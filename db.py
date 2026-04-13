"""
db.py — Think AI Supabase Storage Backend (FINANCE SAFE)
========================================================
Single module that handles all reads and writes to Supabase.

Responsibilities:
- Read full JSON documents from per-company rows
- Upsert full JSON documents safely
- Delete company rows safely
- Provide a health check
- Protect against invalid tables, oversize payloads, and transient failures
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("think-ai.db")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

_SUPABASE_URL = os.getenv("SUPABASE_URL", "")
_SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

_DB_MAX_DOC_BYTES = int(os.getenv("THINK_AI_DB_MAX_DOC_BYTES", "5242880"))  # 5 MB
_DB_RETRIES = int(os.getenv("THINK_AI_DB_RETRIES", "3"))

_VALID_TABLES = {
    "memory_store",
    "audit_store",
    "conditions_store",
    "feedback_store",
    "marketplace_store",
}

_client = None


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_table(table: str) -> bool:
    if table not in _VALID_TABLES:
        logger.error("Invalid table name: %s", table)
        return False
    return True


def _normalize_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Ensures payload is JSON-serializable and stable for nested finance docs.
    """
    try:
        normalized = json.loads(json.dumps(data, default=str))
        return normalized
    except Exception as e:
        logger.error("Failed to normalize data payload | %s", e)
        raise


def _payload_size_bytes(data: dict[str, Any]) -> int:
    return len(json.dumps(data, default=str).encode("utf-8"))


def _get_client():
    global _client
    if _client is None:
        if not _SUPABASE_URL or not _SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment")
        try:
            from supabase import create_client
            _client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
            logger.info("✅ Supabase client initialized")
        except ImportError:
            raise RuntimeError("supabase package not installed. Run: pip install supabase")
    return _client


async def _run_with_retries(fn, *, op_name: str, table: str, company_id_hash: str):
    last_error: Exception | None = None
    for attempt in range(1, _DB_RETRIES + 1):
        try:
            return await asyncio.to_thread(fn)
        except Exception as e:
            last_error = e
            if attempt >= _DB_RETRIES:
                break
            wait = (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            logger.warning(
                "%s retry | table=%s | company_hash=%s | attempt=%d/%d | wait=%.2fs | %s",
                op_name,
                table,
                company_id_hash[:12],
                attempt,
                _DB_RETRIES,
                wait,
                e,
            )
            await asyncio.sleep(wait)
    raise last_error if last_error else RuntimeError(f"{op_name} failed unexpectedly")


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

async def read_store(
    table: str,
    company_id_hash: str,
) -> dict[str, Any] | None:
    """
    Read one company's document from a Supabase table.
    Returns the parsed dict or None if not found.
    """
    if not _validate_table(table):
        return None

    try:
        client = _get_client()

        response = await _run_with_retries(
            lambda: client.table(table)
                .select("data")
                .eq("company_id_hash", company_id_hash)
                .limit(1)
                .execute(),
            op_name="read_store",
            table=table,
            company_id_hash=company_id_hash,
        )

        if response.data and len(response.data) > 0:
            raw = response.data[0].get("data")
            if isinstance(raw, dict):
                return raw
            if raw is None:
                return None
            # safety fallback if DB returns a JSON string
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except Exception:
                    logger.error(
                        "read_store received non-JSON string | table=%s | company_hash=%s",
                        table, company_id_hash[:12],
                    )
                    return None
        return None

    except Exception as e:
        logger.error(
            "read_store error | table=%s | company_hash=%s | %s",
            table, company_id_hash[:12], e,
        )
        return None


async def write_store(
    table: str,
    company_id_hash: str,
    data: dict[str, Any],
) -> bool:
    """
    Write (upsert) one company's document to a Supabase table.
    Creates the row if it doesn't exist, updates it if it does.
    """
    if not _validate_table(table):
        return False

    try:
        client = _get_client()
        normalized = _normalize_data(data)
        payload_size = _payload_size_bytes(normalized)

        if payload_size > _DB_MAX_DOC_BYTES:
            logger.error(
                "write_store rejected oversized payload | table=%s | company_hash=%s | size=%d > max=%d",
                table, company_id_hash[:12], payload_size, _DB_MAX_DOC_BYTES,
            )
            return False

        row = {
            "company_id_hash": company_id_hash,
            "data": normalized,
            "updated_at": _utc_now_iso(),
        }

        await _run_with_retries(
            lambda: client.table(table).upsert(row).execute(),
            op_name="write_store",
            table=table,
            company_id_hash=company_id_hash,
        )

        logger.debug(
            "write_store ok | table=%s | company_hash=%s | bytes=%d",
            table, company_id_hash[:12], payload_size,
        )
        return True

    except Exception as e:
        logger.error(
            "write_store error | table=%s | company_hash=%s | %s",
            table, company_id_hash[:12], e,
        )
        return False


async def delete_store(
    table: str,
    company_id_hash: str,
) -> bool:
    """
    Delete one company's document from a Supabase table.
    Used by GDPR erasure and company reset flows.
    """
    if not _validate_table(table):
        return False

    try:
        client = _get_client()

        response = await _run_with_retries(
            lambda: client.table(table)
                .delete()
                .eq("company_id_hash", company_id_hash)
                .execute(),
            op_name="delete_store",
            table=table,
            company_id_hash=company_id_hash,
        )

        logger.info(
            "🗑️ delete_store ok | table=%s | company_hash=%s",
            table, company_id_hash[:12],
        )
        return True

    except Exception as e:
        logger.error(
            "delete_store error | table=%s | company_hash=%s | %s",
            table, company_id_hash[:12], e,
        )
        return False


async def health_check() -> bool:
    """
    Quick connectivity check — returns True if Supabase is reachable.
    """
    try:
        client = _get_client()

        await _run_with_retries(
            lambda: client.table("memory_store")
                .select("company_id_hash")
                .limit(1)
                .execute(),
            op_name="health_check",
            table="memory_store",
            company_id_hash="healthcheck",
        )

        logger.info("✅ Supabase health check passed")
        return True

    except Exception as e:
        logger.error("❌ Supabase health check failed | %s", e)
        return False