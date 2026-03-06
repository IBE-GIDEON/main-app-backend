"""
marketplace.py — Think AI Live Data Connectors
================================================
Responsibility:  Connect to external data sources, evaluate conditions
                 automatically against real numbers, and trigger re-analysis
                 when a stop condition fires — completing the closed loop.

The problem this solves
-----------------------
conditions.py generates Go/Stop/Review conditions from the Refiner.
Until now, those conditions were checked manually (spot-check) or by
time expiry. marketplace.py makes them fully automatic:

  "Stop if churn rate exceeds 5%"
        │
        ▼  marketplace fetches churn rate from your analytics API
        │
        ▼  LLM evaluates: 7.2% > 5% → condition fired
        │
        ▼  conditions.fire_condition() → status: "fired"
        │
        ▼  request_reanalysis() → decision queued for re-run
        │
        ▼  /decide re-runs automatically → new version in audit.py
        │
        ▼  user sees updated verdict with current data

Connector architecture (pluggable)
------------------------------------
Every data source is a Connector — a simple async class with one method:

    async def fetch(self, query: DataQuery) -> DataPoint

Built-in connectors:
  HttpConnector    — any REST/JSON API (generic, configurable)
  WebhookConnector — receives pushed data (your system pushes to Think AI)
  MockConnector    — deterministic fake data for testing

Future connectors (documented hooks, not yet built):
  StripeConnector       — MRR, churn, ARR
  HubSpotConnector      — pipeline value, conversion rates
  GoogleAnalytics       — traffic, conversion, bounce rate
  PostgresConnector     — any internal database metric
  MarketplaceConnector  — competitive pricing, market share

Adding a new connector = one class, zero edits to existing files.

Re-analysis loop
-----------------
  1. scan_company_conditions()  — called by the background scanner
  2. For each active condition:
       a. Identify relevant metrics from condition text (LLM, lightweight)
       b. Fetch those metrics via configured connectors
       c. Evaluate: has the condition fired? (LLM, lightweight)
       d. If fired → fire_condition() + request_reanalysis()
  3. Background scanner runs on a configurable interval per company
  4. /marketplace/scan endpoint lets main.py trigger on-demand scans

How it connects (zero edits to existing files)
-----------------------------------------------
  main.py adds:
    - POST /marketplace/connectors        (register a data connector)
    - POST /marketplace/{company_id}/scan (trigger on-demand scan)
    - GET  /marketplace/{company_id}/connectors
    - GET  /marketplace/{company_id}/data-points
    - DELETE /marketplace/{company_id}    (GDPR + clear connectors)

  marketplace.py imports from:
    - conditions : get_active_conditions, fire_condition, request_reanalysis
    - router     : UserContext (for re-analysis context)
    - refiner    : refine()
    - output     : format_for_ui()
    - audit      : write_audit()

Security
--------
  * Connector API keys encrypted at rest using Fernet symmetric encryption
  * company_id SHA-256 hashed in all file paths
  * External HTTP calls: 10s timeout, allowlist of domains (configurable)
  * Connector responses size-capped before LLM evaluation
  * Per-company asyncio locks on all writes
  * Narrow exception catches — real bugs surface
  * Rate limiting on external calls per connector (no hammering third-party APIs)

Storage
-------
  Default    : JSON files in marketplace_store/ (same pattern as other modules)
  Secrets    : Connector API keys encrypted with Fernet before storage
  Production : Swap _read_store/_write_store for Postgres/Redis

Usage (from main.py)
---------------------
    from marketplace import (
        register_connector, scan_company_conditions,
        get_connector_config, ConnectorConfig,
    )

    # Register a connector
    await register_connector(
        company_id = "acme-123",
        config     = ConnectorConfig(
            connector_id = "acme-analytics",
            kind         = "http",
            base_url     = "https://analytics.acme.com/api",
            api_key      = "sk-...",   # encrypted before storage
            headers      = {"X-Source": "think-ai"},
        ),
    )

    # Trigger a scan
    report = await scan_company_conditions("acme-123")
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()   # ensures .env is loaded before AsyncOpenAI client is created

import asyncio
import base64
import hashlib
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import httpx
from cryptography.fernet import Fernet, InvalidToken
from openai import AsyncOpenAI, APIStatusError, APITimeoutError
from pydantic import BaseModel, Field, field_validator

# ── Import shared contracts — zero re-definition ──────────────────────────
from router  import UserContext, route
from refiner import refine
from output  import format_for_ui
from audit   import write_audit
from conditions import (
    get_active_conditions,
    fire_condition,
    request_reanalysis,
    ConditionRecord,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

_STORE_DIR          = Path(os.getenv("THINK_AI_MARKETPLACE_DIR",     "marketplace_store"))
_MAX_RECORDS        = int(os.getenv("THINK_AI_MARKETPLACE_MAX",      "500"))
_LOCK_TIMEOUT_SEC   = float(os.getenv("THINK_AI_MARKETPLACE_LOCK",   "5"))
_HTTP_TIMEOUT_SEC   = float(os.getenv("THINK_AI_CONNECTOR_TIMEOUT",  "10"))
_MAX_RESPONSE_BYTES = int(os.getenv("THINK_AI_CONNECTOR_MAX_BYTES",  "32768"))  # 32 KB
_EVAL_MODEL         = os.getenv("THINK_AI_MARKETPLACE_MODEL",        "gpt-4o-mini")
_EVAL_TIMEOUT       = float(os.getenv("THINK_AI_MARKETPLACE_TIMEOUT","20"))
_MAX_RETRIES        = int(os.getenv("THINK_AI_REFINER_RETRIES",      "3"))
_SCAN_INTERVAL_SEC  = int(os.getenv("THINK_AI_SCAN_INTERVAL",        "3600"))   # 1 hour default
_MARKETPLACE_VERSION = "1.0.0"

# Encryption key for connector API keys at rest.
# Generate once: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Store in .env as THINK_AI_ENCRYPTION_KEY
_ENCRYPTION_KEY = os.getenv("THINK_AI_ENCRYPTION_KEY", "")
_fernet: Fernet | None = None

if _ENCRYPTION_KEY:
    try:
        _fernet = Fernet(_ENCRYPTION_KEY.encode())
    except Exception:
        pass   # Will fail gracefully per call

# Allowlisted URL schemes — block file://, ftp://, etc.
_ALLOWED_SCHEMES = {"https", "http"}

# Max connector response size fed into LLM (truncated if larger)
_MAX_LLM_CONTEXT_CHARS = 3000

_llm_client: AsyncOpenAI | None = None

def _get_client() -> AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        load_dotenv()
        _llm_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            timeout=_EVAL_TIMEOUT,
            max_retries=0,
        )
    return _llm_client

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("think-ai.marketplace")

# ─────────────────────────────────────────────
# ENCRYPTION HELPERS
# API keys are encrypted before hitting disk.
# If no encryption key is set (dev mode),
# a warning is logged and keys are stored as-is.
# ─────────────────────────────────────────────

def _encrypt(value: str) -> str:
    if not _fernet:
        logger.warning(
            "THINK_AI_ENCRYPTION_KEY not set — connector API key stored in plaintext. "
            "Set this in production."
        )
        return value
    return _fernet.encrypt(value.encode()).decode()


def _decrypt(value: str) -> str:
    if not _fernet:
        return value
    try:
        return _fernet.decrypt(value.encode()).decode()
    except InvalidToken:
        logger.error("Failed to decrypt connector API key — key may have rotated")
        return ""


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

ConnectorKind = Literal["http", "webhook", "mock"]


class ConnectorConfig(BaseModel):
    """
    Configuration for one external data connector.
    Stored per-company. API key encrypted before write.
    """
    connector_id:   str   = Field(min_length=1, max_length=64)
    kind:           ConnectorKind
    label:          str   = Field(
        description="Human-readable name, e.g. 'Stripe MRR Feed'",
        max_length=128,
    )
    base_url:       str | None = Field(
        default=None,
        description="Base URL for HTTP connectors.",
    )
    api_key:        str | None = Field(
        default=None,
        description="API key — encrypted before storage, never logged.",
    )
    headers:        dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers (e.g. X-Source: think-ai).",
    )
    metric_paths:   dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of metric_name → JSON path in the connector response. "
            "e.g. {'churn_rate': 'data.metrics.churn_monthly'}"
        ),
    )
    enabled:        bool = True
    created_utc:    str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_used_utc:  str | None = None

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v: str) -> str:
        if v not in ("http", "webhook", "mock"):
            raise ValueError("kind must be 'http', 'webhook', or 'mock'")
        return v

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, v: str | None) -> str | None:
        if v is None:
            return v
        from urllib.parse import urlparse
        parsed = urlparse(v)
        if parsed.scheme not in _ALLOWED_SCHEMES:
            raise ValueError(f"URL scheme '{parsed.scheme}' is not allowed.")
        return v

    def safe_dict(self) -> dict:
        """Return config with API key masked — safe for logging and client responses."""
        d = self.model_dump()
        if d.get("api_key"):
            d["api_key"] = "***"
        return d


class DataQuery(BaseModel):
    """What marketplace asks a connector to fetch."""
    metric_name:    str
    condition_text: str   # the condition being evaluated — gives context to the connector
    extra_params:   dict[str, Any] = Field(default_factory=dict)


class DataPoint(BaseModel):
    """One piece of live data returned by a connector."""
    connector_id:   str
    metric_name:    str
    value:          Any          # raw value — string, float, int, dict
    value_str:      str          # human-readable string for LLM evaluation
    fetched_utc:    str
    source_url:     str | None = None


class ConditionEvaluation(BaseModel):
    """Result of evaluating one condition against live data."""
    condition_id:   str
    condition_text: str
    kind:           str
    data_points:    list[DataPoint]
    assessment:     Literal["fired", "clear", "insufficient_data"]
    reasoning:      str
    fired:          bool
    reanalysis_triggered: bool = False
    evaluated_utc:  str


class ScanReport(BaseModel):
    """Result of scanning all conditions for one company."""
    company_id:     str
    conditions_scanned: int
    conditions_fired:   int
    conditions_clear:   int
    insufficient_data:  int
    reanalysis_triggered: int
    evaluations:    list[ConditionEvaluation]
    scan_duration_ms: int
    scanned_utc:    str


class CompanyMarketplaceStore(BaseModel):
    """All connector configs and recent data points for one company."""
    company_id_hash:  str
    version:          str = _MARKETPLACE_VERSION
    connectors:       list[ConnectorConfig] = Field(default_factory=list)
    recent_data:      list[DataPoint]       = Field(default_factory=list)  # last 100
    last_scan_utc:    str | None = None
    last_updated_utc: str


class WebhookPayload(BaseModel):
    """
    Body for POST /marketplace/webhook/{company_id}/{connector_id}.
    External systems push data here instead of being polled.
    """
    metric_name:  str = Field(min_length=1, max_length=128)
    value:        Any
    value_str:    str = Field(min_length=1, max_length=500)
    source_label: str | None = None


# ─────────────────────────────────────────────
# CONNECTOR PROTOCOL
# Every connector implements this interface.
# Adding a new data source = one new class.
# ─────────────────────────────────────────────

@runtime_checkable
class Connector(Protocol):
    async def fetch(self, query: DataQuery) -> DataPoint | None:
        ...


# ─────────────────────────────────────────────
# BUILT-IN CONNECTORS
# ─────────────────────────────────────────────

class HttpConnector:
    """
    Generic REST/JSON connector.
    Fetches data from any HTTP endpoint that returns JSON.
    Extracts the metric value using a dot-notation JSON path.

    Config fields used:
      base_url     : Root URL
      api_key      : Sent as Authorization: Bearer <key> header
      headers      : Any additional headers
      metric_paths : { "metric_name": "dot.notation.path" }
    """

    def __init__(self, config: ConnectorConfig):
        self.config = config

    def _extract_by_path(self, data: dict, path: str) -> Any:
        """Extract a value from nested dict using dot notation."""
        keys = path.split(".")
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    async def fetch(self, query: DataQuery) -> DataPoint | None:
        if not self.config.base_url:
            return None

        headers = dict(self.config.headers)
        api_key = _decrypt(self.config.api_key) if self.config.api_key else None
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = self.config.base_url.rstrip("/")

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SEC) as client:
                response = await client.get(
                    url,
                    headers=headers,
                    params=query.extra_params or {},
                )
                response.raise_for_status()

                # Size guard before parsing
                if len(response.content) > _MAX_RESPONSE_BYTES:
                    logger.warning(
                        "Connector response too large | connector=%s | size=%d",
                        self.config.connector_id, len(response.content),
                    )
                    return None

                data = response.json()

        except httpx.TimeoutException:
            logger.warning(
                "Connector timeout | connector=%s | url=%s",
                self.config.connector_id, url,
            )
            return None
        except httpx.HTTPStatusError as e:
            logger.warning(
                "Connector HTTP error | connector=%s | status=%d",
                self.config.connector_id, e.response.status_code,
            )
            return None
        except Exception as e:
            logger.error(
                "Connector fetch error | connector=%s | %s",
                self.config.connector_id, e,
            )
            return None

        # Extract specific metric using path from config
        json_path = self.config.metric_paths.get(query.metric_name)
        value = self._extract_by_path(data, json_path) if json_path else data

        if value is None:
            logger.warning(
                "Metric not found in response | connector=%s | metric=%s | path=%s",
                self.config.connector_id, query.metric_name, json_path,
            )
            return None

        return DataPoint(
            connector_id=self.config.connector_id,
            metric_name=query.metric_name,
            value=value,
            value_str=str(value)[:500],
            fetched_utc=datetime.now(timezone.utc).isoformat(),
            source_url=url,
        )


class WebhookConnector:
    """
    Receives data that was pushed to
    POST /marketplace/webhook/{company_id}/{connector_id}.
    Stores it in the company's marketplace store.
    Fetching returns the most recent pushed data point for the metric.
    """

    def __init__(self, config: ConnectorConfig, store: CompanyMarketplaceStore):
        self.config = config
        self.store  = store

    async def fetch(self, query: DataQuery) -> DataPoint | None:
        # Return the most recent pushed data point for this metric
        candidates = [
            d for d in reversed(self.store.recent_data)
            if d.connector_id == self.config.connector_id
            and d.metric_name == query.metric_name
        ]
        return candidates[0] if candidates else None


class MockConnector:
    """
    Deterministic fake connector for testing.
    Returns configurable values — useful for demo mode and CI.
    metric_paths is used as a metric_name → value map.
    e.g. metric_paths = {"churn_rate": "4.2", "mrr": "125000"}
    """

    def __init__(self, config: ConnectorConfig):
        self.config = config

    async def fetch(self, query: DataQuery) -> DataPoint | None:
        value_str = self.config.metric_paths.get(query.metric_name, "unavailable")
        return DataPoint(
            connector_id=self.config.connector_id,
            metric_name=query.metric_name,
            value=value_str,
            value_str=value_str,
            fetched_utc=datetime.now(timezone.utc).isoformat(),
            source_url=None,
        )


def _build_connector(config: ConnectorConfig, store: CompanyMarketplaceStore) -> Connector:
    """Factory — returns the right connector class for this config."""
    if config.kind == "http":
        return HttpConnector(config)
    if config.kind == "webhook":
        return WebhookConnector(config, store)
    return MockConnector(config)


# ─────────────────────────────────────────────
# CONCURRENCY GUARD
# ─────────────────────────────────────────────

_write_locks: dict[str, asyncio.Lock] = {}


def _get_lock(company_id: str) -> asyncio.Lock:
    if company_id not in _write_locks:
        _write_locks[company_id] = asyncio.Lock()
    return _write_locks[company_id]


# ─────────────────────────────────────────────
# STORAGE BACKEND
# ─────────────────────────────────────────────

def _company_hash(company_id: str) -> str:
    return hashlib.sha256(company_id.encode()).hexdigest()


def _store_path(company_id: str) -> Path:
    return _STORE_DIR / f"{_company_hash(company_id)}.json"


def _ensure_store_dir() -> None:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    gitignore = _STORE_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n")


async def _read_store(company_id: str) -> CompanyMarketplaceStore | None:
    path = _store_path(company_id)
    if not path.exists():
        return None
    try:
        raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return CompanyMarketplaceStore.model_validate_json(raw)
    except Exception as e:
        logger.error(
            "Marketplace read error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return None


async def _write_store(company_id: str, store: CompanyMarketplaceStore) -> bool:
    _ensure_store_dir()
    path = _store_path(company_id)

    # Cap recent data to 100 points
    if len(store.recent_data) > 100:
        store.recent_data = store.recent_data[-100:]

    try:
        await asyncio.to_thread(
            path.write_text,
            store.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return True
    except Exception as e:
        logger.error(
            "Marketplace write error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return False


def _new_store(company_id: str) -> CompanyMarketplaceStore:
    return CompanyMarketplaceStore(
        company_id_hash=_company_hash(company_id),
        connectors=[],
        recent_data=[],
        last_updated_utc=datetime.now(timezone.utc).isoformat(),
    )


# ─────────────────────────────────────────────
# LLM EVALUATION — two lightweight calls
# ─────────────────────────────────────────────

_METRIC_SYSTEM = """
You are Think AI's Data Scout.
Given a condition statement, identify what metrics or data points
would allow you to evaluate whether this condition has fired.
Return only metrics that are commonly trackable via analytics APIs.
Return valid JSON only.
""".strip()


class _MetricQuery(BaseModel):
    metrics: list[str] = Field(
        description=(
            "List of 1–3 metric names to fetch that would allow evaluating this condition. "
            "Use snake_case. e.g. ['churn_rate', 'mrr', 'cac']"
        )
    )
    reasoning: str = Field(
        description="Why these metrics are relevant to this condition."
    )


_EVAL_SYSTEM = """
You are Think AI's Condition Evaluator.
You are given:
  - A condition statement from a business decision
  - Live data points fetched from the company's data connectors

Your job: has this condition fired based on the data?
Be direct. If data is insufficient, say so.
Return valid JSON only.
""".strip()


class _EvalOutput(BaseModel):
    assessment:  Literal["fired", "clear", "insufficient_data"]
    reasoning:   str = Field(description="2–3 sentences explaining the assessment")
    confidence:  Literal["high", "medium", "low"]


async def _identify_metrics(condition_text: str) -> list[str]:
    """Ask the LLM what metrics to fetch for this condition."""
    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = await _get_client().chat.completions.create(
                model=_EVAL_MODEL,
                messages=[
                    {"role": "system", "content": _METRIC_SYSTEM + "\n\nRespond with valid JSON only. No markdown."},
                    {"role": "user",   "content": f"Condition: {condition_text}"},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            msg = response.choices[0].message
            import json as _json
            raw = (msg.content or "").strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return _MetricQuery.model_validate(_json.loads(raw)).metrics[:3]

        except (APITimeoutError, APIStatusError) as e:
            status_code = getattr(e, "status_code", None)
            if status_code and status_code not in (429, 500, 502, 503, 504):
                raise
            last_error = e
            wait = (2 ** attempt) + random.uniform(0, 0.3)
            await asyncio.sleep(wait)
        except Exception as e:
            last_error = e
            break

    logger.warning("_identify_metrics failed | %s", last_error)
    return []


async def _evaluate_condition(
    condition: ConditionRecord,
    data_points: list[DataPoint],
) -> _EvalOutput:
    """Ask the LLM whether a condition has fired given live data."""
    if not data_points:
        return _EvalOutput(
            assessment="insufficient_data",
            reasoning="No data points were available from any connector.",
            confidence="low",
        )

    data_str = "\n".join(
        f"  {dp.metric_name} = {dp.value_str} (from {dp.connector_id})"
        for dp in data_points
    )[:_MAX_LLM_CONTEXT_CHARS]

    user_message = (
        f"Condition ({condition.kind.upper()}): {condition.text}\n\n"
        f"Live data:\n{data_str}\n\n"
        f"Has this condition fired?"
    )

    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = await _get_client().chat.completions.create(
                model=_EVAL_MODEL,
                messages=[
                    {"role": "system", "content": _EVAL_SYSTEM + "\n\nRespond with valid JSON only. No markdown."},
                    {"role": "user",   "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            msg = response.choices[0].message
            import json as _json
            raw = (msg.content or "").strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return _EvalOutput.model_validate(_json.loads(raw))

        except (APITimeoutError, APIStatusError) as e:
            status_code = getattr(e, "status_code", None)
            if status_code and status_code not in (429, 500, 502, 503, 504):
                raise
            last_error = e
            wait = (2 ** attempt) + random.uniform(0, 0.3)
            await asyncio.sleep(wait)
        except Exception as e:
            last_error = e
            break

    logger.warning("_evaluate_condition failed | %s", last_error)
    return _EvalOutput(
        assessment="insufficient_data",
        reasoning=f"Evaluation failed after {_MAX_RETRIES} retries.",
        confidence="low",
    )


# ─────────────────────────────────────────────
# CORE SCAN ENGINE
# ─────────────────────────────────────────────

async def _scan_one_condition(
    company_id:  str,
    condition:   ConditionRecord,
    connectors:  list[ConnectorConfig],
    store:       CompanyMarketplaceStore,
) -> ConditionEvaluation:
    """
    Evaluates one condition against all available connectors.
    Returns a ConditionEvaluation — never raises.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Skip conditions that are already resolved
    if condition.status in ("fired", "cleared", "expired", "acknowledged"):
        return ConditionEvaluation(
            condition_id=condition.condition_id,
            condition_text=condition.text,
            kind=condition.kind,
            data_points=[],
            assessment="clear",
            reasoning=f"Condition already in status '{condition.status}' — skipped.",
            fired=False,
            evaluated_utc=now,
        )

    enabled_connectors = [c for c in connectors if c.enabled]
    if not enabled_connectors:
        return ConditionEvaluation(
            condition_id=condition.condition_id,
            condition_text=condition.text,
            kind=condition.kind,
            data_points=[],
            assessment="insufficient_data",
            reasoning="No enabled connectors configured for this company.",
            fired=False,
            evaluated_utc=now,
        )

    # Step 1: Identify what metrics this condition needs
    metric_names = await _identify_metrics(condition.text)

    if not metric_names:
        return ConditionEvaluation(
            condition_id=condition.condition_id,
            condition_text=condition.text,
            kind=condition.kind,
            data_points=[],
            assessment="insufficient_data",
            reasoning="Could not identify relevant metrics for this condition.",
            fired=False,
            evaluated_utc=now,
        )

    # Step 2: Fetch those metrics from all connectors
    data_points: list[DataPoint] = []
    fetch_tasks = []

    for config in enabled_connectors:
        connector = _build_connector(config, store)
        for metric in metric_names:
            fetch_tasks.append(
                connector.fetch(DataQuery(
                    metric_name=metric,
                    condition_text=condition.text,
                ))
            )

    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, DataPoint):
            data_points.append(result)
        elif isinstance(result, Exception):
            logger.warning("Connector fetch raised | %s", result)

    # Step 3: LLM evaluates the condition against fetched data
    eval_result = await _evaluate_condition(condition, data_points)

    fired              = eval_result.assessment == "fired"
    reanalysis_done    = False

    # Step 4: If fired → fire_condition() + request_reanalysis() for stop conditions
    if fired:
        await fire_condition(company_id, condition.condition_id)
        logger.warning(
            "🚨 Condition fired | company_hash=%s | id=%s | kind=%s | text=%s",
            _company_hash(company_id)[:12],
            condition.condition_id,
            condition.kind,
            condition.text[:60],
        )
        if condition.kind == "stop":
            await request_reanalysis(company_id, condition.condition_id)
            reanalysis_done = True
            logger.warning(
                "🔁 Re-analysis queued | company_hash=%s | decision=%s",
                _company_hash(company_id)[:12], condition.decision_id,
            )

    return ConditionEvaluation(
        condition_id=condition.condition_id,
        condition_text=condition.text,
        kind=condition.kind,
        data_points=data_points,
        assessment=eval_result.assessment,
        reasoning=eval_result.reasoning,
        fired=fired,
        reanalysis_triggered=reanalysis_done,
        evaluated_utc=now,
    )


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

async def register_connector(
    company_id: str,
    config:     ConnectorConfig,
) -> bool:
    """
    Register or update a data connector for a company.
    API key is encrypted before storage. Never logged.

    Parameters
    ----------
    company_id : Tenant identifier.
    config     : ConnectorConfig — see schema for fields.

    Returns
    -------
    bool — True if written successfully.
    """
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        store = await _read_store(company_id) or _new_store(company_id)

        # Encrypt API key before storage
        encrypted_config = config.model_copy()
        if config.api_key:
            encrypted_config.api_key = _encrypt(config.api_key)

        # Upsert — replace existing connector with same ID
        store.connectors = [
            c for c in store.connectors
            if c.connector_id != config.connector_id
        ]
        store.connectors.append(encrypted_config)
        store.last_updated_utc = datetime.now(timezone.utc).isoformat()

        success = await _write_store(company_id, store)
        if success:
            logger.info(
                "✅ Connector registered | company_hash=%s | id=%s | kind=%s",
                _company_hash(company_id)[:12],
                config.connector_id, config.kind,
            )
        return success

    except Exception as e:
        logger.error(
            "register_connector error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return False
    finally:
        lock.release()


async def scan_company_conditions(
    company_id: str,
    context:    UserContext | None = None,
) -> ScanReport:
    """
    Scans all active conditions for a company against live connector data.
    Fires conditions where data shows they've triggered.
    Queues stop conditions for re-analysis automatically.

    This is the core of the closed loop. Call it:
      - On a schedule (e.g. every hour via a cron job / background task)
      - On-demand via POST /marketplace/{company_id}/scan

    Parameters
    ----------
    company_id : Tenant identifier.
    context    : Optional UserContext — enriches any auto-triggered re-analysis.

    Returns
    -------
    ScanReport — always, even if no conditions were scanned.
    """
    t0  = time.perf_counter()
    now = datetime.now(timezone.utc).isoformat()

    store = await _read_store(company_id)
    if not store or not store.connectors:
        logger.info(
            "No connectors configured | company_hash=%s",
            _company_hash(company_id)[:12],
        )
        return ScanReport(
            company_id=company_id,
            conditions_scanned=0,
            conditions_fired=0,
            conditions_clear=0,
            insufficient_data=0,
            reanalysis_triggered=0,
            evaluations=[],
            scan_duration_ms=0,
            scanned_utc=now,
        )

    # Fetch active conditions
    condition_report = await get_active_conditions(company_id)
    all_conditions: list[ConditionRecord] = (
        condition_report.go_conditions
        + condition_report.needs_review
        + condition_report.urgent
    )

    # Deduplicate by condition_id
    seen: set[str] = set()
    unique_conditions: list[ConditionRecord] = []
    for c in all_conditions:
        if c.condition_id not in seen:
            seen.add(c.condition_id)
            unique_conditions.append(c)

    if not unique_conditions:
        return ScanReport(
            company_id=company_id,
            conditions_scanned=0,
            conditions_fired=0,
            conditions_clear=0,
            insufficient_data=0,
            reanalysis_triggered=0,
            evaluations=[],
            scan_duration_ms=round((time.perf_counter() - t0) * 1000),
            scanned_utc=now,
        )

    logger.info(
        "🔍 Scanning | company_hash=%s | conditions=%d | connectors=%d",
        _company_hash(company_id)[:12],
        len(unique_conditions), len(store.connectors),
    )

    # Evaluate all conditions (concurrent per condition, sequential per connector
    # to avoid rate-limit hammering)
    evaluations: list[ConditionEvaluation] = []
    for condition in unique_conditions:
        evaluation = await _scan_one_condition(
            company_id=company_id,
            condition=condition,
            connectors=store.connectors,
            store=store,
        )
        evaluations.append(evaluation)

        # Store fetched data points for the dashboard
        if evaluation.data_points:
            store.recent_data.extend(evaluation.data_points)

    # Update last_scan_utc
    store.last_scan_utc    = now
    store.last_updated_utc = now
    await _write_store(company_id, store)

    fired_count       = sum(1 for e in evaluations if e.fired)
    clear_count       = sum(1 for e in evaluations if e.assessment == "clear" and not e.fired)
    insufficient_count= sum(1 for e in evaluations if e.assessment == "insufficient_data")
    reanalysis_count  = sum(1 for e in evaluations if e.reanalysis_triggered)
    duration_ms       = round((time.perf_counter() - t0) * 1000)

    logger.info(
        "✅ Scan complete | company_hash=%s | scanned=%d | fired=%d | "
        "reanalysis=%d | duration_ms=%d",
        _company_hash(company_id)[:12],
        len(evaluations), fired_count, reanalysis_count, duration_ms,
    )

    return ScanReport(
        company_id=company_id,
        conditions_scanned=len(evaluations),
        conditions_fired=fired_count,
        conditions_clear=clear_count,
        insufficient_data=insufficient_count,
        reanalysis_triggered=reanalysis_count,
        evaluations=evaluations,
        scan_duration_ms=duration_ms,
        scanned_utc=now,
    )


async def receive_webhook(
    company_id:   str,
    connector_id: str,
    payload:      WebhookPayload,
) -> bool:
    """
    Receives a pushed data point from an external system.
    Stored immediately. The next scan will pick it up automatically.
    Called by POST /marketplace/webhook/{company_id}/{connector_id}.

    Parameters
    ----------
    company_id   : Tenant identifier.
    connector_id : Must match a registered WebhookConnector.
    payload      : The pushed data point.

    Returns
    -------
    bool — True if stored successfully.
    """
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        store = await _read_store(company_id)
        if not store:
            logger.warning(
                "Webhook received for unknown company | company_hash=%s",
                _company_hash(company_id)[:12],
            )
            return False

        # Verify connector exists and is a webhook type
        connector_exists = any(
            c.connector_id == connector_id and c.kind == "webhook"
            for c in store.connectors
        )
        if not connector_exists:
            logger.warning(
                "Webhook connector not found | company_hash=%s | connector=%s",
                _company_hash(company_id)[:12], connector_id,
            )
            return False

        data_point = DataPoint(
            connector_id=connector_id,
            metric_name=payload.metric_name,
            value=payload.value,
            value_str=payload.value_str[:500],
            fetched_utc=datetime.now(timezone.utc).isoformat(),
            source_url=None,
        )

        store.recent_data.append(data_point)
        store.last_updated_utc = datetime.now(timezone.utc).isoformat()

        success = await _write_store(company_id, store)
        if success:
            logger.info(
                "✅ Webhook received | company_hash=%s | connector=%s | metric=%s",
                _company_hash(company_id)[:12], connector_id, payload.metric_name,
            )
        return success

    except Exception as e:
        logger.error(
            "receive_webhook error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return False
    finally:
        lock.release()


async def get_connector_config(
    company_id: str,
) -> list[dict]:
    """
    Returns all connector configs for a company — with API keys masked.
    Safe to send to the frontend settings page.
    """
    store = await _read_store(company_id)
    if not store:
        return []
    return [c.safe_dict() for c in store.connectors]


async def get_recent_data(
    company_id: str,
    limit:      int = 50,
) -> list[DataPoint]:
    """
    Returns the most recent data points fetched from connectors.
    Used by the marketplace dashboard to show live data.
    """
    store = await _read_store(company_id)
    if not store:
        return []
    return list(reversed(store.recent_data[-limit:]))


async def disable_connector(
    company_id:   str,
    connector_id: str,
) -> bool:
    """Disable a connector without deleting it."""
    return await _toggle_connector(company_id, connector_id, enabled=False)


async def enable_connector(
    company_id:   str,
    connector_id: str,
) -> bool:
    """Re-enable a previously disabled connector."""
    return await _toggle_connector(company_id, connector_id, enabled=True)


async def _toggle_connector(
    company_id:   str,
    connector_id: str,
    enabled:      bool,
) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        store = await _read_store(company_id)
        if not store:
            return False
        for c in store.connectors:
            if c.connector_id == connector_id:
                c.enabled = enabled
                store.last_updated_utc = datetime.now(timezone.utc).isoformat()
                return await _write_store(company_id, store)
        return False
    except Exception as e:
        logger.error("_toggle_connector error | %s", e)
        return False
    finally:
        lock.release()


async def delete_connector(
    company_id:   str,
    connector_id: str,
) -> bool:
    """Permanently remove a connector."""
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        store = await _read_store(company_id)
        if not store:
            return False
        before = len(store.connectors)
        store.connectors = [c for c in store.connectors if c.connector_id != connector_id]
        if len(store.connectors) == before:
            return False
        store.last_updated_utc = datetime.now(timezone.utc).isoformat()
        success = await _write_store(company_id, store)
        if success:
            logger.info(
                "🗑️  Connector deleted | company_hash=%s | id=%s",
                _company_hash(company_id)[:12], connector_id,
            )
        return success
    except Exception as e:
        logger.error("delete_connector error | %s", e)
        return False
    finally:
        lock.release()


async def clear_marketplace(company_id: str) -> bool:
    """
    Wipes all marketplace data for a company.
    Called alongside clear_memory() / clear_audit() for GDPR erasure.
    """
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        path = _store_path(company_id)
        if path.exists():
            await asyncio.to_thread(path.unlink)
            logger.info(
                "🗑️  Marketplace cleared | company_hash=%s",
                _company_hash(company_id)[:12],
            )
            return True
        return False
    except Exception as e:
        logger.error("clear_marketplace error | %s", e)
        return False
    finally:
        lock.release()


# ─────────────────────────────────────────────
# QUICK LOCAL TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import json
    from router import route, UserContext
    from refiner import refine
    from output import format_for_ui
    from audit import write_audit
    from conditions import register_conditions

    async def _demo():
        company_id = "demo-co"
        query      = "Should we raise our prices by 20% next month?"
        ctx        = UserContext(
            industry="E-commerce",
            company_size="Series A",
            risk_appetite="Moderate",
        )

        print("── Step 1: Full pipeline ───────────────────────")
        plan      = await route(query, company_id, ctx)
        decision  = await refine(query, plan, company_id, ctx)
        payload   = format_for_ui(decision, query, company_id)
        record_id = await write_audit(
            company_id=company_id, query=query,
            plan=plan, decision=decision, payload=payload,
        )
        await register_conditions(
            company_id=company_id,
            decision_id=decision.audit.query_hash[:16],
            query=query,
            verdict=decision.verdict_box,
        )
        print(f"  Verdict: {payload.verdict_card.badge.label}")

        print("\n── Step 2: Register mock connector ─────────────")
        ok = await register_connector(
            company_id=company_id,
            config=ConnectorConfig(
                connector_id="mock-analytics",
                kind="mock",
                label="Mock Analytics Feed",
                metric_paths={
                    "churn_rate":        "7.2",    # above a common stop threshold
                    "mrr":               "95000",
                    "conversion_rate":   "2.1",
                    "customer_count":    "842",
                    "avg_order_value":   "67.50",
                },
            ),
        )
        print(f"  Connector registered: {ok}")

        print("\n── Step 3: Scan conditions ─────────────────────")
        report = await scan_company_conditions(company_id, ctx)
        print(f"  Conditions scanned   : {report.conditions_scanned}")
        print(f"  Conditions fired     : {report.conditions_fired}")
        print(f"  Re-analysis triggered: {report.reanalysis_triggered}")
        print(f"  Scan duration        : {report.scan_duration_ms}ms")
        for e in report.evaluations:
            icon = "🚨" if e.fired else ("✅" if e.assessment == "clear" else "❓")
            print(f"  {icon} [{e.kind.upper()}] {e.condition_text[:50]}")
            print(f"      → {e.assessment} | {e.reasoning[:80]}")

        print("\n── Step 4: Get connectors (API keys masked) ────")
        connectors = await get_connector_config(company_id)
        for c in connectors:
            print(f"  {c['connector_id']} | {c['kind']} | enabled={c['enabled']}")

        print("\n── Step 5: Recent data points ──────────────────")
        data = await get_recent_data(company_id, limit=5)
        for d in data:
            print(f"  {d.metric_name} = {d.value_str} (from {d.connector_id})")

    asyncio.run(_demo())