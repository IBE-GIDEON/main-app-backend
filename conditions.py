"""
conditions.py — Three AI Finance Condition Monitor
Storage backend: Supabase

Finance-first changes:
- Adds finance-aware metric_targets and threshold_hint to conditions
- Deterministic finance trigger evaluation before LLM fallback
- Stores last_evaluated_metrics and evaluation_source
- Keeps public API shape compatible with existing main.py usage
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import asyncio
import hashlib
import logging
import os
import random
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Literal

from openai import AsyncOpenAI, APIStatusError, APITimeoutError
from pydantic import BaseModel, Field, field_validator

from router import UserContext
from refiner import VerdictBox
from db import read_store, write_store, delete_store

_TABLE              = "conditions_store"
_LOCK_TIMEOUT_SEC   = float(os.getenv("THINK_AI_CONDITIONS_LOCK", "5"))
_MAX_RECORDS        = int(os.getenv("THINK_AI_CONDITIONS_MAX", "200"))
_MAX_RETRIES        = int(os.getenv("THINK_AI_REFINER_RETRIES", "3"))
_SPOT_CHECK_MODEL   = os.getenv("THINK_AI_CONDITIONS_MODEL", "gpt-4o-mini")
_SPOT_CHECK_TIMEOUT = float(os.getenv("THINK_AI_CONDITIONS_TIMEOUT", "20"))
_AUTO_REVIEW_DAYS   = int(os.getenv("THINK_AI_CONDITIONS_REVIEW_DAYS", "30"))
_CONDITIONS_VERSION = "2.0.0-finance"

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        load_dotenv()
        _client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            timeout=_SPOT_CHECK_TIMEOUT,
            max_retries=0,
        )
    return _client


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("three-ai.conditions")

ConditionKind = Literal["go", "stop", "review"]
ConditionStatus = Literal["active", "fired", "acknowledged", "cleared", "expired"]


FINANCE_KEYWORDS: dict[str, list[str]] = {
    "runway": ["runway_months", "cash_balance", "monthly_burn"],
    "cash": ["cash_balance", "available_liquidity"],
    "liquidity": ["cash_balance", "available_liquidity", "current_ratio", "quick_ratio"],
    "burn": ["monthly_burn", "cash_balance"],
    "revenue": ["revenue_last_30d", "revenue_growth_pct", "mrr", "arr"],
    "mrr": ["mrr"],
    "arr": ["arr"],
    "margin": ["gross_margin_pct", "ebitda_margin_pct", "net_margin_pct"],
    "gross margin": ["gross_margin_pct"],
    "ebitda": ["ebitda_margin_pct"],
    "collections": ["ar_total", "ar_overdue_30_plus", "failed_payment_rate_pct"],
    "receivable": ["ar_total", "ar_overdue_30_plus"],
    "receivables": ["ar_total", "ar_overdue_30_plus"],
    "ar": ["ar_total", "ar_overdue_30_plus"],
    "ap": ["ap_total", "ap_due_30d"],
    "working capital": ["ar_total", "ar_overdue_30_plus", "ap_total", "ap_due_30d"],
    "failed payment": ["failed_payment_rate_pct"],
    "failed payments": ["failed_payment_rate_pct"],
    "churn": ["revenue_churn_pct", "logo_churn_pct", "nrr_pct"],
    "nrr": ["nrr_pct"],
    "customer concentration": ["customer_concentration_pct", "top_customer_share_pct"],
    "top customer": ["top_customer_share_pct", "customer_concentration_pct"],
    "forecast": ["forecast_vs_actual_pct", "pipeline_coverage"],
    "pipeline": ["pipeline_coverage"],
    "debt": ["debt_service_coverage_ratio", "covenant_headroom_pct"],
    "covenant": ["debt_service_coverage_ratio", "covenant_headroom_pct"],
    "current ratio": ["current_ratio"],
    "quick ratio": ["quick_ratio"],
    "headcount": ["headcount"],
}


class ConditionRecord(BaseModel):
    condition_id: str
    decision_id: str
    company_id: str
    query_preview: str
    kind: ConditionKind
    text: str
    status: ConditionStatus = "active"
    created_utc: str
    last_checked_utc: str | None = None
    fired_utc: str | None = None
    acknowledged_utc: str | None = None
    cleared_utc: str | None = None
    expires_utc: str
    spot_check_note: str | None = None
    reanalysis_requested: bool = False

    # finance-aware additions
    metric_targets: list[str] = Field(default_factory=list)
    threshold_hint: float | None = None
    last_evaluated_metrics: dict[str, Any] = Field(default_factory=dict)
    evaluation_source: str | None = None

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v):
        return v if v in ("go", "stop", "review") else "review"

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        return v if v in ("active", "fired", "acknowledged", "cleared", "expired") else "active"


class CompanyConditionStore(BaseModel):
    company_id_hash: str
    version: str = _CONDITIONS_VERSION
    records: list[ConditionRecord] = Field(default_factory=list)
    last_updated_utc: str


class ConditionReport(BaseModel):
    company_id: str
    total_active: int
    total_fired: int
    total_acknowledged: int
    urgent: list[ConditionRecord]
    needs_review: list[ConditionRecord]
    go_conditions: list[ConditionRecord]
    generated_utc: str


class SpotCheckRequest(BaseModel):
    company_id: str = Field(min_length=1, max_length=128)
    condition_id: str = Field(min_length=1, max_length=64)
    user_update: str = Field(min_length=3, max_length=1000)
    context: dict[str, Any] | None = None


class SpotCheckResult(BaseModel):
    condition_id: str
    condition_text: str
    kind: ConditionKind
    user_update: str
    assessment: Literal["likely_fired", "likely_clear", "uncertain"]
    reasoning: str
    recommended_action: str
    new_status: ConditionStatus


_write_locks: dict[str, asyncio.Lock] = {}


def _get_lock(company_id: str) -> asyncio.Lock:
    if company_id not in _write_locks:
        _write_locks[company_id] = asyncio.Lock()
    return _write_locks[company_id]


def _company_hash(company_id: str) -> str:
    return hashlib.sha256(company_id.encode()).hexdigest()


async def _read_store(company_id: str) -> CompanyConditionStore | None:
    raw = await read_store(_TABLE, _company_hash(company_id))
    if raw is None:
        return None
    try:
        return CompanyConditionStore.model_validate(raw)
    except Exception as e:
        logger.error("Conditions parse error | %s", e)
        return None


async def _write_store(company_id: str, store: CompanyConditionStore) -> bool:
    if len(store.records) > _MAX_RECORDS:
        droppable = [r for r in store.records if r.status in ("cleared", "expired", "acknowledged")]
        droppable.sort(key=lambda r: r.created_utc)
        to_drop = set(r.condition_id for r in droppable[:len(store.records) - _MAX_RECORDS])
        store.records = [r for r in store.records if r.condition_id not in to_drop]
    return await write_store(_TABLE, _company_hash(company_id), store.model_dump())


def _new_store(company_id: str) -> CompanyConditionStore:
    return CompanyConditionStore(
        company_id_hash=_company_hash(company_id),
        records=[],
        last_updated_utc=datetime.now(timezone.utc).isoformat(),
    )


def _condition_id(decision_id: str, kind: str, text: str) -> str:
    raw = f"{decision_id}:{kind}:{text.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _expires_utc(days: int = _AUTO_REVIEW_DAYS) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def _check_expiry(record: ConditionRecord) -> ConditionRecord:
    if record.status not in ("active", "fired"):
        return record
    try:
        if datetime.now(timezone.utc) > datetime.fromisoformat(record.expires_utc):
            record.status = "expired"
    except Exception:
        pass
    return record


def _infer_metrics_from_text(text: str) -> list[str]:
    text_l = text.lower()
    found: list[str] = []
    for keyword, metrics in FINANCE_KEYWORDS.items():
        if keyword in text_l:
            for metric in metrics:
                if metric not in found:
                    found.append(metric)
    return found[:5]


def _extract_threshold(text: str) -> float | None:
    text_l = text.lower()

    # prioritize explicit comparison phrases
    patterns = [
        r"(?:less than|below|under|greater than|more than|above|over)\s+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:%|months|month|x)?",
    ]
    for pattern in patterns:
        m = re.search(pattern, text_l)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def _build_records_from_verdict(
    company_id: str,
    decision_id: str,
    query: str,
    verdict: VerdictBox,
) -> list[ConditionRecord]:
    now = datetime.now(timezone.utc).isoformat()
    records: list[ConditionRecord] = []
    expiry_days = {
        "go": _AUTO_REVIEW_DAYS,
        "stop": _AUTO_REVIEW_DAYS * 2,
        "review": max(7, _AUTO_REVIEW_DAYS // 2),
    }

    for kind, texts in [
        ("go", verdict.go_conditions),
        ("stop", verdict.stop_conditions),
        ("review", verdict.review_triggers),
    ]:
        for text in texts:
            if not text or not text.strip():
                continue

            clean_text = text.strip()
            cid = _condition_id(decision_id, kind, clean_text)
            metrics = _infer_metrics_from_text(clean_text)
            threshold = _extract_threshold(clean_text)

            records.append(
                ConditionRecord(
                    condition_id=cid,
                    decision_id=decision_id,
                    company_id=company_id,
                    query_preview=query[:80],
                    kind=kind,
                    text=clean_text,
                    status="active",
                    created_utc=now,
                    expires_utc=_expires_utc(expiry_days[kind]),
                    metric_targets=metrics,
                    threshold_hint=threshold,
                )
            )
    return records


_SPOT_CHECK_SYSTEM = """
You are Three AI's Finance Condition Evaluator.
A user has provided an update about their company situation.
Assess whether a specific finance condition has likely fired, is likely still clear, or is uncertain.
Be conservative and practical.
Return valid JSON only.
""".strip()


class _SpotCheckOutput(BaseModel):
    assessment: Literal["likely_fired", "likely_clear", "uncertain"]
    reasoning: str
    recommended_action: str


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _snapshot_from_context(context: UserContext | None) -> dict[str, Any]:
    if not context or not getattr(context, "extra", None):
        return {}
    if not isinstance(context.extra, dict):
        return {}
    return context.extra


def _deterministic_finance_assessment(condition: ConditionRecord, snapshot: dict[str, Any]) -> tuple[str, str] | None:
    """
    Returns:
      ("likely_fired", reason)
      ("likely_clear", reason)
      or None if insufficient deterministic evidence
    """
    text = condition.text.lower()
    threshold = condition.threshold_hint

    def val(name: str) -> float | None:
        return _to_float(snapshot.get(name))

    # runway checks
    if "runway" in text:
        runway = val("runway_months")
        if runway is not None and threshold is not None:
            if any(p in text for p in ("less than", "below", "under", "<")):
                return (
                    "likely_fired" if runway < threshold else "likely_clear",
                    f"Runway is {runway} months versus threshold {threshold}.",
                )
            if any(p in text for p in ("greater than", "above", "over", ">")):
                return (
                    "likely_fired" if runway > threshold else "likely_clear",
                    f"Runway is {runway} months versus threshold {threshold}.",
                )

    # failed payment rate
    if "failed payment" in text:
        rate = val("failed_payment_rate_pct")
        if rate is not None and threshold is not None:
            if any(p in text for p in ("greater than", "above", "over", "exceeds", ">")):
                return (
                    "likely_fired" if rate > threshold else "likely_clear",
                    f"Failed payment rate is {rate}% versus threshold {threshold}%.",
                )
            return (
                "likely_fired" if rate > threshold else "likely_clear",
                f"Failed payment rate is {rate}% versus threshold {threshold}%.",
            )

    # overdue receivables ratio
    if any(k in text for k in ("overdue", "collections", "receivable", "receivables", "ar ")):
        ar_total = val("ar_total")
        overdue = val("ar_overdue_30_plus")
        if ar_total not in (None, 0) and overdue is not None and threshold is not None:
            overdue_pct = round((overdue / ar_total) * 100, 2)
            if any(p in text for p in ("greater than", "above", "over", "exceeds", ">")):
                return (
                    "likely_fired" if overdue_pct > threshold else "likely_clear",
                    f"Overdue AR is {overdue_pct}% of total AR versus threshold {threshold}%.",
                )
            return (
                "likely_fired" if overdue_pct > threshold else "likely_clear",
                f"Overdue AR is {overdue_pct}% of total AR versus threshold {threshold}%.",
            )

    # customer concentration
    if "customer concentration" in text or "top customer" in text:
        metric = "top_customer_share_pct" if "top customer" in text else "customer_concentration_pct"
        concentration = val(metric)
        if concentration is not None and threshold is not None:
            return (
                "likely_fired" if concentration > threshold else "likely_clear",
                f"{metric} is {concentration}% versus threshold {threshold}%.",
            )

    # revenue growth going negative / below threshold
    if "revenue" in text and "growth" in text:
        growth = val("revenue_growth_pct")
        if growth is not None and threshold is not None:
            if any(p in text for p in ("below", "under", "<", "less than")):
                return (
                    "likely_fired" if growth < threshold else "likely_clear",
                    f"Revenue growth is {growth}% versus threshold {threshold}%.",
                )
            if any(p in text for p in ("greater than", "above", "over", ">")):
                return (
                    "likely_fired" if growth > threshold else "likely_clear",
                    f"Revenue growth is {growth}% versus threshold {threshold}%.",
                )

    # quick ratio / current ratio
    if "quick ratio" in text:
        qr = val("quick_ratio")
        if qr is not None and threshold is not None:
            return (
                "likely_fired" if qr < threshold else "likely_clear",
                f"Quick ratio is {qr} versus threshold {threshold}.",
            )

    if "current ratio" in text:
        cr = val("current_ratio")
        if cr is not None and threshold is not None:
            return (
                "likely_fired" if cr < threshold else "likely_clear",
                f"Current ratio is {cr} versus threshold {threshold}.",
            )

    # debt / covenant
    if "covenant" in text:
        headroom = val("covenant_headroom_pct")
        if headroom is not None and threshold is not None:
            return (
                "likely_fired" if headroom < threshold else "likely_clear",
                f"Covenant headroom is {headroom}% versus threshold {threshold}%.",
            )

    if "debt service" in text:
        dscr = val("debt_service_coverage_ratio")
        if dscr is not None and threshold is not None:
            return (
                "likely_fired" if dscr < threshold else "likely_clear",
                f"Debt service coverage ratio is {dscr} versus threshold {threshold}.",
            )

    return None


async def _run_spot_check(
    condition: ConditionRecord,
    user_update: str,
    context: UserContext | None = None,
) -> _SpotCheckOutput:
    snapshot = _snapshot_from_context(context)

    deterministic = _deterministic_finance_assessment(condition, snapshot)
    if deterministic is not None:
        assessment, reason = deterministic
        return _SpotCheckOutput(
            assessment=assessment,
            reasoning=reason,
            recommended_action="Review the finance snapshot and update the linked decision if needed.",
        )

    last_error: Exception | None = None
    user_message = f"""
Original decision: {condition.query_preview}
Condition type: {condition.kind.upper()}
Condition text: {condition.text}
Tracked metrics: {condition.metric_targets}
Threshold hint: {condition.threshold_hint}
User's current update: {user_update}
Structured finance snapshot: {snapshot if snapshot else 'None'}

Has this finance condition fired, is it still clear, or is it uncertain?
""".strip()

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = await _get_client().chat.completions.create(
                model=_SPOT_CHECK_MODEL,
                messages=[
                    {"role": "system", "content": _SPOT_CHECK_SYSTEM + "\n\nRespond with valid JSON only."},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = (response.choices[0].message.content or "").strip()
            
            # Bulletproof Markdown Stripping
            if raw.startswith("```json"):
                raw = raw[7:]
            elif raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            
            return _SpotCheckOutput.model_validate_json(raw)
        except (APITimeoutError, APIStatusError) as e:
            status_code = getattr(e, "status_code", None)
            if status_code and status_code not in (429, 500, 502, 503, 504):
                raise
            last_error = e
            await asyncio.sleep((2 ** attempt) + random.uniform(0, 0.5))
        except Exception as e:
            last_error = e
            break

    raise RuntimeError("Spot-check failed after retries") from last_error


# ── PUBLIC API ────────────────────────────────────────────────────────────

async def register_conditions(
    company_id: str,
    decision_id: str,
    query: str,
    verdict: VerdictBox,
) -> bool:
    new_records = _build_records_from_verdict(company_id, decision_id, query, verdict)
    if not new_records:
        return True

    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        t0 = time.perf_counter()
        store = await _read_store(company_id) or _new_store(company_id)
        existing_ids = {r.condition_id for r in store.records}
        to_add = [r for r in new_records if r.condition_id not in existing_ids]
        store.records.extend(to_add)
        store.last_updated_utc = datetime.now(timezone.utc).isoformat()
        success = await _write_store(company_id, store)
        logger.info(
            "✅ Conditions registered | company_hash=%s | decision=%s | added=%d | total=%d | latency_ms=%d",
            _company_hash(company_id)[:12],
            decision_id,
            len(to_add),
            len(store.records),
            round((time.perf_counter() - t0) * 1000),
        )
        return success
    except Exception as e:
        logger.error("register_conditions error | %s", e)
        return False
    finally:
        lock.release()


async def get_active_conditions(company_id: str) -> ConditionReport:
    store = await _read_store(company_id)
    now = datetime.now(timezone.utc).isoformat()
    if not store:
        return ConditionReport(
            company_id=company_id,
            total_active=0,
            total_fired=0,
            total_acknowledged=0,
            urgent=[],
            needs_review=[],
            go_conditions=[],
            generated_utc=now,
        )

    updated = [_check_expiry(r) for r in store.records]
    if any(r.status == "expired" for r in updated):
        store.records = updated
        store.last_updated_utc = now
        asyncio.create_task(_write_store(company_id, store))

    active = [r for r in store.records if r.status == "active"]
    fired = [r for r in store.records if r.status == "fired"]
    acked = [r for r in store.records if r.status == "acknowledged"]

    urgent = sorted(
        [r for r in fired if r.kind == "stop"],
        key=lambda r: r.fired_utc or r.created_utc,
    )
    needs_review = sorted(
        [r for r in fired if r.kind in ("review", "go")],
        key=lambda r: r.fired_utc or r.created_utc,
    )
    go_all = sorted(
        [r for r in store.records if r.kind == "go" and r.status in ("active", "fired")],
        key=lambda r: r.status,
    )

    return ConditionReport(
        company_id=company_id,
        total_active=len(active),
        total_fired=len(fired),
        total_acknowledged=len(acked),
        urgent=urgent,
        needs_review=needs_review,
        go_conditions=go_all,
        generated_utc=now,
    )


async def acknowledge_condition(company_id: str, condition_id: str) -> bool:
    return await _update_condition_status(company_id, condition_id, "acknowledged", "acknowledged_utc")


async def clear_condition(company_id: str, condition_id: str) -> bool:
    return await _update_condition_status(company_id, condition_id, "cleared", "cleared_utc")


async def fire_condition(company_id: str, condition_id: str) -> bool:
    return await _update_condition_status(company_id, condition_id, "fired", "fired_utc")


async def _update_condition_status(
    company_id: str,
    condition_id: str,
    new_status: ConditionStatus,
    timestamp_field: str,
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

        for record in store.records:
            if record.condition_id == condition_id:
                record.status = new_status
                setattr(record, timestamp_field, datetime.now(timezone.utc).isoformat())
                record.last_checked_utc = datetime.now(timezone.utc).isoformat()
                store.last_updated_utc = datetime.now(timezone.utc).isoformat()
                return await _write_store(company_id, store)
        return False
    except Exception as e:
        logger.error("_update_condition_status error | %s", e)
        return False
    finally:
        lock.release()


async def run_spot_check(
    company_id: str,
    condition_id: str,
    user_update: str,
    context: UserContext | None = None,
) -> SpotCheckResult | None:
    store = await _read_store(company_id)
    if not store:
        return None

    target = next((r for r in store.records if r.condition_id == condition_id), None)
    if target is None or target.status in ("cleared", "expired"):
        return None

    snapshot = _snapshot_from_context(context)

    try:
        result = await _run_spot_check(target, user_update, context=context)
    except RuntimeError as e:
        logger.error("Spot-check LLM failed | %s", e)
        return None

    status_map: dict[str, ConditionStatus] = {
        "likely_fired": "fired",
        "likely_clear": "active",
        "uncertain": "active",
    }
    new_status: ConditionStatus = status_map[result.assessment]

    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        pass
    else:
        try:
            store2 = await _read_store(company_id)
            if store2:
                for r in store2.records:
                    if r.condition_id == condition_id:
                        r.status = new_status
                        r.last_checked_utc = datetime.now(timezone.utc).isoformat()
                        r.spot_check_note = result.reasoning
                        r.last_evaluated_metrics = {k: snapshot.get(k) for k in r.metric_targets if k in snapshot}
                        r.evaluation_source = "deterministic_or_llm"
                        if new_status == "fired":
                            r.fired_utc = datetime.now(timezone.utc).isoformat()
                        break
                store2.last_updated_utc = datetime.now(timezone.utc).isoformat()
                await _write_store(company_id, store2)
        except Exception as e:
            logger.error("Spot-check write error | %s", e)
        finally:
            lock.release()

    return SpotCheckResult(
        condition_id=condition_id,
        condition_text=target.text,
        kind=target.kind,
        user_update=user_update,
        assessment=result.assessment,
        reasoning=result.reasoning,
        recommended_action=result.recommended_action,
        new_status=new_status,
    )


async def request_reanalysis(company_id: str, condition_id: str) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        store = await _read_store(company_id)
        if not store:
            return False

        for record in store.records:
            if record.condition_id == condition_id:
                record.reanalysis_requested = True
                record.last_checked_utc = datetime.now(timezone.utc).isoformat()
                store.last_updated_utc = datetime.now(timezone.utc).isoformat()
                return await _write_store(company_id, store)
        return False
    except Exception as e:
        logger.error("request_reanalysis error | %s", e)
        return False
    finally:
        lock.release()


async def get_reanalysis_queue(company_id: str) -> list[ConditionRecord]:
    store = await _read_store(company_id)
    if not store:
        return []
    return [r for r in store.records if r.reanalysis_requested]


async def clear_all_conditions(company_id: str) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        result = await delete_store(_TABLE, _company_hash(company_id))
        if result:
            logger.info("🗑️ Conditions cleared | company_hash=%s", _company_hash(company_id)[:12])
        return result
    except Exception as e:
        logger.error("clear_all_conditions error | %s", e)
        return False
    finally:
        lock.release()