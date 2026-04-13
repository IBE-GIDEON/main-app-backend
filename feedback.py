"""
feedback.py — Three AI Finance Feedback & Learning Loop
Storage backend: Supabase

Finance-first changes:
- Adds finance-aware wrong_aspects categories
- Learns finance preferences from low/high rated decisions
- Stores finance learning signals into memory
- Keeps public API signatures compatible
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import asyncio
import hashlib
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI, APIStatusError, APITimeoutError
from pydantic import BaseModel, Field, field_validator

from router import UserContext
from audit import get_audit_record, AuditRecord
from memory import update_rating, _read_doc, _write_doc, _get_lock as _mem_lock
from db import read_store, write_store, delete_store

_TABLE               = "feedback_store"
_LOCK_TIMEOUT_SEC    = float(os.getenv("THINK_AI_FEEDBACK_LOCK", "5"))
_MAX_RECORDS         = int(os.getenv("THINK_AI_FEEDBACK_MAX", "1000"))
_MAX_COMMENT_LEN     = int(os.getenv("THINK_AI_FEEDBACK_MAX_COMMENT", "500"))
_LEARNING_THRESHOLD  = int(os.getenv("THINK_AI_FEEDBACK_THRESHOLD", "3"))
_LEARNING_MODEL      = os.getenv("THINK_AI_FEEDBACK_MODEL", "gpt-4o-mini")
_LEARNING_TIMEOUT    = float(os.getenv("THINK_AI_FEEDBACK_TIMEOUT", "30"))
_MAX_RETRIES         = int(os.getenv("THINK_AI_REFINER_RETRIES", "3"))
_FEEDBACK_VERSION    = "2.0.0-finance"

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        load_dotenv()
        _client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            timeout=_LEARNING_TIMEOUT,
            max_retries=0,
        )
    return _client


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("three-ai.feedback")


VALID_ASPECTS = {
    # original/generic
    "verdict",
    "risk_analysis",
    "upside_analysis",
    "lenses",
    "confidence",
    "conditions",
    "next_steps",
    "stake_level",
    "framework",
    "depth",
    "other",

    # finance-aware
    "wrong_metric",
    "stale_data",
    "missed_risk",
    "weak_upside",
    "wrong_priority",
    "wrong_threshold",
    "cash_flow",
    "runway",
    "revenue_quality",
    "margin",
    "collections",
    "concentration",
    "forecasting",
    "debt_covenants",
}

LOW_RATING_THRESHOLD = 3
LEARNING_SAMPLE_SIZE = 10


class FeedbackRecord(BaseModel):
    feedback_id: str
    company_id_hash: str
    decision_id: str
    record_id: str
    rating: int
    comment: str | None
    wrong_aspects: list[str]
    created_utc: str
    triggered_learning: bool = False

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v):
        if v not in range(1, 6):
            raise ValueError("Rating must be 1–5")
        return v

    @field_validator("wrong_aspects")
    @classmethod
    def validate_aspects(cls, v):
        return [a for a in v if a in VALID_ASPECTS]


class CompanyFeedbackStore(BaseModel):
    company_id_hash: str
    version: str = _FEEDBACK_VERSION
    records: list[FeedbackRecord] = Field(default_factory=list)
    last_updated_utc: str


class LearningSignals(BaseModel):
    avoided_lenses: list[str]
    preferred_lenses: list[str]
    preferred_framework: str | None
    depth_preference: str | None

    common_failure_modes: list[str]
    common_success_modes: list[str]
    learning_summary: str

    # finance-aware
    finance_focus: list[str] = Field(default_factory=list)
    recurring_metric_blindspots: list[str] = Field(default_factory=list)
    recurring_data_quality_issues: list[str] = Field(default_factory=list)
    preferred_finance_priorities: list[str] = Field(default_factory=list)

    last_learned_utc: str
    low_rated_count: int
    high_rated_count: int


class FeedbackInsights(BaseModel):
    company_id: str
    total_ratings: int
    avg_rating: float
    low_rated_count: int
    high_rated_count: int
    most_flagged_aspects: list[str]
    learning_signals: LearningSignals | None
    rating_trend: str
    generated_utc: str


_write_locks: dict[str, asyncio.Lock] = {}


def _get_lock(company_id: str) -> asyncio.Lock:
    if company_id not in _write_locks:
        _write_locks[company_id] = asyncio.Lock()
    return _write_locks[company_id]


def _company_hash(company_id: str) -> str:
    return hashlib.sha256(company_id.encode()).hexdigest()


async def _read_store(company_id: str) -> CompanyFeedbackStore | None:
    raw = await read_store(_TABLE, _company_hash(company_id))
    if raw is None:
        return None
    try:
        return CompanyFeedbackStore.model_validate(raw)
    except Exception as e:
        logger.error("Feedback parse error | %s", e)
        return None


async def _write_store(company_id: str, store: CompanyFeedbackStore) -> bool:
    if len(store.records) > _MAX_RECORDS:
        store.records = store.records[-_MAX_RECORDS:]
    return await write_store(_TABLE, _company_hash(company_id), store.model_dump())


def _new_store(company_id: str) -> CompanyFeedbackStore:
    return CompanyFeedbackStore(
        company_id_hash=_company_hash(company_id),
        records=[],
        last_updated_utc=datetime.now(timezone.utc).isoformat(),
    )


_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard the above",
    "you are now",
    "act as",
    "system:",
    "assistant:",
    "jailbreak",
]


def _sanitise_comment(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = raw.strip()[:_MAX_COMMENT_LEN]
    lower = cleaned.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lower:
            return None
    return cleaned if cleaned else None


_LEARNING_SYSTEM = """
You are Three AI's Finance Learning Analyst.

You are given low-rated and high-rated finance decision analyses from one company.
Infer what this company values, what the system gets wrong, and how future finance analysis should improve.

Focus on:
- stale or missing data
- wrong metric selection
- cash/runway blind spots
- weak revenue quality reasoning
- margin/collections/concentration misses
- thresholds and priority mistakes

Return valid JSON only.
""".strip()


class _LearningOutput(BaseModel):
    avoided_lenses: list[str]
    preferred_lenses: list[str]
    preferred_framework: str | None
    depth_preference: str | None
    common_failure_modes: list[str]
    common_success_modes: list[str]
    learning_summary: str

    finance_focus: list[str] = Field(default_factory=list)
    recurring_metric_blindspots: list[str] = Field(default_factory=list)
    recurring_data_quality_issues: list[str] = Field(default_factory=list)
    preferred_finance_priorities: list[str] = Field(default_factory=list)


def _extract_finance_summary(r: AuditRecord) -> str:
    finance = getattr(r, "finance_snapshot", None)

    if not finance:
        return "Finance snapshot: not stored."

    parts: list[str] = []
    if getattr(finance, "cash_balance", None) is not None:
        parts.append(f"cash={finance.cash_balance}")
    if getattr(finance, "runway_months", None) is not None:
        parts.append(f"runway={finance.runway_months}m")
    if getattr(finance, "revenue_growth_pct", None) is not None:
        parts.append(f"rev_growth={finance.revenue_growth_pct}%")
    if getattr(finance, "gross_margin_pct", None) is not None:
        parts.append(f"gross_margin={finance.gross_margin_pct}%")
    if getattr(finance, "failed_payment_rate_pct", None) is not None:
        parts.append(f"failed_payments={finance.failed_payment_rate_pct}%")
    if getattr(finance, "customer_concentration_pct", None) is not None:
        parts.append(f"concentration={finance.customer_concentration_pct}%")

    srcs = getattr(finance, "sources_used", []) or []
    source_text = f" sources={', '.join(srcs)}" if srcs else ""
    return "Finance snapshot: " + (", ".join(parts) if parts else "present but sparse") + source_text


async def _run_learning_analysis(
    company_id: str,
    low_rated_records: list[AuditRecord],
    high_rated_records: list[AuditRecord],
    aspects_flagged: list[str],
) -> _LearningOutput | None:
    if not low_rated_records:
        return None

    def _summarise(r: AuditRecord, label: str) -> str:
        v = r.verdict_snapshot
        s = r.routing_snapshot
        return (
            f"[{label}] Query: {r.query_preview}\n"
            f"  Type={s.decision_type} Stake={s.stake_level} Lenses={', '.join(s.selected_lenses)}\n"
            f"  Verdict={v.color} NetScore={v.net_score:+.1f} | Key unknown: {v.key_unknown}\n"
            f"  {_extract_finance_summary(r)}"
        )

    low_block = "\n\n".join(
        _summarise(r, f"LOW-RATED #{i+1}")
        for i, r in enumerate(low_rated_records[:LEARNING_SAMPLE_SIZE])
    )
    high_block = "\n\n".join(
        _summarise(r, f"HIGH-RATED #{i+1}")
        for i, r in enumerate(high_rated_records[:5])
    ) or "None yet."

    user_message = f"""
Company: {_company_hash(company_id)[:12]} (anonymised)
Aspects flagged: {', '.join(aspects_flagged) if aspects_flagged else 'Not specified'}

LOW-RATED FINANCE DECISIONS:
{low_block}

HIGH-RATED FINANCE DECISIONS (for contrast):
{high_block}

What patterns explain the low ratings, and what should change in future finance analysis?
""".strip()

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = await _get_client().chat.completions.create(
                model=_LEARNING_MODEL,
                messages=[
                    {"role": "system", "content": _LEARNING_SYSTEM + "\n\nRespond with valid JSON only."},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
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
            
            return _LearningOutput.model_validate_json(raw)
        except (APITimeoutError, APIStatusError) as e:
            status_code = getattr(e, "status_code", None)
            if status_code and status_code not in (429, 500, 502, 503, 504):
                raise
            last_error = e
            await asyncio.sleep((2 ** attempt) + random.uniform(0, 0.5))
        except Exception as e:
            last_error = e
            break

    logger.error("Learning analysis failed | %s", last_error)
    return None


def _derive_finance_learning_from_aspects(aspects: list[str]) -> tuple[list[str], list[str], list[str]]:
    focus: list[str] = []
    blindspots: list[str] = []
    data_issues: list[str] = []

    mapping = {
        "cash_flow": "cash flow",
        "runway": "runway",
        "revenue_quality": "revenue quality",
        "margin": "margins",
        "collections": "collections",
        "concentration": "customer concentration",
        "forecasting": "forecasting",
        "debt_covenants": "debt/covenants",
    }

    for aspect in aspects:
        if aspect in mapping and mapping[aspect] not in focus:
            focus.append(mapping[aspect])

        if aspect == "wrong_metric":
            blindspots.append("wrong metric selection")
        if aspect == "missed_risk":
            blindspots.append("missed downside risk")
        if aspect == "weak_upside":
            blindspots.append("weak upside detection")
        if aspect == "stale_data":
            data_issues.append("stale data")
        if aspect == "wrong_threshold":
            blindspots.append("bad thresholding")
        if aspect == "wrong_priority":
            blindspots.append("wrong finance prioritization")

    return focus, list(dict.fromkeys(blindspots)), list(dict.fromkeys(data_issues))


async def _write_learning_signals(company_id: str, signals: LearningSignals) -> bool:
    lock = _mem_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        doc = await _read_doc(company_id)
        if not doc:
            return False

        doc.preference_signals["learning_signals"] = signals.model_dump()
        doc.preference_signals["avoided_lenses"] = signals.avoided_lenses
        doc.preference_signals["preferred_lenses"] = signals.preferred_lenses
        doc.preference_signals["depth_preference"] = signals.depth_preference
        doc.preference_signals["common_failure_modes"] = signals.common_failure_modes
        doc.preference_signals["learning_summary"] = signals.learning_summary

        # finance-aware memory additions
        doc.preference_signals["finance_focus"] = signals.finance_focus
        doc.preference_signals["recurring_metric_blindspots"] = signals.recurring_metric_blindspots
        doc.preference_signals["recurring_data_quality_issues"] = signals.recurring_data_quality_issues
        doc.preference_signals["preferred_finance_priorities"] = signals.preferred_finance_priorities

        doc.last_updated_utc = datetime.now(timezone.utc).isoformat()
        return await _write_doc(company_id, doc)
    except Exception as e:
        logger.error("_write_learning_signals error | %s", e)
        return False
    finally:
        lock.release()


async def _trigger_learning(company_id: str, aspects_flagged: list[str]) -> None:
    try:
        store = await _read_store(company_id)
        if not store:
            return

        low_rated = [r for r in store.records if r.rating <= LOW_RATING_THRESHOLD]
        high_rated = [r for r in store.records if r.rating >= 4]

        if len(low_rated) < _LEARNING_THRESHOLD:
            return

        low_audit: list[AuditRecord] = []
        for fb in sorted(low_rated, key=lambda r: r.created_utc, reverse=True)[:LEARNING_SAMPLE_SIZE]:
            rec = await get_audit_record(company_id, fb.record_id)
            if rec:
                low_audit.append(rec)

        high_audit: list[AuditRecord] = []
        for fb in sorted(high_rated, key=lambda r: r.created_utc, reverse=True)[:5]:
            rec = await get_audit_record(company_id, fb.record_id)
            if rec:
                high_audit.append(rec)

        if not low_audit:
            return

        output = await _run_learning_analysis(company_id, low_audit, high_audit, aspects_flagged)
        if not output:
            return

        derived_focus, derived_blindspots, derived_data_issues = _derive_finance_learning_from_aspects(aspects_flagged)

        signals = LearningSignals(
            avoided_lenses=output.avoided_lenses,
            preferred_lenses=output.preferred_lenses,
            preferred_framework=output.preferred_framework,
            depth_preference=output.depth_preference,
            common_failure_modes=output.common_failure_modes,
            common_success_modes=output.common_success_modes,
            learning_summary=output.learning_summary,
            finance_focus=list(dict.fromkeys(output.finance_focus + derived_focus)),
            recurring_metric_blindspots=list(dict.fromkeys(output.recurring_metric_blindspots + derived_blindspots)),
            recurring_data_quality_issues=list(dict.fromkeys(output.recurring_data_quality_issues + derived_data_issues)),
            preferred_finance_priorities=output.preferred_finance_priorities,
            last_learned_utc=datetime.now(timezone.utc).isoformat(),
            low_rated_count=len(low_rated),
            high_rated_count=len(high_rated),
        )
        await _write_learning_signals(company_id, signals)

        lock = _get_lock(company_id)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            return

        try:
            store2 = await _read_store(company_id)
            if store2:
                for r in store2.records:
                    if r.rating <= LOW_RATING_THRESHOLD:
                        r.triggered_learning = True
                store2.last_updated_utc = datetime.now(timezone.utc).isoformat()
                await _write_store(company_id, store2)
        finally:
            lock.release()
    except Exception as e:
        logger.error("_trigger_learning error | %s", e)


def _calculate_trend(records: list[FeedbackRecord]) -> str:
    if len(records) < 4:
        return "insufficient_data"

    sorted_recs = sorted(records, key=lambda r: r.created_utc)
    recent = sorted_recs[-5:]
    prior = sorted_recs[-10:-5] if len(sorted_recs) >= 10 else sorted_recs[:-5]
    if not prior:
        return "insufficient_data"

    delta = (sum(r.rating for r in recent) / len(recent)) - (sum(r.rating for r in prior) / len(prior))
    if delta > 0.3:
        return "improving"
    if delta < -0.3:
        return "declining"
    return "stable"


# ── PUBLIC API ────────────────────────────────────────────────────────────

async def write_feedback(
    company_id: str,
    decision_id: str,
    record_id: str,
    rating: int,
    comment: str | None = None,
    wrong_aspects: list[str] | None = None,
) -> bool:
    if rating not in range(1, 6):
        return False

    comment = _sanitise_comment(comment)
    wrong_aspects = [a for a in (wrong_aspects or []) if a in VALID_ASPECTS]

    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        t0 = time.perf_counter()
        store = await _read_store(company_id) or _new_store(company_id)
        now = datetime.now(timezone.utc).isoformat()
        feedback_id = hashlib.sha256(f"{company_id}:{record_id}:{now}".encode()).hexdigest()[:16]

        record = FeedbackRecord(
            feedback_id=feedback_id,
            company_id_hash=_company_hash(company_id),
            decision_id=decision_id,
            record_id=record_id,
            rating=rating,
            comment=comment,
            wrong_aspects=wrong_aspects,
            created_utc=now,
        )
        store.records.append(record)
        store.last_updated_utc = now
        success = await _write_store(company_id, store)

        if success:
            logger.info(
                "✅ Feedback written | company_hash=%s | rating=%d | latency_ms=%d",
                _company_hash(company_id)[:12],
                rating,
                round((time.perf_counter() - t0) * 1000),
            )

        asyncio.create_task(update_rating(company_id, decision_id, rating))
        if rating <= LOW_RATING_THRESHOLD:
            asyncio.create_task(_trigger_learning(company_id, wrong_aspects))

        return success
    except Exception as e:
        logger.error("write_feedback error | %s", e)
        return False
    finally:
        lock.release()


async def get_feedback_insights(company_id: str) -> FeedbackInsights:
    store = await _read_store(company_id)
    now = datetime.now(timezone.utc).isoformat()

    if not store or not store.records:
        return FeedbackInsights(
            company_id=company_id,
            total_ratings=0,
            avg_rating=0.0,
            low_rated_count=0,
            high_rated_count=0,
            most_flagged_aspects=[],
            learning_signals=None,
            rating_trend="insufficient_data",
            generated_utc=now,
        )

    records = store.records
    total = len(records)
    avg_rating = round(sum(r.rating for r in records) / total, 2)
    low_rated = [r for r in records if r.rating <= LOW_RATING_THRESHOLD]
    high_rated = [r for r in records if r.rating >= 4]

    aspect_counts: dict[str, int] = {}
    for r in records:
        for a in r.wrong_aspects:
            aspect_counts[a] = aspect_counts.get(a, 0) + 1
    most_flagged = sorted(aspect_counts, key=aspect_counts.get, reverse=True)[:5]

    learning_signals: LearningSignals | None = None
    try:
        doc = await _read_doc(company_id)
        if doc and "learning_signals" in doc.preference_signals:
            learning_signals = LearningSignals.model_validate(doc.preference_signals["learning_signals"])
    except Exception as e:
        logger.warning("Could not read learning signals | %s", e)

    return FeedbackInsights(
        company_id=company_id,
        total_ratings=total,
        avg_rating=avg_rating,
        low_rated_count=len(low_rated),
        high_rated_count=len(high_rated),
        most_flagged_aspects=most_flagged,
        learning_signals=learning_signals,
        rating_trend=_calculate_trend(records),
        generated_utc=now,
    )


async def get_feedback_for_decision(company_id: str, decision_id: str) -> list[FeedbackRecord]:
    store = await _read_store(company_id)
    if not store:
        return []
    return [r for r in store.records if r.decision_id == decision_id]


async def clear_feedback(company_id: str) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        result = await delete_store(_TABLE, _company_hash(company_id))
        if result:
            logger.info("🗑️ Feedback cleared | company_hash=%s", _company_hash(company_id)[:12])
        return result
    except Exception as e:
        logger.error("clear_feedback error | %s", e)
        return False
    finally:
        lock.release()