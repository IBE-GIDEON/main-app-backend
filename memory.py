"""
memory.py — Three AI Finance Memory System (COMPLETE)
Storage backend: Supabase
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import asyncio
import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from router import UserContext, RoutingPlan
from output import UIDecisionPayload
from db import read_store, write_store, delete_store

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

_TABLE              = "memory_store"
_MAX_HISTORY        = int(os.getenv("THINK_AI_MEMORY_MAX_HISTORY", "50"))
_MAX_ASSUMPTIONS    = int(os.getenv("THINK_AI_MEMORY_MAX_ASSUMPTIONS", "20"))
_LOCK_TIMEOUT_SEC   = float(os.getenv("THINK_AI_MEMORY_LOCK_TIMEOUT", "5"))
_MEMORY_VERSION     = "2.0.0-finance"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("three-ai.memory")


# ─────────────────────────────────────────────
# LOCKS
# ─────────────────────────────────────────────

_write_locks: dict[str, asyncio.Lock] = {}

def _get_lock(company_id: str) -> asyncio.Lock:
    if company_id not in _write_locks:
        _write_locks[company_id] = asyncio.Lock()
    return _write_locks[company_id]


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class CompanyProfile(BaseModel):
    industry: str | None = None
    company_size: str | None = None
    risk_appetite: str | None = None
    primary_market: str | None = None
    team_size: str | None = None
    client_count: str | None = None
    revenue_range: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class FinanceProfile(BaseModel):
    reporting_currency: str | None = None
    planning_horizon_days: int | None = None

    prefers_growth: bool = False
    prefers_profitability: bool = False
    prefers_cash_safety: bool = False

    primary_kpis: list[str] = Field(default_factory=list)
    connected_sources: list[str] = Field(default_factory=list)

    last_known_runway: float | None = None
    last_known_cash: float | None = None
    last_finance_update_utc: str | None = None


class DecisionSummary(BaseModel):
    decision_id: str
    query_preview: str
    decision_type: str
    stake_level: str
    framework: str
    verdict_color: str
    confidence: float
    timestamp_utc: str
    user_rating: int | None = None


class BehaviorPattern(BaseModel):
    most_common_types: list[str] = Field(default_factory=list)
    most_used_lenses: list[str] = Field(default_factory=list)
    avg_stake_level: str = "Medium"
    prefers_quick: bool = False
    high_risk_ratio: float = 0.0
    total_decisions: int = 0


class MemoryDocument(BaseModel):
    company_id_hash: str
    version: str = _MEMORY_VERSION
    created_utc: str
    last_updated_utc: str

    profile: CompanyProfile = Field(default_factory=CompanyProfile)
    finance: FinanceProfile = Field(default_factory=FinanceProfile)

    behavior: BehaviorPattern = Field(default_factory=BehaviorPattern)
    decision_history: list[DecisionSummary] = Field(default_factory=list)

    running_assumptions: list[str] = Field(default_factory=list)
    preference_signals: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────

def _company_hash(company_id: str) -> str:
    return hashlib.sha256(company_id.encode()).hexdigest()


async def _read_doc(company_id: str) -> MemoryDocument | None:
    raw = await read_store(_TABLE, _company_hash(company_id))
    if raw is None:
        return None
    try:
        return MemoryDocument.model_validate(raw)
    except Exception as e:
        logger.error("Memory parse error | company_hash=%s | %s", _company_hash(company_id)[:12], e)
        return None


async def _write_doc(company_id: str, doc: MemoryDocument) -> bool:
    if len(doc.decision_history) > _MAX_HISTORY:
        doc.decision_history = doc.decision_history[-_MAX_HISTORY:]
    return await write_store(_TABLE, _company_hash(company_id), doc.model_dump())


def _new_document(company_id: str) -> MemoryDocument:
    now = datetime.now(timezone.utc).isoformat()
    return MemoryDocument(
        company_id_hash=_company_hash(company_id),
        created_utc=now,
        last_updated_utc=now,
    )


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _merge_profile(doc: MemoryDocument, ctx: UserContext) -> MemoryDocument:
    p = doc.profile
    if ctx.industry and not p.industry:
        p.industry = ctx.industry
    if ctx.company_size and not p.company_size:
        p.company_size = ctx.company_size
    if ctx.risk_appetite and not p.risk_appetite:
        p.risk_appetite = ctx.risk_appetite

    for k, v in (ctx.extra or {}).items():
        if k not in p.extra and not str(k).startswith("_"):
            p.extra[k] = str(v)

    return doc


def _update_finance_memory(doc: MemoryDocument, payload: UIDecisionPayload) -> MemoryDocument:
    snap = getattr(payload, "finance_snapshot", None)
    if not snap:
        return doc

    if hasattr(snap, "model_dump"):
        s = snap.model_dump()
    elif isinstance(snap, dict):
        s = snap
    else:
        s = dict(snap)

    f = doc.finance

    if s.get("currency"):
        f.reporting_currency = s["currency"]

    if s.get("analysis_horizon_days") is not None:
        try:
            f.planning_horizon_days = int(s["analysis_horizon_days"])
        except Exception:
            pass

    sources = s.get("sources_used") or []
    if isinstance(sources, list):
        f.connected_sources = list(dict.fromkeys([*f.connected_sources, *[str(x) for x in sources]]))

    if s.get("runway_months") is not None:
        try:
            f.last_known_runway = float(s["runway_months"])
        except Exception:
            pass

    if s.get("cash_balance") is not None:
        try:
            f.last_known_cash = float(s["cash_balance"])
        except Exception:
            pass

    if s.get("as_of_utc"):
        f.last_finance_update_utc = str(s["as_of_utc"])

    if s.get("runway_months") is not None:
        try:
            if float(s["runway_months"]) < 6:
                f.prefers_cash_safety = True
        except Exception:
            pass

    if s.get("revenue_growth_pct") is not None:
        try:
            if float(s["revenue_growth_pct"]) > 20:
                f.prefers_growth = True
        except Exception:
            pass

    if s.get("gross_margin_pct") is not None:
        try:
            if float(s["gross_margin_pct"]) > 50:
                f.prefers_profitability = True
        except Exception:
            pass

    kpis = [
        "cash_balance",
        "runway_months",
        "monthly_burn",
        "revenue_last_30d",
        "revenue_growth_pct",
        "gross_margin_pct",
        "failed_payment_rate_pct",
        "customer_concentration_pct",
    ]
    for k in kpis:
        if s.get(k) is not None and k not in f.primary_kpis:
            f.primary_kpis.append(k)

    return doc


def _update_behavior(doc: MemoryDocument, plan: RoutingPlan) -> MemoryDocument:
    b = doc.behavior
    history = doc.decision_history

    b.total_decisions += 1

    all_types = [h.decision_type for h in history] + [plan.decision_type]
    type_freq = {t: all_types.count(t) for t in set(all_types)}
    b.most_common_types = sorted(type_freq, key=type_freq.get, reverse=True)[:3]

    if plan.selected_lenses:
        lens_counts = doc.preference_signals.get("lens_counts", {})
        for lens in plan.selected_lenses:
            lens_counts[lens] = lens_counts.get(lens, 0) + 1
        doc.preference_signals["lens_counts"] = lens_counts
        b.most_used_lenses = sorted(lens_counts, key=lens_counts.get, reverse=True)[:5]

    high_count = sum(1 for h in history if h.stake_level == "High")
    if plan.stake_level == "High":
        high_count += 1
    b.high_risk_ratio = round(high_count / b.total_decisions, 2)

    stake_weights = {"Low": 1, "Medium": 2, "High": 3}
    all_stakes = [h.stake_level for h in history] + [plan.stake_level]
    avg_weight = sum(stake_weights.get(s, 2) for s in all_stakes) / len(all_stakes)
    b.avg_stake_level = "High" if avg_weight > 2.4 else ("Low" if avg_weight < 1.6 else "Medium")

    low_count = sum(1 for h in history if h.stake_level == "Low")
    if plan.stake_level == "Low":
        low_count += 1
    b.prefers_quick = (low_count / b.total_decisions) > 0.6

    return doc


def _append_history(
    doc: MemoryDocument,
    query: str,
    plan: RoutingPlan,
    payload: UIDecisionPayload,
) -> MemoryDocument:
    decision_id = hashlib.sha256(
        f"{query}{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()[:16]

    verdict_color = "Yellow"
    try:
        verdict_color = payload.verdict_card.badge.raw if getattr(payload.verdict_card, "badge", None) else getattr(payload.verdict_card, "color", "Yellow")
    except Exception:
        pass

    summary = DecisionSummary(
        decision_id=decision_id,
        query_preview=query[:80],
        decision_type=plan.decision_type,
        stake_level=plan.stake_level,
        framework=plan.framework,
        verdict_color=verdict_color,
        confidence=payload.confidence,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    doc.decision_history.append(summary)

    if len(doc.decision_history) > _MAX_HISTORY:
        doc.decision_history = doc.decision_history[-_MAX_HISTORY:]

    return doc


def _merge_assumptions(doc: MemoryDocument, plan: RoutingPlan) -> MemoryDocument:
    existing = set(doc.running_assumptions)
    for assumption in plan.assumptions:
        if assumption not in existing:
            doc.running_assumptions.append(assumption)
            existing.add(assumption)

    if len(doc.running_assumptions) > _MAX_ASSUMPTIONS:
        doc.running_assumptions = doc.running_assumptions[-_MAX_ASSUMPTIONS:]

    return doc


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

async def enrich_context(company_id: str, ctx: UserContext | None) -> UserContext:
    t0 = time.perf_counter()
    doc = await _read_doc(company_id)

    if doc is None:
        logger.info("No memory yet | company_hash=%s", _company_hash(company_id)[:12])
        return ctx or UserContext()

    p = doc.profile
    f = doc.finance
    b = doc.behavior
    base = ctx or UserContext()

    finance_priority = (
        "cash"
        if f.prefers_cash_safety
        else "growth"
        if f.prefers_growth
        else "profitability"
        if f.prefers_profitability
        else ""
    )

    enriched = UserContext(
        industry=base.industry or p.industry,
        company_size=base.company_size or p.company_size,
        risk_appetite=base.risk_appetite or p.risk_appetite,
        extra={
            **p.extra,
            **(base.extra or {}),
            "_total_decisions": str(b.total_decisions),
            "_avg_stake": b.avg_stake_level,
            "_prefers_quick": str(b.prefers_quick),
            "_common_types": ", ".join(b.most_common_types) if b.most_common_types else "",
            "_preferred_lenses": ", ".join(b.most_used_lenses) if b.most_used_lenses else "",
            "_running_assumptions": "; ".join(doc.running_assumptions[-5:]),
            "_currency": f.reporting_currency or "",
            "_runway": str(f.last_known_runway or ""),
            "_cash": str(f.last_known_cash or ""),
            "_finance_sources": ", ".join(f.connected_sources),
            "_finance_priority": finance_priority,
            "_finance_kpis": ", ".join(f.primary_kpis),
            "_finance_last_update": f.last_finance_update_utc or "",
        },
    )

    latency_ms = round((time.perf_counter() - t0) * 1000)
    logger.info(
        "✅ Context enriched | company_hash=%s | decisions_seen=%d | latency_ms=%d",
        _company_hash(company_id)[:12], b.total_decisions, latency_ms,
    )
    return enriched


async def update_memory(
    company_id: str,
    query: str,
    plan: RoutingPlan,
    payload: UIDecisionPayload,
) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning("Memory lock timeout | company_hash=%s", _company_hash(company_id)[:12])
        return False

    try:
        t0 = time.perf_counter()
        doc = await _read_doc(company_id) or _new_document(company_id)
        doc = _update_finance_memory(doc, payload)
        doc = _update_behavior(doc, plan)
        doc = _append_history(doc, query, plan, payload)
        doc = _merge_assumptions(doc, plan)
        doc.last_updated_utc = datetime.now(timezone.utc).isoformat()

        success = await _write_doc(company_id, doc)
        latency_ms = round((time.perf_counter() - t0) * 1000)
        if success:
            logger.info(
                "✅ Memory updated | company_hash=%s | total_decisions=%d | latency_ms=%d",
                _company_hash(company_id)[:12], doc.behavior.total_decisions, latency_ms,
            )
        return success
    except Exception as e:
        logger.error("Memory update error | company_hash=%s | %s", _company_hash(company_id)[:12], e)
        return False
    finally:
        lock.release()


async def update_profile(company_id: str, profile_data: dict[str, Any]) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        doc = await _read_doc(company_id) or _new_document(company_id)
        p = doc.profile
        f = doc.finance

        known_profile = {
            "industry", "company_size", "risk_appetite",
            "primary_market", "team_size", "client_count", "revenue_range",
        }
        known_finance = {
            "reporting_currency", "planning_horizon_days",
            "prefers_growth", "prefers_profitability", "prefers_cash_safety",
        }

        for k, v in profile_data.items():
            if k in known_profile:
                setattr(p, k, str(v))
            elif k in known_finance:
                if k == "planning_horizon_days":
                    try:
                        setattr(f, k, int(v))
                    except Exception:
                        pass
                elif k in {"prefers_growth", "prefers_profitability", "prefers_cash_safety"}:
                    setattr(f, k, bool(v))
                else:
                    setattr(f, k, str(v))
            else:
                p.extra[k] = str(v)

        doc.last_updated_utc = datetime.now(timezone.utc).isoformat()
        success = await _write_doc(company_id, doc)
        if success:
            logger.info("✅ Profile updated | company_hash=%s", _company_hash(company_id)[:12])
        return success
    except Exception as e:
        logger.error("Profile update error | company_hash=%s | %s", _company_hash(company_id)[:12], e)
        return False
    finally:
        lock.release()


async def update_rating(company_id: str, decision_id: str, rating: int) -> bool:
    if rating not in range(1, 6):
        return False

    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        doc = await _read_doc(company_id)
        if not doc:
            return False

        for summary in doc.decision_history:
            if summary.decision_id == decision_id:
                summary.user_rating = rating

                if rating < 3:
                    signals = doc.preference_signals.get("low_rated_types", [])
                    if summary.decision_type not in signals:
                        signals.append(summary.decision_type)
                    doc.preference_signals["low_rated_types"] = signals

                    neg_verdicts = doc.preference_signals.get("low_rated_verdicts", [])
                    if summary.verdict_color not in neg_verdicts:
                        neg_verdicts.append(summary.verdict_color)
                    doc.preference_signals["low_rated_verdicts"] = neg_verdicts

                doc.last_updated_utc = datetime.now(timezone.utc).isoformat()
                success = await _write_doc(company_id, doc)
                logger.info(
                    "✅ Rating saved | company_hash=%s | decision=%s | rating=%d",
                    _company_hash(company_id)[:12], decision_id, rating,
                )
                return success

        return False
    except Exception as e:
        logger.error("Rating update error | company_hash=%s | %s", _company_hash(company_id)[:12], e)
        return False
    finally:
        lock.release()


async def get_memory_summary(company_id: str) -> dict[str, Any] | None:
    doc = await _read_doc(company_id)
    if not doc:
        return None

    return {
        "total_decisions": doc.behavior.total_decisions,
        "most_common_types": doc.behavior.most_common_types,
        "most_used_lenses": doc.behavior.most_used_lenses,
        "avg_stake_level": doc.behavior.avg_stake_level,
        "high_risk_ratio": doc.behavior.high_risk_ratio,
        "profile": {
            "industry": doc.profile.industry,
            "company_size": doc.profile.company_size,
            "risk_appetite": doc.profile.risk_appetite,
        },
        "finance": {
            "reporting_currency": doc.finance.reporting_currency,
            "planning_horizon_days": doc.finance.planning_horizon_days,
            "connected_sources": doc.finance.connected_sources,
            "last_known_runway": doc.finance.last_known_runway,
            "last_known_cash": doc.finance.last_known_cash,
            "primary_kpis": doc.finance.primary_kpis,
            "last_finance_update_utc": doc.finance.last_finance_update_utc,
        },
        "last_updated_utc": doc.last_updated_utc,
        "memory_version": doc.version,
    }


async def clear_memory(company_id: str) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        result = await delete_store(_TABLE, _company_hash(company_id))
        if result:
            logger.info("🗑️ Memory cleared | company_hash=%s", _company_hash(company_id)[:12])
        return result
    except Exception as e:
        logger.error("Memory clear error | %s", e)
        return False
    finally:
        lock.release()