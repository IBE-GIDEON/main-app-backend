"""
audit.py — Three AI Decision Audit & Record Keeping (FINANCE COMPLETE)
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

from router import RoutingPlan
from refiner import RefinedDecision
from output import UIDecisionPayload
from db import read_store, write_store, delete_store

_TABLE            = "audit_store"
_MAX_RECORDS      = int(os.getenv("THINK_AI_AUDIT_MAX", "500"))
_LOCK_TIMEOUT_SEC = float(os.getenv("THINK_AI_AUDIT_LOCK", "5"))
_AUDIT_VERSION    = "2.0.0-finance"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("three-ai.audit")


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class FinanceSnapshotRecord(BaseModel):
    as_of_utc: str | None = None
    is_live_data: bool = False

    sources_used: list[str] = Field(default_factory=list)
    source_freshness: dict[str, str] = Field(default_factory=dict)

    cash_balance: float | None = None
    runway_months: float | None = None
    monthly_burn: float | None = None

    revenue_last_30d: float | None = None
    revenue_growth_pct: float | None = None
    gross_margin_pct: float | None = None

    ar_overdue_30_plus: float | None = None
    failed_payment_rate_pct: float | None = None
    customer_concentration_pct: float | None = None

    notes: list[str] = Field(default_factory=list)


class DecisionSnapshot(BaseModel):
    decision_type: str
    stake_level: str
    framework: str
    selected_lenses: list[str]
    box_count: int
    reasoning_depth: int
    is_reversible: bool
    required_questions: list[str]
    assumptions: list[str]
    constraints_detected: list[str]
    confidence: float
    intent: str | None = None
    is_live_query: bool | None = None
    analysis_horizon_days: int | None = None
    finance_priority: str | None = None


class VerdictSnapshot(BaseModel):
    color: str
    headline: str
    rationale: str
    net_score: float
    go_conditions: list[str]
    stop_conditions: list[str]
    review_triggers: list[str]
    key_unknown: str
    flip_factor: str


class BoxSnapshot(BaseModel):
    box_type: str
    title: str
    color: str | None = None
    claim: str
    probability: int | None = None
    impact: int | None = None
    risk_score: float | None = None
    evidence_or_reasoning: str | None = None
    follow_up_actions: list[str] = Field(default_factory=list)
    spawn_questions: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class PerformanceSnapshot(BaseModel):
    model_used: str
    pass_latencies_ms: list[int]
    total_latency_ms: int
    output_latency_ms: int
    total_tokens_in: int
    total_tokens_out: int
    estimated_cost_usd: float


class AuditRecord(BaseModel):
    record_id: str
    decision_id: str
    company_id_hash: str
    version: int
    is_branch: bool

    query_hash: str
    query_preview: str
    user_id_hash: str | None = None

    routing_snapshot: DecisionSnapshot
    verdict_snapshot: VerdictSnapshot
    boxes: list[BoxSnapshot]
    performance: PerformanceSnapshot
    finance_snapshot: FinanceSnapshotRecord | None = None

    created_utc: str
    audit_version: str = _AUDIT_VERSION


class CompanyAuditStore(BaseModel):
    company_id_hash: str
    version: str = _AUDIT_VERSION
    records: list[AuditRecord] = Field(default_factory=list)
    last_updated_utc: str


class AuditListItem(BaseModel):
    record_id: str
    decision_id: str
    version: int
    query_preview: str
    decision_type: str
    stake_level: str
    verdict_color: str
    net_score: float
    confidence: float
    total_boxes: int
    is_branch: bool
    created_utc: str


class VersionDiff(BaseModel):
    decision_id: str
    version_a: int
    version_b: int
    changed_fields: list[str]
    verdict_changed: bool
    stake_changed: bool
    confidence_delta: float
    net_score_delta: float
    new_conditions: list[str]
    removed_conditions: list[str]
    summary: str


class AuditExport(BaseModel):
    decision_id: str
    version: int
    query_preview: str
    created_utc: str
    what_was_decided: str
    why: str
    key_assumptions: list[str]
    what_we_knew: str
    what_we_didnt_know: str
    go_conditions: list[str]
    stop_conditions: list[str]
    review_triggers: list[str]
    next_step: str
    performance: str


# ─────────────────────────────────────────────
# LOCKS
# ─────────────────────────────────────────────

_write_locks: dict[str, asyncio.Lock] = {}

def _get_lock(company_id: str) -> asyncio.Lock:
    if company_id not in _write_locks:
        _write_locks[company_id] = asyncio.Lock()
    return _write_locks[company_id]


# ─────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────

def _company_hash(company_id: str) -> str:
    return hashlib.sha256(company_id.encode()).hexdigest()


async def _read_store(company_id: str) -> CompanyAuditStore | None:
    raw = await read_store(_TABLE, _company_hash(company_id))
    if raw is None:
        return None
    try:
        return CompanyAuditStore.model_validate(raw)
    except Exception as e:
        logger.error("Audit parse error | %s", e)
        return None


async def _write_store(company_id: str, store: CompanyAuditStore) -> bool:
    if len(store.records) > _MAX_RECORDS:
        store.records = store.records[-_MAX_RECORDS:]
    return await write_store(_TABLE, _company_hash(company_id), store.model_dump())


def _new_store(company_id: str) -> CompanyAuditStore:
    return CompanyAuditStore(
        company_id_hash=_company_hash(company_id),
        records=[],
        last_updated_utc=datetime.now(timezone.utc).isoformat(),
    )


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _decision_id(company_id: str, query: str) -> str:
    raw = f"{company_id}:{query.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _record_id(decision_id: str, version: int) -> str:
    raw = f"{decision_id}:v{version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _hash_user_id(user_id: str) -> str:
    if "@" in user_id or len(user_id) > 32:
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    return user_id


def _estimate_cost(tokens_in: int, tokens_out: int, model: str) -> float:
    rates: dict[str, tuple[float, float]] = {
        "gpt-4o-mini": (0.000_000_150, 0.000_000_600),
        "gpt-4o":      (0.000_005_000, 0.000_015_000),
    }
    model_key = "gpt-4o" if "gpt-4o" in model and "mini" not in model else "gpt-4o-mini"
    rate_in, rate_out = rates.get(model_key, rates["gpt-4o-mini"])
    return round(tokens_in * rate_in + tokens_out * rate_out, 6)


def _build_routing_snapshot(plan: RoutingPlan) -> DecisionSnapshot:
    payload = plan.model_dump()
    return DecisionSnapshot(**payload)


def _build_verdict_snapshot(decision: RefinedDecision) -> VerdictSnapshot:
    vb = decision.verdict_box
    payload = vb.model_dump()
    return VerdictSnapshot(**payload)


def _build_box_snapshots(decision: RefinedDecision) -> list[BoxSnapshot]:
    boxes: list[BoxSnapshot] = []

    for b in decision.upside_boxes:
        b_dict = b.model_dump()
        b_dict["box_type"] = "upside"
        boxes.append(BoxSnapshot(**b_dict))

    for b in decision.risk_boxes:
        b_dict = b.model_dump()
        b_dict["box_type"] = "risk"
        boxes.append(BoxSnapshot(**b_dict))

    return boxes


def _build_performance_snapshot(
    decision: RefinedDecision,
    payload: UIDecisionPayload,
) -> PerformanceSnapshot:
    a = decision.audit
    total_lat = sum(a.pass_latencies_ms) + (payload.audit.output_latency_ms if payload.audit else 0)
    return PerformanceSnapshot(
        model_used=a.model_used,
        pass_latencies_ms=a.pass_latencies_ms,
        total_latency_ms=total_lat,
        output_latency_ms=payload.audit.output_latency_ms if payload.audit else 0,
        total_tokens_in=a.total_tokens_in,
        total_tokens_out=a.total_tokens_out,
        estimated_cost_usd=_estimate_cost(a.total_tokens_in, a.total_tokens_out, a.model_used),
    )


def _build_finance_snapshot(payload: UIDecisionPayload) -> FinanceSnapshotRecord | None:
    snap = getattr(payload, "finance_snapshot", None)
    if not snap:
        return None

    if hasattr(snap, "model_dump"):
        data = snap.model_dump()
    elif isinstance(snap, dict):
        data = snap
    else:
        data = dict(snap)

    return FinanceSnapshotRecord(**data)


def _diff_records(a: AuditRecord, b: AuditRecord) -> VersionDiff:
    changed: list[str] = []

    verdict_changed = a.verdict_snapshot.color != b.verdict_snapshot.color
    if verdict_changed:
        changed.append(f"Verdict changed: {a.verdict_snapshot.color} → {b.verdict_snapshot.color}")

    if a.verdict_snapshot.headline != b.verdict_snapshot.headline:
        changed.append("Verdict headline updated")

    stake_changed = a.routing_snapshot.stake_level != b.routing_snapshot.stake_level
    if stake_changed:
        changed.append(f"Stake level: {a.routing_snapshot.stake_level} → {b.routing_snapshot.stake_level}")

    conf_delta = round(b.routing_snapshot.confidence - a.routing_snapshot.confidence, 3)
    score_delta = round(b.verdict_snapshot.net_score - a.verdict_snapshot.net_score, 2)

    if abs(conf_delta) > 0.05:
        changed.append(f"Confidence {'↑' if conf_delta > 0 else '↓'} {abs(conf_delta):.0%}")
    if abs(score_delta) > 0.5:
        changed.append(f"Net score {'↑' if score_delta > 0 else '↓'} {abs(score_delta):.1f}")

    old_go = set(a.verdict_snapshot.go_conditions)
    new_go = set(b.verdict_snapshot.go_conditions)
    old_stop = set(a.verdict_snapshot.stop_conditions)
    new_stop = set(b.verdict_snapshot.stop_conditions)
    old_rev = set(a.verdict_snapshot.review_triggers)
    new_rev = set(b.verdict_snapshot.review_triggers)

    all_old = old_go | old_stop | old_rev
    all_new = new_go | new_stop | new_rev

    new_conds = list(all_new - all_old)
    removed_conds = list(all_old - all_new)

    if new_conds:
        changed.append(f"{len(new_conds)} new condition(s) added")
    if removed_conds:
        changed.append(f"{len(removed_conds)} condition(s) removed")

    if not changed:
        summary = "No significant changes detected between these two versions."
    elif verdict_changed:
        summary = f"Verdict flipped from {a.verdict_snapshot.color} to {b.verdict_snapshot.color}."
    elif stake_changed:
        summary = f"Stake level changed from {a.routing_snapshot.stake_level} to {b.routing_snapshot.stake_level}."
    else:
        summary = f"{len(changed)} aspect(s) changed since version {a.version}."

    return VersionDiff(
        decision_id=a.decision_id,
        version_a=a.version,
        version_b=b.version,
        changed_fields=changed,
        verdict_changed=verdict_changed,
        stake_changed=stake_changed,
        confidence_delta=conf_delta,
        net_score_delta=score_delta,
        new_conditions=new_conds,
        removed_conditions=removed_conds,
        summary=summary,
    )


def _build_export(record: AuditRecord) -> AuditExport:
    p = record.performance
    perf_str = (
        f"Analysed in {p.total_latency_ms / 1000:.1f}s using {p.model_used}. "
        f"{p.total_tokens_in + p.total_tokens_out:,} tokens. "
        f"Cost: ${p.estimated_cost_usd:.4f}."
    )
    r = record.routing_snapshot
    v = record.verdict_snapshot

    return AuditExport(
        decision_id=record.decision_id,
        version=record.version,
        query_preview=record.query_preview,
        created_utc=record.created_utc,
        what_was_decided=f"{v.headline} (Verdict: {v.color}, Net score: {v.net_score:+.1f})",
        why=v.rationale,
        key_assumptions=r.assumptions,
        what_we_knew=(
            f"Decision classified as {r.decision_type} with {r.stake_level} stakes. "
            f"Analysed through: {', '.join(r.selected_lenses)}."
        ),
        what_we_didnt_know=v.key_unknown,
        go_conditions=v.go_conditions,
        stop_conditions=v.stop_conditions,
        review_triggers=v.review_triggers,
        next_step=v.flip_factor,
        performance=perf_str,
    )


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

async def write_audit(
    company_id: str,
    query: str,
    plan: RoutingPlan,
    decision: RefinedDecision,
    payload: UIDecisionPayload,
    user_id: str | None = None,
) -> str | None:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning("Audit write lock timeout | company_hash=%s", _company_hash(company_id)[:12])
        return None

    try:
        t0 = time.perf_counter()
        store = await _read_store(company_id) or _new_store(company_id)
        d_id = _decision_id(company_id, query)
        existing_versions = [r.version for r in store.records if r.decision_id == d_id]
        version = max(existing_versions) + 1 if existing_versions else 1
        r_id = _record_id(d_id, version)
        now = datetime.now(timezone.utc).isoformat()

        record = AuditRecord(
            record_id=r_id,
            decision_id=d_id,
            company_id_hash=_company_hash(company_id),
            version=version,
            is_branch=bool(getattr(payload, "is_branch", False)),
            query_hash=decision.audit.query_hash,
            query_preview=query[:80],
            user_id_hash=_hash_user_id(user_id) if user_id else None,
            routing_snapshot=_build_routing_snapshot(plan),
            verdict_snapshot=_build_verdict_snapshot(decision),
            boxes=_build_box_snapshots(decision),
            performance=_build_performance_snapshot(decision, payload),
            finance_snapshot=_build_finance_snapshot(payload),
            created_utc=now,
        )

        store.records.append(record)
        store.last_updated_utc = now
        success = await _write_store(company_id, store)
        latency_ms = round((time.perf_counter() - t0) * 1000)

        if success:
            logger.info(
                "✅ Audit written | company_hash=%s | decision=%s | v%d | record=%s | latency_ms=%d",
                _company_hash(company_id)[:12], d_id[:12], version, r_id, latency_ms,
            )
            return r_id
        return None
    except Exception as e:
        logger.error("write_audit error | %s", e)
        return None
    finally:
        lock.release()


async def get_audit_record(company_id: str, record_id: str) -> AuditRecord | None:
    store = await _read_store(company_id)
    if not store:
        return None
    for r in store.records:
        if r.record_id == record_id:
            return r
    return None


async def get_decision_history(company_id: str, decision_id: str) -> list[AuditRecord]:
    store = await _read_store(company_id)
    if not store:
        return []
    versions = [r for r in store.records if r.decision_id == decision_id]
    return sorted(versions, key=lambda r: r.version)


async def list_audit_records(
    company_id: str,
    limit: int = 20,
    offset: int = 0,
    decision_type: str | None = None,
    stake_level: str | None = None,
    verdict_color: str | None = None,
) -> list[AuditListItem]:
    limit = min(limit, 100)
    store = await _read_store(company_id)
    if not store:
        return []

    records = sorted(store.records, key=lambda r: r.created_utc, reverse=True)
    if decision_type:
        records = [r for r in records if r.routing_snapshot.decision_type == decision_type]
    if stake_level:
        records = [r for r in records if r.routing_snapshot.stake_level == stake_level]
    if verdict_color:
        records = [r for r in records if r.verdict_snapshot.color == verdict_color]

    page = records[offset: offset + limit]
    return [
        AuditListItem(
            record_id=r.record_id,
            decision_id=r.decision_id,
            version=r.version,
            query_preview=r.query_preview,
            decision_type=r.routing_snapshot.decision_type,
            stake_level=r.routing_snapshot.stake_level,
            verdict_color=r.verdict_snapshot.color,
            net_score=r.verdict_snapshot.net_score,
            confidence=r.routing_snapshot.confidence,
            total_boxes=len(r.boxes),
            is_branch=r.is_branch,
            created_utc=r.created_utc,
        )
        for r in page
    ]


async def diff_versions(
    company_id: str,
    decision_id: str,
    version_a: int | None = None,
    version_b: int | None = None,
) -> VersionDiff | None:
    history = await get_decision_history(company_id, decision_id)
    if len(history) < 2:
        return None

    rec_a = history[-2] if version_a is None else next((r for r in history if r.version == version_a), history[-2])
    rec_b = history[-1] if version_b is None else next((r for r in history if r.version == version_b), history[-1])

    if rec_a.record_id == rec_b.record_id:
        return None

    return _diff_records(rec_a, rec_b)


async def export_record(company_id: str, record_id: str) -> AuditExport | None:
    record = await get_audit_record(company_id, record_id)
    if not record:
        return None
    return _build_export(record)


async def audit_stats(company_id: str) -> dict[str, Any]:
    store = await _read_store(company_id)
    if not store or not store.records:
        return {
            "total_decisions": 0,
            "total_versions": 0,
            "verdict_breakdown": {},
            "stake_breakdown": {},
            "type_breakdown": {},
            "avg_confidence": 0.0,
            "avg_net_score": 0.0,
            "avg_latency_ms": 0,
            "total_cost_usd": 0.0,
        }

    records = store.records
    unique_decisions = len({r.decision_id for r in records})
    verdict_counts: dict[str, int] = {}
    stake_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    confidences: list[float] = []
    net_scores: list[float] = []
    latencies: list[int] = []
    total_cost = 0.0

    for r in records:
        verdict_counts[r.verdict_snapshot.color] = verdict_counts.get(r.verdict_snapshot.color, 0) + 1
        stake_counts[r.routing_snapshot.stake_level] = stake_counts.get(r.routing_snapshot.stake_level, 0) + 1
        type_counts[r.routing_snapshot.decision_type] = type_counts.get(r.routing_snapshot.decision_type, 0) + 1
        confidences.append(r.routing_snapshot.confidence)
        net_scores.append(r.verdict_snapshot.net_score)
        latencies.append(r.performance.total_latency_ms)
        total_cost += r.performance.estimated_cost_usd

    return {
        "total_decisions": unique_decisions,
        "total_versions": len(records),
        "verdict_breakdown": verdict_counts,
        "stake_breakdown": stake_counts,
        "type_breakdown": type_counts,
        "avg_confidence": round(sum(confidences) / len(confidences), 3),
        "avg_net_score": round(sum(net_scores) / len(net_scores), 2),
        "avg_latency_ms": round(sum(latencies) / len(latencies)),
        "total_cost_usd": round(total_cost, 4),
    }


async def clear_audit(company_id: str) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        result = await delete_store(_TABLE, _company_hash(company_id))
        if result:
            logger.info("🗑️ Audit cleared | company_hash=%s", _company_hash(company_id)[:12])
        return result
    except Exception as e:
        logger.error("clear_audit error | %s", e)
        return False
    finally:
        lock.release()


async def delete_audit_record(company_id: str, record_id: str) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning("Audit delete lock timeout | company_hash=%s", _company_hash(company_id)[:12])
        return False

    try:
        store = await _read_store(company_id)
        if not store:
            return False

        before_count = len(store.records)
        store.records = [r for r in store.records if r.record_id != record_id]

        if len(store.records) == before_count:
            return False

        store.last_updated_utc = datetime.now(timezone.utc).isoformat()
        success = await _write_store(company_id, store)

        if success:
            logger.info(
                "🗑️ Audit record deleted | company_hash=%s | record_id=%s",
                _company_hash(company_id)[:12], record_id
            )
        return success

    except Exception as e:
        logger.error("delete_audit_record error | %s", e)
        return False
    finally:
        lock.release()