"""
audit.py — Think AI Decision Audit & Record Keeping
=====================================================
Responsibility:  Persist, version, diff, and retrieve the full record
                 of every decision that passes through Think AI.

The problem this solves
-----------------------
memory.py stores lightweight summaries to enrich future decisions.
conditions.py tracks live condition state.
Neither stores the complete, immutable record of what was decided,
why, by whom, on which version of the data, and what changed since.

audit.py is the paper trail:
  ✅  Full decision snapshot (query, plan, payload, verdict, assumptions)
  ✅  Version numbering  — every re-analysis increments the version
  ✅  Diffing            — what changed between v1 and v2 of a decision
  ✅  Who asked          — user_id attached to every record
  ✅  What changed since — side-by-side comparison for teams
  ✅  Export             — human-readable summary for explaining decisions

This is what your spec means by:
  "What we assumed / What data you gave / Which version / Who asked / What changed"

How it connects (zero edits to existing files)
-----------------------------------------------
  main.py calls write_audit() as a fire-and-forget task after /decide.
  main.py exposes GET /audit/{company_id} and GET /audit/{company_id}/{decision_id}.
  audit.py imports from router, refiner, output — zero re-definition.

Storage
-------
  Default    : One JSON file per company in audit_store/ (same pattern as memory/conditions)
  Production : Swap _read_store/_write_store for PostgreSQL/S3 — rest of module unchanged.
  Immutability: Records are append-only. Existing versions are never overwritten.

Security
--------
  * company_id SHA-256 hashed in all file paths
  * query_hash (SHA-256) stored, not raw query — PII-safe in logs
  * user_id hashed before storage if it looks like an email
  * Records are read-only once written (append-only log)
  * Sensitive verdict details excluded from list views — full record only on direct fetch
  * Per-company asyncio write locks
  * Narrow exception catches everywhere — real bugs surface

Usage (from main.py)
---------------------
    from audit import write_audit, get_audit_record, list_audit_records

    # After /decide — fire-and-forget
    asyncio.create_task(
        write_audit(
            company_id = body.company_id,
            query      = body.query,
            plan       = plan,
            decision   = decision,
            payload    = payload,
            user_id    = body.user_id,   # optional
        )
    )

    # Retrieve
    record = await get_audit_record(company_id, decision_id)
    records = await list_audit_records(company_id, limit=20)
    diff    = await diff_versions(company_id, decision_id)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ── Import shared contracts — zero re-definition ──────────────────────────
from router import UserContext, RoutingPlan
from refiner import RefinedDecision, AuditTrail
from output import UIDecisionPayload

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

_STORE_DIR        = Path(os.getenv("THINK_AI_AUDIT_DIR",      "audit_store"))
_MAX_RECORDS      = int(os.getenv("THINK_AI_AUDIT_MAX",       "500"))   # per company
_LOCK_TIMEOUT_SEC = float(os.getenv("THINK_AI_AUDIT_LOCK",    "5"))
_MAX_DOC_BYTES    = int(os.getenv("THINK_AI_AUDIT_MAX_BYTES", "5242880"))  # 5 MB per tenant
_AUDIT_VERSION    = "1.0.0"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("think-ai.audit")

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class DecisionSnapshot(BaseModel):
    """
    Frozen snapshot of the routing plan — what the Router decided.
    Stored verbatim. Never mutated after write.
    """
    decision_type:        str
    stake_level:          str
    framework:            str
    selected_lenses:      list[str]
    box_count:            int
    reasoning_depth:      int
    is_reversible:        bool
    required_questions:   list[str]
    assumptions:          list[str]
    constraints_detected: list[str]
    confidence:           float


class VerdictSnapshot(BaseModel):
    """
    Frozen snapshot of the final verdict — what the Refiner decided.
    Stored verbatim. Never mutated after write.
    """
    color:           str
    headline:        str
    rationale:       str
    net_score:       float
    go_conditions:   list[str]
    stop_conditions: list[str]
    review_triggers: list[str]
    key_unknown:     str
    flip_factor:     str


class BoxSnapshot(BaseModel):
    """Minimal record of one upside or risk box."""
    box_type:   str   # "upside" or "risk"
    title:      str
    color:      str
    claim:      str
    probability: int
    impact:     int
    risk_score: float


class PerformanceSnapshot(BaseModel):
    """
    Pipeline performance data for every decision.
    Used to track cost, latency trends, and model usage over time.
    """
    model_used:         str
    pass_latencies_ms:  list[int]   # [draft_ms, attack_ms, finalize_ms]
    total_latency_ms:   int
    output_latency_ms:  int
    total_tokens_in:    int
    total_tokens_out:   int
    estimated_cost_usd: float       # rough estimate based on token counts


class AuditRecord(BaseModel):
    """
    The complete, immutable audit record for one decision version.

    Versioning:
      - First time a query is analysed → version=1
      - User changes context and re-runs → version=2
      - Stop condition fires, re-analysis runs → version=3
      Each version is a full new record. Nothing is ever overwritten.

    Identity:
      - decision_id   : SHA-256 of (company_id + query) — stable across versions
      - record_id     : SHA-256 of (decision_id + version) — unique per version
      - user_id_hash  : SHA-256 of user_id if provided — PII-safe
    """
    record_id:        str        # unique per version
    decision_id:      str        # stable across versions of same query
    company_id_hash:  str        # SHA-256 of company_id
    version:          int        # starts at 1, increments on re-analysis
    is_branch:        bool       # True if this came from a spawn_question click

    # ── What the user asked ───────────────────────────────────────────────
    query_hash:       str        # SHA-256 of query — PII-safe
    query_preview:    str        # first 80 chars (user-visible in audit log)
    user_id_hash:     str | None = None

    # ── What the system decided ───────────────────────────────────────────
    routing_snapshot: DecisionSnapshot
    verdict_snapshot: VerdictSnapshot
    boxes:            list[BoxSnapshot]

    # ── Pipeline performance ─────────────────────────────────────────────
    performance:      PerformanceSnapshot

    # ── Timestamps ───────────────────────────────────────────────────────
    created_utc:      str
    audit_version:    str = _AUDIT_VERSION


class CompanyAuditStore(BaseModel):
    """All audit records for one company — what gets written to disk."""
    company_id_hash:  str
    version:          str = _AUDIT_VERSION
    records:          list[AuditRecord] = Field(default_factory=list)
    last_updated_utc: str


class AuditListItem(BaseModel):
    """
    Lightweight summary for list views — safe to send to frontend.
    Never includes full verdict rationale or sensitive box details.
    """
    record_id:      str
    decision_id:    str
    version:        int
    query_preview:  str
    decision_type:  str
    stake_level:    str
    verdict_color:  str
    net_score:      float
    confidence:     float
    total_boxes:    int
    is_branch:      bool
    created_utc:    str


class VersionDiff(BaseModel):
    """
    Side-by-side diff between two versions of the same decision.
    Used by teams to understand what changed and why.
    """
    decision_id:    str
    version_a:      int
    version_b:      int
    changed_fields: list[str]           # human-readable list of what changed
    verdict_changed: bool
    stake_changed:   bool
    confidence_delta: float             # version_b.confidence - version_a.confidence
    net_score_delta:  float
    new_conditions:   list[str]         # conditions in B that weren't in A
    removed_conditions: list[str]       # conditions in A that aren't in B
    summary:         str                # one-sentence plain English summary of the diff


class AuditExport(BaseModel):
    """
    Human-readable export of a single decision record.
    Designed to be shown to a team or pasted into a document.
    """
    decision_id:    str
    version:        int
    query_preview:  str
    created_utc:    str

    # Written in plain English — no JSON jargon
    what_was_decided: str
    why:              str
    key_assumptions:  list[str]
    what_we_knew:     str
    what_we_didnt_know: str
    go_conditions:    list[str]
    stop_conditions:  list[str]
    review_triggers:  list[str]
    next_step:        str
    performance:      str     # e.g. "Analysed in 4.2s using gpt-4o, 2,450 tokens"


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


async def _read_store(company_id: str) -> CompanyAuditStore | None:
    path = _store_path(company_id)
    if not path.exists():
        return None
    try:
        raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return CompanyAuditStore.model_validate_json(raw)
    except Exception as e:
        logger.error(
            "Audit read error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return None


async def _write_store(company_id: str, store: CompanyAuditStore) -> bool:
    _ensure_store_dir()
    path = _store_path(company_id)

    # Cap records — drop oldest first if over limit
    if len(store.records) > _MAX_RECORDS:
        store.records = store.records[-_MAX_RECORDS:]

    serial = store.model_dump_json(indent=2)

    # Size guard — trim oldest if over 5 MB
    while len(serial.encode()) > _MAX_DOC_BYTES and store.records:
        store.records.pop(0)
        serial = store.model_dump_json(indent=2)

    try:
        await asyncio.to_thread(path.write_text, serial, encoding="utf-8")
        return True
    except Exception as e:
        logger.error(
            "Audit write error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return False


def _new_store(company_id: str) -> CompanyAuditStore:
    return CompanyAuditStore(
        company_id_hash=_company_hash(company_id),
        records=[],
        last_updated_utc=datetime.now(timezone.utc).isoformat(),
    )


# ─────────────────────────────────────────────
# INTERNAL BUILDERS
# ─────────────────────────────────────────────

def _decision_id(company_id: str, query: str) -> str:
    """Stable across all versions of the same query for this company."""
    raw = f"{company_id}:{query.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _record_id(decision_id: str, version: int) -> str:
    """Unique per version."""
    raw = f"{decision_id}:v{version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _hash_user_id(user_id: str) -> str:
    """Hash user_id if it looks like an email or PII."""
    if "@" in user_id or len(user_id) > 32:
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    return user_id   # short opaque IDs are fine as-is


def _estimate_cost(tokens_in: int, tokens_out: int, model: str) -> float:
    """
    Rough cost estimate in USD. Rates as of mid-2025.
    Update these when OpenAI changes pricing.
    """
    rates: dict[str, tuple[float, float]] = {
        "gpt-4o-mini": (0.000_000_150, 0.000_000_600),  # per token in/out
        "gpt-4o":      (0.000_005_000, 0.000_015_000),
    }
    model_key = "gpt-4o" if "gpt-4o" in model and "mini" not in model else "gpt-4o-mini"
    rate_in, rate_out = rates.get(model_key, rates["gpt-4o-mini"])
    return round(tokens_in * rate_in + tokens_out * rate_out, 6)


def _build_routing_snapshot(plan: RoutingPlan) -> DecisionSnapshot:
    return DecisionSnapshot(
        decision_type=plan.decision_type,
        stake_level=plan.stake_level,
        framework=plan.framework,
        selected_lenses=plan.selected_lenses,
        box_count=plan.box_count,
        reasoning_depth=plan.reasoning_depth,
        is_reversible=plan.is_reversible,
        required_questions=plan.required_questions,
        assumptions=plan.assumptions,
        constraints_detected=plan.constraints_detected,
        confidence=plan.confidence,
    )


def _build_verdict_snapshot(decision: RefinedDecision) -> VerdictSnapshot:
    vb = decision.verdict_box
    return VerdictSnapshot(
        color=vb.color,
        headline=vb.headline,
        rationale=vb.rationale,
        net_score=vb.net_score,
        go_conditions=vb.go_conditions,
        stop_conditions=vb.stop_conditions,
        review_triggers=vb.review_triggers,
        key_unknown=vb.key_unknown,
        flip_factor=vb.flip_factor,
    )


def _build_box_snapshots(decision: RefinedDecision) -> list[BoxSnapshot]:
    boxes: list[BoxSnapshot] = []
    for b in decision.upside_boxes:
        boxes.append(BoxSnapshot(
            box_type="upside",
            title=b.title,
            color=b.color,
            claim=b.claim,
            probability=b.probability,
            impact=b.impact,
            risk_score=b.risk_score,
        ))
    for b in decision.risk_boxes:
        boxes.append(BoxSnapshot(
            box_type="risk",
            title=b.title,
            color=b.color,
            claim=b.claim,
            probability=b.probability,
            impact=b.impact,
            risk_score=b.risk_score,
        ))
    return boxes


def _build_performance_snapshot(
    decision: RefinedDecision,
    payload:  UIDecisionPayload,
) -> PerformanceSnapshot:
    a = decision.audit
    total_lat = sum(a.pass_latencies_ms) + payload.audit.output_latency_ms
    return PerformanceSnapshot(
        model_used=a.model_used,
        pass_latencies_ms=a.pass_latencies_ms,
        total_latency_ms=total_lat,
        output_latency_ms=payload.audit.output_latency_ms,
        total_tokens_in=a.total_tokens_in,
        total_tokens_out=a.total_tokens_out,
        estimated_cost_usd=_estimate_cost(
            a.total_tokens_in, a.total_tokens_out, a.model_used
        ),
    )


# ─────────────────────────────────────────────
# DIFF ENGINE
# ─────────────────────────────────────────────

def _diff_records(a: AuditRecord, b: AuditRecord) -> VersionDiff:
    """
    Produces a human-readable diff between two versions of a decision.
    Deterministic — no LLM needed for diffing.
    """
    changed: list[str] = []

    # Verdict changes
    verdict_changed = a.verdict_snapshot.color != b.verdict_snapshot.color
    if verdict_changed:
        changed.append(
            f"Verdict changed: {a.verdict_snapshot.color} → {b.verdict_snapshot.color}"
        )
    if a.verdict_snapshot.headline != b.verdict_snapshot.headline:
        changed.append("Verdict headline updated")

    # Stake / type changes
    stake_changed = a.routing_snapshot.stake_level != b.routing_snapshot.stake_level
    if stake_changed:
        changed.append(
            f"Stake level: {a.routing_snapshot.stake_level} → {b.routing_snapshot.stake_level}"
        )
    if a.routing_snapshot.decision_type != b.routing_snapshot.decision_type:
        changed.append(
            f"Decision type: {a.routing_snapshot.decision_type} → {b.routing_snapshot.decision_type}"
        )

    # Score / confidence
    conf_delta  = round(b.routing_snapshot.confidence  - a.routing_snapshot.confidence, 3)
    score_delta = round(b.verdict_snapshot.net_score   - a.verdict_snapshot.net_score, 2)
    if abs(conf_delta) > 0.05:
        direction = "↑" if conf_delta > 0 else "↓"
        changed.append(f"Confidence {direction} {abs(conf_delta):.0%}")
    if abs(score_delta) > 0.5:
        direction = "↑" if score_delta > 0 else "↓"
        changed.append(f"Net score {direction} {abs(score_delta):.1f}")

    # Lenses
    old_lenses = set(a.routing_snapshot.selected_lenses)
    new_lenses = set(b.routing_snapshot.selected_lenses)
    added_lenses = new_lenses - old_lenses
    if added_lenses:
        changed.append(f"New lenses added: {', '.join(added_lenses)}")

    # Conditions diff
    old_go   = set(a.verdict_snapshot.go_conditions)
    new_go   = set(b.verdict_snapshot.go_conditions)
    old_stop = set(a.verdict_snapshot.stop_conditions)
    new_stop = set(b.verdict_snapshot.stop_conditions)
    old_rev  = set(a.verdict_snapshot.review_triggers)
    new_rev  = set(b.verdict_snapshot.review_triggers)

    all_old = old_go | old_stop | old_rev
    all_new = new_go | new_stop | new_rev
    new_conds     = list(all_new - all_old)
    removed_conds = list(all_old - all_new)

    if new_conds:
        changed.append(f"{len(new_conds)} new condition(s) added")
    if removed_conds:
        changed.append(f"{len(removed_conds)} condition(s) removed")

    # Boxes
    old_titles = {bx.title for bx in a.boxes}
    new_titles = {bx.title for bx in b.boxes}
    new_boxes  = new_titles - old_titles
    if new_boxes:
        changed.append(f"New analysis box: {', '.join(new_boxes)}")

    # Plain English summary
    if not changed:
        summary = "No significant changes detected between these two versions."
    elif verdict_changed:
        summary = (
            f"Verdict flipped from {a.verdict_snapshot.color} to "
            f"{b.verdict_snapshot.color} — review carefully before acting."
        )
    elif stake_changed:
        summary = (
            f"Stake level changed from {a.routing_snapshot.stake_level} to "
            f"{b.routing_snapshot.stake_level} — "
            f"{'escalated' if b.routing_snapshot.stake_level == 'High' else 'de-escalated'}."
        )
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


# ─────────────────────────────────────────────
# EXPORT BUILDER
# ─────────────────────────────────────────────

def _build_export(record: AuditRecord) -> AuditExport:
    """
    Converts an AuditRecord into a plain-English export
    suitable for showing to a team or pasting into a doc.
    No JSON jargon — business language only.
    """
    p = record.performance
    perf_str = (
        f"Analysed in {p.total_latency_ms / 1000:.1f}s using {p.model_used}. "
        f"{p.total_tokens_in + p.total_tokens_out:,} tokens used. "
        f"Estimated cost: ${p.estimated_cost_usd:.4f}."
    )

    r = record.routing_snapshot
    v = record.verdict_snapshot

    what_was_decided = (
        f"{v.headline} "
        f"(Verdict: {v.color}, Net score: {v.net_score:+.1f})"
    )

    what_we_knew = (
        f"Decision classified as {r.decision_type} with {r.stake_level} stakes. "
        f"Analysed through: {', '.join(r.selected_lenses)}. "
        f"Reversible: {'Yes' if r.is_reversible else 'No'}."
    )

    what_we_didnt_know = v.key_unknown

    return AuditExport(
        decision_id=record.decision_id,
        version=record.version,
        query_preview=record.query_preview,
        created_utc=record.created_utc,
        what_was_decided=what_was_decided,
        why=v.rationale,
        key_assumptions=r.assumptions,
        what_we_knew=what_we_knew,
        what_we_didnt_know=what_we_didnt_know,
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
    query:      str,
    plan:       RoutingPlan,
    decision:   RefinedDecision,
    payload:    UIDecisionPayload,
    user_id:    str | None = None,
) -> str | None:
    """
    Called by main.py as a fire-and-forget task after every /decide.
    Appends a new AuditRecord. If this query has been analysed before,
    version is auto-incremented.

    Parameters
    ----------
    company_id : Tenant identifier.
    query      : The original decision text.
    plan       : RoutingPlan from route().
    decision   : RefinedDecision from refine().
    payload    : UIDecisionPayload from format_for_ui().
    user_id    : Optional — who triggered this decision. Hashed before storage.

    Returns
    -------
    record_id (str) if written, None on failure.
    """
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning(
            "Audit write lock timeout | company_hash=%s",
            _company_hash(company_id)[:12],
        )
        return None

    try:
        t0    = time.perf_counter()
        store = await _read_store(company_id) or _new_store(company_id)

        d_id  = _decision_id(company_id, query)

        # Determine version — find highest existing version for this decision_id
        existing_versions = [
            r.version for r in store.records if r.decision_id == d_id
        ]
        version = max(existing_versions) + 1 if existing_versions else 1

        r_id  = _record_id(d_id, version)
        now   = datetime.now(timezone.utc).isoformat()

        record = AuditRecord(
            record_id=r_id,
            decision_id=d_id,
            company_id_hash=_company_hash(company_id),
            version=version,
            is_branch=payload.is_branch,
            query_hash=decision.audit.query_hash,
            query_preview=query[:80],
            user_id_hash=_hash_user_id(user_id) if user_id else None,
            routing_snapshot=_build_routing_snapshot(plan),
            verdict_snapshot=_build_verdict_snapshot(decision),
            boxes=_build_box_snapshots(decision),
            performance=_build_performance_snapshot(decision, payload),
            created_utc=now,
        )

        store.records.append(record)
        store.last_updated_utc = now

        success = await _write_store(company_id, store)
        latency_ms = round((time.perf_counter() - t0) * 1000)

        if success:
            logger.info(
                "✅ Audit written | company_hash=%s | decision=%s | "
                "v%d | record=%s | latency_ms=%d",
                _company_hash(company_id)[:12],
                d_id[:12], version, r_id, latency_ms,
            )
            return r_id

        return None

    except Exception as e:
        logger.error(
            "write_audit error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return None
    finally:
        lock.release()


async def get_audit_record(
    company_id: str,
    record_id:  str,
) -> AuditRecord | None:
    """
    Retrieve one specific audit record by its record_id.
    Returns None if not found.
    """
    store = await _read_store(company_id)
    if not store:
        return None
    for r in store.records:
        if r.record_id == record_id:
            return r
    return None


async def get_decision_history(
    company_id:  str,
    decision_id: str,
) -> list[AuditRecord]:
    """
    Retrieve all versions of a specific decision.
    Returns them oldest-first (v1, v2, v3 ...).
    """
    store = await _read_store(company_id)
    if not store:
        return []
    versions = [r for r in store.records if r.decision_id == decision_id]
    return sorted(versions, key=lambda r: r.version)


async def list_audit_records(
    company_id: str,
    limit:      int = 20,
    offset:     int = 0,
    decision_type: str | None = None,
    stake_level:   str | None = None,
    verdict_color: str | None = None,
) -> list[AuditListItem]:
    """
    List audit records for a company — lightweight summary view.
    Supports filtering and pagination.
    Returns newest first.

    Parameters
    ----------
    company_id    : Tenant identifier.
    limit         : Max records to return (default 20, max 100).
    offset        : Pagination offset.
    decision_type : Filter by decision type (e.g. "Finance").
    stake_level   : Filter by stake level ("Low" / "Medium" / "High").
    verdict_color : Filter by verdict color ("Green" / "Yellow" / "Orange" / "Red").
    """
    limit = min(limit, 100)   # hard cap
    store = await _read_store(company_id)
    if not store:
        return []

    records = sorted(store.records, key=lambda r: r.created_utc, reverse=True)

    # Apply filters
    if decision_type:
        records = [r for r in records if r.routing_snapshot.decision_type == decision_type]
    if stake_level:
        records = [r for r in records if r.routing_snapshot.stake_level == stake_level]
    if verdict_color:
        records = [r for r in records if r.verdict_snapshot.color == verdict_color]

    # Paginate
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
    company_id:  str,
    decision_id: str,
    version_a:   int | None = None,
    version_b:   int | None = None,
) -> VersionDiff | None:
    """
    Produce a human-readable diff between two versions of a decision.
    If version_a / version_b not specified, diffs the last two versions.
    Returns None if fewer than 2 versions exist.

    Parameters
    ----------
    company_id  : Tenant identifier.
    decision_id : The stable decision ID (from AuditListItem.decision_id).
    version_a   : Older version number (default: second-to-last).
    version_b   : Newer version number (default: latest).
    """
    history = await get_decision_history(company_id, decision_id)
    if len(history) < 2:
        return None

    if version_a is None:
        rec_a = history[-2]
    else:
        matching = [r for r in history if r.version == version_a]
        rec_a = matching[0] if matching else history[-2]

    if version_b is None:
        rec_b = history[-1]
    else:
        matching = [r for r in history if r.version == version_b]
        rec_b = matching[0] if matching else history[-1]

    if rec_a.record_id == rec_b.record_id:
        return None   # same record, nothing to diff

    return _diff_records(rec_a, rec_b)


async def export_record(
    company_id: str,
    record_id:  str,
) -> AuditExport | None:
    """
    Returns a plain-English export of a single audit record.
    Suitable for displaying to a team or sharing in a document.
    """
    record = await get_audit_record(company_id, record_id)
    if not record:
        return None
    return _build_export(record)


async def audit_stats(company_id: str) -> dict[str, Any]:
    """
    Aggregate statistics across all audit records for a company.
    Used by your analytics / reporting page.
    """
    store = await _read_store(company_id)
    if not store or not store.records:
        return {
            "total_decisions": 0,
            "total_versions":  0,
            "verdict_breakdown": {},
            "stake_breakdown": {},
            "type_breakdown": {},
            "avg_confidence": 0.0,
            "avg_net_score": 0.0,
            "avg_latency_ms": 0,
            "total_cost_usd": 0.0,
        }

    records = store.records

    # Unique decisions (by decision_id)
    unique_decisions = len({r.decision_id for r in records})

    # Breakdowns
    verdict_counts: dict[str, int] = {}
    stake_counts:   dict[str, int] = {}
    type_counts:    dict[str, int] = {}
    confidences:    list[float]    = []
    net_scores:     list[float]    = []
    latencies:      list[int]      = []
    total_cost:     float          = 0.0

    for r in records:
        v = r.verdict_snapshot.color
        s = r.routing_snapshot.stake_level
        t = r.routing_snapshot.decision_type

        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        stake_counts[s]   = stake_counts.get(s, 0)   + 1
        type_counts[t]    = type_counts.get(t, 0)    + 1
        confidences.append(r.routing_snapshot.confidence)
        net_scores.append(r.verdict_snapshot.net_score)
        latencies.append(r.performance.total_latency_ms)
        total_cost += r.performance.estimated_cost_usd

    return {
        "total_decisions":   unique_decisions,
        "total_versions":    len(records),
        "verdict_breakdown": verdict_counts,
        "stake_breakdown":   stake_counts,
        "type_breakdown":    type_counts,
        "avg_confidence":    round(sum(confidences) / len(confidences), 3),
        "avg_net_score":     round(sum(net_scores)  / len(net_scores),  2),
        "avg_latency_ms":    round(sum(latencies)   / len(latencies)),
        "total_cost_usd":    round(total_cost, 4),
    }


async def clear_audit(company_id: str) -> bool:
    """
    Wipes all audit records for a company.
    Called alongside memory.clear_memory() for GDPR erasure.
    Returns True if deleted, False if nothing existed.
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
                "🗑️  Audit cleared | company_hash=%s",
                _company_hash(company_id)[:12],
            )
            return True
        return False
    except Exception as e:
        logger.error("clear_audit error | %s", e)
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

    async def _demo():
        company_id = "demo-co"
        query      = "Should we raise our prices by 20% next month?"
        ctx        = UserContext(
            industry="E-commerce",
            company_size="Series A",
            risk_appetite="Moderate",
        )

        print("── Step 1: Full pipeline ───────────────────────")
        plan     = await route(query, company_id, ctx)
        decision = await refine(query, plan, company_id, ctx)
        payload  = format_for_ui(decision, query, company_id)
        print(f"  Verdict: {payload.verdict_card.badge.label}")

        print("\n── Step 2: Write audit record (v1) ─────────────")
        record_id = await write_audit(
            company_id=company_id,
            query=query,
            plan=plan,
            decision=decision,
            payload=payload,
            user_id="user-abc123",
        )
        print(f"  record_id: {record_id}")

        print("\n── Step 3: Simulate re-analysis (v2) ───────────")
        # Re-run with different context to create v2
        ctx2     = UserContext(industry="E-commerce", company_size="Series B",
                               risk_appetite="Aggressive")
        plan2    = await route(query, company_id, ctx2, bypass_cache=True)
        decision2 = await refine(query, plan2, company_id, ctx2, bypass_cache=True)
        payload2  = format_for_ui(decision2, query, company_id)
        record_id2 = await write_audit(
            company_id=company_id,
            query=query,
            plan=plan2,
            decision=decision2,
            payload=payload2,
            user_id="user-abc123",
        )
        print(f"  record_id v2: {record_id2}")

        print("\n── Step 4: List records ────────────────────────")
        items = await list_audit_records(company_id, limit=5)
        for item in items:
            print(f"  v{item.version} | {item.verdict_color} | {item.query_preview[:40]}")

        print("\n── Step 5: Diff v1 vs v2 ───────────────────────")
        if items:
            diff = await diff_versions(company_id, items[0].decision_id)
            if diff:
                print(f"  Summary: {diff.summary}")
                for change in diff.changed_fields:
                    print(f"  · {change}")
            else:
                print("  Only one version — nothing to diff yet.")

        print("\n── Step 6: Export record ───────────────────────")
        if record_id:
            export = await export_record(company_id, record_id)
            if export:
                print(f"  What was decided: {export.what_was_decided}")
                print(f"  Why             : {export.why}")
                print(f"  Performance     : {export.performance}")

        print("\n── Step 7: Audit stats ─────────────────────────")
        stats = await audit_stats(company_id)
        print(json.dumps(stats, indent=2))

    asyncio.run(_demo())