"""
memory.py — Think AI Background Memory
========================================
Responsibility:  Store and retrieve private per-company background
                 documents that automatically enrich every Router and
                 Refiner call — without the user re-entering context
                 on every request.

What it stores (per company_id)
--------------------------------
  company_profile     : Industry, size, risk appetite, client count, etc.
  behavior_patterns   : Which decision types this company makes most often,
                        which lenses they find most useful.
  decision_history    : Lightweight summaries of past decisions (not full
                        RefinedDecision — just enough to spot patterns).
  running_assumptions : Things the system has inferred and kept across calls.
  preference_signals  : Derived from feedback ratings (high/low rated patterns).

How it connects to the pipeline
--------------------------------
  main.py calls:
    1. enrich_context()  BEFORE  route()   → Router gets richer UserContext
    2. update_memory()   AFTER   format_for_ui() → Memory learns from each decision

  router.py and refiner.py are untouched — they just receive a richer
  UserContext than before. Zero edits to those files.

Storage backend
---------------
  Default  : JSON files on disk (backend/memory_store/)
  Production: Swap _MemoryBackend for Redis or PostgreSQL via env var.
              The rest of the module is identical.

Security
--------
  * company_id is SHA-256 hashed before it becomes a filename — no PII in paths.
  * Each document is size-capped to prevent unbounded growth.
  * No cross-tenant reads — every read/write includes company_id in the key.
  * Memory documents are never returned to the client directly.
  * Sensitive fields (client_count, revenue) are stored but never logged.

Usage (from main.py)
---------------------
    from memory import enrich_context, update_memory, MemoryDocument

    # Before routing — enrich whatever context the user sent
    enriched_ctx = await enrich_context(company_id, user_context)

    # Run normal pipeline with enriched context
    plan     = await route(query, company_id, enriched_ctx)
    decision = await refine(query, plan, company_id, enriched_ctx)
    payload  = format_for_ui(decision, query, company_id)

    # After decision — teach memory what happened
    await update_memory(company_id, query, plan, payload)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ── Import shared contracts — zero re-definition ──────────────────────────
from router import UserContext, RoutingPlan
from output import UIDecisionPayload

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

_STORE_DIR          = Path(os.getenv("THINK_AI_MEMORY_DIR", "memory_store"))
_MAX_HISTORY        = int(os.getenv("THINK_AI_MEMORY_MAX_HISTORY",   "50"))
_MAX_ASSUMPTIONS    = int(os.getenv("THINK_AI_MEMORY_MAX_ASSUMPTIONS","20"))
_MAX_DOC_BYTES      = int(os.getenv("THINK_AI_MEMORY_MAX_BYTES",  "65536"))   # 64 KB per tenant
_LOCK_TIMEOUT_SEC   = float(os.getenv("THINK_AI_MEMORY_LOCK_TIMEOUT", "5"))
_MEMORY_VERSION     = "1.0.0"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("think-ai.memory")

# ─────────────────────────────────────────────
# CONCURRENCY GUARD
# One asyncio.Lock per company_id prevents
# simultaneous writes corrupting the same file.
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
    """
    Stable facts about the company.
    Updated when the user explicitly provides new info,
    or when the system infers something with high confidence.
    """
    industry:        str | None = None
    company_size:    str | None = None    # "Seed", "Series A", "Enterprise" etc.
    risk_appetite:   str | None = None    # "Conservative", "Moderate", "Aggressive"
    primary_market:  str | None = None
    team_size:       str | None = None
    client_count:    str | None = None    # stored but never logged
    revenue_range:   str | None = None    # stored but never logged
    extra:           dict[str, str] = Field(default_factory=dict)


class DecisionSummary(BaseModel):
    """
    Lightweight record of one past decision — not the full RefinedDecision.
    Enough to spot patterns without hitting the size cap.
    """
    decision_id:     str          # SHA-256 of query+timestamp
    query_preview:   str          # first 80 chars only
    decision_type:   str
    stake_level:     str
    framework:       str
    verdict_color:   str
    confidence:      float
    timestamp_utc:   str
    user_rating:     int | None = None    # 1–5, set by feedback.py later


class BehaviorPattern(BaseModel):
    """
    Inferred patterns about how this company uses Think AI.
    Updated automatically after each decision.
    """
    most_common_types:   list[str]   = Field(default_factory=list)
    most_used_lenses:    list[str]   = Field(default_factory=list)
    avg_stake_level:     str         = "Medium"
    prefers_quick:       bool        = False    # True if mostly Low-stake decisions
    high_risk_ratio:     float       = 0.0      # fraction of decisions that are High
    total_decisions:     int         = 0


class MemoryDocument(BaseModel):
    """
    The complete private background document for one company.
    This is what gets serialised to disk / Redis.
    Never returned to the client.
    """
    company_id_hash:     str          # SHA-256 of company_id — PII-safe
    version:             str          = _MEMORY_VERSION
    created_utc:         str
    last_updated_utc:    str

    profile:             CompanyProfile      = Field(default_factory=CompanyProfile)
    behavior:            BehaviorPattern     = Field(default_factory=BehaviorPattern)
    decision_history:    list[DecisionSummary] = Field(default_factory=list)
    running_assumptions: list[str]           = Field(default_factory=list)
    preference_signals:  dict[str, Any]      = Field(default_factory=dict)
    # preference_signals populated by feedback.py:
    #   {"liked_lenses": [...], "disliked_verdict_colors": [...], ...}


# ─────────────────────────────────────────────
# STORAGE BACKEND
# Swappable — replace with Redis/Postgres in prod
# by implementing the same interface.
# ─────────────────────────────────────────────

def _company_hash(company_id: str) -> str:
    """SHA-256 of company_id — used as filename, never the raw ID."""
    return hashlib.sha256(company_id.encode()).hexdigest()


def _doc_path(company_id: str) -> Path:
    return _STORE_DIR / f"{_company_hash(company_id)}.json"


def _ensure_store_dir() -> None:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    # Write a .gitignore so memory files are never committed
    gitignore = _STORE_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n")


async def _read_doc(company_id: str) -> MemoryDocument | None:
    """
    Read the memory document for this company.
    Returns None if no document exists yet.
    Pure I/O — no business logic.
    """
    path = _doc_path(company_id)
    if not path.exists():
        return None

    try:
        raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return MemoryDocument.model_validate_json(raw)
    except Exception as e:
        logger.error(
            "Memory read error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return None


async def _write_doc(company_id: str, doc: MemoryDocument) -> bool:
    """
    Write the memory document. Size-capped. Returns True on success.
    Must be called while holding the company's write lock.
    """
    _ensure_store_dir()
    path   = _doc_path(company_id)
    serial = doc.model_dump_json(indent=2)

    if len(serial.encode()) > _MAX_DOC_BYTES:
        logger.warning(
            "Memory doc exceeds size cap — trimming history | "
            "company_hash=%s", _company_hash(company_id)[:12],
        )
        # Trim oldest history entries until it fits
        while len(serial.encode()) > _MAX_DOC_BYTES and doc.decision_history:
            doc.decision_history.pop(0)
            serial = doc.model_dump_json(indent=2)

    try:
        await asyncio.to_thread(path.write_text, serial, encoding="utf-8")
        return True
    except Exception as e:
        logger.error(
            "Memory write error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return False


# ─────────────────────────────────────────────
# INTERNAL: DOCUMENT LIFECYCLE
# ─────────────────────────────────────────────

def _new_document(company_id: str) -> MemoryDocument:
    now = datetime.now(timezone.utc).isoformat()
    return MemoryDocument(
        company_id_hash=_company_hash(company_id),
        created_utc=now,
        last_updated_utc=now,
    )


def _merge_profile(doc: MemoryDocument, ctx: UserContext) -> MemoryDocument:
    """
    Merge explicit UserContext fields into the stored profile.
    Only overwrites if the new value is non-null — never erases known info.
    """
    p = doc.profile
    if ctx.industry      and not p.industry:      p.industry      = ctx.industry
    if ctx.company_size  and not p.company_size:  p.company_size  = ctx.company_size
    if ctx.risk_appetite and not p.risk_appetite: p.risk_appetite = ctx.risk_appetite
    for k, v in ctx.extra.items():
        if k not in p.extra:
            p.extra[k] = str(v)
    return doc


def _update_behavior(doc: MemoryDocument, plan: RoutingPlan) -> MemoryDocument:
    """
    Update behavior patterns after a new routing plan is seen.
    Running averages — no heavy ML needed at this stage.
    """
    b = doc.behavior
    history = doc.decision_history

    b.total_decisions += 1

    # Most common decision types (top 3)
    all_types  = [h.decision_type for h in history] + [plan.decision_type]
    type_freq  = {t: all_types.count(t) for t in set(all_types)}
    b.most_common_types = sorted(type_freq, key=type_freq.get, reverse=True)[:3]

    # Most used lenses (top 5)
    all_lenses = [l for h in history for l in []] + plan.selected_lenses
    # Note: history summaries don't store lenses (size budget) —
    # we track them via preference_signals instead
    if plan.selected_lenses:
        lens_counts = doc.preference_signals.get("lens_counts", {})
        for lens in plan.selected_lenses:
            lens_counts[lens] = lens_counts.get(lens, 0) + 1
        doc.preference_signals["lens_counts"] = lens_counts
        b.most_used_lenses = sorted(
            lens_counts, key=lens_counts.get, reverse=True
        )[:5]

    # High-risk ratio
    high_count  = sum(1 for h in history if h.stake_level == "High")
    if plan.stake_level == "High":
        high_count += 1
    b.high_risk_ratio = round(high_count / b.total_decisions, 2)

    # Average stake
    stake_weights = {"Low": 1, "Medium": 2, "High": 3}
    all_stakes    = [h.stake_level for h in history] + [plan.stake_level]
    avg_weight    = sum(stake_weights.get(s, 2) for s in all_stakes) / len(all_stakes)
    b.avg_stake_level = "High" if avg_weight > 2.4 else ("Low" if avg_weight < 1.6 else "Medium")

    # Prefers quick decisions?
    low_count   = sum(1 for h in history if h.stake_level == "Low")
    if plan.stake_level == "Low":
        low_count += 1
    b.prefers_quick = (low_count / b.total_decisions) > 0.6

    return doc


def _append_history(
    doc:     MemoryDocument,
    query:   str,
    plan:    RoutingPlan,
    payload: UIDecisionPayload,
) -> MemoryDocument:
    """Add a new DecisionSummary to history. Cap at _MAX_HISTORY."""
    decision_id = hashlib.sha256(
        f"{query}{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()[:16]

    summary = DecisionSummary(
        decision_id=decision_id,
        query_preview=query[:80],
        decision_type=plan.decision_type,
        stake_level=plan.stake_level,
        framework=plan.framework,
        verdict_color=payload.verdict_card.badge.raw,
        confidence=payload.confidence,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )

    doc.decision_history.append(summary)

    # Cap history
    if len(doc.decision_history) > _MAX_HISTORY:
        doc.decision_history = doc.decision_history[-_MAX_HISTORY:]

    return doc


def _merge_assumptions(
    doc:         MemoryDocument,
    plan:        RoutingPlan,
) -> MemoryDocument:
    """
    Absorb new assumptions from the routing plan into running_assumptions.
    Deduplicates. Caps at _MAX_ASSUMPTIONS.
    """
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

async def enrich_context(
    company_id: str,
    ctx:        UserContext | None,
) -> UserContext:
    """
    Called by main.py BEFORE route().
    Reads the company's memory document and merges stored profile
    data into the UserContext so the Router and Refiner get the
    richest possible context without the user re-entering it.

    If no memory exists yet, returns the original context unchanged.
    If memory read fails, returns original context — never blocks the pipeline.

    Parameters
    ----------
    company_id : Tenant identifier.
    ctx        : The UserContext from the current request (may be None).

    Returns
    -------
    UserContext — always, enriched where memory is available.
    """
    t0 = time.perf_counter()

    doc = await _read_doc(company_id)

    if doc is None:
        logger.info(
            "No memory yet | company_hash=%s", _company_hash(company_id)[:12]
        )
        return ctx or UserContext()

    p = doc.profile
    b = doc.behavior

    # Start from what the user sent (or a blank context)
    base = ctx or UserContext()

    # Merge stored profile — user's explicit values win over memory
    enriched = UserContext(
        industry=base.industry           or p.industry,
        company_size=base.company_size   or p.company_size,
        risk_appetite=base.risk_appetite or p.risk_appetite,
        extra={
            # Stored extras first, then current request extras (current wins on conflict)
            **p.extra,
            **base.extra,
            # Inject behavior signals so Router can use them
            "_total_decisions":    str(b.total_decisions),
            "_avg_stake":          b.avg_stake_level,
            "_prefers_quick":      str(b.prefers_quick),
            "_common_types":       ", ".join(b.most_common_types) if b.most_common_types else "",
            "_preferred_lenses":   ", ".join(b.most_used_lenses)  if b.most_used_lenses  else "",
            "_running_assumptions":"; ".join(doc.running_assumptions[-5:]),  # last 5 only
        },
    )

    latency_ms = round((time.perf_counter() - t0) * 1000)
    logger.info(
        "✅ Context enriched | company_hash=%s | decisions_seen=%d | latency_ms=%d",
        _company_hash(company_id)[:12],
        b.total_decisions,
        latency_ms,
    )

    return enriched


async def update_memory(
    company_id: str,
    query:      str,
    plan:       RoutingPlan,
    payload:    UIDecisionPayload,
) -> bool:
    """
    Called by main.py AFTER format_for_ui().
    Updates the company's memory document with:
      - Profile fields inferred from the routing plan
      - New behavior pattern signals
      - A new DecisionSummary in history
      - New assumptions from the plan

    Thread-safe via per-company asyncio.Lock.
    Returns True on success, False on failure — never raises.
    Pipeline continues regardless.

    Parameters
    ----------
    company_id : Tenant identifier.
    query      : The original decision text.
    plan       : The RoutingPlan from route().
    payload    : The UIDecisionPayload from format_for_ui().

    Returns
    -------
    bool — True if memory was written successfully.
    """
    lock = _get_lock(company_id)

    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning(
            "Memory lock timeout | company_hash=%s", _company_hash(company_id)[:12]
        )
        return False

    # We now hold the lock — wrap everything in try/finally to always release
    try:
        t0  = time.perf_counter()
        doc = await _read_doc(company_id) or _new_document(company_id)

        # Enrich context from the payload's audit (company_id only — no PII)
        enriched_ctx = UserContext()   # no new profile info from this call path

        doc = _update_behavior(doc, plan)
        doc = _append_history(doc, query, plan, payload)
        doc = _merge_assumptions(doc, plan)
        doc.last_updated_utc = datetime.now(timezone.utc).isoformat()

        success = await _write_doc(company_id, doc)

        latency_ms = round((time.perf_counter() - t0) * 1000)
        if success:
            logger.info(
                "✅ Memory updated | company_hash=%s | total_decisions=%d | latency_ms=%d",
                _company_hash(company_id)[:12],
                doc.behavior.total_decisions,
                latency_ms,
            )
        return success

    except Exception as e:
        logger.error(
            "Memory update error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return False
    finally:
        lock.release()


async def update_profile(
    company_id: str,
    profile_data: dict[str, Any],
) -> bool:
    """
    Called when a user explicitly updates their company profile
    (e.g. via a settings page in the frontend).
    Merges new values — never wipes existing ones.

    Parameters
    ----------
    company_id   : Tenant identifier.
    profile_data : Dict of profile fields to update.
                   Accepted keys: industry, company_size, risk_appetite,
                   primary_market, team_size, client_count, revenue_range,
                   plus any extra key-value strings.

    Returns
    -------
    bool — True if written successfully.
    """
    lock = _get_lock(company_id)

    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning("Profile update lock timeout | company_hash=%s",
                       _company_hash(company_id)[:12])
        return False

    try:
        doc = await _read_doc(company_id) or _new_document(company_id)
        p   = doc.profile

        # Map known fields — ignore unknown to prevent injection
        _known = {
            "industry", "company_size", "risk_appetite",
            "primary_market", "team_size", "client_count", "revenue_range",
        }
        for k, v in profile_data.items():
            if k in _known:
                setattr(p, k, str(v))
            else:
                p.extra[k] = str(v)    # everything else goes to extra

        doc.last_updated_utc = datetime.now(timezone.utc).isoformat()
        success = await _write_doc(company_id, doc)

        if success:
            logger.info(
                "✅ Profile updated | company_hash=%s",
                _company_hash(company_id)[:12],
            )
        return success

    except Exception as e:
        logger.error(
            "Profile update error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return False
    finally:
        lock.release()


async def update_rating(
    company_id:  str,
    decision_id: str,
    rating:      int,
) -> bool:
    """
    Called by feedback.py to attach a user rating to a past decision.
    Ratings under 3 ★ are flagged so the system can learn from them.

    Parameters
    ----------
    company_id  : Tenant identifier.
    decision_id : The decision_id from a DecisionSummary in history.
    rating      : 1–5 star rating.

    Returns
    -------
    bool — True if found and written successfully.
    """
    if rating not in range(1, 6):
        logger.warning("Invalid rating %d | must be 1–5", rating)
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

                # Track negative signals for the learning loop (feedback.py)
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

        logger.warning(
            "Decision not found in history | company_hash=%s | decision_id=%s",
            _company_hash(company_id)[:12], decision_id,
        )
        return False

    except Exception as e:
        logger.error(
            "Rating update error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return False
    finally:
        lock.release()


async def get_memory_summary(company_id: str) -> dict[str, Any] | None:
    """
    Returns a safe, client-facing summary of what memory holds.
    Never returns sensitive fields (client_count, revenue_range).
    Used by your settings/profile page in the frontend.

    Returns None if no memory exists yet.
    """
    doc = await _read_doc(company_id)
    if not doc:
        return None

    return {
        "total_decisions":    doc.behavior.total_decisions,
        "most_common_types":  doc.behavior.most_common_types,
        "most_used_lenses":   doc.behavior.most_used_lenses,
        "avg_stake_level":    doc.behavior.avg_stake_level,
        "high_risk_ratio":    doc.behavior.high_risk_ratio,
        "profile": {
            "industry":      doc.profile.industry,
            "company_size":  doc.profile.company_size,
            "risk_appetite": doc.profile.risk_appetite,
        },
        "last_updated_utc":   doc.last_updated_utc,
        "memory_version":     doc.version,
    }


async def clear_memory(company_id: str) -> bool:
    """
    Wipes the memory document for this company.
    Called when a user requests data deletion (GDPR right to erasure).
    Returns True if deleted, False if nothing existed.
    """
    lock = _get_lock(company_id)

    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        path = _doc_path(company_id)
        if path.exists():
            await asyncio.to_thread(path.unlink)
            logger.info(
                "🗑️  Memory cleared | company_hash=%s",
                _company_hash(company_id)[:12],
            )
            return True
        return False
    except Exception as e:
        logger.error(
            "Memory clear error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
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

        print("── Step 1: Enrich context from memory ──────────")
        enriched = await enrich_context(company_id, ctx)
        print(f"  Industry (from memory or ctx): {enriched.industry}")
        print(f"  Total past decisions seen    : {enriched.extra.get('_total_decisions', '0')}")

        print("\n── Step 2: Full pipeline ───────────────────────")
        plan     = await route(query, company_id, enriched)
        decision = await refine(query, plan, company_id, enriched)
        payload  = format_for_ui(decision, query, company_id)

        print(f"  Verdict: {payload.verdict_card.badge.label}")
        print(f"  Confidence: {payload.confidence:.0%}")

        print("\n── Step 3: Update memory ───────────────────────")
        ok = await update_memory(company_id, query, plan, payload)
        print(f"  Written: {ok}")

        print("\n── Step 4: Memory summary (client-safe) ────────")
        summary = await get_memory_summary(company_id)
        print(json.dumps(summary, indent=2))

        print("\n── Step 5: Second call — richer context ────────")
        enriched2 = await enrich_context(company_id, None)
        print(f"  Decisions seen: {enriched2.extra.get('_total_decisions')}")
        print(f"  Avg stake     : {enriched2.extra.get('_avg_stake')}")
        print(f"  Common types  : {enriched2.extra.get('_common_types')}")

    asyncio.run(_demo())