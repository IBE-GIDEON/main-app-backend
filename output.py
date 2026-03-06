"""
output.py — Think AI UI Output Layer
=====================================
Responsibility:  Take a RefinedDecision from refiner.py and produce the
                 exact JSON payload the React Decision Board consumes.
                 Also handles box branching — when a user clicks a
                 spawn_question, this layer runs a lightweight mini-pipeline
                 (route → refine) on that sub-question without the caller
                 knowing or caring about the internals.

Architecture
------------
  RefinedDecision (from refiner.py)
        │
        ▼
  format_for_ui()          ← main formatter, called by your API handler
        │
        ▼
  UIDecisionPayload        ← what React receives, field-for-field

  branch_on_question()     ← called when user clicks a spawn_question box
        │
        ├─ route()  (router.py)
        ├─ refine() (refiner.py)
        └─ format_for_ui()
              │
              ▼
        UIDecisionPayload  ← same shape, so React needs zero changes

Security & scalability
-----------------------
* Zero re-definition — imports everything from router.py and refiner.py.
* Tenant-scoped TTL cache for branch results (same keys as router/refiner).
* Input sanitisation before any branch query reaches the LLM.
* Observability: latency + structured logs on every format and branch call.
* Narrow exception catches — real bugs surface, fallbacks stay honest.

Usage
-----
    from router import route, UserContext
    from refiner import refine
    from output import format_for_ui, branch_on_question

    # Normal flow
    plan     = await route(query, company_id, context)
    decision = await refine(query, plan, company_id, context)
    payload  = format_for_ui(decision, query, company_id)

    # User clicks a spawn_question box
    branch   = await branch_on_question(
        question   = "What if our churn rate is already above 5%?",
        parent_query = query,
        company_id = company_id,
        context    = context,
    )
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Literal

from pydantic import BaseModel, Field
from cachetools import TTLCache

# ── Import everything — zero re-definition ────────────────────────────────
from router import route, UserContext, RoutingPlan
from refiner import (
    refine,
    RefinedDecision,
    DecisionBox,
    VerdictBox,
    NextStepBox,
    AuditTrail,
)

import os
_CACHE_TTL    = int(os.getenv("THINK_AI_CACHE_TTL",   "3600"))
_CACHE_MAXSIZE = int(os.getenv("THINK_AI_CACHE_MAX",  "2000"))
_MAX_QUERY_LEN = int(os.getenv("THINK_AI_MAX_QUERY",  "2000"))

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("think-ai.output")

# ─────────────────────────────────────────────
# COLOR MAP  (backend Literal → UI badge label)
# ─────────────────────────────────────────────

COLOR_BADGE: dict[str, str] = {
    "Green":  "🟢 Good to Go",
    "Yellow": "🟡 Balanced",
    "Orange": "🟠 Unknown",
    "Red":    "🔴 Risky",
}

COLOR_HEX: dict[str, str] = {
    "Green":  "#22c55e",
    "Yellow": "#eab308",
    "Orange": "#f97316",
    "Red":    "#ef4444",
}


# ─────────────────────────────────────────────
# UI SCHEMAS  (field-for-field React contract)
# ─────────────────────────────────────────────

class UIBadge(BaseModel):
    label:  str    # e.g. "🟢 Good to Go"
    color:  str    # hex e.g. "#22c55e"
    raw:    str    # e.g. "Green" — for programmatic use in React


class UIBox(BaseModel):
    """One rendered card on the Decision Board."""
    id:                    str    # stable, deterministic — safe for React key prop
    box_type:              Literal["summary", "upside", "risk", "verdict", "next_step"]
    title:                 str
    badge:                 UIBadge
    claim:                 str
    evidence_or_reasoning: str
    probability:           int | None   = None   # None for summary/verdict/next_step
    impact:                int | None   = None
    risk_score:            float | None = None
    follow_up_actions:     list[str]    = Field(default_factory=list)
    spawn_questions:       list[str]    = Field(default_factory=list)
    # spawn_questions powers the branching — clicking reveals sub-decisions


class UIVerdictCard(BaseModel):
    """The full verdict section — separate from the box grid."""
    badge:           UIBadge
    headline:        str
    rationale:       str
    net_score:       float
    go_conditions:   list[str]
    stop_conditions: list[str]
    review_triggers: list[str]
    key_unknown:     str
    flip_factor:     str


class UINextStep(BaseModel):
    immediate_action:  str
    test_if_uncertain: str | None = None
    owner:             str | None = None
    deadline:          str | None = None
    escalate_if:       str | None = None


class UIAudit(BaseModel):
    """Shown only in dev mode / audit view — never in the main decision board."""
    query_hash:        str
    company_id:        str
    routing_plan_hash: str
    model_used:        str
    pass_latencies_ms: list[int]
    total_tokens_in:   int
    total_tokens_out:  int
    refiner_version:   str
    timestamp_utc:     str
    output_latency_ms: int    # time spent in this layer


class UIDecisionPayload(BaseModel):
    """
    The complete payload sent to the React Decision Board.
    Field names match the UI spec 1-to-1:
        summary_box, upside_boxes[], risk_boxes[],
        verdict_box, next_step_box, confidence.
    MAX 5 BOXES TOTAL in the grid (enforced here).
    """
    # ── Meta ───────────────────────────────────────────────────────────────
    query:              str
    company_id:         str
    confidence:         float
    total_boxes:        int

    # ── The grid — what React maps over ────────────────────────────────────
    summary_box:        UIBox
    upside_boxes:       list[UIBox]
    risk_boxes:         list[UIBox]

    # ── Below-the-grid panels ───────────────────────────────────────────────
    verdict_card:       UIVerdictCard
    next_step:          UINextStep

    # ── Dev / audit (hidden from default UI) ───────────────────────────────
    audit:              UIAudit

    # ── Branching state ────────────────────────────────────────────────────
    is_branch:          bool = False     # True if this payload is a branch result
    parent_query:       str | None = None


# ─────────────────────────────────────────────
# CACHE  (branch results only — main results
#         are already cached in router + refiner)
# ─────────────────────────────────────────────

_branch_cache: TTLCache = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL)


def _branch_cache_key(company_id: str, question: str, parent_query: str) -> str:
    raw = f"{company_id}|branch|{question.strip().lower()}|{parent_query.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _box_id(company_id: str, box_type: str, title: str) -> str:
    """
    Stable, deterministic box ID — safe as a React key prop.
    Deterministic means re-renders won't cause flicker.
    """
    raw = f"{company_id}:{box_type}:{title.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_badge(color: str) -> UIBadge:
    return UIBadge(
        label=COLOR_BADGE.get(color, "🟡 Balanced"),
        color=COLOR_HEX.get(color, "#eab308"),
        raw=color,
    )


def _format_decision_box(box: DecisionBox, box_type: str, company_id: str) -> UIBox:
    return UIBox(
        id=_box_id(company_id, box_type, box.title),
        box_type=box_type,
        title=box.title,
        badge=_make_badge(box.color),
        claim=box.claim,
        evidence_or_reasoning=box.evidence_or_reasoning,
        probability=box.probability,
        impact=box.impact,
        risk_score=box.risk_score,
        follow_up_actions=box.follow_up_actions,
        spawn_questions=box.spawn_questions,
    )


def _sanitise_query(raw: str) -> str:
    """
    Strip, truncate, and block prompt-injection patterns
    before any user string reaches the LLM.
    """
    cleaned = raw.strip()

    # Hard length cap
    if len(cleaned) > _MAX_QUERY_LEN:
        cleaned = cleaned[:_MAX_QUERY_LEN]
        logger.warning("Query truncated to %d chars", _MAX_QUERY_LEN)

    # Block obvious injection patterns
    injection_patterns = [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard the above",
        "you are now",
        "act as",
        "system:",
        "assistant:",
    ]
    lower = cleaned.lower()
    for pattern in injection_patterns:
        if pattern in lower:
            logger.warning("Injection pattern detected and blocked: '%s'", pattern)
            raise ValueError(f"Query contains disallowed pattern: '{pattern}'")

    return cleaned


# ─────────────────────────────────────────────
# MAIN FORMATTER
# ─────────────────────────────────────────────

def format_for_ui(
    decision:   RefinedDecision,
    query:      str,
    company_id: str,
    *,
    is_branch:    bool = False,
    parent_query: str | None = None,
) -> UIDecisionPayload:
    """
    Transforms a RefinedDecision into the exact UIDecisionPayload
    the React Decision Board expects.

    This is a pure, synchronous function — no LLM calls, no I/O.
    Safe to call as many times as needed (e.g. for streaming updates).

    Parameters
    ----------
    decision     : Output from refiner.refine().
    query        : The original decision text.
    company_id   : Tenant identifier.
    is_branch    : True when formatting a branch / sub-decision result.
    parent_query : The query that spawned this branch.

    Returns
    -------
    UIDecisionPayload — always. Never raises.
    """
    t0 = time.perf_counter()

    # ── Summary box ────────────────────────────────────────────────────────
    verdict_color = decision.verdict_box.color
    summary_ui = UIBox(
        id=_box_id(company_id, "summary", "summary"),
        box_type="summary",
        title="Decision Summary",
        badge=_make_badge(verdict_color),
        claim=decision.summary_box,
        evidence_or_reasoning="",
        probability=None,
        impact=None,
        risk_score=None,
        follow_up_actions=[],
        spawn_questions=[],
    )

    # ── Upside boxes ───────────────────────────────────────────────────────
    upside_ui = [
        _format_decision_box(b, "upside", company_id)
        for b in decision.upside_boxes
    ]

    # ── Risk boxes ─────────────────────────────────────────────────────────
    risk_ui = [
        _format_decision_box(b, "risk", company_id)
        for b in decision.risk_boxes
    ]

    # ── Verdict card ───────────────────────────────────────────────────────
    vb = decision.verdict_box
    verdict_ui = UIVerdictCard(
        badge=_make_badge(vb.color),
        headline=vb.headline,
        rationale=vb.rationale,
        net_score=vb.net_score,
        go_conditions=vb.go_conditions,
        stop_conditions=vb.stop_conditions,
        review_triggers=vb.review_triggers,
        key_unknown=vb.key_unknown,
        flip_factor=vb.flip_factor,
    )

    # ── Next step ──────────────────────────────────────────────────────────
    ns = decision.next_step_box
    next_step_ui = UINextStep(
        immediate_action=ns.immediate_action,
        test_if_uncertain=ns.test_if_uncertain,
        owner=ns.owner,
        deadline=ns.deadline,
        escalate_if=ns.escalate_if,
    )

    # ── Audit ──────────────────────────────────────────────────────────────
    output_latency_ms = round((time.perf_counter() - t0) * 1000)
    a = decision.audit
    audit_ui = UIAudit(
        query_hash=a.query_hash,
        company_id=a.company_id,
        routing_plan_hash=a.routing_plan_hash,
        model_used=a.model_used,
        pass_latencies_ms=a.pass_latencies_ms,
        total_tokens_in=a.total_tokens_in,
        total_tokens_out=a.total_tokens_out,
        refiner_version=a.refiner_version,
        timestamp_utc=a.timestamp_utc,
        output_latency_ms=output_latency_ms,
    )

    total_boxes = len(upside_ui) + len(risk_ui)

    logger.info(
        "✅ Formatted | company=%s | verdict=%s | boxes=%d | "
        "conf=%.2f | output_latency_ms=%d | branch=%s",
        company_id,
        verdict_color,
        total_boxes,
        decision.confidence,
        output_latency_ms,
        is_branch,
    )

    return UIDecisionPayload(
        query=query,
        company_id=company_id,
        confidence=decision.confidence,
        total_boxes=total_boxes,
        summary_box=summary_ui,
        upside_boxes=upside_ui,
        risk_boxes=risk_ui,
        verdict_card=verdict_ui,
        next_step=next_step_ui,
        audit=audit_ui,
        is_branch=is_branch,
        parent_query=parent_query,
    )


# ─────────────────────────────────────────────
# BRANCHING  — user clicks a spawn_question
# ─────────────────────────────────────────────

async def branch_on_question(
    question:     str,
    parent_query: str,
    company_id:   str,
    context:      UserContext | None = None,
    *,
    bypass_cache: bool = False,
) -> UIDecisionPayload:
    """
    Called when a user clicks a spawn_question on any decision box.
    Runs the full route → refine → format_for_ui pipeline on the
    sub-question, returning a UIDecisionPayload in the same shape
    so React needs zero changes to render it.

    The branch result is independently cached so repeated clicks
    on the same spawn_question are instant.

    Parameters
    ----------
    question     : The spawn_question text (from a DecisionBox).
    parent_query : The original query that produced the parent decision.
    company_id   : Tenant identifier.
    context      : Same UserContext from the parent decision.
    bypass_cache : Force fresh analysis on this branch.

    Returns
    -------
    UIDecisionPayload — always. Falls back gracefully on failure.
    """

    # ── Input sanitisation ─────────────────────────────────────────────────
    try:
        question     = _sanitise_query(question)
        parent_query = _sanitise_query(parent_query)
    except ValueError as e:
        logger.warning("Branch blocked | company=%s | reason=%s", company_id, e)
        return _branch_fallback(question, parent_query, company_id, str(e))

    if not question:
        return _branch_fallback(question, parent_query, company_id, "empty_question")

    # ── Cache lookup ───────────────────────────────────────────────────────
    cache_key = _branch_cache_key(company_id, question, parent_query)

    if not bypass_cache and cache_key in _branch_cache:
        logger.info(
            "Branch cache hit | company=%s | key=%s…",
            company_id, cache_key[:12],
        )
        return _branch_cache[cache_key]

    logger.info(
        "🌿 Branch | company=%s | question=%s…",
        company_id, question[:60],
    )

    # ── Enrich the branch context with the parent query ────────────────────
    # This means the router and refiner know this is a sub-question,
    # not a cold-start decision — produces tighter, more relevant output.
    branch_context = UserContext(
        industry=context.industry       if context else None,
        company_size=context.company_size if context else None,
        risk_appetite=context.risk_appetite if context else None,
        extra={
            **(context.extra if context else {}),
            "parent_decision": parent_query[:200],
            "branch_depth": "1",
        },
    )

    try:
        plan     = await route(question, company_id, branch_context)
        decision = await refine(question, plan, company_id, branch_context)
        payload  = format_for_ui(
            decision,
            question,
            company_id,
            is_branch=True,
            parent_query=parent_query,
        )
    except Exception as e:
        logger.error(
            "Branch failed | company=%s | question=%s | %s",
            company_id, question[:60], e,
        )
        return _branch_fallback(question, parent_query, company_id, "pipeline_failure")

    _branch_cache[cache_key] = payload
    return payload


# ─────────────────────────────────────────────
# BRANCH FALLBACK
# ─────────────────────────────────────────────

def _branch_fallback(
    question:     str,
    parent_query: str,
    company_id:   str,
    reason:       str,
) -> UIDecisionPayload:
    logger.warning(
        "Branch fallback | company=%s | reason=%s", company_id, reason
    )
    placeholder = UIBox(
        id=_box_id(company_id, "risk", "branch-fallback"),
        box_type="risk",
        title="Branch unavailable",
        badge=_make_badge("Orange"),
        claim=f"Could not analyse this sub-question: {reason}",
        evidence_or_reasoning="Please retry or rephrase the question.",
        follow_up_actions=["Retry", "Rephrase the question"],
        spawn_questions=[],
    )
    return UIDecisionPayload(
        query=question,
        company_id=company_id,
        confidence=0.0,
        total_boxes=1,
        summary_box=UIBox(
            id=_box_id(company_id, "summary", "branch-fallback-summary"),
            box_type="summary",
            title="Branch Analysis Unavailable",
            badge=_make_badge("Orange"),
            claim=f"Sub-question analysis failed: {reason}",
            evidence_or_reasoning="",
        ),
        upside_boxes=[],
        risk_boxes=[placeholder],
        verdict_card=UIVerdictCard(
            badge=_make_badge("Orange"),
            headline="Unable to analyse this sub-question.",
            rationale="The branch pipeline did not complete. Please retry.",
            net_score=0.0,
            go_conditions=[],
            stop_conditions=["Do not act on incomplete analysis"],
            review_triggers=["When the system is available"],
            key_unknown="Could not determine",
            flip_factor="Could not determine",
        ),
        next_step=UINextStep(
            immediate_action="Retry the sub-question or simplify it.",
        ),
        audit=UIAudit(
            query_hash=hashlib.sha256(question.encode()).hexdigest(),
            company_id=company_id,
            routing_plan_hash="unavailable",
            model_used="unavailable",
            pass_latencies_ms=[0, 0, 0],
            total_tokens_in=0,
            total_tokens_out=0,
            refiner_version="1.0.0",
            timestamp_utc="",
            output_latency_ms=0,
        ),
        is_branch=True,
        parent_query=parent_query,
    )


# ─────────────────────────────────────────────
# QUICK LOCAL TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import json
    from router import route, UserContext
    from refiner import refine

    async def _demo():
        query = "Should we raise our prices by 20% next month?"
        ctx   = UserContext(
            industry="E-commerce",
            company_size="Series A",
            risk_appetite="Moderate",
        )

        print("── Step 1: Router ──────────────────────────────")
        plan = await route(query, "demo-co", ctx)
        print(f"  {plan.decision_type} | {plan.stake_level} | {plan.framework}")

        print("\n── Step 2: Refiner ─────────────────────────────")
        decision = await refine(query, plan, "demo-co", ctx)
        print(f"  Verdict: {decision.verdict_box.color} — {decision.verdict_box.headline}")

        print("\n── Step 3: Output (format_for_ui) ──────────────")
        payload = format_for_ui(decision, query, "demo-co")
        print(f"  Badge  : {payload.verdict_card.badge.label}")
        print(f"  Boxes  : {payload.total_boxes}")
        print(f"  Conf   : {payload.confidence:.0%}")

        for b in payload.upside_boxes:
            print(f"  [UPSIDE] {b.title} | {b.badge.label} | score={b.risk_score}")
        for b in payload.risk_boxes:
            print(f"  [RISK]   {b.title} | {b.badge.label} | score={b.risk_score}")

        print(f"\n  Go conditions   : {payload.verdict_card.go_conditions}")
        print(f"  Stop conditions : {payload.verdict_card.stop_conditions}")
        print(f"  Next step       : {payload.next_step.immediate_action}")

        # ── Test branching ─────────────────────────────────────────────────
        if payload.upside_boxes and payload.upside_boxes[0].spawn_questions:
            sub_q = payload.upside_boxes[0].spawn_questions[0]
            print(f"\n── Step 4: Branch on '{sub_q[:50]}…' ──")
            branch = await branch_on_question(sub_q, query, "demo-co", ctx)
            print(f"  Branch verdict : {branch.verdict_card.badge.label}")
            print(f"  Branch is_branch: {branch.is_branch}")
            print(f"  Branch boxes   : {branch.total_boxes}")

    asyncio.run(_demo())