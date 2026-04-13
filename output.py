"""
output.py — Three AI Finance UI Output Layer

Finance-first changes:
- Adds finance_snapshot to UIDecisionPayload
- Surfaces finance metrics, source freshness, and evidence notes
- Extends audit output with finance-specific fields when available
- Keeps branching + caching behavior
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Literal, Any

from pydantic import BaseModel, Field
from cachetools import TTLCache

from router import route, UserContext, RoutingPlan
from refiner import (
    refine, RefinedDecision, DecisionBox, VerdictBox,
    NextStepBox, AuditTrail, ReasoningTrail,
)

import os
_CACHE_TTL     = int(os.getenv("THINK_AI_CACHE_TTL",  "3600"))
_CACHE_MAXSIZE = int(os.getenv("THINK_AI_CACHE_MAX",  "2000"))
_MAX_QUERY_LEN = int(os.getenv("THINK_AI_MAX_QUERY",  "2000"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("three-ai.output")

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
# UI SCHEMAS
# ─────────────────────────────────────────────

class UIBadge(BaseModel):
    label: str
    color: str
    raw: str


class UIBox(BaseModel):
    id: str
    box_type: Literal["summary", "upside", "risk", "verdict", "next_step"]
    title: str
    badge: UIBadge
    claim: str
    evidence_or_reasoning: str
    probability: int | None = None
    impact: int | None = None
    risk_score: float | None = None
    follow_up_actions: list[str] = Field(default_factory=list)
    spawn_questions: list[str] = Field(default_factory=list)


class UIVerdictCard(BaseModel):
    badge: UIBadge
    headline: str
    rationale: str
    net_score: float
    go_conditions: list[str]
    stop_conditions: list[str]
    review_triggers: list[str]
    key_unknown: str
    flip_factor: str


class UINextStep(BaseModel):
    immediate_action: str
    test_if_uncertain: str | None = None
    owner: str | None = None
    deadline: str | None = None
    escalate_if: str | None = None


class UIAudit(BaseModel):
    query_hash: str
    company_id: str
    routing_plan_hash: str
    model_used: str
    pass_latencies_ms: list[int]
    total_tokens_in: int
    total_tokens_out: int
    refiner_version: str
    timestamp_utc: str
    output_latency_ms: int

    # finance-aware audit fields
    finance_snapshot_hash: str | None = None
    data_as_of_utc: str | None = None
    cache_bypassed: bool | None = None


class UIFinanceSnapshot(BaseModel):
    as_of_utc: str | None = None
    currency: str | None = None
    reporting_period: str | None = None
    analysis_horizon_days: int | None = None
    is_live_data: bool = False

    sources_used: list[str] = Field(default_factory=list)
    source_freshness: dict[str, str] = Field(default_factory=dict)

    cash_balance: float | None = None
    available_liquidity: float | None = None
    monthly_burn: float | None = None
    runway_months: float | None = None

    mrr: float | None = None
    arr: float | None = None
    revenue_last_30d: float | None = None
    revenue_prev_30d: float | None = None
    revenue_growth_pct: float | None = None

    gross_margin_pct: float | None = None
    ebitda_margin_pct: float | None = None
    net_margin_pct: float | None = None

    opex_last_30d: float | None = None
    opex_prev_30d: float | None = None
    opex_growth_pct: float | None = None

    ar_total: float | None = None
    ar_overdue_30_plus: float | None = None
    ap_total: float | None = None
    ap_due_30d: float | None = None

    failed_payment_rate_pct: float | None = None
    customer_concentration_pct: float | None = None
    top_customer_share_pct: float | None = None
    logo_churn_pct: float | None = None
    revenue_churn_pct: float | None = None
    nrr_pct: float | None = None

    pipeline_coverage: float | None = None
    forecast_vs_actual_pct: float | None = None

    debt_service_coverage_ratio: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    covenant_headroom_pct: float | None = None

    headcount: int | None = None
    notes: list[str] = Field(default_factory=list)


class UIDecisionPayload(BaseModel):
    query: str
    company_id: str
    confidence: float
    total_boxes: int

    summary_box: UIBox
    upside_boxes: list[UIBox]
    risk_boxes: list[UIBox]

    verdict_card: UIVerdictCard
    next_step: UINextStep
    audit: UIAudit

    is_branch: bool = False
    parent_query: str | None = None

    reasoning_trail: ReasoningTrail | None = None
    finance_snapshot: UIFinanceSnapshot | None = None


# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────

_branch_cache: TTLCache = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL)


def _branch_cache_key(company_id: str, question: str, parent_query: str) -> str:
    raw = f"{company_id}|branch|{question.strip().lower()}|{parent_query.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _box_id(company_id: str, box_type: str, title: str) -> str:
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
    cleaned = raw.strip()
    if len(cleaned) > _MAX_QUERY_LEN:
        cleaned = cleaned[:_MAX_QUERY_LEN]
        logger.warning("Query truncated to %d chars", _MAX_QUERY_LEN)

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


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _build_finance_snapshot(context: UserContext | None) -> UIFinanceSnapshot | None:
    if not context or not getattr(context, "extra", None):
        return None

    extra = context.extra or {}

    sources_used = extra.get("sources_used", [])
    if not isinstance(sources_used, list):
        sources_used = []

    source_freshness = extra.get("source_freshness", {})
    if not isinstance(source_freshness, dict):
        source_freshness = {}

    notes = extra.get("evidence_notes", [])
    if not isinstance(notes, list):
        notes = []

    snapshot = UIFinanceSnapshot(
        as_of_utc=extra.get("as_of_utc"),
        currency=extra.get("currency"),
        reporting_period=extra.get("reporting_period"),
        analysis_horizon_days=_to_int(extra.get("analysis_horizon_days")),
        is_live_data=bool(extra.get("is_live_data", False)),

        sources_used=[str(x) for x in sources_used],
        source_freshness={str(k): str(v) for k, v in source_freshness.items()},

        cash_balance=_to_float(extra.get("cash_balance")),
        available_liquidity=_to_float(extra.get("available_liquidity")),
        monthly_burn=_to_float(extra.get("monthly_burn")),
        runway_months=_to_float(extra.get("runway_months")),

        mrr=_to_float(extra.get("mrr")),
        arr=_to_float(extra.get("arr")),
        revenue_last_30d=_to_float(extra.get("revenue_last_30d")),
        revenue_prev_30d=_to_float(extra.get("revenue_prev_30d")),
        revenue_growth_pct=_to_float(extra.get("revenue_growth_pct")),

        gross_margin_pct=_to_float(extra.get("gross_margin_pct")),
        ebitda_margin_pct=_to_float(extra.get("ebitda_margin_pct")),
        net_margin_pct=_to_float(extra.get("net_margin_pct")),

        opex_last_30d=_to_float(extra.get("opex_last_30d")),
        opex_prev_30d=_to_float(extra.get("opex_prev_30d")),
        opex_growth_pct=_to_float(extra.get("opex_growth_pct")),

        ar_total=_to_float(extra.get("ar_total")),
        ar_overdue_30_plus=_to_float(extra.get("ar_overdue_30_plus")),
        ap_total=_to_float(extra.get("ap_total")),
        ap_due_30d=_to_float(extra.get("ap_due_30d")),

        failed_payment_rate_pct=_to_float(extra.get("failed_payment_rate_pct")),
        customer_concentration_pct=_to_float(extra.get("customer_concentration_pct")),
        top_customer_share_pct=_to_float(extra.get("top_customer_share_pct")),
        logo_churn_pct=_to_float(extra.get("logo_churn_pct")),
        revenue_churn_pct=_to_float(extra.get("revenue_churn_pct")),
        nrr_pct=_to_float(extra.get("nrr_pct")),

        pipeline_coverage=_to_float(extra.get("pipeline_coverage")),
        forecast_vs_actual_pct=_to_float(extra.get("forecast_vs_actual_pct")),

        debt_service_coverage_ratio=_to_float(extra.get("debt_service_coverage_ratio")),
        current_ratio=_to_float(extra.get("current_ratio")),
        quick_ratio=_to_float(extra.get("quick_ratio")),
        covenant_headroom_pct=_to_float(extra.get("covenant_headroom_pct")),

        headcount=_to_int(extra.get("headcount")),
        notes=[str(x) for x in notes],
    )

    has_real_data = any(
        getattr(snapshot, field_name) is not None
        for field_name in [
            "cash_balance", "monthly_burn", "runway_months",
            "mrr", "arr", "revenue_last_30d", "revenue_growth_pct",
            "gross_margin_pct", "ar_overdue_30_plus",
            "failed_payment_rate_pct", "customer_concentration_pct"
        ]
    ) or bool(snapshot.sources_used)

    return snapshot if has_real_data else None


# ─────────────────────────────────────────────
# MAIN FORMATTER
# ─────────────────────────────────────────────

def format_for_ui(
    decision: RefinedDecision,
    query: str,
    company_id: str,
    *,
    context: UserContext | None = None,
    is_branch: bool = False,
    parent_query: str | None = None,
) -> UIDecisionPayload:
    t0 = time.perf_counter()

    verdict_color = decision.verdict_box.color
    summary_ui = UIBox(
        id=_box_id(company_id, "summary", "summary"),
        box_type="summary",
        title="Decision Summary",
        badge=_make_badge(verdict_color),
        claim=decision.summary_box,
        evidence_or_reasoning="",
    )

    upside_ui = [_format_decision_box(b, "upside", company_id) for b in decision.upside_boxes]
    risk_ui = [_format_decision_box(b, "risk", company_id) for b in decision.risk_boxes]

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

    ns = decision.next_step_box
    next_step_ui = UINextStep(
        immediate_action=ns.immediate_action,
        test_if_uncertain=ns.test_if_uncertain,
        owner=ns.owner,
        deadline=ns.deadline,
        escalate_if=ns.escalate_if,
    )

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
        finance_snapshot_hash=getattr(a, "finance_snapshot_hash", None),
        data_as_of_utc=getattr(a, "data_as_of_utc", None),
        cache_bypassed=getattr(a, "cache_bypassed", None),
    )

    finance_snapshot = _build_finance_snapshot(context)
    total_boxes = len(upside_ui) + len(risk_ui)

    logger.info(
        "✅ Formatted | company=%s | verdict=%s | boxes=%d | conf=%.2f | output_latency_ms=%d | branch=%s | finance_snapshot=%s",
        company_id,
        verdict_color,
        total_boxes,
        decision.confidence,
        output_latency_ms,
        is_branch,
        bool(finance_snapshot),
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
        reasoning_trail=decision.reasoning_trail,
        finance_snapshot=finance_snapshot,
    )


# ─────────────────────────────────────────────
# BRANCHING
# ─────────────────────────────────────────────

async def branch_on_question(
    question: str,
    parent_query: str,
    company_id: str,
    context: UserContext | None = None,
    *,
    bypass_cache: bool = False,
) -> UIDecisionPayload:
    try:
        question = _sanitise_query(question)
        parent_query = _sanitise_query(parent_query)
    except ValueError as e:
        logger.warning("Branch blocked | company=%s | reason=%s", company_id, e)
        return _branch_fallback(question, parent_query, company_id, str(e), context=context)

    if not question:
        return _branch_fallback(question, parent_query, company_id, "empty_question", context=context)

    cache_key = _branch_cache_key(company_id, question, parent_query)
    if not bypass_cache and cache_key in _branch_cache:
        logger.info("Branch cache hit | company=%s | key=%s…", company_id, cache_key[:12])
        return _branch_cache[cache_key]

    logger.info("🌿 Branch | company=%s | question=%s…", company_id, question[:60])

    branch_context = UserContext(
        industry=context.industry if context else None,
        company_size=context.company_size if context else None,
        risk_appetite=context.risk_appetite if context else None,
        extra={
            **(context.extra if context else {}),
            "parent_decision": parent_query[:200],
            "branch_depth": "1",
        },
    )

    try:
        plan = await route(question, company_id, branch_context, bypass_cache=bypass_cache)
        decision = await refine(question, plan, company_id, branch_context, bypass_cache=bypass_cache)
        payload = format_for_ui(
            decision,
            question,
            company_id,
            context=branch_context,
            is_branch=True,
            parent_query=parent_query,
        )
    except Exception as e:
        logger.error("Branch failed | company=%s | question=%s | %s", company_id, question[:60], e)
        return _branch_fallback(question, parent_query, company_id, "pipeline_failure", context=branch_context)

    _branch_cache[cache_key] = payload
    return payload


def _branch_fallback(
    question: str,
    parent_query: str,
    company_id: str,
    reason: str,
    *,
    context: UserContext | None = None,
) -> UIDecisionPayload:
    logger.warning("Branch fallback | company=%s | reason=%s", company_id, reason)

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
            immediate_action="Retry the sub-question or simplify it."
        ),
        audit=UIAudit(
            query_hash=hashlib.sha256(question.encode()).hexdigest(),
            company_id=company_id,
            routing_plan_hash="unavailable",
            model_used="unavailable",
            pass_latencies_ms=[0, 0, 0],
            total_tokens_in=0,
            total_tokens_out=0,
            refiner_version="2.0.0-finance",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            output_latency_ms=0,
            finance_snapshot_hash=None,
            data_as_of_utc=None,
            cache_bypassed=True,
        ),
        is_branch=True,
        parent_query=parent_query,
        reasoning_trail=None,
        finance_snapshot=_build_finance_snapshot(context),
    )