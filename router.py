"""
router.py — Three AI Finance Triage Router
==========================================

Responsibility:
- Classify an incoming query into a finance-first RoutingPlan.
- Detect whether the query is a finance decision, finance analysis request,
  general chat, or an action request.
- Emit structured finance-native metadata for the downstream Finance Refiner.

Design goals:
- Strong bias toward enterprise finance routing when finance language is present.
- Better handling of real-time / current-state financial questions.
- Replace generic business lenses with finance-native analytical lenses.
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
from typing import Any, Literal

from cachetools import TTLCache
from openai import AsyncOpenAI, APIStatusError, APITimeoutError
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

_MODEL = os.getenv("THINK_AI_ROUTER_MODEL", "gpt-4o-mini")
_TIMEOUT_SEC = float(os.getenv("THINK_AI_ROUTER_TIMEOUT", "15"))
_MAX_RETRIES = int(os.getenv("THINK_AI_ROUTER_RETRIES", "3"))
_CACHE_TTL = int(os.getenv("THINK_AI_CACHE_TTL", "3600"))
_CACHE_MAXSIZE = int(os.getenv("THINK_AI_CACHE_MAX", "2000"))
_LIVE_CACHE_TTL = int(os.getenv("THINK_AI_LIVE_CACHE_TTL", "120"))

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        load_dotenv()
        _client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            timeout=_TIMEOUT_SEC,
            max_retries=0,
        )
    return _client


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("three-ai.finance-router")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

INTENTS = ["decision", "chat", "action"]

DECISION_TYPES = [
    "cash_flow_risk",
    "liquidity_runway",
    "revenue_quality",
    "pricing_strategy",
    "margin_pressure",
    "working_capital",
    "collections_risk",
    "customer_concentration",
    "forecast_reliability",
    "budget_allocation",
    "debt_covenants",
    "capital_allocation",
    "unit_economics",
    "board_finance",
    "enterprise_finance",
    "general",
]

STAKE_LEVELS = ["Low", "Medium", "High"]

AVAILABLE_LENSES = [
    "Liquidity",
    "Cash Runway",
    "Revenue Durability",
    "Margin Quality",
    "Working Capital",
    "Collections Risk",
    "Customer Concentration",
    "Forecast Reliability",
    "Downside Severity",
    "Upside Leverage",
    "Debt and Covenants",
    "Capital Efficiency",
    "Timing Mismatch",
    "Sensitivity Analysis",
    "Data Freshness",
]

FRAMEWORKS = [
    "Finance Quick View",
    "Finance Deep Dive",
    "Board / CFO Review",
]

BOXES_BY_STAKE: dict[str, int] = {
    "Low": 2,
    "Medium": 4,
    "High": 6,
}

DEPTH_BY_STAKE: dict[str, int] = {
    "Low": 2,
    "Medium": 4,
    "High": 5,
}

FINANCE_KEYWORDS = [
    "cash",
    "runway",
    "burn",
    "revenue",
    "arr",
    "mrr",
    "margin",
    "gross margin",
    "ebitda",
    "profit",
    "pricing",
    "invoice",
    "receivables",
    "ar",
    "ap",
    "working capital",
    "collections",
    "failed payments",
    "churn",
    "nrr",
    "forecast",
    "budget",
    "board",
    "debt",
    "covenant",
    "liquidity",
    "customer concentration",
    "top customer",
    "unit economics",
    "opex",
    "finance",
]

LIVE_QUERY_HINTS = [
    "today",
    "right now",
    "currently",
    "latest",
    "live",
    "this week",
    "this month",
    "as of now",
    "current",
    "real-time",
    "real time",
    "just now",
]


# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────

class UserContext(BaseModel):
    industry: str | None = None
    company_size: str | None = None
    risk_appetite: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class RoutingPlan(BaseModel):
    intent: Literal["decision", "chat", "action"] = Field(default="decision")
    decision_type: str
    stake_level: str
    framework: str
    selected_lenses: list[str] = Field(min_length=1, max_length=6)
    box_count: int = Field(ge=1, le=6)
    required_questions: list[str] = Field(max_length=5)
    assumptions: list[str]
    constraints_detected: list[str]
    reasoning_depth: int = Field(ge=1, le=5)
    is_reversible: bool
    confidence: float = Field(ge=0.0, le=1.0)
    is_live_query: bool = False
    analysis_horizon_days: int = Field(default=90, ge=1, le=3650)
    finance_priority: Literal["low", "medium", "high"] = "medium"

    @field_validator("intent", mode="before")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        return v if v in INTENTS else "decision"

    @field_validator("decision_type")
    @classmethod
    def validate_decision_type(cls, v: str) -> str:
        return v if v in DECISION_TYPES else "general"

    @field_validator("stake_level")
    @classmethod
    def validate_stake_level(cls, v: str) -> str:
        return v if v in STAKE_LEVELS else "Medium"

    @field_validator("framework")
    @classmethod
    def validate_framework(cls, v: str) -> str:
        return v if v in FRAMEWORKS else "Finance Deep Dive"

    @field_validator("selected_lenses")
    @classmethod
    def validate_lenses(cls, v: list[str]) -> list[str]:
        valid = [lens for lens in v if lens in AVAILABLE_LENSES]
        return valid[:6] or ["Liquidity", "Downside Severity"]

    @field_validator("box_count", mode="before")
    @classmethod
    def clamp_box_count(cls, v: int) -> int:
        return max(1, min(6, int(v)))

    @field_validator("analysis_horizon_days", mode="before")
    @classmethod
    def clamp_horizon(cls, v: int) -> int:
        return max(1, min(3650, int(v)))


# ─────────────────────────────────────────────
# DETERMINISTIC HELPERS
# ─────────────────────────────────────────────

def _contains_finance_language(query: str, context: UserContext | None) -> bool:
    q = query.lower()
    if any(keyword in q for keyword in FINANCE_KEYWORDS):
        return True

    extra = context.extra if context else {}
    finance_metric_keys = {
        "cash_balance",
        "available_liquidity",
        "monthly_burn",
        "runway_months",
        "mrr",
        "arr",
        "revenue_last_30d",
        "revenue_growth_pct",
        "gross_margin_pct",
        "ebitda_margin_pct",
        "net_margin_pct",
        "opex_growth_pct",
        "ar_total",
        "ar_overdue_30_plus",
        "ap_due_30d",
        "failed_payment_rate_pct",
        "customer_concentration_pct",
        "top_customer_share_pct",
        "logo_churn_pct",
        "revenue_churn_pct",
        "nrr_pct",
        "pipeline_coverage",
        "forecast_vs_actual_pct",
        "debt_service_coverage_ratio",
        "current_ratio",
        "quick_ratio",
        "covenant_headroom_pct",
    }
    return any(key in extra for key in finance_metric_keys)


def _detect_live_query(query: str, context: UserContext | None) -> bool:
    q = query.lower()
    if any(hint in q for hint in LIVE_QUERY_HINTS):
        return True

    extra = context.extra if context else {}
    return bool(extra.get("is_live_data", False))


def _infer_horizon_days(query: str) -> int:
    q = query.lower()

    if "today" in q or "this week" in q:
        return 7
    if "this month" in q or "30 days" in q or "next 30 days" in q:
        return 30
    if "quarter" in q or "90 days" in q or "next quarter" in q:
        return 90
    if "6 months" in q or "180 days" in q:
        return 180
    if "12 months" in q or "1 year" in q or "12-month" in q:
        return 365

    return 90


def _infer_decision_type(query: str) -> str:
    q = query.lower()

    if any(x in q for x in ("runway", "liquidity", "cash balance", "burn", "cash flow")):
        return "liquidity_runway"
    if any(x in q for x in ("revenue", "mrr", "arr", "nrr", "churn", "growth")):
        return "revenue_quality"
    if any(x in q for x in ("price", "pricing", "discount", "package")):
        return "pricing_strategy"
    if any(x in q for x in ("margin", "gross margin", "ebitda", "profitability", "opex")):
        return "margin_pressure"
    if any(x in q for x in ("receivables", "collections", "ar", "ap", "invoice", "working capital")):
        return "working_capital"
    if any(x in q for x in ("collections", "overdue", "failed payments")):
        return "collections_risk"
    if any(x in q for x in ("customer concentration", "top customer", "concentration")):
        return "customer_concentration"
    if any(x in q for x in ("forecast", "budget", "plan vs actual", "board pack")):
        return "forecast_reliability"
    if any(x in q for x in ("debt", "covenant", "loan", "facility")):
        return "debt_covenants"
    if any(x in q for x in ("capital allocation", "investment", "hire", "expansion")):
        return "capital_allocation"
    if any(x in q for x in ("unit economics", "cac", "ltv", "payback")):
        return "unit_economics"
    if any(x in q for x in ("board", "cfo", "finance leadership")):
        return "board_finance"

    return "enterprise_finance"


def _infer_finance_priority(decision_type: str, stake_level: str, is_live_query: bool) -> str:
    if is_live_query:
        return "high"
    if stake_level == "High":
        return "high"
    if decision_type in {
        "liquidity_runway",
        "debt_covenants",
        "working_capital",
        "collections_risk",
    }:
        return "high"
    if decision_type in {
        "revenue_quality",
        "margin_pressure",
        "forecast_reliability",
        "customer_concentration",
    }:
        return "medium"
    return "low"


def _default_lenses_for_type(decision_type: str, is_live_query: bool) -> list[str]:
    base = ["Downside Severity", "Data Freshness"]

    mapping: dict[str, list[str]] = {
        "cash_flow_risk": ["Liquidity", "Cash Runway", "Timing Mismatch", "Downside Severity"],
        "liquidity_runway": ["Liquidity", "Cash Runway", "Timing Mismatch", "Downside Severity"],
        "revenue_quality": ["Revenue Durability", "Customer Concentration", "Upside Leverage", "Forecast Reliability"],
        "pricing_strategy": ["Revenue Durability", "Margin Quality", "Sensitivity Analysis", "Upside Leverage"],
        "margin_pressure": ["Margin Quality", "Capital Efficiency", "Timing Mismatch", "Downside Severity"],
        "working_capital": ["Working Capital", "Collections Risk", "Timing Mismatch", "Liquidity"],
        "collections_risk": ["Collections Risk", "Working Capital", "Liquidity", "Downside Severity"],
        "customer_concentration": ["Customer Concentration", "Revenue Durability", "Downside Severity", "Sensitivity Analysis"],
        "forecast_reliability": ["Forecast Reliability", "Sensitivity Analysis", "Data Freshness", "Downside Severity"],
        "budget_allocation": ["Capital Efficiency", "Upside Leverage", "Downside Severity", "Sensitivity Analysis"],
        "debt_covenants": ["Debt and Covenants", "Liquidity", "Timing Mismatch", "Downside Severity"],
        "capital_allocation": ["Capital Efficiency", "Upside Leverage", "Sensitivity Analysis", "Downside Severity"],
        "unit_economics": ["Margin Quality", "Capital Efficiency", "Revenue Durability", "Sensitivity Analysis"],
        "board_finance": ["Liquidity", "Revenue Durability", "Margin Quality", "Forecast Reliability", "Downside Severity"],
        "enterprise_finance": ["Liquidity", "Revenue Durability", "Margin Quality", "Forecast Reliability", "Downside Severity"],
        "general": ["Liquidity", "Downside Severity"],
    }

    lenses = mapping.get(decision_type, base)
    if is_live_query and "Data Freshness" not in lenses:
        lenses = ["Data Freshness"] + lenses
    return lenses[:6]


def _infer_stake_level(query: str, decision_type: str, is_live_query: bool) -> str:
    q = query.lower()

    if any(x in q for x in ("debt", "covenant", "runway", "bankruptcy", "insolvency", "fundraising", "board")):
        return "High"

    if decision_type in {
        "liquidity_runway",
        "debt_covenants",
        "board_finance",
        "capital_allocation",
    }:
        return "High"

    if decision_type in {
        "revenue_quality",
        "margin_pressure",
        "working_capital",
        "forecast_reliability",
        "customer_concentration",
        "collections_risk",
    }:
        return "Medium"

    if is_live_query:
        return "Medium"

    return "Low"


def _infer_reversibility(query: str, decision_type: str) -> bool:
    q = query.lower()

    if any(x in q for x in ("acquire", "debt", "loan", "fundraise", "restructure", "layoff", "shut down")):
        return False

    if decision_type in {"debt_covenants", "capital_allocation", "board_finance"}:
        return False

    return True


def _enforce_stake_rules(plan: "RoutingPlan") -> "RoutingPlan":
    if plan.decision_type in {"debt_covenants", "board_finance", "liquidity_runway"}:
        plan.stake_level = "High"

    if not plan.is_reversible and plan.stake_level == "Low":
        plan.stake_level = "Medium"

    if plan.is_live_query and plan.stake_level == "Low":
        plan.stake_level = "Medium"

    return plan


def _enforce_framework(plan: "RoutingPlan") -> "RoutingPlan":
    if plan.stake_level == "High":
        plan.framework = "Board / CFO Review"
    elif plan.stake_level == "Low":
        plan.framework = "Finance Quick View"
    else:
        plan.framework = "Finance Deep Dive"
    return plan


def _compute_confidence(plan: "RoutingPlan", context: "UserContext | None") -> float:
    score = 1.0

    if plan.required_questions:
        score *= 0.70

    if plan.decision_type == "general":
        score *= 0.75

    if context is None:
        score *= 0.90

    if plan.is_live_query:
        score *= 0.95

    extra = context.extra if context else {}
    if isinstance(extra, dict):
        if extra.get("is_live_data", False):
            score *= 1.0
        if extra.get("as_of_utc") is None and plan.is_live_query:
            score *= 0.90

    return round(max(0.0, min(score, 1.0)), 2)


def _required_questions_for_query(
    query: str,
    decision_type: str,
    context: UserContext | None,
) -> list[str]:
    questions: list[str] = []
    extra = context.extra if context else {}

    if decision_type in {"liquidity_runway", "cash_flow_risk"}:
        if "cash_balance" not in extra:
            questions.append("What is the current cash balance?")
        if "monthly_burn" not in extra and "runway_months" not in extra:
            questions.append("What is the current monthly burn or runway?")

    if decision_type == "revenue_quality":
        if "revenue_growth_pct" not in extra and "mrr" not in extra and "arr" not in extra:
            questions.append("What current revenue trend or recurring revenue metric should be used?")
        if "nrr_pct" not in extra and "revenue_churn_pct" not in extra:
            questions.append("Do you have retention or churn metrics?")

    if decision_type in {"working_capital", "collections_risk"}:
        if "ar_total" not in extra and "ar_overdue_30_plus" not in extra:
            questions.append("What receivables or overdue invoice data is available?")
        if "ap_due_30d" not in extra:
            questions.append("What payables are due in the next 30 days?")

    if decision_type == "debt_covenants":
        if "debt_service_coverage_ratio" not in extra and "covenant_headroom_pct" not in extra:
            questions.append("What debt covenant or debt service metrics are available?")

    return questions[:5]


def _detect_constraints(query: str, context: UserContext | None) -> list[str]:
    constraints: list[str] = []
    q = query.lower()

    if any(x in q for x in ("next 7 days", "this week", "today", "immediately", "urgent")):
        constraints.append("short decision window")
    if any(x in q for x in ("board", "investor", "lender", "bank", "audit")):
        constraints.append("external stakeholder scrutiny")
    if any(x in q for x in ("budget", "cash", "headcount freeze", "freeze")):
        constraints.append("resource constraint")
    if any(x in q for x in ("compliance", "covenant", "debt")):
        constraints.append("financing/compliance constraint")

    extra = context.extra if context else {}
    if isinstance(extra, dict) and extra.get("is_live_data", False):
        constraints.append("must use freshest available data")

    # dedupe preserve order
    seen = set()
    out = []
    for item in constraints:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _assumptions_for_plan(
    decision_type: str,
    context: UserContext | None,
    is_live_query: bool,
) -> list[str]:
    assumptions = [
        "Analysis should prioritize financially material outcomes over generic strategy advice.",
        "Supplied metrics and source timestamps are assumed to be directionally accurate unless contradicted.",
    ]

    if is_live_query:
        assumptions.append("The latest available financial data is more important than historical narrative.")

    if context is None:
        assumptions.append("Missing company context will be handled conservatively.")

    extra = context.extra if context else {}
    if isinstance(extra, dict):
        if not extra:
            assumptions.append("No structured finance snapshot was supplied.")
        if "currency" not in extra:
            assumptions.append("Currency defaults to the company’s primary reporting currency or USD.")
        if "analysis_horizon_days" not in extra:
            assumptions.append("Analysis horizon defaults to 90 days unless the question implies another window.")

    return assumptions[:5]


# ─────────────────────────────────────────────
# FALLBACK
# ─────────────────────────────────────────────

def _safe_fallback(reason: str) -> RoutingPlan:
    logger.warning("Finance router fallback | reason=%s", reason)
    return RoutingPlan(
        intent="chat",
        decision_type="general",
        stake_level="Medium",
        framework="Finance Deep Dive",
        selected_lenses=["Liquidity", "Downside Severity", "Data Freshness"],
        box_count=4,
        required_questions=["What financial question do you want answered?"],
        assumptions=["Insufficient finance context — conservative routing fallback used."],
        constraints_detected=[],
        reasoning_depth=4,
        is_reversible=True,
        confidence=0.35,
        is_live_query=False,
        analysis_horizon_days=90,
        finance_priority="medium",
    )


# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────

_cache: TTLCache = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL)
_live_cache: TTLCache = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_LIVE_CACHE_TTL)


def _cache_key(company_id: str, query: str, context: UserContext | None) -> str:
    raw = f"{company_id}|{query.strip().lower()}"
    if context:
        raw += f"|{context.model_dump_json(exclude_none=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = f"""
You are the Finance Triage Router for Three AI.

Your ONLY job is to classify the incoming query and return JSON metadata.

PRIMARY GOAL:
Strongly prefer finance-first routing whenever the question is about company performance,
cash, runway, revenue, margins, pricing, collections, debt, forecasts, budgets,
customer concentration, board finance, or enterprise financial risk/upside.

INTENTS:
- "chat": factual explanation, simple Q&A, definitions, coding, brainstorming, or non-decision analysis.
- "decision": use when the user wants a finance judgement, risk/upside analysis, prioritization, or tradeoff.
- "action": use only when the user wants generated content or a task drafted.

DECISION TYPES:
{", ".join(DECISION_TYPES)}

STAKE LEVELS:
- High: board/CFO/lender level, runway, debt, covenants, major capital allocation, existential risk
- Medium: meaningful revenue, margin, collections, forecast, pricing, customer risk
- Low: narrower, reversible, or exploratory finance questions

FRAMEWORKS:
- Finance Quick View
- Finance Deep Dive
- Board / CFO Review

AVAILABLE LENSES:
{", ".join(AVAILABLE_LENSES)}

You MUST return ONLY this exact JSON structure:
{{
  "intent": "decision or chat or action",
  "decision_type": "one of the decision types above",
  "stake_level": "Low or Medium or High",
  "framework": "one of the frameworks above",
  "selected_lenses": ["lens1", "lens2"],
  "box_count": 4,
  "required_questions": [],
  "assumptions": ["assumption 1"],
  "constraints_detected": ["constraint 1"],
  "reasoning_depth": 4,
  "is_reversible": true,
  "confidence": 0.8,
  "is_live_query": false,
  "analysis_horizon_days": 90,
  "finance_priority": "medium"
}}

Rules:
- Prefer finance-specific decision_type over "general" whenever finance language appears.
- box_count must align to stakes: Low=2, Medium=4, High=6
- reasoning_depth must align to stakes: Low=2, Medium=4, High=5
- selected_lenses must be finance-native and relevant.
- is_live_query should be true for current-state or freshest-data questions.
- analysis_horizon_days should match the implied time window when possible.
- required_questions should only appear if critical finance data is missing.
- Return ONLY valid JSON. No markdown.

Examples:
Query: "How much runway do we have right now?"
=> intent: "decision", decision_type: "liquidity_runway", is_live_query: true

Query: "Why is revenue quality getting worse?"
=> intent: "decision", decision_type: "revenue_quality"

Query: "Explain what EBITDA means"
=> intent: "chat", decision_type: "general"

Query: "Draft a board memo about cash risks"
=> intent: "action", decision_type: "board_finance"
""".strip()


def _build_user_message(query: str, context: UserContext | None) -> str:
    parts = [f"Query: {query}"]
    if context:
        if context.industry:
            parts.append(f"Industry: {context.industry}")
        if context.company_size:
            parts.append(f"Company size/stage: {context.company_size}")
        if context.risk_appetite:
            parts.append(f"Risk appetite: {context.risk_appetite}")
        for k, v in context.extra.items():
            parts.append(f"{k}: {v}")
    return "\n".join(parts)


# ─────────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────────

async def _call_api(user_message: str, company_id: str, context: UserContext | None) -> RoutingPlan:
    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()

            import json as _json

            response = await asyncio.wait_for(
                _get_client().chat.completions.create(
                    model=_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": _SYSTEM_PROMPT + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, just the JSON object."
                        },
                        {"role": "user", "content": user_message},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                ),
                timeout=20,
            )

            latency_ms = round((time.perf_counter() - t0) * 1000)
            msg = response.choices[0].message

            raw_json = (msg.content or "").strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            logger.info("Finance router raw response | chars=%d | preview=%s", len(raw_json), raw_json[:120])

            plan = RoutingPlan.model_validate(_json.loads(raw_json))

            # deterministic enforcement
            plan = _enforce_stake_rules(plan)
            plan = _enforce_framework(plan)
            plan.box_count = BOXES_BY_STAKE[plan.stake_level]
            plan.reasoning_depth = DEPTH_BY_STAKE[plan.stake_level]
            plan.confidence = _compute_confidence(plan, context)

            logger.info(
                "✅ Routed | company=%s | intent=%s | type=%s | stakes=%s | live=%s | latency_ms=%d",
                company_id,
                plan.intent,
                plan.decision_type,
                plan.stake_level,
                plan.is_live_query,
                latency_ms,
            )

            return plan

        except (APITimeoutError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if status and status not in (429, 500, 502, 503, 504):
                raise

            last_error = e
            wait = (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Transient router error | attempt=%d/%d | wait=%.1fs | %s",
                attempt,
                _MAX_RETRIES,
                wait,
                e,
            )
            await asyncio.sleep(wait)

        except Exception as e:
            last_error = e
            break

    raise RuntimeError("Finance router failed after retries") from last_error


# ─────────────────────────────────────────────
# DETERMINISTIC ROUTING OVERRIDE
# ─────────────────────────────────────────────

def _deterministic_finance_route(query: str, context: UserContext | None) -> RoutingPlan | None:
    q = query.strip()
    q_lower = q.lower()

    has_finance = _contains_finance_language(q, context)
    is_live = _detect_live_query(q, context)
    horizon = _infer_horizon_days(q)

    if not has_finance:
        return None

    # detect non-finance actions / chat
    if any(q_lower.startswith(prefix) for prefix in ("draft ", "write ", "prepare ", "create memo", "compose ")):
        intent = "action"
    elif any(x in q_lower for x in ("what is ", "define ", "explain ", "how does ") ) and not any(
        x in q_lower for x in ("should we", "risk", "upside", "what should", "analyze", "evaluate")
    ):
        intent = "chat"
    else:
        intent = "decision"

    decision_type = _infer_decision_type(q)
    stake_level = _infer_stake_level(q, decision_type, is_live)
    is_reversible = _infer_reversibility(q, decision_type)
    lenses = _default_lenses_for_type(decision_type, is_live)
    constraints = _detect_constraints(q, context)
    assumptions = _assumptions_for_plan(decision_type, context, is_live)
    required_questions = _required_questions_for_query(q, decision_type, context)
    framework = "Finance Deep Dive"
    finance_priority = _infer_finance_priority(decision_type, stake_level, is_live)

    plan = RoutingPlan(
        intent=intent,
        decision_type=decision_type,
        stake_level=stake_level,
        framework=framework,
        selected_lenses=lenses,
        box_count=BOXES_BY_STAKE[stake_level],
        required_questions=required_questions,
        assumptions=assumptions,
        constraints_detected=constraints,
        reasoning_depth=DEPTH_BY_STAKE[stake_level],
        is_reversible=is_reversible,
        confidence=0.88 if decision_type != "general" else 0.72,
        is_live_query=is_live,
        analysis_horizon_days=horizon,
        finance_priority=finance_priority,
    )

    plan = _enforce_stake_rules(plan)
    plan = _enforce_framework(plan)
    plan.box_count = BOXES_BY_STAKE[plan.stake_level]
    plan.reasoning_depth = DEPTH_BY_STAKE[plan.stake_level]
    plan.confidence = _compute_confidence(plan, context)

    return plan


# ─────────────────────────────────────────────
# PUBLIC ENTRY
# ─────────────────────────────────────────────

async def route(
    query: str,
    company_id: str,
    context: UserContext | None = None,
    *,
    bypass_cache: bool = False,
) -> RoutingPlan:
    if not query or not query.strip():
        return _safe_fallback("empty_query")

    query = query.strip()

    deterministic_plan = _deterministic_finance_route(query, context)
    if deterministic_plan is not None:
        cache_key = _cache_key(company_id, query, context)
        cache = _live_cache if deterministic_plan.is_live_query else _cache

        if not bypass_cache and cache_key in cache:
            logger.info("Cache hit | company=%s | deterministic=true | key=%s…", company_id, cache_key[:12])
            return cache[cache_key]

        logger.info(
            "🚀 Deterministic finance routing | company=%s | type=%s | live=%s",
            company_id,
            deterministic_plan.decision_type,
            deterministic_plan.is_live_query,
        )
        cache[cache_key] = deterministic_plan
        return deterministic_plan

    cache_key = _cache_key(company_id, query, context)
    if not bypass_cache and cache_key in _cache:
        logger.info("Cache hit | company=%s | key=%s…", company_id, cache_key[:12])
        return _cache[cache_key]

    logger.info("🚀 Model routing fallback | company=%s | query_preview=%s", company_id, query[:80])
    user_message = _build_user_message(query, context)

    try:
        plan = await _call_api(user_message, company_id, context)
    except (APITimeoutError, asyncio.TimeoutError):
        return _safe_fallback("timeout")
    except RuntimeError:
        return _safe_fallback("api_failure")

    _cache[cache_key] = plan
    return plan