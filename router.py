"""
router.py — Think AI Strategic Triage Router
============================================
Responsibility:  Classify an incoming decision query and emit a
                 fully-structured RoutingPlan that downstream agents
                 (Refiner, UI layer) consume without any ambiguity.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()   # ensures .env is loaded before AsyncOpenAI client is created

import asyncio
import hashlib
import logging
import os
import time
import random
from typing import Annotated, Any

from openai import AsyncOpenAI, APIStatusError, APITimeoutError
from pydantic import BaseModel, Field, field_validator
from cachetools import TTLCache

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

_MODEL          = os.getenv("THINK_AI_ROUTER_MODEL", "gpt-4o-mini")
_TIMEOUT_SEC    = float(os.getenv("THINK_AI_ROUTER_TIMEOUT", "15"))
_MAX_RETRIES    = int(os.getenv("THINK_AI_ROUTER_RETRIES", "3"))
_CACHE_TTL      = int(os.getenv("THINK_AI_CACHE_TTL", "3600"))
_CACHE_MAXSIZE  = int(os.getenv("THINK_AI_CACHE_MAX", "2000"))

# Client is created lazily on first use — guarantees .env is loaded first
_client: AsyncOpenAI | None = None

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        load_dotenv()  # reload in case subprocess missed it
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
logger = logging.getLogger("think-ai.router")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

DECISION_TYPES = [
    "Strategy",
    "Hiring",
    "Finance",
    "Pricing",
    "Ops",
    "Legal",
    "Reputation",
    "Product",
    "General",
]

STAKE_LEVELS = ["Low", "Medium", "High"]

AVAILABLE_LENSES = [
    "Reversibility",
    "Expected Value",
    "Opportunity Cost",
    "Ripple Effects",
    "Pre-Mortem",
    "Resource Reality",
    "Asymmetric Risk",
]

FRAMEWORKS = ["Good/Bad/Ugly", "Dynamic Lenses", "Quick Decision"]

BOXES_BY_STAKE: dict[str, int] = {
    "Low": 2,
    "Medium": 3,
    "High": 5,
}

DEPTH_BY_STAKE = {
    "Low": 2,
    "Medium": 3,
    "High": 5,
}

# ─────────────────────────────────────────────
# DETERMINISTIC ENFORCEMENT LAYER
# ─────────────────────────────────────────────

def _enforce_stake_rules(plan: "RoutingPlan") -> "RoutingPlan":
    if plan.decision_type == "Legal":
        plan.stake_level = "High"

    if plan.decision_type == "Reputation" and plan.stake_level == "Low":
        plan.stake_level = "Medium"

    if not plan.is_reversible and plan.stake_level == "Low":
        plan.stake_level = "Medium"

    return plan


def _enforce_framework(plan: "RoutingPlan") -> "RoutingPlan":
    if plan.stake_level == "High":
        plan.framework = "Good/Bad/Ugly"
    elif plan.stake_level == "Low":
        plan.framework = "Quick Decision"
    else:
        plan.framework = "Dynamic Lenses"
    return plan


def _compute_confidence(plan: "RoutingPlan", context: "UserContext | None") -> float:
    score = 1.0

    if plan.required_questions:
        score *= 0.6

    if plan.decision_type == "General":
        score *= 0.75

    if context is None:
        score *= 0.85

    return round(max(0.0, min(score, 1.0)), 2)

# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────

class UserContext(BaseModel):
    industry:       str | None = None
    company_size:   str | None = None
    risk_appetite:  str | None = None
    extra:          dict[str, Any] = Field(default_factory=dict)


class RoutingPlan(BaseModel):

    decision_type: str
    stake_level: str
    framework: str
    selected_lenses: list[str] = Field(min_length=1, max_length=5)
    box_count: int = Field(ge=1, le=5)
    required_questions: list[str] = Field(max_length=5)
    assumptions: list[str]
    constraints_detected: list[str]
    reasoning_depth: int = Field(ge=1, le=5)
    is_reversible: bool
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("decision_type")
    @classmethod
    def validate_decision_type(cls, v: str) -> str:
        return v if v in DECISION_TYPES else "General"

    @field_validator("stake_level")
    @classmethod
    def validate_stake_level(cls, v: str) -> str:
        return v if v in STAKE_LEVELS else "Medium"

    @field_validator("framework")
    @classmethod
    def validate_framework(cls, v: str) -> str:
        return v if v in FRAMEWORKS else "Dynamic Lenses"

    @field_validator("selected_lenses")
    @classmethod
    def validate_lenses(cls, v: list[str]) -> list[str]:
        valid = [l for l in v if l in AVAILABLE_LENSES]
        return valid[:5] or ["Expected Value"]

    @field_validator("box_count", mode="before")
    @classmethod
    def clamp_box_count(cls, v: int) -> int:
        return max(1, min(5, int(v)))

# ─────────────────────────────────────────────
# FALLBACK
# ─────────────────────────────────────────────

def _safe_fallback(reason: str) -> RoutingPlan:
    logger.warning("Router fallback | reason=%s", reason)
    return RoutingPlan(
        decision_type="General",
        stake_level="Medium",
        framework="Dynamic Lenses",
        selected_lenses=["Expected Value", "Reversibility"],
        box_count=3,
        required_questions=["Could you describe the decision in more detail?"],
        assumptions=["Insufficient context — using conservative defaults"],
        constraints_detected=[],
        reasoning_depth=3,
        is_reversible=True,
        confidence=0.3,
    )

# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────

_cache: TTLCache = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL)

def _cache_key(company_id: str, query: str, context: UserContext | None) -> str:
    raw = f"{company_id}|{query.strip().lower()}"
    if context:
        raw += f"|{context.model_dump_json(exclude_none=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()

# ─────────────────────────────────────────────
# RESTORED: PROMPT BUILDER
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = f"""
You are the Strategic Triage Agent for Think AI.
Your ONLY job is to classify a business decision and return JSON metadata.
You NEVER answer or advise on the decision itself.

DECISION TYPES (pick one): {", ".join(DECISION_TYPES)}
STAKE LEVELS: High (>$10k impact, irreversible, legal, Personal), Medium (moderate), Low (easily undone)
FRAMEWORKS: Good/Bad/Ugly (High stakes), Dynamic Lenses (Medium), Quick Decision (Low)
LENSES (pick 1-5): {", ".join(AVAILABLE_LENSES)}

You MUST return ONLY this exact JSON structure with these exact field names:
{{
  "decision_type": "one of the decision types above",
  "stake_level": "Low or Medium or High",
  "framework": "one of the three frameworks above",
  "selected_lenses": ["lens1", "lens2"],
  "box_count": 3,
  "required_questions": [],
  "assumptions": ["what you assumed"],
  "constraints_detected": ["any constraints spotted"],
  "reasoning_depth": 3,
  "is_reversible": true,
  "confidence": 0.8
}}

Rules:
- box_count must match stake_level: Low=2, Medium=3, High=5
- reasoning_depth must match stake_level: Low=2, Medium=3, High=5
- required_questions: only ask if critical info is truly missing (max 5)
- confidence: 0.0 to 1.0 — how certain you are in this classification
- Return ONLY the JSON object. No explanation, no markdown, no extra text.
""".strip()


def _build_user_message(query: str, context: UserContext | None) -> str:
    parts = [f"Decision: {query}"]
    if context:
        if context.industry:
            parts.append(f"Industry: {context.industry}")
        if context.company_size:
            parts.append(f"Stage: {context.company_size}")
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
                        {"role": "system", "content": _SYSTEM_PROMPT + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, just the JSON object."},
                        {"role": "user",   "content": user_message},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                ),
                timeout=20,
            )

            latency_ms = round((time.perf_counter() - t0) * 1000)
            msg        = response.choices[0].message
            usage      = response.usage

            raw_json = (msg.content or "").strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            logger.info("Router raw response | chars=%d | preview=%s", len(raw_json), raw_json[:100])
            plan = RoutingPlan.model_validate(_json.loads(raw_json))

            # Deterministic enforcement
            plan = _enforce_stake_rules(plan)
            plan = _enforce_framework(plan)
            plan.box_count       = BOXES_BY_STAKE[plan.stake_level]
            plan.reasoning_depth = DEPTH_BY_STAKE[plan.stake_level]
            plan.confidence      = _compute_confidence(plan, context)

            # RESTORED: observability
            logger.info(
                "✅ Routed | company=%s | type=%s | stakes=%s | lenses=%s | "
                "depth=%d | conf=%.2f | latency_ms=%d | tokens_in=%d | tokens_out=%d",
                company_id,
                plan.decision_type,
                plan.stake_level,
                plan.selected_lenses,
                plan.reasoning_depth,
                plan.confidence,
                latency_ms,
                usage.prompt_tokens     if usage else 0,
                usage.completion_tokens if usage else 0,
            )

            return plan

        except (APITimeoutError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if status and status not in (429, 500, 502, 503, 504):
                raise

            last_error = e
            wait = (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Transient error | attempt=%d/%d | wait=%.1fs | %s",
                attempt, _MAX_RETRIES, wait, e,
            )
            await asyncio.sleep(wait)

        except Exception as e:
            last_error = e
            break

    raise RuntimeError("Router failed after retries") from last_error

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
    cache_key = _cache_key(company_id, query, context)

    if not bypass_cache and cache_key in _cache:
        logger.info("Cache hit | company=%s | key=%s…", company_id, cache_key[:12])
        return _cache[cache_key]

    logger.info("🚀 Routing | company=%s | query_preview=%s", company_id, query[:60])

    # RESTORED: context is now injected into the message
    user_message = _build_user_message(query, context)

    try:
        plan = await _call_api(user_message, company_id, context)
    except (APITimeoutError, asyncio.TimeoutError):   # FIXED: narrow catch
        return _safe_fallback("timeout")
    except RuntimeError:                               # FIXED: only catches router failure
        return _safe_fallback("api_failure")
    # everything else bubbles up so real bugs are visible

    _cache[cache_key] = plan
    return plan


# ─────────────────────────────────────────────
# QUICK LOCAL TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json

    async def _demo():
        ctx = UserContext(
            industry="E-commerce",
            company_size="Series A",
            risk_appetite="Moderate",
        )
        plan = await route(
            query="Should we raise our prices by 20% next month?",
            company_id="demo-co",
            context=ctx,
        )
        print(json.dumps(plan.model_dump(), indent=2))

    asyncio.run(_demo())