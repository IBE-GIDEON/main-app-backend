"""
refiner.py — Three AI Decision Refiner
=======================================
Changes from v1:
  - Added BoxReasoningTrail schema — captures draft claim, attack critique,
    and what changed per box. Sent to UI layer via RefinedDecision.
  - Added ReasoningTrail to RefinedDecision — full per-box trail.
  - _assemble() now builds reasoning trails from pass data.
  - Zero changes to the 3-pass pipeline logic itself.
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
from typing import Any, Literal

from openai import AsyncOpenAI, APIStatusError, APITimeoutError
from pydantic import BaseModel, Field, field_validator
from cachetools import TTLCache

from router import UserContext, RoutingPlan, BOXES_BY_STAKE

_MODEL_STANDARD = os.getenv("THINK_AI_REFINER_MODEL_STD",  "gpt-4o-mini")
_MODEL_HIGH     = os.getenv("THINK_AI_REFINER_MODEL_HIGH", "gpt-4o")
_TIMEOUT_SEC    = float(os.getenv("THINK_AI_REFINER_TIMEOUT",  "20"))
_MAX_RETRIES    = int(os.getenv("THINK_AI_REFINER_RETRIES",    "3"))
_CACHE_TTL      = int(os.getenv("THINK_AI_CACHE_TTL",         "3600"))
_CACHE_MAXSIZE  = int(os.getenv("THINK_AI_CACHE_MAX",         "2000"))

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("three-ai.refiner")

VerdictColor = Literal["Green", "Yellow", "Orange", "Red"]

BOX_SPLIT: dict[str, dict[str, int]] = {
    "Low":    {"upside": 1, "risk": 1},
    "Medium": {"upside": 2, "risk": 1},
    "High":   {"upside": 3, "risk": 2},
}

# ─────────────────────────────────────────────
# NEW: REASONING TRAIL SCHEMA
# ─────────────────────────────────────────────

class BoxReasoningTrail(BaseModel):
    """
    The full reasoning chain for one decision box.
    Captures what changed across all 3 passes.
    Sent to the UI layer so users can see HOW the AI reached its conclusion.
    """
    box_title:          str
    box_type:           Literal["upside", "risk"]

    # Pass 1 — what the AI first thought
    draft_claim:        str   = Field(description="Raw claim from Pass 1 before any critique")
    draft_probability:  int   = Field(description="Initial probability estimate")
    draft_impact:       int   = Field(description="Initial impact estimate")
    draft_reasoning:    str   = Field(description="Initial reasoning before attack")

    # Pass 2 — what the devil's advocate said about this specific box
    attack_weakness:    str   = Field(description="What Pass 2 found wrong or overconfident")
    attack_correction:  str   = Field(description="What Pass 2 said should be said instead")

    # What changed between draft and final
    claim_changed:      bool  = Field(description="Whether the claim changed after critique")
    probability_delta:  int   = Field(description="How much probability changed: final - draft")
    impact_delta:       int   = Field(description="How much impact changed: final - draft")
    what_changed:       str   = Field(description="Plain English summary of what the attack changed")

    # Lenses applied to this specific box
    lenses_applied:     list[str] = Field(description="Which reasoning lenses shaped this box")


class ReasoningTrail(BaseModel):
    """
    Full reasoning trail for the entire decision.
    Attached to RefinedDecision and passed through to UI.
    """
    box_trails:         list[BoxReasoningTrail]

    # Company context that was injected
    company_industry:   str | None
    company_size:       str | None
    risk_appetite:      str | None
    extra_context:      dict[str, str] = Field(default_factory=dict)

    # Lenses and framework used
    lenses_used:        list[str]
    framework:          str
    decision_type:      str
    stake_level:        str

    # Confidence breakdown
    router_confidence:      float = Field(description="Router's classification confidence")
    refiner_confidence:     float = Field(description="Refiner's analysis confidence")
    combined_confidence:    float = Field(description="router × refiner = final confidence")
    confidence_explanation: str   = Field(description="Plain English why this confidence score")

    # Attack summary
    overall_attack_assessment: str = Field(description="Pass 2 overall critique summary")
    real_world_failure_modes:  list[str]
    missing_risks_found:       list[str]


# ─────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────

class DecisionBox(BaseModel):
    title:                str
    color:                VerdictColor
    claim:                str
    evidence_or_reasoning: str
    probability:          int   = Field(ge=0, le=100)
    impact:               int   = Field(ge=1, le=10)
    risk_score:           float = Field(ge=0.0, le=10.0)
    follow_up_actions:    list[str] = Field(default_factory=list, max_length=3)
    spawn_questions:      list[str] = Field(default_factory=list, max_length=3)

    @field_validator("color")
    @classmethod
    def validate_color(cls, v: str) -> str:
        return v if v in ("Green", "Yellow", "Orange", "Red") else "Yellow"

    @field_validator("risk_score", mode="before")
    @classmethod
    def clamp_risk_score(cls, v: float) -> float:
        return round(max(0.0, min(10.0, float(v))), 2)


class VerdictBox(BaseModel):
    color:          VerdictColor
    headline:       str
    rationale:      str
    net_score:      float
    go_conditions:      list[str]
    stop_conditions:    list[str]
    review_triggers:    list[str]
    key_unknown:    str
    flip_factor:    str

    @field_validator("color")
    @classmethod
    def validate_color(cls, v: str) -> str:
        return v if v in ("Green", "Yellow", "Orange", "Red") else "Yellow"


class NextStepBox(BaseModel):
    immediate_action:   str
    test_if_uncertain:  str | None = None
    owner:              str | None = None
    deadline:           str | None = None
    escalate_if:        str | None = None


class AuditTrail(BaseModel):
    query_hash:         str
    company_id:         str
    routing_plan_hash:  str
    model_used:         str
    pass_latencies_ms:  list[int]
    total_tokens_in:    int
    total_tokens_out:   int
    refiner_version:    str = Field(default="1.0.0")
    timestamp_utc:      str


class RefinedDecision(BaseModel):
    summary_box:    str
    upside_boxes:   list[DecisionBox]
    risk_boxes:     list[DecisionBox]
    verdict_box:    VerdictBox
    next_step_box:  NextStepBox
    confidence:     float = Field(ge=0.0, le=1.0)
    audit:          AuditTrail
    reasoning_trail: ReasoningTrail | None = None  # NEW — full reasoning chain

    @property
    def total_boxes(self) -> int:
        return len(self.upside_boxes) + len(self.risk_boxes)


# ─────────────────────────────────────────────
# INTERNAL PASS SCHEMAS
# ─────────────────────────────────────────────

class _DraftBox(BaseModel):
    box_type:    Literal["upside", "risk"]
    title:       str
    claim:       str
    reasoning:   str
    probability: int = Field(ge=0, le=100)
    impact:      int = Field(ge=1, le=10)


class _DraftOutput(BaseModel):
    summary:     str
    boxes:       list[_DraftBox]
    assumptions: list[str]


class _AttackReport(BaseModel):
    overall_assessment:       str
    box_critiques:            list[dict[str, str]]
    missing_risks:            list[str]
    overconfident_claims:     list[str]
    real_world_failure_modes: list[str]


class _FinalBox(BaseModel):
    box_type:               Literal["upside", "risk"]
    title:                  str
    color:                  VerdictColor
    claim:                  str
    evidence_or_reasoning:  str
    probability:            int = Field(ge=0, le=100)
    impact:                 int = Field(ge=1, le=10)
    follow_up_actions:      list[str] = Field(default_factory=list)
    spawn_questions:        list[str] = Field(default_factory=list)


class _FinalOutput(BaseModel):
    summary_box:        str
    boxes:              list[_FinalBox]
    verdict_color:      VerdictColor
    verdict_headline:   str
    verdict_rationale:  str
    go_conditions:      list[str]
    stop_conditions:    list[str]
    review_triggers:    list[str]
    key_unknown:        str
    flip_factor:        str
    immediate_action:   str
    test_if_uncertain:  str | None = None
    owner:              str | None = None
    deadline:           str | None = None
    escalate_if:        str | None = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


# ─────────────────────────────────────────────
# VERDICT COLOR ENFORCEMENT
# ─────────────────────────────────────────────

def _enforce_verdict_color(net_score: float, has_stop_conditions: bool) -> VerdictColor:
    if has_stop_conditions and net_score < 0:
        return "Red"
    if net_score >= 4.0:
        return "Green"
    if net_score >= 0.5:
        return "Yellow"
    if net_score >= -2.0:
        return "Orange"
    return "Red"


def _compute_risk_score(probability: int, impact: int, box_type: str) -> float:
    if box_type == "upside":
        return round((probability / 100) * impact, 2)
    else:
        return round(((100 - probability) / 100) * impact, 2)


# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────

_cache: TTLCache = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL)


def _cache_key(company_id: str, query: str, plan: RoutingPlan) -> str:
    raw = f"{company_id}|{query.strip().lower()}|{plan.model_dump_json()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _select_model(plan: RoutingPlan) -> str:
    return _MODEL_HIGH if plan.stake_level == "High" else _MODEL_STANDARD


# ─────────────────────────────────────────────
# PROMPT BUILDERS (unchanged)
# ─────────────────────────────────────────────

def _build_pass1_prompt(query: str, plan: RoutingPlan, context: UserContext | None) -> str:
    ctx_block = ""
    if context:
        parts = []
        if context.industry:      parts.append(f"Industry: {context.industry}")
        if context.company_size:  parts.append(f"Stage: {context.company_size}")
        if context.risk_appetite: parts.append(f"Risk appetite: {context.risk_appetite}")
        for k, v in context.extra.items(): parts.append(f"{k}: {v}")
        ctx_block = "\n".join(parts)

    split = BOX_SPLIT[plan.stake_level]

    return f"""
## Decision
{query}

## Context
{ctx_block or "No additional context provided."}

## Routing Classification
- Type: {plan.decision_type}
- Stakes: {plan.stake_level}
- Framework: {plan.framework}
- Lenses to apply: {", ".join(plan.selected_lenses)}
- Reversible: {plan.is_reversible}
- Constraints detected: {", ".join(plan.constraints_detected) or "None"}
- Router assumptions: {", ".join(plan.assumptions)}

## Your task — PASS 1: DRAFT
Generate exactly {split["upside"]} upside box(es) and {split["risk"]} risk box(es).
- Use the lenses above as your analytical frame.
- For each box: write a falsifiable claim, your reasoning, probability (0–100%), impact (1–10).
- Write a plain-language summary of the decision situation.
- List any assumptions you are making that the user did not state.
- Be specific — avoid vague claims like "could be risky."
""".strip()


def _build_pass2_prompt(query: str, draft: _DraftOutput) -> str:
    boxes_text = "\n".join(
        f"[{b.box_type.upper()}] {b.title}: {b.claim} "
        f"(prob={b.probability}%, impact={b.impact})"
        for b in draft.boxes
    )
    return f"""
## Original Decision
{query}

## Pass 1 Draft Summary
{draft.summary}

## Draft Boxes
{boxes_text}

## Draft Assumptions
{chr(10).join(draft.assumptions)}

## Your task — PASS 2: ATTACK
You are a ruthless devil's advocate. Your job is to find every weakness in Pass 1.

For each box, identify:
1. What claim is overstated or underweighted?
2. What does the real world usually do in cases like this?
3. What breaks this in practice (regulations, team capacity, market conditions)?

Also identify:
- Any critical risks that Pass 1 completely missed.
- Any claims that sound confident but have thin evidence.
- The 2–3 most likely real-world failure modes for this specific decision.

Be specific. Vague critiques are useless.
""".strip()


def _build_pass3_prompt(
    query: str, plan: RoutingPlan,
    draft: _DraftOutput, attack: _AttackReport,
    context: UserContext | None,
) -> str:
    ctx_block = ""
    if context:
        parts = []
        if context.industry:      parts.append(f"Industry: {context.industry}")
        if context.company_size:  parts.append(f"Stage: {context.company_size}")
        if context.risk_appetite: parts.append(f"Risk appetite: {context.risk_appetite}")
        ctx_block = "\n".join(parts)

    split = BOX_SPLIT[plan.stake_level]

    return f"""
## Original Decision
{query}

## Context
{ctx_block or "None"}

## Routing Plan
Type: {plan.decision_type} | Stakes: {plan.stake_level} | Reversible: {plan.is_reversible}
Lenses: {", ".join(plan.selected_lenses)}

## Pass 1 Summary
{draft.summary}

## Pass 2 Attack Findings
Overall: {attack.overall_assessment}
Failure modes: {", ".join(attack.real_world_failure_modes)}
Missing risks: {", ".join(attack.missing_risks)}
Overconfident claims: {", ".join(attack.overconfident_claims)}

## Your task — PASS 3: FINALIZE
Produce the final output incorporating all critique from Pass 2.

Rules:
1. Exactly {split["upside"]} upside box(es) and {split["risk"]} risk box(es).
2. Assign each box a color: Green (clear benefit/manageable risk), Yellow (moderate),
   Orange (uncertain/context-dependent), Red (serious concern).
3. For verdict_color: Green = benefits clearly beat risks, Yellow = balanced,
   Orange = depends on unknowns, Red = downsides clearly beat benefits.
4. go_conditions / stop_conditions / review_triggers must be specific and actionable.
5. key_unknown = the single most important thing not yet known.
6. flip_factor = what one change would reverse the verdict entirely.
7. immediate_action must be doable within 24 hours.
8. overall_confidence = your genuine certainty in this analysis (0.0–1.0).
9. spawn_questions for each box = what a user would naturally ask by clicking it.
10. Be specific — no generic advice. This decision is about: {query[:80]}
""".strip()


# ─────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────

async def _parse_call(
    system: str, user: str,
    response_format: type, model: str,
    company_id: str, pass_name: str,
) -> tuple[Any, int, int, int]:
    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()

            response = await asyncio.wait_for(
                _get_client().chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, just the JSON object."},
                        {"role": "user",   "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2 if pass_name == "attack" else 0,
                ),
                timeout=25,
            )

            latency_ms = round((time.perf_counter() - t0) * 1000)
            msg        = response.choices[0].message
            usage      = response.usage

            # Bulletproof Markdown Stripping
            raw_json = (msg.content or "").strip()
            if raw_json.startswith("```json"):
                raw_json = raw_json[7:]
            elif raw_json.startswith("```"):
                raw_json = raw_json[3:]
            if raw_json.endswith("```"):
                raw_json = raw_json[:-3]
            raw_json = raw_json.strip()

            logger.info("Pass=%s raw response | company=%s | chars=%d | preview=%s",
                        pass_name, company_id, len(raw_json), raw_json[:120])

            # Pydantic Native JSON Validation (Safer and Faster)
            parsed = response_format.model_validate_json(raw_json)

            logger.info(
                "✅ Pass=%s | company=%s | model=%s | latency_ms=%d | tokens_in=%d | tokens_out=%d",
                pass_name, company_id, model, latency_ms,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )

            return (
                parsed, latency_ms,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )

        except asyncio.TimeoutError:
            last_error = RuntimeError(f"Pass {pass_name} timed out after 25s")
            logger.warning("⏰ Timeout | pass=%s | attempt=%d/%d", pass_name, attempt, _MAX_RETRIES)

        except (APITimeoutError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if status and status not in (429, 500, 502, 503, 504):
                raise
            last_error = e
            wait = (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning("Transient error | pass=%s | attempt=%d/%d | wait=%.1fs | %s",
                           pass_name, attempt, _MAX_RETRIES, wait, e)
            await asyncio.sleep(wait)

        except Exception as e:
            last_error = e
            logger.error("Pass=%s failed | company=%s | error=%s | type=%s",
                         pass_name, company_id, e, type(e).__name__)
            break

    raise RuntimeError(f"Refiner pass '{pass_name}' failed after {_MAX_RETRIES} attempts") from last_error


# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────

def _build_pass1_system(split: dict) -> str:
    upside_count = split["upside"]
    risk_count   = split["risk"]
    upside_examples = ",\n    ".join([
        f'''{{"box_type": "upside", "title": "Upside {i+1}", "claim": "...", "reasoning": "...", "probability": 70, "impact": 7}}'''
        for i in range(upside_count)
    ])
    risk_examples = ",\n    ".join([
        f'''{{"box_type": "risk", "title": "Risk {i+1}", "claim": "...", "reasoning": "...", "probability": 40, "impact": 6}}'''
        for i in range(risk_count)
    ])
    return f"""
You are Three AI's Decision Analyst — Pass 1: DRAFT.
Produce an initial structured analysis of a business decision.
Use numbers. Be specific. Apply the reasoning lenses provided.

You MUST return ONLY this exact JSON structure:
{{
  "summary": "2-3 sentence plain-English summary of the decision",
  "boxes": [
    {upside_examples},
    {risk_examples}
  ],
  "assumptions": ["assumption 1", "assumption 2"]
}}

Rules:
- You MUST produce exactly {upside_count} upside box(es) and {risk_count} risk box(es) — {upside_count + risk_count} boxes total.
- box_type must be exactly "upside" or "risk"
- probability is 0-100 (integer), impact is 1-10 (integer)
- Return ONLY the JSON object. No explanation, no markdown.
""".strip()

_PASS2_SYSTEM = """
You are Three AI's Devil's Advocate — Pass 2: ATTACK.
Ruthlessly challenge the draft analysis. Find overconfident claims, missing risks,
survivorship bias, and real-world failure modes.

You MUST return ONLY this exact JSON structure with these exact field names:
{
  "overall_assessment": "2-3 sentence overall critique",
  "box_critiques": [
    {
      "box_title": "title of the box being critiqued",
      "weakness": "what is wrong or overconfident",
      "correction": "what should be said instead"
    }
  ],
  "missing_risks": ["risk that was overlooked"],
  "overconfident_claims": ["claim that is too confident"],
  "real_world_failure_modes": ["how this actually fails in practice"]
}

Return ONLY the JSON object. No explanation, no markdown.
""".strip()

def _build_pass3_system(split: dict) -> str:
    upside_count = split["upside"]
    risk_count   = split["risk"]
    upside_examples = ",\n    ".join([
        f'''{{"box_type": "upside", "title": "Upside {i+1}", "color": "Green", "claim": "...", "evidence_or_reasoning": "...", "probability": 70, "impact": 7, "follow_up_actions": ["action"], "spawn_questions": ["question"]}}'''
        for i in range(upside_count)
    ])
    risk_examples = ",\n    ".join([
        f'''{{"box_type": "risk", "title": "Risk {i+1}", "color": "Red", "claim": "...", "evidence_or_reasoning": "...", "probability": 40, "impact": 8, "follow_up_actions": ["action"], "spawn_questions": ["question"]}}'''
        for i in range(risk_count)
    ])
    return f"""
You are Three AI's Decision Synthesiser — Pass 3: FINALIZE.
Produce the final polished decision analysis combining the draft and the critique.
Every claim must be specific. Every action must be concrete.

You MUST return ONLY this exact JSON structure:
{{
  "summary_box": "2-3 sentence plain-English TL;DR of the decision",
  "boxes": [
    {upside_examples},
    {risk_examples}
  ],
  "verdict_color": "Yellow",
  "verdict_headline": "One decisive sentence",
  "verdict_rationale": "2-3 sentences explaining the verdict",
  "go_conditions": ["proceed if this is true"],
  "stop_conditions": ["halt if this happens"],
  "review_triggers": ["re-evaluate when this changes"],
  "key_unknown": "The single biggest unknown",
  "flip_factor": "What would change the verdict completely",
  "immediate_action": "Do this within 24 hours",
  "test_if_uncertain": "Cheapest way to validate the key unknown",
  "owner": null,
  "deadline": null,
  "escalate_if": null,
  "overall_confidence": 0.75
}}

Rules:
- You MUST produce exactly {upside_count} upside box(es) and {risk_count} risk box(es) — {upside_count + risk_count} boxes total.
- box_type must be exactly "upside" or "risk"
- color must be exactly "Green", "Yellow", "Orange", or "Red"
- verdict_color must be exactly "Green", "Yellow", "Orange", or "Red"
- probability is 0-100 (integer), impact is 1-10 (integer)
- overall_confidence is 0.0 to 1.0
- Return ONLY the JSON object. No explanation, no markdown.
""".strip()


# ─────────────────────────────────────────────
# FALLBACK
# ─────────────────────────────────────────────

def _safe_fallback(query: str, company_id: str, reason: str) -> RefinedDecision:
    logger.warning("Refiner fallback | company=%s | reason=%s", company_id, reason)

    placeholder_box = DecisionBox(
        title="Analysis unavailable", color="Orange",
        claim="The refiner could not complete analysis for this decision.",
        evidence_or_reasoning=f"Reason: {reason}",
        probability=50, impact=5, risk_score=2.5,
        follow_up_actions=["Try again", "Simplify your query"],
        spawn_questions=["What went wrong?"],
    )

    return RefinedDecision(
        summary_box=f"Analysis unavailable due to: {reason}. Please try again.",
        upside_boxes=[placeholder_box],
        risk_boxes=[placeholder_box],
        verdict_box=VerdictBox(
            color="Orange",
            headline="Unable to reach a verdict — please retry.",
            rationale="The analysis pipeline did not complete successfully.",
            net_score=0.0,
            go_conditions=[],
            stop_conditions=["Do not act without completing a full analysis"],
            review_triggers=["When the system is available again"],
            key_unknown="Could not determine",
            flip_factor="Could not determine",
        ),
        next_step_box=NextStepBox(immediate_action="Retry this query with more context."),
        confidence=0.0,
        audit=AuditTrail(
            query_hash=hashlib.sha256(query.encode()).hexdigest(),
            company_id=company_id,
            routing_plan_hash="unavailable",
            model_used="unavailable",
            pass_latencies_ms=[0, 0, 0],
            total_tokens_in=0,
            total_tokens_out=0,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        ),
        reasoning_trail=None,
    )


# ─────────────────────────────────────────────
# NEW: REASONING TRAIL BUILDER
# Builds per-box trail by matching draft boxes to
# attack critiques and final boxes.
# ─────────────────────────────────────────────

def _build_reasoning_trail(
    plan:    RoutingPlan,
    context: UserContext | None,
    draft:   _DraftOutput,
    attack:  _AttackReport,
    final:   _FinalOutput,
    refiner_confidence: float,
    router_confidence:  float,
) -> ReasoningTrail:
    """
    Builds the full reasoning trail for the UI metrics page.
    Matches draft boxes → attack critiques → final boxes by title.
    """
    box_trails: list[BoxReasoningTrail] = []

    for final_box in final.boxes:
        # Find matching draft box by title (case-insensitive partial match)
        draft_box = next(
            (d for d in draft.boxes if d.title.lower().strip() == final_box.title.lower().strip()),
            draft.boxes[0] if draft.boxes else None,
        )

        # Find matching attack critique by title
        attack_critique = next(
            (c for c in attack.box_critiques
             if c.get("box_title", "").lower().strip() == final_box.title.lower().strip()),
            None,
        )

        draft_claim       = draft_box.claim       if draft_box else "Draft not available"
        draft_probability = draft_box.probability if draft_box else final_box.probability
        draft_impact      = draft_box.impact      if draft_box else final_box.impact
        draft_reasoning   = draft_box.reasoning   if draft_box else "Draft reasoning not available"

        attack_weakness   = attack_critique.get("weakness",   "No specific weakness identified") if attack_critique else "No critique for this box"
        attack_correction = attack_critique.get("correction", "No correction suggested")         if attack_critique else "No correction suggested"

        claim_changed      = draft_claim.lower().strip() != final_box.claim.lower().strip()
        probability_delta  = final_box.probability - draft_probability
        impact_delta       = final_box.impact - draft_impact

        # Build plain English what-changed summary
        changes = []
        if claim_changed:
            changes.append("claim was rewritten after critique")
        if abs(probability_delta) >= 5:
            direction = "raised" if probability_delta > 0 else "lowered"
            changes.append(f"probability {direction} by {abs(probability_delta)}%")
        if abs(impact_delta) >= 1:
            direction = "raised" if impact_delta > 0 else "lowered"
            changes.append(f"impact {direction} by {abs(impact_delta)}")
        what_changed = (
            "After the attack pass: " + ", ".join(changes) + "."
            if changes else
            "No significant changes after the attack pass."
        )

        box_trails.append(BoxReasoningTrail(
            box_title=final_box.title,
            box_type=final_box.box_type,
            draft_claim=draft_claim,
            draft_probability=draft_probability,
            draft_impact=draft_impact,
            draft_reasoning=draft_reasoning,
            attack_weakness=attack_weakness,
            attack_correction=attack_correction,
            claim_changed=claim_changed,
            probability_delta=probability_delta,
            impact_delta=impact_delta,
            what_changed=what_changed,
            lenses_applied=plan.selected_lenses,
        ))

    # Confidence explanation
    combined = round(refiner_confidence * router_confidence, 2)
    if combined >= 0.7:
        conf_explanation = f"High confidence — router classified this clearly ({router_confidence:.0%}) and refiner found strong evidence ({refiner_confidence:.0%})."
    elif combined >= 0.4:
        conf_explanation = f"Moderate confidence — router was {router_confidence:.0%} certain on classification and refiner was {refiner_confidence:.0%} certain on analysis. Some unknowns remain."
    else:
        conf_explanation = f"Low confidence — limited context provided. Router was only {router_confidence:.0%} certain and refiner was {refiner_confidence:.0%} certain. Add more company context to improve this."

    # Extra context (filter out internal _ keys)
    extra_ctx = {}
    if context and context.extra:
        extra_ctx = {
            k: str(v) for k, v in context.extra.items()
            if not k.startswith("_") and k not in ("parent_decision", "branch_depth")
        }

    return ReasoningTrail(
        box_trails=box_trails,
        company_industry=context.industry   if context else None,
        company_size=context.company_size   if context else None,
        risk_appetite=context.risk_appetite if context else None,
        extra_context=extra_ctx,
        lenses_used=plan.selected_lenses,
        framework=plan.framework,
        decision_type=plan.decision_type,
        stake_level=plan.stake_level,
        router_confidence=router_confidence,
        refiner_confidence=refiner_confidence,
        combined_confidence=combined,
        confidence_explanation=conf_explanation,
        overall_attack_assessment=attack.overall_assessment,
        real_world_failure_modes=attack.real_world_failure_modes,
        missing_risks_found=attack.missing_risks,
    )


# ─────────────────────────────────────────────
# ASSEMBLER
# ─────────────────────────────────────────────

def _assemble(
    query:      str,
    company_id: str,
    plan:       RoutingPlan,
    context:    UserContext | None,
    draft:      _DraftOutput,
    attack:     _AttackReport,
    final:      _FinalOutput,
    model:      str,
    latencies:  list[int],
    tokens_in:  int,
    tokens_out: int,
) -> RefinedDecision:

    upside_boxes: list[DecisionBox] = []
    risk_boxes:   list[DecisionBox] = []

    for fb in final.boxes:
        rs = _compute_risk_score(fb.probability, fb.impact, fb.box_type)
        box = DecisionBox(
            title=fb.title, color=fb.color, claim=fb.claim,
            evidence_or_reasoning=fb.evidence_or_reasoning,
            probability=fb.probability, impact=fb.impact, risk_score=rs,
            follow_up_actions=fb.follow_up_actions[:3],
            spawn_questions=fb.spawn_questions[:3],
        )
        if fb.box_type == "upside":
            upside_boxes.append(box)
        else:
            risk_boxes.append(box)

    split        = BOX_SPLIT[plan.stake_level]
    upside_boxes = upside_boxes[:split["upside"]]
    risk_boxes   = risk_boxes[:split["risk"]]

    net_score      = sum(b.risk_score for b in upside_boxes) - sum(b.risk_score for b in risk_boxes)
    enforced_color = _enforce_verdict_color(net_score, bool(final.stop_conditions))

    verdict = VerdictBox(
        color=enforced_color,
        headline=final.verdict_headline,
        rationale=final.verdict_rationale,
        net_score=round(net_score, 2),
        go_conditions=final.go_conditions,
        stop_conditions=final.stop_conditions,
        review_triggers=final.review_triggers,
        key_unknown=final.key_unknown,
        flip_factor=final.flip_factor,
    )

    next_step = NextStepBox(
        immediate_action=final.immediate_action,
        test_if_uncertain=final.test_if_uncertain,
        owner=final.owner,
        deadline=final.deadline,
        escalate_if=final.escalate_if,
    )

    audit = AuditTrail(
        query_hash=hashlib.sha256(query.encode()).hexdigest(),
        company_id=company_id,
        routing_plan_hash=hashlib.sha256(plan.model_dump_json().encode()).hexdigest(),
        model_used=model,
        pass_latencies_ms=latencies,
        total_tokens_in=tokens_in,
        total_tokens_out=tokens_out,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )

    combined_confidence = round(final.overall_confidence * plan.confidence, 2)

    # Build reasoning trail
    reasoning_trail = _build_reasoning_trail(
        plan=plan, context=context,
        draft=draft, attack=attack, final=final,
        refiner_confidence=final.overall_confidence,
        router_confidence=plan.confidence,
    )

    return RefinedDecision(
        summary_box=final.summary_box,
        upside_boxes=upside_boxes,
        risk_boxes=risk_boxes,
        verdict_box=verdict,
        next_step_box=next_step,
        confidence=combined_confidence,
        audit=audit,
        reasoning_trail=reasoning_trail,
    )


# ─────────────────────────────────────────────
# CORE 3-PASS PIPELINE
# ─────────────────────────────────────────────

async def _run_three_pass(
    query:      str,
    plan:       RoutingPlan,
    company_id: str,
    context:    UserContext | None,
    model:      str,
) -> RefinedDecision:
    total_in  = 0
    total_out = 0
    latencies = []

    logger.info("🟡 Pass 1 (Draft)  | company=%s", company_id)
    draft_user = _build_pass1_prompt(query, plan, context)
    draft, lat1, ti1, to1 = await _parse_call(
        system=_build_pass1_system(BOX_SPLIT[plan.stake_level]),
        user=draft_user, response_format=_DraftOutput,
        model=model, company_id=company_id, pass_name="draft",
    )
    latencies.append(lat1); total_in += ti1; total_out += to1

    logger.info("🔴 Pass 2 (Attack) | company=%s", company_id)
    attack_user = _build_pass2_prompt(query, draft)
    attack, lat2, ti2, to2 = await _parse_call(
        system=_PASS2_SYSTEM, user=attack_user,
        response_format=_AttackReport, model=model,
        company_id=company_id, pass_name="attack",
    )
    latencies.append(lat2); total_in += ti2; total_out += to2

    logger.info("🟢 Pass 3 (Final)  | company=%s", company_id)
    final_user = _build_pass3_prompt(query, plan, draft, attack, context)
    final, lat3, ti3, to3 = await _parse_call(
        system=_build_pass3_system(BOX_SPLIT[plan.stake_level]),
        user=final_user, response_format=_FinalOutput,
        model=model, company_id=company_id, pass_name="finalize",
    )
    latencies.append(lat3); total_in += ti3; total_out += to3

    # Pass all 3 pass outputs to assembler so it can build the reasoning trail
    decision = _assemble(
        query=query, company_id=company_id,
        plan=plan, context=context,
        draft=draft, attack=attack, final=final,
        model=model, latencies=latencies,
        tokens_in=total_in, tokens_out=total_out,
    )

    logger.info(
        "✅ Refined | company=%s | verdict=%s | conf=%.2f | "
        "total_latency_ms=%d | total_tokens=%d",
        company_id, decision.verdict_box.color,
        decision.confidence, sum(latencies), total_in + total_out,
    )

    return decision


# ─────────────────────────────────────────────
# PUBLIC ENTRY
# ─────────────────────────────────────────────

async def refine(
    query:      str,
    plan:       RoutingPlan,
    company_id: str,
    context:    UserContext | None = None,
    *,
    bypass_cache: bool = False,
) -> RefinedDecision:
    if not query or not query.strip():
        return _safe_fallback(query, company_id, "empty_query")

    query     = query.strip()
    cache_key = _cache_key(company_id, query, plan)

    if not bypass_cache and cache_key in _cache:
        logger.info("Cache hit | company=%s | key=%s…", company_id, cache_key[:12])
        return _cache[cache_key]

    model = _select_model(plan)
    logger.info(
        "🚀 Refining | company=%s | model=%s | stakes=%s | depth=%d",
        company_id, model, plan.stake_level, plan.reasoning_depth,
    )

    try:
        decision = await _run_three_pass(query, plan, company_id, context, model)
    except (APITimeoutError, asyncio.TimeoutError):
        return _safe_fallback(query, company_id, "timeout")
    except RuntimeError:
        return _safe_fallback(query, company_id, "api_failure")

    _cache[cache_key] = decision
    return decision