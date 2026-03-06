"""
feedback.py — Think AI Rating & Learning Loop
===============================================
Responsibility:  Collect user ratings on decisions, store them with
                 optional written comments, and run a learning loop
                 that continuously improves what the Router and Refiner
                 produce for this company over time.

The problem this solves
-----------------------
Without feedback.py, ratings are stored in memory.py's preference_signals
as raw lists of low-rated types and verdict colours. That's useful but blunt.

feedback.py goes further:
  1. Stores the full feedback record — rating, comment, what specifically
     was wrong (wrong_aspects), which box the user disliked, timestamp.
  2. Analyses patterns across all low-rated decisions using a lightweight
     LLM call to identify WHY they failed — not just WHAT was rated badly.
  3. Writes those patterns back into memory.preference_signals as structured
     learning signals — so enrich_context() in memory.py injects them
     into every future Router and Refiner call automatically.
  4. Exposes an insights endpoint so the frontend can show the company
     "Here's what Think AI has learned about your preferences."

How the learning loop works
-----------------------------
  User rates decision ≤ 2 stars
        │
        ▼
  write_feedback()         → stores FeedbackRecord
        │
        ▼
  _trigger_learning()      → if ≥ 3 low-rated decisions exist, runs analysis
        │
        ▼
  _run_learning_analysis() → lightweight LLM call reads low-rated audit
                             records and identifies failure patterns
        │
        ▼
  _write_learning_signals()→ updates memory.preference_signals with:
                             - avoided_lenses (lenses that keep failing)
                             - preferred_frameworks
                             - depth_preference (go deeper / stay quick)
                             - common_failure_modes (plain English)
                             - learning_summary (one sentence)
        │
        ▼
  enrich_context() in memory.py picks up these signals
  and injects them into UserContext.extra on every future call
        │
        ▼
  Router and Refiner receive richer context → better output → higher ratings

How it connects (zero edits to existing files)
-----------------------------------------------
  main.py calls write_feedback() from the POST /feedback endpoint.
  feedback.py calls memory.update_rating() and memory._read_doc() internally.
  feedback.py calls audit.list_audit_records() to fetch low-rated records.
  feedback.py calls audit.get_audit_record() to fetch full record detail.
  feedback.py writes learning signals directly to memory via update_preference_signals().
  router.py and refiner.py are untouched — they receive richer UserContext.

Storage
-------
  Default    : JSON files in feedback_store/ (same pattern as memory/conditions/audit)
  Production : Swap _read_store/_write_store for Postgres/Redis — rest unchanged.

Security
--------
  * company_id SHA-256 hashed in all file paths — no PII on disk
  * comment text sanitised — length capped, injection patterns blocked
  * wrong_aspects validated against a known allowlist
  * LLM learning analysis is read-only — never modifies an AuditRecord
  * Per-company asyncio write locks on all writes
  * Narrow exception catches throughout

Usage (from main.py)
---------------------
    from feedback import write_feedback, get_feedback_insights

    # POST /feedback
    result = await write_feedback(
        company_id  = "acme-123",
        decision_id = "abc123",
        record_id   = "def456",
        rating      = 2,
        comment     = "The risk analysis felt too generic.",
        wrong_aspects = ["risk_analysis", "lenses"],
    )

    # GET /feedback/{company_id}/insights
    insights = await get_feedback_insights("acme-123")
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()   # ensures .env is loaded before AsyncOpenAI client is created

import asyncio
import hashlib
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, APIStatusError, APITimeoutError
from pydantic import BaseModel, Field, field_validator

# ── Import shared contracts — zero re-definition ──────────────────────────
from router import UserContext
from audit  import list_audit_records, get_audit_record, AuditRecord

# memory functions we call directly
from memory import update_rating, _read_doc, _write_doc, _get_lock as _mem_lock
from memory import _company_hash as _mem_hash

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

_STORE_DIR           = Path(os.getenv("THINK_AI_FEEDBACK_DIR",      "feedback_store"))
_MAX_RECORDS         = int(os.getenv("THINK_AI_FEEDBACK_MAX",        "1000"))
_LOCK_TIMEOUT_SEC    = float(os.getenv("THINK_AI_FEEDBACK_LOCK",     "5"))
_MAX_COMMENT_LEN     = int(os.getenv("THINK_AI_FEEDBACK_MAX_COMMENT","500"))
_LEARNING_THRESHOLD  = int(os.getenv("THINK_AI_FEEDBACK_THRESHOLD",  "3"))   # low-rated before learning triggers
_LEARNING_MODEL      = os.getenv("THINK_AI_FEEDBACK_MODEL",          "gpt-4o-mini")
_LEARNING_TIMEOUT    = float(os.getenv("THINK_AI_FEEDBACK_TIMEOUT",  "30"))
_MAX_RETRIES         = int(os.getenv("THINK_AI_REFINER_RETRIES",     "3"))
_FEEDBACK_VERSION    = "1.0.0"

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

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("think-ai.feedback")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Allowlist of aspects a user can flag as wrong.
# Validated on input — prevents arbitrary strings entering the learning loop.
VALID_ASPECTS = {
    "verdict",           # overall verdict was wrong
    "risk_analysis",     # risk boxes were off
    "upside_analysis",   # upside boxes were off
    "lenses",            # wrong reasoning lenses chosen
    "confidence",        # confidence score felt wrong
    "conditions",        # go/stop/review conditions were unhelpful
    "next_steps",        # recommended actions were vague or wrong
    "stake_level",       # decision was over or under-weighted
    "framework",         # wrong framework chosen
    "depth",             # analysis too shallow or too deep
    "other",             # catch-all
}

# Ratings below this threshold trigger the learning loop
LOW_RATING_THRESHOLD = 3

# How many low-rated records to feed into the learning analysis
LEARNING_SAMPLE_SIZE = 10


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class FeedbackRecord(BaseModel):
    """
    One user rating on one decision version.
    Stored immutably — never overwritten, only appended.
    """
    feedback_id:     str           # SHA-256 of company_id + record_id + timestamp
    company_id_hash: str           # SHA-256 of company_id
    decision_id:     str           # stable across versions
    record_id:       str           # the specific version being rated
    rating:          int           # 1–5
    comment:         str | None    # sanitised free text, max 500 chars
    wrong_aspects:   list[str]     # validated against VALID_ASPECTS
    created_utc:     str
    triggered_learning: bool = False   # True if this rating triggered the learning loop

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: int) -> int:
        if v not in range(1, 6):
            raise ValueError("Rating must be 1–5")
        return v

    @field_validator("wrong_aspects")
    @classmethod
    def validate_aspects(cls, v: list[str]) -> list[str]:
        return [a for a in v if a in VALID_ASPECTS]


class CompanyFeedbackStore(BaseModel):
    """All feedback records for one company."""
    company_id_hash:  str
    version:          str = _FEEDBACK_VERSION
    records:          list[FeedbackRecord] = Field(default_factory=list)
    last_updated_utc: str


class LearningSignals(BaseModel):
    """
    Structured signals written to memory.preference_signals
    after the learning loop runs.
    These are injected into UserContext.extra by enrich_context()
    so every future Router and Refiner call benefits automatically.
    """
    avoided_lenses:       list[str]   # lenses that correlate with low ratings
    preferred_lenses:     list[str]   # lenses that correlate with high ratings
    preferred_framework:  str | None  # framework that gets highest ratings
    depth_preference:     str | None  # "deeper" / "quicker" / "balanced"
    common_failure_modes: list[str]   # plain English — what keeps going wrong
    common_success_modes: list[str]   # plain English — what this company values
    learning_summary:     str         # one sentence — injected into Router prompt
    last_learned_utc:     str
    low_rated_count:      int
    high_rated_count:     int


class FeedbackInsights(BaseModel):
    """
    Client-facing insights report — what has been learned so far.
    Safe to display on a company's settings or analytics page.
    """
    company_id:           str
    total_ratings:        int
    avg_rating:           float
    low_rated_count:      int       # ratings ≤ 2
    high_rated_count:     int       # ratings ≥ 4
    most_flagged_aspects: list[str]
    learning_signals:     LearningSignals | None
    rating_trend:         str       # "improving" / "declining" / "stable" / "insufficient_data"
    generated_utc:        str


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


async def _read_store(company_id: str) -> CompanyFeedbackStore | None:
    path = _store_path(company_id)
    if not path.exists():
        return None
    try:
        raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return CompanyFeedbackStore.model_validate_json(raw)
    except Exception as e:
        logger.error(
            "Feedback read error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return None


async def _write_store(company_id: str, store: CompanyFeedbackStore) -> bool:
    _ensure_store_dir()
    path = _store_path(company_id)
    if len(store.records) > _MAX_RECORDS:
        store.records = store.records[-_MAX_RECORDS:]
    try:
        await asyncio.to_thread(
            path.write_text,
            store.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return True
    except Exception as e:
        logger.error(
            "Feedback write error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return False


def _new_store(company_id: str) -> CompanyFeedbackStore:
    return CompanyFeedbackStore(
        company_id_hash=_company_hash(company_id),
        records=[],
        last_updated_utc=datetime.now(timezone.utc).isoformat(),
    )


# ─────────────────────────────────────────────
# INPUT SANITISATION
# ─────────────────────────────────────────────

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
    lower   = cleaned.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lower:
            logger.warning("Injection pattern in feedback comment — stripped")
            return None
    return cleaned if cleaned else None


# ─────────────────────────────────────────────
# LEARNING ANALYSIS — LLM CALL
# ─────────────────────────────────────────────

_LEARNING_SYSTEM = """
You are Think AI's Learning Analyst.
You are given a set of low-rated decision analyses from a specific company.
Your job is to identify patterns in what went wrong and what this company values.

Be specific. "Analysis was too generic" is not useful.
"The risk boxes underweighted regulatory risk for a fintech company" is useful.

Return valid JSON only.
""".strip()


class _LearningOutput(BaseModel):
    avoided_lenses:       list[str] = Field(
        description="Lenses that appear in low-rated decisions — the company finds them unhelpful"
    )
    preferred_lenses:     list[str] = Field(
        description="Lenses the company would benefit from more"
    )
    preferred_framework:  str | None = Field(
        description="Framework that would suit this company best"
    )
    depth_preference:     str | None = Field(
        description="'deeper' if analyses are consistently too shallow, "
                    "'quicker' if too slow/verbose, 'balanced' if fine"
    )
    common_failure_modes: list[str] = Field(
        description="Specific, concrete things that went wrong across these decisions"
    )
    common_success_modes: list[str] = Field(
        description="What this company tends to value in a good analysis"
    )
    learning_summary:     str = Field(
        description="One sentence that can be injected into future Router prompts "
                    "to improve output for this company"
    )


async def _run_learning_analysis(
    company_id: str,
    low_rated_records: list[AuditRecord],
    high_rated_records: list[AuditRecord],
    aspects_flagged: list[str],
) -> _LearningOutput | None:
    """
    Lightweight LLM call — one pass, no 3-step loop needed.
    Reads low-rated audit records and identifies failure patterns.
    """
    if not low_rated_records:
        return None

    def _summarise_record(r: AuditRecord, label: str) -> str:
        v = r.verdict_snapshot
        s = r.routing_snapshot
        boxes = ", ".join(f"{b.box_type}:{b.title}" for b in r.boxes)
        return (
            f"[{label}] Query: {r.query_preview}\n"
            f"  Type={s.decision_type} Stake={s.stake_level} "
            f"Framework={s.framework} Lenses={', '.join(s.selected_lenses)}\n"
            f"  Verdict={v.color} NetScore={v.net_score:+.1f}\n"
            f"  Boxes: {boxes}\n"
            f"  Key unknown: {v.key_unknown}"
        )

    low_block  = "\n\n".join(
        _summarise_record(r, f"LOW-RATED #{i+1}")
        for i, r in enumerate(low_rated_records[:LEARNING_SAMPLE_SIZE])
    )
    high_block = "\n\n".join(
        _summarise_record(r, f"HIGH-RATED #{i+1}")
        for i, r in enumerate(high_rated_records[:5])
    ) or "No high-rated decisions yet."

    aspects_str = ", ".join(aspects_flagged) if aspects_flagged else "Not specified"

    user_message = f"""
Company: {_company_hash(company_id)[:12]} (anonymised)
Aspects flagged as wrong by users: {aspects_str}

LOW-RATED DECISIONS (rated ≤ 2 stars):
{low_block}

HIGH-RATED DECISIONS (rated ≥ 4 stars, for contrast):
{high_block}

What patterns explain the low ratings? What should future analyses do differently?
""".strip()

    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = await _get_client().chat.completions.create(
                model=_LEARNING_MODEL,
                messages=[
                    {"role": "system", "content": _LEARNING_SYSTEM + "\n\nRespond with valid JSON only. No markdown."},
                    {"role": "user",   "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            msg = response.choices[0].message
            usage = response.usage
            import json as _json
            raw = (msg.content or "").strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = _LearningOutput.model_validate(_json.loads(raw))

            logger.info(
                "✅ Learning analysis done | company_hash=%s | "
                "tokens_in=%d | tokens_out=%d",
                _company_hash(company_id)[:12],
                usage.prompt_tokens     if usage else 0,
                usage.completion_tokens if usage else 0,
            )
            return parsed

        except (APITimeoutError, APIStatusError) as e:
            status_code = getattr(e, "status_code", None)
            if status_code and status_code not in (429, 500, 502, 503, 504):
                raise
            last_error = e
            wait = (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Learning analysis transient error | attempt=%d/%d | %.1fs | %s",
                attempt, _MAX_RETRIES, wait, e,
            )
            await asyncio.sleep(wait)

        except Exception as e:
            last_error = e
            break

    logger.error("Learning analysis failed | company_hash=%s | %s",
                 _company_hash(company_id)[:12], last_error)
    return None


# ─────────────────────────────────────────────
# WRITE LEARNING SIGNALS INTO MEMORY
# ─────────────────────────────────────────────

async def _write_learning_signals(
    company_id: str,
    signals:    LearningSignals,
) -> bool:
    """
    Writes the learning signals directly into memory.preference_signals.
    enrich_context() in memory.py picks these up automatically on every
    future call — no changes needed to router.py or refiner.py.
    """
    # Use memory's own lock to safely update the document
    lock = _mem_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning(
            "Learning signals write lock timeout | company_hash=%s",
            _company_hash(company_id)[:12],
        )
        return False

    try:
        doc = await _read_doc(company_id)
        if not doc:
            logger.warning(
                "No memory doc found — learning signals not written | company_hash=%s",
                _company_hash(company_id)[:12],
            )
            return False

        # Write structured signals into preference_signals
        # These are read by enrich_context() and injected into UserContext.extra
        doc.preference_signals["learning_signals"]     = signals.model_dump()
        doc.preference_signals["avoided_lenses"]       = signals.avoided_lenses
        doc.preference_signals["preferred_lenses"]     = signals.preferred_lenses
        doc.preference_signals["depth_preference"]     = signals.depth_preference
        doc.preference_signals["common_failure_modes"] = signals.common_failure_modes
        doc.preference_signals["learning_summary"]     = signals.learning_summary
        doc.last_updated_utc = datetime.now(timezone.utc).isoformat()

        success = await _write_doc(company_id, doc)
        if success:
            logger.info(
                "✅ Learning signals written to memory | company_hash=%s | summary=%s",
                _company_hash(company_id)[:12],
                signals.learning_summary[:60],
            )
        return success

    except Exception as e:
        logger.error(
            "_write_learning_signals error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return False
    finally:
        lock.release()


# ─────────────────────────────────────────────
# LEARNING TRIGGER
# ─────────────────────────────────────────────

async def _trigger_learning(company_id: str, aspects_flagged: list[str]) -> None:
    """
    Called after a low rating is written.
    If enough low-rated decisions have accumulated, run the learning analysis
    and write the results back into memory.

    This is fire-and-forget — called via asyncio.create_task().
    Never raises. Never blocks the rating response.
    """
    try:
        store = await _read_store(company_id)
        if not store:
            return

        # Count low-rated records
        low_rated  = [r for r in store.records if r.rating <= LOW_RATING_THRESHOLD]
        high_rated = [r for r in store.records if r.rating >= 4]

        if len(low_rated) < _LEARNING_THRESHOLD:
            logger.info(
                "Learning not triggered yet | company_hash=%s | "
                "low_rated=%d / threshold=%d",
                _company_hash(company_id)[:12],
                len(low_rated), _LEARNING_THRESHOLD,
            )
            return

        logger.info(
            "🧠 Learning triggered | company_hash=%s | low_rated=%d",
            _company_hash(company_id)[:12], len(low_rated),
        )

        # Fetch full audit records for low-rated decisions
        low_audit_records: list[AuditRecord] = []
        for fb in sorted(low_rated, key=lambda r: r.created_utc, reverse=True)[:LEARNING_SAMPLE_SIZE]:
            rec = await get_audit_record(company_id, fb.record_id)
            if rec:
                low_audit_records.append(rec)

        # Fetch full audit records for high-rated decisions (for contrast)
        high_audit_records: list[AuditRecord] = []
        for fb in sorted(high_rated, key=lambda r: r.created_utc, reverse=True)[:5]:
            rec = await get_audit_record(company_id, fb.record_id)
            if rec:
                high_audit_records.append(rec)

        if not low_audit_records:
            logger.warning(
                "Low-rated audit records not found | company_hash=%s",
                _company_hash(company_id)[:12],
            )
            return

        # Run LLM analysis
        output = await _run_learning_analysis(
            company_id=company_id,
            low_rated_records=low_audit_records,
            high_rated_records=high_audit_records,
            aspects_flagged=aspects_flagged,
        )

        if not output:
            return

        # Build structured learning signals
        signals = LearningSignals(
            avoided_lenses=output.avoided_lenses,
            preferred_lenses=output.preferred_lenses,
            preferred_framework=output.preferred_framework,
            depth_preference=output.depth_preference,
            common_failure_modes=output.common_failure_modes,
            common_success_modes=output.common_success_modes,
            learning_summary=output.learning_summary,
            last_learned_utc=datetime.now(timezone.utc).isoformat(),
            low_rated_count=len(low_rated),
            high_rated_count=len(high_rated),
        )

        # Write signals into memory — enrich_context() picks them up automatically
        await _write_learning_signals(company_id, signals)

        # Mark the records that triggered learning
        lock = _get_lock(company_id)
        try:
            async with asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC):
                pass
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
        logger.error(
            "_trigger_learning error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )


# ─────────────────────────────────────────────
# TREND CALCULATION
# ─────────────────────────────────────────────

def _calculate_trend(records: list[FeedbackRecord]) -> str:
    """
    Compares avg rating of the last 5 records vs the 5 before that.
    Returns 'improving', 'declining', 'stable', or 'insufficient_data'.
    """
    if len(records) < 4:
        return "insufficient_data"
    sorted_recs = sorted(records, key=lambda r: r.created_utc)
    recent = sorted_recs[-5:]
    prior  = sorted_recs[-10:-5] if len(sorted_recs) >= 10 else sorted_recs[:-5]
    if not prior:
        return "insufficient_data"
    avg_recent = sum(r.rating for r in recent)  / len(recent)
    avg_prior  = sum(r.rating for r in prior)   / len(prior)
    delta = avg_recent - avg_prior
    if delta >  0.3:
        return "improving"
    if delta < -0.3:
        return "declining"
    return "stable"


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

async def write_feedback(
    company_id:    str,
    decision_id:   str,
    record_id:     str,
    rating:        int,
    comment:       str | None       = None,
    wrong_aspects: list[str] | None = None,
) -> bool:
    """
    Main entry point. Called by main.py POST /feedback.
    Stores the feedback record, updates the rating in memory,
    and triggers the learning loop for low ratings.

    Parameters
    ----------
    company_id    : Tenant identifier.
    decision_id   : Stable decision ID (from AuditListItem.decision_id).
    record_id     : Specific version record ID (from AuditRecord.record_id).
    rating        : 1–5 star rating.
    comment       : Optional written feedback (sanitised, max 500 chars).
    wrong_aspects : List of aspects that were wrong (validated against VALID_ASPECTS).

    Returns
    -------
    bool — True if written successfully.
    """
    if rating not in range(1, 6):
        logger.warning("Invalid rating %d | company_hash=%s",
                       rating, _company_hash(company_id)[:12])
        return False

    # Sanitise inputs
    comment       = _sanitise_comment(comment)
    wrong_aspects = [a for a in (wrong_aspects or []) if a in VALID_ASPECTS]

    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning(
            "Feedback write lock timeout | company_hash=%s",
            _company_hash(company_id)[:12],
        )
        return False

    try:
        t0    = time.perf_counter()
        store = await _read_store(company_id) or _new_store(company_id)
        now   = datetime.now(timezone.utc).isoformat()

        feedback_id = hashlib.sha256(
            f"{company_id}:{record_id}:{now}".encode()
        ).hexdigest()[:16]

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
        latency_ms = round((time.perf_counter() - t0) * 1000)

        if success:
            logger.info(
                "✅ Feedback written | company_hash=%s | decision=%s | "
                "rating=%d | aspects=%s | latency_ms=%d",
                _company_hash(company_id)[:12],
                decision_id[:12], rating,
                wrong_aspects, latency_ms,
            )

        # ── Update memory rating (existing memory.py hook) ─────────────────
        # Non-fatal — feedback is stored regardless of memory update result
        asyncio.create_task(
            update_rating(company_id, decision_id, rating)
        )

        # ── Trigger learning loop for low ratings ──────────────────────────
        if rating <= LOW_RATING_THRESHOLD:
            asyncio.create_task(
                _trigger_learning(company_id, wrong_aspects)
            )

        return success

    except Exception as e:
        logger.error(
            "write_feedback error | company_hash=%s | %s",
            _company_hash(company_id)[:12], e,
        )
        return False
    finally:
        lock.release()


async def get_feedback_insights(company_id: str) -> FeedbackInsights:
    """
    Returns a client-facing insights report — what has been learned
    about this company's preferences from their ratings.
    Safe to display on a settings or analytics page.
    Never exposes raw decision content — only aggregated signals.

    Returns
    -------
    FeedbackInsights — always. Empty report if no feedback yet.
    """
    store = await _read_store(company_id)
    now   = datetime.now(timezone.utc).isoformat()

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

    records    = store.records
    total      = len(records)
    avg_rating = round(sum(r.rating for r in records) / total, 2)
    low_rated  = [r for r in records if r.rating <= LOW_RATING_THRESHOLD]
    high_rated = [r for r in records if r.rating >= 4]

    # Most flagged aspects
    aspect_counts: dict[str, int] = {}
    for r in records:
        for a in r.wrong_aspects:
            aspect_counts[a] = aspect_counts.get(a, 0) + 1
    most_flagged = sorted(aspect_counts, key=aspect_counts.get, reverse=True)[:5]

    # Read learning signals from memory
    learning_signals: LearningSignals | None = None
    try:
        doc = await _read_doc(company_id)
        if doc and "learning_signals" in doc.preference_signals:
            learning_signals = LearningSignals.model_validate(
                doc.preference_signals["learning_signals"]
            )
    except Exception as e:
        logger.warning(
            "Could not read learning signals from memory | %s", e
        )

    trend = _calculate_trend(records)

    logger.info(
        "Feedback insights | company_hash=%s | total=%d | avg=%.2f | trend=%s",
        _company_hash(company_id)[:12], total, avg_rating, trend,
    )

    return FeedbackInsights(
        company_id=company_id,
        total_ratings=total,
        avg_rating=avg_rating,
        low_rated_count=len(low_rated),
        high_rated_count=len(high_rated),
        most_flagged_aspects=most_flagged,
        learning_signals=learning_signals,
        rating_trend=trend,
        generated_utc=now,
    )


async def get_feedback_for_decision(
    company_id:  str,
    decision_id: str,
) -> list[FeedbackRecord]:
    """
    Returns all feedback records for one specific decision (all versions).
    Used by the audit detail page to show ratings alongside the record.
    """
    store = await _read_store(company_id)
    if not store:
        return []
    return [r for r in store.records if r.decision_id == decision_id]


async def clear_feedback(company_id: str) -> bool:
    """
    Wipes all feedback for this company.
    Called alongside clear_memory() / clear_audit() for GDPR erasure.
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
                "🗑️  Feedback cleared | company_hash=%s",
                _company_hash(company_id)[:12],
            )
            return True
        return False
    except Exception as e:
        logger.error("clear_feedback error | %s", e)
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
    from audit  import write_audit

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

        print("\n── Step 2: Write audit record ──────────────────")
        record_id = await write_audit(
            company_id=company_id,
            query=query,
            plan=plan,
            decision=decision,
            payload=payload,
        )
        print(f"  record_id: {record_id}")

        print("\n── Step 3: Submit low rating (triggers learning)─")
        from audit import list_audit_records
        items = await list_audit_records(company_id, limit=1)
        if items and record_id:
            ok = await write_feedback(
                company_id=company_id,
                decision_id=items[0].decision_id,
                record_id=record_id,
                rating=2,
                comment="The risk analysis felt too generic. Missed regulatory risk.",
                wrong_aspects=["risk_analysis", "lenses"],
            )
            print(f"  Feedback written: {ok}")

            # Submit more low ratings to cross the learning threshold
            for _ in range(_LEARNING_THRESHOLD):
                await write_feedback(
                    company_id=company_id,
                    decision_id=items[0].decision_id,
                    record_id=record_id,
                    rating=2,
                    wrong_aspects=["risk_analysis"],
                )

            # Give the learning task time to run
            await asyncio.sleep(5)

        print("\n── Step 4: Feedback insights ───────────────────")
        insights = await get_feedback_insights(company_id)
        print(f"  Total ratings   : {insights.total_ratings}")
        print(f"  Avg rating      : {insights.avg_rating}")
        print(f"  Trend           : {insights.rating_trend}")
        print(f"  Most flagged    : {insights.most_flagged_aspects}")
        if insights.learning_signals:
            print(f"  Learning summary: {insights.learning_signals.learning_summary}")
            print(f"  Avoided lenses  : {insights.learning_signals.avoided_lenses}")
            print(f"  Failure modes   : {insights.learning_signals.common_failure_modes}")

    asyncio.run(_demo())