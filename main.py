"""
main.py — Three AI Finance API Gateway
======================================
Complete entry point including routing, streaming, memory, 
audit, conditions, feedback, and marketplace endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from dotenv import load_dotenv
load_dotenv()

# All required FastAPI imports
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from openai import AsyncOpenAI
import PyPDF2

# ─────────────────────────────────────────────
# THREE AI MODULE IMPORTS
# ─────────────────────────────────────────────
from router import route, UserContext, RoutingPlan
from refiner import refine, RefinedDecision
from output import (
    UIDecisionPayload,
    UIAssistantEnvelope,
    branch_on_question,
    build_assistant_envelope_from_decision,
    build_assistant_envelope_from_text,
    format_for_ui,
)
from finance_runtime import build_finance_snapshot_for_plan
from finance_sync_api import finance_sync_router
from documents import inject_documents_into_context

from memory import (
    enrich_context, update_memory, update_profile, update_rating,
    get_memory_summary, clear_memory,
)
from conditions import (
    register_conditions, get_active_conditions, acknowledge_condition,
    clear_condition, fire_condition, run_spot_check, request_reanalysis,
    get_reanalysis_queue, clear_all_conditions,
    ConditionReport, SpotCheckRequest, SpotCheckResult,
)
from audit import (
    write_audit, get_audit_record, get_decision_history, list_audit_records,
    diff_versions, export_record, audit_stats, clear_audit, delete_audit_record,
    rename_audit_record,
    AuditRecord, AuditListItem, AuditExport, VersionDiff,
)
from feedback import (
    write_feedback, get_feedback_insights, get_feedback_for_decision,
    clear_feedback, FeedbackInsights, VALID_ASPECTS,
)
from marketplace import (
    register_connector, scan_company_conditions, receive_webhook,
    get_connector_config, get_recent_data, disable_connector,
    enable_connector, delete_connector, clear_marketplace,
    ConnectorConfig, ScanReport, WebhookPayload,
)
from insights_runtime import (
    InsightAskIfResponse,
    InsightSnapshotResponse,
    ask_if_insight,
    enroll_company_for_background_insights,
    refresh_company_insights,
    run_background_insights_monitor,
)
from billing import (
    BillingStatusResponse,
    PaystackVerifyRequest,
    get_billing_status,
    verify_paystack_payment,
)

# ─────────────────────────────────────────────
# CONFIG & CACHE
# ─────────────────────────────────────────────

_API_KEY         = os.getenv("THINK_AI_API_KEY", "")
_ALLOWED_ORIGINS = os.getenv("THINK_AI_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
_ENV             = os.getenv("THINK_AI_ENV", "development")
_MAX_BODY_BYTES  = int(os.getenv("THINK_AI_MAX_BODY", str(20 * 1024 * 1024)))

_RATE_DECIDE  = os.getenv("THINK_AI_RATE_DECIDE",  "20/minute")
_RATE_BRANCH  = os.getenv("THINK_AI_RATE_BRANCH",  "40/minute")
_RATE_ROUTE   = os.getenv("THINK_AI_RATE_ROUTE",   "60/minute")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("three-ai.main")

limiter = Limiter(key_func=get_remote_address)

# 🚀 IN-MEMORY CACHE (Bypasses all DB import errors!)

# ─────────────────────────────────────────────
# FORMAT CONTRACTS
# ─────────────────────────────────────────────

_FORMAT_CONTRACTS: dict[str, str] = {
    "compare": (
        "Respond with a single markdown table comparing the options. "
        "Add one bold verdict line above the table. Nothing else."
    ),
    "explain": (
        "Use ### headings and bullet points. Maximum 3 sections. "
        "Each section must have a heading and 2-4 bullets. No walls of text."
    ),
    "summarize": (
        "Return exactly three labelled blocks:\n"
        "**TL;DR:** (1 sentence)\n"
        "**Key Points:** (3-5 bullets)\n"
        "**So What:** (1 sentence action or implication)\n"
        "No other text."
    ),
    "plan": (
        "Return numbered steps only. "
        "Each step: **Bold Title** — 1-2 sentence description. "
        "End with a **Expected Outcome:** line. No other text."
    ),
    "email": (
        "Return exactly two labelled fields:\n"
        "**Subject:** ...\n\n"
        "**Body:** ...\n\n"
        "Professional tone. No preamble."
    ),
    "data": (
        "Return a markdown table only. No prose before or after. "
        "If data is unavailable, return a single line: *Insufficient data to generate table.*"
    ),
    "default": (
        "Write like a senior analyst briefing an executive: "
        "structured, direct, no filler. Use ### headings if the response exceeds 3 paragraphs. "
        "Never open with affirmations or greetings."
    ),
}

def _detect_format_contract(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["compare", " vs ", "versus", "difference between", "which is better", "better than"]):
        return _FORMAT_CONTRACTS["compare"]
    if any(w in q for w in ["explain", "how does", "how do", "what is", "what are", "why does", "why is"]):
        return _FORMAT_CONTRACTS["explain"]
    if any(w in q for w in ["summarize", "summary", "tl;dr", "tldr", "overview", "recap"]):
        return _FORMAT_CONTRACTS["summarize"]
    if any(w in q for w in ["plan", "steps", "how to", "roadmap", "process for", "guide"]):
        return _FORMAT_CONTRACTS["plan"]
    if any(w in q for w in ["email", "write to", "draft a", "draft an", "message to", "send to"]):
        return _FORMAT_CONTRACTS["email"]
    if any(w in q for w in ["table", "list", "show me", "data", "numbers", "breakdown"]):
        return _FORMAT_CONTRACTS["data"]
    return _FORMAT_CONTRACTS["default"]


# ─────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Three AI starting | env=%s", _ENV)
    if not os.getenv("OPENAI_API_KEY"):
        logger.critical("OPENAI_API_KEY is not set — all LLM calls will fail")
    if not _API_KEY and _ENV == "production":
        logger.warning("THINK_AI_API_KEY is not set in production — API is unprotected")
    app.state.insights_stop_event = asyncio.Event()
    app.state.insights_monitor_task = asyncio.create_task(
        run_background_insights_monitor(app.state.insights_stop_event)
    )
    logger.info("✅ Three AI ready")
    try:
        yield
    finally:
        app.state.insights_stop_event.set()
        app.state.insights_monitor_task.cancel()
        try:
            await app.state.insights_monitor_task
        except asyncio.CancelledError:
            pass
        logger.info("🛑 Three AI shutting down")


# ─────────────────────────────────────────────
# APP & MIDDLEWARE
# ─────────────────────────────────────────────

app = FastAPI(
    title="Three AI Finance Engine",
    version="2.0.0-finance",
    docs_url="/docs" if _ENV != "production" else None,
    redoc_url="/redoc" if _ENV != "production" else None,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.started_at = time.perf_counter()

    response = await call_next(request)

    latency_ms = round((time.perf_counter() - request.state.started_at) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-MS"] = str(latency_ms)

    if request.url.path not in ("/health", "/ready"):
        logger.info(
            "%s %s | status=%d | latency_ms=%d | req_id=%s",
            request.method, request.url.path,
            response.status_code, latency_ms, request_id,
        )
    return response

@app.middleware("http")
async def body_size_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type or request.url.path.startswith("/finance/upload"):
        return await call_next(request)
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_BODY_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"detail": f"Request body exceeds {_MAX_BODY_BYTES} bytes."},
        )
    return await call_next(request)

# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    if _ENV == "production" and not _API_KEY:
        logger.critical("SECURITY BREACH PREVENTED: Missing THINK_AI_API_KEY in production!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration. API closed.",
        )
    if _API_KEY and key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


app.include_router(finance_sync_router, dependencies=[Depends(require_api_key)])


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class DecisionRequest(BaseModel):
    query:        str = Field(min_length=3, max_length=2000)
    company_id:   str = Field(min_length=1, max_length=128)
    context:      dict[str, Any] | None = Field(default=None)
    bypass_cache: bool = Field(default=False)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()

    @field_validator("company_id")
    @classmethod
    def strip_company_id(cls, v: str) -> str:
        return v.strip()

    def to_user_context(self) -> UserContext | None:
        if not self.context:
            return None
        return UserContext(
            industry=self.context.get("industry"),
            company_size=self.context.get("company_size"),
            risk_appetite=self.context.get("risk_appetite"),
            extra={k: v for k, v in self.context.items()
                   if k not in ("industry", "company_size", "risk_appetite")},
        )

class BranchRequest(BaseModel):
    question:     str = Field(min_length=3, max_length=2000)
    parent_query: str = Field(min_length=3, max_length=2000)
    company_id:   str = Field(min_length=1, max_length=128)
    context:      dict[str, Any] | None = Field(default=None)
    bypass_cache: bool = Field(default=False)

    @field_validator("question", "parent_query", "company_id")
    @classmethod
    def strip_fields(cls, v: str) -> str:
        return v.strip()

    def to_user_context(self) -> UserContext | None:
        if not self.context:
            return None
        return UserContext(
            industry=self.context.get("industry"),
            company_size=self.context.get("company_size"),
            risk_appetite=self.context.get("risk_appetite"),
            extra={k: v for k, v in self.context.items()
                   if k not in ("industry", "company_size", "risk_appetite")},
        )

class RouteOnlyResponse(BaseModel):
    intent:                 str
    decision_type:          str
    stake_level:            str
    framework:              str
    selected_lenses:        list[str]
    box_count:              int
    required_questions:     list[str]
    assumptions:            list[str]
    constraints_detected:   list[str]
    reasoning_depth:        int
    is_reversible:          bool
    confidence:             float
    is_live_query:          bool
    analysis_horizon_days:  int
    finance_priority:       str

class HealthResponse(BaseModel):
    status:  str
    version: str
    env:     str

class ReadyResponse(BaseModel):
    status:     str
    openai_key: str

class ProfileUpdateRequest(BaseModel):
    company_id:   str = Field(min_length=1, max_length=128)
    profile_data: dict[str, Any]

class AuditRenameRequest(BaseModel):
    query_preview: str = Field(min_length=3, max_length=160)

    @field_validator("query_preview")
    @classmethod
    def strip_query_preview(cls, v: str) -> str:
        return " ".join(v.strip().split())

class RatingRequest(BaseModel):
    company_id:    str = Field(min_length=1, max_length=128)
    decision_id:   str = Field(min_length=1, max_length=64)
    record_id:     str = Field(min_length=1, max_length=64)
    rating:        int = Field(ge=1, le=5)
    comment:       str | None = Field(default=None, max_length=500)
    wrong_aspects: list[str] = Field(default_factory=list)

class AcknowledgeRequest(BaseModel):
    company_id:   str = Field(min_length=1, max_length=128)
    condition_id: str = Field(min_length=1, max_length=64)

class ConditionActionRequest(BaseModel):
    company_id:   str = Field(min_length=1, max_length=128)
    condition_id: str = Field(min_length=1, max_length=64)
    action:       Literal["fire", "clear", "reanalysis"]

class InsightAskIfRequest(BaseModel):
    company_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=3, max_length=1500)
    force_refresh: bool = False

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        return " ".join(value.strip().split())

class ConnectorToggleRequest(BaseModel):
    company_id:   str = Field(min_length=1, max_length=128)
    connector_id: str = Field(min_length=1, max_length=64)

# ─────────────────────────────────────────────
# DOCUMENT INJECTION & UPLOAD (GLOBAL CACHE FIX)
# ─────────────────────────────────────────────

async def _inject_documents(company_id: str, ctx: UserContext | None) -> UserContext:
    return await inject_documents_into_context(company_id, ctx)
    if ctx is None:
        ctx = UserContext(extra={})
    if ctx.extra is None:
        ctx.extra = {}
        
    # 🚨 THE FIX: Read from the GLOBAL key, ignoring the random company_id
    doc_text = _DOCUMENT_CACHE.get("GLOBAL_DOC")
    
    if doc_text:
        ctx.extra["Uploaded Financial Documents"] = doc_text
        logger.info(f"✅ BLINDFOLD OFF: Injected {len(doc_text)} chars of PDF text into AI context!")
    else:
        logger.warning(f"⚠️ No PDF text found in global cache.")
        
    return ctx

@app.post("/finance/upload/legacy-global", tags=["Sync"], include_in_schema=False)
async def upload_financial_context(
    company_id: str = Form(...),
    files: list[UploadFile] = File(...)
):
    try:
        combined_text = f"--- UPLOADED FINANCIAL DOCUMENTATION FOR {company_id} ---\n\n"
        
        for file in files:
            combined_text += f"Document Name: {file.filename}\n"
            if file.filename.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file.file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        combined_text += text + "\n"
            elif file.filename.endswith(".csv"):
                content = await file.read()
                combined_text += content.decode("utf-8", errors="ignore") + "\n"
                
            combined_text += "\n------------------------\n"

        # 🚨 THE FIX: Save straight to the GLOBAL key
        _DOCUMENT_CACHE["GLOBAL_DOC"] = combined_text
        
        logger.info(f"✅ UPLOAD SUCCESS: Cached {len(combined_text)} characters globally!")
        return {"ok": True, "message": "Files uploaded and text extracted."}

    except Exception as e:
        logger.error(f"❌ Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────
# UPLOAD ENDPOINT
# ─────────────────────────────────────────────

@app.post("/finance/upload/legacy-company", tags=["Sync"], include_in_schema=False)
async def upload_financial_context(
    company_id: str = Form(...),
    files: list[UploadFile] = File(...)
):
    try:
        combined_text = f"--- UPLOADED FINANCIAL DOCUMENTATION FOR {company_id} ---\n\n"
        
        for file in files:
            combined_text += f"Document Name: {file.filename}\n"
            if file.filename.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file.file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        combined_text += text + "\n"
            elif file.filename.endswith(".csv"):
                content = await file.read()
                combined_text += content.decode("utf-8", errors="ignore") + "\n"
                
            combined_text += "\n------------------------\n"

        # Save straight to server memory
        _DOCUMENT_CACHE[company_id] = combined_text
        
        logger.info(f"✅ UPLOAD SUCCESS: Cached {len(combined_text)} characters for {company_id}")
        return {"ok": True, "message": "Files uploaded and text extracted."}

    except Exception as e:
        logger.error(f"❌ Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────
# STREAMING GENERATOR
# ─────────────────────────────────────────────

def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _action_contract() -> str:
    return (
        "Respond ONLY in this exact structure:\n"
        "**Action:** (what you are doing, 1 line)\n"
        "**Steps:**\n"
        "1. (step)\n"
        "2. (step)\n"
        "**Expected Output:** (what the user gets)\n"
        "**Risks:** (1-2 bullet points or 'None identified')"
    )


def _build_llm_messages(query: str, ctx: UserContext, kind: Literal["chat", "action"]) -> list[dict[str, str]]:
    if kind == "action":
        format_instruction = _action_contract()
        user_content = query
    else:
        format_instruction = _detect_format_contract(query)
        user_content = f"User Context:\n{json.dumps(ctx.extra or {}, default=str)}\n\nQuery:\n{query}"

    system_setup = f"""You are Three AI, an advanced finance-first enterprise engine.
COMPANY CONTEXT:
- Industry: {ctx.industry or 'Unknown'}
- Size: {ctx.company_size or 'Unknown'}
- Uploaded Documents Available: {"Yes" if ctx.extra and "Uploaded Financial Documents" in ctx.extra else "No"}
- Finance Snapshot Available: {"Yes" if ctx.extra else "No"}
OUTPUT CONTRACT:
{format_instruction}
"""
    return [
        {"role": "system", "content": system_setup},
        {"role": "user", "content": user_content},
    ]


async def _prepare_context_and_plan(body: DecisionRequest) -> tuple[UserContext, RoutingPlan]:
    ctx = body.to_user_context()
    ctx = await enrich_context(body.company_id, ctx)
    ctx = await _inject_documents(body.company_id, ctx)
    plan = await asyncio.wait_for(
        route(
            query=body.query,
            company_id=body.company_id,
            context=ctx,
            bypass_cache=body.bypass_cache,
        ),
        timeout=20,
    )
    ctx = await build_finance_snapshot_for_plan(
        company_id=body.company_id,
        plan=plan,
        ctx=ctx,
    )
    return ctx, plan


async def _complete_non_decision_response(query: str, ctx: UserContext, kind: Literal["chat", "action"]) -> str:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    response = await client.chat.completions.create(
        model=os.getenv("THINK_AI_ROUTER_MODEL", "gpt-4o-mini"),
        messages=_build_llm_messages(query, ctx, kind),
        temperature=0.3 if kind == "action" else 0.4,
    )
    message = response.choices[0].message.content
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return "".join(
            part.get("text", "")
            for part in message
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(message or "").strip()


def _decision_id_from_payload(payload: UIDecisionPayload) -> str:
    audit = getattr(payload, "audit", None)
    if audit and getattr(audit, "query_hash", None):
        return audit.query_hash[:16]
    return uuid.uuid4().hex[:16]


def _queue_post_decision_tasks(
    *,
    company_id: str,
    query: str,
    plan: RoutingPlan,
    decision: RefinedDecision,
    payload: UIDecisionPayload,
    user_id: str | None,
) -> None:
    asyncio.create_task(update_memory(company_id, query, plan, payload))
    asyncio.create_task(
        write_audit(
            company_id=company_id,
            query=query,
            plan=plan,
            decision=decision,
            payload=payload,
            user_id=user_id,
        )
    )
    asyncio.create_task(
        register_conditions(
            company_id=company_id,
            decision_id=_decision_id_from_payload(payload),
            query=query,
            verdict=decision.verdict_box,
        )
    )


async def generate_decision_stream(body: DecisionRequest, request: Request):
    try:
        ctx, plan = await _prepare_context_and_plan(body)

        if plan.intent in {"action", "chat"}:
            kind = plan.intent
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
            response_stream = await client.chat.completions.create(
                model=os.getenv("THINK_AI_ROUTER_MODEL", "gpt-4o-mini"),
                messages=_build_llm_messages(body.query, ctx, kind),
                stream=True,
                temperature=0.3 if kind == "action" else 0.4,
            )
            streamed_parts: list[str] = []

            yield _sse_event({"type": "status", "text": "Routing complete. Generating structured response."})

            async for chunk in response_stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                streamed_parts.append(delta)
                yield _sse_event({"type": "chunk", "text": delta})

            envelope = build_assistant_envelope_from_text("".join(streamed_parts), kind=kind)
            yield _sse_event({"type": "assistant", "payload": envelope.model_dump(mode="json")})
            yield "data: [DONE]\n\n"
            return

        yield _sse_event(
            {
                "type": "status",
                "text": f"Finance triage routed via {plan.framework} ({plan.stake_level} stakes, {plan.decision_type}).",
            }
        )
        await asyncio.sleep(0.1)

        decision = await asyncio.wait_for(
            refine(
                query=body.query,
                plan=plan,
                company_id=body.company_id,
                context=ctx,
                bypass_cache=body.bypass_cache,
            ),
            timeout=90,
        )

        yield _sse_event({"type": "status", "text": "Finance synthesis complete. Building decision board."})

        payload = format_for_ui(decision=decision, query=body.query, company_id=body.company_id, context=ctx)
        envelope = build_assistant_envelope_from_decision(payload)

        _queue_post_decision_tasks(
            company_id=body.company_id,
            query=body.query,
            plan=plan,
            decision=decision,
            payload=payload,
            user_id=getattr(request.state, "user_id", None),
        )

        yield _sse_event({"type": "assistant", "payload": envelope.model_dump(mode="json")})
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error("Stream failed | company=%s | %s", body.company_id, e)
        yield _sse_event({"type": "error", "error": "Three AI encountered an error."})
        yield "data: [DONE]\n\n"

@app.post("/decide/stream", tags=["Decisions"])
@limiter.limit(_RATE_DECIDE)
async def decide_stream(request: Request, body: DecisionRequest, _auth: None = Depends(require_api_key)):
    return StreamingResponse(generate_decision_stream(body, request), media_type="text/event-stream")

# ─────────────────────────────────────────────
# DECISION ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/decide", response_model=UIAssistantEnvelope, tags=["Decisions"])
@limiter.limit(_RATE_DECIDE)
async def decide(request: Request, body: DecisionRequest, _auth: None = Depends(require_api_key)) -> UIAssistantEnvelope:
    ctx, plan = await _prepare_context_and_plan(body)

    if plan.intent in {"chat", "action"}:
        reply = await _complete_non_decision_response(body.query, ctx, kind=plan.intent)
        return build_assistant_envelope_from_text(reply, kind=plan.intent)

    decision = await refine(
        query=body.query,
        plan=plan,
        company_id=body.company_id,
        context=ctx,
        bypass_cache=body.bypass_cache,
    )
    payload = format_for_ui(decision=decision, query=body.query, company_id=body.company_id, context=ctx)

    _queue_post_decision_tasks(
        company_id=body.company_id,
        query=body.query,
        plan=plan,
        decision=decision,
        payload=payload,
        user_id=getattr(request.state, "user_id", None),
    )
    return build_assistant_envelope_from_decision(payload)


@app.post("/branch", response_model=UIAssistantEnvelope, tags=["Decisions"])
@limiter.limit(_RATE_BRANCH)
async def branch(request: Request, body: BranchRequest, _auth: None = Depends(require_api_key)) -> UIAssistantEnvelope:
    ctx = body.to_user_context()
    ctx = await enrich_context(body.company_id, ctx)
    ctx = await _inject_documents(body.company_id, ctx)

    payload = await branch_on_question(question=body.question, parent_query=body.parent_query, company_id=body.company_id, context=ctx, bypass_cache=body.bypass_cache)
    return build_assistant_envelope_from_decision(payload)

@app.post("/route-only", response_model=RouteOnlyResponse, tags=["Decisions"])
@limiter.limit(_RATE_ROUTE)
async def route_only(request: Request, body: DecisionRequest, _auth: None = Depends(require_api_key)) -> RouteOnlyResponse:
    ctx = body.to_user_context()
    ctx = await enrich_context(body.company_id, ctx)
    ctx = await _inject_documents(body.company_id, ctx)
    plan = await route(query=body.query, company_id=body.company_id, context=ctx, bypass_cache=body.bypass_cache)
    return RouteOnlyResponse(**plan.model_dump())

# ─────────────────────────────────────────────
# OTHER ENDPOINTS (SYSTEM/MEMORY/AUDIT)
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="2.0.0-finance", env=_ENV)

@app.get("/ready", response_model=ReadyResponse, tags=["System"])
async def ready() -> ReadyResponse:
    return ReadyResponse(status="ready", openai_key="set")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "-")
    logger.exception("Unhandled application error | req_id=%s", req_id, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Three AI encountered an internal error.", "request_id": req_id},
    )

@app.get("/memory/{company_id}", tags=["Memory"])
@limiter.limit("30/minute")
async def get_memory(request: Request, company_id: str, _auth: None = Depends(require_api_key)) -> dict:
    return await get_memory_summary(company_id.strip()) or {}

@app.post("/memory/profile", tags=["Memory"])
@limiter.limit("10/minute")
async def update_company_profile(request: Request, body: ProfileUpdateRequest, _auth: None = Depends(require_api_key)) -> dict:
    await update_profile(body.company_id, body.profile_data)
    return {"status": "updated", "company_id": body.company_id}

@app.delete("/memory/{company_id}", tags=["Memory"])
@limiter.limit("5/minute")
async def delete_memory(request: Request, company_id: str, _auth: None = Depends(require_api_key)) -> dict:
    await clear_memory(company_id.strip())
    return {"status": "cleared", "company_id": company_id}



    # ─────────────────────────────────────────────
@app.get("/billing/{company_id}", response_model=BillingStatusResponse, tags=["Billing"])
@limiter.limit("30/minute")
async def read_billing_status(
    request: Request,
    company_id: str,
    _auth: None = Depends(require_api_key),
) -> BillingStatusResponse:
    return await get_billing_status(company_id.strip())


@app.post("/billing/paystack/verify", response_model=BillingStatusResponse, tags=["Billing"])
@limiter.limit("20/minute")
async def verify_paystack_route(
    request: Request,
    body: PaystackVerifyRequest,
    _auth: None = Depends(require_api_key),
) -> BillingStatusResponse:
    try:
        return await verify_paystack_payment(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/conditions/{company_id}", response_model=ConditionReport, tags=["Conditions"])
@limiter.limit("30/minute")
async def list_conditions(request: Request, company_id: str, _auth: None = Depends(require_api_key)) -> ConditionReport:
    return await get_active_conditions(company_id.strip())


@app.get("/conditions/{company_id}/scan", response_model=ScanReport, tags=["Conditions"])
@limiter.limit("15/minute")
async def scan_conditions(request: Request, company_id: str, _auth: None = Depends(require_api_key)) -> ScanReport:
    return await scan_company_conditions(company_id.strip())


@app.post("/conditions/acknowledge", tags=["Conditions"])
@limiter.limit("20/minute")
async def acknowledge_condition_route(
    request: Request,
    body: AcknowledgeRequest,
    _auth: None = Depends(require_api_key),
) -> dict:
    acknowledged = await acknowledge_condition(body.company_id, body.condition_id)
    if not acknowledged:
        raise HTTPException(status_code=404, detail="Condition not found")
    return {"status": "acknowledged", "condition_id": body.condition_id}


@app.post("/conditions/action", tags=["Conditions"])
@limiter.limit("20/minute")
async def condition_action(
    request: Request,
    body: ConditionActionRequest,
    _auth: None = Depends(require_api_key),
) -> dict:
    if body.action == "fire":
        changed = await fire_condition(body.company_id, body.condition_id)
    elif body.action == "clear":
        changed = await clear_condition(body.company_id, body.condition_id)
    else:
        changed = await request_reanalysis(body.company_id, body.condition_id)

    if not changed:
        raise HTTPException(status_code=404, detail="Condition not found")

    return {
        "status": "ok",
        "condition_id": body.condition_id,
        "action": body.action,
    }


@app.post("/conditions/spot-check", response_model=SpotCheckResult, tags=["Conditions"])
@limiter.limit("15/minute")
async def spot_check_condition(
    request: Request,
    body: SpotCheckRequest,
    _auth: None = Depends(require_api_key),
) -> SpotCheckResult:
    return await run_spot_check(body)


@app.get("/insights/{company_id}", response_model=InsightSnapshotResponse, tags=["Insights"])
@limiter.limit("20/minute")
async def get_insights_snapshot(
    request: Request,
    company_id: str,
    force_refresh: bool = False,
    _auth: None = Depends(require_api_key),
) -> InsightSnapshotResponse:
    company_id = company_id.strip()
    await enroll_company_for_background_insights(
        company_id,
        reason="insights-open",
    )
    return await refresh_company_insights(company_id, force_refresh=force_refresh)


@app.post("/insights/{company_id}/refresh", response_model=InsightSnapshotResponse, tags=["Insights"])
@limiter.limit("10/minute")
async def force_refresh_insights(
    request: Request,
    company_id: str,
    _auth: None = Depends(require_api_key),
) -> InsightSnapshotResponse:
    company_id = company_id.strip()
    await enroll_company_for_background_insights(
        company_id,
        reason="manual-refresh",
    )
    return await refresh_company_insights(company_id, force_refresh=True)


@app.post("/insights/ask-if", response_model=InsightAskIfResponse, tags=["Insights"])
@limiter.limit("20/minute")
async def ask_if_route(
    request: Request,
    body: InsightAskIfRequest,
    _auth: None = Depends(require_api_key),
) -> InsightAskIfResponse:
    await enroll_company_for_background_insights(
        body.company_id,
        reason="ask-if",
    )
    return await ask_if_insight(
        company_id=body.company_id,
        question=body.question,
        force_refresh=body.force_refresh,
    )


# AUDIT ENDPOINTS (RESTORED)
# ─────────────────────────────────────────────

@app.get("/audit/{company_id}", tags=["Audit"])
@limiter.limit("30/minute")
async def get_audit_history(request: Request, company_id: str, limit: int = 50, _auth: None = Depends(require_api_key)):
    return await list_audit_records(company_id.strip(), limit=limit)

@app.get("/audit/{company_id}/{record_id}", tags=["Audit"])
@limiter.limit("30/minute")
async def get_single_audit(request: Request, company_id: str, record_id: str, _auth: None = Depends(require_api_key)):
    record = await get_audit_record(company_id.strip(), record_id.strip())
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record

@app.delete("/audit/{company_id}/{record_id}", tags=["Audit"])
@limiter.limit("30/minute")
async def delete_audit(request: Request, company_id: str, record_id: str, _auth: None = Depends(require_api_key)):
    deleted = await delete_audit_record(company_id.strip(), record_id.strip())
    if not deleted:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"status": "deleted"}

@app.patch("/audit/{company_id}/{record_id}", response_model=AuditListItem, tags=["Audit"])
@limiter.limit("30/minute")
async def rename_audit(
    request: Request,
    company_id: str,
    record_id: str,
    body: AuditRenameRequest,
    _auth: None = Depends(require_api_key),
):
    item = await rename_audit_record(company_id.strip(), record_id.strip(), body.query_preview)
    if not item:
        raise HTTPException(status_code=404, detail="Record not found")
    return item

app.router.routes = [
    route
    for route in app.router.routes
    if getattr(route, "path", None) not in {"/finance/upload/legacy-global", "/finance/upload/legacy-company"}
]

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=(_ENV == "development"),
        log_level="info",
    )
