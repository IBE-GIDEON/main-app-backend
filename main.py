"""
main.py — Think AI API Gateway
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from router import route, UserContext, RoutingPlan
from refiner import refine, RefinedDecision
from output import format_for_ui, branch_on_question, UIDecisionPayload
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
    diff_versions, export_record, audit_stats, clear_audit,
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

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

_API_KEY         = os.getenv("THINK_AI_API_KEY", "")
_ALLOWED_ORIGINS = os.getenv("THINK_AI_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
_ENV             = os.getenv("THINK_AI_ENV", "development")
_MAX_BODY_BYTES  = int(os.getenv("THINK_AI_MAX_BODY", "65536"))

_RATE_DECIDE  = os.getenv("THINK_AI_RATE_DECIDE",  "20/minute")
_RATE_BRANCH  = os.getenv("THINK_AI_RATE_BRANCH",  "40/minute")
_RATE_ROUTE   = os.getenv("THINK_AI_RATE_ROUTE",   "60/minute")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("think-ai.main")

limiter = Limiter(key_func=get_remote_address)

# ─────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Think AI starting | env=%s", _ENV)
    if not os.getenv("OPENAI_API_KEY"):
        logger.critical("OPENAI_API_KEY is not set — all LLM calls will fail")
    if not _API_KEY and _ENV == "production":
        logger.warning("THINK_AI_API_KEY is not set in production — API is unprotected")
    logger.info("✅ Think AI ready")
    yield
    logger.info("🛑 Think AI shutting down")

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Think AI Decision Engine",
    version="1.0.0",
    docs_url="/docs"  if _ENV != "production" else None,
    redoc_url="/redoc" if _ENV != "production" else None,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS — must be added BEFORE any other middleware ──────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────

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
    # Let OPTIONS preflight through untouched — CORS middleware handles it
    if request.method == "OPTIONS":
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
    if not _API_KEY:
        return
    if key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

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
    decision_type:        str
    stake_level:          str
    framework:            str
    selected_lenses:      list[str]
    box_count:            int
    required_questions:   list[str]
    assumptions:          list[str]
    constraints_detected: list[str]
    reasoning_depth:      int
    is_reversible:        bool
    confidence:           float


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

    @field_validator("company_id")
    @classmethod
    def strip_id(cls, v: str) -> str:
        return v.strip()


class RatingRequest(BaseModel):
    company_id:    str = Field(min_length=1, max_length=128)
    decision_id:   str = Field(min_length=1, max_length=64)
    record_id:     str = Field(min_length=1, max_length=64)
    rating:        int = Field(ge=1, le=5)
    comment:       str | None = Field(default=None, max_length=500)
    wrong_aspects: list[str] = Field(default_factory=list)

    @field_validator("company_id", "decision_id", "record_id")
    @classmethod
    def strip_fields(cls, v: str) -> str:
        return v.strip()


class AcknowledgeRequest(BaseModel):
    company_id:   str = Field(min_length=1, max_length=128)
    condition_id: str = Field(min_length=1, max_length=64)

    @field_validator("company_id", "condition_id")
    @classmethod
    def strip_fields(cls, v: str) -> str:
        return v.strip()


class ConditionActionRequest(BaseModel):
    company_id:   str = Field(min_length=1, max_length=128)
    condition_id: str = Field(min_length=1, max_length=64)
    action:       Literal["fire", "clear", "reanalysis"]

    @field_validator("company_id", "condition_id")
    @classmethod
    def strip_fields(cls, v: str) -> str:
        return v.strip()


class ConnectorToggleRequest(BaseModel):
    company_id:   str = Field(min_length=1, max_length=128)
    connector_id: str = Field(min_length=1, max_length=64)

    @field_validator("company_id", "connector_id")
    @classmethod
    def strip_fields(cls, v: str) -> str:
        return v.strip()


# ─────────────────────────────────────────────
# DECISION ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/decide", response_model=UIDecisionPayload, tags=["Decisions"])
@limiter.limit(_RATE_DECIDE)
async def decide(
    request: Request,
    body:    DecisionRequest,
    _auth:   None = Depends(require_api_key),
) -> UIDecisionPayload:
    logger.info("POST /decide | company=%s | req_id=%s",
                body.company_id, getattr(request.state, "request_id", "-"))

    ctx = body.to_user_context()
    ctx = await enrich_context(body.company_id, ctx)

    try:
        plan = await asyncio.wait_for(
            route(query=body.query, company_id=body.company_id,
                  context=ctx, bypass_cache=body.bypass_cache),
            timeout=20,
        )
    except asyncio.TimeoutError:
        logger.error("Router timeout | company=%s", body.company_id)
        raise HTTPException(status_code=504, detail="Routing timed out. Please retry.")
    except Exception as e:
        logger.error("Router failed | company=%s | %s", body.company_id, e)
        raise HTTPException(status_code=502, detail="Decision routing is temporarily unavailable.")

    try:
        decision = await asyncio.wait_for(
            refine(query=body.query, plan=plan, company_id=body.company_id,
                   context=ctx, bypass_cache=body.bypass_cache),
            timeout=90,
        )
    except asyncio.TimeoutError:
        logger.error("Refiner timeout | company=%s", body.company_id)
        raise HTTPException(status_code=504, detail="Analysis timed out. Please retry.")
    except Exception as e:
        logger.error("Refiner failed | company=%s | %s", body.company_id, e)
        raise HTTPException(status_code=502, detail="Decision analysis is temporarily unavailable.")

    payload = format_for_ui(decision=decision, query=body.query, company_id=body.company_id)

    asyncio.create_task(update_memory(body.company_id, body.query, plan, payload))
    asyncio.create_task(register_conditions(
        company_id=body.company_id,
        decision_id=decision.audit.query_hash[:16],
        query=body.query,
        verdict=decision.verdict_box,
    ))
    asyncio.create_task(write_audit(
        company_id=body.company_id, query=body.query,
        plan=plan, decision=decision, payload=payload,
        user_id=getattr(request.state, "user_id", None),
    ))

    return payload


@app.post("/branch", response_model=UIDecisionPayload, tags=["Decisions"])
@limiter.limit(_RATE_BRANCH)
async def branch(
    request: Request,
    body:    BranchRequest,
    _auth:   None = Depends(require_api_key),
) -> UIDecisionPayload:
    logger.info("POST /branch | company=%s | req_id=%s",
                body.company_id, getattr(request.state, "request_id", "-"))
    ctx = body.to_user_context()
    try:
        payload = await branch_on_question(
            question=body.question, parent_query=body.parent_query,
            company_id=body.company_id, context=ctx, bypass_cache=body.bypass_cache,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Branch failed | company=%s | %s", body.company_id, e)
        raise HTTPException(status_code=502, detail="Branch analysis is temporarily unavailable.")
    return payload


@app.post("/route-only", response_model=RouteOnlyResponse, tags=["Decisions"])
@limiter.limit(_RATE_ROUTE)
async def route_only(
    request: Request,
    body:    DecisionRequest,
    _auth:   None = Depends(require_api_key),
) -> RouteOnlyResponse:
    ctx = body.to_user_context()
    try:
        plan = await route(query=body.query, company_id=body.company_id,
                           context=ctx, bypass_cache=body.bypass_cache)
    except Exception as e:
        logger.error("Route-only failed | company=%s | %s", body.company_id, e)
        raise HTTPException(status_code=502, detail="Routing is temporarily unavailable.")
    return RouteOnlyResponse(
        decision_type=plan.decision_type, stake_level=plan.stake_level,
        framework=plan.framework, selected_lenses=plan.selected_lenses,
        box_count=plan.box_count, required_questions=plan.required_questions,
        assumptions=plan.assumptions, constraints_detected=plan.constraints_detected,
        reasoning_depth=plan.reasoning_depth, is_reversible=plan.is_reversible,
        confidence=plan.confidence,
    )

# ─────────────────────────────────────────────
# SYSTEM
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="1.0.0", env=_ENV)


@app.get("/ready", response_model=ReadyResponse, tags=["System"])
async def ready() -> ReadyResponse:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")
    return ReadyResponse(status="ready", openai_key="set")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "-")
    logger.critical("Unhandled exception | path=%s | req_id=%s | %s",
                    request.url.path, req_id, exc)
    detail = str(exc) if _ENV != "production" else "An unexpected error occurred."
    return JSONResponse(status_code=500, content={"detail": detail, "request_id": req_id})

# ─────────────────────────────────────────────
# MEMORY ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/memory/{company_id}", tags=["Memory"])
@limiter.limit("30/minute")
async def get_memory(request: Request, company_id: str,
                     _auth: None = Depends(require_api_key)) -> dict:
    summary = await get_memory_summary(company_id.strip())
    if summary is None:
        raise HTTPException(status_code=404, detail="No memory found for this company.")
    return summary


@app.post("/memory/profile", tags=["Memory"])
@limiter.limit("10/minute")
async def update_company_profile(request: Request, body: ProfileUpdateRequest,
                                  _auth: None = Depends(require_api_key)) -> dict:
    ok = await update_profile(body.company_id, body.profile_data)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to update profile.")
    return {"status": "updated", "company_id": body.company_id}


@app.post("/memory/rate", tags=["Memory"])
@limiter.limit("30/minute")
async def rate_decision(request: Request, body: RatingRequest,
                         _auth: None = Depends(require_api_key)) -> dict:
    ok = await write_feedback(
        company_id=body.company_id, decision_id=body.decision_id,
        record_id=body.record_id, rating=body.rating,
        comment=body.comment, wrong_aspects=body.wrong_aspects,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Decision not found or rating could not be saved.")
    return {"status": "rated", "decision_id": body.decision_id, "rating": body.rating}


@app.delete("/memory/{company_id}", tags=["Memory"])
@limiter.limit("5/minute")
async def delete_memory(request: Request, company_id: str,
                         _auth: None = Depends(require_api_key)) -> dict:
    mem_ok      = await clear_memory(company_id.strip())
    cond_ok     = await clear_all_conditions(company_id.strip())
    audit_ok    = await clear_audit(company_id.strip())
    feedback_ok = await clear_feedback(company_id.strip())
    market_ok   = await clear_marketplace(company_id.strip())
    if not any([mem_ok, cond_ok, audit_ok, feedback_ok, market_ok]):
        raise HTTPException(status_code=404, detail="No data found for this company.")
    return {"status": "cleared", "company_id": company_id}

# ─────────────────────────────────────────────
# FEEDBACK ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/feedback", tags=["Feedback"])
@limiter.limit("20/minute")
async def submit_feedback(request: Request, body: RatingRequest,
                           _auth: None = Depends(require_api_key)) -> dict:
    ok = await write_feedback(
        company_id=body.company_id, decision_id=body.decision_id,
        record_id=body.record_id, rating=body.rating,
        comment=body.comment, wrong_aspects=body.wrong_aspects,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Feedback could not be saved.")
    return {"status": "received", "decision_id": body.decision_id,
            "rating": body.rating, "learning": body.rating <= 2}


@app.get("/feedback/{company_id}/insights", response_model=FeedbackInsights, tags=["Feedback"])
@limiter.limit("20/minute")
async def feedback_insights(request: Request, company_id: str,
                              _auth: None = Depends(require_api_key)) -> FeedbackInsights:
    return await get_feedback_insights(company_id.strip())


@app.get("/feedback/{company_id}/decision/{decision_id}", tags=["Feedback"])
@limiter.limit("20/minute")
async def feedback_for_decision(request: Request, company_id: str, decision_id: str,
                                  _auth: None = Depends(require_api_key)) -> dict:
    records = await get_feedback_for_decision(company_id.strip(), decision_id.strip())
    return {"decision_id": decision_id, "total": len(records),
            "records": [r.model_dump() for r in records]}

# ─────────────────────────────────────────────
# CONDITIONS ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/conditions/{company_id}", response_model=ConditionReport, tags=["Conditions"])
@limiter.limit("60/minute")
async def get_conditions(request: Request, company_id: str,
                          _auth: None = Depends(require_api_key)) -> ConditionReport:
    return await get_active_conditions(company_id.strip())


@app.post("/conditions/acknowledge", tags=["Conditions"])
@limiter.limit("30/minute")
async def acknowledge(request: Request, body: AcknowledgeRequest,
                       _auth: None = Depends(require_api_key)) -> dict:
    ok = await acknowledge_condition(body.company_id, body.condition_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Condition not found.")
    return {"status": "acknowledged", "condition_id": body.condition_id}


@app.post("/conditions/action", tags=["Conditions"])
@limiter.limit("30/minute")
async def condition_action(request: Request, body: ConditionActionRequest,
                            _auth: None = Depends(require_api_key)) -> dict:
    if body.action == "fire":
        ok = await fire_condition(body.company_id, body.condition_id)
    elif body.action == "clear":
        ok = await clear_condition(body.company_id, body.condition_id)
    else:
        ok = await request_reanalysis(body.company_id, body.condition_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Condition not found or action failed.")
    return {"status": body.action, "condition_id": body.condition_id}


@app.post("/conditions/spot-check", response_model=SpotCheckResult, tags=["Conditions"])
@limiter.limit("10/minute")
async def spot_check(request: Request, body: SpotCheckRequest,
                      _auth: None = Depends(require_api_key)) -> SpotCheckResult:
    ctx = None
    if body.context:
        ctx = UserContext(
            industry=body.context.get("industry"),
            company_size=body.context.get("company_size"),
            risk_appetite=body.context.get("risk_appetite"),
            extra={k: v for k, v in body.context.items()
                   if k not in ("industry", "company_size", "risk_appetite")},
        )
    result = await run_spot_check(company_id=body.company_id, condition_id=body.condition_id,
                                   user_update=body.user_update, context=ctx)
    if result is None:
        raise HTTPException(status_code=404, detail="Condition not found or spot-check failed.")
    return result


@app.get("/conditions/{company_id}/reanalysis-queue", tags=["Conditions"])
@limiter.limit("10/minute")
async def reanalysis_queue(request: Request, company_id: str,
                            _auth: None = Depends(require_api_key)) -> dict:
    queue = await get_reanalysis_queue(company_id.strip())
    return {"company_id": company_id, "queued": len(queue),
            "conditions": [r.model_dump() for r in queue]}

# ─────────────────────────────────────────────
# AUDIT ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/audit/{company_id}", response_model=list[AuditListItem], tags=["Audit"])
@limiter.limit("30/minute")
async def list_audits(request: Request, company_id: str, limit: int = 20,
                       offset: int = 0, decision_type: str | None = None,
                       stake_level: str | None = None, verdict_color: str | None = None,
                       _auth: None = Depends(require_api_key)) -> list[AuditListItem]:
    return await list_audit_records(company_id=company_id.strip(), limit=limit,
                                     offset=offset, decision_type=decision_type,
                                     stake_level=stake_level, verdict_color=verdict_color)


@app.get("/audit/{company_id}/{record_id}", response_model=AuditRecord, tags=["Audit"])
@limiter.limit("30/minute")
async def get_audit(request: Request, company_id: str, record_id: str,
                     _auth: None = Depends(require_api_key)) -> AuditRecord:
    record = await get_audit_record(company_id.strip(), record_id.strip())
    if not record:
        raise HTTPException(status_code=404, detail="Audit record not found.")
    return record


@app.get("/audit/{company_id}/decision/{decision_id}/history",
         response_model=list[AuditRecord], tags=["Audit"])
@limiter.limit("20/minute")
async def decision_history(request: Request, company_id: str, decision_id: str,
                            _auth: None = Depends(require_api_key)) -> list[AuditRecord]:
    return await get_decision_history(company_id.strip(), decision_id.strip())


@app.get("/audit/{company_id}/decision/{decision_id}/diff",
         response_model=VersionDiff, tags=["Audit"])
@limiter.limit("20/minute")
async def decision_diff(request: Request, company_id: str, decision_id: str,
                         version_a: int | None = None, version_b: int | None = None,
                         _auth: None = Depends(require_api_key)) -> VersionDiff:
    diff = await diff_versions(company_id=company_id.strip(), decision_id=decision_id.strip(),
                                version_a=version_a, version_b=version_b)
    if diff is None:
        raise HTTPException(status_code=404, detail="Need at least 2 versions to diff.")
    return diff


@app.get("/audit/{company_id}/{record_id}/export", response_model=AuditExport, tags=["Audit"])
@limiter.limit("20/minute")
async def export_audit(request: Request, company_id: str, record_id: str,
                        _auth: None = Depends(require_api_key)) -> AuditExport:
    result = await export_record(company_id.strip(), record_id.strip())
    if result is None:
        raise HTTPException(status_code=404, detail="Audit record not found.")
    return result


@app.get("/audit/{company_id}/stats", tags=["Audit"])
@limiter.limit("20/minute")
async def get_audit_stats(request: Request, company_id: str,
                           _auth: None = Depends(require_api_key)) -> dict:
    return await audit_stats(company_id.strip())

# ─────────────────────────────────────────────
# MARKETPLACE ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/marketplace/connectors", tags=["Marketplace"])
@limiter.limit("10/minute")
async def add_connector(request: Request, company_id: str, body: ConnectorConfig,
                         _auth: None = Depends(require_api_key)) -> dict:
    ok = await register_connector(company_id.strip(), body)
    if not ok:
        raise HTTPException(status_code=500, detail="Connector could not be registered.")
    return {"status": "registered", "connector_id": body.connector_id, "kind": body.kind}


@app.get("/marketplace/{company_id}/connectors", tags=["Marketplace"])
@limiter.limit("30/minute")
async def list_connectors(request: Request, company_id: str,
                           _auth: None = Depends(require_api_key)) -> list[dict]:
    return await get_connector_config(company_id.strip())


@app.post("/marketplace/{company_id}/scan", response_model=ScanReport, tags=["Marketplace"])
@limiter.limit("5/minute")
async def trigger_scan(request: Request, company_id: str,
                        _auth: None = Depends(require_api_key)) -> ScanReport:
    ctx = UserContext()
    return await scan_company_conditions(company_id.strip(), ctx)


@app.post("/marketplace/webhook/{company_id}/{connector_id}", tags=["Marketplace"])
@limiter.limit("120/minute")
async def webhook_receive(request: Request, company_id: str, connector_id: str,
                           body: WebhookPayload,
                           _auth: None = Depends(require_api_key)) -> dict:
    ok = await receive_webhook(company_id=company_id.strip(),
                                connector_id=connector_id.strip(), payload=body)
    if not ok:
        raise HTTPException(status_code=404, detail="Connector not found.")
    return {"status": "received", "connector_id": connector_id, "metric": body.metric_name}


@app.get("/marketplace/{company_id}/data", tags=["Marketplace"])
@limiter.limit("30/minute")
async def recent_data(request: Request, company_id: str, limit: int = 50,
                       _auth: None = Depends(require_api_key)) -> list[dict]:
    points = await get_recent_data(company_id.strip(), limit=min(limit, 100))
    return [p.model_dump() for p in points]


@app.post("/marketplace/connectors/toggle", tags=["Marketplace"])
@limiter.limit("10/minute")
async def toggle_connector(request: Request, body: ConnectorToggleRequest,
                            action: Literal["enable", "disable"] = "disable",
                            _auth: None = Depends(require_api_key)) -> dict:
    ok = await enable_connector(body.company_id, body.connector_id) if action == "enable" \
         else await disable_connector(body.company_id, body.connector_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Connector not found.")
    return {"status": action + "d", "connector_id": body.connector_id}


@app.delete("/marketplace/{company_id}/connectors/{connector_id}", tags=["Marketplace"])
@limiter.limit("10/minute")
async def remove_connector(request: Request, company_id: str, connector_id: str,
                            _auth: None = Depends(require_api_key)) -> dict:
    ok = await delete_connector(company_id.strip(), connector_id.strip())
    if not ok:
        raise HTTPException(status_code=404, detail="Connector not found.")
    return {"status": "deleted", "connector_id": connector_id}

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