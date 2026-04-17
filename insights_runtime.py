from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from conditions import ConditionReport, get_active_conditions, register_conditions
from finance_runtime import build_finance_snapshot_for_plan
from finance_scheduler import run_company_sync
from marketplace import DataPoint, ScanReport, get_connector_config, get_recent_data, scan_company_conditions
from memory import enrich_context, get_memory_summary, update_memory
from output import UIDecisionPayload, format_for_ui
from refiner import refine
from router import UserContext, route
from audit import write_audit

logger = logging.getLogger("three-ai.insights")

_REFRESH_INTERVAL_SECONDS = 300
_MONITOR_POLL_SECONDS = int(os.getenv("THINK_AI_INSIGHTS_POLL_SECONDS", "60"))
_MONITOR_RETRY_SECONDS = int(os.getenv("THINK_AI_INSIGHTS_RETRY_SECONDS", "120"))
_REGISTRY_PATH = Path(
    os.getenv(
        "THINK_AI_INSIGHTS_REGISTRY_PATH",
        str(Path(__file__).with_name("insights_registry.json")),
    )
)
_cache: dict[str, "InsightSnapshotResponse"] = {}
_cache_locks: dict[str, asyncio.Lock] = {}
_registry_lock = asyncio.Lock()


class InsightConnectorStatus(BaseModel):
    connector_id: str
    label: str
    enabled: bool
    status: str
    detail: str | None = None
    last_used_utc: str | None = None
    category: str = "finance"


class InsightMonitorCard(BaseModel):
    monitor_id: str
    title: str
    description: str
    question: str
    decision: UIDecisionPayload
    generated_utc: str


class InsightOverview(BaseModel):
    company_id: str
    generated_utc: str
    next_refresh_utc: str
    refresh_interval_seconds: int = _REFRESH_INTERVAL_SECONDS
    source_count: int = 0
    connected_sources: list[str] = Field(default_factory=list)
    datapoint_count: int = 0
    monitor_count: int = 0
    active_conditions: int = 0
    fired_conditions: int = 0
    urgent_conditions: int = 0


class InsightWatchStatus(BaseModel):
    company_id: str
    active: bool = True
    status: str = "idle"
    enrolled_utc: str
    last_requested_utc: str | None = None
    last_completed_utc: str | None = None
    next_due_utc: str | None = None
    last_duration_ms: int | None = None
    last_error: str | None = None
    last_reason: str | None = None


class InsightWatchRegistry(BaseModel):
    version: str = "1.0"
    updated_utc: str
    companies: list[InsightWatchStatus] = Field(default_factory=list)


class InsightSnapshotResponse(BaseModel):
    overview: InsightOverview
    connector_status: list[InsightConnectorStatus] = Field(default_factory=list)
    sync_result: dict[str, Any] = Field(default_factory=dict)
    recent_data: list[DataPoint] = Field(default_factory=list)
    conditions: ConditionReport
    scan_report: ScanReport
    monitors: list[InsightMonitorCard] = Field(default_factory=list)
    watch_status: InsightWatchStatus | None = None
    generated_utc: str
    cached: bool = False


class InsightAskIfResponse(BaseModel):
    company_id: str
    query: str
    decision: UIDecisionPayload
    watch_status: InsightWatchStatus | None = None
    generated_utc: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _future_iso(seconds: int) -> str:
    return (_utc_now() + timedelta(seconds=seconds)).isoformat()


def _watch_due_now(status: InsightWatchStatus) -> bool:
    if not status.active:
        return False
    if not status.next_due_utc:
        return True
    try:
        return datetime.fromisoformat(status.next_due_utc) <= _utc_now()
    except Exception:
        return True


def _new_watch_status(company_id: str, reason: str) -> InsightWatchStatus:
    now = _utc_now_iso()
    return InsightWatchStatus(
        company_id=company_id,
        active=True,
        status="scheduled",
        enrolled_utc=now,
        last_requested_utc=now,
        next_due_utc=now,
        last_reason=reason,
    )


def _registry_default() -> InsightWatchRegistry:
    return InsightWatchRegistry(updated_utc=_utc_now_iso())


def _load_registry_sync() -> InsightWatchRegistry:
    if not _REGISTRY_PATH.exists():
        return _registry_default()

    try:
        raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        return InsightWatchRegistry.model_validate(raw)
    except Exception as exc:
        logger.warning("Insights registry parse failed | %s", exc)
        return _registry_default()


def _save_registry_sync(registry: InsightWatchRegistry) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _REGISTRY_PATH.with_suffix(".tmp")
    tmp_path.write_text(
        registry.model_dump_json(indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(_REGISTRY_PATH)


async def _load_registry() -> InsightWatchRegistry:
    return await asyncio.to_thread(_load_registry_sync)


async def _save_registry(registry: InsightWatchRegistry) -> None:
    registry.updated_utc = _utc_now_iso()
    await asyncio.to_thread(_save_registry_sync, registry)


async def _upsert_watch_status(status: InsightWatchStatus) -> InsightWatchStatus:
    async with _registry_lock:
        registry = await _load_registry()
        deduped: dict[str, InsightWatchStatus] = {
            item.company_id: item for item in registry.companies
        }
        deduped[status.company_id] = status
        registry.companies = sorted(deduped.values(), key=lambda item: item.company_id)
        await _save_registry(registry)
        return status


async def enroll_company_for_background_insights(
    company_id: str,
    *,
    reason: str = "insights",
) -> InsightWatchStatus:
    company_id = company_id.strip()
    if not company_id:
        raise ValueError("company_id is required")

    async with _registry_lock:
        registry = await _load_registry()
        by_company = {item.company_id: item for item in registry.companies}
        status = by_company.get(company_id) or _new_watch_status(company_id, reason)
        now = _utc_now_iso()

        status.active = True
        status.status = "scheduled"
        status.last_requested_utc = now
        status.last_reason = reason
        status.next_due_utc = now

        by_company[company_id] = status
        registry.companies = sorted(by_company.values(), key=lambda item: item.company_id)
        await _save_registry(registry)
        return status


async def get_company_watch_status(company_id: str) -> InsightWatchStatus | None:
    company_id = company_id.strip()
    if not company_id:
        return None

    async with _registry_lock:
        registry = await _load_registry()
        return next(
            (item for item in registry.companies if item.company_id == company_id),
            None,
        )


async def _record_watch_run(
    company_id: str,
    *,
    status: str,
    duration_ms: int | None = None,
    error: str | None = None,
    reason: str | None = None,
    next_due_seconds: int = _REFRESH_INTERVAL_SECONDS,
) -> InsightWatchStatus:
    company_id = company_id.strip()
    now = _utc_now_iso()

    async with _registry_lock:
        registry = await _load_registry()
        by_company = {item.company_id: item for item in registry.companies}
        current = by_company.get(company_id) or _new_watch_status(company_id, reason or "background")

        current.active = True
        current.status = status
        current.last_completed_utc = now
        current.last_duration_ms = duration_ms
        current.last_error = error
        if reason:
            current.last_reason = reason
        current.next_due_utc = _future_iso(next_due_seconds)

        by_company[company_id] = current
        registry.companies = sorted(by_company.values(), key=lambda item: item.company_id)
        await _save_registry(registry)
        return current


def _get_lock(company_id: str) -> asyncio.Lock:
    if company_id not in _cache_locks:
        _cache_locks[company_id] = asyncio.Lock()
    return _cache_locks[company_id]


def _recent_metric_names(points: list[DataPoint]) -> set[str]:
    return {point.metric_name for point in points}


def _connected_sources(memory_summary: dict[str, Any] | None) -> list[str]:
    finance = (memory_summary or {}).get("finance") or {}
    sources = finance.get("connected_sources") or []
    if not isinstance(sources, list):
        return []
    return [str(source) for source in sources if str(source).strip()]


def _monitor_templates(
    memory_summary: dict[str, Any] | None,
    connectors: list[dict[str, Any]],
    recent_data: list[DataPoint],
) -> list[dict[str, str]]:
    recent_metrics = _recent_metric_names(recent_data)
    connected_source_labels = {
        str((connector.get("label") or connector.get("connector_id") or "")).lower()
        for connector in connectors
    }

    templates: list[dict[str, str]] = [
        {
            "monitor_id": "liquidity-sentinel",
            "title": "Liquidity sentinel",
            "description": "Checks cash, burn, runway, and near-term survivability.",
            "question": "Using the latest connected company finance data, should leadership worry about liquidity, burn, or runway right now? Return the most important upside, risk, and explicit go, stop, and review conditions.",
        },
        {
            "monitor_id": "revenue-quality-radar",
            "title": "Revenue quality radar",
            "description": "Tracks revenue durability, growth quality, and retention risk.",
            "question": "Using the latest connected company finance data, is revenue quality getting stronger or weaker right now, and under what conditions would that verdict change?",
        },
        {
            "monitor_id": "margin-pressure-watch",
            "title": "Margin pressure watch",
            "description": "Looks for gross margin, net margin, and operating pressure.",
            "question": "Using the latest connected company finance data, are margins under pressure right now, and what would finance leadership need to see before acting or stopping?",
        },
        {
            "monitor_id": "board-finance-pulse",
            "title": "Board finance pulse",
            "description": "Compresses the live financial picture into a board-level call.",
            "question": "Based on the latest connected company data, what is the single most important board-level finance call leadership should make right now, and what conditions would flip that recommendation?",
        },
    ]

    if {"ar_total", "ar_overdue_30_plus", "ap_total", "ap_due_30d"} & recent_metrics:
        templates.append(
            {
                "monitor_id": "working-capital-watch",
                "title": "Working capital watch",
                "description": "Examines receivables, payables, and collections stress.",
                "question": "Using the latest connected company finance data, is working capital creating a near-term risk right now, and what conditions say go, stop, or review?",
            }
        )

    if {"customer_concentration_pct", "top_customer_share_pct", "logo_churn_pct", "nrr_pct"} & recent_metrics:
        templates.append(
            {
                "monitor_id": "concentration-retention-watch",
                "title": "Concentration and retention watch",
                "description": "Flags top-customer exposure, churn, and retention fragility.",
                "question": "Using the latest connected company finance data, is customer concentration or churn becoming a material financial risk right now, and what are the live go, stop, and review conditions?",
            }
        )

    if {"forecast_vs_actual_pct", "pipeline_coverage"} & recent_metrics:
        templates.append(
            {
                "monitor_id": "forecast-reliability-watch",
                "title": "Forecast reliability watch",
                "description": "Tests whether planning assumptions are still holding.",
                "question": "Using the latest connected company finance data, how reliable is the current forecast right now, and under what conditions should the team trust it, stop on it, or review it?",
            }
        )

    if {"debt_service_coverage_ratio", "covenant_headroom_pct", "current_ratio", "quick_ratio"} & recent_metrics:
        templates.append(
            {
                "monitor_id": "debt-covenant-watch",
                "title": "Debt and covenant watch",
                "description": "Monitors covenant headroom, liquidity ratios, and financing risk.",
                "question": "Using the latest connected company finance data, is debt or covenant pressure becoming dangerous right now, and what conditions should trigger stop, review, or proceed decisions?",
            }
        )

    if any("stripe" in label for label in connected_source_labels) or {"failed_payment_rate_pct", "mrr", "arr"} & recent_metrics:
        templates.append(
            {
                "monitor_id": "payments-subscriptions-watch",
                "title": "Payments and subscriptions watch",
                "description": "Looks for failed payment, subscription, and billing risk signals.",
                "question": "Using the latest connected Stripe or subscription finance data, are payments, collections, or subscription trends creating a financial risk right now, and what conditions would trigger action?",
            }
        )

    # Keep the monitoring page dense but still manageable.
    return templates[:6]


def _build_connector_status(
    connectors: list[dict[str, Any]],
    sync_result: dict[str, Any],
) -> list[InsightConnectorStatus]:
    results = (sync_result or {}).get("results") or {}
    statuses: list[InsightConnectorStatus] = []

    for connector in connectors:
        connector_id = str(connector.get("connector_id") or "unknown")
        label = str(connector.get("label") or connector_id)
        enabled = bool(connector.get("enabled", False))
        status = "idle"
        detail = None

        if not enabled:
            status = "disabled"
        elif "stripe" in connector_id and "stripe" in results:
            stripe_result = results.get("stripe") or {}
            if stripe_result.get("error"):
                status = "error"
                detail = str(stripe_result.get("error"))
            else:
                status = "connected"
                detail = "Stripe sync completed."
        elif "quickbooks" in connector_id and "quickbooks" in results:
            qb_result = results.get("quickbooks") or {}
            if qb_result.get("error"):
                status = "error"
                detail = str(qb_result.get("error"))
            else:
                status = "connected"
                detail = "QuickBooks sync completed."
        elif "csv" in connector_id:
            status = "connected"
            detail = "CSV finance stream available."
        elif enabled:
            status = "connected"
            detail = "Connector enabled."

        statuses.append(
            InsightConnectorStatus(
                connector_id=connector_id,
                label=label,
                enabled=enabled,
                status=status,
                detail=detail,
                last_used_utc=connector.get("last_used_utc"),
            )
        )

    return statuses


async def _run_monitor(
    company_id: str,
    base_context: UserContext,
    template: dict[str, str],
    *,
    bypass_cache: bool,
) -> InsightMonitorCard:
    query = template["question"]

    plan = await route(query, company_id, base_context, bypass_cache=bypass_cache)
    scenario_context = await build_finance_snapshot_for_plan(
        company_id=company_id,
        plan=plan,
        ctx=base_context,
    )
    decision = await refine(
        query=query,
        plan=plan,
        company_id=company_id,
        context=scenario_context,
        bypass_cache=bypass_cache,
    )
    payload = format_for_ui(
        decision=decision,
        query=query,
        company_id=company_id,
        context=scenario_context,
    )

    await register_conditions(
        company_id=company_id,
        decision_id=payload.audit.query_hash[:16],
        query=query,
        verdict=decision.verdict_box,
    )

    return InsightMonitorCard(
        monitor_id=template["monitor_id"],
        title=template["title"],
        description=template["description"],
        question=query,
        decision=payload,
        generated_utc=_utc_now_iso(),
    )


async def refresh_company_insights(
    company_id: str,
    *,
    force_refresh: bool = False,
) -> InsightSnapshotResponse:
    lock = _get_lock(company_id)
    async with lock:
        cached = _cache.get(company_id)
        if cached and not force_refresh:
            generated = datetime.fromisoformat(cached.generated_utc)
            if (_utc_now() - generated).total_seconds() < _REFRESH_INTERVAL_SECONDS:
                return cached.model_copy(
                    update={
                        "cached": True,
                        "watch_status": await get_company_watch_status(company_id),
                    }
                )

        t0 = time.perf_counter()
        generated_utc = _utc_now_iso()

        connectors = await get_connector_config(company_id)
        sync_result = await run_company_sync(company_id)
        memory_summary = await get_memory_summary(company_id)
        base_context = await enrich_context(company_id, UserContext())

        recent_data = await get_recent_data(company_id, limit=60)
        templates = _monitor_templates(memory_summary, connectors, recent_data)
        monitors: list[InsightMonitorCard] = []

        for template in templates:
            try:
                monitor = await _run_monitor(
                    company_id=company_id,
                    base_context=base_context,
                    template=template,
                    bypass_cache=force_refresh,
                )
                monitors.append(monitor)
            except Exception as exc:
                logger.error(
                    "Insight monitor failed | company=%s | monitor=%s | %s",
                    company_id,
                    template["monitor_id"],
                    exc,
                )

        condition_report = await get_active_conditions(company_id)
        scan_report = await scan_company_conditions(company_id, base_context)
        recent_data = await get_recent_data(company_id, limit=60)
        connector_status = _build_connector_status(connectors, sync_result)
        connected_sources = _connected_sources(memory_summary)

        overview = InsightOverview(
            company_id=company_id,
            generated_utc=generated_utc,
            next_refresh_utc=_future_iso(_REFRESH_INTERVAL_SECONDS),
            source_count=len(connector_status),
            connected_sources=connected_sources,
            datapoint_count=len(recent_data),
            monitor_count=len(monitors),
            active_conditions=condition_report.total_active,
            fired_conditions=condition_report.total_fired,
            urgent_conditions=len(condition_report.urgent),
        )

        duration_ms = round((time.perf_counter() - t0) * 1000)
        watch_status = await _record_watch_run(
            company_id,
            status="healthy",
            duration_ms=duration_ms,
            reason="refresh",
        )

        response = InsightSnapshotResponse(
            overview=overview,
            connector_status=connector_status,
            sync_result={
                **sync_result,
                "duration_ms": duration_ms,
            },
            recent_data=recent_data,
            conditions=condition_report,
            scan_report=scan_report,
            monitors=monitors,
            watch_status=watch_status,
            generated_utc=generated_utc,
            cached=False,
        )

        _cache[company_id] = response
        return response


async def ask_if_insight(
    company_id: str,
    question: str,
    *,
    force_refresh: bool = False,
) -> InsightAskIfResponse:
    t0 = time.perf_counter()
    # Keep the ask-if flow pointed at the freshest available finance context.
    await run_company_sync(company_id)

    base_context = await enrich_context(company_id, UserContext())
    plan = await route(question, company_id, base_context, bypass_cache=force_refresh)
    scenario_context = await build_finance_snapshot_for_plan(
        company_id=company_id,
        plan=plan,
        ctx=base_context,
    )
    decision = await refine(
        query=question,
        plan=plan,
        company_id=company_id,
        context=scenario_context,
        bypass_cache=force_refresh,
    )
    payload = format_for_ui(
        decision=decision,
        query=question,
        company_id=company_id,
        context=scenario_context,
    )

    await register_conditions(
        company_id=company_id,
        decision_id=payload.audit.query_hash[:16],
        query=question,
        verdict=decision.verdict_box,
    )
    await update_memory(company_id, question, plan, payload)
    await write_audit(
        company_id=company_id,
        query=question,
        plan=plan,
        decision=decision,
        payload=payload,
        user_id=None,
    )

    watch_status = await _record_watch_run(
        company_id,
        status="healthy",
        duration_ms=round((time.perf_counter() - t0) * 1000),
        reason="ask-if",
    )

    return InsightAskIfResponse(
        company_id=company_id,
        query=question,
        decision=payload,
        watch_status=watch_status,
        generated_utc=_utc_now_iso(),
    )


async def run_background_insights_monitor(stop_event: asyncio.Event) -> None:
    logger.info(
        "Insights background monitor started | poll_seconds=%d | registry=%s",
        _MONITOR_POLL_SECONDS,
        _REGISTRY_PATH,
    )

    while not stop_event.is_set():
        try:
            registry = await _load_registry()
            due_companies = [
                company for company in registry.companies if _watch_due_now(company)
            ]

            for company in due_companies:
                started = time.perf_counter()
                try:
                    await refresh_company_insights(company.company_id, force_refresh=True)
                    logger.info(
                        "Insights background refresh ok | company=%s",
                        company.company_id,
                    )
                except Exception as exc:
                    logger.error(
                        "Insights background refresh failed | company=%s | %s",
                        company.company_id,
                        exc,
                    )
                    await _record_watch_run(
                        company.company_id,
                        status="error",
                        duration_ms=round((time.perf_counter() - started) * 1000),
                        error=str(exc)[:500],
                        reason="background",
                        next_due_seconds=_MONITOR_RETRY_SECONDS,
                    )

            await asyncio.wait_for(stop_event.wait(), timeout=_MONITOR_POLL_SECONDS)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Insights background monitor loop error | %s", exc)
            await asyncio.sleep(min(_MONITOR_POLL_SECONDS, _MONITOR_RETRY_SECONDS))

    logger.info("Insights background monitor stopped")
