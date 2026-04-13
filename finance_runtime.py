from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from router import RoutingPlan, UserContext
from marketplace import get_recent_data
from finance_defaults import get_default_kpis_for_stage, build_finance_health_summary

logger = logging.getLogger("think-ai.finance-runtime")

DECISION_METRIC_MAP: dict[str, list[str]] = {
    "liquidity_runway": [
        "cash_balance",
        "available_liquidity",
        "monthly_burn",
        "runway_months",
        "ap_due_30d",
        "current_ratio",
        "quick_ratio",
    ],
    "cash_flow_risk": [
        "cash_balance",
        "available_liquidity",
        "monthly_burn",
        "runway_months",
        "ar_total",
        "ar_overdue_30_plus",
        "ap_due_30d",
    ],
    "revenue_quality": [
        "mrr",
        "arr",
        "revenue_last_30d",
        "revenue_prev_30d",
        "revenue_growth_pct",
        "logo_churn_pct",
        "revenue_churn_pct",
        "nrr_pct",
        "failed_payment_rate_pct",
        "customer_concentration_pct",
        "top_customer_share_pct",
    ],
    "pricing_strategy": [
        "mrr",
        "arr",
        "revenue_last_30d",
        "revenue_growth_pct",
        "gross_margin_pct",
        "nrr_pct",
        "customer_concentration_pct",
    ],
    "margin_pressure": [
        "gross_margin_pct",
        "ebitda_margin_pct",
        "net_margin_pct",
        "opex_last_30d",
        "opex_prev_30d",
        "opex_growth_pct",
        "revenue_growth_pct",
    ],
    "working_capital": [
        "ar_total",
        "ar_overdue_30_plus",
        "ap_total",
        "ap_due_30d",
        "cash_balance",
    ],
    "collections_risk": [
        "ar_total",
        "ar_overdue_30_plus",
        "failed_payment_rate_pct",
        "cash_balance",
    ],
    "customer_concentration": [
        "customer_concentration_pct",
        "top_customer_share_pct",
        "revenue_last_30d",
        "revenue_growth_pct",
    ],
    "forecast_reliability": [
        "forecast_vs_actual_pct",
        "pipeline_coverage",
        "revenue_growth_pct",
        "mrr",
        "arr",
    ],
    "debt_covenants": [
        "debt_service_coverage_ratio",
        "covenant_headroom_pct",
        "current_ratio",
        "quick_ratio",
        "cash_balance",
        "runway_months",
    ],
    "capital_allocation": [
        "cash_balance",
        "runway_months",
        "gross_margin_pct",
        "ebitda_margin_pct",
        "revenue_growth_pct",
    ],
    "unit_economics": [
        "gross_margin_pct",
        "ebitda_margin_pct",
        "net_margin_pct",
        "revenue_growth_pct",
        "nrr_pct",
    ],
    "board_finance": [
        "cash_balance",
        "runway_months",
        "revenue_growth_pct",
        "gross_margin_pct",
        "ar_overdue_30_plus",
        "customer_concentration_pct",
        "forecast_vs_actual_pct",
    ],
    "enterprise_finance": [
        "cash_balance",
        "runway_months",
        "revenue_last_30d",
        "revenue_growth_pct",
        "gross_margin_pct",
        "ar_overdue_30_plus",
        "failed_payment_rate_pct",
        "customer_concentration_pct",
        "forecast_vs_actual_pct",
    ],
}

CORE_FALLBACK_METRICS = [
    "cash_balance",
    "monthly_burn",
    "runway_months",
    "revenue_last_30d",
    "revenue_growth_pct",
    "gross_margin_pct",
    "ar_overdue_30_plus",
    "failed_payment_rate_pct",
]

PERCENT_KEYS = {
    "revenue_growth_pct",
    "gross_margin_pct",
    "ebitda_margin_pct",
    "net_margin_pct",
    "opex_growth_pct",
    "failed_payment_rate_pct",
    "customer_concentration_pct",
    "top_customer_share_pct",
    "logo_churn_pct",
    "revenue_churn_pct",
    "nrr_pct",
    "forecast_vs_actual_pct",
    "covenant_headroom_pct",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_percent(metric_name: str, value: Any) -> float | None:
    num = _to_float(value)
    if num is None:
        return None
    if metric_name in PERCENT_KEYS and 0 <= num <= 1:
        return round(num * 100, 2)
    return round(num, 2)


def _extract_point(point: Any) -> tuple[str | None, Any, str | None, str | None]:
    if hasattr(point, "model_dump"):
        raw = point.model_dump()
    elif isinstance(point, dict):
        raw = point
    else:
        raw = {
            "metric_name": getattr(point, "metric_name", None),
            "value": getattr(point, "value", None),
            "connector_id": getattr(point, "connector_id", None),
            "source_label": getattr(point, "source_label", None),
            "freshness_utc": getattr(point, "freshness_utc", None),
            "fetched_utc": getattr(point, "fetched_utc", None),
        }

    metric_name = raw.get("metric_name")
    value = raw.get("value")
    source = raw.get("source_label") or raw.get("connector_id")
    freshness = raw.get("freshness_utc") or raw.get("fetched_utc")
    return metric_name, value, source, freshness


def _compute_derived_metrics(snapshot: dict[str, Any]) -> dict[str, Any]:
    cash_balance = _to_float(snapshot.get("cash_balance"))
    monthly_burn = _to_float(snapshot.get("monthly_burn"))
    revenue_last_30d = _to_float(snapshot.get("revenue_last_30d"))
    revenue_prev_30d = _to_float(snapshot.get("revenue_prev_30d"))
    opex_last_30d = _to_float(snapshot.get("opex_last_30d"))
    opex_prev_30d = _to_float(snapshot.get("opex_prev_30d"))
    ar_total = _to_float(snapshot.get("ar_total"))
    ar_overdue_30_plus = _to_float(snapshot.get("ar_overdue_30_plus"))
    ap_total = _to_float(snapshot.get("ap_total"))
    ap_due_30d = _to_float(snapshot.get("ap_due_30d"))

    notes: list[str] = list(snapshot.get("evidence_notes", []))

    if snapshot.get("runway_months") is None and cash_balance is not None and monthly_burn and monthly_burn > 0:
        snapshot["runway_months"] = round(cash_balance / monthly_burn, 2)

    if snapshot.get("revenue_growth_pct") is None and revenue_last_30d is not None and revenue_prev_30d not in (None, 0):
        snapshot["revenue_growth_pct"] = round(((revenue_last_30d - revenue_prev_30d) / revenue_prev_30d) * 100, 2)

    if snapshot.get("opex_growth_pct") is None and opex_last_30d is not None and opex_prev_30d not in (None, 0):
        snapshot["opex_growth_pct"] = round(((opex_last_30d - opex_prev_30d) / opex_prev_30d) * 100, 2)

    if snapshot.get("collections_risk_ratio_pct") is None and ar_total not in (None, 0) and ar_overdue_30_plus is not None:
        snapshot["collections_risk_ratio_pct"] = round((ar_overdue_30_plus / ar_total) * 100, 2)

    if snapshot.get("payables_pressure_ratio_pct") is None and ap_total not in (None, 0) and ap_due_30d is not None:
        snapshot["payables_pressure_ratio_pct"] = round((ap_due_30d / ap_total) * 100, 2)

    runway = _to_float(snapshot.get("runway_months"))
    revenue_growth = _to_float(snapshot.get("revenue_growth_pct"))
    concentration = _to_float(snapshot.get("customer_concentration_pct"))
    failed_payment_rate = _to_float(snapshot.get("failed_payment_rate_pct"))
    quick_ratio = _to_float(snapshot.get("quick_ratio"))

    if runway is not None:
        if runway < 6:
            notes.append(f"Runway is critically tight at {runway} months.")
        elif runway < 12:
            notes.append(f"Runway is below 12 months at {runway} months.")

    if revenue_growth is not None and revenue_growth < 0:
        notes.append(f"Revenue growth is negative at {revenue_growth}%.")

    if concentration is not None and concentration >= 35:
        notes.append(f"Customer concentration is elevated at {concentration}%.")

    if failed_payment_rate is not None and failed_payment_rate >= 3:
        notes.append(f"Failed payment rate is elevated at {failed_payment_rate}%.")

    if quick_ratio is not None and quick_ratio < 1:
        notes.append(f"Quick ratio is below 1.0 at {quick_ratio}.")

    snapshot["evidence_notes"] = list(dict.fromkeys(notes))
    return snapshot


def _pick_metric_names(plan: RoutingPlan) -> list[str]:
    metrics = DECISION_METRIC_MAP.get(plan.decision_type, [])
    if not metrics:
        metrics = CORE_FALLBACK_METRICS
    return metrics


async def build_finance_snapshot_for_plan(
    company_id: str,
    plan: RoutingPlan,
    ctx: UserContext | None,
    *,
    recent_limit: int = 200,
) -> UserContext:
    """
    Pull only the most relevant finance metrics for the current decision_type,
    compute derived metrics, add default KPI coverage by company stage,
    and merge them into ctx.extra.
    """
    if ctx is None:
        ctx = UserContext(extra={})
    if ctx.extra is None:
        ctx.extra = {}

    recent_points = await get_recent_data(company_id, limit=recent_limit)

    company_size = getattr(ctx, "company_size", None) if ctx else None
    default_kpis = get_default_kpis_for_stage(company_size)
    desired_metrics = set(_pick_metric_names(plan)) | set(default_kpis)

    latest_by_metric: dict[str, Any] = {}
    sources_used: set[str] = set()
    source_freshness: dict[str, str] = {}

    for point in recent_points:
        metric_name, value, source, freshness = _extract_point(point)
        if not metric_name:
            continue
        if metric_name not in desired_metrics:
            continue
        if metric_name in latest_by_metric:
            continue

        normalized = _normalize_percent(metric_name, value)
        if normalized is None and value is not None:
            normalized = value

        latest_by_metric[metric_name] = normalized

        if source:
            sources_used.add(str(source))
            if freshness:
                source_freshness[str(source)] = str(freshness)

    merged = dict(ctx.extra)
    merged.update(latest_by_metric)

    merged["is_live_data"] = True
    merged["as_of_utc"] = _utc_now_iso()
    merged["analysis_horizon_days"] = getattr(plan, "analysis_horizon_days", merged.get("analysis_horizon_days", 90))
    merged["sources_used"] = sorted(set([*(merged.get("sources_used", []) or []), *sources_used]))
    merged["source_freshness"] = {
        **(merged.get("source_freshness", {}) or {}),
        **source_freshness,
    }

    merged = _compute_derived_metrics(merged)
    merged["health_summary"] = build_finance_health_summary(merged)

    ctx.extra = merged

    logger.info(
        "Finance snapshot runtime built | company=%s | decision_type=%s | metrics_loaded=%d | sources=%s",
        company_id,
        plan.decision_type,
        len(latest_by_metric),
        ",".join(sorted(sources_used)) or "none",
    )
    return ctx