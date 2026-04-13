from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ThresholdBand:
    green: float | None = None
    yellow: float | None = None
    red: float | None = None
    direction: str = "higher_is_better"  # or "lower_is_better"


@dataclass(frozen=True)
class KPIConfig:
    metric_name: str
    label: str
    unit: str
    band: ThresholdBand
    description: str


COMPANY_STAGE_KPI_PACKS: dict[str, list[str]] = {
    "startup": [
        "cash_balance",
        "runway_months",
        "monthly_burn",
        "revenue_growth_pct",
        "gross_margin_pct",
        "failed_payment_rate_pct",
    ],
    "smb": [
        "cash_balance",
        "runway_months",
        "revenue_last_30d",
        "revenue_growth_pct",
        "gross_margin_pct",
        "ar_overdue_30_plus",
        "failed_payment_rate_pct",
        "customer_concentration_pct",
    ],
    "growth": [
        "cash_balance",
        "runway_months",
        "mrr",
        "arr",
        "revenue_growth_pct",
        "gross_margin_pct",
        "nrr_pct",
        "customer_concentration_pct",
        "forecast_vs_actual_pct",
    ],
    "enterprise": [
        "cash_balance",
        "available_liquidity",
        "runway_months",
        "gross_margin_pct",
        "ebitda_margin_pct",
        "ar_overdue_30_plus",
        "customer_concentration_pct",
        "forecast_vs_actual_pct",
        "debt_service_coverage_ratio",
        "covenant_headroom_pct",
    ],
    "default": [
        "cash_balance",
        "runway_months",
        "revenue_growth_pct",
        "gross_margin_pct",
        "failed_payment_rate_pct",
    ],
}


KPI_LIBRARY: dict[str, KPIConfig] = {
    "cash_balance": KPIConfig(
        metric_name="cash_balance",
        label="Cash Balance",
        unit="currency",
        band=ThresholdBand(direction="higher_is_better"),
        description="Total cash currently available.",
    ),
    "available_liquidity": KPIConfig(
        metric_name="available_liquidity",
        label="Available Liquidity",
        unit="currency",
        band=ThresholdBand(direction="higher_is_better"),
        description="Cash plus immediately available liquidity sources.",
    ),
    "monthly_burn": KPIConfig(
        metric_name="monthly_burn",
        label="Monthly Burn",
        unit="currency",
        band=ThresholdBand(direction="lower_is_better"),
        description="Average monthly net cash outflow.",
    ),
    "runway_months": KPIConfig(
        metric_name="runway_months",
        label="Runway",
        unit="months",
        band=ThresholdBand(green=12, yellow=6, red=3, direction="higher_is_better"),
        description="Estimated months before cash runs out at current burn.",
    ),
    "revenue_last_30d": KPIConfig(
        metric_name="revenue_last_30d",
        label="Revenue (30d)",
        unit="currency",
        band=ThresholdBand(direction="higher_is_better"),
        description="Revenue over the last 30 days.",
    ),
    "revenue_growth_pct": KPIConfig(
        metric_name="revenue_growth_pct",
        label="Revenue Growth",
        unit="percent",
        band=ThresholdBand(green=15, yellow=0, red=-10, direction="higher_is_better"),
        description="Recent revenue growth percentage.",
    ),
    "gross_margin_pct": KPIConfig(
        metric_name="gross_margin_pct",
        label="Gross Margin",
        unit="percent",
        band=ThresholdBand(green=60, yellow=40, red=25, direction="higher_is_better"),
        description="Gross margin percentage.",
    ),
    "ebitda_margin_pct": KPIConfig(
        metric_name="ebitda_margin_pct",
        label="EBITDA Margin",
        unit="percent",
        band=ThresholdBand(green=15, yellow=0, red=-15, direction="higher_is_better"),
        description="EBITDA margin percentage.",
    ),
    "net_margin_pct": KPIConfig(
        metric_name="net_margin_pct",
        label="Net Margin",
        unit="percent",
        band=ThresholdBand(green=10, yellow=0, red=-15, direction="higher_is_better"),
        description="Net margin percentage.",
    ),
    "ar_overdue_30_plus": KPIConfig(
        metric_name="ar_overdue_30_plus",
        label="Overdue AR (30+)",
        unit="currency",
        band=ThresholdBand(direction="lower_is_better"),
        description="Accounts receivable overdue by 30+ days.",
    ),
    "failed_payment_rate_pct": KPIConfig(
        metric_name="failed_payment_rate_pct",
        label="Failed Payment Rate",
        unit="percent",
        band=ThresholdBand(green=1.5, yellow=3, red=5, direction="lower_is_better"),
        description="Percentage of failed payments.",
    ),
    "customer_concentration_pct": KPIConfig(
        metric_name="customer_concentration_pct",
        label="Customer Concentration",
        unit="percent",
        band=ThresholdBand(green=20, yellow=35, red=50, direction="lower_is_better"),
        description="Revenue concentration among top customers.",
    ),
    "top_customer_share_pct": KPIConfig(
        metric_name="top_customer_share_pct",
        label="Top Customer Share",
        unit="percent",
        band=ThresholdBand(green=10, yellow=20, red=30, direction="lower_is_better"),
        description="Revenue share of the top single customer.",
    ),
    "logo_churn_pct": KPIConfig(
        metric_name="logo_churn_pct",
        label="Logo Churn",
        unit="percent",
        band=ThresholdBand(green=2, yellow=5, red=8, direction="lower_is_better"),
        description="Percentage of customers lost.",
    ),
    "revenue_churn_pct": KPIConfig(
        metric_name="revenue_churn_pct",
        label="Revenue Churn",
        unit="percent",
        band=ThresholdBand(green=2, yellow=5, red=8, direction="lower_is_better"),
        description="Percentage of revenue lost from churn.",
    ),
    "nrr_pct": KPIConfig(
        metric_name="nrr_pct",
        label="Net Revenue Retention",
        unit="percent",
        band=ThresholdBand(green=110, yellow=100, red=90, direction="higher_is_better"),
        description="Net revenue retention percentage.",
    ),
    "pipeline_coverage": KPIConfig(
        metric_name="pipeline_coverage",
        label="Pipeline Coverage",
        unit="x",
        band=ThresholdBand(green=3, yellow=2, red=1.2, direction="higher_is_better"),
        description="Pipeline coverage relative to target.",
    ),
    "forecast_vs_actual_pct": KPIConfig(
        metric_name="forecast_vs_actual_pct",
        label="Forecast vs Actual",
        unit="percent",
        band=ThresholdBand(green=5, yellow=10, red=20, direction="lower_is_better"),
        description="Absolute variance between forecast and actuals.",
    ),
    "debt_service_coverage_ratio": KPIConfig(
        metric_name="debt_service_coverage_ratio",
        label="DSCR",
        unit="x",
        band=ThresholdBand(green=1.5, yellow=1.2, red=1.0, direction="higher_is_better"),
        description="Debt service coverage ratio.",
    ),
    "current_ratio": KPIConfig(
        metric_name="current_ratio",
        label="Current Ratio",
        unit="x",
        band=ThresholdBand(green=1.5, yellow=1.1, red=1.0, direction="higher_is_better"),
        description="Current assets divided by current liabilities.",
    ),
    "quick_ratio": KPIConfig(
        metric_name="quick_ratio",
        label="Quick Ratio",
        unit="x",
        band=ThresholdBand(green=1.2, yellow=1.0, red=0.8, direction="higher_is_better"),
        description="Quick assets divided by current liabilities.",
    ),
    "covenant_headroom_pct": KPIConfig(
        metric_name="covenant_headroom_pct",
        label="Covenant Headroom",
        unit="percent",
        band=ThresholdBand(green=20, yellow=10, red=5, direction="higher_is_better"),
        description="Buffer before debt covenant breach.",
    ),
}


DEFAULT_FINANCE_ALERTS: list[dict[str, Any]] = [
    {
        "kind": "stop",
        "text": "Runway below 6 months",
        "metric_targets": ["runway_months"],
        "threshold": 6,
        "operator": "lt",
        "priority": "critical",
    },
    {
        "kind": "review",
        "text": "Failed payment rate above 3%",
        "metric_targets": ["failed_payment_rate_pct"],
        "threshold": 3,
        "operator": "gt",
        "priority": "high",
    },
    {
        "kind": "review",
        "text": "Customer concentration above 35%",
        "metric_targets": ["customer_concentration_pct"],
        "threshold": 35,
        "operator": "gt",
        "priority": "high",
    },
    {
        "kind": "review",
        "text": "Quick ratio below 1.0",
        "metric_targets": ["quick_ratio"],
        "threshold": 1.0,
        "operator": "lt",
        "priority": "high",
    },
    {
        "kind": "review",
        "text": "Covenant headroom below 10%",
        "metric_targets": ["covenant_headroom_pct"],
        "threshold": 10,
        "operator": "lt",
        "priority": "high",
    },
    {
        "kind": "review",
        "text": "Revenue growth below 0%",
        "metric_targets": ["revenue_growth_pct"],
        "threshold": 0,
        "operator": "lt",
        "priority": "medium",
    },
]


def infer_company_stage(company_size: str | None) -> str:
    if not company_size:
        return "default"

    value = company_size.strip().lower()

    if any(x in value for x in ["startup", "seed", "pre-seed", "early"]):
        return "startup"
    if any(x in value for x in ["small", "smb", "business"]):
        return "smb"
    if any(x in value for x in ["growth", "scale", "mid-market", "series"]):
        return "growth"
    if any(x in value for x in ["enterprise", "large", "public"]):
        return "enterprise"

    return "default"


def get_default_kpis_for_stage(company_size: str | None) -> list[str]:
    stage = infer_company_stage(company_size)
    return COMPANY_STAGE_KPI_PACKS.get(stage, COMPANY_STAGE_KPI_PACKS["default"])


def get_kpi_config(metric_name: str) -> KPIConfig | None:
    return KPI_LIBRARY.get(metric_name)


def evaluate_band(metric_name: str, value: float | int | None) -> str | None:
    if value is None:
        return None

    config = KPI_LIBRARY.get(metric_name)
    if not config:
        return None

    band = config.band
    v = float(value)

    if band.direction == "higher_is_better":
        if band.green is not None and v >= band.green:
            return "green"
        if band.yellow is not None and v >= band.yellow:
            return "yellow"
        return "red"

    if band.direction == "lower_is_better":
        if band.green is not None and v <= band.green:
            return "green"
        if band.yellow is not None and v <= band.yellow:
            return "yellow"
        return "red"

    return None


def build_finance_health_summary(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for metric_name, config in KPI_LIBRARY.items():
        value = snapshot.get(metric_name)
        if value is None:
            continue

        out.append(
            {
                "metric_name": metric_name,
                "label": config.label,
                "value": value,
                "unit": config.unit,
                "status": evaluate_band(metric_name, value),
                "description": config.description,
            }
        )

    return out