"""
marketplace.py — Three AI Finance Connectors
Storage backend: Supabase
Finance-first canonical data normalization.
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
import json
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, runtime_checkable

import httpx
from cryptography.fernet import Fernet, InvalidToken
from openai import AsyncOpenAI, APIStatusError, APITimeoutError
from pydantic import BaseModel, Field, field_validator

from router import UserContext
from conditions import (
    get_active_conditions,
    fire_condition,
    request_reanalysis,
    ConditionRecord,
)
from db import read_store, write_store, delete_store

_TABLE                = "marketplace_store"
_LOCK_TIMEOUT_SEC     = float(os.getenv("THINK_AI_MARKETPLACE_LOCK", "5"))
_MAX_RECORDS          = int(os.getenv("THINK_AI_MARKETPLACE_MAX", "1000"))
_HTTP_TIMEOUT_SEC     = float(os.getenv("THINK_AI_CONNECTOR_TIMEOUT", "10"))
_MAX_RESPONSE_BYTES   = int(os.getenv("THINK_AI_CONNECTOR_MAX_BYTES", "65536"))
_EVAL_MODEL           = os.getenv("THINK_AI_MARKETPLACE_MODEL", "gpt-4o-mini")
_EVAL_TIMEOUT         = float(os.getenv("THINK_AI_MARKETPLACE_TIMEOUT", "20"))
_MAX_RETRIES          = int(os.getenv("THINK_AI_REFINER_RETRIES", "3"))
_MARKETPLACE_VERSION  = "2.0.0-finance"
_MAX_LLM_CONTEXT_CHARS = 4000
_ALLOWED_SCHEMES = {"https", "http"}

_ENCRYPTION_KEY = os.getenv("THINK_AI_ENCRYPTION_KEY", "")
_fernet: Fernet | None = None
if _ENCRYPTION_KEY:
    try:
        _fernet = Fernet(_ENCRYPTION_KEY.encode())
    except Exception:
        pass

_llm_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        load_dotenv()
        _llm_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            timeout=_EVAL_TIMEOUT,
            max_retries=0,
        )
    return _llm_client


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("three-ai.marketplace")


def _encrypt(value: str) -> str:
    if not _fernet:
        logger.warning("THINK_AI_ENCRYPTION_KEY not set — API key stored in plaintext.")
        return value
    return _fernet.encrypt(value.encode()).decode()


def _decrypt(value: str) -> str:
    if not _fernet:
        return value
    try:
        return _fernet.decrypt(value.encode()).decode()
    except InvalidToken:
        logger.error("Failed to decrypt connector API key")
        return ""


ConnectorKind = Literal["http", "webhook", "mock"]
FinanceDataKind = Literal["metric", "ratio", "currency_amount", "count", "flag", "text"]

CANONICAL_FINANCE_METRICS = {
    "cash_balance",
    "available_liquidity",
    "monthly_burn",
    "runway_months",
    "mrr",
    "arr",
    "revenue_last_30d",
    "revenue_prev_30d",
    "revenue_growth_pct",
    "gross_margin_pct",
    "ebitda_margin_pct",
    "net_margin_pct",
    "opex_last_30d",
    "opex_prev_30d",
    "opex_growth_pct",
    "ar_total",
    "ar_overdue_30_plus",
    "ap_total",
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
    "headcount",
}

METRIC_ALIASES: dict[str, str] = {
    "cash": "cash_balance",
    "cash_balance": "cash_balance",
    "bank_balance": "cash_balance",
    "ending_cash": "cash_balance",
    "available_liquidity": "available_liquidity",
    "liquidity": "available_liquidity",
    "monthly_burn": "monthly_burn",
    "burn": "monthly_burn",
    "runway": "runway_months",
    "runway_months": "runway_months",
    "mrr": "mrr",
    "arr": "arr",
    "revenue": "revenue_last_30d",
    "revenue_last_30d": "revenue_last_30d",
    "revenue_prev_30d": "revenue_prev_30d",
    "revenue_growth": "revenue_growth_pct",
    "revenue_growth_pct": "revenue_growth_pct",
    "gross_margin": "gross_margin_pct",
    "gross_margin_pct": "gross_margin_pct",
    "ebitda_margin": "ebitda_margin_pct",
    "ebitda_margin_pct": "ebitda_margin_pct",
    "net_margin": "net_margin_pct",
    "net_margin_pct": "net_margin_pct",
    "opex": "opex_last_30d",
    "opex_last_30d": "opex_last_30d",
    "opex_prev_30d": "opex_prev_30d",
    "opex_growth": "opex_growth_pct",
    "opex_growth_pct": "opex_growth_pct",
    "ar_total": "ar_total",
    "accounts_receivable": "ar_total",
    "ar_overdue": "ar_overdue_30_plus",
    "ar_overdue_30_plus": "ar_overdue_30_plus",
    "ap_total": "ap_total",
    "accounts_payable": "ap_total",
    "ap_due_30d": "ap_due_30d",
    "failed_payments": "failed_payment_rate_pct",
    "failed_payment_rate": "failed_payment_rate_pct",
    "failed_payment_rate_pct": "failed_payment_rate_pct",
    "customer_concentration": "customer_concentration_pct",
    "customer_concentration_pct": "customer_concentration_pct",
    "top_customer_share": "top_customer_share_pct",
    "top_customer_share_pct": "top_customer_share_pct",
    "logo_churn": "logo_churn_pct",
    "logo_churn_pct": "logo_churn_pct",
    "revenue_churn": "revenue_churn_pct",
    "revenue_churn_pct": "revenue_churn_pct",
    "nrr": "nrr_pct",
    "nrr_pct": "nrr_pct",
    "pipeline_coverage": "pipeline_coverage",
    "forecast_vs_actual": "forecast_vs_actual_pct",
    "forecast_vs_actual_pct": "forecast_vs_actual_pct",
    "debt_service_coverage_ratio": "debt_service_coverage_ratio",
    "current_ratio": "current_ratio",
    "quick_ratio": "quick_ratio",
    "covenant_headroom": "covenant_headroom_pct",
    "covenant_headroom_pct": "covenant_headroom_pct",
    "headcount": "headcount",
}

FINANCE_METRIC_HINTS: dict[str, list[str]] = {
    "liquidity_runway": ["cash_balance", "available_liquidity", "monthly_burn", "runway_months"],
    "cash_flow_risk": ["cash_balance", "available_liquidity", "monthly_burn", "ap_due_30d", "ar_overdue_30_plus"],
    "revenue_quality": ["mrr", "arr", "revenue_last_30d", "revenue_growth_pct", "nrr_pct", "revenue_churn_pct"],
    "margin_pressure": ["gross_margin_pct", "ebitda_margin_pct", "net_margin_pct", "opex_growth_pct"],
    "working_capital": ["ar_total", "ar_overdue_30_plus", "ap_total", "ap_due_30d"],
    "collections_risk": ["ar_total", "ar_overdue_30_plus", "failed_payment_rate_pct"],
    "customer_concentration": ["customer_concentration_pct", "top_customer_share_pct"],
    "forecast_reliability": ["forecast_vs_actual_pct", "pipeline_coverage", "revenue_growth_pct"],
    "debt_covenants": ["debt_service_coverage_ratio", "covenant_headroom_pct", "current_ratio", "quick_ratio"],
    "unit_economics": ["gross_margin_pct", "ebitda_margin_pct", "revenue_growth_pct"],
    "board_finance": ["cash_balance", "runway_months", "revenue_growth_pct", "gross_margin_pct", "customer_concentration_pct"],
    "enterprise_finance": ["cash_balance", "runway_months", "revenue_growth_pct", "gross_margin_pct", "ar_overdue_30_plus"],
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_metric_name(name: str | None) -> str | None:
    if not name:
        return None
    clean = str(name).strip().lower().replace(" ", "_")
    return METRIC_ALIASES.get(clean, clean if clean in CANONICAL_FINANCE_METRICS else None)


def _infer_data_kind(metric_name: str) -> FinanceDataKind:
    if metric_name in {
        "cash_balance", "available_liquidity", "monthly_burn", "mrr", "arr",
        "revenue_last_30d", "revenue_prev_30d", "opex_last_30d", "opex_prev_30d",
        "ar_total", "ar_overdue_30_plus", "ap_total", "ap_due_30d",
    }:
        return "currency_amount"
    if metric_name in {
        "gross_margin_pct", "ebitda_margin_pct", "net_margin_pct", "revenue_growth_pct",
        "opex_growth_pct", "failed_payment_rate_pct", "customer_concentration_pct",
        "top_customer_share_pct", "logo_churn_pct", "revenue_churn_pct",
        "nrr_pct", "forecast_vs_actual_pct", "covenant_headroom_pct",
    }:
        return "ratio"
    if metric_name in {"headcount"}:
        return "count"
    return "metric"


def _to_number(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    try:
        num = float(value)
        if num.is_integer():
            return int(num)
        return num
    except (TypeError, ValueError):
        return None


def _normalize_metric_value(metric_name: str, value: Any) -> Any:
    num = _to_number(value)
    if num is None:
        return value

    if metric_name == "headcount":
        return int(num)

    if metric_name.endswith("_pct") and 0 <= float(num) <= 1:
        return round(float(num) * 100, 2)

    return round(float(num), 2) if isinstance(num, float) else num


class ConnectorConfig(BaseModel):
    connector_id: str = Field(min_length=1, max_length=64)
    kind: ConnectorKind
    label: str = Field(max_length=128)
    base_url: str | None = None
    api_key: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    metric_paths: dict[str, str] = Field(default_factory=dict)
    metric_aliases: dict[str, str] = Field(default_factory=dict)
    default_currency: str | None = "USD"
    default_period: str | None = "monthly"
    enabled: bool = True
    created_utc: str = Field(default_factory=_utc_now_iso)
    last_used_utc: str | None = None

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v):
        if v not in ("http", "webhook", "mock"):
            raise ValueError("kind must be http, webhook, or mock")
        return v

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, v):
        if v is None:
            return v
        from urllib.parse import urlparse
        if urlparse(v).scheme not in _ALLOWED_SCHEMES:
            raise ValueError("URL scheme not allowed")
        return v

    def safe_dict(self) -> dict:
        d = self.model_dump()
        if d.get("api_key"):
            d["api_key"] = "***"
        return d


class DataQuery(BaseModel):
    metric_name: str
    condition_text: str
    extra_params: dict[str, Any] = Field(default_factory=dict)


class DataPoint(BaseModel):
    connector_id: str
    source_label: str | None = None
    metric_name: str
    value: Any
    value_str: str
    kind: FinanceDataKind = "metric"
    currency: str | None = None
    period: str | None = None
    fetched_utc: str
    recorded_utc: str | None = None
    freshness_utc: str | None = None
    source_url: str | None = None


class ConditionEvaluation(BaseModel):
    condition_id: str
    condition_text: str
    kind: str
    data_points: list[DataPoint]
    assessment: Literal["fired", "clear", "insufficient_data"]
    reasoning: str
    fired: bool
    reanalysis_triggered: bool = False
    evaluated_utc: str


class ScanReport(BaseModel):
    company_id: str
    conditions_scanned: int
    conditions_fired: int
    conditions_clear: int
    insufficient_data: int
    reanalysis_triggered: int
    evaluations: list[ConditionEvaluation]
    scan_duration_ms: int
    scanned_utc: str


class CompanyMarketplaceStore(BaseModel):
    company_id_hash: str
    version: str = _MARKETPLACE_VERSION
    connectors: list[ConnectorConfig] = Field(default_factory=list)
    recent_data: list[DataPoint] = Field(default_factory=list)
    last_scan_utc: str | None = None
    last_updated_utc: str


class WebhookPayload(BaseModel):
    metric_name: str = Field(min_length=1, max_length=128)
    value: Any
    value_str: str | None = Field(default=None, max_length=500)
    kind: FinanceDataKind | None = None
    currency: str | None = None
    period: str | None = None
    recorded_utc: str | None = None
    freshness_utc: str | None = None
    source_label: str | None = None


@runtime_checkable
class Connector(Protocol):
    async def fetch(self, query: DataQuery) -> DataPoint | None: ...


class HttpConnector:
    def __init__(self, config: ConnectorConfig):
        self.config = config

    def _extract_by_path(self, data: Any, path: str) -> Any:
        current = data
        for key in path.split("."):
            if isinstance(current, list):
                try:
                    current = current[int(key)]
                    continue
                except Exception:
                    return None
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _resolve_metric_name(self, raw_metric_name: str) -> str | None:
        alias_name = self.config.metric_aliases.get(raw_metric_name, raw_metric_name)
        return _normalize_metric_name(alias_name)

    async def fetch(self, query: DataQuery) -> DataPoint | None:
        if not self.config.base_url:
            return None

        headers = dict(self.config.headers)
        api_key = _decrypt(self.config.api_key) if self.config.api_key else None
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SEC) as client:
                response = await client.get(
                    self.config.base_url.rstrip("/"),
                    headers=headers,
                    params=query.extra_params or {},
                )
                response.raise_for_status()
                if len(response.content) > _MAX_RESPONSE_BYTES:
                    return None
                data = response.json()
        except Exception as e:
            logger.warning("Connector fetch error | %s | %s", self.config.connector_id, e)
            return None

        metric_name = self._resolve_metric_name(query.metric_name)
        if not metric_name:
            return None

        json_path = self.config.metric_paths.get(query.metric_name) or self.config.metric_paths.get(metric_name)
        value = self._extract_by_path(data, json_path) if json_path else data
        if value is None:
            return None

        normalized_value = _normalize_metric_value(metric_name, value)

        return DataPoint(
            connector_id=self.config.connector_id,
            source_label=self.config.label,
            metric_name=metric_name,
            value=normalized_value,
            value_str=str(normalized_value)[:500],
            kind=_infer_data_kind(metric_name),
            currency=self.config.default_currency,
            period=self.config.default_period,
            fetched_utc=_utc_now_iso(),
            recorded_utc=None,
            freshness_utc=_utc_now_iso(),
            source_url=self.config.base_url,
        )


class WebhookConnector:
    def __init__(self, config: ConnectorConfig, store: CompanyMarketplaceStore):
        self.config = config
        self.store = store

    async def fetch(self, query: DataQuery) -> DataPoint | None:
        normalized_metric = _normalize_metric_name(query.metric_name)
        if not normalized_metric:
            return None

        candidates = [
            d for d in reversed(self.store.recent_data)
            if d.connector_id == self.config.connector_id and d.metric_name == normalized_metric
        ]
        return candidates[0] if candidates else None


class MockConnector:
    def __init__(self, config: ConnectorConfig):
        self.config = config

    async def fetch(self, query: DataQuery) -> DataPoint | None:
        metric_name = _normalize_metric_name(query.metric_name)
        if not metric_name:
            return None

        value_str = self.config.metric_paths.get(query.metric_name, "unavailable")
        return DataPoint(
            connector_id=self.config.connector_id,
            source_label=self.config.label,
            metric_name=metric_name,
            value=value_str,
            value_str=str(value_str)[:500],
            kind=_infer_data_kind(metric_name),
            currency=self.config.default_currency,
            period=self.config.default_period,
            fetched_utc=_utc_now_iso(),
            recorded_utc=None,
            freshness_utc=_utc_now_iso(),
            source_url=None,
        )


def _build_connector(config: ConnectorConfig, store: CompanyMarketplaceStore) -> Connector:
    if config.kind == "http":
        return HttpConnector(config)
    if config.kind == "webhook":
        return WebhookConnector(config, store)
    return MockConnector(config)


_write_locks: dict[str, asyncio.Lock] = {}


def _get_lock(company_id: str) -> asyncio.Lock:
    if company_id not in _write_locks:
        _write_locks[company_id] = asyncio.Lock()
    return _write_locks[company_id]


def _company_hash(company_id: str) -> str:
    return hashlib.sha256(company_id.encode()).hexdigest()


async def _read_store(company_id: str) -> CompanyMarketplaceStore | None:
    raw = await read_store(_TABLE, _company_hash(company_id))
    if raw is None:
        return None
    try:
        return CompanyMarketplaceStore.model_validate(raw)
    except Exception as e:
        logger.error("Marketplace parse error | %s", e)
        return None


async def _write_store(company_id: str, store: CompanyMarketplaceStore) -> bool:
    if len(store.recent_data) > _MAX_RECORDS:
        store.recent_data = store.recent_data[-_MAX_RECORDS:]
    return await write_store(_TABLE, _company_hash(company_id), store.model_dump())


def _new_store(company_id: str) -> CompanyMarketplaceStore:
    return CompanyMarketplaceStore(
        company_id_hash=_company_hash(company_id),
        connectors=[],
        recent_data=[],
        last_updated_utc=_utc_now_iso(),
    )


_METRIC_SYSTEM = "You are Three AI's Finance Metric Scout. Return valid JSON only."
_EVAL_SYSTEM = "You are Three AI's Finance Condition Evaluator. Return valid JSON only."


class _MetricQuery(BaseModel):
    metrics: list[str]
    reasoning: str


class _EvalOutput(BaseModel):
    assessment: Literal["fired", "clear", "insufficient_data"]
    reasoning: str
    confidence: Literal["high", "medium", "low"]


def _deterministic_metrics_from_condition(condition_text: str) -> list[str]:
    text = condition_text.lower()
    found: list[str] = []

    keyword_map = {
        "runway": ["cash_balance", "monthly_burn", "runway_months"],
        "cash": ["cash_balance", "available_liquidity"],
        "liquidity": ["cash_balance", "available_liquidity", "current_ratio", "quick_ratio"],
        "burn": ["monthly_burn", "cash_balance"],
        "revenue": ["revenue_last_30d", "revenue_growth_pct", "mrr", "arr"],
        "mrr": ["mrr"],
        "arr": ["arr"],
        "margin": ["gross_margin_pct", "ebitda_margin_pct", "net_margin_pct"],
        "collections": ["ar_total", "ar_overdue_30_plus", "failed_payment_rate_pct"],
        "receivable": ["ar_total", "ar_overdue_30_plus"],
        "receivables": ["ar_total", "ar_overdue_30_plus"],
        "ar": ["ar_total", "ar_overdue_30_plus"],
        "ap": ["ap_total", "ap_due_30d"],
        "working capital": ["ar_total", "ap_total", "ap_due_30d"],
        "failed payment": ["failed_payment_rate_pct"],
        "churn": ["revenue_churn_pct", "logo_churn_pct", "nrr_pct"],
        "nrr": ["nrr_pct"],
        "customer concentration": ["customer_concentration_pct", "top_customer_share_pct"],
        "top customer": ["top_customer_share_pct", "customer_concentration_pct"],
        "forecast": ["forecast_vs_actual_pct", "pipeline_coverage"],
        "pipeline": ["pipeline_coverage"],
        "debt": ["debt_service_coverage_ratio", "covenant_headroom_pct"],
        "covenant": ["debt_service_coverage_ratio", "covenant_headroom_pct"],
        "headcount": ["headcount"],
    }

    for keyword, metrics in keyword_map.items():
        if keyword in text:
            for metric in metrics:
                if metric not in found:
                    found.append(metric)

    return found[:5]


async def _identify_metrics(condition_text: str) -> list[str]:
    deterministic = _deterministic_metrics_from_condition(condition_text)
    if deterministic:
        return deterministic

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = await _get_client().chat.completions.create(
                model=_EVAL_MODEL,
                messages=[
                    {"role": "system", "content": _METRIC_SYSTEM + "\n\nChoose canonical finance metric names only. Return valid JSON only."},
                    {"role": "user", "content": f"Condition: {condition_text}\n\nAllowed metrics: {sorted(CANONICAL_FINANCE_METRICS)}"},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = (response.choices[0].message.content or "").strip()
            
            # Bulletproof Markdown Stripping
            if raw.startswith("```json"):
                raw = raw[7:]
            elif raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            
            parsed = _MetricQuery.model_validate_json(raw)
            out: list[str] = []
            for m in parsed.metrics:
                nm = _normalize_metric_name(m)
                if nm and nm not in out:
                    out.append(nm)
            return out[:5]
        except (APITimeoutError, APIStatusError) as e:
            status_code = getattr(e, "status_code", None)
            if status_code and status_code not in (429, 500, 502, 503, 504):
                raise
            last_error = e
            await asyncio.sleep((2 ** attempt) + random.uniform(0, 0.3))
        except Exception as e:
            last_error = e
            break

    logger.warning("Metric identification fallback | %s", last_error)
    return []


def _finance_condition_eval_deterministic(condition: ConditionRecord, data_points: list[DataPoint]) -> _EvalOutput | None:
    if not data_points:
        return None

    values = {dp.metric_name: dp.value for dp in data_points}
    text = condition.text.lower()

    def num(metric: str) -> float | None:
        try:
            val = values.get(metric)
            if val is None or val == "":
                return None
            return float(val)
        except Exception:
            return None

    runway = num("runway_months")
    if runway is not None and "runway" in text:
        if "less than" in text or "below" in text or "<" in text:
            import re
            m = re.search(r"(\d+(\.\d+)?)", text)
            if m:
                threshold = float(m.group(1))
                fired = runway < threshold
                return _EvalOutput(
                    assessment="fired" if fired else "clear",
                    reasoning=f"Runway is {runway} months versus threshold {threshold}.",
                    confidence="high",
                )

    failed_rate = num("failed_payment_rate_pct")
    if failed_rate is not None and "failed payment" in text:
        import re
        m = re.search(r"(\d+(\.\d+)?)", text)
        if m:
            threshold = float(m.group(1))
            fired = failed_rate > threshold
            return _EvalOutput(
                assessment="fired" if fired else "clear",
                reasoning=f"Failed payment rate is {failed_rate}% versus threshold {threshold}%.",
                confidence="high",
            )

    overdue_ar = num("ar_overdue_30_plus")
    total_ar = num("ar_total")
    if overdue_ar is not None and total_ar not in (None, 0) and ("overdue" in text or "collections" in text):
        overdue_pct = (overdue_ar / total_ar) * 100
        import re
        m = re.search(r"(\d+(\.\d+)?)", text)
        if m:
            threshold = float(m.group(1))
            fired = overdue_pct > threshold
            return _EvalOutput(
                assessment="fired" if fired else "clear",
                reasoning=f"Overdue AR is {round(overdue_pct, 2)}% of total AR versus threshold {threshold}%.",
                confidence="high",
            )

    return None


async def _evaluate_condition(condition: ConditionRecord, data_points: list[DataPoint]) -> _EvalOutput:
    deterministic = _finance_condition_eval_deterministic(condition, data_points)
    if deterministic is not None:
        return deterministic

    if not data_points:
        return _EvalOutput(
            assessment="insufficient_data",
            reasoning="No data points available.",
            confidence="low",
        )

    data_str = "\n".join(
        f"- {dp.metric_name} = {dp.value_str} ({dp.kind}, source={dp.connector_id}, freshness={dp.freshness_utc or dp.fetched_utc})"
        for dp in data_points
    )[:_MAX_LLM_CONTEXT_CHARS]

    user_message = (
        f"Condition ({condition.kind.upper()}): {condition.text}\n\n"
        f"Live finance data:\n{data_str}\n\n"
        f"Decide whether the condition has fired."
    )

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = await _get_client().chat.completions.create(
                model=_EVAL_MODEL,
                messages=[
                    {"role": "system", "content": _EVAL_SYSTEM + "\n\nRespond with valid JSON only."},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = (response.choices[0].message.content or "").strip()
            
            # Bulletproof Markdown Stripping
            if raw.startswith("```json"):
                raw = raw[7:]
            elif raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            
            return _EvalOutput.model_validate_json(raw)
        except (APITimeoutError, APIStatusError) as e:
            status_code = getattr(e, "status_code", None)
            if status_code and status_code not in (429, 500, 502, 503, 504):
                raise
            last_error = e
            await asyncio.sleep((2 ** attempt) + random.uniform(0, 0.3))
        except Exception as e:
            last_error = e
            break

    logger.warning("Condition evaluation fallback | %s", last_error)
    return _EvalOutput(
        assessment="insufficient_data",
        reasoning=f"Evaluation failed after {_MAX_RETRIES} retries.",
        confidence="low",
    )


async def _scan_one_condition(
    company_id: str,
    condition: ConditionRecord,
    connectors: list[ConnectorConfig],
    store: CompanyMarketplaceStore,
) -> ConditionEvaluation:
    now = _utc_now_iso()

    if condition.status in ("fired", "cleared", "expired", "acknowledged"):
        return ConditionEvaluation(
            condition_id=condition.condition_id,
            condition_text=condition.text,
            kind=condition.kind,
            data_points=[],
            assessment="clear",
            reasoning=f"Already '{condition.status}' — skipped.",
            fired=False,
            evaluated_utc=now,
        )

    enabled = [c for c in connectors if c.enabled]
    if not enabled:
        return ConditionEvaluation(
            condition_id=condition.condition_id,
            condition_text=condition.text,
            kind=condition.kind,
            data_points=[],
            assessment="insufficient_data",
            reasoning="No enabled connectors.",
            fired=False,
            evaluated_utc=now,
        )

    metric_names = await _identify_metrics(condition.text)
    if not metric_names:
        return ConditionEvaluation(
            condition_id=condition.condition_id,
            condition_text=condition.text,
            kind=condition.kind,
            data_points=[],
            assessment="insufficient_data",
            reasoning="Could not identify finance metrics.",
            fired=False,
            evaluated_utc=now,
        )

    fetch_tasks = [
        _build_connector(c, store).fetch(DataQuery(metric_name=m, condition_text=condition.text))
        for c in enabled for m in metric_names
    ]

    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    data_points = [r for r in results if isinstance(r, DataPoint)]

    eval_result = await _evaluate_condition(condition, data_points)
    fired = eval_result.assessment == "fired"
    reanalysis = False

    if fired:
        await fire_condition(company_id, condition.condition_id)
        if condition.kind == "stop":
            await request_reanalysis(company_id, condition.condition_id)
            reanalysis = True

    return ConditionEvaluation(
        condition_id=condition.condition_id,
        condition_text=condition.text,
        kind=condition.kind,
        data_points=data_points,
        assessment=eval_result.assessment,
        reasoning=eval_result.reasoning,
        fired=fired,
        reanalysis_triggered=reanalysis,
        evaluated_utc=now,
    )


# ── PUBLIC API ────────────────────────────────────────────────────────────

async def register_connector(company_id: str, config: ConnectorConfig) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        store = await _read_store(company_id) or _new_store(company_id)
        encrypted = config.model_copy()
        if config.api_key:
            encrypted.api_key = _encrypt(config.api_key)

        store.connectors = [c for c in store.connectors if c.connector_id != config.connector_id]
        store.connectors.append(encrypted)
        store.last_updated_utc = _utc_now_iso()

        success = await _write_store(company_id, store)
        if success:
            logger.info("✅ Connector registered | %s | %s", _company_hash(company_id)[:12], config.connector_id)
        return success
    except Exception as e:
        logger.error("register_connector error | %s", e)
        return False
    finally:
        lock.release()


async def scan_company_conditions(company_id: str, context: UserContext | None = None) -> ScanReport:
    t0 = time.perf_counter()
    now = _utc_now_iso()

    store = await _read_store(company_id)
    if not store or not store.connectors:
        return ScanReport(
            company_id=company_id,
            conditions_scanned=0,
            conditions_fired=0,
            conditions_clear=0,
            insufficient_data=0,
            reanalysis_triggered=0,
            evaluations=[],
            scan_duration_ms=0,
            scanned_utc=now,
        )

    condition_report = await get_active_conditions(company_id)
    all_conditions = condition_report.go_conditions + condition_report.needs_review + condition_report.urgent

    seen: set[str] = set()
    unique = [c for c in all_conditions if not (c.condition_id in seen or seen.add(c.condition_id))]

    if not unique:
        return ScanReport(
            company_id=company_id,
            conditions_scanned=0,
            conditions_fired=0,
            conditions_clear=0,
            insufficient_data=0,
            reanalysis_triggered=0,
            evaluations=[],
            scan_duration_ms=round((time.perf_counter() - t0) * 1000),
            scanned_utc=now,
        )

    evaluations: list[ConditionEvaluation] = []
    for condition in unique:
        evaluation = await _scan_one_condition(company_id, condition, store.connectors, store)
        evaluations.append(evaluation)
        if evaluation.data_points:
            store.recent_data.extend(evaluation.data_points)

    store.last_scan_utc = now
    store.last_updated_utc = now
    await _write_store(company_id, store)

    fired_count = sum(1 for e in evaluations if e.fired)
    duration_ms = round((time.perf_counter() - t0) * 1000)

    logger.info(
        "✅ Scan complete | %s | scanned=%d | fired=%d | duration_ms=%d",
        _company_hash(company_id)[:12], len(evaluations), fired_count, duration_ms
    )

    return ScanReport(
        company_id=company_id,
        conditions_scanned=len(evaluations),
        conditions_fired=fired_count,
        conditions_clear=sum(1 for e in evaluations if e.assessment == "clear" and not e.fired),
        insufficient_data=sum(1 for e in evaluations if e.assessment == "insufficient_data"),
        reanalysis_triggered=sum(1 for e in evaluations if e.reanalysis_triggered),
        evaluations=evaluations,
        scan_duration_ms=duration_ms,
        scanned_utc=now,
    )


async def receive_webhook(company_id: str, connector_id: str, payload: WebhookPayload) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        store = await _read_store(company_id)
        if not store:
            return False

        connector = next(
            (c for c in store.connectors if c.connector_id == connector_id and c.kind == "webhook"),
            None,
        )
        if not connector:
            return False

        metric_name = _normalize_metric_name(payload.metric_name)
        if not metric_name:
            logger.warning("Rejected webhook metric | unknown metric_name=%s", payload.metric_name)
            return False

        normalized_value = _normalize_metric_value(metric_name, payload.value)
        value_str = payload.value_str or str(normalized_value)

        store.recent_data.append(
            DataPoint(
                connector_id=connector_id,
                source_label=payload.source_label or connector.label,
                metric_name=metric_name,
                value=normalized_value,
                value_str=str(value_str)[:500],
                kind=payload.kind or _infer_data_kind(metric_name),
                currency=payload.currency or connector.default_currency,
                period=payload.period or connector.default_period,
                fetched_utc=_utc_now_iso(),
                recorded_utc=payload.recorded_utc,
                freshness_utc=payload.freshness_utc or _utc_now_iso(),
                source_url=connector.base_url,
            )
        )

        connector.last_used_utc = _utc_now_iso()
        store.last_updated_utc = _utc_now_iso()
        return await _write_store(company_id, store)
    except Exception as e:
        logger.error("receive_webhook error | %s", e)
        return False
    finally:
        lock.release()


async def get_connector_config(company_id: str) -> list[dict]:
    store = await _read_store(company_id)
    return [c.safe_dict() for c in store.connectors] if store else []


async def get_recent_data(company_id: str, limit: int = 50) -> list[DataPoint]:
    store = await _read_store(company_id)
    return list(reversed(store.recent_data[-limit:])) if store else []


async def _toggle_connector(company_id: str, connector_id: str, enabled: bool) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        store = await _read_store(company_id)
        if not store:
            return False

        for c in store.connectors:
            if c.connector_id == connector_id:
                c.enabled = enabled
                store.last_updated_utc = _utc_now_iso()
                return await _write_store(company_id, store)
        return False
    except Exception as e:
        logger.error("_toggle_connector error | %s", e)
        return False
    finally:
        lock.release()


async def disable_connector(company_id: str, connector_id: str) -> bool:
    return await _toggle_connector(company_id, connector_id, False)


async def enable_connector(company_id: str, connector_id: str) -> bool:
    return await _toggle_connector(company_id, connector_id, True)


async def delete_connector(company_id: str, connector_id: str) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        store = await _read_store(company_id)
        if not store:
            return False

        before = len(store.connectors)
        store.connectors = [c for c in store.connectors if c.connector_id != connector_id]
        if len(store.connectors) == before:
            return False

        store.last_updated_utc = _utc_now_iso()
        return await _write_store(company_id, store)
    except Exception as e:
        logger.error("delete_connector error | %s", e)
        return False
    finally:
        lock.release()


async def clear_marketplace(company_id: str) -> bool:
    lock = _get_lock(company_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return False

    try:
        result = await delete_store(_TABLE, _company_hash(company_id))
        if result:
            logger.info("🗑️ Marketplace cleared | %s", _company_hash(company_id)[:12])
        return result
    except Exception as e:
        logger.error("clear_marketplace error | %s", e)
        return False
    finally:
        lock.release()