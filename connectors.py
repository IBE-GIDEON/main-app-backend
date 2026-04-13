"""
connectors.py — Three AI Finance Data Connectors
================================================
Direct integrations for Stripe, QuickBooks, and CSV parsing.
Normalizes external data and pushes it into the Marketplace webhook.
"""

from __future__ import annotations

import csv
import io
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from marketplace import register_connector, receive_webhook, ConnectorConfig, WebhookPayload

logger = logging.getLogger("three-ai.finance-connectors")

STRIPE_CONNECTOR_ID = "stripe_finance"
QUICKBOOKS_CONNECTOR_ID = "quickbooks_finance"
CSV_CONNECTOR_ID = "csv_finance"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _days_ago_ts(days: int) -> int:
    return int((_utc_now() - timedelta(days=days)).timestamp())


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _minor_to_major(amount: int | float | None, currency: str = "USD") -> float | None:
    if amount is None:
        return None
    # MVP: assume standard 2-decimal currencies
    return round(float(amount) / 100.0, 2)


async def ensure_default_finance_connectors(company_id: str) -> dict[str, bool]:
    """
    Creates webhook-style connectors so sync jobs can write normalized finance metrics
    into your marketplace store using receive_webhook().
    """
    results: dict[str, bool] = {}

    configs = [
        ConnectorConfig(
            connector_id=STRIPE_CONNECTOR_ID,
            kind="webhook",
            label="Stripe Finance",
            default_currency="USD",
            default_period="monthly",
        ),
        ConnectorConfig(
            connector_id=QUICKBOOKS_CONNECTOR_ID,
            kind="webhook",
            label="QuickBooks Finance",
            default_currency="USD",
            default_period="monthly",
        ),
        ConnectorConfig(
            connector_id=CSV_CONNECTOR_ID,
            kind="webhook",
            label="CSV Finance",
            default_currency="USD",
            default_period="monthly",
        ),
    ]

    for cfg in configs:
        ok = await register_connector(company_id, cfg)
        results[cfg.connector_id] = ok

    return results


async def _ingest_metrics(
    company_id: str,
    connector_id: str,
    metrics: dict[str, Any],
    *,
    currency: str = "USD",
    period: str = "monthly",
    source_label: str | None = None,
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    now_iso = _utc_now_iso()

    for metric_name, value in metrics.items():
        if value is None:
            continue

        ok = await receive_webhook(
            company_id=company_id,
            connector_id=connector_id,
            payload=WebhookPayload(
                metric_name=metric_name,
                value=value,
                value_str=str(value),
                currency=currency,
                period=period,
                recorded_utc=now_iso,
                freshness_utc=now_iso,
                source_label=source_label,
            ),
        )
        results[metric_name] = ok

    return results


# ─────────────────────────────────────────────
# STRIPE
# ─────────────────────────────────────────────

async def _stripe_get(
    client: httpx.AsyncClient,
    secret_key: str,
    path: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resp = await client.get(
        f"https://api.stripe.com{path}",
        headers={"Authorization": f"Bearer {secret_key}"},
        params=params or {},
    )
    resp.raise_for_status()
    return resp.json()


def _stripe_subscription_mrr(sub: dict[str, Any]) -> float:
    items = ((sub.get("items") or {}).get("data") or [])
    total_mrr = 0.0

    for item in items:
        price = item.get("price") or {}
        recurring = price.get("recurring") or {}
        interval = recurring.get("interval")
        interval_count = recurring.get("interval_count") or 1
        unit_amount = price.get("unit_amount")
        quantity = item.get("quantity") or 1

        amount = _minor_to_major(unit_amount) or 0.0
        amount *= quantity

        if interval == "month":
            total_mrr += amount / interval_count
        elif interval == "year":
            total_mrr += amount / 12.0 / interval_count
        elif interval == "week":
            total_mrr += amount * 4.345 / interval_count
        elif interval == "day":
            total_mrr += amount * 30.0 / interval_count

    return round(total_mrr, 2)


async def sync_stripe_metrics(
    company_id: str,
    *,
    secret_key: str,
    currency: str = "USD",
) -> dict[str, Any]:
    await ensure_default_finance_connectors(company_id)

    async with httpx.AsyncClient(timeout=30.0) as client:
        balance = await _stripe_get(client, secret_key, "/v1/balance")
        subscriptions_active = await _stripe_get(
            client,
            secret_key,
            "/v1/subscriptions",
            {"status": "active", "limit": 100},
        )
        subscriptions_all = await _stripe_get(
            client,
            secret_key,
            "/v1/subscriptions",
            {"status": "all", "limit": 100},
        )
        charges_60d = await _stripe_get(
            client,
            secret_key,
            "/v1/charges",
            {"limit": 100, "created[gte]": _days_ago_ts(60)},
        )
        payment_intents_30d = await _stripe_get(
            client,
            secret_key,
            "/v1/payment_intents",
            {"limit": 100, "created[gte]": _days_ago_ts(30)},
        )

    available_balances = balance.get("available", []) or []
    pending_balances = balance.get("pending", []) or []

    available_total = 0.0
    pending_total = 0.0

    for row in available_balances:
        if row.get("currency", "").upper() == currency.upper():
            available_total += _minor_to_major(row.get("amount"), currency) or 0.0

    for row in pending_balances:
        if row.get("currency", "").upper() == currency.upper():
            pending_total += _minor_to_major(row.get("amount"), currency) or 0.0

    cash_balance = round(available_total, 2)
    available_liquidity = round(available_total + pending_total, 2)

    active_subs = subscriptions_active.get("data", []) or []
    all_subs = subscriptions_all.get("data", []) or []
    charges = charges_60d.get("data", []) or []
    pis = payment_intents_30d.get("data", []) or []

    mrr = round(sum(_stripe_subscription_mrr(sub) for sub in active_subs), 2)
    arr = round(mrr * 12, 2)

    now = _utc_now()
    revenue_last_30d = 0.0
    revenue_prev_30d = 0.0
    customer_revenue_last_30d: dict[str, float] = defaultdict(float)

    for ch in charges:
        if not ch.get("paid"):
            continue
        created = datetime.fromtimestamp(ch["created"], tz=timezone.utc)
        amount = _minor_to_major(ch.get("amount"), currency) or 0.0
        customer_id = ch.get("customer") or "unknown"

        if created >= now - timedelta(days=30):
            revenue_last_30d += amount
            customer_revenue_last_30d[customer_id] += amount
        elif created >= now - timedelta(days=60):
            revenue_prev_30d += amount

    revenue_last_30d = round(revenue_last_30d, 2)
    revenue_prev_30d = round(revenue_prev_30d, 2)

    revenue_growth_pct = None
    if revenue_prev_30d > 0:
        revenue_growth_pct = round(((revenue_last_30d - revenue_prev_30d) / revenue_prev_30d) * 100, 2)

    failed = 0
    total_pi = 0
    for pi in pis:
        total_pi += 1
        status = (pi.get("status") or "").lower()
        if status in {"canceled", "requires_payment_method"}:
            failed += 1

    failed_payment_rate_pct = round((failed / total_pi) * 100, 2) if total_pi > 0 else None

    canceled_last_30d = 0
    for sub in all_subs:
        canceled_at = sub.get("canceled_at")
        if canceled_at:
            canceled_dt = datetime.fromtimestamp(canceled_at, tz=timezone.utc)
            if canceled_dt >= now - timedelta(days=30):
                canceled_last_30d += 1

    logo_churn_pct = None
    active_count = len(active_subs)
    denominator = active_count + canceled_last_30d
    if denominator > 0:
        logo_churn_pct = round((canceled_last_30d / denominator) * 100, 2)

    customer_concentration_pct = None
    top_customer_share_pct = None
    if revenue_last_30d > 0 and customer_revenue_last_30d:
        sorted_vals = sorted(customer_revenue_last_30d.values(), reverse=True)
        top_3 = sum(sorted_vals[:3])
        top_1 = sorted_vals[0]
        customer_concentration_pct = round((top_3 / revenue_last_30d) * 100, 2)
        top_customer_share_pct = round((top_1 / revenue_last_30d) * 100, 2)

    metrics = {
        "cash_balance": cash_balance,
        "available_liquidity": available_liquidity,
        "mrr": mrr,
        "arr": arr,
        "revenue_last_30d": revenue_last_30d,
        "revenue_prev_30d": revenue_prev_30d,
        "revenue_growth_pct": revenue_growth_pct,
        "failed_payment_rate_pct": failed_payment_rate_pct,
        "logo_churn_pct": logo_churn_pct,
        "customer_concentration_pct": customer_concentration_pct,
        "top_customer_share_pct": top_customer_share_pct,
    }

    ingest = await _ingest_metrics(
        company_id,
        STRIPE_CONNECTOR_ID,
        metrics,
        currency=currency,
        period="monthly",
        source_label="Stripe Finance",
    )

    return {
        "connector": "stripe",
        "metrics": metrics,
        "ingested": ingest,
    }


# ─────────────────────────────────────────────
# QUICKBOOKS
# ─────────────────────────────────────────────

def _qb_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/text",
    }


def _qb_find_first_value(rows: list[dict[str, Any]], names: list[str]) -> float | None:
    """
    Recursively scan QBO report rows for summary rows with matching labels.
    """
    target_names = {n.lower() for n in names}

    def walk(items: list[dict[str, Any]]) -> float | None:
        for item in items:
            header = item.get("Header", {}) or {}
            coldata = header.get("ColData", []) or []
            row_name = ""
            if coldata:
                row_name = (coldata[0].get("value") or "").strip().lower()

            summary = item.get("Summary", {}) or {}
            summary_cols = summary.get("ColData", []) or []

            if row_name in target_names and len(summary_cols) > 1:
                return _safe_float(summary_cols[-1].get("value"))

            subrows = ((item.get("Rows") or {}).get("Row")) or []
            found = walk(subrows)
            if found is not None:
                return found
        return None

    return walk(rows)


async def _qb_query(
    client: httpx.AsyncClient,
    realm_id: str,
    access_token: str,
    query: str,
) -> dict[str, Any]:
    resp = await client.post(
        f"https://quickbooks.api.intuit.com/v3/company/{realm_id}/query",
        headers=_qb_headers(access_token),
        params={"minorversion": "75"},
        content=query,
    )
    resp.raise_for_status()
    return resp.json()


async def _qb_report(
    client: httpx.AsyncClient,
    realm_id: str,
    access_token: str,
    report_name: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resp = await client.get(
        f"https://quickbooks.api.intuit.com/v3/company/{realm_id}/reports/{report_name}",
        headers=_qb_headers(access_token),
        params={"minorversion": "75", **(params or {})},
    )
    resp.raise_for_status()
    return resp.json()


async def sync_quickbooks_metrics(
    company_id: str,
    *,
    realm_id: str,
    access_token: str,
    currency: str = "USD",
) -> dict[str, Any]:
    await ensure_default_finance_connectors(company_id)

    today = _utc_now().date()
    start_30 = today - timedelta(days=29)
    start_prev_30 = today - timedelta(days=59)
    end_prev_30 = today - timedelta(days=30)

    async with httpx.AsyncClient(timeout=30.0) as client:
        balance_sheet = await _qb_report(client, realm_id, access_token, "BalanceSheet")
        pnl_30 = await _qb_report(
            client,
            realm_id,
            access_token,
            "ProfitAndLoss",
            {"start_date": str(start_30), "end_date": str(today)},
        )
        pnl_prev_30 = await _qb_report(
            client,
            realm_id,
            access_token,
            "ProfitAndLoss",
            {"start_date": str(start_prev_30), "end_date": str(end_prev_30)},
        )
        invoice_query = await _qb_query(
            client,
            realm_id,
            access_token,
            "select Id, Balance, DueDate, TotalAmt from Invoice where Balance > '0' maxresults 1000",
        )
        bill_query = await _qb_query(
            client,
            realm_id,
            access_token,
            "select Id, Balance, DueDate, TotalAmt from Bill where Balance > '0' maxresults 1000",
        )

    bs_rows = ((balance_sheet.get("Rows") or {}).get("Row")) or []
    pnl_rows = ((pnl_30.get("Rows") or {}).get("Row")) or []
    pnl_prev_rows = ((pnl_prev_30.get("Rows") or {}).get("Row")) or []

    cash_balance = _qb_find_first_value(bs_rows, ["cash and cash equivalents", "cash", "bank accounts"])
    ar_total = _qb_find_first_value(bs_rows, ["accounts receivable", "total accounts receivable"])
    ap_total = _qb_find_first_value(bs_rows, ["accounts payable", "total accounts payable"])
    current_assets = _qb_find_first_value(bs_rows, ["total current assets"])
    current_liabilities = _qb_find_first_value(bs_rows, ["total current liabilities"])

    income_30 = _qb_find_first_value(pnl_rows, ["total income", "income", "total revenue"])
    cogs_30 = _qb_find_first_value(pnl_rows, ["total cost of goods sold", "cost of goods sold"])
    net_income_30 = _qb_find_first_value(pnl_rows, ["net income"])
    gross_profit_30 = _qb_find_first_value(pnl_rows, ["gross profit"])

    income_prev = _qb_find_first_value(pnl_prev_rows, ["total income", "income", "total revenue"])

    revenue_last_30d = round(income_30 or 0.0, 2)
    revenue_prev_30d = round(income_prev or 0.0, 2)

    revenue_growth_pct = None
    if revenue_prev_30d > 0:
        revenue_growth_pct = round(((revenue_last_30d - revenue_prev_30d) / revenue_prev_30d) * 100, 2)

    gross_margin_pct = None
    if revenue_last_30d > 0:
        if gross_profit_30 is not None:
            gross_margin_pct = round((gross_profit_30 / revenue_last_30d) * 100, 2)
        elif cogs_30 is not None:
            gross_margin_pct = round(((revenue_last_30d - cogs_30) / revenue_last_30d) * 100, 2)

    net_margin_pct = None
    if revenue_last_30d > 0 and net_income_30 is not None:
        net_margin_pct = round((net_income_30 / revenue_last_30d) * 100, 2)

    invoices = (((invoice_query.get("QueryResponse") or {}).get("Invoice")) or [])
    bills = (((bill_query.get("QueryResponse") or {}).get("Bill")) or [])

    ar_overdue_30_plus = 0.0
    for inv in invoices:
        due = inv.get("DueDate")
        bal = _safe_float(inv.get("Balance")) or 0.0
        if due:
            try:
                due_dt = datetime.fromisoformat(due).date()
                if due_dt <= today - timedelta(days=30):
                    ar_overdue_30_plus += bal
            except Exception:
                pass

    ap_due_30d = 0.0
    for bill in bills:
        due = bill.get("DueDate")
        bal = _safe_float(bill.get("Balance")) or 0.0
        if due:
            try:
                due_dt = datetime.fromisoformat(due).date()
                if due_dt <= today + timedelta(days=30):
                    ap_due_30d += bal
            except Exception:
                pass

    current_ratio = None
    if current_assets is not None and current_liabilities not in (None, 0):
        current_ratio = round(current_assets / current_liabilities, 2)

    metrics = {
        "cash_balance": round(cash_balance or 0.0, 2) if cash_balance is not None else None,
        "ar_total": round(ar_total or 0.0, 2) if ar_total is not None else None,
        "ar_overdue_30_plus": round(ar_overdue_30_plus, 2),
        "ap_total": round(ap_total or 0.0, 2) if ap_total is not None else None,
        "ap_due_30d": round(ap_due_30d, 2),
        "revenue_last_30d": revenue_last_30d,
        "revenue_prev_30d": revenue_prev_30d,
        "revenue_growth_pct": revenue_growth_pct,
        "gross_margin_pct": gross_margin_pct,
        "net_margin_pct": net_margin_pct,
        "current_ratio": current_ratio,
    }

    ingest = await _ingest_metrics(
        company_id,
        QUICKBOOKS_CONNECTOR_ID,
        metrics,
        currency=currency,
        period="monthly",
        source_label="QuickBooks Finance",
    )

    return {
        "connector": "quickbooks",
        "metrics": metrics,
        "ingested": ingest,
    }


# ─────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────

async def sync_csv_metrics(
    company_id: str,
    *,
    csv_text: str,
    default_currency: str = "USD",
    default_period: str = "monthly",
) -> dict[str, Any]:
    """
    Expected CSV columns:
    metric_name,value,currency,period,recorded_utc
    """
    await ensure_default_finance_connectors(company_id)

    reader = csv.DictReader(io.StringIO(csv_text))
    ingested: dict[str, bool] = {}
    parsed_metrics: dict[str, Any] = {}

    for row in reader:
        metric_name = (row.get("metric_name") or "").strip()
        value = row.get("value")
        currency = (row.get("currency") or default_currency).strip()
        period = (row.get("period") or default_period).strip()
        recorded_utc = (row.get("recorded_utc") or _utc_now_iso()).strip()

        num = _safe_float(value)
        final_value: Any = num if num is not None else value

        ok = await receive_webhook(
            company_id=company_id,
            connector_id=CSV_CONNECTOR_ID,
            payload=WebhookPayload(
                metric_name=metric_name,
                value=final_value,
                value_str=str(final_value),
                currency=currency,
                period=period,
                recorded_utc=recorded_utc,
                freshness_utc=_utc_now_iso(),
                source_label="CSV Finance",
            ),
        )

        ingested[metric_name] = ok
        parsed_metrics[metric_name] = final_value

    return {
        "connector": "csv",
        "metrics": parsed_metrics,
        "ingested": ingested,
    }