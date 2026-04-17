"""
billing.py - Three AI subscription billing and Paystack verification.

This module keeps billing state keyed to company_id so the frontend can read a
single subscription truth across checkout, account, and upgrade screens.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from db import read_store, write_store

load_dotenv()

logger = logging.getLogger("three-ai.billing")

_TABLE = "billing_store"
_BILLING_VERSION = "1.0.0"
_PAYSTACK_BASE_URL = os.getenv("THINK_AI_PAYSTACK_BASE_URL", "https://api.paystack.co")
_PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY", "")
_PAYSTACK_TIMEOUT_SEC = float(os.getenv("THINK_AI_PAYSTACK_TIMEOUT_SEC", "20"))
_STORE_DIR = Path(os.getenv("THINK_AI_BILLING_STORE_DIR", Path(__file__).with_name("billing_store")))
_DEFAULT_PERIOD_DAYS = int(os.getenv("THINK_AI_BILLING_PERIOD_DAYS", "30"))
_DEFAULT_CURRENCY = os.getenv("THINK_AI_BILLING_CURRENCY", "USD").upper()
_DEFAULT_PLAN_ID = os.getenv("THINK_AI_BILLING_PLAN_ID", "mvp_monthly")
_DEFAULT_PLAN_NAME = os.getenv("THINK_AI_BILLING_PLAN_NAME", "MVP Plan")
_DEFAULT_PLAN_AMOUNT_MINOR = int(os.getenv("THINK_AI_BILLING_PLAN_AMOUNT_MINOR", "56000"))
_DEFAULT_DECISION_LIMIT = int(os.getenv("THINK_AI_BILLING_DECISION_LIMIT", "50"))
_DEFAULT_MANUAL_UPLOAD_LIMIT = int(os.getenv("THINK_AI_BILLING_MANUAL_UPLOAD_LIMIT", "250"))
_MAX_PAYMENT_HISTORY = int(os.getenv("THINK_AI_BILLING_MAX_PAYMENT_HISTORY", "25"))

SubscriptionStatus = Literal["inactive", "active", "expired", "canceled"]


class PlanDefinition(BaseModel):
    plan_id: str
    name: str
    amount_minor: int
    currency: str
    interval_days: int
    max_decisions_per_month: int | None = None
    max_manual_uploads: int | None = None
    feature_flags: list[str] = Field(default_factory=list)


_PLAN_CATALOG: dict[str, PlanDefinition] = {
    _DEFAULT_PLAN_ID: PlanDefinition(
        plan_id=_DEFAULT_PLAN_ID,
        name=_DEFAULT_PLAN_NAME,
        amount_minor=_DEFAULT_PLAN_AMOUNT_MINOR,
        currency=_DEFAULT_CURRENCY,
        interval_days=_DEFAULT_PERIOD_DAYS,
        max_decisions_per_month=_DEFAULT_DECISION_LIMIT,
        max_manual_uploads=_DEFAULT_MANUAL_UPLOAD_LIMIT,
        feature_flags=[
            "recursive_decisions",
            "audit_history",
            "connector_context",
            "manual_data_uploads",
            "insights_monitoring",
            "verdict_email",
        ],
    ),
}


class BillingEntitlements(BaseModel):
    has_paid_access: bool = False
    max_decisions_per_month: int | None = 5
    max_manual_uploads: int | None = 5
    feature_flags: list[str] = Field(default_factory=list)


class BillingPaymentRecord(BaseModel):
    provider: Literal["paystack"] = "paystack"
    reference: str
    amount_minor: int
    currency: str
    paid_at_utc: str
    status: Literal["success"] = "success"
    email: str | None = None
    provider_transaction_id: str | None = None
    provider_channel: str | None = None


class CompanyBillingStore(BaseModel):
    company_id_hash: str
    version: str = _BILLING_VERSION
    status: SubscriptionStatus = "inactive"
    plan_id: str | None = None
    plan_name: str | None = None
    billing_provider: str | None = None
    customer_email: str | None = None
    customer_name: str | None = None
    activated_at_utc: str | None = None
    current_period_start_utc: str | None = None
    current_period_end_utc: str | None = None
    cancel_at_period_end: bool = False
    last_payment_reference: str | None = None
    last_payment_amount_minor: int | None = None
    last_payment_currency: str | None = None
    last_payment_at_utc: str | None = None
    entitlements: BillingEntitlements = Field(default_factory=BillingEntitlements)
    payments: list[BillingPaymentRecord] = Field(default_factory=list)
    updated_utc: str


class BillingStatusResponse(BaseModel):
    company_id: str
    status: SubscriptionStatus
    is_active: bool
    has_paid_access: bool
    plan_id: str | None = None
    plan_name: str | None = None
    billing_provider: str | None = None
    customer_email: str | None = None
    customer_name: str | None = None
    activated_at_utc: str | None = None
    current_period_start_utc: str | None = None
    current_period_end_utc: str | None = None
    last_payment_reference: str | None = None
    last_payment_amount_minor: int | None = None
    last_payment_currency: str | None = None
    last_payment_at_utc: str | None = None
    cancel_at_period_end: bool = False
    entitlements: BillingEntitlements = Field(default_factory=BillingEntitlements)
    recent_payments: list[BillingPaymentRecord] = Field(default_factory=list)
    updated_utc: str


class PaystackVerifyRequest(BaseModel):
    company_id: str = Field(min_length=1, max_length=128)
    reference: str = Field(min_length=6, max_length=128)
    email: str | None = Field(default=None, max_length=320)
    name: str | None = Field(default=None, max_length=160)
    plan_id: str | None = Field(default=None, max_length=64)


_write_locks: dict[str, asyncio.Lock] = {}


def _get_lock(company_id: str) -> asyncio.Lock:
    if company_id not in _write_locks:
        _write_locks[company_id] = asyncio.Lock()
    return _write_locks[company_id]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _company_hash(company_id: str) -> str:
    return hashlib.sha256(company_id.encode("utf-8")).hexdigest()


def _coerce_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _serialize_store(store: CompanyBillingStore) -> dict:
    return store.model_dump(mode="json")


def _storage_path(company_id_hash: str) -> Path:
    return _STORE_DIR / f"{company_id_hash}.json"


async def _read_fallback_store(company_id_hash: str) -> CompanyBillingStore | None:
    path = _storage_path(company_id_hash)
    if not path.exists():
        return None

    try:
        raw_text = await asyncio.to_thread(path.read_text, "utf-8")
        return CompanyBillingStore.model_validate(json.loads(raw_text))
    except Exception as exc:
        logger.warning("Fallback billing read failed | company_hash=%s | %s", company_id_hash[:12], exc)
        return None


async def _write_fallback_store(store: CompanyBillingStore) -> None:
    await asyncio.to_thread(_STORE_DIR.mkdir, parents=True, exist_ok=True)
    payload = json.dumps(_serialize_store(store), ensure_ascii=True, indent=2)
    await asyncio.to_thread(_storage_path(store.company_id_hash).write_text, payload, "utf-8")


async def _read_company_store(company_id: str) -> CompanyBillingStore | None:
    company_id_hash = _company_hash(company_id)
    raw = await read_store(_TABLE, company_id_hash)
    if raw is not None:
        try:
            return CompanyBillingStore.model_validate(raw)
        except Exception as exc:
            logger.warning("Billing store parse failed | company_hash=%s | %s", company_id_hash[:12], exc)

    return await _read_fallback_store(company_id_hash)


async def _persist_company_store(store: CompanyBillingStore) -> None:
    payload = _serialize_store(store)
    remote_ok = await write_store(_TABLE, store.company_id_hash, payload)
    try:
        await _write_fallback_store(store)
    except Exception as exc:
        logger.warning("Fallback billing write failed | company_hash=%s | %s", store.company_id_hash[:12], exc)

    if not remote_ok:
        logger.info("Billing persisted with local fallback only | company_hash=%s", store.company_id_hash[:12])


def _resolve_plan(plan_id: str | None) -> PlanDefinition:
    if not plan_id:
        return _PLAN_CATALOG[_DEFAULT_PLAN_ID]
    return _PLAN_CATALOG.get(plan_id, _PLAN_CATALOG[_DEFAULT_PLAN_ID])


def _free_entitlements() -> BillingEntitlements:
    return BillingEntitlements(
        has_paid_access=False,
        max_decisions_per_month=5,
        max_manual_uploads=5,
        feature_flags=[],
    )


def _plan_entitlements(plan: PlanDefinition) -> BillingEntitlements:
    return BillingEntitlements(
        has_paid_access=True,
        max_decisions_per_month=plan.max_decisions_per_month,
        max_manual_uploads=plan.max_manual_uploads,
        feature_flags=plan.feature_flags[:],
    )


def _normalize_store_status(store: CompanyBillingStore) -> CompanyBillingStore:
    period_end = _coerce_datetime(store.current_period_end_utc)
    if store.status == "active" and period_end and period_end < _utc_now():
        store.status = "expired"
        store.entitlements = _free_entitlements()
        store.updated_utc = _utc_now_iso()
    return store


def _to_status_response(company_id: str, store: CompanyBillingStore | None) -> BillingStatusResponse:
    now_iso = _utc_now_iso()
    if store is None:
        return BillingStatusResponse(
            company_id=company_id,
            status="inactive",
            is_active=False,
            has_paid_access=False,
            entitlements=_free_entitlements(),
            updated_utc=now_iso,
        )

    normalized = _normalize_store_status(store)
    period_end = _coerce_datetime(normalized.current_period_end_utc)
    has_access = normalized.status == "active" and bool(period_end and period_end >= _utc_now())

    if normalized.status == "canceled" and period_end and period_end >= _utc_now():
        has_access = True

    return BillingStatusResponse(
        company_id=company_id,
        status=normalized.status,
        is_active=has_access,
        has_paid_access=has_access and normalized.entitlements.has_paid_access,
        plan_id=normalized.plan_id,
        plan_name=normalized.plan_name,
        billing_provider=normalized.billing_provider,
        customer_email=normalized.customer_email,
        customer_name=normalized.customer_name,
        activated_at_utc=normalized.activated_at_utc,
        current_period_start_utc=normalized.current_period_start_utc,
        current_period_end_utc=normalized.current_period_end_utc,
        last_payment_reference=normalized.last_payment_reference,
        last_payment_amount_minor=normalized.last_payment_amount_minor,
        last_payment_currency=normalized.last_payment_currency,
        last_payment_at_utc=normalized.last_payment_at_utc,
        cancel_at_period_end=normalized.cancel_at_period_end,
        entitlements=normalized.entitlements,
        recent_payments=normalized.payments[:10],
        updated_utc=normalized.updated_utc,
    )


async def get_billing_status(company_id: str) -> BillingStatusResponse:
    company_id = company_id.strip()
    store = await _read_company_store(company_id)
    if store is None:
        return _to_status_response(company_id, None)

    status_before = store.status
    normalized = _normalize_store_status(store)
    if normalized.status != status_before:
        await _persist_company_store(normalized)
    return _to_status_response(company_id, normalized)


async def _verify_paystack_reference(reference: str) -> dict:
    if not _PAYSTACK_SECRET_KEY:
        raise RuntimeError("PAYSTACK_SECRET_KEY is not configured on the backend.")

    headers = {"Authorization": f"Bearer {_PAYSTACK_SECRET_KEY}"}
    url = f"{_PAYSTACK_BASE_URL.rstrip('/')}/transaction/verify/{quote(reference)}"

    async with httpx.AsyncClient(timeout=_PAYSTACK_TIMEOUT_SEC) as client:
        response = await client.get(url, headers=headers)

    try:
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError("Paystack verification request failed.") from exc

    data = response.json()
    if not data.get("status"):
        raise ValueError("Paystack could not verify this payment reference.")
    return data.get("data") or {}


async def verify_paystack_payment(payload: PaystackVerifyRequest) -> BillingStatusResponse:
    company_id = payload.company_id.strip()
    reference = payload.reference.strip()
    plan = _resolve_plan(payload.plan_id)

    lock = _get_lock(company_id)
    async with lock:
        existing_store = await _read_company_store(company_id)
        if existing_store:
            previous_status = existing_store.status
            existing_store = _normalize_store_status(existing_store)
            if existing_store.status != previous_status:
                await _persist_company_store(existing_store)
            if any(payment.reference == reference for payment in existing_store.payments):
                return _to_status_response(company_id, existing_store)

        verification = await _verify_paystack_reference(reference)
        provider_reference = str(verification.get("reference") or "").strip()
        if provider_reference and provider_reference != reference:
            raise ValueError("Verified payment reference does not match the submitted reference.")

        provider_status = (verification.get("status") or "").strip().lower()
        if provider_status != "success":
            raise ValueError("Paystack has not marked this transaction as successful.")

        amount_minor = int(verification.get("amount") or 0)
        currency = str(verification.get("currency") or plan.currency).upper()

        if amount_minor != plan.amount_minor:
            raise ValueError("Verified payment amount does not match the selected plan.")
        if currency != plan.currency:
            raise ValueError("Verified payment currency does not match the selected plan.")

        paid_at = (
            _coerce_datetime(verification.get("paid_at"))
            or _coerce_datetime(verification.get("transaction_date"))
            or _utc_now()
        )

        current_period_start = paid_at
        if existing_store and existing_store.status == "active":
            existing_period_end = _coerce_datetime(existing_store.current_period_end_utc)
            if existing_period_end and existing_period_end > current_period_start:
                current_period_start = existing_period_end

        current_period_end = current_period_start + timedelta(days=plan.interval_days)
        customer = verification.get("customer") or {}
        customer_email = customer.get("email") or payload.email or (existing_store.customer_email if existing_store else None)
        customer_name = payload.name or (existing_store.customer_name if existing_store else None)

        payment_record = BillingPaymentRecord(
            reference=reference,
            amount_minor=amount_minor,
            currency=currency,
            paid_at_utc=paid_at.isoformat(),
            email=customer_email,
            provider_transaction_id=str(verification.get("id")) if verification.get("id") is not None else None,
            provider_channel=verification.get("channel"),
        )

        payments = [payment_record]
        if existing_store:
            payments.extend(existing_store.payments)
        payments = payments[:_MAX_PAYMENT_HISTORY]

        company_id_hash = _company_hash(company_id)
        store = CompanyBillingStore(
            company_id_hash=company_id_hash,
            version=_BILLING_VERSION,
            status="active",
            plan_id=plan.plan_id,
            plan_name=plan.name,
            billing_provider="paystack",
            customer_email=customer_email,
            customer_name=customer_name,
            activated_at_utc=(existing_store.activated_at_utc if existing_store and existing_store.activated_at_utc else paid_at.isoformat()),
            current_period_start_utc=current_period_start.isoformat(),
            current_period_end_utc=current_period_end.isoformat(),
            cancel_at_period_end=False,
            last_payment_reference=reference,
            last_payment_amount_minor=amount_minor,
            last_payment_currency=currency,
            last_payment_at_utc=paid_at.isoformat(),
            entitlements=_plan_entitlements(plan),
            payments=payments,
            updated_utc=_utc_now_iso(),
        )

        await _persist_company_store(store)
        return _to_status_response(company_id, store)
