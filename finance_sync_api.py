from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from connectors import (
    ensure_default_finance_connectors,
    sync_csv_metrics,
    sync_quickbooks_metrics,
    sync_stripe_metrics,
)
from delivery import VerdictEmailRequest, is_email_delivery_available, send_verdict_email
from documents import (
    build_document_bundle_response,
    build_document_detail_response,
    build_document_upload_response,
    delete_uploaded_document,
    get_uploaded_document_asset,
    get_uploaded_document_detail,
    get_uploaded_documents,
    store_uploaded_documents,
)
from insights_runtime import enroll_company_for_background_insights

logger = logging.getLogger("three-ai.finance-api")

finance_sync_router = APIRouter(prefix="/finance", tags=["Finance"])


class ConnectorSetupRequest(BaseModel):
    company_id: str = Field(min_length=1, max_length=128)


class StripeSyncRequest(BaseModel):
    company_id: str = Field(min_length=1, max_length=128)
    secret_key: str = Field(min_length=10)
    currency: str = Field(default="USD", min_length=3, max_length=3)


class QuickBooksSyncRequest(BaseModel):
    company_id: str = Field(min_length=1, max_length=128)
    realm_id: str = Field(min_length=1)
    access_token: str = Field(min_length=20)
    currency: str = Field(default="USD", min_length=3, max_length=3)


class CSVSyncRequest(BaseModel):
    company_id: str = Field(min_length=1, max_length=128)
    csv_text: str = Field(min_length=5)
    default_currency: str = Field(default="USD", min_length=3, max_length=3)
    default_period: str = Field(default="monthly", min_length=3, max_length=20)


@finance_sync_router.post("/connectors/setup")
async def setup_finance_connectors(body: ConnectorSetupRequest):
    result = await ensure_default_finance_connectors(body.company_id)
    await enroll_company_for_background_insights(
        body.company_id,
        reason="connector-setup",
    )
    return {"ok": True, "connectors": result}


@finance_sync_router.post("/sync/stripe")
async def run_stripe_sync(body: StripeSyncRequest):
    try:
        result = await sync_stripe_metrics(
            body.company_id,
            secret_key=body.secret_key,
            currency=body.currency.upper(),
        )
        await enroll_company_for_background_insights(
            body.company_id,
            reason="stripe-sync",
        )
        return {"ok": True, "result": result}
    except Exception as exc:
        logger.error("Stripe sync failed | company=%s | %s", body.company_id, exc)
        raise HTTPException(status_code=500, detail=f"Stripe sync failed: {exc}") from exc


@finance_sync_router.post("/sync/quickbooks")
async def run_quickbooks_sync(body: QuickBooksSyncRequest):
    try:
        result = await sync_quickbooks_metrics(
            body.company_id,
            realm_id=body.realm_id,
            access_token=body.access_token,
            currency=body.currency.upper(),
        )
        await enroll_company_for_background_insights(
            body.company_id,
            reason="quickbooks-sync",
        )
        return {"ok": True, "result": result}
    except Exception as exc:
        logger.error("QuickBooks sync failed | company=%s | %s", body.company_id, exc)
        raise HTTPException(status_code=500, detail=f"QuickBooks sync failed: {exc}") from exc


@finance_sync_router.post("/sync/csv")
async def run_csv_sync(body: CSVSyncRequest):
    try:
        result = await sync_csv_metrics(
            body.company_id,
            csv_text=body.csv_text,
            default_currency=body.default_currency.upper(),
            default_period=body.default_period,
        )
        await enroll_company_for_background_insights(
            body.company_id,
            reason="csv-sync",
        )
        return {"ok": True, "result": result}
    except Exception as exc:
        logger.error("CSV sync failed | company=%s | %s", body.company_id, exc)
        raise HTTPException(status_code=500, detail=f"CSV sync failed: {exc}") from exc


@finance_sync_router.post("/upload")
async def upload_financial_context(
    company_id: str = Form(...),
    files: list[UploadFile] = File(...),
):
    try:
        bundle = await store_uploaded_documents(company_id, files)
        await enroll_company_for_background_insights(
            company_id,
            reason="document-upload",
        )
        return build_document_upload_response(bundle)
    except Exception as exc:
        logger.error("Financial document upload failed | company=%s | %s", company_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@finance_sync_router.get("/uploads/{company_id}")
async def list_uploaded_financial_context(company_id: str):
    try:
        bundle = get_uploaded_documents(company_id)
        return build_document_bundle_response(bundle)
    except Exception as exc:
        logger.error("Financial document list failed | company=%s | %s", company_id, exc)
        raise HTTPException(status_code=500, detail="Failed to load manual data deck.") from exc


@finance_sync_router.get("/uploads/{company_id}/{document_id}")
async def get_uploaded_financial_document(company_id: str, document_id: str):
    try:
        detail = get_uploaded_document_detail(company_id, document_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        return build_document_detail_response(detail)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Financial document detail failed | company=%s | document_id=%s | %s",
            company_id,
            document_id,
            exc,
        )
        raise HTTPException(status_code=500, detail="Failed to load document detail.") from exc


@finance_sync_router.get("/uploads/{company_id}/{document_id}/download")
async def download_uploaded_financial_document(company_id: str, document_id: str):
    try:
        asset = get_uploaded_document_asset(company_id, document_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="Document not found.")

        file_path, meta = asset
        return FileResponse(
            path=file_path,
            media_type=meta.content_type or "application/octet-stream",
            filename=meta.filename,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Financial document download failed | company=%s | document_id=%s | %s",
            company_id,
            document_id,
            exc,
        )
        raise HTTPException(status_code=500, detail="Failed to download document.") from exc


@finance_sync_router.delete("/uploads/{company_id}/{document_id}")
async def delete_uploaded_financial_document(company_id: str, document_id: str):
    try:
        bundle = await delete_uploaded_document(company_id, document_id)
        await enroll_company_for_background_insights(
            company_id,
            reason="document-delete",
        )
        return {
            **build_document_bundle_response(bundle),
            "message": "Manual data document deleted.",
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc
    except Exception as exc:
        logger.error(
            "Financial document delete failed | company=%s | document_id=%s | %s",
            company_id,
            document_id,
            exc,
        )
        raise HTTPException(status_code=500, detail="Failed to delete document.") from exc


@finance_sync_router.get("/delivery/status")
async def get_delivery_status():
    return {
        "ok": True,
        "email_delivery_available": is_email_delivery_available(),
    }


@finance_sync_router.post("/deliver/email")
async def deliver_verdict_email(body: VerdictEmailRequest):
    try:
        result = await send_verdict_email(body)
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Verdict email delivery failed | company=%s | %s", body.company_id, exc)
        raise HTTPException(status_code=500, detail="Failed to send verdict email.") from exc
