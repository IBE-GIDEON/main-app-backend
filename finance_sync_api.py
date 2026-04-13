from __future__ import annotations

import PyPDF2 # <-- Added for reading PDFs
import hashlib
from fastapi import APIRouter, HTTPException, UploadFile, File, Form # <-- Added File/Form uploads
from pydantic import BaseModel, Field

# Make sure you import your database connection so we can save the text!
# (Adjust this import if your Supabase client lives in a different file like 'memory.py')
from db import supabase 

from finance_connectors import (
    ensure_default_finance_connectors,
    sync_stripe_metrics,
    sync_quickbooks_metrics,
    sync_csv_metrics,
)

finance_sync_router = APIRouter(prefix="/finance", tags=["Finance Sync"])


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
    return {"ok": True, "connectors": result}


@finance_sync_router.post("/sync/stripe")
async def run_stripe_sync(body: StripeSyncRequest):
    try:
        result = await sync_stripe_metrics(
            body.company_id,
            secret_key=body.secret_key,
            currency=body.currency.upper(),
        )
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe sync failed: {e}")


@finance_sync_router.post("/sync/quickbooks")
async def run_quickbooks_sync(body: QuickBooksSyncRequest):
    try:
        result = await sync_quickbooks_metrics(
            body.company_id,
            realm_id=body.realm_id,
            access_token=body.access_token,
            currency=body.currency.upper(),
        )
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QuickBooks sync failed: {e}")


@finance_sync_router.post("/sync/csv")
async def run_csv_sync(body: CSVSyncRequest):
    try:
        result = await sync_csv_metrics(
            body.company_id,
            csv_text=body.csv_text,
            default_currency=body.default_currency.upper(),
            default_period=body.default_period,
        )
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV sync failed: {e}")


# 🚀 THE MISSING ENDPOINT: This catches the dropped PDF, extracts text, and saves it
@finance_sync_router.post("/upload")
async def upload_financial_context(
    company_id: str = Form(...),
    files: list[UploadFile] = File(...)
):
    try:
        combined_text = f"--- UPLOADED FINANCIAL DOCUMENTATION FOR {company_id} ---\n\n"
        
        for file in files:
            combined_text += f"Document Name: {file.filename}\n"
            
            # Extract PDF Text
            if file.filename.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file.file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        combined_text += text + "\n"
                        
            # Extract CSV Text
            elif file.filename.endswith(".csv"):
                content = await file.read()
                combined_text += content.decode("utf-8", errors="ignore") + "\n"
                
            combined_text += "\n------------------------\n"

        # Generate the company hash to match your memory_store format
        company_hash = hashlib.sha256(company_id.encode()).hexdigest()

        data = {
            "company_id_hash": company_hash,
            "data": combined_text,
            "session_id": "document_upload"
        }
        
        # Save to Supabase (Change 'memory_store' if you use a different table)
        supabase.table("memory_store").insert(data).execute()
        
        print(f"✅ Successfully extracted and saved context for {company_id}")
        return {"ok": True, "message": "Files uploaded and text extracted."}

    except Exception as e:
        print(f"❌ Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))