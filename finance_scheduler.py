"""
finance_scheduler.py — Three AI Automated Sync & Cron
=====================================================
Safely pulls encrypted credentials from the marketplace store
and runs background syncs for Stripe and QuickBooks.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from marketplace import _read_store, _decrypt
from connectors import sync_stripe_metrics, sync_quickbooks_metrics, STRIPE_CONNECTOR_ID, QUICKBOOKS_CONNECTOR_ID

logger = logging.getLogger("three-ai.finance-scheduler")

async def run_company_sync(company_id: str) -> dict[str, Any]:
    """
    Looks up a company's registered connectors, decrypts their API keys securely
    in memory, and runs the data syncs.
    """
    logger.info("Starting automated sync | company_id=%s", company_id)
    
    store = await _read_store(company_id)
    if not store or not store.connectors:
        logger.warning("No connectors found | company_id=%s", company_id)
        return {"status": "no_connectors_configured"}

    results: dict[str, Any] = {}

    for config in store.connectors:
        if not config.enabled:
            continue

        # ─────────────────────────────────────────────
        # STRIPE SYNC
        # ─────────────────────────────────────────────
        if config.connector_id == STRIPE_CONNECTOR_ID:
            if not config.api_key:
                results["stripe"] = {"error": "Missing API key"}
                continue
                
            try:
                # Decrypt the secret key safely in memory
                secret_key = _decrypt(config.api_key)
                if not secret_key:
                    results["stripe"] = {"error": "Decryption failed"}
                    continue
                    
                stripe_result = await sync_stripe_metrics(
                    company_id=company_id,
                    secret_key=secret_key,
                    currency=config.default_currency or "USD"
                )
                results["stripe"] = stripe_result
                logger.info("✅ Stripe sync complete | company_id=%s", company_id)
            except Exception as e:
                logger.error("Stripe sync failed | company_id=%s | error=%s", company_id, e)
                results["stripe"] = {"error": str(e)}

        # ─────────────────────────────────────────────
        # QUICKBOOKS SYNC
        # ─────────────────────────────────────────────
        elif config.connector_id == QUICKBOOKS_CONNECTOR_ID:
            if not config.api_key:
                results["quickbooks"] = {"error": "Missing Access Token"}
                continue
                
            # For QB, we assume the realm_id is stored in the config headers dict
            realm_id = config.headers.get("realm_id")
            if not realm_id:
                results["quickbooks"] = {"error": "Missing realm_id in config headers"}
                continue
                
            try:
                # Decrypt the access token safely in memory
                access_token = _decrypt(config.api_key)
                if not access_token:
                    results["quickbooks"] = {"error": "Decryption failed"}
                    continue
                    
                qb_result = await sync_quickbooks_metrics(
                    company_id=company_id,
                    realm_id=realm_id,
                    access_token=access_token,
                    currency=config.default_currency or "USD"
                )
                results["quickbooks"] = qb_result
                logger.info("✅ QuickBooks sync complete | company_id=%s", company_id)
            except Exception as e:
                logger.error("QuickBooks sync failed | company_id=%s | error=%s", company_id, e)
                results["quickbooks"] = {"error": str(e)}

    return {
        "status": "sync_completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": results
    }