from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

import httpx
from pydantic import BaseModel, Field

from output import UIDecisionPayload

logger = logging.getLogger("three-ai.delivery")

_RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
_SMTP_HOST = os.getenv("SMTP_HOST", "")
_SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
_SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
_SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
_SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() != "false"
_MAIL_FROM = (
    os.getenv("THINK_AI_VERDICT_EMAIL_FROM")
    or os.getenv("SMTP_FROM")
    or os.getenv("RESEND_FROM")
    or ""
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _looks_like_email(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value.strip()))


def _render_subject(query: str, subject: str | None = None) -> str:
    if subject and subject.strip():
        return subject.strip()
    return f"Three AI verdict: {query[:72]}".strip()


def _render_plaintext(query: str, decision: UIDecisionPayload) -> str:
    verdict = decision.verdict_card
    next_step = decision.next_step
    snapshot = decision.finance_snapshot
    immediate_action = next_step.immediate_action if next_step else "Review verdict in Three AI"
    go_conditions = [f"- {item}" for item in verdict.go_conditions] or ["- None"]
    stop_conditions = [f"- {item}" for item in verdict.stop_conditions] or ["- None"]
    review_triggers = [f"- {item}" for item in verdict.review_triggers] or ["- None"]

    lines = [
        f"Query: {query}",
        "",
        f"Verdict: {verdict.headline}",
        verdict.rationale,
        "",
        "Go conditions:",
        *go_conditions,
        "",
        "Stop conditions:",
        *stop_conditions,
        "",
        "Review triggers:",
        *review_triggers,
        "",
        f"Immediate action: {immediate_action}",
    ]

    if snapshot:
        lines.extend(
            [
                "",
                "Finance snapshot:",
                f"- Cash balance: {snapshot.cash_balance if snapshot.cash_balance is not None else 'n/a'}",
                f"- Runway months: {snapshot.runway_months if snapshot.runway_months is not None else 'n/a'}",
                f"- Revenue growth: {snapshot.revenue_growth_pct if snapshot.revenue_growth_pct is not None else 'n/a'}",
                f"- Gross margin: {snapshot.gross_margin_pct if snapshot.gross_margin_pct is not None else 'n/a'}",
            ]
        )

    return "\n".join(lines)


def _render_html(query: str, decision: UIDecisionPayload) -> str:
    verdict = decision.verdict_card
    next_step = decision.next_step
    snapshot = decision.finance_snapshot

    def render_list(items: list[str]) -> str:
        if not items:
            return "<li>None</li>"
        return "".join(f"<li>{html.escape(item)}</li>" for item in items)

    metrics_html = ""
    if snapshot:
        metrics_html = f"""
        <h3>Finance snapshot</h3>
        <table cellpadding="8" cellspacing="0" border="1" style="border-collapse: collapse; width: 100%;">
          <tr><th align="left">Metric</th><th align="left">Value</th></tr>
          <tr><td>Cash balance</td><td>{snapshot.cash_balance if snapshot.cash_balance is not None else 'n/a'}</td></tr>
          <tr><td>Runway months</td><td>{snapshot.runway_months if snapshot.runway_months is not None else 'n/a'}</td></tr>
          <tr><td>Revenue growth</td><td>{snapshot.revenue_growth_pct if snapshot.revenue_growth_pct is not None else 'n/a'}</td></tr>
          <tr><td>Gross margin</td><td>{snapshot.gross_margin_pct if snapshot.gross_margin_pct is not None else 'n/a'}</td></tr>
        </table>
        """

    immediate_action = html.escape(next_step.immediate_action if next_step else "Review verdict in Three AI")

    return f"""
    <html>
      <body style="font-family: Arial, Helvetica, sans-serif; color: #111827; line-height: 1.5;">
        <h2>Three AI finance verdict</h2>
        <p><strong>Query:</strong> {html.escape(query)}</p>
        <h3>{html.escape(verdict.headline)}</h3>
        <p>{html.escape(verdict.rationale)}</p>
        <h3>Go conditions</h3>
        <ul>{render_list(verdict.go_conditions)}</ul>
        <h3>Stop conditions</h3>
        <ul>{render_list(verdict.stop_conditions)}</ul>
        <h3>Review triggers</h3>
        <ul>{render_list(verdict.review_triggers)}</ul>
        <p><strong>Immediate action:</strong> {immediate_action}</p>
        {metrics_html}
      </body>
    </html>
    """.strip()


def _send_smtp_email(to: str, subject: str, plain: str, html: str) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = _MAIL_FROM
    message["To"] = to
    message.set_content(plain)
    message.add_alternative(html, subtype="html")

    with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=20) as server:
        if _SMTP_USE_TLS:
            server.starttls()
        if _SMTP_USERNAME:
            server.login(_SMTP_USERNAME, _SMTP_PASSWORD)
        server.send_message(message)


async def _send_resend_email(to: str, subject: str, plain: str, html: str) -> None:
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {_RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": _MAIL_FROM,
                "to": [to],
                "subject": subject,
                "text": plain,
                "html": html,
            },
        )
        response.raise_for_status()


class VerdictEmailRequest(BaseModel):
    company_id: str = Field(min_length=1, max_length=128)
    to: str = Field(min_length=5, max_length=320)
    query: str = Field(min_length=3, max_length=2000)
    subject: str | None = Field(default=None, max_length=160)
    decision: UIDecisionPayload


class DeliveryResult(BaseModel):
    ok: bool
    provider: str
    to: str
    subject: str
    sent_utc: str


def is_email_delivery_available() -> bool:
    return bool(_MAIL_FROM and (_RESEND_API_KEY or _SMTP_HOST))


async def send_verdict_email(request: VerdictEmailRequest) -> DeliveryResult:
    to = request.to.strip()
    if not _looks_like_email(to):
        raise ValueError("A valid destination email is required.")
    if not is_email_delivery_available():
        raise RuntimeError("Email delivery is not configured on the server.")

    subject = _render_subject(request.query, request.subject)
    plain = _render_plaintext(request.query, request.decision)
    html = _render_html(request.query, request.decision)

    if _RESEND_API_KEY:
        await _send_resend_email(to, subject, plain, html)
        provider = "resend"
    else:
        await asyncio.to_thread(_send_smtp_email, to, subject, plain, html)
        provider = "smtp"

    result = DeliveryResult(
        ok=True,
        provider=provider,
        to=to,
        subject=subject,
        sent_utc=_utc_now_iso(),
    )
    logger.info("Verdict email sent | company=%s | to=%s | provider=%s", request.company_id, to, provider)
    return result
