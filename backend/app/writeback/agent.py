"""
Write-Back Agent (spec Section 3 & 6, step 6).

Closes the loop: "the next engineer, or the next agent, inherits the
finding instead of starting from zero." Writes the diagnosis back into
DataHub as a structured incident, and opens a GitHub issue with the
concrete suggested fix.
"""
from __future__ import annotations

import logging

import httpx

from app.config import settings
from app.lineage.datahub_client import BaseLineageClient
from app.models.schemas import RootCauseReport, WriteBackResult

logger = logging.getLogger(__name__)


GITHUB_API = "https://api.github.com"


def _render_incident_payload(report: RootCauseReport) -> dict:
    return {
        "summary": report.summary,
        "detailed_explanation": report.detailed_explanation,
        "root_causes": report.root_causes,
        "confidence": report.confidence,
        "suggested_fixes": [f.model_dump() for f in report.suggested_fixes],
        "model_urn": report.model_urn,
        "generated_at": report.generated_at.isoformat(),
        "generated_by": "causal-drift-sentinel-agent",
    }


def _render_github_issue_body(report: RootCauseReport) -> str:
    lines = [
        f"**Model:** `{report.model_urn}`",
        f"**Confidence:** {report.confidence}",
        "",
        "## Summary",
        report.summary,
        "",
        "## Root Cause Analysis",
        report.detailed_explanation,
        "",
        "## Isolated Root Cause(s)",
    ]
    lines += [f"- `{rc}`" for rc in report.root_causes] or ["- (none isolated)"]
    lines += ["", "## Suggested Fixes"]
    for fix in report.suggested_fixes:
        lines.append(f"- **{fix.action}** on `{fix.target_urn}` — {fix.rationale}")
    lines += [
        "",
        "## Statistical Evidence",
        f"- Prediction drift: `{report.raw_trace.prediction_drift.method.value}` "
        f"statistic={report.raw_trace.prediction_drift.statistic:.4f}, "
        f"p={report.raw_trace.prediction_drift.p_value:.4g}, "
        f"severity={report.raw_trace.prediction_drift.severity.value}",
        "",
        "| Candidate | Hops | Method | Statistic | Intervention Δ | Genuine Cause? |",
        "|---|---|---|---|---|---|",
    ]
    for c in report.raw_trace.candidates_examined:
        lines.append(
            f"| `{c.node_name}` | {c.hops_from_model} | {c.drift_result.method.value} "
            f"| {c.drift_result.statistic:.4f} | {c.intervention_delta:.4f} | "
            f"{'✅' if c.is_genuine_cause else '❌'} |"
        )
    lines += [
        "",
        "---",
        "_Opened automatically by Causal Drift Sentinel — root cause isolated "
        "algorithmically via lineage graph traversal + intervention-style drift "
        "testing; this text is the LLM explanation layer phrasing that finding._",
    ]
    return "\n".join(lines)


class WriteBackAgent:
    def __init__(self, lineage_client: BaseLineageClient):
        self.lineage_client = lineage_client

    async def write_datahub_incident(self, report: RootCauseReport) -> str | None:
        payload = _render_incident_payload(report)
        return await self.lineage_client.write_incident(report.model_urn, payload)

    async def open_github_issue(self, report: RootCauseReport) -> str | None:
        if not settings.WRITEBACK_ENABLED:
            return None  # safety gate: disabled by default for public demo deployments
        if not settings.GITHUB_TOKEN or not settings.GITHUB_REPO:
            return None
        title = f"[Drift Detected] {report.summary[:80]}"
        body = _render_github_issue_body(report)
        async with httpx.AsyncClient(
            base_url=GITHUB_API,
            headers={
                "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
            },
            timeout=15.0,
        ) as client:
            resp = await client.post(
                f"/repos/{settings.GITHUB_REPO}/issues",
                json={"title": title, "body": body, "labels": ["drift-detected", "auto-generated"]},
            )
            resp.raise_for_status()
            return resp.json().get("html_url")

    async def run(self, report: RootCauseReport) -> WriteBackResult:
        incident_urn = None
        try:
            incident_urn = await self.write_datahub_incident(report)
        except Exception as exc:  # noqa: BLE001
            # DataHub write-back is best-effort, same as GitHub below — a
            # misconfigured or unreachable DataHub instance shouldn't crash
            # the whole /investigate request when the causal finding itself
            # is already valid and returned regardless.
            logger.warning("DataHub write-back failed: %s", exc, exc_info=True)
            incident_urn = None

        issue_url = None
        try:
            issue_url = await self.open_github_issue(report)
        except httpx.HTTPError as exc:
            logger.warning("GitHub write-back failed: %s", exc)
            issue_url = None  # non-fatal: GitHub write-back is best-effort in demo mode

        return WriteBackResult(
            datahub_incident_urn=incident_urn,
            github_issue_url=issue_url,
            github_pr_url=None,
            status="completed" if incident_urn else "partial",
        )
