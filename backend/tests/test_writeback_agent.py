"""
Tests for the Write-Back Agent's two fixes found this round:

1. A latent crash introduced by the drift engine's own fix from a prior
   round: PredictionDriftResult.p_value can now legitimately be None (for
   DriftSeverity.INSUFFICIENT_DATA), but _render_github_issue_body's
   f-string formatting assumed it was always a real float and crashed
   with an unhandled TypeError.

2. The exact same GitHub-issue idempotency gap already found and fixed in
   the Executor's escalation ticket handler — open_github_issue had zero
   protection against creating duplicate real issues for the same
   underlying finding.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import settings
from app.models.schemas import (
    DriftMethod,
    DriftSeverity,
    PredictionDriftResult,
    RootCauseReport,
    RootCauseTrace,
)
from app.utils.github_dedup import compute_issue_fingerprint
from app.writeback.agent import WriteBackAgent, _render_github_issue_body


def _make_report(p_value, severity, root_causes=None) -> RootCauseReport:
    pred = PredictionDriftResult(
        model_urn="urn:test:model", method=DriftMethod.KS_TEST, statistic=0.0,
        p_value=p_value, severity=severity, detected_at=datetime.now(timezone.utc),
    )
    trace = RootCauseTrace(
        model_urn="urn:test:model", prediction_drift=pred,
        candidates_examined=[], isolated_root_causes=[], graph_path=[],
    )
    return RootCauseReport(
        model_urn="urn:test:model", generated_at=datetime.now(timezone.utc),
        summary="s", detailed_explanation="d", root_causes=root_causes or [],
        confidence="high", suggested_fixes=[], raw_trace=trace,
    )


class TestRenderGithubIssueBodyPValueNoneSafety:
    def test_none_p_value_does_not_crash(self):
        report = _make_report(p_value=None, severity=DriftSeverity.INSUFFICIENT_DATA)
        body = _render_github_issue_body(report)
        assert "n/a" in body.lower()

    def test_real_p_value_still_formats_normally(self):
        report = _make_report(p_value=0.00123, severity=DriftSeverity.CRITICAL)
        body = _render_github_issue_body(report)
        assert "0.00123" in body or "0.001230" in body

    def test_zero_p_value_formats_as_number_not_none_path(self):
        report = _make_report(p_value=0.0, severity=DriftSeverity.CRITICAL)
        body = _render_github_issue_body(report)
        assert "n/a" not in body.lower()


class TestOpenGithubIssueIdempotency:
    @pytest.mark.asyncio
    async def test_skips_creation_when_matching_open_issue_exists(self, monkeypatch):
        monkeypatch.setattr(settings, "WRITEBACK_ENABLED", True)
        monkeypatch.setattr(settings, "GITHUB_TOKEN", "fake-token")
        monkeypatch.setattr(settings, "GITHUB_REPO", "owner/repo")

        report = _make_report(p_value=0.01, severity=DriftSeverity.HIGH, root_causes=["urn:cause:a"])
        fingerprint = compute_issue_fingerprint(report.model_urn, ",".join(sorted(report.root_causes)))

        mock_search_response = MagicMock()
        mock_search_response.raise_for_status = MagicMock()
        mock_search_response.json.return_value = {
            "items": [{
                "title": f"[nexus:{fingerprint}] [Drift Detected] something",
                "state": "open",
                "html_url": "https://github.com/owner/repo/issues/7",
            }]
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_search_response
        mock_client.post = AsyncMock()

        agent = WriteBackAgent(lineage_client=MagicMock())
        with patch("httpx.AsyncClient") as mock_ac:
            mock_ac.return_value.__aenter__.return_value = mock_client
            url = await agent.open_github_issue(report)

        mock_client.post.assert_not_called()
        assert url == "https://github.com/owner/repo/issues/7"

    @pytest.mark.asyncio
    async def test_creates_new_issue_when_no_match(self, monkeypatch):
        monkeypatch.setattr(settings, "WRITEBACK_ENABLED", True)
        monkeypatch.setattr(settings, "GITHUB_TOKEN", "fake-token")
        monkeypatch.setattr(settings, "GITHUB_REPO", "owner/repo")

        report = _make_report(p_value=0.01, severity=DriftSeverity.HIGH, root_causes=["urn:cause:a"])

        mock_search_response = MagicMock()
        mock_search_response.raise_for_status = MagicMock()
        mock_search_response.json.return_value = {"items": []}

        mock_create_response = MagicMock()
        mock_create_response.raise_for_status = MagicMock()
        mock_create_response.json.return_value = {"html_url": "https://github.com/owner/repo/issues/55"}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_search_response
        mock_client.post.return_value = mock_create_response

        agent = WriteBackAgent(lineage_client=MagicMock())
        with patch("httpx.AsyncClient") as mock_ac:
            mock_ac.return_value.__aenter__.return_value = mock_client
            url = await agent.open_github_issue(report)

        mock_client.post.assert_called_once()
        assert url == "https://github.com/owner/repo/issues/55"

    @pytest.mark.asyncio
    async def test_different_root_causes_produce_different_fingerprints(self, monkeypatch):
        monkeypatch.setattr(settings, "WRITEBACK_ENABLED", True)
        monkeypatch.setattr(settings, "GITHUB_TOKEN", "fake-token")
        monkeypatch.setattr(settings, "GITHUB_REPO", "owner/repo")

        report_a = _make_report(p_value=0.01, severity=DriftSeverity.HIGH, root_causes=["urn:cause:a"])
        report_b = _make_report(p_value=0.01, severity=DriftSeverity.HIGH, root_causes=["urn:cause:b"])

        fp_a = compute_issue_fingerprint(report_a.model_urn, ",".join(sorted(report_a.root_causes)))
        fp_b = compute_issue_fingerprint(report_b.model_urn, ",".join(sorted(report_b.root_causes)))
        assert fp_a != fp_b

    @pytest.mark.asyncio
    async def test_search_failure_falls_through_to_create(self, monkeypatch):
        monkeypatch.setattr(settings, "WRITEBACK_ENABLED", True)
        monkeypatch.setattr(settings, "GITHUB_TOKEN", "fake-token")
        monkeypatch.setattr(settings, "GITHUB_REPO", "owner/repo")

        report = _make_report(p_value=0.01, severity=DriftSeverity.HIGH, root_causes=["urn:cause:a"])

        mock_create_response = MagicMock()
        mock_create_response.raise_for_status = MagicMock()
        mock_create_response.json.return_value = {"html_url": "https://github.com/owner/repo/issues/88"}

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("rate limited")
        mock_client.post.return_value = mock_create_response

        agent = WriteBackAgent(lineage_client=MagicMock())
        with patch("httpx.AsyncClient") as mock_ac:
            mock_ac.return_value.__aenter__.return_value = mock_client
            url = await agent.open_github_issue(report)

        mock_client.post.assert_called_once()
        assert url == "https://github.com/owner/repo/issues/88"

    @pytest.mark.asyncio
    async def test_disabled_writeback_returns_none_without_any_http_calls(self, monkeypatch):
        monkeypatch.setattr(settings, "WRITEBACK_ENABLED", False)
        report = _make_report(p_value=0.01, severity=DriftSeverity.HIGH)
        agent = WriteBackAgent(lineage_client=MagicMock())
        with patch("httpx.AsyncClient") as mock_ac:
            url = await agent.open_github_issue(report)
        mock_ac.assert_not_called()
        assert url is None


class TestSharedDedupModuleUsedConsistently:
    def test_executor_and_writeback_import_same_fingerprint_function(self):
        from app.agents.executor import compute_issue_fingerprint as exec_fp
        from app.writeback.agent import compute_issue_fingerprint as wb_fp
        assert exec_fp is wb_fp

    def test_executor_and_writeback_import_same_matcher_function(self):
        from app.agents.executor import find_matching_open_issue_url as exec_match
        from app.writeback.agent import find_matching_open_issue_url as wb_match
        assert exec_match is wb_match
