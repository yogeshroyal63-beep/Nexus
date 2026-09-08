"""
Tests for the Executor's GitHub-issue idempotency fix.

THE GAP THESE TESTS PROVE IS CLOSED: _open_incident_ticket previously had
no protection against creating duplicate real GitHub issues if execute()
ran twice for the same underlying escalation (duplicate SNS delivery, a
retry after a network blip, two overlapping coordinator runs). These tests
cover the two pure functions that make the fix work — fingerprint
computation and search-result matching — plus an end-to-end mocked-httpx
test proving a duplicate create is actually avoided.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.executor import (
    ExecutorAgent,
    _compute_escalation_fingerprint,
    _find_matching_open_issue_url,
)
from app.config import settings
from app.models.schemas import RemediationActionType, RemediationPlan, RiskLevel


def _make_plan(model_urn="urn:test:model", target_urn="urn:test:model") -> RemediationPlan:
    return RemediationPlan(
        model_urn=model_urn, action_type=RemediationActionType.OPEN_INCIDENT_TICKET,
        target_urn=target_urn, rationale="r", confidence=0.9,
        risk_level=RiskLevel.LOW, requires_human_approval=False,
    )


class TestComputeEscalationFingerprint:
    def test_deterministic_for_same_inputs(self):
        a = _compute_escalation_fingerprint("urn:m", "urn:t", "trigger_retrain")
        b = _compute_escalation_fingerprint("urn:m", "urn:t", "trigger_retrain")
        assert a == b

    def test_different_for_different_model_urn(self):
        a = _compute_escalation_fingerprint("urn:m1", "urn:t", "trigger_retrain")
        b = _compute_escalation_fingerprint("urn:m2", "urn:t", "trigger_retrain")
        assert a != b

    def test_different_for_different_target_urn(self):
        a = _compute_escalation_fingerprint("urn:m", "urn:t1", "trigger_retrain")
        b = _compute_escalation_fingerprint("urn:m", "urn:t2", "trigger_retrain")
        assert a != b

    def test_different_for_different_action_type(self):
        a = _compute_escalation_fingerprint("urn:m", "urn:t", "trigger_retrain")
        b = _compute_escalation_fingerprint("urn:m", "urn:t", "rollback_model_version")
        assert a != b

    def test_is_short_and_url_safe(self):
        fp = _compute_escalation_fingerprint("urn:m", "urn:t", "trigger_retrain")
        assert len(fp) == 12
        assert fp.isalnum()


class TestFindMatchingOpenIssueUrl:
    def test_finds_matching_open_issue(self):
        fingerprint = "abc123def456"
        search_response = {
            "items": [
                {"title": f"[nexus:{fingerprint}] Escalated action needed: retrain", "state": "open", "html_url": "https://github.com/x/y/issues/1"},
            ]
        }
        url = _find_matching_open_issue_url(search_response, fingerprint)
        assert url == "https://github.com/x/y/issues/1"

    def test_ignores_closed_issue_with_matching_fingerprint(self):
        """A CLOSED issue with the same fingerprint means the previous
        escalation was already resolved — must not treat it as 'already
        open', or a genuinely new occurrence would never get a ticket."""
        fingerprint = "abc123def456"
        search_response = {
            "items": [
                {"title": f"[nexus:{fingerprint}] Escalated action needed: retrain", "state": "closed", "html_url": "https://github.com/x/y/issues/1"},
            ]
        }
        url = _find_matching_open_issue_url(search_response, fingerprint)
        assert url is None

    def test_ignores_non_matching_fingerprint(self):
        search_response = {
            "items": [
                {"title": "[nexus:zzz999zzz999] Some other escalation", "state": "open", "html_url": "https://github.com/x/y/issues/2"},
            ]
        }
        url = _find_matching_open_issue_url(search_response, "abc123def456")
        assert url is None

    def test_empty_items_returns_none(self):
        assert _find_matching_open_issue_url({"items": []}, "abc123def456") is None

    def test_missing_items_key_returns_none_not_crash(self):
        assert _find_matching_open_issue_url({}, "abc123def456") is None

    def test_case_insensitive_fingerprint_match(self):
        fingerprint = "ABC123DEF456"
        search_response = {
            "items": [
                {"title": f"[Nexus:{fingerprint.lower()}] Escalated action", "state": "open", "html_url": "https://github.com/x/y/issues/3"},
            ]
        }
        url = _find_matching_open_issue_url(search_response, fingerprint.lower())
        assert url == "https://github.com/x/y/issues/3"


class TestOpenIncidentTicketIdempotency:
    """End-to-end (mocked httpx) proof that a duplicate create is avoided."""

    @pytest.mark.asyncio
    async def test_skips_creation_when_matching_open_issue_exists(self, monkeypatch):
        monkeypatch.setattr(settings, "WRITEBACK_ENABLED", True)
        monkeypatch.setattr(settings, "GITHUB_TOKEN", "fake-token")
        monkeypatch.setattr(settings, "GITHUB_REPO", "owner/repo")

        plan = _make_plan()
        fingerprint = _compute_escalation_fingerprint(
            plan.model_urn, plan.target_urn, plan.action_type.value
        )

        mock_search_response = MagicMock()
        mock_search_response.raise_for_status = MagicMock()
        mock_search_response.json.return_value = {
            "items": [{
                "title": f"[nexus:{fingerprint}] Escalated action needed: x",
                "state": "open",
                "html_url": "https://github.com/owner/repo/issues/42",
            }]
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_search_response
        mock_client.post = AsyncMock()  # must NOT be called

        with patch("httpx.AsyncClient") as mock_ac:
            mock_ac.return_value.__aenter__.return_value = mock_client
            executor = ExecutorAgent()
            outcome = await executor._open_incident_ticket(plan)

        mock_client.post.assert_not_called()
        assert outcome.executed is True
        assert "already exists" in outcome.execution_detail
        assert "issues/42" in outcome.execution_detail

    @pytest.mark.asyncio
    async def test_creates_new_issue_when_no_match_found(self, monkeypatch):
        monkeypatch.setattr(settings, "WRITEBACK_ENABLED", True)
        monkeypatch.setattr(settings, "GITHUB_TOKEN", "fake-token")
        monkeypatch.setattr(settings, "GITHUB_REPO", "owner/repo")

        plan = _make_plan()

        mock_search_response = MagicMock()
        mock_search_response.raise_for_status = MagicMock()
        mock_search_response.json.return_value = {"items": []}

        mock_create_response = MagicMock()
        mock_create_response.raise_for_status = MagicMock()
        mock_create_response.json.return_value = {"html_url": "https://github.com/owner/repo/issues/99"}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_search_response
        mock_client.post.return_value = mock_create_response

        with patch("httpx.AsyncClient") as mock_ac:
            mock_ac.return_value.__aenter__.return_value = mock_client
            executor = ExecutorAgent()
            outcome = await executor._open_incident_ticket(plan)

        mock_client.post.assert_called_once()
        assert outcome.executed is True
        assert "issues/99" in outcome.execution_detail

    @pytest.mark.asyncio
    async def test_search_failure_falls_through_to_create_not_silently_skips(self, monkeypatch):
        """A transient search failure (rate limit, network blip) must not
        cause the escalation to be silently dropped — fail open to
        creating a ticket, since a missed escalation is worse than an
        occasional duplicate."""
        monkeypatch.setattr(settings, "WRITEBACK_ENABLED", True)
        monkeypatch.setattr(settings, "GITHUB_TOKEN", "fake-token")
        monkeypatch.setattr(settings, "GITHUB_REPO", "owner/repo")

        plan = _make_plan()

        mock_create_response = MagicMock()
        mock_create_response.raise_for_status = MagicMock()
        mock_create_response.json.return_value = {"html_url": "https://github.com/owner/repo/issues/100"}

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("rate limited")
        mock_client.post.return_value = mock_create_response

        with patch("httpx.AsyncClient") as mock_ac:
            mock_ac.return_value.__aenter__.return_value = mock_client
            executor = ExecutorAgent()
            outcome = await executor._open_incident_ticket(plan)

        mock_client.post.assert_called_once()
        assert outcome.executed is True
        assert "issues/100" in outcome.execution_detail

    @pytest.mark.asyncio
    async def test_fingerprint_is_embedded_in_created_issue_title(self, monkeypatch):
        """Confirm future searches will actually be able to find this
        issue — the fingerprint must be in the title GitHub indexes."""
        monkeypatch.setattr(settings, "WRITEBACK_ENABLED", True)
        monkeypatch.setattr(settings, "GITHUB_TOKEN", "fake-token")
        monkeypatch.setattr(settings, "GITHUB_REPO", "owner/repo")

        plan = _make_plan()
        expected_fingerprint = _compute_escalation_fingerprint(
            plan.model_urn, plan.target_urn, plan.action_type.value
        )

        mock_search_response = MagicMock()
        mock_search_response.raise_for_status = MagicMock()
        mock_search_response.json.return_value = {"items": []}

        mock_create_response = MagicMock()
        mock_create_response.raise_for_status = MagicMock()
        mock_create_response.json.return_value = {"html_url": "https://github.com/owner/repo/issues/1"}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_search_response
        mock_client.post.return_value = mock_create_response

        with patch("httpx.AsyncClient") as mock_ac:
            mock_ac.return_value.__aenter__.return_value = mock_client
            executor = ExecutorAgent()
            await executor._open_incident_ticket(plan)

        call_kwargs = mock_client.post.call_args
        posted_title = call_kwargs.kwargs["json"]["title"]
        assert f"[nexus:{expected_fingerprint}]" in posted_title
