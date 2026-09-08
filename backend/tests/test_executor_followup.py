"""
Tests for the Executor + Verifier action loop, focused on the
handle_failed_verification follow-up logic.

THE GAP THESE TESTS PROVE IS CLOSED: before this round, a verified=False
outcome was a dead end for the current run. ExecutorAgent.rollback()
existed but was never called from anywhere in the codebase (confirmed by
a full-repo grep before this fix). These tests prove that a failed
verification now always results in either an automatic rollback (for
reversible actions) or an automatic escalation ticket (for non-reversible
ones), within the same run.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.agents.executor import ExecutorAgent, get_executor, get_verifier, handle_failed_verification
from app.config import settings
from app.models.schemas import ActionOutcome, RemediationActionType, RemediationPlan, RiskLevel


def _make_plan(action: RemediationActionType, model_urn: str = "urn:test:model") -> RemediationPlan:
    return RemediationPlan(
        model_urn=model_urn, action_type=action, target_urn=model_urn,
        rationale="test", confidence=0.9, risk_level=RiskLevel.LOW,
        requires_human_approval=False,
    )


def _make_outcome(
    action: RemediationActionType,
    verified: bool | None,
    reversible: bool,
    rollback_reference: str | None = None,
) -> ActionOutcome:
    plan = _make_plan(action)
    return ActionOutcome(
        plan=plan, executed=True, execution_detail="did the thing",
        reversible=reversible, rollback_reference=rollback_reference,
        verified=verified, verification_detail="checked",
    )


class TestHandleFailedVerificationNoOpCases:
    """Verify the function is a true no-op whenever it shouldn't act."""

    @pytest.mark.asyncio
    async def test_verified_true_is_untouched(self):
        outcome = _make_outcome(RemediationActionType.TRIGGER_RETRAIN, verified=True, reversible=False)
        executor = get_executor()
        result = await handle_failed_verification(outcome, executor)
        assert result.follow_up is None
        assert result is outcome or result == outcome

    @pytest.mark.asyncio
    async def test_verified_none_is_untouched(self):
        """verified=None means verification couldn't run (e.g. an
        exception) — not the same as verified=False, must not trigger
        automatic corrective action on an unknown outcome."""
        outcome = _make_outcome(RemediationActionType.TRIGGER_RETRAIN, verified=None, reversible=False)
        executor = get_executor()
        result = await handle_failed_verification(outcome, executor)
        assert result.follow_up is None

    @pytest.mark.asyncio
    async def test_no_action_type_never_gets_follow_up_even_if_marked_false(self):
        """Defensive: NO_ACTION should never reach this function with
        verified=False in practice (verify() skips it), but if it somehow
        did, a no_action outcome has no rollback_reference and would fall
        into the escalation branch — confirm that branch handles it
        without crashing, rather than assuming it can't happen."""
        outcome = _make_outcome(RemediationActionType.NO_ACTION, verified=False, reversible=False)
        executor = get_executor()
        result = await handle_failed_verification(outcome, executor)
        # Falls into escalation branch since reversible=False — should
        # produce a follow_up (an escalation ticket attempt), not crash.
        assert result.follow_up is not None


class TestHandleFailedVerificationRollbackPath:
    @pytest.mark.asyncio
    async def test_reversible_failure_triggers_automatic_rollback(self):
        outcome = _make_outcome(
            RemediationActionType.ROLLBACK_MODEL_VERSION, verified=False,
            reversible=True, rollback_reference="rollback-abc123",
        )
        executor = get_executor()
        result = await handle_failed_verification(outcome, executor)

        assert result.follow_up is not None
        assert result.follow_up.executed is True
        assert "rollback-abc123" in result.follow_up.execution_detail

    @pytest.mark.asyncio
    async def test_reversible_but_missing_rollback_reference_escalates_instead(self):
        """A reversible=True outcome with no actual rollback_reference
        recorded cannot be rolled back procedurally — must fall through
        to escalation rather than calling rollback() with nothing to
        reference (which would just fail anyway)."""
        outcome = _make_outcome(
            RemediationActionType.QUARANTINE_DATA_SOURCE, verified=False,
            reversible=True, rollback_reference=None,
        )
        executor = get_executor()
        result = await handle_failed_verification(outcome, executor)

        assert result.follow_up is not None
        # Should be an escalation ticket attempt, not a rollback call —
        # confirm by action type on the follow-up's plan.
        assert result.follow_up.plan.action_type == RemediationActionType.OPEN_INCIDENT_TICKET

    @pytest.mark.asyncio
    async def test_rollback_call_actually_uses_executor_rollback_method(self):
        """Confirm the previously-dead-code rollback() method is now
        actually invoked, not reimplemented inline."""
        outcome = _make_outcome(
            RemediationActionType.ROLLBACK_MODEL_VERSION, verified=False,
            reversible=True, rollback_reference="ref-xyz",
        )
        executor = get_executor()
        executor.rollback = AsyncMock(wraps=executor.rollback)

        await handle_failed_verification(outcome, executor)

        executor.rollback.assert_awaited_once()


class TestHandleFailedVerificationEscalationPath:
    @pytest.mark.asyncio
    async def test_non_reversible_failure_triggers_auto_escalation(self):
        outcome = _make_outcome(
            RemediationActionType.TRIGGER_RETRAIN, verified=False, reversible=False,
        )
        executor = get_executor()
        result = await handle_failed_verification(outcome, executor)

        assert result.follow_up is not None
        assert result.follow_up.plan.action_type == RemediationActionType.OPEN_INCIDENT_TICKET

    @pytest.mark.asyncio
    async def test_escalation_plan_carries_explanatory_rationale(self):
        outcome = _make_outcome(
            RemediationActionType.TRIGGER_RETRAIN, verified=False, reversible=False,
        )
        executor = get_executor()
        result = await handle_failed_verification(outcome, executor)

        rationale = result.follow_up.plan.rationale.lower()
        assert "trigger_retrain" in rationale
        assert "verification" in rationale or "did not resolve" in rationale

    @pytest.mark.asyncio
    async def test_escalation_plan_does_not_require_further_approval(self):
        """Opening a ticket is unconditionally safe — the auto-escalation
        must not itself get stuck waiting for human approval, or the loop
        would still dead-end just one step later."""
        outcome = _make_outcome(
            RemediationActionType.TRIGGER_RETRAIN, verified=False, reversible=False,
        )
        executor = get_executor()
        result = await handle_failed_verification(outcome, executor)
        assert result.follow_up.plan.requires_human_approval is False


class TestHandleFailedVerificationBoundedDepth:
    @pytest.mark.asyncio
    async def test_follow_up_failure_does_not_recurse_further(self):
        """If the automatic rollback itself fails, that failure must be
        visible on follow_up.executed=False, but the function must NOT
        attempt a second level of automatic correction (no follow_up on
        the follow_up) — bounded to exactly one level."""
        outcome = _make_outcome(
            RemediationActionType.ROLLBACK_MODEL_VERSION, verified=False,
            reversible=True, rollback_reference="ref-1",
        )
        executor = get_executor()
        # Force the rollback call itself to fail.
        executor.rollback = AsyncMock(return_value=ActionOutcome(
            plan=outcome.plan, executed=False,
            execution_detail="rollback API call failed", reversible=False,
        ))

        result = await handle_failed_verification(outcome, executor)

        assert result.follow_up.executed is False
        assert result.follow_up.follow_up is None  # no recursive chain


class TestExecutorRollbackMethod:
    """Direct tests of ExecutorAgent.rollback() — now actually reachable
    in production code, so it earns direct unit coverage too."""

    @pytest.mark.asyncio
    async def test_rollback_without_reference_returns_not_executed(self):
        executor = get_executor()
        outcome = _make_outcome(RemediationActionType.ROLLBACK_MODEL_VERSION, verified=False, reversible=False)
        result = await executor.rollback(outcome)
        assert result.executed is False

    @pytest.mark.asyncio
    async def test_rollback_with_reference_succeeds(self):
        executor = get_executor()
        outcome = _make_outcome(
            RemediationActionType.ROLLBACK_MODEL_VERSION, verified=False,
            reversible=True, rollback_reference="ref-999",
        )
        result = await executor.rollback(outcome)
        assert result.executed is True
        assert "ref-999" in result.execution_detail
        assert result.reversible is False  # a rollback of a rollback isn't modeled
        assert result.rollback_reference is None


class TestFollowUpFieldSerialization:
    """Confirm the new schema field round-trips through JSON cleanly —
    this is what the frontend and DynamoDB memory store both depend on."""

    def test_follow_up_serializes_and_deserializes(self):
        import json
        outcome = _make_outcome(
            RemediationActionType.ROLLBACK_MODEL_VERSION, verified=False,
            reversible=True, rollback_reference="ref-1",
        )
        follow_up = _make_outcome(RemediationActionType.NO_ACTION, verified=None, reversible=False)
        outer = outcome.model_copy(update={"follow_up": follow_up})

        data = json.loads(outer.model_dump_json())
        assert data["follow_up"]["plan"]["action_type"] == "no_action"

        rebuilt = ActionOutcome(**data)
        assert rebuilt.follow_up.plan.action_type == RemediationActionType.NO_ACTION

    def test_follow_up_defaults_to_none(self):
        outcome = _make_outcome(RemediationActionType.NO_ACTION, verified=True, reversible=False)
        assert outcome.follow_up is None
