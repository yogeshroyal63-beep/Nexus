"""
Tests for the Strands Planner agent.

The most important tests here are the ones proving the risk-level
self-report bypass (documented in planner.py's module docstring) is
actually closed: a model that picks a high-risk action but self-reports
"low" risk must NOT get its self-report honored.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.agents.planner import (
    StrandsPlannerAgent,
    _assess_risk_level_logic,
    _build_planner_context,
    _check_past_incidents_logic,
    _enforce_target_grounding,
    _extract_json_object,
    _history_to_summary,
    _safe_escalation_plan,
)
from app.models.schemas import (
    ActionOutcome,
    IncidentRecord,
    PredictionDriftResult,
    DriftMethod,
    DriftSeverity,
    RemediationActionType,
    RemediationPlan,
    RiskLevel,
    RootCauseReport,
    RootCauseTrace,
    CausalCandidate,
    FeatureDriftResult,
)


def _make_trace(model_urn: str = "urn:test:model", root_cause_urn: str | None = "urn:test:cause"):
    pred = PredictionDriftResult(
        model_urn=model_urn, method=DriftMethod.KS_TEST, statistic=0.5,
        severity=DriftSeverity.HIGH, detected_at=datetime.now(timezone.utc),
    )
    isolated = []
    if root_cause_urn:
        drift_result = FeatureDriftResult(
            node_urn=root_cause_urn, feature_name="f", method=DriftMethod.KS_TEST,
            statistic=0.4, p_value=0.01, severity=DriftSeverity.HIGH,
            baseline_window="training", current_window="last_7d",
            sample_size_baseline=300, sample_size_current=300,
        )
        isolated = [CausalCandidate(
            node_urn=root_cause_urn, node_name="cause_node", hops_from_model=1,
            drift_result=drift_result, is_genuine_cause=True,
            fdr_adjusted_p_value=0.01, survived_fdr_correction=True,
            intervention_delta=0.3,
        )]
    return RootCauseTrace(
        model_urn=model_urn, prediction_drift=pred, candidates_examined=isolated,
        isolated_root_causes=isolated, graph_path=[root_cause_urn] if root_cause_urn else [],
    )


def _make_report(model_urn: str = "urn:test:model", confidence: str = "high", root_cause_urn="urn:test:cause"):
    trace = _make_trace(model_urn, root_cause_urn)
    return RootCauseReport(
        model_urn=model_urn, generated_at=datetime.now(timezone.utc),
        summary="s", detailed_explanation="d",
        root_causes=[root_cause_urn] if root_cause_urn else [],
        confidence=confidence, suggested_fixes=[], raw_trace=trace,
    )


def _make_incident(action: str, verified: bool, model_urn: str = "urn:test:model") -> IncidentRecord:
    plan = RemediationPlan(
        model_urn=model_urn, action_type=RemediationActionType(action), target_urn=model_urn,
        rationale="r", confidence=0.8, risk_level=RiskLevel.LOW, requires_human_approval=False,
    )
    outcome = ActionOutcome(plan=plan, executed=True, execution_detail="d", reversible=False, verified=verified)
    return IncidentRecord(
        incident_id="inc-1", model_urn=model_urn, created_at=datetime.now(timezone.utc),
        report=_make_report(model_urn), plan=plan, outcome=outcome,
    )


class TestAssessRiskLevelLogic:
    """Pure function — deterministic classification, exhaustively tested."""

    def test_rollback_is_always_high_risk(self):
        result = _assess_risk_level_logic("rollback_model_version", confidence=0.99, has_similar_failures=False)
        assert result["risk_level"] == "high"

    def test_retrain_is_medium_risk(self):
        result = _assess_risk_level_logic("trigger_retrain", confidence=0.99, has_similar_failures=False)
        assert result["risk_level"] == "medium"

    def test_quarantine_is_low_risk(self):
        result = _assess_risk_level_logic("quarantine_data_source", confidence=0.99, has_similar_failures=False)
        assert result["risk_level"] == "low"

    def test_open_ticket_is_low_risk(self):
        result = _assess_risk_level_logic("open_incident_ticket", confidence=0.99, has_similar_failures=False)
        assert result["risk_level"] == "low"

    def test_no_action_is_low_risk(self):
        result = _assess_risk_level_logic("no_action", confidence=0.99, has_similar_failures=False)
        assert result["risk_level"] == "low"

    def test_unknown_action_fails_closed_to_high_risk(self):
        """An unrecognized action string must never silently default to low risk."""
        result = _assess_risk_level_logic("some_made_up_action", confidence=0.99, has_similar_failures=False)
        assert result["risk_level"] == "high"
        assert result["requires_approval"] is True

    def test_high_risk_always_requires_approval_regardless_of_confidence(self):
        result = _assess_risk_level_logic("rollback_model_version", confidence=1.0, has_similar_failures=False)
        assert result["requires_approval"] is True

    def test_low_confidence_requires_approval_even_for_low_risk_action(self):
        result = _assess_risk_level_logic("no_action", confidence=0.1, has_similar_failures=False)
        assert result["requires_approval"] is True

    def test_similar_failures_forces_approval_even_for_low_risk_high_confidence(self):
        result = _assess_risk_level_logic("quarantine_data_source", confidence=0.99, has_similar_failures=True)
        assert result["requires_approval"] is True

    def test_medium_risk_high_confidence_no_failures_does_not_require_approval(self):
        result = _assess_risk_level_logic("trigger_retrain", confidence=0.9, has_similar_failures=False)
        assert result["requires_approval"] is False

    def test_auto_execute_disabled_forces_approval_regardless(self):
        with patch("app.agents.planner.settings") as mock_settings:
            mock_settings.AUTO_EXECUTE_ENABLED = False
            mock_settings.AUTO_EXECUTE_MIN_CONFIDENCE = 0.75
            result = _assess_risk_level_logic("no_action", confidence=1.0, has_similar_failures=False)
            assert result["requires_approval"] is True


class TestCheckPastIncidentsLogic:
    def test_no_incidents_no_failures(self):
        result = _check_past_incidents_logic([], "trigger_retrain")
        assert result["has_failures"] is False
        assert result["failure_count"] == 0

    def test_detects_matching_failed_action(self):
        incidents = [{"action_taken": "trigger_retrain", "verified_success": False}]
        result = _check_past_incidents_logic(incidents, "trigger_retrain")
        assert result["has_failures"] is True
        assert result["failure_count"] == 1

    def test_ignores_different_action_failures(self):
        incidents = [{"action_taken": "rollback_model_version", "verified_success": False}]
        result = _check_past_incidents_logic(incidents, "trigger_retrain")
        assert result["has_failures"] is False

    def test_ignores_successful_past_actions(self):
        incidents = [{"action_taken": "trigger_retrain", "verified_success": True}]
        result = _check_past_incidents_logic(incidents, "trigger_retrain")
        assert result["has_failures"] is False

    def test_ignores_unverified_past_actions(self):
        incidents = [{"action_taken": "trigger_retrain", "verified_success": None}]
        result = _check_past_incidents_logic(incidents, "trigger_retrain")
        assert result["has_failures"] is False

    def test_counts_multiple_failures(self):
        incidents = [
            {"action_taken": "trigger_retrain", "verified_success": False},
            {"action_taken": "trigger_retrain", "verified_success": False},
            {"action_taken": "trigger_retrain", "verified_success": True},
        ]
        result = _check_past_incidents_logic(incidents, "trigger_retrain")
        assert result["failure_count"] == 2


class TestExtractJsonObject:
    def test_extracts_clean_json(self):
        text = '{"a": 1, "b": 2}'
        assert _extract_json_object(text) == {"a": 1, "b": 2}

    def test_extracts_json_with_surrounding_prose(self):
        text = 'Here is my answer:\n{"action_type": "no_action", "confidence": 0.5}\nDone.'
        result = _extract_json_object(text)
        assert result["action_type"] == "no_action"

    def test_handles_braces_inside_rationale_string(self):
        """The critical case naive rfind() breaks on: braces INSIDE a
        string value should not be mistaken for object boundaries."""
        text = '{"action_type": "no_action", "rationale": "See config {retries: 3} for details", "confidence": 0.5}'
        result = _extract_json_object(text)
        assert result["rationale"] == "See config {retries: 3} for details"
        assert result["action_type"] == "no_action"

    def test_handles_nested_json_example_in_prose_before_real_answer(self):
        text = (
            'The format should look like {"example": true}. '
            'My actual answer: {"action_type": "trigger_retrain", "confidence": 0.8, '
            '"target_urn": "urn:x", "rationale": "r", "risk_level": "medium"}'
        )
        result = _extract_json_object(text)
        # First balanced object wins, deterministically.
        assert result == {"example": True}

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            _extract_json_object("no json here at all")

    def test_raises_on_unbalanced_braces(self):
        with pytest.raises(ValueError, match="Unbalanced"):
            _extract_json_object('{"a": 1, "b": {"c": 2}')


class TestEnforceTargetGrounding:
    def test_allows_model_urn_as_target(self):
        report = _make_report()
        parsed = {"target_urn": report.model_urn, "action_type": "trigger_retrain"}
        result = _enforce_target_grounding(dict(parsed), report)
        assert result["target_urn"] == report.model_urn
        assert result["action_type"] == "trigger_retrain"

    def test_allows_isolated_root_cause_urn_as_target(self):
        report = _make_report(root_cause_urn="urn:test:cause")
        parsed = {"target_urn": "urn:test:cause", "action_type": "quarantine_data_source"}
        result = _enforce_target_grounding(dict(parsed), report)
        assert result["target_urn"] == "urn:test:cause"

    def test_forces_no_action_on_hallucinated_target(self):
        report = _make_report(root_cause_urn="urn:test:cause")
        parsed = {
            "target_urn": "urn:completely:made:up",
            "action_type": "rollback_model_version",
            "confidence": 0.99,
            "risk_level": "high",
        }
        result = _enforce_target_grounding(dict(parsed), report)
        assert result["action_type"] == "no_action"
        assert result["target_urn"] == report.model_urn
        assert result["confidence"] == 0.0


class TestSafeEscalationPlan:
    def test_always_requires_approval(self):
        plan = _safe_escalation_plan(_make_report(), "timeout")
        assert plan.requires_human_approval is True
        assert plan.risk_level == RiskLevel.HIGH
        assert plan.action_type == RemediationActionType.OPEN_INCIDENT_TICKET


class TestPlanRiskOverrideSafety:
    """
    THE critical safety tests: prove the model cannot bypass deterministic
    risk assessment by self-reporting a lower risk_level than its chosen
    action_type actually warrants.
    """

    def _stub_agent_returning(self, json_payload: dict):
        planner = StrandsPlannerAgent()
        fake_response = json.dumps(json_payload)
        fake_agent = MagicMock(return_value=fake_response)
        planner._get_agent = MagicMock(return_value=fake_agent)
        return planner

    @pytest.mark.asyncio
    async def test_model_self_reporting_low_risk_for_high_risk_action_is_overridden(self):
        """
        THE core vulnerability this session's audit found and fixed:
        model picks rollback_model_version (tool-classified HIGH risk) but
        writes "risk_level": "low" in its JSON. The final RemediationPlan
        must show risk_level=HIGH and requires_human_approval=True
        regardless of what the model claimed.
        """
        report = _make_report(root_cause_urn="urn:test:cause")
        planner = self._stub_agent_returning({
            "action_type": "rollback_model_version",
            "target_urn": "urn:test:cause",
            "rationale": "Rolling back to fix it",
            "confidence": 0.95,
            "risk_level": "low",
        })

        plan = await planner.plan(report, history=[])

        assert plan.risk_level == RiskLevel.HIGH, (
            "Model's false 'low' self-report must NOT override the "
            "deterministic classification of rollback_model_version as high risk"
        )
        assert plan.requires_human_approval is True, (
            "A high-risk action must always require human approval, "
            "regardless of what risk_level the model claimed"
        )

    @pytest.mark.asyncio
    async def test_model_self_reporting_high_risk_for_low_risk_action_is_corrected(self):
        report = _make_report(root_cause_urn="urn:test:cause")
        planner = self._stub_agent_returning({
            "action_type": "no_action",
            "target_urn": report.model_urn,
            "rationale": "Nothing to do",
            "confidence": 0.95,
            "risk_level": "high",
        })

        plan = await planner.plan(report, history=[])
        assert plan.risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_past_failure_forces_approval_even_with_low_risk_action(self):
        report = _make_report(root_cause_urn="urn:test:cause")
        history = [_make_incident("quarantine_data_source", verified=False)]
        planner = self._stub_agent_returning({
            "action_type": "quarantine_data_source",
            "target_urn": "urn:test:cause",
            "rationale": "Quarantine it",
            "confidence": 0.9,
            "risk_level": "low",
        })

        plan = await planner.plan(report, history=history)
        assert plan.requires_human_approval is True

    @pytest.mark.asyncio
    async def test_confidence_out_of_range_is_clamped_not_rejected(self):
        report = _make_report(root_cause_urn="urn:test:cause")
        planner = self._stub_agent_returning({
            "action_type": "no_action",
            "target_urn": report.model_urn,
            "rationale": "r",
            "confidence": 1.4,
            "risk_level": "low",
        })
        plan = await planner.plan(report, history=[])
        assert plan.confidence == 1.0


class TestPlanShortCircuits:
    @pytest.mark.asyncio
    async def test_no_root_causes_returns_no_action_without_calling_agent(self):
        report = _make_report(root_cause_urn=None, confidence="low")
        planner = StrandsPlannerAgent()
        planner._get_agent = MagicMock(side_effect=AssertionError("should not be called"))
        plan = await planner.plan(report, history=[])
        assert plan.action_type == RemediationActionType.NO_ACTION
        assert plan.requires_human_approval is False


class TestPlanRetryAndEscalation:
    @pytest.mark.asyncio
    async def test_retries_once_on_malformed_json_then_succeeds(self):
        report = _make_report(root_cause_urn="urn:test:cause")
        planner = StrandsPlannerAgent()

        call_count = {"n": 0}

        def fake_agent(_message):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "not valid json at all"
            return json.dumps({
                "action_type": "no_action",
                "target_urn": report.model_urn,
                "rationale": "r",
                "confidence": 0.5,
                "risk_level": "low",
            })

        planner._get_agent = MagicMock(return_value=fake_agent)
        plan = await planner.plan(report, history=[])

        assert call_count["n"] == 2
        assert plan.action_type == RemediationActionType.NO_ACTION

    @pytest.mark.asyncio
    async def test_escalates_after_exhausting_retries(self):
        report = _make_report(root_cause_urn="urn:test:cause")
        planner = StrandsPlannerAgent()
        planner._get_agent = MagicMock(return_value=lambda _msg: "still not json")

        plan = await planner.plan(report, history=[])

        assert plan.requires_human_approval is True
        assert plan.action_type == RemediationActionType.OPEN_INCIDENT_TICKET

    @pytest.mark.asyncio
    async def test_invalid_action_type_string_triggers_retry_not_crash(self):
        report = _make_report(root_cause_urn="urn:test:cause")
        planner = StrandsPlannerAgent()
        planner._get_agent = MagicMock(return_value=lambda _msg: json.dumps({
            "action_type": "delete_the_entire_database",
            "target_urn": report.model_urn,
            "rationale": "r",
            "confidence": 0.9,
            "risk_level": "low",
        }))

        plan = await planner.plan(report, history=[])
        assert plan.requires_human_approval is True


class TestBuildPlannerContext:
    def test_includes_valid_target_urns(self):
        report = _make_report(root_cause_urn="urn:test:cause")
        context = _build_planner_context(report, history=[])
        payload = json.loads(context)
        assert report.model_urn in payload["valid_target_urns"]
        assert "urn:test:cause" in payload["valid_target_urns"]

    def test_includes_history_summary(self):
        report = _make_report()
        history = [_make_incident("trigger_retrain", verified=True)]
        context = _build_planner_context(report, history)
        payload = json.loads(context)
        assert len(payload["past_incidents_this_model"]) == 1
        assert payload["past_incidents_this_model"][0]["action_taken"] == "trigger_retrain"


class TestHistoryToSummary:
    def test_handles_incident_with_no_plan_or_outcome(self):
        report = _make_report()
        incident = IncidentRecord(
            incident_id="x", model_urn="urn:test:model",
            created_at=datetime.now(timezone.utc), report=report,
        )
        summary = _history_to_summary([incident])
        assert summary[0]["action_taken"] is None
        assert summary[0]["verified_success"] is None
