"""
Tests for the LLM Reasoning Layer's two fixes found this round:

1. CRITICAL: pydantic.ValidationError was not in the retry-worthy
   exception tuple. Syntactically valid JSON with a schema violation
   (e.g. a suggested_fix missing its required 'rationale' field)
   propagated as an uncaught 500 error instead of retrying or falling
   back to the deterministic report -- defeating the explicit purpose of
   generate_report's retry loop.

2. The Planner's ONLY low-confidence safety gate is an exact string
   comparison (report.confidence == "low"). Nothing previously validated
   or normalized the LLM's confidence value, so "Low", "very low", or any
   other off-vocabulary value would silently bypass that gate entirely.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from groq import GroqError

from app.llm.reasoning import LLMReasoningLayer, _enforce_grounding
from app.models.schemas import (
    CausalCandidate,
    DriftMethod,
    DriftSeverity,
    FeatureDriftResult,
    PredictionDriftResult,
    RootCauseTrace,
)


def _make_trace(with_candidate: bool = True) -> RootCauseTrace:
    pred = PredictionDriftResult(
        model_urn="urn:test:model", method=DriftMethod.KS_TEST, statistic=0.5,
        p_value=0.01, severity=DriftSeverity.HIGH, detected_at=datetime.now(timezone.utc),
    )
    candidates = []
    if with_candidate:
        drift_result = FeatureDriftResult(
            node_urn="urn:cause", feature_name="f", method=DriftMethod.KS_TEST,
            statistic=0.4, p_value=0.01, severity=DriftSeverity.HIGH,
            baseline_window="training", current_window="last_7d",
            sample_size_baseline=300, sample_size_current=300,
        )
        candidates = [CausalCandidate(
            node_urn="urn:cause", node_name="cause_node", hops_from_model=1,
            drift_result=drift_result, is_genuine_cause=True,
            fdr_adjusted_p_value=0.01, survived_fdr_correction=True,
            intervention_delta=0.3,
        )]
    return RootCauseTrace(
        model_urn="urn:test:model", prediction_drift=pred,
        candidates_examined=candidates, isolated_root_causes=candidates,
        graph_path=["urn:cause"] if with_candidate else [],
    )


def _make_layer_with_response(response_content: str) -> LLMReasoningLayer:
    mock_choice = MagicMock()
    mock_choice.message.content = response_content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    layer = LLMReasoningLayer.__new__(LLMReasoningLayer)
    layer.model = "test-model"
    layer._client = MagicMock()
    layer._client.chat.completions.create.return_value = mock_response
    return layer


class TestValidationErrorNoLongerCrashes:
    """THE critical bug: a schema-invalid (but JSON-valid) LLM response
    must never propagate as an uncaught exception."""

    def test_missing_required_field_in_suggested_fix_falls_back_gracefully(self):
        trace = _make_trace()
        response = json.dumps({
            "summary": "s", "detailed_explanation": "d", "root_causes": ["cause_node"],
            "confidence": "high",
            "suggested_fixes": [{"action": "retrain", "target_urn": "urn:cause"}],
        })
        layer = _make_layer_with_response(response)

        report = layer.generate_report(trace)

        assert report is not None
        assert report.confidence == "low"
        assert "unavailable" in report.detailed_explanation.lower()

    def test_falls_back_to_deterministic_report_content(self):
        trace = _make_trace()
        response = json.dumps({
            "summary": "s", "detailed_explanation": "d", "root_causes": [],
            "confidence": "high", "suggested_fixes": [{"action": "x"}],
        })
        layer = _make_layer_with_response(response)
        report = layer.generate_report(trace)
        assert report.root_causes == ["cause_node"]

    def test_retries_before_falling_back(self):
        trace = _make_trace()
        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            mock_choice = MagicMock()
            mock_choice.message.content = json.dumps({
                "summary": "s", "detailed_explanation": "d", "root_causes": ["cause_node"],
                "confidence": "high",
                "suggested_fixes": [{"action": "x", "target_urn": "urn:cause"}],
            })
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            return mock_response

        layer = LLMReasoningLayer.__new__(LLMReasoningLayer)
        layer.model = "test-model"
        layer._client = MagicMock()
        layer._client.chat.completions.create.side_effect = side_effect

        layer.generate_report(trace, _retries=1)
        assert call_count["n"] == 2

    def test_valid_response_with_complete_fix_succeeds_without_fallback(self):
        trace = _make_trace()
        response = json.dumps({
            "summary": "s", "detailed_explanation": "d", "root_causes": ["cause_node"],
            "confidence": "high",
            "suggested_fixes": [{"action": "retrain", "target_urn": "urn:cause", "rationale": "because"}],
        })
        layer = _make_layer_with_response(response)
        report = layer.generate_report(trace)
        assert report.summary == "s"
        assert len(report.suggested_fixes) == 1

    def test_groq_error_still_handled_separately(self):
        trace = _make_trace()
        layer = LLMReasoningLayer.__new__(LLMReasoningLayer)
        layer.model = "test-model"
        layer._client = MagicMock()
        layer._client.chat.completions.create.side_effect = GroqError("API down")

        report = layer.generate_report(trace)
        assert report is not None


class TestConfidenceNormalization:
    """THE second bug: the Planner's low-confidence gate is an exact
    string comparison against "low" -- anything off-vocabulary must be
    normalized or fail closed, never silently bypass the check."""

    def test_wrong_case_is_normalized_to_lowercase(self):
        trace = _make_trace(with_candidate=False)
        parsed = {"root_causes": [], "suggested_fixes": [], "confidence": "Low"}
        result = _enforce_grounding(parsed, trace)
        assert result["confidence"] == "low"

    def test_uppercase_is_normalized(self):
        trace = _make_trace(with_candidate=False)
        parsed = {"root_causes": [], "suggested_fixes": [], "confidence": "HIGH"}
        result = _enforce_grounding(parsed, trace)
        assert result["confidence"] == "high"

    def test_whitespace_is_stripped(self):
        trace = _make_trace(with_candidate=False)
        parsed = {"root_causes": [], "suggested_fixes": [], "confidence": "  moderate  "}
        result = _enforce_grounding(parsed, trace)
        assert result["confidence"] == "moderate"

    def test_unrecognized_value_fails_closed_to_low(self):
        trace = _make_trace(with_candidate=False)
        parsed = {"root_causes": [], "suggested_fixes": [], "confidence": "very confident"}
        result = _enforce_grounding(parsed, trace)
        assert result["confidence"] == "low"

    def test_missing_confidence_key_fails_closed_to_low(self):
        trace = _make_trace(with_candidate=False)
        parsed = {"root_causes": [], "suggested_fixes": []}
        result = _enforce_grounding(parsed, trace)
        assert result["confidence"] == "low"

    def test_valid_values_all_pass_through_unchanged(self):
        trace = _make_trace(with_candidate=False)
        for value in ("low", "moderate", "high"):
            parsed = {"root_causes": [], "suggested_fixes": [], "confidence": value}
            result = _enforce_grounding(parsed, trace)
            assert result["confidence"] == value

    def test_normalized_confidence_actually_triggers_planner_gate(self):
        trace = _make_trace()
        response = json.dumps({
            "summary": "s", "detailed_explanation": "d", "root_causes": ["cause_node"],
            "confidence": "Low",
            "suggested_fixes": [],
        })
        layer = _make_layer_with_response(response)
        report = layer.generate_report(trace)
        assert report.confidence == "low"


class TestRootCausesDeduplication:
    """Minor but real gap: an LLM repeating a name twice in its own JSON
    output flows through undeduplicated, and the frontend uses each name
    as a React list key -- a duplicate produces a real console warning
    and undefined reconciliation behavior for information the repetition
    doesn't actually add."""

    def test_duplicate_root_causes_are_deduplicated(self):
        trace = _make_trace()
        parsed = {
            "root_causes": ["cause_node", "cause_node", "cause_node"],
            "suggested_fixes": [], "confidence": "high",
        }
        result = _enforce_grounding(parsed, trace)
        assert result["root_causes"] == ["cause_node"]

    def test_dedup_preserves_first_occurrence_order(self):
        trace = _make_trace(with_candidate=True)
        # Add a second candidate so there are two distinct valid names to
        # order-check against.
        from app.models.schemas import CausalCandidate, FeatureDriftResult, DriftMethod, DriftSeverity
        drift_result = FeatureDriftResult(
            node_urn="urn:cause2", feature_name="f2", method=DriftMethod.KS_TEST,
            statistic=0.3, p_value=0.02, severity=DriftSeverity.MODERATE,
            baseline_window="training", current_window="last_7d",
            sample_size_baseline=300, sample_size_current=300,
        )
        trace.isolated_root_causes.append(CausalCandidate(
            node_urn="urn:cause2", node_name="cause_2", hops_from_model=1,
            drift_result=drift_result, is_genuine_cause=True,
            fdr_adjusted_p_value=0.02, survived_fdr_correction=True,
            intervention_delta=0.2,
        ))
        parsed = {
            "root_causes": ["cause_2", "cause_node", "cause_2"],
            "suggested_fixes": [], "confidence": "high",
        }
        result = _enforce_grounding(parsed, trace)
        assert result["root_causes"] == ["cause_2", "cause_node"]
