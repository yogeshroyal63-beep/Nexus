"""
Tests for schemas.py's closed-type tightening.

THE GAP THIS CLOSES: RootCauseReport.confidence was typed as an
unconstrained str, even though it's the exact field the Planner's ONLY
low-confidence safety gate checks against (report.confidence == "low").
Every other enum-like field in this schema (RiskLevel, RemediationActionType,
DriftSeverity, DriftMethod, NodeType) is a proper closed type; this was
relying entirely on llm/reasoning.py's caller-level normalization rather
than a type-system guarantee. Tightened to Literal["low","moderate","high"]
so an invalid value is impossible to construct at all, not just
discouraged by convention.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    DriftMethod,
    DriftSeverity,
    PredictionDriftResult,
    RootCauseReport,
    RootCauseTrace,
)


def _make_trace() -> RootCauseTrace:
    pred = PredictionDriftResult(
        model_urn="urn:t", method=DriftMethod.KS_TEST, statistic=0.5,
        p_value=0.01, severity=DriftSeverity.HIGH, detected_at=datetime.now(timezone.utc),
    )
    return RootCauseTrace(
        model_urn="urn:t", prediction_drift=pred,
        candidates_examined=[], isolated_root_causes=[], graph_path=[],
    )


def _base_report_kwargs(confidence):
    return dict(
        model_urn="urn:t", generated_at=datetime.now(timezone.utc),
        summary="s", detailed_explanation="d", root_causes=[],
        confidence=confidence, suggested_fixes=[], raw_trace=_make_trace(),
    )


class TestConfidenceIsClosedType:
    def test_low_is_accepted(self):
        report = RootCauseReport(**_base_report_kwargs("low"))
        assert report.confidence == "low"

    def test_moderate_is_accepted(self):
        report = RootCauseReport(**_base_report_kwargs("moderate"))
        assert report.confidence == "moderate"

    def test_high_is_accepted(self):
        report = RootCauseReport(**_base_report_kwargs("high"))
        assert report.confidence == "high"

    def test_arbitrary_string_is_rejected_at_construction(self):
        """THE core fix: an invalid value must be impossible to construct,
        not just discouraged by a docstring convention."""
        with pytest.raises(ValidationError):
            RootCauseReport(**_base_report_kwargs("very confident"))

    def test_wrong_case_is_rejected_at_construction(self):
        """The schema itself does not normalize case -- that's
        llm/reasoning.py's job (already fixed in an earlier round). The
        schema's job is to reject anything that isn't already exactly
        right, which is what makes the caller-level normalization
        actually load-bearing instead of optional."""
        with pytest.raises(ValidationError):
            RootCauseReport(**_base_report_kwargs("Low"))

    def test_empty_string_is_rejected(self):
        with pytest.raises(ValidationError):
            RootCauseReport(**_base_report_kwargs(""))

    def test_none_is_rejected(self):
        with pytest.raises(ValidationError):
            RootCauseReport(**_base_report_kwargs(None))
