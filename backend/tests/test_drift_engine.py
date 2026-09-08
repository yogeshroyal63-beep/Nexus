"""
Tests for the Drift Detection Engine's sample-size reliability guard.

THE CRITICAL BUG THESE TESTS PROVE IS FIXED: before this round, an empty
`current` array (a realistic production scenario — an upstream pipeline
outage producing zero rows for a time window) caused scipy's ks_2samp to
return statistic=nan, p_value=nan. Because NaN comparisons are always
False in IEEE-754, the severity classification's cascading if/elif chain
fell through every bucket and hit the CRITICAL catch-all — silently
turning "we have no data" into "this is the worst possible drift finding",
which would have driven the Strands Planner into a real remediation
action in response to what was actually just missing data.

PSI had the same misclassification bug plus a second, worse failure: an
empty baseline array crashed outright with an unhandled IndexError.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.drift.engine import (
    _insufficient_sample_size,
    embedding_centroid_drift,
    is_drift_alerting,
    ks_test_drift,
    prediction_output_drift,
    psi_drift,
)
from app.models.schemas import DriftSeverity


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestInsufficientSampleSizeHelper:
    def test_below_floor_on_either_side_is_insufficient(self):
        floor = settings.MIN_SAMPLE_SIZE_FOR_DRIFT_TEST
        assert _insufficient_sample_size(floor - 1, floor + 100) is True
        assert _insufficient_sample_size(floor + 100, floor - 1) is True

    def test_exactly_at_floor_is_sufficient(self):
        floor = settings.MIN_SAMPLE_SIZE_FOR_DRIFT_TEST
        assert _insufficient_sample_size(floor, floor) is False

    def test_both_above_floor_is_sufficient(self):
        floor = settings.MIN_SAMPLE_SIZE_FOR_DRIFT_TEST
        assert _insufficient_sample_size(floor + 50, floor + 50) is False

    def test_zero_is_insufficient(self):
        assert _insufficient_sample_size(0, 300) is True
        assert _insufficient_sample_size(300, 0) is True


class TestKSTestDriftDegenerateInput:
    def test_empty_current_no_longer_misclassified_as_critical(self, rng):
        """THE core bug: this used to return DriftSeverity.CRITICAL via
        NaN-comparison fallthrough. Must now return INSUFFICIENT_DATA."""
        baseline = rng.normal(0, 1, 300)
        result = ks_test_drift(baseline, np.array([]), "urn:x", "f", "training", "now")
        assert result.severity == DriftSeverity.INSUFFICIENT_DATA
        assert result.p_value is None
        assert not np.isnan(result.statistic)  # never leak a raw NaN into the schema

    def test_empty_baseline_no_longer_misclassified(self, rng):
        current = rng.normal(0, 1, 300)
        result = ks_test_drift(np.array([]), current, "urn:x", "f", "training", "now")
        assert result.severity == DriftSeverity.INSUFFICIENT_DATA

    def test_both_empty_no_crash(self):
        result = ks_test_drift(np.array([]), np.array([]), "urn:x", "f", "training", "now")
        assert result.severity == DriftSeverity.INSUFFICIENT_DATA

    def test_single_sample_below_floor(self, rng):
        baseline = rng.normal(0, 1, 300)
        result = ks_test_drift(baseline, np.array([5.0]), "urn:x", "f", "training", "now")
        assert result.severity == DriftSeverity.INSUFFICIENT_DATA

    def test_insufficient_data_never_triggers_alerting(self, rng):
        baseline = rng.normal(0, 1, 300)
        result = ks_test_drift(baseline, np.array([]), "urn:x", "f", "training", "now")
        assert is_drift_alerting(result) is False

    def test_sample_sizes_recorded_even_when_insufficient(self, rng):
        """The result must still honestly report how many samples were
        actually available, for diagnosability."""
        baseline = rng.normal(0, 1, 300)
        result = ks_test_drift(baseline, np.array([1.0, 2.0]), "urn:x", "f", "training", "now")
        assert result.sample_size_baseline == 300
        assert result.sample_size_current == 2

    def test_normal_case_unaffected_still_detects_real_drift(self, rng):
        """The fix must not make the engine less sensitive to genuine
        drift with adequate sample sizes."""
        baseline = rng.normal(0, 1, 300)
        current = rng.normal(3, 1, 300)  # large, obvious shift
        result = ks_test_drift(baseline, current, "urn:x", "f", "training", "now")
        assert result.severity == DriftSeverity.CRITICAL
        assert is_drift_alerting(result) is True

    def test_normal_case_no_drift_still_returns_none(self, rng):
        baseline = rng.normal(0, 1, 300)
        current = rng.normal(0, 1, 300)
        result = ks_test_drift(baseline, current, "urn:x", "f", "training", "now")
        assert result.severity in (DriftSeverity.NONE, DriftSeverity.LOW)
        assert not np.isnan(result.statistic)


class TestPSIDriftDegenerateInput:
    def test_empty_current_no_longer_extreme_psi(self, rng):
        """THE second bug: this used to compute PSI≈11.5 (far outside any
        real drift range) and bucket it as CRITICAL."""
        baseline = rng.normal(0, 1, 300)
        result = psi_drift(baseline, np.array([]), "urn:x", "f", "training", "now")
        assert result.severity == DriftSeverity.INSUFFICIENT_DATA
        assert result.psi_score is None

    def test_empty_baseline_no_longer_crashes(self, rng):
        """THE crash bug: baseline.min()/.max() on an empty array raised
        an unhandled IndexError before this fix."""
        current = rng.normal(0, 1, 300)
        result = psi_drift(np.array([]), current, "urn:x", "f", "training", "now")
        assert result.severity == DriftSeverity.INSUFFICIENT_DATA

    def test_normal_case_unaffected(self, rng):
        baseline = rng.normal(0, 1, 300)
        current = rng.normal(2, 1, 300)
        result = psi_drift(baseline, current, "urn:x", "f", "training", "now")
        assert result.severity != DriftSeverity.INSUFFICIENT_DATA
        assert result.psi_score is not None
        assert result.psi_score > 0


class TestEmbeddingCentroidDriftDegenerateInput:
    def test_empty_current_embeddings_no_crash(self, rng):
        baseline = rng.normal(0, 1, (300, 8))
        current = np.empty((0, 8))
        result = embedding_centroid_drift(baseline, current, "urn:x", "f", "training", "now")
        assert result.severity == DriftSeverity.INSUFFICIENT_DATA

    def test_normal_case_unaffected(self, rng):
        baseline = rng.normal(0, 1, (300, 8))
        current = rng.normal(5, 1, (300, 8))
        result = embedding_centroid_drift(baseline, current, "urn:x", "f", "training", "now")
        assert result.severity != DriftSeverity.INSUFFICIENT_DATA


class TestPredictionOutputDriftDegenerateInput:
    def test_empty_current_predictions_no_longer_critical(self, rng):
        baseline = rng.normal(0, 1, 300)
        result = prediction_output_drift(baseline, np.array([]), "urn:model")
        assert result.severity == DriftSeverity.INSUFFICIENT_DATA
        assert is_drift_alerting(result) is False

    def test_normal_case_unaffected(self, rng):
        baseline = rng.normal(0, 1, 300)
        current = rng.normal(3, 1, 300)
        result = prediction_output_drift(baseline, current, "urn:model")
        assert result.severity == DriftSeverity.CRITICAL
        assert is_drift_alerting(result) is True


class TestIsDriftAlertingExcludesBothNonAlertingStates:
    def test_none_does_not_alert(self):
        from datetime import datetime, timezone
        from app.models.schemas import DriftMethod, PredictionDriftResult
        result = PredictionDriftResult(
            model_urn="urn:m", method=DriftMethod.KS_TEST, statistic=0.01,
            p_value=0.9, severity=DriftSeverity.NONE, detected_at=datetime.now(timezone.utc),
        )
        assert is_drift_alerting(result) is False

    def test_insufficient_data_does_not_alert(self):
        from datetime import datetime, timezone
        from app.models.schemas import DriftMethod, PredictionDriftResult
        result = PredictionDriftResult(
            model_urn="urn:m", method=DriftMethod.KS_TEST, statistic=0.0,
            p_value=None, severity=DriftSeverity.INSUFFICIENT_DATA,
            detected_at=datetime.now(timezone.utc),
        )
        assert is_drift_alerting(result) is False

    def test_low_does_alert(self):
        from datetime import datetime, timezone
        from app.models.schemas import DriftMethod, PredictionDriftResult
        result = PredictionDriftResult(
            model_urn="urn:m", method=DriftMethod.KS_TEST, statistic=0.08,
            p_value=0.03, severity=DriftSeverity.LOW, detected_at=datetime.now(timezone.utc),
        )
        assert is_drift_alerting(result) is True
