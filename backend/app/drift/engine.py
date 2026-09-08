"""
Statistical Drift Detection Engine (spec Section 3 & 6, step 3).

Real applied statistics, not "ask the LLM if this looks different":
  - KS-test for numeric feature distribution drift.
  - Population Stability Index (PSI) for numeric/categorical drift, which
    is more standard in ML monitoring because it gives a single
    interpretable magnitude rather than just a p-value.
  - Embedding centroid cosine-distance drift for unstructured/text features.
  - The same KS-test machinery is reused for prediction-output drift
    (comparing the live prediction distribution to a training-time /
    recent-baseline distribution).

FIXED — degenerate input silently misclassified as CRITICAL (found via
testing with realistic failure inputs, not observed in production, but
directly exploitable): every one of the four functions below assumed
well-formed, reasonably-sized, non-empty input arrays with no validation.
An empty `current` array — a completely realistic production scenario,
e.g. an upstream pipeline outage producing zero rows for a time window —
caused `scipy.stats.ks_2samp` to return `statistic=nan, p_value=nan`.
Because NaN comparisons are always False in IEEE-754 (`nan >= x` and
`nan < x` are both False for any x), `_severity_from_pvalue`'s cascading
if/elif chain fell through EVERY bucket check and hit the final
`return DriftSeverity.CRITICAL` catch-all. The practical consequence: a
data outage — not model drift at all — would be classified as the most
severe possible drift finding and could drive the Strands Planner into a
real remediation action (trigger_retrain, even rollback_model_version)
in response to what was actually just missing data.

PSI had the same class of bug (empty `current` produced PSI≈11.5, far
outside any real drift range, still bucketed as CRITICAL) plus a second,
worse failure: an empty `baseline` array crashed outright with an
unhandled IndexError from `baseline.min()`/`.max()`.

Fixed with an explicit minimum-sample-size guard at the top of every
function (`settings.MIN_SAMPLE_SIZE_FOR_DRIFT_TEST`, default 20 — a
widely-used rule of thumb below which KS/PSI are known to be statistically
unreliable). Below the floor, functions return
`DriftSeverity.INSUFFICIENT_DATA` — a new, explicit third state distinct
from both NONE (measured, genuinely no drift) and CRITICAL (measured,
severe drift) — rather than attempting a computation the underlying
statistics can't support. `is_drift_alerting()` treats INSUFFICIENT_DATA
as non-alerting (an autonomous action must never fire on "we don't know"),
while the result itself remains fully visible in the report/UI so a human
can tell a data-quality problem apart from actual drift.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from scipy import stats

from app.config import settings
from app.models.schemas import (
    DriftMethod,
    DriftSeverity,
    FeatureDriftResult,
    PredictionDriftResult,
)


# ---------------------------------------------------------------------------
# Sample-size reliability guard
# ---------------------------------------------------------------------------

def _insufficient_sample_size(baseline_len: int, current_len: int) -> bool:
    floor = settings.MIN_SAMPLE_SIZE_FOR_DRIFT_TEST
    return baseline_len < floor or current_len < floor


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

def _severity_from_psi(psi: float) -> DriftSeverity:
    if psi < settings.PSI_LOW_THRESHOLD:
        return DriftSeverity.NONE
    if psi < settings.PSI_MODERATE_THRESHOLD:
        return DriftSeverity.LOW
    if psi < settings.PSI_HIGH_THRESHOLD:
        return DriftSeverity.MODERATE
    return DriftSeverity.HIGH if psi < 0.5 else DriftSeverity.CRITICAL


def _severity_from_pvalue(p_value: float, statistic: float) -> DriftSeverity:
    if p_value >= settings.KS_PVALUE_ALERT_THRESHOLD:
        return DriftSeverity.NONE
    if statistic < 0.1:
        return DriftSeverity.LOW
    if statistic < 0.2:
        return DriftSeverity.MODERATE
    if statistic < 0.35:
        return DriftSeverity.HIGH
    return DriftSeverity.CRITICAL


def _severity_from_cosine(distance: float) -> DriftSeverity:
    t = settings.EMBEDDING_COSINE_DRIFT_THRESHOLD
    if distance < t:
        return DriftSeverity.NONE
    if distance < t * 1.5:
        return DriftSeverity.LOW
    if distance < t * 2.5:
        return DriftSeverity.MODERATE
    if distance < t * 4:
        return DriftSeverity.HIGH
    return DriftSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Core statistical tests
# ---------------------------------------------------------------------------

def ks_test_drift(
    baseline: np.ndarray,
    current: np.ndarray,
    node_urn: str,
    feature_name: str,
    baseline_window: str,
    current_window: str,
) -> FeatureDriftResult:
    """Two-sample Kolmogorov-Smirnov test for numeric feature drift."""
    if _insufficient_sample_size(len(baseline), len(current)):
        return FeatureDriftResult(
            node_urn=node_urn,
            feature_name=feature_name,
            method=DriftMethod.KS_TEST,
            statistic=0.0,
            p_value=None,
            severity=DriftSeverity.INSUFFICIENT_DATA,
            baseline_window=baseline_window,
            current_window=current_window,
            sample_size_baseline=len(baseline),
            sample_size_current=len(current),
        )

    statistic, p_value = stats.ks_2samp(baseline, current)
    return FeatureDriftResult(
        node_urn=node_urn,
        feature_name=feature_name,
        method=DriftMethod.KS_TEST,
        statistic=float(statistic),
        p_value=float(p_value),
        severity=_severity_from_pvalue(p_value, statistic),
        baseline_window=baseline_window,
        current_window=current_window,
        sample_size_baseline=len(baseline),
        sample_size_current=len(current),
    )


def psi_drift(
    baseline: np.ndarray,
    current: np.ndarray,
    node_urn: str,
    feature_name: str,
    baseline_window: str,
    current_window: str,
    n_bins: int = 10,
) -> FeatureDriftResult:
    """
    Population Stability Index.

    PSI = sum( (current_pct - baseline_pct) * ln(current_pct / baseline_pct) )
    over quantile bins fit on the baseline distribution. This is the metric
    most production ML monitoring stacks use to give a single interpretable
    magnitude for how much a distribution has shifted, complementing the
    KS-test's significance test.
    """
    if _insufficient_sample_size(len(baseline), len(current)):
        return FeatureDriftResult(
            node_urn=node_urn,
            feature_name=feature_name,
            method=DriftMethod.PSI,
            statistic=0.0,
            psi_score=None,
            severity=DriftSeverity.INSUFFICIENT_DATA,
            baseline_window=baseline_window,
            current_window=current_window,
            sample_size_baseline=len(baseline),
            sample_size_current=len(current),
        )

    eps = 1e-6
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(baseline, quantiles))
    if len(bin_edges) < 3:
        # Degenerate / near-constant baseline distribution; fall back to
        # min/max range binning so PSI is still computable.
        bin_edges = np.linspace(baseline.min(), baseline.max() + eps, n_bins + 1)

    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    baseline_pct = np.clip(baseline_counts / max(len(baseline), 1), eps, None)
    current_pct = np.clip(current_counts / max(len(current), 1), eps, None)

    psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))

    return FeatureDriftResult(
        node_urn=node_urn,
        feature_name=feature_name,
        method=DriftMethod.PSI,
        statistic=psi,
        psi_score=psi,
        severity=_severity_from_psi(psi),
        baseline_window=baseline_window,
        current_window=current_window,
        sample_size_baseline=len(baseline),
        sample_size_current=len(current),
    )


def embedding_centroid_drift(
    baseline_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
    node_urn: str,
    feature_name: str,
    baseline_window: str,
    current_window: str,
) -> FeatureDriftResult:
    """
    Cosine distance between the centroid (mean vector) of baseline vs.
    current embeddings, for unstructured/text features where KS/PSI don't
    apply directly to raw values.
    """
    if _insufficient_sample_size(len(baseline_embeddings), len(current_embeddings)):
        return FeatureDriftResult(
            node_urn=node_urn,
            feature_name=feature_name,
            method=DriftMethod.EMBEDDING_COSINE,
            statistic=0.0,
            severity=DriftSeverity.INSUFFICIENT_DATA,
            baseline_window=baseline_window,
            current_window=current_window,
            sample_size_baseline=len(baseline_embeddings),
            sample_size_current=len(current_embeddings),
        )

    baseline_centroid = baseline_embeddings.mean(axis=0)
    current_centroid = current_embeddings.mean(axis=0)

    denom = np.linalg.norm(baseline_centroid) * np.linalg.norm(current_centroid)
    cosine_similarity = float(np.dot(baseline_centroid, current_centroid) / denom) if denom > 0 else 1.0
    cosine_distance = 1.0 - cosine_similarity

    return FeatureDriftResult(
        node_urn=node_urn,
        feature_name=feature_name,
        method=DriftMethod.EMBEDDING_COSINE,
        statistic=cosine_distance,
        severity=_severity_from_cosine(cosine_distance),
        baseline_window=baseline_window,
        current_window=current_window,
        sample_size_baseline=len(baseline_embeddings),
        sample_size_current=len(current_embeddings),
    )


def prediction_output_drift(
    baseline_predictions: np.ndarray,
    current_predictions: np.ndarray,
    model_urn: str,
) -> PredictionDriftResult:
    """KS-test on the model's own output distribution over time — this is
    what actually triggers an investigation (spec Section 2, step 1-2)."""
    if _insufficient_sample_size(len(baseline_predictions), len(current_predictions)):
        return PredictionDriftResult(
            model_urn=model_urn,
            method=DriftMethod.KS_TEST,
            statistic=0.0,
            p_value=None,
            severity=DriftSeverity.INSUFFICIENT_DATA,
            detected_at=datetime.now(timezone.utc),
        )

    statistic, p_value = stats.ks_2samp(baseline_predictions, current_predictions)
    return PredictionDriftResult(
        model_urn=model_urn,
        method=DriftMethod.KS_TEST,
        statistic=float(statistic),
        p_value=float(p_value),
        severity=_severity_from_pvalue(p_value, statistic),
        detected_at=datetime.now(timezone.utc),
    )


def is_drift_alerting(result: FeatureDriftResult | PredictionDriftResult) -> bool:
    # INSUFFICIENT_DATA must never be treated as alerting — an autonomous
    # remediation action must never fire on "we don't have enough data to
    # tell", only on a genuinely measured drift finding.
    return result.severity not in (DriftSeverity.NONE, DriftSeverity.INSUFFICIENT_DATA)
