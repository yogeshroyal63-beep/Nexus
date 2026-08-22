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
    return result.severity not in (DriftSeverity.NONE,)
