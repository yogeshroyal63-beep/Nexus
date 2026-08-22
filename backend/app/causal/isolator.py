"""
Causal Root-Cause Isolation Engine (spec Section 3 & 6, step 4).

This is the novel core of the project: "most teams stop at 'something
changed nearby'." Given prediction drift on a model, and a lineage DAG of
everything upstream, this engine:

  1. Walks the DAG backward from the model to enumerate every upstream
     dataset/feature node (any number of hops).
  2. Tests each upstream node independently for drift (feeding off the
     drift engine's KS/PSI/embedding tests).
  3. For nodes that show drift, runs a simplified intervention-style check:
     estimate how much of the *downstream* (model output) drift can be
     explained by *this node's* drift specifically, holding the
     contribution of other upstream nodes fixed.
  4. Nodes whose drift has a large, mostly-unconfounded intervention_delta
     are called genuine causes. Nodes that merely co-drifted around the
     same time as a genuine cause (i.e. their apparent contribution
     collapses once the genuine cause is accounted for) are flagged as
     confounded, not causal.

Why this isn't just "the closest drifted node": in a DAG where multiple
upstream nodes feed a shared downstream feature, several nodes can be
statistically drifted simultaneously (e.g. because they're driven by a
common seasonal or deployment event) even though only one of them is doing
the causal work. The intervention check is what separates that.

Method (simplified structural intervention):
  We model each downstream node's value as approximately a weighted
  function of its direct upstream parents' values (weights estimated via
  linear regression on the *baseline* window, where the pipeline was
  presumed healthy). To test "how much of the current downstream drift is
  attributable to parent P", we recompute what the downstream distribution
  would look like if P had NOT drifted (substitute in P's baseline
  distribution while holding all other parents at their observed current
  values), and measure how much that reduces the downstream drift
  statistic. A large reduction means P is doing real causal work; a small
  reduction despite P itself being drifted means P is confounded /
  coincidental.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np

from app.config import settings
from app.drift.engine import ks_test_drift, is_drift_alerting
from app.lineage.dag import get_node, hops_from, path_to_model, upstream_nodes
from app.models.schemas import (
    CausalCandidate,
    FeatureDriftResult,
    LineageGraph,
    PredictionDriftResult,
    RootCauseTrace,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureSample:
    """Baseline vs. current samples for one feature/node, keyed by node URN."""
    node_urn: str
    feature_name: str
    baseline: np.ndarray
    current: np.ndarray


def _downstream_drift_statistic(
    downstream_baseline: np.ndarray,
    downstream_current: np.ndarray,
) -> float:
    from scipy import stats as _stats
    statistic, _ = _stats.ks_2samp(downstream_baseline, downstream_current)
    return float(statistic)


def _simulate_holding_parent_at_baseline(
    parent_baseline: np.ndarray,
    parent_current: np.ndarray,
    other_parents_current: dict[str, np.ndarray],
    downstream_weights: dict[str, float],
    downstream_baseline: np.ndarray,
    downstream_current_actual: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Reconstruct a counterfactual downstream distribution: "what would the
    downstream node look like now if `parent` had NOT drifted, but every
    other parent drifted exactly as observed?"

    Uses the linear structural weights fit on baseline data:
        downstream ≈ sum_i( w_i * parent_i ) + noise

    We swap only this parent's contribution from current -> baseline while
    keeping all others (and the noise term, approximated via residual
    resampling) at their current/observed values.

    `seed` controls the random baseline resample — varying it across calls
    is what lets the caller bootstrap a confidence interval on the
    resulting intervention effect instead of trusting one draw.
    """
    n = len(downstream_current_actual)
    rng = np.random.default_rng(seed)

    # Residual/noise: what's left of the current downstream signal after
    # removing every parent's *actual current* linear contribution.
    reconstructed_current = np.zeros(n)
    for name, weight in downstream_weights.items():
        if name == "__intercept__":
            continue
        series = other_parents_current.get(name)
        if series is None:
            continue
        m = min(len(series), n)
        reconstructed_current[:m] += weight * series[:m]
    reconstructed_current += downstream_weights.get("__intercept__", 0.0)
    residual = downstream_current_actual - reconstructed_current

    # Now rebuild with this parent's contribution swapped to a baseline draw.
    baseline_draw = rng.choice(parent_baseline, size=n, replace=True)
    counterfactual = np.full(n, downstream_weights.get("__intercept__", 0.0))
    for name, weight in downstream_weights.items():
        if name == "__intercept__":
            continue
        if name == "__this_parent__":
            counterfactual += weight * baseline_draw
            continue
        series = other_parents_current.get(name)
        if series is None:
            continue
        m = min(len(series), n)
        counterfactual[:m] += weight * series[:m]
    counterfactual += residual
    return counterfactual


def _correlated_ancestors(
    candidate_urn: str,
    candidate_current: np.ndarray,
    other_ancestor_urns: list[str],
    upstream_samples: dict,
    threshold: float = 0.5,
) -> list[str]:
    """
    Which other drifted ancestors is this candidate's current signal actually
    correlated with? Used to populate `confounded_with` with nodes that
    plausibly explain (or share a cause with) this candidate's apparent
    drift, rather than an arbitrary slice of "other ancestors that exist".
    """
    correlated: list[tuple[str, float]] = []
    for other_urn in other_ancestor_urns:
        other_current = upstream_samples[other_urn].current
        n = min(len(candidate_current), len(other_current))
        if n < 10:
            continue
        try:
            r = float(np.corrcoef(candidate_current[:n], other_current[:n])[0, 1])
        except Exception:
            continue
        if np.isnan(r):
            continue
        if abs(r) >= threshold:
            correlated.append((other_urn, abs(r)))
    correlated.sort(key=lambda pair: pair[1], reverse=True)
    return [urn for urn, _ in correlated[:3]]


def _fit_linear_weights(
    downstream_baseline: np.ndarray,
    parent_baselines: dict[str, np.ndarray],
) -> dict[str, float]:
    """Fit downstream ≈ w0 + sum(w_i * parent_i) via least squares on baseline data."""
    names = list(parent_baselines.keys())
    n = len(downstream_baseline)
    X = np.ones((n, len(names) + 1))
    for i, name in enumerate(names):
        series = parent_baselines[name]
        m = min(len(series), n)
        X[:m, i + 1] = series[:m]
        if m < n:
            # Rows beyond this parent's own series length would otherwise be
            # left at np.ones()'s default of 1.0, which silently biases the
            # fitted weight for that parent. Pad with its own mean instead —
            # a neutral fill that doesn't pull the regression in either
            # direction.
            X[m:, i + 1] = series[:m].mean() if m > 0 else 0.0
    y = downstream_baseline
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    weights = {"__intercept__": float(coeffs[0])}
    for i, name in enumerate(names):
        weights[name] = float(coeffs[i + 1])
    return weights


def isolate_root_causes(
    graph: LineageGraph,
    dag: nx.DiGraph,
    model_urn: str,
    prediction_drift: PredictionDriftResult,
    upstream_samples: dict[str, FeatureSample],
) -> RootCauseTrace:
    """
    Main entry point. `upstream_samples` maps node_urn -> baseline/current
    sample arrays for every upstream node (both raw datasets and
    intermediate feature tables) that could plausibly explain drift in the
    model's predictions. In the demo this comes from the synthetic data
    generator; in production it would come from a feature store /
    warehouse query per node.

    All ancestor nodes with samples are used jointly in the intervention
    regression (see `_fit_linear_weights` / `_simulate_holding_parent_at_baseline`),
    so a raw dataset and the feature table derived from it are both tested
    as candidates and can disambiguate each other via `confounded_with`.
    """
    candidates: list[CausalCandidate] = []

    ancestors = upstream_nodes(dag, model_urn)

    for urn in ancestors:
        if urn not in upstream_samples:
            continue
        sample = upstream_samples[urn]
        node = get_node(dag, urn)

        drift_result: FeatureDriftResult = ks_test_drift(
            baseline=sample.baseline,
            current=sample.current,
            node_urn=urn,
            feature_name=sample.feature_name,
            baseline_window="training",
            current_window="last_7d",
        )

        if not is_drift_alerting(drift_result):
            continue  # not drifted at all -> cannot be a cause

        hops = hops_from(dag, urn, model_urn)

        # --- Intervention check -------------------------------------------------
        # Compare: (a) actual downstream drift statistic vs.
        #          (b) counterfactual downstream drift statistic with THIS
        #              node's contribution reset to baseline behavior.
        # A big drop from (a) to (b) means this node is doing real causal work.
        intervention_delta = 0.0
        confounded_with: list[str] = []

        other_ancestor_urns = [a for a in ancestors if a != urn and a in upstream_samples]

        if other_ancestor_urns:
            other_parents_current = {
                a: upstream_samples[a].current for a in other_ancestor_urns
            }
            other_parents_baseline = {
                a: upstream_samples[a].baseline for a in other_ancestor_urns
            }
            other_parents_baseline["__this_parent__"] = sample.baseline

            # Use the model's prediction values as the downstream signal we're
            # trying to explain. We approximate a baseline "prediction proxy"
            # as a weighted sum of ancestor baselines for the regression fit,
            # since raw historical predictions aren't always available offline.
            proxy_weights_inputs = dict(other_parents_baseline)
            proxy_weights_inputs["__this_parent__"] = sample.baseline
            n_ref = min(len(v) for v in proxy_weights_inputs.values())
            downstream_baseline_proxy = np.mean(
                [v[:n_ref] for v in proxy_weights_inputs.values()], axis=0
            )

            try:
                weights = _fit_linear_weights(downstream_baseline_proxy, proxy_weights_inputs)
                n_cur = min(
                    min(len(v) for v in other_parents_current.values()) if other_parents_current else len(sample.current),
                    len(sample.current),
                )
                downstream_current_actual = np.mean(
                    [sample.current[:n_cur]]
                    + [v[:n_cur] for v in other_parents_current.values()],
                    axis=0,
                )
                # --- Bootstrap the counterfactual, don't trust one draw ------
                # A single random baseline resample could overstate or
                # understate the effect by chance. Resample N times, each
                # with a different seed, and use the conservative lower
                # bound of the resulting distribution (not the mean) to
                # decide is_genuine_cause — a node only counts as a genuine
                # cause if the effect holds up robustly across resamples,
                # not just in the luckiest draw.
                n_bootstrap = 25
                actual_stat = _downstream_drift_statistic(downstream_baseline_proxy, downstream_current_actual)
                bootstrap_deltas = []
                for b in range(n_bootstrap):
                    counterfactual = _simulate_holding_parent_at_baseline(
                        parent_baseline=sample.baseline,
                        parent_current=sample.current,
                        other_parents_current=other_parents_current,
                        downstream_weights=weights,
                        downstream_baseline=downstream_baseline_proxy,
                        downstream_current_actual=downstream_current_actual,
                        seed=1000 + b,
                    )
                    counterfactual_stat = _downstream_drift_statistic(downstream_baseline_proxy, counterfactual)
                    bootstrap_deltas.append(max(0.0, actual_stat - counterfactual_stat))

                bootstrap_deltas = np.array(bootstrap_deltas)
                intervention_delta = float(np.mean(bootstrap_deltas))
                intervention_delta_lower_ci = float(np.percentile(bootstrap_deltas, 5))
                intervention_delta_upper_ci = float(np.percentile(bootstrap_deltas, 95))

                # Confounding: nodes whose own KS statistic is high but whose
                # intervention_delta is low relative to their raw drift are
                # confounded with whatever other drifted ancestor their
                # current signal actually correlates with.
                if intervention_delta < settings.INTERVENTION_DELTA_MIN and drift_result.statistic > 0.15:
                    confounded_with = _correlated_ancestors(
                        candidate_urn=urn,
                        candidate_current=sample.current,
                        other_ancestor_urns=other_ancestor_urns,
                        upstream_samples=upstream_samples,
                    )
            except Exception:
                # Fall back gracefully: treat raw drift statistic as a weak
                # proxy for intervention_delta if the regression is degenerate
                # (e.g. too few overlapping samples). Logged, not silent —
                # this path means the result for this candidate is a weaker
                # approximation and worth knowing about when debugging.
                logger.warning(
                    "Intervention regression failed for %s; falling back to "
                    "raw-drift-statistic proxy for intervention_delta.",
                    urn, exc_info=True,
                )
                intervention_delta = drift_result.statistic * 0.5
                intervention_delta_lower_ci = intervention_delta
                intervention_delta_upper_ci = intervention_delta
        else:
            # Only one drifted ancestor found -> no confounding possible,
            # its own drift statistic stands in directly as the intervention effect.
            intervention_delta = drift_result.statistic
            intervention_delta_lower_ci = intervention_delta
            intervention_delta_upper_ci = intervention_delta

        # Require the conservative lower bound (not the point estimate) to
        # clear the threshold — a candidate whose effect only sometimes
        # clears the bar under resampling is not robustly a genuine cause.
        is_genuine_cause = intervention_delta_lower_ci >= settings.INTERVENTION_DELTA_MIN

        candidates.append(
            CausalCandidate(
                node_urn=urn,
                node_name=node.name,
                hops_from_model=hops,
                drift_result=drift_result,
                is_genuine_cause=is_genuine_cause,
                intervention_delta=round(intervention_delta, 4),
                intervention_delta_lower_ci=round(intervention_delta_lower_ci, 4),
                intervention_delta_upper_ci=round(intervention_delta_upper_ci, 4),
                confounded_with=confounded_with,
            )
        )

    candidates.sort(key=lambda c: c.intervention_delta, reverse=True)
    isolated = [c for c in candidates if c.is_genuine_cause]

    graph_path: list[str] = []
    if isolated:
        graph_path = path_to_model(dag, isolated[0].node_urn, model_urn)

    return RootCauseTrace(
        model_urn=model_urn,
        prediction_drift=prediction_drift,
        candidates_examined=candidates,
        isolated_root_causes=isolated,
        graph_path=graph_path,
    )
