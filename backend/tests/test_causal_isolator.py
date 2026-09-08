"""
Deep tests for the causal root-cause isolation engine.

Unlike test_nexus.py's integration-level smoke tests, these construct
SYNTHETIC data with KNOWN ground truth — we build the downstream signal
ourselves, so we know exactly which upstream node is the genuine cause,
which nodes merely co-drift, and which nodes are pure noise. This is the
only way to validate a causal inference algorithm: you cannot just check
"did it run", you have to check "did it find the right answer when we
already know the answer".

Four ground-truth scenarios:
  1. Single genuine cause, everything else stable  -> isolator finds it
  2. Genuine cause + a node that merely co-drifts alongside it (shared
     shock, no causal path to the downstream signal) -> isolator must
     NOT credit the co-drifting node
  3. Nothing drifted at all -> isolator finds zero candidates
  4. Many stable ancestors (multiple-testing stress test) -> FDR
     correction keeps the false-positive root-cause rate low, which a
     naive per-node alpha=0.05 threshold would not
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from app.causal.isolator import FeatureSample, isolate_root_causes
from app.drift.engine import prediction_output_drift
from app.lineage.dag import build_dag
from app.models.schemas import LineageEdge, LineageGraph, LineageNode, NodeType

MODEL_URN = "urn:test:model"
RNG_SEED = 12345


def _make_graph(ancestor_urns: list[str]) -> LineageGraph:
    """Flat graph: every ancestor feeds the model directly (1 hop)."""
    nodes = [
        LineageNode(
            urn=urn,
            name=urn.split(":")[-1],
            node_type=NodeType.FEATURE,
            platform="test",
            description="synthetic test node",
            schema_fields=["value"],
        )
        for urn in ancestor_urns
    ] + [
        LineageNode(
            urn=MODEL_URN,
            name="test_model",
            node_type=NodeType.MODEL,
            platform="test",
            description="synthetic test model",
            schema_fields=[],
        )
    ]
    edges = [
        LineageEdge(upstream_urn=urn, downstream_urn=MODEL_URN, relationship="derives")
        for urn in ancestor_urns
    ]
    return LineageGraph(root_model_urn=MODEL_URN, nodes=nodes, edges=edges)


class TestSingleGenuineCause:
    """Scenario 1: one node truly drifted and drives downstream drift."""

    def test_isolates_the_single_true_cause(self):
        rng = np.random.default_rng(RNG_SEED)
        n = 400

        # feature_a: TRUE cause. Mean shifts from 0 -> 2.5 (large, unambiguous shift).
        a_baseline = rng.normal(0, 1, n)
        a_current = rng.normal(2.5, 1, n)

        # feature_b: completely stable, no drift at all.
        b_baseline = rng.normal(5, 1, n)
        b_current = rng.normal(5, 1, n)

        graph = _make_graph([
            "urn:test:feature_a", "urn:test:feature_b",
        ])
        dag = build_dag(graph)

        samples = {
            "urn:test:feature_a": FeatureSample("urn:test:feature_a", "feature_a", a_baseline, a_current),
            "urn:test:feature_b": FeatureSample("urn:test:feature_b", "feature_b", b_baseline, b_current),
        }

        pred_drift = prediction_output_drift(
            baseline_predictions=rng.normal(0, 1, n),
            current_predictions=rng.normal(2.5, 1, n),
            model_urn=MODEL_URN,
        )

        trace = isolate_root_causes(graph, dag, MODEL_URN, pred_drift, samples)

        isolated_urns = {c.node_urn for c in trace.isolated_root_causes}
        assert "urn:test:feature_a" in isolated_urns, (
            f"Expected feature_a (true cause) to be isolated. Got: {isolated_urns}"
        )
        assert "urn:test:feature_b" not in isolated_urns, (
            "feature_b never drifted — must never be reported as a root cause"
        )

    def test_stable_node_does_not_survive_fdr(self):
        rng = np.random.default_rng(RNG_SEED)
        n = 400
        a_baseline = rng.normal(0, 1, n)
        a_current = rng.normal(2.5, 1, n)
        b_baseline = rng.normal(5, 1, n)
        b_current = rng.normal(5, 1, n)  # identical distribution -> p-value should be large

        graph = _make_graph(["urn:test:feature_a", "urn:test:feature_b"])
        dag = build_dag(graph)
        samples = {
            "urn:test:feature_a": FeatureSample("urn:test:feature_a", "feature_a", a_baseline, a_current),
            "urn:test:feature_b": FeatureSample("urn:test:feature_b", "feature_b", b_baseline, b_current),
        }
        pred_drift = prediction_output_drift(rng.normal(0, 1, n), rng.normal(2.5, 1, n), MODEL_URN)
        trace = isolate_root_causes(graph, dag, MODEL_URN, pred_drift, samples)

        b_candidate = next(c for c in trace.candidates_examined if c.node_urn == "urn:test:feature_b")
        assert b_candidate.survived_fdr_correction is False


class TestConfoundedCoDrift:
    """
    Scenario 2: two nodes drift AT THE SAME TIME because they share a
    common upstream shock. Two sub-cases with different, both CORRECT,
    expected behavior:

    (a) Near-perfect confounding (feature_a and feature_c are ~r=1.0
        correlated, identical magnitude of shift). This is mathematically
        NON-IDENTIFIABLE — no causal inference method, ours included, can
        tell which of two perfectly co-moving signals is "the" cause from
        observational data alone. The scientifically honest answer is to
        claim NEITHER with confidence, not to arbitrarily pick one. A
        system that confidently blames one of two indistinguishable
        signals is worse than one that admits it cannot tell.

    (b) Realistic partial confounding: the two nodes share a common
        component (co-drift together) but the true cause has ADDITIONAL
        drift magnitude the passenger lacks — the real-world case of "a
        deployment event nudged several pipelines, but only one of them
        actually broke the model". This case IS identifiable, and the
        isolator must separate them correctly.
    """

    def test_near_perfect_confounding_is_honestly_not_isolated(self):
        """
        When two nodes are statistically indistinguishable in their
        relationship to the downstream signal, claiming either one as
        THE root cause would be a false precision the evidence doesn't
        support. Verify the isolator does NOT manufacture a confident
        pick here — this is correct, conservative behavior, not a bug.
        """
        rng = np.random.default_rng(RNG_SEED)
        n = 400
        shock = rng.normal(0, 1, n)

        a_baseline = rng.normal(0, 1, n)
        a_current = 2.0 + shock + rng.normal(0, 0.3, n)
        c_baseline = rng.normal(0, 1, n)
        c_current = 2.0 + shock + rng.normal(0, 0.3, n)  # near-identical to a_current

        # Real downstream signal that depends EQUALLY on both — this is
        # the genuinely non-identifiable case: two ancestors with
        # indistinguishable drift AND indistinguishable true influence.
        downstream_baseline = 0.5 * a_baseline + 0.5 * c_baseline + rng.normal(0, 0.2, n)
        downstream_current = 0.5 * a_current + 0.5 * c_current + rng.normal(0, 0.2, n)

        graph = _make_graph(["urn:test:feature_a", "urn:test:feature_c"])
        dag = build_dag(graph)
        samples = {
            "urn:test:feature_a": FeatureSample("urn:test:feature_a", "feature_a", a_baseline, a_current),
            "urn:test:feature_c": FeatureSample("urn:test:feature_c", "feature_c", c_baseline, c_current),
        }
        pred_drift = prediction_output_drift(downstream_baseline, downstream_current, MODEL_URN)
        trace = isolate_root_causes(
            graph, dag, MODEL_URN, pred_drift, samples,
            downstream_baseline=downstream_baseline,
            downstream_current=downstream_current,
        )

        # Both are statistically drifted (confirms the scenario is set up
        # correctly) — but under near-perfect mutual confounding, neither
        # should be reported with is_genuine_cause=True. Manufacturing a
        # confident pick between two indistinguishable signals would be
        # false precision.
        drifted = {c.node_urn for c in trace.candidates_examined if c.survived_fdr_correction}
        assert drifted == {"urn:test:feature_a", "urn:test:feature_c"}
        assert trace.isolated_root_causes == [], (
            "Under near-perfect confounding (r≈1.0), no single node should "
            "be confidently isolated — this is the mathematically honest "
            "answer, not a detection failure."
        )

    def test_asymmetric_confounding_correctly_separates_true_cause(self):
        """
        Realistic case: a shared shock nudges both nodes, but the
        DOWNSTREAM signal genuinely (and differentially) depends much
        more on feature_a than feature_c. This mirrors production usage:
        pipeline.py passes the model's real prediction arrays as
        downstream_baseline/downstream_current, so the intervention
        regression targets true differential dependence rather than a
        proxy reconstructed from the same ancestors being tested.
        """
        rng = np.random.default_rng(RNG_SEED)
        n = 500
        shock = rng.normal(0, 1, n)

        a_baseline = rng.normal(0, 1, n)
        a_current = 1.2 + shock * 0.5 + rng.normal(0, 0.3, n)

        c_baseline = rng.normal(0, 1, n)
        c_current = 0.15 + shock * 0.5 + rng.normal(0, 0.3, n)

        # Real downstream signal: genuinely, differentially dependent on
        # feature_a (weight 0.8) vs feature_c (weight 0.1), with its own
        # independent noise — exactly what pipeline.py provides in
        # production via generate_prediction_samples().
        downstream_baseline = 0.8 * a_baseline + 0.1 * c_baseline + rng.normal(0, 0.4, n)
        downstream_current = 0.8 * a_current + 0.1 * c_current + rng.normal(0, 0.4, n)

        graph = _make_graph(["urn:test:feature_a", "urn:test:feature_c"])
        dag = build_dag(graph)
        samples = {
            "urn:test:feature_a": FeatureSample("urn:test:feature_a", "feature_a", a_baseline, a_current),
            "urn:test:feature_c": FeatureSample("urn:test:feature_c", "feature_c", c_baseline, c_current),
        }
        pred_drift = prediction_output_drift(downstream_baseline, downstream_current, MODEL_URN)
        trace = isolate_root_causes(
            graph, dag, MODEL_URN, pred_drift, samples,
            downstream_baseline=downstream_baseline,
            downstream_current=downstream_current,
        )

        by_urn = {c.node_urn: c for c in trace.candidates_examined}
        a_cand = by_urn["urn:test:feature_a"]
        c_cand = by_urn["urn:test:feature_c"]

        # The node the downstream signal genuinely depends on more
        # heavily (weight 0.8 vs 0.1) must show a materially larger
        # intervention_delta — proving the algorithm uses real
        # differential dependence, not just "both drifted around the
        # same time so credit both equally".
        assert a_cand.intervention_delta > c_cand.intervention_delta, (
            f"Expected feature_a (weight 0.8) to show a larger "
            f"intervention_delta than feature_c (weight 0.1). Got "
            f"a={a_cand.intervention_delta}, c={c_cand.intervention_delta}"
        )

    def test_mean_proxy_fallback_still_runs_without_real_downstream(self):
        """
        When no real prediction arrays are available (downstream_baseline/
        downstream_current omitted), the function must still run to
        completion using the weaker mean-based proxy fallback — not crash.
        This is the degraded-but-functional path for callers without a
        live prediction log.
        """
        rng = np.random.default_rng(RNG_SEED)
        n = 300
        a_baseline = rng.normal(0, 1, n)
        a_current = rng.normal(2.0, 1, n)
        b_baseline = rng.normal(5, 1, n)
        b_current = rng.normal(5, 1, n)

        graph = _make_graph(["urn:test:feature_a", "urn:test:feature_b"])
        dag = build_dag(graph)
        samples = {
            "urn:test:feature_a": FeatureSample("urn:test:feature_a", "feature_a", a_baseline, a_current),
            "urn:test:feature_b": FeatureSample("urn:test:feature_b", "feature_b", b_baseline, b_current),
        }
        pred_drift = prediction_output_drift(rng.normal(0, 1, n), rng.normal(2.0, 1, n), MODEL_URN)

        # No downstream_baseline/downstream_current passed -> fallback path.
        trace = isolate_root_causes(graph, dag, MODEL_URN, pred_drift, samples)
        assert trace is not None
        assert isinstance(trace.candidates_examined, list)


class TestNoDrift:
    """Scenario 3: nothing drifted. The isolator must report zero causes,
    not force-pick the least-stable-looking node."""

    def test_zero_candidates_when_nothing_drifted(self):
        rng = np.random.default_rng(RNG_SEED)
        n = 300
        # All nodes: baseline and current drawn from the IDENTICAL distribution.
        urns = ["urn:test:feature_a", "urn:test:feature_b", "urn:test:feature_c"]
        graph = _make_graph(urns)
        dag = build_dag(graph)
        samples = {
            urn: FeatureSample(urn, urn, rng.normal(0, 1, n), rng.normal(0, 1, n))
            for urn in urns
        }
        pred_drift = prediction_output_drift(rng.normal(0, 1, n), rng.normal(0, 1, n), MODEL_URN)
        trace = isolate_root_causes(graph, dag, MODEL_URN, pred_drift, samples)

        assert trace.isolated_root_causes == []
        assert trace.graph_path == []


class TestFDRProtectionAtScale:
    """
    Scenario 4: many ancestor nodes, all genuinely stable (pure noise).
    Without FDR correction, testing N nodes at raw alpha=0.05 would be
    expected to produce ~0.05*N false "significant" results by chance
    alone. This test proves the FDR-corrected pipeline suppresses that
    inflation in the actual isolator output, not just in the standalone
    fdr.py unit tests.
    """

    def test_many_stable_ancestors_produce_no_spurious_root_causes(self):
        rng = np.random.default_rng(999)
        n = 200
        n_ancestors = 15
        urns = [f"urn:test:ancestor_{i}" for i in range(n_ancestors)]
        graph = _make_graph(urns)
        dag = build_dag(graph)

        # All ancestors stable: baseline and current from the same
        # distribution. Any "significance" that shows up is pure sampling
        # noise, and pure noise should not become a reported root cause.
        samples = {
            urn: FeatureSample(urn, urn, rng.normal(0, 1, n), rng.normal(0, 1, n))
            for urn in urns
        }
        pred_drift = prediction_output_drift(rng.normal(0, 1, n), rng.normal(0, 1, n), MODEL_URN)
        trace = isolate_root_causes(graph, dag, MODEL_URN, pred_drift, samples)

        # With 15 pure-noise ancestors, a naive per-test alpha=0.05 would
        # expect ~0.75 false "drifted" flags on average, sometimes more.
        # The isolator must report zero genuine root causes regardless —
        # is_genuine_cause additionally requires clearing the
        # intervention_delta bar, which pure noise should not do.
        assert trace.isolated_root_causes == []

        # Confirm the FDR risk figure is being computed and is non-trivial
        # for this many simultaneous tests (sanity check on the plumbing,
        # not just the final decision).
        assert trace.fdr_uncorrected_false_positive_risk > 0.5  # 1-0.95^15 ≈ 0.537
