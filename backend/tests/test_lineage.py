"""
Tests for the lineage DAG utilities and DataHub response parser.

THE TWO GAPS THESE TESTS PROVE ARE CLOSED:
1. A single malformed entity/relationship (missing its identifying URN
   field) in an otherwise-valid DataHub response crashed the ENTIRE
   lineage parse with an uncaught KeyError. Confirmed exploitable, and
   directly relevant given this module's own docstring documents that
   exact field names shift across mcp-server-datahub versions.
2. A response in a genuinely unrecognized shape (e.g. the plain-text
   fallback path when a tool's output isn't valid JSON) silently produced
   an EMPTY LineageGraph instead of an error -- masking a lineage
   ingestion FAILURE as "this model has no upstream dependencies", a
   fundamentally different and much worse conclusion.
"""
from __future__ import annotations

import pytest

from app.lineage.dag import build_dag, get_node, hops_from, path_to_model, upstream_nodes
from app.lineage.datahub_client import (
    MockDataHubClient,
    _parse_datahub_lineage_response,
)
from app.models.schemas import LineageEdge, LineageGraph, LineageNode, NodeType


def _make_graph():
    nodes = [
        LineageNode(urn="urn:a", name="a", node_type=NodeType.DATASET, platform="p"),
        LineageNode(urn="urn:b", name="b", node_type=NodeType.FEATURE, platform="p"),
        LineageNode(urn="urn:model", name="model", node_type=NodeType.MODEL, platform="p"),
    ]
    edges = [
        LineageEdge(upstream_urn="urn:a", downstream_urn="urn:b"),
        LineageEdge(upstream_urn="urn:b", downstream_urn="urn:model"),
    ]
    return LineageGraph(nodes=nodes, edges=edges, root_model_urn="urn:model")


class TestBuildDag:
    def test_builds_dag_from_valid_graph(self):
        dag = build_dag(_make_graph())
        assert dag.number_of_nodes() == 3
        assert dag.number_of_edges() == 2

    def test_raises_on_cyclic_graph(self):
        nodes = [
            LineageNode(urn="urn:x", name="x", node_type=NodeType.DATASET, platform="p"),
            LineageNode(urn="urn:y", name="y", node_type=NodeType.DATASET, platform="p"),
        ]
        edges = [
            LineageEdge(upstream_urn="urn:x", downstream_urn="urn:y"),
            LineageEdge(upstream_urn="urn:y", downstream_urn="urn:x"),  # cycle
        ]
        cyclic_graph = LineageGraph(nodes=nodes, edges=edges, root_model_urn="urn:x")
        with pytest.raises(ValueError, match="cycle"):
            build_dag(cyclic_graph)


class TestUpstreamNodes:
    def test_returns_all_ancestors(self):
        dag = build_dag(_make_graph())
        ancestors = upstream_nodes(dag, "urn:model")
        assert set(ancestors) == {"urn:a", "urn:b"}

    def test_returns_empty_for_unknown_model(self):
        dag = build_dag(_make_graph())
        assert upstream_nodes(dag, "urn:does-not-exist") == []

    def test_returns_empty_for_leaf_node_with_no_ancestors(self):
        dag = build_dag(_make_graph())
        assert upstream_nodes(dag, "urn:a") == []


class TestHopsFrom:
    def test_direct_edge_is_one_hop(self):
        dag = build_dag(_make_graph())
        assert hops_from(dag, "urn:b", "urn:model") == 1

    def test_two_hop_ancestor(self):
        dag = build_dag(_make_graph())
        assert hops_from(dag, "urn:a", "urn:model") == 2

    def test_no_path_returns_negative_one(self):
        dag = build_dag(_make_graph())
        assert hops_from(dag, "urn:model", "urn:a") == -1  # wrong direction

    def test_unknown_node_returns_negative_one(self):
        dag = build_dag(_make_graph())
        assert hops_from(dag, "urn:nonexistent", "urn:model") == -1


class TestPathToModel:
    def test_returns_full_path(self):
        dag = build_dag(_make_graph())
        path = path_to_model(dag, "urn:a", "urn:model")
        assert path == ["urn:a", "urn:b", "urn:model"]

    def test_falls_back_to_direct_pair_on_no_path(self):
        dag = build_dag(_make_graph())
        path = path_to_model(dag, "urn:model", "urn:a")
        assert path == ["urn:model", "urn:a"]


class TestGetNode:
    def test_reconstructs_lineage_node(self):
        dag = build_dag(_make_graph())
        node = get_node(dag, "urn:a")
        assert node.urn == "urn:a"
        assert node.name == "a"
        assert node.node_type == NodeType.DATASET


class TestParseDataHubLineageResponseMalformedEntities:
    """THE crash bug: one malformed item must not take down the whole parse."""

    def test_entity_missing_urn_is_skipped_not_crashed(self):
        raw = {
            "entities": [
                {"name": "no-urn-here"},  # malformed
                {"urn": "urn:good", "name": "good", "entityType": "DATASET"},
            ],
            "relationships": [],
        }
        result = _parse_datahub_lineage_response(raw, "urn:model")
        assert len(result.nodes) == 1
        assert result.nodes[0].urn == "urn:good"

    def test_relationship_missing_upstream_urn_is_skipped(self):
        raw = {
            "entities": [{"urn": "urn:a", "entityType": "DATASET"}],
            "relationships": [
                {"downstreamUrn": "urn:model"},  # missing upstreamUrn
                {"upstreamUrn": "urn:a", "downstreamUrn": "urn:model"},
            ],
        }
        result = _parse_datahub_lineage_response(raw, "urn:model")
        assert len(result.edges) == 1
        assert result.edges[0].upstream_urn == "urn:a"

    def test_all_entities_malformed_returns_empty_but_valid_graph(self):
        raw = {"entities": [{"name": "x"}, {"name": "y"}], "relationships": []}
        result = _parse_datahub_lineage_response(raw, "urn:model")
        assert result.nodes == []  # degraded gracefully, not crashed

    def test_results_shape_missing_urn_is_skipped(self):
        raw = {"results": [{"paths": []}, {"urn": "urn:good", "paths": []}]}
        result = _parse_datahub_lineage_response(raw, "urn:model")
        assert len(result.nodes) == 1
        assert result.nodes[0].urn == "urn:good"

    def test_well_formed_response_still_parses_normally(self):
        """Sanity check: the fix didn't break the happy path."""
        raw = {
            "entities": [
                {"urn": "urn:a", "name": "a", "entityType": "DATASET"},
                {"urn": "urn:model", "name": "model", "entityType": "ML_MODEL"},
            ],
            "relationships": [
                {"upstreamUrn": "urn:a", "downstreamUrn": "urn:model", "type": "derives"},
            ],
        }
        result = _parse_datahub_lineage_response(raw, "urn:model")
        assert len(result.nodes) == 2
        assert len(result.edges) == 1


class TestParseDataHubLineageResponseUnrecognizedShape:
    """THE silent-failure bug: an unrecognized shape must raise, never
    silently produce an empty-but-'successful' graph."""

    def test_plain_string_response_raises_not_silently_empty(self):
        with pytest.raises(ValueError, match="Unrecognized"):
            _parse_datahub_lineage_response("some plain text, not JSON", "urn:model")

    def test_dict_without_entities_or_results_raises(self):
        with pytest.raises(ValueError, match="Unrecognized"):
            _parse_datahub_lineage_response({"unexpected_key": []}, "urn:model")

    def test_none_response_raises(self):
        with pytest.raises(ValueError, match="Unrecognized"):
            _parse_datahub_lineage_response(None, "urn:model")

    def test_error_message_includes_model_urn_for_diagnosability(self):
        with pytest.raises(ValueError, match="urn:my-model"):
            _parse_datahub_lineage_response("garbage", "urn:my-model")


class TestMockDataHubClient:
    """Confirm the mock (the one actually exercised in the demo) still
    produces a well-formed graph after all these changes."""

    @pytest.mark.asyncio
    async def test_produces_valid_lineage_graph(self):
        client = MockDataHubClient()
        graph = await client.get_ml_lineage("urn:li:mlModel:(demo,fraud_model_v3,PROD)")
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0
        dag = build_dag(graph)  # must not raise (no cycles, well-formed)
        assert dag.number_of_nodes() == len(graph.nodes)
