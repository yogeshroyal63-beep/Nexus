"""
Materializes a LineageGraph into a traversable networkx DiGraph and
provides the upstream-walk utilities the causal engine depends on.
"""
from __future__ import annotations

import networkx as nx

from app.models.schemas import LineageGraph, LineageNode


def build_dag(graph: LineageGraph) -> nx.DiGraph:
    """
    Build a directed graph with edges pointing upstream -> downstream
    (i.e. data flows in edge direction, same as LineageEdge semantics).
    """
    dag = nx.DiGraph()
    for node in graph.nodes:
        dag.add_node(node.urn, **node.model_dump())
    for edge in graph.edges:
        dag.add_edge(edge.upstream_urn, edge.downstream_urn, relationship=edge.relationship)

    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Lineage graph contains a cycle; cannot treat as a DAG for causal traversal.")
    return dag


def upstream_nodes(dag: nx.DiGraph, model_urn: str) -> list[str]:
    """All ancestors (any number of hops) of the given model node, i.e. every
    dataset/feature that could plausibly have caused drift in this model."""
    if model_urn not in dag:
        return []
    return list(nx.ancestors(dag, model_urn))


def hops_from(dag: nx.DiGraph, source_urn: str, target_urn: str) -> int:
    try:
        return nx.shortest_path_length(dag, source=source_urn, target=target_urn)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return -1


def path_to_model(dag: nx.DiGraph, source_urn: str, model_urn: str) -> list[str]:
    try:
        return nx.shortest_path(dag, source=source_urn, target=model_urn)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [source_urn, model_urn]


def get_node(dag: nx.DiGraph, urn: str) -> LineageNode:
    data = dict(dag.nodes[urn])
    return LineageNode(**data)
