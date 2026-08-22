"""
Lineage Graph Ingestion Layer (spec Section 3 & 6, step 2).

Responsible for pulling ML lineage from DataHub (via the official DataHub
MCP Server, acryldata/mcp-server-datahub) and materializing it into our
internal LineageGraph model.

Two implementations are provided behind the same interface:
  - DataHubMCPClient: talks to the real DataHub MCP Server, either as a
    local stdio subprocess (self-hosted DataHub — the standard way MCP
    clients like Claude Desktop/Cursor connect to it) or over SSE (DataHub
    Cloud's managed MCP server, if DATAHUB_MCP_URL is set).
  - MockDataHubClient: synthesizes a realistic demo lineage graph so the
    full pipeline can run end-to-end without a live DataHub instance
    (used for `USE_MOCK_DATAHUB=true`, e.g. local dev / hackathon demo).

Swapping between them is a one-line change in get_lineage_client() below —
nothing downstream (drift engine, causal engine, LLM layer) needs to know
which one is active, since both return the same LineageGraph shape.

IMPORTANT — real tool names, verified against DataHub's published docs
(https://docs.datahub.com/docs/features/feature-guides/mcp):
  Read-only:  search, get_entities, get_lineage, get_dataset_queries,
              list_schema_fields, get_lineage_paths_between
  Mutation (opt-in via TOOLS_IS_MUTATION_ENABLED=true on the server):
              add_tags, remove_tags, update_description, add_owners,
              set_domains

There is NO incident-creation tool in the official server, so our
write-back path uses `add_tags` + `update_description` to annotate the
model entity directly, rather than the (non-existent) "create_incident"
this file originally assumed. Exact per-tool parameter names have shifted
across mcp-server-datahub releases (e.g. the `filter` param was renamed
from a dict to a string between versions) — before your demo, run
`session.list_tools()` once against your installed version and confirm
the argument names below still match; adjust `_call_tool` arguments if not.
"""
from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any

import abc

from app.config import settings
from app.models.schemas import LineageEdge, LineageGraph, LineageNode, NodeType


class BaseLineageClient(abc.ABC):
    @abc.abstractmethod
    async def get_ml_lineage(self, model_urn: str) -> LineageGraph:
        """Return the full multi-hop upstream lineage DAG for a given model URN."""
        raise NotImplementedError

    @abc.abstractmethod
    async def write_incident(self, model_urn: str, incident_payload: dict[str, Any]) -> str:
        """Annotate the model entity with the diagnosis. Returns an identifier for the write."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release any held resources (subprocess, connection). No-op by default."""
        return None


class DataHubMCPClient(BaseLineageClient):
    """
    Talks to the real DataHub MCP Server using the official `mcp` Python SDK.

    Self-hosted DataHub: connects over stdio, spawning `mcp-server-datahub`
    as a subprocess (DATAHUB_MCP_COMMAND/DATAHUB_MCP_ARGS), exactly like
    Claude Desktop or Cursor would, with DATAHUB_GMS_URL/DATAHUB_GMS_TOKEN
    passed through as env vars to that subprocess.

    DataHub Cloud (managed MCP server): connects over SSE instead, if
    DATAHUB_MCP_URL is set.
    """

    def __init__(self):
        self._session = None
        self._exit_stack: AsyncExitStack | None = None

    async def _ensure_session(self):
        if self._session is not None:
            return self._session

        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        self._exit_stack = AsyncExitStack()

        if settings.DATAHUB_MCP_URL:
            from mcp.client.sse import sse_client
            read, write = await self._exit_stack.enter_async_context(
                sse_client(settings.DATAHUB_MCP_URL)
            )
        else:
            server_params = StdioServerParameters(
                command=settings.DATAHUB_MCP_COMMAND,
                args=settings.DATAHUB_MCP_ARGS.split(),
                env={
                    "DATAHUB_GMS_URL": settings.DATAHUB_GMS_URL,
                    "DATAHUB_GMS_TOKEN": settings.DATAHUB_GMS_TOKEN,
                    "TOOLS_IS_MUTATION_ENABLED": str(settings.DATAHUB_MUTATION_ENABLED).lower(),
                },
            )
            read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))

        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._session = session
        return session

    async def aclose(self):
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        session = await self._ensure_session()
        result = await session.call_tool(tool_name, arguments)
        if result.isError:
            raise RuntimeError(f"MCP tool '{tool_name}' failed: {result.content}")
        # Tool results come back as a list of content blocks; text blocks carry JSON.
        text_parts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
        raw_text = "\n".join(text_parts)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text  # some tools return plain text rather than JSON

    async def get_ml_lineage(self, model_urn: str) -> LineageGraph:
        raw = await self._call_tool(
            "get_lineage",
            {"urn": model_urn, "direction": "upstream"},
        )
        return _parse_datahub_lineage_response(raw, model_urn)

    async def write_incident(self, model_urn: str, incident_payload: dict[str, Any]) -> str:
        """
        No native incident tool exists, so we annotate the entity directly:
        tag it as drift-detected and append the diagnosis to its description.
        Requires DATAHUB_MUTATION_ENABLED=true (mirrors the server's
        TOOLS_IS_MUTATION_ENABLED flag) — otherwise this raises clearly
        instead of silently no-op-ing.
        """
        if not settings.DATAHUB_MUTATION_ENABLED:
            raise RuntimeError(
                "DataHub write-back requires DATAHUB_MUTATION_ENABLED=true "
                "(and the server's TOOLS_IS_MUTATION_ENABLED=true) — mutation "
                "tools are opt-in on both sides."
            )
        await self._call_tool("add_tags", {"urn": model_urn, "tags": ["drift-detected"]})
        summary = incident_payload.get("summary", "Model drift detected.")
        note = f"\n\n[causal-drift-sentinel] {summary}"
        await self._call_tool("update_description", {"urn": model_urn, "description_append": note})
        return f"annotated:{model_urn}"


def _parse_datahub_lineage_response(raw: Any, root_model_urn: str) -> LineageGraph:
    """
    Translate DataHub's get_lineage tool response into our internal
    LineageGraph. The exact response shape can vary slightly by
    mcp-server-datahub version — this handles the common
    {"entities": [...], "relationships": [...]} shape and falls back to a
    flatter {"results": [{"urn", "hops", "paths"}, ...]} shape some versions
    return. Verify against your installed version's actual output (call
    `list_tools()` / run the tool once) and adjust here if it differs.
    """
    nodes: list[LineageNode] = []
    edges: list[LineageEdge] = []
    type_map = {
        "DATASET": NodeType.DATASET,
        "ML_FEATURE_TABLE": NodeType.FEATURE,
        "ML_MODEL": NodeType.MODEL,
        "ML_MODEL_DEPLOYMENT": NodeType.DEPLOYMENT,
    }

    if isinstance(raw, dict) and "entities" in raw:
        for entity in raw.get("entities", []):
            nodes.append(
                LineageNode(
                    urn=entity["urn"],
                    name=entity.get("name", entity["urn"]),
                    node_type=type_map.get(entity.get("entityType", "DATASET"), NodeType.DATASET),
                    platform=entity.get("platform"),
                    description=entity.get("description"),
                    schema_fields=entity.get("schemaFieldNames", []),
                    tags=entity.get("tags", []),
                )
            )
        for rel in raw.get("relationships", []):
            edges.append(
                LineageEdge(
                    upstream_urn=rel["upstreamUrn"],
                    downstream_urn=rel["downstreamUrn"],
                    relationship=rel.get("type", "derives_from"),
                )
            )
    elif isinstance(raw, dict) and "results" in raw:
        for item in raw.get("results", []):
            urn = item["urn"]
            nodes.append(LineageNode(urn=urn, name=urn, node_type=NodeType.DATASET))
            for path in item.get("paths", []):
                for a, b in zip(path, path[1:]):
                    edges.append(LineageEdge(upstream_urn=a, downstream_urn=b))

    return LineageGraph(nodes=nodes, edges=edges, root_model_urn=root_model_urn)


class MockDataHubClient(BaseLineageClient):
    """
    Synthesizes a realistic fraud-detection-style ML lineage graph for demo
    purposes (spec Section 4, "Data for the Demo"):

        raw_transactions, raw_user_profiles, raw_device_signals
              -> feature_txn_velocity, feature_user_risk_score, feature_device_trust
                    -> fraud_model_v3
                          -> fraud_model_v3_prod_deployment

    This lets the full pipeline (ingestion -> drift -> causal -> LLM ->
    write-back -> frontend) run end-to-end without a live DataHub instance,
    while keeping the exact same LineageGraph contract a real DataHub client
    would produce.
    """

    async def get_ml_lineage(self, model_urn: str) -> LineageGraph:
        nodes = [
            LineageNode(
                urn="urn:li:dataset:(demo,raw_transactions,PROD)",
                name="raw_transactions",
                node_type=NodeType.DATASET,
                platform="snowflake",
                description="Raw transaction events ingested from the payments service.",
                schema_fields=["txn_id", "user_id", "amount", "currency", "merchant_category", "ts"],
            ),
            LineageNode(
                urn="urn:li:dataset:(demo,raw_user_profiles,PROD)",
                name="raw_user_profiles",
                node_type=NodeType.DATASET,
                platform="postgres",
                description="User account and KYC profile data.",
                schema_fields=["user_id", "account_age_days", "kyc_tier", "country"],
            ),
            LineageNode(
                urn="urn:li:dataset:(demo,raw_device_signals,PROD)",
                name="raw_device_signals",
                node_type=NodeType.DATASET,
                platform="kafka",
                description="Device fingerprint and network signals per session.",
                schema_fields=["session_id", "user_id", "device_id", "ip_risk_score"],
            ),
            LineageNode(
                urn="urn:li:mlFeatureTable:(demo,feature_txn_velocity)",
                name="feature_txn_velocity",
                node_type=NodeType.FEATURE,
                platform="feast",
                description="Rolling 1h/24h transaction count and sum per user.",
                schema_fields=["txn_count_1h", "txn_sum_24h"],
            ),
            LineageNode(
                urn="urn:li:mlFeatureTable:(demo,feature_user_risk_score)",
                name="feature_user_risk_score",
                node_type=NodeType.FEATURE,
                platform="feast",
                description="Composite user risk score derived from profile + history.",
                schema_fields=["user_risk_score"],
            ),
            LineageNode(
                urn="urn:li:mlFeatureTable:(demo,feature_device_trust)",
                name="feature_device_trust",
                node_type=NodeType.FEATURE,
                platform="feast",
                description="Device trust score derived from device signals.",
                schema_fields=["device_trust_score"],
            ),
            LineageNode(
                urn="urn:li:mlModel:(demo,fraud_model_v3,PROD)",
                name="fraud_model_v3",
                node_type=NodeType.MODEL,
                platform="mlflow",
                description="Gradient-boosted fraud classifier, trained 2026-06-01.",
            ),
            LineageNode(
                urn="urn:li:mlModelDeployment:(demo,fraud_model_v3_prod)",
                name="fraud_model_v3_prod_deployment",
                node_type=NodeType.DEPLOYMENT,
                platform="sagemaker",
                description="Live production endpoint for fraud_model_v3.",
            ),
        ]
        edges = [
            LineageEdge(upstream_urn="urn:li:dataset:(demo,raw_transactions,PROD)",
                        downstream_urn="urn:li:mlFeatureTable:(demo,feature_txn_velocity)"),
            LineageEdge(upstream_urn="urn:li:dataset:(demo,raw_user_profiles,PROD)",
                        downstream_urn="urn:li:mlFeatureTable:(demo,feature_user_risk_score)"),
            LineageEdge(upstream_urn="urn:li:dataset:(demo,raw_transactions,PROD)",
                        downstream_urn="urn:li:mlFeatureTable:(demo,feature_user_risk_score)"),
            LineageEdge(upstream_urn="urn:li:dataset:(demo,raw_device_signals,PROD)",
                        downstream_urn="urn:li:mlFeatureTable:(demo,feature_device_trust)"),
            LineageEdge(upstream_urn="urn:li:mlFeatureTable:(demo,feature_txn_velocity)",
                        downstream_urn="urn:li:mlModel:(demo,fraud_model_v3,PROD)"),
            LineageEdge(upstream_urn="urn:li:mlFeatureTable:(demo,feature_user_risk_score)",
                        downstream_urn="urn:li:mlModel:(demo,fraud_model_v3,PROD)"),
            LineageEdge(upstream_urn="urn:li:mlFeatureTable:(demo,feature_device_trust)",
                        downstream_urn="urn:li:mlModel:(demo,fraud_model_v3,PROD)"),
            LineageEdge(upstream_urn="urn:li:mlModel:(demo,fraud_model_v3,PROD)",
                        downstream_urn="urn:li:mlModelDeployment:(demo,fraud_model_v3_prod)"),
        ]
        return LineageGraph(
            nodes=nodes,
            edges=edges,
            root_model_urn=model_urn or "urn:li:mlModelDeployment:(demo,fraud_model_v3_prod)",
        )

    async def write_incident(self, model_urn: str, incident_payload: dict[str, Any]) -> str:
        # In demo mode we just fabricate a plausible incident URN; no network call.
        return f"urn:li:incident:(demo,{abs(hash(json.dumps(incident_payload, default=str))) % 10**8})"


def get_lineage_client() -> BaseLineageClient:
    if settings.USE_MOCK_DATAHUB:
        return MockDataHubClient()
    return DataHubMCPClient()
