"""
End-to-end pipeline orchestration, matching the architecture diagram in
spec Section 4:

  Lineage Ingestion -> Drift Detection -> Causal Root-Cause Engine
    -> LLM Reasoning Layer -> Write-Back Agent
"""
from __future__ import annotations

from app.causal.isolator import isolate_root_causes
from app.drift.engine import prediction_output_drift
from app.lineage.dag import build_dag
from app.lineage.datahub_client import get_lineage_client
from app.llm.reasoning import get_reasoning_layer
from app.models.schemas import LineageGraph, RootCauseReport, WriteBackResult
from app.utils.demo_data import generate_demo_samples, generate_prediction_samples
from app.writeback.agent import WriteBackAgent


async def run_full_pipeline(
    model_urn: str = "urn:li:mlModel:(demo,fraud_model_v3,PROD)",
    inject_drift: bool = True,
    write_back: bool = True,
) -> dict:
    """
    Runs the complete Causal Drift Sentinel pipeline once, end-to-end.
    Powers both the API's /investigate endpoint and the "Replay a failure"
    demo mode in the frontend.
    """
    lineage_client = get_lineage_client()

    try:
        # 1. Lineage ingestion
        graph: LineageGraph = await lineage_client.get_ml_lineage(model_urn)
        dag = build_dag(graph)

        # 2. Drift detection (demo data stands in for a real feature-store /
        #    warehouse query in this build; the statistical machinery is real)
        feature_samples = generate_demo_samples(inject_drift=inject_drift)
        pred_baseline, pred_current = generate_prediction_samples(feature_samples)
        prediction_drift = prediction_output_drift(pred_baseline, pred_current, model_urn)

        # 3. Causal root-cause isolation
        trace = isolate_root_causes(
            graph=graph,
            dag=dag,
            model_urn=model_urn,
            prediction_drift=prediction_drift,
            upstream_samples=feature_samples,
        )

        result: dict = {"graph": graph, "trace": trace, "report": None, "writeback": None}

        if trace.isolated_root_causes:
            # 4. LLM reasoning & explanation layer
            reasoning = get_reasoning_layer()
            report: RootCauseReport = reasoning.generate_report(trace)
            result["report"] = report

            # 5. Write-back agent
            if write_back:
                agent = WriteBackAgent(lineage_client)
                wb: WriteBackResult = await agent.run(report)
                result["writeback"] = wb

        return result
    finally:
        # Real DataHubMCPClient holds a subprocess/session; must be released
        # after every run or long-lived deployments leak subprocesses.
        # MockDataHubClient's aclose() is a no-op inherited from the base class.
        await lineage_client.aclose()
