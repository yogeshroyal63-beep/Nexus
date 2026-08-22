"""
Coordinator — orchestrates the full Nexus autonomous loop:

    Detect -> Diagnose -> (Memory lookup) -> Plan (Strands+Bedrock) ->
    [Execute -> Verify | Escalate] -> Report -> Remember

This is what makes Nexus a Taskmaster-track agent: it doesn't stop at
producing a report, it decides whether to act, acts within a safety gate,
checks whether the action worked, and remembers the outcome for next time.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.agents.executor import get_executor, get_verifier
from app.agents.memory import get_memory_store
from app.agents.planner import get_planner
from app.drift.engine import is_drift_alerting, prediction_output_drift
from app.models.schemas import IncidentRecord, SentinelRunResult
from app.pipeline import run_full_pipeline
from app.utils.demo_data import generate_demo_samples, generate_prediction_samples

logger = logging.getLogger(__name__)


async def _rerun_drift_check(model_urn: str) -> bool:
    samples = generate_demo_samples(inject_drift=False)
    baseline, current = generate_prediction_samples(samples)
    result = prediction_output_drift(baseline, current, model_urn)
    return is_drift_alerting(result)


async def run_nexus(
    model_urn: str = "urn:li:mlModel:(demo,fraud_model_v3,PROD)",
    inject_drift: bool = True,
    write_back: bool = True,
) -> SentinelRunResult:
    """
    Single entry point for the full Nexus agentic loop.
    Powers /api/run and any scheduled/SNS-triggered background run.
    """
    incident_id = f"nxs-{uuid.uuid4().hex[:12]}"
    memory = get_memory_store()

    # 1-3: Detect, Diagnose, Explain — pipeline
    pipeline_result = await run_full_pipeline(
        model_urn=model_urn, inject_drift=inject_drift, write_back=write_back
    )
    trace = pipeline_result["trace"]
    report = pipeline_result["report"]
    writeback = pipeline_result["writeback"]

    result = SentinelRunResult(
        incident_id=incident_id,
        model_urn=model_urn,
        trace=trace,
        report=report,
        writeback=writeback,
    )

    if report is None:
        await memory.save_incident(
            IncidentRecord(
                incident_id=incident_id,
                model_urn=model_urn,
                created_at=datetime.now(timezone.utc),
                report=_placeholder_report(trace),
            )
        )
        return result

    # Memory lookup — informs Strands Planner
    history = await memory.find_similar(model_urn)
    result.similar_past_incidents = [h.incident_id for h in history]

    # 4: Plan via Strands Agents SDK + Bedrock
    planner = get_planner()
    plan = await planner.plan(report, history)
    result.plan = plan

    # 5: Execute (gated) or escalate
    outcome = None
    if plan.requires_human_approval:
        result.escalated = True
        logger.info(
            "Incident %s escalated (confidence=%.2f, risk=%s): %s",
            incident_id, plan.confidence, plan.risk_level.value, plan.rationale,
        )
    else:
        executor = get_executor()
        outcome = await executor.execute(plan)

        # 6: Verify
        verifier = get_verifier()
        outcome = await verifier.verify(outcome, _rerun_drift_check)
        result.outcome = outcome

    # 7: Remember
    await memory.save_incident(
        IncidentRecord(
            incident_id=incident_id,
            model_urn=model_urn,
            created_at=datetime.now(timezone.utc),
            report=report,
            plan=plan,
            outcome=outcome,
            writeback=writeback,
        )
    )

    return result


def _placeholder_report(trace):
    from app.models.schemas import RootCauseReport
    return RootCauseReport(
        model_urn=trace.model_urn,
        generated_at=datetime.now(timezone.utc),
        summary="No root cause isolated in this run.",
        detailed_explanation="Drift detection and causal isolation ran but found no genuine root cause.",
        root_causes=[],
        confidence="low",
        suggested_fixes=[],
        raw_trace=trace,
    )
