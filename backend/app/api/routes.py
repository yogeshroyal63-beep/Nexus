"""
Nexus API routes.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.agents.coordinator import run_nexus
from app.agents.executor import get_executor
from app.agents.memory import get_memory_store
from app.config import settings
from app.lineage.dag import build_dag
from app.models.schemas import ActionOutcome, LineageGraph
from app.pipeline import run_full_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/lineage/{model_urn:path}")
async def get_lineage(model_urn: str) -> LineageGraph:
    from app.lineage.datahub_client import get_lineage_client
    client = get_lineage_client()
    try:
        graph = await client.get_ml_lineage(model_urn)
        build_dag(graph)
        return graph
    except Exception as exc:
        detail = str(exc) if settings.ENV == "development" else "Lineage fetch failed."
        raise HTTPException(status_code=500, detail=detail) from exc
    finally:
        await client.aclose()


@router.post("/investigate")
async def investigate(
    model_urn: str = "urn:li:mlModel:(demo,fraud_model_v3,PROD)",
    inject_drift: bool = True,
):
    """Report-only pipeline (non-agentic). Kept for backward compat."""
    try:
        result = await run_full_pipeline(model_urn=model_urn, inject_drift=inject_drift)
    except Exception as exc:
        detail = str(exc) if settings.ENV == "development" else "Pipeline failed."
        raise HTTPException(status_code=500, detail=detail) from exc
    return {
        "graph": result["graph"],
        "trace": result["trace"],
        "report": result["report"],
        "writeback": result["writeback"],
    }


@router.post("/run")
async def run_agent(
    model_urn: str = "urn:li:mlModel:(demo,fraud_model_v3,PROD)",
    inject_drift: bool = True,
    write_back: bool = True,
):
    """
    Full Nexus agentic loop: detect -> diagnose -> plan (Strands+Bedrock) ->
    execute (or escalate) -> verify -> remember.
    """
    try:
        result = await run_nexus(model_urn=model_urn, inject_drift=inject_drift, write_back=write_back)
    except Exception as exc:
        detail = str(exc) if settings.ENV == "development" else "Agentic pipeline failed."
        raise HTTPException(status_code=500, detail=detail) from exc
    return result


@router.get("/incidents")
async def list_incidents(limit: int = 50):
    memory = get_memory_store()
    return await memory.all_incidents(limit=limit)


@router.get("/incidents/{incident_id}")
async def get_incident(incident_id: str):
    memory = get_memory_store()
    record = await memory.get_incident(incident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return record


@router.post("/incidents/{incident_id}/approve")
async def approve_incident(incident_id: str):
    """Human-in-the-loop approval for escalated plans."""
    memory = get_memory_store()

    existing = await memory.get_incident(incident_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    if existing.plan is None:
        raise HTTPException(status_code=400, detail="Incident has no plan to approve")

    record = await memory.claim_incident_for_approval(incident_id)
    if record is None:
        raise HTTPException(
            status_code=409,
            detail="Incident already has an executed outcome or approval is in progress.",
        )

    from app.agents.coordinator import _rerun_drift_check
    from app.agents.executor import get_verifier

    try:
        executor = get_executor()
        outcome = await executor.execute(record.plan)
        verifier = get_verifier()
        outcome = await verifier.verify(outcome, _rerun_drift_check)
        record.outcome = outcome
        await memory.save_incident(record)
        return record
    except Exception as exc:
        logger.error("Approval of incident %s failed: %s", incident_id, exc, exc_info=True)
        try:
            record.outcome = ActionOutcome(
                plan=record.plan,
                executed=False,
                execution_detail=f"Approval failed: {exc.__class__.__name__}: {exc}",
                reversible=False,
            )
            await memory.save_incident(record)
        except Exception:
            logger.error("Could not persist failure outcome for incident %s.", incident_id, exc_info=True)
        detail = str(exc) if settings.ENV == "development" else "Approval failed."
        raise HTTPException(status_code=500, detail=detail) from exc


@router.post("/sns/drift-check")
async def sns_drift_check(request: Request):
    """
    AWS SNS push endpoint — triggers background autonomous run.
    Wire an SNS subscription to POST here for scheduled/event-driven runs.
    """
    import base64, json as _json

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {exc}") from exc

    # Handle SNS notification format
    message_str = body.get("Message", "{}")
    try:
        payload = _json.loads(message_str) if message_str.strip() else {}
    except Exception:
        payload = {}

    kwargs = {}
    if "model_urn" in payload:
        kwargs["model_urn"] = payload["model_urn"]
    if "inject_drift" in payload:
        kwargs["inject_drift"] = bool(payload["inject_drift"])
    if "write_back" in payload:
        kwargs["write_back"] = bool(payload["write_back"])

    try:
        result = await run_nexus(**kwargs)
        logger.info("SNS-triggered run complete: incident_id=%s", result.incident_id)
        return {"status": "processed", "incident_id": result.incident_id}
    except Exception as exc:
        logger.error("SNS-triggered run failed: %s", exc, exc_info=True)
        return {"status": "error", "detail": str(exc)}


@router.get("/health")
async def health():
    return {"status": "ok", "service": "nexus"}
