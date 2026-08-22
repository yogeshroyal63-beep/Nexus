"""
Nexus backend test suite.
Tests: memory store, executor, coordinator pipeline, API routes.
Run: pytest tests/ -v
"""
import asyncio
import json
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.agents.executor import get_executor, get_verifier
from app.agents.memory import LocalJSONMemoryStore
from app.main import app
from app.models.schemas import (
    ActionOutcome,
    DriftMethod,
    DriftSeverity,
    IncidentRecord,
    PredictionDriftResult,
    RemediationActionType,
    RemediationPlan,
    RiskLevel,
    RootCauseReport,
    RootCauseTrace,
)

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_trace(model_urn: str = "urn:test:model") -> RootCauseTrace:
    pred = PredictionDriftResult(
        model_urn=model_urn,
        method=DriftMethod.KS_TEST,
        statistic=0.45,
        severity=DriftSeverity.HIGH,
        detected_at=datetime.now(timezone.utc),
    )
    return RootCauseTrace(
        model_urn=model_urn,
        prediction_drift=pred,
        candidates_examined=[],
        isolated_root_causes=[],
        graph_path=[],
    )


def _make_report(model_urn: str = "urn:test:model") -> RootCauseReport:
    return RootCauseReport(
        model_urn=model_urn,
        generated_at=datetime.now(timezone.utc),
        summary="Test summary",
        detailed_explanation="Test explanation",
        root_causes=[],
        confidence="low",
        suggested_fixes=[],
        raw_trace=_make_trace(model_urn),
    )


def _make_plan(
    model_urn: str = "urn:test:model",
    action: RemediationActionType = RemediationActionType.NO_ACTION,
    confidence: float = 0.9,
    risk: RiskLevel = RiskLevel.LOW,
    requires_approval: bool = False,
) -> RemediationPlan:
    return RemediationPlan(
        model_urn=model_urn,
        action_type=action,
        target_urn=model_urn,
        rationale="Test rationale",
        confidence=confidence,
        risk_level=risk,
        requires_human_approval=requires_approval,
    )


def _make_incident(
    iid: str | None = None,
    model_urn: str = "urn:test:model",
    with_plan: bool = False,
    with_outcome: bool = False,
) -> IncidentRecord:
    iid = iid or f"test-{uuid.uuid4().hex[:8]}"
    plan = _make_plan(model_urn) if with_plan else None
    outcome = None
    if with_outcome and plan:
        outcome = ActionOutcome(
            plan=plan,
            executed=True,
            execution_detail="Simulated execution",
            reversible=True,
            rollback_reference="ref-123",
        )
    return IncidentRecord(
        incident_id=iid,
        model_urn=model_urn,
        created_at=datetime.now(timezone.utc),
        report=_make_report(model_urn),
        plan=plan,
        outcome=outcome,
    )


# ── Memory store tests ────────────────────────────────────────────────────────

class TestLocalJSONMemoryStore:
    def setup_method(self):
        import tempfile, os
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.store = LocalJSONMemoryStore(path=self.tmp.name)

    def teardown_method(self):
        import os
        try:
            os.unlink(self.tmp.name)
        except FileNotFoundError:
            pass

    def test_save_and_retrieve(self):
        record = _make_incident()
        asyncio.get_event_loop().run_until_complete(self.store.save_incident(record))
        fetched = asyncio.get_event_loop().run_until_complete(
            self.store.get_incident(record.incident_id)
        )
        assert fetched is not None
        assert fetched.incident_id == record.incident_id

    def test_get_nonexistent_returns_none(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.store.get_incident("does-not-exist")
        )
        assert result is None

    def test_find_similar_filters_by_model(self):
        r1 = _make_incident(model_urn="urn:model:a")
        r2 = _make_incident(model_urn="urn:model:b")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.store.save_incident(r1))
        loop.run_until_complete(self.store.save_incident(r2))
        results = loop.run_until_complete(self.store.find_similar("urn:model:a"))
        assert all(r.model_urn == "urn:model:a" for r in results)
        assert len(results) == 1

    def test_all_incidents_ordering(self):
        loop = asyncio.get_event_loop()
        for _ in range(3):
            loop.run_until_complete(self.store.save_incident(_make_incident()))
        all_inc = loop.run_until_complete(self.store.all_incidents())
        assert len(all_inc) == 3

    def test_upsert_replaces_existing(self):
        record = _make_incident()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.store.save_incident(record))
        record.report = _make_report()
        loop.run_until_complete(self.store.save_incident(record))
        all_inc = loop.run_until_complete(self.store.all_incidents())
        same_id = [r for r in all_inc if r.incident_id == record.incident_id]
        assert len(same_id) == 1

    def test_claim_for_approval_works_once(self):
        record = _make_incident(with_plan=True)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.store.save_incident(record))
        claimed = loop.run_until_complete(
            self.store.claim_incident_for_approval(record.incident_id)
        )
        assert claimed is not None
        # second claim should fail
        claimed2 = loop.run_until_complete(
            self.store.claim_incident_for_approval(record.incident_id)
        )
        assert claimed2 is None

    def test_claim_returns_none_without_plan(self):
        record = _make_incident(with_plan=False)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.store.save_incident(record))
        result = loop.run_until_complete(
            self.store.claim_incident_for_approval(record.incident_id)
        )
        assert result is None

    def test_claim_returns_none_already_has_outcome(self):
        record = _make_incident(with_plan=True, with_outcome=True)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.store.save_incident(record))
        result = loop.run_until_complete(
            self.store.claim_incident_for_approval(record.incident_id)
        )
        assert result is None


# ── Executor tests ─────────────────────────────────────────────────────────────

class TestExecutor:
    def test_no_action_executes_cleanly(self):
        plan = _make_plan(action=RemediationActionType.NO_ACTION)
        executor = get_executor()
        outcome = asyncio.get_event_loop().run_until_complete(executor.execute(plan))
        assert outcome.executed is True
        assert "No action" in outcome.execution_detail

    def test_open_ticket_without_token_returns_outcome(self):
        plan = _make_plan(action=RemediationActionType.OPEN_INCIDENT_TICKET)
        executor = get_executor()
        outcome = asyncio.get_event_loop().run_until_complete(executor.execute(plan))
        # Without GitHub creds executor degrades gracefully — outcome always returned
        assert outcome is not None
        assert isinstance(outcome.execution_detail, str)
        assert len(outcome.execution_detail) > 0

    def test_trigger_retrain_simulates(self):
        plan = _make_plan(action=RemediationActionType.TRIGGER_RETRAIN)
        executor = get_executor()
        outcome = asyncio.get_event_loop().run_until_complete(executor.execute(plan))
        assert outcome.executed is True

    def test_rollback_simulates(self):
        plan = _make_plan(action=RemediationActionType.ROLLBACK_MODEL_VERSION)
        executor = get_executor()
        outcome = asyncio.get_event_loop().run_until_complete(executor.execute(plan))
        assert outcome.executed is True
        assert outcome.reversible is True

    def test_quarantine_simulates(self):
        plan = _make_plan(action=RemediationActionType.QUARANTINE_DATA_SOURCE)
        executor = get_executor()
        outcome = asyncio.get_event_loop().run_until_complete(executor.execute(plan))
        assert outcome.executed is True


# ── Verifier tests ─────────────────────────────────────────────────────────────

class TestVerifier:
    def test_verifies_as_resolved_when_no_drift(self):
        # Must use a non-NO_ACTION plan so verifier actually runs
        plan = _make_plan(action=RemediationActionType.TRIGGER_RETRAIN)
        outcome = ActionOutcome(
            plan=plan,
            executed=True,
            execution_detail="Done",
            reversible=False,
        )
        verifier = get_verifier()

        async def no_drift(_): return False

        result = asyncio.get_event_loop().run_until_complete(
            verifier.verify(outcome, no_drift)
        )
        assert result.verified is True

    def test_verifies_as_unresolved_when_drift_remains(self):
        plan = _make_plan(action=RemediationActionType.TRIGGER_RETRAIN)
        outcome = ActionOutcome(
            plan=plan,
            executed=True,
            execution_detail="Done",
            reversible=False,
        )
        verifier = get_verifier()

        async def still_drifting(_): return True

        result = asyncio.get_event_loop().run_until_complete(
            verifier.verify(outcome, still_drifting)
        )
        assert result.verified is False

    def test_verifier_skips_no_action(self):
        plan = _make_plan(action=RemediationActionType.NO_ACTION)
        outcome = ActionOutcome(
            plan=plan,
            executed=True,
            execution_detail="No action",
            reversible=False,
        )
        verifier = get_verifier()

        async def no_drift(_): return False

        result = asyncio.get_event_loop().run_until_complete(
            verifier.verify(outcome, no_drift)
        )
        # NO_ACTION skips verification — verified stays None
        assert result.verified is None


# ── API route tests ────────────────────────────────────────────────────────────

class TestAPIRoutes:
    def test_health_ok(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_lineage_returns_graph(self):
        r = client.get("/api/lineage/urn:li:mlModel:(demo,fraud_model_v3,PROD)")
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert len(data["nodes"]) > 0

    def test_incidents_list_returns_array(self):
        r = client.get("/api/incidents")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_nonexistent_incident_404(self):
        r = client.get("/api/incidents/does-not-exist")
        assert r.status_code == 404

    def test_approve_nonexistent_incident_404(self):
        r = client.post("/api/incidents/does-not-exist/approve")
        assert r.status_code == 404

    def test_rate_limit_on_run_endpoint(self):
        # 4th call within window should get 429
        responses = []
        for _ in range(4):
            r = client.post(
                "/api/run",
                params={"model_urn": "urn:li:mlModel:(demo,fraud_model_v3,PROD)", "inject_drift": "true"},
            )
            responses.append(r.status_code)
        assert 429 in responses

    def test_sns_endpoint_handles_empty_body(self):
        r = client.post(
            "/api/sns/drift-check",
            json={"Message": "{}"},
        )
        # Should process or return a meaningful response (not 500)
        assert r.status_code in (200, 422)

    def test_investigate_returns_trace(self):
        r = client.post(
            "/api/investigate",
            params={"model_urn": "urn:li:mlModel:(demo,fraud_model_v3,PROD)", "inject_drift": "true"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "trace" in data
        assert "graph" in data
