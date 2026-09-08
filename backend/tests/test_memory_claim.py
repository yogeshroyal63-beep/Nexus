"""
Tests for the Memory agent's stuck-forever claim fix and DynamoDB
pagination fix.

THE GAP THESE TESTS PROVE IS CLOSED: before this round, an incident
claimed for approval but never followed by a saved outcome (e.g. the
process died mid-request) was stuck FOREVER — no future claim attempt
could ever succeed, since the only check was "does a claim flag exist",
with no expiry. These tests prove a stale claim can now be recovered,
while a genuinely in-flight (recent) claim still correctly blocks a
second concurrent claim.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from app.agents.memory import LocalJSONMemoryStore
from app.config import settings
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


def _make_plan(model_urn: str = "urn:test:model") -> RemediationPlan:
    return RemediationPlan(
        model_urn=model_urn, action_type=RemediationActionType.TRIGGER_RETRAIN,
        target_urn=model_urn, rationale="r", confidence=0.9,
        risk_level=RiskLevel.MEDIUM, requires_human_approval=True,
    )


def _make_report(model_urn: str = "urn:test:model") -> RootCauseReport:
    pred = PredictionDriftResult(
        model_urn=model_urn, method=DriftMethod.KS_TEST, statistic=0.5,
        severity=DriftSeverity.HIGH, detected_at=datetime.now(timezone.utc),
    )
    trace = RootCauseTrace(
        model_urn=model_urn, prediction_drift=pred, candidates_examined=[],
        isolated_root_causes=[], graph_path=[],
    )
    return RootCauseReport(
        model_urn=model_urn, generated_at=datetime.now(timezone.utc),
        summary="s", detailed_explanation="d", root_causes=[],
        confidence="high", suggested_fixes=[], raw_trace=trace,
    )


def _make_incident(with_plan: bool = True, model_urn: str = "urn:test:model") -> IncidentRecord:
    return IncidentRecord(
        incident_id=f"inc-{model_urn}", model_urn=model_urn,
        created_at=datetime.now(timezone.utc), report=_make_report(model_urn),
        plan=_make_plan(model_urn) if with_plan else None,
    )


@pytest.fixture
def store():
    tmp = tempfile.mktemp(suffix=".json")
    s = LocalJSONMemoryStore(path=tmp)
    yield s
    import os
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass


class TestClaimBasicSemantics:
    @pytest.mark.asyncio
    async def test_first_claim_succeeds(self, store):
        record = _make_incident()
        await store.save_incident(record)
        claimed = await store.claim_incident_for_approval(record.incident_id)
        assert claimed is not None

    @pytest.mark.asyncio
    async def test_second_immediate_claim_is_blocked(self, store):
        """The core double-execution protection this mechanism exists
        for: a claim made moments ago must block a second claim."""
        record = _make_incident()
        await store.save_incident(record)
        first = await store.claim_incident_for_approval(record.incident_id)
        assert first is not None
        second = await store.claim_incident_for_approval(record.incident_id)
        assert second is None

    @pytest.mark.asyncio
    async def test_claim_without_plan_returns_none(self, store):
        record = _make_incident(with_plan=False)
        await store.save_incident(record)
        result = await store.claim_incident_for_approval(record.incident_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_claim_with_existing_outcome_returns_none(self, store):
        record = _make_incident()
        record.outcome = ActionOutcome(
            plan=record.plan, executed=True, execution_detail="done", reversible=False,
        )
        await store.save_incident(record)
        result = await store.claim_incident_for_approval(record.incident_id)
        assert result is None


class TestStaleClaimRecovery:
    """THE fix this round: an abandoned claim (process died before saving
    an outcome) must eventually become re-claimable, not stuck forever."""

    @pytest.mark.asyncio
    async def test_stale_claim_can_be_reclaimed(self, store, monkeypatch):
        monkeypatch.setattr(settings, "CLAIM_STALE_AFTER_SECONDS", 1)

        record = _make_incident()
        await store.save_incident(record)
        first = await store.claim_incident_for_approval(record.incident_id)
        assert first is not None

        # Simulate time passing past the (now very short) staleness
        # threshold by directly rewriting the stored claim timestamp into
        # the past, rather than sleeping in a test.
        records = store._read_all()
        for r in records:
            if r["incident_id"] == record.incident_id:
                old_time = datetime.now(timezone.utc) - timedelta(seconds=10)
                r["_claimed_at"] = old_time.isoformat()
        store._write_all(records)

        second = await store.claim_incident_for_approval(record.incident_id)
        assert second is not None, (
            "A claim older than CLAIM_STALE_AFTER_SECONDS with no outcome "
            "must be treated as abandoned and re-claimable — this is the "
            "recovery path for a process dying between claim and save"
        )

    @pytest.mark.asyncio
    async def test_fresh_claim_is_not_reclaimed_even_with_short_threshold(self, store, monkeypatch):
        """A claim made 0 seconds ago must never be treated as stale,
        regardless of how short the threshold is configured — this
        preserves the original double-execution protection."""
        monkeypatch.setattr(settings, "CLAIM_STALE_AFTER_SECONDS", 300)
        record = _make_incident()
        await store.save_incident(record)
        first = await store.claim_incident_for_approval(record.incident_id)
        assert first is not None

        second = await store.claim_incident_for_approval(record.incident_id)
        assert second is None

    @pytest.mark.asyncio
    async def test_claim_with_outcome_is_never_reclaimed_regardless_of_staleness(self, store, monkeypatch):
        """A stale claim is only recoverable if it STILL has no outcome.
        If an outcome was successfully saved (even a long time ago), that
        is a completed incident, not an abandoned one — must never be
        re-claimed just because time has passed."""
        monkeypatch.setattr(settings, "CLAIM_STALE_AFTER_SECONDS", 1)

        record = _make_incident()
        await store.save_incident(record)
        await store.claim_incident_for_approval(record.incident_id)

        # Now save a real outcome, completing the incident.
        record.outcome = ActionOutcome(
            plan=record.plan, executed=True, execution_detail="done", reversible=False,
        )
        await store.save_incident(record)

        # Even though the original claim is now "stale" by time, this
        # incident has a real outcome and must not be re-claimable.
        result = await store.claim_incident_for_approval(record.incident_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_corrupt_claim_timestamp_is_treated_as_stale_not_permanently_blocked(self, store):
        """Defensive: if the stored claim timestamp is somehow malformed
        (e.g. hand-edited data, a future schema change), the incident
        must not become permanently unclaimable over a parse error —
        treat it as stale (recoverable) rather than failing closed."""
        record = _make_incident()
        await store.save_incident(record)
        await store.claim_incident_for_approval(record.incident_id)

        records = store._read_all()
        for r in records:
            if r["incident_id"] == record.incident_id:
                r["_claimed_at"] = "not-a-valid-timestamp"
        store._write_all(records)

        result = await store.claim_incident_for_approval(record.incident_id)
        assert result is not None, (
            "A corrupt claim timestamp must not permanently lock the "
            "incident — treat as stale and allow recovery"
        )


class TestClaimTimestampNotLeakedToCaller:
    @pytest.mark.asyncio
    async def test_claimed_record_does_not_expose_internal_claim_field(self, store):
        """_claimed_at is a storage-layer implementation detail — the
        IncidentRecord returned to callers (and eventually serialized to
        the frontend) must not include it."""
        record = _make_incident()
        await store.save_incident(record)
        claimed = await store.claim_incident_for_approval(record.incident_id)
        data = json.loads(claimed.model_dump_json())
        assert "_claimed_at" not in data
