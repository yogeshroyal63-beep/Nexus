"""
Tests for the SNS webhook authentication fix.

THE GAP THIS CLOSES: /api/sns/drift-check had zero authentication and was
absent from rate limiting entirely -- once deployed, this URL is publicly
reachable (SNS itself must reach it over the internet), so anyone who
discovered it could trigger the full agentic pipeline (real Bedrock/Groq
calls, real GitHub issue creation if write-back is enabled) on demand,
with no rate limit slowing them down.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_secret():
    original = settings.SNS_WEBHOOK_SECRET
    yield
    settings.SNS_WEBHOOK_SECRET = original


class TestSnsWebhookAuth:
    def test_works_without_secret_when_none_configured(self):
        """Demo-friendly default: no extra setup required out of the box."""
        settings.SNS_WEBHOOK_SECRET = ""
        r = client.post("/api/sns/drift-check", json={"Message": "{}"})
        assert r.status_code == 200

    def test_rejects_missing_secret_when_one_is_configured(self):
        settings.SNS_WEBHOOK_SECRET = "test-secret"
        r = client.post("/api/sns/drift-check", json={"Message": "{}"})
        assert r.status_code == 403

    def test_rejects_wrong_secret(self):
        settings.SNS_WEBHOOK_SECRET = "test-secret"
        r = client.post("/api/sns/drift-check?secret=wrong", json={"Message": "{}"})
        assert r.status_code == 403

    def test_accepts_correct_secret_via_query_param(self):
        settings.SNS_WEBHOOK_SECRET = "test-secret"
        r = client.post("/api/sns/drift-check?secret=test-secret", json={"Message": "{}"})
        assert r.status_code == 200

    def test_accepts_correct_secret_via_header(self):
        settings.SNS_WEBHOOK_SECRET = "test-secret"
        r = client.post(
            "/api/sns/drift-check",
            json={"Message": "{}"},
            headers={"x-nexus-webhook-secret": "test-secret"},
        )
        assert r.status_code == 200

    def test_header_and_query_param_both_work_independently(self):
        settings.SNS_WEBHOOK_SECRET = "test-secret"
        # Header alone, no query param
        r1 = client.post(
            "/api/sns/drift-check",
            json={"Message": "{}"},
            headers={"x-nexus-webhook-secret": "test-secret"},
        )
        assert r1.status_code == 200
        # Query param alone, no header
        r2 = client.post("/api/sns/drift-check?secret=test-secret", json={"Message": "{}"})
        assert r2.status_code == 200


class TestSnsEndpointRateLimited:
    def test_endpoint_is_now_covered_by_rate_limiting(self):
        """THE gap: this endpoint was previously absent from RATE_LIMITS
        entirely. Confirm it now has an entry."""
        from app.main import RATE_LIMITS
        assert "/api/sns/drift-check" in RATE_LIMITS

    def test_rate_limit_actually_triggers(self):
        settings.SNS_WEBHOOK_SECRET = ""
        from app.main import _rate_store
        _rate_store.clear()

        limit, _window = __import__("app.main", fromlist=["RATE_LIMITS"]).RATE_LIMITS["/api/sns/drift-check"]
        statuses = []
        for _ in range(limit + 2):
            r = client.post("/api/sns/drift-check", json={"Message": "{}"})
            statuses.append(r.status_code)
        assert 429 in statuses
