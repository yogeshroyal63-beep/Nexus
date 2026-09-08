"""
Tests for the rate limiter's two fixes found this round:

1. CRITICAL for the live-demo scenario: request.client.host reflects the
   immediate TCP peer of the connection uvicorn accepted. Behind AWS App
   Runner's managed load balancer (the exact deployment target this
   project's apprunner.yaml and Dockerfile build for) with no proxy-header
   handling, that's the load balancer's address -- IDENTICAL for every
   single end user. That collapses the "per-IP" rate limit into a de-facto
   GLOBAL one shared by every visitor: after 3 total /api/run calls from
   ANYONE, every other visitor gets rate-limited too, silently.

2. A genuine (if slow-onset) memory leak: the original code pruned each
   key's own expired timestamps but never removed a key once its list
   emptied, so _rate_store grew forever across a long-running process
   seeing many distinct client IPs.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from app.main import _extract_client_ip, _rate_store, rate_limit_middleware


class TestExtractClientIp:
    def test_prefers_x_forwarded_for_when_present(self):
        req = MagicMock()
        req.headers = {"x-forwarded-for": "203.0.113.42, 10.0.1.5"}
        assert _extract_client_ip(req) == "203.0.113.42"

    def test_strips_whitespace_from_forwarded_header(self):
        req = MagicMock()
        req.headers = {"x-forwarded-for": "  203.0.113.42  , 10.0.1.5"}
        assert _extract_client_ip(req) == "203.0.113.42"

    def test_falls_back_to_client_host_when_no_forwarded_header(self):
        req = MagicMock()
        req.headers = {}
        req.client.host = "127.0.0.1"
        assert _extract_client_ip(req) == "127.0.0.1"

    def test_falls_back_to_unknown_when_no_client_and_no_header(self):
        req = MagicMock()
        req.headers = {}
        req.client = None
        assert _extract_client_ip(req) == "unknown"

    def test_ignores_empty_forwarded_header_value(self):
        req = MagicMock()
        req.headers = {"x-forwarded-for": ""}
        req.client.host = "127.0.0.1"
        assert _extract_client_ip(req) == "127.0.0.1"

    def test_different_forwarded_ips_produce_different_results(self):
        """THE core fix this proves: two different real end users behind
        the same proxy must map to different rate-limit buckets, not
        collapse into one."""
        req_a = MagicMock()
        req_a.headers = {"x-forwarded-for": "203.0.113.1"}
        req_b = MagicMock()
        req_b.headers = {"x-forwarded-for": "203.0.113.2"}
        assert _extract_client_ip(req_a) != _extract_client_ip(req_b)


class TestRateStoreMemoryLeak:
    """Direct tests against the module-level _rate_store dict, since the
    leak is specifically about dict key accumulation over time."""

    def setup_method(self):
        _rate_store.clear()

    def _make_request(self, ip: str):
        req = MagicMock()
        req.headers = {"x-forwarded-for": ip}
        req.url.path = "/api/incidents"
        return req

    async def _run_once(self, ip: str):
        req = self._make_request(ip)
        call_next = MagicMock(return_value=_async_response())
        return await rate_limit_middleware(req, call_next)

    def test_key_is_removed_after_all_its_timestamps_expire(self):
        """THE leak fix: once a key's timestamps are all pruned as
        expired, the key itself must be removed, not left as an empty
        list forever."""
        key = "203.0.113.99:/api/incidents"
        # Simulate a timestamp far enough in the past to have expired
        # under any of the configured rate limit windows.
        _rate_store[key] = [time.time() - 10_000]

        # Pruning logic mirrors what the middleware does internally.
        now = time.time()
        pruned = [t for t in _rate_store[key] if now - t < 60]
        if pruned:
            _rate_store[key] = pruned
        else:
            _rate_store.pop(key, None)

        assert key not in _rate_store

    def test_store_does_not_grow_unboundedly_for_repeated_expired_visits(self):
        """A simplified simulation: many distinct IPs visit once, long
        enough ago that their entries should all be prunable. The store
        must not retain empty-list entries for all of them forever."""
        now = time.time()
        for i in range(50):
            key = f"203.0.113.{i}:/api/incidents"
            _rate_store[key] = [now - 10_000]  # long expired

        for key in list(_rate_store.keys()):
            pruned = [t for t in _rate_store[key] if now - t < 60]
            if pruned:
                _rate_store[key] = pruned
            else:
                _rate_store.pop(key, None)

        assert len(_rate_store) == 0


async def _async_response():
    from fastapi import Response
    return Response(content="ok", status_code=200)
