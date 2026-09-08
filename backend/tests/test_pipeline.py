"""
Tests for pipeline.py's guard against silently investigating a model_urn
the lineage source doesn't actually represent.

THE GAP THIS CLOSES: MockDataHubClient always returns the SAME fixed demo
graph regardless of the requested model_urn -- it only stamps
root_model_urn with whatever was asked for, without that URN actually
being a node in the returned graph. upstream_nodes() correctly returns []
for a model_urn not present in the graph (the right behavior for that
function in isolation), but left unchecked at the pipeline level this
meant requesting an unsupported model_urn -- a completely realistic case,
since /api/investigate accepts model_urn as an open query parameter --
silently produced isolated_root_causes=[] and report=None: a result
INDISTINGUISHABLE from a legitimate "investigated, found nothing wrong."
"""
from __future__ import annotations

import warnings

import pytest

from app.pipeline import run_full_pipeline


class TestUnrepresentedModelUrnGuard:
    @pytest.mark.asyncio
    async def test_unsupported_model_urn_raises_clear_error(self):
        with pytest.raises(ValueError, match="not represented in the lineage graph"):
            await run_full_pipeline(
                model_urn="urn:li:mlModel:(totally,unrelated,model)",
                write_back=False,
            )

    @pytest.mark.asyncio
    async def test_error_message_includes_the_requested_urn(self):
        with pytest.raises(ValueError, match="totally,unrelated,model"):
            await run_full_pipeline(
                model_urn="urn:li:mlModel:(totally,unrelated,model)",
                write_back=False,
            )

    @pytest.mark.asyncio
    async def test_error_explicitly_distinguishes_from_no_drift_found(self):
        """The error message itself must make the distinction explicit,
        since this is precisely the confusion the bug caused."""
        with pytest.raises(ValueError, match="NOT the same as"):
            await run_full_pipeline(
                model_urn="urn:li:mlModel:(some,other,model)",
                write_back=False,
            )

    @pytest.mark.asyncio
    async def test_supported_demo_model_still_works_normally(self):
        """Confirm the fix didn't break the one model_urn the demo data
        generator actually supports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = await run_full_pipeline(
                model_urn="urn:li:mlModel:(demo,fraud_model_v3,PROD)",
                inject_drift=True,
                write_back=False,
            )
        assert result["report"] is not None
        assert len(result["graph"].nodes) > 0

    @pytest.mark.asyncio
    async def test_supported_demo_model_with_no_drift_still_returns_none_report_legitimately(self):
        """Sanity check the fix doesn't over-correct: a SUPPORTED model
        with genuinely no injected drift should still be allowed to
        legitimately return report=None -- that guard only fires for a
        model_urn absent from the graph entirely, not for a real
        'nothing found' result on a real, present model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = await run_full_pipeline(
                model_urn="urn:li:mlModel:(demo,fraud_model_v3,PROD)",
                inject_drift=False,
                write_back=False,
            )
        # Should not raise, and report may legitimately be None here.
        assert "report" in result
