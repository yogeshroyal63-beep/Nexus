"""
LLM Reasoning & Explanation Layer.

Uses Groq for the explanation/report generation step.
The Planner agent (agents/planner.py) uses AWS Bedrock via Strands SDK.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from groq import Groq, GroqError
from pydantic import ValidationError

from app.config import settings
from app.models.schemas import RootCauseReport, RootCauseTrace, SuggestedFix

logger = logging.getLogger(__name__)

_VALID_CONFIDENCE_VALUES = {"low", "moderate", "high"}

SYSTEM_PROMPT = """You are the reasoning layer of Nexus, an autonomous developer agent. \
You will be given a structured RootCauseTrace: statistical drift evidence and a causal \
graph trace that an algorithmic engine has ALREADY computed and isolated.

STRICT RULES:
1. NEVER invent or guess a root cause not present in `isolated_root_causes`. Explain only \
what the algorithm already isolated.
2. If `isolated_root_causes` is empty, say so plainly.
3. Ground every claim in the numeric evidence provided (p-values, PSI scores, intervention_delta).
4. Confidence must reflect the evidence: high requires large intervention_delta (>0.15).
5. Suggested fixes must be tied to a specific node URN from the trace.

Respond with ONLY a JSON object (no markdown fences, no preamble):
{
  "summary": "one or two sentence plain-language summary",
  "detailed_explanation": "several sentences walking through the evidence",
  "root_causes": ["human-readable node names from isolated_root_causes only"],
  "confidence": "low | moderate | high",
  "suggested_fixes": [
    {"action": "...", "target_urn": "...", "rationale": "..."}
  ]
}"""


def _enforce_grounding(parsed: dict, trace: RootCauseTrace) -> dict:
    allowed_names = {c.node_name for c in trace.isolated_root_causes}
    allowed_urns = {c.node_urn for c in trace.isolated_root_causes} | {trace.model_urn}

    # dict.fromkeys(...) dedupes while preserving order — a plain list
    # comprehension would let the model repeat a name twice in its own
    # JSON output straight through, which the frontend then uses as a
    # React list key (RootCauseReportView.jsx), producing a duplicate-key
    # warning and undefined reconciliation behavior for a purely cosmetic
    # LLM repetition that carries no additional information anyway.
    root_causes = list(dict.fromkeys(
        n for n in parsed.get("root_causes", []) if n in allowed_names
    ))
    if not root_causes and allowed_names:
        logger.warning("LLM named no valid root causes; falling back to algorithm output.")
        root_causes = sorted(allowed_names)
    parsed["root_causes"] = root_causes

    fixes = parsed.get("suggested_fixes", [])
    filtered = [f for f in fixes if f.get("target_urn") in allowed_urns]
    parsed["suggested_fixes"] = filtered

    # Confidence normalization: the Planner's ONLY low-confidence safety
    # gate is an exact string comparison (report.confidence == "low"). The
    # system prompt asks for exactly "low | moderate | high", but nothing
    # previously enforced that — a model returning "Low" (wrong case),
    # "very low", or any other off-vocabulary value would silently bypass
    # that check entirely, letting the Planner proceed to plan/act on a
    # report the model itself considered too uncertain to trust. Normalize
    # case/whitespace, and fail closed to "low" (the conservative,
    # act-less direction) for anything that still doesn't match one of the
    # three values the rest of the system understands — never fail open
    # to "high" on a value we don't recognize.
    raw_confidence = str(parsed.get("confidence", "")).strip().lower()
    if raw_confidence not in _VALID_CONFIDENCE_VALUES:
        logger.warning(
            "LLM returned unrecognized confidence value %r; defaulting to 'low' "
            "(fail closed) rather than risk silently bypassing the Planner's "
            "low-confidence safety gate.",
            parsed.get("confidence"),
        )
        raw_confidence = "low"
    parsed["confidence"] = raw_confidence

    return parsed


class LLMReasoningLayer:
    def __init__(self, model: str | None = None):
        self.model = model or settings.LLM_MODEL
        # Client construction deliberately deferred out of __init__ and into
        # generate_report()'s own try/except (see _get_client below).
        #
        # FIXED — client-construction crash bypassed the fallback safety net
        # entirely (found by running the documented zero-cost/no-API-key
        # quickstart path, not observed in production): Groq(api_key=...)
        # was previously called here, in __init__. Any failure at
        # construction time — no API key configured, or (as actually
        # happened) an installed groq/httpx combination where groq's
        # internal client still passed a `proxies` kwarg that httpx had
        # already removed — raised immediately, before generate_report()'s
        # try/except (built specifically to catch GroqError and fall back
        # to _deterministic_fallback_report) ever ran. The exception
        # propagated straight out of get_reasoning_layer() as an uncaught
        # 500, defeating the whole purpose of the fallback path on exactly
        # the "no GROQ_API_KEY set" scenario the README's quickstart
        # describes as supported. Fixed by constructing the client lazily,
        # inside generate_report()'s existing try/except, so any
        # construction-time failure is caught by the same
        # retry-then-deterministic-fallback logic as an API call failure.
        self._client: Groq | None = None

    def _get_client(self) -> Groq:
        if self._client is None:
            self._client = Groq(api_key=settings.GROQ_API_KEY)
        return self._client

    def generate_report(self, trace: RootCauseTrace, _retries: int = 1) -> RootCauseReport:
        evidence_payload = trace.model_dump(mode="json")
        last_error: Exception | None = None

        for attempt in range(_retries + 1):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Here is the RootCauseTrace evidence:\n\n"
                                f"{json.dumps(evidence_payload, indent=2)}"
                            ),
                        },
                    ],
                )
                text = response.choices[0].message.content
                text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(text)
                parsed = _enforce_grounding(parsed, trace)

                return RootCauseReport(
                    model_urn=trace.model_urn,
                    generated_at=datetime.now(timezone.utc),
                    summary=parsed["summary"],
                    detailed_explanation=parsed["detailed_explanation"],
                    root_causes=parsed["root_causes"],
                    confidence=parsed["confidence"],
                    suggested_fixes=[SuggestedFix(**f) for f in parsed["suggested_fixes"]],
                    raw_trace=trace,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
                # ValidationError added after finding it slipped through
                # uncaught: syntactically valid JSON with a schema violation
                # (e.g. a suggested_fix missing its required 'rationale'
                # field) raised pydantic.ValidationError, which is NOT a
                # subclass of any of the other three exception types here.
                # That let a single malformed field bypass this entire
                # retry-then-fallback safety net and propagate as an
                # uncaught 500 error instead of gracefully retrying or
                # falling back to the deterministic report — defeating the
                # explicit purpose of this loop.
                last_error = exc
                logger.warning("LLM response malformed on attempt %d/%d: %s", attempt + 1, _retries + 1, exc)
            except GroqError as exc:
                last_error = exc
                logger.warning("LLM API call failed on attempt %d/%d: %s", attempt + 1, _retries + 1, exc)
            except Exception as exc:
                # Catches client-construction failures surfaced via
                # _get_client() (missing/invalid API key, an
                # installed-package-version incompatibility raising
                # TypeError, etc.) that are not GroqError instances. These
                # are not worth retrying — the same construction call will
                # fail identically on every attempt — but they must still
                # land on the deterministic fallback below rather than
                # propagate as an uncaught 500.
                last_error = exc
                logger.error(
                    "LLM reasoning layer hit an unexpected error on attempt %d/%d: %s",
                    attempt + 1, _retries + 1, exc,
                )
                break

        logger.error("LLM reasoning layer failed after %d attempts; using deterministic fallback.", _retries + 1)
        return _deterministic_fallback_report(trace)


def _deterministic_fallback_report(trace: RootCauseTrace) -> RootCauseReport:
    names = [c.node_name for c in trace.isolated_root_causes]
    return RootCauseReport(
        model_urn=trace.model_urn,
        generated_at=datetime.now(timezone.utc),
        summary=(
            f"Root cause isolated algorithmically: {', '.join(names)}."
            if names else "No root cause could be isolated from the current evidence."
        ),
        detailed_explanation=(
            "The LLM explanation layer was unavailable. This is a deterministic summary "
            "generated directly from the causal isolation engine's trace."
        ),
        root_causes=names,
        confidence="low",
        suggested_fixes=[],
        raw_trace=trace,
    )


def get_reasoning_layer() -> LLMReasoningLayer:
    return LLMReasoningLayer()
