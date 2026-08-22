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

from app.config import settings
from app.models.schemas import RootCauseReport, RootCauseTrace, SuggestedFix

logger = logging.getLogger(__name__)

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

    root_causes = [n for n in parsed.get("root_causes", []) if n in allowed_names]
    if not root_causes and allowed_names:
        logger.warning("LLM named no valid root causes; falling back to algorithm output.")
        root_causes = sorted(allowed_names)
    parsed["root_causes"] = root_causes

    fixes = parsed.get("suggested_fixes", [])
    filtered = [f for f in fixes if f.get("target_urn") in allowed_urns]
    parsed["suggested_fixes"] = filtered
    return parsed


class LLMReasoningLayer:
    def __init__(self, model: str | None = None):
        self.model = model or settings.LLM_MODEL
        self._client = Groq(api_key=settings.GROQ_API_KEY)

    def generate_report(self, trace: RootCauseTrace, _retries: int = 1) -> RootCauseReport:
        evidence_payload = trace.model_dump(mode="json")
        last_error: Exception | None = None

        for attempt in range(_retries + 1):
            try:
                response = self._client.chat.completions.create(
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
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                last_error = exc
                logger.warning("LLM response malformed on attempt %d/%d: %s", attempt + 1, _retries + 1, exc)
            except GroqError as exc:
                last_error = exc
                logger.warning("LLM API call failed on attempt %d/%d: %s", attempt + 1, _retries + 1, exc)

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
