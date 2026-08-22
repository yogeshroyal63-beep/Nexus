# How I Built Nexus — An Autonomous Developer Agent with Strands Agents SDK and Amazon Bedrock

*This post was created for the purposes of entering the Agents for Humans Hackathon.*

---

## The Problem

Every ML team I have talked to has the same story. A model drifts in production. Someone notices it two days later in a standup. Three people spend a morning tracing the lineage, running statistical tests, reading logs, and opening tickets. By the time the root cause is found and a fix is deployed, it has been a full week.

None of that work required creativity. It was all mechanical — read the numbers, follow the graph, decide the action. The kind of work that should not need a human at all.

That is the problem Nexus solves.

---

## What Nexus Does

Nexus is an autonomous developer agent that runs in the background, monitors ML model performance, detects drift, isolates the root cause using causal analysis, and decides and executes the right remediation action on its own. It only surfaces to the developer when a real decision is needed — high risk action, low confidence, or something it has never seen before.

The loop looks like this:

```
Detect drift → Isolate root cause → Reason with LLM → Plan with Strands → Execute → Verify → Remember
```

It runs on a schedule via AWS SNS and EventBridge. Most of the time it runs, resolves the issue, and logs the incident without anyone knowing. You check the Incident History tab in the morning and see what it handled overnight.

---

## Why Strands Agents SDK

I had used LangGraph before for agent pipelines. Strands felt fundamentally different in one important way: tools are just Python functions.

```python
from strands import Agent, tool
from strands.models import BedrockModel

@tool
def assess_risk_level(action_type: str, confidence: float, has_similar_failures: bool) -> str:
    """
    Deterministically assess risk level for a proposed action.
    Never trust the LLM to self-report risk — compute it in code.
    """
    high_risk = {"rollback_model_version"}
    medium_risk = {"trigger_retrain"}
    
    risk = "high" if action_type in high_risk else "medium" if action_type in medium_risk else "low"
    requires_approval = (
        confidence < 0.75 or risk == "high" or has_similar_failures
    )
    return json.dumps({"risk_level": risk, "requires_approval": requires_approval})

agent = Agent(
    model=BedrockModel(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0"),
    system_prompt=PLANNER_SYSTEM_PROMPT,
    tools=[assess_risk_level, check_past_incidents],
)
```

The key design choice here: `assess_risk_level` is deterministic. The LLM calls it and gets back a computed answer. It cannot hallucinate a risk level. This is what makes the agent trustworthy enough to act autonomously — the safety gate is code, not model judgment.

---

## The Architecture

The backend is FastAPI with four main agents:

**Planner** — powered by Strands Agents SDK + Amazon Bedrock Claude 3.5 Sonnet. Gets the diagnosed report, calls its tools, returns an action decision.

**Executor** — carries out the action: retrain trigger, rollback, data quarantine, or GitHub issue. Degrades gracefully when credentials are not configured.

**Verifier** — re-runs drift detection after execution to check if the action worked. Sets `verified: true` or `verified: false` on the outcome.

**Memory** — DynamoDB in production, local JSON in dev. The Planner checks past incidents before deciding — if an action failed verification twice on the same model, it will not repeat it.

The SNS endpoint (`/api/sns/drift-check`) receives push notifications from EventBridge on a 30-minute schedule. It runs the full loop in the background. The developer sees the result in the Incident History view.

---

## The Part That Took the Longest

Getting the Strands agent to ground its decisions in actual evidence rather than hallucinating target URNs took real work.

The system prompt has strict rules — `target_urn` must be the model URN or one of the root cause node URNs from the trace. But LLMs drift under pressure. So I added a hard enforcement layer in Python:

```python
def _enforce_target_grounding(parsed: dict, report: RootCauseReport) -> dict:
    allowed = {report.model_urn} | {c.node_urn for c in report.raw_trace.isolated_root_causes}
    if parsed.get("target_urn") not in allowed:
        parsed["action_type"] = "no_action"
        parsed["target_urn"] = report.model_urn
        parsed["confidence"] = 0.0
        parsed["rationale"] = "Target could not be grounded — defaulted to no_action."
    return parsed
```

If the agent proposes a target that is not in the trace, the code overrides it to `no_action` and escalates. The LLM never gets to act on a hallucinated URN.

---

## What AWS Makes Possible

DynamoDB as the memory backend is what makes the agent genuinely cross-session. Every incident is stored with its full report, plan, outcome, and verification result. When the Planner sees the same model drifting again, it checks whether a retrain worked last time. If it failed verification twice, it escalates instead of repeating the same mistake.

That persistent, queryable memory is what separates an agent from a script that runs once and forgets everything.

The SNS + EventBridge trigger is what makes it truly autonomous. No cron job on a developer's laptop. The agent wakes up every 30 minutes, checks the model, handles what it can, and goes back to sleep.

---

## Try It

The full source is at: https://github.com/yogeshroyal63-beep/nexus

```bash
cd backend
cp .env.example .env
# Fill in AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, GROQ_API_KEY
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open the frontend at http://localhost:5173 and click "Run Nexus".

---

*Built for the Agents for Humans Hackathon using AWS Strands Agents SDK, Amazon Bedrock, DynamoDB, SNS, and EventBridge. #AllThingsAgenticHackathon*
