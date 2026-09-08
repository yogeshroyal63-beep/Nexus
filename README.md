# Nexus — Autonomous Developer Agent

> **AWS Agents for Humans Hackathon** · Professional Agents track

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![AWS Strands](https://img.shields.io/badge/AWS-Strands%20Agents%20SDK-orange)](https://github.com/strands-agents/sdk-python)
[![Amazon Bedrock](https://img.shields.io/badge/Amazon-Bedrock-FF9900)](https://aws.amazon.com/bedrock/)

Nexus watches your ML systems while you work. It detects drift, isolates root causes using causal analysis, and autonomously decides and executes the right remediation action — only surfacing when a human decision is genuinely needed.

---

## The Problem

Every ML team has the same story. A model drifts in production. Someone notices two days later. Three people spend a morning tracing lineage, running tests, reading logs, opening tickets. None of that needed a human. It was all mechanical: read the numbers, follow the graph, decide the action.

Nexus handles that loop automatically, end to end.

---

## The Autonomous Loop

```
Detect drift → Isolate root cause (causal DAG) → Reason (Groq LLaMA 3.3)
    → Plan (Strands SDK + Bedrock Claude 3.5) → Execute (gated) → Verify → Remember (DynamoDB)
```

- Runs on schedule via **AWS SNS + EventBridge** — no developer intervention
- Only surfaces when action is **high risk or low confidence**
- Learns from past incidents stored in **DynamoDB** — won't repeat a failed action

---

## Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | **AWS Strands Agents SDK** |
| LLM reasoning (Planner) | **Amazon Bedrock** — Claude 3.5 Sonnet |
| LLM reasoning (Explainer) | Groq — LLaMA 3.3 70B |
| Memory | **AWS DynamoDB** (local JSON fallback) |
| Background triggers | **AWS SNS + EventBridge** |
| Deployment | **AWS App Runner** |
| Backend | FastAPI + Python 3.12 |
| Frontend | React 18 + Vite + Recharts + ReactFlow |

---

## Quickstart (Local — Zero Cost)

### Prerequisites
- Python 3.11+
- Node.js 18+
- A free [Groq API key](https://console.groq.com)
- AWS credentials (for Bedrock + DynamoDB — or use local JSON fallback)

### Backend

```bash
cd backend
cp .env.example .env
# Edit .env — minimum required: GROQ_API_KEY
# For full Strands+Bedrock: also set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The backend starts at **http://localhost:8000**. Without AWS credentials it runs with:
- Local JSON memory (`data/nexus_incidents.json`)
- Planner falls back to safe escalation (no Bedrock call)
- Full drift detection and causal isolation still works

### Frontend

```bash
cd frontend
npm install
npm run dev
# Opens http://localhost:5173
```

Click **Run Nexus** to trigger the full agentic loop.

---

## AWS Deployment

### 1. Set up infrastructure

```bash
chmod +x scripts/setup-aws.sh
AWS_REGION=us-east-1 ./scripts/setup-aws.sh
```

This creates:
- DynamoDB table `nexus-incidents`
- SNS topic `nexus-drift-check`
- EventBridge rule (every 30 minutes)

### 2. Deploy backend to App Runner

```bash
cd backend

# Build and push to ECR
aws ecr create-repository --repository-name nexus-backend --region us-east-1
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t nexus-backend .
docker tag nexus-backend:latest <account>.dkr.ecr.us-east-1.amazonaws.com/nexus-backend:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/nexus-backend:latest

# Deploy via App Runner (console or CLI)
# Set env vars: AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
#               GROQ_API_KEY, MEMORY_BACKEND=dynamodb
```

### 3. Subscribe SNS to your deployed endpoint

```bash
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:<account>:nexus-drift-check \
  --protocol https \
  --notification-endpoint https://<your-app-runner-url>/api/sns/drift-check \
  --region us-east-1
```

Nexus now runs autonomously every 30 minutes without any developer action.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM reasoning layer |
| `AWS_REGION` | For AWS | AWS region (default: us-east-1) |
| `AWS_ACCESS_KEY_ID` | For AWS | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | For AWS | AWS secret key |
| `BEDROCK_MODEL_ID` | For Strands | Bedrock model (default: claude-3-5-sonnet) |
| `MEMORY_BACKEND` | No | `local` (default) or `dynamodb` |
| `GITHUB_TOKEN` | No | For GitHub issue write-back |
| `GITHUB_REPO` | No | `owner/repo` for write-back |
| `AUTO_EXECUTE_ENABLED` | No | Enable autonomous execution (default: true) |
| `AUTO_EXECUTE_MIN_CONFIDENCE` | No | Minimum confidence to auto-execute (default: 0.75) |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/lineage/{model_urn}` | Fetch ML lineage graph |
| `POST` | `/api/investigate` | Report-only pipeline (no agent action) |
| `POST` | `/api/run` | **Full agentic loop** (Strands + Bedrock) |
| `GET` | `/api/incidents` | List all stored incidents |
| `GET` | `/api/incidents/{id}` | Get single incident |
| `POST` | `/api/incidents/{id}/approve` | Human approve an escalated plan |
| `POST` | `/api/sns/drift-check` | SNS push endpoint for background runs |

---

## Running Tests

```bash
cd backend
pip install pytest pytest-asyncio
pytest tests/ -v
# Expected: 24 passed
```

---

## Project Structure

```
nexus/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── coordinator.py   # Full autonomous loop
│   │   │   ├── planner.py       # Strands SDK + Bedrock planner
│   │   │   ├── executor.py      # Action execution + verification
│   │   │   └── memory.py        # DynamoDB + local JSON backends
│   │   ├── drift/engine.py      # KS test, PSI, cosine drift detection
│   │   ├── causal/isolator.py   # Causal DAG root cause isolation
│   │   ├── llm/reasoning.py     # Groq LLM explanation layer
│   │   ├── lineage/             # DataHub client (mock mode)
│   │   ├── writeback/           # GitHub write-back agent
│   │   ├── api/routes.py        # FastAPI routes
│   │   └── main.py              # App entrypoint
│   ├── tests/test_nexus.py      # 24 tests
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── views/               # Lineage, Drift, Report, Agent, History
│   │   ├── components/          # TopBar, StatusFooter, ErrorBoundary
│   │   ├── theme/               # Dark/light ThemeContext
│   │   └── api/client.js        # API client
│   └── package.json
├── scripts/setup-aws.sh         # AWS infrastructure setup
├── apprunner.yaml               # App Runner config
├── architecture.svg             # Architecture diagram
├── BUILDER_POST.md              # builder.aws.com post
├── LICENSE                      # MIT
└── README.md
```

---

## Judging Criteria Alignment

| Criterion | Nexus |
|-----------|-------|
| **Technological Implementation** | Strands SDK with 2 custom tools, Bedrock as LLM, DynamoDB memory, SNS trigger, AgentCore-compatible |
| **Design** | Full product: lineage graph, drift timeline, agent decision view, incident history, dark/light theme |
| **Potential Impact** | ML engineers lose hours to drift response — Nexus resolves most incidents autonomously |
| **Creativity** | Non-obvious Strands use: causal isolation + safety-gated remediation, not just Q&A |
| **Presentation** | Live demo, architecture diagram, 24 passing tests, reproducible setup |

---

## License

MIT © 2026 Yogesh Rayal — see [LICENSE](LICENSE)
