# Retail Banking Chatbot — Mistral Workflow Prototype

Multi-agent retail banking chatbot deployed as a Mistral conversational Workflow.

## Architecture

- **Orchestrator**: RetailBankingChatbotWorkflow — deterministic intent routing in workflow code
- **FAQ Agent**: Mistral LLM agent grounded on a curated knowledge base (agency hours, fees)
- **Account Agent**: read-only tool activity (fetch_balances) on a mock core-banking store
- **Transaction Agent**: card lock/unlock with step-up auth (demo OTP 123456), explicit confirmation (wait_for_input), idempotent execution and audit logging

## Multi-agent safety demo

"Lock my credit card" runs the full Tier-2 flow: OTP verification → card selection → explicit Yes/No confirmation → audited write. Cancelling at any step executes nothing.

## Deploy (managed deployment)

1. Install the Mistral GitHub App on this repo: https://github.com/apps/mistralai
2. Connect GitHub in Studio: https://console.mistral.ai/connectors/github_app
3. Create a deployment from this repo (branch main, Dockerfile at root).
4. Requires pay-as-you-go enabled on the organization (or Enterprise plan).
5. Once deployed, the workflow retail-banking-chatbot runs in the Studio Console and appears in Vibe Work under + > Workflows.

## Try it

Input: { "user_id": "demo" } then chat. Example requests:
- "What are my agency's opening hours?"
- "What's my account balance?"
- "Lock my credit card" (demo OTP: 123456)

Note: the banking store is in-memory for the prototype — card locks reset on worker restart. Swap the activities for real Core Banking API calls in production.

## Standalone chat UI + log/trace viewer (`ui/`)

Besides Vibe Work and Studio, you can chat with the **same deployed workflow** from a local web UI, and browse its logs and OpenTelemetry trace in your own viewer — no Studio needed.

```bash
pip install -r ui/requirements.txt
export MISTRAL_API_KEY=...        # same key/console project as the deployment
uvicorn ui.app:app --reload --port 8000   # run from the repo root
```

- **http://localhost:8000** — chat UI. Starts a real execution, streams the assistant's messages, shows suggestion chips and the OTP/confirmation flows.
- **http://localhost:8000/logs** — log viewer. Per execution: lifecycle + assistant-message events, worker logs, and the execution's OTel spans with a small timeline.

### How it works

- The chat UI drives the managed deployment through the public Workflows API: `execute_workflow` to start, the `__get_pending_inputs` query to discover what the workflow is waiting for, and the `__submit_input` update to send each user message — the same protocol Vibe Work uses.
- Instead of redirecting OTLP from the worker (not reachable when the worker is managed by Mistral), the UI pulls each execution's OpenTelemetry trace from the platform (`GET /v1/workflows/executions/{id}/trace/otel`) plus its logs and event history, and stores everything in a local SQLite database (`ui/data/ui.db`, git-ignored). The `/logs` page reads only from SQLite.
- If you later self-host the worker, you can additionally point its standard OpenTelemetry exporter (`OTEL_EXPORTER_OTLP_ENDPOINT`) at a local collector (e.g. Jaeger all-in-one) for real-time OTLP; the SQLite capture here keeps working unchanged.

Configuration (env vars): `MISTRAL_API_KEY` (required), `WORKFLOW_NAME` (default `retail-banking-chatbot`), `UI_DB_PATH` (SQLite location), `UI_POLL_SECONDS`.
