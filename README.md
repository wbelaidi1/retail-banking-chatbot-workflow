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
