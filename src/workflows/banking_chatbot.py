"""BNP-style retail banking multi-agent chatbot — Mistral Workflow prototype.

Conversational InteractiveWorkflow surfaced in Vibe Work (+ > Workflows).
Simulates the multi-agent architecture:
  - Orchestrator: deterministic intent routing (workflow code, not a model call)
  - FAQ Agent: Mistral LLM agent answering from a curated knowledge base
  - Account Agent: read-only tool calls against a mock core-banking store
  - Transaction Agent: step-up auth + explicit confirmation + audited card lock/unlock
"""

from pydantic import BaseModel

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai

# ---------------------------------------------------------------------------
# Mock core banking store (in-memory; resets on worker restart — prototype only)
# ---------------------------------------------------------------------------
BANK_DB: dict = {
    "customers": {
        "demo": {
            "name": "Sophie Martin",
            "agency": "Paris Opéra (Agence 2041)",
            "auth_level": 1,
            "accounts": [
                {"name": "Compte Courant •2201", "balance": 1830.55},
                {"name": "Compte sur Livre •8873", "balance": 4250.12},
            ],
            "cards": {
                "c1": {"name": "Visa Premier", "last4": "4532", "locked": False},
                "c2": {"name": "Mastercard Gold", "last4": "7781", "locked": False},
            },
        }
    },
    "audit_log": [],
}

# Curated FAQ knowledge base used by the FAQ Agent.
FAQ_KB = """Agency: Paris Opéra (Agence 2041), 12 Boulevard des Italiens.
Opening hours: Monday-Friday 9:00-17:30, Saturday by appointment only. Closed on public holidays.
Phone support: +33 1 40 14 XX XX, Monday-Saturday 8:00-20:00.
Current account fees: card EUR 45/year, account maintenance EUR 6.50/month, SEPA transfers free within the EU.
Card lock: instant, free, fully reversible from the app or this chat.
Lost/stolen card: lock the card immediately, then call 0 800 000 000 (24/7, free)."""

INTENT_KEYWORDS = [
    ("faq", ["hour", "open", "agency", "branch", "agence", "horaire", "fee", "fees", "charge", "cost", "frais"]),
    ("balance", ["balance", "solde", "how much", "accounts"]),
    ("transactions", ["transaction", "history", "statement", "releve", "relevé"]),
    ("lock", ["lock", "block", "freeze", "bloque", "gel"]),
    ("unlock", ["unlock", "unfreeze", "reactivate", "unblock", "debloque", "débloque"]),
]


def classify_intent(message: str) -> str:
    t = message.lower()
    for intent, keywords in INTENT_KEYWORDS:
        if any(k in t for k in keywords):
            return intent
    return "unknown"


def _text(inp) -> str:
    return "".join(c.text for c in (inp.message or [])) if inp and inp.message else ""


def resolve_user(user_id: str) -> str:
    """Map any user_id to a known demo profile instead of failing."""
    return user_id if user_id in BANK_DB["customers"] else "demo"


# ---------------------------------------------------------------------------
# Tool layer — deterministic activities (least privilege: each tool does one thing)
# ---------------------------------------------------------------------------
@workflows.activity()
async def fetch_balances(user_id: str) -> list:
    customer = BANK_DB["customers"].get(resolve_user(user_id))
    return customer["accounts"]


@workflows.activity()
async def fetch_cards(user_id: str) -> list:
    customer = BANK_DB["customers"].get(resolve_user(user_id))
    return [{"id": cid, **card} for cid, card in customer["cards"].items()]


@workflows.activity()
async def set_card_status(user_id: str, card_id: str, locked: bool) -> dict:
    """Tool call to the (mock) Card API. Idempotent and reversible."""
    customer = BANK_DB["customers"].get(resolve_user(user_id))
    if card_id not in customer["cards"]:
        raise ValueError("Card not found for this customer")
    customer["cards"][card_id]["locked"] = locked
    return {"id": card_id, **customer["cards"][card_id]}


@workflows.activity()
async def write_audit(user_id: str, action: str, details: str) -> dict:
    import datetime

    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "user_id": user_id,
        "action": action,
        "details": details,
    }
    BANK_DB["audit_log"].append(entry)
    return entry


# ---------------------------------------------------------------------------
# Orchestrator workflow
# ---------------------------------------------------------------------------
class ChatbotInput(BaseModel):
    user_id: str = "demo"


@workflows.workflow.define(
    name="retail-banking-chatbot",
    workflow_display_name="Retail Banking Chatbot",
    workflow_description=(
        "Multi-agent retail banking chatbot prototype: FAQs (agency hours, fees), "
        "account balance, and audited card lock/unlock with step-up authentication "
        "and explicit confirmation. Demo customer: user_id 'demo'."
    ),
)
class RetailBankingChatbotWorkflow(workflows.InteractiveWorkflow):
    @workflows.workflow.entrypoint
    async def run(self, input: ChatbotInput) -> workflows_mistralai.ChatAssistantWorkflowOutput:
        user_id = input.user_id
        if user_id not in BANK_DB["customers"]:
            await workflows_mistralai.send_assistant_message(
                f"No test profile for '{user_id}' in this prototype — continuing with the "
                "demo customer (Sophie Martin)."
            )
            user_id = "demo"
        customer = BANK_DB["customers"][user_id]
        first_name = customer["name"].split()[0]

        await workflows_mistralai.send_assistant_message(
            f"Hello {first_name}! I'm your BNP banking assistant. I can answer FAQs "
            "(agency opening hours, fees), check your balance, or lock/unlock your cards. "
            "How can I help?"
        )
        user_input = await self.wait_for_input(
            workflows_mistralai.ChatInput(
                "What would you like to do?",
                suggestions=[
                    [workflows_mistralai.TextChunk(text="What are my agency's opening hours?")],
                    [workflows_mistralai.TextChunk(text="What's my account balance?")],
                    [workflows_mistralai.TextChunk(text="Lock my credit card")],
                ],
            ),
            timeout=900,
        )
        message = _text(user_input)

        for _ in range(10):
            intent = classify_intent(message)
            await self._handle_intent(user_id, intent, message)

            await workflows_mistralai.send_assistant_message(
                "Anything else I can do for you?"
            )
            nxt = await self.wait_for_input(
                workflows_mistralai.ChatInput("Type your request or say 'no' to finish."),
                timeout=900,
            )
            message = _text(nxt)
            if message.strip().lower() in {"no", "nope", "nothing", "non", "no thanks", "done"}:
                break

        return workflows_mistralai.ChatAssistantWorkflowOutput(
            content=[workflows_mistralai.TextOutput(text="Thank you for banking with us. Have a great day!")]
        )

    async def _handle_intent(self, user_id: str, intent: str, message: str) -> None:
        if intent == "faq":
            await self._faq_agent(user_id, message)
        elif intent == "balance":
            await self._account_agent(user_id)
        elif intent == "transactions":
            await self._account_agent(user_id)
        elif intent in ("lock", "unlock"):
            await self._card_action(user_id, intent)
        else:
            await workflows_mistralai.send_assistant_message(
                "I'm not sure I understood. I can help with agency opening hours, fees, "
                "your balance, or locking/unlocking a card."
            )

    # --- FAQ Agent: LLM grounded on the curated knowledge base ---
    async def _faq_agent(self, user_id: str, question: str) -> None:
        customer = BANK_DB["customers"].get(resolve_user(user_id), {})
        try:
            session = workflows_mistralai.RemoteSession(stream=True)
            faq_agent = workflows_mistralai.Agent(
                model="mistral-small-latest",
                name="faq-agent",
                description="Answers retail banking FAQ questions from a curated knowledge base.",
                instructions=(
                    "You are the FAQ Agent of a retail bank chatbot. Answer ONLY from the "
                    "knowledge base below. Be concise and friendly (2-4 sentences). If the "
                    "answer is not in the knowledge base, say you don't have that "
                    "information and suggest contacting phone support.\n\n"
                    f"Customer home agency: {customer.get('agency', 'their home agency')}.\n\n"
                    f"KNOWLEDGE BASE:\n{FAQ_KB}"
                ),
            )
            await workflows_mistralai.Runner.run(agent=faq_agent, inputs=question, session=session)
        except Exception:
            await workflows_mistralai.send_assistant_message(
                "Your home agency is " + customer.get("agency", "Paris Opéra (Agence 2041)")
                + ". Opening hours: Mon-Fri 9:00-17:30, Saturday by appointment."
            )

    # --- Account Agent: read-only tool calls ---
    async def _account_agent(self, user_id: str) -> None:
        accounts = await fetch_balances(user_id)
        lines = "\n".join(f"• {a['name']}: €{a['balance']:.2f}" for a in accounts)
        await workflows_mistralai.send_assistant_message(
            f"Here are your accounts (read-only lookup, Level 1 auth):\n{lines}"
        )
        await write_audit(user_id, "READ_BALANCES", "Balance lookup performed (read-only).")

    # --- Transaction Agent: step-up auth + confirmation + audited write ---
    async def _card_action(self, user_id: str, intent: str) -> None:
        verb = "lock" if intent == "lock" else "unlock"
        cards = await fetch_cards(user_id)

        verify_item = workflows_mistralai.TodoListItem(title="Verify identity (OTP)", description="Step-up authentication")
        select_item = workflows_mistralai.TodoListItem(title="Select card", description="Identify the card concerned")
        confirm_item = workflows_mistralai.TodoListItem(title="Confirm action", description="Explicit user confirmation")
        exec_item = workflows_mistralai.TodoListItem(title=f"Execute & audit", description=f"{verb.capitalize()} card via Card API")

        async with workflows_mistralai.TodoList(items=[verify_item, select_item, confirm_item, exec_item]):
            # 1. Step-up authentication (demo OTP)
            async with verify_item:
                await workflows_mistralai.send_assistant_message(
                    f"🔒 {verb.capitalize()}ing a card is a sensitive action, so I need to verify "
                    "your identity. I've sent a 6-digit code by SMS. For this prototype, the code is **123456**."
                )
                ok = False
                for _ in range(3):
                    otp = await self.wait_for_input(
                        workflows_mistralai.ChatInput("Enter the 6-digit code sent by SMS."),
                        timeout=300,
                    )
                    if _text(otp).strip() == "123456":
                        ok = True
                        break
                    await workflows_mistralai.send_assistant_message("That code doesn't match. Please try again (demo code: 123456).")
                if not ok:
                    await workflows_mistralai.send_assistant_message(
                        "I couldn't verify your identity, so no action was taken. You can try again anytime."
                    )
                    return

            # 2. Card selection
            async with select_item:
                if len(cards) == 1:
                    card = cards[0]
                else:
                    card_suggestions = [
                        [workflows_mistralai.TextChunk(text=f"{c['name']} •{c['last4']}")]
                        for c in cards
                    ]
                    sel = await self.wait_for_input(
                        workflows_mistralai.ChatInput("Which card is concerned?", suggestions=card_suggestions),
                        timeout=300,
                    )
                    chosen = _text(sel).lower()
                    card = next(
                        (c for c in cards if c["last4"] in chosen or c["name"].lower() in chosen),
                        cards[0],
                    )

            # 3. Explicit confirmation
            async with confirm_item:
                conf = await self.wait_for_input(
                    workflows_mistralai.ChatInput(
                        f"Please confirm: **{verb} card {card['name']} •{card['last4']}**?",
                        suggestions=[
                            [workflows_mistralai.TextChunk(text=f"Yes, {verb} my card")],
                            [workflows_mistralai.TextChunk(text="No, cancel")],
                        ],
                    ),
                    timeout=300,
                )
                if "yes" not in _text(conf).lower():
                    await write_audit(user_id, f"CANCEL_{intent.upper()}", f"User cancelled {verb} of {card['name']} •{card['last4']}.")
                    await workflows_mistralai.send_assistant_message("No problem, nothing was changed.")
                    return

            # 4. Execute + audit
            async with exec_item:
                locked = intent == "lock"
                updated = await set_card_status(user_id, card["id"], locked)
                await write_audit(
                    user_id,
                    f"{'LOCK' if locked else 'UNLOCK'}_CARD",
                    f"Card {updated['name']} •{updated['last4']} {'locked' if locked else 'unlocked'} — auth L2 (OTP), explicit confirmation received.",
                )
                await workflows_mistralai.send_assistant_message(
                    f"✅ Done! Your {updated['name']} •{updated['last4']} is now "
                    f"**{'locked' if locked else 'active'}**. This action has been logged for audit. "
                    + ("You can unlock it anytime by asking me." if locked else "Happy spending!")
                )
