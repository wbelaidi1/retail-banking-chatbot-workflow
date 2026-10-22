"""Worker entrypoint: registers the retail banking chatbot workflow."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mistralai.workflows as workflows

from src.workflows.banking_chatbot import RetailBankingChatbotWorkflow


async def main() -> None:
    await workflows.run_worker([RetailBankingChatbotWorkflow])


if __name__ == "__main__":
    asyncio.run(main())
