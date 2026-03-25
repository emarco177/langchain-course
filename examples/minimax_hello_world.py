"""Example: Using MiniMax with LangChain.

This script demonstrates how to use MiniMax's OpenAI-compatible API
with LangChain via the ``utils.llm_provider`` helper.

Prerequisites
-------------
1. Set your MiniMax API key::

       export MINIMAX_API_KEY="your-api-key-here"

2. Install dependencies::

       pip install langchain langchain-openai python-dotenv

3. Run::

       python examples/minimax_hello_world.py

   Or explicitly set the provider::

       LLM_PROVIDER=minimax python examples/minimax_hello_world.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from utils.llm_provider import get_chat_model


def main():
    llm = get_chat_model(provider="minimax")
    print(f"Using model: {llm.model_name}")

    response = llm.invoke(
        "Give me 3 interesting facts about the Python programming language."
    )
    print(response.content)


if __name__ == "__main__":
    main()
