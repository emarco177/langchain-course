"""Example: Using multiple LLM providers with LangChain.

This script demonstrates how to switch between different LLM providers
(OpenAI, MiniMax, Anthropic, Google, Ollama) using the same code.

Usage
-----
Set the ``LLM_PROVIDER`` environment variable to choose a provider::

    LLM_PROVIDER=openai   python examples/multi_provider_example.py
    LLM_PROVIDER=minimax  python examples/multi_provider_example.py
    LLM_PROVIDER=ollama   python examples/multi_provider_example.py

Or let the script auto-detect from available API keys.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

from utils.llm_provider import get_chat_model, _detect_provider


def main():
    provider = _detect_provider()
    llm = get_chat_model()
    print(f"Auto-detected provider: {provider}")
    print(f"Using model: {llm.model_name if hasattr(llm, 'model_name') else 'N/A'}")
    print("-" * 50)

    summary_template = PromptTemplate(
        input_variables=["topic"],
        template="Give me a brief summary about {topic} in 2-3 sentences.",
    )

    chain = summary_template | llm

    response = chain.invoke({"topic": "the LangChain framework"})
    content = response.content if hasattr(response, "content") else str(response)
    print(content)


if __name__ == "__main__":
    main()
