"""01 · A support agent with no middleware: model + tools + prompt.

Needs OPENAI_API_KEY.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent

from desk import REFUNDS, SYSTEM_PROMPT, TOOLS, ask, reset, traced

load_dotenv()

agent = create_agent("openai:gpt-6-luna", tools=TOOLS, system_prompt=SYSTEM_PROMPT)


if __name__ == "__main__":
    for task in [
        "What did I buy in order A100?",
        "Refund order B200 in full, it arrived broken.",
    ]:
        reset()
        with traced(task):
            ask(agent, task)
            print(f"refunds: {REFUNDS}")
