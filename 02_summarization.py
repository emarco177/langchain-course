"""02 · Summarization: a built-in middleware that rewrites state when the history grows.

Needs OPENAI_API_KEY.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from desk import SYSTEM_PROMPT, TOOLS, reset, short

load_dotenv()

agent = create_agent(
    "openai:gpt-6-luna",
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        # Tiny limits so it fires in a short demo. Real agents use ("fraction", 0.8) or ("tokens", ...).
        SummarizationMiddleware(
            model="openai:gpt-6-luna",
            trigger=("messages", 8),
            keep=("messages", 4),
        ),
    ],
    checkpointer=InMemorySaver(),
)


def is_summary(message) -> bool:
    return message.additional_kwargs.get("lc_source") == "summarization"


if __name__ == "__main__":
    reset()
    config = {"configurable": {"thread_id": "dana"}}
    for task in [
        "Hi, I'm Dana. What did I buy in order A100?",
        "And what's in order B200?",
        "Refund $20 on A100, the case was scratched.",
        "Thanks. Which orders have we talked about, and what did you refund?",
    ]:
        result = agent.invoke({"messages": [{"role": "user", "content": task}]}, config)
        messages = result["messages"]
        summarized = any(is_summary(m) for m in messages)
        print(f"user:      {task!r}")
        print(f"answer:    {short(messages[-1].text)}")
        print(f"messages:  {len(messages)}{'  ← summary + recent' if summarized else ''}")
        print()

    summary = next(m for m in messages if is_summary(m))
    print("summary in state:")
    print(summary.text)
