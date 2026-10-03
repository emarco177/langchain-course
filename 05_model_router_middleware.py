"""05 · ModelRouterMiddleware: Jev picks the fast or powerful model per request.

Needs TYPESAFE_API_KEY and OPENAI_API_KEY.
"""

from langchain.agents import create_agent
from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
)

from models import FAST_MODEL, POWERFUL_MODEL

router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(
            model=FAST_MODEL,
            criteria="Direct lookups, short factual answers, and small localized edits.",
        ),
        "powerful": ModelChoice(
            model=POWERFUL_MODEL,
            criteria="Multi-step reasoning, architecture, and high-stakes decisions.",
        ),
    },
    instructions="Choose the least costly model that can complete the task well.",
)

# Placeholder model; the router picks the real one.
agent = create_agent(FAST_MODEL, middleware=[router])


def run(task: str) -> None:
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})

    # The full ChoiceAnswer lives in state.
    route = result["model_route"]
    reply = result["messages"][-1]
    print(f"task:   {task!r}")
    print(f"jev:    {route.choice}  (confidence {route.confidence:.3f})")
    print(f"model:  {reply.response_metadata.get('model_name')}")
    print(f"answer: {reply.text[:160]!r}{'…' if len(reply.text) > 160 else ''}")
    print()


run("What is the capital of Australia? Answer in one word.")

run(
    "We're migrating a monolith with a 2TB Postgres database to microservices "
    "with zero downtime. Propose a phased plan and call out the two riskiest steps."
)
