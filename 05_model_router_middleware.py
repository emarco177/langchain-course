"""05 · Jev inside the agent loop: ModelRouterMiddleware.

Everything so far was Jev as a standalone classifier. This example puts it INSIDE
a LangChain agent, as middleware. The question a router answers is one of the
most common decisions in production agents: "does this task need the expensive
model, or will the cheap one do?" Answering that with another LLM call is slow
and costs as much as the call you were trying to save. Answering it with Jev
takes milliseconds and a fraction of a cent.

`ModelRouterMiddleware` (experimental, in `langchain_typesafe.experimental`)
hooks two points of the agent lifecycle:

  * `before_agent`   — classify the latest human message with a `Choice`
                       question whose labels are your model names, and store the
                       full `ChoiceAnswer` in agent state under `model_route`.
  * `wrap_model_call` — swap in the chosen model for every model call in the run.

You describe each model with a `criteria` string; that becomes the label
description Jev classifies against. The base model you pass to `create_agent`
is only a placeholder — the router overrides it.

Note the failure mode: a classifier error ends the run rather than silently
picking a default model. Routing is treated as part of the contract.

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

# The model string here is a placeholder; the router picks the real one per run.
agent = create_agent(FAST_MODEL, middleware=[router])


def run(task: str) -> None:
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})

    # The middleware keeps the WHOLE answer in state, not just the label, so the
    # probabilities and confidence are available to you (and to LangSmith traces).
    route = result["model_route"]
    print(f"task:        {task!r}")
    print(f"route:       {route.choice}  (confidence {route.confidence:.3f})")
    print(f"distribution:{ {k: round(v, 3) for k, v in route.probabilities.items()} }")
    answer = result["messages"][-1].content
    print(f"answer:      {answer[:160]!r}{'…' if len(answer) > 160 else ''}")
    print()


run("What is the capital of Australia? Answer in one word.")

run(
    "We're migrating a monolith with a 2TB Postgres database to microservices "
    "with zero downtime. Propose a phased plan and call out the two riskiest steps."
)
