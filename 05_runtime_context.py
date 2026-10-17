"""05 · Steer: per-request context rewrites the prompt, model and tools.

The agent is built once; middleware reads who is calling before every model call.

Needs OPENAI_API_KEY.
"""

from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, wrap_model_call
from langchain.chat_models import init_chat_model

from desk import REFUNDS, SYSTEM_PROMPT, TOOLS, ask, reset

load_dotenv()

FAST = init_chat_model("openai:gpt-6-luna")
POWERFUL = init_chat_model("openai:gpt-6-sol")


@dataclass
class Context:
    user_name: str
    tier: str  # "free" | "pro"


@dynamic_prompt
def personalize(request) -> str:
    # dynamic_prompt replaces the system message, so build on SYSTEM_PROMPT
    ctx = request.runtime.context
    return (
        f"{SYSTEM_PROMPT} You are helping {ctx.user_name} ({ctx.tier} plan). "
        "If you cannot do something with your tools, say it was escalated to a human."
    )


@wrap_model_call
def by_tier(request, handler):
    if request.runtime.context.tier == "pro":
        return handler(request.override(model=POWERFUL))
    tools = [t for t in request.tools if t.name != "refund_order"]
    return handler(request.override(model=FAST, tools=tools))


agent = create_agent(
    FAST,
    tools=TOOLS,
    context_schema=Context,
    middleware=[personalize, by_tier],
)


if __name__ == "__main__":
    for ctx in [Context("Dana", "free"), Context("Sam", "pro")]:
        reset()
        print(f"caller:  {ctx.user_name} ({ctx.tier})")
        result = ask(
            agent, "Refund order A100, the headphones stopped working.", context=ctx
        )
        print(f"model:   {result['messages'][-1].response_metadata['model_name']}")
        print(f"refunds: {REFUNDS}")
        print()
