"""04 · A hook can pause the loop: a human approves, edits or rejects the refund.

The pause is saved state: a checkpointer keeps the run under its thread_id until we resume it.

Needs OPENAI_API_KEY.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from desk import REFUNDS, SYSTEM_PROMPT, TOOLS, reset, short

load_dotenv()

TASK = "Refund order B200 in full, it arrived broken."
DECISIONS = {
    "approve": {"type": "approve"},
    "edit": {
        "type": "edit",
        "edited_action": {
            "name": "refund_order",
            "args": {"order_id": "B200", "amount": 100.0},
        },
    },
    "reject": {"type": "reject", "message": "Send it to the returns team instead."},
}

agent = create_agent(
    "openai:gpt-6-luna",
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    # Interrupts need somewhere to park the state until a human answers.
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "refund_order": {"allowed_decisions": ["approve", "edit", "reject"]}
            }
        )
    ],
)


if __name__ == "__main__":
    for thread_id, decision in DECISIONS.items():
        reset()
        config = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": TASK}]}, config, version="v2"
        )
        action = result.interrupts[0].value["action_requests"][0]
        print(f"thread:   {thread_id}")
        print(f"paused:   {action['name']}({action['args']})")
        print(f"refunds:  {REFUNDS}  (before resume)")
        # Resuming is a new invoke on the same thread, carrying the human's decision.
        result = agent.invoke(
            Command(resume={"decisions": [decision]}), config, version="v2"
        )
        print(f"decision: {decision}")
        print(f"refunds:  {REFUNDS}  (after resume)")
        if result.interrupts:  # the model asked for another gated call
            again = result.interrupts[0].value["action_requests"][0]
            print(f"paused:   again, {again['name']}({again['args']})")
        else:
            print(f"answer:   {short(result.value['messages'][-1].text)}")
        print()
