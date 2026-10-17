"""04 · A hook can pause the loop: a human approves, edits or rejects the refund.

The pause is saved state: a checkpointer keeps the run under its thread_id until we resume it.

Needs OPENAI_API_KEY.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from desk import REFUNDS, SYSTEM_PROMPT, TOOLS, reset, short, traced

load_dotenv()

TASK = "Refund order B200 in full, it arrived broken."


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


def ask_human(action: dict) -> dict:
    """You are the human: approve, edit or reject the paused tool call."""
    print(f"paused:   {action['name']}({action['args']})")
    print(f"refunds:  {REFUNDS}  (nothing ran yet)")
    choice = input("          [a]pprove / [e]dit / [r]eject? ").strip().lower()[:1]
    if choice == "e":
        amount = float(input("          new amount: $"))
        args = {**action["args"], "amount": amount}
        return {"type": "edit", "edited_action": {"name": action["name"], "args": args}}
    if choice == "r":
        reason = input("          reason for the agent: ")
        return {"type": "reject", "message": reason}
    return {"type": "approve"}


if __name__ == "__main__":
    reset()
    config = {"configurable": {"thread_id": "desk-1"}}
    print(f"user:     {TASK!r}")
    with traced(TASK):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": TASK}]}, config, version="v2"
        )
        # Every pause is saved state; resuming is a new invoke carrying your decision.
        while result.interrupts:
            action = result.interrupts[0].value["action_requests"][0]
            decision = ask_human(action)
            result = agent.invoke(
                Command(resume={"decisions": [decision]}), config, version="v2"
            )
            print(f"refunds:  {REFUNDS}  (after {decision['type']})")
        print(f"answer:   {short(result.value['messages'][-1].text)}")
