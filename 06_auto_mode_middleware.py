"""06 · Jev as a tool gate: AutoModeMiddleware.

The other decision every agent harness makes, over and over: "should this tool
call be allowed to run?" Coding agents like Claude Code, Codex, and Cursor all
have an "auto mode" that lets safe calls through and stops dangerous ones — and
until now that classifier was closed-source. `AutoModeMiddleware` (experimental)
gives you the same thing, powered by Jev, in one line.

How it works:

  * It hooks `wrap_tool_call`, immediately before a tool executes.
  * For each tool you LIST, it asks Jev a `Noul` question — "is this call risky?"
    — over the recent messages plus the proposed call (name, args, description).
  * Probability >= 0.5 → the tool is NOT run; the agent receives an error
    `ToolMessage` explaining the block and carries on from there.
  * Tools you did not list bypass classification entirely.
  * It fails CLOSED: if the classifier errors, the tool does not run.

The default instructions are about authorization — only explicit user messages
can authorize an action, and tool output is treated as data, not instructions,
which is what defends against prompt injection. Here we turn the `criteria` knob
to a simpler, deterministic policy — "writes are risky, reads are not" — so you
can see the gate fire predictably: a read request goes through, a delete request
is blocked, and the agent tells the user it couldn't do it.

Needs TYPESAFE_API_KEY and OPENAI_API_KEY.
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_typesafe import NoulCriteria
from langchain_typesafe.experimental.middleware import AutoModeMiddleware

from models import FAST_MODEL

# A tiny in-memory "notes app" so the tools have something real to do.
NOTES: dict[str, str] = {
    "groceries": "eggs, oat milk, coffee beans",
    "ideas": "lesson on system one models",
}


@tool
def read_note(title: str) -> str:
    """Return the contents of the note with the given title."""
    return NOTES.get(title, f"no note titled {title!r}")


@tool
def delete_note(title: str) -> str:
    """Permanently delete the note with the given title."""
    NOTES.pop(title, None)
    return f"deleted {title!r}"


# Only `delete_note` is gated. `read_note` is never classified.
auto_mode = AutoModeMiddleware(
    tools=[delete_note],
    criteria=NoulCriteria(
        true="The call deletes, overwrites, or otherwise changes stored data.",
        false="The call only reads data and changes nothing.",
    ),
)

agent = create_agent(
    FAST_MODEL,
    tools=[read_note, delete_note],
    middleware=[auto_mode],
)


def run(task: str) -> None:
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    print(f"task:   {task!r}")
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            status = "BLOCKED" if message.status == "error" else "ran"
            print(f"  tool {message.name!r:<14} {status:<8} → {message.content}")
    print(f"answer: {result['messages'][-1].content!r}")
    print(f"notes now: {sorted(NOTES)}")
    print()


run("What's on my groceries note?")
run("Delete my groceries note.")
