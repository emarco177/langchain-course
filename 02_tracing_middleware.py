"""02 · Observe: middleware sees every step of the loop, so tracing is just a middleware.

Needs OPENAI_API_KEY. Prints a LangSmith link when LANGSMITH_TRACING is on.
"""

import time

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    TracePolicy,
    omit_payload,
    wrap_model_call,
)
from langchain_core.tracers import LangChainTracer
from langsmith.utils import tracing_is_enabled

from desk import SYSTEM_PROMPT, TOOLS, ask, reset

load_dotenv()


class TimelineMiddleware(AgentMiddleware):
    # Hooks receive the whole state (the customer's card number included): keep it off their spans.
    trace_policy = TracePolicy(process_inputs=omit_payload)

    def log(self, text: str, depth: int = 0) -> None:
        self.step += 1
        print(f"  {self.step:>2} {'  ' * depth}{text}")

    def before_agent(self, state, runtime):
        self.step, self.start = 0, time.perf_counter()
        self.log("before_agent")

    def before_model(self, state, runtime):
        self.log(f"before_model    {len(state['messages'])} messages")

    def after_model(self, state, runtime):
        calls = [c["name"] for c in state["messages"][-1].tool_calls]
        self.log(f"after_model     → {', '.join(calls) or 'final answer'}")

    def wrap_tool_call(self, request, handler):
        start = time.perf_counter()
        result = handler(request)
        ms = (time.perf_counter() - start) * 1000
        self.log(f"tool {request.tool_call['name']:<14} {ms:.0f} ms", depth=1)
        return result

    def after_agent(self, state, runtime):
        self.log(f"after_agent     total {time.perf_counter() - self.start:.1f} s")


TIMELINE = TimelineMiddleware()


@wrap_model_call
def stopwatch(request, handler):
    start = time.perf_counter()
    response = handler(request)
    ms = (time.perf_counter() - start) * 1000
    TIMELINE.log(f"stopwatch       model {ms:.0f} ms", depth=1)
    return response


agent = create_agent(
    "openai:gpt-6-luna",
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    middleware=[TIMELINE, stopwatch],
)


if __name__ == "__main__":
    for task in [
        "What did I buy in order A100?",
        "Refund order B200 in full, it arrived broken.",
    ]:
        reset()
        tracer = LangChainTracer() if tracing_is_enabled() else None
        config = {"tags": ["step-02"], "metadata": {"lesson": "observe"}}
        ask(agent, task, config={**config, "callbacks": [tracer] if tracer else []})
        if tracer:
            tracer.wait_for_futures()
            print(f"trace:   {tracer.get_run_url()}")
        print()
