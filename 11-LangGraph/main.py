from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph,END

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON="agent_reason"
ACT= "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")]})
    print(res["messages"][LAST].content)