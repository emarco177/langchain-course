"""03 · Host #2: the same tools in a graph you build yourself.

Same server, same client, same tools as 02. Only the host changed.
Needs OPENAI_API_KEY.
"""

import asyncio

from langchain.chat_models import init_chat_model
from langchain.mcp import MCPAdapter
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from settings import MCP_URL, MODEL, QUESTION, SYSTEM_PROMPT


def build_graph(model, tools):
    model = model.bind_tools(tools)

    async def call_model(state: MessagesState):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]]
        return {"messages": [await model.ainvoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model", tools_condition
    )  # tool call? -> tools, else END
    graph.add_edge("tools", "model")
    return graph.compile()


async def main():
    async with MCPAdapter(MCP_URL) as adapter:
        tools = await adapter.list_tools()

    graph = build_graph(init_chat_model(MODEL), tools)

    async for step in graph.astream(
        {"messages": [{"role": "user", "content": QUESTION}]}, stream_mode="updates"
    ):
        for node, update in step.items():
            print(f"--- {node} ---")
            update["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
