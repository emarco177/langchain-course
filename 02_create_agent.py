"""02 · Host #1: a LangChain agent. MCP tools go straight into create_agent.

Needs OPENAI_API_KEY.
"""

import asyncio

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

from settings import MCP_URL, MODEL, QUESTION, SYSTEM_PROMPT


async def main():
    async with MCPAdapter(MCP_URL) as adapter:  # the MCP client
        tools = await adapter.list_tools()  # MCP tools -> LangChain tools
    print("tools:", [tool.name for tool in tools])

    agent = create_agent(MODEL, tools, system_prompt=SYSTEM_PROMPT)  # the host

    result = await agent.ainvoke({"messages": [{"role": "user", "content": QUESTION}]})
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
