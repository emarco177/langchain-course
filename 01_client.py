"""01 · A bare MCP client: connect, list the tools, call one. No model, no agent.

This is what the MCP Inspector does when you click through it.
Needs no API key.
"""

import asyncio
import json

from fastmcp import Client

from settings import MCP_URL


def pick_search_tool(tools):
    """The server names its own tools, so find the search tool instead of hardcoding it."""
    for tool in tools:
        if "search" in tool.name.lower():
            return tool
    return tools[0]


async def main():
    print(f"server: {MCP_URL}\n")

    async with Client(MCP_URL) as client:  # initialize
        tools = await client.list_tools()  # tools/list
        for tool in tools:
            params = list(tool.input_schema.get("properties", {}))
            print(f"- {tool.name}({', '.join(params)})")
            print(f"    {(tool.description or '').strip().splitlines()[0][:100]}")

        tool = pick_search_tool(tools)
        schema = tool.input_schema
        arg = (schema.get("required") or list(schema.get("properties", {})))[0]
        call = {arg: "short-term memory"}
        print(f"\ncalling {tool.name}({json.dumps(call)})\n")

        result = await client.call_tool(tool.name, call)  # tools/call
        for block in result.content:
            print(getattr(block, "text", block)[:800])


if __name__ == "__main__":
    asyncio.run(main())
