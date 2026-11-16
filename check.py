"""Smoke test: is the MCP server still there, and does a tool call still work?

Run it before recording, or let .github/workflows/mcp-smoke.yml run it weekly.
Needs no API key. Exits non-zero when something is broken.
"""

import asyncio
import sys

from fastmcp import Client

from settings import MCP_URL


async def main() -> int:
    try:
        async with Client(MCP_URL, timeout=30) as client:
            tools = await client.list_tools()
            assert tools, "server returned no tools"
            names = [tool.name for tool in tools]
            search = next((t for t in tools if "search" in t.name.lower()), None)
            assert search, f"no search tool among {names}"
            schema = search.input_schema
            arg = (schema.get("required") or list(schema.get("properties", {})))[0]
            result = await client.call_tool(search.name, {arg: "create_agent"})
            assert result.content, f"{search.name} returned nothing"
    except Exception as error:  # noqa: BLE001 - any failure is a failed check
        print(f"FAIL  {MCP_URL}\n      {type(error).__name__}: {error}")
        print(
            "      Fallback: uv run python 05_docs_server.py, then MCP_URL=http://localhost:8000/mcp"
        )
        return 1
    print(f"OK    {MCP_URL}\n      tools: {names}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
