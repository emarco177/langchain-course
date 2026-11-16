"""05 · Your own MCP server: two tools over the LangChain docs index (llms.txt).

It is also the fallback for the whole section. If the hosted server is down or
moves, run this and set MCP_URL=http://localhost:8000/mcp. Lessons 01-04 work
unchanged, because they only know "an MCP server at MCP_URL".

Needs no API key.

    uv run python 05_docs_server.py
"""

from urllib.parse import urlparse

import httpx
from fastmcp import FastMCP

from settings import LLMS_TXT_URL

mcp = FastMCP("docs")

MAX_CHARS = 20_000


async def fetch(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as http:
        response = await http.get(url)
        response.raise_for_status()
        return response.text


@mcp.tool
async def search_docs(query: str) -> str:
    """Search the LangChain, LangGraph and LangSmith docs index. Returns matching pages as markdown links."""
    words = query.lower().split()
    index = await fetch(LLMS_TXT_URL)
    links = [line for line in index.splitlines() if "](" in line]
    scored = [(sum(word in line.lower() for word in words), line) for line in links]
    hits = [line for score, line in sorted(scored, key=lambda s: -s[0]) if score][:8]
    return (
        "\n".join(hits) or f"No pages matched {query!r}. Try fewer or different words."
    )


@mcp.tool
async def read_doc(url: str) -> str:
    """Read one docs page. Pass a URL returned by search_docs."""
    allowed = urlparse(LLMS_TXT_URL).hostname
    if urlparse(url).hostname != allowed:
        return f"Refused: this server only reads pages on {allowed}."
    return (await fetch(url))[:MAX_CHARS]


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)
