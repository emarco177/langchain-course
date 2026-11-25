"""The only place a URL or model name is written down.

Every lesson imports from here, so when a URL moves or a model is retired you
change one line (or set it in .env) and all the lessons follow.
"""

import os
import warnings

from dotenv import load_dotenv

load_dotenv()

# langchain.mcp is in beta and says so on import. uv.lock pins the version
# these lessons were verified against, so the warning is noise here.
warnings.filterwarnings("ignore", message=".*langchain.mcp.*beta.*")

# The MCP server. Hosted by LangChain, public, no API key.
MCP_URL = os.getenv("MCP_URL", "https://docs.langchain.com/mcp")

# The chat model for the agents in 02 and 03.
MODEL = os.getenv("MODEL", "openai:gpt-6.2-sol")

# The docs index 05_docs_server.py serves.
LLMS_TXT_URL = os.getenv("LLMS_TXT_URL", "https://docs.langchain.com/llms.txt")

QUESTION = os.getenv(
    "QUESTION",
    "How do I add short-term memory to a LangChain agent? Link the docs page you used.",
)

SYSTEM_PROMPT = (
    "You answer questions about LangChain, LangGraph and LangSmith. "
    "Look the answer up with your tools before you reply, and link your sources."
)
