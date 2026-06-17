from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient

load_dotenv()


class Source(BaseModel):
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate an answer")


tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Search the web and return a compact JSON payload.
    """
    print(f"searching for: {query}")
    results = tavily.search(
        query=query,
        max_results=3,   # keep it small
    )

    # Keep only compact fields to reduce token usage
    compact = []
    for item in results.get("results", []):
        compact.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "content": item.get("content", "")[:500],  # trim long snippets
        })

    return json.dumps(compact, ensure_ascii=False)


llm = ChatOllama(
    model="mymodel",   # e.g. "jobhelper"
    temperature=0,
    num_ctx=16384,                    # increase context window
)

agent = create_agent(model=llm, tools=[search])


def main():
    result = agent.invoke({
        "messages": [
            HumanMessage(content="Find three AI engineer jobs in the Sweden on LinkedIn")
        ]
    })
    print(result)


if __name__ == "__main__":
    main()