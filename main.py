"""
from dotenv import load_dotenv

# import os

load_dotenv()


def main():
    print("Hello from langchain-course!")
    # print(os.environ.get("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
"""

from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
# from tavily import TavilyClient
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek

# tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query) # "Tokyo weather is sunny"

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatDeepSeek(model="deepseek-chat") # ChatOllama(model="deepseek-r1:latest") # ChatOpenAI(model="gpt-5")
tools = [TavilySearch()] # [search] works with tavily_client()
agent = create_agent(model=llm, tools=tools) # , response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                # content="What is the weather in Tokyo?" # 
                content = "search for 3 job postings for an ai engineer using langchain in the Boston area on linkedin and list their details"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()