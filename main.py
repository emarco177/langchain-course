import os
from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

tavily = TavilyClient()

@tool(description="useful for when you need to answer questions about current events")
def custom_search(query: str) -> dict:
    """
    Tool that searches the web for the query.
    Args:
    query: The search query
    :return:
    :return: A string containing search results
    """
    print(f"Searching for {query}...")
    return tavily.search(query=query)


llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatAnthropic(temperature=0, model="claude-haiku-4-5-20251001")
# tools = [TavilySearch()]
tools = [custom_search]
# agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
agent = create_agent(llm, tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            # "messages": HumanMessage(
            #     content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
            # )
            "messages": HumanMessage(content="What is the weather in Osaka, Japan?")
        }
    )
    print(result)


if __name__ == "__main__":
    main()
