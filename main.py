import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

load_dotenv(override=True)

class SearchResults(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the search result")


class WebSearchResults(BaseModel):
    """Schema for agent response with answer and source"""

    answer: str = Field(description="The answer to the user's question")
    sources: List[SearchResults] = Field(description="The sources used to answer the question")


# tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# @tool
# def search_engine(query: str) -> str:
#     """Tool that searches over the web for information
    
#     Args:
#         query: The query to search for
    
#     Returns:
#         The search results
#     """
#     print(f"Searching the web for: {query}")
#     return tavily.search(query)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# tools = [search_engine]
tools = [TavilySearch()]
inputs = {
    "messages": [
        HumanMessage(content="Get me the recent job openings for a Python developer in around the New York area and the posted days is not more than 7 days")
    ]
}

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=WebSearchResults
)

# messages = [HumanMessage(content="Get me the recent job openings for a Python developer in around the New York area and the posted days is not more than 7 days")]
response = agent.invoke(inputs)
print(response)