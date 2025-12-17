import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

class Source(BaseModel):
    """
    Scheme for a source used by the agent
    """
    url: str = Field(description="URL of the source")

class AgentResponse(BaseModel):
    """
    Scheme for the agent response with sources and final answer
    """
    answer: str = Field(description="The final answer from the agent")
    sources: List[Source] = Field(default_factory=list, description="List of sources used by the agent")

@tool
def brave_search(query: str) -> str:
    """
    Tool that searches over internet
    
    Args:
        query (str): query to search for
    Returns:
        the search results
    """
    print(f"Searching for: {query}")
    return "cold weather in India"

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
tools = [brave_search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="Give me a summary of the current weather in India with sources.")]})
    print(f"Agent result: {result}")
    
if __name__ == "__main__":
    main()