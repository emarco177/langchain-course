from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv
    
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema of the source used by the agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")
    
# using existing tool by tavily
llm = ChatOllama(model="llama3.2")
tool = [TavilySearch()]
agent = create_agent(model=llm, tools=tool, response_format=AgentResponse)

def main():
    print("Hello from LangChain!")
    result = agent.invoke({"messages": HumanMessage(content="Search for ai engineering jobs through linkedin in bengalore india")})
    # we need to pass dictionary containing key as "messages" to agent which is going to contain the input query
    print(result)

if __name__ == "__main__":
    main()