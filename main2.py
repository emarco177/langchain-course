from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent

from langchain.tools import tool
from langchain_core.messages import HumanMessage

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily = TavilyClient()

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer:str = Field(description="The agents answer to the query")
    sources: list[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatOllama(temperature=0, model="llama3.2:3b")
# llm = ChatOllama(temperature=0, model="gemma3:4b")
# llm = ChatGroq(temperature=0, model="llama-3.1-8b-instant")


tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job posts about data engineer in madrid in linkedin using langchain and list their details")})
    print(result)

if __name__ == "__main__":
    main()


# llm_with_tools = llm.bind_tools(tools)
