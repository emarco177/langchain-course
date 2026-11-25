
# import os
# from langchain_core.prompts import PromptTemplate 
# # from langchain_openai import ChatOpenAI
# from langchain_community.llms import Ollama
# from langchain_groq import ChatGroq

from langchain_community.llms import llamacpp
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model

from tavily import TavilyClient



MODEL = "mymodel"




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

#  tavily = TavilyClient()
@tool
def search(query: str) -> str:
    
    """
        Get the current weather for a given city.

        Args:
            city: Name of the city.

        Returns:
            A string describing the current weather conditions.
        """

    print(f"searching for {query}")
    return tavily.search(query=query)
    # return "Weather in Tokyo is sunny"



# llm = ChatOpenAI()
llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
tools = [search]
# agent = create_agent(model=llm, tools=tools)
# agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
agent = create_agent(model=llm, tools=tools)



# llm = ChatOpenAI()
# tools = [TavilySearch]
# agent = create_agent(model=llm, tools=tools)
def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="Find three ai engineer jobs in bay area in linkedin")})
    print(result)
    # # print(os.environ.get("OPENAI_API_KEY"))

    # result = agent.invoke(
    #     {"messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?")}
        
    # )
    # print(result)


if __name__ == "__main__":
    main()


