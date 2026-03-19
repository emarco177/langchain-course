from dotenv import load_dotenv
load_dotenv() 

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

import os

tavily = TavilyClient()
os.getenv("OPENAI_API_KEY")
@tool
def search(query: str) -> str:
    """
     Tool that searches over internet
    Args: 
      query: The query search for  
    Returns: 
       The search results 
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)


# llm = ChatGroq(model="llama-3.1-8b-instant")

llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main(): 
    print("Hello from langchain-course!")
    # result = agent.invoke({"messages" :"What is the weather in Tokyo? "})
    result = agent.invoke({"messages" : HumanMessage(content="What is the weather in Tokyo? ")})
    print(result)

if __name__ == "__main__":
    main()
