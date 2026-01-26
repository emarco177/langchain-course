from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent

from langchain.tools import tool
from langchain_core.messages import HumanMessage

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from tavily import TavilyClient
from langchain_tavily import TavilySearchResults

tavily = TavilyClient()

llm = ChatOllama(temperature=0, model="llama3.2:3b")
# llm = ChatOllama(temperature=0, model="gemma3:4b")
# llm = ChatGroq(temperature=0, model="llama-3.1-8b-instant")

'''

@tool("brave_search")
def search(query: str) -> str:
    """
    Tool that searches over the internet
    Args:
        query: The query to search for
    Returns:
        The search result

    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job posts about data engineer in madrid in linkedin using langchain and list their details")})
    print(result)

'''



tools = [TavilySearchResults()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job posts about data engineer in madrid in linkedin using langchain and list their details")})
    print(result)

if __name__ == "__main__":
    main()


# llm_with_tools = llm.bind_tools(tools)
