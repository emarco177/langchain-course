from dotenv import load_dotenv
    
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient

tavily = TavilyClient()

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
    return tavily.search(query=query)

llm = ChatOllama(model="llama3.2")
tool = [search]
agent = create_agent(model=llm, tools=tool)

def main():
    print("Hello from LangChain!")
    result = agent.invoke({"messages": HumanMessage(content="Search for ai engineering jobs through linkedin post in last 1 hour in bengalore india")})
    # we need to pass dictionary containing key as "messages" to agent which is going to contain the input query
    print(result)

if __name__ == "__main__":
    main()