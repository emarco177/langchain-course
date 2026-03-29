from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from travily import TravilyClient

travily = TravilyClient()


@tool
def search(query: str) -> str:
    """
    tool that searches the web for a given query
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for {query}")
    return travily.search(query=query)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="what is the weather in tokyo?")]})
    print(result)
     

if __name__ == "__main__":
    main()
