from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage 
from langchain_openai import ChatOpenAI 
from tavily import TavilyClient


tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """Searches the web for the given query and returns the results.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """ 

    print(f"Search results for query: {query}") 
    return tavily.search(query)


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for 3 job popstings of an AI Engineer using langchain in the bay area on linkedin and list their details")], "tool_choice": "auto"})
                              
    print(result)


if __name__ == "__main__":
    main()
