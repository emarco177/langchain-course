from dotenv import load_dotenv

load_dotenv()
import os


from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from langchain_tavily import TavilySearch


# tavily_client = TavilyClient()


# CUSTOM TOOL
# @tool
# def search_web(query: str) -> str:
#     """Search the web for information about the query"""
#     # search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
#     # return search.run(query)
#     print(f"Runnint tool with query: {query}")
#     response = tavily_client.search(query=query)
#     return response


llm = ChatOpenAI(model="gpt-5", temperature=0)
# tools = [search_web]
tools = [TavilySearch()]

agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Search for 3 job postings for an ai engineer using langchain on linkedin and list their details"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
