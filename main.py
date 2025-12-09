import os
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch 



# Use GitHub Models via OpenAI-compatible endpoint
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or "gpt-4o-mini" / another model you enabled on GitHub
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com",  # GitHub Models endpoint
    temperature=0,
)

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course2!")
    result = agent.invoke({
        "messages": [HumanMessage(content="Search for three job postings for an AI engineer in chain in the Bay area in LinkedIn, and to list all of their details here.")]
    })

    print(result)

if __name__ == "__main__":
    main()


'''import os
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
    """
    Tool that searches over internet
    Args:
        query: The query to search for

    Returns:
        The search results
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

# Use GitHub Models via OpenAI-compatible endpoint
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or "gpt-4o-mini" / another model you enabled on GitHub
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com",  # GitHub Models endpoint
    temperature=0,
)

tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course2!")
    result = agent.invoke({
        "messages": [HumanMessage(content="Search for three job postings for an AI engineer in chain in the Bay area in LinkedIn, and to list all of their details here.")]
    })

    print(result)

if __name__ == "__main__":
    main()'''
