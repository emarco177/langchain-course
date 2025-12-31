from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# from tavily import TavilyClient
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """The schema for a source used by the agent."""

    url: str = Field(description="The URL of the source")


# tavily = TavilyClient()


class AgentResponse(BaseModel):
    """The schema for the agent response with answer and sources."""

    answer: str = Field(description="The agent's answer to the user's query")
    sources: list[Source] = Field(
        default_factory=list, description="The sources used to generate the response"
    )


llm = ChatOpenAI(model="gpt-5")
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")

    invoke_args = {
        "messages": [
            HumanMessage(
                content="Find 3 data engineering jobs in Linkedin for Los Angeles?"
            )
        ]
    }
    result = agent.invoke(invoke_args)
    print(result)


# @tool
# def search(query):
#     """
#     Tool that searches the web

#     Args:
#         query (str): The query to search for
#     returns:
#         The result of the search
#     """
#     print(f"searching for {query}")
#     return tavily.search(query)
#     # return "Tokyo weather is sunny"

if __name__ == "__main__":
    main()
