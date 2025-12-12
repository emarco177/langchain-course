from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
from langchain_tavily import TavilySearch
load_dotenv()

# tavily = TavilyClient()




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


llm = ChatOpenAI(model="gpt-5")
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(llm, tools=tools)

invoke_args = {"messages": [HumanMessage(content="Find 3 dataengineering jobs in Linkedin for Los Angeles?")]}

result = agent.invoke(invoke_args)
print(result)
