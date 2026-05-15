from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage 
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

def main():
    print("Hello from langchain-course!")

llm = ChatOpenAI();
tools = [TavilySearch()];
agent = create_agent(model=llm,tools=tools)
response = agent.invoke({"messages":HumanMessage(content="search for 2 job postings for an ai engineer using langchain in the banglore area on linked in and list their details")});
print(response);


if __name__ == "__main__":
    main()
