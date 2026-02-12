from dotenv import load_dotenv
load_dotenv()
from langgraph.prebuilt.tool_node import tools_condition
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

load_dotenv()




llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

tools = [TavilySearch()]
agent = create_agent(model = llm, tools = tools)

def main():
    result = agent.invoke({"messages": HumanMessage(content="Give me 3 AI Engineer job post in india with apply link")})
    print(result)


if __name__ == "__main__":
    main()
