from dotenv import load_dotenv
load_dotenv()
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

llm = ChatGroq(temperature=0, model="qwen/qwen3-32b") # more reliable tool use
tools = [TavilySearch()]
agent = create_react_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({
        "messages": HumanMessage(
            content="Search for 3 job postings for an AI engineer using LangChain in the Bay Area on LinkedIn. List their title, company, location, and job description details."
        )
    })
    # Print just the final response
    print(result["messages"][-1].content)
    return result


if __name__ == "__main__":
    main()