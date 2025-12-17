import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    
    Args:
        query (str): query to search for
    Returns:
        the search results
    """
    print(f"Searching for: {query}")
    return "cold weather in India"

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in India?")]})
    print(f"Agent result: {result}")
    
if __name__ == "__main__":
    main()