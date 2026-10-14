from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from tavily import TavilyClient

# Load environment variables (.env)
load_dotenv()

# Initialize Tavily client
tavily = TavilyClient()

# ✅ Simple search tool (works well with Ollama)
@tool
def search_web(query: str) -> str:
    """
    Search the web for up-to-date information.

    Args:
        query: What you want to search for

    Returns:
        Search results as text
    """
    print(f"🔍 Searching: {query}")
    result = tavily.search(query=query)
    return str(result)   # Convert to string (important)


# ✅ Initialize Ollama model
llm = init_chat_model("ollama:mymodel", temperature=0)

# ✅ Attach tools
tools = [search_web]

# ✅ Create agent
agent = create_agent(
    model=llm,
    tools=tools
)


def pretty_print(result):
    print("\n==============================")
    print("🤖 AI RESPONSE")
    print("==============================\n")

    for msg in result["messages"]:
        if msg.type == "ai" and msg.content:
            print(msg.content)

    print("\n==============================\n")


# Use it


def main():
    print("🌍 Web Search Agent (Ollama)")

    user_query = "Find 3 AI engineer jobs in Sweden using LangChain with links"

    result = agent.invoke({
        "messages": HumanMessage(content=user_query)
    })

    print("\n✅ Result:\n")
    pretty_print(result)

if __name__ == "__main__":
    main()