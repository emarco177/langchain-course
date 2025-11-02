from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from schemas import AgentResponse

load_dotenv()

# ---- Tools ----
tools = [TavilySearch()]  # ensure TAVILY_API_KEY is set

# ---- LLM ----
llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
    num_predict=1024,
    repeat_penalty=1.2,
)

# ---- System Prompt ----
system_prompt = """You are a helpful AI assistant with access to search tools.
When answering questions, use the available tools to find accurate, up-to-date information.
Always cite your sources by including the URLs you found information from."""

# ---- Create Agent with Structured Output ----
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    response_format=AgentResponse,  # Built-in structured output support
)

def main():
    # Invoke agent with messages
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "What is the current weather in Cairo, Egypt?"}
        ]
    })
    
    # Access structured response
    print("Structured Response:")
    print(f"Answer: {result['structured_response']}")

if __name__ == "__main__":
    main()
