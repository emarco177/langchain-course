from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# LangChain Classic (v1)
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain_tavily import TavilySearch
from schemas import AgentResponse
from prompt import REACT_CHAT_JSON_FINAL

load_dotenv()

# ---- Tools (for the AGENT only) ----
tools = [TavilySearch()]  # ensure TAVILY_API_KEY is set

# ---- LLM (no tool_choice binding) ----
llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
    num_predict=1024,
    repeat_penalty=1.2,
)

# ---- Build the prompt with format-instructions ONLY for the Final Answer ----
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt = ChatPromptTemplate.from_template(REACT_CHAT_JSON_FINAL).partial(
    format_instructions=output_parser.get_format_instructions()
)

# ---- ReAct (text) agent ----
agent = create_react_agent(llm, tools, prompt=react_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=250,
    return_intermediate_steps=False,
)

def main():
    res = agent_executor.invoke({
        "input": "What is the current weather in Cairo, Egypt?",
        "chat_history": [],
    })
    print(res["output"])  # expected to be JSON per AgentResponse

if __name__ == "__main__":
    main()
