from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")


react_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that can use tools to answer questions.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question""",
        ),
        ("human", "Question: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Use what you have
agent = create_agent(llm=llm, tools=tools, system_prompt=react_prompt)
# executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# result = executor.invoke({"input": "What's the weather?"})
# print(result["output"])


def main():
    result = agent.invoke({"input": "search for 3 jobs for a data engineer on linkedin and list their details"}) # agent.invoke({"input": "search for 3 jobs for a data engineer on linkedin and list their details"})
    print(result)


if __name__ == "__main__":
    main()
