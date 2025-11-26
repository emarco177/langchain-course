from os import name
from dotenv import load_dotenv
from typing import Union, List
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description, tool
from langchain_openai import ChatOpenAI
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser

from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case

    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"{tool_name} not found.")


def main():
    print("Hello from ReAct LangChain!")
    # print(get_text_length(text='Dog'))


if __name__ == "__main__":
    main()
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    llm = ChatOpenAI(temperature=0, stop=["\nObservation", "Observation"], callbacks=[AgentCallbackHandler()])

    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    # agent = prompt | llm | ReActSingleInputOutputParser()

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step = agent.invoke(
            {"input": "What is the length of the text: 'Dog'", "agent_scratchpad": intermediate_steps}
        )

        if isinstance(agent_step, AgentAction):
            tool = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool)
            tool_input = agent_step.tool_input
            observation = tool_to_use.func(str(tool_input))
            print(f"observation = {observation}")
            intermediate_steps.append((agent_step, str(observation)))

    
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)