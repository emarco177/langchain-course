from typing import Any

from dotenv import load_dotenv
from langchain.tools import BaseTool, tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

load_dotenv()


class AgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for agent events"""
    
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        print("\n🤖 Ag-CB: LLM Start")
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        print("✅ Ag-CB: LLM End")
    
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        print(f"\n🔧 Ag-CB: Tool Start: {serialized.get('name')}")
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        print(f"✅ Ag-CB: Tool End: {output}")


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length called with: {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: list[BaseTool], tool_name: str) -> BaseTool:
    for _tool in tools:
        if _tool.name == tool_name:
            return _tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("Hello LangChain Tools (.bind_tools)!")

    tools = [get_text_length]

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    )

    llm_with_tools = llm.bind_tools(tools)

    messages = [HumanMessage(content="What is the length of the word: DOG")]

    while True:
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = getattr(ai_message, "tool_calls", None) or []

        if len(tool_calls) > 0:
            messages.append(ai_message)

            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)

                print(f"Result: {observation}")

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            continue

        print(f"\n✨ Final Answer: {ai_message.content}")
        break
