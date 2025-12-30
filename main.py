from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain.tools import tool, BaseTool
from azure_env import llm
from dotenv import load_dotenv

from callbacks import AgentCallbackHandler

from azure_env import llm

@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case

    return len(text)


def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    print("Hello LangChain Tools (.bind_tools)!")
    print("=" * 60)
    
    tools = [get_text_length]
    print(f"📦 Registered tools: {[tool.name for tool in tools]}")

    llm_with_tools = llm.bind_tools(tools)

    # Start conversation
    messages: List[BaseMessage] = [HumanMessage(content="What is the length of the word: DOG")]

    iteration = 0
    while True:
        iteration += 1
        print(f"\n🔄 ITERATION {iteration}")
        print(f"📨 Sending {len(messages)} message(s) to LLM...")
        print(f"\n📋 CURRENT MESSAGES LIST:")
        for i, msg in enumerate(messages, 1):
            print(f"   [{i}] {type(msg).__name__}: {msg.content[:100] if hasattr(msg, 'content') else msg}")
        
        ai_message = llm_with_tools.invoke(messages)
        
        print(f"\n🤖 AI Response received:")
        print(f"   Type: {type(ai_message).__name__}")
        print(f"   Content preview: {ai_message.content[:100] if ai_message.content else 'None'}")
        print(f"\n🔍 FULL AI_MESSAGE DEBUG:")
        print(f"{ai_message}")
        print(f"\n🔍 AI_MESSAGE ATTRIBUTES:")
        print(f"   - content: {ai_message.content}")
        print(f"   - response_metadata: {getattr(ai_message, 'response_metadata', 'N/A')}")
        print(f"   - tool_calls: {getattr(ai_message, 'tool_calls', 'N/A')}")
        
        # If the model decides to call tools, execute them and return results
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        print(f"\n   Tool calls found: {len(tool_calls)}")
        
        if len(tool_calls) > 0:
            print(f"\n🔧 MODEL WANTS TO USE TOOLS:")
            messages.append(ai_message)
            print(f"   ➕ Added AIMessage to messages list (now {len(messages)} messages)")
            
            for idx, tool_call in enumerate(tool_calls, 1):
                print(f"\n   Tool Call #{idx}:")
                print(f"      Full tool_call dict: {tool_call}")
                # tool_call is typically a dict with keys: id, type, name, args
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")
                
                print(f"      - Name: {tool_name}")
                print(f"      - Args: {tool_args}")
                print(f"      - Call ID: {tool_call_id}")

                print(f"\n   ⚙️  Executing tool '{tool_name}'...")
                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"   ✅ Tool result (observation): {observation}")

                tool_message = ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                messages.append(tool_message)
                print(f"   📝 Added ToolMessage to conversation (now {len(messages)} messages)")
            
            print(f"\n↩️  Continuing loop to let model see the tool results...")
            print("=" * 60)
            continue

        # No tool calls -> final answer
        print(f"\n✨ FINAL ANSWER (no tool calls):")
        print(f"   {ai_message.content}")
        print("=" * 60)
        print(f"\n🎯 Conversation complete after {iteration} iteration(s)\n")
        break