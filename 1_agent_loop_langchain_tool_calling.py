from dotenv import load_dotenv 

load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langsmith import traceable
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"Looking up price for product: {product}")
    prices = {"laptop": 1000.0, "phone": 500.10, "tablet": 300.40}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price and return the final price.
    Available discount tiers are "bronze", "silver", and "gold".
    """
    print(f"Applying discount: {discount_tier} to price: {price}")
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 50}
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1 - discount / 100), 2)

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature = 0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. Call only ONE tool per turn. Wait for its result before "
                "calling the next tool.\n"
                "5. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1,MAX_ITERATIONS + 1):
        print(f"\n---Iteration {iteration}---")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        #If no tool calls were made, this is the final answer from the AI
        if not tool_calls:
            print(f"Final answer from AI: {ai_message.content}")
            return {"answer": ai_message.content, "messages": messages + [ai_message]}

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" [Tool Selected: {tool_name} with args: {tool_args}]")

        tools_to_use = tools_dict.get(tool_name)
        if not tools_to_use:
            raise ValueError(f"Tool '{tool_name}' not found.")

        try:
            observation = tools_to_use.invoke(tool_args)
        except Exception as e:
            # Let the model self-correct (e.g. it passed a placeholder instead of a real price)
            observation = (
                f"Error calling {tool_name}: {e}. Call tools one at a time and "
                "pass real values returned by previous tool calls."
            )

        print(f" [Tool Result: {observation}]")

        # Keep only the executed call so every tool_call has a matching ToolMessage;
        # dependent calls issued in parallel (e.g. apply_discount before the price is known) are dropped.
        ai_message.tool_calls = [tool_call]
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("Max iterations reached without a final answer.")
    return {"answer": None, "messages": messages}

if __name__ == "__main__":
    print("Hello LangChain Agent!")
    print()
    result =  run_agent("what is the price of laptop after applying a gold discount?")
    print(result["answer"])