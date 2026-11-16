from dotenv import load_dotenv 

load_dotenv()

import ollama

from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@traceable(name="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"Looking up price for product: {product}")
    prices = {"laptop": 1000.0, "phone": 500.10, "tablet": 300.40}
    return prices.get(product, 0)

@traceable(name="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price and return the final price.
    Available discount tiers are "bronze", "silver", and "gold".
    """
    print(f"Applying discount: {discount_tier} to price: {price}")
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 50}
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1 - discount / 100), 2)

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'phone', 'tablet'"
                    },
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount to a price and return the final price.",
            "parameters": {
            "type": "object",
            "properties": {
                "price": {
                    "type": "number",
                    "description": "The original price of the product"
                },
                "discount_tier": {
                    "type": "string",
                    "description": "The discount tier, e.g. 'bronze', 'silver', 'gold'"
                }
            },
            "required": ["price", "discount_tier"]
            }
        }
    }
]

@traceable(name="Ollama chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }



    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
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
        }
       ,
       {"role": "user", "content": question}
    ]

    for iteration in range(1,MAX_ITERATIONS + 1):
        print(f"\n---Iteration {iteration}---")

        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        #If no tool calls were made, this is the final answer from the AI
        if not tool_calls:
            print(f"Final answer from AI: {ai_message.content}")
            return {"answer": ai_message.content, "messages": messages + [ai_message]}

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
   
        print(f" [Tool Selected: {tool_name} with args: {tool_args}]")

        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"Tool '{tool_name}' not found.")

        observation = tool_to_use(**tool_args)

        print(f" [Tool Result: {observation}]")

        # Keep only the executed call so every tool_call has a matching ToolMessage;
        # dependent calls issued in parallel (e.g. apply_discount before the price is known) are dropped.
        ai_message.tool_calls = [tool_call]
        messages.append(ai_message)
        messages.append(
           {
               "role": "tool",
               "content": str(observation)
           }
        )

    print("Max iterations reached without a final answer.")
    return {"answer": None, "messages": messages}

if __name__ == "__main__":
    print("Hello LangChain Agent!")
    print()
    result =  run_agent("what is the price of laptop after applying a gold discount?")
    print(result["answer"])