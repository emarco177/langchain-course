from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from pydantic import BaseModel
import datetime

tavily = TavilyClient()
load_dotenv()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

# @tool
def calculate(a: float,b: float, operation: str) -> float:
    """
    Tool that performs basic arithmetic operations
    Args:
        a: The first number
        b: The second number
        operation: The operation to perform (add, subtract, multiply, divide)
    Returns:
        The result of the operation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    else:
        raise ValueError(f"Invalid operation: {operation}")

def date_time(timezone: str) -> str:
    '''Get the current time.'''
@tool
def unit_converter(value: float, conversion: float) -> float:
    '''A tool that converts unit
    value: User input
    Operation: To convert the units (Km_to_miles, miles_to_km, kg_to_pounds, pounds_to_kg, celsius_to_farenheit, farenheit_to_celsius)
    Returns:
        The result of the operation
    '''
    conversions = {
        "km_to_miles": 0.621371,
        "miles_to_km": 1.60934,
        "kg_to_pounds": 2.20462,
        "pounds_to_kg": 0.453592,
    }
    if conversion in conversions:
        return value * conversions[conversion]
    if conversion == "celcius_to_farenheit":
        return value * 9 / 5 + 32;
    if conversion =="farenheit_to_celcius":
        return (value - 32) * 5 / 9
    raise ValueError(f"Invalid Conversion{conversion}")

#Implementing pydantic 
class Conversion_result(BaseModel):
    original_value : float
    converted_value : float
    unit :str

result = Conversion_result(
    original_value = 10.00,
    converted_value = 6.213,
    unit = "miles"
)


# print(result)
# print(result.original_value)
# print(result.converted_value)
# print(result.unit)


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature = 0)
tools = [search, calculate, unit_converter]
agent = create_agent(model=llm,tools=tools)


def main():
    print("Hello from langchain-course!")
    # result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")})
    # result = agent.invoke({"messages":HumanMessage(content="Who is the current CEO of NVIDIA?")})
    # result = agent.invoke({"messages":HumanMessage(content="Who is the founder of Adamas University")})
    # result = agent.invoke({"messages" : HumanMessage(content="Add 2 + 3")})
    # result = agent.invoke({"messages" : HumanMessage(content = "What is the current time in Kolkata according to Time and Date?")})
    # result = agent.invoke({"messages" : HumanMessage(content = "Convert 100 km_to_miles")})
    # result = agent.invoke({"messages" : HumanMessage(content = "Convert 75 kg_to_pounds, Convert 96 celcius_to_farenheit")})
    result = agent.invoke({"messages" : HumanMessage(content = "Convert 10 km to miles")})
    print(result["messages"][-1].content)

structured_llm = llm.with_structured_output(Conversion_result)
result = structured_llm.invoke(
    "Convert 10 kilometers to miles"
)


if __name__ == "__main__":
    main()