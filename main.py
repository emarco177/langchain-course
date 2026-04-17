from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Mukesh Dhirubhai Ambani (born 19 April 1957) is an Indian businessman. He is the chairman and managing director of Reliance Industries, the largest public company in India by market capitalisation in 2025. As of December 2025, he is the richest person in Asia and the 16th richest in the world, with a net worth of US$112.8 billion.[5][6] He has attracted fame due to his growth and wealth, and criticism for being a plutocrat,[7] and reports of market manipulation, political corruption, cronyism, and exploitation.[8][9][10][11]
    Born in 1957 to Dhirubhai Ambani, the founder of Reliance Industries, and Kokilaben in Aden, Mukesh Ambani completed his studies at St. Xavier's College, Mumbai and Institute of Chemical Technology. He dropped out of Stanford University in 1980 to join Reliance Industries. He expanded the energy ventures of the company, and directed the set up of its largest petroleum refinery at Jamnagar. He took an increasing role in the company after his father developed health issues in the late 1980s. After the death of his father in 2002, the ownership of the companies in the group was divided between him and his younger brother Anil Ambani.[12] Later, he expanded the portfolio of the group and launched or acquired several ventures in energy, finance, healthcare, media, telecommunications, and sports."""
    summary_template = """
    given the information {information} about a person i want you to create:
    1. A short summary
    2.two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template
    )
    llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
