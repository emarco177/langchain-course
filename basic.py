from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
load_dotenv()

def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")

    info="""
    Elon Musk is a business magnate and investor. He is the founder, CEO, and chief engineer of SpaceX;
    early-stage investor, CEO, and product architect of Tesla, Inc.; founder of The Boring Company; 
    and co-founder of Neuralink and OpenAI. With an estimated net worth of around $211 billion as of June 2026,
    Musk is the wealthiest person in the world according to the Bloomberg Billionaires Index and Forbes' real-time billionaires list.
    """

    summary_template = """
    You are a helpful assistant that summarizes information.
    Give a summary about this info {info} in following format:
    1) Tell the name of the person
    2) Give 2 fun facts about the person
    """

    summary_prompt = PromptTemplate(
        input_variables=["info"],
        template=summary_template,
    )

    # llm = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     temperature=0,
    # )
    llm=ChatOllama(
        model="gemma3:270m",
        temperature=0,
    )

    chain = summary_prompt | llm
    response = chain.invoke({"info": info})
    print(response.content)


  


if __name__ == "__main__":
    main()
