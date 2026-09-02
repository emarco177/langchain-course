from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


def main() -> None:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GEMINI_API_KEY")

    information = """
Elon Reeve Musk (born June 28, 1971) is a businessman and former public official who is the CEO and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and briefly became the only trillionaire (in terms of US dollars) in June 2026; as of August 14, 2026, Forbes estimates his net worth to be US$864 billion.

Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

"""

    summary_template = f"""
Given the following information {information} about a person, I want to create:
(1) Short Summary
(2) Two interesting facts about them

Here is the information:
{information}
"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOpenAI(model="gpt-5", temperature=0)
    # llm_google = ChatGoogleGenerativeAI(
    #    model="gemini-3.1-flash-lite", api_key=google_api_key
    # )
    # print("Calling Gemini...")

    llm_ollama = ChatOllama(model="gemma4:latest", temperature=0)

    #chain = summary_prompt_template | llm_google
    chain = summary_prompt_template | llm_ollama
    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
