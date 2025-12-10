# import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    # print(os.getenv("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
