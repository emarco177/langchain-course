import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from langchain-course!")
    print("Your OpenAI API Key is:", os.environ.get("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
