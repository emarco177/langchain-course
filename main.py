import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from langchain-course!")
    print(os.getenv("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
