import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from langchain-course!")
    print("Environment variable EXAMPLE_VAR:", os.getenv("EXAMPLE_VAR"))


if __name__ == "__main__":
    main()
