from dotenv import load_dotenv
import os
load_dotenv()

def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    print(openai_api_key)
    print("Hello from langchain-learning!")


if __name__ == "__main__":
    main()
