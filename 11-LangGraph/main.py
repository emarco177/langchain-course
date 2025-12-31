from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    print("Hello from ReAct LangGraph!")
    print(f"OPENAI_API_KEY: '{os.getenv('OPENAI_API_KEY')}'")