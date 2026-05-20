from dotenv import load_dotenv

load_dotenv()

# from graph.graph import app
from agentic_rag_13.graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "what is agent memory?"}))
    # print(app.invoke(input={"question": "how to make pizaa?"}))
