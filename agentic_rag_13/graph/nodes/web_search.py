import os
import sys
from typing import Any

from dotenv import load_dotenv

# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

# from agentic_rag_13.graph.state import GraphState
from agentic_rag_13.graph.state import GraphState

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])# state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})
    
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results['results']]
    )
    web_results = Document(page_content=joined_tavily_result)
    # if documents is not None:
    #     documents.append(web_results)
    # else:
    #     documents = [web_results]
    # return {"documents": documents, "question": question}
    return {"documents": documents + [web_results]}


# if __name__ == "__main__":
    # print("cwd:", os.getcwd())
    # print("sys.path:", sys.path)
    # print("hello, web_search")
    # print("/Users/behzad.pirvali/src/udemy/langchain-course" in sys.path)
    # web_search(state={"question": "agent memory", "documents": None})
