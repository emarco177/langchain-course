from typing import Any

from agentic_rag_13.graph.chains.generation import generation_chain
from agentic_rag_13.graph.state import GraphState


def generate(state: GraphState) -> dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"question": question, "context": documents})
    # return {"question": question, "documents": documents, "generation": generation}
    return {"generation": generation}
