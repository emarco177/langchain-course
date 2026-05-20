import typing

from ingestion import retriever

from graph.state import GraphState


def retrieve(state: GraphState) -> dict[str, typing.Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    # return {"documents": documents, "question": question}
    return {"documents": documents}