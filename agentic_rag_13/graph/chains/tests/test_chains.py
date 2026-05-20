# def test_foo()->None:
#     print("\nrunning test_foo()...")
#     assert True
from pprint import pprint

from dotenv import load_dotenv

load_dotenv()


from agentic_rag_13.graph.chains import hallucination_grader
from agentic_rag_13.graph.chains.generation import generation_chain
from agentic_rag_13.graph.chains.retrieval_grader import (
    GradeDocuments,
    retrieval_grader,
)
from agentic_rag_13.graph.chains.router import RouteQuery, question_router
from agentic_rag_13.ingestion import retriever


def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})
    pprint(generation)
    # doc_txt = docs[1].page_content

    # res: GradeDocuments = retrieval_grader.invoke(
    #     {"question": question, "document": doc_txt}
    # )

    # assert res.binary_score == "yes"


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: hallucination_grader.GradeHallucinations = (
        hallucination_grader.hallucin_grader.invoke(
            {"documents": docs, "generation": generation}
        )
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: hallucination_grader.GradeHallucinations = hallucination_grader.hallucin_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"
