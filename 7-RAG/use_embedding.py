import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

load_dotenv()
print("Initializing components...")
embeddings = OpenAIEmbeddings(
    model= "text-embedding-3-small",
    dimensions= 1024,       
    openai_api_type=os.environ["OPENAI_API_KEY"]
)



print("Initializing vector store...")
vector_store = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX_NAME"],
    embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:"""
)
def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

print("Initializing LLM...")
llm = ChatOpenAI(
    model="gpt-5.2",
)

def retrieval_chain_without_lcel(query):
    docs = retriever.invoke(query)
    context = format_docs(docs)

    prompt_template.format_messages(
        context=context,
        question=query
    )

    docs = vector_store.similarity_search(query, k=3)
    context = format_docs(docs)
    msgs =prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(msgs)

    return response.content

def create_retrieval_chain_with_lcel():
    """Create a retrieval chain with LCEL.
    
    Returns:
        RetrievalChain: A retrieval chain that can be invoked with a query.
    """
    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") |retriever | format_docs
        )
        | prompt_template 
        | llm 
        | StrOutputParser()

    )
    return chain
    # docs = retriever.invoke(query)
    # context = format_docs(docs)

    # prompt_template.format_messages(
    #     context=context,
    #     question=query
    # )

    # docs = vector_store.similarity_search(query, k=3)
    # context = format_docs(docs)
    # msgs =prompt_template.format_messages(context=context, question=query)
    # response = llm.invoke(msgs)

    # return response.content

if __name__ == "__main__":
    print("Starting conversation...")
    query = "What is pinecone in machine learning?"
    print(f"Query: {query}\n")

    # -------------------------------
    # Option 0: without RAG
    # -------------------------------
    print("\n" + "-" * 70)
    print("Option 0: without RAG...")

    # result_0 = llm.invoke(
    #     [HumanMessage(
    #         content=query
    #     )]
    # )
    # print(f"Answer:\n{result_0.content}")   
    print("\n" + "-" * 70)
    # -------------------------------
    # Option 1: with RAG
    # -------------------------------
    
    print("Option 1: with RAG without LCEL...")
    # res_without_lcel = retrieval_chain_without_lcel(query)
    # print(f"Answer:\n{res_without_lcel}")
    print("\n" + "-" * 70)
    # -------------------------------
    # Option 2: with RAG+LCEL
    # -------------------------------
    chain = create_retrieval_chain_with_lcel()

    print("Option 2: with RAG+LCEL...")
    res_with_lcel = chain.invoke({"question": query})
    print(f"Answer:\n{res_with_lcel}")

