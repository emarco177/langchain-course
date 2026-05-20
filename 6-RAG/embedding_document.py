# import os

import os
from dotenv import load_dotenv

# from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_classic import hub

load_dotenv()


def main():
    print("Ingesting...")
    hub.
    loader = TextLoader(
        "RAG/data/medium-blog-1.txt"
    )
    doc = loader.load()
    
    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(doc)
    print(f"Split into {len(texts)} chunks")

    print("Embedding...")
    embeddings = OpenAIEmbeddings(
        model= "text-embedding-3-small",
        dimensions= 1024,       
        openai_api_type=os.environ["OPENAI_API_KEY"])


    print("Storing...")
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ["PINECONE_INDEX_NAME"]
    )

    print("Done!")
if __name__ == "__main__":
    main()
