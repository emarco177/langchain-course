# import asyncio
# import os
# import ssl
# from typing import Any, Dict, List

# import certifi
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap


import os
import ssl
import asyncio

import certifi
from dotenv import load_dotenv

# from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyMap, tavily_crawl, tavily_extract
from langchain_core.documents import Document

from logger  import log_header, log_info, log_error, log_warning, log_success, Colors
load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)

# Chroma = chroma(
#     persist_directory="chroma_db_dir", # os.environ["CHROMA_PERSIST_DIRECTORY"]
#     embedding_function=embeddings
# )
vectorstore = PineconeVectorStore(
    index_name="langchain-docs-2025",  # os.environ["PINECONE_INDEX_NAME"],
    embedding=embeddings,
)
tavily_extract = tavily_extract.TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = tavily_crawl.TavilyCrawl()

async def main():
    """ Main function for the script to ingest data using Tavily """
    URL_DOC = "https://python.langchain.com/"
    log_header("DOC INGESTION PIPELINE")
    log_info(
        f"TavilyCrawl: Crawling {URL_DOC} ...",
        Colors.PURPLE
    )

    res = tavily_crawl.invoke(
        {
            "url": f"{URL_DOC}",
            "max_depth": 1,
            "extract_depth": "advanced",
            # "instructions": "content on ai agents"
        })
    # all_docs = res["results"]
    all_docs = [ Document(
                        page_content=result["raw_content"]
                        , metadata={"source": result["url"]}
                        ) 
                        for result in res["results"]
                    ]
    
    log_success(f"Crawled {len(all_docs)} URLs from the doc-site")
    # await tavily_crawl.run()
    # await tavily_extract.run()
    # await tavily_map.run()


# vectorstore = PineconeVectorStore(
#     index_name=os.environ["PINECONE_INDEX_NAME"],
#     embedding=embeddings
# )

if __name__ == "__main__":
    asyncio.run(main())
