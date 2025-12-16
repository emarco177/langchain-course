# RAG Tutorial with LangChain

A step-by-step tutorial demonstrating how to build a Retrieval Augmented Generation (RAG) system using LangChain, OpenAI, and Pinecone.

## Overview

This tutorial progressively builds a complete RAG pipeline:
1. **Document Ingestion** - Load, chunk, embed, and store documents in a vector database
2. **Naive RAG** - Implement a basic retrieval chain using manual function calls
3. **LCEL RAG** - Refactor to use LangChain Expression Language for a cleaner, more powerful approach

## Tutorial Progression

Follow the commits in order to learn RAG concepts incrementally:

| Step | Commit | Description |
|------|--------|-------------|
| 1 | `598dee4` | **Initial Setup** - Project structure with dependencies (LangChain, OpenAI, Pinecone), sample data, and basic configuration |
| 2 | `2e34caf` | **Add Imports** - Import LangChain components for document processing: TextLoader, CharacterTextSplitter, OpenAIEmbeddings, PineconeVectorStore |
| 3 | `9066596` | **Document Ingestion Pipeline** - Complete ingestion: load text documents, split into chunks, generate embeddings, store in Pinecone |
| 4 | `5b0c33f` | **Naive RAG Implementation** - Manual step-by-step retrieval chain demonstrating core RAG concepts without LCEL |
| 5 | `5e4d009` | **LCEL-Based RAG** - Declarative retrieval chain using LangChain Expression Language with streaming, async, and composability |

## Key Components

### Ingestion (`ingestion.py`)
- **TextLoader**: Load documents from text files
- **CharacterTextSplitter**: Split documents into manageable chunks (1000 chars)
- **OpenAIEmbeddings**: Convert text chunks to vector embeddings
- **PineconeVectorStore**: Store and index vectors for similarity search

### Retrieval (`main.py`)
- **Raw LLM**: Direct query to LLM (no context) - baseline comparison
- **Naive RAG**: Manual retrieval → format → prompt → LLM pipeline
- **LCEL RAG**: Declarative chain using `|` operator with built-in streaming/async

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set environment variables:
```bash
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
INDEX_NAME=your_index_name
```

3. Run ingestion to populate the vector store:
```bash
python ingestion.py
```

4. Run the RAG pipeline:
```bash
python main.py
```

## Why LCEL?

The tutorial demonstrates two approaches to building RAG:

| Feature | Naive Approach | LCEL Approach |
|---------|----------------|---------------|
| Code style | Imperative | Declarative |
| Streaming | Manual implementation | Built-in `.stream()` |
| Async | Manual implementation | Built-in `.ainvoke()` |
| Composability | Limited | Pipe operator `\|` |
| Batch processing | Manual loops | Built-in `.batch()` |

## Technologies

- **LangChain** - Framework for building LLM applications
- **OpenAI** - Embeddings and chat completions
- **Pinecone** - Vector database for similarity search
- **Python** - 3.12+
