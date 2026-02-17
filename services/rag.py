from pathlib import Path
from typing import List

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    LLM_MODEL_NAME,
    SEARCH_LIMIT,
    CONFLUENCE_BASE_URL,
    CONFLUENCE_USERNAME,
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_SPACE_KEY
)
from services.embedding import embed_text, embed_texts
from services.vectorstore import ensure_collection, search_similar, upsert_chunks
from services.confluence import fetch_confluence_pages

# ensure OpenAI key exists
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY environment variable is required for ChatOpenAI")

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0)

def extract_text_from_pdf(path: Path) -> str:
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    return "\n".join(doc.page_content for doc in documents)

def chunk_text(text: str) -> List[str]:
    return _splitter.split_text(text)

def ingest_pdf(path: Path, source_name: str) -> int:
    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    ensure_collection()
    return upsert_chunks(chunks, vectors, source=source_name)

def ingest_confluence() -> int:
    pages = fetch_confluence_pages(
        CONFLUENCE_BASE_URL,
        CONFLUENCE_USERNAME,
        CONFLUENCE_API_TOKEN,
        CONFLUENCE_SPACE_KEY
    )
    total_chunks = 0
    ensure_collection()
    for page in pages:
        text = page['content']
        chunks = chunk_text(text)
        vectors = embed_texts(chunks)
        chunks_added = upsert_chunks(chunks, vectors, source="confluence")
        total_chunks += chunks_added
    return total_chunks

def build_context(results) -> str:
    return "\n\n".join([res.payload.get("text", "") for res in results if res.payload])

def build_prompt(context: str, question: str) -> str:
    return f"""
    You are a helpful assistant.

    Answer the question ONLY using the provided context.
    If the answer is not found in the context, say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """

def answer_question(question: str, limit: int = SEARCH_LIMIT) -> tuple[str, int]:
    query_vector = embed_text(question)
    ensure_collection()
    results = search_similar(query_vector, limit)
    context = build_context(results)
    prompt = build_prompt(context, question)
    response = llm.invoke(prompt)
    return response.content, len(results)
