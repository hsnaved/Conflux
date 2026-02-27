from pathlib import Path
from typing import List, Tuple
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    LLM_MODEL_NAME,
    SEARCH_LIMIT,
    OPENAI_API_KEY,
    CONFLUENCE_BASE_URL,
    CONFLUENCE_USERNAME,
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_SPACE_KEY,
)

from services.embedding import embed_text, embed_texts
from services.vectorstore import ensure_collection, search_similar, upsert_chunks
from services.confluence import fetch_confluence_pages


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

_llm = None


def get_llm() -> ChatOpenAI:
    global _llm

    if _llm:
        return _llm

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    _llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    return _llm


def chunk_text(text: str) -> List[str]:
    if not text:
        return []
    return _splitter.split_text(text)


def answer_question(question: str, limit: int = SEARCH_LIMIT) -> Tuple[str, int]:
    if not question:
        return "No question provided.", 0

    query_vector = embed_text(question)

    if not query_vector:
        return "Embedding failed.", 0

    ensure_collection()
    results = search_similar(query_vector, limit)

    context = "\n\n".join(
        res.payload.get("text", "")
        for res in results
        if res.payload and res.payload.get("text")
    )

    if not context:
        return "I don't know.", 0

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.
If answer is not found, say "I don't know".

Context:
{context}

Question:
{question}
"""

    llm = get_llm()
    response = llm.invoke(prompt)

    return response.content, len(results)