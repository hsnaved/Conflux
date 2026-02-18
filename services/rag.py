from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from config import CHUNK_OVERLAP, CHUNK_SIZE, LLM_MODEL_NAME, SEARCH_LIMIT, logger
from services.embedding import embed_text, embed_texts
from services.vectorstore import ensure_collection, search_similar, upsert_chunks

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
# llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0)


def extract_text_from_pdf(path: Path) -> str:
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    return "\n".join(doc.page_content for doc in documents)


def chunk_text(text: str) -> List[str]:
    return _splitter.split_text(text)


def ingest_pdf(path: Path, source_name: str) -> int:
    text = extract_text_from_pdf(path)
    logger.info("Extracted %d characters from PDF", len(text))
    chunks = chunk_text(text)
    logger.info("Split text into %d chunks", len(chunks))
    vectors = embed_texts(chunks)
    ensure_collection()
    return upsert_chunks(chunks, vectors, source=source_name)


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
    logger.info("Answering question: %s (limit=%d)", question, limit)
    query_vector = embed_text(question)
    logger.info(query_vector)
    ensure_collection()
    results = search_similar(query_vector, limit)
    # Extract text chunks
    contexts = [point.payload["text"] for point in results]

    # Join them
    context_text = "\n".join(contexts)
    # context = build_context(results)
    # prompt = build_prompt(context, question)
    # response = llm.invoke(prompt)
    # return response.content, len(results)
    return context_text, len(results)
