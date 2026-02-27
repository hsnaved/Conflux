import sys
import traceback
from fastmcp import FastMCP

from services.embedding import embed_text, embed_texts
from services.vectorstore import (
    search_similar,
    ensure_collection,
    upsert_chunks,
)
from services.confluence import fetch_confluence_pages
from services.rag import chunk_text
from config import (
    CONFLUENCE_BASE_URL,
    CONFLUENCE_USERNAME,
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_SPACE_KEY,
    SEARCH_LIMIT,
)

mcp = FastMCP("Conflux RAG Server")


@mcp.tool()
def retrieve_chunks(question: str, limit: int = SEARCH_LIMIT) -> dict:
    try:
        ensure_collection()
        query_vector = embed_text(question)

        if not query_vector:
            return {"status": "error", "message": "Embedding failed"}

        results = search_similar(query_vector, limit)
        chunks = [r.payload.get("text", "") for r in results if r.payload]

        return {"status": "success", "chunks": chunks, "count": len(chunks)}

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {"status": "error", "message": str(e)}


@mcp.tool()
def ingest_confluence_data() -> dict:
    try:
        pages = fetch_confluence_pages(
            CONFLUENCE_BASE_URL,
            CONFLUENCE_USERNAME,
            CONFLUENCE_API_TOKEN,
            CONFLUENCE_SPACE_KEY,
        )

        total_chunks = 0
        ensure_collection()

        for page in pages:
            page_id = page.get("id")
            text = page.get("content", "")

            chunks = chunk_text(text)
            if not chunks:
                continue

            vectors = embed_texts(chunks)

            total_chunks += upsert_chunks(
                chunks,
                vectors,
                source=f"confluence:{page_id}",
                doc_id=page_id,
            )

        return {
            "status": "success",
            "pages": len(pages),
            "total_chunks": total_chunks,
        }

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {"status": "error", "message": str(e)}


@mcp.tool()
def ask_question(question: str, limit: int = SEARCH_LIMIT) -> dict:
    try:
        ensure_collection()
        query_vector = embed_text(question)

        if not query_vector:
            return {"status": "error", "message": "Embedding failed"}

        results = search_similar(query_vector, limit)
        chunks = [r.payload.get("text", "") for r in results if r.payload]

        context = "\n\n".join(chunks)

        return {
            "status": "success",
            "question": question,
            "context": context,
            "chunks_retrieved": len(chunks),
        }

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    mcp.run()