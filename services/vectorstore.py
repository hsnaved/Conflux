import uuid
from typing import Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from config import COLLECTION_NAME, QDRANT_URL, VECTOR_SIZE

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def ensure_collection() -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )


def upsert_chunks(
    chunks: Iterable[str],
    vectors: Iterable[List[float]],
    source: str,
    doc_id: Optional[str] = None,
) -> int:
    ensure_collection()
    client = get_client()

    doc_id = doc_id or str(uuid.uuid4())
    points = []

    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "source": source,
                    "chunk_index": idx,
                    "doc_id": doc_id,
                },
            )
        )

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    return len(points)


def search_similar(
    vector: List[float],
    limit: int = 5,
    source_filter: Optional[str] = None,
):
    ensure_collection()
    client = get_client()

    query_filter = None

    if source_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source_filter),
                )
            ]
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )

    return response.points


def delete_by_source(source: str) -> None:
    client = get_client()

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source),
                )
            ]
        ),
    )