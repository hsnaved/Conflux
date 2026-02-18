import uuid
from typing import Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, ScoredPoint, VectorParams

from config import COLLECTION_NAME, QDRANT_URL, VECTOR_SIZE


_client = QdrantClient(url=QDRANT_URL)


def ensure_collection() -> None:
    existing = {collection.name for collection in _client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
         )


def upsert_chunks(chunks: Iterable[str], vectors: Iterable[List[float]], source: str) -> int:
    points: List[PointStruct] = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk, "source": source},
            )
        )

    if points:
        _client.upsert(collection_name=COLLECTION_NAME, points=points)

    return len(points)


def search_similar(vector: List[float], limit: int) -> List[ScoredPoint]:
    # ensure_collection()
    # return _client.search(collection_name=COLLECTION_NAME, query_vector=vector, limit=limit)
    response = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=limit,
        score_threshold=0.5
    )

    return response.points
