from functools import lru_cache
from typing import Iterable, List

from sentence_transformers import SentenceTransformer
import torch

from config import MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(MODEL_NAME, device=device)


def embed_text(text: str) -> List[float]:
    if not text or not isinstance(text, str):
        return []

    model = get_embedding_model()

    vector = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return vector.tolist()


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    text_list = [t for t in texts if isinstance(t, str) and t.strip()]

    if not text_list:
        return []

    model = get_embedding_model()

    vectors = model.encode(
        text_list,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return [vec.tolist() for vec in vectors]