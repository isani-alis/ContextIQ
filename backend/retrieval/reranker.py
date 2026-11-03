from __future__ import annotations

from typing import List, Tuple

from sentence_transformers import CrossEncoder

from backend.ingestion.parser import Chunk

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CONFIDENCE_THRESHOLD = -15.0  # cross-encoder ms-marco scores typically range -15 to +10

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank(query: str, candidates: List[Tuple[Chunk, float]],
           top_k: int = 5) -> List[Tuple[Chunk, float]]:
    """Re-score candidates with a cross-encoder, return top_k sorted by score."""
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, chunk.text) for chunk, _ in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip([c for c, _ in candidates], scores),
                    key=lambda x: x[1], reverse=True)
    return [(chunk, float(score)) for chunk, score in ranked[:top_k]]


def is_confident(results: List[Tuple[Chunk, float]]) -> bool:
    """True if the top reranked result clears the confidence threshold."""
    if not results:
        return False
    return results[0][1] >= CONFIDENCE_THRESHOLD
