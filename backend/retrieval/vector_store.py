from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.ingestion.parser import Chunk

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = Path(__file__).parent.parent.parent / "data" / "index"


class VectorStore:
    """FAISS dense index + BM25 sparse index over the same chunk list."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.bm25: BM25Okapi | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]

        # Dense embeddings (normalised for inner-product = cosine)
        embeddings = self.embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings.astype(np.float32))

        # BM25 sparse index
        tokenised = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenised)

        self._save()

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def dense_search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.faiss_index.search(q_emb, top_k)
        return [(self.chunks[i], float(scores[0][rank])) for rank, i in enumerate(indices[0]) if i != -1]

    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def hybrid_search(self, query: str, top_k: int = 10,
                      dense_weight: float = 0.6, sparse_weight: float = 0.4) -> List[Tuple[Chunk, float]]:
        """Combine dense + sparse scores via weighted reciprocal rank fusion."""
        dense_results = self.dense_search(query, top_k * 2)
        sparse_results = self.sparse_search(query, top_k * 2)

        # Normalise sparse scores to [0, 1]
        if sparse_results:
            max_s = max(s for _, s in sparse_results) or 1.0
            sparse_results = [(c, s / max_s) for c, s in sparse_results]

        scores: dict[int, float] = {}
        for chunk, score in dense_results:
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + dense_weight * score
        for chunk, score in sparse_results:
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + sparse_weight * score

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        id_to_chunk = {c.chunk_id: c for c, _ in dense_results + sparse_results}
        return [(id_to_chunk[cid], scores[cid]) for cid in sorted_ids if cid in id_to_chunk]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(INDEX_DIR / "faiss.index"))
        with open(INDEX_DIR / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(INDEX_DIR / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls) -> "VectorStore":
        store = cls()
        store.faiss_index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        with open(INDEX_DIR / "bm25.pkl", "rb") as f:
            store.bm25 = pickle.load(f)
        with open(INDEX_DIR / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store

    @classmethod
    def exists(cls) -> bool:
        return (INDEX_DIR / "faiss.index").exists()
