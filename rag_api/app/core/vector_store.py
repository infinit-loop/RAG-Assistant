"""FAISS vector store.

Embeddings: we use the project's TF-IDF vectorizer to turn chunks into dense
float vectors, L2-normalize them, and index with FAISS IndexFlatIP (inner
product on normalized vectors == cosine similarity). This keeps the service
fully offline and lightweight (no torch / no model download) while giving a
real FAISS-backed similarity index. Swapping in sentence-transformer embeddings
later only requires changing the `embed` function — the FAISS layer is unchanged.
"""
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer


class FaissVectorStore:
    def __init__(self):
        self._vectorizer: TfidfVectorizer | None = None
        self._index: faiss.Index | None = None
        self._chunks: list[str] = []
        self._sources: list[str] = []

    def build(self, chunks: list[str], sources: list[str]) -> None:
        self._chunks = chunks
        self._sources = sources
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = self._vectorizer.fit_transform(chunks).toarray().astype("float32")
        faiss.normalize_L2(matrix)
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)

    def _embed(self, text: str) -> np.ndarray:
        vec = self._vectorizer.transform([text]).toarray().astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def search(self, query: str, k: int = 3):
        """Return list of (chunk, source, score) sorted by descending score."""
        if self._index is None:
            return []
        scores, idx = self._index.search(self._embed(query), k)
        out = []
        for score, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            out.append((self._chunks[i], self._sources[i], float(score)))
        return out

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index is not None else 0
