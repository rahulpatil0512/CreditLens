"""Embedding backends for RAG.

Two implementations behind one interface:

* ``SentenceTransformerEmbedder`` - semantic, 384-dim. Best quality, but the
  first run downloads the model from Hugging Face (needs internet once).
* ``LexicalEmbedder`` - a stateless hashing vectorizer (scikit-learn), fixed
  1024-dim, L2-normalised. Zero download, fully offline, deterministic. Lexical
  (keyword) matching rather than semantic, which is perfectly serviceable for a
  small policy corpus.

``build_embedder`` selects the semantic model when it is importable AND loads
successfully; otherwise it falls back to lexical. The choice (type + dimension)
is recorded with the vector store so ingestion and querying always match.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class LexicalEmbedder:
    """Deterministic, offline hashing embedder (keyword matching)."""

    def __init__(self, n_features: int = 1024):
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.name = f"lexical-hashing-{n_features}"
        self.dim = n_features

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._vec.transform(texts).toarray().astype(np.float32)


class SentenceTransformerEmbedder:
    """Semantic embedder (downloads the model on first use)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        # Prefer the local cache: online loading makes Hugging Face Hub update
        # checks that can stall for a minute on slow or blocked networks. Only
        # go online when the model has never been downloaded.
        try:
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self._model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)


def build_embedder(model_name: str, prefer_semantic: bool = True) -> Embedder:
    """Return the best available embedder, falling back to lexical if needed."""
    if prefer_semantic:
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception:
            # sentence-transformers not installed, or model download failed
            # (e.g. offline). Fall through to the lexical backend.
            pass
    return LexicalEmbedder()
