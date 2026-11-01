"""Vector store: Qdrant in embedded/local mode (no Docker, no server).

If ``qdrant-client`` is unavailable for any reason, a tiny numpy cosine store is
used instead so RAG still works. Both persist to ``artifacts/vector_store`` and
record the embedder (type + dimension) so queries never mix incompatible
vectors.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_META_FILE = "embedder_meta.json"


class _QdrantStore:
    def __init__(self, path: Path, collection: str, dim: int):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.collection = collection
        self.client = QdrantClient(path=str(path / "qdrant"))
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def reset(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def add(self, vectors: np.ndarray, payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=i, vector=vectors[i].tolist(), payload=payloads[i])
            for i in range(len(payloads))
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, vector: np.ndarray, top_k: int) -> list[dict]:
        hits = self.client.query_points(
            self.collection, query=vector.tolist(), limit=top_k, with_payload=True
        ).points
        return [{"score": float(h.score), **h.payload} for h in hits]

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


class _NumpyStore:
    """Fallback cosine store persisted via .npz + .json."""

    def __init__(self, path: Path, collection: str, dim: int):
        self.path = path
        self.collection = collection
        self.vec_file = path / "vectors.npz"
        self.pay_file = path / "payloads.json"
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._payloads: list[dict] = []
        if self.vec_file.exists() and self.pay_file.exists():
            self._vectors = np.load(self.vec_file)["v"]
            self._payloads = json.loads(self.pay_file.read_text())

    def reset(self, dim: int) -> None:
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._payloads = []

    def add(self, vectors: np.ndarray, payloads: list[dict]) -> None:
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        self._payloads.extend(payloads)
        np.savez_compressed(self.vec_file, v=self._vectors)
        self.pay_file.write_text(json.dumps(self._payloads))

    def search(self, vector: np.ndarray, top_k: int) -> list[dict]:
        if len(self._payloads) == 0:
            return []
        v = vector.astype(np.float32)
        # vectors are L2-normalised by the embedders -> dot product == cosine.
        sims = self._vectors @ v
        idx = np.argsort(-sims)[:top_k]
        return [{"score": float(sims[i]), **self._payloads[i]} for i in idx]

    def close(self) -> None:
        pass


def open_store(path: Path, collection: str, dim: int):
    """Open Qdrant, falling back to the numpy mirror.

    Embedded Qdrant allows one process per storage folder, so while the API
    server holds it, other processes (tests, demo script) read the mirror that
    ingestion writes alongside it.
    """
    path.mkdir(parents=True, exist_ok=True)
    try:
        return _QdrantStore(path, collection, dim)
    except Exception:
        return _NumpyStore(path, collection, dim)


def write_mirror(path: Path, collection: str, vectors: np.ndarray,
                 payloads: list[dict]) -> None:
    """Persist the index as a numpy store too (the fallback for a locked Qdrant)."""
    mirror = _NumpyStore(path, collection, vectors.shape[1])
    mirror.reset(vectors.shape[1])
    mirror.add(vectors, payloads)


def save_embedder_meta(path: Path, embedder_name: str, dim: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / _META_FILE).write_text(
        json.dumps({"embedder": embedder_name, "dim": dim}, indent=2)
    )


def load_embedder_meta(path: Path) -> dict[str, Any] | None:
    f = path / _META_FILE
    return json.loads(f.read_text()) if f.exists() else None
