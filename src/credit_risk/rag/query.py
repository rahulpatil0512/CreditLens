"""RAG query: retrieve policy/regulatory evidence for a question or decision.

Rebuilds the SAME embedder that was used at ingestion (recorded in the store
metadata) so query vectors are compatible with stored vectors.
"""
from __future__ import annotations

import atexit
from functools import lru_cache
from typing import Any

from credit_risk.config import Config, get_config
from credit_risk.rag.embeddings import (
    LexicalEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from credit_risk.rag import vector_store as vs


class RagNotIngestedError(RuntimeError):
    pass


def _embedder_from_meta(meta: dict, cfg: Config):
    name = meta.get("embedder", "")
    if name.startswith("st:"):
        model_name = name.split("st:", 1)[1]
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception as e:  # semantic store but ST now unavailable
            raise RagNotIngestedError(
                "Vector store was built with a semantic embedder that is not "
                "available now. Re-run ingestion (scripts/ingest_rag.py)."
            ) from e
    if name.startswith("lexical"):
        dim = int(meta.get("dim", 1024))
        return LexicalEmbedder(n_features=dim)
    return build_embedder(str(cfg.get("rag.embedding_model")))


@lru_cache(maxsize=1)
def _load(cfg_id: int):
    cfg = get_config()
    store_dir = cfg.path("paths.vector_store_dir")
    meta = vs.load_embedder_meta(store_dir)
    if meta is None:
        raise RagNotIngestedError(
            "RAG index not found. Run: python scripts/ingest_rag.py"
        )
    embedder = _embedder_from_meta(meta, cfg)
    collection = str(cfg.get("rag.collection_name", "credit_policy"))
    store = vs.open_store(store_dir, collection, embedder.dim)
    atexit.register(store.close)  # clean shutdown (avoids Qdrant __del__ race)
    return embedder, store, meta


def query(text: str, top_k: int | None = None,
          cfg: Config | None = None) -> dict[str, Any]:
    cfg = cfg or get_config()
    k = int(top_k or cfg.get("rag.top_k", 4))
    embedder, store, meta = _load(0)
    vector = embedder.encode([text])[0]
    hits = store.search(vector, k)
    return {
        "query": text,
        "embedder": meta.get("embedder"),
        "retrieval_mode": "semantic" if str(meta.get("embedder", "")).startswith("st:")
                          else "lexical",
        "results": [
            {
                "rank": i + 1,
                "source": h.get("source"),
                "score": round(float(h.get("score", 0.0)), 4),
                "text": h.get("text"),
            }
            for i, h in enumerate(hits)
        ],
    }


def evidence_for_decision(decision: str, top_factors: list[str] | None = None,
                          cfg: Config | None = None) -> dict[str, Any]:
    """Convenience: build a retrieval query from a decision + its drivers."""
    factors = ", ".join(top_factors or [])
    q = (f"Policy and regulatory basis for a credit {decision.lower()} decision "
         f"and its explanation. Key factors: {factors}. Adverse action reasons, "
         f"affordability, expected loss, model governance.")
    return query(q, cfg=cfg)
