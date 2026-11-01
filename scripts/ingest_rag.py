#!/usr/bin/env python
"""Entry point: ingest policy/regulatory documents into the RAG vector store.

Reads every .md/.txt/.pdf in docs/policy/ (drop real EBA/ECB/BaFin PDFs there
first for proper grounding; see docs/policy/SOURCES.md).

Usage:
    python scripts/ingest_rag.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.rag.ingest import ingest  # noqa: E402


def main() -> None:
    info = ingest()
    print(f"[ingest] documents : {', '.join(info['documents'])}")
    print(f"[ingest] chunks    : {info['n_chunks']}")
    print(f"[ingest] embedder  : {info['embedder']} (dim {info['embedding_dim']})")
    if info["embedder"].startswith("lexical"):
        print("[ingest] NOTE: using the offline LEXICAL embedder. For semantic "
              "retrieval, `pip install sentence-transformers` (downloads a model "
              "once) and re-run.")
    print(f"[ingest] stored in : {info['store_dir']}")


if __name__ == "__main__":
    main()
