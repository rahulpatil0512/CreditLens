"""RAG ingestion: read policy documents -> chunk -> embed -> store.

Reads every ``.md``, ``.txt`` and ``.pdf`` in ``docs/policy/``. PDFs require
``pypdf`` (optional); if it is missing, PDFs are skipped with a warning and the
text files are still ingested.
"""
from __future__ import annotations

from pathlib import Path

from credit_risk.config import Config, get_config
from credit_risk.rag.embeddings import build_embedder
from credit_risk.rag import vector_store as vs


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        print(f"[ingest] SKIP {path.name}: install 'pypdf' to ingest PDFs.")
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:  # pragma: no cover
        print(f"[ingest] WARN could not read {path.name}: {e}")
        return ""


def _read_doc(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    """Sliding-window character chunks that prefer to break on a newline."""
    text = text.strip()
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            nl = text.rfind("\n", start + int(size * 0.5), end)
            if nl != -1:
                end = nl
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def ingest(cfg: Config | None = None) -> dict:
    cfg = cfg or get_config()
    docs_dir = cfg.path("paths.policy_docs_dir")
    store_dir = cfg.path("paths.vector_store_dir")
    collection = str(cfg.get("rag.collection_name", "credit_policy"))
    size = int(cfg.get("rag.chunk_size", 900))
    overlap = int(cfg.get("rag.chunk_overlap", 150))
    prefer_semantic = bool(cfg.get("rag.prefer_semantic", True))
    model_name = str(cfg.get("rag.embedding_model"))

    files = sorted(
        p for p in docs_dir.glob("*")
        if p.suffix.lower() in {".md", ".txt", ".pdf"} and p.name != "SOURCES.md"
    )
    if not files:
        raise FileNotFoundError(f"No policy documents found in {docs_dir}.")

    texts, payloads = [], []
    for f in files:
        content = _read_doc(f)
        for i, ch in enumerate(_chunk(content, size, overlap)):
            texts.append(ch)
            payloads.append({"text": ch, "source": f.name, "chunk_index": i})

    if not texts:
        raise ValueError("No text chunks produced from policy documents.")

    embedder = build_embedder(model_name, prefer_semantic=prefer_semantic)
    vectors = embedder.encode(texts)

    store = vs.open_store(store_dir, collection, embedder.dim)
    store.reset(embedder.dim)
    store.add(vectors, payloads)
    store.close()
    vs.write_mirror(store_dir, collection, vectors, payloads)
    vs.save_embedder_meta(store_dir, embedder.name, embedder.dim)

    return {
        "documents": [f.name for f in files],
        "n_chunks": len(texts),
        "embedder": embedder.name,
        "embedding_dim": embedder.dim,
        "store_dir": str(store_dir),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(ingest(), indent=2))
