#!/usr/bin/env python
"""One-command launcher: build anything missing, start the API + web UI.

Runs data prep / training / RAG ingestion only if their artifacts are absent,
then serves the FastAPI backend and the web UI on the same port and opens the
browser.

Usage:
    python scripts/serve.py              # http://127.0.0.1:8000
    python scripts/serve.py --port 8080 --no-browser
    python scripts/serve.py --rebuild    # force prepare + train + ingest
"""
import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.config import get_config  # noqa: E402


def ensure_artifacts(rebuild: bool) -> None:
    cfg = get_config()
    model = cfg.path("paths.artifacts_dir") / "model_calibrated.joblib"
    test_split = cfg.path("paths.processed_dir") / "test.parquet"
    rag_meta = cfg.path("paths.vector_store_dir") / "embedder_meta.json"

    if rebuild or not test_split.exists():
        from credit_risk.data.prepare import prepare

        print("[serve] preparing data ...")
        p = prepare(write=True)
        print(f"[serve]   source={p.source} train={len(p.train)} test={len(p.test)}")
    if rebuild or not model.exists():
        from credit_risk.model.train import train

        print("[serve] training model (a few minutes on the full dataset) ...")
        m = train()
        print(f"[serve]   test AUC={m['metrics']['test_auc']}")
    if rebuild or not rag_meta.exists():
        from credit_risk.rag.ingest import ingest

        print("[serve] indexing policy documents ...")
        r = ingest()
        print(f"[serve]   {r['n_chunks']} chunks, embedder={r['embedder']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-run data prep, training and RAG ingestion")
    args = ap.parse_args()

    ensure_artifacts(args.rebuild)

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"[serve] UI  -> {url}")
    print(f"[serve] API -> {url}/docs")
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(2), webbrowser.open(url)),
                         daemon=True).start()
    uvicorn.run("credit_risk.api.main:app", host=args.host, port=args.port,
                app_dir=str(ROOT / "src"), log_level="info")


if __name__ == "__main__":
    main()
