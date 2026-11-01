"""FastAPI application exposing the credit decision-support backend.

Endpoints
---------
GET  /health              liveness + model/RAG availability
GET  /meta/options        category values the model knows (for UI dropdowns)
GET  /meta/demo-applicants  bundled demo applicants
GET  /model/info          model card: metrics, thresholds, importance, fairness
GET  /model/calibration-plot  calibration curve PNG
POST /predict             full assessment (PD, decision, SHAP, CF, RAG, narrative) + audit
POST /explain             SHAP reason codes (applicant or audit_id)
POST /counterfactual      actionable counterfactuals (applicant or audit_id)
POST /stress-test         scenario sensitivity (applicant and/or portfolio)
POST /rag/query           retrieve policy/regulatory evidence
GET  /audit               recent decisions + decision counts
GET  /audit/{id}          fetch a stored decision record
GET  /                    web UI (static files in ``frontend/``)

CORS is enabled so an external frontend can call the API directly.
"""
from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from credit_risk.config import get_config
from credit_risk.api import schemas as s

cfg = get_config()
FRONTEND_DIR = cfg.root / "frontend"


def _warm_up() -> None:
    """Load model, SHAP explainer and RAG embedder so first requests are fast."""
    try:
        from credit_risk.model.predict import get_model_and_meta
        from credit_risk.explain.shap_explainer import _load_explainer

        get_model_and_meta()
        _load_explainer()
    except Exception:
        pass  # surfaced via /health and per-request errors instead
    try:
        from credit_risk.rag.query import _load as load_rag

        load_rag(0)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title=str(cfg.get("api.title", "Credit Decision API")),
              version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(cfg.get("api.cors_origins", ["*"])),
    allow_credentials=False,  # no cookies are used; "*" + credentials is invalid CORS
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_applicant(applicant, audit_id) -> dict[str, Any]:
    """Return a base-field applicant dict from either a payload or an audit_id."""
    if applicant is not None:
        return applicant.to_dict()
    if audit_id:
        from credit_risk.audit import store

        rec = store.get(audit_id)
        if not rec:
            raise HTTPException(status_code=404, detail=f"audit_id '{audit_id}' not found")
        return rec["applicant"]
    raise HTTPException(status_code=422, detail="Provide either 'applicant' or 'audit_id'.")


@app.get("/health")
def health() -> dict[str, Any]:
    out: dict[str, Any] = {"status": "ok"}
    try:
        from credit_risk.model.predict import model_metadata

        meta = model_metadata()
        out["model"] = {
            "available": True,
            "model_version": meta.get("model_version"),
            "data_source": meta.get("data_source"),
            "test_auc": meta.get("metrics", {}).get("test_auc"),
        }
    except Exception as e:
        out["model"] = {"available": False, "error": f"{type(e).__name__}: {e}"}
    try:
        from credit_risk.rag import vector_store as vs

        meta = vs.load_embedder_meta(cfg.path("paths.vector_store_dir"))
        out["rag"] = {"available": meta is not None, "embedder": (meta or {}).get("embedder")}
    except Exception:
        out["rag"] = {"available": False}
    out["llm"] = {"enabled": bool(cfg.get("llm.enabled")), "available": cfg.llm_available,
                  "provider": cfg.llm_provider}
    return out


@app.get("/meta/options")
def meta_options() -> dict[str, Any]:
    from credit_risk.model.predict import category_options, ModelNotTrainedError

    try:
        return {"categorical": category_options()}
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/meta/demo-applicants")
def demo_applicants() -> list[dict[str, Any]]:
    path = cfg.root / "data" / "demo_applicants.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    from credit_risk.model.predict import (
        ModelNotTrainedError, global_importance, model_metadata,
    )
    from credit_risk.risk.engine import compute_thresholds

    try:
        meta = dict(model_metadata())
        meta["global_importance"] = global_importance()
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    meta["thresholds"] = compute_thresholds(cfg).as_dict()
    meta["risk_assumptions"] = {
        "lgd": cfg.get("risk.lgd_assumption"),
        "ead_source": cfg.get("risk.ead_source"),
        "annual_margin": cfg.get("risk.annual_margin_assumption"),
    }
    meta["scenarios"] = {
        k: v.get("description", "") for k, v in dict(cfg.get("stress.scenarios", {})).items()
    }
    meta["calibration_plot_available"] = cfg.path("paths.calibration_plot").exists()
    return meta


@app.get("/model/calibration-plot")
def calibration_plot() -> FileResponse:
    path = cfg.path("paths.calibration_plot")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Calibration plot not found; train first.")
    return FileResponse(path, media_type="image/png")


@app.post("/predict")
def predict(req: s.PredictRequest) -> dict[str, Any]:
    from credit_risk.service import assess
    from credit_risk.model.predict import ModelNotTrainedError

    try:
        return assess(
            req.applicant.to_dict(),
            do_explain=req.include_explanation,
            do_counterfactual=req.include_counterfactual,
            do_rag=req.include_rag,
            do_llm=req.include_llm,
            persist=req.persist,
        )
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/explain")
def explain_endpoint(req: s.ApplicantOrAudit) -> dict[str, Any]:
    from credit_risk.explain.shap_explainer import explain
    from credit_risk.model.predict import ModelNotTrainedError

    applicant = _resolve_applicant(req.applicant, req.audit_id)
    try:
        return explain(applicant)
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/counterfactual")
def counterfactual_endpoint(req: s.ApplicantOrAudit) -> dict[str, Any]:
    from credit_risk.counterfactual.engine import generate
    from credit_risk.model.predict import ModelNotTrainedError

    applicant = _resolve_applicant(req.applicant, req.audit_id)
    try:
        return generate(applicant)
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/stress-test")
def stress_endpoint(req: s.StressRequest) -> dict[str, Any]:
    from credit_risk.stress.engine import (
        stress_applicant, stress_portfolio, load_portfolio_sample,
    )
    from credit_risk.model.predict import ModelNotTrainedError

    out: dict[str, Any] = {}
    try:
        if req.applicant is not None or req.audit_id:
            applicant = _resolve_applicant(req.applicant, req.audit_id)
            out["applicant"] = stress_applicant(applicant)
        if req.portfolio_sample and req.portfolio_sample > 0:
            sample = load_portfolio_sample(req.portfolio_sample)
            out["portfolio"] = stress_portfolio(sample)
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if not out:
        raise HTTPException(status_code=422,
                            detail="Provide an applicant/audit_id and/or portfolio_sample>0.")
    return out


@app.post("/rag/query")
def rag_query_endpoint(req: s.RagQueryRequest) -> dict[str, Any]:
    from credit_risk.rag.query import query, RagNotIngestedError

    try:
        return query(req.query, top_k=req.top_k)
    except RagNotIngestedError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/audit")
def list_audit(limit: int = Query(25, ge=1, le=200)) -> dict[str, Any]:
    from credit_risk.audit import store

    return {"stats": store.stats(), "records": store.list_recent(limit)}


@app.get("/audit/{audit_id}")
def get_audit(audit_id: str) -> dict[str, Any]:
    from credit_risk.audit import store

    rec = store.get(audit_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"audit_id '{audit_id}' not found")
    return rec


# Web UI. Mounted last so every API route above takes precedence.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
