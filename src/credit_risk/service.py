"""Orchestration layer.

Ties the independent components into one decision record:
    predict -> risk -> explain (SHAP) -> counterfactual -> RAG evidence
            -> narrative (template or LLM) -> audit.

Each stage is optional so individual API endpoints can run just the piece they
need, but ``assess`` produces the complete, auditable record required by the
spec (model version, PD, decision, SHAP factors, counterfactuals, risk metrics,
retrieved sources and explanation).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from credit_risk.config import Config, get_config
from credit_risk.model.predict import predict_pd, model_metadata
from credit_risk.risk import engine as risk_engine


def _snippet(text: str, n: int = 320) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rstrip() + "..."


def assess(
    applicant: dict[str, Any],
    cfg: Config | None = None,
    do_explain: bool = True,
    do_counterfactual: bool = True,
    do_rag: bool = True,
    do_llm: bool = True,
    persist: bool = True,
) -> dict[str, Any]:
    cfg = cfg or get_config()

    # 1) PD + risk (always)
    pd = predict_pd(applicant, cfg)
    risk = risk_engine.assess(pd, applicant, cfg)
    decision = risk.decision

    record: dict[str, Any] = {
        "audit_id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_metadata().get("model_version"),
        "data_source": model_metadata().get("data_source"),
        "applicant": applicant,
        "pd": round(pd, 6),
        "decision": decision,
        "thresholds": risk.thresholds.as_dict(),
        "risk": {
            "lgd": risk.lgd,
            "ead": risk.ead,
            "expected_loss": risk.expected_loss,
            "assumptions": risk.assumptions,
        },
    }

    top_factor_names: list[str] = []

    # 2) explainability
    if do_explain:
        from credit_risk.explain.shap_explainer import explain

        exp = explain(applicant, cfg=cfg)
        record["top_factors"] = exp["top_factors"]
        record["adverse_action_reasons"] = exp["adverse_action_reasons"]
        record["explanation_meta"] = {
            "backend": exp["backend"], "note": exp["note"],
            "base_value_log_odds": exp["base_value_log_odds"],
        }
        top_factor_names = [f["feature"] for f in exp["top_factors"][:4]]

    # 3) counterfactuals (only when not already at target)
    if do_counterfactual and decision != risk_engine.APPROVE:
        from credit_risk.counterfactual.engine import generate

        record["counterfactuals"] = generate(applicant, cfg=cfg)

    # 4) RAG evidence
    if do_rag:
        try:
            from credit_risk.rag.query import evidence_for_decision

            ev = evidence_for_decision(decision, top_factor_names, cfg=cfg)
            record["rag_evidence"] = [
                {"source": r["source"], "score": r["score"],
                 "snippet": _snippet(r["text"])}
                for r in ev["results"]
            ]
            record["rag_meta"] = {"retrieval_mode": ev["retrieval_mode"],
                                  "embedder": ev["embedder"]}
        except Exception as e:
            record["rag_evidence"] = []
            record["rag_meta"] = {"error": f"{type(e).__name__}: {e}"}

    # 5) narrative (template or LLM)
    if do_llm:
        from credit_risk.llm.explain import generate_explanation

        facts = {
            "decision": decision,
            "pd": pd,
            "risk": record["risk"],
            "top_factors": record.get("top_factors", []),
            "adverse_action_reasons": record.get("adverse_action_reasons", []),
            "counterfactuals": record.get("counterfactuals"),
            "rag_evidence": record.get("rag_evidence", []),
        }
        record["narrative"] = generate_explanation(facts, cfg)

    # 6) audit
    if persist:
        from credit_risk.audit import store

        store.save(record, cfg)
    return record
