"""Integration tests: model + explainability + CF + stress + RAG + API + audit.

A session fixture builds the artifacts (data prep, training, RAG ingestion) once
if they are not already present, so a fresh clone can run ``pytest`` directly.
These are slower than test_core.py because they exercise the real model.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.config import get_config  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ensure_setup():
    cfg = get_config()
    art = cfg.path("paths.artifacts_dir")
    if not (art / "model_calibrated.joblib").exists():
        from credit_risk.data.prepare import prepare
        from credit_risk.model.train import train
        prepare()
        train()
    if not (cfg.path("paths.vector_store_dir") / "embedder_meta.json").exists():
        from credit_risk.rag.ingest import ingest
        ingest()
    yield


@pytest.fixture(scope="session")
def demo_applicants():
    return json.loads((ROOT / "data" / "demo_applicants.json").read_text())


# --- prediction + calibration sanity ------------------------------------
def test_pd_in_unit_interval(demo_applicants):
    from credit_risk.model.predict import predict_pd
    for a in demo_applicants:
        p = predict_pd(a)
        assert 0.0 < p < 1.0


def test_higher_risk_applicant_has_higher_pd(demo_applicants):
    from credit_risk.model.predict import predict_pd
    low = predict_pd(demo_applicants[0])
    high = predict_pd(demo_applicants[2])
    assert high > low


# --- explainability ------------------------------------------------------
def test_explain_returns_ranked_factors(demo_applicants):
    from credit_risk.explain.shap_explainer import explain
    r = explain(demo_applicants[2])
    assert len(r["top_factors"]) >= 3
    contribs = [abs(f["contribution"]) for f in r["top_factors"]]
    assert contribs == sorted(contribs, reverse=True)  # ranked by |contribution|


# --- counterfactuals -----------------------------------------------------
def test_counterfactual_feasible_lowers_pd(demo_applicants):
    from credit_risk.counterfactual.engine import generate
    r = generate(demo_applicants[3])  # borderline Review -> should be fixable
    assert r["baseline_decision"] == "Review"
    assert r["feasible"] is True and r["counterfactuals"]
    for cf in r["counterfactuals"]:
        assert cf["resulting_pd"] < r["baseline_pd"]


def test_counterfactual_only_changes_actionable_features(demo_applicants):
    from credit_risk.counterfactual.engine import generate
    cfg = get_config()
    allowed = set(cfg.get("counterfactual.actionable_features").keys())
    r = generate(demo_applicants[3])
    assert r["counterfactuals"]
    for cf in r["counterfactuals"]:
        for change in cf["changes"]:
            assert change["feature"] in allowed


# --- stress --------------------------------------------------------------
def test_stress_unemployment_raises_pd(demo_applicants):
    from credit_risk.stress.engine import stress_applicant
    r = stress_applicant(demo_applicants[1])
    scen = {s["scenario"]: s for s in r["scenarios"]}
    assert scen["unemployment_shock"]["pd_delta"] > 0


# --- RAG -----------------------------------------------------------------
def test_rag_returns_results():
    from credit_risk.rag.query import query
    r = query("adverse action reasons and affordability", top_k=3)
    assert len(r["results"]) >= 1
    assert all("source" in x for x in r["results"])


# --- service + audit -----------------------------------------------------
def test_service_builds_full_record_and_audits(demo_applicants):
    from credit_risk.service import assess
    from credit_risk.audit import store
    rec = assess(demo_applicants[2])
    for key in ("pd", "decision", "risk", "top_factors", "rag_evidence", "narrative"):
        assert key in rec
    assert store.get(rec["audit_id"]) is not None


def test_llm_falls_back_to_template_without_key(demo_applicants):
    # No API key in test env -> narrative must be the deterministic template.
    from credit_risk.service import assess
    rec = assess(demo_applicants[0])
    assert rec["narrative"]["mode"] == "template"


# --- API -----------------------------------------------------------------
@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from credit_risk.api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model"]["available"] is True


def test_predict_endpoint(client, demo_applicants):
    r = client.post("/predict", json={"applicant": demo_applicants[2]})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in ("Approve", "Review", "Reject")
    assert "audit_id" in body


def test_audit_roundtrip_via_api(client, demo_applicants):
    aid = client.post("/predict", json={"applicant": demo_applicants[1]}).json()["audit_id"]
    r = client.get(f"/audit/{aid}")
    assert r.status_code == 200 and r.json()["audit_id"] == aid


def test_invalid_applicant_rejected(client, demo_applicants):
    bad = {**demo_applicants[1], "AMT_INCOME_TOTAL": -1}
    assert client.post("/predict", json={"applicant": bad}).status_code == 422


def test_missing_audit_returns_404(client):
    assert client.get("/audit/nope").status_code == 404


def test_meta_options_lists_model_categories(client):
    r = client.get("/meta/options")
    assert r.status_code == 200
    cats = r.json()["categorical"]
    assert "Cash loans" in cats["NAME_CONTRACT_TYPE"]
    assert set(cats) == set(get_config().get("features.categorical"))


def test_model_info_has_metrics_thresholds_importance(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    thr = body["thresholds"]
    assert thr["approve_below"] < thr["reject_above"]
    assert body["metrics"]["test_auc"] > 0.5
    imp = body["global_importance"]
    assert abs(sum(i["importance"] for i in imp) - 1.0) < 0.01


def test_audit_list_includes_new_decision(client, demo_applicants):
    aid = client.post("/predict", json={"applicant": demo_applicants[0]}).json()["audit_id"]
    body = client.get("/audit?limit=10").json()
    assert body["stats"]["total"] >= 1
    assert aid in [r["audit_id"] for r in body["records"]]


def test_predict_without_persist_is_not_audited(client, demo_applicants):
    r = client.post("/predict", json={
        "applicant": demo_applicants[0], "persist": False, "include_explanation": False,
        "include_counterfactual": False, "include_rag": False, "include_llm": False,
    })
    assert r.status_code == 200
    assert client.get(f"/audit/{r.json()['audit_id']}").status_code == 404


def test_web_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Credit Risk Intelligence" in r.text
    assert client.get("/app.js").status_code == 200
