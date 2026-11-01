"""Scenario sensitivity analysis (a.k.a. 'stress testing' in the UI).

Terminology note (deliberate, not a buzzword)
----------------------------------------------
This is *feature-shock scenario sensitivity analysis*: we apply shocks to input
features (income down, job loss, higher annuity from a rate rise) and recompute
PD and Expected Loss. It is NOT regulatory macro stress testing (EBA/CCAR),
which links macro variables (GDP, unemployment) to default rates through a
separate macro model we do not have. The README states this limitation plainly.

Scenarios are defined in ``config.yaml`` under ``stress.scenarios``. Each shock
is either ``{multiply: x}`` or ``{set: v}`` applied to a BASE feature; derived
ratios are then recomputed by the shared feature engineering.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from credit_risk.config import Config, get_config
from credit_risk.features import BASE_NUMERIC, CATEGORICAL
from credit_risk.model.predict import predict_pd, predict_pd_batch
from credit_risk.risk import engine as risk_engine


def _apply_shocks(base: dict[str, Any], shocks: dict[str, dict]) -> dict[str, Any]:
    out = dict(base)
    for feat, op in shocks.items():
        if feat not in out or out[feat] is None:
            continue
        if "multiply" in op:
            out[feat] = round(float(out[feat]) * float(op["multiply"]), 2)
        elif "set" in op:
            out[feat] = op["set"]
    return out


def stress_applicant(applicant: dict[str, Any], cfg: Config | None = None) -> dict[str, Any]:
    """Per-scenario PD / decision / EL for one applicant vs baseline."""
    cfg = cfg or get_config()
    scenarios: dict[str, dict] = dict(cfg.get("stress.scenarios", {}))
    thr = risk_engine.compute_thresholds(cfg)
    ead_col = str(cfg.get("risk.ead_source", "AMT_CREDIT"))

    base_pd = predict_pd(applicant, cfg)
    base = {
        "pd": round(base_pd, 4),
        "decision": risk_engine.decide(base_pd, thr),
        "expected_loss": risk_engine.expected_loss(base_pd, applicant[ead_col], cfg)["expected_loss"],
    }

    rows = []
    for name, spec in scenarios.items():
        shocked = _apply_shocks(applicant, dict(spec.get("shocks", {})))
        spd = predict_pd(shocked, cfg)
        # EAD may itself move if the shock touched AMT_CREDIT.
        sead = float(shocked.get(ead_col, applicant[ead_col]))
        sel = risk_engine.expected_loss(spd, sead, cfg)["expected_loss"]
        rows.append({
            "scenario": name,
            "description": spec.get("description", ""),
            "pd": round(spd, 4),
            "pd_delta": round(spd - base_pd, 4),
            "decision": risk_engine.decide(spd, thr),
            "decision_changed": risk_engine.decide(spd, thr) != base["decision"],
            "expected_loss": sel,
            "expected_loss_delta": round(sel - base["expected_loss"], 2),
        })
    return {"baseline": base, "scenarios": rows}


def stress_portfolio(applicants: list[dict[str, Any]],
                     cfg: Config | None = None) -> dict[str, Any]:
    """Aggregate scenario impact across a set of applicants."""
    cfg = cfg or get_config()
    scenarios: dict[str, dict] = dict(cfg.get("stress.scenarios", {}))
    thr = risk_engine.compute_thresholds(cfg)
    ead_col = str(cfg.get("risk.ead_source", "AMT_CREDIT"))
    lgd = float(cfg.get("risk.lgd_assumption", 0.45))

    def portfolio_metrics(apps: list[dict[str, Any]]) -> dict[str, float]:
        pds = predict_pd_batch(apps, cfg)
        els = [p * lgd * float(a[ead_col]) for p, a in zip(pds, apps)]
        decisions = [risk_engine.decide(p, thr) for p in pds]
        n = len(apps)
        return {
            "n": n,
            "mean_pd": round(sum(pds) / n, 4),
            "total_expected_loss": round(sum(els), 2),
            "approve_rate": round(decisions.count(risk_engine.APPROVE) / n, 4),
            "reject_rate": round(decisions.count(risk_engine.REJECT) / n, 4),
        }

    baseline = portfolio_metrics(applicants)
    rows = []
    for name, spec in scenarios.items():
        shocks = dict(spec.get("shocks", {}))
        shocked_apps = [_apply_shocks(a, shocks) for a in applicants]
        m = portfolio_metrics(shocked_apps)
        rows.append({
            "scenario": name,
            "description": spec.get("description", ""),
            "mean_pd": m["mean_pd"],
            "mean_pd_delta": round(m["mean_pd"] - baseline["mean_pd"], 4),
            "total_expected_loss": m["total_expected_loss"],
            "expected_loss_increase_pct": round(
                100 * (m["total_expected_loss"] - baseline["total_expected_loss"])
                / max(baseline["total_expected_loss"], 1e-9), 1),
            "approve_rate": m["approve_rate"],
        })
    return {"baseline": baseline, "scenarios": rows}


def load_portfolio_sample(n: int = 500, cfg: Config | None = None) -> list[dict[str, Any]]:
    """Load a base-field sample from the processed test split (demo convenience)."""
    cfg = cfg or get_config()
    path = cfg.path("paths.processed_dir") / "test.parquet"
    if not path.exists():
        raise FileNotFoundError("Run data prep first (scripts/prepare_data.py).")
    df = pd.read_parquet(path)
    cols = [c for c in (BASE_NUMERIC + CATEGORICAL) if c in df.columns]
    sample = df[cols].head(n)
    return sample.to_dict(orient="records")
