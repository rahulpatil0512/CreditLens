"""Deterministic unit tests for the phase 1-3 core (no model / network needed).

Run: pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import get_config
from credit_risk.features import engineer_applicant
from credit_risk.model.evaluate import population_stability_index, ks_statistic
from credit_risk.risk import engine


# --- feature engineering -------------------------------------------------
def test_engineer_applicant_recomputes_ratios():
    base = {"AMT_INCOME_TOTAL": 100000, "AMT_CREDIT": 500000, "AMT_ANNUITY": 25000}
    out = engineer_applicant(base)
    assert out["CREDIT_INCOME_RATIO"] == pytest.approx(5.0)
    assert out["ANNUITY_INCOME_RATIO"] == pytest.approx(0.25)
    assert out["CREDIT_TERM"] == pytest.approx(20.0)  # 500000 / 25000


def test_engineer_applicant_handles_zero_income():
    out = engineer_applicant({"AMT_INCOME_TOTAL": 0, "AMT_CREDIT": 500000,
                              "AMT_ANNUITY": 25000})
    assert out["CREDIT_INCOME_RATIO"] is None  # no divide-by-zero crash


# --- risk engine decisions ----------------------------------------------
def test_cost_based_thresholds_match_breakeven():
    cfg = get_config()
    thr = engine.compute_thresholds(cfg)
    margin = float(cfg.get("risk.annual_margin_assumption"))
    lgd = float(cfg.get("risk.lgd_assumption"))
    assert thr.breakeven_pd == pytest.approx(margin / (margin + lgd), abs=1e-4)
    assert thr.approve_below < thr.breakeven_pd < thr.reject_above


def test_decision_bands():
    thr = engine.compute_thresholds()
    assert engine.decide(thr.approve_below - 0.01, thr) == engine.APPROVE
    assert engine.decide((thr.approve_below + thr.reject_above) / 2, thr) == engine.REVIEW
    assert engine.decide(thr.reject_above + 0.01, thr) == engine.REJECT


def test_expected_loss_formula():
    cfg = get_config()
    lgd = float(cfg.get("risk.lgd_assumption"))
    el = engine.expected_loss(pd=0.10, ead=500000, cfg=cfg)
    assert el["expected_loss"] == pytest.approx(0.10 * lgd * 500000, abs=1.0)


def test_assess_rejects_invalid_pd():
    with pytest.raises(ValueError):
        engine.assess(1.5, {"AMT_CREDIT": 100000})


def test_assess_includes_labeled_assumptions():
    r = engine.assess(0.05, {"AMT_CREDIT": 400000})
    joined = " ".join(r.assumptions).lower()
    assert "assumption" in joined and "lgd" in joined and "ead" in joined


# --- evaluation metrics --------------------------------------------------
def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_positive_for_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(1.5, 1, 5000)  # clearly shifted
    assert population_stability_index(a, b) > 0.25


def test_ks_in_unit_range():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = rng.random(2000)
    assert 0.0 <= ks_statistic(y, p) <= 1.0
