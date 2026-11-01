"""Evaluation utilities for the credit-risk model.

Includes the metrics a credit-risk reviewer actually looks at: discrimination
(ROC-AUC, PR-AUC, KS), calibration quality (Brier, reliability curve), a
Population Stability Index helper for drift monitoring, and a basic group
disparity check on the protected attribute held out of the model.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def roc_auc(y_true, p) -> float:
    return float(roc_auc_score(y_true, p))


def ks_statistic(y_true: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between good/bad score distributions."""
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(tpr - fpr))


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between a reference (expected) and new (actual) score distribution.

    Rule of thumb used in banking: <0.1 stable, 0.1-0.25 moderate shift,
    >0.25 significant shift (investigate / consider retraining).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_perc = np.histogram(expected, bins=edges)[0] / len(expected)
    a_perc = np.histogram(actual, bins=edges)[0] / len(actual)
    eps = 1e-6
    e_perc = np.clip(e_perc, eps, None)
    a_perc = np.clip(a_perc, eps, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def full_report(
    y_true: np.ndarray, pd_raw: np.ndarray, pd_cal: np.ndarray, cfg=None
) -> dict:
    """Discrimination + calibration metrics, comparing raw vs calibrated PD."""
    return {
        "test_auc": round(roc_auc(y_true, pd_cal), 4),
        "test_pr_auc": round(float(average_precision_score(y_true, pd_cal)), 4),
        "test_ks": round(ks_statistic(y_true, pd_cal), 4),
        "base_rate": round(float(np.mean(y_true)), 4),
        # Calibration quality: lower Brier is better. Shows WHY calibration ran.
        "brier_raw": round(float(brier_score_loss(y_true, pd_raw)), 5),
        "brier_calibrated": round(float(brier_score_loss(y_true, pd_cal)), 5),
        "logloss_raw": round(float(log_loss(y_true, pd_raw, labels=[0, 1])), 5),
        "logloss_calibrated": round(float(log_loss(y_true, pd_cal, labels=[0, 1])), 5),
        "mean_pd_raw": round(float(np.mean(pd_raw)), 4),
        "mean_pd_calibrated": round(float(np.mean(pd_cal)), 4),
    }


def save_calibration_plot(
    y_true: np.ndarray, pd_raw: np.ndarray, pd_cal: np.ndarray, out_path: Path
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frac_raw, mean_raw = calibration_curve(y_true, pd_raw, n_bins=10, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_true, pd_cal, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    ax.plot(mean_raw, frac_raw, "o-", label="Raw model score")
    ax.plot(mean_cal, frac_cal, "s-", label="Calibrated PD (isotonic)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default frequency")
    ax.set_title("Reliability curve: raw score vs calibrated PD")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def group_disparity(
    pd_cal: np.ndarray, protected: pd.DataFrame, y_true: np.ndarray, cfg
) -> dict:
    """Basic fairness check: approval-rate and mean-PD by protected group.

    Uses the fixed approval cutoff as a reference point (the live decision policy
    is cost-based, but a single cutoff makes the disparity interpretable). This
    is a lightweight audit signal, NOT a full fairness assessment.
    """
    approve_below = float(cfg.get("decision.fixed.approve_below", 0.08))
    report: dict = {
        "reference_approve_cutoff": approve_below,
        "method": "approval rate = share with calibrated PD < cutoff",
        "groups": {},
    }
    approved = pd_cal < approve_below
    for col in protected.columns:
        groups: dict = {}
        series = protected[col].astype("object").fillna("Unknown")
        for value, idx in series.groupby(series).groups.items():
            mask = series.index.isin(idx)
            if mask.sum() < 30:  # skip tiny groups to avoid noise
                continue
            groups[str(value)] = {
                "n": int(mask.sum()),
                "approval_rate": round(float(approved[mask].mean()), 4),
                "mean_pd": round(float(pd_cal[mask].mean()), 4),
                "observed_default_rate": round(float(np.asarray(y_true)[mask].mean()), 4),
            }
        rates = [g["approval_rate"] for g in groups.values()]
        if len(rates) >= 2 and max(rates) > 0:
            # Disparate-impact-style ratio: min/max approval rate across groups.
            report["approval_rate_ratio_min_over_max"] = round(min(rates) / max(rates), 3)
        report["groups"][col] = groups
    return report
