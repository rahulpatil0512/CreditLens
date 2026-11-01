"""SHAP explanations framed as adverse-action reason codes.

Lenders must give specific reasons for a credit decision (ECOA in the US;
EBA loan-origination guidance in the EU). SHAP attributions over the features
the model actually used ARE those reasons. This module:

1. runs TreeSHAP on the (uncalibrated) LightGBM model - calibration is a
   monotonic post-transform, so it rescales PD but does not change which
   factors drive the score or their ranking;
2. aggregates one-hot columns back to their original categorical feature so a
   reason reads "Education = Secondary", not a dummy column;
3. returns the top drivers with sign and a plain-language direction, plus the
   subset that *increase* risk (the principal reasons for a decline).

SHAP values are in log-odds (margin) space. ``base_value + sum(contributions)``
equals the model's raw margin for the applicant.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from credit_risk.config import Config, get_config
from credit_risk.features import to_model_frame
from credit_risk.model.predict import get_model_and_meta  # noqa: F401 (ensures trained)

_RISK_UP = "increases risk"
_RISK_DOWN = "reduces risk"


@lru_cache(maxsize=1)
def _load_explainer():
    """Build a TreeSHAP explainer on the raw LightGBM pipeline (cached)."""
    import joblib

    cfg = get_config()
    raw_pipe = joblib.load(cfg.path("paths.artifacts_dir") / "model_raw.joblib")
    pre = raw_pipe.named_steps["preprocessor"]
    clf = raw_pipe.named_steps["classifier"]
    feat_names = list(pre.get_feature_names_out())

    try:
        import shap

        explainer = shap.TreeExplainer(clf)
        backend = "shap.TreeExplainer"
    except Exception:
        explainer = None  # fall back to LightGBM native TreeSHAP (pred_contrib)
        backend = "lightgbm.pred_contrib"
    return raw_pipe, pre, clf, feat_names, explainer, backend


def _raw_feature(transformed_name: str, numeric: list[str],
                 categorical: list[str]) -> str:
    """Map 'num__AMT_CREDIT' / 'cat__NAME_X_Value' back to the original feature."""
    name = transformed_name
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name in numeric:
        return name
    # categorical one-hot: 'NAME_EDUCATION_TYPE_Higher education'
    for cat in categorical:
        if name.startswith(cat + "_"):
            return cat
    return name


def _shap_for_row(X_trans: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (per-feature contributions, base_value) for a single row."""
    _, _, clf, feat_names, explainer, _ = _load_explainer()
    if explainer is not None:
        exp = explainer(X_trans)
        vals = np.asarray(exp.values)
        base = exp.base_values
        # Binary classifiers may return a trailing class axis; take positive class.
        if vals.ndim == 3:
            vals = vals[:, :, 1]
            base = np.asarray(base)[:, 1]
        contributions = vals[0]
        base_value = float(np.ravel(base)[0])
    else:
        # LightGBM native TreeSHAP: last column is the base (expected) value.
        booster = clf.booster_
        contrib = booster.predict(X_trans, pred_contrib=True)
        contrib = np.asarray(contrib)[0]
        contributions = contrib[:-1]
        base_value = float(contrib[-1])
    return contributions, base_value


def explain(applicant: dict[str, Any], top_n: int = 6,
            cfg: Config | None = None) -> dict[str, Any]:
    """Return SHAP reason codes for one applicant (base-field dict)."""
    cfg = cfg or get_config()
    numeric = list(cfg.get("features.numeric"))
    categorical = list(cfg.get("features.categorical"))

    raw_pipe, pre, _, feat_names, _, backend = _load_explainer()
    X = to_model_frame(applicant, numeric, categorical)
    X_trans = pre.transform(X)
    contributions, base_value = _shap_for_row(X_trans)

    # Aggregate transformed-feature contributions back to original features.
    agg: dict[str, float] = {}
    for fname, val in zip(feat_names, contributions):
        orig = _raw_feature(fname, numeric, categorical)
        agg[orig] = agg.get(orig, 0.0) + float(val)

    from credit_risk.features import engineer_applicant

    eng = engineer_applicant(applicant)
    factors = []
    for feat, contrib in sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True):
        factors.append(
            {
                "feature": feat,
                "value": eng.get(feat),
                "contribution": round(contrib, 4),  # log-odds
                "direction": _RISK_UP if contrib > 0 else _RISK_DOWN,
            }
        )

    top = factors[:top_n]
    # Principal reasons for a decline = the risk-INCREASING factors, largest first.
    adverse = [f for f in top if f["contribution"] > 0]

    return {
        "backend": backend,
        "base_value_log_odds": round(base_value, 4),
        "note": (
            "Contributions are in log-odds (margin) space from the uncalibrated "
            "model; calibration is a monotonic rescaling of the final PD and does "
            "not change factor ranking."
        ),
        "top_factors": top,
        "adverse_action_reasons": adverse,
    }
