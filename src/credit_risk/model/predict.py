"""Inference layer: load saved artifacts once, return calibrated PD.

Nothing here retrains. The calibrated model is memoised so repeated API calls
are cheap. ``predict_pd`` is the single entry point used by the API, demo,
counterfactual engine and stress tests.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from credit_risk.config import Config, get_config
from credit_risk.features import to_model_frame


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load(cfg_id: int):
    cfg = get_config()
    art = cfg.path("paths.artifacts_dir")
    cal_path = art / "model_calibrated.joblib"
    meta_path = art / "model_metadata.json"
    if not cal_path.exists() or not meta_path.exists():
        raise ModelNotTrainedError(
            "No trained model found. Run: python scripts/train_model.py"
        )
    model = joblib.load(cal_path)
    meta = json.loads(meta_path.read_text())
    return model, meta


def get_model_and_meta() -> tuple[Any, dict]:
    return _load(0)


def model_metadata() -> dict:
    return _load(0)[1]


def predict_pd(applicant: dict[str, Any], cfg: Config | None = None) -> float:
    """Return calibrated Probability of Default for one applicant (base fields)."""
    cfg = cfg or get_config()
    numeric = list(cfg.get("features.numeric"))
    categorical = list(cfg.get("features.categorical"))
    model, _ = get_model_and_meta()
    X = to_model_frame(applicant, numeric, categorical)
    raw = float(model.predict_proba(X)[:, 1][0])
    return _floor_pd(raw)


# Isotonic calibration can emit exactly 0.0 / 1.0 in its extreme bins, which
# makes Expected Loss degenerate (no real loan has 0% default risk). Clip to a
# tiny floor/ceiling. Documented in the README.
_PD_FLOOR = 5e-4


def _floor_pd(p: float) -> float:
    return float(min(max(p, _PD_FLOOR), 1.0 - _PD_FLOOR))


def predict_pd_batch(applicants: list[dict[str, Any]],
                     cfg: Config | None = None) -> list[float]:
    cfg = cfg or get_config()
    numeric = list(cfg.get("features.numeric"))
    categorical = list(cfg.get("features.categorical"))
    model, _ = get_model_and_meta()
    from credit_risk.features import engineer_applicant
    rows = [engineer_applicant(a) for a in applicants]
    X = pd.DataFrame(
        [{c: r.get(c) for c in (numeric + categorical)} for r in rows],
        columns=numeric + categorical,
    )
    return [_floor_pd(float(p)) for p in model.predict_proba(X)[:, 1]]


@lru_cache(maxsize=1)
def _load_raw_pipeline():
    cfg = get_config()
    path = cfg.path("paths.artifacts_dir") / "model_raw.joblib"
    if not path.exists():
        raise ModelNotTrainedError(
            "No trained model found. Run: python scripts/train_model.py"
        )
    return joblib.load(path)


def category_options(cfg: Config | None = None) -> dict[str, list[str]]:
    """Category values seen in training for each categorical feature.

    Read from the fitted one-hot encoder, so a UI can offer exactly the values
    the model knows (unseen values are tolerated but carry no signal).
    """
    cfg = cfg or get_config()
    categorical = list(cfg.get("features.categorical"))
    pre = _load_raw_pipeline().named_steps["preprocessor"]
    onehot = pre.named_transformers_["cat"].named_steps["onehot"]
    return {
        feat: [str(v) for v in cats]
        for feat, cats in zip(categorical, onehot.categories_)
    }


def global_importance(cfg: Config | None = None) -> list[dict[str, Any]]:
    """Split-gain importance aggregated back to original features (normalised)."""
    cfg = cfg or get_config()
    numeric = list(cfg.get("features.numeric"))
    categorical = list(cfg.get("features.categorical"))
    pipe = _load_raw_pipeline()
    clf = pipe.named_steps["classifier"]
    names = list(pipe.named_steps["preprocessor"].get_feature_names_out())
    if hasattr(clf, "booster_"):
        values = clf.booster_.feature_importance(importance_type="gain")
    elif hasattr(clf, "coef_"):
        values = abs(clf.coef_[0])
    else:
        return []

    agg: dict[str, float] = {}
    for name, val in zip(names, values):
        raw = name.split("__", 1)[-1]
        feat = raw if raw in numeric else next(
            (c for c in categorical if raw.startswith(c + "_")), raw)
        agg[feat] = agg.get(feat, 0.0) + float(val)
    total = sum(agg.values()) or 1.0
    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    return [{"feature": f, "importance": round(v / total, 4)} for f, v in ranked]
