"""Train the baseline + primary credit-risk models and calibrate PD.

Why calibration is not optional here
------------------------------------
LightGBM (and especially LightGBM with ``class_weight="balanced"``) outputs a
*score* that rank-orders risk well but is **not** a true probability. Every
downstream number in this system - Expected Loss = PD x LGD x EAD, the
cost-based decision thresholds, the stress-test deltas - assumes PD is a real
probability. So we fit the model on the training split, then learn a monotonic
isotonic mapping on a held-out calibration split that turns raw scores into
calibrated probabilities matching observed default frequencies.

Artifacts written to ``artifacts/``:
    model_calibrated.joblib  -- CalibratedClassifierCV (USE THIS for PD)
    model_raw.joblib         -- fitted pipeline (uncalibrated; used by SHAP)
    baseline_lr.joblib       -- Logistic Regression pipeline (reference)
    model_metadata.json      -- version, source, metrics, feature lists, timestamp
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_risk.config import Config, get_config
from credit_risk.model.preprocessing import build_preprocessor
from credit_risk.model import evaluate as ev


def _load_splits(cfg: Config) -> dict[str, pd.DataFrame]:
    d = cfg.path("paths.processed_dir")
    needed = ["train", "calib", "test", "protected_test"]
    missing = [n for n in needed if not (d / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed splits {missing}. Run the data-prep step first "
            "(python scripts/prepare_data.py)."
        )
    return {n: pd.read_parquet(d / f"{n}.parquet") for n in needed}


def _build_lgbm(cfg: Config) -> LGBMClassifier:
    p = cfg.get("model.lightgbm", {})
    return LGBMClassifier(
        n_estimators=int(p.get("n_estimators", 400)),
        learning_rate=float(p.get("learning_rate", 0.03)),
        num_leaves=int(p.get("num_leaves", 31)),
        max_depth=int(p.get("max_depth", -1)),
        min_child_samples=int(p.get("min_child_samples", 100)),
        subsample=float(p.get("subsample", 0.8)),
        subsample_freq=1,  # required for subsample<1 to take effect in LightGBM
        colsample_bytree=float(p.get("colsample_bytree", 0.8)),
        reg_lambda=float(p.get("reg_lambda", 1.0)),
        class_weight=p.get("class_weight", "balanced"),
        n_jobs=int(p.get("n_jobs", -1)),
        random_state=int(cfg.get("project.random_seed", 42)),
        verbosity=-1,
    )


def _build_lr(cfg: Config) -> LogisticRegression:
    p = cfg.get("model.logistic_regression", {})
    return LogisticRegression(
        C=float(p.get("C", 1.0)),
        max_iter=int(p.get("max_iter", 2000)),
        class_weight=p.get("class_weight", "balanced"),
        random_state=int(cfg.get("project.random_seed", 42)),
    )


def train(cfg: Config | None = None) -> dict:
    cfg = cfg or get_config()
    numeric = list(cfg.get("features.numeric"))
    categorical = list(cfg.get("features.categorical"))
    target = str(cfg.get("data.target_column", "TARGET"))

    splits = _load_splits(cfg)
    Xtr, ytr = splits["train"][numeric + categorical], splits["train"][target]
    Xcal, ycal = splits["calib"][numeric + categorical], splits["calib"][target]
    Xte, yte = splits["test"][numeric + categorical], splits["test"][target]

    primary = str(cfg.get("model.primary", "lightgbm"))

    # --- baseline: Logistic Regression ------------------------------------
    lr_pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric, categorical, scale_numeric=True)),
            ("classifier", _build_lr(cfg)),
        ]
    )
    lr_pipe.fit(Xtr, ytr)
    lr_auc = ev.roc_auc(yte, lr_pipe.predict_proba(Xte)[:, 1])

    # --- primary model (uncalibrated pipeline) ----------------------------
    if primary == "logistic_regression":
        raw_pipe = lr_pipe
    else:
        raw_pipe = Pipeline(
            [
                ("preprocessor", build_preprocessor(numeric, categorical, scale_numeric=False)),
                ("classifier", _build_lgbm(cfg)),
            ]
        )
        raw_pipe.fit(Xtr, ytr)

    # --- calibration: raw score -> true PD on held-out calib split --------
    method = str(cfg.get("model.calibration_method", "isotonic"))
    calibrated = CalibratedClassifierCV(FrozenEstimator(raw_pipe), method=method)
    calibrated.fit(Xcal, ycal)

    # --- evaluation on the untouched test split ---------------------------
    pd_raw = raw_pipe.predict_proba(Xte)[:, 1]
    pd_cal = calibrated.predict_proba(Xte)[:, 1]
    metrics = ev.full_report(yte.to_numpy(), pd_raw, pd_cal, cfg)

    # Calibration curve figure (raw vs calibrated).
    ev.save_calibration_plot(
        yte.to_numpy(), pd_raw, pd_cal, cfg.path("paths.calibration_plot")
    )

    # Fairness: approval-rate disparity by protected group (audit only).
    disparity = ev.group_disparity(
        pd_cal, splits["protected_test"], yte.to_numpy(), cfg
    )

    # --- persist artifacts ------------------------------------------------
    art = cfg.path("paths.artifacts_dir")
    art.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, art / "model_calibrated.joblib")
    joblib.dump(raw_pipe, art / "model_raw.joblib")
    joblib.dump(lr_pipe, art / "baseline_lr.joblib")

    meta = {
        "model_version": str(cfg.get("project.model_version", "0.0.0")),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": _data_source(cfg),
        "primary_model": primary,
        "calibration_method": method,
        "random_seed": int(cfg.get("project.random_seed", 42)),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target": target,
        "metrics": metrics,
        "baseline_logreg_test_auc": round(float(lr_auc), 4),
        "fairness_disparity": disparity,
        "notes": (
            "Prototype. Metrics on SYNTHETIC data unless data_source == 'real'. "
            "Protected attributes excluded from model inputs; disparity reported "
            "for audit only. Not a compliance certification."
        ),
    }
    with open(art / "model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def _data_source(cfg: Config) -> str:
    meta_path = cfg.path("paths.processed_dir") / "feature_metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text()).get("source", "unknown")
    return "unknown"


if __name__ == "__main__":
    m = train()
    print(json.dumps({k: m[k] for k in ("data_source", "primary_model", "metrics",
                                        "baseline_logreg_test_auc")}, indent=2))
