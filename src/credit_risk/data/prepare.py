"""Data preparation: load -> clean -> engineer features -> split.

Runs the **same** logic on the real Home Credit ``application_train.csv`` and on
the synthetic fallback, because the synthetic generator emits the same raw
schema. Protected attributes are split off into a separate frame used only for
the fairness audit and never fed to the model.

Outputs (written to ``data/processed/``):
    train.parquet, calib.parquet, test.parquet   -- model features + TARGET
    protected_test.parquet                        -- protected attrs for audit only
    feature_metadata.json                         -- feature lists + provenance
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import Config, get_config
from credit_risk.data import synthetic

# Raw columns we read from the source file (main application table only).
_RAW_COLUMNS = [
    "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2",
    "EXT_SOURCE_3", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_INCOME_TYPE", "NAME_HOUSING_TYPE",
]

DAYS_EMPLOYED_ANOMALY = 365243  # sentinel in real Home Credit for the non-working


@dataclass
class PreparedData:
    train: pd.DataFrame
    calib: pd.DataFrame
    test: pd.DataFrame
    protected_test: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]
    target: str
    source: str  # "real" or "synthetic"


def _load_raw(cfg: Config) -> tuple[pd.DataFrame, str]:
    """Load the real CSV if present, else the synthetic fallback."""
    raw_path = cfg.path("paths.raw_data")
    # The repo ships the dataset zipped (GitHub's 100 MB file limit); pandas
    # reads a single-file zip directly, so no manual extraction is needed.
    zipped = raw_path.with_name(raw_path.name + ".zip")
    if not raw_path.exists() and zipped.exists():
        raw_path = zipped
    if raw_path.exists():
        df = pd.read_csv(raw_path, usecols=lambda c: c in _RAW_COLUMNS)
        missing = [c for c in _RAW_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Real data is missing expected columns: {missing}")
        return df, "real"
    if not bool(cfg.get("data.allow_synthetic_fallback", True)):
        raise FileNotFoundError(
            f"Raw data not found at {raw_path} and synthetic fallback is disabled."
        )
    df = synthetic.generate(
        n_rows=int(cfg.get("data.synthetic_n_rows", 30000)),
        seed=int(cfg.get("project.random_seed", 42)),
    )
    return df, "synthetic"


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw quirks and build the engineered features the model consumes."""
    out = df.copy()

    # Age from negative day counts.
    out["AGE_YEARS"] = (-out["DAYS_BIRTH"] / 365.25).round(1)

    # Employment length: replace the 365243 anomaly with NaN, then to years.
    emp = out["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan)
    out["YEARS_EMPLOYED"] = (-emp / 365.25)
    # Anomalous / non-working rows get 0 years employed (documented choice).
    out["YEARS_EMPLOYED"] = out["YEARS_EMPLOYED"].fillna(0).clip(lower=0).round(2)

    # Normalise the rare 'XNA' gender to missing (audit-only column anyway).
    out["CODE_GENDER"] = out["CODE_GENDER"].replace("XNA", np.nan)

    # Ratio features (guard against divide-by-zero / missing).
    income = out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    annuity = out["AMT_ANNUITY"].replace(0, np.nan)
    out["CREDIT_INCOME_RATIO"] = (out["AMT_CREDIT"] / income).round(3)
    out["ANNUITY_INCOME_RATIO"] = (out["AMT_ANNUITY"] / income).round(4)
    out["CREDIT_TERM"] = (out["AMT_CREDIT"] / annuity).round(1)  # approx months

    return out


def prepare(cfg: Config | None = None, write: bool = True) -> PreparedData:
    cfg = cfg or get_config()
    seed = int(cfg.get("project.random_seed", 42))
    target = str(cfg.get("data.target_column", "TARGET"))
    numeric = list(cfg.get("features.numeric"))
    categorical = list(cfg.get("features.categorical"))
    protected = list(cfg.get("data.protected_attributes", []))

    raw, source = _load_raw(cfg)
    eng = _engineer(raw)

    model_cols = numeric + categorical + [target]
    missing = [c for c in model_cols if c not in eng.columns]
    if missing:
        raise ValueError(f"Engineered frame missing required columns: {missing}")

    X = eng[numeric + categorical].copy()
    y = eng[target].astype(int)
    prot = eng[[c for c in protected if c in eng.columns]].copy()

    # Split: hold out test, then carve a calibration slice from the remainder.
    test_size = float(cfg.get("data.test_size", 0.2))
    calib_size = float(cfg.get("data.calibration_size", 0.15))

    X_tmp, X_test, y_tmp, y_test, prot_tmp, prot_test = train_test_split(
        X, y, prot, test_size=test_size, random_state=seed, stratify=y
    )
    # calib_size is expressed as a fraction of the ORIGINAL data.
    calib_rel = calib_size / (1.0 - test_size)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_tmp, y_tmp, test_size=calib_rel, random_state=seed, stratify=y_tmp
    )

    def _bundle(Xp: pd.DataFrame, yp: pd.Series) -> pd.DataFrame:
        b = Xp.copy()
        b[target] = yp.values
        return b.reset_index(drop=True)

    prepared = PreparedData(
        train=_bundle(X_train, y_train),
        calib=_bundle(X_calib, y_calib),
        test=_bundle(X_test, y_test),
        protected_test=prot_test.reset_index(drop=True),
        numeric_features=numeric,
        categorical_features=categorical,
        target=target,
        source=source,
    )

    if write:
        out_dir = cfg.path("paths.processed_dir")
        out_dir.mkdir(parents=True, exist_ok=True)
        prepared.train.to_parquet(out_dir / "train.parquet", index=False)
        prepared.calib.to_parquet(out_dir / "calib.parquet", index=False)
        prepared.test.to_parquet(out_dir / "test.parquet", index=False)
        prepared.protected_test.to_parquet(out_dir / "protected_test.parquet", index=False)
        meta = {
            "source": source,
            "n_train": len(prepared.train),
            "n_calib": len(prepared.calib),
            "n_test": len(prepared.test),
            "default_rate_train": round(float(prepared.train[target].mean()), 4),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "protected_attributes_excluded_from_model": protected,
            "target": target,
        }
        with open(out_dir / "feature_metadata.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    return prepared


if __name__ == "__main__":
    p = prepare()
    print(f"source={p.source}  train={len(p.train)}  calib={len(p.calib)}  test={len(p.test)}")
    print(f"train default rate = {p.train[p.target].mean():.4f}")
