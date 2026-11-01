"""Shared applicant feature engineering.

A single source of truth for turning a raw applicant record (base economic +
demographic fields) into the exact model feature vector. Reused by prediction,
counterfactuals and stress testing so derived ratios are ALWAYS recomputed
consistently after any field is changed.

Applicant input schema (what callers provide)
----------------------------------------------
Base numeric : AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE,
               AGE_YEARS, YEARS_EMPLOYED, EXT_SOURCE_1, EXT_SOURCE_2,
               EXT_SOURCE_3, CNT_CHILDREN, CNT_FAM_MEMBERS
Categorical  : NAME_CONTRACT_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS,
               NAME_INCOME_TYPE, NAME_HOUSING_TYPE

Derived here : CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_TERM
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Base fields a caller supplies directly (everything except derived ratios).
BASE_NUMERIC = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "AGE_YEARS", "YEARS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2",
    "EXT_SOURCE_3", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
]
CATEGORICAL = [
    "NAME_CONTRACT_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE", "NAME_HOUSING_TYPE",
]
DERIVED_NUMERIC = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM"]


def _safe_div(a: float, b: float) -> float | None:
    if b in (None, 0) or (isinstance(b, float) and np.isnan(b)):
        return None
    return a / b


def engineer_applicant(base: dict[str, Any]) -> dict[str, Any]:
    """Return a full model-feature dict with derived ratios recomputed."""
    out = dict(base)
    income = out.get("AMT_INCOME_TOTAL")
    credit = out.get("AMT_CREDIT")
    annuity = out.get("AMT_ANNUITY")

    cir = _safe_div(credit, income) if credit is not None else None
    air = _safe_div(annuity, income) if annuity is not None else None
    term = _safe_div(credit, annuity) if credit is not None else None

    out["CREDIT_INCOME_RATIO"] = round(cir, 3) if cir is not None else None
    out["ANNUITY_INCOME_RATIO"] = round(air, 4) if air is not None else None
    out["CREDIT_TERM"] = round(term, 1) if term is not None else None
    return out


def to_model_frame(base: dict[str, Any], numeric: list[str],
                   categorical: list[str]) -> pd.DataFrame:
    """Engineer one applicant and return a 1-row DataFrame in model schema."""
    full = engineer_applicant(base)
    row = {c: full.get(c) for c in (numeric + categorical)}
    return pd.DataFrame([row], columns=numeric + categorical)
