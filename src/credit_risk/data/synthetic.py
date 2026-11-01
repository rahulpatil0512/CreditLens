"""Synthetic data generator mimicking the Home Credit ``application_train`` schema.

Purpose
-------
The real Home Credit Default Risk dataset requires a Kaggle account to download.
To keep the project fully runnable offline (and testable in CI), this module
produces a dataframe with the **same raw column names and quirks** as the real
file, including:

* ``DAYS_BIRTH`` stored as a negative integer (days before application),
* the infamous ``DAYS_EMPLOYED == 365243`` anomaly for the non-working,
* external scores in ``[0, 1]`` with realistic missingness,
* an imbalanced ``TARGET`` (~8% positive) driven by a genuine latent signal.

Because the schema matches, ``data/prepare.py`` runs the **identical** cleaning
and feature-engineering code on synthetic and real data. This is clearly
labelled synthetic everywhere and must never be presented as real outcomes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

_EDUCATION = [
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]
_FAMILY = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
_INCOME_TYPE = ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"]
_HOUSING = ["House / apartment", "With parents", "Rented apartment", "Municipal apartment", "Office apartment"]


def generate(n_rows: int = 30000, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic dataframe shaped like Home Credit ``application_train``."""
    rng = np.random.default_rng(seed)
    n = int(n_rows)

    # --- demographics -----------------------------------------------------
    age_years = rng.normal(43, 11, n).clip(21, 69)
    days_birth = (-age_years * 365.25).astype(int)

    income_type = rng.choice(_INCOME_TYPE, n, p=[0.52, 0.23, 0.18, 0.06, 0.01])
    # Employment length; pensioners/unemployed carry the 365243 anomaly sentinel.
    years_employed = rng.gamma(shape=2.0, scale=3.5, size=n).clip(0, 45)
    days_employed = (-years_employed * 365.25).astype(int)
    non_working = np.isin(income_type, ["Pensioner", "Unemployed"])
    days_employed = np.where(non_working, 365243, days_employed)  # real-data anomaly

    gender = rng.choice(["F", "M", "XNA"], n, p=[0.64, 0.355, 0.005])  # XNA like real data

    # --- money ------------------------------------------------------------
    income = rng.lognormal(mean=11.95, sigma=0.45, size=n).clip(27000, 1_200_000)
    income = np.round(income / 4500) * 4500
    credit = (income * rng.uniform(0.8, 6.0, n)).clip(45000, 4_000_000)
    credit = np.round(credit / 4500) * 4500
    goods_price = (credit * rng.uniform(0.80, 1.0, n)).round(-3)
    term_months = rng.uniform(12, 60, n)
    annuity = (credit / term_months).round(-2).clip(1500, None)

    children = rng.choice([0, 1, 2, 3, 4], n, p=[0.58, 0.25, 0.12, 0.04, 0.01])
    fam_members = (children + rng.choice([1, 2], n, p=[0.35, 0.65])).astype(float)

    # --- external credit scores (strong, partly missing) ------------------
    base_quality = rng.beta(2.4, 2.0, n)  # latent creditworthiness in [0, 1]
    ext1 = (base_quality + rng.normal(0, 0.14, n)).clip(0, 1)
    ext2 = (base_quality + rng.normal(0, 0.12, n)).clip(0, 1)
    ext3 = (base_quality + rng.normal(0, 0.16, n)).clip(0, 1)
    ext1[rng.random(n) < 0.56] = np.nan  # EXT_SOURCE_1 is missing ~56% in real data
    ext2[rng.random(n) < 0.01] = np.nan
    ext3[rng.random(n) < 0.20] = np.nan

    # --- latent default propensity (the signal the model should recover) ---
    credit_income = credit / income
    annuity_income = annuity / income
    ext_stack = np.vstack([ext1, ext2, ext3])
    with warnings.catch_warnings():  # rows with all three scores missing -> NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ext_mean = np.nanmean(ext_stack, axis=0)
    ext_mean = np.where(np.isnan(ext_mean), 0.5, ext_mean)  # impute to neutral

    logit = (
        -2.65
        - 3.4 * (ext_mean - 0.5)          # better external score -> lower risk
        + 0.45 * (credit_income - 3.0)     # higher leverage -> higher risk
        + 2.0 * (annuity_income - 0.22)    # higher annuity burden -> higher risk
        - 0.020 * (age_years - 43)         # younger -> slightly higher risk
        - 0.030 * years_employed.clip(0, 20)
        + 0.15 * (children)
        + rng.normal(0, 0.55, n)           # irreducible noise
    )
    p_default = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.random(n) < p_default).astype(int)

    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100001, 100001 + n),
            "TARGET": target,
            "NAME_CONTRACT_TYPE": rng.choice(
                ["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]
            ),
            "CODE_GENDER": gender,
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods_price,
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "CNT_CHILDREN": children,
            "CNT_FAM_MEMBERS": fam_members,
            "NAME_EDUCATION_TYPE": rng.choice(_EDUCATION, n, p=[0.71, 0.24, 0.03, 0.015, 0.005]),
            "NAME_FAMILY_STATUS": rng.choice(_FAMILY, n, p=[0.64, 0.15, 0.10, 0.07, 0.04]),
            "NAME_INCOME_TYPE": income_type,
            "NAME_HOUSING_TYPE": rng.choice(_HOUSING, n, p=[0.89, 0.05, 0.03, 0.02, 0.01]),
        }
    )
    return df


if __name__ == "__main__":  # quick manual check
    d = generate(5000)
    print(d.shape)
    print("default rate:", round(d["TARGET"].mean(), 4))
    print(d.head(3).T)
