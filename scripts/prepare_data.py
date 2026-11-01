#!/usr/bin/env python
"""Entry point: prepare data (real Home Credit CSV if present, else synthetic).

Usage:
    python scripts/prepare_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.data.prepare import prepare  # noqa: E402


def main() -> None:
    p = prepare(write=True)
    print(f"[prepare] source        : {p.source}")
    print(f"[prepare] train / calib / test = {len(p.train)} / {len(p.calib)} / {len(p.test)}")
    print(f"[prepare] train default rate   = {p.train[p.target].mean():.4f}")
    print("[prepare] wrote data/processed/{train,calib,test,protected_test}.parquet")
    if p.source == "synthetic":
        print("[prepare] NOTE: using SYNTHETIC data. Drop the real "
              "application_train.csv into data/raw/ to use real data.")


if __name__ == "__main__":
    main()
