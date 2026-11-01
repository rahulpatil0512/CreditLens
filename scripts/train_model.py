#!/usr/bin/env python
"""Entry point: train baseline + primary model, calibrate PD, write artifacts.

Usage:
    python scripts/train_model.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.model.train import train  # noqa: E402


def main() -> None:
    meta = train()
    m = meta["metrics"]
    print(f"[train] data source           : {meta['data_source']}")
    print(f"[train] primary model         : {meta['primary_model']}")
    print(f"[train] baseline LogReg AUC   : {meta['baseline_logreg_test_auc']}")
    print(f"[train] test AUC / PR-AUC / KS: {m['test_auc']} / {m['test_pr_auc']} / {m['test_ks']}")
    print(f"[train] base rate             : {m['base_rate']}")
    print(f"[train] Brier  raw -> calib   : {m['brier_raw']} -> {m['brier_calibrated']}")
    print(f"[train] mean PD raw -> calib  : {m['mean_pd_raw']} -> {m['mean_pd_calibrated']}  "
          f"(should land near base rate)")
    print(f"[train] wrote artifacts/model_calibrated.joblib (+ raw, baseline, metadata, plot)")
    if meta["data_source"] == "synthetic":
        print("[train] NOTE: metrics are on SYNTHETIC data, for pipeline validation only.")


if __name__ == "__main__":
    main()
