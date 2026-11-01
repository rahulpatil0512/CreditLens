"""Constrained counterfactual search.

Finds the smallest realistic change to *actionable* loan terms that would move a
Reject/Review to Approve. Only the features in ``counterfactual.actionable_features``
may move, and only in their allowed direction and range. Immutable attributes
(age, external scores, protected attributes) are frozen.

Honesty by design: for some applicants no feasible change exists within the
allowed range. The engine returns ``feasible: false`` rather than inventing one.

Search: enumerate allowed multipliers per actionable feature, evaluate all
single-feature changes and all pairs (``max_features_changed``), keep those that
reach the target decision, and rank by total relative change (prefer the
smallest, simplest change).
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Any

from credit_risk.config import Config, get_config
from credit_risk.model.predict import predict_pd_batch
from credit_risk.risk import engine as risk_engine


def _multipliers(spec: dict) -> list[float]:
    step = float(spec.get("step_factor", 0.05))
    direction = spec.get("direction", "decrease")
    out: list[float] = []
    if direction == "decrease":
        lo = float(spec.get("min_factor", 0.5))
        m = 1.0 - step
        while m >= lo - 1e-9:
            out.append(round(m, 4))
            m -= step
    else:  # increase
        hi = float(spec.get("max_factor", 1.5))
        m = 1.0 + step
        while m <= hi + 1e-9:
            out.append(round(m, 4))
            m += step
    return out


def _apply(applicant: dict[str, Any], changes: dict[str, float]) -> dict[str, Any]:
    """Return a copy of the applicant with multiplicative changes applied."""
    out = dict(applicant)
    for feat, mult in changes.items():
        if out.get(feat) is not None:
            out[feat] = round(float(out[feat]) * mult, 2)
    return out


def _cost(changes: dict[str, float]) -> float:
    return sum(abs(1.0 - m) for m in changes.values())


def generate(applicant: dict[str, Any], max_alternatives: int = 3,
             cfg: Config | None = None) -> dict[str, Any]:
    cfg = cfg or get_config()
    actionable: dict[str, dict] = dict(cfg.get("counterfactual.actionable_features", {}))
    max_changed = int(cfg.get("counterfactual.max_features_changed", 2))
    target = str(cfg.get("counterfactual.target_decision", "Approve"))
    thr = risk_engine.compute_thresholds(cfg)

    base_pd = predict_pd_batch([applicant], cfg)[0]
    base_decision = risk_engine.decide(base_pd, thr)

    result: dict[str, Any] = {
        "baseline_pd": round(base_pd, 4),
        "baseline_decision": base_decision,
        "target_decision": target,
        "actionable_features": list(actionable.keys()),
    }
    if base_decision == target:
        result.update({"feasible": True, "already_at_target": True,
                       "counterfactuals": []})
        return result

    # Candidate change-sets: all singles + all pairs (up to max_changed).
    feats = [f for f in actionable if applicant.get(f) is not None]
    grids = {f: _multipliers(actionable[f]) for f in feats}

    change_sets: list[dict[str, float]] = []
    for f in feats:
        for m in grids[f]:
            change_sets.append({f: m})
    if max_changed >= 2:
        for f1, f2 in combinations(feats, 2):
            for m1, m2 in product(grids[f1], grids[f2]):
                change_sets.append({f1: m1, f2: m2})

    if not change_sets:
        result.update({"feasible": False, "reason": "No actionable features present.",
                       "counterfactuals": []})
        return result

    # Batch-evaluate all candidates.
    candidates = [_apply(applicant, cs) for cs in change_sets]
    pds = predict_pd_batch(candidates, cfg)

    feasible = []
    for cs, cand, pd_ in zip(change_sets, candidates, pds):
        if risk_engine.decide(pd_, thr) == target:
            feasible.append((cs, cand, pd_))

    if not feasible:
        result.update({
            "feasible": False,
            "reason": (
                "No change to the allowed loan terms, within realistic limits, "
                "reaches the target decision. The decline is driven mainly by "
                "non-actionable factors (e.g. external credit scores)."
            ),
            "counterfactuals": [],
        })
        return result

    # Rank by smallest total relative change, then by lowest resulting PD.
    feasible.sort(key=lambda t: (_cost(t[0]), t[2]))

    out = []
    for cs, cand, pd_ in feasible[:max_alternatives]:
        changes = []
        for feat, mult in cs.items():
            changes.append({
                "feature": feat,
                "from": round(float(applicant[feat]), 2),
                "to": round(float(applicant[feat]) * mult, 2),
                "pct_change": round((mult - 1.0) * 100, 1),
            })
        out.append({
            "changes": changes,
            "resulting_pd": round(pd_, 4),
            "resulting_decision": target,
            "total_relative_change": round(_cost(cs), 4),
        })

    result.update({"feasible": True, "already_at_target": False, "counterfactuals": out})
    return result
