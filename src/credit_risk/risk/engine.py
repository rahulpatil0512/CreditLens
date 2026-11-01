"""Risk engine: PD -> decision, and Expected Loss = PD x LGD x EAD.

Decision policy
---------------
Rather than hard-coding arbitrary cutoffs, the Approve / Review / Reject bands
are anchored to a break-even argument. A loan is (very roughly) profitable to
approve when its expected margin exceeds its expected loss:

    (1 - PD) * margin * EAD  >  PD * LGD * EAD
        =>  PD  <  margin / (margin + LGD)   =:  PD*  (break-even PD)

We Approve below ``PD* - buffer``, Reject above ``PD* + buffer``, and send the
uncertain middle band to a human ("Review"). A ``fixed`` mode is available to
pin explicit thresholds instead.

IMPORTANT - labelled assumptions
--------------------------------
The Home Credit dataset ships no recovery or exposure outcomes, so LGD and EAD
cannot be estimated from it. The values used here are **assumptions**, surfaced
in every result via the ``assumptions`` field and recorded in the audit trail.
This is a decision-support prototype, not a validated capital model.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from credit_risk.config import Config, get_config

APPROVE, REVIEW, REJECT = "Approve", "Review", "Reject"


@dataclass
class DecisionThresholds:
    mode: str
    approve_below: float
    reject_above: float
    breakeven_pd: float | None
    review_buffer: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskResult:
    pd: float
    decision: str
    thresholds: DecisionThresholds
    lgd: float
    ead: float
    expected_loss: float
    assumptions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["thresholds"] = self.thresholds.as_dict()
        return d


def compute_thresholds(cfg: Config | None = None) -> DecisionThresholds:
    cfg = cfg or get_config()
    mode = str(cfg.get("decision.mode", "cost_based"))
    if mode == "fixed":
        return DecisionThresholds(
            mode="fixed",
            approve_below=float(cfg.get("decision.fixed.approve_below", 0.08)),
            reject_above=float(cfg.get("decision.fixed.reject_above", 0.20)),
            breakeven_pd=None,
            review_buffer=None,
        )
    margin = float(cfg.get("risk.annual_margin_assumption", 0.08))
    lgd = float(cfg.get("risk.lgd_assumption", 0.45))
    buffer = float(cfg.get("decision.review_buffer", 0.03))
    breakeven = margin / (margin + lgd)
    return DecisionThresholds(
        mode="cost_based",
        approve_below=max(0.0, breakeven - buffer),
        reject_above=min(1.0, breakeven + buffer),
        breakeven_pd=round(breakeven, 4),
        review_buffer=buffer,
    )


def decide(pd: float, thresholds: DecisionThresholds | None = None,
           cfg: Config | None = None) -> str:
    thr = thresholds or compute_thresholds(cfg)
    if pd < thr.approve_below:
        return APPROVE
    if pd > thr.reject_above:
        return REJECT
    return REVIEW


def expected_loss(pd: float, ead: float, cfg: Config | None = None) -> dict[str, Any]:
    """EL = PD x LGD x EAD. LGD and EAD are assumptions (see module docstring)."""
    cfg = cfg or get_config()
    lgd = float(cfg.get("risk.lgd_assumption", 0.45))
    el = float(pd) * lgd * float(ead)
    return {"lgd": lgd, "ead": float(ead), "expected_loss": round(el, 2)}


def _ead_from_applicant(applicant: dict[str, Any], cfg: Config) -> float:
    source_col = str(cfg.get("risk.ead_source", "AMT_CREDIT"))
    if source_col not in applicant or applicant[source_col] is None:
        raise ValueError(f"Cannot compute EAD: applicant missing '{source_col}'.")
    return float(applicant[source_col])


def assess(pd: float, applicant: dict[str, Any], cfg: Config | None = None) -> RiskResult:
    """Full risk assessment for one applicant given an already-computed PD."""
    cfg = cfg or get_config()
    if not 0.0 <= pd <= 1.0:
        raise ValueError(f"PD must be in [0, 1], got {pd}.")

    thr = compute_thresholds(cfg)
    decision = decide(pd, thr)
    ead = _ead_from_applicant(applicant, cfg)
    el = expected_loss(pd, ead, cfg)

    lgd = el["lgd"]
    assumptions = [
        f"LGD fixed at {lgd:.2f} (assumption; not estimated from data).",
        f"EAD proxied by '{cfg.get('risk.ead_source', 'AMT_CREDIT')}' "
        f"= {ead:,.0f} (assumption: full drawn exposure).",
    ]
    if thr.mode == "cost_based":
        assumptions.append(
            f"Decision bands derived from break-even PD* = {thr.breakeven_pd} "
            f"(margin/(margin+LGD)); +/-{thr.review_buffer} review buffer. "
            "Single-period simplification."
        )
    return RiskResult(
        pd=round(float(pd), 6),
        decision=decision,
        thresholds=thr,
        lgd=lgd,
        ead=ead,
        expected_loss=el["expected_loss"],
        assumptions=assumptions,
    )


if __name__ == "__main__":
    thr = compute_thresholds()
    print("thresholds:", thr.as_dict())
    for test_pd in (0.03, 0.14, 0.30):
        r = assess(test_pd, {"AMT_CREDIT": 500000}, )
        print(f"PD={test_pd} -> {r.decision}  EL={r.expected_loss:,.0f}")
