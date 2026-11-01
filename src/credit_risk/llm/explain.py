"""Narrative generation for a decision.

Design rule (enforced by architecture, not trust)
--------------------------------------------------
Every number, decision, factor and source is computed upstream and passed in as
``facts``. The deterministic template turns those facts into text. The LLM, when
available, is asked ONLY to re-phrase that text more fluently - it is never the
source of any figure. If the LLM is disabled, unavailable, errors, or returns
output that drops/contradicts the decision, we use the template text. So the
system is fully functional with no API key and can never have an LLM invent a PD,
a calculation, a decision, or a regulatory claim.
"""
from __future__ import annotations

from typing import Any

from credit_risk.config import Config, get_config

_SYSTEM = (
    "You are a communication assistant for a credit decision-support tool. "
    "You will be given a factual explanation that is already complete and "
    "correct. Your ONLY job is to rewrite it in clear, professional, plain "
    "language for an internal credit officer. Strict rules: do not add, remove "
    "or change any number, probability, amount, decision or factor; do not "
    "introduce new facts; do not cite any regulation that is not already "
    "mentioned; never state or imply that the decision is legally or "
    "regulatorily compliant. If you are unsure, keep the original wording."
)


def _fmt_money(x: float) -> str:
    return f"{x:,.0f}"


def render_template(facts: dict[str, Any]) -> str:
    """Deterministic, offline explanation built purely from facts."""
    d = facts["decision"]
    pd = facts["pd"]
    lines: list[str] = []
    lines.append(f"Decision: {d} (Probability of Default = {pd:.1%}).")

    risk = facts.get("risk", {})
    if risk:
        lines.append(
            f"Estimated Expected Loss is {_fmt_money(risk['expected_loss'])} "
            f"(PD x LGD x EAD, with LGD = {risk['lgd']:.2f} and "
            f"EAD = {_fmt_money(risk['ead'])})."
        )

    reasons = facts.get("adverse_action_reasons") or []
    if d != "Approve" and reasons:
        top = "; ".join(f"{r['feature']} ({r['direction']})" for r in reasons[:4])
        lines.append(f"Principal factors increasing risk: {top}.")
    elif d == "Approve":
        pos = [f for f in facts.get("top_factors", []) if f["contribution"] < 0][:3]
        if pos:
            drivers = "; ".join(f["feature"] for f in pos)
            lines.append(f"Main factors supporting approval: {drivers}.")

    cf = facts.get("counterfactuals")
    if cf and cf.get("feasible") and cf.get("counterfactuals"):
        best = cf["counterfactuals"][0]
        changes = "; ".join(
            f"{c['feature']} {c['pct_change']:+.0f}%" for c in best["changes"]
        )
        lines.append(
            f"A different outcome ({cf['target_decision']}) could be reached by: "
            f"{changes} (projected PD {best['resulting_pd']:.1%})."
        )
    elif cf and not cf.get("feasible"):
        lines.append(
            "No realistic change to the requested loan terms would reach approval; "
            "the decline is driven mainly by non-actionable factors."
        )

    ev = facts.get("rag_evidence") or []
    if ev:
        srcs = ", ".join(sorted({e["source"] for e in ev}))
        lines.append(f"Supporting policy/reference evidence retrieved from: {srcs}.")

    for a in risk.get("assumptions", []):
        lines.append(f"Assumption: {a}")

    lines.append(
        "Note: regulation-grounded decision-support prototype; not a compliance "
        "certification."
    )
    return "\n".join(lines)


def _looks_valid(llm_text: str, facts: dict[str, Any]) -> bool:
    """Light guardrail: keep LLM text only if it preserves the decision and
    makes no compliance claim."""
    if not llm_text or len(llm_text.strip()) < 20:
        return False
    low = llm_text.lower()
    if facts["decision"].lower() not in low:
        return False
    banned = ["fully compliant", "is compliant", "guarantee", "guaranteed",
              "legally compliant", "regulatory compliant"]
    return not any(b in low for b in banned)


def generate_explanation(facts: dict[str, Any], cfg: Config | None = None) -> dict[str, Any]:
    cfg = cfg or get_config()
    template_text = render_template(facts)

    if not cfg.llm_available:
        return {"mode": "template", "text": template_text,
                "reason": "LLM disabled or no API key."}

    from credit_risk.llm import providers

    user = (
        "Rewrite the following explanation in clear, professional language for a "
        "credit officer. Keep every number, decision and factor exactly as given.\n\n"
        f"{template_text}"
    )
    try:
        out = providers.complete(_SYSTEM, user, cfg)
    except providers.LLMUnavailable as e:
        return {"mode": "template", "text": template_text,
                "reason": f"LLM unavailable: {e}"}
    except Exception as e:  # any provider/runtime error -> safe fallback
        return {"mode": "template", "text": template_text,
                "reason": f"LLM error: {type(e).__name__}"}

    if not _looks_valid(out, facts):
        return {"mode": "template", "text": template_text,
                "reason": "LLM output failed guardrail check; used template."}
    return {"mode": "llm", "text": out.strip(), "provider": cfg.llm_provider,
            "model": str(cfg.get("llm.model"))}
