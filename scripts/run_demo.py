#!/usr/bin/env python
"""End-to-end demo on the bundled synthetic applicants (no manual input needed).

Runs the full assessment for each demo applicant: PD -> decision -> SHAP reason
codes -> counterfactuals -> RAG evidence -> narrative -> audit. Then shows a
single-applicant stress test and a RAG query. Works fully offline (template
narrative) and automatically uses the LLM if a key is configured.

Usage:
    python scripts/run_demo.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.service import assess  # noqa: E402
from credit_risk.stress.engine import stress_applicant  # noqa: E402
from credit_risk.rag.query import query  # noqa: E402

LINE = "=" * 78


def main() -> None:
    applicants = json.loads((ROOT / "data" / "demo_applicants.json").read_text())

    for a in applicants:
        rec = assess(a)
        print(LINE)
        print(f"{a.get('applicant_id', '?')}  |  {a.get('label_hint', '')}")
        print(LINE)
        print(f"  PD            : {rec['pd']:.2%}")
        print(f"  Decision      : {rec['decision']}")
        print(f"  Expected Loss : {rec['risk']['expected_loss']:,.0f} "
              f"(LGD={rec['risk']['lgd']}, EAD={rec['risk']['ead']:,.0f})")
        aar = rec.get("adverse_action_reasons", [])
        if aar:
            print("  Risk drivers  : " + ", ".join(r["feature"] for r in aar[:4]))
        cf = rec.get("counterfactuals")
        if cf:
            if cf["feasible"] and cf["counterfactuals"]:
                best = cf["counterfactuals"][0]
                chg = ", ".join(f"{c['feature']} {c['pct_change']:+.0f}%"
                                for c in best["changes"])
                print(f"  To approve    : {chg} -> PD {best['resulting_pd']:.2%}")
            elif not cf["feasible"]:
                print("  To approve    : no feasible change to loan terms")
        print(f"  RAG evidence  : "
              f"{', '.join(sorted({e['source'] for e in rec.get('rag_evidence', [])}))}")
        print(f"  Narrative     : [{rec['narrative']['mode']}]")
        print("  " + rec["narrative"]["text"].replace("\n", "\n  "))
        print(f"  audit_id      : {rec['audit_id']}")
        print()

    # Stress test on the borderline applicant
    mid = next((a for a in applicants if "MID" in a.get("applicant_id", "")), applicants[0])
    print(LINE)
    print(f"STRESS / SCENARIO SENSITIVITY  ({mid.get('applicant_id')})")
    print(LINE)
    st = stress_applicant(mid)
    print(f"  baseline: PD {st['baseline']['pd']:.2%}  {st['baseline']['decision']}")
    for s in st["scenarios"]:
        flag = "  <-- decision changed" if s["decision_changed"] else ""
        print(f"  {s['scenario']:18s} PD {s['pd']:.2%} ({s['pd_delta']:+.2%}) "
              f"-> {s['decision']}{flag}")
    print()

    # RAG query
    print(LINE)
    print("RAG QUERY: 'What are the model governance and monitoring expectations?'")
    print(LINE)
    r = query("What are the model governance and monitoring expectations?", top_k=2)
    print(f"  retrieval mode: {r['retrieval_mode']}  ({r['embedder']})")
    for res in r["results"]:
        print(f"  [{res['rank']}] {res['source']} (score {res['score']})")
        print("      " + " ".join(res["text"].split())[:150] + "...")


if __name__ == "__main__":
    main()
