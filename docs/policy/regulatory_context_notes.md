# Regulatory Context Notes (PREPARED SUMMARY — not official text)

> **These are high-level, plain-language notes prepared for this prototype.**
> They are **not** quotations of, or substitutes for, any official regulation,
> and they do not state specific legal requirements. They name real frameworks
> only so a user knows which authoritative documents to obtain (see SOURCES.md)
> and drop into this folder for proper grounding. Always consult the official
> source for anything authoritative.

## Why this layer exists
A credit decision in the EU sits inside a web of supervisory expectations.
This prototype does not claim to implement any of them; it is structured so that
retrieved evidence can accompany a decision, which is the shape such frameworks
tend to encourage.

## Frameworks commonly relevant to retail credit decisioning
The following bodies and documents are the authoritative references a real
deployment would ground against. They are named here for orientation only.

- **European Banking Authority (EBA)** — guidelines on loan origination and
  monitoring; and separate guidance on the estimation of risk parameters such as
  probability of default and loss given default.
- **European Central Bank (ECB)** — supervisory guidance relevant to internal
  models used by significant institutions.
- **BaFin** — the German federal financial supervisory authority, relevant for
  institutions operating in Germany.
- **Basel framework** — the international standards underlying internal
  ratings-based approaches to credit risk capital.
- **GDPR** — includes provisions on automated individual decision-making and on
  a data subject's ability to obtain meaningful information about the logic
  involved.

## Themes this prototype deliberately mirrors
Without claiming compliance, the system is built around themes these frameworks
broadly share:
- creditworthiness and affordability should be assessed, not just a single score;
- decisions, and especially declines, should be explainable to the applicant;
- models should be governed, versioned, validated and monitored over time;
- protected characteristics should not drive decisions;
- decisions and their basis should be documented and auditable.

## What this prototype does NOT do
- It does not certify compliance with any of the above.
- It does not reproduce or paraphrase specific articles, clauses or thresholds
  from any regulation.
- It treats all loss-given-default and exposure figures as labelled assumptions.
