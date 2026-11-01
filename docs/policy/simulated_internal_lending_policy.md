# SIMULATED Internal Retail Lending Policy

> **THIS IS A SIMULATED / FICTIONAL DOCUMENT.**
> It was written for this prototype to give the retrieval layer something
> concrete and internally consistent to cite. It is **not** the policy of any
> real bank and carries no regulatory authority. Decision rules below are
> illustrative and aligned to this prototype's configuration.

## 1. Scope
This policy governs automated and semi-automated decisioning for unsecured
retail consumer loans. It applies to the credit decision engine and the staff
who review its output.

## 2. Probability of Default (PD) and decision bands
- Every application receives a calibrated Probability of Default (PD) from the
  approved model. PD must be a calibrated probability, not an uncalibrated score.
- Decision bands:
  - **Approve**: PD below the approval threshold defined by the current
    cost-based policy.
  - **Refer to manual review**: PD in the intermediate band.
  - **Decline**: PD above the decline threshold.
- Thresholds are derived from a break-even argument comparing expected interest
  margin against expected credit loss, and are reviewed quarterly.

## 3. Affordability and leverage
- The debt-service burden is assessed using the annuity-to-income ratio and the
  total-credit-to-income ratio.
- Applications with very high leverage are referred to manual review even when
  the model PD is within the approval band.

## 4. Expected Loss
- Expected Loss is computed as PD x LGD x EAD.
- Where loss-given-default and exposure-at-default cannot be estimated from
  internal data, documented portfolio assumptions are used and must be labelled
  as assumptions in the decision record.

## 5. Explainability and adverse-action
- Every automated decision must be accompanied by the principal factors that
  drove it, expressed in terms the applicant can understand.
- For a decline, the applicant is entitled to the specific principal reasons for
  the decision (adverse-action reasons).
- Where feasible, the applicant should be told what realistic change to the
  requested loan terms could lead to a different outcome.

## 6. Fairness and protected attributes
- Protected attributes (including gender) must not be used as model inputs.
- The portfolio is monitored for disparate outcomes across groups; material
  disparities trigger investigation.

## 7. Model governance and monitoring
- Models are version-controlled; every decision records the model version used.
- Model calibration is checked on a held-out sample before deployment.
- Population Stability Index (PSI) is monitored on key inputs and on the score;
  a PSI above 0.25 triggers review and possible retraining.

## 8. Human oversight
- Applications in the review band, and any decline the applicant contests, are
  reviewed by a credit officer who may override the automated decision with a
  documented rationale.

## 9. Record-keeping
- Each decision is retained with: model version, calibrated PD, decision, the
  principal factors, any counterfactual guidance, the risk metrics and their
  assumptions, and the evidence retrieved in support of the decision.
