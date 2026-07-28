# Research Plan

## Claim boundary
This project does not claim a new lending model or legal fairness determination. Its contribution is a reproducible protocol for assessing predictive performance, calibration, group metrics, and prediction sensitivity under an explicitly documented distribution shift.

## Phases
1. **Literature and gap map (1-2 weeks):** collect 15-25 primary papers; record exact task, dataset, split, metrics, and limitations.
2. **Dataset governance (1 week):** choose 2-3 real, license-compatible datasets with temporal fields where possible; create a dataset card for each.
3. **Baselines (2 weeks):** logistic regression, gradient boosting, random forest; locked preprocessing; leakage review.
4. **Shift protocol (2 weeks):** temporal holdout first; geographic/population shift only when dataset permits; no unsupported claims.
5. **Trust evaluation (2 weeks):** discrimination, calibration, group metrics, prediction sensitivity; bootstrap confidence intervals as a later extension.
6. **Paper and release (2 weeks):** write results honestly, release code/configs/model cards, record all failed experiments.

## Success criteria
- Two or more real datasets, each with a dataset card and reproducible acquisition instructions
- All models evaluated with a locked protocol and uncertainty estimates
- At least one non-obvious finding, including a negative result if supported
- External review by a teacher, professor, or practitioner

## Paper outline
Abstract; Introduction; Related Work; Data and Governance; Evaluation Protocol; Models; Results; Failure Analysis; Limitations/Ethics; Reproducibility.
