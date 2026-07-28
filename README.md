# StressTest-Fin

An open research toolkit for evaluating whether financial-risk models remain **accurate, calibrated, fair, and explanation-stable** under dataset shift. It is for research and education only, and must not be used to make lending, investment, employment, insurance, or other high-impact decisions.

## Research question
When a credit-risk model is transferred across a time period or defined population shift, how do discrimination, calibration, fairness, and explanation stability change?

## What this repository does
- Loads a tabular credit-risk dataset from a local CSV
- Applies an explicit chronological split if a time column is available; otherwise uses a documented proxy split
- Trains logistic regression, histogram gradient boosting, and random forest baselines
- Reports ROC-AUC, average precision, Brier score, expected calibration error, and threshold metrics
- Audits group-level error-rate gaps when a protected-group column is explicitly provided
- Measures local explanation stability through perturbation tests
- Saves a versioned experiment report and a model card

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
python -m stresstest_fin.run --config configs/example.yaml
pytest
```

Put a legally usable dataset at `data/raw/credit_risk.csv`; configure the target, ID, temporal, and optional protected-group columns in `configs/example.yaml`. Do not commit raw personal or restricted data.

## Suggested commit sequence
1. `chore: scaffold reproducible research repository`
2. `feat: add schema validation and dataset cards`
3. `feat: add temporal split and baseline models`
4. `feat: add calibration and fairness evaluation`
5. `feat: add explanation stability audit`
6. `docs: add findings, limitations, and reproducibility guide`

## Limits
Public credit datasets may contain historical bias, missing context, and imperfect labels. Metrics here do not establish legal compliance or fairness. Never infer protected attributes; evaluate them only where lawfully available and appropriate.
