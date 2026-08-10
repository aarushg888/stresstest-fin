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

## Supported datasets / experiments

| study | status | location |
| --- | --- | --- |
| FICO HELOC (missing vs preserve-codes vs sentinel-aware) | completed | `configs/heloc*.yaml`, `artifacts/heloc_*` |
| Home Credit Default Risk (external validation, baseline vs `DAYS_EMPLOYED` sentinel-aware) | completed | `configs/home_credit*.yaml`, `artifacts/home_credit_*` |

See `docs/experiment_protocol.md` for the full methodology and
`docs/home_credit_results.md` for the Home Credit numbers.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
python -m stresstest_fin.run --config configs/example.yaml
pytest
```

Put a legally usable dataset at `data/raw/credit_risk.csv`; configure the target, ID, temporal, and optional protected-group columns in `configs/example.yaml`. Do not commit raw personal or restricted data.

### Home Credit (external validation)

Raw data is NOT committed (see `.gitignore`). Place it at:

```
data/raw/home_credit/application_train.csv
```

Then reproduce the pipeline:

```bash
python scripts/audit_home_credit.py                                  # Phase 1 audit
python scripts/convert_home_credit_sentinel_aware.py                 # Phase 4 transform
# Phase 3 baseline (5 seeds) + Phase 5 permutation stability
scripts/run_home_credit_seeds.sh configs/home_credit_baseline.yaml home_credit_baseline --skip-permutation
scripts/run_home_credit_seeds.sh configs/home_credit_baseline.yaml home_credit_baseline_permutation --n-repeats 3 --max-eval 2000
# Phase 4 sentinel-aware (5 seeds) + permutation
scripts/run_home_credit_seeds.sh configs/home_credit_sentinel_aware.yaml home_credit_sentinel_aware --skip-permutation
scripts/run_home_credit_seeds.sh configs/home_credit_sentinel_aware.yaml home_credit_sentinel_aware_permutation --n-repeats 3 --max-eval 2000
# Summaries
python scripts/summarize_home_credit.py --inputs "artifacts/home_credit_baseline_seed_*.json" --output artifacts/home_credit_baseline_summary.csv --label baseline
python scripts/summarize_home_credit.py --inputs "artifacts/home_credit_sentinel_aware_seed_*.json" --output artifacts/home_credit_sentinel_aware_summary.csv --label sentinel_aware
python scripts/compare_home_credit_variants.py --baseline artifacts/home_credit_baseline_summary.csv --sentinel artifacts/home_credit_sentinel_aware_summary.csv --output artifacts/home_credit_baseline_vs_sentinel_aware.csv
python scripts/summarize_home_credit_permutation.py --inputs "artifacts/home_credit_*_permutation_seed_*.json"
```

## Suggested commit sequence
1. `chore: scaffold reproducible research repository`
2. `feat: add schema validation and dataset cards`
3. `feat: add temporal split and baseline models`
4. `feat: add calibration and fairness evaluation`
5. `feat: add explanation stability audit`
6. `docs: add findings, limitations, and reproducibility guide`

## Limits
Public credit datasets may contain historical bias, missing context, and imperfect labels. Metrics here do not establish legal compliance or fairness. Never infer protected attributes; evaluate them only where lawfully available and appropriate.
