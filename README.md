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
| Home Credit Phase 2 (enriched aux tables, temporal recency split, LR calibration, HGB balance) | completed | `configs/home_credit_{enriched,temporal}.yaml`, `artifacts/home_credit_{enriched,temporal,cal,hgb}*` |

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

## Key Results (Home Credit Default Risk)

| Phase | Models | AUC (mean) | AP (mean) | Brier (mean) | ECE-10 (mean) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Raw features, stratified (5 seeds) | LR / HGB / RF | 0.741 / 0.744 / 0.737 | 0.215 / 0.222 / 0.211 | 0.203 / 0.069 / 0.117 | 0.337 / 0.003 / 0.209 | Baseline |
| Raw + sentinel-aware (5 seeds) | LR / HGB / RF | 0.741 / 0.743 / 0.736 | 0.215 / 0.220 / 0.209 | 0.202 / 0.069 / 0.117 | 0.336 / 0.002 / 0.209 | ΔAUC < 0.002 |
| **Enriched (aux tables, 5 seeds)** | **LR / HGB / RF** | **0.758 / 0.759 / 0.750** | **0.231 / 0.239 / 0.222** | **0.194 / 0.068 / 0.100** | **0.319 / 0.004 / 0.171** | **+0.013–0.017 AUC** |
| Enriched + temporal (5 seeds) | LR / HGB / RF | 0.758 / 0.757 / 0.749 | 0.248 / 0.254 / 0.239 | 0.207 / 0.074 / 0.108 | 0.327 / 0.005 / 0.177 | Drift: Brier/ECE ↑ |
| **Enriched + LR isotonic cal. (5 seeds)** | **LR** | **0.758** | **0.231** | **0.068** | **0.002** | **ECE 0.319→0.002** |
| **Enriched full-scale (2 seeds, 230k)** | **LR / HGB / RF** | **0.765 / 0.773 / 0.760** | **0.241 / 0.259 / 0.233** | **0.195 / 0.067 / 0.099** | **0.325 / 0.002 / 0.169** | **+0.007–0.013 AUC** |

**Takeaways:**
- Auxiliary-table enrichment is the single biggest lever (+0.013–0.017 AUC).
- Temporal split holds AUC but degrades calibration (expected drift signature).
- Isotonic calibration fixes LR's ECE (0.32→0.002) at zero AUC cost — LR becomes best-calibrated model.
- Full-scale training (230k rows) adds +0.007–0.013 AUC; HGB reaches 0.773 AUC, near published full-data baselines.
- The `DAYS_EMPLOYED==365243` sentinel indicator has negligible predictive effect on top of enrichment.

See `docs/home_credit_results.md` for full tables with ±std and artifact paths.

## Suggested commit sequence
1. `chore: scaffold reproducible research repository`
2. `feat: add schema validation and dataset cards`
3. `feat: add temporal split and baseline models`
4. `feat: add calibration and fairness evaluation`
5. `feat: add explanation stability audit`
6. `docs: add findings, limitations, and reproducibility guide`

## Limits
Public credit datasets may contain historical bias, missing context, and imperfect labels. Metrics here do not establish legal compliance or fairness. Never infer protected attributes; evaluate them only where lawfully available and appropriate.
