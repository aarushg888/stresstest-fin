# Home Credit Default Risk — External Validation Results

Research/education only. Not suitable for any real lending or automated
decision. All numbers below come from actual local runs; artifact paths
are listed next to every table.

## Setup

- Dataset: `data/raw/home_credit/application_train.csv` (307,511 × 122)
- Variants:
  - `baseline`: raw table, standard preprocessing (median/most-frequent imputation + one-hot)
  - `sentinel_aware`: `DAYS_EMPLOYED__was_special` indicator + `DAYS_EMPLOYED == 365243` → NaN
- Splits: stratified random, `test_size=0.25`, seeds 11/22/33/44/55 (+ smoke 42)
- Training rows capped at 60,000 (stratified, fixed per seed).
  Test metrics on the full 76,878-row test partition.
- Permutation stability: deterministic 2,000-row test subsample (3,000 for the
  seed-42 smoke), `n_repeats=3` (5 for smoke), `top_k=10`, ROC-AUC scoring, `n_jobs=1`.
- Models: logistic_regression, hist_gradient_boosting, random_forest (see `docs/experiment_protocol.md`).

## Metrics legend

| metric | better |
| --- | --- |
| ROC-AUC | higher |
| average precision | higher |
| Brier | lower |
| ECE-10 | lower |
| top-k Jaccard (permutation) | higher |
| mean abs importance change | lower |
| prediction-sensitivity proxy | lower |

## Baseline (seed 42 smoke test)

Artifact: `artifacts/home_credit_baseline_seed_42.json`
(eval sample 3,000 rows, n_repeats=5)

| model | ROC-AUC | AP | Brier | ECE-10 | pred-sens |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.7433 | 0.2176 | 0.2046 | 0.3412 | 0.0069 |
| hist_gradient_boosting | 0.7440 | 0.2278 | 0.0686 | 0.0018 | 0.0082 |
| random_forest | 0.7406 | 0.2156 | 0.1177 | 0.2117 | 0.0160 |

## Baseline — aggregate (seeds 11/22/33/44/55)

Artifact: `artifacts/home_credit_baseline_summary.csv` (raw: `..._raw.csv`)

| model | ROC-AUC (mean±std) | AP (mean±std) | Brier (mean±std) | ECE-10 (mean±std) | pred-sens (mean±std) |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.7411±0.0024 | 0.2147±0.0037 | 0.2028±0.0011 | 0.3373±0.0024 | 0.0150±0.0088 |
| hist_gradient_boosting | 0.7436±0.0023 | 0.2223±0.0039 | 0.0688±0.0002 | 0.0026±0.0010 | 0.0098±0.0020 |
| random_forest | 0.7373±0.0022 | 0.2109±0.0038 | 0.1174±0.0004 | 0.2093±0.0015 | 0.0159±0.0014 |

## Sentinel-aware — aggregate (seeds 11/22/33/44/55)

Artifact: `artifacts/home_credit_sentinel_aware_summary.csv`

| model | ROC-AUC (mean±std) | AP (mean±std) | Brier (mean±std) | ECE-10 (mean±std) | pred-sens (mean±std) |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.7413±0.0024 | 0.2150±0.0039 | 0.2023±0.0007 | 0.3361±0.0017 | 0.0029±0.0032 |
| hist_gradient_boosting | 0.7430±0.0019 | 0.2202±0.0031 | 0.0689±0.0001 | 0.0023±0.0010 | 0.0070±0.0015 |
| random_forest | 0.7364±0.0018 | 0.2093±0.0032 | 0.1174±0.0005 | 0.2088±0.0012 | 0.0087±0.0008 |

## Baseline vs sentinel-aware deltas (sentinel − baseline)

Artifact: `artifacts/home_credit_baseline_vs_sentinel_aware.csv`

| model | Δ ROC-AUC | Δ AP | Δ Brier | Δ ECE-10 | Δ pred-sens |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | +0.0002 | +0.0003 | −0.0005 | −0.0011 | −0.0121 |
| hist_gradient_boosting | −0.0006 | −0.0021 | +0.0001 | −0.0004 | −0.0029 |
| random_forest | −0.0009 | −0.0016 | +0.0000 | −0.0005 | −0.0072 |

Predictive deltas are within cross-seed noise (±0.002). The clearest effect is on
the prediction-sensitivity proxy: sentinel-aware logistic regression drops from
0.0150 to 0.0029 (−0.0121), and RF from 0.0159 to 0.0087 (−0.0072).

## Permutation-importance stability

Artifacts:
- `artifacts/home_credit_permutation_stability_by_seed.csv`
- `artifacts/home_credit_permutation_stability_summary.csv`

| variant | model | top-10 Jaccard (mean±std) | mean abs importance change (mean±std) |
| --- | --- | --- | --- |
| baseline | logistic_regression | 0.8606±0.1415 | 0.0016±0.0003 |
| baseline | hist_gradient_boosting | 0.6970±0.0678 | 0.0006±0.0001 |
| baseline | random_forest | 0.6713±0.0991 | 0.0006±0.0001 |
| sentinel_aware | logistic_regression | 0.8242±0.1181 | 0.0015±0.0002 |
| sentinel_aware | hist_gradient_boosting | 0.6760±0.1400 | 0.0006±0.0001 |
| sentinel_aware | random_forest | 0.6237±0.1474 | 0.0005±0.0001 |

Model ordering (LR most stable, then HGB, then RF) matches HELOC. Absolute
Jaccard values are lower than HELOC (LR 1.000, HGB 0.939, RF 0.884) because
Home Credit has ~250 transformed features vs 87 for HELOC, and n_repeats is 3
vs 10. The sentinel-aware variant shows slightly lower Jaccard than baseline
for all three models; the differences are within cross-seed std, so we do not
claim sentinel-aware improves permutation stability here.

## Interpretation notes

1. **External validation reproduces the HELOC ranking**: HGB has the best
   calibration (ECE ≈ 0.002-0.003) and best overall balance (AUC 0.744,
   Brier 0.069); LR and RF are poorly calibrated on this imbalanced target
   (ECE 0.21-0.34) under `class_weight="balanced"` without calibration.
2. **The DAYS_EMPLOYED sentinel indicator has negligible predictive impact**
   (all deltas within ±0.002 AUC). It is a single feature among ~120; the
   ablation is therefore a weak test of the sentinel-aware hypothesis. The
   effect is clearest on the prediction-sensitivity proxy for LR.
3. **AUC ~0.74 on a 60k training subsample** is below the ~0.78-0.80 that
   gradient-boosting entries reach on the full 307k training table; do not
   compare these absolute numbers to published full-data results.
4. **Random stratified split, not temporal validation** — see limitations in
   `docs/experiment_protocol.md`.
