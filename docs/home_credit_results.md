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

---

# Phase 2 — Enriched features + temporal split + calibration

Research/education only. Phase 2 adds the Home Credit auxiliary tables
(previous_application, bureau, bureau_balance, credit_card_balance,
installments_payments, POS_CASH_balance) aggregated per customer
(`scripts/build_home_credit_features.py`), giving 171 new features
(307,511 × 293). Raw tables downloaded via
`scripts/download_home_credit_tables.py` (KAGGLE_API_TOKEN required).

## Enriched — aggregate (seeds 11/22/33/44/55, 60k train cap)

Artifact: `artifacts/home_credit_enriched_summary.csv`

| model | ROC-AUC (mean±std) | AP (mean±std) | Brier (mean±std) | ECE-10 (mean±std) | pred-sens (mean±std) |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.7582±0.0032 | 0.2310±0.0032 | 0.1942±0.0010 | 0.3194±0.0020 | 0.0169±0.0078 |
| hist_gradient_boosting | 0.7594±0.0019 | 0.2389±0.0020 | 0.0680±0.0001 | 0.0037±0.0003 | 0.0161±0.0073 |
| random_forest | 0.7501±0.0033 | 0.2216±0.0027 | 0.1002±0.0004 | 0.1713±0.0007 | 0.0295±0.0167 |

## Enriched sentinel-aware — aggregate

Artifact: `artifacts/home_credit_enriched_sentinel_aware_summary.csv`

| model | ROC-AUC (mean±std) | AP (mean±std) | Brier (mean±std) | ECE-10 (mean±std) | pred-sens (mean±std) |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.7585±0.0030 | 0.2315±0.0029 | 0.1942±0.0010 | 0.3192±0.0018 | 0.0035±0.0024 |
| hist_gradient_boosting | 0.7594±0.0024 | 0.2384±0.0029 | 0.0680±0.0001 | 0.0043±0.0010 | 0.0175±0.0106 |
| random_forest | 0.7500±0.0032 | 0.2209±0.0030 | 0.1003±0.0004 | 0.1715±0.0008 | 0.0292±0.0210 |

## Enriched vs raw-feature deltas (enriched − baseline)

| model | Δ ROC-AUC | Δ AP | Δ Brier | Δ ECE-10 |
| --- | --- | --- | --- | --- |
| logistic_regression | +0.0171 | +0.0163 | −0.0086 | −0.0179 |
| hist_gradient_boosting | +0.0158 | +0.0166 | −0.0008 | +0.0011 |
| random_forest | +0.0128 | +0.0107 | −0.0172 | −0.0380 |

The auxiliary tables add real signal: AUC improves +0.013 to +0.017 across
models, AP +0.011 to +0.017, and Brier/ECE improve for LR and RF. The
sentinel-aware indicator again has negligible effect on top of enrichment.

## Temporal split (recency proxy) — aggregate (seeds 11/22/33/44/55)

Artifact: `artifacts/home_credit_temporal_summary.csv`
Split: `chronological` on `prev_APP_RECENCY_DAYS` (max DAYS_DECISION per
customer = most recent previous-application decision, days before the
current application). Train = 230,634 earliest rows, test = 76,877 most
recent rows. Target rate: train 7.79%, test 8.94% — real distribution drift.

| model | ROC-AUC (mean±std) | AP (mean±std) | Brier (mean±std) | ECE-10 (mean±std) | pred-sens (mean±std) |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.7578±0.0018 | 0.2484±0.0029 | 0.2070±0.0044 | 0.3268±0.0060 | 0.0220±0.0093 |
| hist_gradient_boosting | 0.7574±0.0026 | 0.2537±0.0038 | 0.0743±0.0003 | 0.0054±0.0015 | 0.0199±0.0065 |
| random_forest | 0.7491±0.0023 | 0.2387±0.0035 | 0.1080±0.0006 | 0.1771±0.0018 | 0.0244±0.0084 |

Under the temporal split, AUC holds within ~0.001-0.003 of the random-split
enriched numbers (no catastrophic degradation), AP is higher (test has more
positives), but Brier/ECE degrade slightly for all models — the models are
less well-calibrated on the shifted test distribution. This is the expected
signature of mild distribution drift, not a failure.

## LR calibration (Platt/isotonic) — aggregate (seeds 11/22/33/44/55)

Artifacts: `artifacts/home_credit_enriched_cal_sigmoid_summary.csv`,
`artifacts/home_credit_enriched_cal_isotonic_summary.csv`
Applied via CalibratedClassifierCV(cv=3) on the enriched LR pipeline.
AUC is unchanged (calibration is rank-preserving).

| method | ROC-AUC | AP | Brier (mean±std) | ECE-10 (mean±std) |
| --- | --- | --- | --- | --- |
| none (enriched LR) | 0.7582 | 0.2310 | 0.1942±0.0010 | 0.3194±0.0020 |
| sigmoid (Platt) | 0.7583 | 0.2308 | 0.0685±0.0004 | 0.0133±0.0069 |
| isotonic | 0.7584 | 0.2314 | 0.0681±0.0002 | 0.0023±0.0006 |

Both methods fix LR calibration dramatically: Brier drops from 0.194 to
~0.068 (matching HGB) and ECE from 0.319 to 0.013 (sigmoid) / 0.002
(isotonic, matching HGB). Isotonic is the best-calibrating method here.

## HGB balanced sample weights — aggregate

Artifact: `artifacts/home_credit_enriched_hgb_balanced_summary.csv`

| variant | ROC-AUC | AP | Brier (mean±std) | ECE-10 (mean±std) |
| --- | --- | --- | --- | --- |
| HGB plain | 0.7594 | 0.2389 | 0.0680±0.0001 | 0.0037±0.0003 |
| HGB balanced | 0.7613 | 0.2423 | 0.1717±0.0030 | 0.2922±0.0066 |

Balanced sample weights improve ranking slightly (AUC +0.002, AP +0.003)
but destroy calibration (Brier 0.068→0.172, ECE 0.004→0.292): the weights
shift the predicted probability distribution away from the true base rate.
If used, must be paired with recalibration.

## Phase 2 interpretation notes

1. **Enrichment is the single biggest lever** found so far (+0.013 to
   +0.017 AUC). The auxiliary tables carry signal that the application
   table alone does not.
2. **Temporal split exposes mild drift**: AUC holds, calibration degrades.
   The recency proxy is a documented approximation of true calendar time.
3. **LR calibration is a near-free win**: isotonic recalibration makes LR
   the best-calibrated model (ECE 0.002) at zero AUC cost.
4. **Balanced HGB weights trade calibration for ranking** — not a
   free lunch without recalibration.
5. Full-scale (no 60k cap) results appended when complete
   (`artifacts/home_credit_enriched_fullscale_seed_*.json`).

## Full-scale (no training cap) — seeds 11/22

Artifacts: `artifacts/home_credit_enriched_fullscale_seed_11.json`,
`artifacts/home_credit_enriched_fullscale_seed_22.json`
Training: all available rows after chronological/temporal split
(230,633 rows for these seeds). No permutation (--skip-permutation).

| seed | model | ROC-AUC | AP | Brier | ECE-10 |
| --- | --- | --- | --- | --- | --- |
| 11 | logistic_regression | 0.7677 | 0.2411 | 0.1950 | 0.3264 |
| 11 | hist_gradient_boosting | 0.7726 | 0.2581 | 0.0670 | 0.0015 |
| 11 | random_forest | 0.7588 | 0.2319 | 0.0993 | 0.1694 |
| 22 | logistic_regression | 0.7629 | 0.2398 | 0.1940 | 0.3235 |
| 22 | hist_gradient_boosting | 0.7730 | 0.2607 | 0.0669 | 0.0032 |
| 22 | random_forest | 0.7602 | 0.2344 | 0.0991 | 0.1689 |

| model | mean AUC | mean AP | mean Brier | mean ECE-10 | vs 60k cap (Δ AUC) |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.7653 | 0.2405 | 0.1945 | 0.3250 | +0.0071 |
| hist_gradient_boosting | 0.7728 | 0.2594 | 0.0670 | 0.0024 | +0.0134 |
| random_forest | 0.7595 | 0.2332 | 0.0992 | 0.1692 | +0.0094 |

Removing the 60k training cap adds 3.8× more training data (230k vs 60k)
and yields consistent AUC gains (+0.007 to +0.013). HGB at 0.773 AUC
approaches published full-data baselines (~0.78-0.80). LR calibration
remains poor (ECE ~0.32) without explicit calibration; see Phase 2
calibration results.

---

*End of Phase 2. All experimental sections complete.*
