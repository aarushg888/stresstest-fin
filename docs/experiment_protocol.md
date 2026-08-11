# Experiment Protocol

Research/education only. None of these models are used for any real lending
or automated decision. This document describes the methodology shared
across the FICO HELOC and Home Credit Default Risk studies in this repo.

## Research objective

Quantify how the choice of representation for missing / sentinel values in
credit-risk tabular data affects three downstream properties:

1. **Predictive performance** — ROC-AUC, average precision, Brier score.
2. **Calibration** — ECE-10 (equal-width bins).
3. **Explanation stability** — top-k Jaccard similarity and mean absolute
   importance change of permutation-importance rankings before and after
   a small synthetic noise perturbation.

We deliberately treat explanation stability as an **exploratory** measure:
small-perturbation permutation stability is not a complete account of
explanation quality.

## HELOC preprocessing variants

| variant | description |
| --- | --- |
| `A_missing` | FICO special codes `-9`, `-8`, `-7` → NaN; no indicators. |
| `B_preserve_codes` | Special codes retained as numeric values; no indicators. |
| `C_sentinel_aware` | For every numeric feature and each code (`-9`, `-8`, `-7`), add a `__was_<code>` indicator (int8), then replace the special code with NaN. |

Conversion scripts:
- `scripts/convert_heloc.py`            → variant A
- `scripts/convert_heloc_preserve_codes.py` → variant B
- `scripts/convert_heloc_sentinel_aware.py` → variant C

Per-variant configs: `configs/heloc*.yaml`.
Per-seed configs:    `configs/heloc_seed_<seed>.yaml`,
                     `configs/heloc_sentinel_aware_seed_<seed>.yaml`,
                     `configs/heloc_preserve_codes_seed_<seed>.yaml`.

## Home Credit external validation

- Raw: `data/raw/home_credit/application_train.csv` (not committed; see `.gitignore`).
- Rows: 307,511; columns: 122 (123 after sentinel-aware indicator).
- Target: `TARGET` ∈ {0, 1}, positive prevalence ≈ 0.0807.
- ID column: `SK_ID_CURR` (excluded from features).
- Documented sentinel: `DAYS_EMPLOYED == 365243` (≈ 18 % of rows). All
  other negative values are normal (days before application) and are NOT
  treated as sentinels.

Variants:
| variant | description |
| --- | --- |
| `baseline` | Pass the table through unchanged; the model pipeline imputes missing values normally. |
| `sentinel_aware` | Add `DAYS_EMPLOYED__was_special` indicator; replace `365243` with NaN. |

Conversion: `scripts/convert_home_credit_sentinel_aware.py`.
Configs:    `configs/home_credit_baseline.yaml`,
            `configs/home_credit_sentinel_aware.yaml`.

## Home Credit Phase 2 — enriched features + temporal split + calibration

Phase 2 (added after the initial external validation) extends the study:

1. **Feature enrichment** — the six auxiliary tables (previous_application,
   bureau, bureau_balance, credit_card_balance, installments_payments,
   POS_CASH_balance) are aggregated per customer (count, mean/max/min/sum/std
   on informative numeric columns, nunique on categoricals) and left-joined
   into the application table → 307,511 × 293.
   - Scripts: `scripts/download_home_credit_tables.py` (requires
     `KAGGLE_API_TOKEN` env var — the modern KGAT_ token is Bearer-auth and
     the legacy kaggle.json username/key Basic-auth is NOT accepted by
     kagglehub 1.0.2 for this competition), `scripts/build_home_credit_features.py`.
   - Configs: `configs/home_credit_enriched.yaml`,
     `configs/home_credit_enriched_sentinel_aware.yaml`.
2. **Temporal (recency-proxy) split** — a `chronological` split on
   `prev_APP_RECENCY_DAYS` (max DAYS_DECISION per customer = most recent
   previous-application decision, days before the current application).
   Earliest 75% train, latest 25% test; rows with no previous applications
   (NaN recency) sort first and land in train. This is a recency ordering,
   NOT true calendar time, because all DAYS_* values are relative to each
   customer's own application date. Config: `configs/home_credit_temporal.yaml`.
   Split support: `data.split_data` handles numeric time columns.
3. **LR calibration** — `--calibrate sigmoid|isotonic` wraps LR in
   `CalibratedClassifierCV(cv=3)`. Rank-preserving (AUC unchanged).
4. **HGB balanced sample weights** — `--hgb-balanced-weights` passes
   inverse-frequency sample weights to HistGradientBoostingClassifier.

## Five-seed protocol

For every (variant, model) pair we run a stratified random train/test
split with `test_size = 0.25` and `random_state ∈ {11, 22, 33, 44, 55}`.
This is a *random* split, not a temporal split.

For Home Credit, training rows are stratified-subsampled to a fixed cap
per seed (see "Runtime constraints" below). The cap and seed are
recorded in every output JSON.

## Models

All three pipelines share the same preprocessing logic so results are
comparable across models.

| model | preprocessing | notes |
| --- | --- | --- |
| `logistic_regression` | numeric: median impute + StandardScaler; categorical: most-frequent + one-hot (sparse) | `class_weight="balanced"`, `max_iter=2000` |
| `hist_gradient_boosting` | numeric: median impute; categorical: most-frequent + one-hot (dense) | sklearn HGB |
| `random_forest` | numeric: median impute; categorical: most-frequent + one-hot (dense) | 400 trees, `min_samples_leaf=5`, `class_weight="balanced"`, `n_jobs=1` |

Tree models use a dense one-hot output because `HistGradientBoostingClassifier`
does not accept sparse matrices. Numeric scaling is dropped for trees.

## Metrics and direction of improvement

| metric | direction |
| --- | --- |
| ROC-AUC | higher is better |
| average precision | higher is better |
| Brier score | lower is better |
| ECE-10 | lower is better |
| top-k Jaccard stability (permutation importance) | higher is better |
| mean absolute importance change | lower is better |
| prediction-sensitivity proxy (mean abs prob change) | lower is better |

## Runtime constraints (research-safe reductions)

- **Training subsample** for Home Credit: stratified sample of the
  training partition, fixed per seed, recorded in the output JSON.
  Rationale: fitting 3 sklearn models on 230k × ~250-feature data on a
  single Mac is the bottleneck; the cap keeps runs reproducible and
  bounded.
- **Evaluation subsample** for permutation importance and the
  prediction-sensitivity proxy: a fixed deterministic sample of the
  test partition (3,000 rows for the seed-42 smoke run; 2,000 rows for
  the formal 5-seed sweep), recorded in the output JSON. Same sample
  across models and seeds.
- **Permutation repeats**: `n_repeats=5` for the seed-42 smoke run;
  `n_repeats=3` for the formal 5-seed sweep (reduced from HELOC's 10
  because Home Credit has far more features). Same `n_repeats` across
  models and seeds within each phase.
- **`n_jobs=1`** for both random forest and permutation importance to
  avoid joblib/Loky warnings and improve reproducibility.

## Limitations (must be reported)

1. **Random stratified splitting is not temporal validation.** HELOC and
   the initial Home Credit runs use i.i.d. stratified splits. Phase 2 adds
   a Home Credit temporal (recency-proxy) split, which is a documented
   approximation of true calendar time — all DAYS_* values are relative to
   each customer's own application date, so the proxy orders by recency of
   previous-application activity rather than absolute date. Real production
   decisions face distribution shift; the temporal results estimate
   performance under mild drift but should not be over-generalized.
2. **Both datasets concern high-impact credit decisions and are
   research-only.** No model output here is suitable for any real
   lending or automated decision.
3. **No fairness claim is made.** Valid fairness analysis requires
   justified protected-group variables and a rigorous methodology.
   None of the configs set `protected_group_column`, so no group
   audit is produced.
4. **Permutation stability under small synthetic perturbations is not
   a complete measure of explanation quality.** It captures one
   dimension of stability and should not be over-interpreted.
5. **External validation on one additional benchmark is not universal
   generalization.** Adding Home Credit strengthens the case but does
   not establish that sentinel-aware preprocessing generalizes to all
   credit-risk datasets.

## Reproducibility checklist

- `random_state` set on every model and every random source.
- `np.random.default_rng(seed)` used for subsampling.
- All output JSONs contain `config_path`, `n_train_used`, `n_eval_used`,
  `max_train_cap`, `max_eval_cap`.
- Seed-list sweep: 11, 22, 33, 44, 55 (and 42 for the smoke test).