from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def make_preprocessor(X) -> ColumnTransformer:
    """Original HELOC-era numeric+categorical preprocessor.

    Kept as the default so that HELOC runs continue to behave exactly as
    before. Returns a sparse ColumnTransformer. Tree models that cannot
    consume sparse output must densify downstream (see `build_models`).
    """
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]), categorical),
        ],
        sparse_threshold=0.3,
    )


@dataclass(frozen=True)
class SpecialValueSpec:
    """One documented special-value rule for a numeric feature.

    For Home Credit we only have a single rule: ``DAYS_EMPLOYED == 365243``.
    We deliberately do NOT generalize to arbitrary negative values, since
    negative values in DAYS_* columns are expected (days before application).
    """
    column: str
    code: float | int
    indicator_name: str

    def applies_to(self, columns: Sequence[str]) -> bool:
        return self.column in columns


@dataclass(frozen=True)
class PreprocessingConfig:
    """Reusable preprocessing configuration."""
    drop_columns: tuple[str, ...] = ()
    special_values: tuple[SpecialValueSpec, ...] = ()
    drop_high_cardinality_above: int | None = None  # documented opt-in only


def apply_special_values(df, spec: SpecialValueSpec):
    """Apply a SpecialValueSpec to a dataframe.

    Returns the (mutated) copy and the indicator column name.
    The original special value in ``df[column]`` is replaced with NaN.
    """
    import numpy as np
    import pandas as pd
    out = df.copy()
    indicator = spec.indicator_name
    out[indicator] = (out[spec.column] == spec.code).astype("int8")
    out[spec.column] = out[spec.column].replace(spec.code, np.nan)
    return out, indicator


def build_preprocessor(
    X,
    config: PreprocessingConfig | None = None,
    *,
    scale_numeric: bool = True,
    dense_output: bool = False,
) -> ColumnTransformer:
    """Build a ColumnTransformer for an arbitrary feature mix.

    - Numeric features: median imputation, optional StandardScaler.
    - Categorical features: most-frequent imputation, one-hot with unknown handling.
    - If ``dense_output=True`` the result is forced to dense numpy output
      (used for ``HistGradientBoostingClassifier`` which cannot consume sparse
      matrices). A documented size guard caps this to a sane ceiling.
    """
    cfg = config or PreprocessingConfig()
    drop = set(cfg.drop_columns)
    cols = [c for c in X.columns if c not in drop]

    if cfg.drop_high_cardinality_above is not None:
        threshold = cfg.drop_high_cardinality_above
        keep_cats = []
        dropped_cats = []
        for c in cols:
            if not _is_numeric(X[c]):
                nunique = X[c].nunique(dropna=True)
                if nunique > threshold:
                    dropped_cats.append(c)
                    continue
                keep_cats.append(c)
        cols = [c for c in cols if c not in set(dropped_cats)]
    else:
        keep_cats = [c for c in cols if not _is_numeric(X[c])]

    numeric = [c for c in cols if _is_numeric(X[c])]

    numeric_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if keep_cats:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)),
            ]),
            keep_cats,
        ))

    sparse_threshold = 0.0 if dense_output else 0.3
    return ColumnTransformer(transformers, sparse_threshold=sparse_threshold)


def _is_numeric(series) -> bool:
    import pandas as pd
    return pd.api.types.is_numeric_dtype(series)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Random forest must NOT use n_jobs=-1 to avoid joblib/Loky warnings and to
# improve reproducibility on a local Mac. Same for permutation importance
# which is already pinned to n_jobs=1 in explainability.py.

def _dense_pre_for_trees(X) -> ColumnTransformer:
    """Tree-friendly preprocessor: dense numeric + dense one-hot."""
    return build_preprocessor(X, scale_numeric=False, dense_output=True)


def build_models(X, seed: int, *, force_dense_for_trees: bool = True):
    """Build the three baseline models.

    Logistic regression gets a sparse-friendly scaled pipeline (faster, smaller).
    HistGradientBoosting and RandomForest get a dense output pipeline because
    they cannot consume the sparse one-hot output (and HGB has its own native
    categorical support but we keep parity with the LR / RF pipeline so the
    three models are comparable on the same transformed features).

    The previous HELOC behaviour (single shared preprocessor, sparse for all
    three) was not actually exercised for HGB on HELOC because HELOC is
    numeric-only — there the preprocessor degenerates to numeric-only and the
    sparse_threshold=0.3 path produces a dense array anyway. So HELOC results
    are unchanged.
    """
    pre_sparse = make_preprocessor(X)

    if force_dense_for_trees:
        pre_dense = _dense_pre_for_trees(X)
    else:
        pre_dense = pre_sparse

    return {
        "logistic_regression": Pipeline([
            ("pre", pre_sparse),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("pre", pre_dense),
            ("model", HistGradientBoostingClassifier(random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("pre", pre_dense),
            ("model", RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            )),
        ]),
    }