"""Reproducible audit for the Home Credit Default Risk application_train.csv.

Research/education only. Never the basis of a real lending decision.

Reports:
  - shape and target distribution
  - dtype / numeric vs categorical counts
  - top-30 columns by missingness
  - categorical cardinality
  - constant / near-constant columns
  - DAYS_EMPLOYED == 365243 special-value count (documented Home Credit sentinel)
  - generic negative-value count in numeric columns (NOT assumed to be sentinel)
  - saves JSON summary and CSV tables to artifacts/

Usage:
    python scripts/audit_home_credit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "data" / "raw" / "home_credit" / "application_train.csv"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def audit_application_train(csv_path: Path = DEFAULT_PATH) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected Home Credit file at {csv_path}. Place application_train.csv "
            f"there before running this audit."
        )

    df = pd.read_csv(csv_path)

    n_rows, n_cols = df.shape

    target_col = "TARGET"
    target = df[target_col].astype(int)
    target_counts = target.value_counts().to_dict()
    positive_prevalence = float(target.mean())

    dtypes = df.dtypes.astype(str)
    dtype_counts = dtypes.value_counts().to_dict()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    missingness = df.isna().mean().sort_values(ascending=False)
    top_missing = (missingness.head(30) * 100).round(4).to_dict()

    cardinality = {c: int(df[c].nunique(dropna=True)) for c in categorical_cols}
    cardinality_sorted = dict(
        sorted(cardinality.items(), key=lambda kv: kv[1], reverse=True)
    )

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    near_constant_cols = [
        c for c in df.columns
        if df[c].nunique(dropna=True) > 1
        and df[c].value_counts(normalize=True, dropna=True).iloc[0] >= 0.995
    ]

    days_employed_special = int((df["DAYS_EMPLOYED"] == 365243).sum()) if "DAYS_EMPLOYED" in df.columns else None

    neg_value_counts = {}
    for col in numeric_cols:
        col_min = df[col].min(skipna=True)
        if pd.notna(col_min) and col_min < 0:
            neg_value_counts[col] = int((df[col] < 0).sum())
    neg_value_counts_top = dict(
        sorted(neg_value_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
    )

    summary = {
        "dataset_path": str(csv_path),
        "shape": {"n_rows": int(n_rows), "n_cols": int(n_cols)},
        "target": {
            "column": target_col,
            "value_counts": {str(k): int(v) for k, v in target_counts.items()},
            "positive_prevalence": positive_prevalence,
        },
        "dtypes": {k: int(v) for k, v in dtype_counts.items()},
        "n_numeric_columns": len(numeric_cols),
        "n_categorical_columns": len(categorical_cols),
        "top_30_missing_pct": top_missing,
        "categorical_cardinality_top_15": {
            k: cardinality_sorted[k] for k in list(cardinality_sorted)[:15]
        },
        "constant_columns": constant_cols,
        "near_constant_columns_geq_99_5_pct": near_constant_cols,
        "days_employed_eq_365243_count": days_employed_special,
        "note_on_negative_values": (
            "Negative values in DAYS_* columns are expected (days before the application); "
            "they are NOT treated as sentinels. Only DAYS_EMPLOYED == 365243 is a documented "
            "special value in Home Credit."
        ),
        "numeric_columns_with_negative_values_top_15": neg_value_counts_top,
    }

    pd.DataFrame(
        [(c, float(missingness[c] * 100)) for c in missingness.index],
        columns=["column", "missing_pct"],
    ).to_csv(ARTIFACTS_DIR / "home_credit_missingness.csv", index=False)

    pd.DataFrame(
        [(c, cardinality[c]) for c in categorical_cols],
        columns=["column", "cardinality"],
    ).sort_values("cardinality", ascending=False).to_csv(
        ARTIFACTS_DIR / "home_credit_categorical_cardinality.csv", index=False
    )

    (ARTIFACTS_DIR / "home_credit_audit.json").write_text(
        json.dumps(summary, indent=2)
    )

    return summary


def print_summary(summary: dict) -> None:
    print(f"Dataset: {summary['dataset_path']}")
    print(f"Shape:   {summary['shape']['n_rows']} rows x {summary['shape']['n_cols']} cols")
    print(f"Target:  {summary['target']['value_counts']}  (pos prev {summary['target']['positive_prevalence']:.4f})")
    print(f"Numeric columns:     {summary['n_numeric_columns']}")
    print(f"Categorical columns: {summary['n_categorical_columns']}")
    print(f"DAYS_EMPLOYED == 365243 (documented special): {summary['days_employed_eq_365243_count']}")
    print(f"Constant columns:    {summary['constant_columns']}")
    print(f"Near-constant >=99.5%: {len(summary['near_constant_columns_geq_99_5_pct'])} cols")
    print("Top 10 missingness (%):")
    for k, v in list(summary["top_30_missing_pct"].items())[:10]:
        print(f"  {k:40s} {v:8.2f}")


if __name__ == "__main__":
    s = audit_application_train()
    print_summary(s)
    print(f"\nWrote artifacts/home_credit_audit.json, "
          f"home_credit_missingness.csv, home_credit_categorical_cardinality.csv")