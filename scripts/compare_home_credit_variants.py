"""Compare baseline vs sentinel-aware Home Credit summary CSVs.

Outputs a wide CSV with mean+/-std for each (model, metric, condition) plus a
delta column.

Usage:
    python scripts/compare_home_credit_variants.py \
        --baseline artifacts/home_credit_baseline_summary.csv \
        --sentinel artifacts/home_credit_sentinel_aware_summary.csv \
        --output artifacts/home_credit_baseline_vs_sentinel_aware.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["roc_auc", "average_precision", "brier", "ece_10", "mean_probability_change"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--sentinel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    b = pd.read_csv(args.baseline)
    s = pd.read_csv(args.sentinel)

    b = b.drop(columns=["condition"], errors="ignore")
    s = s.drop(columns=["condition"], errors="ignore")

    b = b.rename(columns={c: f"{c}__baseline" for c in b.columns if c != "model"})
    s = s.rename(columns={c: f"{c}__sentinel" for c in s.columns if c != "model"})

    merged = b.merge(s, on="model", how="outer", suffixes=("", ""))
    rows = []
    for _, r in merged.iterrows():
        row = {"model": r["model"]}
        for m in METRICS:
            mean_b = r.get(f"{m}_mean__baseline")
            std_b = r.get(f"{m}_std__baseline")
            mean_s = r.get(f"{m}_mean__sentinel")
            std_s = r.get(f"{m}_std__sentinel")
            row[f"{m}_baseline_mean"] = mean_b
            row[f"{m}_baseline_std"] = std_b
            row[f"{m}_sentinel_mean"] = mean_s
            row[f"{m}_sentinel_std"] = std_s
            if pd.notna(mean_b) and pd.notna(mean_s):
                row[f"{m}_delta"] = mean_s - mean_b
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(out.round(4).to_string(index=False))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()