"""Aggregate Home Credit per-seed evaluation JSON files into a CSV.

Usage:
    python scripts/summarize_home_credit.py \
        --inputs "artifacts/home_credit_baseline_seed_*.json" \
        --output artifacts/home_credit_baseline_summary.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


METRICS = [
    "roc_auc",
    "average_precision",
    "brier",
    "ece_10",
]
SENS_KEY = "mean_absolute_probability_change"


def collect(json_paths):
    rows = []
    for jp in sorted(json_paths):
        d = json.loads(Path(jp).read_text())
        seed = d.get("random_seed") or d.get("seed") or Path(jp).stem.split("_")[-1]
        for model_name, model_res in d["models"].items():
            pm = model_res["predictive_metrics"]
            row = {
                "seed": int(seed) if str(seed).isdigit() else seed,
                "model": model_name,
                "roc_auc": pm["roc_auc"],
                "average_precision": pm["average_precision"],
                "brier": pm["brier"],
                "ece_10": pm["ece_10"],
                "mean_probability_change": (
                    model_res.get("stability", {}).get(SENS_KEY)
                    if model_res.get("stability", {}).get("available") else None
                ),
                "n_test_used": d.get("n_test_used_for_metrics"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = METRICS + ["mean_probability_change"]
    grouped = df.groupby("model")[metric_cols].agg(["mean", "std"])
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    return grouped.reset_index()


def print_table(df: pd.DataFrame) -> None:
    cols = ["model"] + [f"{m}_{s}" for m in METRICS + ["mean_probability_change"] for s in ("mean", "std")]
    cols = [c for c in cols if c in df.columns]
    pretty = df[cols].copy()
    for c in pretty.columns:
        if c != "model":
            pretty[c] = pretty[c].astype(float).round(4)
    print(pretty.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True,
                        help="Glob pattern for per-seed JSON files.")
    parser.add_argument("--output", required=True,
                        help="Output CSV path.")
    parser.add_argument("--label", default="condition",
                        help="Label column for this condition (e.g. baseline, sentinel_aware).")
    args = parser.parse_args()

    paths = sorted(Path(".").glob(args.inputs))
    if not paths:
        sys.exit(f"No input files matched: {args.inputs}")

    df = collect(paths)
    df.insert(0, "condition", args.label)
    df.insert(1, "n_seeds", df.groupby("model")["seed"].transform("count"))

    df.to_csv(args.output.replace(".csv", "_raw.csv"), index=False)

    agg = aggregate(df)
    agg.insert(0, "condition", args.label)
    agg.to_csv(args.output, index=False)

    print(f"Wrote {args.output} and {args.output.replace('.csv', '_raw.csv')}")
    print_table(agg)


if __name__ == "__main__":
    main()