"""Aggregate Home Credit permutation-importance stability into by-seed and
summary CSVs.

Reads per-seed JSON artifacts produced by run.py (which include the
`permutation_importance_stability` block) and produces:

- artifacts/home_credit_permutation_stability_by_seed.csv
- artifacts/home_credit_permutation_stability_summary.csv

Usage:
    python scripts/summarize_home_credit_permutation.py \
        --inputs "artifacts/home_credit_*_permutation_seed_*.json"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True,
                        help="Glob matching per-seed permutation JSON files.")
    parser.add_argument("--by-seed", default="artifacts/home_credit_permutation_stability_by_seed.csv")
    parser.add_argument("--summary", default="artifacts/home_credit_permutation_stability_summary.csv")
    args = parser.parse_args()

    paths = sorted(Path(".").glob(args.inputs))
    if not paths:
        raise SystemExit(f"No files matched: {args.inputs}")

    rows = []
    for jp in paths:
        d = json.loads(jp.read_text())
        seed = d.get("random_seed")
        # variant = e.g. "baseline" / "sentinel_aware" from filename
        stem = jp.stem
        m = re.search(r"home_credit_(.*?)_permutation_seed_", stem)
        variant = m.group(1) if m else "unknown"
        for model_name, model_res in d["models"].items():
            pi = model_res.get("permutation_importance_stability") or {}
            if not pi.get("available"):
                continue
            rows.append({
                "variant": variant,
                "seed": int(seed),
                "model": model_name,
                "top_10_jaccard": pi.get("top_k_jaccard_similarity"),
                "mean_importance_change": pi.get("mean_absolute_importance_change"),
                "shared_top_features": ";".join(pi.get("shared_top_features", [])),
            })

    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(args.by_seed, index=False)
    print(f"Wrote {args.by_seed}")

    if by_seed.empty:
        raise SystemExit("No permutation-stability rows found.")

    summary = (
        by_seed.groupby(["variant", "model"])[["top_10_jaccard", "mean_importance_change"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.summary, index=False)
    print(f"Wrote {args.summary}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()