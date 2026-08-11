"""Convert application_train.csv into a sentinel-aware variant.

Adds a single documented indicator for the one Home Credit special value
we have evidence for: ``DAYS_EMPLOYED == 365243`` (employment duration
sentinel meaning "missing/unknown"). Then replaces the special value with
NaN so downstream imputation handles it normally.

Negative values in DAYS_* columns are NOT treated as sentinels — those
are just days *before* the application date.

Usage:
    python scripts/convert_home_credit_sentinel_aware.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "data" / "raw" / "home_credit" / "application_train.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "home_credit" / "application_train_sentinel_aware.csv"

SPECIAL_COLUMN = "DAYS_EMPLOYED"
SPECIAL_CODE = 365243
INDICATOR_NAME = "DAYS_EMPLOYED__was_special"


def convert(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    if SPECIAL_COLUMN not in df.columns:
        raise ValueError(f"Expected column {SPECIAL_COLUMN!r} not in dataframe")

    special_count_before = int((df[SPECIAL_COLUMN] == SPECIAL_CODE).sum())
    df[INDICATOR_NAME] = (df[SPECIAL_COLUMN] == SPECIAL_CODE).astype("int8")
    df[SPECIAL_COLUMN] = df[SPECIAL_COLUMN].replace(SPECIAL_CODE, np.nan)
    special_count_after = int(df[INDICATOR_NAME].sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "indicator_name": INDICATOR_NAME,
        "special_value": SPECIAL_CODE,
        "rows_with_special_before": special_count_before,
        "rows_with_indicator_after": special_count_after,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    convert(args.input, args.output)