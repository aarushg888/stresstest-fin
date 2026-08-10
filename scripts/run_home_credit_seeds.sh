#!/usr/bin/env bash
# Run the Home Credit 5-seed sweep for one config, writing per-seed JSON.
# Usage: scripts/run_home_credit_seeds.sh <config.yaml> <output_prefix> [extra_args...]
# Examples:
#   scripts/run_home_credit_seeds.sh configs/home_credit_baseline.yaml home_credit_baseline --skip-permutation
#   scripts/run_home_credit_seeds.sh configs/home_credit_baseline.yaml home_credit_baseline_permutation --n-repeats 3 --perm-top-k 10
set -euo pipefail
CONFIG="$1"; shift
PREFIX="$1"; shift
EXTRA="$@"
SEEDS=(11 22 33 44 55)
for s in "${SEEDS[@]}"; do
    OUT="artifacts/${PREFIX}_seed_${s}.json"
    echo "=== running ${CONFIG} seed=${s} -> ${OUT} ==="
    .venv/bin/python -m src.stresstest_fin.run \
        --config "${CONFIG}" \
        --seed "${s}" \
        --output "${OUT}" \
        --max-train 60000 \
        --max-eval 3000 \
        --n-repeats 5 \
        --perm-top-k 10 \
        ${EXTRA}
done
echo "DONE ${PREFIX}"