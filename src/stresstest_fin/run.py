import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import load_config
from .data import features_and_target, load_and_validate, split_data
from .evaluate import (
    explanation_stability_proxy,
    group_audit,
    predictive_metrics,
)
from .explainability import permutation_importance_stability
from .models import build_models


def _maybe_subsample(X, y, max_train: int, seed: int):
    """Optionally subsample training rows with a fixed seed.

    Subsampling is only applied when len(X) > max_train. Stratified on y
    when possible. The subsample seed is derived from the parent seed so
    runs are reproducible.
    """
    import numpy as np
    import pandas as pd
    if len(X) <= max_train:
        return X, y
    rng = np.random.default_rng(seed)
    positives = y == 1
    neg_idx = np.where(~positives.to_numpy())[0]
    pos_idx = np.where(positives.to_numpy())[0]
    n_pos_target = min(len(pos_idx), max(1, int(max_train * float(y.mean()))))
    n_neg_target = max_train - n_pos_target
    n_neg_target = min(n_neg_target, len(neg_idx))
    n_pos_target = min(n_pos_target, len(pos_idx))
    sel_pos = rng.choice(pos_idx, size=n_pos_target, replace=False)
    sel_neg = rng.choice(neg_idx, size=n_neg_target, replace=False)
    sel = np.concatenate([sel_pos, sel_neg])
    sel.sort()
    return X.iloc[sel].reset_index(drop=True), y.iloc[sel].reset_index(drop=True)


def _maybe_subsample_eval(X, y, max_eval: int, seed: int):
    """Subsample evaluation rows deterministically (used for permutation
    importance and prediction-sensitivity which are expensive)."""
    import numpy as np
    if len(X) <= max_eval:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_eval, replace=False)
    idx.sort()
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the random seed from the config file.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. Defaults to artifacts/evaluation.json.")
    parser.add_argument("--skip-permutation", action="store_true",
                        help="Skip the permutation-importance step (used for cheap seed sweeps).")
    parser.add_argument("--max-train", type=int, default=0,
                        help="Cap on stratified training rows. 0 = no cap.")
    parser.add_argument("--max-eval", type=int, default=0,
                        help="Cap on deterministic evaluation subsample for "
                             "permutation importance / prediction sensitivity. 0 = no cap.")
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="Permutation-importance repeats.")
    parser.add_argument("--perm-top-k", type=int, default=10,
                        help="Top-K for permutation Jaccard.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = replace(cfg, random_seed=args.seed)

    df = load_and_validate(cfg)
    train, test, split_type = split_data(df, cfg)

    X_train, y_train = features_and_target(train, cfg)
    X_test, y_test = features_and_target(test, cfg)

    if args.max_train and len(X_train) > args.max_train:
        X_train, y_train = _maybe_subsample(
            X_train, y_train, args.max_train, cfg.random_seed
        )

    if args.max_eval and len(X_test) > args.max_eval:
        X_eval, y_eval = _maybe_subsample_eval(
            X_test, y_test, args.max_eval, cfg.random_seed
        )
    else:
        X_eval, y_eval = X_test, y_test.reset_index(drop=True)

    models = build_models(X_train, cfg.random_seed)

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": args.config,
        "dataset_path": cfg.dataset_path,
        "random_seed": cfg.random_seed,
        "split_type": split_type,
        "n_train_full": len(train),
        "n_test_full": len(test),
        "n_train_used": len(X_train),
        "n_test_used_for_metrics": len(X_test),
        "n_eval_used_for_permutation_and_sensitivity": len(X_eval),
        "max_train_cap": args.max_train,
        "max_eval_cap": args.max_eval,
        "warning": (
            "Research/education only. Do not use results for automated "
            "or high-impact decisions."
        ),
        "models": {},
    }

    if args.max_train and args.max_train < len(train):
        results["subsampling_note"] = (
            f"Training rows stratified-subsampled to {args.max_train} "
            f"(seed={cfg.random_seed}). Predictive metrics computed on the "
            f"unsubs-sampled test set ({len(X_test)} rows)."
        )

    for name in cfg.models:
        model = models[name].fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]

        prediction_stability = explanation_stability_proxy(
            model,
            X_eval,
            list(X_eval.select_dtypes(include="number").columns),
            fraction=(cfg.explanation or {}).get("fraction", 0.02),
            sample_size=(cfg.explanation or {}).get("sample_size", 100),
            seed=cfg.random_seed,
        )

        importance_stability = None
        if not args.skip_permutation:
            importance_stability = permutation_importance_stability(
                model,
                X_eval,
                y_eval.to_numpy(),
                n_repeats=args.n_repeats,
                top_k=args.perm_top_k,
                random_state=cfg.random_seed,
            )

        result = {
            "predictive_metrics": predictive_metrics(
                y_test.to_numpy(),
                probabilities,
            ),
            "stability": prediction_stability,
        }
        if importance_stability is not None:
            result["permutation_importance_stability"] = importance_stability

        if cfg.protected_group_column:
            result["group_audit"] = group_audit(
                y_test.to_numpy(),
                probabilities,
                test[cfg.protected_group_column].reset_index(drop=True),
            )

        results["models"][name] = result

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    output_path = Path(args.output) if args.output else output_dir / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()