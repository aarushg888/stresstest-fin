import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .data import features_and_target, load_and_validate, split_data
from .evaluate import (
    explanation_stability_proxy,
    group_audit,
    predictive_metrics,
)
from .explainability import permutation_importance_stability
from .models import build_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    df = load_and_validate(cfg)
    train, test, split_type = split_data(df, cfg)

    X_train, y_train = features_and_target(train, cfg)
    X_test, y_test = features_and_target(test, cfg)

    models = build_models(X_train, cfg.random_seed)

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": cfg.dataset_path,
        "split_type": split_type,
        "n_train": len(train),
        "n_test": len(test),
        "warning": (
            "Research/education only. Do not use results for automated "
            "or high-impact decisions."
        ),
        "models": {},
    }

    for name in cfg.models:
        model = models[name].fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]

        prediction_stability = explanation_stability_proxy(
            model,
            X_test,
            list(X_test.select_dtypes(include="number").columns),
            **(cfg.explanation or {}),
            seed=cfg.random_seed,
        )

        importance_stability = permutation_importance_stability(
            model,
            X_test,
            y_test.to_numpy(),
            random_state=cfg.random_seed,
        )

        result = {
            "predictive_metrics": predictive_metrics(
                y_test.to_numpy(),
                probabilities,
            ),
            "stability": prediction_stability,
            "permutation_importance_stability": importance_stability,
        }

        if cfg.protected_group_column:
            result["group_audit"] = group_audit(
                y_test.to_numpy(),
                probabilities,
                test[cfg.protected_group_column].reset_index(drop=True),
            )

        results["models"][name] = result

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "evaluation.json"
    output_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
