import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_stability(
    model,
    X,
    y,
    *,
    n_repeats=10,
    top_k=10,
    random_state=42,
):
    baseline = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=1,
    )

    perturbed = X.copy()
    numeric_columns = list(perturbed.select_dtypes(include="number").columns)

    rng = np.random.default_rng(random_state)
    for column in numeric_columns:
        std = perturbed[column].std(skipna=True)
        if pd.notna(std) and std > 0:
            noise = rng.normal(0, 0.01 * std, size=len(perturbed))
            perturbed[column] = perturbed[column] + noise

    after = permutation_importance(
        model,
        perturbed,
        y,
        n_repeats=n_repeats,
        random_state=random_state + 1,
        scoring="roc_auc",
        n_jobs=1,
    )

    before_series = pd.Series(
        baseline.importances_mean,
        index=X.columns,
        name="before",
    )
    after_series = pd.Series(
        after.importances_mean,
        index=X.columns,
        name="after",
    )

    ranking = pd.DataFrame(
        {
            "before": before_series,
            "after": after_series,
        }
    )

    top_before = set(
        ranking["before"].nlargest(top_k).index
    )
    top_after = set(
        ranking["after"].nlargest(top_k).index
    )

    shared_top_features = sorted(top_before & top_after)
    jaccard = len(shared_top_features) / len(top_before | top_after)

    mean_absolute_change = float(
        (ranking["before"] - ranking["after"]).abs().mean()
    )

    return {
        "available": True,
        "method": "permutation_importance_roc_auc",
        "noise_fraction_of_feature_std": 0.01,
        "n_repeats": n_repeats,
        "top_k": top_k,
        "top_k_jaccard_similarity": float(jaccard),
        "mean_absolute_importance_change": mean_absolute_change,
        "shared_top_features": shared_top_features,
    }
