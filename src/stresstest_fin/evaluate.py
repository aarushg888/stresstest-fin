import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_score, recall_score

def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any(): ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)

def predictive_metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)), "ece_10": expected_calibration_error(np.asarray(y), np.asarray(p)), "precision_at_0_5": float(precision_score(y, pred, zero_division=0)), "recall_at_0_5": float(recall_score(y, pred, zero_division=0))}

def group_audit(y, p, group, threshold=0.5):
    rows = {}; pred = (p >= threshold).astype(int)
    for g in sorted(group.dropna().unique()):
        idx = group == g
        if idx.sum() < 20: continue
        rows[str(g)] = {"n": int(idx.sum()), "selection_rate": float(pred[idx].mean()), "recall": float(recall_score(y[idx], pred[idx], zero_division=0)), "mean_predicted_risk": float(p[idx].mean())}
    return rows

def explanation_stability_proxy(model, X, numeric_columns, fraction=0.02, sample_size=100, seed=42):
    if not numeric_columns: return {"available": False, "reason": "no numeric features"}
    sample = X.sample(min(sample_size, len(X)), random_state=seed).copy()
    base = model.predict_proba(sample)[:, 1]
    perturbed = sample.copy()
    for c in numeric_columns:
        scale = max(float(sample[c].std(skipna=True) or 0), 1e-8)
        perturbed[c] = perturbed[c] + fraction * scale
    shifted = model.predict_proba(perturbed)[:, 1]
    return {"available": True, "mean_absolute_probability_change": float(np.mean(np.abs(base - shifted))), "max_absolute_probability_change": float(np.max(np.abs(base - shifted))), "note": "Prediction-sensitivity proxy; add SHAP/permutation attribution stability only after validating methodology."}
