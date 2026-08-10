import numpy as np
import pandas as pd

from stresstest_fin.models import (
    SpecialValueSpec,
    apply_special_values,
    build_models,
    build_preprocessor,
)


def _toy_df():
    return pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0],
        "num2": [10, 365243, 30, 40],
        "cat1": ["a", "b", "a", "c"],
        "id_col": [1, 2, 3, 4],
    })


def test_apply_special_values_adds_indicator_and_nans_code():
    df = _toy_df()
    spec = SpecialValueSpec(column="num2", code=365243, indicator_name="num2__was_special")
    out, indicator = apply_special_values(df, spec)
    assert indicator == "num2__was_special"
    assert out["num2__was_special"].tolist() == [0, 1, 0, 0]
    assert pd.isna(out.loc[1, "num2"])
    assert out.loc[0, "num2"] == 10.0


def test_build_preprocessor_handles_mixed_types():
    X = _toy_df().drop(columns=["id_col"])
    pre = build_preprocessor(X, scale_numeric=True, dense_output=False)
    out = pre.fit_transform(X)
    assert out.shape[0] == 4
    # numeric 2 + one-hot 3 categories = 5 columns
    assert out.shape[1] == 5


def test_build_models_predict_proba_on_mixed_frame():
    df = _toy_df().drop(columns=["id_col"])
    y = pd.Series([0, 1, 0, 1])
    models = build_models(df, seed=42)
    for name, model in models.items():
        model.fit(df, y)
        p = model.predict_proba(df)[:, 1]
        assert p.shape == (4,)
        assert np.isfinite(p).all()


def test_sentinel_indicator_flows_through_pipeline():
    df = _toy_df().drop(columns=["id_col"])
    spec = SpecialValueSpec(column="num2", code=365243, indicator_name="num2__was_special")
    out, _ = apply_special_values(df, spec)
    models = build_models(out, seed=42)
    y = pd.Series([0, 1, 0, 1])
    model = models["logistic_regression"].fit(out, y)
    assert np.isfinite(model.predict_proba(out)[:, 1]).all()
