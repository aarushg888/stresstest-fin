from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

def make_preprocessor(X):
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

def build_models(X, seed):
    pre = make_preprocessor(X)
    return {
        "logistic_regression": Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
        "hist_gradient_boosting": Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier(random_state=seed))]),
        "random_forest": Pipeline([("pre", pre), ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=-1))]),
    }
