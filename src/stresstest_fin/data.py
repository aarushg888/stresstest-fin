import pandas as pd
from sklearn.model_selection import train_test_split
from .config import Config

def load_and_validate(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.dataset_path)
    required = [cfg.target_column] + [x for x in [cfg.id_column, cfg.time_column, cfg.protected_group_column] if x]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    if df[cfg.target_column].isna().any(): raise ValueError("Target contains missing values")
    if df[cfg.target_column].nunique() != 2: raise ValueError("This starter supports binary targets only")
    return df

def split_data(df: pd.DataFrame, cfg: Config):
    if cfg.time_column:
        col = df[cfg.time_column]
        # Numeric recency proxies (e.g. days-before-application) sort as-is;
        # actual datetimes are parsed. Missing values are placed first so
        # they land in the training (past) partition.
        if pd.api.types.is_numeric_dtype(col):
            ordered = df.sort_values(
                cfg.time_column, na_position="first"
            ).reset_index(drop=True)
        else:
            ordered = (
                df.assign(**{cfg.time_column: pd.to_datetime(col)})
                .sort_values(cfg.time_column)
                .reset_index(drop=True)
            )
        n_test = max(1, int(len(ordered) * cfg.test_size))
        train, test = ordered.iloc[:-n_test].copy(), ordered.iloc[-n_test:].copy()
        split_type = "chronological"
    else:
        train, test = train_test_split(df, test_size=cfg.test_size, random_state=cfg.random_seed, stratify=df[cfg.target_column])
        split_type = "stratified_random_proxy"
    return train, test, split_type

def features_and_target(df: pd.DataFrame, cfg: Config):
    drop = [cfg.target_column] + [c for c in [cfg.id_column, cfg.time_column, cfg.protected_group_column] if c]
    X = df.drop(columns=drop, errors="ignore")
    y = (df[cfg.target_column] == cfg.positive_label).astype(int)
    return X, y
