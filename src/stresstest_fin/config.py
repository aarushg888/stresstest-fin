from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class Config:
    dataset_path: str
    target_column: str
    id_column: str | None = None
    time_column: str | None = None
    protected_group_column: str | None = None
    positive_label: str | int = 1
    random_seed: int = 42
    test_size: float = 0.25
    models: tuple[str, ...] = ("logistic_regression",)
    explanation: dict | None = None
    calibration: str | None = None  # "sigmoid" | "isotonic" — applied to LR only
    hgb_sample_weight_balanced: bool = False  # inverse-frequency weights for HGB

def load_config(path: str) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    raw["models"] = tuple(raw.get("models", []))
    return Config(**raw)
