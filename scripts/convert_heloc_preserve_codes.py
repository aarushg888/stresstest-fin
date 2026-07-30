from pathlib import Path
import arff
import pandas as pd

source = Path("data/raw/heloc.arff")
destination = Path("data/raw/heloc_preserve_codes.csv")

with source.open(encoding="utf-8") as file:
    raw = arff.load(file)

columns = [name for name, _ in raw["attributes"]]
df = pd.DataFrame(raw["data"], columns=columns)

target = "RiskPerformance"
features = [column for column in df.columns if column != target]

df[target] = df[target].astype(str)
df[features] = df[features].apply(pd.to_numeric, errors="coerce")
df.to_csv(destination, index=False)

print(f"Created {destination} with shape {df.shape}")
