from pathlib import Path
import arff
import numpy as np
import pandas as pd

input_path = Path("data/raw/heloc.arff")
output_path = Path("data/raw/heloc_sentinel_aware.csv")

with input_path.open(encoding="utf-8") as file:
    raw = arff.load(file)

columns = [name for name, _ in raw["attributes"]]
df = pd.DataFrame(raw["data"], columns=columns)

target = "RiskPerformance"
features = [column for column in df.columns if column != target]

df[target] = df[target].astype(str)
df[features] = df[features].apply(pd.to_numeric, errors="coerce")

for feature in features:
    for code in (-9, -8, -7):
        df[f"{feature}__was_{code}"] = (df[feature] == code).astype("int8")

df[features] = df[features].replace([-9, -8, -7], np.nan)

df.to_csv(output_path, index=False)

indicator_columns = [c for c in df.columns if "__was_" in c]
print(f"Created: {output_path}")
print(f"Rows, columns: {df.shape}")
print(f"Added indicators: {len(indicator_columns)}")
print("Non-zero indicators:")
print(df[indicator_columns].sum().sort_values(ascending=False).head(15))
