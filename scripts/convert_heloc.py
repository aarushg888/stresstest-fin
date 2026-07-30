from pathlib import Path
import arff
import numpy as np
import pandas as pd

input_path = Path("data/raw/heloc.arff")
output_path = Path("data/raw/heloc.csv")

with input_path.open("r", encoding="utf-8") as f:
    raw = arff.load(f)

columns = [name for name, _ in raw["attributes"]]
df = pd.DataFrame(raw["data"], columns=columns)

target_column = "RiskPerformance"
df[target_column] = df[target_column].astype(str)

feature_columns = [c for c in df.columns if c != target_column]
df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")

special_codes = [-9, -8, -7]
special_code_counts = {
    str(code): int((df[feature_columns] == code).sum().sum())
    for code in special_codes
}

df[feature_columns] = df[feature_columns].replace(special_codes, np.nan)

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Saved: {output_path}")
print(f"Shape: {df.shape}")
print("Class counts:")
print(df[target_column].value_counts())
print("Special values converted to missing:")
print(special_code_counts)
print("Missingness after conversion:")
print((df.isna().mean().sort_values(ascending=False).head(10) * 100).round(2))