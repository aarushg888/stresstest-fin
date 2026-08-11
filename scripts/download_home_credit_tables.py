"""Download the Home Credit auxiliary tables from the Kaggle competition.

Requires a valid Kaggle API token. The modern KGAT_ token must be passed
via the KAGGLE_API_TOKEN environment variable (Bearer auth); the legacy
kaggle.json username/key pair (Basic auth) is NOT sufficient for this
competition in kagglehub 1.0.2.

  KAGGLE_API_TOKEN=KGAT_... python scripts/download_home_credit_tables.py

Also requires acceptance of the competition rules:
https://www.kaggle.com/competitions/home-credit-default-risk/rules

Outputs are written to data/raw/home_credit/ (gitignored). kagglehub caches
a copy under ~/.cache/kagglehub; we copy the target files into the repo.
"""
import os
import shutil
from pathlib import Path

import kagglehub

COMPETITION = "home-credit-default-risk"
FILES = [
    "previous_application.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
]

DEST = Path("data/raw/home_credit")
DEST.mkdir(parents=True, exist_ok=True)

for name in FILES:
    path = kagglehub.competition_download(COMPETITION, path=name)
    dst = DEST / name
    shutil.copy(path, dst)
    size_mb = os.path.getsize(dst) / 1e6
    print(f"OK {name}: {size_mb:.1f} MB -> {dst}")

print("ALL DONE")
