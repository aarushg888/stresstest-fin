"""Aggregate the Home Credit auxiliary tables into application-level features.

For each auxiliary table we compute a small, documented set of per-customer
aggregates (count, mean/max/min/sum/std on the most informative numeric
columns, nunique on a few categorical columns) and left-join them onto
application_train by SK_ID_CURR. bureau_balance is aggregated per
SK_ID_BUREAU first, then merged into bureau before the per-customer pass.

Output: data/raw/home_credit/application_train_enriched.csv (gitignored).

Research/education only. Features are computed from pre-application history
only, so joining them before the train/test split introduces no label
leakage; the split itself is applied later by run.py.
"""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/home_credit")
OUT = RAW / "application_train_enriched.csv"

app = pd.read_csv(RAW / "application_train.csv", low_memory=False)

NUMERIC_OPS = ["mean", "max", "min", "sum", "std"]


def agg_table(df, key, num_cols, cat_nunique=None):
    """Return per-<key> aggregates for one auxiliary table."""
    cat_nunique = cat_nunique or []
    agg = {"__count__": (key, "count")}
    for c in num_cols:
        for op in NUMERIC_OPS:
            agg[f"{c}_{op}"] = (c, op)
    for c in cat_nunique:
        agg[f"{c}_nunique"] = (c, "nunique")
    g = df.groupby(key).agg(**agg).reset_index()
    return g


# 1) previous_application -> per-customer history of prior loans
prev = pd.read_csv(RAW / "previous_application.csv", low_memory=False)
prev_num = [
    "AMT_APPLICATION", "AMT_CREDIT", "AMT_ANNUITY", "RATE_INTEREST_PRIMARY",
    "CNT_PAYMENT", "DAYS_DECISION", "DAYS_FIRST_DUE", "DAYS_LAST_DUE",
]
prev_cat = ["NAME_CONTRACT_TYPE", "NAME_PAYMENT_TYPE", "NAME_CLIENT_TYPE"]
prev_agg = agg_table(prev, "SK_ID_CURR", prev_num, prev_cat)
# Recency proxy for temporal validation: the most recent previous application
# decision, expressed in days before the current application. Customers with
# no previous applications get NaN here (handled at split time).
_recency = (
    prev.groupby("SK_ID_CURR")
    .agg(max_dec=("DAYS_DECISION", "max"))
    .reset_index()
    .rename(columns={"max_dec": "APP_RECENCY_DAYS"})
)
prev_recency = _recency
prev_agg = prev_agg.merge(prev_recency, on="SK_ID_CURR", how="left")
prev_agg = prev_agg.add_prefix("prev_")
prev_agg = prev_agg.rename(columns={"prev_SK_ID_CURR": "SK_ID_CURR"})
print(f"previous_application aggregates: {prev_agg.shape[1] - 1} features")

# 2) bureau (credit bureau reports) + bureau_balance (monthly statuses)
bureau = pd.read_csv(RAW / "bureau.csv", low_memory=False)
bb = pd.read_csv(RAW / "bureau_balance.csv", low_memory=False)
bb_num = ["MONTHS_BALANCE"]
bb_cat = ["STATUS"]
bb_agg = agg_table(bb, "SK_ID_BUREAU", bb_num, bb_cat)
bb_agg = bb_agg.add_prefix("bb_")
bb_agg = bb_agg.rename(columns={"bb_SK_ID_BUREAU": "SK_ID_BUREAU"})
bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
bureau_num = [
    "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
    "AMT_ANNUITY", "DAYS_CREDIT", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT",
    "CNT_CREDIT_PROLONG",
]
bureau_cat = ["CREDIT_ACTIVE", "CREDIT_TYPE"]
bureau_agg = agg_table(bureau, "SK_ID_CURR", bureau_num, bureau_cat)
bureau_agg = bureau_agg.add_prefix("bureau_")
bureau_agg = bureau_agg.rename(columns={"bureau_SK_ID_CURR": "SK_ID_CURR"})
print(f"bureau aggregates: {bureau_agg.shape[1] - 1} features")

# 3) credit_card_balance
cc = pd.read_csv(RAW / "credit_card_balance.csv", low_memory=False)
cc_num = [
    "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT",
    "AMT_DRAWINGS_CURRENT", "CNT_DRAWINGS_ATM_CURRENT", "SK_DPD", "SK_DPD_DEF",
]
cc_agg = agg_table(cc, "SK_ID_CURR", cc_num)
cc_agg = cc_agg.add_prefix("cc_")
cc_agg = cc_agg.rename(columns={"cc_SK_ID_CURR": "SK_ID_CURR"})
print(f"credit_card_balance aggregates: {cc_agg.shape[1] - 1} features")

# 4) installments_payments
inst = pd.read_csv(RAW / "installments_payments.csv", low_memory=False)
inst_num = ["AMT_INSTALMENT", "AMT_PAYMENT", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"]
inst_agg = agg_table(inst, "SK_ID_CURR", inst_num)
inst_agg = inst_agg.add_prefix("inst_")
inst_agg = inst_agg.rename(columns={"inst_SK_ID_CURR": "SK_ID_CURR"})
print(f"installments_payments aggregates: {inst_agg.shape[1] - 1} features")

# 5) POS_CASH_balance
pos = pd.read_csv(RAW / "POS_CASH_balance.csv", low_memory=False)
pos_num = [
    "MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE",
    "SK_DPD", "SK_DPD_DEF",
]
pos_agg = agg_table(pos, "SK_ID_CURR", pos_num)
pos_agg = pos_agg.add_prefix("pos_")
pos_agg = pos_agg.rename(columns={"pos_SK_ID_CURR": "SK_ID_CURR"})
print(f"POS_CASH_balance aggregates: {pos_agg.shape[1] - 1} features")

# 6) Join everything (left join keeps applicants with no auxiliary history)
for label, g in [
    ("prev", prev_agg), ("bureau", bureau_agg), ("cc", cc_agg),
    ("inst", inst_agg), ("pos", pos_agg),
]:
    before = app.shape[1]
    app = app.merge(g, on="SK_ID_CURR", how="left")
    print(f"joined {label}: +{app.shape[1] - before} cols, rows={app.shape[0]}")

app.to_csv(OUT, index=False)
print(f"\nWROTE {OUT}  shape={app.shape}  ({app.shape[1] - 122} new features)")
