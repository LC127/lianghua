
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT / "output"
OUT = R / "M1_day18/04_step4_duplicate_key_review/final_check"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Stock code history
p = R / "M1_day1/01_stage1/stock_code_history.parquet"
d = pd.read_parquet(p)
key = ["security_id", "stock_code"]

dup = d[d.duplicated(key, keep=False)].sort_values(key)
missing = d[d[["security_id", "stock_code"]].isna().any(axis=1)]

dup.to_csv(OUT / "stock_code_history_duplicates.csv",
           index=False, encoding="utf-8-sig")
missing.to_csv(OUT / "stock_code_history_missing_identity.csv",
               index=False, encoding="utf-8-sig")

print("Stock history:")
print("Duplicate-group rows:", len(dup))
print("Exact duplicate excess rows:", int(d.duplicated().sum()))
print("Missing identity rows:", len(missing))

# 2. Stock master issues: preserve all original rows
p = R / "M1_day1/01_stage1_fixed/stock_master_issues.csv"
d = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
exact = d[d.duplicated(keep=False)]

exact.to_csv(OUT / "stock_master_issues_exact_duplicates.csv",
             index=False, encoding="utf-8-sig")

print("\nStock master issues:")
print("Original rows:", len(d))
print("Exact duplicate excess rows:", int(d.duplicated().sum()))
print("Rows involved in duplicate groups:", len(exact))
print("Unique rows after exact deduplication:", len(d.drop_duplicates()))
print("Output:", OUT)
