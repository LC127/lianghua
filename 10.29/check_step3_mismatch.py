
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(r"D:\M1_StockNetwork")
INV = ROOT / (
    r"output\M1_day18\02_step2_source_inventory"
    r"\run_20261029_120201_266156\day18_source_inventory.csv"
)
TARGET = ROOT / "source_inventory.py"

df = pd.read_csv(INV)
row = df[
    df["path"].astype(str).str.casefold()
    == str(TARGET).casefold()
]

print("Step 2 inventory record:")
print(row.T.to_string(header=False) if not row.empty else "NOT FOUND")

print("\nCurrent file:")
print("Size:", TARGET.stat().st_size)
print("Modified:", datetime.fromtimestamp(TARGET.stat().st_mtime))
