from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================
PROJECT_DIR = Path("stock_network")

PROCESSED_DIR = (
    PROJECT_DIR /
    "data" /
    "processed"
)

INPUT_RETURNS = (
    PROCESSED_DIR /
    "stock_returns.csv"
)

returns = pd.read_csv(
    INPUT_RETURNS,
    index_col=0,
    parse_dates=True
)

returns.columns = (
    returns.columns
    .astype(str)
    .str.zfill(6)
)

returns = returns.sort_index()

OUTPUT_DIR = Path("regime_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Regime boundaries identified from the previous
#    Rolling-GLasso regime analysis
# ============================================================

REGIME_PERIODS = {
    "R1": ("2024-01-16", "2024-07-18"),
    "R2": ("2024-08-15", "2024-12-16"),
    "R3": ("2025-01-14", "2025-10-16"),
    "R4": ("2025-11-13", "2026-02-09"),
    "R5": ("2026-03-17", "2026-07-14"),
}


# ============================================================
# 3. Load daily log returns
# ============================================================

returns = pd.read_csv(INPUT_RETURNS)

# The original file uses the Chinese column name "日期".
DATE_COL = "日期"

returns[DATE_COL] = pd.to_datetime(returns[DATE_COL])

returns = (
    returns
    .sort_values(DATE_COL)
    .drop_duplicates(subset=DATE_COL)
    .reset_index(drop=True)
)


# Stock columns
stock_cols = [c for c in returns.columns if c != DATE_COL]

print("=" * 70)
print("Full return data")
print("=" * 70)

print(f"Date range : "
      f"{returns[DATE_COL].min().date()} "
      f"to {returns[DATE_COL].max().date()}")

print(f"Number of trading days : {len(returns)}")
print(f"Number of stocks       : {len(stock_cols)}")

if len(stock_cols) != 15:
    print(
        f"WARNING: Expected 15 stocks, "
        f"but found {len(stock_cols)}."
    )


# ============================================================
# 4. Basic data-quality checks
# ============================================================

# Convert all stock-return columns to numeric.
returns[stock_cols] = returns[stock_cols].apply(
    pd.to_numeric,
    errors="coerce"
)

missing_count = returns[stock_cols].isna().sum()

if missing_count.sum() > 0:
    print("\nMissing values detected:")
    print(missing_count[missing_count > 0])

    # For graphical-model estimation, all stocks need to be
    # observed on the same date.
    returns = returns.dropna(
        subset=stock_cols
    ).reset_index(drop=True)

print(
    "\nRemaining complete trading days:",
    len(returns)
)


# ============================================================
# 5. Build non-overlapping regime-level samples
# ============================================================

regime_samples: dict[str, pd.DataFrame] = {}

summary_rows = []

for regime, (start, end) in REGIME_PERIODS.items():

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    mask = (
        (returns[DATE_COL] >= start)
        &
        (returns[DATE_COL] <= end)
    )

    sample = (
        returns.loc[
            mask,
            [DATE_COL] + stock_cols
        ]
        .copy()
        .reset_index(drop=True)
    )

    if sample.empty:
        raise ValueError(
            f"No observations found for {regime}: "
            f"{start.date()} -- {end.date()}"
        )

    regime_samples[regime] = sample

    n = len(sample)
    p = len(stock_cols)

    # -----------------------------------------
    # Descriptive diagnostics
    # -----------------------------------------

    X = sample[stock_cols]

    mean_abs_return = (
        X.abs().mean().mean()
    )

    mean_stock_volatility = (
        X.std(ddof=1).mean()
    )

    # Average absolute pairwise Pearson correlation
    corr = X.corr().to_numpy()

    upper = np.triu_indices_from(
        corr,
        k=1
    )

    mean_abs_corr = np.mean(
        np.abs(corr[upper])
    )

    summary_rows.append(
        {
            "regime": regime,
            "start_date":
                sample[DATE_COL].min().date(),
            "end_date":
                sample[DATE_COL].max().date(),
            "n_days": n,
            "p_stocks": p,
            "n_over_p": n / p,
            "n_missing":
                int(X.isna().sum().sum()),
            "mean_abs_daily_return":
                mean_abs_return,
            "mean_stock_volatility":
                mean_stock_volatility,
            "mean_abs_pairwise_corr":
                mean_abs_corr,
        }
    )

    # -----------------------------------------
    # Save regime sample
    # -----------------------------------------

    sample.to_csv(
        OUTPUT_DIR / f"{regime}_returns.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 6. Build regime summary table
# ============================================================

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_DIR / "regime_sample_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. Verify that regime samples do not overlap
# ============================================================

print("\n" + "=" * 70)
print("Regime samples")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)


for i in range(len(REGIME_PERIODS) - 1):

    r1 = f"R{i + 1}"
    r2 = f"R{i + 2}"

    dates_1 = set(
        regime_samples[r1][DATE_COL]
    )

    dates_2 = set(
        regime_samples[r2][DATE_COL]
    )

    overlap = dates_1.intersection(
        dates_2
    )

    if overlap:
        raise RuntimeError(
            f"{r1} and {r2} overlap: "
            f"{len(overlap)} dates."
        )


print("\nNo overlapping trading dates between regimes.")


# ============================================================
# 8. Check whether each regime has enough observations
# ============================================================

print("\n" + "=" * 70)
print("Sample-size diagnostics")
print("=" * 70)

for row in summary.itertuples():

    if row.n_days <= row.p_stocks:
        status = "WARNING: n <= p"
    elif row.n_days < 2 * row.p_stocks:
        status = "CAUTION: relatively small n"
    else:
        status = "OK"

    print(
        f"{row.regime}: "
        f"n={row.n_days}, "
        f"p={row.p_stocks}, "
        f"n/p={row.n_over_p:.2f} "
        f"--> {status}"
    )


print("\nFiles written to:")
print(OUTPUT_DIR.resolve())