from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. Raw daily market-data directory
# ============================================================

DAILY_TEMP3_DIR = Path(
    r"D:\lowfreq\daily_temp3"
)


# ============================================================
# 1. Sampling settings
#
# Enough for a unit audit; no need to load all daily files.
# ============================================================

MAX_SAMPLE_FILES = 150


# ============================================================
# 2. Discover CSV files
# ============================================================

if not DAILY_TEMP3_DIR.exists():

    raise FileNotFoundError(
        f"Directory does not exist:\n"
        f"{DAILY_TEMP3_DIR}"
    )


csv_files = sorted(
    DAILY_TEMP3_DIR.glob("*.csv")
)


if len(csv_files) == 0:

    raise RuntimeError(
        f"No CSV files found under:\n"
        f"{DAILY_TEMP3_DIR}"
    )


print("=" * 80)
print("Trade-Value Unit Audit")
print("=" * 80)

print(
    f"Total daily CSV files: "
    f"{len(csv_files):,}"
)

print(
    f"First file: {csv_files[0].name}"
)

print(
    f"Last file : {csv_files[-1].name}"
)


# ============================================================
# 3. Confirm actual schema
# ============================================================

header = pd.read_csv(
    csv_files[0],
    nrows=0,
)


required_columns = [
    "turnoverVol",
    "turnoverValue",
    "vwap",
    "isOpen",
]


missing = [
    col
    for col in required_columns
    if col not in header.columns
]


if missing:

    raise RuntimeError(
        "Missing required columns:\n"
        f"{missing}\n\n"
        "Available columns:\n"
        f"{header.columns.tolist()}"
    )


print()
print("Columns confirmed:")

for col in required_columns:

    print(
        f"  {col}"
    )


# ============================================================
# 4. Evenly sample dates over the whole history
# ============================================================

sample_n = min(
    MAX_SAMPLE_FILES,
    len(csv_files),
)


indices = np.linspace(
    0,
    len(csv_files) - 1,
    sample_n,
)


indices = np.unique(
    np.round(
        indices
    ).astype(int)
)


sample_files = [
    csv_files[i]
    for i in indices
]


print()
print(
    f"Sampled files: "
    f"{len(sample_files):,}"
)


# ============================================================
# 5. Read sample
# ============================================================

frames = []


for k, path in enumerate(
    sample_files,
    start=1,
):

    if (
        k == 1
        or
        k % 25 == 0
        or
        k == len(sample_files)
    ):

        print(
            f"Reading "
            f"{k}/{len(sample_files)}: "
            f"{path.name}"
        )


    temp = pd.read_csv(
        path,
        usecols=required_columns,
        low_memory=False,
    )


    for col in required_columns:

        temp[col] = pd.to_numeric(
            temp[col],
            errors="coerce",
        )


    # --------------------------------------------------------
    # Normal open trading observations only
    # --------------------------------------------------------

    temp = temp[
        (
            temp["isOpen"] == 1
        )
        &
        (
            temp["turnoverVol"] > 0
        )
        &
        (
            temp["turnoverValue"] > 0
        )
        &
        (
            temp["vwap"] > 0
        )
    ].copy()


    if len(temp) == 0:

        continue


    temp[
        "source_file"
    ] = path.name


    frames.append(
        temp
    )


if len(frames) == 0:

    raise RuntimeError(
        "No valid open-trading observations found."
    )


sample = pd.concat(
    frames,
    ignore_index=True,
)


# ============================================================
# 6. Core unit identity
#
# turnoverValue / (turnoverVol * vwap)
# ============================================================

sample[
    "implied_value"
] = (
    sample[
        "turnoverVol"
    ]
    *
    sample[
        "vwap"
    ]
)


sample[
    "value_ratio"
] = (
    sample[
        "turnoverValue"
    ]
    /
    sample[
        "implied_value"
    ]
)


ratio = (
    sample[
        "value_ratio"
    ]
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
    .dropna()
)


if len(ratio) == 0:

    raise RuntimeError(
        "No valid turnoverValue/"
        "(turnoverVol*vwap) observations."
    )


# ============================================================
# 7. Overall diagnostics
# ============================================================

print()
print("=" * 80)

print(
    "turnoverValue / "
    "(turnoverVol * vwap)"
)

print("=" * 80)

print(
    f"Valid observations: "
    f"{len(ratio):,}"
)

print()

quantiles = ratio.quantile(
    [
        0.001,
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
        0.999,
    ]
)

print("Quantiles:")
print(
    quantiles.to_string()
)

print()

print(
    f"Mean   = "
    f"{ratio.mean():.12g}"
)

print(
    f"Median = "
    f"{ratio.median():.12g}"
)

print()


for lower, upper in [
    (0.999, 1.001),
    (0.99, 1.01),
    (0.95, 1.05),
    (0.90, 1.10),
]:

    share = ratio.between(
        lower,
        upper,
    ).mean()

    print(
        f"Share in "
        f"[{lower}, {upper}] = "
        f"{share:.6%}"
    )


# ============================================================
# 8. Check whether scale is stable over time
# ============================================================

daily_summary = (

    sample.groupby(
        "source_file"
    )[
        "value_ratio"
    ]
    .agg(
        n="count",
        median_ratio="median",
        mean_ratio="mean",
    )
    .reset_index()
)


print()
print("=" * 80)
print("Across-date median-ratio diagnostics")
print("=" * 80)

print(
    daily_summary[
        "median_ratio"
    ]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
        ]
    )
    .to_string()
)


# ============================================================
# 9. Numerical interpretation
# ============================================================

median_ratio = float(
    ratio.median()
)


print()
print("=" * 80)
print("Numerical interpretation")
print("=" * 80)


if (
    0.95
    <= median_ratio
    <= 1.05
):

    print(
        "turnoverValue ≈ turnoverVol × vwap."
    )

    print()

    print(
        "If the official source definition confirms:"
    )

    print(
        "  turnoverVol   = shares"
    )

    print(
        "  vwap          = CNY/share"
    )

    print(
        "  turnoverValue = transaction value"
    )

    print()

    print(
        "then turnoverValue is consistent "
        "with CNY."
    )

    print()

    print(
        "For Step 3 use:"
    )

    print(
        "EXPLICIT_VALUE_UNIT_CONFIRMED = True"
    )

    print(
        "EXPLICIT_VALUE_SCALE_TO_CNY = 1.0"
    )


elif (
    95
    <= median_ratio
    <= 105
):

    print(
        "turnoverValue ≈ "
        "100 × turnoverVol × vwap."
    )

    print()

    print(
        "This strongly suggests turnoverVol "
        "may be recorded in lots (手), "
        "where 1 lot = 100 shares."
    )

    print()

    print(
        "Do NOT rescale turnoverValue "
        "based on this alone."
    )


elif (
    0.00095
    <= median_ratio
    <= 0.00105
):

    print(
        "turnoverValue ≈ "
        "(turnoverVol × vwap) / 1000."
    )

    print()

    print(
        "This is numerically consistent with "
        "turnoverValue being in thousand CNY "
        "IF turnoverVol is shares."
    )

    print()

    print(
        "Official documentation is still required."
    )


else:

    print(
        f"Median ratio = "
        f"{median_ratio:.12g}"
    )

    print()

    print(
        "No simple unit relationship was identified."
    )

    print(
        "Do NOT set the Step-3 CNY scale yet."
    )


print()
print("=" * 80)
print("Audit complete")
print("=" * 80)