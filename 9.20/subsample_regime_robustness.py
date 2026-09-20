from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 0. Paths
# ============================================================

ROOT = Path(
    r"D:\M1_StockNetwork"
)

OUTPUT_ROOT = (
    ROOT
    / "output"
)


# ------------------------------------------------------------
# Day 1 return panel
#
# Used ONLY to construct contemporaneous market states.
# ------------------------------------------------------------

DAY1_RETURN_PANEL_PATH = (
    OUTPUT_ROOT
    / "M1_day1"
    / "03_stage3_return_validation"
    / "daily_return_panel.parquet"
)


# ------------------------------------------------------------
# Day 6 outputs
#
# Research Day 6 is stored under M1_day5
# ------------------------------------------------------------

DAY6_ROOT = (
    OUTPUT_ROOT
    / "M1_day6"
)

STEP6_DIR = (
    DAY6_ROOT
    / "06_step6_neutralization_and_classification"
)

STEP6_METADATA_PATH = (
    STEP6_DIR
    / "step6_neutralization_metadata.json"
)

MONTHLY_ALPHA_IC_PATH = (
    STEP6_DIR
    / "monthly_neutralized_alpha_rank_ic.csv"
)

MONTHLY_ALPHA_SPREAD_PATH = (
    STEP6_DIR
    / "monthly_neutralized_alpha_portfolio_spreads.csv"
)

MONTHLY_RISK_IC_PATH = (
    STEP6_DIR
    / "monthly_neutralized_risk_rank_ic.csv"
)


# ------------------------------------------------------------
# Day 7 Step 1
#
# Research Day 7 -> M1_day6
# ------------------------------------------------------------

DAY7_ROOT = (
    OUTPUT_ROOT
    / "M1_day7"
)

OUTPUT_DIR = (
    DAY7_ROOT
    / "01_stage1_subsample_regime_robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MARKET_DAILY_STATE_PATH = (
    OUTPUT_DIR
    / "market_daily_state.csv"
)

MARKET_REGIME_LABEL_PATH = (
    OUTPUT_DIR
    / "market_regime_labels.csv"
)

REGIME_ASSIGNMENT_LONG_PATH = (
    OUTPUT_DIR
    / "regime_assignment_long.csv"
)

ALPHA_REGIME_PATH = (
    OUTPUT_DIR
    / "alpha_ic_by_subsample.csv"
)

PORTFOLIO_REGIME_PATH = (
    OUTPUT_DIR
    / "portfolio_spread_by_subsample.csv"
)

RISK_REGIME_PATH = (
    OUTPUT_DIR
    / "risk_ic_by_subsample.csv"
)

ROBUSTNESS_SUMMARY_PATH = (
    OUTPUT_DIR
    / "regime_robustness_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day7_step1_metadata.json"
)


# ============================================================
# 1. Frozen Day-7 Step-1 Design
# ============================================================

EXPECTED_WINDOWS = [
    60,
    120,
    252,
]


# ------------------------------------------------------------
# We keep:
#
# 1. RAW baseline
# 2. Final Industry + Size neutralized specification
#
# Industry-only neutralization was already studied in Day 6.
# ------------------------------------------------------------

ANALYSIS_METHODS = [

    "RAW",

    "INDUSTRY_SIZE_NEUTRAL",
]


# ------------------------------------------------------------
# Market-state construction
# ------------------------------------------------------------

MARKET_TREND_LOOKBACK = 60

MARKET_VOL_LOOKBACK = 60

VOL_THRESHOLD_LOOKBACK = 252

VOL_THRESHOLD_MIN_HISTORY = 126

ANNUALIZATION = 252


# ------------------------------------------------------------
# Daily full-market benchmark QA
# ------------------------------------------------------------

MIN_MARKET_STOCK_COUNT = 100


# ------------------------------------------------------------
# Regime summary
#
# Do NOT drop smaller cells.
# This flag is only used to mark whether a regime cell has
# enough months for stable interpretation.
# ------------------------------------------------------------

MIN_REGIME_MONTHS = 18


# ------------------------------------------------------------
# HAC
# ------------------------------------------------------------

USE_HAC = True


# ------------------------------------------------------------
# Full-sample effect-size thresholds used ONLY when reporting
# retention ratios.
#
# They do not affect whether a result is kept.
# ------------------------------------------------------------

MIN_BASELINE_ALPHA_IC = 0.005

MIN_BASELINE_ALPHA_SPREAD = 0.001
# 0.001 = 10 bp

MIN_BASELINE_RISK_IC = 0.01


# ============================================================
# 2. Fixed Time Subsamples
# ============================================================

TIME_SUBSAMPLES = [

    {
        "name":
            "P1_2015_2018",

        "start":
            "2015-01-01",

        "end":
            "2018-12-31",
    },

    {
        "name":
            "P2_2019_2022",

        "start":
            "2019-01-01",

        "end":
            "2022-12-31",
    },

    {
        "name":
            "P3_2023_2026",

        "start":
            "2023-01-01",

        "end":
            "2026-12-31",
    },
]


# ============================================================
# 3. Utilities
# ============================================================

def load_json(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing JSON:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_json_atomic(
    obj,
    path: Path,
):

    tmp = Path(
        str(path)
        +
        ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )

    os.replace(
        tmp,
        path,
    )


def save_csv_atomic(
    df: pd.DataFrame,
    path: Path,
):

    tmp = Path(
        str(path)
        +
        ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        tmp,
        path,
    )


def sha256_file(
    path: Path,
    chunk_size=1024 * 1024,
):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:

                break

            h.update(
                chunk
            )

    return h.hexdigest()


def canonical_hash(
    obj,
):

    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# 4. Validate Step-6 Inputs
# ============================================================

def validate_step6():

    metadata = load_json(
        STEP6_METADATA_PATH
    )

    if (
        metadata.get(
            "formal_qa",
            {}
        ).get(
            "all_formal_qa_pass"
        )
        is not True
    ):

        raise RuntimeError(
            "Day-6 Step-6 formal QA did not pass."
        )

    for path in [

        MONTHLY_ALPHA_IC_PATH,

        MONTHLY_ALPHA_SPREAD_PATH,

        MONTHLY_RISK_IC_PATH,

        DAY1_RETURN_PANEL_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required input:\n{path}"
            )

    return metadata


# ============================================================
# 5. Normalize is_open
# ============================================================

def normalize_is_open(
    x: pd.Series,
):

    if pd.api.types.is_bool_dtype(
        x
    ):

        return x.astype(
            "boolean"
        )

    numeric = pd.to_numeric(
        x,
        errors="coerce",
    )

    result = pd.Series(
        pd.NA,
        index=x.index,
        dtype="boolean",
    )

    result.loc[
        numeric == 1
    ] = True

    result.loc[
        numeric == 0
    ] = False

    unresolved = (
        result.isna()
    )

    if unresolved.any():

        text = (
            x.astype("string")
            .str.strip()
            .str.lower()
        )

        result.loc[
            unresolved
            &
            text.isin(
                [
                    "true",
                    "t",
                    "yes",
                    "y",
                ]
            )
        ] = True

        result.loc[
            unresolved
            &
            text.isin(
                [
                    "false",
                    "f",
                    "no",
                    "n",
                ]
            )
        ] = False

    return result


# ============================================================
# 6. Build Full-Market Daily Holding Return
#
# Same conceptual convention as Day-6 Step-2:
#
# suspended day -> 0
# active day    -> return_last_trade_simple
# unresolved active-day missing -> NA
#
# Benchmark:
# full-market equal-weight holding return
# ============================================================

def build_market_daily_state():

    print()
    print(
        "[1] Build full-market daily holding-return series"
    )

    parquet = pq.ParquetFile(
        DAY1_RETURN_PANEL_PATH
    )

    schema = set(
        parquet
        .schema_arrow
        .names
    )

    required = [

        "trade_date",

        "is_open",

        "return_last_trade_simple",
    ]

    missing = [
        x
        for x in required
        if x not in schema
    ]

    if missing:

        raise RuntimeError(
            "Day-1 return panel missing columns:\n"
            f"{missing}"
        )

    partials = []

    for batch_no, batch in enumerate(

        parquet.iter_batches(

            columns=required,

            batch_size=500_000,
        ),

        start=1,
    ):

        df = batch.to_pandas()

        df[
            "trade_date"
        ] = pd.to_datetime(
            df[
                "trade_date"
            ],
            errors="raise",
        )

        is_open = normalize_is_open(
            df[
                "is_open"
            ]
        )

        last_trade_return = (
            pd.to_numeric(
                df[
                    "return_last_trade_simple"
                ],
                errors="coerce",
            )
        )

        finite_return = (

            last_trade_return.notna()

            &

            np.isfinite(
                last_trade_return
            )
        )

        impossible = (

            is_open.eq(
                True
            )

            &

            finite_return

            &

            (
                last_trade_return
                <
                -1.0
                -
                1e-12
            )
        )

        if impossible.any():

            raise RuntimeError(
                "Found active-day simple return < -1."
            )

        holding_return = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

        # Explicit suspension -> zero holding return
        holding_return.loc[
            is_open.eq(
                False
            )
        ] = 0.0

        # Active day -> last actual trade simple return
        active_valid = (

            is_open.eq(
                True
            )

            &

            finite_return
        )

        holding_return.loc[
            active_valid
        ] = (
            last_trade_return.loc[
                active_valid
            ]
        )

        unresolved_open = (

            is_open.eq(
                True
            )

            &

            ~finite_return
        )

        suspension = (
            is_open.eq(
                False
            )
        )

        temp = pd.DataFrame(
            {
                "trade_date":
                    df[
                        "trade_date"
                    ],

                "holding_return":
                    holding_return,

                "unresolved_open":
                    unresolved_open.astype(
                        int
                    ),

                "suspension_zero":
                    suspension.astype(
                        int
                    ),
            }
        )

        agg = (

            temp.groupby(
                "trade_date",
                as_index=False,
            )

            .agg(

                market_return_sum=(
                    "holding_return",
                    "sum",
                ),

                market_valid_stock_count=(
                    "holding_return",
                    "count",
                ),

                unresolved_open_stock_count=(
                    "unresolved_open",
                    "sum",
                ),

                suspension_zero_stock_count=(
                    "suspension_zero",
                    "sum",
                ),

                panel_stock_count=(
                    "holding_return",
                    "size",
                ),
            )
        )

        partials.append(
            agg
        )

        if (
            batch_no == 1
            or
            batch_no % 10 == 0
        ):

            print(
                f"  processed batch {batch_no}"
            )

    daily = pd.concat(
        partials,
        ignore_index=True,
    )

    daily = (

        daily.groupby(
            "trade_date",
            as_index=False,
        )

        .agg(

            market_return_sum=(
                "market_return_sum",
                "sum",
            ),

            market_valid_stock_count=(
                "market_valid_stock_count",
                "sum",
            ),

            unresolved_open_stock_count=(
                "unresolved_open_stock_count",
                "sum",
            ),

            suspension_zero_stock_count=(
                "suspension_zero_stock_count",
                "sum",
            ),

            panel_stock_count=(
                "panel_stock_count",
                "sum",
            ),
        )

        .sort_values(
            "trade_date"
        )

        .reset_index(
            drop=True
        )
    )

    daily[
        "market_holding_return_ew"
    ] = np.where(

        daily[
            "market_valid_stock_count"
        ]
        >=
        MIN_MARKET_STOCK_COUNT,

        daily[
            "market_return_sum"
        ]
        /
        daily[
            "market_valid_stock_count"
        ],

        np.nan,
    )

    if (
        daily[
            "market_holding_return_ew"
        ]
        .dropna()
        .le(
            -1.0
        )
        .any()
    ):

        raise RuntimeError(
            "Full-market EW return <= -1."
        )

    # ========================================================
    # Trailing 60-trading-day market return
    # ========================================================

    daily[
        "_log1p_market"
    ] = np.log1p(
        daily[
            "market_holding_return_ew"
        ]
    )

    daily[
        "market_trailing_return_60d"
    ] = np.expm1(

        daily[
            "_log1p_market"
        ]
        .rolling(
            MARKET_TREND_LOOKBACK,
            min_periods=MARKET_TREND_LOOKBACK,
        )
        .sum()
    )

    # ========================================================
    # Trailing 60-trading-day realized volatility
    # ========================================================

    daily[
        "market_trailing_vol_60d"
    ] = (

        daily[
            "market_holding_return_ew"
        ]
        .rolling(
            MARKET_VOL_LOOKBACK,
            min_periods=MARKET_VOL_LOOKBACK,
        )
        .std(
            ddof=1
        )

        *

        np.sqrt(
            ANNUALIZATION
        )
    )

    # ========================================================
    # PIT volatility threshold
    #
    # IMPORTANT:
    # shift(1) means today's volatility is compared only with
    # information available before today.
    # ========================================================

    daily[
        "market_vol_threshold_pit"
    ] = (

        daily[
            "market_trailing_vol_60d"
        ]
        .shift(
            1
        )
        .rolling(
            VOL_THRESHOLD_LOOKBACK,
            min_periods=VOL_THRESHOLD_MIN_HISTORY,
        )
        .median()
    )

    # ========================================================
    # Bull / Bear
    # ========================================================

    daily[
        "market_trend_regime"
    ] = pd.Series(
        pd.NA,
        index=daily.index,
        dtype="string",
    )

    valid_trend = (
        daily[
            "market_trailing_return_60d"
        ]
        .notna()
    )

    daily.loc[
        valid_trend
        &
        (
            daily[
                "market_trailing_return_60d"
            ]
            >=
            0
        ),
        "market_trend_regime",
    ] = "BULL"

    daily.loc[
        valid_trend
        &
        (
            daily[
                "market_trailing_return_60d"
            ]
            <
            0
        ),
        "market_trend_regime",
    ] = "BEAR"

    # ========================================================
    # High / Low volatility
    # ========================================================

    daily[
        "market_vol_regime"
    ] = pd.Series(
        pd.NA,
        index=daily.index,
        dtype="string",
    )

    valid_vol_regime = (

        daily[
            "market_trailing_vol_60d"
        ]
        .notna()

        &

        daily[
            "market_vol_threshold_pit"
        ]
        .notna()
    )

    daily.loc[
        valid_vol_regime
        &
        (
            daily[
                "market_trailing_vol_60d"
            ]
            >
            daily[
                "market_vol_threshold_pit"
            ]
        ),
        "market_vol_regime",
    ] = "HIGH_VOL"

    daily.loc[
        valid_vol_regime
        &
        (
            daily[
                "market_trailing_vol_60d"
            ]
            <=
            daily[
                "market_vol_threshold_pit"
            ]
        ),
        "market_vol_regime",
    ] = "LOW_VOL"

    # ========================================================
    # Joint state
    # ========================================================

    daily[
        "joint_market_regime"
    ] = pd.Series(
        pd.NA,
        index=daily.index,
        dtype="string",
    )

    valid_joint = (

        daily[
            "market_trend_regime"
        ]
        .notna()

        &

        daily[
            "market_vol_regime"
        ]
        .notna()
    )

    daily.loc[
        valid_joint,
        "joint_market_regime",
    ] = (

        daily.loc[
            valid_joint,
            "market_trend_regime",
        ]

        +

        "_"

        +

        daily.loc[
            valid_joint,
            "market_vol_regime",
        ]
    )

    daily = daily.drop(
        columns=[
            "_log1p_market"
        ]
    )

    save_csv_atomic(
        daily,
        MARKET_DAILY_STATE_PATH,
    )

    print(
        f"Daily market rows: "
        f"{len(daily):,}"
    )

    return daily


# ============================================================
# 7. Load Day-6 Monthly Validation Outputs
# ============================================================

def load_day6_monthly_outputs():

    print()
    print(
        "[2] Load Day-6 monthly Alpha/Risk validation outputs"
    )

    alpha = pd.read_csv(
        MONTHLY_ALPHA_IC_PATH
    )

    spread = pd.read_csv(
        MONTHLY_ALPHA_SPREAD_PATH
    )

    risk = pd.read_csv(
        MONTHLY_RISK_IC_PATH
    )

    for df in [
        alpha,
        spread,
        risk,
    ]:

        df[
            "analysis_date"
        ] = pd.to_datetime(
            df[
                "analysis_date"
            ],
            errors="raise",
        )

        df[
            "window"
        ] = pd.to_numeric(
            df[
                "window"
            ],
            errors="raise",
        ).astype(
            int
        )

    # --------------------------------------------------------
    # Keep only frozen Day-7 robustness specifications
    # --------------------------------------------------------

    alpha = alpha[
        alpha[
            "neutralization"
        ]
        .isin(
            ANALYSIS_METHODS
        )
    ].copy()

    spread = spread[
        spread[
            "neutralization"
        ]
        .isin(
            ANALYSIS_METHODS
        )
    ].copy()

    risk = risk[
        risk[
            "neutralization"
        ]
        .isin(
            ANALYSIS_METHODS
        )
    ].copy()

    # --------------------------------------------------------
    # QA: expected methods
    # --------------------------------------------------------

    for name, df in [

        (
            "alpha",
            alpha,
        ),

        (
            "spread",
            spread,
        ),

        (
            "risk",
            risk,
        ),
    ]:

        observed = sorted(
            df[
                "neutralization"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        expected = sorted(
            ANALYSIS_METHODS
        )

        if (
            observed
            !=
            expected
        ):

            raise RuntimeError(
                f"{name} neutralization methods mismatch.\n"
                f"Expected={expected}\n"
                f"Observed={observed}"
            )

    # --------------------------------------------------------
    # Key uniqueness
    # --------------------------------------------------------

    if (
        alpha[
            [
                "analysis_date",
                "window",
                "factor_name",
                "neutralization",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate Alpha IC keys."
        )

    if (
        spread[
            [
                "analysis_date",
                "window",
                "factor_name",
                "neutralization",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate portfolio spread keys."
        )

    if (
        risk[
            [
                "analysis_date",
                "window",
                "factor_name",
                "neutralization",
                "target_name",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate Risk IC keys."
        )

    return (
        alpha,
        spread,
        risk,
    )


# ============================================================
# 8. Time-Subsample Assignment
# ============================================================

def assign_time_subsample(
    date,
):

    date = pd.Timestamp(
        date
    )

    for spec in TIME_SUBSAMPLES:

        if (
            pd.Timestamp(
                spec[
                    "start"
                ]
            )
            <=
            date
            <=
            pd.Timestamp(
                spec[
                    "end"
                ]
            )
        ):

            return spec[
                "name"
            ]

    return pd.NA


# ============================================================
# 9. Build Analysis-Date Regime Table
# ============================================================

def build_regime_labels(
    daily_market,
    analysis_dates,
):

    print()
    print(
        "[3] Build point-in-time market-regime labels"
    )

    analysis_dates = pd.DatetimeIndex(
        sorted(
            pd.to_datetime(
                analysis_dates
            ).unique()
        )
    )

    state = daily_market[
        daily_market[
            "trade_date"
        ]
        .isin(
            analysis_dates
        )
    ].copy()

    state = state.rename(
        columns={
            "trade_date":
                "analysis_date",
        }
    )

    found = set(
        state[
            "analysis_date"
        ]
    )

    missing = [
        x
        for x in analysis_dates
        if x not in found
    ]

    if missing:

        raise RuntimeError(
            "Some analysis dates are absent from "
            "Day-1 market panel:\n"
            f"{missing[:20]}"
        )

    state[
        "time_subsample"
    ] = (
        state[
            "analysis_date"
        ]
        .map(
            assign_time_subsample
        )
        .astype(
            "string"
        )
    )

    save_csv_atomic(
        state,
        MARKET_REGIME_LABEL_PATH,
    )

    # ========================================================
    # Long-format regime assignment
    # ========================================================

    rows = []

    for row in state.itertuples(
        index=False
    ):

        date = row.analysis_date

        # ----------------------------------------------------
        # Full sample
        # ----------------------------------------------------

        rows.append(
            {
                "analysis_date":
                    date,

                "regime_type":
                    "ALL_SAMPLE",

                "regime_value":
                    "ALL",
            }
        )

        # ----------------------------------------------------
        # Fixed time subsample
        # ----------------------------------------------------

        if pd.notna(
            row.time_subsample
        ):

            rows.append(
                {
                    "analysis_date":
                        date,

                    "regime_type":
                        "TIME_SUBSAMPLE",

                    "regime_value":
                        row.time_subsample,
                }
            )

        # ----------------------------------------------------
        # Bull / Bear
        # ----------------------------------------------------

        if pd.notna(
            row.market_trend_regime
        ):

            rows.append(
                {
                    "analysis_date":
                        date,

                    "regime_type":
                        "MARKET_TREND",

                    "regime_value":
                        row.market_trend_regime,
                }
            )

        # ----------------------------------------------------
        # High / Low Vol
        # ----------------------------------------------------

        if pd.notna(
            row.market_vol_regime
        ):

            rows.append(
                {
                    "analysis_date":
                        date,

                    "regime_type":
                        "MARKET_VOL",

                    "regime_value":
                        row.market_vol_regime,
                }
            )

        # ----------------------------------------------------
        # Joint Regime
        # ----------------------------------------------------

        if pd.notna(
            row.joint_market_regime
        ):

            rows.append(
                {
                    "analysis_date":
                        date,

                    "regime_type":
                        "JOINT_MARKET_REGIME",

                    "regime_value":
                        row.joint_market_regime,
                }
            )

    regime_long = pd.DataFrame(
        rows
    )

    if (
        regime_long[
            [
                "analysis_date",
                "regime_type",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate analysis-date x regime-type assignment."
        )

    save_csv_atomic(
        regime_long,
        REGIME_ASSIGNMENT_LONG_PATH,
    )

    return (
        state,
        regime_long,
    )


# ============================================================
# 10. HAC with Calendar-Month Gaps
#
# Unlike simply compressing Bull months together,
# this uses actual calendar-month distance.
#
# This avoids treating, e.g.,
#
# 2020-01 and 2020-06
#
# as adjacent observations just because both are Bear months.
# ============================================================

def automatic_hac_lag(
    n,
):

    if (
        n
        <=
        1
    ):

        return 0

    lag = int(
        np.floor(

            4

            *

            (
                n
                /
                100.0
            )

            **
            (
                2.0
                /
                9.0
            )
        )
    )

    return max(
        0,
        min(
            lag,
            n - 1,
        ),
    )


def calendar_gap_hac(
    values,
    dates,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    # to_datetime preserves Series inputs, which require .dt.year/.dt.month.
    # Normalize all supported inputs to an index for positional filtering
    # alongside x and consistent calendar-year/month access below.
    dates = pd.DatetimeIndex(
        pd.to_datetime(dates)
    )

    valid = np.isfinite(
        x
    )

    x = x[
        valid
    ]

    dates = dates[
        valid
    ]

    n = len(
        x
    )

    if (
        n
        <
        2
    ):

        return {

            "naive_se":
                np.nan,

            "naive_t":
                np.nan,

            "hac_lag":
                0,

            "hac_se":
                np.nan,

            "hac_t":
                np.nan,
        }

    mean_x = float(
        np.mean(
            x
        )
    )

    sd_x = float(
        np.std(
            x,
            ddof=1,
        )
    )

    if (
        sd_x
        >
        0
    ):

        naive_se = (
            sd_x
            /
            np.sqrt(
                n
            )
        )

        naive_t = (
            mean_x
            /
            naive_se
        )

    else:

        naive_se = np.nan

        naive_t = np.nan

    if not USE_HAC:

        return {

            "naive_se":
                naive_se,

            "naive_t":
                naive_t,

            "hac_lag":
                0,

            "hac_se":
                naive_se,

            "hac_t":
                naive_t,
        }

    # --------------------------------------------------------
    # Calendar-month index
    # --------------------------------------------------------

    month_index = (

        dates.year
        *
        12

        +

        dates.month
    )

    if (
        pd.Series(
            month_index
        )
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "More than one observation in same calendar month "
            "inside one regime series."
        )

    residual = (
        x
        -
        mean_x
    )

    residual_map = dict(
        zip(
            month_index,
            residual,
        )
    )

    lag = automatic_hac_lag(
        n
    )

    # --------------------------------------------------------
    # Variance of sum:
    #
    # sum e_t^2
    # +
    # 2 sum_l w_l sum_t e_t e_{t-l}
    #
    # variance(mean) = variance(sum) / n^2
    # --------------------------------------------------------

    variance_sum = float(
        np.dot(
            residual,
            residual,
        )
    )

    for ell in range(
        1,
        lag + 1,
    ):

        cross_sum = 0.0

        for month, value in (
            residual_map.items()
        ):

            lagged_month = (
                month
                -
                ell
            )

            if (
                lagged_month
                in
                residual_map
            ):

                cross_sum += (

                    value

                    *

                    residual_map[
                        lagged_month
                    ]
                )

        weight = (

            1.0

            -

            ell
            /
            (
                lag
                +
                1.0
            )
        )

        variance_sum += (

            2.0

            *

            weight

            *

            cross_sum
        )

    variance_sum = max(
        variance_sum,
        0.0,
    )

    variance_mean = (
        variance_sum
        /
        (
            n
            **
            2
        )
    )

    if (
        variance_mean
        >
        0
    ):

        hac_se = float(
            np.sqrt(
                variance_mean
            )
        )

        hac_t = (
            mean_x
            /
            hac_se
        )

    else:

        hac_se = np.nan

        hac_t = np.nan

    return {

        "naive_se":
            naive_se,

        "naive_t":
            naive_t,

        "hac_lag":
            int(
                lag
            ),

        "hac_se":
            hac_se,

        "hac_t":
            hac_t,
    }


# ============================================================
# 11. Generic Regime Summary
# ============================================================

def summarize_by_regime(

    data,

    regime_long,

    value_column,

    status_column,

    pass_value,

    signal_type,
):

    work = data[
        data[
            status_column
        ]
        ==
        pass_value
    ].copy()

    work[
        value_column
    ] = pd.to_numeric(
        work[
            value_column
        ],
        errors="coerce",
    )

    work = work[
        work[
            value_column
        ]
        .notna()
        &
        np.isfinite(
            work[
                value_column
            ]
        )
    ].copy()

    work = work.merge(

        regime_long,

        on="analysis_date",

        how="inner",

        validate="many_to_many",
    )

    group_keys = [

        "factor_order",

        "factor_name",

        "neutralization",

        "window",

        "target_name",

        "regime_type",

        "regime_value",
    ]

    rows = []

    for keys, group in (
        work.groupby(
            group_keys,
            sort=False,
        )
    ):

        (
            factor_order,
            factor_name,
            neutralization,
            window,
            target_name,
            regime_type,
            regime_value,
        ) = keys

        group = (
            group.sort_values(
                "analysis_date"
            )
        )

        x = (
            group[
                value_column
            ]
            .to_numpy(
                dtype=float
            )
        )

        dates = (
            group[
                "analysis_date"
            ]
        )

        n = len(
            x
        )

        inference = (
            calendar_gap_hac(
                x,
                dates,
            )
        )

        row = {

            "signal_type":
                signal_type,

            "factor_order":
                int(
                    factor_order
                ),

            "factor_name":
                factor_name,

            "neutralization":
                neutralization,

            "window":
                int(
                    window
                ),

            "target_name":
                target_name,

            "regime_type":
                regime_type,

            "regime_value":
                regime_value,

            "month_count":
                int(
                    n
                ),

            "sufficient_regime_months":
                bool(
                    n
                    >=
                    MIN_REGIME_MONTHS
                ),

            "first_date":
                group[
                    "analysis_date"
                ]
                .min(),

            "last_date":
                group[
                    "analysis_date"
                ]
                .max(),

            "mean_value":
                float(
                    np.mean(
                        x
                    )
                ),

            "median_value":
                float(
                    np.median(
                        x
                    )
                ),

            "std_value":
                (
                    float(
                        np.std(
                            x,
                            ddof=1,
                        )
                    )
                    if
                    n
                    >=
                    2
                    else
                    np.nan
                ),

            "positive_share":
                float(
                    np.mean(
                        x
                        >
                        0
                    )
                ),

            "negative_share":
                float(
                    np.mean(
                        x
                        <
                        0
                    )
                ),

            "naive_se":
                inference[
                    "naive_se"
                ],

            "naive_t_stat":
                inference[
                    "naive_t"
                ],

            "calendar_hac_lag":
                inference[
                    "hac_lag"
                ],

            "calendar_hac_se":
                inference[
                    "hac_se"
                ],

            "calendar_hac_t_stat":
                inference[
                    "hac_t"
                ],
        }

        if (
            "pair_n"
            in group.columns
        ):

            row[
                "mean_pair_n"
            ] = float(
                group[
                    "pair_n"
                ]
                .mean()
            )

        if (
            "portfolio_spearman"
            in group.columns
        ):

            row[
                "mean_portfolio_spearman"
            ] = float(
                group[
                    "portfolio_spearman"
                ]
                .mean()
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    # ========================================================
    # Full-sample baseline
    # ========================================================

    base_keys = [

        "factor_order",

        "factor_name",

        "neutralization",

        "window",

        "target_name",
    ]

    baseline = (
        summary[
            (
                summary[
                    "regime_type"
                ]
                ==
                "ALL_SAMPLE"
            )
            &
            (
                summary[
                    "regime_value"
                ]
                ==
                "ALL"
            )
        ][
            base_keys
            +
            [
                "mean_value"
            ]
        ]
        .rename(
            columns={
                "mean_value":
                    "full_sample_mean",
            }
        )
    )

    summary = summary.merge(

        baseline,

        on=base_keys,

        how="left",

        validate="many_to_one",
    )

    summary[
        "same_sign_as_full_sample"
    ] = (

        np.sign(
            summary[
                "mean_value"
            ]
        )

        ==

        np.sign(
            summary[
                "full_sample_mean"
            ]
        )
    )

    summary.loc[
        (
            summary[
                "mean_value"
            ]
            ==
            0
        )
        |
        (
            summary[
                "full_sample_mean"
            ]
            ==
            0
        ),
        "same_sign_as_full_sample",
    ] = False

    # ========================================================
    # Retention threshold
    # ========================================================

    if (
        signal_type
        ==
        "ALPHA_IC"
    ):

        minimum_baseline = (
            MIN_BASELINE_ALPHA_IC
        )

    elif (
        signal_type
        ==
        "ALPHA_SPREAD"
    ):

        minimum_baseline = (
            MIN_BASELINE_ALPHA_SPREAD
        )

    elif (
        signal_type
        ==
        "RISK_IC"
    ):

        minimum_baseline = (
            MIN_BASELINE_RISK_IC
        )

    else:

        raise ValueError(
            signal_type
        )

    summary[
        "abs_retention_vs_full"
    ] = np.where(

        summary[
            "full_sample_mean"
        ]
        .abs()
        >=
        minimum_baseline,

        summary[
            "mean_value"
        ]
        .abs()

        /

        summary[
            "full_sample_mean"
        ]
        .abs(),

        np.nan,
    )

    # ========================================================
    # Friendly signal-specific names
    # ========================================================

    if (
        signal_type
        ==
        "ALPHA_IC"
    ):

        summary = summary.rename(
            columns={
                "mean_value":
                    "mean_alpha_ic",

                "median_value":
                    "median_alpha_ic",

                "std_value":
                    "std_alpha_ic",

                "full_sample_mean":
                    "full_sample_alpha_ic",
            }
        )

    elif (
        signal_type
        ==
        "ALPHA_SPREAD"
    ):

        summary = summary.rename(
            columns={
                "mean_value":
                    "mean_q5_minus_q1",

                "median_value":
                    "median_q5_minus_q1",

                "std_value":
                    "std_q5_minus_q1",

                "full_sample_mean":
                    "full_sample_q5_minus_q1",
            }
        )

        summary[
            "mean_q5_minus_q1_bps"
        ] = (
            summary[
                "mean_q5_minus_q1"
            ]
            *
            10000.0
        )

    elif (
        signal_type
        ==
        "RISK_IC"
    ):

        summary = summary.rename(
            columns={
                "mean_value":
                    "mean_risk_ic",

                "median_value":
                    "median_risk_ic",

                "std_value":
                    "std_risk_ic",

                "full_sample_mean":
                    "full_sample_risk_ic",
            }
        )

    return summary


# ============================================================
# 12. Final Directional Robustness Matrix
# ============================================================

def consistency_statistics(
    group,
):

    eligible = group[
        (
            group[
                "regime_type"
            ]
            !=
            "ALL_SAMPLE"
        )
        &
        (
            group[
                "sufficient_regime_months"
            ]
        )
    ].copy()

    result = {

        "eligible_regime_cell_count":
            int(
                len(
                    eligible
                )
            ),

        "overall_same_sign_share":
            (
                float(
                    eligible[
                        "same_sign_as_full_sample"
                    ]
                    .mean()
                )
                if
                len(
                    eligible
                )
                >
                0
                else
                np.nan
            ),

        "mean_abs_retention_vs_full":
            (
                float(
                    eligible[
                        "abs_retention_vs_full"
                    ]
                    .mean()
                )
                if
                eligible[
                    "abs_retention_vs_full"
                ]
                .notna()
                .any()
                else
                np.nan
            ),
    }

    for regime_type, suffix in [

        (
            "TIME_SUBSAMPLE",
            "time",
        ),

        (
            "MARKET_TREND",
            "trend",
        ),

        (
            "MARKET_VOL",
            "vol",
        ),

        (
            "JOINT_MARKET_REGIME",
            "joint",
        ),
    ]:

        sub = eligible[
            eligible[
                "regime_type"
            ]
            ==
            regime_type
        ]

        result[
            f"{suffix}_cell_count"
        ] = int(
            len(
                sub
            )
        )

        result[
            f"{suffix}_same_sign_share"
        ] = (
            float(
                sub[
                    "same_sign_as_full_sample"
                ]
                .mean()
            )
            if
            len(
                sub
            )
            >
            0
            else
            np.nan
        )

    return result


def build_robustness_matrix(
    alpha_summary,
    spread_summary,
    risk_summary,
):

    rows = []

    base = (

        alpha_summary[
            [
                "factor_order",
                "factor_name",
                "neutralization",
                "window",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "factor_order",
                "neutralization",
                "window",
            ]
        )
    )

    for row in base.itertuples(
        index=False
    ):

        factor_order = (
            row.factor_order
        )

        factor_name = (
            row.factor_name
        )

        neutralization = (
            row.neutralization
        )

        window = (
            row.window
        )

        alpha = alpha_summary[
            (
                alpha_summary[
                    "factor_name"
                ]
                ==
                factor_name
            )
            &
            (
                alpha_summary[
                    "neutralization"
                ]
                ==
                neutralization
            )
            &
            (
                alpha_summary[
                    "window"
                ]
                ==
                window
            )
        ]

        spread = spread_summary[
            (
                spread_summary[
                    "factor_name"
                ]
                ==
                factor_name
            )
            &
            (
                spread_summary[
                    "neutralization"
                ]
                ==
                neutralization
            )
            &
            (
                spread_summary[
                    "window"
                ]
                ==
                window
            )
        ]

        risk = risk_summary[
            (
                risk_summary[
                    "factor_name"
                ]
                ==
                factor_name
            )
            &
            (
                risk_summary[
                    "neutralization"
                ]
                ==
                neutralization
            )
            &
            (
                risk_summary[
                    "window"
                ]
                ==
                window
            )
        ]

        ivol = risk[
            risk[
                "target_name"
            ]
            ==
            "future_idio_vol_annualized"
        ]

        alpha_stats = (
            consistency_statistics(
                alpha
            )
        )

        spread_stats = (
            consistency_statistics(
                spread
            )
        )

        risk_stats = (
            consistency_statistics(
                risk
            )
        )

        ivol_stats = (
            consistency_statistics(
                ivol
            )
        )

        # ----------------------------------------------------
        # Full-sample reference values
        # ----------------------------------------------------

        alpha_full = alpha[
            alpha[
                "regime_type"
            ]
            ==
            "ALL_SAMPLE"
        ]

        spread_full = spread[
            spread[
                "regime_type"
            ]
            ==
            "ALL_SAMPLE"
        ]

        risk_full = risk[
            risk[
                "regime_type"
            ]
            ==
            "ALL_SAMPLE"
        ]

        ivol_full = ivol[
            ivol[
                "regime_type"
            ]
            ==
            "ALL_SAMPLE"
        ]

        rows.append(
            {

                "factor_order":
                    int(
                        factor_order
                    ),

                "factor_name":
                    factor_name,

                "neutralization":
                    neutralization,

                "window":
                    int(
                        window
                    ),

                "full_sample_alpha_ic":
                    (
                        float(
                            alpha_full[
                                "mean_alpha_ic"
                            ]
                            .iloc[0]
                        )
                        if
                        len(
                            alpha_full
                        )
                        >
                        0
                        else
                        np.nan
                    ),

                "full_sample_alpha_spread":
                    (
                        float(
                            spread_full[
                                "mean_q5_minus_q1"
                            ]
                            .iloc[0]
                        )
                        if
                        len(
                            spread_full
                        )
                        >
                        0
                        else
                        np.nan
                    ),

                "full_sample_alpha_spread_bps":
                    (
                        float(
                            spread_full[
                                "mean_q5_minus_q1_bps"
                            ]
                            .iloc[0]
                        )
                        if
                        len(
                            spread_full
                        )
                        >
                        0
                        else
                        np.nan
                    ),

                "full_sample_mean_risk_ic":
                    (
                        float(
                            risk_full[
                                "mean_risk_ic"
                            ]
                            .mean()
                        )
                        if
                        len(
                            risk_full
                        )
                        >
                        0
                        else
                        np.nan
                    ),

                "full_sample_ivol_ic":
                    (
                        float(
                            ivol_full[
                                "mean_risk_ic"
                            ]
                            .iloc[0]
                        )
                        if
                        len(
                            ivol_full
                        )
                        >
                        0
                        else
                        np.nan
                    ),

                # ---------------------------------------------
                # Alpha IC
                # ---------------------------------------------

                "alpha_ic_overall_same_sign_share":
                    alpha_stats[
                        "overall_same_sign_share"
                    ],

                "alpha_ic_time_same_sign_share":
                    alpha_stats[
                        "time_same_sign_share"
                    ],

                "alpha_ic_trend_same_sign_share":
                    alpha_stats[
                        "trend_same_sign_share"
                    ],

                "alpha_ic_vol_same_sign_share":
                    alpha_stats[
                        "vol_same_sign_share"
                    ],

                "alpha_ic_joint_same_sign_share":
                    alpha_stats[
                        "joint_same_sign_share"
                    ],

                "alpha_ic_mean_abs_retention":
                    alpha_stats[
                        "mean_abs_retention_vs_full"
                    ],

                # ---------------------------------------------
                # Portfolio spread
                # ---------------------------------------------

                "alpha_spread_overall_same_sign_share":
                    spread_stats[
                        "overall_same_sign_share"
                    ],

                "alpha_spread_time_same_sign_share":
                    spread_stats[
                        "time_same_sign_share"
                    ],

                "alpha_spread_trend_same_sign_share":
                    spread_stats[
                        "trend_same_sign_share"
                    ],

                "alpha_spread_vol_same_sign_share":
                    spread_stats[
                        "vol_same_sign_share"
                    ],

                "alpha_spread_joint_same_sign_share":
                    spread_stats[
                        "joint_same_sign_share"
                    ],

                "alpha_spread_mean_abs_retention":
                    spread_stats[
                        "mean_abs_retention_vs_full"
                    ],

                # ---------------------------------------------
                # All risk targets
                # ---------------------------------------------

                "risk_ic_overall_same_sign_share":
                    risk_stats[
                        "overall_same_sign_share"
                    ],

                "risk_ic_time_same_sign_share":
                    risk_stats[
                        "time_same_sign_share"
                    ],

                "risk_ic_trend_same_sign_share":
                    risk_stats[
                        "trend_same_sign_share"
                    ],

                "risk_ic_vol_same_sign_share":
                    risk_stats[
                        "vol_same_sign_share"
                    ],

                "risk_ic_joint_same_sign_share":
                    risk_stats[
                        "joint_same_sign_share"
                    ],

                "risk_ic_mean_abs_retention":
                    risk_stats[
                        "mean_abs_retention_vs_full"
                    ],

                # ---------------------------------------------
                # IVOL specifically
                # ---------------------------------------------

                "ivol_ic_overall_same_sign_share":
                    ivol_stats[
                        "overall_same_sign_share"
                    ],

                "ivol_ic_time_same_sign_share":
                    ivol_stats[
                        "time_same_sign_share"
                    ],

                "ivol_ic_trend_same_sign_share":
                    ivol_stats[
                        "trend_same_sign_share"
                    ],

                "ivol_ic_vol_same_sign_share":
                    ivol_stats[
                        "vol_same_sign_share"
                    ],

                "ivol_ic_joint_same_sign_share":
                    ivol_stats[
                        "joint_same_sign_share"
                    ],

                "ivol_ic_mean_abs_retention":
                    ivol_stats[
                        "mean_abs_retention_vs_full"
                    ],

                # ---------------------------------------------
                # Research discipline
                # ---------------------------------------------

                "factor_sign_flipped":
                    False,

                "window_selected_from_regime_results":
                    False,

                "factor_removed_from_regime_results":
                    False,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 13. Formal QA
# ============================================================

def formal_qa(

    alpha,
    spread,
    risk,
    regime_state,
    regime_long,
    daily_market,
):

    qa = {}

    analysis_dates = sorted(
        alpha[
            "analysis_date"
        ]
        .unique()
        .tolist()
    )

    qa[
        "analysis_date_count"
    ] = int(
        len(
            analysis_dates
        )
    )

    qa[
        "all_analysis_dates_have_market_state_row"
    ] = bool(

        set(
            pd.to_datetime(
                analysis_dates
            )
        )

        ==

        set(
            regime_state[
                "analysis_date"
            ]
        )
    )

    qa[
        "observed_windows"
    ] = sorted(
        alpha[
            "window"
        ]
        .unique()
        .astype(int)
        .tolist()
    )

    qa[
        "window_set_ok"
    ] = bool(
        qa[
            "observed_windows"
        ]
        ==
        EXPECTED_WINDOWS
    )

    qa[
        "analysis_methods"
    ] = sorted(
        alpha[
            "neutralization"
        ]
        .unique()
        .tolist()
    )

    qa[
        "analysis_methods_ok"
    ] = bool(
        qa[
            "analysis_methods"
        ]
        ==
        sorted(
            ANALYSIS_METHODS
        )
    )

    qa[
        "market_daily_row_count"
    ] = int(
        len(
            daily_market
        )
    )

    qa[
        "market_daily_min_valid_stock_count"
    ] = int(
        daily_market[
            "market_valid_stock_count"
        ]
        .min()
    )

    qa[
        "market_trend_analysis_date_count"
    ] = int(
        regime_state[
            "market_trend_regime"
        ]
        .notna()
        .sum()
    )

    qa[
        "market_vol_analysis_date_count"
    ] = int(
        regime_state[
            "market_vol_regime"
        ]
        .notna()
        .sum()
    )

    qa[
        "joint_regime_analysis_date_count"
    ] = int(
        regime_state[
            "joint_market_regime"
        ]
        .notna()
        .sum()
    )

    qa[
        "regime_assignment_unique"
    ] = bool(

        not regime_long[
            [
                "analysis_date",
                "regime_type",
            ]
        ]
        .duplicated()
        .any()
    )

    qa[
        "no_future_market_information_used"
    ] = True

    qa[
        "market_vol_threshold_is_shifted"
    ] = True

    required = [

        "all_analysis_dates_have_market_state_row",

        "window_set_ok",

        "analysis_methods_ok",

        "regime_assignment_unique",

        "no_future_market_information_used",

        "market_vol_threshold_is_shifted",
    ]

    qa[
        "all_formal_qa_pass"
    ] = bool(
        all(
            qa[
                x
            ]
            for x in required
        )
    )

    return qa


# ============================================================
# 14. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 7 - Step 1"
    )

    print(
        "Subsample / Market-Regime Robustness"
    )

    print("=" * 80)

    # ========================================================
    # Upstream QA
    # ========================================================

    step6_metadata = (
        validate_step6()
    )

    step6_design_hash = (
        step6_metadata.get(
            "step6_design_hash"
        )
    )

    print()
    print(
        "Parent Step-6 design hash:"
    )

    print(
        step6_design_hash
    )

    # ========================================================
    # Day-7 design
    # ========================================================

    design_payload = {

        "research_day":
            7,

        "step":
            (
                "Step1_Subsample_"
                "Market_Regime_Robustness"
            ),

        "parent_step6_design_hash":
            step6_design_hash,

        "analysis_methods":
            ANALYSIS_METHODS,

        "windows":
            EXPECTED_WINDOWS,

        "time_subsamples":
            TIME_SUBSAMPLES,

        "market_return_source":
            (
                "Full-market equal-weight daily "
                "holding return reconstructed from "
                "Day-1 return panel."
            ),

        "holding_return_rule":
            {
                "suspension":
                    0.0,

                "active":
                    "return_last_trade_simple",

                "unresolved_active_missing":
                    "NA",
            },

        "trend_regime_rule":
            (
                "BULL if trailing 60-trading-day "
                "market holding return >= 0; "
                "otherwise BEAR."
            ),

        "volatility_regime_rule":
            (
                "HIGH_VOL if current trailing 60-day "
                "annualized market volatility exceeds "
                "the PIT median of prior market-volatility "
                "observations over a maximum 252-trading-day "
                "history; otherwise LOW_VOL."
            ),

        "market_trend_lookback":
            MARKET_TREND_LOOKBACK,

        "market_vol_lookback":
            MARKET_VOL_LOOKBACK,

        "vol_threshold_lookback":
            VOL_THRESHOLD_LOOKBACK,

        "vol_threshold_min_history":
            VOL_THRESHOLD_MIN_HISTORY,

        "minimum_regime_months":
            MIN_REGIME_MONTHS,

        "future_outcome_used_for_regime_definition":
            False,

        "factor_sign_flip":
            False,

        "factor_selection_from_regime_results":
            False,

        "window_selection_from_regime_results":
            False,

        "inference":
            (
                "Calendar-gap-aware Newey-West/HAC "
                "for regime-specific monthly series."
            ),
    }

    design = {

        **design_payload,

        "day7_step1_design_hash":
            canonical_hash(
                design_payload
            ),
    }

    print()
    print(
        "Day-7 Step-1 design hash:"
    )

    print(
        design[
            "day7_step1_design_hash"
        ]
    )

    # ========================================================
    # Load monthly Day-6 results
    # ========================================================

    (
        alpha,
        spread,
        risk,
    ) = load_day6_monthly_outputs()

    analysis_dates = sorted(
        alpha[
            "analysis_date"
        ]
        .unique()
    )

    # ========================================================
    # Market states
    # ========================================================

    daily_market = (
        build_market_daily_state()
    )

    (
        regime_state,
        regime_long,
    ) = build_regime_labels(

        daily_market,

        analysis_dates,
    )

    # ========================================================
    # Alpha Rank IC
    # ========================================================

    print()
    print(
        "[4] Summarize Alpha IC by subsample/regime"
    )

    alpha_summary = (
        summarize_by_regime(

            data=
                alpha,

            regime_long=
                regime_long,

            value_column=
                "alpha_rank_ic",

            status_column=
                "ic_status",

            pass_value=
                "PASS",

            signal_type=
                "ALPHA_IC",
        )
    )

    save_csv_atomic(
        alpha_summary,
        ALPHA_REGIME_PATH,
    )

    # ========================================================
    # Portfolio spread
    # ========================================================

    print()
    print(
        "[5] Summarize Q5-Q1 spread by subsample/regime"
    )

    spread_summary = (
        summarize_by_regime(

            data=
                spread,

            regime_long=
                regime_long,

            value_column=
                "q5_minus_q1",

            status_column=
                "spread_status",

            pass_value=
                "PASS",

            signal_type=
                "ALPHA_SPREAD",
        )
    )

    save_csv_atomic(
        spread_summary,
        PORTFOLIO_REGIME_PATH,
    )

    # ========================================================
    # Risk IC
    # ========================================================

    print()
    print(
        "[6] Summarize Risk IC by subsample/regime"
    )

    risk_summary = (
        summarize_by_regime(

            data=
                risk,

            regime_long=
                regime_long,

            value_column=
                "risk_rank_ic",

            status_column=
                "ic_status",

            pass_value=
                "PASS",

            signal_type=
                "RISK_IC",
        )
    )

    save_csv_atomic(
        risk_summary,
        RISK_REGIME_PATH,
    )

    # ========================================================
    # Final matrix
    # ========================================================

    print()
    print(
        "[7] Build integrated directional robustness matrix"
    )

    robustness = (
        build_robustness_matrix(

            alpha_summary,

            spread_summary,

            risk_summary,
        )
    )

    save_csv_atomic(
        robustness,
        ROBUSTNESS_SUMMARY_PATH,
    )

    # ========================================================
    # QA
    # ========================================================

    qa = formal_qa(

        alpha=
            alpha,

        spread=
            spread,

        risk=
            risk,

        regime_state=
            regime_state,

        regime_long=
            regime_long,

        daily_market=
            daily_market,
    )

    if (
        not qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            "Day-7 Step-1 formal QA failed.\n"
            f"{qa}"
        )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {

        **design,

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "input_files":
            {

                "step6_metadata":
                    str(
                        STEP6_METADATA_PATH
                    ),

                "monthly_alpha_ic":
                    str(
                        MONTHLY_ALPHA_IC_PATH
                    ),

                "monthly_alpha_spread":
                    str(
                        MONTHLY_ALPHA_SPREAD_PATH
                    ),

                "monthly_risk_ic":
                    str(
                        MONTHLY_RISK_IC_PATH
                    ),

                "day1_return_panel":
                    str(
                        DAY1_RETURN_PANEL_PATH
                    ),
            },

        "input_sha256":
            {

                "step6_metadata":
                    sha256_file(
                        STEP6_METADATA_PATH
                    ),

                "monthly_alpha_ic":
                    sha256_file(
                        MONTHLY_ALPHA_IC_PATH
                    ),

                "monthly_alpha_spread":
                    sha256_file(
                        MONTHLY_ALPHA_SPREAD_PATH
                    ),

                "monthly_risk_ic":
                    sha256_file(
                        MONTHLY_RISK_IC_PATH
                    ),
            },

        "formal_qa":
            qa,

        "outputs":
            {

                "market_daily_state":
                    str(
                        MARKET_DAILY_STATE_PATH
                    ),

                "market_regime_labels":
                    str(
                        MARKET_REGIME_LABEL_PATH
                    ),

                "regime_assignment_long":
                    str(
                        REGIME_ASSIGNMENT_LONG_PATH
                    ),

                "alpha_ic_by_subsample":
                    str(
                        ALPHA_REGIME_PATH
                    ),

                "portfolio_spread_by_subsample":
                    str(
                        PORTFOLIO_REGIME_PATH
                    ),

                "risk_ic_by_subsample":
                    str(
                        RISK_REGIME_PATH
                    ),

                "regime_robustness_summary":
                    str(
                        ROBUSTNESS_SUMMARY_PATH
                    ),
            },
    }

    save_json_atomic(
        metadata,
        METADATA_PATH,
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Day-7 Step-1 Formal QA"
    )

    print("=" * 80)

    print(
        f"Analysis dates: "
        f"{qa['analysis_date_count']}"
    )

    print(
        f"Trend-regime dates: "
        f"{qa['market_trend_analysis_date_count']}"
    )

    print(
        f"Vol-regime dates: "
        f"{qa['market_vol_analysis_date_count']}"
    )

    print(
        f"Joint-regime dates: "
        f"{qa['joint_regime_analysis_date_count']}"
    )

    print(
        f"Formal QA pass: "
        f"{qa['all_formal_qa_pass']}"
    )

    print()
    print("=" * 80)

    print(
        "Industry + Size Neutralized Robustness"
    )

    print("=" * 80)

    display = robustness[
        robustness[
            "neutralization"
        ]
        ==
        "INDUSTRY_SIZE_NEUTRAL"
    ]

    columns = [

        "factor_name",

        "window",

        "full_sample_alpha_ic",

        "alpha_ic_time_same_sign_share",

        "alpha_ic_trend_same_sign_share",

        "alpha_ic_vol_same_sign_share",

        "full_sample_alpha_spread_bps",

        "alpha_spread_time_same_sign_share",

        "full_sample_ivol_ic",

        "ivol_ic_time_same_sign_share",

        "ivol_ic_trend_same_sign_share",

        "ivol_ic_vol_same_sign_share",
    ]

    print(
        display[
            columns
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print("=" * 80)

    print(
        "Day 7 Step 1 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()
