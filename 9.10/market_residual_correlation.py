from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


# ============================================================
# 0. Paths
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day2"
    / "06_step6_market_residual"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Day 1
# ------------------------------------------------------------

DAY1_STAGE1_DIR = Path(
    r"D:\M1_StockNetwork\output\M1_day1\01_stage1_final"
)

STOCK_MASTER_PATH = (
    DAY1_STAGE1_DIR
    / "stock_master.parquet"
)

TRADE_CALENDAR_PATH = (
    DAY1_STAGE1_DIR
    / "trade_calendar_used.csv"
)


DAY1_STAGE3_DIR = Path(
    r"D:\M1_StockNetwork\output\M1_day1\03_stage3_return_validation"
)

RETURN_PANEL_PATH = (
    DAY1_STAGE3_DIR
    / "daily_return_panel.parquet"
)


# ------------------------------------------------------------
# Day 2 Step 1
# ------------------------------------------------------------

STEP1_DIR_CANDIDATES = [

    OUTPUT_ROOT
    / "M1_day2"
    / "01_step1_baseline_config",

    Path(
        r"D:\M1_StockNetwork\output\M1_day2\01_step1_baseline_config"
    ),
]


def resolve_step1_dir():

    for path in STEP1_DIR_CANDIDATES:

        if (
            path.exists()
            and
            (
                path
                / "day2_analysis_config.json"
            ).exists()
        ):

            return path

    raise FileNotFoundError(
        "没有找到 Day 2 Step 1 配置目录。"
    )


STEP1_DIR = resolve_step1_dir()

CONFIG_PATH = (
    STEP1_DIR
    / "day2_analysis_config.json"
)

ANALYSIS_DATES_PATH = (
    STEP1_DIR
    / "analysis_dates_by_window.csv"
)

REPRESENTATIVE_DATES_PATH = (
    STEP1_DIR
    / "representative_year_end_dates.csv"
)


# ============================================================
# 1. Step 6 Parameters
# ============================================================

MAIN_INDUSTRY_LEVEL = (
    "industry_id1"
)

INDUSTRY_LEVELS = [
    "industry_id1",
    "industry_id2",
]

PAIR_SAMPLE_SIZE = 20_000

PAIR_BATCH_SIZE = 2_000

PARQUET_BATCH_SIZE = 300_000

RANDOM_SEED = 20260910

TOP_TAIL_FRACTIONS = [
    0.05,
    0.01,
]

SAVE_REPRESENTATIVE_PAIRS = True


# ============================================================
# 2. Config
# ============================================================

def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    rolling = config[
        "rolling_design"
    ]

    return_info = config[
        "return"
    ]

    windows = [
        int(x)
        for x in rolling[
            "windows"
        ]
    ]

    stock_valid_ratio = float(
        rolling[
            "baseline_valid_ratio"
        ]
    )

    pair_valid_ratio = float(
        rolling[
            "pairwise_min_common_ratio"
        ]
    )

    return_column = (
        return_info[
            "baseline_column"
        ]
    )

    return (
        config,
        windows,
        stock_valid_ratio,
        pair_valid_ratio,
        return_column,
    )


# ============================================================
# 3. Basic Data
# ============================================================

def load_trade_dates():

    df = pd.read_csv(
        TRADE_CALENDAR_PATH
    )

    if "trade_date" not in df.columns:

        raise ValueError(
            "trade_calendar_used.csv 缺少 trade_date。"
        )

    dates = pd.to_datetime(
        df["trade_date"],
        errors="coerce",
    )

    dates = (
        dates
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    return pd.DatetimeIndex(
        dates
    )


def load_stock_master():

    master = pd.read_parquet(
        STOCK_MASTER_PATH
    )

    required = [
        "security_id",
        "stock_code",
        "stock_name",
        "exchange",
        "list_date",
        "delist_date",
    ]

    missing = [
        x
        for x in required
        if x not in master.columns
    ]

    if missing:

        raise ValueError(
            f"stock_master 缺少字段：{missing}"
        )

    master = master.copy()

    master["security_id"] = (
        master["security_id"]
        .astype("string")
        .str.strip()
    )

    master["stock_code"] = (
        master["stock_code"]
        .astype("string")
    )

    master["stock_name"] = (
        master["stock_name"]
        .astype("string")
    )

    master["exchange"] = (
        master["exchange"]
        .astype("string")
    )

    master["list_date"] = pd.to_datetime(
        master["list_date"],
        errors="coerce",
    )

    master["delist_date"] = pd.to_datetime(
        master["delist_date"],
        errors="coerce",
    )

    if (
        master["security_id"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "stock_master 中 security_id 重复。"
        )

    return (
        master
        .sort_values(
            "security_id"
        )
        .reset_index(
            drop=True
        )
    )


def load_analysis_dates():

    df = pd.read_csv(
        ANALYSIS_DATES_PATH
    )

    df["analysis_date"] = pd.to_datetime(
        df["analysis_date"]
    )

    df["window"] = (
        df["window"]
        .astype(int)
    )

    if "window_start" in df.columns:

        df["window_start"] = pd.to_datetime(
            df["window_start"]
        )

    return df


def load_representative_dates():

    if not REPRESENTATIVE_DATES_PATH.exists():

        return set()

    df = pd.read_csv(
        REPRESENTATIVE_DATES_PATH
    )

    if df.empty:

        return set()

    df["analysis_date"] = pd.to_datetime(
        df["analysis_date"]
    )

    df["window"] = (
        df["window"]
        .astype(int)
    )

    return {
        (
            int(row.window),
            pd.Timestamp(
                row.analysis_date
            ),
        )
        for row
        in df.itertuples(index=False)
    }


# ============================================================
# 4. Return Matrix + Month-End Industry Snapshots
# ============================================================

def build_return_matrix_and_industry_snapshots(
    trade_dates,
    master,
    analysis_dates,
    return_column,
):

    print()
    print("=" * 76)
    print("构造 Return Matrix + Industry Snapshots")
    print("=" * 76)

    security_ids = (
        master["security_id"]
        .astype("string")
        .tolist()
    )

    T = len(
        trade_dates
    )

    P = len(
        security_ids
    )

    print(
        f"T = {T:,}"
    )

    print(
        f"P = {P:,}"
    )

    returns = np.full(
        (
            T,
            P,
        ),
        np.nan,
        dtype=np.float32,
    )

    date_to_idx = {
        pd.Timestamp(date): i
        for i, date
        in enumerate(trade_dates)
    }

    security_to_idx = {
        sid: i
        for i, sid
        in enumerate(security_ids)
    }

    analysis_date_set = set(
        pd.to_datetime(
            analysis_dates[
                "analysis_date"
            ]
        )
        .unique()
    )

    snapshot_parts = []

    parquet_file = pq.ParquetFile(
        RETURN_PANEL_PATH
    )

    required_columns = [
        "trade_date",
        "security_id",
        return_column,
        "industry_id1",
        "industry_id2",
    ]

    missing = [
        x
        for x in required_columns
        if x not in (
            parquet_file
            .schema_arrow
            .names
        )
    ]

    if missing:

        raise ValueError(
            f"Return Panel 缺少字段：{missing}"
        )

    total_rows = 0

    for batch_no, batch in enumerate(

        parquet_file.iter_batches(
            batch_size=
                PARQUET_BATCH_SIZE,

            columns=
                required_columns,
        ),

        start=1,
    ):

        df = batch.to_pandas()

        df["trade_date"] = pd.to_datetime(
            df["trade_date"],
            errors="coerce",
        )

        df["security_id"] = (
            df["security_id"]
            .astype("string")
            .str.strip()
        )

        row_idx = (
            df["trade_date"]
            .map(
                date_to_idx
            )
        )

        col_idx = (
            df["security_id"]
            .map(
                security_to_idx
            )
        )

        ret = pd.to_numeric(
            df[
                return_column
            ],
            errors="coerce",
        )

        mapped = (
            row_idx.notna()
            &
            col_idx.notna()
        )

        if mapped.any():

            r = (
                row_idx.loc[
                    mapped
                ]
                .astype(np.int32)
                .to_numpy()
            )

            c = (
                col_idx.loc[
                    mapped
                ]
                .astype(np.int32)
                .to_numpy()
            )

            values = (
                ret.loc[
                    mapped
                ]
                .to_numpy(
                    dtype=float
                )
            )

            finite = np.isfinite(
                values
            )

            returns[
                r[finite],
                c[finite],
            ] = (
                values[
                    finite
                ]
                .astype(
                    np.float32
                )
            )

        # ----------------------------------------------------
        # Month-end industry snapshots
        # ----------------------------------------------------

        snapshot_mask = (
            mapped
            &
            df["trade_date"].isin(
                analysis_date_set
            )
        )

        if snapshot_mask.any():

            temp = df.loc[
                snapshot_mask,
                [
                    "trade_date",
                    "security_id",
                    "industry_id1",
                    "industry_id2",
                ]
            ].copy()

            temp[
                "master_index"
            ] = (
                col_idx.loc[
                    snapshot_mask
                ]
                .astype(np.int32)
                .to_numpy()
            )

            snapshot_parts.append(
                temp
            )

        total_rows += len(
            df
        )

        if (
            batch_no == 1
            or
            batch_no % 10 == 0
        ):

            print(
                f"Batch {batch_no:4d}: "
                f"{total_rows:,} rows"
            )

    snapshot_df = pd.concat(
        snapshot_parts,
        ignore_index=True,
    )

    snapshot_df[
        "trade_date"
    ] = pd.to_datetime(
        snapshot_df[
            "trade_date"
        ]
    )

    # --------------------------------------------------------
    # Encode industry labels
    # --------------------------------------------------------

    industry_snapshots = {
        level: {}
        for level
        in INDUSTRY_LEVELS
    }

    codebooks = {}

    for level in INDUSTRY_LEVELS:

        values = (
            snapshot_df[level]
            .astype("string")
        )

        codes, uniques = pd.factorize(
            values,
            sort=True,
        )

        snapshot_df[
            f"{level}_code"
        ] = codes.astype(
            np.int32
        )

        codebooks[level] = {
            int(i): str(value)
            for i, value
            in enumerate(uniques)
        }

    for date, g in (
        snapshot_df.groupby(
            "trade_date",
            sort=False,
        )
    ):

        idx = (
            g["master_index"]
            .to_numpy(
                dtype=np.int32
            )
        )

        for level in INDUSTRY_LEVELS:

            arr = np.full(
                P,
                -1,
                dtype=np.int32,
            )

            arr[idx] = (
                g[
                    f"{level}_code"
                ]
                .to_numpy(
                    dtype=np.int32
                )
            )

            industry_snapshots[
                level
            ][
                pd.Timestamp(date)
            ] = arr

    print()
    print(
        "Industry snapshot dates: "
        f"{snapshot_df['trade_date'].nunique():,}"
    )

    return (
        returns,
        industry_snapshots,
        codebooks,
    )


# ============================================================
# 5. PIT Universe
# ============================================================

def current_pit_mask(
    master,
    date,
):

    date = pd.Timestamp(
        date
    )

    return (
        master[
            "list_date"
        ].le(date)
        &
        (
            master[
                "delist_date"
            ].isna()
            |
            master[
                "delist_date"
            ].gt(date)
        )
    ).to_numpy(
        dtype=bool
    )


# ============================================================
# 6. Daily PIT EW Market Factor
# ============================================================

def build_daily_pit_ew_factor(
    trade_dates,
    master,
    returns,
):

    print()
    print("=" * 76)
    print("构造 Daily PIT Equal-Weighted Market Factor")
    print("=" * 76)

    T, P = returns.shape

    market_sum = np.full(
        T,
        np.nan,
        dtype=np.float64,
    )

    market_count = np.zeros(
        T,
        dtype=np.int32,
    )

    market_factor = np.full(
        T,
        np.nan,
        dtype=np.float64,
    )

    for t, date in enumerate(
        trade_dates
    ):

        pit = current_pit_mask(
            master,
            date,
        )

        r = (
            returns[t]
            .astype(
                np.float64,
                copy=False,
            )
        )

        valid = (
            pit
            &
            np.isfinite(r)
        )

        n = int(
            valid.sum()
        )

        market_count[t] = n

        if n > 0:

            total = float(
                r[
                    valid
                ].sum()
            )

            market_sum[t] = total

            market_factor[t] = (
                total / n
            )

    factor_df = pd.DataFrame(
        {
            "trade_date":
                trade_dates,

            "pit_ew_market_return":
                market_factor,

            "pit_valid_stock_count":
                market_count,
        }
    )

    factor_df.to_csv(
        OUTPUT_DIR
        / "daily_pit_ew_market_factor.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return (
        market_sum,
        market_count,
        market_factor,
    )


# ============================================================
# 7. Leave-One-Out Factor Matrix
# ============================================================

def build_leave_one_out_factor(
    X,
    market_sum_window,
    market_count_window,
):

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    finite = np.isfinite(
        X
    )

    own_return = np.where(
        finite,
        X,
        0.0,
    )

    own_count = (
        finite.astype(
            np.int32
        )
    )

    numerator = (
        market_sum_window[
            :,
            None
        ]
        -
        own_return
    )

    denominator = (
        market_count_window[
            :,
            None
        ]
        -
        own_count
    )

    F = np.full(
        X.shape,
        np.nan,
        dtype=np.float64,
    )

    usable = (
        denominator > 0
    )

    F[
        usable
    ] = (
        numerator[
            usable
        ]
        /
        denominator[
            usable
        ]
    )

    return F


# ============================================================
# 8. Stock-Level Market Regression + Residuals
#
# r_i = alpha_i + beta_i F_{-i} + epsilon_i
# ============================================================

def residualize_market_factor(
    X,
    F,
    required_obs,
):

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    F = np.asarray(
        F,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(X)
        &
        np.isfinite(F)
    )

    n = (
        valid.sum(
            axis=0
        )
        .astype(
            np.int32
        )
    )

    x0 = np.where(
        valid,
        X,
        0.0,
    )

    f0 = np.where(
        valid,
        F,
        0.0,
    )

    sum_x = (
        x0.sum(
            axis=0
        )
    )

    sum_f = (
        f0.sum(
            axis=0
        )
    )

    sum_x2 = (
        (
            x0 * x0
        )
        .sum(
            axis=0
        )
    )

    sum_f2 = (
        (
            f0 * f0
        )
        .sum(
            axis=0
        )
    )

    sum_fx = (
        (
            f0 * x0
        )
        .sum(
            axis=0
        )
    )

    n_float = (
        n.astype(
            np.float64
        )
    )

    safe_n = np.where(
        n > 0,
        n_float,
        np.nan,
    )

    mean_x = (
        sum_x
        /
        safe_n
    )

    mean_f = (
        sum_f
        /
        safe_n
    )

    ss_x = (
        sum_x2
        -
        sum_x
        *
        sum_x
        /
        safe_n
    )

    ss_f = (
        sum_f2
        -
        sum_f
        *
        sum_f
        /
        safe_n
    )

    cov_fx = (
        sum_fx
        -
        sum_f
        *
        sum_x
        /
        safe_n
    )

    usable = (
        (n >= required_obs)
        &
        np.isfinite(ss_x)
        &
        np.isfinite(ss_f)
        &
        (ss_x > 0)
        &
        (ss_f > 0)
    )

    alpha = np.full(
        X.shape[1],
        np.nan,
        dtype=float,
    )

    beta = np.full(
        X.shape[1],
        np.nan,
        dtype=float,
    )

    r2 = np.full(
        X.shape[1],
        np.nan,
        dtype=float,
    )

    beta[
        usable
    ] = (
        cov_fx[
            usable
        ]
        /
        ss_f[
            usable
        ]
    )

    alpha[
        usable
    ] = (
        mean_x[
            usable
        ]
        -
        beta[
            usable
        ]
        *
        mean_f[
            usable
        ]
    )

    r2[
        usable
    ] = (
        cov_fx[
            usable
        ] ** 2
        /
        (
            ss_x[
                usable
            ]
            *
            ss_f[
                usable
            ]
        )
    )

    r2 = np.clip(
        r2,
        0.0,
        1.0,
    )

    prediction = (
        alpha[
            None,
            :
        ]
        +
        beta[
            None,
            :
        ]
        *
        F
    )

    residual = np.where(
        valid
        &
        usable[
            None,
            :
        ],
        X - prediction,
        np.nan,
    )

    return (
        residual,
        alpha,
        beta,
        r2,
        n,
    )


# ============================================================
# 9. Correct Uniform Pair Sampling
#
# 在所有 C(p,2) unordered pairs 的 rank 中
# 直接无放回均匀抽样。
# ============================================================

def sample_uniform_pairs(
    n_stocks,
    sample_size,
    rng,
):

    total_pairs = (
        n_stocks
        *
        (
            n_stocks - 1
        )
        //
        2
    )

    target = min(
        sample_size,
        total_pairs,
    )

    if target <= 0:

        return np.empty(
            (
                0,
                2,
            ),
            dtype=np.int32,
        )

    pair_ranks = rng.choice(
        total_pairs,
        size=target,
        replace=False,
    )

    # --------------------------------------------------------
    # cumulative pair count before first index i:
    #
    # C_i =
    # i(2n-i-1)/2
    # --------------------------------------------------------

    i_grid = np.arange(
        n_stocks - 1,
        dtype=np.int64,
    )

    cumulative = (
        i_grid
        *
        (
            2 * n_stocks
            -
            i_grid
            -
            1
        )
        //
        2
    )

    i = (
        np.searchsorted(
            cumulative,
            pair_ranks,
            side="right",
        )
        -
        1
    )

    j = (
        i
        +
        1
        +
        (
            pair_ranks
            -
            cumulative[i]
        )
    )

    pairs = np.column_stack(
        [
            i,
            j,
        ]
    ).astype(
        np.int32
    )

    return pairs


# ============================================================
# 10. Pairwise Pearson
# ============================================================

def calculate_pairwise_correlations(
    matrix,
    stock_a,
    stock_b,
    required_common_days,
):

    n_pairs = len(
        stock_a
    )

    corr = np.full(
        n_pairs,
        np.nan,
        dtype=float,
    )

    common_obs = np.zeros(
        n_pairs,
        dtype=np.int16,
    )

    for start in range(
        0,
        n_pairs,
        PAIR_BATCH_SIZE,
    ):

        end = min(
            start
            +
            PAIR_BATCH_SIZE,
            n_pairs,
        )

        a = stock_a[
            start:end
        ]

        b = stock_b[
            start:end
        ]

        x = (
            matrix[
                :,
                a
            ]
            .astype(
                np.float64,
                copy=False,
            )
        )

        y = (
            matrix[
                :,
                b
            ]
            .astype(
                np.float64,
                copy=False,
            )
        )

        valid = (
            np.isfinite(x)
            &
            np.isfinite(y)
        )

        n = (
            valid.sum(
                axis=0
            )
            .astype(
                np.int32
            )
        )

        common_obs[
            start:end
        ] = (
            n.astype(
                np.int16
            )
        )

        x0 = np.where(
            valid,
            x,
            0.0,
        )

        y0 = np.where(
            valid,
            y,
            0.0,
        )

        sum_x = (
            x0.sum(
                axis=0
            )
        )

        sum_y = (
            y0.sum(
                axis=0
            )
        )

        sum_x2 = (
            (
                x0 * x0
            )
            .sum(
                axis=0
            )
        )

        sum_y2 = (
            (
                y0 * y0
            )
            .sum(
                axis=0
            )
        )

        sum_xy = (
            (
                x0 * y0
            )
            .sum(
                axis=0
            )
        )

        n_float = (
            n.astype(
                np.float64
            )
        )

        safe_n = np.where(
            n > 0,
            n_float,
            np.nan,
        )

        cov_num = (
            sum_xy
            -
            sum_x
            *
            sum_y
            /
            safe_n
        )

        var_x = (
            sum_x2
            -
            sum_x
            *
            sum_x
            /
            safe_n
        )

        var_y = (
            sum_y2
            -
            sum_y
            *
            sum_y
            /
            safe_n
        )

        denominator = np.sqrt(
            np.maximum(
                var_x,
                0.0,
            )
            *
            np.maximum(
                var_y,
                0.0,
            )
        )

        usable = (
            (n >= required_common_days)
            &
            np.isfinite(
                denominator
            )
            &
            (denominator > 0)
        )

        current = np.full(
            end - start,
            np.nan,
            dtype=float,
        )

        current[
            usable
        ] = (
            cov_num[
                usable
            ]
            /
            denominator[
                usable
            ]
        )

        corr[
            start:end
        ] = np.clip(
            current,
            -1.0,
            1.0,
        )

    return (
        corr,
        common_obs,
    )


# ============================================================
# 11. Helpers
# ============================================================

def safe_corr(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    if valid.sum() < 3:

        return np.nan

    xv = x[
        valid
    ]

    yv = y[
        valid
    ]

    if (
        np.std(xv) <= 0
        or
        np.std(yv) <= 0
    ):

        return np.nan

    return float(
        np.corrcoef(
            xv,
            yv,
        )[0, 1]
    )


def summarize_corr_group(
    values,
    prefix,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:

        return {
            f"{prefix}_pair_count":
                0
        }

    q = np.quantile(
        values,
        [
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ],
    )

    return {
        f"{prefix}_pair_count":
            int(
                len(values)
            ),

        f"{prefix}_mean_corr":
            float(
                np.mean(values)
            ),

        f"{prefix}_median_corr":
            float(
                np.median(values)
            ),

        f"{prefix}_std_corr":
            float(
                np.std(
                    values,
                    ddof=1,
                )
            ),

        f"{prefix}_mean_abs_corr":
            float(
                np.mean(
                    np.abs(values)
                )
            ),

        f"{prefix}_q10_corr":
            float(q[0]),

        f"{prefix}_q25_corr":
            float(q[1]),

        f"{prefix}_q50_corr":
            float(q[2]),

        f"{prefix}_q75_corr":
            float(q[3]),

        f"{prefix}_q90_corr":
            float(q[4]),

        f"{prefix}_q95_corr":
            float(q[5]),

        f"{prefix}_q99_corr":
            float(q[6]),

        f"{prefix}_share_negative":
            float(
                (
                    values < 0
                ).mean()
            ),

        f"{prefix}_share_gt_0_1":
            float(
                (
                    values > 0.10
                ).mean()
            ),

        f"{prefix}_share_gt_0_2":
            float(
                (
                    values > 0.20
                ).mean()
            ),

        f"{prefix}_share_gt_0_3":
            float(
                (
                    values > 0.30
                ).mean()
            ),

        f"{prefix}_share_gt_0_5":
            float(
                (
                    values > 0.50
                ).mean()
            ),
    }


def exact_industry_pair_structure(
    industry_codes,
):

    industry_codes = np.asarray(
        industry_codes,
        dtype=np.int32,
    )

    valid = (
        industry_codes >= 0
    )

    codes = industry_codes[
        valid
    ]

    n = len(
        codes
    )

    if n < 2:

        return {
            "industry_valid_stock_count":
                n,

            "industry_class_count":
                0,

            "exact_same_pair_share":
                np.nan,
        }

    counts = np.bincount(
        codes
    )

    counts = counts[
        counts > 0
    ]

    total_pairs = (
        n
        *
        (
            n - 1
        )
        //
        2
    )

    same_pairs = int(
        np.sum(
            counts
            *
            (
                counts - 1
            )
            //
            2
        )
    )

    return {
        "industry_valid_stock_count":
            int(n),

        "industry_class_count":
            int(
                len(counts)
            ),

        "exact_total_pair_count":
            int(
                total_pairs
            ),

        "exact_same_pair_count":
            same_pairs,

        "exact_cross_pair_count":
            int(
                total_pairs
                -
                same_pairs
            ),

        "exact_same_pair_share":
            float(
                same_pairs
                /
                total_pairs
            ),
    }


# ============================================================
# 12. Same-industry Enrichment
# ============================================================

def calculate_same_enrichment(
    corr,
    same_mask,
    exact_same_share,
    prefix,
):

    corr = np.asarray(
        corr,
        dtype=float,
    )

    same_mask = np.asarray(
        same_mask,
        dtype=bool,
    )

    valid = np.isfinite(
        corr
    )

    corr = corr[
        valid
    ]

    same = same_mask[
        valid
    ]

    output = {}

    if (
        len(corr) == 0
        or
        not np.isfinite(
            exact_same_share
        )
        or
        exact_same_share <= 0
    ):

        return output

    output[
        f"{prefix}_sample_same_share"
    ] = float(
        same.mean()
    )

    for tail_fraction in (
        TOP_TAIL_FRACTIONS
    ):

        threshold = float(
            np.quantile(
                corr,
                1.0
                -
                tail_fraction,
            )
        )

        top = (
            corr >= threshold
        )

        top_same_share = float(
            same[
                top
            ].mean()
        )

        pct = int(
            round(
                tail_fraction
                *
                100
            )
        )

        output[
            f"{prefix}_top{pct}_corr_threshold"
        ] = threshold

        output[
            f"{prefix}_top{pct}_same_share"
        ] = top_same_share

        output[
            f"{prefix}_top{pct}_same_enrichment"
        ] = (
            top_same_share
            /
            exact_same_share
        )

    abs_corr = np.abs(
        corr
    )

    threshold = float(
        np.quantile(
            abs_corr,
            0.99,
        )
    )

    top_abs = (
        abs_corr
        >=
        threshold
    )

    top_abs_same_share = float(
        same[
            top_abs
        ].mean()
    )

    output[
        f"{prefix}_top1_abs_same_share"
    ] = top_abs_same_share

    output[
        f"{prefix}_top1_abs_same_enrichment"
    ] = (
        top_abs_same_share
        /
        exact_same_share
    )

    return output


# ============================================================
# 13. Raw vs Residual Edge-Rank Overlap
# ============================================================

def top_rank_overlap(
    raw_corr,
    residual_corr,
    tail_fraction,
):

    raw_corr = np.asarray(
        raw_corr,
        dtype=float,
    )

    residual_corr = np.asarray(
        residual_corr,
        dtype=float,
    )

    valid = (
        np.isfinite(raw_corr)
        &
        np.isfinite(
            residual_corr
        )
    )

    raw = raw_corr[
        valid
    ]

    residual = residual_corr[
        valid
    ]

    n = len(
        raw
    )

    if n < 2:

        return np.nan

    k = max(
        1,
        int(
            round(
                n
                *
                tail_fraction
            )
        ),
    )

    raw_top = np.argpartition(
        raw,
        -k,
    )[
        -k:
    ]

    residual_top = np.argpartition(
        residual,
        -k,
    )[
        -k:
    ]

    overlap = np.intersect1d(
        raw_top,
        residual_top,
        assume_unique=False,
    )

    return float(
        len(overlap)
        /
        k
    )


# ============================================================
# 14. Main Analysis
# ============================================================

def run_analysis(
    trade_dates,
    master,
    returns,
    industry_snapshots,
    analysis_dates,
    representative_dates,
    windows,
    stock_valid_ratio,
    pair_valid_ratio,
    market_sum,
    market_count,
):

    date_to_idx = {
        pd.Timestamp(date): i
        for i, date
        in enumerate(trade_dates)
    }

    finite_return = np.isfinite(
        returns
    )

    cumulative_valid = np.zeros(
        (
            returns.shape[0] + 1,
            returns.shape[1],
        ),
        dtype=np.int32,
    )

    cumulative_valid[
        1:
    ] = np.cumsum(
        finite_return,
        axis=0,
        dtype=np.int32,
    )

    summary_rows = []

    representative_parts = []

    for window in windows:

        print()
        print("=" * 76)
        print(
            f"Market Residual Correlation: W={window}"
        )
        print("=" * 76)

        required_stock_days = int(
            math.ceil(
                stock_valid_ratio
                *
                window
            )
        )

        required_pair_days = int(
            math.ceil(
                pair_valid_ratio
                *
                window
            )
        )

        dates_w = (
            analysis_dates[
                analysis_dates[
                    "window"
                ].eq(window)
            ]
            .sort_values(
                "analysis_date"
            )
        )

        for counter, row in enumerate(
            dates_w.itertuples(
                index=False
            ),
            start=1,
        ):

            date = pd.Timestamp(
                row.analysis_date
            )

            end_idx = (
                date_to_idx[
                    date
                ]
            )

            start_idx = (
                end_idx
                -
                window
                +
                1
            )

            if start_idx < 0:

                continue

            window_start = (
                trade_dates[
                    start_idx
                ]
            )

            # =================================================
            # Eligible universe
            # =================================================

            pit_mask = current_pit_mask(
                master,
                date,
            )

            valid_count = (
                cumulative_valid[
                    end_idx + 1
                ]
                -
                cumulative_valid[
                    start_idx
                ]
            )

            eligible_mask = (
                pit_mask
                &
                (
                    valid_count
                    >=
                    required_stock_days
                )
            )

            eligible_idx = np.flatnonzero(
                eligible_mask
            )

            p = len(
                eligible_idx
            )

            if p < 2:

                continue

            X = (
                returns[
                    start_idx:
                    end_idx + 1,
                    :
                ][
                    :,
                    eligible_idx
                ]
                .astype(
                    np.float64
                )
            )

            # =================================================
            # Leave-one-out daily PIT market factor
            # =================================================

            F_loo = (
                build_leave_one_out_factor(
                    X=
                        X,

                    market_sum_window=
                        market_sum[
                            start_idx:
                            end_idx + 1
                        ],

                    market_count_window=
                        market_count[
                            start_idx:
                            end_idx + 1
                        ],
                )
            )

            # =================================================
            # Residualization
            # =================================================

            (
                residual,
                alpha,
                beta,
                r2,
                regression_n,
            ) = (
                residualize_market_factor(
                    X=
                        X,

                    F=
                        F_loo,

                    required_obs=
                        required_stock_days,
                )
            )

            regression_valid = (
                np.isfinite(beta)
                &
                np.isfinite(r2)
            )

            beta_valid = beta[
                regression_valid
            ]

            r2_valid = r2[
                regression_valid
            ]

            # =================================================
            # Correct uniform pair sample
            # =================================================

            seed = (
                RANDOM_SEED
                +
                window
                *
                10_000_000
                +
                int(
                    date.strftime(
                        "%Y%m%d"
                    )
                )
            ) % (
                2**32 - 1
            )

            rng = np.random.default_rng(
                seed
            )

            local_pairs = (
                sample_uniform_pairs(
                    n_stocks=p,
                    sample_size=
                        PAIR_SAMPLE_SIZE,
                    rng=rng,
                )
            )

            local_a = (
                local_pairs[
                    :,
                    0
                ]
            )

            local_b = (
                local_pairs[
                    :,
                    1
                ]
            )

            # =================================================
            # Raw and residual correlations
            # =================================================

            (
                raw_corr,
                raw_common_obs,
            ) = (
                calculate_pairwise_correlations(
                    matrix=X,

                    stock_a=
                        local_a,

                    stock_b=
                        local_b,

                    required_common_days=
                        required_pair_days,
                )
            )

            (
                residual_corr,
                residual_common_obs,
            ) = (
                calculate_pairwise_correlations(
                    matrix=
                        residual,

                    stock_a=
                        local_a,

                    stock_b=
                        local_b,

                    required_common_days=
                        required_pair_days,
                )
            )

            raw_all_summary = (
                summarize_corr_group(
                    raw_corr,
                    "raw_all",
                )
            )

            residual_all_summary = (
                summarize_corr_group(
                    residual_corr,
                    "residual_all",
                )
            )

            common_pair = (
                np.isfinite(
                    raw_corr
                )
                &
                np.isfinite(
                    residual_corr
                )
            )

            raw_residual_corr = (
                safe_corr(
                    raw_corr,
                    residual_corr,
                )
            )

            sign_change_share = np.nan

            if common_pair.any():

                sign_change_share = float(
                    (
                        np.sign(
                            raw_corr[
                                common_pair
                            ]
                        )
                        !=
                        np.sign(
                            residual_corr[
                                common_pair
                            ]
                        )
                    ).mean()
                )

            top5_overlap = (
                top_rank_overlap(
                    raw_corr,
                    residual_corr,
                    0.05,
                )
            )

            top1_overlap = (
                top_rank_overlap(
                    raw_corr,
                    residual_corr,
                    0.01,
                )
            )

            # =================================================
            # Representative pair file
            # =================================================

            save_rep = (
                SAVE_REPRESENTATIVE_PAIRS
                and
                (
                    window,
                    date,
                )
                in
                representative_dates
            )

            representative_base = None

            if save_rep:

                global_a = (
                    eligible_idx[
                        local_a
                    ]
                )

                global_b = (
                    eligible_idx[
                        local_b
                    ]
                )

                representative_base = (
                    pd.DataFrame(
                        {
                            "analysis_date":
                                date,

                            "window":
                                window,

                            "security_id_i":
                                master.iloc[
                                    global_a
                                ][
                                    "security_id"
                                ].to_numpy(),

                            "stock_code_i":
                                master.iloc[
                                    global_a
                                ][
                                    "stock_code"
                                ].to_numpy(),

                            "stock_name_i":
                                master.iloc[
                                    global_a
                                ][
                                    "stock_name"
                                ].to_numpy(),

                            "security_id_j":
                                master.iloc[
                                    global_b
                                ][
                                    "security_id"
                                ].to_numpy(),

                            "stock_code_j":
                                master.iloc[
                                    global_b
                                ][
                                    "stock_code"
                                ].to_numpy(),

                            "stock_name_j":
                                master.iloc[
                                    global_b
                                ][
                                    "stock_name"
                                ].to_numpy(),

                            "raw_common_obs":
                                raw_common_obs,

                            "residual_common_obs":
                                residual_common_obs,

                            "raw_corr":
                                raw_corr,

                            "residual_corr":
                                residual_corr,

                            "corr_change":
                                (
                                    residual_corr
                                    -
                                    raw_corr
                                ),
                        }
                    )
                )

            # =================================================
            # Industry levels
            # =================================================

            for industry_level in (
                INDUSTRY_LEVELS
            ):

                if (
                    date
                    not in
                    industry_snapshots[
                        industry_level
                    ]
                ):

                    raise ValueError(
                        f"{industry_level} "
                        f"缺少 {date.date()} "
                        "行业快照。"
                    )

                industry_all = (
                    industry_snapshots[
                        industry_level
                    ][
                        date
                    ]
                )

                industry_eligible = (
                    industry_all[
                        eligible_idx
                    ]
                )

                exact_structure = (
                    exact_industry_pair_structure(
                        industry_eligible
                    )
                )

                industry_a = (
                    industry_eligible[
                        local_a
                    ]
                )

                industry_b = (
                    industry_eligible[
                        local_b
                    ]
                )

                valid_industry = (
                    (industry_a >= 0)
                    &
                    (industry_b >= 0)
                )

                same_mask = (
                    valid_industry
                    &
                    (
                        industry_a
                        ==
                        industry_b
                    )
                )

                cross_mask = (
                    valid_industry
                    &
                    (
                        industry_a
                        !=
                        industry_b
                    )
                )

                # ---------------------------------------------
                # Partition QA
                # ---------------------------------------------

                if np.any(
                    same_mask
                    &
                    cross_mask
                ):

                    raise RuntimeError(
                        "Same/Cross masks overlap."
                    )

                if (
                    (
                        same_mask
                        |
                        cross_mask
                    ).sum()
                    !=
                    valid_industry.sum()
                ):

                    raise RuntimeError(
                        "Same/Cross partition error."
                    )

                # ---------------------------------------------
                # Raw
                # ---------------------------------------------

                raw_same = raw_corr[
                    same_mask
                    &
                    np.isfinite(
                        raw_corr
                    )
                ]

                raw_cross = raw_corr[
                    cross_mask
                    &
                    np.isfinite(
                        raw_corr
                    )
                ]

                raw_same_summary = (
                    summarize_corr_group(
                        raw_same,
                        "raw_same",
                    )
                )

                raw_cross_summary = (
                    summarize_corr_group(
                        raw_cross,
                        "raw_cross",
                    )
                )

                # ---------------------------------------------
                # Residual
                # ---------------------------------------------

                residual_same = residual_corr[
                    same_mask
                    &
                    np.isfinite(
                        residual_corr
                    )
                ]

                residual_cross = residual_corr[
                    cross_mask
                    &
                    np.isfinite(
                        residual_corr
                    )
                ]

                residual_same_summary = (
                    summarize_corr_group(
                        residual_same,
                        "residual_same",
                    )
                )

                residual_cross_summary = (
                    summarize_corr_group(
                        residual_cross,
                        "residual_cross",
                    )
                )

                # ---------------------------------------------
                # Enrichment
                # ---------------------------------------------

                exact_same_share = (
                    exact_structure[
                        "exact_same_pair_share"
                    ]
                )

                raw_enrichment = (
                    calculate_same_enrichment(
                        corr=
                            raw_corr[
                                valid_industry
                            ],

                        same_mask=
                            same_mask[
                                valid_industry
                            ],

                        exact_same_share=
                            exact_same_share,

                        prefix=
                            "raw",
                    )
                )

                residual_enrichment = (
                    calculate_same_enrichment(
                        corr=
                            residual_corr[
                                valid_industry
                            ],

                        same_mask=
                            same_mask[
                                valid_industry
                            ],

                        exact_same_share=
                            exact_same_share,

                        prefix=
                            "residual",
                    )
                )

                # ---------------------------------------------
                # Raw / Residual gaps
                # ---------------------------------------------

                raw_same_mean = (
                    raw_same_summary.get(
                        "raw_same_mean_corr",
                        np.nan,
                    )
                )

                raw_cross_mean = (
                    raw_cross_summary.get(
                        "raw_cross_mean_corr",
                        np.nan,
                    )
                )

                residual_same_mean = (
                    residual_same_summary.get(
                        "residual_same_mean_corr",
                        np.nan,
                    )
                )

                residual_cross_mean = (
                    residual_cross_summary.get(
                        "residual_cross_mean_corr",
                        np.nan,
                    )
                )

                raw_gap = (
                    raw_same_mean
                    -
                    raw_cross_mean
                )

                residual_gap = (
                    residual_same_mean
                    -
                    residual_cross_mean
                )

                gap_retention = np.nan

                if (
                    np.isfinite(
                        raw_gap
                    )
                    and
                    abs(raw_gap)
                    >
                    1e-12
                ):

                    gap_retention = (
                        residual_gap
                        /
                        raw_gap
                    )

                raw_all_mean = (
                    raw_all_summary.get(
                        "raw_all_mean_corr",
                        np.nan,
                    )
                )

                residual_all_mean = (
                    residual_all_summary.get(
                        "residual_all_mean_corr",
                        np.nan,
                    )
                )

                mean_corr_removed = (
                    raw_all_mean
                    -
                    residual_all_mean
                )

                reduction_fraction = np.nan

                if (
                    np.isfinite(
                        raw_all_mean
                    )
                    and
                    abs(raw_all_mean)
                    >
                    1e-12
                ):

                    reduction_fraction = (
                        mean_corr_removed
                        /
                        raw_all_mean
                    )

                summary_rows.append(
                    {
                        "analysis_date":
                            date,

                        "window":
                            window,

                        "window_start":
                            window_start,

                        "industry_level":
                            industry_level,

                        "is_main_industry_level":
                            (
                                industry_level
                                ==
                                MAIN_INDUSTRY_LEVEL
                            ),

                        "stock_valid_ratio":
                            stock_valid_ratio,

                        "pair_valid_ratio":
                            pair_valid_ratio,

                        "required_stock_days":
                            required_stock_days,

                        "required_pair_days":
                            required_pair_days,

                        "current_pit_count":
                            int(
                                pit_mask.sum()
                            ),

                        "eligible_stock_count":
                            p,

                        "market_regression_stock_count":
                            int(
                                regression_valid.sum()
                            ),

                        "market_beta_mean":
                            float(
                                np.mean(
                                    beta_valid
                                )
                            ),

                        "market_beta_median":
                            float(
                                np.median(
                                    beta_valid
                                )
                            ),

                        "market_r2_mean":
                            float(
                                np.mean(
                                    r2_valid
                                )
                            ),

                        "market_r2_median":
                            float(
                                np.median(
                                    r2_valid
                                )
                            ),

                        "sampled_pair_count":
                            int(
                                len(
                                    local_pairs
                                )
                            ),

                        "raw_residual_pair_corr":
                            raw_residual_corr,

                        "raw_residual_sign_change_share":
                            sign_change_share,

                        "raw_residual_top5_overlap":
                            top5_overlap,

                        "raw_residual_top1_overlap":
                            top1_overlap,

                        **raw_all_summary,

                        **residual_all_summary,

                        **exact_structure,

                        **raw_same_summary,

                        **raw_cross_summary,

                        **residual_same_summary,

                        **residual_cross_summary,

                        **raw_enrichment,

                        **residual_enrichment,

                        "raw_same_cross_gap":
                            raw_gap,

                        "residual_same_cross_gap":
                            residual_gap,

                        "industry_gap_retention_ratio":
                            gap_retention,

                        "raw_minus_residual_mean_corr":
                            mean_corr_removed,

                        "overall_corr_reduction_fraction":
                            reduction_fraction,
                    }
                )

                # ---------------------------------------------
                # Representative pairs:
                # save both industry classifications
                # ---------------------------------------------

                if (
                    save_rep
                    and
                    representative_base
                    is not None
                ):

                    representative_base[
                        f"{industry_level}_same"
                    ] = same_mask

                    representative_base[
                        f"{industry_level}_code_i"
                    ] = industry_a

                    representative_base[
                        f"{industry_level}_code_j"
                    ] = industry_b

            if (
                save_rep
                and
                representative_base
                is not None
            ):

                representative_parts.append(
                    representative_base
                )

            if (
                counter == 1
                or
                counter % 12 == 0
                or
                counter
                ==
                len(dates_w)
            ):

                current = [
                    x
                    for x in summary_rows
                    if (
                        x[
                            "analysis_date"
                        ]
                        ==
                        date
                        and
                        x[
                            "window"
                        ]
                        ==
                        window
                        and
                        x[
                            "industry_level"
                        ]
                        ==
                        MAIN_INDUSTRY_LEVEL
                    )
                ][-1]

                print(
                    f"[{counter:3d}/"
                    f"{len(dates_w):3d}] "
                    f"{date.date()} | "
                    f"p={p:,} | "
                    f"Raw={current['raw_all_mean_corr']:.4f} | "
                    f"Residual={current['residual_all_mean_corr']:.4f} | "
                    f"Raw Gap={current['raw_same_cross_gap']:.4f} | "
                    f"Residual Gap={current['residual_same_cross_gap']:.4f}"
                )

    summary_df = pd.DataFrame(
        summary_rows
    )

    if representative_parts:

        representative_df = pd.concat(
            representative_parts,
            ignore_index=True,
        )

    else:

        representative_df = pd.DataFrame()

    return (
        summary_df,
        representative_df,
    )


# ============================================================
# 15. Window Summary
# ============================================================

def build_window_comparison(
    summary_df,
):

    result = (
        summary_df
        .groupby(
            [
                "industry_level",
                "window",
            ],
            as_index=False,
        )
        .agg(

            analysis_date_count=
                (
                    "analysis_date",
                    "count",
                ),

            mean_market_r2=
                (
                    "market_r2_mean",
                    "mean",
                ),

            median_market_r2=
                (
                    "market_r2_median",
                    "median",
                ),

            raw_mean_corr=
                (
                    "raw_all_mean_corr",
                    "mean",
                ),

            residual_mean_corr=
                (
                    "residual_all_mean_corr",
                    "mean",
                ),

            mean_corr_removed=
                (
                    "raw_minus_residual_mean_corr",
                    "mean",
                ),

            corr_reduction_fraction=
                (
                    "overall_corr_reduction_fraction",
                    "mean",
                ),

            raw_same_corr=
                (
                    "raw_same_mean_corr",
                    "mean",
                ),

            raw_cross_corr=
                (
                    "raw_cross_mean_corr",
                    "mean",
                ),

            raw_same_cross_gap=
                (
                    "raw_same_cross_gap",
                    "mean",
                ),

            residual_same_corr=
                (
                    "residual_same_mean_corr",
                    "mean",
                ),

            residual_cross_corr=
                (
                    "residual_cross_mean_corr",
                    "mean",
                ),

            residual_same_cross_gap=
                (
                    "residual_same_cross_gap",
                    "mean",
                ),

            industry_gap_retention=
                (
                    "industry_gap_retention_ratio",
                    "mean",
                ),

            raw_top1_same_enrichment=
                (
                    "raw_top1_same_enrichment",
                    "mean",
                ),

            residual_top1_same_enrichment=
                (
                    "residual_top1_same_enrichment",
                    "mean",
                ),

            raw_top5_same_enrichment=
                (
                    "raw_top5_same_enrichment",
                    "mean",
                ),

            residual_top5_same_enrichment=
                (
                    "residual_top5_same_enrichment",
                    "mean",
                ),

            raw_residual_pair_corr=
                (
                    "raw_residual_pair_corr",
                    "mean",
                ),

            raw_residual_top5_overlap=
                (
                    "raw_residual_top5_overlap",
                    "mean",
                ),

            raw_residual_top1_overlap=
                (
                    "raw_residual_top1_overlap",
                    "mean",
                ),

            sign_change_share=
                (
                    "raw_residual_sign_change_share",
                    "mean",
                ),
        )
    )

    return result


# ============================================================
# 16. Yearly Summary
# ============================================================

def build_yearly_summary(
    summary_df,
):

    df = summary_df.copy()

    df["year"] = (
        pd.to_datetime(
            df[
                "analysis_date"
            ]
        )
        .dt.year
    )

    result = (
        df.groupby(
            [
                "year",
                "window",
                "industry_level",
            ],
            as_index=False,
        )
        .agg(

            analysis_month_count=
                (
                    "analysis_date",
                    "count",
                ),

            market_r2_median=
                (
                    "market_r2_median",
                    "mean",
                ),

            raw_mean_corr=
                (
                    "raw_all_mean_corr",
                    "mean",
                ),

            residual_mean_corr=
                (
                    "residual_all_mean_corr",
                    "mean",
                ),

            raw_same_cross_gap=
                (
                    "raw_same_cross_gap",
                    "mean",
                ),

            residual_same_cross_gap=
                (
                    "residual_same_cross_gap",
                    "mean",
                ),

            gap_retention_ratio=
                (
                    "industry_gap_retention_ratio",
                    "mean",
                ),

            raw_top1_same_enrichment=
                (
                    "raw_top1_same_enrichment",
                    "mean",
                ),

            residual_top1_same_enrichment=
                (
                    "residual_top1_same_enrichment",
                    "mean",
                ),

            raw_residual_top1_overlap=
                (
                    "raw_residual_top1_overlap",
                    "mean",
                ),
        )
    )

    return result


# ============================================================
# 17. Market Mode vs Residual Industry Structure
# ============================================================

def build_relationship_summary(
    summary_df,
):

    rows = []

    for (
        industry_level,
        window,
    ), g in (
        summary_df.groupby(
            [
                "industry_level",
                "window",
            ]
        )
    ):

        rows.append(
            {
                "industry_level":
                    industry_level,

                "window":
                    int(window),

                "month_count":
                    int(
                        len(g)
                    ),

                "corr_market_r2_residual_gap":
                    safe_corr(
                        g[
                            "market_r2_median"
                        ],
                        g[
                            "residual_same_cross_gap"
                        ],
                    ),

                "corr_raw_mean_residual_gap":
                    safe_corr(
                        g[
                            "raw_all_mean_corr"
                        ],
                        g[
                            "residual_same_cross_gap"
                        ],
                    ),

                "corr_market_r2_residual_top1_enrichment":
                    safe_corr(
                        g[
                            "market_r2_median"
                        ],
                        g[
                            "residual_top1_same_enrichment"
                        ],
                    ),

                "corr_raw_gap_residual_gap":
                    safe_corr(
                        g[
                            "raw_same_cross_gap"
                        ],
                        g[
                            "residual_same_cross_gap"
                        ],
                    ),

                "share_residual_gap_positive":
                    float(
                        (
                            g[
                                "residual_same_cross_gap"
                            ]
                            >
                            0
                        ).mean()
                    ),

                "share_residual_top1_enrichment_gt_1":
                    float(
                        (
                            g[
                                "residual_top1_same_enrichment"
                            ]
                            >
                            1
                        ).mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 18. Figures
# ============================================================

def plot_results(
    summary_df,
    windows,
):

    main = (
        summary_df[
            summary_df[
                "industry_level"
            ]
            .eq(
                MAIN_INDUSTRY_LEVEL
            )
        ]
        .copy()
    )

    for window in windows:

        df = (
            main[
                main[
                    "window"
                ]
                .eq(
                    window
                )
            ]
            .sort_values(
                "analysis_date"
            )
        )

        # ----------------------------------------------------
        # Raw vs residual overall correlation
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.plot(
            df["analysis_date"],
            df["raw_all_mean_corr"],
            label="Raw",
        )

        ax.plot(
            df["analysis_date"],
            df["residual_all_mean_corr"],
            label="Market residual",
        )

        ax.axhline(
            0,
            linewidth=1,
        )

        ax.set_title(
            f"Raw vs Market-Residual Mean Correlation - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Mean Pairwise Correlation"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / f"raw_vs_residual_mean_corr_W{window}.png",
            dpi=180,
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Residual Same vs Cross
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.plot(
            df["analysis_date"],
            df["residual_same_mean_corr"],
            label="Residual Same Industry",
        )

        ax.plot(
            df["analysis_date"],
            df["residual_cross_mean_corr"],
            label="Residual Cross Industry",
        )

        ax.axhline(
            0,
            linewidth=1,
        )

        ax.set_title(
            f"Same vs Cross Industry Residual Correlation - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Mean Residual Correlation"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / f"residual_same_cross_mean_corr_W{window}.png",
            dpi=180,
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Raw gap vs residual gap
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.plot(
            df["analysis_date"],
            df["raw_same_cross_gap"],
            label="Raw Industry Gap",
        )

        ax.plot(
            df["analysis_date"],
            df["residual_same_cross_gap"],
            label="Residual Industry Gap",
        )

        ax.axhline(
            0,
            linewidth=1,
        )

        ax.set_title(
            f"Raw vs Residual Industry Gap - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Same - Cross Correlation"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / f"raw_vs_residual_industry_gap_W{window}.png",
            dpi=180,
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Top 1% industry enrichment
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.plot(
            df["analysis_date"],
            df[
                "raw_top1_same_enrichment"
            ],
            label="Raw Top 1%",
        )

        ax.plot(
            df["analysis_date"],
            df[
                "residual_top1_same_enrichment"
            ],
            label="Residual Top 1%",
        )

        ax.axhline(
            1,
            linewidth=1,
        )

        ax.set_title(
            f"Same-Industry Enrichment Before/After Market Removal - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Enrichment Ratio"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / f"raw_vs_residual_top1_enrichment_W{window}.png",
            dpi=180,
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Top 1% edge overlap
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.plot(
            df["analysis_date"],
            df[
                "raw_residual_top1_overlap"
            ],
        )

        ax.set_title(
            f"Top 1% Edge Overlap: Raw vs Market Residual - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Top 1% Edge Overlap"
        )

        ax.set_ylim(
            0,
            1,
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / f"raw_residual_top1_edge_overlap_W{window}.png",
            dpi=180,
        )

        plt.close(fig)


# ============================================================
# 19. Metadata
# ============================================================

def save_metadata(
    windows,
    stock_valid_ratio,
    pair_valid_ratio,
    return_column,
):

    metadata = {

        "step":
            "Day2_Step6_Market_Residual_Correlation",

        "return_column":
            return_column,

        "windows":
            windows,

        "stock_valid_ratio":
            stock_valid_ratio,

        "pair_valid_ratio":
            pair_valid_ratio,

        "main_industry_level":
            MAIN_INDUSTRY_LEVEL,

        "market_factor":
            (
                "Daily equal-weighted market return "
                "constructed from the contemporaneous "
                "PIT universe and valid network returns."
            ),

        "stock_regression_factor":
            (
                "Leave-one-out version of the daily PIT "
                "equal-weighted market factor."
            ),

        "residual_model":
            (
                "r_i,s = alpha_i,t,W + "
                "beta_i,t,W F_{-i,s} + epsilon_i,s"
            ),

        "residual_correlation":
            (
                "Pairwise-complete Pearson correlation "
                "between rolling market-factor residuals."
            ),

        "pair_sampling":
            (
                "Uniform sampling without replacement "
                "from integer ranks of all eligible "
                "unordered pairs."
            ),

        "important_notes": [

            (
                "The pair sampler differs from the earlier "
                "legacy Step 2/5 implementation because the "
                "old np.unique(... )[:target] truncation can "
                "induce index-order bias."
            ),

            (
                "Raw and residual correlations inside Step 6 "
                "are computed on exactly the same corrected "
                "pair sample, so their comparison is valid."
            ),

            (
                "Residual correlation is descriptive evidence "
                "of conditional dependence after removal of "
                "one broad market factor. It is not a causal "
                "measure and is not yet the final sparse network."
            ),

            (
                "The Same/Cross classification uses the "
                "historically indexed industry label at "
                "the network endpoint."
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        / "step6_market_residual_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 20. Main
# ============================================================

def main():

    print("=" * 76)
    print("M1 Day 2 - Step 6")
    print("Market-Residual Correlation Structure")
    print("=" * 76)

    (
        config,
        windows,
        stock_valid_ratio,
        pair_valid_ratio,
        return_column,
    ) = load_config()

    print()
    print("[1] Config")

    print(
        f"Return        : {return_column}"
    )

    print(
        f"Windows       : {windows}"
    )

    print(
        f"Stock c       : {stock_valid_ratio}"
    )

    print(
        f"Pair c        : {pair_valid_ratio}"
    )

    trade_dates = load_trade_dates()

    master = load_stock_master()

    analysis_dates = (
        load_analysis_dates()
    )

    representative_dates = (
        load_representative_dates()
    )

    print()
    print(
        "[2] Return Matrix + Industry Snapshots"
    )

    (
        returns,
        industry_snapshots,
        industry_codebooks,
    ) = (
        build_return_matrix_and_industry_snapshots(
            trade_dates=
                trade_dates,

            master=
                master,

            analysis_dates=
                analysis_dates,

            return_column=
                return_column,
        )
    )

    with open(
        OUTPUT_DIR
        / "industry_codebooks.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            industry_codebooks,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "[3] Daily PIT EW Market Factor"
    )

    (
        market_sum,
        market_count,
        market_factor,
    ) = (
        build_daily_pit_ew_factor(
            trade_dates=
                trade_dates,

            master=
                master,

            returns=
                returns,
        )
    )

    print()
    print(
        "[4] Rolling Market Residual Correlation"
    )

    (
        summary_df,
        representative_df,
    ) = (
        run_analysis(
            trade_dates=
                trade_dates,

            master=
                master,

            returns=
                returns,

            industry_snapshots=
                industry_snapshots,

            analysis_dates=
                analysis_dates,

            representative_dates=
                representative_dates,

            windows=
                windows,

            stock_valid_ratio=
                stock_valid_ratio,

            pair_valid_ratio=
                pair_valid_ratio,

            market_sum=
                market_sum,

            market_count=
                market_count,
        )
    )

    window_df = (
        build_window_comparison(
            summary_df
        )
    )

    yearly_df = (
        build_yearly_summary(
            summary_df
        )
    )

    relationship_df = (
        build_relationship_summary(
            summary_df
        )
    )

    print()
    print("[5] Save Results")

    summary_df.to_csv(
        OUTPUT_DIR
        / "market_residual_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    window_df.to_csv(
        OUTPUT_DIR
        / "market_residual_window_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    yearly_df.to_csv(
        OUTPUT_DIR
        / "market_residual_yearly_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    relationship_df.to_csv(
        OUTPUT_DIR
        / "market_residual_relationship_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not representative_df.empty:

        representative_df.to_parquet(
            OUTPUT_DIR
            / "representative_raw_residual_pairs.parquet",
            index=False,
        )

    print()
    print("[6] Figures")

    plot_results(
        summary_df=
            summary_df,

        windows=
            windows,
    )

    save_metadata(
        windows=
            windows,

        stock_valid_ratio=
            stock_valid_ratio,

        pair_valid_ratio=
            pair_valid_ratio,

        return_column=
            return_column,
    )

    print()
    print("=" * 76)
    print("Day 2 Step 6 完成")
    print("=" * 76)

    show_cols = [
        "industry_level",
        "window",
        "raw_mean_corr",
        "residual_mean_corr",
        "corr_reduction_fraction",
        "raw_same_cross_gap",
        "residual_same_cross_gap",
        "industry_gap_retention",
        "raw_top1_same_enrichment",
        "residual_top1_same_enrichment",
        "raw_residual_top1_overlap",
    ]

    print()
    print(
        window_df[
            show_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()