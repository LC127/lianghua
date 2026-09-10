from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib.pyplot as plt


# ============================================================
# 0. 路径配置
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day2"
    / "05_step5_same_cross_industry"
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


# ------------------------------------------------------------
# Day 2 Step 2
# ------------------------------------------------------------

STEP2_DIR = (
    OUTPUT_ROOT
    / "M1_day2"
    / "02_step2_raw_correlation"
)

RAW_CORRELATION_PATH = (
    STEP2_DIR
    / "raw_correlation_summary.csv"
)


# ------------------------------------------------------------
# Day 2 Step 3
# ------------------------------------------------------------

STEP3_DIR = (
    OUTPUT_ROOT
    / "M1_day2"
    / "03_step3_common_factor"
)

COMMON_FACTOR_PATH = (
    STEP3_DIR
    / "common_factor_summary.csv"
)


# ------------------------------------------------------------
# Day 2 Step 4
# ------------------------------------------------------------

STEP4_DIR = (
    OUTPUT_ROOT
    / "M1_day2"
    / "04_step4_historical_industry_validation"
)

INDUSTRY_CHANGE_EVENTS_PATH = (
    STEP4_DIR
    / "historical_industry_change_events.csv"
)


# ============================================================
# 1. Step 5 参数
# ============================================================

# ------------------------------------------------------------
# 主行业层级 + robustness
# ------------------------------------------------------------

INDUSTRY_LEVELS = [

    "industry_id1",
    "industry_id2",
]


MAIN_INDUSTRY_LEVEL = (
    "industry_id1"
)


# ------------------------------------------------------------
# 与 Step 2 完全一致：
# 每个 Month-End × W 随机抽 20,000 个 pair
#
# 因为使用相同 seed 和 sampling rule，
# 理论上可以复现 Step 2 的 pair sample。
# ------------------------------------------------------------

PAIR_SAMPLE_SIZE = 20_000

PAIR_BATCH_SIZE = 2_000

PARQUET_BATCH_SIZE = 300_000

RANDOM_SEED = 20260910


# ------------------------------------------------------------
# Strong association thresholds
# ------------------------------------------------------------

STRONG_CORRELATION_THRESHOLDS = [
    0.30,
    0.50,
]


# ------------------------------------------------------------
# Top-tail
# ------------------------------------------------------------

TOP_QUANTILES = [
    0.95,
    0.99,
]


# ------------------------------------------------------------
# 如果随机样本中的 Same pair 太少，
# 给出 sample-size warning
# ------------------------------------------------------------

MIN_SAME_PAIR_SAMPLE = 200


# ------------------------------------------------------------
# 分类变更集中日期警告阈值
#
# 仅用于 diagnostic flag，
# 不是统计检验阈值。
# ------------------------------------------------------------

INDUSTRY_REVISION_ALERT_COUNT = 100


# ------------------------------------------------------------
# Representative dates 是否保存 pair-level sample
# ------------------------------------------------------------

SAVE_REPRESENTATIVE_PAIRS = True


# ============================================================
# 2. 读取 Config
# ============================================================

def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    rolling = (
        config["rolling_design"]
    )

    return_info = (
        config["return"]
    )

    windows = [
        int(x)
        for x in rolling["windows"]
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
# 3. Trading Calendar
# ============================================================

def load_trade_dates():

    df = pd.read_csv(
        TRADE_CALENDAR_PATH
    )

    if "trade_date" not in df.columns:

        raise ValueError(
            "trade_calendar_used.csv "
            "缺少 trade_date。"
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


# ============================================================
# 4. Stock Master
# ============================================================

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
            f"stock_master 缺字段：{missing}"
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


# ============================================================
# 5. Analysis Dates
# ============================================================

def load_analysis_dates():

    df = pd.read_csv(
        ANALYSIS_DATES_PATH
    )

    df["analysis_date"] = pd.to_datetime(
        df["analysis_date"]
    )

    df["window_start"] = pd.to_datetime(
        df["window_start"]
    )

    df["window"] = (
        df["window"]
        .astype(int)
    )

    return df


def load_representative_dates():

    if not (
        REPRESENTATIVE_DATES_PATH.exists()
    ):

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

        for row in df.itertuples(
            index=False
        )
    }


# ============================================================
# 6. 构造 Return Matrix
#    + Month-End Industry Snapshots
#
# Return:
#
#   T × P
#
# Industry:
#
#   只保存 Month-End，
#   不保存完整 T × P 字符串矩阵。
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

        pd.Timestamp(date):
            i

        for i, date in enumerate(
            trade_dates
        )
    }

    security_to_idx = {

        sid: i

        for i, sid in enumerate(
            security_ids
        )
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
            f"Return Panel 缺字段：{missing}"
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
                r[
                    finite
                ],
                c[
                    finite
                ],
            ] = (
                values[
                    finite
                ]
                .astype(
                    np.float32
                )
            )

        # ====================================================
        # Month-End Industry Snapshot
        # ====================================================

        is_analysis_date = (
            df["trade_date"]
            .isin(
                analysis_date_set
            )
        )

        snapshot_mask = (
            mapped
            &
            is_analysis_date
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
                .astype(
                    np.int32
                )
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

    # ========================================================
    # Combine Month-End Industry Data
    # ========================================================

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
    # Encode industry labels as integers
    # --------------------------------------------------------

    level_codebooks = {}

    for level in INDUSTRY_LEVELS:

        values = (
            snapshot_df[
                level
            ]
            .astype("string")
        )

        codes, uniques = (
            pd.factorize(
                values,
                sort=True,
            )
        )

        snapshot_df[
            f"{level}_code"
        ] = (
            codes.astype(
                np.int32
            )
        )

        level_codebooks[
            level
        ] = {

            int(i):
                str(value)

            for i, value in enumerate(
                uniques
            )
        }

    # ========================================================
    # Build date -> industry-code array
    # ========================================================

    industry_snapshots = {

        level: {}

        for level in INDUSTRY_LEVELS
    }

    for date, g in (
        snapshot_df.groupby(
            "trade_date",
            sort=False,
        )
    ):

        master_indices = (
            g[
                "master_index"
            ]
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

            arr[
                master_indices
            ] = (
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
        level_codebooks,
    )


# ============================================================
# 7. PIT Mask
# ============================================================

def current_pit_mask(
    master,
    date,
):

    date = pd.Timestamp(
        date
    )

    return (

        master["list_date"]
        .le(date)

        &

        (
            master[
                "delist_date"
            ]
            .isna()

            |

            master[
                "delist_date"
            ]
            .gt(date)
        )
    ).to_numpy(
        dtype=bool
    )


# ============================================================
# 8. 随机抽不重复 Pair
#
# 与 Step 2 使用相同算法。
# ============================================================

def sample_unique_pairs(
    n_stocks,
    sample_size,
    rng,
):

    total_possible = (
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
        total_possible,
    )

    if target <= 0:

        return np.empty(
            (
                0,
                2,
            ),
            dtype=np.int32,
        )

    collected = np.empty(
        (
            0,
            2,
        ),
        dtype=np.int32,
    )

    while (
        len(collected)
        <
        target
    ):

        remaining = (
            target
            -
            len(collected)
        )

        draw_size = max(
            int(
                remaining
                *
                1.30
            ),
            1000,
        )

        a = rng.integers(
            0,
            n_stocks,
            size=draw_size,
            dtype=np.int32,
        )

        b = rng.integers(
            0,
            n_stocks,
            size=draw_size,
            dtype=np.int32,
        )

        different = (
            a != b
        )

        a = a[
            different
        ]

        b = b[
            different
        ]

        low = np.minimum(
            a,
            b,
        )

        high = np.maximum(
            a,
            b,
        )

        new_pairs = np.column_stack(
            [
                low,
                high,
            ]
        )

        collected = np.vstack(
            [
                collected,
                new_pairs,
            ]
        )

        collected = np.unique(
            collected,
            axis=0,
        )

        if (
            len(collected)
            >
            target
        ):

            collected = (
                collected[
                    :target
                ]
            )

    return collected


# ============================================================
# 9. Exact Pairwise Pearson
# ============================================================

def calculate_pairwise_correlations(
    window_returns,
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
            window_returns[
                :,
                a
            ]
            .astype(
                np.float64,
                copy=False,
            )
        )

        y = (
            window_returns[
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

        sufficient = (
            n
            >=
            required_common_days
        )

        if not (
            sufficient.any()
        ):

            continue

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
            n_float > 0,
            n_float,
            np.nan,
        )

        cov_num = (

            sum_xy

            -

            (
                sum_x
                *
                sum_y
                /
                safe_n
            )
        )

        var_x = (

            sum_x2

            -

            (
                sum_x
                *
                sum_x
                /
                safe_n
            )
        )

        var_y = (

            sum_y2

            -

            (
                sum_y
                *
                sum_y
                /
                safe_n
            )
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

            sufficient

            &

            np.isfinite(
                denominator
            )

            &

            (
                denominator > 0
            )
        )

        current_corr = np.full(
            end - start,
            np.nan,
            dtype=float,
        )

        current_corr[
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
            current_corr,
            -1.0,
            1.0,
        )

    return (
        corr,
        common_obs,
    )


# ============================================================
# 10. Correlation Distribution Summary
# ============================================================

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

    abs_values = np.abs(
        values
    )

    return {

        f"{prefix}_pair_count":
            int(
                len(values)
            ),

        f"{prefix}_mean_corr":
            float(
                np.mean(
                    values
                )
            ),

        f"{prefix}_median_corr":
            float(
                np.median(
                    values
                )
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
                    abs_values
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


# ============================================================
# 11. Exact Same-Industry Pair Baseline
#
# 不使用随机样本估计 baseline。
#
# 对 eligible 股票的行业 group size：
#
# n_g
#
# 精确计算：
#
# SamePairs = sum_g C(n_g,2)
# ============================================================

def calculate_exact_industry_pair_structure(
    industry_codes,
):

    industry_codes = np.asarray(
        industry_codes,
        dtype=np.int32,
    )

    industry_codes = industry_codes[
        industry_codes >= 0
    ]

    n = len(
        industry_codes
    )

    if n < 2:

        return {

            "industry_valid_stock_count":
                n,

            "industry_class_count":
                0,

            "exact_total_pair_count":
                0,

            "exact_same_pair_count":
                0,

            "exact_cross_pair_count":
                0,

            "exact_same_pair_share":
                np.nan,
        }

    counts = np.bincount(
        industry_codes
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

    cross_pairs = (
        total_pairs
        -
        same_pairs
    )

    return {

        "industry_valid_stock_count":
            int(n),

        "industry_class_count":
            int(
                len(
                    counts
                )
            ),

        "exact_total_pair_count":
            int(
                total_pairs
            ),

        "exact_same_pair_count":
            same_pairs,

        "exact_cross_pair_count":
            int(
                cross_pairs
            ),

        "exact_same_pair_share":
            float(
                same_pairs
                /
                total_pairs
            ),
    }


# ============================================================
# 12. Strong-edge Same-Industry Enrichment
# ============================================================

def calculate_top_edge_enrichment(
    corr,
    same_mask,
    exact_same_pair_share,
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

    result = {}

    if (
        len(corr) == 0
        or
        not np.isfinite(
            exact_same_pair_share
        )
        or
        exact_same_pair_share <= 0
    ):

        return result

    # --------------------------------------------------------
    # Sampled same share
    # --------------------------------------------------------

    result[
        "sample_same_pair_share"
    ] = float(
        same.mean()
    )

    result[
        "sample_same_share_minus_exact"
    ] = float(
        same.mean()
        -
        exact_same_pair_share
    )

    # ========================================================
    # Top positive correlation
    # ========================================================

    for q in TOP_QUANTILES:

        threshold = float(
            np.quantile(
                corr,
                q,
            )
        )

        top = (
            corr
            >=
            threshold
        )

        top_same_share = float(
            same[
                top
            ].mean()
        )

        suffix = int(
            round(
                (
                    1.0
                    -
                    q
                )
                *
                100
            )
        )

        result[
            f"top{suffix}_corr_threshold"
        ] = threshold

        result[
            f"top{suffix}_same_share"
        ] = top_same_share

        result[
            f"top{suffix}_same_enrichment"
        ] = (
            top_same_share
            /
            exact_same_pair_share
        )

    # ========================================================
    # Top absolute correlation
    # ========================================================

    abs_corr = np.abs(
        corr
    )

    threshold_abs = float(
        np.quantile(
            abs_corr,
            0.99,
        )
    )

    top_abs = (
        abs_corr
        >=
        threshold_abs
    )

    top_abs_same_share = float(
        same[
            top_abs
        ].mean()
    )

    result[
        "top1_abs_corr_threshold"
    ] = threshold_abs

    result[
        "top1_abs_same_share"
    ] = top_abs_same_share

    result[
        "top1_abs_same_enrichment"
    ] = (
        top_abs_same_share
        /
        exact_same_pair_share
    )

    return result


# ============================================================
# 13. Step 4 Industry Change Diagnostics
# ============================================================

def load_industry_change_events():

    if not (
        INDUSTRY_CHANGE_EVENTS_PATH.exists()
    ):

        return pd.DataFrame()

    df = pd.read_csv(
        INDUSTRY_CHANGE_EVENTS_PATH
    )

    if df.empty:

        return df

    df["change_date"] = pd.to_datetime(
        df["change_date"],
        errors="coerce",
    )

    return df


def summarize_industry_changes_in_window(
    events,
    industry_level,
    start_date,
    end_date,
):

    if events.empty:

        return {

            "industry_change_event_count":
                0,

            "industry_change_stock_count":
                0,

            "max_daily_industry_change_count":
                0,

            "industry_revision_alert":
                False,
        }

    sub = events[
        (
            events["source"]
            ==
            industry_level
        )
        &
        (
            events["change_date"]
            >=
            start_date
        )
        &
        (
            events["change_date"]
            <=
            end_date
        )
    ]

    if sub.empty:

        return {

            "industry_change_event_count":
                0,

            "industry_change_stock_count":
                0,

            "max_daily_industry_change_count":
                0,

            "industry_revision_alert":
                False,
        }

    daily_count = (
        sub.groupby(
            "change_date"
        )
        .size()
    )

    max_daily = int(
        daily_count.max()
    )

    return {

        "industry_change_event_count":
            int(
                len(sub)
            ),

        "industry_change_stock_count":
            int(
                sub[
                    "security_id"
                ]
                .nunique()
            ),

        "max_daily_industry_change_count":
            max_daily,

        "industry_revision_alert":
            bool(
                max_daily
                >=
                INDUSTRY_REVISION_ALERT_COUNT
            ),
    }


# ============================================================
# 14. 主分析
# ============================================================

def run_same_cross_analysis(
    trade_dates,
    master,
    returns,
    industry_snapshots,
    analysis_dates,
    representative_dates,
    industry_change_events,
    windows,
    stock_valid_ratio,
    pair_valid_ratio,
):

    date_to_idx = {

        pd.Timestamp(date):
            i

        for i, date in enumerate(
            trade_dates
        )
    }

    # --------------------------------------------------------
    # Rolling valid count
    # --------------------------------------------------------

    valid = np.isfinite(
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
        valid,
        axis=0,
        dtype=np.int32,
    )

    summary_rows = []

    representative_pair_parts = []

    # ========================================================
    # Window
    # ========================================================

    for window in windows:

        print()
        print("=" * 76)
        print(
            f"Same/Cross Industry: W={window}"
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
                ]
                .eq(
                    window
                )
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
            # 1. Same eligible universe as Step 2
            # =================================================

            pit_mask = (
                current_pit_mask(
                    master,
                    date,
                )
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

            eligible_idx = (
                np.flatnonzero(
                    eligible_mask
                )
            )

            p = len(
                eligible_idx
            )

            if p < 2:

                continue

            # =================================================
            # 2. Same deterministic sample as Step 2
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
            )

            rng = np.random.default_rng(
                seed
            )

            local_pairs = (
                sample_unique_pairs(
                    n_stocks=p,
                    sample_size=
                        PAIR_SAMPLE_SIZE,
                    rng=rng,
                )
            )

            stock_a = (
                eligible_idx[
                    local_pairs[
                        :,
                        0
                    ]
                ]
            )

            stock_b = (
                eligible_idx[
                    local_pairs[
                        :,
                        1
                    ]
                ]
            )

            # =================================================
            # 3. Pairwise Pearson
            # =================================================

            window_returns = (
                returns[
                    start_idx:
                    end_idx + 1,
                    :
                ]
            )

            (
                corr,
                common_obs,
            ) = (
                calculate_pairwise_correlations(
                    window_returns=
                        window_returns,

                    stock_a=
                        stock_a,

                    stock_b=
                        stock_b,

                    required_common_days=
                        required_pair_days,
                )
            )

            valid_corr = (
                np.isfinite(
                    corr
                )
            )

            # =================================================
            # 4. 每一个行业层级
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
                        "的历史行业快照。"
                    )

                industry_all = (
                    industry_snapshots[
                        industry_level
                    ][
                        date
                    ]
                )

                eligible_industry = (
                    industry_all[
                        eligible_idx
                    ]
                )

                exact_structure = (
                    calculate_exact_industry_pair_structure(
                        eligible_industry
                    )
                )

                industry_a = (
                    industry_all[
                        stock_a
                    ]
                )

                industry_b = (
                    industry_all[
                        stock_b
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

                same_valid = (
                    same_mask
                    &
                    valid_corr
                )

                cross_valid = (
                    cross_mask
                    &
                    valid_corr
                )

                same_corr = (
                    corr[
                        same_valid
                    ]
                )

                cross_corr = (
                    corr[
                        cross_valid
                    ]
                )

                all_labeled_valid = (
                    valid_corr
                    &
                    valid_industry
                )

                # =============================================
                # Group summaries
                # =============================================

                same_summary = (
                    summarize_corr_group(
                        same_corr,
                        "same",
                    )
                )

                cross_summary = (
                    summarize_corr_group(
                        cross_corr,
                        "cross",
                    )
                )

                # =============================================
                # Gaps
                # =============================================

                same_mean = (
                    same_summary.get(
                        "same_mean_corr",
                        np.nan,
                    )
                )

                cross_mean = (
                    cross_summary.get(
                        "cross_mean_corr",
                        np.nan,
                    )
                )

                same_median = (
                    same_summary.get(
                        "same_median_corr",
                        np.nan,
                    )
                )

                cross_median = (
                    cross_summary.get(
                        "cross_median_corr",
                        np.nan,
                    )
                )

                same_abs = (
                    same_summary.get(
                        "same_mean_abs_corr",
                        np.nan,
                    )
                )

                cross_abs = (
                    cross_summary.get(
                        "cross_mean_abs_corr",
                        np.nan,
                    )
                )

                # =============================================
                # Top-edge enrichment
                # =============================================

                enrichment = (
                    calculate_top_edge_enrichment(
                        corr=
                            corr[
                                all_labeled_valid
                            ],

                        same_mask=
                            same_mask[
                                all_labeled_valid
                            ],

                        exact_same_pair_share=
                            exact_structure[
                                "exact_same_pair_share"
                            ],
                    )
                )

                # =============================================
                # Industry classification change diagnostics
                # =============================================

                change_diag = (
                    summarize_industry_changes_in_window(
                        events=
                            industry_change_events,

                        industry_level=
                            industry_level,

                        start_date=
                            window_start,

                        end_date=
                            date,
                    )
                )

                # =============================================
                # Row
                # =============================================

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

                        "sampled_pair_count":
                            int(
                                len(
                                    local_pairs
                                )
                            ),

                        "valid_corr_pair_count":
                            int(
                                valid_corr.sum()
                            ),

                        "valid_industry_pair_count":
                            int(
                                all_labeled_valid.sum()
                            ),

                        **exact_structure,

                        **same_summary,

                        **cross_summary,

                        "same_minus_cross_mean_corr":
                            (
                                same_mean
                                -
                                cross_mean
                            ),

                        "same_minus_cross_median_corr":
                            (
                                same_median
                                -
                                cross_median
                            ),

                        "same_minus_cross_mean_abs_corr":
                            (
                                same_abs
                                -
                                cross_abs
                            ),

                        "same_to_cross_mean_corr_ratio":
                            (
                                same_mean
                                /
                                cross_mean
                                if (
                                    np.isfinite(
                                        cross_mean
                                    )
                                    and
                                    abs(
                                        cross_mean
                                    )
                                    >
                                    1e-12
                                )
                                else
                                np.nan
                            ),

                        "same_pair_sample_warning":
                            bool(
                                len(
                                    same_corr
                                )
                                <
                                MIN_SAME_PAIR_SAMPLE
                            ),

                        **enrichment,

                        **change_diag,
                    }
                )

                # =============================================
                # Representative pair-level output
                # =============================================

                if (
                    SAVE_REPRESENTATIVE_PAIRS
                    and
                    (
                        window,
                        date,
                    )
                    in
                    representative_dates
                ):

                    keep = (
                        all_labeled_valid
                    )

                    if keep.any():

                        rep = pd.DataFrame(
                            {

                                "analysis_date":
                                    date,

                                "window":
                                    window,

                                "industry_level":
                                    industry_level,

                                "security_id_i":
                                    master.iloc[
                                        stock_a[
                                            keep
                                        ]
                                    ][
                                        "security_id"
                                    ]
                                    .to_numpy(),

                                "stock_code_i":
                                    master.iloc[
                                        stock_a[
                                            keep
                                        ]
                                    ][
                                        "stock_code"
                                    ]
                                    .to_numpy(),

                                "stock_name_i":
                                    master.iloc[
                                        stock_a[
                                            keep
                                        ]
                                    ][
                                        "stock_name"
                                    ]
                                    .to_numpy(),

                                "security_id_j":
                                    master.iloc[
                                        stock_b[
                                            keep
                                        ]
                                    ][
                                        "security_id"
                                    ]
                                    .to_numpy(),

                                "stock_code_j":
                                    master.iloc[
                                        stock_b[
                                            keep
                                        ]
                                    ][
                                        "stock_code"
                                    ]
                                    .to_numpy(),

                                "stock_name_j":
                                    master.iloc[
                                        stock_b[
                                            keep
                                        ]
                                    ][
                                        "stock_name"
                                    ]
                                    .to_numpy(),

                                "industry_code_i":
                                    industry_a[
                                        keep
                                    ],

                                "industry_code_j":
                                    industry_b[
                                        keep
                                    ],

                                "same_industry":
                                    same_mask[
                                        keep
                                    ],

                                "common_obs":
                                    common_obs[
                                        keep
                                    ],

                                "correlation":
                                    corr[
                                        keep
                                    ],

                                "abs_correlation":
                                    np.abs(
                                        corr[
                                            keep
                                        ]
                                    ),
                            }
                        )

                        representative_pair_parts.append(
                            rep
                        )

            # =================================================
            # Console
            # =================================================

            if (
                counter == 1
                or
                counter % 12 == 0
                or
                counter
                ==
                len(
                    dates_w
                )
            ):

                current_rows = [
                    x
                    for x in summary_rows
                    if (
                        x[
                            "window"
                        ]
                        ==
                        window
                        and
                        x[
                            "analysis_date"
                        ]
                        ==
                        date
                        and
                        x[
                            "industry_level"
                        ]
                        ==
                        MAIN_INDUSTRY_LEVEL
                    )
                ]

                if current_rows:

                    current = (
                        current_rows[-1]
                    )

                    print(
                        f"[{counter:3d}/"
                        f"{len(dates_w):3d}] "
                        f"{date.date()} | "
                        f"p={p:,} | "
                        f"Same={current.get('same_mean_corr', np.nan):.4f} | "
                        f"Cross={current.get('cross_mean_corr', np.nan):.4f} | "
                        f"Gap={current.get('same_minus_cross_mean_corr', np.nan):.4f}"
                    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    if representative_pair_parts:

        representative_df = (
            pd.concat(
                representative_pair_parts,
                ignore_index=True,
            )
        )

    else:

        representative_df = (
            pd.DataFrame()
        )

    return (
        summary_df,
        representative_df,
    )


# ============================================================
# 15. Window Comparison
# ============================================================

def build_window_comparison(
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
                    int(
                        window
                    ),

                "analysis_date_count":
                    int(
                        len(g)
                    ),

                "mean_exact_same_pair_share":
                    float(
                        g[
                            "exact_same_pair_share"
                        ]
                        .mean()
                    ),

                "mean_same_corr":
                    float(
                        g[
                            "same_mean_corr"
                        ]
                        .mean()
                    ),

                "mean_cross_corr":
                    float(
                        g[
                            "cross_mean_corr"
                        ]
                        .mean()
                    ),

                "mean_same_cross_gap":
                    float(
                        g[
                            "same_minus_cross_mean_corr"
                        ]
                        .mean()
                    ),

                "median_same_cross_gap":
                    float(
                        g[
                            "same_minus_cross_mean_corr"
                        ]
                        .median()
                    ),

                "mean_same_abs_corr":
                    float(
                        g[
                            "same_mean_abs_corr"
                        ]
                        .mean()
                    ),

                "mean_cross_abs_corr":
                    float(
                        g[
                            "cross_mean_abs_corr"
                        ]
                        .mean()
                    ),

                "mean_top5_same_enrichment":
                    float(
                        g[
                            "top5_same_enrichment"
                        ]
                        .mean()
                    ),

                "median_top5_same_enrichment":
                    float(
                        g[
                            "top5_same_enrichment"
                        ]
                        .median()
                    ),

                "mean_top1_same_enrichment":
                    float(
                        g[
                            "top1_same_enrichment"
                        ]
                        .mean()
                    ),

                "median_top1_same_enrichment":
                    float(
                        g[
                            "top1_same_enrichment"
                        ]
                        .median()
                    ),

                "mean_top1_abs_same_enrichment":
                    float(
                        g[
                            "top1_abs_same_enrichment"
                        ]
                        .mean()
                    ),

                "mean_same_share_gt_0_5":
                    float(
                        g[
                            "same_share_gt_0_5"
                        ]
                        .mean()
                    ),

                "mean_cross_share_gt_0_5":
                    float(
                        g[
                            "cross_share_gt_0_5"
                        ]
                        .mean()
                    ),

                "industry_revision_alert_months":
                    int(
                        g[
                            "industry_revision_alert"
                        ]
                        .sum()
                    ),

                "same_pair_low_sample_months":
                    int(
                        g[
                            "same_pair_sample_warning"
                        ]
                        .sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 16. Yearly Summary
# ============================================================

def build_yearly_summary(
    summary_df,
):

    df = summary_df.copy()

    df["year"] = (
        pd.to_datetime(
            df["analysis_date"]
        )
        .dt.year
    )

    return (

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

            exact_same_pair_share=
                (
                    "exact_same_pair_share",
                    "mean",
                ),

            same_mean_corr=
                (
                    "same_mean_corr",
                    "mean",
                ),

            cross_mean_corr=
                (
                    "cross_mean_corr",
                    "mean",
                ),

            same_cross_gap=
                (
                    "same_minus_cross_mean_corr",
                    "mean",
                ),

            same_mean_abs_corr=
                (
                    "same_mean_abs_corr",
                    "mean",
                ),

            cross_mean_abs_corr=
                (
                    "cross_mean_abs_corr",
                    "mean",
                ),

            top5_same_enrichment=
                (
                    "top5_same_enrichment",
                    "mean",
                ),

            top1_same_enrichment=
                (
                    "top1_same_enrichment",
                    "mean",
                ),

            top1_abs_same_enrichment=
                (
                    "top1_abs_same_enrichment",
                    "mean",
                ),

            max_daily_industry_change_count=
                (
                    "max_daily_industry_change_count",
                    "max",
                ),
        )
    )


# ============================================================
# 17. 合并 Step 2 Raw + Step 3 Common Factor
# ============================================================

def merge_external_diagnostics(
    summary_df,
):

    result = summary_df.copy()

    # ========================================================
    # Step 2
    # ========================================================

    if RAW_CORRELATION_PATH.exists():

        raw = pd.read_csv(
            RAW_CORRELATION_PATH
        )

        raw[
            "analysis_date"
        ] = pd.to_datetime(
            raw[
                "analysis_date"
            ]
        )

        raw_keep = [

            "analysis_date",
            "window",
            "mean_corr",
            "median_corr",
            "mean_abs_corr",
        ]

        raw_keep = [

            x
            for x in raw_keep

            if x in raw.columns
        ]

        raw = (
            raw[
                raw_keep
            ]
            .rename(
                columns={

                    "mean_corr":
                        "raw_mean_corr",

                    "median_corr":
                        "raw_median_corr",

                    "mean_abs_corr":
                        "raw_mean_abs_corr",
                }
            )
        )

        result = result.merge(
            raw,
            on=[
                "analysis_date",
                "window",
            ],
            how="left",
            validate=
                "many_to_one",
        )

    # ========================================================
    # Step 3
    # ========================================================

    if COMMON_FACTOR_PATH.exists():

        common = pd.read_csv(
            COMMON_FACTOR_PATH
        )

        common[
            "analysis_date"
        ] = pd.to_datetime(
            common[
                "analysis_date"
            ]
        )

        candidate = [

            "analysis_date",
            "window",

            "pca_pc1_evr",
            "pca_pc1_5_evr",

            "ew_r2_mean",
            "ew_r2_median",

            "vw_r2_mean",
            "vw_r2_median",
        ]

        available = [

            x
            for x in candidate

            if x in common.columns
        ]

        common = common[
            available
        ]

        result = result.merge(
            common,
            on=[
                "analysis_date",
                "window",
            ],
            how="left",
            validate=
                "many_to_one",
        )

    return result


# ============================================================
# 18. Safe correlation
# ============================================================

def safe_corr(
    x,
    y,
):

    x = pd.to_numeric(
        x,
        errors="coerce",
    )

    y = pd.to_numeric(
        y,
        errors="coerce",
    )

    valid = (
        x.notna()
        &
        y.notna()
    )

    if (
        valid.sum()
        <
        3
    ):

        return np.nan

    xv = (
        x[
            valid
        ]
        .to_numpy(
            dtype=float
        )
    )

    yv = (
        y[
            valid
        ]
        .to_numpy(
            dtype=float
        )
    )

    if (
        np.std(xv)
        <=
        0
        or
        np.std(yv)
        <=
        0
    ):

        return np.nan

    return float(
        np.corrcoef(
            xv,
            yv,
        )[0, 1]
    )


# ============================================================
# 19. 行业结构 vs Market Mode
# ============================================================

def build_market_mode_relationship_summary(
    merged,
):

    rows = []

    for (
        industry_level,
        window,
    ), g in (

        merged.groupby(
            [
                "industry_level",
                "window",
            ]
        )
    ):

        row = {

            "industry_level":
                industry_level,

            "window":
                int(
                    window
                ),

            "month_count":
                int(
                    len(g)
                ),
        }

        relationships = {

            # ------------------------------------------------
            # 行业 gap 与 Raw Market Synchronization
            # ------------------------------------------------

            "corr_gap_raw_mean_corr":
                (
                    "same_minus_cross_mean_corr",
                    "raw_mean_corr",
                ),

            # ------------------------------------------------
            # 行业 gap 与 PC1
            # ------------------------------------------------

            "corr_gap_pc1_evr":
                (
                    "same_minus_cross_mean_corr",
                    "pca_pc1_evr",
                ),

            # ------------------------------------------------
            # 行业 gap 与 Market R2
            # ------------------------------------------------

            "corr_gap_ew_median_r2":
                (
                    "same_minus_cross_mean_corr",
                    "ew_r2_median",
                ),

            # ------------------------------------------------
            # Top 1% 行业 enrichment 与 Market Mode
            # ------------------------------------------------

            "corr_top1_enrichment_pc1_evr":
                (
                    "top1_same_enrichment",
                    "pca_pc1_evr",
                ),

            "corr_top1_enrichment_ew_median_r2":
                (
                    "top1_same_enrichment",
                    "ew_r2_median",
                ),

            # ------------------------------------------------
            # Top 5%
            # ------------------------------------------------

            "corr_top5_enrichment_pc1_evr":
                (
                    "top5_same_enrichment",
                    "pca_pc1_evr",
                ),
        }

        for name, (
            x_col,
            y_col,
        ) in relationships.items():

            if (
                x_col in g.columns
                and
                y_col in g.columns
            ):

                row[name] = safe_corr(
                    g[
                        x_col
                    ],
                    g[
                        y_col
                    ],
                )

            else:

                row[name] = np.nan

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 20. Step 2 Replication QA
#
# 因为 Step 5 使用与 Step 2 相同 seed 和 pair sample，
# 所以重新计算的 overall mean corr 应该与
# raw_correlation_summary.csv 非常接近/一致。
# ============================================================

def build_step2_replication_qa(
    merged,
):

    if (
        "raw_mean_corr"
        not in merged.columns
    ):

        return pd.DataFrame()

    main = merged[
        merged[
            "industry_level"
        ]
        .eq(
            MAIN_INDUSTRY_LEVEL
        )
    ].copy()

    same_n = (
        main[
            "same_pair_count"
        ]
        .astype(float)
    )

    cross_n = (
        main[
            "cross_pair_count"
        ]
        .astype(float)
    )

    denominator = (
        same_n
        +
        cross_n
    )

    reconstructed = np.where(

        denominator
        >
        0,

        (
            same_n
            *
            main[
                "same_mean_corr"
            ]

            +

            cross_n
            *
            main[
                "cross_mean_corr"
            ]
        )
        /
        denominator,

        np.nan,
    )

    qa = main[
        [
            "analysis_date",
            "window",
            "raw_mean_corr",
        ]
    ].copy()

    qa[
        "step5_reconstructed_mean_corr"
    ] = reconstructed

    qa[
        "difference"
    ] = (
        qa[
            "step5_reconstructed_mean_corr"
        ]
        -
        qa[
            "raw_mean_corr"
        ]
    )

    qa[
        "abs_difference"
    ] = np.abs(
        qa[
            "difference"
        ]
    )

    return qa


# ============================================================
# 21. 绘图
#
# 主图只使用 industry_id1。
# industry_id2 保存在 CSV 中作为 robustness。
# ============================================================

def plot_main_industry_results(
    merged,
    windows,
):

    main = merged[
        merged[
            "industry_level"
        ]
        .eq(
            MAIN_INDUSTRY_LEVEL
        )
    ].copy()

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

        # ====================================================
        # Figure 1:
        # Same vs Cross Mean Correlation
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(
                12,
                6,
            )
        )

        ax.plot(
            df[
                "analysis_date"
            ],
            df[
                "same_mean_corr"
            ],
            label=
                "Same Industry",
        )

        ax.plot(
            df[
                "analysis_date"
            ],
            df[
                "cross_mean_corr"
            ],
            label=
                "Cross Industry",
        )

        ax.set_title(
            f"Same vs Cross Industry Raw Correlation - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Mean Correlation"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            /
            f"same_cross_mean_corr_W{window}.png",
            dpi=180,
        )

        plt.close(
            fig
        )

        # ====================================================
        # Figure 2:
        # Same-Cross Gap
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(
                12,
                6,
            )
        )

        ax.plot(
            df[
                "analysis_date"
            ],
            df[
                "same_minus_cross_mean_corr"
            ],
        )

        ax.axhline(
            0,
            linewidth=1,
        )

        ax.set_title(
            f"Same-Cross Industry Correlation Gap - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Mean Corr(Same) - Mean Corr(Cross)"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            /
            f"same_cross_gap_W{window}.png",
            dpi=180,
        )

        plt.close(
            fig
        )

        # ====================================================
        # Figure 3:
        # Top Edge Enrichment
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(
                12,
                6,
            )
        )

        ax.plot(
            df[
                "analysis_date"
            ],
            df[
                "top5_same_enrichment"
            ],
            label=
                "Top 5%",
        )

        ax.plot(
            df[
                "analysis_date"
            ],
            df[
                "top1_same_enrichment"
            ],
            label=
                "Top 1%",
        )

        ax.axhline(
            1,
            linewidth=1,
        )

        ax.set_title(
            f"Same-Industry Enrichment among Strong Raw Correlations - W={window}"
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
            /
            f"same_industry_top_edge_enrichment_W{window}.png",
            dpi=180,
        )

        plt.close(
            fig
        )

        # ====================================================
        # Figure 4:
        # Industry Gap vs PC1 EVR
        # ====================================================

        if (
            "pca_pc1_evr"
            in df.columns
        ):

            valid = (
                df[
                    "pca_pc1_evr"
                ]
                .notna()
                &
                df[
                    "same_minus_cross_mean_corr"
                ]
                .notna()
            )

            if (
                valid.sum()
                >=
                3
            ):

                fig, ax = plt.subplots(
                    figsize=(
                        7,
                        6,
                    )
                )

                ax.scatter(
                    df.loc[
                        valid,
                        "pca_pc1_evr"
                    ],
                    df.loc[
                        valid,
                        "same_minus_cross_mean_corr"
                    ],
                )

                ax.set_title(
                    f"Industry Gap vs Market Mode - W={window}"
                )

                ax.set_xlabel(
                    "PC1 Explained Variance Ratio"
                )

                ax.set_ylabel(
                    "Same-Cross Correlation Gap"
                )

                ax.grid(
                    alpha=0.25
                )

                fig.tight_layout()

                fig.savefig(
                    OUTPUT_DIR
                    /
                    f"industry_gap_vs_pc1_evr_W{window}.png",
                    dpi=180,
                )

                plt.close(
                    fig
                )


# ============================================================
# 22. Metadata
# ============================================================

def save_metadata(
    windows,
    stock_valid_ratio,
    pair_valid_ratio,
    return_column,
):

    metadata = {

        "step":
            "Day2_Step5_Same_Cross_Industry_Association",

        "output_dir":
            str(
                OUTPUT_DIR
            ),

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

        "robustness_industry_level":
            "industry_id2",

        "industry_classification_rule":
            (
                "Same/Cross status is defined using the "
                "historically indexed industry label observed "
                "at the network analysis date (window endpoint)."
            ),

        "correlation_method":
            "Pearson pairwise-complete raw return correlation",

        "pair_sampling":
            {

                "sample_size_per_window_date":
                    PAIR_SAMPLE_SIZE,

                "sampling_rule":
                    (
                        "Uniform random sampling among eligible "
                        "undirected stock pairs using the same "
                        "random seed rule as Day 2 Step 2."
                    ),
            },

        "exact_baseline":
            (
                "The unconditional Same-Industry pair share "
                "is computed exactly from eligible industry "
                "group sizes rather than estimated from the "
                "random pair sample."
            ),

        "top_edge_enrichment":
            (
                "Same-industry share among the sampled top "
                "correlation tail divided by the exact "
                "same-industry share among all possible "
                "eligible pairs."
            ),

        "important_notes": [

            (
                "This step studies raw-return Same/Cross "
                "industry structure. It does not yet remove "
                "the market common factor."
            ),

            (
                "Pair-level observations are dependent because "
                "different pairs can share stocks. The analysis "
                "is therefore descriptive; naive independent-"
                "pair t-tests are intentionally not reported."
            ),

            (
                "industry_id1 is the primary classification "
                "because Step 4 showed essentially exact "
                "historical correspondence with the external "
                "historical industry file."
            ),

            (
                "industry_id2 is retained as a finer-industry "
                "robustness analysis."
            ),

            (
                "Concentrated historical industry "
                "reclassification dates can mechanically change "
                "Same/Cross labels. Window-level classification "
                "change diagnostics are therefore retained."
            ),

            (
                "A positive Same-Cross gap does not yet imply "
                "stock-specific dependence because both Same "
                "and Cross correlations still contain the "
                "market-wide common component diagnosed in "
                "Step 3."
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        / "step5_same_cross_metadata.json",
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
# 23. 主程序
# ============================================================

def main():

    print("=" * 76)
    print("M1 Day 2 - Step 5")
    print("Same / Cross Industry Association Structure")
    print("=" * 76)

    # ========================================================
    # 1. Config
    # ========================================================

    print()
    print("[1] 读取 Step 1 Config")

    (
        config,
        windows,
        stock_valid_ratio,
        pair_valid_ratio,
        return_column,
    ) = (
        load_config()
    )

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

    print(
        f"Main Industry : {MAIN_INDUSTRY_LEVEL}"
    )

    # ========================================================
    # 2. Basic
    # ========================================================

    print()
    print(
        "[2] 读取 Calendar / Stock Master / Analysis Dates"
    )

    trade_dates = (
        load_trade_dates()
    )

    master = (
        load_stock_master()
    )

    analysis_dates = (
        load_analysis_dates()
    )

    representative_dates = (
        load_representative_dates()
    )

    # ========================================================
    # 3. Return + Industry
    # ========================================================

    print()
    print(
        "[3] 构造 Return Matrix + Month-End Industry Snapshots"
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

    # 保存行业编码映射
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

    # ========================================================
    # 4. Step 4 change events
    # ========================================================

    industry_change_events = (
        load_industry_change_events()
    )

    # ========================================================
    # 5. Same/Cross
    # ========================================================

    print()
    print("[4] Same/Cross Industry Analysis")

    (
        summary_df,
        representative_pairs,
    ) = (
        run_same_cross_analysis(

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

            industry_change_events=
                industry_change_events,

            windows=
                windows,

            stock_valid_ratio=
                stock_valid_ratio,

            pair_valid_ratio=
                pair_valid_ratio,
        )
    )

    # ========================================================
    # 6. Aggregate
    # ========================================================

    window_comparison = (
        build_window_comparison(
            summary_df
        )
    )

    yearly_summary = (
        build_yearly_summary(
            summary_df
        )
    )

    # ========================================================
    # 7. Merge Step 2 + Step 3
    # ========================================================

    print()
    print(
        "[5] 合并 Raw Correlation + Common Factor"
    )

    merged = (
        merge_external_diagnostics(
            summary_df
        )
    )

    relationship_summary = (
        build_market_mode_relationship_summary(
            merged
        )
    )

    # ========================================================
    # 8. Step 2 replication QA
    # ========================================================

    replication_qa = (
        build_step2_replication_qa(
            merged
        )
    )

    # ========================================================
    # 9. Save
    # ========================================================

    summary_df.to_csv(
        OUTPUT_DIR
        / "same_cross_industry_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    window_comparison.to_csv(
        OUTPUT_DIR
        / "same_cross_industry_window_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    yearly_summary.to_csv(
        OUTPUT_DIR
        / "same_cross_industry_yearly_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    merged.to_csv(
        OUTPUT_DIR
        / "same_cross_vs_market_mode.csv",
        index=False,
        encoding="utf-8-sig",
    )

    relationship_summary.to_csv(
        OUTPUT_DIR
        / "same_cross_vs_market_mode_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not (
        replication_qa.empty
    ):

        replication_qa.to_csv(
            OUTPUT_DIR
            / "step2_replication_qa.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if not (
        representative_pairs.empty
    ):

        representative_pairs.to_parquet(
            OUTPUT_DIR
            / "representative_same_cross_pairs.parquet",
            index=False,
        )

    # ========================================================
    # 10. Figures
    # ========================================================

    print()
    print("[6] 绘图")

    plot_main_industry_results(
        merged=
            merged,

        windows=
            windows,
    )

    # ========================================================
    # 11. Metadata
    # ========================================================

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

    # ========================================================
    # 12. Console
    # ========================================================

    print()
    print("=" * 76)
    print("Day 2 Step 5 完成")
    print("=" * 76)

    print()
    print("Window Comparison:")

    show = [

        "industry_level",
        "window",

        "mean_exact_same_pair_share",

        "mean_same_corr",
        "mean_cross_corr",

        "mean_same_cross_gap",

        "mean_top5_same_enrichment",
        "mean_top1_same_enrichment",

        "industry_revision_alert_months",
    ]

    print(
        window_comparison[
            show
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Industry Structure vs Market Mode:"
    )

    print(
        relationship_summary
        .to_string(
            index=False
        )
    )

    if not replication_qa.empty:

        print()
        print(
            "Step 2 replication QA:"
        )

        print(
            "Mean absolute difference = "
            f"{replication_qa['abs_difference'].mean():.8f}"
        )

        print(
            "Max absolute difference  = "
            f"{replication_qa['abs_difference'].max():.8f}"
        )

    print()
    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()