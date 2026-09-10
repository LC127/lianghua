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
    / "08_robustness_market_industry"
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
# 1. Robustness Design
# ============================================================

PAIR_SAMPLE_SIZE = 20_000

PAIR_BATCH_SIZE = 2_000

PARQUET_BATCH_SIZE = 300_000

RANDOM_SEED = 20260910

MAIN_INDUSTRY_LEVEL = (
    "industry_id1"
)


# ------------------------------------------------------------
# Sampling robustness:
#
# 推荐使用所有 Month-End，
# 因为这里专门检查 Step 2/5 的 sampling artifact。
#
# 如果只是先测试代码，可以改成：
#
# SAMPLER_DATE_MODE = "representative"
# ------------------------------------------------------------

SAMPLER_DATE_MODE = "all"


# ------------------------------------------------------------
# Specification robustness:
#
# 默认为 representative year-end，
# 可以显著降低运行时间。
#
# 最终正式论文版本如果计算资源允许，
# 可以改成 "all"。
# ------------------------------------------------------------

SPEC_DATE_MODE = "representative"


# ------------------------------------------------------------
# 四个清晰 specification
# ------------------------------------------------------------

ROBUSTNESS_CONFIGS = [

    {
        "config_id":
            "BASE",

        "return_column":
            "return_network",

        "stock_valid_ratio":
            0.90,

        "pair_valid_ratio":
            0.90,

        "description":
            "Baseline specification",
    },

    {
        "config_id":
            "RETURN",

        "return_column":
            "return_last_trade_log",

        "stock_valid_ratio":
            0.90,

        "pair_valid_ratio":
            0.90,

        "description":
            "Alternative last-trade return",
    },

    {
        "config_id":
            "C80",

        "return_column":
            "return_network",

        "stock_valid_ratio":
            0.80,

        "pair_valid_ratio":
            0.80,

        "description":
            "Looser 80% valid-observation requirement",
    },

    {
        "config_id":
            "C95",

        "return_column":
            "return_network",

        "stock_valid_ratio":
            0.95,

        "pair_valid_ratio":
            0.95,

        "description":
            "Stricter 95% valid-observation requirement",
    },
]


# ============================================================
# 2. Basic Loaders
# ============================================================

def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    windows = [
        int(x)
        for x
        in config[
            "rolling_design"
        ][
            "windows"
        ]
    ]

    return (
        config,
        windows,
    )


def load_trade_dates():

    df = pd.read_csv(
        TRADE_CALENDAR_PATH
    )

    if "trade_date" not in df.columns:

        raise ValueError(
            "trade_calendar_used.csv 缺少 trade_date。"
        )

    dates = pd.to_datetime(
        df[
            "trade_date"
        ],
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
            f"stock_master 缺字段：{missing}"
        )

    master = master.copy()

    master[
        "security_id"
    ] = (
        master[
            "security_id"
        ]
        .astype("string")
        .str.strip()
    )

    master[
        "list_date"
    ] = pd.to_datetime(
        master[
            "list_date"
        ],
        errors="coerce",
    )

    master[
        "delist_date"
    ] = pd.to_datetime(
        master[
            "delist_date"
        ],
        errors="coerce",
    )

    if (
        master[
            "security_id"
        ]
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

    df[
        "analysis_date"
    ] = pd.to_datetime(
        df[
            "analysis_date"
        ]
    )

    df[
        "window"
    ] = (
        pd.to_numeric(
            df[
                "window"
            ],
            errors="raise",
        )
        .astype(int)
    )

    return df


def load_representative_dates():

    if not (
        REPRESENTATIVE_DATES_PATH.exists()
    ):

        raise FileNotFoundError(
            f"没有找到：{REPRESENTATIVE_DATES_PATH}"
        )

    df = pd.read_csv(
        REPRESENTATIVE_DATES_PATH
    )

    df[
        "analysis_date"
    ] = pd.to_datetime(
        df[
            "analysis_date"
        ]
    )

    df[
        "window"
    ] = (
        pd.to_numeric(
            df[
                "window"
            ],
            errors="raise",
        )
        .astype(int)
    )

    return df


# ============================================================
# 3. Build Two Return Matrices + Industry Snapshot
#
# 一次扫描 parquet，同时读取：
#
# return_network
# return_last_trade_log
# industry_id1
#
# 避免重复读取 1100 多万行数据。
# ============================================================

def build_data_matrices(
    trade_dates,
    master,
    analysis_dates,
):

    print()
    print("=" * 76)
    print(
        "Build Return Matrices + Industry Snapshots"
    )
    print("=" * 76)

    security_ids = (
        master[
            "security_id"
        ]
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

    return_matrices = {

        "return_network":
            np.full(
                (T, P),
                np.nan,
                dtype=np.float32,
            ),

        "return_last_trade_log":
            np.full(
                (T, P),
                np.nan,
                dtype=np.float32,
            ),
    }

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

    industry_parts = []

    parquet_file = pq.ParquetFile(
        RETURN_PANEL_PATH
    )

    required = [
        "trade_date",
        "security_id",
        "return_network",
        "return_last_trade_log",
        MAIN_INDUSTRY_LEVEL,
    ]

    missing = [
        x
        for x in required
        if x not in (
            parquet_file
            .schema_arrow
            .names
        )
    ]

    if missing:

        raise ValueError(
            f"daily_return_panel 缺字段：{missing}"
        )

    total_rows = 0

    for batch_no, batch in enumerate(

        parquet_file.iter_batches(
            batch_size=
                PARQUET_BATCH_SIZE,

            columns=
                required,
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
            errors="coerce",
        )

        df[
            "security_id"
        ] = (
            df[
                "security_id"
            ]
            .astype("string")
            .str.strip()
        )

        row_idx = (
            df[
                "trade_date"
            ]
            .map(
                date_to_idx
            )
        )

        col_idx = (
            df[
                "security_id"
            ]
            .map(
                security_to_idx
            )
        )

        mapped = (
            row_idx.notna()
            &
            col_idx.notna()
        )

        if mapped.any():

            rr = (
                row_idx.loc[
                    mapped
                ]
                .astype(np.int32)
                .to_numpy()
            )

            cc = (
                col_idx.loc[
                    mapped
                ]
                .astype(np.int32)
                .to_numpy()
            )

            for return_column in (
                return_matrices
            ):

                values = pd.to_numeric(
                    df.loc[
                        mapped,
                        return_column,
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )

                finite = np.isfinite(
                    values
                )

                return_matrices[
                    return_column
                ][
                    rr[
                        finite
                    ],
                    cc[
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

        # ----------------------------------------------------
        # Industry snapshots only at analysis dates
        # ----------------------------------------------------

        snap_mask = (
            mapped
            &
            df[
                "trade_date"
            ]
            .isin(
                analysis_date_set
            )
        )

        if snap_mask.any():

            temp = df.loc[
                snap_mask,
                [
                    "trade_date",
                    "security_id",
                    MAIN_INDUSTRY_LEVEL,
                ]
            ].copy()

            temp[
                "master_index"
            ] = (
                col_idx.loc[
                    snap_mask
                ]
                .astype(
                    np.int32
                )
                .to_numpy()
            )

            industry_parts.append(
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
    # Industry encoding
    # ========================================================

    industry_df = pd.concat(
        industry_parts,
        ignore_index=True,
    )

    industry_df[
        "trade_date"
    ] = pd.to_datetime(
        industry_df[
            "trade_date"
        ]
    )

    values = (
        industry_df[
            MAIN_INDUSTRY_LEVEL
        ]
        .astype("string")
    )

    codes, uniques = (
        pd.factorize(
            values,
            sort=True,
        )
    )

    industry_df[
        "industry_code"
    ] = codes.astype(
        np.int32
    )

    codebook = {
        int(i): str(x)
        for i, x
        in enumerate(uniques)
    }

    industry_snapshots = {}

    for date, g in (
        industry_df.groupby(
            "trade_date",
            sort=False,
        )
    ):

        arr = np.full(
            P,
            -1,
            dtype=np.int32,
        )

        idx = (
            g[
                "master_index"
            ]
            .to_numpy(
                dtype=np.int32
            )
        )

        arr[
            idx
        ] = (
            g[
                "industry_code"
            ]
            .to_numpy(
                dtype=np.int32
            )
        )

        industry_snapshots[
            pd.Timestamp(date)
        ] = arr

    print()
    print(
        "Industry snapshot dates: "
        f"{len(industry_snapshots):,}"
    )

    return (
        return_matrices,
        industry_snapshots,
        codebook,
    )


# ============================================================
# 4. PIT
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
# 5. Legacy Sampler
#
# 这里故意复现旧实现，
# 用于 robustness comparison。
# ============================================================

def sample_pairs_legacy(
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
            (0, 2),
            dtype=np.int32,
        )

    collected = np.empty(
        (0, 2),
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

        valid = (
            a != b
        )

        low = np.minimum(
            a[
                valid
            ],
            b[
                valid
            ],
        )

        high = np.maximum(
            a[
                valid
            ],
            b[
                valid
            ],
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

        # ----------------------------------------------------
        # np.unique 会排序
        # ----------------------------------------------------

        collected = np.unique(
            collected,
            axis=0,
        )

        # ----------------------------------------------------
        # 这就是旧代码潜在 bias 的来源
        # ----------------------------------------------------

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
# 6. Correct Uniform Sampler
#
# 从全部 C(p,2) pair ranks 中
# 直接无放回均匀抽样。
# ============================================================

def sample_pairs_uniform(
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
            (0, 2),
            dtype=np.int32,
        )

    ranks = rng.choice(
        total_pairs,
        size=target,
        replace=False,
    )

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
            ranks,
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
            ranks
            -
            cumulative[
                i
            ]
        )
    )

    return (
        np.column_stack(
            [
                i,
                j,
            ]
        )
        .astype(
            np.int32
        )
    )


# ============================================================
# 7. Pairwise Pearson
# ============================================================

def calculate_pairwise_corr(
    matrix,
    stock_a,
    stock_b,
    required_common_days,
):

    n_pairs = len(
        stock_a
    )

    result = np.full(
        n_pairs,
        np.nan,
        dtype=float,
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

        x = matrix[
            :,
            a
        ].astype(
            np.float64,
            copy=False,
        )

        y = matrix[
            :,
            b
        ].astype(
            np.float64,
            copy=False,
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
                x0
                *
                x0
            )
            .sum(
                axis=0
            )
        )

        sum_y2 = (
            (
                y0
                *
                y0
            )
            .sum(
                axis=0
            )
        )

        sum_xy = (
            (
                x0
                *
                y0
            )
            .sum(
                axis=0
            )
        )

        safe_n = np.where(
            n > 0,
            n.astype(float),
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

        denom = np.sqrt(
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
                denom
            )
            &
            (denom > 0)
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
            denom[
                usable
            ]
        )

        result[
            start:end
        ] = np.clip(
            current,
            -1.0,
            1.0,
        )

    return result


# ============================================================
# 8. Industry Metrics
# ============================================================

def exact_same_pair_share(
    industry_codes,
):

    codes = np.asarray(
        industry_codes,
        dtype=np.int32,
    )

    codes = codes[
        codes >= 0
    ]

    n = len(
        codes
    )

    if n < 2:

        return np.nan

    counts = np.bincount(
        codes
    )

    counts = counts[
        counts > 0
    ]

    same_pairs = np.sum(
        counts
        *
        (
            counts - 1
        )
        //
        2
    )

    total_pairs = (
        n
        *
        (
            n - 1
        )
        //
        2
    )

    return float(
        same_pairs
        /
        total_pairs
    )


def calculate_structure_metrics(
    corr,
    industry_a,
    industry_b,
    exact_same_share,
):

    corr = np.asarray(
        corr,
        dtype=float,
    )

    valid_industry = (
        (industry_a >= 0)
        &
        (industry_b >= 0)
    )

    valid_corr = np.isfinite(
        corr
    )

    same = (
        valid_industry
        &
        (
            industry_a
            ==
            industry_b
        )
    )

    cross = (
        valid_industry
        &
        (
            industry_a
            !=
            industry_b
        )
    )

    same_values = corr[
        same
        &
        valid_corr
    ]

    cross_values = corr[
        cross
        &
        valid_corr
    ]

    all_values = corr[
        valid_industry
        &
        valid_corr
    ]

    same_valid_mask = (
        same[
            valid_industry
            &
            valid_corr
        ]
    )

    if len(
        all_values
    ) == 0:

        return {}

    mean_corr = float(
        np.mean(
            all_values
        )
    )

    same_mean = (
        float(
            np.mean(
                same_values
            )
        )
        if len(
            same_values
        ) > 0
        else
        np.nan
    )

    cross_mean = (
        float(
            np.mean(
                cross_values
            )
        )
        if len(
            cross_values
        ) > 0
        else
        np.nan
    )

    gap = (
        same_mean
        -
        cross_mean
    )

    # --------------------------------------------------------
    # Top 1%
    # --------------------------------------------------------

    threshold = float(
        np.quantile(
            all_values,
            0.99,
        )
    )

    top = (
        all_values
        >=
        threshold
    )

    top_same_share = float(
        same_valid_mask[
            top
        ]
        .mean()
    )

    enrichment = (
        top_same_share
        /
        exact_same_share
        if (
            np.isfinite(
                exact_same_share
            )
            and
            exact_same_share > 0
        )
        else
        np.nan
    )

    return {

        "valid_pair_count":
            int(
                len(
                    all_values
                )
            ),

        "mean_corr":
            mean_corr,

        "same_mean_corr":
            same_mean,

        "cross_mean_corr":
            cross_mean,

        "same_cross_gap":
            gap,

        "top1_threshold":
            threshold,

        "top1_same_share":
            top_same_share,

        "top1_same_enrichment":
            enrichment,
    }


# ============================================================
# 9. Daily PIT EW Factor
# ============================================================

def build_daily_market_factor_components(
    trade_dates,
    master,
    returns,
):

    T = len(
        trade_dates
    )

    market_sum = np.full(
        T,
        np.nan,
        dtype=np.float64,
    )

    market_count = np.zeros(
        T,
        dtype=np.int32,
    )

    for t, date in enumerate(
        trade_dates
    ):

        pit = current_pit_mask(
            master,
            date,
        )

        r = (
            returns[
                t
            ]
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

        market_count[
            t
        ] = n

        if n > 0:

            market_sum[
                t
            ] = float(
                r[
                    valid
                ]
                .sum()
            )

    return (
        market_sum,
        market_count,
    )


# ============================================================
# 10. Leave-One-Out Market Factor
# ============================================================

def build_loo_factor(
    X,
    market_sum_window,
    market_count_window,
):

    finite = np.isfinite(
        X
    )

    own_return = np.where(
        finite,
        X,
        0.0,
    )

    own_count = (
        finite
        .astype(
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

    factor = np.full(
        X.shape,
        np.nan,
        dtype=np.float64,
    )

    usable = (
        denominator > 0
    )

    factor[
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

    return factor


# ============================================================
# 11. Market Residualization
# ============================================================

def residualize(
    X,
    F,
    required_obs,
):

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
            x0
            *
            x0
        )
        .sum(
            axis=0
        )
    )

    sum_f2 = (
        (
            f0
            *
            f0
        )
        .sum(
            axis=0
        )
    )

    sum_fx = (
        (
            f0
            *
            x0
        )
        .sum(
            axis=0
        )
    )

    safe_n = np.where(
        n > 0,
        n.astype(float),
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
        np.isfinite(
            ss_x
        )
        &
        np.isfinite(
            ss_f
        )
        &
        (ss_x > 0)
        &
        (ss_f > 0)
    )

    beta = np.full(
        X.shape[1],
        np.nan,
        dtype=float,
    )

    alpha = np.full(
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
        X
        -
        prediction,
        np.nan,
    )

    return (
        residual,
        beta,
        r2,
    )


# ============================================================
# 12. Top Edge Overlap
# ============================================================

def top1_overlap(
    raw_corr,
    residual_corr,
):

    valid = (
        np.isfinite(
            raw_corr
        )
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

    if n < 100:

        return np.nan

    k = max(
        1,
        int(
            round(
                0.01
                *
                n
            )
        ),
    )

    raw_top = np.argpartition(
        raw,
        -k,
    )[
        -k:
    ]

    res_top = np.argpartition(
        residual,
        -k,
    )[
        -k:
    ]

    overlap = np.intersect1d(
        raw_top,
        res_top,
    )

    return float(
        len(
            overlap
        )
        /
        k
    )


# ============================================================
# 13. Select Dates
# ============================================================

def select_dates(
    analysis_dates,
    representative_dates,
    mode,
):

    if mode == "all":

        return (
            analysis_dates[
                [
                    "analysis_date",
                    "window",
                ]
            ]
            .drop_duplicates()
            .copy()
        )

    if mode == "representative":

        return (
            representative_dates[
                [
                    "analysis_date",
                    "window",
                ]
            ]
            .drop_duplicates()
            .copy()
        )

    raise ValueError(
        f"Unknown date mode: {mode}"
    )


# ============================================================
# 14. Sampling Robustness
#
# BASE only:
#
# return_network
# c_stock = c_pair = 0.90
#
# Compare:
#
# Legacy vs Corrected Uniform
# ============================================================

def run_sampler_robustness(
    trade_dates,
    master,
    returns,
    industry_snapshots,
    dates_to_run,
):

    print()
    print("=" * 76)
    print("Part A - Sampler Robustness")
    print("=" * 76)

    date_to_idx = {
        pd.Timestamp(date): i
        for i, date
        in enumerate(
            trade_dates
        )
    }

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

    rows = []

    for counter, row in enumerate(

        dates_to_run
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .itertuples(
            index=False
        ),

        start=1,
    ):

        window = int(
            row.window
        )

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

        required_days = int(
            math.ceil(
                0.90
                *
                window
            )
        )

        pit = current_pit_mask(
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

        eligible = (
            pit
            &
            (
                valid_count
                >=
                required_days
            )
        )

        eligible_idx = (
            np.flatnonzero(
                eligible
            )
        )

        p = len(
            eligible_idx
        )

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

        industry = (
            industry_snapshots[
                date
            ][
                eligible_idx
            ]
        )

        exact_share = (
            exact_same_pair_share(
                industry
            )
        )

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

        for sampler_name in [
            "legacy",
            "uniform",
        ]:

            rng = np.random.default_rng(
                seed
            )

            if (
                sampler_name
                ==
                "legacy"
            ):

                pairs = (
                    sample_pairs_legacy(
                        p,
                        PAIR_SAMPLE_SIZE,
                        rng,
                    )
                )

            else:

                pairs = (
                    sample_pairs_uniform(
                        p,
                        PAIR_SAMPLE_SIZE,
                        rng,
                    )
                )

            a = pairs[
                :,
                0
            ]

            b = pairs[
                :,
                1
            ]

            corr = (
                calculate_pairwise_corr(
                    matrix=X,

                    stock_a=a,

                    stock_b=b,

                    required_common_days=
                        required_days,
                )
            )

            metrics = (
                calculate_structure_metrics(
                    corr=
                        corr,

                    industry_a=
                        industry[
                            a
                        ],

                    industry_b=
                        industry[
                            b
                        ],

                    exact_same_share=
                        exact_share,
                )
            )

            rows.append(
                {

                    "analysis_date":
                        date,

                    "window":
                        window,

                    "sampler":
                        sampler_name,

                    "eligible_stock_count":
                        p,

                    "sampled_pair_count":
                        len(
                            pairs
                        ),

                    # -----------------------------------------
                    # Direct sampler index diagnostics
                    #
                    # 对真正均匀 unordered pair：
                    # E[min index]/(p-1) 约为 1/3
                    # E[max index]/(p-1) 约为 2/3
                    # -----------------------------------------

                    "mean_first_index_ratio":
                        float(
                            np.mean(a)
                            /
                            max(
                                p - 1,
                                1,
                            )
                        ),

                    "mean_second_index_ratio":
                        float(
                            np.mean(b)
                            /
                            max(
                                p - 1,
                                1,
                            )
                        ),

                    **metrics,
                }
            )

        if (
            counter == 1
            or
            counter % 20 == 0
            or
            counter
            ==
            len(
                dates_to_run
            )
        ):

            print(
                f"[{counter:3d}/"
                f"{len(dates_to_run):3d}] "
                f"W={window} "
                f"{date.date()} "
                f"p={p:,}"
            )

    detail = pd.DataFrame(
        rows
    )

    return detail


# ============================================================
# 15. Sampler Comparison Summary
# ============================================================

def summarize_sampler_robustness(
    detail,
):

    wide = (
        detail.pivot(
            index=[
                "analysis_date",
                "window",
            ],

            columns=
                "sampler",

            values=[
                "mean_corr",
                "same_cross_gap",
                "top1_same_enrichment",
                "mean_first_index_ratio",
                "mean_second_index_ratio",
            ],
        )
    )

    wide.columns = [
        f"{metric}_{sampler}"
        for metric, sampler
        in wide.columns
    ]

    wide = (
        wide
        .reset_index()
    )

    for metric in [
        "mean_corr",
        "same_cross_gap",
        "top1_same_enrichment",
    ]:

        wide[
            f"{metric}_uniform_minus_legacy"
        ] = (
            wide[
                f"{metric}_uniform"
            ]
            -
            wide[
                f"{metric}_legacy"
            ]
        )

        wide[
            f"{metric}_abs_difference"
        ] = np.abs(
            wide[
                f"{metric}_uniform_minus_legacy"
            ]
        )

    summary = (
        wide.groupby(
            "window",
            as_index=False,
        )
        .agg(

            month_count=
                (
                    "analysis_date",
                    "count",
                ),

            legacy_mean_corr=
                (
                    "mean_corr_legacy",
                    "mean",
                ),

            uniform_mean_corr=
                (
                    "mean_corr_uniform",
                    "mean",
                ),

            mean_abs_diff_mean_corr=
                (
                    "mean_corr_abs_difference",
                    "mean",
                ),

            legacy_industry_gap=
                (
                    "same_cross_gap_legacy",
                    "mean",
                ),

            uniform_industry_gap=
                (
                    "same_cross_gap_uniform",
                    "mean",
                ),

            mean_abs_diff_industry_gap=
                (
                    "same_cross_gap_abs_difference",
                    "mean",
                ),

            legacy_top1_enrichment=
                (
                    "top1_same_enrichment_legacy",
                    "mean",
                ),

            uniform_top1_enrichment=
                (
                    "top1_same_enrichment_uniform",
                    "mean",
                ),

            mean_abs_diff_top1_enrichment=
                (
                    "top1_same_enrichment_abs_difference",
                    "mean",
                ),

            legacy_first_index_ratio=
                (
                    "mean_first_index_ratio_legacy",
                    "mean",
                ),

            uniform_first_index_ratio=
                (
                    "mean_first_index_ratio_uniform",
                    "mean",
                ),

            legacy_second_index_ratio=
                (
                    "mean_second_index_ratio_legacy",
                    "mean",
                ),

            uniform_second_index_ratio=
                (
                    "mean_second_index_ratio_uniform",
                    "mean",
                ),
        )
    )

    return (
        wide,
        summary,
    )


# ============================================================
# 16. Specification Robustness
# ============================================================

def run_specification_robustness(
    trade_dates,
    master,
    return_matrices,
    industry_snapshots,
    dates_to_run,
):

    print()
    print("=" * 76)
    print("Part B - Return / Coverage Robustness")
    print("=" * 76)

    date_to_idx = {
        pd.Timestamp(date): i
        for i, date
        in enumerate(
            trade_dates
        )
    }

    # --------------------------------------------------------
    # 为两个 return definition 分别建立：
    #
    # cumulative valid count
    # daily PIT market factor components
    # --------------------------------------------------------

    return_info = {}

    for return_column, returns in (
        return_matrices.items()
    ):

        print()
        print(
            f"Prepare: {return_column}"
        )

        finite = np.isfinite(
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
            finite,
            axis=0,
            dtype=np.int32,
        )

        (
            market_sum,
            market_count,
        ) = (
            build_daily_market_factor_components(
                trade_dates=
                    trade_dates,

                master=
                    master,

                returns=
                    returns,
            )
        )

        return_info[
            return_column
        ] = {

            "returns":
                returns,

            "cumulative_valid":
                cumulative_valid,

            "market_sum":
                market_sum,

            "market_count":
                market_count,
        }

    rows = []

    total_jobs = (
        len(
            ROBUSTNESS_CONFIGS
        )
        *
        len(
            dates_to_run
        )
    )

    job_no = 0

    for specification in (
        ROBUSTNESS_CONFIGS
    ):

        config_id = (
            specification[
                "config_id"
            ]
        )

        return_column = (
            specification[
                "return_column"
            ]
        )

        stock_ratio = float(
            specification[
                "stock_valid_ratio"
            ]
        )

        pair_ratio = float(
            specification[
                "pair_valid_ratio"
            ]
        )

        info = (
            return_info[
                return_column
            ]
        )

        returns = (
            info[
                "returns"
            ]
        )

        cumulative_valid = (
            info[
                "cumulative_valid"
            ]
        )

        market_sum = (
            info[
                "market_sum"
            ]
        )

        market_count = (
            info[
                "market_count"
            ]
        )

        print()
        print(
            "-" * 76
        )

        print(
            f"Specification: {config_id}"
        )

        print(
            f"Return={return_column} | "
            f"Stock c={stock_ratio} | "
            f"Pair c={pair_ratio}"
        )

        print(
            "-" * 76
        )

        for row in (
            dates_to_run
            .sort_values(
                [
                    "window",
                    "analysis_date",
                ]
            )
            .itertuples(
                index=False
            )
        ):

            job_no += 1

            date = pd.Timestamp(
                row.analysis_date
            )

            window = int(
                row.window
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

            required_stock_days = int(
                math.ceil(
                    stock_ratio
                    *
                    window
                )
            )

            required_pair_days = int(
                math.ceil(
                    pair_ratio
                    *
                    window
                )
            )

            pit = current_pit_mask(
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

            eligible = (
                pit
                &
                (
                    valid_count
                    >=
                    required_stock_days
                )
            )

            eligible_idx = (
                np.flatnonzero(
                    eligible
                )
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

            industry = (
                industry_snapshots[
                    date
                ][
                    eligible_idx
                ]
            )

            exact_share = (
                exact_same_pair_share(
                    industry
                )
            )

            # =================================================
            # Market residual
            # =================================================

            F_loo = (
                build_loo_factor(
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

            (
                residual,
                beta,
                r2,
            ) = (
                residualize(
                    X=
                        X,

                    F=
                        F_loo,

                    required_obs=
                        required_stock_days,
                )
            )

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
            )

            rng = np.random.default_rng(
                seed
            )

            pairs = (
                sample_pairs_uniform(
                    n_stocks=
                        p,

                    sample_size=
                        PAIR_SAMPLE_SIZE,

                    rng=
                        rng,
                )
            )

            a = (
                pairs[
                    :,
                    0
                ]
            )

            b = (
                pairs[
                    :,
                    1
                ]
            )

            raw_corr = (
                calculate_pairwise_corr(
                    matrix=
                        X,

                    stock_a=
                        a,

                    stock_b=
                        b,

                    required_common_days=
                        required_pair_days,
                )
            )

            residual_corr = (
                calculate_pairwise_corr(
                    matrix=
                        residual,

                    stock_a=
                        a,

                    stock_b=
                        b,

                    required_common_days=
                        required_pair_days,
                )
            )

            raw_metrics = (
                calculate_structure_metrics(
                    corr=
                        raw_corr,

                    industry_a=
                        industry[
                            a
                        ],

                    industry_b=
                        industry[
                            b
                        ],

                    exact_same_share=
                        exact_share,
                )
            )

            residual_metrics = (
                calculate_structure_metrics(
                    corr=
                        residual_corr,

                    industry_a=
                        industry[
                            a
                        ],

                    industry_b=
                        industry[
                            b
                        ],

                    exact_same_share=
                        exact_share,
                )
            )

            valid_r2 = (
                r2[
                    np.isfinite(
                        r2
                    )
                ]
            )

            overlap = (
                top1_overlap(
                    raw_corr,
                    residual_corr,
                )
            )

            rows.append(
                {

                    "config_id":
                        config_id,

                    "description":
                        specification[
                            "description"
                        ],

                    "return_column":
                        return_column,

                    "stock_valid_ratio":
                        stock_ratio,

                    "pair_valid_ratio":
                        pair_ratio,

                    "analysis_date":
                        date,

                    "window":
                        window,

                    "eligible_stock_count":
                        p,

                    "exact_same_pair_share":
                        exact_share,

                    "market_r2_median":
                        float(
                            np.median(
                                valid_r2
                            )
                        )
                        if len(
                            valid_r2
                        ) > 0
                        else
                        np.nan,

                    "raw_mean_corr":
                        raw_metrics.get(
                            "mean_corr",
                            np.nan,
                        ),

                    "raw_same_cross_gap":
                        raw_metrics.get(
                            "same_cross_gap",
                            np.nan,
                        ),

                    "raw_top1_same_enrichment":
                        raw_metrics.get(
                            "top1_same_enrichment",
                            np.nan,
                        ),

                    "residual_mean_corr":
                        residual_metrics.get(
                            "mean_corr",
                            np.nan,
                        ),

                    "residual_same_cross_gap":
                        residual_metrics.get(
                            "same_cross_gap",
                            np.nan,
                        ),

                    "residual_top1_same_enrichment":
                        residual_metrics.get(
                            "top1_same_enrichment",
                            np.nan,
                        ),

                    "raw_residual_top1_overlap":
                        overlap,

                    "residual_gap_positive":
                        bool(
                            residual_metrics.get(
                                "same_cross_gap",
                                np.nan,
                            )
                            >
                            0
                        ),

                    "residual_enrichment_gt_1":
                        bool(
                            residual_metrics.get(
                                "top1_same_enrichment",
                                np.nan,
                            )
                            >
                            1
                        ),
                }
            )

            if (
                job_no == 1
                or
                job_no % 20 == 0
                or
                job_no == total_jobs
            ):

                print(
                    f"[{job_no:3d}/"
                    f"{total_jobs:3d}] "
                    f"{config_id} | "
                    f"W={window} | "
                    f"{date.date()} | "
                    f"p={p:,}"
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 17. Specification Summary
# ============================================================

def summarize_specifications(
    detail,
):

    summary = (
        detail.groupby(
            [
                "config_id",
                "description",
                "return_column",
                "stock_valid_ratio",
                "pair_valid_ratio",
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

            mean_eligible_stock_count=
                (
                    "eligible_stock_count",
                    "mean",
                ),

            mean_market_r2_median=
                (
                    "market_r2_median",
                    "mean",
                ),

            mean_raw_corr=
                (
                    "raw_mean_corr",
                    "mean",
                ),

            mean_residual_corr=
                (
                    "residual_mean_corr",
                    "mean",
                ),

            mean_raw_industry_gap=
                (
                    "raw_same_cross_gap",
                    "mean",
                ),

            mean_residual_industry_gap=
                (
                    "residual_same_cross_gap",
                    "mean",
                ),

            mean_raw_top1_enrichment=
                (
                    "raw_top1_same_enrichment",
                    "mean",
                ),

            mean_residual_top1_enrichment=
                (
                    "residual_top1_same_enrichment",
                    "mean",
                ),

            mean_raw_residual_top1_overlap=
                (
                    "raw_residual_top1_overlap",
                    "mean",
                ),

            share_residual_gap_positive=
                (
                    "residual_gap_positive",
                    "mean",
                ),

            share_residual_enrichment_gt_1=
                (
                    "residual_enrichment_gt_1",
                    "mean",
                ),
        )
    )

    return summary


# ============================================================
# 18. Compare each robustness specification with BASE
# ============================================================

def build_conclusion_matrix(
    summary,
):

    base = (
        summary[
            summary[
                "config_id"
            ]
            .eq(
                "BASE"
            )
        ][
            [
                "window",
                "mean_residual_corr",
                "mean_residual_industry_gap",
                "mean_residual_top1_enrichment",
            ]
        ]
        .copy()
    )

    base = base.rename(
        columns={

            "mean_residual_corr":
                "base_residual_corr",

            "mean_residual_industry_gap":
                "base_residual_gap",

            "mean_residual_top1_enrichment":
                "base_residual_enrichment",
        }
    )

    result = (
        summary.merge(
            base,
            on="window",
            how="left",
            validate=
                "many_to_one",
        )
    )

    result[
        "delta_residual_mean_corr"
    ] = (
        result[
            "mean_residual_corr"
        ]
        -
        result[
            "base_residual_corr"
        ]
    )

    result[
        "delta_residual_industry_gap"
    ] = (
        result[
            "mean_residual_industry_gap"
        ]
        -
        result[
            "base_residual_gap"
        ]
    )

    result[
        "residual_gap_ratio_to_base"
    ] = (
        result[
            "mean_residual_industry_gap"
        ]
        /
        result[
            "base_residual_gap"
        ]
    )

    result[
        "delta_residual_top1_enrichment"
    ] = (
        result[
            "mean_residual_top1_enrichment"
        ]
        -
        result[
            "base_residual_enrichment"
        ]
    )

    # --------------------------------------------------------
    # 不做任意综合分数。
    #
    # 这里只记录核心结论是否保留。
    # --------------------------------------------------------

    result[
        "core_residual_gap_positive"
    ] = (
        result[
            "mean_residual_industry_gap"
        ]
        >
        0
    )

    result[
        "core_residual_enrichment_gt_1"
    ] = (
        result[
            "mean_residual_top1_enrichment"
        ]
        >
        1
    )

    result[
        "core_monthly_gap_consistency"
    ] = (
        result[
            "share_residual_gap_positive"
        ]
        >=
        0.90
    )

    result[
        "core_monthly_enrichment_consistency"
    ] = (
        result[
            "share_residual_enrichment_gt_1"
        ]
        >=
        0.90
    )

    result[
        "core_conclusion_preserved"
    ] = (
        result[
            [
                "core_residual_gap_positive",
                "core_residual_enrichment_gt_1",
                "core_monthly_gap_consistency",
                "core_monthly_enrichment_consistency",
            ]
        ]
        .all(
            axis=1
        )
    )

    return result


# ============================================================
# 19. Figures
# ============================================================

def plot_robustness(
    sampler_summary,
    specification_summary,
):

    # --------------------------------------------------------
    # Figure 1
    # Legacy vs Uniform industry gap
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    x = np.arange(
        len(
            sampler_summary
        )
    )

    width = 0.35

    ax.bar(
        x - width / 2,
        sampler_summary[
            "legacy_industry_gap"
        ],
        width,
        label="Legacy sampler",
    )

    ax.bar(
        x + width / 2,
        sampler_summary[
            "uniform_industry_gap"
        ],
        width,
        label="Uniform sampler",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        sampler_summary[
            "window"
        ].astype(str)
    )

    ax.set_xlabel(
        "Window"
    )

    ax.set_ylabel(
        "Same-Cross Industry Gap"
    )

    ax.set_title(
        "Sampler Robustness of Industry Gap"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "sampler_robustness_industry_gap.png",
        dpi=180,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Figure 2
    # Residual gap across specifications
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for config_id, g in (
        specification_summary.groupby(
            "config_id"
        )
    ):

        g = g.sort_values(
            "window"
        )

        ax.plot(
            g[
                "window"
            ],
            g[
                "mean_residual_industry_gap"
            ],
            marker="o",
            label=config_id,
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Window"
    )

    ax.set_ylabel(
        "Residual Same-Cross Gap"
    )

    ax.set_title(
        "Residual Industry Gap across Robustness Specifications"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "specification_robustness_residual_gap.png",
        dpi=180,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Figure 3
    # Residual Top1 enrichment
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for config_id, g in (
        specification_summary.groupby(
            "config_id"
        )
    ):

        g = g.sort_values(
            "window"
        )

        ax.plot(
            g[
                "window"
            ],
            g[
                "mean_residual_top1_enrichment"
            ],
            marker="o",
            label=config_id,
        )

    ax.axhline(
        1,
        linewidth=1,
    )

    ax.set_xlabel(
        "Window"
    )

    ax.set_ylabel(
        "Residual Top-1% Same-Industry Enrichment"
    )

    ax.set_title(
        "Industry Enrichment across Robustness Specifications"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "specification_robustness_top1_enrichment.png",
        dpi=180,
    )

    plt.close(
        fig
    )


# ============================================================
# 20. Metadata
# ============================================================

def save_metadata(
    windows,
):

    metadata = {

        "analysis":
            (
                "Supplementary Market-Industry "
                "Decomposition Robustness"
            ),

        "windows":
            windows,

        "main_industry_level":
            MAIN_INDUSTRY_LEVEL,

        "pair_sample_size":
            PAIR_SAMPLE_SIZE,

        "sampler_robustness_date_mode":
            SAMPLER_DATE_MODE,

        "specification_robustness_date_mode":
            SPEC_DATE_MODE,

        "robustness_configurations":
            ROBUSTNESS_CONFIGS,

        "sampler_test":
            (
                "Legacy np.unique + first-target truncation "
                "versus direct uniform sampling without "
                "replacement from all unordered pair ranks."
            ),

        "main_quantities":
            [
                "mean raw correlation",
                "raw/residual Same-Cross industry gap",
                "raw/residual Top-1% industry enrichment",
                "raw-residual Top-1% edge overlap",
            ],

        "interpretation":
            (
                "The robustness exercise is designed to test "
                "whether the main Market-Industry decomposition "
                "conclusions depend on pair-sampling implementation, "
                "return definition, or valid-observation thresholds."
            ),

        "important_note":
            (
                "The 60-, 120-, and 252-day windows are retained "
                "because window-length robustness has already been "
                "examined throughout Steps 2-7."
            ),
    }

    with open(
        OUTPUT_DIR
        / "robustness_metadata.json",
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
# 21. Main
# ============================================================

def main():

    print("=" * 76)
    print(
        "Supplementary Robustness"
    )
    print(
        "Market-Industry Decomposition"
    )
    print("=" * 76)

    (
        config,
        windows,
    ) = load_config()

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

    print()
    print(
        "[1] Build matrices"
    )

    (
        return_matrices,
        industry_snapshots,
        industry_codebook,
    ) = (
        build_data_matrices(
            trade_dates=
                trade_dates,

            master=
                master,

            analysis_dates=
                analysis_dates,
        )
    )

    with open(
        OUTPUT_DIR
        / "industry_codebook.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            industry_codebook,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Part A
    # ========================================================

    sampler_dates = (
        select_dates(
            analysis_dates=
                analysis_dates,

            representative_dates=
                representative_dates,

            mode=
                SAMPLER_DATE_MODE,
        )
    )

    sampler_detail = (
        run_sampler_robustness(
            trade_dates=
                trade_dates,

            master=
                master,

            returns=
                return_matrices[
                    "return_network"
                ],

            industry_snapshots=
                industry_snapshots,

            dates_to_run=
                sampler_dates,
        )
    )

    (
        sampler_comparison,
        sampler_summary,
    ) = (
        summarize_sampler_robustness(
            sampler_detail
        )
    )

    sampler_detail.to_csv(
        OUTPUT_DIR
        / "sampler_robustness_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sampler_comparison.to_csv(
        OUTPUT_DIR
        / "sampler_robustness_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sampler_summary.to_csv(
        OUTPUT_DIR
        / "sampler_robustness_window_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Part B
    # ========================================================

    spec_dates = (
        select_dates(
            analysis_dates=
                analysis_dates,

            representative_dates=
                representative_dates,

            mode=
                SPEC_DATE_MODE,
        )
    )

    specification_detail = (
        run_specification_robustness(
            trade_dates=
                trade_dates,

            master=
                master,

            return_matrices=
                return_matrices,

            industry_snapshots=
                industry_snapshots,

            dates_to_run=
                spec_dates,
        )
    )

    specification_summary = (
        summarize_specifications(
            specification_detail
        )
    )

    conclusion_matrix = (
        build_conclusion_matrix(
            specification_summary
        )
    )

    specification_detail.to_csv(
        OUTPUT_DIR
        / "specification_robustness_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    specification_summary.to_csv(
        OUTPUT_DIR
        / "specification_robustness_window_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    conclusion_matrix.to_csv(
        OUTPUT_DIR
        / "robustness_core_conclusion_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Figures
    # ========================================================

    print()
    print(
        "[3] Figures"
    )

    plot_robustness(
        sampler_summary=
            sampler_summary,

        specification_summary=
            specification_summary,
    )

    save_metadata(
        windows=
            windows
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 76)
    print(
        "Sampler Robustness"
    )
    print("=" * 76)

    print(
        sampler_summary[
            [
                "window",
                "legacy_mean_corr",
                "uniform_mean_corr",
                "legacy_industry_gap",
                "uniform_industry_gap",
                "legacy_top1_enrichment",
                "uniform_top1_enrichment",
                "legacy_first_index_ratio",
                "uniform_first_index_ratio",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 76)
    print(
        "Specification Robustness"
    )
    print("=" * 76)

    print(
        specification_summary[
            [
                "config_id",
                "window",
                "mean_raw_corr",
                "mean_residual_corr",
                "mean_residual_industry_gap",
                "mean_residual_top1_enrichment",
                "share_residual_gap_positive",
                "share_residual_enrichment_gt_1",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 76)
    print(
        "Core Conclusion Matrix"
    )
    print("=" * 76)

    print(
        conclusion_matrix[
            [
                "config_id",
                "window",
                "residual_gap_ratio_to_base",
                "core_residual_gap_positive",
                "core_residual_enrichment_gt_1",
                "core_conclusion_preserved",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()