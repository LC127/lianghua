from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib.pyplot as plt


# ============================================================
# 0. 路径
# ============================================================

# ------------------------------------------------------------
# Step 1
# ------------------------------------------------------------

STAGE1_DIR = Path(
    r"D:\output\M1_day1\01_stage1_final"
)

STOCK_MASTER_PATH = (
    STAGE1_DIR
    / "stock_master.parquet"
)

TRADE_CALENDAR_PATH = (
    STAGE1_DIR
    / "trade_calendar_used.csv"
)


# ------------------------------------------------------------
# Step 3
# ------------------------------------------------------------

STAGE3_DIR = Path(
    r"D:\output\M1_day1\03_stage3_return_validation"
)

RETURN_PANEL_PATH = (
    STAGE3_DIR
    / "daily_return_panel.parquet"
)


# ------------------------------------------------------------
# Step 4
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"D:\output\M1_day1\04_step4_network_feasibility"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. 参数
# ============================================================

START_DATE = pd.Timestamp(
    "2015-01-05"
)

END_DATE = pd.Timestamp(
    "2026-09-03"
)

EXPECTED_TRADE_DAY_COUNT = 2837


# ============================================================
# 2. 网络窗口
# ============================================================

WINDOWS = [
    60,
    120,
    252,
]


# ============================================================
# 3. 最低有效收益比例
# ============================================================

MIN_VALID_RATIOS = [
    0.80,
    0.90,
    0.95,
]


# ============================================================
# 4. 使用哪一种收益
#
# 主分析：
# return_network
#
# 稳健性分析以后可以改成：
# return_last_trade_log
# ============================================================

RETURN_COLUMN = "return_network"


# ============================================================
# 5. Pairwise overlap 辅助诊断
#
# 这里只对最新日期、c=0.90 进行随机抽样，
# 避免真正构造 5000 x 5000 x 多日期的巨大对象。
# ============================================================

RUN_PAIRWISE_DIAGNOSTIC = True

PAIRWISE_VALID_RATIO = 0.90

PAIR_SAMPLE_SIZE = 50_000

PAIR_SAMPLE_BATCH_SIZE = 5_000

RANDOM_SEED = 20260908


# ============================================================
# 6. Parquet 读取批大小
# ============================================================

PARQUET_BATCH_SIZE = 300_000


# ============================================================
# 7. 工具函数：交易日
# ============================================================

def load_trade_dates() -> pd.DatetimeIndex:

    if not TRADE_CALENDAR_PATH.exists():

        raise FileNotFoundError(
            f"找不到交易日历：{TRADE_CALENDAR_PATH}"
        )

    calendar = pd.read_csv(
        TRADE_CALENDAR_PATH
    )

    if "trade_date" not in calendar.columns:

        raise ValueError(
            "trade_calendar_used.csv 缺少 trade_date。"
        )

    dates = pd.to_datetime(
        calendar["trade_date"],
        errors="coerce",
    )

    dates = (
        dates
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    dates = dates[
        (dates >= START_DATE)
        &
        (dates <= END_DATE)
    ]

    if len(dates) != EXPECTED_TRADE_DAY_COUNT:

        raise ValueError(
            f"交易日数量={len(dates)}, "
            f"预期={EXPECTED_TRADE_DAY_COUNT}"
        )

    return pd.DatetimeIndex(
        dates
    )


# ============================================================
# 8. 读取 Stock Master
# ============================================================

def load_stock_master() -> pd.DataFrame:

    if not STOCK_MASTER_PATH.exists():

        raise FileNotFoundError(
            f"找不到：{STOCK_MASTER_PATH}"
        )

    master = pd.read_parquet(
        STOCK_MASTER_PATH
    )

    master = master.copy()

    master["security_id"] = (
        master["security_id"]
        .astype("string")
        .str.strip()
    )

    master = (
        master
        .drop_duplicates(
            subset=["security_id"]
        )
        .sort_values(
            "security_id"
        )
        .reset_index(
            drop=True
        )
    )

    return master


# ============================================================
# 9. 从 Daily Return Panel 构造两个矩阵
#
# present[t, i]：
#   股票 i 在日期 t 是否属于研究面板
#
# valid[t, i]：
#   return_network 是否非缺失
#
# 注意：
# 5553 x 2837 只有约 1575 万个 bool，
# 因此可以安全放在内存中。
# ============================================================

def build_presence_and_valid_matrix(
    trade_dates: pd.DatetimeIndex,
    stock_master: pd.DataFrame,
):

    if not RETURN_PANEL_PATH.exists():

        raise FileNotFoundError(
            f"找不到：{RETURN_PANEL_PATH}"
        )

    security_ids = (
        stock_master[
            "security_id"
        ]
        .astype("string")
        .tolist()
    )

    n_dates = len(
        trade_dates
    )

    n_stocks = len(
        security_ids
    )

    print()
    print("=" * 72)
    print("构造 Presence / Return Validity Matrix")
    print("=" * 72)

    print(
        f"交易日：{n_dates:,}"
    )

    print(
        f"历史证券主体：{n_stocks:,}"
    )

    present = np.zeros(
        (
            n_dates,
            n_stocks,
        ),
        dtype=bool,
    )

    valid = np.zeros(
        (
            n_dates,
            n_stocks,
        ),
        dtype=bool,
    )

    # --------------------------------------------------------
    # 索引映射
    # --------------------------------------------------------

    security_to_index = {

        sid: i

        for i, sid
        in enumerate(
            security_ids
        )
    }

    date_to_index = {

        pd.Timestamp(date):
            i

        for i, date
        in enumerate(
            trade_dates
        )
    }

    # --------------------------------------------------------
    # 流式读取 Parquet
    # --------------------------------------------------------

    parquet_file = pq.ParquetFile(
        RETURN_PANEL_PATH
    )

    required_columns = [
        "trade_date",
        "security_id",
        RETURN_COLUMN,
    ]

    total_rows = 0
    matched_rows = 0
    valid_return_rows = 0

    for batch_no, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=PARQUET_BATCH_SIZE,
            columns=required_columns,
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
                date_to_index
            )
        )

        col_idx = (
            df["security_id"]
            .map(
                security_to_index
            )
        )

        usable = (
            row_idx.notna()
            &
            col_idx.notna()
        )

        if usable.any():

            rows = (
                row_idx.loc[
                    usable
                ]
                .astype(
                    np.int32
                )
                .to_numpy()
            )

            cols = (
                col_idx.loc[
                    usable
                ]
                .astype(
                    np.int32
                )
                .to_numpy()
            )

            present[
                rows,
                cols,
            ] = True

            returns = pd.to_numeric(
                df.loc[
                    usable,
                    RETURN_COLUMN,
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            return_valid = np.isfinite(
                returns
            )

            if return_valid.any():

                valid[
                    rows[
                        return_valid
                    ],
                    cols[
                        return_valid
                    ],
                ] = True

            matched_rows += len(
                rows
            )

            valid_return_rows += int(
                return_valid.sum()
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
                f"累计读取 {total_rows:,} 行"
            )

    print()
    print(
        f"Panel 总行数："
        f"{total_rows:,}"
    )

    print(
        f"成功映射记录："
        f"{matched_rows:,}"
    )

    print(
        f"{RETURN_COLUMN} 非缺失："
        f"{valid_return_rows:,}"
    )

    return (
        present,
        valid,
        security_ids,
    )


# ============================================================
# 10. Rolling Valid Counts
#
# 对每只股票计算：
#
# n_valid(i,t,W)
# =
# sum_{s=t-W+1}^t
# 1{return_network != NA}
# ============================================================

def rolling_valid_counts(
    valid: np.ndarray,
    window: int,
) -> np.ndarray:

    n_dates = valid.shape[0]

    if window > n_dates:

        raise ValueError(
            f"window={window} 大于总交易日数量。"
        )

    # --------------------------------------------------------
    # cumulative sum
    #
    # 最大值只有 2837，
    # int32 足够。
    # --------------------------------------------------------

    cs = np.cumsum(
        valid,
        axis=0,
        dtype=np.int32,
    )

    # 第一个完整窗口：
    # cs[W-1]
    #
    # 后续：
    # cs[t] - cs[t-W]
    # --------------------------------------------------------

    counts = (
        cs[
            window - 1:
        ]
        .copy()
    )

    if counts.shape[0] > 1:

        counts[
            1:
        ] -= (
            cs[
                : n_dates - window
            ]
        )

    return counts


# ============================================================
# 11. Jaccard
# ============================================================

def binary_jaccard(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    union = (
        a
        |
        b
    ).sum()

    if union == 0:

        return np.nan

    intersection = (
        a
        &
        b
    ).sum()

    return (
        intersection
        /
        union
    )


# ============================================================
# 12. 主网络可行性诊断
# ============================================================

def run_network_feasibility(
    trade_dates,
    stock_master,
    present,
    valid,
):

    all_results = []

    coverage_distribution_rows = []

    stock_summary_parts = []

    latest_stock_parts = []

    # 保存最新窗口，给 pairwise diagnostic 使用
    latest_window_cache = {}

    for window in WINDOWS:

        print()
        print("=" * 72)
        print(
            f"分析滚动窗口 W = {window}"
        )
        print("=" * 72)

        # ----------------------------------------------------
        # rolling valid observations
        # ----------------------------------------------------

        counts = (
            rolling_valid_counts(
                valid,
                window,
            )
        )

        # counts shape：
        # T-W+1 x P

        current_days_by_stock = (
            present[
                window - 1:
            ]
            .sum(
                axis=0
            )
        )

        eligible_day_counts = {

            c:
                np.zeros(
                    present.shape[1],
                    dtype=np.int32,
                )

            for c in MIN_VALID_RATIOS
        }

        previous_eligible = {

            c: None

            for c in MIN_VALID_RATIOS
        }

        # ----------------------------------------------------
        # 逐个窗口结束日期
        # ----------------------------------------------------

        for local_idx in range(
            counts.shape[0]
        ):

            global_idx = (
                window
                -
                1
                +
                local_idx
            )

            date = (
                trade_dates[
                    global_idx
                ]
            )

            count_t = (
                counts[
                    local_idx
                ]
            )

            # ------------------------------------------------
            # 当前时点仍在研究 Panel 的股票
            # ------------------------------------------------

            current_mask = (
                present[
                    global_idx
                ]
            )

            current_universe_count = int(
                current_mask.sum()
            )

            # ------------------------------------------------
            # 当前 Universe 内的 window coverage
            # ------------------------------------------------

            if current_universe_count > 0:

                current_ratios = (
                    count_t[
                        current_mask
                    ]
                    /
                    window
                )

                quantiles = np.quantile(
                    current_ratios,
                    [
                        0.00,
                        0.10,
                        0.25,
                        0.50,
                        0.75,
                        0.90,
                        0.95,
                        1.00,
                    ],
                )

            else:

                quantiles = (
                    np.full(
                        8,
                        np.nan,
                    )
                )

            coverage_distribution_rows.append(
                {

                    "trade_date":
                        date,

                    "window":
                        window,

                    "current_universe_count":
                        current_universe_count,

                    "valid_ratio_min":
                        quantiles[0],

                    "valid_ratio_q10":
                        quantiles[1],

                    "valid_ratio_q25":
                        quantiles[2],

                    "valid_ratio_median":
                        quantiles[3],

                    "valid_ratio_q75":
                        quantiles[4],

                    "valid_ratio_q90":
                        quantiles[5],

                    "valid_ratio_q95":
                        quantiles[6],

                    "valid_ratio_max":
                        quantiles[7],
                }
            )

            # =================================================
            # 不同 c
            # =================================================

            for c in MIN_VALID_RATIOS:

                required_days = int(
                    math.ceil(
                        c
                        *
                        window
                    )
                )

                eligible = (

                    current_mask

                    &

                    (
                        count_t
                        >=
                        required_days
                    )
                )

                p = int(
                    eligible.sum()
                )

                eligible_day_counts[
                    c
                ] += (
                    eligible.astype(
                        np.int32
                    )
                )

                eligible_ratio = (
                    p
                    /
                    current_universe_count
                    if current_universe_count > 0
                    else np.nan
                )

                p_over_w = (
                    p
                    /
                    window
                )

                # ------------------------------------------------
                # 完整相关矩阵包含多少 pair
                # ------------------------------------------------

                pair_count = (
                    p
                    *
                    (
                        p - 1
                    )
                    //
                    2
                )

                # ------------------------------------------------
                # 一个 dense p x p float64 matrix
                # 仅计算理论存储大小
                # ------------------------------------------------

                dense_matrix_mb = (
                    p
                    *
                    p
                    *
                    8
                    /
                    1024 ** 2
                )

                # ------------------------------------------------
                # 如果完整数据且经过中心化，
                #
                # sample covariance rank <= W - 1
                #
                # 因此 p >= W 时一定维数奇异。
                # ------------------------------------------------

                dimensionally_singular = (
                    p
                    >=
                    window
                )

                rank_upper_bound = min(
                    p,
                    window - 1,
                )

                minimum_rank_deficiency = max(
                    p
                    -
                    (
                        window - 1
                    ),
                    0,
                )

                # ------------------------------------------------
                # Universe stability
                # ------------------------------------------------

                previous = (
                    previous_eligible[
                        c
                    ]
                )

                if previous is None:

                    jaccard = np.nan
                    entered = np.nan
                    exited = np.nan

                else:

                    jaccard = (
                        binary_jaccard(
                            eligible,
                            previous,
                        )
                    )

                    entered = int(
                        (
                            eligible
                            &
                            ~previous
                        ).sum()
                    )

                    exited = int(
                        (
                            previous
                            &
                            ~eligible
                        ).sum()
                    )

                previous_eligible[
                    c
                ] = (
                    eligible.copy()
                )

                all_results.append(
                    {

                        "trade_date":
                            date,

                        "window":
                            window,

                        "min_valid_ratio":
                            c,

                        "required_valid_days":
                            required_days,

                        "current_universe_count":
                            current_universe_count,

                        "eligible_stock_count":
                            p,

                        "eligible_stock_ratio":
                            eligible_ratio,

                        "p_over_w":
                            p_over_w,

                        "possible_pair_count":
                            pair_count,

                        "dense_matrix_memory_mb_float64":
                            dense_matrix_mb,

                        "covariance_dimensionally_singular":
                            dimensionally_singular,

                        "covariance_rank_upper_bound":
                            rank_upper_bound,

                        "minimum_rank_deficiency":
                            minimum_rank_deficiency,

                        "eligible_jaccard_vs_previous_day":
                            jaccard,

                        "entered_count":
                            entered,

                        "exited_count":
                            exited,
                    }
                )

        # ====================================================
        # Stock-level feasibility
        # ====================================================

        meta = (
            stock_master[
                [
                    "security_id",
                    "stock_code",
                    "stock_name",
                    "exchange",
                ]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        for c in MIN_VALID_RATIOS:

            eligible_days = (
                eligible_day_counts[
                    c
                ]
            )

            ratio = np.divide(

                eligible_days,

                current_days_by_stock,

                out=np.full(
                    len(
                        eligible_days
                    ),
                    np.nan,
                    dtype=float,
                ),

                where=(
                    current_days_by_stock
                    >
                    0
                ),
            )

            stock_result = (
                meta.copy()
            )

            stock_result[
                "window"
            ] = window

            stock_result[
                "min_valid_ratio"
            ] = c

            stock_result[
                "current_universe_days"
            ] = (
                current_days_by_stock
            )

            stock_result[
                "eligible_days"
            ] = eligible_days

            stock_result[
                "eligible_day_ratio"
            ] = ratio

            stock_summary_parts.append(
                stock_result
            )

        # ====================================================
        # 最新日期逐股 Snapshot
        # ====================================================

        latest_counts = (
            counts[
                -1
            ]
        )

        latest_current = (
            present[
                -1
            ]
        )

        latest_meta = (
            meta.copy()
        )

        latest_meta[
            "trade_date"
        ] = trade_dates[
            -1
        ]

        latest_meta[
            "window"
        ] = window

        latest_meta[
            "is_current_universe"
        ] = latest_current

        latest_meta[
            "valid_return_count"
        ] = latest_counts

        latest_meta[
            "valid_return_ratio"
        ] = (
            latest_counts
            /
            window
        )

        for c in MIN_VALID_RATIOS:

            required_days = int(
                math.ceil(
                    c
                    *
                    window
                )
            )

            latest_meta[
                f"eligible_c{int(c * 100)}"
            ] = (

                latest_current

                &

                (
                    latest_counts
                    >=
                    required_days
                )
            )

        latest_stock_parts.append(
            latest_meta
        )

        # Pairwise 最新窗口缓存
        latest_window_cache[
            window
        ] = {

            "latest_counts":
                latest_counts.copy(),

            "latest_current":
                latest_current.copy(),
        }

        del counts

    # ========================================================
    # 整理结果
    # ========================================================

    feasibility_df = pd.DataFrame(
        all_results
    )

    coverage_distribution_df = pd.DataFrame(
        coverage_distribution_rows
    )

    stock_feasibility_df = pd.concat(
        stock_summary_parts,
        ignore_index=True,
    )

    latest_stock_df = pd.concat(
        latest_stock_parts,
        ignore_index=True,
    )

    return (
        feasibility_df,
        coverage_distribution_df,
        stock_feasibility_df,
        latest_stock_df,
        latest_window_cache,
    )


# ============================================================
# 13. 汇总 W × c
# ============================================================

def summarize_feasibility(
    feasibility_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        window,
        c,
    ), group in (
        feasibility_df.groupby(
            [
                "window",
                "min_valid_ratio",
            ]
        )
    ):

        p = (
            group[
                "eligible_stock_count"
            ]
        )

        p_over_w = (
            group[
                "p_over_w"
            ]
        )

        jaccard = (
            group[
                "eligible_jaccard_vs_previous_day"
            ]
            .dropna()
        )

        rows.append(
            {

                "window":
                    window,

                "min_valid_ratio":
                    c,

                "required_valid_days":
                    int(
                        group[
                            "required_valid_days"
                        ].iloc[0]
                    ),

                "evaluation_day_count":
                    len(
                        group
                    ),

                "eligible_stock_count_min":
                    int(
                        p.min()
                    ),

                "eligible_stock_count_median":
                    float(
                        p.median()
                    ),

                "eligible_stock_count_mean":
                    float(
                        p.mean()
                    ),

                "eligible_stock_count_max":
                    int(
                        p.max()
                    ),

                "eligible_ratio_median":
                    float(
                        group[
                            "eligible_stock_ratio"
                        ].median()
                    ),

                "p_over_w_min":
                    float(
                        p_over_w.min()
                    ),

                "p_over_w_median":
                    float(
                        p_over_w.median()
                    ),

                "p_over_w_mean":
                    float(
                        p_over_w.mean()
                    ),

                "p_over_w_max":
                    float(
                        p_over_w.max()
                    ),

                "share_days_p_ge_w":
                    float(
                        (
                            p
                            >=
                            window
                        ).mean()
                    ),

                "share_days_p_ge_2w":
                    float(
                        (
                            p
                            >=
                            2
                            *
                            window
                        ).mean()
                    ),

                "share_days_p_ge_5w":
                    float(
                        (
                            p
                            >=
                            5
                            *
                            window
                        ).mean()
                    ),

                "share_days_p_ge_10w":
                    float(
                        (
                            p
                            >=
                            10
                            *
                            window
                        ).mean()
                    ),

                "median_possible_pair_count":
                    float(
                        group[
                            "possible_pair_count"
                        ].median()
                    ),

                "median_dense_matrix_memory_mb":
                    float(
                        group[
                            "dense_matrix_memory_mb_float64"
                        ].median()
                    ),

                "median_daily_universe_jaccard":
                    (
                        float(
                            jaccard.median()
                        )
                        if not jaccard.empty
                        else np.nan
                    ),

                "mean_daily_entered_count":
                    float(
                        group[
                            "entered_count"
                        ].mean()
                    ),

                "mean_daily_exited_count":
                    float(
                        group[
                            "exited_count"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 14. 最新日期 Snapshot
# ============================================================

def build_latest_snapshot(
    feasibility_df: pd.DataFrame,
) -> pd.DataFrame:

    latest_date = (
        feasibility_df[
            "trade_date"
        ].max()
    )

    result = (
        feasibility_df[
            feasibility_df[
                "trade_date"
            ].eq(
                latest_date
            )
        ]
        .copy()
    )

    return (
        result
        .sort_values(
            [
                "window",
                "min_valid_ratio",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 15. Pairwise overlap 抽样诊断
#
# 单股满足 >=90% 并不意味着任意两个股票
# 都有 >=90% 的共同有效收益。
#
# 因此对最新日期抽 50,000 对股票，
# 查看共同有效收益比例。
# ============================================================

def run_latest_pairwise_diagnostic(
    valid: np.ndarray,
    latest_window_cache,
):

    if not RUN_PAIRWISE_DIAGNOSTIC:

        return pd.DataFrame()

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rows = []

    n_dates = (
        valid.shape[0]
    )

    for window in WINDOWS:

        if window > n_dates:

            continue

        cache = (
            latest_window_cache[
                window
            ]
        )

        latest_counts = (
            cache[
                "latest_counts"
            ]
        )

        latest_current = (
            cache[
                "latest_current"
            ]
        )

        required_days = int(
            math.ceil(
                PAIRWISE_VALID_RATIO
                *
                window
            )
        )

        eligible = (

            latest_current

            &

            (
                latest_counts
                >=
                required_days
            )
        )

        eligible_indices = np.flatnonzero(
            eligible
        )

        p = len(
            eligible_indices
        )

        if p < 2:

            continue

        pair_draw_count = (
            PAIR_SAMPLE_SIZE
        )

        # ----------------------------------------------------
        # 随机 pair draws
        #
        # 这里允许极少量重复 pair，
        # 对分布诊断没有实质影响。
        # ----------------------------------------------------

        common_counts = []

        remaining = (
            pair_draw_count
        )

        window_valid = (
            valid[
                -window:
            ]
        )

        while remaining > 0:

            batch_size = min(
                PAIR_SAMPLE_BATCH_SIZE,
                remaining,
            )

            a = rng.integers(
                0,
                p,
                size=batch_size,
            )

            b = rng.integers(
                0,
                p,
                size=batch_size,
            )

            # 防止自己和自己配对
            same = (
                a == b
            )

            while same.any():

                b[
                    same
                ] = rng.integers(
                    0,
                    p,
                    size=int(
                        same.sum()
                    ),
                )

                same = (
                    a == b
                )

            stock_a = (
                eligible_indices[
                    a
                ]
            )

            stock_b = (
                eligible_indices[
                    b
                ]
            )

            common = (

                window_valid[
                    :,
                    stock_a,
                ]

                &

                window_valid[
                    :,
                    stock_b,
                ]
            ).sum(
                axis=0
            )

            common_counts.append(
                common.astype(
                    np.int16
                )
            )

            remaining -= (
                batch_size
            )

        common_counts = np.concatenate(
            common_counts
        )

        common_ratio = (
            common_counts
            /
            window
        )

        q = np.quantile(
            common_ratio,
            [
                0.00,
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                1.00,
            ],
        )

        rows.append(
            {

                "trade_date":
                    END_DATE,

                "window":
                    window,

                "individual_min_valid_ratio":
                    PAIRWISE_VALID_RATIO,

                "eligible_stock_count":
                    p,

                "sample_pair_draw_count":
                    len(
                        common_ratio
                    ),

                "pair_common_ratio_mean":
                    float(
                        common_ratio.mean()
                    ),

                "pair_common_ratio_min":
                    q[0],

                "pair_common_ratio_q01":
                    q[1],

                "pair_common_ratio_q05":
                    q[2],

                "pair_common_ratio_q10":
                    q[3],

                "pair_common_ratio_q25":
                    q[4],

                "pair_common_ratio_median":
                    q[5],

                "pair_common_ratio_q75":
                    q[6],

                "pair_common_ratio_q90":
                    q[7],

                "pair_common_ratio_q95":
                    q[8],

                "pair_common_ratio_q99":
                    q[9],

                "pair_common_ratio_max":
                    q[10],

                "share_pairs_common_ge_80pct":
                    float(
                        (
                            common_ratio
                            >=
                            0.80
                        ).mean()
                    ),

                "share_pairs_common_ge_90pct":
                    float(
                        (
                            common_ratio
                            >=
                            0.90
                        ).mean()
                    ),

                "share_pairs_common_ge_95pct":
                    float(
                        (
                            common_ratio
                            >=
                            0.95
                        ).mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 16. 画图
# ============================================================

def plot_feasibility(
    feasibility_df,
):

    for window in WINDOWS:

        subset = (
            feasibility_df[
                feasibility_df[
                    "window"
                ]
                .eq(
                    window
                )
            ]
        )

        # ----------------------------------------------------
        # 图 1：
        # Eligible Stocks
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(
                12,
                6,
            )
        )

        for c in MIN_VALID_RATIOS:

            part = (
                subset[
                    subset[
                        "min_valid_ratio"
                    ]
                    .eq(
                        c
                    )
                ]
            )

            ax.plot(
                part[
                    "trade_date"
                ],
                part[
                    "eligible_stock_count"
                ],
                label=f"c={c:.2f}",
            )

        ax.set_title(
            f"Eligible Stocks under Rolling Window W={window}"
        )

        ax.set_xlabel(
            "Date"
        )

        ax.set_ylabel(
            "Eligible Stock Count"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / f"eligible_stock_count_W{window}.png",
            dpi=180,
        )

        plt.close(
            fig
        )

        # ----------------------------------------------------
        # 图 2：
        # p / W
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(
                12,
                6,
            )
        )

        for c in MIN_VALID_RATIOS:

            part = (
                subset[
                    subset[
                        "min_valid_ratio"
                    ]
                    .eq(
                        c
                    )
                ]
            )

            ax.plot(
                part[
                    "trade_date"
                ],
                part[
                    "p_over_w"
                ],
                label=f"c={c:.2f}",
            )

        ax.axhline(
            y=1.0,
            linestyle="--",
            linewidth=1,
            label="p/W = 1",
        )

        ax.set_title(
            f"Dimension-to-Window Ratio p/W, W={window}"
        )

        ax.set_xlabel(
            "Date"
        )

        ax.set_ylabel(
            "p / W"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / f"p_over_w_W{window}.png",
            dpi=180,
        )

        plt.close(
            fig
        )


# ============================================================
# 17. 自动生成诊断结论表
# ============================================================

def build_dimension_diagnostic_table(
    latest_snapshot,
):

    result = (
        latest_snapshot[
            [
                "trade_date",
                "window",
                "min_valid_ratio",
                "required_valid_days",
                "current_universe_count",
                "eligible_stock_count",
                "eligible_stock_ratio",
                "p_over_w",
                "possible_pair_count",
                "dense_matrix_memory_mb_float64",
                "covariance_dimensionally_singular",
                "minimum_rank_deficiency",
            ]
        ]
        .copy()
    )

    # 这里不用 LOW/HIGH 等主观标签，
    # 只报告客观维度关系。

    return result


# ============================================================
# 18. 主程序
# ============================================================

def main():

    print("=" * 72)
    print("M1 Day 1 - Step 4")
    print("Full-Market Network Feasibility Diagnostics")
    print("=" * 72)

    # --------------------------------------------------------
    # A. 基础数据
    # --------------------------------------------------------

    trade_dates = (
        load_trade_dates()
    )

    stock_master = (
        load_stock_master()
    )

    # --------------------------------------------------------
    # B. 构造矩阵
    # --------------------------------------------------------

    (
        present,
        valid,
        security_ids,
    ) = (
        build_presence_and_valid_matrix(
            trade_dates,
            stock_master,
        )
    )

    # --------------------------------------------------------
    # C. 网络可行性
    # --------------------------------------------------------

    (
        feasibility_df,
        coverage_distribution_df,
        stock_feasibility_df,
        latest_stock_df,
        latest_window_cache,
    ) = (
        run_network_feasibility(
            trade_dates,
            stock_master,
            present,
            valid,
        )
    )

    # --------------------------------------------------------
    # D. Summary
    # --------------------------------------------------------

    summary_df = (
        summarize_feasibility(
            feasibility_df
        )
    )

    latest_snapshot = (
        build_latest_snapshot(
            feasibility_df
        )
    )

    dimension_table = (
        build_dimension_diagnostic_table(
            latest_snapshot
        )
    )

    # --------------------------------------------------------
    # E. Pairwise common observations
    # --------------------------------------------------------

    pairwise_df = (
        run_latest_pairwise_diagnostic(
            valid,
            latest_window_cache,
        )
    )

    # --------------------------------------------------------
    # F. 保存
    # --------------------------------------------------------

    feasibility_df.to_csv(
        OUTPUT_DIR
        / "network_feasibility_by_date.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coverage_distribution_df.to_csv(
        OUTPUT_DIR
        / "window_valid_ratio_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "network_feasibility_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    latest_snapshot.to_csv(
        OUTPUT_DIR
        / "network_feasibility_latest_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dimension_table.to_csv(
        OUTPUT_DIR
        / "latest_dimension_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stock_feasibility_df.to_parquet(
        OUTPUT_DIR
        / "stock_level_feasibility_summary.parquet",
        index=False,
    )

    latest_stock_df.to_parquet(
        OUTPUT_DIR
        / "latest_stock_window_coverage.parquet",
        index=False,
    )

    if not pairwise_df.empty:

        pairwise_df.to_csv(
            OUTPUT_DIR
            / "pairwise_overlap_latest.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # G. 画图
    # --------------------------------------------------------

    plot_feasibility(
        feasibility_df
    )

    # --------------------------------------------------------
    # H. Metadata
    # --------------------------------------------------------

    metadata = {

        "research_start":
            str(
                START_DATE.date()
            ),

        "research_end":
            str(
                END_DATE.date()
            ),

        "return_panel":
            str(
                RETURN_PANEL_PATH
            ),

        "return_column":
            RETURN_COLUMN,

        "windows":
            WINDOWS,

        "min_valid_ratios":
            MIN_VALID_RATIOS,

        "window_definition":
            (
                "W consecutive market trading dates"
            ),

        "eligible_stock_definition":
            (
                "Stock must be present in the research panel "
                "at window end and have at least ceil(c*W) "
                "non-missing return_network observations "
                "during the rolling window."
            ),

        "p_over_w_definition":
            (
                "eligible_stock_count / rolling_window_length"
            ),

        "covariance_singularity_note":
            (
                "For a demeaned W x p complete-data matrix, "
                "the sample covariance rank is at most W-1. "
                "Therefore p >= W implies dimensional "
                "singularity before considering additional "
                "missing-data complications."
            ),

        "pairwise_diagnostic":
            {

                "enabled":
                    RUN_PAIRWISE_DIAGNOSTIC,

                "date":
                    str(
                        END_DATE.date()
                    ),

                "individual_min_valid_ratio":
                    PAIRWISE_VALID_RATIO,

                "pair_sample_size":
                    PAIR_SAMPLE_SIZE,

                "random_seed":
                    RANDOM_SEED,
            },

        "important_note":
            (
                "This step evaluates statistical/data feasibility "
                "only. It does not select the final network method "
                "and does not construct the formal Pearson, "
                "partial-correlation, Graphical Lasso, or "
                "factor-residual network."
            ),
    }

    with open(
        OUTPUT_DIR
        / "step4_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # I. Console
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("Step 4 完成")
    print("=" * 72)

    print()
    print("最新日期网络维度诊断：")

    show_cols = [

        "window",
        "min_valid_ratio",
        "current_universe_count",
        "eligible_stock_count",
        "eligible_stock_ratio",
        "p_over_w",
        "possible_pair_count",
        "covariance_dimensionally_singular",
    ]

    print(
        latest_snapshot[
            show_cols
        ].to_string(
            index=False
        )
    )

    if not pairwise_df.empty:

        print()
        print("最新日期 Pairwise Overlap 抽样：")

        print(
            pairwise_df.to_string(
                index=False
            )
        )

    print()
    print(
        f"输出目录：{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()