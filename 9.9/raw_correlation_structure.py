from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib.pyplot as plt


# ============================================================
# 0. 项目路径
# ============================================================

# ------------------------------------------------------------
# 你指定的项目输出根目录
# ------------------------------------------------------------

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day2"
    / "02_step2_raw_correlation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. Day 1 数据输入
# ============================================================

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


# ============================================================
# 2. Day 2 Step 1 配置路径
#
# 同时兼容：
#
# D:\M1_StockNetwork\output\01_step1_baseline_config
#
# 和之前的：
#
# D:\M1_day2\01_step1_baseline_config
# ============================================================

STEP1_DIR_CANDIDATES = [

    OUTPUT_ROOT
    / "01_step1_baseline_config",

    Path(
        r"D:\M1_StockNetwork\output\M1_day2\01_step1_baseline_config"
    ),
]


def resolve_step1_dir():

    for path in (
        STEP1_DIR_CANDIDATES
    ):

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
        "未找到 Day 2 Step 1 输出目录。\n"
        "已尝试：\n"
        +
        "\n".join(
            str(x)
            for x
            in STEP1_DIR_CANDIDATES
        )
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
# 3. Raw Correlation 参数
# ============================================================

# ------------------------------------------------------------
# 每个 Month-End × W 随机抽多少 pair
# ------------------------------------------------------------

MONTHLY_PAIR_SAMPLE_SIZE = 20_000


# ------------------------------------------------------------
# 计算 correlation 时分批处理
# 防止一次生成巨大 W x N 矩阵
# ------------------------------------------------------------

PAIR_BATCH_SIZE = 2_000


# ------------------------------------------------------------
# 年度代表日期保留多少 pair
# ------------------------------------------------------------

SAVE_REPRESENTATIVE_PAIR_SAMPLE = True


# ------------------------------------------------------------
# 每个代表性日期保存 sampled pair 中
# |rho| 最大的多少条
# ------------------------------------------------------------

TOP_PAIR_COUNT = 100


# ------------------------------------------------------------
# 固定随机种子
# ------------------------------------------------------------

RANDOM_SEED = 20260910


# ------------------------------------------------------------
# Parquet 流式读取批大小
# ------------------------------------------------------------

PARQUET_BATCH_SIZE = 300_000


# ============================================================
# 4. 读取 Step 1 Config
# ============================================================

def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(
            f
        )

    rolling = config[
        "rolling_design"
    ]

    return_info = config[
        "return"
    ]

    windows = [
        int(x)
        for x
        in rolling[
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
# 5. Trading Calendar
# ============================================================

def load_trade_dates():

    calendar = pd.read_csv(
        TRADE_CALENDAR_PATH
    )

    if (
        "trade_date"
        not in calendar.columns
    ):

        raise ValueError(
            "trade_calendar_used.csv "
            "缺少 trade_date。"
        )

    dates = pd.to_datetime(
        calendar[
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


# ============================================================
# 6. Stock Master
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
            f"stock_master 缺字段："
            f"{missing}"
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
        "stock_code"
    ] = (
        master[
            "stock_code"
        ]
        .astype("string")
    )

    master[
        "stock_name"
    ] = (
        master[
            "stock_name"
        ]
        .astype("string")
    )

    master[
        "exchange"
    ] = (
        master[
            "exchange"
        ]
        .astype("string")
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
            "stock_master 中 "
            "security_id 存在重复。"
        )

    master = (
        master
        .sort_values(
            "security_id"
        )
        .reset_index(
            drop=True
        )
    )

    return master


# ============================================================
# 7. 分析日期
# ============================================================

def load_analysis_dates():

    df = pd.read_csv(
        ANALYSIS_DATES_PATH
    )

    required = {

        "window",
        "analysis_date",
        "window_start",
    }

    if not required.issubset(
        df.columns
    ):

        raise ValueError(
            "analysis_dates_by_window.csv "
            "字段不完整。"
        )

    df = df.copy()

    df[
        "analysis_date"
    ] = pd.to_datetime(
        df[
            "analysis_date"
        ]
    )

    df[
        "window_start"
    ] = pd.to_datetime(
        df[
            "window_start"
        ]
    )

    df[
        "window"
    ] = (
        df[
            "window"
        ]
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
        df[
            "window"
        ]
        .astype(int)
    )

    return {

        (
            int(row.window),
            pd.Timestamp(
                row.analysis_date
            ),
        )

        for row in (
            df.itertuples(
                index=False
            )
        )
    }


# ============================================================
# 8. 构造全历史 T × P 收益矩阵
#
# T ≈ 2837
# P ≈ 5550
#
# float32 约 60 MB，
# 比反复读取 1100 万行 Panel 更高效。
# ============================================================

def build_return_matrix(
    trade_dates,
    master,
    return_column,
):

    print()
    print("=" * 76)
    print("构造 Full-History Return Matrix")
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

    # --------------------------------------------------------
    # float32 足够用于存储收益。
    #
    # 后续 Pearson 计算会转成 float64。
    # --------------------------------------------------------

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

        for i, date
        in enumerate(
            trade_dates
        )
    }

    security_to_idx = {

        sid: i

        for i, sid
        in enumerate(
            security_ids
        )
    }

    parquet_file = pq.ParquetFile(
        RETURN_PANEL_PATH
    )

    required_columns = [

        "trade_date",
        "security_id",
        return_column,
    ]

    panel_columns = (
        parquet_file
        .schema_arrow
        .names
    )

    missing = [

        x
        for x in required_columns

        if x not in panel_columns
    ]

    if missing:

        raise ValueError(
            f"Return Panel 缺字段："
            f"{missing}"
        )

    total_rows = 0
    mapped_rows = 0
    valid_rows = 0

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
                .astype(
                    np.int32
                )
                .to_numpy()
            )

            c = (
                col_idx.loc[
                    mapped
                ]
                .astype(
                    np.int32
                )
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

            # NaN 保留为 missing
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

            mapped_rows += len(
                r
            )

            valid_rows += int(
                finite.sum()
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

    print()
    print(
        f"Panel 总记录："
        f"{total_rows:,}"
    )

    print(
        f"成功映射："
        f"{mapped_rows:,}"
    )

    print(
        f"有效 {return_column}："
        f"{valid_rows:,}"
    )

    return (
        returns,
        security_ids,
    )


# ============================================================
# 9. 当前时点 PIT Universe
# ============================================================

def current_pit_mask(
    master,
    date,
):

    date = pd.Timestamp(
        date
    )

    mask = (

        master[
            "list_date"
        ]
        .le(
            date
        )

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
            .gt(
                date
            )
        )
    )

    return (
        mask
        .to_numpy(
            dtype=bool
        )
    )


# ============================================================
# 10. 随机抽取不重复股票 Pair
#
# 返回 eligible local indices：
#
# pair[:,0], pair[:,1]
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
            n_stocks
            -
            1
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

    while len(
        collected
    ) < target:

        remaining = (
            target
            -
            len(
                collected
            )
        )

        # 多抽一些，减少重复后不足
        draw_size = max(
            int(
                remaining
                *
                1.3
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
            a
            !=
            b
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

        if len(
            collected
        ) > target:

            collected = (
                collected[
                    :target
                ]
            )

    return collected


# ============================================================
# 11. Exact Pairwise Pearson Correlation
#
# 每个 pair 仅使用二者共同非缺失日期。
#
# 不进行 mean imputation。
# 不把停牌收益设为 0。
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

        # shape:
        # W x batch
        x = (
            window_returns[
                :,
                a,
            ]
            .astype(
                np.float64,
                copy=False,
            )
        )

        y = (
            window_returns[
                :,
                b,
            ]
            .astype(
                np.float64,
                copy=False,
            )
        )

        valid = (
            np.isfinite(
                x
            )
            &
            np.isfinite(
                y
            )
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

        if not sufficient.any():

            continue

        # ----------------------------------------------------
        # Pairwise-complete sums
        # ----------------------------------------------------

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

        sum_x = x0.sum(
            axis=0
        )

        sum_y = y0.sum(
            axis=0
        )

        sum_x2 = (
            x0
            *
            x0
        ).sum(
            axis=0
        )

        sum_y2 = (
            y0
            *
            y0
        ).sum(
            axis=0
        )

        sum_xy = (
            x0
            *
            y0
        ).sum(
            axis=0
        )

        n_float = (
            n.astype(
                np.float64
            )
        )

        # ----------------------------------------------------
        # 未归一化 covariance numerator
        #
        # sum[(x-xbar)(y-ybar)]
        # ----------------------------------------------------

        cov_num = (
            sum_xy
            -
            (
                sum_x
                *
                sum_y
                /
                n_float
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
                n_float
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
                n_float
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
                denominator
                >
                0
            )
        )

        corr_batch = np.full(
            end - start,
            np.nan,
            dtype=float,
        )

        corr_batch[
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

        # 数值误差可能导致
        # 1.0000000002
        corr_batch = np.clip(
            corr_batch,
            -1.0,
            1.0,
        )

        corr[
            start:end
        ] = corr_batch

    return (
        corr,
        common_obs,
    )


# ============================================================
# 12. Correlation Distribution Summary
# ============================================================

def summarize_correlations(
    corr,
    common_obs,
    window,
    sampled_pair_count,
):

    valid = np.isfinite(
        corr
    )

    valid_corr = corr[
        valid
    ]

    valid_common = (
        common_obs[
            valid
        ]
    )

    if len(
        valid_corr
    ) == 0:

        return {

            "sampled_pair_count":
                sampled_pair_count,

            "valid_pair_count":
                0,

            "valid_pair_ratio":
                0.0,
        }

    q = np.quantile(
        valid_corr,
        [
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ],
    )

    abs_q = np.quantile(
        np.abs(
            valid_corr
        ),
        [
            0.50,
            0.90,
            0.95,
            0.99,
        ],
    )

    return {

        "sampled_pair_count":
            int(
                sampled_pair_count
            ),

        "valid_pair_count":
            int(
                len(
                    valid_corr
                )
            ),

        "valid_pair_ratio":
            float(
                len(
                    valid_corr
                )
                /
                sampled_pair_count
            ),

        "mean_corr":
            float(
                np.mean(
                    valid_corr
                )
            ),

        "median_corr":
            float(
                np.median(
                    valid_corr
                )
            ),

        "std_corr":
            float(
                np.std(
                    valid_corr,
                    ddof=1,
                )
            ),

        "mean_abs_corr":
            float(
                np.mean(
                    np.abs(
                        valid_corr
                    )
                )
            ),

        "q01_corr":
            float(
                q[0]
            ),

        "q05_corr":
            float(
                q[1]
            ),

        "q10_corr":
            float(
                q[2]
            ),

        "q25_corr":
            float(
                q[3]
            ),

        "q50_corr":
            float(
                q[4]
            ),

        "q75_corr":
            float(
                q[5]
            ),

        "q90_corr":
            float(
                q[6]
            ),

        "q95_corr":
            float(
                q[7]
            ),

        "q99_corr":
            float(
                q[8]
            ),

        "median_abs_corr":
            float(
                abs_q[0]
            ),

        "q90_abs_corr":
            float(
                abs_q[1]
            ),

        "q95_abs_corr":
            float(
                abs_q[2]
            ),

        "q99_abs_corr":
            float(
                abs_q[3]
            ),

        "share_corr_negative":
            float(
                (
                    valid_corr
                    <
                    0
                ).mean()
            ),

        "share_corr_gt_0_1":
            float(
                (
                    valid_corr
                    >
                    0.10
                ).mean()
            ),

        "share_corr_gt_0_2":
            float(
                (
                    valid_corr
                    >
                    0.20
                ).mean()
            ),

        "share_corr_gt_0_3":
            float(
                (
                    valid_corr
                    >
                    0.30
                ).mean()
            ),

        "share_corr_gt_0_5":
            float(
                (
                    valid_corr
                    >
                    0.50
                ).mean()
            ),

        "share_abs_corr_gt_0_3":
            float(
                (
                    np.abs(
                        valid_corr
                    )
                    >
                    0.30
                ).mean()
            ),

        "share_abs_corr_gt_0_5":
            float(
                (
                    np.abs(
                        valid_corr
                    )
                    >
                    0.50
                ).mean()
            ),

        "mean_common_obs":
            float(
                np.mean(
                    valid_common
                )
            ),

        "median_common_obs":
            float(
                np.median(
                    valid_common
                )
            ),

        "mean_common_ratio":
            float(
                np.mean(
                    valid_common
                    /
                    window
                )
            ),

        "min_common_obs":
            int(
                np.min(
                    valid_common
                )
            ),
    }


# ============================================================
# 13. Main Raw Correlation Analysis
# ============================================================

def run_raw_correlation_analysis(
    trade_dates,
    master,
    returns,
    analysis_dates,
    representative_dates,
    windows,
    stock_valid_ratio,
    pair_valid_ratio,
):

    date_to_idx = {

        pd.Timestamp(date):
            i

        for i, date
        in enumerate(
            trade_dates
        )
    }

    # --------------------------------------------------------
    # Return validity cumulative sum
    #
    # shape:
    # (T+1) x P
    # --------------------------------------------------------

    valid_indicator = (
        np.isfinite(
            returns
        )
    )

    cumulative_valid = np.zeros(
        (
            valid_indicator.shape[0]
            +
            1,

            valid_indicator.shape[1],
        ),
        dtype=np.int32,
    )

    cumulative_valid[
        1:
    ] = np.cumsum(
        valid_indicator,
        axis=0,
        dtype=np.int32,
    )

    summary_rows = []

    representative_samples = []

    sampled_top_pairs = []

    # ========================================================
    # 每个 W
    # ========================================================

    for window in windows:

        print()
        print("=" * 76)
        print(
            f"Raw Correlation: W = {window}"
        )
        print("=" * 76)

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

        for counter, row in enumerate(

            dates_w.itertuples(
                index=False
            ),

            start=1,
        ):

            date = pd.Timestamp(
                row.analysis_date
            )

            if (
                date
                not in
                date_to_idx
            ):

                raise ValueError(
                    f"{date} 不在交易日历。"
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

            # =================================================
            # 1. 当前 PIT Universe
            # =================================================

            pit_mask = (
                current_pit_mask(
                    master,
                    date,
                )
            )

            current_universe_count = int(
                pit_mask.sum()
            )

            # =================================================
            # 2. 单股票有效观测数量
            # =================================================

            valid_counts = (

                cumulative_valid[
                    end_idx
                    +
                    1
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
                    valid_counts
                    >=
                    required_stock_days
                )
            )

            eligible_indices = np.flatnonzero(
                eligible_mask
            )

            p = len(
                eligible_indices
            )

            if p < 2:

                continue

            # =================================================
            # 3. 固定且可复现的随机种子
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

            rng = (
                np.random.default_rng(
                    seed
                )
            )

            # =================================================
            # 4. 从 Eligible 全市场中随机抽 Pair
            # =================================================

            local_pairs = (
                sample_unique_pairs(
                    n_stocks=p,
                    sample_size=
                        MONTHLY_PAIR_SAMPLE_SIZE,
                    rng=rng,
                )
            )

            stock_a = (
                eligible_indices[
                    local_pairs[
                        :,
                        0
                    ]
                ]
            )

            stock_b = (
                eligible_indices[
                    local_pairs[
                        :,
                        1
                    ]
                ]
            )

            # =================================================
            # 5. 当前 W 日收益窗口
            # =================================================

            window_returns = (
                returns[
                    start_idx:
                    end_idx
                    +
                    1,
                    :
                ]
            )

            # =================================================
            # 6. Exact Pairwise Pearson
            # =================================================

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

            # =================================================
            # 7. Distribution Summary
            # =================================================

            stats = (
                summarize_correlations(
                    corr=
                        corr,

                    common_obs=
                        common_obs,

                    window=
                        window,

                    sampled_pair_count=
                        len(
                            local_pairs
                        ),
                )
            )

            summary_rows.append(
                {

                    "analysis_date":
                        date,

                    "window":
                        window,

                    "window_start":
                        trade_dates[
                            start_idx
                        ],

                    "stock_min_valid_ratio":
                        stock_valid_ratio,

                    "pair_min_common_ratio":
                        pair_valid_ratio,

                    "required_stock_valid_days":
                        required_stock_days,

                    "required_pair_common_days":
                        required_pair_days,

                    "current_pit_universe_count":
                        current_universe_count,

                    "eligible_stock_count":
                        p,

                    "eligible_stock_ratio":
                        (
                            p
                            /
                            current_universe_count
                            if
                            current_universe_count
                            >
                            0
                            else
                            np.nan
                        ),

                    "possible_pair_count":
                        (
                            p
                            *
                            (
                                p
                                -
                                1
                            )
                            //
                            2
                        ),

                    **stats,
                }
            )

            # =================================================
            # 8. 年度代表日期：保存 sampled pairs
            # =================================================

            if (
                SAVE_REPRESENTATIVE_PAIR_SAMPLE
                and
                (
                    window,
                    date,
                )
                in
                representative_dates
            ):

                valid_pair = np.isfinite(
                    corr
                )

                if valid_pair.any():

                    rep = pd.DataFrame(
                        {

                            "analysis_date":
                                date,

                            "window":
                                window,

                            "security_id_i":
                                master.iloc[
                                    stock_a[
                                        valid_pair
                                    ]
                                ][
                                    "security_id"
                                ]
                                .to_numpy(),

                            "stock_code_i":
                                master.iloc[
                                    stock_a[
                                        valid_pair
                                    ]
                                ][
                                    "stock_code"
                                ]
                                .to_numpy(),

                            "stock_name_i":
                                master.iloc[
                                    stock_a[
                                        valid_pair
                                    ]
                                ][
                                    "stock_name"
                                ]
                                .to_numpy(),

                            "security_id_j":
                                master.iloc[
                                    stock_b[
                                        valid_pair
                                    ]
                                ][
                                    "security_id"
                                ]
                                .to_numpy(),

                            "stock_code_j":
                                master.iloc[
                                    stock_b[
                                        valid_pair
                                    ]
                                ][
                                    "stock_code"
                                ]
                                .to_numpy(),

                            "stock_name_j":
                                master.iloc[
                                    stock_b[
                                        valid_pair
                                    ]
                                ][
                                    "stock_name"
                                ]
                                .to_numpy(),

                            "common_obs":
                                common_obs[
                                    valid_pair
                                ],

                            "common_ratio":
                                (
                                    common_obs[
                                        valid_pair
                                    ]
                                    /
                                    window
                                ),

                            "correlation":
                                corr[
                                    valid_pair
                                ],

                            "abs_correlation":
                                np.abs(
                                    corr[
                                        valid_pair
                                    ]
                                ),
                        }
                    )

                    representative_samples.append(
                        rep
                    )

                    top = (
                        rep
                        .sort_values(
                            "abs_correlation",
                            ascending=False,
                        )
                        .head(
                            TOP_PAIR_COUNT
                        )
                        .copy()
                    )

                    # 强调这只是 sampled top
                    top[
                        "top_pair_definition"
                    ] = (
                        "Top |rho| within random pair sample; "
                        "not global full-market top edge"
                    )

                    sampled_top_pairs.append(
                        top
                    )

            if (
                counter == 1
                or
                counter
                %
                12
                ==
                0
                or
                counter
                ==
                len(
                    dates_w
                )
            ):

                mean_corr = (
                    stats.get(
                        "mean_corr",
                        np.nan,
                    )
                )

                q95 = (
                    stats.get(
                        "q95_corr",
                        np.nan,
                    )
                )

                print(
                    f"[{counter:3d}/"
                    f"{len(dates_w):3d}] "
                    f"{date.date()} | "
                    f"p={p:,} | "
                    f"mean rho="
                    f"{mean_corr:.4f} | "
                    f"Q95="
                    f"{q95:.4f}"
                )

    # ========================================================
    # Results
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    if representative_samples:

        representative_df = pd.concat(
            representative_samples,
            ignore_index=True,
        )

    else:

        representative_df = (
            pd.DataFrame()
        )

    if sampled_top_pairs:

        top_df = pd.concat(
            sampled_top_pairs,
            ignore_index=True,
        )

    else:

        top_df = (
            pd.DataFrame()
        )

    return (
        summary_df,
        representative_df,
        top_df,
    )


# ============================================================
# 14. Window-level Summary
# ============================================================

def build_window_comparison(
    summary_df,
):

    rows = []

    for window, group in (
        summary_df.groupby(
            "window"
        )
    ):

        rows.append(
            {

                "window":
                    int(
                        window
                    ),

                "analysis_date_count":
                    len(
                        group
                    ),

                "mean_eligible_stock_count":
                    float(
                        group[
                            "eligible_stock_count"
                        ]
                        .mean()
                    ),

                "median_eligible_stock_count":
                    float(
                        group[
                            "eligible_stock_count"
                        ]
                        .median()
                    ),

                "mean_raw_correlation":
                    float(
                        group[
                            "mean_corr"
                        ]
                        .mean()
                    ),

                "median_of_monthly_median_corr":
                    float(
                        group[
                            "median_corr"
                        ]
                        .median()
                    ),

                "mean_absolute_correlation":
                    float(
                        group[
                            "mean_abs_corr"
                        ]
                        .mean()
                    ),

                "median_q90_corr":
                    float(
                        group[
                            "q90_corr"
                        ]
                        .median()
                    ),

                "median_q95_corr":
                    float(
                        group[
                            "q95_corr"
                        ]
                        .median()
                    ),

                "median_q99_corr":
                    float(
                        group[
                            "q99_corr"
                        ]
                        .median()
                    ),

                "mean_share_negative":
                    float(
                        group[
                            "share_corr_negative"
                        ]
                        .mean()
                    ),

                "mean_share_corr_gt_0_3":
                    float(
                        group[
                            "share_corr_gt_0_3"
                        ]
                        .mean()
                    ),

                "mean_share_corr_gt_0_5":
                    float(
                        group[
                            "share_corr_gt_0_5"
                        ]
                        .mean()
                    ),

                "mean_pair_valid_ratio":
                    float(
                        group[
                            "valid_pair_ratio"
                        ]
                        .mean()
                    ),

                "mean_common_ratio":
                    float(
                        group[
                            "mean_common_ratio"
                        ]
                        .mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 15. Year-level Summary
# ============================================================

def build_yearly_summary(
    summary_df,
):

    df = summary_df.copy()

    df[
        "year"
    ] = (
        pd.to_datetime(
            df[
                "analysis_date"
            ]
        )
        .dt.year
    )

    yearly = (
        df
        .groupby(
            [
                "year",
                "window",
            ],
            as_index=False,
        )
        .agg(

            analysis_month_count=
                (
                    "analysis_date",
                    "count",
                ),

            mean_corr=
                (
                    "mean_corr",
                    "mean",
                ),

            median_corr=
                (
                    "median_corr",
                    "mean",
                ),

            mean_abs_corr=
                (
                    "mean_abs_corr",
                    "mean",
                ),

            q95_corr=
                (
                    "q95_corr",
                    "mean",
                ),

            q99_corr=
                (
                    "q99_corr",
                    "mean",
                ),

            share_negative=
                (
                    "share_corr_negative",
                    "mean",
                ),

            share_corr_gt_0_3=
                (
                    "share_corr_gt_0_3",
                    "mean",
                ),

            share_corr_gt_0_5=
                (
                    "share_corr_gt_0_5",
                    "mean",
                ),

            eligible_stock_count=
                (
                    "eligible_stock_count",
                    "mean",
                ),
        )
    )

    return yearly


# ============================================================
# 16. 绘图
# ============================================================

def plot_raw_correlation_summary(
    summary_df,
    windows,
):

    for window in windows:

        df = (
            summary_df[
                summary_df[
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
        # Figure 1:
        # Mean / Median Raw Correlation
        # ----------------------------------------------------

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
                "mean_corr"
            ],
            label=
                "Mean correlation",
        )

        ax.plot(
            df[
                "analysis_date"
            ],
            df[
                "median_corr"
            ],
            label=
                "Median correlation",
        )

        ax.set_title(
            f"Raw Correlation Level - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Correlation"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            /
            f"raw_corr_mean_median_W{window}.png",
            dpi=180,
        )

        plt.close(
            fig
        )

        # ----------------------------------------------------
        # Figure 2:
        # Mean Absolute Correlation
        # ----------------------------------------------------

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
                "mean_abs_corr"
            ],
        )

        ax.set_title(
            f"Mean Absolute Raw Correlation - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Mean |Correlation|"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            /
            f"raw_corr_mean_abs_W{window}.png",
            dpi=180,
        )

        plt.close(
            fig
        )

        # ----------------------------------------------------
        # Figure 3:
        # Upper Tail
        # ----------------------------------------------------

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
                "q90_corr"
            ],
            label="Q90",
        )

        ax.plot(
            df[
                "analysis_date"
            ],
            df[
                "q95_corr"
            ],
            label="Q95",
        )

        ax.plot(
            df[
                "analysis_date"
            ],
            df[
                "q99_corr"
            ],
            label="Q99",
        )

        ax.set_title(
            f"Upper Tail of Raw Correlation - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Correlation Quantile"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            /
            f"raw_corr_upper_tail_W{window}.png",
            dpi=180,
        )

        plt.close(
            fig
        )

        # ----------------------------------------------------
        # Figure 4:
        # Strong Positive Correlation Share
        # ----------------------------------------------------

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
                "share_corr_gt_0_3"
            ],
            label=
                "rho > 0.3",
        )

        ax.plot(
            df[
                "analysis_date"
            ],
            df[
                "share_corr_gt_0_5"
            ],
            label=
                "rho > 0.5",
        )

        ax.set_title(
            f"Share of Strong Positive Raw Correlations - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Pair Share"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            /
            f"raw_corr_strong_share_W{window}.png",
            dpi=180,
        )

        plt.close(
            fig
        )

        # ----------------------------------------------------
        # Figure 5:
        # Negative Correlation Share
        # ----------------------------------------------------

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
                "share_corr_negative"
            ],
        )

        ax.set_title(
            f"Share of Negative Raw Correlations - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Negative Pair Share"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            /
            f"raw_corr_negative_share_W{window}.png",
            dpi=180,
        )

        plt.close(
            fig
        )


# ============================================================
# 17. Metadata
# ============================================================

def save_metadata(
    config,
    windows,
    stock_valid_ratio,
    pair_valid_ratio,
    return_column,
    summary_df,
):

    metadata = {

        "step":
            "Day2_Step2_Raw_Correlation_Structure",

        "output_dir":
            str(
                OUTPUT_DIR
            ),

        "return_panel":
            str(
                RETURN_PANEL_PATH
            ),

        "return_column":
            return_column,

        "windows":
            windows,

        "stock_min_valid_ratio":
            stock_valid_ratio,

        "pair_min_common_ratio":
            pair_valid_ratio,

        "analysis_frequency":
            "month_end",

        "monthly_pair_sample_size":
            MONTHLY_PAIR_SAMPLE_SIZE,

        "correlation_method":
            "Pearson",

        "missing_data_rule":
            (
                "Pairwise complete observations. "
                "No zero filling and no mean imputation."
            ),

        "pair_sampling_rule":
            (
                "For every month-end and rolling window, "
                "randomly sample stock pairs from the full "
                "eligible PIT universe. Distributional "
                "statistics therefore represent Monte Carlo "
                "estimates of the full-market pair distribution."
            ),

        "analysis_date_count":
            int(
                len(
                    summary_df
                )
            ),

        "important_notes": [

            (
                "This step studies unconditional raw return "
                "correlation only."
            ),

            (
                "No market-factor residualization is performed."
            ),

            (
                "No industry adjustment is performed."
            ),

            (
                "No correlation threshold is used to define "
                "network edges."
            ),

            (
                "Sampled top pairs are top pairs only within "
                "the random sample and are not claimed to be "
                "the exact global strongest full-market pairs."
            ),

            (
                "The purpose is to establish the raw "
                "association benchmark before common-factor, "
                "Same/Cross Industry and residual-network "
                "diagnostics."
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        / "step2_raw_correlation_metadata.json",
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
# 18. 主程序
# ============================================================

def main():

    print("=" * 76)
    print("M1 Day 2 - Step 2")
    print("Full-Market Raw Correlation Structure")
    print("=" * 76)

    # ========================================================
    # 1. Config
    # ========================================================

    print()
    print("[1] 读取 Day 2 Step 1 配置")

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
        f"Return Column   : {return_column}"
    )

    print(
        f"Windows         : {windows}"
    )

    print(
        f"Stock c         : {stock_valid_ratio}"
    )

    print(
        f"Pair c          : {pair_valid_ratio}"
    )

    # ========================================================
    # 2. Calendar + Master
    # ========================================================

    print()
    print("[2] 读取交易日历和 Stock Master")

    trade_dates = (
        load_trade_dates()
    )

    master = (
        load_stock_master()
    )

    print(
        f"Trading Days    : {len(trade_dates):,}"
    )

    print(
        f"Historical IDs  : {len(master):,}"
    )

    # ========================================================
    # 3. Analysis Dates
    # ========================================================

    print()
    print("[3] 读取 Month-End 分析日期")

    analysis_dates = (
        load_analysis_dates()
    )

    representative_dates = (
        load_representative_dates()
    )

    print(
        "Month-End × Window rows: "
        f"{len(analysis_dates):,}"
    )

    print(
        "Representative rows     : "
        f"{len(representative_dates):,}"
    )

    # ========================================================
    # 4. Return Matrix
    # ========================================================

    (
        returns,
        security_ids,
    ) = (
        build_return_matrix(
            trade_dates=
                trade_dates,

            master=
                master,

            return_column=
                return_column,
        )
    )

    # ========================================================
    # 5. Raw Correlation
    # ========================================================

    print()
    print("[4] 开始 Raw Correlation 结构诊断")

    (
        summary_df,
        representative_df,
        top_df,
    ) = (
        run_raw_correlation_analysis(

            trade_dates=
                trade_dates,

            master=
                master,

            returns=
                returns,

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
        )
    )

    # ========================================================
    # 6. Summary
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
    # 7. 输出
    # ========================================================

    summary_df.to_csv(
        OUTPUT_DIR
        /
        "raw_correlation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    window_comparison.to_csv(
        OUTPUT_DIR
        /
        "raw_correlation_window_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    yearly_summary.to_csv(
        OUTPUT_DIR
        /
        "raw_correlation_yearly_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not representative_df.empty:

        representative_df.to_parquet(
            OUTPUT_DIR
            /
            "raw_correlation_pair_sample.parquet",
            index=False,
        )

    if not top_df.empty:

        top_df.to_csv(
            OUTPUT_DIR
            /
            "raw_sampled_top_pairs.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # ========================================================
    # 8. Figures
    # ========================================================

    print()
    print("[5] 绘制 Raw Correlation 时间序列")

    plot_raw_correlation_summary(
        summary_df=
            summary_df,

        windows=
            windows,
    )

    # ========================================================
    # 9. Metadata
    # ========================================================

    save_metadata(
        config=
            config,

        windows=
            windows,

        stock_valid_ratio=
            stock_valid_ratio,

        pair_valid_ratio=
            pair_valid_ratio,

        return_column=
            return_column,

        summary_df=
            summary_df,
    )

    # ========================================================
    # 10. Console
    # ========================================================

    print()
    print("=" * 76)
    print("Day 2 Step 2 完成")
    print("=" * 76)

    print()
    print("Window Comparison:")

    show_columns = [

        "window",

        "mean_eligible_stock_count",

        "mean_raw_correlation",

        "mean_absolute_correlation",

        "median_q95_corr",

        "median_q99_corr",

        "mean_share_negative",

        "mean_share_corr_gt_0_3",

        "mean_share_corr_gt_0_5",

        "mean_pair_valid_ratio",
    ]

    print(
        window_comparison[
            show_columns
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"输出目录：{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()