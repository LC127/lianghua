from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib.pyplot as plt

try:
    from sklearn.utils.extmath import randomized_svd
except ImportError as e:
    raise ImportError(
        "Step 3 需要 scikit-learn：\n"
        "pip install scikit-learn"
    ) from e


# ============================================================
# 0. 项目目录
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day2"
    / "03_step3_common_factor"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. Day 1 数据
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
# 2. Day 2 Step 1 配置
# ============================================================

STEP1_DIR_CANDIDATES = [

    OUTPUT_ROOT
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
        "没有找到 Day 2 Step 1 配置目录。\n"
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
# 3. Day 2 Step 2 Raw Correlation
# ============================================================

STEP2_DIR = (
    OUTPUT_ROOT
    / "M1_day2"
    / "02_step2_raw_correlation"
)

RAW_CORRELATION_SUMMARY_PATH = (
    STEP2_DIR
    / "raw_correlation_summary.csv"
)


# ============================================================
# 4. 参数
# ============================================================

PARQUET_BATCH_SIZE = 300_000

RANDOM_SEED = 20260910


# ------------------------------------------------------------
# PCA
#
# 使用前 5 个 principal components
# ------------------------------------------------------------

PCA_COMPONENTS = 5

PCA_N_ITER = 3


# ------------------------------------------------------------
# 是否保存所有 Month-End 的逐股 beta/R2
#
# 不建议第一版打开，会产生较大的文件。
# ------------------------------------------------------------

SAVE_ALL_STOCK_REGRESSIONS = False


# ------------------------------------------------------------
# 每年代表日期保存逐股 beta/R2
# ------------------------------------------------------------

SAVE_REPRESENTATIVE_STOCK_REGRESSIONS = True


# ------------------------------------------------------------
# 市值必须为正
# ------------------------------------------------------------

MIN_MARKET_VALUE = 0.0


# ============================================================
# 5. Step 1 Config
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

    return_column = (
        return_info[
            "baseline_column"
        ]
    )

    return (
        config,
        windows,
        stock_valid_ratio,
        return_column,
    )


# ============================================================
# 6. Trading Calendar
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
# 7. Stock Master
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

    master["list_date"] = (
        pd.to_datetime(
            master["list_date"],
            errors="coerce",
        )
    )

    master["delist_date"] = (
        pd.to_datetime(
            master["delist_date"],
            errors="coerce",
        )
    )

    if (
        master["security_id"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "stock_master 中 security_id 重复。"
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
# 8. Analysis Dates
# ============================================================

def load_analysis_dates():

    df = pd.read_csv(
        ANALYSIS_DATES_PATH
    )

    df["analysis_date"] = (
        pd.to_datetime(
            df["analysis_date"]
        )
    )

    df["window_start"] = (
        pd.to_datetime(
            df["window_start"]
        )
    )

    df["window"] = (
        df["window"]
        .astype(int)
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

    df["analysis_date"] = (
        pd.to_datetime(
            df["analysis_date"]
        )
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
# 9. 构造 T x P Return + Market Value Matrix
# ============================================================

def build_return_and_market_value_matrix(
    trade_dates,
    master,
    return_column,
):

    print()
    print("=" * 76)
    print("构造 Return / Market Value Matrix")
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

    returns = np.full(
        (
            T,
            P,
        ),
        np.nan,
        dtype=np.float32,
    )

    market_value = np.full(
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
        "market_value",
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
    valid_return_rows = 0
    valid_mv_rows = 0

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

        df["trade_date"] = (
            pd.to_datetime(
                df["trade_date"],
                errors="coerce",
            )
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

            ret = (
                pd.to_numeric(
                    df.loc[
                        mapped,
                        return_column,
                    ],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=float
                )
            )

            mv = (
                pd.to_numeric(
                    df.loc[
                        mapped,
                        "market_value",
                    ],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=float
                )
            )

            valid_ret = np.isfinite(
                ret
            )

            valid_mv = (
                np.isfinite(
                    mv
                )
                &
                (
                    mv
                    >
                    MIN_MARKET_VALUE
                )
            )

            returns[
                r[valid_ret],
                c[valid_ret],
            ] = (
                ret[valid_ret]
                .astype(np.float32)
            )

            market_value[
                r[valid_mv],
                c[valid_mv],
            ] = (
                mv[valid_mv]
                .astype(np.float32)
            )

            valid_return_rows += int(
                valid_ret.sum()
            )

            valid_mv_rows += int(
                valid_mv.sum()
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
        f"Panel rows       : {total_rows:,}"
    )

    print(
        f"Valid returns    : {valid_return_rows:,}"
    )

    print(
        f"Valid market cap : {valid_mv_rows:,}"
    )

    return (
        returns,
        market_value,
    )


# ============================================================
# 10. PIT Universe Mask
# ============================================================

def current_pit_mask(
    master,
    date,
):

    date = pd.Timestamp(
        date
    )

    mask = (

        master["list_date"]
        .le(date)

        &

        (
            master["delist_date"]
            .isna()

            |

            master["delist_date"]
            .gt(date)
        )
    )

    return (
        mask
        .to_numpy(
            dtype=bool
        )
    )


# ============================================================
# 11. Lagged Market Value Window
#
# 对窗口日期 s 使用 MV_{s-1}
# ============================================================

def get_lagged_market_value_window(
    market_value,
    start_idx,
    end_idx,
    eligible_idx,
):

    W = (
        end_idx
        -
        start_idx
        +
        1
    )

    if start_idx > 0:

        lag_mv = (
            market_value[
                start_idx - 1:
                end_idx,
                :
            ][
                :,
                eligible_idx
            ]
        )

    else:

        lag_mv = np.full(
            (
                W,
                len(
                    eligible_idx
                ),
            ),
            np.nan,
            dtype=np.float32,
        )

        if W > 1:

            lag_mv[
                1:
            ] = (
                market_value[
                    0:
                    end_idx,
                    :
                ][
                    :,
                    eligible_idx
                ]
            )

    return lag_mv


# ============================================================
# 12. 构造 EW / VW Market Factor
# ============================================================

def construct_market_factors(
    X,
    lagged_market_value,
):

    X64 = X.astype(
        np.float64,
        copy=False,
    )

    # ========================================================
    # Equal Weight
    # ========================================================

    valid_return = np.isfinite(
        X64
    )

    ew_count = (
        valid_return
        .sum(
            axis=1
        )
    )

    ew_sum = np.where(
        valid_return,
        X64,
        0.0,
    ).sum(
        axis=1
    )

    ew_factor = np.divide(

        ew_sum,

        ew_count,

        out=np.full(
            len(
                ew_count
            ),
            np.nan,
            dtype=float,
        ),

        where=(
            ew_count
            >
            0
        ),
    )

    # ========================================================
    # Value Weight
    #
    # 使用 lagged market value
    # ========================================================

    mv = (
        lagged_market_value
        .astype(
            np.float64,
            copy=False,
        )
    )

    valid_vw = (

        valid_return

        &

        np.isfinite(
            mv
        )

        &

        (
            mv > 0
        )
    )

    weight = np.where(
        valid_vw,
        mv,
        0.0,
    )

    weighted_return = np.where(
        valid_vw,
        X64
        *
        mv,
        0.0,
    )

    denominator = (
        weight.sum(
            axis=1
        )
    )

    numerator = (
        weighted_return.sum(
            axis=1
        )
    )

    vw_factor = np.divide(

        numerator,

        denominator,

        out=np.full(
            len(
                denominator
            ),
            np.nan,
            dtype=float,
        ),

        where=(
            denominator
            >
            0
        ),
    )

    vw_count = (
        valid_vw.sum(
            axis=1
        )
    )

    return (
        ew_factor,
        vw_factor,
        ew_count,
        vw_count,
    )


# ============================================================
# 13. 向量化 Rolling Factor Regression
#
# r_i = alpha_i + beta_i F + epsilon_i
#
# 返回：
# alpha
# beta
# R2
# n_obs
# ============================================================

def factor_regression(
    X,
    factor,
    required_obs,
):

    X = X.astype(
        np.float64,
        copy=False,
    )

    f = np.asarray(
        factor,
        dtype=np.float64,
    )

    valid = (

        np.isfinite(
            X
        )

        &

        np.isfinite(
            f
        )[
            :,
            None
        ]
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
        f[
            :,
            None
        ],
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

    n_float = (
        n.astype(
            np.float64
        )
    )

    usable_n = (
        n
        >=
        required_obs
    )

    safe_n = np.where(
        n_float
        >
        0,
        n_float,
        1.0,
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
        (
            sum_x
            *
            sum_x
            /
            safe_n
        )
    )

    ss_f = (
        sum_f2
        -
        (
            sum_f
            *
            sum_f
            /
            safe_n
        )
    )

    cov_fx = (
        sum_fx
        -
        (
            sum_f
            *
            sum_x
            /
            safe_n
        )
    )

    usable = (

        usable_n

        &

        np.isfinite(
            ss_x
        )

        &

        np.isfinite(
            ss_f
        )

        &

        (
            ss_x > 0
        )

        &

        (
            ss_f > 0
        )
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
        ]
        ** 2

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

    return (
        alpha,
        beta,
        r2,
        n,
    )


# ============================================================
# 14. Regression Summary
# ============================================================

def summarize_factor_regression(
    beta,
    r2,
    prefix,
):

    valid = (

        np.isfinite(
            beta
        )

        &

        np.isfinite(
            r2
        )
    )

    beta = beta[
        valid
    ]

    r2 = r2[
        valid
    ]

    if len(beta) == 0:

        return {
            f"{prefix}_regression_stock_count":
                0
        }

    beta_q = np.quantile(
        beta,
        [
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ],
    )

    r2_q = np.quantile(
        r2,
        [
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ],
    )

    return {

        f"{prefix}_regression_stock_count":
            int(
                len(beta)
            ),

        f"{prefix}_beta_mean":
            float(
                np.mean(beta)
            ),

        f"{prefix}_beta_q10":
            float(
                beta_q[0]
            ),

        f"{prefix}_beta_q25":
            float(
                beta_q[1]
            ),

        f"{prefix}_beta_median":
            float(
                beta_q[2]
            ),

        f"{prefix}_beta_q75":
            float(
                beta_q[3]
            ),

        f"{prefix}_beta_q90":
            float(
                beta_q[4]
            ),

        f"{prefix}_r2_mean":
            float(
                np.mean(r2)
            ),

        f"{prefix}_r2_q10":
            float(
                r2_q[0]
            ),

        f"{prefix}_r2_q25":
            float(
                r2_q[1]
            ),

        f"{prefix}_r2_median":
            float(
                r2_q[2]
            ),

        f"{prefix}_r2_q75":
            float(
                r2_q[3]
            ),

        f"{prefix}_r2_q90":
            float(
                r2_q[4]
            ),

        f"{prefix}_share_r2_gt_0_3":
            float(
                (
                    r2
                    >
                    0.30
                ).mean()
            ),

        f"{prefix}_share_r2_gt_0_5":
            float(
                (
                    r2
                    >
                    0.50
                ).mean()
            ),

        f"{prefix}_share_r2_gt_0_7":
            float(
                (
                    r2
                    >
                    0.70
                ).mean()
            ),
    }


# ============================================================
# 15. 安全相关系数
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

    if (
        valid.sum()
        <
        3
    ):

        return np.nan

    xv = x[
        valid
    ]

    yv = y[
        valid
    ]

    if (
        np.std(xv) == 0
        or
        np.std(yv) == 0
    ):

        return np.nan

    return float(
        np.corrcoef(
            xv,
            yv,
        )[0, 1]
    )


# ============================================================
# 16. Correlation-Based PCA / Market Mode
#
# 1. 每只股票使用其有效样本估计 mean/std
# 2. 标准化
# 3. 缺失值在“标准化以后”置 0
#
# 因为 0 表示该股票自己的样本均值，而不是
# 把原始收益人为置 0。
#
# 4. 根据有效样本数重新缩放列范数，
#    降低少量 missing 对 PCA 权重的影响。
#
# 5. 使用 randomized SVD，只计算前 5 个主成分。
# ============================================================

def pca_market_mode(
    X,
    ew_factor,
    vw_factor,
    random_state,
):

    X = X.astype(
        np.float64,
        copy=False,
    )

    valid = np.isfinite(
        X
    )

    n_obs = (
        valid.sum(
            axis=0
        )
    )

    means = np.nanmean(
        X,
        axis=0,
    )

    stds = np.nanstd(
        X,
        axis=0,
        ddof=1,
    )

    usable_stock = (

        np.isfinite(
            means
        )

        &

        np.isfinite(
            stds
        )

        &

        (
            stds
            >
            1e-12
        )

        &

        (
            n_obs
            >
            2
        )
    )

    X_use = (
        X[
            :,
            usable_stock
        ]
    )

    means_use = (
        means[
            usable_stock
        ]
    )

    stds_use = (
        stds[
            usable_stock
        ]
    )

    n_use = (
        n_obs[
            usable_stock
        ]
    )

    if (
        X_use.shape[1]
        <
        2
    ):

        return {
            "pca_stock_count": 0
        }

    # --------------------------------------------------------
    # Z-score
    # --------------------------------------------------------

    Z = (
        X_use
        -
        means_use[
            None,
            :
        ]
    ) / (
        stds_use[
            None,
            :
        ]
    )

    # 缺失值在 demean/standardize 后填 0
    Z[
        ~np.isfinite(
            Z
        )
    ] = 0.0

    # --------------------------------------------------------
    # 每一列理论标准化平方和约为 n_i - 1。
    #
    # 调整到约 W - 1，
    # 避免 missing 较多股票在 PCA 中机械降权。
    # --------------------------------------------------------

    W = (
        X.shape[0]
    )

    scale = np.sqrt(

        (
            W - 1
        )

        /

        np.maximum(
            n_use
            -
            1,
            1,
        )
    )

    Z *= scale[
        None,
        :
    ]

    n_components = min(
        PCA_COMPONENTS,
        Z.shape[0],
        Z.shape[1],
    )

    (
        U,
        singular_values,
        Vt,
    ) = randomized_svd(

        Z,

        n_components=
            n_components,

        n_iter=
            PCA_N_ITER,

        random_state=
            random_state,
    )

    # --------------------------------------------------------
    # Explained Variance Ratio
    #
    # 对 SVD:
    #
    # singular_value_k^2
    #
    # 相对于整个标准化数据矩阵 Frobenius norm^2。
    # --------------------------------------------------------

    total_ss = float(
        np.sum(
            Z
            *
            Z
        )
    )

    explained_ratio = (

        singular_values
        ** 2

        /

        total_ss
    )

    # --------------------------------------------------------
    # PC1 factor score
    # --------------------------------------------------------

    pc1_score = (
        U[
            :,
            0
        ]
        *
        singular_values[
            0
        ]
    )

    pc1_loading = (
        Vt[
            0,
            :
        ]
        .copy()
    )

    # --------------------------------------------------------
    # PCA 符号不唯一。
    #
    # 用 EW Market Factor 对齐方向。
    # --------------------------------------------------------

    corr_pc1_ew = (
        safe_corr(
            pc1_score,
            ew_factor,
        )
    )

    if (
        np.isfinite(
            corr_pc1_ew
        )
        and
        corr_pc1_ew
        <
        0
    ):

        pc1_score *= -1

        pc1_loading *= -1

    corr_pc1_ew = (
        safe_corr(
            pc1_score,
            ew_factor,
        )
    )

    corr_pc1_vw = (
        safe_corr(
            pc1_score,
            vw_factor,
        )
    )

    loading_q = np.quantile(
        pc1_loading,
        [
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ],
    )

    result = {

        "pca_stock_count":
            int(
                X_use.shape[1]
            ),

        "pca_pc1_evr":
            float(
                explained_ratio[0]
            ),

        "pca_pc1_2_evr":
            float(
                explained_ratio[
                    :min(
                        2,
                        len(
                            explained_ratio
                        ),
                    )
                ].sum()
            ),

        "pca_pc1_3_evr":
            float(
                explained_ratio[
                    :min(
                        3,
                        len(
                            explained_ratio
                        ),
                    )
                ].sum()
            ),

        "pca_pc1_5_evr":
            float(
                explained_ratio.sum()
            ),

        "pca_pc1_corr_ew_factor":
            corr_pc1_ew,

        "pca_pc1_corr_vw_factor":
            corr_pc1_vw,

        "pca_pc1_loading_mean":
            float(
                np.mean(
                    pc1_loading
                )
            ),

        "pca_pc1_loading_q10":
            float(
                loading_q[0]
            ),

        "pca_pc1_loading_q25":
            float(
                loading_q[1]
            ),

        "pca_pc1_loading_median":
            float(
                loading_q[2]
            ),

        "pca_pc1_loading_q75":
            float(
                loading_q[3]
            ),

        "pca_pc1_loading_q90":
            float(
                loading_q[4]
            ),

        "pca_pc1_loading_positive_share":
            float(
                (
                    pc1_loading
                    >
                    0
                ).mean()
            ),
    }

    return result


# ============================================================
# 17. Step 3 Main Analysis
# ============================================================

def run_common_factor_analysis(
    trade_dates,
    master,
    returns,
    market_value,
    analysis_dates,
    representative_dates,
    windows,
    stock_valid_ratio,
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
    # Rolling valid count
    # --------------------------------------------------------

    valid_return = (
        np.isfinite(
            returns
        )
    )

    cumulative_valid = np.zeros(
        (
            len(
                trade_dates
            )
            +
            1,

            returns.shape[1],
        ),
        dtype=np.int32,
    )

    cumulative_valid[
        1:
    ] = np.cumsum(
        valid_return,
        axis=0,
        dtype=np.int32,
    )

    summary_rows = []

    factor_ts_parts = []

    representative_regression_parts = []

    all_regression_parts = []

    # ========================================================
    # Window
    # ========================================================

    for window in windows:

        print()
        print("=" * 76)
        print(
            f"Common Factor Diagnostics: W={window}"
        )
        print("=" * 76)

        required_obs = int(
            math.ceil(
                stock_valid_ratio
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

            # =================================================
            # 1. PIT + Valid Return Universe
            # =================================================

            pit_mask = (
                current_pit_mask(
                    master,
                    date,
                )
            )

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
                    required_obs
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
            # 2. W x p Return Matrix
            # =================================================

            X = (
                returns[
                    start_idx:
                    end_idx
                    +
                    1,
                    :
                ][
                    :,
                    eligible_idx
                ]
            )

            # =================================================
            # 3. Lagged Market Value
            # =================================================

            lag_mv = (
                get_lagged_market_value_window(
                    market_value=
                        market_value,

                    start_idx=
                        start_idx,

                    end_idx=
                        end_idx,

                    eligible_idx=
                        eligible_idx,
                )
            )

            # =================================================
            # 4. EW / VW Market Factors
            # =================================================

            (
                ew_factor,
                vw_factor,
                ew_count,
                vw_count,
            ) = (
                construct_market_factors(
                    X=
                        X,

                    lagged_market_value=
                        lag_mv,
                )
            )

            factor_corr = (
                safe_corr(
                    ew_factor,
                    vw_factor,
                )
            )

            # =================================================
            # 5. EW Factor Regression
            # =================================================

            (
                ew_alpha,
                ew_beta,
                ew_r2,
                ew_n,
            ) = (
                factor_regression(
                    X=
                        X,

                    factor=
                        ew_factor,

                    required_obs=
                        required_obs,
                )
            )

            ew_summary = (
                summarize_factor_regression(
                    beta=
                        ew_beta,

                    r2=
                        ew_r2,

                    prefix=
                        "ew",
                )
            )

            # =================================================
            # 6. VW Factor Regression
            # =================================================

            (
                vw_alpha,
                vw_beta,
                vw_r2,
                vw_n,
            ) = (
                factor_regression(
                    X=
                        X,

                    factor=
                        vw_factor,

                    required_obs=
                        required_obs,
                )
            )

            vw_summary = (
                summarize_factor_regression(
                    beta=
                        vw_beta,

                    r2=
                        vw_r2,

                    prefix=
                        "vw",
                )
            )

            # =================================================
            # 7. PCA / Market Mode
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

            pca_summary = (
                pca_market_mode(
                    X=
                        X,

                    ew_factor=
                        ew_factor,

                    vw_factor=
                        vw_factor,

                    random_state=
                        seed,
                )
            )

            # =================================================
            # 8. Monthly Summary
            # =================================================

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

                    "required_valid_days":
                        required_obs,

                    "current_pit_universe_count":
                        int(
                            pit_mask.sum()
                        ),

                    "eligible_stock_count":
                        p,

                    "eligible_stock_ratio":
                        float(
                            p
                            /
                            pit_mask.sum()
                        ),

                    # -----------------------------------------
                    # Market Factor
                    # -----------------------------------------

                    "ew_factor_mean":
                        float(
                            np.nanmean(
                                ew_factor
                            )
                        ),

                    "ew_factor_std":
                        float(
                            np.nanstd(
                                ew_factor,
                                ddof=1,
                            )
                        ),

                    "vw_factor_mean":
                        float(
                            np.nanmean(
                                vw_factor
                            )
                        ),

                    "vw_factor_std":
                        float(
                            np.nanstd(
                                vw_factor,
                                ddof=1,
                            )
                        ),

                    "ew_vw_factor_corr":
                        factor_corr,

                    "ew_mean_contributor_count":
                        float(
                            np.mean(
                                ew_count
                            )
                        ),

                    "vw_mean_contributor_count":
                        float(
                            np.mean(
                                vw_count
                            )
                        ),

                    **ew_summary,

                    **vw_summary,

                    **pca_summary,
                }
            )

            # =================================================
            # 9. 保存 Factor Time Series
            # =================================================

            factor_ts = pd.DataFrame(
                {

                    "analysis_date":
                        date,

                    "window":
                        window,

                    "factor_date":
                        trade_dates[
                            start_idx:
                            end_idx
                            +
                            1
                        ],

                    "ew_market_factor":
                        ew_factor,

                    "vw_market_factor":
                        vw_factor,

                    "ew_contributor_count":
                        ew_count,

                    "vw_contributor_count":
                        vw_count,
                }
            )

            factor_ts_parts.append(
                factor_ts
            )

            # =================================================
            # 10. Stock-Level Regression
            # =================================================

            save_stock_level = (

                SAVE_ALL_STOCK_REGRESSIONS

                or

                (
                    SAVE_REPRESENTATIVE_STOCK_REGRESSIONS

                    and

                    (
                        window,
                        date,
                    )
                    in
                    representative_dates
                )
            )

            if save_stock_level:

                meta = (
                    master.iloc[
                        eligible_idx
                    ][
                        [
                            "security_id",
                            "stock_code",
                            "stock_name",
                            "exchange",
                        ]
                    ]
                    .reset_index(
                        drop=True
                    )
                )

                stock_result = (
                    meta.copy()
                )

                stock_result[
                    "analysis_date"
                ] = date

                stock_result[
                    "window"
                ] = window

                stock_result[
                    "ew_alpha"
                ] = ew_alpha

                stock_result[
                    "ew_beta"
                ] = ew_beta

                stock_result[
                    "ew_r2"
                ] = ew_r2

                stock_result[
                    "ew_n_obs"
                ] = ew_n

                stock_result[
                    "vw_alpha"
                ] = vw_alpha

                stock_result[
                    "vw_beta"
                ] = vw_beta

                stock_result[
                    "vw_r2"
                ] = vw_r2

                stock_result[
                    "vw_n_obs"
                ] = vw_n

                if (
                    SAVE_ALL_STOCK_REGRESSIONS
                ):

                    all_regression_parts.append(
                        stock_result
                    )

                else:

                    representative_regression_parts.append(
                        stock_result
                    )

            # =================================================
            # Console
            # =================================================

            if (
                counter == 1
                or
                counter % 12 == 0
                or
                counter == len(
                    dates_w
                )
            ):

                print(
                    f"[{counter:3d}/"
                    f"{len(dates_w):3d}] "
                    f"{date.date()} | "
                    f"p={p:,} | "
                    f"EW median R2="
                    f"{ew_summary.get('ew_r2_median', np.nan):.4f} | "
                    f"PC1 EVR="
                    f"{pca_summary.get('pca_pc1_evr', np.nan):.4f}"
                )

    # ========================================================
    # Output DataFrames
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    factor_ts_df = pd.concat(
        factor_ts_parts,
        ignore_index=True,
    )

    if representative_regression_parts:

        representative_regression_df = (
            pd.concat(
                representative_regression_parts,
                ignore_index=True,
            )
        )

    else:

        representative_regression_df = (
            pd.DataFrame()
        )

    if all_regression_parts:

        all_regression_df = pd.concat(
            all_regression_parts,
            ignore_index=True,
        )

    else:

        all_regression_df = (
            pd.DataFrame()
        )

    return (
        summary_df,
        factor_ts_df,
        representative_regression_df,
        all_regression_df,
    )


# ============================================================
# 18. Window-Level Summary
# ============================================================

def build_window_comparison(
    summary_df,
):

    rows = []

    for window, df in (
        summary_df.groupby(
            "window"
        )
    ):

        rows.append(
            {

                "window":
                    int(window),

                "analysis_date_count":
                    int(
                        len(df)
                    ),

                "mean_eligible_stock_count":
                    float(
                        df[
                            "eligible_stock_count"
                        ].mean()
                    ),

                # --------------------------------------------
                # Market Factor
                # --------------------------------------------

                "mean_ew_factor_vol":
                    float(
                        df[
                            "ew_factor_std"
                        ].mean()
                    ),

                "mean_vw_factor_vol":
                    float(
                        df[
                            "vw_factor_std"
                        ].mean()
                    ),

                "median_ew_vw_factor_corr":
                    float(
                        df[
                            "ew_vw_factor_corr"
                        ].median()
                    ),

                # --------------------------------------------
                # EW Regression
                # --------------------------------------------

                "mean_ew_beta_median":
                    float(
                        df[
                            "ew_beta_median"
                        ].mean()
                    ),

                "mean_ew_r2_mean":
                    float(
                        df[
                            "ew_r2_mean"
                        ].mean()
                    ),

                "median_ew_r2_median":
                    float(
                        df[
                            "ew_r2_median"
                        ].median()
                    ),

                "mean_ew_share_r2_gt_0_5":
                    float(
                        df[
                            "ew_share_r2_gt_0_5"
                        ].mean()
                    ),

                # --------------------------------------------
                # VW Regression
                # --------------------------------------------

                "mean_vw_beta_median":
                    float(
                        df[
                            "vw_beta_median"
                        ].mean()
                    ),

                "mean_vw_r2_mean":
                    float(
                        df[
                            "vw_r2_mean"
                        ].mean()
                    ),

                "median_vw_r2_median":
                    float(
                        df[
                            "vw_r2_median"
                        ].median()
                    ),

                "mean_vw_share_r2_gt_0_5":
                    float(
                        df[
                            "vw_share_r2_gt_0_5"
                        ].mean()
                    ),

                # --------------------------------------------
                # PCA
                # --------------------------------------------

                "mean_pc1_evr":
                    float(
                        df[
                            "pca_pc1_evr"
                        ].mean()
                    ),

                "median_pc1_evr":
                    float(
                        df[
                            "pca_pc1_evr"
                        ].median()
                    ),

                "mean_pc1_5_evr":
                    float(
                        df[
                            "pca_pc1_5_evr"
                        ].mean()
                    ),

                "median_pc1_corr_ew":
                    float(
                        df[
                            "pca_pc1_corr_ew_factor"
                        ].median()
                    ),

                "median_pc1_corr_vw":
                    float(
                        df[
                            "pca_pc1_corr_vw_factor"
                        ].median()
                    ),

                "mean_pc1_loading_positive_share":
                    float(
                        df[
                            "pca_pc1_loading_positive_share"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 19. 合并 Step 2 Raw Correlation
# ============================================================

def merge_with_raw_correlation(
    common_factor_df,
):

    if not (
        RAW_CORRELATION_SUMMARY_PATH.exists()
    ):

        print(
            "[WARNING] "
            "没有找到 raw_correlation_summary.csv"
        )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    raw = pd.read_csv(
        RAW_CORRELATION_SUMMARY_PATH
    )

    raw[
        "analysis_date"
    ] = pd.to_datetime(
        raw[
            "analysis_date"
        ]
    )

    raw[
        "window"
    ] = (
        raw[
            "window"
        ]
        .astype(int)
    )

    raw_columns = [

        "analysis_date",
        "window",

        "mean_corr",
        "median_corr",
        "mean_abs_corr",

        "q95_corr",
        "q99_corr",

        "share_corr_negative",
        "share_corr_gt_0_3",
        "share_corr_gt_0_5",
    ]

    available = [

        x
        for x in raw_columns

        if x in raw.columns
    ]

    merged = (
        common_factor_df
        .merge(
            raw[
                available
            ],
            on=[
                "analysis_date",
                "window",
            ],
            how="left",
        )
        .rename(
            columns={

                "mean_corr":
                    "raw_mean_corr",

                "median_corr":
                    "raw_median_corr",

                "mean_abs_corr":
                    "raw_mean_abs_corr",

                "q95_corr":
                    "raw_q95_corr",

                "q99_corr":
                    "raw_q99_corr",

                "share_corr_negative":
                    "raw_share_negative",

                "share_corr_gt_0_3":
                    "raw_share_corr_gt_0_3",

                "share_corr_gt_0_5":
                    "raw_share_corr_gt_0_5",
            }
        )
    )

    relationship_rows = []

    for window, df in (
        merged.groupby(
            "window"
        )
    ):

        relationship_rows.append(
            {

                "window":
                    int(window),

                "month_count":
                    int(
                        len(df)
                    ),

                # --------------------------------------------
                # Raw Mean Corr vs PCA
                # --------------------------------------------

                "corr_raw_mean_corr_pc1_evr":
                    safe_corr(
                        df[
                            "raw_mean_corr"
                        ],
                        df[
                            "pca_pc1_evr"
                        ],
                    ),

                "corr_raw_mean_corr_pc1_5_evr":
                    safe_corr(
                        df[
                            "raw_mean_corr"
                        ],
                        df[
                            "pca_pc1_5_evr"
                        ],
                    ),

                # --------------------------------------------
                # Raw Mean Corr vs Regression R2
                # --------------------------------------------

                "corr_raw_mean_corr_ew_median_r2":
                    safe_corr(
                        df[
                            "raw_mean_corr"
                        ],
                        df[
                            "ew_r2_median"
                        ],
                    ),

                "corr_raw_mean_corr_vw_median_r2":
                    safe_corr(
                        df[
                            "raw_mean_corr"
                        ],
                        df[
                            "vw_r2_median"
                        ],
                    ),

                # --------------------------------------------
                # Strong correlation share
                # --------------------------------------------

                "corr_raw_strong_share_pc1_evr":
                    safe_corr(
                        df[
                            "raw_share_corr_gt_0_3"
                        ],
                        df[
                            "pca_pc1_evr"
                        ],
                    ),

                # --------------------------------------------
                # Market factor volatility
                # --------------------------------------------

                "corr_raw_mean_corr_ew_factor_vol":
                    safe_corr(
                        df[
                            "raw_mean_corr"
                        ],
                        df[
                            "ew_factor_std"
                        ],
                    ),

                "corr_raw_mean_corr_vw_factor_vol":
                    safe_corr(
                        df[
                            "raw_mean_corr"
                        ],
                        df[
                            "vw_factor_std"
                        ],
                    ),
            }
        )

    relationship_df = pd.DataFrame(
        relationship_rows
    )

    return (
        merged,
        relationship_df,
    )


# ============================================================
# 20. 年度汇总
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

            eligible_stock_count=
                (
                    "eligible_stock_count",
                    "mean",
                ),

            ew_factor_vol=
                (
                    "ew_factor_std",
                    "mean",
                ),

            vw_factor_vol=
                (
                    "vw_factor_std",
                    "mean",
                ),

            ew_r2_mean=
                (
                    "ew_r2_mean",
                    "mean",
                ),

            ew_r2_median=
                (
                    "ew_r2_median",
                    "mean",
                ),

            vw_r2_mean=
                (
                    "vw_r2_mean",
                    "mean",
                ),

            vw_r2_median=
                (
                    "vw_r2_median",
                    "mean",
                ),

            pc1_evr=
                (
                    "pca_pc1_evr",
                    "mean",
                ),

            pc1_5_evr=
                (
                    "pca_pc1_5_evr",
                    "mean",
                ),

            pc1_corr_ew=
                (
                    "pca_pc1_corr_ew_factor",
                    "mean",
                ),

            pc1_corr_vw=
                (
                    "pca_pc1_corr_vw_factor",
                    "mean",
                ),

            pc1_positive_loading_share=
                (
                    "pca_pc1_loading_positive_share",
                    "mean",
                ),
        )
    )

    return yearly


# ============================================================
# 21. Plot
# ============================================================

def zscore_series(
    x,
):

    x = pd.to_numeric(
        x,
        errors="coerce",
    )

    std = (
        x.std(
            ddof=1
        )
    )

    if (
        not np.isfinite(
            std
        )
        or
        std == 0
    ):

        return (
            x
            *
            np.nan
        )

    return (
        (
            x
            -
            x.mean()
        )
        /
        std
    )


def plot_common_factor(
    summary_df,
    merged_df,
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

        # ====================================================
        # Figure 1:
        # EW / VW Median R2
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
                "ew_r2_median"
            ],
            label=
                "EW market factor",
        )

        ax.plot(
            df[
                "analysis_date"
            ],
            df[
                "vw_r2_median"
            ],
            label=
                "VW market factor",
        )

        ax.set_title(
            f"Median Stock-Level Market-Factor R-squared - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Median R-squared"
        )

        ax.set_ylim(
            0,
            1,
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            /
            f"market_factor_median_r2_W{window}.png",
            dpi=180,
        )

        plt.close(
            fig
        )

        # ====================================================
        # Figure 2:
        # PCA EVR
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
                "pca_pc1_evr"
            ],
            label=
                "PC1 EVR",
        )

        ax.plot(
            df[
                "analysis_date"
            ],
            df[
                "pca_pc1_5_evr"
            ],
            label=
                "PC1-PC5 EVR",
        )

        ax.set_title(
            f"PCA Common-Factor Strength - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Explained Variance Ratio"
        )

        ax.set_ylim(
            0,
            1,
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            /
            f"pca_common_factor_evr_W{window}.png",
            dpi=180,
        )

        plt.close(
            fig
        )

        # ====================================================
        # Figure 3:
        # PC1 vs EW/VW correlation
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
                "pca_pc1_corr_ew_factor"
            ],
            label=
                "Corr(PC1, EW)",
        )

        ax.plot(
            df[
                "analysis_date"
            ],
            df[
                "pca_pc1_corr_vw_factor"
            ],
            label=
                "Corr(PC1, VW)",
        )

        ax.set_title(
            f"PC1 Alignment with Market Factors - W={window}"
        )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            "Correlation"
        )

        ax.set_ylim(
            -1,
            1,
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            /
            f"pc1_market_factor_alignment_W{window}.png",
            dpi=180,
        )

        plt.close(
            fig
        )

        # ====================================================
        # Figure 4:
        # Raw Correlation vs Common Factor
        #
        # 转成 z-score，仅比较时间共变，不比较绝对单位。
        # ====================================================

        if not merged_df.empty:

            m = (
                merged_df[
                    merged_df[
                        "window"
                    ]
                    .eq(
                        window
                    )
                ]
                .sort_values(
                    "analysis_date"
                )
                .copy()
            )

            if (
                "raw_mean_corr"
                in m.columns
            ):

                fig, ax = plt.subplots(
                    figsize=(
                        12,
                        6,
                    )
                )

                ax.plot(
                    m[
                        "analysis_date"
                    ],
                    zscore_series(
                        m[
                            "raw_mean_corr"
                        ]
                    ),
                    label=
                        "Raw Mean Corr",
                )

                ax.plot(
                    m[
                        "analysis_date"
                    ],
                    zscore_series(
                        m[
                            "pca_pc1_evr"
                        ]
                    ),
                    label=
                        "PC1 EVR",
                )

                ax.plot(
                    m[
                        "analysis_date"
                    ],
                    zscore_series(
                        m[
                            "ew_r2_median"
                        ]
                    ),
                    label=
                        "EW Median R2",
                )

                ax.set_title(
                    f"Raw Correlation vs Common-Factor Strength - W={window}"
                )

                ax.set_xlabel(
                    "Analysis Date"
                )

                ax.set_ylabel(
                    "Standardized Value"
                )

                ax.legend()

                ax.grid(
                    alpha=0.25
                )

                fig.tight_layout()

                fig.savefig(
                    OUTPUT_DIR
                    /
                    f"raw_corr_vs_common_factor_W{window}.png",
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
    return_column,
):

    metadata = {

        "step":
            "Day2_Step3_Common_Factor_Diagnostics",

        "output_dir":
            str(
                OUTPUT_DIR
            ),

        "return_column":
            return_column,

        "windows":
            windows,

        "stock_min_valid_ratio":
            stock_valid_ratio,

        "analysis_frequency":
            "month_end",

        "market_factors": {

            "equal_weighted":
                (
                    "Daily equal-weighted mean of valid "
                    "return_network observations among the "
                    "window-end eligible PIT universe."
                ),

            "value_weighted":
                (
                    "Daily value-weighted mean using previous "
                    "market trading day's market_value."
                ),
        },

        "factor_regression":
            (
                "r_i = alpha_i + beta_i * F + epsilon_i; "
                "OLS with intercept using valid stock-factor "
                "observations only."
            ),

        "pca_method":
            (
                "Correlation-based PCA using stock-wise "
                "standardized returns. Missing observations "
                "are filled with zero only after demeaning and "
                "standardization, so zero represents the "
                "stock-specific sample mean in standardized "
                "space. Column norms are rescaled for small "
                "differences in observation counts. "
                "Top components are computed using randomized SVD."
            ),

        "pca_components":
            PCA_COMPONENTS,

        "important_notes": [

            (
                "This step diagnoses common market factors; "
                "it does not yet construct residual networks."
            ),

            (
                "EW/VW market factors include each stock itself. "
                "With several thousand stocks, own-stock weight "
                "is small. Leave-one-out market factors may be "
                "used later as a robustness check."
            ),

            (
                "PCA is used as a diagnostic of the broad "
                "correlation mode rather than as the final "
                "factor model."
            ),

            (
                "A high Raw Correlation alone does not prove "
                "that the market factor causes stock dependence. "
                "Evidence should be based jointly on beta/R2, "
                "PC1 explained variance, PC1 market-factor "
                "alignment, and their co-movement with Raw "
                "Correlation."
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        / "step3_common_factor_metadata.json",
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
    print("M1 Day 2 - Step 3")
    print("Common Factor Diagnostics")
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
        return_column,
    ) = (
        load_config()
    )

    print(
        f"Return  : {return_column}"
    )

    print(
        f"Windows : {windows}"
    )

    print(
        f"c       : {stock_valid_ratio}"
    )

    # ========================================================
    # 2. Basic Data
    # ========================================================

    print()
    print("[2] 读取交易日历 / Stock Master")

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

    print(
        f"Trading days  : {len(trade_dates):,}"
    )

    print(
        f"Historical IDs: {len(master):,}"
    )

    print(
        "Analysis rows : "
        f"{len(analysis_dates):,}"
    )

    # ========================================================
    # 3. Return / Market Value Matrices
    # ========================================================

    print()
    print("[3] 构造 Return / Market Value 矩阵")

    (
        returns,
        market_value,
    ) = (
        build_return_and_market_value_matrix(

            trade_dates=
                trade_dates,

            master=
                master,

            return_column=
                return_column,
        )
    )

    # ========================================================
    # 4. Common Factor
    # ========================================================

    print()
    print("[4] 开始 Common Factor Diagnostics")

    (
        summary_df,
        factor_ts_df,
        representative_regression_df,
        all_regression_df,
    ) = (
        run_common_factor_analysis(

            trade_dates=
                trade_dates,

            master=
                master,

            returns=
                returns,

            market_value=
                market_value,

            analysis_dates=
                analysis_dates,

            representative_dates=
                representative_dates,

            windows=
                windows,

            stock_valid_ratio=
                stock_valid_ratio,
        )
    )

    # ========================================================
    # 5. Summaries
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

    (
        merged_raw_df,
        raw_relationship_df,
    ) = (
        merge_with_raw_correlation(
            summary_df
        )
    )

    # ========================================================
    # 6. Output
    # ========================================================

    summary_df.to_csv(
        OUTPUT_DIR
        /
        "common_factor_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    window_comparison.to_csv(
        OUTPUT_DIR
        /
        "common_factor_window_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    yearly_summary.to_csv(
        OUTPUT_DIR
        /
        "common_factor_yearly_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    factor_ts_df.to_parquet(
        OUTPUT_DIR
        /
        "rolling_market_factor_timeseries.parquet",
        index=False,
    )

    if not representative_regression_df.empty:

        representative_regression_df.to_parquet(
            OUTPUT_DIR
            /
            "representative_stock_factor_regression.parquet",
            index=False,
        )

    if (
        SAVE_ALL_STOCK_REGRESSIONS
        and
        not all_regression_df.empty
    ):

        all_regression_df.to_parquet(
            OUTPUT_DIR
            /
            "all_monthly_stock_factor_regression.parquet",
            index=False,
        )

    if not merged_raw_df.empty:

        merged_raw_df.to_csv(
            OUTPUT_DIR
            /
            "common_factor_vs_raw_correlation.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if not raw_relationship_df.empty:

        raw_relationship_df.to_csv(
            OUTPUT_DIR
            /
            "common_factor_vs_raw_window_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # ========================================================
    # 7. Figures
    # ========================================================

    print()
    print("[5] 绘图")

    plot_common_factor(
        summary_df=
            summary_df,

        merged_df=
            merged_raw_df,

        windows=
            windows,
    )

    # ========================================================
    # 8. Metadata
    # ========================================================

    save_metadata(

        windows=
            windows,

        stock_valid_ratio=
            stock_valid_ratio,

        return_column=
            return_column,
    )

    # ========================================================
    # 9. Console
    # ========================================================

    print()
    print("=" * 76)
    print("Day 2 Step 3 完成")
    print("=" * 76)

    print()
    print("Window Comparison:")

    show_columns = [

        "window",

        "median_ew_vw_factor_corr",

        "mean_ew_r2_mean",

        "median_ew_r2_median",

        "mean_vw_r2_mean",

        "median_vw_r2_median",

        "mean_pc1_evr",

        "mean_pc1_5_evr",

        "median_pc1_corr_ew",

        "median_pc1_corr_vw",

        "mean_pc1_loading_positive_share",
    ]

    print(
        window_comparison[
            show_columns
        ]
        .to_string(
            index=False
        )
    )

    if not raw_relationship_df.empty:

        print()
        print(
            "Raw Correlation "
            "vs Common Factor:"
        )

        print(
            raw_relationship_df
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