from __future__ import annotations

from pathlib import Path
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 0. Paths
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)

STEP1_DIR = (
    OUTPUT_ROOT
    / "M1_day4"
    / "01_step1_network_design"
)

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day4"
    / "02_step2_matched_networks"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Day 1 data foundation
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
# Day 4 Step 1
# ------------------------------------------------------------

NETWORK_CONFIG_PATH = (
    STEP1_DIR
    / "day4_network_design_config.json"
)

ANALYSIS_DATES_PATH = (
    STEP1_DIR
    / "day4_analysis_dates.csv"
)


# ============================================================
# 1. Output sub-directories
# ============================================================

EDGE_ROOT = (
    OUTPUT_DIR
    / "edge_master"
)

NODE_ROOT = (
    OUTPUT_DIR
    / "node_universe"
)

MARKER_ROOT = (
    OUTPUT_DIR
    / "job_markers"
)

EDGE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

NODE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MARKER_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Computational settings
# ============================================================

# ------------------------------------------------------------
# Block size:
#
# 512 is usually a reasonable compromise between BLAS efficiency
# and memory usage.
#
# If memory is limited, reduce to 256.
# ------------------------------------------------------------

PAIR_BLOCK_SIZE = 512


# ------------------------------------------------------------
# Return-panel parquet scan
# ------------------------------------------------------------

PARQUET_BATCH_SIZE = 300_000


# ------------------------------------------------------------
# Formal analysis runs all month-end dates.
#
# For first smoke test:
#
# TEST_LATEST_ONLY = True
#
# This will run only the latest date for each W.
#
# After confirming outputs, change back to False.
# ------------------------------------------------------------

TEST_LATEST_ONLY = False


# ------------------------------------------------------------
# Output compression
# ------------------------------------------------------------

PARQUET_COMPRESSION = "zstd"


# ============================================================
# 3. Load frozen Step 1 design
# ============================================================

def load_network_design():

    if not NETWORK_CONFIG_PATH.exists():

        raise FileNotFoundError(
            f"Step 1 config 不存在：\n"
            f"{NETWORK_CONFIG_PATH}"
        )

    with open(
        NETWORK_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    data_design = config[
        "data_design"
    ]

    primary = config[
        "primary_network_comparison"
    ]

    density_robustness = config[
        "density_robustness"
    ]

    return_column = (
        data_design[
            "return_column"
        ]
    )

    windows = [
        int(x)
        for x
        in data_design[
            "windows"
        ]
    ]

    stock_valid_ratio = float(
        data_design[
            "stock_valid_ratio"
        ]
    )

    pair_valid_ratio = float(
        data_design[
            "pair_valid_ratio"
        ]
    )

    q_grid = sorted(
        float(x)
        for x
        in density_robustness[
            "q_grid"
        ]
    )

    primary_q = float(
        primary[
            "primary_density"
        ]
    )

    return (
        config,
        return_column,
        windows,
        stock_valid_ratio,
        pair_valid_ratio,
        q_grid,
        primary_q,
    )


# ============================================================
# 4. Basic loaders
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
        col
        for col in required
        if col not in master.columns
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
            "stock_master 中 security_id 重复。"
        )

    # --------------------------------------------------------
    # Important:
    #
    # sort by security_id so master_index itself can be used
    # as deterministic tie-break order.
    # --------------------------------------------------------

    master = (
        master
        .sort_values(
            "security_id"
        )
        .reset_index(
            drop=True
        )
    )

    master[
        "master_index"
    ] = np.arange(
        len(master),
        dtype=np.int32,
    )

    return master


def load_analysis_dates(
    windows,
):

    if not ANALYSIS_DATES_PATH.exists():

        raise FileNotFoundError(
            f"不存在：{ANALYSIS_DATES_PATH}"
        )

    df = pd.read_csv(
        ANALYSIS_DATES_PATH
    )

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
    ] = (
        pd.to_numeric(
            df[
                "window"
            ],
            errors="raise",
        )
        .astype(int)
    )

    df = (
        df[
            df[
                "window"
            ]
            .isin(
                windows
            )
        ]
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .drop_duplicates(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if TEST_LATEST_ONLY:

        df = (
            df.groupby(
                "window",
                as_index=False,
            )
            .tail(1)
            .sort_values(
                "window"
            )
            .reset_index(
                drop=True
            )
        )

    return df


# ============================================================
# 5. Build return matrix + industry snapshots
#
# Read daily_return_panel only once.
# ============================================================

def build_market_data(
    trade_dates,
    master,
    analysis_dates,
    return_column,
):

    print()
    print("=" * 80)
    print(
        "Build Full-Market Return Matrix + Industry Snapshots"
    )
    print("=" * 80)

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

    date_to_idx = {
        pd.Timestamp(date): i
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
        col
        for col in required_columns
        if col not in (
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

            returns[
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
                    "industry_id1",
                    "industry_id2",
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

    # ========================================================
    # Encode historical industry labels
    # ========================================================

    industry_snapshots = {
        "industry_id1": {},
        "industry_id2": {},
    }

    industry_codebooks = {}

    for industry_level in [
        "industry_id1",
        "industry_id2",
    ]:

        values = (
            snapshot_df[
                industry_level
            ]
            .astype("string")
        )

        codes, uniques = pd.factorize(
            values,
            sort=True,
        )

        snapshot_df[
            f"{industry_level}_code"
        ] = (
            codes.astype(
                np.int16
            )
        )

        industry_codebooks[
            industry_level
        ] = {
            int(i): str(value)
            for i, value
            in enumerate(
                uniques
            )
        }

    for date, g in (
        snapshot_df.groupby(
            "trade_date",
            sort=False,
        )
    ):

        idx = (
            g[
                "master_index"
            ]
            .to_numpy(
                dtype=np.int32
            )
        )

        for industry_level in [
            "industry_id1",
            "industry_id2",
        ]:

            arr = np.full(
                P,
                -1,
                dtype=np.int16,
            )

            arr[
                idx
            ] = (
                g[
                    f"{industry_level}_code"
                ]
                .to_numpy(
                    dtype=np.int16
                )
            )

            industry_snapshots[
                industry_level
            ][
                pd.Timestamp(date)
            ] = arr

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

    return (
        returns,
        industry_snapshots,
    )


# ============================================================
# 6. PIT universe
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
# 7. Daily PIT equal-weight market factor
# ============================================================

def build_daily_pit_market_components(
    trade_dates,
    master,
    returns,
):

    print()
    print("=" * 80)
    print(
        "Build Daily PIT Equal-Weighted Market Factor Components"
    )
    print("=" * 80)

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
            np.isfinite(
                r
            )
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

    factor = np.divide(
        market_sum,
        market_count,
        out=np.full(
            T,
            np.nan,
            dtype=float,
        ),
        where=(
            market_count
            >
            0
        ),
    )

    pd.DataFrame(
        {
            "trade_date":
                trade_dates,

            "pit_market_return":
                factor,

            "pit_valid_stock_count":
                market_count,
        }
    ).to_csv(
        OUTPUT_DIR
        / "daily_pit_ew_market_factor.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return (
        market_sum,
        market_count,
    )


# ============================================================
# 8. Leave-one-out market factor
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
        denominator
        >
        0
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
# 9. Rolling market residualization
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
        np.isfinite(
            X
        )
        &
        np.isfinite(
            F
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

    sum_x = x0.sum(
        axis=0
    )

    sum_f = f0.sum(
        axis=0
    )

    sum_x2 = (
        x0
        *
        x0
    ).sum(
        axis=0
    )

    sum_f2 = (
        f0
        *
        f0
    ).sum(
        axis=0
    )

    sum_fx = (
        f0
        *
        x0
    ).sum(
        axis=0
    )

    safe_n = np.where(
        n > 0,
        n.astype(
            np.float64
        ),
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

    alpha = np.full(
        X.shape[1],
        np.nan,
        dtype=np.float64,
    )

    beta = np.full(
        X.shape[1],
        np.nan,
        dtype=np.float64,
    )

    r2 = np.full(
        X.shape[1],
        np.nan,
        dtype=np.float64,
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
        X
        -
        prediction,
        np.nan,
    )

    return (
        residual,
        alpha,
        beta,
        r2,
        n,
        usable,
    )


# ============================================================
# 10. Exact pairwise Pearson for one block
#
# Pairwise-complete correlation.
#
# For each pair (i,j), all sums are calculated only
# on days where BOTH stocks are observed.
# ============================================================

def pairwise_corr_block(
    XA,
    XB,
    required_common_days,
):

    XA = np.asarray(
        XA,
        dtype=np.float64,
    )

    XB = np.asarray(
        XB,
        dtype=np.float64,
    )

    valid_a = np.isfinite(
        XA
    )

    valid_b = np.isfinite(
        XB
    )

    A = np.where(
        valid_a,
        XA,
        0.0,
    )

    B = np.where(
        valid_b,
        XB,
        0.0,
    )

    MA = valid_a.astype(
        np.float64
    )

    MB = valid_b.astype(
        np.float64
    )

    # --------------------------------------------------------
    # Number of common valid observations
    # --------------------------------------------------------

    n = (
        MA.T
        @
        MB
    )

    # --------------------------------------------------------
    # Pair-specific sums over common observations
    # --------------------------------------------------------

    sum_x = (
        A.T
        @
        MB
    )

    sum_y = (
        MA.T
        @
        B
    )

    sum_x2 = (
        (A * A).T
        @
        MB
    )

    sum_y2 = (
        MA.T
        @
        (B * B)
    )

    sum_xy = (
        A.T
        @
        B
    )

    safe_n = np.where(
        n > 0,
        n,
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

    corr = np.full(
        n.shape,
        np.nan,
        dtype=np.float64,
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

    corr[
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

    corr = np.clip(
        corr,
        -1.0,
        1.0,
    )

    return (
        corr,
        n,
    )


# ============================================================
# 11. Top-edge accumulator
#
# We do not store all ~12.5 million pair correlations.
#
# We keep only enough candidates to recover the maximum
# requested density (2%).
# ============================================================

@dataclass
class TopEdgeAccumulator:

    max_keep: int

    def __post_init__(
        self
    ):

        self.score = np.empty(
            0,
            dtype=np.float64,
        )

        self.corr = np.empty(
            0,
            dtype=np.float64,
        )

        self.i = np.empty(
            0,
            dtype=np.int32,
        )

        self.j = np.empty(
            0,
            dtype=np.int32,
        )

    def add(
        self,
        score,
        corr,
        i,
        j,
    ):

        if len(
            score
        ) == 0:

            return

        score = np.asarray(
            score,
            dtype=np.float64,
        )

        corr = np.asarray(
            corr,
            dtype=np.float64,
        )

        i = np.asarray(
            i,
            dtype=np.int32,
        )

        j = np.asarray(
            j,
            dtype=np.int32,
        )

        self.score = np.concatenate(
            [
                self.score,
                score,
            ]
        )

        self.corr = np.concatenate(
            [
                self.corr,
                corr,
            ]
        )

        self.i = np.concatenate(
            [
                self.i,
                i,
            ]
        )

        self.j = np.concatenate(
            [
                self.j,
                j,
            ]
        )

        if (
            len(
                self.score
            )
            >
            self.max_keep
        ):

            keep = np.argpartition(
                self.score,
                -self.max_keep,
            )[
                -self.max_keep:
            ]

            self.score = (
                self.score[
                    keep
                ]
            )

            self.corr = (
                self.corr[
                    keep
                ]
            )

            self.i = (
                self.i[
                    keep
                ]
            )

            self.j = (
                self.j[
                    keep
                ]
            )

    def finalize(
        self
    ):

        if len(
            self.score
        ) == 0:

            return pd.DataFrame(
                columns=[
                    "master_index_i",
                    "master_index_j",
                    "correlation",
                    "edge_score",
                ]
            )

        # ----------------------------------------------------
        # Deterministic final ranking:
        #
        # 1. score descending
        # 2. master_index_i ascending
        # 3. master_index_j ascending
        #
        # Since master is sorted by security_id,
        # master_index order is identical to security_id order.
        # ----------------------------------------------------

        order = np.lexsort(
            (
                self.j,
                self.i,
                -self.score,
            )
        )

        result = pd.DataFrame(
            {
                "master_index_i":
                    self.i[
                        order
                    ],

                "master_index_j":
                    self.j[
                        order
                    ],

                "correlation":
                    self.corr[
                        order
                    ].astype(
                        np.float32
                    ),

                "edge_score":
                    self.score[
                        order
                    ].astype(
                        np.float32
                    ),
            }
        )

        result[
            "rank"
        ] = np.arange(
            1,
            len(
                result
            ) + 1,
            dtype=np.int32,
        )

        return result


# ============================================================
# 12. Atomic parquet writer
# ============================================================

def atomic_to_parquet(
    df,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = Path(
        str(path)
        +
        ".tmp.parquet"
    )

    df.to_parquet(
        temp_path,
        index=False,
        compression=
            PARQUET_COMPRESSION,
    )

    os.replace(
        temp_path,
        path,
    )


# ============================================================
# 13. Construct one Date × Window matched network pair
# ============================================================

def construct_one_job(
    date,
    window,
    trade_dates,
    master,
    returns,
    industry_snapshots,
    market_sum,
    market_count,
    stock_valid_ratio,
    pair_valid_ratio,
    q_grid,
):

    date = pd.Timestamp(
        date
    )

    date_string = (
        date.strftime(
            "%Y-%m-%d"
        )
    )

    marker_path = (
        MARKER_ROOT
        /
        f"W{window}_{date_string}.json"
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if marker_path.exists():

        print(
            f"SKIP completed: "
            f"W={window} {date_string}"
        )

        return

    date_to_idx = {
        pd.Timestamp(d): i
        for i, d
        in enumerate(
            trade_dates
        )
    }

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

        raise ValueError(
            f"W={window}, {date_string}: "
            "insufficient historical observations."
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

    # ========================================================
    # 13.1 Raw eligible universe
    # ========================================================

    X_full = (
        returns[
            start_idx:
            end_idx + 1,
            :
        ]
        .astype(
            np.float64
        )
    )

    valid_count = (
        np.isfinite(
            X_full
        )
        .sum(
            axis=0
        )
    )

    pit = current_pit_mask(
        master,
        date,
    )

    raw_eligible = (
        pit
        &
        (
            valid_count
            >=
            required_stock_days
        )
    )

    eligible_idx = np.flatnonzero(
        raw_eligible
    )

    X_eligible = (
        X_full[
            :,
            eligible_idx
        ]
    )

    if len(
        eligible_idx
    ) < 2:

        raise RuntimeError(
            "Eligible universe contains fewer than 2 stocks."
        )

    # ========================================================
    # 13.2 PIT leave-one-out market factor
    # ========================================================

    F_loo = build_loo_factor(
        X=
            X_eligible,

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

    (
        residual_eligible,
        alpha,
        beta,
        r2,
        regression_n,
        regression_usable,
    ) = (
        residualize_market_factor(
            X=
                X_eligible,

            F=
                F_loo,

            required_obs=
                required_stock_days,
        )
    )

    # ========================================================
    # 13.3 Common node universe
    #
    # Raw and Residual networks must use identical nodes.
    # ========================================================

    common_local = np.flatnonzero(
        regression_usable
    )

    common_master_idx = (
        eligible_idx[
            common_local
        ]
    )

    X = (
        X_eligible[
            :,
            common_local
        ]
    )

    residual = (
        residual_eligible[
            :,
            common_local
        ]
    )

    beta_common = (
        beta[
            common_local
        ]
    )

    r2_common = (
        r2[
            common_local
        ]
    )

    p = len(
        common_master_idx
    )

    if p < 2:

        raise RuntimeError(
            "Common Raw/Residual node universe "
            "contains fewer than 2 stocks."
        )

    total_possible_pairs = (
        p
        *
        (
            p - 1
        )
        //
        2
    )

    max_q = max(
        q_grid
    )

    # --------------------------------------------------------
    # Upper bound sufficient to recover all top max-q edges.
    # Actual common valid pair count <= total_possible_pairs.
    # --------------------------------------------------------

    max_keep = int(
        math.ceil(
            max_q
            *
            total_possible_pairs
        )
    )

    # Small safety margin
    max_keep += 100

    raw_top = TopEdgeAccumulator(
        max_keep=
            max_keep
    )

    residual_top = (
        TopEdgeAccumulator(
            max_keep=
                max_keep
        )
    )

    # ========================================================
    # Industry snapshots on common nodes
    # ========================================================

    ind1 = (
        industry_snapshots[
            "industry_id1"
        ][
            date
        ][
            common_master_idx
        ]
    )

    ind2 = (
        industry_snapshots[
            "industry_id2"
        ][
            date
        ][
            common_master_idx
        ]
    )

    # ========================================================
    # 13.4 Exact full-market pair scan
    # ========================================================

    common_pair_count = 0

    common_industry_pair_count = 0

    common_same_industry_count = 0

    raw_positive_count = 0

    residual_positive_count = 0

    block_starts = list(
        range(
            0,
            p,
            PAIR_BLOCK_SIZE,
        )
    )

    n_blocks = len(
        block_starts
    )

    block_job = 0

    total_block_jobs = (
        n_blocks
        *
        (
            n_blocks + 1
        )
        //
        2
    )

    for block_a, i0 in enumerate(
        block_starts
    ):

        i1 = min(
            i0
            +
            PAIR_BLOCK_SIZE,
            p,
        )

        XA = (
            X[
                :,
                i0:i1
            ]
        )

        RA = (
            residual[
                :,
                i0:i1
            ]
        )

        for block_b in range(
            block_a,
            n_blocks,
        ):

            j0 = (
                block_starts[
                    block_b
                ]
            )

            j1 = min(
                j0
                +
                PAIR_BLOCK_SIZE,
                p,
            )

            block_job += 1

            XB = (
                X[
                    :,
                    j0:j1
                ]
            )

            RB = (
                residual[
                    :,
                    j0:j1
                ]
            )

            (
                raw_corr,
                raw_n,
            ) = pairwise_corr_block(
                XA,
                XB,
                required_common_days=
                    required_pair_days,
            )

            (
                residual_corr,
                residual_n,
            ) = pairwise_corr_block(
                RA,
                RB,
                required_common_days=
                    required_pair_days,
            )

            # ------------------------------------------------
            # Common valid pair universe
            # ------------------------------------------------

            common_valid = (
                np.isfinite(
                    raw_corr
                )
                &
                np.isfinite(
                    residual_corr
                )
            )

            # ------------------------------------------------
            # Diagonal block:
            # keep only i < j
            # ------------------------------------------------

            if block_a == block_b:

                triangle = np.triu(
                    np.ones(
                        common_valid.shape,
                        dtype=bool,
                    ),
                    k=1,
                )

                common_valid &= (
                    triangle
                )

            block_common_count = int(
                common_valid.sum()
            )

            common_pair_count += (
                block_common_count
            )

            if (
                block_common_count
                ==
                0
            ):

                continue

            # =================================================
            # Exact industry baseline among COMMON valid pairs
            # =================================================

            industry_valid = (
                (ind1[
                    i0:i1
                ][
                    :,
                    None
                ] >= 0)
                &
                (ind1[
                    j0:j1
                ][
                    None,
                    :
                ] >= 0)
            )

            labeled_common = (
                common_valid
                &
                industry_valid
            )

            common_industry_pair_count += int(
                labeled_common.sum()
            )

            same_industry_matrix = (
                ind1[
                    i0:i1
                ][
                    :,
                    None
                ]
                ==
                ind1[
                    j0:j1
                ][
                    None,
                    :
                ]
            )

            common_same_industry_count += int(
                (
                    labeled_common
                    &
                    same_industry_matrix
                ).sum()
            )

            # =================================================
            # RAW positive candidates
            # =================================================

            raw_positive = (
                common_valid
                &
                (
                    raw_corr
                    >
                    0
                )
            )

            raw_positive_count += int(
                raw_positive.sum()
            )

            if raw_positive.any():

                rr, cc = np.nonzero(
                    raw_positive
                )

                raw_i = (
                    common_master_idx[
                        i0
                        +
                        rr
                    ]
                )

                raw_j = (
                    common_master_idx[
                        j0
                        +
                        cc
                    ]
                )

                values = (
                    raw_corr[
                        raw_positive
                    ]
                )

                raw_top.add(
                    score=
                        values,

                    corr=
                        values,

                    i=
                        raw_i,

                    j=
                        raw_j,
                )

            # =================================================
            # RESIDUAL positive candidates
            # =================================================

            residual_positive = (
                common_valid
                &
                (
                    residual_corr
                    >
                    0
                )
            )

            residual_positive_count += int(
                residual_positive.sum()
            )

            if residual_positive.any():

                rr, cc = np.nonzero(
                    residual_positive
                )

                residual_i = (
                    common_master_idx[
                        i0
                        +
                        rr
                    ]
                )

                residual_j = (
                    common_master_idx[
                        j0
                        +
                        cc
                    ]
                )

                values = (
                    residual_corr[
                        residual_positive
                    ]
                )

                residual_top.add(
                    score=
                        values,

                    corr=
                        values,

                    i=
                        residual_i,

                    j=
                        residual_j,
                )

            if (
                block_job == 1
                or
                block_job % 10 == 0
                or
                block_job
                ==
                total_block_jobs
            ):

                print(
                    f"  Block "
                    f"{block_job:3d}/"
                    f"{total_block_jobs:3d} | "
                    f"common pairs so far="
                    f"{common_pair_count:,}"
                )

    # ========================================================
    # 13.5 Exact edge budgets
    # ========================================================

    if common_pair_count <= 0:

        raise RuntimeError(
            "No common valid pair was found."
        )

    density_rows = []

    edge_budgets = {}

    for q in q_grid:

        K = int(
            math.floor(
                q
                *
                common_pair_count
            )
        )

        K = max(
            K,
            1,
        )

        edge_budgets[
            q
        ] = K

    max_required_k = max(
        edge_budgets.values()
    )

    if (
        raw_positive_count
        <
        max_required_k
    ):

        raise RuntimeError(
            "Raw positive pairs are insufficient "
            "for the requested maximum density."
        )

    if (
        residual_positive_count
        <
        max_required_k
    ):

        raise RuntimeError(
            "Residual positive pairs are insufficient "
            "for the requested maximum density."
        )

    # ========================================================
    # 13.6 Final top edge tables
    # ========================================================

    raw_edges = (
        raw_top.finalize()
    )

    residual_edges = (
        residual_top.finalize()
    )

    if len(
        raw_edges
    ) < max_required_k:

        raise RuntimeError(
            "Raw accumulator retained fewer edges "
            "than required."
        )

    if len(
        residual_edges
    ) < max_required_k:

        raise RuntimeError(
            "Residual accumulator retained fewer edges "
            "than required."
        )

    raw_edges = (
        raw_edges
        .iloc[
            :max_required_k
        ]
        .copy()
    )

    residual_edges = (
        residual_edges
        .iloc[
            :max_required_k
        ]
        .copy()
    )

    # ========================================================
    # Add compact metadata to edge master
    # ========================================================

    ind1_all = (
        industry_snapshots[
            "industry_id1"
        ][
            date
        ]
    )

    ind2_all = (
        industry_snapshots[
            "industry_id2"
        ][
            date
        ]
    )

    def decorate_edges(
        edges,
        network_id,
    ):

        edges = edges.copy()

        mi = (
            edges[
                "master_index_i"
            ]
            .to_numpy(
                dtype=np.int32
            )
        )

        mj = (
            edges[
                "master_index_j"
            ]
            .to_numpy(
                dtype=np.int32
            )
        )

        edges[
            "analysis_date"
        ] = date

        edges[
            "window"
        ] = (
            np.int16(
                window
            )
        )

        edges[
            "network_id"
        ] = network_id

        edges[
            "edge_weight"
        ] = (
            edges[
                "correlation"
            ]
            .astype(
                np.float32
            )
        )

        edges[
            "edge_sign"
        ] = np.int8(
            1
        )

        edges[
            "rank_fraction_common"
        ] = (
            edges[
                "rank"
            ]
            .to_numpy(
                dtype=float
            )
            /
            common_pair_count
        ).astype(
            np.float32
        )

        edges[
            "industry_id1_i"
        ] = (
            ind1_all[
                mi
            ]
        )

        edges[
            "industry_id1_j"
        ] = (
            ind1_all[
                mj
            ]
        )

        edges[
            "same_industry_id1"
        ] = (
            (
                ind1_all[
                    mi
                ]
                >=
                0
            )
            &
            (
                ind1_all[
                    mj
                ]
                >=
                0
            )
            &
            (
                ind1_all[
                    mi
                ]
                ==
                ind1_all[
                    mj
                ]
            )
        )

        edges[
            "industry_id2_i"
        ] = (
            ind2_all[
                mi
            ]
        )

        edges[
            "industry_id2_j"
        ] = (
            ind2_all[
                mj
            ]
        )

        edges[
            "same_industry_id2"
        ] = (
            (
                ind2_all[
                    mi
                ]
                >=
                0
            )
            &
            (
                ind2_all[
                    mj
                ]
                >=
                0
            )
            &
            (
                ind2_all[
                    mi
                ]
                ==
                ind2_all[
                    mj
                ]
            )
        )

        return edges

    raw_edges = decorate_edges(
        raw_edges,
        "B0_RAW_POSITIVE",
    )

    residual_edges = decorate_edges(
        residual_edges,
        "M1_RESIDUAL_POSITIVE",
    )

    # ========================================================
    # 13.7 Common node table
    # ========================================================

    node_df = (
        master.iloc[
            common_master_idx
        ][
            [
                "master_index",
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

    node_df[
        "analysis_date"
    ] = date

    node_df[
        "window"
    ] = (
        np.int16(
            window
        )
    )

    node_df[
        "market_beta"
    ] = (
        beta_common.astype(
            np.float32
        )
    )

    node_df[
        "market_r2"
    ] = (
        r2_common.astype(
            np.float32
        )
    )

    node_df[
        "industry_id1"
    ] = (
        ind1.astype(
            np.int16
        )
    )

    node_df[
        "industry_id2"
    ] = (
        ind2.astype(
            np.int16
        )
    )

    # ========================================================
    # 13.8 Save compact max-density edge masters
    # ========================================================

    raw_path = (
        EDGE_ROOT
        /
        "B0_RAW_POSITIVE"
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )

    residual_path = (
        EDGE_ROOT
        /
        "M1_RESIDUAL_POSITIVE"
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )

    node_path = (
        NODE_ROOT
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )

    atomic_to_parquet(
        raw_edges,
        raw_path,
    )

    atomic_to_parquet(
        residual_edges,
        residual_path,
    )

    atomic_to_parquet(
        node_df,
        node_path,
    )

    # ========================================================
    # 13.9 Density-level QA + overlap
    # ========================================================

    industry_baseline = np.nan

    if (
        common_industry_pair_count
        >
        0
    ):

        industry_baseline = (
            common_same_industry_count
            /
            common_industry_pair_count
        )

    for q in q_grid:

        K = (
            edge_budgets[
                q
            ]
        )

        raw_q = (
            raw_edges.iloc[
                :K
            ]
        )

        residual_q = (
            residual_edges.iloc[
                :K
            ]
        )

        raw_keys = (
            raw_q[
                "master_index_i"
            ]
            .to_numpy(
                dtype=np.int64
            )
            *
            len(
                master
            )
            +
            raw_q[
                "master_index_j"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        residual_keys = (
            residual_q[
                "master_index_i"
            ]
            .to_numpy(
                dtype=np.int64
            )
            *
            len(
                master
            )
            +
            residual_q[
                "master_index_j"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        overlap_count = len(
            np.intersect1d(
                raw_keys,
                residual_keys,
                assume_unique=True,
            )
        )

        edge_overlap_fraction = (
            overlap_count
            /
            K
        )

        edge_jaccard = (
            overlap_count
            /
            (
                2 * K
                -
                overlap_count
            )
        )

        raw_same_share = float(
            raw_q[
                "same_industry_id1"
            ]
            .mean()
        )

        residual_same_share = float(
            residual_q[
                "same_industry_id1"
            ]
            .mean()
        )

        raw_enrichment = np.nan
        residual_enrichment = np.nan

        if (
            np.isfinite(
                industry_baseline
            )
            and
            industry_baseline
            >
            0
        ):

            raw_enrichment = (
                raw_same_share
                /
                industry_baseline
            )

            residual_enrichment = (
                residual_same_share
                /
                industry_baseline
            )

        density_rows.append(
            {
                "analysis_date":
                    date_string,

                "window":
                    window,

                "density_fraction":
                    q,

                "density_percent":
                    100 * q,

                "common_node_count":
                    p,

                "common_valid_pair_count":
                    common_pair_count,

                "edge_budget_k":
                    K,

                "raw_positive_pair_count":
                    raw_positive_count,

                "residual_positive_pair_count":
                    residual_positive_count,

                "raw_edge_threshold":
                    float(
                        raw_q[
                            "correlation"
                        ]
                        .iloc[
                            -1
                        ]
                    ),

                "residual_edge_threshold":
                    float(
                        residual_q[
                            "correlation"
                        ]
                        .iloc[
                            -1
                        ]
                    ),

                "raw_residual_overlap_count":
                    overlap_count,

                "raw_residual_overlap_fraction":
                    edge_overlap_fraction,

                "raw_residual_jaccard":
                    edge_jaccard,

                "common_industry_labeled_pair_count":
                    common_industry_pair_count,

                "common_same_industry_pair_share":
                    industry_baseline,

                "raw_same_industry_edge_share":
                    raw_same_share,

                "residual_same_industry_edge_share":
                    residual_same_share,

                "raw_industry_enrichment":
                    raw_enrichment,

                "residual_industry_enrichment":
                    residual_enrichment,
            }
        )

    # ========================================================
    # 13.10 Save completion marker
    # ========================================================

    job_summary = {

        "analysis_date":
            date_string,

        "window":
            window,

        "window_start":
            str(
                trade_dates[
                    start_idx
                ].date()
            ),

        "required_stock_days":
            required_stock_days,

        "required_pair_days":
            required_pair_days,

        "pit_stock_count":
            int(
                pit.sum()
            ),

        "raw_eligible_stock_count":
            int(
                len(
                    eligible_idx
                )
            ),

        "common_node_count":
            p,

        "total_possible_common_node_pairs":
            int(
                total_possible_pairs
            ),

        "common_valid_pair_count":
            int(
                common_pair_count
            ),

        "common_valid_pair_ratio":
            float(
                common_pair_count
                /
                total_possible_pairs
            ),

        "raw_positive_pair_count":
            int(
                raw_positive_count
            ),

        "residual_positive_pair_count":
            int(
                residual_positive_count
            ),

        "market_beta_mean":
            float(
                np.nanmean(
                    beta_common
                )
            ),

        "market_beta_median":
            float(
                np.nanmedian(
                    beta_common
                )
            ),

        "market_r2_mean":
            float(
                np.nanmean(
                    r2_common
                )
            ),

        "market_r2_median":
            float(
                np.nanmedian(
                    r2_common
                )
            ),

        "common_industry_pair_count":
            int(
                common_industry_pair_count
            ),

        "common_same_industry_pair_count":
            int(
                common_same_industry_count
            ),

        "common_same_industry_pair_share":
            (
                float(
                    industry_baseline
                )
                if np.isfinite(
                    industry_baseline
                )
                else None
            ),

        "raw_edge_master_path":
            str(
                raw_path
            ),

        "residual_edge_master_path":
            str(
                residual_path
            ),

        "node_path":
            str(
                node_path
            ),
    }

    marker_payload = {

        "job_summary":
            job_summary,

        "density_rows":
            density_rows,
    }

    temp_marker = Path(
        str(
            marker_path
        )
        +
        ".tmp"
    )

    with open(
        temp_marker,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            marker_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temp_marker,
        marker_path,
    )

    print()
    print(
        f"DONE W={window} "
        f"{date_string}"
    )

    print(
        f"  Common nodes : {p:,}"
    )

    print(
        f"  Common pairs : "
        f"{common_pair_count:,}"
    )

    print(
        f"  Raw + pairs  : "
        f"{raw_positive_count:,}"
    )

    print(
        f"  Res + pairs  : "
        f"{residual_positive_count:,}"
    )


# ============================================================
# 14. Consolidate all completion markers
# ============================================================

def consolidate_markers():

    job_rows = []

    density_rows = []

    marker_files = sorted(
        MARKER_ROOT.glob(
            "W*_*.json"
        )
    )

    for path in marker_files:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            payload = json.load(
                f
            )

        job_rows.append(
            payload[
                "job_summary"
            ]
        )

        density_rows.extend(
            payload[
                "density_rows"
            ]
        )

    job_df = pd.DataFrame(
        job_rows
    )

    density_df = pd.DataFrame(
        density_rows
    )

    if not job_df.empty:

        job_df[
            "analysis_date"
        ] = pd.to_datetime(
            job_df[
                "analysis_date"
            ]
        )

        job_df = (
            job_df.sort_values(
                [
                    "window",
                    "analysis_date",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        job_df.to_csv(
            OUTPUT_DIR
            / "network_construction_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if not density_df.empty:

        density_df[
            "analysis_date"
        ] = pd.to_datetime(
            density_df[
                "analysis_date"
            ]
        )

        density_df = (
            density_df.sort_values(
                [
                    "window",
                    "density_fraction",
                    "analysis_date",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        density_df.to_csv(
            OUTPUT_DIR
            / "matched_density_network_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return (
        job_df,
        density_df,
    )


# ============================================================
# 15. Save stock mapping once
# ============================================================

def save_master_mapping(
    master,
):

    mapping = (
        master[
            [
                "master_index",
                "security_id",
                "stock_code",
                "stock_name",
                "exchange",
                "list_date",
                "delist_date",
            ]
        ]
        .copy()
    )

    mapping.to_parquet(
        OUTPUT_DIR
        / "stock_master_mapping.parquet",
        index=False,
        compression=
            PARQUET_COMPRESSION,
    )


# ============================================================
# 16. Metadata
# ============================================================

def save_metadata(
    return_column,
    windows,
    stock_valid_ratio,
    pair_valid_ratio,
    q_grid,
    primary_q,
):

    metadata = {

        "research_day":
            4,

        "step":
            (
                "Step2_Construct_Matched_Density_"
                "Raw_and_Market_Residual_Networks"
            ),

        "output_directory":
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

        "density_grid":
            q_grid,

        "primary_density":
            primary_q,

        "pairwise_correlation":
            (
                "Exact pairwise-complete Pearson correlation "
                "calculated blockwise."
            ),

        "market_factor":
            (
                "Daily PIT equal-weight market factor "
                "with stock-specific leave-one-out adjustment."
            ),

        "node_comparison":
            (
                "Raw and Residual networks use the "
                "same common node universe."
            ),

        "pair_comparison":
            (
                "Raw and Residual networks use the "
                "intersection of their valid pair universes."
            ),

        "edge_budget":
            (
                "K=floor(q*M_common_valid_pairs)"
            ),

        "primary_network_type":
            (
                "Positive-correlation network."
            ),

        "storage_strategy":
            (
                "Only the maximum requested density edge master "
                "is stored. Lower density networks are nested "
                "subsets selected by edge rank."
            ),

        "important_notes": [

            (
                "Industry labels are not used for edge selection."
            ),

            (
                "Edge tables store compact integer master_index "
                "identifiers; stock metadata are stored separately "
                "in stock_master_mapping.parquet."
            ),

            (
                "The script supports restart/resume through "
                "per-date completion markers."
            ),

            (
                "For a first smoke test, set "
                "TEST_LATEST_ONLY=True. Formal results require "
                "TEST_LATEST_ONLY=False."
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        / "step2_network_construction_metadata.json",
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
# 17. Main
# ============================================================

def main():

    print("=" * 80)
    print(
        "M1 - Research Day 4 - Step 2"
    )
    print(
        "Construct Matched-Density Raw & "
        "Market-Residual Networks"
    )
    print("=" * 80)

    (
        design_config,
        return_column,
        windows,
        stock_valid_ratio,
        pair_valid_ratio,
        q_grid,
        primary_q,
    ) = load_network_design()

    print()
    print("[1] Frozen design")

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
        f"Density grid  : {q_grid}"
    )

    print(
        f"Primary q     : {primary_q}"
    )

    print(
        f"Test mode     : {TEST_LATEST_ONLY}"
    )

    trade_dates = (
        load_trade_dates()
    )

    master = (
        load_stock_master()
    )

    analysis_dates = (
        load_analysis_dates(
            windows
        )
    )

    save_master_mapping(
        master
    )

    # ========================================================
    # Full return matrix
    # ========================================================

    print()
    print(
        "[2] Build return matrix + industries"
    )

    (
        returns,
        industry_snapshots,
    ) = (
        build_market_data(
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

    # ========================================================
    # Daily market factor
    # ========================================================

    print()
    print(
        "[3] Build PIT market factor components"
    )

    (
        market_sum,
        market_count,
    ) = (
        build_daily_pit_market_components(
            trade_dates=
                trade_dates,

            master=
                master,

            returns=
                returns,
        )
    )

    # ========================================================
    # Construct all Date × Window networks
    # ========================================================

    print()
    print(
        "[4] Construct matched-density networks"
    )

    total_jobs = len(
        analysis_dates
    )

    for job_no, row in enumerate(
        analysis_dates.itertuples(
            index=False
        ),
        start=1,
    ):

        date = pd.Timestamp(
            row.analysis_date
        )

        window = int(
            row.window
        )

        print()
        print("=" * 80)

        print(
            f"Job {job_no}/{total_jobs}"
        )

        print(
            f"W={window} | "
            f"Date={date.date()}"
        )

        print("=" * 80)

        construct_one_job(
            date=
                date,

            window=
                window,

            trade_dates=
                trade_dates,

            master=
                master,

            returns=
                returns,

            industry_snapshots=
                industry_snapshots,

            market_sum=
                market_sum,

            market_count=
                market_count,

            stock_valid_ratio=
                stock_valid_ratio,

            pair_valid_ratio=
                pair_valid_ratio,

            q_grid=
                q_grid,
        )

        # ----------------------------------------------------
        # Continuously refresh summary files.
        #
        # If the run is interrupted, completed jobs remain usable.
        # ----------------------------------------------------

        consolidate_markers()

    # ========================================================
    # Final summaries
    # ========================================================

    print()
    print(
        "[5] Consolidate final outputs"
    )

    (
        job_summary,
        density_summary,
    ) = (
        consolidate_markers()
    )

    save_metadata(
        return_column=
            return_column,

        windows=
            windows,

        stock_valid_ratio=
            stock_valid_ratio,

        pair_valid_ratio=
            pair_valid_ratio,

        q_grid=
            q_grid,

        primary_q=
            primary_q,
    )

    print()
    print("=" * 80)
    print(
        "Day 4 Step 2 Complete"
    )
    print("=" * 80)

    if not density_summary.empty:

        primary = (
            density_summary[
                np.isclose(
                    density_summary[
                        "density_fraction"
                    ],
                    primary_q,
                )
            ]
        )

        print()
        print(
            "Primary q network summary:"
        )

        show_cols = [
            "analysis_date",
            "window",
            "common_node_count",
            "common_valid_pair_count",
            "edge_budget_k",
            "raw_edge_threshold",
            "residual_edge_threshold",
            "raw_residual_overlap_fraction",
            "raw_industry_enrichment",
            "residual_industry_enrichment",
        ]

        print(
            primary[
                show_cols
            ]
            .tail(12)
            .to_string(
                index=False
            )
        )

    print()
    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()