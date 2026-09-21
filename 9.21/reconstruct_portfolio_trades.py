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
# Day 7 Step 2
# Traditional characteristics / current information
# ------------------------------------------------------------

DAY7_STEP2_DIR = (
    OUTPUT_ROOT
    / "M1_day7"
    / "02_stage2_traditional_characteristic_controls"
)

TRADITIONAL_PANEL_PATH = (
    DAY7_STEP2_DIR
    / "traditional_characteristic_panel.parquet"
)

# Day-7 writes controls at stock/date level; network factors and
# their actual estimation windows remain in the Day-5 feature panel.
FACTOR_PANEL_PATH = (
    OUTPUT_ROOT
    / "M1_day5"
    / "05_step5_community_features"
    / "community_bridge_feature_panel.parquet"
)


# ------------------------------------------------------------
# Day 6 Step 2
# Forward total return is used ONLY to drift PREVIOUS weights
# from the previous formation date to the current date.
#
# It is NOT used to form the current portfolio.
# ------------------------------------------------------------

FORWARD_LABEL_DIR = (
    OUTPUT_ROOT
    / "M1_day6"
    / "02_step2_forward_labels"
)


# ------------------------------------------------------------
# Day 7 Step 4
# Optional diagnostic comparison only.
# ------------------------------------------------------------

STEP4_ECONOMIC_PATH = (
    OUTPUT_ROOT
    / "M1_day7"
    / "04_stage4_turnover_transaction_cost"
    / "turnover_economic_summary.csv"
)


# ------------------------------------------------------------
# Day 8 Step 2
# ------------------------------------------------------------

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day8"
    / "02_stage2_trade_reconstruction"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET_WEIGHT_PATH = (
    OUTPUT_DIR
    / "portfolio_target_weights.parquet"
)

TRADE_PANEL_PATH = (
    OUTPUT_DIR
    / "portfolio_trade_weight_panel.parquet"
)

REBALANCE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "portfolio_rebalance_summary.csv"
)

AUM_SCENARIO_SUMMARY_PATH = (
    OUTPUT_DIR
    / "portfolio_trade_value_scenario_summary.csv"
)

SORT_QA_PATH = (
    OUTPUT_DIR
    / "portfolio_sort_qa.csv"
)

STEP4_COMPARISON_PATH = (
    OUTPUT_DIR
    / "step4_turnover_reconstruction_comparison.csv"
)

FORMAL_QA_PATH = (
    OUTPUT_DIR
    / "portfolio_trade_reconstruction_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day8_step2_metadata.json"
)


# ============================================================
# 1. Frozen research design
# ============================================================

WINDOWS = [
    60,
    120,
    252,
]


FACTOR_NAMES = [

    "residual_degree_percentile",

    "delta_degree_percentile",

    "cross_industry_degree_percentile",

    "cross_industry_degree_ratio",

    "delta_residual_degree_percentile_1m",

    "delta_cross_industry_degree_percentile_1m",

    "neighbor_jaccard_1m",

    "neighbor_retention_1m",

    "outside_community_degree_percentile",
]


# ------------------------------------------------------------
# Main Day-8 capacity analysis:
#
# use the FINAL specification after traditional controls.
#
# This is NOT factor/window selection.
# All 9 factors and all 3 windows remain.
#
# If later you want RAW / IND+SIZE capacity comparison,
# set this True.
# ------------------------------------------------------------

INCLUDE_COMPARISON_METHODS = False


if INCLUDE_COMPARISON_METHODS:

    METHODS = [

        "RAW_COMMON",

        "INDUSTRY_SIZE_COMMON",

        "FULL_CHARACTERISTIC_NEUTRAL",
    ]

else:

    METHODS = [

        "FULL_CHARACTERISTIC_NEUTRAL",
    ]


# ------------------------------------------------------------
# Factor sort
# ------------------------------------------------------------

N_PORTFOLIOS = 5

Q1_LABEL = 1
Q5_LABEL = 5

MIN_CROSS_SECTION_OBS = 100


# ------------------------------------------------------------
# Traditional controls
# ------------------------------------------------------------

INDUSTRY_COLUMN = (
    "industry_id1"
)

CONTROL_COLUMNS = [

    "log_market_value",

    "momentum_120_20",

    "reversal_20",

    "volatility_60",

    "log_turnover_20",
]


SIZE_COLUMN = (
    "log_market_value"
)


# ------------------------------------------------------------
# Must match Day-7 Step-2 preprocessing.
# ------------------------------------------------------------

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

ZSCORE_DDOF = 0


# ------------------------------------------------------------
# AUM scenarios
#
# Strategy NAV:
# each Q1 / Q5 leg has notional = NAV.
#
# 1e8  = 1 亿元
# 5e8  = 5 亿元
# 1e9  = 10 亿元
# 2e9  = 20 亿元
# 5e9  = 50 亿元
# 1e10 = 100 亿元
# ------------------------------------------------------------

AUM_SCENARIOS_CNY = [

    1e8,

    5e8,

    1e9,

    2e9,

    5e9,

    1e10,
]


# ------------------------------------------------------------
# Instead of exploding the stock-level table by all AUMs,
# store monetary trade values for two convenient reference NAVs.
#
# Any other NAV is a linear rescaling.
# ------------------------------------------------------------

REFERENCE_NAV_100M_CNY = 1e8

REFERENCE_NAV_1B_CNY = 1e9


# ============================================================
# 2. Utilities
# ============================================================

def save_json_atomic(
    obj,
    path: Path,
):

    temp = Path(
        str(path)
        +
        ".tmp"
    )

    with open(
        temp,
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
        temp,
        path,
    )


def save_csv_atomic(
    df: pd.DataFrame,
    path: Path,
):

    temp = Path(
        str(path)
        +
        ".tmp"
    )

    df.to_csv(
        temp,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        temp,
        path,
    )


def save_parquet_atomic(
    df: pd.DataFrame,
    path: Path,
):

    temp = Path(
        str(path)
        +
        ".tmp"
    )

    df.to_parquet(
        temp,
        index=False,
        compression="snappy",
    )

    os.replace(
        temp,
        path,
    )


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


def normalize_security_id(
    x,
):

    return (
        x.astype("string")
        .str.strip()
    )


def require_columns(
    df,
    columns,
    name,
):

    missing = [

        c
        for c in columns
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{name} missing required columns:\n"
            f"{missing}"
        )


def parquet_columns(
    path: Path,
):

    return (
        pq.ParquetFile(
            path
        )
        .schema
        .names
    )


# ============================================================
# 3. Find forward-label file automatically
# ============================================================

def find_forward_label_file():

    required = {

        "analysis_date",
        "security_id",
        "window",
        "future_total_return",
    }

    candidates = []

    for path in sorted(
        FORWARD_LABEL_DIR.rglob(
            "*.parquet"
        )
    ):

        try:

            cols = set(
                parquet_columns(
                    path
                )
            )

        except Exception:

            continue

        if required.issubset(
            cols
        ):

            score = 0

            name = (
                path.name
                .lower()
            )

            if "forward" in name:

                score += 2

            if "label" in name:

                score += 2

            candidates.append(
                (
                    score,
                    path,
                )
            )

    for path in sorted(
        FORWARD_LABEL_DIR.rglob(
            "*.csv"
        )
    ):

        try:

            cols = set(
                pd.read_csv(
                    path,
                    nrows=0,
                ).columns
            )

        except Exception:

            continue

        if required.issubset(
            cols
        ):

            score = 0

            name = (
                path.name
                .lower()
            )

            if "forward" in name:

                score += 2

            if "label" in name:

                score += 2

            candidates.append(
                (
                    score,
                    path,
                )
            )

    if not candidates:

        raise FileNotFoundError(
            "Cannot find a Day-6 forward-label file "
            "containing:\n"
            f"{sorted(required)}\n"
            f"under:\n{FORWARD_LABEL_DIR}"
        )

    candidates.sort(
        key=lambda x: (
            # Prefer the active panel over recursively discovered archives.
            x[1] != FORWARD_LABEL_DIR / "forward_label_panel.parquet",
            -x[0],
            str(
                x[1]
            ),
        )
    )

    selected = (
        candidates[0][1]
    )

    print(
        "Forward-label source:"
    )

    print(
        selected
    )

    return selected


# ============================================================
# 4. Load current-information panel
#
# Supports separate stock/date controls plus the Day-5 factor panel,
# as well as two already-combined layouts:
#
# (A) LONG:
#     factor_name / factor_value
#
# (B) WIDE:
#     one column per factor
# ============================================================

def load_current_information_panel():

    if (
        not
        TRADITIONAL_PANEL_PATH.exists()
    ):

        raise FileNotFoundError(
            f"Missing panel:\n"
            f"{TRADITIONAL_PANEL_PATH}"
        )

    df = pd.read_parquet(
        TRADITIONAL_PANEL_PATH
    )

    if "window" not in df.columns:
        # Reconstruct the same many-to-one current-information join used
        # by Day 7. Never fabricate windows or join future return labels
        # here: outcome availability must not select today's portfolio.
        stock_keys = ["analysis_date", "security_id"]
        control_columns = stock_keys + [INDUSTRY_COLUMN, *CONTROL_COLUMNS]
        require_columns(df, control_columns, "traditional_characteristic_panel")
        if not FACTOR_PANEL_PATH.exists():
            raise FileNotFoundError(f"Missing network factor panel:\n{FACTOR_PANEL_PATH}")
        factor_columns = stock_keys + ["window", *FACTOR_NAMES]
        missing = sorted(set(factor_columns) - set(parquet_columns(FACTOR_PANEL_PATH)))
        if missing:
            raise RuntimeError(f"Network factor panel missing required columns:\n{missing}")
        factors = pd.read_parquet(FACTOR_PANEL_PATH, columns=factor_columns)
        controls = df[control_columns].copy()
        for frame in (factors, controls):
            frame["analysis_date"] = pd.to_datetime(frame["analysis_date"], errors="raise")
            frame["security_id"] = normalize_security_id(frame["security_id"])
            if frame[stock_keys].isna().any().any():
                raise RuntimeError("Missing date/security keys in current-information inputs.")
        if controls.duplicated(stock_keys).any():
            raise RuntimeError("Duplicate stock-date keys in traditional_characteristic_panel.")
        factors["window"] = pd.to_numeric(factors["window"], errors="raise").astype(int)
        if factors.duplicated(stock_keys + ["window"]).any():
            raise RuntimeError("Duplicate stock-date-window keys in network factor panel.")
        df = factors.merge(
            controls,
            on=stock_keys,
            how="left",
            validate="many_to_one",
        )
        print(f"Current network factor source: {FACTOR_PANEL_PATH}")

    base_required = [

        "analysis_date",

        "security_id",

        "window",

        INDUSTRY_COLUMN,

        *CONTROL_COLUMNS,
    ]

    require_columns(
        df,
        base_required,
        "traditional_characteristic_panel",
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
        "security_id"
    ] = normalize_security_id(
        df[
            "security_id"
        ]
    )

    df[
        "window"
    ] = pd.to_numeric(
        df[
            "window"
        ],
        errors="raise",
    ).astype(int)

    df = df[
        df[
            "window"
        ]
        .isin(
            WINDOWS
        )
    ].copy()

    # --------------------------------------------------------
    # LONG layout
    # --------------------------------------------------------

    if (
        "factor_name"
        in
        df.columns
        and
        "factor_value"
        in
        df.columns
    ):

        long_df = df[
            df[
                "factor_name"
            ]
            .isin(
                FACTOR_NAMES
            )
        ].copy()

    # --------------------------------------------------------
    # WIDE layout
    # --------------------------------------------------------

    else:

        missing_factors = [

            factor
            for factor
            in FACTOR_NAMES
            if factor not in df.columns
        ]

        if missing_factors:

            raise RuntimeError(
                "Panel is neither a valid LONG factor panel "
                "nor a WIDE panel containing all frozen factors.\n"
                f"Missing factor columns:\n"
                f"{missing_factors}"
            )

        id_vars = [

            "analysis_date",

            "security_id",

            "window",

            INDUSTRY_COLUMN,

            *CONTROL_COLUMNS,
        ]

        long_df = df[
            id_vars
            +
            FACTOR_NAMES
        ].melt(

            id_vars=id_vars,

            value_vars=FACTOR_NAMES,

            var_name="factor_name",

            value_name="factor_value",
        )

    long_df[
        "factor_value"
    ] = pd.to_numeric(
        long_df[
            "factor_value"
        ],
        errors="coerce",
    )

    for column in CONTROL_COLUMNS:

        long_df[
            column
        ] = pd.to_numeric(
            long_df[
                column
            ],
            errors="coerce",
        )

    long_df[
        INDUSTRY_COLUMN
    ] = (
        long_df[
            INDUSTRY_COLUMN
        ]
        .astype("string")
        .str.strip()
    )

    duplicate = long_df[
        [
            "analysis_date",
            "window",
            "factor_name",
            "security_id",
        ]
    ].duplicated(
        keep=False
    )

    if duplicate.any():

        raise RuntimeError(
            "Duplicate current-information "
            "factor-stock rows detected.\n"
            f"Count={int(duplicate.sum())}"
        )

    return (
        long_df.sort_values(
            [
                "analysis_date",
                "window",
                "factor_name",
                "security_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 5. Traditional-control transformation
# ============================================================

def winsorize_and_zscore(
    x: pd.Series,
):

    x = pd.to_numeric(
        x,
        errors="coerce",
    ).astype(float)

    lower = x.quantile(
        WINSOR_LOWER
    )

    upper = x.quantile(
        WINSOR_UPPER
    )

    clipped = x.clip(
        lower=lower,
        upper=upper,
    )

    mean = clipped.mean()

    std = clipped.std(
        ddof=ZSCORE_DDOF
    )

    if (
        not np.isfinite(
            std
        )
        or
        std
        <=
        1e-14
    ):

        return None

    return (
        clipped
        -
        mean
    ) / std


def residualize_with_industry_fe(
    y: pd.Series,
    controls: pd.DataFrame,
    industry: pd.Series,
):

    temp = pd.DataFrame(
        {
            "_y":
                pd.to_numeric(
                    y,
                    errors="coerce",
                ),

            "_industry":
                industry.astype(
                    "string"
                ),
        },
        index=y.index,
    )

    for column in controls.columns:

        temp[
            column
        ] = pd.to_numeric(
            controls[
                column
            ],
            errors="coerce",
        )

    # --------------------------------------------------------
    # FWL:
    # remove industry-group means from y and X.
    # --------------------------------------------------------

    y_dm = (

        temp[
            "_y"
        ]

        -

        temp.groupby(
            "_industry",
            observed=True,
        )[
            "_y"
        ]
        .transform(
            "mean"
        )
    )

    X_dm = pd.DataFrame(
        index=temp.index
    )

    for column in controls.columns:

        X_dm[
            column
        ] = (

            temp[
                column
            ]

            -

            temp.groupby(
                "_industry",
                observed=True,
            )[
                column
            ]
            .transform(
                "mean"
            )
        )

    X = X_dm.to_numpy(
        dtype=float
    )

    Y = y_dm.to_numpy(
        dtype=float
    )

    if X.shape[1] == 0:

        return pd.Series(
            Y,
            index=y.index,
        )

    beta, *_ = np.linalg.lstsq(
        X,
        Y,
        rcond=None,
    )

    residual = (

        Y

        -

        X
        @
        beta
    )

    return pd.Series(
        residual,
        index=y.index,
    )


# ============================================================
# 6. Conservative quintile sort
# ============================================================

def assign_quintiles(
    signal: pd.Series,
):

    signal = pd.to_numeric(
        signal,
        errors="coerce",
    )

    if (
        signal.notna().sum()
        <
        MIN_CROSS_SECTION_OBS
    ):

        return (
            None,
            "INSUFFICIENT_OBS",
        )

    if (
        signal.nunique(
            dropna=True
        )
        <
        N_PORTFOLIOS
    ):

        return (
            None,
            "INSUFFICIENT_UNIQUE_VALUES",
        )

    try:

        bins = pd.qcut(

            signal,

            q=N_PORTFOLIOS,

            labels=False,

            duplicates="drop",
        )

    except Exception:

        return (
            None,
            "SORT_FAILURE",
        )

    unique_bins = sorted(
        pd.Series(
            bins
        )
        .dropna()
        .unique()
        .tolist()
    )

    if unique_bins != [
        0,
        1,
        2,
        3,
        4,
    ]:

        return (
            None,
            "SORT_FAILURE",
        )

    quintile = (
        pd.Series(
            bins,
            index=signal.index,
        )
        +
        1
    )

    return (
        quintile,
        "PASS",
    )


# ============================================================
# 7. Build target portfolios
#
# IMPORTANT:
#
# Current target portfolios depend ONLY on current information.
# No future-return availability is used.
# ============================================================

def build_target_portfolios(
    panel,
):

    target_frames = []

    qa_rows = []

    group_columns = [

        "analysis_date",

        "window",

        "factor_name",
    ]

    groups = panel.groupby(
        group_columns,
        sort=True,
    )

    total_groups = groups.ngroups

    for group_no, (
        keys,
        group,
    ) in enumerate(
        groups,
        start=1,
    ):

        (
            analysis_date,
            window,
            factor_name,
        ) = keys

        if (
            group_no == 1
            or
            group_no % 250 == 0
            or
            group_no == total_groups
        ):

            print(
                f"Portfolio formation: "
                f"{group_no}/{total_groups}"
            )

        # ----------------------------------------------------
        # Common current-information sample.
        #
        # NO future outcome condition here.
        # ----------------------------------------------------

        valid = (

            group[
                "factor_value"
            ]
            .notna()

            &

            group[
                INDUSTRY_COLUMN
            ]
            .notna()
        )

        for column in CONTROL_COLUMNS:

            valid &= (
                group[
                    column
                ]
                .notna()
            )

        common = (
            group.loc[
                valid
            ]
            .copy()
        )

        if (
            len(
                common
            )
            <
            MIN_CROSS_SECTION_OBS
        ):

            for method in METHODS:

                qa_rows.append(
                    {
                        "analysis_date":
                            analysis_date,

                        "window":
                            int(
                                window
                            ),

                        "factor_name":
                            factor_name,

                        "method":
                            method,

                        "common_sample_n":
                            int(
                                len(
                                    common
                                )
                            ),

                        "status":
                            "INSUFFICIENT_OBS",

                        "q1_n":
                            0,

                        "q5_n":
                            0,
                    }
                )

            continue

        # ----------------------------------------------------
        # Controls:
        # 1/99 winsorization then cross-sectional z-score.
        # ----------------------------------------------------

        transformed_controls = pd.DataFrame(
            index=common.index
        )

        transform_failure = False

        for column in CONTROL_COLUMNS:

            transformed = (
                winsorize_and_zscore(
                    common[
                        column
                    ]
                )
            )

            if transformed is None:

                transform_failure = True

                break

            transformed_controls[
                column
            ] = transformed

        if transform_failure:

            for method in METHODS:

                qa_rows.append(
                    {
                        "analysis_date":
                            analysis_date,

                        "window":
                            int(
                                window
                            ),

                        "factor_name":
                            factor_name,

                        "method":
                            method,

                        "common_sample_n":
                            int(
                                len(
                                    common
                                )
                            ),

                        "status":
                            "CONTROL_TRANSFORM_FAILURE",

                        "q1_n":
                            0,

                        "q5_n":
                            0,
                    }
                )

            continue

        signals = {}

        # ----------------------------------------------------
        # RAW common sample
        # ----------------------------------------------------

        if (
            "RAW_COMMON"
            in
            METHODS
        ):

            signals[
                "RAW_COMMON"
            ] = common[
                "factor_value"
            ].astype(float)

        # ----------------------------------------------------
        # Industry + Size neutral
        # ----------------------------------------------------

        if (
            "INDUSTRY_SIZE_COMMON"
            in
            METHODS
        ):

            signals[
                "INDUSTRY_SIZE_COMMON"
            ] = (
                residualize_with_industry_fe(

                    y=common[
                        "factor_value"
                    ],

                    controls=
                        transformed_controls[
                            [
                                SIZE_COLUMN
                            ]
                        ],

                    industry=
                        common[
                            INDUSTRY_COLUMN
                        ],
                )
            )

        # ----------------------------------------------------
        # Full traditional-characteristic neutral
        # ----------------------------------------------------

        if (
            "FULL_CHARACTERISTIC_NEUTRAL"
            in
            METHODS
        ):

            signals[
                "FULL_CHARACTERISTIC_NEUTRAL"
            ] = (
                residualize_with_industry_fe(

                    y=common[
                        "factor_value"
                    ],

                    controls=
                        transformed_controls[
                            CONTROL_COLUMNS
                        ],

                    industry=
                        common[
                            INDUSTRY_COLUMN
                        ],
                )
            )

        # ----------------------------------------------------
        # Q1 / Q5 targets
        # ----------------------------------------------------

        for method in METHODS:

            signal = signals[
                method
            ]

            quintile, status = (
                assign_quintiles(
                    signal
                )
            )

            if status != "PASS":

                qa_rows.append(
                    {
                        "analysis_date":
                            analysis_date,

                        "window":
                            int(
                                window
                            ),

                        "factor_name":
                            factor_name,

                        "method":
                            method,

                        "common_sample_n":
                            int(
                                len(
                                    common
                                )
                            ),

                        "status":
                            status,

                        "q1_n":
                            0,

                        "q5_n":
                            0,
                    }
                )

                continue

            q1_index = quintile[
                quintile
                ==
                Q1_LABEL
            ].index

            q5_index = quintile[
                quintile
                ==
                Q5_LABEL
            ].index

            q1_n = len(
                q1_index
            )

            q5_n = len(
                q5_index
            )

            qa_rows.append(
                {
                    "analysis_date":
                        analysis_date,

                    "window":
                        int(
                            window
                        ),

                    "factor_name":
                        factor_name,

                    "method":
                        method,

                    "common_sample_n":
                        int(
                            len(
                                common
                            )
                        ),

                    "status":
                        "PASS",

                    "q1_n":
                        int(
                            q1_n
                        ),

                    "q5_n":
                        int(
                            q5_n
                        ),
                }
            )

            for leg, index in [

                (
                    "Q1",
                    q1_index,
                ),

                (
                    "Q5",
                    q5_index,
                ),
            ]:

                n = len(
                    index
                )

                temp = common.loc[
                    index,
                    [
                        "analysis_date",
                        "window",
                        "factor_name",
                        "security_id",
                    ],
                ].copy()

                temp[
                    "method"
                ] = method

                temp[
                    "portfolio_leg"
                ] = leg

                temp[
                    "signal_value"
                ] = signal.loc[
                    index
                ].to_numpy(
                    dtype=float
                )

                temp[
                    "target_weight"
                ] = (
                    1.0
                    /
                    float(
                        n
                    )
                )

                target_frames.append(
                    temp
                )

    if not target_frames:

        raise RuntimeError(
            "No valid target portfolios were formed."
        )

    targets = pd.concat(
        target_frames,
        ignore_index=True,
    )

    qa = pd.DataFrame(
        qa_rows
    )

    key_columns = [

        "analysis_date",

        "window",

        "factor_name",

        "method",

        "portfolio_leg",

        "security_id",
    ]

    if targets[
        key_columns
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate target-weight rows."
        )

    return (
        targets,
        qa,
    )


# ============================================================
# 8. Load previous-period total returns
# ============================================================

def load_forward_total_returns():

    path = find_forward_label_file()

    required = [

        "analysis_date",

        "security_id",

        "window",

        "future_total_return",
    ]

    if (
        path.suffix.lower()
        ==
        ".parquet"
    ):

        available = parquet_columns(
            path
        )

        columns = required.copy()

        if (
            "next_analysis_date"
            in
            available
        ):

            columns.append(
                "next_analysis_date"
            )

        labels = pd.read_parquet(
            path,
            columns=columns,
        )

    else:

        header = (
            pd.read_csv(
                path,
                nrows=0,
            )
            .columns
            .tolist()
        )

        columns = required.copy()

        if (
            "next_analysis_date"
            in
            header
        ):

            columns.append(
                "next_analysis_date"
            )

        labels = pd.read_csv(
            path,
            usecols=columns,
            low_memory=False,
        )

    labels[
        "analysis_date"
    ] = pd.to_datetime(
        labels[
            "analysis_date"
        ],
        errors="raise",
    )

    if (
        "next_analysis_date"
        in
        labels.columns
    ):

        labels[
            "next_analysis_date"
        ] = pd.to_datetime(
            labels[
                "next_analysis_date"
            ],
            errors="coerce",
        )

    labels[
        "security_id"
    ] = normalize_security_id(
        labels[
            "security_id"
        ]
    )

    labels[
        "window"
    ] = pd.to_numeric(
        labels[
            "window"
        ],
        errors="raise",
    ).astype(int)

    labels[
        "future_total_return"
    ] = pd.to_numeric(
        labels[
            "future_total_return"
        ],
        errors="coerce",
    )

    labels = labels[
        labels[
            "window"
        ]
        .isin(
            WINDOWS
        )
    ].copy()

    # --------------------------------------------------------
    # Labels may have been repeated across factors.
    #
    # A return label is unique at:
    #
    # analysis_date × window × security.
    # --------------------------------------------------------

    key = [

        "analysis_date",

        "window",

        "security_id",
    ]

    duplicate = labels[
        key
    ].duplicated(
        keep=False
    )

    if duplicate.any():

        check = (

            labels.loc[
                duplicate
            ]
            .groupby(
                key,
                dropna=False,
            )[
                "future_total_return"
            ]
            .nunique(
                dropna=False
            )
        )

        if (
            check
            >
            1
        ).any():

            raise RuntimeError(
                "Conflicting future_total_return "
                "values for the same stock-date-window."
            )

        labels = (
            labels.drop_duplicates(
                subset=key,
                keep="first",
            )
        )

    return (
        labels.sort_values(
            key
        )
        .reset_index(
            drop=True
        ),
        path,
    )


# ============================================================
# 9. Build fast return lookup maps
# ============================================================

def build_return_maps(
    labels,
):

    return_maps = {}

    next_date_maps = {}

    for (
        analysis_date,
        window,
    ), group in labels.groupby(
        [
            "analysis_date",
            "window",
        ],
        sort=False,
    ):

        return_maps[
            (
                analysis_date,
                int(
                    window
                ),
            )
        ] = pd.Series(

            group[
                "future_total_return"
            ]
            .to_numpy(
                dtype=float
            ),

            index=group[
                "security_id"
            ]
            .astype(str)
            .to_numpy(),
        )

        if (
            "next_analysis_date"
            in
            group.columns
        ):

            values = (
                group[
                    "next_analysis_date"
                ]
                .dropna()
                .unique()
            )

            if len(
                values
            ) == 1:

                next_date_maps[
                    (
                        analysis_date,
                        int(
                            window
                        ),
                    )
                ] = pd.Timestamp(
                    values[0]
                )

    return (
        return_maps,
        next_date_maps,
    )


# ============================================================
# 10. Reconstruct drift-adjusted stock-level trades
# ============================================================

def reconstruct_trades(
    targets,
    labels,
):

    (
        return_maps,
        next_date_maps,
    ) = build_return_maps(
        labels
    )

    trade_frames = []

    summary_rows = []

    portfolio_keys = [

        "window",

        "factor_name",

        "method",

        "portfolio_leg",
    ]

    groups = targets.groupby(
        portfolio_keys,
        sort=False,
    )

    total_groups = (
        groups.ngroups
    )

    for group_no, (
        keys,
        group,
    ) in enumerate(
        groups,
        start=1,
    ):

        (
            window,
            factor_name,
            method,
            portfolio_leg,
        ) = keys

        if (
            group_no == 1
            or
            group_no % 20 == 0
            or
            group_no == total_groups
        ):

            print(
                f"Trade reconstruction: "
                f"{group_no}/{total_groups}"
            )

        by_date = {

            pd.Timestamp(
                date
            ):
                temp[
                    [
                        "security_id",
                        "target_weight",
                        "signal_value",
                    ]
                ]
                .copy()
                .set_index(
                    "security_id"
                )

            for date, temp
            in group.groupby(
                "analysis_date",
                sort=True,
            )
        }

        dates = sorted(
            by_date.keys()
        )

        if len(
            dates
        ) == 0:

            continue

        # ----------------------------------------------------
        # First formation:
        #
        # no previous portfolio;
        # no normal rebalance turnover/cost.
        # ----------------------------------------------------

        first_date = dates[0]

        first_target = by_date[
            first_date
        ]

        summary_rows.append(
            {
                "analysis_date":
                    first_date,

                "previous_analysis_date":
                    pd.NaT,

                "window":
                    int(
                        window
                    ),

                "factor_name":
                    factor_name,

                "method":
                    method,

                "portfolio_leg":
                    portfolio_leg,

                "status":
                    "NO_PREVIOUS_FORMATION",

                "previous_n":
                    0,

                "current_n":
                    int(
                        len(
                            first_target
                        )
                    ),

                "union_n":
                    int(
                        len(
                            first_target
                        )
                    ),

                "missing_drift_return_n":
                    0,

                "pretrade_weight_sum":
                    np.nan,

                "target_weight_sum":
                    float(
                        first_target[
                            "target_weight"
                        ]
                        .sum()
                    ),

                "turnover":
                    np.nan,

                "buy_weight":
                    np.nan,

                "sell_weight":
                    np.nan,

                "total_abs_trade_weight":
                    np.nan,

                "trade_notional_per_100m_cny":
                    np.nan,

                "trade_notional_per_1bn_cny":
                    np.nan,
            }
        )

        # ----------------------------------------------------
        # Rebalancing dates
        # ----------------------------------------------------

        for date_no in range(
            1,
            len(
                dates
            ),
        ):

            previous_date = (
                dates[
                    date_no
                    -
                    1
                ]
            )

            current_date = (
                dates[
                    date_no
                ]
            )

            previous = by_date[
                previous_date
            ]

            current = by_date[
                current_date
            ]

            previous_weights = (
                previous[
                    "target_weight"
                ]
                .astype(float)
            )

            current_weights = (
                current[
                    "target_weight"
                ]
                .astype(float)
            )

            return_key = (

                previous_date,

                int(
                    window
                ),
            )

            returns = return_maps.get(
                return_key
            )

            if returns is None:

                status = (
                    "MISSING_INTERVAL_RETURN_MAP"
                )

                missing_return_n = int(
                    len(
                        previous_weights
                    )
                )

                summary_rows.append(
                    {
                        "analysis_date":
                            current_date,

                        "previous_analysis_date":
                            previous_date,

                        "window":
                            int(
                                window
                            ),

                        "factor_name":
                            factor_name,

                        "method":
                            method,

                        "portfolio_leg":
                            portfolio_leg,

                        "status":
                            status,

                        "previous_n":
                            int(
                                len(
                                    previous_weights
                                )
                            ),

                        "current_n":
                            int(
                                len(
                                    current_weights
                                )
                            ),

                        "union_n":
                            int(
                                len(
                                    previous_weights.index.union(
                                        current_weights.index
                                    )
                                )
                            ),

                        "missing_drift_return_n":
                            missing_return_n,

                        "pretrade_weight_sum":
                            np.nan,

                        "target_weight_sum":
                            float(
                                current_weights.sum()
                            ),

                        "turnover":
                            np.nan,

                        "buy_weight":
                            np.nan,

                        "sell_weight":
                            np.nan,

                        "total_abs_trade_weight":
                            np.nan,

                        "trade_notional_per_100m_cny":
                            np.nan,

                        "trade_notional_per_1bn_cny":
                            np.nan,
                    }
                )

                continue

            # ------------------------------------------------
            # Optional interval-date consistency check.
            # ------------------------------------------------

            expected_next_date = (
                next_date_maps.get(
                    return_key
                )
            )

            if (
                expected_next_date
                is not None
                and
                pd.Timestamp(
                    expected_next_date
                )
                !=
                current_date
            ):

                summary_rows.append(
                    {
                        "analysis_date":
                            current_date,

                        "previous_analysis_date":
                            previous_date,

                        "window":
                            int(
                                window
                            ),

                        "factor_name":
                            factor_name,

                        "method":
                            method,

                        "portfolio_leg":
                            portfolio_leg,

                        "status":
                            "INTERVAL_DATE_MISMATCH",

                        "previous_n":
                            int(
                                len(
                                    previous_weights
                                )
                            ),

                        "current_n":
                            int(
                                len(
                                    current_weights
                                )
                            ),

                        "union_n":
                            int(
                                len(
                                    previous_weights.index.union(
                                        current_weights.index
                                    )
                                )
                            ),

                        "missing_drift_return_n":
                            np.nan,

                        "pretrade_weight_sum":
                            np.nan,

                        "target_weight_sum":
                            float(
                                current_weights.sum()
                            ),

                        "turnover":
                            np.nan,

                        "buy_weight":
                            np.nan,

                        "sell_weight":
                            np.nan,

                        "total_abs_trade_weight":
                            np.nan,

                        "trade_notional_per_100m_cny":
                            np.nan,

                        "trade_notional_per_1bn_cny":
                            np.nan,
                    }
                )

                continue

            # ------------------------------------------------
            # Vectorized lookup.
            #
            # No repeated:
            #     sid_series == sid
            # scans.
            # ------------------------------------------------

            previous_returns = (
                returns.reindex(
                    previous_weights.index
                )
            )

            missing_return = (
                previous_returns
                .isna()
                |
                ~np.isfinite(
                    previous_returns
                )
            )

            missing_return_n = int(
                missing_return.sum()
            )

            if missing_return_n > 0:

                status = (
                    "INCOMPLETE_DRIFT_RETURN"
                )

                summary_rows.append(
                    {
                        "analysis_date":
                            current_date,

                        "previous_analysis_date":
                            previous_date,

                        "window":
                            int(
                                window
                            ),

                        "factor_name":
                            factor_name,

                        "method":
                            method,

                        "portfolio_leg":
                            portfolio_leg,

                        "status":
                            status,

                        "previous_n":
                            int(
                                len(
                                    previous_weights
                                )
                            ),

                        "current_n":
                            int(
                                len(
                                    current_weights
                                )
                            ),

                        "union_n":
                            int(
                                len(
                                    previous_weights.index.union(
                                        current_weights.index
                                    )
                                )
                            ),

                        "missing_drift_return_n":
                            missing_return_n,

                        "pretrade_weight_sum":
                            np.nan,

                        "target_weight_sum":
                            float(
                                current_weights.sum()
                            ),

                        "turnover":
                            np.nan,

                        "buy_weight":
                            np.nan,

                        "sell_weight":
                            np.nan,

                        "total_abs_trade_weight":
                            np.nan,

                        "trade_notional_per_100m_cny":
                            np.nan,

                        "trade_notional_per_1bn_cny":
                            np.nan,
                    }
                )

                continue

            # ------------------------------------------------
            # Gross-value evolution.
            # ------------------------------------------------

            drift_value = (

                previous_weights

                *

                (
                    1.0
                    +
                    previous_returns
                )
            )

            denominator = float(
                drift_value.sum()
            )

            if (
                not np.isfinite(
                    denominator
                )
                or
                denominator
                <=
                0
            ):

                summary_rows.append(
                    {
                        "analysis_date":
                            current_date,

                        "previous_analysis_date":
                            previous_date,

                        "window":
                            int(
                                window
                            ),

                        "factor_name":
                            factor_name,

                        "method":
                            method,

                        "portfolio_leg":
                            portfolio_leg,

                        "status":
                            "INVALID_DRIFT_DENOMINATOR",

                        "previous_n":
                            int(
                                len(
                                    previous_weights
                                )
                            ),

                        "current_n":
                            int(
                                len(
                                    current_weights
                                )
                            ),

                        "union_n":
                            int(
                                len(
                                    previous_weights.index.union(
                                        current_weights.index
                                    )
                                )
                            ),

                        "missing_drift_return_n":
                            0,

                        "pretrade_weight_sum":
                            np.nan,

                        "target_weight_sum":
                            float(
                                current_weights.sum()
                            ),

                        "turnover":
                            np.nan,

                        "buy_weight":
                            np.nan,

                        "sell_weight":
                            np.nan,

                        "total_abs_trade_weight":
                            np.nan,

                        "trade_notional_per_100m_cny":
                            np.nan,

                        "trade_notional_per_1bn_cny":
                            np.nan,
                    }
                )

                continue

            pretrade_weights = (

                drift_value

                /

                denominator
            )

            union_index = (
                previous_weights.index.union(
                    current_weights.index
                )
            )

            pretrade_union = (
                pretrade_weights.reindex(
                    union_index,
                    fill_value=0.0,
                )
            )

            target_union = (
                current_weights.reindex(
                    union_index,
                    fill_value=0.0,
                )
            )

            delta_weight = (

                target_union

                -

                pretrade_union
            )

            abs_delta_weight = (
                delta_weight.abs()
            )

            buy_weight = (
                delta_weight.clip(
                    lower=0
                )
            )

            sell_weight = (

                -
                delta_weight.clip(
                    upper=0
                )
            )

            total_abs_trade_weight = float(
                abs_delta_weight.sum()
            )

            turnover = (

                0.5

                *

                total_abs_trade_weight
            )

            # ------------------------------------------------
            # Stock-level trade records
            # ------------------------------------------------

            stock_trade = pd.DataFrame(
                {
                    "analysis_date":
                        current_date,

                    "previous_analysis_date":
                        previous_date,

                    "window":
                        int(
                            window
                        ),

                    "factor_name":
                        factor_name,

                    "method":
                        method,

                    "portfolio_leg":
                        portfolio_leg,

                    "security_id":
                        union_index.astype(
                            str
                        ),

                    "previous_target_weight":
                        previous_weights.reindex(
                            union_index,
                            fill_value=0.0,
                        )
                        .to_numpy(
                            dtype=float
                        ),

                    "holding_period_total_return":
                        previous_returns.reindex(
                            union_index
                        )
                        .to_numpy(
                            dtype=float
                        ),

                    "pretrade_weight":
                        pretrade_union.to_numpy(
                            dtype=float
                        ),

                    "target_weight":
                        target_union.to_numpy(
                            dtype=float
                        ),

                    "delta_weight":
                        delta_weight.to_numpy(
                            dtype=float
                        ),

                    "abs_delta_weight":
                        abs_delta_weight.to_numpy(
                            dtype=float
                        ),

                    "buy_weight":
                        buy_weight.to_numpy(
                            dtype=float
                        ),

                    "sell_weight":
                        sell_weight.to_numpy(
                            dtype=float
                        ),
                }
            )

            # ------------------------------------------------
            # Previous returns are only defined for previous
            # members. New current names legitimately have NA
            # in holding_period_total_return.
            #
            # This is NOT a missing-drift problem because their
            # pretrade weight is zero.
            # ------------------------------------------------

            stock_trade[
                "trade_side"
            ] = np.where(

                stock_trade[
                    "delta_weight"
                ]
                >
                1e-15,

                "BUY",

                np.where(

                    stock_trade[
                        "delta_weight"
                    ]
                    <
                    -1e-15,

                    "SELL",

                    "NONE",
                )
            )

            # ------------------------------------------------
            # Monetary trade values at two reference NAVs.
            #
            # These do NOT need the ADV unit yet.
            # ------------------------------------------------

            stock_trade[
                "trade_value_per_100m_nav_cny"
            ] = (

                stock_trade[
                    "abs_delta_weight"
                ]

                *

                REFERENCE_NAV_100M_CNY
            )

            stock_trade[
                "trade_value_per_1bn_nav_cny"
            ] = (

                stock_trade[
                    "abs_delta_weight"
                ]

                *

                REFERENCE_NAV_1B_CNY
            )

            # Keep all union members for reproducibility,
            # including very small weight changes.
            trade_frames.append(
                stock_trade
            )

            summary_rows.append(
                {
                    "analysis_date":
                        current_date,

                    "previous_analysis_date":
                        previous_date,

                    "window":
                        int(
                            window
                        ),

                    "factor_name":
                        factor_name,

                    "method":
                        method,

                    "portfolio_leg":
                        portfolio_leg,

                    "status":
                        "PASS",

                    "previous_n":
                        int(
                            len(
                                previous_weights
                            )
                        ),

                    "current_n":
                        int(
                            len(
                                current_weights
                            )
                        ),

                    "union_n":
                        int(
                            len(
                                union_index
                            )
                        ),

                    "missing_drift_return_n":
                        0,

                    "pretrade_weight_sum":
                        float(
                            pretrade_union.sum()
                        ),

                    "target_weight_sum":
                        float(
                            target_union.sum()
                        ),

                    "turnover":
                        float(
                            turnover
                        ),

                    "buy_weight":
                        float(
                            buy_weight.sum()
                        ),

                    "sell_weight":
                        float(
                            sell_weight.sum()
                        ),

                    "total_abs_trade_weight":
                        total_abs_trade_weight,

                    "trade_notional_per_100m_cny":
                        (
                            total_abs_trade_weight
                            *
                            REFERENCE_NAV_100M_CNY
                        ),

                    "trade_notional_per_1bn_cny":
                        (
                            total_abs_trade_weight
                            *
                            REFERENCE_NAV_1B_CNY
                        ),
                }
            )

    if trade_frames:

        trades = pd.concat(
            trade_frames,
            ignore_index=True,
        )

    else:

        trades = pd.DataFrame()

    summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "analysis_date",
                "window",
                "factor_name",
                "method",
                "portfolio_leg",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        trades,
        summary,
    )


# ============================================================
# 11. Build AUM scenario summary
#
# Do NOT duplicate the stock-level table six times.
# ============================================================

def build_aum_scenario_summary(
    rebalance_summary,
):

    valid = rebalance_summary[
        rebalance_summary[
            "status"
        ]
        ==
        "PASS"
    ].copy()

    rows = []

    for row in valid.itertuples(
        index=False
    ):

        for aum in AUM_SCENARIOS_CNY:

            rows.append(
                {
                    "analysis_date":
                        row.analysis_date,

                    "previous_analysis_date":
                        row.previous_analysis_date,

                    "window":
                        int(
                            row.window
                        ),

                    "factor_name":
                        row.factor_name,

                    "method":
                        row.method,

                    "portfolio_leg":
                        row.portfolio_leg,

                    "strategy_nav_cny":
                        float(
                            aum
                        ),

                    "leg_gross_notional_cny":
                        float(
                            aum
                        ),

                    "strategy_gross_exposure":
                        2.0,

                    "turnover":
                        float(
                            row.turnover
                        ),

                    "total_abs_trade_weight":
                        float(
                            row.total_abs_trade_weight
                        ),

                    "trade_notional_cny":
                        (
                            float(
                                aum
                            )
                            *
                            float(
                                row.total_abs_trade_weight
                            )
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 12. Optional comparison with Day-7 Step-4
#
# This is DIAGNOSTIC ONLY because Day-8 target formation is
# explicitly current-information based.
# ============================================================

def compare_with_step4(
    rebalance_summary,
):

    if not STEP4_ECONOMIC_PATH.exists():

        return pd.DataFrame()

    old = pd.read_csv(
        STEP4_ECONOMIC_PATH
    )

    required = [

        "factor_name",

        "method",

        "window",

        "mean_q1_drift_turnover",

        "mean_q5_drift_turnover",

        "mean_roundtrip_traded_notional_ratio",
    ]

    if any(
        column not in old.columns
        for column in required
    ):

        return pd.DataFrame()

    current = rebalance_summary[
        rebalance_summary[
            "status"
        ]
        ==
        "PASS"
    ].copy()

    grouped = (

        current.groupby(
            [
                "factor_name",
                "method",
                "window",
                "portfolio_leg",
            ],
            as_index=False,
        )[
            "turnover"
        ]
        .mean()
    )

    pivot = grouped.pivot(

        index=[
            "factor_name",
            "method",
            "window",
        ],

        columns="portfolio_leg",

        values="turnover",
    ).reset_index()

    if (
        "Q1"
        not in pivot.columns
        or
        "Q5"
        not in pivot.columns
    ):

        return pd.DataFrame()

    pivot = pivot.rename(
        columns={
            "Q1":
                "reconstructed_mean_q1_turnover",

            "Q5":
                "reconstructed_mean_q5_turnover",
        }
    )

    pivot[
        "reconstructed_roundtrip_traded_notional_ratio"
    ] = (

        2.0

        *

        (
            pivot[
                "reconstructed_mean_q1_turnover"
            ]

            +

            pivot[
                "reconstructed_mean_q5_turnover"
            ]
        )
    )

    old = old[
        old[
            "method"
        ]
        .isin(
            METHODS
        )
    ].copy()

    result = pivot.merge(

        old[
            required
        ],

        on=[
            "factor_name",
            "method",
            "window",
        ],

        how="left",

        validate="one_to_one",
    )

    result[
        "q1_turnover_difference"
    ] = (

        result[
            "reconstructed_mean_q1_turnover"
        ]

        -

        result[
            "mean_q1_drift_turnover"
        ]
    )

    result[
        "q5_turnover_difference"
    ] = (

        result[
            "reconstructed_mean_q5_turnover"
        ]

        -

        result[
            "mean_q5_drift_turnover"
        ]
    )

    result[
        "roundtrip_difference"
    ] = (

        result[
            "reconstructed_roundtrip_traded_notional_ratio"
        ]

        -

        result[
            "mean_roundtrip_traded_notional_ratio"
        ]
    )

    return result


# ============================================================
# 13. Formal QA
# ============================================================

def formal_qa(
    targets,
    trades,
    rebalance_summary,
    sort_qa,
):

    qa = {}

    # --------------------------------------------------------
    # Target weights
    # --------------------------------------------------------

    target_sum = (

        targets.groupby(
            [
                "analysis_date",
                "window",
                "factor_name",
                "method",
                "portfolio_leg",
            ]
        )[
            "target_weight"
        ]
        .sum()
    )

    qa[
        "max_abs_target_weight_sum_minus_1"
    ] = float(
        (
            target_sum
            -
            1.0
        )
        .abs()
        .max()
    )

    qa[
        "target_weight_sum_ok"
    ] = bool(
        qa[
            "max_abs_target_weight_sum_minus_1"
        ]
        <
        1e-12
    )

    # --------------------------------------------------------
    # Trade panel uniqueness
    # --------------------------------------------------------

    if len(
        trades
    ) > 0:

        trade_key = [

            "analysis_date",

            "window",

            "factor_name",

            "method",

            "portfolio_leg",

            "security_id",
        ]

        qa[
            "trade_keys_unique"
        ] = bool(
            not trades[
                trade_key
            ]
            .duplicated()
            .any()
        )

        # ----------------------------------------------------
        # delta = target - pretrade
        # ----------------------------------------------------

        delta_error = (

            trades[
                "delta_weight"
            ]

            -

            (
                trades[
                    "target_weight"
                ]

                -

                trades[
                    "pretrade_weight"
                ]
            )
        ).abs()

        qa[
            "max_delta_weight_identity_error"
        ] = float(
            delta_error.max()
        )

        qa[
            "delta_weight_identity_ok"
        ] = bool(
            delta_error.max()
            <
            1e-14
        )

    else:

        qa[
            "trade_keys_unique"
        ] = False

        qa[
            "max_delta_weight_identity_error"
        ] = np.nan

        qa[
            "delta_weight_identity_ok"
        ] = False

    # --------------------------------------------------------
    # PASS rebalance groups
    # --------------------------------------------------------

    valid = rebalance_summary[
        rebalance_summary[
            "status"
        ]
        ==
        "PASS"
    ].copy()

    qa[
        "pass_rebalance_count"
    ] = int(
        len(
            valid
        )
    )

    if len(
        valid
    ) > 0:

        qa[
            "max_abs_pretrade_weight_sum_minus_1"
        ] = float(
            (
                valid[
                    "pretrade_weight_sum"
                ]
                -
                1.0
            )
            .abs()
            .max()
        )

        qa[
            "max_abs_current_target_weight_sum_minus_1"
        ] = float(
            (
                valid[
                    "target_weight_sum"
                ]
                -
                1.0
            )
            .abs()
            .max()
        )

        qa[
            "pretrade_weight_sum_ok"
        ] = bool(
            qa[
                "max_abs_pretrade_weight_sum_minus_1"
            ]
            <
            1e-12
        )

        qa[
            "current_target_weight_sum_ok"
        ] = bool(
            qa[
                "max_abs_current_target_weight_sum_minus_1"
            ]
            <
            1e-12
        )

        qa[
            "turnover_within_0_1"
        ] = bool(
            (
                valid[
                    "turnover"
                ]
                >=
                -1e-14
            )
            .all()

            and

            (
                valid[
                    "turnover"
                ]
                <=
                1.0
                +
                1e-12
            )
            .all()
        )

        # ----------------------------------------------------
        # For two fully invested portfolios:
        #
        # sum(delta)=0
        # => buy weight == sell weight == turnover
        # ----------------------------------------------------

        buy_error = (

            valid[
                "buy_weight"
            ]

            -

            valid[
                "turnover"
            ]
        ).abs()

        sell_error = (

            valid[
                "sell_weight"
            ]

            -

            valid[
                "turnover"
            ]
        ).abs()

        qa[
            "max_buy_turnover_difference"
        ] = float(
            buy_error.max()
        )

        qa[
            "max_sell_turnover_difference"
        ] = float(
            sell_error.max()
        )

        qa[
            "buy_sell_turnover_identity_ok"
        ] = bool(
            buy_error.max()
            <
            1e-12

            and

            sell_error.max()
            <
            1e-12
        )

        notional_error = (

            valid[
                "trade_notional_per_1bn_cny"
            ]

            -

            (
                2.0
                *
                REFERENCE_NAV_1B_CNY
                *
                valid[
                    "turnover"
                ]
            )
        ).abs()

        qa[
            "max_1bn_notional_identity_error_cny"
        ] = float(
            notional_error.max()
        )

        qa[
            "monetary_notional_identity_ok"
        ] = bool(
            notional_error.max()
            <
            1e-4
        )

    else:

        qa[
            "pretrade_weight_sum_ok"
        ] = False

        qa[
            "current_target_weight_sum_ok"
        ] = False

        qa[
            "turnover_within_0_1"
        ] = False

        qa[
            "buy_sell_turnover_identity_ok"
        ] = False

        qa[
            "monetary_notional_identity_ok"
        ] = False

    # --------------------------------------------------------
    # Sort QA
    # --------------------------------------------------------

    qa[
        "sort_pass_count"
    ] = int(
        (
            sort_qa[
                "status"
            ]
            ==
            "PASS"
        )
        .sum()
    )

    qa[
        "sort_failure_count"
    ] = int(
        (
            sort_qa[
                "status"
            ]
            !=
            "PASS"
        )
        .sum()
    )

    # --------------------------------------------------------
    # Governance
    # --------------------------------------------------------

    qa[
        "future_outcome_used_to_form_current_portfolio"
    ] = False

    qa[
        "previous_interval_total_return_used_for_weight_drift"
    ] = True

    qa[
        "factor_sign_flip"
    ] = False

    qa[
        "factor_selection"
    ] = False

    qa[
        "window_selection"
    ] = False

    qa[
        "initial_formation_treated_as_rebalance"
    ] = False

    qa[
        "terminal_liquidation_assumed"
    ] = False

    required = [

        "target_weight_sum_ok",

        "trade_keys_unique",

        "delta_weight_identity_ok",

        "pretrade_weight_sum_ok",

        "current_target_weight_sum_ok",

        "turnover_within_0_1",

        "buy_sell_turnover_identity_ok",

        "monetary_notional_identity_ok",
    ]

    qa[
        "all_formal_qa_pass"
    ] = bool(
        all(
            qa[
                key
            ]
            for key in required
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
        "Day 8 - Step 2"
    )

    print(
        "Reconstruct Portfolio-Level Monetary Trades"
    )

    print("=" * 80)

    # ========================================================
    # Current information
    # ========================================================

    print()
    print(
        "[1] Load current-information factor/control panel"
    )

    panel = (
        load_current_information_panel()
    )

    print(
        f"Factor-stock rows: "
        f"{len(panel):,}"
    )

    print(
        f"Analysis dates: "
        f"{panel['analysis_date'].nunique():,}"
    )

    print(
        f"Factors: "
        f"{panel['factor_name'].nunique():,}"
    )

    # ========================================================
    # Target portfolios
    # ========================================================

    print()
    print(
        "[2] Reconstruct current-information Q1/Q5 targets"
    )

    (
        targets,
        sort_qa,
    ) = build_target_portfolios(
        panel
    )

    save_parquet_atomic(
        targets,
        TARGET_WEIGHT_PATH,
    )

    save_csv_atomic(
        sort_qa,
        SORT_QA_PATH,
    )

    print(
        f"Target-weight rows: "
        f"{len(targets):,}"
    )

    # ========================================================
    # Forward total return
    #
    # Previous interval only.
    # ========================================================

    print()
    print(
        "[3] Load previous-period total returns for weight drift"
    )

    (
        labels,
        label_source_path,
    ) = load_forward_total_returns()

    # ========================================================
    # Trade reconstruction
    # ========================================================

    print()
    print(
        "[4] Reconstruct drift-adjusted stock-level trades"
    )

    (
        trades,
        rebalance_summary,
    ) = reconstruct_trades(

        targets,

        labels,
    )

    save_parquet_atomic(
        trades,
        TRADE_PANEL_PATH,
    )

    save_csv_atomic(
        rebalance_summary,
        REBALANCE_SUMMARY_PATH,
    )

    print(
        f"Stock-level trade rows: "
        f"{len(trades):,}"
    )

    # ========================================================
    # AUM scenarios
    # ========================================================

    print()
    print(
        "[5] Build compact AUM scenario summary"
    )

    aum_summary = (
        build_aum_scenario_summary(
            rebalance_summary
        )
    )

    save_csv_atomic(
        aum_summary,
        AUM_SCENARIO_SUMMARY_PATH,
    )

    # ========================================================
    # Step-4 diagnostic comparison
    # ========================================================

    print()
    print(
        "[6] Compare reconstructed turnover "
        "with Day-7 Step-4 where available"
    )

    comparison = (
        compare_with_step4(
            rebalance_summary
        )
    )

    if len(
        comparison
    ) > 0:

        save_csv_atomic(
            comparison,
            STEP4_COMPARISON_PATH,
        )

        print(
            "Max absolute round-trip turnover difference:"
        )

        print(
            comparison[
                "roundtrip_difference"
            ]
            .abs()
            .max()
        )

    else:

        print(
            "Step-4 comparison unavailable; "
            "continuing because it is diagnostic only."
        )

    # ========================================================
    # Formal QA
    # ========================================================

    print()
    print(
        "[7] Formal QA"
    )

    qa = formal_qa(

        targets,

        trades,

        rebalance_summary,

        sort_qa,
    )

    qa_df = pd.DataFrame(
        [
            {
                "qa_name":
                    key,

                "qa_value":
                    value,
            }

            for key, value
            in qa.items()
        ]
    )

    save_csv_atomic(
        qa_df,
        FORMAL_QA_PATH,
    )

    if (
        not qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            "Day-8 Step-2 formal QA failed.\n"
            f"{qa}"
        )

    # ========================================================
    # Metadata
    # ========================================================

    design_payload = {

        "research_day":
            8,

        "step":
            "Step2_Reconstruct_Portfolio_Level_Monetary_Trades",

        "factor_count":
            len(
                FACTOR_NAMES
            ),

        "factors":
            FACTOR_NAMES,

        "windows":
            WINDOWS,

        "methods":
            METHODS,

        "main_method":
            "FULL_CHARACTERISTIC_NEUTRAL",

        "portfolio_sort":
            "Conservative cross-sectional quintile qcut",

        "portfolio_legs":
            [
                "Q1",
                "Q5",
            ],

        "weighting":
            "Equal weight within each leg",

        "current_portfolio_formation":
            (
                "Current-information only; future outcome "
                "availability is not used."
            ),

        "drift_return":
            "future_total_return from previous formation interval",

        "drift_formula":
            (
                "w_pre = w_prev * (1 + R_total) "
                "/ sum_j[w_prev_j * (1 + R_total_j)]"
            ),

        "trade_weight":
            "delta_w = target_w - pretrade_w",

        "turnover":
            "0.5 * sum(abs(delta_w))",

        "aum_definition":
            (
                "Strategy NAV; each Q1/Q5 leg has gross "
                "notional equal to strategy NAV, "
                "so total gross exposure is 200%."
            ),

        "aum_scenarios_cny":
            AUM_SCENARIOS_CNY,

        "initial_formation_in_turnover":
            False,

        "terminal_liquidation":
            False,

        "missing_drift_return_zero_filled":
            False,

        "factor_sign_flip":
            False,

        "factor_selection":
            False,

        "window_selection":
            False,

        "future_outcome_used_for_current_portfolio":
            False,
    }

    design_hash = canonical_hash(
        design_payload
    )

    metadata = {

        **design_payload,

        "day8_step2_design_hash":
            design_hash,

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "forward_label_source":
            str(
                label_source_path
            ),

        "formal_qa":
            qa,

        "outputs":
            {

                "portfolio_target_weights":
                    str(
                        TARGET_WEIGHT_PATH
                    ),

                "portfolio_trade_weight_panel":
                    str(
                        TRADE_PANEL_PATH
                    ),

                "portfolio_rebalance_summary":
                    str(
                        REBALANCE_SUMMARY_PATH
                    ),

                "portfolio_trade_value_scenario_summary":
                    str(
                        AUM_SCENARIO_SUMMARY_PATH
                    ),

                "portfolio_sort_qa":
                    str(
                        SORT_QA_PATH
                    ),

                "formal_qa":
                    str(
                        FORMAL_QA_PATH
                    ),
            },

        "important_notes":
            [

                (
                    "Current target portfolios are formed "
                    "without conditioning on future-return "
                    "availability."
                ),

                (
                    "Previous-interval future_total_return is "
                    "used only after it has become realized, "
                    "to drift the previous portfolio into the "
                    "current rebalance date."
                ),

                (
                    "Stock-level monetary trade values are "
                    "linear in strategy NAV; therefore the "
                    "stock-level panel is not duplicated for "
                    "every AUM scenario."
                ),

                (
                    "The ADV source-value unit is not required "
                    "for Step 2, but must be confirmed before "
                    "Step 3 participation-rate calculations."
                ),

                (
                    "Day-7 Step-4 turnover comparison is "
                    "diagnostic because Day-8 explicitly uses "
                    "a current-information-only portfolio "
                    "formation universe."
                ),
            ],
    }

    save_json_atomic(
        metadata,
        METADATA_PATH,
    )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Day-8 Step-2 Formal QA"
    )

    print("=" * 80)

    print(
        f"Target weight sum error: "
        f"{qa['max_abs_target_weight_sum_minus_1']}"
    )

    print(
        f"Pre-trade weight sum OK: "
        f"{qa['pretrade_weight_sum_ok']}"
    )

    print(
        f"Delta-weight identity OK: "
        f"{qa['delta_weight_identity_ok']}"
    )

    print(
        f"Turnover in [0,1]: "
        f"{qa['turnover_within_0_1']}"
    )

    print(
        f"Buy/Sell/Turnover identity OK: "
        f"{qa['buy_sell_turnover_identity_ok']}"
    )

    print(
        f"Monetary notional identity OK: "
        f"{qa['monetary_notional_identity_ok']}"
    )

    print(
        f"Formal QA pass: "
        f"{qa['all_formal_qa_pass']}"
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
        "Day 8 Step 2 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()
