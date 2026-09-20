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
# Day 1
# ------------------------------------------------------------

DAY1_RETURN_PANEL_PATH = (
    OUTPUT_ROOT
    / "M1_day1"
    / "03_stage3_return_validation"
    / "daily_return_panel.parquet"
)


# ------------------------------------------------------------
# Day 5 factor panel
#
# Research Day 5 -> M1_day4
# ------------------------------------------------------------

DAY5_ROOT = (
    OUTPUT_ROOT
    / "M1_day5"
)

FACTOR_PANEL_PATH = (
    DAY5_ROOT
    / "05_step5_community_features"
    / "community_bridge_feature_panel.parquet"
)


# ------------------------------------------------------------
# Day 6
#
# Research Day 6 -> M1_day5
# ------------------------------------------------------------

DAY6_ROOT = (
    OUTPUT_ROOT
    / "M1_day6"
)

FROZEN_CORE_PATH = (
    DAY6_ROOT
    / "01_step1_freeze_reduced_factor_set"
    / "frozen_core_factor_set.csv"
)

LABEL_PANEL_PATH = (
    DAY6_ROOT
    / "02_step2_forward_labels"
    / "forward_label_panel.parquet"
)

STEP6_METADATA_PATH = (
    DAY6_ROOT
    / "06_step6_neutralization_and_classification"
    / "step6_neutralization_metadata.json"
)


# ------------------------------------------------------------
# Day 7 Step 1 lineage
# ------------------------------------------------------------

DAY7_ROOT = (
    OUTPUT_ROOT
    / "M1_day7"
)

STEP1_METADATA_PATH = (
    DAY7_ROOT
    / "01_stage1_subsample_regime_robustness"
    / "day7_step1_metadata.json"
)


# ------------------------------------------------------------
# Day 7 Step 2 output
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY7_ROOT
    / "02_stage2_traditional_characteristic_controls"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CHARACTERISTIC_PANEL_PATH = (
    OUTPUT_DIR
    / "traditional_characteristic_panel.parquet"
)

CHARACTERISTIC_QA_PATH = (
    OUTPUT_DIR
    / "traditional_characteristic_qa.csv"
)

EXPOSURE_PATH = (
    OUTPUT_DIR
    / "factor_control_exposure_summary.csv"
)

MONTHLY_ALPHA_IC_PATH = (
    OUTPUT_DIR
    / "monthly_controlled_alpha_rank_ic.csv"
)

ALPHA_SUMMARY_PATH = (
    OUTPUT_DIR
    / "controlled_alpha_rank_ic_summary.csv"
)

MONTHLY_ALPHA_SPREAD_PATH = (
    OUTPUT_DIR
    / "monthly_controlled_alpha_portfolio_spreads.csv"
)

ALPHA_PORTFOLIO_SUMMARY_PATH = (
    OUTPUT_DIR
    / "controlled_alpha_portfolio_summary.csv"
)

MONTHLY_RISK_IC_PATH = (
    OUTPUT_DIR
    / "monthly_controlled_risk_rank_ic.csv"
)

RISK_SUMMARY_PATH = (
    OUTPUT_DIR
    / "controlled_risk_rank_ic_summary.csv"
)

RETENTION_PATH = (
    OUTPUT_DIR
    / "traditional_control_signal_retention.csv"
)

NEUTRALIZATION_QA_PATH = (
    OUTPUT_DIR
    / "traditional_control_neutralization_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day7_step2_metadata.json"
)


# ============================================================
# 1. Frozen Design
# ============================================================

EXPECTED_WINDOWS = [
    60,
    120,
    252,
]

EXPECTED_FACTOR_COUNT = 9


METHODS = [

    "RAW_COMMON",

    "INDUSTRY_SIZE_COMMON",

    "FULL_CHARACTERISTIC_NEUTRAL",
]


# ------------------------------------------------------------
# Traditional characteristics
# ------------------------------------------------------------

CONTROL_COLUMNS = [

    "log_market_value",

    "momentum_120_20",

    "reversal_20",

    "volatility_60",

    "log_turnover_20",
]


FULL_CONTROL_COLUMNS = CONTROL_COLUMNS.copy()

SIZE_ONLY_COLUMNS = [
    "log_market_value"
]


# ------------------------------------------------------------
# Feature lookbacks
# ------------------------------------------------------------

MOMENTUM_TOTAL_LOOKBACK = 120

MOMENTUM_SKIP_RECENT = 20

MOMENTUM_EFFECTIVE_WINDOW = (
    MOMENTUM_TOTAL_LOOKBACK
    -
    MOMENTUM_SKIP_RECENT
)

REVERSAL_LOOKBACK = 20

VOL_LOOKBACK = 60

TURNOVER_LOOKBACK = 20

ANNUALIZATION = 252


# ------------------------------------------------------------
# Cross-sectional design
# ------------------------------------------------------------

CONTROL_WINSOR_LOW = 0.01

CONTROL_WINSOR_HIGH = 0.99

MIN_CROSS_SECTION_OBS = 100

MIN_SORT_OBS = 100

MIN_PORTFOLIO_OBS = 20

N_PORTFOLIOS = 5

MIN_CONTROL_COVERAGE = 0.50


# ------------------------------------------------------------
# HAC
# ------------------------------------------------------------

USE_HAC = True


# ------------------------------------------------------------
# Retention ratios
# ------------------------------------------------------------

MIN_ALPHA_IC_BASELINE = 0.005

MIN_ALPHA_SPREAD_BASELINE = 0.001

MIN_RISK_IC_BASELINE = 0.01


# ============================================================
# 2. Targets
# ============================================================

ALPHA_TARGET = (
    "future_excess_return"
)

RISK_TARGETS = [

    "future_realized_vol_annualized",

    "future_downside_vol_annualized",

    "future_max_drawdown",

    "future_idio_vol_annualized",
]


KEY_COLUMNS = [

    "analysis_date",

    "window",

    "master_index",

    "security_id",
]


# ============================================================
# 3. Utilities
# ============================================================

def load_json(path):

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
    path,
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
    df,
    path,
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


def canonical_hash(obj):

    payload = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
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


def resolve_column(
    schema_names,
    candidates,
):

    for name in candidates:

        if name in schema_names:

            return name

    raise RuntimeError(
        f"None of these columns were found:\n"
        f"{candidates}"
    )


# ============================================================
# 4. Upstream Validation
# ============================================================

def validate_upstream():

    step6 = load_json(
        STEP6_METADATA_PATH
    )

    if (
        step6.get(
            "formal_qa",
            {}
        ).get(
            "all_formal_qa_pass"
        )
        is not True
    ):

        raise RuntimeError(
            "Day-6 Step-6 QA failed."
        )

    step1 = load_json(
        STEP1_METADATA_PATH
    )

    if (
        step1.get(
            "formal_qa",
            {}
        ).get(
            "all_formal_qa_pass"
        )
        is not True
    ):

        raise RuntimeError(
            "Day-7 Step-1 QA failed."
        )

    return (
        step6,
        step1,
    )


# ============================================================
# 5. Frozen Core Factors
# ============================================================

def load_core():

    core = pd.read_csv(
        FROZEN_CORE_PATH
    )

    if (
        "factor_name"
        not in core.columns
    ):

        raise RuntimeError(
            "factor_name missing."
        )

    if len(core) != EXPECTED_FACTOR_COUNT:

        raise RuntimeError(
            "Unexpected number of frozen factors."
        )

    if core[
        "factor_name"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate factor names."
        )

    if (
        "priority"
        not in core.columns
    ):

        core[
            "priority"
        ] = (
            np.arange(
                len(core)
            )
            +
            1
        )

    return (
        core.sort_values(
            "priority"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 6. Normalize Open Flag
# ============================================================

def normalize_is_open(x):

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
# 7. Construct Traditional Characteristics
# ============================================================

def build_characteristic_panel(
    analysis_dates,
):

    print()
    print(
        "[1] Construct PIT traditional stock characteristics"
    )

    parquet = pq.ParquetFile(
        DAY1_RETURN_PANEL_PATH
    )

    schema = set(
        parquet
        .schema_arrow
        .names
    )

    date_col = resolve_column(
        schema,
        [
            "trade_date",
        ],
    )

    security_col = resolve_column(
        schema,
        [
            "security_id",
        ],
    )

    open_col = resolve_column(
        schema,
        [
            "is_open",
            "isOpen",
        ],
    )

    return_col = resolve_column(
        schema,
        [
            "return_last_trade_simple",
        ],
    )

    market_value_col = resolve_column(
        schema,
        [
            "market_value",
            "marketValue",
        ],
    )

    turnover_col = resolve_column(
        schema,
        [
            "turnoverRate",
            "turnover_rate",
        ],
    )

    industry_col = resolve_column(
        schema,
        [
            "industry_id1",
            "industryID1",
        ],
    )

    columns = [

        date_col,

        security_col,

        open_col,

        return_col,

        market_value_col,

        turnover_col,

        industry_col,
    ]

    print(
        "Loading selected Day-1 columns..."
    )

    daily = pd.read_parquet(
        DAY1_RETURN_PANEL_PATH,
        columns=columns,
    )

    rename = {

        date_col:
            "trade_date",

        security_col:
            "security_id",

        open_col:
            "is_open",

        return_col:
            "return_last_trade_simple",

        market_value_col:
            "market_value",

        turnover_col:
            "turnover_rate",

        industry_col:
            "industry_id1",
    }

    daily = daily.rename(
        columns=rename
    )

    daily[
        "trade_date"
    ] = pd.to_datetime(
        daily[
            "trade_date"
        ]
    )

    daily[
        "security_id"
    ] = (
        daily[
            "security_id"
        ]
        .astype("string")
        .str.strip()
    )

    daily[
        "industry_id1"
    ] = (
        daily[
            "industry_id1"
        ]
        .astype("string")
        .str.strip()
    )

    daily[
        "market_value"
    ] = pd.to_numeric(
        daily[
            "market_value"
        ],
        errors="coerce",
    )

    daily[
        "turnover_rate"
    ] = pd.to_numeric(
        daily[
            "turnover_rate"
        ],
        errors="coerce",
    )

    daily[
        "return_last_trade_simple"
    ] = pd.to_numeric(
        daily[
            "return_last_trade_simple"
        ],
        errors="coerce",
    )

    is_open = normalize_is_open(
        daily[
            "is_open"
        ]
    )

    # ========================================================
    # Holding return
    # ========================================================

    holding = pd.Series(
        np.nan,
        index=daily.index,
        dtype=float,
    )

    holding.loc[
        is_open.eq(
            False
        )
    ] = 0.0

    valid_active = (

        is_open.eq(
            True
        )

        &

        daily[
            "return_last_trade_simple"
        ]
        .notna()

        &

        np.isfinite(
            daily[
                "return_last_trade_simple"
            ]
        )
    )

    holding.loc[
        valid_active
    ] = (
        daily.loc[
            valid_active,
            "return_last_trade_simple",
        ]
    )

    if (
        holding.dropna()
        <
        -1.0
        -
        1e-12
    ).any():

        raise RuntimeError(
            "Holding simple return < -1 detected."
        )

    daily[
        "_holding_return"
    ] = holding

    daily[
        "_log1p_return"
    ] = np.log1p(
        daily[
            "_holding_return"
        ]
    )

    # ========================================================
    # Sorting is essential for rolling features
    # ========================================================

    daily = (
        daily.sort_values(
            [
                "security_id",
                "trade_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # log size
    # --------------------------------------------------------

    daily[
        "log_market_value"
    ] = np.where(

        daily[
            "market_value"
        ]
        >
        0,

        np.log(
            daily[
                "market_value"
            ]
        ),

        np.nan,
    )

    # --------------------------------------------------------
    # 20-day short-term reversal
    #
    # t-19 ... t
    # --------------------------------------------------------

    reversal_log = (

        daily.groupby(
            "security_id",
            sort=False,
        )[
            "_log1p_return"
        ]

        .rolling(
            REVERSAL_LOOKBACK,
            min_periods=REVERSAL_LOOKBACK,
        )

        .sum()

        .reset_index(
            level=0,
            drop=True,
        )
    )

    daily[
        "reversal_20"
    ] = np.expm1(
        reversal_log
    )

    # --------------------------------------------------------
    # Momentum 120-20
    #
    # uses t-119 ... t-20
    #
    # Most recent 20 trading days are excluded.
    # --------------------------------------------------------

    daily[
        "_log1p_return_lag20"
    ] = (

        daily.groupby(
            "security_id",
            sort=False,
        )[
            "_log1p_return"
        ]
        .shift(
            MOMENTUM_SKIP_RECENT
        )
    )

    momentum_log = (

        daily.groupby(
            "security_id",
            sort=False,
        )[
            "_log1p_return_lag20"
        ]

        .rolling(
            MOMENTUM_EFFECTIVE_WINDOW,
            min_periods=MOMENTUM_EFFECTIVE_WINDOW,
        )

        .sum()

        .reset_index(
            level=0,
            drop=True,
        )
    )

    daily[
        "momentum_120_20"
    ] = np.expm1(
        momentum_log
    )

    # --------------------------------------------------------
    # 60-day realized volatility
    # --------------------------------------------------------

    daily[
        "volatility_60"
    ] = (

        daily.groupby(
            "security_id",
            sort=False,
        )[
            "_holding_return"
        ]

        .rolling(
            VOL_LOOKBACK,
            min_periods=VOL_LOOKBACK,
        )

        .std(
            ddof=1
        )

        .reset_index(
            level=0,
            drop=True,
        )

        *

        np.sqrt(
            ANNUALIZATION
        )
    )

    # --------------------------------------------------------
    # 20-day turnover
    # --------------------------------------------------------

    turnover = daily[
        "turnover_rate"
    ].where(
        daily[
            "turnover_rate"
        ]
        >=
        0
    )

    daily[
        "_valid_turnover"
    ] = turnover

    mean_turnover = (

        daily.groupby(
            "security_id",
            sort=False,
        )[
            "_valid_turnover"
        ]

        .rolling(
            TURNOVER_LOOKBACK,
            min_periods=TURNOVER_LOOKBACK,
        )

        .mean()

        .reset_index(
            level=0,
            drop=True,
        )
    )

    daily[
        "log_turnover_20"
    ] = np.log1p(
        mean_turnover
    )

    # ========================================================
    # Keep analysis dates only
    # ========================================================

    analysis_dates = set(
        pd.to_datetime(
            analysis_dates
        )
    )

    keep_columns = [

        "trade_date",

        "security_id",

        "industry_id1",

        "log_market_value",

        "momentum_120_20",

        "reversal_20",

        "volatility_60",

        "log_turnover_20",
    ]

    result = daily[
        daily[
            "trade_date"
        ]
        .isin(
            analysis_dates
        )
    ][
        keep_columns
    ].copy()

    result = result.rename(
        columns={
            "trade_date":
                "analysis_date",
        }
    )

    if (
        result[
            [
                "analysis_date",
                "security_id",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate characteristic keys."
        )

    result.to_parquet(
        CHARACTERISTIC_PANEL_PATH,
        index=False,
        compression="zstd",
    )

    del daily

    return result


# ============================================================
# 8. Load Factor + Label Panel
# ============================================================

def load_analysis_panel(
    core,
):

    factor_names = (
        core[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    factor_columns = (
        KEY_COLUMNS
        +
        factor_names
    )

    factors = pd.read_parquet(
        FACTOR_PANEL_PATH,
        columns=factor_columns,
    )

    label_columns = (

        KEY_COLUMNS

        +

        [
            "future_return_label_valid",

            "future_risk_label_valid",

            ALPHA_TARGET,
        ]

        +

        RISK_TARGETS
    )

    labels = pd.read_parquet(
        LABEL_PANEL_PATH,
        columns=label_columns,
    )

    for df in [
        factors,
        labels,
    ]:

        df[
            "analysis_date"
        ] = pd.to_datetime(
            df[
                "analysis_date"
            ]
        )

        df[
            "window"
        ] = pd.to_numeric(
            df[
                "window"
            ]
        ).astype(int)

        df[
            "security_id"
        ] = (
            df[
                "security_id"
            ]
            .astype("string")
            .str.strip()
        )

    if factors[
        KEY_COLUMNS
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate factor keys."
        )

    if labels[
        KEY_COLUMNS
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate label keys."
        )

    panel = factors.merge(

        labels,

        on=KEY_COLUMNS,

        how="left",

        validate="one_to_one",

        indicator=True,
    )

    if (
        panel[
            "_merge"
        ]
        !=
        "both"
    ).any():

        raise RuntimeError(
            "Factor/label mismatch."
        )

    panel = panel.drop(
        columns="_merge"
    )

    return panel


# ============================================================
# 9. Winsorize + Standardize Controls
# ============================================================

def prepare_controls(
    group,
):

    result = pd.DataFrame(
        index=group.index
    )

    for column in CONTROL_COLUMNS:

        x = pd.to_numeric(
            group[
                column
            ],
            errors="coerce",
        )

        valid = (
            x.notna()
            &
            np.isfinite(
                x
            )
        )

        result[
            column
        ] = np.nan

        if valid.sum() < 2:

            continue

        low = x.loc[
            valid
        ].quantile(
            CONTROL_WINSOR_LOW
        )

        high = x.loc[
            valid
        ].quantile(
            CONTROL_WINSOR_HIGH
        )

        w = (
            x.loc[
                valid
            ]
            .clip(
                lower=low,
                upper=high,
            )
        )

        mean = float(
            w.mean()
        )

        sd = float(
            w.std(
                ddof=0
            )
        )

        if (
            np.isfinite(sd)
            and
            sd > 0
        ):

            result.loc[
                valid,
                column,
            ] = (
                w
                -
                mean
            ) / sd

        else:

            result.loc[
                valid,
                column,
            ] = 0.0

    return result


# ============================================================
# 10. FWL Neutralization
# ============================================================

def residualize_factor(
    factor,
    industry,
    controls,
):

    factor = pd.to_numeric(
        factor,
        errors="coerce",
    )

    industry = (
        industry
        .astype("string")
    )

    result = pd.Series(
        np.nan,
        index=factor.index,
        dtype=float,
    )

    valid = (

        factor.notna()

        &

        np.isfinite(
            factor
        )

        &

        industry.notna()

        &

        controls.notna().all(
            axis=1
        )

        &

        np.isfinite(
            controls.to_numpy(
                dtype=float
            )
        ).all(
            axis=1
        )
    )

    if (
        valid.sum()
        <
        MIN_CROSS_SECTION_OBS
    ):

        return (
            result,
            {
                "valid_n":
                    int(
                        valid.sum()
                    ),

                "matrix_rank":
                    0,

                "control_count":
                    int(
                        controls.shape[1]
                    ),

                "max_abs_residual_industry_mean":
                    np.nan,

                "max_abs_residual_control_corr":
                    np.nan,
            },
        )

    work = pd.DataFrame(
        {
            "factor":
                factor.loc[
                    valid
                ],

            "industry":
                industry.loc[
                    valid
                ],
        }
    )

    C = (
        controls.loc[
            valid
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Within-industry factor
    # --------------------------------------------------------

    x_within = (

        work[
            "factor"
        ]

        -

        work.groupby(
            "industry"
        )[
            "factor"
        ]
        .transform(
            "mean"
        )
    )

    # --------------------------------------------------------
    # Within-industry controls
    # --------------------------------------------------------

    C_within = pd.DataFrame(
        index=C.index
    )

    for c in C.columns:

        temp = pd.DataFrame(
            {
                "value":
                    C[
                        c
                    ],

                "industry":
                    work[
                        "industry"
                    ],
            }
        )

        C_within[
            c
        ] = (

            temp[
                "value"
            ]

            -

            temp.groupby(
                "industry"
            )[
                "value"
            ]
            .transform(
                "mean"
            )
        )

    X = C_within.to_numpy(
        dtype=float
    )

    y = x_within.to_numpy(
        dtype=float
    )

    beta, _, rank, _ = (
        np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )
    )

    resid = (
        y
        -
        X @ beta
    )

    result.loc[
        valid
    ] = resid

    # ========================================================
    # QA
    # ========================================================

    qa_df = pd.DataFrame(
        {
            "resid":
                resid,

            "industry":
                work[
                    "industry"
                ]
                .to_numpy(),
        }
    )

    max_ind_mean = float(

        qa_df.groupby(
            "industry"
        )[
            "resid"
        ]
        .mean()
        .abs()
        .max()
    )

    max_corr = 0.0

    if np.std(
        resid
    ) > 0:

        for j in range(
            X.shape[1]
        ):

            if np.std(
                X[:, j]
            ) > 0:

                corr = abs(
                    float(
                        np.corrcoef(
                            resid,
                            X[:, j],
                        )[0, 1]
                    )
                )

                max_corr = max(
                    max_corr,
                    corr,
                )

    diagnostics = {

        "valid_n":
            int(
                valid.sum()
            ),

        "matrix_rank":
            int(
                rank
            ),

        "control_count":
            int(
                X.shape[1]
            ),

        "max_abs_residual_industry_mean":
            max_ind_mean,

        "max_abs_residual_control_corr":
            float(
                max_corr
            ),
    }

    return (
        result,
        diagnostics,
    )


# ============================================================
# 11. Spearman IC
# ============================================================

def spearman_ic(
    score,
    target,
    valid_outcome,
):

    score = pd.to_numeric(
        score,
        errors="coerce",
    )

    target = pd.to_numeric(
        target,
        errors="coerce",
    )

    valid = (

        valid_outcome

        &

        score.notna()

        &

        target.notna()

        &

        np.isfinite(
            score
        )

        &

        np.isfinite(
            target
        )
    )

    x = score.loc[
        valid
    ]

    y = target.loc[
        valid
    ]

    n = len(x)

    if (
        n
        <
        MIN_CROSS_SECTION_OBS
    ):

        return (
            np.nan,
            n,
            "INSUFFICIENT_OBS",
        )

    if (
        x.nunique()
        <
        2
        or
        y.nunique()
        <
        2
    ):

        return (
            np.nan,
            n,
            "NO_VARIATION",
        )

    ic = float(
        x.rank(
            method="average"
        ).corr(
            y.rank(
                method="average"
            )
        )
    )

    return (
        ic,
        n,
        "PASS",
    )


# ============================================================
# 12. Portfolio Sort
# ============================================================

def portfolio_spread(
    score,
    future_return,
    valid_return,
):

    score = pd.to_numeric(
        score,
        errors="coerce",
    )

    current_valid = (

        score.notna()

        &

        np.isfinite(
            score
        )
    )

    x = score.loc[
        current_valid
    ]

    if (
        len(x)
        <
        MIN_SORT_OBS
    ):

        return {
            "status":
                "INSUFFICIENT_SORT_OBS",
        }

    if (
        x.nunique()
        <
        N_PORTFOLIOS
    ):

        return {
            "status":
                "INSUFFICIENT_UNIQUE_VALUES",
        }

    try:

        q = pd.qcut(
            x,
            q=N_PORTFOLIOS,
            labels=False,
            duplicates="drop",
        )

    except ValueError:

        return {
            "status":
                "QCUT_FAILURE",
        }

    if q.nunique() != N_PORTFOLIOS:

        return {
            "status":
                "FEWER_THAN_5_PORTFOLIOS",
        }

    portfolio = pd.Series(
        pd.NA,
        index=score.index,
        dtype="Int64",
    )

    portfolio.loc[
        x.index
    ] = (
        q.astype(int)
        +
        1
    )

    future_return = pd.to_numeric(
        future_return,
        errors="coerce",
    )

    q_return = {}

    q_n = {}

    for k in range(
        1,
        6,
    ):

        mask = (

            valid_return

            &

            future_return.notna()

            &

            np.isfinite(
                future_return
            )

            &

            (
                portfolio
                ==
                k
            )
        )

        values = (
            future_return.loc[
                mask
            ]
        )

        q_n[k] = len(
            values
        )

        if (
            len(values)
            >=
            MIN_PORTFOLIO_OBS
        ):

            q_return[k] = float(
                values.mean()
            )

        else:

            q_return[k] = np.nan

    values = np.array(
        [
            q_return[k]
            for k in range(
                1,
                6,
            )
        ],
        dtype=float,
    )

    if (
        not np.isfinite(
            values
        ).all()
    ):

        return {
            "status":
                "INSUFFICIENT_PORTFOLIO_OBS",
        }

    spread = float(
        q_return[5]
        -
        q_return[1]
    )

    rho = float(
        pd.Series(
            values
        ).corr(
            pd.Series(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                ]
            ),
            method="spearman",
        )
    )

    return {

        "status":
            "PASS",

        "q1":
            q_return[1],

        "q2":
            q_return[2],

        "q3":
            q_return[3],

        "q4":
            q_return[4],

        "q5":
            q_return[5],

        "q1_n":
            q_n[1],

        "q5_n":
            q_n[5],

        "spread":
            spread,

        "portfolio_spearman":
            rho,
    }


# ============================================================
# 13. HAC
# ============================================================

def automatic_hac_lag(n):

    if n <= 1:

        return 0

    lag = int(
        np.floor(
            4
            *
            (
                n
                /
                100
            )
            **
            (
                2
                /
                9
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


def hac_inference(values):

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    n = len(x)

    if n < 2:

        return (
            np.nan,
            np.nan,
            0,
        )

    mean_x = float(
        x.mean()
    )

    residual = (
        x
        -
        mean_x
    )

    lag = (
        automatic_hac_lag(
            n
        )
        if USE_HAC
        else
        0
    )

    gamma0 = float(
        np.dot(
            residual,
            residual,
        )
        /
        n
    )

    long_run_var = gamma0

    for ell in range(
        1,
        lag + 1,
    ):

        gamma = float(
            np.dot(
                residual[
                    ell:
                ],
                residual[
                    :-ell
                ],
            )
            /
            n
        )

        weight = (
            1
            -
            ell
            /
            (
                lag
                +
                1
            )
        )

        long_run_var += (
            2
            *
            weight
            *
            gamma
        )

    long_run_var = max(
        long_run_var,
        0
    )

    se = np.sqrt(
        long_run_var
        /
        n
    )

    t = (
        mean_x
        /
        se
        if se > 0
        else np.nan
    )

    return (
        float(se),
        float(t),
        int(lag),
    )


# ============================================================
# 14. Main Monthly Analysis
# ============================================================

def run_monthly_analysis(
    panel,
    core,
):

    factor_names = (
        core[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    alpha_rows = []

    spread_rows = []

    risk_rows = []

    qa_rows = []

    exposure_rows = []

    grouped = panel.groupby(
        [
            "analysis_date",
            "window",
        ],
        sort=True,
    )

    total_jobs = grouped.ngroups

    for job_no, (
        (
            analysis_date,
            window,
        ),
        group,
    ) in enumerate(
        grouped,
        start=1,
    ):

        print(
            f"[{job_no:3d}/{total_jobs}] "
            f"{analysis_date.date()} | "
            f"W={window}"
        )

        group = (
            group.copy()
            .reset_index(
                drop=True
            )
        )

        controls = prepare_controls(
            group
        )

        industry = (
            group[
                "industry_id1"
            ]
            .astype("string")
        )

        complete_controls = (

            industry.notna()

            &

            controls.notna().all(
                axis=1
            )

            &

            np.isfinite(
                controls.to_numpy(
                    dtype=float
                )
            ).all(
                axis=1
            )
        )

        control_coverage = float(
            complete_controls.mean()
        )

        return_valid = (
            group[
                "future_return_label_valid"
            ]
            .fillna(False)
            .astype(bool)
        )

        risk_valid = (
            group[
                "future_risk_label_valid"
            ]
            .fillna(False)
            .astype(bool)
        )

        for (
            factor_order,
            factor_name,
        ) in enumerate(
            factor_names,
            start=1,
        ):

            factor = pd.to_numeric(
                group[
                    factor_name
                ],
                errors="coerce",
            )

            factor_valid = (

                factor.notna()

                &

                np.isfinite(
                    factor
                )
            )

            # =================================================
            # Common current-information sample
            # =================================================

            common = (

                factor_valid

                &

                complete_controls
            )

            raw_common = factor.where(
                common
            )

            # =================================================
            # Factor-control exposure diagnostics
            # =================================================

            for control_name in (
                CONTROL_COLUMNS
            ):

                mask = (

                    common

                    &

                    controls[
                        control_name
                    ]
                    .notna()
                )

                if (
                    mask.sum()
                    >=
                    MIN_CROSS_SECTION_OBS
                ):

                    rho = float(

                        factor.loc[
                            mask
                        ]
                        .rank(
                            method="average"
                        )

                        .corr(

                            controls.loc[
                                mask,
                                control_name,
                            ]
                            .rank(
                                method="average"
                            )
                        )
                    )

                else:

                    rho = np.nan

                exposure_rows.append(
                    {

                        "analysis_date":
                            analysis_date,

                        "window":
                            int(window),

                        "factor_order":
                            factor_order,

                        "factor_name":
                            factor_name,

                        "control_name":
                            control_name,

                        "pair_n":
                            int(
                                mask.sum()
                            ),

                        "spearman_exposure":
                            rho,
                    }
                )

            # =================================================
            # Industry + Size on same common sample
            # =================================================

            size_controls = (
                controls[
                    SIZE_ONLY_COLUMNS
                ]
                .copy()
            )

            size_controls.loc[
                ~common,
                :
            ] = np.nan

            (
                ind_size_score,
                ind_size_qa,
            ) = residualize_factor(

                factor=
                    factor.where(
                        common
                    ),

                industry=
                    industry,

                controls=
                    size_controls,
            )

            # =================================================
            # Full traditional controls
            # =================================================

            full_controls = (
                controls[
                    FULL_CONTROL_COLUMNS
                ]
                .copy()
            )

            full_controls.loc[
                ~common,
                :
            ] = np.nan

            (
                full_score,
                full_qa,
            ) = residualize_factor(

                factor=
                    factor.where(
                        common
                    ),

                industry=
                    industry,

                controls=
                    full_controls,
            )

            qa_rows.append(
                {

                    "analysis_date":
                        analysis_date,

                    "window":
                        int(window),

                    "factor_order":
                        factor_order,

                    "factor_name":
                        factor_name,

                    "factor_universe_n":
                        int(
                            len(group)
                        ),

                    "factor_nonmissing_n":
                        int(
                            factor_valid.sum()
                        ),

                    "common_sample_n":
                        int(
                            common.sum()
                        ),

                    "control_complete_coverage":
                        control_coverage,

                    "industry_size_rank":
                        ind_size_qa[
                            "matrix_rank"
                        ],

                    "full_control_rank":
                        full_qa[
                            "matrix_rank"
                        ],

                    "full_control_count":
                        full_qa[
                            "control_count"
                        ],

                    "ind_size_max_abs_industry_mean":
                        ind_size_qa[
                            "max_abs_residual_industry_mean"
                        ],

                    "ind_size_max_abs_control_corr":
                        ind_size_qa[
                            "max_abs_residual_control_corr"
                        ],

                    "full_max_abs_industry_mean":
                        full_qa[
                            "max_abs_residual_industry_mean"
                        ],

                    "full_max_abs_control_corr":
                        full_qa[
                            "max_abs_residual_control_corr"
                        ],
                }
            )

            scores = {

                "RAW_COMMON":
                    raw_common,

                "INDUSTRY_SIZE_COMMON":
                    ind_size_score,

                "FULL_CHARACTERISTIC_NEUTRAL":
                    full_score,
            }

            # =================================================
            # Same future outcomes for all three methods
            # =================================================

            for method, score in (
                scores.items()
            ):

                # ---------------------------------------------
                # Alpha Rank IC
                # ---------------------------------------------

                (
                    alpha_ic,
                    pair_n,
                    status,
                ) = spearman_ic(

                    score,

                    group[
                        ALPHA_TARGET
                    ],

                    return_valid,
                )

                alpha_rows.append(
                    {

                        "analysis_date":
                            analysis_date,

                        "window":
                            int(window),

                        "factor_order":
                            factor_order,

                        "factor_name":
                            factor_name,

                        "method":
                            method,

                        "target_name":
                            ALPHA_TARGET,

                        "common_sample_n":
                            int(
                                common.sum()
                            ),

                        "pair_n":
                            int(
                                pair_n
                            ),

                        "alpha_rank_ic":
                            alpha_ic,

                        "ic_status":
                            status,
                    }
                )

                # ---------------------------------------------
                # Alpha Portfolio
                # ---------------------------------------------

                p = portfolio_spread(

                    score,

                    group[
                        ALPHA_TARGET
                    ],

                    return_valid,
                )

                spread_rows.append(
                    {

                        "analysis_date":
                            analysis_date,

                        "window":
                            int(window),

                        "factor_order":
                            factor_order,

                        "factor_name":
                            factor_name,

                        "method":
                            method,

                        "target_name":
                            ALPHA_TARGET,

                        "common_sample_n":
                            int(
                                common.sum()
                            ),

                        "spread_status":
                            p.get(
                                "status"
                            ),

                        "q1_return":
                            p.get(
                                "q1",
                                np.nan,
                            ),

                        "q2_return":
                            p.get(
                                "q2",
                                np.nan,
                            ),

                        "q3_return":
                            p.get(
                                "q3",
                                np.nan,
                            ),

                        "q4_return":
                            p.get(
                                "q4",
                                np.nan,
                            ),

                        "q5_return":
                            p.get(
                                "q5",
                                np.nan,
                            ),

                        "q5_minus_q1":
                            p.get(
                                "spread",
                                np.nan,
                            ),

                        "q5_minus_q1_bps":
                            (
                                p.get(
                                    "spread",
                                    np.nan,
                                )
                                *
                                10000
                            ),

                        "portfolio_spearman":
                            p.get(
                                "portfolio_spearman",
                                np.nan,
                            ),
                    }
                )

                # ---------------------------------------------
                # Risk IC
                # ---------------------------------------------

                for target in (
                    RISK_TARGETS
                ):

                    (
                        risk_ic,
                        risk_n,
                        risk_status,
                    ) = spearman_ic(

                        score,

                        group[
                            target
                        ],

                        risk_valid,
                    )

                    risk_rows.append(
                        {

                            "analysis_date":
                                analysis_date,

                            "window":
                                int(window),

                            "factor_order":
                                factor_order,

                            "factor_name":
                                factor_name,

                            "method":
                                method,

                            "target_name":
                                target,

                            "common_sample_n":
                                int(
                                    common.sum()
                                ),

                            "pair_n":
                                int(
                                    risk_n
                                ),

                            "risk_rank_ic":
                                risk_ic,

                            "ic_status":
                                risk_status,
                        }
                    )

    return (

        pd.DataFrame(
            alpha_rows
        ),

        pd.DataFrame(
            spread_rows
        ),

        pd.DataFrame(
            risk_rows
        ),

        pd.DataFrame(
            qa_rows
        ),

        pd.DataFrame(
            exposure_rows
        ),
    )


# ============================================================
# 15. Summary Helpers
# ============================================================

def summarize_signal(
    df,
    value_col,
    status_col,
):

    passed = df[
        df[
            status_col
        ]
        ==
        "PASS"
    ].copy()

    group_cols = [

        "factor_order",

        "factor_name",

        "method",

        "window",

        "target_name",
    ]

    rows = []

    for keys, group in (
        passed.groupby(
            group_cols,
            sort=False,
        )
    ):

        (
            factor_order,
            factor_name,
            method,
            window,
            target_name,
        ) = keys

        group = (
            group.sort_values(
                "analysis_date"
            )
        )

        x = (
            group[
                value_col
            ]
            .to_numpy(
                dtype=float
            )
        )

        se, t_stat, lag = (
            hac_inference(
                x
            )
        )

        rows.append(
            {

                "factor_order":
                    int(
                        factor_order
                    ),

                "factor_name":
                    factor_name,

                "method":
                    method,

                "window":
                    int(
                        window
                    ),

                "target_name":
                    target_name,

                "month_count":
                    int(
                        len(x)
                    ),

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
                        len(x)
                        >=
                        2
                        else
                        np.nan
                    ),

                "positive_share":
                    float(
                        np.mean(
                            x > 0
                        )
                    ),

                "negative_share":
                    float(
                        np.mean(
                            x < 0
                        )
                    ),

                "hac_lag":
                    lag,

                "hac_se":
                    se,

                "hac_t_stat":
                    t_stat,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 16. Retention Table
# ============================================================

def get_method_value(
    df,
    factor,
    window,
    method,
    value_column,
    target=None,
):

    x = df[
        (
            df[
                "factor_name"
            ]
            ==
            factor
        )
        &
        (
            df[
                "window"
            ]
            ==
            window
        )
        &
        (
            df[
                "method"
            ]
            ==
            method
        )
    ]

    if target is not None:

        x = x[
            x[
                "target_name"
            ]
            ==
            target
        ]

    if len(x) != 1:

        return np.nan

    return float(
        x[
            value_column
        ]
        .iloc[0]
    )


def build_retention(
    alpha_summary,
    spread_summary,
    risk_summary,
    core,
):

    rows = []

    for row in core.itertuples(
        index=False
    ):

        factor = row.factor_name

        factor_order = row.priority

        for window in EXPECTED_WINDOWS:

            raw_alpha = get_method_value(
                alpha_summary,
                factor,
                window,
                "RAW_COMMON",
                "mean_value",
                ALPHA_TARGET,
            )

            size_alpha = get_method_value(
                alpha_summary,
                factor,
                window,
                "INDUSTRY_SIZE_COMMON",
                "mean_value",
                ALPHA_TARGET,
            )

            full_alpha = get_method_value(
                alpha_summary,
                factor,
                window,
                "FULL_CHARACTERISTIC_NEUTRAL",
                "mean_value",
                ALPHA_TARGET,
            )

            raw_spread = get_method_value(
                spread_summary,
                factor,
                window,
                "RAW_COMMON",
                "mean_value",
                ALPHA_TARGET,
            )

            size_spread = get_method_value(
                spread_summary,
                factor,
                window,
                "INDUSTRY_SIZE_COMMON",
                "mean_value",
                ALPHA_TARGET,
            )

            full_spread = get_method_value(
                spread_summary,
                factor,
                window,
                "FULL_CHARACTERISTIC_NEUTRAL",
                "mean_value",
                ALPHA_TARGET,
            )

            risk_values = {}

            for method in METHODS:

                temp = risk_summary[
                    (
                        risk_summary[
                            "factor_name"
                        ]
                        ==
                        factor
                    )
                    &
                    (
                        risk_summary[
                            "window"
                        ]
                        ==
                        window
                    )
                    &
                    (
                        risk_summary[
                            "method"
                        ]
                        ==
                        method
                    )
                ]

                risk_values[
                    method
                ] = (
                    float(
                        temp[
                            "mean_value"
                        ]
                        .mean()
                    )
                    if
                    len(temp) > 0
                    else
                    np.nan
                )

            raw_ivol = get_method_value(
                risk_summary,
                factor,
                window,
                "RAW_COMMON",
                "mean_value",
                "future_idio_vol_annualized",
            )

            size_ivol = get_method_value(
                risk_summary,
                factor,
                window,
                "INDUSTRY_SIZE_COMMON",
                "mean_value",
                "future_idio_vol_annualized",
            )

            full_ivol = get_method_value(
                risk_summary,
                factor,
                window,
                "FULL_CHARACTERISTIC_NEUTRAL",
                "mean_value",
                "future_idio_vol_annualized",
            )

            rows.append(
                {

                    "factor_order":
                        factor_order,

                    "factor_name":
                        factor,

                    "window":
                        window,

                    # -----------------------------------------
                    # Alpha IC
                    # -----------------------------------------

                    "raw_common_alpha_ic":
                        raw_alpha,

                    "industry_size_common_alpha_ic":
                        size_alpha,

                    "full_control_alpha_ic":
                        full_alpha,

                    "full_vs_ind_size_alpha_same_sign":
                        (
                            np.sign(
                                full_alpha
                            )
                            ==
                            np.sign(
                                size_alpha
                            )
                            if
                            np.isfinite(
                                full_alpha
                            )
                            and
                            np.isfinite(
                                size_alpha
                            )
                            else
                            False
                        ),

                    "full_vs_ind_size_alpha_abs_retention":
                        (
                            abs(
                                full_alpha
                            )
                            /
                            abs(
                                size_alpha
                            )
                            if
                            np.isfinite(
                                size_alpha
                            )
                            and
                            abs(
                                size_alpha
                            )
                            >=
                            MIN_ALPHA_IC_BASELINE
                            and
                            np.isfinite(
                                full_alpha
                            )
                            else
                            np.nan
                        ),

                    # -----------------------------------------
                    # Portfolio spread
                    # -----------------------------------------

                    "raw_common_spread":
                        raw_spread,

                    "industry_size_common_spread":
                        size_spread,

                    "full_control_spread":
                        full_spread,

                    "full_control_spread_bps":
                        (
                            full_spread
                            *
                            10000
                            if
                            np.isfinite(
                                full_spread
                            )
                            else
                            np.nan
                        ),

                    "full_vs_ind_size_spread_same_sign":
                        (
                            np.sign(
                                full_spread
                            )
                            ==
                            np.sign(
                                size_spread
                            )
                            if
                            np.isfinite(
                                full_spread
                            )
                            and
                            np.isfinite(
                                size_spread
                            )
                            else
                            False
                        ),

                    "full_vs_ind_size_spread_abs_retention":
                        (
                            abs(
                                full_spread
                            )
                            /
                            abs(
                                size_spread
                            )
                            if
                            np.isfinite(
                                size_spread
                            )
                            and
                            abs(
                                size_spread
                            )
                            >=
                            MIN_ALPHA_SPREAD_BASELINE
                            and
                            np.isfinite(
                                full_spread
                            )
                            else
                            np.nan
                        ),

                    # -----------------------------------------
                    # Mean Risk IC
                    # -----------------------------------------

                    "raw_common_mean_risk_ic":
                        risk_values[
                            "RAW_COMMON"
                        ],

                    "industry_size_common_mean_risk_ic":
                        risk_values[
                            "INDUSTRY_SIZE_COMMON"
                        ],

                    "full_control_mean_risk_ic":
                        risk_values[
                            "FULL_CHARACTERISTIC_NEUTRAL"
                        ],

                    # -----------------------------------------
                    # IVOL
                    # -----------------------------------------

                    "raw_common_ivol_ic":
                        raw_ivol,

                    "industry_size_common_ivol_ic":
                        size_ivol,

                    "full_control_ivol_ic":
                        full_ivol,

                    "full_vs_ind_size_ivol_same_sign":
                        (
                            np.sign(
                                full_ivol
                            )
                            ==
                            np.sign(
                                size_ivol
                            )
                            if
                            np.isfinite(
                                full_ivol
                            )
                            and
                            np.isfinite(
                                size_ivol
                            )
                            else
                            False
                        ),

                    "full_vs_ind_size_ivol_abs_retention":
                        (
                            abs(
                                full_ivol
                            )
                            /
                            abs(
                                size_ivol
                            )
                            if
                            np.isfinite(
                                size_ivol
                            )
                            and
                            abs(
                                size_ivol
                            )
                            >=
                            MIN_RISK_IC_BASELINE
                            and
                            np.isfinite(
                                full_ivol
                            )
                            else
                            np.nan
                        ),

                    "factor_sign_flipped":
                        False,

                    "window_selected":
                        False,

                    "factor_removed":
                        False,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 17. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 7 - Step 2"
    )

    print(
        "Traditional Stock Characteristic Controls"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Upstream
    # --------------------------------------------------------

    step6_metadata, step1_metadata = (
        validate_upstream()
    )

    core = load_core()

    factor_names = (
        core[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # Load factor/outcome panel
    # --------------------------------------------------------

    print()
    print(
        "[0] Load frozen factor/outcome panel"
    )

    panel = load_analysis_panel(
        core
    )

    analysis_dates = sorted(
        panel[
            "analysis_date"
        ]
        .unique()
    )

    # --------------------------------------------------------
    # Build characteristics
    # --------------------------------------------------------

    characteristics = (
        build_characteristic_panel(
            analysis_dates
        )
    )

    panel = panel.merge(

        characteristics,

        on=[
            "analysis_date",
            "security_id",
        ],

        how="left",

        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Characteristic coverage QA
    # --------------------------------------------------------

    qa_rows = []

    for (
        analysis_date,
        window,
    ), group in panel.groupby(
        [
            "analysis_date",
            "window",
        ],
        sort=True,
    ):

        row = {

            "analysis_date":
                analysis_date,

            "window":
                int(window),

            "universe_n":
                int(
                    len(group)
                ),
        }

        for c in (
            [
                "industry_id1"
            ]
            +
            CONTROL_COLUMNS
        ):

            row[
                f"{c}_coverage"
            ] = float(
                group[
                    c
                ]
                .notna()
                .mean()
            )

        row[
            "all_controls_complete_share"
        ] = float(

            (
                group[
                    CONTROL_COLUMNS
                ]
                .notna()
                .all(
                    axis=1
                )

                &

                group[
                    "industry_id1"
                ]
                .notna()
            )
            .mean()
        )

        qa_rows.append(
            row
        )

    characteristic_qa = pd.DataFrame(
        qa_rows
    )

    save_csv_atomic(
        characteristic_qa,
        CHARACTERISTIC_QA_PATH,
    )

    # --------------------------------------------------------
    # Run monthly analysis
    # --------------------------------------------------------

    print()
    print(
        "[2] Run common-sample traditional-control validation"
    )

    (
        alpha_monthly,
        spread_monthly,
        risk_monthly,
        neutralization_qa,
        exposure_monthly,
    ) = run_monthly_analysis(
        panel,
        core,
    )

    save_csv_atomic(
        alpha_monthly,
        MONTHLY_ALPHA_IC_PATH,
    )

    save_csv_atomic(
        spread_monthly,
        MONTHLY_ALPHA_SPREAD_PATH,
    )

    save_csv_atomic(
        risk_monthly,
        MONTHLY_RISK_IC_PATH,
    )

    save_csv_atomic(
        neutralization_qa,
        NEUTRALIZATION_QA_PATH,
    )

    # --------------------------------------------------------
    # Factor-control exposure summary
    # --------------------------------------------------------

    exposure_summary = (

        exposure_monthly.groupby(
            [
                "factor_order",
                "factor_name",
                "control_name",
                "window",
            ],
            as_index=False,
        )

        .agg(

            month_count=(
                "spearman_exposure",
                "count",
            ),

            mean_spearman_exposure=(
                "spearman_exposure",
                "mean",
            ),

            median_spearman_exposure=(
                "spearman_exposure",
                "median",
            ),
        )
    )

    save_csv_atomic(
        exposure_summary,
        EXPOSURE_PATH,
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    print()
    print(
        "[3] Build Alpha IC summary"
    )

    alpha_summary = summarize_signal(

        alpha_monthly,

        "alpha_rank_ic",

        "ic_status",
    )

    alpha_summary = alpha_summary.rename(
        columns={
            "mean_value":
                "mean_alpha_ic",

            "median_value":
                "median_alpha_ic",

            "std_value":
                "std_alpha_ic",
        }
    )

    save_csv_atomic(
        alpha_summary,
        ALPHA_SUMMARY_PATH,
    )

    print()
    print(
        "[4] Build Alpha portfolio summary"
    )

    spread_summary = summarize_signal(

        spread_monthly,

        "q5_minus_q1",

        "spread_status",
    )

    spread_summary[
        "mean_q5_minus_q1_bps"
    ] = (
        spread_summary[
            "mean_value"
        ]
        *
        10000
    )

    save_csv_atomic(
        spread_summary,
        ALPHA_PORTFOLIO_SUMMARY_PATH,
    )

    print()
    print(
        "[5] Build Risk IC summary"
    )

    risk_summary = summarize_signal(

        risk_monthly,

        "risk_rank_ic",

        "ic_status",
    )

    risk_summary = risk_summary.rename(
        columns={
            "mean_value":
                "mean_risk_ic",

            "median_value":
                "median_risk_ic",

            "std_value":
                "std_risk_ic",
        }
    )

    save_csv_atomic(
        risk_summary,
        RISK_SUMMARY_PATH,
    )

    # --------------------------------------------------------
    # Build compatible copies for retention helper
    # --------------------------------------------------------

    alpha_retention = (
        alpha_summary.rename(
            columns={
                "mean_alpha_ic":
                    "mean_value",
            }
        )
    )

    risk_retention = (
        risk_summary.rename(
            columns={
                "mean_risk_ic":
                    "mean_value",
            }
        )
    )

    # --------------------------------------------------------
    # Retention
    # --------------------------------------------------------

    print()
    print(
        "[6] Build incremental-signal retention table"
    )

    retention = build_retention(

        alpha_retention,

        spread_summary,

        risk_retention,

        core,
    )

    save_csv_atomic(
        retention,
        RETENTION_PATH,
    )

    # --------------------------------------------------------
    # Formal QA
    # --------------------------------------------------------

    max_full_industry_mean = (
        neutralization_qa[
            "full_max_abs_industry_mean"
        ]
        .dropna()
        .abs()
        .max()
    )

    max_full_control_corr = (
        neutralization_qa[
            "full_max_abs_control_corr"
        ]
        .dropna()
        .abs()
        .max()
    )

    formal_qa = {

        "factor_count":
            int(
                len(core)
            ),

        "factor_count_ok":
            bool(
                len(core)
                ==
                EXPECTED_FACTOR_COUNT
            ),

        "observed_windows":
            sorted(
                panel[
                    "window"
                ]
                .unique()
                .astype(int)
                .tolist()
            ),

        "window_set_ok":
            bool(
                sorted(
                    panel[
                        "window"
                    ]
                    .unique()
                    .astype(int)
                    .tolist()
                )
                ==
                EXPECTED_WINDOWS
            ),

        "minimum_all_control_coverage":
            float(
                characteristic_qa[
                    "all_controls_complete_share"
                ]
                .min()
            ),

        "max_abs_full_residual_industry_mean":
            float(
                max_full_industry_mean
            ),

        "max_abs_full_residual_control_corr":
            float(
                max_full_control_corr
            ),

        "neutralization_orthogonality_ok":
            bool(

                max_full_industry_mean
                <
                1e-8

                and

                max_full_control_corr
                <
                1e-8
            ),

        "outcome_used_to_construct_controls":
            False,

        "factor_sign_flip":
            False,

        "window_selection":
            False,

        "factor_selection":
            False,
    }

    formal_qa[
        "all_formal_qa_pass"
    ] = bool(

        formal_qa[
            "factor_count_ok"
        ]

        and

        formal_qa[
            "window_set_ok"
        ]

        and

        formal_qa[
            "neutralization_orthogonality_ok"
        ]
    )

    if (
        not formal_qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            "Day-7 Step-2 formal QA failed.\n"
            f"{formal_qa}"
        )

    # --------------------------------------------------------
    # Design metadata
    # --------------------------------------------------------

    design_payload = {

        "research_day":
            7,

        "step":
            (
                "Step2_Traditional_"
                "Stock_Characteristic_Controls"
            ),

        "parent_day6_step6_design_hash":
            step6_metadata.get(
                "step6_design_hash"
            ),

        "parent_day7_step1_design_hash":
            step1_metadata.get(
                "day7_step1_design_hash"
            ),

        "factors":
            factor_names,

        "windows":
            EXPECTED_WINDOWS,

        "comparison_methods":
            METHODS,

        "traditional_controls":
            {

                "industry":
                    "industry_id1",

                "size":
                    "log(market_value_t)",

                "momentum":
                    (
                        "Compounded holding return "
                        "over trading days t-119 to t-20."
                    ),

                "short_term_reversal":
                    (
                        "Compounded holding return "
                        "over trading days t-19 to t."
                    ),

                "volatility":
                    (
                        "Annualized standard deviation "
                        "of holding returns over t-59 to t."
                    ),

                "turnover":
                    (
                        "log(1 + mean turnoverRate "
                        "over t-19 to t)."
                    ),
            },

        "holding_return_rule":
            {

                "suspension":
                    0.0,

                "active":
                    "return_last_trade_simple",

                "unresolved":
                    "NA",
            },

        "control_preprocessing":
            (
                "Cross-sectional 1%/99% winsorization "
                "followed by z-standardization."
            ),

        "full_neutralization":
            (
                "Factor residual from industry fixed effects "
                "+ size + momentum + reversal + volatility "
                "+ turnover using FWL."
            ),

        "common_sample_design":
            True,

        "future_outcome_used_in_control_construction":
            False,

        "factor_sign_flip":
            False,

        "factor_selection":
            False,

        "window_selection":
            False,
    }

    metadata = {

        **design_payload,

        "day7_step2_design_hash":
            canonical_hash(
                design_payload
            ),

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "formal_qa":
            formal_qa,

        "outputs":
            {

                "characteristic_panel":
                    str(
                        CHARACTERISTIC_PANEL_PATH
                    ),

                "characteristic_qa":
                    str(
                        CHARACTERISTIC_QA_PATH
                    ),

                "factor_control_exposure":
                    str(
                        EXPOSURE_PATH
                    ),

                "alpha_summary":
                    str(
                        ALPHA_SUMMARY_PATH
                    ),

                "portfolio_summary":
                    str(
                        ALPHA_PORTFOLIO_SUMMARY_PATH
                    ),

                "risk_summary":
                    str(
                        RISK_SUMMARY_PATH
                    ),

                "signal_retention":
                    str(
                        RETENTION_PATH
                    ),

                "neutralization_qa":
                    str(
                        NEUTRALIZATION_QA_PATH
                    ),
            },
    }

    save_json_atomic(
        metadata,
        METADATA_PATH,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "Day-7 Step-2 Formal QA"
    )

    print("=" * 80)

    print(
        f"Minimum control-complete coverage: "
        f"{formal_qa['minimum_all_control_coverage']:.4%}"
    )

    print(
        f"Max |residual industry mean|: "
        f"{formal_qa['max_abs_full_residual_industry_mean']:.3e}"
    )

    print(
        f"Max |residual-control correlation|: "
        f"{formal_qa['max_abs_full_residual_control_corr']:.3e}"
    )

    print(
        f"Formal QA pass: "
        f"{formal_qa['all_formal_qa_pass']}"
    )

    print()
    print("=" * 80)

    print(
        "Incremental Signal Retention"
    )

    print("=" * 80)

    display_columns = [

        "factor_name",

        "window",

        "industry_size_common_alpha_ic",

        "full_control_alpha_ic",

        "full_vs_ind_size_alpha_abs_retention",

        "industry_size_common_spread",

        "full_control_spread_bps",

        "industry_size_common_ivol_ic",

        "full_control_ivol_ic",

        "full_vs_ind_size_ivol_abs_retention",
    ]

    print(
        retention[
            display_columns
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
        "Day 7 Step 2 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()