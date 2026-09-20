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
# Day 7 Step 1
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
# Day 7 Step 2
# ------------------------------------------------------------

STEP2_DIR = (
    DAY7_ROOT
    / "02_stage2_traditional_characteristic_controls"
)

CHARACTERISTIC_PANEL_PATH = (
    STEP2_DIR
    / "traditional_characteristic_panel.parquet"
)

STEP2_METADATA_PATH = (
    STEP2_DIR
    / "day7_step2_metadata.json"
)


# ------------------------------------------------------------
# Day 7 Step 3
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY7_ROOT
    / "03_stage3_fama_macbeth_regression"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MONTHLY_COEF_PATH = (
    OUTPUT_DIR
    / "monthly_fmb_coefficients.csv"
)

FMB_SUMMARY_PATH = (
    OUTPUT_DIR
    / "fama_macbeth_summary.csv"
)

MODEL_COMPARISON_PATH = (
    OUTPUT_DIR
    / "fmb_model_comparison.csv"
)

CROSS_WINDOW_PATH = (
    OUTPUT_DIR
    / "fmb_cross_window_consistency.csv"
)

CROSS_SECTION_QA_PATH = (
    OUTPUT_DIR
    / "fmb_cross_section_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day7_step3_metadata.json"
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


# ------------------------------------------------------------
# Fama-MacBeth model specifications
# ------------------------------------------------------------

MODEL_SPECS = [

    "M1_NETWORK_ONLY",

    "M2_INDUSTRY_SIZE",

    "M3_FULL_CONTROLS",
]


# ------------------------------------------------------------
# Outcome targets
# ------------------------------------------------------------

PRIMARY_TARGET = (
    "future_excess_return"
)

ROBUSTNESS_TARGET = (
    "future_total_return"
)

TARGETS = [

    PRIMARY_TARGET,

    ROBUSTNESS_TARGET,
]


# ------------------------------------------------------------
# Traditional controls
# ------------------------------------------------------------

CONTROL_COLUMNS = [

    "log_market_value",

    "momentum_120_20",

    "reversal_20",

    "volatility_60",

    "log_turnover_20",
]

SIZE_CONTROL = [
    "log_market_value"
]

FULL_CONTROLS = CONTROL_COLUMNS.copy()


# ------------------------------------------------------------
# Pre-processing
# ------------------------------------------------------------

CONTROL_WINSOR_LOW = 0.01

CONTROL_WINSOR_HIGH = 0.99


# Network factor:
#
# - no winsorization
# - cross-sectional z-standardization only
#
# Therefore FMB beta = future return associated with
# a one-standard-deviation increase in the network factor.
# ------------------------------------------------------------


# ------------------------------------------------------------
# Cross-sectional minimum sample
# ------------------------------------------------------------

MIN_CROSS_SECTION_OBS = 100


# ------------------------------------------------------------
# Numerical tolerances
# ------------------------------------------------------------

MIN_FACTOR_SD = 1e-12

MIN_RESIDUAL_VARIANCE = 1e-14


# ------------------------------------------------------------
# HAC
# ------------------------------------------------------------

USE_HAC = True


# ------------------------------------------------------------
# Retention ratio guard
#
# 0.0002 = 2 bp per one-SD network-factor increase.
# Avoid unstable ratios when M2 coefficient is nearly zero.
# ------------------------------------------------------------

MIN_BETA_BASELINE = 0.0002


# ============================================================
# 2. Keys
# ============================================================

KEY_COLUMNS = [

    "analysis_date",

    "window",

    "master_index",

    "security_id",
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
# 4. Upstream Validation
# ============================================================

def validate_upstream():

    step6 = load_json(
        STEP6_METADATA_PATH
    )

    step1 = load_json(
        STEP1_METADATA_PATH
    )

    step2 = load_json(
        STEP2_METADATA_PATH
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
            "Day-6 Step-6 formal QA failed."
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
            "Day-7 Step-1 formal QA failed."
        )

    if (
        step2.get(
            "formal_qa",
            {}
        ).get(
            "all_formal_qa_pass"
        )
        is not True
    ):

        raise RuntimeError(
            "Day-7 Step-2 formal QA failed."
        )

    for path in [

        FACTOR_PANEL_PATH,

        LABEL_PANEL_PATH,

        CHARACTERISTIC_PANEL_PATH,

        FROZEN_CORE_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing input:\n{path}"
            )

    return (
        step6,
        step1,
        step2,
    )


# ============================================================
# 5. Frozen Core Factors
# ============================================================

def load_core_factors():

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

    if (
        len(core)
        !=
        EXPECTED_FACTOR_COUNT
    ):

        raise RuntimeError(
            "Unexpected frozen factor count."
        )

    if (
        core[
            "factor_name"
        ]
        .duplicated()
        .any()
    ):

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
# 6. Load Factor + Outcome + Characteristic Panel
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

    # --------------------------------------------------------
    # Factor panel
    # --------------------------------------------------------

    factor_columns = (
        KEY_COLUMNS
        +
        factor_names
    )

    factors = pd.read_parquet(
        FACTOR_PANEL_PATH,
        columns=factor_columns,
    )

    # --------------------------------------------------------
    # Forward-return labels
    # --------------------------------------------------------

    label_columns = (

        KEY_COLUMNS

        +

        [
            "future_return_label_valid",

            PRIMARY_TARGET,

            ROBUSTNESS_TARGET,
        ]
    )

    labels = pd.read_parquet(
        LABEL_PANEL_PATH,
        columns=label_columns,
    )

    # --------------------------------------------------------
    # Traditional characteristics
    # --------------------------------------------------------

    characteristic_columns = [

        "analysis_date",

        "security_id",

        "industry_id1",

    ] + CONTROL_COLUMNS

    characteristics = pd.read_parquet(
        CHARACTERISTIC_PANEL_PATH,
        columns=characteristic_columns,
    )

    # --------------------------------------------------------
    # Normalize keys
    # --------------------------------------------------------

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
            ],
            errors="raise",
        ).astype(int)

        df[
            "master_index"
        ] = pd.to_numeric(
            df[
                "master_index"
            ],
            errors="raise",
        ).astype(
            np.int64
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

    characteristics[
        "analysis_date"
    ] = pd.to_datetime(
        characteristics[
            "analysis_date"
        ]
    )

    characteristics[
        "security_id"
    ] = (
        characteristics[
            "security_id"
        ]
        .astype("string")
        .str.strip()
    )

    characteristics[
        "industry_id1"
    ] = (
        characteristics[
            "industry_id1"
        ]
        .astype("string")
        .str.strip()
    )

    # --------------------------------------------------------
    # Duplicate QA
    # --------------------------------------------------------

    if (
        factors[
            KEY_COLUMNS
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate factor keys."
        )

    if (
        labels[
            KEY_COLUMNS
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate label keys."
        )

    if (
        characteristics[
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

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

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
            "Factor / label mismatch."
        )

    panel = panel.drop(
        columns="_merge"
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

    windows = sorted(
        panel[
            "window"
        ]
        .unique()
        .astype(int)
        .tolist()
    )

    if (
        windows
        !=
        EXPECTED_WINDOWS
    ):

        raise RuntimeError(
            f"Unexpected windows: {windows}"
        )

    return panel


# ============================================================
# 7. Control Preprocessing
#
# Controls are transformed using CURRENT information only.
#
# Important:
# future-return availability does NOT enter the winsorization
# or standardization step.
# ============================================================

def prepare_controls(
    group,
):

    raw = pd.DataFrame(
        index=group.index
    )

    for c in CONTROL_COLUMNS:

        raw[
            c
        ] = pd.to_numeric(
            group[
                c
            ],
            errors="coerce",
        )

    control_complete = np.ones(
        len(group),
        dtype=bool,
    )

    for c in CONTROL_COLUMNS:

        control_complete &= (

            raw[
                c
            ]
            .notna()
            .to_numpy()

            &

            np.isfinite(
                raw[
                    c
                ]
                .to_numpy(
                    dtype=float
                )
            )
        )

    industry = (
        group[
            "industry_id1"
        ]
        .astype("string")
    )

    control_complete &= (
        industry.notna()
        .to_numpy()
    )

    control_complete = pd.Series(
        control_complete,
        index=group.index,
    )

    z = pd.DataFrame(
        np.nan,
        index=group.index,
        columns=CONTROL_COLUMNS,
        dtype=float,
    )

    if (
        control_complete.sum()
        <
        2
    ):

        return (
            z,
            control_complete,
        )

    for c in CONTROL_COLUMNS:

        x = raw.loc[
            control_complete,
            c,
        ]

        low = float(
            x.quantile(
                CONTROL_WINSOR_LOW
            )
        )

        high = float(
            x.quantile(
                CONTROL_WINSOR_HIGH
            )
        )

        x = x.clip(
            lower=low,
            upper=high,
        )

        mean_x = float(
            x.mean()
        )

        sd_x = float(
            x.std(
                ddof=0
            )
        )

        if (
            np.isfinite(
                sd_x
            )
            and
            sd_x
            >
            0
        ):

            z.loc[
                control_complete,
                c,
            ] = (
                x
                -
                mean_x
            ) / sd_x

        else:

            z.loc[
                control_complete,
                c,
            ] = 0.0

    return (
        z,
        control_complete,
    )


# ============================================================
# 8. Factor Standardization
#
# IMPORTANT:
#
# - No factor winsorization.
# - No sign flip.
# - No rank transformation.
# - Standardization uses current-information common universe.
# ============================================================

def standardize_factor(
    factor,
    current_common,
):

    x = pd.to_numeric(
        factor,
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=factor.index,
        dtype=float,
    )

    valid = (

        current_common

        &

        x.notna()

        &

        np.isfinite(
            x
        )
    )

    if (
        valid.sum()
        <
        2
    ):

        return result

    values = x.loc[
        valid
    ]

    mean_x = float(
        values.mean()
    )

    sd_x = float(
        values.std(
            ddof=0
        )
    )

    if (
        not np.isfinite(
            sd_x
        )
        or
        sd_x
        <=
        MIN_FACTOR_SD
    ):

        return result

    result.loc[
        valid
    ] = (
        values
        -
        mean_x
    ) / sd_x

    return result


# ============================================================
# 9. Within-Industry Demeaning
# ============================================================

def within_demean_series(
    x,
    industry,
):

    temp = pd.DataFrame(
        {
            "x":
                np.asarray(
                    x,
                    dtype=float,
                ),

            "industry":
                np.asarray(
                    industry,
                ),
        }
    )

    return (

        temp[
            "x"
        ]

        -

        temp.groupby(
            "industry"
        )[
            "x"
        ]
        .transform(
            "mean"
        )
    ).to_numpy(
        dtype=float
    )


def within_demean_matrix(
    X,
    industry,
):

    X = np.asarray(
        X,
        dtype=float,
    )

    industry = np.asarray(
        industry,
    )

    result = np.empty_like(
        X,
        dtype=float,
    )

    for j in range(
        X.shape[1]
    ):

        result[
            :,
            j
        ] = within_demean_series(

            X[
                :,
                j
            ],

            industry,
        )

    return result


# ============================================================
# 10. Cross-Sectional Regression
#
# M1:
#
#   y = a + beta * F + e
#
# M2/M3 use Frisch-Waugh-Lovell:
#
#   residualize y and F with respect to
#   industry fixed effects + controls
#
#   beta = Cov(F_resid, y_resid) / Var(F_resid)
#
# This beta is exactly the coefficient on F in the full OLS.
# ============================================================

def fit_cross_section(
    y,
    factor_z,
    industry,
    controls,
    model_spec,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    x = np.asarray(
        factor_z,
        dtype=float,
    )

    industry = np.asarray(
        industry,
    )

    n = len(
        y
    )

    if (
        n
        <
        MIN_CROSS_SECTION_OBS
    ):

        return {
            "status":
                "INSUFFICIENT_OBS",

            "nobs":
                n,
        }

    industry_count = int(
        pd.Series(
            industry
        )
        .nunique()
    )

    industry_sizes = (
        pd.Series(
            industry
        )
        .value_counts()
    )

    singleton_industry_count = int(
        (
            industry_sizes
            ==
            1
        )
        .sum()
    )

    # ========================================================
    # M1: Network only
    # ========================================================

    if (
        model_spec
        ==
        "M1_NETWORK_ONLY"
    ):

        X = np.column_stack(
            [
                np.ones(
                    n
                ),
                x,
            ]
        )

        coef, _, rank, _ = (
            np.linalg.lstsq(
                X,
                y,
                rcond=None,
            )
        )

        if (
            rank
            <
            2
        ):

            return {
                "status":
                    "RANK_DEFICIENT",

                "nobs":
                    n,
            }

        beta = float(
            coef[
                1
            ]
        )

        fitted = (
            X
            @
            coef
        )

        resid = (
            y
            -
            fitted
        )

        sst = float(
            np.sum(
                (
                    y
                    -
                    y.mean()
                )
                **
                2
            )
        )

        sse = float(
            np.sum(
                resid
                **
                2
            )
        )

        r2 = (
            1.0
            -
            sse
            /
            sst
            if
            sst
            >
            0
            else
            np.nan
        )

        return {

            "status":
                "PASS",

            "nobs":
                int(
                    n
                ),

            "industry_count":
                industry_count,

            "singleton_industry_count":
                singleton_industry_count,

            "control_count":
                0,

            "control_rank":
                0,

            "network_beta":
                beta,

            "network_beta_bps":
                beta
                *
                10000.0,

            "partial_r2":
                r2,

            "factor_residual_sd":
                float(
                    np.std(
                        x,
                        ddof=0,
                    )
                ),
        }

    # ========================================================
    # M2 / M3 controls
    # ========================================================

    if (
        model_spec
        ==
        "M2_INDUSTRY_SIZE"
    ):

        control_names = (
            SIZE_CONTROL
        )

    elif (
        model_spec
        ==
        "M3_FULL_CONTROLS"
    ):

        control_names = (
            FULL_CONTROLS
        )

    else:

        raise ValueError(
            f"Unknown model spec: {model_spec}"
        )

    C = (
        controls[
            control_names
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Industry fixed effects:
    # remove industry means from y, x and controls.
    # --------------------------------------------------------

    y_w = within_demean_series(
        y,
        industry,
    )

    x_w = within_demean_series(
        x,
        industry,
    )

    C_w = within_demean_matrix(
        C,
        industry,
    )

    control_rank = int(
        np.linalg.matrix_rank(
            C_w
        )
    )

    required_rank = int(
        C_w.shape[
            1
        ]
    )

    if (
        control_rank
        <
        required_rank
    ):

        return {

            "status":
                "CONTROL_RANK_DEFICIENT",

            "nobs":
                int(
                    n
                ),

            "industry_count":
                industry_count,

            "singleton_industry_count":
                singleton_industry_count,

            "control_count":
                required_rank,

            "control_rank":
                control_rank,
        }

    # --------------------------------------------------------
    # FWL residual of network factor
    # --------------------------------------------------------

    beta_x, _, _, _ = (
        np.linalg.lstsq(
            C_w,
            x_w,
            rcond=None,
        )
    )

    x_resid = (
        x_w
        -
        C_w
        @
        beta_x
    )

    # --------------------------------------------------------
    # FWL residual of future return
    # --------------------------------------------------------

    beta_y, _, _, _ = (
        np.linalg.lstsq(
            C_w,
            y_w,
            rcond=None,
        )
    )

    y_resid = (
        y_w
        -
        C_w
        @
        beta_y
    )

    denominator = float(
        np.dot(
            x_resid,
            x_resid,
        )
    )

    if (
        denominator
        <=
        MIN_RESIDUAL_VARIANCE
    ):

        return {

            "status":
                "NO_FACTOR_RESIDUAL_VARIATION",

            "nobs":
                int(
                    n
                ),

            "industry_count":
                industry_count,

            "singleton_industry_count":
                singleton_industry_count,

            "control_count":
                required_rank,

            "control_rank":
                control_rank,
        }

    network_beta = float(

        np.dot(
            x_resid,
            y_resid,
        )

        /

        denominator
    )

    # --------------------------------------------------------
    # Partial R^2 of network factor conditional on controls
    # --------------------------------------------------------

    restricted_sse = float(
        np.dot(
            y_resid,
            y_resid,
        )
    )

    full_resid = (
        y_resid
        -
        network_beta
        *
        x_resid
    )

    full_sse = float(
        np.dot(
            full_resid,
            full_resid,
        )
    )

    partial_r2 = (
        1.0
        -
        full_sse
        /
        restricted_sse
        if
        restricted_sse
        >
        0
        else
        np.nan
    )

    return {

        "status":
            "PASS",

        "nobs":
            int(
                n
            ),

        "industry_count":
            industry_count,

        "singleton_industry_count":
            singleton_industry_count,

        "control_count":
            required_rank,

        "control_rank":
            control_rank,

        "network_beta":
            network_beta,

        "network_beta_bps":
            network_beta
            *
            10000.0,

        "partial_r2":
            partial_r2,

        "factor_residual_sd":
            float(
                np.std(
                    x_resid,
                    ddof=0,
                )
            ),
    }


# ============================================================
# 11. Calendar-Gap-Aware HAC
#
# Uses actual calendar-month distance rather than compressing
# missing months into artificial adjacent observations.
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

    # to_datetime preserves Series inputs, which do not expose .year/.month.
    # Normalize before positional filtering so Series, lists and indexes all
    # use the same calendar-month lookup without changing the HAC formula.
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

            "hac_lag":
                0,

            "hac_se":
                np.nan,

            "hac_t":
                np.nan,

            "naive_se":
                np.nan,

            "naive_t":
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

    naive_se = (
        sd_x
        /
        np.sqrt(
            n
        )
        if
        sd_x
        >
        0
        else
        np.nan
    )

    naive_t = (
        mean_x
        /
        naive_se
        if
        np.isfinite(
            naive_se
        )
        and
        naive_se
        >
        0
        else
        np.nan
    )

    if not USE_HAC:

        return {

            "hac_lag":
                0,

            "hac_se":
                naive_se,

            "hac_t":
                naive_t,

            "naive_se":
                naive_se,

            "naive_t":
                naive_t,
        }

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
            "Duplicate calendar month in FMB time series."
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

            previous_month = (
                month
                -
                ell
            )

            if (
                previous_month
                in
                residual_map
            ):

                cross_sum += (

                    value

                    *

                    residual_map[
                        previous_month
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

        "hac_lag":
            int(
                lag
            ),

        "hac_se":
            hac_se,

        "hac_t":
            hac_t,

        "naive_se":
            naive_se,

        "naive_t":
            naive_t,
    }


# ============================================================
# 12. Run Monthly Cross-Sectional Regressions
# ============================================================

def run_monthly_fmb(
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

    coef_rows = []

    qa_rows = []

    grouped = panel.groupby(
        [
            "analysis_date",
            "window",
        ],
        sort=True,
    )

    total_jobs = (
        grouped.ngroups
    )

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

        industry = (
            group[
                "industry_id1"
            ]
            .astype("string")
        )

        # ====================================================
        # Current-information controls
        # ====================================================

        (
            controls_z,
            control_complete,
        ) = prepare_controls(
            group
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
            # Current-information common sample
            #
            # Outcome availability is NOT used here.
            # =================================================

            current_common = (

                control_complete

                &

                factor_valid

                &

                industry.notna()
            )

            factor_z = standardize_factor(

                factor,

                current_common,
            )

            qa_base = {

                "analysis_date":
                    analysis_date,

                "window":
                    int(
                        window
                    ),

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

                "control_complete_n":
                    int(
                        control_complete.sum()
                    ),

                "current_common_n":
                    int(
                        current_common.sum()
                    ),

                "current_common_share":
                    float(
                        current_common.mean()
                    ),
            }

            # =================================================
            # Outcomes
            # =================================================

            formal_return_valid = (

                group[
                    "future_return_label_valid"
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
            )

            for target_name in TARGETS:

                y = pd.to_numeric(
                    group[
                        target_name
                    ],
                    errors="coerce",
                )

                outcome_valid = (

                    formal_return_valid

                    &

                    y.notna()

                    &

                    np.isfinite(
                        y
                    )
                )

                # =================================================
                # Exact same regression sample M1/M2/M3
                # =================================================

                regression_sample = (

                    current_common

                    &

                    factor_z.notna()

                    &

                    outcome_valid
                )

                nobs = int(
                    regression_sample.sum()
                )

                qa_rows.append(
                    {

                        **qa_base,

                        "target_name":
                            target_name,

                        "regression_n":
                            nobs,

                        "regression_share_of_current_common":
                            (
                                nobs
                                /
                                current_common.sum()
                                if
                                current_common.sum()
                                >
                                0
                                else
                                np.nan
                            ),
                    }
                )

                if (
                    nobs
                    <
                    MIN_CROSS_SECTION_OBS
                ):

                    for model_spec in MODEL_SPECS:

                        coef_rows.append(
                            {

                                "analysis_date":
                                    analysis_date,

                                "window":
                                    int(
                                        window
                                    ),

                                "factor_order":
                                    factor_order,

                                "factor_name":
                                    factor_name,

                                "factor_orientation":
                                    "ORIGINAL",

                                "target_name":
                                    target_name,

                                "model_spec":
                                    model_spec,

                                "nobs":
                                    nobs,

                                "network_beta":
                                    np.nan,

                                "network_beta_bps":
                                    np.nan,

                                "partial_r2":
                                    np.nan,

                                "status":
                                    "INSUFFICIENT_OBS",
                            }
                        )

                    continue

                idx = group.index[
                    regression_sample
                ]

                y_reg = (
                    y.loc[
                        idx
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                x_reg = (
                    factor_z.loc[
                        idx
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                industry_reg = (
                    industry.loc[
                        idx
                    ]
                    .to_numpy()
                )

                controls_reg = (
                    controls_z.loc[
                        idx,
                        CONTROL_COLUMNS,
                    ]
                )

                for model_spec in MODEL_SPECS:

                    fit = fit_cross_section(

                        y=
                            y_reg,

                        factor_z=
                            x_reg,

                        industry=
                            industry_reg,

                        controls=
                            controls_reg,

                        model_spec=
                            model_spec,
                    )

                    coef_rows.append(
                        {

                            "analysis_date":
                                analysis_date,

                            "window":
                                int(
                                    window
                                ),

                            "factor_order":
                                factor_order,

                            "factor_name":
                                factor_name,

                            "factor_orientation":
                                "ORIGINAL",

                            "target_name":
                                target_name,

                            "model_spec":
                                model_spec,

                            "nobs":
                                fit.get(
                                    "nobs",
                                    nobs,
                                ),

                            "industry_count":
                                fit.get(
                                    "industry_count",
                                    np.nan,
                                ),

                            "singleton_industry_count":
                                fit.get(
                                    "singleton_industry_count",
                                    np.nan,
                                ),

                            "control_count":
                                fit.get(
                                    "control_count",
                                    np.nan,
                                ),

                            "control_rank":
                                fit.get(
                                    "control_rank",
                                    np.nan,
                                ),

                            "network_beta":
                                fit.get(
                                    "network_beta",
                                    np.nan,
                                ),

                            "network_beta_bps":
                                fit.get(
                                    "network_beta_bps",
                                    np.nan,
                                ),

                            "partial_r2":
                                fit.get(
                                    "partial_r2",
                                    np.nan,
                                ),

                            "factor_residual_sd":
                                fit.get(
                                    "factor_residual_sd",
                                    np.nan,
                                ),

                            "status":
                                fit[
                                    "status"
                                ],
                        }
                    )

    coef_df = (
        pd.DataFrame(
            coef_rows
        )
        .sort_values(
            [
                "target_name",
                "factor_order",
                "window",
                "analysis_date",
                "model_spec",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    qa_df = (
        pd.DataFrame(
            qa_rows
        )
        .sort_values(
            [
                "target_name",
                "factor_order",
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        coef_df,
        qa_df,
    )


# ============================================================
# 13. Fama-MacBeth Second Stage
# ============================================================

def summarize_fmb(
    monthly_coef,
):

    passed = (
        monthly_coef[
            monthly_coef[
                "status"
            ]
            ==
            "PASS"
        ]
    )

    rows = []

    group_cols = [

        "factor_order",

        "factor_name",

        "target_name",

        "model_spec",

        "window",
    ]

    for keys, group in (
        passed.groupby(
            group_cols,
            sort=False,
        )
    ):

        (
            factor_order,
            factor_name,
            target_name,
            model_spec,
            window,
        ) = keys

        group = (
            group.sort_values(
                "analysis_date"
            )
        )

        beta = (
            group[
                "network_beta"
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

        inference = (
            calendar_gap_hac(
                beta,
                dates,
            )
        )

        rows.append(
            {

                "factor_order":
                    factor_order,

                "factor_name":
                    factor_name,

                "factor_orientation":
                    "ORIGINAL",

                "target_name":
                    target_name,

                "model_spec":
                    model_spec,

                "window":
                    int(
                        window
                    ),

                "month_count":
                    int(
                        len(
                            group
                        )
                    ),

                "mean_network_beta":
                    float(
                        np.mean(
                            beta
                        )
                    ),

                "mean_network_beta_bps_per_1sd":
                    float(
                        np.mean(
                            beta
                        )
                        *
                        10000.0
                    ),

                "median_network_beta":
                    float(
                        np.median(
                            beta
                        )
                    ),

                "std_network_beta":
                    (
                        float(
                            np.std(
                                beta,
                                ddof=1,
                            )
                        )
                        if
                        len(
                            beta
                        )
                        >=
                        2
                        else
                        np.nan
                    ),

                "positive_beta_share":
                    float(
                        np.mean(
                            beta
                            >
                            0
                        )
                    ),

                "negative_beta_share":
                    float(
                        np.mean(
                            beta
                            <
                            0
                        )
                    ),

                "mean_cross_section_n":
                    float(
                        group[
                            "nobs"
                        ]
                        .mean()
                    ),

                "mean_partial_r2":
                    float(
                        group[
                            "partial_r2"
                        ]
                        .mean()
                    ),

                "naive_se":
                    inference[
                        "naive_se"
                    ],

                "naive_t_stat":
                    inference[
                        "naive_t"
                    ],

                "hac_lag":
                    inference[
                        "hac_lag"
                    ],

                "hac_se":
                    inference[
                        "hac_se"
                    ],

                "hac_t_stat":
                    inference[
                        "hac_t"
                    ],

                "factor_sign_flipped":
                    False,

                "window_selected":
                    False,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "target_name",
                "factor_order",
                "window",
                "model_spec",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 14. Model Comparison
# ============================================================

def build_model_comparison(
    summary,
):

    rows = []

    keys = [

        "factor_order",

        "factor_name",

        "target_name",

        "window",
    ]

    for group_keys, group in (
        summary.groupby(
            keys,
            sort=False,
        )
    ):

        (
            factor_order,
            factor_name,
            target_name,
            window,
        ) = group_keys

        g = group.set_index(
            "model_spec"
        )

        required = set(
            MODEL_SPECS
        )

        if (
            not required
            .issubset(
                set(
                    g.index
                )
            )
        ):

            continue

        def value(
            model,
            column,
        ):

            return float(
                g.loc[
                    model,
                    column,
                ]
            )

        m1_beta = value(
            "M1_NETWORK_ONLY",
            "mean_network_beta",
        )

        m2_beta = value(
            "M2_INDUSTRY_SIZE",
            "mean_network_beta",
        )

        m3_beta = value(
            "M3_FULL_CONTROLS",
            "mean_network_beta",
        )

        m3_t = value(
            "M3_FULL_CONTROLS",
            "hac_t_stat",
        )

        m3_r2 = value(
            "M3_FULL_CONTROLS",
            "mean_partial_r2",
        )

        rows.append(
            {

                "factor_order":
                    factor_order,

                "factor_name":
                    factor_name,

                "target_name":
                    target_name,

                "window":
                    int(
                        window
                    ),

                "m1_network_only_beta":
                    m1_beta,

                "m1_beta_bps_per_1sd":
                    (
                        m1_beta
                        *
                        10000
                    ),

                "m2_industry_size_beta":
                    m2_beta,

                "m2_beta_bps_per_1sd":
                    (
                        m2_beta
                        *
                        10000
                    ),

                "m3_full_control_beta":
                    m3_beta,

                "m3_beta_bps_per_1sd":
                    (
                        m3_beta
                        *
                        10000
                    ),

                "m3_hac_t_stat":
                    m3_t,

                "m3_mean_partial_r2":
                    m3_r2,

                "m3_vs_m2_same_sign":
                    bool(

                        np.sign(
                            m3_beta
                        )

                        ==

                        np.sign(
                            m2_beta
                        )

                        and

                        np.sign(
                            m2_beta
                        )
                        !=
                        0
                    ),

                "m3_vs_m2_abs_retention":
                    (
                        abs(
                            m3_beta
                        )
                        /
                        abs(
                            m2_beta
                        )
                        if
                        abs(
                            m2_beta
                        )
                        >=
                        MIN_BETA_BASELINE
                        else
                        np.nan
                    ),

                "m3_minus_m2_beta":
                    (
                        m3_beta
                        -
                        m2_beta
                    ),

                "m3_minus_m2_bps":
                    (
                        m3_beta
                        -
                        m2_beta
                    )
                    *
                    10000,

                "m2_vs_m1_same_sign":
                    bool(

                        np.sign(
                            m2_beta
                        )

                        ==

                        np.sign(
                            m1_beta
                        )

                        and

                        np.sign(
                            m1_beta
                        )
                        !=
                        0
                    ),

                "factor_sign_flipped":
                    False,

                "factor_selected_from_fmb":
                    False,

                "window_selected_from_fmb":
                    False,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "target_name",
                "factor_order",
                "window",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 15. Cross-Window Consistency
# ============================================================

def same_nonzero_sign(
    values,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    if (
        len(
            x
        )
        !=
        3
    ):

        return False

    if (
        not np.isfinite(
            x
        )
        .all()
    ):

        return False

    signs = np.sign(
        x
    )

    return bool(

        np.all(
            signs
            ==
            signs[
                0
            ]
        )

        and

        np.all(
            signs
            !=
            0
        )
    )


def build_cross_window(
    summary,
):

    # --------------------------------------------------------
    # Final specification only
    # --------------------------------------------------------

    final = summary[
        summary[
            "model_spec"
        ]
        ==
        "M3_FULL_CONTROLS"
    ].copy()

    rows = []

    for keys, group in (
        final.groupby(
            [
                "factor_order",
                "factor_name",
                "target_name",
            ],
            sort=False,
        )
    ):

        (
            factor_order,
            factor_name,
            target_name,
        ) = keys

        group = (
            group.sort_values(
                "window"
            )
        )

        beta_map = dict(
            zip(
                group[
                    "window"
                ],
                group[
                    "mean_network_beta"
                ],
            )
        )

        t_map = dict(
            zip(
                group[
                    "window"
                ],
                group[
                    "hac_t_stat"
                ],
            )
        )

        beta_values = [

            beta_map.get(
                60,
                np.nan,
            ),

            beta_map.get(
                120,
                np.nan,
            ),

            beta_map.get(
                252,
                np.nan,
            ),
        ]

        rows.append(
            {

                "factor_order":
                    factor_order,

                "factor_name":
                    factor_name,

                "target_name":
                    target_name,

                "model_spec":
                    "M3_FULL_CONTROLS",

                "beta_w60":
                    beta_values[
                        0
                    ],

                "beta_w120":
                    beta_values[
                        1
                    ],

                "beta_w252":
                    beta_values[
                        2
                    ],

                "beta_bps_w60":
                    beta_values[
                        0
                    ]
                    *
                    10000
                    if
                    np.isfinite(
                        beta_values[
                            0
                        ]
                    )
                    else
                    np.nan,

                "beta_bps_w120":
                    beta_values[
                        1
                    ]
                    *
                    10000
                    if
                    np.isfinite(
                        beta_values[
                            1
                        ]
                    )
                    else
                    np.nan,

                "beta_bps_w252":
                    beta_values[
                        2
                    ]
                    *
                    10000
                    if
                    np.isfinite(
                        beta_values[
                            2
                        ]
                    )
                    else
                    np.nan,

                "same_beta_sign_all_windows":
                    same_nonzero_sign(
                        beta_values
                    ),

                "hac_t_w60":
                    t_map.get(
                        60,
                        np.nan,
                    ),

                "hac_t_w120":
                    t_map.get(
                        120,
                        np.nan,
                    ),

                "hac_t_w252":
                    t_map.get(
                        252,
                        np.nan,
                    ),

                "hac_abs_ge_1p96_count":
                    int(
                        np.sum(
                            [
                                (
                                    abs(
                                        x
                                    )
                                    >=
                                    1.96
                                )
                                if
                                np.isfinite(
                                    x
                                )
                                else
                                False
                                for x in [
                                    t_map.get(
                                        60,
                                        np.nan,
                                    ),
                                    t_map.get(
                                        120,
                                        np.nan,
                                    ),
                                    t_map.get(
                                        252,
                                        np.nan,
                                    ),
                                ]
                            ]
                        )
                    ),

                "window_selected":
                    False,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "target_name",
                "factor_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 16. Formal QA
# ============================================================

def formal_qa(
    monthly,
    summary,
    cross_section_qa,
    core,
):

    qa = {}

    qa[
        "factor_count"
    ] = int(
        len(
            core
        )
    )

    qa[
        "factor_count_ok"
    ] = bool(
        len(
            core
        )
        ==
        EXPECTED_FACTOR_COUNT
    )

    qa[
        "observed_windows"
    ] = sorted(
        monthly[
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
        "observed_models"
    ] = sorted(
        monthly[
            "model_spec"
        ]
        .unique()
        .tolist()
    )

    qa[
        "model_set_ok"
    ] = bool(
        qa[
            "observed_models"
        ]
        ==
        sorted(
            MODEL_SPECS
        )
    )

    qa[
        "observed_targets"
    ] = sorted(
        monthly[
            "target_name"
        ]
        .unique()
        .tolist()
    )

    qa[
        "target_set_ok"
    ] = bool(
        qa[
            "observed_targets"
        ]
        ==
        sorted(
            TARGETS
        )
    )

    qa[
        "monthly_key_unique"
    ] = bool(

        not monthly[
            [
                "analysis_date",
                "window",
                "factor_name",
                "target_name",
                "model_spec",
            ]
        ]
        .duplicated()
        .any()
    )

    qa[
        "status_counts"
    ] = (
        monthly[
            "status"
        ]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # Same nobs across all three model specifications
    # --------------------------------------------------------

    nobs_check = (

        monthly.groupby(
            [
                "analysis_date",
                "window",
                "factor_name",
                "target_name",
            ]
        )[
            "nobs"
        ]
        .nunique(
            dropna=False
        )
    )

    qa[
        "same_sample_across_models"
    ] = bool(
        (
            nobs_check
            <=
            1
        )
        .all()
    )

    # --------------------------------------------------------
    # Control rank QA on PASS rows
    # --------------------------------------------------------

    m2 = monthly[
        (
            monthly[
                "model_spec"
            ]
            ==
            "M2_INDUSTRY_SIZE"
        )
        &
        (
            monthly[
                "status"
            ]
            ==
            "PASS"
        )
    ]

    m3 = monthly[
        (
            monthly[
                "model_spec"
            ]
            ==
            "M3_FULL_CONTROLS"
        )
        &
        (
            monthly[
                "status"
            ]
            ==
            "PASS"
        )
    ]

    qa[
        "m2_control_rank_ok"
    ] = bool(
        (
            m2[
                "control_rank"
            ]
            ==
            1
        )
        .all()
    )

    qa[
        "m3_control_rank_ok"
    ] = bool(
        (
            m3[
                "control_rank"
            ]
            ==
            5
        )
        .all()
    )

    qa[
        "minimum_regression_n"
    ] = int(
        cross_section_qa[
            "regression_n"
        ]
        .min()
    )

    qa[
        "median_regression_n"
    ] = float(
        cross_section_qa[
            "regression_n"
        ]
        .median()
    )

    qa[
        "predictor_preprocessing_uses_future_outcome"
    ] = False

    qa[
        "factor_sign_flip"
    ] = False

    qa[
        "factor_selection"
    ] = False

    qa[
        "window_selection"
    ] = False

    required = [

        "factor_count_ok",

        "window_set_ok",

        "model_set_ok",

        "target_set_ok",

        "monthly_key_unique",

        "same_sample_across_models",

        "m2_control_rank_ok",

        "m3_control_rank_ok",
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
# 17. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 7 - Step 3"
    )

    print(
        "Fama-MacBeth Cross-Sectional Regression"
    )

    print("=" * 80)

    # ========================================================
    # Upstream lineage
    # ========================================================

    (
        step6_metadata,
        step1_metadata,
        step2_metadata,
    ) = validate_upstream()

    core = load_core_factors()

    # ========================================================
    # Freeze design before regression
    # ========================================================

    design_payload = {

        "research_day":
            7,

        "step":
            "Step3_Fama_MacBeth_Regression",

        "parent_day6_step6_design_hash":
            step6_metadata.get(
                "step6_design_hash"
            ),

        "parent_day7_step1_design_hash":
            step1_metadata.get(
                "day7_step1_design_hash"
            ),

        "parent_day7_step2_design_hash":
            step2_metadata.get(
                "day7_step2_design_hash"
            ),

        "factor_set":
            core[
                "factor_name"
            ]
            .astype(str)
            .tolist(),

        "windows":
            EXPECTED_WINDOWS,

        "targets":
            {

                "primary":
                    PRIMARY_TARGET,

                "robustness":
                    ROBUSTNESS_TARGET,
            },

        "models":
            {

                "M1_NETWORK_ONLY":
                    (
                        "future return ~ standardized "
                        "network factor"
                    ),

                "M2_INDUSTRY_SIZE":
                    (
                        "future return ~ network factor "
                        "+ industry fixed effects "
                        "+ log market value"
                    ),

                "M3_FULL_CONTROLS":
                    (
                        "future return ~ network factor "
                        "+ industry fixed effects "
                        "+ size + momentum + reversal "
                        "+ past volatility + turnover"
                    ),
            },

        "common_sample_across_models":
            True,

        "network_factor_winsorization":
            False,

        "network_factor_transform":
            (
                "Cross-sectional z-standardization only."
            ),

        "control_transform":
            (
                "1%/99% cross-sectional winsorization "
                "followed by z-standardization."
            ),

        "industry_control":
            "industry_id1",

        "fmb_second_stage":
            (
                "Equal-weight time-series average "
                "of monthly cross-sectional network "
                "coefficients."
            ),

        "inference":
            (
                "Calendar-gap-aware Newey-West/HAC "
                "standard error for mean monthly beta."
            ),

        "factor_sign_flip":
            False,

        "factor_selection_from_results":
            False,

        "window_selection_from_results":
            False,
    }

    design = {

        **design_payload,

        "day7_step3_design_hash":
            canonical_hash(
                design_payload
            ),
    }

    print()
    print(
        "Day-7 Step-3 design hash:"
    )

    print(
        design[
            "day7_step3_design_hash"
        ]
    )

    # ========================================================
    # Panel
    # ========================================================

    print()
    print(
        "[1] Load factor, return and characteristic panel"
    )

    panel = load_analysis_panel(
        core
    )

    print(
        f"Panel rows: "
        f"{len(panel):,}"
    )

    # ========================================================
    # First-stage cross-sectional regressions
    # ========================================================

    print()
    print(
        "[2] Run monthly cross-sectional regressions"
    )

    (
        monthly_coef,
        cross_section_qa,
    ) = run_monthly_fmb(
        panel,
        core,
    )

    save_csv_atomic(
        monthly_coef,
        MONTHLY_COEF_PATH,
    )

    save_csv_atomic(
        cross_section_qa,
        CROSS_SECTION_QA_PATH,
    )

    # ========================================================
    # Second-stage FMB
    # ========================================================

    print()
    print(
        "[3] Compute Fama-MacBeth coefficient averages"
    )

    fmb_summary = summarize_fmb(
        monthly_coef
    )

    save_csv_atomic(
        fmb_summary,
        FMB_SUMMARY_PATH,
    )

    # ========================================================
    # Model comparison
    # ========================================================

    print()
    print(
        "[4] Compare M1 / M2 / M3"
    )

    model_comparison = (
        build_model_comparison(
            fmb_summary
        )
    )

    save_csv_atomic(
        model_comparison,
        MODEL_COMPARISON_PATH,
    )

    # ========================================================
    # Cross-window
    # ========================================================

    print()
    print(
        "[5] Build M3 cross-window consistency"
    )

    cross_window = (
        build_cross_window(
            fmb_summary
        )
    )

    save_csv_atomic(
        cross_window,
        CROSS_WINDOW_PATH,
    )

    # ========================================================
    # Formal QA
    # ========================================================

    qa = formal_qa(

        monthly=
            monthly_coef,

        summary=
            fmb_summary,

        cross_section_qa=
            cross_section_qa,

        core=
            core,
    )

    if (
        not qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            "Day-7 Step-3 formal QA failed.\n"
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

        "formal_qa":
            qa,

        "interpretation": {

            "network_beta":
                (
                    "Future-return change associated "
                    "with a one-standard-deviation "
                    "increase in the contemporaneous "
                    "network factor."
                ),

            "network_beta_bps":
                (
                    "Same coefficient expressed in "
                    "basis points per forward holding interval."
                ),

            "m3_coefficient":
                (
                    "Conditional network-factor coefficient "
                    "after industry, size, momentum, reversal, "
                    "past volatility and turnover controls."
                ),

            "partial_r2":
                (
                    "Incremental cross-sectional variation "
                    "in future return explained by the network "
                    "factor conditional on the model controls."
                ),
        },

        "outputs": {

            "monthly_fmb_coefficients":
                str(
                    MONTHLY_COEF_PATH
                ),

            "fama_macbeth_summary":
                str(
                    FMB_SUMMARY_PATH
                ),

            "fmb_model_comparison":
                str(
                    MODEL_COMPARISON_PATH
                ),

            "fmb_cross_window_consistency":
                str(
                    CROSS_WINDOW_PATH
                ),

            "fmb_cross_section_qa":
                str(
                    CROSS_SECTION_QA_PATH
                ),
        },
    }

    save_json_atomic(
        metadata,
        METADATA_PATH,
    )

    # ========================================================
    # Console display
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Day-7 Step-3 Formal QA"
    )

    print("=" * 80)

    print(
        f"Status counts: "
        f"{qa['status_counts']}"
    )

    print(
        f"Same sample across models: "
        f"{qa['same_sample_across_models']}"
    )

    print(
        f"M2 rank OK: "
        f"{qa['m2_control_rank_ok']}"
    )

    print(
        f"M3 rank OK: "
        f"{qa['m3_control_rank_ok']}"
    )

    print(
        f"Median regression N: "
        f"{qa['median_regression_n']:.1f}"
    )

    print(
        f"Formal QA pass: "
        f"{qa['all_formal_qa_pass']}"
    )

    print()
    print("=" * 80)

    print(
        "Primary Target: M3 Full-Control FMB Results"
    )

    print("=" * 80)

    display = fmb_summary[
        (
            fmb_summary[
                "target_name"
            ]
            ==
            PRIMARY_TARGET
        )
        &
        (
            fmb_summary[
                "model_spec"
            ]
            ==
            "M3_FULL_CONTROLS"
        )
    ]

    display_columns = [

        "factor_name",

        "window",

        "month_count",

        "mean_network_beta_bps_per_1sd",

        "positive_beta_share",

        "negative_beta_share",

        "mean_partial_r2",

        "hac_t_stat",
    ]

    print(
        display[
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
        "Day 7 Step 3 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()
