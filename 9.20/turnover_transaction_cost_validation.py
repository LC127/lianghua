from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd


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
# Research Day 5 -> M1_day4
# ------------------------------------------------------------

FACTOR_PANEL_PATH = (
    OUTPUT_ROOT
    / "M1_day5"
    / "05_step5_community_features"
    / "community_bridge_feature_panel.parquet"
)


# ------------------------------------------------------------
# Day 6
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


# ------------------------------------------------------------
# Day 7 Step 2
# ------------------------------------------------------------

DAY7_ROOT = (
    OUTPUT_ROOT
    / "M1_day7"
)

STEP2_DIR = (
    DAY7_ROOT
    / "02_stage2_traditional_characteristic_controls"
)

CHARACTERISTIC_PANEL_PATH = (
    STEP2_DIR
    / "traditional_characteristic_panel.parquet"
)

STEP2_MONTHLY_SPREAD_PATH = (
    STEP2_DIR
    / "monthly_controlled_alpha_portfolio_spreads.csv"
)

STEP2_METADATA_PATH = (
    STEP2_DIR
    / "day7_step2_metadata.json"
)


# ------------------------------------------------------------
# Day 7 Step 3
# ------------------------------------------------------------

STEP3_METADATA_PATH = (
    DAY7_ROOT
    / "03_stage3_fama_macbeth_regression"
    / "day7_step3_metadata.json"
)


# ------------------------------------------------------------
# Day 7 Step 4
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY7_ROOT
    / "04_stage4_turnover_transaction_cost"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MONTHLY_TURNOVER_PATH = (
    OUTPUT_DIR
    / "monthly_portfolio_turnover.csv"
)

MONTHLY_COST_PATH = (
    OUTPUT_DIR
    / "monthly_cost_adjusted_spreads.csv"
)

TURNOVER_SUMMARY_PATH = (
    OUTPUT_DIR
    / "turnover_summary.csv"
)

COST_SUMMARY_PATH = (
    OUTPUT_DIR
    / "transaction_cost_summary.csv"
)

ECONOMIC_SUMMARY_PATH = (
    OUTPUT_DIR
    / "turnover_economic_summary.csv"
)

QA_PATH = (
    OUTPUT_DIR
    / "turnover_transaction_cost_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day7_step4_metadata.json"
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


CONTROL_COLUMNS = [

    "log_market_value",

    "momentum_120_20",

    "reversal_20",

    "volatility_60",

    "log_turnover_20",
]


SIZE_ONLY_COLUMNS = [
    "log_market_value"
]


PRIMARY_RETURN = (
    "future_excess_return"
)

DRIFT_RETURN = (
    "future_total_return"
)


N_PORTFOLIOS = 5

MIN_SORT_OBS = 100

MIN_PORTFOLIO_RETURN_OBS = 20


CONTROL_WINSOR_LOW = 0.01

CONTROL_WINSOR_HIGH = 0.99


# ------------------------------------------------------------
# One-way trading-cost scenarios
#
# 1 bp = 0.0001
# ------------------------------------------------------------

COST_SCENARIOS_BPS = [

    5,

    10,

    20,

    30,
]


# ------------------------------------------------------------
# Exact drift turnover:
# require every previous target weight to have a valid
# realized total return.
#
# We do NOT silently fill missing drift returns with zero.
# ------------------------------------------------------------

DRIFT_COVERAGE_TOL = 1e-10


USE_HAC = True


KEY_COLUMNS = [

    "analysis_date",

    "window",

    "master_index",

    "security_id",
]


# ============================================================
# 2. Utilities
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
    df,
    path,
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


def canonical_hash(obj):

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
# 3. Upstream Validation
# ============================================================

def validate_upstream():

    step2 = load_json(
        STEP2_METADATA_PATH
    )

    step3 = load_json(
        STEP3_METADATA_PATH
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

    if (
        step3.get(
            "formal_qa",
            {}
        ).get(
            "all_formal_qa_pass"
        )
        is not True
    ):

        raise RuntimeError(
            "Day-7 Step-3 formal QA failed."
        )

    required = [

        FACTOR_PANEL_PATH,

        LABEL_PANEL_PATH,

        CHARACTERISTIC_PANEL_PATH,

        STEP2_MONTHLY_SPREAD_PATH,

        FROZEN_CORE_PATH,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required input:\n{path}"
            )

    return (
        step2,
        step3,
    )


# ============================================================
# 4. Frozen Factors
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

    if (
        len(core)
        !=
        EXPECTED_FACTOR_COUNT
    ):

        raise RuntimeError(
            "Unexpected number of frozen factors."
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
# 5. Load Analysis Panel
# ============================================================

def load_panel(
    core,
):

    factor_names = (
        core[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    factors = pd.read_parquet(

        FACTOR_PANEL_PATH,

        columns=(
            KEY_COLUMNS
            +
            factor_names
        ),
    )

    labels = pd.read_parquet(

        LABEL_PANEL_PATH,

        columns=(
            KEY_COLUMNS

            +

            [
                "future_return_label_valid",

                PRIMARY_RETURN,

                DRIFT_RETURN,
            ]
        ),
    )

    characteristics = pd.read_parquet(

        CHARACTERISTIC_PANEL_PATH,

        columns=(

            [
                "analysis_date",

                "security_id",

                "industry_id1",
            ]

            +

            CONTROL_COLUMNS
        ),
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
# 6. Control Preparation
#
# EXACTLY aligned with Day-7 Step-2.
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

        if (
            valid.sum()
            <
            2
        ):

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
            np.isfinite(
                sd
            )
            and
            sd
            >
            0
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
# 7. FWL Factor Residualization
#
# EXACTLY aligned with Day-7 Step-2.
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
        MIN_SORT_OBS
    ):

        return result

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

    factor_within = (

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

    controls_within = pd.DataFrame(
        index=C.index
    )

    for column in C.columns:

        tmp = pd.DataFrame(
            {
                "value":
                    C[
                        column
                    ],

                "industry":
                    work[
                        "industry"
                    ],
            }
        )

        controls_within[
            column
        ] = (

            tmp[
                "value"
            ]

            -

            tmp.groupby(
                "industry"
            )[
                "value"
            ]
            .transform(
                "mean"
            )
        )

    X = controls_within.to_numpy(
        dtype=float
    )

    y = factor_within.to_numpy(
        dtype=float
    )

    beta, _, _, _ = (
        np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )
    )

    residual = (
        y
        -
        X @ beta
    )

    result.loc[
        valid
    ] = residual

    return result


# ============================================================
# 8. Conservative Quintile Assignment
#
# No artificial tie-breaking.
# ============================================================

def assign_quintiles(
    score,
):

    score = pd.to_numeric(
        score,
        errors="coerce",
    )

    valid = (

        score.notna()

        &

        np.isfinite(
            score
        )
    )

    x = score.loc[
        valid
    ]

    result = pd.Series(
        pd.NA,
        index=score.index,
        dtype="Int64",
    )

    if (
        len(x)
        <
        MIN_SORT_OBS
    ):

        return (
            result,
            "INSUFFICIENT_SORT_OBS",
        )

    if (
        x.nunique()
        <
        N_PORTFOLIOS
    ):

        return (
            result,
            "INSUFFICIENT_UNIQUE_VALUES",
        )

    try:

        q = pd.qcut(

            x,

            q=N_PORTFOLIOS,

            labels=False,

            duplicates="drop",
        )

    except ValueError:

        return (
            result,
            "QCUT_FAILURE",
        )

    if (
        q.nunique()
        !=
        N_PORTFOLIOS
    ):

        return (
            result,
            "FEWER_THAN_5_PORTFOLIOS",
        )

    result.loc[
        x.index
    ] = (
        q.astype(int)
        +
        1
    )

    return (
        result,
        "PASS",
    )


# ============================================================
# 9. Equal-Weight Portfolio
# ============================================================

def equal_weight_dict(
    security_ids,
):

    ids = (
        pd.Series(
            security_ids,
            dtype="string",
        )
        .dropna()
        .astype(str)
        .tolist()
    )

    n = len(
        ids
    )

    if (
        n
        ==
        0
    ):

        return {}

    weight = (
        1.0
        /
        n
    )

    return {
        sid:
            weight
        for sid in ids
    }


# ============================================================
# 10. Portfolio Return
# ============================================================

def portfolio_future_return(
    group,
    portfolio,
    q,
):

    y = pd.to_numeric(
        group[
            PRIMARY_RETURN
        ],
        errors="coerce",
    )

    valid_label = (
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

    mask = (

        valid_label

        &

        y.notna()

        &

        np.isfinite(
            y
        )

        &

        (
            portfolio
            ==
            q
        )
    )

    values = (
        y.loc[
            mask
        ]
    )

    n = len(
        values
    )

    if (
        n
        <
        MIN_PORTFOLIO_RETURN_OBS
    ):

        return (
            np.nan,
            n,
        )

    return (
        float(
            values.mean()
        ),
        n,
    )


# ============================================================
# 11. Target-to-Target Turnover
# ============================================================

def target_turnover(
    previous_weights,
    current_weights,
):

    union = (
        set(
            previous_weights
        )
        |
        set(
            current_weights
        )
    )

    value = (
        0.5
        *
        sum(

            abs(

                current_weights.get(
                    sid,
                    0.0,
                )

                -

                previous_weights.get(
                    sid,
                    0.0,
                )
            )

            for sid in union
        )
    )

    return float(
        value
    )


# ============================================================
# 12. Drift-Adjusted Turnover
# ============================================================

def drift_adjusted_turnover(

    previous_weights,

    current_weights,

    previous_total_returns,
):

    if (
        not previous_weights
        or
        not current_weights
    ):

        return {

            "status":
                "EMPTY_PORTFOLIO",

            "turnover":
                np.nan,

            "return_weight_coverage":
                np.nan,
        }

    valid_weight = 0.0

    grown = {}

    for sid, weight in (
        previous_weights.items()
    ):

        r = previous_total_returns.get(
            sid,
            np.nan,
        )

        if (
            pd.notna(
                r
            )
            and
            np.isfinite(
                r
            )
        ):

            if (
                r
                <
                -1.0
                -
                1e-12
            ):

                raise RuntimeError(
                    f"Simple return < -1 for {sid}: {r}"
                )

            valid_weight += weight

            grown[
                sid
            ] = (

                weight

                *

                (
                    1.0
                    +
                    r
                )
            )

    coverage = float(
        valid_weight
    )

    # --------------------------------------------------------
    # Do not silently fill missing realized returns.
    # --------------------------------------------------------

    if (
        abs(
            coverage
            -
            1.0
        )
        >
        DRIFT_COVERAGE_TOL
    ):

        return {

            "status":
                "INCOMPLETE_DRIFT_RETURN",

            "turnover":
                np.nan,

            "return_weight_coverage":
                coverage,
        }

    total_value = float(
        sum(
            grown.values()
        )
    )

    if (
        not np.isfinite(
            total_value
        )
        or
        total_value
        <=
        0
    ):

        return {

            "status":
                "INVALID_DRIFTED_PORTFOLIO_VALUE",

            "turnover":
                np.nan,

            "return_weight_coverage":
                coverage,
        }

    pretrade_weights = {

        sid:
            value
            /
            total_value

        for sid, value in (
            grown.items()
        )
    }

    union = (

        set(
            pretrade_weights
        )

        |

        set(
            current_weights
        )
    )

    turnover = (

        0.5

        *

        sum(

            abs(

                current_weights.get(
                    sid,
                    0.0,
                )

                -

                pretrade_weights.get(
                    sid,
                    0.0,
                )
            )

            for sid in union
        )
    )

    return {

        "status":
            "PASS",

        "turnover":
            float(
                turnover
            ),

        "return_weight_coverage":
            coverage,
    }


# ============================================================
# 13. HAC
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

    # to_datetime preserves Series inputs, which lack .year/.month attributes.
    # Use one index type for positional filtering and calendar-month access.
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

            "lag":
                0,

            "se":
                np.nan,

            "t":
                np.nan,
        }

    mean_x = float(
        x.mean()
    )

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
            "Duplicate calendar month."
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

    lag = (
        automatic_hac_lag(
            n
        )
        if USE_HAC
        else
        0
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

            prev_month = (
                month
                -
                ell
            )

            if (
                prev_month
                in
                residual_map
            ):

                cross_sum += (

                    value

                    *

                    residual_map[
                        prev_month
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

        se = float(
            np.sqrt(
                variance_mean
            )
        )

        t_stat = (
            mean_x
            /
            se
        )

    else:

        se = np.nan

        t_stat = np.nan

    return {

        "lag":
            int(
                lag
            ),

        "se":
            se,

        "t":
            t_stat,
    }


# ============================================================
# 14. Main Monthly Turnover Construction
# ============================================================

def run_turnover_analysis(
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

    turnover_rows = []

    cost_rows = []

    # --------------------------------------------------------
    # State keyed by:
    #
    # (window, factor_name, method)
    #
    # Stores the immediately previous successful portfolio.
    # --------------------------------------------------------

    state = {}

    # --------------------------------------------------------
    # Expected previous date for each window
    # --------------------------------------------------------

    previous_date_map = {}

    for window in EXPECTED_WINDOWS:

        dates = sorted(
            panel.loc[
                panel[
                    "window"
                ]
                ==
                window,
                "analysis_date",
            ]
            .drop_duplicates()
            .tolist()
        )

        for i, date in enumerate(
            dates
        ):

            previous_date_map[
                (
                    window,
                    date,
                )
            ] = (
                dates[
                    i - 1
                ]
                if
                i
                >
                0
                else
                None
            )

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

            common = (

                factor_valid

                &

                complete_controls
            )

            raw_score = (
                factor.where(
                    common
                )
            )

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

            ind_size_score = (
                residualize_factor(

                    factor.where(
                        common
                    ),

                    industry,

                    size_controls,
                )
            )

            full_controls = (
                controls[
                    CONTROL_COLUMNS
                ]
                .copy()
            )

            full_controls.loc[
                ~common,
                :
            ] = np.nan

            full_score = (
                residualize_factor(

                    factor.where(
                        common
                    ),

                    industry,

                    full_controls,
                )
            )

            scores = {

                "RAW_COMMON":
                    raw_score,

                "INDUSTRY_SIZE_COMMON":
                    ind_size_score,

                "FULL_CHARACTERISTIC_NEUTRAL":
                    full_score,
            }

            for method, score in (
                scores.items()
            ):

                key = (

                    int(
                        window
                    ),

                    factor_name,

                    method,
                )

                (
                    portfolio,
                    sort_status,
                ) = assign_quintiles(
                    score
                )

                expected_previous_date = (
                    previous_date_map[
                        (
                            int(
                                window
                            ),
                            analysis_date,
                        )
                    ]
                )

                # =================================================
                # Sort failure
                # =================================================

                if (
                    sort_status
                    !=
                    "PASS"
                ):

                    turnover_rows.append(
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

                            "method":
                                method,

                            "sort_status":
                                sort_status,

                            "turnover_status":
                                "SORT_FAILURE",

                            "gross_spread_status":
                                "SORT_FAILURE",

                            "gross_q5_minus_q1":
                                np.nan,
                        }
                    )

                    # Break continuity.
                    state[
                        key
                    ] = None

                    continue

                # =================================================
                # Current holdings
                # =================================================

                q1_ids = (
                    group.loc[
                        portfolio
                        ==
                        1,
                        "security_id",
                    ]
                )

                q5_ids = (
                    group.loc[
                        portfolio
                        ==
                        5,
                        "security_id",
                    ]
                )

                q1_weights = (
                    equal_weight_dict(
                        q1_ids
                    )
                )

                q5_weights = (
                    equal_weight_dict(
                        q5_ids
                    )
                )

                # =================================================
                # Current-period gross return
                # =================================================

                (
                    q1_return,
                    q1_return_n,
                ) = portfolio_future_return(

                    group,

                    portfolio,

                    1,
                )

                (
                    q5_return,
                    q5_return_n,
                ) = portfolio_future_return(

                    group,

                    portfolio,

                    5,
                )

                if (
                    np.isfinite(
                        q1_return
                    )
                    and
                    np.isfinite(
                        q5_return
                    )
                ):

                    gross_spread = (

                        q5_return

                        -

                        q1_return
                    )

                    gross_status = (
                        "PASS"
                    )

                else:

                    gross_spread = np.nan

                    gross_status = (
                        "INSUFFICIENT_PORTFOLIO_RETURN_OBS"
                    )

                # =================================================
                # Turnover against previous formation
                # =================================================

                previous = state.get(
                    key
                )

                if (
                    previous is None
                ):

                    turnover_status = (
                        "NO_PREVIOUS_FORMATION"
                    )

                    q1_target_to = np.nan
                    q5_target_to = np.nan

                    q1_drift_to = np.nan
                    q5_drift_to = np.nan

                    q1_drift_coverage = np.nan
                    q5_drift_coverage = np.nan

                elif (
                    previous[
                        "analysis_date"
                    ]
                    !=
                    expected_previous_date
                ):

                    turnover_status = (
                        "NONCONSECUTIVE_PREVIOUS_FORMATION"
                    )

                    q1_target_to = np.nan
                    q5_target_to = np.nan

                    q1_drift_to = np.nan
                    q5_drift_to = np.nan

                    q1_drift_coverage = np.nan
                    q5_drift_coverage = np.nan

                else:

                    q1_target_to = (
                        target_turnover(

                            previous[
                                "q1_weights"
                            ],

                            q1_weights,
                        )
                    )

                    q5_target_to = (
                        target_turnover(

                            previous[
                                "q5_weights"
                            ],

                            q5_weights,
                        )
                    )

                    q1_drift = (
                        drift_adjusted_turnover(

                            previous[
                                "q1_weights"
                            ],

                            q1_weights,

                            previous[
                                "future_total_return_map"
                            ],
                        )
                    )

                    q5_drift = (
                        drift_adjusted_turnover(

                            previous[
                                "q5_weights"
                            ],

                            q5_weights,

                            previous[
                                "future_total_return_map"
                            ],
                        )
                    )

                    q1_drift_to = (
                        q1_drift[
                            "turnover"
                        ]
                    )

                    q5_drift_to = (
                        q5_drift[
                            "turnover"
                        ]
                    )

                    q1_drift_coverage = (
                        q1_drift[
                            "return_weight_coverage"
                        ]
                    )

                    q5_drift_coverage = (
                        q5_drift[
                            "return_weight_coverage"
                        ]
                    )

                    if (
                        q1_drift[
                            "status"
                        ]
                        ==
                        "PASS"

                        and

                        q5_drift[
                            "status"
                        ]
                        ==
                        "PASS"
                    ):

                        turnover_status = (
                            "PASS"
                        )

                    else:

                        turnover_status = (

                            "Q1_"
                            +
                            q1_drift[
                                "status"
                            ]

                            +
                            "__Q5_"

                            +
                            q5_drift[
                                "status"
                            ]
                        )

                # =================================================
                # Save monthly turnover
                # =================================================

                turnover_rows.append(
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

                        "method":
                            method,

                        "sort_status":
                            sort_status,

                        "q1_sort_n":
                            len(
                                q1_weights
                            ),

                        "q5_sort_n":
                            len(
                                q5_weights
                            ),

                        "q1_return_n":
                            q1_return_n,

                        "q5_return_n":
                            q5_return_n,

                        "q1_future_excess_return":
                            q1_return,

                        "q5_future_excess_return":
                            q5_return,

                        "gross_q5_minus_q1":
                            gross_spread,

                        "gross_q5_minus_q1_bps":
                            (
                                gross_spread
                                *
                                10000
                                if
                                np.isfinite(
                                    gross_spread
                                )
                                else
                                np.nan
                            ),

                        "gross_spread_status":
                            gross_status,

                        "q1_target_turnover":
                            q1_target_to,

                        "q5_target_turnover":
                            q5_target_to,

                        "q1_drift_turnover":
                            q1_drift_to,

                        "q5_drift_turnover":
                            q5_drift_to,

                        "q1_drift_return_weight_coverage":
                            q1_drift_coverage,

                        "q5_drift_return_weight_coverage":
                            q5_drift_coverage,

                        "long_short_half_l1_turnover_sum":
                            (
                                q1_drift_to
                                +
                                q5_drift_to
                                if
                                np.isfinite(
                                    q1_drift_to
                                )
                                and
                                np.isfinite(
                                    q5_drift_to
                                )
                                else
                                np.nan
                            ),

                        "roundtrip_traded_notional_ratio":
                            (
                                2.0
                                *
                                (
                                    q1_drift_to
                                    +
                                    q5_drift_to
                                )
                                if
                                np.isfinite(
                                    q1_drift_to
                                )
                                and
                                np.isfinite(
                                    q5_drift_to
                                )
                                else
                                np.nan
                            ),

                        "turnover_status":
                            turnover_status,
                    }
                )

                # =================================================
                # Transaction-cost scenarios
                # =================================================

                if (
                    turnover_status
                    ==
                    "PASS"

                    and

                    gross_status
                    ==
                    "PASS"
                ):

                    total_half_turnover = (

                        q1_drift_to
                        +
                        q5_drift_to
                    )

                    for cost_bps in (
                        COST_SCENARIOS_BPS
                    ):

                        one_way_cost = (

                            cost_bps

                            /
                            10000.0
                        )

                        cost_drag = (

                            2.0

                            *

                            one_way_cost

                            *

                            total_half_turnover
                        )

                        net_spread = (

                            gross_spread

                            -

                            cost_drag
                        )

                        cost_rows.append(
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

                                "method":
                                    method,

                                "one_way_cost_bps":
                                    cost_bps,

                                "gross_q5_minus_q1":
                                    gross_spread,

                                "gross_q5_minus_q1_bps":
                                    (
                                        gross_spread
                                        *
                                        10000
                                    ),

                                "q1_drift_turnover":
                                    q1_drift_to,

                                "q5_drift_turnover":
                                    q5_drift_to,

                                "long_short_half_l1_turnover_sum":
                                    total_half_turnover,

                                "roundtrip_traded_notional_ratio":
                                    (
                                        2.0
                                        *
                                        total_half_turnover
                                    ),

                                "transaction_cost_drag":
                                    cost_drag,

                                "transaction_cost_drag_bps":
                                    (
                                        cost_drag
                                        *
                                        10000
                                    ),

                                "canonical_net_q5_minus_q1":
                                    net_spread,

                                "canonical_net_q5_minus_q1_bps":
                                    (
                                        net_spread
                                        *
                                        10000
                                    ),

                                "factor_sign_flipped":
                                    False,
                            }
                        )

                # =================================================
                # Current total returns for NEXT rebalance drift
                # =================================================

                current_total_return = (
                    pd.to_numeric(
                        group[
                            DRIFT_RETURN
                        ],
                        errors="coerce",
                    )
                )

                current_valid_return = (
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

                total_return_map = {}

                members = (

                    set(
                        q1_weights
                    )

                    |

                    set(
                        q5_weights
                    )
                )

                sid_series = (
                    group[
                        "security_id"
                    ]
                    .astype(str)
                )

                for sid in members:

                    mask = (
                        sid_series
                        ==
                        sid
                    )

                    if (
                        mask.sum()
                        !=
                        1
                    ):

                        total_return_map[
                            sid
                        ] = np.nan

                        continue

                    idx = group.index[
                        mask
                    ][0]

                    value = (
                        current_total_return.loc[
                            idx
                        ]
                    )

                    valid = bool(
                        current_valid_return.loc[
                            idx
                        ]
                    )

                    if (
                        valid
                        and
                        pd.notna(
                            value
                        )
                        and
                        np.isfinite(
                            value
                        )
                    ):

                        total_return_map[
                            sid
                        ] = float(
                            value
                        )

                    else:

                        total_return_map[
                            sid
                        ] = np.nan

                # =================================================
                # Update state
                # =================================================

                state[
                    key
                ] = {

                    "analysis_date":
                        analysis_date,

                    "q1_weights":
                        q1_weights,

                    "q5_weights":
                        q5_weights,

                    "future_total_return_map":
                        total_return_map,
                }

    return (

        pd.DataFrame(
            turnover_rows
        ),

        pd.DataFrame(
            cost_rows
        ),
    )


# ============================================================
# 15. Turnover Summary
# ============================================================

def build_turnover_summary(
    monthly,
):

    passed = monthly[
        monthly[
            "turnover_status"
        ]
        ==
        "PASS"
    ].copy()

    rows = []

    group_cols = [

        "factor_order",

        "factor_name",

        "method",

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
            method,
            window,
        ) = keys

        rows.append(
            {

                "factor_order":
                    factor_order,

                "factor_name":
                    factor_name,

                "method":
                    method,

                "window":
                    int(
                        window
                    ),

                "turnover_month_count":
                    int(
                        len(
                            group
                        )
                    ),

                "mean_q1_target_turnover":
                    float(
                        group[
                            "q1_target_turnover"
                        ]
                        .mean()
                    ),

                "mean_q5_target_turnover":
                    float(
                        group[
                            "q5_target_turnover"
                        ]
                        .mean()
                    ),

                "mean_q1_drift_turnover":
                    float(
                        group[
                            "q1_drift_turnover"
                        ]
                        .mean()
                    ),

                "median_q1_drift_turnover":
                    float(
                        group[
                            "q1_drift_turnover"
                        ]
                        .median()
                    ),

                "mean_q5_drift_turnover":
                    float(
                        group[
                            "q5_drift_turnover"
                        ]
                        .mean()
                    ),

                "median_q5_drift_turnover":
                    float(
                        group[
                            "q5_drift_turnover"
                        ]
                        .median()
                    ),

                "mean_long_short_half_l1_turnover_sum":
                    float(
                        group[
                            "long_short_half_l1_turnover_sum"
                        ]
                        .mean()
                    ),

                "mean_roundtrip_traded_notional_ratio":
                    float(
                        group[
                            "roundtrip_traded_notional_ratio"
                        ]
                        .mean()
                    ),

                "annualized_q1_drift_turnover":
                    float(
                        12
                        *
                        group[
                            "q1_drift_turnover"
                        ]
                        .mean()
                    ),

                "annualized_q5_drift_turnover":
                    float(
                        12
                        *
                        group[
                            "q5_drift_turnover"
                        ]
                        .mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 16. Transaction-Cost Summary
# ============================================================

def build_cost_summary(
    monthly_cost,
):

    rows = []

    group_cols = [

        "factor_order",

        "factor_name",

        "method",

        "window",

        "one_way_cost_bps",
    ]

    for keys, group in (
        monthly_cost.groupby(
            group_cols,
            sort=False,
        )
    ):

        (
            factor_order,
            factor_name,
            method,
            window,
            cost_bps,
        ) = keys

        group = (
            group.sort_values(
                "analysis_date"
            )
        )

        gross = (
            group[
                "gross_q5_minus_q1"
            ]
            .to_numpy(
                dtype=float
            )
        )

        net = (
            group[
                "canonical_net_q5_minus_q1"
            ]
            .to_numpy(
                dtype=float
            )
        )

        gross_hac = calendar_gap_hac(

            gross,

            group[
                "analysis_date"
            ],
        )

        net_hac = calendar_gap_hac(

            net,

            group[
                "analysis_date"
            ],
        )

        net_sd = (

            float(
                np.std(
                    net,
                    ddof=1,
                )
            )

            if
            len(
                net
            )
            >=
            2

            else
            np.nan
        )

        rows.append(
            {

                "factor_order":
                    factor_order,

                "factor_name":
                    factor_name,

                "method":
                    method,

                "window":
                    int(
                        window
                    ),

                "one_way_cost_bps":
                    int(
                        cost_bps
                    ),

                "month_count":
                    int(
                        len(
                            group
                        )
                    ),

                "mean_gross_spread":
                    float(
                        np.mean(
                            gross
                        )
                    ),

                "mean_gross_spread_bps":
                    float(
                        np.mean(
                            gross
                        )
                        *
                        10000
                    ),

                "mean_transaction_cost_drag":
                    float(
                        group[
                            "transaction_cost_drag"
                        ]
                        .mean()
                    ),

                "mean_transaction_cost_drag_bps":
                    float(
                        group[
                            "transaction_cost_drag_bps"
                        ]
                        .mean()
                    ),

                "mean_canonical_net_spread":
                    float(
                        np.mean(
                            net
                        )
                    ),

                "mean_canonical_net_spread_bps":
                    float(
                        np.mean(
                            net
                        )
                        *
                        10000
                    ),

                "annualized_arithmetic_net_spread":
                    float(
                        12
                        *
                        np.mean(
                            net
                        )
                    ),

                "annualized_net_spread_sharpe":
                    (
                        float(

                            np.sqrt(
                                12
                            )

                            *

                            np.mean(
                                net
                            )

                            /

                            net_sd
                        )

                        if
                        np.isfinite(
                            net_sd
                        )
                        and
                        net_sd
                        >
                        0

                        else
                        np.nan
                    ),

                "positive_net_spread_share":
                    float(
                        np.mean(
                            net
                            >
                            0
                        )
                    ),

                "negative_net_spread_share":
                    float(
                        np.mean(
                            net
                            <
                            0
                        )
                    ),

                "gross_hac_t_stat":
                    gross_hac[
                        "t"
                    ],

                "net_hac_t_stat":
                    net_hac[
                        "t"
                    ],

                "factor_sign_flipped":
                    False,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 17. Economic / Break-Even Summary
# ============================================================

def build_economic_summary(
    monthly,
):

    eligible = monthly[
        (
            monthly[
                "turnover_status"
            ]
            ==
            "PASS"
        )
        &
        (
            monthly[
                "gross_spread_status"
            ]
            ==
            "PASS"
        )
    ].copy()

    rows = []

    group_cols = [

        "factor_order",

        "factor_name",

        "method",

        "window",
    ]

    for keys, group in (
        eligible.groupby(
            group_cols,
            sort=False,
        )
    ):

        (
            factor_order,
            factor_name,
            method,
            window,
        ) = keys

        gross = float(
            group[
                "gross_q5_minus_q1"
            ]
            .mean()
        )

        traded_notional = float(
            group[
                "roundtrip_traded_notional_ratio"
            ]
            .mean()
        )

        if (
            traded_notional
            >
            0
        ):

            break_even_bps = (

                abs(
                    gross
                )

                /

                traded_notional

                *

                10000
            )

        else:

            break_even_bps = np.nan

        rows.append(
            {

                "factor_order":
                    factor_order,

                "factor_name":
                    factor_name,

                "method":
                    method,

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

                "mean_gross_q5_minus_q1":
                    gross,

                "mean_gross_q5_minus_q1_bps":
                    gross
                    *
                    10000,

                "mean_q1_drift_turnover":
                    float(
                        group[
                            "q1_drift_turnover"
                        ]
                        .mean()
                    ),

                "mean_q5_drift_turnover":
                    float(
                        group[
                            "q5_drift_turnover"
                        ]
                        .mean()
                    ),

                "mean_roundtrip_traded_notional_ratio":
                    traded_notional,

                "descriptive_break_even_one_way_cost_bps":
                    break_even_bps,

                "break_even_is_ex_post_magnitude_diagnostic":
                    True,

                "factor_sign_flipped":
                    False,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 18. Reproduce Step-2 Gross Spread QA
# ============================================================

def validate_against_step2(
    monthly,
):

    reference = pd.read_csv(
        STEP2_MONTHLY_SPREAD_PATH
    )

    reference[
        "analysis_date"
    ] = pd.to_datetime(
        reference[
            "analysis_date"
        ]
    )

    reference = reference[
        reference[
            "target_name"
        ]
        ==
        PRIMARY_RETURN
    ].copy()

    left = monthly[
        [
            "analysis_date",

            "window",

            "factor_name",

            "method",

            "gross_spread_status",

            "gross_q5_minus_q1",
        ]
    ].copy()

    right = reference[
        [
            "analysis_date",

            "window",

            "factor_name",

            "method",

            "spread_status",

            "q5_minus_q1",
        ]
    ].copy()

    merged = left.merge(

        right,

        on=[
            "analysis_date",
            "window",
            "factor_name",
            "method",
        ],

        how="inner",

        validate="one_to_one",
    )

    comparable = merged[
        (
            merged[
                "gross_spread_status"
            ]
            ==
            "PASS"
        )
        &
        (
            merged[
                "spread_status"
            ]
            ==
            "PASS"
        )
    ].copy()

    if (
        len(
            comparable
        )
        ==
        0
    ):

        max_diff = np.nan

    else:

        max_diff = float(

            (
                comparable[
                    "gross_q5_minus_q1"
                ]

                -

                comparable[
                    "q5_minus_q1"
                ]
            )
            .abs()
            .max()
        )

    return {

        "comparison_row_count":
            int(
                len(
                    comparable
                )
            ),

        "max_abs_gross_spread_difference":
            max_diff,

        "gross_spread_reproduction_ok":
            bool(

                np.isfinite(
                    max_diff
                )

                and

                max_diff
                <
                1e-12
            ),
    }


# ============================================================
# 19. Formal QA
# ============================================================

def formal_qa(
    monthly,
    monthly_cost,
    reproduction_qa,
):

    qa = {}

    qa[
        "turnover_status_counts"
    ] = (
        monthly[
            "turnover_status"
        ]
        .value_counts()
        .to_dict()
    )

    passed = monthly[
        monthly[
            "turnover_status"
        ]
        ==
        "PASS"
    ]

    turnover_values = pd.concat(
        [
            passed[
                "q1_drift_turnover"
            ],

            passed[
                "q5_drift_turnover"
            ],

            passed[
                "q1_target_turnover"
            ],

            passed[
                "q5_target_turnover"
            ],
        ],
        ignore_index=True,
    ).dropna()

    qa[
        "all_turnover_in_0_1"
    ] = bool(

        (
            turnover_values
            >=
            -1e-12
        )
        .all()

        and

        (
            turnover_values
            <=
            1
            +
            1e-12
        )
        .all()
    )

    qa[
        "all_cost_drag_nonnegative"
    ] = bool(
        (
            monthly_cost[
                "transaction_cost_drag"
            ]
            >=
            -1e-15
        )
        .all()
    )

    qa[
        "net_equals_gross_minus_cost"
    ] = bool(

        np.allclose(

            monthly_cost[
                "canonical_net_q5_minus_q1"
            ],

            monthly_cost[
                "gross_q5_minus_q1"
            ]

            -

            monthly_cost[
                "transaction_cost_drag"
            ],

            rtol=0,

            atol=1e-14,
        )
    )

    qa[
        "gross_spread_reproduction_ok"
    ] = (
        reproduction_qa[
            "gross_spread_reproduction_ok"
        ]
    )

    qa[
        "max_abs_step2_spread_difference"
    ] = (
        reproduction_qa[
            "max_abs_gross_spread_difference"
        ]
    )

    qa[
        "future_return_used_to_form_portfolio"
    ] = False

    qa[
        "factor_sign_flip"
    ] = False

    qa[
        "window_selection"
    ] = False

    qa[
        "initial_formation_cost_included"
    ] = False

    qa[
        "terminal_liquidation_cost_included"
    ] = False

    required = [

        "all_turnover_in_0_1",

        "all_cost_drag_nonnegative",

        "net_equals_gross_minus_cost",

        "gross_spread_reproduction_ok",
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
# 20. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 7 - Step 4"
    )

    print(
        "Turnover & Transaction-Cost Validation"
    )

    print("=" * 80)

    # ========================================================
    # Upstream
    # ========================================================

    (
        step2_metadata,
        step3_metadata,
    ) = validate_upstream()

    core = load_core()

    # ========================================================
    # Freeze design
    # ========================================================

    design_payload = {

        "research_day":
            7,

        "step":
            (
                "Step4_Turnover_"
                "and_Transaction_Cost"
            ),

        "parent_day7_step2_design_hash":
            step2_metadata.get(
                "day7_step2_design_hash"
            ),

        "parent_day7_step3_design_hash":
            step3_metadata.get(
                "day7_step3_design_hash"
            ),

        "factors":
            core[
                "factor_name"
            ]
            .astype(str)
            .tolist(),

        "windows":
            EXPECTED_WINDOWS,

        "methods":
            METHODS,

        "portfolio_definition":
            (
                "Q1 = lowest factor values; "
                "Q5 = highest factor values."
            ),

        "canonical_spread":
            "Q5 - Q1",

        "portfolio_weighting":
            "Equal weight",

        "primary_return":
            PRIMARY_RETURN,

        "weight_drift_return":
            DRIFT_RETURN,

        "turnover_definition":
            (
                "0.5 * sum absolute difference "
                "between current target weights and "
                "previous post-return pre-trade weights."
            ),

        "transaction_cost_formula":
            (
                "2 * one_way_cost * "
                "(Q5 half-L1 turnover + "
                "Q1 half-L1 turnover)"
            ),

        "one_way_cost_scenarios_bps":
            COST_SCENARIOS_BPS,

        "missing_drift_return_imputation":
            None,

        "initial_formation_cost_included":
            False,

        "terminal_liquidation_cost_included":
            False,

        "factor_sign_flip":
            False,

        "outcome_based_factor_selection":
            False,

        "outcome_based_window_selection":
            False,
    }

    design = {

        **design_payload,

        "day7_step4_design_hash":
            canonical_hash(
                design_payload
            ),
    }

    print()
    print(
        "Day-7 Step-4 design hash:"
    )

    print(
        design[
            "day7_step4_design_hash"
        ]
    )

    # ========================================================
    # Panel
    # ========================================================

    print()
    print(
        "[1] Load factor, label and characteristic panel"
    )

    panel = load_panel(
        core
    )

    print(
        f"Panel rows: "
        f"{len(panel):,}"
    )

    # ========================================================
    # Monthly turnover
    # ========================================================

    print()
    print(
        "[2] Reconstruct quintiles and compute turnover"
    )

    (
        monthly_turnover,
        monthly_cost,
    ) = run_turnover_analysis(

        panel,

        core,
    )

    save_csv_atomic(
        monthly_turnover,
        MONTHLY_TURNOVER_PATH,
    )

    save_csv_atomic(
        monthly_cost,
        MONTHLY_COST_PATH,
    )

    # ========================================================
    # Reproduce Step-2 spread
    # ========================================================

    print()
    print(
        "[3] Validate reconstructed gross spreads"
    )

    reproduction_qa = (
        validate_against_step2(
            monthly_turnover
        )
    )

    print(
        reproduction_qa
    )

    # ========================================================
    # Summaries
    # ========================================================

    print()
    print(
        "[4] Build turnover summary"
    )

    turnover_summary = (
        build_turnover_summary(
            monthly_turnover
        )
    )

    save_csv_atomic(
        turnover_summary,
        TURNOVER_SUMMARY_PATH,
    )

    print()
    print(
        "[5] Build transaction-cost summary"
    )

    cost_summary = (
        build_cost_summary(
            monthly_cost
        )
    )

    save_csv_atomic(
        cost_summary,
        COST_SUMMARY_PATH,
    )

    print()
    print(
        "[6] Build economic break-even summary"
    )

    economic_summary = (
        build_economic_summary(
            monthly_turnover
        )
    )

    save_csv_atomic(
        economic_summary,
        ECONOMIC_SUMMARY_PATH,
    )

    # ========================================================
    # Formal QA
    # ========================================================

    qa = formal_qa(

        monthly_turnover,

        monthly_cost,

        reproduction_qa,
    )

    qa_df = pd.DataFrame(
        [
            {
                "qa_name":
                    key,

                "qa_value":
                    value,
            }

            for key, value in (
                qa.items()
            )
        ]
    )

    save_csv_atomic(
        qa_df,
        QA_PATH,
    )

    if (
        not qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            "Day-7 Step-4 formal QA failed.\n"
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

        "important_interpretation_rules":
            [

                (
                    "Formal portfolio orientation remains "
                    "Q5 minus Q1 for every factor."
                ),

                (
                    "Negative gross spread is not flipped "
                    "into Q1 minus Q5 after observing outcomes."
                ),

                (
                    "Transaction costs are based on "
                    "drift-adjusted turnover."
                ),

                (
                    "Target-to-target turnover is diagnostic only."
                ),

                (
                    "The first available formation is excluded "
                    "from turnover-cost averaging because there "
                    "is no prior portfolio."
                ),

                (
                    "No terminal liquidation cost is imposed."
                ),

                (
                    "Break-even cost is an ex-post magnitude "
                    "diagnostic, not evidence of a pre-specified "
                    "tradable strategy."
                ),
            ],

        "outputs":
            {

                "monthly_portfolio_turnover":
                    str(
                        MONTHLY_TURNOVER_PATH
                    ),

                "monthly_cost_adjusted_spreads":
                    str(
                        MONTHLY_COST_PATH
                    ),

                "turnover_summary":
                    str(
                        TURNOVER_SUMMARY_PATH
                    ),

                "transaction_cost_summary":
                    str(
                        COST_SUMMARY_PATH
                    ),

                "turnover_economic_summary":
                    str(
                        ECONOMIC_SUMMARY_PATH
                    ),

                "formal_qa":
                    str(
                        QA_PATH
                    ),
            },
    }

    save_json_atomic(
        metadata,
        METADATA_PATH,
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Day-7 Step-4 Formal QA"
    )

    print("=" * 80)

    print(
        f"Turnover statuses: "
        f"{qa['turnover_status_counts']}"
    )

    print(
        f"All turnover in [0,1]: "
        f"{qa['all_turnover_in_0_1']}"
    )

    print(
        f"Step-2 gross-spread reproduction: "
        f"{qa['gross_spread_reproduction_ok']}"
    )

    print(
        f"Max spread difference: "
        f"{qa['max_abs_step2_spread_difference']}"
    )

    print(
        f"Formal QA pass: "
        f"{qa['all_formal_qa_pass']}"
    )

    print()
    print("=" * 80)

    print(
        "Full-Control Turnover Summary"
    )

    print("=" * 80)

    display = economic_summary[
        economic_summary[
            "method"
        ]
        ==
        "FULL_CHARACTERISTIC_NEUTRAL"
    ]

    display_columns = [

        "factor_name",

        "window",

        "mean_gross_q5_minus_q1_bps",

        "mean_q1_drift_turnover",

        "mean_q5_drift_turnover",

        "mean_roundtrip_traded_notional_ratio",

        "descriptive_break_even_one_way_cost_bps",
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
        "Day 7 Step 4 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()
