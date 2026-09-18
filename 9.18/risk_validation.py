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

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)


# ------------------------------------------------------------
# Day 5 final factor panel
# Research Day 5 is stored under M1_day4
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
# Research Day 6 is stored under M1_day5
# ------------------------------------------------------------

DAY6_ROOT = (
    OUTPUT_ROOT
    / "M1_day6"
)


# ------------------------------------------------------------
# Step 1
# ------------------------------------------------------------

STEP1_DIR = (
    DAY6_ROOT
    / "01_step1_freeze_reduced_factor_set"
)

FROZEN_CORE_PATH = (
    STEP1_DIR
    / "frozen_core_factor_set.csv"
)

DAY6_FACTOR_DESIGN_PATH = (
    STEP1_DIR
    / "day6_factor_design.json"
)


# ------------------------------------------------------------
# Step 2
# ------------------------------------------------------------

STEP2_DIR = (
    DAY6_ROOT
    / "02_step2_forward_labels"
)

FORWARD_LABEL_PANEL_PATH = (
    STEP2_DIR
    / "forward_label_panel.parquet"
)

STEP2_LABEL_DESIGN_PATH = (
    STEP2_DIR
    / "step2_label_design.json"
)

STEP2_METADATA_PATH = (
    STEP2_DIR
    / "step2_forward_label_metadata.json"
)


# ------------------------------------------------------------
# Step 3 / Step 4
#
# Only lineage / QA.
# Never use their outcomes to redefine factor signs or windows.
# ------------------------------------------------------------

STEP3_METADATA_PATH = (
    DAY6_ROOT
    / "03_step3_alpha_rank_ic"
    / "step3_alpha_rank_ic_metadata.json"
)

STEP4_METADATA_PATH = (
    DAY6_ROOT
    / "04_step4_portfolio_sort"
    / "step4_portfolio_sort_metadata.json"
)


# ------------------------------------------------------------
# Step 5 output
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY6_ROOT
    / "05_step5_risk_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MONTHLY_IC_PATH = (
    OUTPUT_DIR
    / "monthly_risk_rank_ic.csv"
)

IC_SUMMARY_PATH = (
    OUTPUT_DIR
    / "risk_rank_ic_summary.csv"
)

MONTHLY_PORTFOLIO_PATH = (
    OUTPUT_DIR
    / "monthly_risk_portfolios.csv"
)

MONTHLY_SPREAD_PATH = (
    OUTPUT_DIR
    / "monthly_risk_quintile_spreads.csv"
)

QUINTILE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "risk_quintile_summary.csv"
)

CROSS_WINDOW_PATH = (
    OUTPUT_DIR
    / "risk_cross_window_consistency.csv"
)

COMBINED_SUMMARY_PATH = (
    OUTPUT_DIR
    / "risk_validation_combined_summary.csv"
)

SORT_DIAGNOSTIC_PATH = (
    OUTPUT_DIR
    / "risk_sort_diagnostics.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "step5_risk_validation_metadata.json"
)


# ============================================================
# 1. Frozen Step-5 Design
# ============================================================

EXPECTED_CORE_FACTOR_COUNT = 9

EXPECTED_WINDOWS = [
    60,
    120,
    252,
]


# ------------------------------------------------------------
# Cross-sectional IC rule
# ------------------------------------------------------------

MIN_CROSS_SECTION_OBS = 100


# ------------------------------------------------------------
# Risk-quintile rule
# ------------------------------------------------------------

N_PORTFOLIOS = 5

PORTFOLIO_LABELS = [
    1,
    2,
    3,
    4,
    5,
]

MIN_SORT_OBS = 100

MIN_PORTFOLIO_RISK_OBS = 20


# ------------------------------------------------------------
# Newey-West / HAC
# ------------------------------------------------------------

USE_HAC = True


# ============================================================
# 2. Formal Risk Targets
# ============================================================

RISK_TARGETS = [

    {
        "target_name":
            "future_realized_vol_annualized",

        "target_role":
            "TOTAL_VOL",

        "description":
            "Forward annualized realized volatility.",

        "higher_value_means":
            "HIGHER_RISK",
    },

    {
        "target_name":
            "future_downside_vol_annualized",

        "target_role":
            "DOWNSIDE_VOL",

        "description":
            "Forward annualized downside volatility.",

        "higher_value_means":
            "HIGHER_RISK",
    },

    {
        "target_name":
            "future_max_drawdown",

        "target_role":
            "MAX_DRAWDOWN",

        "description":
            "Forward maximum drawdown magnitude.",

        "higher_value_means":
            "HIGHER_RISK",
    },

    {
        "target_name":
            "future_idio_vol_annualized",

        "target_role":
            "IDIOSYNCRATIC_VOL",

        "description":
            (
                "Forward annualized volatility of "
                "stock holding return minus LOO "
                "equal-weight market holding return."
            ),

        "higher_value_means":
            "HIGHER_RISK",
    },
]


# ============================================================
# 3. Keys
# ============================================================

KEY_COLUMNS = [

    "analysis_date",

    "window",

    "master_index",

    "security_id",
]


LABEL_DIAGNOSTICS = [

    "future_risk_label_valid",

    "future_expected_trade_days",

    "future_valid_return_days",

    "future_return_coverage",
]


# ============================================================
# 4. Utilities
# ============================================================

def load_json(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing JSON file:\n{path}"
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


def canonical_hash(
    obj,
):

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


# ============================================================
# 5. Validate Upstream Research Lineage
# ============================================================

def validate_upstream():

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    factor_design = load_json(
        DAY6_FACTOR_DESIGN_PATH
    )

    if (
        factor_design.get(
            "design_status"
        )
        !=
        "FROZEN"
    ):

        raise RuntimeError(
            "Day-6 factor design is not FROZEN."
        )

    factor_hash = (
        factor_design.get(
            "day6_design_hash"
        )
    )

    if not factor_hash:

        raise RuntimeError(
            "Missing day6_design_hash."
        )

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    label_design = load_json(
        STEP2_LABEL_DESIGN_PATH
    )

    if (
        label_design.get(
            "label_version"
        )
        !=
        "holding_return_v2"
    ):

        raise RuntimeError(
            "Risk validation must use "
            "holding_return_v2 labels."
        )

    if (
        label_design.get(
            "parent_day6_factor_design_hash"
        )
        !=
        factor_hash
    ):

        raise RuntimeError(
            "Factor design / label design mismatch."
        )

    label_hash = (
        label_design.get(
            "step2_label_design_hash"
        )
    )

    step2_metadata = load_json(
        STEP2_METADATA_PATH
    )

    if (
        step2_metadata.get(
            "formal_qa",
            {}
        ).get(
            "all_jobs_pass"
        )
        is not True
    ):

        raise RuntimeError(
            "Step-2 formal QA failed."
        )

    # --------------------------------------------------------
    # Step 3 optional lineage
    # --------------------------------------------------------

    if STEP3_METADATA_PATH.exists():

        step3 = load_json(
            STEP3_METADATA_PATH
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
                "Step-3 QA failed."
            )

    # --------------------------------------------------------
    # Step 4 optional lineage
    # --------------------------------------------------------

    if STEP4_METADATA_PATH.exists():

        step4 = load_json(
            STEP4_METADATA_PATH
        )

        if (
            step4.get(
                "formal_qa",
                {}
            ).get(
                "all_formal_qa_pass"
            )
            is not True
        ):

            raise RuntimeError(
                "Step-4 QA failed."
            )

    return (
        factor_hash,
        label_hash,
    )


# ============================================================
# 6. Load Frozen Factors
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
            "No factor_name in frozen core factor file."
        )

    if (
        core[
            "factor_name"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate frozen factor names."
        )

    if (
        len(
            core
        )
        !=
        EXPECTED_CORE_FACTOR_COUNT
    ):

        raise RuntimeError(
            "Unexpected number of core factors."
        )

    if (
        "priority"
        not in core.columns
    ):

        core[
            "priority"
        ] = (
            np.arange(
                len(
                    core
                )
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
# 7. Validate Parquet Schemas
# ============================================================

def validate_schemas(
    factor_names,
):

    factor_schema = (
        pq.ParquetFile(
            FACTOR_PANEL_PATH
        )
        .schema_arrow
        .names
    )

    missing_factor = [

        x

        for x in (
            KEY_COLUMNS
            +
            factor_names
        )

        if x
        not in factor_schema
    ]

    if missing_factor:

        raise RuntimeError(
            f"Factor columns missing:\n"
            f"{missing_factor}"
        )

    label_schema = (
        pq.ParquetFile(
            FORWARD_LABEL_PANEL_PATH
        )
        .schema_arrow
        .names
    )

    risk_names = [
        x[
            "target_name"
        ]
        for x in RISK_TARGETS
    ]

    missing_label = [

        x

        for x in (
            KEY_COLUMNS
            +
            LABEL_DIAGNOSTICS
            +
            risk_names
        )

        if x
        not in label_schema
    ]

    if missing_label:

        raise RuntimeError(
            f"Risk-label columns missing:\n"
            f"{missing_label}"
        )


# ============================================================
# 8. Load Analysis Panel
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

    risk_names = [
        x[
            "target_name"
        ]
        for x in RISK_TARGETS
    ]

    validate_schemas(
        factor_names
    )

    print(
        "Loading factor panel..."
    )

    factors = pd.read_parquet(
        FACTOR_PANEL_PATH,
        columns=(
            KEY_COLUMNS
            +
            factor_names
        ),
    )

    print(
        f"Factor rows: {len(factors):,}"
    )

    print(
        "Loading forward risk labels..."
    )

    labels = pd.read_parquet(
        FORWARD_LABEL_PANEL_PATH,
        columns=(
            KEY_COLUMNS
            +
            LABEL_DIAGNOSTICS
            +
            risk_names
        ),
    )

    print(
        f"Label rows: {len(labels):,}"
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
        ).astype(
            int
        )

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

    merged = factors.merge(

        labels,

        on=KEY_COLUMNS,

        how="left",

        validate="one_to_one",

        indicator=True,
    )

    if (
        (
            merged[
                "_merge"
            ]
            !=
            "both"
        )
        .any()
    ):

        raise RuntimeError(
            "Factor rows without matching risk labels."
        )

    merged = merged.drop(
        columns="_merge"
    )

    windows = sorted(
        merged[
            "window"
        ]
        .unique()
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

    formal_valid = (
        merged[
            "future_risk_label_valid"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    for risk_name in risk_names:

        if (
            merged.loc[
                formal_valid,
                risk_name,
            ]
            .isna()
            .any()
        ):

            raise RuntimeError(
                f"Valid risk label contains NA: "
                f"{risk_name}"
            )

    return merged


# ============================================================
# 9. Spearman Rank IC
# ============================================================

def spearman_ic(
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

        &

        np.isfinite(
            x
        )

        &

        np.isfinite(
            y
        )
    )

    x = x.loc[
        valid
    ]

    y = y.loc[
        valid
    ]

    n = len(
        x
    )

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
    ):

        return (
            np.nan,
            n,
            "NO_FACTOR_VARIATION",
        )

    if (
        y.nunique()
        <
        2
    ):

        return (
            np.nan,
            n,
            "NO_TARGET_VARIATION",
        )

    xr = (
        x.rank(
            method="average"
        )
        .to_numpy(
            dtype=float
        )
    )

    yr = (
        y.rank(
            method="average"
        )
        .to_numpy(
            dtype=float
        )
    )

    ic = float(
        np.corrcoef(
            xr,
            yr,
        )[
            0,
            1
        ]
    )

    if not np.isfinite(
        ic
    ):

        return (
            np.nan,
            n,
            "NONFINITE_IC",
        )

    return (
        float(
            np.clip(
                ic,
                -1.0,
                1.0,
            )
        ),
        n,
        "PASS",
    )


# ============================================================
# 10. Monthly Risk Rank IC
# ============================================================

def compute_monthly_risk_ic(
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

    rows = []

    grouped = panel.groupby(
        [
            "analysis_date",
            "window",
        ],
        sort=True,
    )

    for (
        analysis_date,
        window,
    ), group in grouped:

        universe_n = len(
            group
        )

        risk_valid = (
            group[
                "future_risk_label_valid"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

        for target_spec in RISK_TARGETS:

            target_name = (
                target_spec[
                    "target_name"
                ]
            )

            target_role = (
                target_spec[
                    "target_role"
                ]
            )

            y = pd.to_numeric(
                group[
                    target_name
                ],
                errors="coerce",
            )

            target_valid = (

                risk_valid

                &

                y.notna()

                &

                np.isfinite(
                    y
                )
            )

            label_valid_n = int(
                target_valid.sum()
            )

            for (
                factor_order,
                factor_name,
            ) in enumerate(
                factor_names,
                start=1,
            ):

                x = pd.to_numeric(
                    group[
                        factor_name
                    ],
                    errors="coerce",
                )

                factor_valid = (

                    x.notna()

                    &

                    np.isfinite(
                        x
                    )
                )

                pair_mask = (

                    target_valid

                    &

                    factor_valid
                )

                (
                    ic,
                    pair_n,
                    status,
                ) = spearman_ic(

                    x.loc[
                        pair_mask
                    ],

                    y.loc[
                        pair_mask
                    ],
                )

                rows.append(
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

                        "target_role":
                            target_role,

                        "higher_target_value_means":
                            "HIGHER_RISK",

                        "factor_universe_n":
                            int(
                                universe_n
                            ),

                        "label_valid_n":
                            label_valid_n,

                        "factor_nonmissing_n":
                            int(
                                factor_valid.sum()
                            ),

                        "pair_n":
                            int(
                                pair_n
                            ),

                        "pair_share":
                            (
                                pair_n
                                /
                                universe_n
                                if
                                universe_n
                                >
                                0
                                else
                                np.nan
                            ),

                        "risk_rank_ic":
                            ic,

                        "ic_status":
                            status,
                    }
                )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "target_role",
                "factor_order",
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 11. HAC / Newey-West Inference
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


def mean_inference(
    values,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(
            x
        )
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

            "naive_se":
                np.nan,

            "naive_t":
                np.nan,

            "hac_lag":
                0,

            "hac_se":
                np.nan,

            "hac_t":
                np.nan,
        }

    mean_x = float(
        x.mean()
    )

    sd_x = float(
        x.std(
            ddof=1
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

            "naive_se":
                naive_se,

            "naive_t":
                naive_t,

            "hac_lag":
                0,

            "hac_se":
                naive_se,

            "hac_t":
                naive_t,
        }

    lag = automatic_hac_lag(
        n
    )

    resid = (
        x
        -
        mean_x
    )

    gamma0 = float(
        np.dot(
            resid,
            resid,
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
                resid[
                    ell:
                ],
                resid[
                    :-ell
                ],
            )
            /
            n
        )

        weight = (
            1.0
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
        0.0,
    )

    hac_var_mean = (
        long_run_var
        /
        n
    )

    hac_se = (
        float(
            np.sqrt(
                hac_var_mean
            )
        )
        if
        hac_var_mean
        >
        0
        else
        np.nan
    )

    hac_t = (
        mean_x
        /
        hac_se
        if
        np.isfinite(
            hac_se
        )
        and
        hac_se
        >
        0
        else
        np.nan
    )

    return {

        "naive_se":
            naive_se,

        "naive_t":
            naive_t,

        "hac_lag":
            int(
                lag
            ),

        "hac_se":
            hac_se,

        "hac_t":
            hac_t,
    }


# ============================================================
# 12. Risk-IC Summary
# ============================================================

def summarize_risk_ic(
    monthly_ic,
):

    rows = []

    passed = (
        monthly_ic[
            monthly_ic[
                "ic_status"
            ]
            ==
            "PASS"
        ]
    )

    group_cols = [

        "factor_order",

        "factor_name",

        "target_name",

        "target_role",

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
            target_role,
            window,
        ) = keys

        group = (
            group.sort_values(
                "analysis_date"
            )
        )

        x = (
            group[
                "risk_rank_ic"
            ]
            .to_numpy(
                dtype=float
            )
        )

        n = len(
            x
        )

        mean_ic = float(
            np.mean(
                x
            )
        )

        median_ic = float(
            np.median(
                x
            )
        )

        sd_ic = (
            float(
                np.std(
                    x,
                    ddof=1,
                )
            )
            if
            n
            >=
            2
            else
            np.nan
        )

        inference = mean_inference(
            x
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

                "target_role":
                    target_role,

                "window":
                    int(
                        window
                    ),

                "month_count":
                    int(
                        n
                    ),

                "mean_risk_rank_ic":
                    mean_ic,

                "median_risk_rank_ic":
                    median_ic,

                "std_risk_rank_ic":
                    sd_ic,

                "risk_icir_monthly":
                    (
                        mean_ic
                        /
                        sd_ic
                        if
                        np.isfinite(
                            sd_ic
                        )
                        and
                        sd_ic
                        >
                        0
                        else
                        np.nan
                    ),

                "risk_icir_annualized":
                    (
                        np.sqrt(
                            12
                        )
                        *
                        mean_ic
                        /
                        sd_ic
                        if
                        np.isfinite(
                            sd_ic
                        )
                        and
                        sd_ic
                        >
                        0
                        else
                        np.nan
                    ),

                "positive_ic_share":
                    float(
                        np.mean(
                            x
                            >
                            0
                        )
                    ),

                "negative_ic_share":
                    float(
                        np.mean(
                            x
                            <
                            0
                        )
                    ),

                "mean_pair_n":
                    float(
                        group[
                            "pair_n"
                        ]
                        .mean()
                    ),

                "mean_pair_share":
                    float(
                        group[
                            "pair_share"
                        ]
                        .mean()
                    ),

                "hac_lag":
                    inference[
                        "hac_lag"
                    ],

                "hac_ic_se":
                    inference[
                        "hac_se"
                    ],

                "hac_ic_t_stat":
                    inference[
                        "hac_t"
                    ],

                "outcome_based_sign_flip":
                    False,

                "outcome_based_window_selection":
                    False,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "target_role",
                "factor_order",
                "window",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 13. Conservative Quintile Assignment
#
# Ties are NOT broken artificially using future information.
# If qcut cannot construct five genuine groups, the month is
# not used for quintile-gradient analysis.
# ============================================================

def assign_quintiles(
    factor,
):

    factor = pd.to_numeric(
        factor,
        errors="coerce",
    )

    valid = (

        factor.notna()

        &

        np.isfinite(
            factor
        )
    )

    x = factor.loc[
        valid
    ]

    result = pd.Series(
        pd.NA,
        index=factor.index,
        dtype="Int64",
    )

    diag = {

        "factor_nonmissing_n":
            int(
                len(
                    x
                )
            ),

        "factor_unique_n":
            int(
                x.nunique()
            ),

        "sort_status":
            "PASS",

        "portfolio_count":
            0,
    }

    if (
        len(
            x
        )
        <
        MIN_SORT_OBS
    ):

        diag[
            "sort_status"
        ] = "INSUFFICIENT_SORT_OBS"

        return (
            result,
            diag,
        )

    if (
        x.nunique()
        <
        N_PORTFOLIOS
    ):

        diag[
            "sort_status"
        ] = "INSUFFICIENT_UNIQUE_VALUES"

        return (
            result,
            diag,
        )

    try:

        q = pd.qcut(

            x,

            q=N_PORTFOLIOS,

            labels=False,

            duplicates="drop",
        )

    except ValueError:

        diag[
            "sort_status"
        ] = "QCUT_FAILURE"

        return (
            result,
            diag,
        )

    portfolio_count = int(
        q.nunique()
    )

    diag[
        "portfolio_count"
    ] = portfolio_count

    if (
        portfolio_count
        !=
        N_PORTFOLIOS
    ):

        diag[
            "sort_status"
        ] = "FEWER_THAN_5_PORTFOLIOS"

        return (
            result,
            diag,
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
        diag,
    )


# ============================================================
# 14. Risk Quintile Analysis
# ============================================================

def compute_risk_quintiles(
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

    portfolio_rows = []

    spread_rows = []

    diagnostic_rows = []

    grouped = panel.groupby(
        [
            "analysis_date",
            "window",
        ],
        sort=True,
    )

    for (
        analysis_date,
        window,
    ), group in grouped:

        group = (
            group.copy()
            .reset_index(
                drop=True
            )
        )

        risk_valid = (
            group[
                "future_risk_label_valid"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

        for (
            factor_order,
            factor_name,
        ) in enumerate(
            factor_names,
            start=1,
        ):

            (
                quintile,
                diag,
            ) = assign_quintiles(
                group[
                    factor_name
                ]
            )

            work = group.copy()

            work[
                "portfolio"
            ] = quintile

            counts = (
                work[
                    "portfolio"
                ]
                .value_counts()
                .to_dict()
            )

            diagnostic_rows.append(
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

                    **diag,

                    "q1_sort_n":
                        int(
                            counts.get(
                                1,
                                0,
                            )
                        ),

                    "q5_sort_n":
                        int(
                            counts.get(
                                5,
                                0,
                            )
                        ),
                }
            )

            if (
                diag[
                    "sort_status"
                ]
                !=
                "PASS"
            ):

                continue

            for target_spec in RISK_TARGETS:

                target_name = (
                    target_spec[
                        "target_name"
                    ]
                )

                target_role = (
                    target_spec[
                        "target_role"
                    ]
                )

                y = pd.to_numeric(
                    work[
                        target_name
                    ],
                    errors="coerce",
                )

                valid_target = (

                    risk_valid

                    &

                    y.notna()

                    &

                    np.isfinite(
                        y
                    )

                    &

                    work[
                        "portfolio"
                    ]
                    .notna()
                )

                q_values = {}

                q_ns = {}

                for q in PORTFOLIO_LABELS:

                    mask = (

                        valid_target

                        &

                        (
                            work[
                                "portfolio"
                            ]
                            ==
                            q
                        )
                    )

                    values = y.loc[
                        mask
                    ]

                    n_q = len(
                        values
                    )

                    q_ns[
                        q
                    ] = int(
                        n_q
                    )

                    if (
                        n_q
                        >=
                        MIN_PORTFOLIO_RISK_OBS
                    ):

                        mean_risk = float(
                            values.mean()
                        )

                        median_risk = float(
                            values.median()
                        )

                    else:

                        mean_risk = np.nan

                        median_risk = np.nan

                    q_values[
                        q
                    ] = mean_risk

                    portfolio_rows.append(
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

                            "target_name":
                                target_name,

                            "target_role":
                                target_role,

                            "portfolio":
                                q,

                            "return_observation_n":
                                n_q,

                            "mean_future_risk":
                                mean_risk,

                            "median_future_risk":
                                median_risk,
                        }
                    )

                values = np.array(
                    [
                        q_values[
                            q
                        ]
                        for q in PORTFOLIO_LABELS
                    ],
                    dtype=float,
                )

                if (
                    np.isfinite(
                        values
                    )
                    .all()
                ):

                    spread = float(
                        values[
                            4
                        ]
                        -
                        values[
                            0
                        ]
                    )

                    portfolio_rank_corr = float(
                        pd.Series(
                            values
                        )
                        .corr(
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

                    diff = np.diff(
                        values
                    )

                    increasing = bool(
                        np.all(
                            diff
                            >
                            0
                        )
                    )

                    decreasing = bool(
                        np.all(
                            diff
                            <
                            0
                        )
                    )

                    status = "PASS"

                else:

                    spread = np.nan

                    portfolio_rank_corr = np.nan

                    increasing = False

                    decreasing = False

                    status = (
                        "INSUFFICIENT_PORTFOLIO_RISK_OBS"
                    )

                spread_rows.append(
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

                        "target_role":
                            target_role,

                        "q1_n":
                            q_ns[
                                1
                            ],

                        "q2_n":
                            q_ns[
                                2
                            ],

                        "q3_n":
                            q_ns[
                                3
                            ],

                        "q4_n":
                            q_ns[
                                4
                            ],

                        "q5_n":
                            q_ns[
                                5
                            ],

                        "q1_risk":
                            q_values[
                                1
                            ],

                        "q2_risk":
                            q_values[
                                2
                            ],

                        "q3_risk":
                            q_values[
                                3
                            ],

                        "q4_risk":
                            q_values[
                                4
                            ],

                        "q5_risk":
                            q_values[
                                5
                            ],

                        "q5_minus_q1_risk":
                            spread,

                        "q5_minus_q1_percentage_points":
                            (
                                spread
                                *
                                100
                                if
                                np.isfinite(
                                    spread
                                )
                                else
                                np.nan
                            ),

                        "portfolio_risk_spearman":
                            portfolio_rank_corr,

                        "strictly_increasing_risk":
                            increasing,

                        "strictly_decreasing_risk":
                            decreasing,

                        "spread_status":
                            status,
                    }
                )

    return (
        pd.DataFrame(
            portfolio_rows
        ),
        pd.DataFrame(
            spread_rows
        ),
        pd.DataFrame(
            diagnostic_rows
        ),
    )


# ============================================================
# 15. Quintile Risk Summary
# ============================================================

def summarize_quintile_risk(
    spread_df,
):

    passed = (
        spread_df[
            spread_df[
                "spread_status"
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

        "target_role",

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
            target_role,
            window,
        ) = keys

        group = (
            group.sort_values(
                "analysis_date"
            )
        )

        spread = (
            group[
                "q5_minus_q1_risk"
            ]
            .to_numpy(
                dtype=float
            )
        )

        inference = mean_inference(
            spread
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

                "target_role":
                    target_role,

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

                "mean_q1_risk":
                    float(
                        group[
                            "q1_risk"
                        ]
                        .mean()
                    ),

                "mean_q2_risk":
                    float(
                        group[
                            "q2_risk"
                        ]
                        .mean()
                    ),

                "mean_q3_risk":
                    float(
                        group[
                            "q3_risk"
                        ]
                        .mean()
                    ),

                "mean_q4_risk":
                    float(
                        group[
                            "q4_risk"
                        ]
                        .mean()
                    ),

                "mean_q5_risk":
                    float(
                        group[
                            "q5_risk"
                        ]
                        .mean()
                    ),

                "mean_q5_minus_q1_risk":
                    float(
                        np.mean(
                            spread
                        )
                    ),

                "mean_q5_minus_q1_percentage_points":
                    float(
                        np.mean(
                            spread
                        )
                        *
                        100
                    ),

                "median_q5_minus_q1_risk":
                    float(
                        np.median(
                            spread
                        )
                    ),

                "positive_risk_spread_share":
                    float(
                        np.mean(
                            spread
                            >
                            0
                        )
                    ),

                "negative_risk_spread_share":
                    float(
                        np.mean(
                            spread
                            <
                            0
                        )
                    ),

                "mean_portfolio_risk_spearman":
                    float(
                        group[
                            "portfolio_risk_spearman"
                        ]
                        .mean()
                    ),

                "strictly_increasing_risk_share":
                    float(
                        group[
                            "strictly_increasing_risk"
                        ]
                        .mean()
                    ),

                "strictly_decreasing_risk_share":
                    float(
                        group[
                            "strictly_decreasing_risk"
                        ]
                        .mean()
                    ),

                "hac_lag":
                    inference[
                        "hac_lag"
                    ],

                "hac_risk_spread_se":
                    inference[
                        "hac_se"
                    ],

                "hac_risk_spread_t_stat":
                    inference[
                        "hac_t"
                    ],

                "outcome_based_sign_flip":
                    False,

                "outcome_based_window_selection":
                    False,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "target_role",
                "factor_order",
                "window",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 16. Combined IC + Quintile Summary
# ============================================================

def build_combined_summary(
    ic_summary,
    quintile_summary,
):

    keys = [

        "factor_order",

        "factor_name",

        "factor_orientation",

        "target_name",

        "target_role",

        "window",
    ]

    result = ic_summary.merge(

        quintile_summary,

        on=keys,

        how="left",

        validate="one_to_one",

        suffixes=(
            "_ic",
            "_sort",
        ),
    )

    return (
        result.sort_values(
            [
                "target_role",
                "factor_order",
                "window",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 17. Cross-Window Consistency
# ============================================================

def build_cross_window(
    combined,
):

    rows = []

    group_cols = [

        "factor_order",

        "factor_name",

        "target_name",

        "target_role",
    ]

    for keys, group in (
        combined.groupby(
            group_cols,
            sort=False,
        )
    ):

        (
            factor_order,
            factor_name,
            target_name,
            target_role,
        ) = keys

        g = group.set_index(
            "window"
        )

        ic_map = {}

        spread_map = {}

        for w in EXPECTED_WINDOWS:

            if w in g.index:

                ic_map[
                    w
                ] = float(
                    g.loc[
                        w,
                        "mean_risk_rank_ic",
                    ]
                )

                spread_value = (
                    g.loc[
                        w,
                        "mean_q5_minus_q1_risk",
                    ]
                )

                spread_map[
                    w
                ] = (
                    float(
                        spread_value
                    )
                    if
                    pd.notna(
                        spread_value
                    )
                    else
                    np.nan
                )

            else:

                ic_map[
                    w
                ] = np.nan

                spread_map[
                    w
                ] = np.nan

        ic_values = np.array(
            [
                ic_map[
                    60
                ],
                ic_map[
                    120
                ],
                ic_map[
                    252
                ],
            ],
            dtype=float,
        )

        if (
            np.isfinite(
                ic_values
            )
            .all()
        ):

            ic_signs = np.sign(
                ic_values
            )

            same_ic_sign = bool(

                np.all(
                    ic_signs
                    ==
                    ic_signs[
                        0
                    ]
                )

                and

                np.all(
                    ic_signs
                    !=
                    0
                )
            )

        else:

            same_ic_sign = False

        spread_values = np.array(
            [
                spread_map[
                    60
                ],
                spread_map[
                    120
                ],
                spread_map[
                    252
                ],
            ],
            dtype=float,
        )

        if (
            np.isfinite(
                spread_values
            )
            .all()
        ):

            spread_signs = np.sign(
                spread_values
            )

            same_spread_sign = bool(

                np.all(
                    spread_signs
                    ==
                    spread_signs[
                        0
                    ]
                )

                and

                np.all(
                    spread_signs
                    !=
                    0
                )
            )

        else:

            same_spread_sign = False

        rows.append(
            {

                "factor_order":
                    factor_order,

                "factor_name":
                    factor_name,

                "target_name":
                    target_name,

                "target_role":
                    target_role,

                "mean_ic_w60":
                    ic_map[
                        60
                    ],

                "mean_ic_w120":
                    ic_map[
                        120
                    ],

                "mean_ic_w252":
                    ic_map[
                        252
                    ],

                "same_ic_sign_all_windows":
                    same_ic_sign,

                "risk_spread_w60":
                    spread_map[
                        60
                    ],

                "risk_spread_w120":
                    spread_map[
                        120
                    ],

                "risk_spread_w252":
                    spread_map[
                        252
                    ],

                "same_risk_spread_sign_all_windows":
                    same_spread_sign,

                "factor_orientation":
                    "ORIGINAL",

                "window_selected_from_outcome":
                    False,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "target_role",
                "factor_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 18. Formal QA
# ============================================================

def formal_qa(
    monthly_ic,
    ic_summary,
    spread_df,
    combined,
    core,
):

    qa = {}

    qa[
        "core_factor_count"
    ] = int(
        len(
            core
        )
    )

    qa[
        "core_factor_count_ok"
    ] = bool(
        len(
            core
        )
        ==
        EXPECTED_CORE_FACTOR_COUNT
    )

    qa[
        "monthly_ic_key_unique"
    ] = bool(

        not monthly_ic[
            [
                "analysis_date",
                "window",
                "factor_name",
                "target_name",
            ]
        ]
        .duplicated()
        .any()
    )

    qa[
        "monthly_ic_row_count"
    ] = int(
        len(
            monthly_ic
        )
    )

    qa[
        "ic_status_counts"
    ] = (
        monthly_ic[
            "ic_status"
        ]
        .value_counts()
        .to_dict()
    )

    valid_ic = (
        monthly_ic[
            "risk_rank_ic"
        ]
        .dropna()
    )

    qa[
        "all_ic_in_unit_interval"
    ] = bool(

        (
            valid_ic
            >=
            -1
            -
            1e-10
        )
        .all()

        and

        (
            valid_ic
            <=
            1
            +
            1e-10
        )
        .all()
    )

    qa[
        "spread_key_unique"
    ] = bool(

        not spread_df[
            [
                "analysis_date",
                "window",
                "factor_name",
                "target_name",
            ]
        ]
        .duplicated()
        .any()
    )

    qa[
        "combined_window_set"
    ] = sorted(
        combined[
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
            "combined_window_set"
        ]
        ==
        EXPECTED_WINDOWS
    )

    qa[
        "all_factor_orientation_original"
    ] = bool(
        (
            ic_summary[
                "factor_orientation"
            ]
            ==
            "ORIGINAL"
        )
        .all()
    )

    checks = [

        "core_factor_count_ok",

        "monthly_ic_key_unique",

        "all_ic_in_unit_interval",

        "spread_key_unique",

        "window_set_ok",

        "all_factor_orientation_original",
    ]

    qa[
        "all_formal_qa_pass"
    ] = bool(
        all(
            qa[
                x
            ]
            for x in checks
        )
    )

    return qa


# ============================================================
# 19. Step-5 Design
# ============================================================

def build_step5_design(
    factor_hash,
    label_hash,
    core,
):

    payload = {

        "research_day":
            6,

        "step":
            "Step5_Risk_Validation",

        "parent_factor_design_hash":
            factor_hash,

        "parent_label_design_hash":
            label_hash,

        "factor_set":
            core[
                "factor_name"
            ]
            .astype(str)
            .tolist(),

        "windows":
            EXPECTED_WINDOWS,

        "risk_targets":
            RISK_TARGETS,

        "primary_method":
            (
                "Monthly cross-sectional "
                "Spearman Rank IC."
            ),

        "secondary_method":
            (
                "Monthly factor quintile "
                "future-risk gradient."
            ),

        "quintile_orientation":
            (
                "Q1=lowest factor values; "
                "Q5=highest factor values."
            ),

        "risk_spread_definition":
            "Q5 future risk - Q1 future risk",

        "positive_ic_interpretation":
            (
                "Higher factor value predicts "
                "higher future risk."
            ),

        "negative_ic_interpretation":
            (
                "Higher factor value predicts "
                "lower future risk."
            ),

        "neutralization":
            False,

        "factor_sign_flip":
            False,

        "outcome_based_window_selection":
            False,

        "multiple_testing_adjustment":
            False,

        "hac_inference":
            USE_HAC,
    }

    return {

        **payload,

        "step5_design_hash":
            canonical_hash(
                payload
            ),
    }


# ============================================================
# 20. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 6 - Step 5"
    )

    print(
        "Risk Validation"
    )

    print("=" * 80)

    # ========================================================
    # 20.1 upstream
    # ========================================================

    print()
    print(
        "[1] Validate upstream frozen designs"
    )

    (
        factor_hash,
        label_hash,
    ) = validate_upstream()

    print(
        f"Factor design hash:\n"
        f"{factor_hash}"
    )

    print()
    print(
        f"Label design hash:\n"
        f"{label_hash}"
    )

    # ========================================================
    # 20.2 frozen core
    # ========================================================

    print()
    print(
        "[2] Load frozen core factors"
    )

    core = load_core_factors()

    print(
        core[
            [
                "priority",
                "factor_name",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 20.3 step5 design
    # ========================================================

    design = build_step5_design(

        factor_hash=
            factor_hash,

        label_hash=
            label_hash,

        core=
            core,
    )

    print()
    print(
        "Step-5 design hash:"
    )

    print(
        design[
            "step5_design_hash"
        ]
    )

    # ========================================================
    # 20.4 panel
    # ========================================================

    print()
    print(
        "[3] Load factor + forward-risk panel"
    )

    panel = load_analysis_panel(
        core
    )

    print(
        f"Analysis rows: "
        f"{len(panel):,}"
    )

    # ========================================================
    # 20.5 monthly risk IC
    # ========================================================

    print()
    print(
        "[4] Compute monthly cross-sectional Risk Rank IC"
    )

    monthly_ic = (
        compute_monthly_risk_ic(
            panel,
            core,
        )
    )

    save_csv_atomic(
        monthly_ic,
        MONTHLY_IC_PATH,
    )

    # ========================================================
    # 20.6 IC summary
    # ========================================================

    print()
    print(
        "[5] Summarize Risk Rank IC"
    )

    ic_summary = (
        summarize_risk_ic(
            monthly_ic
        )
    )

    save_csv_atomic(
        ic_summary,
        IC_SUMMARY_PATH,
    )

    # ========================================================
    # 20.7 risk quintile
    # ========================================================

    print()
    print(
        "[6] Compute future-risk quintile gradients"
    )

    (
        portfolio_df,
        spread_df,
        diagnostics_df,
    ) = compute_risk_quintiles(
        panel,
        core,
    )

    save_csv_atomic(
        portfolio_df,
        MONTHLY_PORTFOLIO_PATH,
    )

    save_csv_atomic(
        spread_df,
        MONTHLY_SPREAD_PATH,
    )

    save_csv_atomic(
        diagnostics_df,
        SORT_DIAGNOSTIC_PATH,
    )

    # ========================================================
    # 20.8 quintile summary
    # ========================================================

    print()
    print(
        "[7] Summarize risk quintile gradients"
    )

    quintile_summary = (
        summarize_quintile_risk(
            spread_df
        )
    )

    save_csv_atomic(
        quintile_summary,
        QUINTILE_SUMMARY_PATH,
    )

    # ========================================================
    # 20.9 combined
    # ========================================================

    print()
    print(
        "[8] Build combined IC + quintile summary"
    )

    combined = (
        build_combined_summary(

            ic_summary,

            quintile_summary,
        )
    )

    save_csv_atomic(
        combined,
        COMBINED_SUMMARY_PATH,
    )

    # ========================================================
    # 20.10 cross-window
    # ========================================================

    cross_window = (
        build_cross_window(
            combined
        )
    )

    save_csv_atomic(
        cross_window,
        CROSS_WINDOW_PATH,
    )

    # ========================================================
    # 20.11 QA
    # ========================================================

    qa = formal_qa(

        monthly_ic=
            monthly_ic,

        ic_summary=
            ic_summary,

        spread_df=
            spread_df,

        combined=
            combined,

        core=
            core,
    )

    if (
        not qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            f"Step-5 QA failed:\n{qa}"
        )

    # ========================================================
    # 20.12 metadata
    # ========================================================

    metadata = {

        **design,

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "formal_qa":
            qa,

        "outputs": {

            "monthly_risk_rank_ic":
                str(
                    MONTHLY_IC_PATH
                ),

            "risk_rank_ic_summary":
                str(
                    IC_SUMMARY_PATH
                ),

            "monthly_risk_portfolios":
                str(
                    MONTHLY_PORTFOLIO_PATH
                ),

            "monthly_risk_quintile_spreads":
                str(
                    MONTHLY_SPREAD_PATH
                ),

            "risk_quintile_summary":
                str(
                    QUINTILE_SUMMARY_PATH
                ),

            "risk_cross_window_consistency":
                str(
                    CROSS_WINDOW_PATH
                ),

            "risk_validation_combined_summary":
                str(
                    COMBINED_SUMMARY_PATH
                ),
        },
    }

    save_json_atomic(
        metadata,
        METADATA_PATH,
    )

    # ========================================================
    # 20.13 Console summary
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Step-5 Formal QA"
    )

    print("=" * 80)

    print(
        f"Monthly Risk-IC rows: "
        f"{qa['monthly_ic_row_count']:,}"
    )

    print(
        f"IC status counts: "
        f"{qa['ic_status_counts']}"
    )

    print(
        f"Formal QA pass: "
        f"{qa['all_formal_qa_pass']}"
    )

    print()
    print("=" * 80)

    print(
        "Risk Validation Summary"
    )

    print("=" * 80)

    display_cols = [

        "factor_name",

        "target_role",

        "window",

        "mean_risk_rank_ic",

        "positive_ic_share",

        "hac_ic_t_stat",

        "mean_q5_minus_q1_risk",

        "mean_q5_minus_q1_percentage_points",

        "mean_portfolio_risk_spearman",

        "hac_risk_spread_t_stat",
    ]

    print(
        combined[
            display_cols
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
        "Day 6 Step 5 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()