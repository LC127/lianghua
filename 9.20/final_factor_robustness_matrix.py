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


# ------------------------------------------------------------
# Day 7
# Research Day 7 -> M1_day6
# ------------------------------------------------------------

DAY7_ROOT = (
    OUTPUT_ROOT
    / "M1_day7"
)


# ============================================================
# Step 1
# ============================================================

STEP1_DIR = (
    DAY7_ROOT
    / "01_stage1_subsample_regime_robustness"
)

STEP1_MATRIX_PATH = (
    STEP1_DIR
    / "regime_robustness_summary.csv"
)

STEP1_METADATA_PATH = (
    STEP1_DIR
    / "day7_step1_metadata.json"
)


# ============================================================
# Step 2
# ============================================================

STEP2_DIR = (
    DAY7_ROOT
    / "02_stage2_traditional_characteristic_controls"
)

STEP2_RETENTION_PATH = (
    STEP2_DIR
    / "traditional_control_signal_retention.csv"
)

STEP2_ALPHA_PATH = (
    STEP2_DIR
    / "controlled_alpha_rank_ic_summary.csv"
)

STEP2_PORTFOLIO_PATH = (
    STEP2_DIR
    / "controlled_alpha_portfolio_summary.csv"
)

STEP2_RISK_PATH = (
    STEP2_DIR
    / "controlled_risk_rank_ic_summary.csv"
)

STEP2_METADATA_PATH = (
    STEP2_DIR
    / "day7_step2_metadata.json"
)


# ============================================================
# Step 3
# ============================================================

STEP3_DIR = (
    DAY7_ROOT
    / "03_stage3_fama_macbeth_regression"
)

STEP3_MODEL_PATH = (
    STEP3_DIR
    / "fmb_model_comparison.csv"
)

STEP3_CROSS_WINDOW_PATH = (
    STEP3_DIR
    / "fmb_cross_window_consistency.csv"
)

STEP3_METADATA_PATH = (
    STEP3_DIR
    / "day7_step3_metadata.json"
)


# ============================================================
# Step 4
# ============================================================

STEP4_DIR = (
    DAY7_ROOT
    / "04_stage4_turnover_transaction_cost"
)

STEP4_ECONOMIC_PATH = (
    STEP4_DIR
    / "turnover_economic_summary.csv"
)

STEP4_COST_PATH = (
    STEP4_DIR
    / "transaction_cost_summary.csv"
)

STEP4_METADATA_PATH = (
    STEP4_DIR
    / "day7_step4_metadata.json"
)


# ============================================================
# Day 7 Step 5
# ============================================================

OUTPUT_DIR = (
    DAY7_ROOT
    / "05_stage5_final_factor_robustness_matrix"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


WINDOW_MATRIX_PATH = (
    OUTPUT_DIR
    / "final_factor_robustness_matrix_by_window.csv"
)

FACTOR_MATRIX_PATH = (
    OUTPUT_DIR
    / "final_factor_robustness_matrix.csv"
)

ROLE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "final_factor_role_summary.csv"
)

QA_PATH = (
    OUTPUT_DIR
    / "final_factor_robustness_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day7_step5_metadata.json"
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


PRIMARY_TARGET = (
    "future_excess_return"
)


REGIME_METHOD = (
    "INDUSTRY_SIZE_NEUTRAL"
)

FULL_CONTROL_METHOD = (
    "FULL_CHARACTERISTIC_NEUTRAL"
)


# ------------------------------------------------------------
# Risk targets
# ------------------------------------------------------------

RISK_TARGETS = {

    "future_realized_vol_annualized":
        "total_vol",

    "future_downside_vol_annualized":
        "downside_vol",

    "future_max_drawdown":
        "max_drawdown",

    "future_idio_vol_annualized":
        "ivol",
}


# ------------------------------------------------------------
# Transaction cost scenarios
# ------------------------------------------------------------

EXPECTED_COST_BPS = [
    5,
    10,
    20,
    30,
]


# ============================================================
# 2. Descriptive Evidence Rules
#
# IMPORTANT:
#
# These rules are ONLY used to assign descriptive evidence
# patterns.
#
# They do NOT:
#
# - remove factors
# - select windows
# - flip signs
# - construct a factor ranking
# ============================================================

ALPHA_IC_REFERENCE = 0.005

RISK_IC_REFERENCE = 0.01

NOMINAL_T_REFERENCE = 1.96


# ------------------------------------------------------------
# Transaction-cost descriptive benchmark
#
# 10 bp is used only as a reporting benchmark.
# All 5/10/20/30 bp results remain in the detailed matrix.
# ------------------------------------------------------------

REFERENCE_COST_BPS = 10


# ============================================================
# 3. Utilities
# ============================================================

def load_json(path: Path):

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


def require_columns(
    df: pd.DataFrame,
    columns,
    name,
):

    missing = [
        x
        for x in columns
        if x not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{name} missing required columns:\n"
            f"{missing}"
        )


def same_nonzero_sign(
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

    if len(x) == 0:

        return False

    signs = np.sign(
        x
    )

    if np.any(
        signs == 0
    ):

        return False

    return bool(
        np.all(
            signs
            ==
            signs[0]
        )
    )


def safe_mean(
    values,
):

    x = pd.to_numeric(
        pd.Series(
            values
        ),
        errors="coerce",
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if len(x) == 0:

        return np.nan

    return float(
        x.mean()
    )


def safe_min(
    values,
):

    x = pd.to_numeric(
        pd.Series(
            values
        ),
        errors="coerce",
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if len(x) == 0:

        return np.nan

    return float(
        x.min()
    )


def count_abs_t_ge(
    values,
    threshold=NOMINAL_T_REFERENCE,
):

    x = pd.to_numeric(
        pd.Series(
            values
        ),
        errors="coerce",
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    return int(
        (
            x.abs()
            >=
            threshold
        )
        .sum()
    )


# ============================================================
# 4. Upstream QA
# ============================================================

def validate_upstream():

    metadata_paths = {

        "step1":
            STEP1_METADATA_PATH,

        "step2":
            STEP2_METADATA_PATH,

        "step3":
            STEP3_METADATA_PATH,

        "step4":
            STEP4_METADATA_PATH,
    }

    metadata = {}

    for name, path in (
        metadata_paths.items()
    ):

        meta = load_json(
            path
        )

        if (
            meta.get(
                "formal_qa",
                {}
            ).get(
                "all_formal_qa_pass"
            )
            is not True
        ):

            raise RuntimeError(
                f"{name} formal QA did not pass."
            )

        metadata[
            name
        ] = meta

    required_files = [

        FROZEN_CORE_PATH,

        STEP1_MATRIX_PATH,

        STEP2_RETENTION_PATH,

        STEP2_ALPHA_PATH,

        STEP2_PORTFOLIO_PATH,

        STEP2_RISK_PATH,

        STEP3_MODEL_PATH,

        STEP3_CROSS_WINDOW_PATH,

        STEP4_ECONOMIC_PATH,

        STEP4_COST_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required input:\n{path}"
            )

    return metadata


# ============================================================
# 5. Core Factors
# ============================================================

def load_core():

    core = pd.read_csv(
        FROZEN_CORE_PATH
    )

    require_columns(
        core,
        [
            "factor_name",
        ],
        "frozen_core_factor_set",
    )

    if len(core) != EXPECTED_FACTOR_COUNT:

        raise RuntimeError(
            "Unexpected factor count."
        )

    if core[
        "factor_name"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate frozen factors."
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

    core[
        "priority"
    ] = pd.to_numeric(
        core[
            "priority"
        ],
        errors="raise",
    ).astype(int)

    return (
        core[
            [
                "priority",
                "factor_name",
            ]
        ]
        .sort_values(
            "priority"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 6. Base Factor x Window Grid
# ============================================================

def build_base_grid(
    core,
):

    rows = []

    for row in core.itertuples(
        index=False
    ):

        for window in EXPECTED_WINDOWS:

            rows.append(
                {

                    "factor_order":
                        int(
                            row.priority
                        ),

                    "factor_name":
                        row.factor_name,

                    "window":
                        int(
                            window
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 7. Step-1 Regime Evidence
# ============================================================

def load_regime_evidence():

    df = pd.read_csv(
        STEP1_MATRIX_PATH
    )

    require_columns(
        df,
        [
            "factor_name",
            "window",
            "neutralization",

            "alpha_ic_overall_same_sign_share",
            "alpha_ic_time_same_sign_share",
            "alpha_ic_trend_same_sign_share",
            "alpha_ic_vol_same_sign_share",
            "alpha_ic_joint_same_sign_share",

            "alpha_spread_overall_same_sign_share",

            "ivol_ic_overall_same_sign_share",
            "ivol_ic_time_same_sign_share",
            "ivol_ic_trend_same_sign_share",
            "ivol_ic_vol_same_sign_share",
            "ivol_ic_joint_same_sign_share",
        ],
        "regime_robustness_summary",
    )

    df[
        "window"
    ] = pd.to_numeric(
        df[
            "window"
        ]
    ).astype(int)

    df = df[
        df[
            "neutralization"
        ]
        ==
        REGIME_METHOD
    ].copy()

    if (
        df[
            [
                "factor_name",
                "window",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate Step-1 factor-window rows."
        )

    rename = {

        "alpha_ic_overall_same_sign_share":
            "regime_alpha_ic_overall_same_sign_share",

        "alpha_ic_time_same_sign_share":
            "regime_alpha_ic_time_same_sign_share",

        "alpha_ic_trend_same_sign_share":
            "regime_alpha_ic_trend_same_sign_share",

        "alpha_ic_vol_same_sign_share":
            "regime_alpha_ic_vol_same_sign_share",

        "alpha_ic_joint_same_sign_share":
            "regime_alpha_ic_joint_same_sign_share",

        "alpha_spread_overall_same_sign_share":
            "regime_alpha_spread_overall_same_sign_share",

        "ivol_ic_overall_same_sign_share":
            "regime_ivol_overall_same_sign_share",

        "ivol_ic_time_same_sign_share":
            "regime_ivol_time_same_sign_share",

        "ivol_ic_trend_same_sign_share":
            "regime_ivol_trend_same_sign_share",

        "ivol_ic_vol_same_sign_share":
            "regime_ivol_vol_same_sign_share",

        "ivol_ic_joint_same_sign_share":
            "regime_ivol_joint_same_sign_share",
    }

    keep = [

        "factor_name",
        "window",

    ] + list(
        rename.keys()
    )

    return (
        df[
            keep
        ]
        .rename(
            columns=rename
        )
    )


# ============================================================
# 8. Step-2 Traditional-Control Evidence
# ============================================================

def load_step2_retention():

    df = pd.read_csv(
        STEP2_RETENTION_PATH
    )

    require_columns(
        df,
        [
            "factor_name",
            "window",

            "industry_size_common_alpha_ic",
            "full_control_alpha_ic",
            "full_vs_ind_size_alpha_same_sign",
            "full_vs_ind_size_alpha_abs_retention",

            "industry_size_common_spread",
            "full_control_spread",
            "full_control_spread_bps",
            "full_vs_ind_size_spread_same_sign",
            "full_vs_ind_size_spread_abs_retention",

            "industry_size_common_mean_risk_ic",
            "full_control_mean_risk_ic",

            "industry_size_common_ivol_ic",
            "full_control_ivol_ic",
            "full_vs_ind_size_ivol_same_sign",
            "full_vs_ind_size_ivol_abs_retention",
        ],
        "traditional_control_signal_retention",
    )

    df[
        "window"
    ] = pd.to_numeric(
        df[
            "window"
        ]
    ).astype(int)

    if (
        df[
            [
                "factor_name",
                "window",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate Step-2 retention rows."
        )

    keep = [

        "factor_name",
        "window",

        "industry_size_common_alpha_ic",
        "full_control_alpha_ic",
        "full_vs_ind_size_alpha_same_sign",
        "full_vs_ind_size_alpha_abs_retention",

        "industry_size_common_spread",
        "full_control_spread",
        "full_control_spread_bps",
        "full_vs_ind_size_spread_same_sign",
        "full_vs_ind_size_spread_abs_retention",

        "industry_size_common_mean_risk_ic",
        "full_control_mean_risk_ic",

        "industry_size_common_ivol_ic",
        "full_control_ivol_ic",
        "full_vs_ind_size_ivol_same_sign",
        "full_vs_ind_size_ivol_abs_retention",
    ]

    return df[
        keep
    ].copy()


def load_step2_alpha():

    df = pd.read_csv(
        STEP2_ALPHA_PATH
    )

    require_columns(
        df,
        [
            "factor_name",
            "method",
            "window",
            "target_name",
            "mean_alpha_ic",
            "hac_t_stat",
        ],
        "controlled_alpha_rank_ic_summary",
    )

    df[
        "window"
    ] = pd.to_numeric(
        df[
            "window"
        ]
    ).astype(int)

    df = df[
        (
            df[
                "method"
            ]
            ==
            FULL_CONTROL_METHOD
        )
        &
        (
            df[
                "target_name"
            ]
            ==
            PRIMARY_TARGET
        )
    ].copy()

    return (
        df[
            [
                "factor_name",
                "window",
                "mean_alpha_ic",
                "hac_t_stat",
            ]
        ]
        .rename(
            columns={
                "mean_alpha_ic":
                    "full_control_alpha_ic_from_summary",

                "hac_t_stat":
                    "full_control_alpha_ic_hac_t",
            }
        )
    )


def load_step2_portfolio():

    df = pd.read_csv(
        STEP2_PORTFOLIO_PATH
    )

    require_columns(
        df,
        [
            "factor_name",
            "method",
            "window",
            "target_name",
            "mean_q5_minus_q1_bps",
            "hac_t_stat",
        ],
        "controlled_alpha_portfolio_summary",
    )

    df[
        "window"
    ] = pd.to_numeric(
        df[
            "window"
        ]
    ).astype(int)

    df = df[
        (
            df[
                "method"
            ]
            ==
            FULL_CONTROL_METHOD
        )
        &
        (
            df[
                "target_name"
            ]
            ==
            PRIMARY_TARGET
        )
    ].copy()

    return (
        df[
            [
                "factor_name",
                "window",
                "mean_q5_minus_q1_bps",
                "hac_t_stat",
            ]
        ]
        .rename(
            columns={
                "mean_q5_minus_q1_bps":
                    "full_control_spread_summary_bps",

                "hac_t_stat":
                    "full_control_spread_hac_t",
            }
        )
    )


# ============================================================
# 9. Step-2 Risk Evidence
# ============================================================

def load_step2_risk():

    df = pd.read_csv(
        STEP2_RISK_PATH
    )

    require_columns(
        df,
        [
            "factor_name",
            "method",
            "window",
            "target_name",
            "mean_risk_ic",
            "hac_t_stat",
        ],
        "controlled_risk_rank_ic_summary",
    )

    df[
        "window"
    ] = pd.to_numeric(
        df[
            "window"
        ]
    ).astype(int)

    df = df[
        df[
            "method"
        ]
        ==
        FULL_CONTROL_METHOD
    ].copy()

    rows = []

    group_cols = [
        "factor_name",
        "window",
    ]

    for keys, group in (
        df.groupby(
            group_cols,
            sort=False,
        )
    ):

        (
            factor_name,
            window,
        ) = keys

        row = {

            "factor_name":
                factor_name,

            "window":
                int(
                    window
                ),
        }

        for target, short_name in (
            RISK_TARGETS.items()
        ):

            temp = group[
                group[
                    "target_name"
                ]
                ==
                target
            ]

            if len(temp) == 1:

                row[
                    f"{short_name}_ic"
                ] = float(
                    temp[
                        "mean_risk_ic"
                    ]
                    .iloc[0]
                )

                row[
                    f"{short_name}_hac_t"
                ] = float(
                    temp[
                        "hac_t_stat"
                    ]
                    .iloc[0]
                )

            else:

                row[
                    f"{short_name}_ic"
                ] = np.nan

                row[
                    f"{short_name}_hac_t"
                ] = np.nan

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 10. Step-3 Fama-MacBeth Evidence
# ============================================================

def load_fmb_evidence():

    df = pd.read_csv(
        STEP3_MODEL_PATH
    )

    require_columns(
        df,
        [
            "factor_name",
            "target_name",
            "window",

            "m1_beta_bps_per_1sd",
            "m2_beta_bps_per_1sd",
            "m3_beta_bps_per_1sd",

            "m3_hac_t_stat",
            "m3_mean_partial_r2",

            "m3_vs_m2_same_sign",
            "m3_vs_m2_abs_retention",
        ],
        "fmb_model_comparison",
    )

    df[
        "window"
    ] = pd.to_numeric(
        df[
            "window"
        ]
    ).astype(int)

    df = df[
        df[
            "target_name"
        ]
        ==
        PRIMARY_TARGET
    ].copy()

    if (
        df[
            [
                "factor_name",
                "window",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate FMB factor-window rows."
        )

    keep = [

        "factor_name",
        "window",

        "m1_beta_bps_per_1sd",
        "m2_beta_bps_per_1sd",
        "m3_beta_bps_per_1sd",

        "m3_hac_t_stat",
        "m3_mean_partial_r2",

        "m3_vs_m2_same_sign",
        "m3_vs_m2_abs_retention",
    ]

    return df[
        keep
    ].copy()


# ============================================================
# 11. Step-4 Turnover / Cost Evidence
# ============================================================

def load_turnover_evidence():

    df = pd.read_csv(
        STEP4_ECONOMIC_PATH
    )

    require_columns(
        df,
        [
            "factor_name",
            "method",
            "window",

            "mean_gross_q5_minus_q1_bps",

            "mean_q1_drift_turnover",
            "mean_q5_drift_turnover",

            "mean_roundtrip_traded_notional_ratio",

            "descriptive_break_even_one_way_cost_bps",
        ],
        "turnover_economic_summary",
    )

    df[
        "window"
    ] = pd.to_numeric(
        df[
            "window"
        ]
    ).astype(int)

    df = df[
        df[
            "method"
        ]
        ==
        FULL_CONTROL_METHOD
    ].copy()

    keep = [

        "factor_name",
        "window",

        "mean_gross_q5_minus_q1_bps",

        "mean_q1_drift_turnover",
        "mean_q5_drift_turnover",

        "mean_roundtrip_traded_notional_ratio",

        "descriptive_break_even_one_way_cost_bps",
    ]

    return df[
        keep
    ].copy()


def load_cost_evidence():

    df = pd.read_csv(
        STEP4_COST_PATH
    )

    require_columns(
        df,
        [
            "factor_name",
            "method",
            "window",
            "one_way_cost_bps",

            "mean_gross_spread_bps",

            "mean_transaction_cost_drag_bps",

            "mean_canonical_net_spread_bps",

            "net_hac_t_stat",
        ],
        "transaction_cost_summary",
    )

    df[
        "window"
    ] = pd.to_numeric(
        df[
            "window"
        ]
    ).astype(int)

    df[
        "one_way_cost_bps"
    ] = pd.to_numeric(
        df[
            "one_way_cost_bps"
        ]
    ).astype(int)

    df = df[
        df[
            "method"
        ]
        ==
        FULL_CONTROL_METHOD
    ].copy()

    rows = []

    for keys, group in (
        df.groupby(
            [
                "factor_name",
                "window",
            ],
            sort=False,
        )
    ):

        (
            factor_name,
            window,
        ) = keys

        row = {

            "factor_name":
                factor_name,

            "window":
                int(
                    window
                ),
        }

        for cost in EXPECTED_COST_BPS:

            temp = group[
                group[
                    "one_way_cost_bps"
                ]
                ==
                cost
            ]

            if len(temp) != 1:

                row[
                    f"cost_drag_{cost}bp"
                ] = np.nan

                row[
                    f"canonical_net_spread_{cost}bp"
                ] = np.nan

                row[
                    f"abs_gross_minus_cost_{cost}bp"
                ] = np.nan

                row[
                    f"gross_magnitude_exceeds_cost_{cost}bp"
                ] = False

                continue

            gross = float(
                temp[
                    "mean_gross_spread_bps"
                ]
                .iloc[0]
            )

            drag = float(
                temp[
                    "mean_transaction_cost_drag_bps"
                ]
                .iloc[0]
            )

            canonical_net = float(
                temp[
                    "mean_canonical_net_spread_bps"
                ]
                .iloc[0]
            )

            magnitude_remaining = (

                abs(
                    gross
                )

                -

                drag
            )

            row[
                f"cost_drag_{cost}bp"
            ] = drag

            row[
                f"canonical_net_spread_{cost}bp"
            ] = canonical_net

            # ------------------------------------------------
            # This is NOT a strategy return.
            #
            # It is only:
            #
            # |gross Q5-Q1 magnitude| - cost drag.
            #
            # Useful for negative factors without reversing
            # the formal Q5-Q1 orientation.
            # ------------------------------------------------

            row[
                f"abs_gross_minus_cost_{cost}bp"
            ] = magnitude_remaining

            row[
                f"gross_magnitude_exceeds_cost_{cost}bp"
            ] = bool(
                magnitude_remaining
                >
                0
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 12. Build 9 x 3 Detailed Robustness Matrix
# ============================================================

def build_window_matrix(
    core,
):

    matrix = build_base_grid(
        core
    )

    sources = [

        load_regime_evidence(),

        load_step2_retention(),

        load_step2_alpha(),

        load_step2_portfolio(),

        load_step2_risk(),

        load_fmb_evidence(),

        load_turnover_evidence(),

        load_cost_evidence(),
    ]

    for source in sources:

        matrix = matrix.merge(

            source,

            on=[
                "factor_name",
                "window",
            ],

            how="left",

            validate="one_to_one",
        )

    # ========================================================
    # Internal QA:
    #
    # Step-2 retention and Step-2 alpha summary should agree.
    # ========================================================

    matrix[
        "step2_alpha_ic_internal_difference"
    ] = (

        matrix[
            "full_control_alpha_ic"
        ]

        -

        matrix[
            "full_control_alpha_ic_from_summary"
        ]
    )

    # --------------------------------------------------------
    # Nominal evidence markers
    #
    # They are diagnostics only.
    # No multiple-testing adjustment is implied.
    # --------------------------------------------------------

    matrix[
        "alpha_ic_nominal_abs_t_ge_1p96"
    ] = (

        matrix[
            "full_control_alpha_ic_hac_t"
        ]
        .abs()
        >=
        NOMINAL_T_REFERENCE
    )

    matrix[
        "alpha_spread_nominal_abs_t_ge_1p96"
    ] = (

        matrix[
            "full_control_spread_hac_t"
        ]
        .abs()
        >=
        NOMINAL_T_REFERENCE
    )

    matrix[
        "fmb_m3_nominal_abs_t_ge_1p96"
    ] = (

        matrix[
            "m3_hac_t_stat"
        ]
        .abs()
        >=
        NOMINAL_T_REFERENCE
    )

    matrix[
        "ivol_nominal_abs_t_ge_1p96"
    ] = (

        matrix[
            "ivol_hac_t"
        ]
        .abs()
        >=
        NOMINAL_T_REFERENCE
    )

    # --------------------------------------------------------
    # Cost benchmark
    # --------------------------------------------------------

    matrix[
        "cost_10bp_magnitude_survives"
    ] = matrix[
        "gross_magnitude_exceeds_cost_10bp"
    ]

    # --------------------------------------------------------
    # No selection flags
    # --------------------------------------------------------

    matrix[
        "factor_selected"
    ] = False

    matrix[
        "factor_removed"
    ] = False

    matrix[
        "factor_sign_flipped"
    ] = False

    matrix[
        "window_selected"
    ] = False

    return (
        matrix.sort_values(
            [
                "factor_order",
                "window",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 13. Factor-Level Aggregation
# ============================================================

def classify_evidence_role(
    alpha_evidence,
    risk_evidence,
):

    if (
        alpha_evidence
        and
        risk_evidence
    ):

        return (
            "RETURN_AND_RISK_EVIDENCE"
        )

    if (
        risk_evidence
        and
        not alpha_evidence
    ):

        return (
            "RISK_DOMINANT_EVIDENCE"
        )

    if (
        alpha_evidence
        and
        not risk_evidence
    ):

        return (
            "RETURN_DOMINANT_EVIDENCE"
        )

    return (
        "WEAK_OR_MIXED_EVIDENCE"
    )


def classify_cost_profile(
    survival_count_10bp,
    mean_break_even,
):

    # --------------------------------------------------------
    # Descriptive only.
    # Does not rank/select factors.
    # --------------------------------------------------------

    if (
        survival_count_10bp
        ==
        3
        and
        np.isfinite(
            mean_break_even
        )
        and
        mean_break_even
        >=
        20
    ):

        return (
            "LOWER_COST_SENSITIVITY"
        )

    if (
        survival_count_10bp
        >=
        1
        and
        np.isfinite(
            mean_break_even
        )
        and
        mean_break_even
        >=
        10
    ):

        return (
            "MODERATE_COST_SENSITIVITY"
        )

    return (
        "HIGH_COST_SENSITIVITY"
    )


def build_factor_matrix(
    window_matrix,
):

    rows = []

    for (
        factor_order,
        factor_name,
    ), group in (
        window_matrix.groupby(
            [
                "factor_order",
                "factor_name",
            ],
            sort=False,
        )
    ):

        group = (
            group.sort_values(
                "window"
            )
        )

        # ====================================================
        # Alpha
        # ====================================================

        alpha_ic = (
            group[
                "full_control_alpha_ic"
            ]
            .to_numpy(
                dtype=float
            )
        )

        alpha_ic_mean = safe_mean(
            alpha_ic
        )

        alpha_ic_same_sign = (
            same_nonzero_sign(
                alpha_ic
            )
        )

        alpha_ic_nominal_count = (
            count_abs_t_ge(
                group[
                    "full_control_alpha_ic_hac_t"
                ]
            )
        )

        alpha_retention_mean = safe_mean(
            group[
                "full_vs_ind_size_alpha_abs_retention"
            ]
        )

        # ====================================================
        # Portfolio
        # ====================================================

        spread = (
            group[
                "full_control_spread_bps"
            ]
            .to_numpy(
                dtype=float
            )
        )

        spread_mean = safe_mean(
            spread
        )

        spread_same_sign = (
            same_nonzero_sign(
                spread
            )
        )

        spread_nominal_count = (
            count_abs_t_ge(
                group[
                    "full_control_spread_hac_t"
                ]
            )
        )

        # ====================================================
        # FMB
        # ====================================================

        fmb_beta = (
            group[
                "m3_beta_bps_per_1sd"
            ]
            .to_numpy(
                dtype=float
            )
        )

        fmb_mean = safe_mean(
            fmb_beta
        )

        fmb_same_sign = (
            same_nonzero_sign(
                fmb_beta
            )
        )

        fmb_nominal_count = (
            count_abs_t_ge(
                group[
                    "m3_hac_t_stat"
                ]
            )
        )

        fmb_partial_r2_mean = (
            safe_mean(
                group[
                    "m3_mean_partial_r2"
                ]
            )
        )

        # ====================================================
        # Regime
        # ====================================================

        regime_alpha_min = safe_min(
            group[
                "regime_alpha_ic_overall_same_sign_share"
            ]
        )

        regime_ivol_min = safe_min(
            group[
                "regime_ivol_overall_same_sign_share"
            ]
        )

        regime_alpha_strict = bool(

            np.isfinite(
                regime_alpha_min
            )

            and

            np.isclose(
                regime_alpha_min,
                1.0,
            )
        )

        regime_ivol_strict = bool(

            np.isfinite(
                regime_ivol_min
            )

            and

            np.isclose(
                regime_ivol_min,
                1.0,
            )
        )

        # ====================================================
        # Risk
        # ====================================================

        ivol = (
            group[
                "ivol_ic"
            ]
            .to_numpy(
                dtype=float
            )
        )

        ivol_mean = safe_mean(
            ivol
        )

        ivol_same_sign = (
            same_nonzero_sign(
                ivol
            )
        )

        ivol_nominal_count = (
            count_abs_t_ge(
                group[
                    "ivol_hac_t"
                ]
            )
        )

        stable_risk_target_count = 0

        risk_target_signs = {}

        for short_name in (
            RISK_TARGETS.values()
        ):

            values = (
                group[
                    f"{short_name}_ic"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            stable = (
                same_nonzero_sign(
                    values
                )
            )

            if stable:

                stable_risk_target_count += 1

            risk_target_signs[
                f"{short_name}_same_sign_all_windows"
            ] = stable

            risk_target_signs[
                f"{short_name}_mean_ic"
            ] = safe_mean(
                values
            )

        # ====================================================
        # Economic implementation
        # ====================================================

        mean_turnover = safe_mean(
            group[
                "mean_roundtrip_traded_notional_ratio"
            ]
        )

        mean_break_even = safe_mean(
            group[
                "descriptive_break_even_one_way_cost_bps"
            ]
        )

        cost_survival_5 = int(
            group[
                "gross_magnitude_exceeds_cost_5bp"
            ]
            .fillna(False)
            .sum()
        )

        cost_survival_10 = int(
            group[
                "gross_magnitude_exceeds_cost_10bp"
            ]
            .fillna(False)
            .sum()
        )

        cost_survival_20 = int(
            group[
                "gross_magnitude_exceeds_cost_20bp"
            ]
            .fillna(False)
            .sum()
        )

        cost_survival_30 = int(
            group[
                "gross_magnitude_exceeds_cost_30bp"
            ]
            .fillna(False)
            .sum()
        )

        # ====================================================
        # Descriptive evidence patterns
        #
        # IMPORTANT:
        #
        # This is not a factor-selection rule.
        # ====================================================

        alpha_evidence = bool(

            np.isfinite(
                alpha_ic_mean
            )

            and

            abs(
                alpha_ic_mean
            )
            >=
            ALPHA_IC_REFERENCE

            and

            alpha_ic_same_sign

            and

            fmb_same_sign

            and

            (
                alpha_ic_nominal_count
                >=
                1

                or

                fmb_nominal_count
                >=
                1
            )
        )

        risk_evidence = bool(

            np.isfinite(
                ivol_mean
            )

            and

            abs(
                ivol_mean
            )
            >=
            RISK_IC_REFERENCE

            and

            ivol_same_sign

            and

            ivol_nominal_count
            >=
            1
        )

        evidence_role = (
            classify_evidence_role(

                alpha_evidence,

                risk_evidence,
            )
        )

        cost_profile = (
            classify_cost_profile(

                cost_survival_10,

                mean_break_even,
            )
        )

        row = {

            "factor_order":
                int(
                    factor_order
                ),

            "factor_name":
                factor_name,

            # ================================================
            # Alpha
            # ================================================

            "full_control_mean_alpha_ic":
                alpha_ic_mean,

            "full_control_alpha_ic_same_sign_all_windows":
                alpha_ic_same_sign,

            "full_control_alpha_ic_nominal_t_count":
                alpha_ic_nominal_count,

            "mean_alpha_ic_abs_retention_vs_industry_size":
                alpha_retention_mean,

            "full_control_mean_spread_bps":
                spread_mean,

            "full_control_spread_same_sign_all_windows":
                spread_same_sign,

            "full_control_spread_nominal_t_count":
                spread_nominal_count,

            # ================================================
            # FMB
            # ================================================

            "m3_mean_beta_bps_per_1sd":
                fmb_mean,

            "m3_beta_same_sign_all_windows":
                fmb_same_sign,

            "m3_nominal_t_count":
                fmb_nominal_count,

            "m3_mean_partial_r2":
                fmb_partial_r2_mean,

            # ================================================
            # Regime
            # ================================================

            "regime_alpha_min_same_sign_share":
                regime_alpha_min,

            "regime_alpha_strict_all_windows":
                regime_alpha_strict,

            "regime_ivol_min_same_sign_share":
                regime_ivol_min,

            "regime_ivol_strict_all_windows":
                regime_ivol_strict,

            # ================================================
            # Risk
            # ================================================

            "full_control_mean_ivol_ic":
                ivol_mean,

            "full_control_ivol_same_sign_all_windows":
                ivol_same_sign,

            "full_control_ivol_nominal_t_count":
                ivol_nominal_count,

            "stable_risk_target_count_out_of_4":
                stable_risk_target_count,

            # ================================================
            # Cost
            # ================================================

            "mean_roundtrip_traded_notional_ratio":
                mean_turnover,

            "mean_descriptive_break_even_cost_bps":
                mean_break_even,

            "cost_magnitude_survival_windows_5bp":
                cost_survival_5,

            "cost_magnitude_survival_windows_10bp":
                cost_survival_10,

            "cost_magnitude_survival_windows_20bp":
                cost_survival_20,

            "cost_magnitude_survival_windows_30bp":
                cost_survival_30,

            # ================================================
            # Descriptive labels
            # ================================================

            "alpha_evidence_flag":
                alpha_evidence,

            "risk_evidence_flag":
                risk_evidence,

            "evidence_role":
                evidence_role,

            "cost_profile":
                cost_profile,

            # ================================================
            # Research discipline
            # ================================================

            "factor_selected":
                False,

            "factor_removed":
                False,

            "factor_sign_flipped":
                False,

            "window_selected":
                False,
        }

        row.update(
            risk_target_signs
        )

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "factor_order"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 14. Compact Role Summary
# ============================================================

def build_role_summary(
    factor_matrix,
):

    columns = [

        "factor_order",

        "factor_name",

        "evidence_role",

        "cost_profile",

        "full_control_mean_alpha_ic",

        "m3_mean_beta_bps_per_1sd",

        "full_control_mean_spread_bps",

        "full_control_mean_ivol_ic",

        "stable_risk_target_count_out_of_4",

        "regime_alpha_strict_all_windows",

        "regime_ivol_strict_all_windows",

        "cost_magnitude_survival_windows_10bp",

        "mean_roundtrip_traded_notional_ratio",

        "mean_descriptive_break_even_cost_bps",
    ]

    return factor_matrix[
        columns
    ].copy()


# ============================================================
# 15. Formal QA
# ============================================================

def formal_qa(
    window_matrix,
    factor_matrix,
    upstream_metadata,
):

    qa = {}

    qa[
        "expected_window_rows"
    ] = int(
        EXPECTED_FACTOR_COUNT
        *
        len(
            EXPECTED_WINDOWS
        )
    )

    qa[
        "actual_window_rows"
    ] = int(
        len(
            window_matrix
        )
    )

    qa[
        "window_row_count_ok"
    ] = bool(
        len(
            window_matrix
        )
        ==
        qa[
            "expected_window_rows"
        ]
    )

    qa[
        "factor_row_count"
    ] = int(
        len(
            factor_matrix
        )
    )

    qa[
        "factor_row_count_ok"
    ] = bool(
        len(
            factor_matrix
        )
        ==
        EXPECTED_FACTOR_COUNT
    )

    qa[
        "factor_window_keys_unique"
    ] = bool(

        not window_matrix[
            [
                "factor_name",
                "window",
            ]
        ]
        .duplicated()
        .any()
    )

    qa[
        "factor_keys_unique"
    ] = bool(

        not factor_matrix[
            "factor_name"
        ]
        .duplicated()
        .any()
    )

    observed_windows = sorted(
        window_matrix[
            "window"
        ]
        .unique()
        .astype(int)
        .tolist()
    )

    qa[
        "observed_windows"
    ] = observed_windows

    qa[
        "window_set_ok"
    ] = bool(
        observed_windows
        ==
        EXPECTED_WINDOWS
    )

    # --------------------------------------------------------
    # Step-2 internal consistency
    # --------------------------------------------------------

    alpha_diff = (
        window_matrix[
            "step2_alpha_ic_internal_difference"
        ]
        .abs()
        .dropna()
    )

    qa[
        "max_abs_step2_alpha_ic_internal_difference"
    ] = (
        float(
            alpha_diff.max()
        )
        if
        len(
            alpha_diff
        )
        >
        0
        else
        np.nan
    )

    qa[
        "step2_alpha_internal_consistency_ok"
    ] = bool(

        len(
            alpha_diff
        )
        >
        0

        and

        alpha_diff.max()
        <
        1e-12
    )

    # --------------------------------------------------------
    # Missing evidence
    # --------------------------------------------------------

    key_evidence_columns = [

        "full_control_alpha_ic",

        "full_control_spread_bps",

        "ivol_ic",

        "m3_beta_bps_per_1sd",

        "mean_roundtrip_traded_notional_ratio",

        "descriptive_break_even_one_way_cost_bps",
    ]

    for column in key_evidence_columns:

        qa[
            f"missing_{column}_count"
        ] = int(
            window_matrix[
                column
            ]
            .isna()
            .sum()
        )

    # --------------------------------------------------------
    # Parent QAs
    # --------------------------------------------------------

    qa[
        "all_parent_qa_pass"
    ] = bool(
        all(

            upstream_metadata[
                step
            ]
            .get(
                "formal_qa",
                {}
            )
            .get(
                "all_formal_qa_pass"
            )
            is True

            for step in [
                "step1",
                "step2",
                "step3",
                "step4",
            ]
        )
    )

    # --------------------------------------------------------
    # No selection / no sign flip
    # --------------------------------------------------------

    qa[
        "factor_selection_performed"
    ] = False

    qa[
        "factor_removal_performed"
    ] = False

    qa[
        "sign_flip_performed"
    ] = False

    qa[
        "window_selection_performed"
    ] = False

    required = [

        "window_row_count_ok",

        "factor_row_count_ok",

        "factor_window_keys_unique",

        "factor_keys_unique",

        "window_set_ok",

        "step2_alpha_internal_consistency_ok",

        "all_parent_qa_pass",
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
# 16. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 7 - Step 5"
    )

    print(
        "Final Factor Robustness Matrix"
    )

    print("=" * 80)

    # ========================================================
    # Parent QA
    # ========================================================

    upstream_metadata = (
        validate_upstream()
    )

    core = load_core()

    # ========================================================
    # Freeze Step-5 design
    # ========================================================

    design_payload = {

        "research_day":
            7,

        "step":
            "Step5_Final_Factor_Robustness_Matrix",

        "parent_design_hashes":
            {

                "step1":
                    upstream_metadata[
                        "step1"
                    ].get(
                        "day7_step1_design_hash"
                    ),

                "step2":
                    upstream_metadata[
                        "step2"
                    ].get(
                        "day7_step2_design_hash"
                    ),

                "step3":
                    upstream_metadata[
                        "step3"
                    ].get(
                        "day7_step3_design_hash"
                    ),

                "step4":
                    upstream_metadata[
                        "step4"
                    ].get(
                        "day7_step4_design_hash"
                    ),
            },

        "factor_count":
            EXPECTED_FACTOR_COUNT,

        "windows":
            EXPECTED_WINDOWS,

        "primary_alpha_target":
            PRIMARY_TARGET,

        "regime_specification":
            REGIME_METHOD,

        "traditional_control_specification":
            FULL_CONTROL_METHOD,

        "risk_targets":
            list(
                RISK_TARGETS.keys()
            ),

        "transaction_cost_scenarios_bps":
            EXPECTED_COST_BPS,

        "reference_transaction_cost_bps":
            REFERENCE_COST_BPS,

        "classification_rules":
            {

                "alpha_ic_reference":
                    ALPHA_IC_REFERENCE,

                "risk_ic_reference":
                    RISK_IC_REFERENCE,

                "nominal_t_reference":
                    NOMINAL_T_REFERENCE,

                "purpose":
                    (
                        "Descriptive evidence classification "
                        "only; not factor selection."
                    ),
            },

        "factor_ranking":
            False,

        "factor_selection":
            False,

        "factor_removal":
            False,

        "factor_sign_flip":
            False,

        "window_selection":
            False,
    }

    design = {

        **design_payload,

        "day7_step5_design_hash":
            canonical_hash(
                design_payload
            ),
    }

    print()
    print(
        "Day-7 Step-5 design hash:"
    )

    print(
        design[
            "day7_step5_design_hash"
        ]
    )

    # ========================================================
    # Detailed 9 x 3 matrix
    # ========================================================

    print()
    print(
        "[1] Build factor x window robustness matrix"
    )

    window_matrix = (
        build_window_matrix(
            core
        )
    )

    save_csv_atomic(
        window_matrix,
        WINDOW_MATRIX_PATH,
    )

    # ========================================================
    # Factor-level matrix
    # ========================================================

    print()
    print(
        "[2] Aggregate evidence across W60/W120/W252"
    )

    factor_matrix = (
        build_factor_matrix(
            window_matrix
        )
    )

    save_csv_atomic(
        factor_matrix,
        FACTOR_MATRIX_PATH,
    )

    # ========================================================
    # Compact role table
    # ========================================================

    print()
    print(
        "[3] Build compact evidence-role summary"
    )

    role_summary = (
        build_role_summary(
            factor_matrix
        )
    )

    save_csv_atomic(
        role_summary,
        ROLE_SUMMARY_PATH,
    )

    # ========================================================
    # QA
    # ========================================================

    print()
    print(
        "[4] Formal QA"
    )

    qa = formal_qa(

        window_matrix,

        factor_matrix,

        upstream_metadata,
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
            "Day-7 Step-5 formal QA failed.\n"
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

        "interpretation_rules":
            {

                "RETURN_AND_RISK_EVIDENCE":
                    (
                        "Both conditional return evidence "
                        "and future-IVOL evidence satisfy "
                        "the pre-specified descriptive rules."
                    ),

                "RISK_DOMINANT_EVIDENCE":
                    (
                        "Future-IVOL evidence is stronger "
                        "than conditional return evidence."
                    ),

                "RETURN_DOMINANT_EVIDENCE":
                    (
                        "Conditional return evidence is present "
                        "while IVOL evidence is comparatively weak."
                    ),

                "WEAK_OR_MIXED_EVIDENCE":
                    (
                        "Evidence is weak, unstable, or mixed "
                        "across the pre-specified dimensions."
                    ),

                "LOWER_COST_SENSITIVITY":
                    (
                        "Gross Q5-Q1 magnitude exceeds a 10bp "
                        "one-way cost drag in all three windows "
                        "and mean descriptive break-even cost "
                        "is at least 20bp."
                    ),

                "MODERATE_COST_SENSITIVITY":
                    (
                        "At least one window retains gross "
                        "magnitude beyond the 10bp cost burden "
                        "and mean break-even cost is at least 10bp."
                    ),

                "HIGH_COST_SENSITIVITY":
                    (
                        "The gross economic magnitude is generally "
                        "small relative to turnover-based cost drag."
                    ),
            },

        "important_caveats":
            [

                (
                    "Nominal HAC |t| >= 1.96 markers are "
                    "descriptive and are not adjusted for "
                    "multiple testing."
                ),

                (
                    "The robustness matrix does not rank "
                    "or select factors."
                ),

                (
                    "The matrix does not choose an optimal "
                    "network window."
                ),

                (
                    "No factor direction is flipped after "
                    "observing outcomes."
                ),

                (
                    "For negative Q5-Q1 factors, "
                    "|gross spread| minus cost drag is only "
                    "an economic-magnitude diagnostic and "
                    "must not be interpreted as the realized "
                    "return of an ex-ante Q1-Q5 strategy."
                ),

                (
                    "Step-1 regime robustness is based on "
                    "Industry+Size neutralization, while "
                    "Step-2/3/4 final conditional evidence "
                    "uses the full traditional-control design."
                ),
            ],

        "outputs":
            {

                "window_level_matrix":
                    str(
                        WINDOW_MATRIX_PATH
                    ),

                "factor_level_matrix":
                    str(
                        FACTOR_MATRIX_PATH
                    ),

                "role_summary":
                    str(
                        ROLE_SUMMARY_PATH
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
        "Final Factor Robustness Matrix"
    )

    print("=" * 80)

    display_columns = [

        "factor_name",

        "evidence_role",

        "cost_profile",

        "full_control_mean_alpha_ic",

        "m3_mean_beta_bps_per_1sd",

        "full_control_mean_spread_bps",

        "full_control_mean_ivol_ic",

        "stable_risk_target_count_out_of_4",

        "regime_alpha_strict_all_windows",

        "regime_ivol_strict_all_windows",

        "cost_magnitude_survival_windows_10bp",
    ]

    print(
        factor_matrix[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 80)

    print(
        "Formal QA"
    )

    print("=" * 80)

    print(
        f"Window rows: "
        f"{qa['actual_window_rows']} "
        f"/ {qa['expected_window_rows']}"
    )

    print(
        f"Factor rows: "
        f"{qa['factor_row_count']}"
    )

    print(
        f"Max Step-2 IC internal difference: "
        f"{qa['max_abs_step2_alpha_ic_internal_difference']}"
    )

    print(
        f"Parent QA pass: "
        f"{qa['all_parent_qa_pass']}"
    )

    print(
        f"Final QA pass: "
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
        "Day 7 Step 5 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()