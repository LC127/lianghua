from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd


# ============================================================
# 0. Project configuration
# ============================================================

ROOT = Path(
    r"D:\M1_StockNetwork"
)

OUTPUT_ROOT = (
    ROOT
    / "output"
)


# ------------------------------------------------------------
# IMPORTANT:
#
# Previously frozen convention:
#
# Research Day 8 physical outputs are stored under:
#
#     D:\M1_StockNetwork\output\M1_day7
#
# If your ENTIRE Day-8 chain has already been intentionally
# migrated to M1_day8, change ONLY this line.
# ------------------------------------------------------------

DAY8_STORAGE_FOLDER = "M1_day8"

DAY8_ROOT = (
    OUTPUT_ROOT
    / DAY8_STORAGE_FOLDER
)


# ============================================================
# 1. Import frozen Step-4 mathematics
# ============================================================

if str(ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT),
    )


try:

    from capacity_limit_estimation import (
        coverage_from_breakpoints,
        solve_capacity_limit,
    )

except Exception as exc:

    raise RuntimeError(
        "Cannot import the frozen Step-4 module:\n\n"
        "D:\\M1_StockNetwork\\capacity_limit_estimation.py\n\n"
        "Step 5 intentionally reuses the validated Step-4 "
        "capacity mathematics. Please confirm that the Step-4 "
        "script exists and can be imported.\n\n"
        f"Original error:\n{exc}"
    ) from exc


# ============================================================
# 2. Frozen Step-3 inputs
# ============================================================

STEP3_DIR = (
    DAY8_ROOT
    / "03_stage3_participation_capacity"
)

STEP3_PANEL_PATH = (
    STEP3_DIR
    / "stock_participation_base_panel.parquet"
)

STEP3_QA_PATH = (
    STEP3_DIR
    / "participation_capacity_qa.csv"
)


# ============================================================
# 3. Frozen Step-4 inputs
# ============================================================

STEP4_DIR = (
    DAY8_ROOT
    / "04_stage4_capacity_limits"
)

STEP4_CAPACITY_PATH = (
    STEP4_DIR
    / "capacity_limit_factor_window.csv"
)

STEP4_QA_PATH = (
    STEP4_DIR
    / "capacity_limit_qa.csv"
)


# ============================================================
# 4. Step-5 outputs
# ============================================================

OUTPUT_DIR = (
    DAY8_ROOT
    / "05_stage5_capacity_robustness_matrix"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ROBUSTNESS_MATRIX_PATH = (
    OUTPUT_DIR
    / "capacity_robustness_matrix.csv"
)

ROBUSTNESS_MATRIX_WIDE_PATH = (
    OUTPUT_DIR
    / "capacity_robustness_matrix_wide.csv"
)

SCENARIO_SUMMARY_PATH = (
    OUTPUT_DIR
    / "capacity_robustness_scenario_summary.csv"
)

FAMILY_SUMMARY_PATH = (
    OUTPUT_DIR
    / "capacity_robustness_family_summary.csv"
)

PRIMARY_CROSSCHECK_PATH = (
    OUTPUT_DIR
    / "capacity_robustness_primary_crosscheck.csv"
)

ZERO_CAPACITY_PATH = (
    OUTPUT_DIR
    / "capacity_robustness_zero_capacity_detail.csv"
)

QA_PATH = (
    OUTPUT_DIR
    / "capacity_robustness_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day8_step5_metadata.json"
)


# ============================================================
# 5. Frozen robustness design
# ============================================================

# ------------------------------------------------------------
# Step-4 PRIMARY specification:
#
# ADV20
# 5% ADV participation
# 95% mean executable share
# 1-day execution
#
# Step 5 does NOT replace this primary definition.
# It only evaluates pre-specified sensitivity dimensions.
# ------------------------------------------------------------

PRIMARY_LIQUIDITY_BASIS = "ADV20"

PRIMARY_PARTICIPATION_CAP = 0.05

PRIMARY_TARGET_EXECUTABLE_SHARE = 0.95

EXECUTION_DAYS = 1


# ------------------------------------------------------------
# Robustness dimensions
#
# Full factorial:
#
# 2 liquidity definitions
# × 3 participation limits
# × 3 executable-share requirements
#
# = 18 scenarios
#
# No scenario is selected based on results.
# ------------------------------------------------------------

LIQUIDITY_BASES = [
    "ADV20",
    "ADV60",
]

PARTICIPATION_CAPS = [
    0.01,
    0.05,
    0.10,
]

TARGET_EXECUTABLE_SHARES = [
    0.90,
    0.95,
    0.975,
]


EXPECTED_SCENARIO_N = (
    len(
        LIQUIDITY_BASES
    )
    *
    len(
        PARTICIPATION_CAPS
    )
    *
    len(
        TARGET_EXECUTABLE_SHARES
    )
)


# ------------------------------------------------------------
# IMPORTANT:
#
# We deliberately keep EXECUTION_DAYS = 1.
#
# A 3-day/5-day horizon would require another explicit
# assumption regarding future reopening and/or static ADV.
#
# Step 5 therefore does NOT introduce such assumptions.
# ------------------------------------------------------------


TRADE_WEIGHT_TOL = 1e-15

FLOAT_TOL = 1e-12


# ============================================================
# 6. Fixed descriptive factor families
#
# These families were defined before Step 5 and are used only
# for descriptive robustness summaries.
# ============================================================

STRUCTURAL_FACTORS = {
    "residual_degree_percentile",
    "delta_degree_percentile",
    "cross_industry_degree_percentile",
    "cross_industry_degree_ratio",
    "outside_community_degree_percentile",
}

DYNAMIC_FACTORS = {
    "delta_residual_degree_percentile_1m",
    "delta_cross_industry_degree_percentile_1m",
    "neighbor_jaccard_1m",
    "neighbor_retention_1m",
}


def factor_family(
    factor_name: str,
) -> str:

    if factor_name in STRUCTURAL_FACTORS:

        return "STRUCTURAL"

    if factor_name in DYNAMIC_FACTORS:

        return "DYNAMIC"

    return "UNCLASSIFIED"


# ============================================================
# 7. I/O utilities
# ============================================================

def save_csv_atomic(
    df: pd.DataFrame,
    path: Path,
) -> None:

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


def save_json_atomic(
    obj,
    path: Path,
) -> None:

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


def canonical_hash(
    obj,
) -> str:

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


def parse_bool(
    x,
) -> bool:

    if isinstance(
        x,
        bool,
    ):

        return x

    value = (
        str(x)
        .strip()
        .lower()
    )

    if value in {
        "true",
        "1",
        "yes",
        "y",
    }:

        return True

    if value in {
        "false",
        "0",
        "no",
        "n",
    }:

        return False

    raise ValueError(
        f"Cannot parse bool value: {x}"
    )


def read_qa(
    path: Path,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"QA file not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    if not {
        "qa_name",
        "qa_value",
    }.issubset(
        df.columns
    ):

        raise RuntimeError(
            f"Unexpected QA schema:\n{path}"
        )

    return dict(
        zip(
            df[
                "qa_name"
            ],
            df[
                "qa_value"
            ],
        )
    )


# ============================================================
# 8. Scenario construction
# ============================================================

def participation_label(
    value: float,
) -> str:

    pct = (
        value
        *
        100
    )

    if float(
        pct
    ).is_integer():

        return f"P{int(pct):02d}"

    return (
        "P"
        +
        str(
            pct
        )
        .replace(
            ".",
            "p",
        )
    )


def executable_label(
    value: float,
) -> str:

    pct = (
        value
        *
        100
    )

    if float(
        pct
    ).is_integer():

        return f"E{int(pct)}"

    return (
        "E"
        +
        str(
            pct
        )
        .replace(
            ".",
            "p",
        )
    )


def make_scenario_id(
    liquidity_basis: str,
    participation_cap: float,
    target_share: float,
) -> str:

    return (
        f"{liquidity_basis}_"
        f"{participation_label(participation_cap)}_"
        f"{executable_label(target_share)}"
    )


def build_scenario_table():

    rows = []

    scenario_order = 0

    for liquidity_basis in (
        LIQUIDITY_BASES
    ):

        for participation_cap in (
            PARTICIPATION_CAPS
        ):

            for target_share in (
                TARGET_EXECUTABLE_SHARES
            ):

                scenario_order += 1

                scenario_id = (
                    make_scenario_id(
                        liquidity_basis,
                        participation_cap,
                        target_share,
                    )
                )

                is_primary = bool(
                    liquidity_basis
                    ==
                    PRIMARY_LIQUIDITY_BASIS

                    and

                    np.isclose(
                        participation_cap,
                        PRIMARY_PARTICIPATION_CAP,
                    )

                    and

                    np.isclose(
                        target_share,
                        PRIMARY_TARGET_EXECUTABLE_SHARE,
                    )
                )

                rows.append(
                    {
                        "scenario_order":
                            scenario_order,

                        "scenario_id":
                            scenario_id,

                        "liquidity_basis":
                            liquidity_basis,

                        "participation_cap":
                            participation_cap,

                        "target_executable_share":
                            target_share,

                        "execution_days":
                            EXECUTION_DAYS,

                        "is_primary":
                            is_primary,
                    }
                )

    scenarios = pd.DataFrame(
        rows
    )

    return scenarios


# ============================================================
# 9. Validate frozen Step 3 and Step 4
# ============================================================

def validate_upstream():

    step3_qa = (
        read_qa(
            STEP3_QA_PATH
        )
    )

    step4_qa = (
        read_qa(
            STEP4_QA_PATH
        )
    )

    if (
        "all_formal_qa_pass"
        not in step3_qa
    ):

        raise RuntimeError(
            "Step-3 QA does not contain "
            "`all_formal_qa_pass`."
        )

    if (
        "all_formal_qa_pass"
        not in step4_qa
    ):

        raise RuntimeError(
            "Step-4 QA does not contain "
            "`all_formal_qa_pass`."
        )

    if not parse_bool(
        step3_qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            "Step 3 is not formally frozen/PASS."
        )

    if not parse_bool(
        step4_qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            "Step 4 is not formally frozen/PASS."
        )

    return (
        step3_qa,
        step4_qa,
    )


# ============================================================
# 10. Load minimal frozen Step-3 stock-level data
#
# IMPORTANT FOR SPEED:
#
# Step 5 reads only the columns actually needed.
# ============================================================

def load_step3_panel():

    if not STEP3_PANEL_PATH.exists():

        raise FileNotFoundError(
            "Frozen Step-3 stock panel not found:\n"
            f"{STEP3_PANEL_PATH}"
        )

    columns = [
        "analysis_date",
        "previous_analysis_date",
        "window",
        "factor_name",
        "method",
        "portfolio_leg",
        "capacity_abs_trade_weight",
        "effective_adv20_1d_cny",
        "effective_adv60_1d_cny",
    ]

    df = pd.read_parquet(
        STEP3_PANEL_PATH,
        columns=columns,
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
        "previous_analysis_date"
    ] = pd.to_datetime(
        df[
            "previous_analysis_date"
        ],
        errors="raise",
    )

    df[
        "window"
    ] = pd.to_numeric(
        df[
            "window"
        ],
        errors="raise",
    ).astype(int)

    numeric_columns = [
        "capacity_abs_trade_weight",
        "effective_adv20_1d_cny",
        "effective_adv60_1d_cny",
    ]

    for column in numeric_columns:

        df[
            column
        ] = pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )

    if (
        df[
            "capacity_abs_trade_weight"
        ]
        <
        -TRADE_WEIGHT_TOL
    ).any():

        raise RuntimeError(
            "Negative absolute trade weight detected."
        )

    trade = (
        df[
            "capacity_abs_trade_weight"
        ]
        >
        TRADE_WEIGHT_TOL
    )

    for column in [
        "effective_adv20_1d_cny",
        "effective_adv60_1d_cny",
    ]:

        if (
            df.loc[
                trade,
                column,
            ]
            .isna()
            .any()
        ):

            raise RuntimeError(
                f"Missing {column} for actual trades."
            )

        if (
            df.loc[
                trade,
                column,
            ]
            <
            0
        ).any():

            raise RuntimeError(
                f"Negative {column} detected."
            )

    return df


# ============================================================
# 11. Prepare trade weights ONCE
#
# This is the main Step-5 performance optimization.
#
# We do NOT repeat groupby operations for every scenario.
# ============================================================

def prepare_trade_panel(
    panel: pd.DataFrame,
):

    # --------------------------------------------------------
    # Count every rebalance, including theoretically no-trade
    # rebalances.
    # --------------------------------------------------------

    leg_spec_cols = [
        "factor_name",
        "method",
        "window",
        "portfolio_leg",
    ]

    leg_rebalance_cols = [
        "analysis_date",
        "previous_analysis_date",
        "factor_name",
        "method",
        "window",
        "portfolio_leg",
    ]

    pair_spec_cols = [
        "factor_name",
        "method",
        "window",
    ]

    pair_rebalance_cols = [
        "analysis_date",
        "previous_analysis_date",
        "factor_name",
        "method",
        "window",
    ]

    leg_rebalance_counts = (

        panel[
            leg_rebalance_cols
        ]
        .drop_duplicates()

        .groupby(
            leg_spec_cols,
            observed=True,
        )
        .size()
        .rename(
            "total_rebalance_n"
        )
        .reset_index()
    )

    pair_rebalance_counts = (

        panel[
            pair_rebalance_cols
        ]
        .drop_duplicates()

        .groupby(
            pair_spec_cols,
            observed=True,
        )
        .size()
        .rename(
            "total_rebalance_n"
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # From now on only actual market trades are needed.
    # --------------------------------------------------------

    trade = panel[
        panel[
            "capacity_abs_trade_weight"
        ]
        >
        TRADE_WEIGHT_TOL
    ].copy()

    # --------------------------------------------------------
    # Leg total trade weight within each rebalance.
    # --------------------------------------------------------

    trade[
        "_leg_rebalance_total_d"
    ] = (

        trade.groupby(
            leg_rebalance_cols,
            observed=True,
        )[
            "capacity_abs_trade_weight"
        ]
        .transform(
            "sum"
        )
    )

    # --------------------------------------------------------
    # Pair total trade weight within each rebalance.
    #
    # Both Q1 and Q5 stock trades are pooled, matching
    # frozen Step-3 pair construction.
    # --------------------------------------------------------

    trade[
        "_pair_rebalance_total_d"
    ] = (

        trade.groupby(
            pair_rebalance_cols,
            observed=True,
        )[
            "capacity_abs_trade_weight"
        ]
        .transform(
            "sum"
        )
    )

    if (
        trade[
            "_leg_rebalance_total_d"
        ]
        <=
        0
    ).any():

        raise RuntimeError(
            "Invalid leg rebalance total trade weight."
        )

    if (
        trade[
            "_pair_rebalance_total_d"
        ]
        <=
        0
    ).any():

        raise RuntimeError(
            "Invalid pair rebalance total trade weight."
        )

    return (
        trade,
        leg_rebalance_counts,
        pair_rebalance_counts,
    )


# ============================================================
# 12. Exact piecewise capacity-curve preparation
#
# For one factor/window specification:
#
# G(A)
# =
# constant_share
# +
# sum_i q_i min(1, b_i/A)
#
# where
#
# b_i = ADV_i / |delta_w_i|
#
# Participation cap p is applied later:
#
# c_i = p * b_i.
#
#
# Since multiplying all breakpoints by p multiplies the
# capacity by exactly p, we solve the p=1 base curve ONCE.
# ============================================================

def prepare_piecewise_curve(
    q,
    base_breakpoints,
    constant_share: float,
):

    q = np.asarray(
        q,
        dtype=float,
    )

    b = np.asarray(
        base_breakpoints,
        dtype=float,
    )

    valid = (
        np.isfinite(
            q
        )
        &
        np.isfinite(
            b
        )
        &
        (
            q
            >
            0
        )
        &
        (
            b
            >=
            0
        )
    )

    q = q[
        valid
    ]

    b = b[
        valid
    ]

    if len(
        q
    ) == 0:

        return {
            "constant_share":
                float(
                    constant_share
                ),

            "total_q":
                0.0,

            "baseline_share":
                float(
                    constant_share
                ),

            "breakpoint":
                np.array(
                    [],
                    dtype=float,
                ),

            "prefix_q":
                np.array(
                    [],
                    dtype=float,
                ),

            "prefix_qb":
                np.array(
                    [],
                    dtype=float,
                ),
        }

    order = np.argsort(
        b,
        kind="mergesort",
    )

    b = b[
        order
    ]

    q = q[
        order
    ]

    # --------------------------------------------------------
    # Aggregate duplicate breakpoint values.
    #
    # This avoids numerical ambiguity at tied boundaries.
    # --------------------------------------------------------

    unique_b, start_idx = np.unique(
        b,
        return_index=True,
    )

    q_by_breakpoint = np.add.reduceat(
        q,
        start_idx,
    )

    qb_by_breakpoint = np.add.reduceat(
        q
        *
        b,
        start_idx,
    )

    prefix_q = np.cumsum(
        q_by_breakpoint
    )

    prefix_qb = np.cumsum(
        qb_by_breakpoint
    )

    total_q = float(
        prefix_q[-1]
    )

    positive_breakpoint_q = float(
        q[
            b
            >
            0
        ].sum()
    )

    baseline = (
        float(
            constant_share
        )
        +
        positive_breakpoint_q
    )

    return {
        "constant_share":
            float(
                constant_share
            ),

        "total_q":
            total_q,

        "baseline_share":
            float(
                baseline
            ),

        "breakpoint":
            unique_b,

        "prefix_q":
            prefix_q,

        "prefix_qb":
            prefix_qb,
    }


# ============================================================
# 13. Fast exact capacity solver
#
# Within a segment:
#
# b_k <= A <= b_{k+1}
#
# G(A)
# =
# constant
# + (total_q - Q_k)
# + C_k / A
#
# Setting G(A)=target gives:
#
# A
# =
# C_k
# /
# [
#   target
#   - constant
#   - total_q
#   + Q_k
# ].
#
#
# This avoids dozens of repeated passes over millions of rows.
# ============================================================

def solve_prepared_curve(
    prepared,
    target: float,
):

    constant_share = float(
        prepared[
            "constant_share"
        ]
    )

    total_q = float(
        prepared[
            "total_q"
        ]
    )

    baseline = float(
        prepared[
            "baseline_share"
        ]
    )

    b = prepared[
        "breakpoint"
    ]

    prefix_q = prepared[
        "prefix_q"
    ]

    prefix_qb = prepared[
        "prefix_qb"
    ]

    # --------------------------------------------------------
    # No trade.
    # --------------------------------------------------------

    if len(
        b
    ) == 0:

        if (
            constant_share
            >=
            target
        ):

            return {
                "base_capacity":
                    np.inf,

                "status":
                    "NO_TRADE_UNCONSTRAINED",

                "baseline_share":
                    constant_share,
            }

        return {
            "base_capacity":
                0.0,

            "status":
                "NO_FEASIBLE_CAPACITY",

            "baseline_share":
                constant_share,
        }

    # --------------------------------------------------------
    # Even infinitesimal NAV cannot meet target.
    #
    # This is caused by zero immediate liquidity weight.
    # --------------------------------------------------------

    if (
        baseline
        <
        target
        -
        1e-14
    ):

        return {
            "base_capacity":
                0.0,

            "status":
                "ZERO_CAPACITY_BASELINE_BELOW_TARGET",

            "baseline_share":
                baseline,
        }

    if (
        constant_share
        >=
        target
    ):

        return {
            "base_capacity":
                np.inf,

            "status":
                "UNBOUNDED_BY_LIQUIDITY_TEST",

            "baseline_share":
                baseline,
        }

    denominator = (

        target

        -

        constant_share

        -

        total_q

        +

        prefix_q
    )

    valid_denom = (
        denominator
        >
        0
    )

    candidate = np.full(
        len(
            b
        ),
        np.nan,
        dtype=float,
    )

    candidate[
        valid_denom
    ] = (

        prefix_qb[
            valid_denom
        ]

        /

        denominator[
            valid_denom
        ]
    )

    next_b = np.empty_like(
        b
    )

    if len(
        b
    ) > 1:

        next_b[
            :-1
        ] = b[
            1:
        ]

    next_b[
        -1
    ] = np.inf

    lower_ok = (
        candidate
        >=
        b
        -
        FLOAT_TOL
        *
        np.maximum(
            1.0,
            np.abs(
                b
            ),
        )
    )

    upper_ok = (
        candidate
        <=
        next_b
        +
        FLOAT_TOL
        *
        np.maximum(
            1.0,
            np.where(
                np.isfinite(
                    next_b
                ),
                np.abs(
                    next_b
                ),
                1.0,
            ),
        )
    )

    valid_solution = (
        valid_denom
        &
        np.isfinite(
            candidate
        )
        &
        lower_ok
        &
        upper_ok
    )

    positions = np.flatnonzero(
        valid_solution
    )

    if len(
        positions
    ) > 0:

        capacity = float(
            candidate[
                positions[0]
            ]
        )

        return {
            "base_capacity":
                capacity,

            "status":
                "PASS_EXACT",

            "baseline_share":
                baseline,
        }

    # ========================================================
    # Numerical fallback
    #
    # This should be extremely rare.
    #
    # We intentionally call the frozen Step-4 solver rather
    # than inventing another convention.
    # ========================================================

    raise ArithmeticError(
        "Exact segment solver could not identify "
        "a numerically valid segment."
    )


# ============================================================
# 14. Safe solver wrapper with frozen Step-4 fallback
# ============================================================

def solve_capacity_fast(
    q,
    base_breakpoint,
    constant_share,
    participation_cap,
    target_share,
    prepared=None,
):

    if prepared is None:

        prepared = (
            prepare_piecewise_curve(
                q,
                base_breakpoint,
                constant_share,
            )
        )

    try:

        solved = (
            solve_prepared_curve(
                prepared,
                target_share,
            )
        )

        base_capacity = (
            solved[
                "base_capacity"
            ]
        )

        if np.isfinite(
            base_capacity
        ):

            capacity_cny = (
                base_capacity
                *
                participation_cap
                *
                EXECUTION_DAYS
            )

        else:

            capacity_cny = np.inf

        status = (
            solved[
                "status"
            ]
        )

        solver_used = (
            "EXACT_PIECEWISE"
        )

    except ArithmeticError:

        # ----------------------------------------------------
        # Frozen Step-4 bisection fallback.
        # ----------------------------------------------------

        actual_breakpoint = (

            np.asarray(
                base_breakpoint,
                dtype=float,
            )

            *

            participation_cap

            *

            EXECUTION_DAYS
        )

        fallback = (
            solve_capacity_limit(
                np.asarray(
                    q,
                    dtype=float,
                ),

                actual_breakpoint,

                target_share,

                constant_share,
            )
        )

        capacity_cny = (
            fallback[
                "capacity_cny"
            ]
        )

        status = (
            fallback[
                "status"
            ]
        )

        solver_used = (
            "STEP4_BISECTION_FALLBACK"
        )

    return {
        "capacity_cny":
            float(
                capacity_cny
            ),

        "capacity_status":
            status,

        "solver_used":
            solver_used,

        "baseline_executable_share":
            float(
                prepared[
                    "baseline_share"
                ]
            ),
    }


# ============================================================
# 15. Build one factor-window specification
# ============================================================

def evaluate_specification(
    group: pd.DataFrame,
    total_rebalance_n: int,
    scope: str,
    portfolio_scope: str,
    factor_name: str,
    method: str,
    window: int,
    rebalance_total_col: str,
    scenarios: pd.DataFrame,
):

    if total_rebalance_n <= 0:

        raise RuntimeError(
            "Invalid rebalance count."
        )

    # --------------------------------------------------------
    # Number of rebalances that contain actual trades.
    # --------------------------------------------------------

    if scope == "PORTFOLIO_LEG":

        rebalance_key = [
            "analysis_date",
            "previous_analysis_date",
            "factor_name",
            "method",
            "window",
            "portfolio_leg",
        ]

    else:

        rebalance_key = [
            "analysis_date",
            "previous_analysis_date",
            "factor_name",
            "method",
            "window",
        ]

    trade_rebalance_n = int(
        group[
            rebalance_key
        ]
        .drop_duplicates()
        .shape[0]
    )

    no_trade_rebalance_n = (
        total_rebalance_n
        -
        trade_rebalance_n
    )

    if no_trade_rebalance_n < 0:

        raise RuntimeError(
            "Negative no-trade rebalance count."
        )

    constant_share = (
        no_trade_rebalance_n
        /
        total_rebalance_n
    )

    d = (
        group[
            "capacity_abs_trade_weight"
        ]
        .to_numpy(
            dtype=float
        )
    )

    rebalance_total_d = (
        group[
            rebalance_total_col
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Equal weight across rebalance dates.
    #
    # Within a rebalance:
    #
    # q_it = d_it / sum_j d_j
    #
    # Across T dates:
    #
    # q_it / T
    # --------------------------------------------------------

    q = (

        d

        /

        rebalance_total_d

        /

        total_rebalance_n
    )

    # --------------------------------------------------------
    # QA:
    #
    # sum(q) + no_trade/T must equal 1.
    # --------------------------------------------------------

    q_identity = float(
        q.sum()
        +
        constant_share
    )

    if not np.isclose(
        q_identity,
        1.0,
        rtol=1e-11,
        atol=1e-11,
    ):

        raise RuntimeError(
            "Equal-rebalance weight identity failed:\n"
            f"{scope} | "
            f"{factor_name} | "
            f"W{window} | "
            f"{portfolio_scope}\n"
            f"sum(q)+constant = {q_identity}"
        )

    adv20 = (
        group[
            "effective_adv20_1d_cny"
        ]
        .to_numpy(
            dtype=float
        )
    )

    adv60 = (
        group[
            "effective_adv60_1d_cny"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Base breakpoint:
    #
    # ADV / |delta_w|
    #
    # Participation cap is multiplied later.
    # --------------------------------------------------------

    base20 = (
        adv20
        /
        d
    )

    base60 = (
        adv60
        /
        d
    )

    # --------------------------------------------------------
    # Sort each liquidity curve only ONCE.
    # --------------------------------------------------------

    prepared_by_basis = {
        "ADV20":
            prepare_piecewise_curve(
                q,
                base20,
                constant_share,
            ),

        "ADV60":
            prepare_piecewise_curve(
                q,
                base60,
                constant_share,
            ),
    }

    base_breakpoint_by_basis = {
        "ADV20":
            base20,

        "ADV60":
            base60,
    }

    results = []

    for scenario in scenarios.itertuples(
        index=False
    ):

        basis = (
            scenario.liquidity_basis
        )

        p = float(
            scenario.participation_cap
        )

        target = float(
            scenario.target_executable_share
        )

        solved = (
            solve_capacity_fast(
                q=q,
                base_breakpoint=(
                    base_breakpoint_by_basis[
                        basis
                    ]
                ),
                constant_share=constant_share,
                participation_cap=p,
                target_share=target,
                prepared=(
                    prepared_by_basis[
                        basis
                    ]
                ),
            )
        )

        capacity_cny = (
            solved[
                "capacity_cny"
            ]
        )

        if np.isfinite(
            capacity_cny
        ):

            capacity_yi = (
                capacity_cny
                /
                1e8
            )

        else:

            capacity_yi = np.inf

        # ----------------------------------------------------
        # Verify actual coverage at the solved capacity using
        # the frozen Step-4 coverage function.
        # ----------------------------------------------------

        if (
            np.isfinite(
                capacity_cny
            )
            and
            capacity_cny
            >
            0
        ):

            actual_breakpoint = (

                base_breakpoint_by_basis[
                    basis
                ]

                *

                p

                *

                EXECUTION_DAYS
            )

            coverage_at_capacity = (
                coverage_from_breakpoints(
                    capacity_cny,
                    q,
                    actual_breakpoint,
                    constant_share,
                )
            )

        elif (
            capacity_cny
            ==
            0
        ):

            coverage_at_capacity = (
                solved[
                    "baseline_executable_share"
                ]
            )

        else:

            coverage_at_capacity = (
                constant_share
            )

        if scope == "HYPOTHETICAL_Q1_Q5_PAIR":

            gross_notional = (
                2.0
                *
                capacity_cny
                if np.isfinite(
                    capacity_cny
                )
                else np.inf
            )

        else:

            gross_notional = (
                capacity_cny
            )

        results.append(
            {
                "scope":
                    scope,

                "portfolio_scope":
                    portfolio_scope,

                "factor_family":
                    factor_family(
                        factor_name
                    ),

                "factor_name":
                    factor_name,

                "method":
                    method,

                "window":
                    int(
                        window
                    ),

                "scenario_order":
                    int(
                        scenario.scenario_order
                    ),

                "scenario_id":
                    scenario.scenario_id,

                "liquidity_basis":
                    basis,

                "participation_cap":
                    p,

                "target_executable_share":
                    target,

                "execution_days":
                    EXECUTION_DAYS,

                "is_primary":
                    bool(
                        scenario.is_primary
                    ),

                "rebalance_n":
                    int(
                        total_rebalance_n
                    ),

                "trade_rebalance_n":
                    int(
                        trade_rebalance_n
                    ),

                "no_trade_rebalance_n":
                    int(
                        no_trade_rebalance_n
                    ),

                "constant_no_trade_share":
                    float(
                        constant_share
                    ),

                "baseline_executable_share":
                    solved[
                        "baseline_executable_share"
                    ],

                "capacity_status":
                    solved[
                        "capacity_status"
                    ],

                "solver_used":
                    solved[
                        "solver_used"
                    ],

                "capacity_cny":
                    capacity_cny,

                "capacity_yi_cny":
                    capacity_yi,

                "gross_notional_at_capacity_cny":
                    gross_notional,

                "executable_share_at_capacity":
                    float(
                        coverage_at_capacity
                    ),
            }
        )

    return results


# ============================================================
# 16. Evaluate all leg and pair specifications
# ============================================================

def build_robustness_matrix(
    trade: pd.DataFrame,
    leg_counts: pd.DataFrame,
    pair_counts: pd.DataFrame,
    scenarios: pd.DataFrame,
):

    rows = []

    # ========================================================
    # A. Q1 / Q5 leg specifications
    # ========================================================

    leg_spec_cols = [
        "factor_name",
        "method",
        "window",
        "portfolio_leg",
    ]

    leg_count_lookup = {
        (
            str(row.factor_name),
            str(row.method),
            int(row.window),
            str(row.portfolio_leg),
        ):
            int(
                row.total_rebalance_n
            )

        for row in leg_counts.itertuples(
            index=False
        )
    }

    leg_groups = trade.groupby(
        leg_spec_cols,
        sort=False,
        observed=True,
    )

    leg_group_n = (
        leg_groups.ngroups
    )

    for number, (
        keys,
        group,
    ) in enumerate(
        leg_groups,
        start=1,
    ):

        (
            factor_name,
            method,
            window,
            portfolio_leg,
        ) = keys

        if (
            number == 1
            or
            number % 10 == 0
            or
            number == leg_group_n
        ):

            print(
                "Leg robustness specification: "
                f"{number}/{leg_group_n}"
            )

        count_key = (
            str(
                factor_name
            ),
            str(
                method
            ),
            int(
                window
            ),
            str(
                portfolio_leg
            ),
        )

        total_rebalance_n = (
            leg_count_lookup[
                count_key
            ]
        )

        rows.extend(
            evaluate_specification(
                group=group,
                total_rebalance_n=(
                    total_rebalance_n
                ),
                scope="PORTFOLIO_LEG",
                portfolio_scope=str(
                    portfolio_leg
                ),
                factor_name=str(
                    factor_name
                ),
                method=str(
                    method
                ),
                window=int(
                    window
                ),
                rebalance_total_col=(
                    "_leg_rebalance_total_d"
                ),
                scenarios=scenarios,
            )
        )

    # ========================================================
    # B. Hypothetical Q1-Q5 pair specifications
    # ========================================================

    pair_spec_cols = [
        "factor_name",
        "method",
        "window",
    ]

    pair_count_lookup = {
        (
            str(row.factor_name),
            str(row.method),
            int(row.window),
        ):
            int(
                row.total_rebalance_n
            )

        for row in pair_counts.itertuples(
            index=False
        )
    }

    pair_groups = trade.groupby(
        pair_spec_cols,
        sort=False,
        observed=True,
    )

    pair_group_n = (
        pair_groups.ngroups
    )

    for number, (
        keys,
        group,
    ) in enumerate(
        pair_groups,
        start=1,
    ):

        (
            factor_name,
            method,
            window,
        ) = keys

        if (
            number == 1
            or
            number % 10 == 0
            or
            number == pair_group_n
        ):

            print(
                "Pair robustness specification: "
                f"{number}/{pair_group_n}"
            )

        count_key = (
            str(
                factor_name
            ),
            str(
                method
            ),
            int(
                window
            ),
        )

        total_rebalance_n = (
            pair_count_lookup[
                count_key
            ]
        )

        rows.extend(
            evaluate_specification(
                group=group,
                total_rebalance_n=(
                    total_rebalance_n
                ),
                scope=(
                    "HYPOTHETICAL_Q1_Q5_PAIR"
                ),
                portfolio_scope=(
                    "Q1_Q5_PAIR"
                ),
                factor_name=str(
                    factor_name
                ),
                method=str(
                    method
                ),
                window=int(
                    window
                ),
                rebalance_total_col=(
                    "_pair_rebalance_total_d"
                ),
                scenarios=scenarios,
            )
        )

    matrix = pd.DataFrame(
        rows
    )

    return matrix


# ============================================================
# 17. Add comparison to primary Step-4 capacity
# ============================================================

def add_primary_comparison(
    matrix: pd.DataFrame,
):

    spec_key = [
        "scope",
        "portfolio_scope",
        "factor_name",
        "method",
        "window",
    ]

    primary = (
        matrix[
            matrix[
                "is_primary"
            ]
        ][
            spec_key
            +
            [
                "capacity_cny",
            ]
        ]
        .rename(
            columns={
                "capacity_cny":
                    "step5_primary_capacity_cny",
            }
        )
    )

    if primary[
        spec_key
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Step-5 primary specification."
        )

    matrix = matrix.merge(
        primary,
        on=spec_key,
        how="left",
        validate="many_to_one",
    )

    matrix[
        "capacity_minus_primary_cny"
    ] = (

        matrix[
            "capacity_cny"
        ]

        -

        matrix[
            "step5_primary_capacity_cny"
        ]
    )

    # --------------------------------------------------------
    # Safe ratio.
    # --------------------------------------------------------

    numerator = (
        matrix[
            "capacity_cny"
        ]
        .to_numpy(
            dtype=float
        )
    )

    denominator = (
        matrix[
            "step5_primary_capacity_cny"
        ]
        .to_numpy(
            dtype=float
        )
    )

    ratio = np.full(
        len(
            matrix
        ),
        np.nan,
        dtype=float,
    )

    finite_valid = (
        np.isfinite(
            numerator
        )
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

    ratio[
        finite_valid
    ] = (

        numerator[
            finite_valid
        ]

        /

        denominator[
            finite_valid
        ]
    )

    both_inf = (
        np.isposinf(
            numerator
        )
        &
        np.isposinf(
            denominator
        )
    )

    ratio[
        both_inf
    ] = 1.0

    matrix[
        "capacity_ratio_to_primary"
    ] = ratio

    return matrix


# ============================================================
# 18. Cross-check Step-5 primary against frozen Step 4
# ============================================================

def build_primary_crosscheck(
    matrix: pd.DataFrame,
):

    if not STEP4_CAPACITY_PATH.exists():

        raise FileNotFoundError(
            "Frozen Step-4 capacity file not found:\n"
            f"{STEP4_CAPACITY_PATH}"
        )

    step4 = pd.read_csv(
        STEP4_CAPACITY_PATH
    )

    required = [
        "scope",
        "portfolio_scope",
        "factor_name",
        "method",
        "window",
        "mean_coverage_capacity_cny",
    ]

    missing = [
        column
        for column in required
        if column not in step4.columns
    ]

    if missing:

        raise RuntimeError(
            "Frozen Step-4 capacity table "
            "is missing columns:\n"
            f"{missing}"
        )

    step4[
        "window"
    ] = pd.to_numeric(
        step4[
            "window"
        ],
        errors="raise",
    ).astype(int)

    primary = (
        matrix[
            matrix[
                "is_primary"
            ]
        ].copy()
    )

    key = [
        "scope",
        "portfolio_scope",
        "factor_name",
        "method",
        "window",
    ]

    comparison = primary.merge(
        step4[
            key
            +
            [
                "mean_coverage_capacity_cny",
            ]
        ],
        on=key,
        how="left",
        validate="one_to_one",
    )

    comparison[
        "capacity_abs_difference_cny"
    ] = (

        comparison[
            "capacity_cny"
        ]

        -

        comparison[
            "mean_coverage_capacity_cny"
        ]
    ).abs()

    comparison[
        "capacity_relative_difference"
    ] = (

        comparison[
            "capacity_abs_difference_cny"
        ]

        /

        comparison[
            "mean_coverage_capacity_cny"
        ]
        .abs()
    )

    return comparison


# ============================================================
# 19. Wide robustness matrix
# ============================================================

def build_wide_matrix(
    matrix: pd.DataFrame,
):

    index_columns = [
        "scope",
        "portfolio_scope",
        "factor_family",
        "factor_name",
        "method",
        "window",
    ]

    wide = (
        matrix.pivot(
            index=index_columns,
            columns="scenario_id",
            values="capacity_yi_cny",
        )
        .reset_index()
    )

    wide.columns.name = None

    return wide


# ============================================================
# 20. Scenario-level descriptive summary
# ============================================================

def build_scenario_summary(
    matrix: pd.DataFrame,
):

    group_columns = [
        "scope",
        "portfolio_scope",
        "scenario_order",
        "scenario_id",
        "liquidity_basis",
        "participation_cap",
        "target_executable_share",
        "is_primary",
    ]

    rows = []

    for keys, group in matrix.groupby(
        group_columns,
        sort=False,
        observed=True,
        dropna=False,
    ):

        row = dict(
            zip(
                group_columns,
                keys,
            )
        )

        values = (
            group[
                "capacity_yi_cny"
            ]
            .to_numpy(
                dtype=float
            )
        )

        finite = values[
            np.isfinite(
                values
            )
        ]

        row[
            "specification_n"
        ] = int(
            len(
                group
            )
        )

        row[
            "zero_capacity_n"
        ] = int(
            np.sum(
                values
                ==
                0
            )
        )

        row[
            "infinite_capacity_n"
        ] = int(
            np.sum(
                np.isposinf(
                    values
                )
            )
        )

        if len(
            finite
        ) > 0:

            row[
                "mean_capacity_yi_cny_finite"
            ] = float(
                np.mean(
                    finite
                )
            )

            row[
                "median_capacity_yi_cny_finite"
            ] = float(
                np.median(
                    finite
                )
            )

            row[
                "p10_capacity_yi_cny_finite"
            ] = float(
                np.quantile(
                    finite,
                    0.10,
                )
            )

            row[
                "p90_capacity_yi_cny_finite"
            ] = float(
                np.quantile(
                    finite,
                    0.90,
                )
            )

        else:

            row[
                "mean_capacity_yi_cny_finite"
            ] = np.nan

            row[
                "median_capacity_yi_cny_finite"
            ] = np.nan

            row[
                "p10_capacity_yi_cny_finite"
            ] = np.nan

            row[
                "p90_capacity_yi_cny_finite"
            ] = np.nan

        ratio = (
            group[
                "capacity_ratio_to_primary"
            ]
            .to_numpy(
                dtype=float
            )
        )

        finite_ratio = ratio[
            np.isfinite(
                ratio
            )
        ]

        row[
            "mean_capacity_ratio_to_primary"
        ] = (
            float(
                np.mean(
                    finite_ratio
                )
            )
            if len(
                finite_ratio
            )
            >
            0
            else np.nan
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 21. Structural vs dynamic robustness summary
# ============================================================

def build_family_summary(
    matrix: pd.DataFrame,
):

    valid = matrix[
        matrix[
            "factor_family"
        ]
        .isin(
            [
                "STRUCTURAL",
                "DYNAMIC",
            ]
        )
    ].copy()

    group_columns = [
        "scope",
        "portfolio_scope",
        "factor_family",
        "window",
        "scenario_order",
        "scenario_id",
        "liquidity_basis",
        "participation_cap",
        "target_executable_share",
        "is_primary",
    ]

    summary = (

        valid.groupby(
            group_columns,
            as_index=False,
            observed=True,
            dropna=False,
        )

        .agg(
            factor_n=(
                "factor_name",
                "nunique",
            ),

            mean_capacity_yi_cny=(
                "capacity_yi_cny",
                "mean",
            ),

            median_capacity_yi_cny=(
                "capacity_yi_cny",
                "median",
            ),

            mean_capacity_ratio_to_primary=(
                "capacity_ratio_to_primary",
                "mean",
            ),
        )
    )

    return summary


# ============================================================
# 22. Formal Step-5 QA
# ============================================================

def formal_qa(
    matrix: pd.DataFrame,
    scenarios: pd.DataFrame,
    primary_crosscheck: pd.DataFrame,
    step3_qa: dict,
    step4_qa: dict,
):

    qa = {}

    qa[
        "step3_formal_qa_pass"
    ] = parse_bool(
        step3_qa[
            "all_formal_qa_pass"
        ]
    )

    qa[
        "step4_formal_qa_pass"
    ] = parse_bool(
        step4_qa[
            "all_formal_qa_pass"
        ]
    )

    qa[
        "scenario_count"
    ] = int(
        len(
            scenarios
        )
    )

    qa[
        "expected_scenario_count"
    ] = int(
        EXPECTED_SCENARIO_N
    )

    qa[
        "scenario_count_ok"
    ] = bool(
        len(
            scenarios
        )
        ==
        EXPECTED_SCENARIO_N
    )

    qa[
        "primary_scenario_count"
    ] = int(
        scenarios[
            "is_primary"
        ]
        .sum()
    )

    qa[
        "exactly_one_primary_scenario"
    ] = bool(
        qa[
            "primary_scenario_count"
        ]
        ==
        1
    )

    # ========================================================
    # Specification count
    # ========================================================

    spec_key = [
        "scope",
        "portfolio_scope",
        "factor_name",
        "method",
        "window",
    ]

    spec_n = int(
        matrix[
            spec_key
        ]
        .drop_duplicates()
        .shape[0]
    )

    qa[
        "specification_count"
    ] = spec_n

    qa[
        "matrix_row_count"
    ] = int(
        len(
            matrix
        )
    )

    qa[
        "expected_matrix_row_count"
    ] = int(
        spec_n
        *
        EXPECTED_SCENARIO_N
    )

    qa[
        "matrix_row_count_ok"
    ] = bool(
        len(
            matrix
        )
        ==
        spec_n
        *
        EXPECTED_SCENARIO_N
    )

    qa[
        "duplicate_spec_scenario_count"
    ] = int(
        matrix[
            spec_key
            +
            [
                "scenario_id",
            ]
        ]
        .duplicated()
        .sum()
    )

    # ========================================================
    # Factor-family completeness
    # ========================================================

    qa[
        "unclassified_factor_row_count"
    ] = int(
        (
            matrix[
                "factor_family"
            ]
            ==
            "UNCLASSIFIED"
        )
        .sum()
    )

    # ========================================================
    # Exact solver fallback count
    # ========================================================

    qa[
        "step4_bisection_fallback_count"
    ] = int(
        (
            matrix[
                "solver_used"
            ]
            ==
            "STEP4_BISECTION_FALLBACK"
        )
        .sum()
    )

    # Fallback is allowed, not failure.

    # ========================================================
    # Root accuracy
    # ========================================================

    positive_finite = (
        np.isfinite(
            matrix[
                "capacity_cny"
            ]
        )
        &
        (
            matrix[
                "capacity_cny"
            ]
            >
            0
        )
    )

    root_error = (

        matrix.loc[
            positive_finite,
            "executable_share_at_capacity",
        ]

        -

        matrix.loc[
            positive_finite,
            "target_executable_share",
        ]
    ).abs()

    qa[
        "max_capacity_root_error"
    ] = float(
        root_error.max()
        if len(
            root_error
        )
        >
        0
        else 0.0
    )

    qa[
        "capacity_root_ok"
    ] = bool(
        qa[
            "max_capacity_root_error"
        ]
        <
        1e-8
    )

    # ========================================================
    # Primary Step-4 reproduction
    # ========================================================

    qa[
        "primary_crosscheck_missing_step4_count"
    ] = int(
        primary_crosscheck[
            "mean_coverage_capacity_cny"
        ]
        .isna()
        .sum()
    )

    qa[
        "max_primary_capacity_abs_difference_cny"
    ] = float(
        primary_crosscheck[
            "capacity_abs_difference_cny"
        ]
        .max()
    )

    qa[
        "max_primary_capacity_relative_difference"
    ] = float(
        primary_crosscheck[
            "capacity_relative_difference"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .max()
    )

    qa[
        "step4_primary_reproduced"
    ] = bool(
        np.allclose(
            primary_crosscheck[
                "capacity_cny"
            ]
            .to_numpy(
                dtype=float
            ),

            primary_crosscheck[
                "mean_coverage_capacity_cny"
            ]
            .to_numpy(
                dtype=float
            ),

            rtol=1e-8,
            atol=1e-2,
            equal_nan=False,
        )
    )

    # ========================================================
    # Monotonicity 1:
    #
    # Larger participation limit must not reduce capacity.
    # ========================================================

    monotonic_p_violation = 0

    p_group_key = (
        spec_key
        +
        [
            "liquidity_basis",
            "target_executable_share",
        ]
    )

    for _, group in matrix.groupby(
        p_group_key,
        sort=False,
        observed=True,
    ):

        group = group.sort_values(
            "participation_cap"
        )

        values = (
            group[
                "capacity_cny"
            ]
            .to_numpy(
                dtype=float
            )
        )

        finite_values = np.where(
            np.isposinf(
                values
            ),
            np.finfo(
                float
            ).max,
            values,
        )

        if (
            np.diff(
                finite_values
            )
            <
            -1e-5
        ).any():

            monotonic_p_violation += 1

    qa[
        "participation_monotonicity_violation_count"
    ] = int(
        monotonic_p_violation
    )

    qa[
        "participation_monotonicity_ok"
    ] = bool(
        monotonic_p_violation
        ==
        0
    )

    # ========================================================
    # Monotonicity 2:
    #
    # More stringent executable-share target must not
    # increase capacity.
    # ========================================================

    target_violation = 0

    target_group_key = (
        spec_key
        +
        [
            "liquidity_basis",
            "participation_cap",
        ]
    )

    for _, group in matrix.groupby(
        target_group_key,
        sort=False,
        observed=True,
    ):

        group = group.sort_values(
            "target_executable_share"
        )

        values = (
            group[
                "capacity_cny"
            ]
            .to_numpy(
                dtype=float
            )
        )

        finite_values = np.where(
            np.isposinf(
                values
            ),
            np.finfo(
                float
            ).max,
            values,
        )

        if (
            np.diff(
                finite_values
            )
            >
            1e-5
        ).any():

            target_violation += 1

    qa[
        "target_stringency_monotonicity_violation_count"
    ] = int(
        target_violation
    )

    qa[
        "target_stringency_monotonicity_ok"
    ] = bool(
        target_violation
        ==
        0
    )

    # ========================================================
    # Exact scaling identity:
    #
    # At fixed basis + target,
    #
    # capacity is linear in participation cap p.
    #
    # A_cap(p2) / A_cap(p1) = p2 / p1.
    # ========================================================

    scaling_errors = []

    scale_group_key = (
        spec_key
        +
        [
            "liquidity_basis",
            "target_executable_share",
        ]
    )

    for _, group in matrix.groupby(
        scale_group_key,
        sort=False,
        observed=True,
    ):

        finite = group[
            np.isfinite(
                group[
                    "capacity_cny"
                ]
            )
            &
            (
                group[
                    "capacity_cny"
                ]
                >
                0
            )
        ].copy()

        if len(
            finite
        ) < 2:

            continue

        scaled = (

            finite[
                "capacity_cny"
            ]

            /

            finite[
                "participation_cap"
            ]
        )

        reference = float(
            scaled.iloc[
                0
            ]
        )

        relative_error = (

            (
                scaled
                -
                reference
            )
            .abs()

            /

            max(
                abs(
                    reference
                ),
                1.0,
            )
        )

        scaling_errors.extend(
            relative_error.tolist()
        )

    qa[
        "max_participation_scaling_relative_error"
    ] = (
        float(
            np.max(
                scaling_errors
            )
        )
        if len(
            scaling_errors
        )
        >
        0
        else 0.0
    )

    qa[
        "participation_scaling_identity_ok"
    ] = bool(
        qa[
            "max_participation_scaling_relative_error"
        ]
        <
        1e-10
    )

    # ========================================================
    # Governance
    # ========================================================

    qa[
        "future_liquidity_used"
    ] = False

    qa[
        "future_reopen_information_used"
    ] = False

    qa[
        "adv_winsorized"
    ] = False

    qa[
        "scenario_selected_from_results"
    ] = False

    qa[
        "factor_selection"
    ] = False

    qa[
        "window_selection"
    ] = False

    qa[
        "factor_sign_flip"
    ] = False

    qa[
        "transaction_cost_used_to_define_capacity"
    ] = False

    qa[
        "short_borrowability_assumed"
    ] = False

    qa[
        "multi_day_future_execution_assumed"
    ] = False

    # ========================================================
    # Formal PASS
    # ========================================================

    required_true = [
        "step3_formal_qa_pass",
        "step4_formal_qa_pass",
        "scenario_count_ok",
        "exactly_one_primary_scenario",
        "matrix_row_count_ok",
        "capacity_root_ok",
        "step4_primary_reproduced",
        "participation_monotonicity_ok",
        "target_stringency_monotonicity_ok",
        "participation_scaling_identity_ok",
    ]

    qa[
        "all_formal_qa_pass"
    ] = bool(

        all(
            qa[
                name
            ]
            for name in required_true
        )

        and

        qa[
            "duplicate_spec_scenario_count"
        ]
        ==
        0

        and

        qa[
            "primary_crosscheck_missing_step4_count"
        ]
        ==
        0

        and

        qa[
            "unclassified_factor_row_count"
        ]
        ==
        0
    )

    return qa


# ============================================================
# 23. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 8 - Step 5"
    )

    print(
        "Capacity Robustness Matrix"
    )

    print("=" * 80)

    # ========================================================
    # 1. Scenario design
    # ========================================================

    print()
    print(
        "[1] Build frozen robustness scenarios"
    )

    scenarios = (
        build_scenario_table()
    )

    print(
        f"Scenario count: "
        f"{len(scenarios)}"
    )

    print()

    print(
        scenarios[
            [
                "scenario_id",
                "liquidity_basis",
                "participation_cap",
                "target_executable_share",
                "is_primary",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 2. Upstream QA
    # ========================================================

    print()
    print(
        "[2] Validate frozen Step 3 and Step 4"
    )

    (
        step3_qa,
        step4_qa,
    ) = (
        validate_upstream()
    )

    print(
        "Step 3: PASS + FROZEN"
    )

    print(
        "Step 4: PASS + FROZEN"
    )

    # ========================================================
    # 3. Load stock panel
    # ========================================================

    print()
    print(
        "[3] Load minimal frozen Step-3 stock panel"
    )

    panel = (
        load_step3_panel()
    )

    print(
        f"Stock-level rows: "
        f"{len(panel):,}"
    )

    # ========================================================
    # 4. Precompute rebalance weights
    # ========================================================

    print()
    print(
        "[4] Precompute trade/rebalance weights"
    )

    (
        trade,
        leg_counts,
        pair_counts,
    ) = (
        prepare_trade_panel(
            panel
        )
    )

    print(
        f"Actual stock-trade rows: "
        f"{len(trade):,}"
    )

    print(
        f"Leg specifications: "
        f"{len(leg_counts):,}"
    )

    print(
        f"Pair specifications: "
        f"{len(pair_counts):,}"
    )

    # --------------------------------------------------------
    # Free the large full panel before expensive calculations.
    # --------------------------------------------------------

    del panel

    # ========================================================
    # 5. Robustness matrix
    # ========================================================

    print()
    print(
        "[5] Estimate complete robustness matrix"
    )

    matrix = (
        build_robustness_matrix(
            trade,
            leg_counts,
            pair_counts,
            scenarios,
        )
    )

    matrix = (
        add_primary_comparison(
            matrix
        )
    )

    matrix = matrix.sort_values(
        [
            "scope",
            "portfolio_scope",
            "factor_name",
            "window",
            "scenario_order",
        ]
    ).reset_index(
        drop=True
    )

    save_csv_atomic(
        matrix,
        ROBUSTNESS_MATRIX_PATH,
    )

    # ========================================================
    # 6. Wide table
    # ========================================================

    print()
    print(
        "[6] Build wide robustness matrix"
    )

    wide = (
        build_wide_matrix(
            matrix
        )
    )

    save_csv_atomic(
        wide,
        ROBUSTNESS_MATRIX_WIDE_PATH,
    )

    # ========================================================
    # 7. Primary cross-check
    # ========================================================

    print()
    print(
        "[7] Cross-check primary scenario against Step 4"
    )

    primary_crosscheck = (
        build_primary_crosscheck(
            matrix
        )
    )

    save_csv_atomic(
        primary_crosscheck,
        PRIMARY_CROSSCHECK_PATH,
    )

    # ========================================================
    # 8. Scenario summary
    # ========================================================

    print()
    print(
        "[8] Build scenario-level summaries"
    )

    scenario_summary = (
        build_scenario_summary(
            matrix
        )
    )

    save_csv_atomic(
        scenario_summary,
        SCENARIO_SUMMARY_PATH,
    )

    # ========================================================
    # 9. Family summary
    # ========================================================

    print()
    print(
        "[9] Build structural/dynamic summaries"
    )

    family_summary = (
        build_family_summary(
            matrix
        )
    )

    save_csv_atomic(
        family_summary,
        FAMILY_SUMMARY_PATH,
    )

    # ========================================================
    # 10. Zero-capacity scenarios
    # ========================================================

    zero_capacity = matrix[
        matrix[
            "capacity_cny"
        ]
        ==
        0
    ].copy()

    save_csv_atomic(
        zero_capacity,
        ZERO_CAPACITY_PATH,
    )

    # ========================================================
    # 11. Formal QA
    # ========================================================

    print()
    print(
        "[10] Formal Step-5 QA"
    )

    qa = (
        formal_qa(
            matrix,
            scenarios,
            primary_crosscheck,
            step3_qa,
            step4_qa,
        )
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
        QA_PATH,
    )

    print()
    print(
        "Key QA:"
    )

    print(
        "  Scenarios: "
        f"{qa['scenario_count']}"
    )

    print(
        "  Specifications: "
        f"{qa['specification_count']}"
    )

    print(
        "  Matrix rows: "
        f"{qa['matrix_row_count']:,}"
    )

    print(
        "  Max root error: "
        f"{qa['max_capacity_root_error']:.6e}"
    )

    print(
        "  Max Step-4 primary absolute difference: "
        f"{qa['max_primary_capacity_abs_difference_cny']:.6e} CNY"
    )

    print(
        "  Step-4 bisection fallback count: "
        f"{qa['step4_bisection_fallback_count']}"
    )

    print(
        "  Participation monotonicity violations: "
        f"{qa['participation_monotonicity_violation_count']}"
    )

    print(
        "  Target-stringency violations: "
        f"{qa['target_stringency_monotonicity_violation_count']}"
    )

    print(
        "  Formal QA PASS: "
        f"{qa['all_formal_qa_pass']}"
    )

    if not qa[
        "all_formal_qa_pass"
    ]:

        print()
        print(
            "STEP 5 QA FAILED."
        )

        print(
            "Do not interpret robustness results."
        )

        print()
        print(
            "See:"
        )

        print(
            QA_PATH
        )

        print(
            PRIMARY_CROSSCHECK_PATH
        )

        raise RuntimeError(
            "Step-5 formal QA failed."
        )

    # ========================================================
    # 12. Metadata
    # ========================================================

    print()
    print(
        "[11] Save metadata"
    )

    design = {

        "research_day":
            8,

        "step":
            "Step5_Capacity_Robustness_Matrix",

        "input_step3_panel":
            str(
                STEP3_PANEL_PATH
            ),

        "input_step3_qa":
            str(
                STEP3_QA_PATH
            ),

        "input_step4_capacity":
            str(
                STEP4_CAPACITY_PATH
            ),

        "input_step4_qa":
            str(
                STEP4_QA_PATH
            ),

        "primary_definition":
            {
                "liquidity_basis":
                    PRIMARY_LIQUIDITY_BASIS,

                "participation_cap":
                    PRIMARY_PARTICIPATION_CAP,

                "target_executable_share":
                    PRIMARY_TARGET_EXECUTABLE_SHARE,

                "execution_days":
                    EXECUTION_DAYS,
            },

        "robustness_dimensions":
            {
                "liquidity_bases":
                    LIQUIDITY_BASES,

                "participation_caps":
                    PARTICIPATION_CAPS,

                "target_executable_shares":
                    TARGET_EXECUTABLE_SHARES,

                "execution_days":
                    [
                        EXECUTION_DAYS
                    ],
            },

        "scenario_count":
            EXPECTED_SCENARIO_N,

        "scenario_selection":
            (
                "Full pre-specified factorial matrix. "
                "No scenario is selected from results."
            ),

        "capacity_definition":
            (
                "Supremum strategy NAV A such that the "
                "equal-rebalance mean executable trade-value "
                "share reaches the scenario-specific target "
                "under the scenario-specific ADV basis and "
                "name-level participation constraint."
            ),

        "solver":
            (
                "Exact piecewise capacity solution with "
                "frozen Step-4 bisection fallback and "
                "Step-4 primary-capacity cross-check."
            ),

        "execution_horizon_note":
            (
                "Execution horizon remains one day. "
                "No multi-day future reopening information "
                "or future volume is introduced."
            ),

        "future_liquidity_used":
            False,

        "future_reopen_information_used":
            False,

        "adv_winsorized":
            False,

        "factor_selection":
            False,

        "window_selection":
            False,

        "factor_sign_flip":
            False,

        "transaction_cost_used_to_define_capacity":
            False,

        "short_borrowability_assumed":
            False,
    }

    metadata = {

        **design,

        "design_hash":
            canonical_hash(
                design
            ),

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "formal_qa":
            qa,

        "outputs":
            {
                "robustness_matrix":
                    str(
                        ROBUSTNESS_MATRIX_PATH
                    ),

                "robustness_matrix_wide":
                    str(
                        ROBUSTNESS_MATRIX_WIDE_PATH
                    ),

                "scenario_summary":
                    str(
                        SCENARIO_SUMMARY_PATH
                    ),

                "family_summary":
                    str(
                        FAMILY_SUMMARY_PATH
                    ),

                "primary_crosscheck":
                    str(
                        PRIMARY_CROSSCHECK_PATH
                    ),

                "zero_capacity_detail":
                    str(
                        ZERO_CAPACITY_PATH
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
    # 13. Final output
    # ========================================================

    print()
    print("=" * 80)

    print(
        "DAY 8 STEP 5 COMPLETE"
    )

    print("=" * 80)

    print(
        f"Robustness scenarios : "
        f"{EXPECTED_SCENARIO_N}"
    )

    print(
        f"Specifications       : "
        f"{qa['specification_count']}"
    )

    print(
        f"Matrix rows          : "
        f"{len(matrix):,}"
    )

    print(
        f"Formal QA pass       : "
        f"{qa['all_formal_qa_pass']}"
    )

    print()
    print(
        "Primary output:"
    )

    print(
        ROBUSTNESS_MATRIX_PATH
    )

    print()
    print(
        "Wide matrix:"
    )

    print(
        ROBUSTNESS_MATRIX_WIDE_PATH
    )

    print()
    print(
        "Scenario summary:"
    )

    print(
        SCENARIO_SUMMARY_PATH
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


if __name__ == "__main__":

    main()