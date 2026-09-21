from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd


# ============================================================
# 0. Project paths
# ============================================================

ROOT = Path(
    r"D:\M1_StockNetwork"
)

OUTPUT_ROOT = (
    ROOT
    / "output"
)


# ------------------------------------------------------------
# Frozen Day-8 physical directory convention.
#
# Day 8 outputs are stored under M1_day8.
# ------------------------------------------------------------

DAY8_ROOT = (
    OUTPUT_ROOT
    / "M1_day8"
)


# ============================================================
# Frozen Step 3
# ============================================================

STEP3_DIR = (
    DAY8_ROOT
    / "03_stage3_participation_capacity"
)

STEP3_BASE_PANEL_PATH = (
    STEP3_DIR
    / "stock_participation_base_panel.parquet"
)

STEP3_QA_PATH = (
    STEP3_DIR
    / "participation_capacity_qa.csv"
)

STEP3_FACTOR_WINDOW_PATH = (
    STEP3_DIR
    / "participation_factor_window_summary.csv"
)


# ============================================================
# Step 4 output
# ============================================================

OUTPUT_DIR = (
    DAY8_ROOT
    / "04_stage4_capacity_limits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


LEG_REBALANCE_CAPACITY_PATH = (
    OUTPUT_DIR
    / "capacity_limit_rebalance_leg.csv"
)

PAIR_REBALANCE_CAPACITY_PATH = (
    OUTPUT_DIR
    / "capacity_limit_rebalance_pair.csv"
)

FACTOR_WINDOW_CAPACITY_PATH = (
    OUTPUT_DIR
    / "capacity_limit_factor_window.csv"
)

GRID_VALIDATION_PATH = (
    OUTPUT_DIR
    / "capacity_limit_grid_validation.csv"
)

ZERO_CAPACITY_DETAIL_PATH = (
    OUTPUT_DIR
    / "zero_capacity_rebalance_detail.csv"
)

QA_PATH = (
    OUTPUT_DIR
    / "capacity_limit_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day8_step4_metadata.json"
)


# ============================================================
# 1. Frozen Step-4 design
# ============================================================

# ------------------------------------------------------------
# PRIMARY participation constraint
# ------------------------------------------------------------

PRIMARY_LIQUIDITY_BASIS = "ADV20"

PRIMARY_PARTICIPATION_CAP = 0.05


# ------------------------------------------------------------
# Minimum required executable fraction
# ------------------------------------------------------------

TARGET_EXECUTABLE_SHARE = 0.95


# ------------------------------------------------------------
# Execution horizon
#
# One current market day.
#
# The effective ADV already incorporates:
#
# current suspension -> 0 immediate capacity.
# ------------------------------------------------------------

EXECUTION_DAYS = 1


# ------------------------------------------------------------
# Frozen Step-3 AUM grid.
#
# Used ONLY as a validation/bracketing diagnostic.
#
# Capacity itself is solved continuously from stock-level data.
# ------------------------------------------------------------

STEP3_AUM_GRID_CNY = [
    1e8,      # 1亿元
    5e8,      # 5亿元
    1e9,      # 10亿元
    2e9,      # 20亿元
    5e9,      # 50亿元
    1e10,     # 100亿元
]


TRADE_WEIGHT_TOL = 1e-15

ROOT_REL_TOL = 1e-11

ROOT_MAX_ITER = 120

QA_TOL = 1e-10


# ============================================================
# 2. Atomic I/O
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
        f"Cannot parse bool: {x}"
    )


# ============================================================
# 3. Validate frozen Step 3
# ============================================================

def validate_step3():

    if not STEP3_QA_PATH.exists():

        raise FileNotFoundError(
            "Frozen Step-3 QA not found:\n"
            f"{STEP3_QA_PATH}\n\n"
            "Do not run Step 4 before Step 3 is frozen."
        )

    qa = pd.read_csv(
        STEP3_QA_PATH
    )

    if not {
        "qa_name",
        "qa_value",
    }.issubset(
        qa.columns
    ):

        raise RuntimeError(
            "Unexpected Step-3 QA schema."
        )

    qa_map = dict(
        zip(
            qa[
                "qa_name"
            ],
            qa[
                "qa_value"
            ],
        )
    )

    required_true = [
        "all_formal_qa_pass",
        "canonical_step2_freeze_pass",
        "value_unit_confirmed",
        "normal_capacity_delta_unchanged",
        "cash_redeployment_identity_ok",
        "participation20_identity_ok",
    ]

    for name in required_true:

        if name not in qa_map:

            raise RuntimeError(
                f"Step-3 QA missing `{name}`."
            )

        if not parse_bool(
            qa_map[
                name
            ]
        ):

            raise RuntimeError(
                "Frozen Step-3 QA failed:\n"
                f"{name} = "
                f"{qa_map[name]}"
            )

    required_zero = [
        "positive_capacity_trade_unresolved_primary_liquidity_count",
        "positive_capacity_trade_missing_adv20_count",
        "positive_capacity_trade_missing_current_open_count",
    ]

    for name in required_zero:

        if name not in qa_map:

            raise RuntimeError(
                f"Step-3 QA missing `{name}`."
            )

        value = int(
            float(
                qa_map[
                    name
                ]
            )
        )

        if value != 0:

            raise RuntimeError(
                "Frozen Step-3 primary liquidity "
                "is incomplete:\n"
                f"{name} = {value}"
            )

    return qa_map


# ============================================================
# 4. Load frozen stock-level Step-3 panel
# ============================================================

def load_step3_panel():

    if not STEP3_BASE_PANEL_PATH.exists():

        raise FileNotFoundError(
            "Step-3 base panel not found:\n"
            f"{STEP3_BASE_PANEL_PATH}"
        )

    columns = [
        "analysis_date",
        "previous_analysis_date",
        "window",
        "factor_name",
        "method",
        "portfolio_leg",
        "security_id",
        "capacity_abs_trade_weight",
        "capacity_delta_weight",
        "effective_adv20_1d_cny",
        "current_is_open",
        "liquidity_source",
    ]

    df = pd.read_parquet(
        STEP3_BASE_PANEL_PATH,
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

    df[
        "capacity_abs_trade_weight"
    ] = pd.to_numeric(
        df[
            "capacity_abs_trade_weight"
        ],
        errors="raise",
    )

    df[
        "capacity_delta_weight"
    ] = pd.to_numeric(
        df[
            "capacity_delta_weight"
        ],
        errors="raise",
    )

    df[
        "effective_adv20_1d_cny"
    ] = pd.to_numeric(
        df[
            "effective_adv20_1d_cny"
        ],
        errors="coerce",
    )

    df[
        "current_is_open"
    ] = pd.to_numeric(
        df[
            "current_is_open"
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Only actual current-market trades need liquidity.
    # --------------------------------------------------------

    trade_mask = (
        df[
            "capacity_abs_trade_weight"
        ]
        >
        TRADE_WEIGHT_TOL
    )

    if (
        df.loc[
            trade_mask,
            "effective_adv20_1d_cny",
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Step-3 base panel still contains "
            "missing primary effective ADV20 "
            "for actual trades."
        )

    if (
        df.loc[
            trade_mask,
            "effective_adv20_1d_cny",
        ]
        <
        0
    ).any():

        raise RuntimeError(
            "Negative effective ADV20 detected."
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

    return df


# ============================================================
# 5. Capacity mathematics
#
# For trade i:
#
# required_i(A) = A * d_i
#
# executable_i(A) =
#     min(
#         A*d_i,
#         p*ADV_i*D
#     )
#
#
# Define:
#
# q_i = d_i / sum_j d_j
#
# c_i = p*ADV_i*D / d_i
#
#
# Then:
#
# G(A)
# =
# sum_i q_i * min(1, c_i/A)
#
# where c_i is the name-level maximum NAV before that
# individual trade exceeds the participation constraint.
# ============================================================

def coverage_from_breakpoints(
    aum_cny: float,
    q: np.ndarray,
    capacity_breakpoints: np.ndarray,
    constant_share: float = 0.0,
) -> float:

    q = np.asarray(
        q,
        dtype=float,
    )

    c = np.asarray(
        capacity_breakpoints,
        dtype=float,
    )

    if len(
        q
    ) == 0:

        return float(
            constant_share
        )

    if (
        aum_cny
        <=
        0
    ):

        # ----------------------------------------------------
        # Limit A -> 0+.
        #
        # c_i > 0:
        # eventually fully executable.
        #
        # c_i = 0:
        # never immediately executable.
        # ----------------------------------------------------

        return float(
            constant_share
            +
            q[
                c
                >
                0
            ].sum()
        )

    if np.isinf(
        aum_cny
    ):

        return float(
            constant_share
        )

    ratios = np.divide(
        c,
        aum_cny,
        out=np.zeros_like(
            c,
            dtype=float,
        ),
        where=np.ones_like(
            c,
            dtype=bool,
        ),
    )

    executable_fraction = np.minimum(
        1.0,
        ratios,
    )

    coverage = (
        constant_share
        +
        np.sum(
            q
            *
            executable_fraction
        )
    )

    return float(
        coverage
    )


# ============================================================
# 6. Solve continuous capacity limit
#
# Capacity:
#
# sup {
#       A:
#       coverage(A) >= TARGET
#     }
#
# No interpolation from the Step-3 AUM grid is used.
# ============================================================

def solve_capacity_limit(
    q: np.ndarray,
    capacity_breakpoints: np.ndarray,
    target: float,
    constant_share: float = 0.0,
):

    q = np.asarray(
        q,
        dtype=float,
    )

    c = np.asarray(
        capacity_breakpoints,
        dtype=float,
    )

    if len(
        q
    ) == 0:

        if (
            constant_share
            >=
            target
        ):

            return {
                "capacity_cny":
                    np.inf,

                "status":
                    "NO_TRADE_UNCONSTRAINED",

                "baseline_coverage":
                    float(
                        constant_share
                    ),

                "coverage_at_capacity":
                    float(
                        constant_share
                    ),

                "upper_bracket_cny":
                    np.inf,

                "iterations":
                    0,
            }

        return {
            "capacity_cny":
                0.0,

            "status":
                "NO_FEASIBLE_CAPACITY",

            "baseline_coverage":
                float(
                    constant_share
                ),

            "coverage_at_capacity":
                float(
                    constant_share
                ),

            "upper_bracket_cny":
                0.0,

            "iterations":
                0,
        }

    if (
        np.isnan(
            q
        ).any()
        or
        np.isnan(
            c
        ).any()
    ):

        raise RuntimeError(
            "NaN detected in capacity inputs."
        )

    if (
        q
        <
        0
    ).any():

        raise RuntimeError(
            "Negative normalized trade weights."
        )

    if (
        c
        <
        0
    ).any():

        raise RuntimeError(
            "Negative capacity breakpoints."
        )

    baseline = coverage_from_breakpoints(
        0.0,
        q,
        c,
        constant_share,
    )

    # --------------------------------------------------------
    # If even infinitesimal AUM cannot attain 95%,
    # the capacity is zero.
    #
    # Typical reason:
    #
    # >5% of required trade weight is currently suspended
    # or otherwise has zero immediate ADV.
    # --------------------------------------------------------

    if (
        baseline
        <
        target
        -
        1e-14
    ):

        return {
            "capacity_cny":
                0.0,

            "status":
                "ZERO_CAPACITY_BASELINE_BELOW_TARGET",

            "baseline_coverage":
                float(
                    baseline
                ),

            "coverage_at_capacity":
                float(
                    baseline
                ),

            "upper_bracket_cny":
                0.0,

            "iterations":
                0,
        }

    # --------------------------------------------------------
    # If the constant no-trade share alone reaches target,
    # capacity would be unbounded by this implementation test.
    #
    # This should not normally occur in our portfolios.
    # --------------------------------------------------------

    if (
        constant_share
        >=
        target
    ):

        return {
            "capacity_cny":
                np.inf,

            "status":
                "UNBOUNDED_BY_LIQUIDITY_TEST",

            "baseline_coverage":
                float(
                    baseline
                ),

            "coverage_at_capacity":
                float(
                    constant_share
                ),

            "upper_bracket_cny":
                np.inf,

            "iterations":
                0,
        }

    # --------------------------------------------------------
    # We have:
    #
    # coverage(A)
    #
    # <=
    #
    # constant
    # +
    # sum_i q_i c_i / A.
    #
    # Therefore the following is a valid upper bracket:
    #
    # high =
    # sum_i q_i c_i / (target - constant).
    # --------------------------------------------------------

    weighted_breakpoint_sum = float(
        np.sum(
            q
            *
            c
        )
    )

    denominator = (
        target
        -
        constant_share
    )

    if (
        weighted_breakpoint_sum
        <=
        0
    ):

        return {
            "capacity_cny":
                0.0,

            "status":
                "ZERO_CAPACITY_NO_POSITIVE_LIQUIDITY",

            "baseline_coverage":
                float(
                    baseline
                ),

            "coverage_at_capacity":
                float(
                    baseline
                ),

            "upper_bracket_cny":
                0.0,

            "iterations":
                0,
        }

    high = (
        weighted_breakpoint_sum
        /
        denominator
    )

    # --------------------------------------------------------
    # Numerical safeguard.
    # --------------------------------------------------------

    high_coverage = (
        coverage_from_breakpoints(
            high,
            q,
            c,
            constant_share,
        )
    )

    safeguard_n = 0

    while (
        high_coverage
        >
        target
        +
        1e-13
    ):

        high *= 2.0

        high_coverage = (
            coverage_from_breakpoints(
                high,
                q,
                c,
                constant_share,
            )
        )

        safeguard_n += 1

        if safeguard_n > 100:

            raise RuntimeError(
                "Unable to construct capacity upper bracket."
            )

    low = 0.0

    iterations = 0

    for iterations in range(
        1,
        ROOT_MAX_ITER + 1,
    ):

        mid = (
            low
            +
            high
        ) / 2.0

        coverage_mid = (
            coverage_from_breakpoints(
                mid,
                q,
                c,
                constant_share,
            )
        )

        if (
            coverage_mid
            >=
            target
        ):

            low = mid

        else:

            high = mid

        if (
            high
            -
            low
            <=
            ROOT_REL_TOL
            *
            max(
                high,
                1.0,
            )
        ):

            break

    capacity = float(
        low
    )

    coverage_at_capacity = (
        coverage_from_breakpoints(
            capacity,
            q,
            c,
            constant_share,
        )
    )

    return {
        "capacity_cny":
            capacity,

        "status":
            "PASS",

        "baseline_coverage":
            float(
                baseline
            ),

        "coverage_at_capacity":
            float(
                coverage_at_capacity
            ),

        "upper_bracket_cny":
            float(
                high
            ),

        "iterations":
            int(
                iterations
            ),
    }


# ============================================================
# 7. Convert a stock-trade group to q_i and c_i
# ============================================================

def make_rebalance_capacity_inputs(
    group: pd.DataFrame,
):

    trade = group[
        group[
            "capacity_abs_trade_weight"
        ]
        >
        TRADE_WEIGHT_TOL
    ].copy()

    if len(
        trade
    ) == 0:

        return {
            "q":
                np.array(
                    [],
                    dtype=float,
                ),

            "c":
                np.array(
                    [],
                    dtype=float,
                ),

            "total_trade_weight":
                0.0,

            "trade_name_n":
                0,

            "zero_adv_trade_weight_share":
                0.0,
        }

    d = (
        trade[
            "capacity_abs_trade_weight"
        ]
        .to_numpy(
            dtype=float
        )
    )

    adv = (
        trade[
            "effective_adv20_1d_cny"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if np.isnan(
        adv
    ).any():

        raise RuntimeError(
            "Primary ADV20 missing "
            "during Step-4 capacity estimation."
        )

    if (
        adv
        <
        0
    ).any():

        raise RuntimeError(
            "Negative effective ADV20."
        )

    total_d = float(
        d.sum()
    )

    if total_d <= 0:

        raise RuntimeError(
            "Invalid total trade weight."
        )

    q = (
        d
        /
        total_d
    )

    c = (
        PRIMARY_PARTICIPATION_CAP
        *
        adv
        *
        EXECUTION_DAYS
        /
        d
    )

    zero_adv_share = float(
        q[
            adv
            <=
            0
        ].sum()
    )

    return {
        "q":
            q,

        "c":
            c,

        "total_trade_weight":
            total_d,

        "trade_name_n":
            int(
                len(
                    trade
                )
            ),

        "zero_adv_trade_weight_share":
            zero_adv_share,
    }


# ============================================================
# 8. Rebalance-level capacity
# ============================================================

def estimate_rebalance_capacities(
    panel: pd.DataFrame,
    pair_mode: bool,
):

    if pair_mode:

        group_key = [
            "analysis_date",
            "previous_analysis_date",
            "window",
            "factor_name",
            "method",
        ]

        scope = (
            "HYPOTHETICAL_Q1_Q5_PAIR"
        )

        portfolio_scope = (
            "Q1_Q5_PAIR"
        )

    else:

        group_key = [
            "analysis_date",
            "previous_analysis_date",
            "window",
            "factor_name",
            "method",
            "portfolio_leg",
        ]

        scope = "PORTFOLIO_LEG"

        portfolio_scope = None

    groups = panel.groupby(
        group_key,
        sort=False,
        observed=True,
    )

    group_n = (
        groups.ngroups
    )

    rows = []

    for number, (
        keys,
        group,
    ) in enumerate(
        groups,
        start=1,
    ):

        if (
            number == 1
            or
            number % 500 == 0
            or
            number == group_n
        ):

            label = (
                "Pair"
                if pair_mode
                else
                "Leg"
            )

            print(
                f"{label} rebalance capacity: "
                f"{number:,}/"
                f"{group_n:,}"
            )

        base = dict(
            zip(
                group_key,
                keys,
            )
        )

        inputs = (
            make_rebalance_capacity_inputs(
                group
            )
        )

        result = (
            solve_capacity_limit(
                inputs[
                    "q"
                ],
                inputs[
                    "c"
                ],
                TARGET_EXECUTABLE_SHARE,
                constant_share=0.0,
            )
        )

        if pair_mode:

            current_portfolio_scope = (
                portfolio_scope
            )

        else:

            current_portfolio_scope = (
                base[
                    "portfolio_leg"
                ]
            )

        capacity_cny = (
            result[
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
        # Name-level NAV breakpoints:
        #
        # c_i =
        # p * ADV_i / |delta_w_i|.
        #
        # These are diagnostic only.
        # ----------------------------------------------------

        c = inputs[
            "c"
        ]

        positive_c = c[
            np.isfinite(
                c
            )
            &
            (
                c
                >
                0
            )
        ]

        if len(
            positive_c
        ) > 0:

            name_capacity_p10 = float(
                np.quantile(
                    positive_c,
                    0.10,
                )
            )

            name_capacity_p50 = float(
                np.quantile(
                    positive_c,
                    0.50,
                )
            )

            name_capacity_p90 = float(
                np.quantile(
                    positive_c,
                    0.90,
                )
            )

        else:

            name_capacity_p10 = np.nan

            name_capacity_p50 = np.nan

            name_capacity_p90 = np.nan

        row = {
            **base,

            "scope":
                scope,

            "portfolio_scope":
                current_portfolio_scope,

            "liquidity_basis":
                PRIMARY_LIQUIDITY_BASIS,

            "participation_cap":
                PRIMARY_PARTICIPATION_CAP,

            "target_executable_share":
                TARGET_EXECUTABLE_SHARE,

            "execution_days":
                EXECUTION_DAYS,

            "trade_name_n":
                inputs[
                    "trade_name_n"
                ],

            "total_abs_trade_weight":
                inputs[
                    "total_trade_weight"
                ],

            "zero_adv_trade_weight_share":
                inputs[
                    "zero_adv_trade_weight_share"
                ],

            "capacity_status":
                result[
                    "status"
                ],

            "capacity_cny":
                capacity_cny,

            "capacity_yi_cny":
                capacity_yi,

            "baseline_executable_share":
                result[
                    "baseline_coverage"
                ],

            "coverage_at_capacity":
                result[
                    "coverage_at_capacity"
                ],

            "root_upper_bracket_cny":
                result[
                    "upper_bracket_cny"
                ],

            "root_iterations":
                result[
                    "iterations"
                ],

            "name_capacity_p10_cny":
                name_capacity_p10,

            "name_capacity_p50_cny":
                name_capacity_p50,

            "name_capacity_p90_cny":
                name_capacity_p90,
        }

        # ----------------------------------------------------
        # For a Q1-Q5 pair:
        #
        # A is strategy NAV / each-leg notional.
        #
        # Gross notional = 2A.
        # ----------------------------------------------------

        if pair_mode:

            row[
                "gross_notional_at_capacity_cny"
            ] = (
                2.0
                *
                capacity_cny
                if np.isfinite(
                    capacity_cny
                )
                else np.inf
            )

        else:

            row[
                "gross_notional_at_capacity_cny"
            ] = (
                capacity_cny
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 9. Build mean-coverage Factor × Window capacity
#
# This is the PRIMARY Step-4 result.
#
#
# For T rebalances:
#
#   G_bar(A)
#   =
#   1/T * sum_t G_t(A).
#
#
# Within each rebalance:
#
#   q_it =
#   d_it / sum_j d_j.
#
#
# Therefore:
#
#   G_bar(A)
#   =
#   sum_{t,i}
#       [q_it / T]
#       min(1, c_it/A).
#
#
# So the same monotone solver can be used directly.
# ============================================================

def make_factor_window_inputs(
    group: pd.DataFrame,
    rebalance_columns,
):

    all_rebalances = (
        group[
            rebalance_columns
        ]
        .drop_duplicates()
    )

    total_rebalance_n = int(
        len(
            all_rebalances
        )
    )

    if total_rebalance_n == 0:

        raise RuntimeError(
            "No rebalance dates in factor-window group."
        )

    trade = group[
        group[
            "capacity_abs_trade_weight"
        ]
        >
        TRADE_WEIGHT_TOL
    ].copy()

    if len(
        trade
    ) == 0:

        return {
            "q":
                np.array(
                    [],
                    dtype=float,
                ),

            "c":
                np.array(
                    [],
                    dtype=float,
                ),

            "constant_share":
                1.0,

            "rebalance_n":
                total_rebalance_n,

            "no_trade_rebalance_n":
                total_rebalance_n,
        }

    # --------------------------------------------------------
    # Total trade weight within each rebalance.
    # --------------------------------------------------------

    trade[
        "_rebalance_total_trade_weight"
    ] = (

        trade.groupby(
            rebalance_columns,
            observed=True,
        )[
            "capacity_abs_trade_weight"
        ]
        .transform(
            "sum"
        )
    )

    trade_rebalance_n = int(
        trade[
            rebalance_columns
        ]
        .drop_duplicates()
        .shape[0]
    )

    no_trade_rebalance_n = (
        total_rebalance_n
        -
        trade_rebalance_n
    )

    constant_share = (
        no_trade_rebalance_n
        /
        total_rebalance_n
    )

    d = (
        trade[
            "capacity_abs_trade_weight"
        ]
        .to_numpy(
            dtype=float
        )
    )

    rebalance_total = (
        trade[
            "_rebalance_total_trade_weight"
        ]
        .to_numpy(
            dtype=float
        )
    )

    adv = (
        trade[
            "effective_adv20_1d_cny"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if np.isnan(
        adv
    ).any():

        raise RuntimeError(
            "Primary ADV20 missing "
            "in factor-window capacity."
        )

    # --------------------------------------------------------
    # Each rebalance receives equal weight 1/T.
    # --------------------------------------------------------

    q = (
        d
        /
        rebalance_total
        /
        total_rebalance_n
    )

    c = (
        PRIMARY_PARTICIPATION_CAP
        *
        adv
        *
        EXECUTION_DAYS
        /
        d
    )

    return {
        "q":
            q,

        "c":
            c,

        "constant_share":
            float(
                constant_share
            ),

        "rebalance_n":
            total_rebalance_n,

        "no_trade_rebalance_n":
            int(
                no_trade_rebalance_n
            ),
    }


# ============================================================
# 10. Fixed-AUM grid coverage from stock-level data
# ============================================================

def build_grid_rows(
    base,
    q,
    c,
    constant_share,
):

    rows = []

    for aum in (
        STEP3_AUM_GRID_CNY
    ):

        coverage = (
            coverage_from_breakpoints(
                aum,
                q,
                c,
                constant_share,
            )
        )

        rows.append(
            {
                **base,

                "strategy_nav_cny":
                    float(
                        aum
                    ),

                "recomputed_mean_executable_share":
                    float(
                        coverage
                    ),
            }
        )

    return rows


# ============================================================
# 11. Factor × Window capacity table
# ============================================================

def estimate_factor_window_capacity(
    panel: pd.DataFrame,
    leg_rebalance_capacity: pd.DataFrame,
    pair_rebalance_capacity: pd.DataFrame,
):

    result_rows = []

    grid_rows = []

    # ========================================================
    # Leg specs
    # ========================================================

    leg_spec_key = [
        "factor_name",
        "method",
        "window",
        "portfolio_leg",
    ]

    leg_rebalance_cols = [
        "analysis_date",
        "previous_analysis_date",
    ]

    leg_groups = panel.groupby(
        leg_spec_key,
        sort=False,
        observed=True,
    )

    for keys, group in leg_groups:

        spec = dict(
            zip(
                leg_spec_key,
                keys,
            )
        )

        portfolio_scope = (
            spec[
                "portfolio_leg"
            ]
        )

        inputs = (
            make_factor_window_inputs(
                group,
                leg_rebalance_cols,
            )
        )

        solved = (
            solve_capacity_limit(
                inputs[
                    "q"
                ],
                inputs[
                    "c"
                ],
                TARGET_EXECUTABLE_SHARE,
                inputs[
                    "constant_share"
                ],
            )
        )

        base = {
            "scope":
                "PORTFOLIO_LEG",

            "portfolio_scope":
                portfolio_scope,

            "factor_name":
                spec[
                    "factor_name"
                ],

            "method":
                spec[
                    "method"
                ],

            "window":
                spec[
                    "window"
                ],

            "liquidity_basis":
                PRIMARY_LIQUIDITY_BASIS,

            "participation_cap":
                PRIMARY_PARTICIPATION_CAP,

            "target_executable_share":
                TARGET_EXECUTABLE_SHARE,

            "execution_days":
                EXECUTION_DAYS,
        }

        # ----------------------------------------------------
        # Rebalance-capacity distribution
        # ----------------------------------------------------

        reb = leg_rebalance_capacity[
            (
                leg_rebalance_capacity[
                    "factor_name"
                ]
                ==
                spec[
                    "factor_name"
                ]
            )
            &
            (
                leg_rebalance_capacity[
                    "method"
                ]
                ==
                spec[
                    "method"
                ]
            )
            &
            (
                leg_rebalance_capacity[
                    "window"
                ]
                ==
                spec[
                    "window"
                ]
            )
            &
            (
                leg_rebalance_capacity[
                    "portfolio_scope"
                ]
                ==
                portfolio_scope
            )
        ].copy()

        result_rows.append(
            build_factor_window_result_row(
                base,
                inputs,
                solved,
                reb,
            )
        )

        grid_rows.extend(
            build_grid_rows(
                base,
                inputs[
                    "q"
                ],
                inputs[
                    "c"
                ],
                inputs[
                    "constant_share"
                ],
            )
        )

    # ========================================================
    # Pair specs
    # ========================================================

    pair_spec_key = [
        "factor_name",
        "method",
        "window",
    ]

    pair_rebalance_cols = [
        "analysis_date",
        "previous_analysis_date",
    ]

    pair_groups = panel.groupby(
        pair_spec_key,
        sort=False,
        observed=True,
    )

    for keys, group in pair_groups:

        spec = dict(
            zip(
                pair_spec_key,
                keys,
            )
        )

        inputs = (
            make_factor_window_inputs(
                group,
                pair_rebalance_cols,
            )
        )

        solved = (
            solve_capacity_limit(
                inputs[
                    "q"
                ],
                inputs[
                    "c"
                ],
                TARGET_EXECUTABLE_SHARE,
                inputs[
                    "constant_share"
                ],
            )
        )

        base = {
            "scope":
                "HYPOTHETICAL_Q1_Q5_PAIR",

            "portfolio_scope":
                "Q1_Q5_PAIR",

            "factor_name":
                spec[
                    "factor_name"
                ],

            "method":
                spec[
                    "method"
                ],

            "window":
                spec[
                    "window"
                ],

            "liquidity_basis":
                PRIMARY_LIQUIDITY_BASIS,

            "participation_cap":
                PRIMARY_PARTICIPATION_CAP,

            "target_executable_share":
                TARGET_EXECUTABLE_SHARE,

            "execution_days":
                EXECUTION_DAYS,
        }

        reb = pair_rebalance_capacity[
            (
                pair_rebalance_capacity[
                    "factor_name"
                ]
                ==
                spec[
                    "factor_name"
                ]
            )
            &
            (
                pair_rebalance_capacity[
                    "method"
                ]
                ==
                spec[
                    "method"
                ]
            )
            &
            (
                pair_rebalance_capacity[
                    "window"
                ]
                ==
                spec[
                    "window"
                ]
            )
        ].copy()

        result_rows.append(
            build_factor_window_result_row(
                base,
                inputs,
                solved,
                reb,
            )
        )

        grid_rows.extend(
            build_grid_rows(
                base,
                inputs[
                    "q"
                ],
                inputs[
                    "c"
                ],
                inputs[
                    "constant_share"
                ],
            )
        )

    return (
        pd.DataFrame(
            result_rows
        ),
        pd.DataFrame(
            grid_rows
        ),
    )


# ============================================================
# 12. Factor-window output row
# ============================================================

def build_factor_window_result_row(
    base,
    inputs,
    solved,
    rebalance_capacity,
):

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

    capacities = pd.to_numeric(
        rebalance_capacity[
            "capacity_cny"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    finite = capacities[
        np.isfinite(
            capacities
        )
    ]

    zero_share = float(
        np.mean(
            capacities
            ==
            0
        )
    ) if len(
        capacities
    ) > 0 else np.nan

    infinite_share = float(
        np.mean(
            np.isposinf(
                capacities
            )
        )
    ) if len(
        capacities
    ) > 0 else np.nan

    if len(
        finite
    ) > 0:

        reb_p10 = float(
            np.quantile(
                finite,
                0.10,
            )
        )

        reb_p25 = float(
            np.quantile(
                finite,
                0.25,
            )
        )

        reb_p50 = float(
            np.quantile(
                finite,
                0.50,
            )
        )

        reb_p75 = float(
            np.quantile(
                finite,
                0.75,
            )
        )

        reb_p90 = float(
            np.quantile(
                finite,
                0.90,
            )
        )

        reb_mean = float(
            np.mean(
                finite
            )
        )

    else:

        reb_p10 = np.nan
        reb_p25 = np.nan
        reb_p50 = np.nan
        reb_p75 = np.nan
        reb_p90 = np.nan
        reb_mean = np.nan

    # --------------------------------------------------------
    # At aggregate capacity:
    #
    # share of individual rebalances that themselves still
    # satisfy the 95% criterion.
    # --------------------------------------------------------

    if (
        len(
            capacities
        )
        >
        0
        and
        np.isfinite(
            capacity_cny
        )
    ):

        individual_pass_share = float(
            np.mean(
                capacities
                >=
                capacity_cny
            )
        )

    else:

        individual_pass_share = np.nan

    row = {
        **base,

        "rebalance_n":
            int(
                inputs[
                    "rebalance_n"
                ]
            ),

        "no_trade_rebalance_n":
            int(
                inputs[
                    "no_trade_rebalance_n"
                ]
            ),

        "capacity_status":
            solved[
                "status"
            ],

        # ----------------------------------------------------
        # PRIMARY Step-4 capacity.
        # ----------------------------------------------------

        "mean_coverage_capacity_cny":
            capacity_cny,

        "mean_coverage_capacity_yi_cny":
            capacity_yi,

        "baseline_mean_executable_share":
            solved[
                "baseline_coverage"
            ],

        "mean_executable_share_at_capacity":
            solved[
                "coverage_at_capacity"
            ],

        "root_upper_bracket_cny":
            solved[
                "upper_bracket_cny"
            ],

        "root_iterations":
            solved[
                "iterations"
            ],

        # ----------------------------------------------------
        # Distribution of rebalance-level capacity.
        # ----------------------------------------------------

        "rebalance_capacity_zero_share":
            zero_share,

        "rebalance_capacity_infinite_share":
            infinite_share,

        "rebalance_capacity_mean_cny_finite":
            reb_mean,

        "rebalance_capacity_p10_cny_finite":
            reb_p10,

        "rebalance_capacity_p25_cny_finite":
            reb_p25,

        "rebalance_capacity_p50_cny_finite":
            reb_p50,

        "rebalance_capacity_p75_cny_finite":
            reb_p75,

        "rebalance_capacity_p90_cny_finite":
            reb_p90,

        "share_rebalances_meeting_95pct_at_aggregate_capacity":
            individual_pass_share,
    }

    if (
        base[
            "scope"
        ]
        ==
        "HYPOTHETICAL_Q1_Q5_PAIR"
    ):

        row[
            "gross_notional_at_capacity_cny"
        ] = (
            2.0
            *
            capacity_cny
            if np.isfinite(
                capacity_cny
            )
            else np.inf
        )

    else:

        row[
            "gross_notional_at_capacity_cny"
        ] = (
            capacity_cny
        )

    return row


# ============================================================
# 13. Compare Step-4 fixed-grid coverage to frozen Step 3
# ============================================================

def validate_against_step3_grid(
    recomputed_grid: pd.DataFrame,
):

    if not STEP3_FACTOR_WINDOW_PATH.exists():

        raise FileNotFoundError(
            "Frozen Step-3 factor-window summary "
            "not found:\n"
            f"{STEP3_FACTOR_WINDOW_PATH}"
        )

    step3 = pd.read_csv(
        STEP3_FACTOR_WINDOW_PATH
    )

    required_columns = [
        "scope",
        "portfolio_scope",
        "factor_name",
        "method",
        "window",
        "strategy_nav_cny",
        "liquidity_basis",
        "mean_executable_trade_value_share_at_5pct",
    ]

    missing = [
        column
        for column in required_columns
        if column not in step3.columns
    ]

    if missing:

        raise RuntimeError(
            "Step-3 factor-window summary "
            "is missing columns:\n"
            f"{missing}"
        )

    step3[
        "window"
    ] = pd.to_numeric(
        step3[
            "window"
        ],
        errors="raise",
    ).astype(int)

    step3[
        "strategy_nav_cny"
    ] = pd.to_numeric(
        step3[
            "strategy_nav_cny"
        ],
        errors="raise",
    )

    step3 = step3[
        step3[
            "liquidity_basis"
        ]
        ==
        PRIMARY_LIQUIDITY_BASIS
    ].copy()

    merge_key = [
        "scope",
        "portfolio_scope",
        "factor_name",
        "method",
        "window",
        "strategy_nav_cny",
    ]

    if step3[
        merge_key
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Step-3 fixed-grid keys."
        )

    comparison = (
        recomputed_grid.merge(
            step3[
                merge_key
                +
                [
                    "mean_executable_trade_value_share_at_5pct",
                ]
            ],
            on=merge_key,
            how="left",
            validate="one_to_one",
        )
    )

    comparison[
        "abs_difference"
    ] = (

        comparison[
            "recomputed_mean_executable_share"
        ]

        -

        comparison[
            "mean_executable_trade_value_share_at_5pct"
        ]
    ).abs()

    return comparison


# ============================================================
# 14. Add frozen-grid capacity brackets
# ============================================================

def add_grid_brackets(
    factor_window_capacity: pd.DataFrame,
    grid_validation: pd.DataFrame,
):

    rows = []

    spec_key = [
        "scope",
        "portfolio_scope",
        "factor_name",
        "method",
        "window",
    ]

    grouped = grid_validation.groupby(
        spec_key,
        sort=False,
        observed=True,
    )

    for keys, group in grouped:

        group = group.sort_values(
            "strategy_nav_cny"
        )

        passes = group[
            group[
                "recomputed_mean_executable_share"
            ]
            >=
            TARGET_EXECUTABLE_SHARE
        ]

        fails = group[
            group[
                "recomputed_mean_executable_share"
            ]
            <
            TARGET_EXECUTABLE_SHARE
        ]

        if len(
            passes
        ) > 0:

            lower = float(
                passes[
                    "strategy_nav_cny"
                ]
                .max()
            )

        else:

            lower = np.nan

        if len(
            fails
        ) > 0:

            if np.isfinite(
                lower
            ):

                fail_above = fails[
                    fails[
                        "strategy_nav_cny"
                    ]
                    >
                    lower
                ]

                if len(
                    fail_above
                ) > 0:

                    upper = float(
                        fail_above[
                            "strategy_nav_cny"
                        ]
                        .min()
                    )

                else:

                    upper = np.nan

            else:

                upper = float(
                    fails[
                        "strategy_nav_cny"
                    ]
                    .min()
                )

        else:

            upper = np.nan

        row = dict(
            zip(
                spec_key,
                keys,
            )
        )

        row[
            "step3_grid_lower_pass_cny"
        ] = lower

        row[
            "step3_grid_upper_fail_cny"
        ] = upper

        rows.append(
            row
        )

    bracket = pd.DataFrame(
        rows
    )

    result = (
        factor_window_capacity.merge(
            bracket,
            on=spec_key,
            how="left",
            validate="one_to_one",
        )
    )

    capacity = (
        result[
            "mean_coverage_capacity_cny"
        ]
    )

    lower = (
        result[
            "step3_grid_lower_pass_cny"
        ]
    )

    upper = (
        result[
            "step3_grid_upper_fail_cny"
        ]
    )

    within_lower = (
        lower.isna()
        |
        (
            capacity
            >=
            lower
            -
            1e-6
        )
    )

    within_upper = (
        upper.isna()
        |
        (
            capacity
            <=
            upper
            +
            1e-6
        )
    )

    result[
        "capacity_within_step3_grid_bracket"
    ] = (
        within_lower
        &
        within_upper
    )

    return result


# ============================================================
# 15. Formal Step-4 QA
# ============================================================

def formal_qa(
    step3_qa,
    panel,
    leg_capacity,
    pair_capacity,
    factor_window_capacity,
    grid_validation,
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
        "primary_liquidity_basis_is_adv20"
    ] = bool(
        PRIMARY_LIQUIDITY_BASIS
        ==
        "ADV20"
    )

    qa[
        "participation_cap_is_5pct"
    ] = bool(
        np.isclose(
            PRIMARY_PARTICIPATION_CAP,
            0.05,
        )
    )

    qa[
        "target_executable_share_is_95pct"
    ] = bool(
        np.isclose(
            TARGET_EXECUTABLE_SHARE,
            0.95,
        )
    )

    qa[
        "execution_days_is_one"
    ] = bool(
        EXECUTION_DAYS
        ==
        1
    )

    # ========================================================
    # Group counts
    # ========================================================

    qa[
        "leg_rebalance_capacity_row_count"
    ] = int(
        len(
            leg_capacity
        )
    )

    qa[
        "pair_rebalance_capacity_row_count"
    ] = int(
        len(
            pair_capacity
        )
    )

    expected_leg_n = int(
        float(
            step3_qa[
                "canonical_trade_leg_group_count"
            ]
        )
    )

    expected_pair_n = int(
        float(
            step3_qa[
                "canonical_trade_pair_group_count"
            ]
        )
    )

    qa[
        "expected_leg_rebalance_count"
    ] = expected_leg_n

    qa[
        "expected_pair_rebalance_count"
    ] = expected_pair_n

    qa[
        "leg_rebalance_count_matches_step3"
    ] = bool(
        len(
            leg_capacity
        )
        ==
        expected_leg_n
    )

    qa[
        "pair_rebalance_count_matches_step3"
    ] = bool(
        len(
            pair_capacity
        )
        ==
        expected_pair_n
    )

    # ========================================================
    # No primary liquidity missing
    # ========================================================

    trade_mask = (
        panel[
            "capacity_abs_trade_weight"
        ]
        >
        TRADE_WEIGHT_TOL
    )

    qa[
        "primary_effective_adv_missing_trade_count"
    ] = int(
        panel.loc[
            trade_mask,
            "effective_adv20_1d_cny",
        ]
        .isna()
        .sum()
    )

    # ========================================================
    # No-trade groups
    # ========================================================

    qa[
        "leg_no_trade_rebalance_count"
    ] = int(
        (
            leg_capacity[
                "capacity_status"
            ]
            ==
            "NO_TRADE_UNCONSTRAINED"
        )
        .sum()
    )

    qa[
        "pair_no_trade_rebalance_count"
    ] = int(
        (
            pair_capacity[
                "capacity_status"
            ]
            ==
            "NO_TRADE_UNCONSTRAINED"
        )
        .sum()
    )

    # ========================================================
    # Zero-capacity diagnostics
    #
    # Zero capacity is NOT a QA failure.
    #
    # It means baseline immediate executable share <95%,
    # typically due to current suspensions.
    # ========================================================

    qa[
        "leg_zero_capacity_rebalance_count"
    ] = int(
        (
            leg_capacity[
                "capacity_cny"
            ]
            ==
            0
        )
        .sum()
    )

    qa[
        "pair_zero_capacity_rebalance_count"
    ] = int(
        (
            pair_capacity[
                "capacity_cny"
            ]
            ==
            0
        )
        .sum()
    )

    # ========================================================
    # Root accuracy
    # ========================================================

    fw = factor_window_capacity.copy()

    finite_positive = (
        np.isfinite(
            fw[
                "mean_coverage_capacity_cny"
            ]
        )
        &
        (
            fw[
                "mean_coverage_capacity_cny"
            ]
            >
            0
        )
        &
        (
            fw[
                "capacity_status"
            ]
            ==
            "PASS"
        )
    )

    root_error = (

        fw.loc[
            finite_positive,
            "mean_executable_share_at_capacity",
        ]

        -

        TARGET_EXECUTABLE_SHARE
    ).abs()

    qa[
        "max_factor_window_capacity_root_error"
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
        "factor_window_capacity_root_ok"
    ] = bool(
        qa[
            "max_factor_window_capacity_root_error"
        ]
        <
        1e-8
    )

    # ========================================================
    # Step-3 fixed-grid reproduction
    # ========================================================

    qa[
        "grid_validation_missing_step3_count"
    ] = int(
        grid_validation[
            "mean_executable_trade_value_share_at_5pct"
        ]
        .isna()
        .sum()
    )

    qa[
        "max_step3_grid_executable_share_difference"
    ] = float(
        grid_validation[
            "abs_difference"
        ]
        .max()
    )

    qa[
        "step3_grid_reproduced"
    ] = bool(
        np.allclose(
            grid_validation[
                "recomputed_mean_executable_share"
            ]
            .to_numpy(
                dtype=float
            ),

            grid_validation[
                "mean_executable_trade_value_share_at_5pct"
            ]
            .to_numpy(
                dtype=float
            ),

            rtol=1e-10,
            atol=1e-10,
            equal_nan=False,
        )
    )

    # ========================================================
    # Monotonicity over frozen AUM grid
    # ========================================================

    spec_key = [
        "scope",
        "portfolio_scope",
        "factor_name",
        "method",
        "window",
    ]

    monotonic_violation_n = 0

    for _, group in (
        grid_validation.groupby(
            spec_key,
            sort=False,
            observed=True,
        )
    ):

        values = (
            group.sort_values(
                "strategy_nav_cny"
            )[
                "recomputed_mean_executable_share"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if (
            np.diff(
                values
            )
            >
            1e-12
        ).any():

            monotonic_violation_n += 1

    qa[
        "capacity_grid_monotonicity_violation_count"
    ] = int(
        monotonic_violation_n
    )

    qa[
        "capacity_grid_monotonicity_ok"
    ] = bool(
        monotonic_violation_n
        ==
        0
    )

    # ========================================================
    # Exact capacity should lie inside frozen Step-3 bracket
    # whenever a finite grid bracket exists.
    # ========================================================

    qa[
        "factor_window_capacity_outside_grid_bracket_count"
    ] = int(
        (
            ~factor_window_capacity[
                "capacity_within_step3_grid_bracket"
            ]
        )
        .sum()
    )

    qa[
        "factor_window_capacity_within_grid_brackets"
    ] = bool(
        qa[
            "factor_window_capacity_outside_grid_bracket_count"
        ]
        ==
        0
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
        "capacity_threshold_selected_from_results"
    ] = False

    qa[
        "participation_cap_selected_from_results"
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
        "capacity_estimated_by_grid_interpolation"
    ] = False

    # ========================================================
    # Formal PASS
    # ========================================================

    required_true = [
        "step3_formal_qa_pass",
        "primary_liquidity_basis_is_adv20",
        "participation_cap_is_5pct",
        "target_executable_share_is_95pct",
        "execution_days_is_one",
        "leg_rebalance_count_matches_step3",
        "pair_rebalance_count_matches_step3",
        "factor_window_capacity_root_ok",
        "step3_grid_reproduced",
        "capacity_grid_monotonicity_ok",
        "factor_window_capacity_within_grid_brackets",
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
            "primary_effective_adv_missing_trade_count"
        ]
        ==
        0

        and

        qa[
            "grid_validation_missing_step3_count"
        ]
        ==
        0
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
        "Day 8 - Step 4"
    )

    print(
        "Capacity Limit Estimation"
    )

    print("=" * 80)

    print()
    print(
        "Primary definition:"
    )

    print(
        f"  Liquidity basis       : "
        f"{PRIMARY_LIQUIDITY_BASIS}"
    )

    print(
        f"  Participation cap     : "
        f"{PRIMARY_PARTICIPATION_CAP:.1%}"
    )

    print(
        f"  Executable-share floor: "
        f"{TARGET_EXECUTABLE_SHARE:.1%}"
    )

    print(
        f"  Execution horizon     : "
        f"{EXECUTION_DAYS} day"
    )

    # ========================================================
    # 1. Validate Step 3
    # ========================================================

    print()
    print(
        "[1] Validate frozen Step 3"
    )

    step3_qa = (
        validate_step3()
    )

    print(
        "Step-3 formal QA: PASS"
    )

    # ========================================================
    # 2. Load stock-level base
    # ========================================================

    print()
    print(
        "[2] Load frozen Step-3 stock participation panel"
    )

    panel = (
        load_step3_panel()
    )

    print(
        "Stock-level rows: "
        f"{len(panel):,}"
    )

    # ========================================================
    # 3. Rebalance-level leg capacity
    # ========================================================

    print()
    print(
        "[3] Estimate rebalance-level Q1/Q5 capacities"
    )

    leg_capacity = (
        estimate_rebalance_capacities(
            panel,
            pair_mode=False,
        )
    )

    save_csv_atomic(
        leg_capacity,
        LEG_REBALANCE_CAPACITY_PATH,
    )

    # ========================================================
    # 4. Rebalance-level pair capacity
    # ========================================================

    print()
    print(
        "[4] Estimate rebalance-level "
        "hypothetical Q1-Q5 capacities"
    )

    pair_capacity = (
        estimate_rebalance_capacities(
            panel,
            pair_mode=True,
        )
    )

    save_csv_atomic(
        pair_capacity,
        PAIR_REBALANCE_CAPACITY_PATH,
    )

    # ========================================================
    # 5. Factor × Window capacities
    # ========================================================

    print()
    print(
        "[5] Estimate Factor × Window "
        "mean-coverage capacity limits"
    )

    (
        factor_window_capacity,
        recomputed_grid,
    ) = (
        estimate_factor_window_capacity(
            panel,
            leg_capacity,
            pair_capacity,
        )
    )

    # ========================================================
    # 6. Reproduce frozen Step-3 grid
    # ========================================================

    print()
    print(
        "[6] Validate against frozen Step-3 AUM grid"
    )

    grid_validation = (
        validate_against_step3_grid(
            recomputed_grid
        )
    )

    save_csv_atomic(
        grid_validation,
        GRID_VALIDATION_PATH,
    )

    # ========================================================
    # 7. Add Step-3 brackets
    # ========================================================

    factor_window_capacity = (
        add_grid_brackets(
            factor_window_capacity,
            grid_validation,
        )
    )

    save_csv_atomic(
        factor_window_capacity,
        FACTOR_WINDOW_CAPACITY_PATH,
    )

    # ========================================================
    # 8. Zero-capacity diagnostics
    # ========================================================

    zero_leg = (
        leg_capacity[
            leg_capacity[
                "capacity_cny"
            ]
            ==
            0
        ]
        .copy()
    )

    zero_leg[
        "rebalance_type"
    ] = "PORTFOLIO_LEG"

    zero_pair = (
        pair_capacity[
            pair_capacity[
                "capacity_cny"
            ]
            ==
            0
        ]
        .copy()
    )

    zero_pair[
        "rebalance_type"
    ] = "Q1_Q5_PAIR"

    zero_detail = pd.concat(
        [
            zero_leg,
            zero_pair,
        ],
        ignore_index=True,
        sort=False,
    )

    save_csv_atomic(
        zero_detail,
        ZERO_CAPACITY_DETAIL_PATH,
    )

    # ========================================================
    # 9. Formal QA
    # ========================================================

    print()
    print(
        "[7] Formal Step-4 QA"
    )

    qa = formal_qa(
        step3_qa,
        panel,
        leg_capacity,
        pair_capacity,
        factor_window_capacity,
        grid_validation,
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
        "  Step-3 grid max difference: "
        f"{qa['max_step3_grid_executable_share_difference']:.6e}"
    )

    print(
        "  Capacity-root max error: "
        f"{qa['max_factor_window_capacity_root_error']:.6e}"
    )

    print(
        "  Grid monotonicity violations: "
        f"{qa['capacity_grid_monotonicity_violation_count']}"
    )

    print(
        "  Capacity outside Step-3 brackets: "
        f"{qa['factor_window_capacity_outside_grid_bracket_count']}"
    )

    print(
        "  Zero-capacity leg rebalances: "
        f"{qa['leg_zero_capacity_rebalance_count']:,}"
    )

    print(
        "  Zero-capacity pair rebalances: "
        f"{qa['pair_zero_capacity_rebalance_count']:,}"
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
            "Step-4 QA FAILED."
        )

        print(
            "Do not interpret capacity estimates."
        )

        print()
        print(
            "See:"
        )

        print(
            QA_PATH
        )

        print(
            GRID_VALIDATION_PATH
        )

        raise RuntimeError(
            "Step-4 formal QA failed."
        )

    # ========================================================
    # 10. Metadata
    # ========================================================

    print()
    print(
        "[8] Save metadata"
    )

    design = {

        "research_day":
            8,

        "step":
            "Step4_Capacity_Limit_Estimation",

        "input_step3_panel":
            str(
                STEP3_BASE_PANEL_PATH
            ),

        "input_step3_qa":
            str(
                STEP3_QA_PATH
            ),

        "primary_liquidity_basis":
            PRIMARY_LIQUIDITY_BASIS,

        "participation_cap":
            PRIMARY_PARTICIPATION_CAP,

        "target_executable_share":
            TARGET_EXECUTABLE_SHARE,

        "execution_days":
            EXECUTION_DAYS,

        "capacity_definition":
            (
                "sup A such that the equal-rebalance mean "
                "of executable trade-value share is at least "
                "95%, under a 5% one-day ADV20 name-level "
                "participation constraint."
            ),

        "rebalance_coverage_definition":
            (
                "sum_i min(A*abs(delta_weight_i), "
                "0.05*effective_ADV20_i) "
                "/ (A*sum_i abs(delta_weight_i))"
            ),

        "factor_window_aggregation":
            (
                "Equal weight across rebalance dates, "
                "matching Step-3 factor-window mean "
                "executable-share summaries."
            ),

        "capacity_solver":
            (
                "Continuous monotone bisection on "
                "stock-level trade-demand and ADV data; "
                "no interpolation of Step-3 AUM grid."
            ),

        "step3_grid_used_for":
            (
                "Validation and bracket diagnostics only."
            ),

        "current_suspension_rule":
            (
                "effective_ADV20=0; therefore current "
                "suspended trades have zero immediate "
                "executable amount."
            ),

        "zero_capacity_rule":
            (
                "Capacity=0 when the AUM->0+ executable "
                "share is already below 95%."
            ),

        "pair_interpretation":
            (
                "A is strategy NAV and each Q1/Q5 leg "
                "has gross notional A. Pair gross notional "
                "at capacity is 2A. Short borrowability "
                "is not modeled."
            ),

        "future_liquidity_used":
            False,

        "future_reopen_information_used":
            False,

        "transaction_cost_used":
            False,

        "factor_selection":
            False,

        "window_selection":
            False,

        "sign_flip":
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

        "outputs": {

            "rebalance_leg_capacity":
                str(
                    LEG_REBALANCE_CAPACITY_PATH
                ),

            "rebalance_pair_capacity":
                str(
                    PAIR_REBALANCE_CAPACITY_PATH
                ),

            "factor_window_capacity":
                str(
                    FACTOR_WINDOW_CAPACITY_PATH
                ),

            "grid_validation":
                str(
                    GRID_VALIDATION_PATH
                ),

            "zero_capacity_detail":
                str(
                    ZERO_CAPACITY_DETAIL_PATH
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
    # 11. Console summary
    # ========================================================

    print()
    print("=" * 80)

    print(
        "DAY 8 STEP 4 COMPLETE"
    )

    print("=" * 80)

    print(
        "Leg rebalance capacities: "
        f"{len(leg_capacity):,}"
    )

    print(
        "Pair rebalance capacities: "
        f"{len(pair_capacity):,}"
    )

    print(
        "Factor-window capacity specs: "
        f"{len(factor_window_capacity):,}"
    )

    print(
        "Formal QA pass: "
        f"{qa['all_formal_qa_pass']}"
    )

    print()
    print(
        "Main result:"
    )

    print(
        FACTOR_WINDOW_CAPACITY_PATH
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