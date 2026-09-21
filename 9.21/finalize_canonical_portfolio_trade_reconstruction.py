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
# Frozen target portfolios from original Day-8 Step 2
#
# IMPORTANT:
# Current portfolios are NOT reconstructed again.
# ------------------------------------------------------------

FROZEN_STEP2_DIR = (
    OUTPUT_ROOT
    / "M1_day8"
    / "02_stage2_trade_reconstruction"
)

TARGET_WEIGHT_PATH = (
    FROZEN_STEP2_DIR
    / "portfolio_target_weights.parquet"
)


# ------------------------------------------------------------
# Step 2C audited dedicated drift returns
# ------------------------------------------------------------

STEP2C_DIR = (
    OUTPUT_ROOT
    / "M1_day8"
    / "02d_stage2c_delisting_missing_record_audit"
)

AUDITED_DRIFT_PATH = (
    STEP2C_DIR
    / "dedicated_drift_return_panel_audited.parquet"
)

STEP2C_QA_PATH = (
    STEP2C_DIR
    / "step2c_formal_qa.csv"
)


# ------------------------------------------------------------
# Previous pre-audit dedicated-drift rerun
#
# Diagnostic comparison only.
# ------------------------------------------------------------

PRE_AUDIT_STEP2_DIR = (
    OUTPUT_ROOT
    / "M1_day8"
    / "02c_stage2_rerun_with_dedicated_drift"
)

PRE_AUDIT_REBALANCE_PATH = (
    PRE_AUDIT_STEP2_DIR
    / "portfolio_rebalance_summary_dedicated_drift.csv"
)


# ------------------------------------------------------------
# Day-7 Step-4 historical comparison
# ------------------------------------------------------------

DAY7_STEP4_PATH = (
    OUTPUT_ROOT
    / "M1_day7"
    / "04_stage4_turnover_transaction_cost"
    / "turnover_economic_summary.csv"
)


# ------------------------------------------------------------
# FINAL CANONICAL Step 2
# ------------------------------------------------------------

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day8"
    / "02e_stage2_final_canonical_trade_reconstruction"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRADE_PANEL_PATH = (
    OUTPUT_DIR
    / "canonical_portfolio_trade_weight_panel.parquet"
)

REBALANCE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "canonical_portfolio_rebalance_summary.csv"
)

PAIR_SUMMARY_PATH = (
    OUTPUT_DIR
    / "canonical_portfolio_pair_summary.csv"
)

AUM_SUMMARY_PATH = (
    OUTPUT_DIR
    / "canonical_portfolio_trade_value_scenario_summary.csv"
)

DRIFT_PROVENANCE_PATH = (
    OUTPUT_DIR
    / "canonical_drift_provenance_summary.csv"
)

PRE_AUDIT_COMPARISON_PATH = (
    OUTPUT_DIR
    / "canonical_vs_pre_audit_comparison.csv"
)

STEP4_COMPARISON_PATH = (
    OUTPUT_DIR
    / "canonical_vs_day7_step4_comparison.csv"
)

QA_PATH = (
    OUTPUT_DIR
    / "canonical_portfolio_trade_reconstruction_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day8_canonical_step2_metadata.json"
)


# ============================================================
# 1. Frozen conventions
# ============================================================

VALID_DRIFT_STATUSES = {
    "PASS",
    "PASS_DELIST_CASH_CARRY",
}


AUM_SCENARIOS_CNY = [
    1e8,    # 1亿元
    5e8,    # 5亿元
    1e9,    # 10亿元
    2e9,    # 20亿元
    5e9,    # 50亿元
    1e10,   # 100亿元
]


REFERENCE_NAV_100M_CNY = 1e8
REFERENCE_NAV_1B_CNY = 1e9


TOL = 1e-12


# ============================================================
# 2. Utility functions
# ============================================================

def save_json_atomic(
    obj,
    path: Path,
):
    tmp = Path(
        str(path) + ".tmp"
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
        str(path) + ".tmp"
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


def save_parquet_atomic(
    df: pd.DataFrame,
    path: Path,
):
    tmp = Path(
        str(path) + ".tmp"
    )

    df.to_parquet(
        tmp,
        index=False,
        compression="snappy",
    )

    os.replace(
        tmp,
        path,
    )


def normalize_security_id(
    x: pd.Series,
):
    return (
        x.astype("string")
        .str.strip()
    )


def canonical_hash(
    obj,
):
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


def parse_bool(
    x,
):
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
    }:
        return True

    if value in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Cannot parse bool value: {x}"
    )


# ============================================================
# 3. Stable DataFrame fingerprint
#
# Used only to document that the frozen portfolio design
# is exactly the portfolio panel used in this canonical run.
# ============================================================

def dataframe_fingerprint(
    df: pd.DataFrame,
    columns,
):
    temp = (
        df[
            columns
        ]
        .copy()
        .sort_values(
            columns[:-1]
        )
        .reset_index(
            drop=True
        )
    )

    hashed = pd.util.hash_pandas_object(
        temp,
        index=False,
    ).to_numpy(
        dtype=np.uint64
    )

    return hashlib.sha256(
        hashed.tobytes()
    ).hexdigest()


# ============================================================
# 4. Validate Step-2C upstream QA
# ============================================================

def validate_step2c_qa():

    if not STEP2C_QA_PATH.exists():

        raise FileNotFoundError(
            f"Missing Step-2C QA:\n"
            f"{STEP2C_QA_PATH}"
        )

    qa = pd.read_csv(
        STEP2C_QA_PATH
    )

    required = {
        "qa_name",
        "qa_value",
    }

    if not required.issubset(
        qa.columns
    ):

        raise RuntimeError(
            "Unexpected Step-2C QA schema."
        )

    qa_map = dict(
        zip(
            qa["qa_name"],
            qa["qa_value"],
        )
    )

    required_true = [
        "all_formal_qa_pass",
        "audit_row_count_matches_issue_count",
        "classification_complete",
        "original_pass_status_unchanged",
        "original_pass_return_unchanged",
        "no_true_source_gap_auto_fill",
    ]

    for name in required_true:

        if name not in qa_map:

            raise RuntimeError(
                f"Step-2C QA missing `{name}`."
            )

        if not parse_bool(
            qa_map[
                name
            ]
        ):

            raise RuntimeError(
                f"Step-2C upstream QA failed: {name}"
            )

    # --------------------------------------------------------
    # This is now required for canonical freezing.
    # --------------------------------------------------------

    if (
        "audited_remaining_nonpass_count"
        not in qa_map
    ):

        raise RuntimeError(
            "Step-2C QA missing "
            "`audited_remaining_nonpass_count`."
        )

    remaining = int(
        float(
            qa_map[
                "audited_remaining_nonpass_count"
            ]
        )
    )

    if remaining != 0:

        raise RuntimeError(
            "Canonical Step 2 cannot be frozen because "
            f"{remaining} audited drift observations "
            "remain unresolved."
        )

    return qa_map


# ============================================================
# 5. Load frozen target portfolios
# ============================================================

def load_target_weights():

    if not TARGET_WEIGHT_PATH.exists():

        raise FileNotFoundError(
            f"Missing frozen target weights:\n"
            f"{TARGET_WEIGHT_PATH}"
        )

    df = pd.read_parquet(
        TARGET_WEIGHT_PATH
    )

    required = [
        "analysis_date",
        "window",
        "factor_name",
        "method",
        "portfolio_leg",
        "security_id",
        "target_weight",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Frozen target-weight panel missing:\n"
            f"{missing}"
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

    df[
        "target_weight"
    ] = pd.to_numeric(
        df[
            "target_weight"
        ],
        errors="raise",
    )

    key = [
        "analysis_date",
        "window",
        "factor_name",
        "method",
        "portfolio_leg",
        "security_id",
    ]

    if df[
        key
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate frozen target-weight keys."
        )

    if (
        ~np.isfinite(
            df[
                "target_weight"
            ]
        )
    ).any():

        raise RuntimeError(
            "Non-finite frozen target weights detected."
        )

    if (
        df[
            "target_weight"
        ]
        <=
        0
    ).any():

        raise RuntimeError(
            "Non-positive target weights detected."
        )

    return (
        df.sort_values(
            key
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 6. Load audited dedicated drift returns
# ============================================================

def load_audited_drift_returns():

    if not AUDITED_DRIFT_PATH.exists():

        raise FileNotFoundError(
            f"Missing audited drift panel:\n"
            f"{AUDITED_DRIFT_PATH}"
        )

    df = pd.read_parquet(
        AUDITED_DRIFT_PATH
    )

    required = [
        "previous_analysis_date",
        "analysis_date",
        "security_id",
        "status",
        "drift_total_return",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Audited drift panel missing:\n"
            f"{missing}"
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
        "drift_total_return"
    ] = pd.to_numeric(
        df[
            "drift_total_return"
        ],
        errors="coerce",
    )

    key = [
        "previous_analysis_date",
        "analysis_date",
        "security_id",
    ]

    if df[
        key
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate audited-drift keys."
        )

    # --------------------------------------------------------
    # Canonical version requires 100% valid audited drift.
    # --------------------------------------------------------

    invalid_status = (
        ~df[
            "status"
        ]
        .isin(
            VALID_DRIFT_STATUSES
        )
    )

    if invalid_status.any():

        counts = (
            df.loc[
                invalid_status,
                "status",
            ]
            .value_counts()
        )

        raise RuntimeError(
            "Canonical audited drift contains "
            "non-valid status rows:\n"
            f"{counts}"
        )

    if (
        df[
            "drift_total_return"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Canonical audited drift contains "
            "missing drift_total_return."
        )

    if (
        ~np.isfinite(
            df[
                "drift_total_return"
            ]
        )
    ).any():

        raise RuntimeError(
            "Non-finite canonical drift return detected."
        )

    if (
        df[
            "drift_total_return"
        ]
        <
        -1.0
        -
        TOL
    ).any():

        raise RuntimeError(
            "Canonical drift return below -100% detected."
        )

    return (
        df.sort_values(
            key
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 7. Build audited drift map
# ============================================================

def build_drift_maps(
    drift,
):

    maps = {}

    for (
        previous_date,
        current_date,
    ), group in drift.groupby(
        [
            "previous_analysis_date",
            "analysis_date",
        ],
        sort=False,
    ):

        temp = (
            group[
                [
                    "security_id",
                    "status",
                    "drift_total_return",
                ]
            ]
            .copy()
            .set_index(
                "security_id"
            )
        )

        maps[
            (
                pd.Timestamp(
                    previous_date
                ),
                pd.Timestamp(
                    current_date
                ),
            )
        ] = temp

    return maps


# ============================================================
# 8. Canonical trade reconstruction
# ============================================================

def reconstruct_canonical_trades(
    targets,
    drift,
):

    drift_maps = (
        build_drift_maps(
            drift
        )
    )

    trade_frames = []
    summary_rows = []

    portfolio_keys = [
        "window",
        "factor_name",
        "method",
        "portfolio_leg",
    ]

    grouped = targets.groupby(
        portfolio_keys,
        sort=False,
    )

    total_groups = (
        grouped.ngroups
    )

    for group_no, (
        keys,
        group,
    ) in enumerate(
        grouped,
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
                "Canonical reconstruction: "
                f"{group_no}/{total_groups}"
            )

        by_date = {}

        for date, temp in group.groupby(
            "analysis_date",
            sort=True,
        ):

            weights = (
                temp[
                    [
                        "security_id",
                        "target_weight",
                    ]
                ]
                .copy()
                .set_index(
                    "security_id"
                )[
                    "target_weight"
                ]
                .astype(float)
            )

            by_date[
                pd.Timestamp(
                    date
                )
            ] = weights

        dates = sorted(
            by_date.keys()
        )

        if len(dates) == 0:
            continue

        # ----------------------------------------------------
        # Initial formation
        # ----------------------------------------------------

        first_date = (
            dates[0]
        )

        first_target = (
            by_date[
                first_date
            ]
        )

        summary_rows.append(
            {
                "analysis_date":
                    first_date,

                "previous_analysis_date":
                    pd.NaT,

                "window":
                    int(window),

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

                "pass_drift_n":
                    0,

                "delist_cash_carry_n":
                    0,

                "pretrade_weight_sum":
                    np.nan,

                "target_weight_sum":
                    float(
                        first_target.sum()
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
        # Normal rebalance dates
        # ----------------------------------------------------

        for date_no in range(
            1,
            len(dates),
        ):

            previous_date = (
                dates[
                    date_no - 1
                ]
            )

            current_date = (
                dates[
                    date_no
                ]
            )

            previous_weights = (
                by_date[
                    previous_date
                ]
            )

            current_weights = (
                by_date[
                    current_date
                ]
            )

            pair_key = (
                previous_date,
                current_date,
            )

            pair_drift = (
                drift_maps.get(
                    pair_key
                )
            )

            if pair_drift is None:

                raise RuntimeError(
                    "Canonical drift date-pair map missing:\n"
                    f"{previous_date} -> {current_date}\n"
                    f"{factor_name}, W{window}, "
                    f"{method}, {portfolio_leg}"
                )

            aligned = (
                pair_drift.reindex(
                    previous_weights.index
                )
            )

            # ------------------------------------------------
            # Canonical freeze:
            #
            # ANY missing previous member is a hard error.
            # ------------------------------------------------

            missing_key = (
                aligned[
                    "status"
                ]
                .isna()
            )

            if missing_key.any():

                bad_ids = (
                    aligned.index[
                        missing_key
                    ]
                    .tolist()
                )

                raise RuntimeError(
                    "Canonical drift key missing for "
                    "previous portfolio member.\n"
                    f"Date pair: "
                    f"{previous_date} -> {current_date}\n"
                    f"Factor: {factor_name}\n"
                    f"Window: {window}\n"
                    f"Leg: {portfolio_leg}\n"
                    f"Examples: {bad_ids[:20]}"
                )

            valid_status = (
                aligned[
                    "status"
                ]
                .isin(
                    VALID_DRIFT_STATUSES
                )
            )

            valid_return = (

                aligned[
                    "drift_total_return"
                ]
                .notna()

                &

                np.isfinite(
                    aligned[
                        "drift_total_return"
                    ]
                )
            )

            if not (
                valid_status
                &
                valid_return
            ).all():

                bad = aligned.loc[
                    ~(
                        valid_status
                        &
                        valid_return
                    )
                ]

                raise RuntimeError(
                    "Canonical audited drift has invalid "
                    "previous portfolio members:\n"
                    f"{bad.head(20)}"
                )

            previous_returns = (
                aligned[
                    "drift_total_return"
                ]
                .astype(float)
            )

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

                raise RuntimeError(
                    "Invalid canonical drift denominator:\n"
                    f"{previous_date} -> {current_date}\n"
                    f"{factor_name}, W{window}, "
                    f"{portfolio_leg}"
                )

            pretrade_weights = (

                drift_value

                /

                denominator
            )

            union_index = (
                previous_weights
                .index
                .union(
                    current_weights.index
                )
            )

            previous_target_union = (
                previous_weights.reindex(
                    union_index,
                    fill_value=0.0,
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
                    lower=0.0
                )
            )

            sell_weight = (

                -
                delta_weight.clip(
                    upper=0.0
                )
            )

            total_abs_trade_weight = float(
                abs_delta_weight.sum()
            )

            turnover = float(
                0.5
                *
                total_abs_trade_weight
            )

            # ------------------------------------------------
            # Preserve drift provenance.
            # ------------------------------------------------

            drift_status_union = pd.Series(
                "NOT_REQUIRED_NEW_ENTRY",
                index=union_index,
                dtype="string",
            )

            drift_return_union = pd.Series(
                np.nan,
                index=union_index,
                dtype=float,
            )

            previous_ids = (
                previous_weights.index
            )

            drift_status_union.loc[
                previous_ids
            ] = (
                aligned.loc[
                    previous_ids,
                    "status",
                ]
                .astype("string")
            )

            drift_return_union.loc[
                previous_ids
            ] = (
                aligned.loc[
                    previous_ids,
                    "drift_total_return",
                ]
                .astype(float)
            )

            stock_trade = pd.DataFrame(
                {
                    "analysis_date":
                        current_date,

                    "previous_analysis_date":
                        previous_date,

                    "window":
                        int(window),

                    "factor_name":
                        factor_name,

                    "method":
                        method,

                    "portfolio_leg":
                        portfolio_leg,

                    "security_id":
                        union_index.astype(str),

                    "previous_target_weight":
                        previous_target_union
                        .to_numpy(
                            dtype=float
                        ),

                    "drift_return_required":
                        (
                            previous_target_union
                            >
                            0
                        )
                        .to_numpy(
                            dtype=bool
                        ),

                    "dedicated_drift_status":
                        drift_status_union
                        .to_numpy(),

                    "holding_period_total_return":
                        drift_return_union
                        .to_numpy(
                            dtype=float
                        ),

                    "pretrade_weight":
                        pretrade_union
                        .to_numpy(
                            dtype=float
                        ),

                    "target_weight":
                        target_union
                        .to_numpy(
                            dtype=float
                        ),

                    "delta_weight":
                        delta_weight
                        .to_numpy(
                            dtype=float
                        ),

                    "abs_delta_weight":
                        abs_delta_weight
                        .to_numpy(
                            dtype=float
                        ),

                    "buy_weight":
                        buy_weight
                        .to_numpy(
                            dtype=float
                        ),

                    "sell_weight":
                        sell_weight
                        .to_numpy(
                            dtype=float
                        ),
                }
            )

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
                ),
            )

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

            trade_frames.append(
                stock_trade
            )

            pass_drift_n = int(
                (
                    aligned[
                        "status"
                    ]
                    ==
                    "PASS"
                )
                .sum()
            )

            delist_cash_n = int(
                (
                    aligned[
                        "status"
                    ]
                    ==
                    "PASS_DELIST_CASH_CARRY"
                )
                .sum()
            )

            summary_rows.append(
                {
                    "analysis_date":
                        current_date,

                    "previous_analysis_date":
                        previous_date,

                    "window":
                        int(window),

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

                    "pass_drift_n":
                        pass_drift_n,

                    "delist_cash_carry_n":
                        delist_cash_n,

                    "pretrade_weight_sum":
                        float(
                            pretrade_union.sum()
                        ),

                    "target_weight_sum":
                        float(
                            target_union.sum()
                        ),

                    "turnover":
                        turnover,

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

    trades = pd.concat(
        trade_frames,
        ignore_index=True,
    )

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
# 9. Correct pair-level summary
#
# IMPORTANT:
#
# Do NOT use:
#
# pivot_table(..., dropna=False)
#
# because that can create Cartesian combinations.
# ============================================================

def build_pair_summary(
    rebalance_summary,
):

    # --------------------------------------------------------
    # analysis_date × window × factor × method
    #
    # uniquely defines the Q1-Q5 pair.
    #
    # previous_analysis_date is attached separately,
    # avoiding NaT problems for initial formation.
    # --------------------------------------------------------

    pair_key = [
        "analysis_date",
        "window",
        "factor_name",
        "method",
    ]

    full_key = (
        pair_key
        +
        [
            "portfolio_leg"
        ]
    )

    if rebalance_summary[
        full_key
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate portfolio-leg rows "
            "in canonical rebalance summary."
        )

    previous_date_table = (

        rebalance_summary[
            pair_key
            +
            [
                "previous_analysis_date"
            ]
        ]
        .drop_duplicates()
    )

    if previous_date_table[
        pair_key
    ].duplicated().any():

        raise RuntimeError(
            "Multiple previous_analysis_date values "
            "for the same portfolio pair."
        )

    status = (
        rebalance_summary[
            full_key
            +
            [
                "status"
            ]
        ]
        .pivot(
            index=pair_key,
            columns="portfolio_leg",
            values="status",
        )
        .reset_index()
    )

    status.columns.name = None

    turnover = (
        rebalance_summary[
            full_key
            +
            [
                "turnover"
            ]
        ]
        .pivot(
            index=pair_key,
            columns="portfolio_leg",
            values="turnover",
        )
        .reset_index()
    )

    turnover.columns.name = None

    turnover = turnover.rename(
        columns={
            "Q1":
                "q1_turnover",

            "Q5":
                "q5_turnover",
        }
    )

    status = status.rename(
        columns={
            "Q1":
                "q1_status",

            "Q5":
                "q5_status",
        }
    )

    for column in [
        "q1_status",
        "q5_status",
    ]:

        if column not in status.columns:

            status[
                column
            ] = np.nan

    for column in [
        "q1_turnover",
        "q5_turnover",
    ]:

        if column not in turnover.columns:

            turnover[
                column
            ] = np.nan

    pair = status.merge(
        turnover,
        on=pair_key,
        how="left",
        validate="one_to_one",
    )

    pair = pair.merge(
        previous_date_table,
        on=pair_key,
        how="left",
        validate="one_to_one",
    )

    pair[
        "both_initial_formation"
    ] = (

        pair[
            "q1_status"
        ]
        .eq(
            "NO_PREVIOUS_FORMATION"
        )

        &

        pair[
            "q5_status"
        ]
        .eq(
            "NO_PREVIOUS_FORMATION"
        )
    )

    pair[
        "both_legs_pass"
    ] = (

        pair[
            "q1_status"
        ]
        .eq(
            "PASS"
        )

        &

        pair[
            "q5_status"
        ]
        .eq(
            "PASS"
        )
    )

    pair[
        "pair_status"
    ] = np.select(
        [
            pair[
                "both_initial_formation"
            ],

            pair[
                "both_legs_pass"
            ],
        ],
        [
            "BOTH_INITIAL",
            "BOTH_PASS",
        ],
        default="PAIR_FAILURE",
    )

    pair[
        "roundtrip_traded_notional_ratio"
    ] = np.where(
        pair[
            "both_legs_pass"
        ],

        2.0
        *
        (
            pair[
                "q1_turnover"
            ]
            +
            pair[
                "q5_turnover"
            ]
        ),

        np.nan,
    )

    return (
        pair.sort_values(
            pair_key
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 10. AUM scenarios
# ============================================================

def build_aum_summary(
    rebalance_summary,
):

    valid = rebalance_summary[
        rebalance_summary[
            "status"
        ]
        ==
        "PASS"
    ].copy()

    frames = []

    for aum in AUM_SCENARIOS_CNY:

        temp = valid[
            [
                "analysis_date",
                "previous_analysis_date",
                "window",
                "factor_name",
                "method",
                "portfolio_leg",
                "turnover",
                "total_abs_trade_weight",
            ]
        ].copy()

        temp[
            "strategy_nav_cny"
        ] = float(
            aum
        )

        temp[
            "leg_gross_notional_cny"
        ] = float(
            aum
        )

        temp[
            "strategy_gross_exposure"
        ] = 2.0

        temp[
            "trade_notional_cny"
        ] = (

            float(
                aum
            )

            *

            temp[
                "total_abs_trade_weight"
            ]
        )

        frames.append(
            temp
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# 11. Drift provenance
# ============================================================

def build_drift_provenance_summary(
    trades,
):

    required = trades[
        trades[
            "drift_return_required"
        ]
    ].copy()

    summary = (

        required.groupby(
            [
                "dedicated_drift_status"
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            stock_trade_rows=(
                "security_id",
                "size",
            ),

            unique_securities=(
                "security_id",
                "nunique",
            ),

            unique_rebalance_dates=(
                "analysis_date",
                "nunique",
            ),
        )
    )

    return summary


# ============================================================
# 12. Compare with pre-audit dedicated rerun
# ============================================================

def compare_with_pre_audit(
    canonical_summary,
):

    if not PRE_AUDIT_REBALANCE_PATH.exists():

        return pd.DataFrame()

    old = pd.read_csv(
        PRE_AUDIT_REBALANCE_PATH
    )

    for column in [
        "analysis_date",
        "previous_analysis_date",
    ]:

        old[
            column
        ] = pd.to_datetime(
            old[
                column
            ],
            errors="coerce",
        )

    key = [
        "analysis_date",
        "window",
        "factor_name",
        "method",
        "portfolio_leg",
    ]

    old = old[
        key
        +
        [
            "status",
            "turnover",
        ]
    ].rename(
        columns={
            "status":
                "pre_audit_status",

            "turnover":
                "pre_audit_turnover",
        }
    )

    new = canonical_summary[
        key
        +
        [
            "status",
            "turnover",
            "delist_cash_carry_n",
        ]
    ].rename(
        columns={
            "status":
                "canonical_status",

            "turnover":
                "canonical_turnover",
        }
    )

    comparison = new.merge(
        old,
        on=key,
        how="left",
        validate="one_to_one",
    )

    comparison[
        "status_transition"
    ] = (

        comparison[
            "pre_audit_status"
        ]
        .astype("string")

        +

        " -> "

        +

        comparison[
            "canonical_status"
        ]
        .astype("string")
    )

    comparison[
        "turnover_difference"
    ] = (

        comparison[
            "canonical_turnover"
        ]

        -

        comparison[
            "pre_audit_turnover"
        ]
    )

    return comparison


# ============================================================
# 13. Compare with Day-7 Step-4
# ============================================================

def compare_with_day7_step4(
    canonical_summary,
):

    if not DAY7_STEP4_PATH.exists():

        return pd.DataFrame()

    old = pd.read_csv(
        DAY7_STEP4_PATH
    )

    required = [
        "factor_name",
        "method",
        "window",
        "mean_q1_drift_turnover",
        "mean_q5_drift_turnover",
        "mean_roundtrip_traded_notional_ratio",
    ]

    if not all(
        column in old.columns
        for column in required
    ):

        return pd.DataFrame()

    valid = canonical_summary[
        canonical_summary[
            "status"
        ]
        ==
        "PASS"
    ].copy()

    current = (

        valid.groupby(
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

    pivot = current.pivot(
        index=[
            "factor_name",
            "method",
            "window",
        ],
        columns="portfolio_leg",
        values="turnover",
    ).reset_index()

    pivot.columns.name = None

    pivot = pivot.rename(
        columns={
            "Q1":
                "canonical_mean_q1_turnover",

            "Q5":
                "canonical_mean_q5_turnover",
        }
    )

    pivot[
        "canonical_roundtrip_traded_notional_ratio"
    ] = (

        2.0

        *

        (
            pivot[
                "canonical_mean_q1_turnover"
            ]

            +

            pivot[
                "canonical_mean_q5_turnover"
            ]
        )
    )

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
            "canonical_mean_q1_turnover"
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
            "canonical_mean_q5_turnover"
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
            "canonical_roundtrip_traded_notional_ratio"
        ]

        -

        result[
            "mean_roundtrip_traded_notional_ratio"
        ]
    )

    return result


# ============================================================
# 14. Formal canonical QA
# ============================================================

def formal_qa(
    targets,
    drift,
    trades,
    summary,
    pair_summary,
    upstream_qa,
):

    qa = {}

    # --------------------------------------------------------
    # Target portfolio QA
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
        TOL
    )

    # --------------------------------------------------------
    # Audited drift QA
    # --------------------------------------------------------

    qa[
        "audited_drift_row_count"
    ] = int(
        len(
            drift
        )
    )

    qa[
        "audited_drift_all_valid"
    ] = bool(
        drift[
            "status"
        ]
        .isin(
            VALID_DRIFT_STATUSES
        )
        .all()
    )

    qa[
        "audited_drift_missing_return_count"
    ] = int(
        drift[
            "drift_total_return"
        ]
        .isna()
        .sum()
    )

    qa[
        "audited_drift_pass_count"
    ] = int(
        (
            drift[
                "status"
            ]
            ==
            "PASS"
        )
        .sum()
    )

    qa[
        "audited_drift_delist_cash_carry_count"
    ] = int(
        (
            drift[
                "status"
            ]
            ==
            "PASS_DELIST_CASH_CARRY"
        )
        .sum()
    )

    qa[
        "upstream_step2c_formal_qa_pass"
    ] = parse_bool(
        upstream_qa[
            "all_formal_qa_pass"
        ]
    )

    # --------------------------------------------------------
    # Rebalance coverage
    # --------------------------------------------------------

    qa[
        "rebalance_row_count"
    ] = int(
        len(
            summary
        )
    )

    qa[
        "rebalance_pass_count"
    ] = int(
        (
            summary[
                "status"
            ]
            ==
            "PASS"
        )
        .sum()
    )

    qa[
        "initial_formation_count"
    ] = int(
        (
            summary[
                "status"
            ]
            ==
            "NO_PREVIOUS_FORMATION"
        )
        .sum()
    )

    qa[
        "noninitial_rebalance_count"
    ] = int(
        (
            summary[
                "status"
            ]
            !=
            "NO_PREVIOUS_FORMATION"
        )
        .sum()
    )

    qa[
        "noninitial_rebalance_pass_count"
    ] = int(
        (
            summary[
                "status"
            ]
            ==
            "PASS"
        )
        .sum()
    )

    qa[
        "noninitial_rebalance_pass_share"
    ] = (

        qa[
            "noninitial_rebalance_pass_count"
        ]

        /

        qa[
            "noninitial_rebalance_count"
        ]
    )

    qa[
        "all_noninitial_rebalances_pass"
    ] = bool(
        np.isclose(
            qa[
                "noninitial_rebalance_pass_share"
            ],
            1.0,
        )
    )

    # --------------------------------------------------------
    # Pair QA
    # --------------------------------------------------------

    qa[
        "pair_row_count"
    ] = int(
        len(
            pair_summary
        )
    )

    qa[
        "initial_pair_count"
    ] = int(
        pair_summary[
            "both_initial_formation"
        ]
        .sum()
    )

    noninitial_pairs = pair_summary[
        ~pair_summary[
            "both_initial_formation"
        ]
    ].copy()

    qa[
        "noninitial_pair_count"
    ] = int(
        len(
            noninitial_pairs
        )
    )

    qa[
        "both_legs_pass_count_noninitial"
    ] = int(
        noninitial_pairs[
            "both_legs_pass"
        ]
        .sum()
    )

    qa[
        "both_legs_pass_share_noninitial"
    ] = (

        qa[
            "both_legs_pass_count_noninitial"
        ]

        /

        qa[
            "noninitial_pair_count"
        ]
    )

    qa[
        "all_noninitial_pairs_both_legs_pass"
    ] = bool(
        np.isclose(
            qa[
                "both_legs_pass_share_noninitial"
            ],
            1.0,
        )
    )

    qa[
        "pair_failure_count"
    ] = int(
        (
            pair_summary[
                "pair_status"
            ]
            ==
            "PAIR_FAILURE"
        )
        .sum()
    )

    # --------------------------------------------------------
    # Stock-level accounting identities
    # --------------------------------------------------------

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

    valid = summary[
        summary[
            "status"
        ]
        ==
        "PASS"
    ].copy()

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
        "pretrade_weight_sum_ok"
    ] = bool(
        qa[
            "max_abs_pretrade_weight_sum_minus_1"
        ]
        <
        TOL
    )

    qa[
        "max_abs_target_weight_sum_rebalance_minus_1"
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
        "rebalance_target_weight_sum_ok"
    ] = bool(
        qa[
            "max_abs_target_weight_sum_rebalance_minus_1"
        ]
        <
        TOL
    )

    qa[
        "turnover_within_0_1"
    ] = bool(
        (
            valid[
                "turnover"
            ]
            >=
            -TOL
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
            TOL
        )
        .all()
    )

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
        TOL

        and

        sell_error.max()
        <
        TOL
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

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    previous_rows = trades[
        trades[
            "drift_return_required"
        ]
    ]

    qa[
        "all_required_trade_rows_have_valid_drift_status"
    ] = bool(
        previous_rows[
            "dedicated_drift_status"
        ]
        .isin(
            VALID_DRIFT_STATUSES
        )
        .all()
    )

    qa[
        "required_trade_rows_missing_return_count"
    ] = int(
        previous_rows[
            "holding_period_total_return"
        ]
        .isna()
        .sum()
    )

    qa[
        "delist_cash_carry_used_in_trade_panel"
    ] = bool(
        (
            previous_rows[
                "dedicated_drift_status"
            ]
            ==
            "PASS_DELIST_CASH_CARRY"
        )
        .any()
    )

    # --------------------------------------------------------
    # Research governance
    # --------------------------------------------------------

    qa[
        "target_portfolios_recomputed"
    ] = False

    qa[
        "frozen_target_portfolios_used"
    ] = True

    qa[
        "statistical_future_label_used_for_weight_drift"
    ] = False

    qa[
        "audited_dedicated_drift_used"
    ] = True

    qa[
        "missing_drift_return_zero_filled"
    ] = False

    qa[
        "missing_drift_stock_dropped"
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

    qa[
        "initial_formation_treated_as_rebalance"
    ] = False

    qa[
        "terminal_liquidation_assumed"
    ] = False

    # --------------------------------------------------------
    # Canonical freeze criteria
    # --------------------------------------------------------

    required_true = [
        "target_weight_sum_ok",
        "audited_drift_all_valid",
        "upstream_step2c_formal_qa_pass",
        "all_noninitial_rebalances_pass",
        "all_noninitial_pairs_both_legs_pass",
        "trade_keys_unique",
        "delta_weight_identity_ok",
        "pretrade_weight_sum_ok",
        "rebalance_target_weight_sum_ok",
        "turnover_within_0_1",
        "buy_sell_turnover_identity_ok",
        "monetary_notional_identity_ok",
        "all_required_trade_rows_have_valid_drift_status",
    ]

    qa[
        "canonical_freeze_pass"
    ] = bool(
        all(
            qa[
                name
            ]
            for name in required_true
        )

        and

        qa[
            "audited_drift_missing_return_count"
        ]
        ==
        0

        and

        qa[
            "pair_failure_count"
        ]
        ==
        0

        and

        qa[
            "required_trade_rows_missing_return_count"
        ]
        ==
        0
    )

    return qa


# ============================================================
# 15. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 8 - FINAL CANONICAL STEP 2"
    )

    print(
        "Audited Dedicated-Drift Portfolio "
        "Trade Reconstruction"
    )

    print("=" * 80)

    # ========================================================
    # Step-2C QA
    # ========================================================

    print()
    print(
        "[1] Validate Step-2C upstream QA"
    )

    upstream_qa = (
        validate_step2c_qa()
    )

    # ========================================================
    # Frozen portfolios
    # ========================================================

    print()
    print(
        "[2] Load frozen target portfolios"
    )

    targets = (
        load_target_weights()
    )

    target_fingerprint = (
        dataframe_fingerprint(
            targets,
            [
                "analysis_date",
                "window",
                "factor_name",
                "method",
                "portfolio_leg",
                "security_id",
                "target_weight",
            ],
        )
    )

    print(
        f"Target rows: "
        f"{len(targets):,}"
    )

    print(
        "Target portfolio fingerprint:"
    )

    print(
        target_fingerprint
    )

    # ========================================================
    # Audited drift
    # ========================================================

    print()
    print(
        "[3] Load audited dedicated drift returns"
    )

    drift = (
        load_audited_drift_returns()
    )

    print(
        f"Audited drift rows: "
        f"{len(drift):,}"
    )

    print(
        "Drift status counts:"
    )

    print(
        drift[
            "status"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Reconstruct
    # ========================================================

    print()
    print(
        "[4] Reconstruct canonical stock-level trades"
    )

    (
        trades,
        rebalance_summary,
    ) = reconstruct_canonical_trades(
        targets,
        drift,
    )

    # ========================================================
    # Pair summary
    # ========================================================

    print()
    print(
        "[5] Build corrected Q1/Q5 pair summary"
    )

    pair_summary = (
        build_pair_summary(
            rebalance_summary
        )
    )

    # ========================================================
    # AUM
    # ========================================================

    print()
    print(
        "[6] Build AUM scenario summary"
    )

    aum_summary = (
        build_aum_summary(
            rebalance_summary
        )
    )

    # ========================================================
    # Provenance
    # ========================================================

    print()
    print(
        "[7] Build drift provenance summary"
    )

    provenance = (
        build_drift_provenance_summary(
            trades
        )
    )

    # ========================================================
    # Comparisons
    # ========================================================

    print()
    print(
        "[8] Compare with pre-audit rerun"
    )

    pre_audit_comparison = (
        compare_with_pre_audit(
            rebalance_summary
        )
    )

    print()
    print(
        "[9] Compare with Day-7 Step-4"
    )

    step4_comparison = (
        compare_with_day7_step4(
            rebalance_summary
        )
    )

    # ========================================================
    # Formal QA BEFORE writing canonical outputs
    # ========================================================

    print()
    print(
        "[10] Canonical formal QA"
    )

    qa = formal_qa(
        targets,
        drift,
        trades,
        rebalance_summary,
        pair_summary,
        upstream_qa,
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

    if not qa[
        "canonical_freeze_pass"
    ]:

        raise RuntimeError(
            "CANONICAL FREEZE FAILED.\n\n"
            "No canonical trade files were finalized.\n\n"
            f"QA:\n{qa}"
        )

    # ========================================================
    # Only now write canonical outputs
    # ========================================================

    print()
    print(
        "[11] Canonical QA passed - "
        "write final frozen outputs"
    )

    save_parquet_atomic(
        trades,
        TRADE_PANEL_PATH,
    )

    save_csv_atomic(
        rebalance_summary,
        REBALANCE_SUMMARY_PATH,
    )

    save_csv_atomic(
        pair_summary,
        PAIR_SUMMARY_PATH,
    )

    save_csv_atomic(
        aum_summary,
        AUM_SUMMARY_PATH,
    )

    save_csv_atomic(
        provenance,
        DRIFT_PROVENANCE_PATH,
    )

    if len(
        pre_audit_comparison
    ) > 0:

        save_csv_atomic(
            pre_audit_comparison,
            PRE_AUDIT_COMPARISON_PATH,
        )

    if len(
        step4_comparison
    ) > 0:

        save_csv_atomic(
            step4_comparison,
            STEP4_COMPARISON_PATH,
        )

    # ========================================================
    # Metadata / freeze record
    # ========================================================

    design = {

        "research_day":
            8,

        "step":
            "Final_Canonical_Step2_Portfolio_Trade_Reconstruction",

        "canonical":
            True,

        "target_portfolio_source":
            str(
                TARGET_WEIGHT_PATH
            ),

        "target_portfolio_fingerprint":
            target_fingerprint,

        "target_portfolios_recomputed":
            False,

        "target_portfolios_frozen":
            True,

        "audited_drift_source":
            str(
                AUDITED_DRIFT_PATH
            ),

        "valid_drift_statuses":
            sorted(
                VALID_DRIFT_STATUSES
            ),

        "drift_return_type":
            "audited_portfolio_accounting_return",

        "statistical_future_label_used":
            False,

        "drift_formula":
            (
                "w_pre_i = "
                "w_prev_i * (1 + R_drift_audited_i) / "
                "sum_j[w_prev_j * "
                "(1 + R_drift_audited_j)]"
            ),

        "trade_weight_formula":
            "delta_w = target_w - pretrade_w",

        "turnover_formula":
            "0.5 * sum(abs(delta_w))",

        "aum_definition":
            (
                "Strategy NAV; each Q1/Q5 leg gross "
                "notional equals NAV; total gross "
                "exposure = 200%."
            ),

        "aum_scenarios_cny":
            AUM_SCENARIOS_CNY,

        "initial_formation_in_turnover":
            False,

        "terminal_liquidation":
            False,

        "missing_return_zero_fill":
            False,

        "factor_sign_flip":
            False,

        "factor_selection":
            False,

        "window_selection":
            False,

        "pair_summary_method":
            (
                "Observed-combination pivot without "
                "pivot_table(dropna=False) Cartesian expansion."
            ),
    }

    metadata = {

        **design,

        "canonical_design_hash":
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

            "canonical_trade_panel":
                str(
                    TRADE_PANEL_PATH
                ),

            "canonical_rebalance_summary":
                str(
                    REBALANCE_SUMMARY_PATH
                ),

            "canonical_pair_summary":
                str(
                    PAIR_SUMMARY_PATH
                ),

            "canonical_aum_summary":
                str(
                    AUM_SUMMARY_PATH
                ),

            "drift_provenance_summary":
                str(
                    DRIFT_PROVENANCE_PATH
                ),

            "pre_audit_comparison":
                str(
                    PRE_AUDIT_COMPARISON_PATH
                ),

            "day7_step4_comparison":
                str(
                    STEP4_COMPARISON_PATH
                ),

            "canonical_qa":
                str(
                    QA_PATH
                ),
        },

        "canonical_freeze_statement":
            (
                "This directory is the canonical Day-8 "
                "portfolio-trade reconstruction for subsequent "
                "participation-rate and capacity analysis. "
                "Target portfolios are frozen; audited dedicated "
                "drift returns are used for portfolio wealth "
                "evolution; no outcome-driven factor, sign, or "
                "window selection is performed."
            ),
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
        "CANONICAL STEP 2 FREEZE PASSED"
    )

    print("=" * 80)

    print(
        f"Rebalance rows: "
        f"{qa['rebalance_row_count']:,}"
    )

    print(
        f"Non-initial rebalances: "
        f"{qa['noninitial_rebalance_count']:,}"
    )

    print(
        "Non-initial rebalance PASS share: "
        f"{qa['noninitial_rebalance_pass_share']:.6%}"
    )

    print(
        f"Portfolio pairs: "
        f"{qa['pair_row_count']:,}"
    )

    print(
        f"Non-initial pairs: "
        f"{qa['noninitial_pair_count']:,}"
    )

    print(
        "Both-leg PASS share "
        "among non-initial pairs: "
        f"{qa['both_legs_pass_share_noninitial']:.6%}"
    )

    print(
        "PASS drift observations: "
        f"{qa['audited_drift_pass_count']:,}"
    )

    print(
        "DELIST cash-carry drift observations: "
        f"{qa['audited_drift_delist_cash_carry_count']:,}"
    )

    print(
        "Canonical freeze pass: "
        f"{qa['canonical_freeze_pass']}"
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