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
# Original Day-8 Step 2
#
# IMPORTANT:
# We freeze the already-constructed current-information
# target portfolios.
#
# Therefore the ONLY substantive change in this rerun is:
#
# old statistical forward return
#       ->
# dedicated portfolio-accounting drift return
# ------------------------------------------------------------

OLD_STEP2_DIR = (
    OUTPUT_ROOT
    / "M1_day8"
    / "02_stage2_trade_reconstruction"
)

TARGET_WEIGHT_PATH = (
    OLD_STEP2_DIR
    / "portfolio_target_weights.parquet"
)

OLD_REBALANCE_SUMMARY_PATH = (
    OLD_STEP2_DIR
    / "portfolio_rebalance_summary.csv"
)


# ------------------------------------------------------------
# Day-8 Step 2B
# ------------------------------------------------------------

STEP2B_DIR = (
    OUTPUT_ROOT
    / "M1_day8"
    / "02b_stage2b_dedicated_drift_return"
)

DEDICATED_DRIFT_RETURN_PATH = (
    STEP2B_DIR
    / "dedicated_drift_return_panel.parquet"
)

DEDICATED_DRIFT_FORMAL_QA_PATH = (
    STEP2B_DIR
    / "dedicated_drift_return_formal_qa.csv"
)


# ------------------------------------------------------------
# Day-7 Step 4 comparison
# ------------------------------------------------------------

DAY7_STEP4_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "M1_day7"
    / "04_stage4_turnover_transaction_cost"
    / "turnover_economic_summary.csv"
)


# ------------------------------------------------------------
# New rerun output
# ------------------------------------------------------------

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day8"
    / "02c_stage2_rerun_with_dedicated_drift"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRADE_PANEL_PATH = (
    OUTPUT_DIR
    / "portfolio_trade_weight_panel_dedicated_drift.parquet"
)

REBALANCE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "portfolio_rebalance_summary_dedicated_drift.csv"
)

AUM_SCENARIO_PATH = (
    OUTPUT_DIR
    / "portfolio_trade_value_scenario_summary_dedicated_drift.csv"
)

DRIFT_ISSUE_PATH = (
    OUTPUT_DIR
    / "portfolio_drift_issue_detail.csv"
)

PAIR_SUMMARY_PATH = (
    OUTPUT_DIR
    / "portfolio_pair_rebalance_summary.csv"
)

OLD_COMPARISON_PATH = (
    OUTPUT_DIR
    / "old_vs_dedicated_drift_comparison.csv"
)

OLD_TRANSITION_PATH = (
    OUTPUT_DIR
    / "old_vs_dedicated_drift_transition_summary.csv"
)

STEP4_COMPARISON_PATH = (
    OUTPUT_DIR
    / "step4_turnover_reconstruction_comparison_dedicated_drift.csv"
)

QA_PATH = (
    OUTPUT_DIR
    / "portfolio_trade_reconstruction_qa_dedicated_drift.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day8_step2_dedicated_drift_metadata.json"
)


# ============================================================
# 1. Frozen conventions
# ============================================================

AUM_SCENARIOS_CNY = [
    1e8,   # 1亿元
    5e8,   # 5亿元
    1e9,   # 10亿元
    2e9,   # 20亿元
    5e9,   # 50亿元
    1e10,  # 100亿元
]


REFERENCE_NAV_100M_CNY = 1e8
REFERENCE_NAV_1B_CNY = 1e9


TOL = 1e-12


# ============================================================
# 2. Utilities
# ============================================================

def save_json_atomic(
    obj,
    path: Path,
):
    temp = Path(
        str(path) + ".tmp"
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
        str(path) + ".tmp"
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
        str(path) + ".tmp"
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
        payload.encode("utf-8")
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
        f"Cannot parse bool: {x}"
    )


# ============================================================
# 3. Validate upstream Step-2B QA
# ============================================================

def validate_step2b_upstream():

    if not DEDICATED_DRIFT_FORMAL_QA_PATH.exists():

        raise FileNotFoundError(
            "Missing Step-2B formal QA:\n"
            f"{DEDICATED_DRIFT_FORMAL_QA_PATH}"
        )

    qa = pd.read_csv(
        DEDICATED_DRIFT_FORMAL_QA_PATH
    )

    required = {
        "qa_name",
        "qa_value",
    }

    if not required.issubset(
        qa.columns
    ):
        raise RuntimeError(
            "Unexpected Step-2B QA schema."
        )

    qa_map = dict(
        zip(
            qa["qa_name"],
            qa["qa_value"],
        )
    )

    if (
        "all_formal_qa_pass"
        not in
        qa_map
    ):
        raise RuntimeError(
            "Step-2B QA does not contain "
            "`all_formal_qa_pass`."
        )

    upstream_pass = parse_bool(
        qa_map[
            "all_formal_qa_pass"
        ]
    )

    if not upstream_pass:

        raise RuntimeError(
            "Step-2B formal QA did not pass. "
            "Do not rerun portfolio accounting."
        )

    return qa_map


# ============================================================
# 4. Load frozen target portfolios
# ============================================================

def load_target_weights():

    if not TARGET_WEIGHT_PATH.exists():

        raise FileNotFoundError(
            f"Missing target weights:\n"
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
            "Target portfolio panel missing:\n"
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
            "Duplicate target-weight keys."
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
# 5. Load dedicated drift returns
# ============================================================

def load_dedicated_drift_returns():

    if (
        not
        DEDICATED_DRIFT_RETURN_PATH.exists()
    ):

        raise FileNotFoundError(
            "Missing dedicated drift returns:\n"
            f"{DEDICATED_DRIFT_RETURN_PATH}"
        )

    df = pd.read_parquet(
        DEDICATED_DRIFT_RETURN_PATH
    )

    required = [
        "previous_analysis_date",
        "analysis_date",
        "security_id",
        "drift_total_return",
        "status",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Dedicated drift panel missing:\n"
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
            "Duplicate dedicated-drift keys."
        )

    pass_mask = (
        df[
            "status"
        ]
        ==
        "PASS"
    )

    if (
        df.loc[
            pass_mask,
            "drift_total_return",
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "PASS dedicated-drift rows "
            "contain missing returns."
        )

    if (
        df.loc[
            ~pass_mask,
            "drift_total_return",
        ]
        .notna()
        .any()
    ):

        raise RuntimeError(
            "Non-PASS dedicated-drift rows "
            "unexpectedly contain returns."
        )

    return df


# ============================================================
# 6. Build date-pair drift maps
#
# IMPORTANT:
#
# Dedicated drift return does NOT depend on
# factor / window / method / portfolio leg.
#
# Key:
#
# (previous_analysis_date, analysis_date)
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
                    "drift_total_return",
                    "status",
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
# 7. Reconstruct stock-level trades
# ============================================================

def reconstruct_trades(
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
    issue_rows = []

    portfolio_columns = [
        "window",
        "factor_name",
        "method",
        "portfolio_leg",
    ]

    portfolio_groups = targets.groupby(
        portfolio_columns,
        sort=False,
    )

    total_groups = (
        portfolio_groups.ngroups
    )

    for group_no, (
        keys,
        group,
    ) in enumerate(
        portfolio_groups,
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
                "Trade reconstruction: "
                f"{group_no}/{total_groups}"
            )

        by_date = {}

        for date, temp in group.groupby(
            "analysis_date",
            sort=True,
        ):

            x = (
                temp[
                    [
                        "security_id",
                        "target_weight",
                    ]
                ]
                .copy()
                .set_index(
                    "security_id"
                )
            )

            by_date[
                pd.Timestamp(date)
            ] = x

        dates = sorted(
            by_date.keys()
        )

        if len(dates) == 0:
            continue

        # ----------------------------------------------------
        # Initial portfolio formation
        # ----------------------------------------------------

        first_date = dates[0]

        first_target = (
            by_date[
                first_date
            ][
                "target_weight"
            ]
            .astype(float)
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

                "dedicated_drift_required_n":
                    0,

                "dedicated_drift_missing_n":
                    0,

                "dedicated_drift_missing_weight":
                    0.0,

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
        # Subsequent rebalances
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
                ][
                    "target_weight"
                ]
                .astype(float)
            )

            current_weights = (
                by_date[
                    current_date
                ][
                    "target_weight"
                ]
                .astype(float)
            )

            pair_key = (
                previous_date,
                current_date,
            )

            pair_map = (
                drift_maps.get(
                    pair_key
                )
            )

            # =================================================
            # Missing complete date-pair map
            # =================================================

            if pair_map is None:

                for security_id, weight in (
                    previous_weights.items()
                ):

                    issue_rows.append(
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
                                security_id,

                            "previous_target_weight":
                                float(weight),

                            "dedicated_drift_status":
                                "MISSING_DATE_PAIR_MAP",

                            "drift_total_return":
                                np.nan,
                        }
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
                            "MISSING_DEDICATED_DRIFT_MAP",

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
                                    previous_weights
                                    .index
                                    .union(
                                        current_weights.index
                                    )
                                )
                            ),

                        "dedicated_drift_required_n":
                            int(
                                len(
                                    previous_weights
                                )
                            ),

                        "dedicated_drift_missing_n":
                            int(
                                len(
                                    previous_weights
                                )
                            ),

                        "dedicated_drift_missing_weight":
                            1.0,

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

            # =================================================
            # Align dedicated drift to PREVIOUS portfolio
            # members only.
            # =================================================

            aligned = pair_map.reindex(
                previous_weights.index
            )

            valid = (

                aligned[
                    "status"
                ]
                .eq(
                    "PASS"
                )

                &

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

            missing_mask = (
                ~valid
            )

            missing_n = int(
                missing_mask.sum()
            )

            missing_weight = float(
                previous_weights.loc[
                    missing_mask
                ]
                .sum()
            )

            # =================================================
            # Conservative policy:
            #
            # ANY unresolved previous member
            # -> entire leg rebalance remains incomplete.
            #
            # Do NOT:
            # - fill 0
            # - drop stock
            # - renormalize around missing stock
            # =================================================

            if missing_n > 0:

                bad_ids = (
                    previous_weights
                    .index[
                        missing_mask
                    ]
                )

                for security_id in bad_ids:

                    if (
                        security_id
                        in
                        aligned.index
                    ):

                        drift_status = (
                            aligned.loc[
                                security_id,
                                "status",
                            ]
                        )

                        drift_return = (
                            aligned.loc[
                                security_id,
                                "drift_total_return",
                            ]
                        )

                        if pd.isna(
                            drift_status
                        ):
                            drift_status = (
                                "MISSING_DEDICATED_DRIFT_KEY"
                            )

                    else:

                        drift_status = (
                            "MISSING_DEDICATED_DRIFT_KEY"
                        )

                        drift_return = np.nan

                    issue_rows.append(
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
                                security_id,

                            "previous_target_weight":
                                float(
                                    previous_weights.loc[
                                        security_id
                                    ]
                                ),

                            "dedicated_drift_status":
                                drift_status,

                            "drift_total_return":
                                drift_return,
                        }
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
                            "INCOMPLETE_DEDICATED_DRIFT_RETURN",

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
                                    previous_weights
                                    .index
                                    .union(
                                        current_weights.index
                                    )
                                )
                            ),

                        "dedicated_drift_required_n":
                            int(
                                len(
                                    previous_weights
                                )
                            ),

                        "dedicated_drift_missing_n":
                            missing_n,

                        "dedicated_drift_missing_weight":
                            missing_weight,

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

            # =================================================
            # All previous members have valid dedicated drift
            # =================================================

            previous_returns = (
                aligned[
                    "drift_total_return"
                ]
                .astype(float)
            )

            # -------------------------------------------------
            # Drift previous portfolio wealth
            # -------------------------------------------------

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
                denominator <= 0
            ):

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
                                    previous_weights
                                    .index
                                    .union(
                                        current_weights.index
                                    )
                                )
                            ),

                        "dedicated_drift_required_n":
                            int(
                                len(
                                    previous_weights
                                )
                            ),

                        "dedicated_drift_missing_n":
                            0,

                        "dedicated_drift_missing_weight":
                            0.0,

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

            # -------------------------------------------------
            # Stock-level panel
            # -------------------------------------------------

            previous_return_union = (
                previous_returns.reindex(
                    union_index
                )
            )

            drift_required = (
                previous_target_union
                >
                0
            )

            drift_status_union = pd.Series(
                "NOT_REQUIRED_NEW_ENTRY",
                index=union_index,
                dtype="string",
            )

            drift_status_union.loc[
                drift_required
            ] = "PASS"

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
                        union_index.astype(
                            str
                        ),

                    "previous_target_weight":
                        previous_target_union
                        .to_numpy(
                            dtype=float
                        ),

                    "drift_return_required":
                        drift_required
                        .to_numpy(
                            dtype=bool
                        ),

                    "dedicated_drift_status":
                        drift_status_union
                        .to_numpy(),

                    "holding_period_total_return":
                        previous_return_union
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

                    "dedicated_drift_required_n":
                        int(
                            len(
                                previous_weights
                            )
                        ),

                    "dedicated_drift_missing_n":
                        0,

                    "dedicated_drift_missing_weight":
                        0.0,

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

    issues = pd.DataFrame(
        issue_rows
    )

    return (
        trades,
        summary,
        issues,
    )


# ============================================================
# 8. Build Q1/Q5 pair-level eligibility
# ============================================================

def build_pair_summary(
    rebalance_summary,
):

    key = [
        "analysis_date",
        "previous_analysis_date",
        "window",
        "factor_name",
        "method",
    ]

    status = rebalance_summary.pivot_table(
        index=key,
        columns="portfolio_leg",
        values="status",
        aggfunc="first",
        dropna=False,
    ).reset_index()

    status.columns.name = None

    status = status.rename(
        columns={
            "Q1":
                "q1_status",

            "Q5":
                "q5_status",
        }
    )

    if (
        "q1_status"
        not in status.columns
    ):
        status[
            "q1_status"
        ] = np.nan

    if (
        "q5_status"
        not in status.columns
    ):
        status[
            "q5_status"
        ] = np.nan

    status[
        "both_legs_pass"
    ] = (

        status[
            "q1_status"
        ]
        .eq("PASS")

        &

        status[
            "q5_status"
        ]
        .eq("PASS")
    )

    status[
        "both_initial_formation"
    ] = (

        status[
            "q1_status"
        ]
        .eq(
            "NO_PREVIOUS_FORMATION"
        )

        &

        status[
            "q5_status"
        ]
        .eq(
            "NO_PREVIOUS_FORMATION"
        )
    )

    return status


# ============================================================
# 9. AUM scenario summary
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

    frames = []

    for aum in (
        AUM_SCENARIOS_CNY
    ):

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

            temp[
                "total_abs_trade_weight"
            ]

            *

            float(
                aum
            )
        )

        frames.append(
            temp
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# 10. Compare old Step 2 vs dedicated-drift rerun
# ============================================================

def compare_with_old_step2(
    new_summary,
):

    if (
        not
        OLD_REBALANCE_SUMMARY_PATH.exists()
    ):

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    old = pd.read_csv(
        OLD_REBALANCE_SUMMARY_PATH
    )

    for column in [
        "analysis_date",
        "previous_analysis_date",
    ]:

        if column in old.columns:

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
        "previous_analysis_date",
        "window",
        "factor_name",
        "method",
        "portfolio_leg",
    ]

    old_keep = (
        old[
            key
            +
            [
                "status",
                "turnover",
            ]
        ]
        .rename(
            columns={
                "status":
                    "old_status",

                "turnover":
                    "old_turnover",
            }
        )
    )

    new_keep = (
        new_summary[
            key
            +
            [
                "status",
                "turnover",
                "dedicated_drift_missing_n",
                "dedicated_drift_missing_weight",
            ]
        ]
        .rename(
            columns={
                "status":
                    "new_status",

                "turnover":
                    "new_turnover",
            }
        )
    )

    comparison = new_keep.merge(
        old_keep,
        on=key,
        how="outer",
        validate="one_to_one",
    )

    comparison[
        "status_transition"
    ] = (

        comparison[
            "old_status"
        ]
        .astype("string")

        +

        " -> "

        +

        comparison[
            "new_status"
        ]
        .astype("string")
    )

    comparison[
        "turnover_difference"
    ] = (

        comparison[
            "new_turnover"
        ]

        -

        comparison[
            "old_turnover"
        ]
    )

    transition = (

        comparison.groupby(
            [
                "old_status",
                "new_status",
            ],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "count"
            }
        )
    )

    return (
        comparison,
        transition,
    )


# ============================================================
# 11. Compare with Day-7 Step-4 aggregate turnover
# ============================================================

def compare_with_day7_step4(
    new_summary,
):

    if (
        not
        DAY7_STEP4_SUMMARY_PATH.exists()
    ):

        return pd.DataFrame()

    old = pd.read_csv(
        DAY7_STEP4_SUMMARY_PATH
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
        c in old.columns
        for c in required
    ):

        return pd.DataFrame()

    valid = new_summary[
        new_summary[
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
                "dedicated_mean_q1_turnover",

            "Q5":
                "dedicated_mean_q5_turnover",
        }
    )

    pivot[
        "dedicated_roundtrip_traded_notional_ratio"
    ] = (

        2.0

        *

        (
            pivot[
                "dedicated_mean_q1_turnover"
            ]

            +

            pivot[
                "dedicated_mean_q5_turnover"
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
            "dedicated_mean_q1_turnover"
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
            "dedicated_mean_q5_turnover"
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
            "dedicated_roundtrip_traded_notional_ratio"
        ]

        -

        result[
            "mean_roundtrip_traded_notional_ratio"
        ]
    )

    return result


# ============================================================
# 12. Formal QA
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
    # Frozen target portfolio weights
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
    # Dedicated drift source
    # --------------------------------------------------------

    drift_key = [
        "previous_analysis_date",
        "analysis_date",
        "security_id",
    ]

    qa[
        "dedicated_drift_keys_unique"
    ] = bool(
        not drift[
            drift_key
        ]
        .duplicated()
        .any()
    )

    qa[
        "dedicated_drift_source_pass_share"
    ] = float(
        (
            drift[
                "status"
            ]
            ==
            "PASS"
        )
        .mean()
    )

    qa[
        "dedicated_drift_source_nonpass_count"
    ] = int(
        (
            drift[
                "status"
            ]
            !=
            "PASS"
        )
        .sum()
    )

    qa[
        "upstream_step2b_formal_qa_pass"
    ] = parse_bool(
        upstream_qa[
            "all_formal_qa_pass"
        ]
    )

    # --------------------------------------------------------
    # Rebalance status coverage
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
        "rebalance_pass_share"
    ] = float(
        (
            summary[
                "status"
            ]
            ==
            "PASS"
        )
        .mean()
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
        "incomplete_dedicated_drift_rebalance_count"
    ] = int(
        (
            summary[
                "status"
            ]
            ==
            "INCOMPLETE_DEDICATED_DRIFT_RETURN"
        )
        .sum()
    )

    qa[
        "both_legs_pass_count"
    ] = int(
        pair_summary[
            "both_legs_pass"
        ]
        .sum()
    )

    qa[
        "both_legs_pass_share"
    ] = float(
        pair_summary[
            "both_legs_pass"
        ]
        .mean()
    )

    # --------------------------------------------------------
    # Stock-level trade QA
    # --------------------------------------------------------

    if len(
        trades
    ) == 0:

        qa[
            "trade_keys_unique"
        ] = False

        qa[
            "delta_weight_identity_ok"
        ] = False

        qa[
            "pretrade_weight_sum_ok"
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

    else:

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
    # Research governance
    # --------------------------------------------------------

    qa[
        "target_portfolios_recomputed"
    ] = False

    qa[
        "future_outcome_used_to_form_current_portfolio"
    ] = False

    qa[
        "statistical_future_label_used_for_weight_drift"
    ] = False

    qa[
        "dedicated_portfolio_accounting_drift_used"
    ] = True

    qa[
        "missing_drift_return_zero_filled"
    ] = False

    qa[
        "missing_drift_stock_dropped_and_renormalized"
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

    required = [
        "target_weight_sum_ok",
        "dedicated_drift_keys_unique",
        "upstream_step2b_formal_qa_pass",
        "trade_keys_unique",
        "delta_weight_identity_ok",
        "pretrade_weight_sum_ok",
        "turnover_within_0_1",
        "buy_sell_turnover_identity_ok",
        "monetary_notional_identity_ok",
    ]

    qa[
        "all_formal_qa_pass"
    ] = bool(
        all(
            qa[key]
            for key in required
        )
    )

    return qa


# ============================================================
# 13. Main
# ============================================================

def main():

    print("=" * 80)
    print("M1 A-Share Full-Market Stock Network")
    print("Day 8 - Step 2 Rerun")
    print("Dedicated Drift Return Portfolio Reconstruction")
    print("=" * 80)

    # --------------------------------------------------------
    # Upstream QA
    # --------------------------------------------------------

    print()
    print("[1] Validate Step-2B upstream QA")

    upstream_qa = (
        validate_step2b_upstream()
    )

    # --------------------------------------------------------
    # Frozen target portfolios
    # --------------------------------------------------------

    print()
    print("[2] Load frozen Step-2 target portfolios")

    targets = (
        load_target_weights()
    )

    print(
        f"Target rows: {len(targets):,}"
    )

    print(
        "Portfolio specifications: "
        f"{targets[['window', 'factor_name', 'method', 'portfolio_leg']].drop_duplicates().shape[0]:,}"
    )

    # --------------------------------------------------------
    # Dedicated drift returns
    # --------------------------------------------------------

    print()
    print("[3] Load dedicated drift returns")

    drift = (
        load_dedicated_drift_returns()
    )

    print(
        f"Dedicated drift rows: "
        f"{len(drift):,}"
    )

    print(
        "Dedicated PASS share: "
        f"{(drift['status'] == 'PASS').mean():.8%}"
    )

    # --------------------------------------------------------
    # Reconstruct
    # --------------------------------------------------------

    print()
    print(
        "[4] Reconstruct drift-adjusted "
        "portfolio trades"
    )

    (
        trades,
        rebalance_summary,
        issues,
    ) = reconstruct_trades(
        targets,
        drift,
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
        issues,
        DRIFT_ISSUE_PATH,
    )

    # --------------------------------------------------------
    # Q1/Q5 pair eligibility
    # --------------------------------------------------------

    print()
    print("[5] Build Q1/Q5 pair-level summary")

    pair_summary = (
        build_pair_summary(
            rebalance_summary
        )
    )

    save_csv_atomic(
        pair_summary,
        PAIR_SUMMARY_PATH,
    )

    # --------------------------------------------------------
    # AUM
    # --------------------------------------------------------

    print()
    print("[6] Build AUM trade-value scenarios")

    aum_summary = (
        build_aum_scenario_summary(
            rebalance_summary
        )
    )

    save_csv_atomic(
        aum_summary,
        AUM_SCENARIO_PATH,
    )

    # --------------------------------------------------------
    # Old Step-2 comparison
    # --------------------------------------------------------

    print()
    print(
        "[7] Compare old Step-2 "
        "vs dedicated-drift rerun"
    )

    (
        old_comparison,
        transition,
    ) = compare_with_old_step2(
        rebalance_summary
    )

    if len(
        old_comparison
    ) > 0:

        save_csv_atomic(
            old_comparison,
            OLD_COMPARISON_PATH,
        )

        save_csv_atomic(
            transition,
            OLD_TRANSITION_PATH,
        )

        recovered = (

            old_comparison[
                "old_status"
            ]
            .eq(
                "INCOMPLETE_DRIFT_RETURN"
            )

            &

            old_comparison[
                "new_status"
            ]
            .eq(
                "PASS"
            )
        )

        print(
            "Recovered old incomplete "
            f"rebalances: {int(recovered.sum()):,}"
        )

    # --------------------------------------------------------
    # Day-7 Step-4 comparison
    # --------------------------------------------------------

    print()
    print(
        "[8] Compare with Day-7 Step-4 "
        "aggregate turnover"
    )

    step4_comparison = (
        compare_with_day7_step4(
            rebalance_summary
        )
    )

    if len(
        step4_comparison
    ) > 0:

        save_csv_atomic(
            step4_comparison,
            STEP4_COMPARISON_PATH,
        )

        print(
            "Max abs round-trip difference "
            "vs Day-7 Step-4:"
        )

        print(
            step4_comparison[
                "roundtrip_difference"
            ]
            .abs()
            .max()
        )

    # --------------------------------------------------------
    # Formal QA
    # --------------------------------------------------------

    print()
    print("[9] Formal QA")

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

    if (
        not
        qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            "Dedicated-drift Step-2 rerun "
            "formal QA failed.\n"
            f"{qa}"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    design = {
        "research_day":
            8,

        "step":
            "Step2_Rerun_With_Dedicated_Drift_Return",

        "target_portfolio_source":
            str(
                TARGET_WEIGHT_PATH
            ),

        "dedicated_drift_source":
            str(
                DEDICATED_DRIFT_RETURN_PATH
            ),

        "target_portfolios_recomputed":
            False,

        "target_portfolios_frozen":
            True,

        "drift_key":
            (
                "previous_analysis_date, "
                "analysis_date, security_id"
            ),

        "drift_return":
            "drift_total_return",

        "drift_source_type":
            "portfolio_accounting_return",

        "statistical_future_label_used":
            False,

        "drift_formula":
            (
                "pretrade_w_i = "
                "prev_target_w_i * "
                "(1 + drift_total_return_i) "
                "/ sum_j(prev_target_w_j * "
                "(1 + drift_total_return_j))"
            ),

        "trade_weight":
            "delta_w = target_w - pretrade_w",

        "turnover":
            "0.5 * sum(abs(delta_w))",

        "missing_drift_policy":
            (
                "If any previous portfolio member "
                "has unresolved dedicated drift return, "
                "the entire leg rebalance is marked "
                "incomplete. No zero fill, stock drop, "
                "or renormalization is allowed."
            ),

        "aum_scenarios_cny":
            AUM_SCENARIOS_CNY,

        "strategy_nav_definition":
            (
                "Each Q1/Q5 leg notional equals "
                "strategy NAV; two-leg gross exposure 200%."
            ),

        "factor_sign_flip":
            False,

        "factor_selection":
            False,

        "window_selection":
            False,

        "initial_formation_in_turnover":
            False,

        "terminal_liquidation":
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
            "trade_panel":
                str(
                    TRADE_PANEL_PATH
                ),

            "rebalance_summary":
                str(
                    REBALANCE_SUMMARY_PATH
                ),

            "aum_scenario_summary":
                str(
                    AUM_SCENARIO_PATH
                ),

            "drift_issue_detail":
                str(
                    DRIFT_ISSUE_PATH
                ),

            "pair_summary":
                str(
                    PAIR_SUMMARY_PATH
                ),

            "old_step2_comparison":
                str(
                    OLD_COMPARISON_PATH
                ),

            "old_transition_summary":
                str(
                    OLD_TRANSITION_PATH
                ),

            "day7_step4_comparison":
                str(
                    STEP4_COMPARISON_PATH
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

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Dedicated-Drift Step-2 Rerun Summary")
    print("=" * 80)

    print(
        "PASS rebalances: "
        f"{qa['rebalance_pass_count']:,}"
    )

    print(
        "PASS share: "
        f"{qa['rebalance_pass_share']:.4%}"
    )

    print(
        "Incomplete dedicated-drift rebalances: "
        f"{qa['incomplete_dedicated_drift_rebalance_count']:,}"
    )

    print(
        "Both-leg PASS count: "
        f"{qa['both_legs_pass_count']:,}"
    )

    print(
        "Both-leg PASS share: "
        f"{qa['both_legs_pass_share']:.4%}"
    )

    print(
        "Dedicated source non-PASS rows: "
        f"{qa['dedicated_drift_source_nonpass_count']:,}"
    )

    print(
        "Formal QA pass: "
        f"{qa['all_formal_qa_pass']}"
    )

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    print()
    print("=" * 80)
    print("Day 8 Step 2 Dedicated-Drift Rerun Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()