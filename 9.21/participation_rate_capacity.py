from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


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
# IMPORTANT:
#
# According to the previously frozen project convention:
#
# Day 8 physical outputs are stored under:
#
#     D:\M1_StockNetwork\output\M1_day8
# ------------------------------------------------------------

DAY8_ROOT = (
    OUTPUT_ROOT
    / "M1_day8"
)


# ============================================================
# Step 1: PIT liquidity
# ============================================================

STEP1_DIR = (
    DAY8_ROOT
    / "01_stage1_liquidity_panel"
)

LIQUIDITY_PANEL_PATH = (
    STEP1_DIR
    / "stock_liquidity_panel.parquet"
)


# ============================================================
# Canonical Step 2
# ============================================================

CANONICAL_STEP2_DIR = (
    DAY8_ROOT
    / "02e_stage2_final_canonical_trade_reconstruction"
)

CANONICAL_TRADE_PATH = (
    CANONICAL_STEP2_DIR
    / "canonical_portfolio_trade_weight_panel.parquet"
)

CANONICAL_QA_PATH = (
    CANONICAL_STEP2_DIR
    / "canonical_portfolio_trade_reconstruction_qa.csv"
)


# ============================================================
# Raw market data
#
# One CSV per market trading day:
#
# D:\lowfreq\daily_temp3\20150105.csv
# ...
# ============================================================

RAW_DAILY_DIR = Path(
    r"D:\lowfreq\daily_temp3"
)


# ============================================================
# Market trading calendar
# ============================================================

TRADE_CALENDAR_PATH = Path(
    r"D:\lowfreq\trade_calendar\trade_date.csv"
)


# ============================================================
# Step 3 output
# ============================================================

OUTPUT_DIR = (
    DAY8_ROOT
    / "03_stage3_participation_capacity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


STOCK_BASE_PATH = (
    OUTPUT_DIR
    / "stock_participation_base_panel.parquet"
)

LEG_SUMMARY_PATH = (
    OUTPUT_DIR
    / "participation_leg_summary.csv"
)

PAIR_SUMMARY_PATH = (
    OUTPUT_DIR
    / "participation_pair_summary.csv"
)

FACTOR_WINDOW_SUMMARY_PATH = (
    OUTPUT_DIR
    / "participation_factor_window_summary.csv"
)


# ------------------------------------------------------------
# New trade-universe liquidity supplement
# ------------------------------------------------------------

LIQUIDITY_SUPPLEMENT_PATH = (
    OUTPUT_DIR
    / "trade_universe_liquidity_supplement.parquet"
)

LIQUIDITY_SUPPLEMENT_QA_PATH = (
    OUTPUT_DIR
    / "trade_universe_liquidity_supplement_qa.csv"
)

ORIGINAL_LIQUIDITY_GAP_PATH = (
    OUTPUT_DIR
    / "original_trade_universe_liquidity_gap_detail.csv"
)


# ------------------------------------------------------------
# Post-supplement unresolved diagnostics
# ------------------------------------------------------------

LIQUIDITY_JOIN_ISSUE_PATH = (
    OUTPUT_DIR
    / "liquidity_join_issue_detail.csv"
)

CURRENT_SUSPENSION_PATH = (
    OUTPUT_DIR
    / "current_suspension_trade_detail.csv"
)

ADV60_MISSING_PATH = (
    OUTPUT_DIR
    / "adv60_missing_trade_detail.csv"
)


QA_PATH = (
    OUTPUT_DIR
    / "participation_capacity_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day8_step3_metadata.json"
)


# ============================================================
# 1. Frozen research design
# ============================================================

# ------------------------------------------------------------
# Monetary-unit audit
#
# Independently verified:
#
# turnoverValue / (turnoverVol * vwap) == 1
#
# on:
#     599,920 stock-day observations
#     150 sampled trading dates
#     2015-01-05 through 2026-09-03
#
# Therefore:
#
#     source turnoverValue scale -> CNY = 1.0
#
# The audit is completed before Step 3 and is NOT inferred
# from the magnitude of ADV.
# ------------------------------------------------------------

VALUE_UNIT_CONFIRMED = True

VALUE_SCALE_TO_CNY = 1.0

VALUE_SOURCE_FIELD = "turnoverValue"

VALUE_UNIT_AUDIT_N = 599_920

VALUE_UNIT_AUDIT_DATE_N = 150

VALUE_UNIT_AUDIT_START = "2015-01-05"

VALUE_UNIT_AUDIT_END = "2026-09-03"


# ------------------------------------------------------------
# Frozen AUM scenarios
#
# A = strategy NAV.
#
# Each Q1/Q5 leg has gross notional A.
# Hypothetical Q1-Q5 gross exposure = 200%.
# ------------------------------------------------------------

AUM_SCENARIOS_CNY = [
    1e8,     # 1亿元
    5e8,     # 5亿元
    1e9,     # 10亿元
    2e9,     # 20亿元
    5e9,     # 50亿元
    1e10,    # 100亿元
]


# ------------------------------------------------------------
# Frozen participation thresholds
# ------------------------------------------------------------

PARTICIPATION_CAPS = [
    0.01,
    0.05,
    0.10,
]


PRIMARY_LIQUIDITY_BASIS = "ADV20"

ROBUSTNESS_LIQUIDITY_BASIS = "ADV60"


# ------------------------------------------------------------
# Main Step-3 execution horizon
#
# One current trading day.
#
# No future reopening information is used.
# ------------------------------------------------------------

EXECUTION_DAYS = 1


TRADE_WEIGHT_TOL = 1e-15

QA_TOL = 1e-12


VALID_DRIFT_STATUSES = {
    "PASS",
    "PASS_DELIST_CASH_CARRY",
}


# ============================================================
# 2. Column aliases
# ============================================================

DATE_ALIASES = [
    "analysis_date",
    "trade_date",
    "tradeDate",
    "calendarDate",
    "date",
]

SECURITY_ALIASES = [
    "security_id",
    "secID",
    "sec_id",
]

ADV20_ALIASES = [
    "adv20",
    "ADV20",
    "adv_20",
    "adv20_value",
    "value_adv20",
    "mean_value_20",
    "adv_value_20",
    "adv20_source_native",
]

ADV60_ALIASES = [
    "adv60",
    "ADV60",
    "adv_60",
    "adv60_value",
    "value_adv60",
    "mean_value_60",
    "adv_value_60",
    "adv60_source_native",
]

CURRENT_OPEN_ALIASES = [
    "current_is_open",
    "current_isOpen",
    "is_open_current",
    "isOpen_current",
    "isOpen",
    "is_open",
]


# ============================================================
# 3. Utility functions
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


def save_parquet_atomic(
    df: pd.DataFrame,
    path: Path,
) -> None:

    tmp = Path(
        str(path)
        +
        ".tmp"
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


def normalize_security_id(
    x: pd.Series,
) -> pd.Series:

    return (
        x.astype("string")
        .str.strip()
    )


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


def resolve_column(
    columns,
    aliases,
    field_name: str,
) -> str:

    columns = list(
        columns
    )

    for candidate in aliases:

        if candidate in columns:

            return candidate

    raise RuntimeError(
        f"Cannot resolve `{field_name}`.\n\n"
        f"Available columns:\n"
        f"{columns}"
    )


def normalize_binary_open(
    x: pd.Series,
) -> pd.Series:

    numeric = pd.to_numeric(
        x,
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=x.index,
        dtype=float,
    )

    valid_numeric = (
        numeric.isin(
            [
                0,
                1,
            ]
        )
    )

    result.loc[
        valid_numeric
    ] = numeric.loc[
        valid_numeric
    ]

    text = (
        x.astype("string")
        .str.strip()
        .str.lower()
    )

    result.loc[
        text.isin(
            [
                "true",
                "yes",
                "y",
                "open",
            ]
        )
    ] = 1.0

    result.loc[
        text.isin(
            [
                "false",
                "no",
                "n",
                "closed",
                "close",
            ]
        )
    ] = 0.0

    return result


# ============================================================
# 4. Validate Canonical Step 2
# ============================================================

def validate_canonical_step2():

    if not CANONICAL_QA_PATH.exists():

        raise FileNotFoundError(
            "Canonical Step-2 QA file not found:\n"
            f"{CANONICAL_QA_PATH}"
        )

    qa = pd.read_csv(
        CANONICAL_QA_PATH
    )

    if not {
        "qa_name",
        "qa_value",
    }.issubset(
        qa.columns
    ):

        raise RuntimeError(
            "Unexpected Canonical Step-2 QA schema."
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
        "canonical_freeze_pass",
        "all_noninitial_rebalances_pass",
        "all_noninitial_pairs_both_legs_pass",
        "delta_weight_identity_ok",
        "pretrade_weight_sum_ok",
        "monetary_notional_identity_ok",
    ]

    for name in required_true:

        if name not in qa_map:

            raise RuntimeError(
                "Canonical Step-2 QA missing:\n"
                f"{name}"
            )

        if not parse_bool(
            qa_map[
                name
            ]
        ):

            raise RuntimeError(
                "Canonical Step-2 QA failed:\n"
                f"{name} = "
                f"{qa_map[name]}"
            )

    return qa_map


# ============================================================
# 5. Load Canonical Step-2 stock-level trades
# ============================================================

def load_trade_panel():

    if not CANONICAL_TRADE_PATH.exists():

        raise FileNotFoundError(
            "Canonical trade panel not found:\n"
            f"{CANONICAL_TRADE_PATH}"
        )

    required = [
        "analysis_date",
        "previous_analysis_date",
        "window",
        "factor_name",
        "method",
        "portfolio_leg",
        "security_id",
        "dedicated_drift_status",
        "pretrade_weight",
        "target_weight",
        "delta_weight",
        "abs_delta_weight",
        "buy_weight",
        "sell_weight",
        "trade_side",
    ]

    df = pd.read_parquet(
        CANONICAL_TRADE_PATH,
        columns=required,
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

    numeric_columns = [
        "pretrade_weight",
        "target_weight",
        "delta_weight",
        "abs_delta_weight",
        "buy_weight",
        "sell_weight",
    ]

    for column in numeric_columns:

        df[
            column
        ] = pd.to_numeric(
            df[
                column
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
            "Duplicate canonical stock-trade keys."
        )

    delta_error = (

        df[
            "delta_weight"
        ]

        -

        (
            df[
                "target_weight"
            ]

            -

            df[
                "pretrade_weight"
            ]
        )
    ).abs()

    if (
        delta_error.max()
        >
        1e-14
    ):

        raise RuntimeError(
            "Canonical delta-weight identity failed."
        )

    return df


# ============================================================
# 6. Load Step-1 PIT liquidity
# ============================================================

def load_liquidity_panel():

    if not LIQUIDITY_PANEL_PATH.exists():

        raise FileNotFoundError(
            "Step-1 liquidity panel not found:\n"
            f"{LIQUIDITY_PANEL_PATH}"
        )

    dataset = ds.dataset(
        str(
            LIQUIDITY_PANEL_PATH
        ),
        format="parquet",
    )

    columns = (
        dataset.schema.names
    )

    date_col = resolve_column(
        columns,
        DATE_ALIASES,
        "analysis_date",
    )

    security_col = resolve_column(
        columns,
        SECURITY_ALIASES,
        "security_id",
    )

    adv20_col = resolve_column(
        columns,
        ADV20_ALIASES,
        "ADV20",
    )

    adv60_col = resolve_column(
        columns,
        ADV60_ALIASES,
        "ADV60",
    )

    open_col = resolve_column(
        columns,
        CURRENT_OPEN_ALIASES,
        "current_is_open",
    )

    print()
    print(
        "Resolved Step-1 liquidity columns:"
    )

    print(
        f"  analysis_date : {date_col}"
    )

    print(
        f"  security_id   : {security_col}"
    )

    print(
        f"  ADV20         : {adv20_col}"
    )

    print(
        f"  ADV60         : {adv60_col}"
    )

    print(
        f"  current open  : {open_col}"
    )

    table = dataset.to_table(
        columns=[
            date_col,
            security_col,
            adv20_col,
            adv60_col,
            open_col,
        ]
    )

    df = table.to_pandas()

    df = df.rename(
        columns={
            date_col:
                "analysis_date",

            security_col:
                "security_id",

            adv20_col:
                "adv20_source_native",

            adv60_col:
                "adv60_source_native",

            open_col:
                "current_is_open",
        }
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
        "adv20_source_native"
    ] = pd.to_numeric(
        df[
            "adv20_source_native"
        ],
        errors="coerce",
    )

    df[
        "adv60_source_native"
    ] = pd.to_numeric(
        df[
            "adv60_source_native"
        ],
        errors="coerce",
    )

    df[
        "current_is_open"
    ] = normalize_binary_open(
        df[
            "current_is_open"
        ]
    )

    key = [
        "analysis_date",
        "security_id",
    ]

    if df[
        key
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate stock-date keys "
            "in Step-1 liquidity panel."
        )

    if (
        df[
            "adv20_source_native"
        ]
        .dropna()
        <
        0
    ).any():

        raise RuntimeError(
            "Negative ADV20 detected."
        )

    if (
        df[
            "adv60_source_native"
        ]
        .dropna()
        <
        0
    ).any():

        raise RuntimeError(
            "Negative ADV60 detected."
        )

    return df


# ============================================================
# 7. Load market trading calendar
# ============================================================

def load_trade_calendar():

    if not TRADE_CALENDAR_PATH.exists():

        raise FileNotFoundError(
            "Trading calendar not found:\n"
            f"{TRADE_CALENDAR_PATH}"
        )

    raw = pd.read_csv(
        TRADE_CALENDAR_PATH
    )

    date_col = resolve_column(
        raw.columns,
        DATE_ALIASES,
        "trade calendar date",
    )

    calendar = pd.DataFrame(
        {
            "trade_date":
                pd.to_datetime(
                    raw[
                        date_col
                    ],
                    errors="coerce",
                )
        }
    )

    # --------------------------------------------------------
    # If the calendar contains an open flag, keep market-open
    # dates only. If not, assume the file itself already stores
    # trading dates only.
    # --------------------------------------------------------

    open_col = None

    for candidate in [
        "isOpen",
        "is_open",
    ]:

        if candidate in raw.columns:

            open_col = candidate

            break

    if open_col is not None:

        market_open = pd.to_numeric(
            raw[
                open_col
            ],
            errors="coerce",
        )

        calendar = calendar[
            market_open
            ==
            1
        ].copy()

    calendar = (
        calendar[
            calendar[
                "trade_date"
            ]
            .notna()
        ]
        .drop_duplicates(
            subset=[
                "trade_date"
            ]
        )
        .sort_values(
            "trade_date"
        )
        .reset_index(
            drop=True
        )
    )

    calendar[
        "market_index"
    ] = np.arange(
        len(
            calendar
        ),
        dtype=np.int64,
    )

    return calendar


# ============================================================
# 8. Current-market execution demand
# ============================================================

def build_execution_demand(
    trades: pd.DataFrame,
):

    df = trades.copy()

    delist_cash = (
        df[
            "dedicated_drift_status"
        ]
        ==
        "PASS_DELIST_CASH_CARRY"
    )

    # --------------------------------------------------------
    # A stock already converted to cash cannot be in the
    # current target portfolio.
    # --------------------------------------------------------

    invalid_target = (
        delist_cash
        &
        (
            df[
                "target_weight"
            ]
            >
            TRADE_WEIGHT_TOL
        )
    )

    if invalid_target.any():

        bad = df.loc[
            invalid_target,
            [
                "analysis_date",
                "security_id",
                "factor_name",
                "window",
                "method",
                "portfolio_leg",
                "target_weight",
            ],
        ].head(
            20
        )

        raise RuntimeError(
            "PASS_DELIST_CASH_CARRY security "
            "appears in current target portfolio:\n\n"
            f"{bad}"
        )

    # --------------------------------------------------------
    # Normal rows:
    #
    # capacity delta = canonical delta.
    #
    # Delisted cash-carry rows:
    #
    # wealth is already cash before current rebalance,
    # therefore there is NO current-date stock SELL.
    # --------------------------------------------------------

    df[
        "capacity_delta_weight"
    ] = df[
        "delta_weight"
    ]

    df.loc[
        delist_cash,
        "capacity_delta_weight",
    ] = 0.0

    df[
        "capacity_abs_trade_weight"
    ] = (
        df[
            "capacity_delta_weight"
        ]
        .abs()
    )

    df[
        "cash_redeployment_weight"
    ] = np.where(
        delist_cash,
        df[
            "pretrade_weight"
        ],
        0.0,
    )

    df[
        "capacity_trade_side"
    ] = np.where(
        delist_cash,

        "CASH_CARRY_NO_CURRENT_TRADE",

        np.where(
            df[
                "capacity_delta_weight"
            ]
            >
            TRADE_WEIGHT_TOL,

            "BUY",

            np.where(
                df[
                    "capacity_delta_weight"
                ]
                <
                -TRADE_WEIGHT_TOL,

                "SELL",

                "NONE",
            ),
        ),
    )

    return df


# ============================================================
# 9. Build strict PIT trade-universe liquidity supplement
# ============================================================

def build_trade_universe_liquidity_supplement(
    missing_keys: pd.DataFrame,
    calendar: pd.DataFrame,
):

    keys = (
        missing_keys[
            [
                "analysis_date",
                "security_id",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    keys[
        "analysis_date"
    ] = pd.to_datetime(
        keys[
            "analysis_date"
        ],
        errors="raise",
    )

    keys[
        "security_id"
    ] = normalize_security_id(
        keys[
            "security_id"
        ]
    )

    print(
        "Unique liquidity-supplement keys: "
        f"{len(keys):,}"
    )

    if len(
        keys
    ) == 0:

        return pd.DataFrame(
            columns=[
                "analysis_date",
                "security_id",
                "adv20_source_native",
                "adv60_source_native",
                "current_is_open",
                "adv20_observed_days",
                "adv20_resolved_days",
                "adv20_full_coverage",
                "adv60_observed_days",
                "adv60_resolved_days",
                "adv60_full_coverage",
                "current_record_present",
                "liquidity_source",
            ]
        )

    # ========================================================
    # Trading-calendar index
    # ========================================================

    calendar_dates = (
        calendar[
            "trade_date"
        ]
        .tolist()
    )

    calendar_index = {
        pd.Timestamp(date):
            i

        for i, date
        in enumerate(
            calendar_dates
        )
    }

    # ========================================================
    # Build strict 20/60 MARKET-DAY windows ending at t.
    #
    # Current analysis date is included.
    # ========================================================

    analysis_windows = {}

    for analysis_date in (
        keys[
            "analysis_date"
        ]
        .drop_duplicates()
        .sort_values()
    ):

        analysis_date = pd.Timestamp(
            analysis_date
        )

        if (
            analysis_date
            not in
            calendar_index
        ):

            raise RuntimeError(
                "Analysis date is not in trading calendar:\n"
                f"{analysis_date}"
            )

        index = (
            calendar_index[
                analysis_date
            ]
        )

        dates20 = (
            calendar_dates[
                max(
                    0,
                    index - 19,
                ):
                index + 1
            ]
        )

        dates60 = (
            calendar_dates[
                max(
                    0,
                    index - 59,
                ):
                index + 1
            ]
        )

        analysis_windows[
            analysis_date
        ] = {
            "dates20":
                dates20,

            "dates60":
                dates60,
        }

    # ========================================================
    # Determine the stock subset required from every raw date.
    #
    # Read each daily CSV only once.
    # ========================================================

    securities_by_analysis_date = {

        pd.Timestamp(
            analysis_date
        ):
            set(
                group[
                    "security_id"
                ]
                .astype(str)
                .tolist()
            )

        for analysis_date, group
        in keys.groupby(
            "analysis_date"
        )
    }

    securities_needed_by_raw_date = (
        defaultdict(
            set
        )
    )

    for (
        analysis_date,
        security_ids,
    ) in securities_by_analysis_date.items():

        dates60 = (
            analysis_windows[
                analysis_date
            ][
                "dates60"
            ]
        )

        for raw_date in dates60:

            securities_needed_by_raw_date[
                pd.Timestamp(
                    raw_date
                )
            ].update(
                security_ids
            )

    # ========================================================
    # Read required raw CSVs
    # ========================================================

    daily_lookup = {}

    raw_dates = sorted(
        securities_needed_by_raw_date.keys()
    )

    print(
        "Raw market dates required for supplement: "
        f"{len(raw_dates):,}"
    )

    missing_raw_files = []

    for number, raw_date in enumerate(
        raw_dates,
        start=1,
    ):

        if (
            number == 1
            or
            number % 250 == 0
            or
            number == len(
                raw_dates
            )
        ):

            print(
                "Reading raw liquidity data: "
                f"{number:,}/"
                f"{len(raw_dates):,}"
            )

        file_path = (
            RAW_DAILY_DIR
            /
            (
                raw_date.strftime(
                    "%Y%m%d"
                )
                +
                ".csv"
            )
        )

        needed_security_ids = (
            securities_needed_by_raw_date[
                raw_date
            ]
        )

        if not file_path.exists():

            missing_raw_files.append(
                str(
                    file_path
                )
            )

            continue

        raw = pd.read_csv(
            file_path,
            usecols=[
                "secID",
                "turnoverValue",
                "isOpen",
            ],
            low_memory=False,
        )

        raw[
            "secID"
        ] = (
            raw[
                "secID"
            ]
            .astype("string")
            .str.strip()
        )

        raw = raw[
            raw[
                "secID"
            ]
            .isin(
                needed_security_ids
            )
        ].copy()

        if raw[
            "secID"
        ].duplicated().any():

            duplicated_ids = (
                raw.loc[
                    raw[
                        "secID"
                    ]
                    .duplicated(
                        keep=False
                    ),
                    "secID",
                ]
                .tolist()
            )

            raise RuntimeError(
                "Duplicate secID rows found in raw "
                f"daily file:\n"
                f"{file_path}\n\n"
                f"Examples:\n"
                f"{duplicated_ids[:20]}"
            )

        raw[
            "turnoverValue"
        ] = pd.to_numeric(
            raw[
                "turnoverValue"
            ],
            errors="coerce",
        )

        raw[
            "isOpen"
        ] = pd.to_numeric(
            raw[
                "isOpen"
            ],
            errors="coerce",
        )

        for row in raw.itertuples(
            index=False
        ):

            security_id = str(
                row.secID
            )

            is_open = (
                row.isOpen
            )

            turnover_value = (
                row.turnoverValue
            )

            # --------------------------------------------
            # EXACT Step-1 capacity-value semantics:
            #
            # closed -> 0
            #
            # open + valid turnoverValue -> turnoverValue
            #
            # otherwise unresolved
            # --------------------------------------------

            if (
                pd.notna(
                    is_open
                )
                and
                float(
                    is_open
                )
                ==
                0.0
            ):

                cap_value = 0.0

                resolved = True

            elif (
                pd.notna(
                    is_open
                )
                and
                float(
                    is_open
                )
                ==
                1.0
                and
                pd.notna(
                    turnover_value
                )
                and
                np.isfinite(
                    float(
                        turnover_value
                    )
                )
                and
                float(
                    turnover_value
                )
                >=
                0.0
            ):

                cap_value = float(
                    turnover_value
                )

                resolved = True

            else:

                cap_value = np.nan

                resolved = False

            daily_lookup[
                (
                    pd.Timestamp(
                        raw_date
                    ),
                    security_id,
                )
            ] = {
                "is_open":
                    (
                        float(
                            is_open
                        )
                        if pd.notna(
                            is_open
                        )
                        else np.nan
                    ),

                "cap_value":
                    cap_value,

                "resolved":
                    bool(
                        resolved
                    ),
            }

    if len(
        missing_raw_files
    ) > 0:

        print()
        print(
            "WARNING: missing raw daily files:"
        )

        for item in (
            missing_raw_files[
                :20
            ]
        ):

            print(
                item
            )

    # ========================================================
    # Calculate strict ADV20 / ADV60 for every missing key
    # ========================================================

    output_rows = []

    for number, row in enumerate(
        keys.itertuples(
            index=False
        ),
        start=1,
    ):

        if (
            number == 1
            or
            number % 500 == 0
            or
            number == len(
                keys
            )
        ):

            print(
                "Constructing supplement ADV: "
                f"{number:,}/"
                f"{len(keys):,}"
            )

        analysis_date = pd.Timestamp(
            row.analysis_date
        )

        security_id = str(
            row.security_id
        )

        windows = (
            analysis_windows[
                analysis_date
            ]
        )

        def calculate_adv(
            dates,
            required_n,
        ):

            observed_n = 0

            resolved_n = 0

            values = []

            for date in dates:

                item = daily_lookup.get(
                    (
                        pd.Timestamp(
                            date
                        ),
                        security_id,
                    )
                )

                if item is None:

                    continue

                observed_n += 1

                if item[
                    "resolved"
                ]:

                    resolved_n += 1

                    values.append(
                        item[
                            "cap_value"
                        ]
                    )

            full_coverage = bool(
                len(
                    dates
                )
                ==
                required_n
                and
                observed_n
                ==
                required_n
                and
                resolved_n
                ==
                required_n
            )

            if full_coverage:

                adv = float(
                    np.mean(
                        values
                    )
                )

            else:

                adv = np.nan

            return {
                "adv":
                    adv,

                "market_window_n":
                    int(
                        len(
                            dates
                        )
                    ),

                "observed_n":
                    int(
                        observed_n
                    ),

                "resolved_n":
                    int(
                        resolved_n
                    ),

                "full_coverage":
                    full_coverage,
            }

        result20 = calculate_adv(
            windows[
                "dates20"
            ],
            20,
        )

        result60 = calculate_adv(
            windows[
                "dates60"
            ],
            60,
        )

        current_item = daily_lookup.get(
            (
                analysis_date,
                security_id,
            )
        )

        if current_item is None:

            current_record_present = False

            current_is_open = np.nan

        else:

            current_record_present = True

            current_is_open = (
                current_item[
                    "is_open"
                ]
            )

        output_rows.append(
            {
                "analysis_date":
                    analysis_date,

                "security_id":
                    security_id,

                "adv20_source_native":
                    result20[
                        "adv"
                    ],

                "adv60_source_native":
                    result60[
                        "adv"
                    ],

                "current_is_open":
                    current_is_open,

                "adv20_market_window_n":
                    result20[
                        "market_window_n"
                    ],

                "adv20_observed_days":
                    result20[
                        "observed_n"
                    ],

                "adv20_resolved_days":
                    result20[
                        "resolved_n"
                    ],

                "adv20_full_coverage":
                    result20[
                        "full_coverage"
                    ],

                "adv60_market_window_n":
                    result60[
                        "market_window_n"
                    ],

                "adv60_observed_days":
                    result60[
                        "observed_n"
                    ],

                "adv60_resolved_days":
                    result60[
                        "resolved_n"
                    ],

                "adv60_full_coverage":
                    result60[
                        "full_coverage"
                    ],

                "current_record_present":
                    current_record_present,

                "liquidity_source":
                    "TRADE_UNIVERSE_SUPPLEMENT",
            }
        )

    supplement = pd.DataFrame(
        output_rows
    )

    key = [
        "analysis_date",
        "security_id",
    ]

    if supplement[
        key
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate supplement stock-date keys."
        )

    return supplement


# ============================================================
# 10. Supplement QA
# ============================================================

def summarize_liquidity_supplement(
    supplement: pd.DataFrame,
):

    if len(
        supplement
    ) == 0:

        return {
            "supplement_key_count":
                0,

            "supplement_keys_unique":
                True,

            "supplement_current_record_present_share":
                1.0,

            "supplement_adv20_full_coverage_share":
                1.0,

            "supplement_adv60_full_coverage_share":
                1.0,

            "supplement_adv20_missing_count":
                0,

            "supplement_adv60_missing_count":
                0,

            "supplement_current_open_missing_count":
                0,
        }

    qa = {
        "supplement_key_count":
            int(
                len(
                    supplement
                )
            ),

        "supplement_keys_unique":
            bool(
                not supplement[
                    [
                        "analysis_date",
                        "security_id",
                    ]
                ]
                .duplicated()
                .any()
            ),

        "supplement_current_record_present_share":
            float(
                supplement[
                    "current_record_present"
                ]
                .mean()
            ),

        "supplement_adv20_full_coverage_share":
            float(
                supplement[
                    "adv20_full_coverage"
                ]
                .mean()
            ),

        "supplement_adv60_full_coverage_share":
            float(
                supplement[
                    "adv60_full_coverage"
                ]
                .mean()
            ),

        "supplement_adv20_missing_count":
            int(
                supplement[
                    "adv20_source_native"
                ]
                .isna()
                .sum()
            ),

        "supplement_adv60_missing_count":
            int(
                supplement[
                    "adv60_source_native"
                ]
                .isna()
                .sum()
            ),

        "supplement_current_open_missing_count":
            int(
                supplement[
                    "current_is_open"
                ]
                .isna()
                .sum()
            ),
    }

    return qa


# ============================================================
# 11. Merge Step-1 liquidity + trade-universe supplement
# ============================================================

def merge_liquidity_with_trade_universe_supplement(
    execution_panel: pd.DataFrame,
    liquidity: pd.DataFrame,
    calendar: pd.DataFrame,
):

    # ========================================================
    # Stage A:
    # merge original Step-1 liquidity.
    # ========================================================

    df = execution_panel.merge(
        liquidity,
        on=[
            "analysis_date",
            "security_id",
        ],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    df[
        "original_liquidity_joined"
    ] = (
        df[
            "_merge"
        ]
        ==
        "both"
    )

    df = df.drop(
        columns=[
            "_merge"
        ]
    )

    delist_cash = (
        df[
            "dedicated_drift_status"
        ]
        ==
        "PASS_DELIST_CASH_CARRY"
    )

    normal_market_trade = (
        (
            df[
                "capacity_abs_trade_weight"
            ]
            >
            TRADE_WEIGHT_TOL
        )
        &
        ~delist_cash
    )

    # --------------------------------------------------------
    # A real market trade needs primary ADV20 and current-open
    # information.
    #
    # These are exactly the cases requiring supplement.
    # --------------------------------------------------------

    needs_supplement = (
        normal_market_trade
        &
        (
            ~df[
                "original_liquidity_joined"
            ]
            |
            df[
                "adv20_source_native"
            ]
            .isna()
            |
            df[
                "current_is_open"
            ]
            .isna()
        )
    )

    original_gap_detail = (
        df.loc[
            needs_supplement
        ]
        .copy()
    )

    missing_keys = (
        df.loc[
            needs_supplement,
            [
                "analysis_date",
                "security_id",
            ],
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    print(
        "Original trade rows requiring "
        "liquidity supplement: "
        f"{int(needs_supplement.sum()):,}"
    )

    print(
        "Unique trade-universe supplement keys: "
        f"{len(missing_keys):,}"
    )

    # ========================================================
    # Build strict PIT supplement
    # ========================================================

    supplement = (
        build_trade_universe_liquidity_supplement(
            missing_keys,
            calendar,
        )
    )

    # ========================================================
    # Stage B:
    # merge supplement.
    # ========================================================

    if len(
        supplement
    ) == 0:

        df[
            "supplement_adv20"
        ] = np.nan

        df[
            "supplement_adv60"
        ] = np.nan

        df[
            "supplement_current_is_open"
        ] = np.nan

        df[
            "supplement_row_joined"
        ] = False

    else:

        supplement_merge = (
            supplement[
                [
                    "analysis_date",
                    "security_id",
                    "adv20_source_native",
                    "adv60_source_native",
                    "current_is_open",
                ]
            ]
            .rename(
                columns={
                    "adv20_source_native":
                        "supplement_adv20",

                    "adv60_source_native":
                        "supplement_adv60",

                    "current_is_open":
                        "supplement_current_is_open",
                }
            )
        )

        df = df.merge(
            supplement_merge,
            on=[
                "analysis_date",
                "security_id",
            ],
            how="left",
            validate="many_to_one",
            indicator="_supplement_merge",
        )

        df[
            "supplement_row_joined"
        ] = (
            df[
                "_supplement_merge"
            ]
            ==
            "both"
        )

        df = df.drop(
            columns=[
                "_supplement_merge"
            ]
        )

    # ========================================================
    # Fill ONLY missing original values.
    #
    # Never overwrite valid Step-1 values.
    # ========================================================

    fill_adv20 = (
        df[
            "adv20_source_native"
        ]
        .isna()
        &
        df[
            "supplement_adv20"
        ]
        .notna()
    )

    fill_adv60 = (
        df[
            "adv60_source_native"
        ]
        .isna()
        &
        df[
            "supplement_adv60"
        ]
        .notna()
    )

    fill_open = (
        df[
            "current_is_open"
        ]
        .isna()
        &
        df[
            "supplement_current_is_open"
        ]
        .notna()
    )

    df.loc[
        fill_adv20,
        "adv20_source_native",
    ] = df.loc[
        fill_adv20,
        "supplement_adv20",
    ]

    df.loc[
        fill_adv60,
        "adv60_source_native",
    ] = df.loc[
        fill_adv60,
        "supplement_adv60",
    ]

    df.loc[
        fill_open,
        "current_is_open",
    ] = df.loc[
        fill_open,
        "supplement_current_is_open",
    ]

    used_supplement = (
        fill_adv20
        |
        fill_adv60
        |
        fill_open
    )

    df[
        "liquidity_source"
    ] = np.select(
        [
            (
                df[
                    "original_liquidity_joined"
                ]
                &
                ~used_supplement
            ),

            (
                df[
                    "original_liquidity_joined"
                ]
                &
                used_supplement
            ),

            (
                ~df[
                    "original_liquidity_joined"
                ]
                &
                used_supplement
            ),

            (
                ~normal_market_trade
            ),
        ],
        [
            "STEP1_PANEL",

            "STEP1_PANEL_PLUS_SUPPLEMENT",

            "TRADE_UNIVERSE_SUPPLEMENT",

            "NOT_REQUIRED_FOR_CURRENT_MARKET_TRADE",
        ],
        default="UNRESOLVED",
    )

    df = df.drop(
        columns=[
            "supplement_adv20",
            "supplement_adv60",
            "supplement_current_is_open",
        ]
    )

    return (
        df,
        supplement,
        original_gap_detail,
    )


# ============================================================
# 12. Effective one-day ADV
# ============================================================

def build_effective_adv(
    adv: pd.Series,
    current_is_open: pd.Series,
) -> pd.Series:

    result = pd.Series(
        np.nan,
        index=adv.index,
        dtype=float,
    )

    open_mask = (
        current_is_open
        ==
        1
    )

    closed_mask = (
        current_is_open
        ==
        0
    )

    # --------------------------------------------------------
    # Open today:
    # use PIT trailing ADV.
    # --------------------------------------------------------

    result.loc[
        open_mask
    ] = adv.loc[
        open_mask
    ]

    # --------------------------------------------------------
    # Suspended today:
    # immediate one-day execution capacity = 0.
    #
    # This uses NO future reopening information.
    # --------------------------------------------------------

    result.loc[
        closed_mask
    ] = 0.0

    return result


# ============================================================
# 13. Participation base
# ============================================================

def build_participation_base(
    trade_weight: pd.Series,
    effective_adv: pd.Series,
) -> pd.Series:

    result = pd.Series(
        np.nan,
        index=trade_weight.index,
        dtype=float,
    )

    no_trade = (
        trade_weight
        <=
        TRADE_WEIGHT_TOL
    )

    positive_trade = (
        trade_weight
        >
        TRADE_WEIGHT_TOL
    )

    positive_adv = (
        effective_adv
        >
        0
    )

    zero_adv = (
        effective_adv
        ==
        0
    )

    result.loc[
        no_trade
    ] = 0.0

    valid = (
        positive_trade
        &
        positive_adv
    )

    result.loc[
        valid
    ] = (

        trade_weight.loc[
            valid
        ]

        /

        effective_adv.loc[
            valid
        ]
    )

    blocked = (
        positive_trade
        &
        zero_adv
    )

    result.loc[
        blocked
    ] = np.inf

    # Missing effective ADV remains NaN.
    return result


# ============================================================
# 14. Finalize stock-level participation panel
# ============================================================

def finalize_participation_panel(
    df: pd.DataFrame,
):

    df = df.copy()

    # --------------------------------------------------------
    # Convert source-native liquidity to CNY.
    # --------------------------------------------------------

    df[
        "adv20_cny"
    ] = (

        df[
            "adv20_source_native"
        ]

        *

        VALUE_SCALE_TO_CNY
    )

    df[
        "adv60_cny"
    ] = (

        df[
            "adv60_source_native"
        ]

        *

        VALUE_SCALE_TO_CNY
    )

    # --------------------------------------------------------
    # Immediate execution liquidity
    # --------------------------------------------------------

    df[
        "effective_adv20_1d_cny"
    ] = build_effective_adv(
        df[
            "adv20_cny"
        ],
        df[
            "current_is_open"
        ],
    )

    df[
        "effective_adv60_1d_cny"
    ] = build_effective_adv(
        df[
            "adv60_cny"
        ],
        df[
            "current_is_open"
        ],
    )

    # --------------------------------------------------------
    # Base participation:
    #
    # participation(A)
    # =
    # A * participation_per_cny_nav
    # --------------------------------------------------------

    df[
        "participation20_per_cny_nav"
    ] = build_participation_base(
        df[
            "capacity_abs_trade_weight"
        ],
        df[
            "effective_adv20_1d_cny"
        ],
    )

    df[
        "participation60_per_cny_nav"
    ] = build_participation_base(
        df[
            "capacity_abs_trade_weight"
        ],
        df[
            "effective_adv60_1d_cny"
        ],
    )

    # --------------------------------------------------------
    # Convenient reference sizes
    # --------------------------------------------------------

    df[
        "participation20_at_100m"
    ] = (
        df[
            "participation20_per_cny_nav"
        ]
        *
        1e8
    )

    df[
        "participation20_at_1bn"
    ] = (
        df[
            "participation20_per_cny_nav"
        ]
        *
        1e9
    )

    df[
        "participation60_at_100m"
    ] = (
        df[
            "participation60_per_cny_nav"
        ]
        *
        1e8
    )

    df[
        "participation60_at_1bn"
    ] = (
        df[
            "participation60_per_cny_nav"
        ]
        *
        1e9
    )

    return df


# ============================================================
# 15. Weighted quantile
#
# +inf is intentionally retained.
#
# A current suspension may legitimately make an upper
# participation quantile infinite.
# ============================================================

def weighted_quantile(
    values,
    weights,
    q: float,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid = (
        ~np.isnan(
            values
        )
        &
        np.isfinite(
            weights
        )
        &
        (
            weights
            >
            0
        )
    )

    values = values[
        valid
    ]

    weights = weights[
        valid
    ]

    if len(
        values
    ) == 0:

        return np.nan

    order = np.argsort(
        values
    )

    values = values[
        order
    ]

    weights = weights[
        order
    ]

    cumulative = np.cumsum(
        weights
    )

    target = (
        q
        *
        cumulative[-1]
    )

    index = np.searchsorted(
        cumulative,
        target,
        side="left",
    )

    index = min(
        index,
        len(
            values
        )
        -
        1,
    )

    return float(
        values[
            index
        ]
    )


# ============================================================
# 16. Summarize one portfolio
# ============================================================

def summarize_one_group(
    group: pd.DataFrame,
    aum_cny: float,
    liquidity_basis: str,
):

    if liquidity_basis == "ADV20":

        participation_base_col = (
            "participation20_per_cny_nav"
        )

        effective_adv_col = (
            "effective_adv20_1d_cny"
        )

    elif liquidity_basis == "ADV60":

        participation_base_col = (
            "participation60_per_cny_nav"
        )

        effective_adv_col = (
            "effective_adv60_1d_cny"
        )

    else:

        raise ValueError(
            f"Unknown liquidity basis: "
            f"{liquidity_basis}"
        )

    # --------------------------------------------------------
    # Current stock-market trades only.
    # --------------------------------------------------------

    trade = group[
        group[
            "capacity_abs_trade_weight"
        ]
        >
        TRADE_WEIGHT_TOL
    ].copy()

    cash_redeployment_weight = float(
        group[
            "cash_redeployment_weight"
        ]
        .sum()
    )

    result = {
        "strategy_nav_cny":
            float(
                aum_cny
            ),

        "liquidity_basis":
            liquidity_basis,

        "execution_days":
            EXECUTION_DAYS,

        "cash_redeployment_weight":
            cash_redeployment_weight,

        "cash_redeployment_cny":
            float(
                aum_cny
                *
                cash_redeployment_weight
            ),
    }

    if len(
        trade
    ) == 0:

        result.update(
            {
                "trade_name_n":
                    0,

                "evaluable_trade_name_n":
                    0,

                "unknown_liquidity_name_n":
                    0,

                "total_abs_trade_weight":
                    0.0,

                "total_trade_value_cny":
                    0.0,

                "buy_weight":
                    0.0,

                "sell_weight":
                    0.0,

                "buy_notional_cny":
                    0.0,

                "sell_notional_cny":
                    0.0,

                "current_closed_trade_name_n":
                    0,

                "current_closed_trade_value_share":
                    0.0,

                "unknown_liquidity_trade_value_share":
                    0.0,

                "participation_weighted_p50":
                    np.nan,

                "participation_weighted_p90":
                    np.nan,

                "participation_weighted_p95":
                    np.nan,

                "participation_weighted_p99":
                    np.nan,

                "max_participation":
                    np.nan,
            }
        )

        for cap in (
            PARTICIPATION_CAPS
        ):

            pct = int(
                round(
                    cap
                    *
                    100
                )
            )

            result[
                f"over_{pct}pct_name_share_evaluable"
            ] = np.nan

            result[
                f"over_{pct}pct_trade_value_share_evaluable"
            ] = np.nan

            result[
                f"over_{pct}pct_trade_value_share_total"
            ] = 0.0

            result[
                f"executable_trade_value_share_at_{pct}pct"
            ] = 1.0

            result[
                f"unexecuted_trade_value_share_at_{pct}pct"
            ] = 0.0

        return result

    trade_weight = (
        trade[
            "capacity_abs_trade_weight"
        ]
        .to_numpy(
            dtype=float
        )
    )

    trade_value = (
        trade_weight
        *
        float(
            aum_cny
        )
    )

    participation = (
        trade[
            participation_base_col
        ]
        .to_numpy(
            dtype=float
        )
        *
        float(
            aum_cny
        )
    )

    effective_adv = (
        trade[
            effective_adv_col
        ]
        .to_numpy(
            dtype=float
        )
    )

    total_abs_trade_weight = float(
        trade_weight.sum()
    )

    total_trade_value = float(
        trade_value.sum()
    )

    buy_mask = (
        trade[
            "capacity_delta_weight"
        ]
        .to_numpy(
            dtype=float
        )
        >
        TRADE_WEIGHT_TOL
    )

    sell_mask = (
        trade[
            "capacity_delta_weight"
        ]
        .to_numpy(
            dtype=float
        )
        <
        -TRADE_WEIGHT_TOL
    )

    buy_weight = float(
        trade.loc[
            buy_mask,
            "capacity_delta_weight",
        ]
        .sum()
    )

    sell_weight = float(
        -trade.loc[
            sell_mask,
            "capacity_delta_weight",
        ]
        .sum()
    )

    # --------------------------------------------------------
    # +inf is known:
    # zero current execution capacity.
    #
    # NaN is unknown liquidity.
    # --------------------------------------------------------

    known = (
        ~np.isnan(
            participation
        )
    )

    unknown = (
        ~known
    )

    current_closed = (
        trade[
            "current_is_open"
        ]
        .eq(
            0
        )
        .to_numpy()
    )

    known_trade_value = float(
        trade_value[
            known
        ].sum()
    )

    unknown_trade_value = float(
        trade_value[
            unknown
        ].sum()
    )

    result.update(
        {
            "trade_name_n":
                int(
                    len(
                        trade
                    )
                ),

            "evaluable_trade_name_n":
                int(
                    known.sum()
                ),

            "unknown_liquidity_name_n":
                int(
                    unknown.sum()
                ),

            "total_abs_trade_weight":
                total_abs_trade_weight,

            "total_trade_value_cny":
                total_trade_value,

            "buy_weight":
                buy_weight,

            "sell_weight":
                sell_weight,

            "buy_notional_cny":
                float(
                    aum_cny
                    *
                    buy_weight
                ),

            "sell_notional_cny":
                float(
                    aum_cny
                    *
                    sell_weight
                ),

            "current_closed_trade_name_n":
                int(
                    current_closed.sum()
                ),

            "current_closed_trade_value_share":
                (
                    float(
                        trade_value[
                            current_closed
                        ].sum()
                        /
                        total_trade_value
                    )
                    if total_trade_value
                    >
                    0
                    else np.nan
                ),

            "unknown_liquidity_trade_value_share":
                (
                    float(
                        unknown_trade_value
                        /
                        total_trade_value
                    )
                    if total_trade_value
                    >
                    0
                    else np.nan
                ),

            "participation_weighted_p50":
                weighted_quantile(
                    participation,
                    trade_value,
                    0.50,
                ),

            "participation_weighted_p90":
                weighted_quantile(
                    participation,
                    trade_value,
                    0.90,
                ),

            "participation_weighted_p95":
                weighted_quantile(
                    participation,
                    trade_value,
                    0.95,
                ),

            "participation_weighted_p99":
                weighted_quantile(
                    participation,
                    trade_value,
                    0.99,
                ),

            "max_participation":
                (
                    float(
                        np.nanmax(
                            participation
                        )
                    )
                    if known.any()
                    else np.nan
                ),
        }
    )

    # ========================================================
    # Threshold diagnostics
    # ========================================================

    for cap in (
        PARTICIPATION_CAPS
    ):

        pct = int(
            round(
                cap
                *
                100
            )
        )

        over = (
            known
            &
            (
                participation
                >
                cap
            )
        )

        if known.sum() > 0:

            name_share = float(
                over.sum()
                /
                known.sum()
            )

        else:

            name_share = np.nan

        if known_trade_value > 0:

            over_value_share_evaluable = float(
                trade_value[
                    over
                ].sum()
                /
                known_trade_value
            )

        else:

            over_value_share_evaluable = np.nan

        if total_trade_value > 0:

            over_value_share_total = float(
                trade_value[
                    over
                ].sum()
                /
                total_trade_value
            )

        else:

            over_value_share_total = np.nan

        # ----------------------------------------------------
        # Conservative executable amount:
        #
        # min(
        #     required trade,
        #     cap * effective ADV * execution_days
        # )
        #
        # NaN liquidity -> zero executable amount.
        # Suspension -> effective ADV=0 -> zero executable.
        # ----------------------------------------------------

        executable_value = np.zeros(
            len(
                trade
            ),
            dtype=float,
        )

        usable_adv = (
            ~np.isnan(
                effective_adv
            )
            &
            np.isfinite(
                effective_adv
            )
        )

        executable_value[
            usable_adv
        ] = np.minimum(
            trade_value[
                usable_adv
            ],
            (
                cap
                *
                effective_adv[
                    usable_adv
                ]
                *
                EXECUTION_DAYS
            ),
        )

        if total_trade_value > 0:

            executable_share = float(
                executable_value.sum()
                /
                total_trade_value
            )

        else:

            executable_share = np.nan

        result[
            f"over_{pct}pct_name_share_evaluable"
        ] = name_share

        result[
            f"over_{pct}pct_trade_value_share_evaluable"
        ] = over_value_share_evaluable

        result[
            f"over_{pct}pct_trade_value_share_total"
        ] = over_value_share_total

        result[
            f"executable_trade_value_share_at_{pct}pct"
        ] = executable_share

        result[
            f"unexecuted_trade_value_share_at_{pct}pct"
        ] = (
            1.0
            -
            executable_share
            if np.isfinite(
                executable_share
            )
            else np.nan
        )

    return result


# ============================================================
# 17. Build leg/pair summaries
# ============================================================

def build_participation_summaries(
    panel: pd.DataFrame,
):

    leg_rows = []

    pair_rows = []

    leg_key = [
        "analysis_date",
        "previous_analysis_date",
        "window",
        "factor_name",
        "method",
        "portfolio_leg",
    ]

    pair_key = [
        "analysis_date",
        "previous_analysis_date",
        "window",
        "factor_name",
        "method",
    ]

    # ========================================================
    # Q1 / Q5 legs
    # ========================================================

    leg_groups = panel.groupby(
        leg_key,
        sort=False,
        observed=True,
    )

    leg_group_n = (
        leg_groups.ngroups
    )

    for group_number, (
        keys,
        group,
    ) in enumerate(
        leg_groups,
        start=1,
    ):

        if (
            group_number == 1
            or
            group_number % 500 == 0
            or
            group_number == leg_group_n
        ):

            print(
                "Leg participation summary: "
                f"{group_number:,}/"
                f"{leg_group_n:,}"
            )

        base = dict(
            zip(
                leg_key,
                keys,
            )
        )

        for aum in (
            AUM_SCENARIOS_CNY
        ):

            for basis in [
                "ADV20",
                "ADV60",
            ]:

                metrics = (
                    summarize_one_group(
                        group,
                        aum,
                        basis,
                    )
                )

                row = {
                    **base,
                    **metrics,
                }

                row[
                    "scope"
                ] = (
                    "PORTFOLIO_LEG"
                )

                leg_rows.append(
                    row
                )

    # ========================================================
    # Hypothetical Q1-Q5 combined gross-liquidity diagnostic
    #
    # This follows the Day-7 convention of evaluating both
    # legs' gross market trading demand.
    #
    # It does NOT imply actual A-share short borrowability.
    # ========================================================

    pair_groups = panel.groupby(
        pair_key,
        sort=False,
        observed=True,
    )

    pair_group_n = (
        pair_groups.ngroups
    )

    for group_number, (
        keys,
        group,
    ) in enumerate(
        pair_groups,
        start=1,
    ):

        if (
            group_number == 1
            or
            group_number % 500 == 0
            or
            group_number == pair_group_n
        ):

            print(
                "Pair participation summary: "
                f"{group_number:,}/"
                f"{pair_group_n:,}"
            )

        base = dict(
            zip(
                pair_key,
                keys,
            )
        )

        for aum in (
            AUM_SCENARIOS_CNY
        ):

            for basis in [
                "ADV20",
                "ADV60",
            ]:

                metrics = (
                    summarize_one_group(
                        group,
                        aum,
                        basis,
                    )
                )

                row = {
                    **base,
                    **metrics,
                }

                row[
                    "scope"
                ] = (
                    "HYPOTHETICAL_Q1_Q5_PAIR"
                )

                pair_rows.append(
                    row
                )

    return (
        pd.DataFrame(
            leg_rows
        ),
        pd.DataFrame(
            pair_rows
        ),
    )


# ============================================================
# 18. Factor × Window aggregate summary
# ============================================================

def build_factor_window_summary(
    leg_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
):

    leg = (
        leg_summary.copy()
    )

    leg[
        "portfolio_scope"
    ] = leg[
        "portfolio_leg"
    ]

    pair = (
        pair_summary.copy()
    )

    pair[
        "portfolio_scope"
    ] = "Q1_Q5_PAIR"

    combined = pd.concat(
        [
            leg,
            pair,
        ],
        ignore_index=True,
        sort=False,
    )

    group_columns = [
        "scope",
        "portfolio_scope",
        "factor_name",
        "method",
        "window",
        "strategy_nav_cny",
        "liquidity_basis",
    ]

    metrics = [
        "total_abs_trade_weight",
        "total_trade_value_cny",
        "cash_redeployment_weight",
        "current_closed_trade_value_share",
        "unknown_liquidity_trade_value_share",
        "participation_weighted_p50",
        "participation_weighted_p90",
        "participation_weighted_p95",
        "participation_weighted_p99",
    ]

    for cap in (
        PARTICIPATION_CAPS
    ):

        pct = int(
            round(
                cap
                *
                100
            )
        )

        metrics.extend(
            [
                f"over_{pct}pct_name_share_evaluable",

                f"over_{pct}pct_trade_value_share_evaluable",

                f"over_{pct}pct_trade_value_share_total",

                f"executable_trade_value_share_at_{pct}pct",

                f"unexecuted_trade_value_share_at_{pct}pct",
            ]
        )

    rows = []

    groups = combined.groupby(
        group_columns,
        sort=False,
        dropna=False,
        observed=True,
    )

    for keys, group in groups:

        row = dict(
            zip(
                group_columns,
                keys,
            )
        )

        row[
            "rebalance_n"
        ] = int(
            len(
                group
            )
        )

        for metric in metrics:

            values = pd.to_numeric(
                group[
                    metric
                ],
                errors="coerce",
            )

            row[
                f"mean_{metric}"
            ] = float(
                values.mean()
            )

            row[
                f"median_{metric}"
            ] = float(
                values.median()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 19. Formal Step-3 QA
# ============================================================

def formal_qa(
    panel: pd.DataFrame,
    canonical_qa,
    supplement: pd.DataFrame,
    original_gap_detail: pd.DataFrame,
):

    qa = {}

    leg_key = [
        "analysis_date",
        "window",
        "factor_name",
        "method",
        "portfolio_leg",
    ]

    pair_key = [
        "analysis_date",
        "window",
        "factor_name",
        "method",
    ]

    # ========================================================
    # Upstream freeze
    # ========================================================

    qa[
        "canonical_step2_freeze_pass"
    ] = parse_bool(
        canonical_qa[
            "canonical_freeze_pass"
        ]
    )

    qa[
        "value_unit_confirmed"
    ] = bool(
        VALUE_UNIT_CONFIRMED
    )

    qa[
        "value_scale_to_cny"
    ] = float(
        VALUE_SCALE_TO_CNY
    )

    # ========================================================
    # Group coverage
    # ========================================================

    qa[
        "stock_panel_row_count"
    ] = int(
        len(
            panel
        )
    )

    qa[
        "canonical_trade_leg_group_count"
    ] = int(
        panel[
            leg_key
        ]
        .drop_duplicates()
        .shape[0]
    )

    qa[
        "canonical_trade_pair_group_count"
    ] = int(
        panel[
            pair_key
        ]
        .drop_duplicates()
        .shape[0]
    )

    expected_leg_count = int(
        float(
            canonical_qa[
                "noninitial_rebalance_count"
            ]
        )
    )

    expected_pair_count = int(
        float(
            canonical_qa[
                "noninitial_pair_count"
            ]
        )
    )

    qa[
        "expected_noninitial_leg_count"
    ] = (
        expected_leg_count
    )

    qa[
        "expected_noninitial_pair_count"
    ] = (
        expected_pair_count
    )

    qa[
        "trade_leg_group_count_matches_canonical"
    ] = bool(
        qa[
            "canonical_trade_leg_group_count"
        ]
        ==
        expected_leg_count
    )

    qa[
        "trade_pair_group_count_matches_canonical"
    ] = bool(
        qa[
            "canonical_trade_pair_group_count"
        ]
        ==
        expected_pair_count
    )

    # ========================================================
    # Original Step-1 universe mismatch
    #
    # This is diagnostic, NOT a final failure after supplement.
    # ========================================================

    qa[
        "original_trade_rows_requiring_liquidity_supplement"
    ] = int(
        len(
            original_gap_detail
        )
    )

    qa[
        "original_unique_liquidity_supplement_key_count"
    ] = int(
        original_gap_detail[
            [
                "analysis_date",
                "security_id",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    qa[
        "supplement_key_count"
    ] = int(
        len(
            supplement
        )
    )

    # ========================================================
    # Delisting execution mapping
    # ========================================================

    delist_cash = (
        panel[
            "dedicated_drift_status"
        ]
        ==
        "PASS_DELIST_CASH_CARRY"
    )

    qa[
        "delist_cash_carry_stock_rows"
    ] = int(
        delist_cash.sum()
    )

    qa[
        "delist_cash_target_positive_count"
    ] = int(
        (
            delist_cash
            &
            (
                panel[
                    "target_weight"
                ]
                >
                TRADE_WEIGHT_TOL
            )
        )
        .sum()
    )

    qa[
        "delist_cash_nonzero_capacity_trade_count"
    ] = int(
        (
            delist_cash
            &
            (
                panel[
                    "capacity_abs_trade_weight"
                ]
                >
                TRADE_WEIGHT_TOL
            )
        )
        .sum()
    )

    # ========================================================
    # Normal rows must preserve Canonical Step-2 delta
    # ========================================================

    normal = (
        ~delist_cash
    )

    normal_delta_error = (

        panel.loc[
            normal,
            "capacity_delta_weight",
        ]

        -

        panel.loc[
            normal,
            "delta_weight",
        ]
    ).abs()

    qa[
        "max_normal_capacity_delta_change"
    ] = float(
        normal_delta_error.max()
        if len(
            normal_delta_error
        )
        >
        0
        else 0.0
    )

    qa[
        "normal_capacity_delta_unchanged"
    ] = bool(
        qa[
            "max_normal_capacity_delta_change"
        ]
        <
        1e-14
    )

    # --------------------------------------------------------
    # Capacity adjustment can only REMOVE the artificial
    # current-date trade of a delisted cash-carry stock.
    #
    # It must never increase stock-market trade demand.
    # --------------------------------------------------------

    abs_trade_increase = (

        panel[
            "capacity_abs_trade_weight"
        ]

        -

        panel[
            "abs_delta_weight"
        ]
    )

    qa[
        "max_capacity_abs_trade_weight_increase"
    ] = float(
        abs_trade_increase.max()
    )

    qa[
        "capacity_trade_not_inflated"
    ] = bool(
        abs_trade_increase.max()
        <=
        QA_TOL
    )

    # ========================================================
    # Actual market trades
    # ========================================================

    capacity_trade = (
        panel[
            "capacity_abs_trade_weight"
        ]
        >
        TRADE_WEIGHT_TOL
    )

    normal_capacity_trade = (
        capacity_trade
        &
        normal
    )

    qa[
        "capacity_trade_row_count"
    ] = int(
        capacity_trade.sum()
    )

    # ========================================================
    # FINAL primary liquidity coverage after supplement
    # ========================================================

    qa[
        "positive_capacity_trade_missing_adv20_count"
    ] = int(
        (
            normal_capacity_trade
            &
            panel[
                "adv20_cny"
            ]
            .isna()
        )
        .sum()
    )

    qa[
        "positive_capacity_trade_missing_current_open_count"
    ] = int(
        (
            normal_capacity_trade
            &
            panel[
                "current_is_open"
            ]
            .isna()
        )
        .sum()
    )

    qa[
        "positive_capacity_trade_unresolved_primary_liquidity_count"
    ] = int(
        (
            normal_capacity_trade
            &
            (
                panel[
                    "adv20_cny"
                ]
                .isna()
                |
                panel[
                    "current_is_open"
                ]
                .isna()
            )
        )
        .sum()
    )

    # --------------------------------------------------------
    # ADV60 is robustness only.
    # Missing ADV60 is reported but is not a formal failure.
    # --------------------------------------------------------

    qa[
        "positive_capacity_trade_missing_adv60_count"
    ] = int(
        (
            normal_capacity_trade
            &
            panel[
                "adv60_cny"
            ]
            .isna()
        )
        .sum()
    )

    # ========================================================
    # Current suspension
    #
    # Genuine capacity constraint, NOT missing data.
    # ========================================================

    current_suspended_trade = (
        normal_capacity_trade
        &
        panel[
            "current_is_open"
        ]
        .eq(
            0
        )
    )

    qa[
        "current_suspended_trade_row_count"
    ] = int(
        current_suspended_trade.sum()
    )

    if current_suspended_trade.any():

        suspended_participation = (
            panel.loc[
                current_suspended_trade,
                "participation20_per_cny_nav",
            ]
            .to_numpy(
                dtype=float
            )
        )

        qa[
            "all_current_suspended_trades_have_infinite_primary_participation"
        ] = bool(
            np.isposinf(
                suspended_participation
            )
            .all()
        )

    else:

        qa[
            "all_current_suspended_trades_have_infinite_primary_participation"
        ] = True

    # ========================================================
    # Cash-redeployment identity
    #
    # Canonical:
    #
    #     sum(delta_weight) = 0
    #
    # After removing delisted-stock fake SELL:
    #
    #     sum(capacity_delta_weight)
    #     =
    #     cash_redeployment_weight
    # ========================================================

    cash_check = (

        panel.groupby(
            leg_key,
            as_index=False,
            observed=True,
        )
        .agg(
            capacity_delta_sum=(
                "capacity_delta_weight",
                "sum",
            ),

            cash_redeployment_weight=(
                "cash_redeployment_weight",
                "sum",
            ),
        )
    )

    cash_error = (

        cash_check[
            "capacity_delta_sum"
        ]

        -

        cash_check[
            "cash_redeployment_weight"
        ]
    ).abs()

    qa[
        "max_cash_redeployment_identity_error"
    ] = float(
        cash_error.max()
    )

    qa[
        "cash_redeployment_identity_ok"
    ] = bool(
        cash_error.max()
        <
        QA_TOL
    )

    # ========================================================
    # Participation identity
    #
    # At NAV = 1bn:
    #
    # participation =
    #
    # 1e9 * abs(delta_w) / effective_ADV20
    #
    # Use relative+absolute floating-point tolerance rather
    # than an overly strict absolute 1e-14 threshold.
    # ========================================================

    positive_adv20 = (
        capacity_trade
        &
        (
            panel[
                "effective_adv20_1d_cny"
            ]
            >
            0
        )
        &
        panel[
            "effective_adv20_1d_cny"
        ]
        .notna()
    )

    expected_p20_1bn = (

        1e9

        *

        panel.loc[
            positive_adv20,
            "capacity_abs_trade_weight",
        ]

        /

        panel.loc[
            positive_adv20,
            "effective_adv20_1d_cny",
        ]
    )

    actual_p20_1bn = (
        panel.loc[
            positive_adv20,
            "participation20_at_1bn",
        ]
    )

    participation_error = (

        expected_p20_1bn

        -

        actual_p20_1bn
    ).abs()

    qa[
        "max_participation20_1bn_identity_error"
    ] = float(
        participation_error.max()
        if len(
            participation_error
        )
        >
        0
        else 0.0
    )

    if len(
        expected_p20_1bn
    ) > 0:

        qa[
            "participation20_identity_ok"
        ] = bool(
            np.allclose(
                expected_p20_1bn.to_numpy(
                    dtype=float
                ),

                actual_p20_1bn.to_numpy(
                    dtype=float
                ),

                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            )
        )

    else:

        qa[
            "participation20_identity_ok"
        ] = True

    # ========================================================
    # Research governance
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
        "aum_scenarios_selected_from_results"
    ] = False

    qa[
        "participation_caps_selected_from_results"
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
        "short_borrowability_assumed"
    ] = False

    # ========================================================
    # Formal PASS
    #
    # Do NOT require ADV60 completeness.
    #
    # Do NOT require zero current suspensions.
    #
    # Those are real robustness/execution features.
    # ========================================================

    required_true = [
        "canonical_step2_freeze_pass",
        "value_unit_confirmed",
        "trade_leg_group_count_matches_canonical",
        "trade_pair_group_count_matches_canonical",
        "normal_capacity_delta_unchanged",
        "capacity_trade_not_inflated",
        "cash_redeployment_identity_ok",
        "participation20_identity_ok",
        "all_current_suspended_trades_have_infinite_primary_participation",
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
            "delist_cash_target_positive_count"
        ]
        ==
        0

        and

        qa[
            "delist_cash_nonzero_capacity_trade_count"
        ]
        ==
        0

        and

        qa[
            "positive_capacity_trade_unresolved_primary_liquidity_count"
        ]
        ==
        0
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
        "Day 8 - Step 3"
    )

    print(
        "Participation-Rate Capacity Analysis"
    )

    print("=" * 80)

    # ========================================================
    # 1. Monetary-unit confirmation
    # ========================================================

    print()
    print(
        "[1] Confirm monetary unit"
    )

    if not VALUE_UNIT_CONFIRMED:

        raise RuntimeError(
            "Transaction-value unit is not confirmed."
        )

    if (
        not np.isfinite(
            VALUE_SCALE_TO_CNY
        )
        or
        VALUE_SCALE_TO_CNY
        <=
        0
    ):

        raise RuntimeError(
            "Invalid VALUE_SCALE_TO_CNY."
        )

    print(
        f"Source field      : "
        f"{VALUE_SOURCE_FIELD}"
    )

    print(
        f"Scale to CNY      : "
        f"{VALUE_SCALE_TO_CNY}"
    )

    print(
        f"Unit-audit rows   : "
        f"{VALUE_UNIT_AUDIT_N:,}"
    )

    print(
        f"Unit-audit dates  : "
        f"{VALUE_UNIT_AUDIT_DATE_N}"
    )

    print(
        f"Unit-audit period : "
        f"{VALUE_UNIT_AUDIT_START} "
        f"to "
        f"{VALUE_UNIT_AUDIT_END}"
    )

    # ========================================================
    # 2. Canonical Step 2
    # ========================================================

    print()
    print(
        "[2] Validate Canonical Step 2"
    )

    canonical_qa = (
        validate_canonical_step2()
    )

    print(
        "Canonical Step-2 freeze: PASS"
    )

    # ========================================================
    # 3. Load trades
    # ========================================================

    print()
    print(
        "[3] Load Canonical Step-2 trade panel"
    )

    trades = (
        load_trade_panel()
    )

    print(
        "Canonical stock-trade rows: "
        f"{len(trades):,}"
    )

    # ========================================================
    # 4. Execution mapping
    # ========================================================

    print()
    print(
        "[4] Build current-market execution demand"
    )

    execution_panel = (
        build_execution_demand(
            trades
        )
    )

    # ========================================================
    # 5. Original Step-1 liquidity
    # ========================================================

    print()
    print(
        "[5] Load Step-1 PIT liquidity"
    )

    liquidity = (
        load_liquidity_panel()
    )

    print(
        "Step-1 liquidity rows: "
        f"{len(liquidity):,}"
    )

    # ========================================================
    # 6. Trading calendar
    # ========================================================

    print()
    print(
        "[6] Load market trading calendar"
    )

    calendar = (
        load_trade_calendar()
    )

    print(
        "Market trading days: "
        f"{len(calendar):,}"
    )

    # ========================================================
    # 7. Trade-universe supplement
    # ========================================================

    print()
    print(
        "[7] Build trade-universe PIT liquidity supplement"
    )

    (
        liquidity_complete_panel,
        supplement,
        original_gap_detail,
    ) = (
        merge_liquidity_with_trade_universe_supplement(
            execution_panel,
            liquidity,
            calendar,
        )
    )

    # --------------------------------------------------------
    # Save original mismatch for research audit trail.
    # --------------------------------------------------------

    save_csv_atomic(
        original_gap_detail,
        ORIGINAL_LIQUIDITY_GAP_PATH,
    )

    # --------------------------------------------------------
    # Save supplement itself.
    # --------------------------------------------------------

    save_parquet_atomic(
        supplement,
        LIQUIDITY_SUPPLEMENT_PATH,
    )

    supplement_qa = (
        summarize_liquidity_supplement(
            supplement
        )
    )

    supplement_qa_df = pd.DataFrame(
        [
            {
                "qa_name":
                    key,

                "qa_value":
                    value,
            }

            for key, value
            in supplement_qa.items()
        ]
    )

    save_csv_atomic(
        supplement_qa_df,
        LIQUIDITY_SUPPLEMENT_QA_PATH,
    )

    print()
    print(
        "Liquidity supplement QA:"
    )

    for key, value in (
        supplement_qa.items()
    ):

        print(
            f"  {key}: "
            f"{value}"
        )

    # ========================================================
    # 8. Final participation panel
    # ========================================================

    print()
    print(
        "[8] Finalize stock-level participation panel"
    )

    panel = (
        finalize_participation_panel(
            liquidity_complete_panel
        )
    )

    # ========================================================
    # 9. Diagnostic outputs before formal QA
    # ========================================================

    delist_cash = (
        panel[
            "dedicated_drift_status"
        ]
        ==
        "PASS_DELIST_CASH_CARRY"
    )

    normal_market_trade = (
        (
            panel[
                "capacity_abs_trade_weight"
            ]
            >
            TRADE_WEIGHT_TOL
        )
        &
        ~delist_cash
    )

    # --------------------------------------------------------
    # Final unresolved primary-liquidity cases.
    #
    # Ideally zero.
    # --------------------------------------------------------

    final_liquidity_issue = panel[
        normal_market_trade
        &
        (
            panel[
                "adv20_cny"
            ]
            .isna()
            |
            panel[
                "current_is_open"
            ]
            .isna()
        )
    ].copy()

    save_csv_atomic(
        final_liquidity_issue,
        LIQUIDITY_JOIN_ISSUE_PATH,
    )

    # --------------------------------------------------------
    # Current suspension is NOT a failure.
    # --------------------------------------------------------

    current_suspension = panel[
        normal_market_trade
        &
        panel[
            "current_is_open"
        ]
        .eq(
            0
        )
    ].copy()

    save_csv_atomic(
        current_suspension,
        CURRENT_SUSPENSION_PATH,
    )

    # --------------------------------------------------------
    # ADV60 missing is robustness coverage only.
    # --------------------------------------------------------

    adv60_missing = panel[
        normal_market_trade
        &
        panel[
            "adv60_cny"
        ]
        .isna()
    ].copy()

    save_csv_atomic(
        adv60_missing,
        ADV60_MISSING_PATH,
    )

    # ========================================================
    # 10. Formal QA
    # ========================================================

    print()
    print(
        "[9] Formal Step-3 QA"
    )

    qa = formal_qa(
        panel,
        canonical_qa,
        supplement,
        original_gap_detail,
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
        "Key QA results:"
    )

    print(
        "  Original trade rows requiring supplement: "
        f"{qa['original_trade_rows_requiring_liquidity_supplement']:,}"
    )

    print(
        "  Unique supplement keys: "
        f"{qa['original_unique_liquidity_supplement_key_count']:,}"
    )

    print(
        "  Final unresolved primary-liquidity trades: "
        f"{qa['positive_capacity_trade_unresolved_primary_liquidity_count']:,}"
    )

    print(
        "  ADV60 missing trade rows: "
        f"{qa['positive_capacity_trade_missing_adv60_count']:,}"
    )

    print(
        "  Current suspended trade rows: "
        f"{qa['current_suspended_trade_row_count']:,}"
    )

    print(
        "  Max participation identity error: "
        f"{qa['max_participation20_1bn_identity_error']:.6e}"
    )

    print(
        "  Participation identity PASS: "
        f"{qa['participation20_identity_ok']}"
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
            "Step-3 QA FAILED."
        )

        print()
        print(
            "See:"
        )

        print(
            QA_PATH
        )

        print(
            LIQUIDITY_JOIN_ISSUE_PATH
        )

        print(
            LIQUIDITY_SUPPLEMENT_QA_PATH
        )

        raise RuntimeError(
            "Step-3 formal QA failed. "
            "Do not use participation results."
        )

    # ========================================================
    # 11. Save formal stock-level base panel
    # ========================================================

    print()
    print(
        "[10] Save stock-level participation base panel"
    )

    save_parquet_atomic(
        panel,
        STOCK_BASE_PATH,
    )

    # ========================================================
    # 12. Leg / pair results
    # ========================================================

    print()
    print(
        "[11] Build Q1/Q5 leg and pair participation summaries"
    )

    (
        leg_summary,
        pair_summary,
    ) = (
        build_participation_summaries(
            panel
        )
    )

    save_csv_atomic(
        leg_summary,
        LEG_SUMMARY_PATH,
    )

    save_csv_atomic(
        pair_summary,
        PAIR_SUMMARY_PATH,
    )

    # ========================================================
    # 13. Factor × Window summary
    # ========================================================

    print()
    print(
        "[12] Build Factor × Window aggregate summary"
    )

    factor_window_summary = (
        build_factor_window_summary(
            leg_summary,
            pair_summary,
        )
    )

    save_csv_atomic(
        factor_window_summary,
        FACTOR_WINDOW_SUMMARY_PATH,
    )

    # ========================================================
    # 14. Metadata
    # ========================================================

    print()
    print(
        "[13] Save metadata"
    )

    design = {

        "research_day":
            8,

        "step":
            "Step3_Participation_Rate_Capacity",

        "canonical_trade_source":
            str(
                CANONICAL_TRADE_PATH
            ),

        "step1_liquidity_source":
            str(
                LIQUIDITY_PANEL_PATH
            ),

        "trade_universe_supplement_source":
            str(
                RAW_DAILY_DIR
            ),

        "trade_calendar_source":
            str(
                TRADE_CALENDAR_PATH
            ),

        "value_source_field":
            VALUE_SOURCE_FIELD,

        "value_unit_confirmed":
            VALUE_UNIT_CONFIRMED,

        "value_scale_to_cny":
            VALUE_SCALE_TO_CNY,

        "value_unit_audit":
            {
                "observation_n":
                    VALUE_UNIT_AUDIT_N,

                "sampled_date_n":
                    VALUE_UNIT_AUDIT_DATE_N,

                "start":
                    VALUE_UNIT_AUDIT_START,

                "end":
                    VALUE_UNIT_AUDIT_END,

                "identity":
                    (
                        "turnoverValue / "
                        "(turnoverVol * vwap) = 1"
                    ),
            },

        "liquidity_universe_rule":
            (
                "Primary Step-1 current-universe PIT liquidity "
                "is supplemented only for actual Canonical "
                "Step-2 market-trade stock-date keys not "
                "covered by the Step-1 panel."
            ),

        "supplement_adv_definition":
            (
                "Strict arithmetic mean over exactly the most "
                "recent 20/60 MARKET TRADING DAYS ending at "
                "analysis_date, with suspension turnover value "
                "set to zero and unresolved source records "
                "left missing."
            ),

        "aum_scenarios_cny":
            AUM_SCENARIOS_CNY,

        "participation_caps":
            PARTICIPATION_CAPS,

        "primary_liquidity_basis":
            PRIMARY_LIQUIDITY_BASIS,

        "robustness_liquidity_basis":
            ROBUSTNESS_LIQUIDITY_BASIS,

        "execution_days":
            EXECUTION_DAYS,

        "participation_definition":
            (
                "A * abs(current-market delta weight) "
                "/ effective trailing ADV"
            ),

        "effective_adv_rule":
            (
                "current_is_open == 1 -> trailing PIT ADV; "
                "current_is_open == 0 -> effective ADV = 0; "
                "unknown status remains missing."
            ),

        "delisting_execution_rule":
            (
                "PASS_DELIST_CASH_CARRY wealth is already "
                "cash at the current rebalance and therefore "
                "does not create a current-date stock SELL."
            ),

        "executable_share_definition":
            (
                "sum_i min(required_trade_i, "
                "participation_cap * effective_ADV_i "
                "* execution_days) / total_required_trade"
            ),

        "adv_winsorization":
            False,

        "future_liquidity_used":
            False,

        "future_reopen_information_used":
            False,

        "factor_selection":
            False,

        "window_selection":
            False,

        "factor_sign_flip":
            False,

        "q1_q5_pair_interpretation":
            (
                "Hypothetical gross market-liquidity "
                "diagnostic matching the two-leg convention. "
                "A-share stock-borrow availability and "
                "short-borrow costs are not modeled."
            ),
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

        "supplement_qa":
            supplement_qa,

        "formal_qa":
            qa,

        "outputs": {

            "stock_participation_base_panel":
                str(
                    STOCK_BASE_PATH
                ),

            "participation_leg_summary":
                str(
                    LEG_SUMMARY_PATH
                ),

            "participation_pair_summary":
                str(
                    PAIR_SUMMARY_PATH
                ),

            "participation_factor_window_summary":
                str(
                    FACTOR_WINDOW_SUMMARY_PATH
                ),

            "trade_universe_liquidity_supplement":
                str(
                    LIQUIDITY_SUPPLEMENT_PATH
                ),

            "trade_universe_liquidity_supplement_qa":
                str(
                    LIQUIDITY_SUPPLEMENT_QA_PATH
                ),

            "original_trade_universe_liquidity_gap_detail":
                str(
                    ORIGINAL_LIQUIDITY_GAP_PATH
                ),

            "final_liquidity_join_issue_detail":
                str(
                    LIQUIDITY_JOIN_ISSUE_PATH
                ),

            "current_suspension_detail":
                str(
                    CURRENT_SUSPENSION_PATH
                ),

            "adv60_missing_detail":
                str(
                    ADV60_MISSING_PATH
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
    # 15. Final console output
    # ========================================================

    print()
    print("=" * 80)

    print(
        "DAY 8 STEP 3 COMPLETE"
    )

    print("=" * 80)

    print(
        "Canonical leg groups: "
        f"{qa['canonical_trade_leg_group_count']:,}"
    )

    print(
        "Canonical pair groups: "
        f"{qa['canonical_trade_pair_group_count']:,}"
    )

    print(
        "Current-market trade rows: "
        f"{qa['capacity_trade_row_count']:,}"
    )

    print(
        "Original liquidity-gap trade rows: "
        f"{qa['original_trade_rows_requiring_liquidity_supplement']:,}"
    )

    print(
        "Unique supplement stock-date keys: "
        f"{qa['original_unique_liquidity_supplement_key_count']:,}"
    )

    print(
        "Final unresolved primary-liquidity trades: "
        f"{qa['positive_capacity_trade_unresolved_primary_liquidity_count']:,}"
    )

    print(
        "Current suspended trade rows: "
        f"{qa['current_suspended_trade_row_count']:,}"
    )

    print(
        "ADV60 missing trade rows: "
        f"{qa['positive_capacity_trade_missing_adv60_count']:,}"
    )

    print(
        "Formal QA pass: "
        f"{qa['all_formal_qa_pass']}"
    )

    print()
    print(
        "Primary liquidity basis:"
    )

    print(
        PRIMARY_LIQUIDITY_BASIS
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