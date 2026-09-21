from __future__ import annotations

from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


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
# Day 1 validated return panel
# ------------------------------------------------------------

DAY1_RETURN_PANEL_PATH = (
    OUTPUT_ROOT
    / "M1_day1"
    / "03_stage3_return_validation"
    / "daily_return_panel.parquet"
)


# ------------------------------------------------------------
# Trading calendar
# ------------------------------------------------------------

TRADE_CALENDAR_PATH = Path(
    r"D:\lowfreq\trade_calendar\trade_date.csv"
)


# ------------------------------------------------------------
# Day 8 Step 2 target portfolios
#
# We use this file to determine which exact
#
# previous formation date -> current formation date
#
# intervals are actually required, and which PREVIOUS
# portfolio members require a drift return.
# ------------------------------------------------------------

TARGET_WEIGHT_PATH = (
    OUTPUT_ROOT
    / "M1_day8"
    / "02_stage2_trade_reconstruction"
    / "portfolio_target_weights.parquet"
)


# ------------------------------------------------------------
# Day 8 Step 2B
# ------------------------------------------------------------

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day8"
    / "02b_stage2b_dedicated_drift_return"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DRIFT_RETURN_PATH = (
    OUTPUT_DIR
    / "dedicated_drift_return_panel.parquet"
)

INTERVAL_QA_PATH = (
    OUTPUT_DIR
    / "dedicated_drift_return_interval_qa.csv"
)

ISSUE_DETAIL_PATH = (
    OUTPUT_DIR
    / "dedicated_drift_return_issue_detail.csv"
)

SOURCE_QA_PATH = (
    OUTPUT_DIR
    / "dedicated_drift_return_source_qa.csv"
)

FORMAL_QA_PATH = (
    OUTPUT_DIR
    / "dedicated_drift_return_formal_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day8_step2b_metadata.json"
)


# ============================================================
# 1. Fixed design
# ============================================================

WINDOWS = [
    60,
    120,
    252,
]


# ------------------------------------------------------------
# Exact Day-1 holding-return convention.
# ------------------------------------------------------------

RETURN_COLUMN_CANDIDATES = [

    "return_last_trade_simple",

    "last_trade_simple_return",

    "return_last_trade",
]


IS_OPEN_COLUMN_CANDIDATES = [

    "isOpen",

    "is_open",
]


DATE_COLUMN_CANDIDATES = [

    "trade_date",

    "tradeDate",

    "calendarDate",

    "date",
]


SECURITY_COLUMN_CANDIDATES = [

    "security_id",

    "secID",

    "sec_id",
]


# ============================================================
# 2. Utilities
# ============================================================

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


def save_parquet_atomic(
    df: pd.DataFrame,
    path: Path,
):

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


def normalize_security_id(
    x,
):

    return (
        x.astype("string")
        .str.strip()
    )


def resolve_column(
    columns,
    candidates,
    name,
):

    columns = list(
        columns
    )

    for candidate in candidates:

        if candidate in columns:

            return candidate

    raise RuntimeError(
        f"Cannot resolve column `{name}`.\n"
        f"Available columns:\n"
        f"{columns}"
    )


def normalize_is_open(
    x,
):

    numeric = pd.to_numeric(
        x,
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=x.index,
        dtype=float,
    )

    numeric_valid = numeric.notna()

    # to_numeric preserves boolean input. Explicitly convert valid values
    # to the float output dtype (True -> 1.0, False -> 0.0); pandas 3
    # rejects assigning bool values directly into a float64 Series.
    result.loc[
        numeric_valid
    ] = numeric.loc[
        numeric_valid
    ].astype(float)

    string_x = (
        x.astype("string")
        .str.strip()
        .str.lower()
    )

    true_values = {
        "true",
        "t",
        "yes",
        "y",
        "open",
    }

    false_values = {
        "false",
        "f",
        "no",
        "n",
        "closed",
        "close",
    }

    result.loc[
        string_x.isin(
            true_values
        )
    ] = 1.0

    result.loc[
        string_x.isin(
            false_values
        )
    ] = 0.0

    invalid = (

        result.notna()

        &

        ~result.isin(
            [
                0.0,
                1.0,
            ]
        )
    )

    if invalid.any():

        raise RuntimeError(
            "Invalid isOpen values detected."
        )

    return result


# ============================================================
# 3. Load trading calendar
# ============================================================

def load_trade_calendar():

    if (
        not
        TRADE_CALENDAR_PATH.exists()
    ):

        raise FileNotFoundError(
            f"Missing calendar:\n"
            f"{TRADE_CALENDAR_PATH}"
        )

    raw = pd.read_csv(
        TRADE_CALENDAR_PATH
    )

    date_column = resolve_column(

        raw.columns,

        DATE_COLUMN_CANDIDATES,

        "trade_date",
    )

    calendar = pd.DataFrame(
        {
            "trade_date":
                pd.to_datetime(
                    raw[
                        date_column
                    ],
                    errors="coerce",
                )
        }
    )

    # --------------------------------------------------------
    # If the calendar itself contains isOpen,
    # keep market trading days only.
    # --------------------------------------------------------

    open_column = None

    for candidate in (
        IS_OPEN_COLUMN_CANDIDATES
    ):

        if candidate in raw.columns:

            open_column = candidate

            break

    if open_column is not None:

        market_open = (
            normalize_is_open(
                raw[
                    open_column
                ]
            )
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
# 4. Load Step-2 target portfolios
# ============================================================

def load_target_weights():

    if (
        not
        TARGET_WEIGHT_PATH.exists()
    ):

        raise FileNotFoundError(
            f"Missing target weights:\n"
            f"{TARGET_WEIGHT_PATH}"
        )

    targets = pd.read_parquet(
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
        if c not in targets.columns
    ]

    if missing:

        raise RuntimeError(
            "Target weight panel missing columns:\n"
            f"{missing}"
        )

    targets[
        "analysis_date"
    ] = pd.to_datetime(
        targets[
            "analysis_date"
        ],
        errors="raise",
    )

    targets[
        "security_id"
    ] = normalize_security_id(
        targets[
            "security_id"
        ]
    )

    targets[
        "window"
    ] = pd.to_numeric(
        targets[
            "window"
        ],
        errors="raise",
    ).astype(int)

    targets[
        "target_weight"
    ] = pd.to_numeric(
        targets[
            "target_weight"
        ],
        errors="raise",
    )

    targets = targets[
        targets[
            "window"
        ]
        .isin(
            WINDOWS
        )
    ].copy()

    key = [

        "analysis_date",

        "window",

        "factor_name",

        "method",

        "portfolio_leg",

        "security_id",
    ]

    if targets[
        key
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate target-weight keys."
        )

    return targets


# ============================================================
# 5. Determine exactly which drift returns are required
#
# Important:
#
# We do NOT blindly create every stock × every month.
#
# A drift return is required only if a stock belonged to
# the PREVIOUS target portfolio.
#
# Return itself is independent of:
#
# factor / window / method / leg
#
# so requirements are deduplicated across portfolio contexts.
# ============================================================

def build_drift_requirements(
    targets,
    calendar,
):

    pair_to_securities = defaultdict(
        set
    )

    portfolio_group_columns = [

        "window",

        "factor_name",

        "method",

        "portfolio_leg",
    ]

    for _, group in targets.groupby(
        portfolio_group_columns,
        sort=False,
    ):

        by_date = {

            pd.Timestamp(
                date
            ):
                set(
                    temp[
                        "security_id"
                    ]
                    .astype(str)
                    .tolist()
                )

            for date, temp
            in group.groupby(
                "analysis_date",
                sort=True,
            )
        }

        dates = sorted(
            by_date.keys()
        )

        for previous_date, current_date in zip(
            dates[:-1],
            dates[1:],
        ):

            pair_to_securities[
                (
                    previous_date,
                    current_date,
                )
            ].update(
                by_date[
                    previous_date
                ]
            )

    if not pair_to_securities:

        raise RuntimeError(
            "No drift-return intervals constructed."
        )

    rows = []

    market_map = (
        calendar.set_index(
            "trade_date"
        )[
            "market_index"
        ]
    )

    for (
        previous_date,
        current_date,
    ), security_set in (
        pair_to_securities.items()
    ):

        previous_market_index = (
            market_map.get(
                previous_date,
                np.nan,
            )
        )

        current_market_index = (
            market_map.get(
                current_date,
                np.nan,
            )
        )

        if (
            pd.isna(
                previous_market_index
            )
            or
            pd.isna(
                current_market_index
            )
        ):

            raise RuntimeError(
                "Formation date not found "
                "in trading calendar:\n"
                f"{previous_date} -> {current_date}"
            )

        previous_market_index = int(
            previous_market_index
        )

        current_market_index = int(
            current_market_index
        )

        interval_days = (

            current_market_index

            -

            previous_market_index
        )

        if interval_days <= 0:

            raise RuntimeError(
                "Invalid formation-date ordering:\n"
                f"{previous_date} -> {current_date}"
            )

        for security_id in (
            security_set
        ):

            rows.append(
                {
                    "previous_analysis_date":
                        previous_date,

                    "analysis_date":
                        current_date,

                    "previous_market_index":
                        previous_market_index,

                    "current_market_index":
                        current_market_index,

                    "interval_market_days":
                        interval_days,

                    "security_id":
                        security_id,
                }
            )

    requirements = pd.DataFrame(
        rows
    )

    requirements = (

        requirements
        .drop_duplicates(
            subset=[
                "previous_analysis_date",
                "analysis_date",
                "security_id",
            ]
        )
        .sort_values(
            [
                "security_id",
                "previous_analysis_date",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return requirements


# ============================================================
# 6. Inspect Day-1 return-panel schema
# ============================================================

def get_return_panel_columns():

    if (
        not
        DAY1_RETURN_PANEL_PATH.exists()
    ):

        raise FileNotFoundError(
            f"Missing Day-1 return panel:\n"
            f"{DAY1_RETURN_PANEL_PATH}"
        )

    dataset = ds.dataset(
        str(
            DAY1_RETURN_PANEL_PATH
        ),
        format="parquet",
    )

    return dataset.schema.names


# ============================================================
# 7. Load Day-1 holding-return ingredients
# ============================================================

def load_daily_holding_return_source(
    requirements,
):

    columns = (
        get_return_panel_columns()
    )

    date_column = resolve_column(

        columns,

        DATE_COLUMN_CANDIDATES,

        "trade_date",
    )

    security_column = resolve_column(

        columns,

        SECURITY_COLUMN_CANDIDATES,

        "security_id",
    )

    is_open_column = resolve_column(

        columns,

        IS_OPEN_COLUMN_CANDIDATES,

        "is_open",
    )

    return_column = resolve_column(

        columns,

        RETURN_COLUMN_CANDIDATES,

        "return_last_trade_simple",
    )

    source_columns = [

        date_column,

        security_column,

        is_open_column,

        return_column,
    ]

    daily = pd.read_parquet(

        DAY1_RETURN_PANEL_PATH,

        columns=source_columns,
    )

    daily = daily.rename(
        columns={
            date_column:
                "trade_date",

            security_column:
                "security_id",

            is_open_column:
                "is_open",

            return_column:
                "return_last_trade_simple",
        }
    )

    daily[
        "trade_date"
    ] = pd.to_datetime(
        daily[
            "trade_date"
        ],
        errors="coerce",
    )

    daily[
        "security_id"
    ] = normalize_security_id(
        daily[
            "security_id"
        ]
    )

    daily[
        "is_open"
    ] = normalize_is_open(
        daily[
            "is_open"
        ]
    )

    daily[
        "return_last_trade_simple"
    ] = pd.to_numeric(
        daily[
            "return_last_trade_simple"
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Restrict to relevant dates/stocks.
    # --------------------------------------------------------

    min_date = (
        requirements[
            "previous_analysis_date"
        ]
        .min()
    )

    max_date = (
        requirements[
            "analysis_date"
        ]
        .max()
    )

    required_securities = set(
        requirements[
            "security_id"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    daily = daily[
        (
            daily[
                "trade_date"
            ]
            >
            min_date
        )
        &
        (
            daily[
                "trade_date"
            ]
            <=
            max_date
        )
        &
        (
            daily[
                "security_id"
            ]
            .astype(str)
            .isin(
                required_securities
            )
        )
    ].copy()

    daily = daily[
        daily[
            "trade_date"
        ]
        .notna()
        &
        daily[
            "security_id"
        ]
        .notna()
    ].copy()

    duplicate = daily[
        [
            "trade_date",
            "security_id",
        ]
    ].duplicated(
        keep=False
    )

    duplicate_count = int(
        duplicate.sum()
    )

    if duplicate_count > 0:

        raise RuntimeError(
            "Duplicate stock-date rows in "
            "Day-1 return panel.\n"
            f"Count={duplicate_count}"
        )

    # --------------------------------------------------------
    # Source sanity:
    #
    # simple returns cannot be below -100%.
    # --------------------------------------------------------

    invalid_return = (

        daily[
            "return_last_trade_simple"
        ]
        <
        -1.0
        -
        1e-12
    )

    invalid_return_count = int(
        invalid_return.sum()
    )

    if invalid_return_count > 0:

        raise RuntimeError(
            "return_last_trade_simple < -1 detected.\n"
            f"Count={invalid_return_count}"
        )

    # --------------------------------------------------------
    # Build portfolio-accounting daily return.
    #
    # closed:
    #     0
    #
    # open + valid last-trade return:
    #     return_last_trade_simple
    #
    # otherwise:
    #     NA
    # --------------------------------------------------------

    daily[
        "holding_return"
    ] = np.nan

    closed = (

        daily[
            "is_open"
        ]
        ==
        0
    )

    open_valid = (

        daily[
            "is_open"
        ]
        ==
        1

    ) & (

        daily[
            "return_last_trade_simple"
        ]
        .notna()

    ) & (

        np.isfinite(
            daily[
                "return_last_trade_simple"
            ]
        )
    )

    daily.loc[
        closed,
        "holding_return",
    ] = 0.0

    daily.loc[
        open_valid,
        "holding_return",
    ] = daily.loc[
        open_valid,
        "return_last_trade_simple",
    ]

    source_qa = {

        "daily_row_count":
            int(
                len(
                    daily
                )
            ),

        "security_count":
            int(
                daily[
                    "security_id"
                ]
                .nunique()
            ),

        "date_min":
            daily[
                "trade_date"
            ]
            .min(),

        "date_max":
            daily[
                "trade_date"
            ]
            .max(),

        "duplicate_stock_date_count":
            duplicate_count,

        "invalid_return_below_minus_one_count":
            invalid_return_count,

        "open_missing_last_trade_return_count":
            int(
                (
                    (
                        daily[
                            "is_open"
                        ]
                        ==
                        1
                    )
                    &
                    (
                        daily[
                            "return_last_trade_simple"
                        ]
                        .isna()
                    )
                )
                .sum()
            ),

        "closed_day_count":
            int(
                closed.sum()
            ),

        "resolved_holding_return_count":
            int(
                daily[
                    "holding_return"
                ]
                .notna()
                .sum()
            ),
    }

    return (
        daily,
        source_qa,
    )


# ============================================================
# 8. Attach market index
# ============================================================

def attach_market_index(
    daily,
    calendar,
):

    daily = daily.merge(

        calendar[
            [
                "trade_date",
                "market_index",
            ]
        ],

        on="trade_date",

        how="left",

        validate="many_to_one",
    )

    if (
        daily[
            "market_index"
        ]
        .isna()
        .any()
    ):

        bad_dates = (

            daily.loc[
                daily[
                    "market_index"
                ]
                .isna(),
                "trade_date",
            ]
            .drop_duplicates()
            .head(
                20
            )
            .tolist()
        )

        raise RuntimeError(
            "Return panel contains dates not "
            "found in trading calendar:\n"
            f"{bad_dates}"
        )

    daily[
        "market_index"
    ] = daily[
        "market_index"
    ].astype(
        np.int64
    )

    return (
        daily.sort_values(
            [
                "security_id",
                "market_index",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 9. Construct dedicated drift returns
#
# Uses prefix statistics within each security.
#
# This avoids scanning the full 11m-row table once per interval.
# ============================================================

def construct_drift_returns(
    requirements,
    daily,
):

    output_frames = []

    daily_groups = daily.groupby(
        "security_id",
        sort=False,
    )

    available_securities = set(
        daily_groups.groups.keys()
    )

    requirement_groups = (
        requirements.groupby(
            "security_id",
            sort=False,
        )
    )

    total_securities = (
        requirement_groups.ngroups
    )

    for security_no, (
        security_id,
        req,
    ) in enumerate(
        requirement_groups,
        start=1,
    ):

        if (
            security_no == 1
            or
            security_no % 250 == 0
            or
            security_no == total_securities
        ):

            print(
                f"Drift-return construction: "
                f"{security_no}/"
                f"{total_securities}"
            )

        req = (
            req.sort_values(
                [
                    "previous_market_index",
                    "current_market_index",
                ]
            )
            .copy()
        )

        # ----------------------------------------------------
        # No source history at all.
        # ----------------------------------------------------

        if (
            security_id
            not in
            available_securities
        ):

            temp = req.copy()

            temp[
                "observed_market_days"
            ] = 0

            temp[
                "resolved_holding_days"
            ] = 0

            temp[
                "open_days"
            ] = 0

            temp[
                "suspended_days"
            ] = 0

            temp[
                "missing_record_days"
            ] = temp[
                "interval_market_days"
            ]

            temp[
                "unresolved_return_days"
            ] = 0

            temp[
                "zero_gross_return_days"
            ] = 0

            temp[
                "drift_total_return"
            ] = np.nan

            temp[
                "status"
            ] = "MISSING_SOURCE_HISTORY"

            output_frames.append(
                temp
            )

            continue

        history = (
            daily_groups
            .get_group(
                security_id
            )
            .sort_values(
                "market_index"
            )
        )

        market_index = (
            history[
                "market_index"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        holding_return = (
            history[
                "holding_return"
            ]
            .to_numpy(
                dtype=float
            )
        )

        is_open = (
            history[
                "is_open"
            ]
            .to_numpy(
                dtype=float
            )
        )

        valid_return = np.isfinite(
            holding_return
        )

        gross = (
            1.0
            +
            holding_return
        )

        invalid_negative_gross = (

            valid_return

            &

            (
                gross
                <
                -1e-12
            )
        )

        if invalid_negative_gross.any():

            raise RuntimeError(
                "Negative gross-return factor "
                f"for security {security_id}."
            )

        # ----------------------------------------------------
        # Return = -1 exactly implies wealth goes to zero.
        #
        # We handle that separately because log(0)
        # cannot be accumulated normally.
        # ----------------------------------------------------

        zero_gross = (

            valid_return

            &

            np.isclose(
                gross,
                0.0,
                atol=1e-15,
                rtol=0.0,
            )
        )

        positive_gross = (

            valid_return

            &

            (
                gross
                >
                0
            )
        )

        log_gross = np.zeros(
            len(
                history
            ),
            dtype=float,
        )

        log_gross[
            positive_gross
        ] = np.log(
            gross[
                positive_gross
            ]
        )

        # ----------------------------------------------------
        # Prefix arrays
        # ----------------------------------------------------

        cumulative_valid = np.concatenate(
            [
                np.array(
                    [
                        0
                    ],
                    dtype=np.int64,
                ),

                np.cumsum(
                    valid_return.astype(
                        np.int64
                    )
                ),
            ]
        )

        cumulative_log_gross = np.concatenate(
            [
                np.array(
                    [
                        0.0
                    ]
                ),

                np.cumsum(
                    log_gross
                ),
            ]
        )

        cumulative_zero_gross = np.concatenate(
            [
                np.array(
                    [
                        0
                    ],
                    dtype=np.int64,
                ),

                np.cumsum(
                    zero_gross.astype(
                        np.int64
                    )
                ),
            ]
        )

        cumulative_open = np.concatenate(
            [
                np.array(
                    [
                        0
                    ],
                    dtype=np.int64,
                ),

                np.cumsum(
                    (
                        is_open
                        ==
                        1
                    )
                    .astype(
                        np.int64
                    )
                ),
            ]
        )

        cumulative_suspended = np.concatenate(
            [
                np.array(
                    [
                        0
                    ],
                    dtype=np.int64,
                ),

                np.cumsum(
                    (
                        is_open
                        ==
                        0
                    )
                    .astype(
                        np.int64
                    )
                ),
            ]
        )

        previous_index = (
            req[
                "previous_market_index"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        current_index = (
            req[
                "current_market_index"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        # ----------------------------------------------------
        # We need:
        #
        # previous_index < market_index <= current_index
        # ----------------------------------------------------

        left = np.searchsorted(

            market_index,

            previous_index,

            side="right",
        )

        right = np.searchsorted(

            market_index,

            current_index,

            side="right",
        )

        observed_days = (
            right
            -
            left
        )

        resolved_days = (

            cumulative_valid[
                right
            ]

            -

            cumulative_valid[
                left
            ]
        )

        open_days = (

            cumulative_open[
                right
            ]

            -

            cumulative_open[
                left
            ]
        )

        suspended_days = (

            cumulative_suspended[
                right
            ]

            -

            cumulative_suspended[
                left
            ]
        )

        zero_gross_days = (

            cumulative_zero_gross[
                right
            ]

            -

            cumulative_zero_gross[
                left
            ]
        )

        expected_days = (
            req[
                "interval_market_days"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        missing_record_days = (

            expected_days

            -

            observed_days
        )

        unresolved_return_days = (

            observed_days

            -

            resolved_days
        )

        full_coverage = (

            (
                observed_days
                ==
                expected_days
            )

            &

            (
                resolved_days
                ==
                expected_days
            )
        )

        interval_log_gross = (

            cumulative_log_gross[
                right
            ]

            -

            cumulative_log_gross[
                left
            ]
        )

        drift_return = np.full(
            len(
                req
            ),
            np.nan,
            dtype=float,
        )

        full_rows = np.where(
            full_coverage
        )[0]

        if len(
            full_rows
        ) > 0:

            no_zero = (

                zero_gross_days[
                    full_rows
                ]
                ==
                0
            )

            nonzero_rows = (
                full_rows[
                    no_zero
                ]
            )

            zero_rows = (
                full_rows[
                    ~no_zero
                ]
            )

            drift_return[
                nonzero_rows
            ] = (

                np.exp(
                    interval_log_gross[
                        nonzero_rows
                    ]
                )

                -

                1.0
            )

            # Complete wealth loss.
            drift_return[
                zero_rows
            ] = -1.0

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        status = np.full(
            len(
                req
            ),
            "PASS",
            dtype=object,
        )

        missing_only = (

            (
                missing_record_days
                >
                0
            )

            &

            (
                unresolved_return_days
                ==
                0
            )
        )

        unresolved_only = (

            (
                missing_record_days
                ==
                0
            )

            &

            (
                unresolved_return_days
                >
                0
            )
        )

        both = (

            (
                missing_record_days
                >
                0
            )

            &

            (
                unresolved_return_days
                >
                0
            )
        )

        status[
            missing_only
        ] = "MISSING_SOURCE_RECORD"

        status[
            unresolved_only
        ] = "UNRESOLVED_HOLDING_RETURN"

        status[
            both
        ] = "MISSING_AND_UNRESOLVED"

        temp = req.copy()

        temp[
            "observed_market_days"
        ] = observed_days

        temp[
            "resolved_holding_days"
        ] = resolved_days

        temp[
            "open_days"
        ] = open_days

        temp[
            "suspended_days"
        ] = suspended_days

        temp[
            "missing_record_days"
        ] = missing_record_days

        temp[
            "unresolved_return_days"
        ] = unresolved_return_days

        temp[
            "zero_gross_return_days"
        ] = zero_gross_days

        temp[
            "drift_total_return"
        ] = drift_return

        temp[
            "status"
        ] = status

        output_frames.append(
            temp
        )

    result = pd.concat(
        output_frames,
        ignore_index=True,
    )

    result[
        "short_interval_lt_10_days"
    ] = (

        result[
            "interval_market_days"
        ]
        <
        10
    )

    return (
        result.sort_values(
            [
                "previous_analysis_date",
                "analysis_date",
                "security_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 10. Interval QA
# ============================================================

def build_interval_qa(
    drift,
):

    rows = []

    for (
        previous_date,
        current_date,
    ), group in drift.groupby(
        [
            "previous_analysis_date",
            "analysis_date",
        ],
        sort=True,
    ):

        status_counts = (

            group[
                "status"
            ]
            .value_counts()
            .to_dict()
        )

        pass_n = int(
            status_counts.get(
                "PASS",
                0,
            )
        )

        n = int(
            len(
                group
            )
        )

        rows.append(
            {
                "previous_analysis_date":
                    previous_date,

                "analysis_date":
                    current_date,

                "interval_market_days":
                    int(
                        group[
                            "interval_market_days"
                        ]
                        .iloc[0]
                    ),

                "required_security_n":
                    n,

                "pass_n":
                    pass_n,

                "pass_share":
                    (
                        pass_n
                        /
                        n
                        if n > 0
                        else np.nan
                    ),

                "missing_source_record_n":
                    int(
                        status_counts.get(
                            "MISSING_SOURCE_RECORD",
                            0,
                        )
                    ),

                "unresolved_holding_return_n":
                    int(
                        status_counts.get(
                            "UNRESOLVED_HOLDING_RETURN",
                            0,
                        )
                    ),

                "missing_and_unresolved_n":
                    int(
                        status_counts.get(
                            "MISSING_AND_UNRESOLVED",
                            0,
                        )
                    ),

                "short_interval_lt_10_days":
                    bool(
                        group[
                            "short_interval_lt_10_days"
                        ]
                        .iloc[0]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 11. Formal QA
# ============================================================

def formal_qa(
    drift,
    source_qa,
):

    qa = {}

    key = [

        "previous_analysis_date",

        "analysis_date",

        "security_id",
    ]

    qa[
        "drift_row_count"
    ] = int(
        len(
            drift
        )
    )

    qa[
        "drift_keys_unique"
    ] = bool(
        not drift[
            key
        ]
        .duplicated()
        .any()
    )

    qa[
        "interval_pair_count"
    ] = int(
        drift[
            [
                "previous_analysis_date",
                "analysis_date",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    qa[
        "security_count"
    ] = int(
        drift[
            "security_id"
        ]
        .nunique()
    )

    qa[
        "short_interval_row_count"
    ] = int(
        drift[
            "short_interval_lt_10_days"
        ]
        .sum()
    )

    qa[
        "pass_count"
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
        "nonpass_count"
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
        "pass_share"
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

    pass_rows = (

        drift[
            "status"
        ]
        ==
        "PASS"
    )

    nonpass_rows = (
        ~pass_rows
    )

    qa[
        "pass_rows_have_return"
    ] = bool(
        drift.loc[
            pass_rows,
            "drift_total_return",
        ]
        .notna()
        .all()
    )

    qa[
        "nonpass_rows_have_no_return"
    ] = bool(
        drift.loc[
            nonpass_rows,
            "drift_total_return",
        ]
        .isna()
        .all()
    )

    qa[
        "drift_return_not_below_minus_one"
    ] = bool(
        (
            drift[
                "drift_total_return"
            ]
            .dropna()
            >=
            -1.0
            -
            1e-12
        )
        .all()
    )

    qa[
        "pass_rows_have_full_market_day_coverage"
    ] = bool(

        (
            drift.loc[
                pass_rows,
                "observed_market_days",
            ]

            ==

            drift.loc[
                pass_rows,
                "interval_market_days",
            ]
        )
        .all()

        and

        (
            drift.loc[
                pass_rows,
                "resolved_holding_days",
            ]

            ==

            drift.loc[
                pass_rows,
                "interval_market_days",
            ]
        )
        .all()
    )

    qa[
        "source_duplicate_stock_date_count"
    ] = int(
        source_qa[
            "duplicate_stock_date_count"
        ]
    )

    qa[
        "source_invalid_return_below_minus_one_count"
    ] = int(
        source_qa[
            "invalid_return_below_minus_one_count"
        ]
    )

    qa[
        "minimum_valid_interval_days_imposed"
    ] = False

    qa[
        "future_information_used"
    ] = False

    qa[
        "missing_return_zero_filled"
    ] = False

    qa[
        "suspension_return_zero_by_design"
    ] = True

    qa[
        "interval_left_endpoint_included"
    ] = False

    qa[
        "interval_right_endpoint_included"
    ] = True

    # --------------------------------------------------------
    # This flag is informative rather than required.
    # Do not silently invent an arbitrary coverage threshold.
    # --------------------------------------------------------

    qa[
        "all_required_drift_returns_resolved"
    ] = bool(
        qa[
            "nonpass_count"
        ]
        ==
        0
    )

    required = [

        "drift_keys_unique",

        "pass_rows_have_return",

        "nonpass_rows_have_no_return",

        "drift_return_not_below_minus_one",

        "pass_rows_have_full_market_day_coverage",
    ]

    qa[
        "all_formal_qa_pass"
    ] = bool(
        all(
            qa[
                key
            ]
            for key in required
        )

        and

        qa[
            "source_duplicate_stock_date_count"
        ]
        ==
        0

        and

        qa[
            "source_invalid_return_below_minus_one_count"
        ]
        ==
        0
    )

    return qa


# ============================================================
# 12. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 8 - Step 2B"
    )

    print(
        "Dedicated Drift Return Construction"
    )

    print("=" * 80)

    # ========================================================
    # Calendar
    # ========================================================

    print()
    print(
        "[1] Load trading calendar"
    )

    calendar = load_trade_calendar()

    # ========================================================
    # Current portfolio targets
    # ========================================================

    print()
    print(
        "[2] Load Step-2 target portfolios"
    )

    targets = load_target_weights()

    print(
        f"Target rows: "
        f"{len(targets):,}"
    )

    # ========================================================
    # Required drift intervals
    # ========================================================

    print()
    print(
        "[3] Build exact drift-return requirements"
    )

    requirements = (
        build_drift_requirements(
            targets,
            calendar,
        )
    )

    print(
        f"Unique drift-return requirements: "
        f"{len(requirements):,}"
    )

    print(
        f"Unique interval pairs: "
        f"{requirements[['previous_analysis_date', 'analysis_date']].drop_duplicates().shape[0]:,}"
    )

    # ========================================================
    # Day-1 holding-return source
    # ========================================================

    print()
    print(
        "[4] Load Day-1 holding-return source"
    )

    (
        daily,
        source_qa,
    ) = load_daily_holding_return_source(
        requirements
    )

    daily = attach_market_index(
        daily,
        calendar,
    )

    source_qa_df = pd.DataFrame(
        [
            {
                "qa_name":
                    key,

                "qa_value":
                    value,
            }

            for key, value
            in source_qa.items()
        ]
    )

    save_csv_atomic(
        source_qa_df,
        SOURCE_QA_PATH,
    )

    # ========================================================
    # Dedicated drift returns
    # ========================================================

    print()
    print(
        "[5] Construct dedicated interval drift returns"
    )

    drift = construct_drift_returns(

        requirements,

        daily,
    )

    save_parquet_atomic(
        drift,
        DRIFT_RETURN_PATH,
    )

    # ========================================================
    # Interval QA
    # ========================================================

    print()
    print(
        "[6] Build interval-level QA"
    )

    interval_qa = (
        build_interval_qa(
            drift
        )
    )

    save_csv_atomic(
        interval_qa,
        INTERVAL_QA_PATH,
    )

    # ========================================================
    # Problem cases
    # ========================================================

    issues = drift[
        drift[
            "status"
        ]
        !=
        "PASS"
    ].copy()

    save_csv_atomic(
        issues,
        ISSUE_DETAIL_PATH,
    )

    # ========================================================
    # Formal QA
    # ========================================================

    print()
    print(
        "[7] Formal QA"
    )

    qa = formal_qa(

        drift,

        source_qa,
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
        FORMAL_QA_PATH,
    )

    if (
        not
        qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            "Day-8 Step-2B formal QA failed.\n"
            f"{qa}"
        )

    # ========================================================
    # Metadata
    # ========================================================

    design_payload = {

        "research_day":
            8,

        "step":
            "Step2B_Dedicated_Drift_Return_Construction",

        "source":
            str(
                DAY1_RETURN_PANEL_PATH
            ),

        "target_weight_source":
            str(
                TARGET_WEIGHT_PATH
            ),

        "daily_holding_return_definition":
            (
                "0 when isOpen==0; "
                "return_last_trade_simple when isOpen==1 "
                "and valid; otherwise NA."
            ),

        "interval_definition":
            "(previous_analysis_date, analysis_date]",

        "drift_total_return_definition":
            (
                "product(1 + holding_return) - 1 "
                "over all market trading days in the interval."
            ),

        "minimum_valid_interval_days":
            None,

        "requires_complete_market_day_path":
            True,

        "missing_source_record_zero_filled":
            False,

        "unresolved_return_zero_filled":
            False,

        "factor_selection":
            False,

        "window_selection":
            False,

        "future_outcome_used":
            False,
    }

    metadata = {

        **design_payload,

        "day8_step2b_design_hash":
            canonical_hash(
                design_payload
            ),

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "formal_qa":
            qa,

        "source_qa":
            source_qa,

        "outputs":
            {

                "dedicated_drift_return_panel":
                    str(
                        DRIFT_RETURN_PATH
                    ),

                "interval_qa":
                    str(
                        INTERVAL_QA_PATH
                    ),

                "issue_detail":
                    str(
                        ISSUE_DETAIL_PATH
                    ),

                "source_qa":
                    str(
                        SOURCE_QA_PATH
                    ),

                "formal_qa":
                    str(
                        FORMAL_QA_PATH
                    ),
            },

        "important_notes":
            [

                (
                    "This return is a portfolio-accounting "
                    "return, not a statistical prediction label."
                ),

                (
                    "No minimum 10-day rule is imposed."
                ),

                (
                    "A 3-day interval is valid if all three "
                    "market-day holding returns are resolved."
                ),

                (
                    "Suspensions contribute zero daily return; "
                    "the first valid reopening-day "
                    "return_last_trade_simple captures the "
                    "full last-trade price gap."
                ),

                (
                    "Missing or unresolved holding returns "
                    "are never replaced by zero."
                ),

                (
                    "The left formation date is excluded and "
                    "the current formation date is included."
                ),
            ],
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
        "Day-8 Step-2B Summary"
    )

    print("=" * 80)

    print(
        f"Drift-return rows: "
        f"{qa['drift_row_count']:,}"
    )

    print(
        f"Interval pairs: "
        f"{qa['interval_pair_count']:,}"
    )

    print(
        f"PASS: "
        f"{qa['pass_count']:,}"
    )

    print(
        f"Non-PASS: "
        f"{qa['nonpass_count']:,}"
    )

    print(
        f"PASS share: "
        f"{qa['pass_share']:.8%}"
    )

    print(
        f"Short (<10 day) rows: "
        f"{qa['short_interval_row_count']:,}"
    )

    print(
        f"All required returns resolved: "
        f"{qa['all_required_drift_returns_resolved']}"
    )

    print(
        f"Formal QA pass: "
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
        "Day 8 Step 2B Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()
