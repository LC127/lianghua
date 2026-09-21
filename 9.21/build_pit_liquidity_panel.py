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
# Raw daily market data
#
# If your actual daily_temp3 path differs,
# ONLY change this path.
# ------------------------------------------------------------

DAILY_SOURCE_PATH = Path(
    r"D:\lowfreq\daily_temp3"
)


# ------------------------------------------------------------
# Trading calendar
# ------------------------------------------------------------

TRADE_CALENDAR_PATH = Path(
    r"D:\lowfreq\trade_calendar\trade_date.csv"
)

# ------------------------------------------------------------
# Day 5 factor panel
#
# Used ONLY to define:
#
# analysis_date × security_id current universe.
#
# No future outcome is used.
# ------------------------------------------------------------

FACTOR_PANEL_PATH = (
    OUTPUT_ROOT
    / "M1_day5"
    / "05_step5_community_features"
    / "community_bridge_feature_panel.parquet"
)


# ------------------------------------------------------------
# Day 8 -> M1_day7
# ------------------------------------------------------------

DAY8_ROOT = (
    OUTPUT_ROOT
    / "M1_day8"
)

OUTPUT_DIR = (
    DAY8_ROOT
    / "01_stage1_liquidity_panel"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


LIQUIDITY_PANEL_PATH = (
    OUTPUT_DIR
    / "stock_liquidity_panel.parquet"
)

COVERAGE_QA_PATH = (
    OUTPUT_DIR
    / "liquidity_coverage_qa.csv"
)

DISTRIBUTION_PATH = (
    OUTPUT_DIR
    / "liquidity_distribution_summary.csv"
)

SOURCE_QA_PATH = (
    OUTPUT_DIR
    / "liquidity_source_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day8_step1_metadata.json"
)


# ============================================================
# 1. Frozen Step-1 design
# ============================================================

LOOKBACK_WINDOWS = [
    20,
    60,
]


# ------------------------------------------------------------
# IMPORTANT:
#
# Leave as None until the vendor unit of `value`
# has been explicitly verified.
#
# Examples:
#
# value already in RMB yuan:
#     TRADE_VALUE_SCALE_TO_CNY = 1.0
#
# value in thousand RMB:
#     TRADE_VALUE_SCALE_TO_CNY = 1000.0
# ------------------------------------------------------------

TRADE_VALUE_SCALE_TO_CNY = None


# ------------------------------------------------------------
# Main ADV requires complete market-day coverage.
#
# No relaxed ADV is used as the formal capacity measure.
# ------------------------------------------------------------

REQUIRE_FULL_WINDOW_COVERAGE = True


# ------------------------------------------------------------
# Only for descriptive QA.
# ------------------------------------------------------------

RELAXED_COVERAGE_REFERENCE = 0.80


# ============================================================
# 2. Source-column aliases
# ============================================================

COLUMN_ALIASES = {

    "trade_date": [
        "trade_date",
        "tradeDate",
        "date",
        "calendarDate",
    ],

    "security_id": [
        "security_id",
        "secID",
        "sec_id",
    ],

    "value": [
        "value",
        "turnoverValue",
        "turnover_value",
        "amount",
    ],

    "is_open": [
        "isOpen",
        "is_open",
    ],
}


CALENDAR_DATE_ALIASES = [
    "trade_date",
    "tradeDate",
    "calendarDate",
    "date",
]


CALENDAR_OPEN_ALIASES = [
    "isOpen",
    "is_open",
]


# ============================================================
# 3. Utilities
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


def resolve_column(
    columns,
    aliases,
    canonical_name,
):

    columns = list(
        columns
    )

    for candidate in aliases:

        if candidate in columns:

            return candidate

    raise RuntimeError(
        f"Cannot resolve column "
        f"`{canonical_name}`.\n"
        f"Available columns:\n"
        f"{columns}"
    )


def normalize_security_id(
    x,
):

    return (
        x.astype("string")
        .str.strip()
    )


def normalize_is_open(
    x,
):

    # --------------------------------------------------------
    # Numeric cases
    # --------------------------------------------------------

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

    result.loc[
        numeric_valid
    ] = numeric.loc[
        numeric_valid
    ]

    # --------------------------------------------------------
    # String cases
    # --------------------------------------------------------

    string_value = (
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
        string_value.isin(
            true_values
        )
    ] = 1.0

    result.loc[
        string_value.isin(
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

        examples = (
            x.loc[
                invalid
            ]
            .head(
                10
            )
            .tolist()
        )

        raise RuntimeError(
            "isOpen contains values outside "
            "{0,1} after normalization.\n"
            f"Examples: {examples}"
        )

    return result


# ============================================================
# 4. Trading calendar
# ============================================================

def load_trade_calendar():

    if not TRADE_CALENDAR_PATH.exists():

        raise FileNotFoundError(
            f"Missing trade calendar:\n"
            f"{TRADE_CALENDAR_PATH}"
        )

    raw = pd.read_csv(
        TRADE_CALENDAR_PATH
    )

    date_column = resolve_column(

        raw.columns,

        CALENDAR_DATE_ALIASES,

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
    # If calendar file contains open/closed calendar dates,
    # retain trading days only.
    # --------------------------------------------------------

    open_column = None

    for candidate in (
        CALENDAR_OPEN_ALIASES
    ):

        if candidate in raw.columns:

            open_column = candidate

            break

    if open_column is not None:

        open_flag = normalize_is_open(
            raw[
                open_column
            ]
        )

        calendar = calendar[
            open_flag
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

    if (
        calendar[
            "trade_date"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate trading dates "
            "in calendar."
        )

    return calendar


# ============================================================
# 5. Current factor universe
# ============================================================

def load_current_universe(
    calendar,
):

    if not FACTOR_PANEL_PATH.exists():

        raise FileNotFoundError(
            f"Missing factor panel:\n"
            f"{FACTOR_PANEL_PATH}"
        )

    universe = pd.read_parquet(

        FACTOR_PANEL_PATH,

        columns=[
            "analysis_date",
            "security_id",
        ],
    )

    universe[
        "analysis_date"
    ] = pd.to_datetime(
        universe[
            "analysis_date"
        ],
        errors="raise",
    )

    universe[
        "security_id"
    ] = normalize_security_id(
        universe[
            "security_id"
        ]
    )

    # --------------------------------------------------------
    # Factor panel contains the same stock repeatedly across
    # W60/W120/W252.
    #
    # Liquidity is independent of network window,
    # therefore keep one row per date-stock.
    # --------------------------------------------------------

    universe = (
        universe[
            [
                "analysis_date",
                "security_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "analysis_date",
                "security_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if universe[
        [
            "analysis_date",
            "security_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate current-universe keys."
        )

    market_map = (
        calendar.set_index(
            "trade_date"
        )[
            "market_index"
        ]
    )

    universe[
        "market_index"
    ] = universe[
        "analysis_date"
    ].map(
        market_map
    )

    if universe[
        "market_index"
    ].isna().any():

        missing_dates = (
            universe.loc[
                universe[
                    "market_index"
                ]
                .isna(),
                "analysis_date",
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        raise RuntimeError(
            "Some analysis dates are not "
            "trading dates:\n"
            f"{missing_dates[:20]}"
        )

    universe[
        "market_index"
    ] = universe[
        "market_index"
    ].astype(
        np.int64
    )

    return universe


# ============================================================
# 6. Discover daily-market files
# ============================================================

def discover_daily_files():

    if not DAILY_SOURCE_PATH.exists():

        raise FileNotFoundError(
            f"Missing daily source:\n"
            f"{DAILY_SOURCE_PATH}"
        )

    if DAILY_SOURCE_PATH.is_file():

        return [
            DAILY_SOURCE_PATH
        ]

    parquet_files = sorted(
        DAILY_SOURCE_PATH.rglob(
            "*.parquet"
        )
    )

    if parquet_files:

        return parquet_files

    csv_files = sorted(
        DAILY_SOURCE_PATH.rglob(
            "*.csv"
        )
    )

    if csv_files:

        return csv_files

    raise RuntimeError(
        "No parquet/csv files found under:\n"
        f"{DAILY_SOURCE_PATH}"
    )


def get_file_columns(
    path: Path,
):

    suffix = (
        path.suffix
        .lower()
    )

    if suffix == ".parquet":

        schema = (
            pq.ParquetFile(
                path
            )
            .schema
        )

        return schema.names

    if suffix == ".csv":

        return (
            pd.read_csv(
                path,
                nrows=0,
            )
            .columns
            .tolist()
        )

    raise RuntimeError(
        f"Unsupported file type: {path}"
    )


# ============================================================
# 7. Load raw daily data
# ============================================================

def load_daily_market(
    universe,
    calendar,
):

    files = discover_daily_files()

    print(
        f"Daily files found: "
        f"{len(files)}"
    )

    first_columns = (
        get_file_columns(
            files[0]
        )
    )

    source_columns = {

        canonical:
            resolve_column(
                first_columns,
                aliases,
                canonical,
            )

        for canonical, aliases
        in COLUMN_ALIASES.items()
    }

    print(
        "Resolved source columns:"
    )

    print(
        source_columns
    )

    # --------------------------------------------------------
    # Determine the minimum date required.
    #
    # Need 59 previous market dates for ADV60.
    # --------------------------------------------------------

    min_analysis_index = int(
        universe[
            "market_index"
        ]
        .min()
    )

    max_analysis_index = int(
        universe[
            "market_index"
        ]
        .max()
    )

    history_start_index = max(
        0,
        min_analysis_index
        -
        (
            max(
                LOOKBACK_WINDOWS
            )
            -
            1
        ),
    )

    history_start_date = (
        calendar.loc[
            calendar[
                "market_index"
            ]
            ==
            history_start_index,
            "trade_date",
        ]
        .iloc[0]
    )

    history_end_date = (
        calendar.loc[
            calendar[
                "market_index"
            ]
            ==
            max_analysis_index,
            "trade_date",
        ]
        .iloc[0]
    )

    security_set = set(
        universe[
            "security_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    frames = []

    usecols = list(
        source_columns.values()
    )

    for file_no, path in enumerate(
        files,
        start=1,
    ):

        if (
            file_no
            ==
            1
            or
            file_no
            %
            25
            ==
            0
            or
            file_no
            ==
            len(
                files
            )
        ):

            print(
                f"Reading daily file "
                f"{file_no}/{len(files)}"
            )

        suffix = (
            path.suffix
            .lower()
        )

        if suffix == ".parquet":

            temp = pd.read_parquet(
                path,
                columns=usecols,
            )

        else:

            temp = pd.read_csv(
                path,
                usecols=usecols,
                low_memory=False,
            )

        temp = temp.rename(
            columns={
                source:
                    canonical

                for canonical, source
                in source_columns.items()
            }
        )

        temp[
            "trade_date"
        ] = pd.to_datetime(
            temp[
                "trade_date"
            ],
            errors="coerce",
        )

        temp[
            "security_id"
        ] = normalize_security_id(
            temp[
                "security_id"
            ]
        )

        # ----------------------------------------------------
        # Restrict as early as possible.
        # ----------------------------------------------------

        temp = temp[
            (
                temp[
                    "trade_date"
                ]
                >=
                history_start_date
            )
            &
            (
                temp[
                    "trade_date"
                ]
                <=
                history_end_date
            )
            &
            (
                temp[
                    "security_id"
                ]
                .astype(str)
                .isin(
                    security_set
                )
            )
        ].copy()

        if len(temp) == 0:

            continue

        temp[
            "value"
        ] = pd.to_numeric(
            temp[
                "value"
            ],
            errors="coerce",
        )

        temp[
            "is_open"
        ] = normalize_is_open(
            temp[
                "is_open"
            ]
        )

        frames.append(
            temp[
                [
                    "trade_date",
                    "security_id",
                    "value",
                    "is_open",
                ]
            ]
        )

    if not frames:

        raise RuntimeError(
            "No daily market rows remain "
            "after date/security filtering."
        )

    daily = pd.concat(
        frames,
        ignore_index=True,
    )

    del frames

    # --------------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------------

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

    duplicate_mask = daily[
        [
            "trade_date",
            "security_id",
        ]
    ].duplicated(
        keep=False
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:

        examples = (
            daily.loc[
                duplicate_mask,
                [
                    "trade_date",
                    "security_id",
                ],
            ]
            .head(
                20
            )
        )

        raise RuntimeError(
            "Duplicate stock-date rows found "
            "in daily source.\n"
            f"Count={duplicate_count}\n"
            f"Examples:\n{examples}"
        )

    # --------------------------------------------------------
    # Calendar index
    # --------------------------------------------------------

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

    if daily[
        "market_index"
    ].isna().any():

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
            "Daily source contains dates "
            "not present in trading calendar:\n"
            f"{bad_dates}"
        )

    daily[
        "market_index"
    ] = daily[
        "market_index"
    ].astype(
        np.int64
    )

    # --------------------------------------------------------
    # Source QA
    # --------------------------------------------------------

    negative_value = (

        daily[
            "value"
        ]
        <
        0
    )

    negative_value_count = int(
        negative_value.sum()
    )

    if negative_value_count > 0:

        raise RuntimeError(
            "Negative turnover value found. "
            f"Count={negative_value_count}"
        )

    open_missing_value = (

        daily[
            "is_open"
        ]
        ==
        1

    ) & (

        daily[
            "value"
        ]
        .isna()

    )

    closed_positive_value = (

        daily[
            "is_open"
        ]
        ==
        0

    ) & (

        daily[
            "value"
        ]
        .fillna(
            0
        )
        >
        0

    )

    # --------------------------------------------------------
    # Capacity-oriented trade value:
    #
    # closed -> 0
    # open + valid value -> value
    # unresolved -> NA
    # --------------------------------------------------------

    daily[
        "trade_value_capacity"
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
            "value"
        ]
        .notna()

    ) & (

        np.isfinite(
            daily[
                "value"
            ]
        )

    ) & (

        daily[
            "value"
        ]
        >=
        0
    )

    daily.loc[
        closed,
        "trade_value_capacity",
    ] = 0.0

    daily.loc[
        open_valid,
        "trade_value_capacity",
    ] = daily.loc[
        open_valid,
        "value",
    ]

    daily = (
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

        "negative_value_count":
            negative_value_count,

        "open_missing_value_count":
            int(
                open_missing_value.sum()
            ),

        "closed_positive_value_count":
            int(
                closed_positive_value.sum()
            ),

        "history_start_date":
            history_start_date,

        "history_end_date":
            history_end_date,

        "value_scale_to_cny":
            TRADE_VALUE_SCALE_TO_CNY,

        "value_unit_confirmed":
            bool(
                TRADE_VALUE_SCALE_TO_CNY
                is not None
            ),
    }

    return (
        daily,
        source_qa,
        source_columns,
    )


# ============================================================
# 8. Build PIT liquidity panel
#
# Important:
#
# This calculation does NOT rely on pandas row-count rolling.
#
# For each analysis date, the code explicitly asks:
#
#     which raw observations fall inside
#     [t-W+1, t]
#     in MARKET-TRADING-DAY INDEX?
#
# Therefore missing stock-date rows cannot silently turn a
# 60-market-day window into "the last 60 observations".
# ============================================================

def build_liquidity_panel(
    daily,
    universe,
):

    output_frames = []

    daily_grouped = daily.groupby(
        "security_id",
        sort=False,
    )

    daily_group_keys = set(
        daily_grouped.groups.keys()
    )

    target_grouped = universe.groupby(
        "security_id",
        sort=False,
    )

    total_securities = (
        target_grouped.ngroups
    )

    for security_no, (
        security_id,
        target,
    ) in enumerate(
        target_grouped,
        start=1,
    ):

        if (
            security_no
            ==
            1
            or
            security_no
            %
            250
            ==
            0
            or
            security_no
            ==
            total_securities
        ):

            print(
                f"Liquidity calculation: "
                f"{security_no}/"
                f"{total_securities}"
            )

        target = (
            target.sort_values(
                "market_index"
            )
            .copy()
        )

        target_index = (
            target[
                "market_index"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        # ----------------------------------------------------
        # No raw history for this stock.
        # ----------------------------------------------------

        if (
            security_id
            not in
            daily_group_keys
        ):

            temp = target[
                [
                    "analysis_date",
                    "security_id",
                    "market_index",
                ]
            ].copy()

            temp[
                "current_record_present"
            ] = False

            temp[
                "current_is_open"
            ] = np.nan

            temp[
                "current_trade_value"
            ] = np.nan

            for window in LOOKBACK_WINDOWS:

                temp[
                    f"observed_days_{window}"
                ] = 0

                temp[
                    f"resolved_days_{window}"
                ] = 0

                temp[
                    f"coverage_{window}"
                ] = 0.0

                temp[
                    f"open_days_{window}"
                ] = np.nan

                temp[
                    f"suspended_days_{window}"
                ] = np.nan

                temp[
                    f"full_coverage_{window}"
                ] = False

                temp[
                    f"adv{window}"
                ] = np.nan

                temp[
                    f"median_value_{window}"
                ] = np.nan

            output_frames.append(
                temp
            )

            continue

        history = (
            daily_grouped
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

        trade_value = (
            history[
                "trade_value_capacity"
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

        # ----------------------------------------------------
        # Prefix sums for O(1) rolling mean/count calculations.
        # ----------------------------------------------------

        value_valid = np.isfinite(
            trade_value
        )

        open_known = np.isfinite(
            is_open
        )

        cumulative_value = np.concatenate(
            [
                np.array(
                    [
                        0.0
                    ]
                ),

                np.cumsum(
                    np.where(
                        value_valid,
                        trade_value,
                        0.0,
                    )
                ),
            ]
        )

        cumulative_valid = np.concatenate(
            [
                np.array(
                    [
                        0
                    ],
                    dtype=np.int64,
                ),

                np.cumsum(
                    value_valid.astype(
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
                    np.where(
                        open_known,
                        is_open,
                        0.0,
                    ).astype(
                        np.int64
                    )
                ),
            ]
        )

        cumulative_open_known = np.concatenate(
            [
                np.array(
                    [
                        0
                    ],
                    dtype=np.int64,
                ),

                np.cumsum(
                    open_known.astype(
                        np.int64
                    )
                ),
            ]
        )

        temp = target[
            [
                "analysis_date",
                "security_id",
                "market_index",
            ]
        ].copy()

        # ----------------------------------------------------
        # Current-date state
        # ----------------------------------------------------

        current_pos = np.searchsorted(
            market_index,
            target_index,
            side="left",
        )

        current_exact = (

            current_pos
            <
            len(
                market_index
            )
        )

        exact_indices = np.where(
            current_exact
        )[0]

        if len(
            exact_indices
        ) > 0:

            current_exact[
                exact_indices
            ] &= (

                market_index[
                    current_pos[
                        exact_indices
                    ]
                ]

                ==

                target_index[
                    exact_indices
                ]
            )

        current_open = np.full(
            len(
                target
            ),
            np.nan,
            dtype=float,
        )

        current_value = np.full(
            len(
                target
            ),
            np.nan,
            dtype=float,
        )

        exact_rows = np.where(
            current_exact
        )[0]

        if len(
            exact_rows
        ) > 0:

            source_positions = (
                current_pos[
                    exact_rows
                ]
            )

            current_open[
                exact_rows
            ] = is_open[
                source_positions
            ]

            current_value[
                exact_rows
            ] = trade_value[
                source_positions
            ]

        temp[
            "current_record_present"
        ] = current_exact

        temp[
            "current_is_open"
        ] = current_open

        temp[
            "current_trade_value"
        ] = current_value

        # ----------------------------------------------------
        # ADV20 / ADV60
        # ----------------------------------------------------

        for window in LOOKBACK_WINDOWS:

            start_index = (

                target_index

                -

                (
                    window
                    -
                    1
                )
            )

            left = np.searchsorted(
                market_index,
                start_index,
                side="left",
            )

            right = np.searchsorted(
                market_index,
                target_index,
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

            open_known_days = (

                cumulative_open_known[
                    right
                ]

                -

                cumulative_open_known[
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

            # ------------------------------------------------
            # Full coverage means:
            #
            # 1. exactly W market-date records exist,
            # 2. every isOpen is known,
            # 3. every capacity trade value is resolved.
            #
            # Because the market-index interval itself has W
            # dates, observed_days == W also guarantees that
            # no market date is missing inside the window.
            # ------------------------------------------------

            full_coverage = (

                (
                    observed_days
                    ==
                    window
                )

                &

                (
                    resolved_days
                    ==
                    window
                )

                &

                (
                    open_known_days
                    ==
                    window
                )
            )

            total_value = (

                cumulative_value[
                    right
                ]

                -

                cumulative_value[
                    left
                ]
            )

            adv = np.full(
                len(
                    target
                ),
                np.nan,
                dtype=float,
            )

            adv[
                full_coverage
            ] = (

                total_value[
                    full_coverage
                ]

                /

                float(
                    window
                )
            )

            # ------------------------------------------------
            # Median:
            #
            # computed only for formal full-coverage windows.
            # ------------------------------------------------

            median_value = np.full(
                len(
                    target
                ),
                np.nan,
                dtype=float,
            )

            full_rows = np.where(
                full_coverage
            )[0]

            for row_no in full_rows:

                values = trade_value[
                    left[
                        row_no
                    ]
                    :
                    right[
                        row_no
                    ]
                ]

                # All values must be finite by construction.
                median_value[
                    row_no
                ] = float(
                    np.median(
                        values
                    )
                )

            temp[
                f"observed_days_{window}"
            ] = observed_days

            temp[
                f"resolved_days_{window}"
            ] = resolved_days

            temp[
                f"coverage_{window}"
            ] = (

                resolved_days

                /

                float(
                    window
                )
            )

            temp[
                f"open_days_{window}"
            ] = np.where(
                open_known_days
                ==
                window,
                open_days,
                np.nan,
            )

            temp[
                f"suspended_days_{window}"
            ] = np.where(
                open_known_days
                ==
                window,
                window
                -
                open_days,
                np.nan,
            )

            temp[
                f"full_coverage_{window}"
            ] = full_coverage

            temp[
                f"adv{window}"
            ] = adv

            temp[
                f"median_value_{window}"
            ] = median_value

        output_frames.append(
            temp
        )

    panel = pd.concat(
        output_frames,
        ignore_index=True,
    )

    panel = (
        panel.sort_values(
            [
                "analysis_date",
                "security_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Optional RMB-scaled copies.
    # --------------------------------------------------------

    if (
        TRADE_VALUE_SCALE_TO_CNY
        is not None
    ):

        scale = float(
            TRADE_VALUE_SCALE_TO_CNY
        )

        panel[
            "current_trade_value_cny"
        ] = (

            panel[
                "current_trade_value"
            ]
            *
            scale
        )

        for window in LOOKBACK_WINDOWS:

            panel[
                f"adv{window}_cny"
            ] = (

                panel[
                    f"adv{window}"
                ]
                *
                scale
            )

            panel[
                f"median_value_{window}_cny"
            ] = (

                panel[
                    f"median_value_{window}"
                ]
                *
                scale
            )

    return panel


# ============================================================
# 9. Coverage QA
# ============================================================

def build_coverage_qa(
    panel,
):

    rows = []

    for (
        analysis_date,
        group,
    ) in panel.groupby(
        "analysis_date",
        sort=True,
    ):

        row = {

            "analysis_date":
                analysis_date,

            "universe_n":
                int(
                    len(
                        group
                    )
                ),

            "current_record_n":
                int(
                    group[
                        "current_record_present"
                    ]
                    .sum()
                ),

            "current_record_share":
                float(
                    group[
                        "current_record_present"
                    ]
                    .mean()
                ),

            "current_open_known_n":
                int(
                    group[
                        "current_is_open"
                    ]
                    .notna()
                    .sum()
                ),

            "current_open_share":
                float(
                    (
                        group[
                            "current_is_open"
                        ]
                        ==
                        1
                    )
                    .mean()
                ),

            "current_suspended_n":
                int(
                    (
                        group[
                            "current_is_open"
                        ]
                        ==
                        0
                    )
                    .sum()
                ),
        }

        for window in LOOKBACK_WINDOWS:

            row[
                f"full_coverage_{window}_n"
            ] = int(
                group[
                    f"full_coverage_{window}"
                ]
                .sum()
            )

            row[
                f"full_coverage_{window}_share"
            ] = float(
                group[
                    f"full_coverage_{window}"
                ]
                .mean()
            )

            row[
                f"coverage_{window}_ge_80pct_share"
            ] = float(
                (
                    group[
                        f"coverage_{window}"
                    ]
                    >=
                    RELAXED_COVERAGE_REFERENCE
                )
                .mean()
            )

            row[
                f"adv{window}_nonmissing_share"
            ] = float(
                group[
                    f"adv{window}"
                ]
                .notna()
                .mean()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 10. Liquidity distribution summary
# ============================================================

def build_distribution_summary(
    panel,
):

    rows = []

    quantiles = [
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]

    for (
        analysis_date,
        group,
    ) in panel.groupby(
        "analysis_date",
        sort=True,
    ):

        row = {

            "analysis_date":
                analysis_date,

            "universe_n":
                int(
                    len(
                        group
                    )
                ),
        }

        for window in LOOKBACK_WINDOWS:

            x = pd.to_numeric(
                group[
                    f"adv{window}"
                ],
                errors="coerce",
            )

            x = x[
                x.notna()
                &
                np.isfinite(
                    x
                )
            ]

            row[
                f"adv{window}_n"
            ] = int(
                len(
                    x
                )
            )

            row[
                f"adv{window}_mean"
            ] = (
                float(
                    x.mean()
                )
                if
                len(
                    x
                )
                >
                0
                else
                np.nan
            )

            for q in quantiles:

                label = (
                    f"p"
                    f"{int(q * 100):02d}"
                )

                row[
                    f"adv{window}_{label}"
                ] = (
                    float(
                        x.quantile(
                            q
                        )
                    )
                    if
                    len(
                        x
                    )
                    >
                    0
                    else
                    np.nan
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 11. Formal QA
# ============================================================

def formal_qa(
    panel,
    universe,
    source_qa,
):

    qa = {}

    qa[
        "panel_row_count"
    ] = int(
        len(
            panel
        )
    )

    qa[
        "universe_row_count"
    ] = int(
        len(
            universe
        )
    )

    qa[
        "row_count_matches_universe"
    ] = bool(
        len(
            panel
        )
        ==
        len(
            universe
        )
    )

    qa[
        "panel_keys_unique"
    ] = bool(
        not panel[
            [
                "analysis_date",
                "security_id",
            ]
        ]
        .duplicated()
        .any()
    )

    qa[
        "analysis_date_count"
    ] = int(
        panel[
            "analysis_date"
        ]
        .nunique()
    )

    qa[
        "security_count"
    ] = int(
        panel[
            "security_id"
        ]
        .nunique()
    )

    qa[
        "raw_duplicate_stock_date_count"
    ] = int(
        source_qa[
            "duplicate_stock_date_count"
        ]
    )

    qa[
        "raw_negative_value_count"
    ] = int(
        source_qa[
            "negative_value_count"
        ]
    )

    qa[
        "raw_open_missing_value_count"
    ] = int(
        source_qa[
            "open_missing_value_count"
        ]
    )

    qa[
        "raw_closed_positive_value_count"
    ] = int(
        source_qa[
            "closed_positive_value_count"
        ]
    )

    # --------------------------------------------------------
    # ADV must never be negative.
    # --------------------------------------------------------

    for window in LOOKBACK_WINDOWS:

        adv = panel[
            f"adv{window}"
        ]

        qa[
            f"adv{window}_nonnegative"
        ] = bool(
            (
                adv.dropna()
                >=
                0
            )
            .all()
        )

        # ----------------------------------------------------
        # Main ADV exists iff formal full coverage exists.
        # ----------------------------------------------------

        qa[
            f"adv{window}_matches_full_coverage_flag"
        ] = bool(

            (
                panel[
                    f"adv{window}"
                ]
                .notna()

                ==

                panel[
                    f"full_coverage_{window}"
                ]
            )
            .all()
        )

        qa[
            f"coverage_{window}_within_0_1"
        ] = bool(

            (
                panel[
                    f"coverage_{window}"
                ]
                >=
                0
            )
            .all()

            and

            (
                panel[
                    f"coverage_{window}"
                ]
                <=
                1
                +
                1e-12
            )
            .all()
        )

    # --------------------------------------------------------
    # Closed current stocks must have zero capacity trade value
    # whenever the current record is resolved.
    # --------------------------------------------------------

    current_closed = (

        panel[
            "current_is_open"
        ]
        ==
        0
    )

    qa[
        "current_closed_trade_value_zero"
    ] = bool(

        (
            panel.loc[
                current_closed,
                "current_trade_value",
            ]
            .fillna(
                0
            )
            .abs()
            <=
            1e-14
        )
        .all()
    )

    qa[
        "future_information_used"
    ] = False

    qa[
        "outcome_used"
    ] = False

    qa[
        "factor_selection_performed"
    ] = False

    qa[
        "window_selection_performed"
    ] = False

    qa[
        "trade_value_unit_confirmed_for_cny_capacity"
    ] = bool(
        TRADE_VALUE_SCALE_TO_CNY
        is not None
    )

    # --------------------------------------------------------
    # Unit confirmation is NOT required to construct the
    # source-native liquidity panel.
    #
    # It IS required before monetary AUM capacity analysis.
    # --------------------------------------------------------

    required = [

        "row_count_matches_universe",

        "panel_keys_unique",

        "adv20_nonnegative",

        "adv60_nonnegative",

        "adv20_matches_full_coverage_flag",

        "adv60_matches_full_coverage_flag",

        "coverage_20_within_0_1",

        "coverage_60_within_0_1",

        "current_closed_trade_value_zero",
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
        "Day 8 - Step 1"
    )

    print(
        "Liquidity Data Construction & QA"
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

    print(
        f"Trading days: "
        f"{len(calendar):,}"
    )

    # ========================================================
    # Current universe
    # ========================================================

    print()
    print(
        "[2] Load current factor universe"
    )

    universe = load_current_universe(
        calendar
    )

    print(
        f"Current universe rows: "
        f"{len(universe):,}"
    )

    print(
        f"Analysis dates: "
        f"{universe['analysis_date'].nunique():,}"
    )

    print(
        f"Securities: "
        f"{universe['security_id'].nunique():,}"
    )

    # ========================================================
    # Daily market
    # ========================================================

    print()
    print(
        "[3] Load raw daily market liquidity data"
    )

    (
        daily,
        source_qa,
        source_columns,
    ) = load_daily_market(

        universe,

        calendar,
    )

    print(
        f"Relevant daily rows: "
        f"{len(daily):,}"
    )

    # ========================================================
    # Source QA output
    # ========================================================

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
    # PIT liquidity
    # ========================================================

    print()
    print(
        "[4] Build PIT ADV20 / ADV60"
    )

    liquidity_panel = (
        build_liquidity_panel(

            daily,

            universe,
        )
    )

    save_parquet_atomic(
        liquidity_panel,
        LIQUIDITY_PANEL_PATH,
    )

    # ========================================================
    # Coverage QA
    # ========================================================

    print()
    print(
        "[5] Build liquidity coverage QA"
    )

    coverage_qa = (
        build_coverage_qa(
            liquidity_panel
        )
    )

    save_csv_atomic(
        coverage_qa,
        COVERAGE_QA_PATH,
    )

    # ========================================================
    # Distribution
    # ========================================================

    print()
    print(
        "[6] Build liquidity distribution summary"
    )

    distribution = (
        build_distribution_summary(
            liquidity_panel
        )
    )

    save_csv_atomic(
        distribution,
        DISTRIBUTION_PATH,
    )

    # ========================================================
    # Formal QA
    # ========================================================

    print()
    print(
        "[7] Formal QA"
    )

    qa = formal_qa(

        liquidity_panel,

        universe,

        source_qa,
    )

    if (
        not qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            "Day-8 Step-1 formal QA failed.\n"
            f"{qa}"
        )

    # ========================================================
    # Frozen design
    # ========================================================

    design_payload = {

        "research_day":
            8,

        "step":
            "Step1_Liquidity_Data_Construction_and_QA",

        "current_universe_source":
            str(
                FACTOR_PANEL_PATH
            ),

        "daily_source":
            str(
                DAILY_SOURCE_PATH
            ),

        "trade_calendar":
            str(
                TRADE_CALENDAR_PATH
            ),

        "lookback_windows":
            LOOKBACK_WINDOWS,

        "adv_definition":
            (
                "Arithmetic mean of capacity-oriented "
                "daily turnover value across exactly W "
                "market trading days ending at analysis_date."
            ),

        "suspension_treatment":
            (
                "isOpen == 0 -> daily trade value = 0."
            ),

        "open_valid_treatment":
            (
                "isOpen == 1 and valid value -> "
                "daily trade value = source value."
            ),

        "unresolved_missing_treatment":
            (
                "Unresolved daily value remains NA."
            ),

        "formal_adv_coverage":
            (
                "Complete W/W market-day coverage required."
            ),

        "median_liquidity_robustness":
            True,

        "value_scale_to_cny":
            TRADE_VALUE_SCALE_TO_CNY,

        "value_unit_confirmed":
            bool(
                TRADE_VALUE_SCALE_TO_CNY
                is not None
            ),

        "future_information_used":
            False,

        "future_outcome_used":
            False,

        "factor_selection":
            False,

        "window_selection":
            False,
    }

    design_hash = canonical_hash(
        design_payload
    )

    metadata = {

        **design_payload,

        "day8_step1_design_hash":
            design_hash,

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "resolved_source_columns":
            source_columns,

        "source_qa":
            source_qa,

        "formal_qa":
            qa,

        "outputs":
            {

                "stock_liquidity_panel":
                    str(
                        LIQUIDITY_PANEL_PATH
                    ),

                "liquidity_coverage_qa":
                    str(
                        COVERAGE_QA_PATH
                    ),

                "liquidity_distribution_summary":
                    str(
                        DISTRIBUTION_PATH
                    ),

                "liquidity_source_qa":
                    str(
                        SOURCE_QA_PATH
                    ),
            },

        "important_notes":
            [

                (
                    "ADV is based on market trading days, "
                    "not the last W observed stock rows."
                ),

                (
                    "Suspension days contribute zero turnover "
                    "rather than being removed from the denominator."
                ),

                (
                    "Formal ADV20/ADV60 require complete "
                    "20/60-day resolved histories."
                ),

                (
                    "The liquidity panel is constructed only "
                    "from dates <= analysis_date."
                ),

                (
                    "Source turnover-value units must be "
                    "confirmed before monetary AUM capacity "
                    "is interpreted in RMB."
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
        "Day-8 Step-1 Formal QA"
    )

    print("=" * 80)

    print(
        f"Panel rows: "
        f"{qa['panel_row_count']:,}"
    )

    print(
        f"Analysis dates: "
        f"{qa['analysis_date_count']:,}"
    )

    print(
        f"Securities: "
        f"{qa['security_count']:,}"
    )

    print(
        f"ADV20 nonnegative: "
        f"{qa['adv20_nonnegative']}"
    )

    print(
        f"ADV60 nonnegative: "
        f"{qa['adv60_nonnegative']}"
    )

    print(
        f"ADV20/full-coverage consistency: "
        f"{qa['adv20_matches_full_coverage_flag']}"
    )

    print(
        f"ADV60/full-coverage consistency: "
        f"{qa['adv60_matches_full_coverage_flag']}"
    )

    print(
        f"Formal QA pass: "
        f"{qa['all_formal_qa_pass']}"
    )

    print()

    if (
        TRADE_VALUE_SCALE_TO_CNY
        is None
    ):

        print(
            "WARNING:"
        )

        print(
            "The source `value` unit has NOT yet been "
            "confirmed in RMB."
        )

        print(
            "Step 1 is valid in source-native units, "
            "but confirm the vendor unit before Step 2 "
            "AUM/ADV capacity analysis."
        )

    else:

        print(
            "Turnover value scale to CNY: "
            f"{TRADE_VALUE_SCALE_TO_CNY}"
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
        "Day 8 Step 1 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()