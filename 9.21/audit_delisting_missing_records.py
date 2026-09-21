from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
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
# Trading calendar
# ------------------------------------------------------------

TRADE_CALENDAR_PATH = Path(
    r"D:\lowfreq\trade_calendar\trade_date.csv"
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
# Day 8 Step 2B
# ------------------------------------------------------------

STEP2B_DIR = (
    OUTPUT_ROOT
    / "M1_day8"
    / "02b_stage2b_dedicated_drift_return"
)

STEP2B_DRIFT_PATH = (
    STEP2B_DIR
    / "dedicated_drift_return_panel.parquet"
)

STEP2B_ISSUE_PATH = (
    STEP2B_DIR
    / "dedicated_drift_return_issue_detail.csv"
)

STEP2B_FORMAL_QA_PATH = (
    STEP2B_DIR
    / "dedicated_drift_return_formal_qa.csv"
)


# ------------------------------------------------------------
# Stock master
#
# If you KNOW the exact stock-master path,
# set it explicitly here.
#
# Example:
#
# STOCK_MASTER_PATH = (
#     ROOT
#     / "status"
#     / "equ.csv"
# )
#
# Otherwise leave None.
#
# The script will search under:
#
#     D:\M1_StockNetwork\status
#     D:\M1_StockNetwork\master
#     D:\M1_StockNetwork\basic
#
# for a file containing:
#
# security_id/secID
# list_date/listDate
# delist_date/delistDate
# ------------------------------------------------------------

# Use the final Day-1 stock master, not older stage1/fixed snapshots.
STOCK_MASTER_PATH = (
    OUTPUT_ROOT
    / "M1_day1"
    / "01_stage1_final"
    / "stock_master.parquet"
)


# ------------------------------------------------------------
# Day 8 Step 2C
#
# 02c is already occupied by the dedicated-drift Step-2 rerun,
# therefore use 02d physically while keeping the research
# label "Step 2C".
# ------------------------------------------------------------

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day8"
    / "02d_stage2c_delisting_missing_record_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


AUDIT_DETAIL_PATH = (
    OUTPUT_DIR
    / "missing_record_audit_detail.csv"
)

DAILY_AUDIT_DETAIL_PATH = (
    OUTPUT_DIR
    / "missing_record_daily_audit_detail.csv"
)

AUDIT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "missing_record_audit_summary.csv"
)

AUDITED_DRIFT_PATH = (
    OUTPUT_DIR
    / "dedicated_drift_return_panel_audited.parquet"
)

FORMAL_QA_PATH = (
    OUTPUT_DIR
    / "step2c_formal_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day8_step2c_metadata.json"
)


# ============================================================
# 1. Fixed design
# ============================================================

# ------------------------------------------------------------
# Strict rule:
#
# Only a delisting-boundary case satisfying ALL conditions
# below can be automatically repaired:
#
# 1. Previous portfolio date is before delist_date.
# 2. delist_date lies inside (previous_date, current_date].
# 3. Every market day before delist_date inside the holding
#    interval has a stock source record.
# 4. Every required pre-delist holding return is resolved.
# 5. No non-zero holding return is observed on/after
#    delist_date.
#
# Then post-delist time is treated as cash with zero return.
# ------------------------------------------------------------

APPLY_DELIST_CASH_CARRY_REPAIRS = True


TOL = 1e-12


# ============================================================
# 2. Column aliases
# ============================================================

SECURITY_ALIASES = [
    "security_id",
    "secID",
    "sec_id",
]

DATE_ALIASES = [
    "trade_date",
    "tradeDate",
    "calendarDate",
    "date",
]

IS_OPEN_ALIASES = [
    "isOpen",
    "is_open",
]

RETURN_ALIASES = [
    "return_last_trade_simple",
    "last_trade_simple_return",
    "return_last_trade",
]

LIST_DATE_ALIASES = [
    "list_date",
    "listDate",
    "listed_date",
    "listedDate",
]

DELIST_DATE_ALIASES = [
    "delist_date",
    "delistDate",
    "delisted_date",
    "delistedDate",
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
    aliases,
    name,
):

    columns = list(
        columns
    )

    for candidate in aliases:

        if candidate in columns:

            return candidate

    raise RuntimeError(
        f"Cannot resolve `{name}`.\n"
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

    # to_numeric preserves bool input; pandas 3 requires an explicit
    # conversion before assigning it to the float64 result.
    result.loc[
        numeric.notna()
    ] = numeric.loc[
        numeric.notna()
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
            "Invalid isOpen values found."
        )

    return result


# ============================================================
# 4. Validate Step-2B upstream QA
# ============================================================

def validate_step2b_qa():

    if not STEP2B_FORMAL_QA_PATH.exists():

        raise FileNotFoundError(
            f"Missing Step-2B QA:\n"
            f"{STEP2B_FORMAL_QA_PATH}"
        )

    qa = pd.read_csv(
        STEP2B_FORMAL_QA_PATH
    )

    if not {
        "qa_name",
        "qa_value",
    }.issubset(
        qa.columns
    ):

        raise RuntimeError(
            "Unexpected Step-2B QA schema."
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

    value = str(
        qa_map.get(
            "all_formal_qa_pass",
            ""
        )
    ).strip().lower()

    if value not in {
        "true",
        "1",
    }:

        raise RuntimeError(
            "Step-2B formal QA did not pass."
        )

    return qa_map


# ============================================================
# 5. Load trading calendar
# ============================================================

def load_trade_calendar():

    raw = pd.read_csv(
        TRADE_CALENDAR_PATH
    )

    date_column = resolve_column(
        raw.columns,
        DATE_ALIASES,
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

    open_column = None

    for candidate in (
        IS_OPEN_ALIASES
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
# 6. Load Step-2B issue rows
# ============================================================

def load_issue_rows():

    if not STEP2B_ISSUE_PATH.exists():

        raise FileNotFoundError(
            f"Missing Step-2B issue detail:\n"
            f"{STEP2B_ISSUE_PATH}"
        )

    issue = pd.read_csv(
        STEP2B_ISSUE_PATH
    )

    required = [
        "previous_analysis_date",
        "analysis_date",
        "security_id",
        "status",
    ]

    missing = [
        column
        for column in required
        if column not in issue.columns
    ]

    if missing:

        raise RuntimeError(
            "Issue detail missing columns:\n"
            f"{missing}"
        )

    issue[
        "previous_analysis_date"
    ] = pd.to_datetime(
        issue[
            "previous_analysis_date"
        ],
        errors="raise",
    )

    issue[
        "analysis_date"
    ] = pd.to_datetime(
        issue[
            "analysis_date"
        ],
        errors="raise",
    )

    issue[
        "security_id"
    ] = normalize_security_id(
        issue[
            "security_id"
        ]
    )

    # --------------------------------------------------------
    # Step-2B showed all 107 issues as MISSING_SOURCE_RECORD.
    #
    # Keep this check explicit rather than silently assuming it.
    # --------------------------------------------------------

    unexpected = (
        ~issue[
            "status"
        ]
        .eq(
            "MISSING_SOURCE_RECORD"
        )
    )

    if unexpected.any():

        print(
            "WARNING:"
        )

        print(
            "Issue file contains statuses other than "
            "MISSING_SOURCE_RECORD."
        )

        print(
            issue.loc[
                unexpected,
                "status",
            ]
            .value_counts()
        )

    key = [
        "previous_analysis_date",
        "analysis_date",
        "security_id",
    ]

    if issue[
        key
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Step-2B issue keys."
        )

    return issue


# ============================================================
# 7. Stock-master discovery
# ============================================================

def get_file_columns(
    path: Path,
):

    suffix = (
        path.suffix
        .lower()
    )

    if suffix == ".csv":

        return (
            pd.read_csv(
                path,
                nrows=0,
            )
            .columns
            .tolist()
        )

    if suffix == ".parquet":

        return (
            pq.ParquetFile(
                path
            )
            .schema
            .names
        )

    return []


def has_any(
    columns,
    aliases,
):

    return any(
        x in columns
        for x in aliases
    )


def discover_stock_master():

    if STOCK_MASTER_PATH is not None:

        path = Path(
            STOCK_MASTER_PATH
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Configured STOCK_MASTER_PATH "
                f"does not exist:\n{path}"
            )

        return path

    search_dirs = [
        ROOT / "status",
        ROOT / "master",
        ROOT / "basic",
    ]

    candidates = []

    for directory in search_dirs:

        if not directory.exists():

            continue

        files = (
            list(
                directory.rglob(
                    "*.csv"
                )
            )
            +
            list(
                directory.rglob(
                    "*.parquet"
                )
            )
        )

        for path in files:

            try:

                columns = (
                    get_file_columns(
                        path
                    )
                )

            except Exception:

                continue

            if not (
                has_any(
                    columns,
                    SECURITY_ALIASES,
                )
                and
                has_any(
                    columns,
                    LIST_DATE_ALIASES,
                )
                and
                has_any(
                    columns,
                    DELIST_DATE_ALIASES,
                )
            ):

                continue

            name = (
                path.name
                .lower()
            )

            score = 0

            for word in [
                "master",
                "equ",
                "stock",
                "security",
            ]:

                if word in name:

                    score += 1

            candidates.append(
                (
                    score,
                    path,
                )
            )

    if not candidates:

        raise FileNotFoundError(
            "Cannot automatically find a stock-master file "
            "containing security_id/secID + listDate + "
            "delistDate.\n\n"
            "Please set STOCK_MASTER_PATH explicitly near "
            "the top of this script."
        )

    candidates.sort(
        key=lambda x: (
            -x[0],
            str(
                x[1]
            ),
        )
    )

    selected = (
        candidates[0][1]
    )

    print(
        "Automatically selected stock master:"
    )

    print(
        selected
    )

    if len(
        candidates
    ) > 1:

        print()
        print(
            "Other compatible stock-master candidates:"
        )

        for score, path in (
            candidates[
                1:6
            ]
        ):

            print(
                f"  score={score}: {path}"
            )

    return selected


# ============================================================
# 8. Load stock master
# ============================================================

def load_stock_master(
    issue_ids,
):

    path = discover_stock_master()

    columns = get_file_columns(
        path
    )

    security_column = resolve_column(
        columns,
        SECURITY_ALIASES,
        "security_id",
    )

    list_column = resolve_column(
        columns,
        LIST_DATE_ALIASES,
        "list_date",
    )

    delist_column = resolve_column(
        columns,
        DELIST_DATE_ALIASES,
        "delist_date",
    )

    usecols = [
        security_column,
        list_column,
        delist_column,
    ]

    if (
        path.suffix.lower()
        ==
        ".csv"
    ):

        master = pd.read_csv(
            path,
            usecols=usecols,
            low_memory=False,
        )

    else:

        master = pd.read_parquet(
            path,
            columns=usecols,
        )

    master = master.rename(
        columns={
            security_column:
                "security_id",

            list_column:
                "list_date",

            delist_column:
                "delist_date",
        }
    )

    master[
        "security_id"
    ] = normalize_security_id(
        master[
            "security_id"
        ]
    )

    master[
        "list_date"
    ] = pd.to_datetime(
        master[
            "list_date"
        ],
        errors="coerce",
    )

    master[
        "delist_date"
    ] = pd.to_datetime(
        master[
            "delist_date"
        ],
        errors="coerce",
    )

    master = master[
        master[
            "security_id"
        ]
        .isin(
            issue_ids
        )
    ].copy()

    # --------------------------------------------------------
    # Allow exact duplicate records, but not conflicting
    # list/delist records.
    # --------------------------------------------------------

    master = master.drop_duplicates()

    duplicate_ids = master[
        "security_id"
    ].duplicated(
        keep=False
    )

    if duplicate_ids.any():

        conflict = (

            master.loc[
                duplicate_ids
            ]
            .groupby(
                "security_id"
            )[
                [
                    "list_date",
                    "delist_date",
                ]
            ]
            .nunique(
                dropna=False
            )
        )

        conflict = conflict[
            (
                conflict[
                    "list_date"
                ]
                >
                1
            )
            |
            (
                conflict[
                    "delist_date"
                ]
                >
                1
            )
        ]

        if len(
            conflict
        ) > 0:

            raise RuntimeError(
                "Conflicting list/delist dates in stock master:\n"
                f"{conflict.head(20)}"
            )

        master = (
            master.drop_duplicates(
                subset=[
                    "security_id"
                ]
            )
        )

    return (
        master,
        path,
    )


# ============================================================
# 9. Load Day-1 returns for ONLY issue securities
# ============================================================

def load_issue_daily_returns(
    issue,
):

    dataset = ds.dataset(
        str(
            DAY1_RETURN_PANEL_PATH
        ),
        format="parquet",
    )

    columns = (
        dataset.schema.names
    )

    security_column = resolve_column(
        columns,
        SECURITY_ALIASES,
        "security_id",
    )

    date_column = resolve_column(
        columns,
        DATE_ALIASES,
        "trade_date",
    )

    is_open_column = resolve_column(
        columns,
        IS_OPEN_ALIASES,
        "is_open",
    )

    return_column = resolve_column(
        columns,
        RETURN_ALIASES,
        "return_last_trade_simple",
    )

    issue_ids = (
        issue[
            "security_id"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # Filter by issue securities inside Arrow.
    #
    # Date is restricted later in pandas to avoid dtype
    # mismatches if the parquet date field is stored as string.
    # --------------------------------------------------------

    filter_expression = (
        ds.field(
            security_column
        )
        .isin(
            issue_ids
        )
    )

    table = dataset.to_table(
        columns=[
            security_column,
            date_column,
            is_open_column,
            return_column,
        ],
        filter=filter_expression,
    )

    daily = table.to_pandas()

    daily = daily.rename(
        columns={
            security_column:
                "security_id",

            date_column:
                "trade_date",

            is_open_column:
                "is_open",

            return_column:
                "return_last_trade_simple",
        }
    )

    daily[
        "security_id"
    ] = normalize_security_id(
        daily[
            "security_id"
        ]
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

    min_date = (
        issue[
            "previous_analysis_date"
        ]
        .min()
    )

    max_date = (
        issue[
            "analysis_date"
        ]
        .max()
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
    ].copy()

    duplicate = daily[
        [
            "security_id",
            "trade_date",
        ]
    ].duplicated(
        keep=False
    )

    if duplicate.any():

        raise RuntimeError(
            "Duplicate security-date rows found "
            "in Day-1 return source."
        )

    # --------------------------------------------------------
    # Reconstruct the same holding-return semantics used
    # throughout the project.
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

    invalid = (
        daily[
            "holding_return"
        ]
        .dropna()
        <
        -1.0
        -
        TOL
    )

    if invalid.any():

        raise RuntimeError(
            "Holding return below -100% detected."
        )

    return (
        daily.sort_values(
            [
                "security_id",
                "trade_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 10. Safe return compounding
# ============================================================

def compound_return(
    returns: pd.Series,
):

    r = pd.to_numeric(
        returns,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if len(
        r
    ) == 0:

        return 0.0

    if (
        ~np.isfinite(
            r
        )
    ).any():

        return np.nan

    if (
        r
        <
        -1.0
        -
        TOL
    ).any():

        return np.nan

    gross = (
        1.0
        +
        r
    )

    if (
        np.isclose(
            gross,
            0.0,
            atol=1e-15,
            rtol=0.0,
        )
    ).any():

        return -1.0

    return float(
        np.exp(
            np.log(
                gross
            ).sum()
        )
        -
        1.0
    )


# ============================================================
# 11. Audit every Step-2B issue
# ============================================================

def audit_issues(
    issue,
    master,
    daily,
    calendar,
):

    master_map = (
        master.set_index(
            "security_id"
        )
    )

    daily_groups = {
        security_id:
            temp.set_index(
                "trade_date"
            )

        for security_id, temp
        in daily.groupby(
            "security_id",
            sort=False,
        )
    }

    audit_rows = []
    daily_rows = []

    for row in issue.itertuples(
        index=False
    ):

        security_id = str(
            row.security_id
        )

        previous_date = pd.Timestamp(
            row.previous_analysis_date
        )

        current_date = pd.Timestamp(
            row.analysis_date
        )

        expected_dates = (
            calendar.loc[
                (
                    calendar[
                        "trade_date"
                    ]
                    >
                    previous_date
                )
                &
                (
                    calendar[
                        "trade_date"
                    ]
                    <=
                    current_date
                ),
                "trade_date",
            ]
            .tolist()
        )

        # ----------------------------------------------------
        # Stock-master state
        # ----------------------------------------------------

        master_found = (
            security_id
            in
            master_map.index
        )

        if master_found:

            master_row = (
                master_map.loc[
                    security_id
                ]
            )

            list_date = (
                master_row[
                    "list_date"
                ]
            )

            delist_date = (
                master_row[
                    "delist_date"
                ]
            )

        else:

            list_date = pd.NaT
            delist_date = pd.NaT

        # ----------------------------------------------------
        # Source rows
        # ----------------------------------------------------

        source = daily_groups.get(
            security_id
        )

        per_day = []

        for date in expected_dates:

            record_present = (
                source is not None
                and
                date in source.index
            )

            if record_present:

                source_row = (
                    source.loc[
                        date
                    ]
                )

                # Should be scalar because uniqueness
                # was checked.
                is_open = (
                    source_row[
                        "is_open"
                    ]
                )

                raw_return = (
                    source_row[
                        "return_last_trade_simple"
                    ]
                )

                holding_return = (
                    source_row[
                        "holding_return"
                    ]
                )

            else:

                is_open = np.nan
                raw_return = np.nan
                holding_return = np.nan

            if pd.notna(
                list_date
            ):

                listed_by_date = (
                    date
                    >=
                    list_date
                )

            else:

                listed_by_date = np.nan

            if pd.notna(
                delist_date
            ):

                before_delist = (
                    date
                    <
                    delist_date
                )

                post_delist = (
                    date
                    >=
                    delist_date
                )

            else:

                before_delist = True
                post_delist = False

            if pd.notna(
                list_date
            ):

                should_be_active = bool(
                    listed_by_date
                    and
                    before_delist
                )

            else:

                should_be_active = bool(
                    before_delist
                )

            holding_resolved = bool(
                record_present
                and
                pd.notna(
                    holding_return
                )
                and
                np.isfinite(
                    holding_return
                )
            )

            per_day.append(
                {
                    "previous_analysis_date":
                        previous_date,

                    "analysis_date":
                        current_date,

                    "security_id":
                        security_id,

                    "trade_date":
                        date,

                    "list_date":
                        list_date,

                    "delist_date":
                        delist_date,

                    "should_be_active":
                        should_be_active,

                    "post_delist":
                        bool(
                            post_delist
                        ),

                    "source_record_present":
                        bool(
                            record_present
                        ),

                    "is_open":
                        is_open,

                    "return_last_trade_simple":
                        raw_return,

                    "holding_return":
                        holding_return,

                    "holding_return_resolved":
                        holding_resolved,
                }
            )

        detail = pd.DataFrame(
            per_day
        )

        # ----------------------------------------------------
        # Missing / unresolved counts
        # ----------------------------------------------------

        active_mask = (
            detail[
                "should_be_active"
            ]
        )

        post_delist_mask = (
            detail[
                "post_delist"
            ]
        )

        active_missing_n = int(
            (
                active_mask
                &
                ~detail[
                    "source_record_present"
                ]
            )
            .sum()
        )

        active_unresolved_n = int(
            (
                active_mask
                &
                detail[
                    "source_record_present"
                ]
                &
                ~detail[
                    "holding_return_resolved"
                ]
            )
            .sum()
        )

        post_delist_missing_n = int(
            (
                post_delist_mask
                &
                ~detail[
                    "source_record_present"
                ]
            )
            .sum()
        )

        post_delist_present_n = int(
            (
                post_delist_mask
                &
                detail[
                    "source_record_present"
                ]
            )
            .sum()
        )

        post_delist_nonzero_return_n = int(
            (
                post_delist_mask
                &
                detail[
                    "holding_return_resolved"
                ]
                &
                (
                    detail[
                        "holding_return"
                    ]
                    .abs()
                    >
                    TOL
                )
            )
            .sum()
        )

        source_missing_total_n = int(
            (
                ~detail[
                    "source_record_present"
                ]
            )
            .sum()
        )

        resolved_active_returns = (
            detail.loc[
                active_mask
                &
                detail[
                    "holding_return_resolved"
                ],
                "holding_return",
            ]
        )

        active_compounded_return = (
            compound_return(
                resolved_active_returns
            )
        )

        # ----------------------------------------------------
        # Last available source information
        # ----------------------------------------------------

        available = detail[
            detail[
                "source_record_present"
            ]
        ]

        if len(
            available
        ) > 0:

            last_source_date = (
                available[
                    "trade_date"
                ]
                .max()
            )

        else:

            last_source_date = pd.NaT

        active_available = detail[
            active_mask
            &
            detail[
                "source_record_present"
            ]
        ]

        if len(
            active_available
        ) > 0:

            last_active_source_date = (
                active_available[
                    "trade_date"
                ]
                .max()
            )

        else:

            last_active_source_date = pd.NaT

        # ====================================================
        # Classification
        # ====================================================

        repair_eligible = False

        audited_return = np.nan

        if not master_found:

            classification = (
                "MASTER_SECURITY_NOT_FOUND"
            )

            recommendation = (
                "DO_NOT_REPAIR"
            )

        elif (
            pd.notna(
                list_date
            )
            and
            list_date
            >
            previous_date
        ):

            classification = (
                "LIST_DATE_CONFLICT_PREVIOUS_HOLDING"
            )

            recommendation = (
                "DO_NOT_REPAIR_AUDIT_MASTER"
            )

        elif (
            pd.notna(
                delist_date
            )
            and
            delist_date
            <=
            previous_date
        ):

            classification = (
                "DELIST_DATE_CONFLICT_PREVIOUS_HOLDING"
            )

            recommendation = (
                "DO_NOT_REPAIR_AUDIT_MASTER"
            )

        elif (
            pd.notna(
                delist_date
            )
            and
            previous_date
            <
            delist_date
            <=
            current_date
        ):

            # ------------------------------------------------
            # Delisting occurred during the holding interval.
            # ------------------------------------------------

            if (
                active_missing_n
                >
                0
                or
                active_unresolved_n
                >
                0
            ):

                classification = (
                    "PRE_DELIST_SOURCE_GAP"
                )

                recommendation = (
                    "DO_NOT_REPAIR"
                )

            elif (
                post_delist_nonzero_return_n
                >
                0
            ):

                classification = (
                    "POST_DELIST_SOURCE_CONFLICT"
                )

                recommendation = (
                    "DO_NOT_REPAIR_AUDIT_MASTER"
                )

            else:

                classification = (
                    "DELIST_POST_BOUNDARY_ONLY"
                )

                recommendation = (
                    "REPAIR_WITH_DELIST_CASH_CARRY"
                )

                repair_eligible = True

                audited_return = (
                    active_compounded_return
                )

        else:

            # ------------------------------------------------
            # Stock should still be active over the interval,
            # but records are missing.
            # ------------------------------------------------

            classification = (
                "ACTIVE_PERIOD_TRUE_SOURCE_GAP"
            )

            recommendation = (
                "DO_NOT_REPAIR"
            )

        # ----------------------------------------------------
        # Safety:
        #
        # Never mark a return repairable unless it is finite
        # and >= -1.
        # ----------------------------------------------------

        if repair_eligible:

            if (
                pd.isna(
                    audited_return
                )
                or
                not np.isfinite(
                    audited_return
                )
                or
                audited_return
                <
                -1.0
                -
                TOL
            ):

                repair_eligible = False

                classification = (
                    "DELIST_RETURN_RECONSTRUCTION_FAILURE"
                )

                recommendation = (
                    "DO_NOT_REPAIR"
                )

                audited_return = np.nan

        # ----------------------------------------------------
        # Date strings useful for manual audit
        # ----------------------------------------------------

        missing_dates = (
            detail.loc[
                ~detail[
                    "source_record_present"
                ],
                "trade_date",
            ]
            .dt.strftime(
                "%Y-%m-%d"
            )
            .tolist()
        )

        active_missing_dates = (
            detail.loc[
                active_mask
                &
                ~detail[
                    "source_record_present"
                ],
                "trade_date",
            ]
            .dt.strftime(
                "%Y-%m-%d"
            )
            .tolist()
        )

        post_delist_missing_dates = (
            detail.loc[
                post_delist_mask
                &
                ~detail[
                    "source_record_present"
                ],
                "trade_date",
            ]
            .dt.strftime(
                "%Y-%m-%d"
            )
            .tolist()
        )

        audit_rows.append(
            {
                "previous_analysis_date":
                    previous_date,

                "analysis_date":
                    current_date,

                "security_id":
                    security_id,

                "step2b_status":
                    row.status,

                "step2b_missing_record_days":
                    getattr(
                        row,
                        "missing_record_days",
                        np.nan,
                    ),

                "master_found":
                    master_found,

                "list_date":
                    list_date,

                "delist_date":
                    delist_date,

                "delist_in_holding_interval":
                    bool(
                        pd.notna(
                            delist_date
                        )
                        and
                        previous_date
                        <
                        delist_date
                        <=
                        current_date
                    ),

                "expected_market_days":
                    int(
                        len(
                            detail
                        )
                    ),

                "source_missing_total_n":
                    source_missing_total_n,

                "active_missing_n":
                    active_missing_n,

                "active_unresolved_n":
                    active_unresolved_n,

                "post_delist_missing_n":
                    post_delist_missing_n,

                "post_delist_present_n":
                    post_delist_present_n,

                "post_delist_nonzero_return_n":
                    post_delist_nonzero_return_n,

                "last_source_date":
                    last_source_date,

                "last_active_source_date":
                    last_active_source_date,

                "active_compounded_return":
                    active_compounded_return,

                "audit_classification":
                    classification,

                "audit_recommendation":
                    recommendation,

                "repair_eligible":
                    bool(
                        repair_eligible
                    ),

                "audited_drift_total_return":
                    audited_return,

                "all_missing_dates":
                    ";".join(
                        missing_dates
                    ),

                "active_missing_dates":
                    ";".join(
                        active_missing_dates
                    ),

                "post_delist_missing_dates":
                    ";".join(
                        post_delist_missing_dates
                    ),
            }
        )

        daily_rows.extend(
            per_day
        )

    audit = pd.DataFrame(
        audit_rows
    )

    daily_audit = pd.DataFrame(
        daily_rows
    )

    return (
        audit,
        daily_audit,
    )


# ============================================================
# 12. Build audited dedicated-drift panel
# ============================================================

def build_audited_drift_panel(
    audit,
):

    if not STEP2B_DRIFT_PATH.exists():

        raise FileNotFoundError(
            f"Missing Step-2B drift panel:\n"
            f"{STEP2B_DRIFT_PATH}"
        )

    drift = pd.read_parquet(
        STEP2B_DRIFT_PATH
    )

    drift[
        "previous_analysis_date"
    ] = pd.to_datetime(
        drift[
            "previous_analysis_date"
        ],
        errors="raise",
    )

    drift[
        "analysis_date"
    ] = pd.to_datetime(
        drift[
            "analysis_date"
        ],
        errors="raise",
    )

    drift[
        "security_id"
    ] = normalize_security_id(
        drift[
            "security_id"
        ]
    )

    key = [
        "previous_analysis_date",
        "analysis_date",
        "security_id",
    ]

    if drift[
        key
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Step-2B drift keys."
        )

    # --------------------------------------------------------
    # Preserve original values explicitly.
    # --------------------------------------------------------

    drift[
        "status_original"
    ] = drift[
        "status"
    ]

    drift[
        "drift_total_return_original"
    ] = drift[
        "drift_total_return"
    ]

    audit_merge_columns = (
        key
        +
        [
            "audit_classification",
            "audit_recommendation",
            "repair_eligible",
            "audited_drift_total_return",
            "list_date",
            "delist_date",
        ]
    )

    drift = drift.merge(
        audit[
            audit_merge_columns
        ],
        on=key,
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Canonical audited values start from original values.
    # --------------------------------------------------------

    drift[
        "status_audited"
    ] = drift[
        "status_original"
    ]

    drift[
        "drift_total_return_audited"
    ] = drift[
        "drift_total_return_original"
    ]

    repair_mask = (
        drift[
            "repair_eligible"
        ]
        .fillna(
            False
        )
        .astype(bool)
    )

    if (
        APPLY_DELIST_CASH_CARRY_REPAIRS
    ):

        drift.loc[
            repair_mask,
            "status_audited",
        ] = (
            "PASS_DELIST_CASH_CARRY"
        )

        drift.loc[
            repair_mask,
            "drift_total_return_audited",
        ] = drift.loc[
            repair_mask,
            "audited_drift_total_return",
        ]

    # --------------------------------------------------------
    # To make downstream usage explicit:
    #
    # canonical `status` and `drift_total_return`
    # now represent the audited version.
    #
    # Original values remain available in *_original.
    # --------------------------------------------------------

    drift[
        "status"
    ] = drift[
        "status_audited"
    ]

    drift[
        "drift_total_return"
    ] = drift[
        "drift_total_return_audited"
    ]

    return drift


# ============================================================
# 13. Summary
# ============================================================

def build_audit_summary(
    audit,
):

    summary = (

        audit.groupby(
            [
                "audit_classification",
                "audit_recommendation",
                "repair_eligible",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            issue_n=(
                "security_id",
                "size",
            ),

            security_n=(
                "security_id",
                "nunique",
            ),

            mean_missing_days=(
                "source_missing_total_n",
                "mean",
            ),

            max_missing_days=(
                "source_missing_total_n",
                "max",
            ),
        )
    )

    return summary


# ============================================================
# 14. Formal QA
# ============================================================

def formal_qa(
    issue,
    audit,
    audited_drift,
):

    qa = {}

    key = [
        "previous_analysis_date",
        "analysis_date",
        "security_id",
    ]

    qa[
        "input_issue_count"
    ] = int(
        len(
            issue
        )
    )

    qa[
        "audit_row_count"
    ] = int(
        len(
            audit
        )
    )

    qa[
        "audit_row_count_matches_issue_count"
    ] = bool(
        len(
            issue
        )
        ==
        len(
            audit
        )
    )

    qa[
        "audit_keys_unique"
    ] = bool(
        not audit[
            key
        ]
        .duplicated()
        .any()
    )

    qa[
        "classification_complete"
    ] = bool(
        audit[
            "audit_classification"
        ]
        .notna()
        .all()
    )

    repair = (
        audit[
            "repair_eligible"
        ]
    )

    qa[
        "repair_eligible_count"
    ] = int(
        repair.sum()
    )

    qa[
        "remaining_unrepaired_issue_count"
    ] = int(
        (
            ~repair
        )
        .sum()
    )

    # --------------------------------------------------------
    # Every auto-repair MUST be a strict delisting boundary.
    # --------------------------------------------------------

    repair_rows = audit[
        repair
    ].copy()

    qa[
        "all_repairs_are_delist_boundary_only"
    ] = bool(
        (
            repair_rows[
                "audit_classification"
            ]
            ==
            "DELIST_POST_BOUNDARY_ONLY"
        )
        .all()
    )

    qa[
        "all_repairs_have_delist_in_interval"
    ] = bool(
        repair_rows[
            "delist_in_holding_interval"
        ]
        .all()
    )

    qa[
        "all_repairs_have_zero_active_missing"
    ] = bool(
        (
            repair_rows[
                "active_missing_n"
            ]
            ==
            0
        )
        .all()
    )

    qa[
        "all_repairs_have_zero_active_unresolved"
    ] = bool(
        (
            repair_rows[
                "active_unresolved_n"
            ]
            ==
            0
        )
        .all()
    )

    qa[
        "all_repairs_have_no_post_delist_nonzero_return"
    ] = bool(
        (
            repair_rows[
                "post_delist_nonzero_return_n"
            ]
            ==
            0
        )
        .all()
    )

    qa[
        "all_repaired_returns_valid"
    ] = bool(
        (
            repair_rows[
                "audited_drift_total_return"
            ]
            .notna()
        )
        .all()

        and

        (
            np.isfinite(
                repair_rows[
                    "audited_drift_total_return"
                ]
            )
        )
        .all()

        and

        (
            repair_rows[
                "audited_drift_total_return"
            ]
            >=
            -1.0
            -
            TOL
        )
        .all()
    )

    # --------------------------------------------------------
    # Original PASS rows must NEVER be changed.
    # --------------------------------------------------------

    original_pass = (

        audited_drift[
            "status_original"
        ]
        ==
        "PASS"
    )

    qa[
        "original_pass_status_unchanged"
    ] = bool(
        (
            audited_drift.loc[
                original_pass,
                "status",
            ]
            ==
            "PASS"
        )
        .all()
    )

    original_return_diff = (

        audited_drift.loc[
            original_pass,
            "drift_total_return"
        ]

        -

        audited_drift.loc[
            original_pass,
            "drift_total_return_original"
        ]
    ).abs()

    qa[
        "max_original_pass_return_change"
    ] = float(
        original_return_diff.max()
        if len(
            original_return_diff
        ) > 0
        else 0.0
    )

    qa[
        "original_pass_return_unchanged"
    ] = bool(
        qa[
            "max_original_pass_return_change"
        ]
        <
        1e-14
    )

    qa[
        "audited_drift_keys_unique"
    ] = bool(
        not audited_drift[
            key
        ]
        .duplicated()
        .any()
    )

    valid_status = (
        audited_drift[
            "status"
        ]
        .isin(
            [
                "PASS",
                "PASS_DELIST_CASH_CARRY",
            ]
        )
    )

    qa[
        "audited_valid_return_count"
    ] = int(
        valid_status.sum()
    )

    qa[
        "audited_remaining_nonpass_count"
    ] = int(
        (
            ~valid_status
        )
        .sum()
    )

    qa[
        "valid_status_rows_have_return"
    ] = bool(
        audited_drift.loc[
            valid_status,
            "drift_total_return",
        ]
        .notna()
        .all()
    )

    qa[
        "true_source_gap_auto_filled"
    ] = bool(
        (
            audit[
                "audit_classification"
            ]
            .eq(
                "ACTIVE_PERIOD_TRUE_SOURCE_GAP"
            )
            &
            audit[
                "repair_eligible"
            ]
        )
        .any()
    )

    # This should be False.
    qa[
        "no_true_source_gap_auto_fill"
    ] = bool(
        not qa[
            "true_source_gap_auto_filled"
        ]
    )

    qa[
        "missing_record_zero_filled"
    ] = False

    qa[
        "pre_delist_gap_zero_filled"
    ] = False

    qa[
        "future_information_used"
    ] = False

    qa[
        "factor_selection"
    ] = False

    qa[
        "window_selection"
    ] = False

    required = [
        "audit_row_count_matches_issue_count",
        "audit_keys_unique",
        "classification_complete",
        "all_repairs_are_delist_boundary_only",
        "all_repairs_have_delist_in_interval",
        "all_repairs_have_zero_active_missing",
        "all_repairs_have_zero_active_unresolved",
        "all_repairs_have_no_post_delist_nonzero_return",
        "all_repaired_returns_valid",
        "original_pass_status_unchanged",
        "original_pass_return_unchanged",
        "audited_drift_keys_unique",
        "valid_status_rows_have_return",
        "no_true_source_gap_auto_fill",
    ]

    qa[
        "all_formal_qa_pass"
    ] = bool(
        all(
            qa[
                name
            ]
            for name in required
        )
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
        "Day 8 - Step 2C"
    )

    print(
        "Delisting / Missing-Record Audit"
    )

    print("=" * 80)

    # ========================================================
    # Upstream QA
    # ========================================================

    print()
    print(
        "[1] Validate Step-2B upstream QA"
    )

    validate_step2b_qa()

    # ========================================================
    # Inputs
    # ========================================================

    print()
    print(
        "[2] Load Step-2B missing-record issues"
    )

    issue = load_issue_rows()

    print(
        f"Issue rows: "
        f"{len(issue):,}"
    )

    issue_ids = set(
        issue[
            "security_id"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    # ========================================================
    # Calendar
    # ========================================================

    print()
    print(
        "[3] Load trading calendar"
    )

    calendar = (
        load_trade_calendar()
    )

    # ========================================================
    # Master
    # ========================================================

    print()
    print(
        "[4] Load stock master"
    )

    (
        master,
        master_path,
    ) = load_stock_master(
        issue_ids
    )

    print(
        f"Issue securities found in master: "
        f"{master['security_id'].nunique():,}"
        f"/{len(issue_ids):,}"
    )

    # ========================================================
    # Daily returns
    # ========================================================

    print()
    print(
        "[5] Load Day-1 returns for issue securities"
    )

    daily = load_issue_daily_returns(
        issue
    )

    print(
        f"Relevant daily rows: "
        f"{len(daily):,}"
    )

    # ========================================================
    # Audit
    # ========================================================

    print()
    print(
        "[6] Audit delisting / source-record structure"
    )

    (
        audit,
        daily_audit,
    ) = audit_issues(
        issue,
        master,
        daily,
        calendar,
    )

    save_csv_atomic(
        audit,
        AUDIT_DETAIL_PATH,
    )

    save_csv_atomic(
        daily_audit,
        DAILY_AUDIT_DETAIL_PATH,
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = (
        build_audit_summary(
            audit
        )
    )

    save_csv_atomic(
        summary,
        AUDIT_SUMMARY_PATH,
    )

    print()
    print(
        "Audit classification:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # Build audited drift panel
    # ========================================================

    print()
    print(
        "[7] Build audited dedicated-drift panel"
    )

    audited_drift = (
        build_audited_drift_panel(
            audit
        )
    )

    save_parquet_atomic(
        audited_drift,
        AUDITED_DRIFT_PATH,
    )

    # ========================================================
    # Formal QA
    # ========================================================

    print()
    print(
        "[8] Formal QA"
    )

    qa = formal_qa(
        issue,
        audit,
        audited_drift,
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

    if not qa[
        "all_formal_qa_pass"
    ]:

        raise RuntimeError(
            "Step-2C formal QA failed.\n"
            f"{qa}"
        )

    # ========================================================
    # Metadata
    # ========================================================

    design = {

        "research_day":
            8,

        "step":
            "Step2C_Delisting_Missing_Record_Audit",

        "stock_active_rule":
            "list_date <= t < delist_date",

        "step2b_issue_source":
            str(
                STEP2B_ISSUE_PATH
            ),

        "stock_master_source":
            str(
                master_path
            ),

        "day1_return_source":
            str(
                DAY1_RETURN_PANEL_PATH
            ),

        "repair_rule":
            (
                "Auto repair only when delist_date lies in "
                "(previous_analysis_date, analysis_date], "
                "all pre-delist market days are observed and "
                "resolved, and no non-zero holding return is "
                "observed on/after delist_date."
            ),

        "repair_accounting":
            (
                "Compound pre-delist holding returns, then "
                "carry resulting wealth as cash with zero return "
                "through the current rebalance date."
            ),

        "apply_repairs":
            APPLY_DELIST_CASH_CARRY_REPAIRS,

        "missing_source_gap_zero_fill":
            False,

        "pre_delist_gap_zero_fill":
            False,

        "future_information_used":
            False,

        "factor_selection":
            False,

        "window_selection":
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

                "audit_detail":
                    str(
                        AUDIT_DETAIL_PATH
                    ),

                "daily_audit_detail":
                    str(
                        DAILY_AUDIT_DETAIL_PATH
                    ),

                "audit_summary":
                    str(
                        AUDIT_SUMMARY_PATH
                    ),

                "audited_drift_panel":
                    str(
                        AUDITED_DRIFT_PATH
                    ),

                "formal_qa":
                    str(
                        FORMAL_QA_PATH
                    ),
            },

        "important_note":
            (
                "DELIST_CASH_CARRY is a project portfolio-"
                "accounting convention based on the PIT stock-"
                "master active interval. It should not be "
                "interpreted as proof of legal liquidation "
                "proceeds or exact delisting settlement value."
            ),
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
        "Step-2C Summary"
    )

    print("=" * 80)

    print(
        f"Original issue rows: "
        f"{qa['input_issue_count']:,}"
    )

    print(
        f"Repair eligible: "
        f"{qa['repair_eligible_count']:,}"
    )

    print(
        f"Remaining unresolved: "
        f"{qa['remaining_unrepaired_issue_count']:,}"
    )

    print(
        f"Audited drift remaining non-PASS: "
        f"{qa['audited_remaining_nonpass_count']:,}"
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
        "Day 8 Step 2C Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()
