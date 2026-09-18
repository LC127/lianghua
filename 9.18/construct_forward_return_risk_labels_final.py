from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 0. Paths
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)


# ------------------------------------------------------------
# Day 5 final factor panel
#
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
# Day 6 Step 1 frozen factor design
#
# Research Day 6 is stored under M1_day5
# ------------------------------------------------------------

DAY6_ROOT = (
    OUTPUT_ROOT
    / "M1_day6"
)

DAY6_STEP1_DIR = (
    DAY6_ROOT
    / "01_step1_freeze_reduced_factor_set"
)

DAY6_DESIGN_PATH = (
    DAY6_STEP1_DIR
    / "day6_factor_design.json"
)


# ------------------------------------------------------------
# Day 1 validated return panel
# ------------------------------------------------------------

DAY1_STEP3_DIR = (
    OUTPUT_ROOT
    / "M1_day1"
    / "03_stage3_return_validation"
)

RETURN_PANEL_PATH = (
    DAY1_STEP3_DIR
    / "daily_return_panel.parquet"
)

DAY1_METADATA_PATH = (
    DAY1_STEP3_DIR
    / "stage3_metadata.json"
)


# ------------------------------------------------------------
# Trading calendar
# ------------------------------------------------------------

PRIMARY_CALENDAR_PATH = Path(
    r"D:\lowfreq\trade_calendar\trade_date.csv"
)

ALTERNATIVE_CALENDAR_PATHS = [

    Path(
        r"D:\M1_StockNetwork"
        r"\trade_calendar"
        r"\trade_date.csv"
    ),

    Path(
        r"D:\M1_StockNetwork\data"
        r"\trade_calendar"
        r"\trade_date.csv"
    ),
]


# ------------------------------------------------------------
# Day 6 Step 2 output
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY6_ROOT
    / "02_step2_forward_labels"
)

PARTITION_DIR = (
    OUTPUT_DIR
    / "panel_parts"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARTITION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FINAL_PANEL_PATH = (
    OUTPUT_DIR
    / "forward_label_panel.parquet"
)

INTERVAL_PATH = (
    OUTPUT_DIR
    / "forward_interval_table.csv"
)

JOB_QA_PATH = (
    OUTPUT_DIR
    / "forward_label_job_qa.csv"
)

COVERAGE_PATH = (
    OUTPUT_DIR
    / "forward_label_coverage_summary.csv"
)

DISTRIBUTION_PATH = (
    OUTPUT_DIR
    / "forward_label_distribution_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "step2_forward_label_metadata.json"
)

LABEL_DESIGN_PATH = (
    OUTPUT_DIR
    / "step2_label_design.json"
)


# ============================================================
# 1. Label Version
# ============================================================

LABEL_VERSION = "holding_return_v2"


# ============================================================
# 2. Formal Label Design
# ============================================================

ANNUALIZATION_DAYS = 252

MIN_RETURN_COVERAGE = 0.80

MIN_VALID_RETURN_DAYS = 10

MIN_VALID_RISK_DAYS = 10

TAIL_QUANTILE = 0.05

MIN_MARKET_STOCKS_PER_DAY = 100


# ------------------------------------------------------------
# Conservative data-quality rule
#
# isOpen=True but return_last_trade_simple is missing:
# do not silently treat it as zero.
#
# Such a stock-period cannot be a formal label if any such
# unresolved open-day missing return exists.
# ------------------------------------------------------------

REQUIRE_NO_UNRESOLVED_OPEN_MISSING = True


# ------------------------------------------------------------
# Existing Step-2 files are backed up before overwrite.
# ------------------------------------------------------------

BACKUP_EXISTING_OUTPUTS = True

SAVE_PARTITIONS = True

STRICT_QA = True


# ============================================================
# 3. Formal Labels
# ============================================================

RETURN_LABELS = [

    "future_total_return",

    "future_loo_market_return",

    "future_excess_return",

    "future_relative_return",
]


RISK_LABELS = [

    "future_realized_vol_annualized",

    "future_downside_vol_annualized",

    "future_max_drawdown",

    "future_idio_vol_annualized",

    "future_worst_daily_return",

    "future_var_5pct_daily",
]


ALL_LABELS = (
    RETURN_LABELS
    +
    RISK_LABELS
)


# ============================================================
# 4. Day-1 source columns
# ============================================================

SOURCE_COLUMNS = [

    "security_id",

    "trade_date",

    "is_open",

    "is_suspended",

    "return_last_trade_simple",

    "is_reopen_day",

    "days_since_last_trade",
]


# ============================================================
# 5. Utilities
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

    temp_path = Path(
        str(path)
        +
        ".tmp"
    )

    with open(
        temp_path,
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
        temp_path,
        path,
    )


def canonical_json_hash(
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


def resolve_path(
    primary: Path,
    alternatives: list[Path],
    label: str,
):

    if primary.exists():

        return primary

    existing = [
        x
        for x in alternatives
        if x.exists()
    ]

    if len(existing) == 1:

        return existing[0]

    if len(existing) == 0:

        raise FileNotFoundError(
            f"{label} not found.\n"
            f"Primary:\n{primary}\n\n"
            f"Alternatives:\n"
            +
            "\n".join(
                str(x)
                for x in alternatives
            )
        )

    raise RuntimeError(
        f"Multiple candidate {label} files found:\n"
        +
        "\n".join(
            str(x)
            for x in existing
        )
        +
        "\nPlease explicitly set PRIMARY path."
    )


# ============================================================
# 6. Backup Existing Formal Step-2 Outputs
# ============================================================

def backup_existing_outputs():

    if not BACKUP_EXISTING_OUTPUTS:

        return None

    formal_files = [

        FINAL_PANEL_PATH,

        INTERVAL_PATH,

        JOB_QA_PATH,

        COVERAGE_PATH,

        DISTRIBUTION_PATH,

        METADATA_PATH,

        LABEL_DESIGN_PATH,
    ]

    existing = [
        path
        for path in formal_files
        if path.exists()
    ]

    if len(existing) == 0:

        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    archive_dir = (
        OUTPUT_DIR
        /
        f"archive_before_{LABEL_VERSION}_{timestamp}"
    )

    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in existing:

        shutil.copy2(
            path,
            archive_dir
            / path.name,
        )

    print(
        f"Old Step-2 outputs backed up to:\n"
        f"{archive_dir}"
    )

    return archive_dir


# ============================================================
# 7. Validate Day-6 Frozen Factor Design
# ============================================================

def load_day6_design():

    design = load_json(
        DAY6_DESIGN_PATH
    )

    if (
        design.get(
            "design_status"
        )
        !=
        "FROZEN"
    ):

        raise RuntimeError(
            "Day-6 factor design is not FROZEN."
        )

    design_hash = (
        design.get(
            "day6_design_hash"
        )
    )

    if not design_hash:

        raise RuntimeError(
            "Missing day6_design_hash."
        )

    factor_sets = (
        design.get(
            "factor_sets",
            {},
        )
    )

    if (
        int(
            factor_sets.get(
                "core_factor_count",
                -1,
            )
        )
        !=
        9
    ):

        raise RuntimeError(
            "Expected exactly 9 frozen core factors."
        )

    return (
        design,
        design_hash,
    )


# ============================================================
# 8. Validate Day-1 Return Design
# ============================================================

def validate_day1_return_design():

    if not RETURN_PANEL_PATH.exists():

        raise FileNotFoundError(
            f"Day-1 return panel missing:\n"
            f"{RETURN_PANEL_PATH}"
        )

    parquet_file = pq.ParquetFile(
        RETURN_PANEL_PATH
    )

    columns = set(
        parquet_file
        .schema_arrow
        .names
    )

    missing = [
        col
        for col in SOURCE_COLUMNS
        if col not in columns
    ]

    if missing:

        raise RuntimeError(
            "Day-1 return panel does not contain "
            "required holding-return fields:\n"
            f"{missing}"
        )

    metadata = {}

    if DAY1_METADATA_PATH.exists():

        metadata = load_json(
            DAY1_METADATA_PATH
        )

    print()
    print(
        "Day-1 return design:"
    )

    print(
        "  Network return:"
    )

    print(
        "    consecutive-open market-day return"
    )

    print(
        "  Forward-label source:"
    )

    print(
        "    return_last_trade_simple"
    )

    print(
        "  Suspension treatment:"
    )

    print(
        "    is_open=False -> holding_return=0"
    )

    print(
        "  Reopen treatment:"
    )

    print(
        "    use return_last_trade_simple"
    )

    return metadata


# ============================================================
# 9. Step-2 Label Design Hash
# ============================================================

def build_label_design(
    day6_design_hash,
):

    payload = {

        "label_version":
            LABEL_VERSION,

        "parent_day6_factor_design_hash":
            day6_design_hash,

        "source_panel":
            str(
                RETURN_PANEL_PATH
            ),

        "source_return_column":
            "return_last_trade_simple",

        "source_return_type":
            "simple",

        "holding_return_definition": {

            "suspended_day":
                (
                    "holding_return = 0 "
                    "when is_open == False"
                ),

            "active_day":
                (
                    "holding_return = "
                    "return_last_trade_simple "
                    "when is_open == True "
                    "and return_last_trade_simple "
                    "is observed"
                ),

            "active_missing":
                (
                    "holding_return = NA "
                    "when is_open == True "
                    "but return_last_trade_simple "
                    "is missing"
                ),

            "reopen_day":
                (
                    "use return_last_trade_simple; "
                    "this preserves the cumulative "
                    "adjusted-price change since the "
                    "previous actual trading day"
                ),
        },

        "forward_interval":
            "(analysis_date, next_analysis_date]",

        "market_benchmark":
            (
                "daily leave-one-out equal-weight "
                "full-market holding return"
            ),

        "annualization_days":
            ANNUALIZATION_DAYS,

        "minimum_return_coverage":
            MIN_RETURN_COVERAGE,

        "minimum_valid_return_days":
            MIN_VALID_RETURN_DAYS,

        "minimum_valid_risk_days":
            MIN_VALID_RISK_DAYS,

        "require_no_unresolved_open_missing":
            REQUIRE_NO_UNRESOLVED_OPEN_MISSING,

        "tail_quantile":
            TAIL_QUANTILE,
    }

    design_hash = canonical_json_hash(
        payload
    )

    final = {
        **payload,

        "step2_label_design_hash":
            design_hash,
    }

    return (
        final,
        design_hash,
    )


# ============================================================
# 10. Load Factor Universe
# ============================================================

def load_factor_universe():

    if not FACTOR_PANEL_PATH.exists():

        raise FileNotFoundError(
            f"Factor panel missing:\n"
            f"{FACTOR_PANEL_PATH}"
        )

    pf = pq.ParquetFile(
        FACTOR_PANEL_PATH
    )

    columns = (
        pf
        .schema_arrow
        .names
    )

    required = [

        "analysis_date",

        "window",

        "master_index",

        "security_id",
    ]

    missing = [
        col
        for col in required
        if col not in columns
    ]

    if missing:

        raise RuntimeError(
            "Factor panel missing key columns:\n"
            f"{missing}"
        )

    optional = [

        "stock_code",

        "stock_name",

        "industry_id1",

        "industry_id2",
    ]

    use_columns = (
        required
        +
        [
            col
            for col in optional
            if col in columns
        ]
    )

    df = pd.read_parquet(
        FACTOR_PANEL_PATH,
        columns=use_columns,
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

    key = [
        "analysis_date",
        "window",
        "master_index",
    ]

    if (
        df[
            key
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate factor-panel key found."
        )

    return df


# ============================================================
# 11. Derive Frozen Analysis-Date Schedule
# ============================================================

def build_analysis_schedule(
    factor_universe,
):

    schedule = (
        factor_universe[
            [
                "analysis_date",
                "window",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    schedule[
        "next_analysis_date"
    ] = (
        schedule.groupby(
            "window"
        )[
            "analysis_date"
        ]
        .shift(-1)
    )

    if (
        len(
            schedule
        )
        !=
        402
    ):

        raise RuntimeError(
            "Unexpected Date x Window job count.\n"
            f"Expected 402, got {len(schedule)}."
        )

    return schedule


# ============================================================
# 12. Trading Calendar
# ============================================================

def load_trade_calendar():

    path = resolve_path(

        PRIMARY_CALENDAR_PATH,

        ALTERNATIVE_CALENDAR_PATHS,

        "trading calendar",
    )

    calendar = pd.read_csv(
        path
    )

    date_candidates = [

        "trade_date",

        "date",

        "calendarDate",

        "tradeDate",
    ]

    matches = [
        col
        for col in date_candidates
        if col in calendar.columns
    ]

    if len(matches) != 1:

        raise RuntimeError(
            "Cannot uniquely identify calendar date column.\n"
            f"Candidates found: {matches}"
        )

    date_col = matches[0]

    if "isOpen" in calendar.columns:

        calendar = (
            calendar[
                pd.to_numeric(
                    calendar[
                        "isOpen"
                    ],
                    errors="coerce",
                )
                ==
                1
            ]
        )

    dates = pd.to_datetime(
        calendar[
            date_col
        ],
        errors="raise",
    )

    dates = (
        pd.DatetimeIndex(
            dates
        )
        .drop_duplicates()
        .sort_values()
    )

    return (
        dates,
        path,
    )


# ============================================================
# 13. Load Day-1 Panel and Construct Holding Return
# ============================================================

def load_holding_return_panel():

    print()
    print(
        "Loading Day-1 validated return panel..."
    )

    df = pd.read_parquet(
        RETURN_PANEL_PATH,
        columns=SOURCE_COLUMNS,
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

    df[
        "trade_date"
    ] = pd.to_datetime(
        df[
            "trade_date"
        ],
        errors="raise",
    )

    df[
        "return_last_trade_simple"
    ] = pd.to_numeric(
        df[
            "return_last_trade_simple"
        ],
        errors="coerce",
    )

    df[
        "days_since_last_trade"
    ] = pd.to_numeric(
        df[
            "days_since_last_trade"
        ],
        errors="coerce",
    )

    df[
        "is_open"
    ] = (
        df[
            "is_open"
        ]
        .astype("boolean")
    )

    df[
        "is_suspended"
    ] = (
        df[
            "is_suspended"
        ]
        .astype("boolean")
    )

    df[
        "is_reopen_day"
    ] = (
        df[
            "is_reopen_day"
        ]
        .astype("boolean")
    )

    if (
        df[
            [
                "security_id",
                "trade_date",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate security-date rows "
            "in Day-1 return panel."
        )

    # ========================================================
    # Holding-period return
    # ========================================================

    df[
        "holding_return"
    ] = np.nan

    df[
        "holding_return_source"
    ] = "UNRESOLVED"

    # --------------------------------------------------------
    # Suspension:
    #
    # investor's marked value remains unchanged
    # until a new actual trading price appears.
    # --------------------------------------------------------

    suspension_mask = (
        df[
            "is_open"
        ]
        .eq(
            False
        )
        .fillna(
            False
        )
    )

    df.loc[
        suspension_mask,
        "holding_return",
    ] = 0.0

    df.loc[
        suspension_mask,
        "holding_return_source",
    ] = "SUSPENSION_ZERO"

    # --------------------------------------------------------
    # Actual trading day:
    #
    # use price change from previous actual trading day.
    #
    # For normal consecutive trading days this equals the
    # ordinary daily adjusted-price return.
    #
    # For reopen days it captures the entire price jump from
    # the last actual trading price.
    # --------------------------------------------------------

    active_valid_mask = (

        df[
            "is_open"
        ]
        .eq(
            True
        )
        .fillna(
            False
        )

        &

        df[
            "return_last_trade_simple"
        ]
        .notna()
    )

    df.loc[
        active_valid_mask,
        "holding_return",
    ] = (
        df.loc[
            active_valid_mask,
            "return_last_trade_simple",
        ]
    )

    normal_active_mask = (

        active_valid_mask

        &

        ~df[
            "is_reopen_day"
        ]
        .eq(
            True
        )
        .fillna(
            False
        )
    )

    reopen_valid_mask = (

        active_valid_mask

        &

        df[
            "is_reopen_day"
        ]
        .eq(
            True
        )
        .fillna(
            False
        )
    )

    df.loc[
        normal_active_mask,
        "holding_return_source",
    ] = "ACTIVE_LAST_TRADE"

    df.loc[
        reopen_valid_mask,
        "holding_return_source",
    ] = "REOPEN_LAST_TRADE"

    # --------------------------------------------------------
    # Explicit unresolved open-day missing
    # --------------------------------------------------------

    df[
        "unresolved_open_missing"
    ] = (

        df[
            "is_open"
        ]
        .eq(
            True
        )
        .fillna(
            False
        )

        &

        df[
            "holding_return"
        ]
        .isna()
    )

    df[
        "unresolved_reopen_missing"
    ] = (

        df[
            "is_reopen_day"
        ]
        .eq(
            True
        )
        .fillna(
            False
        )

        &

        df[
            "holding_return"
        ]
        .isna()
    )

    # --------------------------------------------------------
    # Basic mathematical QA
    # --------------------------------------------------------

    invalid_return = (

        df[
            "holding_return"
        ]
        .notna()

        &

        (
            df[
                "holding_return"
            ]
            <
            -1.0
            -
            1e-12
        )
    )

    if invalid_return.any():

        bad = (
            df.loc[
                invalid_return,
                [
                    "security_id",
                    "trade_date",
                    "is_open",
                    "is_reopen_day",
                    "return_last_trade_simple",
                    "holding_return",
                ],
            ]
            .head(
                20
            )
        )

        raise RuntimeError(
            "Holding simple return below -100%.\n"
            f"{bad}"
        )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    source_counts = (
        df[
            "holding_return_source"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()
    print(
        "Holding-return source counts:"
    )

    print(
        source_counts.to_string()
    )

    print()

    print(
        "Unresolved open-day missing:"
    )

    print(
        int(
            df[
                "unresolved_open_missing"
            ]
            .sum()
        )
    )

    print(
        "Unresolved reopen-day missing:"
    )

    print(
        int(
            df[
                "unresolved_reopen_missing"
            ]
            .sum()
        )
    )

    df = (
        df.sort_values(
            [
                "trade_date",
                "security_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# 14. Fast Date Slicing
# ============================================================

def build_date_index(
    holding_panel,
):

    return (
        holding_panel[
            "trade_date"
        ]
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )


def slice_date_range(
    holding_panel,
    date_values,
    start_date,
    end_date,
):

    start64 = np.datetime64(
        pd.Timestamp(
            start_date
        )
    )

    end64 = np.datetime64(
        pd.Timestamp(
            end_date
        )
    )

    left = np.searchsorted(
        date_values,
        start64,
        side="left",
    )

    right = np.searchsorted(
        date_values,
        end64,
        side="right",
    )

    return (
        holding_panel
        .iloc[
            left:right
        ]
        .copy()
    )


# ============================================================
# 15. Expected Forward Trading Dates
# ============================================================

def get_expected_trade_dates(
    calendar_dates,
    analysis_date,
    next_analysis_date,
):

    return calendar_dates[
        (
            calendar_dates
            >
            pd.Timestamp(
                analysis_date
            )
        )
        &
        (
            calendar_dates
            <=
            pd.Timestamp(
                next_analysis_date
            )
        )
    ]


# ============================================================
# 16. Empty Formal Label Panel
# ============================================================

def initialize_label_columns(
    base,
):

    result = base.copy()

    result[
        "label_version"
    ] = LABEL_VERSION

    result[
        "has_forward_horizon"
    ] = False

    result[
        "next_analysis_date"
    ] = pd.NaT

    result[
        "future_period_start"
    ] = pd.NaT

    result[
        "future_period_end"
    ] = pd.NaT

    result[
        "future_expected_trade_days"
    ] = 0

    # --------------------------------------------------------
    # Holding-path diagnostics
    # --------------------------------------------------------

    result[
        "future_panel_record_days"
    ] = 0

    result[
        "future_no_panel_record_days"
    ] = 0

    result[
        "future_open_days"
    ] = 0

    result[
        "future_suspended_days"
    ] = 0

    result[
        "future_reopen_days"
    ] = 0

    result[
        "future_reopen_return_days"
    ] = 0

    result[
        "future_unresolved_open_missing_days"
    ] = 0

    result[
        "future_unresolved_reopen_missing_days"
    ] = 0

    result[
        "future_valid_return_days"
    ] = 0

    result[
        "future_return_coverage"
    ] = np.nan

    result[
        "future_return_label_valid"
    ] = False

    result[
        "future_risk_label_valid"
    ] = False

    for label in ALL_LABELS:

        result[
            label
        ] = np.nan

    return result


# ============================================================
# 17. Build One Date x Window Job
# ============================================================

def build_labels_one_job(
    base,
    holding_panel,
    date_values,
    calendar_dates,
    analysis_date,
    next_analysis_date,
):

    result = initialize_label_columns(
        base
    )

    # --------------------------------------------------------
    # Last frozen analysis date:
    # no future t+1 date -> no labels.
    # --------------------------------------------------------

    if pd.isna(
        next_analysis_date
    ):

        diagnostics = {

            "expected_trade_days":
                0,

            "market_trade_day_count":
                0,

            "market_date_coverage":
                np.nan,

            "min_market_stock_count":
                np.nan,

            "suspension_zero_stock_days":
                0,

            "reopen_stock_days":
                0,

            "reopen_return_recovered_stock_days":
                0,

            "unresolved_open_missing_stock_days":
                0,

            "unresolved_reopen_missing_stock_days":
                0,
        }

        return (
            result,
            diagnostics,
        )

    analysis_date = pd.Timestamp(
        analysis_date
    )

    next_analysis_date = pd.Timestamp(
        next_analysis_date
    )

    if (
        next_analysis_date
        <=
        analysis_date
    ):

        raise RuntimeError(
            "next_analysis_date must be "
            "strictly after analysis_date."
        )

    expected_dates = (
        get_expected_trade_dates(

            calendar_dates,

            analysis_date,

            next_analysis_date,
        )
    )

    expected_days = len(
        expected_dates
    )

    if expected_days <= 0:

        raise RuntimeError(
            "No trading day found in "
            "forward interval."
        )

    start_date = (
        expected_dates[
            0
        ]
    )

    end_date = (
        expected_dates[
            -1
        ]
    )

    if (
        start_date
        <=
        analysis_date
    ):

        raise RuntimeError(
            "Future period starts on or before "
            "analysis_date."
        )

    if (
        end_date
        !=
        next_analysis_date
    ):

        raise RuntimeError(
            "next_analysis_date does not agree "
            "with trading calendar.\n"
            f"analysis_date={analysis_date}\n"
            f"next_analysis_date={next_analysis_date}\n"
            f"calendar_end={end_date}"
        )

    # ========================================================
    # Full-market holding-return interval
    # ========================================================

    interval_all = (
        slice_date_range(

            holding_panel,

            date_values,

            start_date,

            end_date,
        )
    )

    # ========================================================
    # Daily full-market EW holding return
    #
    # Suspended stocks are explicitly included with return=0.
    #
    # Unresolved active missing returns are not silently
    # included.
    # ========================================================

    market_stats = (
        interval_all.groupby(
            "trade_date"
        )[
            "holding_return"
        ]
        .agg(
            market_sum="sum",
            market_count="count",
        )
        .reset_index()
    )

    market_trade_day_count = int(
        market_stats[
            "trade_date"
        ]
        .nunique()
    )

    market_date_coverage = (

        market_trade_day_count

        /

        expected_days
    )

    min_market_stock_count = int(
        market_stats[
            "market_count"
        ]
        .min()
    )

    # ========================================================
    # Only now restrict to current t-factor universe
    # ========================================================

    current_ids = set(
        result[
            "security_id"
        ]
        .astype(str)
        .tolist()
    )

    work = (
        interval_all[
            interval_all[
                "security_id"
            ]
            .astype(str)
            .isin(
                current_ids
            )
        ]
        .copy()
    )

    work = work.merge(

        market_stats,

        on="trade_date",

        how="left",

        validate="many_to_one",
    )

    # ========================================================
    # Leave-one-out EW holding-market return
    #
    # Only defined on stock-days with an observed holding
    # return, so stock and benchmark are compounded over the
    # same valid days.
    # ========================================================

    valid_loo = (

        work[
            "holding_return"
        ]
        .notna()

        &

        (
            work[
                "market_count"
            ]
            >
            1
        )
    )

    work[
        "loo_market_return"
    ] = np.nan

    work.loc[
        valid_loo,
        "loo_market_return"
    ] = (

        (
            work.loc[
                valid_loo,
                "market_sum"
            ]

            -

            work.loc[
                valid_loo,
                "holding_return"
            ]
        )

        /

        (
            work.loc[
                valid_loo,
                "market_count"
            ]

            -
            1
        )
    )

    work = (
        work.sort_values(
            [
                "security_id",
                "trade_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Daily path variables
    # ========================================================

    stock_return = (
        work[
            "holding_return"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    market_return = (
        work[
            "loo_market_return"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        work[
            "_stock_log_gross"
        ] = np.log1p(
            stock_return
        )

        work[
            "_market_log_gross"
        ] = np.log1p(
            market_return
        )

    work[
        "_active_return"
    ] = (

        work[
            "holding_return"
        ]

        -

        work[
            "loo_market_return"
        ]
    )

    work[
        "_downside_sq"
    ] = (

        np.minimum(
            stock_return,
            0.0,
        )

        ** 2
    )

    work[
        "_gross"
    ] = (

        1.0

        +

        work[
            "holding_return"
        ]
    )

    # ========================================================
    # Holding-period wealth path / MDD
    #
    # Suspended days have gross return = 1.
    # Reopen-day jump enters on the first actual trading day.
    # ========================================================

    work[
        "_wealth"
    ] = (

        work.groupby(
            "security_id",
            sort=False,
        )[
            "_gross"
        ]
        .cumprod()
    )

    work[
        "_running_peak"
    ] = (

        work.groupby(
            "security_id",
            sort=False,
        )[
            "_wealth"
        ]
        .cummax()
    )

    work[
        "_running_peak"
    ] = np.maximum(

        work[
            "_running_peak"
        ]
        .to_numpy(
            dtype=np.float64
        ),

        1.0,
    )

    work[
        "_drawdown"
    ] = (

        1.0

        -

        (
            work[
                "_wealth"
            ]

            /

            work[
                "_running_peak"
            ]
        )
    )

    # ========================================================
    # Stock-level group
    # ========================================================

    group = work.groupby(
        "security_id",
        sort=False,
    )

    # --------------------------------------------------------
    # Path diagnostics
    # --------------------------------------------------------

    panel_record_days = (
        group[
            "trade_date"
        ]
        .count()
    )

    open_days = (
        work.assign(
            _open_flag=(
                work[
                    "is_open"
                ]
                .eq(
                    True
                )
                .fillna(
                    False
                )
                .astype(int)
            )
        )
        .groupby(
            "security_id"
        )[
            "_open_flag"
        ]
        .sum()
    )

    suspended_days = (
        work.assign(
            _suspend_flag=(
                work[
                    "is_open"
                ]
                .eq(
                    False
                )
                .fillna(
                    False
                )
                .astype(int)
            )
        )
        .groupby(
            "security_id"
        )[
            "_suspend_flag"
        ]
        .sum()
    )

    reopen_days = (
        work.assign(
            _reopen_flag=(
                work[
                    "is_reopen_day"
                ]
                .eq(
                    True
                )
                .fillna(
                    False
                )
                .astype(int)
            )
        )
        .groupby(
            "security_id"
        )[
            "_reopen_flag"
        ]
        .sum()
    )

    reopen_return_days = (
        work.assign(
            _reopen_return_flag=(
                (
                    work[
                        "is_reopen_day"
                    ]
                    .eq(
                        True
                    )
                    .fillna(
                        False
                    )
                )
                &
                (
                    work[
                        "holding_return"
                    ]
                    .notna()
                )
            ).astype(int)
        )
        .groupby(
            "security_id"
        )[
            "_reopen_return_flag"
        ]
        .sum()
    )

    unresolved_open_days = (
        group[
            "unresolved_open_missing"
        ]
        .sum()
    )

    unresolved_reopen_days = (
        group[
            "unresolved_reopen_missing"
        ]
        .sum()
    )

    n_valid = (
        group[
            "holding_return"
        ]
        .count()
    )

    # ========================================================
    # Return labels
    # ========================================================

    stock_log_sum = (
        group[
            "_stock_log_gross"
        ]
        .sum(
            min_count=1
        )
    )

    market_log_sum = (
        group[
            "_market_log_gross"
        ]
        .sum(
            min_count=1
        )
    )

    future_total_return = (
        np.expm1(
            stock_log_sum
        )
    )

    future_market_return = (
        np.expm1(
            market_log_sum
        )
    )

    future_excess_return = (

        future_total_return

        -

        future_market_return
    )

    future_relative_return = (

        (
            1.0
            +
            future_total_return
        )

        /

        (
            1.0
            +
            future_market_return
        )

        -
        1.0
    )

    # ========================================================
    # Risk labels
    # ========================================================

    realized_vol = (

        group[
            "holding_return"
        ]
        .std(
            ddof=1
        )

        *
        np.sqrt(
            ANNUALIZATION_DAYS
        )
    )

    downside_vol = (

        np.sqrt(
            group[
                "_downside_sq"
            ]
            .mean()
        )

        *
        np.sqrt(
            ANNUALIZATION_DAYS
        )
    )

    idio_vol = (

        group[
            "_active_return"
        ]
        .std(
            ddof=1
        )

        *
        np.sqrt(
            ANNUALIZATION_DAYS
        )
    )

    max_drawdown = (
        group[
            "_drawdown"
        ]
        .max()
    )

    worst_daily_return = (
        group[
            "holding_return"
        ]
        .min()
    )

    var_5pct = (
        group[
            "holding_return"
        ]
        .quantile(
            TAIL_QUANTILE
        )
    )

    # ========================================================
    # Stock-level label table
    # ========================================================

    labels = pd.DataFrame(
        {

            "security_id":
                n_valid.index.astype(
                    str
                ),

            "future_panel_record_days":
                panel_record_days.to_numpy(
                    dtype=np.int64
                ),

            "future_open_days":
                open_days.reindex(
                    n_valid.index,
                    fill_value=0,
                ).to_numpy(
                    dtype=np.int64
                ),

            "future_suspended_days":
                suspended_days.reindex(
                    n_valid.index,
                    fill_value=0,
                ).to_numpy(
                    dtype=np.int64
                ),

            "future_reopen_days":
                reopen_days.reindex(
                    n_valid.index,
                    fill_value=0,
                ).to_numpy(
                    dtype=np.int64
                ),

            "future_reopen_return_days":
                reopen_return_days.reindex(
                    n_valid.index,
                    fill_value=0,
                ).to_numpy(
                    dtype=np.int64
                ),

            "future_unresolved_open_missing_days":
                unresolved_open_days.reindex(
                    n_valid.index,
                    fill_value=0,
                ).to_numpy(
                    dtype=np.int64
                ),

            "future_unresolved_reopen_missing_days":
                unresolved_reopen_days.reindex(
                    n_valid.index,
                    fill_value=0,
                ).to_numpy(
                    dtype=np.int64
                ),

            "future_valid_return_days":
                n_valid.to_numpy(
                    dtype=np.int64
                ),

            "future_total_return":
                future_total_return.to_numpy(
                    dtype=np.float64
                ),

            "future_loo_market_return":
                future_market_return.to_numpy(
                    dtype=np.float64
                ),

            "future_excess_return":
                future_excess_return.to_numpy(
                    dtype=np.float64
                ),

            "future_relative_return":
                future_relative_return.to_numpy(
                    dtype=np.float64
                ),

            "future_realized_vol_annualized":
                realized_vol.to_numpy(
                    dtype=np.float64
                ),

            "future_downside_vol_annualized":
                downside_vol.to_numpy(
                    dtype=np.float64
                ),

            "future_max_drawdown":
                max_drawdown.to_numpy(
                    dtype=np.float64
                ),

            "future_idio_vol_annualized":
                idio_vol.to_numpy(
                    dtype=np.float64
                ),

            "future_worst_daily_return":
                worst_daily_return.to_numpy(
                    dtype=np.float64
                ),

            "future_var_5pct_daily":
                var_5pct.to_numpy(
                    dtype=np.float64
                ),
        }
    )

    # ========================================================
    # LEFT JOIN onto frozen factor universe
    # ========================================================

    diagnostic_columns = [

        "future_panel_record_days",

        "future_open_days",

        "future_suspended_days",

        "future_reopen_days",

        "future_reopen_return_days",

        "future_unresolved_open_missing_days",

        "future_unresolved_reopen_missing_days",

        "future_valid_return_days",
    ]

    # Replace initialized placeholders with the computed stock-level values.
    # Keeping ALL_LABELS here would create _x/_y columns during the merge,
    # leaving the canonical label columns missing when validity masks run.
    result = result.drop(
        columns=diagnostic_columns + ALL_LABELS,
    )

    result = result.merge(

        labels,

        on="security_id",

        how="left",

        validate="one_to_one",
    )

    for col in diagnostic_columns:

        result[
            col
        ] = (

            result[
                col
            ]
            .fillna(
                0
            )
            .astype(
                np.int64
            )
        )

    result[
        "future_no_panel_record_days"
    ] = (

        int(
            expected_days
        )

        -

        result[
            "future_panel_record_days"
        ]
    )

    result[
        "future_no_panel_record_days"
    ] = (

        result[
            "future_no_panel_record_days"
        ]
        .clip(
            lower=0
        )
        .astype(
            np.int64
        )
    )

    result[
        "has_forward_horizon"
    ] = True

    result[
        "next_analysis_date"
    ] = next_analysis_date

    result[
        "future_period_start"
    ] = start_date

    result[
        "future_period_end"
    ] = end_date

    result[
        "future_expected_trade_days"
    ] = int(
        expected_days
    )

    result[
        "future_return_coverage"
    ] = (

        result[
            "future_valid_return_days"
        ]

        /

        float(
            expected_days
        )
    )

    # ========================================================
    # Formal validity rules
    # ========================================================

    sufficient_return_days = (

        result[
            "future_valid_return_days"
        ]

        >=

        MIN_VALID_RETURN_DAYS
    )

    sufficient_risk_days = (

        result[
            "future_valid_return_days"
        ]

        >=

        MIN_VALID_RISK_DAYS
    )

    sufficient_coverage = (

        result[
            "future_return_coverage"
        ]

        >=

        MIN_RETURN_COVERAGE
    )

    if REQUIRE_NO_UNRESOLVED_OPEN_MISSING:

        no_unresolved_open_missing = (

            result[
                "future_unresolved_open_missing_days"
            ]

            ==
            0
        )

    else:

        no_unresolved_open_missing = (
            pd.Series(
                True,
                index=result.index,
            )
        )

    result[
        "future_return_label_valid"
    ] = (

        sufficient_return_days

        &

        sufficient_coverage

        &

        no_unresolved_open_missing
    )

    result[
        "future_risk_label_valid"
    ] = (

        sufficient_risk_days

        &

        sufficient_coverage

        &

        no_unresolved_open_missing
    )

    # ========================================================
    # Invalid formal labels -> NA
    #
    # Never drop the factor-universe row.
    # ========================================================

    invalid_return = (

        ~result[
            "future_return_label_valid"
        ]
    )

    result.loc[
        invalid_return,
        RETURN_LABELS,
    ] = np.nan

    invalid_risk = (

        ~result[
            "future_risk_label_valid"
        ]
    )

    result.loc[
        invalid_risk,
        RISK_LABELS,
    ] = np.nan

    # ========================================================
    # Job diagnostics
    # ========================================================

    suspension_zero_stock_days = int(
        (
            work[
                "holding_return_source"
            ]
            ==
            "SUSPENSION_ZERO"
        )
        .sum()
    )

    reopen_stock_days = int(
        work[
            "is_reopen_day"
        ]
        .eq(
            True
        )
        .fillna(
            False
        )
        .sum()
    )

    reopen_return_recovered_stock_days = int(
        (
            (
                work[
                    "is_reopen_day"
                ]
                .eq(
                    True
                )
                .fillna(
                    False
                )
            )
            &
            (
                work[
                    "holding_return"
                ]
                .notna()
            )
        )
        .sum()
    )

    unresolved_open_missing_stock_days = int(
        work[
            "unresolved_open_missing"
        ]
        .sum()
    )

    unresolved_reopen_missing_stock_days = int(
        work[
            "unresolved_reopen_missing"
        ]
        .sum()
    )

    diagnostics = {

        "expected_trade_days":
            int(
                expected_days
            ),

        "market_trade_day_count":
            int(
                market_trade_day_count
            ),

        "market_date_coverage":
            float(
                market_date_coverage
            ),

        "min_market_stock_count":
            int(
                min_market_stock_count
            ),

        "suspension_zero_stock_days":
            suspension_zero_stock_days,

        "reopen_stock_days":
            reopen_stock_days,

        "reopen_return_recovered_stock_days":
            reopen_return_recovered_stock_days,

        "unresolved_open_missing_stock_days":
            unresolved_open_missing_stock_days,

        "unresolved_reopen_missing_stock_days":
            unresolved_reopen_missing_stock_days,
    }

    return (
        result,
        diagnostics,
    )


# ============================================================
# 18. Formal QA for One Job
# ============================================================

def qa_one_job(
    panel,
    expected_rows,
    analysis_date,
    next_analysis_date,
    diagnostics,
):

    errors = []

    is_last_date = pd.isna(
        next_analysis_date
    )

    # ========================================================
    # Current-factor-universe preservation
    # ========================================================

    row_count_ok = (
        len(
            panel
        )
        ==
        int(
            expected_rows
        )
    )

    if not row_count_ok:

        errors.append(
            "row_count_mismatch"
        )

    key_unique = (

        not panel[
            [
                "analysis_date",
                "window",
                "master_index",
            ]
        ]
        .duplicated()
        .any()
    )

    if not key_unique:

        errors.append(
            "duplicate_key"
        )

    security_unique = (

        not panel[
            "security_id"
        ]
        .duplicated()
        .any()
    )

    if not security_unique:

        errors.append(
            "duplicate_security_id"
        )

    # ========================================================
    # Coverage
    # ========================================================

    coverage = (
        panel[
            "future_return_coverage"
        ]
        .dropna()
    )

    coverage_range_ok = bool(

        (
            coverage
            >=
            -1e-12
        ).all()

        and

        (
            coverage
            <=
            1.0
            +
            1e-12
        ).all()
    )

    if not coverage_range_ok:

        errors.append(
            "coverage_out_of_range"
        )

    # ========================================================
    # Last-date policy
    # ========================================================

    last_date_policy_ok = True

    if is_last_date:

        if (
            panel[
                "has_forward_horizon"
            ]
            .any()
        ):

            last_date_policy_ok = False

        if (
            panel[
                ALL_LABELS
            ]
            .notna()
            .any()
            .any()
        ):

            last_date_policy_ok = False

    if not last_date_policy_ok:

        errors.append(
            "last_date_policy_failure"
        )

    # ========================================================
    # Forward direction
    # ========================================================

    forward_direction_ok = True

    if not is_last_date:

        start_dates = (
            panel[
                "future_period_start"
            ]
            .dropna()
            .unique()
        )

        end_dates = (
            panel[
                "future_period_end"
            ]
            .dropna()
            .unique()
        )

        if (
            len(
                start_dates
            )
            !=
            1
            or
            len(
                end_dates
            )
            !=
            1
        ):

            forward_direction_ok = False

        else:

            start = pd.Timestamp(
                start_dates[
                    0
                ]
            )

            end = pd.Timestamp(
                end_dates[
                    0
                ]
            )

            if (
                start
                <=
                pd.Timestamp(
                    analysis_date
                )
            ):

                forward_direction_ok = False

            if (
                end
                !=
                pd.Timestamp(
                    next_analysis_date
                )
            ):

                forward_direction_ok = False

    if not forward_direction_ok:

        errors.append(
            "forward_direction_failure"
        )

    # ========================================================
    # Market benchmark QA
    # ========================================================

    market_calendar_ok = True

    market_count_ok = True

    if not is_last_date:

        market_calendar_ok = bool(
            np.isclose(
                diagnostics[
                    "market_date_coverage"
                ],
                1.0,
            )
        )

        market_count_ok = (

            diagnostics[
                "min_market_stock_count"
            ]

            >=

            MIN_MARKET_STOCKS_PER_DAY
        )

    if not market_calendar_ok:

        errors.append(
            "market_calendar_failure"
        )

    if not market_count_ok:

        errors.append(
            "market_stock_count_failure"
        )

    # ========================================================
    # Reopen recovery QA
    # ========================================================

    reopen_recovery_ok = True

    if not is_last_date:

        unresolved_reopen = (

            diagnostics[
                "unresolved_reopen_missing_stock_days"
            ]
        )

        reopen_recovery_ok = (
            unresolved_reopen
            ==
            0
        )

    if not reopen_recovery_ok:

        errors.append(
            "unresolved_reopen_return"
        )

    # ========================================================
    # Formal-label completeness
    # ========================================================

    valid_return = (
        panel[
            "future_return_label_valid"
        ]
    )

    return_label_complete = True

    if (
        valid_return.sum()
        >
        0
    ):

        return_label_complete = bool(

            panel.loc[
                valid_return,
                RETURN_LABELS,
            ]
            .notna()
            .all()
            .all()
        )

    if not return_label_complete:

        errors.append(
            "valid_return_label_missing"
        )

    valid_risk = (
        panel[
            "future_risk_label_valid"
        ]
    )

    risk_label_complete = True

    if (
        valid_risk.sum()
        >
        0
    ):

        risk_label_complete = bool(

            panel.loc[
                valid_risk,
                RISK_LABELS,
            ]
            .notna()
            .all()
            .all()
        )

    if not risk_label_complete:

        errors.append(
            "valid_risk_label_missing"
        )

    # ========================================================
    # Mathematical range QA
    # ========================================================

    vol_nonnegative = bool(

        (
            panel[
                "future_realized_vol_annualized"
            ]
            .dropna()
            >=
            -1e-12
        )
        .all()

        and

        (
            panel[
                "future_downside_vol_annualized"
            ]
            .dropna()
            >=
            -1e-12
        )
        .all()

        and

        (
            panel[
                "future_idio_vol_annualized"
            ]
            .dropna()
            >=
            -1e-12
        )
        .all()
    )

    if not vol_nonnegative:

        errors.append(
            "negative_volatility"
        )

    drawdown = (
        panel[
            "future_max_drawdown"
        ]
        .dropna()
    )

    drawdown_range_ok = bool(

        (
            drawdown
            >=
            -1e-12
        ).all()

        and

        (
            drawdown
            <=
            1.0
            +
            1e-10
        ).all()
    )

    if not drawdown_range_ok:

        errors.append(
            "drawdown_out_of_range"
        )

    total_return = (
        panel[
            "future_total_return"
        ]
        .dropna()
    )

    total_return_range_ok = bool(

        (
            total_return
            >=
            -1.0
            -
            1e-10
        ).all()
    )

    if not total_return_range_ok:

        errors.append(
            "return_below_minus_one"
        )

    qa_pass = (
        len(
            errors
        )
        ==
        0
    )

    return {

        "analysis_date":
            pd.Timestamp(
                analysis_date
            ),

        "window":
            int(
                panel[
                    "window"
                ]
                .iloc[
                    0
                ]
            ),

        "next_analysis_date":
            (
                pd.Timestamp(
                    next_analysis_date
                )
                if not is_last_date
                else pd.NaT
            ),

        "is_last_date":
            bool(
                is_last_date
            ),

        "expected_rows":
            int(
                expected_rows
            ),

        "actual_rows":
            int(
                len(
                    panel
                )
            ),

        "expected_trade_days":
            diagnostics[
                "expected_trade_days"
            ],

        "market_trade_day_count":
            diagnostics[
                "market_trade_day_count"
            ],

        "market_date_coverage":
            diagnostics[
                "market_date_coverage"
            ],

        "min_market_stock_count":
            diagnostics[
                "min_market_stock_count"
            ],

        "suspension_zero_stock_days":
            diagnostics[
                "suspension_zero_stock_days"
            ],

        "reopen_stock_days":
            diagnostics[
                "reopen_stock_days"
            ],

        "reopen_return_recovered_stock_days":
            diagnostics[
                "reopen_return_recovered_stock_days"
            ],

        "unresolved_open_missing_stock_days":
            diagnostics[
                "unresolved_open_missing_stock_days"
            ],

        "unresolved_reopen_missing_stock_days":
            diagnostics[
                "unresolved_reopen_missing_stock_days"
            ],

        "mean_return_coverage":
            (
                float(
                    panel[
                        "future_return_coverage"
                    ]
                    .mean()
                )
                if not is_last_date
                else np.nan
            ),

        "return_label_valid_share":
            (
                float(
                    panel[
                        "future_return_label_valid"
                    ]
                    .mean()
                )
                if not is_last_date
                else np.nan
            ),

        "risk_label_valid_share":
            (
                float(
                    panel[
                        "future_risk_label_valid"
                    ]
                    .mean()
                )
                if not is_last_date
                else np.nan
            ),

        "mean_suspended_days_per_stock":
            (
                float(
                    panel[
                        "future_suspended_days"
                    ]
                    .mean()
                )
                if not is_last_date
                else np.nan
            ),

        "mean_unresolved_open_missing_days":
            (
                float(
                    panel[
                        "future_unresolved_open_missing_days"
                    ]
                    .mean()
                )
                if not is_last_date
                else np.nan
            ),

        "row_count_ok":
            row_count_ok,

        "key_unique":
            key_unique,

        "security_unique":
            security_unique,

        "coverage_range_ok":
            coverage_range_ok,

        "last_date_policy_ok":
            last_date_policy_ok,

        "forward_direction_ok":
            forward_direction_ok,

        "market_calendar_ok":
            market_calendar_ok,

        "market_count_ok":
            market_count_ok,

        "reopen_recovery_ok":
            reopen_recovery_ok,

        "return_label_complete":
            return_label_complete,

        "risk_label_complete":
            risk_label_complete,

        "vol_nonnegative":
            vol_nonnegative,

        "drawdown_range_ok":
            drawdown_range_ok,

        "total_return_range_ok":
            total_return_range_ok,

        "qa_pass":
            qa_pass,

        "qa_errors":
            "|".join(
                errors
            ),
    }


# ============================================================
# 19. Save Partition
# ============================================================

def build_label_table(panel, schema, context=""):
    """Serialize every job with the frozen column order and safe Arrow types."""
    if not panel.columns.is_unique:
        raise RuntimeError(f"Duplicate forward-label columns. {context}")

    missing = sorted(set(schema.names) - set(panel.columns))
    unexpected = sorted(set(panel.columns) - set(schema.names))
    if missing or unexpected:
        # from_pandas(schema=...) can ignore extra columns; never allow that.
        raise RuntimeError(
            f"Forward-label columns changed. {context}\n"
            f"Missing: {missing}\nUnexpected: {unexpected}"
        )

    try:
        return pa.Table.from_pandas(
            panel,
            schema=schema,
            preserve_index=False,
            safe=True,
        )
    except (pa.ArrowException, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Cannot safely convert forward-label types. {context}\n{exc}"
        ) from exc


def save_partition(
    table,
    window,
    date,
):

    date_string = (
        pd.Timestamp(
            date
        )
        .strftime(
            "%Y-%m-%d"
        )
    )

    path = (
        PARTITION_DIR
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = Path(
        str(path)
        +
        ".tmp"
    )

    pq.write_table(
        table,
        temp,
        compression="zstd",
    )

    os.replace(
        temp,
        path,
    )


# ============================================================
# 20. Coverage Summary
# ============================================================

def build_coverage_summary(
    qa,
):

    rows = []

    for window, group in (
        qa.groupby(
            "window"
        )
    ):

        valid_horizon = (
            group[
                ~group[
                    "is_last_date"
                ]
            ]
        )

        rows.append(
            {

                "window":
                    int(
                        window
                    ),

                "job_count":
                    int(
                        len(
                            group
                        )
                    ),

                "forward_interval_count":
                    int(
                        len(
                            valid_horizon
                        )
                    ),

                "qa_pass_share":
                    float(
                        group[
                            "qa_pass"
                        ]
                        .mean()
                    ),

                "mean_return_coverage":
                    float(
                        valid_horizon[
                            "mean_return_coverage"
                        ]
                        .mean()
                    ),

                "min_return_coverage":
                    float(
                        valid_horizon[
                            "mean_return_coverage"
                        ]
                        .min()
                    ),

                "mean_return_label_valid_share":
                    float(
                        valid_horizon[
                            "return_label_valid_share"
                        ]
                        .mean()
                    ),

                "min_return_label_valid_share":
                    float(
                        valid_horizon[
                            "return_label_valid_share"
                        ]
                        .min()
                    ),

                "mean_risk_label_valid_share":
                    float(
                        valid_horizon[
                            "risk_label_valid_share"
                        ]
                        .mean()
                    ),

                "min_risk_label_valid_share":
                    float(
                        valid_horizon[
                            "risk_label_valid_share"
                        ]
                        .min()
                    ),

                "total_suspension_zero_stock_days":
                    int(
                        valid_horizon[
                            "suspension_zero_stock_days"
                        ]
                        .sum()
                    ),

                "total_reopen_stock_days":
                    int(
                        valid_horizon[
                            "reopen_stock_days"
                        ]
                        .sum()
                    ),

                "total_reopen_return_recovered_stock_days":
                    int(
                        valid_horizon[
                            "reopen_return_recovered_stock_days"
                        ]
                        .sum()
                    ),

                "total_unresolved_open_missing_stock_days":
                    int(
                        valid_horizon[
                            "unresolved_open_missing_stock_days"
                        ]
                        .sum()
                    ),

                "total_unresolved_reopen_missing_stock_days":
                    int(
                        valid_horizon[
                            "unresolved_reopen_missing_stock_days"
                        ]
                        .sum()
                    ),

                "min_market_date_coverage":
                    float(
                        valid_horizon[
                            "market_date_coverage"
                        ]
                        .min()
                    ),

                "min_market_stock_count":
                    int(
                        valid_horizon[
                            "min_market_stock_count"
                        ]
                        .min()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 21. Distribution Summary
# ============================================================

def build_distribution_summary(
    panel,
):

    rows = []

    for window in sorted(
        panel[
            "window"
        ]
        .unique()
    ):

        sub = (
            panel[
                panel[
                    "window"
                ]
                ==
                window
            ]
        )

        for label in ALL_LABELS:

            x = (
                pd.to_numeric(
                    sub[
                        label
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            if len(x) == 0:

                continue

            q = x.quantile(
                [
                    0.01,
                    0.05,
                    0.50,
                    0.95,
                    0.99,
                ]
            )

            rows.append(
                {

                    "window":
                        int(
                            window
                        ),

                    "label_name":
                        label,

                    "n":
                        int(
                            len(
                                x
                            )
                        ),

                    "mean":
                        float(
                            x.mean()
                        ),

                    "std":
                        float(
                            x.std(
                                ddof=0
                            )
                        ),

                    "p01":
                        float(
                            q.loc[
                                0.01
                            ]
                        ),

                    "p05":
                        float(
                            q.loc[
                                0.05
                            ]
                        ),

                    "median":
                        float(
                            q.loc[
                                0.50
                            ]
                        ),

                    "p95":
                        float(
                            q.loc[
                                0.95
                            ]
                        ),

                    "p99":
                        float(
                            q.loc[
                                0.99
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 22. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 6 - Step 2"
    )

    print(
        "Construct Forward Return & Risk Labels"
    )

    print(
        f"Label version: {LABEL_VERSION}"
    )

    print("=" * 80)

    # ========================================================
    # 22.1 Frozen factor design
    # ========================================================

    print()
    print(
        "[1] Validate Day-6 frozen factor design"
    )

    (
        day6_design,
        day6_design_hash,
    ) = load_day6_design()

    print(
        "Day-6 factor design hash:"
    )

    print(
        day6_design_hash
    )

    # ========================================================
    # 22.2 Day-1 return design
    # ========================================================

    print()
    print(
        "[2] Validate Day-1 return source"
    )

    day1_metadata = (
        validate_day1_return_design()
    )

    # ========================================================
    # 22.3 Freeze Step-2 label design
    # ========================================================

    print()
    print(
        "[3] Freeze Step-2 label design"
    )

    (
        label_design,
        label_design_hash,
    ) = build_label_design(
        day6_design_hash
    )

    print(
        "Step-2 label design hash:"
    )

    print(
        label_design_hash
    )

    # ========================================================
    # 22.4 Backup old results
    # ========================================================

    backup_existing_outputs()

    save_json_atomic(
        label_design,
        LABEL_DESIGN_PATH,
    )

    # ========================================================
    # 22.5 Factor universe
    # ========================================================

    print()
    print(
        "[4] Load frozen factor universe"
    )

    factor_universe = (
        load_factor_universe()
    )

    print(
        f"Factor-universe rows: "
        f"{len(factor_universe):,}"
    )

    schedule = (
        build_analysis_schedule(
            factor_universe
        )
    )

    print(
        f"Date-window jobs: "
        f"{len(schedule):,}"
    )

    # ========================================================
    # 22.6 Calendar
    # ========================================================

    print()
    print(
        "[5] Load trading calendar"
    )

    (
        calendar_dates,
        calendar_path,
    ) = load_trade_calendar()

    print(
        f"Calendar: {calendar_path}"
    )

    # ========================================================
    # 22.7 Holding return panel
    # ========================================================

    print()
    print(
        "[6] Construct holding-return source panel"
    )

    holding_panel = (
        load_holding_return_panel()
    )

    print(
        f"Day-1 panel rows: "
        f"{len(holding_panel):,}"
    )

    date_values = (
        build_date_index(
            holding_panel
        )
    )

    # ========================================================
    # 22.8 Prepare final parquet
    # ========================================================

    temp_final_path = Path(
        str(
            FINAL_PANEL_PATH
        )
        +
        ".tmp"
    )

    if temp_final_path.exists():

        temp_final_path.unlink()

    writer = None

    # Freeze the schema from the empty, typed output template, not from a
    # particular job. Final dates skip the merge, so their column order,
    # string storage and NaT timestamp units can differ from ordinary jobs.
    output_schema = pa.Table.from_pandas(
        initialize_label_columns(factor_universe.iloc[:0]),
        preserve_index=False,
    ).schema

    qa_rows = []

    interval_rows = []

    distribution_parts = []

    total_rows = 0

    # ========================================================
    # 22.9 Date x Window loop
    # ========================================================

    try:

        total_jobs = len(
            schedule
        )

        for job_no, row in enumerate(

            schedule.itertuples(
                index=False
            ),

            start=1,
        ):

            analysis_date = pd.Timestamp(
                row.analysis_date
            )

            window = int(
                row.window
            )

            next_analysis_date = (
                pd.Timestamp(
                    row.next_analysis_date
                )
                if pd.notna(
                    row.next_analysis_date
                )
                else pd.NaT
            )

            print()
            print(
                f"[{job_no}/{total_jobs}] "
                f"W={window} | "
                f"{analysis_date.date()}"
            )

            base = (
                factor_universe[
                    (
                        factor_universe[
                            "analysis_date"
                        ]
                        ==
                        analysis_date
                    )
                    &
                    (
                        factor_universe[
                            "window"
                        ]
                        ==
                        window
                    )
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            if len(base) == 0:

                raise RuntimeError(
                    "Empty current factor universe:\n"
                    f"W={window}, "
                    f"date={analysis_date.date()}"
                )

            expected_rows = len(
                base
            )

            (
                label_panel,
                diagnostics,
            ) = build_labels_one_job(

                base=
                    base,

                holding_panel=
                    holding_panel,

                date_values=
                    date_values,

                calendar_dates=
                    calendar_dates,

                analysis_date=
                    analysis_date,

                next_analysis_date=
                    next_analysis_date,
            )

            qa = qa_one_job(

                panel=
                    label_panel,

                expected_rows=
                    expected_rows,

                analysis_date=
                    analysis_date,

                next_analysis_date=
                    next_analysis_date,

                diagnostics=
                    diagnostics,
            )

            qa_rows.append(
                qa
            )

            interval_rows.append(
                {

                    "analysis_date":
                        analysis_date,

                    "window":
                        window,

                    "next_analysis_date":
                        next_analysis_date,

                    "future_period_start":
                        label_panel[
                            "future_period_start"
                        ]
                        .iloc[
                            0
                        ],

                    "future_period_end":
                        label_panel[
                            "future_period_end"
                        ]
                        .iloc[
                            0
                        ],

                    "expected_trade_days":
                        diagnostics[
                            "expected_trade_days"
                        ],

                    "market_trade_day_count":
                        diagnostics[
                            "market_trade_day_count"
                        ],

                    "market_date_coverage":
                        diagnostics[
                            "market_date_coverage"
                        ],

                    "suspension_zero_stock_days":
                        diagnostics[
                            "suspension_zero_stock_days"
                        ],

                    "reopen_stock_days":
                        diagnostics[
                            "reopen_stock_days"
                        ],

                    "reopen_return_recovered_stock_days":
                        diagnostics[
                            "reopen_return_recovered_stock_days"
                        ],

                    "unresolved_open_missing_stock_days":
                        diagnostics[
                            "unresolved_open_missing_stock_days"
                        ],

                    "unresolved_reopen_missing_stock_days":
                        diagnostics[
                            "unresolved_reopen_missing_stock_days"
                        ],
                }
            )

            if not qa[
                "is_last_date"
            ]:

                print(
                    "  Expected trade days = "
                    f"{qa['expected_trade_days']}"
                )

                print(
                    "  Mean coverage = "
                    f"{qa['mean_return_coverage']:.4f}"
                )

                print(
                    "  Return valid share = "
                    f"{qa['return_label_valid_share']:.4f}"
                )

                print(
                    "  Risk valid share = "
                    f"{qa['risk_label_valid_share']:.4f}"
                )

                print(
                    "  Suspension-zero stock-days = "
                    f"{qa['suspension_zero_stock_days']:,}"
                )

                print(
                    "  Reopen stock-days = "
                    f"{qa['reopen_stock_days']:,}"
                )

                print(
                    "  Reopen returns recovered = "
                    f"{qa['reopen_return_recovered_stock_days']:,}"
                )

                print(
                    "  Unresolved open missing = "
                    f"{qa['unresolved_open_missing_stock_days']:,}"
                )

            else:

                print(
                    "  Last frozen analysis date: "
                    "forward labels intentionally NA."
                )

            if (
                STRICT_QA
                and
                not qa[
                    "qa_pass"
                ]
            ):

                raise RuntimeError(
                    "Forward-label QA failed.\n"
                    f"W={window}\n"
                    f"date={analysis_date.date()}\n"
                    f"errors={qa['qa_errors']}"
                )

            # =================================================
            # Normalize once for both partition and streaming output.
            # =================================================

            table = build_label_table(
                label_panel,
                output_schema,
                context=f"W={window}, date={analysis_date.date()}",
            )

            if SAVE_PARTITIONS:

                save_partition(

                    table=
                        table,

                    window=
                        window,

                    date=
                        analysis_date,
                )

            # =================================================
            # Streaming final parquet
            # =================================================

            if writer is None:

                writer = pq.ParquetWriter(

                    temp_final_path,

                    table.schema,

                    compression="zstd",
                )

            else:

                if (
                    table.schema
                    !=
                    writer.schema
                ):

                    raise RuntimeError(
                        "Parquet schema changed "
                        "across jobs."
                    )

            writer.write_table(
                table
            )

            total_rows += len(
                label_panel
            )

            distribution_parts.append(
                label_panel[
                    [
                        "window",
                    ]
                    +
                    ALL_LABELS
                ]
            )

    finally:

        if writer is not None:

            writer.close()

    # ========================================================
    # 22.10 Finalize parquet
    # ========================================================

    if not temp_final_path.exists():

        raise RuntimeError(
            "Temporary output parquet "
            "was not created."
        )

    os.replace(
        temp_final_path,
        FINAL_PANEL_PATH,
    )

    # ========================================================
    # 22.11 QA tables
    # ========================================================

    qa_df = (
        pd.DataFrame(
            qa_rows
        )
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    qa_df.to_csv(

        JOB_QA_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    interval_df = (
        pd.DataFrame(
            interval_rows
        )
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    interval_df.to_csv(

        INTERVAL_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    coverage_df = (
        build_coverage_summary(
            qa_df
        )
    )

    coverage_df.to_csv(

        COVERAGE_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    summary_panel = pd.concat(
        distribution_parts,
        ignore_index=True,
    )

    distribution_df = (
        build_distribution_summary(
            summary_panel
        )
    )

    distribution_df.to_csv(

        DISTRIBUTION_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 22.12 Metadata
    # ========================================================

    metadata = {

        "research_day":
            6,

        "step":
            (
                "Step2_Construct_Forward_"
                "Return_and_Risk_Labels"
            ),

        "label_version":
            LABEL_VERSION,

        "day6_factor_design_hash":
            day6_design_hash,

        "step2_label_design_hash":
            label_design_hash,

        "factor_universe_source":
            str(
                FACTOR_PANEL_PATH
            ),

        "return_panel_source":
            str(
                RETURN_PANEL_PATH
            ),

        "daily_return_source":
            "return_last_trade_simple",

        "daily_return_source_type":
            "simple",

        "holding_return_definition":
            label_design[
                "holding_return_definition"
            ],

        "forward_interval_rule":
            (
                "(analysis_date, next_analysis_date]"
            ),

        "market_benchmark":
            (
                "Daily leave-one-out equal-weight "
                "full-market holding return; "
                "explicit suspension days enter "
                "the market with return zero."
            ),

        "annualization_days":
            ANNUALIZATION_DAYS,

        "coverage_rule": {

            "minimum_return_coverage":
                MIN_RETURN_COVERAGE,

            "minimum_valid_return_days":
                MIN_VALID_RETURN_DAYS,

            "minimum_valid_risk_days":
                MIN_VALID_RISK_DAYS,

            "require_no_unresolved_open_missing":
                REQUIRE_NO_UNRESOLVED_OPEN_MISSING,
        },

        "formal_return_labels":
            RETURN_LABELS,

        "formal_risk_labels":
            RISK_LABELS,

        "important_rules": [

            (
                "The contemporaneous factor universe "
                "is fixed before future outcomes "
                "are observed."
            ),

            (
                "Explicit suspension days are assigned "
                "holding return zero."
            ),

            (
                "Active trading days use "
                "return_last_trade_simple."
            ),

            (
                "Reopen-day returns preserve the "
                "cumulative adjusted-price change "
                "since the previous actual trade."
            ),

            (
                "is_open=True with missing "
                "return_last_trade_simple is never "
                "silently filled with zero."
            ),

            (
                "Rows with unavailable future labels "
                "remain in the factor universe."
            ),

            (
                "The last frozen analysis date has "
                "no forward label."
            ),
        ],

        "output_row_count":
            int(
                total_rows
            ),

        "formal_qa": {

            "job_count":
                int(
                    len(
                        qa_df
                    )
                ),

            "passed_job_count":
                int(
                    qa_df[
                        "qa_pass"
                    ]
                    .sum()
                ),

            "failed_job_count":
                int(
                    (
                        ~qa_df[
                            "qa_pass"
                        ]
                    )
                    .sum()
                ),

            "all_jobs_pass":
                bool(
                    qa_df[
                        "qa_pass"
                    ]
                    .all()
                ),
        },
    }

    save_json_atomic(
        metadata,
        METADATA_PATH,
    )

    # ========================================================
    # 22.13 Console summary
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Forward Holding-Return Label Summary"
    )

    print("=" * 80)

    print(
        f"Total rows: "
        f"{total_rows:,}"
    )

    print(
        f"QA jobs: "
        f"{len(qa_df):,}"
    )

    print(
        f"QA pass: "
        f"{int(qa_df['qa_pass'].sum()):,}"
        f"/{len(qa_df):,}"
    )

    print()
    print(
        coverage_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Final label panel:"
    )

    print(
        FINAL_PANEL_PATH
    )

    print()
    print(
        "Step-2 label design hash:"
    )

    print(
        label_design_hash
    )

    print()
    print("=" * 80)

    print(
        "Day 6 Step 2 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()
