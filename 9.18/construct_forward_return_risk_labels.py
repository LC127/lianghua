from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 0. Paths
# ============================================================

STEP2_DIR = Path(
    r"D:\M1_StockNetwork\output\M1_day6"
    r"\02_step2_forward_labels"
)

FORWARD_PANEL_PATH = (
    STEP2_DIR
    / "forward_label_panel.parquet"
)

JOB_QA_PATH = (
    STEP2_DIR
    / "forward_label_job_qa.csv"
)

INTERVAL_PATH = (
    STEP2_DIR
    / "forward_interval_table.csv"
)

RETURN_PANEL_PATH = Path(
    r"D:\M1_StockNetwork\output\M1_day1"
    r"\03_stage3_return_validation"
    r"\daily_return_panel.parquet"
)

CALENDAR_PATH = Path(
    r"D:\lowfreq\trade_calendar\trade_date.csv"
)


# ------------------------------------------------------------
# IMPORTANT:
# 根据你实际原始行情目录确认。
#
# 如果主数据确实放在 D:\lowfreq\daily_temp3，
# 这里无需修改。
# ------------------------------------------------------------

RAW_DAILY_ROOT = Path(
    r"D:\lowfreq\daily_temp3"
)


OUTPUT_DIR = (
    STEP2_DIR
    / "coverage_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. Settings
# ============================================================

# 只检查 2015 年
TARGET_YEAR = 2015

# 哪些 Date × Window 认为值得检查
LOW_VALID_SHARE_THRESHOLD = 0.90

# 个股原 Step-2 coverage 低于此值，重点检查
LOW_STOCK_COVERAGE_THRESHOLD = 0.80


# ============================================================
# 2. Resolve raw daily file
# ============================================================

def resolve_raw_daily_file(date: pd.Timestamp) -> Path:

    date = pd.Timestamp(date)

    strings = [
        date.strftime("%Y-%m-%d"),
        date.strftime("%Y%m%d"),
    ]

    candidates = []

    for s in strings:

        candidates.extend(
            RAW_DAILY_ROOT.glob(f"*{s}*.parquet")
        )

        candidates.extend(
            RAW_DAILY_ROOT.glob(f"*{s}*.csv")
        )

    candidates = list(
        dict.fromkeys(candidates)
    )

    if len(candidates) == 1:

        return candidates[0]

    if len(candidates) == 0:

        raise FileNotFoundError(
            f"No raw daily file found for {date.date()}\n"
            f"Root = {RAW_DAILY_ROOT}"
        )

    raise RuntimeError(
        f"Multiple raw files found for {date.date()}:\n"
        +
        "\n".join(
            str(x)
            for x in candidates
        )
    )


# ============================================================
# 3. Read raw daily isOpen
# ============================================================

def load_raw_daily_status(
    date: pd.Timestamp
) -> pd.DataFrame:

    path = resolve_raw_daily_file(
        date
    )

    if path.suffix.lower() == ".parquet":

        schema = (
            pq.ParquetFile(path)
            .schema_arrow
            .names
        )

        security_candidates = [
            "security_id",
            "secID",
        ]

        security_col = next(
            (
                c
                for c in security_candidates
                if c in schema
            ),
            None,
        )

        if security_col is None:

            raise ValueError(
                f"No security ID column in {path}"
            )

        if "isOpen" not in schema:

            raise ValueError(
                f"No isOpen column in {path}"
            )

        df = pd.read_parquet(
            path,
            columns=[
                security_col,
                "isOpen",
            ],
        )

    else:

        df = pd.read_csv(
            path,
            usecols=lambda c: (
                c in {
                    "security_id",
                    "secID",
                    "isOpen",
                }
            ),
        )

        security_col = (
            "security_id"
            if "security_id" in df.columns
            else "secID"
        )

    df = df.rename(
        columns={
            security_col:
                "security_id"
        }
    )

    df["security_id"] = (
        df["security_id"]
        .astype("string")
        .str.strip()
    )

    df["isOpen"] = pd.to_numeric(
        df["isOpen"],
        errors="coerce",
    )

    df["trade_date"] = pd.Timestamp(
        date
    )

    if df["security_id"].duplicated().any():

        raise RuntimeError(
            f"Duplicate security IDs in {path}"
        )

    return df[
        [
            "security_id",
            "trade_date",
            "isOpen",
        ]
    ]


# ============================================================
# 4. Load trading calendar
# ============================================================

def load_calendar():

    cal = pd.read_csv(
        CALENDAR_PATH
    )

    date_candidates = [
        "trade_date",
        "date",
        "calendarDate",
        "tradeDate",
    ]

    date_col = next(
        (
            c
            for c in date_candidates
            if c in cal.columns
        ),
        None,
    )

    if date_col is None:

        raise ValueError(
            "Cannot find date column "
            "in trading calendar."
        )

    if "isOpen" in cal.columns:

        cal = cal[
            pd.to_numeric(
                cal["isOpen"],
                errors="coerce",
            )
            ==
            1
        ]

    dates = pd.to_datetime(
        cal[date_col],
        errors="raise",
    )

    return (
        pd.DatetimeIndex(dates)
        .drop_duplicates()
        .sort_values()
    )


# ============================================================
# 5. Load return observations
# ============================================================

def load_returns():

    pf = pq.ParquetFile(
        RETURN_PANEL_PATH
    )

    columns = pf.schema_arrow.names

    required = [
        "security_id",
        "trade_date",
        "return_network",
    ]

    missing = [
        c
        for c in required
        if c not in columns
    ]

    if missing:

        raise RuntimeError(
            f"Return panel missing {missing}"
        )

    df = pd.read_parquet(
        RETURN_PANEL_PATH,
        columns=required,
    )

    df["security_id"] = (
        df["security_id"]
        .astype("string")
        .str.strip()
    )

    df["trade_date"] = pd.to_datetime(
        df["trade_date"]
    )

    df["return_network"] = pd.to_numeric(
        df["return_network"],
        errors="coerce",
    )

    return df


# ============================================================
# 6. Identify low-coverage jobs
# ============================================================

def identify_problem_jobs():

    qa = pd.read_csv(
        JOB_QA_PATH
    )

    qa["analysis_date"] = pd.to_datetime(
        qa["analysis_date"]
    )

    qa["next_analysis_date"] = pd.to_datetime(
        qa["next_analysis_date"]
    )

    problem = qa[
        (
            qa["analysis_date"].dt.year
            ==
            TARGET_YEAR
        )
        &
        (
            ~qa["is_last_date"]
        )
        &
        (
            qa["return_label_valid_share"]
            <
            LOW_VALID_SHARE_THRESHOLD
        )
    ].copy()

    return problem.sort_values(
        [
            "window",
            "analysis_date",
        ]
    )


# ============================================================
# 7. Diagnose one Date × Window job
# ============================================================

def diagnose_job(
    analysis_date,
    next_analysis_date,
    window,
    forward_panel,
    return_panel,
    calendar_dates,
):

    analysis_date = pd.Timestamp(
        analysis_date
    )

    next_analysis_date = pd.Timestamp(
        next_analysis_date
    )

    # --------------------------------------------------------
    # Current factor universe
    # --------------------------------------------------------

    current = forward_panel[
        (
            forward_panel["analysis_date"]
            ==
            analysis_date
        )
        &
        (
            forward_panel["window"]
            ==
            int(window)
        )
    ].copy()

    if len(current) == 0:

        raise RuntimeError(
            "No forward-panel rows found."
        )

    # --------------------------------------------------------
    # Focus on stocks with low observed coverage
    # --------------------------------------------------------

    low = current[
        current["future_return_coverage"]
        <
        LOW_STOCK_COVERAGE_THRESHOLD
    ].copy()

    if len(low) == 0:

        return None, None

    stock_ids = (
        low["security_id"]
        .astype(str)
        .unique()
    )

    # --------------------------------------------------------
    # Expected exchange trading dates
    # --------------------------------------------------------

    expected_dates = calendar_dates[
        (
            calendar_dates
            >
            analysis_date
        )
        &
        (
            calendar_dates
            <=
            next_analysis_date
        )
    ]

    expected_days = len(
        expected_dates
    )

    # ========================================================
    # Build stock × expected-date grid
    # ========================================================

    grid = pd.MultiIndex.from_product(
        [
            stock_ids,
            expected_dates,
        ],
        names=[
            "security_id",
            "trade_date",
        ],
    ).to_frame(
        index=False
    )

    # ========================================================
    # Did the validated return panel contain an observation?
    # ========================================================

    ret = return_panel[
        (
            return_panel["trade_date"]
            >
            analysis_date
        )
        &
        (
            return_panel["trade_date"]
            <=
            next_analysis_date
        )
        &
        (
            return_panel["security_id"]
            .isin(stock_ids)
        )
    ][
        [
            "security_id",
            "trade_date",
            "return_network",
        ]
    ].copy()

    ret["return_observed"] = (
        ret["return_network"]
        .notna()
    )

    grid = grid.merge(
        ret,
        on=[
            "security_id",
            "trade_date",
        ],
        how="left",
        validate="one_to_one",
    )

    grid["return_observed"] = (
        grid["return_observed"]
        .fillna(False)
    )

    # ========================================================
    # Load raw daily status for all expected dates
    # ========================================================

    raw_parts = []

    for date in expected_dates:

        raw_parts.append(
            load_raw_daily_status(
                date
            )
        )

    raw_status = pd.concat(
        raw_parts,
        ignore_index=True,
    )

    raw_status = raw_status[
        raw_status["security_id"]
        .isin(stock_ids)
    ].copy()

    raw_status["raw_record_exists"] = True

    grid = grid.merge(
        raw_status,
        on=[
            "security_id",
            "trade_date",
        ],
        how="left",
        validate="one_to_one",
    )

    grid["raw_record_exists"] = (
        grid["raw_record_exists"]
        .fillna(False)
    )

    # ========================================================
    # Missing-day attribution
    # ========================================================

    conditions = [

        grid["return_observed"],

        (
            ~grid["return_observed"]
            &
            grid["raw_record_exists"]
            &
            (
                grid["isOpen"]
                ==
                0
            )
        ),

        (
            ~grid["return_observed"]
            &
            grid["raw_record_exists"]
            &
            (
                grid["isOpen"]
                ==
                1
            )
        ),

        (
            ~grid["return_observed"]
            &
            ~grid["raw_record_exists"]
        ),
    ]

    choices = [

        "RETURN_OBSERVED",

        "EXPLICIT_SUSPENSION",

        "OPEN_BUT_RETURN_MISSING",

        "NO_RAW_RECORD",
    ]

    grid["day_status"] = np.select(
        conditions,
        choices,
        default="OTHER",
    )

    grid["analysis_date"] = (
        analysis_date
    )

    grid["next_analysis_date"] = (
        next_analysis_date
    )

    grid["window"] = int(
        window
    )

    # ========================================================
    # Aggregate stock-level attribution
    # ========================================================

    status_counts = (
        grid.groupby(
            [
                "security_id",
                "day_status",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    for col in [
        "RETURN_OBSERVED",
        "EXPLICIT_SUSPENSION",
        "OPEN_BUT_RETURN_MISSING",
        "NO_RAW_RECORD",
        "OTHER",
    ]:

        if col not in status_counts.columns:

            status_counts[col] = 0

    status_counts[
        "expected_trade_days"
    ] = expected_days

    status_counts[
        "diagnostic_raw_coverage"
    ] = (

        status_counts[
            "RETURN_OBSERVED"
        ]

        /

        expected_days
    )

    status_counts[
        "suspension_adjusted_coverage"
    ] = (

        (
            status_counts[
                "RETURN_OBSERVED"
            ]

            +

            status_counts[
                "EXPLICIT_SUSPENSION"
            ]
        )

        /

        expected_days
    )

    status_counts[
        "missing_day_count"
    ] = (

        expected_days

        -

        status_counts[
            "RETURN_OBSERVED"
        ]
    )

    status_counts[
        "suspension_share_of_missing"
    ] = np.where(

        status_counts[
            "missing_day_count"
        ]
        >
        0,

        (
            status_counts[
                "EXPLICIT_SUSPENSION"
            ]

            /

            status_counts[
                "missing_day_count"
            ]
        ),

        np.nan,
    )

    status_counts[
        "analysis_date"
    ] = analysis_date

    status_counts[
        "next_analysis_date"
    ] = next_analysis_date

    status_counts[
        "window"
    ] = int(
        window
    )

    # --------------------------------------------------------
    # Add Step-2 coverage for exact comparison
    # --------------------------------------------------------

    status_counts = status_counts.merge(

        low[
            [
                "security_id",
                "future_return_coverage",
                "future_valid_return_days",
            ]
        ],

        on="security_id",

        how="left",

        validate="one_to_one",
    )

    return (
        grid,
        status_counts,
    )


# ============================================================
# 8. Main
# ============================================================

def main():

    print("=" * 80)
    print("Forward Coverage Suspension Diagnostic")
    print("=" * 80)

    problem_jobs = identify_problem_jobs()

    print("\nProblem jobs:")
    print(
        problem_jobs[
            [
                "analysis_date",
                "next_analysis_date",
                "window",
                "mean_return_coverage",
                "return_label_valid_share",
            ]
        ].to_string(
            index=False
        )
    )

    if len(problem_jobs) == 0:

        print(
            "\nNo low-valid-share jobs found."
        )

        return

    calendar_dates = load_calendar()

    return_panel = load_returns()

    forward_columns = [
        "analysis_date",
        "window",
        "master_index",
        "security_id",
        "future_valid_return_days",
        "future_return_coverage",
        "future_return_label_valid",
    ]

    forward_panel = pd.read_parquet(
        FORWARD_PANEL_PATH,
        columns=forward_columns,
    )

    forward_panel["analysis_date"] = (
        pd.to_datetime(
            forward_panel["analysis_date"]
        )
    )

    all_day_detail = []

    all_stock_summary = []

    for row in problem_jobs.itertuples(
        index=False
    ):

        print()
        print(
            f"Diagnosing "
            f"W={row.window} | "
            f"{row.analysis_date.date()} "
            f"-> "
            f"{row.next_analysis_date.date()}"
        )

        (
            detail,
            stock_summary,
        ) = diagnose_job(

            analysis_date=
                row.analysis_date,

            next_analysis_date=
                row.next_analysis_date,

            window=
                row.window,

            forward_panel=
                forward_panel,

            return_panel=
                return_panel,

            calendar_dates=
                calendar_dates,
        )

        if detail is not None:

            all_day_detail.append(
                detail
            )

            all_stock_summary.append(
                stock_summary
            )

    if len(
        all_stock_summary
    ) == 0:

        print(
            "No low-coverage stock observations found."
        )

        return

    detail_df = pd.concat(
        all_day_detail,
        ignore_index=True,
    )

    stock_df = pd.concat(
        all_stock_summary,
        ignore_index=True,
    )

    # ========================================================
    # Job-level summary
    # ========================================================

    job_summary = (
        stock_df.groupby(
            [
                "analysis_date",
                "next_analysis_date",
                "window",
            ]
        )
        .agg(

            low_coverage_stock_count=(
                "security_id",
                "count",
            ),

            total_missing_days=(
                "missing_day_count",
                "sum",
            ),

            total_suspension_days=(
                "EXPLICIT_SUSPENSION",
                "sum",
            ),

            total_open_but_missing_days=(
                "OPEN_BUT_RETURN_MISSING",
                "sum",
            ),

            total_no_raw_record_days=(
                "NO_RAW_RECORD",
                "sum",
            ),

            mean_raw_coverage=(
                "diagnostic_raw_coverage",
                "mean",
            ),

            mean_suspension_adjusted_coverage=(
                "suspension_adjusted_coverage",
                "mean",
            ),

            mean_suspension_share_of_missing=(
                "suspension_share_of_missing",
                "mean",
            ),

        )
        .reset_index()
    )

    job_summary[
        "aggregate_suspension_share_of_missing"
    ] = (

        job_summary[
            "total_suspension_days"
        ]

        /

        job_summary[
            "total_missing_days"
        ]
    )

    # ========================================================
    # Save outputs
    # ========================================================

    detail_df.to_csv(
        OUTPUT_DIR
        / "coverage_missing_day_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stock_df.to_csv(
        OUTPUT_DIR
        / "coverage_stock_level_attribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    job_summary.to_csv(
        OUTPUT_DIR
        / "coverage_job_level_attribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    problem_jobs.to_csv(
        OUTPUT_DIR
        / "low_coverage_jobs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("Job-level attribution")
    print("=" * 80)

    print(
        job_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Output directory:"
    )
    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()