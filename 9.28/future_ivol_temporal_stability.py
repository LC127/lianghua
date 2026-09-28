from pathlib import Path
import json
import os

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
OUT = ROOT / "output"

SOURCE = (
    OUT / "M1_day7"
    / "02_stage2_traditional_characteristic_controls"
    / "monthly_controlled_risk_rank_ic.csv"
)

OUTDIR = (
    OUT / "M1_day10"
    / "04_stage4_ivol_temporal_stability"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

MONTHLY_PATH = OUTDIR / "delta_degree_ivol_monthly.csv"
YEAR_PATH = OUTDIR / "delta_degree_ivol_by_year.csv"
SUMMARY_PATH = OUTDIR / "delta_degree_ivol_temporal_summary.csv"
QA_PATH = OUTDIR / "day10_step4_qa.csv"
META_PATH = OUTDIR / "day10_step4_metadata.json"


# ============================================================
# 1. Frozen design
# ============================================================

FACTOR = "delta_degree_percentile"
TARGET = "future_idio_vol_annualized"
METHOD = "FULL_CHARACTERISTIC_NEUTRAL"
WINDOWS = [60, 120, 252]


# ============================================================
# 2. Helpers
# ============================================================

def save(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def find_col(cols, aliases):
    lookup = {str(c).lower(): c for c in cols}
    for x in aliases:
        if x.lower() in lookup:
            return lookup[x.lower()]
    return None


# ============================================================
# 3. Load frozen monthly IVOL IC
# ============================================================

# ============================================================
# 3. Load frozen monthly IVOL IC
# ============================================================

def load_data():

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(SOURCE)
    cols = list(df.columns)

    date_col = find_col(
        cols,
        ["analysis_date", "date"]
    )

    window_col = find_col(
        cols,
        ["window", "window_length"]
    )

    factor_col = find_col(
        cols,
        ["factor_name", "factor"]
    )

    method_col = find_col(
        cols,
        ["method", "control_method"]
    )

    target_col = find_col(
        cols,
        [
            "target_name",
            "target",
            "risk_target",
            "outcome",
        ]
    )

    value_col = find_col(
        cols,
        ["risk_rank_ic"]
    )

    required = {
        "date": date_col,
        "window": window_col,
        "factor": factor_col,
        "method": method_col,
        "target": target_col,
        "value": value_col,
    }

    missing = [
        k for k, v in required.items()
        if v is None
    ]

    if missing:
        raise RuntimeError(
            f"Missing required fields: {missing}\n"
            f"Columns: {cols}"
        )

    # --------------------------------------------------------
    # Frozen specification only
    # --------------------------------------------------------

    x = df[
        (df[factor_col] == FACTOR)
        & (df[method_col] == METHOD)
        & (df[target_col] == TARGET)
    ][
        [
            date_col,
            window_col,
            value_col,
        ]
    ].copy()

    if x.empty:
        raise RuntimeError(
            "No rows match the frozen specification:\n"
            f"factor={FACTOR}\n"
            f"method={METHOD}\n"
            f"target={TARGET}"
        )

    x.columns = [
        "analysis_date",
        "window",
        "risk_rank_ic",
    ]

    x["analysis_date"] = pd.to_datetime(
        x["analysis_date"],
        errors="raise",
    )

    x["window"] = pd.to_numeric(
        x["window"],
        errors="raise",
    ).astype(int)

    x["risk_rank_ic"] = pd.to_numeric(
        x["risk_rank_ic"],
        errors="coerce",
    )

    x = x[
        x["window"].isin(WINDOWS)
        & x["risk_rank_ic"].notna()
    ].copy()

    if x.empty:
        raise RuntimeError(
            "No valid IVOL rank-IC observations "
            "after frozen filters."
        )

    if x[
        ["analysis_date", "window"]
    ].duplicated().any():
        dup = x[
            x[
                ["analysis_date", "window"]
            ].duplicated(
                keep=False
            )
        ]

        raise RuntimeError(
            "Duplicate date-window rows found:\n"
            f"{dup.head(20)}"
        )

    # All three frozen windows must exist.
    found_windows = sorted(
        x["window"].unique().tolist()
    )

    if found_windows != WINDOWS:
        raise RuntimeError(
            "Unexpected frozen windows.\n"
            f"Expected: {WINDOWS}\n"
            f"Found: {found_windows}"
        )

    print(f"Source        : {SOURCE}")
    print(f"Factor        : {FACTOR}")
    print(f"Method        : {METHOD}")
    print(f"Target        : {TARGET}")
    print(f"Target column : {target_col}")
    print(f"Value column  : {value_col}")
    print(f"Rows          : {len(x):,}")
    print(f"Windows       : {found_windows}")

    return (
        x.sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# 4. Annual summaries
# ============================================================

def annual_summary(df, scope):

    x = df.copy()
    x["calendar_year"] = (
        x["analysis_date"].dt.year
    )

    out = (
        x.groupby(
            ["window", "calendar_year"],
            as_index=False,
        )
        .agg(
            month_n=("risk_rank_ic", "size"),
            mean_ic=("risk_rank_ic", "mean"),
            median_ic=("risk_rank_ic", "median"),
            min_ic=("risk_rank_ic", "min"),
            max_ic=("risk_rank_ic", "max"),
            positive_month_n=(
                "risk_rank_ic",
                lambda s: int((s > 0).sum()),
            ),
            negative_month_n=(
                "risk_rank_ic",
                lambda s: int((s < 0).sum()),
            ),
        )
    )

    out["positive_month_share"] = (
        out["positive_month_n"]
        / out["month_n"]
    )

    out["annual_mean_positive"] = (
        out["mean_ic"] > 0
    )

    out["sample_scope"] = scope

    return out


# ============================================================
# 5. Main
# ============================================================

def main():

    print("=" * 72)
    print("M1 - Research Day 10 - Step 4")
    print("Future-IVOL Temporal Stability")
    print("=" * 72)

    x = load_data()

    # --------------------------------------------------------
    # Common dates across W60/W120/W252
    # --------------------------------------------------------

    nwin = (
        x.groupby("analysis_date")["window"]
        .nunique()
    )

    common_dates = set(
        nwin[nwin == len(WINDOWS)].index
    )

    native = x.copy()

    common = x[
        x["analysis_date"].isin(
            common_dates
        )
    ].copy()

    monthly = pd.concat(
        [
            native.assign(
                sample_scope="NATIVE"
            ),
            common.assign(
                sample_scope="COMMON_DATE"
            ),
        ],
        ignore_index=True,
    )

    save(monthly, MONTHLY_PATH)

    # --------------------------------------------------------
    # Annual results
    # --------------------------------------------------------

    annual = pd.concat(
        [
            annual_summary(
                native, "NATIVE"
            ),
            annual_summary(
                common, "COMMON_DATE"
            ),
        ],
        ignore_index=True,
    ).sort_values(
        [
            "sample_scope",
            "window",
            "calendar_year",
        ]
    )

    save(annual, YEAR_PATH)

    # --------------------------------------------------------
    # Window-level temporal stability
    # --------------------------------------------------------

    rows = []

    for (scope, w), g in annual.groupby(
        ["sample_scope", "window"]
    ):

        monthly_g = monthly[
            (monthly["sample_scope"] == scope)
            & (monthly["window"] == w)
        ]

        rows.append({
            "sample_scope": scope,
            "window": int(w),

            "month_n":
                int(len(monthly_g)),

            "calendar_year_n":
                int(g["calendar_year"].nunique()),

            "positive_year_n":
                int(g["annual_mean_positive"].sum()),

            "negative_year_n":
                int((~g["annual_mean_positive"]).sum()),

            "positive_year_share":
                float(g["annual_mean_positive"].mean()),

            "mean_annual_ic":
                float(g["mean_ic"].mean()),

            "median_annual_ic":
                float(g["mean_ic"].median()),

            "min_annual_ic":
                float(g["mean_ic"].min()),

            "max_annual_ic":
                float(g["mean_ic"].max()),

            "monthly_mean_ic":
                float(
                    monthly_g["risk_rank_ic"].mean()
                ),

            "monthly_median_ic":
                float(
                    monthly_g["risk_rank_ic"].median()
                ),

            "positive_month_share":
                float(
                    (
                        monthly_g["risk_rank_ic"] > 0
                    ).mean()
                ),
        })

    summary = pd.DataFrame(rows)

    save(summary, SUMMARY_PATH)

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    qa = {
        "source_exists": True,

        "factor":
            FACTOR,

        "target":
            TARGET,

        "method":
            METHOD,

        "native_row_count":
            int(len(native)),

        "common_date_count":
            int(len(common_dates)),

        "common_row_count":
            int(len(common)),

        "native_window_count":
            int(native["window"].nunique()),

        "duplicate_date_window_count":
            int(
                native[
                    ["analysis_date", "window"]
                ].duplicated().sum()
            ),

        "annual_result_row_count":
            int(len(annual)),

        "summary_row_count":
            int(len(summary)),

        "expected_summary_row_count":
            2 * len(WINDOWS),

        "risk_ic_reestimated":
            False,

        "factor_reestimated":
            False,

        "factor_selected_from_results":
            False,

        "window_selected_from_results":
            False,

        "year_selected_from_results":
            False,

        "future_information_used":
            False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa["native_window_count"]
        == len(WINDOWS)
        and
        qa["duplicate_date_window_count"] == 0
        and
        qa["summary_row_count"]
        == qa["expected_summary_row_count"]
        and
        qa["native_row_count"] > 0
        and
        qa["common_row_count"] > 0
    )

    save(
        pd.DataFrame(
            [
                {
                    "qa_name": k,
                    "qa_value": v,
                }
                for k, v in qa.items()
            ]
        ),
        QA_PATH,
    )

    metadata = {
        "research_day": 10,
        "step":
            "Step4_Future_IVOL_Temporal_Stability",

        "source":
            str(SOURCE),

        "factor":
            FACTOR,

        "target":
            TARGET,

        "method":
            METHOD,

        "windows":
            WINDOWS,

        "sample_scopes":
            ["NATIVE", "COMMON_DATE"],

        "annual_statistics":
            [
                "mean IC",
                "median IC",
                "positive-month share",
            ],

        "primary_temporal_metric":
            "positive_year_share",

        "formal_annual_significance_test":
            False,

        "interpretation":
            (
                "Temporal stability of the frozen "
                "delta-degree relation with future "
                "idiosyncratic volatility."
            ),

        "formal_qa":
            qa,
    }

    with open(
        META_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print()
    print("Temporal stability summary:")
    print(summary.to_string(index=False))

    print()
    print("Formal QA:")
    for k, v in qa.items():
        print(f"  {k}: {v}")

    if not qa["all_formal_qa_pass"]:
        raise RuntimeError(
            "Day-10 Step-4 QA failed."
        )

    print()
    print("=" * 72)
    print("DAY 10 STEP 4 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()