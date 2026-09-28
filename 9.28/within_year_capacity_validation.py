from pathlib import Path
import json
import os

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
DAY10 = ROOT / "output" / "M1_day10"

STEP1 = DAY10 / "01_stage1_secular_liquidity_panel"
STEP2 = DAY10 / "02_stage2_time_adjusted_capacity"

INPUT = STEP1 / "calendar_liquidity_network_panel.csv"
STEP1_QA = STEP1 / "day10_step1_qa.csv"
STEP2_QA = STEP2 / "day10_step2_qa.csv"

OUTDIR = DAY10 / "03_stage3_within_year_capacity"
OUTDIR.mkdir(parents=True, exist_ok=True)

YEAR_PATH = OUTDIR / "within_year_capacity_by_year.csv"
OVERLAP_PATH = OUTDIR / "within_year_capacity_overlap_only.csv"
SUMMARY_PATH = OUTDIR / "within_year_capacity_summary.csv"
QA_PATH = OUTDIR / "day10_step3_qa.csv"
META_PATH = OUTDIR / "day10_step3_metadata.json"


# ============================================================
# 1. Frozen design
# ============================================================

REGIME = "network_stress_regime"
Y = "median_pair_capacity_yi"

WINDOWS = [60, 120, 252]
REGIMES = ["LOW", "MEDIUM", "HIGH"]


# ============================================================
# 2. Helpers
# ============================================================

def save(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def qa_pass(path):
    q = pd.read_csv(path)
    d = dict(zip(q["qa_name"], q["qa_value"]))
    return str(d["all_formal_qa_pass"]).lower() == "true"


# ============================================================
# 3. Annual HIGH-vs-LOW comparison
# ============================================================

def annual_table(df, scope):
    rows = []

    for (w, year), g in df.groupby(
        ["window", "calendar_year"]
    ):
        lo = g.loc[g[REGIME] == "LOW", Y].dropna()
        md = g.loc[g[REGIME] == "MEDIUM", Y].dropna()
        hi = g.loc[g[REGIME] == "HIGH", Y].dropna()

        overlap = len(lo) > 0 and len(hi) > 0

        rows.append({
            "sample_scope": scope,
            "window": int(w),
            "calendar_year": int(year),

            "total_n": int(len(g)),
            "low_n": int(len(lo)),
            "medium_n": int(len(md)),
            "high_n": int(len(hi)),

            "low_mean_yi":
                lo.mean() if len(lo) else np.nan,
            "high_mean_yi":
                hi.mean() if len(hi) else np.nan,

            "low_median_yi":
                lo.median() if len(lo) else np.nan,
            "high_median_yi":
                hi.median() if len(hi) else np.nan,

            "mean_high_minus_low_yi":
                hi.mean() - lo.mean()
                if overlap else np.nan,

            "median_high_minus_low_yi":
                hi.median() - lo.median()
                if overlap else np.nan,

            "has_high_low_overlap": overlap,
        })

    return pd.DataFrame(rows)


# ============================================================
# 4. Main
# ============================================================

def main():

    print("=" * 72)
    print("M1 - Research Day 10 - Step 3")
    print("Within-Year Capacity Validation")
    print("=" * 72)

    # --------------------------------------------------------
    # Upstream QA
    # --------------------------------------------------------

    if not qa_pass(STEP1_QA):
        raise RuntimeError("Step-1 QA is not PASS.")

    if not qa_pass(STEP2_QA):
        raise RuntimeError("Step-2 QA is not PASS.")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(INPUT)

    required = [
        "analysis_date",
        "window",
        "calendar_year",
        REGIME,
        Y,
        "capacity_factor_n",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    df["analysis_date"] = pd.to_datetime(
        df["analysis_date"],
        errors="raise",
    )

    df["window"] = pd.to_numeric(
        df["window"],
        errors="raise",
    ).astype(int)

    # Same effective sample as Step 2.
    x = df[
        df[REGIME].isin(REGIMES)
        & df[Y].notna()
        & df["calendar_year"].notna()
    ].copy()

    if not (x["capacity_factor_n"] == 9).all():
        raise RuntimeError(
            "Not all effective rows use 9 frozen factors."
        )

    # --------------------------------------------------------
    # Common dates across all 3 windows
    # --------------------------------------------------------

    nwin = (
        x.groupby("analysis_date")["window"]
        .nunique()
    )

    common_dates = set(
        nwin[nwin == len(WINDOWS)].index
    )

    # --------------------------------------------------------
    # Native + Common-Date
    # --------------------------------------------------------

    annual_parts = []

    for scope in ["NATIVE", "COMMON_DATE"]:

        z = (
            x
            if scope == "NATIVE"
            else x[x["analysis_date"].isin(common_dates)]
        )

        annual_parts.append(
            annual_table(z, scope)
        )

    annual = (
        pd.concat(annual_parts, ignore_index=True)
        .sort_values(
            ["sample_scope", "window", "calendar_year"]
        )
        .reset_index(drop=True)
    )

    overlap = annual[
        annual["has_high_low_overlap"]
    ].copy()

    save(annual, YEAR_PATH)
    save(overlap, OVERLAP_PATH)

    # --------------------------------------------------------
    # Window-level support summary
    # --------------------------------------------------------

    rows = []

    for (scope, w), g in annual.groupby(
        ["sample_scope", "window"]
    ):

        o = g[g["has_high_low_overlap"]]

        d = o["mean_high_minus_low_yi"].dropna()

        rows.append({
            "sample_scope": scope,
            "window": int(w),

            "calendar_year_n":
                int(g["calendar_year"].nunique()),

            "overlap_year_n":
                int(len(o)),

            "positive_gap_year_n":
                int((d > 0).sum()),

            "negative_gap_year_n":
                int((d < 0).sum()),

            "zero_gap_year_n":
                int((d == 0).sum()),

            "unweighted_mean_annual_gap_yi":
                d.mean() if len(d) else np.nan,

            "median_annual_gap_yi":
                d.median() if len(d) else np.nan,

            "min_annual_gap_yi":
                d.min() if len(d) else np.nan,

            "max_annual_gap_yi":
                d.max() if len(d) else np.nan,

            "positive_year_share":
                (d > 0).mean() if len(d) else np.nan,
        })

    summary = pd.DataFrame(rows)

    save(summary, SUMMARY_PATH)

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    expected_summary_rows = 2 * len(WINDOWS)

    qa = {
        "step1_formal_qa_pass": True,
        "step2_formal_qa_pass": True,

        "effective_row_count":
            int(len(x)),

        "common_date_count":
            int(len(common_dates)),

        "effective_duplicate_date_window_count":
            int(
                x[
                    ["analysis_date", "window"]
                ].duplicated().sum()
            ),

        "all_effective_rows_use_9_factors":
            bool((x["capacity_factor_n"] == 9).all()),

        "annual_row_count":
            int(len(annual)),

        "overlap_row_count":
            int(len(overlap)),

        "summary_row_count":
            int(len(summary)),

        "expected_summary_row_count":
            expected_summary_rows,

        "capacity_reestimated": False,
        "regime_reestimated": False,
        "factor_selected_from_results": False,
        "window_selected_from_results": False,
        "year_selected_from_results": False,
        "future_information_used": False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa[
            "effective_duplicate_date_window_count"
        ] == 0
        and
        qa["all_effective_rows_use_9_factors"]
        and
        qa["summary_row_count"]
        == expected_summary_rows
    )

    save(
        pd.DataFrame(
            [
                {"qa_name": k, "qa_value": v}
                for k, v in qa.items()
            ]
        ),
        QA_PATH,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "research_day": 10,
        "step":
            "Step3_Within_Year_Capacity_Validation",
        "input": str(INPUT),
        "dependent_variable": Y,
        "regime": REGIME,
        "contrast": "HIGH_MINUS_LOW_WITHIN_YEAR",
        "sample_scopes":
            ["NATIVE", "COMMON_DATE"],
        "statistical_role":
            "descriptive within-year support diagnostic",
        "new_regression": False,
        "formal_significance_test": False,
        "interpretation": (
            "Directly evaluates whether HIGH-vs-LOW "
            "capacity differences exist within the same "
            "calendar year, rather than between different "
            "calendar periods."
        ),
        "formal_qa": qa,
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
    print("Within-year support summary:")
    print(summary.to_string(index=False))

    print()
    print("Formal QA:")
    for k, v in qa.items():
        print(f"  {k}: {v}")

    if not qa["all_formal_qa_pass"]:
        raise RuntimeError(
            "Day-10 Step-3 QA failed."
        )

    print()
    print("=" * 72)
    print("DAY 10 STEP 3 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()