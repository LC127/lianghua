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
    / "05_stage5_factor_temporal_stability"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

ANNUAL_PATH = OUTDIR / "frozen_9factor_ivol_by_year.csv"
SUMMARY_PATH = OUTDIR / "frozen_9factor_temporal_summary.csv"
MATRIX_PATH = OUTDIR / "frozen_9factor_temporal_matrix.csv"
QA_PATH = OUTDIR / "day10_step5_qa.csv"
META_PATH = OUTDIR / "day10_step5_metadata.json"


# ============================================================
# 1. Frozen design
# ============================================================

TARGET = "future_idio_vol_annualized"
METHOD = "FULL_CHARACTERISTIC_NEUTRAL"
WINDOWS = [60, 120, 252]

FACTORS = [
    "residual_degree_percentile",
    "delta_degree_percentile",
    "cross_industry_degree_percentile",
    "cross_industry_degree_ratio",
    "delta_residual_degree_percentile_1m",
    "delta_cross_industry_degree_percentile_1m",
    "neighbor_jaccard_1m",
    "neighbor_retention_1m",
    "outside_community_degree_percentile",
]

CORE_FACTOR = "delta_degree_percentile"


# ============================================================
# 2. Helpers
# ============================================================

def save(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def annual_stats(x, scope):

    z = x.copy()
    z["calendar_year"] = z["analysis_date"].dt.year

    g = (
        z.groupby(
            ["factor_name", "window", "calendar_year"],
            as_index=False,
        )
        .agg(
            month_n=("risk_rank_ic", "size"),
            mean_ic=("risk_rank_ic", "mean"),
            median_ic=("risk_rank_ic", "median"),
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

    g["positive_month_share"] = (
        g["positive_month_n"] / g["month_n"]
    )

    g["sample_scope"] = scope
    return g


# ============================================================
# 3. Main
# ============================================================

def main():

    print("=" * 72)
    print("M1 - Research Day 10 - Step 5")
    print("Frozen 9-Factor Temporal Stability Matrix")
    print("=" * 72)

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(SOURCE)

    required = [
        "analysis_date",
        "window",
        "factor_order",
        "factor_name",
        "method",
        "target_name",
        "risk_rank_ic",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Frozen specification
    # --------------------------------------------------------

    x = df[
        (df["method"] == METHOD)
        & (df["target_name"] == TARGET)
        & df["factor_name"].isin(FACTORS)
        & df["window"].isin(WINDOWS)
    ].copy()

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

    x = x[x["risk_rank_ic"].notna()].copy()

    # --------------------------------------------------------
    # Frozen factor validation
    # --------------------------------------------------------

    found = set(x["factor_name"].unique())

    if found != set(FACTORS):
        raise RuntimeError(
            "Frozen factor mismatch.\n"
            f"Missing: {set(FACTORS) - found}\n"
            f"Extra: {found - set(FACTORS)}"
        )

    if x[
        ["analysis_date", "window", "factor_name"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate date-window-factor rows."
        )

    # --------------------------------------------------------
    # COMMON_DATE:
    # require all 9 factors × all 3 windows on the date
    # --------------------------------------------------------

    expected_cells = len(FACTORS) * len(WINDOWS)

    date_cells = (
        x.groupby("analysis_date")
        .size()
    )

    common_dates = set(
        date_cells[
            date_cells == expected_cells
        ].index
    )

    native = x.copy()

    common = x[
        x["analysis_date"].isin(common_dates)
    ].copy()

    # --------------------------------------------------------
    # Annual panel
    # --------------------------------------------------------

    annual = pd.concat(
        [
            annual_stats(native, "NATIVE"),
            annual_stats(common, "COMMON_DATE"),
        ],
        ignore_index=True,
    )

    factor_order = {
        f: i + 1
        for i, f in enumerate(FACTORS)
    }

    annual["factor_order"] = (
        annual["factor_name"]
        .map(factor_order)
    )

    annual = annual.sort_values(
        [
            "sample_scope",
            "factor_order",
            "window",
            "calendar_year",
        ]
    ).reset_index(drop=True)

    save(annual, ANNUAL_PATH)

    # --------------------------------------------------------
    # Temporal summary
    # --------------------------------------------------------

    monthly = pd.concat(
        [
            native.assign(sample_scope="NATIVE"),
            common.assign(sample_scope="COMMON_DATE"),
        ],
        ignore_index=True,
    )

    rows = []

    for (scope, factor, w), g in annual.groupby(
        ["sample_scope", "factor_name", "window"]
    ):

        m = monthly[
            (monthly["sample_scope"] == scope)
            & (monthly["factor_name"] == factor)
            & (monthly["window"] == w)
        ]

        full_mean = float(m["risk_rank_ic"].mean())
        full_sign = int(np.sign(full_mean))

        annual_sign = np.sign(
            g["mean_ic"].to_numpy()
        )

        same_sign = (
            annual_sign == full_sign
        ) if full_sign != 0 else np.zeros(
            len(g), dtype=bool
        )

        rows.append({
            "sample_scope": scope,
            "factor_order": factor_order[factor],
            "factor_name": factor,
            "window": int(w),

            "month_n": int(len(m)),
            "calendar_year_n": int(len(g)),

            "full_sample_mean_ic": full_mean,
            "full_sample_median_ic":
                float(m["risk_rank_ic"].median()),

            "positive_year_n":
                int((g["mean_ic"] > 0).sum()),
            "negative_year_n":
                int((g["mean_ic"] < 0).sum()),

            "positive_year_share":
                float((g["mean_ic"] > 0).mean()),

            "negative_year_share":
                float((g["mean_ic"] < 0).mean()),

            # Direction consistency relative to the
            # observed full-sample IC sign.
            "same_sign_year_share":
                float(same_sign.mean()),

            "mean_annual_ic":
                float(g["mean_ic"].mean()),

            "median_annual_ic":
                float(g["mean_ic"].median()),

            "min_annual_ic":
                float(g["mean_ic"].min()),

            "max_annual_ic":
                float(g["mean_ic"].max()),

            "positive_month_share":
                float((m["risk_rank_ic"] > 0).mean()),

            "is_core_factor":
                factor == CORE_FACTOR,
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "sample_scope",
                "factor_order",
                "window",
            ]
        )
        .reset_index(drop=True)
    )

    save(summary, SUMMARY_PATH)

    # --------------------------------------------------------
    # Compact matrix for reporting
    # Rows = factor; columns = W60/W120/W252
    # Metric = same-sign annual share
    # --------------------------------------------------------

    matrix = (
        summary.pivot(
            index=[
                "sample_scope",
                "factor_order",
                "factor_name",
            ],
            columns="window",
            values="same_sign_year_share",
        )
        .rename(
            columns={
                60: "W60_sign_consistency",
                120: "W120_sign_consistency",
                252: "W252_sign_consistency",
            }
        )
        .reset_index()
        .sort_values(
            ["sample_scope", "factor_order"]
        )
    )

    matrix["mean_sign_consistency"] = (
        matrix[
            [
                "W60_sign_consistency",
                "W120_sign_consistency",
                "W252_sign_consistency",
            ]
        ].mean(axis=1)
    )

    matrix["is_core_factor"] = (
        matrix["factor_name"] == CORE_FACTOR
    )

    save(matrix, MATRIX_PATH)

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    expected_summary_rows = (
        2 * len(FACTORS) * len(WINDOWS)
    )

    expected_matrix_rows = (
        2 * len(FACTORS)
    )

    qa = {
        "source_exists": True,

        "frozen_factor_count":
            int(x["factor_name"].nunique()),

        "expected_factor_count":
            len(FACTORS),

        "window_count":
            int(x["window"].nunique()),

        "expected_window_count":
            len(WINDOWS),

        "native_row_count":
            int(len(native)),

        "common_date_count":
            int(len(common_dates)),

        "common_row_count":
            int(len(common)),

        "duplicate_date_window_factor_count":
            int(
                x[
                    [
                        "analysis_date",
                        "window",
                        "factor_name",
                    ]
                ].duplicated().sum()
            ),

        "summary_row_count":
            int(len(summary)),

        "expected_summary_row_count":
            expected_summary_rows,

        "matrix_row_count":
            int(len(matrix)),

        "expected_matrix_row_count":
            expected_matrix_rows,

        "risk_ic_reestimated": False,
        "factor_reestimated": False,
        "factor_sign_flipped": False,
        "factor_selected_from_results": False,
        "window_selected_from_results": False,
        "year_selected_from_results": False,
        "future_information_used": False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa["frozen_factor_count"]
        == qa["expected_factor_count"]
        and
        qa["window_count"]
        == qa["expected_window_count"]
        and
        qa["duplicate_date_window_factor_count"] == 0
        and
        qa["summary_row_count"]
        == qa["expected_summary_row_count"]
        and
        qa["matrix_row_count"]
        == qa["expected_matrix_row_count"]
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
            "Step5_Frozen_9Factor_Temporal_Stability",
        "source": str(SOURCE),
        "target": TARGET,
        "method": METHOD,
        "frozen_factors": FACTORS,
        "windows": WINDOWS,
        "sample_scopes":
            ["NATIVE", "COMMON_DATE"],
        "matrix_metric":
            "same_sign_year_share",
        "matrix_metric_definition": (
            "Share of calendar years whose annual mean IC "
            "has the same sign as that factor-window's "
            "full-sample monthly mean IC."
        ),
        "important_note": (
            "The sign is not flipped and the metric is "
            "descriptive only; it is not used for factor "
            "or window selection."
        ),
        "formal_significance_test": False,
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
    print("Temporal-stability matrix:")
    print(matrix.to_string(index=False))

    print()
    print("Formal QA:")
    for k, v in qa.items():
        print(f"  {k}: {v}")

    if not qa["all_formal_qa_pass"]:
        raise RuntimeError(
            "Day-10 Step-5 QA failed."
        )

    print()
    print("=" * 72)
    print("DAY 10 STEP 5 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()