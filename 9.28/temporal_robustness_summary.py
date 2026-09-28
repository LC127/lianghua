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

S2 = DAY10 / "02_stage2_time_adjusted_capacity"
S3 = DAY10 / "03_stage3_within_year_capacity"
S4 = DAY10 / "04_stage4_ivol_temporal_stability"
S5 = DAY10 / "05_stage5_factor_temporal_stability"

CAP_RESULT = S2 / "time_adjusted_capacity_results.csv"
CAP_YEAR = S3 / "within_year_capacity_summary.csv"
DELTA_IVOL = S4 / "delta_degree_ivol_temporal_summary.csv"
FACTOR_MATRIX = S5 / "frozen_9factor_temporal_matrix.csv"
FACTOR_SUMMARY = S5 / "frozen_9factor_temporal_summary.csv"

QA_FILES = {
    "step2": S2 / "day10_step2_qa.csv",
    "step3": S3 / "day10_step3_qa.csv",
    "step4": S4 / "day10_step4_qa.csv",
    "step5": S5 / "day10_step5_qa.csv",
}

OUTDIR = DAY10 / "06_stage6_temporal_robustness_summary"
OUTDIR.mkdir(parents=True, exist_ok=True)

CAP_OUT = OUTDIR / "capacity_temporal_robustness_summary.csv"
RISK_OUT = OUTDIR / "ivol_temporal_robustness_summary.csv"
MAIN_OUT = OUTDIR / "day10_temporal_robustness_summary.csv"
QA_OUT = OUTDIR / "day10_step6_qa.csv"
META_OUT = OUTDIR / "day10_step6_metadata.json"


# ============================================================
# 1. Frozen design
# ============================================================

SCOPES = ["NATIVE", "COMMON_DATE"]
WINDOWS = [60, 120, 252]

M0 = "M0_REGIME_ONLY"
M3 = "M3_ADV_PLUS_YEAR_FE"

CORE_FACTOR = "delta_degree_percentile"
EXPECTED_FACTOR_N = 9


# ============================================================
# 2. Helpers
# ============================================================

def save(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def qa_pass(path):
    if not path.exists():
        raise FileNotFoundError(path)

    q = pd.read_csv(path)
    d = dict(zip(q["qa_name"], q["qa_value"]))

    return str(
        d.get("all_formal_qa_pass", "")
    ).strip().lower() == "true"


def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{name} missing columns: {missing}"
        )


# ============================================================
# 3. Capacity summary: Step 2 + Step 3
# ============================================================

def build_capacity_summary():

    r = pd.read_csv(CAP_RESULT)
    y = pd.read_csv(CAP_YEAR)

    require_cols(
        r,
        [
            "sample_scope",
            "window",
            "model",
            "high_minus_low_yi",
            "high_minus_low_t",
            "high_minus_low_q_bh",
        ],
        "Step2",
    )

    require_cols(
        y,
        [
            "sample_scope",
            "window",
            "overlap_year_n",
            "positive_gap_year_n",
            "negative_gap_year_n",
            "median_annual_gap_yi",
        ],
        "Step3",
    )

    key = ["sample_scope", "window"]

    raw = r[
        r["model"] == M0
    ][
        key
        + [
            "high_minus_low_yi",
            "high_minus_low_t",
        ]
    ].rename(
        columns={
            "high_minus_low_yi":
                "raw_capacity_hl_yi",
            "high_minus_low_t":
                "raw_capacity_hl_t",
        }
    )

    adj = r[
        r["model"] == M3
    ][
        key
        + [
            "high_minus_low_yi",
            "high_minus_low_t",
            "high_minus_low_q_bh",
        ]
    ].rename(
        columns={
            "high_minus_low_yi":
                "adjusted_capacity_hl_yi",
            "high_minus_low_t":
                "adjusted_capacity_hl_t",
            "high_minus_low_q_bh":
                "adjusted_capacity_hl_q",
        }
    )

    out = (
        raw.merge(
            adj,
            on=key,
            validate="one_to_one",
        )
        .merge(
            y[
                key
                + [
                    "overlap_year_n",
                    "positive_gap_year_n",
                    "negative_gap_year_n",
                    "median_annual_gap_yi",
                ]
            ],
            on=key,
            validate="one_to_one",
        )
    )

    out["adjustment_change_yi"] = (
        out["adjusted_capacity_hl_yi"]
        - out["raw_capacity_hl_yi"]
    )

    out["sign_reversal_after_adjustment"] = (
        np.sign(out["raw_capacity_hl_yi"])
        !=
        np.sign(out["adjusted_capacity_hl_yi"])
    )

    return out.sort_values(key).reset_index(drop=True)


# ============================================================
# 4. IVOL temporal robustness: Step 4 + Step 5
# ============================================================

def build_risk_summary():

    delta = pd.read_csv(DELTA_IVOL)
    matrix = pd.read_csv(FACTOR_MATRIX)
    full = pd.read_csv(FACTOR_SUMMARY)

    require_cols(
        delta,
        [
            "sample_scope",
            "window",
            "monthly_mean_ic",
            "positive_year_share",
            "median_annual_ic",
            "positive_month_share",
        ],
        "Step4",
    )

    require_cols(
        matrix,
        [
            "sample_scope",
            "factor_name",
            "W60_sign_consistency",
            "W120_sign_consistency",
            "W252_sign_consistency",
        ],
        "Step5 matrix",
    )

    require_cols(
        full,
        [
            "sample_scope",
            "factor_name",
            "window",
            "full_sample_mean_ic",
            "same_sign_year_share",
        ],
        "Step5 summary",
    )

    # --------------------------------------------------------
    # Broad frozen-factor stability by window
    # --------------------------------------------------------

    m = matrix.melt(
        id_vars=[
            "sample_scope",
            "factor_name",
        ],
        value_vars=[
            "W60_sign_consistency",
            "W120_sign_consistency",
            "W252_sign_consistency",
        ],
        var_name="window_name",
        value_name="sign_consistency",
    )

    m["window"] = (
        m["window_name"]
        .str.extract(r"W(\d+)")[0]
        .astype(int)
    )

    broad = (
        m.groupby(
            ["sample_scope", "window"],
            as_index=False,
        )
        .agg(
            frozen_factor_n=(
                "factor_name",
                "nunique",
            ),
            mean_factor_sign_consistency=(
                "sign_consistency",
                "mean",
            ),
            median_factor_sign_consistency=(
                "sign_consistency",
                "median",
            ),
            min_factor_sign_consistency=(
                "sign_consistency",
                "min",
            ),
            perfect_sign_factor_n=(
                "sign_consistency",
                lambda s: int(
                    np.isclose(s, 1.0).sum()
                ),
            ),
        )
    )

    out = delta[
        [
            "sample_scope",
            "window",
            "monthly_mean_ic",
            "positive_year_share",
            "median_annual_ic",
            "positive_month_share",
        ]
    ].rename(
        columns={
            "monthly_mean_ic":
                "delta_degree_mean_ic",
            "positive_year_share":
                "delta_degree_positive_year_share",
            "median_annual_ic":
                "delta_degree_median_annual_ic",
            "positive_month_share":
                "delta_degree_positive_month_share",
        }
    )

    out = out.merge(
        broad,
        on=["sample_scope", "window"],
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Internal consistency:
    # Step4 Native delta-degree must reproduce Step5 Native.
    # --------------------------------------------------------

    delta5 = full[
        full["factor_name"] == CORE_FACTOR
    ][
        [
            "sample_scope",
            "window",
            "full_sample_mean_ic",
            "same_sign_year_share",
        ]
    ]

    out = out.merge(
        delta5,
        on=["sample_scope", "window"],
        validate="one_to_one",
    )

    return out.sort_values(
        ["sample_scope", "window"]
    ).reset_index(drop=True)


# ============================================================
# 5. Main
# ============================================================

def main():

    print("=" * 72)
    print("M1 - Research Day 10 - Step 6")
    print("Temporal Robustness Summary")
    print("=" * 72)

    # --------------------------------------------------------
    # Upstream QA
    # --------------------------------------------------------

    upstream = {
        name: qa_pass(path)
        for name, path in QA_FILES.items()
    }

    if not all(upstream.values()):
        raise RuntimeError(
            f"Upstream QA failed: {upstream}"
        )

    # --------------------------------------------------------
    # Integrated summaries
    # --------------------------------------------------------

    cap = build_capacity_summary()
    risk = build_risk_summary()

    save(cap, CAP_OUT)
    save(risk, RISK_OUT)

    key = ["sample_scope", "window"]

    main = (
        cap.merge(
            risk,
            on=key,
            validate="one_to_one",
        )
        .sort_values(key)
        .reset_index(drop=True)
    )

    save(main, MAIN_OUT)

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    native = main[
        main["sample_scope"] == "NATIVE"
    ]

    native_delta_match = bool(
        np.allclose(
            native["delta_degree_mean_ic"],
            native["full_sample_mean_ic"],
            atol=1e-12,
            rtol=0,
        )
        and
        np.allclose(
            native[
                "delta_degree_positive_year_share"
            ],
            native["same_sign_year_share"],
            atol=1e-12,
            rtol=0,
        )
    )

    qa = {
        "step2_qa_pass": upstream["step2"],
        "step3_qa_pass": upstream["step3"],
        "step4_qa_pass": upstream["step4"],
        "step5_qa_pass": upstream["step5"],

        "capacity_summary_row_count":
            int(len(cap)),

        "risk_summary_row_count":
            int(len(risk)),

        "integrated_summary_row_count":
            int(len(main)),

        "expected_summary_row_count":
            2 * len(WINDOWS),

        "all_windows_present":
            sorted(main["window"].unique().tolist())
            == WINDOWS,

        "all_scopes_present":
            set(main["sample_scope"])
            == set(SCOPES),

        "all_broad_summaries_use_9_factors":
            bool(
                (
                    main["frozen_factor_n"]
                    == EXPECTED_FACTOR_N
                ).all()
            ),

        "native_delta_step4_step5_match":
            native_delta_match,

        "new_capacity_estimation": False,
        "new_risk_estimation": False,
        "new_significance_test": False,
        "factor_selected_from_summary": False,
        "window_selected_from_summary": False,
        "future_information_used": False,
    }

    qa["all_formal_qa_pass"] = bool(
        all(upstream.values())
        and
        qa["capacity_summary_row_count"]
        == qa["expected_summary_row_count"]
        and
        qa["risk_summary_row_count"]
        == qa["expected_summary_row_count"]
        and
        qa["integrated_summary_row_count"]
        == qa["expected_summary_row_count"]
        and
        qa["all_windows_present"]
        and
        qa["all_scopes_present"]
        and
        qa["all_broad_summaries_use_9_factors"]
        and
        qa["native_delta_step4_step5_match"]
    )

    save(
        pd.DataFrame(
            [
                {"qa_name": k, "qa_value": v}
                for k, v in qa.items()
            ]
        ),
        QA_OUT,
    )

    metadata = {
        "research_day": 10,
        "step":
            "Step6_Temporal_Robustness_Summary",

        "capacity_inputs": {
            "time_adjusted":
                str(CAP_RESULT),
            "within_year":
                str(CAP_YEAR),
        },

        "risk_inputs": {
            "delta_degree":
                str(DELTA_IVOL),
            "frozen_9factor_matrix":
                str(FACTOR_MATRIX),
        },

        "capacity_primary_comparison":
            "M0_REGIME_ONLY vs M3_ADV_PLUS_YEAR_FE",

        "risk_primary_metrics": [
            "delta_degree_positive_year_share",
            "factor-level same-sign annual consistency",
        ],

        "new_statistical_estimation": False,

        "interpretation": (
            "Integrates the temporal-confounding evidence "
            "for implementation capacity with the temporal "
            "persistence evidence for future-IVOL information."
        ),

        "formal_qa": qa,
    }

    with open(
        META_OUT,
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
    print("Integrated Day-10 summary:")
    print(
        main[
            [
                "sample_scope",
                "window",
                "raw_capacity_hl_yi",
                "adjusted_capacity_hl_yi",
                "adjusted_capacity_hl_q",
                "overlap_year_n",
                "negative_gap_year_n",
                "delta_degree_mean_ic",
                "delta_degree_positive_year_share",
                "mean_factor_sign_consistency",
                "perfect_sign_factor_n",
            ]
        ].to_string(index=False)
    )

    print()
    print("Formal QA:")
    for k, v in qa.items():
        print(f"  {k}: {v}")

    if not qa["all_formal_qa_pass"]:
        raise RuntimeError(
            "Day-10 Step-6 QA failed."
        )

    print()
    print("=" * 72)
    print("DAY 10 STEP 6 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()