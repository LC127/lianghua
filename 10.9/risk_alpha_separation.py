from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# Day 13 - Step 3
# Risk-Alpha Separation Final Summary
# Pure summary of frozen Step-2 evidence; no re-estimation.
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
S2 = ROOT / "output" / "M1_day13" / "02_step2_frozen_9factor_evidence"
OUT = ROOT / "output" / "M1_day13" / "03_step3_risk_alpha_separation_summary"
OUT.mkdir(parents=True, exist_ok=True)

IN = S2 / "m1_frozen_9factor_final_evidence_matrix_compact.csv"
QA_IN = S2 / "day13_step2_qa.csv"

SUMMARY_OUT = OUT / "risk_alpha_separation_final_summary.csv"
OOS_OUT = OUT / "risk_system_oos_context.csv"
QA_OUT = OUT / "day13_step3_qa.csv"
META_OUT = OUT / "day13_step3_metadata.json"

EXPECTED_FACTORS = 9
EXPECTED_WINDOWS = [60, 120, 252]


def as_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


# ============================================================
# 1. Load frozen Step-2 source and QA
# ============================================================

if not IN.exists() or not QA_IN.exists():
    raise FileNotFoundError("Step-2 compact matrix or QA file is missing.")

q = pd.read_csv(QA_IN)
qd = dict(zip(q["qa_name"], q["qa_value"]))

if not as_bool(qd.get("compact_and_research_all_qa_pass", False)):
    raise RuntimeError("Day-13 Step-2 compact/research QA has not passed.")

df = pd.read_csv(IN, low_memory=False)

required = [
    "factor_order", "factor_name", "risk_block",
    "risk_block_factor_n", "window",

    "risk_neutral_ivol_ic",
    "risk_neutral_ivol_hac_t",
    "risk_same_sign_year_share",

    "oos_all_network_delta_ic",
    "oos_all_network_hac_t",
    "oos_all_network_positive_month_share",
    "oos_all_network_positive_year_share",
    "oos_all_network_evidence_level",

    "oos_block_marginal_ic",
    "oos_block_positive_year_share",
    "oos_block_evidence_level",

    "raw_mean_ic",
    "neutral_mean_ic",

    "raw_20bps_mean_net_hl",
    "raw_20bps_mean_net_lh",
    "neutral_20bps_mean_net_hl",
    "neutral_20bps_mean_net_lh",

    "raw_strategy_capacity_95pct_dates_cny",
    "neutral_strategy_capacity_95pct_dates_cny",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing Step-2 columns: {missing}")

if len(df) != 27 or df[["factor_name", "window"]].duplicated().any():
    raise RuntimeError("Step-2 factor-window structure is invalid.")

if sorted(df["window"].unique().tolist()) != EXPECTED_WINDOWS:
    raise RuntimeError("Unexpected frozen windows.")

# OOS evidence-level consistency checks
if not (
    df.groupby("window")["oos_all_network_delta_ic"].nunique() == 1
).all():
    raise RuntimeError("All-network OOS is not unique within window.")

if not (
    df.groupby(["risk_block", "window"])["oos_block_marginal_ic"].nunique() == 1
).all():
    raise RuntimeError("Block OOS is not unique within block-window.")


# ============================================================
# 2. Factor-level Risk-Alpha separation summary
# ============================================================

keys = [
    "factor_order",
    "factor_name",
    "risk_block",
    "risk_block_factor_n",
]

summary = (
    df.groupby(keys, as_index=False)
    .agg(
        window_n=("window", "nunique"),

        # Risk side
        risk_ivol_ic_mean=("risk_neutral_ivol_ic", "mean"),
        risk_ivol_abs_ic_mean=(
            "risk_neutral_ivol_ic",
            lambda x: x.abs().mean()
        ),
        risk_ivol_abs_hac_t_mean=(
            "risk_neutral_ivol_hac_t",
            lambda x: x.abs().mean()
        ),
        risk_same_sign_year_share_mean=(
            "risk_same_sign_year_share",
            "mean"
        ),

        # Conditional block-level OOS evidence
        block_oos_marginal_ic_mean=(
            "oos_block_marginal_ic",
            "mean"
        ),
        block_oos_positive_year_share_mean=(
            "oos_block_positive_year_share",
            "mean"
        ),
        block_oos_evidence_level=(
            "oos_block_evidence_level",
            "first"
        ),

        # Alpha information
        raw_alpha_ic_mean=("raw_mean_ic", "mean"),
        raw_alpha_abs_ic_mean=(
            "raw_mean_ic",
            lambda x: x.abs().mean()
        ),
        neutral_alpha_ic_mean=("neutral_mean_ic", "mean"),
        neutral_alpha_abs_ic_mean=(
            "neutral_mean_ic",
            lambda x: x.abs().mean()
        ),

        # 20-bps implementation evidence
        raw_20bps_net_hl_mean=(
            "raw_20bps_mean_net_hl",
            "mean"
        ),
        raw_20bps_net_lh_mean=(
            "raw_20bps_mean_net_lh",
            "mean"
        ),
        neutral_20bps_net_hl_mean=(
            "neutral_20bps_mean_net_hl",
            "mean"
        ),
        neutral_20bps_net_lh_mean=(
            "neutral_20bps_mean_net_lh",
            "mean"
        ),

        # Number of windows with positive frozen 95%-dates capacity
        raw_positive_capacity_window_n=(
            "raw_strategy_capacity_95pct_dates_cny",
            lambda x: int((x > 0).sum())
        ),
        neutral_positive_capacity_window_n=(
            "neutral_strategy_capacity_95pct_dates_cny",
            lambda x: int((x > 0).sum())
        ),
    )
)

# Neutralization effect on absolute alpha IC
summary["neutral_minus_raw_abs_ic"] = (
    summary["neutral_alpha_abs_ic_mean"]
    - summary["raw_alpha_abs_ic_mean"]
)

summary["neutral_over_raw_abs_ic_ratio"] = np.where(
    summary["raw_alpha_abs_ic_mean"] > 0,
    summary["neutral_alpha_abs_ic_mean"]
    / summary["raw_alpha_abs_ic_mean"],
    np.nan,
)

summary = (
    summary
    .sort_values("factor_order")
    .reset_index(drop=True)
)


# ============================================================
# 3. Keep all-network OOS at correct window level
# ============================================================

oos = (
    df[
        [
            "window",
            "oos_all_network_delta_ic",
            "oos_all_network_hac_t",
            "oos_all_network_positive_month_share",
            "oos_all_network_positive_year_share",
            "oos_all_network_evidence_level",
        ]
    ]
    .drop_duplicates()
    .sort_values("window")
    .reset_index(drop=True)
)


# ============================================================
# 4. Formal QA
# ============================================================

qa = {
    "step2_compact_qa_pass": True,
    "factor_summary_row_count": len(summary),
    "expected_factor_summary_row_count": EXPECTED_FACTORS,
    "system_oos_row_count": len(oos),
    "expected_system_oos_row_count": len(EXPECTED_WINDOWS),

    "all_factors_have_3_windows":
        bool((summary["window_n"] == 3).all()),

    "factor_summary_missing_count":
        int(summary.isna().sum().sum()),

    "system_oos_missing_count":
        int(oos.isna().sum().sum()),

    "all_network_oos_is_window_level":
        bool(
            (
                oos["oos_all_network_evidence_level"]
                == "WINDOW_LEVEL_SHARED_ACROSS_ALL_9_FACTORS"
            ).all()
        ),

    "factor_selected_from_results": False,
    "window_selected_from_results": False,
    "direction_selected_from_results": False,
    "cost_level_selected_from_results": False,
    "risk_or_alpha_reestimated": False,
}

qa["all_formal_qa_pass"] = bool(
    qa["factor_summary_row_count"] == EXPECTED_FACTORS
    and qa["system_oos_row_count"] == len(EXPECTED_WINDOWS)
    and qa["all_factors_have_3_windows"]
    and qa["factor_summary_missing_count"] == 0
    and qa["system_oos_missing_count"] == 0
    and qa["all_network_oos_is_window_level"]
)

if not qa["all_formal_qa_pass"]:
    raise RuntimeError(f"Day-13 Step-3 QA failed: {qa}")


# ============================================================
# 5. Save outputs
# ============================================================

summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
oos.to_csv(OOS_OUT, index=False, encoding="utf-8-sig")

pd.DataFrame(
    [{"qa_name": k, "qa_value": v} for k, v in qa.items()]
).to_csv(QA_OUT, index=False, encoding="utf-8-sig")

meta = {
    "research_day": 13,
    "step": "Step3_Risk_Alpha_Separation_Final_Summary",

    "principle": (
        "Summarize frozen Step-2 evidence only. "
        "No factor/window/direction/cost selection, "
        "no new risk or alpha estimation."
    ),

    "source": str(IN),

    "outputs": {
        "factor_summary": str(SUMMARY_OUT),
        "system_oos_context": str(OOS_OUT),
    },

    "aggregation": {
        "factor_summary":
            "Mean across frozen W60/W120/W252 for each factor.",
        "absolute_ic":
            "Mean absolute IC across the three frozen windows.",
        "neutral_minus_raw_abs_ic":
            "Neutral absolute alpha IC minus Raw absolute alpha IC.",
        "neutral_over_raw_abs_ic_ratio":
            "Neutral absolute alpha IC divided by Raw absolute alpha IC.",
        "system_oos":
            "Kept at window level because all-network OOS is not factor-specific.",
        "block_oos":
            "Conditional block-level LOBO evidence; not factor-specific.",
    },

    "reporting_cost_bps": 20,
    "all_formal_qa_pass": True,
}

with open(META_OUT, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("=" * 72)
print("Day 13 Step 3 - Risk-Alpha Separation Final Summary")
print("=" * 72)
print(f"Factor summary : {SUMMARY_OUT}")
print(f"System OOS     : {OOS_OUT}")
print(f"QA             : {QA_OUT}")
print(f"Metadata       : {META_OUT}")
print("Formal QA      : PASS")