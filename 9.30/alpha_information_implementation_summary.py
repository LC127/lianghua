from pathlib import Path
import json, os
import numpy as np
import pandas as pd

# ============================================================
# Day 12 - Step 6
# Alpha Information-Implementation Joint Summary
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
D12 = ROOT / "output" / "M1_day12"

S2 = D12 / "02_stage2_alpha_ic_validation"
S3 = D12 / "03_stage3_monotonic_portfolio_validation"
S4 = D12 / "04_stage4_turnover_cost_validation"
S5 = D12 / "05_stage5_capacity_constrained_alpha"

IC_FILE = S2 / "alpha_ic_comparison_summary.csv"
PORT_FILE = S3 / "portfolio_validation_summary.csv"
COST_FILE = S4 / "turnover_cost_summary.csv"
CAP_FILE = S5 / "capacity_alpha_summary.csv"

QA2 = S2 / "day12_step2_qa.csv"
QA4 = S4 / "day12_step4_qa.csv"
QA5 = S5 / "day12_step5_qa.csv"

OUT = D12 / "06_stage6_alpha_joint_summary"
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = [60, 120, 252]
COST_BPS = [10, 20, 30, 50]

RAW = "RAW_FACTOR_SORT"
NEUTRAL = "CHARACTERISTIC_NEUTRAL_FACTOR_SORT"

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

KEY = ["sample_scope", "method", "window", "factor"]


# ============================================================
# Helpers
# ============================================================

def save_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)

def as_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes"}

def qa_pass(path):
    q = pd.read_csv(path)
    d = dict(zip(q["qa_name"], q["qa_value"]))
    return as_bool(d.get("all_formal_qa_pass", False))

def require(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"{name} missing columns: {miss}")

def check_key(df, name):
    if df[KEY].duplicated().any():
        raise RuntimeError(f"{name} has duplicate joint keys.")


# ============================================================
# Step 2 -> method-long information table
# ============================================================

def load_information():
    x = pd.read_csv(IC_FILE)

    require(
        x,
        [
            "sample_scope", "window", "factor",
            "mean_raw_ic", "hac_t_raw_ic", "bh_q_raw_ic",
            "positive_raw_month_share",
            "mean_neutral_ic", "hac_t_neutral_ic", "bh_q_neutral_ic",
            "positive_neutral_month_share",
            "same_sign_neutral_year_share",
        ],
        "Step-2 IC summary",
    )

    base = ["sample_scope", "window", "factor"]

    raw = x[base].copy()
    raw["method"] = RAW
    raw["mean_ic"] = x["mean_raw_ic"]
    raw["hac_t_ic"] = x["hac_t_raw_ic"]
    raw["bh_q_ic"] = x["bh_q_raw_ic"]
    raw["same_sign_month_share"] = x["positive_raw_month_share"]
    raw["same_sign_year_share_ic"] = np.nan

    neu = x[base].copy()
    neu["method"] = NEUTRAL
    neu["mean_ic"] = x["mean_neutral_ic"]
    neu["hac_t_ic"] = x["hac_t_neutral_ic"]
    neu["bh_q_ic"] = x["bh_q_neutral_ic"]
    neu["same_sign_month_share"] = x["positive_neutral_month_share"]
    neu["same_sign_year_share_ic"] = x["same_sign_neutral_year_share"]

    out = pd.concat([raw, neu], ignore_index=True)
    return out[KEY + [
        "mean_ic", "hac_t_ic", "bh_q_ic",
        "same_sign_month_share", "same_sign_year_share_ic",
    ]]


# ============================================================
# Main joint table
# ============================================================

def build_joint(info, port, cost, cap):
    require(
        port,
        KEY + [
            "date_n", "mean_high_minus_low",
            "annualized_sharpe_hl",
            "mean_quintile_curve_spearman",
            "same_sign_hl_year_share",
            "bh_q_high_minus_low",
            "mean_q1_return", "mean_q5_return",
        ],
        "Step-3 portfolio summary",
    )

    require(
        cost,
        KEY + [
            "mean_one_way_trade_notional",
            "mean_half_l1_turnover",
            "break_even_bps_hl",
            "break_even_bps_lh",
        ],
        "Step-4 cost summary",
    )

    require(
        cap,
        KEY + [
            "capacity_match_share",
            "strategy_capacity_all_dates_cny",
            "strategy_capacity_95pct_dates_cny",
        ],
        "Step-5 capacity summary",
    )

    for df, name in [
        (info, "Step-2 information"),
        (port, "Step-3 portfolio"),
        (cost, "Step-4 cost"),
        (cap, "Step-5 capacity"),
    ]:
        check_key(df, name)

    p = port[KEY + [
        "date_n", "mean_high_minus_low",
        "annualized_sharpe_hl",
        "mean_quintile_curve_spearman",
        "same_sign_hl_year_share",
        "bh_q_high_minus_low",
        "mean_q1_return", "mean_q5_return",
    ]]

    c = cost[KEY + [
        "mean_one_way_trade_notional",
        "mean_half_l1_turnover",
        "break_even_bps_hl",
        "break_even_bps_lh",
    ]]

    a = cap[KEY + [
        "capacity_match_share",
        "strategy_capacity_all_dates_cny",
        "strategy_capacity_95pct_dates_cny",
    ]]

    out = (
        info.merge(p, on=KEY, how="inner", validate="one_to_one")
            .merge(c, on=KEY, how="inner", validate="one_to_one")
            .merge(a, on=KEY, how="inner", validate="one_to_one")
    )

    out["ic_portfolio_sign_consistent"] = (
        np.sign(out["mean_ic"])
        == np.sign(out["mean_high_minus_low"])
    )

    return out.sort_values(KEY).reset_index(drop=True)


# ============================================================
# Cost x capacity stress table
# ============================================================

def build_stress(joint, cost, cap):
    rows = []

    c = cost.set_index(KEY)
    a = cap.set_index(KEY)

    for _, r in joint.iterrows():
        key = tuple(r[k] for k in KEY)
        cr, ar = c.loc[key], a.loc[key]

        for bps in COST_BPS:
            rows.append({
                **{k: r[k] for k in KEY},
                "cost_bps": bps,
                "mean_ic": r["mean_ic"],
                "mean_gross_hl": r["mean_high_minus_low"],
                "mean_one_way_trade_notional":
                    r["mean_one_way_trade_notional"],
                "mean_net_hl": cr[f"mean_net_hl_{bps}bps"],
                "mean_net_lh": cr[f"mean_net_lh_{bps}bps"],
                "bh_q_net_hl": cr[f"bh_q_net_hl_{bps}bps"],
                "bh_q_net_lh": cr[f"bh_q_net_lh_{bps}bps"],
                "capacity_95pct_dates_cny":
                    r["strategy_capacity_95pct_dates_cny"],
                "annual_pnl_95pct_hl_cny":
                    ar[f"annual_pnl_95pct_dates_hl_{bps}bps_cny"],
                "annual_pnl_95pct_lh_cny":
                    ar[f"annual_pnl_95pct_dates_lh_{bps}bps_cny"],
            })

    return pd.DataFrame(rows).sort_values(
        KEY + ["cost_bps"]
    ).reset_index(drop=True)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 82)
    print("M1 - Day 12 - Step 6")
    print("Alpha Information-Implementation Joint Summary")
    print("=" * 82)

    for p in [IC_FILE, PORT_FILE, COST_FILE, CAP_FILE, QA2, QA4, QA5]:
        if not p.exists():
            raise FileNotFoundError(p)

    if not qa_pass(QA2):
        raise RuntimeError("Step 2 QA has not passed.")
    if not qa_pass(QA4):
        raise RuntimeError("Step 4 QA has not passed.")
    if not qa_pass(QA5):
        raise RuntimeError("Step 5 QA has not passed.")

    info = load_information()
    port = pd.read_csv(PORT_FILE)
    cost = pd.read_csv(COST_FILE)
    cap = pd.read_csv(CAP_FILE)

    joint = build_joint(info, port, cost, cap)
    stress = build_stress(joint, cost, cap)

    expected_joint = 2 * 2 * len(WINDOWS) * len(FACTORS)   # 108
    expected_stress = expected_joint * len(COST_BPS)       # 432

    core = [
        "mean_ic", "mean_high_minus_low",
        "mean_one_way_trade_notional",
        "strategy_capacity_95pct_dates_cny",
    ]

    qa = {
        "step2_formal_qa_pass": True,
        "step4_formal_qa_pass": True,
        "step5_formal_qa_pass": True,
        "factor_count": len(FACTORS),
        "window_count": len(WINDOWS),
        "method_count": int(joint["method"].nunique()),
        "sample_scope_count": int(joint["sample_scope"].nunique()),
        "joint_row_count": len(joint),
        "expected_joint_row_count": expected_joint,
        "joint_duplicate_key_count": int(joint[KEY].duplicated().sum()),
        "joint_core_missing_count": int(joint[core].isna().sum().sum()),
        "stress_row_count": len(stress),
        "expected_stress_row_count": expected_stress,
        "stress_duplicate_key_count": int(
            stress[KEY + ["cost_bps"]].duplicated().sum()
        ),
        "cost_grid_bps": "|".join(map(str, COST_BPS)),
        "factor_selected_from_results": False,
        "window_selected_from_results": False,
        "direction_selected_from_results": False,
        "cost_level_selected_from_results": False,
        "capacity_reestimated": False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa["factor_count"] == 9
        and qa["window_count"] == 3
        and qa["method_count"] == 2
        and qa["sample_scope_count"] == 2
        and qa["joint_row_count"] == expected_joint
        and qa["joint_duplicate_key_count"] == 0
        and qa["joint_core_missing_count"] == 0
        and qa["stress_row_count"] == expected_stress
        and qa["stress_duplicate_key_count"] == 0
    )

    save_csv(joint, OUT / "alpha_information_implementation_joint_summary.csv")
    save_csv(stress, OUT / "alpha_cost_capacity_stress_matrix.csv")
    save_csv(
        pd.DataFrame(
            [{"qa_name": k, "qa_value": v} for k, v in qa.items()]
        ),
        OUT / "day12_step6_qa.csv",
    )

    meta = {
        "research_day": 12,
        "step": "Step6_Alpha_Information_Implementation_Joint_Summary",
        "joint_key": KEY,
        "cost_grid_bps": COST_BPS,
        "information_source": str(IC_FILE),
        "portfolio_source": str(PORT_FILE),
        "turnover_cost_source": str(COST_FILE),
        "capacity_source": str(CAP_FILE),
        "joint_summary_rule": (
            "One row per scope-method-window-factor. Step 2 information, "
            "Step 3 gross portfolio evidence, Step 4 turnover/break-even cost, "
            "and Step 5 frozen capacity are joined without selecting factors, "
            "windows, directions, or cost levels."
        ),
        "stress_matrix_rule": (
            "Report both canonical Q5-Q1 and mechanical Q1-Q5 net outcomes "
            "at 10/20/30/50 bps, together with frozen 95%-date capacity and "
            "capacity-scaled annual P&L. No direction is selected."
        ),
        "formal_qa": qa,
    }

    with open(
        OUT / "day12_step6_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nJoint summary preview:")
    show = [
        "sample_scope", "method", "window", "factor",
        "mean_ic", "mean_high_minus_low",
        "mean_one_way_trade_notional",
        "break_even_bps_hl", "break_even_bps_lh",
        "strategy_capacity_95pct_dates_cny",
        "ic_portfolio_sign_consistent",
    ]
    print(joint[show].to_string(index=False))

    print("\nFormal QA:")
    for k, v in qa.items():
        print(f"  {k}: {v}")

    if not qa["all_formal_qa_pass"]:
        raise RuntimeError(
            "Day-12 Step-6 formal QA failed; inspect day12_step6_qa.csv."
        )

    print(f"\nPASS. Output: {OUT}")


if __name__ == "__main__":
    main()
