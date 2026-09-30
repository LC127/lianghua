from pathlib import Path
import json, os
import numpy as np
import pandas as pd

# ============================================================
# Day 12 - Step 5
# Capacity-Constrained Alpha Validation
# Reuse frozen Day-8 rebalance-level capacity; do NOT re-estimate it.
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
D12 = ROOT / "output" / "M1_day12"

S4 = D12 / "04_stage4_turnover_cost_validation"
S4_MONTHLY = S4 / "monthly_turnover_cost_stress.csv"
S4_QA = S4 / "day12_step4_qa.csv"

DAY8_CAP = (
    ROOT / "output" / "M1_day8"
    / "04_stage4_capacity_limits"
    / "capacity_limit_rebalance_pair.csv"
)

OUT = D12 / "05_stage5_capacity_constrained_alpha"
OUT.mkdir(parents=True, exist_ok=True)

MONTHLY_OUT = OUT / "monthly_capacity_alpha.csv"
SUMMARY_OUT = OUT / "capacity_alpha_summary.csv"
QA_OUT = OUT / "day12_step5_qa.csv"
META_OUT = OUT / "day12_step5_metadata.json"

WINDOWS = [60, 120, 252]
COST_BPS = [10, 20, 30, 50]

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

# Frozen Day-8 implementation convention.
PARTICIPATION_RATE = 0.05
EXECUTION_DAYS = 1
EXECUTABLE_SHARE_THRESHOLD = 0.95

# Source-match QA only; does not redefine Day-8 capacity.
MIN_CAPACITY_MATCH_SHARE = 0.95


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

def find_col(cols, aliases):
    lower = {str(c).strip().lower(): c for c in cols}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None

def empirical_lower_quantile(x, q):
    """
    Conservative empirical quantile using an observed value.
    q=0.05 gives a capital level met/exceeded on about 95% of dates.
    """
    x = np.sort(np.asarray(x, float))
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    k = int(np.floor(q * (len(x) - 1)))
    return float(x[k])

def resolve_factor_col(df):
    c = find_col(
        df.columns,
        ["factor", "factor_name", "network_factor", "signal", "feature"],
    )
    if c is not None:
        return c

    target = set(FACTORS)
    best_col, best_overlap = None, 0

    for col in df.columns:
        if df[col].dtype.kind not in {"O", "U", "S"} and not pd.api.types.is_string_dtype(df[col]):
            continue
        vals = set(df[col].dropna().astype(str).unique())
        overlap = len(vals & target)
        if overlap > best_overlap:
            best_col, best_overlap = col, overlap

    if best_overlap < 1:
        raise RuntimeError("Cannot resolve factor column in Day-8 capacity source.")
    return best_col

def load_capacity():
    cap = pd.read_csv(DAY8_CAP, low_memory=False)

    date_col = find_col(
        cap.columns,
        [
            "analysis_date",
            "rebalance_date",
            "to_analysis_date",
            "current_analysis_date",
            "date",
        ],
    )
    win_col = find_col(
        cap.columns,
        ["window", "window_length", "network_window"],
    )
    cap_col = find_col(
        cap.columns,
        ["capacity_cny", "capacity", "capacity_limit_cny"],
    )
    factor_col = resolve_factor_col(cap)

    missing = [
        name for name, col in {
            "date": date_col,
            "window": win_col,
            "capacity": cap_col,
        }.items()
        if col is None
    ]
    if missing:
        save_csv(
            pd.DataFrame({"source_column": cap.columns}),
            OUT / "capacity_source_columns.csv",
        )
        raise RuntimeError(
            f"Cannot resolve Day-8 capacity columns: {missing}. "
            "See capacity_source_columns.csv."
        )

    cap = cap.rename(
        columns={
            date_col: "analysis_date",
            win_col: "window",
            factor_col: "factor",
            cap_col: "capacity_cny",
        }
    )

    cap["analysis_date"] = pd.to_datetime(cap["analysis_date"], errors="raise")
    cap["window"] = pd.to_numeric(cap["window"], errors="raise").astype(int)
    cap["factor"] = cap["factor"].astype(str)
    cap["capacity_cny"] = pd.to_numeric(cap["capacity_cny"], errors="coerce")

    cap = cap.loc[
        cap["window"].isin(WINDOWS)
        & cap["factor"].isin(FACTORS),
        ["analysis_date", "window", "factor", "capacity_cny"],
    ].copy()

    key = ["analysis_date", "window", "factor"]

    # Duplicates are allowed only when capacity is identical.
    if cap[key].duplicated().any():
        bad = (
            cap.groupby(key)["capacity_cny"]
            .nunique(dropna=False)
            .gt(1)
        )
        if bad.any():
            raise RuntimeError(
                "Day-8 capacity source has non-invariant duplicate "
                "date-window-factor rows."
            )
        cap = cap.drop_duplicates(key)

    return cap


# ============================================================
# Summary
# ============================================================

def summarize(monthly):
    rows = []

    for key, g in monthly.groupby(
        ["sample_scope", "method", "window", "factor"],
        sort=True,
    ):
        scope, method, window, factor = key
        c = g["capacity_cny"].dropna().to_numpy(float)
        n = len(g)

        row = {
            "sample_scope": scope,
            "method": method,
            "window": int(window),
            "factor": factor,
            "date_n": n,
            "capacity_date_n": len(c),
            "capacity_match_share": len(c) / n if n else np.nan,
            "min_rebalance_capacity_cny": float(np.min(c)) if len(c) else np.nan,
            "p05_rebalance_capacity_cny": empirical_lower_quantile(c, 0.05),
            "p10_rebalance_capacity_cny": empirical_lower_quantile(c, 0.10),
            "median_rebalance_capacity_cny": float(np.median(c)) if len(c) else np.nan,
            "mean_rebalance_capacity_cny": float(np.mean(c)) if len(c) else np.nan,
        }

        # Strategy-level interpretations:
        # - strict capacity: executable under the frozen Day-8 rule on every matched date;
        # - 95%-date capacity: frozen Day-8 rule holds on about 95% of matched dates.
        row["strategy_capacity_all_dates_cny"] = row["min_rebalance_capacity_cny"]
        row["strategy_capacity_95pct_dates_cny"] = row["p05_rebalance_capacity_cny"]

        for bps in COST_BPS:
            for direction in ["hl", "lh"]:
                col = f"net_{direction}_{bps}bps"
                v = g[col].to_numpy(float)
                mean_net = float(np.mean(v))
                ann_net = 12.0 * mean_net

                row[f"mean_net_{direction}_{bps}bps"] = mean_net
                row[f"annualized_net_{direction}_{bps}bps"] = ann_net
                row[f"positive_net_{direction}_{bps}bps_month_share"] = float(
                    np.mean(v > 0)
                )

                # Scale-normalized alpha converted into an illustrative annual
                # CNY P&L at the two frozen strategy-capacity summaries.
                row[f"annual_pnl_all_dates_{direction}_{bps}bps_cny"] = (
                    ann_net * row["strategy_capacity_all_dates_cny"]
                )
                row[f"annual_pnl_95pct_dates_{direction}_{bps}bps_cny"] = (
                    ann_net * row["strategy_capacity_95pct_dates_cny"]
                )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["sample_scope", "method", "window", "factor"])
        .reset_index(drop=True)
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 82)
    print("M1 - Day 12 - Step 5: Capacity-Constrained Alpha Validation")
    print("=" * 82)

    for p in [S4_MONTHLY, S4_QA, DAY8_CAP]:
        if not p.exists():
            raise FileNotFoundError(p)

    if not qa_pass(S4_QA):
        raise RuntimeError("Day-12 Step-4 QA has not passed.")

    cap = load_capacity()

    s4 = pd.read_csv(S4_MONTHLY, low_memory=False)
    s4["analysis_date"] = pd.to_datetime(s4["analysis_date"], errors="raise")
    s4["window"] = pd.to_numeric(s4["window"], errors="raise").astype(int)

    # Step 4 has existed in two compatible naming versions:
    #   compact : one_way_trade_notional
    #   frozen  : total_one_way_trade_notional
    # Normalize either name to one canonical column. Step 5 does not
    # re-estimate turnover/cost; it only reuses Step-4 net returns and
    # Day-8 frozen capacity.
    trade_col = find_col(
        s4.columns,
        ["one_way_trade_notional", "total_one_way_trade_notional"],
    )
    if trade_col is not None and trade_col != "one_way_trade_notional":
        s4["one_way_trade_notional"] = pd.to_numeric(
            s4[trade_col], errors="coerce"
        )

    required = [
        "sample_scope", "method", "window", "analysis_date", "factor",
        "high_minus_low",
    ] + [
        f"net_{d}_{c}bps"
        for c in COST_BPS
        for d in ["hl", "lh"]
    ]

    missing = [c for c in required if c not in s4.columns]
    if missing:
        raise RuntimeError(f"Step-4 monthly file missing columns: {missing}")

    key5 = ["sample_scope", "method", "window", "analysis_date", "factor"]
    if s4[key5].duplicated().any():
        raise RuntimeError("Duplicate key in Step-4 monthly file.")

    monthly = s4.merge(
        cap,
        on=["analysis_date", "window", "factor"],
        how="left",
        validate="many_to_one",
    )

    valid_cap = np.isfinite(
        pd.to_numeric(monthly["capacity_cny"], errors="coerce").to_numpy(float)
    ) & (monthly["capacity_cny"].fillna(-1).to_numpy(float) >= 0)

    monthly["capacity_available"] = valid_cap
    summary = summarize(monthly)

    match_share = float(monthly["capacity_available"].mean())
    expected_summary_rows = 2 * 2 * 3 * 9

    qa = {
        "step4_formal_qa_pass": True,
        "day8_capacity_source_exists": True,
        "frozen_factor_count": len(FACTORS),
        "window_count": len(WINDOWS),
        "capacity_source_row_count": len(cap),
        "capacity_source_duplicate_key_count": int(
            cap[["analysis_date", "window", "factor"]].duplicated().sum()
        ),
        "negative_capacity_count": int((cap["capacity_cny"] < 0).sum()),
        "monthly_row_count": len(monthly),
        "monthly_duplicate_key_count": int(monthly[key5].duplicated().sum()),
        "step4_trade_notional_source_column": (
            trade_col if trade_col is not None else "NOT_REQUIRED"
        ),
        "step4_trade_notional_normalized": bool(trade_col is not None),
        "capacity_match_share": match_share,
        "minimum_required_capacity_match_share": MIN_CAPACITY_MATCH_SHARE,
        "summary_row_count": len(summary),
        "expected_summary_row_count": expected_summary_rows,
        "participation_rate": PARTICIPATION_RATE,
        "execution_days": EXECUTION_DAYS,
        "day8_executable_share_threshold": EXECUTABLE_SHARE_THRESHOLD,
        "capacity_reestimated": False,
        "day8_capacity_reused_directly": True,
        "strategy_capacity_all_dates_definition": "MIN_REBALANCE_CAPACITY",
        "strategy_capacity_95pct_dates_definition": "EMPIRICAL_5PCT_REBALANCE_CAPACITY",
        "factor_selected_from_results": False,
        "window_selected_from_results": False,
        "cost_level_selected_from_results": False,
        "direction_selected_from_results": False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa["capacity_source_duplicate_key_count"] == 0
        and qa["negative_capacity_count"] == 0
        and qa["monthly_duplicate_key_count"] == 0
        and qa["capacity_match_share"] >= MIN_CAPACITY_MATCH_SHARE
        and qa["summary_row_count"] == expected_summary_rows
    )

    save_csv(monthly, MONTHLY_OUT)
    save_csv(summary, SUMMARY_OUT)
    save_csv(
        pd.DataFrame(
            [{"qa_name": k, "qa_value": v} for k, v in qa.items()]
        ),
        QA_OUT,
    )

    meta = {
        "research_day": 12,
        "step": "Step5_Capacity_Constrained_Alpha_Validation",
        "capacity_source": str(DAY8_CAP),
        "capacity_column": "capacity_cny",
        "frozen_day8_capacity_rule": {
            "participation_rate": PARTICIPATION_RATE,
            "execution_days": EXECUTION_DAYS,
            "executable_share_threshold": EXECUTABLE_SHARE_THRESHOLD,
        },
        "rebalance_capacity_rule": (
            "Use frozen Day-8 capacity_cny directly; no capacity is re-estimated."
        ),
        "strategy_capacity_rule": (
            "Report both min rebalance capacity (strict all-date capacity) and "
            "empirical 5th-percentile rebalance capacity (capacity met or exceeded "
            "on about 95% of matched evaluation dates)."
        ),
        "alpha_link": (
            "Join the frozen Day-8 rebalance capacity to the frozen Step-4 "
            "gross/net alpha observations by analysis_date, window, and factor. "
            "Step-4 turnover is not re-estimated in Step 5; either historical "
            "turnover column name is accepted for compatibility. No factor/"
            "window/cost/direction is selected."
        ),
        "important_limitation": (
            "If Day-8 capacity has no method/scope dimension, the same frozen "
            "factor-date-window capacity is used as the implementation-capacity "
            "benchmark for RAW and characteristic-neutral Step-4 portfolios. "
            "This is a frozen capacity benchmark, not a re-estimated method-specific "
            "capacity curve."
        ),
        "formal_qa": qa,
    }

    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    print("\nCapacity summary preview:")
    show = [
        "sample_scope", "method", "window", "factor",
        "capacity_match_share",
        "strategy_capacity_all_dates_cny",
        "strategy_capacity_95pct_dates_cny",
        "median_rebalance_capacity_cny",
        "mean_net_lh_20bps",
        "annual_pnl_95pct_dates_lh_20bps_cny",
    ]
    print(summary[show].to_string(index=False))

    print("\nFormal QA:")
    for k, v in qa.items():
        print(f"  {k}: {v}")

    if not qa["all_formal_qa_pass"]:
        raise RuntimeError(
            "Day-12 Step-5 formal QA failed; inspect day12_step5_qa.csv."
        )

    print(f"\nPASS. Output: {OUT}")


if __name__ == "__main__":
    main()
