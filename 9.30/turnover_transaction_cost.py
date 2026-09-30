from pathlib import Path
import json, math, os
import numpy as np
import pandas as pd

# ============================================================
# 0. Paths
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
DAY12 = ROOT / "output" / "M1_day12"

STEP1 = DAY12 / "01_stage1_frozen_alpha_evaluation_panel"
STEP3 = DAY12 / "03_stage3_monotonic_portfolio_validation"

PANEL_PATH = STEP1 / "frozen_alpha_evaluation_panel.parquet"
STEP1_QA = STEP1 / "day12_step1_qa.csv"

STEP3_MONTHLY = STEP3 / "monthly_quintile_portfolio_returns.csv"
STEP3_QA = STEP3 / "day12_step3_qa.csv"
STEP3_COMMON = STEP3 / "portfolio_common_date_manifest.csv"

OUTDIR = DAY12 / "04_stage4_turnover_cost_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

MONTHLY_OUT = OUTDIR / "monthly_turnover_cost_stress.csv"
ANNUAL_OUT = OUTDIR / "annual_turnover_cost_summary.csv"
SUMMARY_OUT = OUTDIR / "turnover_cost_summary.csv"
QA_OUT = OUTDIR / "day12_step4_qa.csv"
META_OUT = OUTDIR / "day12_step4_metadata.json"

# ============================================================
# 1. Frozen design
# ============================================================

WINDOWS = [60, 120, 252]
N_GROUPS = 5

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

INDUSTRY = "industry_id1"
CONTROL_NUMERIC = [
    "log_market_value",
    "momentum_120_20",
    "reversal_20",
    "volatility_60",
    "log_turnover_20",
]

RAW = "RAW_FACTOR_SORT"
NEUTRAL = "CHARACTERISTIC_NEUTRAL_FACTOR_SORT"
NATIVE = "NATIVE"
COMMON = "COMMON_DATE"

# Same formation rules as Step 3.
MIN_FORMATION_N = 100
MIN_GROUP_FORMATION_N = 15

# One-way transaction-cost stress grid.
# Example: 10 bps = 0.001 per unit of traded notional.
COST_BPS = [10, 20, 30, 50]

PERIODS_PER_YEAR = 12
NW_LAG = 3
EPS = 1e-12
LSTSQ_RCOND = 1e-10

# ============================================================
# 2. Small helpers
# ============================================================

def atomic_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)

def as_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes"}

def load_qa(path):
    q = pd.read_csv(path)
    return dict(zip(q["qa_name"], q["qa_value"]))

def rank_pct(x):
    return pd.Series(np.asarray(x, float)).rank(
        method="average", pct=True
    ).to_numpy(float)

def rank_z(x):
    r = rank_pct(x)
    if len(r) < 2:
        return None
    s = np.std(r, ddof=1)
    if not np.isfinite(s) or s <= EPS:
        return None
    return (r - r.mean()) / s

def rank_z_frame(df, cols):
    z = np.zeros((len(df), len(cols)), float)
    active = np.zeros(len(cols), bool)
    for j, c in enumerate(cols):
        v = rank_z(df[c].to_numpy(float))
        if v is not None:
            z[:, j] = v
            active[j] = True
    return z, active

def quintile(score):
    g = np.ceil(rank_pct(score) * N_GROUPS)
    return np.clip(g, 1, N_GROUPS).astype(int)

# ============================================================
# 3. Same neutral score definition as Step 3
# ============================================================

def neutral_scores(df):
    factor_z, active = rank_z_frame(df, FACTORS)
    if not active.all():
        return None

    control_z, active_c = rank_z_frame(df, CONTROL_NUMERIC)
    cont = control_z[:, active_c] if active_c.any() else np.empty((len(df), 0))

    ind = df[INDUSTRY].astype("string").astype(str).to_numpy()
    levels = sorted(pd.unique(ind).tolist())
    dummies = np.column_stack([(ind == x).astype(float) for x in levels])

    X = np.column_stack([cont, dummies])
    if X.shape[0] <= X.shape[1]:
        return None

    beta = np.linalg.lstsq(
        X, factor_z, rcond=LSTSQ_RCOND
    )[0]
    return factor_z - X @ beta

# ============================================================
# 4. Weight / turnover construction
# ============================================================

def leg_weights(ids, groups, q):
    """
    Equal-weight one-leg portfolio.
    Returns {security_id: weight}; weights sum to 1.
    """
    mask = (groups == q)
    names = np.asarray(ids)[mask]
    n = len(names)
    if n < MIN_GROUP_FORMATION_N:
        return None
    w = 1.0 / n
    return {str(s): w for s in names}

def traded_notional(new_w, old_w):
    """
    One-way traded notional = sum_i |w_new - w_old|.

    For a fully-invested long-only leg:
      first establishment = 1
      complete replacement = 2

    Step-3 H-L uses one unit long Q5 and one unit short Q1.
    Therefore total strategy traded notional is:
      long trade + short trade.
    """
    if old_w is None:
        return float(sum(abs(v) for v in new_w.values()))

    names = set(new_w) | set(old_w)
    return float(sum(
        abs(new_w.get(s, 0.0) - old_w.get(s, 0.0))
        for s in names
    ))

def formation_universe(g, scope, method, is_step2_common):
    if scope == COMMON and not is_step2_common:
        return None

    network_ok = g["network_feature_complete"].fillna(False).astype(bool)
    control_ok = (
        network_ok
        & g["control_complete"].fillna(False).astype(bool)
        & g["control_numeric_finite"].fillna(False).astype(bool)
    )

    if scope == COMMON:
        # Both methods share the neutral current-information universe.
        return g.loc[control_ok].copy().reset_index(drop=True)

    if method == RAW:
        return g.loc[network_ok].copy().reset_index(drop=True)

    return g.loc[control_ok].copy().reset_index(drop=True)

def build_turnover(panel, step2_common_dates):
    """
    Reconstruct Step-3 formation weights using CURRENT information only.

    Turnover state advances on every eligible formation date, even if
    that date later has no valid realized portfolio return. This avoids
    understating trading between evaluation dates.
    """
    prev = {}
    rows = []

    for (window, date), g in panel.groupby(
        ["window", "analysis_date"], sort=True
    ):
        date = pd.Timestamp(date)
        is_common = date in step2_common_dates

        for scope in [NATIVE, COMMON]:
            for method in [RAW, NEUTRAL]:
                u = formation_universe(
                    g, scope, method, is_common
                )
                if u is None or len(u) < MIN_FORMATION_N:
                    continue

                if method == RAW:
                    score_mat = u[FACTORS].to_numpy(float)
                else:
                    score_mat = neutral_scores(u)
                    if score_mat is None:
                        continue

                ids = u["security_id"].astype(str).to_numpy()

                for j, factor in enumerate(FACTORS):
                    score = score_mat[:, j]
                    if (
                        not np.isfinite(score).all()
                        or np.std(score, ddof=1) <= EPS
                    ):
                        continue

                    groups = quintile(score)
                    q1 = leg_weights(ids, groups, 1)
                    q5 = leg_weights(ids, groups, 5)
                    if q1 is None or q5 is None:
                        continue

                    key = (scope, method, int(window), factor)
                    old = prev.get(key)

                    q1_trade = traded_notional(
                        q1, None if old is None else old["q1"]
                    )
                    q5_trade = traded_notional(
                        q5, None if old is None else old["q5"]
                    )

                    rows.append({
                        "sample_scope": scope,
                        "method": method,
                        "window": int(window),
                        "analysis_date": date,
                        "factor": factor,
                        "q1_formation_n": len(q1),
                        "q5_formation_n": len(q5),
                        "q1_one_way_trade_notional": q1_trade,
                        "q5_one_way_trade_notional": q5_trade,
                        "total_one_way_trade_notional": q1_trade + q5_trade,
                        "half_l1_turnover": 0.5 * (q1_trade + q5_trade),
                    })

                    prev[key] = {"q1": q1, "q5": q5}

    return pd.DataFrame(rows)

# ============================================================
# 5. HAC / BH helpers
# ============================================================

def nw_mean_test(values, lag=3):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 3:
        return np.nan, np.nan, np.nan

    mu = x.mean()
    u = x - mu
    L = min(int(lag), T - 1)
    lr = np.dot(u, u) / T

    for ell in range(1, L + 1):
        w = 1.0 - ell / (L + 1.0)
        lr += 2.0 * w * np.dot(u[ell:], u[:-ell]) / T

    se = math.sqrt(max(lr, 0.0) / T)
    if not np.isfinite(se) or se <= EPS:
        return float(mu), np.nan, np.nan

    t = mu / se
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return float(mu), float(t), float(p)

def bh_qvalues(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    good = np.isfinite(p)
    if not good.any():
        return q

    idx = np.where(good)[0]
    pv = p[idx]
    order = np.argsort(pv)
    s = pv[order]
    m = len(s)

    qq = s * m / np.arange(1, m + 1)
    qq = np.minimum.accumulate(qq[::-1])[::-1]
    q[idx[order]] = np.clip(qq, 0, 1)
    return q

# ============================================================
# 6. Cost application and summaries
# ============================================================

def add_cost_columns(x):
    out = x.copy()

    gross = out["high_minus_low"].to_numpy(float)
    trade = out["total_one_way_trade_notional"].to_numpy(float)

    for bps in COST_BPS:
        cost = trade * (bps / 10000.0)

        out[f"cost_{bps}bps"] = cost

        # Canonical Step-3 direction: Q5 - Q1.
        out[f"net_hl_{bps}bps"] = gross - cost

        # Mechanical reverse direction, reported for reference only.
        # No factor is selected or sign-flipped.
        out[f"net_lh_{bps}bps"] = -gross - cost

    return out

def build_annual(monthly):
    x = monthly.copy()
    x["year"] = x["analysis_date"].dt.year.astype(int)

    value_cols = [
        "high_minus_low",
        "total_one_way_trade_notional",
        "half_l1_turnover",
    ]

    for bps in COST_BPS:
        value_cols += [
            f"net_hl_{bps}bps",
            f"net_lh_{bps}bps",
        ]

    return (
        x.groupby(
            ["sample_scope", "method", "window", "factor", "year"],
            as_index=False,
        )[value_cols]
        .mean()
        .sort_values(
            ["sample_scope", "method", "window", "factor", "year"]
        )
        .reset_index(drop=True)
    )

def build_summary(monthly):
    rows = []

    for key, g in monthly.groupby(
        ["sample_scope", "method", "window", "factor"],
        sort=True,
    ):
        scope, method, window, factor = key
        g = g.sort_values("analysis_date")

        gross = g["high_minus_low"].to_numpy(float)
        trade = g["total_one_way_trade_notional"].to_numpy(float)

        mu_g, t_g, p_g = nw_mean_test(gross, NW_LAG)

        row = {
            "sample_scope": scope,
            "method": method,
            "window": int(window),
            "factor": factor,
            "date_n": len(g),
            "mean_gross_hl": float(np.mean(gross)),
            "median_gross_hl": float(np.median(gross)),
            "positive_gross_hl_month_share": float(np.mean(gross > 0)),
            "mean_one_way_trade_notional": float(np.mean(trade)),
            "median_one_way_trade_notional": float(np.median(trade)),
            "p90_one_way_trade_notional": float(np.quantile(trade, 0.90)),
            "mean_half_l1_turnover": float(g["half_l1_turnover"].mean()),
            "annualized_one_way_trade_notional": float(
                PERIODS_PER_YEAR * np.mean(trade)
            ),
            "hac_t_gross_hl": t_g,
            "hac_p_gross_hl": p_g,
        }

        mean_trade = np.mean(trade)
        row["break_even_bps_hl"] = (
            10000.0 * np.mean(gross) / mean_trade
            if mean_trade > EPS else np.nan
        )
        row["break_even_bps_lh"] = (
            -10000.0 * np.mean(gross) / mean_trade
            if mean_trade > EPS else np.nan
        )

        for bps in COST_BPS:
            for label in ["hl", "lh"]:
                col = f"net_{label}_{bps}bps"
                r = g[col].to_numpy(float)

                mu, t, p = nw_mean_test(r, NW_LAG)
                sd = np.std(r, ddof=1)

                row[f"mean_{col}"] = float(np.mean(r))
                row[f"positive_{col}_month_share"] = float(np.mean(r > 0))
                row[f"annualized_{col}"] = float(PERIODS_PER_YEAR * np.mean(r))
                row[f"sharpe_{col}"] = (
                    math.sqrt(PERIODS_PER_YEAR) * np.mean(r) / sd
                    if np.isfinite(sd) and sd > EPS
                    else np.nan
                )
                row[f"hac_t_{col}"] = t
                row[f"hac_p_{col}"] = p

        rows.append(row)

    out = pd.DataFrame(rows)

    # BH q-values for canonical and reverse net returns,
    # separately by scope x method x cost.
    for bps in COST_BPS:
        for label in ["hl", "lh"]:
            qcol = f"bh_q_net_{label}_{bps}bps"
            out[qcol] = np.nan

            for scope in [NATIVE, COMMON]:
                for method in [RAW, NEUTRAL]:
                    mask = (
                        (out["sample_scope"] == scope)
                        & (out["method"] == method)
                    )
                    pcol = f"hac_p_net_{label}_{bps}bps"
                    out.loc[mask, qcol] = bh_qvalues(
                        out.loc[mask, pcol].to_numpy(float)
                    )

    return (
        out.sort_values(
            ["sample_scope", "method", "window", "factor"]
        )
        .reset_index(drop=True)
    )

# ============================================================
# 7. Main
# ============================================================

def main():
    print("=" * 84)
    print("M1 - Research Day 12 - Step 4")
    print("Turnover and Transaction-Cost Stress Test")
    print("=" * 84)

    for p in [
        PANEL_PATH,
        STEP1_QA,
        STEP3_MONTHLY,
        STEP3_QA,
        STEP3_COMMON,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    if not as_bool(load_qa(STEP1_QA).get("all_formal_qa_pass", False)):
        raise RuntimeError("Step 1 QA has not passed.")

    if not as_bool(load_qa(STEP3_QA).get("all_formal_qa_pass", False)):
        raise RuntimeError("Step 3 QA has not passed.")

    # --------------------------------------------------------
    # Step-3 frozen common-date calendar
    # --------------------------------------------------------

    cm = pd.read_csv(STEP3_COMMON)
    cm["analysis_date"] = pd.to_datetime(cm["analysis_date"])

    step2_common_dates = set(cm["analysis_date"].tolist())
    portfolio_common_dates = set(
        cm.loc[
            cm["portfolio_common_date"].map(as_bool),
            "analysis_date",
        ].tolist()
    )

    # --------------------------------------------------------
    # Read current-information formation variables only
    # --------------------------------------------------------

    columns = [
        "security_id",
        "analysis_date",
        "window",
        "network_feature_complete",
        "control_complete",
        "control_numeric_finite",
        INDUSTRY,
        *CONTROL_NUMERIC,
        *FACTORS,
    ]

    panel = pd.read_parquet(
        PANEL_PATH,
        columns=columns,
    )

    panel["analysis_date"] = pd.to_datetime(panel["analysis_date"])
    panel["window"] = panel["window"].astype(int)

    dup = int(
        panel[
            ["security_id", "analysis_date", "window"]
        ].duplicated().sum()
    )

    if dup:
        raise RuntimeError(
            f"Duplicate Step-1 panel key count: {dup}"
        )

    # --------------------------------------------------------
    # Reconstruct formation weights and turnover
    # --------------------------------------------------------

    print("Reconstructing frozen Step-3 portfolio memberships...")
    turnover = build_turnover(
        panel,
        step2_common_dates,
    )

    # --------------------------------------------------------
    # Load realized Step-3 gross portfolio outcomes
    # --------------------------------------------------------

    s3 = pd.read_csv(STEP3_MONTHLY)
    s3["analysis_date"] = pd.to_datetime(s3["analysis_date"])

    required = [
        "sample_scope",
        "method",
        "window",
        "analysis_date",
        "factor",
        "portfolio_valid",
        "q1_formation_n",
        "q5_formation_n",
        "high_minus_low",
    ]

    missing = [c for c in required if c not in s3.columns]
    if missing:
        raise RuntimeError(
            f"Step-3 monthly file missing columns: {missing}"
        )

    valid = s3.loc[
        s3["portfolio_valid"].map(as_bool)
    ].copy()

    # COMMON_DATE final evaluation calendar is the 95-date
    # frozen portfolio common calendar from Step 3.
    valid = valid.loc[
        (valid["sample_scope"] != COMMON)
        |
        valid["analysis_date"].isin(portfolio_common_dates)
    ].copy()

    merge_key = [
        "sample_scope",
        "method",
        "window",
        "analysis_date",
        "factor",
    ]

    monthly = valid.merge(
        turnover,
        on=merge_key,
        how="left",
        validate="one_to_one",
        suffixes=("_step3", "_reconstructed"),
    )

    # Exact formation replication audit.
    q1_mismatch = int(
        (
            monthly["q1_formation_n_step3"]
            != monthly["q1_formation_n_reconstructed"]
        ).fillna(True).sum()
    )

    q5_mismatch = int(
        (
            monthly["q5_formation_n_step3"]
            != monthly["q5_formation_n_reconstructed"]
        ).fillna(True).sum()
    )

    missing_turnover_n = int(
        monthly["total_one_way_trade_notional"].isna().sum()
    )

    monthly = add_cost_columns(monthly)

    annual = build_annual(monthly)
    summary = build_summary(monthly)

    # --------------------------------------------------------
    # Formal QA
    # --------------------------------------------------------

    expected_summary_rows = (
        2 * 2 * len(WINDOWS) * len(FACTORS)
    )

    common_rows = monthly.loc[
        monthly["sample_scope"] == COMMON
    ]

    common_date_counts = (
        common_rows.groupby(
            ["method", "window", "factor"]
        )["analysis_date"]
        .nunique()
    )

    common_identical_dates = bool(
        len(common_date_counts)
        == 2 * len(WINDOWS) * len(FACTORS)
        and
        (common_date_counts == len(portfolio_common_dates)).all()
    )

    nonfinite_core = int(
        (
            ~np.isfinite(
                monthly[
                    [
                        "high_minus_low",
                        "total_one_way_trade_notional",
                        "half_l1_turnover",
                    ]
                ].to_numpy(float)
            )
        ).sum()
    )

    negative_trade_n = int(
        (
            monthly["total_one_way_trade_notional"] < -EPS
        ).sum()
    )

    qa = {
        "step1_formal_qa_pass": True,
        "step3_formal_qa_pass": True,
        "frozen_factor_count": len(FACTORS),
        "window_count": len(WINDOWS),
        "cost_grid_bps": "|".join(map(str, COST_BPS)),
        "portfolio_common_date_count": len(portfolio_common_dates),
        "monthly_evaluation_row_count": len(monthly),
        "missing_turnover_row_count": missing_turnover_n,
        "q1_formation_count_mismatch_count": q1_mismatch,
        "q5_formation_count_mismatch_count": q5_mismatch,
        "common_cells_share_identical_dates": common_identical_dates,
        "summary_row_count": len(summary),
        "expected_summary_row_count": expected_summary_rows,
        "nonfinite_core_metric_count": nonfinite_core,
        "negative_trade_notional_count": negative_trade_n,
        "turnover_definition": "ONE_WAY_SUM_ABS_WEIGHT_CHANGE",
        "half_l1_turnover_reported": True,
        "initial_position_establishment_charged": True,
        "cost_applied_to_one_way_traded_notional": True,
        "primary_spread_definition": "Q5_MINUS_Q1",
        "reverse_direction_reported_mechanically": True,
        "factor_sign_flipped": False,
        "factor_selected_from_results": False,
        "window_selected_from_results": False,
        "cost_level_selected_from_results": False,
        "future_return_used_for_formation": False,
        "liquidity_filter_applied": False,
        "capacity_filter_applied": False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa["step1_formal_qa_pass"]
        and qa["step3_formal_qa_pass"]
        and qa["frozen_factor_count"] == 9
        and qa["window_count"] == 3
        and qa["missing_turnover_row_count"] == 0
        and qa["q1_formation_count_mismatch_count"] == 0
        and qa["q5_formation_count_mismatch_count"] == 0
        and qa["common_cells_share_identical_dates"]
        and qa["summary_row_count"] == expected_summary_rows
        and qa["nonfinite_core_metric_count"] == 0
        and qa["negative_trade_notional_count"] == 0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    atomic_csv(monthly, MONTHLY_OUT)
    atomic_csv(annual, ANNUAL_OUT)
    atomic_csv(summary, SUMMARY_OUT)

    atomic_csv(
        pd.DataFrame(
            [{"qa_name": k, "qa_value": v} for k, v in qa.items()]
        ),
        QA_OUT,
    )

    metadata = {
        "research_day": 12,
        "step": "Step4_Turnover_and_Transaction_Cost_Stress_Test",
        "input_panel": str(PANEL_PATH),
        "input_step3_monthly": str(STEP3_MONTHLY),
        "windows": WINDOWS,
        "frozen_factors": FACTORS,
        "cost_grid_bps": COST_BPS,
        "portfolio_weighting": "EQUAL_WEIGHT",
        "turnover_definition": (
            "For each Q1/Q5 leg, one-way traded notional equals "
            "sum_i |w_i,t - w_i,t-1|. The first establishment is "
            "charged from zero holdings. Strategy trade is Q1+Q5."
        ),
        "cost_rule": (
            "cost_t(c) = one_way_trade_notional_t * c / 10000. "
            "Canonical net H-L = (Q5-Q1) - cost. A mechanical "
            "reverse L-H series = -(Q5-Q1) - cost is also reported "
            "for reference, without selecting a direction."
        ),
        "break_even_rule": (
            "break_even_bps = 10000 * mean(gross spread) / "
            "mean(one-way traded notional), separately for H-L "
            "and the mechanical reverse L-H."
        ),
        "common_date_rule": (
            "COMMON_DATE uses the 95-date portfolio-common calendar "
            "frozen in Step 3 for realized evaluation. Turnover state "
            "is advanced on all 126 Step-2 common formation dates, "
            "so skipped evaluation dates do not mechanically suppress "
            "trading between evaluated dates."
        ),
        "research_boundary": (
            "No liquidity or capacity filter is applied in Step 4. "
            "Those implementation constraints remain reserved for Step 5."
        ),
        "formal_qa": qa,
    }

    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print("Turnover / cost summary:")
    show = [
        "sample_scope",
        "method",
        "window",
        "factor",
        "date_n",
        "mean_gross_hl",
        "mean_one_way_trade_notional",
        "break_even_bps_hl",
        "break_even_bps_lh",
        "mean_net_hl_20bps",
        "mean_net_lh_20bps",
    ]
    print(summary[show].to_string(index=False))

    print()
    print("Formal QA:")
    for k, v in qa.items():
        print(f"  {k}: {v}")

    if not qa["all_formal_qa_pass"]:
        raise RuntimeError(
            "Day-12 Step-4 formal QA failed. "
            "Inspect day12_step4_qa.csv before proceeding."
        )

    print()
    print("=" * 84)
    print("DAY 12 STEP 4 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 84)


if __name__ == "__main__":
    main()
