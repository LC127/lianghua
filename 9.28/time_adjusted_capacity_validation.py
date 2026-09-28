from pathlib import Path
import json
import math
import os

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
OUT = ROOT / "output"

# If needed, set the exact Step-1 directory here.
STEP1_DIR_OVERRIDE = None

CANDIDATES = [
    OUT / "M1_day10" / "01_stage1_secular_liquidity_panel",
    OUT / "M1_day9" / "01_stage1_secular_liquidity_panel",
]


def resolve_step1_dir():
    if STEP1_DIR_OVERRIDE is not None:
        d = Path(STEP1_DIR_OVERRIDE)
        if not (d / "calendar_liquidity_network_panel.csv").exists():
            raise FileNotFoundError(d)
        return d

    valid = [
        d for d in CANDIDATES
        if (
            (d / "calendar_liquidity_network_panel.csv").exists()
            and
            (d / "day10_step1_qa.csv").exists()
        )
    ]

    if len(valid) != 1:
        raise RuntimeError(
            "Cannot uniquely locate Day-10 Step-1 output.\n"
            f"Candidates found: {valid}\n"
            "Set STEP1_DIR_OVERRIDE explicitly."
        )

    return valid[0]


STEP1 = resolve_step1_dir()

INPUT = STEP1 / "calendar_liquidity_network_panel.csv"
STEP1_QA = STEP1 / "day10_step1_qa.csv"

OUTDIR = STEP1.parent / "02_stage2_time_adjusted_capacity"
OUTDIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUTDIR / "time_adjusted_capacity_results.csv"
ATTEN_PATH = OUTDIR / "time_adjusted_capacity_attenuation.csv"
QA_PATH = OUTDIR / "day10_step2_qa.csv"
META_PATH = OUTDIR / "day10_step2_metadata.json"


# ============================================================
# 1. Frozen design
# ============================================================

REGIME = "network_stress_regime"
Y = "median_pair_capacity_yi"

HAC_LAGS = 3
REGIMES = ["LOW", "MEDIUM", "HIGH"]
WINDOWS = [60, 120, 252]

# M3 is primary; M4 is liquidity robustness.
MODELS = {
    "M0_REGIME_ONLY": (),
    "M1_PLUS_MARKET_ADV":
        ("log_market_adv20",),
    "M2_PLUS_YEAR_FE":
        ("year_fe",),
    "M3_ADV_PLUS_YEAR_FE":
        ("log_market_adv20", "year_fe"),
    "M4_MEDIAN_ADV_PLUS_YEAR_FE":
        ("log_median_adv20", "year_fe"),
}

PRIMARY_MODEL = "M3_ADV_PLUS_YEAR_FE"


# ============================================================
# 2. Helpers
# ============================================================

def save(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def as_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


def newey_west(y, X, lags=3):
    y = np.asarray(y, float)
    X = np.asarray(X, float)

    inv = np.linalg.pinv(X.T @ X)
    beta = inv @ X.T @ y
    resid = y - X @ beta

    xu = X * resid[:, None]
    S = xu.T @ xu

    for lag in range(1, min(lags, len(y) - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = xu[lag:].T @ xu[:-lag]
        S += w * (G + G.T)

    cov = inv @ S @ inv
    return beta, cov, resid


def make_design(g, spec):
    X = pd.DataFrame(index=g.index)

    X["Intercept"] = 1.0
    X["MEDIUM"] = (g[REGIME] == "MEDIUM").astype(float)
    X["HIGH"] = (g[REGIME] == "HIGH").astype(float)

    if "log_market_adv20" in spec:
        X["log_market_adv20"] = g["log_market_adv20"]

    if "log_median_adv20" in spec:
        X["log_median_adv20"] = g["log_median_adv20"]

    if "year_fe" in spec:
        yd = pd.get_dummies(
            g["calendar_year"].astype(int).astype(str),
            prefix="Y",
            drop_first=True,
            dtype=float,
        )
        X = pd.concat([X, yd], axis=1)

    return X


# ============================================================
# 3. Fit one window/model
# ============================================================

def fit_one(g, scope, window, model, spec):
    g = g.sort_values("analysis_date").copy()

    Xdf = make_design(g, spec)
    X = Xdf.to_numpy(float)
    y = g[Y].to_numpy(float)

    rank = np.linalg.matrix_rank(X)

    if rank != X.shape[1]:
        raise RuntimeError(
            f"Rank-deficient design: "
            f"{scope}, W{window}, {model}"
        )

    beta, cov, resid = newey_west(
        y, X, HAC_LAGS
    )

    pos = {
        c: i
        for i, c in enumerate(Xdf.columns)
    }

    ih = pos["HIGH"]
    im = pos["MEDIUM"]

    se_h = np.sqrt(
        max(float(cov[ih, ih]), 0.0)
    )

    se_m = np.sqrt(
        max(float(cov[im, im]), 0.0)
    )

    beta_h = float(beta[ih])
    beta_m = float(beta[im])

    t_h = (
        beta_h / se_h
        if se_h > 0
        else np.nan
    )

    t_m = (
        beta_m / se_m
        if se_m > 0
        else np.nan
    )

    sst = float(
        ((y - y.mean()) ** 2).sum()
    )

    r2 = (
        1.0
        - float(resid @ resid) / sst
        if sst > 0
        else np.nan
    )

    year_sets = (
        g.groupby("calendar_year")[REGIME]
        .apply(set)
    )

    overlap_year_n = sum(
        ("HIGH" in s and "LOW" in s)
        for s in year_sets
    )

    counts = g[REGIME].value_counts()

    return {
        "sample_scope": scope,
        "window": int(window),
        "model": model,
        "is_primary": model == PRIMARY_MODEL,

        "n": int(len(g)),
        "low_n": int(counts.get("LOW", 0)),
        "medium_n": int(counts.get("MEDIUM", 0)),
        "high_n": int(counts.get("HIGH", 0)),
        "calendar_year_n":
            int(g["calendar_year"].nunique()),
        "high_low_same_year_n":
            int(overlap_year_n),

        # LOW is reference category.
        "medium_minus_low_yi": beta_m,
        "medium_minus_low_se": se_m,
        "medium_minus_low_t": t_m,

        "high_minus_low_yi": beta_h,
        "high_minus_low_se": se_h,
        "high_minus_low_t": t_h,
        "high_minus_low_p":
            math.erfc(abs(t_h) / math.sqrt(2))
            if np.isfinite(t_h)
            else np.nan,

        "r2": r2,
        "design_k": int(X.shape[1]),
        "design_rank": int(rank),
        "full_rank": rank == X.shape[1],
    }


# ============================================================
# 4. BH correction across W=60/120/252
# ============================================================

def add_bh(df):
    df["high_minus_low_q_bh"] = np.nan

    for _, idx in df.groupby(
        ["sample_scope", "model"]
    ).groups.items():

        idx = list(idx)
        p = df.loc[idx, "high_minus_low_p"]
        valid = p.notna()

        if not valid.any():
            continue

        pv = p[valid].to_numpy()
        order = np.argsort(pv)
        ranked = pv[order]
        m = len(ranked)

        q = ranked * m / np.arange(1, m + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.minimum(q, 1.0)

        back = np.empty_like(q)
        back[order] = q

        df.loc[
            p[valid].index,
            "high_minus_low_q_bh",
        ] = back

    return df


# ============================================================
# 5. Main
# ============================================================

def main():
    print("=" * 72)
    print("M1 - Research Day 10 - Step 2")
    print("Time-Adjusted Network-Stress Capacity Validation")
    print("=" * 72)

    # --------------------------------------------------------
    # Step-1 QA
    # --------------------------------------------------------

    q = pd.read_csv(STEP1_QA)
    q = dict(zip(q["qa_name"], q["qa_value"]))

    if not as_bool(q["all_formal_qa_pass"]):
        raise RuntimeError(
            "Day-10 Step-1 QA is not PASS."
        )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    df = pd.read_csv(INPUT)

    required = [
        "analysis_date",
        "window",
        REGIME,
        Y,
        "market_adv20_total_yi",
        "median_adv20_cny",
        "calendar_year",
        "capacity_factor_n",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    df["analysis_date"] = pd.to_datetime(
        df["analysis_date"],
        errors="raise",
    )

    df["window"] = pd.to_numeric(
        df["window"],
        errors="raise",
    ).astype(int)

    # Same estimation sample for all M0-M4.
    valid = (
        df[REGIME].isin(REGIMES)
        &
        df[Y].notna()
        &
        (df["market_adv20_total_yi"] > 0)
        &
        (df["median_adv20_cny"] > 0)
        &
        df["calendar_year"].notna()
    )

    x = df[valid].copy()

    x["log_market_adv20"] = np.log(
        x["market_adv20_total_yi"]
    )

    x["log_median_adv20"] = np.log(
        x["median_adv20_cny"]
    )

    if not (
        x["capacity_factor_n"] == 9
    ).all():
        raise RuntimeError(
            "Effective date-window rows are not "
            "all based on 9 frozen factors."
        )

    # --------------------------------------------------------
    # Common dates:
    # valid Network-Stress regime + capacity for all 3 W.
    # --------------------------------------------------------

    c = (
        x.groupby("analysis_date")["window"]
        .nunique()
    )

    common_dates = set(
        c[c == len(WINDOWS)].index
    )

    # --------------------------------------------------------
    # Estimation
    # --------------------------------------------------------

    rows = []

    for scope in ["NATIVE", "COMMON_DATE"]:

        z = (
            x
            if scope == "NATIVE"
            else x[
                x["analysis_date"].isin(
                    common_dates
                )
            ]
        )

        for window in WINDOWS:

            g = z[
                z["window"] == window
            ].copy()

            if g.empty:
                raise RuntimeError(
                    f"No observations: {scope}, W{window}"
                )

            for model, spec in MODELS.items():
                rows.append(
                    fit_one(
                        g,
                        scope,
                        window,
                        model,
                        spec,
                    )
                )

    result = add_bh(
        pd.DataFrame(rows)
    )

    save(
        result,
        RESULT_PATH,
    )

    # --------------------------------------------------------
    # Attenuation relative to M0
    # --------------------------------------------------------

    key = [
        "sample_scope",
        "window",
    ]

    base = result[
        result["model"] == "M0_REGIME_ONLY"
    ][
        key + ["high_minus_low_yi"]
    ].rename(
        columns={
            "high_minus_low_yi":
                "m0_high_minus_low_yi"
        }
    )

    attenuation = (
        result[
            result["model"]
            != "M0_REGIME_ONLY"
        ]
        .merge(
            base,
            on=key,
            validate="many_to_one",
        )
        .copy()
    )

    attenuation[
        "effect_ratio_to_m0"
    ] = (
        attenuation["high_minus_low_yi"]
        /
        attenuation["m0_high_minus_low_yi"]
    )

    attenuation[
        "attenuation_fraction"
    ] = (
        1.0
        -
        attenuation["effect_ratio_to_m0"]
    )

    attenuation[
        "sign_reversal_vs_m0"
    ] = (
        np.sign(
            attenuation["high_minus_low_yi"]
        )
        !=
        np.sign(
            attenuation["m0_high_minus_low_yi"]
        )
    )

    save(
        attenuation,
        ATTEN_PATH,
    )

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    expected_fit_n = (
        2
        * len(WINDOWS)
        * len(MODELS)
    )

    qa = {
        "step1_formal_qa_pass": True,

        "input_row_count":
            int(len(df)),

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
            bool(
                (x["capacity_factor_n"] == 9).all()
            ),

        "expected_fit_count":
            expected_fit_n,

        "actual_fit_count":
            int(len(result)),

        "full_rank_fit_count":
            int(result["full_rank"].sum()),

        "primary_model_fit_count":
            int(
                (
                    result["model"]
                    == PRIMARY_MODEL
                ).sum()
            ),

        "capacity_reestimated": False,
        "regime_reestimated": False,
        "liquidity_reestimated": False,
        "model_selected_from_results": False,
        "window_selected_from_results": False,
        "future_information_used": False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa[
            "effective_duplicate_date_window_count"
        ] == 0
        and
        qa[
            "all_effective_rows_use_9_factors"
        ]
        and
        qa["actual_fit_count"]
        == expected_fit_n
        and
        qa["full_rank_fit_count"]
        == expected_fit_n
        and
        qa["primary_model_fit_count"]
        == 6
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
            "Step2_Time_Adjusted_Network_Stress_Capacity",
        "input":
            str(INPUT),
        "dependent_variable":
            Y,
        "regime":
            REGIME,
        "reference_regime":
            "LOW",
        "primary_model":
            PRIMARY_MODEL,
        "models":
            MODELS,
        "hac_lags":
            HAC_LAGS,
        "sample_scopes":
            ["NATIVE", "COMMON_DATE"],
        "capacity_definition":
            (
                "Median frozen rebalance-level pair capacity "
                "across the 9 frozen factors within date-window."
            ),
        "interpretation":
            (
                "Validation of Network-Stress HIGH-minus-LOW "
                "capacity association after controlling for "
                "secular market-liquidity and calendar-year effects."
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
    print("Primary M3 results:")
    print(
        result[
            result["model"] == PRIMARY_MODEL
        ][
            [
                "sample_scope",
                "window",
                "n",
                "low_n",
                "high_n",
                "high_low_same_year_n",
                "high_minus_low_yi",
                "high_minus_low_t",
                "high_minus_low_q_bh",
                "r2",
            ]
        ].to_string(index=False)
    )

    print()
    print("Formal QA:")
    for k, v in qa.items():
        print(f"  {k}: {v}")

    if not qa["all_formal_qa_pass"]:
        raise RuntimeError(
            "Day-10 Step-2 QA failed."
        )

    print()
    print("=" * 72)
    print("DAY 10 STEP 2 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()