from __future__ import annotations

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

# Day 8 -> physical M1_day8
STEP4_DIR = (
    OUT
    / "M1_day8"
    / "04_stage4_capacity_limits"
)

CAPACITY_PATH = (
    STEP4_DIR
    / "capacity_limit_rebalance_pair.csv"
)

STEP4_QA_PATH = (
    STEP4_DIR
    / "capacity_limit_qa.csv"
)

# Day 9 -> physical M1_day9
REGIME_DIR = (
    OUT
    / "M1_day9"
    / "02_stage2_pit_regime_classification"
)

REGIME_PARQUET = (
    REGIME_DIR
    / "pit_regime_panel.parquet"
)

REGIME_CSV = (
    REGIME_DIR
    / "pit_regime_panel.csv"
)

STEP2_QA_PATH = (
    REGIME_DIR
    / "pit_regime_qa.csv"
)

OUTPUT_DIR = (
    OUT
    / "M1_day9"
    / "05_stage5_regime_conditional_capacity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "regime_capacity_summary.csv"
)

CONTRAST_PATH = (
    OUTPUT_DIR
    / "regime_capacity_high_low_contrast.csv"
)

OVERALL_PATH = (
    OUTPUT_DIR
    / "regime_capacity_overall_summary.csv"
)

QA_PATH = (
    OUTPUT_DIR
    / "regime_capacity_qa.csv"
)

SOURCE_PATH = (
    OUTPUT_DIR
    / "regime_capacity_source_manifest.csv"
)

META_PATH = (
    OUTPUT_DIR
    / "day9_step5_metadata.json"
)


# ============================================================
# 1. Frozen design
# ============================================================

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

WINDOWS = [60, 120, 252]

REGIMES = {
    "NETWORK":
        "network_stress_regime",

    "RECONFIG":
        "network_reconfig_regime",
}

REGIME_ORDER = [
    "LOW",
    "MEDIUM",
    "HIGH",
]

EXPECTED_SPECS = (
    len(FACTORS)
    *
    len(WINDOWS)
)  # 27

HAC_LAGS = 3
SMALL_SAMPLE_N = 12
ZERO_TOL_CNY = 1e-8

# If automatic capacity-column detection is wrong,
# set the exact column name here.
CAPACITY_COL = None


# ============================================================
# 2. Utilities
# ============================================================

def save_csv(df, path):

    tmp = Path(
        str(path) + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        tmp,
        path,
    )


def parse_bool(x):

    return (
        str(x)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


def read_qa(path):

    q = pd.read_csv(path)

    return dict(
        zip(
            q["qa_name"],
            q["qa_value"],
        )
    )


# ============================================================
# 3. NumPy Newey-West
# ============================================================

def newey_west_ols(
    y,
    X,
    maxlags=3,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    X = np.asarray(
        X,
        dtype=float,
    )

    inv = np.linalg.pinv(
        X.T @ X
    )

    beta = (
        inv
        @ X.T
        @ y
    )

    resid = (
        y
        -
        X @ beta
    )

    xu = (
        X
        *
        resid[:, None]
    )

    S = (
        xu.T
        @ xu
    )

    L = min(
        maxlags,
        len(y) - 1,
    )

    for lag in range(
        1,
        L + 1,
    ):

        w = (
            1.0
            -
            lag / (L + 1.0)
        )

        G = (
            xu[lag:].T
            @
            xu[:-lag]
        )

        S += (
            w
            *
            (
                G
                +
                G.T
            )
        )

    cov = (
        inv
        @ S
        @ inv
    )

    return beta, cov


# ============================================================
# 4. Load frozen pair capacity
# ============================================================

def detect_capacity_column(df):

    if CAPACITY_COL is not None:

        if CAPACITY_COL not in df.columns:
            raise RuntimeError(
                f"CAPACITY_COL not found: "
                f"{CAPACITY_COL}"
            )

        return CAPACITY_COL

    preferred = [
        "capacity_limit_cny",
        "capacity_aum_cny",
        "capacity_cny",
        "capacity_limit",
        "capacity_aum",
    ]

    for col in preferred:

        if col in df.columns:
            return col

    candidates = [
        c
        for c in df.columns
        if (
            "capacity" in c.lower()
            and
            (
                "cny" in c.lower()
                or
                "aum" in c.lower()
                or
                "limit" in c.lower()
            )
            and
            "status" not in c.lower()
            and
            "target" not in c.lower()
        )
    ]

    numeric = [
        c
        for c in candidates
        if pd.api.types.is_numeric_dtype(
            df[c]
        )
    ]

    if len(numeric) != 1:

        raise RuntimeError(
            "Cannot uniquely detect capacity column.\n"
            f"Candidates: {numeric}\n"
            "Set CAPACITY_COL manually."
        )

    return numeric[0]


def load_capacity():

    if not CAPACITY_PATH.exists():

        raise FileNotFoundError(
            CAPACITY_PATH
        )

    df = pd.read_csv(
        CAPACITY_PATH
    )

    required = [
        "analysis_date",
        "window",
        "factor_name",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Capacity file missing: "
            f"{missing}"
        )

    cap_col = detect_capacity_column(
        df
    )

    print(
        f"Capacity source : {CAPACITY_PATH}"
    )

    print(
        f"Capacity column : {cap_col}"
    )

    df["analysis_date"] = pd.to_datetime(
        df["analysis_date"],
        errors="raise",
    )

    df["window"] = pd.to_numeric(
        df["window"],
        errors="raise",
    ).astype(int)

    df["capacity_cny"] = pd.to_numeric(
        df[cap_col],
        errors="coerce",
    )

    df = df[
        df["factor_name"]
        .isin(FACTORS)
        &
        df["window"]
        .isin(WINDOWS)
    ].copy()

    key = [
        "analysis_date",
        "window",
        "factor_name",
    ]

    if df[key].duplicated().any():

        raise RuntimeError(
            "Duplicate pair capacity "
            "date-window-factor rows."
        )

    if df["capacity_cny"].isna().any():

        raise RuntimeError(
            "Missing frozen pair capacity."
        )

    if (
        df["capacity_cny"]
        <
        -ZERO_TOL_CNY
    ).any():

        raise RuntimeError(
            "Negative capacity detected."
        )

    df["capacity_yi"] = (
        df["capacity_cny"]
        /
        1e8
    )

    df["zero_capacity"] = (
        df["capacity_cny"]
        <=
        ZERO_TOL_CNY
    )

    return (
        df[
            key
            +
            [
                "capacity_cny",
                "capacity_yi",
                "zero_capacity",
            ]
        ],
        cap_col,
    )


# ============================================================
# 5. Load frozen PIT regimes
# ============================================================

def load_regimes():

    q2 = read_qa(
        STEP2_QA_PATH
    )

    q4 = read_qa(
        STEP4_QA_PATH
    )

    if not parse_bool(
        q2["all_formal_qa_pass"]
    ):

        raise RuntimeError(
            "Day 9 Step 2 QA is not PASS."
        )

    if not parse_bool(
        q4["all_formal_qa_pass"]
    ):

        raise RuntimeError(
            "Day 8 Step 4 QA is not PASS."
        )

    if REGIME_PARQUET.exists():

        x = pd.read_parquet(
            REGIME_PARQUET
        )

    else:

        x = pd.read_csv(
            REGIME_CSV
        )

    x["analysis_date"] = pd.to_datetime(
        x["analysis_date"],
        errors="raise",
    )

    x["window"] = pd.to_numeric(
        x["window"],
        errors="raise",
    ).astype(int)

    x = x[
        [
            "analysis_date",
            "window",
            *REGIMES.values(),
        ]
    ].copy()

    if x[
        [
            "analysis_date",
            "window",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate regime keys."
        )

    return x


# ============================================================
# 6. Regime descriptive statistics
# ============================================================

def summarize_regimes(
    data,
    regime_col,
    regime_type,
    scope,
):

    rows = []

    for (
        factor,
        window,
        regime
    ), g in data.groupby(
        [
            "factor_name",
            "window",
            regime_col,
        ],
        observed=True,
    ):

        if regime not in REGIME_ORDER:
            continue

        x = g[
            "capacity_yi"
        ].to_numpy()

        rows.append(
            {
                "regime_type":
                    regime_type,

                "sample_scope":
                    scope,

                "factor_name":
                    factor,

                "window":
                    int(window),

                "regime":
                    regime,

                "date_n":
                    int(len(x)),

                "mean_capacity_yi":
                    float(
                        np.mean(x)
                    ),

                "median_capacity_yi":
                    float(
                        np.median(x)
                    ),

                "p10_capacity_yi":
                    float(
                        np.quantile(
                            x,
                            0.10,
                        )
                    ),

                "p25_capacity_yi":
                    float(
                        np.quantile(
                            x,
                            0.25,
                        )
                    ),

                "p75_capacity_yi":
                    float(
                        np.quantile(
                            x,
                            0.75,
                        )
                    ),

                "p90_capacity_yi":
                    float(
                        np.quantile(
                            x,
                            0.90,
                        )
                    ),

                "zero_capacity_share":
                    float(
                        g[
                            "zero_capacity"
                        ]
                        .mean()
                    ),
            }
        )

    return rows


# ============================================================
# 7. HIGH - LOW capacity contrast
# ============================================================

def fit_high_low(
    g,
    regime_col,
    regime_type,
    scope,
):

    g = (
        g[
            g[regime_col]
            .isin(REGIME_ORDER)
        ]
        .sort_values(
            "analysis_date"
        )
        .copy()
    )

    D = (
        pd.get_dummies(
            g[regime_col],
            dtype=float,
        )
        .reindex(
            columns=REGIME_ORDER,
            fill_value=0.0,
        )
    )

    if (
        D["LOW"].sum() == 0
        or
        D["HIGH"].sum() == 0
    ):

        return None

    beta, cov = newey_west_ols(
        g["capacity_yi"],
        D,
        HAC_LAGS,
    )

    beta = pd.Series(
        beta,
        index=REGIME_ORDER,
    )

    cov = pd.DataFrame(
        cov,
        index=REGIME_ORDER,
        columns=REGIME_ORDER,
    )

    mean_diff = float(
        beta["HIGH"]
        -
        beta["LOW"]
    )

    var = (
        cov.loc[
            "HIGH",
            "HIGH",
        ]
        +
        cov.loc[
            "LOW",
            "LOW",
        ]
        -
        2
        *
        cov.loc[
            "HIGH",
            "LOW",
        ]
    )

    se = float(
        np.sqrt(
            max(
                float(var),
                0.0,
            )
        )
    )

    low = g[
        g[regime_col]
        ==
        "LOW"
    ]

    high = g[
        g[regime_col]
        ==
        "HIGH"
    ]

    t = (
        mean_diff / se
        if se > 0
        else np.nan
    )

    return {
        "regime_type":
            regime_type,

        "sample_scope":
            scope,

        "factor_name":
            g.iloc[0][
                "factor_name"
            ],

        "window":
            int(
                g.iloc[0]["window"]
            ),

        "low_n":
            int(len(low)),

        "high_n":
            int(len(high)),

        "mean_high_minus_low_yi":
            mean_diff,

        "median_high_minus_low_yi":
            float(
                high[
                    "capacity_yi"
                ].median()
                -
                low[
                    "capacity_yi"
                ].median()
            ),

        "zero_share_high_minus_low":
            float(
                high[
                    "zero_capacity"
                ].mean()
                -
                low[
                    "zero_capacity"
                ].mean()
            ),

        "hac_se_yi":
            se,

        "hac_t":
            t,

        "p_nominal":
            (
                math.erfc(
                    abs(t)
                    /
                    math.sqrt(2)
                )
                if np.isfinite(t)
                else np.nan
            ),

        "small_sample":
            min(
                len(low),
                len(high),
            )
            <
            SMALL_SAMPLE_N,
    }


# ============================================================
# 8. BH correction
# ============================================================

def add_bh_qvalues(df):

    df["q_bh"] = np.nan

    for _, idx in (
        df.groupby(
            [
                "regime_type",
                "sample_scope",
            ]
        )
        .groups.items()
    ):

        idx = list(idx)

        p = df.loc[
            idx,
            "p_nominal",
        ]

        valid = p.notna()

        if not valid.any():
            continue

        pv = p[
            valid
        ].to_numpy()

        order = np.argsort(
            pv
        )

        ranked = pv[
            order
        ]

        m = len(ranked)

        q = (
            ranked
            *
            m
            /
            np.arange(
                1,
                m + 1,
            )
        )

        q = np.minimum.accumulate(
            q[::-1]
        )[::-1]

        q = np.minimum(
            q,
            1.0,
        )

        back = np.empty_like(
            q
        )

        back[
            order
        ] = q

        df.loc[
            p[valid].index,
            "q_bh",
        ] = back

    return df


# ============================================================
# 9. Main analysis
# ============================================================

def run_analysis(
    capacity,
    regimes,
):

    data = capacity.merge(
        regimes,
        on=[
            "analysis_date",
            "window",
        ],
        how="left",
        validate="many_to_one",
    )

    summary_rows = []
    contrast_rows = []

    n_windows = len(
        WINDOWS
    )

    for regime_type, regime_col in (
        REGIMES.items()
    ):

        # Common dates:
        # all 3 W have a valid regime.
        c = (
            regimes[
                regimes[
                    regime_col
                ].notna()
            ]
            .groupby(
                "analysis_date"
            )["window"]
            .nunique()
        )

        common_dates = set(
            c[
                c == n_windows
            ].index
        )

        for scope in [
            "NATIVE",
            "COMMON_DATE",
        ]:

            x = data

            if scope == "COMMON_DATE":

                x = x[
                    x[
                        "analysis_date"
                    ]
                    .isin(
                        common_dates
                    )
                ]

            summary_rows.extend(
                summarize_regimes(
                    x,
                    regime_col,
                    regime_type,
                    scope,
                )
            )

            for _, g in x.groupby(
                [
                    "factor_name",
                    "window",
                ],
                observed=True,
                sort=False,
            ):

                row = fit_high_low(
                    g,
                    regime_col,
                    regime_type,
                    scope,
                )

                if row is not None:

                    contrast_rows.append(
                        row
                    )

    summary = pd.DataFrame(
        summary_rows
    )

    contrast = pd.DataFrame(
        contrast_rows
    )

    contrast = add_bh_qvalues(
        contrast
    )

    return (
        data,
        summary,
        contrast,
    )


# ============================================================
# 10. Main
# ============================================================

def main():

    print("=" * 72)
    print("M1 - Research Day 9 - Step 5")
    print("Regime-Conditional Capacity Validation")
    print("=" * 72)

    capacity, cap_col = (
        load_capacity()
    )

    regimes = load_regimes()

    print(
        f"Frozen capacity rows: "
        f"{len(capacity):,}"
    )

    (
        merged,
        summary,
        contrast,
    ) = run_analysis(
        capacity,
        regimes,
    )

    save_csv(
        summary,
        SUMMARY_PATH,
    )

    save_csv(
        contrast,
        CONTRAST_PATH,
    )

    # --------------------------------------------------------
    # Overall descriptive summary across frozen 27 specs
    # --------------------------------------------------------

    overall = (
        summary.groupby(
            [
                "regime_type",
                "sample_scope",
                "regime",
            ],
            as_index=False,
        )
        .agg(
            specification_n=(
                "factor_name",
                "size",
            ),

            avg_mean_capacity_yi=(
                "mean_capacity_yi",
                "mean",
            ),

            avg_median_capacity_yi=(
                "median_capacity_yi",
                "mean",
            ),

            avg_p10_capacity_yi=(
                "p10_capacity_yi",
                "mean",
            ),

            avg_zero_capacity_share=(
                "zero_capacity_share",
                "mean",
            ),
        )
    )

    save_csv(
        overall,
        OVERALL_PATH,
    )

    save_csv(
        pd.DataFrame(
            [
                {
                    "source":
                        "Day8_Step4_frozen_pair_capacity",

                    "path":
                        str(CAPACITY_PATH),

                    "capacity_column":
                        cap_col,

                    "primary_definition":
                        (
                            "ADV20 + 5% participation + "
                            "1-day + 95% mean executable"
                        ),
                },

                {
                    "source":
                        "Day9_Step2_frozen_PIT_regime",

                    "path":
                        str(
                            REGIME_PARQUET
                            if REGIME_PARQUET.exists()
                            else REGIME_CSV
                        ),

                    "capacity_column":
                        "",

                    "primary_definition":
                        "Frozen expanding-history PIT regimes",
                },
            ]
        ),
        SOURCE_PATH,
    )

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    spec_n = int(
        capacity[
            [
                "factor_name",
                "window",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    duplicate_n = int(
        capacity[
            [
                "analysis_date",
                "window",
                "factor_name",
            ]
        ]
        .duplicated()
        .sum()
    )

    missing_regime_n = {
        name:
            int(
                merged[col]
                .isna()
                .sum()
            )
        for name, col
        in REGIMES.items()
    }

    qa = {
        "day8_step4_formal_qa_pass":
            True,

        "day9_step2_formal_qa_pass":
            True,

        "expected_specification_count":
            EXPECTED_SPECS,

        "actual_specification_count":
            spec_n,

        "capacity_row_count":
            int(
                len(capacity)
            ),

        "duplicate_capacity_key_count":
            duplicate_n,

        "network_unclassified_row_count":
            missing_regime_n[
                "NETWORK"
            ],

        "reconfig_unclassified_row_count":
            missing_regime_n[
                "RECONFIG"
            ],

        "summary_row_count":
            int(
                len(summary)
            ),

        "contrast_row_count":
            int(
                len(contrast)
            ),

        "small_sample_contrast_count":
            int(
                contrast[
                    "small_sample"
                ].sum()
            ),

        "capacity_reestimated":
            False,

        "capacity_definition_changed":
            False,

        "adv_definition_changed":
            False,

        "participation_threshold_changed":
            False,

        "factor_selected_from_results":
            False,

        "window_selected_from_results":
            False,

        "regime_reestimated":
            False,

        "future_information_used":
            False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa[
            "actual_specification_count"
        ]
        ==
        EXPECTED_SPECS
        and
        qa[
            "duplicate_capacity_key_count"
        ]
        == 0
        and
        qa[
            "summary_row_count"
        ]
        > 0
        and
        qa[
            "contrast_row_count"
        ]
        > 0
    )

    save_csv(
        pd.DataFrame(
            [
                {
                    "qa_name":
                        k,

                    "qa_value":
                        v,
                }
                for k, v
                in qa.items()
            ]
        ),
        QA_PATH,
    )

    metadata = {
        "research_day":
            9,

        "step":
            (
                "Step5_Regime_Conditional_"
                "Capacity_Validation"
            ),

        "capacity_source":
            str(
                CAPACITY_PATH
            ),

        "capacity_reestimated":
            False,

        "primary_capacity_definition":
            (
                "Frozen Day-8 Step-4 pair capacity: "
                "ADV20, 5% participation, one-day "
                "execution, 95% mean executable share."
            ),

        "regime_dimensions":
            REGIMES,

        "sample_scopes": [
            "NATIVE",
            "COMMON_DATE",
        ],

        "primary_comparison":
            "HIGH_MINUS_LOW",

        "hac_lags":
            HAC_LAGS,

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
    print("Formal QA:")

    for k, v in qa.items():

        print(
            f"  {k}: {v}"
        )

    if not qa[
        "all_formal_qa_pass"
    ]:

        raise RuntimeError(
            "Day 9 Step 5 QA failed."
        )

    print()
    print("=" * 72)
    print("DAY 9 STEP 5 COMPLETE")
    print(
        f"Output: {OUTPUT_DIR}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()