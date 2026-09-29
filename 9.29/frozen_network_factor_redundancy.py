from pathlib import Path
import json
import os

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
DAY11 = ROOT / "output" / "M1_day11"

STEP1 = (
    DAY11
    / "01_stage1_frozen_prediction_panel"
)

PANEL = STEP1 / "frozen_prediction_panel.parquet"
STEP1_QA = STEP1 / "day11_step1_qa.csv"
STEP1_META = STEP1 / "day11_step1_metadata.json"

OUTDIR = (
    DAY11
    / "02_stage2_factor_redundancy"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

DATE_QA_PATH = (
    OUTDIR
    / "redundancy_date_window_qa.csv"
)

PAIR_DATE_PATH = (
    OUTDIR
    / "factor_pair_spearman_by_date.csv"
)

PAIR_SUMMARY_PATH = (
    OUTDIR
    / "factor_pair_redundancy_summary.csv"
)

VIF_DATE_PATH = (
    OUTDIR
    / "factor_vif_by_date.csv"
)

VIF_SUMMARY_PATH = (
    OUTDIR
    / "factor_vif_summary.csv"
)

MULTI_DATE_PATH = (
    OUTDIR
    / "multivariate_redundancy_by_date.csv"
)

MULTI_SUMMARY_PATH = (
    OUTDIR
    / "multivariate_redundancy_summary.csv"
)

QA_PATH = (
    OUTDIR
    / "day11_step2_qa.csv"
)

META_PATH = (
    OUTDIR
    / "day11_step2_metadata.json"
)


# ============================================================
# 1. Frozen design
# ============================================================

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

FACTOR_ORDER = {
    f: i + 1
    for i, f in enumerate(FACTORS)
}

# Fixed before examining redundancy results.
MIN_CROSS_SECTION_N = 100

EPS = 1e-10


# ============================================================
# 2. Helpers
# ============================================================

def save(df, path):

    tmp = Path(str(path) + ".tmp")

    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(tmp, path)


def as_bool(x):

    return (
        str(x)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


def load_step1_qa():

    q = pd.read_csv(STEP1_QA)

    d = dict(
        zip(
            q["qa_name"],
            q["qa_value"],
        )
    )

    if not as_bool(
        d.get("all_formal_qa_pass", False)
    ):
        raise RuntimeError(
            "Day-11 Step-1 QA is not PASS."
        )

    return d


# ============================================================
# 3. Rank-standardize one cross-section
# ============================================================

def rank_standardize(g):

    x = g[FACTORS].copy()

    valid = (
        x.notna().all(axis=1)
        &
        np.isfinite(
            x.to_numpy(dtype=float)
        ).all(axis=1)
    )

    x = x.loc[valid]

    n = len(x)

    if n < MIN_CROSS_SECTION_N:
        return None, n, "INSUFFICIENT_COMPLETE_CASES"

    # Spearman framework:
    # Pearson correlation of within-date ranks.
    r = x.rank(
        axis=0,
        method="average",
        pct=True,
    )

    z = r.to_numpy(dtype=float)

    mean = z.mean(axis=0)
    sd = z.std(axis=0, ddof=1)

    constant = [
        FACTORS[j]
        for j in range(len(FACTORS))
        if (
            not np.isfinite(sd[j])
            or sd[j] <= EPS
        )
    ]

    if constant:
        return (
            None,
            n,
            "CONSTANT:" + "|".join(constant),
        )

    z = (
        z - mean
    ) / sd

    return z, n, "VALID"


# ============================================================
# 4. VIF
# ============================================================

def compute_vif(z):

    rows = []

    k = z.shape[1]

    for j in range(k):

        y = z[:, j]

        other = [
            x for x in range(k)
            if x != j
        ]

        X = z[:, other]

        beta = np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )[0]

        fitted = X @ beta

        rss = float(
            np.sum(
                (y - fitted) ** 2
            )
        )

        tss = float(
            np.sum(y ** 2)
        )

        if tss <= EPS:
            r2 = np.nan
            vif = np.nan
        else:
            r2 = 1.0 - rss / tss

            # Numerical protection only.
            r2 = min(
                max(r2, 0.0),
                1.0,
            )

            denom = 1.0 - r2

            vif = (
                np.inf
                if denom <= EPS
                else 1.0 / denom
            )

        rows.append(
            {
                "factor_name":
                    FACTORS[j],

                "factor_order":
                    FACTOR_ORDER[
                        FACTORS[j]
                    ],

                "r2_from_other_factors":
                    r2,

                "vif":
                    vif,
            }
        )

    return rows


# ============================================================
# 5. Multivariate diagnostics
# ============================================================

def multivariate_metrics(corr):

    eig = np.linalg.eigvalsh(
        corr
    )

    # Remove tiny negative numerical artifacts.
    eig = np.clip(
        eig,
        0.0,
        None,
    )

    eig_desc = eig[::-1]

    total = float(
        eig_desc.sum()
    )

    min_eig = float(
        eig_desc[-1]
    )

    max_eig = float(
        eig_desc[0]
    )

    condition = (
        np.inf
        if min_eig <= EPS
        else max_eig / min_eig
    )

    if total <= EPS:
        return None

    share = eig_desc / total

    cumulative = np.cumsum(
        share
    )

    pc80 = int(
        np.searchsorted(
            cumulative,
            0.80,
        )
        + 1
    )

    pc90 = int(
        np.searchsorted(
            cumulative,
            0.90,
        )
        + 1
    )

    # Participation-ratio effective dimension.
    effective_rank = (
        total ** 2
        /
        float(
            np.sum(
                eig_desc ** 2
            )
        )
    )

    return {
        "min_eigenvalue":
            min_eig,

        "max_eigenvalue":
            max_eig,

        "condition_number":
            condition,

        "pc1_variance_share":
            float(share[0]),

        "pc2_cumulative_share":
            float(
                share[:2].sum()
            ),

        "pc3_cumulative_share":
            float(
                share[:3].sum()
            ),

        "pc_n_80pct":
            pc80,

        "pc_n_90pct":
            pc90,

        "participation_effective_rank":
            float(effective_rank),
    }


# ============================================================
# 6. Compute date-window diagnostics
# ============================================================

def compute_diagnostics(df):

    pair_rows = []
    vif_rows = []
    multi_rows = []
    qa_rows = []

    grouped = df.groupby(
        [
            "window",
            "analysis_date",
        ],
        sort=True,
    )

    for (window, date), g in grouped:

        z, n_complete, status = (
            rank_standardize(g)
        )

        qa_rows.append(
            {
                "window": int(window),
                "analysis_date": date,

                "stock_n":
                    int(len(g)),

                "complete_9factor_n":
                    int(n_complete),

                "complete_share":
                    (
                        n_complete / len(g)
                        if len(g)
                        else np.nan
                    ),

                "status":
                    status,
            }
        )

        if status != "VALID":
            continue

        # ----------------------------------------------------
        # Spearman correlation matrix
        # ----------------------------------------------------

        corr = np.corrcoef(
            z,
            rowvar=False,
        )

        if not np.isfinite(corr).all():
            qa_rows[-1]["status"] = (
                "NUMERICAL_CORRELATION_FAILURE"
            )
            continue

        k = len(FACTORS)

        for i in range(k):
            for j in range(i + 1, k):

                rho = float(
                    corr[i, j]
                )

                pair_rows.append(
                    {
                        "window":
                            int(window),

                        "analysis_date":
                            date,

                        "complete_n":
                            int(n_complete),

                        "factor_i":
                            FACTORS[i],

                        "factor_j":
                            FACTORS[j],

                        "factor_i_order":
                            FACTOR_ORDER[
                                FACTORS[i]
                            ],

                        "factor_j_order":
                            FACTOR_ORDER[
                                FACTORS[j]
                            ],

                        "spearman_rho":
                            rho,

                        "abs_spearman_rho":
                            abs(rho),
                    }
                )

        # ----------------------------------------------------
        # VIF
        # ----------------------------------------------------

        for r in compute_vif(z):

            r.update(
                {
                    "window":
                        int(window),

                    "analysis_date":
                        date,

                    "complete_n":
                        int(n_complete),
                }
            )

            vif_rows.append(r)

        # ----------------------------------------------------
        # Eigenvalue / effective dimension
        # ----------------------------------------------------

        metrics = multivariate_metrics(
            corr
        )

        if metrics is None:
            qa_rows[-1]["status"] = (
                "EIGENVALUE_FAILURE"
            )
            continue

        metrics.update(
            {
                "window":
                    int(window),

                "analysis_date":
                    date,

                "complete_n":
                    int(n_complete),
            }
        )

        multi_rows.append(
            metrics
        )

    return (
        pd.DataFrame(pair_rows),
        pd.DataFrame(vif_rows),
        pd.DataFrame(multi_rows),
        pd.DataFrame(qa_rows),
    )


# ============================================================
# 7. Native / Common-Date scopes
# ============================================================

def add_scope(df, common_dates):

    native = df.copy()
    native["sample_scope"] = "NATIVE"

    common = df[
        df["analysis_date"].isin(
            common_dates
        )
    ].copy()

    common["sample_scope"] = (
        "COMMON_DATE"
    )

    return pd.concat(
        [native, common],
        ignore_index=True,
    )


# ============================================================
# 8. Aggregate pairwise correlations
# ============================================================

def summarize_pairs(df):

    def q10(s):
        return s.quantile(0.10)

    def q90(s):
        return s.quantile(0.90)

    out = (
        df.groupby(
            [
                "sample_scope",
                "window",
                "factor_i",
                "factor_j",
                "factor_i_order",
                "factor_j_order",
            ],
            as_index=False,
        )
        .agg(
            date_n=(
                "analysis_date",
                "nunique",
            ),

            mean_spearman=(
                "spearman_rho",
                "mean",
            ),

            median_spearman=(
                "spearman_rho",
                "median",
            ),

            q10_spearman=(
                "spearman_rho",
                q10,
            ),

            q90_spearman=(
                "spearman_rho",
                q90,
            ),

            median_abs_spearman=(
                "abs_spearman_rho",
                "median",
            ),

            share_abs_rho_ge_080=(
                "abs_spearman_rho",
                lambda s:
                    float(
                        (s >= 0.80).mean()
                    ),
            ),

            share_abs_rho_ge_090=(
                "abs_spearman_rho",
                lambda s:
                    float(
                        (s >= 0.90).mean()
                    ),
            ),
        )
    )

    return out.sort_values(
        [
            "sample_scope",
            "window",
            "factor_i_order",
            "factor_j_order",
        ]
    ).reset_index(drop=True)


# ============================================================
# 9. Aggregate VIF
# ============================================================

def summarize_vif(df):

    def p90(s):

        s = s.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        return (
            s.quantile(0.90)
            if len(s)
            else np.nan
        )

    out = (
        df.groupby(
            [
                "sample_scope",
                "window",
                "factor_order",
                "factor_name",
            ],
            as_index=False,
        )
        .agg(
            date_n=(
                "analysis_date",
                "nunique",
            ),

            median_vif=(
                "vif",
                lambda s:
                    s.replace(
                        [np.inf, -np.inf],
                        np.nan,
                    ).median(),
            ),

            p90_vif=(
                "vif",
                p90,
            ),

            max_finite_vif=(
                "vif",
                lambda s:
                    s.replace(
                        [np.inf, -np.inf],
                        np.nan,
                    ).max(),
            ),

            infinite_vif_date_n=(
                "vif",
                lambda s:
                    int(
                        np.isinf(
                            s.to_numpy(
                                dtype=float
                            )
                        ).sum()
                    ),
            ),

            share_vif_ge_5=(
                "vif",
                lambda s:
                    float(
                        (
                            s.to_numpy(
                                dtype=float
                            )
                            >= 5
                        ).mean()
                    ),
            ),

            share_vif_ge_10=(
                "vif",
                lambda s:
                    float(
                        (
                            s.to_numpy(
                                dtype=float
                            )
                            >= 10
                        ).mean()
                    ),
            ),
        )
    )

    return out.sort_values(
        [
            "sample_scope",
            "window",
            "factor_order",
        ]
    ).reset_index(drop=True)


# ============================================================
# 10. Aggregate multivariate diagnostics
# ============================================================

def summarize_multivariate(df):

    def finite_median(s):

        x = (
            s.replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        return (
            x.median()
            if len(x)
            else np.nan
        )

    def finite_p90(s):

        x = (
            s.replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        return (
            x.quantile(0.90)
            if len(x)
            else np.nan
        )

    return (
        df.groupby(
            [
                "sample_scope",
                "window",
            ],
            as_index=False,
        )
        .agg(
            date_n=(
                "analysis_date",
                "nunique",
            ),

            median_complete_n=(
                "complete_n",
                "median",
            ),

            median_condition_number=(
                "condition_number",
                finite_median,
            ),

            p90_condition_number=(
                "condition_number",
                finite_p90,
            ),

            singular_date_n=(
                "condition_number",
                lambda s:
                    int(
                        np.isinf(
                            s.to_numpy(
                                dtype=float
                            )
                        ).sum()
                    ),
            ),

            mean_pc1_variance_share=(
                "pc1_variance_share",
                "mean",
            ),

            median_pc1_variance_share=(
                "pc1_variance_share",
                "median",
            ),

            median_pc3_cumulative_share=(
                "pc3_cumulative_share",
                "median",
            ),

            median_pc_n_80pct=(
                "pc_n_80pct",
                "median",
            ),

            median_pc_n_90pct=(
                "pc_n_90pct",
                "median",
            ),

            median_effective_rank=(
                "participation_effective_rank",
                "median",
            ),
        )
    )


# ============================================================
# 11. Main
# ============================================================

def main():

    print("=" * 76)
    print("M1 - Research Day 11 - Step 2")
    print("Frozen Network-Factor Redundancy Diagnostic")
    print("=" * 76)

    # --------------------------------------------------------
    # Step-1 provenance / QA
    # --------------------------------------------------------

    step1_qa = load_step1_qa()

    with open(
        STEP1_META,
        "r",
        encoding="utf-8",
    ) as f:
        meta = json.load(f)

    if meta.get(
        "frozen_factors"
    ) != FACTORS:
        raise RuntimeError(
            "Step-1 frozen factor list does not "
            "match Step-2 frozen design."
        )

    if meta.get(
        "windows"
    ) != WINDOWS:
        raise RuntimeError(
            "Step-1 windows do not match."
        )

    # --------------------------------------------------------
    # Read only columns needed for redundancy diagnostics
    # --------------------------------------------------------

    cols = [
        "security_id",
        "analysis_date",
        "window",
    ] + FACTORS

    df = pd.read_parquet(
        PANEL,
        columns=cols,
    )

    df["analysis_date"] = pd.to_datetime(
        df["analysis_date"],
        errors="raise",
    )

    df["window"] = pd.to_numeric(
        df["window"],
        errors="raise",
    ).astype(int)

    df = df[
        df["window"].isin(WINDOWS)
    ].copy()

    key = [
        "security_id",
        "analysis_date",
        "window",
    ]

    if df[key].duplicated().any():
        raise RuntimeError(
            "Duplicate stock-date-window keys."
        )

    # --------------------------------------------------------
    # Main diagnostics
    # --------------------------------------------------------

    (
        pair,
        vif,
        multi,
        date_qa,
    ) = compute_diagnostics(df)

    valid_dates = (
        date_qa[
            date_qa["status"] == "VALID"
        ][
            [
                "analysis_date",
                "window",
            ]
        ]
    )

    nwin = (
        valid_dates
        .groupby("analysis_date")[
            "window"
        ]
        .nunique()
    )

    common_dates = set(
        nwin[
            nwin == len(WINDOWS)
        ].index
    )

    pair_scoped = add_scope(
        pair,
        common_dates,
    )

    vif_scoped = add_scope(
        vif,
        common_dates,
    )

    multi_scoped = add_scope(
        multi,
        common_dates,
    )

    # --------------------------------------------------------
    # Summary outputs
    # --------------------------------------------------------

    pair_summary = summarize_pairs(
        pair_scoped
    )

    vif_summary = summarize_vif(
        vif_scoped
    )

    multi_summary = (
        summarize_multivariate(
            multi_scoped
        )
        .sort_values(
            [
                "sample_scope",
                "window",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save(
        date_qa,
        DATE_QA_PATH,
    )

    save(
        pair_scoped,
        PAIR_DATE_PATH,
    )

    save(
        pair_summary,
        PAIR_SUMMARY_PATH,
    )

    save(
        vif_scoped,
        VIF_DATE_PATH,
    )

    save(
        vif_summary,
        VIF_SUMMARY_PATH,
    )

    save(
        multi_scoped,
        MULTI_DATE_PATH,
    )

    save(
        multi_summary,
        MULTI_SUMMARY_PATH,
    )

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    valid_n = int(
        (
            date_qa["status"] == "VALID"
        ).sum()
    )

    unexpected_failure_n = int(
        date_qa[
            ~date_qa["status"].isin(
                [
                    "VALID",
                    "INSUFFICIENT_COMPLETE_CASES",
                ]
            )
        ].shape[0]
    )

    expected_pairs = (
        len(FACTORS)
        * (len(FACTORS) - 1)
        // 2
    )

    native_pair_counts = (
        pair_summary[
            pair_summary[
                "sample_scope"
            ] == "NATIVE"
        ]
        .groupby("window")
        .size()
    )

    qa = {
        "step1_formal_qa_pass":
            True,

        "panel_row_count":
            int(len(df)),

        "duplicate_stock_date_window_count":
            int(
                df[key]
                .duplicated()
                .sum()
            ),

        "frozen_factor_count":
            len(FACTORS),

        "window_count":
            int(
                df["window"].nunique()
            ),

        "date_window_count":
            int(len(date_qa)),

        "valid_date_window_count":
            valid_n,

        "common_valid_date_count":
            int(len(common_dates)),

        "unexpected_failure_count":
            unexpected_failure_n,

        "expected_factor_pair_count_per_window":
            expected_pairs,

        "all_native_windows_have_36_pairs":
            bool(
                all(
                    native_pair_counts.get(
                        w, 0
                    )
                    == expected_pairs
                    for w in WINDOWS
                )
            ),

        "network_features_reestimated":
            False,

        "network_factor_deleted":
            False,

        "network_factor_combined":
            False,

        "factor_selected_from_results":
            False,

        "window_selected_from_results":
            False,

        "future_target_read":
            False,

        "future_information_used":
            False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa[
            "duplicate_stock_date_window_count"
        ] == 0
        and
        qa["frozen_factor_count"] == 9
        and
        qa["window_count"] == 3
        and
        qa["valid_date_window_count"] > 0
        and
        qa["common_valid_date_count"] > 0
        and
        qa["unexpected_failure_count"] == 0
        and
        qa[
            "all_native_windows_have_36_pairs"
        ]
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
        "research_day": 11,

        "step":
            "Step2_Frozen_Network_Factor_Redundancy_Diagnostic",

        "input":
            str(PANEL),

        "frozen_factors":
            FACTORS,

        "windows":
            WINDOWS,

        "minimum_cross_section_n":
            MIN_CROSS_SECTION_N,

        "correlation_metric":
            (
                "Within-date cross-sectional "
                "Spearman correlation."
            ),

        "vif_metric":
            (
                "VIF calculated by regressing each "
                "rank-standardized network factor on "
                "the other eight network factors."
            ),

        "multivariate_metrics": [
            "correlation-matrix condition number",
            "PC1 variance share",
            "PC3 cumulative share",
            "number of PCs explaining 80%",
            "number of PCs explaining 90%",
            "participation-ratio effective rank",
        ],

        "sample_scopes": [
            "NATIVE",
            "COMMON_DATE",
        ],

        "important_note":
            (
                "Redundancy diagnostics are descriptive. "
                "No factor is removed, combined, sign-flipped, "
                "or selected using these results."
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

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("Multivariate redundancy summary:")
    print(
        multi_summary.to_string(
            index=False
        )
    )

    print()
    print("Largest median |Spearman| pairs (Native):")

    top = (
        pair_summary[
            pair_summary[
                "sample_scope"
            ] == "NATIVE"
        ]
        .sort_values(
            [
                "window",
                "median_abs_spearman",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "window",
            group_keys=False,
        )
        .head(5)
    )

    print(
        top[
            [
                "window",
                "factor_i",
                "factor_j",
                "median_spearman",
                "median_abs_spearman",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Formal QA:")

    for k, v in qa.items():
        print(f"  {k}: {v}")

    if not qa[
        "all_formal_qa_pass"
    ]:
        raise RuntimeError(
            "Day-11 Step-2 formal QA failed."
        )

    print()
    print("=" * 76)
    print("DAY 11 STEP 2 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 76)


if __name__ == "__main__":
    main()