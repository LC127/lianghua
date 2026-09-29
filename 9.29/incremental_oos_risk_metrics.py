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
DAY11 = ROOT / "output" / "M1_day11"

STEP3 = (
    DAY11
    / "03_stage3_strict_oos_forecasting"
)

PRED_PATH = (
    STEP3
    / "oos_predictions.parquet"
)

FORECAST_SUMMARY_PATH = (
    STEP3
    / "oos_forecast_summary.csv"
)

STEP3_QA_PATH = (
    STEP3
    / "day11_step3_qa.csv"
)

STEP3_META_PATH = (
    STEP3
    / "day11_step3_metadata.json"
)

OUTDIR = (
    DAY11
    / "04_stage4_incremental_oos_metrics"
)
OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

MONTHLY_OUT = (
    OUTDIR
    / "monthly_oos_risk_metrics.csv"
)

ANNUAL_OUT = (
    OUTDIR
    / "annual_oos_risk_metrics.csv"
)

SUMMARY_OUT = (
    OUTDIR
    / "oos_incremental_risk_summary.csv"
)

EVAL_DATE_OUT = (
    OUTDIR
    / "oos_evaluation_date_manifest.csv"
)

QA_OUT = (
    OUTDIR
    / "day11_step4_qa.csv"
)

META_OUT = (
    OUTDIR
    / "day11_step4_metadata.json"
)


# ============================================================
# 1. Frozen design
# ============================================================

WINDOWS = [60, 120, 252]

TARGET = "future_idio_vol_annualized"

BASE_SCORE = "baseline_score"
NETWORK_SCORE = "network_score"

MIN_EVAL_N = 100

# Descriptive HAC inference for monthly Delta IC.
NW_LAG = 3

SCOPES = [
    "PRIMARY",
    "FULL_RANK",
]


# ============================================================
# 2. Helpers
# ============================================================

def save_csv(df, path):

    tmp = Path(str(path) + ".tmp")

    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        tmp,
        path,
    )


def as_bool(x):

    return (
        str(x)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


def load_qa(path):

    x = pd.read_csv(path)

    return dict(
        zip(
            x["qa_name"],
            x["qa_value"],
        )
    )


# ============================================================
# 3. Rank utilities
# ============================================================

def percentile_rank(x):

    return (
        pd.Series(x)
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(
            dtype=float
        )
    )


def rank_z(x):

    r = percentile_rank(x)

    if len(r) < 2:
        return None

    sd = np.std(
        r,
        ddof=1,
    )

    if (
        not np.isfinite(sd)
        or sd <= 1e-12
    ):
        return None

    return (
        r - np.mean(r)
    ) / sd


def spearman_corr(x, y):

    rx = percentile_rank(x)
    ry = percentile_rank(y)

    if (
        len(rx) < 2
        or
        np.std(rx, ddof=1) <= 1e-12
        or
        np.std(ry, ddof=1) <= 1e-12
    ):
        return np.nan

    return float(
        np.corrcoef(
            rx,
            ry,
        )[0, 1]
    )


# ============================================================
# 4. Newey-West mean inference
# ============================================================

def nw_mean_test(
    values,
    lag=3,
):

    """
    HAC inference for H0: E[value] = 0.

    Returns:
        mean
        HAC standard error
        t statistic
        two-sided normal p-value
        one-sided p-value for mean > 0
    """

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    T = len(x)

    if T < 3:
        return {
            "mean": np.nan,
            "se": np.nan,
            "t": np.nan,
            "p_two_sided": np.nan,
            "p_gt0": np.nan,
        }

    mu = float(
        np.mean(x)
    )

    u = x - mu

    L = min(
        int(lag),
        T - 1,
    )

    gamma0 = float(
        np.dot(u, u)
        / T
    )

    long_run = gamma0

    for ell in range(
        1,
        L + 1,
    ):

        weight = (
            1.0
            - ell / (L + 1.0)
        )

        gamma = float(
            np.dot(
                u[ell:],
                u[:-ell],
            )
            / T
        )

        long_run += (
            2.0
            * weight
            * gamma
        )

    long_run = max(
        long_run,
        0.0,
    )

    se = math.sqrt(
        long_run / T
    )

    if se <= 0:
        t_stat = np.nan
        p_two = np.nan
        p_gt0 = np.nan

    else:

        t_stat = (
            mu / se
        )

        # Normal approximation.
        p_two = math.erfc(
            abs(t_stat)
            / math.sqrt(2.0)
        )

        # H1: mean > 0
        p_gt0 = (
            0.5
            * math.erfc(
                t_stat
                / math.sqrt(2.0)
            )
        )

    return {
        "mean": mu,
        "se": se,
        "t": t_stat,
        "p_two_sided": p_two,
        "p_gt0": p_gt0,
    }


# ============================================================
# 5. Evaluate one Date × Window
# ============================================================

def evaluate_one_date(g):

    """
    Baseline and Network are evaluated on the exact
    same stock sample.
    """

    valid = (
        g[
            [
                TARGET,
                BASE_SCORE,
                NETWORK_SCORE,
            ]
        ]
        .notna()
        .all(axis=1)
    )

    sub = (
        g.loc[valid]
        .copy()
        .reset_index(drop=True)
    )

    n = len(sub)

    if n < MIN_EVAL_N:
        return None

    y = sub[
        TARGET
    ].to_numpy(
        dtype=float
    )

    pb = sub[
        BASE_SCORE
    ].to_numpy(
        dtype=float
    )

    pn = sub[
        NETWORK_SCORE
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # 5.1 OOS Spearman IC
    # --------------------------------------------------------

    ic_b = spearman_corr(
        pb,
        y,
    )

    ic_n = spearman_corr(
        pn,
        y,
    )

    delta_ic = (
        ic_n - ic_b
    )

    # --------------------------------------------------------
    # 5.2 Rank-z errors
    # --------------------------------------------------------

    yz = rank_z(y)
    bz = rank_z(pb)
    nz = rank_z(pn)

    if (
        yz is None
        or bz is None
        or nz is None
    ):
        return None

    mse_b = float(
        np.mean(
            (bz - yz) ** 2
        )
    )

    mse_n = float(
        np.mean(
            (nz - yz) ** 2
        )
    )

    mae_b = float(
        np.mean(
            np.abs(
                bz - yz
            )
        )
    )

    mae_n = float(
        np.mean(
            np.abs(
                nz - yz
            )
        )
    )

    # --------------------------------------------------------
    # 5.3 Predicted high-risk capture
    # --------------------------------------------------------

    y_pct = percentile_rank(
        y
    )

    pb_pct = percentile_rank(
        pb
    )

    pn_pct = percentile_rank(
        pn
    )

    b_top = (
        pb_pct >= 0.90
    )

    b_bottom = (
        pb_pct <= 0.10
    )

    n_top = (
        pn_pct >= 0.90
    )

    n_bottom = (
        pn_pct <= 0.10
    )

    b_top_mean = float(
        np.mean(
            y_pct[b_top]
        )
    )

    n_top_mean = float(
        np.mean(
            y_pct[n_top]
        )
    )

    b_spread = float(
        np.mean(
            y_pct[b_top]
        )
        -
        np.mean(
            y_pct[b_bottom]
        )
    )

    n_spread = float(
        np.mean(
            y_pct[n_top]
        )
        -
        np.mean(
            y_pct[n_bottom]
        )
    )

    return {
        "evaluation_n":
            int(n),

        "baseline_ic":
            ic_b,

        "network_ic":
            ic_n,

        "delta_ic":
            delta_ic,

        "baseline_rank_mse":
            mse_b,

        "network_rank_mse":
            mse_n,

        "delta_rank_mse":
            mse_n - mse_b,

        "baseline_rank_mae":
            mae_b,

        "network_rank_mae":
            mae_n,

        "delta_rank_mae":
            mae_n - mae_b,

        "baseline_top_decile_target_rank":
            b_top_mean,

        "network_top_decile_target_rank":
            n_top_mean,

        "delta_top_decile_target_rank":
            n_top_mean - b_top_mean,

        "baseline_top_bottom_spread":
            b_spread,

        "network_top_bottom_spread":
            n_spread,

        "delta_top_bottom_spread":
            n_spread - b_spread,
    }


# ============================================================
# 6. Build evaluation scopes
# ============================================================

def build_date_manifest(
    forecast_summary,
):

    x = forecast_summary[
        forecast_summary[
            "forecast_generated"
        ]
    ].copy()

    x["full_rank"] = (
        x[
            "baseline_matrix_rank"
        ]
        ==
        x[
            "baseline_active_parameter_n"
        ]
    ) & (
        x[
            "network_matrix_rank"
        ]
        ==
        x[
            "network_active_parameter_n"
        ]
    )

    x["has_realized_target"] = (
        x[
            "forecast_target_available_n"
        ] > 0
    )

    x["primary_evaluable"] = (
        x[
            "has_realized_target"
        ]
    )

    x["full_rank_evaluable"] = (
        x[
            "has_realized_target"
        ]
        &
        x[
            "full_rank"
        ]
    )

    keep = [
        "window",
        "forecast_date",
        "forecast_row_n",
        "forecast_target_available_n",
        "baseline_active_parameter_n",
        "baseline_matrix_rank",
        "baseline_condition_number",
        "network_active_parameter_n",
        "network_matrix_rank",
        "network_condition_number",
        "full_rank",
        "has_realized_target",
        "primary_evaluable",
        "full_rank_evaluable",
    ]

    return (
        x[keep]
        .sort_values(
            [
                "window",
                "forecast_date",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# 7. Monthly OOS metrics
# ============================================================

def build_monthly_metrics(
    predictions,
    manifest,
):

    rows = []

    lookup = manifest.set_index(
        [
            "window",
            "forecast_date",
        ]
    )

    for (
        window,
        date
    ), g in predictions.groupby(
        [
            "window",
            "analysis_date",
        ],
        sort=True,
    ):

        key = (
            int(window),
            pd.Timestamp(date),
        )

        if key not in lookup.index:
            continue

        info = lookup.loc[key]

        if isinstance(
            info,
            pd.DataFrame,
        ):
            raise RuntimeError(
                f"Duplicate manifest key: {key}"
            )

        # -----------------------------------------------
        # PRIMARY
        # -----------------------------------------------

        if as_bool(
            info[
                "primary_evaluable"
            ]
        ):

            metric = (
                evaluate_one_date(g)
            )

            if metric is not None:

                rows.append(
                    {
                        "sample_scope":
                            "PRIMARY",

                        "window":
                            int(window),

                        "analysis_date":
                            pd.Timestamp(date),

                        "full_rank":
                            bool(
                                info[
                                    "full_rank"
                                ]
                            ),

                        **metric,
                    }
                )

        # -----------------------------------------------
        # FULL-RANK robustness
        # -----------------------------------------------

        if as_bool(
            info[
                "full_rank_evaluable"
            ]
        ):

            metric = (
                evaluate_one_date(g)
            )

            if metric is not None:

                rows.append(
                    {
                        "sample_scope":
                            "FULL_RANK",

                        "window":
                            int(window),

                        "analysis_date":
                            pd.Timestamp(date),

                        "full_rank":
                            True,

                        **metric,
                    }
                )

    out = pd.DataFrame(
        rows
    )

    if not out.empty:

        out["year"] = (
            out[
                "analysis_date"
            ]
            .dt.year
            .astype(int)
        )

        out = (
            out.sort_values(
                [
                    "sample_scope",
                    "window",
                    "analysis_date",
                ]
            )
            .reset_index(drop=True)
        )

    return out


# ============================================================
# 8. Annual summary
# ============================================================

def build_annual_summary(
    monthly,
):

    return (
        monthly.groupby(
            [
                "sample_scope",
                "window",
                "year",
            ],
            as_index=False,
        )
        .agg(
            date_n=(
                "analysis_date",
                "nunique",
            ),

            mean_baseline_ic=(
                "baseline_ic",
                "mean",
            ),

            mean_network_ic=(
                "network_ic",
                "mean",
            ),

            mean_delta_ic=(
                "delta_ic",
                "mean",
            ),

            median_delta_ic=(
                "delta_ic",
                "median",
            ),

            positive_delta_ic_month_share=(
                "delta_ic",
                lambda s:
                    float(
                        (s > 0).mean()
                    ),
            ),

            mean_delta_rank_mse=(
                "delta_rank_mse",
                "mean",
            ),

            mean_delta_top_bottom_spread=(
                "delta_top_bottom_spread",
                "mean",
            ),
        )
        .sort_values(
            [
                "sample_scope",
                "window",
                "year",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# 9. Main OOS summary
# ============================================================

def build_oos_summary(
    monthly,
    annual,
):

    rows = []

    for (
        scope,
        window
    ), g in monthly.groupby(
        [
            "sample_scope",
            "window",
        ],
        sort=True,
    ):

        g = g.sort_values(
            "analysis_date"
        )

        ann = annual[
            (
                annual[
                    "sample_scope"
                ] == scope
            )
            &
            (
                annual[
                    "window"
                ] == window
            )
        ]

        nw = nw_mean_test(
            g["delta_ic"],
            lag=NW_LAG,
        )

        mean_mse_b = float(
            g[
                "baseline_rank_mse"
            ].mean()
        )

        mean_mse_n = float(
            g[
                "network_rank_mse"
            ].mean()
        )

        relative_oos_r2 = (
            1.0
            -
            mean_mse_n
            / mean_mse_b
            if mean_mse_b > 0
            else np.nan
        )

        rows.append(
            {
                "sample_scope":
                    scope,

                "window":
                    int(window),

                "date_n":
                    int(
                        g[
                            "analysis_date"
                        ].nunique()
                    ),

                "evaluation_row_n":
                    int(
                        g[
                            "evaluation_n"
                        ].sum()
                    ),

                "first_evaluation_date":
                    g[
                        "analysis_date"
                    ].min(),

                "last_evaluation_date":
                    g[
                        "analysis_date"
                    ].max(),

                "mean_baseline_ic":
                    float(
                        g[
                            "baseline_ic"
                        ].mean()
                    ),

                "median_baseline_ic":
                    float(
                        g[
                            "baseline_ic"
                        ].median()
                    ),

                "mean_network_ic":
                    float(
                        g[
                            "network_ic"
                        ].mean()
                    ),

                "median_network_ic":
                    float(
                        g[
                            "network_ic"
                        ].median()
                    ),

                "mean_delta_ic":
                    float(
                        g[
                            "delta_ic"
                        ].mean()
                    ),

                "median_delta_ic":
                    float(
                        g[
                            "delta_ic"
                        ].median()
                    ),

                "positive_delta_ic_month_share":
                    float(
                        (
                            g[
                                "delta_ic"
                            ] > 0
                        ).mean()
                    ),

                "hac_se_delta_ic":
                    nw["se"],

                "hac_t_delta_ic":
                    nw["t"],

                "hac_p_two_sided_delta_ic":
                    nw[
                        "p_two_sided"
                    ],

                "hac_p_gt0_delta_ic":
                    nw[
                        "p_gt0"
                    ],

                "positive_delta_ic_year_share":
                    (
                        float(
                            (
                                ann[
                                    "mean_delta_ic"
                                ] > 0
                            ).mean()
                        )
                        if len(ann)
                        else np.nan
                    ),

                "year_n":
                    int(
                        ann[
                            "year"
                        ].nunique()
                    ),

                "mean_baseline_rank_mse":
                    mean_mse_b,

                "mean_network_rank_mse":
                    mean_mse_n,

                "mean_delta_rank_mse":
                    float(
                        g[
                            "delta_rank_mse"
                        ].mean()
                    ),

                "relative_oos_r2_rank":
                    relative_oos_r2,

                "mean_baseline_rank_mae":
                    float(
                        g[
                            "baseline_rank_mae"
                        ].mean()
                    ),

                "mean_network_rank_mae":
                    float(
                        g[
                            "network_rank_mae"
                        ].mean()
                    ),

                "mean_delta_rank_mae":
                    float(
                        g[
                            "delta_rank_mae"
                        ].mean()
                    ),

                "mean_baseline_top_bottom_spread":
                    float(
                        g[
                            "baseline_top_bottom_spread"
                        ].mean()
                    ),

                "mean_network_top_bottom_spread":
                    float(
                        g[
                            "network_top_bottom_spread"
                        ].mean()
                    ),

                "mean_delta_top_bottom_spread":
                    float(
                        g[
                            "delta_top_bottom_spread"
                        ].mean()
                    ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "sample_scope",
                "window",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# 10. Main
# ============================================================

def main():

    print("=" * 78)
    print("M1 - Research Day 11 - Step 4")
    print("Incremental OOS Risk Metrics")
    print("=" * 78)

    # ========================================================
    # 10.1 Upstream QA
    # ========================================================

    step3_qa = load_qa(
        STEP3_QA_PATH
    )

    if not as_bool(
        step3_qa.get(
            "all_formal_qa_pass",
            False,
        )
    ):
        raise RuntimeError(
            "Day-11 Step-3 formal QA "
            "has not passed."
        )

    if not as_bool(
        step3_qa.get(
            "step1_control_set_confirmed",
            False,
        )
    ):
        raise RuntimeError(
            "Step-1 control set was not "
            "explicitly frozen in Step 3."
        )

    # ========================================================
    # 10.2 Load frozen OOS predictions
    # ========================================================

    cols = [
        "security_id",
        "analysis_date",
        "window",
        TARGET,
        BASE_SCORE,
        NETWORK_SCORE,
    ]

    pred = pd.read_parquet(
        PRED_PATH,
        columns=cols,
    )

    pred[
        "analysis_date"
    ] = pd.to_datetime(
        pred[
            "analysis_date"
        ],
        errors="raise",
    )

    pred["window"] = (
        pd.to_numeric(
            pred["window"],
            errors="raise",
        )
        .astype(int)
    )

    # No target filtering yet:
    # target availability must never alter
    # the original Step-3 prediction file.

    # ========================================================
    # 10.3 Load Step-3 forecast diagnostics
    # ========================================================

    forecast = pd.read_csv(
        FORECAST_SUMMARY_PATH
    )

    forecast[
        "forecast_date"
    ] = pd.to_datetime(
        forecast[
            "forecast_date"
        ],
        errors="raise",
    )

    forecast["window"] = (
        pd.to_numeric(
            forecast["window"],
            errors="raise",
        )
        .astype(int)
    )

    # ========================================================
    # 10.4 Evaluation-date manifest
    # ========================================================

    manifest = (
        build_date_manifest(
            forecast
        )
    )

    save_csv(
        manifest,
        EVAL_DATE_OUT,
    )

    # ========================================================
    # 10.5 Monthly metrics
    # ========================================================

    monthly = (
        build_monthly_metrics(
            pred,
            manifest,
        )
    )

    if monthly.empty:
        raise RuntimeError(
            "No evaluable OOS observations."
        )

    save_csv(
        monthly,
        MONTHLY_OUT,
    )

    # ========================================================
    # 10.6 Annual descriptive stability
    # ========================================================

    annual = (
        build_annual_summary(
            monthly
        )
    )

    save_csv(
        annual,
        ANNUAL_OUT,
    )

    # ========================================================
    # 10.7 Overall incremental summary
    # ========================================================

    summary = (
        build_oos_summary(
            monthly,
            annual,
        )
    )

    save_csv(
        summary,
        SUMMARY_OUT,
    )

    # ========================================================
    # 10.8 Formal QA
    # ========================================================

    primary = monthly[
        monthly[
            "sample_scope"
        ] == "PRIMARY"
    ]

    full_rank = monthly[
        monthly[
            "sample_scope"
        ] == "FULL_RANK"
    ]

    primary_window_n = int(
        primary[
            "window"
        ].nunique()
    )

    full_rank_window_n = int(
        full_rank[
            "window"
        ].nunique()
    )

    primary_dup_n = int(
        primary[
            [
                "window",
                "analysis_date",
            ]
        ]
        .duplicated()
        .sum()
    )

    full_rank_dup_n = int(
        full_rank[
            [
                "window",
                "analysis_date",
            ]
        ]
        .duplicated()
        .sum()
    )

    nonfinite_metric_n = int(
        (
            ~np.isfinite(
                monthly[
                    [
                        "baseline_ic",
                        "network_ic",
                        "delta_ic",
                        "baseline_rank_mse",
                        "network_rank_mse",
                    ]
                ]
                .to_numpy(
                    dtype=float
                )
            )
        ).sum()
    )

    # FULL_RANK dates must be subset of PRIMARY.
    p_keys = set(
        zip(
            primary[
                "window"
            ],
            primary[
                "analysis_date"
            ],
        )
    )

    f_keys = set(
        zip(
            full_rank[
                "window"
            ],
            full_rank[
                "analysis_date"
            ],
        )
    )

    full_rank_subset = (
        f_keys.issubset(
            p_keys
        )
    )

    expected_summary_rows = (
        len(WINDOWS)
        * len(SCOPES)
    )

    qa = {
        "step3_formal_qa_pass":
            True,

        "prediction_row_count":
            int(len(pred)),

        "primary_evaluation_date_window_count":
            int(len(primary)),

        "full_rank_evaluation_date_window_count":
            int(len(full_rank)),

        "primary_window_count":
            primary_window_n,

        "full_rank_window_count":
            full_rank_window_n,

        "primary_duplicate_date_window_count":
            primary_dup_n,

        "full_rank_duplicate_date_window_count":
            full_rank_dup_n,

        "full_rank_is_subset_of_primary":
            full_rank_subset,

        "summary_row_count":
            int(len(summary)),

        "expected_summary_row_count":
            expected_summary_rows,

        "nonfinite_core_metric_count":
            nonfinite_metric_n,

        "model_refitted":
            False,

        "factor_reestimated":
            False,

        "factor_selected_from_results":
            False,

        "factor_removed_from_results":
            False,

        "window_selected_from_results":
            False,

        "hyperparameter_tuned":
            False,

        "future_target_used_for_model_selection":
            False,

        "target_availability_used_only_for_evaluation":
            True,

        "baseline_and_network_same_stock_sample":
            True,

        "full_rank_scope_defined_without_outcome":
            True,
    }

    qa[
        "all_formal_qa_pass"
    ] = bool(
        qa[
            "primary_window_count"
        ] == 3
        and
        qa[
            "full_rank_window_count"
        ] == 3
        and
        qa[
            "primary_duplicate_date_window_count"
        ] == 0
        and
        qa[
            "full_rank_duplicate_date_window_count"
        ] == 0
        and
        qa[
            "full_rank_is_subset_of_primary"
        ]
        and
        qa[
            "summary_row_count"
        ]
        == expected_summary_rows
        and
        qa[
            "nonfinite_core_metric_count"
        ] == 0
    )

    save_csv(
        pd.DataFrame(
            [
                {
                    "qa_name": k,
                    "qa_value": v,
                }
                for k, v in qa.items()
            ]
        ),
        QA_OUT,
    )

    # ========================================================
    # 10.9 Metadata
    # ========================================================

    metadata = {
        "research_day": 11,

        "step":
            "Step4_Incremental_OOS_Risk_Metrics",

        "input_predictions":
            str(PRED_PATH),

        "target":
            TARGET,

        "windows":
            WINDOWS,

        "sample_scopes": {
            "PRIMARY":
                (
                    "All Step-3 OOS forecasts with "
                    "realized future-IVOL target."
                ),

            "FULL_RANK":
                (
                    "Robustness subset requiring both "
                    "baseline and network design matrices "
                    "to be full rank at the forecast date."
                ),
        },

        "primary_metric":
            (
                "delta_ic = network Spearman OOS IC "
                "- baseline Spearman OOS IC"
            ),

        "secondary_metrics": [
            "rank-scale MSE",
            "rank-scale MAE",
            "relative rank OOS R2",
            "predicted top-vs-bottom realized-risk spread",
        ],

        "delta_metric_direction": {
            "delta_ic":
                "positive is better",

            "delta_rank_mse":
                "negative is better",

            "delta_rank_mae":
                "negative is better",

            "delta_top_bottom_spread":
                "positive is better",

            "relative_oos_r2_rank":
                "positive is better",
        },

        "hac_inference": {
            "metric":
                "monthly delta_ic",

            "newey_west_lag":
                NW_LAG,

            "role":
                (
                    "supporting inference only; "
                    "not used for model selection"
                ),
        },

        "important_note":
            (
                "Step 4 does not refit either model. "
                "All scores are frozen Step-3 OOS scores."
            ),

        "formal_qa":
            qa,
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

    # ========================================================
    # 10.10 Console
    # ========================================================

    show = summary[
        [
            "sample_scope",
            "window",
            "date_n",
            "mean_baseline_ic",
            "mean_network_ic",
            "mean_delta_ic",
            "positive_delta_ic_month_share",
            "positive_delta_ic_year_share",
            "relative_oos_r2_rank",
            "hac_t_delta_ic",
            "hac_p_two_sided_delta_ic",
        ]
    ]

    print()
    print(
        "Incremental OOS summary:"
    )

    print(
        show.to_string(
            index=False
        )
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
            "Day-11 Step-4 formal QA failed."
        )

    print()
    print("=" * 78)
    print("DAY 11 STEP 4 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()