from pathlib import Path
from collections import OrderedDict
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

STEP1 = (
    DAY11
    / "01_stage1_frozen_prediction_panel"
)

STEP3 = (
    DAY11
    / "03_stage3_strict_oos_forecasting"
)

STEP4 = (
    DAY11
    / "04_stage4_incremental_oos_metrics"
)

PANEL = (
    STEP1
    / "frozen_prediction_panel.parquet"
)

CALENDAR = (
    STEP1
    / "oos_training_calendar.csv"
)

STEP3_PRED = (
    STEP3
    / "oos_predictions.parquet"
)

STEP3_SUMMARY = (
    STEP3
    / "oos_forecast_summary.csv"
)

STEP3_QA = (
    STEP3
    / "day11_step3_qa.csv"
)

STEP4_MONTHLY = (
    STEP4
    / "monthly_oos_risk_metrics.csv"
)

STEP4_ANNUAL = (
    STEP4
    / "annual_oos_risk_metrics.csv"
)

STEP4_QA = (
    STEP4
    / "day11_step4_qa.csv"
)

OUTDIR = (
    DAY11
    / "05_stage5_oos_temporal_block_validation"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

TEMPORAL_YEAR_OUT = (
    OUTDIR
    / "temporal_oos_by_year.csv"
)

TEMPORAL_SUMMARY_OUT = (
    OUTDIR
    / "temporal_oos_summary.csv"
)

BLOCK_MONTHLY_OUT = (
    OUTDIR
    / "block_oos_monthly_metrics.csv"
)

BLOCK_ANNUAL_OUT = (
    OUTDIR
    / "block_oos_annual_metrics.csv"
)

BLOCK_SUMMARY_OUT = (
    OUTDIR
    / "block_oos_summary.csv"
)

RANK_DIAG_OUT = (
    OUTDIR
    / "block_rank_diagnostic_summary.csv"
)

QA_OUT = (
    OUTDIR
    / "day11_step5_qa.csv"
)

META_OUT = (
    OUTDIR
    / "day11_step5_metadata.json"
)


# ============================================================
# 1. Frozen design
# ============================================================

WINDOWS = [60, 120, 252]

TARGET = "future_idio_vol_annualized"

INDUSTRY = "industry_id1"

CONTROL_NUMERIC = [
    "log_market_value",
    "momentum_120_20",
    "reversal_20",
    "volatility_60",
    "log_turnover_20",
]

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


# ============================================================
# 2. Pre-frozen structural blocks
# ============================================================

BLOCKS = OrderedDict(
    [
        (
            "LEVEL_POSITION",
            [
                "residual_degree_percentile",
                "cross_industry_degree_percentile",
                "outside_community_degree_percentile",
            ],
        ),

        (
            "CROSS_INDUSTRY_COMPOSITION",
            [
                "cross_industry_degree_ratio",
            ],
        ),

        (
            "RECONFIGURATION",
            [
                "delta_degree_percentile",
                "delta_residual_degree_percentile_1m",
                "delta_cross_industry_degree_percentile_1m",
            ],
        ),

        (
            "NEIGHBOR_STABILITY",
            [
                "neighbor_jaccard_1m",
                "neighbor_retention_1m",
            ],
        ),
    ]
)

MIN_TRAIN_PERIODS = 24
MIN_CROSS_SECTION_N = 100

DATE_BALANCED_WEIGHTING = True

PINV_RCOND = 1e-10
EPS = 1e-12

NW_LAG = 3

REPRO_TOL = 1e-8


# ============================================================
# 3. Utilities
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


def rank_z_vector(x):

    r = percentile_rank(x)

    if len(r) < 2:
        return None

    sd = np.std(
        r,
        ddof=1,
    )

    if (
        not np.isfinite(sd)
        or sd <= EPS
    ):
        return None

    return (
        r - np.mean(r)
    ) / sd


def rank_z_frame(
    df,
    cols,
):

    rank = (
        df[cols]
        .rank(
            axis=0,
            method="average",
            pct=True,
        )
        .astype(float)
    )

    z = np.zeros(
        rank.shape,
        dtype=float,
    )

    for j, col in enumerate(cols):

        x = rank[
            col
        ].to_numpy(
            dtype=float
        )

        sd = np.std(
            x,
            ddof=1,
        )

        if (
            np.isfinite(sd)
            and sd > EPS
        ):

            z[:, j] = (
                x - np.mean(x)
            ) / sd

        else:

            z[:, j] = 0.0

    return z


def spearman_corr(
    x,
    y,
):

    rx = percentile_rank(x)
    ry = percentile_rank(y)

    if (
        len(rx) < 2
        or
        np.std(rx, ddof=1) <= EPS
        or
        np.std(ry, ddof=1) <= EPS
    ):
        return np.nan

    return float(
        np.corrcoef(
            rx,
            ry,
        )[0, 1]
    )


# ============================================================
# 4. Newey-West inference
# ============================================================

def nw_mean_test(
    values,
    lag=3,
):

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
            "p_gt0": np.nan,
        }

    mu = float(
        np.mean(x)
    )

    u = x - mu

    L = min(
        lag,
        T - 1,
    )

    long_run = float(
        np.dot(u, u)
        / T
    )

    for ell in range(
        1,
        L + 1,
    ):

        weight = (
            1
            - ell
            / (L + 1)
        )

        gamma = float(
            np.dot(
                u[ell:],
                u[:-ell],
            )
            / T
        )

        long_run += (
            2
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
        p_gt0 = np.nan

    else:

        t_stat = (
            mu / se
        )

        p_gt0 = (
            0.5
            * math.erfc(
                t_stat
                / math.sqrt(2)
            )
        )

    return {
        "mean": mu,
        "se": se,
        "t": t_stat,
        "p_gt0": p_gt0,
    }


# ============================================================
# 5. Base cross-sectional design
# ============================================================

def build_base_design(
    g,
    industry_levels,
):

    """
    IMPORTANT:
    Always require ALL nine frozen factors,
    even when a block is omitted.

    Therefore every LOBO model uses exactly the same
    stock sample as the frozen Step-3 model.
    """

    numeric = (
        CONTROL_NUMERIC
        + FACTORS
    )

    required = (
        numeric
        + [INDUSTRY]
    )

    complete = (
        g[required]
        .notna()
        .all(axis=1)
    ).to_numpy()

    finite = np.isfinite(
        g[numeric]
        .to_numpy(
            dtype=float
        )
    ).all(axis=1)

    mask = (
        complete
        & finite
    )

    sub = (
        g.loc[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(sub) < MIN_CROSS_SECTION_N:
        return None

    control_z = (
        rank_z_frame(
            sub,
            CONTROL_NUMERIC,
        )
    )

    network_z = (
        rank_z_frame(
            sub,
            FACTORS,
        )
    )

    ind = (
        sub[
            INDUSTRY
        ]
        .astype(str)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Preserve the frozen Step-3 industry coding:
    # intercept + K-1 industry dummies.
    # --------------------------------------------------------

    dummy_levels = (
        industry_levels[1:]
    )

    if dummy_levels:

        dummies = np.column_stack(
            [
                (
                    ind == level
                ).astype(float)
                for level
                in dummy_levels
            ]
        )

    else:

        dummies = np.empty(
            (
                len(sub),
                0,
            ),
            dtype=float,
        )

    intercept = np.ones(
        (
            len(sub),
            1,
        ),
        dtype=float,
    )

    return {
        "data":
            sub,

        "intercept":
            intercept,

        "control_z":
            control_z,

        "network_z":
            network_z,

        "dummies":
            dummies,
    }


# ============================================================
# 6. Build one LOBO model matrix
# ============================================================

def model_matrix(
    design,
    factor_names,
):

    idx = [
        FACTORS.index(f)
        for f in factor_names
    ]

    network_part = (
        design[
            "network_z"
        ][:, idx]
        if idx
        else np.empty(
            (
                len(
                    design["data"]
                ),
                0,
            ),
            dtype=float,
        )
    )

    return np.column_stack(
        [
            design[
                "intercept"
            ],

            design[
                "control_z"
            ],

            network_part,

            design[
                "dummies"
            ],
        ]
    )


# ============================================================
# 7. OLS
# ============================================================

def solve_ols(
    xtx,
    xty,
):

    diagonal = np.diag(
        xtx
    )

    active = (
        np.isfinite(diagonal)
        &
        (diagonal > EPS)
    )

    beta = np.zeros(
        len(xty),
        dtype=float,
    )

    if active.sum() == 0:

        return (
            beta,
            0,
            0,
            np.nan,
        )

    A = xtx[
        np.ix_(
            active,
            active,
        )
    ]

    b = xty[
        active
    ]

    rank = int(
        np.linalg.matrix_rank(
            A
        )
    )

    active_n = int(
        active.sum()
    )

    condition = float(
        np.linalg.cond(
            A
        )
    )

    beta[
        active
    ] = (
        np.linalg.pinv(
            A,
            rcond=PINV_RCOND,
        )
        @ b
    )

    return (
        beta,
        rank,
        active_n,
        condition,
    )


# ============================================================
# 8. Evaluation metrics
# ============================================================

def score_metrics(
    score,
    target,
):

    ic = spearman_corr(
        score,
        target,
    )

    sz = rank_z_vector(
        score
    )

    yz = rank_z_vector(
        target
    )

    if (
        sz is None
        or yz is None
    ):
        return None

    mse = float(
        np.mean(
            (
                sz - yz
            ) ** 2
        )
    )

    mae = float(
        np.mean(
            np.abs(
                sz - yz
            )
        )
    )

    score_pct = percentile_rank(
        score
    )

    target_pct = percentile_rank(
        target
    )

    top = (
        score_pct >= 0.90
    )

    bottom = (
        score_pct <= 0.10
    )

    spread = float(
        np.mean(
            target_pct[top]
        )
        -
        np.mean(
            target_pct[bottom]
        )
    )

    return {
        "ic": ic,
        "rank_mse": mse,
        "rank_mae": mae,
        "top_bottom_spread": spread,
    }


# ============================================================
# 9. One historical training block
# ============================================================

def build_training_block(
    design,
    model_factor_sets,
):

    sub = design[
        "data"
    ]

    usable = (
        sub[
            TARGET
        ].notna()
        &
        sub[
            "label_end_date"
        ].notna()
    ).to_numpy()

    n = int(
        usable.sum()
    )

    if n < MIN_CROSS_SECTION_N:
        return None

    label_end = (
        pd.to_datetime(
            sub.loc[
                usable,
                "label_end_date",
            ]
        )
        .dropna()
        .drop_duplicates()
    )

    if len(label_end) != 1:

        raise RuntimeError(
            "Historical block contains "
            "multiple label_end_date values."
        )

    y = rank_z_vector(
        sub.loc[
            usable,
            TARGET,
        ].to_numpy(
            dtype=float
        )
    )

    if y is None:
        return None

    scale = (
        1.0 / n
        if DATE_BALANCED_WEIGHTING
        else 1.0
    )

    stats = {}

    for model_name, factors \
        in model_factor_sets.items():

        X = model_matrix(
            design,
            factors,
        )[usable]

        stats[
            model_name
        ] = {
            "xtx":
                scale
                * (X.T @ X),

            "xty":
                scale
                * (X.T @ y),
        }

    return {
        "n":
            n,

        "label_end_date":
            pd.Timestamp(
                label_end.iloc[0]
            ),

        "stats":
            stats,
    }


# ============================================================
# 10. Run one network window
# ============================================================

def run_window(
    window,
    forecast_summary,
):

    print()
    print(
        f"[Window W{window}]"
    )

    panel_cols = [
        "security_id",
        "analysis_date",
        "window",
        INDUSTRY,
        *CONTROL_NUMERIC,
        *FACTORS,
        TARGET,
        "label_end_date",
    ]

    df = pd.read_parquet(
        PANEL,
        columns=panel_cols,
        filters=[
            (
                "window",
                "==",
                window,
            )
        ],
    )

    df[
        "analysis_date"
    ] = pd.to_datetime(
        df[
            "analysis_date"
        ],
        errors="raise",
    )

    df[
        "label_end_date"
    ] = pd.to_datetime(
        df[
            "label_end_date"
        ],
        errors="coerce",
    )

    df = (
        df.sort_values(
            [
                "analysis_date",
                "security_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Frozen Step-3 predictions
    # --------------------------------------------------------

    pred = pd.read_parquet(
        STEP3_PRED,
        columns=[
            "security_id",
            "analysis_date",
            "window",
            TARGET,
            "baseline_score",
            "network_score",
        ],
        filters=[
            (
                "window",
                "==",
                window,
            )
        ],
    )

    pred[
        "analysis_date"
    ] = pd.to_datetime(
        pred[
            "analysis_date"
        ],
        errors="raise",
    )

    pred[
        "security_id"
    ] = (
        pred[
            "security_id"
        ]
        .astype(str)
    )

    pred_groups = {
        pd.Timestamp(d):
            x.copy()
        for d, x
        in pred.groupby(
            "analysis_date"
        )
    }

    # --------------------------------------------------------
    # Forecast metadata
    # --------------------------------------------------------

    fs = (
        forecast_summary[
            forecast_summary[
                "window"
            ] == window
        ]
        .copy()
    )

    fs = (
        fs.set_index(
            "forecast_date"
        )
    )

    # --------------------------------------------------------
    # Same industry vocabulary as Step 3
    # --------------------------------------------------------

    industry_levels = sorted(
        df[
            INDUSTRY
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # Leave-one-block-out models
    # --------------------------------------------------------

    model_factor_sets = OrderedDict()

    for block_name, block_factors \
        in BLOCKS.items():

        kept = [
            f for f in FACTORS
            if f not in block_factors
        ]

        model_factor_sets[
            f"WITHOUT_{block_name}"
        ] = kept

    # --------------------------------------------------------
    # Initialize sufficient statistics
    # --------------------------------------------------------

    states = {}

    for model_name, factor_set \
        in model_factor_sets.items():

        p = (
            1
            + len(
                CONTROL_NUMERIC
            )
            + len(
                factor_set
            )
            + max(
                len(
                    industry_levels
                ) - 1,
                0,
            )
        )

        states[
            model_name
        ] = {
            "xtx":
                np.zeros(
                    (
                        p,
                        p,
                    ),
                    dtype=float,
                ),

            "xty":
                np.zeros(
                    p,
                    dtype=float,
                ),
        }

    pending = []

    train_period_n = 0
    train_row_n = 0

    monthly_rows = []
    rank_rows = []

    sample_mismatch_n = 0
    train_period_mismatch_n = 0
    train_row_mismatch_n = 0

    # ========================================================
    # Expanding walk-forward loop
    # ========================================================

    for date, g in df.groupby(
        "analysis_date",
        sort=True,
    ):

        date = pd.Timestamp(
            date
        )

        # ====================================================
        # 10.1 Activate realized historical labels
        # ====================================================

        remain = []

        for block in pending:

            if (
                block[
                    "analysis_date"
                ] < date
                and
                block[
                    "label_end_date"
                ] <= date
            ):

                for model_name \
                    in model_factor_sets:

                    states[
                        model_name
                    ][
                        "xtx"
                    ] += (
                        block[
                            "stats"
                        ][
                            model_name
                        ][
                            "xtx"
                        ]
                    )

                    states[
                        model_name
                    ][
                        "xty"
                    ] += (
                        block[
                            "stats"
                        ][
                            model_name
                        ][
                            "xty"
                        ]
                    )

                train_period_n += 1
                train_row_n += (
                    block["n"]
                )

            else:

                remain.append(
                    block
                )

        pending = remain

        # ====================================================
        # 10.2 Current PIT design
        # ====================================================

        design = (
            build_base_design(
                g,
                industry_levels,
            )
        )

        # ====================================================
        # 10.3 If Step 3 generated a forecast,
        #      generate all LOBO scores
        # ====================================================

        if (
            date in fs.index
            and
            as_bool(
                fs.loc[
                    date,
                    "forecast_generated",
                ]
            )
        ):

            info = fs.loc[
                date
            ]

            if isinstance(
                info,
                pd.DataFrame,
            ):
                raise RuntimeError(
                    "Duplicate Step-3 forecast summary key."
                )

            # -----------------------------------------------
            # Training-history reproduction QA
            # -----------------------------------------------

            if (
                train_period_n
                != int(
                    info[
                        "usable_train_period_n"
                    ]
                )
            ):
                train_period_mismatch_n += 1

            if (
                train_row_n
                != int(
                    info[
                        "usable_train_row_n"
                    ]
                )
            ):
                train_row_mismatch_n += 1

            if design is None:

                raise RuntimeError(
                    f"W{window} {date}: "
                    "Step 3 generated forecast "
                    "but Step 5 design is unavailable."
                )

            if date not in pred_groups:

                raise RuntimeError(
                    f"W{window} {date}: "
                    "missing frozen Step-3 predictions."
                )

            frozen = (
                pred_groups[
                    date
                ]
                .copy()
            )

            frozen[
                "security_id"
            ] = (
                frozen[
                    "security_id"
                ]
                .astype(str)
            )

            base_ids = set(
                design[
                    "data"
                ][
                    "security_id"
                ]
                .astype(str)
            )

            frozen_ids = set(
                frozen[
                    "security_id"
                ]
            )

            if (
                base_ids
                != frozen_ids
            ):
                sample_mismatch_n += 1

            # -----------------------------------------------
            # Common base table
            # -----------------------------------------------

            current = pd.DataFrame(
                {
                    "security_id":
                        design[
                            "data"
                        ][
                            "security_id"
                        ]
                        .astype(str),
                }
            )

            # -----------------------------------------------
            # Fit each LOBO model
            # -----------------------------------------------

            block_diagnostics = {}

            for (
                model_name,
                factor_set,
            ) in model_factor_sets.items():

                beta, rank, active_n, cond = (
                    solve_ols(
                        states[
                            model_name
                        ][
                            "xtx"
                        ],

                        states[
                            model_name
                        ][
                            "xty"
                        ],
                    )
                )

                X_now = (
                    model_matrix(
                        design,
                        factor_set,
                    )
                )

                current[
                    model_name
                ] = (
                    X_now @ beta
                )

                block_diagnostics[
                    model_name
                ] = {
                    "rank":
                        rank,

                    "active_n":
                        active_n,

                    "condition":
                        cond,

                    "full_rank":
                        bool(
                            rank
                            == active_n
                        ),
                }

            current = current.merge(
                frozen[
                    [
                        "security_id",
                        TARGET,
                        "baseline_score",
                        "network_score",
                    ]
                ],
                on="security_id",
                how="inner",
                validate="one_to_one",
            )

            if (
                len(current)
                != len(frozen)
            ):
                sample_mismatch_n += 1

            # -----------------------------------------------
            # Only realized target dates enter evaluation
            # -----------------------------------------------

            valid = (
                current[
                    TARGET
                ].notna()
            )

            eval_df = (
                current.loc[
                    valid
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            if (
                len(eval_df)
                >= MIN_CROSS_SECTION_N
            ):

                y = (
                    eval_df[
                        TARGET
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                baseline_metric = (
                    score_metrics(
                        eval_df[
                            "baseline_score"
                        ].to_numpy(
                            dtype=float
                        ),
                        y,
                    )
                )

                all_metric = (
                    score_metrics(
                        eval_df[
                            "network_score"
                        ].to_numpy(
                            dtype=float
                        ),
                        y,
                    )
                )

                step3_full_rank = bool(
                    (
                        info[
                            "baseline_matrix_rank"
                        ]
                        ==
                        info[
                            "baseline_active_parameter_n"
                        ]
                    )
                    and
                    (
                        info[
                            "network_matrix_rank"
                        ]
                        ==
                        info[
                            "network_active_parameter_n"
                        ]
                    )
                )

                scopes = [
                    "PRIMARY"
                ]

                if step3_full_rank:

                    scopes.append(
                        "FULL_RANK"
                    )

                for (
                    block_name,
                    block_factors,
                ) in BLOCKS.items():

                    model_name = (
                        f"WITHOUT_{block_name}"
                    )

                    minus_metric = (
                        score_metrics(
                            eval_df[
                                model_name
                            ].to_numpy(
                                dtype=float
                            ),
                            y,
                        )
                    )

                    diag = (
                        block_diagnostics[
                            model_name
                        ]
                    )

                    for scope in scopes:

                        monthly_rows.append(
                            {
                                "sample_scope":
                                    scope,

                                "window":
                                    int(window),

                                "analysis_date":
                                    date,

                                "year":
                                    int(
                                        date.year
                                    ),

                                "block_name":
                                    block_name,

                                "removed_factor_n":
                                    len(
                                        block_factors
                                    ),

                                "evaluation_n":
                                    int(
                                        len(
                                            eval_df
                                        )
                                    ),

                                "baseline_ic":
                                    baseline_metric[
                                        "ic"
                                    ],

                                "all_network_ic":
                                    all_metric[
                                        "ic"
                                    ],

                                "without_block_ic":
                                    minus_metric[
                                        "ic"
                                    ],

                                "all_vs_baseline_delta_ic":
                                    (
                                        all_metric[
                                            "ic"
                                        ]
                                        -
                                        baseline_metric[
                                            "ic"
                                        ]
                                    ),

                                "without_block_vs_baseline_delta_ic":
                                    (
                                        minus_metric[
                                            "ic"
                                        ]
                                        -
                                        baseline_metric[
                                            "ic"
                                        ]
                                    ),

                                # Positive = block helps.
                                "block_marginal_ic":
                                    (
                                        all_metric[
                                            "ic"
                                        ]
                                        -
                                        minus_metric[
                                            "ic"
                                        ]
                                    ),

                                "all_network_rank_mse":
                                    all_metric[
                                        "rank_mse"
                                    ],

                                "without_block_rank_mse":
                                    minus_metric[
                                        "rank_mse"
                                    ],

                                # Positive = removing block worsens MSE.
                                "block_marginal_mse_gain":
                                    (
                                        minus_metric[
                                            "rank_mse"
                                        ]
                                        -
                                        all_metric[
                                            "rank_mse"
                                        ]
                                    ),

                                "all_network_top_bottom_spread":
                                    all_metric[
                                        "top_bottom_spread"
                                    ],

                                "without_block_top_bottom_spread":
                                    minus_metric[
                                        "top_bottom_spread"
                                    ],

                                # Positive = block improves separation.
                                "block_marginal_spread_gain":
                                    (
                                        all_metric[
                                            "top_bottom_spread"
                                        ]
                                        -
                                        minus_metric[
                                            "top_bottom_spread"
                                        ]
                                    ),

                                "lobo_matrix_rank":
                                    diag[
                                        "rank"
                                    ],

                                "lobo_active_parameter_n":
                                    diag[
                                        "active_n"
                                    ],

                                "lobo_condition_number":
                                    diag[
                                        "condition"
                                    ],

                                "lobo_full_rank":
                                    diag[
                                        "full_rank"
                                    ],
                            }
                        )

                    rank_rows.append(
                        {
                            "window":
                                int(window),

                            "analysis_date":
                                date,

                            "block_name":
                                block_name,

                            "matrix_rank":
                                diag[
                                    "rank"
                                ],

                            "active_parameter_n":
                                diag[
                                    "active_n"
                                ],

                            "full_rank":
                                diag[
                                    "full_rank"
                                ],

                            "condition_number":
                                diag[
                                    "condition"
                                ],
                        }
                    )

        # ====================================================
        # 10.4 Add current period to pending history
        #      only AFTER forecast generation
        # ====================================================

        if design is not None:

            training_block = (
                build_training_block(
                    design,
                    model_factor_sets,
                )
            )

            if training_block is not None:

                training_block[
                    "analysis_date"
                ] = date

                pending.append(
                    training_block
                )

    return (
        pd.DataFrame(
            monthly_rows
        ),

        pd.DataFrame(
            rank_rows
        ),

        sample_mismatch_n,
        train_period_mismatch_n,
        train_row_mismatch_n,
    )


# ============================================================
# 11. Annual block metrics
# ============================================================

def build_block_annual(
    monthly,
):

    return (
        monthly.groupby(
            [
                "sample_scope",
                "window",
                "block_name",
                "year",
            ],
            as_index=False,
        )
        .agg(
            date_n=(
                "analysis_date",
                "nunique",
            ),

            mean_block_marginal_ic=(
                "block_marginal_ic",
                "mean",
            ),

            median_block_marginal_ic=(
                "block_marginal_ic",
                "median",
            ),

            positive_block_month_share=(
                "block_marginal_ic",
                lambda s:
                    float(
                        (s > 0).mean()
                    ),
            ),

            mean_block_marginal_mse_gain=(
                "block_marginal_mse_gain",
                "mean",
            ),

            mean_block_marginal_spread_gain=(
                "block_marginal_spread_gain",
                "mean",
            ),
        )
        .sort_values(
            [
                "sample_scope",
                "window",
                "block_name",
                "year",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 12. Overall block summary
# ============================================================

def build_block_summary(
    monthly,
    annual,
):

    rows = []

    for (
        scope,
        window,
        block_name,
    ), g in monthly.groupby(
        [
            "sample_scope",
            "window",
            "block_name",
        ],
        sort=True,
    ):

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
            &
            (
                annual[
                    "block_name"
                ] == block_name
            )
        ]

        nw = nw_mean_test(
            g[
                "block_marginal_ic"
            ],
            lag=NW_LAG,
        )

        rows.append(
            {
                "sample_scope":
                    scope,

                "window":
                    int(window),

                "block_name":
                    block_name,

                "removed_factor_n":
                    int(
                        g[
                            "removed_factor_n"
                        ].iloc[0]
                    ),

                "date_n":
                    int(
                        g[
                            "analysis_date"
                        ].nunique()
                    ),

                "year_n":
                    int(
                        ann[
                            "year"
                        ].nunique()
                    ),

                "mean_all_vs_baseline_delta_ic":
                    float(
                        g[
                            "all_vs_baseline_delta_ic"
                        ].mean()
                    ),

                "mean_without_block_vs_baseline_delta_ic":
                    float(
                        g[
                            "without_block_vs_baseline_delta_ic"
                        ].mean()
                    ),

                "mean_block_marginal_ic":
                    float(
                        g[
                            "block_marginal_ic"
                        ].mean()
                    ),

                "median_block_marginal_ic":
                    float(
                        g[
                            "block_marginal_ic"
                        ].median()
                    ),

                "positive_block_month_share":
                    float(
                        (
                            g[
                                "block_marginal_ic"
                            ] > 0
                        ).mean()
                    ),

                "positive_block_year_share":
                    (
                        float(
                            (
                                ann[
                                    "mean_block_marginal_ic"
                                ] > 0
                            ).mean()
                        )
                        if len(ann)
                        else np.nan
                    ),

                "hac_se_block_marginal_ic":
                    nw[
                        "se"
                    ],

                "hac_t_block_marginal_ic":
                    nw[
                        "t"
                    ],

                "hac_p_gt0_block_marginal_ic":
                    nw[
                        "p_gt0"
                    ],

                "mean_block_marginal_mse_gain":
                    float(
                        g[
                            "block_marginal_mse_gain"
                        ].mean()
                    ),

                "mean_block_marginal_spread_gain":
                    float(
                        g[
                            "block_marginal_spread_gain"
                        ].mean()
                    ),

                "lobo_full_rank_date_share":
                    float(
                        g[
                            "lobo_full_rank"
                        ].mean()
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "sample_scope",
                "window",
                "block_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 13. Temporal stability of frozen Step-4 result
# ============================================================

def build_temporal_summary(
    step4_annual,
):

    rows = []

    for (
        scope,
        window,
    ), g in step4_annual.groupby(
        [
            "sample_scope",
            "window",
        ]
    ):

        g = g.sort_values(
            "year"
        )

        worst_idx = (
            g[
                "mean_delta_ic"
            ].idxmin()
        )

        best_idx = (
            g[
                "mean_delta_ic"
            ].idxmax()
        )

        rows.append(
            {
                "sample_scope":
                    scope,

                "window":
                    int(window),

                "year_n":
                    int(
                        g[
                            "year"
                        ].nunique()
                    ),

                "mean_annual_delta_ic":
                    float(
                        g[
                            "mean_delta_ic"
                        ].mean()
                    ),

                "median_annual_delta_ic":
                    float(
                        g[
                            "mean_delta_ic"
                        ].median()
                    ),

                "positive_year_share":
                    float(
                        (
                            g[
                                "mean_delta_ic"
                            ] > 0
                        ).mean()
                    ),

                "worst_year":
                    int(
                        g.loc[
                            worst_idx,
                            "year",
                        ]
                    ),

                "worst_year_delta_ic":
                    float(
                        g.loc[
                            worst_idx,
                            "mean_delta_ic",
                        ]
                    ),

                "best_year":
                    int(
                        g.loc[
                            best_idx,
                            "year",
                        ]
                    ),

                "best_year_delta_ic":
                    float(
                        g.loc[
                            best_idx,
                            "mean_delta_ic",
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 14. Main
# ============================================================

def main():

    print("=" * 80)
    print("M1 - Research Day 11 - Step 5")
    print("OOS Temporal and Block-Incremental Validation")
    print("=" * 80)

    # ========================================================
    # 14.1 Upstream QA
    # ========================================================

    step3_qa = load_qa(
        STEP3_QA
    )

    step4_qa = load_qa(
        STEP4_QA
    )

    if not as_bool(
        step3_qa.get(
            "all_formal_qa_pass",
            False,
        )
    ):
        raise RuntimeError(
            "Step 3 QA has not passed."
        )

    if not as_bool(
        step4_qa.get(
            "all_formal_qa_pass",
            False,
        )
    ):
        raise RuntimeError(
            "Step 4 QA has not passed."
        )

    # ========================================================
    # 14.2 Verify block partition
    # ========================================================

    block_flat = [
        f
        for x
        in BLOCKS.values()
        for f
        in x
    ]

    block_cover_ok = (
        sorted(
            block_flat
        )
        ==
        sorted(
            FACTORS
        )
    )

    block_unique_ok = (
        len(
            block_flat
        )
        ==
        len(
            set(
                block_flat
            )
        )
    )

    if not (
        block_cover_ok
        and
        block_unique_ok
    ):

        raise RuntimeError(
            "Frozen block partition does not "
            "cover the nine factors exactly once."
        )

    # ========================================================
    # 14.3 Load forecast summary
    # ========================================================

    forecast_summary = pd.read_csv(
        STEP3_SUMMARY
    )

    forecast_summary[
        "forecast_date"
    ] = pd.to_datetime(
        forecast_summary[
            "forecast_date"
        ],
        errors="raise",
    )

    forecast_summary[
        "window"
    ] = pd.to_numeric(
        forecast_summary[
            "window"
        ],
        errors="raise",
    ).astype(int)

    # ========================================================
    # 14.4 Run LOBO models
    # ========================================================

    monthly_parts = []
    rank_parts = []

    sample_mismatch_total = 0
    train_period_mismatch_total = 0
    train_row_mismatch_total = 0

    for window in WINDOWS:

        (
            monthly_w,
            rank_w,
            sample_mismatch_n,
            train_period_mismatch_n,
            train_row_mismatch_n,
        ) = run_window(
            window,
            forecast_summary,
        )

        monthly_parts.append(
            monthly_w
        )

        rank_parts.append(
            rank_w
        )

        sample_mismatch_total += (
            sample_mismatch_n
        )

        train_period_mismatch_total += (
            train_period_mismatch_n
        )

        train_row_mismatch_total += (
            train_row_mismatch_n
        )

    monthly = pd.concat(
        monthly_parts,
        ignore_index=True,
    )

    rank_diag = pd.concat(
        rank_parts,
        ignore_index=True,
    )

    monthly = (
        monthly.sort_values(
            [
                "sample_scope",
                "window",
                "block_name",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    save_csv(
        monthly,
        BLOCK_MONTHLY_OUT,
    )

    # ========================================================
    # 14.5 Annual + overall block results
    # ========================================================

    block_annual = (
        build_block_annual(
            monthly
        )
    )

    save_csv(
        block_annual,
        BLOCK_ANNUAL_OUT,
    )

    block_summary = (
        build_block_summary(
            monthly,
            block_annual,
        )
    )

    save_csv(
        block_summary,
        BLOCK_SUMMARY_OUT,
    )

    # ========================================================
    # 14.6 Rank diagnostic summary
    # ========================================================

    rank_summary = (
        rank_diag.groupby(
            [
                "window",
                "block_name",
            ],
            as_index=False,
        )
        .agg(
            forecast_date_n=(
                "analysis_date",
                "nunique",
            ),

            full_rank_date_n=(
                "full_rank",
                "sum",
            ),

            full_rank_date_share=(
                "full_rank",
                "mean",
            ),

            median_condition_number=(
                "condition_number",
                lambda s:
                    s.replace(
                        [np.inf, -np.inf],
                        np.nan,
                    ).median(),
            ),
        )
    )

    save_csv(
        rank_summary,
        RANK_DIAG_OUT,
    )

    # ========================================================
    # 14.7 Temporal validation from frozen Step 4
    # ========================================================

    step4_annual = pd.read_csv(
        STEP4_ANNUAL
    )

    save_csv(
        step4_annual,
        TEMPORAL_YEAR_OUT,
    )

    temporal_summary = (
        build_temporal_summary(
            step4_annual
        )
    )

    save_csv(
        temporal_summary,
        TEMPORAL_SUMMARY_OUT,
    )

    # ========================================================
    # 14.8 Step-4 reproduction QA
    # ========================================================

    step4_monthly = pd.read_csv(
        STEP4_MONTHLY
    )

    step4_monthly[
        "analysis_date"
    ] = pd.to_datetime(
        step4_monthly[
            "analysis_date"
        ],
        errors="raise",
    )

    # Any block has identical baseline / all-network metrics.
    reference_block = (
        list(
            BLOCKS.keys()
        )[0]
    )

    reference = (
        monthly[
            monthly[
                "block_name"
            ] == reference_block
        ][
            [
                "sample_scope",
                "window",
                "analysis_date",
                "baseline_ic",
                "all_network_ic",
                "all_vs_baseline_delta_ic",
            ]
        ]
        .copy()
    )

    check = reference.merge(
        step4_monthly[
            [
                "sample_scope",
                "window",
                "analysis_date",
                "baseline_ic",
                "network_ic",
                "delta_ic",
            ]
        ],
        on=[
            "sample_scope",
            "window",
            "analysis_date",
        ],
        how="inner",
        suffixes=(
            "_step5",
            "_step4",
        ),
        validate="one_to_one",
    )

    reproduction_error = float(
        np.nanmax(
            np.abs(
                np.column_stack(
                    [
                        (
                            check[
                                "baseline_ic_step5"
                            ]
                            -
                            check[
                                "baseline_ic_step4"
                            ]
                        ),

                        (
                            check[
                                "all_network_ic"
                            ]
                            -
                            check[
                                "network_ic"
                            ]
                        ),

                        (
                            check[
                                "all_vs_baseline_delta_ic"
                            ]
                            -
                            check[
                                "delta_ic"
                            ]
                        ),
                    ]
                )
            )
        )
    )

    # ========================================================
    # 14.9 Formal QA
    # ========================================================

    expected_summary_rows = (
        2
        * len(WINDOWS)
        * len(BLOCKS)
    )

    nonfinite_core_n = int(
        (
            ~np.isfinite(
                monthly[
                    [
                        "baseline_ic",
                        "all_network_ic",
                        "without_block_ic",
                        "block_marginal_ic",
                    ]
                ]
                .to_numpy(
                    dtype=float
                )
            )
        ).sum()
    )

    qa = {
        "step3_formal_qa_pass":
            True,

        "step4_formal_qa_pass":
            True,

        "frozen_factor_count":
            len(FACTORS),

        "block_count":
            len(BLOCKS),

        "blocks_cover_all_9_factors":
            bool(
                block_cover_ok
            ),

        "factor_appears_in_exactly_one_block":
            bool(
                block_unique_ok
            ),

        "window_count":
            int(
                monthly[
                    "window"
                ].nunique()
            ),

        "sample_scope_count":
            int(
                monthly[
                    "sample_scope"
                ].nunique()
            ),

        "block_summary_row_count":
            int(
                len(
                    block_summary
                )
            ),

        "expected_block_summary_row_count":
            int(
                expected_summary_rows
            ),

        "forecast_sample_mismatch_count":
            int(
                sample_mismatch_total
            ),

        "training_period_mismatch_count":
            int(
                train_period_mismatch_total
            ),

        "training_row_mismatch_count":
            int(
                train_row_mismatch_total
            ),

        "step4_metric_reproduction_max_abs_error":
            reproduction_error,

        "step4_metric_reproduction_pass":
            bool(
                reproduction_error
                <= REPRO_TOL
            ),

        "nonfinite_core_metric_count":
            nonfinite_core_n,

        "primary_scores_refitted":
            False,

        "leave_one_block_out_models_fitted":
            True,

        "block_definition_data_driven":
            False,

        "factor_selected_from_results":
            False,

        "factor_removed_from_primary_model":
            False,

        "window_selected_from_results":
            False,

        "future_target_used_for_model_selection":
            False,

        "future_information_used_in_training":
            False,
    }

    qa[
        "all_formal_qa_pass"
    ] = bool(
        qa[
            "blocks_cover_all_9_factors"
        ]
        and
        qa[
            "factor_appears_in_exactly_one_block"
        ]
        and
        qa[
            "window_count"
        ] == 3
        and
        qa[
            "sample_scope_count"
        ] == 2
        and
        qa[
            "block_summary_row_count"
        ]
        == expected_summary_rows
        and
        qa[
            "forecast_sample_mismatch_count"
        ] == 0
        and
        qa[
            "training_period_mismatch_count"
        ] == 0
        and
        qa[
            "training_row_mismatch_count"
        ] == 0
        and
        qa[
            "step4_metric_reproduction_pass"
        ]
        and
        qa[
            "nonfinite_core_metric_count"
        ] == 0
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
        QA_OUT,
    )

    # ========================================================
    # 14.10 Metadata
    # ========================================================

    metadata = {
        "research_day":
            11,

        "step":
            (
                "Step5_OOS_Temporal_and_"
                "Block_Incremental_Validation"
            ),

        "frozen_factors":
            FACTORS,

        "blocks":
            BLOCKS,

        "block_definition_rule":
            (
                "Blocks are defined from feature construction "
                "and interpretation, not from Step-2 correlations, "
                "VIF values, or Step-4 OOS performance."
            ),

        "block_metric":
            (
                "block_marginal_ic = "
                "IC(all nine network factors) - "
                "IC(network model without that block)"
            ),

        "block_metric_interpretation":
            (
                "Positive values indicate that removing the block "
                "reduces OOS rank IC conditional on all other "
                "network blocks."
            ),

        "important_caution":
            (
                "LOBO block contributions are conditional and "
                "need not sum to the total All-Network minus "
                "Baseline OOS improvement."
            ),

        "training_scheme":
            (
                "Exactly the frozen Step-3 expanding-window "
                "PIT scheme with date-balanced weighting and "
                "the same complete-nine-factor stock sample."
            ),

        "sample_scopes": [
            "PRIMARY",
            "FULL_RANK",
        ],

        "newey_west_lag":
            NW_LAG,

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
    # 14.11 Console summary
    # ========================================================

    print()
    print(
        "Temporal OOS summary:"
    )

    print(
        temporal_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Block incremental summary:"
    )

    show = block_summary[
        [
            "sample_scope",
            "window",
            "block_name",
            "mean_block_marginal_ic",
            "positive_block_month_share",
            "positive_block_year_share",
            "mean_block_marginal_mse_gain",
            "hac_t_block_marginal_ic",
        ]
    ]

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
            "Day-11 Step-5 formal QA failed."
        )

    print()
    print("=" * 80)
    print("DAY 11 STEP 5 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()