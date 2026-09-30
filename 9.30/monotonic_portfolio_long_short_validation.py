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
DAY12 = ROOT / "output" / "M1_day12"

STEP1 = (
    DAY12
    / "01_stage1_frozen_alpha_evaluation_panel"
)

STEP2 = (
    DAY12
    / "02_stage2_alpha_ic_validation"
)

PANEL_PATH = (
    STEP1
    / "frozen_alpha_evaluation_panel.parquet"
)

STEP1_QA_PATH = (
    STEP1
    / "day12_step1_qa.csv"
)

STEP1_META_PATH = (
    STEP1
    / "day12_step1_metadata.json"
)

STEP2_QA_PATH = (
    STEP2
    / "day12_step2_qa.csv"
)

STEP2_META_PATH = (
    STEP2
    / "day12_step2_metadata.json"
)

STEP2_COMMON_DATE_PATH = (
    STEP2
    / "alpha_ic_common_date_manifest.csv"
)

OUTDIR = (
    DAY12
    / "03_stage3_monotonic_portfolio_validation"
)
OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. Frozen design
# ============================================================

WINDOWS = [60, 120, 252]

TARGET = "future_holding_return"

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

CONTROLS = [
    INDUSTRY,
    *CONTROL_NUMERIC,
]

METHOD_RAW = "RAW_FACTOR_SORT"
METHOD_NEUTRAL = "CHARACTERISTIC_NEUTRAL_FACTOR_SORT"

SCOPE_NATIVE = "NATIVE"
SCOPE_COMMON = "COMMON_DATE"

N_GROUPS = 5

# Formation rules.
MIN_FORMATION_N = 100
MIN_GROUP_FORMATION_N = 15

# Return evaluation rules.
MIN_GROUP_REALIZED_N = 15
MIN_GROUP_RETURN_COVERAGE = 0.95

# Monthly annualization convention.
PERIODS_PER_YEAR = 12

# Supporting inference only.
NW_LAG = 3

EPS = 1e-12
LSTSQ_RCOND = 1e-10


# ============================================================
# 2. Outputs
# ============================================================

MONTHLY_OUT = (
    OUTDIR
    / "monthly_quintile_portfolio_returns.csv"
)

ANNUAL_OUT = (
    OUTDIR
    / "annual_long_short_summary.csv"
)

SUMMARY_OUT = (
    OUTDIR
    / "portfolio_validation_summary.csv"
)

CURVE_OUT = (
    OUTDIR
    / "quintile_curve_summary.csv"
)

CUMULATIVE_OUT = (
    OUTDIR
    / "long_short_cumulative_pnl.csv"
)

DATE_QA_OUT = (
    OUTDIR
    / "portfolio_date_window_qa.csv"
)

PORTFOLIO_COMMON_DATE_OUT = (
    OUTDIR
    / "portfolio_common_date_manifest.csv"
)

QA_OUT = (
    OUTDIR
    / "day12_step3_qa.csv"
)

META_OUT = (
    OUTDIR
    / "day12_step3_metadata.json"
)


# ============================================================
# 3. Helpers
# ============================================================

def atomic_csv(df, path):
    path = Path(path)
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
    q = pd.read_csv(path)

    required = {
        "qa_name",
        "qa_value",
    }

    if not required.issubset(
        q.columns
    ):
        raise RuntimeError(
            f"Invalid QA file: {path}"
        )

    return dict(
        zip(
            q["qa_name"],
            q["qa_value"],
        )
    )


def require_columns(
    df,
    columns,
    name,
):
    missing = [
        c
        for c in columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{name} missing columns: {missing}"
        )


# ============================================================
# 4. Rank / neutralization helpers
# ============================================================

def percentile_rank(x):
    return (
        pd.Series(
            np.asarray(
                x,
                dtype=float,
            )
        )
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

    sd = float(
        np.std(
            r,
            ddof=1,
        )
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
    columns,
):
    z = np.zeros(
        (
            len(df),
            len(columns),
        ),
        dtype=float,
    )

    active = []

    for j, col in enumerate(
        columns
    ):
        v = rank_z(
            df[col]
            .to_numpy(
                dtype=float
            )
        )

        if v is None:
            active.append(False)
            z[:, j] = 0.0
        else:
            active.append(True)
            z[:, j] = v

    return (
        z,
        np.asarray(
            active,
            dtype=bool,
        ),
    )


def build_control_design(df):
    """
    Same current-information control design frozen in Day-12 Step 2:

      - full industry fixed effects;
      - rank-z log market value;
      - rank-z momentum;
      - rank-z reversal;
      - rank-z volatility;
      - rank-z turnover.

    No future return enters score construction.
    """

    control_z, active_numeric = (
        rank_z_frame(
            df,
            CONTROL_NUMERIC,
        )
    )

    if active_numeric.any():
        continuous = (
            control_z[
                :,
                active_numeric,
            ]
        )
    else:
        continuous = np.empty(
            (
                len(df),
                0,
            ),
            dtype=float,
        )

    industry = (
        df[INDUSTRY]
        .astype("string")
        .astype(str)
        .to_numpy()
    )

    industry_levels = sorted(
        pd.unique(
            industry
        ).tolist()
    )

    dummies = np.column_stack(
        [
            (
                industry == level
            ).astype(float)
            for level in industry_levels
        ]
    )

    X = np.column_stack(
        [
            continuous,
            dummies,
        ]
    )

    return {
        "X":
            X,

        "active_numeric_control_n":
            int(
                active_numeric.sum()
            ),

        "industry_level_n":
            int(
                len(
                    industry_levels
                )
            ),

        "design_column_n":
            int(
                X.shape[1]
            ),
    }


def build_neutral_factor_scores(df):
    """
    Characteristic-neutral scores used for portfolio formation.

    IMPORTANT:
    Unlike Step 2 partial-rank IC, Step 3 does NOT residualize
    future return. It residualizes only the CURRENT factor score
    on CURRENT traditional characteristics, because the portfolio
    must earn the actual future holding return.

    For all 9 factors:
      1. rank-z factor within date-window;
      2. regress factor rank on industry FE + 5 ranked controls;
      3. use factor residual as the neutral portfolio signal.
    """

    factor_z, factor_active = (
        rank_z_frame(
            df,
            FACTORS,
        )
    )

    if not factor_active.all():
        return None

    control_info = (
        build_control_design(
            df
        )
    )

    X = control_info[
        "X"
    ]

    if X.shape[0] <= X.shape[1]:
        return None

    beta, _, rank, singular_values = (
        np.linalg.lstsq(
            X,
            factor_z,
            rcond=LSTSQ_RCOND,
        )
    )

    residual = (
        factor_z
        -
        X @ beta
    )

    if (
        singular_values is None
        or len(
            singular_values
        ) == 0
    ):
        condition = np.nan
    else:
        smax = float(
            np.max(
                singular_values
            )
        )

        smin = float(
            np.min(
                singular_values
            )
        )

        condition = (
            smax / smin
            if (
                np.isfinite(
                    smin
                )
                and smin > EPS
            )
            else np.inf
        )

    control_info[
        "design_rank"
    ] = int(rank)

    control_info[
        "design_condition_number"
    ] = float(
        condition
    )

    return {
        "scores":
            residual,

        "control_info":
            control_info,
    }


# ============================================================
# 5. Quintile assignment
# ============================================================

def assign_quantile_group(
    score,
    n_groups=N_GROUPS,
):
    """
    Fixed low-to-high factor ordering.

    Q1 = lowest score
    Q5 = highest score

    No factor sign is flipped.
    """

    pct = percentile_rank(
        score
    )

    group = np.ceil(
        pct
        * n_groups
    )

    group = np.clip(
        group,
        1,
        n_groups,
    )

    return (
        group
        .astype(int)
    )


# ============================================================
# 6. Portfolio evaluation for one score
# ============================================================

def evaluate_one_factor_sort(
    formation_df,
    score,
):
    """
    Portfolio formation uses only current information.

    future_holding_return / return_target_available are used ONLY
    after group assignment to evaluate the realized outcome.

    Within each quintile, realized return is equal-weighted among
    members with a valid frozen return label. Return coverage is
    explicitly audited.
    """

    n = len(
        formation_df
    )

    if n < MIN_FORMATION_N:
        return None

    score = np.asarray(
        score,
        dtype=float,
    )

    if (
        len(score) != n
        or
        not np.isfinite(
            score
        ).all()
    ):
        return None

    if float(
        np.std(
            score,
            ddof=1,
        )
    ) <= EPS:
        return None

    groups = assign_quantile_group(
        score,
        n_groups=N_GROUPS,
    )

    temp = formation_df[
        [
            "security_id",
            TARGET,
            "return_target_available",
        ]
    ].copy()

    temp[
        "group"
    ] = groups

    temp[
        "score"
    ] = score

    returns = {}
    medians = {}
    formation_n = {}
    realized_n = {}
    coverage = {}

    valid_all = True

    for q in range(
        1,
        N_GROUPS + 1,
    ):
        qmask = (
            temp[
                "group"
            ] == q
        )

        formation_q = (
            temp.loc[
                qmask
            ]
        )

        n_form = int(
            len(
                formation_q
            )
        )

        realized_q = (
            formation_q.loc[
                formation_q[
                    "return_target_available"
                ]
                .fillna(False)
                .astype(bool)
                &
                formation_q[
                    TARGET
                ].notna()
            ]
        )

        n_real = int(
            len(
                realized_q
            )
        )

        cov = (
            n_real
            / n_form
            if n_form > 0
            else np.nan
        )

        formation_n[q] = (
            n_form
        )

        realized_n[q] = (
            n_real
        )

        coverage[q] = (
            float(cov)
            if np.isfinite(cov)
            else np.nan
        )

        if (
            n_form
            < MIN_GROUP_FORMATION_N
            or
            n_real
            < MIN_GROUP_REALIZED_N
            or
            not np.isfinite(
                cov
            )
            or
            cov
            < MIN_GROUP_RETURN_COVERAGE
        ):
            valid_all = False

        if n_real > 0:
            r = (
                realized_q[
                    TARGET
                ]
                .to_numpy(
                    dtype=float
                )
            )

            returns[q] = float(
                np.mean(r)
            )

            medians[q] = float(
                np.median(r)
            )
        else:
            returns[q] = np.nan
            medians[q] = np.nan

    if not valid_all:
        return {
            "valid":
                False,

            "formation_n":
                formation_n,

            "realized_n":
                realized_n,

            "coverage":
                coverage,
        }

    q_returns = np.asarray(
        [
            returns[q]
            for q in range(
                1,
                N_GROUPS + 1,
            )
        ],
        dtype=float,
    )

    if not np.isfinite(
        q_returns
    ).all():
        return {
            "valid":
                False,

            "formation_n":
                formation_n,

            "realized_n":
                realized_n,

            "coverage":
                coverage,
        }

    quintile_index = np.arange(
        1,
        N_GROUPS + 1,
        dtype=float,
    )

    # Spearman between Q-number and Q-return in this month.
    quintile_spearman = (
        np.corrcoef(
            percentile_rank(
                quintile_index
            ),
            percentile_rank(
                q_returns
            ),
        )[0, 1]
    )

    slope = float(
        np.polyfit(
            quintile_index,
            q_returns,
            deg=1,
        )[0]
    )

    strict_increasing = bool(
        np.all(
            np.diff(
                q_returns
            ) > 0
        )
    )

    strict_decreasing = bool(
        np.all(
            np.diff(
                q_returns
            ) < 0
        )
    )

    high_minus_low = float(
        q_returns[-1]
        -
        q_returns[0]
    )

    out = {
        "valid":
            True,

        "high_minus_low":
            high_minus_low,

        "quintile_spearman":
            float(
                quintile_spearman
            ),

        "quintile_linear_slope":
            slope,

        "strict_increasing":
            strict_increasing,

        "strict_decreasing":
            strict_decreasing,

        "formation_n":
            formation_n,

        "realized_n":
            realized_n,

        "coverage":
            coverage,

        "returns":
            returns,

        "medians":
            medians,
    }

    return out


# ============================================================
# 7. Newey-West inference
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
        np.isfinite(
            x
        )
    ]

    T = len(x)

    if T < 3:
        return {
            "mean":
                np.nan,

            "se":
                np.nan,

            "t":
                np.nan,

            "p_two_sided":
                np.nan,
        }

    mu = float(
        np.mean(
            x
        )
    )

    u = (
        x - mu
    )

    L = min(
        int(lag),
        T - 1,
    )

    long_run = float(
        np.dot(
            u,
            u,
        )
        / T
    )

    for ell in range(
        1,
        L + 1,
    ):
        weight = (
            1.0
            -
            ell
            / (L + 1.0)
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
        long_run
        / T
    )

    if (
        not np.isfinite(
            se
        )
        or
        se <= EPS
    ):
        return {
            "mean":
                mu,

            "se":
                np.nan,

            "t":
                np.nan,

            "p_two_sided":
                np.nan,
        }

    t_stat = (
        mu / se
    )

    p_two = math.erfc(
        abs(
            t_stat
        )
        / math.sqrt(
            2.0
        )
    )

    return {
        "mean":
            mu,

        "se":
            float(
                se
            ),

        "t":
            float(
                t_stat
            ),

        "p_two_sided":
            float(
                p_two
            ),
    }


# ============================================================
# 8. Benjamini-Hochberg
# ============================================================

def bh_qvalues(
    p_values,
):
    p = np.asarray(
        p_values,
        dtype=float,
    )

    q = np.full(
        len(p),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(
            p
        )
    )

    if not valid.any():
        return q

    idx = np.where(
        valid
    )[0]

    pv = p[
        idx
    ]

    order = np.argsort(
        pv
    )

    sorted_p = (
        pv[
            order
        ]
    )

    m = len(
        sorted_p
    )

    raw_q = (
        sorted_p
        * m
        / np.arange(
            1,
            m + 1,
        )
    )

    monotone = np.minimum.accumulate(
        raw_q[::-1]
    )[::-1]

    monotone = np.clip(
        monotone,
        0.0,
        1.0,
    )

    q[
        idx[
            order
        ]
    ] = (
        monotone
    )

    return q


# ============================================================
# 9. Drawdown helper
# ============================================================

def additive_max_drawdown(
    returns,
):
    """
    Long-short H-L is a self-financing spread, so Step 3 reports
    additive cumulative P&L rather than imposing a particular gross
    leverage convention.

    max_drawdown_additive is measured in return/P&L units.
    """

    x = np.asarray(
        returns,
        dtype=float,
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if len(x) == 0:
        return np.nan

    cumulative = np.cumsum(
        x
    )

    wealth_path = np.concatenate(
        [
            [0.0],
            cumulative,
        ]
    )

    running_peak = np.maximum.accumulate(
        wealth_path
    )

    drawdown = (
        running_peak
        -
        wealth_path
    )

    return float(
        np.max(
            drawdown
        )
    )


# ============================================================
# 10. Build one monthly output row
# ============================================================

def result_to_row(
    result,
    scope,
    method,
    window,
    date,
    factor,
    neutral_control_info=None,
):
    row = {
        "sample_scope":
            scope,

        "method":
            method,

        "window":
            int(
                window
            ),

        "analysis_date":
            pd.Timestamp(
                date
            ),

        "factor":
            factor,

        "portfolio_valid":
            bool(
                result[
                    "valid"
                ]
            ),
    }

    for q in range(
        1,
        N_GROUPS + 1,
    ):
        row[
            f"q{q}_formation_n"
        ] = int(
            result[
                "formation_n"
            ].get(
                q,
                0,
            )
        )

        row[
            f"q{q}_realized_n"
        ] = int(
            result[
                "realized_n"
            ].get(
                q,
                0,
            )
        )

        row[
            f"q{q}_return_coverage"
        ] = (
            result[
                "coverage"
            ].get(
                q,
                np.nan,
            )
        )

        row[
            f"q{q}_mean_return"
        ] = (
            result.get(
                "returns",
                {},
            ).get(
                q,
                np.nan,
            )
        )

        row[
            f"q{q}_median_return"
        ] = (
            result.get(
                "medians",
                {},
            ).get(
                q,
                np.nan,
            )
        )

    row[
        "high_minus_low"
    ] = result.get(
        "high_minus_low",
        np.nan,
    )

    row[
        "quintile_spearman"
    ] = result.get(
        "quintile_spearman",
        np.nan,
    )

    row[
        "quintile_linear_slope"
    ] = result.get(
        "quintile_linear_slope",
        np.nan,
    )

    row[
        "strict_increasing"
    ] = result.get(
        "strict_increasing",
        False,
    )

    row[
        "strict_decreasing"
    ] = result.get(
        "strict_decreasing",
        False,
    )

    if neutral_control_info is None:
        row[
            "control_design_column_n"
        ] = np.nan

        row[
            "control_design_rank"
        ] = np.nan

        row[
            "control_design_condition_number"
        ] = np.nan

    else:
        row[
            "control_design_column_n"
        ] = neutral_control_info[
            "design_column_n"
        ]

        row[
            "control_design_rank"
        ] = neutral_control_info[
            "design_rank"
        ]

        row[
            "control_design_condition_number"
        ] = neutral_control_info[
            "design_condition_number"
        ]

    return row


# ============================================================
# 11. Process one date-window
# ============================================================

def process_date_window(
    g,
    window,
    date,
    common_date_set,
):
    """
    Build NATIVE and COMMON_DATE portfolio sorts.

    Key anti-leakage rule:
        Portfolio groups are formed before checking future return
        availability. Future return availability is used only when
        evaluating realized portfolio returns.
    """

    rows = []
    qa_rows = []

    # --------------------------------------------------------
    # Current-information formation masks
    # --------------------------------------------------------

    raw_current_mask = (
        g[
            "network_feature_complete"
        ]
        .fillna(False)
        .astype(bool)
    )

    neutral_current_mask = (
        raw_current_mask
        &
        g[
            "control_complete"
        ]
        .fillna(False)
        .astype(bool)
        &
        g[
            "control_numeric_finite"
        ]
        .fillna(False)
        .astype(bool)
    )

    raw_native_df = (
        g.loc[
            raw_current_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    neutral_native_df = (
        g.loc[
            neutral_current_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Neutral scores are computed ONCE on current information.
    # --------------------------------------------------------

    neutral_score_result = None

    if (
        len(
            neutral_native_df
        )
        >= MIN_FORMATION_N
    ):
        neutral_score_result = (
            build_neutral_factor_scores(
                neutral_native_df
            )
        )

    # --------------------------------------------------------
    # Native RAW
    # --------------------------------------------------------

    for factor in FACTORS:
        result = evaluate_one_factor_sort(
            raw_native_df,
            raw_native_df[
                factor
            ].to_numpy(
                dtype=float
            ),
        )

        if result is None:
            result = {
                "valid":
                    False,

                "formation_n":
                    {},

                "realized_n":
                    {},

                "coverage":
                    {},
            }

        rows.append(
            result_to_row(
                result=result,
                scope=SCOPE_NATIVE,
                method=METHOD_RAW,
                window=window,
                date=date,
                factor=factor,
                neutral_control_info=None,
            )
        )

    # --------------------------------------------------------
    # Native NEUTRAL
    # --------------------------------------------------------

    if (
        neutral_score_result
        is not None
    ):
        neutral_scores = (
            neutral_score_result[
                "scores"
            ]
        )

        neutral_info = (
            neutral_score_result[
                "control_info"
            ]
        )

        for j, factor in enumerate(
            FACTORS
        ):
            result = evaluate_one_factor_sort(
                neutral_native_df,
                neutral_scores[
                    :,
                    j,
                ],
            )

            if result is None:
                result = {
                    "valid":
                        False,

                    "formation_n":
                        {},

                    "realized_n":
                        {},

                    "coverage":
                        {},
                }

            rows.append(
                result_to_row(
                    result=result,
                    scope=SCOPE_NATIVE,
                    method=METHOD_NEUTRAL,
                    window=window,
                    date=date,
                    factor=factor,
                    neutral_control_info=neutral_info,
                )
            )

    else:
        for factor in FACTORS:
            rows.append(
                result_to_row(
                    result={
                        "valid":
                            False,

                        "formation_n":
                            {},

                        "realized_n":
                            {},

                        "coverage":
                            {},
                    },
                    scope=SCOPE_NATIVE,
                    method=METHOD_NEUTRAL,
                    window=window,
                    date=date,
                    factor=factor,
                    neutral_control_info=None,
                )
            )

    # --------------------------------------------------------
    # COMMON_DATE
    #
    # On common dates BOTH methods use the SAME current-info
    # neutral formation universe, so portfolio differences are
    # not driven by stock-sample composition.
    # --------------------------------------------------------

    if pd.Timestamp(
        date
    ) in common_date_set:

        for factor in FACTORS:
            result = evaluate_one_factor_sort(
                neutral_native_df,
                neutral_native_df[
                    factor
                ].to_numpy(
                    dtype=float
                ),
            )

            if result is None:
                result = {
                    "valid":
                        False,

                    "formation_n":
                        {},

                    "realized_n":
                        {},

                    "coverage":
                        {},
                }

            rows.append(
                result_to_row(
                    result=result,
                    scope=SCOPE_COMMON,
                    method=METHOD_RAW,
                    window=window,
                    date=date,
                    factor=factor,
                    neutral_control_info=None,
                )
            )

        if (
            neutral_score_result
            is not None
        ):
            neutral_scores = (
                neutral_score_result[
                    "scores"
                ]
            )

            neutral_info = (
                neutral_score_result[
                    "control_info"
                ]
            )

            for j, factor in enumerate(
                FACTORS
            ):
                result = evaluate_one_factor_sort(
                    neutral_native_df,
                    neutral_scores[
                        :,
                        j,
                    ],
                )

                if result is None:
                    result = {
                        "valid":
                            False,

                        "formation_n":
                            {},

                        "realized_n":
                            {},

                        "coverage":
                            {},
                    }

                rows.append(
                    result_to_row(
                        result=result,
                        scope=SCOPE_COMMON,
                        method=METHOD_NEUTRAL,
                        window=window,
                        date=date,
                        factor=factor,
                        neutral_control_info=neutral_info,
                    )
                )

        else:
            for factor in FACTORS:
                rows.append(
                    result_to_row(
                        result={
                            "valid":
                                False,

                            "formation_n":
                                {},

                            "realized_n":
                                {},

                            "coverage":
                                {},
                        },
                        scope=SCOPE_COMMON,
                        method=METHOD_NEUTRAL,
                        window=window,
                        date=date,
                        factor=factor,
                        neutral_control_info=None,
                    )
                )

    # --------------------------------------------------------
    # Date-window QA
    # --------------------------------------------------------

    target_valid_n = int(
        g[
            "return_target_available"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    qa_rows.append(
        {
            "window":
                int(
                    window
                ),

            "analysis_date":
                pd.Timestamp(
                    date
                ),

            "is_step2_common_date":
                bool(
                    pd.Timestamp(
                        date
                    )
                    in common_date_set
                ),

            "stock_n":
                int(
                    len(g)
                ),

            "raw_current_formation_n":
                int(
                    len(
                        raw_native_df
                    )
                ),

            "neutral_current_formation_n":
                int(
                    len(
                        neutral_native_df
                    )
                ),

            "return_target_available_n":
                target_valid_n,

            "neutral_score_available":
                bool(
                    neutral_score_result
                    is not None
                ),

            "neutral_control_design_column_n":
                (
                    neutral_score_result[
                        "control_info"
                    ][
                        "design_column_n"
                    ]
                    if neutral_score_result
                    is not None
                    else np.nan
                ),

            "neutral_control_design_rank":
                (
                    neutral_score_result[
                        "control_info"
                    ][
                        "design_rank"
                    ]
                    if neutral_score_result
                    is not None
                    else np.nan
                ),

            "neutral_control_design_condition_number":
                (
                    neutral_score_result[
                        "control_info"
                    ][
                        "design_condition_number"
                    ]
                    if neutral_score_result
                    is not None
                    else np.nan
                ),
        }
    )

    return (
        rows,
        qa_rows,
    )


# ============================================================
# 12. Annual summary
# ============================================================

def build_annual_summary(
    monthly_valid,
):
    x = monthly_valid.copy()

    x[
        "year"
    ] = (
        x[
            "analysis_date"
        ]
        .dt.year
        .astype(int)
    )

    rows = []

    for (
        scope,
        method,
        window,
        factor,
        year,
    ), g in x.groupby(
        [
            "sample_scope",
            "method",
            "window",
            "factor",
            "year",
        ],
        sort=True,
    ):
        hl = (
            g[
                "high_minus_low"
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        rows.append(
            {
                "sample_scope":
                    scope,

                "method":
                    method,

                "window":
                    int(
                        window
                    ),

                "factor":
                    factor,

                "year":
                    int(
                        year
                    ),

                "month_n":
                    int(
                        len(
                            hl
                        )
                    ),

                "mean_high_minus_low":
                    (
                        float(
                            hl.mean()
                        )
                        if len(
                            hl
                        )
                        else np.nan
                    ),

                "median_high_minus_low":
                    (
                        float(
                            hl.median()
                        )
                        if len(
                            hl
                        )
                        else np.nan
                    ),

                "positive_hl_month_share":
                    (
                        float(
                            (
                                hl > 0
                            ).mean()
                        )
                        if len(
                            hl
                        )
                        else np.nan
                    ),

                "mean_quintile_spearman":
                    float(
                        g[
                            "quintile_spearman"
                        ].mean()
                    ),

                "strict_increasing_month_share":
                    float(
                        g[
                            "strict_increasing"
                        ].mean()
                    ),

                "strict_decreasing_month_share":
                    float(
                        g[
                            "strict_decreasing"
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
                "method",
                "window",
                "factor",
                "year",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 13. Quintile curve summary
# ============================================================

def build_curve_summary(
    monthly_valid,
):
    rows = []

    for (
        scope,
        method,
        window,
        factor,
    ), g in monthly_valid.groupby(
        [
            "sample_scope",
            "method",
            "window",
            "factor",
        ],
        sort=True,
    ):
        mean_q = []

        for q in range(
            1,
            N_GROUPS + 1,
        ):
            mean_return = float(
                g[
                    f"q{q}_mean_return"
                ].mean()
            )

            median_return = float(
                g[
                    f"q{q}_mean_return"
                ].median()
            )

            mean_coverage = float(
                g[
                    f"q{q}_return_coverage"
                ].mean()
            )

            mean_q.append(
                mean_return
            )

            rows.append(
                {
                    "sample_scope":
                        scope,

                    "method":
                        method,

                    "window":
                        int(
                            window
                        ),

                    "factor":
                        factor,

                    "quintile":
                        int(
                            q
                        ),

                    "mean_monthly_return":
                        mean_return,

                    "median_monthly_return":
                        median_return,

                    "annualized_arithmetic_return":
                        (
                            PERIODS_PER_YEAR
                            * mean_return
                        ),

                    "mean_return_coverage":
                        mean_coverage,
                }
            )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "sample_scope",
                "method",
                "window",
                "factor",
                "quintile",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 14. Overall validation summary
# ============================================================

def build_validation_summary(
    monthly_valid,
    annual,
):
    rows = []

    for (
        scope,
        method,
        window,
        factor,
    ), g in monthly_valid.groupby(
        [
            "sample_scope",
            "method",
            "window",
            "factor",
        ],
        sort=True,
    ):
        g = g.sort_values(
            "analysis_date"
        )

        hl = (
            g[
                "high_minus_low"
            ]
            .to_numpy(
                dtype=float
            )
        )

        nw = nw_mean_test(
            hl,
            lag=NW_LAG,
        )

        mean_hl = float(
            np.mean(
                hl
            )
        )

        std_hl = float(
            np.std(
                hl,
                ddof=1,
            )
        ) if len(
            hl
        ) >= 2 else np.nan

        ann_mean = (
            PERIODS_PER_YEAR
            * mean_hl
        )

        ann_vol = (
            math.sqrt(
                PERIODS_PER_YEAR
            )
            * std_hl
            if np.isfinite(
                std_hl
            )
            else np.nan
        )

        sharpe = (
            ann_mean
            / ann_vol
            if (
                np.isfinite(
                    ann_vol
                )
                and ann_vol > EPS
            )
            else np.nan
        )

        mean_q = np.asarray(
            [
                float(
                    g[
                        f"q{q}_mean_return"
                    ].mean()
                )
                for q in range(
                    1,
                    N_GROUPS + 1,
                )
            ],
            dtype=float,
        )

        q_index = np.arange(
            1,
            N_GROUPS + 1,
            dtype=float,
        )

        mean_curve_spearman = float(
            np.corrcoef(
                percentile_rank(
                    q_index
                ),
                percentile_rank(
                    mean_q
                ),
            )[0, 1]
        )

        mean_curve_slope = float(
            np.polyfit(
                q_index,
                mean_q,
                deg=1,
            )[0]
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
                    "method"
                ] == method
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
                    "factor"
                ] == factor
            )
        ].copy()

        if (
            mean_hl != 0
            and len(
                ann
            )
        ):
            sign_target = (
                1
                if mean_hl > 0
                else -1
            )

            annual_sign = np.sign(
                ann[
                    "mean_high_minus_low"
                ].to_numpy(
                    dtype=float
                )
            )

            same_sign_year_share = float(
                np.mean(
                    annual_sign
                    == sign_target
                )
            )
        else:
            same_sign_year_share = np.nan

        mean_coverages = [
            float(
                g[
                    f"q{q}_return_coverage"
                ].mean()
            )
            for q in range(
                1,
                N_GROUPS + 1,
            )
        ]

        row = {
            "sample_scope":
                scope,

            "method":
                method,

            "window":
                int(
                    window
                ),

            "factor":
                factor,

            "date_n":
                int(
                    g[
                        "analysis_date"
                    ].nunique()
                ),

            "first_date":
                g[
                    "analysis_date"
                ].min(),

            "last_date":
                g[
                    "analysis_date"
                ].max(),

            "mean_high_minus_low":
                mean_hl,

            "median_high_minus_low":
                float(
                    np.median(
                        hl
                    )
                ),

            "positive_hl_month_share":
                float(
                    np.mean(
                        hl > 0
                    )
                ),

            "negative_hl_month_share":
                float(
                    np.mean(
                        hl < 0
                    )
                ),

            "annualized_arithmetic_hl":
                ann_mean,

            "annualized_volatility_hl":
                ann_vol,

            "annualized_sharpe_hl":
                sharpe,

            "additive_max_drawdown_hl":
                additive_max_drawdown(
                    hl
                ),

            "hac_se_high_minus_low":
                nw[
                    "se"
                ],

            "hac_t_high_minus_low":
                nw[
                    "t"
                ],

            "hac_p_two_sided_high_minus_low":
                nw[
                    "p_two_sided"
                ],

            "mean_monthly_quintile_spearman":
                float(
                    g[
                        "quintile_spearman"
                    ].mean()
                ),

            "median_monthly_quintile_spearman":
                float(
                    g[
                        "quintile_spearman"
                    ].median()
                ),

            "positive_quintile_spearman_month_share":
                float(
                    (
                        g[
                            "quintile_spearman"
                        ] > 0
                    ).mean()
                ),

            "negative_quintile_spearman_month_share":
                float(
                    (
                        g[
                            "quintile_spearman"
                        ] < 0
                    ).mean()
                ),

            "strict_increasing_month_share":
                float(
                    g[
                        "strict_increasing"
                    ].mean()
                ),

            "strict_decreasing_month_share":
                float(
                    g[
                        "strict_decreasing"
                    ].mean()
                ),

            "mean_quintile_curve_spearman":
                mean_curve_spearman,

            "mean_quintile_curve_linear_slope":
                mean_curve_slope,

            "year_n":
                int(
                    ann[
                        "year"
                    ].nunique()
                ),

            "positive_hl_year_share":
                (
                    float(
                        (
                            ann[
                                "mean_high_minus_low"
                            ] > 0
                        ).mean()
                    )
                    if len(
                        ann
                    )
                    else np.nan
                ),

            "same_sign_hl_year_share":
                same_sign_year_share,

            "minimum_mean_quintile_return_coverage":
                float(
                    np.min(
                        mean_coverages
                    )
                ),

            "mean_quintile_return_coverage":
                float(
                    np.mean(
                        mean_coverages
                    )
                ),
        }

        for q in range(
            1,
            N_GROUPS + 1,
        ):
            row[
                f"mean_q{q}_return"
            ] = float(
                g[
                    f"q{q}_mean_return"
                ].mean()
            )

        rows.append(
            row
        )

    out = pd.DataFrame(
        rows
    )

    # BH correction separately by scope x method across 27
    # frozen factor-window cells.
    out[
        "bh_q_high_minus_low"
    ] = np.nan

    for scope in [
        SCOPE_NATIVE,
        SCOPE_COMMON,
    ]:
        for method in [
            METHOD_RAW,
            METHOD_NEUTRAL,
        ]:
            mask = (
                (
                    out[
                        "sample_scope"
                    ] == scope
                )
                &
                (
                    out[
                        "method"
                    ] == method
                )
            )

            out.loc[
                mask,
                "bh_q_high_minus_low",
            ] = bh_qvalues(
                out.loc[
                    mask,
                    "hac_p_two_sided_high_minus_low",
                ].to_numpy(
                    dtype=float
                )
            )

    return (
        out.sort_values(
            [
                "sample_scope",
                "method",
                "window",
                "factor",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 15. Cumulative additive long-short P&L
# ============================================================

def build_cumulative_pnl(
    monthly_valid,
):
    x = (
        monthly_valid[
            [
                "sample_scope",
                "method",
                "window",
                "factor",
                "analysis_date",
                "high_minus_low",
            ]
        ]
        .copy()
        .sort_values(
            [
                "sample_scope",
                "method",
                "window",
                "factor",
                "analysis_date",
            ]
        )
    )

    x[
        "cumulative_high_minus_low_pnl"
    ] = (
        x.groupby(
            [
                "sample_scope",
                "method",
                "window",
                "factor",
            ]
        )[
            "high_minus_low"
        ]
        .cumsum()
    )

    return (
        x.reset_index(
            drop=True
        )
    )


# ============================================================
# 16. Main
# ============================================================

def main():
    print("=" * 86)
    print("M1 - Research Day 12 - Step 3")
    print("Monotonic Portfolio and Long-Short Validation")
    print("=" * 86)

    # --------------------------------------------------------
    # 16.1 Upstream QA
    # --------------------------------------------------------

    for path in [
        PANEL_PATH,
        STEP1_QA_PATH,
        STEP1_META_PATH,
        STEP2_QA_PATH,
        STEP2_META_PATH,
        STEP2_COMMON_DATE_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    step1_qa = load_qa(
        STEP1_QA_PATH
    )

    step2_qa = load_qa(
        STEP2_QA_PATH
    )

    if not as_bool(
        step1_qa.get(
            "all_formal_qa_pass",
            False,
        )
    ):
        raise RuntimeError(
            "Day-12 Step-1 formal QA has not passed."
        )

    if not as_bool(
        step2_qa.get(
            "all_formal_qa_pass",
            False,
        )
    ):
        raise RuntimeError(
            "Day-12 Step-2 formal QA has not passed."
        )

    with open(
        STEP1_META_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        step1_meta = json.load(
            f
        )

    if (
        step1_meta.get(
            "frozen_windows"
        )
        != WINDOWS
    ):
        raise RuntimeError(
            "Step-1 frozen windows mismatch."
        )

    if (
        step1_meta.get(
            "frozen_network_factors"
        )
        != FACTORS
    ):
        raise RuntimeError(
            "Step-1 frozen factors mismatch."
        )

    if (
        step1_meta.get(
            "frozen_controls"
        )
        != CONTROLS
    ):
        raise RuntimeError(
            "Step-1 frozen controls mismatch."
        )

    if (
        step1_meta.get(
            "return_target"
        )
        != TARGET
    ):
        raise RuntimeError(
            "Step-1 target mismatch."
        )

    # --------------------------------------------------------
    # 16.2 Load common-date manifest frozen in Step 2
    # --------------------------------------------------------

    common_manifest = pd.read_csv(
        STEP2_COMMON_DATE_PATH
    )

    require_columns(
        common_manifest,
        [
            "analysis_date",
        ],
        "Step-2 common-date manifest",
    )

    common_manifest[
        "analysis_date"
    ] = pd.to_datetime(
        common_manifest[
            "analysis_date"
        ],
        errors="raise",
    )

    common_date_set = set(
        common_manifest[
            "analysis_date"
        ].tolist()
    )

    if len(
        common_date_set
    ) == 0:
        raise RuntimeError(
            "Step-2 common-date manifest is empty."
        )

    # --------------------------------------------------------
    # 16.3 Read ONLY frozen Step-1 panel
    # --------------------------------------------------------

    columns = [
        "security_id",
        "analysis_date",
        "window",
        TARGET,

        "network_feature_complete",
        "control_complete",
        "control_numeric_finite",
        "return_target_available",

        INDUSTRY,
        *CONTROL_NUMERIC,
        *FACTORS,
    ]

    panel = pd.read_parquet(
        PANEL_PATH,
        columns=columns,
    )

    require_columns(
        panel,
        columns,
        "Frozen alpha evaluation panel",
    )

    panel[
        "analysis_date"
    ] = pd.to_datetime(
        panel[
            "analysis_date"
        ],
        errors="raise",
    )

    panel[
        "window"
    ] = (
        pd.to_numeric(
            panel[
                "window"
            ],
            errors="raise",
        )
        .astype(int)
    )

    key = [
        "security_id",
        "analysis_date",
        "window",
    ]

    duplicate_key_n = int(
        panel[
            key
        ]
        .duplicated()
        .sum()
    )

    if duplicate_key_n > 0:
        raise RuntimeError(
            "Duplicate frozen stock-date-window key."
        )

    # --------------------------------------------------------
    # 16.4 Portfolio construction
    # --------------------------------------------------------

    monthly_rows = []
    date_qa_rows = []

    grouped = panel.groupby(
        [
            "window",
            "analysis_date",
        ],
        sort=True,
    )

    total_groups = (
        grouped.ngroups
    )

    for counter, (
        (
            window,
            date,
        ),
        g,
    ) in enumerate(
        grouped,
        start=1,
    ):
        if (
            counter == 1
            or
            counter % 25 == 0
            or
            counter
            == total_groups
        ):
            print(
                f"[{counter:>3}/{total_groups}] "
                f"W{int(window)} "
                f"{pd.Timestamp(date).date()}"
            )

        (
            rows,
            qa_rows,
        ) = process_date_window(
            g=g,
            window=window,
            date=date,
            common_date_set=common_date_set,
        )

        monthly_rows.extend(
            rows
        )

        date_qa_rows.extend(
            qa_rows
        )

    monthly = pd.DataFrame(
        monthly_rows
    )

    date_qa = pd.DataFrame(
        date_qa_rows
    )

    monthly = (
        monthly.sort_values(
            [
                "sample_scope",
                "method",
                "window",
                "factor",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # 16.5 Build a Step-3 portfolio-common evaluation calendar
    # --------------------------------------------------------
    #
    # Step 2 COMMON_DATE was defined for IC estimation using the
    # Step-1 neutral-alpha evaluable sample. Step 3 deliberately
    # forms portfolios WITHOUT conditioning on future-return
    # availability. Therefore a Step-2 common date can fail the
    # pre-frozen 95% ex-post return-coverage rule in one or more
    # quintile portfolios. Requiring every Step-2 common date to
    # survive would contradict the anti-leakage design.
    #
    # For Step 3 we therefore create a stricter evaluation calendar:
    # a date is PORTFOLIO_COMMON_DATE only if all
    #   2 methods x 3 windows x 9 factors = 54
    # COMMON_DATE portfolio cells are valid under the pre-frozen
    # formation-size and realized-return-coverage rules.
    # This uses outcome AVAILABILITY only for evaluation-quality
    # filtering; it never enters portfolio formation or sorting.

    common_attempts = (
        monthly.loc[
            monthly[
                "sample_scope"
            ] == SCOPE_COMMON
        ]
        .copy()
    )

    expected_common_cells_per_date = (
        2
        * len(WINDOWS)
        * len(FACTORS)
    )

    common_eval_manifest = (
        common_attempts.groupby(
            "analysis_date",
            as_index=False,
        )
        .agg(
            output_cell_n=(
                "factor",
                "size",
            ),
            valid_cell_n=(
                "portfolio_valid",
                "sum",
            ),
        )
    )

    # Ensure every frozen Step-2 common date appears in the audit
    # manifest even if some future code path produces no rows.
    step2_common_df = pd.DataFrame(
        {
            "analysis_date": sorted(
                common_date_set
            )
        }
    )

    common_eval_manifest = (
        step2_common_df.merge(
            common_eval_manifest,
            on="analysis_date",
            how="left",
            validate="one_to_one",
        )
        .fillna(
            {
                "output_cell_n": 0,
                "valid_cell_n": 0,
            }
        )
        .sort_values(
            "analysis_date"
        )
        .reset_index(drop=True)
    )

    common_eval_manifest[
        "output_cell_n"
    ] = (
        common_eval_manifest[
            "output_cell_n"
        ]
        .astype(int)
    )

    common_eval_manifest[
        "valid_cell_n"
    ] = (
        common_eval_manifest[
            "valid_cell_n"
        ]
        .astype(int)
    )

    common_eval_manifest[
        "invalid_cell_n"
    ] = (
        expected_common_cells_per_date
        - common_eval_manifest[
            "valid_cell_n"
        ]
    )

    common_eval_manifest[
        "expected_cell_n"
    ] = (
        expected_common_cells_per_date
    )

    common_eval_manifest[
        "portfolio_common_date"
    ] = (
        (
            common_eval_manifest[
                "output_cell_n"
            ]
            == expected_common_cells_per_date
        )
        &
        (
            common_eval_manifest[
                "valid_cell_n"
            ]
            == expected_common_cells_per_date
        )
    )

    portfolio_common_date_set = set(
        common_eval_manifest.loc[
            common_eval_manifest[
                "portfolio_common_date"
            ],
            "analysis_date",
        ]
        .tolist()
    )

    # NATIVE keeps every valid portfolio date. COMMON_DATE uses the
    # strict Step-3 intersection so every factor/window/method is
    # summarized on exactly the same evaluation dates.
    native_valid = (
        monthly.loc[
            (
                monthly[
                    "sample_scope"
                ] == SCOPE_NATIVE
            )
            &
            (
                monthly[
                    "portfolio_valid"
                ]
                .fillna(False)
                .astype(bool)
            )
        ]
        .copy()
    )

    common_valid = (
        monthly.loc[
            (
                monthly[
                    "sample_scope"
                ] == SCOPE_COMMON
            )
            &
            (
                monthly[
                    "portfolio_valid"
                ]
                .fillna(False)
                .astype(bool)
            )
            &
            (
                monthly[
                    "analysis_date"
                ].isin(
                    portfolio_common_date_set
                )
            )
        ]
        .copy()
    )

    monthly_valid = (
        pd.concat(
            [
                native_valid,
                common_valid,
            ],
            ignore_index=True,
        )
        .sort_values(
            [
                "sample_scope",
                "method",
                "window",
                "factor",
                "analysis_date",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 16.6 Annual / full sample summaries
    # --------------------------------------------------------

    annual = (
        build_annual_summary(
            monthly_valid
        )
    )

    summary = (
        build_validation_summary(
            monthly_valid,
            annual,
        )
    )

    curve = (
        build_curve_summary(
            monthly_valid
        )
    )

    cumulative = (
        build_cumulative_pnl(
            monthly_valid
        )
    )

    # --------------------------------------------------------
    # 16.7 Formal QA
    # --------------------------------------------------------

    expected_summary_rows = (
        2
        * 2
        * len(
            WINDOWS
        )
        * len(
            FACTORS
        )
    )

    expected_curve_rows = (
        expected_summary_rows
        * N_GROUPS
    )

    monthly_duplicate_n = int(
        monthly[
            [
                "sample_scope",
                "method",
                "window",
                "analysis_date",
                "factor",
            ]
        ]
        .duplicated()
        .sum()
    )

    summary_duplicate_n = int(
        summary[
            [
                "sample_scope",
                "method",
                "window",
                "factor",
            ]
        ]
        .duplicated()
        .sum()
    )

    common_date_count_by_cell = (
        common_valid.groupby(
            [
                "method",
                "window",
                "factor",
            ]
        )[
            "analysis_date"
        ]
        .nunique()
    )

    portfolio_common_date_count = int(
        len(
            portfolio_common_date_set
        )
    )

    step2_common_date_count = int(
        len(
            common_date_set
        )
    )

    common_all_step2_dates_preserved = bool(
        portfolio_common_date_count
        == step2_common_date_count
    )

    common_date_retention_share = (
        portfolio_common_date_count
        / step2_common_date_count
        if step2_common_date_count > 0
        else np.nan
    )

    common_output_date_set = set(
        common_attempts[
            "analysis_date"
        ].drop_duplicates().tolist()
    )

    common_output_dates_subset_step2_manifest = bool(
        common_output_date_set.issubset(
            common_date_set
        )
    )

    expected_common_cell_n = (
        2
        * len(WINDOWS)
        * len(FACTORS)
    )

    common_cell_count_complete = bool(
        len(
            common_date_count_by_cell
        )
        == expected_common_cell_n
    )

    common_cells_share_same_dates = bool(
        common_cell_count_complete
        and
        portfolio_common_date_count > 0
        and
        (
            common_date_count_by_cell
            == portfolio_common_date_count
        ).all()
    )

    expected_common_analysis_rows = (
        portfolio_common_date_count
        * expected_common_cell_n
    )

    common_analysis_row_count = int(
        len(
            common_valid
        )
    )

    common_analysis_nonfinite_core_n = int(
        (
            ~np.isfinite(
                common_valid[
                    [
                        "high_minus_low",
                        "quintile_spearman",
                        "quintile_linear_slope",
                    ]
                ]
                .to_numpy(
                    dtype=float
                )
            )
        ).sum()
    )

    # Formation must NOT use future-return availability.
    formation_uses_future_return = False

    # No valid portfolio row may violate the frozen coverage rule.
    coverage_columns = [
        f"q{q}_return_coverage"
        for q in range(
            1,
            N_GROUPS + 1,
        )
    ]

    valid_coverage_violation_n = int(
        (
            (
                monthly_valid[
                    coverage_columns
                ]
                < MIN_GROUP_RETURN_COVERAGE
            )
            .any(
                axis=1
            )
        ).sum()
    )

    nonfinite_valid_core_n = int(
        (
            ~np.isfinite(
                monthly_valid[
                    [
                        "high_minus_low",
                        "quintile_spearman",
                        "quintile_linear_slope",
                    ]
                ]
                .to_numpy(
                    dtype=float
                )
            )
        ).sum()
    )

    neutral_valid = (
        monthly_valid[
            monthly_valid[
                "method"
            ] == METHOD_NEUTRAL
        ]
    )

    neutral_rank_deficient_row_n = int(
        (
            neutral_valid[
                "control_design_rank"
            ]
            <
            neutral_valid[
                "control_design_column_n"
            ]
        )
        .fillna(False)
        .sum()
    )

    qa = {
        "step1_formal_qa_pass":
            True,

        "step2_formal_qa_pass":
            True,

        "frozen_panel_row_count":
            int(
                len(
                    panel
                )
            ),

        "duplicate_stock_date_window_count":
            duplicate_key_n,

        "frozen_factor_count":
            len(
                FACTORS
            ),

        "window_count":
            int(
                panel[
                    "window"
                ].nunique()
            ),

        "portfolio_group_count":
            N_GROUPS,

        "method_count":
            2,

        "sample_scope_count":
            2,

        "step2_common_date_count":
            int(
                len(
                    common_date_set
                )
            ),

        "monthly_output_row_count":
            int(
                len(
                    monthly
                )
            ),

        "monthly_valid_row_count":
            int(
                len(
                    monthly_valid
                )
            ),

        "monthly_duplicate_key_count":
            monthly_duplicate_n,

        "summary_row_count":
            int(
                len(
                    summary
                )
            ),

        "expected_summary_row_count":
            int(
                expected_summary_rows
            ),

        "summary_duplicate_key_count":
            summary_duplicate_n,

        "curve_row_count":
            int(
                len(
                    curve
                )
            ),

        "expected_curve_row_count":
            int(
                expected_curve_rows
            ),

        # Descriptive: Step-3 need not preserve every Step-2 common
        # date because portfolio formation intentionally ignores
        # future-return availability, while the pre-frozen 95%
        # realized-return coverage rule is enforced ex post.
        "common_all_step2_dates_preserved":
            common_all_step2_dates_preserved,

        "portfolio_common_date_count":
            portfolio_common_date_count,

        "step2_common_dates_dropped_for_portfolio_evaluation_count":
            int(
                step2_common_date_count
                - portfolio_common_date_count
            ),

        "portfolio_common_date_retention_share":
            common_date_retention_share,

        "common_output_dates_subset_step2_manifest":
            common_output_dates_subset_step2_manifest,

        "common_analysis_cell_count_complete":
            common_cell_count_complete,

        "common_analysis_cells_share_identical_dates":
            common_cells_share_same_dates,

        "common_analysis_row_count":
            common_analysis_row_count,

        "expected_common_analysis_row_count":
            int(
                expected_common_analysis_rows
            ),

        "common_analysis_nonfinite_core_metric_count":
            common_analysis_nonfinite_core_n,

        "valid_portfolio_return_coverage_violation_count":
            valid_coverage_violation_n,

        "nonfinite_valid_core_metric_count":
            nonfinite_valid_core_n,

        # Diagnostic only.
        "neutral_rank_deficient_valid_row_count":
            neutral_rank_deficient_row_n,

        "portfolio_formation_uses_future_return_availability":
            formation_uses_future_return,

        "future_return_used_only_for_realized_evaluation":
            True,

        "portfolio_weighting":
            "EQUAL_WEIGHT",

        "primary_spread_definition":
            "Q5_MINUS_Q1",

        "factor_sign_flipped":
            False,

        "factor_selected_from_step2_results":
            False,

        "window_selected_from_step2_results":
            False,

        "portfolio_direction_selected_from_step2_results":
            False,

        "transaction_cost_applied":
            False,

        "liquidity_filter_applied":
            False,

        "capacity_filter_applied":
            False,

        "network_factors_reestimated":
            False,

        "traditional_controls_reestimated":
            False,

        "future_returns_reestimated":
            False,
    }

    qa[
        "all_formal_qa_pass"
    ] = bool(
        qa[
            "step1_formal_qa_pass"
        ]
        and
        qa[
            "step2_formal_qa_pass"
        ]
        and
        qa[
            "duplicate_stock_date_window_count"
        ] == 0
        and
        qa[
            "frozen_factor_count"
        ] == 9
        and
        qa[
            "window_count"
        ] == 3
        and
        qa[
            "portfolio_group_count"
        ] == 5
        and
        qa[
            "method_count"
        ] == 2
        and
        qa[
            "sample_scope_count"
        ] == 2
        and
        qa[
            "monthly_duplicate_key_count"
        ] == 0
        and
        qa[
            "summary_row_count"
        ] == expected_summary_rows
        and
        qa[
            "summary_duplicate_key_count"
        ] == 0
        and
        qa[
            "curve_row_count"
        ] == expected_curve_rows
        and
        qa[
            "portfolio_common_date_count"
        ] > 0
        and
        qa[
            "common_output_dates_subset_step2_manifest"
        ]
        and
        qa[
            "common_analysis_cell_count_complete"
        ]
        and
        qa[
            "common_analysis_cells_share_identical_dates"
        ]
        and
        qa[
            "common_analysis_row_count"
        ]
        == qa[
            "expected_common_analysis_row_count"
        ]
        and
        qa[
            "common_analysis_nonfinite_core_metric_count"
        ] == 0
        and
        qa[
            "valid_portfolio_return_coverage_violation_count"
        ] == 0
        and
        qa[
            "nonfinite_valid_core_metric_count"
        ] == 0
        and
        not qa[
            "portfolio_formation_uses_future_return_availability"
        ]
    )

    # --------------------------------------------------------
    # 16.7 Save
    # --------------------------------------------------------

    atomic_csv(
        monthly,
        MONTHLY_OUT,
    )

    atomic_csv(
        annual,
        ANNUAL_OUT,
    )

    atomic_csv(
        summary,
        SUMMARY_OUT,
    )

    atomic_csv(
        curve,
        CURVE_OUT,
    )

    atomic_csv(
        cumulative,
        CUMULATIVE_OUT,
    )

    atomic_csv(
        date_qa,
        DATE_QA_OUT,
    )

    atomic_csv(
        common_eval_manifest,
        PORTFOLIO_COMMON_DATE_OUT,
    )

    atomic_csv(
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

    # --------------------------------------------------------
    # 16.8 Metadata
    # --------------------------------------------------------

    metadata = {
        "research_day":
            12,

        "step":
            (
                "Step3_Monotonic_Portfolio_and_"
                "Long_Short_Validation"
            ),

        "input_panel":
            str(
                PANEL_PATH
            ),

        "step2_common_date_manifest":
            str(
                STEP2_COMMON_DATE_PATH
            ),

        "windows":
            WINDOWS,

        "frozen_factors":
            FACTORS,

        "target":
            TARGET,

        "portfolio_groups":
            {
                "count":
                    N_GROUPS,

                "definition":
                    (
                        "Q1 is the lowest current signal quintile "
                        "and Q5 is the highest current signal quintile."
                    ),
            },

        "methods": {
            METHOD_RAW:
                (
                    "Sort stocks directly by the frozen network "
                    "factor. No factor sign is flipped."
                ),

            METHOD_NEUTRAL:
                (
                    "Rank-z the current network factor and "
                    "residualize it on the same frozen industry FE "
                    "and five ranked traditional characteristics "
                    "used in Step 2. Sort stocks by this current-"
                    "information residual score. Future return is "
                    "NOT residualized in the portfolio backtest."
                ),
        },

        "sample_scopes": {
            SCOPE_NATIVE:
                (
                    "RAW forms portfolios from the current frozen "
                    "network-complete universe. NEUTRAL forms "
                    "portfolios from the current network-complete "
                    "and control-complete universe."
                ),

            SCOPE_COMMON:
                (
                    "Starts from the frozen Step-2 common-date "
                    "manifest. Both RAW and NEUTRAL use the same "
                    "current neutral formation universe. Step 3 then "
                    "forms a stricter portfolio-common evaluation "
                    "calendar containing only dates for which every "
                    "RAW/NEUTRAL x window x factor portfolio cell "
                    "passes the pre-frozen formation-size and 95% "
                    "realized-return-coverage rules. This keeps the "
                    "final COMMON_DATE summaries on identical dates "
                    "without using future-return availability in "
                    "portfolio formation."
                ),
        },

        "anti_leakage_rule":
            (
                "Portfolio assignment is performed before using "
                "future-return availability. The frozen future "
                "holding return and its validity flag are used only "
                "to evaluate realized group returns after formation."
            ),

        "portfolio_common_date_manifest":
            str(
                PORTFOLIO_COMMON_DATE_OUT
            ),

        "portfolio_common_date_rule":
            (
                "A Step-2 common date enters Step-3 COMMON_DATE "
                "summary statistics only if all 54 frozen portfolio "
                "cells (2 methods x 3 windows x 9 factors) pass "
                "the pre-frozen portfolio-validity rules. This is "
                "an evaluation-quality intersection, not a signal-"
                "formation or return-value selection rule."
            ),

        "realized_return_rule":
            (
                "Each quintile's realized return is the equal-"
                "weighted mean of members with valid frozen future "
                "holding returns, provided formation count, realized "
                "count, and return coverage satisfy the pre-frozen "
                "minimum rules."
            ),

        "minimum_formation_n":
            MIN_FORMATION_N,

        "minimum_group_formation_n":
            MIN_GROUP_FORMATION_N,

        "minimum_group_realized_n":
            MIN_GROUP_REALIZED_N,

        "minimum_group_return_coverage":
            MIN_GROUP_RETURN_COVERAGE,

        "primary_spread":
            (
                "Q5 minus Q1. The direction is never changed based "
                "on Step-2 IC results. Therefore a genuinely "
                "negative alpha relation should appear as a "
                "negative Q5-Q1 spread."
            ),

        "monotonicity_metrics":
            [
                "monthly Spearman correlation between quintile number and quintile return",
                "monthly linear slope across Q1-Q5 returns",
                "strict increasing-month share",
                "strict decreasing-month share",
                "Spearman correlation of the full-sample mean Q1-Q5 return curve",
            ],

        "long_short_metrics":
            [
                "mean and median Q5-Q1",
                "positive and negative month shares",
                "annualized arithmetic Q5-Q1",
                "annualized volatility",
                "annualized Sharpe",
                "additive P&L max drawdown",
                "Newey-West HAC t statistic",
                "Benjamini-Hochberg q value",
                "annual sign stability",
            ],

        "cost_capacity_boundary":
            (
                "Step 3 is a gross portfolio-sort validation only. "
                "No transaction cost, ADV filter, turnover penalty, "
                "or capacity constraint is applied here. Those are "
                "reserved for Day-12 Steps 4-5."
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

    # --------------------------------------------------------
    # 16.9 Console
    # --------------------------------------------------------

    print()
    print(
        "Step-2 common dates: "
        f"{len(common_date_set)}"
    )

    print(
        "Step-3 portfolio-common dates: "
        f"{len(portfolio_common_date_set)}"
    )

    print()
    print(
        "Portfolio validation summary:"
    )

    show_cols = [
        "sample_scope",
        "method",
        "window",
        "factor",
        "date_n",
        "mean_q1_return",
        "mean_q5_return",
        "mean_high_minus_low",
        "mean_monthly_quintile_spearman",
        "mean_quintile_curve_spearman",
        "same_sign_hl_year_share",
        "annualized_sharpe_hl",
        "hac_t_high_minus_low",
        "bh_q_high_minus_low",
    ]

    print(
        summary[
            show_cols
        ].to_string(
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
            "Day-12 Step-3 formal QA failed. "
            "Inspect day12_step3_qa.csv before proceeding."
        )

    print()
    print("=" * 86)
    print("DAY 12 STEP 3 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 86)


if __name__ == "__main__":
    main()
