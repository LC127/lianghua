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

STEP1 = DAY12 / "01_stage1_frozen_alpha_evaluation_panel"

PANEL_PATH = STEP1 / "frozen_alpha_evaluation_panel.parquet"
STEP1_QA_PATH = STEP1 / "day12_step1_qa.csv"
STEP1_META_PATH = STEP1 / "day12_step1_metadata.json"

OUTDIR = DAY12 / "02_stage2_alpha_ic_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

MONTHLY_OUT = OUTDIR / "monthly_alpha_rank_ic.csv"
DATE_QA_OUT = OUTDIR / "alpha_ic_date_window_qa.csv"
COMMON_DATE_OUT = OUTDIR / "alpha_ic_common_date_manifest.csv"
ANNUAL_OUT = OUTDIR / "annual_alpha_ic_summary.csv"
SUMMARY_OUT = OUTDIR / "alpha_ic_comparison_summary.csv"
MATRIX_OUT = OUTDIR / "alpha_ic_matrix.csv"
QA_OUT = OUTDIR / "day12_step2_qa.csv"
META_OUT = OUTDIR / "day12_step2_metadata.json"


# ============================================================
# 1. Frozen Day-12 design
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

METHOD_RAW = "RAW_RANK_IC"
METHOD_NEUTRAL = "FULL_CHARACTERISTIC_NEUTRAL"

SCOPE_NATIVE = "NATIVE"
SCOPE_COMMON = "COMMON_DATE"

MIN_CROSS_SECTION_N = 100

# Supporting time-series inference for monthly IC series.
NW_LAG = 3

EPS = 1e-12
LSTSQ_RCOND = 1e-10


# ============================================================
# 2. Helpers
# ============================================================

def atomic_csv(df, path):
    path = Path(path)
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


def load_qa(path):
    q = pd.read_csv(path)

    required = {
        "qa_name",
        "qa_value",
    }

    if not required.issubset(q.columns):
        raise RuntimeError(
            f"Invalid QA file: {path}"
        )

    return dict(
        zip(
            q["qa_name"],
            q["qa_value"],
        )
    )


def require_columns(df, columns, name):
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
# 3. Cross-sectional rank helpers
# ============================================================

def percentile_rank(x):
    """
    Average-tie percentile rank, computed within one date-window
    cross-section.
    """
    return (
        pd.Series(
            np.asarray(x, dtype=float)
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(dtype=float)
    )


def rank_z(x):
    """
    Percentile rank, then z-standardize.

    Returns None when the cross-section is degenerate.
    """
    r = percentile_rank(x)

    if len(r) < 2:
        return None

    sd = float(
        np.std(r, ddof=1)
    )

    if (
        not np.isfinite(sd)
        or sd <= EPS
    ):
        return None

    return (
        r - np.mean(r)
    ) / sd


def rank_z_frame(df, columns):
    """
    Rank-z transform several continuous variables.
    Constant variables are set to zero and marked inactive.
    """
    z = np.zeros(
        (
            len(df),
            len(columns),
        ),
        dtype=float,
    )

    active = []

    for j, col in enumerate(columns):
        v = rank_z(
            df[col].to_numpy(dtype=float)
        )

        if v is None:
            active.append(False)
            z[:, j] = 0.0
        else:
            active.append(True)
            z[:, j] = v

    return (
        z,
        np.asarray(active, dtype=bool),
    )


def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))

    if (
        not np.isfinite(sx)
        or not np.isfinite(sy)
        or sx <= EPS
        or sy <= EPS
    ):
        return np.nan

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def spearman_rank_ic(x, y):
    """
    Spearman IC = Pearson correlation of within-date ranks.
    """
    xz = rank_z(x)
    yz = rank_z(y)

    if (
        xz is None
        or yz is None
    ):
        return np.nan

    return pearson_corr(xz, yz)


# ============================================================
# 4. FULL_CHARACTERISTIC_NEUTRAL
# ============================================================

def build_control_design(df):
    """
    Explicit Day-12 neutralization design.

    Continuous controls:
        within-date percentile rank -> z score

    Industry:
        full observed industry dummy set

    No additional common intercept is added because the full
    industry dummy set already supplies group-specific intercepts.

    Frozen controls:
        industry_id1
        log_market_value
        momentum_120_20
        reversal_20
        volatility_60
        log_turnover_20
    """

    control_z, active_numeric = rank_z_frame(
        df,
        CONTROL_NUMERIC,
    )

    if active_numeric.any():
        continuous = control_z[:, active_numeric]
    else:
        continuous = np.empty(
            (len(df), 0),
            dtype=float,
        )

    industry = (
        df[INDUSTRY]
        .astype("string")
        .astype(str)
        .to_numpy()
    )

    industry_levels = sorted(
        pd.unique(industry).tolist()
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
                len(industry_levels)
            ),

        "design_column_n":
            int(
                X.shape[1]
            ),
    }


def residualize_rank_block(neutral_df):
    """
    Partial-rank / FWL-style characteristic-neutral IC.

    Step A:
        rank-z each network factor
        rank-z future_holding_return
        rank-z each continuous control

    Step B:
        regress both factor rank and return rank on:
            industry FE
            size
            momentum
            reversal
            volatility
            turnover

    Step C:
        neutral IC = corr(factor residual, return residual)

    The function also returns raw Rank IC on the EXACT same stock
    sample, which is used for paired neutralization diagnostics.
    """

    factor_z, factor_active = rank_z_frame(
        neutral_df,
        FACTORS,
    )

    y_z = rank_z(
        neutral_df[
            TARGET
        ].to_numpy(dtype=float)
    )

    if y_z is None:
        return None

    if not factor_active.all():
        return None

    control_info = build_control_design(
        neutral_df
    )

    X = control_info["X"]

    # Regress 9 factor-ranks and 1 return-rank jointly.
    Y = np.column_stack(
        [
            factor_z,
            y_z,
        ]
    )

    if X.shape[0] <= X.shape[1]:
        return None

    beta, _, rank, singular_values = (
        np.linalg.lstsq(
            X,
            Y,
            rcond=LSTSQ_RCOND,
        )
    )

    residual = (
        Y - X @ beta
    )

    factor_resid = residual[:, :len(FACTORS)]
    return_resid = residual[:, -1]

    neutral_ic = {}
    raw_same_sample_ic = {}

    for j, factor in enumerate(FACTORS):
        neutral_ic[factor] = pearson_corr(
            factor_resid[:, j],
            return_resid,
        )

        raw_same_sample_ic[factor] = pearson_corr(
            factor_z[:, j],
            y_z,
        )

    if (
        singular_values is None
        or len(singular_values) == 0
    ):
        condition = np.nan
    else:
        smax = float(np.max(singular_values))
        smin = float(np.min(singular_values))

        condition = (
            smax / smin
            if (
                np.isfinite(smin)
                and smin > EPS
            )
            else np.inf
        )

    control_info["design_rank"] = int(rank)
    control_info[
        "design_condition_number"
    ] = float(condition)

    return {
        "neutral_ic":
            neutral_ic,

        "raw_same_sample_ic":
            raw_same_sample_ic,

        "control_info":
            control_info,
    }


# ============================================================
# 5. Newey-West inference for monthly mean IC
# ============================================================

def nw_mean_test(values, lag=3):
    """
    HAC test of H0: E[IC_t] = 0.

    Uses Bartlett weights and a normal approximation.
    This is supporting inference only.
    """
    x = np.asarray(values, dtype=float)

    x = x[np.isfinite(x)]

    T = len(x)

    if T < 3:
        return {
            "mean": np.nan,
            "se": np.nan,
            "t": np.nan,
            "p_two_sided": np.nan,
        }

    mu = float(np.mean(x))
    u = x - mu

    L = min(
        int(lag),
        T - 1,
    )

    long_run = float(
        np.dot(u, u) / T
    )

    for ell in range(1, L + 1):
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

    if (
        not np.isfinite(se)
        or se <= EPS
    ):
        return {
            "mean": mu,
            "se": np.nan,
            "t": np.nan,
            "p_two_sided": np.nan,
        }

    t_stat = mu / se

    # Two-sided normal-approximation p-value.
    p_two = math.erfc(
        abs(t_stat)
        / math.sqrt(2.0)
    )

    return {
        "mean": mu,
        "se": float(se),
        "t": float(t_stat),
        "p_two_sided": float(p_two),
    }


# ============================================================
# 6. Benjamini-Hochberg q-values
# ============================================================

def bh_qvalues(p_values):
    p = np.asarray(
        p_values,
        dtype=float,
    )

    q = np.full(
        len(p),
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(p)

    if not valid.any():
        return q

    idx = np.where(valid)[0]
    pv = p[idx]

    order = np.argsort(pv)
    sorted_p = pv[order]

    m = len(sorted_p)

    raw_q = (
        sorted_p
        * m
        / np.arange(1, m + 1)
    )

    monotone_q = np.minimum.accumulate(
        raw_q[::-1]
    )[::-1]

    monotone_q = np.clip(
        monotone_q,
        0.0,
        1.0,
    )

    q[
        idx[order]
    ] = monotone_q

    return q


# ============================================================
# 7. One date-window evaluation
# ============================================================

def evaluate_date_window(
    g,
    window,
    date,
):
    rows = []

    # --------------------------------------------------------
    # RAW native sample
    # --------------------------------------------------------

    raw_mask = (
        g[
            "raw_alpha_evaluable"
        ]
        .fillna(False)
        .astype(bool)
    )

    raw = (
        g.loc[
            raw_mask,
            [
                TARGET,
                *FACTORS,
            ],
        ]
        .copy()
    )

    raw_n = int(len(raw))

    raw_valid = (
        raw_n
        >= MIN_CROSS_SECTION_N
    )

    raw_ic = {
        f: np.nan
        for f in FACTORS
    }

    if raw_valid:
        y = raw[
            TARGET
        ].to_numpy(dtype=float)

        for factor in FACTORS:
            raw_ic[factor] = spearman_rank_ic(
                raw[
                    factor
                ].to_numpy(dtype=float),
                y,
            )

    # --------------------------------------------------------
    # Characteristic-neutral sample
    # --------------------------------------------------------

    neutral_mask = (
        g[
            "neutral_alpha_evaluable"
        ]
        .fillna(False)
        .astype(bool)
    )

    neutral = (
        g.loc[
            neutral_mask,
            [
                TARGET,
                INDUSTRY,
                *CONTROL_NUMERIC,
                *FACTORS,
            ],
        ]
        .copy()
    )

    neutral_n = int(len(neutral))

    neutral_valid = (
        neutral_n
        >= MIN_CROSS_SECTION_N
    )

    neutral_result = None

    if neutral_valid:
        neutral_result = residualize_rank_block(
            neutral
        )

    if neutral_result is None:
        neutral_valid = False

        neutral_ic = {
            f: np.nan
            for f in FACTORS
        }

        raw_same_sample_ic = {
            f: np.nan
            for f in FACTORS
        }

        control_info = {
            "active_numeric_control_n": np.nan,
            "industry_level_n": np.nan,
            "design_column_n": np.nan,
            "design_rank": np.nan,
            "design_condition_number": np.nan,
        }

    else:
        neutral_ic = neutral_result[
            "neutral_ic"
        ]

        raw_same_sample_ic = neutral_result[
            "raw_same_sample_ic"
        ]

        control_info = neutral_result[
            "control_info"
        ]

    # --------------------------------------------------------
    # One output row per frozen factor
    # --------------------------------------------------------

    for factor in FACTORS:
        ri = raw_ic[factor]
        rs = raw_same_sample_ic[factor]
        ni = neutral_ic[factor]

        rows.append(
            {
                "window":
                    int(window),

                "analysis_date":
                    pd.Timestamp(date),

                "factor":
                    factor,

                "raw_n":
                    raw_n,

                "neutral_n":
                    neutral_n,

                "raw_native_valid":
                    bool(
                        raw_valid
                        and np.isfinite(ri)
                    ),

                "neutral_valid":
                    bool(
                        neutral_valid
                        and np.isfinite(ni)
                    ),

                # RAW on Step-1 raw-alpha sample.
                "raw_rank_ic":
                    ri,

                # RAW recomputed on exact neutral stock sample.
                "raw_rank_ic_on_neutral_sample":
                    rs,

                # Partial-rank characteristic-neutral IC.
                "neutral_rank_ic":
                    ni,

                # Same-date, same-stock-sample transformation.
                "neutral_minus_raw_same_sample_ic":
                    (
                        ni - rs
                        if (
                            np.isfinite(ni)
                            and np.isfinite(rs)
                        )
                        else np.nan
                    ),

                "active_numeric_control_n":
                    control_info[
                        "active_numeric_control_n"
                    ],

                "industry_level_n":
                    control_info[
                        "industry_level_n"
                    ],

                "control_design_column_n":
                    control_info[
                        "design_column_n"
                    ],

                "control_design_rank":
                    control_info[
                        "design_rank"
                    ],

                "control_design_condition_number":
                    control_info[
                        "design_condition_number"
                    ],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 8. COMMON_DATE construction
# ============================================================

def build_common_date_manifest(monthly):
    """
    COMMON_DATE requires that all 9 factors have a valid
    characteristic-neutral IC in all 3 frozen windows.

    In COMMON_DATE:
      - dates are identical across W60/W120/W252;
      - RAW and NEUTRAL are evaluated on the same neutral stock
        sample for each date-window.
    """

    per_window_date = (
        monthly.groupby(
            [
                "window",
                "analysis_date",
            ],
            as_index=False,
        )
        .agg(
            factor_n=(
                "factor",
                "nunique",
            ),

            neutral_valid_factor_n=(
                "neutral_valid",
                "sum",
            ),
        )
    )

    per_window_date[
        "window_date_valid"
    ] = (
        per_window_date[
            "factor_n"
        ] == len(FACTORS)
    ) & (
        per_window_date[
            "neutral_valid_factor_n"
        ] == len(FACTORS)
    )

    valid = (
        per_window_date.loc[
            per_window_date[
                "window_date_valid"
            ],
            [
                "window",
                "analysis_date",
            ],
        ]
    )

    date_counts = (
        valid.groupby(
            "analysis_date"
        )[
            "window"
        ]
        .nunique()
    )

    common_dates = (
        date_counts[
            date_counts
            == len(WINDOWS)
        ]
        .index
    )

    manifest = (
        pd.DataFrame(
            {
                "analysis_date":
                    pd.to_datetime(
                        common_dates
                    )
            }
        )
        .sort_values(
            "analysis_date"
        )
        .reset_index(drop=True)
    )

    manifest[
        "common_date"
    ] = True

    return (
        manifest,
        per_window_date,
    )


# ============================================================
# 9. Build NATIVE and COMMON_DATE monthly series
# ============================================================

def build_scope_monthly(
    monthly,
    common_dates,
):
    parts = []

    # --------------------------------------------------------
    # NATIVE
    #
    # RAW uses its native raw sample.
    # NEUTRAL uses its native neutral sample.
    # paired_delta always compares neutral vs raw on SAME stocks.
    # --------------------------------------------------------

    native = monthly.copy()

    native[
        "sample_scope"
    ] = SCOPE_NATIVE

    native[
        "raw_ic_for_scope"
    ] = native[
        "raw_rank_ic"
    ]

    native[
        "neutral_ic_for_scope"
    ] = native[
        "neutral_rank_ic"
    ]

    native[
        "paired_delta_ic"
    ] = native[
        "neutral_minus_raw_same_sample_ic"
    ]

    parts.append(native)

    # --------------------------------------------------------
    # COMMON_DATE
    #
    # Same dates across all windows.
    # RAW is also forced onto neutral sample, so Raw-vs-Neutral
    # holds BOTH date composition and stock composition fixed.
    # --------------------------------------------------------

    common_set = set(
        pd.to_datetime(common_dates)
    )

    common = monthly[
        monthly[
            "analysis_date"
        ].isin(common_set)
    ].copy()

    common[
        "sample_scope"
    ] = SCOPE_COMMON

    common[
        "raw_ic_for_scope"
    ] = common[
        "raw_rank_ic_on_neutral_sample"
    ]

    common[
        "neutral_ic_for_scope"
    ] = common[
        "neutral_rank_ic"
    ]

    common[
        "paired_delta_ic"
    ] = common[
        "neutral_minus_raw_same_sample_ic"
    ]

    parts.append(common)

    return (
        pd.concat(
            parts,
            ignore_index=True,
        )
        .sort_values(
            [
                "sample_scope",
                "window",
                "factor",
                "analysis_date",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# 10. Annual summary
# ============================================================

def build_annual_summary(
    scope_monthly,
):
    x = scope_monthly.copy()

    x["year"] = (
        x[
            "analysis_date"
        ]
        .dt.year
        .astype(int)
    )

    rows = []

    for (
        scope,
        window,
        factor,
        year,
    ), g in x.groupby(
        [
            "sample_scope",
            "window",
            "factor",
            "year",
        ],
        sort=True,
    ):
        raw = (
            g["raw_ic_for_scope"]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        neutral = (
            g["neutral_ic_for_scope"]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        delta = (
            g["paired_delta_ic"]
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

                "window":
                    int(window),

                "factor":
                    factor,

                "year":
                    int(year),

                "raw_month_n":
                    int(len(raw)),

                "neutral_month_n":
                    int(len(neutral)),

                "paired_month_n":
                    int(len(delta)),

                "mean_raw_ic":
                    (
                        float(raw.mean())
                        if len(raw)
                        else np.nan
                    ),

                "median_raw_ic":
                    (
                        float(raw.median())
                        if len(raw)
                        else np.nan
                    ),

                "positive_raw_month_share":
                    (
                        float(
                            (raw > 0).mean()
                        )
                        if len(raw)
                        else np.nan
                    ),

                "mean_neutral_ic":
                    (
                        float(
                            neutral.mean()
                        )
                        if len(neutral)
                        else np.nan
                    ),

                "median_neutral_ic":
                    (
                        float(
                            neutral.median()
                        )
                        if len(neutral)
                        else np.nan
                    ),

                "positive_neutral_month_share":
                    (
                        float(
                            (
                                neutral > 0
                            ).mean()
                        )
                        if len(neutral)
                        else np.nan
                    ),

                "mean_paired_delta_ic":
                    (
                        float(
                            delta.mean()
                        )
                        if len(delta)
                        else np.nan
                    ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "sample_scope",
                "window",
                "factor",
                "year",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# 11. Overall summary
# ============================================================

def build_summary(
    scope_monthly,
    annual,
):
    rows = []

    for (
        scope,
        window,
        factor,
    ), g in scope_monthly.groupby(
        [
            "sample_scope",
            "window",
            "factor",
        ],
        sort=True,
    ):
        g = g.sort_values(
            "analysis_date"
        )

        raw = (
            g["raw_ic_for_scope"]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        neutral = (
            g["neutral_ic_for_scope"]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        delta = (
            g["paired_delta_ic"]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        nw_raw = nw_mean_test(
            raw,
            lag=NW_LAG,
        )

        nw_neutral = nw_mean_test(
            neutral,
            lag=NW_LAG,
        )

        nw_delta = nw_mean_test(
            delta,
            lag=NW_LAG,
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
            &
            (
                annual[
                    "factor"
                ] == factor
            )
        ].copy()

        mean_neutral = (
            float(neutral.mean())
            if len(neutral)
            else np.nan
        )

        if (
            np.isfinite(mean_neutral)
            and mean_neutral != 0
            and len(ann)
        ):
            sign_target = (
                1
                if mean_neutral > 0
                else -1
            )

            annual_sign = np.sign(
                ann[
                    "mean_neutral_ic"
                ].to_numpy(dtype=float)
            )

            same_sign_year_share = float(
                np.mean(
                    annual_sign
                    == sign_target
                )
            )
        else:
            same_sign_year_share = np.nan

        positive_neutral_year_share = (
            float(
                (
                    ann[
                        "mean_neutral_ic"
                    ] > 0
                ).mean()
            )
            if len(ann)
            else np.nan
        )

        rows.append(
            {
                "sample_scope":
                    scope,

                "window":
                    int(window),

                "factor":
                    factor,

                "raw_date_n":
                    int(len(raw)),

                "neutral_date_n":
                    int(len(neutral)),

                "paired_date_n":
                    int(len(delta)),

                "mean_raw_ic":
                    (
                        float(raw.mean())
                        if len(raw)
                        else np.nan
                    ),

                "median_raw_ic":
                    (
                        float(raw.median())
                        if len(raw)
                        else np.nan
                    ),

                "positive_raw_month_share":
                    (
                        float(
                            (raw > 0).mean()
                        )
                        if len(raw)
                        else np.nan
                    ),

                "hac_se_raw_ic":
                    nw_raw["se"],

                "hac_t_raw_ic":
                    nw_raw["t"],

                "hac_p_two_sided_raw_ic":
                    nw_raw[
                        "p_two_sided"
                    ],

                "mean_neutral_ic":
                    mean_neutral,

                "median_neutral_ic":
                    (
                        float(
                            neutral.median()
                        )
                        if len(neutral)
                        else np.nan
                    ),

                "positive_neutral_month_share":
                    (
                        float(
                            (
                                neutral > 0
                            ).mean()
                        )
                        if len(neutral)
                        else np.nan
                    ),

                "hac_se_neutral_ic":
                    nw_neutral["se"],

                "hac_t_neutral_ic":
                    nw_neutral["t"],

                "hac_p_two_sided_neutral_ic":
                    nw_neutral[
                        "p_two_sided"
                    ],

                "mean_paired_delta_ic":
                    (
                        float(
                            delta.mean()
                        )
                        if len(delta)
                        else np.nan
                    ),

                "median_paired_delta_ic":
                    (
                        float(
                            delta.median()
                        )
                        if len(delta)
                        else np.nan
                    ),

                "positive_paired_delta_month_share":
                    (
                        float(
                            (
                                delta > 0
                            ).mean()
                        )
                        if len(delta)
                        else np.nan
                    ),

                "hac_se_paired_delta_ic":
                    nw_delta["se"],

                "hac_t_paired_delta_ic":
                    nw_delta["t"],

                "hac_p_two_sided_paired_delta_ic":
                    nw_delta[
                        "p_two_sided"
                    ],

                "year_n":
                    int(
                        ann[
                            "year"
                        ].nunique()
                    ),

                "positive_neutral_year_share":
                    positive_neutral_year_share,

                "same_sign_neutral_year_share":
                    same_sign_year_share,
            }
        )

    out = pd.DataFrame(rows)

    # BH correction separately inside each sample scope.
    out["bh_q_raw_ic"] = np.nan
    out["bh_q_neutral_ic"] = np.nan
    out["bh_q_paired_delta_ic"] = np.nan

    for scope in [
        SCOPE_NATIVE,
        SCOPE_COMMON,
    ]:
        mask = (
            out[
                "sample_scope"
            ] == scope
        )

        out.loc[
            mask,
            "bh_q_raw_ic",
        ] = bh_qvalues(
            out.loc[
                mask,
                "hac_p_two_sided_raw_ic",
            ].to_numpy(dtype=float)
        )

        out.loc[
            mask,
            "bh_q_neutral_ic",
        ] = bh_qvalues(
            out.loc[
                mask,
                "hac_p_two_sided_neutral_ic",
            ].to_numpy(dtype=float)
        )

        out.loc[
            mask,
            "bh_q_paired_delta_ic",
        ] = bh_qvalues(
            out.loc[
                mask,
                "hac_p_two_sided_paired_delta_ic",
            ].to_numpy(dtype=float)
        )

    return (
        out.sort_values(
            [
                "sample_scope",
                "window",
                "factor",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# 12. Compact matrix
# ============================================================

def build_matrix(summary):
    """
    18 rows:
        2 scopes x 3 windows x 3 metrics.
    """
    metric_map = {
        "RAW_MEAN_IC":
            "mean_raw_ic",

        "NEUTRAL_MEAN_IC":
            "mean_neutral_ic",

        "NEUTRAL_MINUS_RAW_SAME_SAMPLE":
            "mean_paired_delta_ic",
    }

    rows = []

    for scope in [
        SCOPE_NATIVE,
        SCOPE_COMMON,
    ]:
        for window in WINDOWS:
            sub = summary[
                (
                    summary[
                        "sample_scope"
                    ] == scope
                )
                &
                (
                    summary[
                        "window"
                    ] == window
                )
            ]

            for metric_name, col \
                in metric_map.items():

                row = {
                    "sample_scope":
                        scope,

                    "window":
                        int(window),

                    "metric":
                        metric_name,
                }

                for factor in FACTORS:
                    hit = sub[
                        sub[
                            "factor"
                        ] == factor
                    ]

                    row[factor] = (
                        float(
                            hit[col].iloc[0]
                        )
                        if (
                            len(hit)
                            and pd.notna(
                                hit[col].iloc[0]
                            )
                        )
                        else np.nan
                    )

                rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 13. Main
# ============================================================

def main():
    print("=" * 84)
    print("M1 - Research Day 12 - Step 2")
    print("Frozen Alpha IC and Characteristic-Neutral Validation")
    print("=" * 84)

    # --------------------------------------------------------
    # 13.1 Validate frozen Step-1 source
    # --------------------------------------------------------

    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            PANEL_PATH
        )

    if not STEP1_QA_PATH.exists():
        raise FileNotFoundError(
            STEP1_QA_PATH
        )

    if not STEP1_META_PATH.exists():
        raise FileNotFoundError(
            STEP1_META_PATH
        )

    step1_qa = load_qa(
        STEP1_QA_PATH
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

    with open(
        STEP1_META_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        step1_meta = json.load(f)

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
            "Step-1 return target mismatch."
        )

    # --------------------------------------------------------
    # 13.2 Read ONLY Step-1 frozen panel
    # --------------------------------------------------------

    columns = [
        "security_id",
        "analysis_date",
        "window",
        TARGET,
        INDUSTRY,
        *CONTROL_NUMERIC,
        *FACTORS,
        "raw_alpha_evaluable",
        "neutral_alpha_evaluable",
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
        panel[key]
        .duplicated()
        .sum()
    )

    if duplicate_key_n > 0:
        raise RuntimeError(
            "Duplicate stock-date-window key in Step-1 panel."
        )

    # --------------------------------------------------------
    # 13.3 Date-window IC calculation
    # --------------------------------------------------------

    monthly_parts = []
    date_qa_rows = []

    grouped = panel.groupby(
        [
            "window",
            "analysis_date",
        ],
        sort=True,
    )

    total_groups = grouped.ngroups

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
            or counter % 25 == 0
            or counter == total_groups
        ):
            print(
                f"[{counter:>3}/{total_groups}] "
                f"W{int(window)} "
                f"{pd.Timestamp(date).date()}"
            )

        result = evaluate_date_window(
            g,
            window,
            date,
        )

        monthly_parts.append(result)

        first = result.iloc[0]

        rank_value = (
            first[
                "control_design_rank"
            ]
        )

        col_value = (
            first[
                "control_design_column_n"
            ]
        )

        control_full_rank = bool(
            pd.notna(rank_value)
            and pd.notna(col_value)
            and int(rank_value)
            == int(col_value)
        )

        date_qa_rows.append(
            {
                "window":
                    int(window),

                "analysis_date":
                    pd.Timestamp(date),

                "raw_n":
                    int(
                        first[
                            "raw_n"
                        ]
                    ),

                "neutral_n":
                    int(
                        first[
                            "neutral_n"
                        ]
                    ),

                "raw_valid_factor_n":
                    int(
                        result[
                            "raw_native_valid"
                        ].sum()
                    ),

                "neutral_valid_factor_n":
                    int(
                        result[
                            "neutral_valid"
                        ].sum()
                    ),

                "active_numeric_control_n":
                    first[
                        "active_numeric_control_n"
                    ],

                "industry_level_n":
                    first[
                        "industry_level_n"
                    ],

                "control_design_column_n":
                    first[
                        "control_design_column_n"
                    ],

                "control_design_rank":
                    first[
                        "control_design_rank"
                    ],

                "control_design_full_rank":
                    control_full_rank,

                "control_design_condition_number":
                    first[
                        "control_design_condition_number"
                    ],
            }
        )

    monthly = (
        pd.concat(
            monthly_parts,
            ignore_index=True,
        )
        .sort_values(
            [
                "window",
                "factor",
                "analysis_date",
            ]
        )
        .reset_index(drop=True)
    )

    date_qa = pd.DataFrame(
        date_qa_rows
    )

    # --------------------------------------------------------
    # 13.4 COMMON_DATE
    # --------------------------------------------------------

    (
        common_manifest,
        common_window_date,
    ) = build_common_date_manifest(
        monthly
    )

    common_dates = (
        common_manifest[
            "analysis_date"
        ]
        .tolist()
    )

    scope_monthly = build_scope_monthly(
        monthly,
        common_dates,
    )

    # --------------------------------------------------------
    # 13.5 Annual and full-sample summaries
    # --------------------------------------------------------

    annual = build_annual_summary(
        scope_monthly
    )

    summary = build_summary(
        scope_monthly,
        annual,
    )

    matrix = build_matrix(
        summary
    )

    # --------------------------------------------------------
    # 13.6 Formal QA
    # --------------------------------------------------------

    date_window_n = int(
        panel[
            [
                "window",
                "analysis_date",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    expected_monthly_rows = (
        date_window_n
        * len(FACTORS)
    )

    expected_summary_rows = (
        2
        * len(WINDOWS)
        * len(FACTORS)
    )

    expected_matrix_rows = (
        2
        * len(WINDOWS)
        * 3
    )

    monthly_dup_n = int(
        monthly[
            [
                "window",
                "analysis_date",
                "factor",
            ]
        ]
        .duplicated()
        .sum()
    )

    summary_dup_n = int(
        summary[
            [
                "sample_scope",
                "window",
                "factor",
            ]
        ]
        .duplicated()
        .sum()
    )

    common_date_count = int(
        len(common_manifest)
    )

    common_rows = scope_monthly[
        scope_monthly[
            "sample_scope"
        ] == SCOPE_COMMON
    ]

    expected_common_rows = (
        common_date_count
        * len(WINDOWS)
        * len(FACTORS)
    )

    common_nonfinite_n = int(
        (
            ~np.isfinite(
                common_rows[
                    [
                        "raw_ic_for_scope",
                        "neutral_ic_for_scope",
                        "paired_delta_ic",
                    ]
                ]
                .to_numpy(dtype=float)
            )
        ).sum()
    )

    neutral_date_qa = date_qa[
        date_qa[
            "neutral_valid_factor_n"
        ] == len(FACTORS)
    ]

    rank_deficient_date_window_n = int(
        (
            ~neutral_date_qa[
                "control_design_full_rank"
            ]
        ).sum()
    )

    qa = {
        "step1_formal_qa_pass":
            True,

        "frozen_panel_row_count":
            int(len(panel)),

        "duplicate_stock_date_window_count":
            duplicate_key_n,

        "frozen_factor_count":
            len(FACTORS),

        "expected_factor_count":
            9,

        "window_count":
            int(
                panel[
                    "window"
                ].nunique()
            ),

        "expected_window_count":
            3,

        "frozen_control_count":
            len(CONTROLS),

        "date_window_count":
            date_window_n,

        "monthly_result_row_count":
            int(len(monthly)),

        "expected_monthly_result_row_count":
            int(
                expected_monthly_rows
            ),

        "monthly_duplicate_date_window_factor_count":
            monthly_dup_n,

        "common_date_count":
            common_date_count,

        "common_scope_row_count":
            int(
                len(common_rows)
            ),

        "expected_common_scope_row_count":
            int(
                expected_common_rows
            ),

        "common_scope_nonfinite_core_metric_count":
            common_nonfinite_n,

        "summary_row_count":
            int(len(summary)),

        "expected_summary_row_count":
            int(
                expected_summary_rows
            ),

        "summary_duplicate_scope_window_factor_count":
            summary_dup_n,

        "matrix_row_count":
            int(len(matrix)),

        "expected_matrix_row_count":
            int(
                expected_matrix_rows
            ),

        # Diagnostic only; not part of pass/fail.
        "neutral_control_rank_deficient_date_window_count":
            rank_deficient_date_window_n,

        "target":
            TARGET,

        "raw_method":
            METHOD_RAW,

        "neutral_method":
            METHOD_NEUTRAL,

        "neutralization_uses_industry_fe":
            True,

        "neutralization_uses_all_5_numeric_controls":
            True,

        "factor_rank_transformed":
            True,

        "return_rank_transformed":
            True,

        "continuous_controls_rank_transformed":
            True,

        "factor_residualized_on_controls":
            True,

        "return_residualized_on_controls":
            True,

        "common_date_raw_and_neutral_use_same_stock_sample":
            True,

        "factor_sign_flipped":
            False,

        "factor_selected_from_results":
            False,

        "window_selected_from_results":
            False,

        "date_selected_from_results":
            False,

        "future_return_used_as_predictor":
            False,

        "return_quality_fields_used_as_predictors":
            False,

        "network_factors_reestimated":
            False,

        "controls_reestimated":
            False,

        "forward_returns_reestimated":
            False,
    }

    qa[
        "all_formal_qa_pass"
    ] = bool(
        qa[
            "step1_formal_qa_pass"
        ]
        and qa[
            "duplicate_stock_date_window_count"
        ] == 0
        and qa[
            "frozen_factor_count"
        ] == 9
        and qa[
            "window_count"
        ] == 3
        and qa[
            "frozen_control_count"
        ] == 6
        and qa[
            "monthly_result_row_count"
        ] == expected_monthly_rows
        and qa[
            "monthly_duplicate_date_window_factor_count"
        ] == 0
        and qa[
            "common_date_count"
        ] > 0
        and qa[
            "common_scope_row_count"
        ] == expected_common_rows
        and qa[
            "common_scope_nonfinite_core_metric_count"
        ] == 0
        and qa[
            "summary_row_count"
        ] == expected_summary_rows
        and qa[
            "summary_duplicate_scope_window_factor_count"
        ] == 0
        and qa[
            "matrix_row_count"
        ] == expected_matrix_rows
    )

    # --------------------------------------------------------
    # 13.7 Save
    # --------------------------------------------------------

    atomic_csv(
        monthly,
        MONTHLY_OUT,
    )

    atomic_csv(
        date_qa,
        DATE_QA_OUT,
    )

    atomic_csv(
        common_manifest,
        COMMON_DATE_OUT,
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
        matrix,
        MATRIX_OUT,
    )

    atomic_csv(
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

    # --------------------------------------------------------
    # 13.8 Metadata
    # --------------------------------------------------------

    metadata = {
        "research_day":
            12,

        "step":
            (
                "Step2_Frozen_Alpha_IC_and_"
                "Characteristic_Neutral_Validation"
            ),

        "input_panel":
            str(PANEL_PATH),

        "input_rule":
            (
                "Read only the frozen Day-12 Step-1 alpha "
                "evaluation panel; do not re-merge Day-5/6/7/8 "
                "sources."
            ),

        "target":
            TARGET,

        "windows":
            WINDOWS,

        "frozen_factors":
            FACTORS,

        "frozen_controls":
            CONTROLS,

        "methods": {
            METHOD_RAW:
                (
                    "Within-date cross-sectional Spearman "
                    "correlation between each frozen network "
                    "factor and frozen future holding return."
                ),

            METHOD_NEUTRAL:
                (
                    "Within-date partial-rank/FWL IC. The network "
                    "factor, future return, and five continuous "
                    "traditional controls are percentile-ranked and "
                    "z-standardized. Both factor rank and return "
                    "rank are residualized on full industry fixed "
                    "effects plus ranked size, momentum, reversal, "
                    "volatility, and turnover. The Pearson "
                    "correlation of the two residuals is reported."
                ),
        },

        "sample_scopes": {
            SCOPE_NATIVE:
                (
                    "RAW uses its Step-1 raw-alpha sample and "
                    "FULL_CHARACTERISTIC_NEUTRAL uses its Step-1 "
                    "neutral-alpha sample. The paired neutral-minus-"
                    "raw diagnostic is always calculated on the same "
                    "neutral stock sample."
                ),

            SCOPE_COMMON:
                (
                    "Only dates valid for all nine frozen factors "
                    "in all three windows are retained. RAW is "
                    "recomputed on the exact same neutral stock "
                    "sample used by FULL_CHARACTERISTIC_NEUTRAL, "
                    "holding both calendar composition and stock "
                    "composition fixed."
                ),
        },

        "minimum_cross_section_n":
            MIN_CROSS_SECTION_N,

        "newey_west_lag":
            NW_LAG,

        "multiple_testing":
            (
                "Benjamini-Hochberg q-values are reported separately "
                "within each sample scope across the 27 frozen "
                "factor-window cells, for raw IC, neutral IC, and "
                "paired neutral-minus-raw IC. They are descriptive "
                "and are not used for factor or window selection."
            ),

        "sign_policy":
            (
                "No factor sign is flipped. Negative IC is retained "
                "as observed rather than mechanically converted into "
                "a positive alpha orientation."
            ),

        "important_method_note":
            (
                "The current retained project metadata identifies "
                "the FULL_CHARACTERISTIC_NEUTRAL method label and "
                "the six frozen controls, but does not independently "
                "document the exact historical Day-7 residualization "
                "equation. This Step-2 script therefore freezes the "
                "explicit partial-rank/FWL definition stated above. "
                "If exact historical Day-7 method replication is "
                "required, compare this definition against the "
                "original Day-7 implementation before claiming exact "
                "methodological identity."
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
    # 13.9 Console
    # --------------------------------------------------------

    print()
    print(
        "Common dates across all three windows: "
        f"{common_date_count}"
    )

    print()
    print(
        "Alpha IC comparison summary:"
    )

    show_cols = [
        "sample_scope",
        "window",
        "factor",
        "raw_date_n",
        "neutral_date_n",
        "mean_raw_ic",
        "mean_neutral_ic",
        "mean_paired_delta_ic",
        "positive_neutral_month_share",
        "same_sign_neutral_year_share",
        "hac_t_neutral_ic",
        "bh_q_neutral_ic",
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
            "Day-12 Step-2 formal QA failed. "
            "Inspect day12_step2_qa.csv before proceeding."
        )

    print()
    print("=" * 84)
    print("DAY 12 STEP 2 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 84)


if __name__ == "__main__":
    main()
