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

STEP2 = (
    DAY11
    / "02_stage2_factor_redundancy"
)

PANEL = STEP1 / "frozen_prediction_panel.parquet"
CALENDAR = STEP1 / "oos_training_calendar.csv"
STEP1_META = STEP1 / "day11_step1_metadata.json"
STEP1_QA = STEP1 / "day11_step1_qa.csv"

STEP2_META = STEP2 / "day11_step2_metadata.json"
STEP2_QA = STEP2 / "day11_step2_qa.csv"

OUTDIR = (
    DAY11
    / "03_stage3_strict_oos_forecasting"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

PRED_OUT = OUTDIR / "oos_predictions.parquet"
SUMMARY_OUT = OUTDIR / "oos_forecast_summary.csv"
QA_OUT = OUTDIR / "day11_step3_qa.csv"
META_OUT = OUTDIR / "day11_step3_metadata.json"


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

EXPECTED_CONTROLS = [
    INDUSTRY,
    *CONTROL_NUMERIC,
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

MIN_TRAIN_PERIODS = 24
MIN_CROSS_SECTION_N = 100

# Each calendar month gets equal total regression weight.
DATE_BALANCED_WEIGHTING = True

# Numerical stabilization only.
# This is still OLS / Moore-Penrose least squares,
# not ridge regularization.
PINV_RCOND = 1e-10
EPS = 1e-12


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

    return dict(
        zip(
            q["qa_name"],
            q["qa_value"],
        )
    )


def rank_z_frame(df, cols):

    """
    Cross-sectional percentile ranks -> z scores.

    No historical/future distribution is used:
    transformation uses only the current date cross-section.
    """

    r = (
        df[cols]
        .rank(
            axis=0,
            method="average",
            pct=True,
        )
        .astype(float)
    )

    z = np.empty(
        r.shape,
        dtype=np.float64,
    )

    constant_cols = []

    for j, col in enumerate(cols):

        v = r[col].to_numpy(
            dtype=float
        )

        mu = np.mean(v)
        sd = np.std(
            v,
            ddof=1,
        )

        if (
            not np.isfinite(sd)
            or sd <= EPS
        ):
            z[:, j] = 0.0
            constant_cols.append(col)

        else:
            z[:, j] = (
                v - mu
            ) / sd

    return z, constant_cols


def rank_z_target(y):

    r = (
        pd.Series(y)
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(
            dtype=float
        )
    )

    sd = np.std(
        r,
        ddof=1,
    )

    if (
        len(r) < 2
        or
        not np.isfinite(sd)
        or
        sd <= EPS
    ):
        return None

    return (
        r - np.mean(r)
    ) / sd


# ============================================================
# 3. Design matrix
# ============================================================

def build_design(
    g,
    industry_levels,
):

    """
    Build current-date predictors.

    IMPORTANT:
    forecast sample is defined only by predictors.
    Current future target is NOT used here.
    """

    required = (
        CONTROL_NUMERIC
        + FACTORS
        + [INDUSTRY]
    )

    numeric = (
        CONTROL_NUMERIC
        + FACTORS
    )

    mask = (
        g[required]
        .notna()
        .all(axis=1)
    )

    finite = np.isfinite(
        g[numeric]
        .to_numpy(
            dtype=float
        )
    ).all(axis=1)

    mask = (
        mask.to_numpy()
        & finite
    )

    sub = (
        g.loc[mask]
        .copy()
        .reset_index(drop=True)
    )

    if len(sub) < MIN_CROSS_SECTION_N:
        return None

    control_z, c_const = (
        rank_z_frame(
            sub,
            CONTROL_NUMERIC,
        )
    )

    network_z, n_const = (
        rank_z_frame(
            sub,
            FACTORS,
        )
    )

    # --------------------------------------------------------
    # Frozen industry vocabulary.
    #
    # First level is the reference category.
    # --------------------------------------------------------

    ind = (
        sub[INDUSTRY]
        .astype(str)
        .to_numpy()
    )

    dummy_levels = (
        industry_levels[1:]
    )

    if dummy_levels:

        dummies = np.column_stack(
            [
                (ind == level).astype(float)
                for level in dummy_levels
            ]
        )

    else:

        dummies = np.empty(
            (len(sub), 0),
            dtype=float,
        )

    intercept = np.ones(
        (len(sub), 1),
        dtype=float,
    )

    # Baseline:
    # intercept + traditional numeric controls + industry FE
    Xb = np.column_stack(
        [
            intercept,
            control_z,
            dummies,
        ]
    )

    # Network model:
    # same baseline + all 9 frozen network factors
    Xn = np.column_stack(
        [
            intercept,
            control_z,
            network_z,
            dummies,
        ]
    )

    return {
        "data": sub,
        "X_baseline": Xb,
        "X_network": Xn,
        "constant_control_cols": c_const,
        "constant_network_cols": n_const,
    }


# ============================================================
# 4. Historical block sufficient statistics
# ============================================================

def make_train_block(
    design,
):

    """
    Construct one historical Date × Window training block.

    The block is NOT added immediately.
    It remains pending until label_end_date <= forecast_date.
    """

    sub = design["data"]

    usable = (
        sub[TARGET].notna()
        &
        sub["label_end_date"].notna()
    ).to_numpy()

    n = int(
        usable.sum()
    )

    if n < MIN_CROSS_SECTION_N:
        return None

    ends = (
        pd.to_datetime(
            sub.loc[
                usable,
                "label_end_date",
            ]
        )
        .dropna()
        .drop_duplicates()
    )

    if len(ends) != 1:
        raise RuntimeError(
            "Historical target block has "
            "multiple label_end_date values."
        )

    y_raw = (
        sub.loc[
            usable,
            TARGET,
        ]
        .to_numpy(
            dtype=float
        )
    )

    y = rank_z_target(
        y_raw
    )

    if y is None:
        return None

    Xb = (
        design["X_baseline"][
            usable
        ]
    )

    Xn = (
        design["X_network"][
            usable
        ]
    )

    # --------------------------------------------------------
    # Equal total weight per historical date.
    # This avoids later high-stock-count years dominating.
    # --------------------------------------------------------

    scale = (
        1.0 / n
        if DATE_BALANCED_WEIGHTING
        else 1.0
    )

    return {
        "label_end_date":
            pd.Timestamp(
                ends.iloc[0]
            ),

        "n":
            n,

        "b_xtx":
            scale * (Xb.T @ Xb),

        "b_xty":
            scale * (Xb.T @ y),

        "n_xtx":
            scale * (Xn.T @ Xn),

        "n_xty":
            scale * (Xn.T @ y),
    }


# ============================================================
# 5. OLS from expanding sufficient statistics
# ============================================================

def solve_ols(
    xtx,
    xty,
):

    """
    Minimum-norm OLS solution.

    Columns not yet observed (e.g. industry categories that
    have not appeared historically) receive coefficient zero.
    """

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
            np.nan,
            0,
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

    cond = float(
        np.linalg.cond(
            A
        )
    )

    beta_active = (
        np.linalg.pinv(
            A,
            rcond=PINV_RCOND,
        )
        @ b
    )

    beta[
        active
    ] = beta_active

    return (
        beta,
        rank,
        cond,
        int(active.sum()),
    )


# ============================================================
# 6. One network window
# ============================================================

def run_window(
    window,
    calendar,
):

    print()
    print(
        f"[Window W{window}]"
    )

    columns = [
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
        columns=columns,
        filters=[
            ("window", "==", window)
        ],
    )

    df["analysis_date"] = pd.to_datetime(
        df["analysis_date"],
        errors="raise",
    )

    df["label_end_date"] = pd.to_datetime(
        df["label_end_date"],
        errors="coerce",
    )

    df = (
        df.sort_values(
            [
                "analysis_date",
                "security_id",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Fixed industry vocabulary.
    #
    # Only category identities are frozen here;
    # no outcome information is used.
    # --------------------------------------------------------

    industry_levels = sorted(
        df[INDUSTRY]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not industry_levels:
        raise RuntimeError(
            f"W{window}: no industry levels."
        )

    p_baseline = (
        1
        + len(CONTROL_NUMERIC)
        + max(
            len(industry_levels) - 1,
            0,
        )
    )

    p_network = (
        p_baseline
        + len(FACTORS)
    )

    # Expanding sufficient statistics.
    BXX = np.zeros(
        (p_baseline, p_baseline),
        dtype=float,
    )

    BXY = np.zeros(
        p_baseline,
        dtype=float,
    )

    NXX = np.zeros(
        (p_network, p_network),
        dtype=float,
    )

    NXY = np.zeros(
        p_network,
        dtype=float,
    )

    train_period_n = 0
    train_row_n = 0

    max_train_label_end = pd.NaT

    # Historical blocks not yet realized.
    pending = []

    prediction_parts = []
    summary_rows = []

    cal_w = (
        calendar[
            calendar["window"] == window
        ]
        .set_index("forecast_date")
    )

    for date, g in df.groupby(
        "analysis_date",
        sort=True,
    ):

        date = pd.Timestamp(
            date
        )

        # ====================================================
        # 6.1 Activate ONLY labels realized by forecast date
        # ====================================================

        remain = []

        for block in pending:

            if (
                block["analysis_date"] < date
                and
                block["label_end_date"] <= date
            ):

                BXX += block["b_xtx"]
                BXY += block["b_xty"]

                NXX += block["n_xtx"]
                NXY += block["n_xty"]

                train_period_n += 1
                train_row_n += block["n"]

                if (
                    pd.isna(
                        max_train_label_end
                    )
                    or
                    block[
                        "label_end_date"
                    ]
                    >
                    max_train_label_end
                ):
                    max_train_label_end = (
                        block[
                            "label_end_date"
                        ]
                    )

            else:
                remain.append(
                    block
                )

        pending = remain

        # ====================================================
        # 6.2 Current predictor block
        # ====================================================

        design = build_design(
            g,
            industry_levels,
        )

        calendar_24m = False

        if date in cal_w.index:

            raw_flag = (
                cal_w.loc[
                    date,
                    "min_24_periods_met",
                ]
            )

            if isinstance(
                raw_flag,
                pd.Series,
            ):
                raw_flag = (
                    raw_flag.iloc[0]
                )

            calendar_24m = as_bool(
                raw_flag
            )

        forecast_generated = False
        forecast_n = 0
        target_available_n = 0

        b_rank = np.nan
        n_rank = np.nan
        b_cond = np.nan
        n_cond = np.nan
        b_active = np.nan
        n_active = np.nan

        status = "NOT_ELIGIBLE"

        # ====================================================
        # 6.3 Strict OOS forecast
        # ====================================================

        eligible = (
            calendar_24m
            and
            train_period_n
            >= MIN_TRAIN_PERIODS
            and
            design is not None
        )

        if eligible:

            (
                beta_b,
                b_rank,
                b_cond,
                b_active,
            ) = solve_ols(
                BXX,
                BXY,
            )

            (
                beta_n,
                n_rank,
                n_cond,
                n_active,
            ) = solve_ols(
                NXX,
                NXY,
            )

            sub = design["data"]

            pred_b = (
                design[
                    "X_baseline"
                ]
                @ beta_b
            )

            pred_n = (
                design[
                    "X_network"
                ]
                @ beta_n
            )

            out = pd.DataFrame(
                {
                    "security_id":
                        sub[
                            "security_id"
                        ].astype(str),

                    "analysis_date":
                        date,

                    "window":
                        int(window),

                    "industry_id1":
                        sub[
                            INDUSTRY
                        ],

                    "future_idio_vol_annualized":
                        sub[
                            TARGET
                        ],

                    "label_end_date":
                        sub[
                            "label_end_date"
                        ],

                    "baseline_score":
                        pred_b,

                    "network_score":
                        pred_n,

                    "train_period_n":
                        train_period_n,

                    "train_row_n":
                        train_row_n,
                }
            )

            prediction_parts.append(
                out
            )

            forecast_generated = True

            forecast_n = int(
                len(out)
            )

            target_available_n = int(
                out[TARGET]
                .notna()
                .sum()
            )

            status = "FORECAST_GENERATED"

        elif design is None:
            status = (
                "INSUFFICIENT_CURRENT_CROSS_SECTION"
            )

        elif not calendar_24m:
            status = (
                "STEP1_24M_NOT_MET"
            )

        elif (
            train_period_n
            < MIN_TRAIN_PERIODS
        ):
            status = (
                "USABLE_24M_NOT_MET"
            )

        # ----------------------------------------------------
        # Leakage audit
        # ----------------------------------------------------

        leakage = bool(
            pd.notna(
                max_train_label_end
            )
            and
            max_train_label_end
            > date
        )

        summary_rows.append(
            {
                "window":
                    int(window),

                "forecast_date":
                    date,

                "status":
                    status,

                "step1_24m_met":
                    calendar_24m,

                "usable_train_period_n":
                    train_period_n,

                "usable_train_row_n":
                    train_row_n,

                "max_train_label_end_date":
                    max_train_label_end,

                "forecast_generated":
                    forecast_generated,

                "forecast_row_n":
                    forecast_n,

                "forecast_target_available_n":
                    target_available_n,

                "baseline_active_parameter_n":
                    b_active,

                "baseline_matrix_rank":
                    b_rank,

                "baseline_condition_number":
                    b_cond,

                "network_active_parameter_n":
                    n_active,

                "network_matrix_rank":
                    n_rank,

                "network_condition_number":
                    n_cond,

                "training_leakage":
                    leakage,

                "constant_control_n":
                    (
                        len(
                            design[
                                "constant_control_cols"
                            ]
                        )
                        if design is not None
                        else np.nan
                    ),

                "constant_network_factor_n":
                    (
                        len(
                            design[
                                "constant_network_cols"
                            ]
                        )
                        if design is not None
                        else np.nan
                    ),
            }
        )

        # ====================================================
        # 6.4 ONLY AFTER forecasting:
        # build current historical block and put it in pending.
        #
        # It cannot enter the model until label_end <=
        # a later forecast date.
        # ====================================================

        if design is not None:

            block = make_train_block(
                design
            )

            if block is not None:

                block[
                    "analysis_date"
                ] = date

                pending.append(
                    block
                )

    predictions = (
        pd.concat(
            prediction_parts,
            ignore_index=True,
        )
        if prediction_parts
        else pd.DataFrame()
    )

    summary = pd.DataFrame(
        summary_rows
    )

    # Save window-level prediction file immediately.
    if not predictions.empty:

        predictions.to_parquet(
            OUTDIR
            / f"oos_predictions_W{window}.parquet",
            index=False,
            compression="zstd",
        )

    return (
        predictions,
        summary,
        industry_levels,
    )


# ============================================================
# 7. Main
# ============================================================

def main():

    print("=" * 78)
    print("M1 - Research Day 11 - Step 3")
    print("Strict Expanding-Window OOS Risk Forecasting")
    print("=" * 78)

    # ========================================================
    # 7.1 Upstream provenance
    # ========================================================

    step1_qa = load_qa(
        STEP1_QA
    )

    step2_qa = load_qa(
        STEP2_QA
    )

    if not as_bool(
        step1_qa.get(
            "all_formal_qa_pass",
            False,
        )
    ):
        raise RuntimeError(
            "Step-1 formal QA has not passed."
        )

    if not as_bool(
        step2_qa.get(
            "all_formal_qa_pass",
            False,
        )
    ):
        raise RuntimeError(
            "Step-2 formal QA has not passed."
        )

    with open(
        STEP1_META,
        "r",
        encoding="utf-8",
    ) as f:
        step1_meta = json.load(
            f
        )

    with open(
        STEP2_META,
        "r",
        encoding="utf-8",
    ) as f:
        step2_meta = json.load(
            f
        )

    # --------------------------------------------------------
    # Formal Step-3 requires explicitly frozen controls.
    # --------------------------------------------------------

    if not step1_meta.get(
        "control_set_explicitly_confirmed",
        False,
    ):
        raise RuntimeError(
            "\nStep-1 control set is not explicitly frozen.\n"
            "Set CONTROL_COLUMNS_OVERRIDE to:\n"
            f"{EXPECTED_CONTROLS}\n"
            "rerun Step 1, then rerun Step 3."
        )

    if (
        step1_meta.get(
            "control_columns"
        )
        != EXPECTED_CONTROLS
    ):
        raise RuntimeError(
            "Frozen control set does not match "
            "the Day-11 Step-3 specification."
        )

    if (
        step1_meta.get(
            "frozen_factors"
        )
        != FACTORS
    ):
        raise RuntimeError(
            "Step-1 factor list mismatch."
        )

    if (
        step2_meta.get(
            "frozen_factors"
        )
        != FACTORS
    ):
        raise RuntimeError(
            "Step-2 factor list mismatch."
        )

    # ========================================================
    # 7.2 Frozen training calendar
    # ========================================================

    calendar = pd.read_csv(
        CALENDAR
    )

    calendar[
        "forecast_date"
    ] = pd.to_datetime(
        calendar[
            "forecast_date"
        ],
        errors="raise",
    )

    calendar["window"] = (
        pd.to_numeric(
            calendar["window"],
            errors="raise",
        )
        .astype(int)
    )

    # ========================================================
    # 7.3 Run W60 / W120 / W252
    # ========================================================

    all_predictions = []
    all_summaries = []
    industry_info = {}

    for window in WINDOWS:

        (
            pred,
            summary,
            industry_levels,
        ) = run_window(
            window,
            calendar,
        )

        if not pred.empty:
            all_predictions.append(
                pred
            )

        all_summaries.append(
            summary
        )

        industry_info[
            str(window)
        ] = {
            "reference_level":
                industry_levels[0],

            "industry_level_n":
                len(industry_levels),
        }

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    # ========================================================
    # 7.4 Save
    # ========================================================

    predictions = (
        predictions.sort_values(
            [
                "window",
                "analysis_date",
                "security_id",
            ]
        )
        .reset_index(drop=True)
    )

    summary = (
        summary.sort_values(
            [
                "window",
                "forecast_date",
            ]
        )
        .reset_index(drop=True)
    )

    predictions.to_parquet(
        PRED_OUT,
        index=False,
        compression="zstd",
    )

    save_csv(
        summary,
        SUMMARY_OUT,
    )

    # ========================================================
    # 7.5 Formal QA
    # ========================================================

    pred_key = [
        "security_id",
        "analysis_date",
        "window",
    ]

    generated = summary[
        summary[
            "forecast_generated"
        ]
    ].copy()

    leakage_n = int(
        generated[
            "training_leakage"
        ].sum()
    )

    too_early_n = int(
        (
            generated[
                "usable_train_period_n"
            ]
            < MIN_TRAIN_PERIODS
        ).sum()
    )

    step1_not_met_n = int(
        (
            ~generated[
                "step1_24m_met"
            ]
        ).sum()
    )

    score_missing_n = int(
        (
            predictions[
                [
                    "baseline_score",
                    "network_score",
                ]
            ]
            .isna()
            .any(axis=1)
        ).sum()
    )

    qa = {
        "step1_formal_qa_pass":
            True,

        "step1_control_set_confirmed":
            True,

        "step2_formal_qa_pass":
            True,

        "prediction_row_count":
            int(
                len(predictions)
            ),

        "forecast_date_window_count":
            int(
                generated.shape[0]
            ),

        "prediction_window_count":
            int(
                predictions[
                    "window"
                ].nunique()
            ),

        "duplicate_prediction_key_count":
            int(
                predictions[
                    pred_key
                ]
                .duplicated()
                .sum()
            ),

        "prediction_score_missing_count":
            score_missing_n,

        "training_leakage_count":
            leakage_n,

        "forecast_before_24_usable_periods_count":
            too_early_n,

        "forecast_when_step1_24m_not_met_count":
            step1_not_met_n,

        "network_factor_count":
            len(FACTORS),

        "numeric_control_count":
            len(CONTROL_NUMERIC),

        "industry_fixed_effect_used":
            True,

        "date_balanced_training_weight":
            DATE_BALANCED_WEIGHTING,

        "current_target_used_to_define_forecast_sample":
            False,

        "same_date_target_used_in_model_fit":
            False,

        "future_target_used_for_model_selection":
            False,

        "factor_selected_from_results":
            False,

        "factor_removed_after_step2":
            False,

        "window_selected_from_results":
            False,

        "hyperparameter_tuned_on_oos":
            False,

        "oos_performance_evaluated_in_step3":
            False,
    }

    qa[
        "all_formal_qa_pass"
    ] = bool(
        qa[
            "prediction_row_count"
        ] > 0
        and
        qa[
            "forecast_date_window_count"
        ] > 0
        and
        qa[
            "prediction_window_count"
        ] == 3
        and
        qa[
            "duplicate_prediction_key_count"
        ] == 0
        and
        qa[
            "prediction_score_missing_count"
        ] == 0
        and
        qa[
            "training_leakage_count"
        ] == 0
        and
        qa[
            "forecast_before_24_usable_periods_count"
        ] == 0
        and
        qa[
            "forecast_when_step1_24m_not_met_count"
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

    metadata = {
        "research_day": 11,

        "step":
            "Step3_Strict_Expanding_Window_OOS_Risk_Forecasting",

        "target":
            TARGET,

        "windows":
            WINDOWS,

        "baseline_model":
            (
                "Industry FE + five frozen traditional "
                "characteristics."
            ),

        "network_model":
            (
                "Same baseline + all nine frozen "
                "network characteristics."
            ),

        "traditional_controls":
            EXPECTED_CONTROLS,

        "network_factors":
            FACTORS,

        "cross_sectional_transform":
            (
                "Within-date percentile rank followed "
                "by z-standardization."
            ),

        "target_transform_for_training":
            (
                "Within-date percentile rank followed "
                "by z-standardization."
            ),

        "training_scheme":
            (
                "Expanding history; a historical block enters "
                "training only when analysis_date < forecast_date "
                "and label_end_date <= forecast_date."
            ),

        "minimum_training_periods":
            MIN_TRAIN_PERIODS,

        "minimum_cross_section_n":
            MIN_CROSS_SECTION_N,

        "date_balanced_weighting":
            DATE_BALANCED_WEIGHTING,

        "estimator":
            (
                "OLS using expanding sufficient statistics "
                "and Moore-Penrose pseudoinverse."
            ),

        "pinv_rcond":
            PINV_RCOND,

        "industry_vocabulary":
            industry_info,

        "performance_evaluation":
            (
                "Deferred to Step 4. Step 3 generates "
                "strict OOS predictions only."
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
    # 7.6 Console
    # ========================================================

    print()
    print("Forecast summary:")

    print(
        generated.groupby(
            "window"
        )
        .agg(
            forecast_date_n=(
                "forecast_date",
                "size",
            ),

            prediction_row_n=(
                "forecast_row_n",
                "sum",
            ),

            evaluable_target_n=(
                "forecast_target_available_n",
                "sum",
            ),

            first_forecast_date=(
                "forecast_date",
                "min",
            ),

            last_forecast_date=(
                "forecast_date",
                "max",
            ),
        )
        .to_string()
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
            "Day-11 Step-3 formal QA failed."
        )

    print()
    print("=" * 78)
    print("DAY 11 STEP 3 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()