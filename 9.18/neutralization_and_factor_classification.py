from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 0. Paths
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)

DAY5_ROOT = (
    OUTPUT_ROOT
    / "M1_day5"
)

DAY6_ROOT = (
    OUTPUT_ROOT
    / "M1_day6"
)


# ------------------------------------------------------------
# Day 5 factor panel
# ------------------------------------------------------------

FACTOR_PANEL_PATH = (
    DAY5_ROOT
    / "05_step5_community_features"
    / "community_bridge_feature_panel.parquet"
)


# ------------------------------------------------------------
# Day 1 PIT controls
#
# industry_id1
# market_value
# ------------------------------------------------------------

DAY1_RETURN_PANEL_PATH = (
    OUTPUT_ROOT
    / "M1_day1"
    / "03_stage3_return_validation"
    / "daily_return_panel.parquet"
)


# ------------------------------------------------------------
# Day 6 Step 1
# ------------------------------------------------------------

STEP1_DIR = (
    DAY6_ROOT
    / "01_step1_freeze_reduced_factor_set"
)

FROZEN_CORE_PATH = (
    STEP1_DIR
    / "frozen_core_factor_set.csv"
)

FACTOR_DESIGN_PATH = (
    STEP1_DIR
    / "day6_factor_design.json"
)


# ------------------------------------------------------------
# Day 6 Step 2
# ------------------------------------------------------------

STEP2_DIR = (
    DAY6_ROOT
    / "02_step2_forward_labels"
)

LABEL_PANEL_PATH = (
    STEP2_DIR
    / "forward_label_panel.parquet"
)

LABEL_DESIGN_PATH = (
    STEP2_DIR
    / "step2_label_design.json"
)

STEP2_METADATA_PATH = (
    STEP2_DIR
    / "step2_forward_label_metadata.json"
)


# ------------------------------------------------------------
# Step 3/4/5 lineage
# ------------------------------------------------------------

STEP3_METADATA_PATH = (
    DAY6_ROOT
    / "03_step3_alpha_rank_ic"
    / "step3_alpha_rank_ic_metadata.json"
)

STEP4_METADATA_PATH = (
    DAY6_ROOT
    / "04_step4_portfolio_sort"
    / "step4_portfolio_sort_metadata.json"
)

STEP5_METADATA_PATH = (
    DAY6_ROOT
    / "05_step5_risk_validation"
    / "step5_risk_validation_metadata.json"
)


# ------------------------------------------------------------
# Step 6 output
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY6_ROOT
    / "06_step6_neutralization_and_classification"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


NEUTRALIZED_PART_DIR = (
    OUTPUT_DIR
    / "neutralized_factor_parts"
)

NEUTRALIZED_PART_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


NEUTRALIZATION_QA_PATH = (
    OUTPUT_DIR
    / "neutralization_qa.csv"
)

MONTHLY_ALPHA_IC_PATH = (
    OUTPUT_DIR
    / "monthly_neutralized_alpha_rank_ic.csv"
)

ALPHA_SUMMARY_PATH = (
    OUTPUT_DIR
    / "neutralized_alpha_rank_ic_summary.csv"
)

MONTHLY_RISK_IC_PATH = (
    OUTPUT_DIR
    / "monthly_neutralized_risk_rank_ic.csv"
)

RISK_SUMMARY_PATH = (
    OUTPUT_DIR
    / "neutralized_risk_rank_ic_summary.csv"
)

MONTHLY_ALPHA_SPREAD_PATH = (
    OUTPUT_DIR
    / "monthly_neutralized_alpha_portfolio_spreads.csv"
)

ALPHA_PORTFOLIO_SUMMARY_PATH = (
    OUTPUT_DIR
    / "neutralized_alpha_portfolio_summary.csv"
)

RETENTION_PATH = (
    OUTPUT_DIR
    / "neutralization_signal_retention.csv"
)

CLASSIFICATION_PATH = (
    OUTPUT_DIR
    / "initial_factor_classification.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "step6_neutralization_metadata.json"
)


# ============================================================
# 1. Frozen Research Design
# ============================================================

EXPECTED_CORE_FACTOR_COUNT = 9

EXPECTED_WINDOWS = [
    60,
    120,
    252,
]


NEUTRALIZATION_METHODS = [

    "RAW",

    "INDUSTRY_NEUTRAL",

    "INDUSTRY_SIZE_NEUTRAL",
]


MIN_CROSS_SECTION_OBS = 100

MIN_SORT_OBS = 100

MIN_PORTFOLIO_OBS = 20

N_PORTFOLIOS = 5

USE_HAC = True

SAVE_NEUTRALIZED_FACTOR_PARTS = True


# ============================================================
# 2. Targets
# ============================================================

ALPHA_TARGET = (
    "future_excess_return"
)


RISK_TARGETS = [

    "future_realized_vol_annualized",

    "future_downside_vol_annualized",

    "future_max_drawdown",

    "future_idio_vol_annualized",
]


# ============================================================
# 3. Keys
# ============================================================

KEY_COLUMNS = [

    "analysis_date",

    "window",

    "master_index",

    "security_id",
]


# ============================================================
# 4. Utilities
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_json_atomic(
    obj,
    path,
):

    temp = Path(
        str(path)
        +
        ".tmp"
    )

    with open(
        temp,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )

    os.replace(
        temp,
        path,
    )


def save_csv_atomic(
    df,
    path,
):

    temp = Path(
        str(path)
        +
        ".tmp"
    )

    df.to_csv(
        temp,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        temp,
        path,
    )


def canonical_hash(obj):

    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# 5. Upstream QA
# ============================================================

def validate_upstream():

    factor_design = load_json(
        FACTOR_DESIGN_PATH
    )

    if (
        factor_design.get(
            "design_status"
        )
        !=
        "FROZEN"
    ):

        raise RuntimeError(
            "Day-6 factor design is not frozen."
        )

    factor_hash = (
        factor_design[
            "day6_design_hash"
        ]
    )

    label_design = load_json(
        LABEL_DESIGN_PATH
    )

    if (
        label_design.get(
            "label_version"
        )
        !=
        "holding_return_v2"
    ):

        raise RuntimeError(
            "Step 6 must use holding_return_v2."
        )

    if (
        label_design.get(
            "parent_day6_factor_design_hash"
        )
        !=
        factor_hash
    ):

        raise RuntimeError(
            "Factor / label design mismatch."
        )

    label_hash = (
        label_design[
            "step2_label_design_hash"
        ]
    )

    step2 = load_json(
        STEP2_METADATA_PATH
    )

    if (
        step2[
            "formal_qa"
        ][
            "all_jobs_pass"
        ]
        is not True
    ):

        raise RuntimeError(
            "Step 2 QA failed."
        )

    # --------------------------------------------------------
    # Require successful Step 3-5 lineage.
    # --------------------------------------------------------

    for name, path in [

        (
            "Step3",
            STEP3_METADATA_PATH,
        ),

        (
            "Step4",
            STEP4_METADATA_PATH,
        ),

        (
            "Step5",
            STEP5_METADATA_PATH,
        ),
    ]:

        metadata = load_json(
            path
        )

        formal_qa = (
            metadata.get(
                "formal_qa",
                {}
            )
        )

        if (
            formal_qa.get(
                "all_formal_qa_pass"
            )
            is not True
        ):

            raise RuntimeError(
                f"{name} formal QA failed."
            )

    return (
        factor_hash,
        label_hash,
    )


# ============================================================
# 6. Frozen Core Factors
# ============================================================

def load_core_factors():

    core = pd.read_csv(
        FROZEN_CORE_PATH
    )

    if (
        "factor_name"
        not in core.columns
    ):

        raise RuntimeError(
            "factor_name missing."
        )

    if (
        core[
            "factor_name"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate factor names."
        )

    if (
        len(core)
        !=
        EXPECTED_CORE_FACTOR_COUNT
    ):

        raise RuntimeError(
            "Core factor count mismatch."
        )

    if (
        "priority"
        not in core.columns
    ):

        core[
            "priority"
        ] = (
            np.arange(
                len(core)
            )
            +
            1
        )

    return (
        core.sort_values(
            "priority"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 7. Extract PIT Industry + Size Controls
#
# Read only four Day-1 columns in batches.
# ============================================================

def load_pit_controls(
    analysis_dates,
):

    required = [

        "trade_date",

        "security_id",

        "industry_id1",

        "market_value",
    ]

    parquet = pq.ParquetFile(
        DAY1_RETURN_PANEL_PATH
    )

    schema = set(
        parquet
        .schema_arrow
        .names
    )

    missing = [
        x
        for x in required
        if x not in schema
    ]

    if missing:

        raise RuntimeError(
            f"Day-1 control columns missing:\n"
            f"{missing}"
        )

    analysis_dates = pd.DatetimeIndex(
        pd.to_datetime(
            analysis_dates
        )
    )

    parts = []

    print(
        "Streaming PIT controls from Day-1 panel..."
    )

    for batch in parquet.iter_batches(

        columns=required,

        batch_size=500_000,
    ):

        df = batch.to_pandas()

        df[
            "trade_date"
        ] = pd.to_datetime(
            df[
                "trade_date"
            ]
        )

        df = df[
            df[
                "trade_date"
            ]
            .isin(
                analysis_dates
            )
        ]

        if len(df) > 0:

            parts.append(
                df
            )

    if not parts:

        raise RuntimeError(
            "No PIT control rows found."
        )

    controls = pd.concat(
        parts,
        ignore_index=True,
    )

    controls = controls.rename(
        columns={
            "trade_date":
                "analysis_date",
        }
    )

    controls[
        "security_id"
    ] = (
        controls[
            "security_id"
        ]
        .astype("string")
        .str.strip()
    )

    controls[
        "industry_id1"
    ] = (
        controls[
            "industry_id1"
        ]
        .astype("string")
        .str.strip()
    )

    controls[
        "market_value"
    ] = pd.to_numeric(
        controls[
            "market_value"
        ],
        errors="coerce",
    )

    if (
        controls[
            [
                "analysis_date",
                "security_id",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate PIT control rows."
        )

    controls[
        "log_market_value"
    ] = np.where(

        controls[
            "market_value"
        ]
        >
        0,

        np.log(
            controls[
                "market_value"
            ]
        ),

        np.nan,
    )

    return controls


# ============================================================
# 8. Load Analysis Panel
# ============================================================

def load_analysis_panel(
    core,
):

    factor_names = (
        core[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    factor_schema = set(
        pq.ParquetFile(
            FACTOR_PANEL_PATH
        )
        .schema_arrow
        .names
    )

    required_factor = (
        KEY_COLUMNS
        +
        factor_names
    )

    missing = [
        x
        for x in required_factor
        if x not in factor_schema
    ]

    if missing:

        raise RuntimeError(
            f"Factor columns missing:\n{missing}"
        )

    factors = pd.read_parquet(
        FACTOR_PANEL_PATH,
        columns=required_factor,
    )

    label_columns = (

        KEY_COLUMNS

        +

        [
            "future_return_label_valid",
            "future_risk_label_valid",
            ALPHA_TARGET,
        ]

        +

        RISK_TARGETS
    )

    labels = pd.read_parquet(
        LABEL_PANEL_PATH,
        columns=label_columns,
    )

    for df in [
        factors,
        labels,
    ]:

        df[
            "analysis_date"
        ] = pd.to_datetime(
            df[
                "analysis_date"
            ]
        )

        df[
            "window"
        ] = pd.to_numeric(
            df[
                "window"
            ]
        ).astype(int)

        df[
            "master_index"
        ] = pd.to_numeric(
            df[
                "master_index"
            ]
        ).astype(
            np.int64
        )

        df[
            "security_id"
        ] = (
            df[
                "security_id"
            ]
            .astype("string")
            .str.strip()
        )

    if (
        factors[
            KEY_COLUMNS
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate factor keys."
        )

    if (
        labels[
            KEY_COLUMNS
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate label keys."
        )

    panel = factors.merge(

        labels,

        on=KEY_COLUMNS,

        how="left",

        validate="one_to_one",

        indicator=True,
    )

    if (
        (
            panel[
                "_merge"
            ]
            !=
            "both"
        )
        .any()
    ):

        raise RuntimeError(
            "Missing label rows."
        )

    panel = panel.drop(
        columns="_merge"
    )

    analysis_dates = (
        panel[
            "analysis_date"
        ]
        .drop_duplicates()
    )

    controls = load_pit_controls(
        analysis_dates
    )

    panel = panel.merge(

        controls,

        on=[
            "analysis_date",
            "security_id",
        ],

        how="left",

        validate="many_to_one",
    )

    return panel


# ============================================================
# 9. Neutralization
# ============================================================

def neutralize_factor(
    group,
    factor_name,
):

    x = pd.to_numeric(
        group[
            factor_name
        ],
        errors="coerce",
    )

    industry = (
        group[
            "industry_id1"
        ]
        .astype("string")
    )

    log_size = pd.to_numeric(
        group[
            "log_market_value"
        ],
        errors="coerce",
    )

    raw = x.copy()

    industry_neutral = pd.Series(
        np.nan,
        index=group.index,
        dtype=float,
    )

    industry_size_neutral = pd.Series(
        np.nan,
        index=group.index,
        dtype=float,
    )

    # ========================================================
    # Industry neutral
    # ========================================================

    valid_ind = (

        x.notna()

        &

        np.isfinite(x)

        &

        industry.notna()
    )

    if valid_ind.any():

        tmp = pd.DataFrame(
            {
                "x":
                    x.loc[
                        valid_ind
                    ],

                "industry":
                    industry.loc[
                        valid_ind
                    ],
            }
        )

        industry_mean = (
            tmp.groupby(
                "industry"
            )[
                "x"
            ]
            .transform(
                "mean"
            )
        )

        industry_neutral.loc[
            valid_ind
        ] = (

            tmp[
                "x"
            ]

            -

            industry_mean
        )

    # ========================================================
    # Industry + Size
    #
    # FWL:
    #
    # demean factor within industry
    # demean log-size within industry
    # residualize factor on demeaned log-size
    # ========================================================

    valid_full = (

        valid_ind

        &

        log_size.notna()

        &

        np.isfinite(
            log_size
        )
    )

    beta_size = np.nan

    if valid_full.any():

        tmp = pd.DataFrame(
            {
                "x":
                    x.loc[
                        valid_full
                    ],

                "size":
                    log_size.loc[
                        valid_full
                    ],

                "industry":
                    industry.loc[
                        valid_full
                    ],
            }
        )

        x_bar = (
            tmp.groupby(
                "industry"
            )[
                "x"
            ]
            .transform(
                "mean"
            )
        )

        size_bar = (
            tmp.groupby(
                "industry"
            )[
                "size"
            ]
            .transform(
                "mean"
            )
        )

        x_within = (
            tmp[
                "x"
            ]
            -
            x_bar
        )

        size_within = (
            tmp[
                "size"
            ]
            -
            size_bar
        )

        denominator = float(
            np.dot(
                size_within,
                size_within,
            )
        )

        if (
            denominator
            >
            1e-14
        ):

            beta_size = float(

                np.dot(
                    size_within,
                    x_within,
                )

                /

                denominator
            )

        else:

            beta_size = 0.0

        residual = (

            x_within

            -

            beta_size
            *
            size_within
        )

        industry_size_neutral.loc[
            valid_full
        ] = residual

    # ========================================================
    # QA
    # ========================================================

    ind_max_abs_mean = np.nan

    if (
        industry_neutral
        .notna()
        .any()
    ):

        qa_df = pd.DataFrame(
            {
                "resid":
                    industry_neutral,

                "industry":
                    industry,
            }
        ).dropna()

        if len(qa_df) > 0:

            means = (
                qa_df.groupby(
                    "industry"
                )[
                    "resid"
                ]
                .mean()
            )

            ind_max_abs_mean = float(
                means.abs().max()
            )

    ind_size_max_abs_mean = np.nan

    size_resid_corr = np.nan

    if (
        industry_size_neutral
        .notna()
        .any()
    ):

        qa_df = pd.DataFrame(
            {
                "resid":
                    industry_size_neutral,

                "industry":
                    industry,

                "size":
                    log_size,
            }
        ).dropna()

        if len(qa_df) > 1:

            means = (
                qa_df.groupby(
                    "industry"
                )[
                    "resid"
                ]
                .mean()
            )

            ind_size_max_abs_mean = float(
                means.abs().max()
            )

            if (
                qa_df[
                    "resid"
                ]
                .std()
                >
                0

                and

                qa_df[
                    "size"
                ]
                .std()
                >
                0
            ):

                size_resid_corr = float(
                    qa_df[
                        "resid"
                    ]
                    .corr(
                        qa_df[
                            "size"
                        ]
                    )
                )

    diagnostics = {

        "factor_name":
            factor_name,

        "raw_nonmissing_n":
            int(
                raw.notna().sum()
            ),

        "industry_neutral_n":
            int(
                industry_neutral
                .notna()
                .sum()
            ),

        "industry_size_neutral_n":
            int(
                industry_size_neutral
                .notna()
                .sum()
            ),

        "industry_count":
            int(
                industry.loc[
                    valid_ind
                ]
                .nunique()
            ),

        "size_beta":
            beta_size,

        "industry_residual_max_abs_mean":
            ind_max_abs_mean,

        "industry_size_residual_max_abs_mean":
            ind_size_max_abs_mean,

        "industry_size_residual_logsize_corr":
            size_resid_corr,
    }

    return {

        "RAW":
            raw,

        "INDUSTRY_NEUTRAL":
            industry_neutral,

        "INDUSTRY_SIZE_NEUTRAL":
            industry_size_neutral,

    }, diagnostics


# ============================================================
# 10. Rank IC
# ============================================================

def rank_ic(
    factor,
    target,
    formal_valid,
):

    factor = pd.to_numeric(
        factor,
        errors="coerce",
    )

    target = pd.to_numeric(
        target,
        errors="coerce",
    )

    valid = (

        formal_valid

        &

        factor.notna()

        &

        target.notna()

        &

        np.isfinite(
            factor
        )

        &

        np.isfinite(
            target
        )
    )

    x = factor.loc[
        valid
    ]

    y = target.loc[
        valid
    ]

    n = len(x)

    if (
        n
        <
        MIN_CROSS_SECTION_OBS
    ):

        return (
            np.nan,
            n,
            "INSUFFICIENT_OBS",
        )

    if (
        x.nunique()
        <
        2
        or
        y.nunique()
        <
        2
    ):

        return (
            np.nan,
            n,
            "NO_VARIATION",
        )

    xr = (
        x.rank(
            method="average"
        )
        .to_numpy()
    )

    yr = (
        y.rank(
            method="average"
        )
        .to_numpy()
    )

    value = float(
        np.corrcoef(
            xr,
            yr,
        )[0, 1]
    )

    return (
        value,
        n,
        "PASS",
    )


# ============================================================
# 11. Quintile Alpha Spread
# ============================================================

def alpha_quintile_spread(
    score,
    future_return,
    formal_valid,
):

    score = pd.to_numeric(
        score,
        errors="coerce",
    )

    valid_score = (

        score.notna()

        &

        np.isfinite(
            score
        )
    )

    x = score.loc[
        valid_score
    ]

    if (
        len(x)
        <
        MIN_SORT_OBS
    ):

        return {
            "status":
                "INSUFFICIENT_SORT_OBS",
        }

    if (
        x.nunique()
        <
        N_PORTFOLIOS
    ):

        return {
            "status":
                "INSUFFICIENT_UNIQUE_VALUES",
        }

    try:

        q = pd.qcut(
            x,
            q=5,
            labels=False,
            duplicates="drop",
        )

    except ValueError:

        return {
            "status":
                "QCUT_FAILURE",
        }

    if (
        q.nunique()
        !=
        5
    ):

        return {
            "status":
                "FEWER_THAN_5_PORTFOLIOS",
        }

    portfolio = pd.Series(
        pd.NA,
        index=score.index,
        dtype="Int64",
    )

    portfolio.loc[
        x.index
    ] = (
        q.astype(int)
        +
        1
    )

    future_return = pd.to_numeric(
        future_return,
        errors="coerce",
    )

    valid_return = (

        formal_valid

        &

        future_return.notna()

        &

        np.isfinite(
            future_return
        )
    )

    q_ret = {}

    q_n = {}

    for k in range(
        1,
        6,
    ):

        mask = (

            valid_return

            &

            (
                portfolio
                ==
                k
            )
        )

        values = (
            future_return.loc[
                mask
            ]
        )

        q_n[
            k
        ] = len(
            values
        )

        if (
            len(values)
            >=
            MIN_PORTFOLIO_OBS
        ):

            q_ret[
                k
            ] = float(
                values.mean()
            )

        else:

            q_ret[
                k
            ] = np.nan

    values = np.array(
        [
            q_ret[k]
            for k in range(
                1,
                6,
            )
        ],
        dtype=float,
    )

    if (
        not np.isfinite(
            values
        )
        .all()
    ):

        return {
            "status":
                "INSUFFICIENT_PORTFOLIO_OBS",
        }

    spread = (
        q_ret[5]
        -
        q_ret[1]
    )

    monotonicity = float(
        pd.Series(
            values
        ).corr(
            pd.Series(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                ]
            ),
            method="spearman",
        )
    )

    return {

        "status":
            "PASS",

        "q1":
            q_ret[1],

        "q2":
            q_ret[2],

        "q3":
            q_ret[3],

        "q4":
            q_ret[4],

        "q5":
            q_ret[5],

        "spread":
            spread,

        "q1_n":
            q_n[1],

        "q5_n":
            q_n[5],

        "portfolio_spearman":
            monotonicity,
    }


# ============================================================
# 12. HAC
# ============================================================

def automatic_hac_lag(
    n,
):

    if n <= 1:

        return 0

    value = int(
        np.floor(
            4
            *
            (
                n
                /
                100
            )
            **
            (
                2
                /
                9
            )
        )
    )

    return max(
        0,
        min(
            value,
            n - 1,
        ),
    )


def hac_inference(
    values,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    n = len(x)

    if n < 2:

        return (
            np.nan,
            np.nan,
            0,
        )

    mean_x = x.mean()

    lag = (
        automatic_hac_lag(
            n
        )
        if USE_HAC
        else
        0
    )

    residual = (
        x
        -
        mean_x
    )

    gamma0 = (

        np.dot(
            residual,
            residual,
        )

        /
        n
    )

    long_run_var = gamma0

    for ell in range(
        1,
        lag + 1,
    ):

        gamma = (

            np.dot(
                residual[
                    ell:
                ],
                residual[
                    :-ell
                ],
            )

            /
            n
        )

        weight = (
            1
            -
            ell
            /
            (
                lag
                +
                1
            )
        )

        long_run_var += (
            2
            *
            weight
            *
            gamma
        )

    long_run_var = max(
        long_run_var,
        0
    )

    se = np.sqrt(
        long_run_var
        /
        n
    )

    t_stat = (
        mean_x
        /
        se
        if
        se > 0
        else
        np.nan
    )

    return (
        float(se),
        float(t_stat),
        lag,
    )


# ============================================================
# 13. Main Monthly Loop
# ============================================================

def run_monthly_analysis(
    panel,
    core,
):

    alpha_rows = []

    risk_rows = []

    portfolio_rows = []

    qa_rows = []

    factor_names = (
        core[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    grouped = panel.groupby(
        [
            "analysis_date",
            "window",
        ],
        sort=True,
    )

    total_jobs = (
        grouped.ngroups
    )

    for job_no, (
        (
            analysis_date,
            window,
        ),
        group,
    ) in enumerate(
        grouped,
        start=1,
    ):

        group = (
            group.copy()
            .reset_index(
                drop=True
            )
        )

        print(
            f"[{job_no:3d}/{total_jobs}] "
            f"W={window} | "
            f"{analysis_date.date()}"
        )

        return_valid = (
            group[
                "future_return_label_valid"
            ]
            .fillna(False)
            .astype(bool)
        )

        risk_valid = (
            group[
                "future_risk_label_valid"
            ]
            .fillna(False)
            .astype(bool)
        )

        # ----------------------------------------------------
        # Optional neutralized-factor partition
        # ----------------------------------------------------

        part = group[
            KEY_COLUMNS
            +
            [
                "industry_id1",
                "market_value",
                "log_market_value",
            ]
        ].copy()

        for (
            factor_order,
            factor_name,
        ) in enumerate(
            factor_names,
            start=1,
        ):

            (
                scores,
                neutral_qa,
            ) = neutralize_factor(

                group,
                factor_name,
            )

            qa_rows.append(
                {

                    "analysis_date":
                        analysis_date,

                    "window":
                        int(window),

                    "factor_order":
                        factor_order,

                    **neutral_qa,
                }
            )

            if SAVE_NEUTRALIZED_FACTOR_PARTS:

                for method in (
                    NEUTRALIZATION_METHODS
                ):

                    part[
                        f"{factor_name}"
                        f"__{method}"
                    ] = (
                        scores[
                            method
                        ]
                    )

            # =================================================
            # Three versions
            # =================================================

            for method in (
                NEUTRALIZATION_METHODS
            ):

                score = (
                    scores[
                        method
                    ]
                )

                # ---------------------------------------------
                # Alpha IC
                # ---------------------------------------------

                (
                    alpha_ic,
                    alpha_n,
                    alpha_status,
                ) = rank_ic(

                    score,

                    group[
                        ALPHA_TARGET
                    ],

                    return_valid,
                )

                alpha_rows.append(
                    {

                        "analysis_date":
                            analysis_date,

                        "window":
                            int(window),

                        "factor_order":
                            factor_order,

                        "factor_name":
                            factor_name,

                        "neutralization":
                            method,

                        "target_name":
                            ALPHA_TARGET,

                        "pair_n":
                            alpha_n,

                        "alpha_rank_ic":
                            alpha_ic,

                        "ic_status":
                            alpha_status,
                    }
                )

                # ---------------------------------------------
                # Alpha portfolio spread
                # ---------------------------------------------

                portfolio_result = (
                    alpha_quintile_spread(

                        score,

                        group[
                            ALPHA_TARGET
                        ],

                        return_valid,
                    )
                )

                portfolio_rows.append(
                    {

                        "analysis_date":
                            analysis_date,

                        "window":
                            int(window),

                        "factor_order":
                            factor_order,

                        "factor_name":
                            factor_name,

                        "neutralization":
                            method,

                        "target_name":
                            ALPHA_TARGET,

                        "spread_status":
                            portfolio_result.get(
                                "status"
                            ),

                        "q1_return":
                            portfolio_result.get(
                                "q1",
                                np.nan,
                            ),

                        "q2_return":
                            portfolio_result.get(
                                "q2",
                                np.nan,
                            ),

                        "q3_return":
                            portfolio_result.get(
                                "q3",
                                np.nan,
                            ),

                        "q4_return":
                            portfolio_result.get(
                                "q4",
                                np.nan,
                            ),

                        "q5_return":
                            portfolio_result.get(
                                "q5",
                                np.nan,
                            ),

                        "q5_minus_q1":
                            portfolio_result.get(
                                "spread",
                                np.nan,
                            ),

                        "q5_minus_q1_bps":
                            (
                                portfolio_result.get(
                                    "spread",
                                    np.nan,
                                )
                                *
                                10000
                            ),

                        "portfolio_spearman":
                            portfolio_result.get(
                                "portfolio_spearman",
                                np.nan,
                            ),
                    }
                )

                # ---------------------------------------------
                # Risk IC
                # ---------------------------------------------

                for target in (
                    RISK_TARGETS
                ):

                    (
                        risk_ic,
                        risk_n,
                        risk_status,
                    ) = rank_ic(

                        score,

                        group[
                            target
                        ],

                        risk_valid,
                    )

                    risk_rows.append(
                        {

                            "analysis_date":
                                analysis_date,

                            "window":
                                int(window),

                            "factor_order":
                                factor_order,

                            "factor_name":
                                factor_name,

                            "neutralization":
                                method,

                            "target_name":
                                target,

                            "pair_n":
                                risk_n,

                            "risk_rank_ic":
                                risk_ic,

                            "ic_status":
                                risk_status,
                        }
                    )

        # ----------------------------------------------------
        # Save neutralized scores
        # ----------------------------------------------------

        if SAVE_NEUTRALIZED_FACTOR_PARTS:

            path = (
                NEUTRALIZED_PART_DIR
                /
                f"W{int(window)}"
                /
                (
                    f"{analysis_date:%Y-%m-%d}"
                    ".parquet"
                )
            )

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            part.to_parquet(
                path,
                index=False,
                compression="zstd",
            )

    return (

        pd.DataFrame(
            alpha_rows
        ),

        pd.DataFrame(
            risk_rows
        ),

        pd.DataFrame(
            portfolio_rows
        ),

        pd.DataFrame(
            qa_rows
        ),
    )


# ============================================================
# 14. Alpha IC Summary
# ============================================================

def summarize_alpha(
    monthly,
):

    rows = []

    passed = monthly[
        monthly[
            "ic_status"
        ]
        ==
        "PASS"
    ]

    for keys, group in passed.groupby(
        [
            "factor_order",
            "factor_name",
            "neutralization",
            "window",
        ],
        sort=False,
    ):

        (
            factor_order,
            factor_name,
            neutralization,
            window,
        ) = keys

        x = (
            group.sort_values(
                "analysis_date"
            )[
                "alpha_rank_ic"
            ]
            .to_numpy(
                dtype=float
            )
        )

        se, t_stat, lag = (
            hac_inference(
                x
            )
        )

        rows.append(
            {

                "factor_order":
                    factor_order,

                "factor_name":
                    factor_name,

                "neutralization":
                    neutralization,

                "window":
                    int(window),

                "month_count":
                    len(x),

                "mean_alpha_ic":
                    float(
                        np.mean(x)
                    ),

                "median_alpha_ic":
                    float(
                        np.median(x)
                    ),

                "std_alpha_ic":
                    float(
                        np.std(
                            x,
                            ddof=1,
                        )
                    ),

                "positive_ic_share":
                    float(
                        np.mean(
                            x > 0
                        )
                    ),

                "negative_ic_share":
                    float(
                        np.mean(
                            x < 0
                        )
                    ),

                "hac_lag":
                    lag,

                "hac_alpha_ic_se":
                    se,

                "hac_alpha_ic_t_stat":
                    t_stat,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "factor_order",
                "neutralization",
                "window",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 15. Risk IC Summary
# ============================================================

def summarize_risk(
    monthly,
):

    rows = []

    passed = monthly[
        monthly[
            "ic_status"
        ]
        ==
        "PASS"
    ]

    for keys, group in passed.groupby(
        [
            "factor_order",
            "factor_name",
            "neutralization",
            "target_name",
            "window",
        ],
        sort=False,
    ):

        (
            factor_order,
            factor_name,
            neutralization,
            target_name,
            window,
        ) = keys

        x = (
            group.sort_values(
                "analysis_date"
            )[
                "risk_rank_ic"
            ]
            .to_numpy(
                dtype=float
            )
        )

        se, t_stat, lag = (
            hac_inference(
                x
            )
        )

        rows.append(
            {

                "factor_order":
                    factor_order,

                "factor_name":
                    factor_name,

                "neutralization":
                    neutralization,

                "target_name":
                    target_name,

                "window":
                    int(window),

                "month_count":
                    len(x),

                "mean_risk_ic":
                    float(
                        np.mean(x)
                    ),

                "median_risk_ic":
                    float(
                        np.median(x)
                    ),

                "positive_ic_share":
                    float(
                        np.mean(
                            x > 0
                        )
                    ),

                "negative_ic_share":
                    float(
                        np.mean(
                            x < 0
                        )
                    ),

                "hac_lag":
                    lag,

                "hac_risk_ic_se":
                    se,

                "hac_risk_ic_t_stat":
                    t_stat,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "factor_order",
                "neutralization",
                "target_name",
                "window",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 16. Alpha Portfolio Summary
# ============================================================

def summarize_portfolio(
    monthly,
):

    passed = monthly[
        monthly[
            "spread_status"
        ]
        ==
        "PASS"
    ]

    rows = []

    for keys, group in passed.groupby(
        [
            "factor_order",
            "factor_name",
            "neutralization",
            "window",
        ],
        sort=False,
    ):

        (
            factor_order,
            factor_name,
            neutralization,
            window,
        ) = keys

        group = group.sort_values(
            "analysis_date"
        )

        spread = (
            group[
                "q5_minus_q1"
            ]
            .to_numpy(
                dtype=float
            )
        )

        se, t_stat, lag = (
            hac_inference(
                spread
            )
        )

        rows.append(
            {

                "factor_order":
                    factor_order,

                "factor_name":
                    factor_name,

                "neutralization":
                    neutralization,

                "window":
                    int(window),

                "month_count":
                    len(spread),

                "mean_q1_return":
                    float(
                        group[
                            "q1_return"
                        ]
                        .mean()
                    ),

                "mean_q5_return":
                    float(
                        group[
                            "q5_return"
                        ]
                        .mean()
                    ),

                "mean_q5_minus_q1":
                    float(
                        np.mean(
                            spread
                        )
                    ),

                "mean_q5_minus_q1_bps":
                    float(
                        np.mean(
                            spread
                        )
                        *
                        10000
                    ),

                "positive_spread_share":
                    float(
                        np.mean(
                            spread > 0
                        )
                    ),

                "negative_spread_share":
                    float(
                        np.mean(
                            spread < 0
                        )
                    ),

                "mean_portfolio_spearman":
                    float(
                        group[
                            "portfolio_spearman"
                        ]
                        .mean()
                    ),

                "hac_lag":
                    lag,

                "hac_spread_se":
                    se,

                "hac_spread_t_stat":
                    t_stat,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "factor_order",
                "neutralization",
                "window",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 17. Direction Helper
# ============================================================

def same_nonzero_sign(
    values,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    if (
        not np.isfinite(x).all()
    ):

        return False

    signs = np.sign(x)

    if (
        np.any(
            signs == 0
        )
    ):

        return False

    return bool(
        np.all(
            signs
            ==
            signs[0]
        )
    )


# ============================================================
# 18. Signal Retention
# ============================================================

def build_retention_table(
    alpha_summary,
    risk_summary,
    portfolio_summary,
):

    rows = []

    factors = (
        alpha_summary[
            [
                "factor_order",
                "factor_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "factor_order"
        )
    )

    for row in factors.itertuples(
        index=False
    ):

        factor_order = (
            row.factor_order
        )

        factor_name = (
            row.factor_name
        )

        for window in (
            EXPECTED_WINDOWS
        ):

            a = alpha_summary[
                (
                    alpha_summary[
                        "factor_name"
                    ]
                    ==
                    factor_name
                )
                &
                (
                    alpha_summary[
                        "window"
                    ]
                    ==
                    window
                )
            ]

            p = portfolio_summary[
                (
                    portfolio_summary[
                        "factor_name"
                    ]
                    ==
                    factor_name
                )
                &
                (
                    portfolio_summary[
                        "window"
                    ]
                    ==
                    window
                )
            ]

            r = risk_summary[
                (
                    risk_summary[
                        "factor_name"
                    ]
                    ==
                    factor_name
                )
                &
                (
                    risk_summary[
                        "window"
                    ]
                    ==
                    window
                )
            ]

            def get_alpha(method):

                x = a[
                    a[
                        "neutralization"
                    ]
                    ==
                    method
                ]

                if len(x) != 1:

                    return np.nan

                return float(
                    x[
                        "mean_alpha_ic"
                    ]
                    .iloc[0]
                )

            def get_spread(method):

                x = p[
                    p[
                        "neutralization"
                    ]
                    ==
                    method
                ]

                if len(x) != 1:

                    return np.nan

                return float(
                    x[
                        "mean_q5_minus_q1"
                    ]
                    .iloc[0]
                )

            def get_risk_average(method):

                x = r[
                    r[
                        "neutralization"
                    ]
                    ==
                    method
                ]

                if len(x) == 0:

                    return np.nan

                return float(
                    x[
                        "mean_risk_ic"
                    ]
                    .mean()
                )

            raw_alpha = get_alpha(
                "RAW"
            )

            ind_alpha = get_alpha(
                "INDUSTRY_NEUTRAL"
            )

            final_alpha = get_alpha(
                "INDUSTRY_SIZE_NEUTRAL"
            )

            raw_risk = get_risk_average(
                "RAW"
            )

            ind_risk = get_risk_average(
                "INDUSTRY_NEUTRAL"
            )

            final_risk = get_risk_average(
                "INDUSTRY_SIZE_NEUTRAL"
            )

            raw_spread = get_spread(
                "RAW"
            )

            ind_spread = get_spread(
                "INDUSTRY_NEUTRAL"
            )

            final_spread = get_spread(
                "INDUSTRY_SIZE_NEUTRAL"
            )

            rows.append(
                {

                    "factor_order":
                        factor_order,

                    "factor_name":
                        factor_name,

                    "window":
                        window,

                    "raw_alpha_ic":
                        raw_alpha,

                    "industry_neutral_alpha_ic":
                        ind_alpha,

                    "industry_size_neutral_alpha_ic":
                        final_alpha,

                    "raw_alpha_to_final_same_sign":
                        (
                            np.sign(
                                raw_alpha
                            )
                            ==
                            np.sign(
                                final_alpha
                            )
                            if
                            np.isfinite(
                                raw_alpha
                            )
                            and
                            np.isfinite(
                                final_alpha
                            )
                            else
                            False
                        ),

                    "alpha_abs_retention_ratio":
                        (
                            abs(
                                final_alpha
                            )
                            /
                            abs(
                                raw_alpha
                            )
                            if
                            np.isfinite(
                                raw_alpha
                            )
                            and
                            abs(
                                raw_alpha
                            )
                            >=
                            0.005
                            and
                            np.isfinite(
                                final_alpha
                            )
                            else
                            np.nan
                        ),

                    "raw_risk_ic_average":
                        raw_risk,

                    "industry_neutral_risk_ic_average":
                        ind_risk,

                    "industry_size_neutral_risk_ic_average":
                        final_risk,

                    "raw_risk_to_final_same_sign":
                        (
                            np.sign(
                                raw_risk
                            )
                            ==
                            np.sign(
                                final_risk
                            )
                            if
                            np.isfinite(
                                raw_risk
                            )
                            and
                            np.isfinite(
                                final_risk
                            )
                            else
                            False
                        ),

                    "raw_alpha_spread":
                        raw_spread,

                    "industry_neutral_alpha_spread":
                        ind_spread,

                    "industry_size_neutral_alpha_spread":
                        final_spread,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 19. Initial Factor Classification
#
# IMPORTANT:
#
# This is descriptive classification.
#
# It does NOT:
# - remove factors,
# - flip factor signs,
# - select windows,
# - redefine the frozen factor set.
# ============================================================

def build_initial_classification(
    alpha_summary,
    risk_summary,
    portfolio_summary,
):

    final_method = (
        "INDUSTRY_SIZE_NEUTRAL"
    )

    alpha = alpha_summary[
        alpha_summary[
            "neutralization"
        ]
        ==
        final_method
    ]

    risk = risk_summary[
        risk_summary[
            "neutralization"
        ]
        ==
        final_method
    ]

    portfolio = portfolio_summary[
        portfolio_summary[
            "neutralization"
        ]
        ==
        final_method
    ]

    factors = (
        alpha[
            [
                "factor_order",
                "factor_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "factor_order"
        )
    )

    rows = []

    for row in factors.itertuples(
        index=False
    ):

        factor_order = (
            row.factor_order
        )

        factor_name = (
            row.factor_name
        )

        a = (
            alpha[
                alpha[
                    "factor_name"
                ]
                ==
                factor_name
            ]
            .sort_values(
                "window"
            )
        )

        p = (
            portfolio[
                portfolio[
                    "factor_name"
                ]
                ==
                factor_name
            ]
            .sort_values(
                "window"
            )
        )

        r = risk[
            risk[
                "factor_name"
            ]
            ==
            factor_name
        ]

        alpha_values = (
            a[
                "mean_alpha_ic"
            ]
            .to_numpy(
                dtype=float
            )
        )

        spread_values = (
            p[
                "mean_q5_minus_q1"
            ]
            .to_numpy(
                dtype=float
            )
        )

        risk_values = (
            r[
                "mean_risk_ic"
            ]
            .to_numpy(
                dtype=float
            )
        )

        alpha_same_sign = (

            len(alpha_values)
            ==
            3

            and

            same_nonzero_sign(
                alpha_values
            )
        )

        spread_same_sign = (

            len(spread_values)
            ==
            3

            and

            same_nonzero_sign(
                spread_values
            )
        )

        alpha_mean = (
            float(
                np.mean(
                    alpha_values
                )
            )
            if
            len(alpha_values)
            >
            0
            else
            np.nan
        )

        spread_mean = (
            float(
                np.mean(
                    spread_values
                )
            )
            if
            len(spread_values)
            >
            0
            else
            np.nan
        )

        alpha_economic_direction_confirmed = (

            alpha_same_sign

            and

            spread_same_sign

            and

            np.sign(
                alpha_mean
            )
            ==
            np.sign(
                spread_mean
            )
        )

        risk_same_sign_all_cells = (

            len(risk_values)
            ==
            12

            and

            same_nonzero_sign(
                risk_values
            )
        )

        risk_mean = (
            float(
                np.mean(
                    risk_values
                )
            )
            if
            len(risk_values)
            >
            0
            else
            np.nan
        )

        alpha_hac_count = int(
            (
                a[
                    "hac_alpha_ic_t_stat"
                ]
                .abs()
                >=
                1.96
            )
            .sum()
        )

        spread_hac_count = int(
            (
                p[
                    "hac_spread_t_stat"
                ]
                .abs()
                >=
                1.96
            )
            .sum()
        )

        risk_hac_count = int(
            (
                r[
                    "hac_risk_ic_t_stat"
                ]
                .abs()
                >=
                1.96
            )
            .sum()
        )

        # ----------------------------------------------------
        # Descriptive directional classification
        # ----------------------------------------------------

        if (
            alpha_economic_direction_confirmed
            and
            risk_same_sign_all_cells
        ):

            if (
                alpha_mean < 0
                and
                risk_mean > 0
            ):

                classification = (
                    "ADVERSE_RETURN_RISK"
                )

            elif (
                alpha_mean > 0
                and
                risk_mean < 0
            ):

                classification = (
                    "FAVORABLE_RETURN_RISK"
                )

            elif (
                alpha_mean > 0
                and
                risk_mean > 0
            ):

                classification = (
                    "POSITIVE_RETURN_HIGHER_RISK"
                )

            elif (
                alpha_mean < 0
                and
                risk_mean < 0
            ):

                classification = (
                    "LOWER_RETURN_LOWER_RISK"
                )

            else:

                classification = (
                    "MIXED_OR_NEAR_ZERO"
                )

        elif (
            risk_same_sign_all_cells
            and
            not alpha_economic_direction_confirmed
        ):

            classification = (
                "RISK_DOMINANT_ALPHA_UNSTABLE"
            )

        elif (
            alpha_economic_direction_confirmed
            and
            not risk_same_sign_all_cells
        ):

            classification = (
                "ALPHA_DIRECTION_STABLE_RISK_MIXED"
            )

        else:

            classification = (
                "MIXED_OR_UNSTABLE"
            )

        def risk_target_average(
            target
        ):

            x = r[
                r[
                    "target_name"
                ]
                ==
                target
            ]

            if len(x) == 0:

                return np.nan

            return float(
                x[
                    "mean_risk_ic"
                ]
                .mean()
            )

        rows.append(
            {

                "factor_order":
                    factor_order,

                "factor_name":
                    factor_name,

                "classification_basis":
                    final_method,

                "mean_alpha_ic_across_windows":
                    alpha_mean,

                "alpha_ic_same_sign_all_windows":
                    alpha_same_sign,

                "mean_alpha_spread_across_windows":
                    spread_mean,

                "alpha_spread_same_sign_all_windows":
                    spread_same_sign,

                "alpha_economic_direction_confirmed":
                    alpha_economic_direction_confirmed,

                "alpha_ic_hac_abs_ge_1p96_count":
                    alpha_hac_count,

                "alpha_spread_hac_abs_ge_1p96_count":
                    spread_hac_count,

                "mean_risk_ic_all_12_cells":
                    risk_mean,

                "risk_same_sign_all_12_cells":
                    risk_same_sign_all_cells,

                "risk_hac_abs_ge_1p96_count":
                    risk_hac_count,

                "mean_total_vol_ic":
                    risk_target_average(
                        "future_realized_vol_annualized"
                    ),

                "mean_downside_vol_ic":
                    risk_target_average(
                        "future_downside_vol_annualized"
                    ),

                "mean_mdd_ic":
                    risk_target_average(
                        "future_max_drawdown"
                    ),

                "mean_ivol_ic":
                    risk_target_average(
                        "future_idio_vol_annualized"
                    ),

                "initial_directional_classification":
                    classification,

                "factor_sign_flipped":
                    False,

                "window_selected":
                    False,

                "factor_removed":
                    False,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 20. Step-6 Formal QA
# ============================================================

def formal_qa(
    alpha_monthly,
    risk_monthly,
    qa_df,
    classification,
):

    checks = {}

    checks[
        "neutralization_methods"
    ] = sorted(
        alpha_monthly[
            "neutralization"
        ]
        .unique()
        .tolist()
    )

    checks[
        "methods_ok"
    ] = (
        checks[
            "neutralization_methods"
        ]
        ==
        sorted(
            NEUTRALIZATION_METHODS
        )
    )

    checks[
        "alpha_key_unique"
    ] = bool(
        not alpha_monthly[
            [
                "analysis_date",
                "window",
                "factor_name",
                "neutralization",
            ]
        ]
        .duplicated()
        .any()
    )

    checks[
        "risk_key_unique"
    ] = bool(
        not risk_monthly[
            [
                "analysis_date",
                "window",
                "factor_name",
                "neutralization",
                "target_name",
            ]
        ]
        .duplicated()
        .any()
    )

    checks[
        "max_abs_industry_residual_mean"
    ] = float(
        qa_df[
            "industry_residual_max_abs_mean"
        ]
        .abs()
        .max()
    )

    checks[
        "max_abs_industry_size_residual_mean"
    ] = float(
        qa_df[
            "industry_size_residual_max_abs_mean"
        ]
        .abs()
        .max()
    )

    checks[
        "max_abs_industry_size_logsize_corr"
    ] = float(
        qa_df[
            "industry_size_residual_logsize_corr"
        ]
        .abs()
        .max()
    )

    checks[
        "industry_neutral_orthogonality_ok"
    ] = bool(
        checks[
            "max_abs_industry_residual_mean"
        ]
        <
        1e-8
    )

    checks[
        "industry_size_neutral_orthogonality_ok"
    ] = bool(

        checks[
            "max_abs_industry_size_residual_mean"
        ]
        <
        1e-8

        and

        checks[
            "max_abs_industry_size_logsize_corr"
        ]
        <
        1e-8
    )

    checks[
        "classification_factor_count"
    ] = len(
        classification
    )

    checks[
        "classification_factor_count_ok"
    ] = (
        len(
            classification
        )
        ==
        EXPECTED_CORE_FACTOR_COUNT
    )

    required = [

        "methods_ok",

        "alpha_key_unique",

        "risk_key_unique",

        "industry_neutral_orthogonality_ok",

        "industry_size_neutral_orthogonality_ok",

        "classification_factor_count_ok",
    ]

    checks[
        "all_formal_qa_pass"
    ] = all(
        checks[
            x
        ]
        for x in required
    )

    return checks


# ============================================================
# 21. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 6 - Step 6"
    )

    print(
        "Neutralization & Initial Factor Classification"
    )

    print("=" * 80)

    # ========================================================
    # Upstream
    # ========================================================

    (
        factor_hash,
        label_hash,
    ) = validate_upstream()

    core = load_core_factors()

    # ========================================================
    # Freeze Step-6 design
    # ========================================================

    design_payload = {

        "research_day":
            6,

        "step":
            (
                "Step6_Neutralization_"
                "and_Initial_Factor_Classification"
            ),

        "parent_factor_design_hash":
            factor_hash,

        "parent_label_design_hash":
            label_hash,

        "factors":
            core[
                "factor_name"
            ]
            .tolist(),

        "windows":
            EXPECTED_WINDOWS,

        "neutralization_methods":
            NEUTRALIZATION_METHODS,

        "industry_control":
            "industry_id1",

        "size_control":
            "log(market_value)",

        "industry_neutralization":
            (
                "Within-industry cross-sectional "
                "demeaning at each analysis_date x window."
            ),

        "industry_size_neutralization":
            (
                "Equal-weight cross-sectional OLS: "
                "factor ~ industry fixed effects "
                "+ log market value; FWL residual used."
            ),

        "primary_alpha_target":
            ALPHA_TARGET,

        "risk_targets":
            RISK_TARGETS,

        "factor_sign_flip":
            False,

        "outcome_based_window_selection":
            False,

        "factor_removal":
            False,

        "classification_is_descriptive":
            True,

        "multiple_testing_adjustment":
            False,
    }

    design = {

        **design_payload,

        "step6_design_hash":
            canonical_hash(
                design_payload
            ),
    }

    print(
        "Step-6 design hash:"
    )

    print(
        design[
            "step6_design_hash"
        ]
    )

    # ========================================================
    # Panel
    # ========================================================

    print()
    print(
        "[1] Load factor, outcome, industry and size panel"
    )

    panel = load_analysis_panel(
        core
    )

    print(
        f"Analysis rows: "
        f"{len(panel):,}"
    )

    print(
        f"Industry coverage: "
        f"{panel['industry_id1'].notna().mean():.4%}"
    )

    print(
        f"Positive market-value coverage: "
        f"{panel['log_market_value'].notna().mean():.4%}"
    )

    # ========================================================
    # Monthly analysis
    # ========================================================

    print()
    print(
        "[2] Neutralize factors and recompute Alpha / Risk signals"
    )

    (
        alpha_monthly,
        risk_monthly,
        portfolio_monthly,
        qa_df,
    ) = run_monthly_analysis(
        panel,
        core,
    )

    save_csv_atomic(
        alpha_monthly,
        MONTHLY_ALPHA_IC_PATH,
    )

    save_csv_atomic(
        risk_monthly,
        MONTHLY_RISK_IC_PATH,
    )

    save_csv_atomic(
        portfolio_monthly,
        MONTHLY_ALPHA_SPREAD_PATH,
    )

    save_csv_atomic(
        qa_df,
        NEUTRALIZATION_QA_PATH,
    )

    # ========================================================
    # Summaries
    # ========================================================

    print()
    print(
        "[3] Build neutralized Alpha summary"
    )

    alpha_summary = summarize_alpha(
        alpha_monthly
    )

    save_csv_atomic(
        alpha_summary,
        ALPHA_SUMMARY_PATH,
    )

    print()
    print(
        "[4] Build neutralized Risk summary"
    )

    risk_summary = summarize_risk(
        risk_monthly
    )

    save_csv_atomic(
        risk_summary,
        RISK_SUMMARY_PATH,
    )

    print()
    print(
        "[5] Build neutralized Alpha portfolio summary"
    )

    portfolio_summary = summarize_portfolio(
        portfolio_monthly
    )

    save_csv_atomic(
        portfolio_summary,
        ALPHA_PORTFOLIO_SUMMARY_PATH,
    )

    # ========================================================
    # Retention
    # ========================================================

    print()
    print(
        "[6] Measure signal retention after neutralization"
    )

    retention = build_retention_table(

        alpha_summary,

        risk_summary,

        portfolio_summary,
    )

    save_csv_atomic(
        retention,
        RETENTION_PATH,
    )

    # ========================================================
    # Classification
    # ========================================================

    print()
    print(
        "[7] Initial descriptive factor classification"
    )

    classification = (
        build_initial_classification(

            alpha_summary,

            risk_summary,

            portfolio_summary,
        )
    )

    save_csv_atomic(
        classification,
        CLASSIFICATION_PATH,
    )

    # ========================================================
    # QA
    # ========================================================

    qa = formal_qa(

        alpha_monthly,

        risk_monthly,

        qa_df,

        classification,
    )

    if (
        not qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            f"Step-6 QA failed:\n{qa}"
        )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {

        **design,

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "formal_qa":
            qa,

        "important_interpretation_rules": [

            (
                "Neutralization uses only "
                "contemporaneous PIT controls."
            ),

            (
                "Future outcomes never enter "
                "neutralization."
            ),

            (
                "Factor signs remain exactly "
                "as frozen before outcome validation."
            ),

            (
                "No network window is selected "
                "using realized outcomes."
            ),

            (
                "Initial classification is descriptive "
                "and does not alter the frozen factor set."
            ),

            (
                "HAC statistics are nominal exploratory "
                "inference; no multiple-testing correction "
                "is applied at this stage."
            ),
        ],

        "outputs": {

            "neutralization_qa":
                str(
                    NEUTRALIZATION_QA_PATH
                ),

            "alpha_summary":
                str(
                    ALPHA_SUMMARY_PATH
                ),

            "risk_summary":
                str(
                    RISK_SUMMARY_PATH
                ),

            "alpha_portfolio_summary":
                str(
                    ALPHA_PORTFOLIO_SUMMARY_PATH
                ),

            "signal_retention":
                str(
                    RETENTION_PATH
                ),

            "initial_classification":
                str(
                    CLASSIFICATION_PATH
                ),
        },
    }

    save_json_atomic(
        metadata,
        METADATA_PATH,
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Step-6 Formal QA"
    )

    print("=" * 80)

    print(
        f"Max |industry residual mean|: "
        f"{qa['max_abs_industry_residual_mean']:.3e}"
    )

    print(
        f"Max |industry+size residual mean|: "
        f"{qa['max_abs_industry_size_residual_mean']:.3e}"
    )

    print(
        f"Max |residual-logSize corr|: "
        f"{qa['max_abs_industry_size_logsize_corr']:.3e}"
    )

    print(
        f"Formal QA pass: "
        f"{qa['all_formal_qa_pass']}"
    )

    print()
    print("=" * 80)

    print(
        "Initial Factor Classification"
    )

    print("=" * 80)

    display_columns = [

        "factor_name",

        "mean_alpha_ic_across_windows",

        "mean_alpha_spread_across_windows",

        "mean_risk_ic_all_12_cells",

        "mean_ivol_ic",

        "alpha_economic_direction_confirmed",

        "risk_same_sign_all_12_cells",

        "initial_directional_classification",
    ]

    print(
        classification[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print("=" * 80)

    print(
        "Day 6 Step 6 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()