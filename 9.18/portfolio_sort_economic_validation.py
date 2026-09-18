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


# ------------------------------------------------------------
# Day 5 final factor panel
# Research Day 5 is stored under M1_day4
# ------------------------------------------------------------

DAY5_ROOT = (
    OUTPUT_ROOT
    / "M1_day5"
)

FACTOR_PANEL_PATH = (
    DAY5_ROOT
    / "05_step5_community_features"
    / "community_bridge_feature_panel.parquet"
)


# ------------------------------------------------------------
# Day 6
# Research Day 6 is stored under M1_day5
# ------------------------------------------------------------

DAY6_ROOT = (
    OUTPUT_ROOT
    / "M1_day6"
)


# ------------------------------------------------------------
# Step 1
# ------------------------------------------------------------

STEP1_DIR = (
    DAY6_ROOT
    / "01_step1_freeze_reduced_factor_set"
)

FROZEN_CORE_PATH = (
    STEP1_DIR
    / "frozen_core_factor_set.csv"
)

DAY6_FACTOR_DESIGN_PATH = (
    STEP1_DIR
    / "day6_factor_design.json"
)


# ------------------------------------------------------------
# Step 2
# ------------------------------------------------------------

STEP2_DIR = (
    DAY6_ROOT
    / "02_step2_forward_labels"
)

FORWARD_LABEL_PANEL_PATH = (
    STEP2_DIR
    / "forward_label_panel.parquet"
)

STEP2_LABEL_DESIGN_PATH = (
    STEP2_DIR
    / "step2_label_design.json"
)

STEP2_METADATA_PATH = (
    STEP2_DIR
    / "step2_forward_label_metadata.json"
)


# ------------------------------------------------------------
# Step 3
# Only used for QA lineage.
# We NEVER select factors from Step-3 IC results.
# ------------------------------------------------------------

STEP3_DIR = (
    DAY6_ROOT
    / "03_step3_alpha_rank_ic"
)

STEP3_METADATA_PATH = (
    STEP3_DIR
    / "step3_alpha_rank_ic_metadata.json"
)


# ------------------------------------------------------------
# Step 4
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY6_ROOT
    / "04_step4_portfolio_sort"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MONTHLY_PORTFOLIO_PATH = (
    OUTPUT_DIR
    / "monthly_portfolio_returns.csv"
)

MONTHLY_SPREAD_PATH = (
    OUTPUT_DIR
    / "monthly_portfolio_spreads.csv"
)

SORT_DIAGNOSTIC_PATH = (
    OUTPUT_DIR
    / "monthly_sort_breakpoints.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "portfolio_sort_summary.csv"
)

YEARLY_PATH = (
    OUTPUT_DIR
    / "portfolio_sort_yearly_summary.csv"
)

CROSS_WINDOW_PATH = (
    OUTPUT_DIR
    / "portfolio_sort_cross_window_consistency.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "step4_portfolio_sort_metadata.json"
)


# ============================================================
# 1. Frozen Step-4 Design
# ============================================================

EXPECTED_CORE_FACTOR_COUNT = 9

EXPECTED_WINDOWS = [
    60,
    120,
    252,
]

N_PORTFOLIOS = 5

PORTFOLIO_LABELS = [
    1,
    2,
    3,
    4,
    5,
]


# ------------------------------------------------------------
# Portfolio formation thresholds
#
# Frozen before examining Step-4 returns.
# ------------------------------------------------------------

MIN_SORT_OBS = 100

MIN_PORTFOLIO_RETURN_OBS = 20


# ------------------------------------------------------------
# Alpha targets
#
# Same targets as Step 3.
# ------------------------------------------------------------

TARGET_SPECS = [

    {
        "target_name":
            "future_excess_return",

        "target_role":
            "PRIMARY_ALPHA",
    },

    {
        "target_name":
            "future_total_return",

        "target_role":
            "ROBUSTNESS_TOTAL_RETURN",
    },
]


# ------------------------------------------------------------
# Portfolio weighting
# ------------------------------------------------------------

PORTFOLIO_WEIGHTING = "EQUAL_WEIGHT"


# ------------------------------------------------------------
# Monthly annualization
# ------------------------------------------------------------

MONTHS_PER_YEAR = 12


# ------------------------------------------------------------
# HAC inference for monthly Q5-Q1 spread
# ------------------------------------------------------------

USE_HAC = True


# ============================================================
# 2. Key Columns
# ============================================================

KEY_COLUMNS = [

    "analysis_date",

    "window",

    "master_index",

    "security_id",
]


LABEL_COLUMNS = [

    "future_return_label_valid",

    "future_expected_trade_days",

    "future_valid_return_days",

    "future_return_coverage",
]


# ============================================================
# 3. Utilities
# ============================================================

def load_json(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing JSON file:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_json_atomic(
    obj,
    path: Path,
):

    temp_path = Path(
        str(path)
        +
        ".tmp"
    )

    with open(
        temp_path,
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
        temp_path,
        path,
    )


def save_csv_atomic(
    df: pd.DataFrame,
    path: Path,
):

    temp_path = Path(
        str(path)
        +
        ".tmp"
    )

    df.to_csv(
        temp_path,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        temp_path,
        path,
    )


def canonical_json_hash(
    obj,
):

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
# 4. Validate Upstream Designs
# ============================================================

def validate_upstream():

    # ========================================================
    # Step 1
    # ========================================================

    factor_design = load_json(
        DAY6_FACTOR_DESIGN_PATH
    )

    if (
        factor_design.get(
            "design_status"
        )
        !=
        "FROZEN"
    ):

        raise RuntimeError(
            "Day-6 factor design is not FROZEN."
        )

    factor_design_hash = (
        factor_design.get(
            "day6_design_hash"
        )
    )

    if not factor_design_hash:

        raise RuntimeError(
            "Missing day6_design_hash."
        )

    # ========================================================
    # Step 2
    # ========================================================

    label_design = load_json(
        STEP2_LABEL_DESIGN_PATH
    )

    if (
        label_design.get(
            "label_version"
        )
        !=
        "holding_return_v2"
    ):

        raise RuntimeError(
            "Step 4 must use holding_return_v2 labels."
        )

    if (
        label_design.get(
            "parent_day6_factor_design_hash"
        )
        !=
        factor_design_hash
    ):

        raise RuntimeError(
            "Factor-design / label-design hash mismatch."
        )

    label_design_hash = (
        label_design.get(
            "step2_label_design_hash"
        )
    )

    if not label_design_hash:

        raise RuntimeError(
            "Missing Step-2 label-design hash."
        )

    step2_metadata = load_json(
        STEP2_METADATA_PATH
    )

    if (
        step2_metadata.get(
            "formal_qa",
            {}
        ).get(
            "all_jobs_pass"
        )
        is not True
    ):

        raise RuntimeError(
            "Step-2 formal QA did not pass."
        )

    # ========================================================
    # Step 3
    #
    # Important:
    # only validate lineage.
    # Do not inspect IC sign for factor selection.
    # ========================================================

    if STEP3_METADATA_PATH.exists():

        step3_metadata = load_json(
            STEP3_METADATA_PATH
        )

        step3_qa = (
            step3_metadata.get(
                "formal_qa",
                {}
            )
        )

        if (
            step3_qa.get(
                "all_formal_qa_pass"
            )
            is not True
        ):

            raise RuntimeError(
                "Step-3 formal QA did not pass."
            )

    else:

        step3_metadata = None

    return (
        factor_design_hash,
        label_design_hash,
        step3_metadata,
    )


# ============================================================
# 5. Load Frozen Core Factors
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
            "No factor_name column in frozen core set."
        )

    if (
        core[
            "factor_name"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate frozen factor names."
        )

    if (
        len(
            core
        )
        !=
        EXPECTED_CORE_FACTOR_COUNT
    ):

        raise RuntimeError(
            "Unexpected frozen core-factor count."
        )

    if (
        "priority"
        not in core.columns
    ):

        core[
            "priority"
        ] = (
            np.arange(
                len(
                    core
                )
            )
            +
            1
        )

    core = (
        core.sort_values(
            "priority"
        )
        .reset_index(
            drop=True
        )
    )

    return core


# ============================================================
# 6. Validate Input Schemas
# ============================================================

def validate_input_schemas(
    factor_names,
):

    factor_columns = (
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
        c
        for c in required_factor
        if c not in factor_columns
    ]

    if missing:

        raise RuntimeError(
            f"Factor panel missing:\n{missing}"
        )

    label_columns = (
        pq.ParquetFile(
            FORWARD_LABEL_PANEL_PATH
        )
        .schema_arrow
        .names
    )

    target_names = [
        x[
            "target_name"
        ]
        for x in TARGET_SPECS
    ]

    required_label = (
        KEY_COLUMNS
        +
        LABEL_COLUMNS
        +
        target_names
    )

    missing = [
        c
        for c in required_label
        if c not in label_columns
    ]

    if missing:

        raise RuntimeError(
            f"Label panel missing:\n{missing}"
        )


# ============================================================
# 7. Load Factor + Forward Label Panel
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

    target_names = [
        x[
            "target_name"
        ]
        for x in TARGET_SPECS
    ]

    validate_input_schemas(
        factor_names
    )

    factors = pd.read_parquet(
        FACTOR_PANEL_PATH,
        columns=(
            KEY_COLUMNS
            +
            factor_names
        ),
    )

    labels = pd.read_parquet(
        FORWARD_LABEL_PANEL_PATH,
        columns=(
            KEY_COLUMNS
            +
            LABEL_COLUMNS
            +
            target_names
        ),
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
            ],
            errors="raise",
        )

        df[
            "window"
        ] = pd.to_numeric(
            df[
                "window"
            ],
            errors="raise",
        ).astype(
            int
        )

        df[
            "master_index"
        ] = pd.to_numeric(
            df[
                "master_index"
            ],
            errors="raise",
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
            "Duplicate factor-panel keys."
        )

    if (
        labels[
            KEY_COLUMNS
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate label-panel keys."
        )

    merged = factors.merge(

        labels,

        on=KEY_COLUMNS,

        how="left",

        validate="one_to_one",

        indicator=True,
    )

    if (
        (
            merged[
                "_merge"
            ]
            !=
            "both"
        )
        .any()
    ):

        raise RuntimeError(
            "Factor row without matching label row."
        )

    merged = merged.drop(
        columns="_merge"
    )

    observed_windows = sorted(
        merged[
            "window"
        ]
        .unique()
        .tolist()
    )

    if (
        observed_windows
        !=
        EXPECTED_WINDOWS
    ):

        raise RuntimeError(
            f"Unexpected windows: {observed_windows}"
        )

    return merged


# ============================================================
# 8. Spearman for 5 Portfolio Returns
# ============================================================

def spearman_5(
    portfolio_returns,
):

    x = np.arange(
        1,
        6,
        dtype=float,
    )

    y = np.asarray(
        portfolio_returns,
        dtype=float,
    )

    if (
        not np.isfinite(
            y
        ).all()
    ):

        return np.nan

    if (
        np.std(
            y
        )
        ==
        0
    ):

        return np.nan

    x_rank = pd.Series(
        x
    ).rank(
        method="average"
    ).to_numpy()

    y_rank = pd.Series(
        y
    ).rank(
        method="average"
    ).to_numpy()

    return float(
        np.corrcoef(
            x_rank,
            y_rank,
        )[
            0,
            1
        ]
    )


# ============================================================
# 9. Build Quintile Assignment
#
# IMPORTANT:
#
# Quintile formation uses contemporaneous factor values ONLY.
#
# Future return availability is NOT used when forming
# breakpoints.
# ============================================================

def assign_quintiles(
    group,
    factor_name,
):

    factor = pd.to_numeric(
        group[
            factor_name
        ],
        errors="coerce",
    )

    factor_valid = (

        factor.notna()

        &

        np.isfinite(
            factor
        )
    )

    x = factor.loc[
        factor_valid
    ]

    n = len(
        x
    )

    result = pd.Series(
        pd.NA,
        index=group.index,
        dtype="Int64",
    )

    diagnostics = {

        "factor_nonmissing_n":
            int(
                n
            ),

        "factor_unique_n":
            int(
                x.nunique()
            ),

        "factor_min":
            (
                float(
                    x.min()
                )
                if
                n
                >
                0
                else
                np.nan
            ),

        "factor_median":
            (
                float(
                    x.median()
                )
                if
                n
                >
                0
                else
                np.nan
            ),

        "factor_max":
            (
                float(
                    x.max()
                )
                if
                n
                >
                0
                else
                np.nan
            ),

        "q20_breakpoint":
            np.nan,

        "q40_breakpoint":
            np.nan,

        "q60_breakpoint":
            np.nan,

        "q80_breakpoint":
            np.nan,

        "portfolio_count":
            0,

        "sort_status":
            "PASS",
    }

    if (
        n
        <
        MIN_SORT_OBS
    ):

        diagnostics[
            "sort_status"
        ] = "INSUFFICIENT_SORT_OBS"

        return (
            result,
            diagnostics,
        )

    if (
        x.nunique()
        <
        N_PORTFOLIOS
    ):

        diagnostics[
            "sort_status"
        ] = "INSUFFICIENT_UNIQUE_VALUES"

        return (
            result,
            diagnostics,
        )

    quantiles = x.quantile(
        [
            0.20,
            0.40,
            0.60,
            0.80,
        ]
    )

    diagnostics[
        "q20_breakpoint"
    ] = float(
        quantiles.loc[
            0.20
        ]
    )

    diagnostics[
        "q40_breakpoint"
    ] = float(
        quantiles.loc[
            0.40
        ]
    )

    diagnostics[
        "q60_breakpoint"
    ] = float(
        quantiles.loc[
            0.60
        ]
    )

    diagnostics[
        "q80_breakpoint"
    ] = float(
        quantiles.loc[
            0.80
        ]
    )

    try:

        assigned = pd.qcut(

            x,

            q=N_PORTFOLIOS,

            labels=False,

            duplicates="drop",
        )

    except ValueError:

        diagnostics[
            "sort_status"
        ] = "QCUT_FAILURE"

        return (
            result,
            diagnostics,
        )

    unique_bins = int(
        assigned.nunique()
    )

    diagnostics[
        "portfolio_count"
    ] = unique_bins

    if (
        unique_bins
        !=
        N_PORTFOLIOS
    ):

        diagnostics[
            "sort_status"
        ] = "FEWER_THAN_5_PORTFOLIOS"

        return (
            result,
            diagnostics,
        )

    result.loc[
        x.index
    ] = (
        assigned.astype(
            int
        )
        +
        1
    )

    return (
        result,
        diagnostics,
    )


# ============================================================
# 10. Monthly Portfolio Sort
# ============================================================

def compute_monthly_portfolios(
    panel,
    core,
):

    factor_names = (
        core[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    portfolio_rows = []

    spread_rows = []

    diagnostic_rows = []

    grouped = panel.groupby(
        [
            "analysis_date",
            "window",
        ],
        sort=True,
    )

    total_jobs = (
        grouped.ngroups
        *
        len(
            factor_names
        )
    )

    counter = 0

    for (
        analysis_date,
        window,
    ), group in grouped:

        group = (
            group.copy()
            .reset_index(
                drop=True
            )
        )

        universe_n = len(
            group
        )

        for (
            factor_order,
            factor_name,
        ) in enumerate(
            factor_names,
            start=1,
        ):

            counter += 1

            if (
                counter == 1
                or
                counter % 100 == 0
                or
                counter == total_jobs
            ):

                print(
                    f"[{counter:4d}/{total_jobs}] "
                    f"W={window} | "
                    f"{analysis_date.date()} | "
                    f"{factor_name}"
                )

            (
                portfolio,
                sort_diag,
            ) = assign_quintiles(
                group=
                    group,

                factor_name=
                    factor_name,
            )

            work = group.copy()

            work[
                "portfolio"
            ] = portfolio

            # ------------------------------------------------
            # Sort diagnostics independent of future returns
            # ------------------------------------------------

            sort_counts = (
                work[
                    "portfolio"
                ]
                .value_counts()
                .to_dict()
            )

            diagnostic_rows.append(
                {

                    "analysis_date":
                        analysis_date,

                    "window":
                        int(
                            window
                        ),

                    "factor_order":
                        factor_order,

                    "factor_name":
                        factor_name,

                    "factor_orientation":
                        "ORIGINAL",

                    "factor_universe_n":
                        int(
                            universe_n
                        ),

                    **sort_diag,

                    "sort_q1_n":
                        int(
                            sort_counts.get(
                                1,
                                0,
                            )
                        ),

                    "sort_q2_n":
                        int(
                            sort_counts.get(
                                2,
                                0,
                            )
                        ),

                    "sort_q3_n":
                        int(
                            sort_counts.get(
                                3,
                                0,
                            )
                        ),

                    "sort_q4_n":
                        int(
                            sort_counts.get(
                                4,
                                0,
                            )
                        ),

                    "sort_q5_n":
                        int(
                            sort_counts.get(
                                5,
                                0,
                            )
                        ),
                }
            )

            # =================================================
            # No valid sort -> record spread status for targets
            # =================================================

            if (
                sort_diag[
                    "sort_status"
                ]
                !=
                "PASS"
            ):

                for target_spec in TARGET_SPECS:

                    spread_rows.append(
                        {

                            "analysis_date":
                                analysis_date,

                            "window":
                                int(
                                    window
                                ),

                            "factor_order":
                                factor_order,

                            "factor_name":
                                factor_name,

                            "factor_orientation":
                                "ORIGINAL",

                            "target_name":
                                target_spec[
                                    "target_name"
                                ],

                            "target_role":
                                target_spec[
                                    "target_role"
                                ],

                            "q1_return":
                                np.nan,

                            "q2_return":
                                np.nan,

                            "q3_return":
                                np.nan,

                            "q4_return":
                                np.nan,

                            "q5_return":
                                np.nan,

                            "q5_minus_q1":
                                np.nan,

                            "q5_minus_q1_bps":
                                np.nan,

                            "portfolio_return_spearman":
                                np.nan,

                            "linear_slope_per_quintile":
                                np.nan,

                            "strictly_increasing":
                                False,

                            "strictly_decreasing":
                                False,

                            "spread_status":
                                sort_diag[
                                    "sort_status"
                                ],
                        }
                    )

                continue

            # =================================================
            # Future outcomes are used only AFTER sort formation
            # =================================================

            formal_valid = (
                work[
                    "future_return_label_valid"
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
            )

            for target_spec in TARGET_SPECS:

                target_name = (
                    target_spec[
                        "target_name"
                    ]
                )

                target_role = (
                    target_spec[
                        "target_role"
                    ]
                )

                target = pd.to_numeric(
                    work[
                        target_name
                    ],
                    errors="coerce",
                )

                target_valid = (

                    formal_valid

                    &

                    target.notna()

                    &

                    np.isfinite(
                        target
                    )

                    &

                    work[
                        "portfolio"
                    ]
                    .notna()
                )

                portfolio_returns = {}

                portfolio_ns = {}

                portfolio_medians = {}

                for q in PORTFOLIO_LABELS:

                    q_mask = (

                        target_valid

                        &

                        (
                            work[
                                "portfolio"
                            ]
                            ==
                            q
                        )
                    )

                    x = target.loc[
                        q_mask
                    ]

                    n_q = len(
                        x
                    )

                    portfolio_ns[
                        q
                    ] = int(
                        n_q
                    )

                    if (
                        n_q
                        >=
                        MIN_PORTFOLIO_RETURN_OBS
                    ):

                        mean_return = float(
                            x.mean()
                        )

                        median_return = float(
                            x.median()
                        )

                    else:

                        mean_return = np.nan

                        median_return = np.nan

                    portfolio_returns[
                        q
                    ] = mean_return

                    portfolio_medians[
                        q
                    ] = median_return

                    portfolio_rows.append(
                        {

                            "analysis_date":
                                analysis_date,

                            "window":
                                int(
                                    window
                                ),

                            "factor_order":
                                factor_order,

                            "factor_name":
                                factor_name,

                            "factor_orientation":
                                "ORIGINAL",

                            "target_name":
                                target_name,

                            "target_role":
                                target_role,

                            "portfolio":
                                int(
                                    q
                                ),

                            "portfolio_weighting":
                                PORTFOLIO_WEIGHTING,

                            "sort_portfolio_n":
                                int(
                                    sort_counts.get(
                                        q,
                                        0,
                                    )
                                ),

                            "return_observation_n":
                                int(
                                    n_q
                                ),

                            "return_coverage_within_portfolio":
                                (
                                    n_q
                                    /
                                    sort_counts.get(
                                        q,
                                        1,
                                    )
                                    if
                                    sort_counts.get(
                                        q,
                                        0,
                                    )
                                    >
                                    0
                                    else
                                    np.nan
                                ),

                            "portfolio_mean_return":
                                mean_return,

                            "portfolio_median_return":
                                median_return,
                        }
                    )

                q_returns = np.array(
                    [
                        portfolio_returns[
                            q
                        ]
                        for q in PORTFOLIO_LABELS
                    ],
                    dtype=float,
                )

                all_portfolios_valid = bool(
                    np.isfinite(
                        q_returns
                    )
                    .all()
                )

                if all_portfolios_valid:

                    spread = float(
                        portfolio_returns[
                            5
                        ]
                        -
                        portfolio_returns[
                            1
                        ]
                    )

                    portfolio_spearman = (
                        spearman_5(
                            q_returns
                        )
                    )

                    linear_slope = float(
                        np.polyfit(
                            np.arange(
                                1,
                                6,
                                dtype=float,
                            ),
                            q_returns,
                            deg=1,
                        )[
                            0
                        ]
                    )

                    diff = np.diff(
                        q_returns
                    )

                    strictly_increasing = bool(
                        np.all(
                            diff
                            >
                            0
                        )
                    )

                    strictly_decreasing = bool(
                        np.all(
                            diff
                            <
                            0
                        )
                    )

                    status = "PASS"

                else:

                    spread = np.nan

                    portfolio_spearman = np.nan

                    linear_slope = np.nan

                    strictly_increasing = False

                    strictly_decreasing = False

                    status = (
                        "INSUFFICIENT_PORTFOLIO_RETURN_OBS"
                    )

                spread_rows.append(
                    {

                        "analysis_date":
                            analysis_date,

                        "window":
                            int(
                                window
                            ),

                        "factor_order":
                            factor_order,

                        "factor_name":
                            factor_name,

                        "factor_orientation":
                            "ORIGINAL",

                        "target_name":
                            target_name,

                        "target_role":
                            target_role,

                        "q1_n":
                            portfolio_ns[
                                1
                            ],

                        "q2_n":
                            portfolio_ns[
                                2
                            ],

                        "q3_n":
                            portfolio_ns[
                                3
                            ],

                        "q4_n":
                            portfolio_ns[
                                4
                            ],

                        "q5_n":
                            portfolio_ns[
                                5
                            ],

                        "q1_return":
                            portfolio_returns[
                                1
                            ],

                        "q2_return":
                            portfolio_returns[
                                2
                            ],

                        "q3_return":
                            portfolio_returns[
                                3
                            ],

                        "q4_return":
                            portfolio_returns[
                                4
                            ],

                        "q5_return":
                            portfolio_returns[
                                5
                            ],

                        "q5_minus_q1":
                            spread,

                        "q5_minus_q1_bps":
                            (
                                spread
                                *
                                10000.0
                                if
                                np.isfinite(
                                    spread
                                )
                                else
                                np.nan
                            ),

                        "portfolio_return_spearman":
                            portfolio_spearman,

                        "linear_slope_per_quintile":
                            linear_slope,

                        "strictly_increasing":
                            strictly_increasing,

                        "strictly_decreasing":
                            strictly_decreasing,

                        "spread_status":
                            status,
                    }
                )

    portfolio_df = pd.DataFrame(
        portfolio_rows
    )

    spread_df = pd.DataFrame(
        spread_rows
    )

    diagnostic_df = pd.DataFrame(
        diagnostic_rows
    )

    return (
        portfolio_df,
        spread_df,
        diagnostic_df,
    )


# ============================================================
# 11. HAC / Newey-West
# ============================================================

def automatic_hac_lag(
    n: int,
):

    if (
        n
        <=
        1
    ):

        return 0

    lag = int(
        np.floor(
            4.0
            *
            (
                n
                /
                100.0
            )
            **
            (
                2.0
                /
                9.0
            )
        )
    )

    return max(
        0,
        min(
            lag,
            n - 1,
        ),
    )


def mean_inference(
    values,
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

    n = len(
        x
    )

    if (
        n
        <
        2
    ):

        return {

            "naive_se":
                np.nan,

            "naive_t":
                np.nan,

            "hac_lag":
                0,

            "hac_se":
                np.nan,

            "hac_t":
                np.nan,
        }

    mean_x = float(
        x.mean()
    )

    sd_x = float(
        x.std(
            ddof=1
        )
    )

    if (
        sd_x
        >
        0
    ):

        naive_se = (
            sd_x
            /
            np.sqrt(
                n
            )
        )

        naive_t = (
            mean_x
            /
            naive_se
        )

    else:

        naive_se = np.nan

        naive_t = np.nan

    if (
        not USE_HAC
        or
        n
        <
        3
    ):

        return {

            "naive_se":
                naive_se,

            "naive_t":
                naive_t,

            "hac_lag":
                0,

            "hac_se":
                naive_se,

            "hac_t":
                naive_t,
        }

    lag = automatic_hac_lag(
        n
    )

    residual = (
        x
        -
        mean_x
    )

    gamma0 = float(
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

        gamma = float(
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
            1.0
            -
            ell
            /
            (
                lag
                +
                1.0
            )
        )

        long_run_var += (
            2.0
            *
            weight
            *
            gamma
        )

    long_run_var = max(
        long_run_var,
        0.0,
    )

    hac_var_mean = (
        long_run_var
        /
        n
    )

    if (
        hac_var_mean
        >
        0
    ):

        hac_se = float(
            np.sqrt(
                hac_var_mean
            )
        )

        hac_t = (
            mean_x
            /
            hac_se
        )

    else:

        hac_se = np.nan

        hac_t = np.nan

    return {

        "naive_se":
            naive_se,

        "naive_t":
            naive_t,

        "hac_lag":
            int(
                lag
            ),

        "hac_se":
            hac_se,

        "hac_t":
            hac_t,
    }


# ============================================================
# 12. Portfolio-Sort Summary
# ============================================================

def summarize_portfolio_sorts(
    spread_df,
):

    rows = []

    passed = (
        spread_df[
            spread_df[
                "spread_status"
            ]
            ==
            "PASS"
        ]
        .copy()
    )

    group_cols = [

        "factor_order",

        "factor_name",

        "target_name",

        "target_role",

        "window",
    ]

    for keys, group in (
        passed.groupby(
            group_cols,
            sort=False,
        )
    ):

        (
            factor_order,
            factor_name,
            target_name,
            target_role,
            window,
        ) = keys

        group = (
            group.sort_values(
                "analysis_date"
            )
        )

        spread = (
            group[
                "q5_minus_q1"
            ]
            .to_numpy(
                dtype=float
            )
        )

        n = len(
            spread
        )

        mean_spread = float(
            np.mean(
                spread
            )
        )

        median_spread = float(
            np.median(
                spread
            )
        )

        std_spread = (
            float(
                np.std(
                    spread,
                    ddof=1,
                )
            )
            if
            n
            >=
            2
            else
            np.nan
        )

        annualized_spread = (
            MONTHS_PER_YEAR
            *
            mean_spread
        )

        if (
            np.isfinite(
                std_spread
            )
            and
            std_spread
            >
            0
        ):

            annualized_sharpe = (

                np.sqrt(
                    MONTHS_PER_YEAR
                )

                *

                mean_spread

                /

                std_spread
            )

        else:

            annualized_sharpe = np.nan

        inference = mean_inference(
            spread
        )

        rows.append(
            {

                "factor_order":
                    int(
                        factor_order
                    ),

                "factor_name":
                    factor_name,

                "factor_orientation":
                    "ORIGINAL",

                "target_name":
                    target_name,

                "target_role":
                    target_role,

                "window":
                    int(
                        window
                    ),

                "month_count":
                    int(
                        n
                    ),

                "first_date":
                    group[
                        "analysis_date"
                    ]
                    .min(),

                "last_date":
                    group[
                        "analysis_date"
                    ]
                    .max(),

                "mean_q1_return":
                    float(
                        group[
                            "q1_return"
                        ]
                        .mean()
                    ),

                "mean_q2_return":
                    float(
                        group[
                            "q2_return"
                        ]
                        .mean()
                    ),

                "mean_q3_return":
                    float(
                        group[
                            "q3_return"
                        ]
                        .mean()
                    ),

                "mean_q4_return":
                    float(
                        group[
                            "q4_return"
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
                    mean_spread,

                "mean_q5_minus_q1_bps":
                    (
                        mean_spread
                        *
                        10000.0
                    ),

                "median_q5_minus_q1":
                    median_spread,

                "std_q5_minus_q1":
                    std_spread,

                "annualized_arithmetic_spread":
                    annualized_spread,

                "annualized_spread_sharpe":
                    annualized_sharpe,

                "positive_spread_share":
                    float(
                        np.mean(
                            spread
                            >
                            0
                        )
                    ),

                "negative_spread_share":
                    float(
                        np.mean(
                            spread
                            <
                            0
                        )
                    ),

                "mean_portfolio_return_spearman":
                    float(
                        group[
                            "portfolio_return_spearman"
                        ]
                        .mean()
                    ),

                "median_portfolio_return_spearman":
                    float(
                        group[
                            "portfolio_return_spearman"
                        ]
                        .median()
                    ),

                "strictly_increasing_share":
                    float(
                        group[
                            "strictly_increasing"
                        ]
                        .mean()
                    ),

                "strictly_decreasing_share":
                    float(
                        group[
                            "strictly_decreasing"
                        ]
                        .mean()
                    ),

                "mean_linear_slope_per_quintile":
                    float(
                        group[
                            "linear_slope_per_quintile"
                        ]
                        .mean()
                    ),

                "mean_q1_n":
                    float(
                        group[
                            "q1_n"
                        ]
                        .mean()
                    ),

                "mean_q5_n":
                    float(
                        group[
                            "q5_n"
                        ]
                        .mean()
                    ),

                "naive_spread_se":
                    inference[
                        "naive_se"
                    ],

                "naive_spread_t_stat":
                    inference[
                        "naive_t"
                    ],

                "hac_lag":
                    inference[
                        "hac_lag"
                    ],

                "hac_spread_se":
                    inference[
                        "hac_se"
                    ],

                "hac_spread_t_stat":
                    inference[
                        "hac_t"
                    ],

                "outcome_based_sign_flip":
                    False,

                "outcome_based_window_selection":
                    False,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "target_role",
                "factor_order",
                "window",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 13. Yearly Spread Summary
# ============================================================

def build_yearly_summary(
    spread_df,
):

    df = (
        spread_df[
            spread_df[
                "spread_status"
            ]
            ==
            "PASS"
        ]
        .copy()
    )

    df[
        "year"
    ] = (
        pd.to_datetime(
            df[
                "analysis_date"
            ]
        )
        .dt.year
    )

    result = (

        df.groupby(
            [
                "factor_order",
                "factor_name",
                "target_name",
                "target_role",
                "window",
                "year",
            ],
            as_index=False,
        )

        .agg(

            month_count=(
                "q5_minus_q1",
                "count",
            ),

            mean_q5_minus_q1=(
                "q5_minus_q1",
                "mean",
            ),

            median_q5_minus_q1=(
                "q5_minus_q1",
                "median",
            ),

            positive_spread_share=(
                "q5_minus_q1",
                lambda x:
                    float(
                        (
                            x
                            >
                            0
                        )
                        .mean()
                    ),
            ),

            mean_portfolio_return_spearman=(
                "portfolio_return_spearman",
                "mean",
            ),
        )
    )

    result[
        "mean_q5_minus_q1_bps"
    ] = (
        result[
            "mean_q5_minus_q1"
        ]
        *
        10000.0
    )

    return result


# ============================================================
# 14. Cross-Window Consistency
# ============================================================

def build_cross_window_consistency(
    summary,
):

    rows = []

    group_cols = [

        "factor_order",

        "factor_name",

        "target_name",

        "target_role",
    ]

    for keys, group in (
        summary.groupby(
            group_cols,
            sort=False,
        )
    ):

        (
            factor_order,
            factor_name,
            target_name,
            target_role,
        ) = keys

        group = (
            group.set_index(
                "window"
            )
        )

        spread_map = {}

        hac_map = {}

        for w in EXPECTED_WINDOWS:

            if (
                w
                in
                group.index
            ):

                spread_map[
                    w
                ] = float(
                    group.loc[
                        w,
                        "mean_q5_minus_q1",
                    ]
                )

                hac_map[
                    w
                ] = float(
                    group.loc[
                        w,
                        "hac_spread_t_stat",
                    ]
                )

            else:

                spread_map[
                    w
                ] = np.nan

                hac_map[
                    w
                ] = np.nan

        values = np.array(
            [
                spread_map[
                    60
                ],
                spread_map[
                    120
                ],
                spread_map[
                    252
                ],
            ],
            dtype=float,
        )

        if (
            np.isfinite(
                values
            )
            .all()
        ):

            signs = np.sign(
                values
            )

            same_sign = bool(

                np.all(
                    signs
                    ==
                    signs[
                        0
                    ]
                )

                and

                np.all(
                    signs
                    !=
                    0
                )
            )

            spread_range = float(
                values.max()
                -
                values.min()
            )

        else:

            same_sign = False

            spread_range = np.nan

        rows.append(
            {

                "factor_order":
                    int(
                        factor_order
                    ),

                "factor_name":
                    factor_name,

                "target_name":
                    target_name,

                "target_role":
                    target_role,

                "mean_spread_w60":
                    spread_map[
                        60
                    ],

                "mean_spread_w120":
                    spread_map[
                        120
                    ],

                "mean_spread_w252":
                    spread_map[
                        252
                    ],

                "hac_t_w60":
                    hac_map[
                        60
                    ],

                "hac_t_w120":
                    hac_map[
                        120
                    ],

                "hac_t_w252":
                    hac_map[
                        252
                    ],

                "same_spread_sign_all_windows":
                    same_sign,

                "cross_window_spread_range":
                    spread_range,

                "factor_orientation":
                    "ORIGINAL",

                "window_selected_from_outcome":
                    False,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "target_role",
                "factor_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 15. Formal QA
# ============================================================

def formal_qa(
    portfolio_df,
    spread_df,
    diagnostics_df,
    summary,
    core,
):

    qa = {}

    qa[
        "core_factor_count"
    ] = int(
        len(
            core
        )
    )

    qa[
        "core_factor_count_ok"
    ] = bool(
        len(
            core
        )
        ==
        EXPECTED_CORE_FACTOR_COUNT
    )

    qa[
        "portfolio_labels"
    ] = sorted(
        portfolio_df[
            "portfolio"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    qa[
        "portfolio_labels_ok"
    ] = bool(
        qa[
            "portfolio_labels"
        ]
        ==
        PORTFOLIO_LABELS
    )

    qa[
        "all_factor_orientation_original"
    ] = bool(
        (
            spread_df[
                "factor_orientation"
            ]
            ==
            "ORIGINAL"
        )
        .all()
    )

    qa[
        "spread_key_unique"
    ] = bool(

        not spread_df[
            [
                "analysis_date",
                "window",
                "factor_name",
                "target_name",
            ]
        ]
        .duplicated()
        .any()
    )

    qa[
        "diagnostic_key_unique"
    ] = bool(

        not diagnostics_df[
            [
                "analysis_date",
                "window",
                "factor_name",
            ]
        ]
        .duplicated()
        .any()
    )

    qa[
        "summary_window_set"
    ] = sorted(
        summary[
            "window"
        ]
        .unique()
        .astype(int)
        .tolist()
    )

    qa[
        "window_set_ok"
    ] = bool(
        qa[
            "summary_window_set"
        ]
        ==
        EXPECTED_WINDOWS
    )

    qa[
        "spread_status_counts"
    ] = (
        spread_df[
            "spread_status"
        ]
        .value_counts()
        .to_dict()
    )

    qa[
        "sort_status_counts"
    ] = (
        diagnostics_df[
            "sort_status"
        ]
        .value_counts()
        .to_dict()
    )

    passed = (
        spread_df[
            spread_df[
                "spread_status"
            ]
            ==
            "PASS"
        ]
    )

    qa[
        "pass_spread_count"
    ] = int(
        len(
            passed
        )
    )

    qa[
        "nonpass_spread_count"
    ] = int(
        len(
            spread_df
        )
        -
        len(
            passed
        )
    )

    finite_check = (
        passed[
            "q5_minus_q1"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .notna()
        .all()
    )

    qa[
        "all_pass_spreads_finite"
    ] = bool(
        finite_check
    )

    required = [

        "core_factor_count_ok",

        "portfolio_labels_ok",

        "all_factor_orientation_original",

        "spread_key_unique",

        "diagnostic_key_unique",

        "window_set_ok",

        "all_pass_spreads_finite",
    ]

    qa[
        "all_formal_qa_pass"
    ] = bool(
        all(
            qa[
                x
            ]
            for x in required
        )
    )

    return qa


# ============================================================
# 16. Step-4 Design
# ============================================================

def build_step4_design(
    factor_design_hash,
    label_design_hash,
    core,
):

    payload = {

        "research_day":
            6,

        "step":
            "Step4_Portfolio_Sort_Economic_Significance",

        "parent_factor_design_hash":
            factor_design_hash,

        "parent_label_design_hash":
            label_design_hash,

        "factor_set":
            core[
                "factor_name"
            ]
            .astype(str)
            .tolist(),

        "windows":
            EXPECTED_WINDOWS,

        "portfolio_count":
            N_PORTFOLIOS,

        "portfolio_weighting":
            PORTFOLIO_WEIGHTING,

        "portfolio_definition":
            (
                "Cross-sectional factor quintiles "
                "formed independently for every "
                "analysis_date x window x factor."
            ),

        "portfolio_orientation":
            (
                "Q1 = lowest factor values; "
                "Q5 = highest factor values."
            ),

        "long_short_definition":
            "Q5 - Q1",

        "factor_orientation":
            "ORIGINAL",

        "outcome_based_sign_flip":
            False,

        "outcome_based_window_selection":
            False,

        "future_label_used_for_breakpoint_formation":
            False,

        "minimum_sort_observations":
            MIN_SORT_OBS,

        "minimum_portfolio_return_observations":
            MIN_PORTFOLIO_RETURN_OBS,

        "targets":
            TARGET_SPECS,

        "primary_target":
            "future_excess_return",

        "transaction_costs_included":
            False,

        "neutralization":
            False,

        "hac_inference":
            USE_HAC,

        "economic_metrics": [

            "Q1-Q5 average returns",

            "Q5-minus-Q1 spread",

            "annualized arithmetic spread",

            "annualized spread Sharpe",

            "portfolio-return monotonicity",

            "spread sign persistence",

            "year-by-year spread stability",
        ],
    }

    design_hash = (
        canonical_json_hash(
            payload
        )
    )

    return {

        **payload,

        "step4_design_hash":
            design_hash,
    }


# ============================================================
# 17. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 6 - Step 4"
    )

    print(
        "Portfolio Sort - Economic Significance"
    )

    print("=" * 80)

    # ========================================================
    # Upstream QA
    # ========================================================

    print()
    print(
        "[1] Validate frozen upstream designs"
    )

    (
        factor_design_hash,
        label_design_hash,
        step3_metadata,
    ) = validate_upstream()

    print(
        f"Factor design hash:\n"
        f"{factor_design_hash}"
    )

    print()

    print(
        f"Label design hash:\n"
        f"{label_design_hash}"
    )

    # ========================================================
    # Frozen factors
    # ========================================================

    print()
    print(
        "[2] Load frozen core factors"
    )

    core = load_core_factors()

    print(
        core[
            [
                "priority",
                "factor_name",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Step-4 design
    # ========================================================

    step4_design = (
        build_step4_design(

            factor_design_hash=
                factor_design_hash,

            label_design_hash=
                label_design_hash,

            core=
                core,
        )
    )

    print()
    print(
        "Step-4 design hash:"
    )

    print(
        step4_design[
            "step4_design_hash"
        ]
    )

    # ========================================================
    # Analysis panel
    # ========================================================

    print()
    print(
        "[3] Load factor + forward-return panel"
    )

    panel = load_analysis_panel(
        core
    )

    print(
        f"Analysis rows: "
        f"{len(panel):,}"
    )

    # ========================================================
    # Portfolio sorts
    # ========================================================

    print()
    print(
        "[4] Run monthly quintile portfolio sorts"
    )

    (
        portfolio_df,
        spread_df,
        diagnostics_df,
    ) = compute_monthly_portfolios(

        panel=
            panel,

        core=
            core,
    )

    save_csv_atomic(
        portfolio_df,
        MONTHLY_PORTFOLIO_PATH,
    )

    save_csv_atomic(
        spread_df,
        MONTHLY_SPREAD_PATH,
    )

    save_csv_atomic(
        diagnostics_df,
        SORT_DIAGNOSTIC_PATH,
    )

    # ========================================================
    # Full-sample summary
    # ========================================================

    print()
    print(
        "[5] Build full-sample economic summary"
    )

    summary = (
        summarize_portfolio_sorts(
            spread_df
        )
    )

    save_csv_atomic(
        summary,
        SUMMARY_PATH,
    )

    # ========================================================
    # Yearly stability
    # ========================================================

    print()
    print(
        "[6] Build yearly spread summary"
    )

    yearly = (
        build_yearly_summary(
            spread_df
        )
    )

    save_csv_atomic(
        yearly,
        YEARLY_PATH,
    )

    # ========================================================
    # Cross-window consistency
    # ========================================================

    print()
    print(
        "[7] Build cross-window consistency"
    )

    cross_window = (
        build_cross_window_consistency(
            summary
        )
    )

    save_csv_atomic(
        cross_window,
        CROSS_WINDOW_PATH,
    )

    # ========================================================
    # QA
    # ========================================================

    qa = formal_qa(

        portfolio_df=
            portfolio_df,

        spread_df=
            spread_df,

        diagnostics_df=
            diagnostics_df,

        summary=
            summary,

        core=
            core,
    )

    if (
        not qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            f"Step-4 formal QA failed:\n{qa}"
        )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {

        **step4_design,

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "formal_qa":
            qa,

        "outputs": {

            "monthly_portfolio_returns":
                str(
                    MONTHLY_PORTFOLIO_PATH
                ),

            "monthly_portfolio_spreads":
                str(
                    MONTHLY_SPREAD_PATH
                ),

            "monthly_sort_breakpoints":
                str(
                    SORT_DIAGNOSTIC_PATH
                ),

            "portfolio_sort_summary":
                str(
                    SUMMARY_PATH
                ),

            "portfolio_sort_yearly_summary":
                str(
                    YEARLY_PATH
                ),

            "cross_window_consistency":
                str(
                    CROSS_WINDOW_PATH
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
        "Step-4 Formal QA"
    )
    print("=" * 80)

    print(
        f"Spread PASS: "
        f"{qa['pass_spread_count']:,}"
    )

    print(
        f"Spread non-PASS: "
        f"{qa['nonpass_spread_count']:,}"
    )

    print(
        f"QA pass: "
        f"{qa['all_formal_qa_pass']}"
    )

    print()
    print("=" * 80)
    print(
        "Primary Alpha Portfolio-Sort Summary"
    )
    print("=" * 80)

    primary = (
        summary[
            summary[
                "target_role"
            ]
            ==
            "PRIMARY_ALPHA"
        ]
    )

    display_columns = [

        "factor_name",

        "window",

        "mean_q1_return",

        "mean_q2_return",

        "mean_q3_return",

        "mean_q4_return",

        "mean_q5_return",

        "mean_q5_minus_q1",

        "mean_q5_minus_q1_bps",

        "annualized_arithmetic_spread",

        "annualized_spread_sharpe",

        "positive_spread_share",

        "mean_portfolio_return_spearman",

        "hac_spread_t_stat",
    ]

    print(
        primary[
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
        "Day 6 Step 4 Complete"
    )
    print("=" * 80)


if __name__ == "__main__":

    main()