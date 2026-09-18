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
#
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
# Day 6 Step 1:
# frozen reduced factor set
#
# Research Day 6 is stored under M1_day5
# ------------------------------------------------------------

DAY6_ROOT = (
    OUTPUT_ROOT
    / "M1_day6"
)

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
# Day 6 Step 2:
# FINAL holding-return-v2 forward labels
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
# Day 6 Step 3
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY6_ROOT
    / "03_step3_alpha_rank_ic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MONTHLY_IC_PATH = (
    OUTPUT_DIR
    / "monthly_rank_ic.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "rank_ic_summary.csv"
)

CROSS_WINDOW_PATH = (
    OUTPUT_DIR
    / "rank_ic_cross_window_consistency.csv"
)

CORE_FACTOR_PATH = (
    OUTPUT_DIR
    / "step3_core_factor_inventory.csv"
)

TARGET_PATH = (
    OUTPUT_DIR
    / "step3_alpha_target_inventory.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "step3_alpha_rank_ic_metadata.json"
)


# ============================================================
# 1. Frozen Step-3 Design
# ============================================================

EXPECTED_CORE_FACTOR_COUNT = 9

EXPECTED_WINDOWS = [
    60,
    120,
    252,
]


# ------------------------------------------------------------
# Rank IC needs enough cross-sectional observations.
#
# This threshold is frozen BEFORE observing IC results.
# ------------------------------------------------------------

MIN_CROSS_SECTION_OBS = 100


# ------------------------------------------------------------
# Main and robustness Alpha targets
# ------------------------------------------------------------

TARGET_SPECS = [

    {
        "target_name":
            "future_excess_return",

        "target_role":
            "PRIMARY_ALPHA",

        "description":
            (
                "Future stock holding-period return "
                "minus future leave-one-out equal-weight "
                "market holding-period return."
            ),
    },

    {
        "target_name":
            "future_total_return",

        "target_role":
            "ROBUSTNESS_TOTAL_RETURN",

        "description":
            (
                "Future stock holding-period total return."
            ),
    },
]


# ------------------------------------------------------------
# Newey-West / HAC
#
# Used only as auxiliary inference for mean IC.
#
# Automatic lag:
#
# floor(4 * (T/100)^(2/9))
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


LABEL_DIAGNOSTIC_COLUMNS = [

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


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot hash missing file:\n{path}"
        )

    hasher = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:

                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


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
# 4. Validate Frozen Research Design
# ============================================================

def validate_designs():

    # --------------------------------------------------------
    # Day-6 factor design
    # --------------------------------------------------------

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

    factor_count = int(
        factor_design[
            "factor_sets"
        ][
            "core_factor_count"
        ]
    )

    if (
        factor_count
        !=
        EXPECTED_CORE_FACTOR_COUNT
    ):

        raise RuntimeError(
            "Unexpected frozen core factor count.\n"
            f"Expected {EXPECTED_CORE_FACTOR_COUNT}, "
            f"got {factor_count}."
        )

    # --------------------------------------------------------
    # Step-2 label design
    # --------------------------------------------------------

    label_design = load_json(
        STEP2_LABEL_DESIGN_PATH
    )

    label_version = (
        label_design.get(
            "label_version"
        )
    )

    if (
        label_version
        !=
        "holding_return_v2"
    ):

        raise RuntimeError(
            "Step 3 must use FINAL "
            "holding_return_v2 labels.\n"
            f"Found label_version={label_version}"
        )

    parent_hash = (
        label_design.get(
            "parent_day6_factor_design_hash"
        )
    )

    if (
        parent_hash
        !=
        factor_design_hash
    ):

        raise RuntimeError(
            "Step-2 labels and Step-1 factor set "
            "belong to different Day-6 designs."
        )

    label_design_hash = (
        label_design.get(
            "step2_label_design_hash"
        )
    )

    if not label_design_hash:

        raise RuntimeError(
            "Missing step2_label_design_hash."
        )

    # --------------------------------------------------------
    # Step-2 metadata QA
    # --------------------------------------------------------

    step2_metadata = load_json(
        STEP2_METADATA_PATH
    )

    metadata_label_version = (
        step2_metadata.get(
            "label_version"
        )
    )

    if (
        metadata_label_version
        !=
        "holding_return_v2"
    ):

        raise RuntimeError(
            "Step-2 metadata does not refer to "
            "holding_return_v2."
        )

    metadata_factor_hash = (
        step2_metadata.get(
            "day6_factor_design_hash"
        )
    )

    if (
        metadata_factor_hash
        !=
        factor_design_hash
    ):

        raise RuntimeError(
            "Step-2 metadata factor hash mismatch."
        )

    metadata_label_hash = (
        step2_metadata.get(
            "step2_label_design_hash"
        )
    )

    if (
        metadata_label_hash
        !=
        label_design_hash
    ):

        raise RuntimeError(
            "Step-2 metadata label hash mismatch."
        )

    formal_qa = (
        step2_metadata.get(
            "formal_qa",
            {}
        )
    )

    if (
        formal_qa.get(
            "all_jobs_pass"
        )
        is not True
    ):

        raise RuntimeError(
            "Step-2 forward labels did not "
            "pass all formal QA."
        )

    return (
        factor_design,
        factor_design_hash,
        label_design,
        label_design_hash,
        step2_metadata,
    )


# ============================================================
# 5. Load Frozen Core Factors
# ============================================================

def load_core_factors():

    if not FROZEN_CORE_PATH.exists():

        raise FileNotFoundError(
            f"Missing frozen core factors:\n"
            f"{FROZEN_CORE_PATH}"
        )

    core = pd.read_csv(
        FROZEN_CORE_PATH
    )

    if (
        "factor_name"
        not in core.columns
    ):

        raise RuntimeError(
            "frozen_core_factor_set.csv "
            "has no factor_name column."
        )

    if (
        core[
            "factor_name"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate core factor names."
        )

    if (
        len(
            core
        )
        !=
        EXPECTED_CORE_FACTOR_COUNT
    ):

        raise RuntimeError(
            "Frozen core factor count mismatch."
        )

    # --------------------------------------------------------
    # Preserve pre-outcome priority/order.
    #
    # Do NOT sort by future IC.
    # --------------------------------------------------------

    if (
        "priority"
        in core.columns
    ):

        core = (
            core.sort_values(
                "priority"
            )
            .reset_index(
                drop=True
            )
        )

    else:

        core = (
            core.reset_index(
                drop=True
            )
        )

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

    core[
        "step3_factor_orientation"
    ] = "ORIGINAL"

    core[
        "outcome_based_sign_flip"
    ] = False

    return core


# ============================================================
# 6. Validate Panel Schemas
# ============================================================

def validate_input_schemas(
    factor_names,
):

    # --------------------------------------------------------
    # Factor panel
    # --------------------------------------------------------

    factor_schema = (
        pq.ParquetFile(
            FACTOR_PANEL_PATH
        )
        .schema_arrow
        .names
    )

    required_factor_columns = (
        KEY_COLUMNS
        +
        list(
            factor_names
        )
    )

    missing_factor_columns = [
        col
        for col in required_factor_columns
        if col not in factor_schema
    ]

    if missing_factor_columns:

        raise RuntimeError(
            "Factor panel missing columns:\n"
            f"{missing_factor_columns}"
        )

    # --------------------------------------------------------
    # Label panel
    # --------------------------------------------------------

    label_schema = (
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

    required_label_columns = (
        KEY_COLUMNS
        +
        LABEL_DIAGNOSTIC_COLUMNS
        +
        target_names
    )

    missing_label_columns = [
        col
        for col in required_label_columns
        if col not in label_schema
    ]

    if missing_label_columns:

        raise RuntimeError(
            "Forward-label panel missing columns:\n"
            f"{missing_label_columns}"
        )


# ============================================================
# 7. Load Analysis Panel
# ============================================================

def load_analysis_panel(
    core_factors,
):

    factor_names = (
        core_factors[
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

    print()
    print(
        "Loading factor panel..."
    )

    factors = pd.read_parquet(
        FACTOR_PANEL_PATH,
        columns=(
            KEY_COLUMNS
            +
            factor_names
        ),
    )

    print(
        f"Factor rows: "
        f"{len(factors):,}"
    )

    print()
    print(
        "Loading FINAL forward labels..."
    )

    labels = pd.read_parquet(
        FORWARD_LABEL_PANEL_PATH,
        columns=(
            KEY_COLUMNS
            +
            LABEL_DIAGNOSTIC_COLUMNS
            +
            target_names
        ),
    )

    print(
        f"Label rows: "
        f"{len(labels):,}"
    )

    # --------------------------------------------------------
    # Normalize keys
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Duplicate-key QA
    # --------------------------------------------------------

    if (
        factors[
            KEY_COLUMNS
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate keys in factor panel."
        )

    if (
        labels[
            KEY_COLUMNS
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate keys in label panel."
        )

    if (
        len(
            factors
        )
        !=
        len(
            labels
        )
    ):

        raise RuntimeError(
            "Factor and label panel row counts differ.\n"
            f"factor={len(factors):,}\n"
            f"label={len(labels):,}"
        )

    # --------------------------------------------------------
    # Strict one-to-one merge
    # --------------------------------------------------------

    merged = factors.merge(

        labels,

        on=KEY_COLUMNS,

        how="left",

        validate="one_to_one",

        indicator=True,
    )

    missing_label_rows = int(
        (
            merged[
                "_merge"
            ]
            !=
            "both"
        )
        .sum()
    )

    if (
        missing_label_rows
        >
        0
    ):

        raise RuntimeError(
            "Some frozen factor rows have no "
            "matching forward-label row.\n"
            f"Missing rows={missing_label_rows:,}"
        )

    merged = merged.drop(
        columns=[
            "_merge"
        ]
    )

    # --------------------------------------------------------
    # Window QA
    # --------------------------------------------------------

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
            "Unexpected window set.\n"
            f"Expected={EXPECTED_WINDOWS}\n"
            f"Observed={observed_windows}"
        )

    # --------------------------------------------------------
    # Valid-label completeness QA
    # --------------------------------------------------------

    valid_label_mask = (
        merged[
            "future_return_label_valid"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    for target in target_names:

        if (
            merged.loc[
                valid_label_mask,
                target,
            ]
            .isna()
            .any()
        ):

            raise RuntimeError(
                "Step-2 says return label is valid, "
                f"but {target} contains NA."
            )

    return merged


# ============================================================
# 8. Spearman Rank IC
# ============================================================

def compute_spearman_rank_ic(
    factor_values,
    target_values,
):

    x = pd.to_numeric(
        factor_values,
        errors="coerce",
    )

    y = pd.to_numeric(
        target_values,
        errors="coerce",
    )

    valid = (

        x.notna()

        &

        y.notna()

        &

        np.isfinite(
            x
        )

        &

        np.isfinite(
            y
        )
    )

    x = x.loc[
        valid
    ]

    y = y.loc[
        valid
    ]

    n = len(
        x
    )

    if (
        n
        <
        MIN_CROSS_SECTION_OBS
    ):

        return {
            "rank_ic":
                np.nan,

            "pair_n":
                n,

            "factor_unique_n":
                int(
                    x.nunique()
                ),

            "target_unique_n":
                int(
                    y.nunique()
                ),

            "status":
                "INSUFFICIENT_OBS",
        }

    factor_unique_n = int(
        x.nunique()
    )

    target_unique_n = int(
        y.nunique()
    )

    if (
        factor_unique_n
        <
        2
    ):

        return {
            "rank_ic":
                np.nan,

            "pair_n":
                n,

            "factor_unique_n":
                factor_unique_n,

            "target_unique_n":
                target_unique_n,

            "status":
                "NO_FACTOR_VARIATION",
        }

    if (
        target_unique_n
        <
        2
    ):

        return {
            "rank_ic":
                np.nan,

            "pair_n":
                n,

            "factor_unique_n":
                factor_unique_n,

            "target_unique_n":
                target_unique_n,

            "status":
                "NO_TARGET_VARIATION",
        }

    # --------------------------------------------------------
    # Spearman correlation =
    # Pearson correlation of average ranks.
    # --------------------------------------------------------

    x_rank = (
        x.rank(
            method="average"
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    y_rank = (
        y.rank(
            method="average"
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    ic = float(
        np.corrcoef(
            x_rank,
            y_rank,
        )[
            0,
            1
        ]
    )

    if (
        not np.isfinite(
            ic
        )
    ):

        return {
            "rank_ic":
                np.nan,

            "pair_n":
                n,

            "factor_unique_n":
                factor_unique_n,

            "target_unique_n":
                target_unique_n,

            "status":
                "NONFINITE_IC",
        }

    # Numerical tolerance
    if (
        abs(
            ic
        )
        >
        1.0
        +
        1e-10
    ):

        raise RuntimeError(
            f"Impossible Rank IC={ic}"
        )

    ic = float(
        np.clip(
            ic,
            -1.0,
            1.0,
        )
    )

    return {
        "rank_ic":
            ic,

        "pair_n":
            n,

        "factor_unique_n":
            factor_unique_n,

        "target_unique_n":
            target_unique_n,

        "status":
            "PASS",
    }


# ============================================================
# 9. Monthly Cross-sectional Rank IC
# ============================================================

def compute_monthly_rank_ic(
    panel,
    core_factors,
):

    factor_names = (
        core_factors[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    rows = []

    grouped = panel.groupby(
        [
            "analysis_date",
            "window",
        ],
        sort=True,
    )

    total_jobs = grouped.ngroups

    print()
    print(
        f"Cross-sectional Date x Window jobs: "
        f"{total_jobs}"
    )

    for (
        job_no,
        (
            (
                analysis_date,
                window,
            ),
            group,
        ),
    ) in enumerate(
        grouped,
        start=1,
    ):

        if (
            job_no == 1
            or
            job_no % 25 == 0
            or
            job_no == total_jobs
        ):

            print(
                f"[{job_no:3d}/{total_jobs}] "
                f"W={window} | "
                f"{analysis_date.date()}"
            )

        factor_universe_n = len(
            group
        )

        formal_valid = (
            group[
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

            target_numeric = pd.to_numeric(
                group[
                    target_name
                ],
                errors="coerce",
            )

            target_valid = (

                formal_valid

                &

                target_numeric.notna()

                &

                np.isfinite(
                    target_numeric
                )
            )

            label_valid_n = int(
                target_valid.sum()
            )

            for (
                factor_order,
                factor_name,
            ) in enumerate(
                factor_names,
                start=1,
            ):

                factor_numeric = (
                    pd.to_numeric(
                        group[
                            factor_name
                        ],
                        errors="coerce",
                    )
                )

                factor_nonmissing = (

                    factor_numeric.notna()

                    &

                    np.isfinite(
                        factor_numeric
                    )
                )

                factor_nonmissing_n = int(
                    factor_nonmissing.sum()
                )

                pair_mask = (

                    target_valid

                    &

                    factor_nonmissing
                )

                result = (
                    compute_spearman_rank_ic(

                        factor_numeric.loc[
                            pair_mask
                        ],

                        target_numeric.loc[
                            pair_mask
                        ],
                    )
                )

                pair_n = int(
                    result[
                        "pair_n"
                    ]
                )

                rows.append(
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

                        "factor_universe_n":
                            int(
                                factor_universe_n
                            ),

                        "label_valid_n":
                            label_valid_n,

                        "factor_nonmissing_n":
                            factor_nonmissing_n,

                        "pair_n":
                            pair_n,

                        "label_valid_share":
                            (
                                label_valid_n
                                /
                                factor_universe_n
                                if
                                factor_universe_n
                                >
                                0
                                else
                                np.nan
                            ),

                        "factor_nonmissing_share":
                            (
                                factor_nonmissing_n
                                /
                                factor_universe_n
                                if
                                factor_universe_n
                                >
                                0
                                else
                                np.nan
                            ),

                        "pair_share_of_factor_universe":
                            (
                                pair_n
                                /
                                factor_universe_n
                                if
                                factor_universe_n
                                >
                                0
                                else
                                np.nan
                            ),

                        "pair_share_of_label_valid":
                            (
                                pair_n
                                /
                                label_valid_n
                                if
                                label_valid_n
                                >
                                0
                                else
                                np.nan
                            ),

                        "factor_unique_n":
                            result[
                                "factor_unique_n"
                            ],

                        "target_unique_n":
                            result[
                                "target_unique_n"
                            ],

                        "rank_ic":
                            result[
                                "rank_ic"
                            ],

                        "ic_status":
                            result[
                                "status"
                            ],
                    }
                )

    monthly_ic = pd.DataFrame(
        rows
    )

    monthly_ic = (
        monthly_ic.sort_values(
            [
                "target_role",
                "factor_order",
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Strict uniqueness
    # --------------------------------------------------------

    unique_key = [

        "analysis_date",

        "window",

        "factor_name",

        "target_name",
    ]

    if (
        monthly_ic[
            unique_key
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate monthly Rank-IC rows."
        )

    valid_ic = (
        monthly_ic[
            "rank_ic"
        ]
        .dropna()
    )

    if (
        (
            valid_ic.abs()
            >
            1.0
            +
            1e-10
        )
        .any()
    ):

        raise RuntimeError(
            "Rank IC outside [-1, 1]."
        )

    return monthly_ic


# ============================================================
# 10. HAC / Newey-West Mean-IC Inference
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
            ** (
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


def mean_ic_inference(
    values,
):

    x = np.asarray(
        values,
        dtype=np.float64,
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
        == 0
    ):

        return {

            "naive_se":
                np.nan,

            "naive_t_stat":
                np.nan,

            "hac_lag":
                np.nan,

            "hac_se":
                np.nan,

            "hac_t_stat":
                np.nan,
        }

    mean_x = float(
        x.mean()
    )

    # --------------------------------------------------------
    # Ordinary time-series t-stat
    # --------------------------------------------------------

    if (
        n
        >=
        2
    ):

        sd = float(
            x.std(
                ddof=1
            )
        )

        if (
            np.isfinite(
                sd
            )
            and
            sd
            >
            0
        ):

            naive_se = (
                sd
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

    else:

        naive_se = np.nan

        naive_t = np.nan

    # --------------------------------------------------------
    # Newey-West / HAC for mean(IC)
    # --------------------------------------------------------

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

            "naive_t_stat":
                naive_t,

            "hac_lag":
                0,

            "hac_se":
                naive_se,

            "hac_t_stat":
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

    long_run_variance = gamma0

    for ell in range(
        1,
        lag + 1,
    ):

        gamma_l = float(
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

        long_run_variance += (
            2.0
            *
            weight
            *
            gamma_l
        )

    long_run_variance = max(
        long_run_variance,
        0.0,
    )

    variance_of_mean = (
        long_run_variance
        /
        n
    )

    if (
        variance_of_mean
        >
        0
    ):

        hac_se = float(
            np.sqrt(
                variance_of_mean
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

        "naive_t_stat":
            naive_t,

        "hac_lag":
            int(
                lag
            ),

        "hac_se":
            hac_se,

        "hac_t_stat":
            hac_t,
    }


# ============================================================
# 11. Rank-IC Time-series Summary
# ============================================================

def summarize_rank_ic(
    monthly_ic,
    core_factors,
):

    priority_map = dict(
        zip(
            core_factors[
                "factor_name"
            ].astype(str),

            core_factors[
                "priority"
            ],
        )
    )

    rows = []

    group_columns = [

        "factor_name",

        "target_name",

        "target_role",

        "window",
    ]

    for keys, group in (
        monthly_ic.groupby(
            group_columns,
            sort=False,
        )
    ):

        (
            factor_name,
            target_name,
            target_role,
            window,
        ) = keys

        passed = (
            group[
                group[
                    "ic_status"
                ]
                ==
                "PASS"
            ]
            .copy()
            .sort_values(
                "analysis_date"
            )
        )

        ic = (
            passed[
                "rank_ic"
            ]
            .dropna()
            .to_numpy(
                dtype=np.float64
            )
        )

        n = len(
            ic
        )

        if (
            n
            >
            0
        ):

            mean_ic = float(
                np.mean(
                    ic
                )
            )

            median_ic = float(
                np.median(
                    ic
                )
            )

            std_ic = (
                float(
                    np.std(
                        ic,
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

            positive_share = float(
                np.mean(
                    ic
                    >
                    0
                )
            )

            negative_share = float(
                np.mean(
                    ic
                    <
                    0
                )
            )

            zero_share = float(
                np.mean(
                    ic
                    ==
                    0
                )
            )

            quantiles = np.quantile(
                ic,
                [
                    0.05,
                    0.25,
                    0.75,
                    0.95,
                ],
            )

            first_ic_date = (
                passed[
                    "analysis_date"
                ]
                .min()
            )

            last_ic_date = (
                passed[
                    "analysis_date"
                ]
                .max()
            )

        else:

            mean_ic = np.nan

            median_ic = np.nan

            std_ic = np.nan

            positive_share = np.nan

            negative_share = np.nan

            zero_share = np.nan

            quantiles = [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]

            first_ic_date = pd.NaT

            last_ic_date = pd.NaT

        if (
            n
            >=
            2
            and
            np.isfinite(
                std_ic
            )
            and
            std_ic
            >
            0
        ):

            icir_monthly = (
                mean_ic
                /
                std_ic
            )

            icir_annualized = (
                np.sqrt(
                    12.0
                )
                *
                icir_monthly
            )

        else:

            icir_monthly = np.nan

            icir_annualized = np.nan

        inference = (
            mean_ic_inference(
                ic
            )
        )

        # ----------------------------------------------------
        # Adjacent-month autocorrelation of IC series
        # ----------------------------------------------------

        if (
            n
            >=
            3
        ):

            ic_lag1_corr = float(
                pd.Series(
                    ic
                )
                .autocorr(
                    lag=1
                )
            )

        else:

            ic_lag1_corr = np.nan

        rows.append(
            {

                "factor_priority":
                    priority_map[
                        factor_name
                    ],

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

                "ic_month_count":
                    int(
                        n
                    ),

                "first_ic_date":
                    first_ic_date,

                "last_ic_date":
                    last_ic_date,

                "mean_rank_ic":
                    mean_ic,

                "median_rank_ic":
                    median_ic,

                "std_rank_ic":
                    std_ic,

                "icir_monthly":
                    icir_monthly,

                "icir_annualized":
                    icir_annualized,

                "positive_ic_share":
                    positive_share,

                "negative_ic_share":
                    negative_share,

                "zero_ic_share":
                    zero_share,

                "p05_rank_ic":
                    quantiles[
                        0
                    ],

                "p25_rank_ic":
                    quantiles[
                        1
                    ],

                "p75_rank_ic":
                    quantiles[
                        2
                    ],

                "p95_rank_ic":
                    quantiles[
                        3
                    ],

                "lag1_ic_autocorrelation":
                    ic_lag1_corr,

                "mean_pair_n":
                    float(
                        passed[
                            "pair_n"
                        ]
                        .mean()
                    )
                    if
                    len(
                        passed
                    )
                    >
                    0
                    else
                    np.nan,

                "min_pair_n":
                    int(
                        passed[
                            "pair_n"
                        ]
                        .min()
                    )
                    if
                    len(
                        passed
                    )
                    >
                    0
                    else
                    0,

                "mean_pair_share_of_label_valid":
                    float(
                        passed[
                            "pair_share_of_label_valid"
                        ]
                        .mean()
                    )
                    if
                    len(
                        passed
                    )
                    >
                    0
                    else
                    np.nan,

                "naive_mean_ic_se":
                    inference[
                        "naive_se"
                    ],

                "naive_mean_ic_t_stat":
                    inference[
                        "naive_t_stat"
                    ],

                "hac_lag":
                    inference[
                        "hac_lag"
                    ],

                "hac_mean_ic_se":
                    inference[
                        "hac_se"
                    ],

                "hac_mean_ic_t_stat":
                    inference[
                        "hac_t_stat"
                    ],

                "outcome_based_sign_flip":
                    False,

                "outcome_based_window_selection":
                    False,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary = (
        summary.sort_values(
            [
                "target_role",
                "factor_priority",
                "window",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# ============================================================
# 12. Cross-window Consistency
# ============================================================

def build_cross_window_consistency(
    summary,
):

    rows = []

    group_columns = [

        "factor_priority",

        "factor_name",

        "target_name",

        "target_role",
    ]

    for keys, group in (
        summary.groupby(
            group_columns,
            sort=False,
        )
    ):

        (
            factor_priority,
            factor_name,
            target_name,
            target_role,
        ) = keys

        by_window = (
            group.set_index(
                "window"
            )
        )

        mean_ic = {}

        median_ic = {}

        month_count = {}

        for window in EXPECTED_WINDOWS:

            if (
                window
                in
                by_window.index
            ):

                row = (
                    by_window.loc[
                        window
                    ]
                )

                mean_ic[
                    window
                ] = (
                    row[
                        "mean_rank_ic"
                    ]
                )

                median_ic[
                    window
                ] = (
                    row[
                        "median_rank_ic"
                    ]
                )

                month_count[
                    window
                ] = (
                    row[
                        "ic_month_count"
                    ]
                )

            else:

                mean_ic[
                    window
                ] = np.nan

                median_ic[
                    window
                ] = np.nan

                month_count[
                    window
                ] = 0

        values = np.array(
            [
                mean_ic[
                    w
                ]
                for w in EXPECTED_WINDOWS
            ],
            dtype=float,
        )

        all_windows_available = bool(
            np.isfinite(
                values
            )
            .all()
        )

        if all_windows_available:

            signs = np.sign(
                values
            )

            nonzero_signs = signs[
                signs
                !=
                0
            ]

            if (
                len(
                    nonzero_signs
                )
                ==
                len(
                    EXPECTED_WINDOWS
                )
            ):

                same_sign_all_windows = bool(
                    np.all(
                        nonzero_signs
                        ==
                        nonzero_signs[
                            0
                        ]
                    )
                )

            else:

                same_sign_all_windows = False

            mean_ic_range = float(
                values.max()
                -
                values.min()
            )

        else:

            same_sign_all_windows = False

            mean_ic_range = np.nan

        rows.append(
            {

                "factor_priority":
                    factor_priority,

                "factor_name":
                    factor_name,

                "target_name":
                    target_name,

                "target_role":
                    target_role,

                "mean_ic_w60":
                    mean_ic[
                        60
                    ],

                "mean_ic_w120":
                    mean_ic[
                        120
                    ],

                "mean_ic_w252":
                    mean_ic[
                        252
                    ],

                "median_ic_w60":
                    median_ic[
                        60
                    ],

                "median_ic_w120":
                    median_ic[
                        120
                    ],

                "median_ic_w252":
                    median_ic[
                        252
                    ],

                "ic_months_w60":
                    month_count[
                        60
                    ],

                "ic_months_w120":
                    month_count[
                        120
                    ],

                "ic_months_w252":
                    month_count[
                        252
                    ],

                "all_windows_available":
                    all_windows_available,

                "same_mean_ic_sign_all_windows":
                    same_sign_all_windows,

                "mean_ic_cross_window_range":
                    mean_ic_range,

                "factor_orientation":
                    "ORIGINAL",

                "window_selected_from_outcome":
                    False,
            }
        )

    result = pd.DataFrame(
        rows
    )

    return (
        result.sort_values(
            [
                "target_role",
                "factor_priority",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 13. Formal QA
# ============================================================

def formal_qa(
    panel,
    monthly_ic,
    summary,
    core_factors,
):

    qa = {}

    # --------------------------------------------------------
    # Analysis panel
    # --------------------------------------------------------

    qa[
        "analysis_panel_row_count"
    ] = int(
        len(
            panel
        )
    )

    qa[
        "analysis_panel_key_unique"
    ] = bool(

        not panel[
            KEY_COLUMNS
        ]
        .duplicated()
        .any()
    )

    # --------------------------------------------------------
    # Frozen factor count
    # --------------------------------------------------------

    qa[
        "core_factor_count"
    ] = int(
        len(
            core_factors
        )
    )

    qa[
        "core_factor_count_ok"
    ] = bool(
        len(
            core_factors
        )
        ==
        EXPECTED_CORE_FACTOR_COUNT
    )

    # --------------------------------------------------------
    # No sign flipping
    # --------------------------------------------------------

    qa[
        "all_factor_orientation_original"
    ] = bool(
        (
            monthly_ic[
                "factor_orientation"
            ]
            ==
            "ORIGINAL"
        )
        .all()
    )

    # --------------------------------------------------------
    # IC ranges
    # --------------------------------------------------------

    ic_nonmissing = (
        monthly_ic[
            "rank_ic"
        ]
        .dropna()
    )

    qa[
        "all_rank_ic_in_unit_interval"
    ] = bool(

        (
            ic_nonmissing
            >=
            -1.0
            -
            1e-10
        )
        .all()

        and

        (
            ic_nonmissing
            <=
            1.0
            +
            1e-10
        )
        .all()
    )

    # --------------------------------------------------------
    # Monthly IC uniqueness
    # --------------------------------------------------------

    monthly_key = [

        "analysis_date",

        "window",

        "factor_name",

        "target_name",
    ]

    qa[
        "monthly_ic_key_unique"
    ] = bool(

        not monthly_ic[
            monthly_key
        ]
        .duplicated()
        .any()
    )

    qa[
        "monthly_ic_row_count"
    ] = int(
        len(
            monthly_ic
        )
    )

    qa[
        "monthly_ic_pass_count"
    ] = int(
        (
            monthly_ic[
                "ic_status"
            ]
            ==
            "PASS"
        )
        .sum()
    )

    qa[
        "monthly_ic_nonpass_count"
    ] = int(
        (
            monthly_ic[
                "ic_status"
            ]
            !=
            "PASS"
        )
        .sum()
    )

    qa[
        "monthly_ic_status_counts"
    ] = (
        monthly_ic[
            "ic_status"
        ]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # Window set
    # --------------------------------------------------------

    qa[
        "observed_windows"
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
            "observed_windows"
        ]
        ==
        EXPECTED_WINDOWS
    )

    # --------------------------------------------------------
    # Formal pass
    # --------------------------------------------------------

    required_boolean_checks = [

        "analysis_panel_key_unique",

        "core_factor_count_ok",

        "all_factor_orientation_original",

        "all_rank_ic_in_unit_interval",

        "monthly_ic_key_unique",

        "window_set_ok",
    ]

    qa[
        "all_formal_qa_pass"
    ] = bool(
        all(
            qa[
                key
            ]
            for key in required_boolean_checks
        )
    )

    return qa


# ============================================================
# 14. Step-3 Design Hash
# ============================================================

def build_step3_design(
    factor_design_hash,
    label_design_hash,
    core_factors,
):

    factor_names = (
        core_factors[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    payload = {

        "research_day":
            6,

        "step":
            "Step3_Initial_Alpha_Screening_Rank_IC",

        "parent_factor_design_hash":
            factor_design_hash,

        "parent_label_design_hash":
            label_design_hash,

        "factor_set":
            factor_names,

        "factor_orientation":
            "ORIGINAL",

        "outcome_based_factor_sign_flip":
            False,

        "outcome_based_window_selection":
            False,

        "windows":
            EXPECTED_WINDOWS,

        "targets":
            TARGET_SPECS,

        "primary_alpha_target":
            "future_excess_return",

        "rank_ic_definition":
            (
                "Monthly cross-sectional Spearman "
                "correlation between contemporaneous "
                "factor and strictly forward return."
            ),

        "cross_section_rule":
            (
                "For each analysis_date x window, "
                "use pairwise complete observations "
                "among the frozen contemporaneous factor "
                "universe and valid Step-2 forward labels."
            ),

        "minimum_cross_section_observations":
            MIN_CROSS_SECTION_OBS,

        "winsorization":
            False,

        "neutralization":
            False,

        "portfolio_sorting":
            False,

        "hac_inference":
            USE_HAC,

        "hac_lag_rule":
            (
                "floor(4 * (T/100)^(2/9))"
                if
                USE_HAC
                else
                None
            ),

        "important_interpretation_rule":
            (
                "Step 3 is descriptive/preliminary Alpha "
                "screening. IC results do not change factor "
                "definitions, factor signs, or the frozen "
                "window design."
            ),
    }

    design_hash = (
        canonical_json_hash(
            payload
        )
    )

    return {
        **payload,

        "step3_design_hash":
            design_hash,
    }


# ============================================================
# 15. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 6 - Step 3"
    )

    print(
        "Initial Alpha Screening - Rank IC"
    )

    print("=" * 80)

    # ========================================================
    # 15.1 Validate upstream frozen designs
    # ========================================================

    print()
    print(
        "[1] Validate frozen factor and label designs"
    )

    (
        factor_design,
        factor_design_hash,
        label_design,
        label_design_hash,
        step2_metadata,
    ) = validate_designs()

    print(
        "Factor design hash:"
    )

    print(
        factor_design_hash
    )

    print()
    print(
        "Forward-label design hash:"
    )

    print(
        label_design_hash
    )

    # ========================================================
    # 15.2 Frozen factors
    # ========================================================

    print()
    print(
        "[2] Load frozen core factors"
    )

    core_factors = (
        load_core_factors()
    )

    print(
        core_factors[
            [
                "priority",
                "factor_name",
            ]
        ]
        .to_string(
            index=False
        )
    )

    save_csv_atomic(
        core_factors,
        CORE_FACTOR_PATH,
    )

    target_inventory = pd.DataFrame(
        TARGET_SPECS
    )

    save_csv_atomic(
        target_inventory,
        TARGET_PATH,
    )

    # ========================================================
    # 15.3 Freeze Step-3 design
    # ========================================================

    step3_design = (
        build_step3_design(

            factor_design_hash=
                factor_design_hash,

            label_design_hash=
                label_design_hash,

            core_factors=
                core_factors,
        )
    )

    step3_design_hash = (
        step3_design[
            "step3_design_hash"
        ]
    )

    print()
    print(
        "Step-3 design hash:"
    )

    print(
        step3_design_hash
    )

    # ========================================================
    # 15.4 Merge factor and outcome panels
    # ========================================================

    print()
    print(
        "[3] Build frozen Factor_t -> Return_{t+1} panel"
    )

    analysis_panel = (
        load_analysis_panel(
            core_factors
        )
    )

    print(
        f"Merged analysis rows: "
        f"{len(analysis_panel):,}"
    )

    # ========================================================
    # 15.5 Monthly Rank IC
    # ========================================================

    print()
    print(
        "[4] Compute monthly cross-sectional Rank IC"
    )

    monthly_ic = (
        compute_monthly_rank_ic(

            panel=
                analysis_panel,

            core_factors=
                core_factors,
        )
    )

    save_csv_atomic(
        monthly_ic,
        MONTHLY_IC_PATH,
    )

    # ========================================================
    # 15.6 Time-series summaries
    # ========================================================

    print()
    print(
        "[5] Summarize Rank-IC time series"
    )

    summary = (
        summarize_rank_ic(

            monthly_ic=
                monthly_ic,

            core_factors=
                core_factors,
        )
    )

    save_csv_atomic(
        summary,
        SUMMARY_PATH,
    )

    # ========================================================
    # 15.7 Cross-window consistency
    # ========================================================

    print()
    print(
        "[6] Build cross-window consistency table"
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
    # 15.8 Formal QA
    # ========================================================

    qa = formal_qa(

        panel=
            analysis_panel,

        monthly_ic=
            monthly_ic,

        summary=
            summary,

        core_factors=
            core_factors,
    )

    if (
        not qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            "Step-3 formal QA failed.\n"
            f"{qa}"
        )

    # ========================================================
    # 15.9 Metadata
    # ========================================================

    metadata = {

        **step3_design,

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "input_files": {

            "frozen_core_factor_set":
                str(
                    FROZEN_CORE_PATH
                ),

            "factor_panel":
                str(
                    FACTOR_PANEL_PATH
                ),

            "forward_label_panel":
                str(
                    FORWARD_LABEL_PANEL_PATH
                ),

            "step2_label_design":
                str(
                    STEP2_LABEL_DESIGN_PATH
                ),
        },

        "input_sha256": {

            "frozen_core_factor_set":
                sha256_file(
                    FROZEN_CORE_PATH
                ),

            "factor_panel":
                sha256_file(
                    FACTOR_PANEL_PATH
                ),

            "forward_label_panel":
                sha256_file(
                    FORWARD_LABEL_PANEL_PATH
                ),

            "step2_label_design":
                sha256_file(
                    STEP2_LABEL_DESIGN_PATH
                ),
        },

        "formal_qa":
            qa,

        "outputs": {

            "monthly_rank_ic":
                str(
                    MONTHLY_IC_PATH
                ),

            "rank_ic_summary":
                str(
                    SUMMARY_PATH
                ),

            "rank_ic_cross_window_consistency":
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
    # 15.10 Console result
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Step-3 Formal QA"
    )

    print("=" * 80)

    print(
        f"Analysis rows: "
        f"{qa['analysis_panel_row_count']:,}"
    )

    print(
        f"Monthly IC rows: "
        f"{qa['monthly_ic_row_count']:,}"
    )

    print(
        f"Monthly IC PASS: "
        f"{qa['monthly_ic_pass_count']:,}"
    )

    print(
        f"Monthly IC non-PASS: "
        f"{qa['monthly_ic_nonpass_count']:,}"
    )

    print()

    print(
        "IC status counts:"
    )

    print(
        qa[
            "monthly_ic_status_counts"
        ]
    )

    print()

    print(
        "Formal QA:"
    )

    print(
        qa[
            "all_formal_qa_pass"
        ]
    )

    print()
    print("=" * 80)

    print(
        "Rank-IC Summary"
    )

    print("=" * 80)

    display_columns = [

        "factor_name",

        "target_name",

        "window",

        "ic_month_count",

        "mean_rank_ic",

        "median_rank_ic",

        "icir_annualized",

        "positive_ic_share",

        "hac_mean_ic_t_stat",
    ]

    print(
        summary[
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
        "Day 6 Step 3 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()