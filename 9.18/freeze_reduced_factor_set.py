from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib
import json
import os

import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)


# ------------------------------------------------------------
# Day 5 outputs
#
# Research Day 5 is stored under M1_day4
# ------------------------------------------------------------

DAY5_ROOT = (
    OUTPUT_ROOT
    / "M1_day5"
)


# ------------------------------------------------------------
# Day 5 Step 1:
# original frozen network-factor design
# ------------------------------------------------------------

DAY5_STEP1_DIR = (
    DAY5_ROOT
    / "01_step1_factor_design"
)

PREVIOUS_FACTOR_CONFIG_PATH = (
    DAY5_STEP1_DIR
    / "factor_design_config.json"
)


# ------------------------------------------------------------
# Day 5 Step 5:
# final stock-level network factor panel
# ------------------------------------------------------------

DAY5_STEP5_DIR = (
    DAY5_ROOT
    / "05_step5_community_features"
)

FACTOR_PANEL_PATH = (
    DAY5_STEP5_DIR
    / "community_bridge_feature_panel.parquet"
)

STEP5_METADATA_PATH = (
    DAY5_STEP5_DIR
    / "step5_community_bridge_metadata.json"
)


# ------------------------------------------------------------
# Day 5 Step 6:
# pre-outcome factor QA and screening
# ------------------------------------------------------------

DAY5_STEP6_DIR = (
    DAY5_ROOT
    / "06_step6_factor_qa_screening"
)

SCREENING_PATH = (
    DAY5_STEP6_DIR
    / "factor_screening_decisions.csv"
)

CORE_SOURCE_PATH = (
    DAY5_STEP6_DIR
    / "recommended_preoutcome_core_set.csv"
)

QA_SUMMARY_PATH = (
    DAY5_STEP6_DIR
    / "factor_qa_summary.csv"
)

REDUNDANCY_PATH = (
    DAY5_STEP6_DIR
    / "factor_redundancy_pairs.csv"
)

TEMPORAL_PATH = (
    DAY5_STEP6_DIR
    / "factor_temporal_stability.csv"
)

STEP6_METADATA_PATH = (
    DAY5_STEP6_DIR
    / "step6_factor_qa_screening_metadata.json"
)


# ------------------------------------------------------------
# Day 6
#
# Research Day 6 is stored under M1_day5
# ------------------------------------------------------------

DAY6_ROOT = (
    OUTPUT_ROOT
    / "M1_day6"
)

OUTPUT_DIR = (
    DAY6_ROOT
    / "01_step1_freeze_reduced_factor_set"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

FROZEN_CORE_PATH = (
    OUTPUT_DIR
    / "frozen_core_factor_set.csv"
)

FROZEN_AUX_PATH = (
    OUTPUT_DIR
    / "frozen_auxiliary_factor_set.csv"
)

FROZEN_EXCLUDED_PATH = (
    OUTPUT_DIR
    / "frozen_excluded_factor_set.csv"
)

FREEZE_MANIFEST_PATH = (
    OUTPUT_DIR
    / "factor_freeze_manifest.csv"
)

DAY6_DESIGN_PATH = (
    OUTPUT_DIR
    / "day6_factor_design.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "day6_step1_freeze_summary.md"
)


# ============================================================
# 1. Expected Research Design
# ============================================================

# ------------------------------------------------------------
# These nine factors come from yesterday's actual
# pre-outcome Step-6 result.
#
# This list is NOT used to re-select factors.
#
# It is used only as a strict consistency check:
# if factor_screening_decisions.csv gives a different set,
# the program stops instead of silently changing the design.
# ------------------------------------------------------------

EXPECTED_CORE_FACTORS = {

    "residual_degree_percentile",

    "delta_degree_percentile",

    "cross_industry_degree_percentile",

    "cross_industry_degree_ratio",

    "delta_residual_degree_percentile_1m",

    "delta_cross_industry_degree_percentile_1m",

    "neighbor_jaccard_1m",

    "neighbor_retention_1m",

    "outside_community_degree_percentile",
}


EXPECTED_CORE_COUNT = 9


# ------------------------------------------------------------
# Formal rolling windows
# ------------------------------------------------------------

WINDOWS = [
    60,
    120,
    252,
]


# ------------------------------------------------------------
# Main network definition
# ------------------------------------------------------------

PRIMARY_NETWORK = (
    "M1_RESIDUAL_POSITIVE"
)

BENCHMARK_NETWORK = (
    "B0_RAW_POSITIVE"
)

PRIMARY_DENSITY = 0.01

MAIN_INDUSTRY_LEVEL = (
    "industry_id1"
)


# ============================================================
# 2. Utilities
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
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


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


# ============================================================
# 3. Validate Previous Frozen Design
# ============================================================

def validate_previous_design():

    previous = load_json(
        PREVIOUS_FACTOR_CONFIG_PATH
    )

    if (
        previous.get(
            "design_status"
        )
        !=
        "FROZEN"
    ):

        raise RuntimeError(
            "Previous Day-5 factor design "
            "is not FROZEN."
        )

    previous_hash = (
        previous.get(
            "design_hash"
        )
    )

    if not previous_hash:

        raise RuntimeError(
            "Previous factor design "
            "does not contain design_hash."
        )

    network_design = (
        previous.get(
            "network_design",
            {},
        )
    )

    if (
        network_design.get(
            "primary_network"
        )
        !=
        PRIMARY_NETWORK
    ):

        raise RuntimeError(
            "Primary network mismatch."
        )

    if (
        network_design.get(
            "benchmark_network"
        )
        !=
        BENCHMARK_NETWORK
    ):

        raise RuntimeError(
            "Benchmark network mismatch."
        )

    previous_windows = sorted(
        network_design.get(
            "windows",
            []
        )
    )

    if (
        previous_windows
        !=
        sorted(
            WINDOWS
        )
    ):

        raise RuntimeError(
            "Window definition mismatch."
        )

    if (
        float(
            network_design.get(
                "primary_density"
            )
        )
        !=
        PRIMARY_DENSITY
    ):

        raise RuntimeError(
            "Primary density mismatch."
        )

    if (
        network_design.get(
            "main_industry_level"
        )
        !=
        MAIN_INDUSTRY_LEVEL
    ):

        raise RuntimeError(
            "Industry definition mismatch."
        )

    return (
        previous,
        previous_hash,
    )


# ============================================================
# 4. Validate Step 5 Factor Panel
# ============================================================

def validate_factor_panel(
    previous_design_hash,
):

    if not FACTOR_PANEL_PATH.exists():

        raise FileNotFoundError(
            "Final stock-level network factor panel "
            "does not exist:\n"
            f"{FACTOR_PANEL_PATH}"
        )

    metadata = load_json(
        STEP5_METADATA_PATH
    )

    if (
        metadata.get(
            "design_hash"
        )
        !=
        previous_design_hash
    ):

        raise RuntimeError(
            "Step-5 factor panel belongs to a "
            "different Day-5 factor design."
        )

    formal_qa = (
        metadata.get(
            "formal_qa",
            {}
        )
    )

    if not formal_qa.get(
        "all_jobs_pass",
        False,
    ):

        raise RuntimeError(
            "Step-5 factor panel did not "
            "pass all formal QA."
        )

    return metadata


# ============================================================
# 5. Validate Step 6 Was Truly Pre-outcome
# ============================================================

def validate_preoutcome_screening(
    previous_design_hash,
):

    metadata = load_json(
        STEP6_METADATA_PATH
    )

    if (
        metadata.get(
            "design_hash"
        )
        !=
        previous_design_hash
    ):

        raise RuntimeError(
            "Step-6 screening belongs to "
            "a different factor design."
        )

    if (
        metadata.get(
            "screening_is_preoutcome"
        )
        is not True
    ):

        raise RuntimeError(
            "Step-6 screening is not marked "
            "as pre-outcome."
        )

    if (
        metadata.get(
            "future_return_used"
        )
        is not False
    ):

        raise RuntimeError(
            "Future return was used in Step-6. "
            "Cannot treat screening as "
            "pre-outcome freeze."
        )

    if (
        metadata.get(
            "future_risk_used"
        )
        is not False
    ):

        raise RuntimeError(
            "Future risk was used in Step-6. "
            "Cannot treat screening as "
            "pre-outcome freeze."
        )

    return metadata


# ============================================================
# 6. Load Screening Decisions
# ============================================================

def load_screening_decisions():

    if not SCREENING_PATH.exists():

        raise FileNotFoundError(
            f"Missing screening file:\n{SCREENING_PATH}"
        )

    df = pd.read_csv(
        SCREENING_PATH
    )

    required_columns = [

        "factor_name",

        "family",

        "tier",

        "priority",

        "screening_status",

        "screening_reason",

        "economic_meaning",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "factor_screening_decisions.csv "
            f"is missing columns:\n{missing}"
        )

    if (
        df[
            "factor_name"
        ]
        .duplicated()
        .any()
    ):

        duplicate_factors = (
            df.loc[
                df[
                    "factor_name"
                ]
                .duplicated(
                    keep=False
                ),
                "factor_name"
            ]
            .tolist()
        )

        raise RuntimeError(
            "Duplicate factor_name found:\n"
            f"{duplicate_factors}"
        )

    return (
        df.sort_values(
            "priority"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 7. Split Core / Auxiliary / Excluded
# ============================================================

def split_factor_sets(
    screening,
):

    core = (
        screening[
            screening[
                "screening_status"
            ]
            ==
            "KEEP_CORE"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    auxiliary = (
        screening[
            screening[
                "screening_status"
            ]
            ==
            "KEEP_AUXILIARY"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    excluded = (
        screening[
            ~screening[
                "screening_status"
            ]
            .isin(
                [
                    "KEEP_CORE",
                    "KEEP_AUXILIARY",
                ]
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        core,
        auxiliary,
        excluded,
    )


# ============================================================
# 8. Strict Core-set Validation
# ============================================================

def validate_core_set(
    core,
):

    actual_core = set(
        core[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    if (
        len(
            actual_core
        )
        !=
        EXPECTED_CORE_COUNT
    ):

        raise RuntimeError(
            "Unexpected core factor count.\n"
            f"Expected: {EXPECTED_CORE_COUNT}\n"
            f"Actual:   {len(actual_core)}"
        )

    missing_expected = (
        EXPECTED_CORE_FACTORS
        -
        actual_core
    )

    unexpected_core = (
        actual_core
        -
        EXPECTED_CORE_FACTORS
    )

    if (
        missing_expected
        or
        unexpected_core
    ):

        raise RuntimeError(
            "Core factor set differs from "
            "the Day-5 documented result.\n\n"
            f"Missing expected factors:\n"
            f"{sorted(missing_expected)}\n\n"
            f"Unexpected KEEP_CORE factors:\n"
            f"{sorted(unexpected_core)}"
        )


# ============================================================
# 9. Cross-check recommended_preoutcome_core_set.csv
# ============================================================

def validate_recommended_core_file(
    core,
):

    if not CORE_SOURCE_PATH.exists():

        raise FileNotFoundError(
            "Missing recommended core set:\n"
            f"{CORE_SOURCE_PATH}"
        )

    recommended = pd.read_csv(
        CORE_SOURCE_PATH
    )

    if (
        "factor_name"
        not in recommended.columns
    ):

        raise ValueError(
            "recommended_preoutcome_core_set.csv "
            "has no factor_name column."
        )

    core_set = set(
        core[
            "factor_name"
        ]
        .astype(str)
    )

    recommended_set = set(
        recommended[
            "factor_name"
        ]
        .astype(str)
    )

    if (
        core_set
        !=
        recommended_set
    ):

        raise RuntimeError(
            "KEEP_CORE set in screening decisions "
            "does not match "
            "recommended_preoutcome_core_set.csv."
        )

    return recommended


# ============================================================
# 10. Add Freeze-specific Columns
# ============================================================

def annotate_frozen_set(
    df,
    freeze_class,
):

    result = (
        df.copy()
    )

    result.insert(
        0,
        "freeze_class",
        freeze_class,
    )

    result.insert(
        1,
        "freeze_status",
        "FROZEN",
    )

    result[
        "allowed_for_day6_primary_validation"
    ] = (
        freeze_class
        ==
        "CORE"
    )

    result[
        "allowed_for_day6_robustness"
    ] = (
        freeze_class
        in {
            "CORE",
            "AUXILIARY",
        }
    )

    result[
        "selection_used_future_outcome"
    ] = False

    return result


# ============================================================
# 11. Build Freeze Manifest
# ============================================================

def build_freeze_manifest(
    core,
    auxiliary,
    excluded,
):

    manifest = pd.concat(
        [

            annotate_frozen_set(
                core,
                "CORE",
            ),

            annotate_frozen_set(
                auxiliary,
                "AUXILIARY",
            ),

            annotate_frozen_set(
                excluded,
                "EXCLUDED",
            ),

        ],
        ignore_index=True,
    )

    manifest = (
        manifest.sort_values(
            "priority"
        )
        .reset_index(
            drop=True
        )
    )

    return manifest


# ============================================================
# 12. Build Day-6 Factor Design
# ============================================================

def build_day6_design(
    previous_design_hash,
    screening_metadata,
    core,
    auxiliary,
    excluded,
):

    core_factors = (
        core[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    auxiliary_factors = (
        auxiliary[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    excluded_factors = (
        excluded[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # This object contains ONLY deterministic design elements.
    #
    # It is used to create the design hash.
    #
    # Do not place runtime timestamps inside it before hashing.
    # --------------------------------------------------------

    frozen_payload = {

        "research_day":
            6,

        "step":
            "Step1_Freeze_Reduced_Factor_Set",

        "design_status":
            "FROZEN",

        "parent_day5_design_hash":
            previous_design_hash,

        "factor_source": {

            "screening_file":
                str(
                    SCREENING_PATH
                ),

            "factor_panel":
                str(
                    FACTOR_PANEL_PATH
                ),

            "screening_is_preoutcome":
                True,

            "future_return_used":
                False,

            "future_risk_used":
                False,
        },

        "network_design": {

            "primary_network":
                PRIMARY_NETWORK,

            "benchmark_network":
                BENCHMARK_NETWORK,

            "primary_density":
                PRIMARY_DENSITY,

            "windows":
                WINDOWS,

            "main_industry_level":
                MAIN_INDUSTRY_LEVEL,
        },

        "factor_sets": {

            "core_factor_count":
                len(
                    core_factors
                ),

            "core_factors":
                core_factors,

            "auxiliary_factor_count":
                len(
                    auxiliary_factors
                ),

            "auxiliary_factors":
                auxiliary_factors,

            "excluded_factor_count":
                len(
                    excluded_factors
                ),

            "excluded_factors":
                excluded_factors,
        },

        "day6_rules": {

            "primary_validation_factor_set":
                "CORE_ONLY",

            "auxiliary_use":
                (
                    "Robustness and interpretation only; "
                    "must not replace a core factor "
                    "because of observed future outcomes."
                ),

            "excluded_factor_use":
                (
                    "Excluded from primary Day-6 "
                    "Alpha/Risk validation."
                ),

            "factor_definition_change_after_freeze":
                False,

            "factor_selection_change_after_future_labels":
                False,

            "future_label_start_rule":
                (
                    "First trading day strictly after "
                    "analysis_date."
                ),

            "future_label_end_rule":
                (
                    "End at the next frozen analysis date "
                    "or the explicitly frozen forward horizon."
                ),

            "primary_cross_section":
                (
                    "Current factor universe at analysis_date; "
                    "must not be conditioned on future "
                    "outcome availability."
                ),

            "missing_future_outcome_policy":
                (
                    "Outcome missingness may remove an "
                    "observation from a particular validation "
                    "statistic, but must not redefine the "
                    "factor universe or factor values."
                ),

            "factor_sign_selection_from_outcome":
                False,

            "window_selection_from_outcome":
                False,
        },

        "source_integrity": {

            "screening_file_sha256":
                sha256_file(
                    SCREENING_PATH
                ),

            "recommended_core_file_sha256":
                sha256_file(
                    CORE_SOURCE_PATH
                ),

            "step6_metadata_sha256":
                sha256_file(
                    STEP6_METADATA_PATH
                ),
        },
    }

    day6_design_hash = (
        canonical_json_hash(
            frozen_payload
        )
    )

    final_design = dict(
        frozen_payload
    )

    final_design[
        "day6_design_hash"
    ] = day6_design_hash

    final_design[
        "freeze_created_at"
    ] = datetime.now().isoformat(
        timespec="seconds"
    )

    return (
        final_design,
        day6_design_hash,
    )


# ============================================================
# 13. Write Markdown Summary
# ============================================================

def build_summary_markdown(
    previous_design_hash,
    day6_design_hash,
    screening,
    core,
    auxiliary,
    excluded,
):

    status_counts = (
        screening[
            "screening_status"
        ]
        .value_counts()
        .to_dict()
    )

    lines = [

        "# M1 Day 6 Step 1 — Freeze Reduced Factor Set",

        "",

        "## 1. Purpose",

        "",

        (
            "Freeze the reduced network-factor set "
            "before any forward return or forward-risk "
            "validation is performed."
        ),

        "",

        "The research sequence is therefore:",

        "",

        r"\[",
        r"\text{Pre-outcome Screening}",
        r"\rightarrow",
        r"\text{Factor Freeze}",
        r"\rightarrow",
        r"\text{Forward Labels}",
        r"\rightarrow",
        r"\text{Alpha / Risk Validation}.",
        r"\]",

        "",

        "## 2. Design Hashes",

        "",

        f"- Parent Day-5 design hash: `{previous_design_hash}`",

        f"- Day-6 frozen design hash: `{day6_design_hash}`",

        "",

        "## 3. Step-6 Screening Result",

        "",
    ]

    for (
        status,
        count,
    ) in sorted(
        status_counts.items()
    ):

        lines.append(
            f"- `{status}`: {count}"
        )

    lines.extend(
        [

            "",

            "## 4. Frozen Core Factor Set",

            "",
        ]
    )

    for factor in core[
        "factor_name"
    ]:

        lines.append(
            f"- `{factor}`"
        )

    lines.extend(
        [

            "",

            (
                f"Total frozen core factors: "
                f"**{len(core)}**."
            ),

            "",

            "## 5. Auxiliary Factor Set",

            "",
        ]
    )

    for factor in auxiliary[
        "factor_name"
    ]:

        lines.append(
            f"- `{factor}`"
        )

    lines.extend(
        [

            "",

            (
                f"Total auxiliary factors: "
                f"**{len(auxiliary)}**."
            ),

            "",

            "## 6. Excluded / Review Factors",

            "",
        ]
    )

    for row in excluded.itertuples(
        index=False
    ):

        lines.append(
            f"- `{row.factor_name}` — "
            f"`{row.screening_status}`"
        )

    lines.extend(
        [

            "",

            "## 7. Frozen Research Rules",

            "",

            (
                "1. Primary Day-6 validation uses only "
                "the frozen CORE factor set."
            ),

            (
                "2. Auxiliary factors may be used only "
                "for robustness or interpretation."
            ),

            (
                "3. Future returns and future-risk labels "
                "must not be used to redefine factors."
            ),

            (
                "4. Factor sign, rolling window, or "
                "factor membership must not be selected "
                "after observing future outcomes."
            ),

            (
                "5. Future labels begin strictly after "
                "the analysis date."
            ),

            (
                "6. Future outcome availability must not "
                "be used to define the contemporaneous "
                "factor universe."
            ),

            "",

            "## 8. Next Step",

            "",

            (
                "Construct strictly forward-looking "
                "return and risk labels while preserving "
                "the frozen factor definitions."
            ),
        ]
    )

    return "\n".join(
        lines
    )


# ============================================================
# 14. Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 6 - Step 1"
    )

    print(
        "Freeze Reduced Factor Set"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # 14.1 Previous design
    # ========================================================

    print()
    print(
        "[1] Validate previous frozen design"
    )

    (
        previous_design,
        previous_design_hash,
    ) = validate_previous_design()

    print(
        "Parent design hash:"
    )

    print(
        previous_design_hash
    )

    # ========================================================
    # 14.2 Step-5 factor panel
    # ========================================================

    print()
    print(
        "[2] Validate final stock-level factor panel"
    )

    step5_metadata = validate_factor_panel(
        previous_design_hash
    )

    print(
        "Step-5 factor panel QA: PASS"
    )

    # ========================================================
    # 14.3 Step-6 pre-outcome screening
    # ========================================================

    print()
    print(
        "[3] Validate pre-outcome screening"
    )

    screening_metadata = (
        validate_preoutcome_screening(
            previous_design_hash
        )
    )

    print(
        "Pre-outcome screening: VERIFIED"
    )

    # ========================================================
    # 14.4 Load screening result
    # ========================================================

    print()
    print(
        "[4] Load screening decisions"
    )

    screening = (
        load_screening_decisions()
    )

    print(
        f"Candidate factor count: "
        f"{len(screening)}"
    )

    print()

    print(
        screening[
            "screening_status"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # 14.5 Split factor sets
    # ========================================================

    (
        core,
        auxiliary,
        excluded,
    ) = split_factor_sets(
        screening
    )

    # ========================================================
    # 14.6 Strict core consistency
    # ========================================================

    print()
    print(
        "[5] Validate core factor set"
    )

    validate_core_set(
        core
    )

    validate_recommended_core_file(
        core
    )

    print(
        f"Core factor count: "
        f"{len(core)}"
    )

    print(
        f"Auxiliary factor count: "
        f"{len(auxiliary)}"
    )

    print(
        f"Excluded/review factor count: "
        f"{len(excluded)}"
    )

    # ========================================================
    # 14.7 Build manifest
    # ========================================================

    manifest = build_freeze_manifest(

        core=
            core,

        auxiliary=
            auxiliary,

        excluded=
            excluded,
    )

    # ========================================================
    # 14.8 Build Day-6 frozen design
    # ========================================================

    (
        day6_design,
        day6_design_hash,
    ) = build_day6_design(

        previous_design_hash=
            previous_design_hash,

        screening_metadata=
            screening_metadata,

        core=
            core,

        auxiliary=
            auxiliary,

        excluded=
            excluded,
    )

    print()
    print(
        "Day-6 frozen design hash:"
    )

    print(
        day6_design_hash
    )

    # ========================================================
    # 14.9 Freeze-specific output tables
    # ========================================================

    frozen_core = annotate_frozen_set(
        core,
        "CORE",
    )

    frozen_auxiliary = annotate_frozen_set(
        auxiliary,
        "AUXILIARY",
    )

    frozen_excluded = annotate_frozen_set(
        excluded,
        "EXCLUDED",
    )

    save_csv_atomic(
        frozen_core,
        FROZEN_CORE_PATH,
    )

    save_csv_atomic(
        frozen_auxiliary,
        FROZEN_AUX_PATH,
    )

    save_csv_atomic(
        frozen_excluded,
        FROZEN_EXCLUDED_PATH,
    )

    save_csv_atomic(
        manifest,
        FREEZE_MANIFEST_PATH,
    )

    # ========================================================
    # 14.10 Save design
    # ========================================================

    save_json_atomic(
        day6_design,
        DAY6_DESIGN_PATH,
    )

    # ========================================================
    # 14.11 Summary MD
    # ========================================================

    summary_md = build_summary_markdown(

        previous_design_hash=
            previous_design_hash,

        day6_design_hash=
            day6_design_hash,

        screening=
            screening,

        core=
            core,

        auxiliary=
            auxiliary,

        excluded=
            excluded,
    )

    temp_summary = Path(
        str(
            SUMMARY_PATH
        )
        +
        ".tmp"
    )

    temp_summary.write_text(
        summary_md,
        encoding="utf-8",
    )

    os.replace(
        temp_summary,
        SUMMARY_PATH,
    )

    # ========================================================
    # 14.12 Final console output
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "Frozen CORE factors"
    )

    print(
        "=" * 80
    )

    print(
        core[
            [
                "factor_name",
                "family",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "Freeze complete"
    )

    print(
        "=" * 80
    )

    print(
        f"CORE:       {len(core)}"
    )

    print(
        f"AUXILIARY:  {len(auxiliary)}"
    )

    print(
        f"EXCLUDED:   {len(excluded)}"
    )

    print()

    print(
        "Day-6 design hash:"
    )

    print(
        day6_design_hash
    )

    print()

    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Next step:"
    )

    print(
        "Construct strictly forward-looking "
        "return and risk labels."
    )


if __name__ == "__main__":

    main()