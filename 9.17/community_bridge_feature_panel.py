from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 0. Paths
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)


# ------------------------------------------------------------
# Research Day 4 outputs
# ------------------------------------------------------------

PREVIOUS_DAY_ROOT = (
    OUTPUT_ROOT
    / "M1_day4"
)


# ------------------------------------------------------------
# Day 4 Step 2:
# Residual edge masters
# ------------------------------------------------------------

NETWORK_STEP2_DIR = (
    PREVIOUS_DAY_ROOT
    / "02_step2_matched_networks"
)

EDGE_ROOT = (
    NETWORK_STEP2_DIR
    / "edge_master"
)


# ------------------------------------------------------------
# Day 4 Step 4:
# Louvain community memberships
# ------------------------------------------------------------

NETWORK_STEP4_DIR = (
    PREVIOUS_DAY_ROOT
    / "04_step4_industry_community_validation"
)

COMMUNITY_ROOT = (
    NETWORK_STEP4_DIR
    / "community_membership"
)


RESIDUAL_NETWORK_ID = (
    "M1_RESIDUAL_POSITIVE"
)


# ------------------------------------------------------------
# Day 5 root
# ------------------------------------------------------------

DAY5_ROOT = (
    OUTPUT_ROOT
    / "M1_day5"
)


# ------------------------------------------------------------
# Day 5 Step 1
# ------------------------------------------------------------

STEP1_DIR = (
    DAY5_ROOT
    / "01_step1_factor_design"
)

FACTOR_CONFIG_PATH = (
    STEP1_DIR
    / "factor_design_config.json"
)

FACTOR_DICTIONARY_PATH = (
    STEP1_DIR
    / "network_factor_dictionary.csv"
)

ANALYSIS_DATES_PATH = (
    STEP1_DIR
    / "factor_analysis_dates.csv"
)


# ------------------------------------------------------------
# Day 5 Step 4:
# Dynamic feature panel
# ------------------------------------------------------------

STEP4_DIR = (
    DAY5_ROOT
    / "04_step4_dynamic_features"
)

STEP4_PARTITION_DIR = (
    STEP4_DIR
    / "panel_parts"
)

STEP4_PANEL_PATH = (
    STEP4_DIR
    / "dynamic_network_feature_panel.parquet"
)

STEP4_METADATA_PATH = (
    STEP4_DIR
    / "step4_dynamic_feature_metadata.json"
)


# ------------------------------------------------------------
# Day 5 Step 5 outputs
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY5_ROOT
    / "05_step5_community_features"
)

PARTITION_DIR = (
    OUTPUT_DIR
    / "panel_parts"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARTITION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FINAL_PANEL_PATH = (
    OUTPUT_DIR
    / "community_bridge_feature_panel.parquet"
)

JOB_QA_PATH = (
    OUTPUT_DIR
    / "community_bridge_job_qa.csv"
)

COVERAGE_PATH = (
    OUTPUT_DIR
    / "community_bridge_coverage_summary.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "community_bridge_feature_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "step5_community_bridge_metadata.json"
)


# ============================================================
# 1. Settings
# ============================================================

WINDOWS = [
    60,
    120,
    252,
]

PRIMARY_DENSITY = 0.01

SAVE_JOB_PARTITIONS = True

STRICT_QA = True


# ------------------------------------------------------------
# Day 4 Step 4 excludes isolated nodes from Louvain.
#
# Every active residual-network node should therefore have
# a community assignment.
# ------------------------------------------------------------

REQUIRE_COMPLETE_ACTIVE_COMMUNITY = True


# ------------------------------------------------------------
# Strength reconstruction tolerance
# ------------------------------------------------------------

STRENGTH_ATOL = 1e-4

STRENGTH_RTOL = 1e-5


# ------------------------------------------------------------
# Generic numerical tolerance
# ------------------------------------------------------------

FLOAT_TOL = 1e-10


# ------------------------------------------------------------
# Cross-sectional percentile convention
# ------------------------------------------------------------

PERCENTILE_TIE_METHOD = "average"


# ============================================================
# 2. Frozen Community Factors
# ============================================================

REQUIRED_FROZEN_FACTORS = {

    "community_size",

    "within_community_degree",

    "outside_community_degree",

    "participation_coefficient",
}


# ============================================================
# 3. Utility Functions
# ============================================================

def load_json(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_json(
    obj,
    path: Path,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def percentile_rank(
    values,
):

    return (
        pd.Series(
            values
        )
        .rank(
            method=PERCENTILE_TIE_METHOD,
            pct=True,
        )
        .to_numpy(
            dtype=np.float64
        )
    )


def percentile_rank_with_na(
    values,
):

    x = pd.Series(
        values,
        dtype="float64",
    )

    result = np.full(
        len(x),
        np.nan,
        dtype=np.float64,
    )

    valid = x.notna()

    if valid.sum() > 0:

        result[
            valid.to_numpy()
        ] = (
            x.loc[
                valid
            ]
            .rank(
                method=PERCENTILE_TIE_METHOD,
                pct=True,
            )
            .to_numpy(
                dtype=np.float64
            )
        )

    return result


def safe_ratio(
    numerator,
    denominator,
):

    numerator = np.asarray(
        numerator,
        dtype=np.float64,
    )

    denominator = np.asarray(
        denominator,
        dtype=np.float64,
    )

    result = np.full(
        len(numerator),
        np.nan,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            numerator
        )
        &
        np.isfinite(
            denominator
        )
        &
        (
            denominator
            >
            0
        )
    )

    result[
        valid
    ] = (
        numerator[
            valid
        ]
        /
        denominator[
            valid
        ]
    )

    return result


# ============================================================
# 4. Load Frozen Design
# ============================================================

def load_frozen_design():

    config = load_json(
        FACTOR_CONFIG_PATH
    )

    if (
        config.get(
            "design_status"
        )
        !=
        "FROZEN"
    ):

        raise RuntimeError(
            "Factor design is not FROZEN."
        )

    design_hash = (
        config.get(
            "design_hash"
        )
    )

    if not design_hash:

        raise RuntimeError(
            "Frozen factor design has no design_hash."
        )

    design = (
        config[
            "network_design"
        ]
    )

    if (
        design[
            "primary_network"
        ]
        !=
        RESIDUAL_NETWORK_ID
    ):

        raise RuntimeError(
            "Primary network mismatch."
        )

    if not np.isclose(
        float(
            design[
                "primary_density"
            ]
        ),
        PRIMARY_DENSITY,
    ):

        raise RuntimeError(
            "Primary density mismatch."
        )

    if (
        sorted(
            design[
                "windows"
            ]
        )
        !=
        sorted(
            WINDOWS
        )
    ):

        raise RuntimeError(
            "Frozen windows mismatch."
        )

    return (
        config,
        design_hash,
    )


# ============================================================
# 5. Validate Step 4 Lineage
# ============================================================

def validate_step4_lineage(
    design_hash,
):

    metadata = load_json(
        STEP4_METADATA_PATH
    )

    if (
        metadata.get(
            "design_hash"
        )
        !=
        design_hash
    ):

        raise RuntimeError(
            "Step 4 dynamic panel was generated "
            "from a different frozen design."
        )

    formal_qa = metadata.get(
        "formal_qa",
        {},
    )

    if not formal_qa.get(
        "all_jobs_pass",
        False,
    ):

        raise RuntimeError(
            "Step 4 dynamic panel did not "
            "pass all formal QA."
        )

    return metadata


# ============================================================
# 6. Validate Frozen Factor Dictionary
# ============================================================

def validate_factor_dictionary():

    if not FACTOR_DICTIONARY_PATH.exists():

        raise FileNotFoundError(
            "Missing factor dictionary:\n"
            f"{FACTOR_DICTIONARY_PATH}"
        )

    df = pd.read_csv(
        FACTOR_DICTIONARY_PATH
    )

    if (
        "factor_name"
        not in df.columns
    ):

        raise ValueError(
            "Factor dictionary has no factor_name."
        )

    existing = set(
        df[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    missing = (
        REQUIRED_FROZEN_FACTORS
        -
        existing
    )

    if missing:

        raise RuntimeError(
            "Frozen factor dictionary misses:\n"
            f"{sorted(missing)}"
        )

    return df


# ============================================================
# 7. Load Analysis Dates
# ============================================================

def load_analysis_dates():

    if not ANALYSIS_DATES_PATH.exists():

        raise FileNotFoundError(
            f"Missing:\n{ANALYSIS_DATES_PATH}"
        )

    df = pd.read_csv(
        ANALYSIS_DATES_PATH
    )

    required = [

        "analysis_date",

        "window",

        "common_node_count",

        "edge_budget_k",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "factor_analysis_dates.csv "
            f"missing columns: {missing}"
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
        "window"
    ] = pd.to_numeric(
        df[
            "window"
        ],
        errors="raise",
    ).astype(int)

    return (
        df.sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 8. Load Step 4 Dynamic Panel Partition
# ============================================================

def dynamic_panel_path(
    window,
    date,
):

    date_string = (
        pd.Timestamp(
            date
        )
        .strftime(
            "%Y-%m-%d"
        )
    )

    return (
        STEP4_PARTITION_DIR
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )


def load_dynamic_panel(
    window,
    date,
):

    path = dynamic_panel_path(
        window,
        date,
    )

    if not path.exists():

        raise FileNotFoundError(
            "Dynamic panel partition not found:\n"
            f"{path}"
        )

    df = pd.read_parquet(
        path
    )

    required = [

        "analysis_date",

        "window",

        "master_index",

        "security_id",

        "residual_degree",

        "residual_strength",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Dynamic panel missing "
            f"columns: {missing}"
        )

    if (
        df[
            "master_index"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate master_index "
            "in dynamic panel."
        )

    return (
        df.sort_values(
            "master_index"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 9. Load Residual Edge Master
# ============================================================

def residual_edge_path(
    window,
    date,
):

    date_string = (
        pd.Timestamp(
            date
        )
        .strftime(
            "%Y-%m-%d"
        )
    )

    return (
        EDGE_ROOT
        /
        RESIDUAL_NETWORK_ID
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )


def load_residual_edges(
    window,
    date,
    edge_budget,
):

    path = residual_edge_path(
        window,
        date,
    )

    if not path.exists():

        raise FileNotFoundError(
            "Residual edge master not found:\n"
            f"{path}"
        )

    edges = pd.read_parquet(
        path
    )

    required = [

        "master_index_i",

        "master_index_j",

        "correlation",

        "rank",
    ]

    missing = [
        col
        for col in required
        if col not in edges.columns
    ]

    if missing:

        raise ValueError(
            "Residual edge master "
            f"missing columns: {missing}"
        )

    edges = (
        edges[
            edges[
                "rank"
            ]
            <=
            int(
                edge_budget
            )
        ]
        .sort_values(
            "rank"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            edges
        )
        !=
        int(
            edge_budget
        )
    ):

        raise RuntimeError(
            "Residual edge count mismatch.\n"
            f"Expected={edge_budget:,}\n"
            f"Actual={len(edges):,}"
        )

    if (
        edges[
            "master_index_i"
        ]
        ==
        edges[
            "master_index_j"
        ]
    ).any():

        raise RuntimeError(
            "Self-loop found in residual network."
        )

    return edges


# ============================================================
# 10. Resolve Community Membership File
#
# This is intentionally flexible because the Day-4 Step-4
# output file name may contain a prefix / suffix.
# ============================================================

def resolve_community_path(
    window,
    date,
):

    date_string = (
        pd.Timestamp(
            date
        )
        .strftime(
            "%Y-%m-%d"
        )
    )

    candidate_dirs = [

        (
            COMMUNITY_ROOT
            /
            RESIDUAL_NETWORK_ID
            /
            f"W{window}"
        ),

        (
            COMMUNITY_ROOT
            /
            f"W{window}"
            /
            RESIDUAL_NETWORK_ID
        ),

        (
            COMMUNITY_ROOT
            /
            f"W{window}"
        ),
    ]

    # --------------------------------------------------------
    # First try exact date filenames.
    # --------------------------------------------------------

    for directory in candidate_dirs:

        for suffix in [
            ".parquet",
            ".csv",
        ]:

            path = (
                directory
                /
                f"{date_string}{suffix}"
            )

            if path.exists():

                return path

    # --------------------------------------------------------
    # Then search for a date-containing file.
    # --------------------------------------------------------

    candidates = []

    for directory in candidate_dirs:

        if not directory.exists():

            continue

        candidates.extend(
            directory.glob(
                f"*{date_string}*.parquet"
            )
        )

        candidates.extend(
            directory.glob(
                f"*{date_string}*.csv"
            )
        )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    candidates = list(
        dict.fromkeys(
            candidates
        )
    )

    if len(
        candidates
    ) == 1:

        return candidates[
            0
        ]

    if len(
        candidates
    ) == 0:

        raise FileNotFoundError(
            "Community membership file not found.\n"
            f"W={window}, date={date_string}\n"
            f"Root={COMMUNITY_ROOT}"
        )

    raise RuntimeError(
        "Multiple community membership files found:\n"
        +
        "\n".join(
            str(x)
            for x in candidates
        )
    )


# ============================================================
# 11. Load Community Membership
# ============================================================

def load_community_membership(
    window,
    date,
):

    path = resolve_community_path(
        window,
        date,
    )

    if (
        path.suffix.lower()
        ==
        ".parquet"
    ):

        df = pd.read_parquet(
            path
        )

    elif (
        path.suffix.lower()
        ==
        ".csv"
    ):

        df = pd.read_csv(
            path
        )

    else:

        raise ValueError(
            f"Unsupported membership file: {path}"
        )

    # --------------------------------------------------------
    # Resolve node identifier
    # --------------------------------------------------------

    master_candidates = [

        "master_index",

        "master_idx",

        "node_master_index",
    ]

    master_col = None

    for candidate in master_candidates:

        if candidate in df.columns:

            master_col = candidate

            break

    if master_col is None:

        raise ValueError(
            "Community membership file "
            "has no master_index-like column.\n"
            f"Columns={list(df.columns)}"
        )

    # --------------------------------------------------------
    # Resolve community column
    # --------------------------------------------------------

    community_candidates = [

        "community_id",

        "community",

        "louvain_community",

        "community_label",

        "cluster_id",

        "cluster",
    ]

    community_col = None

    for candidate in community_candidates:

        if candidate in df.columns:

            community_col = candidate

            break

    # --------------------------------------------------------
    # Fallback:
    # locate a plausible community column.
    # --------------------------------------------------------

    if community_col is None:

        possible = []

        for col in df.columns:

            lower = str(
                col
            ).lower()

            if (
                (
                    "community"
                    in lower
                )
                and
                (
                    "size"
                    not in lower
                )
                and
                (
                    "count"
                    not in lower
                )
                and
                (
                    "degree"
                    not in lower
                )
                and
                (
                    "modularity"
                    not in lower
                )
            ):

                possible.append(
                    col
                )

        if len(
            possible
        ) == 1:

            community_col = (
                possible[
                    0
                ]
            )

    if community_col is None:

        raise ValueError(
            "Cannot identify community column.\n"
            f"Columns={list(df.columns)}"
        )

    membership = df[
        [
            master_col,
            community_col,
        ]
    ].copy()

    membership.columns = [

        "master_index",

        "community_id",
    ]

    membership[
        "master_index"
    ] = pd.to_numeric(
        membership[
            "master_index"
        ],
        errors="raise",
    ).astype(
        np.int64
    )

    membership[
        "community_id"
    ] = (
        membership[
            "community_id"
        ]
        .astype(
            "string"
        )
    )

    # --------------------------------------------------------
    # Remove invalid community labels.
    # --------------------------------------------------------

    invalid = (
        membership[
            "community_id"
        ]
        .isna()
        |
        membership[
            "community_id"
        ]
        .isin(
            [
                "",
                "nan",
                "None",
                "<NA>",
                "-1",
            ]
        )
    )

    membership = (
        membership[
            ~invalid
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Conflicting duplicate assignment is not allowed.
    # --------------------------------------------------------

    conflicts = (
        membership.groupby(
            "master_index"
        )[
            "community_id"
        ]
        .nunique()
    )

    if (
        conflicts
        >
        1
    ).any():

        bad = conflicts[
            conflicts
            >
            1
        ]

        raise RuntimeError(
            "Conflicting community assignments "
            "for the same master_index.\n"
            f"Count={len(bad)}"
        )

    membership = (
        membership.drop_duplicates(
            "master_index"
        )
        .sort_values(
            "master_index"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        membership,
        path,
        community_col,
    )


# ============================================================
# 12. Attach Community Membership to Node Panel
# ============================================================

def attach_community_labels(
    panel,
    membership,
):

    result = (
        panel.copy()
    )

    mapping = (
        membership
        .set_index(
            "master_index"
        )[
            "community_id"
        ]
        .to_dict()
    )

    result[
        "community_id"
    ] = (
        result[
            "master_index"
        ]
        .map(
            mapping
        )
        .astype(
            "string"
        )
    )

    result[
        "community_member"
    ] = (
        result[
            "community_id"
        ]
        .notna()
    )

    return result


# ============================================================
# 13. Encode Communities
# ============================================================

def encode_communities(
    panel,
):

    valid = (
        panel[
            "community_id"
        ]
        .notna()
    )

    codes = np.full(
        len(
            panel
        ),
        -1,
        dtype=np.int32,
    )

    if valid.sum() > 0:

        valid_codes, uniques = (
            pd.factorize(
                panel.loc[
                    valid,
                    "community_id"
                ],
                sort=True,
            )
        )

        codes[
            valid.to_numpy()
        ] = (
            valid_codes.astype(
                np.int32
            )
        )

        n_communities = len(
            uniques
        )

    else:

        n_communities = 0

    return (
        codes,
        n_communities,
    )


# ============================================================
# 14. Build master_index -> local row map
# ============================================================

def build_local_index_map(
    panel,
):

    master_index = (
        panel[
            "master_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    if len(
        master_index
    ) == 0:

        raise RuntimeError(
            "Empty node panel."
        )

    max_master = int(
        master_index.max()
    )

    local_map = np.full(
        max_master + 1,
        -1,
        dtype=np.int32,
    )

    local_map[
        master_index
    ] = np.arange(
        len(
            master_index
        ),
        dtype=np.int32,
    )

    return (
        master_index,
        local_map,
    )


# ============================================================
# 15. Compute Community / Bridge Features
# ============================================================

def build_community_features(
    panel,
    edges,
):

    n = len(
        panel
    )

    result = (
        panel.copy()
    )

    (
        community_codes,
        n_communities,
    ) = encode_communities(
        result
    )

    if (
        n_communities
        ==
        0
    ):

        raise RuntimeError(
            "No valid community assignments."
        )

    # ========================================================
    # Community size
    # ========================================================

    community_counts = np.bincount(
        community_codes[
            community_codes
            >=
            0
        ],
        minlength=n_communities,
    )

    community_size = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    valid_member = (
        community_codes
        >=
        0
    )

    community_size[
        valid_member
    ] = (
        community_counts[
            community_codes[
                valid_member
            ]
        ]
    )

    # ========================================================
    # Node row mapping
    # ========================================================

    (
        master_index,
        local_map,
    ) = build_local_index_map(
        result
    )

    edge_i = (
        edges[
            "master_index_i"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    edge_j = (
        edges[
            "master_index_j"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    weights = (
        edges[
            "correlation"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if (
        edge_i.max()
        >=
        len(
            local_map
        )
        or
        edge_j.max()
        >=
        len(
            local_map
        )
    ):

        raise RuntimeError(
            "Edge endpoint exceeds current "
            "master-index mapping."
        )

    u = local_map[
        edge_i
    ]

    v = local_map[
        edge_j
    ]

    if (
        np.any(
            u < 0
        )
        or
        np.any(
            v < 0
        )
    ):

        raise RuntimeError(
            "Residual edge endpoint is absent "
            "from dynamic panel."
        )

    community_u = (
        community_codes[
            u
        ]
    )

    community_v = (
        community_codes[
            v
        ]
    )

    known_edge = (
        (community_u >= 0)
        &
        (community_v >= 0)
    )

    within_edge = (
        known_edge
        &
        (
            community_u
            ==
            community_v
        )
    )

    outside_edge = (
        known_edge
        &
        (
            community_u
            !=
            community_v
        )
    )

    unknown_edge = (
        ~known_edge
    )

    # ========================================================
    # Degree / strength decomposition arrays
    # ========================================================

    within_degree = np.zeros(
        n,
        dtype=np.int32,
    )

    outside_degree = np.zeros(
        n,
        dtype=np.int32,
    )

    unknown_degree = np.zeros(
        n,
        dtype=np.int32,
    )

    within_strength = np.zeros(
        n,
        dtype=np.float64,
    )

    outside_strength = np.zeros(
        n,
        dtype=np.float64,
    )

    unknown_strength = np.zeros(
        n,
        dtype=np.float64,
    )

    # ========================================================
    # Accumulator
    # ========================================================

    def accumulate(
        mask,
        degree_array,
        strength_array,
    ):

        uu = u[
            mask
        ]

        vv = v[
            mask
        ]

        ww = weights[
            mask
        ]

        np.add.at(
            degree_array,
            uu,
            1,
        )

        np.add.at(
            degree_array,
            vv,
            1,
        )

        np.add.at(
            strength_array,
            uu,
            ww,
        )

        np.add.at(
            strength_array,
            vv,
            ww,
        )

    accumulate(
        within_edge,
        within_degree,
        within_strength,
    )

    accumulate(
        outside_edge,
        outside_degree,
        outside_strength,
    )

    accumulate(
        unknown_edge,
        unknown_degree,
        unknown_strength,
    )

    total_degree_reconstructed = (
        within_degree
        +
        outside_degree
        +
        unknown_degree
    )

    total_strength_reconstructed = (
        within_strength
        +
        outside_strength
        +
        unknown_strength
    )

    # ========================================================
    # Ratios
    # ========================================================

    residual_degree = (
        result[
            "residual_degree"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    within_degree_ratio = safe_ratio(
        within_degree,
        residual_degree,
    )

    outside_degree_ratio = safe_ratio(
        outside_degree,
        residual_degree,
    )

    unknown_degree_ratio = safe_ratio(
        unknown_degree,
        residual_degree,
    )

    # --------------------------------------------------------
    # Use edge-master reconstructed strength in denominator.
    # This guarantees the decomposition and ratios use the
    # exact same edge-weight source.
    # --------------------------------------------------------

    within_strength_ratio = safe_ratio(
        within_strength,
        total_strength_reconstructed,
    )

    outside_strength_ratio = safe_ratio(
        outside_strength,
        total_strength_reconstructed,
    )

    unknown_strength_ratio = safe_ratio(
        unknown_strength,
        total_strength_reconstructed,
    )

    # ========================================================
    # Participation Coefficient
    #
    # For every undirected edge i-j create:
    #
    # i -> community(j)
    # j -> community(i)
    #
    # Then calculate the distribution of each node's links
    # across all neighbor communities.
    # ========================================================

    directed_source = np.concatenate(
        [
            u,
            v,
        ]
    )

    directed_neighbor_community = (
        np.concatenate(
            [
                community_v,
                community_u,
            ]
        )
    )

    directed_weight = np.concatenate(
        [
            weights,
            weights,
        ]
    )

    known_neighbor = (
        directed_neighbor_community
        >=
        0
    )

    unknown_neighbor_count = np.bincount(

        directed_source[
            ~known_neighbor
        ],

        minlength=n,
    )

    unknown_neighbor_strength = np.bincount(

        directed_source[
            ~known_neighbor
        ],

        weights=
            directed_weight[
                ~known_neighbor
            ],

        minlength=n,
    )

    # --------------------------------------------------------
    # One unique key for node × neighbor-community pair.
    # --------------------------------------------------------

    src_known = (
        directed_source[
            known_neighbor
        ]
        .astype(
            np.int64
        )
    )

    nbr_comm_known = (
        directed_neighbor_community[
            known_neighbor
        ]
        .astype(
            np.int64
        )
    )

    weight_known = (
        directed_weight[
            known_neighbor
        ]
        .astype(
            np.float64
        )
    )

    pair_key = (
        src_known
        *
        int(
            n_communities
        )
        +
        nbr_comm_known
    )

    (
        unique_key,
        inverse_index,
    ) = np.unique(
        pair_key,
        return_inverse=True,
    )

    pair_degree = np.bincount(
        inverse_index
    ).astype(
        np.float64
    )

    pair_strength = np.bincount(
        inverse_index,
        weights=
            weight_known,
    ).astype(
        np.float64
    )

    pair_node = (
        unique_key
        //
        int(
            n_communities
        )
    ).astype(
        np.int64
    )

    pair_community = (
        unique_key
        %
        int(
            n_communities
        )
    ).astype(
        np.int64
    )

    # ========================================================
    # Sum k_ic^2 and s_ic^2 by node
    # ========================================================

    sum_squared_degree_by_community = np.zeros(
        n,
        dtype=np.float64,
    )

    np.add.at(

        sum_squared_degree_by_community,

        pair_node,

        pair_degree ** 2,
    )

    sum_squared_strength_by_community = np.zeros(
        n,
        dtype=np.float64,
    )

    np.add.at(

        sum_squared_strength_by_community,

        pair_node,

        pair_strength ** 2,
    )

    # ========================================================
    # Distinct neighbor-community counts
    # ========================================================

    community_diversity_count = np.bincount(

        pair_node,

        minlength=n,
    ).astype(
        np.int32
    )

    # ========================================================
    # Number of distinct EXTERNAL communities
    # ========================================================

    own_community_for_pair = (
        community_codes[
            pair_node
        ]
    )

    external_pair = (
        (own_community_for_pair >= 0)
        &
        (
            pair_community
            !=
            own_community_for_pair
        )
    )

    external_community_count = np.bincount(

        pair_node[
            external_pair
        ],

        minlength=n,
    ).astype(
        np.int32
    )

    # ========================================================
    # Unweighted Participation Coefficient
    # ========================================================

    participation = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    valid_participation = (

        (residual_degree > 0)

        &

        (community_codes >= 0)

        &

        (unknown_neighbor_count == 0)
    )

    participation[
        valid_participation
    ] = (

        1.0

        -

        (
            sum_squared_degree_by_community[
                valid_participation
            ]

            /

            (
                residual_degree[
                    valid_participation
                ]
                ** 2
            )
        )
    )

    participation[
        valid_participation
    ] = np.clip(

        participation[
            valid_participation
        ],

        0.0,

        1.0,
    )

    # ========================================================
    # Weighted Participation Coefficient
    #
    # Auxiliary weighted analogue:
    #
    # Pw_i = 1 - sum_c (s_ic / s_i)^2
    # ========================================================

    weighted_participation = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    valid_weighted = (

        (
            total_strength_reconstructed
            >
            0
        )

        &

        (community_codes >= 0)

        &

        (
            np.abs(
                unknown_neighbor_strength
            )
            <=
            STRENGTH_ATOL
        )
    )

    weighted_participation[
        valid_weighted
    ] = (

        1.0

        -

        (
            sum_squared_strength_by_community[
                valid_weighted
            ]

            /

            (
                total_strength_reconstructed[
                    valid_weighted
                ]
                ** 2
            )
        )
    )

    weighted_participation[
        valid_weighted
    ] = np.clip(

        weighted_participation[
            valid_weighted
        ],

        0.0,

        1.0,
    )

    # ========================================================
    # Cross-sectional percentile versions
    # ========================================================

    within_degree_pct = percentile_rank(
        within_degree
    )

    outside_degree_pct = percentile_rank(
        outside_degree
    )

    within_strength_pct = percentile_rank(
        within_strength
    )

    outside_strength_pct = percentile_rank(
        outside_strength
    )

    participation_pct = percentile_rank_with_na(
        participation
    )

    weighted_participation_pct = (
        percentile_rank_with_na(
            weighted_participation
        )
    )

    community_size_pct = (
        percentile_rank_with_na(
            community_size
        )
    )

    # ========================================================
    # Attach features
    # ========================================================

    result[
        "community_size"
    ] = community_size

    result[
        "community_size_percentile"
    ] = community_size_pct


    result[
        "within_community_degree"
    ] = within_degree

    result[
        "outside_community_degree"
    ] = outside_degree

    result[
        "unknown_community_degree"
    ] = unknown_degree


    result[
        "within_community_degree_ratio"
    ] = within_degree_ratio

    result[
        "outside_community_degree_ratio"
    ] = outside_degree_ratio

    result[
        "unknown_community_degree_ratio"
    ] = unknown_degree_ratio


    result[
        "within_community_strength"
    ] = within_strength

    result[
        "outside_community_strength"
    ] = outside_strength

    result[
        "unknown_community_strength"
    ] = unknown_strength


    result[
        "within_community_strength_ratio"
    ] = within_strength_ratio

    result[
        "outside_community_strength_ratio"
    ] = outside_strength_ratio

    result[
        "unknown_community_strength_ratio"
    ] = unknown_strength_ratio


    result[
        "within_community_degree_percentile"
    ] = within_degree_pct

    result[
        "outside_community_degree_percentile"
    ] = outside_degree_pct

    result[
        "within_community_strength_percentile"
    ] = within_strength_pct

    result[
        "outside_community_strength_percentile"
    ] = outside_strength_pct


    result[
        "participation_coefficient"
    ] = participation

    result[
        "participation_coefficient_percentile"
    ] = participation_pct


    result[
        "weighted_participation_coefficient"
    ] = weighted_participation

    result[
        "weighted_participation_coefficient_percentile"
    ] = weighted_participation_pct


    result[
        "community_diversity_count"
    ] = community_diversity_count

    result[
        "external_community_count"
    ] = external_community_count

    # ========================================================
    # Diagnostics
    # ========================================================

    active = (
        residual_degree
        >
        0
    )

    active_community_coverage = (

        float(
            (
                community_codes[
                    active
                ]
                >=
                0
            )
            .mean()
        )

        if active.sum() > 0

        else np.nan
    )

    diagnostics = {

        "community_count":
            int(
                n_communities
            ),

        "active_node_count":
            int(
                active.sum()
            ),

        "community_assigned_node_count":
            int(
                (
                    community_codes
                    >=
                    0
                )
                .sum()
            ),

        "active_community_coverage":
            active_community_coverage,

        "within_community_edge_count":
            int(
                within_edge.sum()
            ),

        "outside_community_edge_count":
            int(
                outside_edge.sum()
            ),

        "unknown_community_edge_count":
            int(
                unknown_edge.sum()
            ),

        "within_community_edge_share":
            float(
                within_edge.mean()
            ),

        "outside_community_edge_share":
            float(
                outside_edge.mean()
            ),

        "unknown_community_edge_share":
            float(
                unknown_edge.mean()
            ),

        "reconstructed_degree":
            total_degree_reconstructed,

        "reconstructed_strength":
            total_strength_reconstructed,
    }

    return (
        result,
        diagnostics,
    )


# ============================================================
# 16. Formal QA
# ============================================================

def qa_one_job(
    panel,
    diagnostics,
    expected_node_count,
    expected_edge_count,
    analysis_date,
    window,
):

    errors = []

    # ========================================================
    # 16.1 Node count
    # ========================================================

    node_count_ok = (
        len(
            panel
        )
        ==
        int(
            expected_node_count
        )
    )

    if not node_count_ok:

        errors.append(
            "node_count_mismatch"
        )

    # ========================================================
    # 16.2 Unique key
    # ========================================================

    key_unique = (
        not panel[
            [
                "analysis_date",
                "window",
                "master_index",
            ]
        ]
        .duplicated()
        .any()
    )

    if not key_unique:

        errors.append(
            "duplicate_panel_key"
        )

    # ========================================================
    # 16.3 Active community coverage
    # ========================================================

    active_community_coverage_ok = True

    if REQUIRE_COMPLETE_ACTIVE_COMMUNITY:

        active_community_coverage_ok = bool(

            np.isclose(
                diagnostics[
                    "active_community_coverage"
                ],
                1.0,
            )
        )

    if not active_community_coverage_ok:

        errors.append(
            "incomplete_active_community_membership"
        )

    # ========================================================
    # 16.4 Edge-category identity
    # ========================================================

    category_edge_sum = (

        diagnostics[
            "within_community_edge_count"
        ]

        +

        diagnostics[
            "outside_community_edge_count"
        ]

        +

        diagnostics[
            "unknown_community_edge_count"
        ]
    )

    edge_category_identity_ok = (
        category_edge_sum
        ==
        int(
            expected_edge_count
        )
    )

    if not edge_category_identity_ok:

        errors.append(
            "edge_category_identity_failure"
        )

    # ========================================================
    # 16.5 Unknown active-edge community assignment
    # ========================================================

    unknown_community_edge_ok = (

        diagnostics[
            "unknown_community_edge_count"
        ]
        ==
        0
    )

    if (
        REQUIRE_COMPLETE_ACTIVE_COMMUNITY
        and
        not unknown_community_edge_ok
    ):

        errors.append(
            "unknown_community_edge"
        )

    # ========================================================
    # 16.6 Degree reconstruction
    # ========================================================

    reconstructed_degree = (

        panel[
            "within_community_degree"
        ]

        +

        panel[
            "outside_community_degree"
        ]

        +

        panel[
            "unknown_community_degree"
        ]
    )

    degree_diff = (

        reconstructed_degree

        -

        panel[
            "residual_degree"
        ]
    )

    max_degree_reconstruction_error = int(
        degree_diff
        .abs()
        .max()
    )

    degree_reconstruction_ok = (
        max_degree_reconstruction_error
        ==
        0
    )

    if not degree_reconstruction_ok:

        errors.append(
            "degree_reconstruction_failure"
        )

    # ========================================================
    # 16.7 Category degree handshakes
    # ========================================================

    within_degree_handshake_ok = (

        int(
            panel[
                "within_community_degree"
            ]
            .sum()
        )

        ==

        2
        *
        diagnostics[
            "within_community_edge_count"
        ]
    )

    outside_degree_handshake_ok = (

        int(
            panel[
                "outside_community_degree"
            ]
            .sum()
        )

        ==

        2
        *
        diagnostics[
            "outside_community_edge_count"
        ]
    )

    unknown_degree_handshake_ok = (

        int(
            panel[
                "unknown_community_degree"
            ]
            .sum()
        )

        ==

        2
        *
        diagnostics[
            "unknown_community_edge_count"
        ]
    )

    if not (
        within_degree_handshake_ok
        and
        outside_degree_handshake_ok
        and
        unknown_degree_handshake_ok
    ):

        errors.append(
            "category_degree_handshake_failure"
        )

    # ========================================================
    # 16.8 Strength reconstruction
    # ========================================================

    reconstructed_strength = (

        panel[
            "within_community_strength"
        ]

        +

        panel[
            "outside_community_strength"
        ]

        +

        panel[
            "unknown_community_strength"
        ]

    ).to_numpy(
        dtype=np.float64
    )

    source_strength = (
        panel[
            "residual_strength"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    max_strength_reconstruction_error = float(
        np.nanmax(
            np.abs(

                reconstructed_strength

                -

                source_strength
            )
        )
    )

    strength_reconstruction_ok = bool(

        np.allclose(

            reconstructed_strength,

            source_strength,

            atol=
                STRENGTH_ATOL,

            rtol=
                STRENGTH_RTOL,

            equal_nan=True,
        )
    )

    if not strength_reconstruction_ok:

        errors.append(
            "strength_reconstruction_failure"
        )

    # ========================================================
    # 16.9 Ratio validity
    # ========================================================

    ratio_columns = [

        "within_community_degree_ratio",

        "outside_community_degree_ratio",

        "unknown_community_degree_ratio",

        "within_community_strength_ratio",

        "outside_community_strength_ratio",

        "unknown_community_strength_ratio",
    ]

    ratio_valid = True

    for col in ratio_columns:

        values = (
            pd.to_numeric(
                panel[
                    col
                ],
                errors="coerce",
            )
            .dropna()
        )

        if len(
            values
        ) == 0:

            continue

        if (

            (
                values
                <
                -FLOAT_TOL
            )
            .any()

            or

            (
                values
                >
                1.0
                +
                FLOAT_TOL
            )
            .any()
        ):

            ratio_valid = False

    if not ratio_valid:

        errors.append(
            "invalid_ratio"
        )

    # ========================================================
    # 16.10 Participation coefficient range
    # ========================================================

    participation = (
        panel[
            "participation_coefficient"
        ]
        .dropna()
    )

    participation_valid = True

    if len(
        participation
    ) > 0:

        participation_valid = bool(

            (
                participation
                >=
                -FLOAT_TOL
            )
            .all()

            and

            (
                participation
                <=
                1.0
                +
                FLOAT_TOL
            )
            .all()
        )

    if not participation_valid:

        errors.append(
            "invalid_participation_coefficient"
        )

    # ========================================================
    # 16.11 Weighted Participation range
    # ========================================================

    weighted_participation = (
        panel[
            "weighted_participation_coefficient"
        ]
        .dropna()
    )

    weighted_participation_valid = True

    if len(
        weighted_participation
    ) > 0:

        weighted_participation_valid = bool(

            (
                weighted_participation
                >=
                -FLOAT_TOL
            )
            .all()

            and

            (
                weighted_participation
                <=
                1.0
                +
                FLOAT_TOL
            )
            .all()
        )

    if not weighted_participation_valid:

        errors.append(
            "invalid_weighted_participation"
        )

    # ========================================================
    # 16.12 Community size validity
    # ========================================================

    community_size = (
        panel[
            "community_size"
        ]
        .dropna()
    )

    community_size_valid = bool(

        (
            community_size
            >=
            1
        )
        .all()
    )

    if not community_size_valid:

        errors.append(
            "invalid_community_size"
        )

    # ========================================================
    # Descriptive quantities
    # ========================================================

    active = (
        panel[
            "residual_degree"
        ]
        >
        0
    )

    participation_active = (
        panel.loc[
            active,
            "participation_coefficient"
        ]
        .dropna()
    )

    outside_ratio_active = (
        panel.loc[
            active,
            "outside_community_degree_ratio"
        ]
        .dropna()
    )

    external_community_active = (
        panel.loc[
            active,
            "external_community_count"
        ]
        .dropna()
    )

    mean_participation = (

        float(
            participation_active.mean()
        )

        if len(
            participation_active
        ) > 0

        else np.nan
    )

    median_participation = (

        float(
            participation_active.median()
        )

        if len(
            participation_active
        ) > 0

        else np.nan
    )

    mean_outside_degree_ratio = (

        float(
            outside_ratio_active.mean()
        )

        if len(
            outside_ratio_active
        ) > 0

        else np.nan
    )

    mean_external_community_count = (

        float(
            external_community_active.mean()
        )

        if len(
            external_community_active
        ) > 0

        else np.nan
    )

    # ========================================================
    # Final state
    # ========================================================

    qa_pass = (
        len(
            errors
        )
        ==
        0
    )

    return {

        "analysis_date":
            pd.Timestamp(
                analysis_date
            ),

        "window":
            int(
                window
            ),

        "expected_node_count":
            int(
                expected_node_count
            ),

        "actual_node_count":
            int(
                len(
                    panel
                )
            ),

        "expected_edge_count":
            int(
                expected_edge_count
            ),

        # ----------------------------------------------------
        # Community structure
        # ----------------------------------------------------

        "community_count":
            diagnostics[
                "community_count"
            ],

        "active_node_count":
            diagnostics[
                "active_node_count"
            ],

        "community_assigned_node_count":
            diagnostics[
                "community_assigned_node_count"
            ],

        "active_community_coverage":
            diagnostics[
                "active_community_coverage"
            ],

        "within_community_edge_count":
            diagnostics[
                "within_community_edge_count"
            ],

        "outside_community_edge_count":
            diagnostics[
                "outside_community_edge_count"
            ],

        "unknown_community_edge_count":
            diagnostics[
                "unknown_community_edge_count"
            ],

        "within_community_edge_share":
            diagnostics[
                "within_community_edge_share"
            ],

        "outside_community_edge_share":
            diagnostics[
                "outside_community_edge_share"
            ],

        "unknown_community_edge_share":
            diagnostics[
                "unknown_community_edge_share"
            ],

        # ----------------------------------------------------
        # Bridge diagnostics
        # ----------------------------------------------------

        "mean_participation_coefficient":
            mean_participation,

        "median_participation_coefficient":
            median_participation,

        "mean_outside_community_degree_ratio":
            mean_outside_degree_ratio,

        "mean_external_community_count":
            mean_external_community_count,

        # ----------------------------------------------------
        # Reconstruction errors
        # ----------------------------------------------------

        "max_degree_reconstruction_error":
            max_degree_reconstruction_error,

        "max_strength_reconstruction_error":
            max_strength_reconstruction_error,

        # ----------------------------------------------------
        # Formal QA
        # ----------------------------------------------------

        "node_count_ok":
            node_count_ok,

        "key_unique":
            key_unique,

        "active_community_coverage_ok":
            active_community_coverage_ok,

        "edge_category_identity_ok":
            edge_category_identity_ok,

        "unknown_community_edge_ok":
            unknown_community_edge_ok,

        "degree_reconstruction_ok":
            degree_reconstruction_ok,

        "within_degree_handshake_ok":
            within_degree_handshake_ok,

        "outside_degree_handshake_ok":
            outside_degree_handshake_ok,

        "unknown_degree_handshake_ok":
            unknown_degree_handshake_ok,

        "strength_reconstruction_ok":
            strength_reconstruction_ok,

        "ratio_valid":
            ratio_valid,

        "participation_valid":
            participation_valid,

        "weighted_participation_valid":
            weighted_participation_valid,

        "community_size_valid":
            community_size_valid,

        "qa_pass":
            qa_pass,

        "qa_errors":
            "|".join(
                errors
            ),
    }


# ============================================================
# 17. Save Partition
# ============================================================

def save_partition(
    panel,
    window,
    date,
):

    date_string = (
        pd.Timestamp(
            date
        )
        .strftime(
            "%Y-%m-%d"
        )
    )

    path = (
        PARTITION_DIR
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = Path(
        str(
            path
        )
        +
        ".tmp"
    )

    if temp.exists():

        temp.unlink()

    panel.to_parquet(
        temp,
        index=False,
        compression="zstd",
    )

    os.replace(
        temp,
        path,
    )


# ============================================================
# 18. Descriptive Summary
# ============================================================

SUMMARY_FEATURES = [

    "community_size",

    "within_community_degree",

    "outside_community_degree",

    "within_community_degree_ratio",

    "outside_community_degree_ratio",

    "within_community_strength",

    "outside_community_strength",

    "within_community_strength_ratio",

    "outside_community_strength_ratio",

    "participation_coefficient",

    "weighted_participation_coefficient",

    "community_diversity_count",

    "external_community_count",

    "within_community_degree_percentile",

    "outside_community_degree_percentile",

    "participation_coefficient_percentile",
]


def summarize_one_panel(
    panel,
):

    rows = []

    analysis_date = (
        panel[
            "analysis_date"
        ]
        .iloc[
            0
        ]
    )

    window = int(
        panel[
            "window"
        ]
        .iloc[
            0
        ]
    )

    for factor in SUMMARY_FEATURES:

        x = pd.to_numeric(
            panel[
                factor
            ],
            errors="coerce",
        )

        valid = (
            x.dropna()
        )

        if len(
            valid
        ) > 0:

            q = valid.quantile(
                [
                    0.01,
                    0.05,
                    0.50,
                    0.95,
                    0.99,
                ]
            )

            mean_value = float(
                valid.mean()
            )

            std_value = float(
                valid.std(
                    ddof=0
                )
            )

            zero_share = float(
                (
                    valid
                    ==
                    0
                )
                .mean()
            )

            p01 = float(
                q.loc[
                    0.01
                ]
            )

            p05 = float(
                q.loc[
                    0.05
                ]
            )

            median = float(
                q.loc[
                    0.50
                ]
            )

            p95 = float(
                q.loc[
                    0.95
                ]
            )

            p99 = float(
                q.loc[
                    0.99
                ]
            )

        else:

            mean_value = np.nan

            std_value = np.nan

            zero_share = np.nan

            p01 = np.nan

            p05 = np.nan

            median = np.nan

            p95 = np.nan

            p99 = np.nan

        rows.append(
            {

                "analysis_date":
                    analysis_date,

                "window":
                    window,

                "factor_name":
                    factor,

                "n":
                    int(
                        len(
                            x
                        )
                    ),

                "nonmissing_count":
                    int(
                        len(
                            valid
                        )
                    ),

                "missing_share":
                    float(
                        x.isna()
                        .mean()
                    ),

                "zero_share":
                    zero_share,

                "mean":
                    mean_value,

                "std":
                    std_value,

                "p01":
                    p01,

                "p05":
                    p05,

                "median":
                    median,

                "p95":
                    p95,

                "p99":
                    p99,
            }
        )

    return rows


# ============================================================
# 19. Coverage Summary
# ============================================================

def build_coverage_summary(
    qa,
):

    rows = []

    for window, group in (
        qa.groupby(
            "window"
        )
    ):

        rows.append(
            {

                "window":
                    int(
                        window
                    ),

                "analysis_date_count":
                    int(
                        len(
                            group
                        )
                    ),

                "qa_pass_share":
                    float(
                        group[
                            "qa_pass"
                        ]
                        .mean()
                    ),

                "mean_community_count":
                    float(
                        group[
                            "community_count"
                        ]
                        .mean()
                    ),

                "min_community_count":
                    int(
                        group[
                            "community_count"
                        ]
                        .min()
                    ),

                "max_community_count":
                    int(
                        group[
                            "community_count"
                        ]
                        .max()
                    ),

                "mean_active_community_coverage":
                    float(
                        group[
                            "active_community_coverage"
                        ]
                        .mean()
                    ),

                "mean_within_community_edge_share":
                    float(
                        group[
                            "within_community_edge_share"
                        ]
                        .mean()
                    ),

                "mean_outside_community_edge_share":
                    float(
                        group[
                            "outside_community_edge_share"
                        ]
                        .mean()
                    ),

                "mean_participation_coefficient":
                    float(
                        group[
                            "mean_participation_coefficient"
                        ]
                        .mean()
                    ),

                "median_of_monthly_participation":
                    float(
                        group[
                            "median_participation_coefficient"
                        ]
                        .median()
                    ),

                "mean_outside_community_degree_ratio":
                    float(
                        group[
                            "mean_outside_community_degree_ratio"
                        ]
                        .mean()
                    ),

                "mean_external_community_count":
                    float(
                        group[
                            "mean_external_community_count"
                        ]
                        .mean()
                    ),

                "max_degree_reconstruction_error":
                    int(
                        group[
                            "max_degree_reconstruction_error"
                        ]
                        .max()
                    ),

                "max_strength_reconstruction_error":
                    float(
                        group[
                            "max_strength_reconstruction_error"
                        ]
                        .max()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 20. Save Metadata
# ============================================================

def save_metadata(
    design_hash,
    total_rows,
    qa,
    summary,
):

    payload = {

        "research_day":
            5,

        "step":
            "Step5_Community_Bridge_Features",

        "design_hash":
            design_hash,

        "input_dynamic_panel":
            str(
                STEP4_PANEL_PATH
            ),

        "community_source":
            str(
                COMMUNITY_ROOT
            ),

        "residual_edge_source":
            str(
                EDGE_ROOT
                /
                RESIDUAL_NETWORK_ID
            ),

        "output_panel":
            str(
                FINAL_PANEL_PATH
            ),

        "primary_network":
            RESIDUAL_NETWORK_ID,

        "primary_density":
            PRIMARY_DENSITY,

        "windows":
            WINDOWS,

        "row_count":
            int(
                total_rows
            ),

        "community_method":
            (
                "Day-4 Step-4 weighted Louvain "
                "community assignment, resolution=1 "
                "with fixed seed."
            ),

        "core_factor_definitions": {

            "community_size":
                (
                    "Number of active residual-network "
                    "nodes assigned to stock i's "
                    "contemporaneous Louvain community."
                ),

            "within_community_degree":
                (
                    "Number of residual-network neighbors "
                    "inside stock i's own community."
                ),

            "outside_community_degree":
                (
                    "Number of residual-network neighbors "
                    "outside stock i's own community."
                ),

            "participation_coefficient":
                (
                    "1 - sum_c (k_ic / k_i)^2."
                ),
        },

        "auxiliary_features": [

            "within_community_strength",

            "outside_community_strength",

            "outside_community_degree_ratio",

            "outside_community_strength_ratio",

            "weighted_participation_coefficient",

            "community_diversity_count",

            "external_community_count",

            "within_community_degree_percentile",

            "outside_community_degree_percentile",

            "participation_coefficient_percentile",
        ],

        "interpretation": [

            (
                "Within-community connectivity measures "
                "embeddedness in an endogenous network group."
            ),

            (
                "Outside-community connectivity measures "
                "the number/intensity of links extending "
                "beyond the stock's own endogenous group."
            ),

            (
                "Participation coefficient is high when "
                "a node's links are distributed across "
                "multiple communities rather than "
                "concentrated in one community."
            ),

            (
                "A stock with high outside-community degree "
                "but low participation may connect strongly "
                "to only one other community; therefore "
                "outside degree and participation measure "
                "different bridge dimensions."
            ),

            (
                "Community labels are contemporaneous and "
                "their numeric/string IDs are not assumed "
                "to be comparable across dates."
            ),

            (
                "No future return or future-risk outcome "
                "is used in this step."
            ),
        ],

        "formal_qa": {

            "job_count":
                int(
                    len(
                        qa
                    )
                ),

            "all_jobs_pass":
                bool(
                    qa[
                        "qa_pass"
                    ]
                    .all()
                ),

            "failed_job_count":
                int(
                    (
                        ~qa[
                            "qa_pass"
                        ]
                    )
                    .sum()
                ),
        },

        "summary_row_count":
            int(
                len(
                    summary
                )
            ),
    }

    save_json(
        payload,
        METADATA_PATH,
    )


# ============================================================
# 21. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 5 - Step 5"
    )

    print(
        "Community / Bridge Features"
    )

    print("=" * 80)

    # ========================================================
    # 21.1 Frozen design and lineage
    # ========================================================

    print()
    print(
        "[1] Load frozen factor design"
    )

    (
        config,
        design_hash,
    ) = load_frozen_design()

    validate_factor_dictionary()

    validate_step4_lineage(
        design_hash
    )

    analysis_dates = (
        load_analysis_dates()
    )

    print(
        "Design hash:"
    )

    print(
        design_hash
    )

    print(
        f"Date-window jobs: "
        f"{len(analysis_dates):,}"
    )

    # ========================================================
    # 21.2 Prepare streaming output
    # ========================================================

    temp_final_path = Path(
        str(
            FINAL_PANEL_PATH
        )
        +
        ".tmp"
    )

    if temp_final_path.exists():

        temp_final_path.unlink()

    writer = None

    qa_rows = []

    summary_rows = []

    total_rows = 0

    membership_source_records = []

    # ========================================================
    # 21.3 Main loop
    # ========================================================

    try:

        total_jobs = len(
            analysis_dates
        )

        for job_no, row in enumerate(

            analysis_dates.itertuples(
                index=False
            ),

            start=1,
        ):

            date = pd.Timestamp(
                row.analysis_date
            )

            window = int(
                row.window
            )

            expected_nodes = int(
                row.common_node_count
            )

            expected_edges = int(
                row.edge_budget_k
            )

            print()
            print(
                f"[{job_no}/{total_jobs}] "
                f"W={window} | "
                f"{date.date()} | "
                f"N={expected_nodes:,} | "
                f"E={expected_edges:,}"
            )

            # =================================================
            # Current Day-5 Step-4 panel
            # =================================================

            panel = load_dynamic_panel(
                window=
                    window,
                date=
                    date,
            )

            if (
                len(
                    panel
                )
                !=
                expected_nodes
            ):

                raise RuntimeError(
                    "Dynamic panel node count mismatch."
                )

            # =================================================
            # Residual q=1% edge network
            # =================================================

            edges = load_residual_edges(

                window=
                    window,

                date=
                    date,

                edge_budget=
                    expected_edges,
            )

            # =================================================
            # Day-4 Step-4 Louvain membership
            # =================================================

            (
                membership,
                membership_path,
                original_community_column,
            ) = load_community_membership(

                window=
                    window,

                date=
                    date,
            )

            membership_source_records.append(
                {
                    "analysis_date":
                        date,

                    "window":
                        window,

                    "membership_path":
                        str(
                            membership_path
                        ),

                    "source_community_column":
                        str(
                            original_community_column
                        ),
                }
            )

            # =================================================
            # Attach membership
            # =================================================

            panel = attach_community_labels(

                panel=
                    panel,

                membership=
                    membership,
            )

            # =================================================
            # Build Community / Bridge factors
            # =================================================

            (
                community_panel,
                diagnostics,
            ) = build_community_features(

                panel=
                    panel,

                edges=
                    edges,
            )

            # =================================================
            # Formal QA
            # =================================================

            qa = qa_one_job(

                panel=
                    community_panel,

                diagnostics=
                    diagnostics,

                expected_node_count=
                    expected_nodes,

                expected_edge_count=
                    expected_edges,

                analysis_date=
                    date,

                window=
                    window,
            )

            qa_rows.append(
                qa
            )

            print(
                "  Communities = "
                f"{qa['community_count']}"
            )

            print(
                "  Active community coverage = "
                f"{qa['active_community_coverage']:.4f}"
            )

            print(
                "  Within-community edge share = "
                f"{qa['within_community_edge_share']:.4f}"
            )

            print(
                "  Outside-community edge share = "
                f"{qa['outside_community_edge_share']:.4f}"
            )

            print(
                "  Mean participation = "
                f"{qa['mean_participation_coefficient']:.4f}"
            )

            if (
                STRICT_QA
                and
                not qa[
                    "qa_pass"
                ]
            ):

                raise RuntimeError(
                    "QA failed for "
                    f"W={window}, "
                    f"{date.date()}:\n"
                    f"{qa['qa_errors']}"
                )

            # =================================================
            # Descriptive summary
            # =================================================

            summary_rows.extend(
                summarize_one_panel(
                    community_panel
                )
            )

            # =================================================
            # Save partition
            # =================================================

            if SAVE_JOB_PARTITIONS:

                save_partition(

                    panel=
                        community_panel,

                    window=
                        window,

                    date=
                        date,
                )

            # =================================================
            # Streaming final parquet
            # =================================================

            table = pa.Table.from_pandas(

                community_panel,

                preserve_index=False,
            )

            if writer is None:

                writer = pq.ParquetWriter(

                    temp_final_path,

                    table.schema,

                    compression="zstd",
                )

            else:

                if (
                    table.schema
                    !=
                    writer.schema
                ):

                    raise RuntimeError(
                        "Parquet schema changed "
                        "across jobs.\n"
                        f"W={window}, "
                        f"date={date.date()}"
                    )

            writer.write_table(
                table
            )

            total_rows += len(
                community_panel
            )

    finally:

        if writer is not None:

            writer.close()

    # ========================================================
    # 21.4 Atomic finalize
    # ========================================================

    if not temp_final_path.exists():

        raise RuntimeError(
            "Temporary community panel "
            "was not created."
        )

    os.replace(
        temp_final_path,
        FINAL_PANEL_PATH,
    )

    # ========================================================
    # 21.5 QA output
    # ========================================================

    qa_df = (
        pd.DataFrame(
            qa_rows
        )
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    qa_df.to_csv(

        JOB_QA_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 21.6 Coverage summary
    # ========================================================

    coverage_df = (
        build_coverage_summary(
            qa_df
        )
    )

    coverage_df.to_csv(

        COVERAGE_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 21.7 Feature summary
    # ========================================================

    summary_df = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "window",
                "factor_name",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    summary_df.to_csv(

        SUMMARY_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 21.8 Membership source inventory
    # ========================================================

    membership_source_df = pd.DataFrame(
        membership_source_records
    )

    membership_source_df.to_csv(

        OUTPUT_DIR
        /
        "community_membership_source_inventory.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 21.9 Metadata
    # ========================================================

    save_metadata(

        design_hash=
            design_hash,

        total_rows=
            total_rows,

        qa=
            qa_df,

        summary=
            summary_df,
    )

    # ========================================================
    # 21.10 Console summary
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Community / Bridge Feature Summary"
    )

    print("=" * 80)

    print(
        f"Total rows: "
        f"{total_rows:,}"
    )

    print(
        f"QA jobs: "
        f"{len(qa_df):,}"
    )

    print(
        f"QA pass: "
        f"{int(qa_df['qa_pass'].sum()):,}"
        f"/{len(qa_df):,}"
    )

    print()

    print(
        coverage_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Final panel:"
    )

    print(
        FINAL_PANEL_PATH
    )

    print()

    print("=" * 80)

    print(
        "Day 5 Step 5 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()