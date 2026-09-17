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
# Day 4 network outputs
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

RESIDUAL_NETWORK_ID = (
    "M1_RESIDUAL_POSITIVE"
)


# ------------------------------------------------------------
# Day 5
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
# Day 5 Step 3
# ------------------------------------------------------------

STEP3_DIR = (
    DAY5_ROOT
    / "03_step3_industry_decomposition"
)

STEP3_PARTITION_DIR = (
    STEP3_DIR
    / "panel_parts"
)

STEP3_PANEL_PATH = (
    STEP3_DIR
    / "industry_decomposed_feature_panel.parquet"
)

STEP3_METADATA_PATH = (
    STEP3_DIR
    / "step3_industry_decomposition_metadata.json"
)


# ------------------------------------------------------------
# Day 5 Step 4 output
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY5_ROOT
    / "04_step4_dynamic_features"
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
    / "dynamic_network_feature_panel.parquet"
)

JOB_QA_PATH = (
    OUTPUT_DIR
    / "dynamic_feature_job_qa.csv"
)

COVERAGE_PATH = (
    OUTPUT_DIR
    / "dynamic_feature_coverage_summary.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "dynamic_feature_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "step4_dynamic_feature_metadata.json"
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

FLOAT_TOL = 1e-10


# ============================================================
# 2. Frozen Core Dynamic Factors
# ============================================================

REQUIRED_FROZEN_FACTORS = {

    "delta_residual_degree_percentile_1m",

    "delta_residual_strength_percentile_1m",

    "neighbor_jaccard_1m",

    "neighbor_retention_1m",
}


# ============================================================
# 3. Lag-based Features
# ============================================================

# ------------------------------------------------------------
# First four include factors formally defined in Step 1.
#
# Industry-decomposition dynamics are useful extensions of
# Step 3. They are constructed without using any future
# return information.
# ------------------------------------------------------------

LAG_FEATURE_MAP = {

    # ========================================================
    # Overall residual centrality
    # ========================================================

    "residual_degree":
        "delta_residual_degree_1m",

    "residual_strength":
        "delta_residual_strength_1m",

    "residual_degree_percentile":
        "delta_residual_degree_percentile_1m",

    "residual_strength_percentile":
        "delta_residual_strength_percentile_1m",

    # ========================================================
    # Same-industry structure
    # ========================================================

    "same_industry_degree":
        "delta_same_industry_degree_1m",

    "same_industry_strength":
        "delta_same_industry_strength_1m",

    "same_industry_degree_percentile":
        "delta_same_industry_degree_percentile_1m",

    "same_industry_strength_percentile":
        "delta_same_industry_strength_percentile_1m",

    # ========================================================
    # Cross-industry structure
    # ========================================================

    "cross_industry_degree":
        "delta_cross_industry_degree_1m",

    "cross_industry_strength":
        "delta_cross_industry_strength_1m",

    "cross_industry_degree_percentile":
        "delta_cross_industry_degree_percentile_1m",

    "cross_industry_strength_percentile":
        "delta_cross_industry_strength_percentile_1m",

    "cross_industry_degree_ratio":
        "delta_cross_industry_degree_ratio_1m",

    "cross_industry_strength_ratio":
        "delta_cross_industry_strength_ratio_1m",
}


# ============================================================
# 4. Neighbor Dynamic Features
# ============================================================

NEIGHBOR_FEATURE_COLUMNS = [

    # --------------------------------------------------------
    # Main common-node-conditioned metrics
    # --------------------------------------------------------

    "previous_common_neighbor_count_1m",

    "current_common_neighbor_count_1m",

    "retained_neighbor_count_1m",

    "new_neighbor_count_1m",

    "lost_neighbor_count_1m",

    "neighbor_jaccard_1m",

    "neighbor_retention_1m",

    "neighbor_turnover_1m",

    # --------------------------------------------------------
    # Full-neighbor diagnostics
    #
    # These are NOT the preferred main factors because they
    # can be affected by universe entry / exit.
    # --------------------------------------------------------

    "neighbor_jaccard_full_1m",

    "neighbor_retention_full_1m",
]


# ============================================================
# 5. Utility Functions
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


# ============================================================
# 6. Load Frozen Design
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

    design_hash = config.get(
        "design_hash"
    )

    if not design_hash:

        raise RuntimeError(
            "Frozen design has no design_hash."
        )

    design = config[
        "network_design"
    ]

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
            "Window definition mismatch."
        )

    return (
        config,
        design_hash,
    )


# ============================================================
# 7. Validate Step 3 Lineage
# ============================================================

def validate_step3_lineage(
    design_hash,
):

    metadata = load_json(
        STEP3_METADATA_PATH
    )

    if (
        metadata.get(
            "design_hash"
        )
        !=
        design_hash
    ):

        raise RuntimeError(
            "Step 3 panel was generated from "
            "a different frozen design."
        )

    qa = metadata.get(
        "formal_qa",
        {},
    )

    if not qa.get(
        "all_jobs_pass",
        False,
    ):

        raise RuntimeError(
            "Step 3 did not pass all formal QA."
        )

    return metadata


# ============================================================
# 8. Validate Frozen Factor Dictionary
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
# 9. Load Analysis Dates
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

        "previous_analysis_date",
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
        "previous_analysis_date"
    ] = pd.to_datetime(
        df[
            "previous_analysis_date"
        ],
        errors="coerce",
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
# 10. Load Step 3 Panel Partition
# ============================================================

def panel_partition_path(
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
        STEP3_PARTITION_DIR
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )


def load_panel_partition(
    window,
    date,
):

    path = panel_partition_path(
        window,
        date,
    )

    if not path.exists():

        raise FileNotFoundError(
            "Step 3 panel partition not found:\n"
            f"{path}"
        )

    df = pd.read_parquet(
        path
    )

    required = [

        "analysis_date",
        "window",
        "master_index",

        "residual_degree",
        "residual_strength",

        "residual_degree_percentile",
        "residual_strength_percentile",

        "same_industry_degree",
        "same_industry_strength",

        "same_industry_degree_percentile",
        "same_industry_strength_percentile",

        "cross_industry_degree",
        "cross_industry_strength",

        "cross_industry_degree_percentile",
        "cross_industry_strength_percentile",

        "cross_industry_degree_ratio",
        "cross_industry_strength_ratio",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Step 3 panel missing columns:\n{missing}"
        )

    df[
        "analysis_date"
    ] = pd.to_datetime(
        df[
            "analysis_date"
        ]
    ).astype("datetime64[ns]")

    if (
        df[
            "master_index"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate master_index in Step 3 panel."
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
# 11. Load Residual Edge Master
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
            f"Residual edge master missing: {missing}"
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
            "Selected residual edge count mismatch.\n"
            f"Expected: {edge_budget:,}\n"
            f"Actual:   {len(edges):,}"
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
            "Self-loop found in Residual edge master."
        )

    return edges


# ============================================================
# 12. Initialize Dynamic Columns
# ============================================================

def initialize_dynamic_columns(
    panel,
):

    result = (
        panel.copy()
    )

    # Use one timestamp unit in every partition and the streaming writer.
    result["analysis_date"] = pd.to_datetime(
        result["analysis_date"], errors="raise"
    ).astype("datetime64[ns]")

    # --------------------------------------------------------
    # Temporal identifiers
    # --------------------------------------------------------

    result[
        "previous_analysis_date"
    ] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns]",
    )

    result[
        "dynamic_comparable"
    ] = False

    # --------------------------------------------------------
    # Lag-difference columns
    # --------------------------------------------------------

    for output_col in (
        LAG_FEATURE_MAP.values()
    ):

        result[
            output_col
        ] = np.full(
            len(
                result
            ),
            np.nan,
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # Neighbor-dynamic columns
    # --------------------------------------------------------

    for col in NEIGHBOR_FEATURE_COLUMNS:

        result[
            col
        ] = np.full(
            len(
                result
            ),
            np.nan,
            dtype=np.float64,
        )

    return result


# ============================================================
# 13. Add Lagged Node-level Changes
# ============================================================

def add_lag_changes(
    current,
    previous,
    previous_date,
):

    result = (
        initialize_dynamic_columns(
            current
        )
    )

    if previous is None:

        return result

    result[
        "previous_analysis_date"
    ] = pd.Series(
        pd.Timestamp(previous_date),
        index=result.index,
        dtype="datetime64[ns]",
    )

    current_ids = (
        result[
            "master_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    previous_index = (
        previous
        .set_index(
            "master_index",
            drop=False,
        )
    )

    previous_id_set = set(
        previous_index.index
        .astype(int)
        .tolist()
    )

    comparable = np.array(
        [
            int(x)
            in
            previous_id_set
            for x in current_ids
        ],
        dtype=bool,
    )

    result[
        "dynamic_comparable"
    ] = comparable

    # ========================================================
    # Current - Previous
    # ========================================================

    for (
        source_col,
        output_col,
    ) in LAG_FEATURE_MAP.items():

        previous_values = (
            previous_index[
                source_col
            ]
            .reindex(
                current_ids
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        current_values = (
            pd.to_numeric(
                result[
                    source_col
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        delta = (
            current_values
            -
            previous_values
        )

        # ----------------------------------------------------
        # No previous observation -> undefined
        # ----------------------------------------------------

        delta[
            ~comparable
        ] = np.nan

        result[
            output_col
        ] = delta

    return result


# ============================================================
# 14. Build Adjacency
# ============================================================

def build_adjacency(
    edges,
    allowed_nodes=None,
):

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

    # --------------------------------------------------------
    # Common-node filtering
    # --------------------------------------------------------

    if allowed_nodes is not None:

        allowed_nodes = np.asarray(
            allowed_nodes,
            dtype=np.int64,
        )

        if len(
            allowed_nodes
        ) == 0:

            return (
                {},
                0,
            )

        maximum = int(
            max(
                edge_i.max(
                    initial=0
                ),
                edge_j.max(
                    initial=0
                ),
                allowed_nodes.max(
                    initial=0
                ),
            )
        )

        lookup = np.zeros(
            maximum + 1,
            dtype=bool,
        )

        lookup[
            allowed_nodes
        ] = True

        mask = (
            lookup[
                edge_i
            ]
            &
            lookup[
                edge_j
            ]
        )

        edge_i = edge_i[
            mask
        ]

        edge_j = edge_j[
            mask
        ]

    adjacency = {}

    for i, j in zip(
        edge_i,
        edge_j,
    ):

        i = int(
            i
        )

        j = int(
            j
        )

        adjacency.setdefault(
            i,
            set(),
        ).add(
            j
        )

        adjacency.setdefault(
            j,
            set(),
        ).add(
            i
        )

    return (
        adjacency,
        len(
            edge_i
        ),
    )


# ============================================================
# 15. Compute Neighbor Dynamic Metrics
# ============================================================

def add_neighbor_dynamics(
    panel,
    current_edges,
    previous_edges,
    previous_panel,
):

    if (
        previous_panel is None
        or
        previous_edges is None
    ):

        diagnostics = {

            "common_node_count":
                0,

            "current_common_edge_count":
                0,

            "previous_common_edge_count":
                0,
        }

        return (
            panel,
            diagnostics,
        )

    result = (
        panel.copy()
    )

    current_ids = (
        result[
            "master_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    previous_ids = (
        previous_panel[
            "master_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    common_ids = np.intersect1d(
        current_ids,
        previous_ids,
        assume_unique=True,
    )

    common_set = set(
        common_ids.tolist()
    )

    # ========================================================
    # Full adjacency
    #
    # Auxiliary diagnostic only.
    # ========================================================

    current_full_adj, _ = (
        build_adjacency(
            current_edges
        )
    )

    previous_full_adj, _ = (
        build_adjacency(
            previous_edges
        )
    )

    # ========================================================
    # Main common-node-conditioned adjacency
    #
    # Same principle as Day-4 Step 5:
    # filter both networks to nodes observable at t-1 and t.
    #
    # We DO NOT re-rank / re-normalize edges after filtering.
    # ========================================================

    (
        current_common_adj,
        current_common_edge_count,
    ) = build_adjacency(
        current_edges,
        allowed_nodes=common_ids,
    )

    (
        previous_common_adj,
        previous_common_edge_count,
    ) = build_adjacency(
        previous_edges,
        allowed_nodes=common_ids,
    )

    # ========================================================
    # Local row mapping
    # ========================================================

    local_row = {

        int(
            master_index
        ):
            row_no

        for row_no, master_index
        in enumerate(
            current_ids
        )
    }

    n = len(
        result
    )

    previous_common_degree = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    current_common_degree = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    retained_count = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    new_count = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    lost_count = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    jaccard = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    retention = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    turnover = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    full_jaccard = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    full_retention = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    # ========================================================
    # Per-node neighbor comparison
    # ========================================================

    for master_index in common_ids:

        master_index = int(
            master_index
        )

        row_no = (
            local_row[
                master_index
            ]
        )

        # ----------------------------------------------------
        # Common-node-conditioned neighbors
        # ----------------------------------------------------

        prev_neighbors = (
            previous_common_adj.get(
                master_index,
                set(),
            )
        )

        curr_neighbors = (
            current_common_adj.get(
                master_index,
                set(),
            )
        )

        intersection = (
            prev_neighbors
            &
            curr_neighbors
        )

        new_neighbors = (
            curr_neighbors
            -
            prev_neighbors
        )

        lost_neighbors = (
            prev_neighbors
            -
            curr_neighbors
        )

        union = (
            prev_neighbors
            |
            curr_neighbors
        )

        previous_common_degree[
            row_no
        ] = len(
            prev_neighbors
        )

        current_common_degree[
            row_no
        ] = len(
            curr_neighbors
        )

        retained_count[
            row_no
        ] = len(
            intersection
        )

        new_count[
            row_no
        ] = len(
            new_neighbors
        )

        lost_count[
            row_no
        ] = len(
            lost_neighbors
        )

        # ----------------------------------------------------
        # If both sets are empty, Jaccard is undefined rather
        # than mechanically set to 1.
        # ----------------------------------------------------

        if len(
            union
        ) > 0:

            jaccard[
                row_no
            ] = (
                len(
                    intersection
                )
                /
                len(
                    union
                )
            )

            turnover[
                row_no
            ] = (
                (
                    len(
                        new_neighbors
                    )
                    +
                    len(
                        lost_neighbors
                    )
                )
                /
                len(
                    union
                )
            )

        # ----------------------------------------------------
        # Retention requires a non-empty previous set.
        # ----------------------------------------------------

        if len(
            prev_neighbors
        ) > 0:

            retention[
                row_no
            ] = (
                len(
                    intersection
                )
                /
                len(
                    prev_neighbors
                )
            )

        # ----------------------------------------------------
        # Full-neighbor diagnostic
        # ----------------------------------------------------

        prev_full = (
            previous_full_adj.get(
                master_index,
                set(),
            )
        )

        curr_full = (
            current_full_adj.get(
                master_index,
                set(),
            )
        )

        full_intersection = (
            prev_full
            &
            curr_full
        )

        full_union = (
            prev_full
            |
            curr_full
        )

        if len(
            full_union
        ) > 0:

            full_jaccard[
                row_no
            ] = (
                len(
                    full_intersection
                )
                /
                len(
                    full_union
                )
            )

        if len(
            prev_full
        ) > 0:

            full_retention[
                row_no
            ] = (
                len(
                    full_intersection
                )
                /
                len(
                    prev_full
                )
            )

    # ========================================================
    # Attach features
    # ========================================================

    result[
        "previous_common_neighbor_count_1m"
    ] = previous_common_degree

    result[
        "current_common_neighbor_count_1m"
    ] = current_common_degree

    result[
        "retained_neighbor_count_1m"
    ] = retained_count

    result[
        "new_neighbor_count_1m"
    ] = new_count

    result[
        "lost_neighbor_count_1m"
    ] = lost_count

    result[
        "neighbor_jaccard_1m"
    ] = jaccard

    result[
        "neighbor_retention_1m"
    ] = retention

    result[
        "neighbor_turnover_1m"
    ] = turnover

    result[
        "neighbor_jaccard_full_1m"
    ] = full_jaccard

    result[
        "neighbor_retention_full_1m"
    ] = full_retention

    diagnostics = {

        "common_node_count":
            int(
                len(
                    common_ids
                )
            ),

        "current_common_edge_count":
            int(
                current_common_edge_count
            ),

        "previous_common_edge_count":
            int(
                previous_common_edge_count
            ),
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
    current_panel,
    previous_panel,
    diagnostics,
    expected_node_count,
    expected_edge_count,
    analysis_date,
    window,
):

    errors = []

    is_first_date = (
        previous_panel is None
    )

    # ========================================================
    # 16.1 Row count
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
    # 16.2 Key
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
    # 16.3 First date behavior
    #
    # No t-1 -> all dynamic quantities must be undefined.
    # ========================================================

    first_date_policy_ok = True

    if is_first_date:

        if (
            panel[
                "dynamic_comparable"
            ]
            .any()
        ):

            first_date_policy_ok = False

        dynamic_columns = (
            list(
                LAG_FEATURE_MAP.values()
            )
            +
            NEIGHBOR_FEATURE_COLUMNS
        )

        for col in dynamic_columns:

            if (
                panel[
                    col
                ]
                .notna()
                .any()
            ):

                first_date_policy_ok = False

                break

    if not first_date_policy_ok:

        errors.append(
            "first_date_dynamic_policy_failure"
        )

    # ========================================================
    # The rest applies only to actual transitions
    # ========================================================

    comparable_count = 0

    expected_common_node_count = 0

    dynamic_comparable_share = np.nan

    neighbor_identity_ok = True

    neighbor_jaccard_valid = True

    neighbor_retention_valid = True

    common_degree_handshake_ok = True

    delta_identity_ok = True

    max_delta_identity_error = 0.0

    mean_neighbor_jaccard = np.nan

    median_neighbor_jaccard = np.nan

    mean_neighbor_retention = np.nan

    mean_abs_delta_degree_pct = np.nan

    mean_abs_delta_cross_degree_pct = np.nan

    mean_new_neighbors = np.nan

    mean_lost_neighbors = np.nan

    if not is_first_date:

        current_ids = set(
            panel[
                "master_index"
            ]
            .astype(int)
            .tolist()
        )

        previous_ids = set(
            previous_panel[
                "master_index"
            ]
            .astype(int)
            .tolist()
        )

        expected_common_node_count = int(
            len(
                current_ids
                &
                previous_ids
            )
        )

        comparable_count = int(
            panel[
                "dynamic_comparable"
            ]
            .sum()
        )

        if (
            comparable_count
            !=
            expected_common_node_count
        ):

            errors.append(
                "dynamic_comparable_count_failure"
            )

        dynamic_comparable_share = (
            comparable_count
            /
            len(
                panel
            )
        )

        comparable = (
            panel[
                "dynamic_comparable"
            ]
        )

        # ====================================================
        # 16.4 Neighbor count identities
        #
        # Previous degree:
        # retained + lost
        #
        # Current degree:
        # retained + new
        # ====================================================

        prev_degree = (
            panel.loc[
                comparable,
                "previous_common_neighbor_count_1m"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        curr_degree = (
            panel.loc[
                comparable,
                "current_common_neighbor_count_1m"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        retained = (
            panel.loc[
                comparable,
                "retained_neighbor_count_1m"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        new = (
            panel.loc[
                comparable,
                "new_neighbor_count_1m"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        lost = (
            panel.loc[
                comparable,
                "lost_neighbor_count_1m"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        neighbor_identity_ok = bool(

            np.allclose(
                prev_degree,
                retained + lost,
                atol=0,
                rtol=0,
            )

            and

            np.allclose(
                curr_degree,
                retained + new,
                atol=0,
                rtol=0,
            )
        )

        if not neighbor_identity_ok:

            errors.append(
                "neighbor_count_identity_failure"
            )

        # ====================================================
        # 16.5 Jaccard range
        # ====================================================

        jaccard = (
            panel[
                "neighbor_jaccard_1m"
            ]
            .dropna()
        )

        if len(
            jaccard
        ) > 0:

            neighbor_jaccard_valid = bool(

                (
                    jaccard
                    >=
                    -FLOAT_TOL
                )
                .all()

                and

                (
                    jaccard
                    <=
                    1
                    +
                    FLOAT_TOL
                )
                .all()
            )

        if not neighbor_jaccard_valid:

            errors.append(
                "invalid_neighbor_jaccard"
            )

        # ====================================================
        # 16.6 Retention range
        # ====================================================

        retention = (
            panel[
                "neighbor_retention_1m"
            ]
            .dropna()
        )

        if len(
            retention
        ) > 0:

            neighbor_retention_valid = bool(

                (
                    retention
                    >=
                    -FLOAT_TOL
                )
                .all()

                and

                (
                    retention
                    <=
                    1
                    +
                    FLOAT_TOL
                )
                .all()
            )

        if not neighbor_retention_valid:

            errors.append(
                "invalid_neighbor_retention"
            )

        # ====================================================
        # 16.7 Common-edge handshake
        #
        # Sum common-node degree = 2 × common edge count
        # ====================================================

        current_common_degree_sum = int(
            np.nansum(
                panel[
                    "current_common_neighbor_count_1m"
                ]
            )
        )

        previous_common_degree_sum = int(
            np.nansum(
                panel[
                    "previous_common_neighbor_count_1m"
                ]
            )
        )

        common_degree_handshake_ok = bool(

            (
                current_common_degree_sum
                ==
                2
                *
                diagnostics[
                    "current_common_edge_count"
                ]
            )

            and

            (
                previous_common_degree_sum
                ==
                2
                *
                diagnostics[
                    "previous_common_edge_count"
                ]
            )
        )

        if not common_degree_handshake_ok:

            errors.append(
                "common_degree_handshake_failure"
            )

        # ====================================================
        # 16.8 Lag-delta identity
        # ====================================================

        previous_index = (
            previous_panel
            .set_index(
                "master_index"
            )
        )

        current_ids_array = (
            panel[
                "master_index"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        all_errors = []

        for (
            source_col,
            output_col,
        ) in LAG_FEATURE_MAP.items():

            previous_values = (
                previous_index[
                    source_col
                ]
                .reindex(
                    current_ids_array
                )
                .to_numpy(
                    dtype=np.float64
                )
            )

            current_values = (
                pd.to_numeric(
                    panel[
                        source_col
                    ],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=np.float64
                )
            )

            observed_delta = (
                pd.to_numeric(
                    panel[
                        output_col
                    ],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=np.float64
                )
            )

            expected_delta = (
                current_values
                -
                previous_values
            )

            valid = (

                panel[
                    "dynamic_comparable"
                ]
                .to_numpy(
                    dtype=bool
                )

                &

                np.isfinite(
                    expected_delta
                )

                &

                np.isfinite(
                    observed_delta
                )
            )

            if (
                valid.sum()
                >
                0
            ):

                diff = np.abs(
                    expected_delta[
                        valid
                    ]
                    -
                    observed_delta[
                        valid
                    ]
                )

                all_errors.append(
                    float(
                        diff.max()
                    )
                )

        if all_errors:

            max_delta_identity_error = max(
                all_errors
            )

        delta_identity_ok = (
            max_delta_identity_error
            <=
            FLOAT_TOL
        )

        if not delta_identity_ok:

            errors.append(
                "lag_delta_identity_failure"
            )

        # ====================================================
        # 16.9 Descriptive quantities
        # ====================================================

        if len(
            jaccard
        ) > 0:

            mean_neighbor_jaccard = float(
                jaccard.mean()
            )

            median_neighbor_jaccard = float(
                jaccard.median()
            )

        if len(
            retention
        ) > 0:

            mean_neighbor_retention = float(
                retention.mean()
            )

        delta_degree_pct = (
            panel.loc[
                comparable,
                "delta_residual_degree_percentile_1m"
            ]
            .dropna()
        )

        if len(
            delta_degree_pct
        ) > 0:

            mean_abs_delta_degree_pct = float(
                delta_degree_pct
                .abs()
                .mean()
            )

        delta_cross_pct = (
            panel.loc[
                comparable,
                "delta_cross_industry_degree_percentile_1m"
            ]
            .dropna()
        )

        if len(
            delta_cross_pct
        ) > 0:

            mean_abs_delta_cross_degree_pct = float(
                delta_cross_pct
                .abs()
                .mean()
            )

        mean_new_neighbors = float(
            np.nanmean(
                panel[
                    "new_neighbor_count_1m"
                ]
            )
        )

        mean_lost_neighbors = float(
            np.nanmean(
                panel[
                    "lost_neighbor_count_1m"
                ]
            )
        )

    # ========================================================
    # 16.10 Final
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

        "is_first_date":
            bool(
                is_first_date
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

        "expected_common_node_count":
            int(
                expected_common_node_count
            ),

        "dynamic_comparable_count":
            int(
                comparable_count
            ),

        "dynamic_comparable_share":
            dynamic_comparable_share,

        "current_common_edge_count":
            int(
                diagnostics[
                    "current_common_edge_count"
                ]
            ),

        "previous_common_edge_count":
            int(
                diagnostics[
                    "previous_common_edge_count"
                ]
            ),

        # ----------------------------------------------------
        # Descriptive dynamic statistics
        # ----------------------------------------------------

        "mean_neighbor_jaccard":
            mean_neighbor_jaccard,

        "median_neighbor_jaccard":
            median_neighbor_jaccard,

        "mean_neighbor_retention":
            mean_neighbor_retention,

        "mean_abs_delta_residual_degree_percentile":
            mean_abs_delta_degree_pct,

        "mean_abs_delta_cross_industry_degree_percentile":
            mean_abs_delta_cross_degree_pct,

        "mean_new_neighbor_count":
            mean_new_neighbors,

        "mean_lost_neighbor_count":
            mean_lost_neighbors,

        # ----------------------------------------------------
        # Formal QA
        # ----------------------------------------------------

        "node_count_ok":
            node_count_ok,

        "key_unique":
            key_unique,

        "first_date_policy_ok":
            first_date_policy_ok,

        "neighbor_identity_ok":
            neighbor_identity_ok,

        "neighbor_jaccard_valid":
            neighbor_jaccard_valid,

        "neighbor_retention_valid":
            neighbor_retention_valid,

        "common_degree_handshake_ok":
            common_degree_handshake_ok,

        "delta_identity_ok":
            delta_identity_ok,

        "max_delta_identity_error":
            float(
                max_delta_identity_error
            ),

        "qa_pass":
            qa_pass,

        "qa_errors":
            "|".join(
                errors
            ),
    }


# ============================================================
# 17. Save One Partition
# ============================================================

def save_job_partition(
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
# 18. Dynamic Feature Summary
# ============================================================

SUMMARY_FEATURES = [

    "delta_residual_degree_percentile_1m",

    "delta_residual_strength_percentile_1m",

    "delta_same_industry_degree_percentile_1m",

    "delta_cross_industry_degree_percentile_1m",

    "delta_cross_industry_degree_ratio_1m",

    "neighbor_jaccard_1m",

    "neighbor_retention_1m",

    "neighbor_turnover_1m",

    "new_neighbor_count_1m",

    "lost_neighbor_count_1m",

    "retained_neighbor_count_1m",
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

            quantiles = (
                valid.quantile(
                    [
                        0.01,
                        0.05,
                        0.50,
                        0.95,
                        0.99,
                    ]
                )
            )

            mean_value = float(
                valid.mean()
            )

            std_value = float(
                valid.std(
                    ddof=0
                )
            )

            p01 = float(
                quantiles.loc[
                    0.01
                ]
            )

            p05 = float(
                quantiles.loc[
                    0.05
                ]
            )

            median = float(
                quantiles.loc[
                    0.50
                ]
            )

            p95 = float(
                quantiles.loc[
                    0.95
                ]
            )

            p99 = float(
                quantiles.loc[
                    0.99
                ]
            )

        else:

            mean_value = np.nan
            std_value = np.nan

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

        transitions = (
            group[
                ~group[
                    "is_first_date"
                ]
            ]
        )

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

                "transition_count":
                    int(
                        len(
                            transitions
                        )
                    ),

                "qa_pass_share":
                    float(
                        group[
                            "qa_pass"
                        ]
                        .mean()
                    ),

                "mean_dynamic_comparable_share":
                    float(
                        transitions[
                            "dynamic_comparable_share"
                        ]
                        .mean()
                    ),

                "mean_neighbor_jaccard":
                    float(
                        transitions[
                            "mean_neighbor_jaccard"
                        ]
                        .mean()
                    ),

                "median_of_monthly_neighbor_jaccard":
                    float(
                        transitions[
                            "median_neighbor_jaccard"
                        ]
                        .median()
                    ),

                "mean_neighbor_retention":
                    float(
                        transitions[
                            "mean_neighbor_retention"
                        ]
                        .mean()
                    ),

                "mean_abs_delta_residual_degree_percentile":
                    float(
                        transitions[
                            "mean_abs_delta_residual_degree_percentile"
                        ]
                        .mean()
                    ),

                "mean_abs_delta_cross_industry_degree_percentile":
                    float(
                        transitions[
                            "mean_abs_delta_cross_industry_degree_percentile"
                        ]
                        .mean()
                    ),

                "mean_new_neighbor_count":
                    float(
                        transitions[
                            "mean_new_neighbor_count"
                        ]
                        .mean()
                    ),

                "mean_lost_neighbor_count":
                    float(
                        transitions[
                            "mean_lost_neighbor_count"
                        ]
                        .mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 20. Metadata
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
            "Step4_Dynamic_Network_Features",

        "design_hash":
            design_hash,

        "input_panel":
            str(
                STEP3_PANEL_PATH
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

        "time_definition": {

            "1m":
                (
                    "One adjacent network-analysis interval; "
                    "approximately monthly, not a fixed "
                    "30-calendar-day interval."
                ),

            "first_date_policy":
                (
                    "Dynamic features are NaN on the first "
                    "network date within each window."
                ),
        },

        "main_neighbor_definition": {

            "method":
                "common_node_conditioned",

            "description":
                (
                    "For adjacent dates t-1 and t, both "
                    "selected-edge networks are restricted "
                    "to the intersection of their node "
                    "universes. Existing selected edges are "
                    "filtered only; no re-ranking or "
                    "re-normalization is performed."
                ),

            "reason":
                (
                    "Reduce contamination of network "
                    "rewiring measures by stock-universe "
                    "entry and exit."
                ),
        },

        "core_factor_definitions": {

            "delta_residual_degree_percentile_1m":
                (
                    "ResidualDegreePct_t "
                    "- ResidualDegreePct_t-1"
                ),

            "delta_residual_strength_percentile_1m":
                (
                    "ResidualStrengthPct_t "
                    "- ResidualStrengthPct_t-1"
                ),

            "neighbor_jaccard_1m":
                (
                    "|N_i,t ∩ N_i,t-1| / "
                    "|N_i,t ∪ N_i,t-1| "
                    "on common-node-conditioned networks"
                ),

            "neighbor_retention_1m":
                (
                    "|N_i,t ∩ N_i,t-1| / "
                    "|N_i,t-1| "
                    "on common-node-conditioned networks"
                ),
        },

        "auxiliary_features": [

            "delta_same_industry_degree_percentile_1m",

            "delta_cross_industry_degree_percentile_1m",

            "delta_cross_industry_degree_ratio_1m",

            "new_neighbor_count_1m",

            "lost_neighbor_count_1m",

            "neighbor_turnover_1m",
        ],

        "important_interpretation": [

            (
                "Positive delta centrality means a stock "
                "moves upward in the relevant cross-sectional "
                "network ranking."
            ),

            (
                "Low neighbor_jaccard indicates substantial "
                "neighborhood reorganization; high values "
                "indicate persistent network relationships."
            ),

            (
                "New/lost links are computed after "
                "common-node conditioning for the main "
                "dynamic measures."
            ),

            (
                "Dynamic features use t and t-1 only. "
                "No t+1 return or risk outcome is used."
            ),

            (
                "Higher stability for longer rolling windows "
                "may partly arise mechanically from greater "
                "overlap in the underlying return samples."
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
        "Day 5 - Step 4"
    )

    print(
        "Dynamic Network Features"
    )

    print("=" * 80)

    # ========================================================
    # 21.1 Frozen design / lineage
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

    validate_step3_lineage(
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

    # ========================================================
    # 21.3 Process each window sequentially
    #
    # Keeping t-1 in memory avoids reading it again.
    # ========================================================

    try:

        for window in WINDOWS:

            window_jobs = (
                analysis_dates[
                    analysis_dates[
                        "window"
                    ]
                    ==
                    window
                ]
                .sort_values(
                    "analysis_date"
                )
                .reset_index(
                    drop=True
                )
            )

            previous_panel = None

            previous_edges = None

            previous_date = None

            print()
            print(
                "=" * 80
            )

            print(
                f"Processing W={window}"
            )

            print(
                "=" * 80
            )

            for local_job_no, row in enumerate(

                window_jobs.itertuples(
                    index=False
                ),

                start=1,
            ):

                date = pd.Timestamp(
                    row.analysis_date
                )

                expected_nodes = int(
                    row.common_node_count
                )

                expected_edges = int(
                    row.edge_budget_k
                )

                frozen_previous_date = (
                    pd.Timestamp(
                        row.previous_analysis_date
                    )
                    if pd.notna(
                        row.previous_analysis_date
                    )
                    else None
                )

                # ------------------------------------------------
                # Verify the actual sequential loop matches the
                # Step-1 frozen date mapping.
                # ------------------------------------------------

                if previous_date is None:

                    if (
                        frozen_previous_date
                        is not None
                    ):

                        raise RuntimeError(
                            "First date unexpectedly has "
                            "a frozen previous date."
                        )

                else:

                    if (
                        frozen_previous_date
                        !=
                        previous_date
                    ):

                        raise RuntimeError(
                            "Frozen previous date does not "
                            "match sequential processing.\n"
                            f"W={window}\n"
                            f"Current={date.date()}\n"
                            f"Expected previous="
                            f"{previous_date.date()}\n"
                            f"Frozen previous="
                            f"{frozen_previous_date}"
                        )

                print()
                print(
                    f"[W={window} "
                    f"{local_job_no}/{len(window_jobs)}] "
                    f"{date.date()} | "
                    f"N={expected_nodes:,} | "
                    f"E={expected_edges:,}"
                )

                # =============================================
                # Current Step-3 panel
                # =============================================

                current_panel = (
                    load_panel_partition(
                        window=
                            window,

                        date=
                            date,
                    )
                )

                if (
                    len(
                        current_panel
                    )
                    !=
                    expected_nodes
                ):

                    raise RuntimeError(
                        "Current Step-3 node count mismatch."
                    )

                # =============================================
                # Current residual edges
                # =============================================

                current_edges = (
                    load_residual_edges(

                        window=
                            window,

                        date=
                            date,

                        edge_budget=
                            expected_edges,
                    )
                )

                # =============================================
                # Lag-based stock-level changes
                # =============================================

                dynamic_panel = (
                    add_lag_changes(

                        current=
                            current_panel,

                        previous=
                            previous_panel,

                        previous_date=
                            previous_date,
                    )
                )

                # =============================================
                # Neighbor-set dynamics
                # =============================================

                (
                    dynamic_panel,
                    diagnostics,
                ) = add_neighbor_dynamics(

                    panel=
                        dynamic_panel,

                    current_edges=
                        current_edges,

                    previous_edges=
                        previous_edges,

                    previous_panel=
                        previous_panel,
                )

                # =============================================
                # Formal QA
                # =============================================

                qa = (
                    qa_one_job(

                        panel=
                            dynamic_panel,

                        current_panel=
                            current_panel,

                        previous_panel=
                            previous_panel,

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
                )

                qa_rows.append(
                    qa
                )

                if not qa[
                    "is_first_date"
                ]:

                    print(
                        "  Comparable nodes = "
                        f"{qa['dynamic_comparable_share']:.4f}"
                    )

                    print(
                        "  Mean neighbor Jaccard = "
                        f"{qa['mean_neighbor_jaccard']:.4f}"
                    )

                    print(
                        "  Mean neighbor retention = "
                        f"{qa['mean_neighbor_retention']:.4f}"
                    )

                    print(
                        "  Mean |Delta DegreePct| = "
                        f"{qa['mean_abs_delta_residual_degree_percentile']:.4f}"
                    )

                    print(
                        "  Mean |Delta CrossDegreePct| = "
                        f"{qa['mean_abs_delta_cross_industry_degree_percentile']:.4f}"
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

                # =============================================
                # Descriptive summary
                # =============================================

                summary_rows.extend(
                    summarize_one_panel(
                        dynamic_panel
                    )
                )

                # =============================================
                # Save partition
                # =============================================

                if SAVE_JOB_PARTITIONS:

                    save_job_partition(

                        panel=
                            dynamic_panel,

                        window=
                            window,

                        date=
                            date,
                    )

                # =============================================
                # Stream final parquet
                # =============================================

                table = (
                    pa.Table.from_pandas(
                        dynamic_panel,
                        preserve_index=False,
                    )
                )

                if writer is None:

                    writer = (
                        pq.ParquetWriter(

                            temp_final_path,

                            table.schema,

                            compression="zstd",
                        )
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
                    dynamic_panel
                )

                # =============================================
                # Advance t -> t-1
                # =============================================

                previous_panel = (
                    current_panel
                )

                previous_edges = (
                    current_edges
                )

                previous_date = (
                    date
                )

    finally:

        if writer is not None:

            writer.close()

    # ========================================================
    # 21.4 Finalize parquet
    # ========================================================

    if not temp_final_path.exists():

        raise RuntimeError(
            "Temporary dynamic panel "
            "was not created."
        )

    os.replace(
        temp_final_path,
        FINAL_PANEL_PATH,
    )

    # ========================================================
    # 21.5 QA
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
    # 21.6 Coverage
    # ========================================================

    coverage = (
        build_coverage_summary(
            qa_df
        )
    )

    coverage.to_csv(

        COVERAGE_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 21.7 Dynamic factor summary
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
    # 21.8 Metadata
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
    # 21.9 Console summary
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Dynamic Network Feature Summary"
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
        coverage.to_string(
            index=False
        )
    )

    print()

    print(
        "Final dynamic panel:"
    )

    print(
        FINAL_PANEL_PATH
    )

    print()

    print("=" * 80)

    print(
        "Day 5 Step 4 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()
