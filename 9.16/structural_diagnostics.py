from __future__ import annotations

from pathlib import Path
import json
import math
import os

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

import matplotlib.pyplot as plt


# ============================================================
# 0. Paths
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)


# ------------------------------------------------------------
# Day 4 Step 2
# ------------------------------------------------------------

STEP2_DIR = (
    OUTPUT_ROOT
    / "M1_day4"
    / "02_step2_matched_networks"
)

EDGE_ROOT = (
    STEP2_DIR
    / "edge_master"
)

NODE_ROOT = (
    STEP2_DIR
    / "node_universe"
)

MATCHED_SUMMARY_PATH = (
    STEP2_DIR
    / "matched_density_network_summary.csv"
)

STEP2_METADATA_PATH = (
    STEP2_DIR
    / "step2_network_construction_metadata.json"
)

STOCK_MAPPING_PATH = (
    STEP2_DIR
    / "stock_master_mapping.parquet"
)


# ------------------------------------------------------------
# Step 3 output
# ------------------------------------------------------------

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day4"
    / "03_step3_structural_diagnostics"
)

NODE_METRIC_ROOT = (
    OUTPUT_DIR
    / "node_metrics"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NODE_METRIC_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. Analysis Settings
# ============================================================

# ------------------------------------------------------------
# Step 3 正式只分析 Step 1 冻结的 primary density。
#
# 默认从 Step 2 metadata 读取，
# 正常应为 0.01 = 1%。
#
# Step 6 再正式分析：
#
# 0.2%, 0.5%, 1%, 2%
# ------------------------------------------------------------

USE_PRIMARY_DENSITY_ONLY = True


# ------------------------------------------------------------
# 如果希望临时指定 q，可设置为例如：
#
# OVERRIDE_DENSITY = 0.01
#
# 正式运行建议保持 None。
# ------------------------------------------------------------

OVERRIDE_DENSITY = None


# ------------------------------------------------------------
# Exact clustering:
#
# 需要 sparse matrix multiplication。
#
# 对 q=1% 的约 5000 节点网络通常可行，
# 但它是 Step 3 中计算最重的一项。
#
# 第一次 smoke test 可设 False。
# 正式结果建议 True。
# ------------------------------------------------------------

COMPUTE_EXACT_CLUSTERING = True


# ------------------------------------------------------------
# Resume:
#
# Node-level metrics 已存在时直接读取。
# ------------------------------------------------------------

RESUME_NODE_METRICS = True


NETWORK_IDS = [

    "B0_RAW_POSITIVE",

    "M1_RESIDUAL_POSITIVE",
]


# ============================================================
# 2. Load Step 2 Metadata
# ============================================================

def load_step2_metadata():

    if not STEP2_METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Step 2 metadata 不存在：\n"
            f"{STEP2_METADATA_PATH}"
        )

    with open(
        STEP2_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    return metadata


# ============================================================
# 3. Load Matched-density Summary
# ============================================================

def load_matched_summary():

    if not MATCHED_SUMMARY_PATH.exists():

        raise FileNotFoundError(
            f"不存在：\n"
            f"{MATCHED_SUMMARY_PATH}"
        )

    df = pd.read_csv(
        MATCHED_SUMMARY_PATH
    )

    required = [

        "analysis_date",
        "window",
        "density_fraction",
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
            f"matched_density_network_summary "
            f"缺字段：{missing}"
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
    ] = (
        pd.to_numeric(
            df[
                "window"
            ],
            errors="raise",
        )
        .astype(int)
    )

    df[
        "density_fraction"
    ] = pd.to_numeric(
        df[
            "density_fraction"
        ],
        errors="raise",
    )

    return (
        df.sort_values(
            [
                "window",
                "density_fraction",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 4. Determine Step 3 Density
# ============================================================

def resolve_analysis_density(
    metadata,
):

    if (
        OVERRIDE_DENSITY
        is not None
    ):

        return float(
            OVERRIDE_DENSITY
        )

    return float(
        metadata[
            "primary_density"
        ]
    )


# ============================================================
# 5. Utility: Gini
# ============================================================

def gini_coefficient(
    values,
):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:

        return np.nan

    if np.any(
        x < 0
    ):

        raise ValueError(
            "Gini input must be nonnegative."
        )

    total = float(
        x.sum()
    )

    if total <= 0:

        return 0.0

    x = np.sort(
        x
    )

    n = len(
        x
    )

    index = np.arange(
        1,
        n + 1,
        dtype=np.float64,
    )

    gini = (

        2.0
        *
        np.sum(
            index
            *
            x
        )
        /
        (
            n
            *
            total
        )

        -

        (
            n + 1
        )
        /
        n
    )

    return float(
        gini
    )


# ============================================================
# 6. Utility: Top-x% Share
# ============================================================

def top_fraction_share(
    values,
    fraction,
):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    n = len(
        x
    )

    if n == 0:

        return np.nan

    total = float(
        x.sum()
    )

    if total <= 0:

        return 0.0

    k = max(
        1,
        int(
            math.ceil(
                fraction
                *
                n
            )
        ),
    )

    top_values = np.partition(
        x,
        n - k,
    )[
        n - k:
    ]

    return float(
        top_values.sum()
        /
        total
    )


# ============================================================
# 7. Utility: HHI
# ============================================================

def concentration_hhi(
    values,
):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = np.where(
        np.isfinite(x),
        x,
        0.0,
    )

    total = float(
        x.sum()
    )

    if total <= 0:

        return 0.0

    share = (
        x
        /
        total
    )

    return float(
        np.sum(
            share ** 2
        )
    )


# ============================================================
# 8. Load One Node Universe
# ============================================================

def load_nodes(
    date,
    window,
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
        NODE_ROOT
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Node file 不存在：\n"
            f"{path}"
        )

    nodes = pd.read_parquet(
        path
    )

    required = [
        "master_index",
        "security_id",
        "stock_code",
        "stock_name",
        "market_beta",
        "market_r2",
    ]

    missing = [

        col
        for col in required
        if col not in nodes.columns
    ]

    if missing:

        raise ValueError(
            f"Node file 缺字段："
            f"{missing}"
        )

    nodes = (
        nodes
        .sort_values(
            "master_index"
        )
        .reset_index(
            drop=True
        )
    )

    return nodes


# ============================================================
# 9. Load One Edge Master and Cut to q
# ============================================================

def load_edges(
    network_id,
    date,
    window,
    edge_budget,
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
        EDGE_ROOT
        /
        network_id
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Edge master 不存在：\n"
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
            f"Edge master 缺字段："
            f"{missing}"
        )

    # --------------------------------------------------------
    # 不使用固定 percentage 对 parquet 长度切片。
    #
    # 必须使用 Step 2 精确 edge_budget_k。
    # --------------------------------------------------------

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

    if len(
        edges
    ) != int(
        edge_budget
    ):

        raise RuntimeError(
            f"{network_id}, "
            f"W={window}, "
            f"{date_string}: "
            f"实际边数={len(edges):,}, "
            f"expected={edge_budget:,}"
        )

    return edges


# ============================================================
# 10. Convert Edges to Local Node Indices
# ============================================================

def map_edges_to_local_indices(
    nodes,
    edges,
):

    master_index = (
        nodes[
            "master_index"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    node_indexer = pd.Index(
        master_index
    )

    u = (
        node_indexer
        .get_indexer(
            edges[
                "master_index_i"
            ]
            .to_numpy(
                dtype=np.int32
            )
        )
    )

    v = (
        node_indexer
        .get_indexer(
            edges[
                "master_index_j"
            ]
            .to_numpy(
                dtype=np.int32
            )
        )
    )

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
            "Some edge endpoints are absent "
            "from node universe."
        )

    return (
        u.astype(
            np.int32
        ),
        v.astype(
            np.int32
        ),
    )


# ============================================================
# 11. Build Sparse Adjacency Matrix
# ============================================================

def build_sparse_adjacency(
    n_nodes,
    u,
    v,
):

    row = np.concatenate(
        [
            u,
            v,
        ]
    )

    col = np.concatenate(
        [
            v,
            u,
        ]
    )

    data = np.ones(
        len(row),
        dtype=np.int8,
    )

    adjacency = sparse.coo_matrix(

        (
            data,
            (
                row,
                col,
            ),
        ),

        shape=(
            n_nodes,
            n_nodes,
        ),

        dtype=np.int8,
    ).tocsr()

    adjacency.sum_duplicates()

    # Simple graph:
    #
    # any accidental duplicate gets collapsed to 1.
    adjacency.data[:] = 1

    return adjacency


# ============================================================
# 12. Exact Clustering / Triangle Diagnostics
# ============================================================

def calculate_clustering_metrics(
    adjacency,
    degree,
):

    if not COMPUTE_EXACT_CLUSTERING:

        return {
            "triangle_count":
                np.nan,

            "transitivity":
                np.nan,

            "mean_local_clustering":
                np.nan,

            "mean_local_clustering_degree_ge2":
                np.nan,
        }

    # --------------------------------------------------------
    # A^2_ij counts common neighbours of i and j.
    #
    # Keep only existing edges:
    #
    # A_ij * (A^2)_ij.
    #
    # For each node i:
    #
    # sum_j A_ij (A^2)_ij
    #
    # equals twice the number of triangles touching i.
    # --------------------------------------------------------

    adjacency_float = (
        adjacency.astype(
            np.float64
        )
    )

    A2 = (
        adjacency_float
        @
        adjacency_float
    )

    common_on_edges = (
        adjacency_float
        .multiply(
            A2
        )
    )

    closed_wedges_per_node = np.asarray(
        common_on_edges.sum(
            axis=1
        )
    ).ravel()

    # --------------------------------------------------------
    # Every triangle contributes:
    #
    # 2 at each of 3 vertices
    #
    # => total contribution = 6.
    # --------------------------------------------------------

    triangle_count = float(
        closed_wedges_per_node.sum()
        /
        6.0
    )

    denominator = (
        degree
        *
        (
            degree - 1
        )
    ).astype(
        np.float64
    )

    local_clustering = np.zeros(
        len(
            degree
        ),
        dtype=np.float64,
    )

    eligible = (
        degree >= 2
    )

    local_clustering[
        eligible
    ] = (
        closed_wedges_per_node[
            eligible
        ]
        /
        denominator[
            eligible
        ]
    )

    # --------------------------------------------------------
    # Global transitivity:
    #
    # 3*T /
    # sum_i C(d_i,2)
    #
    # equivalent to:
    #
    # sum closed_wedges /
    # sum d_i(d_i-1)
    # --------------------------------------------------------

    total_denominator = float(
        denominator.sum()
    )

    transitivity = (

        float(
            closed_wedges_per_node.sum()
            /
            total_denominator
        )

        if total_denominator > 0

        else np.nan
    )

    result = {

        "triangle_count":
            triangle_count,

        "transitivity":
            transitivity,

        "mean_local_clustering":
            float(
                np.mean(
                    local_clustering
                )
            ),

        "mean_local_clustering_degree_ge2":
            (
                float(
                    np.mean(
                        local_clustering[
                            eligible
                        ]
                    )
                )

                if eligible.any()

                else np.nan
            ),
    }

    del A2
    del common_on_edges

    return result


# ============================================================
# 13. Degree Assortativity
# ============================================================

def degree_assortativity(
    degree,
    u,
    v,
):

    if len(
        u
    ) < 2:

        return np.nan

    # --------------------------------------------------------
    # For an undirected graph, include both orientations.
    # --------------------------------------------------------

    x = np.concatenate(
        [
            degree[
                u
            ],
            degree[
                v
            ],
        ]
    ).astype(
        np.float64
    )

    y = np.concatenate(
        [
            degree[
                v
            ],
            degree[
                u
            ],
        ]
    ).astype(
        np.float64
    )

    if (
        np.std(
            x
        ) == 0
        or
        np.std(
            y
        ) == 0
    ):

        return np.nan

    return float(
        np.corrcoef(
            x,
            y,
        )[
            0,
            1
        ]
    )


# ============================================================
# 14. Analyze One Network
# ============================================================

def analyze_one_network(
    nodes,
    edges,
    network_id,
    analysis_date,
    window,
    density,
):

    n_nodes = len(
        nodes
    )

    n_edges = len(
        edges
    )

    (
        u,
        v,
    ) = map_edges_to_local_indices(
        nodes=
            nodes,

        edges=
            edges,
    )

    weights = (
        edges[
            "correlation"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if np.any(
        weights <= 0
    ):

        raise RuntimeError(
            "Step 3 main structural diagnostics "
            "expect positive-edge networks."
        )

    # ========================================================
    # Degree
    # ========================================================

    degree = np.bincount(
        np.concatenate(
            [
                u,
                v,
            ]
        ),
        minlength=
            n_nodes,
    ).astype(
        np.int32
    )

    # ========================================================
    # Strength
    # ========================================================

    strength = np.zeros(
        n_nodes,
        dtype=np.float64,
    )

    np.add.at(
        strength,
        u,
        weights,
    )

    np.add.at(
        strength,
        v,
        weights,
    )

    # ========================================================
    # Sparse graph
    # ========================================================

    adjacency = (
        build_sparse_adjacency(
            n_nodes=
                n_nodes,

            u=
                u,

            v=
                v,
        )
    )

    # ========================================================
    # Components
    # ========================================================

    (
        n_components,
        component_labels,
    ) = connected_components(

        csgraph=
            adjacency,

        directed=False,

        return_labels=True,
    )

    component_sizes = np.bincount(
        component_labels,
        minlength=
            n_components,
    )

    component_order = np.sort(
        component_sizes
    )[
        ::-1
    ]

    giant_size = int(
        component_order[
            0
        ]
    )

    second_largest_size = (

        int(
            component_order[
                1
            ]
        )

        if len(
            component_order
        ) > 1

        else 0
    )

    giant_component_id = int(
        np.argmax(
            component_sizes
        )
    )

    is_giant = (
        component_labels
        ==
        giant_component_id
    )

    # ========================================================
    # Clustering
    # ========================================================

    clustering = (
        calculate_clustering_metrics(
            adjacency=
                adjacency,

            degree=
                degree,
        )
    )

    # ========================================================
    # Concentration
    # ========================================================

    degree_mean = float(
        np.mean(
            degree
        )
    )

    degree_std = float(
        np.std(
            degree,
            ddof=0,
        )
    )

    degree_cv = (

        degree_std
        /
        degree_mean

        if degree_mean > 0

        else np.nan
    )

    strength_mean = float(
        np.mean(
            strength
        )
    )

    strength_std = float(
        np.std(
            strength,
            ddof=0,
        )
    )

    strength_cv = (

        strength_std
        /
        strength_mean

        if strength_mean > 0

        else np.nan
    )

    # ========================================================
    # Degree assortativity
    # ========================================================

    assortativity = (
        degree_assortativity(
            degree=
                degree,

            u=
                u,

            v=
                v,
        )
    )

    # ========================================================
    # Overall structural summary
    # ========================================================

    possible_edges = (
        n_nodes
        *
        (
            n_nodes - 1
        )
        /
        2
    )

    realized_density = (
        n_edges
        /
        possible_edges
        if possible_edges > 0
        else np.nan
    )

    summary = {

        "analysis_date":
            pd.Timestamp(
                analysis_date
            ),

        "window":
            int(
                window
            ),

        "density_fraction":
            float(
                density
            ),

        "network_id":
            network_id,

        # ----------------------------------------------------
        # Basic size
        # ----------------------------------------------------

        "n_nodes":
            int(
                n_nodes
            ),

        "n_edges":
            int(
                n_edges
            ),

        "realized_density":
            float(
                realized_density
            ),

        # ----------------------------------------------------
        # Connectivity
        # ----------------------------------------------------

        "isolated_node_count":
            int(
                np.sum(
                    degree == 0
                )
            ),

        "isolated_node_share":
            float(
                np.mean(
                    degree == 0
                )
            ),

        "n_connected_components":
            int(
                n_components
            ),

        "giant_component_size":
            giant_size,

        "giant_component_share":
            float(
                giant_size
                /
                n_nodes
            ),

        "second_largest_component_size":
            second_largest_size,

        "second_largest_component_share":
            float(
                second_largest_size
                /
                n_nodes
            ),

        # ----------------------------------------------------
        # Degree distribution
        # ----------------------------------------------------

        "degree_mean":
            degree_mean,

        "degree_median":
            float(
                np.median(
                    degree
                )
            ),

        "degree_std":
            degree_std,

        "degree_cv":
            degree_cv,

        "degree_p90":
            float(
                np.quantile(
                    degree,
                    0.90,
                )
            ),

        "degree_p95":
            float(
                np.quantile(
                    degree,
                    0.95,
                )
            ),

        "degree_p99":
            float(
                np.quantile(
                    degree,
                    0.99,
                )
            ),

        "degree_max":
            int(
                degree.max()
            ),

        "degree_gini":
            gini_coefficient(
                degree
            ),

        "degree_hhi":
            concentration_hhi(
                degree
            ),

        "top1pct_degree_share":
            top_fraction_share(
                degree,
                0.01,
            ),

        "top5pct_degree_share":
            top_fraction_share(
                degree,
                0.05,
            ),

        # ----------------------------------------------------
        # Strength
        # ----------------------------------------------------

        "strength_mean":
            strength_mean,

        "strength_median":
            float(
                np.median(
                    strength
                )
            ),

        "strength_std":
            strength_std,

        "strength_cv":
            strength_cv,

        "strength_p95":
            float(
                np.quantile(
                    strength,
                    0.95,
                )
            ),

        "strength_p99":
            float(
                np.quantile(
                    strength,
                    0.99,
                )
            ),

        "strength_max":
            float(
                strength.max()
            ),

        "strength_gini":
            gini_coefficient(
                strength
            ),

        "strength_hhi":
            concentration_hhi(
                strength
            ),

        "top1pct_strength_share":
            top_fraction_share(
                strength,
                0.01,
            ),

        "top5pct_strength_share":
            top_fraction_share(
                strength,
                0.05,
            ),

        # ----------------------------------------------------
        # Mixing
        # ----------------------------------------------------

        "degree_assortativity":
            assortativity,

        # ----------------------------------------------------
        # Clustering
        # ----------------------------------------------------

        **clustering,
    }

    # ========================================================
    # Node-level table
    # ========================================================

    node_metrics = (
        nodes.copy()
    )

    node_metrics[
        "analysis_date"
    ] = pd.Timestamp(
        analysis_date
    )

    node_metrics[
        "window"
    ] = int(
        window
    )

    node_metrics[
        "density_fraction"
    ] = float(
        density
    )

    node_metrics[
        "network_id"
    ] = network_id

    node_metrics[
        "degree"
    ] = degree

    node_metrics[
        "strength"
    ] = strength.astype(
        np.float32
    )

    node_metrics[
        "component_id"
    ] = component_labels.astype(
        np.int32
    )

    node_metrics[
        "component_size"
    ] = (
        component_sizes[
            component_labels
        ]
        .astype(
            np.int32
        )
    )

    node_metrics[
        "is_giant_component"
    ] = is_giant

    # --------------------------------------------------------
    # Degree / strength percentile ranks
    #
    # pct=True:
    # low degree -> small percentile
    # high degree -> percentile near 1.
    # --------------------------------------------------------

    node_metrics[
        "degree_percentile"
    ] = (
        pd.Series(
            degree
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    node_metrics[
        "strength_percentile"
    ] = (
        pd.Series(
            strength
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    return (
        summary,
        node_metrics,
    )


# ============================================================
# 15. Save Node Metrics
# ============================================================

def node_metric_path(
    network_id,
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
        NODE_METRIC_ROOT
        /
        network_id
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def save_node_metrics(
    df,
    path,
):

    temp_path = Path(
        str(path)
        +
        ".tmp.parquet"
    )

    df.to_parquet(
        temp_path,
        index=False,
        compression="zstd",
    )

    os.replace(
        temp_path,
        path,
    )


# ============================================================
# 16. Hub Set
# ============================================================

def top_hub_set(
    node_metrics,
    column,
    fraction,
):

    n = len(
        node_metrics
    )

    k = max(
        1,
        int(
            math.ceil(
                fraction
                *
                n
            )
        ),
    )

    top = (
        node_metrics
        .nlargest(
            k,
            column,
        )[
            "master_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    return set(
        top.tolist()
    )


# ============================================================
# 17. Raw vs Residual Node-Level Comparison
# ============================================================

def compare_raw_residual_nodes(
    raw_nodes,
    residual_nodes,
    analysis_date,
    window,
    density,
):

    raw = (
        raw_nodes[
            [
                "master_index",
                "degree",
                "strength",
                "degree_percentile",
                "strength_percentile",
            ]
        ]
        .rename(
            columns={

                "degree":
                    "degree_raw",

                "strength":
                    "strength_raw",

                "degree_percentile":
                    "degree_pct_raw",

                "strength_percentile":
                    "strength_pct_raw",
            }
        )
    )

    residual = (
        residual_nodes[
            [
                "master_index",
                "degree",
                "strength",
                "degree_percentile",
                "strength_percentile",
            ]
        ]
        .rename(
            columns={

                "degree":
                    "degree_residual",

                "strength":
                    "strength_residual",

                "degree_percentile":
                    "degree_pct_residual",

                "strength_percentile":
                    "strength_pct_residual",
            }
        )
    )

    merged = raw.merge(
        residual,
        on="master_index",
        how="inner",
        validate="one_to_one",
    )

    if len(
        merged
    ) != len(
        raw_nodes
    ):

        raise RuntimeError(
            "Raw / Residual node universes differ."
        )

    # ========================================================
    # Spearman rank stability across network definitions
    # ========================================================

    degree_spearman = spearmanr(
        merged[
            "degree_raw"
        ],
        merged[
            "degree_residual"
        ],
        nan_policy="omit",
    ).statistic

    strength_spearman = spearmanr(
        merged[
            "strength_raw"
        ],
        merged[
            "strength_residual"
        ],
        nan_policy="omit",
    ).statistic

    # ========================================================
    # Rank changes
    # ========================================================

    degree_rank_change = np.abs(

        merged[
            "degree_pct_residual"
        ]

        -

        merged[
            "degree_pct_raw"
        ]
    )

    strength_rank_change = np.abs(

        merged[
            "strength_pct_residual"
        ]

        -

        merged[
            "strength_pct_raw"
        ]
    )

    # ========================================================
    # Hub overlap
    # ========================================================

    rows = {

        "analysis_date":
            pd.Timestamp(
                analysis_date
            ),

        "window":
            int(
                window
            ),

        "density_fraction":
            float(
                density
            ),

        "degree_spearman_raw_residual":
            float(
                degree_spearman
            ),

        "strength_spearman_raw_residual":
            float(
                strength_spearman
            ),

        "median_abs_degree_percentile_change":
            float(
                np.median(
                    degree_rank_change
                )
            ),

        "p90_abs_degree_percentile_change":
            float(
                np.quantile(
                    degree_rank_change,
                    0.90,
                )
            ),

        "median_abs_strength_percentile_change":
            float(
                np.median(
                    strength_rank_change
                )
            ),

        "p90_abs_strength_percentile_change":
            float(
                np.quantile(
                    strength_rank_change,
                    0.90,
                )
            ),
    }

    for fraction, label in [

        (
            0.01,
            "top1pct",
        ),

        (
            0.05,
            "top5pct",
        ),
    ]:

        raw_degree_hubs = (
            top_hub_set(
                raw_nodes,
                "degree",
                fraction,
            )
        )

        residual_degree_hubs = (
            top_hub_set(
                residual_nodes,
                "degree",
                fraction,
            )
        )

        degree_intersection = len(

            raw_degree_hubs

            &

            residual_degree_hubs
        )

        degree_union = len(

            raw_degree_hubs

            |

            residual_degree_hubs
        )

        rows[
            f"{label}_degree_hub_overlap"
        ] = (

            degree_intersection
            /
            len(
                raw_degree_hubs
            )

            if raw_degree_hubs

            else np.nan
        )

        rows[
            f"{label}_degree_hub_jaccard"
        ] = (

            degree_intersection
            /
            degree_union

            if degree_union > 0

            else np.nan
        )

        raw_strength_hubs = (
            top_hub_set(
                raw_nodes,
                "strength",
                fraction,
            )
        )

        residual_strength_hubs = (
            top_hub_set(
                residual_nodes,
                "strength",
                fraction,
            )
        )

        strength_intersection = len(

            raw_strength_hubs

            &

            residual_strength_hubs
        )

        strength_union = len(

            raw_strength_hubs

            |

            residual_strength_hubs
        )

        rows[
            f"{label}_strength_hub_overlap"
        ] = (

            strength_intersection
            /
            len(
                raw_strength_hubs
            )

            if raw_strength_hubs

            else np.nan
        )

        rows[
            f"{label}_strength_hub_jaccard"
        ] = (

            strength_intersection
            /
            strength_union

            if strength_union > 0

            else np.nan
        )

    return rows


# ============================================================
# 18. Aggregate Structural Diagnostics
# ============================================================

def summarize_structural_diagnostics(
    detail,
):

    numeric_metrics = [

        "n_nodes",
        "n_edges",

        "isolated_node_share",
        "n_connected_components",

        "giant_component_share",
        "second_largest_component_share",

        "degree_median",
        "degree_cv",
        "degree_p95",
        "degree_p99",
        "degree_max",

        "degree_gini",
        "degree_hhi",

        "top1pct_degree_share",
        "top5pct_degree_share",

        "strength_median",
        "strength_cv",
        "strength_p95",
        "strength_p99",
        "strength_max",

        "strength_gini",
        "strength_hhi",

        "top1pct_strength_share",
        "top5pct_strength_share",

        "degree_assortativity",

        "transitivity",
        "mean_local_clustering",
    ]

    agg_dict = {

        metric:
            [
                "mean",
                "median",
            ]

        for metric in
        numeric_metrics
    }

    summary = (
        detail.groupby(
            [
                "window",
                "network_id",
            ]
        )
        .agg(
            agg_dict
        )
    )

    summary.columns = [

        f"{metric}_{stat}"

        for metric, stat
        in summary.columns
    ]

    summary = (
        summary
        .reset_index()
    )

    return summary


# ============================================================
# 19. Aggregate Raw vs Residual Comparison
# ============================================================

def summarize_raw_residual_comparison(
    comparison,
):

    metrics = [

        "degree_spearman_raw_residual",

        "strength_spearman_raw_residual",

        "median_abs_degree_percentile_change",

        "p90_abs_degree_percentile_change",

        "median_abs_strength_percentile_change",

        "p90_abs_strength_percentile_change",

        "top1pct_degree_hub_overlap",
        "top5pct_degree_hub_overlap",

        "top1pct_strength_hub_overlap",
        "top5pct_strength_hub_overlap",
    ]

    agg_dict = {

        metric:
            [
                "mean",
                "median",
            ]

        for metric in
        metrics
    }

    summary = (
        comparison.groupby(
            "window"
        )
        .agg(
            agg_dict
        )
    )

    summary.columns = [

        f"{metric}_{stat}"

        for metric, stat
        in summary.columns
    ]

    return (
        summary
        .reset_index()
    )


# ============================================================
# 20. Raw - Residual Difference Table
# ============================================================

def build_raw_residual_structural_differences(
    detail,
):

    metrics = [

        "isolated_node_share",
        "n_connected_components",
        "giant_component_share",

        "degree_cv",
        "degree_gini",
        "degree_hhi",

        "top1pct_degree_share",
        "top5pct_degree_share",

        "strength_cv",
        "strength_gini",
        "strength_hhi",

        "top1pct_strength_share",
        "top5pct_strength_share",

        "degree_assortativity",

        "transitivity",
        "mean_local_clustering",
    ]

    raw = (

        detail[
            detail[
                "network_id"
            ]
            .eq(
                "B0_RAW_POSITIVE"
            )
        ][
            [
                "analysis_date",
                "window",
            ]
            +
            metrics
        ]
        .copy()
    )

    residual = (

        detail[
            detail[
                "network_id"
            ]
            .eq(
                "M1_RESIDUAL_POSITIVE"
            )
        ][
            [
                "analysis_date",
                "window",
            ]
            +
            metrics
        ]
        .copy()
    )

    raw = raw.rename(
        columns={

            metric:
                f"{metric}_raw"

            for metric in
            metrics
        }
    )

    residual = residual.rename(
        columns={

            metric:
                f"{metric}_residual"

            for metric in
            metrics
        }
    )

    merged = raw.merge(

        residual,

        on=[
            "analysis_date",
            "window",
        ],

        how="inner",

        validate="one_to_one",
    )

    for metric in metrics:

        merged[
            f"{metric}_residual_minus_raw"
        ] = (

            merged[
                f"{metric}_residual"
            ]

            -

            merged[
                f"{metric}_raw"
            ]
        )

    return merged


# ============================================================
# 21. Difference Window Summary
# ============================================================

def summarize_structural_differences(
    difference_df,
):

    difference_columns = [

        col

        for col in difference_df.columns

        if col.endswith(
            "_residual_minus_raw"
        )
    ]

    rows = []

    for window, group in (
        difference_df.groupby(
            "window"
        )
    ):

        row = {

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
        }

        for col in (
            difference_columns
        ):

            values = (
                group[
                    col
                ]
                .dropna()
            )

            row[
                f"{col}_mean"
            ] = (
                float(
                    values.mean()
                )
                if len(
                    values
                ) > 0
                else np.nan
            )

            row[
                f"{col}_median"
            ] = (
                float(
                    values.median()
                )
                if len(
                    values
                ) > 0
                else np.nan
            )

            row[
                f"{col}_share_positive"
            ] = (
                float(
                    (
                        values
                        >
                        0
                    )
                    .mean()
                )
                if len(
                    values
                ) > 0
                else np.nan
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 22. Plot Time Series
# ============================================================

def plot_metric_over_time(
    detail,
    metric,
    ylabel,
    filename_prefix,
):

    for window in sorted(
        detail[
            "window"
        ]
        .unique()
    ):

        subset = (

            detail[
                detail[
                    "window"
                ]
                .eq(
                    window
                )
            ]
            .sort_values(
                "analysis_date"
            )
        )

        fig, ax = plt.subplots(
            figsize=(
                11,
                5.5,
            )
        )

        for network_id, group in (
            subset.groupby(
                "network_id"
            )
        ):

            label = (

                "Raw"

                if network_id
                ==
                "B0_RAW_POSITIVE"

                else
                "Market-Residual"
            )

            ax.plot(

                group[
                    "analysis_date"
                ],

                group[
                    metric
                ],

                label=
                    label,
            )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            f"{ylabel} | W={window}"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(

            FIGURE_DIR
            /
            (
                f"{filename_prefix}"
                f"_W{window}.png"
            ),

            dpi=180,
        )

        plt.close(
            fig
        )


# ============================================================
# 23. Plot Raw vs Residual Node Rank Similarity
# ============================================================

def plot_node_rank_similarity(
    comparison,
):

    for metric, ylabel, filename in [

        (
            "degree_spearman_raw_residual",
            "Raw-Residual Degree Rank Spearman",
            "raw_residual_degree_rank_spearman",
        ),

        (
            "strength_spearman_raw_residual",
            "Raw-Residual Strength Rank Spearman",
            "raw_residual_strength_rank_spearman",
        ),
    ]:

        fig, ax = plt.subplots(
            figsize=(
                11,
                5.5,
            )
        )

        for window, group in (

            comparison
            .sort_values(
                "analysis_date"
            )
            .groupby(
                "window"
            )
        ):

            ax.plot(

                group[
                    "analysis_date"
                ],

                group[
                    metric
                ],

                label=
                    f"W={window}",
            )

        ax.set_xlabel(
            "Analysis Date"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            ylabel
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(

            FIGURE_DIR
            /
            f"{filename}.png",

            dpi=180,
        )

        plt.close(
            fig
        )


# ============================================================
# 24. Metadata
# ============================================================

def save_metadata(
    density,
    metadata,
):

    payload = {

        "research_day":
            4,

        "step":
            (
                "Step3_Structural_Diagnostics"
            ),

        "input_step":
            str(
                STEP2_DIR
            ),

        "output_directory":
            str(
                OUTPUT_DIR
            ),

        "density_fraction":
            float(
                density
            ),

        "density_percent":
            float(
                100
                *
                density
            ),

        "networks": {

            "raw":
                "B0_RAW_POSITIVE",

            "residual":
                "M1_RESIDUAL_POSITIVE",
        },

        "structural_metric_groups": [

            "connectivity",

            "degree distribution",

            "strength distribution",

            "degree/strength concentration",

            "degree assortativity",

            "clustering / transitivity",

            "Raw-vs-Residual hub re-ranking",
        ],

        "important_notes": [

            (
                "Raw and Residual networks have matched "
                "node support and edge budgets by construction."
            ),

            (
                "Mean degree is therefore largely mechanical "
                "and is not treated as a substantive "
                "Raw-versus-Residual result."
            ),

            (
                "Step 3 formally studies the primary density only. "
                "Density robustness remains Step 6."
            ),

            (
                "Industry/community validation is intentionally "
                "left to Step 4."
            ),

            (
                "Adjacent-month temporal stability is intentionally "
                "left to Step 5."
            ),
        ],

        "exact_clustering":
            bool(
                COMPUTE_EXACT_CLUSTERING
            ),
    }

    with open(
        OUTPUT_DIR
        /
        "step3_structural_diagnostics_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 25. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 4 - Step 3"
    )

    print(
        "Structural Diagnostics"
    )

    print("=" * 80)

    # ========================================================
    # 1. Load config / summary
    # ========================================================

    print()
    print(
        "[1] Load Step 2 outputs"
    )

    metadata = (
        load_step2_metadata()
    )

    matched_summary = (
        load_matched_summary()
    )

    density = (
        resolve_analysis_density(
            metadata
        )
    )

    jobs = (

        matched_summary[

            np.isclose(

                matched_summary[
                    "density_fraction"
                ],

                density,
            )
        ]

        .copy()

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

    if jobs.empty:

        raise RuntimeError(
            f"没有找到 density={density} "
            "对应的 Step 2 网络。"
        )

    print(
        f"Primary density = "
        f"{100*density:.2f}%"
    )

    print(
        f"Date × Window jobs = "
        f"{len(jobs):,}"
    )

    # ========================================================
    # 2. Structural diagnostics
    # ========================================================

    structural_rows = []

    comparison_rows = []

    total_jobs = len(
        jobs
    )

    for job_no, row in enumerate(

        jobs.itertuples(
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

        K = int(
            row.edge_budget_k
        )

        print()
        print("=" * 80)

        print(
            f"Job "
            f"{job_no}/{total_jobs}"
        )

        print(
            f"W={window} | "
            f"Date={date.date()} | "
            f"K={K:,}"
        )

        print("=" * 80)

        nodes = (
            load_nodes(
                date=
                    date,

                window=
                    window,
            )
        )

        node_results = {}

        # ====================================================
        # Raw and Residual
        # ====================================================

        for network_id in (
            NETWORK_IDS
        ):

            print(
                f"  Analyze "
                f"{network_id}"
            )

            edge_df = (
                load_edges(

                    network_id=
                        network_id,

                    date=
                        date,

                    window=
                        window,

                    edge_budget=
                        K,
                )
            )

            output_path = (
                node_metric_path(

                    network_id=
                        network_id,

                    window=
                        window,

                    date=
                        date,
                )
            )

            # -----------------------------------------------
            # Structural summary always recomputed.
            #
            # Node parquet may be overwritten because
            # construction is cheap relative to correlation.
            # -----------------------------------------------

            (
                summary,
                node_metrics,
            ) = analyze_one_network(

                nodes=
                    nodes,

                edges=
                    edge_df,

                network_id=
                    network_id,

                analysis_date=
                    date,

                window=
                    window,

                density=
                    density,
            )

            structural_rows.append(
                summary
            )

            node_results[
                network_id
            ] = node_metrics

            save_node_metrics(
                node_metrics,
                output_path,
            )

        # ====================================================
        # Raw vs Residual node re-ranking
        # ====================================================

        comparison = (
            compare_raw_residual_nodes(

                raw_nodes=
                    node_results[
                        "B0_RAW_POSITIVE"
                    ],

                residual_nodes=
                    node_results[
                        "M1_RESIDUAL_POSITIVE"
                    ],

                analysis_date=
                    date,

                window=
                    window,

                density=
                    density,
            )
        )

        comparison_rows.append(
            comparison
        )

    # ========================================================
    # 3. Save detailed structural results
    # ========================================================

    print()
    print(
        "[3] Save structural diagnostics"
    )

    detail = pd.DataFrame(
        structural_rows
    )

    detail = (
        detail
        .sort_values(
            [
                "window",
                "network_id",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    detail.to_csv(

        OUTPUT_DIR
        /
        "structural_diagnostics_detail.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 4. Window summary
    # ========================================================

    window_summary = (
        summarize_structural_diagnostics(
            detail
        )
    )

    window_summary.to_csv(

        OUTPUT_DIR
        /
        "structural_diagnostics_window_summary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 5. Raw / Residual node comparison
    # ========================================================

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison = (

        comparison
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

    comparison.to_csv(

        OUTPUT_DIR
        /
        "raw_residual_node_comparison.csv",

        index=False,

        encoding="utf-8-sig",
    )

    comparison_summary = (
        summarize_raw_residual_comparison(
            comparison
        )
    )

    comparison_summary.to_csv(

        OUTPUT_DIR
        /
        "raw_residual_node_comparison_window_summary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 6. Direct Raw - Residual differences
    # ========================================================

    difference_df = (
        build_raw_residual_structural_differences(
            detail
        )
    )

    difference_df.to_csv(

        OUTPUT_DIR
        /
        "raw_residual_structural_difference_detail.csv",

        index=False,

        encoding="utf-8-sig",
    )

    difference_summary = (
        summarize_structural_differences(
            difference_df
        )
    )

    difference_summary.to_csv(

        OUTPUT_DIR
        /
        "raw_residual_structural_difference_window_summary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 7. Figures
    # ========================================================

    print()
    print(
        "[4] Generate figures"
    )

    plot_metric_over_time(

        detail=
            detail,

        metric=
            "giant_component_share",

        ylabel=
            "Giant Component Share",

        filename_prefix=
            "giant_component_share",
    )

    plot_metric_over_time(

        detail=
            detail,

        metric=
            "degree_gini",

        ylabel=
            "Degree Gini",

        filename_prefix=
            "degree_gini",
    )

    plot_metric_over_time(

        detail=
            detail,

        metric=
            "top1pct_degree_share",

        ylabel=
            "Top 1% Degree Share",

        filename_prefix=
            "top1pct_degree_share",
    )

    if (
        COMPUTE_EXACT_CLUSTERING
    ):

        plot_metric_over_time(

            detail=
                detail,

            metric=
                "transitivity",

            ylabel=
                "Global Transitivity",

            filename_prefix=
                "global_transitivity",
        )

    plot_node_rank_similarity(
        comparison
    )

    # ========================================================
    # 8. Metadata
    # ========================================================

    save_metadata(
        density=
            density,

        metadata=
            metadata,
    )

    # ========================================================
    # 9. Console output
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Window-level Structural Summary"
    )

    print("=" * 80)

    show_cols = [

        "window",
        "network_id",

        "giant_component_share_mean",

        "isolated_node_share_mean",

        "degree_gini_mean",

        "top1pct_degree_share_mean",

        "strength_gini_mean",

        "transitivity_mean",
    ]

    available = [

        col
        for col in show_cols
        if col in (
            window_summary.columns
        )
    ]

    print(

        window_summary[
            available
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 80)

    print(
        "Raw vs Residual Node Ranking"
    )

    print("=" * 80)

    print(
        comparison_summary
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 80)

    print(
        "Day 4 Step 3 Complete"
    )

    print("=" * 80)

    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()