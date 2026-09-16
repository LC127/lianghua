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
# Step 2
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


# ------------------------------------------------------------
# Step 6
# ------------------------------------------------------------

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day4"
    / "06_step6_sparsification_robustness"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

TOPK_QA_DIR = (
    OUTPUT_DIR
    / "per_node_topk_qa"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TOPK_QA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. Analysis Settings
# ============================================================

Q_GRID = [
    0.002,
    0.005,
    0.010,
    0.020,
]

PRIMARY_Q = 0.010


# ------------------------------------------------------------
# Step 1 中预定的 per-node Top-k robustness
# ------------------------------------------------------------

TOPK_GRID = [
    10,
    25,
    50,
]


NETWORK_IDS = [
    "B0_RAW_POSITIVE",
    "M1_RESIDUAL_POSITIVE",
]


# ------------------------------------------------------------
# True:
# 仅在现有 Top-2% edge master 可以严格恢复
# per-node Top-k 时才计算。
#
# 若不能证明 exact，则只输出 QA，不伪造结果。
# ------------------------------------------------------------

CHECK_PER_NODE_TOPK = True


# ------------------------------------------------------------
# 调试：
#
# None = 全部日期
# 例如 12 = 每个 W 只取最后 12 个日期
# ------------------------------------------------------------

TEST_LAST_N_DATES = None


# ============================================================
# 2. Load Step 2 Metadata
# ============================================================

def load_metadata():

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

        return json.load(f)


# ============================================================
# 3. Load Matched Density Summary
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
        "common_valid_pair_count",
        "edge_budget_k",
        "raw_edge_threshold",
        "residual_edge_threshold",
        "raw_residual_overlap_fraction",
        "raw_residual_jaccard",
        "common_same_industry_pair_share",
        "raw_industry_enrichment",
        "residual_industry_enrichment",
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

    df["analysis_date"] = pd.to_datetime(
        df["analysis_date"],
        errors="raise",
    )

    df["window"] = pd.to_numeric(
        df["window"],
        errors="raise",
    ).astype(int)

    df["density_fraction"] = pd.to_numeric(
        df["density_fraction"],
        errors="raise",
    )

    df = (
        df[
            df["density_fraction"]
            .isin(Q_GRID)
        ]
        .copy()
        .sort_values(
            [
                "window",
                "density_fraction",
                "analysis_date",
            ]
        )
        .reset_index(drop=True)
    )

    if TEST_LAST_N_DATES is not None:

        df = (
            df.groupby(
                [
                    "window",
                    "density_fraction",
                ],
                group_keys=False,
            )
            .tail(
                int(TEST_LAST_N_DATES)
            )
            .reset_index(drop=True)
        )

    return df


# ============================================================
# 4. Load Nodes
# ============================================================

def load_nodes(
    date,
    window,
):

    date_string = (
        pd.Timestamp(date)
        .strftime("%Y-%m-%d")
    )

    path = (
        NODE_ROOT
        / f"W{window}"
        / f"{date_string}.parquet"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Node file 不存在：\n{path}"
        )

    nodes = pd.read_parquet(
        path
    )

    required = [
        "master_index",
        "industry_id1",
    ]

    missing = [
        c
        for c in required
        if c not in nodes.columns
    ]

    if missing:

        raise ValueError(
            f"Node file 缺字段：{missing}"
        )

    return (
        nodes.sort_values(
            "master_index"
        )
        .reset_index(drop=True)
    )


# ============================================================
# 5. Load Edge Master and Cut to Exact K
# ============================================================

def load_edges(
    network_id,
    date,
    window,
    edge_budget,
):

    date_string = (
        pd.Timestamp(date)
        .strftime("%Y-%m-%d")
    )

    path = (
        EDGE_ROOT
        / network_id
        / f"W{window}"
        / f"{date_string}.parquet"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Edge master 不存在：\n{path}"
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
        c
        for c in required
        if c not in edges.columns
    ]

    if missing:

        raise ValueError(
            f"Edge master 缺字段：{missing}"
        )

    result = (
        edges[
            edges["rank"]
            <= int(edge_budget)
        ]
        .sort_values("rank")
        .reset_index(drop=True)
    )

    if len(result) != int(edge_budget):

        raise RuntimeError(
            f"{network_id}, "
            f"W={window}, {date_string}: "
            f"实际 K={len(result):,}, "
            f"expected={edge_budget:,}"
        )

    return result


# ============================================================
# 6. Edge Key
# ============================================================

def edge_keys(
    edges,
    multiplier,
):

    i = edges[
        "master_index_i"
    ].to_numpy(
        dtype=np.int64
    )

    j = edges[
        "master_index_j"
    ].to_numpy(
        dtype=np.int64
    )

    a = np.minimum(
        i,
        j,
    )

    b = np.maximum(
        i,
        j,
    )

    return (
        a
        *
        multiplier
        +
        b
    )


# ============================================================
# 7. Local Node Mapping
# ============================================================

def map_edges_to_local(
    nodes,
    edges,
):

    master_index = nodes[
        "master_index"
    ].to_numpy(
        dtype=np.int64
    )

    indexer = pd.Index(
        master_index
    )

    u = indexer.get_indexer(
        edges[
            "master_index_i"
        ].to_numpy(
            dtype=np.int64
        )
    )

    v = indexer.get_indexer(
        edges[
            "master_index_j"
        ].to_numpy(
            dtype=np.int64
        )
    )

    if (
        np.any(u < 0)
        or
        np.any(v < 0)
    ):

        raise RuntimeError(
            "Edge endpoint 不在 node universe 中。"
        )

    return (
        u.astype(np.int32),
        v.astype(np.int32),
    )


# ============================================================
# 8. Gini
# ============================================================

def gini(
    x,
):

    x = np.asarray(
        x,
        dtype=float,
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
            "Gini requires nonnegative values."
        )

    total = float(
        x.sum()
    )

    if total <= 0:

        return 0.0

    x = np.sort(x)

    n = len(x)

    index = np.arange(
        1,
        n + 1,
        dtype=float,
    )

    return float(

        2
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


# ============================================================
# 9. Top Fraction Share
# ============================================================

def top_share(
    x,
    fraction,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:

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
                len(x)
            )
        ),
    )

    top = np.partition(
        x,
        len(x) - k,
    )[
        len(x) - k:
    ]

    return float(
        top.sum()
        /
        total
    )


# ============================================================
# 10. Structural Metrics for One q
# ============================================================

def structural_metrics(
    nodes,
    edges,
):

    n = len(nodes)

    m = len(edges)

    u, v = (
        map_edges_to_local(
            nodes,
            edges,
        )
    )

    weights = edges[
        "correlation"
    ].to_numpy(
        dtype=float
    )

    degree = np.bincount(
        np.concatenate(
            [
                u,
                v,
            ]
        ),
        minlength=n,
    )

    strength = np.zeros(
        n,
        dtype=float,
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

    A = sparse.coo_matrix(
        (
            data,
            (
                row,
                col,
            ),
        ),
        shape=(
            n,
            n,
        ),
    ).tocsr()

    A.data[:] = 1

    (
        n_components,
        labels,
    ) = connected_components(
        A,
        directed=False,
        return_labels=True,
    )

    component_sizes = np.bincount(
        labels
    )

    giant_share = (
        component_sizes.max()
        /
        n
    )

    isolated_share = float(
        np.mean(
            degree == 0
        )
    )

    return {

        "n_nodes":
            int(n),

        "n_edges":
            int(m),

        "mean_degree":
            float(
                degree.mean()
            ),

        "isolated_node_share":
            isolated_share,

        "connected_component_count":
            int(
                n_components
            ),

        "giant_component_share":
            float(
                giant_share
            ),

        "degree_gini":
            gini(
                degree
            ),

        "strength_gini":
            gini(
                strength
            ),

        "top1pct_degree_share":
            top_share(
                degree,
                0.01,
            ),

        "top5pct_degree_share":
            top_share(
                degree,
                0.05,
            ),

        "top1pct_strength_share":
            top_share(
                strength,
                0.01,
            ),

        "top5pct_strength_share":
            top_share(
                strength,
                0.05,
            ),
    }


# ============================================================
# 11. Same-industry Edge Share
# ============================================================

def industry_metrics(
    nodes,
    edges,
    baseline,
):

    industry_map = dict(
        zip(
            nodes[
                "master_index"
            ].astype(int),

            nodes[
                "industry_id1"
            ].astype(int),
        )
    )

    i = edges[
        "master_index_i"
    ].to_numpy(
        dtype=np.int64
    )

    j = edges[
        "master_index_j"
    ].to_numpy(
        dtype=np.int64
    )

    ind_i = np.array(
        [
            industry_map.get(
                int(x),
                -1,
            )
            for x in i
        ],
        dtype=np.int32,
    )

    ind_j = np.array(
        [
            industry_map.get(
                int(x),
                -1,
            )
            for x in j
        ],
        dtype=np.int32,
    )

    labeled = (
        (ind_i >= 0)
        &
        (ind_j >= 0)
    )

    same = (
        labeled
        &
        (
            ind_i
            ==
            ind_j
        )
    )

    labeled_n = int(
        labeled.sum()
    )

    same_share = (

        float(
            same.sum()
            /
            labeled_n
        )

        if labeled_n > 0

        else np.nan
    )

    enrichment = (

        same_share
        /
        baseline

        if (
            np.isfinite(
                baseline
            )
            and
            baseline > 0
        )

        else np.nan
    )

    return {

        "industry_labeled_edge_count":
            labeled_n,

        "same_industry_edge_share":
            same_share,

        "industry_enrichment":
            enrichment,
    }


# ============================================================
# 12. Analyze One Date × W × q
# ============================================================

def analyze_one_job(
    row,
    multiplier,
):

    date = pd.Timestamp(
        row.analysis_date
    )

    window = int(
        row.window
    )

    q = float(
        row.density_fraction
    )

    K = int(
        row.edge_budget_k
    )

    baseline = float(
        row.common_same_industry_pair_share
    )

    nodes = load_nodes(
        date,
        window,
    )

    network_results = {}

    edge_sets = {}

    for network_id in NETWORK_IDS:

        edges = load_edges(
            network_id=
                network_id,

            date=
                date,

            window=
                window,

            edge_budget=
                K,
        )

        structure = (
            structural_metrics(
                nodes,
                edges,
            )
        )

        industry = (
            industry_metrics(
                nodes,
                edges,
                baseline,
            )
        )

        result = {

            "analysis_date":
                date,

            "window":
                window,

            "density_fraction":
                q,

            "density_percent":
                100 * q,

            "network_id":
                network_id,

            **structure,
            **industry,
        }

        network_results[
            network_id
        ] = result

        edge_sets[
            network_id
        ] = set(
            edge_keys(
                edges,
                multiplier,
            ).tolist()
        )

    raw_set = edge_sets[
        "B0_RAW_POSITIVE"
    ]

    residual_set = edge_sets[
        "M1_RESIDUAL_POSITIVE"
    ]

    intersection = (
        raw_set
        &
        residual_set
    )

    union = (
        raw_set
        |
        residual_set
    )

    overlap_fraction = (
        len(intersection)
        /
        len(raw_set)
    )

    jaccard = (
        len(intersection)
        /
        len(union)
    )

    comparison = {

        "analysis_date":
            date,

        "window":
            window,

        "density_fraction":
            q,

        "density_percent":
            100 * q,

        "edge_budget_k":
            K,

        "raw_residual_overlap_fraction":
            overlap_fraction,

        "raw_residual_jaccard":
            jaccard,
    }

    return (
        list(
            network_results.values()
        ),
        comparison,
    )


# ============================================================
# 13. Adjacent Edge Stability for One q
# ============================================================

def adjacent_edge_stability(
    matched_summary,
    multiplier,
):

    rows = []

    for (
        window,
        q
    ), group in matched_summary.groupby(
        [
            "window",
            "density_fraction",
        ]
    ):

        group = (
            group.sort_values(
                "analysis_date"
            )
            .reset_index(drop=True)
        )

        if len(group) < 2:

            continue

        for t in range(
            len(group) - 1
        ):

            row_t = group.iloc[t]
            row_t1 = group.iloc[
                t + 1
            ]

            date_t = pd.Timestamp(
                row_t.analysis_date
            )

            date_t1 = pd.Timestamp(
                row_t1.analysis_date
            )

            nodes_t = load_nodes(
                date_t,
                int(window),
            )

            nodes_t1 = load_nodes(
                date_t1,
                int(window),
            )

            common_nodes = set(
                nodes_t[
                    "master_index"
                ]
                .astype(int)
            ) & set(
                nodes_t1[
                    "master_index"
                ]
                .astype(int)
            )

            for network_id in (
                NETWORK_IDS
            ):

                edges_t = load_edges(

                    network_id,
                    date_t,
                    int(window),
                    int(
                        row_t.edge_budget_k
                    ),
                )

                edges_t1 = load_edges(

                    network_id,
                    date_t1,
                    int(window),
                    int(
                        row_t1.edge_budget_k
                    ),
                )

                # --------------------------------------------
                # Common-node adjusted
                # --------------------------------------------

                mask_t = (
                    edges_t[
                        "master_index_i"
                    ].isin(
                        common_nodes
                    )
                    &
                    edges_t[
                        "master_index_j"
                    ].isin(
                        common_nodes
                    )
                )

                mask_t1 = (
                    edges_t1[
                        "master_index_i"
                    ].isin(
                        common_nodes
                    )
                    &
                    edges_t1[
                        "master_index_j"
                    ].isin(
                        common_nodes
                    )
                )

                e_t = set(
                    edge_keys(
                        edges_t.loc[
                            mask_t
                        ],
                        multiplier,
                    ).tolist()
                )

                e_t1 = set(
                    edge_keys(
                        edges_t1.loc[
                            mask_t1
                        ],
                        multiplier,
                    ).tolist()
                )

                inter = (
                    e_t
                    &
                    e_t1
                )

                union = (
                    e_t
                    |
                    e_t1
                )

                jaccard = (

                    len(inter)
                    /
                    len(union)

                    if len(union) > 0
                    else np.nan
                )

                retention = (

                    len(inter)
                    /
                    len(e_t)

                    if len(e_t) > 0
                    else np.nan
                )

                rows.append(
                    {

                        "window":
                            int(window),

                        "density_fraction":
                            float(q),

                        "density_percent":
                            100 * float(q),

                        "network_id":
                            network_id,

                        "date_t":
                            date_t,

                        "date_t1":
                            date_t1,

                        "common_node_count":
                            len(
                                common_nodes
                            ),

                        "common_edge_jaccard":
                            jaccard,

                        "forward_edge_retention":
                            retention,
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 14. Same vs Cross Dynamic Stability
# ============================================================

def same_cross_dynamic_stability(
    matched_summary,
    multiplier,
):

    rows = []

    for (
        window,
        q
    ), group in matched_summary.groupby(
        [
            "window",
            "density_fraction",
        ]
    ):

        group = (
            group.sort_values(
                "analysis_date"
            )
            .reset_index(drop=True)
        )

        if len(group) < 2:

            continue

        for t in range(
            len(group) - 1
        ):

            row_t = group.iloc[t]
            row_t1 = group.iloc[
                t + 1
            ]

            date_t = pd.Timestamp(
                row_t.analysis_date
            )

            date_t1 = pd.Timestamp(
                row_t1.analysis_date
            )

            nodes_t = load_nodes(
                date_t,
                int(window),
            )

            nodes_t1 = load_nodes(
                date_t1,
                int(window),
            )

            common_nodes = set(
                nodes_t[
                    "master_index"
                ]
                .astype(int)
            ) & set(
                nodes_t1[
                    "master_index"
                ]
                .astype(int)
            )

            industry_map = dict(
                zip(
                    nodes_t[
                        "master_index"
                    ].astype(int),

                    nodes_t[
                        "industry_id1"
                    ].astype(int),
                )
            )

            for network_id in (
                NETWORK_IDS
            ):

                edges_t = load_edges(

                    network_id,
                    date_t,
                    int(window),
                    int(
                        row_t.edge_budget_k
                    ),
                )

                edges_t1 = load_edges(

                    network_id,
                    date_t1,
                    int(window),
                    int(
                        row_t1.edge_budget_k
                    ),
                )

                mask_t = (
                    edges_t[
                        "master_index_i"
                    ].isin(
                        common_nodes
                    )
                    &
                    edges_t[
                        "master_index_j"
                    ].isin(
                        common_nodes
                    )
                )

                mask_t1 = (
                    edges_t1[
                        "master_index_i"
                    ].isin(
                        common_nodes
                    )
                    &
                    edges_t1[
                        "master_index_j"
                    ].isin(
                        common_nodes
                    )
                )

                e_t_df = (
                    edges_t.loc[
                        mask_t
                    ]
                    .copy()
                )

                e_t1_df = (
                    edges_t1.loc[
                        mask_t1
                    ]
                )

                keys_t = edge_keys(
                    e_t_df,
                    multiplier,
                )

                keys_t1 = set(
                    edge_keys(
                        e_t1_df,
                        multiplier,
                    ).tolist()
                )

                i = e_t_df[
                    "master_index_i"
                ].to_numpy(
                    dtype=np.int64
                )

                j = e_t_df[
                    "master_index_j"
                ].to_numpy(
                    dtype=np.int64
                )

                ind_i = np.array(
                    [
                        industry_map.get(
                            int(x),
                            -1,
                        )
                        for x in i
                    ],
                    dtype=np.int32,
                )

                ind_j = np.array(
                    [
                        industry_map.get(
                            int(x),
                            -1,
                        )
                        for x in j
                    ],
                    dtype=np.int32,
                )

                labeled = (
                    (ind_i >= 0)
                    &
                    (ind_j >= 0)
                )

                same = (
                    labeled
                    &
                    (
                        ind_i
                        ==
                        ind_j
                    )
                )

                cross = (
                    labeled
                    &
                    (
                        ind_i
                        !=
                        ind_j
                    )
                )

                persistent = np.array(
                    [
                        int(key)
                        in
                        keys_t1
                        for key in keys_t
                    ],
                    dtype=bool,
                )

                same_retention = (

                    float(
                        (
                            same
                            &
                            persistent
                        ).sum()
                        /
                        same.sum()
                    )

                    if same.sum() > 0
                    else np.nan
                )

                cross_retention = (

                    float(
                        (
                            cross
                            &
                            persistent
                        ).sum()
                        /
                        cross.sum()
                    )

                    if cross.sum() > 0
                    else np.nan
                )

                rows.append(
                    {

                        "window":
                            int(window),

                        "density_fraction":
                            float(q),

                        "network_id":
                            network_id,

                        "date_t":
                            date_t,

                        "date_t1":
                            date_t1,

                        "same_industry_retention":
                            same_retention,

                        "cross_industry_retention":
                            cross_retention,

                        "same_minus_cross_retention":
                            (
                                same_retention
                                -
                                cross_retention

                                if (
                                    np.isfinite(
                                        same_retention
                                    )
                                    and
                                    np.isfinite(
                                        cross_retention
                                    )
                                )

                                else np.nan
                            ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 15. Per-node Top-k Exactness QA
#
# IMPORTANT:
#
# Step 2 only stored global Top-2%.
#
# A node's true top-k neighbours can be recovered from this
# master ONLY if its kth strongest stored incident edge is
# strictly above the global 2% cutoff.
#
# Otherwise unseen edges below / at the global cutoff may alter
# the node's top-k ranking.
# ============================================================

def per_node_topk_exactness_qa(
    matched_summary,
):

    rows = []

    if not CHECK_PER_NODE_TOPK:

        return pd.DataFrame()

    max_q = max(
        Q_GRID
    )

    max_q_rows = (
        matched_summary[
            np.isclose(
                matched_summary[
                    "density_fraction"
                ],
                max_q,
            )
        ]
        .copy()
    )

    for row in (
        max_q_rows.itertuples(
            index=False
        )
    ):

        date = pd.Timestamp(
            row.analysis_date
        )

        window = int(
            row.window
        )

        K_max = int(
            row.edge_budget_k
        )

        nodes = load_nodes(
            date,
            window,
        )

        node_ids = nodes[
            "master_index"
        ].astype(int)

        for network_id in (
            NETWORK_IDS
        ):

            edges = load_edges(
                network_id,
                date,
                window,
                K_max,
            )

            global_cutoff = float(
                edges[
                    "correlation"
                ].min()
            )

            left = edges[
                [
                    "master_index_i",
                    "correlation",
                ]
            ].rename(
                columns={
                    "master_index_i":
                        "master_index"
                }
            )

            right = edges[
                [
                    "master_index_j",
                    "correlation",
                ]
            ].rename(
                columns={
                    "master_index_j":
                        "master_index"
                }
            )

            incident = pd.concat(
                [
                    left,
                    right,
                ],
                ignore_index=True,
            )

            incident = (
                incident.sort_values(
                    [
                        "master_index",
                        "correlation",
                    ],
                    ascending=[
                        True,
                        False,
                    ],
                )
            )

            for k in TOPK_GRID:

                counts = (
                    incident.groupby(
                        "master_index"
                    )
                    .size()
                    .reindex(
                        node_ids,
                        fill_value=0,
                    )
                )

                sufficient_count = (
                    counts >= k
                )

                kth_values = (
                    incident.groupby(
                        "master_index"
                    )[
                        "correlation"
                    ]
                    .nth(
                        k - 1
                    )
                    .reindex(
                        node_ids
                    )
                )

                # Strictly above global cutoff:
                # avoids ambiguity at the truncation boundary.
                exact_node = (

                    sufficient_count

                    &

                    (
                        kth_values
                        >
                        global_cutoff
                        +
                        1e-12
                    )
                )

                exact_share = float(
                    exact_node.mean()
                )

                all_nodes_exact = bool(
                    exact_node.all()
                )

                rows.append(
                    {

                        "analysis_date":
                            date,

                        "window":
                            window,

                        "network_id":
                            network_id,

                        "top_k":
                            int(k),

                        "global_max_density":
                            max_q,

                        "global_cutoff":
                            global_cutoff,

                        "node_count":
                            int(
                                len(
                                    node_ids
                                )
                            ),

                        "nodes_with_at_least_k_stored_edges":
                            int(
                                sufficient_count.sum()
                            ),

                        "nodes_exactly_recoverable":
                            int(
                                exact_node.sum()
                            ),

                        "exactly_recoverable_share":
                            exact_share,

                        "all_nodes_exactly_recoverable":
                            all_nodes_exact,

                        "formal_topk_available_from_step2_master":
                            all_nodes_exact,
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 16. Window × q Summary
# ============================================================

def summarize_network_metrics(
    detail,
):

    metrics = [

        "giant_component_share",
        "isolated_node_share",

        "degree_gini",
        "strength_gini",

        "top1pct_degree_share",
        "top5pct_degree_share",

        "top1pct_strength_share",
        "top5pct_strength_share",

        "same_industry_edge_share",
        "industry_enrichment",
    ]

    agg = {

        metric: [
            "mean",
            "median",
        ]

        for metric in metrics
    }

    result = (
        detail.groupby(
            [
                "window",
                "density_fraction",
                "network_id",
            ]
        )
        .agg(
            agg
        )
    )

    result.columns = [
        f"{metric}_{stat}"
        for metric, stat
        in result.columns
    ]

    return (
        result
        .reset_index()
    )


# ============================================================
# 17. Raw vs Residual Difference
# ============================================================

def build_raw_residual_difference(
    detail,
):

    metrics = [

        "giant_component_share",
        "isolated_node_share",

        "degree_gini",
        "strength_gini",

        "top1pct_degree_share",
        "top1pct_strength_share",

        "same_industry_edge_share",
        "industry_enrichment",
    ]

    raw = (
        detail[
            detail["network_id"]
            ==
            "B0_RAW_POSITIVE"
        ][
            [
                "analysis_date",
                "window",
                "density_fraction",
            ]
            +
            metrics
        ]
        .copy()
    )

    residual = (
        detail[
            detail["network_id"]
            ==
            "M1_RESIDUAL_POSITIVE"
        ][
            [
                "analysis_date",
                "window",
                "density_fraction",
            ]
            +
            metrics
        ]
        .copy()
    )

    raw = raw.rename(
        columns={
            x:
                f"{x}_raw"
            for x in metrics
        }
    )

    residual = residual.rename(
        columns={
            x:
                f"{x}_residual"
            for x in metrics
        }
    )

    merged = raw.merge(

        residual,

        on=[
            "analysis_date",
            "window",
            "density_fraction",
        ],

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
# 18. Robustness Pass Table
#
# 这里的 PASS 只是描述性一致性标准，
# 不是 statistical significance test。
# ============================================================

def build_robustness_pass_table(
    difference,
    overlap_summary,
    dynamic_summary,
    same_cross_summary,
):

    rows = []

    for window in sorted(
        difference[
            "window"
        ]
        .unique()
    ):

        for q in sorted(
            difference[
                "density_fraction"
            ]
            .unique()
        ):

            d = difference[
                (
                    difference[
                        "window"
                    ]
                    ==
                    window
                )
                &
                np.isclose(
                    difference[
                        "density_fraction"
                    ],
                    q,
                )
            ]

            overlap = overlap_summary[
                (
                    overlap_summary[
                        "window"
                    ]
                    ==
                    window
                )
                &
                np.isclose(
                    overlap_summary[
                        "density_fraction"
                    ],
                    q,
                )
            ]

            dyn = dynamic_summary[
                (
                    dynamic_summary[
                        "window"
                    ]
                    ==
                    window
                )
                &
                np.isclose(
                    dynamic_summary[
                        "density_fraction"
                    ],
                    q,
                )
            ]

            sc = same_cross_summary[
                (
                    same_cross_summary[
                        "window"
                    ]
                    ==
                    window
                )
                &
                np.isclose(
                    same_cross_summary[
                        "density_fraction"
                    ],
                    q,
                )
            ]

            if d.empty:

                continue

            industry_advantage_share = float(
                (
                    d[
                        "industry_enrichment_residual_minus_raw"
                    ]
                    >
                    0
                )
                .mean()
            )

            residual_less_concentrated_share = float(
                (
                    d[
                        "degree_gini_residual_minus_raw"
                    ]
                    <
                    0
                )
                .mean()
            )

            residual_more_connected_share = float(
                (
                    d[
                        "giant_component_share_residual_minus_raw"
                    ]
                    >
                    0
                )
                .mean()
            )

            mean_overlap = (

                float(
                    overlap[
                        "raw_residual_overlap_fraction"
                    ]
                    .mean()
                )

                if not overlap.empty

                else np.nan
            )

            # -----------------------------------------------
            # Residual vs Raw dynamic edge stability
            # -----------------------------------------------

            residual_dyn = dyn[
                dyn[
                    "network_id"
                ]
                ==
                "M1_RESIDUAL_POSITIVE"
            ]

            raw_dyn = dyn[
                dyn[
                    "network_id"
                ]
                ==
                "B0_RAW_POSITIVE"
            ]

            dynamic_advantage = np.nan

            if (
                not residual_dyn.empty
                and
                not raw_dyn.empty
            ):

                dynamic_advantage = (

                    float(
                        residual_dyn[
                            "forward_edge_retention"
                        ]
                        .mean()
                    )

                    -

                    float(
                        raw_dyn[
                            "forward_edge_retention"
                        ]
                        .mean()
                    )
                )

            residual_sc = sc[
                sc[
                    "network_id"
                ]
                ==
                "M1_RESIDUAL_POSITIVE"
            ]

            same_cross_gap = (

                float(
                    residual_sc[
                        "same_minus_cross_retention"
                    ]
                    .mean()
                )

                if not residual_sc.empty

                else np.nan
            )

            rows.append(
                {

                    "window":
                        int(window),

                    "density_fraction":
                        float(q),

                    "density_percent":
                        100 * float(q),

                    "industry_advantage_share":
                        industry_advantage_share,

                    "residual_less_concentrated_share":
                        residual_less_concentrated_share,

                    "residual_more_connected_share":
                        residual_more_connected_share,

                    "mean_raw_residual_edge_overlap":
                        mean_overlap,

                    "residual_minus_raw_dynamic_retention":
                        dynamic_advantage,

                    "residual_same_minus_cross_retention":
                        same_cross_gap,

                    # ---------------------------------------
                    # Descriptive robustness checks
                    # ---------------------------------------

                    "industry_structure_preserved":
                        (
                            industry_advantage_share
                            >=
                            0.90
                        ),

                    "lower_hub_concentration_preserved":
                        (
                            residual_less_concentrated_share
                            >=
                            0.90
                        ),

                    "broader_connectivity_preserved":
                        (
                            residual_more_connected_share
                            >=
                            0.90
                        ),

                    "same_cross_persistence_preserved":
                        (
                            np.isfinite(
                                same_cross_gap
                            )
                            and
                            same_cross_gap
                            >
                            0
                        ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        result[
            "core_conclusion_preserved"
        ] = (

            result[
                [
                    "industry_structure_preserved",
                    "lower_hub_concentration_preserved",
                    "broader_connectivity_preserved",
                    "same_cross_persistence_preserved",
                ]
            ]
            .all(
                axis=1
            )
        )

    return result


# ============================================================
# 19. Plot q Sensitivity
# ============================================================

def plot_q_metric(
    summary,
    metric,
    ylabel,
    filename,
):

    fig, ax = plt.subplots(
        figsize=(
            10,
            5.5,
        )
    )

    for (
        window,
        network_id
    ), group in summary.groupby(
        [
            "window",
            "network_id",
        ]
    ):

        label = (

            f"W={window} Raw"

            if network_id
            ==
            "B0_RAW_POSITIVE"

            else
            f"W={window} Residual"
        )

        group = group.sort_values(
            "density_fraction"
        )

        ax.plot(

            100
            *
            group[
                "density_fraction"
            ],

            group[
                metric
            ],

            marker="o",

            label=
                label,
        )

    ax.set_xlabel(
        "Network Density (%)"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        ylabel
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        ncol=2,
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / filename,
        dpi=180,
    )

    plt.close(
        fig
    )


# ============================================================
# 20. Metadata
# ============================================================

def save_metadata():

    payload = {

        "research_day":
            4,

        "step":
            "Step6_Sparsification_Robustness",

        "global_density_grid":
            Q_GRID,

        "primary_density":
            PRIMARY_Q,

        "per_node_topk_grid":
            TOPK_GRID,

        "main_principle":
            (
                "Re-evaluate the substantive conclusions "
                "from Steps 2-5 across fixed global density "
                "levels without recomputing correlations."
            ),

        "important_notes": [

            (
                "Global Top-q robustness is exact because "
                "Step 2 stored the ranked maximum-density "
                "edge master."
            ),

            (
                "Average degree is mechanically determined "
                "by q and is not treated as robustness evidence."
            ),

            (
                "Per-node Top-k cannot in general be recovered "
                "from a truncated global Top-2% edge master."
            ),

            (
                "The script therefore performs an exactness "
                "diagnostic and does not report approximate "
                "Top-k as formal robustness."
            ),

            (
                "The PASS variables are descriptive research "
                "QA criteria, not statistical significance tests."
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        / "step6_sparsification_robustness_metadata.json",
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
# 21. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 4 - Step 6"
    )

    print(
        "Sparsification Robustness"
    )

    print("=" * 80)

    metadata = load_metadata()

    matched_summary = (
        load_matched_summary()
    )

    # ========================================================
    # Edge key multiplier
    # ========================================================

    max_master_index = 0

    unique_jobs = (
        matched_summary[
            [
                "analysis_date",
                "window",
            ]
        ]
        .drop_duplicates()
    )

    for row in (
        unique_jobs.itertuples(
            index=False
        )
    ):

        nodes = load_nodes(
            row.analysis_date,
            int(
                row.window
            ),
        )

        if not nodes.empty:

            max_master_index = max(

                max_master_index,

                int(
                    nodes[
                        "master_index"
                    ]
                    .max()
                ),
            )

    multiplier = (
        max_master_index
        +
        1
    )

    # ========================================================
    # 1. Structural robustness
    # ========================================================

    print()
    print(
        "[1] Global Top-q structural robustness"
    )

    detail_rows = []

    comparison_rows = []

    for job_no, row in enumerate(

        matched_summary.itertuples(
            index=False
        ),

        start=1,
    ):

        if (
            job_no == 1
            or
            job_no % 50 == 0
        ):

            print(
                f"  Job "
                f"{job_no}/"
                f"{len(matched_summary)}"
            )

        network_rows, comparison = (
            analyze_one_job(
                row,
                multiplier,
            )
        )

        detail_rows.extend(
            network_rows
        )

        comparison_rows.append(
            comparison
        )

    detail = pd.DataFrame(
        detail_rows
    )

    comparison = pd.DataFrame(
        comparison_rows
    )

    detail.to_csv(

        OUTPUT_DIR
        / "density_structural_robustness_detail.csv",

        index=False,

        encoding="utf-8-sig",
    )

    comparison.to_csv(

        OUTPUT_DIR
        / "density_raw_residual_overlap_detail.csv",

        index=False,

        encoding="utf-8-sig",
    )

    network_summary = (
        summarize_network_metrics(
            detail
        )
    )

    network_summary.to_csv(

        OUTPUT_DIR
        / "density_structural_robustness_summary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    difference = (
        build_raw_residual_difference(
            detail
        )
    )

    difference.to_csv(

        OUTPUT_DIR
        / "density_raw_residual_difference_detail.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 2. Dynamic robustness across q
    # ========================================================

    print()
    print(
        "[2] Dynamic stability across density"
    )

    dynamic = (
        adjacent_edge_stability(
            matched_summary,
            multiplier,
        )
    )

    dynamic.to_csv(

        OUTPUT_DIR
        / "density_dynamic_stability_detail.csv",

        index=False,

        encoding="utf-8-sig",
    )

    dynamic_summary = (
        dynamic.groupby(
            [
                "window",
                "density_fraction",
                "network_id",
            ],
            as_index=False,
        )
        .agg(

            common_edge_jaccard=
                (
                    "common_edge_jaccard",
                    "mean",
                ),

            forward_edge_retention=
                (
                    "forward_edge_retention",
                    "mean",
                ),
        )
    )

    dynamic_summary.to_csv(

        OUTPUT_DIR
        / "density_dynamic_stability_summary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 3. Same / Cross persistence across q
    # ========================================================

    print()
    print(
        "[3] Same/Cross persistence across density"
    )

    same_cross = (
        same_cross_dynamic_stability(
            matched_summary,
            multiplier,
        )
    )

    same_cross.to_csv(

        OUTPUT_DIR
        / "density_same_cross_stability_detail.csv",

        index=False,

        encoding="utf-8-sig",
    )

    same_cross_summary = (
        same_cross.groupby(
            [
                "window",
                "density_fraction",
                "network_id",
            ],
            as_index=False,
        )
        .agg(

            same_industry_retention=
                (
                    "same_industry_retention",
                    "mean",
                ),

            cross_industry_retention=
                (
                    "cross_industry_retention",
                    "mean",
                ),

            same_minus_cross_retention=
                (
                    "same_minus_cross_retention",
                    "mean",
                ),
        )
    )

    same_cross_summary.to_csv(

        OUTPUT_DIR
        / "density_same_cross_stability_summary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 4. Overlap summary
    # ========================================================

    overlap_summary = (
        comparison.groupby(
            [
                "window",
                "density_fraction",
            ],
            as_index=False,
        )
        .agg(

            raw_residual_overlap_fraction=
                (
                    "raw_residual_overlap_fraction",
                    "mean",
                ),

            raw_residual_jaccard=
                (
                    "raw_residual_jaccard",
                    "mean",
                ),
        )
    )

    overlap_summary.to_csv(

        OUTPUT_DIR
        / "density_raw_residual_overlap_summary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 5. Robustness pass
    # ========================================================

    print()
    print(
        "[4] Build robustness conclusion table"
    )

    pass_table = (
        build_robustness_pass_table(

            difference=
                difference,

            overlap_summary=
                overlap_summary,

            dynamic_summary=
                dynamic_summary,

            same_cross_summary=
                same_cross_summary,
        )
    )

    pass_table.to_csv(

        OUTPUT_DIR
        / "sparsification_robustness_pass_table.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 6. Per-node Top-k feasibility
    # ========================================================

    if CHECK_PER_NODE_TOPK:

        print()
        print(
            "[5] Check exact per-node Top-k feasibility"
        )

        topk_qa = (
            per_node_topk_exactness_qa(
                matched_summary
            )
        )

        topk_qa.to_csv(

            OUTPUT_DIR
            / "per_node_topk_exactness_qa.csv",

            index=False,

            encoding="utf-8-sig",
        )

        if not topk_qa.empty:

            topk_summary = (
                topk_qa.groupby(
                    [
                        "window",
                        "network_id",
                        "top_k",
                    ],
                    as_index=False,
                )
                .agg(

                    mean_exactly_recoverable_share=
                        (
                            "exactly_recoverable_share",
                            "mean",
                        ),

                    min_exactly_recoverable_share=
                        (
                            "exactly_recoverable_share",
                            "min",
                        ),

                    all_dates_fully_exact=
                        (
                            "all_nodes_exactly_recoverable",
                            "all",
                        ),
                )
            )

            topk_summary.to_csv(

                OUTPUT_DIR
                / "per_node_topk_exactness_summary.csv",

                index=False,

                encoding="utf-8-sig",
            )

    # ========================================================
    # 7. Figures
    # ========================================================

    plot_q_metric(

        summary=
            network_summary,

        metric=
            "industry_enrichment_mean",

        ylabel=
            "Industry Enrichment",

        filename=
            "density_vs_industry_enrichment.png",
    )

    plot_q_metric(

        summary=
            network_summary,

        metric=
            "degree_gini_mean",

        ylabel=
            "Degree Gini",

        filename=
            "density_vs_degree_gini.png",
    )

    plot_q_metric(

        summary=
            network_summary,

        metric=
            "giant_component_share_mean",

        ylabel=
            "Giant Component Share",

        filename=
            "density_vs_giant_component_share.png",
    )

    # --------------------------------------------------------
    # Overlap plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    for window, group in (
        overlap_summary.groupby(
            "window"
        )
    ):

        group = group.sort_values(
            "density_fraction"
        )

        ax.plot(

            100
            *
            group[
                "density_fraction"
            ],

            group[
                "raw_residual_overlap_fraction"
            ],

            marker="o",

            label=
                f"W={window}",
        )

    ax.set_xlabel(
        "Network Density (%)"
    )

    ax.set_ylabel(
        "Raw-Residual Edge Overlap"
    )

    ax.set_title(
        "Raw-Residual Edge Overlap across Sparsification"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(

        FIGURE_DIR
        / "density_vs_raw_residual_overlap.png",

        dpi=180,
    )

    plt.close(
        fig
    )

    save_metadata()

    # ========================================================
    # Console output
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Sparsification Robustness"
    )

    print("=" * 80)

    show_cols = [

        "window",
        "density_fraction",

        "industry_advantage_share",
        "residual_less_concentrated_share",
        "residual_more_connected_share",

        "mean_raw_residual_edge_overlap",

        "residual_minus_raw_dynamic_retention",

        "residual_same_minus_cross_retention",

        "core_conclusion_preserved",
    ]

    print(
        pass_table[
            show_cols
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()