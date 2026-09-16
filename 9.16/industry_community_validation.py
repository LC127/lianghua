from __future__ import annotations

from pathlib import Path
import json
import math
import os

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
)

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


# ------------------------------------------------------------
# Day 4 Step 4
# ------------------------------------------------------------

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day4"
    / "04_step4_industry_community_validation"
)

COMMUNITY_ROOT = (
    OUTPUT_DIR
    / "community_membership"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMMUNITY_ROOT.mkdir(
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
# 正式 Step 4：
# 只分析 Step 1 冻结的 primary density。
#
# 正常为：
#
# q = 0.01
# ------------------------------------------------------------

OVERRIDE_DENSITY = None


# ------------------------------------------------------------
# Community detection
# ------------------------------------------------------------

COMMUNITY_RANDOM_SEED = 20260916

LOUVAIN_RESOLUTION = 1.0


# ------------------------------------------------------------
# 第一次测试建议 True。
#
# 正式全历史改成 False。
# ------------------------------------------------------------

TEST_LATEST_ONLY = False


NETWORK_IDS = [

    "B0_RAW_POSITIVE",

    "M1_RESIDUAL_POSITIVE",
]


# ============================================================
# 2. Load Metadata
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

        # Step 2 对 industry_id1
        # 已经计算的 exact feasible-pair baseline
        "common_same_industry_pair_share",
    ]

    missing = [

        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "matched_density_network_summary "
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
# 4. Resolve Density
# ============================================================

def resolve_density(
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
# 5. Select Jobs
# ============================================================

def select_jobs(
    matched_summary,
    density,
):

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
    )

    if TEST_LATEST_ONLY:

        jobs = (
            jobs.sort_values(
                "analysis_date"
            )
            .groupby(
                "window",
                as_index=False,
            )
            .tail(1)
        )

    return (
        jobs.sort_values(
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
# 6. Load Nodes
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

        "industry_id1",
        "industry_id2",
    ]

    missing = [

        col
        for col in required
        if col not in nodes.columns
    ]

    if missing:

        raise ValueError(
            f"Node file 缺字段：{missing}"
        )

    return (
        nodes.sort_values(
            "master_index"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 7. Load Edges
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
            f"Edge file 不存在：\n"
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
            f"Edge file 缺字段：{missing}"
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

    if len(
        edges
    ) != int(
        edge_budget
    ):

        raise RuntimeError(
            f"{network_id} "
            f"W={window} "
            f"{date_string}: "
            f"edge count mismatch."
        )

    return edges


# ============================================================
# 8. Build Graph
# ============================================================

def build_graph(
    nodes,
    edges,
):

    G = nx.Graph()

    # --------------------------------------------------------
    # Add all nodes
    # --------------------------------------------------------

    for row in (
        nodes.itertuples(
            index=False
        )
    ):

        G.add_node(

            int(
                row.master_index
            ),

            industry_id1=
                int(
                    row.industry_id1
                ),

            industry_id2=
                int(
                    row.industry_id2
                ),
        )

    # --------------------------------------------------------
    # Positive weighted edges
    # --------------------------------------------------------

    for row in (
        edges.itertuples(
            index=False
        )
    ):

        weight = float(
            row.correlation
        )

        if (
            not np.isfinite(
                weight
            )
            or
            weight <= 0
        ):

            continue

        G.add_edge(

            int(
                row.master_index_i
            ),

            int(
                row.master_index_j
            ),

            weight=
                weight,
        )

    return G


# ============================================================
# 9. Industry Same-edge Share
# ============================================================

def same_industry_edge_share(
    edges,
    nodes,
    industry_column,
):

    label_map = dict(
        zip(
            nodes[
                "master_index"
            ].astype(int),

            nodes[
                industry_column
            ].astype(int),
        )
    )

    i = (
        edges[
            "master_index_i"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    j = (
        edges[
            "master_index_j"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    li = np.array(
        [
            label_map.get(
                int(x),
                -1,
            )
            for x in i
        ],
        dtype=np.int32,
    )

    lj = np.array(
        [
            label_map.get(
                int(x),
                -1,
            )
            for x in j
        ],
        dtype=np.int32,
    )

    labeled = (
        (li >= 0)
        &
        (lj >= 0)
    )

    if not labeled.any():

        return (
            np.nan,
            0,
            0,
        )

    same = (
        li[
            labeled
        ]
        ==
        lj[
            labeled
        ]
    )

    return (

        float(
            same.mean()
        ),

        int(
            labeled.sum()
        ),

        int(
            same.sum()
        ),
    )


# ============================================================
# 10. Node-pair Industry Baseline
#
# 对 industry_id2 使用。
#
# 注意：
# 这是 node-pair baseline，
# 不等于 Step 2 的 exact common-valid-pair baseline。
#
# industry_id1 enrichment 仍优先使用 Step 2 的 exact denominator。
# ============================================================

def node_pair_same_industry_baseline(
    nodes,
    industry_column,
):

    labels = (
        nodes[
            industry_column
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    labels = labels[
        labels >= 0
    ]

    n = len(
        labels
    )

    if n < 2:

        return np.nan

    counts = (
        pd.Series(
            labels
        )
        .value_counts()
        .to_numpy(
            dtype=np.int64
        )
    )

    same_pairs = int(
        np.sum(
            counts
            *
            (
                counts - 1
            )
            //
            2
        )
    )

    total_pairs = (
        n
        *
        (
            n - 1
        )
        //
        2
    )

    return (
        same_pairs
        /
        total_pairs
        if total_pairs > 0
        else np.nan
    )


# ============================================================
# 11. Categorical Assortativity
#
# Newman nominal assortativity.
#
# 只使用 industry label 已知的 selected edges。
# ============================================================

def categorical_assortativity(
    edges,
    nodes,
    industry_column,
):

    label_map = dict(
        zip(
            nodes[
                "master_index"
            ].astype(int),

            nodes[
                industry_column
            ].astype(int),
        )
    )

    labels_i = np.array(
        [
            label_map.get(
                int(x),
                -1,
            )
            for x in edges[
                "master_index_i"
            ]
        ],
        dtype=np.int32,
    )

    labels_j = np.array(
        [
            label_map.get(
                int(x),
                -1,
            )
            for x in edges[
                "master_index_j"
            ]
        ],
        dtype=np.int32,
    )

    valid = (
        (labels_i >= 0)
        &
        (labels_j >= 0)
    )

    labels_i = labels_i[
        valid
    ]

    labels_j = labels_j[
        valid
    ]

    if len(
        labels_i
    ) == 0:

        return np.nan

    unique_labels = np.unique(
        np.concatenate(
            [
                labels_i,
                labels_j,
            ]
        )
    )

    label_to_local = {

        int(label):
            idx

        for idx, label
        in enumerate(
            unique_labels
        )
    }

    k = len(
        unique_labels
    )

    mixing = np.zeros(
        (
            k,
            k,
        ),
        dtype=np.float64,
    )

    for a, b in zip(
        labels_i,
        labels_j,
    ):

        ia = label_to_local[
            int(a)
        ]

        ib = label_to_local[
            int(b)
        ]

        # Undirected edge:
        # count both orientations.
        mixing[
            ia,
            ib
        ] += 1.0

        mixing[
            ib,
            ia
        ] += 1.0

    total = float(
        mixing.sum()
    )

    if total <= 0:

        return np.nan

    e = (
        mixing
        /
        total
    )

    a = e.sum(
        axis=1
    )

    b = e.sum(
        axis=0
    )

    expected = float(
        np.sum(
            a
            *
            b
        )
    )

    denominator = (
        1.0
        -
        expected
    )

    if abs(
        denominator
    ) < 1e-14:

        return np.nan

    assortativity = (

        np.trace(
            e
        )

        -

        expected

    ) / denominator

    return float(
        assortativity
    )


# ============================================================
# 12. Louvain Community Detection
# ============================================================

def detect_louvain_communities(
    G,
):

    # --------------------------------------------------------
    # Isolated nodes 不参与 community detection。
    #
    # 原因：
    # isolate 没有 edge-based community 信息，
    # 若把每个 isolate 当成一个 singleton community，
    # 会人为影响 NMI / ARI / purity。
    # --------------------------------------------------------

    active_nodes = [

        node

        for node, degree
        in G.degree()

        if degree > 0
    ]

    if len(
        active_nodes
    ) == 0:

        return (
            [],
            {},
            np.nan,
            0,
        )

    H = (
        G.subgraph(
            active_nodes
        )
        .copy()
    )

    communities = (
        nx.community.louvain_communities(

            H,

            weight="weight",

            resolution=
                LOUVAIN_RESOLUTION,

            seed=
                COMMUNITY_RANDOM_SEED,
        )
    )

    modularity = (
        nx.community.modularity(

            H,

            communities,

            weight="weight",
        )
    )

    community_map = {}

    for community_id, community in enumerate(
        communities
    ):

        for node in community:

            community_map[
                int(node)
            ] = int(
                community_id
            )

    return (
        communities,
        community_map,
        float(
            modularity
        ),
        len(
            active_nodes
        ),
    )


# ============================================================
# 13. Community Purity
# ============================================================

def weighted_community_purity(
    community_labels,
    industry_labels,
):

    df = pd.DataFrame(
        {
            "community":
                community_labels,

            "industry":
                industry_labels,
        }
    )

    df = df[
        df[
            "industry"
        ]
        >=
        0
    ]

    if df.empty:

        return np.nan

    correct = 0

    total = len(
        df
    )

    for _, group in (
        df.groupby(
            "community"
        )
    ):

        counts = (
            group[
                "industry"
            ]
            .value_counts()
        )

        correct += int(
            counts.iloc[
                0
            ]
        )

    return float(
        correct
        /
        total
    )


# ============================================================
# 14. Community / Industry Alignment
# ============================================================

def community_industry_alignment(
    nodes,
    community_map,
    industry_column,
):

    community_labels = []

    industry_labels = []

    for row in (
        nodes[
            [
                "master_index",
                industry_column,
            ]
        ]
        .itertuples(
            index=False
        )
    ):

        node = int(
            row.master_index
        )

        industry = int(
            getattr(
                row,
                industry_column
            )
        )

        if (
            node not in
            community_map
        ):

            # isolate:
            # community undefined
            continue

        if industry < 0:

            continue

        community_labels.append(
            community_map[
                node
            ]
        )

        industry_labels.append(
            industry
        )

    if len(
        community_labels
    ) < 2:

        return {
            "n_labeled_active_nodes":
                len(
                    community_labels
                ),

            "nmi":
                np.nan,

            "ari":
                np.nan,

            "weighted_purity":
                np.nan,
        }

    community_labels = np.asarray(
        community_labels,
        dtype=np.int32,
    )

    industry_labels = np.asarray(
        industry_labels,
        dtype=np.int32,
    )

    nmi = (
        normalized_mutual_info_score(

            industry_labels,

            community_labels,
        )
    )

    ari = (
        adjusted_rand_score(

            industry_labels,

            community_labels,
        )
    )

    purity = (
        weighted_community_purity(

            community_labels=
                community_labels,

            industry_labels=
                industry_labels,
        )
    )

    return {

        "n_labeled_active_nodes":
            int(
                len(
                    industry_labels
                )
            ),

        "nmi":
            float(
                nmi
            ),

        "ari":
            float(
                ari
            ),

        "weighted_purity":
            float(
                purity
            ),
    }


# ============================================================
# 15. Community Structural Statistics
# ============================================================

def community_structure_metrics(
    communities,
    active_node_count,
):

    if (
        len(
            communities
        ) == 0
        or
        active_node_count == 0
    ):

        return {

            "community_count":
                0,

            "largest_community_size":
                0,

            "largest_community_share_active":
                np.nan,

            "community_size_median":
                np.nan,

            "community_size_p90":
                np.nan,
        }

    sizes = np.array(
        [
            len(
                community
            )
            for community
            in communities
        ],
        dtype=np.int32,
    )

    return {

        "community_count":
            int(
                len(
                    sizes
                )
            ),

        "largest_community_size":
            int(
                sizes.max()
            ),

        "largest_community_share_active":
            float(
                sizes.max()
                /
                active_node_count
            ),

        "community_size_median":
            float(
                np.median(
                    sizes
                )
            ),

        "community_size_p90":
            float(
                np.quantile(
                    sizes,
                    0.90,
                )
            ),
    }


# ============================================================
# 16. Build Community Membership Table
# ============================================================

def build_community_membership(
    nodes,
    community_map,
    network_id,
    date,
    window,
    density,
):

    df = nodes.copy()

    df[
        "community_id"
    ] = (

        df[
            "master_index"
        ]
        .map(
            community_map
        )
        .fillna(
            -1
        )
        .astype(
            np.int32
        )
    )

    df[
        "is_active_network_node"
    ] = (
        df[
            "community_id"
        ]
        >=
        0
    )

    df[
        "network_id"
    ] = network_id

    df[
        "analysis_date"
    ] = pd.Timestamp(
        date
    )

    df[
        "window"
    ] = int(
        window
    )

    df[
        "density_fraction"
    ] = float(
        density
    )

    return df


# ============================================================
# 17. Save Community Membership
# ============================================================

def save_community_membership(
    membership,
    network_id,
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
        COMMUNITY_ROOT
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

    temp = Path(
        str(
            path
        )
        +
        ".tmp.parquet"
    )

    membership.to_parquet(
        temp,
        index=False,
        compression="zstd",
    )

    os.replace(
        temp,
        path,
    )


# ============================================================
# 18. Analyze One Network
# ============================================================

def analyze_one_network(
    nodes,
    edges,
    network_id,
    date,
    window,
    density,
    exact_industry1_baseline,
):

    G = build_graph(
        nodes=
            nodes,

        edges=
            edges,
    )

    n_nodes = G.number_of_nodes()

    n_edges = G.number_of_edges()

    active_node_count = sum(

        1

        for _, degree
        in G.degree()

        if degree > 0
    )

    active_node_share = (

        active_node_count
        /
        n_nodes

        if n_nodes > 0

        else np.nan
    )

    # ========================================================
    # Industry ID1
    # ========================================================

    (
        same_share_id1,
        labeled_edges_id1,
        same_edges_id1,
    ) = (
        same_industry_edge_share(

            edges=
                edges,

            nodes=
                nodes,

            industry_column=
                "industry_id1",
        )
    )

    assort_id1 = (
        categorical_assortativity(

            edges=
                edges,

            nodes=
                nodes,

            industry_column=
                "industry_id1",
        )
    )

    enrichment_id1 = np.nan

    if (
        np.isfinite(
            exact_industry1_baseline
        )
        and
        exact_industry1_baseline
        >
        0
    ):

        enrichment_id1 = (

            same_share_id1

            /

            exact_industry1_baseline
        )

    # ========================================================
    # Industry ID2
    # ========================================================

    (
        same_share_id2,
        labeled_edges_id2,
        same_edges_id2,
    ) = (
        same_industry_edge_share(

            edges=
                edges,

            nodes=
                nodes,

            industry_column=
                "industry_id2",
        )
    )

    assort_id2 = (
        categorical_assortativity(

            edges=
                edges,

            nodes=
                nodes,

            industry_column=
                "industry_id2",
        )
    )

    # --------------------------------------------------------
    # 注意：
    #
    # ID2 没有 Step 2 exact common-valid-pair denominator。
    #
    # 因此这里只计算 node-pair approximate baseline。
    # --------------------------------------------------------

    baseline_id2_approx = (
        node_pair_same_industry_baseline(

            nodes=
                nodes,

            industry_column=
                "industry_id2",
        )
    )

    enrichment_id2_approx = np.nan

    if (
        np.isfinite(
            baseline_id2_approx
        )
        and
        baseline_id2_approx
        >
        0
    ):

        enrichment_id2_approx = (

            same_share_id2

            /

            baseline_id2_approx
        )

    # ========================================================
    # Louvain communities
    # ========================================================

    (
        communities,
        community_map,
        modularity,
        active_community_nodes,
    ) = (
        detect_louvain_communities(
            G
        )
    )

    community_structure = (
        community_structure_metrics(

            communities=
                communities,

            active_node_count=
                active_community_nodes,
        )
    )

    # ========================================================
    # Community <-> Industry
    # ========================================================

    alignment_id1 = (
        community_industry_alignment(

            nodes=
                nodes,

            community_map=
                community_map,

            industry_column=
                "industry_id1",
        )
    )

    alignment_id2 = (
        community_industry_alignment(

            nodes=
                nodes,

            community_map=
                community_map,

            industry_column=
                "industry_id2",
        )
    )

    result = {

        "analysis_date":
            pd.Timestamp(
                date
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

        "n_nodes":
            int(
                n_nodes
            ),

        "n_edges":
            int(
                n_edges
            ),

        "active_node_count":
            int(
                active_node_count
            ),

        "active_node_share":
            float(
                active_node_share
            ),

        # ====================================================
        # ID1
        # ====================================================

        "industry1_exact_pair_baseline":
            float(
                exact_industry1_baseline
            ),

        "industry1_labeled_edge_count":
            int(
                labeled_edges_id1
            ),

        "industry1_same_edge_count":
            int(
                same_edges_id1
            ),

        "industry1_same_edge_share":
            float(
                same_share_id1
            ),

        "industry1_enrichment_exact":
            float(
                enrichment_id1
            ),

        "industry1_assortativity":
            float(
                assort_id1
            ),

        # ====================================================
        # ID2
        # ====================================================

        "industry2_node_pair_baseline_approx":
            float(
                baseline_id2_approx
            ),

        "industry2_labeled_edge_count":
            int(
                labeled_edges_id2
            ),

        "industry2_same_edge_count":
            int(
                same_edges_id2
            ),

        "industry2_same_edge_share":
            float(
                same_share_id2
            ),

        "industry2_enrichment_approx":
            float(
                enrichment_id2_approx
            ),

        "industry2_assortativity":
            float(
                assort_id2
            ),

        # ====================================================
        # Community
        # ====================================================

        "weighted_modularity":
            float(
                modularity
            ),

        **community_structure,

        # ====================================================
        # Community / Industry ID1
        # ====================================================

        "community_industry1_labeled_active_nodes":
            int(
                alignment_id1[
                    "n_labeled_active_nodes"
                ]
            ),

        "community_industry1_nmi":
            float(
                alignment_id1[
                    "nmi"
                ]
            ),

        "community_industry1_ari":
            float(
                alignment_id1[
                    "ari"
                ]
            ),

        "community_industry1_weighted_purity":
            float(
                alignment_id1[
                    "weighted_purity"
                ]
            ),

        # ====================================================
        # Community / Industry ID2
        # ====================================================

        "community_industry2_labeled_active_nodes":
            int(
                alignment_id2[
                    "n_labeled_active_nodes"
                ]
            ),

        "community_industry2_nmi":
            float(
                alignment_id2[
                    "nmi"
                ]
            ),

        "community_industry2_ari":
            float(
                alignment_id2[
                    "ari"
                ]
            ),

        "community_industry2_weighted_purity":
            float(
                alignment_id2[
                    "weighted_purity"
                ]
            ),
    }

    membership = (
        build_community_membership(

            nodes=
                nodes,

            community_map=
                community_map,

            network_id=
                network_id,

            date=
                date,

            window=
                window,

            density=
                density,
        )
    )

    return (
        result,
        membership,
    )


# ============================================================
# 19. Raw vs Residual Community Partition
# ============================================================

def compare_raw_residual_communities(
    raw_membership,
    residual_membership,
    date,
    window,
    density,
):

    raw = (
        raw_membership[
            [
                "master_index",
                "community_id",
            ]
        ]
        .rename(
            columns={
                "community_id":
                    "community_raw"
            }
        )
    )

    residual = (
        residual_membership[
            [
                "master_index",
                "community_id",
            ]
        ]
        .rename(
            columns={
                "community_id":
                    "community_residual"
            }
        )
    )

    merged = raw.merge(

        residual,

        on="master_index",

        how="inner",

        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Community comparison only where both networks
    # classify the node as active.
    # --------------------------------------------------------

    common_active = (

        (
            merged[
                "community_raw"
            ]
            >=
            0
        )

        &

        (
            merged[
                "community_residual"
            ]
            >=
            0
        )
    )

    x = (
        merged.loc[
            common_active,
            "community_raw",
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    y = (
        merged.loc[
            common_active,
            "community_residual",
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    if len(
        x
    ) >= 2:

        nmi = (
            normalized_mutual_info_score(
                x,
                y,
            )
        )

        ari = (
            adjusted_rand_score(
                x,
                y,
            )
        )

    else:

        nmi = np.nan

        ari = np.nan

    raw_active = (
        merged[
            "community_raw"
        ]
        >=
        0
    )

    residual_active = (
        merged[
            "community_residual"
        ]
        >=
        0
    )

    active_intersection = int(
        (
            raw_active
            &
            residual_active
        )
        .sum()
    )

    active_union = int(
        (
            raw_active
            |
            residual_active
        )
        .sum()
    )

    active_jaccard = (

        active_intersection
        /
        active_union

        if active_union > 0

        else np.nan
    )

    return {

        "analysis_date":
            pd.Timestamp(
                date
            ),

        "window":
            int(
                window
            ),

        "density_fraction":
            float(
                density
            ),

        "common_active_node_count":
            int(
                len(
                    x
                )
            ),

        "raw_residual_active_node_jaccard":
            float(
                active_jaccard
            ),

        "raw_residual_community_nmi":
            float(
                nmi
            ),

        "raw_residual_community_ari":
            float(
                ari
            ),
    }


# ============================================================
# 20. Window Summary
# ============================================================

def build_window_summary(
    detail,
):

    metrics = [

        "active_node_share",

        "industry1_same_edge_share",
        "industry1_enrichment_exact",
        "industry1_assortativity",

        "industry2_same_edge_share",
        "industry2_assortativity",

        "weighted_modularity",

        "community_count",
        "largest_community_share_active",

        "community_industry1_nmi",
        "community_industry1_ari",
        "community_industry1_weighted_purity",

        "community_industry2_nmi",
        "community_industry2_ari",
        "community_industry2_weighted_purity",
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

    result = (
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
# 21. Raw-Residual Difference Detail
# ============================================================

def build_raw_residual_difference(
    detail,
):

    metrics = [

        "active_node_share",

        "industry1_same_edge_share",
        "industry1_enrichment_exact",
        "industry1_assortativity",

        "industry2_same_edge_share",
        "industry2_assortativity",

        "weighted_modularity",

        "community_count",
        "largest_community_share_active",

        "community_industry1_nmi",
        "community_industry1_ari",
        "community_industry1_weighted_purity",

        "community_industry2_nmi",
        "community_industry2_ari",
        "community_industry2_weighted_purity",
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

            for metric in metrics
        }
    )

    residual = residual.rename(
        columns={

            metric:
                f"{metric}_residual"

            for metric in metrics
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
# 22. Difference Window Summary
# ============================================================

def build_difference_window_summary(
    difference,
):

    diff_columns = [

        col

        for col in difference.columns

        if col.endswith(
            "_residual_minus_raw"
        )
    ]

    rows = []

    for window, group in (
        difference.groupby(
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

        for column in (
            diff_columns
        ):

            values = (
                group[
                    column
                ]
                .dropna()
            )

            row[
                f"{column}_mean"
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
                f"{column}_median"
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
                f"{column}_share_positive"
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
# 23. Plot Time Series
# ============================================================

def plot_metric(
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
# 24. Save Metadata
# ============================================================

def save_metadata(
    density,
):

    metadata = {

        "research_day":
            4,

        "step":
            (
                "Step4_Industry_Community_Validation"
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

        "community_method":
            "Louvain",

        "community_resolution":
            float(
                LOUVAIN_RESOLUTION
            ),

        "community_seed":
            int(
                COMMUNITY_RANDOM_SEED
            ),

        "main_industry_level":
            "industry_id1",

        "robustness_industry_level":
            "industry_id2",

        "important_notes": [

            (
                "Industry labels are never used "
                "for edge selection or community detection."
            ),

            (
                "Industry_id1 enrichment uses the exact "
                "common-valid-pair denominator generated in Step 2."
            ),

            (
                "Industry_id2 enrichment uses a node-pair "
                "approximation because Step 2 did not save an "
                "exact common-valid-pair industry_id2 denominator."
            ),

            (
                "Community alignment excludes isolated nodes "
                "because isolates contain no edge-based "
                "community information."
            ),

            (
                "NMI/ARI measure alignment rather than causal "
                "or statistical significance."
            ),

            (
                "Density robustness remains a separate Step 6."
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        /
        "step4_industry_community_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
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
        "Day 4 - Step 4"
    )

    print(
        "Industry / Community Validation"
    )

    print("=" * 80)

    # ========================================================
    # Load design
    # ========================================================

    metadata = (
        load_step2_metadata()
    )

    matched_summary = (
        load_matched_summary()
    )

    density = (
        resolve_density(
            metadata
        )
    )

    jobs = (
        select_jobs(

            matched_summary=
                matched_summary,

            density=
                density,
        )
    )

    print()
    print(
        f"Density = "
        f"{100*density:.2f}%"
    )

    print(
        f"Date × Window jobs = "
        f"{len(jobs):,}"
    )

    detail_rows = []

    partition_rows = []

    total_jobs = len(
        jobs
    )

    # ========================================================
    # All Date × Window
    # ========================================================

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

        exact_id1_baseline = float(
            row.common_same_industry_pair_share
        )

        print()
        print("=" * 80)

        print(
            f"Job "
            f"{job_no}/{total_jobs}"
        )

        print(
            f"W={window} | "
            f"{date.date()} | "
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

        memberships = {}

        # ====================================================
        # Raw + Residual
        # ====================================================

        for network_id in (
            NETWORK_IDS
        ):

            print(
                f"  Analyze "
                f"{network_id}"
            )

            edges = (
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

            (
                result,
                membership,
            ) = analyze_one_network(

                nodes=
                    nodes,

                edges=
                    edges,

                network_id=
                    network_id,

                date=
                    date,

                window=
                    window,

                density=
                    density,

                exact_industry1_baseline=
                    exact_id1_baseline,
            )

            detail_rows.append(
                result
            )

            memberships[
                network_id
            ] = membership

            save_community_membership(

                membership=
                    membership,

                network_id=
                    network_id,

                date=
                    date,

                window=
                    window,
            )

        # ====================================================
        # Raw vs Residual community comparison
        # ====================================================

        partition_comparison = (
            compare_raw_residual_communities(

                raw_membership=
                    memberships[
                        "B0_RAW_POSITIVE"
                    ],

                residual_membership=
                    memberships[
                        "M1_RESIDUAL_POSITIVE"
                    ],

                date=
                    date,

                window=
                    window,

                density=
                    density,
            )
        )

        partition_rows.append(
            partition_comparison
        )

    # ========================================================
    # Save detailed result
    # ========================================================

    detail = pd.DataFrame(
        detail_rows
    )

    detail = (
        detail.sort_values(
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
        "industry_community_validation_detail.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # Window summary
    # ========================================================

    window_summary = (
        build_window_summary(
            detail
        )
    )

    window_summary.to_csv(

        OUTPUT_DIR
        /
        "industry_community_validation_window_summary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # Raw vs residual differences
    # ========================================================

    difference = (
        build_raw_residual_difference(
            detail
        )
    )

    difference.to_csv(

        OUTPUT_DIR
        /
        "raw_residual_industry_community_difference_detail.csv",

        index=False,

        encoding="utf-8-sig",
    )

    difference_summary = (
        build_difference_window_summary(
            difference
        )
    )

    difference_summary.to_csv(

        OUTPUT_DIR
        /
        "raw_residual_industry_community_difference_window_summary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # Raw vs Residual community partition
    # ========================================================

    partition_df = pd.DataFrame(
        partition_rows
    )

    partition_df = (
        partition_df.sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    partition_df.to_csv(

        OUTPUT_DIR
        /
        "raw_residual_community_partition_comparison.csv",

        index=False,

        encoding="utf-8-sig",
    )

    partition_summary = (

        partition_df.groupby(
            "window",
            as_index=False,
        )
        .agg(

            mean_common_active_node_count=
                (
                    "common_active_node_count",
                    "mean",
                ),

            mean_active_node_jaccard=
                (
                    "raw_residual_active_node_jaccard",
                    "mean",
                ),

            mean_community_nmi=
                (
                    "raw_residual_community_nmi",
                    "mean",
                ),

            median_community_nmi=
                (
                    "raw_residual_community_nmi",
                    "median",
                ),

            mean_community_ari=
                (
                    "raw_residual_community_ari",
                    "mean",
                ),

            median_community_ari=
                (
                    "raw_residual_community_ari",
                    "median",
                ),
        )
    )

    partition_summary.to_csv(

        OUTPUT_DIR
        /
        "raw_residual_community_partition_window_summary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # Figures
    # ========================================================

    plot_metric(

        detail=
            detail,

        metric=
            "industry1_enrichment_exact",

        ylabel=
            "Industry-1 Edge Enrichment",

        filename_prefix=
            "industry1_enrichment",
    )

    plot_metric(

        detail=
            detail,

        metric=
            "industry1_assortativity",

        ylabel=
            "Industry-1 Assortativity",

        filename_prefix=
            "industry1_assortativity",
    )

    plot_metric(

        detail=
            detail,

        metric=
            "weighted_modularity",

        ylabel=
            "Weighted Louvain Modularity",

        filename_prefix=
            "weighted_modularity",
    )

    plot_metric(

        detail=
            detail,

        metric=
            "community_industry1_nmi",

        ylabel=
            "Community-Industry NMI",

        filename_prefix=
            "community_industry1_nmi",
    )

    # ========================================================
    # Metadata
    # ========================================================

    save_metadata(
        density=
            density
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Industry / Community Window Summary"
    )

    print("=" * 80)

    show_cols = [

        "window",
        "network_id",

        "active_node_share_mean",

        "industry1_same_edge_share_mean",
        "industry1_enrichment_exact_mean",
        "industry1_assortativity_mean",

        "weighted_modularity_mean",

        "community_industry1_nmi_mean",
        "community_industry1_ari_mean",
        "community_industry1_weighted_purity_mean",
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
        "Raw - Residual Community Partition"
    )

    print("=" * 80)

    print(
        partition_summary
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 80)

    print(
        "Day 4 Step 4 Complete"
    )

    print("=" * 80)

    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()