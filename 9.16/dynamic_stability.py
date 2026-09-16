from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
# Step 3
# ------------------------------------------------------------

STEP3_DIR = (
    OUTPUT_ROOT
    / "M1_day4"
    / "03_step3_structural_diagnostics"
)

NODE_METRIC_ROOT = (
    STEP3_DIR
    / "node_metrics"
)


# ------------------------------------------------------------
# Step 4
# ------------------------------------------------------------

STEP4_DIR = (
    OUTPUT_ROOT
    / "M1_day4"
    / "04_step4_industry_community_validation"
)

COMMUNITY_ROOT = (
    STEP4_DIR
    / "community_membership"
)


# ------------------------------------------------------------
# Step 5 output
# ------------------------------------------------------------

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day4"
    / "05_step5_dynamic_stability"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. Settings
# ============================================================

# ------------------------------------------------------------
# Step 5 正式分析 primary q。
#
# 正常应为：
#
# q = 0.01
# ------------------------------------------------------------

OVERRIDE_DENSITY = None


# ------------------------------------------------------------
# Smoke test:
#
# None:
#     正式全部月份
#
# 例如：
#     TEST_LAST_N_DATES = 6
#
# 每个 W 仅使用最后 6 个日期，
# 可快速检查代码。
# ------------------------------------------------------------

TEST_LAST_N_DATES = None


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

        return json.load(f)


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
        "edge_budget_k",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"matched summary 缺字段：{missing}"
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

    return (
        df.sort_values(
            [
                "window",
                "analysis_date",
                "density_fraction",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# 4. Resolve Primary Density
# ============================================================

def resolve_density(
    metadata,
):

    if OVERRIDE_DENSITY is not None:

        return float(
            OVERRIDE_DENSITY
        )

    return float(
        metadata[
            "primary_density"
        ]
    )


# ============================================================
# 5. Select Dates
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
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(drop=True)
    )

    if TEST_LAST_N_DATES is not None:

        jobs = (
            jobs.groupby(
                "window",
                group_keys=False,
            )
            .tail(
                int(
                    TEST_LAST_N_DATES
                )
            )
            .reset_index(drop=True)
        )

    return jobs


# ============================================================
# 6. Canonical Edge Keys
#
# 使用：
#
# min(master_i, master_j)
# max(master_i, master_j)
#
# 避免 i-j 与 j-i 被当作不同边。
# ============================================================

def canonical_edge_arrays(
    edges,
):

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

    a = np.minimum(
        i,
        j,
    )

    b = np.maximum(
        i,
        j,
    )

    return (
        a,
        b,
    )


def edge_key_set(
    edges,
    multiplier,
):

    a, b = (
        canonical_edge_arrays(
            edges
        )
    )

    keys = (
        a
        *
        multiplier
        +
        b
    )

    return set(
        keys.tolist()
    )


# ============================================================
# 7. Load Nodes
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

    df = pd.read_parquet(
        path
    )

    required = [
        "master_index",
        "industry_id1",
        "industry_id2",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Node file 缺字段：{missing}"
        )

    return df


# ============================================================
# 8. Load Step 2 Edges
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
            f"Edge file 不存在：\n{path}"
        )

    df = pd.read_parquet(
        path
    )

    df = (
        df[
            df["rank"]
            <= int(edge_budget)
        ]
        .sort_values("rank")
        .reset_index(drop=True)
    )

    if len(df) != int(edge_budget):

        raise RuntimeError(
            f"{network_id} "
            f"W={window} "
            f"{date_string}: "
            f"edge count={len(df):,}, "
            f"expected={edge_budget:,}"
        )

    return df


# ============================================================
# 9. Load Step 3 Node Metrics
# ============================================================

def load_node_metrics(
    network_id,
    date,
    window,
):

    date_string = (
        pd.Timestamp(date)
        .strftime("%Y-%m-%d")
    )

    path = (
        NODE_METRIC_ROOT
        / network_id
        / f"W{window}"
        / f"{date_string}.parquet"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Step 3 node metric 不存在：\n"
            f"{path}"
        )

    df = pd.read_parquet(
        path
    )

    required = [
        "master_index",
        "degree",
        "strength",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Node metrics 缺字段：{missing}"
        )

    return df


# ============================================================
# 10. Load Step 4 Community Membership
# ============================================================

def load_community_membership(
    network_id,
    date,
    window,
):

    date_string = (
        pd.Timestamp(date)
        .strftime("%Y-%m-%d")
    )

    path = (
        COMMUNITY_ROOT
        / network_id
        / f"W{window}"
        / f"{date_string}.parquet"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Step 4 community file 不存在：\n"
            f"{path}"
        )

    df = pd.read_parquet(
        path
    )

    required = [
        "master_index",
        "community_id",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Community file 缺字段：{missing}"
        )

    return df


# ============================================================
# 11. Filter Edges to Common Nodes
# ============================================================

def restrict_edges_to_nodes(
    edges,
    common_nodes,
):

    node_set = set(
        common_nodes
    )

    mask = (
        edges[
            "master_index_i"
        ].isin(node_set)

        &

        edges[
            "master_index_j"
        ].isin(node_set)
    )

    return (
        edges.loc[
            mask
        ]
        .copy()
        .reset_index(drop=True)
    )


# ============================================================
# 12. Edge Stability
# ============================================================

def calculate_edge_stability(
    edges_t,
    edges_t1,
    nodes_t,
    nodes_t1,
    multiplier,
):

    # ========================================================
    # Full edge stability
    #
    # 包含 node entry/exit 的影响。
    # ========================================================

    full_t = (
        edge_key_set(
            edges_t,
            multiplier,
        )
    )

    full_t1 = (
        edge_key_set(
            edges_t1,
            multiplier,
        )
    )

    full_intersection = (
        full_t
        &
        full_t1
    )

    full_union = (
        full_t
        |
        full_t1
    )

    full_jaccard = (

        len(
            full_intersection
        )
        /
        len(
            full_union
        )

        if len(full_union) > 0

        else np.nan
    )

    # ========================================================
    # Common nodes
    # ========================================================

    V_t = set(
        nodes_t[
            "master_index"
        ]
        .astype(int)
        .tolist()
    )

    V_t1 = set(
        nodes_t1[
            "master_index"
        ]
        .astype(int)
        .tolist()
    )

    V_common = (
        V_t
        &
        V_t1
    )

    V_union = (
        V_t
        |
        V_t1
    )

    node_jaccard = (

        len(V_common)
        /
        len(V_union)

        if len(V_union) > 0

        else np.nan
    )

    # ========================================================
    # Restrict both edge sets to common node universe
    #
    # 这是 Step 5 的主 edge stability definition。
    # ========================================================

    edges_t_common = (
        restrict_edges_to_nodes(
            edges_t,
            V_common,
        )
    )

    edges_t1_common = (
        restrict_edges_to_nodes(
            edges_t1,
            V_common,
        )
    )

    E_t = (
        edge_key_set(
            edges_t_common,
            multiplier,
        )
    )

    E_t1 = (
        edge_key_set(
            edges_t1_common,
            multiplier,
        )
    )

    intersection = (
        E_t
        &
        E_t1
    )

    union = (
        E_t
        |
        E_t1
    )

    n_intersection = len(
        intersection
    )

    common_jaccard = (

        n_intersection
        /
        len(union)

        if len(union) > 0

        else np.nan
    )

    # --------------------------------------------------------
    # Directional retention:
    #
    # t -> t+1
    # --------------------------------------------------------

    forward_retention = (

        n_intersection
        /
        len(E_t)

        if len(E_t) > 0

        else np.nan
    )

    backward_retention = (

        n_intersection
        /
        len(E_t1)

        if len(E_t1) > 0

        else np.nan
    )

    # --------------------------------------------------------
    # Symmetric Dice-style retention
    # --------------------------------------------------------

    symmetric_retention = (

        2
        *
        n_intersection
        /
        (
            len(E_t)
            +
            len(E_t1)
        )

        if (
            len(E_t)
            +
            len(E_t1)
        ) > 0

        else np.nan
    )

    lost_count = (
        len(E_t)
        -
        n_intersection
    )

    gained_count = (
        len(E_t1)
        -
        n_intersection
    )

    gross_changes = (
        lost_count
        +
        gained_count
    )

    delta_edges = (
        len(E_t1)
        -
        len(E_t)
    )

    common_turnover = (

        1.0
        -
        common_jaccard

        if np.isfinite(
            common_jaccard
        )

        else np.nan
    )

    return {

        "node_count_t":
            len(V_t),

        "node_count_t1":
            len(V_t1),

        "common_node_count":
            len(V_common),

        "node_jaccard":
            node_jaccard,

        "full_edge_count_t":
            len(full_t),

        "full_edge_count_t1":
            len(full_t1),

        "full_edge_jaccard":
            full_jaccard,

        "common_node_edge_count_t":
            len(E_t),

        "common_node_edge_count_t1":
            len(E_t1),

        "common_edge_intersection_count":
            n_intersection,

        "common_edge_jaccard":
            common_jaccard,

        "forward_edge_retention":
            forward_retention,

        "backward_edge_retention":
            backward_retention,

        "symmetric_edge_retention":
            symmetric_retention,

        "edge_turnover":
            common_turnover,

        "lost_edge_count":
            lost_count,

        "gained_edge_count":
            gained_count,

        "gross_edge_changes":
            gross_changes,

        "delta_edge_count":
            delta_edges,

        "_common_nodes":
            V_common,

        "_persistent_edge_keys":
            intersection,

        "_edges_t_common":
            edges_t_common,

        "_edges_t1_common":
            edges_t1_common,
    }


# ============================================================
# 13. Rank Stability
# ============================================================

def safe_spearman(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    if valid.sum() < 3:

        return np.nan

    x = x[valid]
    y = y[valid]

    if (
        np.std(x) == 0
        or
        np.std(y) == 0
    ):

        return np.nan

    return float(
        spearmanr(
            x,
            y,
        ).statistic
    )


# ============================================================
# 14. Hub Overlap
# ============================================================

def top_hub_set(
    df,
    metric,
    fraction,
):

    n = len(
        df
    )

    if n == 0:

        return set()

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
        df.nlargest(
            k,
            metric,
        )[
            "master_index"
        ]
        .astype(int)
        .tolist()
    )

    return set(top)


def hub_overlap(
    df_t,
    df_t1,
    metric,
    fraction,
):

    H_t = top_hub_set(
        df_t,
        metric,
        fraction,
    )

    H_t1 = top_hub_set(
        df_t1,
        metric,
        fraction,
    )

    inter = (
        H_t
        &
        H_t1
    )

    union = (
        H_t
        |
        H_t1
    )

    overlap = (

        len(inter)
        /
        len(H_t)

        if len(H_t) > 0

        else np.nan
    )

    jaccard = (

        len(inter)
        /
        len(union)

        if len(union) > 0

        else np.nan
    )

    return (
        overlap,
        jaccard,
    )


# ============================================================
# 15. Node Stability
# ============================================================

def calculate_node_stability(
    metrics_t,
    metrics_t1,
):

    merged = (
        metrics_t[
            [
                "master_index",
                "degree",
                "strength",
            ]
        ]
        .merge(
            metrics_t1[
                [
                    "master_index",
                    "degree",
                    "strength",
                ]
            ],
            on="master_index",
            how="inner",
            suffixes=(
                "_t",
                "_t1",
            ),
            validate="one_to_one",
        )
    )

    degree_spearman = (
        safe_spearman(
            merged[
                "degree_t"
            ],
            merged[
                "degree_t1"
            ],
        )
    )

    strength_spearman = (
        safe_spearman(
            merged[
                "strength_t"
            ],
            merged[
                "strength_t1"
            ],
        )
    )

    # --------------------------------------------------------
    # Hub overlap:
    #
    # 必须先限制在 common nodes。
    # --------------------------------------------------------

    t_common = (
        metrics_t[
            metrics_t[
                "master_index"
            ]
            .isin(
                merged[
                    "master_index"
                ]
            )
        ]
        .copy()
    )

    t1_common = (
        metrics_t1[
            metrics_t1[
                "master_index"
            ]
            .isin(
                merged[
                    "master_index"
                ]
            )
        ]
        .copy()
    )

    result = {

        "common_node_count":
            int(
                len(
                    merged
                )
            ),

        "degree_rank_spearman":
            degree_spearman,

        "strength_rank_spearman":
            strength_spearman,
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

        (
            overlap_degree,
            jaccard_degree,
        ) = hub_overlap(

            t_common,
            t1_common,
            "degree",
            fraction,
        )

        (
            overlap_strength,
            jaccard_strength,
        ) = hub_overlap(

            t_common,
            t1_common,
            "strength",
            fraction,
        )

        result[
            f"{label}_degree_hub_retention"
        ] = overlap_degree

        result[
            f"{label}_degree_hub_jaccard"
        ] = jaccard_degree

        result[
            f"{label}_strength_hub_retention"
        ] = overlap_strength

        result[
            f"{label}_strength_hub_jaccard"
        ] = jaccard_strength

    return result


# ============================================================
# 16. Community Temporal Stability
# ============================================================

def calculate_community_stability(
    membership_t,
    membership_t1,
):

    merged = (
        membership_t[
            [
                "master_index",
                "community_id",
            ]
        ]
        .merge(
            membership_t1[
                [
                    "master_index",
                    "community_id",
                ]
            ],
            on="master_index",
            how="inner",
            suffixes=(
                "_t",
                "_t1",
            ),
            validate="one_to_one",
        )
    )

    # --------------------------------------------------------
    # Both dates must have active community assignment.
    #
    # community_id = -1 means isolate.
    # --------------------------------------------------------

    common_active = (

        (
            merged[
                "community_id_t"
            ]
            >=
            0
        )

        &

        (
            merged[
                "community_id_t1"
            ]
            >=
            0
        )
    )

    x = (
        merged.loc[
            common_active,
            "community_id_t",
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    y = (
        merged.loc[
            common_active,
            "community_id_t1",
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    if len(x) >= 2:

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

    active_t = set(
        merged.loc[
            merged[
                "community_id_t"
            ]
            >=
            0,
            "master_index",
        ]
        .astype(int)
        .tolist()
    )

    active_t1 = set(
        merged.loc[
            merged[
                "community_id_t1"
            ]
            >=
            0,
            "master_index",
        ]
        .astype(int)
        .tolist()
    )

    active_union = (
        active_t
        |
        active_t1
    )

    active_intersection = (
        active_t
        &
        active_t1
    )

    active_jaccard = (

        len(
            active_intersection
        )
        /
        len(
            active_union
        )

        if len(
            active_union
        ) > 0

        else np.nan
    )

    return {

        "common_active_node_count":
            int(
                len(x)
            ),

        "active_node_jaccard":
            active_jaccard,

        "community_temporal_nmi":
            float(
                nmi
            ),

        "community_temporal_ari":
            float(
                ari
            ),
    }


# ============================================================
# 17. Same / Cross Industry Retention
#
# 用 t 时点行业标签定义 initial edge category。
#
# 回答：
#
# t 时刻的 Same/Cross edges，
# 有多少在 t+1 仍然存在？
# ============================================================

def calculate_same_cross_retention(
    edges_t_common,
    persistent_edge_keys,
    nodes_t,
    multiplier,
):

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

    i = (
        edges_t_common[
            "master_index_i"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    j = (
        edges_t_common[
            "master_index_j"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    a = np.minimum(
        i,
        j,
    )

    b = np.maximum(
        i,
        j,
    )

    keys = (
        a
        *
        multiplier
        +
        b
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
            persistent_edge_keys
            for key in keys
        ],
        dtype=bool,
    )

    same_count = int(
        same.sum()
    )

    cross_count = int(
        cross.sum()
    )

    same_persistent = int(
        (
            same
            &
            persistent
        ).sum()
    )

    cross_persistent = int(
        (
            cross
            &
            persistent
        ).sum()
    )

    same_retention = (

        same_persistent
        /
        same_count

        if same_count > 0

        else np.nan
    )

    cross_retention = (

        cross_persistent
        /
        cross_count

        if cross_count > 0

        else np.nan
    )

    return {

        "initial_same_edge_count":
            same_count,

        "initial_cross_edge_count":
            cross_count,

        "persistent_same_edge_count":
            same_persistent,

        "persistent_cross_edge_count":
            cross_persistent,

        "same_industry_edge_retention":
            same_retention,

        "cross_industry_edge_retention":
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


# ============================================================
# 18. One Adjacent-date Pair
# ============================================================

def analyze_adjacent_pair(
    network_id,
    row_t,
    row_t1,
    multiplier,
    density,
):

    date_t = pd.Timestamp(
        row_t.analysis_date
    )

    date_t1 = pd.Timestamp(
        row_t1.analysis_date
    )

    window = int(
        row_t.window
    )

    K_t = int(
        row_t.edge_budget_k
    )

    K_t1 = int(
        row_t1.edge_budget_k
    )

    # ========================================================
    # Nodes
    # ========================================================

    nodes_t = load_nodes(
        date_t,
        window,
    )

    nodes_t1 = load_nodes(
        date_t1,
        window,
    )

    # ========================================================
    # Edges
    # ========================================================

    edges_t = load_edges(
        network_id,
        date_t,
        window,
        K_t,
    )

    edges_t1 = load_edges(
        network_id,
        date_t1,
        window,
        K_t1,
    )

    # ========================================================
    # Edge stability
    # ========================================================

    edge_result = (
        calculate_edge_stability(

            edges_t=
                edges_t,

            edges_t1=
                edges_t1,

            nodes_t=
                nodes_t,

            nodes_t1=
                nodes_t1,

            multiplier=
                multiplier,
        )
    )

    # ========================================================
    # Node metric stability
    # ========================================================

    metrics_t = (
        load_node_metrics(
            network_id,
            date_t,
            window,
        )
    )

    metrics_t1 = (
        load_node_metrics(
            network_id,
            date_t1,
            window,
        )
    )

    node_result = (
        calculate_node_stability(

            metrics_t=
                metrics_t,

            metrics_t1=
                metrics_t1,
        )
    )

    # ========================================================
    # Community temporal stability
    # ========================================================

    community_t = (
        load_community_membership(

            network_id,
            date_t,
            window,
        )
    )

    community_t1 = (
        load_community_membership(

            network_id,
            date_t1,
            window,
        )
    )

    community_result = (
        calculate_community_stability(

            membership_t=
                community_t,

            membership_t1=
                community_t1,
        )
    )

    # ========================================================
    # Industry persistence
    # ========================================================

    industry_result = (
        calculate_same_cross_retention(

            edges_t_common=
                edge_result[
                    "_edges_t_common"
                ],

            persistent_edge_keys=
                edge_result[
                    "_persistent_edge_keys"
                ],

            nodes_t=
                nodes_t,

            multiplier=
                multiplier,
        )
    )

    # ========================================================
    # Remove private objects before CSV
    # ========================================================

    public_edge_result = {

        key:
            value

        for key, value
        in edge_result.items()

        if not key.startswith(
            "_"
        )
    }

    result = {

        "window":
            window,

        "density_fraction":
            float(
                density
            ),

        "network_id":
            network_id,

        "date_t":
            date_t,

        "date_t1":
            date_t1,

        "calendar_gap_days":
            int(
                (
                    date_t1
                    -
                    date_t
                ).days
            ),

        "edge_budget_t":
            K_t,

        "edge_budget_t1":
            K_t1,

        **public_edge_result,

        **{

            f"node_{key}":
                value

            for key, value
            in node_result.items()
        },

        **{

            f"community_{key}":
                value

            for key, value
            in community_result.items()
        },

        **industry_result,
    }

    return result


# ============================================================
# 19. Window Summary
# ============================================================

def build_window_summary(
    detail,
):

    metrics = [

        # Edge
        "node_jaccard",

        "full_edge_jaccard",
        "common_edge_jaccard",

        "forward_edge_retention",
        "backward_edge_retention",
        "symmetric_edge_retention",

        "edge_turnover",

        # Node centrality
        "node_degree_rank_spearman",
        "node_strength_rank_spearman",

        "node_top1pct_degree_hub_retention",
        "node_top5pct_degree_hub_retention",

        "node_top1pct_strength_hub_retention",
        "node_top5pct_strength_hub_retention",

        # Community
        "community_active_node_jaccard",
        "community_community_temporal_nmi",
        "community_community_temporal_ari",

        # Industry
        "same_industry_edge_retention",
        "cross_industry_edge_retention",
        "same_minus_cross_retention",
    ]

    rows = []

    for (
        window,
        network_id
    ), group in detail.groupby(
        [
            "window",
            "network_id",
        ]
    ):

        row = {

            "window":
                int(
                    window
                ),

            "network_id":
                network_id,

            "transition_count":
                int(
                    len(group)
                ),
        }

        for metric in metrics:

            if metric not in group.columns:

                continue

            values = (
                pd.to_numeric(
                    group[
                        metric
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            row[
                f"{metric}_mean"
            ] = (
                float(
                    values.mean()
                )
                if len(values) > 0
                else np.nan
            )

            row[
                f"{metric}_median"
            ] = (
                float(
                    values.median()
                )
                if len(values) > 0
                else np.nan
            )

            row[
                f"{metric}_p10"
            ] = (
                float(
                    values.quantile(
                        0.10
                    )
                )
                if len(values) > 0
                else np.nan
            )

            row[
                f"{metric}_p90"
            ] = (
                float(
                    values.quantile(
                        0.90
                    )
                )
                if len(values) > 0
                else np.nan
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 20. Raw - Residual Dynamic Difference
# ============================================================

def build_raw_residual_difference(
    detail,
):

    metrics = [

        "common_edge_jaccard",
        "forward_edge_retention",
        "symmetric_edge_retention",

        "node_degree_rank_spearman",
        "node_strength_rank_spearman",

        "node_top1pct_degree_hub_retention",
        "node_top5pct_degree_hub_retention",

        "community_community_temporal_nmi",
        "community_community_temporal_ari",

        "same_industry_edge_retention",
        "cross_industry_edge_retention",
        "same_minus_cross_retention",
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
                "window",
                "date_t",
                "date_t1",
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
                "window",
                "date_t",
                "date_t1",
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
            "window",
            "date_t",
            "date_t1",
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
# 21. Difference Summary
# ============================================================

def build_difference_summary(
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

            "transition_count":
                int(
                    len(group)
                ),
        }

        for column in diff_columns:

            values = (
                pd.to_numeric(
                    group[
                        column
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            row[
                f"{column}_mean"
            ] = (
                float(
                    values.mean()
                )
                if len(values) > 0
                else np.nan
            )

            row[
                f"{column}_median"
            ] = (
                float(
                    values.median()
                )
                if len(values) > 0
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
                if len(values) > 0
                else np.nan
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 22. Plot Dynamic Metric
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
                "date_t1"
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
                    "date_t1"
                ],

                group[
                    metric
                ],

                label=
                    label,
            )

        ax.set_xlabel(
            "Network Date"
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
# 23. Same vs Cross Retention Plot
# ============================================================

def plot_same_cross_retention(
    detail,
):

    for network_id in NETWORK_IDS:

        for window in sorted(
            detail[
                "window"
            ]
            .unique()
        ):

            subset = (
                detail[
                    (
                        detail[
                            "network_id"
                        ]
                        ==
                        network_id
                    )
                    &
                    (
                        detail[
                            "window"
                        ]
                        ==
                        window
                    )
                ]
                .sort_values(
                    "date_t1"
                )
            )

            if subset.empty:

                continue

            fig, ax = plt.subplots(
                figsize=(
                    11,
                    5.5,
                )
            )

            ax.plot(
                subset[
                    "date_t1"
                ],
                subset[
                    "same_industry_edge_retention"
                ],
                label=
                    "Same-industry",
            )

            ax.plot(
                subset[
                    "date_t1"
                ],
                subset[
                    "cross_industry_edge_retention"
                ],
                label=
                    "Cross-industry",
            )

            ax.set_xlabel(
                "Network Date"
            )

            ax.set_ylabel(
                "Forward Edge Retention"
            )

            ax.set_title(
                f"Same vs Cross Edge Retention | "
                f"{network_id} | W={window}"
            )

            ax.legend()

            ax.grid(
                alpha=0.25
            )

            fig.tight_layout()

            safe_name = (
                network_id
                .lower()
            )

            fig.savefig(

                FIGURE_DIR
                /
                (
                    f"same_cross_retention_"
                    f"{safe_name}_"
                    f"W{window}.png"
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

    payload = {

        "research_day":
            4,

        "step":
            "Step5_Dynamic_Stability",

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

        "primary_edge_stability":
            (
                "Adjacent-date edge Jaccard and retention "
                "after restricting both networks to the "
                "intersection of their node universes."
            ),

        "auxiliary_edge_stability":
            (
                "Full edge-set Jaccard before common-node "
                "adjustment."
            ),

        "node_stability":
            (
                "Adjacent-date Spearman stability of degree "
                "and strength on common nodes."
            ),

        "community_stability":
            (
                "Adjacent-date NMI and ARI of Louvain "
                "community assignments on nodes active in "
                "both dates."
            ),

        "industry_stability":
            (
                "Forward retention of same-industry versus "
                "cross-industry edges, with category defined "
                "using date-t historical industry labels."
            ),

        "important_notes": [

            (
                "Common-node-adjusted edge stability is the "
                "main measure because full-set changes mix "
                "true rewiring with node entry/exit."
            ),

            (
                "The analysis is descriptive; Jaccard, "
                "retention, NMI and ARI are not significance "
                "tests."
            ),

            (
                "Step 5 uses the frozen primary density only. "
                "Density robustness remains Step 6."
            ),

            (
                "Same/cross persistence uses historical "
                "industry_id1 labels at the initial date."
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        /
        "step5_dynamic_stability_metadata.json",
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
        "Day 4 - Step 5"
    )

    print(
        "Dynamic Stability"
    )

    print("=" * 80)

    # ========================================================
    # Inputs
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
            matched_summary,
            density,
        )
    )

    print()
    print(
        f"Density = "
        f"{100*density:.2f}%"
    )

    # --------------------------------------------------------
    # master_index currently roughly <= 6000.
    #
    # Use a safe multiplier larger than any master index.
    # --------------------------------------------------------

    max_master_index = 0

    # Find one set of node files to infer safe multiplier.
    for row in jobs.itertuples(
        index=False
    ):

        temp_nodes = load_nodes(
            row.analysis_date,
            int(
                row.window
            ),
        )

        if not temp_nodes.empty:

            max_master_index = max(
                max_master_index,
                int(
                    temp_nodes[
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

    print(
        f"Edge-key multiplier = "
        f"{multiplier:,}"
    )

    rows = []

    # ========================================================
    # Adjacent network dates
    # ========================================================

    for window, group in (
        jobs.groupby(
            "window"
        )
    ):

        group = (
            group.sort_values(
                "analysis_date"
            )
            .reset_index(
                drop=True
            )
        )

        if len(group) < 2:

            continue

        print()
        print("=" * 80)

        print(
            f"W={window}: "
            f"{len(group)} dates, "
            f"{len(group)-1} transitions"
        )

        print("=" * 80)

        for t in range(
            len(group)
            -
            1
        ):

            row_t = (
                group.iloc[
                    t
                ]
            )

            row_t1 = (
                group.iloc[
                    t + 1
                ]
            )

            print(
                f"{row_t.analysis_date.date()} "
                f"-> "
                f"{row_t1.analysis_date.date()}"
            )

            for network_id in (
                NETWORK_IDS
            ):

                result = (
                    analyze_adjacent_pair(

                        network_id=
                            network_id,

                        row_t=
                            row_t,

                        row_t1=
                            row_t1,

                        multiplier=
                            multiplier,

                        density=
                            density,
                    )
                )

                rows.append(
                    result
                )

    # ========================================================
    # Save detail
    # ========================================================

    detail = pd.DataFrame(
        rows
    )

    detail = (
        detail.sort_values(
            [
                "window",
                "network_id",
                "date_t",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    detail.to_csv(

        OUTPUT_DIR
        /
        "dynamic_stability_detail.csv",

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
        "dynamic_stability_window_summary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # Raw vs Residual
    # ========================================================

    difference = (
        build_raw_residual_difference(
            detail
        )
    )

    difference.to_csv(

        OUTPUT_DIR
        /
        "raw_residual_dynamic_difference_detail.csv",

        index=False,

        encoding="utf-8-sig",
    )

    difference_summary = (
        build_difference_summary(
            difference
        )
    )

    difference_summary.to_csv(

        OUTPUT_DIR
        /
        "raw_residual_dynamic_difference_window_summary.csv",

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
            "common_edge_jaccard",

        ylabel=
            "Adjacent Edge Jaccard",

        filename_prefix=
            "adjacent_edge_jaccard",
    )

    plot_metric(

        detail=
            detail,

        metric=
            "forward_edge_retention",

        ylabel=
            "Forward Edge Retention",

        filename_prefix=
            "forward_edge_retention",
    )

    plot_metric(

        detail=
            detail,

        metric=
            "node_degree_rank_spearman",

        ylabel=
            "Degree Rank Stability",

        filename_prefix=
            "degree_rank_stability",
    )

    plot_metric(

        detail=
            detail,

        metric=
            "node_strength_rank_spearman",

        ylabel=
            "Strength Rank Stability",

        filename_prefix=
            "strength_rank_stability",
    )

    plot_metric(

        detail=
            detail,

        metric=
            "community_community_temporal_nmi",

        ylabel=
            "Community Temporal NMI",

        filename_prefix=
            "community_temporal_nmi",
    )

    plot_same_cross_retention(
        detail
    )

    save_metadata(
        density=
            density
    )

    # ========================================================
    # Console Summary
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Dynamic Stability Window Summary"
    )

    print("=" * 80)

    show_cols = [

        "window",
        "network_id",

        "common_edge_jaccard_mean",

        "forward_edge_retention_mean",

        "node_degree_rank_spearman_mean",
        "node_strength_rank_spearman_mean",

        "community_community_temporal_nmi_mean",

        "same_industry_edge_retention_mean",
        "cross_industry_edge_retention_mean",
        "same_minus_cross_retention_mean",
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
        "Day 4 Step 5 Complete"
    )

    print("=" * 80)

    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()