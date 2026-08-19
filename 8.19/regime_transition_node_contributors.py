from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# 0. 中文字体
# ============================================================

def set_chinese_font():

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP"
    ]

    installed = {
        f.name
        for f in font_manager.fontManager.ttflist
    }

    for font_name in candidates:

        if font_name in installed:

            plt.rcParams["font.sans-serif"] = [
                font_name
            ]

            plt.rcParams["axes.unicode_minus"] = False

            print(
                f"使用中文字体：{font_name}"
            )

            return

    print(
        "警告：未找到常见中文字体。"
    )


set_chinese_font()


# ============================================================
# 1. 路径
# ============================================================

PROJECT_DIR = Path(
    "stock_network"
)

PROCESSED_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)

FIGURE_DIR = (
    PROJECT_DIR
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 输入
# ============================================================

EDGE_HISTORY_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_history.csv"
)

REGIME_ASSIGNMENT_FILE = (
    PROCESSED_DIR
    / "network_regime_assignment.csv"
)

EDGE_CHANGE_DETAIL_FILE = (
    PROCESSED_DIR
    / "adjacent_regime_edge_change_detail.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ============================================================
# 3. 输出
# ============================================================

WINDOW_NODE_FILE = (
    PROCESSED_DIR
    / "regime_transition_window_node_metrics.csv"
)

REGIME_NODE_FILE = (
    PROCESSED_DIR
    / "regime_node_metrics.csv"
)

NODE_CHANGE_FILE = (
    PROCESSED_DIR
    / "adjacent_regime_node_change.csv"
)

NODE_BURDEN_FILE = (
    PROCESSED_DIR
    / "regime_transition_node_edge_burden.csv"
)

CONTRIBUTOR_MASTER_FILE = (
    PROCESSED_DIR
    / "regime_transition_node_contributors.csv"
)

TOP_EDGE_CONTRIBUTORS_FILE = (
    PROCESSED_DIR
    / "top_regime_transition_edge_contributors.csv"
)

TOP_ROLE_SHIFTS_FILE = (
    PROCESSED_DIR
    / "top_regime_transition_role_shifts.csv"
)

R34_FILE = (
    PROCESSED_DIR
    / "R3_R4_node_transition_contributors.csv"
)


# ============================================================
# 4. 参数
# ============================================================

TOP_K = 5
TOP_N = 10

MAJOR_PERSISTENCE_CHANGE = 0.50


# ============================================================
# 5. 工具函数
# ============================================================

def normalize_code(x):

    s = str(
        x
    ).strip()

    if s.endswith(".0"):

        s = s[:-2]

    match = re.search(
        r"(\d{6})",
        s
    )

    if match:

        return match.group(1)

    digits = "".join(
        c
        for c in s
        if c.isdigit()
    )

    if digits:

        return digits.zfill(6)

    return s


def canonical_pair(
    a,
    b
):

    a = normalize_code(a)
    b = normalize_code(b)

    if a <= b:

        return a, b

    return b, a


def convert_bool(
    series,
    name
):

    if series.dtype == bool:

        return series

    converted = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False
            }
        )
    )

    if converted.isna().any():

        bad = (
            series[
                converted.isna()
            ]
            .unique()
        )

        raise ValueError(
            f"{name}中存在无法识别的值：{bad}"
        )

    return converted.astype(bool)


# ============================================================
# 6. 股票信息
# ============================================================

stock_info = pd.read_csv(
    STOCK_INFO_FILE,
    dtype=str
)


code_candidates = [
    "code",
    "stock_code",
    "ts_code",
    "symbol"
]

name_candidates = [
    "name",
    "stock_name"
]

industry_candidates = [
    "industry",
    "industry_name"
]


code_col = next(
    (
        c
        for c in code_candidates
        if c in stock_info.columns
    ),
    None
)

name_col = next(
    (
        c
        for c in name_candidates
        if c in stock_info.columns
    ),
    None
)

industry_col = next(
    (
        c
        for c in industry_candidates
        if c in stock_info.columns
    ),
    None
)


if (
    code_col is None
    or
    name_col is None
    or
    industry_col is None
):

    raise ValueError(
        "stock_info.csv缺少代码、名称或行业字段。"
    )


metadata = (
    stock_info[
        [
            code_col,
            name_col,
            industry_col
        ]
    ]
    .rename(
        columns={
            code_col: "stock_code",
            name_col: "stock_name",
            industry_col: "industry"
        }
    )
    .copy()
)


metadata[
    "stock_code"
] = metadata[
    "stock_code"
].apply(
    normalize_code
)


metadata = (
    metadata
    .drop_duplicates(
        subset="stock_code"
    )
    .reset_index(
        drop=True
    )
)


name_map = dict(
    zip(
        metadata[
            "stock_code"
        ],
        metadata[
            "stock_name"
        ]
    )
)


industry_map = dict(
    zip(
        metadata[
            "stock_code"
        ],
        metadata[
            "industry"
        ]
    )
)


# ============================================================
# 7. Regime Assignment
# ============================================================

assignment_df = pd.read_csv(
    REGIME_ASSIGNMENT_FILE
)


assignment_df[
    "network_date"
] = pd.to_datetime(
    assignment_df[
        "network_date"
    ]
)


assignment_df[
    "regime"
] = (
    pd.to_numeric(
        assignment_df[
            "regime"
        ],
        errors="raise"
    )
    .astype(int)
)


assignment_df = (
    assignment_df[
        [
            "network_date",
            "regime"
        ]
    ]
    .drop_duplicates()
)


# ============================================================
# 8. Edge History
# ============================================================

history_df = pd.read_csv(
    EDGE_HISTORY_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


if (
    "network_date"
    not in history_df.columns
):

    if (
        "window_end"
        in history_df.columns
    ):

        history_df[
            "network_date"
        ] = history_df[
            "window_end"
        ]

    else:

        raise ValueError(
            "找不到network_date/window_end。"
        )


history_df[
    "network_date"
] = pd.to_datetime(
    history_df[
        "network_date"
    ]
)


for col in [
    "stock_1",
    "stock_2"
]:

    history_df[col] = (
        history_df[col]
        .apply(
            normalize_code
        )
    )


pairs = history_df.apply(
    lambda row:
        canonical_pair(
            row[
                "stock_1"
            ],
            row[
                "stock_2"
            ]
        ),
    axis=1
)


history_df[
    "stock_1"
] = [
    x[0]
    for x in pairs
]

history_df[
    "stock_2"
] = [
    x[1]
    for x in pairs
]


history_df[
    "selected"
] = convert_bool(
    history_df[
        "selected"
    ],
    "selected"
)


history_df[
    "partial_correlation"
] = pd.to_numeric(
    history_df[
        "partial_correlation"
    ],
    errors="raise"
)


history_df[
    "abs_partial_correlation"
] = (
    history_df[
        "partial_correlation"
    ]
    .abs()
)


history_df = (
    history_df
    .merge(
        assignment_df,
        on="network_date",
        how="inner"
    )
)


nodes = sorted(
    set(
        history_df[
            "stock_1"
        ]
    )
    |
    set(
        history_df[
            "stock_2"
        ]
    )
)


print(
    "股票数量：",
    len(nodes)
)


# ============================================================
# 9. 构造每个窗口 × 每个节点的Centrality
# ============================================================

node_window_rows = []


for (
    network_date,
    regime
), group in history_df.groupby(
    [
        "network_date",
        "regime"
    ]
):

    metrics = {
        code: {
            "degree": 0,
            "strength": 0.0,
            "same_degree": 0,
            "cross_degree": 0,
            "same_strength": 0.0,
            "cross_strength": 0.0
        }
        for code in nodes
    }


    selected = group[
        group[
            "selected"
        ]
    ]


    for row in selected.itertuples():

        a = row.stock_1
        b = row.stock_2

        weight = abs(
            row.partial_correlation
        )


        same = (
            industry_map[a]
            ==
            industry_map[b]
        )


        for node in [
            a,
            b
        ]:

            metrics[
                node
            ][
                "degree"
            ] += 1

            metrics[
                node
            ][
                "strength"
            ] += weight


            if same:

                metrics[
                    node
                ][
                    "same_degree"
                ] += 1

                metrics[
                    node
                ][
                    "same_strength"
                ] += weight

            else:

                metrics[
                    node
                ][
                    "cross_degree"
                ] += 1

                metrics[
                    node
                ][
                    "cross_strength"
                ] += weight


    for code in nodes:

        node_window_rows.append(
            {
                "network_date":
                    network_date,

                "regime":
                    int(regime),

                "stock_code":
                    code,

                "stock_name":
                    name_map[
                        code
                    ],

                "industry":
                    industry_map[
                        code
                    ],

                **metrics[
                    code
                ]
            }
        )


node_window_df = pd.DataFrame(
    node_window_rows
)


# ============================================================
# 10. 每个窗口内部排名
# ============================================================

node_window_df[
    "degree_rank"
] = (
    node_window_df
    .groupby(
        "network_date"
    )[
        "degree"
    ]
    .rank(
        method="average",
        ascending=False
    )
)


node_window_df[
    "strength_rank"
] = (
    node_window_df
    .groupby(
        "network_date"
    )[
        "strength"
    ]
    .rank(
        method="average",
        ascending=False
    )
)


node_window_df[
    "degree_top5"
] = (
    node_window_df[
        "degree_rank"
    ]
    <=
    TOP_K
)


node_window_df[
    "strength_top5"
] = (
    node_window_df[
        "strength_rank"
    ]
    <=
    TOP_K
)


node_window_df.to_csv(
    WINDOW_NODE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. Regime × Node Summary
# ============================================================

regime_node_rows = []


for (
    regime,
    code
), group in node_window_df.groupby(
    [
        "regime",
        "stock_code"
    ]
):

    regime_node_rows.append(
        {
            "regime":
                int(regime),

            "stock_code":
                code,

            "stock_name":
                name_map[
                    code
                ],

            "industry":
                industry_map[
                    code
                ],

            "n_networks":
                len(
                    group
                ),

            "mean_degree":
                group[
                    "degree"
                ]
                .mean(),

            "sd_degree":
                group[
                    "degree"
                ]
                .std(
                    ddof=1
                ),

            "mean_strength":
                group[
                    "strength"
                ]
                .mean(),

            "sd_strength":
                group[
                    "strength"
                ]
                .std(
                    ddof=1
                ),

            "mean_same_degree":
                group[
                    "same_degree"
                ]
                .mean(),

            "mean_cross_degree":
                group[
                    "cross_degree"
                ]
                .mean(),

            "mean_same_strength":
                group[
                    "same_strength"
                ]
                .mean(),

            "mean_cross_strength":
                group[
                    "cross_strength"
                ]
                .mean(),

            "mean_degree_rank":
                group[
                    "degree_rank"
                ]
                .mean(),

            "mean_strength_rank":
                group[
                    "strength_rank"
                ]
                .mean(),

            "degree_top5_share":
                group[
                    "degree_top5"
                ]
                .mean(),

            "strength_top5_share":
                group[
                    "strength_top5"
                ]
                .mean()
        }
    )


regime_node_df = pd.DataFrame(
    regime_node_rows
)


regime_node_df.to_csv(
    REGIME_NODE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 相邻Regime Node Centrality Change
# ============================================================

regimes = sorted(
    regime_node_df[
        "regime"
    ]
    .unique()
)


node_change_rows = []


metric_columns = [
    "mean_degree",
    "mean_strength",
    "mean_same_degree",
    "mean_cross_degree",
    "mean_same_strength",
    "mean_cross_strength",
    "mean_degree_rank",
    "mean_strength_rank",
    "degree_top5_share",
    "strength_top5_share"
]


for idx in range(
    len(regimes) - 1
):

    r_before = regimes[
        idx
    ]

    r_after = regimes[
        idx + 1
    ]


    before = (
        regime_node_df[
            regime_node_df[
                "regime"
            ]
            ==
            r_before
        ]
        .copy()
    )


    after = (
        regime_node_df[
            regime_node_df[
                "regime"
            ]
            ==
            r_after
        ]
        .copy()
    )


    merged = (
        before[
            [
                "stock_code"
            ]
            +
            metric_columns
        ]
        .merge(
            after[
                [
                    "stock_code"
                ]
                +
                metric_columns
            ],

            on="stock_code",

            suffixes=(
                "_before",
                "_after"
            )
        )
    )


    for row in merged.itertuples():

        # Rank improvement:
        # before - after
        # >0 代表排名提高
        degree_rank_improvement = (
            row.mean_degree_rank_before
            -
            row.mean_degree_rank_after
        )


        strength_rank_improvement = (
            row.mean_strength_rank_before
            -
            row.mean_strength_rank_after
        )


        rank_shift_magnitude = np.sqrt(
            degree_rank_improvement ** 2
            +
            strength_rank_improvement ** 2
        )


        node_change_rows.append(
            {
                "regime_before":
                    r_before,

                "regime_after":
                    r_after,

                "stock_code":
                    row.stock_code,

                "stock_name":
                    name_map[
                        row.stock_code
                    ],

                "industry":
                    industry_map[
                        row.stock_code
                    ],

                "mean_degree_before":
                    row.mean_degree_before,

                "mean_degree_after":
                    row.mean_degree_after,

                "delta_mean_degree":
                    (
                        row.mean_degree_after
                        -
                        row.mean_degree_before
                    ),

                "mean_strength_before":
                    row.mean_strength_before,

                "mean_strength_after":
                    row.mean_strength_after,

                "delta_mean_strength":
                    (
                        row.mean_strength_after
                        -
                        row.mean_strength_before
                    ),

                "delta_mean_same_degree":
                    (
                        row.mean_same_degree_after
                        -
                        row.mean_same_degree_before
                    ),

                "delta_mean_cross_degree":
                    (
                        row.mean_cross_degree_after
                        -
                        row.mean_cross_degree_before
                    ),

                "delta_mean_same_strength":
                    (
                        row.mean_same_strength_after
                        -
                        row.mean_same_strength_before
                    ),

                "delta_mean_cross_strength":
                    (
                        row.mean_cross_strength_after
                        -
                        row.mean_cross_strength_before
                    ),

                "mean_degree_rank_before":
                    row.mean_degree_rank_before,

                "mean_degree_rank_after":
                    row.mean_degree_rank_after,

                "degree_rank_improvement":
                    degree_rank_improvement,

                "mean_strength_rank_before":
                    row.mean_strength_rank_before,

                "mean_strength_rank_after":
                    row.mean_strength_rank_after,

                "strength_rank_improvement":
                    strength_rank_improvement,

                "rank_shift_magnitude":
                    rank_shift_magnitude,

                "delta_degree_top5_share":
                    (
                        row.degree_top5_share_after
                        -
                        row.degree_top5_share_before
                    ),

                "delta_strength_top5_share":
                    (
                        row.strength_top5_share_after
                        -
                        row.strength_top5_share_before
                    )
            }
        )


node_change_df = pd.DataFrame(
    node_change_rows
)


node_change_df.to_csv(
    NODE_CHANGE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. 读取Stage 4 Edge Change Detail
# ============================================================

edge_change_df = pd.read_csv(
    EDGE_CHANGE_DETAIL_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


for col in [
    "stock_1",
    "stock_2"
]:

    edge_change_df[col] = (
        edge_change_df[col]
        .apply(
            normalize_code
        )
    )


pairs = edge_change_df.apply(
    lambda row:
        canonical_pair(
            row[
                "stock_1"
            ],
            row[
                "stock_2"
            ]
        ),
    axis=1
)


edge_change_df[
    "stock_1"
] = [
    x[0]
    for x in pairs
]

edge_change_df[
    "stock_2"
] = [
    x[1]
    for x in pairs
]


edge_change_df[
    "regime_before"
] = (
    pd.to_numeric(
        edge_change_df[
            "regime_before"
        ],
        errors="raise"
    )
    .astype(int)
)


edge_change_df[
    "regime_after"
] = (
    pd.to_numeric(
        edge_change_df[
            "regime_after"
        ],
        errors="raise"
    )
    .astype(int)
)


edge_change_df[
    "delta_persistence"
] = pd.to_numeric(
    edge_change_df[
        "delta_persistence"
    ],
    errors="raise"
)


edge_change_df[
    "same_industry"
] = edge_change_df.apply(
    lambda row:
        industry_map[
            row[
                "stock_1"
            ]
        ]
        ==
        industry_map[
            row[
                "stock_2"
            ]
        ],
    axis=1
)


edge_change_df[
    "core_transition_normalized"
] = (
    edge_change_df[
        "core_transition"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# 14. Node Edge-change Burden
# ============================================================

burden_rows = []


transitions = (
    edge_change_df[
        [
            "regime_before",
            "regime_after"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "regime_before",
            "regime_after"
        ]
    )
)


for transition in transitions.itertuples(
    index=False
):

    r_before = (
        transition.regime_before
    )

    r_after = (
        transition.regime_after
    )


    subset = edge_change_df[
        (
            edge_change_df[
                "regime_before"
            ]
            ==
            r_before
        )
        &
        (
            edge_change_df[
                "regime_after"
            ]
            ==
            r_after
        )
    ]


    node_stats = {
        code: {
            "lost_core_incident": 0,
            "gained_core_incident": 0,
            "lost_same_core_incident": 0,
            "lost_cross_core_incident": 0,
            "gained_same_core_incident": 0,
            "gained_cross_core_incident": 0,
            "persistence_drop_burden": 0.0,
            "persistence_gain_burden": 0.0,
            "abs_persistence_shift_burden": 0.0,
            "cross_persistence_drop_burden": 0.0,
            "cross_persistence_gain_burden": 0.0,
            "cross_abs_persistence_shift_burden": 0.0,
            "major_persistence_drop_count": 0,
            "major_persistence_gain_count": 0
        }
        for code in nodes
    }


    for row in subset.itertuples():

        a = row.stock_1
        b = row.stock_2

        delta_p = (
            row.delta_persistence
        )


        same = (
            industry_map[a]
            ==
            industry_map[b]
        )


        transition_label = (
            row.core_transition_normalized
        )


        lost_core = (
            transition_label
            ==
            "lost core"
        )


        gained_core = (
            transition_label
            ==
            "gained core"
        )


        drop = max(
            -delta_p,
            0.0
        )

        gain = max(
            delta_p,
            0.0
        )

        abs_change = abs(
            delta_p
        )


        for code in [
            a,
            b
        ]:

            stats = node_stats[
                code
            ]


            if lost_core:

                stats[
                    "lost_core_incident"
                ] += 1

                if same:

                    stats[
                        "lost_same_core_incident"
                    ] += 1

                else:

                    stats[
                        "lost_cross_core_incident"
                    ] += 1


            if gained_core:

                stats[
                    "gained_core_incident"
                ] += 1

                if same:

                    stats[
                        "gained_same_core_incident"
                    ] += 1

                else:

                    stats[
                        "gained_cross_core_incident"
                    ] += 1


            stats[
                "persistence_drop_burden"
            ] += drop

            stats[
                "persistence_gain_burden"
            ] += gain

            stats[
                "abs_persistence_shift_burden"
            ] += abs_change


            if not same:

                stats[
                    "cross_persistence_drop_burden"
                ] += drop

                stats[
                    "cross_persistence_gain_burden"
                ] += gain

                stats[
                    "cross_abs_persistence_shift_burden"
                ] += abs_change


            if (
                delta_p
                <=
                -MAJOR_PERSISTENCE_CHANGE
            ):

                stats[
                    "major_persistence_drop_count"
                ] += 1


            if (
                delta_p
                >=
                MAJOR_PERSISTENCE_CHANGE
            ):

                stats[
                    "major_persistence_gain_count"
                ] += 1


    transition_node_rows = []


    for code in nodes:

        stats = node_stats[
            code
        ]


        total_core_change_incident = (
            stats[
                "lost_core_incident"
            ]
            +
            stats[
                "gained_core_incident"
            ]
        )


        transition_node_rows.append(
            {
                "regime_before":
                    r_before,

                "regime_after":
                    r_after,

                "stock_code":
                    code,

                "stock_name":
                    name_map[
                        code
                    ],

                "industry":
                    industry_map[
                        code
                    ],

                **stats,

                "total_core_change_incident":
                    total_core_change_incident,

                "net_core_incident_change":
                    (
                        stats[
                            "gained_core_incident"
                        ]
                        -
                        stats[
                            "lost_core_incident"
                        ]
                    ),

                "net_persistence_shift":
                    (
                        stats[
                            "persistence_gain_burden"
                        ]
                        -
                        stats[
                            "persistence_drop_burden"
                        ]
                    )
            }
        )


    temp = pd.DataFrame(
        transition_node_rows
    )


    # --------------------------------------------------------
    # Share:
    # 分母为所有节点的burden总和。
    # 每条edge会贡献给两个endpoint，因此share总和=1。
    # --------------------------------------------------------

    total_abs_burden = (
        temp[
            "abs_persistence_shift_burden"
        ]
        .sum()
    )


    total_core_change_incident = (
        temp[
            "total_core_change_incident"
        ]
        .sum()
    )


    if total_abs_burden > 0:

        temp[
            "persistence_shift_share"
        ] = (
            temp[
                "abs_persistence_shift_burden"
            ]
            /
            total_abs_burden
        )

    else:

        temp[
            "persistence_shift_share"
        ] = 0.0


    if total_core_change_incident > 0:

        temp[
            "core_change_incident_share"
        ] = (
            temp[
                "total_core_change_incident"
            ]
            /
            total_core_change_incident
        )

    else:

        temp[
            "core_change_incident_share"
        ] = 0.0


    temp[
        "cross_shift_ratio"
    ] = np.where(
        temp[
            "abs_persistence_shift_burden"
        ]
        >
        0,

        temp[
            "cross_abs_persistence_shift_burden"
        ]
        /
        temp[
            "abs_persistence_shift_burden"
        ],

        np.nan
    )


    burden_rows.append(
        temp
    )


burden_df = pd.concat(
    burden_rows,
    ignore_index=True
)


burden_df.to_csv(
    NODE_BURDEN_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. 合并Centrality Change + Edge Burden
# ============================================================

contributor_df = (
    node_change_df
    .merge(
        burden_df,

        on=[
            "regime_before",
            "regime_after",
            "stock_code",
            "stock_name",
            "industry"
        ],

        how="left"
    )
)


# ============================================================
# 16. 描述性Transition Mode
# ============================================================

def classify_node_transition(
    row
):

    lost = (
        row[
            "lost_core_incident"
        ]
    )

    gained = (
        row[
            "gained_core_incident"
        ]
    )


    if (
        lost == 0
        and
        gained == 0
    ):

        return (
            "Limited core change"
        )


    if (
        lost > 0
        and
        gained == 0
    ):

        return (
            "Loss-dominated"
        )


    if (
        gained > 0
        and
        lost == 0
    ):

        return (
            "Gain-dominated"
        )


    if (
        lost > 0
        and
        gained > 0
    ):

        return (
            "Rewiring"
        )


    return (
        "Mixed"
    )


contributor_df[
    "node_transition_mode"
] = contributor_df.apply(
    classify_node_transition,
    axis=1
)


contributor_df.to_csv(
    CONTRIBUTOR_MASTER_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. Edge-structure Contributors
#
# 主要按Persistence Shift Share排序。
# ============================================================

top_edge_contributors = (
    contributor_df
    .sort_values(
        [
            "regime_before",
            "regime_after",
            "persistence_shift_share"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )
    .groupby(
        [
            "regime_before",
            "regime_after"
        ],
        group_keys=False
    )
    .head(
        TOP_N
    )
    .reset_index(
        drop=True
    )
)


top_edge_contributors.to_csv(
    TOP_EDGE_CONTRIBUTORS_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. Node Role Shift Ranking
#
# rank_shift_magnitude:
# sqrt(
#   degree_rank_improvement^2
#   +
#   strength_rank_improvement^2
# )
# ============================================================

top_role_shifts = (
    contributor_df
    .sort_values(
        [
            "regime_before",
            "regime_after",
            "rank_shift_magnitude"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )
    .groupby(
        [
            "regime_before",
            "regime_after"
        ],
        group_keys=False
    )
    .head(
        TOP_N
    )
    .reset_index(
        drop=True
    )
)


top_role_shifts.to_csv(
    TOP_ROLE_SHIFTS_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. R3 -> R4专项结果
# ============================================================

r34_df = contributor_df[
    (
        contributor_df[
            "regime_before"
        ]
        ==
        3
    )
    &
    (
        contributor_df[
            "regime_after"
        ]
        ==
        4
    )
].copy()


if len(
    r34_df
) > 0:

    r34_df = (
        r34_df
        .sort_values(
            [
                "persistence_shift_share",
                "rank_shift_magnitude"
            ],
            ascending=[
                False,
                False
            ]
        )
        .reset_index(
            drop=True
        )
    )


    r34_df.to_csv(
        R34_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 20. 输出：每个Transition的Top Contributors
# ============================================================

print(
    "\n======================================"
)

print(
    "Top Edge-structure Contributors"
)

print(
    "======================================"
)


display_columns = [
    "regime_before",
    "regime_after",
    "stock_code",
    "stock_name",
    "industry",
    "lost_core_incident",
    "gained_core_incident",
    "persistence_drop_burden",
    "persistence_gain_burden",
    "persistence_shift_share",
    "cross_shift_ratio",
    "delta_mean_degree",
    "delta_mean_cross_degree",
    "delta_mean_strength",
    "degree_rank_improvement",
    "strength_rank_improvement",
    "node_transition_mode"
]


print(
    top_edge_contributors[
        display_columns
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 21. R3 -> R4重点输出
# ============================================================

if len(
    r34_df
) > 0:

    print(
        "\n======================================"
    )

    print(
        "R3 -> R4 Top Transition Contributors"
    )

    print(
        "======================================"
    )


    print(
        r34_df[
            display_columns
            +
            [
                "rank_shift_magnitude"
            ]
        ]
        .head(
            15
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 22. 图1：R3 -> R4 Persistence Drop Burden
# ============================================================

if len(
    r34_df
) > 0:

    plot_df = (
        r34_df
        .sort_values(
            "persistence_drop_burden",
            ascending=False
        )
        .head(
            10
        )
        .sort_values(
            "persistence_drop_burden",
            ascending=True
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            10,
            7
        )
    )


    ax.barh(
        plot_df[
            "stock_name"
        ],
        plot_df[
            "persistence_drop_burden"
        ]
    )


    ax.set_xlabel(
        "Persistence Drop Burden"
    )


    ax.set_ylabel(
        "Stock"
    )


    ax.set_title(
        "R3→R4 节点Persistence下降负担"
    )


    fig.tight_layout()


    DROP_FIGURE = (
        FIGURE_DIR
        / "R3_R4_node_persistence_drop_burden.png"
    )


    fig.savefig(
        DROP_FIGURE,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()


# ============================================================
# 23. 图2：R3 -> R4 Mean Strength Change
# ============================================================

if len(
    r34_df
) > 0:

    plot_df = (
        r34_df
        .assign(
            abs_strength_change=
                r34_df[
                    "delta_mean_strength"
                ]
                .abs()
        )
        .sort_values(
            "abs_strength_change",
            ascending=False
        )
        .head(
            10
        )
        .sort_values(
            "delta_mean_strength"
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            10,
            7
        )
    )


    ax.barh(
        plot_df[
            "stock_name"
        ],
        plot_df[
            "delta_mean_strength"
        ]
    )


    ax.axvline(
        x=0,
        linewidth=1
    )


    ax.set_xlabel(
        r"$\Delta$ Mean Strength"
    )


    ax.set_ylabel(
        "Stock"
    )


    ax.set_title(
        "R3→R4 节点平均Strength变化"
    )


    fig.tight_layout()


    STRENGTH_FIGURE = (
        FIGURE_DIR
        / "R3_R4_node_strength_change.png"
    )


    fig.savefig(
        STRENGTH_FIGURE,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()


# ============================================================
# 24. 图3：R3 -> R4 Cross Degree Change
# ============================================================

if len(
    r34_df
) > 0:

    plot_df = (
        r34_df
        .assign(
            abs_cross_degree_change=
                r34_df[
                    "delta_mean_cross_degree"
                ]
                .abs()
        )
        .sort_values(
            "abs_cross_degree_change",
            ascending=False
        )
        .head(
            10
        )
        .sort_values(
            "delta_mean_cross_degree"
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            10,
            7
        )
    )


    ax.barh(
        plot_df[
            "stock_name"
        ],
        plot_df[
            "delta_mean_cross_degree"
        ]
    )


    ax.axvline(
        x=0,
        linewidth=1
    )


    ax.set_xlabel(
        r"$\Delta$ Mean Cross-industry Degree"
    )


    ax.set_ylabel(
        "Stock"
    )


    ax.set_title(
        "R3→R4 跨行业Degree变化"
    )


    fig.tight_layout()


    CROSS_DEGREE_FIGURE = (
        FIGURE_DIR
        / "R3_R4_cross_degree_change.png"
    )


    fig.savefig(
        CROSS_DEGREE_FIGURE,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()


# ============================================================
# 25. 完成
# ============================================================

print(
    "\n======================================"
)

print(
    "Stage 5完成"
)

print(
    "======================================"
)


print(
    "\n主要输出文件："
)


for path in [
    WINDOW_NODE_FILE,
    REGIME_NODE_FILE,
    NODE_CHANGE_FILE,
    NODE_BURDEN_FILE,
    CONTRIBUTOR_MASTER_FILE,
    TOP_EDGE_CONTRIBUTORS_FILE,
    TOP_ROLE_SHIFTS_FILE
]:

    print(
        path
    )


if len(
    r34_df
) > 0:

    print(
        R34_FILE
    )