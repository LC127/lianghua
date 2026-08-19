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
        "警告：未找到常见中文字体，"
        "中文标签可能显示异常。"
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
# 2. 输入文件
# ============================================================

REGIME_ASSIGNMENT_FILE = (
    PROCESSED_DIR
    / "network_regime_assignment.csv"
)

EDGE_HISTORY_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_history.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ============================================================
# 3. 输出文件
# ============================================================

# Regime-level主汇总
REGIME_STRUCTURE_FILE = (
    PROCESSED_DIR
    / "regime_network_structure_summary.csv"
)

# 每个Regime × 每条Edge
REGIME_EDGE_PERSISTENCE_FILE = (
    PROCESSED_DIR
    / "regime_edge_persistence.csv"
)

# Persistence >= .8
REGIME_CORE_EDGES_FILE = (
    PROCESSED_DIR
    / "regime_core_edges.csv"
)

# P=1
REGIME_ALWAYS_EDGES_FILE = (
    PROCESSED_DIR
    / "regime_always_edges.csv"
)

# 相邻Regime整体比较
REGIME_TRANSITION_SUMMARY_FILE = (
    PROCESSED_DIR
    / "adjacent_regime_edge_comparison.csv"
)

# 每个Transition × 每条Edge
REGIME_TRANSITION_DETAIL_FILE = (
    PROCESSED_DIR
    / "adjacent_regime_edge_change_detail.csv"
)

# Persistence变化最大的边
TOP_PERSISTENCE_CHANGE_FILE = (
    PROCESSED_DIR
    / "top_regime_persistence_changes.csv"
)


# ============================================================
# 4. 参数
# ============================================================

CORE_THRESHOLD = 0.80

EPS = 1e-12


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
        x
        for x in s
        if x.isdigit()
    )

    if digits:

        return digits.zfill(6)

    return s


def canonical_pair(
    stock_1,
    stock_2
):

    a = normalize_code(
        stock_1
    )

    b = normalize_code(
        stock_2
    )

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

        raise ValueError(
            f"{name}存在无法识别的布尔值："
            f"{series[converted.isna()].unique()}"
        )

    return converted.astype(bool)


# ============================================================
# 6. 股票Metadata
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
            code_col: "code",
            name_col: "name",
            industry_col: "industry"
        }
    )
    .copy()
)


metadata["code"] = (
    metadata[
        "code"
    ]
    .apply(
        normalize_code
    )
)


metadata = (
    metadata
    .drop_duplicates(
        subset="code"
    )
)


name_map = dict(
    zip(
        metadata["code"],
        metadata["name"]
    )
)


industry_map = dict(
    zip(
        metadata["code"],
        metadata["industry"]
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
    assignment_df
    .sort_values(
        "network_date"
    )
    .reset_index(
        drop=True
    )
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


required = [
    "stock_1",
    "stock_2",
    "selected",
    "partial_correlation"
]


for col in required:

    if col not in history_df.columns:

        raise ValueError(
            f"Edge History缺少字段：{col}"
        )


# ------------------------------------------------------------
# Date
# ------------------------------------------------------------

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
            "Edge History缺少network_date/window_end。"
        )


history_df[
    "network_date"
] = pd.to_datetime(
    history_df[
        "network_date"
    ]
)


# ============================================================
# 9. Canonicalize股票代码
# ============================================================

history_df["stock_1"] = (
    history_df["stock_1"]
    .apply(normalize_code)
)

history_df["stock_2"] = (
    history_df["stock_2"]
    .apply(normalize_code)
)


pairs = history_df.apply(
    lambda row:
        canonical_pair(
            row["stock_1"],
            row["stock_2"]
        ),
    axis=1
)


history_df["stock_1"] = [
    x[0]
    for x in pairs
]

history_df["stock_2"] = [
    x[1]
    for x in pairs
]


# ============================================================
# 10. 重新映射Name / Industry
# ============================================================

history_df[
    "name_1"
] = history_df[
    "stock_1"
].map(
    name_map
)


history_df[
    "name_2"
] = history_df[
    "stock_2"
].map(
    name_map
)


history_df[
    "industry_1"
] = history_df[
    "stock_1"
].map(
    industry_map
)


history_df[
    "industry_2"
] = history_df[
    "stock_2"
].map(
    industry_map
)


if (
    history_df[
        [
            "name_1",
            "name_2",
            "industry_1",
            "industry_2"
        ]
    ]
    .isna()
    .any()
    .any()
):

    raise ValueError(
        "股票Metadata映射失败。"
    )


history_df[
    "same_industry"
] = (
    history_df[
        "industry_1"
    ]
    ==
    history_df[
        "industry_2"
    ]
)


history_df[
    "industry_relation"
] = np.where(
    history_df[
        "same_industry"
    ],
    "Same industry",
    "Cross industry"
)


# ============================================================
# 11. 类型
# ============================================================

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


# ============================================================
# 12. 合并Regime
# ============================================================

history_df = (
    history_df
    .merge(
        assignment_df[
            [
                "network_date",
                "regime"
            ]
        ],

        on="network_date",

        how="inner"
    )
)


if len(history_df) == 0:

    raise ValueError(
        "Edge History与Regime Assignment没有匹配日期。"
    )


# ============================================================
# 13. 唯一性检查
# ============================================================

duplicate = history_df.duplicated(
    subset=[
        "network_date",
        "stock_1",
        "stock_2"
    ]
)


if duplicate.any():

    raise ValueError(
        "同一Network Date存在重复股票对。"
    )


# ============================================================
# 14. Regime × Edge Persistence
# ============================================================

edge_rows = []


for (
    regime,
    stock_1,
    stock_2
), group in history_df.groupby(
    [
        "regime",
        "stock_1",
        "stock_2"
    ]
):

    group = (
        group
        .sort_values(
            "network_date"
        )
    )


    n_windows = len(
        group
    )


    n_selected = int(
        group[
            "selected"
        ]
        .sum()
    )


    persistence = (
        n_selected
        /
        n_windows
    )


    selected_group = (
        group[
            group[
                "selected"
            ]
        ]
    )


    if n_selected > 0:

        mean_partial_selected = (
            selected_group[
                "partial_correlation"
            ]
            .mean()
        )


        mean_abs_partial_selected = (
            selected_group[
                "abs_partial_correlation"
            ]
            .mean()
        )


        max_abs_partial_selected = (
            selected_group[
                "abs_partial_correlation"
            ]
            .max()
        )

    else:

        mean_partial_selected = np.nan
        mean_abs_partial_selected = np.nan
        max_abs_partial_selected = np.nan


    edge_rows.append(
        {
            "regime":
                int(regime),

            "stock_1":
                stock_1,

            "name_1":
                group[
                    "name_1"
                ]
                .iloc[0],

            "industry_1":
                group[
                    "industry_1"
                ]
                .iloc[0],

            "stock_2":
                stock_2,

            "name_2":
                group[
                    "name_2"
                ]
                .iloc[0],

            "industry_2":
                group[
                    "industry_2"
                ]
                .iloc[0],

            "same_industry":
                bool(
                    group[
                        "same_industry"
                    ]
                    .iloc[0]
                ),

            "industry_relation":
                group[
                    "industry_relation"
                ]
                .iloc[0],

            "n_regime_windows":
                n_windows,

            "windows_selected":
                n_selected,

            "persistence":
                persistence,

            "is_core_edge":
                (
                    persistence
                    >=
                    CORE_THRESHOLD
                ),

            "is_always_edge":
                np.isclose(
                    persistence,
                    1.0
                ),

            "mean_partial_when_selected":
                mean_partial_selected,

            "mean_abs_partial_when_selected":
                mean_abs_partial_selected,

            "max_abs_partial_when_selected":
                max_abs_partial_selected
        }
    )


edge_persistence_df = pd.DataFrame(
    edge_rows
)


edge_persistence_df.to_csv(
    REGIME_EDGE_PERSISTENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. Core / Always 子集
# ============================================================

core_edges_df = (
    edge_persistence_df[
        edge_persistence_df[
            "is_core_edge"
        ]
    ]
    .copy()
)


always_edges_df = (
    edge_persistence_df[
        edge_persistence_df[
            "is_always_edge"
        ]
    ]
    .copy()
)


core_edges_df.to_csv(
    REGIME_CORE_EDGES_FILE,
    index=False,
    encoding="utf-8-sig"
)


always_edges_df.to_csv(
    REGIME_ALWAYS_EDGES_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. Regime-level Network Structure
# ============================================================

structure_rows = []


for regime, regime_state in assignment_df.groupby(
    "regime"
):

    regime_state = (
        regime_state
        .sort_values(
            "network_date"
        )
    )


    regime_edge = (
        edge_persistence_df[
            edge_persistence_df[
                "regime"
            ]
            ==
            regime
        ]
    )


    core = (
        regime_edge[
            regime_edge[
                "is_core_edge"
            ]
        ]
    )


    always = (
        regime_edge[
            regime_edge[
                "is_always_edge"
            ]
        ]
    )


    same_core = int(
        core[
            "same_industry"
        ]
        .sum()
    )


    cross_core = (
        len(core)
        -
        same_core
    )


    same_always = int(
        always[
            "same_industry"
        ]
        .sum()
    )


    cross_always = (
        len(always)
        -
        same_always
    )


    structure_rows.append(
        {
            "regime":
                int(regime),

            "start_date":
                regime_state[
                    "network_date"
                ]
                .min(),

            "end_date":
                regime_state[
                    "network_date"
                ]
                .max(),

            "n_networks":
                len(
                    regime_state
                ),

            # -----------------------------------------------
            # Network Level
            # -----------------------------------------------

            "mean_edge_count":
                regime_state[
                    "edge_count"
                ]
                .mean(),

            "mean_density":
                regime_state[
                    "density"
                ]
                .mean(),

            "mean_same_edges":
                regime_state[
                    "same_edges"
                ]
                .mean(),

            "mean_cross_edges":
                regime_state[
                    "cross_edges"
                ]
                .mean(),

            "mean_same_industry_ratio":
                regime_state[
                    "same_industry_ratio"
                ]
                .mean(),

            "mean_abs_partial":
                regime_state[
                    "mean_abs_partial"
                ]
                .mean(),

            # -----------------------------------------------
            # Regime Edge Persistence
            # -----------------------------------------------

            "mean_edge_persistence":
                regime_edge[
                    "persistence"
                ]
                .mean(),

            "n_core_edges":
                len(
                    core
                ),

            "n_same_core_edges":
                same_core,

            "n_cross_core_edges":
                cross_core,

            "same_share_of_core":
                (
                    same_core
                    /
                    len(core)
                    if len(core) > 0
                    else np.nan
                ),

            "n_always_edges":
                len(
                    always
                ),

            "n_same_always_edges":
                same_always,

            "n_cross_always_edges":
                cross_always
        }
    )


structure_df = pd.DataFrame(
    structure_rows
)


structure_df.to_csv(
    REGIME_STRUCTURE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. 相邻Regime比较
# ============================================================

regimes = sorted(
    edge_persistence_df[
        "regime"
    ]
    .unique()
)


transition_summary_rows = []

transition_detail_rows = []


for idx in range(
    len(regimes) - 1
):

    regime_a = regimes[
        idx
    ]

    regime_b = regimes[
        idx + 1
    ]


    a = (
        edge_persistence_df[
            edge_persistence_df[
                "regime"
            ]
            ==
            regime_a
        ]
        .copy()
    )


    b = (
        edge_persistence_df[
            edge_persistence_df[
                "regime"
            ]
            ==
            regime_b
        ]
        .copy()
    )


    merged = (
        a[
            [
                "stock_1",
                "stock_2",
                "name_1",
                "name_2",
                "industry_1",
                "industry_2",
                "same_industry",
                "industry_relation",
                "persistence",
                "is_core_edge",
                "is_always_edge",
                "mean_abs_partial_when_selected"
            ]
        ]
        .rename(
            columns={
                "persistence":
                    "persistence_before",

                "is_core_edge":
                    "core_before",

                "is_always_edge":
                    "always_before",

                "mean_abs_partial_when_selected":
                    "mean_abs_partial_before"
            }
        )
        .merge(
            b[
                [
                    "stock_1",
                    "stock_2",
                    "persistence",
                    "is_core_edge",
                    "is_always_edge",
                    "mean_abs_partial_when_selected"
                ]
            ]
            .rename(
                columns={
                    "persistence":
                        "persistence_after",

                    "is_core_edge":
                        "core_after",

                    "is_always_edge":
                        "always_after",

                    "mean_abs_partial_when_selected":
                        "mean_abs_partial_after"
                }
            ),

            on=[
                "stock_1",
                "stock_2"
            ],

            how="inner"
        )
    )


    merged[
        "delta_persistence"
    ] = (
        merged[
            "persistence_after"
        ]
        -
        merged[
            "persistence_before"
        ]
    )


    merged[
        "abs_delta_persistence"
    ] = (
        merged[
            "delta_persistence"
        ]
        .abs()
    )


    # --------------------------------------------------------
    # Core transition classification
    # --------------------------------------------------------

    def classify_core_transition(
        row
    ):

        before = (
            row[
                "core_before"
            ]
        )

        after = (
            row[
                "core_after"
            ]
        )


        if (
            before
            and
            after
        ):

            return (
                "Persistent core"
            )


        if (
            before
            and
            not after
        ):

            return (
                "Lost core"
            )


        if (
            not before
            and
            after
        ):

            return (
                "Gained core"
            )


        return (
            "Non-core both"
        )


    merged[
        "core_transition"
    ] = (
        merged
        .apply(
            classify_core_transition,
            axis=1
        )
    )


    merged[
        "regime_before"
    ] = (
        regime_a
    )


    merged[
        "regime_after"
    ] = (
        regime_b
    )


    transition_detail_rows.append(
        merged
    )


    # --------------------------------------------------------
    # Core Sets
    # --------------------------------------------------------

    core_set_a = set(
        zip(
            a.loc[
                a[
                    "is_core_edge"
                ],
                "stock_1"
            ],

            a.loc[
                a[
                    "is_core_edge"
                ],
                "stock_2"
            ]
        )
    )


    core_set_b = set(
        zip(
            b.loc[
                b[
                    "is_core_edge"
                ],
                "stock_1"
            ],

            b.loc[
                b[
                    "is_core_edge"
                ],
                "stock_2"
            ]
        )
    )


    common_core = (
        core_set_a
        &
        core_set_b
    )


    lost_core = (
        core_set_a
        -
        core_set_b
    )


    gained_core = (
        core_set_b
        -
        core_set_a
    )


    union_core = (
        core_set_a
        |
        core_set_b
    )


    core_jaccard = (
        len(
            common_core
        )
        /
        len(
            union_core
        )
        if len(
            union_core
        ) > 0
        else 1.0
    )


    # --------------------------------------------------------
    # Same/Cross composition of Lost/Gained
    # --------------------------------------------------------

    lost_df = merged[
        merged[
            "core_transition"
        ]
        ==
        "Lost core"
    ]


    gained_df = merged[
        merged[
            "core_transition"
        ]
        ==
        "Gained core"
    ]


    persistent_df = merged[
        merged[
            "core_transition"
        ]
        ==
        "Persistent core"
    ]


    lost_same = int(
        lost_df[
            "same_industry"
        ]
        .sum()
    )


    gained_same = int(
        gained_df[
            "same_industry"
        ]
        .sum()
    )


    lost_cross = (
        len(
            lost_df
        )
        -
        lost_same
    )


    gained_cross = (
        len(
            gained_df
        )
        -
        gained_same
    )


    transition_summary_rows.append(
        {
            "regime_before":
                regime_a,

            "regime_after":
                regime_b,

            "n_core_before":
                len(
                    core_set_a
                ),

            "n_core_after":
                len(
                    core_set_b
                ),

            "common_core_edges":
                len(
                    common_core
                ),

            "lost_core_edges":
                len(
                    lost_core
                ),

            "gained_core_edges":
                len(
                    gained_core
                ),

            "core_jaccard":
                core_jaccard,

            "lost_same_core":
                lost_same,

            "lost_cross_core":
                lost_cross,

            "gained_same_core":
                gained_same,

            "gained_cross_core":
                gained_cross,

            "mean_delta_persistence":
                merged[
                    "delta_persistence"
                ]
                .mean(),

            "mean_abs_delta_persistence":
                merged[
                    "abs_delta_persistence"
                ]
                .mean(),

            "n_large_persistence_drop":
                int(
                    (
                        merged[
                            "delta_persistence"
                        ]
                        <=
                        -0.5
                    )
                    .sum()
                ),

            "n_large_persistence_gain":
                int(
                    (
                        merged[
                            "delta_persistence"
                        ]
                        >=
                        0.5
                    )
                    .sum()
                )
        }
    )


transition_summary_df = pd.DataFrame(
    transition_summary_rows
)


transition_detail_df = pd.concat(
    transition_detail_rows,
    ignore_index=True
)


transition_summary_df.to_csv(
    REGIME_TRANSITION_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


transition_detail_df.to_csv(
    REGIME_TRANSITION_DETAIL_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. 提取Persistence变化最大的边
# ============================================================

top_change_df = (
    transition_detail_df
    .sort_values(
        [
            "regime_before",
            "regime_after",
            "abs_delta_persistence"
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
    .head(15)
    .reset_index(
        drop=True
    )
)


top_change_df.to_csv(
    TOP_PERSISTENCE_CHANGE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. 屏幕输出
# ============================================================

print(
    "\n======================================"
)

print(
    "Regime Network Structure Summary"
)

print(
    "======================================"
)


print(
    structure_df.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Adjacent Regime Comparison"
)

print(
    "======================================"
)


print(
    transition_summary_df.to_string(
        index=False
    )
)


# ============================================================
# 20. R3 -> R4重点输出
# ============================================================

if (
    3 in regimes
    and
    4 in regimes
):

    r34 = (
        transition_detail_df[
            (
                transition_detail_df[
                    "regime_before"
                ]
                ==
                3
            )
            &
            (
                transition_detail_df[
                    "regime_after"
                ]
                ==
                4
            )
        ]
        .copy()
    )


    r34 = (
        r34
        .sort_values(
            "delta_persistence",
            ascending=True
        )
    )


    print(
        "\n======================================"
    )

    print(
        "R3 -> R4 Persistence下降最大的边"
    )

    print(
        "======================================"
    )


    print(
        r34[
            [
                "stock_1",
                "name_1",
                "stock_2",
                "name_2",
                "industry_relation",
                "persistence_before",
                "persistence_after",
                "delta_persistence",
                "core_transition"
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# 21. 图1：Regime Mean Edge Composition
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        6
    )
)


x = np.arange(
    len(
        structure_df
    )
)


width = 0.35


ax.bar(
    x - width / 2,
    structure_df[
        "mean_same_edges"
    ],
    width,
    label="同行业边"
)


ax.bar(
    x + width / 2,
    structure_df[
        "mean_cross_edges"
    ],
    width,
    label="跨行业边"
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    [
        f"R{r}"
        for r
        in structure_df[
            "regime"
        ]
    ]
)


ax.set_ylabel(
    "Mean Number of Edges"
)


ax.set_xlabel(
    "Regime"
)


ax.set_title(
    "不同Regime的同行业与跨行业边结构"
)


ax.legend()


fig.tight_layout()


EDGE_COMPOSITION_FIGURE = (
    FIGURE_DIR
    / "regime_edge_composition.png"
)


fig.savefig(
    EDGE_COMPOSITION_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 22. 图2：Regime Core Edge Count
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        6
    )
)


ax.plot(
    structure_df[
        "regime"
    ],
    structure_df[
        "n_same_core_edges"
    ],
    marker="o",
    label="同行业Core"
)


ax.plot(
    structure_df[
        "regime"
    ],
    structure_df[
        "n_cross_core_edges"
    ],
    marker="o",
    label="跨行业Core"
)


ax.set_xlabel(
    "Regime"
)


ax.set_ylabel(
    "Number of Regime Core Edges"
)


ax.set_xticks(
    structure_df[
        "regime"
    ]
)


ax.set_title(
    "Regime Persistent Core的行业组成"
)


ax.legend()

ax.grid(
    alpha=0.3
)


fig.tight_layout()


CORE_FIGURE = (
    FIGURE_DIR
    / "regime_core_edge_counts.png"
)


fig.savefig(
    CORE_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 23. 图3：Core Jaccard
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


labels = [
    f"R{int(row.regime_before)}→R{int(row.regime_after)}"
    for row
    in transition_summary_df.itertuples()
]


ax.bar(
    labels,
    transition_summary_df[
        "core_jaccard"
    ]
)


ax.set_ylim(
    0,
    1.05
)


ax.set_ylabel(
    "Core-edge Jaccard Similarity"
)


ax.set_xlabel(
    "Adjacent Regime Transition"
)


ax.set_title(
    "相邻Regime核心网络相似度"
)


fig.tight_layout()


JACCARD_FIGURE = (
    FIGURE_DIR
    / "regime_core_jaccard.png"
)


fig.savefig(
    JACCARD_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 24. 图4：R3 -> R4 Persistence Changes
# ============================================================

if (
    3 in regimes
    and
    4 in regimes
):

    r34_plot = (
        transition_detail_df[
            (
                transition_detail_df[
                    "regime_before"
                ]
                ==
                3
            )
            &
            (
                transition_detail_df[
                    "regime_after"
                ]
                ==
                4
            )
        ]
        .copy()
    )


    r34_plot[
        "edge_label"
    ] = (
        r34_plot[
            "name_1"
        ]
        +
        "–"
        +
        r34_plot[
            "name_2"
        ]
    )


    r34_plot = (
        r34_plot
        .sort_values(
            "delta_persistence"
        )
        .head(20)
    )


    fig, ax = plt.subplots(
        figsize=(
            11,
            8
        )
    )


    y = np.arange(
        len(
            r34_plot
        )
    )


    ax.barh(
        y,
        r34_plot[
            "delta_persistence"
        ]
    )


    ax.set_yticks(
        y
    )


    ax.set_yticklabels(
        r34_plot[
            "edge_label"
        ]
    )


    ax.axvline(
        x=0,
        linewidth=1
    )


    ax.set_xlabel(
        r"$\Delta P_{ij}=P_{ij}^{(R4)}-P_{ij}^{(R3)}$"
    )


    ax.set_ylabel(
        "Edge"
    )


    ax.set_title(
        "R3→R4 Persistence下降最大的股票关系"
    )


    fig.tight_layout()


    R34_FIGURE = (
        FIGURE_DIR
        / "R3_R4_edge_persistence_changes.png"
    )


    fig.savefig(
        R34_FIGURE,
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
    "Stage 4完成"
)

print(
    "======================================"
)


print(
    "\n主要输出文件："
)


for path in [
    REGIME_STRUCTURE_FILE,
    REGIME_EDGE_PERSISTENCE_FILE,
    REGIME_CORE_EDGES_FILE,
    REGIME_ALWAYS_EDGES_FILE,
    REGIME_TRANSITION_SUMMARY_FILE,
    REGIME_TRANSITION_DETAIL_FILE,
    TOP_PERSISTENCE_CHANGE_FILE
]:

    print(
        path
    )