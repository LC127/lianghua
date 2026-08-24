from pathlib import Path

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

    installed_fonts = {
        f.name
        for f in font_manager.fontManager.ttflist
    }

    for font_name in candidates:

        if font_name in installed_fonts:

            plt.rcParams["font.sans-serif"] = [
                font_name
            ]

            plt.rcParams["axes.unicode_minus"] = False

            print(
                f"使用中文字体：{font_name}"
            )

            return

    print(
        "未找到常见中文字体，中文可能显示异常。"
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


# ------------------------------------------------------------
# 输入
# ------------------------------------------------------------

ROLLING_EDGE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_history.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ------------------------------------------------------------
# 输出
# ------------------------------------------------------------

WINDOW_DECOMPOSITION_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_industry_edge_decomposition.csv"
)

INDUSTRY_PERSISTENCE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_industry_persistence_summary.csv"
)

EDGE_PERSISTENCE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_persistence_by_industry.csv"
)

INDUSTRY_TURNOVER_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_industry_edge_turnover.csv"
)


# ============================================================
# 2. 工具函数
# ============================================================

def normalize_code(x):

    return str(
        x
    ).strip().zfill(6)


def canonicalize_pair(
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


def convert_boolean_column(
    series,
    name
):

    if series.dtype == bool:

        return series


    converted = (
        series
        .astype(str)
        .str
        .strip()
        .str
        .lower()
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

        bad_values = (
            series[
                converted.isna()
            ]
            .unique()
        )

        raise ValueError(
            f"{name}存在无法识别的值："
            f"{bad_values}"
        )


    return converted.astype(
        bool
    )


# ============================================================
# 3. 读取原始Rolling Edge History
#
# 注意：
# 这里先不要排序股票对。
# 先建立正确的code -> name / industry映射。
# ============================================================

rolling_raw = pd.read_csv(
    ROLLING_EDGE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


rolling_raw[
    "stock_1"
] = (
    rolling_raw[
        "stock_1"
    ]
    .apply(
        normalize_code
    )
)


rolling_raw[
    "stock_2"
] = (
    rolling_raw[
        "stock_2"
    ]
    .apply(
        normalize_code
    )
)


# ============================================================
# 4. 建立正确的股票元数据
#
# 优先使用stock_info.csv。
# ============================================================

if STOCK_INFO_FILE.exists():

    print(
        "使用stock_info.csv作为名称/行业映射。"
    )


    stock_info = pd.read_csv(
        STOCK_INFO_FILE,
        dtype={
            "code": str
        }
    )


    stock_info[
        "code"
    ] = (
        stock_info[
            "code"
        ]
        .apply(
            normalize_code
        )
    )


    required_columns = {
        "code",
        "name",
        "industry"
    }


    missing = (
        required_columns
        -
        set(
            stock_info.columns
        )
    )


    if missing:

        raise ValueError(
            f"stock_info.csv缺少字段：{missing}"
        )


    metadata = (
        stock_info[
            [
                "code",
                "name",
                "industry"
            ]
        ]
        .drop_duplicates(
            subset=[
                "code"
            ]
        )
        .copy()
    )


else:

    print(
        "未找到stock_info.csv，"
        "从原始rolling edge history建立映射。"
    )


    metadata_1 = (
        rolling_raw[
            [
                "stock_1",
                "name_1",
                "industry_1"
            ]
        ]
        .rename(
            columns={
                "stock_1":
                    "code",

                "name_1":
                    "name",

                "industry_1":
                    "industry"
            }
        )
    )


    metadata_2 = (
        rolling_raw[
            [
                "stock_2",
                "name_2",
                "industry_2"
            ]
        ]
        .rename(
            columns={
                "stock_2":
                    "code",

                "name_2":
                    "name",

                "industry_2":
                    "industry"
            }
        )
    )


    metadata = pd.concat(
        [
            metadata_1,
            metadata_2
        ],
        ignore_index=True
    )


    # 检查一个股票代码是否对应多个名称
    name_conflict = (
        metadata
        .groupby(
            "code"
        )[
            "name"
        ]
        .nunique(
            dropna=True
        )
    )


    if (
        name_conflict > 1
    ).any():

        bad_codes = (
            name_conflict[
                name_conflict > 1
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "发现代码对应多个股票名称："
            f"{bad_codes}"
        )


    # 检查行业
    industry_conflict = (
        metadata
        .groupby(
            "code"
        )[
            "industry"
        ]
        .nunique(
            dropna=True
        )
    )


    if (
        industry_conflict > 1
    ).any():

        bad_codes = (
            industry_conflict[
                industry_conflict > 1
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "发现代码对应多个行业："
            f"{bad_codes}"
        )


    metadata = (
        metadata
        .drop_duplicates(
            subset=[
                "code"
            ]
        )
        .copy()
    )


# ============================================================
# 5. 映射字典
# ============================================================

metadata[
    "code"
] = (
    metadata[
        "code"
    ]
    .apply(
        normalize_code
    )
)


name_map = dict(
    zip(
        metadata[
            "code"
        ],
        metadata[
            "name"
        ]
    )
)


industry_map = dict(
    zip(
        metadata[
            "code"
        ],
        metadata[
            "industry"
        ]
    )
)


print(
    "\n股票代码、名称和行业："
)


print(
    metadata
    .sort_values(
        "code"
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 6. 股票对标准化
# ============================================================

rolling_df = rolling_raw.copy()


pairs = rolling_df.apply(

    lambda row:
        canonicalize_pair(
            row[
                "stock_1"
            ],
            row[
                "stock_2"
            ]
        ),

    axis=1
)


rolling_df[
    "stock_1"
] = [
    pair[0]
    for pair in pairs
]


rolling_df[
    "stock_2"
] = [
    pair[1]
    for pair in pairs
]


# ============================================================
# 7. 排序完成后重新映射名称/行业
#
# 完全不依赖旧name_1/name_2。
# ============================================================

rolling_df[
    "name_1"
] = (
    rolling_df[
        "stock_1"
    ]
    .map(
        name_map
    )
)


rolling_df[
    "name_2"
] = (
    rolling_df[
        "stock_2"
    ]
    .map(
        name_map
    )
)


rolling_df[
    "industry_1"
] = (
    rolling_df[
        "stock_1"
    ]
    .map(
        industry_map
    )
)


rolling_df[
    "industry_2"
] = (
    rolling_df[
        "stock_2"
    ]
    .map(
        industry_map
    )
)


# ============================================================
# 8. 检查行业是否缺失
# ============================================================

if (
    rolling_df[
        "industry_1"
    ]
    .isna()
    .any()
    or
    rolling_df[
        "industry_2"
    ]
    .isna()
    .any()
):

    raise ValueError(
        "存在股票缺少行业信息。"
    )


# ============================================================
# 9. same_industry
# ============================================================

rolling_df[
    "same_industry"
] = (
    rolling_df[
        "industry_1"
    ]
    ==
    rolling_df[
        "industry_2"
    ]
)


rolling_df[
    "industry_relation"
] = np.where(
    rolling_df[
        "same_industry"
    ],
    "Same industry",
    "Cross industry"
)


# ============================================================
# 10. selected
# ============================================================

rolling_df[
    "selected"
] = convert_boolean_column(

    rolling_df[
        "selected"
    ],

    "selected"
)


# ============================================================
# 11. 日期
# ============================================================

for col in [
    "window_start",
    "window_end",
    "network_date"
]:

    rolling_df[
        col
    ] = pd.to_datetime(
        rolling_df[
            col
        ]
    )


# ============================================================
# 12. 每个窗口只保留Selected Edges
# ============================================================

selected_df = rolling_df[
    rolling_df[
        "selected"
    ]
].copy()


# ============================================================
# 13. 窗口级动态分解
# ============================================================

window_rows = []


window_ids = sorted(
    rolling_df[
        "window_id"
    ]
    .unique()
)


for window_id in window_ids:

    current_all = rolling_df[
        rolling_df[
            "window_id"
        ]
        ==
        window_id
    ]


    current = selected_df[
        selected_df[
            "window_id"
        ]
        ==
        window_id
    ]


    # 当前网络日期
    network_date = (
        current_all[
            "network_date"
        ]
        .iloc[0]
    )


    # --------------------------------------------
    # Same-industry
    # --------------------------------------------

    same = current[
        current[
            "same_industry"
        ]
    ]


    # --------------------------------------------
    # Cross-industry
    # --------------------------------------------

    cross = current[
        ~current[
            "same_industry"
        ]
    ]


    n_total = len(
        current
    )


    n_same = len(
        same
    )


    n_cross = len(
        cross
    )


    same_ratio = (

        n_same
        /
        n_total

        if n_total > 0

        else np.nan
    )


    cross_ratio = (

        n_cross
        /
        n_total

        if n_total > 0

        else np.nan
    )


    mean_abs_partial_same = (

        same[
            "abs_partial_correlation"
        ]
        .mean()

        if n_same > 0

        else np.nan
    )


    mean_abs_partial_cross = (

        cross[
            "abs_partial_correlation"
        ]
        .mean()

        if n_cross > 0

        else np.nan
    )


    median_abs_partial_same = (

        same[
            "abs_partial_correlation"
        ]
        .median()

        if n_same > 0

        else np.nan
    )


    median_abs_partial_cross = (

        cross[
            "abs_partial_correlation"
        ]
        .median()

        if n_cross > 0

        else np.nan
    )


    window_rows.append(
        {
            "window_id":
                window_id,

            "network_date":
                network_date,

            "total_edges":
                n_total,

            "same_industry_edges":
                n_same,

            "cross_industry_edges":
                n_cross,

            "same_industry_ratio":
                same_ratio,

            "cross_industry_ratio":
                cross_ratio,

            "mean_abs_partial_same":
                mean_abs_partial_same,

            "mean_abs_partial_cross":
                mean_abs_partial_cross,

            "median_abs_partial_same":
                median_abs_partial_same,

            "median_abs_partial_cross":
                median_abs_partial_cross
        }
    )


window_decomposition_df = pd.DataFrame(
    window_rows
)


window_decomposition_df.to_csv(
    WINDOW_DECOMPOSITION_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. 每条边的Persistence
# ============================================================

total_windows = (
    rolling_df[
        "window_id"
    ]
    .nunique()
)


edge_persistence_df = (
    rolling_df
    .groupby(
        [
            "stock_1",
            "stock_2",
            "name_1",
            "name_2",
            "industry_1",
            "industry_2",
            "same_industry",
            "industry_relation"
        ],
        as_index=False
    )
    .agg(

        windows_selected=(
            "selected",
            "sum"
        ),

        mean_partial_all_windows=(
            "partial_correlation",
            "mean"
        ),

        mean_abs_partial_all_windows=(
            "abs_partial_correlation",
            "mean"
        )
    )
)


edge_persistence_df[
    "total_windows"
] = (
    total_windows
)


edge_persistence_df[
    "persistence"
] = (
    edge_persistence_df[
        "windows_selected"
    ]
    /
    total_windows
)


# ============================================================
# 15. 被选中时平均强度
# ============================================================

selected_strength_df = (
    selected_df
    .groupby(
        [
            "stock_1",
            "stock_2"
        ],
        as_index=False
    )
    .agg(

        mean_partial_when_selected=(
            "partial_correlation",
            "mean"
        ),

        mean_abs_partial_when_selected=(
            "abs_partial_correlation",
            "mean"
        )
    )
)


edge_persistence_df = (
    edge_persistence_df
    .merge(
        selected_strength_df,

        on=[
            "stock_1",
            "stock_2"
        ],

        how="left"
    )
)


edge_persistence_df.to_csv(
    EDGE_PERSISTENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 同行业 vs 跨行业Persistence Summary
# ============================================================

industry_persistence_summary_df = (
    edge_persistence_df
    .groupby(
        "industry_relation",
        as_index=False
    )
    .agg(

        n_possible_pairs=(
            "persistence",
            "size"
        ),

        mean_persistence=(
            "persistence",
            "mean"
        ),

        median_persistence=(
            "persistence",
            "median"
        ),

        n_persistence_ge_080=(
            "persistence",
            lambda x:
                int(
                    (
                        x >= 0.80
                    )
                    .sum()
                )
        ),

        n_persistence_equal_1=(
            "persistence",
            lambda x:
                int(
                    np.isclose(
                        x,
                        1.0
                    )
                    .sum()
                )
        ),

        mean_abs_partial_when_selected=(
            "mean_abs_partial_when_selected",
            "mean"
        )
    )
)


industry_persistence_summary_df[
    "share_persistence_ge_080"
] = (
    industry_persistence_summary_df[
        "n_persistence_ge_080"
    ]
    /
    industry_persistence_summary_df[
        "n_possible_pairs"
    ]
)


industry_persistence_summary_df[
    "share_persistence_equal_1"
] = (
    industry_persistence_summary_df[
        "n_persistence_equal_1"
    ]
    /
    industry_persistence_summary_df[
        "n_possible_pairs"
    ]
)


industry_persistence_summary_df.to_csv(
    INDUSTRY_PERSISTENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. 相邻窗口行业Edge Turnover分解
# ============================================================

turnover_rows = []


def make_edge_set(
    df
):

    return {

        (
            row.stock_1,
            row.stock_2
        )

        for row in df.itertuples()
    }


def edge_relation_map_for_window(
    df
):

    return {

        (
            row.stock_1,
            row.stock_2
        ):
            bool(
                row.same_industry
            )

        for row in df.itertuples()
    }


for k in range(
    1,
    len(
        window_ids
    )
):

    previous_id = (
        window_ids[
            k - 1
        ]
    )


    current_id = (
        window_ids[
            k
        ]
    )


    previous = selected_df[
        selected_df[
            "window_id"
        ]
        ==
        previous_id
    ]


    current = selected_df[
        selected_df[
            "window_id"
        ]
        ==
        current_id
    ]


    E_previous = make_edge_set(
        previous
    )


    E_current = make_edge_set(
        current
    )


    relation_previous = (
        edge_relation_map_for_window(
            previous
        )
    )


    relation_current = (
        edge_relation_map_for_window(
            current
        )
    )


    lost_edges = (
        E_previous
        -
        E_current
    )


    gained_edges = (
        E_current
        -
        E_previous
    )


    # --------------------------------------------
    # Lost
    # --------------------------------------------

    lost_same = sum(

        relation_previous[
            edge
        ]

        for edge
        in lost_edges
    )


    lost_cross = (
        len(
            lost_edges
        )
        -
        lost_same
    )


    # --------------------------------------------
    # Gained
    # --------------------------------------------

    gained_same = sum(

        relation_current[
            edge
        ]

        for edge
        in gained_edges
    )


    gained_cross = (
        len(
            gained_edges
        )
        -
        gained_same
    )


    current_date = (
        rolling_df.loc[
            rolling_df[
                "window_id"
            ]
            ==
            current_id,
            "network_date"
        ]
        .iloc[0]
    )


    turnover_rows.append(
        {
            "window_from":
                previous_id,

            "window_to":
                current_id,

            "date_to":
                current_date,

            "lost_total":
                len(
                    lost_edges
                ),

            "lost_same_industry":
                lost_same,

            "lost_cross_industry":
                lost_cross,

            "gained_total":
                len(
                    gained_edges
                ),

            "gained_same_industry":
                gained_same,

            "gained_cross_industry":
                gained_cross,

            "net_same_industry_change":
                gained_same
                -
                lost_same,

            "net_cross_industry_change":
                gained_cross
                -
                lost_cross
        }
    )


industry_turnover_df = pd.DataFrame(
    turnover_rows
)


industry_turnover_df.to_csv(
    INDUSTRY_TURNOVER_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. 屏幕输出
# ============================================================

print(
    "\n======================================"
)

print(
    "窗口级同行业/跨行业动态分解"
)

print(
    "======================================"
)


print(
    window_decomposition_df.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "同行业 vs 跨行业 Persistence"
)

print(
    "======================================"
)


print(
    industry_persistence_summary_df.to_string(
        index=False
    )
)


# ============================================================
# 19. 找同行业比例最高和最低窗口
# ============================================================

highest_same_row = (
    window_decomposition_df
    .loc[
        window_decomposition_df[
            "same_industry_ratio"
        ]
        .idxmax()
    ]
)


lowest_same_row = (
    window_decomposition_df
    .loc[
        window_decomposition_df[
            "same_industry_ratio"
        ]
        .idxmin()
    ]
)


print(
    "\n同行业比例最高窗口："
)


print(
    highest_same_row.to_string()
)


print(
    "\n同行业比例最低窗口："
)


print(
    lowest_same_row.to_string()
)


# ============================================================
# 20. 图1：同行业边与跨行业边数量
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    window_decomposition_df[
        "network_date"
    ],
    window_decomposition_df[
        "same_industry_edges"
    ],
    marker="o",
    label="Same-industry edges"
)


ax.plot(
    window_decomposition_df[
        "network_date"
    ],
    window_decomposition_df[
        "cross_industry_edges"
    ],
    marker="o",
    label="Cross-industry edges"
)


ax.set_xlabel(
    "Window end date"
)

ax.set_ylabel(
    "Number of selected edges"
)

ax.set_title(
    "Rolling GLasso同行业与跨行业边数量变化"
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


EDGE_COUNT_FIGURE = (
    FIGURE_DIR
    / "rolling_glasso_same_vs_cross_edge_counts.png"
)


fig.savefig(
    EDGE_COUNT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 21. 图2：同行业边比例
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    window_decomposition_df[
        "network_date"
    ],
    window_decomposition_df[
        "same_industry_ratio"
    ],
    marker="o"
)


ax.set_xlabel(
    "Window end date"
)

ax.set_ylabel(
    "Same-industry edge ratio"
)

ax.set_title(
    "Rolling GLasso同行业边比例"
)

ax.grid(
    alpha=0.3
)

fig.tight_layout()


RATIO_FIGURE = (
    FIGURE_DIR
    / "rolling_glasso_same_industry_edge_ratio.png"
)


fig.savefig(
    RATIO_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 22. 图3：同行业 vs 跨行业平均条件关联强度
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    window_decomposition_df[
        "network_date"
    ],
    window_decomposition_df[
        "mean_abs_partial_same"
    ],
    marker="o",
    label="Same-industry"
)


ax.plot(
    window_decomposition_df[
        "network_date"
    ],
    window_decomposition_df[
        "mean_abs_partial_cross"
    ],
    marker="o",
    label="Cross-industry"
)


ax.set_xlabel(
    "Window end date"
)

ax.set_ylabel(
    "Mean absolute GLasso partial correlation"
)

ax.set_title(
    "同行业与跨行业边平均条件关联强度"
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


STRENGTH_FIGURE = (
    FIGURE_DIR
    / "rolling_glasso_same_vs_cross_partial_strength.png"
)


fig.savefig(
    STRENGTH_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 23. 图4：Lost edges行业分解
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    industry_turnover_df[
        "date_to"
    ],
    industry_turnover_df[
        "lost_same_industry"
    ],
    marker="o",
    label="Lost same-industry edges"
)


ax.plot(
    industry_turnover_df[
        "date_to"
    ],
    industry_turnover_df[
        "lost_cross_industry"
    ],
    marker="o",
    label="Lost cross-industry edges"
)


ax.set_xlabel(
    "Current window end date"
)

ax.set_ylabel(
    "Number of lost edges"
)

ax.set_title(
    "相邻Rolling网络Lost Edges的行业分解"
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


LOST_FIGURE = (
    FIGURE_DIR
    / "rolling_glasso_lost_edges_by_industry.png"
)


fig.savefig(
    LOST_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 24. 图5：Gained edges行业分解
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    industry_turnover_df[
        "date_to"
    ],
    industry_turnover_df[
        "gained_same_industry"
    ],
    marker="o",
    label="Gained same-industry edges"
)


ax.plot(
    industry_turnover_df[
        "date_to"
    ],
    industry_turnover_df[
        "gained_cross_industry"
    ],
    marker="o",
    label="Gained cross-industry edges"
)


ax.set_xlabel(
    "Current window end date"
)

ax.set_ylabel(
    "Number of gained edges"
)

ax.set_title(
    "相邻Rolling网络Gained Edges的行业分解"
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


GAINED_FIGURE = (
    FIGURE_DIR
    / "rolling_glasso_gained_edges_by_industry.png"
)


fig.savefig(
    GAINED_FIGURE,
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
    "同行业/跨行业动态分解完成"
)

print(
    "======================================"
)


print(
    "\n输出文件："
)


for path in [

    WINDOW_DECOMPOSITION_FILE,
    EDGE_PERSISTENCE_FILE,
    INDUSTRY_PERSISTENCE_FILE,
    INDUSTRY_TURNOVER_FILE,
    EDGE_COUNT_FIGURE,
    RATIO_FIGURE,
    STRENGTH_FIGURE,
    LOST_FIGURE,
    GAINED_FIGURE

]:

    print(
        path
    )