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

EDGE_HISTORY_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_history.csv"
)

SUMMARY_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_network_summary.csv"
)


# ------------------------------------------------------------
# 输出
# ------------------------------------------------------------

TURNOVER_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_adjacent_network_turnover.csv"
)

EDGE_CHANGE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_adjacent_edge_changes.csv"
)

TOP_CHANGE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_top_network_changes.csv"
)


# ============================================================
# 2. 股票代码处理
# ============================================================

def normalize_code(x):

    return str(
        x
    ).strip().zfill(6)


def edge_key(
    stock_1,
    stock_2
):

    return tuple(
        sorted(
            [
                normalize_code(
                    stock_1
                ),
                normalize_code(
                    stock_2
                )
            ]
        )
    )


# ============================================================
# 3. 读取 Rolling Edge History
# ============================================================

edge_df = pd.read_csv(

    EDGE_HISTORY_FILE,

    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


edge_df[
    "stock_1"
] = (
    edge_df[
        "stock_1"
    ]
    .apply(
        normalize_code
    )
)


edge_df[
    "stock_2"
] = (
    edge_df[
        "stock_2"
    ]
    .apply(
        normalize_code
    )
)


# ============================================================
# 4. 处理 selected
#
# 防止CSV读取后selected是字符串
# ============================================================

if (
    edge_df[
        "selected"
    ].dtype
    !=
    bool
):

    edge_df[
        "selected"
    ] = (
        edge_df[
            "selected"
        ]
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


if (
    edge_df[
        "selected"
    ]
    .isna()
    .any()
):

    raise ValueError(
        "selected列中存在无法识别的值，请检查CSV。"
    )


# ============================================================
# 5. 日期处理
# ============================================================

for col in [
    "window_start",
    "window_end",
    "network_date"
]:

    if col in edge_df.columns:

        edge_df[
            col
        ] = pd.to_datetime(
            edge_df[
                col
            ]
        )


# ============================================================
# 6. 读取 Network Summary
# ============================================================

summary_df = pd.read_csv(
    SUMMARY_FILE
)


for col in [
    "window_start",
    "window_end",
    "network_date"
]:

    if col in summary_df.columns:

        summary_df[
            col
        ] = pd.to_datetime(
            summary_df[
                col
            ]
        )


summary_df = (
    summary_df
    .sort_values(
        "window_id"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 7. 数据完整性检查
# ============================================================

window_ids = (
    summary_df[
        "window_id"
    ]
    .astype(int)
    .tolist()
)


print(
    "Rolling窗口数量：",
    len(
        window_ids
    )
)


# 15只股票应该有105个股票对。
# 不强制写死，只输出每个window实际记录数量。
pair_count_by_window = (
    edge_df
    .groupby(
        "window_id"
    )
    .size()
)


print(
    "\n每个窗口记录的股票对数量："
)

print(
    pair_count_by_window
    .describe()
)


# ============================================================
# 8. 为每个窗口建立Edge Set
# ============================================================

edge_sets = {}


for window_id in window_ids:

    current = edge_df[
        (
            edge_df[
                "window_id"
            ]
            ==
            window_id
        )
        &
        (
            edge_df[
                "selected"
            ]
        )
    ]


    current_edges = {

        edge_key(
            row.stock_1,
            row.stock_2
        )

        for row
        in current.itertuples()
    }


    edge_sets[
        window_id
    ] = (
        current_edges
    )


# ============================================================
# 9. 建立股票名称、行业映射
# ============================================================

name_map = {}

industry_map = {}


for row in (
    edge_df
    .drop_duplicates(
        subset=[
            "stock_1"
        ]
    )
    .itertuples()
):

    name_map[
        row.stock_1
    ] = getattr(
        row,
        "name_1",
        row.stock_1
    )

    industry_map[
        row.stock_1
    ] = getattr(
        row,
        "industry_1",
        ""
    )


for row in (
    edge_df
    .drop_duplicates(
        subset=[
            "stock_2"
        ]
    )
    .itertuples()
):

    name_map[
        row.stock_2
    ] = getattr(
        row,
        "name_2",
        row.stock_2
    )

    industry_map[
        row.stock_2
    ] = getattr(
        row,
        "industry_2",
        ""
    )


# ============================================================
# 10. 相邻网络比较
# ============================================================

turnover_rows = []

edge_change_rows = []


for k in range(
    1,
    len(
        summary_df
    )
):

    previous_row = (
        summary_df
        .iloc[
            k - 1
        ]
    )


    current_row = (
        summary_df
        .iloc[
            k
        ]
    )


    previous_id = int(
        previous_row[
            "window_id"
        ]
    )


    current_id = int(
        current_row[
            "window_id"
        ]
    )


    E_previous = (
        edge_sets[
            previous_id
        ]
    )


    E_current = (
        edge_sets[
            current_id
        ]
    )


    # --------------------------------------------------------
    # 10.1 四种集合
    # --------------------------------------------------------

    common_edges = (
        E_previous
        &
        E_current
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


    union_edges = (
        E_previous
        |
        E_current
    )


    # --------------------------------------------------------
    # 10.2 数量
    # --------------------------------------------------------

    n_previous = len(
        E_previous
    )


    n_current = len(
        E_current
    )


    n_common = len(
        common_edges
    )


    n_lost = len(
        lost_edges
    )


    n_gained = len(
        gained_edges
    )


    n_union = len(
        union_edges
    )


    # --------------------------------------------------------
    # 10.3 Jaccard
    # --------------------------------------------------------

    if n_union > 0:

        jaccard = (
            n_common
            /
            n_union
        )

    else:

        # 两张都是空图时，可视为完全相同
        jaccard = 1.0


    # --------------------------------------------------------
    # 10.4 Turnover
    # --------------------------------------------------------

    turnover = (
        1.0
        -
        jaccard
    )


    # 也等价于：
    if n_union > 0:

        turnover_check = (
            (
                n_lost
                +
                n_gained
            )
            /
            n_union
        )

    else:

        turnover_check = 0.0


    if not np.isclose(
        turnover,
        turnover_check
    ):

        raise RuntimeError(
            "Turnover计算校验失败。"
        )


    # --------------------------------------------------------
    # 10.5 方向性指标
    # --------------------------------------------------------

    previous_retention = (

        n_common
        /
        n_previous

        if n_previous > 0

        else np.nan
    )


    current_inherited_share = (

        n_common
        /
        n_current

        if n_current > 0

        else np.nan
    )


    lost_rate_previous = (

        n_lost
        /
        n_previous

        if n_previous > 0

        else np.nan
    )


    gained_rate_current = (

        n_gained
        /
        n_current

        if n_current > 0

        else np.nan
    )


    net_edge_change = (
        n_current
        -
        n_previous
    )


    gross_edge_changes = (
        n_lost
        +
        n_gained
    )


    # --------------------------------------------------------
    # 10.6 保存窗口级结果
    # --------------------------------------------------------

    turnover_rows.append(
        {
            "window_from":
                previous_id,

            "window_to":
                current_id,

            "date_from":
                previous_row[
                    "network_date"
                ],

            "date_to":
                current_row[
                    "network_date"
                ],

            "edges_from":
                n_previous,

            "edges_to":
                n_current,

            "common_edges":
                n_common,

            "lost_edges":
                n_lost,

            "gained_edges":
                n_gained,

            "union_edges":
                n_union,

            "net_edge_change":
                net_edge_change,

            "gross_edge_changes":
                gross_edge_changes,

            "jaccard":
                jaccard,

            "turnover":
                turnover,

            "previous_edge_retention":
                previous_retention,

            "current_inherited_share":
                current_inherited_share,

            "lost_rate_previous":
                lost_rate_previous,

            "gained_rate_current":
                gained_rate_current
        }
    )


    # ========================================================
    # 11. 保存具体Lost Edges
    # ========================================================

    for stock_1, stock_2 in sorted(
        lost_edges
    ):

        edge_change_rows.append(
            {
                "window_from":
                    previous_id,

                "window_to":
                    current_id,

                "date_from":
                    previous_row[
                        "network_date"
                    ],

                "date_to":
                    current_row[
                        "network_date"
                    ],

                "stock_1":
                    stock_1,

                "name_1":
                    name_map.get(
                        stock_1,
                        stock_1
                    ),

                "industry_1":
                    industry_map.get(
                        stock_1,
                        ""
                    ),

                "stock_2":
                    stock_2,

                "name_2":
                    name_map.get(
                        stock_2,
                        stock_2
                    ),

                "industry_2":
                    industry_map.get(
                        stock_2,
                        ""
                    ),

                "change_type":
                    "lost"
            }
        )


    # ========================================================
    # 12. 保存具体Gained Edges
    # ========================================================

    for stock_1, stock_2 in sorted(
        gained_edges
    ):

        edge_change_rows.append(
            {
                "window_from":
                    previous_id,

                "window_to":
                    current_id,

                "date_from":
                    previous_row[
                        "network_date"
                    ],

                "date_to":
                    current_row[
                        "network_date"
                    ],

                "stock_1":
                    stock_1,

                "name_1":
                    name_map.get(
                        stock_1,
                        stock_1
                    ),

                "industry_1":
                    industry_map.get(
                        stock_1,
                        ""
                    ),

                "stock_2":
                    stock_2,

                "name_2":
                    name_map.get(
                        stock_2,
                        stock_2
                    ),

                "industry_2":
                    industry_map.get(
                        stock_2,
                        ""
                    ),

                "change_type":
                    "gained"
            }
        )


# ============================================================
# 13. 转DataFrame
# ============================================================

turnover_df = pd.DataFrame(
    turnover_rows
)


edge_change_df = pd.DataFrame(
    edge_change_rows
)


# ============================================================
# 14. 根据Turnover排序
# ============================================================

turnover_df[
    "turnover_rank"
] = (
    turnover_df[
        "turnover"
    ]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)
)


turnover_df = (
    turnover_df
    .sort_values(
        "window_to"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 15. 保存全部结果
# ============================================================

turnover_df.to_csv(
    TURNOVER_FILE,
    index=False,
    encoding="utf-8-sig"
)


edge_change_df.to_csv(
    EDGE_CHANGE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 网络变化最大的前5个时期
# ============================================================

top_change_df = (
    turnover_df
    .sort_values(
        [
            "turnover",
            "gross_edge_changes"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(
        5
    )
    .copy()
)


top_change_df.to_csv(
    TOP_CHANGE_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n======================================"
)

print(
    "网络Turnover最高的5个相邻窗口"
)

print(
    "======================================"
)


print(
    top_change_df[
        [
            "window_from",
            "window_to",
            "date_from",
            "date_to",
            "edges_from",
            "edges_to",
            "common_edges",
            "lost_edges",
            "gained_edges",
            "jaccard",
            "turnover"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 17. 找到最大Turnover窗口的具体Edge Changes
# ============================================================

max_turnover_row = (
    top_change_df
    .iloc[
        0
    ]
)


max_from = int(
    max_turnover_row[
        "window_from"
    ]
)


max_to = int(
    max_turnover_row[
        "window_to"
    ]
)


max_change_edges = edge_change_df[
    (
        edge_change_df[
            "window_from"
        ]
        ==
        max_from
    )
    &
    (
        edge_change_df[
            "window_to"
        ]
        ==
        max_to
    )
]


print(
    "\n======================================"
)

print(
    "变化率最高时期的具体边变化"
)

print(
    "======================================"
)


print(
    max_change_edges[
        [
            "stock_1",
            "name_1",
            "stock_2",
            "name_2",
            "change_type"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 18. 图1：Jaccard和Turnover随时间变化
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        11,
        6
    )
)


ax.plot(
    turnover_df[
        "date_to"
    ],
    turnover_df[
        "jaccard"
    ],
    marker="o",
    label="Jaccard similarity"
)


ax.plot(
    turnover_df[
        "date_to"
    ],
    turnover_df[
        "turnover"
    ],
    marker="o",
    label="Network turnover"
)


ax.set_xlabel(
    "Current window end date"
)

ax.set_ylabel(
    "Value"
)

ax.set_title(
    "Rolling Graphical Lasso相邻网络结构变化"
)

ax.grid(
    alpha=0.3
)

ax.legend()

fig.tight_layout()


TURNOVER_FIGURE = (
    FIGURE_DIR
    / "rolling_glasso_adjacent_network_turnover.png"
)


fig.savefig(
    TURNOVER_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 19. 图2：Lost / Gained Edge Count
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


x = np.arange(
    len(
        turnover_df
    )
)


width = 0.38


ax.bar(
    x - width / 2,
    turnover_df[
        "lost_edges"
    ],
    width=width,
    label="Lost edges"
)


ax.bar(
    x + width / 2,
    turnover_df[
        "gained_edges"
    ],
    width=width,
    label="Gained edges"
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    turnover_df[
        "date_to"
    ]
    .dt
    .strftime(
        "%Y-%m"
    ),
    rotation=60,
    ha="right"
)


ax.set_xlabel(
    "Current window end date"
)

ax.set_ylabel(
    "Number of edges"
)

ax.set_title(
    "相邻Rolling网络中的边退出与新增"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

fig.tight_layout()


EDGE_CHANGE_FIGURE = (
    FIGURE_DIR
    / "rolling_glasso_lost_gained_edges.png"
)


fig.savefig(
    EDGE_CHANGE_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 20. 图3：Net Edge Change
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.bar(
    turnover_df[
        "date_to"
    ],
    turnover_df[
        "net_edge_change"
    ]
)


ax.axhline(
    y=0,
    linewidth=1
)


ax.set_xlabel(
    "Current window end date"
)

ax.set_ylabel(
    "Net edge change"
)

ax.set_title(
    "Rolling Graphical Lasso网络净边数变化"
)

ax.grid(
    axis="y",
    alpha=0.3
)

fig.autofmt_xdate()

fig.tight_layout()


NET_CHANGE_FIGURE = (
    FIGURE_DIR
    / "rolling_glasso_net_edge_change.png"
)


fig.savefig(
    NET_CHANGE_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 21. 完成
# ============================================================

print(
    "\n======================================"
)

print(
    "相邻网络变化率分析完成"
)

print(
    "======================================"
)


print(
    "\n输出文件："
)


for path in [

    TURNOVER_FILE,
    EDGE_CHANGE_FILE,
    TOP_CHANGE_FILE,
    TURNOVER_FIGURE,
    EDGE_CHANGE_FIGURE,
    NET_CHANGE_FIGURE

]:

    print(
        path
    )