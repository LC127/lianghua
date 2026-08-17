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
# 输入文件
# ------------------------------------------------------------

STATIC_ROBUST_FILE = (
    PROCESSED_DIR
    / "block_length_robust_edges.csv"
)

ROLLING_EDGE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_history.csv"
)

# 如果存在，优先作为股票名称/行业的权威映射
STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ------------------------------------------------------------
# 输出文件
# ------------------------------------------------------------

ALL_PAIRS_FILE = (
    PROCESSED_DIR
    / "static_stable_vs_dynamic_persistent_all_pairs.csv"
)

SUMMARY_FILE = (
    PROCESSED_DIR
    / "static_stable_vs_dynamic_persistent_summary.csv"
)

COMMON_EDGE_FILE = (
    PROCESSED_DIR
    / "static_stable_dynamic_persistent_common_edges.csv"
)

STATIC_ONLY_FILE = (
    PROCESSED_DIR
    / "static_stable_but_not_dynamic_persistent.csv"
)

DYNAMIC_ONLY_FILE = (
    PROCESSED_DIR
    / "dynamic_persistent_but_not_static_stable.csv"
)

ALWAYS_PERSISTENT_FILE = (
    PROCESSED_DIR
    / "static_stable_always_persistent_edges.csv"
)


# ============================================================
# 2. 阈值
# ============================================================

STATIC_STABLE_THRESHOLD = 0.80

DYNAMIC_PERSISTENCE_THRESHOLD = 0.80

ALWAYS_PERSISTENT_THRESHOLD = 1.0


# ============================================================
# 3. 工具函数
# ============================================================

def normalize_code(x):
    """
    将股票代码统一为6位字符串。
    """

    return str(
        x
    ).strip().zfill(6)


def canonicalize_pair(
    stock_1,
    stock_2
):
    """
    将无向股票对统一排序。

    例如：
        600030, 000858
    统一转换成：
        000858, 600030

    注意：
    这里只处理代码，不处理名称和行业。
    名称和行业之后统一根据代码重新映射。
    """

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
    column_name
):
    """
    将CSV中的True/False、1/0等统一为bool。
    """

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
            f"{column_name} 中存在无法识别的布尔值："
            f"{bad_values}"
        )


    return converted.astype(
        bool
    )


# ============================================================
# 4. 读取原始Rolling Edge History
#
# 非常重要：
# 此处先不要canonicalize pair。
# 先利用原始stock_1/name_1与stock_2/name_2
# 建立正确代码 -> 名称/行业映射。
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
# 5. 建立正确的股票代码 -> 名称/行业映射
#
# 优先使用stock_info.csv。
# 如果不存在，则从原始rolling edge history建立。
# ============================================================

if STOCK_INFO_FILE.exists():

    print(
        "\n使用 stock_info.csv 建立股票名称/行业映射。"
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


    missing_columns = (
        required_columns
        -
        set(
            stock_info.columns
        )
    )


    if missing_columns:

        raise ValueError(
            "stock_info.csv 缺少以下字段："
            f"{missing_columns}"
        )


    # 防止同一个code出现多行冲突
    stock_info = (
        stock_info
        .drop_duplicates(
            subset=[
                "code"
            ],
            keep="first"
        )
        .copy()
    )


    metadata = stock_info[
        [
            "code",
            "name",
            "industry"
        ]
    ].copy()


else:

    print(
        "\n未找到 stock_info.csv，"
        "使用原始 rolling_glasso_edge_history.csv "
        "建立股票名称/行业映射。"
    )


    # --------------------------------------------------------
    # stock_1侧
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # stock_2侧
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 检查是否同一代码对应多个名称
    # --------------------------------------------------------

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
        name_conflict
        >
        1
    ).any():

        bad_codes = (
            name_conflict[
                name_conflict > 1
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "原始Rolling文件中存在同一股票代码"
            "对应多个名称的情况："
            f"{bad_codes}"
        )


    # --------------------------------------------------------
    # 检查是否同一代码对应多个行业
    # --------------------------------------------------------

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
        industry_conflict
        >
        1
    ).any():

        bad_codes = (
            industry_conflict[
                industry_conflict > 1
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "原始Rolling文件中存在同一股票代码"
            "对应多个行业的情况："
            f"{bad_codes}"
        )


    metadata = (
        metadata
        .drop_duplicates(
            subset=[
                "code"
            ],
            keep="first"
        )
        .copy()
    )


# ============================================================
# 6. 建立映射字典
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
    "\n股票元数据："
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
# 7. 一个简单的映射检查
# ============================================================

print(
    "\n======================================"
)

print(
    "股票代码名称映射检查"
)

print(
    "======================================"
)


for code in sorted(
    name_map.keys()
):

    print(
        code,
        "->",
        name_map.get(
            code
        ),
        "|",
        industry_map.get(
            code
        )
    )


# ============================================================
# 8. 现在才对Rolling股票对进行canonicalize
#
# 名称/行业字段以后全部重新生成，
# 不再使用原始name_1/name_2进行后续分析。
# ============================================================

rolling_df = rolling_raw.copy()


rolling_pairs = rolling_df.apply(

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
    for pair in rolling_pairs
]


rolling_df[
    "stock_2"
] = [
    pair[1]
    for pair in rolling_pairs
]


# ============================================================
# 9. canonicalize之后重新生成名称与行业
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


# ============================================================
# 10. selected转换为bool
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

    if col in rolling_df.columns:

        rolling_df[
            col
        ] = pd.to_datetime(
            rolling_df[
                col
            ]
        )


# ============================================================
# 12. 读取Static Block-Length Robustness
# ============================================================

static_df = pd.read_csv(
    STATIC_ROBUST_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


static_df[
    "stock_1"
] = (
    static_df[
        "stock_1"
    ]
    .apply(
        normalize_code
    )
)


static_df[
    "stock_2"
] = (
    static_df[
        "stock_2"
    ]
    .apply(
        normalize_code
    )
)


# ============================================================
# 13. Static股票对canonicalize
# ============================================================

static_pairs = static_df.apply(

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


static_df[
    "stock_1"
] = [
    pair[0]
    for pair in static_pairs
]


static_df[
    "stock_2"
] = [
    pair[1]
    for pair in static_pairs
]


# ============================================================
# 14. Static结果中的名称与行业重新映射
#
# 即使原文件已有name，也完全覆盖，
# 防止历史错误继续传播。
# ============================================================

static_df[
    "name_1"
] = (
    static_df[
        "stock_1"
    ]
    .map(
        name_map
    )
)


static_df[
    "name_2"
] = (
    static_df[
        "stock_2"
    ]
    .map(
        name_map
    )
)


static_df[
    "industry_1"
] = (
    static_df[
        "stock_1"
    ]
    .map(
        industry_map
    )
)


static_df[
    "industry_2"
] = (
    static_df[
        "stock_2"
    ]
    .map(
        industry_map
    )
)


static_df[
    "same_industry"
] = (
    static_df[
        "industry_1"
    ]
    ==
    static_df[
        "industry_2"
    ]
)


# ============================================================
# 15. 确定Static Stable
# ============================================================

if (
    "stable_all_block_lengths"
    in static_df.columns
):

    static_df[
        "stable_all_block_lengths"
    ] = convert_boolean_column(

        static_df[
            "stable_all_block_lengths"
        ],

        "stable_all_block_lengths"
    )


    static_df[
        "static_stable"
    ] = (
        static_df[
            "stable_all_block_lengths"
        ]
    )


elif (
    "min_frequency"
    in static_df.columns
):

    static_df[
        "static_stable"
    ] = (
        static_df[
            "min_frequency"
        ]
        >=
        STATIC_STABLE_THRESHOLD
    )


else:

    raise ValueError(
        "block_length_robust_edges.csv中"
        "既不存在stable_all_block_lengths，"
        "也不存在min_frequency。"
    )


# ============================================================
# 16. 检查所有股票代码是否都有名称/行业
# ============================================================

all_codes = set(
    rolling_df[
        "stock_1"
    ]
).union(
    set(
        rolling_df[
            "stock_2"
        ]
    )
)


missing_name_codes = [
    code
    for code in all_codes
    if code not in name_map
]


missing_industry_codes = [
    code
    for code in all_codes
    if code not in industry_map
]


if missing_name_codes:

    raise ValueError(
        "以下股票代码没有名称映射："
        f"{missing_name_codes}"
    )


if missing_industry_codes:

    raise ValueError(
        "以下股票代码没有行业映射："
        f"{missing_industry_codes}"
    )


# ============================================================
# 17. Dynamic Persistence
# ============================================================

total_windows = (
    rolling_df[
        "window_id"
    ]
    .nunique()
)


print(
    "\nRolling窗口总数：",
    total_windows
)


# ------------------------------------------------------------
# 基础动态指标
# ------------------------------------------------------------

dynamic_df = (
    rolling_df
    .groupby(
        [
            "stock_1",
            "stock_2"
        ],
        as_index=False
    )
    .agg(

        windows_selected=(
            "selected",
            "sum"
        ),

        total_windows=(
            "window_id",
            "nunique"
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


dynamic_df[
    "persistence"
] = (
    dynamic_df[
        "windows_selected"
    ]
    /
    dynamic_df[
        "total_windows"
    ]
)


# ============================================================
# 18. 被选择时的Partial Correlation
# ============================================================

selected_dynamic = rolling_df[
    rolling_df[
        "selected"
    ]
].copy()


selected_partial_df = (
    selected_dynamic
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
        ),

        first_selected_date=(
            "network_date",
            "min"
        ),

        last_selected_date=(
            "network_date",
            "max"
        )
    )
)


dynamic_df = dynamic_df.merge(

    selected_partial_df,

    on=[
        "stock_1",
        "stock_2"
    ],

    how="left"
)


# ============================================================
# 19. Dynamic Persistent / Always Persistent
# ============================================================

dynamic_df[
    "dynamic_persistent"
] = (
    dynamic_df[
        "persistence"
    ]
    >=
    DYNAMIC_PERSISTENCE_THRESHOLD
)


dynamic_df[
    "always_persistent"
] = np.isclose(
    dynamic_df[
        "persistence"
    ],
    ALWAYS_PERSISTENT_THRESHOLD
)


# ============================================================
# 20. Dynamic结果重新根据股票代码映射名称与行业
# ============================================================

dynamic_df[
    "name_1"
] = (
    dynamic_df[
        "stock_1"
    ]
    .map(
        name_map
    )
)


dynamic_df[
    "name_2"
] = (
    dynamic_df[
        "stock_2"
    ]
    .map(
        name_map
    )
)


dynamic_df[
    "industry_1"
] = (
    dynamic_df[
        "stock_1"
    ]
    .map(
        industry_map
    )
)


dynamic_df[
    "industry_2"
] = (
    dynamic_df[
        "stock_2"
    ]
    .map(
        industry_map
    )
)


dynamic_df[
    "same_industry"
] = (
    dynamic_df[
        "industry_1"
    ]
    ==
    dynamic_df[
        "industry_2"
    ]
)


# ============================================================
# 21. 整理Static需要保留的变量
# ============================================================

static_keep_columns = [
    "stock_1",
    "stock_2",
    "static_stable"
]


optional_static_columns = [
    "frequency_L10",
    "frequency_L20",
    "frequency_L40",
    "min_frequency",
    "mean_frequency",
    "max_frequency",
    "frequency_range",
    "stable_all_block_lengths",
    "core_all_block_lengths"
]


for col in optional_static_columns:

    if col in static_df.columns:

        static_keep_columns.append(
            col
        )


static_small = static_df[
    static_keep_columns
].copy()


# ============================================================
# 22. Static与Dynamic合并
# ============================================================

comparison_df = dynamic_df.merge(

    static_small,

    on=[
        "stock_1",
        "stock_2"
    ],

    how="left"
)


comparison_df[
    "static_stable"
] = (
    comparison_df[
        "static_stable"
    ]
    .fillna(
        False
    )
    .astype(
        bool
    )
)


# ============================================================
# 23. 合并之后再次重新生成名称/行业
#
# 这是最后一道保险。
# ============================================================

comparison_df[
    "name_1"
] = (
    comparison_df[
        "stock_1"
    ]
    .map(
        name_map
    )
)


comparison_df[
    "name_2"
] = (
    comparison_df[
        "stock_2"
    ]
    .map(
        name_map
    )
)


comparison_df[
    "industry_1"
] = (
    comparison_df[
        "stock_1"
    ]
    .map(
        industry_map
    )
)


comparison_df[
    "industry_2"
] = (
    comparison_df[
        "stock_2"
    ]
    .map(
        industry_map
    )
)


comparison_df[
    "same_industry"
] = (
    comparison_df[
        "industry_1"
    ]
    ==
    comparison_df[
        "industry_2"
    ]
)


# ============================================================
# 24. 四类稳定性分类
# ============================================================

def classify_edge(
    row
):

    static_stable = bool(
        row[
            "static_stable"
        ]
    )

    dynamic_persistent = bool(
        row[
            "dynamic_persistent"
        ]
    )


    if (
        static_stable
        and
        dynamic_persistent
    ):

        return (
            "Static stable + Dynamic persistent"
        )


    if (
        static_stable
        and
        not dynamic_persistent
    ):

        return (
            "Static stable only"
        )


    if (
        not static_stable
        and
        dynamic_persistent
    ):

        return (
            "Dynamic persistent only"
        )


    return (
        "Neither"
    )


comparison_df[
    "stability_class"
] = comparison_df.apply(
    classify_edge,
    axis=1
)


# ============================================================
# 25. 排序
# ============================================================

class_order = {

    "Static stable + Dynamic persistent":
        0,

    "Static stable only":
        1,

    "Dynamic persistent only":
        2,

    "Neither":
        3
}


comparison_df[
    "_class_order"
] = comparison_df[
    "stability_class"
].map(
    class_order
)


comparison_df = (
    comparison_df
    .sort_values(
        [
            "_class_order",
            "persistence",
            "mean_abs_partial_when_selected"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )
    .drop(
        columns=[
            "_class_order"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 26. 建立Edge Sets
# ============================================================

E_static = {

    canonicalize_pair(
        row.stock_1,
        row.stock_2
    )

    for row
    in comparison_df[
        comparison_df[
            "static_stable"
        ]
    ].itertuples()
}


E_dynamic = {

    canonicalize_pair(
        row.stock_1,
        row.stock_2
    )

    for row
    in comparison_df[
        comparison_df[
            "dynamic_persistent"
        ]
    ].itertuples()
}


E_always = {

    canonicalize_pair(
        row.stock_1,
        row.stock_2
    )

    for row
    in comparison_df[
        comparison_df[
            "always_persistent"
        ]
    ].itertuples()
}


E_common = (
    E_static
    &
    E_dynamic
)


E_static_only = (
    E_static
    -
    E_dynamic
)


E_dynamic_only = (
    E_dynamic
    -
    E_static
)


E_union = (
    E_static
    |
    E_dynamic
)


# ============================================================
# 27. 重合指标
# ============================================================

jaccard = (

    len(
        E_common
    )
    /
    len(
        E_union
    )

    if len(
        E_union
    ) > 0

    else np.nan
)


static_temporal_retention = (

    len(
        E_common
    )
    /
    len(
        E_static
    )

    if len(
        E_static
    ) > 0

    else np.nan
)


dynamic_supported_by_static = (

    len(
        E_common
    )
    /
    len(
        E_dynamic
    )

    if len(
        E_dynamic
    ) > 0

    else np.nan
)


static_always_set = (
    E_static
    &
    E_always
)


static_always_persistent_ratio = (

    len(
        static_always_set
    )
    /
    len(
        E_static
    )

    if len(
        E_static
    ) > 0

    else np.nan
)


# ============================================================
# 28. Summary
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "total_pairs":
                len(
                    comparison_df
                ),

            "static_stable_edges":
                len(
                    E_static
                ),

            "dynamic_persistent_edges":
                len(
                    E_dynamic
                ),

            "always_persistent_edges":
                len(
                    E_always
                ),

            "common_edges":
                len(
                    E_common
                ),

            "static_only_edges":
                len(
                    E_static_only
                ),

            "dynamic_only_edges":
                len(
                    E_dynamic_only
                ),

            "union_edges":
                len(
                    E_union
                ),

            "jaccard":
                jaccard,

            "static_temporal_retention":
                static_temporal_retention,

            "dynamic_supported_by_static":
                dynamic_supported_by_static,

            "static_always_persistent_edges":
                len(
                    static_always_set
                ),

            "static_always_persistent_ratio":
                static_always_persistent_ratio
        }
    ]
)


# ============================================================
# 29. 各类别结果
# ============================================================

common_df = comparison_df[
    comparison_df[
        "stability_class"
    ]
    ==
    "Static stable + Dynamic persistent"
].copy()


static_only_df = comparison_df[
    comparison_df[
        "stability_class"
    ]
    ==
    "Static stable only"
].copy()


dynamic_only_df = comparison_df[
    comparison_df[
        "stability_class"
    ]
    ==
    "Dynamic persistent only"
].copy()


always_persistent_df = comparison_df[
    (
        comparison_df[
            "static_stable"
        ]
    )
    &
    (
        comparison_df[
            "always_persistent"
        ]
    )
].copy()


# ============================================================
# 30. 保存CSV
# ============================================================

comparison_df.to_csv(
    ALL_PAIRS_FILE,
    index=False,
    encoding="utf-8-sig"
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


common_df.to_csv(
    COMMON_EDGE_FILE,
    index=False,
    encoding="utf-8-sig"
)


static_only_df.to_csv(
    STATIC_ONLY_FILE,
    index=False,
    encoding="utf-8-sig"
)


dynamic_only_df.to_csv(
    DYNAMIC_ONLY_FILE,
    index=False,
    encoding="utf-8-sig"
)


always_persistent_df.to_csv(
    ALWAYS_PERSISTENT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 31. 输出Summary
# ============================================================

print(
    "\n======================================"
)

print(
    "Static Stable vs Dynamic Persistent"
)

print(
    "======================================"
)


print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 32. 四类边数量
# ============================================================

print(
    "\n四类股票对数量："
)


print(
    comparison_df[
        "stability_class"
    ]
    .value_counts()
)


# ============================================================
# 33. Common Edges
# ============================================================

print(
    "\n======================================"
)

print(
    "Static Stable + Dynamic Persistent"
)

print(
    "======================================"
)


print(
    common_df[
        [
            "stock_1",
            "name_1",
            "industry_1",
            "stock_2",
            "name_2",
            "industry_2",
            "same_industry",
            "windows_selected",
            "persistence",
            "always_persistent",
            "mean_partial_when_selected"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 34. Static-only
# ============================================================

print(
    "\n======================================"
)

print(
    "Static Stable but NOT Dynamic Persistent"
)

print(
    "======================================"
)


if len(
    static_only_df
) > 0:

    print(
        static_only_df[
            [
                "stock_1",
                "name_1",
                "industry_1",
                "stock_2",
                "name_2",
                "industry_2",
                "windows_selected",
                "persistence",
                "min_frequency",
                "mean_partial_when_selected"
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "没有。"
    )


# ============================================================
# 35. Dynamic-only
# ============================================================

print(
    "\n======================================"
)

print(
    "Dynamic Persistent but NOT Static Stable"
)

print(
    "======================================"
)


if len(
    dynamic_only_df
) > 0:

    print(
        dynamic_only_df[
            [
                "stock_1",
                "name_1",
                "industry_1",
                "stock_2",
                "name_2",
                "industry_2",
                "windows_selected",
                "persistence",
                "min_frequency",
                "mean_partial_when_selected"
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "没有。"
    )


# ============================================================
# 36. Always Persistent
# ============================================================

print(
    "\n======================================"
)

print(
    "Static Stable + Always Persistent"
)

print(
    "======================================"
)


print(
    always_persistent_df[
        [
            "stock_1",
            "name_1",
            "industry_1",
            "stock_2",
            "name_2",
            "industry_2",
            "same_industry",
            "persistence",
            "min_frequency",
            "mean_partial_when_selected"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 37. 图1：四类边数量
# ============================================================

class_counts = (

    comparison_df[
        "stability_class"
    ]
    .value_counts()
    .reindex(
        [
            "Static stable + Dynamic persistent",
            "Static stable only",
            "Dynamic persistent only",
            "Neither"
        ],
        fill_value=0
    )
)


fig, ax = plt.subplots(
    figsize=(
        10,
        6
    )
)


ax.bar(
    class_counts.index,
    class_counts.values
)


ax.set_ylabel(
    "Number of stock pairs"
)


ax.set_title(
    "Static Stability与Dynamic Persistence分类"
)


ax.tick_params(
    axis="x",
    rotation=20
)


for i, value in enumerate(
    class_counts.values
):

    ax.text(
        i,
        value + 0.5,
        str(
            value
        ),
        ha="center"
    )


fig.tight_layout()


CLASS_FIGURE = (
    FIGURE_DIR
    / "static_stable_vs_dynamic_persistent_classes.png"
)


fig.savefig(
    CLASS_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 38. 图2：Persistence分布
#
# 注意：
# 新版Matplotlib使用tick_labels，而不是labels。
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


static_values = comparison_df.loc[
    comparison_df[
        "static_stable"
    ],
    "persistence"
]


nonstatic_values = comparison_df.loc[
    ~comparison_df[
        "static_stable"
    ],
    "persistence"
]


ax.boxplot(
    [
        static_values,
        nonstatic_values
    ],

    tick_labels=[
        "Static stable",
        "Not static stable"
    ]
)


ax.axhline(
    y=DYNAMIC_PERSISTENCE_THRESHOLD,
    linestyle="--",
    linewidth=1
)


ax.set_ylabel(
    "Dynamic edge persistence"
)


ax.set_title(
    "Static Stable边与非Static Stable边的动态持续率"
)


fig.tight_layout()


PERSISTENCE_FIGURE = (
    FIGURE_DIR
    / "static_status_vs_dynamic_persistence.png"
)


fig.savefig(
    PERSISTENCE_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 39. 图3：Static Selection Frequency vs Dynamic Persistence
# ============================================================

if (
    "min_frequency"
    in comparison_df.columns
):

    valid_scatter = comparison_df[
        comparison_df[
            "min_frequency"
        ]
        .notna()
    ].copy()


    fig, ax = plt.subplots(
        figsize=(
            8,
            7
        )
    )


    ax.scatter(
        valid_scatter[
            "min_frequency"
        ],
        valid_scatter[
            "persistence"
        ],
        alpha=0.7
    )


    ax.axvline(
        x=STATIC_STABLE_THRESHOLD,
        linestyle="--",
        linewidth=1
    )


    ax.axhline(
        y=DYNAMIC_PERSISTENCE_THRESHOLD,
        linestyle="--",
        linewidth=1
    )


    ax.set_xlabel(
        "Minimum static selection frequency "
        "across block lengths"
    )


    ax.set_ylabel(
        "Dynamic persistence"
    )


    ax.set_title(
        "样本重采样稳定性与时间持续性的关系"
    )


    fig.tight_layout()


    SCATTER_FIGURE = (
        FIGURE_DIR
        / "static_stability_vs_dynamic_persistence_scatter.png"
    )


    fig.savefig(
        SCATTER_FIGURE,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()


# ============================================================
# 40. 最终检查：输出几个关键股票代码
#
# 用于人工确认映射已经正确
# ============================================================

CHECK_CODES = [
    "000001",  # 平安银行
    "000568",  # 泸州老窖
    "000858",  # 五粮液
    "600030",  # 中信证券
    "600036",  # 招商银行
    "600519",  # 贵州茅台
    "601318",  # 中国平安
    "601288",  # 农业银行
    "601398"   # 工商银行
]


print(
    "\n======================================"
)

print(
    "最终名称映射人工检查"
)

print(
    "======================================"
)


for code in CHECK_CODES:

    print(
        f"{code} -> "
        f"{name_map.get(code)} | "
        f"{industry_map.get(code)}"
    )


# ============================================================
# 41. 完成
# ============================================================

print(
    "\n======================================"
)

print(
    "Static Stable vs Dynamic Persistent分析完成"
)

print(
    "======================================"
)


print(
    "\n输出文件："
)


output_paths = [
    ALL_PAIRS_FILE,
    SUMMARY_FILE,
    COMMON_EDGE_FILE,
    STATIC_ONLY_FILE,
    DYNAMIC_ONLY_FILE,
    ALWAYS_PERSISTENT_FILE,
    CLASS_FIGURE,
    PERSISTENCE_FIGURE
]


if (
    "min_frequency"
    in comparison_df.columns
):

    output_paths.append(
        SCATTER_FIGURE
    )


for path in output_paths:

    print(
        path
    )