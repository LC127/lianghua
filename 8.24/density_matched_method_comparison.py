from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径
# ============================================================

PROJECT_DIR = Path("stock_network")

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


GLASSO_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_history.csv"
)

PARTIAL_FILE = (
    PROCESSED_DIR
    / "rolling_partial_edge_history.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ============================================================
# 2. 输出
# ============================================================

MATCHED_PARTIAL_FILE = (
    PROCESSED_DIR
    / "density_matched_partial_edge_history.csv"
)

WINDOW_COMPARISON_FILE = (
    PROCESSED_DIR
    / "density_matched_method_window_comparison.csv"
)

EDGE_PERSISTENCE_FILE = (
    PROCESSED_DIR
    / "density_matched_method_edge_persistence.csv"
)

METHOD_SUMMARY_FILE = (
    PROCESSED_DIR
    / "density_matched_method_summary.csv"
)

CORE_COMPARISON_FILE = (
    PROCESSED_DIR
    / "density_matched_persistent_core_comparison.csv"
)


# ============================================================
# 3. 参数
# ============================================================

PERSISTENCE_THRESHOLD = 0.80


# ============================================================
# 4. 工具函数
# ============================================================

def normalize_code(x):

    s = str(x).strip()

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


def canonical_pair(a, b):

    a = normalize_code(a)
    b = normalize_code(b)

    if a <= b:
        return a, b

    return b, a


def convert_bool(series, name):

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
# 5. Stock Metadata
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
        x
        for x in code_candidates
        if x in stock_info.columns
    ),
    None
)

name_col = next(
    (
        x
        for x in name_candidates
        if x in stock_info.columns
    ),
    None
)

industry_col = next(
    (
        x
        for x in industry_candidates
        if x in stock_info.columns
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
        "stock_info.csv中无法识别代码、名称或行业字段。"
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
# 6. 读取GLasso
# ============================================================

glasso = pd.read_csv(
    GLASSO_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


if "network_date" not in glasso.columns:

    if "window_end" in glasso.columns:

        glasso[
            "network_date"
        ] = glasso[
            "window_end"
        ]

    else:

        raise ValueError(
            "GLasso文件缺少network_date/window_end。"
        )


glasso[
    "network_date"
] = pd.to_datetime(
    glasso[
        "network_date"
    ]
)


for col in [
    "stock_1",
    "stock_2"
]:

    glasso[col] = (
        glasso[col]
        .apply(
            normalize_code
        )
    )


pairs = glasso.apply(
    lambda x:
        canonical_pair(
            x["stock_1"],
            x["stock_2"]
        ),
    axis=1
)


glasso["stock_1"] = [
    x[0]
    for x in pairs
]

glasso["stock_2"] = [
    x[1]
    for x in pairs
]


glasso[
    "selected"
] = convert_bool(
    glasso[
        "selected"
    ],
    "GLasso selected"
)


# ============================================================
# 7. 读取Rolling Partial
# ============================================================

partial = pd.read_csv(
    PARTIAL_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


partial[
    "network_date"
] = pd.to_datetime(
    partial[
        "network_date"
    ]
)


for col in [
    "stock_1",
    "stock_2"
]:

    partial[col] = (
        partial[col]
        .apply(
            normalize_code
        )
    )


pairs = partial.apply(
    lambda x:
        canonical_pair(
            x["stock_1"],
            x["stock_2"]
        ),
    axis=1
)


partial["stock_1"] = [
    x[0]
    for x in pairs
]

partial["stock_2"] = [
    x[1]
    for x in pairs
]


partial[
    "partial_correlation"
] = pd.to_numeric(
    partial[
        "partial_correlation"
    ],
    errors="raise"
)


partial[
    "abs_partial_correlation"
] = (
    partial[
        "partial_correlation"
    ]
    .abs()
)


# ============================================================
# 8. 重新映射Metadata
# ============================================================

for df in [
    glasso,
    partial
]:

    df["name_1"] = (
        df[
            "stock_1"
        ]
        .map(
            name_map
        )
    )

    df["name_2"] = (
        df[
            "stock_2"
        ]
        .map(
            name_map
        )
    )

    df["industry_1"] = (
        df[
            "stock_1"
        ]
        .map(
            industry_map
        )
    )

    df["industry_2"] = (
        df[
            "stock_2"
        ]
        .map(
            industry_map
        )
    )


    if (
        df[
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
            "Metadata映射失败。"
        )


    df[
        "same_industry"
    ] = (
        df[
            "industry_1"
        ]
        ==
        df[
            "industry_2"
        ]
    )


# ============================================================
# 9. 仅保留共同Network Dates
# ============================================================

glasso_dates = set(
    glasso[
        "network_date"
    ]
    .unique()
)


partial_dates = set(
    partial[
        "network_date"
    ]
    .unique()
)


common_dates = sorted(
    glasso_dates
    &
    partial_dates
)


if len(
    common_dates
) == 0:

    raise ValueError(
        "GLasso与Partial没有共同Network Date。"
    )


glasso = (
    glasso[
        glasso[
            "network_date"
        ]
        .isin(
            common_dates
        )
    ]
    .copy()
)


partial = (
    partial[
        partial[
            "network_date"
        ]
        .isin(
            common_dates
        )
    ]
    .copy()
)


print(
    "共同Network Dates：",
    len(
        common_dates
    )
)


# ============================================================
# 10. 验证每个日期股票对完全一致
# ============================================================

for date in common_dates:

    g = glasso[
        glasso[
            "network_date"
        ]
        ==
        date
    ]


    p = partial[
        partial[
            "network_date"
        ]
        ==
        date
    ]


    g_pairs = set(
        zip(
            g[
                "stock_1"
            ],
            g[
                "stock_2"
            ]
        )
    )


    p_pairs = set(
        zip(
            p[
                "stock_1"
            ],
            p[
                "stock_2"
            ]
        )
    )


    if g_pairs != p_pairs:

        raise ValueError(
            f"{date}: GLasso与Partial股票对不一致。"
        )


# ============================================================
# 11. Density Matching
#
# 对每个network_date：
# GLasso有m_t条边，
# Partial选择|rho|最大的前m_t条。
# ============================================================

matched_rows = []


for date in common_dates:

    glasso_t = (
        glasso[
            glasso[
                "network_date"
            ]
            ==
            date
        ]
        .copy()
    )


    partial_t = (
        partial[
            partial[
                "network_date"
            ]
            ==
            date
        ]
        .copy()
    )


    m_t = int(
        glasso_t[
            "selected"
        ]
        .sum()
    )


    if (
        m_t < 0
        or
        m_t > len(
            partial_t
        )
    ):

        raise ValueError(
            f"{date}: 非法的GLasso边数 {m_t}"
        )


    # --------------------------------------------------------
    # 对Partial按照绝对值排序
    #
    # 为避免并列值导致超过m_t条，
    # 用稳定排序 + 精确head(m_t)。
    # --------------------------------------------------------

    partial_t = (
        partial_t
        .sort_values(
            [
                "abs_partial_correlation",
                "stock_1",
                "stock_2"
            ],
            ascending=[
                False,
                True,
                True
            ],
            kind="mergesort"
        )
        .reset_index(
            drop=True
        )
    )


    partial_t[
        "density_match_rank"
    ] = np.arange(
        1,
        len(
            partial_t
        )
        +
        1
    )


    partial_t[
        "selected_density_matched"
    ] = (
        partial_t[
            "density_match_rank"
        ]
        <=
        m_t
    )


    partial_t[
        "glasso_edge_count_target"
    ] = (
        m_t
    )


    # 检查必须完全相等
    selected_count = int(
        partial_t[
            "selected_density_matched"
        ]
        .sum()
    )


    if selected_count != m_t:

        raise RuntimeError(
            f"{date}: density matching失败。"
        )


    matched_rows.append(
        partial_t
    )


matched_partial = pd.concat(
    matched_rows,
    ignore_index=True
)


matched_partial.to_csv(
    MATCHED_PARTIAL_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 每个日期进行方法比较
# ============================================================

window_rows = []


for date in common_dates:

    g = (
        glasso[
            glasso[
                "network_date"
            ]
            ==
            date
        ]
        .copy()
    )


    p = (
        matched_partial[
            matched_partial[
                "network_date"
            ]
            ==
            date
        ]
        .copy()
    )


    g_set = set(
        zip(
            g.loc[
                g[
                    "selected"
                ],
                "stock_1"
            ],

            g.loc[
                g[
                    "selected"
                ],
                "stock_2"
            ]
        )
    )


    p_set = set(
        zip(
            p.loc[
                p[
                    "selected_density_matched"
                ],
                "stock_1"
            ],

            p.loc[
                p[
                    "selected_density_matched"
                ],
                "stock_2"
            ]
        )
    )


    common = (
        g_set
        &
        p_set
    )


    union = (
        g_set
        |
        p_set
    )


    m_t = len(
        g_set
    )


    jaccard = (
        len(
            common
        )
        /
        len(
            union
        )
        if len(
            union
        ) > 0
        else 1.0
    )


    # 因为两个网络边数相同：
    #
    # common / m_t
    #
    # 可以解释为Edge Agreement Rate
    agreement_rate = (
        len(
            common
        )
        /
        m_t
        if m_t > 0
        else 1.0
    )


    g_selected = (
        g[
            g[
                "selected"
            ]
        ]
    )


    p_selected = (
        p[
            p[
                "selected_density_matched"
            ]
        ]
    )


    g_same = int(
        g_selected[
            "same_industry"
        ]
        .sum()
    )


    p_same = int(
        p_selected[
            "same_industry"
        ]
        .sum()
    )


    window_rows.append(
        {
            "network_date":
                date,

            "edge_count":
                m_t,

            "glasso_same_edges":
                g_same,

            "glasso_cross_edges":
                m_t
                -
                g_same,

            "partial_same_edges":
                p_same,

            "partial_cross_edges":
                m_t
                -
                p_same,

            "common_edges":
                len(
                    common
                ),

            "union_edges":
                len(
                    union
                ),

            "edge_agreement_rate":
                agreement_rate,

            "edge_jaccard":
                jaccard,

            "glasso_same_ratio":
                (
                    g_same
                    /
                    m_t
                    if m_t > 0
                    else np.nan
                ),

            "partial_same_ratio":
                (
                    p_same
                    /
                    m_t
                    if m_t > 0
                    else np.nan
                )
        }
    )


window_comparison = pd.DataFrame(
    window_rows
)


window_comparison.to_csv(
    WINDOW_COMPARISON_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. 为两种方法构造统一Edge History
# ============================================================

glasso_history = (
    glasso[
        [
            "network_date",
            "stock_1",
            "stock_2",
            "name_1",
            "name_2",
            "industry_1",
            "industry_2",
            "same_industry",
            "selected"
        ]
    ]
    .rename(
        columns={
            "selected":
                "selected_edge"
        }
    )
    .copy()
)


glasso_history[
    "method"
] = (
    "Graphical Lasso"
)


partial_history = (
    matched_partial[
        [
            "network_date",
            "stock_1",
            "stock_2",
            "name_1",
            "name_2",
            "industry_1",
            "industry_2",
            "same_industry",
            "selected_density_matched"
        ]
    ]
    .rename(
        columns={
            "selected_density_matched":
                "selected_edge"
        }
    )
    .copy()
)


partial_history[
    "method"
] = (
    "Density-matched Partial"
)


combined_history = pd.concat(
    [
        glasso_history,
        partial_history
    ],
    ignore_index=True
)


# ============================================================
# 14. Edge Persistence
# ============================================================

persistence_rows = []


for (
    method,
    stock_1,
    stock_2
), group in combined_history.groupby(
    [
        "method",
        "stock_1",
        "stock_2"
    ]
):

    persistence = (
        group[
            "selected_edge"
        ]
        .mean()
    )


    persistence_rows.append(
        {
            "method":
                method,

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

            "n_windows":
                len(
                    group
                ),

            "windows_selected":
                int(
                    group[
                        "selected_edge"
                    ]
                    .sum()
                ),

            "persistence":
                persistence,

            "persistent_08":
                (
                    persistence
                    >=
                    PERSISTENCE_THRESHOLD
                ),

            "always_selected":
                np.isclose(
                    persistence,
                    1.0
                )
        }
    )


persistence_df = pd.DataFrame(
    persistence_rows
)


persistence_df.to_csv(
    EDGE_PERSISTENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. Turnover / State-change Rate函数
# ============================================================

def calculate_dynamic_metrics(
    history,
    method
):

    history = (
        history[
            history[
                "method"
            ]
            ==
            method
        ]
        .copy()
    )


    dates = sorted(
        history[
            "network_date"
        ]
        .unique()
    )


    turnover_values = []

    gross_values = []

    same_changes = 0

    cross_changes = 0


    for t in range(
        1,
        len(
            dates
        )
    ):

        before = (
            history[
                history[
                    "network_date"
                ]
                ==
                dates[
                    t - 1
                ]
            ]
            .set_index(
                [
                    "stock_1",
                    "stock_2"
                ]
            )
        )


        after = (
            history[
                history[
                    "network_date"
                ]
                ==
                dates[
                    t
                ]
            ]
            .set_index(
                [
                    "stock_1",
                    "stock_2"
                ]
            )
        )


        merged = (
            before[
                [
                    "selected_edge",
                    "same_industry"
                ]
            ]
            .rename(
                columns={
                    "selected_edge":
                        "before"
                }
            )
            .join(
                after[
                    [
                        "selected_edge"
                    ]
                ]
                .rename(
                    columns={
                        "selected_edge":
                            "after"
                    }
                ),

                how="inner"
            )
        )


        b = (
            merged[
                "before"
            ]
            .astype(bool)
        )


        a = (
            merged[
                "after"
            ]
            .astype(bool)
        )


        common = int(
            (
                b
                &
                a
            )
            .sum()
        )


        lost = int(
            (
                b
                &
                ~a
            )
            .sum()
        )


        gained = int(
            (
                ~b
                &
                a
            )
            .sum()
        )


        union = (
            common
            +
            lost
            +
            gained
        )


        turnover = (
            1
            -
            common
            /
            union
            if union > 0
            else 0.0
        )


        gross = (
            lost
            +
            gained
        )


        turnover_values.append(
            turnover
        )


        gross_values.append(
            gross
        )


        changed = (
            b
            !=
            a
        )


        same_mask = (
            merged[
                "same_industry"
            ]
            .astype(bool)
        )


        same_changes += int(
            (
                changed
                &
                same_mask
            )
            .sum()
        )


        cross_changes += int(
            (
                changed
                &
                ~same_mask
            )
            .sum()
        )


    # Candidate pair数量
    pair_info = (
        history[
            [
                "stock_1",
                "stock_2",
                "same_industry"
            ]
        ]
        .drop_duplicates()
    )


    n_same_pairs = int(
        pair_info[
            "same_industry"
        ]
        .sum()
    )


    n_cross_pairs = (
        len(
            pair_info
        )
        -
        n_same_pairs
    )


    n_transitions = (
        len(
            dates
        )
        -
        1
    )


    same_rate = (
        same_changes
        /
        (
            n_same_pairs
            *
            n_transitions
        )
    )


    cross_rate = (
        cross_changes
        /
        (
            n_cross_pairs
            *
            n_transitions
        )
    )


    return {
        "mean_turnover":
            np.mean(
                turnover_values
            ),

        "mean_gross_edge_changes":
            np.mean(
                gross_values
            ),

        "same_state_change_rate":
            same_rate,

        "cross_state_change_rate":
            cross_rate
    }


# ============================================================
# 16. Method Summary
# ============================================================

method_rows = []


for method in [
    "Graphical Lasso",
    "Density-matched Partial"
]:

    persistence_group = (
        persistence_df[
            persistence_df[
                "method"
            ]
            ==
            method
        ]
    )


    same = (
        persistence_group[
            persistence_group[
                "same_industry"
            ]
        ]
    )


    cross = (
        persistence_group[
            ~persistence_group[
                "same_industry"
            ]
        ]
    )


    dynamics = (
        calculate_dynamic_metrics(
            combined_history,
            method
        )
    )


    method_rows.append(
        {
            "method":
                method,

            "n_windows":
                len(
                    common_dates
                ),

            "mean_edge_count":
                combined_history[
                    (
                        combined_history[
                            "method"
                        ]
                        ==
                        method
                    )
                ]
                .groupby(
                    "network_date"
                )[
                    "selected_edge"
                ]
                .sum()
                .mean(),

            "same_mean_persistence":
                same[
                    "persistence"
                ]
                .mean(),

            "cross_mean_persistence":
                cross[
                    "persistence"
                ]
                .mean(),

            "same_persistent_rate_08":
                same[
                    "persistent_08"
                ]
                .mean(),

            "cross_persistent_rate_08":
                cross[
                    "persistent_08"
                ]
                .mean(),

            "same_always_rate":
                same[
                    "always_selected"
                ]
                .mean(),

            "cross_always_rate":
                cross[
                    "always_selected"
                ]
                .mean(),

            **dynamics
        }
    )


method_summary = pd.DataFrame(
    method_rows
)


method_summary.to_csv(
    METHOD_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. Persistent Core比较
# ============================================================

g_core = persistence_df[
    (
        persistence_df[
            "method"
        ]
        ==
        "Graphical Lasso"
    )
    &
    (
        persistence_df[
            "persistent_08"
        ]
    )
].copy()


p_core = persistence_df[
    (
        persistence_df[
            "method"
        ]
        ==
        "Density-matched Partial"
    )
    &
    (
        persistence_df[
            "persistent_08"
        ]
    )
].copy()


g_core_set = set(
    zip(
        g_core[
            "stock_1"
        ],
        g_core[
            "stock_2"
        ]
    )
)


p_core_set = set(
    zip(
        p_core[
            "stock_1"
        ],
        p_core[
            "stock_2"
        ]
    )
)


common_core = (
    g_core_set
    &
    p_core_set
)


union_core = (
    g_core_set
    |
    p_core_set
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


core_rows = []


all_pairs = sorted(
    g_core_set
    |
    p_core_set
)


for stock_1, stock_2 in all_pairs:

    in_glasso = (
        (
            stock_1,
            stock_2
        )
        in
        g_core_set
    )


    in_partial = (
        (
            stock_1,
            stock_2
        )
        in
        p_core_set
    )


    if (
        in_glasso
        and
        in_partial
    ):

        status = (
            "Common persistent core"
        )

    elif in_glasso:

        status = (
            "GLasso-only persistent core"
        )

    else:

        status = (
            "Partial-only persistent core"
        )


    core_rows.append(
        {
            "stock_1":
                stock_1,

            "name_1":
                name_map[
                    stock_1
                ],

            "industry_1":
                industry_map[
                    stock_1
                ],

            "stock_2":
                stock_2,

            "name_2":
                name_map[
                    stock_2
                ],

            "industry_2":
                industry_map[
                    stock_2
                ],

            "same_industry":
                (
                    industry_map[
                        stock_1
                    ]
                    ==
                    industry_map[
                        stock_2
                    ]
                ),

            "in_glasso_core":
                in_glasso,

            "in_partial_core":
                in_partial,

            "core_status":
                status,

            "overall_core_jaccard":
                core_jaccard
        }
    )


core_comparison = pd.DataFrame(
    core_rows
)


core_comparison.to_csv(
    CORE_COMPARISON_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. Persistence相关性
# ============================================================

persistence_wide = (
    persistence_df[
        [
            "method",
            "stock_1",
            "stock_2",
            "persistence"
        ]
    ]
    .pivot(
        index=[
            "stock_1",
            "stock_2"
        ],
        columns="method",
        values="persistence"
    )
)


persistence_spearman = (
    persistence_wide[
        "Graphical Lasso"
    ]
    .corr(
        persistence_wide[
            "Density-matched Partial"
        ],
        method="spearman"
    )
)


# ============================================================
# 19. 屏幕输出
# ============================================================

print(
    "\n======================================"
)

print(
    "Density-matched Method Summary"
)

print(
    "======================================"
)


print(
    method_summary.to_string(
        index=False
    )
)


print(
    "\nMean edge agreement rate：",
    window_comparison[
        "edge_agreement_rate"
    ]
    .mean()
)


print(
    "Mean edge Jaccard：",
    window_comparison[
        "edge_jaccard"
    ]
    .mean()
)


print(
    "\nPersistent core:"
)

print(
    "GLasso =",
    len(
        g_core_set
    )
)

print(
    "Density-matched Partial =",
    len(
        p_core_set
    )
)

print(
    "Common =",
    len(
        common_core
    )
)

print(
    "Core Jaccard =",
    core_jaccard
)


print(
    "\nEdge Persistence Spearman correlation =",
    persistence_spearman
)


# ============================================================
# 20. 图1：每个Window的方法Edge Agreement
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    window_comparison[
        "network_date"
    ],
    window_comparison[
        "edge_agreement_rate"
    ],
    marker="o"
)


ax.set_xlabel(
    "Network Date"
)

ax.set_ylabel(
    "Edge Agreement Rate"
)

ax.set_ylim(
    0,
    1.05
)


ax.set_title(
    "Graphical Lasso vs Density-matched Partial"
)


ax.grid(
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "density_matched_edge_agreement.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 21. 图2：Same/Cross Mean Persistence
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


x = np.arange(
    len(
        method_summary
    )
)


width = 0.35


ax.bar(
    x
    -
    width
    /
    2,

    method_summary[
        "same_mean_persistence"
    ],

    width,

    label="Same industry"
)


ax.bar(
    x
    +
    width
    /
    2,

    method_summary[
        "cross_mean_persistence"
    ],

    width,

    label="Cross industry"
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    method_summary[
        "method"
    ]
)


ax.set_ylabel(
    "Mean Edge Persistence"
)


ax.set_title(
    "Same vs Cross-industry Persistence Across Methods"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "density_matched_same_cross_persistence.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 22. 图3：Same/Cross State-change Rate
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


ax.bar(
    x
    -
    width
    /
    2,

    method_summary[
        "same_state_change_rate"
    ],

    width,

    label="Same industry"
)


ax.bar(
    x
    +
    width
    /
    2,

    method_summary[
        "cross_state_change_rate"
    ],

    width,

    label="Cross industry"
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    method_summary[
        "method"
    ]
)


ax.set_ylabel(
    "Normalized State-change Rate"
)


ax.set_title(
    "Same vs Cross-industry Dynamic Changes Across Methods"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "density_matched_state_change_rate.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 23. 完成
# ============================================================

print(
    "\n======================================"
)

print(
    "Stage 3完成"
)

print(
    "======================================"
)


for path in [
    MATCHED_PARTIAL_FILE,
    WINDOW_COMPARISON_FILE,
    EDGE_PERSISTENCE_FILE,
    METHOD_SUMMARY_FILE,
    CORE_COMPARISON_FILE
]:

    print(
        path
    )