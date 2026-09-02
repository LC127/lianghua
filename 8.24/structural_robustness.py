from __future__ import annotations

from pathlib import Path
import re
from itertools import combinations

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


# ============================================================
# 2. 输入文件
# ============================================================

OVERLAP_SUMMARY_FILE = (
    PROCESSED_DIR
    / "overlap_control_window_summary.csv"
)

OVERLAP_PERSISTENCE_FILE = (
    PROCESSED_DIR
    / "overlap_control_edge_persistence.csv"
)

METHOD_SUMMARY_FILE = (
    PROCESSED_DIR
    / "density_matched_method_summary.csv"
)

METHOD_PERSISTENCE_FILE = (
    PROCESSED_DIR
    / "density_matched_method_edge_persistence.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ============================================================
# 3. 输出文件
# ============================================================

STRUCTURE_SUMMARY_FILE = (
    PROCESSED_DIR
    / "structural_robustness_summary.csv"
)

EDGE_CONSENSUS_FILE = (
    PROCESSED_DIR
    / "structural_robustness_edge_consensus.csv"
)

CORE_OVERLAP_FILE = (
    PROCESSED_DIR
    / "structural_robustness_core_overlap.csv"
)

EVIDENCE_MATRIX_FILE = (
    PROCESSED_DIR
    / "structural_robustness_evidence_matrix.csv"
)


# ============================================================
# 4. 参数
# ============================================================

PERSISTENCE_THRESHOLD = 0.80


# ============================================================
# 5. 股票代码函数
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


def convert_bool(series):

    if series.dtype == bool:
        return series

    result = (
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

    if result.isna().any():

        raise ValueError(
            "存在无法识别的Boolean字段。"
        )

    return result.astype(bool)


# ============================================================
# 6. Metadata
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
        "stock_info.csv缺少代码、名称或行业列。"
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


# ============================================================
# 7. Stage 1：Fixed-overlap GLasso summary
# ============================================================

overlap_summary = pd.read_csv(
    OVERLAP_SUMMARY_FILE
)


fixed_summary = (
    overlap_summary[
        overlap_summary[
            "design"
        ]
        ==
        "fixed_overlap"
    ]
    .copy()
)


expected_windows = {
    126,
    252,
    504
}


actual_windows = set(
    fixed_summary[
        "window_size"
    ]
    .astype(int)
)


if actual_windows != expected_windows:

    raise ValueError(
        "Fixed-overlap结果没有同时包含W=126,252,504。"
    )


# ============================================================
# 8. Stage 3：Method summary
# ============================================================

method_summary = pd.read_csv(
    METHOD_SUMMARY_FILE
)


partial_method_name = (
    "Density-matched Partial"
)


partial_summary = (
    method_summary[
        method_summary[
            "method"
        ]
        ==
        partial_method_name
    ]
    .copy()
)


if len(
    partial_summary
) != 1:

    raise ValueError(
        "无法唯一识别Density-matched Partial结果。"
    )


# ============================================================
# 9. 构造统一Structural Summary
# ============================================================

summary_rows = []


# ------------------------------------------------------------
# GLasso：三个Fixed-overlap尺度
# ------------------------------------------------------------

for _, row in fixed_summary.iterrows():

    window_size = int(
        row[
            "window_size"
        ]
    )


    same_persistence = float(
        row[
            "same_mean_persistence"
        ]
    )

    cross_persistence = float(
        row[
            "cross_mean_persistence"
        ]
    )


    same_persistent_rate = float(
        row[
            "same_persistent_rate_08"
        ]
    )

    cross_persistent_rate = float(
        row[
            "cross_persistent_rate_08"
        ]
    )


    same_change_rate = float(
        row[
            "same_state_change_rate"
        ]
    )

    cross_change_rate = float(
        row[
            "cross_state_change_rate"
        ]
    )


    summary_rows.append(
        {
            "setting":
                f"GLasso_W{window_size}",

            "method":
                "Graphical Lasso",

            "window_size":
                window_size,

            "step":
                int(
                    row[
                        "step"
                    ]
                ),

            "overlap_ratio":
                float(
                    row[
                        "overlap_ratio"
                    ]
                ),

            "same_mean_persistence":
                same_persistence,

            "cross_mean_persistence":
                cross_persistence,

            "persistence_gap_same_minus_cross":
                (
                    same_persistence
                    -
                    cross_persistence
                ),

            "persistence_ratio_same_to_cross":
                (
                    same_persistence
                    /
                    cross_persistence
                    if cross_persistence > 0
                    else np.nan
                ),

            "same_persistent_rate_08":
                same_persistent_rate,

            "cross_persistent_rate_08":
                cross_persistent_rate,

            "persistent_rate_gap_same_minus_cross":
                (
                    same_persistent_rate
                    -
                    cross_persistent_rate
                ),

            "persistent_rate_ratio_same_to_cross":
                (
                    same_persistent_rate
                    /
                    cross_persistent_rate
                    if cross_persistent_rate > 0
                    else np.nan
                ),

            "same_state_change_rate":
                same_change_rate,

            "cross_state_change_rate":
                cross_change_rate,

            "state_change_gap_cross_minus_same":
                (
                    cross_change_rate
                    -
                    same_change_rate
                ),

            "state_change_ratio_cross_to_same":
                (
                    cross_change_rate
                    /
                    same_change_rate
                    if same_change_rate > 0
                    else np.nan
                ),

            "same_persistence_higher":
                (
                    same_persistence
                    >
                    cross_persistence
                ),

            "same_persistent_rate_higher":
                (
                    same_persistent_rate
                    >
                    cross_persistent_rate
                ),

            "cross_state_change_higher":
                (
                    cross_change_rate
                    >
                    same_change_rate
                )
        }
    )


# ------------------------------------------------------------
# Density-matched Partial：W252
# ------------------------------------------------------------

row = (
    partial_summary
    .iloc[
        0
    ]
)


same_persistence = float(
    row[
        "same_mean_persistence"
    ]
)

cross_persistence = float(
    row[
        "cross_mean_persistence"
    ]
)


same_persistent_rate = float(
    row[
        "same_persistent_rate_08"
    ]
)

cross_persistent_rate = float(
    row[
        "cross_persistent_rate_08"
    ]
)


same_change_rate = float(
    row[
        "same_state_change_rate"
    ]
)

cross_change_rate = float(
    row[
        "cross_state_change_rate"
    ]
)


summary_rows.append(
    {
        "setting":
            "Partial_W252",

        "method":
            "Density-matched Partial",

        "window_size":
            252,

        "step":
            20,

        "overlap_ratio":
            1
            -
            20
            /
            252,

        "same_mean_persistence":
            same_persistence,

        "cross_mean_persistence":
            cross_persistence,

        "persistence_gap_same_minus_cross":
            (
                same_persistence
                -
                cross_persistence
            ),

        "persistence_ratio_same_to_cross":
            (
                same_persistence
                /
                cross_persistence
                if cross_persistence > 0
                else np.nan
            ),

        "same_persistent_rate_08":
            same_persistent_rate,

        "cross_persistent_rate_08":
            cross_persistent_rate,

        "persistent_rate_gap_same_minus_cross":
            (
                same_persistent_rate
                -
                cross_persistent_rate
            ),

        "persistent_rate_ratio_same_to_cross":
            (
                same_persistent_rate
                /
                cross_persistent_rate
                if cross_persistent_rate > 0
                else np.nan
            ),

        "same_state_change_rate":
            same_change_rate,

        "cross_state_change_rate":
            cross_change_rate,

        "state_change_gap_cross_minus_same":
            (
                cross_change_rate
                -
                same_change_rate
            ),

        "state_change_ratio_cross_to_same":
            (
                cross_change_rate
                /
                same_change_rate
                if same_change_rate > 0
                else np.nan
            ),

        "same_persistence_higher":
            (
                same_persistence
                >
                cross_persistence
            ),

        "same_persistent_rate_higher":
            (
                same_persistent_rate
                >
                cross_persistent_rate
            ),

        "cross_state_change_higher":
            (
                cross_change_rate
                >
                same_change_rate
            )
    }
)


structural_summary = pd.DataFrame(
    summary_rows
)


structural_summary.to_csv(
    STRUCTURE_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 读取Edge Persistence数据
# ============================================================

overlap_persistence = pd.read_csv(
    OVERLAP_PERSISTENCE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


method_persistence = pd.read_csv(
    METHOD_PERSISTENCE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


# ============================================================
# 11. 标准化股票对
# ============================================================

def prepare_pair_data(df):

    df = df.copy()

    pairs = df.apply(
        lambda x:
            canonical_pair(
                x[
                    "stock_1"
                ],
                x[
                    "stock_2"
                ]
            ),
        axis=1
    )


    df[
        "stock_1"
    ] = [
        x[0]
        for x in pairs
    ]

    df[
        "stock_2"
    ] = [
        x[1]
        for x in pairs
    ]


    # 重新根据stock_info映射Metadata
    df[
        "name_1"
    ] = (
        df[
            "stock_1"
        ]
        .map(
            name_map
        )
    )

    df[
        "name_2"
    ] = (
        df[
            "stock_2"
        ]
        .map(
            name_map
        )
    )

    df[
        "industry_1"
    ] = (
        df[
            "stock_1"
        ]
        .map(
            industry_map
        )
    )

    df[
        "industry_2"
    ] = (
        df[
            "stock_2"
        ]
        .map(
            industry_map
        )
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


    return df


overlap_persistence = prepare_pair_data(
    overlap_persistence
)


method_persistence = prepare_pair_data(
    method_persistence
)


# ============================================================
# 12. 提取四个设定的Persistence
# ============================================================

persistence_tables = {}


for W in [
    126,
    252,
    504
]:

    temp = (
        overlap_persistence[
            (
                overlap_persistence[
                    "design"
                ]
                ==
                "fixed_overlap"
            )
            &
            (
                overlap_persistence[
                    "window_size"
                ]
                ==
                W
            )
        ]
        [
            [
                "stock_1",
                "stock_2",
                "name_1",
                "name_2",
                "industry_1",
                "industry_2",
                "same_industry",
                "persistence"
            ]
        ]
        .copy()
    )


    temp = temp.rename(
        columns={
            "persistence":
                f"p_GLasso_W{W}"
        }
    )


    persistence_tables[
        f"GLasso_W{W}"
    ] = temp


# ------------------------------------------------------------
# Partial W252
# ------------------------------------------------------------

partial_p = (
    method_persistence[
        method_persistence[
            "method"
        ]
        ==
        partial_method_name
    ]
    [
        [
            "stock_1",
            "stock_2",
            "name_1",
            "name_2",
            "industry_1",
            "industry_2",
            "same_industry",
            "persistence"
        ]
    ]
    .copy()
)


partial_p = partial_p.rename(
    columns={
        "persistence":
            "p_Partial_W252"
    }
)


persistence_tables[
    "Partial_W252"
] = (
    partial_p
)


# ============================================================
# 13. 合并成统一Edge Consensus表
# ============================================================

base = (
    persistence_tables[
        "GLasso_W126"
    ]
    .copy()
)


for setting in [
    "GLasso_W252",
    "GLasso_W504",
    "Partial_W252"
]:

    temp = (
        persistence_tables[
            setting
        ]
        .drop(
            columns=[
                "name_1",
                "name_2",
                "industry_1",
                "industry_2",
                "same_industry"
            ]
        )
    )


    base = base.merge(
        temp,
        on=[
            "stock_1",
            "stock_2"
        ],
        how="inner"
    )


p_cols = [
    "p_GLasso_W126",
    "p_GLasso_W252",
    "p_GLasso_W504",
    "p_Partial_W252"
]


# ============================================================
# 14. Consensus指标
# ============================================================

base[
    "mean_persistence_all_settings"
] = (
    base[
        p_cols
    ]
    .mean(
        axis=1
    )
)


base[
    "min_persistence_all_settings"
] = (
    base[
        p_cols
    ]
    .min(
        axis=1
    )
)


base[
    "max_persistence_all_settings"
] = (
    base[
        p_cols
    ]
    .max(
        axis=1
    )
)


base[
    "persistence_range_all_settings"
] = (
    base[
        "max_persistence_all_settings"
    ]
    -
    base[
        "min_persistence_all_settings"
    ]
)


# 每个设定是否Persistent
for col in p_cols:

    base[
        f"{col}_persistent08"
    ] = (
        base[
            col
        ]
        >=
        PERSISTENCE_THRESHOLD
    )


persistent_cols = [
    f"{col}_persistent08"
    for col in p_cols
]


base[
    "n_settings_persistent08"
] = (
    base[
        persistent_cols
    ]
    .sum(
        axis=1
    )
)


# ------------------------------------------------------------
# 三种GLasso尺度全部Persistent
# ------------------------------------------------------------

base[
    "glasso_multiscale_persistent"
] = (
    (
        base[
            "p_GLasso_W126"
        ]
        >=
        PERSISTENCE_THRESHOLD
    )
    &
    (
        base[
            "p_GLasso_W252"
        ]
        >=
        PERSISTENCE_THRESHOLD
    )
    &
    (
        base[
            "p_GLasso_W504"
        ]
        >=
        PERSISTENCE_THRESHOLD
    )
)


# ------------------------------------------------------------
# W252跨方法Persistent
# ------------------------------------------------------------

base[
    "cross_method_W252_persistent"
] = (
    (
        base[
            "p_GLasso_W252"
        ]
        >=
        PERSISTENCE_THRESHOLD
    )
    &
    (
        base[
            "p_Partial_W252"
        ]
        >=
        PERSISTENCE_THRESHOLD
    )
)


# ------------------------------------------------------------
# 最严格定义：
# 三个GLasso尺度 + Partial全部Persistent
# ------------------------------------------------------------

base[
    "full_structural_consensus_core"
] = (
    base[
        p_cols
    ]
    .ge(
        PERSISTENCE_THRESHOLD
    )
    .all(
        axis=1
    )
)


# ------------------------------------------------------------
# 所有设定都P=1
# ------------------------------------------------------------

base[
    "always_in_all_settings"
] = (
    base[
        p_cols
    ]
    .apply(
        lambda row:
            np.allclose(
                row.to_numpy(
                    dtype=float
                ),
                1.0
            ),
        axis=1
    )
)


# ------------------------------------------------------------
# 简单边层级
# ------------------------------------------------------------

def classify_edge(row):

    n = int(
        row[
            "n_settings_persistent08"
        ]
    )

    if row[
        "always_in_all_settings"
    ]:

        return "Universal always core"

    if row[
        "full_structural_consensus_core"
    ]:

        return "Universal persistent core"

    if n >= 3:

        return "Broadly robust edge"

    if n >= 1:

        return "Setting-sensitive edge"

    return "Nonpersistent edge"


base[
    "structural_robustness_class"
] = (
    base.apply(
        classify_edge,
        axis=1
    )
)


base = (
    base
    .sort_values(
        [
            "full_structural_consensus_core",
            "mean_persistence_all_settings",
            "min_persistence_all_settings"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )
)


base.to_csv(
    EDGE_CONSENSUS_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. Persistent Core Sets
# ============================================================

core_sets = {}


setting_to_column = {
    "GLasso_W126":
        "p_GLasso_W126",

    "GLasso_W252":
        "p_GLasso_W252",

    "GLasso_W504":
        "p_GLasso_W504",

    "Partial_W252":
        "p_Partial_W252"
}


for setting, col in setting_to_column.items():

    core = (
        base[
            base[
                col
            ]
            >=
            PERSISTENCE_THRESHOLD
        ]
    )


    core_sets[
        setting
    ] = set(
        zip(
            core[
                "stock_1"
            ],
            core[
                "stock_2"
            ]
        )
    )


# ============================================================
# 16. 两两Core Overlap
# ============================================================

core_overlap_rows = []


for setting_a, setting_b in combinations(
    core_sets.keys(),
    2
):

    set_a = (
        core_sets[
            setting_a
        ]
    )

    set_b = (
        core_sets[
            setting_b
        ]
    )


    common = (
        set_a
        &
        set_b
    )


    union = (
        set_a
        |
        set_b
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


    containment_a_in_b = (
        len(
            common
        )
        /
        len(
            set_a
        )
        if len(
            set_a
        ) > 0
        else np.nan
    )


    containment_b_in_a = (
        len(
            common
        )
        /
        len(
            set_b
        )
        if len(
            set_b
        ) > 0
        else np.nan
    )


    core_overlap_rows.append(
        {
            "setting_a":
                setting_a,

            "setting_b":
                setting_b,

            "core_size_a":
                len(
                    set_a
                ),

            "core_size_b":
                len(
                    set_b
                ),

            "common_core_edges":
                len(
                    common
                ),

            "union_core_edges":
                len(
                    union
                ),

            "core_jaccard":
                jaccard,

            "containment_a_in_b":
                containment_a_in_b,

            "containment_b_in_a":
                containment_b_in_a
        }
    )


core_overlap_df = pd.DataFrame(
    core_overlap_rows
)


core_overlap_df.to_csv(
    CORE_OVERLAP_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. Evidence Matrix
#
# 注意：
# 这不是统计显著性检验。
# 只是描述性“方向一致性”矩阵。
# ============================================================

evidence_rows = []


for _, row in structural_summary.iterrows():

    evidence_rows.append(
        {
            "setting":
                row[
                    "setting"
                ],

            "method":
                row[
                    "method"
                ],

            "window_size":
                row[
                    "window_size"
                ],

            "E1_same_persistence_gt_cross":
                bool(
                    row[
                        "same_persistence_higher"
                    ]
                ),

            "E2_same_persistent_rate_gt_cross":
                bool(
                    row[
                        "same_persistent_rate_higher"
                    ]
                ),

            "E3_cross_change_rate_gt_same":
                bool(
                    row[
                        "cross_state_change_higher"
                    ]
                )
        }
    )


evidence_matrix = pd.DataFrame(
    evidence_rows
)


evidence_cols = [
    "E1_same_persistence_gt_cross",
    "E2_same_persistent_rate_gt_cross",
    "E3_cross_change_rate_gt_same"
]


evidence_matrix[
    "n_supported_structural_patterns"
] = (
    evidence_matrix[
        evidence_cols
    ]
    .sum(
        axis=1
    )
)


evidence_matrix[
    "all_three_supported"
] = (
    evidence_matrix[
        evidence_cols
    ]
    .all(
        axis=1
    )
)


evidence_matrix.to_csv(
    EVIDENCE_MATRIX_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. 屏幕输出
# ============================================================

print(
    "\n============================================"
)

print(
    "Stage 4: Structural Robustness"
)

print(
    "============================================"
)


print(
    "\n--- Structural Summary ---"
)


print(
    structural_summary[
        [
            "setting",
            "same_mean_persistence",
            "cross_mean_persistence",
            "persistence_ratio_same_to_cross",
            "same_persistent_rate_08",
            "cross_persistent_rate_08",
            "same_state_change_rate",
            "cross_state_change_rate",
            "state_change_ratio_cross_to_same"
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\n--- Evidence Matrix ---"
)


print(
    evidence_matrix.to_string(
        index=False
    )
)


# ============================================================
# 19. Consensus Core Summary
# ============================================================

n_multiscale = int(
    base[
        "glasso_multiscale_persistent"
    ]
    .sum()
)


n_cross_method = int(
    base[
        "cross_method_W252_persistent"
    ]
    .sum()
)


n_full_consensus = int(
    base[
        "full_structural_consensus_core"
    ]
    .sum()
)


n_universal_always = int(
    base[
        "always_in_all_settings"
    ]
    .sum()
)


print(
    "\n--- Edge-level Consensus ---"
)


print(
    "GLasso multi-scale persistent edges：",
    n_multiscale
)


print(
    "W252 cross-method persistent edges：",
    n_cross_method
)


print(
    "Full structural consensus core：",
    n_full_consensus
)


print(
    "Always in all settings：",
    n_universal_always
)


# ============================================================
# 20. Consensus Core行业组成
# ============================================================

consensus_core = (
    base[
        base[
            "full_structural_consensus_core"
        ]
    ]
)


if len(
    consensus_core
) > 0:

    n_same_consensus = int(
        consensus_core[
            "same_industry"
        ]
        .sum()
    )

    n_cross_consensus = (
        len(
            consensus_core
        )
        -
        n_same_consensus
    )


    print(
        "\nFull consensus core:"
    )

    print(
        "Total =",
        len(
            consensus_core
        )
    )

    print(
        "Same industry =",
        n_same_consensus
    )

    print(
        "Cross industry =",
        n_cross_consensus
    )

    print(
        "Same-industry share =",
        n_same_consensus
        /
        len(
            consensus_core
        )
    )


# ============================================================
# 21. 图1：Same vs Cross Persistence
# ============================================================

plot_df = (
    structural_summary
    .copy()
)


x = np.arange(
    len(
        plot_df
    )
)


width = 0.35


fig, ax = plt.subplots(
    figsize=(
        10,
        6
    )
)


ax.bar(
    x
    -
    width
    /
    2,

    plot_df[
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

    plot_df[
        "cross_mean_persistence"
    ],

    width,

    label="Cross industry"
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    plot_df[
        "setting"
    ],
    rotation=20
)


ax.set_ylabel(
    "Mean Edge Persistence"
)


ax.set_ylim(
    0,
    1.05
)


ax.set_title(
    "Structural Robustness: Same vs Cross-industry Persistence"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "structural_robustness_persistence.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 22. 图2：State-change Rate
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        6
    )
)


ax.bar(
    x
    -
    width
    /
    2,

    plot_df[
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

    plot_df[
        "cross_state_change_rate"
    ],

    width,

    label="Cross industry"
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    plot_df[
        "setting"
    ],
    rotation=20
)


ax.set_ylabel(
    "Normalized State-change Rate"
)


ax.set_title(
    "Structural Robustness: Same vs Cross-industry Dynamics"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "structural_robustness_state_change.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 23. 图3：Same/Cross Persistence Ratio
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        6
    )
)


ax.bar(
    plot_df[
        "setting"
    ],
    plot_df[
        "persistence_ratio_same_to_cross"
    ]
)


ax.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1
)


ax.set_ylabel(
    "Same / Cross Mean Persistence"
)


ax.set_title(
    "Structural Robustness of Industry Persistence Gap"
)


ax.tick_params(
    axis="x",
    rotation=20
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "structural_robustness_persistence_ratio.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 24. 完成
# ============================================================

print(
    "\n============================================"
)

print(
    "Stage 4完成"
)

print(
    "============================================"
)


for path in [
    STRUCTURE_SUMMARY_FILE,
    EDGE_CONSENSUS_FILE,
    CORE_OVERLAP_FILE,
    EVIDENCE_MATRIX_FILE
]:

    print(
        path
    )