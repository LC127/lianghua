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


RETURNS_FILE = (
    PROCESSED_DIR
    / "stock_returns.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ============================================================
# 2. 输出文件
# ============================================================

# 最重要：
# 每个window × 每条edge的完整partial correlation
EDGE_HISTORY_FILE = (
    PROCESSED_DIR
    / "rolling_partial_edge_history.csv"
)

# 每个window一行
NETWORK_SUMMARY_FILE = (
    PROCESSED_DIR
    / "rolling_partial_network_summary.csv"
)

# 数值稳定性诊断
DIAGNOSTICS_FILE = (
    PROCESSED_DIR
    / "rolling_partial_diagnostics.csv"
)

# 可选：固定0.20阈值下的边
THRESHOLD_EDGES_FILE = (
    PROCESSED_DIR
    / "rolling_partial_threshold_edges.csv"
)


# ============================================================
# 3. Rolling参数
# ============================================================

WINDOW_SIZE = 252

STEP = 20


# ============================================================
# 4. 描述性Partial Network阈值
#
# 这里只用于Stage 2初步观察。
# Stage 3与GLasso正式比较时，
# 不使用这个阈值，而使用density matching。
# ============================================================

PARTIAL_THRESHOLD = 0.20


# ============================================================
# 5. 数值检查阈值
# ============================================================

MIN_EIGENVALUE_TOL = 1e-10

CONDITION_WARNING_THRESHOLD = 1e5


# ============================================================
# 6. 股票代码标准化
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


# ============================================================
# 7. 读取Stock Info
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
        "stock_info.csv中无法识别"
        "股票代码、名称或行业字段。"
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
# 8. 读取收益率
# ============================================================

returns_raw = pd.read_csv(
    RETURNS_FILE
)


date_candidates = [
    "日期",
    "date",
    "trade_date",
    "datetime",
    "Date"
]


date_col = next(
    (
        c
        for c in date_candidates
        if c in returns_raw.columns
    ),
    None
)


if date_col is None:

    raise ValueError(
        "stock_returns.csv中找不到日期列。"
    )


returns_raw[
    date_col
] = pd.to_datetime(
    returns_raw[
        date_col
    ]
)


returns_raw = (
    returns_raw
    .sort_values(
        date_col
    )
    .set_index(
        date_col
    )
)


# ============================================================
# 9. 收益率列名映射成六位股票代码
# ============================================================

rename_dict = {}


for col in returns_raw.columns:

    normalized = normalize_code(
        col
    )

    if normalized in name_map:

        rename_dict[
            col
        ] = normalized


returns_df = returns_raw.rename(
    columns=rename_dict
)


stock_codes = [
    code
    for code in metadata[
        "stock_code"
    ]
    if code in returns_df.columns
]


if len(stock_codes) < 2:

    raise ValueError(
        "收益率文件与stock_info中的"
        "股票代码无法正常匹配。"
    )


returns_df = (
    returns_df[
        stock_codes
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .dropna()
)


N_STOCKS = len(
    stock_codes
)


N_PAIRS = (
    N_STOCKS
    *
    (
        N_STOCKS - 1
    )
    //
    2
)


print(
    "股票数量：",
    N_STOCKS
)

print(
    "候选股票对：",
    N_PAIRS
)

print(
    "收益率观测数：",
    len(
        returns_df
    )
)


# ============================================================
# 10. 检查样本量
# ============================================================

if len(
    returns_df
) < WINDOW_SIZE:

    raise ValueError(
        f"有效收益率观测不足{WINDOW_SIZE}。"
    )


# ============================================================
# 11. Precision -> Partial Correlation
# ============================================================

def precision_to_partial(
    precision: np.ndarray
) -> np.ndarray:

    diagonal = np.diag(
        precision
    )


    if np.any(
        diagonal <= 0
    ):

        raise ValueError(
            "Precision Matrix出现非正对角元素。"
        )


    denominator = np.sqrt(
        np.outer(
            diagonal,
            diagonal
        )
    )


    partial = (
        -
        precision
        /
        denominator
    )


    np.fill_diagonal(
        partial,
        1.0
    )


    # 避免极小数值误差超出[-1,1]
    partial = np.clip(
        partial,
        -1.0,
        1.0
    )


    return partial


# ============================================================
# 12. Rolling Partial Correlation
# ============================================================

edge_rows = []

network_rows = []

diagnostic_rows = []


# 第一个完整252日window：
# index = 251
endpoint_indices = range(
    WINDOW_SIZE - 1,
    len(
        returns_df
    ),
    STEP
)


for window_id, end_idx in enumerate(
    endpoint_indices,
    start=1
):

    start_idx = (
        end_idx
        -
        WINDOW_SIZE
        +
        1
    )


    window = (
        returns_df
        .iloc[
            start_idx:
            end_idx + 1
        ]
        .copy()
    )


    # ========================================================
    # 13. Window内部标准化
    # ========================================================

    means = window.mean(
        axis=0
    )


    sds = window.std(
        axis=0,
        ddof=1
    )


    if (
        sds <= 1e-12
    ).any():

        bad_codes = (
            sds[
                sds <= 1e-12
            ]
            .index
            .tolist()
        )

        raise ValueError(
            f"Window {window_id}中以下股票"
            f"标准差接近0：{bad_codes}"
        )


    Z_df = (
        window
        -
        means
    ) / sds


    Z = Z_df.to_numpy(
        dtype=float
    )


    # ========================================================
    # 14. Sample Covariance
    #
    # 因为已经标准化，
    # S大致也是sample correlation matrix。
    # ========================================================

    covariance = np.cov(
        Z,
        rowvar=False,
        ddof=1
    )


    # 强制数值对称
    covariance = (
        covariance
        +
        covariance.T
    ) / 2.0


    # ========================================================
    # 15. 数值稳定性诊断
    # ========================================================

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )


    min_eigenvalue = float(
        eigenvalues.min()
    )


    max_eigenvalue = float(
        eigenvalues.max()
    )


    if (
        min_eigenvalue
        <=
        MIN_EIGENVALUE_TOL
    ):

        raise np.linalg.LinAlgError(
            f"Window {window_id} "
            f"({window.index[-1].date()}) "
            "的样本协方差矩阵接近奇异；"
            f"minimum eigenvalue = "
            f"{min_eigenvalue:.6e}"
        )


    condition_number = (
        max_eigenvalue
        /
        min_eigenvalue
    )


    # ========================================================
    # 16. Precision Matrix
    #
    # 使用solve而不是直接inv，
    # 数值上通常更稳定。
    #
    # solve(S, I) 等价于 S^{-1}
    # ========================================================

    precision = np.linalg.solve(
        covariance,
        np.eye(
            N_STOCKS
        )
    )


    # 消除浮点误差造成的小非对称
    precision = (
        precision
        +
        precision.T
    ) / 2.0


    # ========================================================
    # 17. 检查反演误差
    # ========================================================

    identity_error = np.max(
        np.abs(
            covariance
            @
            precision
            -
            np.eye(
                N_STOCKS
            )
        )
    )


    # ========================================================
    # 18. Partial Correlation Matrix
    # ========================================================

    partial = precision_to_partial(
        precision
    )


    # ========================================================
    # 19. 先把105条边存进临时列表
    # ========================================================

    current_edges = []


    for i in range(
        N_STOCKS
    ):

        for j in range(
            i + 1,
            N_STOCKS
        ):

            stock_1 = stock_codes[
                i
            ]

            stock_2 = stock_codes[
                j
            ]


            partial_ij = float(
                partial[
                    i,
                    j
                ]
            )


            abs_partial_ij = abs(
                partial_ij
            )


            same_industry = (
                industry_map[
                    stock_1
                ]
                ==
                industry_map[
                    stock_2
                ]
            )


            current_edges.append(
                {
                    "window_id":
                        window_id,

                    "window_start":
                        window.index[
                            0
                        ],

                    "network_date":
                        window.index[
                            -1
                        ],

                    "window_size":
                        WINDOW_SIZE,

                    "step":
                        STEP,

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
                        same_industry,

                    "partial_correlation":
                        partial_ij,

                    "abs_partial_correlation":
                        abs_partial_ij,

                    # ----------------------------------------
                    # 固定阈值仅用于描述性网络
                    # ----------------------------------------

                    "selected_threshold_020":
                        (
                            abs_partial_ij
                            >=
                            PARTIAL_THRESHOLD
                        )
                }
            )


    # ========================================================
    # 20. 对105条Partial按绝对值排名
    #
    # 这一步是为Stage 3的density matching准备。
    # rank=1表示当前window最强的partial edge。
    # ========================================================

    current_df = pd.DataFrame(
        current_edges
    )


    current_df[
        "abs_partial_rank"
    ] = (
        current_df[
            "abs_partial_correlation"
        ]
        .rank(
            method="first",
            ascending=False
        )
        .astype(int)
    )


    edge_rows.extend(
        current_df.to_dict(
            orient="records"
        )
    )


    # ========================================================
    # 21. Window-level描述性汇总
    # ========================================================

    selected_threshold = (
        current_df[
            current_df[
                "selected_threshold_020"
            ]
        ]
    )


    threshold_edge_count = len(
        selected_threshold
    )


    threshold_same_edges = int(
        selected_threshold[
            "same_industry"
        ]
        .sum()
    )


    threshold_cross_edges = (
        threshold_edge_count
        -
        threshold_same_edges
    )


    network_rows.append(
        {
            "window_id":
                window_id,

            "window_start":
                window.index[
                    0
                ],

            "network_date":
                window.index[
                    -1
                ],

            "window_size":
                WINDOW_SIZE,

            "step":
                STEP,

            # -----------------------------------------------
            # 所有105条边的连续Partial指标
            # -----------------------------------------------

            "mean_abs_partial_all_pairs":
                current_df[
                    "abs_partial_correlation"
                ]
                .mean(),

            "median_abs_partial_all_pairs":
                current_df[
                    "abs_partial_correlation"
                ]
                .median(),

            "max_abs_partial":
                current_df[
                    "abs_partial_correlation"
                ]
                .max(),

            # -----------------------------------------------
            # Same/Cross所有候选pair的平均Partial
            # -----------------------------------------------

            "mean_abs_partial_same_pairs":
                current_df.loc[
                    current_df[
                        "same_industry"
                    ],
                    "abs_partial_correlation"
                ]
                .mean(),

            "mean_abs_partial_cross_pairs":
                current_df.loc[
                    ~current_df[
                        "same_industry"
                    ],
                    "abs_partial_correlation"
                ]
                .mean(),

            # -----------------------------------------------
            # 固定阈值0.20的描述性网络
            # -----------------------------------------------

            "threshold":
                PARTIAL_THRESHOLD,

            "threshold_edge_count":
                threshold_edge_count,

            "threshold_density":
                threshold_edge_count
                /
                N_PAIRS,

            "threshold_same_edges":
                threshold_same_edges,

            "threshold_cross_edges":
                threshold_cross_edges,

            "threshold_same_ratio":
                (
                    threshold_same_edges
                    /
                    threshold_edge_count
                    if threshold_edge_count > 0
                    else np.nan
                )
        }
    )


    # ========================================================
    # 22. Diagnostics
    # ========================================================

    diagnostic_rows.append(
        {
            "window_id":
                window_id,

            "window_start":
                window.index[
                    0
                ],

            "network_date":
                window.index[
                    -1
                ],

            "window_size":
                WINDOW_SIZE,

            "step":
                STEP,

            "min_cov_eigenvalue":
                min_eigenvalue,

            "max_cov_eigenvalue":
                max_eigenvalue,

            "cov_condition_number":
                condition_number,

            "condition_warning":
                (
                    condition_number
                    >=
                    CONDITION_WARNING_THRESHOLD
                ),

            "inverse_identity_max_error":
                identity_error
        }
    )


# ============================================================
# 23. 整理输出
# ============================================================

edge_history_df = pd.DataFrame(
    edge_rows
)


network_summary_df = pd.DataFrame(
    network_rows
)


diagnostics_df = pd.DataFrame(
    diagnostic_rows
)


threshold_edges_df = (
    edge_history_df[
        edge_history_df[
            "selected_threshold_020"
        ]
    ]
    .copy()
)


# ============================================================
# 24. 保存
# ============================================================

edge_history_df.to_csv(
    EDGE_HISTORY_FILE,
    index=False,
    encoding="utf-8-sig"
)


network_summary_df.to_csv(
    NETWORK_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


diagnostics_df.to_csv(
    DIAGNOSTICS_FILE,
    index=False,
    encoding="utf-8-sig"
)


threshold_edges_df.to_csv(
    THRESHOLD_EDGES_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 25. 屏幕输出
# ============================================================

print(
    "\n======================================"
)

print(
    "Rolling Partial Correlation Summary"
)

print(
    "======================================"
)


print(
    "Rolling network数量：",
    len(
        network_summary_df
    )
)


print(
    "\n平均所有Pair |Partial|：",
    network_summary_df[
        "mean_abs_partial_all_pairs"
    ]
    .mean()
)


print(
    "平均Same-pair |Partial|：",
    network_summary_df[
        "mean_abs_partial_same_pairs"
    ]
    .mean()
)


print(
    "平均Cross-pair |Partial|：",
    network_summary_df[
        "mean_abs_partial_cross_pairs"
    ]
    .mean()
)


print(
    "\n固定 |partial| >= 0.20 时平均边数：",
    network_summary_df[
        "threshold_edge_count"
    ]
    .mean()
)


print(
    "固定阈值网络平均Same edges：",
    network_summary_df[
        "threshold_same_edges"
    ]
    .mean()
)


print(
    "固定阈值网络平均Cross edges：",
    network_summary_df[
        "threshold_cross_edges"
    ]
    .mean()
)


print(
    "\n最大Covariance Condition Number：",
    diagnostics_df[
        "cov_condition_number"
    ]
    .max()
)


print(
    "Condition warnings：",
    int(
        diagnostics_df[
            "condition_warning"
        ]
        .sum()
    )
)


print(
    "最大inverse check error：",
    diagnostics_df[
        "inverse_identity_max_error"
    ]
    .max()
)


# ============================================================
# 26. 图1：Same vs Cross平均绝对Partial
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    network_summary_df[
        "network_date"
    ],
    network_summary_df[
        "mean_abs_partial_same_pairs"
    ],
    marker="o",
    label="Same-industry pairs"
)


ax.plot(
    network_summary_df[
        "network_date"
    ],
    network_summary_df[
        "mean_abs_partial_cross_pairs"
    ],
    marker="o",
    label="Cross-industry pairs"
)


ax.set_xlabel(
    "Network Date"
)

ax.set_ylabel(
    "Mean Absolute Partial Correlation"
)

ax.set_title(
    "Rolling Partial Correlation: Same vs Cross Industry"
)

ax.legend()

ax.grid(
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "rolling_partial_same_cross_strength.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 27. 图2：Fixed-threshold Edge Count
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    network_summary_df[
        "network_date"
    ],
    network_summary_df[
        "threshold_edge_count"
    ],
    marker="o"
)


ax.set_xlabel(
    "Network Date"
)

ax.set_ylabel(
    "Number of Edges"
)


ax.set_title(
    "Rolling Partial Network "
    "(Descriptive Threshold = 0.20)"
)


ax.grid(
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "rolling_partial_threshold_edge_count.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 28. 图3：Covariance Condition Number
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    diagnostics_df[
        "network_date"
    ],
    diagnostics_df[
        "cov_condition_number"
    ],
    marker="o"
)


ax.axhline(
    y=CONDITION_WARNING_THRESHOLD,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Network Date"
)

ax.set_ylabel(
    "Condition Number"
)


ax.set_title(
    "Rolling Sample Covariance Condition Number"
)


ax.grid(
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "rolling_partial_condition_number.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 29. 完成
# ============================================================

print(
    "\n======================================"
)

print(
    "Stage 2完成"
)

print(
    "======================================"
)


for path in [
    EDGE_HISTORY_FILE,
    NETWORK_SUMMARY_FILE,
    DIAGNOSTICS_FILE,
    THRESHOLD_EDGES_FILE
]:

    print(
        path
    )