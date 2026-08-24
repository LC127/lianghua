from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib import font_manager
from sklearn.covariance import GraphicalLasso
from sklearn.exceptions import ConvergenceWarning


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
        "警告：没有找到常见中文字体。"
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

RETURN_FILE = (
    PROCESSED_DIR
    / "stock_returns.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)

SELECTION_FILE = (
    PROCESSED_DIR
    / "glasso_1se_selection.csv"
)


# ------------------------------------------------------------
# 输出文件
# ------------------------------------------------------------

ROLLING_SUMMARY_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_network_summary.csv"
)

ROLLING_NODE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_node_metrics.csv"
)

ROLLING_EDGE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_history.csv"
)

ROLLING_DIAGNOSTICS_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_diagnostics.csv"
)


# ============================================================
# 2. Rolling参数
# ============================================================

# 每个窗口约1个交易年
WINDOW = 252

# 每约1个月更新一次
STEP = 20

# 数值上的非零判断
ZERO_TOL = 1e-8


# ============================================================
# 3. Graphical Lasso数值参数
# ============================================================

MAX_ITER = 5000

OUTER_TOL = 1e-4

ENET_TOL = 1e-6


# ============================================================
# 4. 工具函数
# ============================================================

def normalize_code(x):

    return str(
        x
    ).strip().zfill(6)


def get_edge_key(
    a,
    b
):

    return tuple(
        sorted(
            [
                normalize_code(a),
                normalize_code(b)
            ]
        )
    )


# ============================================================
# 5. 读取收益率
# ============================================================

returns = pd.read_csv(
    RETURN_FILE,
    index_col=0,
    parse_dates=True
)


returns.columns = [
    normalize_code(
        x
    )

    for x in returns.columns
]


returns = (
    returns
    .sort_index()
    .dropna(
        axis=0,
        how="any"
    )
)


codes = (
    returns.columns
    .tolist()
)


n_samples = len(
    returns
)

n_stocks = len(
    codes
)


print(
    "收益率数据维度：",
    returns.shape
)


print(
    "样本起止日期：",
    returns.index.min(),
    "->",
    returns.index.max()
)


# ============================================================
# 6. 读取固定的alpha_1SE
# ============================================================

selection_df = pd.read_csv(
    SELECTION_FILE
)


if (
    "alpha_1se"
    not in selection_df.columns
):

    raise ValueError(
        "glasso_1se_selection.csv 中不存在 alpha_1se。"
    )


ALPHA_FIXED = float(
    selection_df[
        "alpha_1se"
    ].iloc[0]
)


print(
    "\n固定 Graphical Lasso alpha：",
    ALPHA_FIXED
)


# ============================================================
# 7. 读取股票名称和行业
# ============================================================

name_map = {}

industry_map = {}


if STOCK_INFO_FILE.exists():

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


    if (
        "name"
        in stock_info.columns
    ):

        name_map = dict(
            zip(
                stock_info[
                    "code"
                ],
                stock_info[
                    "name"
                ]
            )
        )


    if (
        "industry"
        in stock_info.columns
    ):

        industry_map = dict(
            zip(
                stock_info[
                    "code"
                ],
                stock_info[
                    "industry"
                ]
            )
        )


def get_name(
    code
):

    value = name_map.get(
        code,
        code
    )

    if pd.isna(
        value
    ):

        return code

    return str(
        value
    )


def get_industry(
    code
):

    value = industry_map.get(
        code,
        "未知行业"
    )

    if pd.isna(
        value
    ):

        return "未知行业"

    return str(
        value
    )


# ============================================================
# 8. Precision -> Partial Correlation
# ============================================================

def precision_to_partial(
    precision
):

    diagonal = np.sqrt(
        np.diag(
            precision
        )
    )


    partial = (
        -precision
        /
        np.outer(
            diagonal,
            diagonal
        )
    )


    np.fill_diagonal(
        partial,
        1.0
    )


    return partial


# ============================================================
# 9. 检查窗口数量
# ============================================================

if WINDOW > n_samples:

    raise ValueError(
        "WINDOW大于总样本量。"
    )


window_starts = list(
    range(
        0,
        n_samples - WINDOW + 1,
        STEP
    )
)


print(
    "\nWINDOW =",
    WINDOW
)

print(
    "STEP =",
    STEP
)

print(
    "预计Rolling窗口数量 =",
    len(
        window_starts
    )
)


# ============================================================
# 10. 保存结果
# ============================================================

summary_rows = []

node_rows = []

edge_rows = []

diagnostic_rows = []


# ============================================================
# 11. Rolling Graphical Lasso主循环
# ============================================================

for window_id, start in enumerate(
    window_starts,
    start=1
):

    end = (
        start
        +
        WINDOW
    )


    window_returns = (
        returns.iloc[
            start:end
        ]
    )


    start_date = (
        window_returns
        .index[
            0
        ]
    )


    end_date = (
        window_returns
        .index[
            -1
        ]
    )


    # 用窗口结束日代表当前网络时间
    network_date = (
        end_date
    )


    # --------------------------------------------------------
    # 11.1 当前窗口数据
    # --------------------------------------------------------

    X_window = (
        window_returns
        .values
        .astype(float)
    )


    # --------------------------------------------------------
    # 11.2 窗口内部重新标准化
    #
    # mean=0, variance=1
    # --------------------------------------------------------

    mean_window = (
        X_window.mean(
            axis=0
        )
    )


    std_window = (
        X_window.std(
            axis=0,
            ddof=0
        )
    )


    if np.any(
        std_window
        <
        1e-12
    ):

        print(
            f"窗口 {window_id} 存在近零方差变量，跳过。"
        )

        diagnostic_rows.append(
            {
                "window_id":
                    window_id,

                "window_start":
                    start_date,

                "window_end":
                    end_date,

                "network_date":
                    network_date,

                "valid":
                    False,

                "converged":
                    False,

                "n_iter":
                    np.nan,

                "dual_gap":
                    np.nan,

                "reason":
                    "near_zero_std"
            }
        )

        continue


    Z_window = (
        X_window
        -
        mean_window
    ) / std_window


    # --------------------------------------------------------
    # 11.3 Graphical Lasso
    # --------------------------------------------------------

    model = GraphicalLasso(

        alpha=
            ALPHA_FIXED,

        mode=
            "cd",

        tol=
            OUTER_TOL,

        enet_tol=
            ENET_TOL,

        max_iter=
            MAX_ITER,

        assume_centered=
            True
    )


    with warnings.catch_warnings(
        record=True
    ) as caught:

        warnings.simplefilter(
            "always"
        )


        try:

            model.fit(
                Z_window
            )

            fit_error = None

        except Exception as exc:

            fit_error = str(
                exc
            )


    # --------------------------------------------------------
    # 11.4 拟合失败
    # --------------------------------------------------------

    if fit_error is not None:

        print(
            f"窗口 {window_id} 拟合失败：{fit_error}"
        )

        diagnostic_rows.append(
            {
                "window_id":
                    window_id,

                "window_start":
                    start_date,

                "window_end":
                    end_date,

                "network_date":
                    network_date,

                "valid":
                    False,

                "converged":
                    False,

                "n_iter":
                    np.nan,

                "dual_gap":
                    np.nan,

                "reason":
                    fit_error
            }
        )

        continue


    # --------------------------------------------------------
    # 11.5 收敛检查
    # --------------------------------------------------------

    convergence_warning = any(

        issubclass(
            w.category,
            ConvergenceWarning
        )

        for w in caught
    )


    precision = (
        model.precision_
    )


    finite_solution = np.all(
        np.isfinite(
            precision
        )
    )


    if (
        hasattr(
            model,
            "costs_"
        )
        and
        len(
            model.costs_
        ) > 0
    ):

        final_gap = float(
            model.costs_[
                -1
            ][1]
        )

    else:

        final_gap = np.nan


    converged = (
        finite_solution
        and
        not convergence_warning
    )


    diagnostic_rows.append(
        {
            "window_id":
                window_id,

            "window_start":
                start_date,

            "window_end":
                end_date,

            "network_date":
                network_date,

            "valid":
                finite_solution,

            "converged":
                converged,

            "n_iter":
                model.n_iter_,

            "dual_gap":
                final_gap,

            "reason":
                (
                    ""
                    if converged
                    else
                    "ConvergenceWarning"
                )
        }
    )


    if not converged:

        print(
            f"警告：窗口 {window_id} 未严格收敛，"
            "该窗口不进入主分析。"
        )

        continue


    # --------------------------------------------------------
    # 11.6 Precision -> Partial
    # --------------------------------------------------------

    partial = precision_to_partial(
        precision
    )


    # ========================================================
    # 12. 构建当前网络
    # ========================================================

    G = nx.Graph()


    for code in codes:

        G.add_node(

            code,

            name=
                get_name(
                    code
                ),

            industry=
                get_industry(
                    code
                )
        )


    current_edge_count = 0

    current_abs_partial = []

    same_industry_edges = 0


    # ========================================================
    # 13. 保存全部105个股票对
    # ========================================================

    for i in range(
        n_stocks
    ):

        for j in range(
            i + 1,
            n_stocks
        ):

            code_i = (
                codes[i]
            )

            code_j = (
                codes[j]
            )


            omega_ij = float(
                precision[
                    i,
                    j
                ]
            )


            rho_ij = float(
                partial[
                    i,
                    j
                ]
            )


            selected = (
                abs(
                    omega_ij
                )
                >
                ZERO_TOL
            )


            same_industry = (
                get_industry(
                    code_i
                )
                ==
                get_industry(
                    code_j
                )
            )


            # ------------------------------------------------
            # 当前网络有边
            # ------------------------------------------------

            if selected:

                current_edge_count += 1


                current_abs_partial.append(
                    abs(
                        rho_ij
                    )
                )


                if same_industry:

                    same_industry_edges += 1


                G.add_edge(

                    code_i,
                    code_j,

                    precision=
                        omega_ij,

                    partial_correlation=
                        rho_ij,

                    strength=
                        abs(
                            rho_ij
                        )
                )


            # ------------------------------------------------
            # 无论有没有边都保存
            # ------------------------------------------------

            edge_rows.append(
                {
                    "window_id":
                        window_id,

                    "window_start":
                        start_date,

                    "window_end":
                        end_date,

                    "network_date":
                        network_date,

                    "stock_1":
                        code_i,

                    "name_1":
                        get_name(
                            code_i
                        ),

                    "industry_1":
                        get_industry(
                            code_i
                        ),

                    "stock_2":
                        code_j,

                    "name_2":
                        get_name(
                            code_j
                        ),

                    "industry_2":
                        get_industry(
                            code_j
                        ),

                    "precision":
                        omega_ij,

                    "partial_correlation":
                        rho_ij,

                    "abs_partial_correlation":
                        abs(
                            rho_ij
                        ),

                    "selected":
                        selected,

                    "same_industry":
                        same_industry
                }
            )


    # ========================================================
    # 14. 当前网络整体指标
    # ========================================================

    n_edges = (
        G.number_of_edges()
    )


    density = (
        nx.density(
            G
        )
    )


    degree_values = [
        degree

        for _, degree
        in G.degree()
    ]


    mean_degree = (
        np.mean(
            degree_values
        )
    )


    max_degree = (
        np.max(
            degree_values
        )
    )


    n_components = (
        nx.number_connected_components(
            G
        )
    )


    n_isolated = len(
        list(
            nx.isolates(
                G
            )
        )
    )


    mean_abs_partial = (

        np.mean(
            current_abs_partial
        )

        if len(
            current_abs_partial
        ) > 0

        else np.nan
    )


    median_abs_partial = (

        np.median(
            current_abs_partial
        )

        if len(
            current_abs_partial
        ) > 0

        else np.nan
    )


    same_industry_ratio = (

        same_industry_edges
        /
        n_edges

        if n_edges > 0

        else np.nan
    )


    summary_rows.append(
        {
            "window_id":
                window_id,

            "window_start":
                start_date,

            "window_end":
                end_date,

            "network_date":
                network_date,

            "window_size":
                WINDOW,

            "step":
                STEP,

            "alpha":
                ALPHA_FIXED,

            "n_nodes":
                G.number_of_nodes(),

            "n_edges":
                n_edges,

            "density":
                density,

            "mean_degree":
                mean_degree,

            "max_degree":
                max_degree,

            "n_components":
                n_components,

            "n_isolated":
                n_isolated,

            "mean_abs_partial":
                mean_abs_partial,

            "median_abs_partial":
                median_abs_partial,

            "same_industry_edges":
                same_industry_edges,

            "same_industry_edge_ratio":
                same_industry_ratio
        }
    )


    # ========================================================
    # 15. 当前窗口节点指标
    # ========================================================

    degree_dict = dict(
        G.degree()
    )


    strength_dict = {

        code:
            sum(
                abs(
                    data[
                        "partial_correlation"
                    ]
                )

                for _, _, data
                in G.edges(
                    code,
                    data=True
                )
            )

        for code in G.nodes()
    }


    for code in codes:

        node_rows.append(
            {
                "window_id":
                    window_id,

                "window_start":
                    start_date,

                "window_end":
                    end_date,

                "network_date":
                    network_date,

                "code":
                    code,

                "name":
                    get_name(
                        code
                    ),

                "industry":
                    get_industry(
                        code
                    ),

                "degree":
                    degree_dict[
                        code
                    ],

                "strength":
                    strength_dict[
                        code
                    ]
            }
        )


    # ========================================================
    # 16. 屏幕输出
    # ========================================================

    print(
        f"Window {window_id:02d}: "
        f"{start_date.date()} -> {end_date.date()} | "
        f"Edges={n_edges:3d} | "
        f"Density={density:.3f} | "
        f"Mean|partial|={mean_abs_partial:.3f} | "
        f"Components={n_components}"
    )


# ============================================================
# 17. 转DataFrame
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


node_df = pd.DataFrame(
    node_rows
)


edge_df = pd.DataFrame(
    edge_rows
)


diagnostics_df = pd.DataFrame(
    diagnostic_rows
)


# ============================================================
# 18. 日期格式
# ============================================================

date_columns_summary = [
    "window_start",
    "window_end",
    "network_date"
]


for col in date_columns_summary:

    if col in summary_df.columns:

        summary_df[
            col
        ] = pd.to_datetime(
            summary_df[
                col
            ]
        )


for col in [
    "window_start",
    "window_end",
    "network_date"
]:

    if col in node_df.columns:

        node_df[
            col
        ] = pd.to_datetime(
            node_df[
                col
            ]
        )


    if col in edge_df.columns:

        edge_df[
            col
        ] = pd.to_datetime(
            edge_df[
                col
            ]
        )


# ============================================================
# 19. 保存结果
# ============================================================

summary_df.to_csv(
    ROLLING_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


node_df.to_csv(
    ROLLING_NODE_FILE,
    index=False,
    encoding="utf-8-sig"
)


edge_df.to_csv(
    ROLLING_EDGE_FILE,
    index=False,
    encoding="utf-8-sig"
)


diagnostics_df.to_csv(
    ROLLING_DIAGNOSTICS_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. 总体输出
# ============================================================

print(
    "\n======================================"
)

print(
    "Rolling Graphical Lasso完成"
)

print(
    "======================================"
)


print(
    "有效窗口数：",
    len(
        summary_df
    )
)


if len(
    summary_df
) > 0:

    print(
        "\n网络边数："
    )

    print(
        "min =",
        summary_df[
            "n_edges"
        ].min()
    )

    print(
        "mean =",
        summary_df[
            "n_edges"
        ].mean()
    )

    print(
        "max =",
        summary_df[
            "n_edges"
        ].max()
    )


    print(
        "\n网络密度："
    )

    print(
        "min =",
        summary_df[
            "density"
        ].min()
    )

    print(
        "mean =",
        summary_df[
            "density"
        ].mean()
    )

    print(
        "max =",
        summary_df[
            "density"
        ].max()
    )


# ============================================================
# 21. 图1：动态网络Density
# ============================================================

plt.figure(
    figsize=(
        11,
        6
    )
)


plt.plot(

    summary_df[
        "network_date"
    ],

    summary_df[
        "density"
    ],

    marker=
        "o"
)


plt.xlabel(
    "Window end date"
)

plt.ylabel(
    "Network density"
)

plt.title(
    "Rolling Graphical Lasso网络密度"
)


plt.grid(
    alpha=
        0.3
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "rolling_glasso_density.png",

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


# ============================================================
# 22. 图2：Mean Absolute Partial
# ============================================================

plt.figure(
    figsize=(
        11,
        6
    )
)


plt.plot(

    summary_df[
        "network_date"
    ],

    summary_df[
        "mean_abs_partial"
    ],

    marker=
        "o"
)


plt.xlabel(
    "Window end date"
)

plt.ylabel(
    "Mean absolute GLasso partial correlation"
)

plt.title(
    "Rolling Graphical Lasso平均条件关联强度"
)


plt.grid(
    alpha=
        0.3
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "rolling_glasso_mean_abs_partial.png",

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


# ============================================================
# 23. 图3：重点股票Degree变化
# ============================================================

FOCUS_CODES = [
    "600030",  # 中信证券
    "601318",  # 中国平安
    "600519",  # 贵州茅台
    "000858",  # 五粮液
    "600036"   # 招商银行
]


plt.figure(
    figsize=(
        12,
        7
    )
)


for code in FOCUS_CODES:

    temp = node_df[
        node_df[
            "code"
        ]
        ==
        code
    ]


    if temp.empty:

        continue


    label = (
        f"{code} "
        f"{get_name(code)}"
    )


    plt.plot(

        temp[
            "network_date"
        ],

        temp[
            "degree"
        ],

        marker=
            "o",

        label=
            label
    )


plt.xlabel(
    "Window end date"
)

plt.ylabel(
    "Degree"
)

plt.title(
    "重点股票在Rolling GLasso网络中的Degree变化"
)


plt.legend()

plt.grid(
    alpha=
        0.3
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "rolling_glasso_focus_stock_degree.png",

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


# ============================================================
# 24. 完成
# ============================================================

print(
    "\n输出文件："
)


for path in [

    ROLLING_SUMMARY_FILE,

    ROLLING_NODE_FILE,

    ROLLING_EDGE_FILE,

    ROLLING_DIAGNOSTICS_FILE,

    FIGURE_DIR
    / "rolling_glasso_density.png",

    FIGURE_DIR
    / "rolling_glasso_mean_abs_partial.png",

    FIGURE_DIR
    / "rolling_glasso_focus_stock_degree.png"

]:

    print(
        path
    )