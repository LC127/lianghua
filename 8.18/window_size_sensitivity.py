from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.covariance import GraphicalLasso
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# 0. 全局参数
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


# ------------------------------------------------------------
# 输入文件
# ------------------------------------------------------------

RETURNS_FILE = (
    PROCESSED_DIR
    / "stock_returns.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)

# 用于stock_info不存在时建立正确名称/行业映射
ROLLING_EDGE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_history.csv"
)

ALPHA_FILE = (
    PROCESSED_DIR
    / "glasso_1se_selection.csv"
)


# ============================================================
# 1. Window-size Sensitivity设置
# ============================================================

WINDOW_SIZES = [
    126,
    252,
    504
]

# 昨天的基准窗口
BASE_WINDOW = 252

# 保持不变
STEP = 20

# 如果设置为None，就从glasso_1se_selection.csv读取
ALPHA_FIXED = None

# 如果读取alpha失败，也可以直接改成：
# ALPHA_FIXED = 0.216910

ZERO_TOL = 1e-8

GLASSO_TOL = 1e-4

ENET_TOL = 1e-6

MAX_ITER = 5000

MODE = "cd"


# ============================================================
# 2. 输出文件
# ============================================================

NETWORK_SUMMARY_FILE = (
    PROCESSED_DIR
    / "window_size_rolling_network_summary.csv"
)

EDGE_HISTORY_FILE = (
    PROCESSED_DIR
    / "window_size_edge_history.csv"
)

DIAGNOSTICS_FILE = (
    PROCESSED_DIR
    / "window_size_diagnostics.csv"
)

TURNOVER_FILE = (
    PROCESSED_DIR
    / "window_size_turnover.csv"
)

EDGE_PERSISTENCE_FILE = (
    PROCESSED_DIR
    / "window_size_edge_persistence.csv"
)

INDUSTRY_PERSISTENCE_FILE = (
    PROCESSED_DIR
    / "window_size_industry_persistence_summary.csv"
)

SENSITIVITY_SUMMARY_FILE = (
    PROCESSED_DIR
    / "window_size_sensitivity_summary.csv"
)


# ============================================================
# 3. 工具函数：股票代码
# ============================================================

def normalize_code(x) -> str:
    """
    将股票代码转换成6位字符串。

    可处理：
        1
        000001
        000001.SZ
        600030.SH
    """

    s = str(x).strip()

    # Excel读取后可能出现 1.0
    if s.endswith(".0"):

        s = s[:-2]


    # 优先寻找连续6位数字
    match = re.search(
        r"(\d{6})",
        s
    )

    if match:

        return match.group(1)


    digits = "".join(
        ch
        for ch in s
        if ch.isdigit()
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

    return tuple(
        sorted(
            [
                a,
                b
            ]
        )
    )


# ============================================================
# 4. 读取股票名称和行业
# ============================================================

def load_stock_metadata():

    # --------------------------------------------------------
    # 优先使用stock_info.csv
    # --------------------------------------------------------

    if STOCK_INFO_FILE.exists():

        info = pd.read_csv(
            STOCK_INFO_FILE,
            dtype=str
        )


        # 兼容可能的字段名
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
                col
                for col
                in code_candidates
                if col in info.columns
            ),
            None
        )

        name_col = next(
            (
                col
                for col
                in name_candidates
                if col in info.columns
            ),
            None
        )

        industry_col = next(
            (
                col
                for col
                in industry_candidates
                if col in info.columns
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
                "stock_info.csv需要包含股票代码、名称和行业字段。"
                f"\n当前字段：{info.columns.tolist()}"
            )


        metadata = (
            info[
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
            metadata["code"]
            .apply(normalize_code)
        )


        metadata = (
            metadata
            .drop_duplicates(
                subset="code"
            )
            .reset_index(
                drop=True
            )
        )


        print(
            "使用stock_info.csv建立股票元数据。"
        )


        return metadata


    # --------------------------------------------------------
    # fallback：
    # 从尚未canonicalize的Rolling文件获取正确映射
    # --------------------------------------------------------

    if not ROLLING_EDGE_FILE.exists():

        raise FileNotFoundError(
            "既没有stock_info.csv，"
            "也没有rolling_glasso_edge_history.csv。"
        )


    raw = pd.read_csv(
        ROLLING_EDGE_FILE,
        dtype={
            "stock_1": str,
            "stock_2": str
        }
    )


    raw["stock_1"] = (
        raw["stock_1"]
        .apply(normalize_code)
    )

    raw["stock_2"] = (
        raw["stock_2"]
        .apply(normalize_code)
    )


    meta_1 = (
        raw[
            [
                "stock_1",
                "name_1",
                "industry_1"
            ]
        ]
        .rename(
            columns={
                "stock_1": "code",
                "name_1": "name",
                "industry_1": "industry"
            }
        )
    )


    meta_2 = (
        raw[
            [
                "stock_2",
                "name_2",
                "industry_2"
            ]
        ]
        .rename(
            columns={
                "stock_2": "code",
                "name_2": "name",
                "industry_2": "industry"
            }
        )
    )


    metadata = pd.concat(
        [
            meta_1,
            meta_2
        ],
        ignore_index=True
    )


    # --------------------------------------------------------
    # 检查名称冲突
    # --------------------------------------------------------

    name_conflict = (
        metadata
        .groupby("code")["name"]
        .nunique(
            dropna=True
        )
    )


    if (
        name_conflict > 1
    ).any():

        bad = (
            name_conflict[
                name_conflict > 1
            ]
            .index
            .tolist()
        )

        raise ValueError(
            f"股票名称映射存在冲突：{bad}"
        )


    industry_conflict = (
        metadata
        .groupby("code")["industry"]
        .nunique(
            dropna=True
        )
    )


    if (
        industry_conflict > 1
    ).any():

        bad = (
            industry_conflict[
                industry_conflict > 1
            ]
            .index
            .tolist()
        )

        raise ValueError(
            f"股票行业映射存在冲突：{bad}"
        )


    metadata = (
        metadata
        .drop_duplicates(
            subset="code"
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "未找到stock_info.csv，"
        "使用原始rolling edge history建立股票元数据。"
    )


    return metadata


# ============================================================
# 5. 读取收益率数据
# ============================================================

def load_returns(
    stock_codes
):

    df = pd.read_csv(
        RETURNS_FILE
    )


    # --------------------------------------------------------
    # 自动识别日期列
    # --------------------------------------------------------

    date_candidates = [
        "date",
        "Date",
        "trade_date",
        "datetime",
        "time"
    ]


    date_col = next(
        (
            col
            for col
            in date_candidates
            if col in df.columns
        ),
        None
    )


    if date_col is None:

        first_col = df.columns[0]

        parsed = pd.to_datetime(
            df[first_col],
            errors="coerce"
        )


        if (
            parsed.notna().mean()
            >
            0.95
        ):

            date_col = first_col

        else:

            raise ValueError(
                "无法自动识别stock_returns.csv的日期列。"
                f"\n当前字段：{df.columns.tolist()}"
            )


    df[date_col] = pd.to_datetime(
        df[date_col]
    )


    df = (
        df
        .sort_values(date_col)
        .drop_duplicates(
            subset=date_col
        )
        .set_index(date_col)
    )


    # --------------------------------------------------------
    # 将收益率列名称标准化为6位股票代码
    # --------------------------------------------------------

    rename_map = {}


    for col in df.columns:

        code = normalize_code(
            col
        )

        if code in stock_codes:

            rename_map[
                col
            ] = code


    df = df.rename(
        columns=rename_map
    )


    missing_codes = [
        code
        for code
        in stock_codes
        if code not in df.columns
    ]


    if missing_codes:

        raise ValueError(
            "收益率文件缺少以下股票："
            f"{missing_codes}"
        )


    returns = (
        df[
            stock_codes
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    n_before = len(
        returns
    )


    returns = (
        returns
        .dropna(
            how="any"
        )
    )


    n_after = len(
        returns
    )


    print(
        f"收益率样本数：{n_after}"
    )


    if n_before != n_after:

        print(
            f"由于缺失值删除了 "
            f"{n_before - n_after} 个交易日。"
        )


    return returns


# ============================================================
# 6. 读取固定alpha_1SE
# ============================================================

def load_alpha():

    if ALPHA_FIXED is not None:

        return float(
            ALPHA_FIXED
        )


    if not ALPHA_FILE.exists():

        raise FileNotFoundError(
            "找不到glasso_1se_selection.csv。\n"
            "可直接设置：ALPHA_FIXED = 0.216910"
        )


    df = pd.read_csv(
        ALPHA_FILE
    )


    candidates = [
        "alpha_1se",
        "alpha_1se_style",
        "one_se_alpha",
        "selected_alpha",
        "alpha"
    ]


    for col in candidates:

        if col not in df.columns:

            continue


        values = (
            pd.to_numeric(
                df[col],
                errors="coerce"
            )
            .dropna()
            .unique()
        )


        if len(values) == 1:

            return float(
                values[0]
            )


    raise ValueError(
        "无法唯一识别1-SE alpha。\n"
        f"当前字段：{df.columns.tolist()}\n"
        "可在代码中直接设置 "
        "ALPHA_FIXED = 0.216910"
    )


# ============================================================
# 7. Precision -> Partial Correlation
# ============================================================

def precision_to_partial(
    precision
):

    diagonal = np.diag(
        precision
    )


    if (
        diagonal <= 0
    ).any():

        raise ValueError(
            "Precision matrix存在非正对角元素。"
        )


    scale = np.sqrt(
        np.outer(
            diagonal,
            diagonal
        )
    )


    partial = (
        -precision
        /
        scale
    )


    np.fill_diagonal(
        partial,
        1.0
    )


    return partial


# ============================================================
# 8. 窗口内标准化
# ============================================================

def standardize_window(
    X
):

    mean = X.mean(
        axis=0
    )

    std = X.std(
        axis=0,
        ddof=0
    )


    if (
        std <= 1e-12
    ).any():

        raise ValueError(
            "某个Rolling window中存在近零标准差股票。"
        )


    Z = (
        X - mean
    ) / std


    return Z


# ============================================================
# 9. 构造共同比较日期
# ============================================================

def build_common_end_indices(
    n_obs
):

    max_window = max(
        WINDOW_SIZES
    )


    # --------------------------------------------------------
    # 以昨天W=252的日期网格为锚点
    #
    # 昨天：
    # end = 251, 271, 291, ...
    # --------------------------------------------------------

    baseline_end_indices = np.arange(
        BASE_WINDOW - 1,
        n_obs,
        STEP
    )


    # --------------------------------------------------------
    # 只保留W=504也有足够历史数据的日期
    # --------------------------------------------------------

    common_end_indices = (
        baseline_end_indices[
            baseline_end_indices
            >=
            max_window - 1
        ]
    )


    if len(
        common_end_indices
    ) < 2:

        raise ValueError(
            "共同Rolling日期太少，"
            "无法进行window-size sensitivity。"
        )


    return common_end_indices


# ============================================================
# 10. 加载数据
# ============================================================

metadata = load_stock_metadata()


metadata["code"] = (
    metadata["code"]
    .apply(normalize_code)
)


metadata = (
    metadata
    .sort_values("code")
    .reset_index(drop=True)
)


stock_codes = (
    metadata[
        "code"
    ]
    .tolist()
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


print(
    "\n股票数量：",
    len(stock_codes)
)


print(
    metadata.to_string(
        index=False
    )
)


returns = load_returns(
    stock_codes
)


alpha_fixed = load_alpha()


print(
    "\n固定Graphical Lasso alpha：",
    alpha_fixed
)


# ============================================================
# 11. 构造完全相同的network dates
# ============================================================

common_end_indices = (
    build_common_end_indices(
        len(returns)
    )
)


print(
    "\n共同比较窗口数量：",
    len(
        common_end_indices
    )
)


print(
    "共同起始Network Date：",
    returns.index[
        common_end_indices[0]
    ]
)


print(
    "共同结束Network Date：",
    returns.index[
        common_end_indices[-1]
    ]
)


# ============================================================
# 12. 主循环：W = 126, 252, 504
# ============================================================

network_rows = []

edge_rows = []

diagnostic_rows = []


for window_size in WINDOW_SIZES:

    print(
        "\n======================================"
    )

    print(
        f"开始运行 WINDOW = {window_size}"
    )

    print(
        "======================================"
    )


    for sensitivity_window_id, end_idx in enumerate(
        common_end_indices,
        start=1
    ):

        start_idx = (
            end_idx
            -
            window_size
            +
            1
        )


        window_returns = (
            returns
            .iloc[
                start_idx:
                end_idx + 1
            ]
            .copy()
        )


        window_start = (
            window_returns
            .index[0]
        )


        window_end = (
            window_returns
            .index[-1]
        )


        # ----------------------------------------------------
        # 昨天W=252时对应的原始window id
        # 方便和Day 7结果对应
        # ----------------------------------------------------

        baseline_window_id = (
            (
                end_idx
                -
                (BASE_WINDOW - 1)
            )
            //
            STEP
            +
            1
        )


        # ----------------------------------------------------
        # 每个window内部重新标准化
        # ----------------------------------------------------

        X = (
            window_returns
            .to_numpy(
                dtype=float
            )
        )


        Z = standardize_window(
            X
        )


        # ----------------------------------------------------
        # Graphical Lasso
        # ----------------------------------------------------

        model = GraphicalLasso(

            alpha=alpha_fixed,

            mode=MODE,

            tol=GLASSO_TOL,

            enet_tol=ENET_TOL,

            max_iter=MAX_ITER,

            assume_centered=True
        )


        fit_error = ""

        convergence_warning = False


        try:

            with warnings.catch_warnings(
                record=True
            ) as caught:

                warnings.simplefilter(
                    "always",
                    ConvergenceWarning
                )


                model.fit(
                    Z
                )


                convergence_warning = any(

                    issubclass(
                        w.category,
                        ConvergenceWarning
                    )

                    for w in caught
                )


        except Exception as exc:

            fit_error = repr(
                exc
            )


            diagnostic_rows.append(
                {
                    "window_size":
                        window_size,

                    "sensitivity_window_id":
                        sensitivity_window_id,

                    "baseline_window_id":
                        baseline_window_id,

                    "window_start":
                        window_start,

                    "window_end":
                        window_end,

                    "network_date":
                        window_end,

                    "fit_ok":
                        False,

                    "convergence_warning":
                        False,

                    "n_iter":
                        np.nan,

                    "last_objective":
                        np.nan,

                    "last_dual_gap":
                        np.nan,

                    "fit_error":
                        fit_error
                }
            )


            continue


        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        costs = getattr(
            model,
            "costs_",
            []
        )


        if len(
            costs
        ) > 0:

            last_objective = float(
                costs[-1][0]
            )

            last_dual_gap = float(
                costs[-1][1]
            )

        else:

            last_objective = np.nan

            last_dual_gap = np.nan


        diagnostic_rows.append(
            {
                "window_size":
                    window_size,

                "sensitivity_window_id":
                    sensitivity_window_id,

                "baseline_window_id":
                    baseline_window_id,

                "window_start":
                    window_start,

                "window_end":
                    window_end,

                "network_date":
                    window_end,

                "fit_ok":
                    True,

                "convergence_warning":
                    convergence_warning,

                "n_iter":
                    model.n_iter_,

                "last_objective":
                    last_objective,

                "last_dual_gap":
                    last_dual_gap,

                "fit_error":
                    ""
            }
        )


        # ----------------------------------------------------
        # Precision / Partial Correlation
        # ----------------------------------------------------

        precision = (
            model.precision_
        )


        partial = (
            precision_to_partial(
                precision
            )
        )


        # ----------------------------------------------------
        # 建图
        # ----------------------------------------------------

        G = nx.Graph()


        for code in stock_codes:

            G.add_node(
                code,

                name=name_map[
                    code
                ],

                industry=industry_map[
                    code
                ]
            )


        selected_abs_partials = []

        same_industry_edges = 0


        # ----------------------------------------------------
        # 保存所有股票对，不只是selected edge
        # ----------------------------------------------------

        for i in range(
            len(stock_codes)
        ):

            for j in range(
                i + 1,
                len(stock_codes)
            ):

                stock_1 = (
                    stock_codes[i]
                )

                stock_2 = (
                    stock_codes[j]
                )


                precision_ij = float(
                    precision[
                        i,
                        j
                    ]
                )


                partial_ij = float(
                    partial[
                        i,
                        j
                    ]
                )


                abs_partial = abs(
                    partial_ij
                )


                selected = (
                    abs(
                        precision_ij
                    )
                    >
                    ZERO_TOL
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


                edge_rows.append(
                    {
                        "window_size":
                            window_size,

                        "sensitivity_window_id":
                            sensitivity_window_id,

                        "baseline_window_id":
                            baseline_window_id,

                        "window_start":
                            window_start,

                        "window_end":
                            window_end,

                        "network_date":
                            window_end,

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

                        "precision":
                            precision_ij,

                        "partial_correlation":
                            partial_ij,

                        "abs_partial_correlation":
                            abs_partial,

                        "selected":
                            selected,

                        "same_industry":
                            same_industry
                    }
                )


                if selected:

                    G.add_edge(
                        stock_1,
                        stock_2,

                        partial_correlation=
                            partial_ij,

                        abs_partial=
                            abs_partial
                    )


                    selected_abs_partials.append(
                        abs_partial
                    )


                    if same_industry:

                        same_industry_edges += 1


        # ----------------------------------------------------
        # 网络级指标
        # ----------------------------------------------------

        n_nodes = (
            G.number_of_nodes()
        )


        n_edges = (
            G.number_of_edges()
        )


        max_possible_edges = (
            n_nodes
            *
            (
                n_nodes - 1
            )
            /
            2
        )


        density = (
            n_edges
            /
            max_possible_edges
        )


        mean_degree = (
            2
            *
            n_edges
            /
            n_nodes
        )


        degrees = dict(
            G.degree()
        )


        max_degree = max(
            degrees.values()
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


        if n_edges > 0:

            mean_abs_partial = float(
                np.mean(
                    selected_abs_partials
                )
            )

            median_abs_partial = float(
                np.median(
                    selected_abs_partials
                )
            )

            same_industry_ratio = (
                same_industry_edges
                /
                n_edges
            )

        else:

            mean_abs_partial = np.nan

            median_abs_partial = np.nan

            same_industry_ratio = np.nan


        network_rows.append(
            {
                "window_size":
                    window_size,

                "sensitivity_window_id":
                    sensitivity_window_id,

                "baseline_window_id":
                    baseline_window_id,

                "window_start":
                    window_start,

                "window_end":
                    window_end,

                "network_date":
                    window_end,

                "alpha":
                    alpha_fixed,

                "n_observations":
                    window_size,

                "n_nodes":
                    n_nodes,

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

                "cross_industry_edges":
                    n_edges
                    -
                    same_industry_edges,

                "same_industry_edge_ratio":
                    same_industry_ratio
            }
        )


# ============================================================
# 13. 转DataFrame并保存原始结果
# ============================================================

network_df = pd.DataFrame(
    network_rows
)


edge_df = pd.DataFrame(
    edge_rows
)


diagnostics_df = pd.DataFrame(
    diagnostic_rows
)


network_df.to_csv(
    NETWORK_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


edge_df.to_csv(
    EDGE_HISTORY_FILE,
    index=False,
    encoding="utf-8-sig"
)


diagnostics_df.to_csv(
    DIAGNOSTICS_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. 检查拟合是否全部成功
# ============================================================

failed = diagnostics_df[
    ~diagnostics_df[
        "fit_ok"
    ]
]


warned = diagnostics_df[
    diagnostics_df[
        "convergence_warning"
    ]
]


print(
    "\n======================================"
)

print(
    "Graphical Lasso Diagnostics"
)

print(
    "======================================"
)


print(
    "总拟合次数：",
    len(
        diagnostics_df
    )
)


print(
    "拟合失败次数：",
    len(
        failed
    )
)


print(
    "ConvergenceWarning次数：",
    len(
        warned
    )
)


if len(
    failed
) > 0:

    raise RuntimeError(
        "存在Graphical Lasso拟合失败。"
        "请先检查window_size_diagnostics.csv。"
    )


# ============================================================
# 15. 每个W计算Edge Persistence
# ============================================================

persistence_df = (
    edge_df
    .groupby(
        [
            "window_size",
            "stock_1",
            "name_1",
            "industry_1",
            "stock_2",
            "name_2",
            "industry_2",
            "same_industry"
        ],
        as_index=False
    )
    .agg(

        windows_selected=(
            "selected",
            "sum"
        ),

        total_windows=(
            "sensitivity_window_id",
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


persistence_df[
    "persistence"
] = (
    persistence_df[
        "windows_selected"
    ]
    /
    persistence_df[
        "total_windows"
    ]
)


# ------------------------------------------------------------
# 被选中时平均partial
# ------------------------------------------------------------

selected_edges = edge_df[
    edge_df[
        "selected"
    ]
]


selected_strength_df = (
    selected_edges
    .groupby(
        [
            "window_size",
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


persistence_df = (
    persistence_df
    .merge(
        selected_strength_df,

        on=[
            "window_size",
            "stock_1",
            "stock_2"
        ],

        how="left"
    )
)


persistence_df.to_csv(
    EDGE_PERSISTENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 每个W计算相邻网络Turnover
# ============================================================

turnover_rows = []


for window_size in WINDOW_SIZES:

    current_edges = edge_df[
        edge_df[
            "window_size"
        ]
        ==
        window_size
    ]


    ids = sorted(
        current_edges[
            "sensitivity_window_id"
        ]
        .unique()
    )


    for k in range(
        1,
        len(
            ids
        )
    ):

        previous_id = (
            ids[
                k - 1
            ]
        )

        current_id = (
            ids[
                k
            ]
        )


        previous_df = current_edges[
            (
                current_edges[
                    "sensitivity_window_id"
                ]
                ==
                previous_id
            )
            &
            (
                current_edges[
                    "selected"
                ]
            )
        ]


        current_df = current_edges[
            (
                current_edges[
                    "sensitivity_window_id"
                ]
                ==
                current_id
            )
            &
            (
                current_edges[
                    "selected"
                ]
            )
        ]


        E_previous = {

            canonical_pair(
                row.stock_1,
                row.stock_2
            )

            for row
            in previous_df.itertuples()
        }


        E_current = {

            canonical_pair(
                row.stock_1,
                row.stock_2
            )

            for row
            in current_df.itertuples()
        }


        common = (
            E_previous
            &
            E_current
        )


        lost = (
            E_previous
            -
            E_current
        )


        gained = (
            E_current
            -
            E_previous
        )


        union = (
            E_previous
            |
            E_current
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


        turnover = (
            1.0
            -
            jaccard
        )


        # ----------------------------------------------------
        # same/cross decomposition
        # ----------------------------------------------------

        def is_same_industry(
            edge
        ):

            stock_1, stock_2 = edge

            return (
                industry_map[
                    stock_1
                ]
                ==
                industry_map[
                    stock_2
                ]
            )


        lost_same = sum(

            is_same_industry(
                edge
            )

            for edge
            in lost
        )


        gained_same = sum(

            is_same_industry(
                edge
            )

            for edge
            in gained
        )


        lost_cross = (
            len(
                lost
            )
            -
            lost_same
        )


        gained_cross = (
            len(
                gained
            )
            -
            gained_same
        )


        current_date = (
            current_df[
                "network_date"
            ]
            .iloc[0]
        )


        turnover_rows.append(
            {
                "window_size":
                    window_size,

                "window_from":
                    previous_id,

                "window_to":
                    current_id,

                "date_to":
                    current_date,

                "edges_from":
                    len(
                        E_previous
                    ),

                "edges_to":
                    len(
                        E_current
                    ),

                "common_edges":
                    len(
                        common
                    ),

                "lost_edges":
                    len(
                        lost
                    ),

                "gained_edges":
                    len(
                        gained
                    ),

                "gross_edge_changes":
                    len(
                        lost
                    )
                    +
                    len(
                        gained
                    ),

                "net_edge_change":
                    len(
                        E_current
                    )
                    -
                    len(
                        E_previous
                    ),

                "jaccard":
                    jaccard,

                "turnover":
                    turnover,

                "lost_same_industry":
                    lost_same,

                "lost_cross_industry":
                    lost_cross,

                "gained_same_industry":
                    gained_same,

                "gained_cross_industry":
                    gained_cross
            }
        )


turnover_df = pd.DataFrame(
    turnover_rows
)


turnover_df.to_csv(
    TURNOVER_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. Same vs Cross Persistence Summary
# ============================================================

industry_persistence_df = (
    persistence_df
    .assign(
        industry_relation=np.where(
            persistence_df[
                "same_industry"
            ],
            "Same industry",
            "Cross industry"
        )
    )
    .groupby(
        [
            "window_size",
            "industry_relation"
        ],
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
                    ).sum()
                )
        ),

        n_persistence_equal_1=(
            "persistence",
            lambda x:
                int(
                    np.isclose(
                        x,
                        1.0
                    ).sum()
                )
        ),

        mean_abs_partial_when_selected=(
            "mean_abs_partial_when_selected",
            "mean"
        )
    )
)


industry_persistence_df[
    "share_persistence_ge_080"
] = (
    industry_persistence_df[
        "n_persistence_ge_080"
    ]
    /
    industry_persistence_df[
        "n_possible_pairs"
    ]
)


industry_persistence_df[
    "share_persistence_equal_1"
] = (
    industry_persistence_df[
        "n_persistence_equal_1"
    ]
    /
    industry_persistence_df[
        "n_possible_pairs"
    ]
)


industry_persistence_df.to_csv(
    INDUSTRY_PERSISTENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. 最终Window-size Summary
# ============================================================

summary_rows = []


for window_size in WINDOW_SIZES:

    net = network_df[
        network_df[
            "window_size"
        ]
        ==
        window_size
    ]


    turn = turnover_df[
        turnover_df[
            "window_size"
        ]
        ==
        window_size
    ]


    persist = persistence_df[
        persistence_df[
            "window_size"
        ]
        ==
        window_size
    ]


    same_persist = persist[
        persist[
            "same_industry"
        ]
    ]


    cross_persist = persist[
        ~persist[
            "same_industry"
        ]
    ]


    persistent_edges = int(
        (
            persist[
                "persistence"
            ]
            >=
            0.80
        )
        .sum()
    )


    always_persistent_edges = int(
        np.isclose(
            persist[
                "persistence"
            ],
            1.0
        )
        .sum()
    )


    changed_same = (
        turn[
            "lost_same_industry"
        ].sum()
        +
        turn[
            "gained_same_industry"
        ].sum()
    )


    changed_cross = (
        turn[
            "lost_cross_industry"
        ].sum()
        +
        turn[
            "gained_cross_industry"
        ].sum()
    )


    total_changes = (
        changed_same
        +
        changed_cross
    )


    cross_change_share = (

        changed_cross
        /
        total_changes

        if total_changes > 0

        else np.nan
    )


    summary_rows.append(
        {
            "window_size":
                window_size,

            "n_common_windows":
                net[
                    "sensitivity_window_id"
                ]
                .nunique(),

            "first_network_date":
                net[
                    "network_date"
                ]
                .min(),

            "last_network_date":
                net[
                    "network_date"
                ]
                .max(),

            "mean_edges":
                net[
                    "n_edges"
                ]
                .mean(),

            "sd_edges":
                net[
                    "n_edges"
                ]
                .std(
                    ddof=1
                ),

            "min_edges":
                net[
                    "n_edges"
                ]
                .min(),

            "max_edges":
                net[
                    "n_edges"
                ]
                .max(),

            "mean_density":
                net[
                    "density"
                ]
                .mean(),

            "mean_abs_partial":
                net[
                    "mean_abs_partial"
                ]
                .mean(),

            "mean_same_industry_ratio":
                net[
                    "same_industry_edge_ratio"
                ]
                .mean(),

            "mean_turnover":
                turn[
                    "turnover"
                ]
                .mean(),

            "median_turnover":
                turn[
                    "turnover"
                ]
                .median(),

            "max_turnover":
                turn[
                    "turnover"
                ]
                .max(),

            "persistent_edges_ge_080":
                persistent_edges,

            "always_persistent_edges":
                always_persistent_edges,

            "same_mean_persistence":
                same_persist[
                    "persistence"
                ]
                .mean(),

            "cross_mean_persistence":
                cross_persist[
                    "persistence"
                ]
                .mean(),

            "same_persistence_ge_080_rate":
                (
                    same_persist[
                        "persistence"
                    ]
                    >=
                    0.80
                )
                .mean(),

            "cross_persistence_ge_080_rate":
                (
                    cross_persist[
                        "persistence"
                    ]
                    >=
                    0.80
                )
                .mean(),

            "same_edge_change_events":
                changed_same,

            "cross_edge_change_events":
                changed_cross,

            "cross_edge_change_share":
                cross_change_share
        }
    )


sensitivity_summary_df = pd.DataFrame(
    summary_rows
)


sensitivity_summary_df.to_csv(
    SENSITIVITY_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n======================================"
)

print(
    "Window-size Sensitivity Summary"
)

print(
    "======================================"
)


print(
    sensitivity_summary_df
    .to_string(
        index=False
    )
)


# ============================================================
# 19. 图1：Edge Count
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


for window_size in WINDOW_SIZES:

    temp = network_df[
        network_df[
            "window_size"
        ]
        ==
        window_size
    ]


    ax.plot(
        temp[
            "network_date"
        ],
        temp[
            "n_edges"
        ],
        marker="o",
        label=f"W={window_size}"
    )


ax.set_xlabel(
    "Network date"
)

ax.set_ylabel(
    "Number of edges"
)

ax.set_title(
    "Window-size Sensitivity: Edge Count"
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "window_size_edge_count_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 20. 图2：Network Turnover
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


for window_size in WINDOW_SIZES:

    temp = turnover_df[
        turnover_df[
            "window_size"
        ]
        ==
        window_size
    ]


    ax.plot(
        temp[
            "date_to"
        ],
        temp[
            "turnover"
        ],
        marker="o",
        label=f"W={window_size}"
    )


ax.set_xlabel(
    "Current network date"
)

ax.set_ylabel(
    "Network turnover"
)

ax.set_title(
    "Window-size Sensitivity: Adjacent Network Turnover"
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "window_size_turnover_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 21. 图3：Same-industry Ratio
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


for window_size in WINDOW_SIZES:

    temp = network_df[
        network_df[
            "window_size"
        ]
        ==
        window_size
    ]


    ax.plot(
        temp[
            "network_date"
        ],
        temp[
            "same_industry_edge_ratio"
        ],
        marker="o",
        label=f"W={window_size}"
    )


ax.set_xlabel(
    "Network date"
)

ax.set_ylabel(
    "Same-industry edge ratio"
)

ax.set_title(
    "Window-size Sensitivity: Same-industry Edge Ratio"
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "window_size_same_industry_ratio.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 22. 图4：Mean Absolute Partial Correlation
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


for window_size in WINDOW_SIZES:

    temp = network_df[
        network_df[
            "window_size"
        ]
        ==
        window_size
    ]


    ax.plot(
        temp[
            "network_date"
        ],
        temp[
            "mean_abs_partial"
        ],
        marker="o",
        label=f"W={window_size}"
    )


ax.set_xlabel(
    "Network date"
)

ax.set_ylabel(
    "Mean absolute GLasso partial correlation"
)

ax.set_title(
    "Window-size Sensitivity: Mean Edge Strength"
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "window_size_mean_abs_partial.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 23. 最终输出
# ============================================================

print(
    "\n======================================"
)

print(
    "Window-size sensitivity分析完成"
)

print(
    "======================================"
)


print(
    "\n主要输出文件："
)


for file in [

    NETWORK_SUMMARY_FILE,
    EDGE_HISTORY_FILE,
    DIAGNOSTICS_FILE,
    TURNOVER_FILE,
    EDGE_PERSISTENCE_FILE,
    INDUSTRY_PERSISTENCE_FILE,
    SENSITIVITY_SUMMARY_FILE

]:

    print(
        file
    )