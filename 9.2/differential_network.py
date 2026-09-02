from __future__ import annotations

from itertools import combinations
from pathlib import Path
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# 0. Project directory
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 1. Paths
# ============================================================

# Stage 2 outputs
REGIME_NETWORK_DIR = (
    BASE_DIR
    / "regime_networks"
)


R3_PARTIAL_FILE = (
    REGIME_NETWORK_DIR
    / "R3_partial.csv"
)


R4_PARTIAL_FILE = (
    REGIME_NETWORK_DIR
    / "R4_partial.csv"
)


# Stock metadata
STOCK_INFO_FILE = (
    BASE_DIR
    / "stock_network"
    / "data"
    / "processed"
    / "stock_info.csv"
)


# Optional Stage-3 output
STAGE3_CHANGE_SUMMARY_FILE = (
    BASE_DIR
    / "regime_network_comparison"
    / "R3_R4_independent_change_summary.csv"
)


# Stage-5 output directory
OUTPUT_DIR = (
    BASE_DIR
    / "stage5_differential_network"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Main settings
# ============================================================

# Numerical tolerance for deciding whether an estimated
# partial correlation corresponds to a selected GLasso edge.
SUPPORT_TOL = 1e-8


# Numerical tolerance for Strengthened / Weakened / Stable.
STRENGTH_TOL = 1e-10


# Number of largest changes shown in the main network figure.
TOP_K_DIFF_EDGES = 20


# Reproducible spring-layout seed.
NETWORK_LAYOUT_SEED = 20260902


# Whether node labels display:
#
#   招商银行
#   600036
#
# If False, only stock name is displayed.
SHOW_STOCK_CODE_IN_LABEL = True


# Label font size
NODE_LABEL_FONT_SIZE = 8


# Network node size
NODE_SIZE = 1200


# ============================================================
# 3. Chinese font configuration
# ============================================================

def setup_chinese_font() -> tuple[FontProperties, str]:
    """
    Detect and configure a Chinese-capable font.

    Priority:
        Microsoft YaHei
        Microsoft YaHei UI
        SimHei
        DengXian
        SimSun
        Noto Sans CJK SC
        Source Han Sans SC

    On Windows, if Matplotlib's font cache does not contain
    these fonts, the function additionally checks common
    C:\\Windows\\Fonts paths directly.

    Returns
    -------
    font_prop
        FontProperties used explicitly when drawing Chinese.

    description
        Human-readable description of selected font.
    """

    preferred_font_names = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "DengXian",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]


    # --------------------------------------------------------
    # Method 1:
    # Search Matplotlib's registered font list.
    # --------------------------------------------------------

    installed_fonts = {
        f.name
        for f in font_manager.fontManager.ttflist
    }


    for font_name in preferred_font_names:

        if font_name in installed_fonts:

            plt.rcParams[
                "font.family"
            ] = "sans-serif"


            plt.rcParams[
                "font.sans-serif"
            ] = [
                font_name,
                "DejaVu Sans",
            ]


            plt.rcParams[
                "axes.unicode_minus"
            ] = False


            font_prop = FontProperties(
                family=font_name
            )


            print(
                f"Chinese font selected: "
                f"{font_name}"
            )


            return (
                font_prop,
                font_name,
            )


    # --------------------------------------------------------
    # Method 2:
    # Directly check Windows font files.
    #
    # This is useful when the font exists in Windows but
    # Matplotlib's font cache has not registered it.
    # --------------------------------------------------------

    windows_font_candidates = [
        (
            "Microsoft YaHei",
            r"C:\Windows\Fonts\msyh.ttc",
        ),
        (
            "Microsoft YaHei Bold",
            r"C:\Windows\Fonts\msyhbd.ttc",
        ),
        (
            "SimHei",
            r"C:\Windows\Fonts\simhei.ttf",
        ),
        (
            "DengXian",
            r"C:\Windows\Fonts\Deng.ttf",
        ),
        (
            "SimSun",
            r"C:\Windows\Fonts\simsun.ttc",
        ),
    ]


    for (
        font_description,
        font_path,
    ) in windows_font_candidates:

        if os.path.exists(
            font_path
        ):

            try:

                # Register font explicitly when possible.
                font_manager.fontManager.addfont(
                    font_path
                )

            except Exception:

                # TTC collections may still work through
                # FontProperties(fname=...) even if addfont
                # behaves differently across versions.
                pass


            font_prop = FontProperties(
                fname=font_path
            )


            plt.rcParams[
                "axes.unicode_minus"
            ] = False


            print(
                "Chinese font selected directly "
                f"from Windows: "
                f"{font_description}"
            )

            print(
                f"Font file: {font_path}"
            )


            return (
                font_prop,
                font_description,
            )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    print(
        "WARNING:"
    )

    print(
        "No Chinese-capable font was detected."
    )

    print(
        "Chinese labels may appear as blank boxes."
    )

    print(
        "Please check C:\\Windows\\Fonts or rebuild "
        "the Matplotlib font cache."
    )


    plt.rcParams[
        "axes.unicode_minus"
    ] = False


    return (
        FontProperties(
            family="DejaVu Sans"
        ),
        "DejaVu Sans (fallback)",
    )


CHINESE_FONT_PROP, CHINESE_FONT_NAME = (
    setup_chinese_font()
)


# ============================================================
# 4. CSV encoding fallback
# ============================================================

def read_csv_with_encoding_fallback(
    path: Path,
    **kwargs,
) -> pd.DataFrame:
    """
    Read CSV using several common Chinese/Windows encodings.
    """

    encodings = [
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
    ]


    last_exception = None


    for encoding in encodings:

        try:

            df = pd.read_csv(
                path,
                encoding=encoding,
                **kwargs,
            )


            print(
                f"Loaded {path.name} "
                f"with encoding={encoding}"
            )


            return df


        except UnicodeDecodeError as exc:

            last_exception = exc


    raise RuntimeError(
        f"Could not decode CSV file:\n"
        f"{path}\n"
        f"Last error: {last_exception}"
    )


# ============================================================
# 5. Normalize stock codes
# ============================================================

def normalize_code(x) -> str:
    """
    Convert stock code to a six-digit string.

    Examples
    --------
    1       -> "000001"
    "1.0"   -> "000001"
    600036  -> "600036"
    """

    x = str(x).strip()


    if x.endswith(
        ".0"
    ):

        x = x[:-2]


    if x.isdigit():

        x = x.zfill(
            6
        )


    return x


# ============================================================
# 6. Load square matrix
# ============================================================

def load_square_matrix(
    path: Path,
) -> pd.DataFrame:
    """
    Load a square stock-by-stock matrix.

    Checks:
        - square shape
        - row/column consistency
        - numerical symmetry
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find matrix:\n"
            f"{path}"
        )


    matrix = read_csv_with_encoding_fallback(
        path,
        index_col=0,
    )


    matrix.index = [
        normalize_code(x)
        for x in matrix.index
    ]


    matrix.columns = [
        normalize_code(x)
        for x in matrix.columns
    ]


    matrix = matrix.apply(
        pd.to_numeric,
        errors="raise",
    )


    if (
        matrix.shape[0]
        !=
        matrix.shape[1]
    ):

        raise ValueError(
            f"{path.name} is not square: "
            f"{matrix.shape}"
        )


    if (
        set(matrix.index)
        !=
        set(matrix.columns)
    ):

        raise ValueError(
            f"Rows and columns differ "
            f"in {path.name}."
        )


    # Reorder columns exactly to row order.
    matrix = (
        matrix.loc[
            matrix.index,
            matrix.index,
        ]
        .copy()
    )


    # Use an independent writable array.
    arr = matrix.to_numpy(
        dtype=float,
        copy=True,
    )


    symmetry_error = float(
        np.max(
            np.abs(
                arr
                -
                arr.T
            )
        )
    )


    if symmetry_error > 1e-6:

        raise ValueError(
            f"{path.name} is not sufficiently symmetric. "
            f"Max error = "
            f"{symmetry_error:.3e}"
        )


    return matrix


# ============================================================
# 7. Metadata-column helper
# ============================================================

def find_column(
    df: pd.DataFrame,
    aliases: list[str],
) -> str | None:

    for col in aliases:

        if col in df.columns:

            return col


    return None


# ============================================================
# 8. Load stock metadata
# ============================================================

if not STOCK_INFO_FILE.exists():

    raise FileNotFoundError(
        "Cannot find stock metadata:\n"
        f"{STOCK_INFO_FILE}"
    )


stock_info_raw = (
    read_csv_with_encoding_fallback(
        STOCK_INFO_FILE,
        dtype=str,
    )
)


COLUMN_ALIASES = {

    "code": [
        "code",
        "Code",
        "股票代码",
        "证券代码",
        "stock_code",
        "ticker",
    ],

    "name": [
        "name",
        "Name",
        "股票名称",
        "股票简称",
        "证券名称",
        "证券简称",
        "stock_name",
        "stockname",
    ],

    "industry": [
        "industry",
        "Industry",
        "行业",
        "所属行业",
        "行业名称",
        "industry_name",
    ],
}


code_col = find_column(
    stock_info_raw,
    COLUMN_ALIASES[
        "code"
    ],
)


name_col = find_column(
    stock_info_raw,
    COLUMN_ALIASES[
        "name"
    ],
)


industry_col = find_column(
    stock_info_raw,
    COLUMN_ALIASES[
        "industry"
    ],
)


if code_col is None:

    raise ValueError(
        "No stock-code column found in stock_info.csv.\n"
        f"Available columns:\n"
        f"{list(stock_info_raw.columns)}"
    )


if industry_col is None:

    raise ValueError(
        "No industry column found in stock_info.csv.\n"
        f"Available columns:\n"
        f"{list(stock_info_raw.columns)}"
    )


rename_map = {
    code_col:
        "code",

    industry_col:
        "industry",
}


if name_col is not None:

    rename_map[
        name_col
    ] = "name"


stock_info = (
    stock_info_raw
    .rename(
        columns=rename_map
    )
    .copy()
)


stock_info[
    "code"
] = (
    stock_info[
        "code"
    ]
    .map(
        normalize_code
    )
)


stock_info = (
    stock_info
    .drop_duplicates(
        subset="code"
    )
    .set_index(
        "code"
    )
)


HAS_NAME = (
    "name"
    in stock_info.columns
)


# ============================================================
# 9. Load R3/R4 partial-correlation matrices
# ============================================================

R3_partial = load_square_matrix(
    R3_PARTIAL_FILE
)


R4_partial = load_square_matrix(
    R4_PARTIAL_FILE
)


nodes_R3 = list(
    R3_partial.index
)


nodes_R4 = list(
    R4_partial.index
)


if set(
    nodes_R3
) != set(
    nodes_R4
):

    raise ValueError(
        "R3 and R4 do not contain "
        "the same stock universe."
    )


# Common node ordering
STOCKS = nodes_R3


R3_partial = (
    R3_partial.loc[
        STOCKS,
        STOCKS,
    ]
    .copy()
)


R4_partial = (
    R4_partial.loc[
        STOCKS,
        STOCKS,
    ]
    .copy()
)


p = len(
    STOCKS
)


missing_metadata = (
    set(STOCKS)
    -
    set(
        stock_info.index
    )
)


if missing_metadata:

    raise ValueError(
        "Missing metadata for stocks:\n"
        f"{sorted(missing_metadata)}"
    )


# ============================================================
# 10. Label diagnostics
# ============================================================

print(
    "\n"
    +
    "=" * 78
)

print(
    "Chinese Font Diagnostic"
)

print(
    "=" * 78
)

print(
    f"Selected font: "
    f"{CHINESE_FONT_NAME}"
)


print(
    "\n"
    +
    "=" * 78
)

print(
    "Stock Label Diagnostic"
)

print(
    "=" * 78
)


print(
    f"Stock-name column detected: "
    f"{HAS_NAME}"
)


if HAS_NAME:

    diagnostic_columns = [
        "name",
        "industry",
    ]


    print(
        stock_info.loc[
            STOCKS,
            diagnostic_columns,
        ]
        .to_string()
    )


else:

    print(
        "WARNING: stock-name column was not detected."
    )

    print(
        "The figure will use stock codes instead."
    )

    print(
        "Metadata columns:"
    )

    print(
        list(
            stock_info.columns
        )
    )


# ============================================================
# 11. Basic diagnostics
# ============================================================

print(
    "\n"
    +
    "=" * 78
)

print(
    "Stage 5: Differential Network Analysis"
)

print(
    "=" * 78
)


print(
    f"Number of stocks: "
    f"{p}"
)


print(
    f"Possible edges: "
    f"{p * (p - 1) // 2}"
)


if p != 15:

    print(
        f"WARNING: expected p=15, "
        f"but found p={p}."
    )


# ============================================================
# 12. Differential partial-correlation matrix
#
# Delta R = R4 - R3
#
# FIX:
# Explicit writable NumPy arrays are created.
# ============================================================

R3_array = R3_partial.to_numpy(
    dtype=float,
    copy=True,
)


R4_array = R4_partial.to_numpy(
    dtype=float,
    copy=True,
)


delta_array = (
    R4_array
    -
    R3_array
)


if not delta_array.flags.writeable:

    delta_array = (
        delta_array.copy()
    )


# Diagonal is irrelevant to edge differences.
np.fill_diagonal(
    delta_array,
    0.0,
)


delta_partial = pd.DataFrame(
    delta_array,
    index=STOCKS,
    columns=STOCKS,
)


delta_partial.to_csv(
    OUTPUT_DIR
    / "R3_R4_delta_partial_matrix.csv",

    encoding="utf-8-sig",
)


print(
    "\nDifferential partial-correlation "
    "matrix constructed successfully."
)


# ============================================================
# 13. Construct edge-level differential table
# ============================================================

edge_rows = []


for (
    stock_i,
    stock_j,
) in combinations(
    STOCKS,
    2,
):

    rho3 = float(
        R3_partial.at[
            stock_i,
            stock_j,
        ]
    )


    rho4 = float(
        R4_partial.at[
            stock_i,
            stock_j,
        ]
    )


    abs3 = abs(
        rho3
    )


    abs4 = abs(
        rho4
    )


    edge_R3 = (
        abs3
        >
        SUPPORT_TOL
    )


    edge_R4 = (
        abs4
        >
        SUPPORT_TOL
    )


    # --------------------------------------------------------
    # Support change
    # --------------------------------------------------------

    if (
        edge_R3
        and
        edge_R4
    ):

        support_change = (
            "Persistent"
        )


    elif (
        edge_R3
        and
        not edge_R4
    ):

        support_change = (
            "Lost"
        )


    elif (
        not edge_R3
        and
        edge_R4
    ):

        support_change = (
            "Gained"
        )


    else:

        support_change = (
            "Absent"
        )


    # --------------------------------------------------------
    # Strength change among persistent edges
    # --------------------------------------------------------

    if (
        support_change
        ==
        "Persistent"
    ):

        delta_abs = (
            abs4
            -
            abs3
        )


        if (
            rho3
            *
            rho4
            < 0
        ):

            strength_change = (
                "SignFlip"
            )


        elif (
            delta_abs
            >
            STRENGTH_TOL
        ):

            strength_change = (
                "Strengthened"
            )


        elif (
            delta_abs
            <
            -STRENGTH_TOL
        ):

            strength_change = (
                "Weakened"
            )


        else:

            strength_change = (
                "Stable"
            )


    else:

        strength_change = (
            "NotApplicable"
        )


    industry_i = str(
        stock_info.at[
            stock_i,
            "industry",
        ]
    )


    industry_j = str(
        stock_info.at[
            stock_j,
            "industry",
        ]
    )


    same_industry = (
        industry_i
        ==
        industry_j
    )


    relation = (
        "Same"
        if same_industry
        else
        "Cross"
    )


    row = {

        "stock_i":
            stock_i,

        "stock_j":
            stock_j,

        "industry_i":
            industry_i,

        "industry_j":
            industry_j,

        "relation":
            relation,

        "same_industry":
            same_industry,

        "partial_R3":
            rho3,

        "partial_R4":
            rho4,

        "abs_partial_R3":
            abs3,

        "abs_partial_R4":
            abs4,

        "delta_partial":
            (
                rho4
                -
                rho3
            ),

        "abs_delta_partial":
            abs(
                rho4
                -
                rho3
            ),

        "delta_abs_partial":
            (
                abs4
                -
                abs3
            ),

        "edge_R3":
            int(
                edge_R3
            ),

        "edge_R4":
            int(
                edge_R4
            ),

        "selected_union":
            int(
                edge_R3
                or
                edge_R4
            ),

        "support_change":
            support_change,

        "strength_change":
            strength_change,
    }


    if HAS_NAME:

        row[
            "name_i"
        ] = stock_info.at[
            stock_i,
            "name",
        ]


        row[
            "name_j"
        ] = stock_info.at[
            stock_j,
            "name",
        ]


    edge_rows.append(
        row
    )


edge_table = pd.DataFrame(
    edge_rows
)


# ============================================================
# 14. Validate pair count
# ============================================================

expected_pairs = (
    p
    *
    (p - 1)
    //
    2
)


if len(
    edge_table
) != expected_pairs:

    raise RuntimeError(
        "Unexpected number of stock pairs: "
        f"{len(edge_table)} "
        f"vs expected "
        f"{expected_pairs}."
    )


# ============================================================
# 15. Structural roles
# ============================================================

def structural_role(
    row: pd.Series,
) -> str:

    if (
        row[
            "support_change"
        ]
        ==
        "Persistent"
    ):

        if (
            row[
                "relation"
            ]
            ==
            "Same"
        ):

            return (
                "Persistent Same-industry"
            )

        return (
            "Persistent Cross-industry"
        )


    if (
        row[
            "support_change"
        ]
        in [
            "Lost",
            "Gained",
        ]
    ):

        if (
            row[
                "relation"
            ]
            ==
            "Same"
        ):

            return (
                "Changing Same-industry"
            )

        return (
            "Changing Cross-industry"
        )


    return (
        "Inactive"
    )


edge_table[
    "structural_role"
] = edge_table.apply(
    structural_role,
    axis=1,
)


# ============================================================
# 16. Sort full edge table
# ============================================================

support_order = {

    "Lost":
        1,

    "Gained":
        2,

    "Persistent":
        3,

    "Absent":
        4,
}


edge_table[
    "_support_order"
] = (
    edge_table[
        "support_change"
    ]
    .map(
        support_order
    )
)


edge_table = (
    edge_table
    .sort_values(
        [
            "_support_order",
            "abs_delta_partial",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .drop(
        columns=[
            "_support_order"
        ]
    )
    .reset_index(
        drop=True
    )
)


edge_table.to_csv(
    OUTPUT_DIR
    / "R3_R4_differential_edges_full.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 17. Union edges
# ============================================================

union_edges = (
    edge_table.loc[
        edge_table[
            "selected_union"
        ]
        ==
        1
    ]
    .copy()
)


union_edges.to_csv(
    OUTPUT_DIR
    / "R3_R4_differential_edges_union.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 18. Top differential edges
# ============================================================

top_changes = (
    union_edges
    .sort_values(
        "abs_delta_partial",
        ascending=False,
    )
    .head(
        TOP_K_DIFF_EDGES
    )
    .copy()
)


top_changes.to_csv(
    OUTPUT_DIR
    / "R3_R4_top_differential_edges.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 19. Same/Cross relation metrics
# ============================================================

def relation_metrics(
    df: pd.DataFrame,
    relation: str,
) -> dict:

    group = (
        df.loc[
            df[
                "relation"
            ]
            ==
            relation
        ]
        .copy()
    )


    possible_pairs = len(
        group
    )


    selected_R3 = int(
        group[
            "edge_R3"
        ]
        .sum()
    )


    selected_R4 = int(
        group[
            "edge_R4"
        ]
        .sum()
    )


    persistent = int(
        (
            group[
                "support_change"
            ]
            ==
            "Persistent"
        )
        .sum()
    )


    lost = int(
        (
            group[
                "support_change"
            ]
            ==
            "Lost"
        )
        .sum()
    )


    gained = int(
        (
            group[
                "support_change"
            ]
            ==
            "Gained"
        )
        .sum()
    )


    gross_support_change = (
        lost
        +
        gained
    )


    union_count = (
        persistent
        +
        lost
        +
        gained
    )


    jaccard = (
        persistent
        /
        union_count

        if union_count > 0
        else np.nan
    )


    retention_from_R3 = (
        persistent
        /
        selected_R3

        if selected_R3 > 0
        else np.nan
    )


    inherited_share_R4 = (
        persistent
        /
        selected_R4

        if selected_R4 > 0
        else np.nan
    )


    support_change_rate = (
        gross_support_change
        /
        possible_pairs

        if possible_pairs > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Transition intensity
    # --------------------------------------------------------

    delta_values = (
        group[
            "delta_partial"
        ]
        .to_numpy(
            dtype=float,
            copy=True,
        )
    )


    intensity_squared = float(
        np.sum(
            delta_values
            ** 2
        )
    )


    transition_intensity = float(
        np.sqrt(
            intensity_squared
        )
    )


    mean_abs_delta_all = float(
        group[
            "abs_delta_partial"
        ]
        .mean()
    )


    selected_union_group = (
        group.loc[
            group[
                "selected_union"
            ]
            ==
            1
        ]
    )


    mean_abs_delta_union = (
        float(
            selected_union_group[
                "abs_delta_partial"
            ]
            .mean()
        )

        if len(
            selected_union_group
        ) > 0

        else np.nan
    )


    return {

        "relation":
            relation,

        "possible_pairs":
            possible_pairs,

        "selected_R3":
            selected_R3,

        "selected_R4":
            selected_R4,

        "persistent":
            persistent,

        "lost":
            lost,

        "gained":
            gained,

        "gross_support_change":
            gross_support_change,

        "jaccard":
            jaccard,

        "retention_from_R3":
            retention_from_R3,

        "inherited_share_R4":
            inherited_share_R4,

        "support_change_rate_possible":
            support_change_rate,

        "transition_intensity":
            transition_intensity,

        "transition_intensity_squared":
            intensity_squared,

        "mean_abs_delta_all_pairs":
            mean_abs_delta_all,

        "mean_abs_delta_selected_union":
            mean_abs_delta_union,
    }


same_metrics = relation_metrics(
    edge_table,
    "Same",
)


cross_metrics = relation_metrics(
    edge_table,
    "Cross",
)


relation_summary = pd.DataFrame(
    [
        same_metrics,
        cross_metrics,
    ]
)


# ============================================================
# 20. Transition intensity shares
# ============================================================

total_intensity_squared = float(
    relation_summary[
        "transition_intensity_squared"
    ]
    .sum()
)


if (
    total_intensity_squared
    >
    0
):

    relation_summary[
        "transition_intensity_share"
    ] = (
        relation_summary[
            "transition_intensity_squared"
        ]
        /
        total_intensity_squared
    )


else:

    relation_summary[
        "transition_intensity_share"
    ] = np.nan


relation_summary.to_csv(
    OUTPUT_DIR
    / "R3_R4_same_cross_decomposition.csv",

    index=False,
    encoding="utf-8-sig",
)


TI_total = float(
    np.sqrt(
        total_intensity_squared
    )
)


TI_same = float(
    relation_summary.loc[
        relation_summary[
            "relation"
        ]
        ==
        "Same",

        "transition_intensity",
    ]
    .iloc[0]
)


TI_cross = float(
    relation_summary.loc[
        relation_summary[
            "relation"
        ]
        ==
        "Cross",

        "transition_intensity",
    ]
    .iloc[0]
)


same_intensity_share = float(
    relation_summary.loc[
        relation_summary[
            "relation"
        ]
        ==
        "Same",

        "transition_intensity_share",
    ]
    .iloc[0]
)


cross_intensity_share = float(
    relation_summary.loc[
        relation_summary[
            "relation"
        ]
        ==
        "Cross",

        "transition_intensity_share",
    ]
    .iloc[0]
)


# ============================================================
# 21. Support vs persistent-strength transition energy
# ============================================================

support_change_mask = (
    edge_table[
        "support_change"
    ]
    .isin(
        [
            "Lost",
            "Gained",
        ]
    )
)


persistent_mask = (
    edge_table[
        "support_change"
    ]
    ==
    "Persistent"
)


support_delta = (
    edge_table.loc[
        support_change_mask,
        "delta_partial",
    ]
    .to_numpy(
        dtype=float,
        copy=True,
    )
)


persistent_delta = (
    edge_table.loc[
        persistent_mask,
        "delta_partial",
    ]
    .to_numpy(
        dtype=float,
        copy=True,
    )
)


support_change_energy = float(
    np.sum(
        support_delta
        ** 2
    )
)


persistent_change_energy = float(
    np.sum(
        persistent_delta
        ** 2
    )
)


support_change_intensity = float(
    np.sqrt(
        support_change_energy
    )
)


persistent_change_intensity = float(
    np.sqrt(
        persistent_change_energy
    )
)


support_change_energy_share = (
    support_change_energy
    /
    total_intensity_squared

    if total_intensity_squared > 0
    else np.nan
)


persistent_change_energy_share = (
    persistent_change_energy
    /
    total_intensity_squared

    if total_intensity_squared > 0
    else np.nan
)


# ============================================================
# 22. Overall support metrics
# ============================================================

n_R3_edges = int(
    edge_table[
        "edge_R3"
    ]
    .sum()
)


n_R4_edges = int(
    edge_table[
        "edge_R4"
    ]
    .sum()
)


persistent_edges = int(
    (
        edge_table[
            "support_change"
        ]
        ==
        "Persistent"
    )
    .sum()
)


lost_edges = int(
    (
        edge_table[
            "support_change"
        ]
        ==
        "Lost"
    )
    .sum()
)


gained_edges = int(
    (
        edge_table[
            "support_change"
        ]
        ==
        "Gained"
    )
    .sum()
)


union_count = (
    persistent_edges
    +
    lost_edges
    +
    gained_edges
)


overall_jaccard = (
    persistent_edges
    /
    union_count

    if union_count > 0
    else np.nan
)


gross_support_changes = (
    lost_edges
    +
    gained_edges
)


net_edge_change = (
    gained_edges
    -
    lost_edges
)


rewiring = (
    gross_support_changes
    -
    abs(
        net_edge_change
    )
)


# ============================================================
# 23. Persistent-edge strength changes
# ============================================================

persistent_edges_df = (
    edge_table.loc[
        persistent_mask
    ]
)


n_strengthened = int(
    (
        persistent_edges_df[
            "strength_change"
        ]
        ==
        "Strengthened"
    )
    .sum()
)


n_weakened = int(
    (
        persistent_edges_df[
            "strength_change"
        ]
        ==
        "Weakened"
    )
    .sum()
)


n_signflip = int(
    (
        persistent_edges_df[
            "strength_change"
        ]
        ==
        "SignFlip"
    )
    .sum()
)


n_stable = int(
    (
        persistent_edges_df[
            "strength_change"
        ]
        ==
        "Stable"
    )
    .sum()
)


# ============================================================
# 24. Transition summary
# ============================================================

transition_summary = pd.DataFrame(
    [
        {

            "transition":
                "R3->R4",

            "R3_edges":
                n_R3_edges,

            "R4_edges":
                n_R4_edges,

            "persistent_edges":
                persistent_edges,

            "lost_edges":
                lost_edges,

            "gained_edges":
                gained_edges,

            "gross_support_changes":
                gross_support_changes,

            "net_edge_change":
                net_edge_change,

            "rewiring":
                rewiring,

            "overall_jaccard":
                overall_jaccard,

            "persistent_strengthened":
                n_strengthened,

            "persistent_weakened":
                n_weakened,

            "persistent_signflip":
                n_signflip,

            "persistent_stable":
                n_stable,

            "transition_intensity_total":
                TI_total,

            "transition_intensity_same":
                TI_same,

            "transition_intensity_cross":
                TI_cross,

            "same_intensity_share":
                same_intensity_share,

            "cross_intensity_share":
                cross_intensity_share,

            "support_change_intensity":
                support_change_intensity,

            "persistent_change_intensity":
                persistent_change_intensity,

            "support_change_energy_share":
                support_change_energy_share,

            "persistent_change_energy_share":
                persistent_change_energy_share,
        }
    ]
)


transition_summary.to_csv(
    OUTPUT_DIR
    / "R3_R4_transition_intensity_summary.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 25. Structural-role summary
# ============================================================

role_summary = (
    edge_table
    .groupby(
        "structural_role",
        observed=False,
    )
    .agg(

        n_pairs=(
            "stock_i",
            "size",
        ),

        mean_abs_delta=(
            "abs_delta_partial",
            "mean",
        ),

        total_squared_change=(
            "delta_partial",
            lambda x: float(
                np.sum(
                    np.asarray(
                        x,
                        dtype=float,
                    )
                    ** 2
                )
            ),
        ),
    )
    .reset_index()
)


role_summary.to_csv(
    OUTPUT_DIR
    / "R3_R4_structural_role_summary.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 26. Stage-3 consistency check
# ============================================================

if STAGE3_CHANGE_SUMMARY_FILE.exists():

    stage3 = (
        read_csv_with_encoding_fallback(
            STAGE3_CHANGE_SUMMARY_FILE
        )
    )


    if len(
        stage3
    ) > 0:

        stage3_row = (
            stage3.iloc[0]
        )


        print(
            "\n"
            +
            "=" * 78
        )


        print(
            "Stage 3 consistency check"
        )


        print(
            "=" * 78
        )


        comparisons = {

            "R3 edges":
                (
                    n_R3_edges,
                    stage3_row.get(
                        "edges_R3",
                        np.nan,
                    ),
                ),

            "R4 edges":
                (
                    n_R4_edges,
                    stage3_row.get(
                        "edges_R4",
                        np.nan,
                    ),
                ),

            "Lost":
                (
                    lost_edges,
                    stage3_row.get(
                        "lost_edges",
                        np.nan,
                    ),
                ),

            "Gained":
                (
                    gained_edges,
                    stage3_row.get(
                        "gained_edges",
                        np.nan,
                    ),
                ),
        }


        for (
            metric_name,
            values,
        ) in comparisons.items():

            stage5_value = (
                values[0]
            )

            stage3_value = (
                values[1]
            )


            print(
                f"{metric_name}: "
                f"Stage5={stage5_value}, "
                f"Stage3={stage3_value}"
            )


# ============================================================
# 27. Figure 1:
#     Same vs Cross transition intensity
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        7,
        5,
    )
)


ax.bar(
    [
        "同行业",
        "跨行业",
    ],
    [
        TI_same,
        TI_cross,
    ],
)


ax.set_ylabel(
    "结构转换强度",
    fontproperties=CHINESE_FONT_PROP,
)


ax.set_title(
    "R3 → R4 偏相关网络转换强度",
    fontproperties=CHINESE_FONT_PROP,
    fontsize=13,
)


# Apply Chinese font to tick labels explicitly
for tick_label in ax.get_xticklabels():

    tick_label.set_fontproperties(
        CHINESE_FONT_PROP
    )


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "R3_R4_same_cross_transition_intensity.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 28. Figure 2:
#     Support-change composition
# ============================================================

support_plot = (
    edge_table.loc[
        edge_table[
            "support_change"
        ]
        .isin(
            [
                "Lost",
                "Gained",
            ]
        )
    ]
    .groupby(
        [
            "support_change",
            "relation",
        ],
        observed=False,
    )
    .size()
    .unstack(
        fill_value=0
    )
)


for col in [
    "Same",
    "Cross",
]:

    if (
        col
        not in
        support_plot.columns
    ):

        support_plot[
            col
        ] = 0


support_plot = (
    support_plot
    .reindex(
        [
            "Lost",
            "Gained",
        ],
        fill_value=0,
    )
)


fig, ax = plt.subplots(
    figsize=(
        7,
        5,
    )
)


support_plot[
    [
        "Same",
        "Cross",
    ]
].plot(
    kind="bar",
    ax=ax,
)


ax.set_xlabel(
    "边支持集变化",
    fontproperties=CHINESE_FONT_PROP,
)


ax.set_ylabel(
    "边数量",
    fontproperties=CHINESE_FONT_PROP,
)


ax.set_title(
    "R3 → R4 同行业与跨行业边变化",
    fontproperties=CHINESE_FONT_PROP,
    fontsize=13,
)


# Replace x tick text by Chinese
ax.set_xticklabels(
    [
        "退出边",
        "新增边",
    ],
    rotation=0,
    fontproperties=CHINESE_FONT_PROP,
)


# Legend can remain English or be replaced.
legend = ax.legend(
    [
        "同行业",
        "跨行业",
    ]
)


for text in (
    legend.get_texts()
):

    text.set_fontproperties(
        CHINESE_FONT_PROP
    )


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "R3_R4_support_change_composition.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 29. Build stock labels
# ============================================================

labels = {}


for stock in STOCKS:

    if HAS_NAME:

        stock_name = (
            stock_info.at[
                stock,
                "name",
            ]
        )


        if (
            pd.isna(
                stock_name
            )
            or
            str(
                stock_name
            ).strip()
            ==
            ""
        ):

            labels[
                stock
            ] = stock


        else:

            stock_name = str(
                stock_name
            ).strip()


            if SHOW_STOCK_CODE_IN_LABEL:

                labels[
                    stock
                ] = (
                    f"{stock_name}\n"
                    f"{stock}"
                )

            else:

                labels[
                    stock
                ] = (
                    stock_name
                )


    else:

        labels[
            stock
        ] = (
            stock
        )


# ------------------------------------------------------------
# Label diagnostic
# ------------------------------------------------------------

print(
    "\n"
    +
    "=" * 78
)


print(
    "Final Network Labels"
)


print(
    "=" * 78
)


for stock in STOCKS:

    printable_label = (
        labels[
            stock
        ]
        .replace(
            "\n",
            " / ",
        )
    )


    print(
        f"{stock}: "
        f"{printable_label}"
    )


# ============================================================
# 30. Figure 3:
#     Descriptive Differential Network
# ============================================================

plot_edges = (
    union_edges
    .sort_values(
        "abs_delta_partial",
        ascending=False,
    )
    .head(
        TOP_K_DIFF_EDGES
    )
    .copy()
)


# ------------------------------------------------------------
# Full union graph for layout
# ------------------------------------------------------------

G_layout = nx.Graph()


G_layout.add_nodes_from(
    STOCKS
)


for row in (
    union_edges.itertuples()
):

    G_layout.add_edge(
        row.stock_i,
        row.stock_j,
        weight=max(
            row.abs_partial_R3,
            row.abs_partial_R4,
        ),
    )


positions = nx.spring_layout(
    G_layout,
    seed=NETWORK_LAYOUT_SEED,
    weight="weight",
)


# ------------------------------------------------------------
# Difference graph only for TOP_K changes
# ------------------------------------------------------------

G_diff = nx.Graph()


G_diff.add_nodes_from(
    STOCKS
)


for row in (
    plot_edges.itertuples()
):

    G_diff.add_edge(
        row.stock_i,
        row.stock_j,
        abs_delta=(
            row.abs_delta_partial
        ),
        support_change=(
            row.support_change
        ),
    )


fig, ax = plt.subplots(
    figsize=(
        13,
        10,
    )
)


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

nx.draw_networkx_nodes(
    G_diff,
    positions,
    node_size=NODE_SIZE,
    ax=ax,
)


# ============================================================
# IMPORTANT FIX:
#
# Do NOT use:
#
#     nx.draw_networkx_labels(...)
#
# Instead explicitly render Chinese labels through Matplotlib
# using FontProperties.
# ============================================================

for stock in STOCKS:

    x, y = (
        positions[
            stock
        ]
    )


    ax.annotate(
        labels[
            stock
        ],

        xy=(
            x,
            y,
        ),

        # Display label slightly above node.
        xytext=(
            0,
            10,
        ),

        textcoords=(
            "offset points"
        ),

        ha="center",

        va="bottom",

        fontsize=(
            NODE_LABEL_FONT_SIZE
        ),

        fontproperties=(
            CHINESE_FONT_PROP
        ),

        bbox={
            "boxstyle":
                "round,pad=0.20",

            "facecolor":
                "white",

            "edgecolor":
                "none",

            "alpha":
                0.82,
        },

        zorder=20,
    )


# ------------------------------------------------------------
# Edge widths according to |Delta partial|
# ------------------------------------------------------------

if len(
    plot_edges
) > 0:

    max_change = float(
        plot_edges[
            "abs_delta_partial"
        ]
        .max()
    )

else:

    max_change = 1.0


if (
    pd.isna(
        max_change
    )
    or
    max_change <= 0
):

    max_change = 1.0


# ------------------------------------------------------------
# Different line styles for support categories
#
# Lost       -> dashed
# Gained     -> dotted
# Persistent -> solid
# ------------------------------------------------------------

for (
    support_type,
    line_style,
) in [

    (
        "Lost",
        "dashed",
    ),

    (
        "Gained",
        "dotted",
    ),

    (
        "Persistent",
        "solid",
    ),
]:

    selected = (
        plot_edges.loc[
            plot_edges[
                "support_change"
            ]
            ==
            support_type
        ]
    )


    edge_list = list(
        zip(
            selected[
                "stock_i"
            ],
            selected[
                "stock_j"
            ],
        )
    )


    widths = (
        1.0
        +
        4.0
        *
        selected[
            "abs_delta_partial"
        ]
        /
        max_change
    ).tolist()


    if len(
        edge_list
    ) > 0:

        nx.draw_networkx_edges(
            G_diff,
            positions,
            edgelist=edge_list,
            width=widths,
            style=line_style,
            ax=ax,
        )


ax.set_title(
    (
        "R3 → R4 描述性差异网络\n"
        f"|偏相关变化|最大的前 "
        f"{TOP_K_DIFF_EDGES} 条边"
    ),
    fontproperties=CHINESE_FONT_PROP,
    fontsize=14,
)


ax.axis(
    "off"
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "R3_R4_descriptive_differential_network.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 31. Final console summaries
# ============================================================

print(
    "\n"
    +
    "=" * 78
)


print(
    "R3 -> R4 support changes"
)


print(
    "=" * 78
)


print(
    f"R3 edges        : "
    f"{n_R3_edges}"
)


print(
    f"R4 edges        : "
    f"{n_R4_edges}"
)


print(
    f"Persistent      : "
    f"{persistent_edges}"
)


print(
    f"Lost            : "
    f"{lost_edges}"
)


print(
    f"Gained          : "
    f"{gained_edges}"
)


print(
    f"Jaccard         : "
    f"{overall_jaccard:.4f}"
)


print(
    f"Gross changes   : "
    f"{gross_support_changes}"
)


print(
    f"Net edge change : "
    f"{net_edge_change}"
)


print(
    f"Rewiring        : "
    f"{rewiring}"
)


print(
    "\n"
    +
    "=" * 78
)


print(
    "Persistent-edge strength changes"
)


print(
    "=" * 78
)


print(
    f"Strengthened    : "
    f"{n_strengthened}"
)


print(
    f"Weakened        : "
    f"{n_weakened}"
)


print(
    f"Sign flip       : "
    f"{n_signflip}"
)


print(
    f"Stable          : "
    f"{n_stable}"
)


print(
    "\n"
    +
    "=" * 78
)


print(
    "Same / Cross decomposition"
)


print(
    "=" * 78
)


print(
    relation_summary.to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 78
)


print(
    "Transition intensity"
)


print(
    "=" * 78
)


print(
    f"Total TI        : "
    f"{TI_total:.6f}"
)


print(
    f"Same TI         : "
    f"{TI_same:.6f}"
)


print(
    f"Cross TI        : "
    f"{TI_cross:.6f}"
)


print(
    f"Same TI share   : "
    f"{same_intensity_share:.2%}"
)


print(
    f"Cross TI share  : "
    f"{cross_intensity_share:.2%}"
)


print(
    f"Support-change energy share    : "
    f"{support_change_energy_share:.2%}"
)


print(
    f"Persistent-change energy share : "
    f"{persistent_change_energy_share:.2%}"
)


print(
    "\n"
    +
    "=" * 78
)


print(
    "Stage 5 completed successfully"
)


print(
    "=" * 78
)


print(
    "\nChinese font used:"
)


print(
    CHINESE_FONT_NAME
)


print(
    "\nOutput directory:"
)


print(
    OUTPUT_DIR.resolve()
)


print(
    "\nMain output files:"
)


main_outputs = [

    "R3_R4_delta_partial_matrix.csv",

    "R3_R4_differential_edges_full.csv",

    "R3_R4_differential_edges_union.csv",

    "R3_R4_top_differential_edges.csv",

    "R3_R4_same_cross_decomposition.csv",

    "R3_R4_transition_intensity_summary.csv",

    "R3_R4_structural_role_summary.csv",

    "R3_R4_same_cross_transition_intensity.png",

    "R3_R4_support_change_composition.png",

    "R3_R4_descriptive_differential_network.png",
]


for filename in main_outputs:

    print(
        f"  - {filename}"
    )