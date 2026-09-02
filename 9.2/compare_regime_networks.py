from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 0. Project directory
# ============================================================

# The script is assumed to be placed in:
#
#   stock-network-lab/
#       compare_regime_networks.py
#
# If your directory structure is different, modify only the
# path definitions in Section 1.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 1. Paths
# ============================================================

# Stage 2 outputs:
#
# regime_networks/
#   R1_edges.csv
#   R1_partial.csv
#   ...
#   R5_edges.csv
#
REGIME_NETWORK_DIR = (
    BASE_DIR
    / "regime_networks"
)


# Authoritative stock metadata
STOCK_INFO_FILE = (
    BASE_DIR
    / "stock_network"
    / "data"
    / "processed"
    / "stock_info.csv"
)


# Stage 3 outputs
OUTPUT_DIR = (
    BASE_DIR
    / "regime_network_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Candidate locations of the ORIGINAL Rolling-Regime summary.
#
# The code will automatically use the first one it finds.
# If none exists, Stage 3 independent-Regime analysis will
# still run normally; only Rolling-vs-Independent comparison
# will be skipped.
# ------------------------------------------------------------

ROLLING_SUMMARY_CANDIDATES = [
    BASE_DIR
    / "network_regime_summary.csv",

    BASE_DIR
    / "stock_network"
    / "data"
    / "processed"
    / "network_regime_summary.csv",

    BASE_DIR
    / "stock_network"
    / "results"
    / "network_regime_summary.csv",

    BASE_DIR
    / "results"
    / "network_regime_summary.csv",
]


# ============================================================
# 2. Main settings
# ============================================================

REGIMES = [
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
]

R3 = "R3"
R4 = "R4"


# Numerical tolerance when deciding whether a common edge
# is "Strengthened", "Weakened", or effectively "Stable".
STRENGTH_TOL = 1e-10


# ============================================================
# 3. Utility functions
# ============================================================

def normalize_code(x) -> str:
    """
    Normalize a stock code to a 6-digit string.

    Examples
    --------
    1        -> "000001"
    "1"      -> "000001"
    "1.0"    -> "000001"
    "600036" -> "600036"
    """

    x = str(x).strip()

    if x.endswith(".0"):
        x = x[:-2]

    if x.isdigit():
        x = x.zfill(6)

    return x


def canonical_pair(
    stock_i: str,
    stock_j: str,
) -> tuple[str, str]:
    """
    Return a canonical undirected edge.

    IMPORTANT:
    Only stock codes are reordered here.

    Stock name / industry information is NOT carried together
    during reordering. Metadata is remapped later from the
    authoritative stock_info.csv.
    """

    a = normalize_code(stock_i)
    b = normalize_code(stock_j)

    return tuple(
        sorted(
            (a, b)
        )
    )


def find_first_existing(
    candidates: list[Path],
) -> Path | None:
    """
    Return the first existing file from a list of candidates.
    """

    for path in candidates:

        if path.exists():
            return path

    return None


def normalize_regime_label(x) -> str:
    """
    Convert regime representations such as:

        3
        3.0
        "3"
        "R3"

    into

        "R3"
    """

    x = str(x).strip()

    if x.upper().startswith("R"):
        return x.upper()

    try:
        return f"R{int(float(x))}"

    except ValueError:
        return x


def direction(
    delta: float,
    tol: float = 1e-12,
) -> str:
    """
    Describe the sign of a change.
    """

    if delta > tol:
        return "increase"

    if delta < -tol:
        return "decrease"

    return "stable"


# ============================================================
# 4. Load authoritative stock metadata
# ============================================================

if not STOCK_INFO_FILE.exists():

    raise FileNotFoundError(
        "\nstock_info.csv was not found.\n"
        f"Expected location:\n{STOCK_INFO_FILE}\n"
    )


# Read as strings first so that 000001 is not converted to 1.
stock_info_raw = pd.read_csv(
    STOCK_INFO_FILE,
    dtype=str,
)


# ------------------------------------------------------------
# Support several possible metadata column names
# ------------------------------------------------------------

COLUMN_ALIASES = {
    "code": [
        "code",
        "股票代码",
        "stock_code",
        "ticker",
    ],

    "name": [
        "name",
        "股票名称",
        "stock_name",
    ],

    "industry": [
        "industry",
        "行业",
        "industry_name",
    ],
}


def find_column(
    df: pd.DataFrame,
    aliases: list[str],
) -> str | None:

    for col in aliases:

        if col in df.columns:
            return col

    return None


code_col = find_column(
    stock_info_raw,
    COLUMN_ALIASES["code"],
)

industry_col = find_column(
    stock_info_raw,
    COLUMN_ALIASES["industry"],
)

name_col = find_column(
    stock_info_raw,
    COLUMN_ALIASES["name"],
)


if code_col is None:

    raise ValueError(
        "Cannot find a stock-code column in stock_info.csv.\n"
        f"Available columns:\n{list(stock_info_raw.columns)}"
    )


if industry_col is None:

    raise ValueError(
        "Cannot find an industry column in stock_info.csv.\n"
        f"Available columns:\n{list(stock_info_raw.columns)}"
    )


rename_map = {
    code_col: "code",
    industry_col: "industry",
}


if name_col is not None:
    rename_map[name_col] = "name"


stock_info = (
    stock_info_raw
    .rename(
        columns=rename_map
    )
    .copy()
)


stock_info["code"] = (
    stock_info["code"]
    .map(normalize_code)
)


stock_info = (
    stock_info
    .drop_duplicates(
        subset="code"
    )
    .set_index("code")
)


HAS_NAME = (
    "name"
    in stock_info.columns
)


print("=" * 78)
print("Stock metadata")
print("=" * 78)

print(
    f"Metadata file: {STOCK_INFO_FILE}"
)

print(
    f"Number of stock records: "
    f"{len(stock_info)}"
)


# ============================================================
# 5. Recover the FULL node set from Stage-2 matrices
# ============================================================

# We should NOT infer the 15-stock universe from edge lists,
# because an isolated stock would not appear in any edge file.
#
# Instead, use R1_partial.csv, whose columns contain all nodes.

REFERENCE_PARTIAL_FILE = (
    REGIME_NETWORK_DIR
    / "R1_partial.csv"
)


if not REFERENCE_PARTIAL_FILE.exists():

    raise FileNotFoundError(
        "\nCannot find R1_partial.csv.\n"
        "Stage 2 must be completed before Stage 3.\n"
        f"Expected:\n{REFERENCE_PARTIAL_FILE}"
    )


reference_partial = pd.read_csv(
    REFERENCE_PARTIAL_FILE,
    index_col=0,
)


NODE_CODES = [
    normalize_code(col)
    for col
    in reference_partial.columns
]


if len(NODE_CODES) != len(set(NODE_CODES)):

    raise ValueError(
        "Duplicated stock codes detected "
        "in R1_partial.csv."
    )


p = len(NODE_CODES)


print("\n" + "=" * 78)
print("Node universe")
print("=" * 78)

print(
    f"Number of nodes p = {p}"
)

print(
    "Stocks:"
)

print(
    NODE_CODES
)


if p != 15:

    print(
        "\nWARNING: "
        f"Expected 15 stocks, but found p={p}."
    )


# ------------------------------------------------------------
# Verify all nodes exist in stock_info.csv
# ------------------------------------------------------------

missing_metadata_codes = (
    set(NODE_CODES)
    -
    set(stock_info.index)
)


if missing_metadata_codes:

    raise ValueError(
        "\nSome Stage-2 stock codes are absent "
        "from stock_info.csv:\n"
        f"{sorted(missing_metadata_codes)}"
    )


# ------------------------------------------------------------
# Verify R1-R5 use the same node set
# ------------------------------------------------------------

for regime in REGIMES:

    partial_file = (
        REGIME_NETWORK_DIR
        / f"{regime}_partial.csv"
    )

    if not partial_file.exists():

        raise FileNotFoundError(
            f"Missing Stage-2 file: "
            f"{partial_file}"
        )


    tmp = pd.read_csv(
        partial_file,
        index_col=0,
    )

    tmp_codes = [
        normalize_code(col)
        for col
        in tmp.columns
    ]


    if set(tmp_codes) != set(NODE_CODES):

        raise ValueError(
            f"{regime} does not use the same "
            "stock universe as R1."
        )


# ============================================================
# 6. Calculate possible Same/Cross-industry pairs
# ============================================================

same_possible = 0
cross_possible = 0


for stock_i, stock_j in combinations(
    NODE_CODES,
    2,
):

    industry_i = (
        stock_info.at[
            stock_i,
            "industry"
        ]
    )

    industry_j = (
        stock_info.at[
            stock_j,
            "industry"
        ]
    )


    if industry_i == industry_j:
        same_possible += 1

    else:
        cross_possible += 1


total_possible = (
    same_possible
    +
    cross_possible
)


expected_total = (
    p
    *
    (p - 1)
    // 2
)


if total_possible != expected_total:

    raise RuntimeError(
        "Possible-pair calculation is inconsistent."
    )


print("\n" + "=" * 78)
print("Possible stock pairs")
print("=" * 78)

print(
    f"Total possible pairs       : "
    f"{total_possible}"
)

print(
    f"Same-industry possible     : "
    f"{same_possible}"
)

print(
    f"Cross-industry possible    : "
    f"{cross_possible}"
)


# ============================================================
# 7. Load one Stage-2 edge list
# ============================================================

def load_regime_edges(
    regime: str,
) -> pd.DataFrame:
    """
    Load one Stage-2 edge file and remap authoritative metadata.
    """

    edge_file = (
        REGIME_NETWORK_DIR
        / f"{regime}_edges.csv"
    )


    if not edge_file.exists():

        raise FileNotFoundError(
            f"Missing Stage-2 edge file:\n"
            f"{edge_file}"
        )


    edges = pd.read_csv(
        edge_file,
        dtype={
            "stock_i": str,
            "stock_j": str,
        },
    )


    required_cols = {
        "stock_i",
        "stock_j",
        "partial_corr",
    }


    missing_cols = (
        required_cols
        -
        set(edges.columns)
    )


    if missing_cols:

        raise ValueError(
            f"{edge_file} is missing columns:\n"
            f"{missing_cols}"
        )


    # --------------------------------------------------------
    # Normalize codes
    # --------------------------------------------------------

    edges["stock_i"] = (
        edges["stock_i"]
        .map(normalize_code)
    )

    edges["stock_j"] = (
        edges["stock_j"]
        .map(normalize_code)
    )


    # --------------------------------------------------------
    # Canonicalize undirected edges
    # --------------------------------------------------------

    pairs = [
        canonical_pair(i, j)
        for i, j
        in zip(
            edges["stock_i"],
            edges["stock_j"],
        )
    ]


    edges["stock_i"] = [
        pair[0]
        for pair in pairs
    ]

    edges["stock_j"] = [
        pair[1]
        for pair in pairs
    ]


    # --------------------------------------------------------
    # Check duplicated undirected edges
    # --------------------------------------------------------

    duplicate_mask = (
        edges.duplicated(
            subset=[
                "stock_i",
                "stock_j",
            ],
            keep=False,
        )
    )


    if duplicate_mask.any():

        duplicated_pairs = (
            edges.loc[
                duplicate_mask,
                [
                    "stock_i",
                    "stock_j",
                    "partial_corr",
                ],
            ]
            .sort_values(
                [
                    "stock_i",
                    "stock_j",
                ]
            )
        )


        raise ValueError(
            f"\nDuplicated undirected edges found "
            f"in {regime}_edges.csv:\n"
            f"{duplicated_pairs.to_string(index=False)}"
        )


    # --------------------------------------------------------
    # Verify all codes belong to the 15-stock universe
    # --------------------------------------------------------

    edge_codes = (
        set(edges["stock_i"])
        |
        set(edges["stock_j"])
    )


    unknown_codes = (
        edge_codes
        -
        set(NODE_CODES)
    )


    if unknown_codes:

        raise ValueError(
            f"{regime} contains unknown stock codes:\n"
            f"{sorted(unknown_codes)}"
        )


    # --------------------------------------------------------
    # Remap authoritative metadata
    # --------------------------------------------------------

    edges["industry_i"] = (
        edges["stock_i"]
        .map(
            stock_info["industry"]
        )
    )

    edges["industry_j"] = (
        edges["stock_j"]
        .map(
            stock_info["industry"]
        )
    )


    if HAS_NAME:

        edges["name_i"] = (
            edges["stock_i"]
            .map(
                stock_info["name"]
            )
        )

        edges["name_j"] = (
            edges["stock_j"]
            .map(
                stock_info["name"]
            )
        )


    edges["same_industry"] = (
        edges["industry_i"]
        ==
        edges["industry_j"]
    )


    edges["relation"] = np.where(
        edges["same_industry"],
        "Same",
        "Cross",
    )


    edges["partial_corr"] = pd.to_numeric(
        edges["partial_corr"],
        errors="raise",
    )


    edges["abs_partial_corr"] = (
        edges["partial_corr"]
        .abs()
    )


    edges["regime"] = regime


    return (
        edges
        .sort_values(
            [
                "stock_i",
                "stock_j",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# 8. Load all five Independent-Regime edge lists
# ============================================================

all_regime_edges = {
    regime:
        load_regime_edges(regime)

    for regime
    in REGIMES
}


# ============================================================
# 9. Task 1-2:
#    Build Independent-Regime industry summaries
# ============================================================

summary_rows = []


for regime in REGIMES:

    edges = (
        all_regime_edges[
            regime
        ]
        .copy()
    )


    n_edges = len(edges)


    same_edges = int(
        edges[
            "same_industry"
        ]
        .sum()
    )


    cross_edges = (
        n_edges
        -
        same_edges
    )


    # --------------------------------------------------------
    # Network density
    # --------------------------------------------------------

    density = (
        n_edges
        / total_possible
        if total_possible > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Composition conditional on selected edges
    # --------------------------------------------------------

    same_edge_ratio = (
        same_edges
        / n_edges
        if n_edges > 0
        else np.nan
    )


    cross_edge_ratio = (
        cross_edges
        / n_edges
        if n_edges > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Coverage relative to POSSIBLE Same/Cross pairs
    #
    # This is important because possible Same and Cross pairs
    # have very different denominators.
    # --------------------------------------------------------

    same_pair_coverage = (
        same_edges
        / same_possible
        if same_possible > 0
        else np.nan
    )


    cross_pair_coverage = (
        cross_edges
        / cross_possible
        if cross_possible > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Overall selected-edge strength
    # --------------------------------------------------------

    mean_abs_partial = (
        edges[
            "abs_partial_corr"
        ]
        .mean()

        if n_edges > 0
        else np.nan
    )


    median_abs_partial = (
        edges[
            "abs_partial_corr"
        ]
        .median()

        if n_edges > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Same/Cross selected-edge strength
    # --------------------------------------------------------

    same_partials = (
        edges.loc[
            edges["same_industry"],
            "abs_partial_corr",
        ]
    )


    cross_partials = (
        edges.loc[
            ~edges["same_industry"],
            "abs_partial_corr",
        ]
    )


    same_mean_abs_partial = (
        same_partials.mean()
        if len(same_partials) > 0
        else np.nan
    )


    cross_mean_abs_partial = (
        cross_partials.mean()
        if len(cross_partials) > 0
        else np.nan
    )


    same_cross_strength_ratio = (
        same_mean_abs_partial
        /
        cross_mean_abs_partial

        if (
            pd.notna(
                same_mean_abs_partial
            )
            and
            pd.notna(
                cross_mean_abs_partial
            )
            and
            cross_mean_abs_partial != 0
        )

        else np.nan
    )


    summary_rows.append(
        {
            "regime":
                regime,

            "n_edges":
                n_edges,

            "density":
                density,

            "same_edges":
                same_edges,

            "cross_edges":
                cross_edges,

            "same_edge_ratio":
                same_edge_ratio,

            "cross_edge_ratio":
                cross_edge_ratio,

            "same_possible_pairs":
                same_possible,

            "cross_possible_pairs":
                cross_possible,

            "same_pair_coverage":
                same_pair_coverage,

            "cross_pair_coverage":
                cross_pair_coverage,

            "mean_abs_partial":
                mean_abs_partial,

            "median_abs_partial":
                median_abs_partial,

            "same_mean_abs_partial":
                same_mean_abs_partial,

            "cross_mean_abs_partial":
                cross_mean_abs_partial,

            "same_cross_strength_ratio":
                same_cross_strength_ratio,
        }
    )


independent_summary = pd.DataFrame(
    summary_rows
)


independent_summary.to_csv(
    OUTPUT_DIR
    / "independent_regime_industry_summary.csv",

    index=False,
    encoding="utf-8-sig",
)


print("\n" + "=" * 78)
print("Independent Regime industry structure")
print("=" * 78)

print(
    independent_summary
    .to_string(
        index=False
    )
)


# ============================================================
# 10. Task 3:
#     R3 -> R4 edge-set comparison
#
# IMPORTANT FIX:
# We now use a REAL MultiIndex:
#
#   index level 1 = stock_i
#   index level 2 = stock_j
#
# instead of an ordinary Index containing Python tuples.
# ============================================================

r3_edges = (
    all_regime_edges[
        R3
    ]
    .copy()
)


r4_edges = (
    all_regime_edges[
        R4
    ]
    .copy()
)


# ------------------------------------------------------------
# Final duplicate check before MultiIndex construction
# ------------------------------------------------------------

for regime, edges in [
    (R3, r3_edges),
    (R4, r4_edges),
]:

    duplicated = edges.duplicated(
        subset=[
            "stock_i",
            "stock_j",
        ],
        keep=False,
    )


    if duplicated.any():

        raise ValueError(
            f"Duplicated pairs remain in {regime}."
        )


# ------------------------------------------------------------
# TRUE MultiIndex
# ------------------------------------------------------------

r3_map = (
    r3_edges
    .set_index(
        [
            "stock_i",
            "stock_j",
        ]
    )
    .sort_index()
)


r4_map = (
    r4_edges
    .set_index(
        [
            "stock_i",
            "stock_j",
        ]
    )
    .sort_index()
)


# Convert MultiIndex entries into ordinary Python tuples
r3_set = set(
    r3_map.index.to_list()
)


r4_set = set(
    r4_map.index.to_list()
)


common_edges = (
    r3_set
    &
    r4_set
)


lost_edges = (
    r3_set
    -
    r4_set
)


gained_edges = (
    r4_set
    -
    r3_set
)


union_edges = (
    r3_set
    |
    r4_set
)


jaccard = (
    len(common_edges)
    /
    len(union_edges)

    if len(union_edges) > 0
    else np.nan
)


print("\n" + "=" * 78)
print("Independent R3 -> R4 edge-set comparison")
print("=" * 78)

print(
    f"R3 edges     : {len(r3_set)}"
)

print(
    f"R4 edges     : {len(r4_set)}"
)

print(
    f"Common edges : {len(common_edges)}"
)

print(
    f"Lost edges   : {len(lost_edges)}"
)

print(
    f"Gained edges : {len(gained_edges)}"
)

print(
    f"Jaccard      : {jaccard:.6f}"
)


# ============================================================
# 11. Build complete R3 -> R4 edge-change table
# ============================================================

change_rows = []


for stock_i, stock_j in sorted(
    union_edges
):

    key = (
        stock_i,
        stock_j,
    )


    in_r3 = (
        key
        in r3_set
    )


    in_r4 = (
        key
        in r4_set
    )


    # --------------------------------------------------------
    # FIX:
    # Scalar access from the MultiIndex uses .at[]
    #
    # This replaces the problematic:
    #
    #     r3_map.loc[key, "partial_corr"]
    #
    # --------------------------------------------------------

    if in_r3:

        partial_r3 = float(
            r3_map.at[
                key,
                "partial_corr",
            ]
        )

    else:

        # "0" means the penalized estimated network did not
        # retain this edge in R3.
        partial_r3 = 0.0


    if in_r4:

        partial_r4 = float(
            r4_map.at[
                key,
                "partial_corr",
            ]
        )

    else:

        partial_r4 = 0.0


    abs_r3 = abs(
        partial_r3
    )


    abs_r4 = abs(
        partial_r4
    )


    delta_partial = (
        partial_r4
        -
        partial_r3
    )


    delta_abs_partial = (
        abs_r4
        -
        abs_r3
    )


    # --------------------------------------------------------
    # Support / strength classification
    # --------------------------------------------------------

    if (
        in_r3
        and
        not in_r4
    ):

        change_type = "Lost"


    elif (
        not in_r3
        and
        in_r4
    ):

        change_type = "Gained"


    else:

        # The edge is selected in BOTH R3 and R4.

        # Check sign reversal first.
        if (
            partial_r3
            *
            partial_r4
            < 0
        ):

            change_type = "SignFlip"


        elif (
            delta_abs_partial
            >
            STRENGTH_TOL
        ):

            change_type = "Strengthened"


        elif (
            delta_abs_partial
            <
            -STRENGTH_TOL
        ):

            change_type = "Weakened"


        else:

            change_type = "Stable"


    # --------------------------------------------------------
    # Authoritative metadata
    # --------------------------------------------------------

    industry_i = (
        stock_info.at[
            stock_i,
            "industry",
        ]
    )


    industry_j = (
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
        else "Cross"
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

        "same_industry":
            same_industry,

        "relation":
            relation,

        "edge_R3":
            int(in_r3),

        "edge_R4":
            int(in_r4),

        "partial_R3":
            partial_r3,

        "partial_R4":
            partial_r4,

        "abs_partial_R3":
            abs_r3,

        "abs_partial_R4":
            abs_r4,

        "delta_partial":
            delta_partial,

        "delta_abs_partial":
            delta_abs_partial,

        "abs_delta_partial":
            abs(
                delta_partial
            ),

        "change_type":
            change_type,
    }


    if HAS_NAME:

        row["name_i"] = (
            stock_info.at[
                stock_i,
                "name",
            ]
        )

        row["name_j"] = (
            stock_info.at[
                stock_j,
                "name",
            ]
        )


    change_rows.append(
        row
    )


change_table = pd.DataFrame(
    change_rows
)


# ------------------------------------------------------------
# Meaningful ordering in the output CSV
# ------------------------------------------------------------

change_order = {
    "Lost": 1,
    "Gained": 2,
    "SignFlip": 3,
    "Strengthened": 4,
    "Weakened": 5,
    "Stable": 6,
}


change_table[
    "_change_order"
] = (
    change_table[
        "change_type"
    ]
    .map(change_order)
)


change_table = (
    change_table
    .sort_values(
        [
            "_change_order",
            "abs_delta_partial",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .drop(
        columns=[
            "_change_order"
        ]
    )
    .reset_index(drop=True)
)


change_table.to_csv(
    OUTPUT_DIR
    / "R3_R4_independent_edge_changes.csv",

    index=False,
    encoding="utf-8-sig",
)


print("\n" + "=" * 78)
print("R3 -> R4 edge-change types")
print("=" * 78)

print(
    change_table[
        "change_type"
    ]
    .value_counts()
)


# ============================================================
# 12. Change type x Same/Cross summary
# ============================================================

change_industry_summary = (
    change_table
    .groupby(
        [
            "change_type",
            "relation",
        ],
        observed=False,
    )
    .size()
    .unstack(
        fill_value=0
    )
    .reset_index()
)


# Ensure both columns exist even if one relation is absent.
for col in [
    "Same",
    "Cross",
]:

    if col not in change_industry_summary.columns:

        change_industry_summary[col] = 0


change_industry_summary[
    "Total"
] = (
    change_industry_summary["Same"]
    +
    change_industry_summary["Cross"]
)


change_industry_summary.to_csv(
    OUTPUT_DIR
    / "R3_R4_edge_type_industry_summary.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 13. Build one-row R3 -> R4 structural summary
# ============================================================

def count_change(
    change_type: str,
    relation: str | None = None,
) -> int:
    """
    Count edge changes by type and optionally Same/Cross relation.
    """

    mask = (
        change_table[
            "change_type"
        ]
        ==
        change_type
    )


    if relation is not None:

        mask = (
            mask
            &
            (
                change_table[
                    "relation"
                ]
                ==
                relation
            )
        )


    return int(
        mask.sum()
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


lost_same = count_change(
    "Lost",
    "Same",
)


lost_cross = count_change(
    "Lost",
    "Cross",
)


gained_same = count_change(
    "Gained",
    "Same",
)


gained_cross = count_change(
    "Gained",
    "Cross",
)


strengthened_same = count_change(
    "Strengthened",
    "Same",
)


strengthened_cross = count_change(
    "Strengthened",
    "Cross",
)


weakened_same = count_change(
    "Weakened",
    "Same",
)


weakened_cross = count_change(
    "Weakened",
    "Cross",
)


signflip_same = count_change(
    "SignFlip",
    "Same",
)


signflip_cross = count_change(
    "SignFlip",
    "Cross",
)


# ------------------------------------------------------------
# Gross / Net / Rewiring
# ------------------------------------------------------------

gross_change = (
    n_lost
    +
    n_gained
)


net_edge_change = (
    n_gained
    -
    n_lost
)


rewiring = (
    gross_change
    -
    abs(
        net_edge_change
    )
)


cross_share_of_lost = (
    lost_cross
    /
    n_lost

    if n_lost > 0
    else np.nan
)


same_share_of_lost = (
    lost_same
    /
    n_lost

    if n_lost > 0
    else np.nan
)


# ------------------------------------------------------------
# R3/R4 summary rows
# ------------------------------------------------------------

summary_indexed = (
    independent_summary
    .set_index(
        "regime"
    )
)


r3_summary = (
    summary_indexed.loc[
        R3
    ]
)


r4_summary = (
    summary_indexed.loc[
        R4
    ]
)


r3_r4_change_summary = pd.DataFrame(
    [
        {
            "transition":
                "R3->R4",

            # ---------------------------------
            # Network size
            # ---------------------------------

            "edges_R3":
                int(
                    r3_summary[
                        "n_edges"
                    ]
                ),

            "edges_R4":
                int(
                    r4_summary[
                        "n_edges"
                    ]
                ),

            "delta_edges":
                int(
                    r4_summary[
                        "n_edges"
                    ]
                    -
                    r3_summary[
                        "n_edges"
                    ]
                ),

            # ---------------------------------
            # Edge support
            # ---------------------------------

            "common_edges":
                n_common,

            "lost_edges":
                n_lost,

            "gained_edges":
                n_gained,

            "gross_changes":
                gross_change,

            "net_edge_change":
                net_edge_change,

            "rewiring":
                rewiring,

            "jaccard":
                jaccard,

            # ---------------------------------
            # Lost/Gained by industry relation
            # ---------------------------------

            "lost_same":
                lost_same,

            "lost_cross":
                lost_cross,

            "same_share_of_lost":
                same_share_of_lost,

            "cross_share_of_lost":
                cross_share_of_lost,

            "gained_same":
                gained_same,

            "gained_cross":
                gained_cross,

            # ---------------------------------
            # Strength changes
            # ---------------------------------

            "strengthened_same":
                strengthened_same,

            "strengthened_cross":
                strengthened_cross,

            "weakened_same":
                weakened_same,

            "weakened_cross":
                weakened_cross,

            "signflip_same":
                signflip_same,

            "signflip_cross":
                signflip_cross,

            # ---------------------------------
            # Same/Cross composition
            # ---------------------------------

            "same_edges_R3":
                int(
                    r3_summary[
                        "same_edges"
                    ]
                ),

            "same_edges_R4":
                int(
                    r4_summary[
                        "same_edges"
                    ]
                ),

            "delta_same_edges":
                int(
                    r4_summary[
                        "same_edges"
                    ]
                    -
                    r3_summary[
                        "same_edges"
                    ]
                ),

            "cross_edges_R3":
                int(
                    r3_summary[
                        "cross_edges"
                    ]
                ),

            "cross_edges_R4":
                int(
                    r4_summary[
                        "cross_edges"
                    ]
                ),

            "delta_cross_edges":
                int(
                    r4_summary[
                        "cross_edges"
                    ]
                    -
                    r3_summary[
                        "cross_edges"
                    ]
                ),

            # ---------------------------------
            # Same ratio
            # ---------------------------------

            "same_ratio_R3":
                r3_summary[
                    "same_edge_ratio"
                ],

            "same_ratio_R4":
                r4_summary[
                    "same_edge_ratio"
                ],

            "delta_same_ratio":
                (
                    r4_summary[
                        "same_edge_ratio"
                    ]
                    -
                    r3_summary[
                        "same_edge_ratio"
                    ]
                ),

            # ---------------------------------
            # Pair coverage
            # ---------------------------------

            "same_coverage_R3":
                r3_summary[
                    "same_pair_coverage"
                ],

            "same_coverage_R4":
                r4_summary[
                    "same_pair_coverage"
                ],

            "cross_coverage_R3":
                r3_summary[
                    "cross_pair_coverage"
                ],

            "cross_coverage_R4":
                r4_summary[
                    "cross_pair_coverage"
                ],

            # ---------------------------------
            # Strength
            # ---------------------------------

            "mean_abs_partial_R3":
                r3_summary[
                    "mean_abs_partial"
                ],

            "mean_abs_partial_R4":
                r4_summary[
                    "mean_abs_partial"
                ],

            "delta_mean_abs_partial":
                (
                    r4_summary[
                        "mean_abs_partial"
                    ]
                    -
                    r3_summary[
                        "mean_abs_partial"
                    ]
                ),

            "same_strength_R3":
                r3_summary[
                    "same_mean_abs_partial"
                ],

            "same_strength_R4":
                r4_summary[
                    "same_mean_abs_partial"
                ],

            "cross_strength_R3":
                r3_summary[
                    "cross_mean_abs_partial"
                ],

            "cross_strength_R4":
                r4_summary[
                    "cross_mean_abs_partial"
                ],
        }
    ]
)


r3_r4_change_summary.to_csv(
    OUTPUT_DIR
    / "R3_R4_independent_change_summary.csv",

    index=False,
    encoding="utf-8-sig",
)


print("\n" + "=" * 78)
print("Independent R3 -> R4 structural summary")
print("=" * 78)

print(
    r3_r4_change_summary
    .to_string(
        index=False
    )
)


# ============================================================
# 14. Task 4:
#     Rolling vs Independent R3 -> R4 consistency
# ============================================================

rolling_summary_file = (
    find_first_existing(
        ROLLING_SUMMARY_CANDIDATES
    )
)


if rolling_summary_file is None:

    print("\n" + "=" * 78)
    print("Rolling comparison skipped")
    print("=" * 78)

    print(
        "No network_regime_summary.csv was found "
        "in the candidate locations."
    )

    print(
        "Independent-Regime Stage 3 outputs "
        "have still been generated."
    )


else:

    print("\n" + "=" * 78)
    print("Rolling Regime summary found")
    print("=" * 78)

    print(
        rolling_summary_file
    )


    rolling = pd.read_csv(
        rolling_summary_file
    )


    if "regime" not in rolling.columns:

        raise ValueError(
            "Rolling summary must contain "
            "a 'regime' column."
        )


    rolling["regime"] = (
        rolling["regime"]
        .map(
            normalize_regime_label
        )
    )


    rolling = (
        rolling
        .set_index(
            "regime"
        )
    )


    if (
        R3 not in rolling.index
        or
        R4 not in rolling.index
    ):

        raise ValueError(
            "Rolling summary does not contain "
            "both R3 and R4."
        )


    # --------------------------------------------------------
    # Flexible column-name lookup
    # --------------------------------------------------------

    ROLLING_COLUMN_ALIASES = {
        "edge_count": [
            "mean_edge_count",
            "edge_count",
            "mean_edges",
        ],

        "same_edges": [
            "mean_same_edges",
            "same_edges",
            "mean_same_industry_edges",
        ],

        "cross_edges": [
            "mean_cross_edges",
            "cross_edges",
            "mean_cross_industry_edges",
        ],

        "same_ratio": [
            "mean_same_industry_ratio",
            "same_industry_ratio",
            "mean_same_edge_ratio",
            "same_edge_ratio",
        ],

        "mean_abs_partial": [
            "mean_abs_partial",
            "mean_selected_abs_partial",
        ],
    }


    def get_rolling_column(
        aliases: list[str],
    ) -> str | None:

        for col in aliases:

            if col in rolling.columns:
                return col

        return None


    rolling_cols = {
        metric:
            get_rolling_column(
                aliases
            )

        for metric, aliases
        in ROLLING_COLUMN_ALIASES.items()
    }


    missing_metrics = [
        metric
        for metric, col
        in rolling_cols.items()
        if col is None
    ]


    if missing_metrics:

        print(
            "\nWARNING: Some Rolling metrics "
            "cannot be compared:"
        )

        print(
            missing_metrics
        )


    independent_col_map = {
        "edge_count":
            "n_edges",

        "same_edges":
            "same_edges",

        "cross_edges":
            "cross_edges",

        "same_ratio":
            "same_edge_ratio",

        "mean_abs_partial":
            "mean_abs_partial",
    }


    roll_r3 = rolling.loc[
        R3
    ]


    roll_r4 = rolling.loc[
        R4
    ]


    ind_r3 = r3_summary
    ind_r4 = r4_summary


    consistency_rows = []


    # --------------------------------------------------------
    # Metric-by-metric comparison
    # --------------------------------------------------------

    for metric in [
        "edge_count",
        "same_edges",
        "cross_edges",
        "same_ratio",
        "mean_abs_partial",
    ]:

        roll_col = (
            rolling_cols[
                metric
            ]
        )


        if roll_col is None:
            continue


        ind_col = (
            independent_col_map[
                metric
            ]
        )


        rolling_R3 = float(
            roll_r3[
                roll_col
            ]
        )


        rolling_R4 = float(
            roll_r4[
                roll_col
            ]
        )


        independent_R3 = float(
            ind_r3[
                ind_col
            ]
        )


        independent_R4 = float(
            ind_r4[
                ind_col
            ]
        )


        rolling_delta = (
            rolling_R4
            -
            rolling_R3
        )


        independent_delta = (
            independent_R4
            -
            independent_R3
        )


        rolling_direction = direction(
            rolling_delta
        )


        independent_direction = direction(
            independent_delta
        )


        consistency_rows.append(
            {
                "metric":
                    metric,

                "rolling_R3":
                    rolling_R3,

                "rolling_R4":
                    rolling_R4,

                "rolling_delta":
                    rolling_delta,

                "rolling_direction":
                    rolling_direction,

                "independent_R3":
                    independent_R3,

                "independent_R4":
                    independent_R4,

                "independent_delta":
                    independent_delta,

                "independent_direction":
                    independent_direction,

                "same_direction":
                    (
                        rolling_direction
                        ==
                        independent_direction
                    ),
            }
        )


    consistency = pd.DataFrame(
        consistency_rows
    )


    consistency.to_csv(
        OUTPUT_DIR
        / "rolling_vs_independent_R3_R4_consistency.csv",

        index=False,
        encoding="utf-8-sig",
    )


    print("\n" + "=" * 78)
    print("Rolling vs Independent R3 -> R4")
    print("=" * 78)

    print(
        consistency
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 15. Core structural-consistency diagnostics
    #
    # We do NOT require every metric to have exactly the same
    # sign. In particular, Rolling Same Edges changed only
    # slightly, so the important question is whether Cross
    # changes dominate Same changes.
    # ========================================================

    required_core_metrics = {
        "edge_count",
        "same_edges",
        "cross_edges",
        "same_ratio",
        "mean_abs_partial",
    }


    available_core_metrics = (
        required_core_metrics
        -
        set(missing_metrics)
    )


    structural_rows = []


    if {
        "edge_count",
        "same_edges",
        "cross_edges",
        "same_ratio",
    }.issubset(
        available_core_metrics
    ):

        roll_edge_delta = (
            float(
                roll_r4[
                    rolling_cols[
                        "edge_count"
                    ]
                ]
            )
            -
            float(
                roll_r3[
                    rolling_cols[
                        "edge_count"
                    ]
                ]
            )
        )


        roll_same_delta = (
            float(
                roll_r4[
                    rolling_cols[
                        "same_edges"
                    ]
                ]
            )
            -
            float(
                roll_r3[
                    rolling_cols[
                        "same_edges"
                    ]
                ]
            )
        )


        roll_cross_delta = (
            float(
                roll_r4[
                    rolling_cols[
                        "cross_edges"
                    ]
                ]
            )
            -
            float(
                roll_r3[
                    rolling_cols[
                        "cross_edges"
                    ]
                ]
            )
        )


        roll_same_ratio_delta = (
            float(
                roll_r4[
                    rolling_cols[
                        "same_ratio"
                    ]
                ]
            )
            -
            float(
                roll_r3[
                    rolling_cols[
                        "same_ratio"
                    ]
                ]
            )
        )


        ind_edge_delta = (
            float(
                ind_r4[
                    "n_edges"
                ]
            )
            -
            float(
                ind_r3[
                    "n_edges"
                ]
            )
        )


        ind_same_delta = (
            float(
                ind_r4[
                    "same_edges"
                ]
            )
            -
            float(
                ind_r3[
                    "same_edges"
                ]
            )
        )


        ind_cross_delta = (
            float(
                ind_r4[
                    "cross_edges"
                ]
            )
            -
            float(
                ind_r3[
                    "cross_edges"
                ]
            )
        )


        ind_same_ratio_delta = (
            float(
                ind_r4[
                    "same_edge_ratio"
                ]
            )
            -
            float(
                ind_r3[
                    "same_edge_ratio"
                ]
            )
        )


        # ----------------------------------------------------
        # Criterion 1:
        # Overall network contracts.
        # ----------------------------------------------------

        structural_rows.append(
            {
                "criterion":
                    "overall_network_contraction",

                "rolling_support":
                    (
                        roll_edge_delta < 0
                    ),

                "independent_support":
                    (
                        ind_edge_delta < 0
                    ),

                "both_support":
                    (
                        roll_edge_delta < 0
                        and
                        ind_edge_delta < 0
                    ),
            }
        )


        # ----------------------------------------------------
        # Criterion 2:
        # Cross-industry links contract.
        # ----------------------------------------------------

        structural_rows.append(
            {
                "criterion":
                    "cross_industry_contraction",

                "rolling_support":
                    (
                        roll_cross_delta < 0
                    ),

                "independent_support":
                    (
                        ind_cross_delta < 0
                    ),

                "both_support":
                    (
                        roll_cross_delta < 0
                        and
                        ind_cross_delta < 0
                    ),
            }
        )


        # ----------------------------------------------------
        # Criterion 3:
        # Cross-industry changes dominate Same-industry
        # changes in absolute edge-count magnitude.
        # ----------------------------------------------------

        rolling_cross_dominant = (
            abs(
                roll_cross_delta
            )
            >
            abs(
                roll_same_delta
            )
        )


        independent_cross_dominant = (
            abs(
                ind_cross_delta
            )
            >
            abs(
                ind_same_delta
            )
        )


        structural_rows.append(
            {
                "criterion":
                    "cross_change_dominates_same",

                "rolling_support":
                    rolling_cross_dominant,

                "independent_support":
                    independent_cross_dominant,

                "both_support":
                    (
                        rolling_cross_dominant
                        and
                        independent_cross_dominant
                    ),
            }
        )


        # ----------------------------------------------------
        # Criterion 4:
        # Same-industry share increases.
        # ----------------------------------------------------

        structural_rows.append(
            {
                "criterion":
                    "same_industry_ratio_increases",

                "rolling_support":
                    (
                        roll_same_ratio_delta > 0
                    ),

                "independent_support":
                    (
                        ind_same_ratio_delta > 0
                    ),

                "both_support":
                    (
                        roll_same_ratio_delta > 0
                        and
                        ind_same_ratio_delta > 0
                    ),
            }
        )


    # --------------------------------------------------------
    # Criterion 5:
    # Sparser but stronger selected-edge structure.
    # --------------------------------------------------------

    if (
        "edge_count"
        in available_core_metrics
        and
        "mean_abs_partial"
        in available_core_metrics
    ):

        roll_strength_delta = (
            float(
                roll_r4[
                    rolling_cols[
                        "mean_abs_partial"
                    ]
                ]
            )
            -
            float(
                roll_r3[
                    rolling_cols[
                        "mean_abs_partial"
                    ]
                ]
            )
        )


        ind_strength_delta = (
            float(
                ind_r4[
                    "mean_abs_partial"
                ]
            )
            -
            float(
                ind_r3[
                    "mean_abs_partial"
                ]
            )
        )


        rolling_sparser_stronger = (
            roll_edge_delta < 0
            and
            roll_strength_delta > 0
        )


        independent_sparser_stronger = (
            ind_edge_delta < 0
            and
            ind_strength_delta > 0
        )


        structural_rows.append(
            {
                "criterion":
                    "sparser_but_stronger",

                "rolling_support":
                    rolling_sparser_stronger,

                "independent_support":
                    independent_sparser_stronger,

                "both_support":
                    (
                        rolling_sparser_stronger
                        and
                        independent_sparser_stronger
                    ),
            }
        )


    structural_consistency = pd.DataFrame(
        structural_rows
    )


    if not structural_consistency.empty:

        structural_consistency.to_csv(
            OUTPUT_DIR
            / "R3_R4_structural_consistency_summary.csv",

            index=False,
            encoding="utf-8-sig",
        )


        print("\n" + "=" * 78)
        print("Core structural consistency")
        print("=" * 78)

        print(
            structural_consistency
            .to_string(
                index=False
            )
        )


        n_criteria = len(
            structural_consistency
        )


        n_joint = int(
            structural_consistency[
                "both_support"
            ]
            .sum()
        )


        print(
            "\nJointly supported criteria: "
            f"{n_joint}/{n_criteria}"
        )


# ============================================================
# 16. Figure 1:
#     Independent R1-R5 Same/Cross edge composition
# ============================================================

plot_df = (
    independent_summary
    .set_index(
        "regime"
    )
    .loc[
        REGIMES
    ]
)


fig, ax = plt.subplots(
    figsize=(
        9,
        5.5,
    )
)


ax.bar(
    plot_df.index,
    plot_df[
        "same_edges"
    ],
    label="Same-industry",
)


ax.bar(
    plot_df.index,
    plot_df[
        "cross_edges"
    ],
    bottom=plot_df[
        "same_edges"
    ],
    label="Cross-industry",
)


ax.set_xlabel(
    "Independent Regime"
)


ax.set_ylabel(
    "Number of selected edges"
)


ax.set_title(
    "Same- and Cross-industry Edge Composition\n"
    "of Independent Regime Networks"
)


ax.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "independent_regime_structure.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 17. Figure 2:
#     R3 -> R4 Lost/Gained composition
# ============================================================

change_counts = pd.DataFrame(
    {
        "Same": [
            lost_same,
            gained_same,
        ],

        "Cross": [
            lost_cross,
            gained_cross,
        ],
    },
    index=[
        "Lost",
        "Gained",
    ],
)


fig, ax = plt.subplots(
    figsize=(
        7.5,
        5,
    )
)


change_counts.plot(
    kind="bar",
    ax=ax,
)


ax.set_xlabel(
    "Support-change type"
)


ax.set_ylabel(
    "Number of edges"
)


ax.set_title(
    "R3 -> R4 Independent-Regime Edge Changes"
)


ax.tick_params(
    axis="x",
    rotation=0,
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "R3_R4_change_composition.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 18. Final output summary
# ============================================================

print("\n" + "=" * 78)
print("Stage 3 completed successfully")
print("=" * 78)


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
    "independent_regime_industry_summary.csv",
    "R3_R4_independent_edge_changes.csv",
    "R3_R4_edge_type_industry_summary.csv",
    "R3_R4_independent_change_summary.csv",
    "independent_regime_structure.png",
    "R3_R4_change_composition.png",
]


for file in main_outputs:

    print(
        "  -",
        file,
    )


if rolling_summary_file is not None:

    print(
        "  -",
        "rolling_vs_independent_R3_R4_consistency.csv",
    )

    if (
        OUTPUT_DIR
        / "R3_R4_structural_consistency_summary.csv"
    ).exists():

        print(
            "  -",
            "R3_R4_structural_consistency_summary.csv",
        )