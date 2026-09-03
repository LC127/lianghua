from __future__ import annotations

from pathlib import Path
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

import numpy as np
import pandas as pd


# ============================================================
# 0. Project root
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 1. Input / output paths
# ============================================================

ROLLING_EDGE_CANDIDATES = [
    BASE_DIR
        / "stock_network"
        / "data"
        / "processed"
        / "rolling_glasso_edge_history.csv"
]

# Optional Stage 6 files. If they are found, the script additionally
# performs Independent-Regime vs Rolling-boundary consistency analysis.
STAGE6_1_CANDIDATES = [
    BASE_DIR
        / "stage6_1_industry_pair_transition"
        / "industry_pair_transition_summary.csv"
]

STAGE6_2_CANDIDATES = [
    BASE_DIR
        / "stage6_2_node_structural_driver"
        / "node_structural_driver_summary.csv"
]

STAGE6_3_CANDIDATES = [
    BASE_DIR
        / "stage6_3_driver_concentration_roles"
        / "node_structural_roles.csv"
]

STAGE6_4_EDGE_CANDIDATES = [
    BASE_DIR
        / "stage6_4_structural_driver_subnetwork"
        / "R3_R4_structural_driver_subnetwork_edges.csv"
]

OUTPUT_DIR = BASE_DIR / "stage7_regime_conditioned_rolling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Settings
# ============================================================

EDGE_TOL = 1e-8
TOP_K_COMPARE = 5
EVENT_RADIUS = 2

# The original regime segmentation on the 31 rolling networks.
REGIME_BY_WINDOW = {
    "R1": range(1, 8),    # windows 1--7
    "R2": range(8, 13),   # windows 8--12
    "R3": range(13, 23),  # windows 13--22
    "R4": range(23, 27),  # windows 23--26
    "R5": range(27, 32),  # windows 27--31
}

TARGET_PREV_WINDOW = 22
TARGET_CURR_WINDOW = 23
TARGET_PREV_DATE = pd.Timestamp("2025-10-16")
TARGET_CURR_DATE = pd.Timestamp("2025-11-13")


# ============================================================
# 3. Utilities
# ============================================================

def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def read_csv_fallback(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(
        f"Unable to decode file:\n{path}\nLast error: {last_error}"
    )


def clean_stock(x) -> str:
    text = str(x).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    converted = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if converted.isna().any():
        bad = series.loc[converted.isna()].drop_duplicates().tolist()
        raise ValueError(
            f"Unable to parse boolean values: {bad}"
        )

    return converted.astype(bool)


def regime_from_window(window_id: int) -> str:
    for regime, window_range in REGIME_BY_WINDOW.items():
        if window_id in window_range:
            return regime
    raise ValueError(f"window_id={window_id} is outside R1--R5.")


def canonical_stock_pair(stock_i: str, stock_j: str) -> str:
    a, b = sorted([clean_stock(stock_i), clean_stock(stock_j)])
    return f"{a}|{b}"


def canonical_industry_pair(industry_i: str, industry_j: str) -> tuple[str, str]:
    a, b = sorted([str(industry_i).strip(), str(industry_j).strip()])
    return a, b


# ============================================================
# 4. Chinese font
# ============================================================

def setup_chinese_font() -> tuple[FontProperties, str]:
    preferred_fonts = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "DengXian",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]

    installed = {f.name for f in font_manager.fontManager.ttflist}

    for name in preferred_fonts:
        if name in installed:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return FontProperties(family=name), name

    windows_fonts = [
        ("Microsoft YaHei", r"C:\Windows\Fonts\msyh.ttc"),
        ("SimHei", r"C:\Windows\Fonts\simhei.ttf"),
        ("DengXian", r"C:\Windows\Fonts\Deng.ttf"),
        ("SimSun", r"C:\Windows\Fonts\simsun.ttc"),
    ]

    for name, path in windows_fonts:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass
            plt.rcParams["axes.unicode_minus"] = False
            return FontProperties(fname=path), name

    plt.rcParams["axes.unicode_minus"] = False
    return FontProperties(family="DejaVu Sans"), "DejaVu Sans (fallback)"


CHINESE_FONT, CHINESE_FONT_NAME = setup_chinese_font()


# ============================================================
# 5. Locate and load the REAL Rolling GLasso history file
# ============================================================

ROLLING_EDGE_FILE = first_existing(ROLLING_EDGE_CANDIDATES)

if ROLLING_EDGE_FILE is None:
    raise FileNotFoundError(
        "Cannot find rolling_glasso_edge_history.csv.\n"
        "Please modify ROLLING_EDGE_CANDIDATES near the top of this script."
    )

rolling = read_csv_fallback(ROLLING_EDGE_FILE)

# Exact real columns from rolling_glasso_edge_history.csv
REQUIRED_COLUMNS = [
    "window_id",
    "window_start",
    "window_end",
    "network_date",
    "stock_1",
    "name_1",
    "industry_1",
    "stock_2",
    "name_2",
    "industry_2",
    "precision",
    "partial_correlation",
    "abs_partial_correlation",
    "selected",
    "same_industry",
]

missing = [col for col in REQUIRED_COLUMNS if col not in rolling.columns]

if missing:
    raise ValueError(
        "The rolling file does not match the expected real schema.\n"
        f"Missing columns: {missing}\n"
        f"Available columns: {list(rolling.columns)}"
    )


# ============================================================
# 6. Clean rolling data
# ============================================================

rolling["window_id"] = pd.to_numeric(
    rolling["window_id"], errors="raise"
).astype(int)

for col in ["window_start", "window_end", "network_date"]:
    rolling[col] = pd.to_datetime(rolling[col], errors="raise")

for col in ["stock_1", "stock_2"]:
    rolling[col] = rolling[col].apply(clean_stock)

for col in [
    "precision",
    "partial_correlation",
    "abs_partial_correlation",
]:
    rolling[col] = pd.to_numeric(rolling[col], errors="raise")

rolling["selected"] = to_bool(rolling["selected"])
rolling["same_industry"] = to_bool(rolling["same_industry"])

for col in ["name_1", "name_2", "industry_1", "industry_2"]:
    rolling[col] = rolling[col].astype(str).str.strip()

rolling["regime"] = rolling["window_id"].apply(regime_from_window)

rolling["pair_key"] = rolling.apply(
    lambda row: canonical_stock_pair(row["stock_1"], row["stock_2"]),
    axis=1,
)

industry_pairs = rolling.apply(
    lambda row: canonical_industry_pair(
        row["industry_1"], row["industry_2"]
    ),
    axis=1,
)

rolling["industry_a"] = [x[0] for x in industry_pairs]
rolling["industry_b"] = [x[1] for x in industry_pairs]


# ============================================================
# 7. Strict sanity checks on the real file
# ============================================================

print("\n" + "=" * 82)
print("Stage 7: Regime-conditioned Rolling Dynamic Validation")
print("=" * 82)

print(f"Input file: {ROLLING_EDGE_FILE.resolve()}")
print(f"Rows: {len(rolling)}")
print(f"Windows: {rolling['window_id'].nunique()}")
print(f"Network dates: {rolling['network_date'].nunique()}")

# Every one of the 31 windows should contain all C(15,2)=105 pairs.
pairs_per_window = rolling.groupby("window_id").size()

if not (pairs_per_window == 105).all():
    raise ValueError(
        "Not every rolling window contains exactly 105 stock pairs.\n"
        f"{pairs_per_window}"
    )

# No duplicated pair within a window.
dup_count = rolling.duplicated(
    subset=["window_id", "pair_key"]
).sum()

if dup_count != 0:
    raise ValueError(
        f"Found {dup_count} duplicate stock pairs within rolling windows."
    )

# selected should agree with partial-correlation sparsity.
selected_from_partial = (
    rolling["abs_partial_correlation"] > EDGE_TOL
)

selection_mismatch = int(
    (selected_from_partial != rolling["selected"]).sum()
)

if selection_mismatch > 0:
    print(
        f"WARNING: {selection_mismatch} rows have selected status "
        "inconsistent with abs_partial_correlation > EDGE_TOL."
    )

# Window-date table.
window_table = (
    rolling[
        [
            "window_id",
            "window_start",
            "window_end",
            "network_date",
            "regime",
        ]
    ]
    .drop_duplicates()
    .sort_values("window_id")
    .reset_index(drop=True)
)

if len(window_table) != 31:
    raise ValueError(
        f"Expected 31 rolling windows, found {len(window_table)}."
    )

window_table.to_csv(
    OUTPUT_DIR / "rolling_window_regime_map.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nWindow / Regime counts:")
print(
    window_table.groupby("regime")["window_id"]
    .count()
    .to_string()
)


# ============================================================
# 8. Metadata tables from the rolling file itself
# ============================================================

metadata_1 = rolling[
    ["stock_1", "name_1", "industry_1"]
].rename(
    columns={
        "stock_1": "stock",
        "name_1": "name",
        "industry_1": "industry",
    }
)

metadata_2 = rolling[
    ["stock_2", "name_2", "industry_2"]
].rename(
    columns={
        "stock_2": "stock",
        "name_2": "name",
        "industry_2": "industry",
    }
)

stock_metadata = (
    pd.concat([metadata_1, metadata_2], ignore_index=True)
    .drop_duplicates()
    .sort_values("stock")
    .reset_index(drop=True)
)

# Each stock should have one unique name/industry.
meta_check = (
    stock_metadata.groupby("stock")
    .agg(
        n_names=("name", "nunique"),
        n_industries=("industry", "nunique"),
    )
)

if (meta_check["n_names"] > 1).any() or (
    meta_check["n_industries"] > 1
).any():
    raise ValueError(
        "Stock name/industry metadata are inconsistent across windows."
    )

stock_metadata = stock_metadata.drop_duplicates(
    subset=["stock"]
).set_index("stock")

stocks = stock_metadata.index.tolist()

print(f"\nStock universe: {len(stocks)}")
print(f"Expected unordered pairs: {len(stocks)*(len(stocks)-1)//2}")


# ============================================================
# 9. Build all 30 adjacent rolling transitions
# ============================================================

transition_rows: list[dict] = []
pair_rows: list[pd.DataFrame] = []
industry_rows: list[dict] = []
node_rows: list[dict] = []


for k in range(1, len(window_table)):

    prev_info = window_table.iloc[k - 1]
    curr_info = window_table.iloc[k]

    prev_id = int(prev_info["window_id"])
    curr_id = int(curr_info["window_id"])

    prev = rolling.loc[
        rolling["window_id"] == prev_id
    ].copy()

    curr = rolling.loc[
        rolling["window_id"] == curr_id
    ].copy()

    # Merge the SAME 105 stock pairs across adjacent windows.
    merged = prev.merge(
        curr,
        on="pair_key",
        how="inner",
        suffixes=("_prev", "_curr"),
        validate="one_to_one",
    )

    if len(merged) != 105:
        raise ValueError(
            f"Transition {prev_id}->{curr_id} has {len(merged)} merged pairs, "
            "not 105."
        )

    # Industry labels and Same/Cross status must remain fixed.
    if not (
        merged["same_industry_prev"].to_numpy()
        == merged["same_industry_curr"].to_numpy()
    ).all():
        raise ValueError(
            f"same_industry metadata changed in transition {prev_id}->{curr_id}."
        )

    # Core continuous network change.
    merged["delta_partial"] = (
        merged["partial_correlation_curr"]
        - merged["partial_correlation_prev"]
    )

    merged["abs_delta_partial"] = (
        merged["delta_partial"].abs()
    )

    merged["transition_energy"] = (
        merged["delta_partial"] ** 2
    )

    # Support transition uses the REAL saved selected flags.
    prev_sel = merged["selected_prev"]
    curr_sel = merged["selected_curr"]

    merged["support_change"] = np.select(
        [
            prev_sel & curr_sel,
            prev_sel & (~curr_sel),
            (~prev_sel) & curr_sel,
        ],
        [
            "Persistent",
            "Lost",
            "Gained",
        ],
        default="Absent",
    )

    merged["relation"] = np.where(
        merged["same_industry_prev"],
        "Same",
        "Cross",
    )

    # Canonical industry pair based on the previous row metadata.
    canonical_pairs = merged.apply(
        lambda row: canonical_industry_pair(
            row["industry_1_prev"],
            row["industry_2_prev"],
        ),
        axis=1,
    )

    merged["industry_a"] = [x[0] for x in canonical_pairs]
    merged["industry_b"] = [x[1] for x in canonical_pairs]

    regime_prev = str(prev_info["regime"])
    regime_curr = str(curr_info["regime"])

    is_boundary = regime_prev != regime_curr

    transition_type = (
        f"{regime_prev}→{regime_curr} Boundary"
        if is_boundary
        else f"Within {regime_curr}"
    )

    date_prev = pd.Timestamp(prev_info["network_date"])
    date_curr = pd.Timestamp(curr_info["network_date"])

    total_energy = float(
        merged["transition_energy"].sum()
    )

    same_energy = float(
        merged.loc[
            merged["relation"] == "Same",
            "transition_energy",
        ].sum()
    )

    cross_energy = float(
        merged.loc[
            merged["relation"] == "Cross",
            "transition_energy",
        ].sum()
    )

    support_mask = merged["support_change"].isin(
        ["Lost", "Gained"]
    )

    persistent_mask = (
        merged["support_change"] == "Persistent"
    )

    support_energy = float(
        merged.loc[
            support_mask,
            "transition_energy",
        ].sum()
    )

    persistent_energy = float(
        merged.loc[
            persistent_mask,
            "transition_energy",
        ].sum()
    )

    lost = int(
        (merged["support_change"] == "Lost").sum()
    )

    gained = int(
        (merged["support_change"] == "Gained").sum()
    )

    persistent = int(
        persistent_mask.sum()
    )

    absent = int(
        (merged["support_change"] == "Absent").sum()
    )

    edges_prev = int(
        merged["selected_prev"].sum()
    )

    edges_curr = int(
        merged["selected_curr"].sum()
    )

    gross_change = lost + gained
    net_edge_change = gained - lost

    # Same definition used earlier:
    # gross support changes minus pure net-size change.
    rewiring = (
        gross_change
        - abs(net_edge_change)
    )

    union_count = persistent + lost + gained

    cross_support_changes = int(
        (
            support_mask
            & (merged["relation"] == "Cross")
        ).sum()
    )

    transition_rows.append(
        {
            "window_prev": prev_id,
            "window_curr": curr_id,
            "date_prev": date_prev,
            "date_curr": date_curr,
            "regime_prev": regime_prev,
            "regime_curr": regime_curr,
            "transition_type": transition_type,
            "is_boundary": is_boundary,
            "edges_prev": edges_prev,
            "edges_curr": edges_curr,
            "persistent": persistent,
            "lost": lost,
            "gained": gained,
            "absent": absent,
            "gross_support_change": gross_change,
            "net_edge_change": net_edge_change,
            "rewiring": rewiring,
            "jaccard": (
                persistent / union_count
                if union_count > 0
                else np.nan
            ),
            "cross_support_change_share": (
                cross_support_changes / gross_change
                if gross_change > 0
                else np.nan
            ),
            "transition_energy": total_energy,
            "TI_total": np.sqrt(total_energy),
            "TI_same": np.sqrt(same_energy),
            "TI_cross": np.sqrt(cross_energy),
            "same_energy_share": (
                same_energy / total_energy
                if total_energy > 0
                else np.nan
            ),
            "cross_energy_share": (
                cross_energy / total_energy
                if total_energy > 0
                else np.nan
            ),
            "support_change_energy": support_energy,
            "persistent_change_energy": persistent_energy,
            "support_energy_share": (
                support_energy / total_energy
                if total_energy > 0
                else np.nan
            ),
            "persistent_energy_share": (
                persistent_energy / total_energy
                if total_energy > 0
                else np.nan
            ),
        }
    )

    # --------------------------------------------------------
    # Pair-level details
    # --------------------------------------------------------

    pair_out = pd.DataFrame(
        {
            "window_prev": prev_id,
            "window_curr": curr_id,
            "date_prev": date_prev,
            "date_curr": date_curr,
            "regime_prev": regime_prev,
            "regime_curr": regime_curr,
            "transition_type": transition_type,
            "pair_key": merged["pair_key"],
            "stock_i": merged["stock_1_prev"],
            "name_i": merged["name_1_prev"],
            "industry_i": merged["industry_1_prev"],
            "stock_j": merged["stock_2_prev"],
            "name_j": merged["name_2_prev"],
            "industry_j": merged["industry_2_prev"],
            "relation": merged["relation"],
            "partial_prev": merged["partial_correlation_prev"],
            "partial_curr": merged["partial_correlation_curr"],
            "delta_partial": merged["delta_partial"],
            "abs_delta_partial": merged["abs_delta_partial"],
            "transition_energy": merged["transition_energy"],
            "selected_prev": merged["selected_prev"],
            "selected_curr": merged["selected_curr"],
            "support_change": merged["support_change"],
        }
    )

    pair_rows.append(pair_out)

    # --------------------------------------------------------
    # Industry-pair dynamic decomposition
    # --------------------------------------------------------

    for (industry_a, industry_b), group in merged.groupby(
        ["industry_a", "industry_b"],
        sort=True,
    ):

        energy = float(
            group["transition_energy"].sum()
        )

        industry_rows.append(
            {
                "window_prev": prev_id,
                "window_curr": curr_id,
                "date_prev": date_prev,
                "date_curr": date_curr,
                "regime_prev": regime_prev,
                "regime_curr": regime_curr,
                "transition_type": transition_type,
                "industry_a": industry_a,
                "industry_b": industry_b,
                "industry_pair": f"{industry_a} - {industry_b}",
                "relation": (
                    "Same"
                    if industry_a == industry_b
                    else "Cross"
                ),
                "possible_pairs": len(group),
                "transition_energy": energy,
                "transition_intensity": np.sqrt(energy),
                "energy_share": (
                    energy / total_energy
                    if total_energy > 0
                    else np.nan
                ),
                "persistent": int(
                    (
                        group["support_change"]
                        == "Persistent"
                    ).sum()
                ),
                "lost": int(
                    (
                        group["support_change"]
                        == "Lost"
                    ).sum()
                ),
                "gained": int(
                    (
                        group["support_change"]
                        == "Gained"
                    ).sum()
                ),
            }
        )

    # --------------------------------------------------------
    # Dynamic node-driver decomposition
    # --------------------------------------------------------

    for stock in stocks:

        incident_mask = (
            (merged["stock_1_prev"] == stock)
            | (merged["stock_2_prev"] == stock)
        )

        incident = merged.loc[
            incident_mask
        ].copy()

        score = float(
            incident["transition_energy"].sum()
        )

        same_score = float(
            incident.loc[
                incident["relation"] == "Same",
                "transition_energy",
            ].sum()
        )

        cross_score = float(
            incident.loc[
                incident["relation"] == "Cross",
                "transition_energy",
            ].sum()
        )

        node_support_mask = (
            incident["support_change"]
            .isin(["Lost", "Gained"])
        )

        support_score = float(
            incident.loc[
                node_support_mask,
                "transition_energy",
            ].sum()
        )

        persistent_score = float(
            incident.loc[
                incident["support_change"]
                == "Persistent",
                "transition_energy",
            ].sum()
        )

        node_rows.append(
            {
                "window_prev": prev_id,
                "window_curr": curr_id,
                "date_prev": date_prev,
                "date_curr": date_curr,
                "regime_prev": regime_prev,
                "regime_curr": regime_curr,
                "transition_type": transition_type,
                "stock": stock,
                "name": stock_metadata.at[stock, "name"],
                "industry": stock_metadata.at[stock, "industry"],
                "driver_score": score,
                "driver_share": (
                    score / (2.0 * total_energy)
                    if total_energy > 0
                    else np.nan
                ),
                "same_driver_score": same_score,
                "cross_driver_score": cross_score,
                "same_driver_share": (
                    same_score / score
                    if score > 0
                    else np.nan
                ),
                "cross_driver_share": (
                    cross_score / score
                    if score > 0
                    else np.nan
                ),
                "support_change_score": support_score,
                "persistent_change_score": persistent_score,
                "support_change_share": (
                    support_score / score
                    if score > 0
                    else np.nan
                ),
                "persistent_change_share": (
                    persistent_score / score
                    if score > 0
                    else np.nan
                ),
                "support_turnover": int(
                    node_support_mask.sum()
                ),
            }
        )


# ============================================================
# 10. Build main Stage 7 tables
# ============================================================

transition_df = pd.DataFrame(
    transition_rows
).sort_values(
    ["window_prev", "window_curr"]
).reset_index(drop=True)

pair_df = pd.concat(
    pair_rows,
    ignore_index=True,
)

industry_df = pd.DataFrame(
    industry_rows
)

node_df = pd.DataFrame(
    node_rows
)

node_df["driver_rank"] = (
    node_df.groupby(
        ["window_prev", "window_curr"]
    )["driver_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

industry_df["energy_rank"] = (
    industry_df.groupby(
        ["window_prev", "window_curr"]
    )["transition_energy"]
    .rank(method="min", ascending=False)
    .astype(int)
)


# ============================================================
# 11. Sanity checks for decompositions
# ============================================================

same_cross_error = np.max(
    np.abs(
        transition_df["TI_total"] ** 2
        - (
            transition_df["TI_same"] ** 2
            + transition_df["TI_cross"] ** 2
        )
    )
)

support_persistent_error = np.nanmax(
    np.abs(
        transition_df["support_energy_share"]
        + transition_df["persistent_energy_share"]
        - 1.0
    )
)

print("\nSanity checks:")
print(
    "Max Same/Cross energy decomposition error:",
    f"{same_cross_error:.12g}",
)
print(
    "Max Support/Persistent share error:",
    f"{support_persistent_error:.12g}",
)


# ============================================================
# 12. Within-Regime summary
# ============================================================

within_df = transition_df.loc[
    ~transition_df["is_boundary"]
].copy()

within_summary = (
    within_df.groupby(
        "regime_curr",
        as_index=False,
    )
    .agg(
        n_transitions=("TI_total", "size"),
        mean_TI=("TI_total", "mean"),
        median_TI=("TI_total", "median"),
        sd_TI=("TI_total", "std"),
        min_TI=("TI_total", "min"),
        max_TI=("TI_total", "max"),
        mean_cross_energy_share=(
            "cross_energy_share", "mean"
        ),
        mean_support_energy_share=(
            "support_energy_share", "mean"
        ),
        mean_persistent_energy_share=(
            "persistent_energy_share", "mean"
        ),
        mean_gross_support_change=(
            "gross_support_change", "mean"
        ),
        mean_jaccard=("jaccard", "mean"),
    )
)


# ============================================================
# 13. Regime-boundary summary
# ============================================================

boundary_df = transition_df.loc[
    transition_df["is_boundary"]
].copy()

boundary_df["boundary"] = (
    boundary_df["regime_prev"]
    + "→"
    + boundary_df["regime_curr"]
)

boundary_df["boundary_TI_rank"] = (
    boundary_df["TI_total"]
    .rank(method="min", ascending=False)
    .astype(int)
)


# ============================================================
# 14. R3->R4 boundary-specific outputs
# ============================================================

target_mask = (
    (transition_df["window_prev"] == TARGET_PREV_WINDOW)
    & (transition_df["window_curr"] == TARGET_CURR_WINDOW)
    & (transition_df["date_prev"] == TARGET_PREV_DATE)
    & (transition_df["date_curr"] == TARGET_CURR_DATE)
)

target_transition = transition_df.loc[
    target_mask
].copy()

if len(target_transition) != 1:
    raise RuntimeError(
        "Could not uniquely identify the R3->R4 rolling boundary "
        "(window 22 -> 23, 2025-10-16 -> 2025-11-13)."
    )

target_transition.to_csv(
    OUTPUT_DIR / "R3_R4_rolling_boundary_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

target_industry = industry_df.loc[
    (industry_df["window_prev"] == TARGET_PREV_WINDOW)
    & (industry_df["window_curr"] == TARGET_CURR_WINDOW)
].sort_values(
    "transition_energy",
    ascending=False,
).reset_index(drop=True)

target_industry.to_csv(
    OUTPUT_DIR / "R3_R4_rolling_industry_pair_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

target_node = node_df.loc[
    (node_df["window_prev"] == TARGET_PREV_WINDOW)
    & (node_df["window_curr"] == TARGET_CURR_WINDOW)
].sort_values(
    "driver_score",
    ascending=False,
).reset_index(drop=True)

target_node.to_csv(
    OUTPUT_DIR / "R3_R4_rolling_node_driver_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

target_pairs = pair_df.loc[
    (pair_df["window_prev"] == TARGET_PREV_WINDOW)
    & (pair_df["window_curr"] == TARGET_CURR_WINDOW)
].sort_values(
    "abs_delta_partial",
    ascending=False,
).reset_index(drop=True)

target_pairs.to_csv(
    OUTPUT_DIR / "R3_R4_rolling_pair_details.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 15. Boundary vs Within-R3 / Within-R4
# ============================================================

target_TI = float(
    target_transition["TI_total"].iloc[0]
)

r3_row = within_summary.loc[
    within_summary["regime_curr"] == "R3"
]

r4_row = within_summary.loc[
    within_summary["regime_curr"] == "R4"
]

if len(r3_row) != 1 or len(r4_row) != 1:
    raise RuntimeError(
        "Cannot find unique Within-R3 / Within-R4 summaries."
    )

mean_r3_TI = float(
    r3_row["mean_TI"].iloc[0]
)

mean_r4_TI = float(
    r4_row["mean_TI"].iloc[0]
)

adjacent_mean_TI = (
    mean_r3_TI + mean_r4_TI
) / 2.0

boundary_vs_within = pd.DataFrame(
    [
        {
            "boundary": "R3→R4",
            "boundary_TI": target_TI,
            "mean_within_R3_TI": mean_r3_TI,
            "mean_within_R4_TI": mean_r4_TI,
            "mean_adjacent_within_TI": adjacent_mean_TI,
            "boundary_vs_R3_ratio": (
                target_TI / mean_r3_TI
                if mean_r3_TI > 0
                else np.nan
            ),
            "boundary_vs_R4_ratio": (
                target_TI / mean_r4_TI
                if mean_r4_TI > 0
                else np.nan
            ),
            "boundary_vs_adjacent_mean_ratio": (
                target_TI / adjacent_mean_TI
                if adjacent_mean_TI > 0
                else np.nan
            ),
        }
    ]
)

boundary_vs_within.to_csv(
    OUTPUT_DIR / "R3_R4_boundary_vs_within_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 16. Event window: abrupt vs gradual
# ============================================================

target_idx = int(
    transition_df.index[target_mask][0]
)

start_idx = max(
    0,
    target_idx - EVENT_RADIUS,
)

end_idx = min(
    len(transition_df),
    target_idx + EVENT_RADIUS + 1,
)

event_window = transition_df.iloc[
    start_idx:end_idx
].copy()

event_window["relative_transition"] = (
    np.arange(start_idx, end_idx)
    - target_idx
)

event_window.to_csv(
    OUTPUT_DIR / "R3_R4_boundary_event_window.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 17. Save core Stage 7 results
# ============================================================

transition_df.to_csv(
    OUTPUT_DIR / "rolling_regime_transition_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

within_summary.to_csv(
    OUTPUT_DIR / "within_regime_transition_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

boundary_df.to_csv(
    OUTPUT_DIR / "regime_boundary_transition_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

industry_df.to_csv(
    OUTPUT_DIR / "rolling_industry_pair_transition.csv",
    index=False,
    encoding="utf-8-sig",
)

node_df.to_csv(
    OUTPUT_DIR / "rolling_node_driver_scores.csv",
    index=False,
    encoding="utf-8-sig",
)

pair_df.to_csv(
    OUTPUT_DIR / "rolling_pair_transition_details.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 18. Optional: Stage 6.1 vs Rolling boundary consistency
# ============================================================

consistency_rows = []

STAGE6_1_FILE = first_existing(
    STAGE6_1_CANDIDATES
)

if STAGE6_1_FILE is not None:

    static_industry = read_csv_fallback(
        STAGE6_1_FILE
    )

    required = [
        "industry_a",
        "industry_b",
        "transition_energy",
    ]

    if all(
        col in static_industry.columns
        for col in required
    ):

        static_industry = static_industry.copy()

        static_pairs = static_industry.apply(
            lambda row: canonical_industry_pair(
                row["industry_a"],
                row["industry_b"],
            ),
            axis=1,
        )

        static_industry["industry_a"] = [
            x[0] for x in static_pairs
        ]

        static_industry["industry_b"] = [
            x[1] for x in static_pairs
        ]

        comparison = static_industry.merge(
            target_industry[
                [
                    "industry_a",
                    "industry_b",
                    "transition_energy",
                    "energy_share",
                    "energy_rank",
                ]
            ],
            on=[
                "industry_a",
                "industry_b",
            ],
            how="outer",
            suffixes=(
                "_independent",
                "_rolling",
            ),
        )

        for col in [
            "transition_energy_independent",
            "transition_energy_rolling",
        ]:
            comparison[col] = comparison[col].fillna(0.0)

        industry_spearman = comparison[
            [
                "transition_energy_independent",
                "transition_energy_rolling",
            ]
        ].corr(
            method="spearman"
        ).iloc[0, 1]

        static_top = set(
            static_industry.nlargest(
                TOP_K_COMPARE,
                "transition_energy",
            ).apply(
                lambda row: (
                    row["industry_a"],
                    row["industry_b"],
                ),
                axis=1,
            )
        )

        rolling_top = set(
            target_industry.nlargest(
                TOP_K_COMPARE,
                "transition_energy",
            ).apply(
                lambda row: (
                    row["industry_a"],
                    row["industry_b"],
                ),
                axis=1,
            )
        )

        intersection = (
            static_top & rolling_top
        )

        union = (
            static_top | rolling_top
        )

        consistency_rows.append(
            {
                "level": "Industry pair",
                "top_k": TOP_K_COMPARE,
                "top_k_overlap": len(intersection),
                "top_k_jaccard": (
                    len(intersection) / len(union)
                    if len(union) > 0
                    else np.nan
                ),
                "rank_spearman": industry_spearman,
            }
        )

        comparison.to_csv(
            OUTPUT_DIR / "Stage6_vs_Rolling_industry_pair_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )


# ============================================================
# 19. Optional: Stage 6.2 vs Rolling boundary node consistency
# ============================================================

STAGE6_2_FILE = first_existing(
    STAGE6_2_CANDIDATES
)

if STAGE6_2_FILE is not None:

    static_node = read_csv_fallback(
        STAGE6_2_FILE
    )

    required = [
        "stock",
        "driver_score",
    ]

    if all(
        col in static_node.columns
        for col in required
    ):

        static_node = static_node.copy()

        static_node["stock"] = (
            static_node["stock"]
            .apply(clean_stock)
        )

        comparison = static_node.merge(
            target_node[
                [
                    "stock",
                    "name",
                    "industry",
                    "driver_score",
                    "driver_share",
                    "driver_rank",
                    "cross_driver_share",
                    "support_change_share",
                    "persistent_change_share",
                    "support_turnover",
                ]
            ],
            on="stock",
            how="outer",
            suffixes=(
                "_independent",
                "_rolling",
            ),
        )

        for col in [
            "driver_score_independent",
            "driver_score_rolling",
        ]:
            comparison[col] = comparison[col].fillna(0.0)

        node_spearman = comparison[
            [
                "driver_score_independent",
                "driver_score_rolling",
            ]
        ].corr(
            method="spearman"
        ).iloc[0, 1]

        static_top_nodes = set(
            static_node.nlargest(
                TOP_K_COMPARE,
                "driver_score",
            )["stock"]
        )

        rolling_top_nodes = set(
            target_node.nlargest(
                TOP_K_COMPARE,
                "driver_score",
            )["stock"]
        )

        intersection = (
            static_top_nodes & rolling_top_nodes
        )

        union = (
            static_top_nodes | rolling_top_nodes
        )

        consistency_rows.append(
            {
                "level": "Node driver",
                "top_k": TOP_K_COMPARE,
                "top_k_overlap": len(intersection),
                "top_k_jaccard": (
                    len(intersection) / len(union)
                    if len(union) > 0
                    else np.nan
                ),
                "rank_spearman": node_spearman,
            }
        )

        comparison.to_csv(
            OUTPUT_DIR / "Stage6_vs_Rolling_node_driver_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )


# ============================================================
# 20. Optional: Stage 6.3 role consistency
# ============================================================

STAGE6_3_FILE = first_existing(
    STAGE6_3_CANDIDATES
)

if STAGE6_3_FILE is not None:

    roles = read_csv_fallback(
        STAGE6_3_FILE
    )

    if "stock" in roles.columns:

        roles = roles.copy()

        roles["stock"] = (
            roles["stock"]
            .apply(clean_stock)
        )

        rolling_roles = target_node.copy()

        rolling_roles["rolling_orientation_role"] = np.where(
            rolling_roles["cross_driver_share"] >= 0.60,
            "Cross-driven",
            np.where(
                rolling_roles["cross_driver_share"] <= 0.40,
                "Same-driven",
                "Mixed Same/Cross",
            ),
        )

        rolling_roles["rolling_mechanism_role"] = np.where(
            rolling_roles["support_change_share"] >= 0.60,
            "Support-rewiring",
            np.where(
                rolling_roles["support_change_share"] <= 0.40,
                "Persistent-strength repricing",
                "Mixed support/strength",
            ),
        )

        role_comparison = roles.merge(
            rolling_roles[
                [
                    "stock",
                    "name",
                    "industry",
                    "driver_score",
                    "driver_rank",
                    "cross_driver_share",
                    "support_change_share",
                    "rolling_orientation_role",
                    "rolling_mechanism_role",
                ]
            ],
            on="stock",
            how="left",
            suffixes=(
                "_independent",
                "_rolling",
            ),
        )

        if "orientation_role" in role_comparison.columns:
            role_comparison[
                "orientation_agree"
            ] = (
                role_comparison[
                    "orientation_role"
                ]
                == role_comparison[
                    "rolling_orientation_role"
                ]
            )

        if "mechanism_role" in role_comparison.columns:
            role_comparison[
                "mechanism_agree"
            ] = (
                role_comparison[
                    "mechanism_role"
                ]
                == role_comparison[
                    "rolling_mechanism_role"
                ]
            )

        role_comparison.to_csv(
            OUTPUT_DIR / "Stage6_vs_Rolling_node_role_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )


# ============================================================
# 21. Optional: Stage 6.4 driver-subnetwork edge consistency
# ============================================================

STAGE6_4_EDGE_FILE = first_existing(
    STAGE6_4_EDGE_CANDIDATES
)

if STAGE6_4_EDGE_FILE is not None:

    static_edges = read_csv_fallback(
        STAGE6_4_EDGE_FILE
    )

    if all(
        col in static_edges.columns
        for col in ["stock_i", "stock_j"]
    ):

        static_edges = static_edges.copy()

        static_edges[
            "edge_key"
        ] = static_edges.apply(
            lambda row: canonical_stock_pair(
                row["stock_i"],
                row["stock_j"],
            ),
            axis=1,
        )

        static_edge_keys = set(
            static_edges["edge_key"]
        )

        boundary_edge_comparison = (
            target_pairs.copy()
        )

        boundary_edge_comparison[
            "is_stage6_4_driver_subnetwork_edge"
        ] = (
            boundary_edge_comparison[
                "pair_key"
            ]
            .isin(static_edge_keys)
        )

        static_cols = [
            "edge_key",
        ]

        for col in [
            "delta_partial",
            "abs_delta_partial",
            "support_change",
        ]:
            if col in static_edges.columns:
                static_cols.append(col)

        static_small = static_edges[
            static_cols
        ].copy()

        rename_map = {
            "delta_partial": "delta_partial_independent",
            "abs_delta_partial": "abs_delta_partial_independent",
            "support_change": "support_change_independent",
        }

        static_small = static_small.rename(
            columns=rename_map
        )

        boundary_edge_comparison = (
            boundary_edge_comparison.merge(
                static_small,
                left_on="pair_key",
                right_on="edge_key",
                how="left",
            )
        )

        if (
            "support_change_independent"
            in boundary_edge_comparison.columns
        ):
            boundary_edge_comparison[
                "support_status_agree"
            ] = (
                boundary_edge_comparison[
                    "support_change"
                ]
                == boundary_edge_comparison[
                    "support_change_independent"
                ]
            )

        boundary_edge_comparison.to_csv(
            OUTPUT_DIR / "Stage6_vs_Rolling_driver_subnetwork_edge_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )


if consistency_rows:
    pd.DataFrame(
        consistency_rows
    ).to_csv(
        OUTPUT_DIR / "Stage6_vs_Rolling_consistency_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 22. Figures
# ============================================================

# Figure 1: all rolling TI, with four Regime boundaries marked.
fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.plot(
    transition_df["date_curr"],
    transition_df["TI_total"],
    marker="o",
)

boundary_points = transition_df.loc[
    transition_df["is_boundary"]
]

ax.scatter(
    boundary_points["date_curr"],
    boundary_points["TI_total"],
    s=100,
    zorder=5,
)

for row in boundary_points.itertuples():
    ax.annotate(
        f"{row.regime_prev}→{row.regime_curr}",
        (row.date_curr, row.TI_total),
        xytext=(5, 7),
        textcoords="offset points",
        fontproperties=CHINESE_FONT,
        fontsize=9,
    )

ax.set_title(
    "Regime-conditioned Rolling Network Transition Intensity",
    fontsize=14,
)

ax.set_xlabel("Rolling Network Date")
ax.set_ylabel("Transition Intensity")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rolling_regime_transition_intensity.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Figure 2: four boundary TIs.
boundary_plot = boundary_df.sort_values(
    "TI_total",
    ascending=True,
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.barh(
    boundary_plot["boundary"],
    boundary_plot["TI_total"],
)

ax.set_xlabel("Transition Intensity")
ax.set_ylabel("Regime Boundary")

ax.set_title(
    "四个 Regime 边界的网络结构转换强度",
    fontproperties=CHINESE_FONT,
    fontsize=14,
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "regime_boundary_TI_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Figure 3: event window around R3->R4.
fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.plot(
    event_window["relative_transition"],
    event_window["TI_total"],
    marker="o",
)

ax.axvline(
    0,
    linestyle="--",
)

ax.set_xlabel(
    "Relative transition to R3→R4 boundary"
)
ax.set_ylabel("Transition Intensity")

ax.set_title(
    "R3→R4 边界前后的 Rolling 网络结构变化",
    fontproperties=CHINESE_FONT,
    fontsize=14,
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "R3_R4_boundary_event_window.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Figure 4: R3->R4 rolling industry-pair energy share.
industry_plot = (
    target_industry.head(10)
    .sort_values(
        "energy_share",
        ascending=True,
    )
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.barh(
    industry_plot["industry_pair"],
    industry_plot["energy_share"],
)

ax.set_xlabel("Rolling Boundary Energy Share")

ax.set_title(
    "R3→R4 Rolling Boundary：主要行业组合结构变化贡献",
    fontproperties=CHINESE_FONT,
    fontsize=14,
)

for tick in ax.get_yticklabels():
    tick.set_fontproperties(
        CHINESE_FONT
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "R3_R4_rolling_industry_pair_energy_share.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Figure 5: R3->R4 rolling top node drivers.
node_plot = (
    target_node.head(10)
    .sort_values(
        "driver_share",
        ascending=True,
    )
    .copy()
)

node_plot["plot_label"] = (
    node_plot["name"].astype(str)
    + "\n"
    + node_plot["stock"].astype(str)
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.barh(
    node_plot["plot_label"],
    node_plot["driver_share"],
)

ax.set_xlabel("Rolling Boundary Node-attributed Driver Share")

ax.set_title(
    "R3→R4 Rolling Boundary：主要结构变化节点",
    fontproperties=CHINESE_FONT,
    fontsize=14,
)

for tick in ax.get_yticklabels():
    tick.set_fontproperties(
        CHINESE_FONT
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "R3_R4_rolling_top_node_drivers.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 23. Console summary
# ============================================================

print("\n" + "=" * 82)
print("Within-Regime Summary")
print("=" * 82)

print(
    within_summary.to_string(
        index=False
    )
)

print("\n" + "=" * 82)
print("Regime Boundary Summary")
print("=" * 82)

print(
    boundary_df[
        [
            "boundary",
            "window_prev",
            "window_curr",
            "date_prev",
            "date_curr",
            "edges_prev",
            "edges_curr",
            "lost",
            "gained",
            "gross_support_change",
            "TI_total",
            "TI_same",
            "TI_cross",
            "cross_energy_share",
            "support_energy_share",
            "persistent_energy_share",
            "boundary_TI_rank",
        ]
    ].to_string(
        index=False
    )
)

print("\n" + "=" * 82)
print("R3 -> R4 Rolling Boundary")
print("=" * 82)

print(
    target_transition.to_string(
        index=False
    )
)

print("\nBoundary-vs-within comparison:")
print(
    boundary_vs_within.to_string(
        index=False
    )
)

print("\nTop R3->R4 rolling industry pairs:")
print(
    target_industry[
        [
            "energy_rank",
            "industry_pair",
            "transition_energy",
            "energy_share",
            "lost",
            "gained",
            "persistent",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nTop R3->R4 rolling node drivers:")
print(
    target_node[
        [
            "driver_rank",
            "stock",
            "name",
            "industry",
            "driver_score",
            "driver_share",
            "cross_driver_share",
            "support_change_share",
            "support_turnover",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

if consistency_rows:
    print("\nStage 6 vs Rolling consistency:")
    print(
        pd.DataFrame(
            consistency_rows
        ).to_string(
            index=False
        )
    )

print("\n" + "=" * 82)
print("Stage 7 completed successfully")
print("=" * 82)

print(f"\nChinese font used: {CHINESE_FONT_NAME}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
