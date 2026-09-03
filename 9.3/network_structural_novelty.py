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

# Stage 7 main transition file.
# Stage 8 can still run without it, but the joint
# Transition-vs-Novelty analysis will be skipped.
STAGE7_TRANSITION_CANDIDATES = [
    BASE_DIR
        / "stage7_regime_conditioned_rolling"
        / "rolling_regime_transition_summary.csv"
]

OUTPUT_DIR = BASE_DIR / "stage8_network_structural_novelty"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Settings
# ============================================================

EDGE_TOL = 1e-8
MIN_HISTORY_FOR_STANDARDIZATION = 8

REGIME_BY_WINDOW = {
    "R1": range(1, 8),
    "R2": range(8, 13),
    "R3": range(13, 23),
    "R4": range(23, 27),
    "R5": range(27, 32),
}

PRE_R4_LAST_WINDOW = 22
FIRST_R4_WINDOW = 23


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
        raise ValueError(f"Unable to parse boolean values: {bad}")

    return converted.astype(bool)


def regime_from_window(window_id: int) -> str:
    for regime, window_range in REGIME_BY_WINDOW.items():
        if window_id in window_range:
            return regime
    raise ValueError(f"window_id={window_id} is outside R1--R5.")


def canonical_stock_pair(stock_i: str, stock_j: str) -> str:
    a, b = sorted([clean_stock(stock_i), clean_stock(stock_j)])
    return f"{a}|{b}"


def canonical_industry_pair(
    industry_i: str,
    industry_j: str,
) -> tuple[str, str]:
    a, b = sorted([str(industry_i).strip(), str(industry_j).strip()])
    return a, b


def safe_jaccard(
    support_a: np.ndarray,
    support_b: np.ndarray,
) -> float:
    support_a = support_a.astype(bool)
    support_b = support_b.astype(bool)

    union = np.logical_or(support_a, support_b).sum()

    if union == 0:
        return 1.0

    intersection = np.logical_and(support_a, support_b).sum()

    return float(intersection / union)


def robust_zscore(
    value: float,
    history: np.ndarray,
) -> float:
    history = np.asarray(history, dtype=float)
    history = history[np.isfinite(history)]

    if len(history) == 0:
        return np.nan

    median = float(np.median(history))
    mad = float(np.median(np.abs(history - median)))

    if mad <= 0:
        return np.nan

    robust_scale = 1.4826 * mad

    return float((value - median) / robust_scale)


def empirical_percentile(
    value: float,
    history: np.ndarray,
) -> float:
    history = np.asarray(history, dtype=float)
    history = history[np.isfinite(history)]

    if len(history) == 0:
        return np.nan

    return float(np.mean(history <= value))


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
# 5. Load the rolling GLasso history
# ============================================================

ROLLING_EDGE_FILE = first_existing(ROLLING_EDGE_CANDIDATES)

if ROLLING_EDGE_FILE is None:
    raise FileNotFoundError(
        "Cannot find rolling_glasso_edge_history.csv.\n"
        "Please check ROLLING_EDGE_CANDIDATES."
    )

rolling = read_csv_fallback(ROLLING_EDGE_FILE)

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

missing = [
    col
    for col in REQUIRED_COLUMNS
    if col not in rolling.columns
]

if missing:
    raise ValueError(
        "Rolling file does not match the expected schema.\n"
        f"Missing columns: {missing}\n"
        f"Available columns: {list(rolling.columns)}"
    )


# ============================================================
# 6. Clean and validate
# ============================================================

rolling["window_id"] = pd.to_numeric(
    rolling["window_id"],
    errors="raise",
).astype(int)

for col in ["window_start", "window_end", "network_date"]:
    rolling[col] = pd.to_datetime(
        rolling[col],
        errors="raise",
    )

for col in ["stock_1", "stock_2"]:
    rolling[col] = rolling[col].apply(clean_stock)

for col in ["precision", "partial_correlation", "abs_partial_correlation"]:
    rolling[col] = pd.to_numeric(
        rolling[col],
        errors="raise",
    )

rolling["selected"] = to_bool(rolling["selected"])
rolling["same_industry"] = to_bool(rolling["same_industry"])

for col in ["name_1", "name_2", "industry_1", "industry_2"]:
    rolling[col] = rolling[col].astype(str).str.strip()

rolling["regime"] = rolling["window_id"].apply(regime_from_window)

rolling["pair_key"] = rolling.apply(
    lambda row: canonical_stock_pair(
        row["stock_1"],
        row["stock_2"],
    ),
    axis=1,
)

industry_pairs = rolling.apply(
    lambda row: canonical_industry_pair(
        row["industry_1"],
        row["industry_2"],
    ),
    axis=1,
)

rolling["industry_a"] = [x[0] for x in industry_pairs]
rolling["industry_b"] = [x[1] for x in industry_pairs]

pairs_per_window = rolling.groupby("window_id").size()

if not (pairs_per_window == 105).all():
    raise ValueError(
        "Not every rolling window contains exactly 105 stock pairs.\n"
        f"{pairs_per_window}"
    )

duplicate_pairs = rolling.duplicated(
    subset=["window_id", "pair_key"]
).sum()

if duplicate_pairs != 0:
    raise ValueError(
        f"Found {duplicate_pairs} duplicate pairs within rolling windows."
    )


# ============================================================
# 7. Window and pair metadata
# ============================================================

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

pair_metadata = (
    rolling[
        [
            "pair_key",
            "stock_1",
            "name_1",
            "industry_1",
            "stock_2",
            "name_2",
            "industry_2",
            "same_industry",
            "industry_a",
            "industry_b",
        ]
    ]
    .drop_duplicates(subset=["pair_key"])
    .sort_values("pair_key")
    .reset_index(drop=True)
)

if len(pair_metadata) != 105:
    raise ValueError(
        f"Expected 105 unique stock pairs, found {len(pair_metadata)}."
    )

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
    pd.concat(
        [metadata_1, metadata_2],
        ignore_index=True,
    )
    .drop_duplicates()
    .sort_values("stock")
    .drop_duplicates(subset=["stock"])
    .set_index("stock")
)

stocks = stock_metadata.index.tolist()


# ============================================================
# 8. Construct 31 complete weighted-network state vectors
# ============================================================

weighted_state = (
    rolling.pivot(
        index="window_id",
        columns="pair_key",
        values="partial_correlation",
    )
    .sort_index()
)

support_state = (
    rolling.pivot(
        index="window_id",
        columns="pair_key",
        values="selected",
    )
    .sort_index()
)

pair_order = weighted_state.columns.tolist()

support_state = support_state[
    pair_order
].astype(bool)

pair_metadata = (
    pair_metadata
    .set_index("pair_key")
    .loc[pair_order]
    .reset_index()
)

if weighted_state.isna().any().any():
    raise ValueError(
        "Weighted network state matrix contains missing values."
    )

if support_state.isna().any().any():
    raise ValueError(
        "Support network state matrix contains missing values."
    )

same_mask = pair_metadata[
    "same_industry"
].to_numpy(dtype=bool)

cross_mask = ~same_mask


# ============================================================
# 9. Expanding historical structural novelty
# ============================================================

novelty_rows = []
industry_novelty_rows = []
node_novelty_rows = []

# Store prior expanding-centroid novelty values for percentile/z calibration.
# Important: after the second network is processed, its novelty is appended.
past_centroid_distances = []


for pos in range(len(window_table)):

    row_info = window_table.iloc[pos]
    window_id = int(row_info["window_id"])

    current_vector = weighted_state.loc[
        window_id
    ].to_numpy(dtype=float)

    current_support = support_state.loc[
        window_id
    ].to_numpy(dtype=bool)

    if pos == 0:
        novelty_rows.append(
            {
                "window_id": window_id,
                "network_date": row_info["network_date"],
                "regime": row_info["regime"],
                "n_prior_states": 0,
                "centroid_novelty": np.nan,
                "nearest_state_distance": np.nan,
                "nearest_prior_window": np.nan,
                "nearest_prior_regime": None,
                "nearest_prior_date": pd.NaT,
                "max_historical_support_jaccard": np.nan,
                "support_novelty": np.nan,
                "same_centroid_novelty": np.nan,
                "cross_centroid_novelty": np.nan,
                "same_novelty_energy_share": np.nan,
                "cross_novelty_energy_share": np.nan,
                "centroid_novelty_percentile": np.nan,
                "centroid_novelty_robust_z": np.nan,
                "is_regime_start": True,
                "is_first_R4_state": False,
            }
        )
        continue

    prior_window_ids = (
        window_table.iloc[:pos]["window_id"]
        .astype(int)
        .tolist()
    )

    prior_vectors = weighted_state.loc[
        prior_window_ids
    ].to_numpy(dtype=float)

    prior_supports = support_state.loc[
        prior_window_ids
    ].to_numpy(dtype=bool)

    historical_centroid = prior_vectors.mean(axis=0)

    delta_from_centroid = (
        current_vector - historical_centroid
    )

    pair_energy = delta_from_centroid ** 2

    total_energy = float(pair_energy.sum())

    centroid_novelty = float(np.sqrt(total_energy))

    same_energy = float(
        pair_energy[same_mask].sum()
    )

    cross_energy = float(
        pair_energy[cross_mask].sum()
    )

    distances_to_prior = np.sqrt(
        np.sum(
            (
                prior_vectors
                - current_vector[None, :]
            )
            ** 2,
            axis=1,
        )
    )

    nearest_pos = int(
        np.argmin(distances_to_prior)
    )

    nearest_window = int(
        prior_window_ids[nearest_pos]
    )

    nearest_info = (
        window_table.loc[
            window_table["window_id"]
            == nearest_window
        ]
        .iloc[0]
    )

    nearest_state_distance = float(
        distances_to_prior[nearest_pos]
    )

    support_jaccards = np.asarray(
        [
            safe_jaccard(
                current_support,
                prior_support,
            )
            for prior_support in prior_supports
        ],
        dtype=float,
    )

    max_support_jaccard = float(
        np.max(support_jaccards)
    )

    support_novelty = float(
        1.0 - max_support_jaccard
    )

    if (
        len(past_centroid_distances)
        >= MIN_HISTORY_FOR_STANDARDIZATION
    ):
        novelty_percentile = empirical_percentile(
            centroid_novelty,
            np.asarray(past_centroid_distances),
        )

        novelty_robust_z = robust_zscore(
            centroid_novelty,
            np.asarray(past_centroid_distances),
        )
    else:
        novelty_percentile = np.nan
        novelty_robust_z = np.nan

    regime_prev = str(
        window_table.iloc[pos - 1]["regime"]
    )

    regime_curr = str(
        row_info["regime"]
    )

    is_regime_start = (
        regime_prev != regime_curr
    )

    novelty_rows.append(
        {
            "window_id": window_id,
            "network_date": row_info["network_date"],
            "regime": regime_curr,
            "n_prior_states": len(prior_window_ids),
            "centroid_novelty": centroid_novelty,
            "nearest_state_distance": nearest_state_distance,
            "nearest_prior_window": nearest_window,
            "nearest_prior_regime": nearest_info["regime"],
            "nearest_prior_date": nearest_info["network_date"],
            "max_historical_support_jaccard": max_support_jaccard,
            "support_novelty": support_novelty,
            "same_centroid_novelty": np.sqrt(same_energy),
            "cross_centroid_novelty": np.sqrt(cross_energy),
            "same_novelty_energy_share": (
                same_energy / total_energy
                if total_energy > 0
                else np.nan
            ),
            "cross_novelty_energy_share": (
                cross_energy / total_energy
                if total_energy > 0
                else np.nan
            ),
            "centroid_novelty_percentile": novelty_percentile,
            "centroid_novelty_robust_z": novelty_robust_z,
            "is_regime_start": is_regime_start,
            "is_first_R4_state": (
                window_id == FIRST_R4_WINDOW
            ),
        }
    )

    temp = pair_metadata.copy()
    temp["novelty_energy"] = pair_energy

    for (
        industry_a,
        industry_b,
    ), group in temp.groupby(
        ["industry_a", "industry_b"],
        sort=True,
    ):
        energy = float(
            group["novelty_energy"].sum()
        )

        industry_novelty_rows.append(
            {
                "window_id": window_id,
                "network_date": row_info["network_date"],
                "regime": regime_curr,
                "industry_a": industry_a,
                "industry_b": industry_b,
                "industry_pair": (
                    f"{industry_a} - {industry_b}"
                ),
                "relation": (
                    "Same"
                    if industry_a == industry_b
                    else "Cross"
                ),
                "novelty_energy": energy,
                "novelty_energy_share": (
                    energy / total_energy
                    if total_energy > 0
                    else np.nan
                ),
            }
        )

    pair_temp = pair_metadata.copy()
    pair_temp["novelty_energy"] = pair_energy

    for stock in stocks:

        incident = pair_temp.loc[
            (
                pair_temp["stock_1"]
                == stock
            )
            |
            (
                pair_temp["stock_2"]
                == stock
            )
        ].copy()

        node_energy = float(
            incident["novelty_energy"].sum()
        )

        node_same_energy = float(
            incident.loc[
                incident["same_industry"],
                "novelty_energy",
            ].sum()
        )

        node_cross_energy = float(
            incident.loc[
                ~incident["same_industry"],
                "novelty_energy",
            ].sum()
        )

        node_novelty_rows.append(
            {
                "window_id": window_id,
                "network_date": row_info["network_date"],
                "regime": regime_curr,
                "stock": stock,
                "name": stock_metadata.at[stock, "name"],
                "industry": stock_metadata.at[stock, "industry"],
                "node_novelty_score": node_energy,
                "node_novelty_share": (
                    node_energy / (2.0 * total_energy)
                    if total_energy > 0
                    else np.nan
                ),
                "same_node_novelty_score": node_same_energy,
                "cross_node_novelty_score": node_cross_energy,
                "cross_node_novelty_share": (
                    node_cross_energy / node_energy
                    if node_energy > 0
                    else np.nan
                ),
            }
        )

    past_centroid_distances.append(
        centroid_novelty
    )


novelty_df = pd.DataFrame(novelty_rows)
industry_novelty_df = pd.DataFrame(industry_novelty_rows)
node_novelty_df = pd.DataFrame(node_novelty_rows)

if len(node_novelty_df) > 0:
    node_novelty_df["node_novelty_rank"] = (
        node_novelty_df.groupby(
            "window_id"
        )["node_novelty_score"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

if len(industry_novelty_df) > 0:
    industry_novelty_df["industry_novelty_rank"] = (
        industry_novelty_df.groupby(
            "window_id"
        )["novelty_energy"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )


# ============================================================
# 10. Frozen pre-R4 historical reference
# ============================================================

pre_r4_window_ids = (
    window_table.loc[
        window_table["window_id"]
        <= PRE_R4_LAST_WINDOW,
        "window_id",
    ]
    .astype(int)
    .tolist()
)

pre_r4_vectors = weighted_state.loc[
    pre_r4_window_ids
].to_numpy(dtype=float)

pre_r4_supports = support_state.loc[
    pre_r4_window_ids
].to_numpy(dtype=bool)

pre_r4_centroid = pre_r4_vectors.mean(axis=0)

frozen_rows = []

post_boundary_windows = window_table.loc[
    window_table["window_id"]
    >= FIRST_R4_WINDOW
]

for row_info in post_boundary_windows.itertuples(
    index=False
):

    window_id = int(row_info.window_id)

    current_vector = weighted_state.loc[
        window_id
    ].to_numpy(dtype=float)

    current_support = support_state.loc[
        window_id
    ].to_numpy(dtype=bool)

    delta = (
        current_vector - pre_r4_centroid
    )

    pair_energy = delta ** 2

    total_energy = float(
        pair_energy.sum()
    )

    distances_to_pre_r4 = np.sqrt(
        np.sum(
            (
                pre_r4_vectors
                - current_vector[None, :]
            )
            ** 2,
            axis=1,
        )
    )

    nearest_pos = int(
        np.argmin(
            distances_to_pre_r4
        )
    )

    nearest_window = int(
        pre_r4_window_ids[nearest_pos]
    )

    support_jaccards = np.asarray(
        [
            safe_jaccard(
                current_support,
                hist_support,
            )
            for hist_support in pre_r4_supports
        ]
    )

    same_energy = float(
        pair_energy[same_mask].sum()
    )

    cross_energy = float(
        pair_energy[cross_mask].sum()
    )

    frozen_rows.append(
        {
            "window_id": window_id,
            "network_date": row_info.network_date,
            "regime": row_info.regime,
            "pre_R4_centroid_distance": np.sqrt(
                total_energy
            ),
            "pre_R4_nearest_state_distance": float(
                distances_to_pre_r4[nearest_pos]
            ),
            "nearest_pre_R4_window": nearest_window,
            "max_pre_R4_support_jaccard": float(
                np.max(support_jaccards)
            ),
            "pre_R4_support_novelty": float(
                1.0 - np.max(support_jaccards)
            ),
            "pre_R4_same_novelty": np.sqrt(
                same_energy
            ),
            "pre_R4_cross_novelty": np.sqrt(
                cross_energy
            ),
            "pre_R4_cross_energy_share": (
                cross_energy / total_energy
                if total_energy > 0
                else np.nan
            ),
        }
    )

pre_r4_novelty_df = pd.DataFrame(
    frozen_rows
)

pre_r4_regime_summary = (
    pre_r4_novelty_df.groupby(
        "regime",
        as_index=False,
    )
    .agg(
        n_networks=("window_id", "size"),
        mean_pre_R4_centroid_distance=(
            "pre_R4_centroid_distance",
            "mean",
        ),
        median_pre_R4_centroid_distance=(
            "pre_R4_centroid_distance",
            "median",
        ),
        mean_pre_R4_nearest_state_distance=(
            "pre_R4_nearest_state_distance",
            "mean",
        ),
        mean_pre_R4_support_novelty=(
            "pre_R4_support_novelty",
            "mean",
        ),
        mean_pre_R4_cross_energy_share=(
            "pre_R4_cross_energy_share",
            "mean",
        ),
    )
)


# ============================================================
# 11. Regime-level expanding novelty summary
# ============================================================

regime_novelty_summary = (
    novelty_df.loc[
        novelty_df["window_id"] >= 2
    ]
    .groupby(
        "regime",
        as_index=False,
    )
    .agg(
        n_network_states=(
            "window_id",
            "size",
        ),
        mean_centroid_novelty=(
            "centroid_novelty",
            "mean",
        ),
        median_centroid_novelty=(
            "centroid_novelty",
            "median",
        ),
        max_centroid_novelty=(
            "centroid_novelty",
            "max",
        ),
        mean_nearest_state_distance=(
            "nearest_state_distance",
            "mean",
        ),
        mean_support_novelty=(
            "support_novelty",
            "mean",
        ),
        mean_cross_novelty_energy_share=(
            "cross_novelty_energy_share",
            "mean",
        ),
        mean_centroid_novelty_robust_z=(
            "centroid_novelty_robust_z",
            "mean",
        ),
    )
)


# ============================================================
# 12. Merge Stage 7 Transition Intensity
# ============================================================

STAGE7_TRANSITION_FILE = first_existing(
    STAGE7_TRANSITION_CANDIDATES
)

joint_df = novelty_df.copy()

if STAGE7_TRANSITION_FILE is not None:

    stage7 = read_csv_fallback(
        STAGE7_TRANSITION_FILE
    )

    required_stage7 = [
        "window_curr",
        "TI_total",
        "transition_type",
        "cross_energy_share",
        "support_energy_share",
        "persistent_energy_share",
    ]

    if all(
        col in stage7.columns
        for col in required_stage7
    ):

        stage7_small = stage7[
            required_stage7
        ].copy()

        stage7_small["window_curr"] = pd.to_numeric(
            stage7_small["window_curr"],
            errors="raise",
        ).astype(int)

        stage7_small = stage7_small.rename(
            columns={
                "window_curr": "window_id",
                "cross_energy_share": (
                    "transition_cross_energy_share"
                ),
                "support_energy_share": (
                    "transition_support_energy_share"
                ),
                "persistent_energy_share": (
                    "transition_persistent_energy_share"
                ),
            }
        )

        joint_df = joint_df.merge(
            stage7_small,
            on="window_id",
            how="left",
            validate="one_to_one",
        )

        classes = []

        for _, row in joint_df.iterrows():

            ti_value = row.get(
                "TI_total",
                np.nan,
            )

            novelty_value = row.get(
                "centroid_novelty",
                np.nan,
            )

            if (
                not np.isfinite(ti_value)
                or
                not np.isfinite(novelty_value)
            ):
                classes.append(None)
                continue

            previous = joint_df.loc[
                joint_df["window_id"]
                <
                row["window_id"]
            ]

            previous = previous.loc[
                previous["TI_total"].notna()
                &
                previous[
                    "centroid_novelty"
                ].notna()
            ]

            if (
                len(previous)
                <
                MIN_HISTORY_FOR_STANDARDIZATION
            ):
                classes.append(None)
                continue

            ti_threshold = float(
                previous["TI_total"].median()
            )

            novelty_threshold = float(
                previous[
                    "centroid_novelty"
                ].median()
            )

            high_ti = (
                ti_value > ti_threshold
            )

            high_novelty = (
                novelty_value > novelty_threshold
            )

            if high_ti and high_novelty:
                label = (
                    "High transition / High novelty"
                )
            elif high_ti and not high_novelty:
                label = (
                    "High transition / Familiar state"
                )
            elif (
                not high_ti
                and high_novelty
            ):
                label = (
                    "Low transition / High novelty"
                )
            else:
                label = (
                    "Routine evolution"
                )

            classes.append(label)

        joint_df[
            "transition_novelty_class"
        ] = classes


# ============================================================
# 13. R4 structural-novelty persistence
# ============================================================

r4_persistence = (
    pre_r4_novelty_df.loc[
        pre_r4_novelty_df["regime"]
        ==
        "R4"
    ]
    .copy()
)

if len(r4_persistence) > 0:

    first_r4_distance = float(
        r4_persistence.iloc[0][
            "pre_R4_centroid_distance"
        ]
    )

    r4_persistence[
        "relative_to_first_R4_novelty"
    ] = (
        r4_persistence[
            "pre_R4_centroid_distance"
        ]
        /
        first_r4_distance
        if first_r4_distance > 0
        else np.nan
    )


# ============================================================
# 14. Save tables
# ============================================================

novelty_df.to_csv(
    OUTPUT_DIR
    / "rolling_network_novelty_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

regime_novelty_summary.to_csv(
    OUTPUT_DIR
    / "regime_novelty_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

pre_r4_novelty_df.to_csv(
    OUTPUT_DIR
    / "pre_R4_reference_network_novelty.csv",
    index=False,
    encoding="utf-8-sig",
)

pre_r4_regime_summary.to_csv(
    OUTPUT_DIR
    / "pre_R4_reference_regime_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

r4_persistence.to_csv(
    OUTPUT_DIR
    / "R4_structural_novelty_persistence.csv",
    index=False,
    encoding="utf-8-sig",
)

industry_novelty_df.to_csv(
    OUTPUT_DIR
    / "rolling_industry_pair_novelty.csv",
    index=False,
    encoding="utf-8-sig",
)

node_novelty_df.to_csv(
    OUTPUT_DIR
    / "rolling_node_novelty_scores.csv",
    index=False,
    encoding="utf-8-sig",
)

joint_df.to_csv(
    OUTPUT_DIR
    / "transition_novelty_joint_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

first_r4_industry = (
    industry_novelty_df.loc[
        industry_novelty_df[
            "window_id"
        ]
        ==
        FIRST_R4_WINDOW
    ]
    .sort_values(
        "novelty_energy",
        ascending=False,
    )
)

first_r4_node = (
    node_novelty_df.loc[
        node_novelty_df[
            "window_id"
        ]
        ==
        FIRST_R4_WINDOW
    ]
    .sort_values(
        "node_novelty_score",
        ascending=False,
    )
)

first_r4_industry.to_csv(
    OUTPUT_DIR
    / "R4_first_state_industry_novelty.csv",
    index=False,
    encoding="utf-8-sig",
)

first_r4_node.to_csv(
    OUTPUT_DIR
    / "R4_first_state_node_novelty.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 15. Figures
# ============================================================

plot_df = novelty_df.loc[
    novelty_df[
        "centroid_novelty"
    ].notna()
].copy()

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.plot(
    plot_df["network_date"],
    plot_df["centroid_novelty"],
    marker="o",
)

regime_starts = plot_df.loc[
    plot_df["is_regime_start"]
]

ax.scatter(
    regime_starts["network_date"],
    regime_starts["centroid_novelty"],
    s=100,
    zorder=5,
)

for row in regime_starts.itertuples():

    ax.annotate(
        str(row.regime),
        (
            row.network_date,
            row.centroid_novelty,
        ),
        xytext=(5, 7),
        textcoords="offset points",
        fontproperties=CHINESE_FONT,
        fontsize=9,
    )

ax.set_title(
    "Rolling Network Structural Novelty Relative to Prior History",
    fontsize=14,
)

ax.set_xlabel("Network Date")
ax.set_ylabel("Distance to Historical Centroid")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "rolling_network_centroid_novelty.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    pre_r4_novelty_df["network_date"],
    pre_r4_novelty_df[
        "pre_R4_centroid_distance"
    ],
    marker="o",
)

ax.set_title(
    "R4/R5 Network Novelty Relative to the Frozen Pre-R4 Reference",
    fontsize=14,
)

ax.set_xlabel("Network Date")
ax.set_ylabel("Distance to Pre-R4 Centroid")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "pre_R4_reference_novelty_path.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


if "TI_total" in joint_df.columns:

    scatter_df = joint_df.loc[
        joint_df["TI_total"].notna()
        &
        joint_df[
            "centroid_novelty"
        ].notna()
    ].copy()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        scatter_df["TI_total"],
        scatter_df["centroid_novelty"],
    )

    first_r4 = scatter_df.loc[
        scatter_df["window_id"]
        ==
        FIRST_R4_WINDOW
    ]

    if len(first_r4) == 1:

        row = first_r4.iloc[0]

        ax.annotate(
            "R3→R4",
            (
                row["TI_total"],
                row["centroid_novelty"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontproperties=CHINESE_FONT,
        )

    ax.set_xlabel(
        "Stage 7 Transition Intensity"
    )

    ax.set_ylabel(
        "Stage 8 Structural Novelty"
    )

    ax.set_title(
        "Local Network Change vs Historical Structural Novelty",
        fontsize=14,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "transition_intensity_vs_structural_novelty.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if len(first_r4_industry) > 0:

    industry_plot = (
        first_r4_industry.head(10)
        .sort_values(
            "novelty_energy_share",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        industry_plot["industry_pair"],
        industry_plot["novelty_energy_share"],
    )

    ax.set_xlabel(
        "Novelty Energy Share"
    )

    ax.set_title(
        "首个 R4 Rolling 状态的行业组合结构新颖性贡献",
        fontproperties=CHINESE_FONT,
        fontsize=14,
    )

    for tick in ax.get_yticklabels():
        tick.set_fontproperties(
            CHINESE_FONT
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "R4_first_state_industry_novelty.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if len(first_r4_node) > 0:

    node_plot = (
        first_r4_node.head(10)
        .sort_values(
            "node_novelty_share",
            ascending=True,
        )
        .copy()
    )

    node_plot["plot_label"] = (
        node_plot["name"].astype(str)
        +
        "\n"
        +
        node_plot["stock"].astype(str)
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        node_plot["plot_label"],
        node_plot["node_novelty_share"],
    )

    ax.set_xlabel(
        "Node-attributed Novelty Share"
    )

    ax.set_title(
        "首个 R4 Rolling 状态的主要结构新颖性节点",
        fontproperties=CHINESE_FONT,
        fontsize=14,
    )

    for tick in ax.get_yticklabels():
        tick.set_fontproperties(
            CHINESE_FONT
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "R4_first_state_node_novelty.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 16. Console summary
# ============================================================

print("\n" + "=" * 84)
print("Stage 8: Network-native Structural Novelty")
print("=" * 84)

print(f"Input file: {ROLLING_EDGE_FILE.resolve()}")
print(f"Rolling networks: {len(window_table)}")
print(f"Stock pairs per network: {len(pair_order)}")

print("\n" + "=" * 84)
print("Regime-level expanding novelty summary")
print("=" * 84)

print(
    regime_novelty_summary.to_string(
        index=False
    )
)

print("\n" + "=" * 84)
print("Frozen pre-R4 reference summary")
print("=" * 84)

print(
    pre_r4_regime_summary.to_string(
        index=False
    )
)

first_r4_summary = novelty_df.loc[
    novelty_df["window_id"]
    ==
    FIRST_R4_WINDOW
]

if len(first_r4_summary) == 1:

    print("\n" + "=" * 84)
    print("First R4 network state")
    print("=" * 84)

    print(
        first_r4_summary[
            [
                "window_id",
                "network_date",
                "regime",
                "centroid_novelty",
                "nearest_state_distance",
                "nearest_prior_window",
                "nearest_prior_regime",
                "support_novelty",
                "cross_novelty_energy_share",
                "centroid_novelty_percentile",
                "centroid_novelty_robust_z",
            ]
        ].to_string(
            index=False
        )
    )

print(
    "\nTop first-R4 industry novelty contributions:"
)

print(
    first_r4_industry[
        [
            "industry_novelty_rank",
            "industry_pair",
            "relation",
            "novelty_energy",
            "novelty_energy_share",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print(
    "\nTop first-R4 node novelty contributions:"
)

print(
    first_r4_node[
        [
            "node_novelty_rank",
            "stock",
            "name",
            "industry",
            "node_novelty_score",
            "node_novelty_share",
            "cross_node_novelty_share",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

if "TI_total" in joint_df.columns:

    first_r4_joint = joint_df.loc[
        joint_df["window_id"]
        ==
        FIRST_R4_WINDOW
    ]

    if len(first_r4_joint) == 1:

        print("\n" + "=" * 84)
        print(
            "Stage 7 Transition vs Stage 8 Novelty at R3->R4"
        )
        print("=" * 84)

        columns_to_show = [
            "window_id",
            "network_date",
            "regime",
            "TI_total",
            "centroid_novelty",
            "nearest_state_distance",
            "support_novelty",
            "transition_novelty_class",
        ]

        columns_to_show = [
            col
            for col in columns_to_show
            if col in first_r4_joint.columns
        ]

        print(
            first_r4_joint[
                columns_to_show
            ].to_string(
                index=False
            )
        )

print("\n" + "=" * 84)
print("Stage 8 completed successfully")
print("=" * 84)

print(
    f"\nChinese font used: "
    f"{CHINESE_FONT_NAME}"
)

print(
    f"Output directory: "
    f"{OUTPUT_DIR.resolve()}"
)
