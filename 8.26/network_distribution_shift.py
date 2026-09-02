from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Project Paths
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
# 2. Input
# ============================================================

INPUT_FILE = (
    PROCESSED_DIR
    / "regime_cross_validation_window_metrics.csv"
)


# ============================================================
# 3. Outputs
# ============================================================

DETAIL_FILE = (
    PROCESSED_DIR
    / "network_distribution_shift_feature_detail.csv"
)

SHIFT_FILE = (
    PROCESSED_DIR
    / "network_distribution_shift.csv"
)

SUMMARY_FILE = (
    PROCESSED_DIR
    / "network_distribution_shift_summary.csv"
)

EVENT_FILE = (
    PROCESSED_DIR
    / "network_distribution_shift_events.csv"
)


# ============================================================
# 4. Parameters
# ============================================================

# At least this many historical Network Dates
# are required before calculating OOD metrics.
MIN_HISTORY = 8


# Descriptive thresholds only.
MODERATE_Z_THRESHOLD = 2.0
EXTREME_Z_THRESHOLD = 3.0


EPS = 1e-12


# ============================================================
# 5. Feature aliases
#
# Compatible with previous Stage files.
# ============================================================

FEATURE_CANDIDATES = {

    "edge_count": [
        "edge_count",
        "matched_edge_count"
    ],

    "same_edge_ratio": [
        "same_edge_ratio",
        "matched_same_ratio"
    ],

    "same_cross_strength_ratio": [
        "same_cross_strength_ratio",
        "raw_same_cross_strength_ratio"
    ],

    "mean_abs_partial": [
        "mean_abs_partial",
        "raw_mean_abs_partial_all"
    ]
}


# Main features used in Joint OOD Score.
JOINT_FEATURES = [
    "edge_count",
    "same_edge_ratio",
    "same_cross_strength_ratio"
]


# ============================================================
# 6. Load Data
# ============================================================

df_raw = pd.read_csv(
    INPUT_FILE
)


if "network_date" not in df_raw.columns:

    raise ValueError(
        "Input file lacks 'network_date'."
    )


if "regime" not in df_raw.columns:

    raise ValueError(
        "Input file lacks 'regime'."
    )


df_raw[
    "network_date"
] = pd.to_datetime(
    df_raw[
        "network_date"
    ]
)


df_raw[
    "regime"
] = (
    df_raw[
        "regime"
    ]
    .astype(str)
    .str.strip()
)


df_raw = (
    df_raw
    .sort_values(
        "network_date"
    )
    .drop_duplicates(
        subset="network_date"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 7. Resolve Feature Columns
# ============================================================

resolved_features = {}


for standard_name, candidates in FEATURE_CANDIDATES.items():

    matched = next(
        (
            col
            for col in candidates
            if col in df_raw.columns
        ),
        None
    )


    if matched is not None:

        resolved_features[
            standard_name
        ] = matched


print(
    "\nResolved Network Features:"
)


for key, value in resolved_features.items():

    print(
        f"{key:30s} <- {value}"
    )


missing_joint = [
    feature
    for feature in JOINT_FEATURES
    if feature not in resolved_features
]


if missing_joint:

    raise ValueError(
        "Missing main OOD features: "
        f"{missing_joint}"
    )


# ============================================================
# 8. Construct Clean Network Dataset
# ============================================================

network_df = pd.DataFrame(
    {
        "network_date":
            df_raw[
                "network_date"
            ],

        "regime":
            df_raw[
                "regime"
            ]
    }
)


for standard_name, original_name in resolved_features.items():

    network_df[
        standard_name
    ] = pd.to_numeric(
        df_raw[
            original_name
        ],
        errors="coerce"
    )


network_df = (
    network_df
    .sort_values(
        "network_date"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 9. Robust Scale Function
# ============================================================

def robust_scale(
    values: pd.Series
) -> tuple[float, float, float]:
    """
    Return:
        median,
        MAD,
        scaled MAD = 1.4826 * MAD
    """

    values = (
        values
        .dropna()
        .astype(float)
    )


    if len(values) == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )


    median = float(
        values.median()
    )


    mad = float(
        (
            values
            -
            median
        )
        .abs()
        .median()
    )


    scaled_mad = (
        1.4826
        *
        mad
    )


    return (
        median,
        mad,
        scaled_mad
    )


# ============================================================
# 10. Expanding Distribution Shift Metrics
#
# IMPORTANT:
#
# Current date t is NEVER included in historical statistics.
#
# history = dates strictly before t
# ============================================================

detail_rows = []


feature_list = list(
    resolved_features.keys()
)


for pos in range(
    len(
        network_df
    )
):

    current_row = (
        network_df
        .iloc[
            pos
        ]
    )


    network_date = (
        current_row[
            "network_date"
        ]
    )


    regime = (
        current_row[
            "regime"
        ]
    )


    history_df = (
        network_df
        .iloc[
            :pos
        ]
    )


    for feature in feature_list:

        current_value = (
            current_row[
                feature
            ]
        )


        history = (
            history_df[
                feature
            ]
            .dropna()
            .astype(float)
        )


        n_history = len(
            history
        )


        # ----------------------------------------------------
        # Not enough history yet
        # ----------------------------------------------------

        if (
            n_history
            <
            MIN_HISTORY
            or
            pd.isna(
                current_value
            )
        ):

            detail_rows.append(
                {
                    "network_date":
                        network_date,

                    "regime":
                        regime,

                    "feature":
                        feature,

                    "current_value":
                        current_value,

                    "n_history":
                        n_history,

                    "history_mean":
                        np.nan,

                    "history_sd":
                        np.nan,

                    "history_median":
                        np.nan,

                    "history_mad":
                        np.nan,

                    "history_min":
                        np.nan,

                    "history_max":
                        np.nan,

                    "expanding_z":
                        np.nan,

                    "robust_z":
                        np.nan,

                    "historical_percentile":
                        np.nan,

                    "range_break":
                        False,

                    "break_direction":
                        "InsufficientHistory",

                    "support_excursion_sd":
                        np.nan,

                    "moderate_ood":
                        False,

                    "extreme_ood":
                        False
                }
            )

            continue


        # ----------------------------------------------------
        # Historical moments
        # ----------------------------------------------------

        history_mean = float(
            history.mean()
        )


        history_sd = float(
            history.std(
                ddof=1
            )
        )


        history_min = float(
            history.min()
        )


        history_max = float(
            history.max()
        )


        (
            history_median,
            history_mad,
            history_robust_scale
        ) = robust_scale(
            history
        )


        # ----------------------------------------------------
        # Expanding Z-score
        # ----------------------------------------------------

        if (
            pd.notna(
                history_sd
            )
            and
            history_sd
            >
            EPS
        ):

            expanding_z = (
                current_value
                -
                history_mean
            ) / history_sd

        else:

            expanding_z = np.nan


        # ----------------------------------------------------
        # Robust Z-score
        # ----------------------------------------------------

        if (
            pd.notna(
                history_robust_scale
            )
            and
            history_robust_scale
            >
            EPS
        ):

            robust_z = (
                current_value
                -
                history_median
            ) / history_robust_scale

        else:

            robust_z = np.nan


        # ----------------------------------------------------
        # Historical Percentile
        #
        # 1.0:
        # higher than/equal to all previous observations
        #
        # 0.0:
        # lower than all previous observations
        # ----------------------------------------------------

        historical_percentile = float(
            (
                history
                <=
                current_value
            )
            .mean()
        )


        # ----------------------------------------------------
        # Historical Range Break
        # ----------------------------------------------------

        if current_value > history_max:

            range_break = True

            break_direction = "AboveHistoricalMax"


        elif current_value < history_min:

            range_break = True

            break_direction = "BelowHistoricalMin"


        else:

            range_break = False

            break_direction = "InsideHistoricalRange"


        # ----------------------------------------------------
        # Support Excursion
        #
        # How far outside the historical range,
        # standardized by historical SD.
        # ----------------------------------------------------

        if (
            range_break
            and
            history_sd
            >
            EPS
        ):

            if current_value > history_max:

                support_excursion_sd = (
                    current_value
                    -
                    history_max
                ) / history_sd


            else:

                support_excursion_sd = (
                    history_min
                    -
                    current_value
                ) / history_sd


        elif range_break:

            support_excursion_sd = np.nan


        else:

            support_excursion_sd = 0.0


        # ----------------------------------------------------
        # Descriptive OOD Flags
        # ----------------------------------------------------

        moderate_ood = (
            pd.notna(
                expanding_z
            )
            and
            abs(
                expanding_z
            )
            >=
            MODERATE_Z_THRESHOLD
        )


        extreme_ood = (
            pd.notna(
                expanding_z
            )
            and
            abs(
                expanding_z
            )
            >=
            EXTREME_Z_THRESHOLD
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        detail_rows.append(
            {
                "network_date":
                    network_date,

                "regime":
                    regime,

                "feature":
                    feature,

                "current_value":
                    current_value,

                "n_history":
                    n_history,

                "history_mean":
                    history_mean,

                "history_sd":
                    history_sd,

                "history_median":
                    history_median,

                "history_mad":
                    history_mad,

                "history_min":
                    history_min,

                "history_max":
                    history_max,

                "expanding_z":
                    expanding_z,

                "robust_z":
                    robust_z,

                "historical_percentile":
                    historical_percentile,

                "range_break":
                    range_break,

                "break_direction":
                    break_direction,

                "support_excursion_sd":
                    support_excursion_sd,

                "moderate_ood":
                    moderate_ood,

                "extreme_ood":
                    extreme_ood
            }
        )


detail_df = pd.DataFrame(
    detail_rows
)


detail_df.to_csv(
    DETAIL_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. Convert to One-row-per-Network-Date Dataset
# ============================================================

base = (
    network_df[
        [
            "network_date",
            "regime"
        ]
        +
        feature_list
    ]
    .copy()
)


metrics_to_pivot = [
    "expanding_z",
    "robust_z",
    "historical_percentile",
    "range_break",
    "support_excursion_sd"
]


shift_df = base.copy()


for metric in metrics_to_pivot:

    temp = (
        detail_df
        .pivot(
            index="network_date",
            columns="feature",
            values=metric
        )
    )


    temp.columns = [
        f"{feature}_{metric}"
        for feature in temp.columns
    ]


    temp = (
        temp
        .reset_index()
    )


    shift_df = pd.merge(
        shift_df,
        temp,
        on="network_date",
        how="left"
    )


# ============================================================
# 12. Joint OOD Metrics
# ============================================================

joint_z_cols = [
    f"{feature}_expanding_z"
    for feature in JOINT_FEATURES
]


joint_robust_z_cols = [
    f"{feature}_robust_z"
    for feature in JOINT_FEATURES
]


joint_range_cols = [
    f"{feature}_range_break"
    for feature in JOINT_FEATURES
]


joint_excursion_cols = [
    f"{feature}_support_excursion_sd"
    for feature in JOINT_FEATURES
]


# ------------------------------------------------------------
# Maximum absolute expanding Z
# ------------------------------------------------------------

shift_df[
    "joint_max_abs_z"
] = (
    shift_df[
        joint_z_cols
    ]
    .abs()
    .max(
        axis=1
    )
)


# ------------------------------------------------------------
# Maximum absolute robust Z
# ------------------------------------------------------------

shift_df[
    "joint_max_abs_robust_z"
] = (
    shift_df[
        joint_robust_z_cols
    ]
    .abs()
    .max(
        axis=1
    )
)


# ------------------------------------------------------------
# Number of features breaking historical range
# ------------------------------------------------------------

shift_df[
    "n_range_breaks"
] = (
    shift_df[
        joint_range_cols
    ]
    .fillna(
        False
    )
    .astype(int)
    .sum(
        axis=1
    )
)


# ------------------------------------------------------------
# Maximum distance outside historical support
# ------------------------------------------------------------

shift_df[
    "max_support_excursion_sd"
] = (
    shift_df[
        joint_excursion_cols
    ]
    .max(
        axis=1
    )
)


# ------------------------------------------------------------
# Overall descriptive OOD flags
# ------------------------------------------------------------

shift_df[
    "joint_moderate_ood"
] = (
    (
        shift_df[
            "joint_max_abs_z"
        ]
        >=
        MODERATE_Z_THRESHOLD
    )
    |
    (
        shift_df[
            "n_range_breaks"
        ]
        >
        0
    )
)


shift_df[
    "joint_extreme_ood"
] = (
    shift_df[
        "joint_max_abs_z"
    ]
    >=
    EXTREME_Z_THRESHOLD
)


# ============================================================
# 13. Rank Network Dates by Distribution Shift Severity
# ============================================================

shift_df[
    "ood_severity_rank"
] = (
    shift_df[
        "joint_max_abs_z"
    ]
    .rank(
        method="min",
        ascending=False
    )
)


shift_df.to_csv(
    SHIFT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. Regime-level Distribution Shift Summary
# ============================================================

summary_rows = []


for regime, group in shift_df.groupby(
    "regime"
):

    valid_group = (
        group[
            group[
                "joint_max_abs_z"
            ]
            .notna()
        ]
    )


    if valid_group.empty:

        continue


    summary_rows.append(
        {
            "regime":
                regime,

            "n_network_dates":
                len(
                    group
                ),

            "n_scored_dates":
                len(
                    valid_group
                ),

            "mean_joint_max_abs_z":
                valid_group[
                    "joint_max_abs_z"
                ]
                .mean(),

            "median_joint_max_abs_z":
                valid_group[
                    "joint_max_abs_z"
                ]
                .median(),

            "max_joint_max_abs_z":
                valid_group[
                    "joint_max_abs_z"
                ]
                .max(),

            "mean_joint_max_abs_robust_z":
                valid_group[
                    "joint_max_abs_robust_z"
                ]
                .mean(),

            "max_joint_max_abs_robust_z":
                valid_group[
                    "joint_max_abs_robust_z"
                ]
                .max(),

            "n_dates_with_range_break":
                int(
                    (
                        valid_group[
                            "n_range_breaks"
                        ]
                        >
                        0
                    )
                    .sum()
                ),

            "n_moderate_ood_dates":
                int(
                    valid_group[
                        "joint_moderate_ood"
                    ]
                    .sum()
                ),

            "n_extreme_ood_dates":
                int(
                    valid_group[
                        "joint_extreme_ood"
                    ]
                    .sum()
                ),

            "max_support_excursion_sd":
                valid_group[
                    "max_support_excursion_sd"
                ]
                .max()
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# Order regimes by first date
regime_order = (
    shift_df[
        [
            "regime",
            "network_date"
        ]
    ]
    .groupby(
        "regime",
        as_index=False
    )[
        "network_date"
    ]
    .min()
    .sort_values(
        "network_date"
    )[
        "regime"
    ]
    .tolist()
)


summary_df[
    "regime"
] = pd.Categorical(
    summary_df[
        "regime"
    ],
    categories=regime_order,
    ordered=True
)


summary_df = (
    summary_df
    .sort_values(
        "regime"
    )
    .reset_index(
        drop=True
    )
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. Extract Important OOD Events
# ============================================================

event_df = (
    shift_df[
        (
            shift_df[
                "joint_moderate_ood"
            ]
        )
        |
        (
            shift_df[
                "n_range_breaks"
            ]
            >
            0
        )
    ]
    .copy()
    .sort_values(
        [
            "joint_max_abs_z",
            "network_date"
        ],
        ascending=[
            False,
            True
        ]
    )
)


event_df.to_csv(
    EVENT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. Identify R3 -> R4 Boundary
# ============================================================

def find_regime_name(
    candidates: list[str]
) -> str | None:

    existing = set(
        shift_df[
            "regime"
        ]
        .astype(str)
    )


    for candidate in candidates:

        if candidate in existing:

            return candidate


    return None


r3_name = find_regime_name(
    [
        "R3",
        "3",
        "Regime3",
        "Regime 3"
    ]
)


r4_name = find_regime_name(
    [
        "R4",
        "4",
        "Regime4",
        "Regime 4"
    ]
)


first_r4_date = pd.NaT


if r4_name is not None:

    first_r4_date = (
        shift_df.loc[
            shift_df[
                "regime"
            ]
            .astype(str)
            ==
            str(
                r4_name
            ),
            "network_date"
        ]
        .min()
    )


# ============================================================
# 17. Print Main Results
# ============================================================

print(
    "\n=============================================="
)

print(
    "Stage 12 - Stage 1"
)

print(
    "Network Distribution Shift Quantification"
)

print(
    "=============================================="
)


print(
    "\nNumber of Network Dates:",
    len(
        shift_df
    )
)


print(
    "\n--- Regime Summary ---"
)


print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\n--- Top Distribution Shift Dates ---"
)


display_cols = [
    "network_date",
    "regime",

    "edge_count",
    "same_edge_ratio",
    "same_cross_strength_ratio",

    "edge_count_expanding_z",
    "same_edge_ratio_expanding_z",
    "same_cross_strength_ratio_expanding_z",

    "joint_max_abs_z",
    "n_range_breaks",
    "max_support_excursion_sd"
]


available_display_cols = [
    col
    for col in display_cols
    if col in event_df.columns
]


print(
    event_df[
        available_display_cols
    ]
    .head(
        15
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 18. Print R3 -> R4 Specific Diagnostics
# ============================================================

if pd.notna(
    first_r4_date
):

    print(
        "\n--- First R4 Network State ---"
    )


    r4_row = (
        shift_df[
            shift_df[
                "network_date"
            ]
            ==
            first_r4_date
        ]
    )


    print(
        r4_row[
            available_display_cols
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# 19. Figure 1:
# Expanding Z-scores of Main Network Features
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


for feature in JOINT_FEATURES:

    col = (
        f"{feature}_expanding_z"
    )


    ax.plot(
        shift_df[
            "network_date"
        ],
        shift_df[
            col
        ],
        marker="o",
        label=feature
    )


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=MODERATE_Z_THRESHOLD,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=-MODERATE_Z_THRESHOLD,
    linestyle="--",
    linewidth=1
)


if pd.notna(
    first_r4_date
):

    ax.axvline(
        x=first_r4_date,
        linestyle="--",
        linewidth=1
    )


ax.set_xlabel(
    "Network Date"
)


ax.set_ylabel(
    "Expanding Historical Z-score"
)


ax.set_title(
    "Network Feature Distribution Shift Over Time"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "network_feature_expanding_zscores.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 20. Figure 2:
# Joint OOD Score
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    shift_df[
        "network_date"
    ],
    shift_df[
        "joint_max_abs_z"
    ],
    marker="o"
)


ax.axhline(
    y=MODERATE_Z_THRESHOLD,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=EXTREME_Z_THRESHOLD,
    linestyle="--",
    linewidth=1
)


if pd.notna(
    first_r4_date
):

    ax.axvline(
        x=first_r4_date,
        linestyle="--",
        linewidth=1
    )


ax.set_xlabel(
    "Network Date"
)


ax.set_ylabel(
    "Joint OOD Score = Max |Historical Z|"
)


ax.set_title(
    "Joint Network Distribution-shift Score"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "joint_network_ood_score.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 21. Figure 3:
# Historical Support Excursion
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    shift_df[
        "network_date"
    ],
    shift_df[
        "max_support_excursion_sd"
    ],
    marker="o"
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


if pd.notna(
    first_r4_date
):

    ax.axvline(
        x=first_r4_date,
        linestyle="--",
        linewidth=1
    )


ax.set_xlabel(
    "Network Date"
)


ax.set_ylabel(
    "Maximum Support Excursion (Historical SD)"
)


ax.set_title(
    "Distance Outside Historical Network-feature Support"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "network_support_excursion.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 22. Complete
# ============================================================

print(
    "\n=============================================="
)

print(
    "Stage 1 Complete"
)

print(
    "=============================================="
)


for path in [
    DETAIL_FILE,
    SHIFT_FILE,
    SUMMARY_FILE,
    EVENT_FILE
]:

    print(
        path
    )