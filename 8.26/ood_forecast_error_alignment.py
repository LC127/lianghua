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
# 2. Input Files
# ============================================================

SHIFT_FILE = (
    PROCESSED_DIR
    / "network_distribution_shift.csv"
)

PREDICTION_FILE = (
    PROCESSED_DIR
    / "incremental_prediction_walk_forward.csv"
)


# ============================================================
# 3. Output Files
# ============================================================

ALIGNMENT_FILE = (
    PROCESSED_DIR
    / "ood_forecast_error_alignment.csv"
)

CORRELATION_FILE = (
    PROCESSED_DIR
    / "ood_forecast_error_correlation.csv"
)

GROUP_SUMMARY_FILE = (
    PROCESSED_DIR
    / "ood_forecast_error_group_summary.csv"
)

EVENT_FILE = (
    PROCESSED_DIR
    / "ood_forecast_error_top_events.csv"
)

SENSITIVITY_FILE = (
    PROCESSED_DIR
    / "ood_forecast_error_sensitivity.csv"
)


# ============================================================
# 4. Model Names
# ============================================================

BASELINE_MODEL = (
    "Baseline_CurrentRelativeVol"
)

MAIN_NETWORK_MODEL = (
    "Augmented_SameEdgeRatio"
)

SECONDARY_NETWORK_MODEL = (
    "Augmented_EdgeCount"
)


# ============================================================
# 5. OOD thresholds
#
# Descriptive thresholds only.
# ============================================================

MODERATE_THRESHOLD = 2.0
EXTREME_THRESHOLD = 3.0

EPS = 1e-12


# ============================================================
# 6. Load Distribution Shift Data
# ============================================================

shift_df = pd.read_csv(
    SHIFT_FILE
)


if "network_date" not in shift_df.columns:

    raise ValueError(
        "network_distribution_shift.csv "
        "lacks network_date."
    )


shift_df[
    "network_date"
] = pd.to_datetime(
    shift_df[
        "network_date"
    ]
)


# ------------------------------------------------------------
# Required OOD variables
# ------------------------------------------------------------

required_shift_cols = [
    "network_date",
    "regime",
    "joint_max_abs_z",
    "n_range_breaks"
]


missing_shift = [
    col
    for col in required_shift_cols
    if col not in shift_df.columns
]


if missing_shift:

    raise ValueError(
        "Distribution-shift file lacks: "
        f"{missing_shift}"
    )


# ------------------------------------------------------------
# Feature-specific Z columns
# ------------------------------------------------------------

feature_specific_cols = [
    "same_edge_ratio_expanding_z",
    "edge_count_expanding_z",
    "same_cross_strength_ratio_expanding_z",
    "max_support_excursion_sd"
]


for col in feature_specific_cols:

    if col not in shift_df.columns:

        shift_df[
            col
        ] = np.nan


# ============================================================
# 7. Load Walk-forward Predictions
# ============================================================

pred_df = pd.read_csv(
    PREDICTION_FILE
)


required_prediction_cols = [
    "model",
    "network_date",
    "actual",
    "prediction",
    "absolute_error",
    "squared_error"
]


missing_pred = [
    col
    for col in required_prediction_cols
    if col not in pred_df.columns
]


if missing_pred:

    raise ValueError(
        "Walk-forward prediction file lacks: "
        f"{missing_pred}"
    )


pred_df[
    "network_date"
] = pd.to_datetime(
    pred_df[
        "network_date"
    ]
)


pred_df[
    "model"
] = (
    pred_df[
        "model"
    ]
    .astype(str)
    .str.strip()
)


# ============================================================
# 8. Check Models
# ============================================================

available_models = set(
    pred_df[
        "model"
    ]
)


if BASELINE_MODEL not in available_models:

    raise ValueError(
        f"Missing baseline model: {BASELINE_MODEL}"
    )


if MAIN_NETWORK_MODEL not in available_models:

    raise ValueError(
        f"Missing main model: {MAIN_NETWORK_MODEL}"
    )


print(
    "\nAvailable Models:"
)

for model in sorted(
    available_models
):

    print(
        model
    )


# ============================================================
# 9. Reshape predictions to one row per date
# ============================================================

keep_cols = [
    "model",
    "network_date",
    "regime",
    "train_size",
    "actual",
    "prediction",
    "absolute_error",
    "squared_error"
]


keep_cols = [
    col
    for col in keep_cols
    if col in pred_df.columns
]


pred_small = (
    pred_df[
        keep_cols
    ]
    .copy()
)


# Actual should be the same across models.
actual_df = (
    pred_small[
        [
            "network_date",
            "actual"
        ]
    ]
    .drop_duplicates(
        subset="network_date"
    )
)


# ------------------------------------------------------------
# Helper to extract one model
# ------------------------------------------------------------

def extract_model(
    model_name: str,
    prefix: str
) -> pd.DataFrame:

    temp = (
        pred_small[
            pred_small[
                "model"
            ]
            ==
            model_name
        ]
        .copy()
    )


    rename_dict = {
        "prediction":
            f"{prefix}_prediction",

        "absolute_error":
            f"{prefix}_absolute_error",

        "squared_error":
            f"{prefix}_squared_error",

        "train_size":
            f"{prefix}_train_size"
    }


    existing_rename = {
        key: value
        for key, value in rename_dict.items()
        if key in temp.columns
    }


    keep = [
        "network_date"
    ] + list(
        existing_rename.keys()
    )


    temp = (
        temp[
            keep
        ]
        .rename(
            columns=existing_rename
        )
    )


    return temp


baseline = extract_model(
    BASELINE_MODEL,
    "baseline"
)


main_network = extract_model(
    MAIN_NETWORK_MODEL,
    "same_ratio"
)


# ============================================================
# 10. Merge Baseline + Main Network
# ============================================================

alignment = pd.merge(
    actual_df,
    baseline,
    on="network_date",
    how="inner"
)


alignment = pd.merge(
    alignment,
    main_network,
    on="network_date",
    how="inner"
)


# ============================================================
# 11. Merge Secondary Edge Count Model if available
# ============================================================

if SECONDARY_NETWORK_MODEL in available_models:

    edge_network = extract_model(
        SECONDARY_NETWORK_MODEL,
        "edge_count"
    )


    alignment = pd.merge(
        alignment,
        edge_network,
        on="network_date",
        how="left"
    )


# ============================================================
# 12. Merge OOD Metrics
# ============================================================

shift_keep = [
    "network_date",
    "regime",
    "joint_max_abs_z",
    "n_range_breaks",
    "max_support_excursion_sd",
    "same_edge_ratio_expanding_z",
    "edge_count_expanding_z",
    "same_cross_strength_ratio_expanding_z"
]


shift_keep = [
    col
    for col in shift_keep
    if col in shift_df.columns
]


alignment = pd.merge(
    alignment,
    shift_df[
        shift_keep
    ],
    on="network_date",
    how="left"
)


alignment = (
    alignment
    .sort_values(
        "network_date"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 13. Absolute Feature-specific OOD
# ============================================================

alignment[
    "same_ratio_abs_z"
] = (
    alignment[
        "same_edge_ratio_expanding_z"
    ]
    .abs()
)


alignment[
    "edge_count_abs_z"
] = (
    alignment[
        "edge_count_expanding_z"
    ]
    .abs()
)


# ============================================================
# 14. Forecast failure measures:
# Main Same Ratio Model
# ============================================================

alignment[
    "same_ratio_excess_absolute_error"
] = (
    alignment[
        "same_ratio_absolute_error"
    ]
    -
    alignment[
        "baseline_absolute_error"
    ]
)


alignment[
    "same_ratio_excess_squared_error"
] = (
    alignment[
        "same_ratio_squared_error"
    ]
    -
    alignment[
        "baseline_squared_error"
    ]
)


alignment[
    "same_ratio_network_win"
] = (
    alignment[
        "same_ratio_absolute_error"
    ]
    <
    alignment[
        "baseline_absolute_error"
    ]
).astype(int)


alignment[
    "same_ratio_error_ratio"
] = (
    alignment[
        "same_ratio_absolute_error"
    ]
    /
    (
        alignment[
            "baseline_absolute_error"
        ]
        +
        EPS
    )
)


# ============================================================
# 15. Forecast failure measures:
# Edge Count Model
# ============================================================

if (
    "edge_count_absolute_error"
    in alignment.columns
):

    alignment[
        "edge_count_excess_absolute_error"
    ] = (
        alignment[
            "edge_count_absolute_error"
        ]
        -
        alignment[
            "baseline_absolute_error"
        ]
    )


    alignment[
        "edge_count_excess_squared_error"
    ] = (
        alignment[
            "edge_count_squared_error"
        ]
        -
        alignment[
            "baseline_squared_error"
        ]
    )


    alignment[
        "edge_count_network_win"
    ] = (
        alignment[
            "edge_count_absolute_error"
        ]
        <
        alignment[
            "baseline_absolute_error"
        ]
    ).astype(int)


# ============================================================
# 16. OOD Groups based purely on Joint |Z|
# ============================================================

def classify_ood(
    value
) -> str:

    if pd.isna(
        value
    ):

        return "Unscored"


    if value < MODERATE_THRESHOLD:

        return "Low OOD"


    if value < EXTREME_THRESHOLD:

        return "Moderate OOD"


    return "Extreme OOD"


alignment[
    "ood_group"
] = (
    alignment[
        "joint_max_abs_z"
    ]
    .apply(
        classify_ood
    )
)


# ============================================================
# 17. Historical Support Break Group
# ============================================================

alignment[
    "support_group"
] = np.where(
    alignment[
        "n_range_breaks"
    ]
    >
    0,
    "Range Break",
    "Inside Historical Range"
)


# ============================================================
# 18. Save Alignment Dataset
# ============================================================

alignment.to_csv(
    ALIGNMENT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. Spearman Correlation Helper
# ============================================================

def spearman_pair(
    data: pd.DataFrame,
    x: str,
    y: str
) -> tuple[int, float]:

    temp = (
        data[
            [
                x,
                y
            ]
        ]
        .dropna()
    )


    n = len(
        temp
    )


    if n < 5:

        return (
            n,
            np.nan
        )


    rho = (
        temp[
            x
        ]
        .corr(
            temp[
                y
            ],
            method="spearman"
        )
    )


    return (
        n,
        rho
    )


# ============================================================
# 20. Task 1 & 2:
# OOD -> Forecast Failure Correlations
# ============================================================

correlation_specs = [
    (
        "joint_max_abs_z",
        "same_ratio_absolute_error",
        "Joint OOD vs SameRatio Model AE"
    ),

    (
        "joint_max_abs_z",
        "same_ratio_excess_absolute_error",
        "Joint OOD vs SameRatio Excess AE"
    ),

    (
        "joint_max_abs_z",
        "same_ratio_excess_squared_error",
        "Joint OOD vs SameRatio Excess SE"
    ),

    (
        "same_ratio_abs_z",
        "same_ratio_absolute_error",
        "SameRatio OOD vs SameRatio Model AE"
    ),

    (
        "same_ratio_abs_z",
        "same_ratio_excess_squared_error",
        "SameRatio OOD vs SameRatio Excess SE"
    ),

    (
        "max_support_excursion_sd",
        "same_ratio_excess_squared_error",
        "Support Excursion vs SameRatio Excess SE"
    )
]


if (
    "edge_count_excess_squared_error"
    in alignment.columns
):

    correlation_specs.extend(
        [
            (
                "joint_max_abs_z",
                "edge_count_absolute_error",
                "Joint OOD vs EdgeCount Model AE"
            ),

            (
                "joint_max_abs_z",
                "edge_count_excess_squared_error",
                "Joint OOD vs EdgeCount Excess SE"
            ),

            (
                "edge_count_abs_z",
                "edge_count_excess_squared_error",
                "EdgeCount OOD vs EdgeCount Excess SE"
            )
        ]
    )


correlation_rows = []


for x, y, description in correlation_specs:

    if (
        x not in alignment.columns
        or
        y not in alignment.columns
    ):

        continue


    n, rho = spearman_pair(
        alignment,
        x,
        y
    )


    correlation_rows.append(
        {
            "description":
                description,

            "x_variable":
                x,

            "y_variable":
                y,

            "n":
                n,

            "spearman_rho":
                rho
        }
    )


correlation_df = pd.DataFrame(
    correlation_rows
)


correlation_df.to_csv(
    CORRELATION_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 21. Task 3:
# OOD Group Summary
# ============================================================

group_rows = []


group_order = [
    "Low OOD",
    "Moderate OOD",
    "Extreme OOD"
]


for group_name in group_order:

    group = (
        alignment[
            alignment[
                "ood_group"
            ]
            ==
            group_name
        ]
    )


    if group.empty:

        continue


    row = {
        "group_type":
            "Joint OOD Group",

        "group":
            group_name,

        "n":
            len(
                group
            ),

        "mean_joint_abs_z":
            group[
                "joint_max_abs_z"
            ]
            .mean(),

        "mean_baseline_AE":
            group[
                "baseline_absolute_error"
            ]
            .mean(),

        "mean_same_ratio_AE":
            group[
                "same_ratio_absolute_error"
            ]
            .mean(),

        "median_same_ratio_AE":
            group[
                "same_ratio_absolute_error"
            ]
            .median(),

        "mean_same_ratio_excess_AE":
            group[
                "same_ratio_excess_absolute_error"
            ]
            .mean(),

        "mean_same_ratio_excess_SE":
            group[
                "same_ratio_excess_squared_error"
            ]
            .mean(),

        "same_ratio_network_win_rate":
            group[
                "same_ratio_network_win"
            ]
            .mean()
    }


    if (
        "edge_count_absolute_error"
        in alignment.columns
    ):

        row[
            "mean_edge_count_AE"
        ] = (
            group[
                "edge_count_absolute_error"
            ]
            .mean()
        )


        row[
            "mean_edge_count_excess_SE"
        ] = (
            group[
                "edge_count_excess_squared_error"
            ]
            .mean()
        )


        row[
            "edge_count_network_win_rate"
        ] = (
            group[
                "edge_count_network_win"
            ]
            .mean()
        )


    group_rows.append(
        row
    )


# ============================================================
# 22. Task 4:
# Historical Range Break Summary
# ============================================================

for group_name in [
    "Inside Historical Range",
    "Range Break"
]:

    group = (
        alignment[
            alignment[
                "support_group"
            ]
            ==
            group_name
        ]
    )


    if group.empty:

        continue


    row = {
        "group_type":
            "Historical Support",

        "group":
            group_name,

        "n":
            len(
                group
            ),

        "mean_joint_abs_z":
            group[
                "joint_max_abs_z"
            ]
            .mean(),

        "mean_baseline_AE":
            group[
                "baseline_absolute_error"
            ]
            .mean(),

        "mean_same_ratio_AE":
            group[
                "same_ratio_absolute_error"
            ]
            .mean(),

        "median_same_ratio_AE":
            group[
                "same_ratio_absolute_error"
            ]
            .median(),

        "mean_same_ratio_excess_AE":
            group[
                "same_ratio_excess_absolute_error"
            ]
            .mean(),

        "mean_same_ratio_excess_SE":
            group[
                "same_ratio_excess_squared_error"
            ]
            .mean(),

        "same_ratio_network_win_rate":
            group[
                "same_ratio_network_win"
            ]
            .mean()
    }


    if (
        "edge_count_absolute_error"
        in alignment.columns
    ):

        row[
            "mean_edge_count_AE"
        ] = (
            group[
                "edge_count_absolute_error"
            ]
            .mean()
        )


        row[
            "mean_edge_count_excess_SE"
        ] = (
            group[
                "edge_count_excess_squared_error"
            ]
            .mean()
        )


        row[
            "edge_count_network_win_rate"
        ] = (
            group[
                "edge_count_network_win"
            ]
            .mean()
        )


    group_rows.append(
        row
    )


group_summary_df = pd.DataFrame(
    group_rows
)


group_summary_df.to_csv(
    GROUP_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 23. Task 5:
# Top OOD / Top Forecast Failure Events
# ============================================================

event_cols = [
    "network_date",
    "regime",
    "joint_max_abs_z",
    "same_ratio_abs_z",
    "edge_count_abs_z",
    "n_range_breaks",
    "max_support_excursion_sd",

    "actual",

    "baseline_prediction",
    "baseline_absolute_error",

    "same_ratio_prediction",
    "same_ratio_absolute_error",
    "same_ratio_excess_absolute_error",
    "same_ratio_excess_squared_error",
    "same_ratio_network_win"
]


event_cols = [
    col
    for col in event_cols
    if col in alignment.columns
]


# ------------------------------------------------------------
# Rank by OOD
# ------------------------------------------------------------

ood_events = (
    alignment[
        event_cols
    ]
    .copy()
)


ood_events[
    "event_selection"
] = "Top OOD"


ood_events = (
    ood_events
    .sort_values(
        "joint_max_abs_z",
        ascending=False
    )
    .head(
        10
    )
)


# ------------------------------------------------------------
# Rank by network excess squared error
# ------------------------------------------------------------

failure_events = (
    alignment[
        event_cols
    ]
    .copy()
)


failure_events[
    "event_selection"
] = "Top Forecast Failure"


failure_events = (
    failure_events
    .sort_values(
        "same_ratio_excess_squared_error",
        ascending=False
    )
    .head(
        10
    )
)


event_df = pd.concat(
    [
        ood_events,
        failure_events
    ],
    ignore_index=True
)


event_df = (
    event_df
    .drop_duplicates(
        subset=[
            "network_date",
            "event_selection"
        ]
    )
)


event_df.to_csv(
    EVENT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 24. Sensitivity Diagnosis
#
# Descriptive only:
#
# Full sample
# Exclude highest OOD date
# Exclude top 2 OOD dates
# Exclude R4
#
# This is NOT a formal robustness proof.
# ============================================================

def summarize_subset(
    data: pd.DataFrame,
    label: str
) -> dict:

    temp = (
        data
        .dropna(
            subset=[
                "joint_max_abs_z",
                "same_ratio_absolute_error",
                "same_ratio_excess_squared_error"
            ]
        )
    )


    if len(
        temp
    ) < 5:

        return {
            "subset":
                label,

            "n":
                len(
                    temp
                ),

            "rho_ood_vs_network_AE":
                np.nan,

            "rho_ood_vs_excess_SE":
                np.nan,

            "mean_network_AE":
                np.nan,

            "mean_excess_SE":
                np.nan,

            "network_win_rate":
                np.nan
        }


    return {
        "subset":
            label,

        "n":
            len(
                temp
            ),

        "rho_ood_vs_network_AE":
            temp[
                "joint_max_abs_z"
            ]
            .corr(
                temp[
                    "same_ratio_absolute_error"
                ],
                method="spearman"
            ),

        "rho_ood_vs_excess_SE":
            temp[
                "joint_max_abs_z"
            ]
            .corr(
                temp[
                    "same_ratio_excess_squared_error"
                ],
                method="spearman"
            ),

        "mean_network_AE":
            temp[
                "same_ratio_absolute_error"
            ]
            .mean(),

        "mean_excess_SE":
            temp[
                "same_ratio_excess_squared_error"
            ]
            .mean(),

        "network_win_rate":
            temp[
                "same_ratio_network_win"
            ]
            .mean()
    }


sensitivity_rows = []


# Full sample
sensitivity_rows.append(
    summarize_subset(
        alignment,
        "Full OOS Sample"
    )
)


# ------------------------------------------------------------
# Exclude highest OOD
# ------------------------------------------------------------

valid_ood = (
    alignment[
        alignment[
            "joint_max_abs_z"
        ]
        .notna()
    ]
    .sort_values(
        "joint_max_abs_z",
        ascending=False
    )
)


if len(
    valid_ood
) >= 1:

    top1_date = (
        valid_ood[
            "network_date"
        ]
        .iloc[
            0
        ]
    )


    subset = (
        alignment[
            alignment[
                "network_date"
            ]
            !=
            top1_date
        ]
    )


    sensitivity_rows.append(
        summarize_subset(
            subset,
            "Exclude Top 1 OOD Date"
        )
    )


# ------------------------------------------------------------
# Exclude top 2 OOD
# ------------------------------------------------------------

if len(
    valid_ood
) >= 2:

    top2_dates = set(
        valid_ood[
            "network_date"
        ]
        .iloc[
            :2
        ]
    )


    subset = (
        alignment[
            ~alignment[
                "network_date"
            ]
            .isin(
                top2_dates
            )
        ]
    )


    sensitivity_rows.append(
        summarize_subset(
            subset,
            "Exclude Top 2 OOD Dates"
        )
    )


# ------------------------------------------------------------
# Exclude R4
# ------------------------------------------------------------

if "regime" in alignment.columns:

    subset = (
        alignment[
            alignment[
                "regime"
            ]
            .astype(str)
            !=
            "R4"
        ]
    )


    sensitivity_rows.append(
        summarize_subset(
            subset,
            "Exclude R4"
        )
    )


sensitivity_df = pd.DataFrame(
    sensitivity_rows
)


sensitivity_df.to_csv(
    SENSITIVITY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 25. Console Output
# ============================================================

print(
    "\n=============================================="
)

print(
    "Stage 12 - Stage 3"
)

print(
    "OOD Severity vs Forecast Failure"
)

print(
    "=============================================="
)


print(
    "\nAligned OOS Forecast Dates:",
    len(
        alignment
    )
)


print(
    "\nDate Range:"
)


print(
    alignment[
        "network_date"
    ]
    .min(),
    "->",
    alignment[
        "network_date"
    ]
    .max()
)


print(
    "\n--- OOD / Forecast Error Correlations ---"
)


print(
    correlation_df.to_string(
        index=False
    )
)


print(
    "\n--- OOD Group Summary ---"
)


print(
    group_summary_df.to_string(
        index=False
    )
)


print(
    "\n--- Top OOD Dates ---"
)


top_display_cols = [
    "network_date",
    "regime",
    "joint_max_abs_z",
    "same_ratio_abs_z",
    "n_range_breaks",
    "same_ratio_absolute_error",
    "same_ratio_excess_squared_error",
    "same_ratio_network_win"
]


top_display_cols = [
    col
    for col in top_display_cols
    if col in alignment.columns
]


print(
    alignment[
        top_display_cols
    ]
    .sort_values(
        "joint_max_abs_z",
        ascending=False
    )
    .head(
        10
    )
    .to_string(
        index=False
    )
)


print(
    "\n--- Sensitivity Diagnosis ---"
)


print(
    sensitivity_df.to_string(
        index=False
    )
)


# ============================================================
# 26. Figure 1:
# Joint OOD vs SameRatio Network Excess SE
# ============================================================

plot_df = (
    alignment[
        [
            "joint_max_abs_z",
            "same_ratio_excess_squared_error",
            "regime"
        ]
    ]
    .dropna()
)


fig, ax = plt.subplots(
    figsize=(
        8,
        6
    )
)


ax.scatter(
    plot_df[
        "joint_max_abs_z"
    ],
    plot_df[
        "same_ratio_excess_squared_error"
    ]
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.axvline(
    x=MODERATE_THRESHOLD,
    linestyle="--",
    linewidth=1
)


ax.axvline(
    x=EXTREME_THRESHOLD,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Joint Network OOD Score"
)


ax.set_ylabel(
    "Excess Squared Error: Network - Baseline"
)


ax.set_title(
    "Network Distribution Shift vs Forecast Deterioration"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "ood_vs_same_ratio_excess_squared_error.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 27. Figure 2:
# Same Ratio OOD vs Same Ratio Forecast Error
# ============================================================

plot_df = (
    alignment[
        [
            "same_ratio_abs_z",
            "same_ratio_absolute_error"
        ]
    ]
    .dropna()
)


fig, ax = plt.subplots(
    figsize=(
        8,
        6
    )
)


ax.scatter(
    plot_df[
        "same_ratio_abs_z"
    ],
    plot_df[
        "same_ratio_absolute_error"
    ]
)


ax.set_xlabel(
    "|Historical Z| of Same Edge Ratio"
)


ax.set_ylabel(
    "Same Ratio Model Absolute Forecast Error"
)


ax.set_title(
    "Same Edge Ratio OOD Severity vs Forecast Error"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "same_ratio_ood_vs_forecast_error.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 28. Figure 3:
# Time Alignment:
# OOD Score and Excess Forecast Error
#
# Use two separate figures rather than dual y-axis
# to avoid visual distortion.
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    alignment[
        "network_date"
    ],
    alignment[
        "joint_max_abs_z"
    ],
    marker="o"
)


ax.axhline(
    y=MODERATE_THRESHOLD,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=EXTREME_THRESHOLD,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Forecast Network Date"
)


ax.set_ylabel(
    "Joint Network OOD Score"
)


ax.set_title(
    "Network OOD Severity Across OOS Forecast Dates"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "stage12_ood_score_over_forecast_dates.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    alignment[
        "network_date"
    ],
    alignment[
        "same_ratio_excess_squared_error"
    ],
    marker="o"
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Forecast Network Date"
)


ax.set_ylabel(
    "Excess Squared Error: Network - Baseline"
)


ax.set_title(
    "Network-model Forecast Deterioration Across Time"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "excess_forecast_error_over_time.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 29. Figure 4:
# Mean Excess SE by OOD Group
# ============================================================

ood_group_plot = (
    group_summary_df[
        group_summary_df[
            "group_type"
        ]
        ==
        "Joint OOD Group"
    ]
    .copy()
)


ood_group_plot[
    "group"
] = pd.Categorical(
    ood_group_plot[
        "group"
    ],
    categories=[
        "Low OOD",
        "Moderate OOD",
        "Extreme OOD"
    ],
    ordered=True
)


ood_group_plot = (
    ood_group_plot
    .sort_values(
        "group"
    )
)


fig, ax = plt.subplots(
    figsize=(
        8,
        6
    )
)


ax.bar(
    ood_group_plot[
        "group"
    ]
    .astype(str),
    ood_group_plot[
        "mean_same_ratio_excess_SE"
    ]
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Network OOD Group"
)


ax.set_ylabel(
    "Mean Excess Squared Error"
)


ax.set_title(
    "Forecast Deterioration Across Network OOD Levels"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "forecast_deterioration_by_ood_group.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 30. Complete
# ============================================================

print(
    "\n=============================================="
)

print(
    "Stage 3 Complete"
)

print(
    "=============================================="
)


for path in [
    ALIGNMENT_FILE,
    CORRELATION_FILE,
    GROUP_SUMMARY_FILE,
    EVENT_FILE,
    SENSITIVITY_FILE
]:

    print(
        path
    )