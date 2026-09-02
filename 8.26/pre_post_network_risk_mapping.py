from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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

FUTURE_FILE = (
    PROCESSED_DIR
    / "network_future_market_dataset.csv"
)

CURRENT_MARKET_FILE = (
    PROCESSED_DIR
    / "external_market_state_window_metrics.csv"
)

SHIFT_FILE = (
    PROCESSED_DIR
    / "network_distribution_shift.csv"
)


# ============================================================
# 3. Output Files
# ============================================================

MAPPING_DATA_FILE = (
    PROCESSED_DIR
    / "network_risk_mapping_dataset.csv"
)

PRE_POST_SUMMARY_FILE = (
    PROCESSED_DIR
    / "pre_post_network_risk_mapping_summary.csv"
)

INTERACTION_FILE = (
    PROCESSED_DIR
    / "network_risk_mapping_interaction_summary.csv"
)

LOO_FILE = (
    PROCESSED_DIR
    / "network_risk_mapping_post_leave_one_out.csv"
)


# ============================================================
# 4. Main Target
# ============================================================

TARGET = (
    "future_relative_volatility_20"
)


# ============================================================
# 5. Candidate Network Features
#
# Analyze separately.
# Do NOT put all of them into one regression.
# ============================================================

FEATURE_CANDIDATES = {
    "same_edge_ratio": [
        "same_edge_ratio",
        "matched_same_ratio"
    ],

    "edge_count": [
        "edge_count",
        "matched_edge_count"
    ],

    "same_cross_strength_ratio": [
        "same_cross_strength_ratio",
        "raw_same_cross_strength_ratio"
    ]
}


# ============================================================
# 6. Load Future Outcome Dataset
# ============================================================

future_df = pd.read_csv(
    FUTURE_FILE
)


future_df[
    "network_date"
] = pd.to_datetime(
    future_df[
        "network_date"
    ]
)


future_df[
    "regime"
] = (
    future_df[
        "regime"
    ]
    .astype(str)
    .str.strip()
)


# ============================================================
# 7. Resolve Network Feature Columns
# ============================================================

resolved_features = {}


for standard_name, candidates in FEATURE_CANDIDATES.items():

    matched = next(
        (
            col
            for col in candidates
            if col in future_df.columns
        ),
        None
    )


    if matched is not None:

        resolved_features[
            standard_name
        ] = matched


missing_features = [
    feature
    for feature in FEATURE_CANDIDATES
    if feature not in resolved_features
]


if missing_features:

    raise ValueError(
        "Future dataset lacks features: "
        f"{missing_features}"
    )


# ============================================================
# 8. Standardize Names
# ============================================================

mapping_df = future_df[
    [
        "network_date",
        "regime",
        TARGET
    ]
].copy()


for standard_name, original_name in resolved_features.items():

    mapping_df[
        standard_name
    ] = pd.to_numeric(
        future_df[
            original_name
        ],
        errors="coerce"
    )


mapping_df[
    TARGET
] = pd.to_numeric(
    mapping_df[
        TARGET
    ],
    errors="coerce"
)


# ============================================================
# 9. Load Current Market State
# ============================================================

current_df = pd.read_csv(
    CURRENT_MARKET_FILE
)


current_df[
    "network_date"
] = pd.to_datetime(
    current_df[
        "network_date"
    ]
)


current_df[
    "annualized_volatility_20"
] = pd.to_numeric(
    current_df[
        "annualized_volatility_20"
    ],
    errors="coerce"
)


current_vol = (
    current_df[
        current_df[
            "index_key"
        ]
        .isin(
            [
                "CSI300",
                "CSI500"
            ]
        )
    ]
    .pivot(
        index="network_date",
        columns="index_key",
        values="annualized_volatility_20"
    )
    .reset_index()
)


if (
    "CSI300" not in current_vol.columns
    or
    "CSI500" not in current_vol.columns
):

    raise ValueError(
        "CSI300 or CSI500 current volatility missing."
    )


current_vol[
    "current_relative_volatility_20"
] = (
    current_vol[
        "CSI500"
    ]
    -
    current_vol[
        "CSI300"
    ]
)


current_vol = current_vol[
    [
        "network_date",
        "current_relative_volatility_20"
    ]
]


# ============================================================
# 10. Merge Current Market Risk
# ============================================================

mapping_df = pd.merge(
    mapping_df,
    current_vol,
    on="network_date",
    how="left"
)


# ============================================================
# 11. Merge Stage 1 OOD Metrics
# ============================================================

if SHIFT_FILE.exists():

    shift_df = pd.read_csv(
        SHIFT_FILE
    )


    shift_df[
        "network_date"
    ] = pd.to_datetime(
        shift_df[
            "network_date"
        ]
    )


    optional_shift_cols = [
        "network_date",
        "joint_max_abs_z",
        "n_range_breaks",
        "max_support_excursion_sd",
        "joint_moderate_ood",
        "joint_extreme_ood"
    ]


    optional_shift_cols = [
        col
        for col in optional_shift_cols
        if col in shift_df.columns
    ]


    mapping_df = pd.merge(
        mapping_df,
        shift_df[
            optional_shift_cols
        ],
        on="network_date",
        how="left"
    )


# ============================================================
# 12. Identify R4 Break Date
# ============================================================

regime_values = set(
    mapping_df[
        "regime"
    ]
)


def find_regime(
    candidates
):

    for candidate in candidates:

        if candidate in regime_values:

            return candidate

    return None


r4_name = find_regime(
    [
        "R4",
        "4",
        "Regime4",
        "Regime 4"
    ]
)


if r4_name is None:

    raise ValueError(
        "Cannot identify R4."
    )


first_r4_date = (
    mapping_df.loc[
        mapping_df[
            "regime"
        ]
        ==
        r4_name,
        "network_date"
    ]
    .min()
)


# ============================================================
# 13. Define Pre/Post
#
# Break point is FIXED from previous Regime analysis.
# It is NOT selected using Future Risk.
# ============================================================

mapping_df[
    "period"
] = np.where(
    mapping_df[
        "network_date"
    ]
    <
    first_r4_date,
    "Pre-R4",
    "Post-R4"
)


mapping_df[
    "post_indicator"
] = (
    mapping_df[
        "period"
    ]
    ==
    "Post-R4"
).astype(int)


# ============================================================
# 14. Convert Future Relative Volatility to percentage points
#
# Example:
# 0.05 -> 5 percentage points
# ============================================================

mapping_df[
    "future_relative_volatility_pp"
] = (
    100.0
    *
    mapping_df[
        TARGET
    ]
)


mapping_df[
    "current_relative_volatility_pp"
] = (
    100.0
    *
    mapping_df[
        "current_relative_volatility_20"
    ]
)


# ============================================================
# 15. Pre-period Standardization
#
# Important:
# Use only Pre-R4 center and scale.
# Post observations are standardized relative
# to the historical Pre-R4 distribution.
# ============================================================

pre_mask = (
    mapping_df[
        "period"
    ]
    ==
    "Pre-R4"
)


for feature in resolved_features:

    pre_values = (
        mapping_df.loc[
            pre_mask,
            feature
        ]
        .dropna()
    )


    pre_mean = (
        pre_values.mean()
    )


    pre_sd = (
        pre_values.std(
            ddof=1
        )
    )


    if (
        pd.isna(
            pre_sd
        )
        or
        pre_sd
        <=
        1e-12
    ):

        raise ValueError(
            f"Invalid pre-period SD for {feature}"
        )


    mapping_df[
        f"{feature}_pre_z"
    ] = (
        mapping_df[
            feature
        ]
        -
        pre_mean
    ) / pre_sd


# Standardize current relative volatility using Pre-R4
current_pre = (
    mapping_df.loc[
        pre_mask,
        "current_relative_volatility_pp"
    ]
    .dropna()
)


current_pre_mean = (
    current_pre.mean()
)


current_pre_sd = (
    current_pre.std(
        ddof=1
    )
)


mapping_df[
    "current_relative_volatility_pre_z"
] = (
    mapping_df[
        "current_relative_volatility_pp"
    ]
    -
    current_pre_mean
) / current_pre_sd


# ============================================================
# 16. Clean Dataset
# ============================================================

required_cols = [
    "network_date",
    "regime",
    "period",
    "post_indicator",
    "future_relative_volatility_pp",
    "current_relative_volatility_pre_z"
]


for feature in resolved_features:

    required_cols.extend(
        [
            feature,
            f"{feature}_pre_z"
        ]
    )


mapping_df = (
    mapping_df
    .dropna(
        subset=[
            "future_relative_volatility_pp",
            "current_relative_volatility_pre_z"
        ]
    )
    .sort_values(
        "network_date"
    )
    .reset_index(
        drop=True
    )
)


mapping_df.to_csv(
    MAPPING_DATA_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. Helper:
# Simple Mapping Statistics
# ============================================================

def simple_mapping_stats(
    data: pd.DataFrame,
    x_col: str,
    y_col: str
) -> dict:

    temp = (
        data[
            [
                x_col,
                y_col
            ]
        ]
        .dropna()
    )


    n = len(
        temp
    )


    if n < 3:

        return {
            "n":
                n,

            "spearman":
                np.nan,

            "intercept":
                np.nan,

            "slope":
                np.nan,

            "r2":
                np.nan
        }


    spearman = (
        temp[
            x_col
        ]
        .corr(
            temp[
                y_col
            ],
            method="spearman"
        )
    )


    X = (
        temp[
            [
                x_col
            ]
        ]
        .to_numpy()
    )


    y = (
        temp[
            y_col
        ]
        .to_numpy()
    )


    model = LinearRegression()


    model.fit(
        X,
        y
    )


    prediction = (
        model.predict(
            X
        )
    )


    return {
        "n":
            n,

        "spearman":
            spearman,

        "intercept":
            float(
                model.intercept_
            ),

        "slope":
            float(
                model.coef_[0]
            ),

        "r2":
            float(
                r2_score(
                    y,
                    prediction
                )
            )
    }


# ============================================================
# 18. Task 1:
# Pre/Post Separate Mapping Summary
# ============================================================

summary_rows = []


for feature in resolved_features:

    z_col = (
        f"{feature}_pre_z"
    )


    for period in [
        "Pre-R4",
        "Post-R4"
    ]:

        group = (
            mapping_df[
                mapping_df[
                    "period"
                ]
                ==
                period
            ]
        )


        stats = simple_mapping_stats(
            data=group,
            x_col=z_col,
            y_col="future_relative_volatility_pp"
        )


        summary_rows.append(
            {
                "network_feature":
                    feature,

                "period":
                    period,

                "n":
                    stats[
                        "n"
                    ],

                "mean_feature_raw":
                    group[
                        feature
                    ]
                    .mean(),

                "sd_feature_raw":
                    group[
                        feature
                    ]
                    .std(
                        ddof=1
                    ),

                "min_feature_raw":
                    group[
                        feature
                    ]
                    .min(),

                "max_feature_raw":
                    group[
                        feature
                    ]
                    .max(),

                "mean_future_relative_volatility_pp":
                    group[
                        "future_relative_volatility_pp"
                    ]
                    .mean(),

                "median_future_relative_volatility_pp":
                    group[
                        "future_relative_volatility_pp"
                    ]
                    .median(),

                "spearman_rho":
                    stats[
                        "spearman"
                    ],

                # slope interpretation:
                # change in future relative volatility
                # percentage points per one Pre-R4 SD of X
                "slope_pp_per_pre_sd":
                    stats[
                        "slope"
                    ],

                "intercept_pp":
                    stats[
                        "intercept"
                    ],

                "r2":
                    stats[
                        "r2"
                    ]
            }
        )


pre_post_summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# 19. Add Pre vs Post Difference
# ============================================================

difference_rows = []


for feature in resolved_features:

    temp = (
        pre_post_summary[
            pre_post_summary[
                "network_feature"
            ]
            ==
            feature
        ]
        .set_index(
            "period"
        )
    )


    if (
        "Pre-R4" in temp.index
        and
        "Post-R4" in temp.index
    ):

        difference_rows.append(
            {
                "network_feature":
                    feature,

                "pre_spearman":
                    temp.loc[
                        "Pre-R4",
                        "spearman_rho"
                    ],

                "post_spearman":
                    temp.loc[
                        "Post-R4",
                        "spearman_rho"
                    ],

                "delta_spearman_post_minus_pre":
                    (
                        temp.loc[
                            "Post-R4",
                            "spearman_rho"
                        ]
                        -
                        temp.loc[
                            "Pre-R4",
                            "spearman_rho"
                        ]
                    ),

                "pre_slope_pp_per_pre_sd":
                    temp.loc[
                        "Pre-R4",
                        "slope_pp_per_pre_sd"
                    ],

                "post_slope_pp_per_pre_sd":
                    temp.loc[
                        "Post-R4",
                        "slope_pp_per_pre_sd"
                    ],

                "delta_slope_post_minus_pre":
                    (
                        temp.loc[
                            "Post-R4",
                            "slope_pp_per_pre_sd"
                        ]
                        -
                        temp.loc[
                            "Pre-R4",
                            "slope_pp_per_pre_sd"
                        ]
                    )
            }
        )


difference_df = pd.DataFrame(
    difference_rows
)


# Save both sections in one long CSV
pre_post_summary = pd.merge(
    pre_post_summary,
    difference_df,
    on="network_feature",
    how="left"
)


pre_post_summary.to_csv(
    PRE_POST_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. Task 2 + Task 3:
#
# Interaction Models
#
# Model A:
#
# Y =
# alpha
# + beta_x X
# + beta_post Post
# + beta_interaction X*Post
#
#
# Model B:
#
# Y =
# alpha
# + gamma CurrentRelativeVol
# + beta_x X
# + beta_post Post
# + beta_interaction X*Post
#
#
# No naive significance claims are made.
# ============================================================

interaction_rows = []


for feature in resolved_features:

    z_col = (
        f"{feature}_pre_z"
    )


    temp = (
        mapping_df[
            [
                "future_relative_volatility_pp",
                "current_relative_volatility_pre_z",
                z_col,
                "post_indicator"
            ]
        ]
        .dropna()
        .copy()
    )


    temp[
        "interaction"
    ] = (
        temp[
            z_col
        ]
        *
        temp[
            "post_indicator"
        ]
    )


    y = (
        temp[
            "future_relative_volatility_pp"
        ]
        .to_numpy()
    )


    # --------------------------------------------------------
    # Model A:
    # Unadjusted interaction model
    # --------------------------------------------------------

    X_A = (
        temp[
            [
                z_col,
                "post_indicator",
                "interaction"
            ]
        ]
        .to_numpy()
    )


    model_A = LinearRegression()


    model_A.fit(
        X_A,
        y
    )


    pred_A = (
        model_A.predict(
            X_A
        )
    )


    beta_x_A = float(
        model_A.coef_[0]
    )


    beta_post_A = float(
        model_A.coef_[1]
    )


    beta_interaction_A = float(
        model_A.coef_[2]
    )


    pre_slope_A = (
        beta_x_A
    )


    post_slope_A = (
        beta_x_A
        +
        beta_interaction_A
    )


    interaction_rows.append(
        {
            "network_feature":
                feature,

            "model":
                "UnadjustedInteraction",

            "n":
                len(
                    temp
                ),

            "intercept":
                float(
                    model_A.intercept_
                ),

            "coef_current_relative_volatility":
                np.nan,

            "coef_network_feature_pre":
                beta_x_A,

            "coef_post_level":
                beta_post_A,

            "coef_feature_x_post":
                beta_interaction_A,

            "pre_network_slope_pp_per_pre_sd":
                pre_slope_A,

            "post_network_slope_pp_per_pre_sd":
                post_slope_A,

            "delta_network_slope":
                (
                    post_slope_A
                    -
                    pre_slope_A
                ),

            "r2":
                float(
                    r2_score(
                        y,
                        pred_A
                    )
                )
        }
    )


    # --------------------------------------------------------
    # Model B:
    # Conditional interaction model
    # controlling current relative volatility
    # --------------------------------------------------------

    X_B = (
        temp[
            [
                "current_relative_volatility_pre_z",
                z_col,
                "post_indicator",
                "interaction"
            ]
        ]
        .to_numpy()
    )


    model_B = LinearRegression()


    model_B.fit(
        X_B,
        y
    )


    pred_B = (
        model_B.predict(
            X_B
        )
    )


    gamma_B = float(
        model_B.coef_[0]
    )


    beta_x_B = float(
        model_B.coef_[1]
    )


    beta_post_B = float(
        model_B.coef_[2]
    )


    beta_interaction_B = float(
        model_B.coef_[3]
    )


    pre_slope_B = (
        beta_x_B
    )


    post_slope_B = (
        beta_x_B
        +
        beta_interaction_B
    )


    interaction_rows.append(
        {
            "network_feature":
                feature,

            "model":
                "ConditionalInteraction",

            "n":
                len(
                    temp
                ),

            "intercept":
                float(
                    model_B.intercept_
                ),

            "coef_current_relative_volatility":
                gamma_B,

            "coef_network_feature_pre":
                beta_x_B,

            "coef_post_level":
                beta_post_B,

            "coef_feature_x_post":
                beta_interaction_B,

            "pre_network_slope_pp_per_pre_sd":
                pre_slope_B,

            "post_network_slope_pp_per_pre_sd":
                post_slope_B,

            "delta_network_slope":
                (
                    post_slope_B
                    -
                    pre_slope_B
                ),

            "r2":
                float(
                    r2_score(
                        y,
                        pred_B
                    )
                )
        }
    )


interaction_df = pd.DataFrame(
    interaction_rows
)


interaction_df.to_csv(
    INTERACTION_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 21. Task 4:
# Post-period Leave-One-Out Sensitivity
#
# Primary question:
# Is Post-R4 slope driven by one observation?
# ============================================================

loo_rows = []


for feature in resolved_features:

    z_col = (
        f"{feature}_pre_z"
    )


    pre_group = (
        mapping_df[
            mapping_df[
                "period"
            ]
            ==
            "Pre-R4"
        ]
    )


    pre_stats = simple_mapping_stats(
        pre_group,
        z_col,
        "future_relative_volatility_pp"
    )


    post_group = (
        mapping_df[
            mapping_df[
                "period"
            ]
            ==
            "Post-R4"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    full_post_stats = simple_mapping_stats(
        post_group,
        z_col,
        "future_relative_volatility_pp"
    )


    for leave_pos in range(
        len(
            post_group
        )
    ):

        left_out = (
            post_group
            .iloc[
                leave_pos
            ]
        )


        reduced = (
            post_group
            .drop(
                index=leave_pos
            )
            .reset_index(
                drop=True
            )
        )


        reduced_stats = simple_mapping_stats(
            reduced,
            z_col,
            "future_relative_volatility_pp"
        )


        loo_rows.append(
            {
                "network_feature":
                    feature,

                "left_out_date":
                    left_out[
                        "network_date"
                    ],

                "left_out_regime":
                    left_out[
                        "regime"
                    ],

                "n_post_after_deletion":
                    reduced_stats[
                        "n"
                    ],

                "pre_full_slope":
                    pre_stats[
                        "slope"
                    ],

                "post_full_slope":
                    full_post_stats[
                        "slope"
                    ],

                "post_loo_slope":
                    reduced_stats[
                        "slope"
                    ],

                "post_loo_spearman":
                    reduced_stats[
                        "spearman"
                    ],

                "delta_loo_slope_vs_pre":
                    (
                        reduced_stats[
                            "slope"
                        ]
                        -
                        pre_stats[
                            "slope"
                        ]
                    )
            }
        )


loo_df = pd.DataFrame(
    loo_rows
)


loo_df.to_csv(
    LOO_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 22. Console Output
# ============================================================

print(
    "\n=============================================="
)

print(
    "Stage 12 - Stage 2"
)

print(
    "Pre/Post Network-Risk Mapping"
)

print(
    "=============================================="
)


print(
    "\nFirst R4 Date:",
    first_r4_date
)


print(
    "\nSample Size:"
)


print(
    mapping_df[
        "period"
    ]
    .value_counts()
)


print(
    "\n--- Pre/Post Mapping Summary ---"
)


display_cols = [
    "network_feature",
    "period",
    "n",
    "spearman_rho",
    "slope_pp_per_pre_sd",
    "r2",
    "delta_spearman_post_minus_pre",
    "delta_slope_post_minus_pre"
]


print(
    pre_post_summary[
        display_cols
    ]
    .to_string(
        index=False
    )
)


print(
    "\n--- Interaction Models ---"
)


interaction_display_cols = [
    "network_feature",
    "model",
    "n",
    "coef_current_relative_volatility",
    "pre_network_slope_pp_per_pre_sd",
    "post_network_slope_pp_per_pre_sd",
    "delta_network_slope",
    "r2"
]


print(
    interaction_df[
        interaction_display_cols
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 23. Figure 1:
# Same Edge Ratio vs Future Relative Volatility
# Pre vs Post
# ============================================================

feature = "same_edge_ratio"


fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


for period in [
    "Pre-R4",
    "Post-R4"
]:

    group = (
        mapping_df[
            mapping_df[
                "period"
            ]
            ==
            period
        ]
        .dropna(
            subset=[
                feature,
                "future_relative_volatility_pp"
            ]
        )
    )


    ax.scatter(
        group[
            feature
        ],
        group[
            "future_relative_volatility_pp"
        ],
        label=period
    )


    if len(
        group
    ) >= 3:

        X = (
            group[
                [
                    feature
                ]
            ]
            .to_numpy()
        )


        y = (
            group[
                "future_relative_volatility_pp"
            ]
            .to_numpy()
        )


        model = LinearRegression()


        model.fit(
            X,
            y
        )


        x_grid = np.linspace(
            group[
                feature
            ]
            .min(),
            group[
                feature
            ]
            .max(),
            100
        )


        y_grid = model.predict(
            x_grid.reshape(
                -1,
                1
            )
        )


        ax.plot(
            x_grid,
            y_grid
        )


ax.set_xlabel(
    "Same-industry Edge Ratio"
)


ax.set_ylabel(
    "Future CSI500 - CSI300 Volatility (percentage points)"
)


ax.set_title(
    "Pre- vs Post-R4 Network-Risk Mapping"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "same_ratio_pre_post_mapping.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 24. Figure 2:
# Pre/Post Slopes Across Network Features
# ============================================================

slope_plot = (
    pre_post_summary[
        [
            "network_feature",
            "period",
            "slope_pp_per_pre_sd"
        ]
    ]
    .pivot(
        index="network_feature",
        columns="period",
        values="slope_pp_per_pre_sd"
    )
)


fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


slope_plot.plot(
    kind="bar",
    ax=ax
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Network Feature"
)


ax.set_ylabel(
    "Slope: Future Relative Volatility pp per Pre-R4 SD"
)


ax.set_title(
    "Change in Network-Risk Mapping Across R4"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "pre_post_network_risk_slopes.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 25. Figure 3:
# Post Leave-One-Out Slope for Same Edge Ratio
# ============================================================

loo_same = (
    loo_df[
        loo_df[
            "network_feature"
        ]
        ==
        "same_edge_ratio"
    ]
    .copy()
)


fig, ax = plt.subplots(
    figsize=(
        11,
        6
    )
)


ax.plot(
    loo_same[
        "left_out_date"
    ],
    loo_same[
        "post_loo_slope"
    ],
    marker="o"
)


if not loo_same.empty:

    pre_slope = (
        loo_same[
            "pre_full_slope"
        ]
        .iloc[
            0
        ]
    )


    full_post_slope = (
        loo_same[
            "post_full_slope"
        ]
        .iloc[
            0
        ]
    )


    ax.axhline(
        y=pre_slope,
        linestyle="--",
        linewidth=1,
        label="Pre-R4 Full Slope"
    )


    ax.axhline(
        y=full_post_slope,
        linestyle=":",
        linewidth=1,
        label="Post-R4 Full Slope"
    )


ax.set_xlabel(
    "Left-out Post-R4 Network Date"
)


ax.set_ylabel(
    "Post-R4 Same Ratio Slope"
)


ax.set_title(
    "Post-R4 Leave-One-Out Stability"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "post_same_ratio_leave_one_out.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 26. Complete
# ============================================================

print(
    "\n=============================================="
)

print(
    "Stage 2 Complete"
)

print(
    "=============================================="
)


for path in [
    MAPPING_DATA_FILE,
    PRE_POST_SUMMARY_FILE,
    INTERACTION_FILE,
    LOO_FILE
]:

    print(
        path
    )