from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 0. Paths
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
OUT = ROOT / "output"

DAY5 = OUT / "M1_day5"
DAY6 = OUT / "M1_day6"
DAY7 = OUT / "M1_day7"
DAY8 = OUT / "M1_day8"

OUTDIR = (
    OUT
    / "M1_day12"
    / "01_stage1_frozen_alpha_evaluation_panel"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Canonical frozen sources
# ------------------------------------------------------------

FACTOR_SOURCE = (
    DAY5
    / "05_step5_community_features"
    / "community_bridge_feature_panel.parquet"
)

LABEL_SOURCE = (
    DAY6
    / "02_step2_forward_labels"
    / "forward_label_panel.parquet"
)

CONTROL_SOURCE = (
    DAY7
    / "02_stage2_traditional_characteristic_controls"
    / "traditional_characteristic_panel.parquet"
)

LIQUIDITY_SOURCE = (
    DAY8
    / "01_stage1_liquidity_panel"
    / "stock_liquidity_panel.parquet"
)


# ============================================================
# 1. Frozen design
# ============================================================

WINDOWS = [60, 120, 252]

FACTORS = [
    "residual_degree_percentile",
    "delta_degree_percentile",
    "cross_industry_degree_percentile",
    "cross_industry_degree_ratio",
    "delta_residual_degree_percentile_1m",
    "delta_cross_industry_degree_percentile_1m",
    "neighbor_jaccard_1m",
    "neighbor_retention_1m",
    "outside_community_degree_percentile",
]

CONTROLS = [
    "industry_id1",
    "log_market_value",
    "momentum_120_20",
    "reversal_20",
    "volatility_60",
    "log_turnover_20",
]

# ------------------------------------------------------------
# IMPORTANT:
# The uploaded Day-6 return-label candidate file shows that
# "future_total_return" is the frozen total holding-period return.
# Freeze it explicitly rather than resolving it heuristically.
# ------------------------------------------------------------

RETURN_COLUMN_OVERRIDE = "future_total_return"

# Canonical name used by all Day-12 downstream code.
RETURN_TARGET = "future_holding_return"

# Day-6 frozen label-validity flag.
RETURN_VALIDITY_COLUMN = "future_return_label_valid"
REQUIRE_RETURN_VALIDITY_FLAG = True

# Keep only label-quality diagnostics that are useful for audit.
# They are NEVER predictors.
RETURN_QUALITY_COLS = [
    "future_return_label_valid",
    "future_return_coverage",
    "future_valid_return_days",
    "future_reopen_return_days",
]

# These future_* columns are allowed to remain in the final panel
# only because they are audit/availability diagnostics.
ALLOWED_FUTURE_AUDIT_COLUMNS = set(RETURN_QUALITY_COLS)

ADV_CANONICAL = "adv20"

RETURN_EQUAL_TOL = 1e-12


# ============================================================
# 2. Column aliases
# ============================================================

ID_ALIASES = [
    "security_id",
    "secID",
    "stock_code",
    "ticker",
]

DATE_ALIASES = [
    "analysis_date",
    "trade_date",
    "date",
]

WINDOW_ALIASES = [
    "window",
    "window_length",
]

LABEL_END_ALIASES = [
    "label_end_date",
    "next_analysis_date",
    "forward_end_date",
    "holding_end_date",
]

ADV_ALIASES = [
    "adv20",
    "ADV20",
    "adv_20",
    "average_dollar_volume_20",
    "avg_amount_20",
]


# ============================================================
# 3. IO helpers
# ============================================================

def atomic_csv(df, path):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")

    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        tmp,
        path,
    )


def read_file(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(
            path,
            low_memory=False,
        )

    raise ValueError(
        f"Unsupported file: {path}"
    )


def get_schema(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        return list(
            pq.ParquetFile(path)
            .schema_arrow
            .names
        )

    if suffix == ".csv":
        return list(
            pd.read_csv(
                path,
                nrows=0,
            ).columns
        )

    return []


def ensure_source(path, name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

    path_lower = str(path).lower()

    if "archive" in path_lower:
        raise RuntimeError(
            f"{name} points to an archive source: {path}"
        )

    if "panel_parts" in path_lower:
        raise RuntimeError(
            f"{name} points to a partition shard: {path}"
        )


def find_col(
    columns,
    aliases,
    required=True,
):
    lookup = {
        str(c).strip().lower(): c
        for c in columns
    }

    for alias in aliases:
        key = alias.lower()

        if key in lookup:
            return lookup[key]

    if required:
        raise RuntimeError(
            "Cannot find column from aliases: "
            f"{aliases}"
        )

    return None


# ============================================================
# 4. Type helpers
# ============================================================

def normalize_bool_series(s, name):
    """
    Convert a boolean-like Series safely.

    Accepted:
      True/False
      1/0
      yes/no
      y/n
      true/false strings

    Missing values remain False for the availability rule,
    while a separate QA count records missing validity flags.
    """

    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        numeric = pd.to_numeric(
            s,
            errors="coerce",
        )

        bad = (
            numeric.notna()
            &
            ~numeric.isin([0, 1])
        )

        if bad.any():
            vals = (
                numeric.loc[bad]
                .drop_duplicates()
                .head(10)
                .tolist()
            )

            raise RuntimeError(
                f"{name} contains numeric values "
                f"outside {{0,1}}: {vals}"
            )

        return (
            numeric
            .fillna(0)
            .astype(int)
            .astype(bool)
        )

    text = (
        s.astype("string")
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
    }

    mapped = text.map(mapping)

    bad = (
        text.notna()
        &
        mapped.isna()
    )

    if bad.any():
        vals = (
            text.loc[bad]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            f"{name} contains unrecognized boolean values: "
            f"{vals}"
        )

    return mapped.fillna(False).astype(bool)


# ============================================================
# 5. Key standardization
# ============================================================

def standardize_keys(
    df,
    require_window,
):
    out = df.copy()

    id_col = find_col(
        out.columns,
        ID_ALIASES,
        required=True,
    )

    date_col = find_col(
        out.columns,
        DATE_ALIASES,
        required=True,
    )

    rename = {}

    if id_col != "security_id":
        rename[id_col] = "security_id"

    if date_col != "analysis_date":
        rename[date_col] = "analysis_date"

    window_col = find_col(
        out.columns,
        WINDOW_ALIASES,
        required=require_window,
    )

    if (
        window_col is not None
        and window_col != "window"
    ):
        rename[window_col] = "window"

    out = out.rename(
        columns=rename
    )

    out["security_id"] = (
        out["security_id"]
        .astype("string")
        .str.strip()
    )

    out["analysis_date"] = pd.to_datetime(
        out["analysis_date"],
        errors="raise",
    )

    if "window" in out.columns:
        out["window"] = (
            pd.to_numeric(
                out["window"],
                errors="raise",
            )
            .astype(int)
        )

    return out


# ============================================================
# 6. Duplicate-collapse helper
# ============================================================

def collapse_invariant(
    df,
    key,
    value_cols,
    source_name,
):
    """
    If duplicated keys exist, allow collapse only when every
    retained value is exactly invariant within each key.
    """

    use_cols = list(
        dict.fromkeys(
            key + value_cols
        )
    )

    x = df[use_cols].copy()

    if not x[key].duplicated().any():
        return x

    bad = []

    grouped = x.groupby(
        key,
        dropna=False,
        sort=False,
    )

    for col in value_cols:
        max_nunique = (
            grouped[col]
            .nunique(dropna=False)
            .max()
        )

        if max_nunique > 1:
            bad.append(col)

    if bad:
        raise RuntimeError(
            f"{source_name}: duplicated keys contain "
            f"non-invariant values: {bad}"
        )

    return (
        x.drop_duplicates(key)
        .copy()
    )


# ============================================================
# 7. Frozen factor base
# ============================================================

def load_factor_base():
    x = standardize_keys(
        read_file(FACTOR_SOURCE),
        require_window=True,
    )

    missing = [
        c
        for c in FACTORS
        if c not in x.columns
    ]

    if missing:
        raise RuntimeError(
            f"Frozen factors missing: {missing}"
        )

    x = x[
        x["window"].isin(WINDOWS)
    ].copy()

    key = [
        "security_id",
        "analysis_date",
        "window",
    ]

    duplicate_n = int(
        x[key]
        .duplicated()
        .sum()
    )

    if duplicate_n > 0:
        raise RuntimeError(
            "Duplicate factor stock-date-window keys: "
            f"{duplicate_n}"
        )

    for c in FACTORS:
        x[c] = pd.to_numeric(
            x[c],
            errors="coerce",
        )

    meta_candidates = [
        "stock_code",
        "ticker",
        "stock_name",
        "exchange",
    ]

    keep = (
        key
        + [
            c
            for c in meta_candidates
            if c in x.columns
        ]
        + FACTORS
    )

    x = x[keep].copy()

    factor_values = (
        x[FACTORS]
        .to_numpy(dtype=float)
    )

    x[
        "network_feature_complete"
    ] = (
        x[FACTORS]
        .notna()
        .all(axis=1)
        &
        np.isfinite(
            factor_values
        ).all(axis=1)
    )

    return x


# ============================================================
# 8. Frozen Day-6 return labels
# ============================================================

def load_return_labels():
    y = standardize_keys(
        read_file(LABEL_SOURCE),
        require_window=False,
    )

    # --------------------------------------------------------
    # 8.1 Explicitly freeze future_total_return
    # --------------------------------------------------------

    if RETURN_COLUMN_OVERRIDE not in y.columns:
        candidates = [
            c
            for c in y.columns
            if "return" in str(c).lower()
        ]

        atomic_csv(
            pd.DataFrame(
                {
                    "candidate_return_column":
                        candidates
                }
            ),
            OUTDIR
            / "return_label_candidates.csv",
        )

        raise RuntimeError(
            "Frozen return column "
            f"{RETURN_COLUMN_OVERRIDE!r} not found in "
            "Day-6 forward_label_panel.parquet. "
            "See return_label_candidates.csv."
        )

    source_return_col = (
        RETURN_COLUMN_OVERRIDE
    )

    # --------------------------------------------------------
    # 8.2 Label-end column
    # --------------------------------------------------------

    label_end_col = find_col(
        y.columns,
        LABEL_END_ALIASES,
        required=False,
    )

    rename = {
        source_return_col:
            RETURN_TARGET,
    }

    if (
        label_end_col is not None
        and label_end_col != "label_end_date"
    ):
        rename[
            label_end_col
        ] = "label_end_date"

    y = y.rename(
        columns=rename
    )

    y[RETURN_TARGET] = pd.to_numeric(
        y[RETURN_TARGET],
        errors="coerce",
    )

    if "label_end_date" in y.columns:
        y["label_end_date"] = pd.to_datetime(
            y["label_end_date"],
            errors="coerce",
        )

    # --------------------------------------------------------
    # 8.3 Require / preserve Day-6 validity flag
    # --------------------------------------------------------

    validity_found = (
        RETURN_VALIDITY_COLUMN
        in y.columns
    )

    if (
        REQUIRE_RETURN_VALIDITY_FLAG
        and not validity_found
    ):
        raise RuntimeError(
            "Day-6 frozen return-validity flag "
            f"{RETURN_VALIDITY_COLUMN!r} is missing."
        )

    validity_missing_count_source = 0

    if validity_found:
        validity_missing_count_source = int(
            y[
                RETURN_VALIDITY_COLUMN
            ]
            .isna()
            .sum()
        )

        y[
            RETURN_VALIDITY_COLUMN
        ] = normalize_bool_series(
            y[
                RETURN_VALIDITY_COLUMN
            ],
            RETURN_VALIDITY_COLUMN,
        )

    # --------------------------------------------------------
    # 8.4 Preserve only audit-quality fields, not other
    #     future return outcomes.
    # --------------------------------------------------------

    quality_found = []

    for col in RETURN_QUALITY_COLS:
        if col not in y.columns:
            continue

        quality_found.append(col)

        if col == RETURN_VALIDITY_COLUMN:
            continue

        y[col] = pd.to_numeric(
            y[col],
            errors="coerce",
        )

    key = [
        "security_id",
        "analysis_date",
    ]

    if "window" in y.columns:
        key.append("window")

    value_cols = [
        RETURN_TARGET,
    ]

    if "label_end_date" in y.columns:
        value_cols.append(
            "label_end_date"
        )

    for col in quality_found:
        if col not in value_cols:
            value_cols.append(col)

    y = collapse_invariant(
        y,
        key,
        value_cols,
        "Day-6 return labels",
    )

    return {
        "data":
            y,

        "source_return_col":
            source_return_col,

        "resolution_method":
            "EXPLICIT_OVERRIDE",

        "validity_found":
            validity_found,

        "validity_missing_count_source":
            validity_missing_count_source,

        "quality_columns_found":
            quality_found,
    }


# ============================================================
# 9. Frozen traditional controls
# ============================================================

def load_controls():
    c = standardize_keys(
        read_file(CONTROL_SOURCE),
        require_window=False,
    )

    missing = [
        col
        for col in CONTROLS
        if col not in c.columns
    ]

    if missing:
        raise RuntimeError(
            f"Frozen controls missing: {missing}"
        )

    for col in CONTROLS:
        if col == "industry_id1":
            continue

        c[col] = pd.to_numeric(
            c[col],
            errors="coerce",
        )

    key = [
        "security_id",
        "analysis_date",
    ]

    if "window" in c.columns:
        key.append("window")

    c = collapse_invariant(
        c,
        key,
        CONTROLS,
        "Day-7 traditional controls",
    )

    return c


# ============================================================
# 10. Frozen stock-level liquidity
# ============================================================

def load_liquidity():
    q = standardize_keys(
        read_file(LIQUIDITY_SOURCE),
        require_window=False,
    )

    adv_col = find_col(
        q.columns,
        ADV_ALIASES,
        required=True,
    )

    source_adv_col = adv_col

    if adv_col != ADV_CANONICAL:
        q = q.rename(
            columns={
                adv_col:
                    ADV_CANONICAL
            }
        )

    q[
        ADV_CANONICAL
    ] = pd.to_numeric(
        q[
            ADV_CANONICAL
        ],
        errors="coerce",
    )

    key = [
        "security_id",
        "analysis_date",
    ]

    if "window" in q.columns:
        key.append("window")

    q = collapse_invariant(
        q,
        key,
        [ADV_CANONICAL],
        "Day-8 stock liquidity",
    )

    return (
        q,
        source_adv_col,
    )


# ============================================================
# 11. Left-join helper
# ============================================================

def left_join_optional_window(
    panel,
    right,
    source_name,
):
    if "window" in right.columns:
        key = [
            "security_id",
            "analysis_date",
            "window",
        ]

        validate = "one_to_one"

    else:
        key = [
            "security_id",
            "analysis_date",
        ]

        validate = "many_to_one"

    n0 = len(panel)

    out = panel.merge(
        right,
        on=key,
        how="left",
        validate=validate,
    )

    if len(out) != n0:
        raise RuntimeError(
            f"{source_name} merge changed "
            "the frozen factor universe."
        )

    return (
        out,
        key,
    )


# ============================================================
# 12. Expected holding-end date
# ============================================================

def add_expected_label_end(panel):
    """
    Compute the next frozen analysis date WITHIN each network
    window. This is safer than one global date map because the
    three frozen windows start at different dates.
    """

    date_table = (
        panel[
            [
                "window",
                "analysis_date",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(drop=True)
    )

    date_table[
        "expected_label_end_date"
    ] = (
        date_table.groupby(
            "window",
            sort=False,
        )[
            "analysis_date"
        ]
        .shift(-1)
    )

    return panel.merge(
        date_table,
        on=[
            "window",
            "analysis_date",
        ],
        how="left",
        validate="many_to_one",
    )


# ============================================================
# 13. Build frozen alpha evaluation panel
# ============================================================

def build_panel(
    base,
    label_info,
    controls,
    liquidity,
):
    panel = base.copy()
    factor_n = len(panel)

    labels = label_info[
        "data"
    ]

    # --------------------------------------------------------
    # 13.1 Return labels
    # --------------------------------------------------------

    panel, label_join_key = (
        left_join_optional_window(
            panel,
            labels,
            "Return label",
        )
    )

    panel = add_expected_label_end(
        panel
    )

    label_end_derived = (
        "label_end_date"
        not in panel.columns
    )

    label_end_mismatch_count = 0

    if label_end_derived:
        panel[
            "label_end_date"
        ] = panel[
            "expected_label_end_date"
        ]

    else:
        comparable = (
            panel[
                RETURN_TARGET
            ].notna()
            &
            panel[
                "label_end_date"
            ].notna()
            &
            panel[
                "expected_label_end_date"
            ].notna()
        )

        label_end_mismatch_count = int(
            (
                panel.loc[
                    comparable,
                    "label_end_date",
                ]
                !=
                panel.loc[
                    comparable,
                    "expected_label_end_date",
                ]
            ).sum()
        )

    # --------------------------------------------------------
    # 13.2 Controls
    # --------------------------------------------------------

    overlap_controls = [
        c
        for c in CONTROLS
        if c in panel.columns
    ]

    if overlap_controls:
        panel = panel.drop(
            columns=overlap_controls
        )

    panel, control_join_key = (
        left_join_optional_window(
            panel,
            controls,
            "Traditional control",
        )
    )

    # --------------------------------------------------------
    # 13.3 Liquidity
    # --------------------------------------------------------

    if ADV_CANONICAL in panel.columns:
        panel = panel.drop(
            columns=[
                ADV_CANONICAL
            ]
        )

    panel, liquidity_join_key = (
        left_join_optional_window(
            panel,
            liquidity,
            "Liquidity",
        )
    )

    if len(panel) != factor_n:
        raise RuntimeError(
            "Final panel does not preserve "
            "the frozen factor universe."
        )

    # --------------------------------------------------------
    # 13.4 Frozen return-label availability
    # --------------------------------------------------------

    source_return_nonmissing = (
        panel[
            RETURN_TARGET
        ].notna()
    )

    source_return_finite = np.isfinite(
        panel[
            RETURN_TARGET
        ].to_numpy(
            dtype=float
        )
    )

    panel[
        "source_return_nonmissing"
    ] = source_return_nonmissing

    panel[
        "source_return_finite"
    ] = source_return_finite

    if (
        RETURN_VALIDITY_COLUMN
        in panel.columns
    ):
        # It was normalized to bool at load time.
        valid_flag = (
            panel[
                RETURN_VALIDITY_COLUMN
            ]
            .fillna(False)
            .astype(bool)
        )

    else:
        valid_flag = pd.Series(
            True,
            index=panel.index,
            dtype=bool,
        )

    panel[
        "return_label_valid_flag"
    ] = valid_flag

    panel[
        "holding_interval_valid"
    ] = (
        panel[
            "label_end_date"
        ].notna()
        &
        (
            panel[
                "label_end_date"
            ]
            >
            panel[
                "analysis_date"
            ]
        )
    )

    panel[
        "return_target_available"
    ] = (
        source_return_nonmissing
        &
        source_return_finite
        &
        valid_flag
        &
        panel[
            "holding_interval_valid"
        ]
    )

    # --------------------------------------------------------
    # 13.5 Control availability
    # --------------------------------------------------------

    panel[
        "control_complete"
    ] = (
        panel[
            CONTROLS
        ]
        .notna()
        .all(axis=1)
    )

    numeric_controls = [
        c
        for c in CONTROLS
        if c != "industry_id1"
    ]

    control_values = (
        panel[
            numeric_controls
        ]
        .to_numpy(
            dtype=float
        )
    )

    panel[
        "control_numeric_finite"
    ] = np.isfinite(
        control_values
    ).all(axis=1)

    # --------------------------------------------------------
    # 13.6 Liquidity availability
    # --------------------------------------------------------

    adv_values = panel[
        ADV_CANONICAL
    ].to_numpy(
        dtype=float
    )

    panel[
        "liquidity_available"
    ] = (
        panel[
            ADV_CANONICAL
        ].notna()
        &
        np.isfinite(
            adv_values
        )
        &
        (
            panel[
                ADV_CANONICAL
            ] > 0
        )
    )

    # --------------------------------------------------------
    # 13.7 Evaluation samples
    # --------------------------------------------------------

    panel[
        "raw_alpha_evaluable"
    ] = (
        panel[
            "network_feature_complete"
        ]
        &
        panel[
            "return_target_available"
        ]
    )

    panel[
        "neutral_alpha_evaluable"
    ] = (
        panel[
            "raw_alpha_evaluable"
        ]
        &
        panel[
            "control_complete"
        ]
        &
        panel[
            "control_numeric_finite"
        ]
    )

    panel[
        "implementation_input_available"
    ] = (
        panel[
            "neutral_alpha_evaluable"
        ]
        &
        panel[
            "liquidity_available"
        ]
    )

    return {
        "panel":
            panel,

        "factor_n":
            factor_n,

        "label_join_key":
            label_join_key,

        "control_join_key":
            control_join_key,

        "liquidity_join_key":
            liquidity_join_key,

        "label_end_derived":
            label_end_derived,

        "label_end_mismatch_count":
            label_end_mismatch_count,
    }


# ============================================================
# 14. Audits
# ============================================================

def cross_window_return_mismatch_count(
    panel,
):
    """
    The holding-period return is not window-specific.
    On valid/evaluable return labels, the same stock-date
    must have the same return across W60/W120/W252.
    """

    x = panel.loc[
        panel[
            "return_target_available"
        ],
        [
            "security_id",
            "analysis_date",
            RETURN_TARGET,
        ],
    ].copy()

    if x.empty:
        return 0

    stat = (
        x.groupby(
            [
                "security_id",
                "analysis_date",
            ],
            sort=False,
        )[
            RETURN_TARGET
        ]
        .agg(
            min_return="min",
            max_return="max",
        )
    )

    return int(
        (
            (
                stat[
                    "max_return"
                ]
                -
                stat[
                    "min_return"
                ]
            ).abs()
            >
            RETURN_EQUAL_TOL
        ).sum()
    )


def make_date_summary(panel):
    out = (
        panel.groupby(
            [
                "window",
                "analysis_date",
            ],
            as_index=False,
        )
        .agg(
            stock_n=(
                "security_id",
                "size",
            ),

            network_complete_n=(
                "network_feature_complete",
                "sum",
            ),

            source_return_nonmissing_n=(
                "source_return_nonmissing",
                "sum",
            ),

            return_label_valid_flag_n=(
                "return_label_valid_flag",
                "sum",
            ),

            return_target_n=(
                "return_target_available",
                "sum",
            ),

            raw_alpha_evaluable_n=(
                "raw_alpha_evaluable",
                "sum",
            ),

            neutral_alpha_evaluable_n=(
                "neutral_alpha_evaluable",
                "sum",
            ),

            implementation_input_n=(
                "implementation_input_available",
                "sum",
            ),

            adv20_available_n=(
                "liquidity_available",
                "sum",
            ),

            label_end_date=(
                "label_end_date",
                "max",
            ),

            expected_label_end_date=(
                "expected_label_end_date",
                "max",
            ),
        )
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(drop=True)
    )

    count_cols = [
        "network_complete_n",
        "source_return_nonmissing_n",
        "return_label_valid_flag_n",
        "return_target_n",
        "raw_alpha_evaluable_n",
        "neutral_alpha_evaluable_n",
        "implementation_input_n",
        "adv20_available_n",
    ]

    for col in count_cols:
        share_col = (
            col[:-2]
            + "_share"
            if col.endswith("_n")
            else col + "_share"
        )

        out[
            share_col
        ] = (
            out[col]
            /
            out["stock_n"]
        )

    return out


# ============================================================
# 15. Main
# ============================================================

def main():
    print("=" * 80)
    print("M1 - Research Day 12 - Step 1")
    print("Frozen Alpha Evaluation Panel & Return-Label Audit")
    print("=" * 80)

    # --------------------------------------------------------
    # 15.1 Source integrity
    # --------------------------------------------------------

    ensure_source(
        FACTOR_SOURCE,
        "Factor source",
    )

    ensure_source(
        LABEL_SOURCE,
        "Return source",
    )

    ensure_source(
        CONTROL_SOURCE,
        "Control source",
    )

    ensure_source(
        LIQUIDITY_SOURCE,
        "Liquidity source",
    )

    print()
    print(
        f"Factor source   : "
        f"{FACTOR_SOURCE}"
    )
    print(
        f"Return source   : "
        f"{LABEL_SOURCE}"
    )
    print(
        f"Control source  : "
        f"{CONTROL_SOURCE}"
    )
    print(
        f"Liquidity source: "
        f"{LIQUIDITY_SOURCE}"
    )

    # --------------------------------------------------------
    # 15.2 Load frozen inputs
    # --------------------------------------------------------

    base = load_factor_base()

    label_info = (
        load_return_labels()
    )

    controls = (
        load_controls()
    )

    (
        liquidity,
        source_adv_col,
    ) = load_liquidity()

    print()
    print(
        "Resolved return column: "
        f"{label_info['source_return_col']}"
    )

    print(
        "Return resolution method: "
        f"{label_info['resolution_method']}"
    )

    print(
        "Return validity flag found: "
        f"{label_info['validity_found']}"
    )

    print(
        "Return quality columns found: "
        f"{label_info['quality_columns_found']}"
    )

    print(
        "Resolved ADV20 source column: "
        f"{source_adv_col}"
    )

    # --------------------------------------------------------
    # 15.3 Build frozen evaluation panel
    # --------------------------------------------------------

    result = build_panel(
        base,
        label_info,
        controls,
        liquidity,
    )

    panel = (
        result[
            "panel"
        ]
        .sort_values(
            [
                "window",
                "analysis_date",
                "security_id",
            ]
        )
        .reset_index(drop=True)
    )

    key = [
        "security_id",
        "analysis_date",
        "window",
    ]

    # --------------------------------------------------------
    # 15.4 Return-label audit
    # --------------------------------------------------------

    valid_target = (
        panel[
            "return_target_available"
        ]
    )

    nonfinite_return_count = int(
        (
            panel[
                "source_return_nonmissing"
            ]
            &
            ~panel[
                "source_return_finite"
            ]
        ).sum()
    )

    return_with_missing_end_count = int(
        (
            valid_target
            &
            panel[
                "label_end_date"
            ].isna()
        ).sum()
    )

    nonforward_return_interval_count = int(
        (
            panel[
                "source_return_nonmissing"
            ]
            &
            panel[
                "label_end_date"
            ].notna()
            &
            (
                panel[
                    "label_end_date"
                ]
                <=
                panel[
                    "analysis_date"
                ]
            )
        ).sum()
    )

    invalid_nonmissing_return_count = int(
        (
            panel[
                "source_return_nonmissing"
            ]
            &
            ~panel[
                "return_label_valid_flag"
            ]
        ).sum()
    )

    valid_flag_true_but_missing_target_count = int(
        (
            panel[
                "return_label_valid_flag"
            ]
            &
            ~panel[
                "source_return_nonmissing"
            ]
        ).sum()
    )

    cross_window_mismatch_count = (
        cross_window_return_mismatch_count(
            panel
        )
    )

    negative_adv20_count = int(
        (
            panel[
                ADV_CANONICAL
            ].notna()
            &
            (
                panel[
                    ADV_CANONICAL
                ] < 0
            )
        ).sum()
    )

    zero_adv20_count = int(
        (
            panel[
                ADV_CANONICAL
            ].notna()
            &
            (
                panel[
                    ADV_CANONICAL
                ] == 0
            )
        ).sum()
    )

    # Future columns other than the frozen outcome and explicitly
    # allowed audit fields are not allowed in the final panel.
    unexpected_future_cols = [
        c
        for c in panel.columns
        if (
            str(c)
            .lower()
            .startswith("future_")
            and
            c != RETURN_TARGET
            and
            c not in ALLOWED_FUTURE_AUDIT_COLUMNS
        )
    ]

    # --------------------------------------------------------
    # 15.5 Summaries
    # --------------------------------------------------------

    date_summary = (
        make_date_summary(
            panel
        )
    )

    eval_calendar = (
        date_summary[
            [
                "window",
                "analysis_date",
                "stock_n",
                "network_complete_n",
                "source_return_nonmissing_n",
                "return_label_valid_flag_n",
                "return_target_n",
                "raw_alpha_evaluable_n",
                "neutral_alpha_evaluable_n",
                "implementation_input_n",
                "label_end_date",
                "expected_label_end_date",
            ]
        ]
        .copy()
    )

    eval_calendar[
        "return_label_realized"
    ] = (
        eval_calendar[
            "return_target_n"
        ] > 0
    )

    # --------------------------------------------------------
    # 15.6 Formal QA
    # --------------------------------------------------------

    qa = {
        "factor_source_exists":
            bool(
                FACTOR_SOURCE.exists()
            ),

        "return_source_exists":
            bool(
                LABEL_SOURCE.exists()
            ),

        "control_source_exists":
            bool(
                CONTROL_SOURCE.exists()
            ),

        "liquidity_source_exists":
            bool(
                LIQUIDITY_SOURCE.exists()
            ),

        "factor_source_row_count":
            int(
                result[
                    "factor_n"
                ]
            ),

        "alpha_panel_row_count":
            int(
                len(panel)
            ),

        "factor_universe_preserved":
            bool(
                len(panel)
                ==
                result[
                    "factor_n"
                ]
            ),

        "duplicate_stock_date_window_count":
            int(
                panel[
                    key
                ]
                .duplicated()
                .sum()
            ),

        "frozen_factor_count":
            int(
                len(FACTORS)
            ),

        "window_count":
            int(
                panel[
                    "window"
                ].nunique()
            ),

        "frozen_control_count":
            int(
                len(CONTROLS)
            ),

        "source_return_column":
            label_info[
                "source_return_col"
            ],

        "expected_source_return_column":
            RETURN_COLUMN_OVERRIDE,

        "return_source_column_is_frozen":
            bool(
                label_info[
                    "source_return_col"
                ]
                ==
                RETURN_COLUMN_OVERRIDE
            ),

        "return_resolution_method":
            label_info[
                "resolution_method"
            ],

        "return_validity_flag_found":
            bool(
                label_info[
                    "validity_found"
                ]
            ),

        "return_validity_flag_missing_source_count":
            int(
                label_info[
                    "validity_missing_count_source"
                ]
            ),

        "return_quality_columns_found":
            "|".join(
                label_info[
                    "quality_columns_found"
                ]
            ),

        "source_return_nonmissing_row_count":
            int(
                panel[
                    "source_return_nonmissing"
                ].sum()
            ),

        "return_label_valid_true_row_count":
            int(
                panel[
                    "return_label_valid_flag"
                ].sum()
            ),

        "return_target_available_row_count":
            int(
                panel[
                    "return_target_available"
                ].sum()
            ),

        "return_target_unavailable_row_count":
            int(
                (
                    ~panel[
                        "return_target_available"
                    ]
                ).sum()
            ),

        # Descriptive only: this may legitimately be > 0.
        "nonmissing_but_invalid_return_count":
            invalid_nonmissing_return_count,

        "valid_flag_true_but_missing_target_count":
            valid_flag_true_but_missing_target_count,

        "return_target_used_to_define_current_universe":
            False,

        "label_end_date_derived":
            bool(
                result[
                    "label_end_derived"
                ]
            ),

        "label_end_date_mismatch_count":
            int(
                result[
                    "label_end_mismatch_count"
                ]
            ),

        "return_with_missing_label_end_count":
            return_with_missing_end_count,

        "nonforward_return_interval_count":
            nonforward_return_interval_count,

        "cross_window_return_mismatch_count":
            cross_window_mismatch_count,

        "nonfinite_source_return_count":
            nonfinite_return_count,

        "negative_adv20_count":
            negative_adv20_count,

        # Descriptive only: zero ADV implies not implementable.
        "zero_adv20_count":
            zero_adv20_count,

        "unexpected_future_column_count":
            int(
                len(
                    unexpected_future_cols
                )
            ),

        "unexpected_future_columns":
            "|".join(
                unexpected_future_cols
            ),

        "control_set_explicitly_frozen":
            True,

        "return_quality_fields_used_as_predictors":
            False,

        "adv20_reestimated":
            False,

        "return_label_reestimated":
            False,

        "network_features_reestimated":
            False,

        "controls_reestimated":
            False,

        "factor_selected_from_results":
            False,

        "window_selected_from_results":
            False,

        "future_information_used_as_predictor":
            False,
    }

    qa[
        "all_formal_qa_pass"
    ] = bool(
        qa[
            "factor_source_exists"
        ]
        and
        qa[
            "return_source_exists"
        ]
        and
        qa[
            "control_source_exists"
        ]
        and
        qa[
            "liquidity_source_exists"
        ]
        and
        qa[
            "factor_universe_preserved"
        ]
        and
        qa[
            "duplicate_stock_date_window_count"
        ] == 0
        and
        qa[
            "frozen_factor_count"
        ] == 9
        and
        qa[
            "window_count"
        ] == 3
        and
        qa[
            "frozen_control_count"
        ] == 6
        and
        qa[
            "return_source_column_is_frozen"
        ]
        and
        (
            (
                not REQUIRE_RETURN_VALIDITY_FLAG
            )
            or
            qa[
                "return_validity_flag_found"
            ]
        )
        and
        qa[
            "valid_flag_true_but_missing_target_count"
        ] == 0
        and
        qa[
            "label_end_date_mismatch_count"
        ] == 0
        and
        qa[
            "return_with_missing_label_end_count"
        ] == 0
        and
        qa[
            "nonforward_return_interval_count"
        ] == 0
        and
        qa[
            "cross_window_return_mismatch_count"
        ] == 0
        and
        qa[
            "nonfinite_source_return_count"
        ] == 0
        and
        qa[
            "negative_adv20_count"
        ] == 0
        and
        qa[
            "unexpected_future_column_count"
        ] == 0
    )

    # --------------------------------------------------------
    # 15.7 Save frozen panel
    # --------------------------------------------------------

    panel.to_parquet(
        OUTDIR
        / "frozen_alpha_evaluation_panel.parquet",
        index=False,
        compression="zstd",
    )

    atomic_csv(
        date_summary,
        OUTDIR
        / "alpha_evaluation_date_summary.csv",
    )

    atomic_csv(
        eval_calendar,
        OUTDIR
        / "alpha_evaluation_calendar.csv",
    )

    # --------------------------------------------------------
    # 15.8 Schema
    # --------------------------------------------------------

    schema_df = pd.DataFrame(
        {
            "column":
                panel.columns,

            "dtype":
                [
                    str(
                        panel[c].dtype
                    )
                    for c
                    in panel.columns
                ],
        }
    )

    roles = []

    for c in panel.columns:
        if c in key:
            roles.append(
                "key"
            )

        elif c in FACTORS:
            roles.append(
                "frozen_network_factor"
            )

        elif c in CONTROLS:
            roles.append(
                "frozen_traditional_control"
            )

        elif c == RETURN_TARGET:
            roles.append(
                "forward_return_outcome"
            )

        elif c == "label_end_date":
            roles.append(
                "outcome_availability_time"
            )

        elif c == "expected_label_end_date":
            roles.append(
                "audit_expected_outcome_end"
            )

        elif c == ADV_CANONICAL:
            roles.append(
                "frozen_liquidity_input"
            )

        elif c in RETURN_QUALITY_COLS:
            roles.append(
                "forward_return_quality_audit_only"
            )

        else:
            roles.append(
                "audit_or_metadata"
            )

    schema_df[
        "role"
    ] = roles

    atomic_csv(
        schema_df,
        OUTDIR
        / "alpha_evaluation_panel_schema.csv",
    )

    # --------------------------------------------------------
    # 15.9 Source manifest
    # --------------------------------------------------------

    manifest = pd.DataFrame(
        [
            {
                "source_role":
                    "frozen_network_factors",

                "path":
                    str(
                        FACTOR_SOURCE
                    ),

                "join_level":
                    "stock-date-window",
            },

            {
                "source_role":
                    "frozen_forward_total_return",

                "path":
                    str(
                        LABEL_SOURCE
                    ),

                "source_column":
                    label_info[
                        "source_return_col"
                    ],

                "canonical_column":
                    RETURN_TARGET,

                "join_level":
                    "+".join(
                        result[
                            "label_join_key"
                        ]
                    ),
            },

            {
                "source_role":
                    "frozen_traditional_controls",

                "path":
                    str(
                        CONTROL_SOURCE
                    ),

                "join_level":
                    "+".join(
                        result[
                            "control_join_key"
                        ]
                    ),
            },

            {
                "source_role":
                    "frozen_stock_liquidity",

                "path":
                    str(
                        LIQUIDITY_SOURCE
                    ),

                "source_column":
                    source_adv_col,

                "canonical_column":
                    ADV_CANONICAL,

                "join_level":
                    "+".join(
                        result[
                            "liquidity_join_key"
                        ]
                    ),
            },
        ]
    )

    atomic_csv(
        manifest,
        OUTDIR
        / "alpha_evaluation_source_manifest.csv",
    )

    # --------------------------------------------------------
    # 15.10 QA
    # --------------------------------------------------------

    atomic_csv(
        pd.DataFrame(
            [
                {
                    "qa_name":
                        k,

                    "qa_value":
                        v,
                }
                for k, v
                in qa.items()
            ]
        ),
        OUTDIR
        / "day12_step1_qa.csv",
    )

    # --------------------------------------------------------
    # 15.11 Metadata
    # --------------------------------------------------------

    metadata = {
        "research_day":
            12,

        "step":
            (
                "Step1_Frozen_Alpha_Evaluation_"
                "Panel_and_Return_Label_Audit"
            ),

        "purpose":
            (
                "Construct the single frozen stock-date-window "
                "panel for Day-12 alpha evaluation without "
                "re-estimating network factors, forward returns, "
                "traditional controls, or liquidity inputs."
            ),

        "frozen_windows":
            WINDOWS,

        "frozen_network_factors":
            FACTORS,

        "frozen_controls":
            CONTROLS,

        "return_target":
            RETURN_TARGET,

        "source_return_column":
            label_info[
                "source_return_col"
            ],

        "return_resolution_method":
            label_info[
                "resolution_method"
            ],

        "return_validity_column":
            RETURN_VALIDITY_COLUMN,

        "return_validity_required":
            REQUIRE_RETURN_VALIDITY_FLAG,

        "return_quality_columns_preserved":
            label_info[
                "quality_columns_found"
            ],

        "return_availability_rule":
            (
                "A row is return-target-available only when "
                "future_total_return is non-missing and finite, "
                "future_return_label_valid is true, and the "
                "holding interval has a valid label_end_date "
                "strictly after analysis_date."
            ),

        "universe_rule":
            (
                "The Day-5 frozen factor universe is the LEFT "
                "base. Future-return availability, return-label "
                "validity, control availability, and liquidity "
                "availability never define the current stock universe."
            ),

        "return_label_rule":
            (
                "The frozen Day-6 future_total_return is used "
                "only as an outcome. Label quality fields are "
                "audit-only and are never predictors."
            ),

        "cross_window_return_rule":
            (
                "For a given security_id and analysis_date, "
                "the valid frozen future holding return must be "
                "invariant across W60/W120/W252 because the "
                "holding-period outcome is not network-window-specific."
            ),

        "day8_capacity_note":
            (
                "Step 1 merges only stock-level frozen ADV20. "
                "Day-8 pair/strategy capacity outputs are not "
                "merged into this stock-level panel and will be "
                "reused only in the later capacity-validation step."
            ),

        "formal_qa":
            qa,
    }

    with open(
        OUTDIR
        / "day12_step1_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # --------------------------------------------------------
    # 15.12 Console summary
    # --------------------------------------------------------

    print()
    print(
        "Date-window coverage:"
    )

    show = (
        date_summary.groupby(
            "window"
        )
        .agg(
            date_n=(
                "analysis_date",
                "nunique",
            ),

            mean_stock_n=(
                "stock_n",
                "mean",
            ),

            mean_source_return_share=(
                "source_return_nonmissing_share",
                "mean",
            ),

            mean_valid_return_share=(
                "return_target_share",
                "mean",
            ),

            mean_raw_alpha_share=(
                "raw_alpha_evaluable_share",
                "mean",
            ),

            mean_neutral_alpha_share=(
                "neutral_alpha_evaluable_share",
                "mean",
            ),

            mean_implementation_share=(
                "implementation_input_share",
                "mean",
            ),
        )
    )

    print(
        show.to_string()
    )

    print()
    print(
        "Formal QA:"
    )

    for k, v in qa.items():
        print(
            f"  {k}: {v}"
        )

    if not qa[
        "all_formal_qa_pass"
    ]:
        raise RuntimeError(
            "Day-12 Step-1 formal QA failed. "
            "Inspect day12_step1_qa.csv before proceeding."
        )

    print()
    print("=" * 80)
    print("DAY 12 STEP 1 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
