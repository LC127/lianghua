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

CONTROL_ROOT = (
    DAY7
    / "02_stage2_traditional_characteristic_controls"
)

OUTDIR = (
    OUT / "M1_day11"
    / "01_stage1_frozen_prediction_panel"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

# If automatic discovery is ambiguous, set exact paths here.
FACTOR_SOURCE_OVERRIDE = None
LABEL_SOURCE_OVERRIDE = (
    r"D:\M1_StockNetwork\output\M1_day6"
    r"\02_step2_forward_labels"
    r"\forward_label_panel.parquet"
)
CONTROL_SOURCE_OVERRIDE = (
    r"D:\M1_StockNetwork\output\M1_day7"
    r"\02_stage2_traditional_characteristic_controls"
    r"\traditional_characteristic_panel.parquet"
)

# IMPORTANT:
# Do not guess the Day-7 FULL control set.
# Once confirmed, e.g.
# CONTROL_COLUMNS_OVERRIDE = ["...", "..."]
CONTROL_COLUMNS_OVERRIDE = [
    "industry_id1",
    "log_market_value",
    "momentum_120_20",
    "reversal_20",
    "volatility_60",
    "log_turnover_20",
]


# ============================================================
# 1. Frozen design
# ============================================================

WINDOWS = [60, 120, 252]

TARGET = "future_idio_vol_annualized"

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

MIN_TRAIN_PERIODS = 24

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

META_KEEP = [
    "stock_code",
    "ticker",
    "exchange",
    "industry_id1",
    "market_value",
    "neg_market_value",
    "turnover_rate",
]

# ============================================================
# 2. IO helpers
# ============================================================

def save_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def read_file(path):
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)

    raise ValueError(f"Unsupported file: {path}")


def schema(path):
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            return list(
                pq.ParquetFile(path).schema_arrow.names
            )

        if path.suffix.lower() == ".csv":
            return list(
                pd.read_csv(path, nrows=0).columns
            )
    except Exception:
        pass

    return []


def find_col(cols, aliases, required=True):
    lookup = {
        str(c).strip().lower(): c
        for c in cols
    }

    for a in aliases:
        if a.lower() in lookup:
            return lookup[a.lower()]

    if required:
        raise RuntimeError(
            f"Cannot find column from aliases: {aliases}"
        )

    return None


# ============================================================
# 3. Source discovery
# ============================================================

def discover(root, kind):

    if not root.exists():
        raise FileNotFoundError(root)

    candidates = []

    for p in root.rglob("*"):

        if (
            not p.is_file()
            or p.suffix.lower()
            not in {".csv", ".parquet", ".pq"}
        ):
            continue

        name = p.name.lower()

        # --------------------------------------------------------
        # Exclude historical archives and partition shards
        # from canonical source discovery.
        # --------------------------------------------------------
        path_lower = str(p).lower()
        
        if "archive" in path_lower:
            continue
        
        if "panel_parts" in path_lower:
            continue

        # Do not select aggregate result files.
        if any(
            x in name
            for x in [
                "summary",
                "qa",
                "metadata",
                "rank_ic",
                "manifest",
            ]
        ):
            continue

        cols = schema(p)

        has_id = any(x in cols for x in ID_ALIASES)
        has_date = any(x in cols for x in DATE_ALIASES)

        if not (has_id and has_date):
            continue

        has_window = any(
            x in cols for x in WINDOW_ALIASES
        )

        if kind == "factor":
            ok = (
                has_window
                and all(f in cols for f in FACTORS)
            )

        elif kind == "label":
            ok = TARGET in cols

        elif kind == "control":
            # Day-7 traditional characteristics are stock-date variables
            # and do NOT have to be network-window specific.  A canonical
            # control panel may therefore be keyed only by
            # (security_id, analysis_date).  If a window column exists we
            # keep it, but it is not required for source discovery.
            ok = (
                "market_value" in cols
                or "industry_id1" in cols
                or "turnover_rate" in cols
            )

        else:
            raise ValueError(kind)

        if not ok:
            continue

        score = (
            5 * ("panel" in name)
            + 4 * (kind in name)
            + 3 * ("control" in name)
            + 2 * (p.suffix.lower() == ".parquet")
        )

        candidates.append(
            {
                "kind": kind,
                "score": score,
                "path": str(p),
                "n_columns": len(cols),
            }
        )

    if not candidates:
        raise RuntimeError(
            f"No {kind} source found under {root}"
        )

    c = pd.DataFrame(candidates).sort_values(
        ["score", "path"],
        ascending=[False, True],
    )

    # Require a unique highest score.
    best_score = c.iloc[0]["score"]
    best = c[c["score"] == best_score]

    if len(best) != 1:
        save_csv(
            c,
            OUTDIR / f"{kind}_source_candidates.csv",
        )
        raise RuntimeError(
            f"Ambiguous {kind} source. "
            f"See {kind}_source_candidates.csv "
            "and set the corresponding *_SOURCE_OVERRIDE."
        )

    save_csv(
        c,
        OUTDIR / f"{kind}_source_candidates.csv",
    )

    return Path(best.iloc[0]["path"])


# ============================================================
# 4. Standardize keys
# ============================================================

def standardize(df, require_window=True):

    df = df.copy()

    sid = find_col(df.columns, ID_ALIASES)
    date = find_col(df.columns, DATE_ALIASES)

    window = find_col(
        df.columns,
        WINDOW_ALIASES,
        required=require_window,
    )

    rename = {
        sid: "security_id",
        date: "analysis_date",
    }

    if window is not None:
        rename[window] = "window"

    df = df.rename(columns=rename)

    df["security_id"] = (
        df["security_id"]
        .astype("string")
        .str.strip()
    )

    df["analysis_date"] = pd.to_datetime(
        df["analysis_date"],
        errors="raise",
    )

    if "window" in df:
        df["window"] = pd.to_numeric(
            df["window"],
            errors="raise",
        ).astype(int)

    return df

# ============================================================
# 5. Resolve sources
# ============================================================

def resolve_sources():

    factor = (
        Path(FACTOR_SOURCE_OVERRIDE)
        if FACTOR_SOURCE_OVERRIDE
        else discover(DAY5, "factor")
    )

    label = (
        Path(LABEL_SOURCE_OVERRIDE)
        if LABEL_SOURCE_OVERRIDE
        else discover(DAY6, "label")
    )

    try:
        control = (
            Path(CONTROL_SOURCE_OVERRIDE)
            if CONTROL_SOURCE_OVERRIDE
            else discover(CONTROL_ROOT, "control")
        )
    except RuntimeError:
        control = None

    return factor, label, control


# ============================================================
# 6. Factor universe = LEFT BASE
# ============================================================

def load_factor_base(path):

    x = standardize(
        read_file(path),
        require_window=True,
    )

    missing = [
        f for f in FACTORS
        if f not in x.columns
    ]

    if missing:
        raise RuntimeError(
            f"Frozen factors missing: {missing}"
        )

    x = x[x["window"].isin(WINDOWS)].copy()

    key = [
        "security_id",
        "analysis_date",
        "window",
    ]

    if x[key].duplicated().any():
        raise RuntimeError(
            "Duplicate stock-date-window "
            "in frozen factor universe."
        )

    keep = key + FACTORS + [
        c for c in META_KEEP
        if c in x.columns
        and c not in key
        and c not in FACTORS
    ]

    x = x[keep].copy()

    x["network_feature_complete"] = (
        x[FACTORS].notna().all(axis=1)
    )

    return x

# ============================================================
# 7. Frozen future-IVOL labels
# ============================================================

def load_labels(path):

    y = standardize(
        read_file(path),
        require_window=False,
    )

    if TARGET not in y.columns:
        raise RuntimeError(
            f"{TARGET} not found in label source."
        )

    end_col = find_col(
        y.columns,
        LABEL_END_ALIASES,
        required=False,
    )

    if (
        end_col is not None
        and end_col != "label_end_date"
    ):
        y = y.rename(
            columns={end_col: "label_end_date"}
        )

    keep = [
        "security_id",
        "analysis_date",
    ]

    if "window" in y.columns:
        keep.append("window")

    keep.append(TARGET)

    if "label_end_date" in y.columns:
        keep.append("label_end_date")

    y = y[keep].copy()

    y[TARGET] = pd.to_numeric(
        y[TARGET],
        errors="coerce",
    )

    if "label_end_date" in y:
        y["label_end_date"] = pd.to_datetime(
            y["label_end_date"],
            errors="coerce",
        )

    key = (
        ["security_id", "analysis_date", "window"]
        if "window" in y.columns
        else ["security_id", "analysis_date"]
    )

    if y[key].duplicated().any():
        raise RuntimeError(
            f"Duplicate label keys: {key}"
        )

    return y
# ============================================================
# 8. Optional frozen control source
# ============================================================

def load_controls(path):

    if path is None:
        return None, []

    # IMPORTANT:
    # Day-7 traditional stock characteristics are usually keyed by
    # (security_id, analysis_date), not by network window.  Therefore
    # window is optional here.  When it is absent, the same PIT control
    # observation is broadcast to W60/W120/W252 during the LEFT join.
    c = standardize(
        read_file(path),
        require_window=False,
    )

    key = [
        "security_id",
        "analysis_date",
    ]

    if "window" in c.columns:
        c = c[c["window"].isin(WINDOWS)].copy()
        key.append("window")

    excluded = set(
        key
        + FACTORS
        + [
            TARGET,
            "factor_name",
            "factor_order",
            "risk_rank_ic",
            "method",
            "target_name",
            "ic_status",
        ]
    )

    # Never allow another forward variable into predictors.
    candidates = [
        col for col in c.columns
        if col not in excluded
        and not str(col).lower().startswith("future_")
        and "label" not in str(col).lower()
        and "outcome" not in str(col).lower()
    ]

    if CONTROL_COLUMNS_OVERRIDE is not None:

        missing = [
            col for col in CONTROL_COLUMNS_OVERRIDE
            if col not in c.columns
        ]

        if missing:
            raise RuntimeError(
                f"Confirmed control columns missing: {missing}"
            )

        candidates = list(
            CONTROL_COLUMNS_OVERRIDE
        )

    # Controls must be unique at their native key.  If the Day-7 panel
    # contains repeated rows (for example because it was stored in long
    # form), duplicates are allowed only when every candidate control is
    # identical within that key.
    if c[key].duplicated().any():

        bad = []

        for col in candidates:
            if (
                c.groupby(key, dropna=False)[col]
                .nunique(dropna=False)
                .max()
                > 1
            ):
                bad.append(col)

        if bad:
            raise RuntimeError(
                "Candidate controls vary inside native control key "
                f"{key}: {bad}"
            )

        c = c[
            key + candidates
        ].drop_duplicates(key)

    else:
        c = c[
            key + candidates
        ].copy()

    return c, candidates

# ============================================================
# 9. Build prediction panel
# ============================================================

def build_panel(base, labels, controls):

    panel = base.copy()
    n0 = len(panel)

    if "window" in labels.columns:
        label_key = [
            "security_id",
            "analysis_date",
            "window",
        ]
        validate = "one_to_one"
    else:
        label_key = [
            "security_id",
            "analysis_date",
        ]
        validate = "many_to_one"

    panel = panel.merge(
        labels,
        on=label_key,
        how="left",
        validate=validate,
    )

    if len(panel) != n0:
        raise RuntimeError(
            "Label merge changed current factor universe."
        )

    # --------------------------------------------------------
    # Derive frozen label end if source does not provide it.
    # Day-6 label semantics:
    # (analysis_date, next_analysis_date]
    # --------------------------------------------------------

    dates = sorted(
        panel["analysis_date"]
        .drop_duplicates()
        .tolist()
    )

    next_map = {
        dates[i]: dates[i + 1]
        for i in range(len(dates) - 1)
    }

    derived_end = panel[
        "analysis_date"
    ].map(next_map)

    label_end_derived = (
        "label_end_date" not in panel.columns
    )

    mismatch_n = 0

    if label_end_derived:
        panel["label_end_date"] = derived_end
    else:
        valid = (
            panel["label_end_date"].notna()
            & derived_end.notna()
        )

        mismatch_n = int(
            (
                panel.loc[valid, "label_end_date"]
                != derived_end.loc[valid]
            ).sum()
        )

        if mismatch_n > 0:
            raise RuntimeError(
                "Existing label_end_date disagrees "
                "with next analysis date."
            )

    # --------------------------------------------------------
    # Controls: still LEFT join; never define universe.
    # --------------------------------------------------------

    if controls is not None:

        # The canonical Day-7 control panel may be stock-date only.
        # In that case the same PIT characteristic vector is correctly
        # reused for each network window on the same date.
        if "window" in controls.columns:
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

        # If Day-5 factor metadata contains a column with the same name
        # as the canonical Day-7 control panel, use the Day-7 value as
        # the control definition rather than creating ambiguous suffixes.
        overlap = [
            col for col in controls.columns
            if col not in key and col in panel.columns
        ]

        if overlap:
            panel = panel.drop(columns=overlap)

        panel = panel.merge(
            controls,
            on=key,
            how="left",
            validate=validate,
        )

        if len(panel) != n0:
            raise RuntimeError(
                "Control merge changed current factor universe."
            )

    panel["target_available"] = (
        panel[TARGET].notna()
    )

    panel["label_interval_valid"] = (
        panel["label_end_date"].notna()
        & (
            panel["label_end_date"]
            > panel["analysis_date"]
        )
    )

    return panel, label_end_derived, mismatch_n

# ============================================================
# 10. OOS training calendar
# ============================================================

def build_training_calendar(panel):

    period = (
        panel.groupby(
            ["window", "analysis_date"],
            as_index=False,
        )
        .agg(
            stock_n=("security_id", "size"),
            feature_complete_n=(
                "network_feature_complete",
                "sum",
            ),
            target_n=(
                "target_available",
                "sum",
            ),
            label_end_date=(
                "label_end_date",
                "max",
            ),
        )
        .sort_values(
            ["window", "analysis_date"]
        )
    )

    rows = []

    for w, g in period.groupby("window"):

        g = g.sort_values(
            "analysis_date"
        ).copy()

        for _, cur in g.iterrows():

            t = cur["analysis_date"]

            # Strict OOS training rule:
            # historical formation date < t
            # AND its forward label is fully realized by t.
            hist = g[
                (g["analysis_date"] < t)
                & (g["label_end_date"] <= t)
                & (g["target_n"] > 0)
            ]

            rows.append({
                "window": int(w),
                "forecast_date": t,

                "forecast_stock_n":
                    int(cur["stock_n"]),

                "forecast_feature_complete_n":
                    int(cur["feature_complete_n"]),

                "forecast_target_n":
                    int(cur["target_n"]),

                "realized_train_period_n":
                    int(len(hist)),

                "realized_train_row_n":
                    int(hist["target_n"].sum()),

                "max_train_label_end_date":
                    (
                        hist["label_end_date"].max()
                        if len(hist)
                        else pd.NaT
                    ),

                "min_24_periods_met":
                    bool(
                        len(hist)
                        >= MIN_TRAIN_PERIODS
                    ),
            })

    return pd.DataFrame(rows)

# ============================================================
# 11. Main
# ============================================================

def main():

    print("=" * 76)
    print("M1 - Research Day 11 - Step 1")
    print("Frozen Prediction Panel & PIT Audit")
    print("=" * 76)

    factor_path, label_path, control_path = (
        resolve_sources()
    )

    print(f"Factor source : {factor_path}")
    print(f"Label source  : {label_path}")
    print(f"Control source: {control_path}")

    base = load_factor_base(
        factor_path
    )

    labels = load_labels(
        label_path
    )

    controls, control_cols = load_controls(
        control_path
    )

    panel, end_derived, end_mismatch = (
        build_panel(
            base,
            labels,
            controls,
        )
    )

    # --------------------------------------------------------
    # Remove accidental future columns other than target
    # --------------------------------------------------------

    other_future = [
        c for c in panel.columns
        if c.startswith("future_")
        and c != TARGET
    ]

    if other_future:
        panel = panel.drop(
            columns=other_future
        )

    panel = panel.sort_values(
        [
            "window",
            "analysis_date",
            "security_id",
        ]
    ).reset_index(drop=True)

    
    date_summary = (
        panel.groupby(
            ["window", "analysis_date"],
            as_index=False,
        )
        .agg(
            stock_n=("security_id", "size"),
            network_complete_n=(
                "network_feature_complete",
                "sum",
            ),
            target_n=(
                "target_available",
                "sum",
            ),
            label_end_date=(
                "label_end_date",
                "max",
            ),
        )
    )

    date_summary[
        "network_complete_share"
    ] = (
        date_summary["network_complete_n"]
        / date_summary["stock_n"]
    )

    date_summary[
        "target_available_share"
    ] = (
        date_summary["target_n"]
        / date_summary["stock_n"]
    )

    train_calendar = (
        build_training_calendar(panel)
    )

    # --------------------------------------------------------
    # PIT QA
    # --------------------------------------------------------

    key = [
        "security_id",
        "analysis_date",
        "window",
    ]

    target_mask = panel[
        "target_available"
    ]

    missing_end_for_target = int(
        panel.loc[
            target_mask,
            "label_end_date",
        ].isna().sum()
    )
    nonforward_target = int(
        (
            panel.loc[
                target_mask,
                "label_end_date",
            ]
            <=
            panel.loc[
                target_mask,
                "analysis_date",
            ]
        ).sum()
    )

    training_leak = int(
        (
            train_calendar[
                "max_train_label_end_date"
            ].notna()
            &
            (
                train_calendar[
                    "max_train_label_end_date"
                ]
                >
                train_calendar[
                    "forecast_date"
                ]
            )
        ).sum()
    )

    qa = {
        "factor_source_row_count":
            int(len(base)),

        "prediction_panel_row_count":
            int(len(panel)),

        "factor_universe_preserved":
            len(base) == len(panel),

        "duplicate_stock_date_window_count":
            int(panel[key].duplicated().sum()),

        "frozen_factor_count":
            len(FACTORS),

        "window_count":
            int(panel["window"].nunique()),

        "target":
            TARGET,

        "target_available_row_count":
            int(panel[TARGET].notna().sum()),

        "target_missing_row_count":
            int(panel[TARGET].isna().sum()),

        "target_used_to_define_current_universe":
            False,

        "label_end_date_derived":
            end_derived,

        "label_end_date_mismatch_count":
            int(end_mismatch),

        "target_with_missing_label_end_count":
            missing_end_for_target,

        "nonforward_target_interval_count":
            nonforward_target,

        "training_calendar_leakage_count":
            training_leak,

        "other_future_columns_in_output":
            0,
        "control_source_found":
            control_path is not None,

        "control_panel_has_window":
            bool(controls is not None and "window" in controls.columns),

        "control_join_key":
            (
                "security_id+analysis_date+window"
                if controls is not None and "window" in controls.columns
                else "security_id+analysis_date"
            ),

        "control_column_count":
            len(control_cols),

        "control_set_explicitly_confirmed":
            CONTROL_COLUMNS_OVERRIDE is not None,

        "network_features_reestimated":
            False,

        "future_ivol_reestimated":
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
    qa["all_formal_qa_pass"] = bool(
        qa["factor_universe_preserved"]
        and
        qa["duplicate_stock_date_window_count"] == 0
        and
        qa["frozen_factor_count"] == 9
        and
        qa["window_count"] == 3
        and
        qa["label_end_date_mismatch_count"] == 0
        and
        qa["target_with_missing_label_end_count"] == 0
        and
        qa["nonforward_target_interval_count"] == 0
        and
        qa["training_calendar_leakage_count"] == 0
        and
        qa["other_future_columns_in_output"] == 0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    panel.to_parquet(
        OUTDIR / "frozen_prediction_panel.parquet",
        index=False,
        compression="zstd",
    )

    save_csv(
        date_summary,
        OUTDIR / "prediction_panel_date_summary.csv",
    )
    save_csv(
        train_calendar,
        OUTDIR / "oos_training_calendar.csv",
    )

    save_csv(
        pd.DataFrame(
            {
                "control_candidate_column":
                    control_cols
            }
        ),
        OUTDIR / "control_candidate_columns.csv",
    )

    schema_df = pd.DataFrame({
        "column": panel.columns,
        "dtype": [
            str(panel[c].dtype)
            for c in panel.columns
        ],
    })

    schema_df["role"] = "preserved_current_information"
    schema_df.loc[
        schema_df["column"].isin(key),
        "role",
    ] = "key"
    schema_df.loc[
        schema_df["column"].isin(FACTORS),
        "role",
    ] = "frozen_network_feature"
    schema_df.loc[
        schema_df["column"] == TARGET,
        "role",
    ] = "future_target_not_predictor"
    schema_df.loc[
        schema_df["column"] == "label_end_date",
        "role",
    ] = "label_timing_only"

    save_csv(
        schema_df,
        OUTDIR / "prediction_panel_schema.csv",
    )
    save_csv(
        pd.DataFrame(
            [
                {
                    "qa_name": k,
                    "qa_value": v,
                }
                for k, v in qa.items()
            ]
        ),
        OUTDIR / "day11_step1_qa.csv",
    )

    manifest = pd.DataFrame([
        {
            "source_type": "factor",
            "path": str(factor_path),
        },
        {
            "source_type": "label",
            "path": str(label_path),
        },
        {
            "source_type": "control",
            "path": (
                str(control_path)
                if control_path
                else ""
            ),
        },
    ])

    save_csv(
        manifest,
        OUTDIR / "prediction_panel_source_manifest.csv",
    )
    metadata = {
        "research_day": 11,
        "step":
            "Step1_Frozen_Prediction_Panel_PIT_Audit",

        "target": TARGET,
        "frozen_factors": FACTORS,
        "windows": WINDOWS,

        "universe_rule":
            (
                "Frozen Day-5 factor universe is the left base. "
                "Future-label availability never defines the "
                "current stock universe."
            ),

        "label_definition":
            (
                "future_idio_vol_annualized over the frozen "
                "forward interval "
                "(analysis_date, next_analysis_date]."
            ),

        "oos_training_rule":
            (
                "For forecast date t, training observations "
                "must satisfy analysis_date < t and "
                "label_end_date <= t."
            ),

        "minimum_training_periods":
            MIN_TRAIN_PERIODS,

        "control_columns":
            control_cols,

        "control_set_explicitly_confirmed":
            CONTROL_COLUMNS_OVERRIDE is not None,

        "important_note":
            (
                "If CONTROL_COLUMNS_OVERRIDE is None, the PIT "
                "panel is valid, but the exact Day-7 FULL "
                "baseline control set must be confirmed before "
                "Step 3 forecasting."
            ),

        "formal_qa": qa,
    }
    with open(
        OUTDIR / "day11_step1_metadata.json",
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

    print()
    print("Formal QA:")
    for k, v in qa.items():
        print(f"  {k}: {v}")

    print()
    print(
        train_calendar.groupby("window")
        .agg(
            forecast_date_n=(
                "forecast_date",
                "size",
            ),
            eligible_24m_n=(
                "min_24_periods_met",
                "sum",
            ),
        )
        .to_string()
    )

    if not qa["all_formal_qa_pass"]:
        raise RuntimeError(
            "Day-11 Step-1 PIT QA failed."
        )

    print()
    print("=" * 76)
    print("DAY 11 STEP 1 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 76)


if __name__ == "__main__":
    main()