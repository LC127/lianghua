from pathlib import Path
import json
import os

import numpy as np
import pandas as pd


# ============================================================
# M1 - Day 13 - Step 2
# Repair Compact Final Evidence Matrix Display Convention
#
# IMPORTANT:
# 1. The full evidence matrix remains the canonical frozen source.
# 2. No risk / OOS / alpha / turnover / cost / capacity result
#    is re-estimated.
# 3. No factor, window, direction, or cost level is selected
#    from outcomes.
# 4. This script only repairs the compact reporting schema.
# ============================================================


# ============================================================
# 1. Paths
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")

DAY13 = ROOT / "output" / "M1_day13"

# Read the existing Day-13 Step-2 canonical artifacts and keep
# all repaired reporting artifacts in that same directory.
STEP2_DIR = (
    DAY13
    / "02_step2_frozen_9factor_evidence"
)

FULL_MATRIX = (
    STEP2_DIR
    / "m1_frozen_9factor_final_evidence_matrix.csv"
)

COMPACT_MATRIX = (
    STEP2_DIR
    / "m1_frozen_9factor_final_evidence_matrix_compact.csv"
)

QA_FILE = (
    STEP2_DIR
    / "day13_step2_qa.csv"
)

META_FILE = (
    STEP2_DIR
    / "day13_step2_metadata.json"
)


# ============================================================
# 2. Frozen reporting conventions
# ============================================================

EXPECTED_FACTOR_COUNT = 9
EXPECTED_WINDOWS = [60, 120, 252]
EXPECTED_ROW_COUNT = 27

REPORTING_COST_BPS = 20

ALL_NETWORK_OOS_LEVEL = (
    "WINDOW_LEVEL_SHARED_ACROSS_ALL_9_FACTORS"
)

BLOCK_OOS_LEVEL = (
    "BLOCK_CONDITIONAL_SHARED_WITHIN_BLOCK"
)

DIRECTION_POLICY = (
    "BOTH_DIRECTIONS_REPORTED_NO_SELECTION"
)

COMPACT_SCHEMA_VERSION = 2


# ============================================================
# 3. Helper functions
# ============================================================

def save_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """
    Atomically save CSV using UTF-8-SIG.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def save_json_atomic(obj: dict, path: Path) -> None:
    """
    Atomically save JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(str(path) + ".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(tmp, path)


def as_bool(x) -> bool:
    return str(x).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    source_name: str,
) -> None:
    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{source_name} is missing required columns:\n"
            + "\n".join(missing)
        )


def max_abs_diff(a, b) -> float:
    """
    Maximum absolute difference ignoring paired NaNs.
    """
    x = pd.to_numeric(a, errors="coerce").to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(float)

    mask = np.isfinite(x) & np.isfinite(y)

    if not mask.any():
        return 0.0

    return float(
        np.max(
            np.abs(x[mask] - y[mask])
        )
    )


def update_qa(
    qa: pd.DataFrame,
    updates: dict,
) -> pd.DataFrame:
    """
    Update existing QA rows and append new ones.
    Keeps one row per qa_name.
    """
    if not {
        "qa_name",
        "qa_value",
    }.issubset(qa.columns):
        raise RuntimeError(
            "QA file must contain "
            "'qa_name' and 'qa_value'."
        )

    q = qa.copy()

    existing = set(
        q["qa_name"].astype(str)
    )

    for name, value in updates.items():

        if name in existing:
            q.loc[
                q["qa_name"] == name,
                "qa_value",
            ] = value

        else:
            q = pd.concat(
                [
                    q,
                    pd.DataFrame(
                        [{
                            "qa_name": name,
                            "qa_value": value,
                        }]
                    ),
                ],
                ignore_index=True,
            )

    q = (
        q
        .drop_duplicates(
            subset=["qa_name"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return q


# ============================================================
# 4. Load frozen full matrix
# ============================================================

def load_and_validate_full_matrix() -> pd.DataFrame:

    if not FULL_MATRIX.exists():
        raise FileNotFoundError(
            f"Full evidence matrix not found:\n"
            f"{FULL_MATRIX}"
        )

    full = pd.read_csv(
        FULL_MATRIX,
        low_memory=False,
    )

    required = [
        # identity
        "factor_order",
        "factor_name",
        "risk_block",
        "window",

        # risk
        "risk_neutral_ivol_ic",
        "risk_neutral_ivol_hac_t",
        "risk_same_sign_year_share",

        # all-network OOS
        "oos_all_network_delta_ic",
        "oos_all_network_hac_t",
        "oos_all_network_positive_month_share",
        "oos_all_network_positive_year_share",

        # block OOS
        "risk_block_factor_n",
        "oos_block_marginal_ic",
        "oos_block_hac_t",
        "oos_block_positive_month_share",
        "oos_block_positive_year_share",
        "oos_evidence_level",

        # alpha information
        "raw_mean_ic",
        "neutral_mean_ic",

        # gross portfolio
        "raw_mean_high_minus_low",
        "neutral_mean_high_minus_low",

        # implementation
        "raw_mean_one_way_trade_notional",
        "neutral_mean_one_way_trade_notional",

        # break-even cost
        "raw_break_even_bps_hl",
        "raw_break_even_bps_lh",
        "neutral_break_even_bps_hl",
        "neutral_break_even_bps_lh",

        # 20bps net
        "raw_20bps_mean_net_hl",
        "raw_20bps_mean_net_lh",
        "neutral_20bps_mean_net_hl",
        "neutral_20bps_mean_net_lh",

        # capacity
        "raw_strategy_capacity_95pct_dates_cny",
        "neutral_strategy_capacity_95pct_dates_cny",
    ]

    require_columns(
        full,
        required,
        "Full evidence matrix",
    )

    # --------------------------------------------------------
    # Frozen structure QA
    # --------------------------------------------------------

    if len(full) != EXPECTED_ROW_COUNT:
        raise RuntimeError(
            f"Unexpected full-matrix row count: "
            f"{len(full)} != {EXPECTED_ROW_COUNT}"
        )

    if (
        full[
            ["factor_name", "window"]
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate factor-window rows "
            "in full evidence matrix."
        )

    factor_count = (
        full["factor_name"]
        .nunique()
    )

    if factor_count != EXPECTED_FACTOR_COUNT:
        raise RuntimeError(
            f"Unexpected factor count: "
            f"{factor_count}"
        )

    windows = sorted(
        pd.to_numeric(
            full["window"],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    if windows != EXPECTED_WINDOWS:
        raise RuntimeError(
            f"Unexpected windows: {windows}"
        )

    # Every factor must have exactly three windows.
    counts = (
        full
        .groupby("factor_name")["window"]
        .nunique()
    )

    if not (
        counts == len(EXPECTED_WINDOWS)
    ).all():
        raise RuntimeError(
            "Not every factor has all "
            "three frozen windows."
        )

    return full


# ============================================================
# 5. Build repaired compact matrix
# ============================================================

def build_compact(
    full: pd.DataFrame,
) -> pd.DataFrame:

    x = full.copy()

    # --------------------------------------------------------
    # 5.1 Explicit OOS evidence hierarchy
    # --------------------------------------------------------

    x[
        "oos_all_network_evidence_level"
    ] = ALL_NETWORK_OOS_LEVEL

    # Full matrix already contains the correct block-level label.
    x[
        "oos_block_evidence_level"
    ] = x["oos_evidence_level"].astype(str)

    # Sanity check:
    # all rows should indicate block-shared conditional evidence.
    allowed_block_levels = {
        BLOCK_OOS_LEVEL,
    }

    observed = set(
        x[
            "oos_block_evidence_level"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    if not observed.issubset(
        allowed_block_levels
    ):
        raise RuntimeError(
            "Unexpected oos_evidence_level values:\n"
            f"{sorted(observed)}"
        )

    # --------------------------------------------------------
    # 5.2 Explicit reporting convention
    # --------------------------------------------------------

    x["reporting_cost_bps"] = (
        REPORTING_COST_BPS
    )

    x["portfolio_direction_policy"] = (
        DIRECTION_POLICY
    )

    # --------------------------------------------------------
    # 5.3 Gross return direction symmetry
    #
    # Full matrix stores H-L gross return.
    # L-H is a deterministic sign reversal only.
    # This is NOT outcome-based direction selection.
    # --------------------------------------------------------

    x[
        "raw_mean_low_minus_high"
    ] = -pd.to_numeric(
        x[
            "raw_mean_high_minus_low"
        ],
        errors="raise",
    )

    x[
        "neutral_mean_low_minus_high"
    ] = -pd.to_numeric(
        x[
            "neutral_mean_high_minus_low"
        ],
        errors="raise",
    )

    # --------------------------------------------------------
    # 5.4 Compact display schema
    # --------------------------------------------------------

    compact_columns = [

        # ====================================================
        # A. Frozen identity / grouping
        # ====================================================
        "factor_order",
        "factor_name",
        "risk_block",
        "risk_block_factor_n",
        "window",

        # ====================================================
        # B. Risk evidence
        # ====================================================
        "risk_neutral_ivol_ic",
        "risk_neutral_ivol_hac_t",
        "risk_same_sign_year_share",

        # ====================================================
        # C. Strict OOS: all-network evidence
        #
        # Same within a window across all nine factor rows.
        # Must NOT be interpreted as factor-specific OOS IC.
        # ====================================================
        "oos_all_network_delta_ic",
        "oos_all_network_hac_t",
        "oos_all_network_positive_month_share",
        "oos_all_network_positive_year_share",
        "oos_all_network_evidence_level",

        # ====================================================
        # D. Strict OOS: conditional LOBO block evidence
        #
        # Same within factors belonging to the same frozen block.
        # Must NOT be interpreted as factor-specific marginal IC.
        # ====================================================
        "oos_block_marginal_ic",
        "oos_block_hac_t",
        "oos_block_positive_month_share",
        "oos_block_positive_year_share",
        "oos_block_evidence_level",

        # ====================================================
        # E. Alpha information
        # ====================================================
        "raw_mean_ic",
        "neutral_mean_ic",

        # ====================================================
        # F. Gross portfolio return
        #
        # Show H-L and L-H together.
        # ====================================================
        "raw_mean_high_minus_low",
        "raw_mean_low_minus_high",

        "neutral_mean_high_minus_low",
        "neutral_mean_low_minus_high",

        # ====================================================
        # G. Trading intensity
        # ====================================================
        "raw_mean_one_way_trade_notional",
        "neutral_mean_one_way_trade_notional",

        # ====================================================
        # H. Break-even cost
        #
        # Both directions are retained.
        # No outcome-based direction choice.
        # ====================================================
        "raw_break_even_bps_hl",
        "raw_break_even_bps_lh",

        "neutral_break_even_bps_hl",
        "neutral_break_even_bps_lh",

        # ====================================================
        # I. 20 bps post-cost return
        #
        # Both directions are retained.
        # ====================================================
        "raw_20bps_mean_net_hl",
        "raw_20bps_mean_net_lh",

        "neutral_20bps_mean_net_hl",
        "neutral_20bps_mean_net_lh",

        # ====================================================
        # J. Frozen 95%-dates capacity
        # ====================================================
        "raw_strategy_capacity_95pct_dates_cny",
        "neutral_strategy_capacity_95pct_dates_cny",

        # ====================================================
        # K. Reporting conventions
        # ====================================================
        "reporting_cost_bps",
        "portfolio_direction_policy",
    ]

    compact = (
        x[compact_columns]
        .sort_values(
            [
                "factor_order",
                "window",
            ]
        )
        .reset_index(drop=True)
    )

    return compact


# ============================================================
# 6. Compact-specific QA
# ============================================================

def compact_qa(
    full: pd.DataFrame,
    compact: pd.DataFrame,
) -> dict:

    key = [
        "factor_name",
        "window",
    ]

    # --------------------------------------------------------
    # Basic structure
    # --------------------------------------------------------

    row_count_ok = (
        len(compact)
        == EXPECTED_ROW_COUNT
    )

    duplicate_count = int(
        compact[key]
        .duplicated()
        .sum()
    )

    missing_count = int(
        compact.isna()
        .sum()
        .sum()
    )

    # --------------------------------------------------------
    # Deterministic L-H reconstruction QA
    # --------------------------------------------------------

    raw_lh_diff = max_abs_diff(
        compact[
            "raw_mean_low_minus_high"
        ],
        -compact[
            "raw_mean_high_minus_low"
        ],
    )

    neutral_lh_diff = max_abs_diff(
        compact[
            "neutral_mean_low_minus_high"
        ],
        -compact[
            "neutral_mean_high_minus_low"
        ],
    )

    # --------------------------------------------------------
    # Break-even directional symmetry
    #
    # Existing full matrix stores both HL and LH.
    # They should be sign reversals.
    # --------------------------------------------------------

    raw_be_symmetry_diff = max_abs_diff(
        compact[
            "raw_break_even_bps_lh"
        ],
        -compact[
            "raw_break_even_bps_hl"
        ],
    )

    neutral_be_symmetry_diff = max_abs_diff(
        compact[
            "neutral_break_even_bps_lh"
        ],
        -compact[
            "neutral_break_even_bps_hl"
        ],
    )

    # --------------------------------------------------------
    # Check that compact numerical source values are reproduced
    # exactly from full matrix.
    # --------------------------------------------------------

    source_cols = [
        "risk_neutral_ivol_ic",
        "risk_neutral_ivol_hac_t",
        "risk_same_sign_year_share",

        "oos_all_network_delta_ic",
        "oos_all_network_hac_t",
        "oos_all_network_positive_month_share",
        "oos_all_network_positive_year_share",

        "risk_block_factor_n",

        "oos_block_marginal_ic",
        "oos_block_hac_t",
        "oos_block_positive_month_share",
        "oos_block_positive_year_share",

        "raw_mean_ic",
        "neutral_mean_ic",

        "raw_mean_high_minus_low",
        "neutral_mean_high_minus_low",

        "raw_mean_one_way_trade_notional",
        "neutral_mean_one_way_trade_notional",

        "raw_break_even_bps_hl",
        "raw_break_even_bps_lh",
        "neutral_break_even_bps_hl",
        "neutral_break_even_bps_lh",

        "raw_20bps_mean_net_hl",
        "raw_20bps_mean_net_lh",
        "neutral_20bps_mean_net_hl",
        "neutral_20bps_mean_net_lh",

        "raw_strategy_capacity_95pct_dates_cny",
        "neutral_strategy_capacity_95pct_dates_cny",
    ]

    f = (
        full[
            key + source_cols
        ]
        .sort_values(key)
        .reset_index(drop=True)
    )

    c = (
        compact[
            key + source_cols
        ]
        .sort_values(key)
        .reset_index(drop=True)
    )

    reproduction_diffs = {}

    for col in source_cols:

        if pd.api.types.is_numeric_dtype(
            f[col]
        ):
            diff = max_abs_diff(
                f[col],
                c[col],
            )

        else:
            diff = float(
                not (
                    f[col]
                    .astype(str)
                    .equals(
                        c[col]
                        .astype(str)
                    )
                )
            )

        reproduction_diffs[col] = diff

    compact_source_max_abs_diff = (
        max(
            reproduction_diffs.values()
        )
        if reproduction_diffs
        else 0.0
    )

    # --------------------------------------------------------
    # OOS level QA
    # --------------------------------------------------------

    all_network_level_ok = bool(
        (
            compact[
                "oos_all_network_evidence_level"
            ]
            == ALL_NETWORK_OOS_LEVEL
        ).all()
    )

    block_level_ok = bool(
        (
            compact[
                "oos_block_evidence_level"
            ]
            == BLOCK_OOS_LEVEL
        ).all()
    )

    direction_policy_ok = bool(
        (
            compact[
                "portfolio_direction_policy"
            ]
            == DIRECTION_POLICY
        ).all()
    )

    cost_level_ok = bool(
        (
            compact[
                "reporting_cost_bps"
            ]
            == REPORTING_COST_BPS
        ).all()
    )

    # --------------------------------------------------------
    # Formal compact display QA
    # --------------------------------------------------------

    all_compact_qa_pass = bool(
        row_count_ok
        and duplicate_count == 0
        and missing_count == 0

        and raw_lh_diff <= 1e-12
        and neutral_lh_diff <= 1e-12

        and raw_be_symmetry_diff <= 1e-10
        and neutral_be_symmetry_diff <= 1e-10

        and compact_source_max_abs_diff <= 1e-12

        and all_network_level_ok
        and block_level_ok
        and direction_policy_ok
        and cost_level_ok
    )

    return {
        "compact_schema_version":
            COMPACT_SCHEMA_VERSION,

        "compact_display_only_change":
            True,

        "compact_row_count":
            len(compact),

        "compact_expected_row_count":
            EXPECTED_ROW_COUNT,

        "compact_duplicate_factor_window_count":
            duplicate_count,

        "compact_missing_count":
            missing_count,

        "compact_all_network_oos_level_explicit":
            all_network_level_ok,

        "compact_block_oos_level_explicit":
            block_level_ok,

        "compact_risk_block_factor_n_included":
            "risk_block_factor_n"
            in compact.columns,

        "compact_direction_policy":
            DIRECTION_POLICY,

        "compact_both_directions_reported":
            True,

        "compact_direction_selected_from_results":
            False,

        "compact_reporting_cost_bps":
            REPORTING_COST_BPS,

        "compact_cost_level_selected_from_results":
            False,

        "compact_raw_lh_return_derivation_max_abs_diff":
            raw_lh_diff,

        "compact_neutral_lh_return_derivation_max_abs_diff":
            neutral_lh_diff,

        "compact_raw_break_even_direction_symmetry_max_abs_diff":
            raw_be_symmetry_diff,

        "compact_neutral_break_even_direction_symmetry_max_abs_diff":
            neutral_be_symmetry_diff,

        "compact_source_reproduction_max_abs_diff":
            compact_source_max_abs_diff,

        "compact_risk_or_alpha_reestimated":
            False,

        "compact_all_formal_qa_pass":
            all_compact_qa_pass,
    }


# ============================================================
# 7. Update Step-2 QA
# ============================================================

def update_step2_qa(
    compact_checks: dict,
) -> None:

    if not QA_FILE.exists():
        raise FileNotFoundError(
            f"Day13 Step2 QA file not found:\n"
            f"{QA_FILE}"
        )

    qa = pd.read_csv(
        QA_FILE,
        low_memory=False,
    )

    qa_map = dict(
        zip(
            qa["qa_name"],
            qa["qa_value"],
        )
    )

    # Existing research-level QA must already have passed.
    original_formal_qa = as_bool(
        qa_map.get(
            "all_formal_qa_pass",
            False,
        )
    )

    if not original_formal_qa:
        raise RuntimeError(
            "Original Day13 Step2 formal QA "
            "has not passed. Compact display "
            "should not be repaired on top of "
            "a failed research matrix."
        )

    updated = update_qa(
        qa,
        compact_checks,
    )

    # Main research-level QA remains unchanged.
    # Add an integrated display QA indicator separately.
    updated = update_qa(
        updated,
        {
            "compact_and_research_all_qa_pass":
                bool(
                    original_formal_qa
                    and compact_checks[
                        "compact_all_formal_qa_pass"
                    ]
                )
        },
    )

    save_csv_atomic(
        updated,
        QA_FILE,
    )


# ============================================================
# 8. Update metadata
# ============================================================

def update_metadata() -> None:

    if not META_FILE.exists():
        raise FileNotFoundError(
            f"Day13 Step2 metadata not found:\n"
            f"{META_FILE}"
        )

    with open(
        META_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        meta = json.load(f)

    # --------------------------------------------------------
    # Preserve original frozen research definition.
    # Add display-schema information only.
    # --------------------------------------------------------

    meta[
        "compact_display_schema"
    ] = {
        "version":
            COMPACT_SCHEMA_VERSION,

        "purpose":
            (
                "Reporting-only compact view of "
                "the frozen full evidence matrix. "
                "No research result is re-estimated "
                "or selected."
            ),

        "canonical_source":
            str(FULL_MATRIX),

        "output":
            str(COMPACT_MATRIX),

        "oos_all_network_interpretation":
            (
                "Window-level evidence shared by "
                "all nine factors. It is not "
                "factor-specific OOS evidence."
            ),

        "oos_all_network_evidence_level":
            ALL_NETWORK_OOS_LEVEL,

        "oos_block_interpretation":
            (
                "Conditional leave-one-block-out "
                "evidence shared by factors belonging "
                "to the same pre-frozen risk block. "
                "It is not factor-specific OOS IC "
                "and block contributions are not additive."
            ),

        "oos_block_evidence_level":
            BLOCK_OOS_LEVEL,

        "risk_block_factor_n_included":
            True,

        "portfolio_direction_policy":
            DIRECTION_POLICY,

        "direction_selected_from_results":
            False,

        "gross_return_display":
            [
                "HIGH_MINUS_LOW",
                "LOW_MINUS_HIGH",
            ],

        "break_even_cost_display":
            [
                "HIGH_MINUS_LOW",
                "LOW_MINUS_HIGH",
            ],

        "post_cost_return_display":
            [
                "HIGH_MINUS_LOW",
                "LOW_MINUS_HIGH",
            ],

        "low_minus_high_gross_return_rule":
            (
                "LOW_MINUS_HIGH is defined mechanically "
                "as negative HIGH_MINUS_LOW."
            ),

        "reporting_cost_bps":
            REPORTING_COST_BPS,

        "cost_level_selected_from_results":
            False,

        "research_result_reestimated":
            False,
    }

    save_json_atomic(
        meta,
        META_FILE,
    )


# ============================================================
# 9. Main
# ============================================================

def main():

    print("=" * 88)
    print(
        "M1 - Day 13 - Step 2 "
        "Compact Evidence Matrix Display Repair"
    )
    print("=" * 88)

    # --------------------------------------------------------
    # 1. Load canonical frozen full matrix
    # --------------------------------------------------------

    full = load_and_validate_full_matrix()

    print(
        f"[OK] Loaded canonical full matrix: "
        f"{len(full)} rows"
    )

    # --------------------------------------------------------
    # 2. Build repaired compact matrix
    # --------------------------------------------------------

    compact = build_compact(full)

    # --------------------------------------------------------
    # 3. Compact-specific QA
    # --------------------------------------------------------

    checks = compact_qa(
        full,
        compact,
    )

    if not checks[
        "compact_all_formal_qa_pass"
    ]:
        print("\nCompact QA details:")
        for k, v in checks.items():
            print(f"  {k}: {v}")

        raise RuntimeError(
            "Compact display QA failed. "
            "No output will be overwritten."
        )

    # --------------------------------------------------------
    # 4. Backup old compact file
    # --------------------------------------------------------

    backup = (
        STEP2_DIR
        / "m1_frozen_9factor_final_evidence_matrix_compact_before_display_fix.csv"
    )

    if COMPACT_MATRIX.exists():

        old = pd.read_csv(
            COMPACT_MATRIX,
            low_memory=False,
        )

        save_csv_atomic(
            old,
            backup,
        )

        print(
            f"[OK] Previous compact matrix backed up:\n"
            f"     {backup}"
        )

    # --------------------------------------------------------
    # 5. Save repaired compact
    # --------------------------------------------------------

    save_csv_atomic(
        compact,
        COMPACT_MATRIX,
    )

    print(
        f"[OK] Repaired compact matrix saved:\n"
        f"     {COMPACT_MATRIX}"
    )

    # --------------------------------------------------------
    # 6. Update QA
    # --------------------------------------------------------

    update_step2_qa(
        checks,
    )

    print(
        f"[OK] QA updated:\n"
        f"     {QA_FILE}"
    )

    # --------------------------------------------------------
    # 7. Update metadata
    # --------------------------------------------------------

    update_metadata()

    print(
        f"[OK] Metadata updated:\n"
        f"     {META_FILE}"
    )

    # --------------------------------------------------------
    # 8. Final console summary
    # --------------------------------------------------------

    print("\n" + "=" * 88)
    print("FINAL COMPACT DISPLAY QA")
    print("=" * 88)

    for k, v in checks.items():
        print(f"{k}: {v}")

    print("\nInterpretation rules:")
    print(
        "1. oos_all_network_delta_ic "
        "= window-level shared evidence."
    )
    print(
        "2. oos_block_marginal_ic "
        "= conditional block-level shared evidence."
    )
    print(
        "3. Neither OOS statistic is "
        "factor-specific."
    )
    print(
        "4. H-L and L-H are both displayed; "
        "no direction is selected from outcomes."
    )
    print(
        "5. Reporting cost remains frozen "
        f"at {REPORTING_COST_BPS} bps."
    )
    print(
        "6. Full matrix remains the canonical "
        "research evidence source."
    )

    print("\n" + "=" * 88)
    print("COMPACT DISPLAY REPAIR: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
