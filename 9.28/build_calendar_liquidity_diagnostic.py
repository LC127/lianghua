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

# Frozen convention
# Day 8  -> M1_day8
# Day 9  -> M1_day9
# Day 10 -> M1_day10
DAY8_ROOT = OUT / "M1_day8"
DAY9_ROOT = OUT / "M1_day9"
DAY10_ROOT = OUT / "M1_day10"

LIQ_DIR = (
    DAY8_ROOT
    / "01_stage1_liquidity_panel"
)

CAP_FILE = (
    DAY8_ROOT
    / "04_stage4_capacity_limits"
    / "capacity_limit_rebalance_pair.csv"
)

REGIME_DIR = (
    DAY9_ROOT
    / "02_stage2_pit_regime_classification"
)

REGIME_FILE = (
    REGIME_DIR
    / "pit_regime_panel.parquet"
)

if not REGIME_FILE.exists():
    REGIME_FILE = (
        REGIME_DIR
        / "pit_regime_panel.csv"
    )

OUTDIR = (
    DAY10_ROOT
    / "01_stage1_secular_liquidity_panel"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

MARKET_FILE = (
    OUTDIR
    / "market_liquidity_diagnostic.csv"
)

PANEL_FILE = (
    OUTDIR
    / "calendar_liquidity_network_panel.csv"
)

CORR_FILE = (
    OUTDIR
    / "calendar_liquidity_correlations.csv"
)

QA_FILE = (
    OUTDIR
    / "day10_step1_qa.csv"
)

META_FILE = (
    OUTDIR
    / "day10_step1_metadata.json"
)


# ============================================================
# 1. Helpers
# ============================================================

def save(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
    )
    os.replace(tmp, path)


def cols(path):
    if path.suffix.lower() == ".csv":
        return list(
            pd.read_csv(path, nrows=0).columns
        )
    return list(
        pq.ParquetFile(path)
        .schema_arrow.names
    )


def find_col(columns, aliases):
    lookup = {
        str(c).strip().lower(): c
        for c in columns
    }
    for x in aliases:
        if x.lower() in lookup:
            return lookup[x.lower()]
    return None


# ============================================================
# 2. Locate frozen Day-8 PIT liquidity panel
# ============================================================

DATE_ALIASES = [
    "analysis_date",
    "trade_date",
    "tradeDate",
    "date",
]

ADV_ALIASES = [
    "adv20_cny",
    "adv20",
    "adv_20_cny",
    "adv_20",
    "average_daily_value_20",
    "average_daily_turnover_value_20",
]

MV_ALIASES = [
    "market_value",
    "market_value_cny",
    "mktcap",
    "market_cap",
]


def find_liquidity_file():

    candidates = []

    for path in LIQ_DIR.rglob("*"):

        if (
            not path.is_file()
            or path.suffix.lower()
            not in {".csv", ".parquet"}
        ):
            continue

        try:
            c = cols(path)
        except Exception:
            continue

        d = find_col(c, DATE_ALIASES)
        a = find_col(c, ADV_ALIASES)

        if d is None or a is None:
            continue

        name = path.name.lower()

        score = (
            10 * ("liquidity" in name)
            + 8 * ("panel" in name)
            + 5 * ("pit" in name)
            + 3 * ("adv" in name)
        )

        candidates.append(
            (score, path)
        )

    if not candidates:
        raise RuntimeError(
            "Cannot locate Day-8 frozen PIT liquidity panel. "
            "Check LIQ_DIR or ADV column aliases."
        )

    candidates.sort(
        key=lambda x: (x[0], str(x[1])),
        reverse=True,
    )

    return candidates[0][1]


# ============================================================
# 3. Build date-level market liquidity state
# ============================================================

def load_market_liquidity():

    path = find_liquidity_file()

    print(f"Liquidity source: {path}")

    df = (
        pd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path)
    )

    c = list(df.columns)

    date_col = find_col(
        c,
        DATE_ALIASES,
    )

    adv_col = find_col(
        c,
        ADV_ALIASES,
    )

    mv_col = find_col(
        c,
        MV_ALIASES,
    )

    if date_col is None or adv_col is None:
        raise RuntimeError(
            "Date or ADV20 column not found."
        )

    df["analysis_date"] = pd.to_datetime(
        df[date_col],
        errors="raise",
    )

    df["adv20_cny"] = pd.to_numeric(
        df[adv_col],
        errors="coerce",
    )

    if (
        df["adv20_cny"].dropna() < 0
    ).any():
        raise RuntimeError(
            "Negative ADV20 found."
        )

    df["positive_adv20"] = (
        df["adv20_cny"] > 0
    )

    # Optional PIT market capitalization
    if mv_col is not None:
        df["market_value_cny"] = pd.to_numeric(
            df[mv_col],
            errors="coerce",
        )

    def agg(g):

        adv = (
            g["adv20_cny"]
            .dropna()
        )

        positive = adv[
            adv > 0
        ]

        out = {
            "stock_n":
                int(len(g)),

            "adv20_valid_n":
                int(len(adv)),

            "positive_adv20_n":
                int((adv > 0).sum()),

            "zero_adv20_share":
                float(
                    (adv <= 0).mean()
                )
                if len(adv)
                else np.nan,

            "market_adv20_total_cny":
                float(positive.sum()),

            "median_adv20_cny":
                float(positive.median())
                if len(positive)
                else np.nan,

            "p25_adv20_cny":
                float(positive.quantile(0.25))
                if len(positive)
                else np.nan,

            "p75_adv20_cny":
                float(positive.quantile(0.75))
                if len(positive)
                else np.nan,
        }

        if "market_value_cny" in g:

            mv = (
                g["market_value_cny"]
                .dropna()
            )

            out[
                "market_value_total_cny"
            ] = float(mv.sum())

            out[
                "median_market_value_cny"
            ] = (
                float(mv.median())
                if len(mv)
                else np.nan
            )

        return pd.Series(out)

    market = (
        df.groupby(
            "analysis_date",
            sort=True,
        )
        .apply(
            agg,
            include_groups=False,
        )
        .reset_index()
        .sort_values("analysis_date")
        .reset_index(drop=True)
    )

    market["time_index"] = np.arange(
        1,
        len(market) + 1,
    )

    market["calendar_year"] = (
        market[
            "analysis_date"
        ].dt.year
    )

    # Convenient reporting units
    market["market_adv20_total_yi"] = (
        market[
            "market_adv20_total_cny"
        ]
        / 1e8
    )

    market["median_adv20_wan"] = (
        market["median_adv20_cny"]
        / 1e4
    )

    return market, path, adv_col, mv_col


# ============================================================
# 4. Load frozen regimes
# ============================================================

def load_regime():

    if not REGIME_FILE.exists():
        raise FileNotFoundError(
            REGIME_FILE
        )

    r = (
        pd.read_parquet(REGIME_FILE)
        if REGIME_FILE.suffix.lower()
        == ".parquet"
        else pd.read_csv(REGIME_FILE)
    )

    needed = [
        "analysis_date",
        "window",
        "network_stress_score",
        "network_stress_regime",
        "network_reconfig_score",
        "network_reconfig_regime",
    ]

    missing = [
        x for x in needed
        if x not in r.columns
    ]

    if missing:
        raise RuntimeError(
            f"Regime file missing: {missing}"
        )

    r["analysis_date"] = pd.to_datetime(
        r["analysis_date"],
        errors="raise",
    )

    r["window"] = pd.to_numeric(
        r["window"],
        errors="raise",
    ).astype(int)

    if r[
        ["analysis_date", "window"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate regime date-window keys."
        )

    return r[needed].copy()


# ============================================================
# 5. Load frozen rebalance-level capacity
# ============================================================

def load_capacity():

    if not CAP_FILE.exists():
        raise FileNotFoundError(
            CAP_FILE
        )

    c = pd.read_csv(
        CAP_FILE
    )

    required = [
        "analysis_date",
        "window",
        "factor_name",
    ]

    for x in required:
        if x not in c.columns:
            raise RuntimeError(
                f"Capacity file missing {x}"
            )

    capacity_cols = [
        x for x in c.columns
        if (
            "capacity" in x.lower()
            and pd.api.types.is_numeric_dtype(c[x])
            and (
                "cny" in x.lower()
                or "aum" in x.lower()
                or "limit" in x.lower()
            )
        )
    ]

    preferred = [
        "capacity_limit_cny",
        "capacity_aum_cny",
        "capacity_cny",
    ]

    cap_col = next(
        (
            x for x in preferred
            if x in c.columns
        ),
        None,
    )

    if cap_col is None:

        if len(capacity_cols) != 1:
            raise RuntimeError(
                "Cannot uniquely detect capacity column: "
                f"{capacity_cols}"
            )

        cap_col = capacity_cols[0]

    print(f"Capacity source : {CAP_FILE}")
    print(f"Capacity column : {cap_col}")

    c["analysis_date"] = pd.to_datetime(
        c["analysis_date"],
        errors="raise",
    )

    c["window"] = pd.to_numeric(
        c["window"],
        errors="raise",
    ).astype(int)

    c["capacity_yi"] = (
        pd.to_numeric(
            c[cap_col],
            errors="coerce",
        )
        / 1e8
    )

    # Remove factor duplication before time diagnostics:
    # one frozen cross-factor median per date × window.
    c = (
        c.groupby(
            ["analysis_date", "window"],
            as_index=False,
        )
        .agg(
            median_pair_capacity_yi=(
                "capacity_yi",
                "median",
            ),
            mean_pair_capacity_yi=(
                "capacity_yi",
                "mean",
            ),
            capacity_factor_n=(
                "factor_name",
                "nunique",
            ),
        )
    )

    return c, cap_col


# ============================================================
# 6. Correlation diagnostics
# ============================================================

def correlations(panel):

    rows = []

    pairs = [
        (
            "network_stress_score",
            "time_index",
        ),
        (
            "network_stress_score",
            "market_adv20_total_yi",
        ),
        (
            "median_pair_capacity_yi",
            "time_index",
        ),
        (
            "median_pair_capacity_yi",
            "market_adv20_total_yi",
        ),
        (
            "network_stress_score",
            "median_pair_capacity_yi",
        ),
    ]

    for window, g in panel.groupby(
        "window"
    ):

        for x, y in pairs:

            z = g[[x, y]].dropna()

            corr = (
                z[x].corr(z[y])
                if len(z) >= 3
                else np.nan
            )

            rows.append({
                "window": int(window),
                "x": x,
                "y": y,
                "n": len(z),
                "pearson_corr": corr,
            })

    return pd.DataFrame(rows)


# ============================================================
# 7. Main
# ============================================================

def main():

    print("=" * 72)
    print("M1 - Research Day 10 - Step 1")
    print("Calendar-Time / Market-Liquidity Diagnostic")
    print("=" * 72)

    market, liq_source, adv_col, mv_col = (
        load_market_liquidity()
    )

    regime = load_regime()
    capacity, cap_col = load_capacity()

    # Only dates actually used by Day-9 network states.
    analysis_dates = (
        regime[
            ["analysis_date"]
        ]
        .drop_duplicates()
    )

    market_analysis = (
        analysis_dates.merge(
            market,
            on="analysis_date",
            how="left",
            validate="one_to_one",
        )
    )

    panel = (
        regime.merge(
            market,
            on="analysis_date",
            how="left",
            validate="many_to_one",
        )
        .merge(
            capacity,
            on=[
                "analysis_date",
                "window",
            ],
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(drop=True)
    )

    corr = correlations(
        panel
    )

    save(
        market_analysis,
        MARKET_FILE,
    )

    save(
        panel,
        PANEL_FILE,
    )

    save(
        corr,
        CORR_FILE,
    )

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    market_missing = int(
        market_analysis[
            "market_adv20_total_cny"
        ].isna().sum()
    )

    capacity_missing = int(
        panel[
            "median_pair_capacity_yi"
        ].isna().sum()
    )

    qa = {
        "liquidity_source_exists":
            True,

        "regime_source_exists":
            True,

        "capacity_source_exists":
            True,

        "analysis_date_count":
            int(
                market_analysis[
                    "analysis_date"
                ].nunique()
            ),

        "panel_row_count":
            int(len(panel)),

        "panel_duplicate_date_window_count":
            int(
                panel[
                    [
                        "analysis_date",
                        "window",
                    ]
                ]
                .duplicated()
                .sum()
            ),

        "market_liquidity_missing_date_count":
            market_missing,

        "capacity_missing_panel_row_count":
            capacity_missing,

        "negative_market_adv20_count":
            int(
                (
                    market[
                        "market_adv20_total_cny"
                    ]
                    < 0
                ).sum()
            ),

        "adv_reestimated":
            False,

        "capacity_reestimated":
            False,

        "regime_reestimated":
            False,

        "future_information_used":
            False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa[
            "panel_duplicate_date_window_count"
        ] == 0
        and
        qa[
            "market_liquidity_missing_date_count"
        ] == 0
        and
        qa[
            "negative_market_adv20_count"
        ] == 0
    )

    save(
        pd.DataFrame(
            [
                {
                    "qa_name": k,
                    "qa_value": v,
                }
                for k, v in qa.items()
            ]
        ),
        QA_FILE,
    )

    metadata = {
        "research_day": 10,
        "step": (
            "Step1_Calendar_Time_"
            "Market_Liquidity_Diagnostic"
        ),
        "liquidity_source":
            str(liq_source),
        "adv_source_column":
            adv_col,
        "market_value_source_column":
            mv_col,
        "capacity_source":
            str(CAP_FILE),
        "capacity_source_column":
            cap_col,
        "regime_source":
            str(REGIME_FILE),
        "capacity_cross_factor_aggregation":
            "median and mean within date-window",
        "interpretation": (
            "Diagnostic only. No causal inference "
            "and no re-estimation of ADV, regime, "
            "or strategy capacity."
        ),
        "formal_qa": qa,
    }

    with open(
        META_FILE,
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
    print("Diagnostic correlations:")
    print(
        corr.to_string(
            index=False
        )
    )

    if not qa[
        "all_formal_qa_pass"
    ]:
        raise RuntimeError(
            "Day 10 Step 1 formal QA failed."
        )

    print()
    print("=" * 72)
    print("DAY 10 STEP 1 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()