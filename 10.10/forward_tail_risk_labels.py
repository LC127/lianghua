"""Day14 Step2: frozen M4 forward labels; no Risk-State outcomes used."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
import pandas as pd

MARKETS = ("CSI300", "CSI500", "SSE_COMPOSITE")
TARGETS = {
    "T1": "fwd_20d_max_drawdown_loss",
    "T2": "fwd_5d_max_drawdown_loss",
    "T3": "fwd_60d_max_drawdown_loss",
    "T4": "fwd_20d_realized_vol_annualized",
    "T5": "fwd_20d_worst_daily_loss",
    "T6": "fwd_20d_downside_semideviation_annualized",
    "T7": "fwd_20d_tail_event_p90",
    "T8": "fwd_20d_tail_event_p95",
}
MIN_CAL = 252  # At least one year of fully realized daily calibration labels.


def check(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_dates(values):
    s = values.astype(str).str.strip()
    fmt = "%Y%m%d" if s.str.fullmatch(r"\d{8}").all() else None
    d = pd.to_datetime(s, format=fmt, errors="raise")
    check(d.notna().all(), "Invalid trading dates")
    return d.dt.normalize()


def load_market(path, market):
    x = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    names = {str(c).lower().strip(): c for c in x.columns}
    dc = next((names[k] for k in ("trade_date", "date", "datetime") if k in names), None)
    pc = next((names[k] for k in ("close", "close_price", "index_close") if k in names), None)
    check(dc is not None and pc is not None,
          f"{market}: requires date/trade_date and close columns: {path}")
    v = pd.to_numeric(x[pc], errors="coerce")
    d = parse_dates(x[dc])
    check(len(x) > 0 and not d.duplicated().any()
          and np.isfinite(v).all() and (v > 0).all(),
          f"{market}: duplicate dates or invalid index close")
    out = pd.DataFrame({"date": d, "close": v.astype(float)}).sort_values("date")
    return out.set_index("date")


def mdd(prices):
    return float(np.max(1.0 - prices / np.maximum.accumulate(prices)))


def measure(close, i, h):
    if i + h >= len(close):
        return np.nan
    p = close[i:i+h+1]
    return mdd(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    ap.add_argument("--market-dir", default=None,
                    help="Exact directory containing CSI300/CSI500/SSE_COMPOSITE CSV or Parquet")
    args = ap.parse_args()
    root = Path(args.root)
    market_dir = Path(args.market_dir) if args.market_dir else root / "data" / "m4_market_indexes"
    d13 = root / "output" / "M1_day13"
    d14 = root / "output" / "M1_day14"
    s1 = d14 / "01_step1_m4_tail_risk_target_specification_freeze"
    out = d14 / "02_step2_m4_forward_tail_risk_labels"
    out.mkdir(parents=True, exist_ok=True)
    labels_file = out / "m4_forward_tail_risk_labels.csv"
    cal_file = out / "m4_tail_event_calibration.csv"
    qa_file = out / "day14_step2_qa.csv"
    meta_file = out / "day14_step2_metadata.json"

    freeze = load_json(d13 / "06_step6_m1_v1_final_freeze" / "m1_v1_0_freeze_metadata.json")
    check(freeze.get("freeze_version") == "M1_v1.0"
          and freeze.get("status") == "FROZEN"
          and freeze.get("formal_qa", {}).get("all_formal_qa_pass") is True,
          "M1 v1.0 not frozen/QA failed")
    interface = Path(freeze["canonical_m1_to_m4_interface"])
    manifest = pd.read_csv(d13 / "06_step6_m1_v1_final_freeze" /
                           "m1_v1_0_final_freeze_manifest.csv", dtype=str)
    match = manifest.loc[manifest["relative_path"].str.replace("\\", "/", regex=False) == interface.relative_to(root).as_posix(), "sha256"]
    check(len(match) == 1 and digest(interface) == match.iloc[0], "M1 interface hash mismatch")
    dates = parse_dates(pd.read_csv(interface, usecols=["analysis_date"])["analysis_date"])
    check(len(dates) > 0 and not dates.duplicated().any(), "Invalid M1 analysis dates")
    first = dates.min()

    step1 = load_json(s1 / "day14_step1_metadata.json")
    sq = pd.read_csv(s1 / "day14_step1_qa.csv")
    sd = dict(zip(sq.qa_name, sq.qa_value.astype(str).str.lower()))
    spec_file = s1 / "m4_tail_risk_target_specification.csv"
    spec = pd.read_csv(spec_file)
    check(step1.get("status") == "FROZEN"
          and step1.get("m1_version") == "M1_v1.0"
          and step1.get("primary_market_proxy") == "CSI300"
          and step1.get("robustness_market_proxies") == ["CSI500", "SSE_COMPOSITE"]
          and step1.get("primary_target") == TARGETS["T1"]
          and step1.get("formal_qa", {}).get("all_formal_qa_pass") is True
          and sd.get("all_formal_qa_pass") == "true"
          and set(spec.target_id) == set(TARGETS)
          and dict(zip(spec.target_id, spec.target_name)) == TARGETS
          and str(first.date()) == step1["formal_qa"]["first_m1_state_date"],
          "Frozen Day14 Step1 specification mismatch")
    check(all(int(row.horizon_td) == int(TARGETS[row.target_id].split("_")[1][:-1])
              for row in spec.itertuples()), "Frozen horizons changed")
    check("exceeds" in str(spec.set_index("target_id").loc["T7", "definition"])
          and "exceeds" in str(spec.set_index("target_id").loc["T8", "definition"]),
          "Event comparator is not the frozen strict '>' rule")

    sources = {}
    for m in MARKETS:
        options = [market_dir / f"{m}{ext}" for ext in (".csv", ".parquet")]
        found = [p for p in options if p.is_file()]
        check(len(found) == 1, f"{m}: provide exactly one {m}.csv or {m}.parquet in {market_dir}")
        sources[m] = found[0]
    inputs = {str(p): digest(p) for p in
              [interface, spec_file, s1 / "day14_step1_metadata.json", *sources.values()]}

    # A second run checks the locked inputs and outputs; it never rewrites the freeze.
    files = (labels_file, cal_file, qa_file, meta_file)
    if any(p.exists() for p in files):
        check(all(p.exists() for p in files), "Incomplete prior freeze: manual audit required")
        old = load_json(meta_file)
        check(old.get("input_sha256") == inputs
              and old.get("output_sha256") == {p.name: digest(p) for p in files[:-1]}
              and old.get("formal_qa", {}).get("all_formal_qa_pass") is True,
              "Frozen Day14 Step2 inputs/outputs drifted; do not overwrite")
        print("Day14 Step2: existing freeze verified, unchanged")
        return

    rows, calibrations = [], []
    for market, path in sources.items():
        price = load_market(path, market)
        idx = price.index
        check((idx.get_indexer(dates) >= 0).all(),
              f"{market}: M1 formation dates missing; do not forward-fill")
        close = price.close.to_numpy()
        # Calibration may use only label windows fully realized BEFORE first M1 date.
        train = [(measure(close, i, 20), idx[i + 20])
                 for i in range(len(idx) - 20) if idx[i + 20] < first]
        check(len(train) >= MIN_CAL, f"{market}: fewer than {MIN_CAL} pre-M1 calibration labels")
        cal = np.array([a for a, _ in train])
        q90, q95 = np.quantile(cal, [0.90, 0.95], method="linear")
        calibrations.append({"market_proxy": market, "calibration_n": len(cal),
                             "last_label_end_date": max(end for _, end in train),
                             "first_m1_state_date": first, "q90": q90, "q95": q95})
        for t in dates:
            i = int(idx.get_loc(t))
            r = {"market_proxy": market, "analysis_date": t}
            for h in (5, 20, 60):
                valid = i + h < len(idx)
                r[f"label_end_{h}d"] = idx[i + h] if valid else pd.NaT
                r[f"available_{h}d"] = valid
                r[f"mdd_{h}d"] = measure(close, i, h)
            r[TARGETS["T1"]] = r.pop("mdd_20d")
            r[TARGETS["T2"]] = r.pop("mdd_5d")
            r[TARGETS["T3"]] = r.pop("mdd_60d")
            if r["available_20d"]:
                ret = close[i+1:i+21] / close[i:i+20] - 1.0
                r[TARGETS["T4"]] = float(np.std(ret, ddof=0) * np.sqrt(252))
                r[TARGETS["T5"]] = float(-np.min(ret))  # Frozen: may be negative.
                r[TARGETS["T6"]] = float(np.sqrt(252 * np.mean(np.minimum(ret, 0) ** 2)))
                dd = r[TARGETS["T1"]]
                r[TARGETS["T7"]] = int(dd > q90)  # Frozen: strict exceedance.
                r[TARGETS["T8"]] = int(dd > q95)
            else:
                for k in ("T4", "T5", "T6", "T7", "T8"):
                    r[TARGETS[k]] = np.nan
            rows.append(r)

    panel = pd.DataFrame(rows)
    cal_table = pd.DataFrame(calibrations)
    end_cols = [f"label_end_{h}d" for h in (5, 20, 60)]
    valid_dates = all(((panel.loc[panel[c].notna(), c] >
                        panel.loc[panel[c].notna(), "analysis_date"]).all()) for c in end_cols)
    nested_events = ((panel[TARGETS["T8"]] <= panel[TARGETS["T7"]]) |
                     panel[TARGETS["T8"]].isna()).all()
    complete = panel["available_60d"]
    ordered_mdd = ((panel.loc[complete, TARGETS["T3"]] + 1e-12 >=
                    panel.loc[complete, TARGETS["T1"]]) &
                   (panel.loc[complete, TARGETS["T1"]] + 1e-12 >=
                    panel.loc[complete, TARGETS["T2"]])).all()
    qa = {
        "m1_v1_0_verified": True, "step1_spec_verified": True,
        "market_count": len(sources), "m1_date_count": len(dates),
        "label_row_count": len(panel), "duplicate_market_date_count":
            int(panel.duplicated(["market_proxy", "analysis_date"]).sum()),
        "min_calibration_n": int(cal_table.calibration_n.min()),
        "all_forward_end_dates_strictly_later": bool(valid_dates),
        "event_p95_subset_p90": bool(nested_events),
        "max_drawdown_horizon_monotonic": bool(ordered_mdd),
        "m1_state_score_joined": False, "target_or_threshold_selected_from_results": False,
        "available_5d_rows": int(panel.available_5d.sum()),
        "available_20d_rows": int(panel.available_20d.sum()),
        "available_60d_rows": int(panel.available_60d.sum()),
    }
    qa["all_formal_qa_pass"] = bool(
        qa["label_row_count"] == len(dates) * len(MARKETS)
        and qa["duplicate_market_date_count"] == 0
        and qa["min_calibration_n"] >= MIN_CAL
        and valid_dates and nested_events and ordered_mdd
        and all(cal_table.last_label_end_date < first))
    check(qa["all_formal_qa_pass"], f"Day14 Step2 QA failed: {qa}")
    # The metadata is written last, acting as the freeze-completion marker.
    panel.to_csv(labels_file, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    cal_table.to_csv(cal_file, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    pd.DataFrame(qa.items(), columns=["qa_name", "qa_value"]).to_csv(
        qa_file, index=False, encoding="utf-8-sig")
    metadata = {
        "research_day": 14, "step": "Step2_M4_Forward_Tail_Risk_Labels_PIT_Audit",
        "status": "FROZEN", "m1_version": "M1_v1.0",
        "market_proxies": list(MARKETS), "min_calibration_n": MIN_CAL,
        "definition_conventions": {
            "price": "positive index closing level; no missing/forward-fill",
            "return": "simple close-to-close daily returns",
            "horizon": "exactly 5/20/60 next observed trading sessions; includes P_t for MDD",
            "rv": "annualized population std (ddof=0) x sqrt(252)",
            "downside": "sqrt(252 x mean(min(simple_return,0)^2))",
            "event": "MDD20 > pre-M1 P90/P95 (per proxy, numpy quantile linear)",
            "calibration": "only windows with label_end < first M1 state date",
            "missing": "insufficient forward history -> NA; never filled with zero",
        },
        "input_sources": {m: str(p) for m, p in sources.items()},
        "input_sha256": inputs,
        "output_sha256": {p.name: digest(p) for p in files[:-1]},
        "formal_qa": qa,
    }
    meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Day14 Step2 FROZEN / QA PASS | rows={len(panel)}, "
          f"min calibration={qa['min_calibration_n']} | {out}")


if __name__ == "__main__":
    main()
