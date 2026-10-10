"""Day14 Step4 | Frozen M1-M4 continuous risk-state validation.
Run: python day14_step4_continuous_risk_state_validation.py --root D:\\M1_StockNetwork
Requires: pandas, numpy, statsmodels. Does not change any upstream freeze.
"""
from pathlib import Path, PureWindowsPath
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
# Import only the estimators/helpers used here. statsmodels.api also loads
# unrelated state-space DLLs that may be blocked by application-control policy.
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.multitest import multipletests

LAGS = {5: 1, 20: 2, 60: 4}          # Prespecified HAC lags in M1 date units
MIN_TRAIN = 36                       # OOS expanding-window training minimum
CONTROLS = ["past20_vol", "past20_mdd"]
MARKETS = ["CSI300", "CSI500", "SSE_COMPOSITE"]


def check(ok, message):
    if not bool(ok):
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def readmeta(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_hash(record, path):
    found = [v for k, v in record.items()
             if k.replace("\\", "/").split("/")[-1] == path.name]
    check(len(found) == 1 and sha(path) == found[0], f"SHA256 mismatch: {path}")


def price_table(path):
    df = pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    dc = next((cols[c] for c in ("trade_date", "date", "datetime") if c in cols), None)
    pc = next((cols[c] for c in ("close", "close_price", "index_close") if c in cols), None)
    check(dc is not None and pc is not None, f"Price fields missing: {path}")
    raw = df[dc].astype(str).str.strip()
    fmt = "%Y%m%d" if raw.str.fullmatch(r"\d{8}").all() else None
    dt = pd.to_datetime(raw, format=fmt, errors="raise").dt.normalize()
    price = pd.to_numeric(df[pc], errors="coerce")
    check(dt.notna().all() and dt.is_unique and np.isfinite(price).all()
          and price.gt(0).all(), f"Invalid date/close: {path}")
    return pd.Series(price.to_numpy(float), index=pd.DatetimeIndex(dt)).sort_index()


def historical_controls(prices, formation_dates):
    """Only closes P[t-20] ... P[t], available as of t after close."""
    idx = prices.index.get_indexer(formation_dates)
    check(np.all(idx >= 20), "Missing formation date or less than 20 past sessions")
    close = prices.to_numpy()
    values = []
    for i in idx:
        p = close[i-20:i+1]
        r = p[1:] / p[:-1] - 1
        values.append((float(np.std(r, ddof=0) * np.sqrt(252)),
                       float(np.max(1 - p / np.maximum.accumulate(p)))))
    return pd.DataFrame(values, columns=CONTROLS)


def regression(df, ycol, horizon):
    d = df.dropna(subset=[ycol, "score", *CONTROLS]).sort_values("analysis_date")
    check(len(d) >= MIN_TRAIN, f"Insufficient observations: {ycol}")
    y = d[ycol].astype(float)
    c = add_constant(d[CONTROLS], has_constant="add")
    u = add_constant(d[["score"]], has_constant="add")
    a = add_constant(d[["score", *CONTROLS]], has_constant="add")
    check(np.linalg.matrix_rank(a.to_numpy()) == a.shape[1], "Rank-deficient design")
    opts = {"cov_type": "HAC", "cov_kwds": {"maxlags": LAGS[horizon]}}
    m0 = OLS(y, c).fit()
    m1 = OLS(y, u).fit(**opts)
    m2 = OLS(y, a).fit(**opts)
    ci = m2.conf_int().loc["score"]
    return dict(n=len(d), hac_lag=LAGS[horizon],
                univ_beta_per_0p1=float(m1.params["score"] / 10),
                univ_p=float(m1.pvalues["score"]),
                adjusted_beta_per_0p1=float(m2.params["score"] / 10),
                adjusted_t=float(m2.tvalues["score"]),
                adjusted_p=float(m2.pvalues["score"]),
                adjusted_ci_low_per_0p1=float(ci.iloc[0] / 10),
                adjusted_ci_high_per_0p1=float(ci.iloc[1] / 10),
                baseline_r2=float(m0.rsquared), augmented_r2=float(m2.rsquared),
                delta_in_sample_r2=float(m2.rsquared - m0.rsquared))


def rolling_primary(df):
    """Forecast at t uses only training labels with label_end_20d < t."""
    d = df.sort_values("analysis_date").dropna(
        subset=["fwd_20d_max_drawdown_loss", "score", *CONTROLS]).copy()
    d["label_end_20d"] = pd.to_datetime(d["label_end_20d"], errors="raise")
    results = []
    for current in d.itertuples(index=False):
        t = current.analysis_date
        train = d.loc[d.label_end_20d < t]
        if len(train) < MIN_TRAIN:
            continue
        y = train.fwd_20d_max_drawdown_loss.to_numpy(float)
        b = np.column_stack([np.ones(len(train)), train[CONTROLS].to_numpy(float)])
        a = np.column_stack([b, train.score.to_numpy(float)])
        xb = np.array([1., current.past20_vol, current.past20_mdd])
        xa = np.r_[xb, current.score]
        pred_b = float(xb @ np.linalg.lstsq(b, y, rcond=None)[0])
        pred_a = float(xa @ np.linalg.lstsq(a, y, rcond=None)[0])
        results.append(dict(forecast_date=t, last_train_label_end=train.label_end_20d.max(),
                            n_train=len(train), actual=float(current.fwd_20d_max_drawdown_loss),
                            baseline_pred=pred_b, augmented_pred=pred_a))
    out = pd.DataFrame(results)
    check(len(out) >= 20, "Not enough strictly available expanding-OOS predictions")
    check((out.last_train_label_end < out.forecast_date).all(), "OOS look-ahead detected")
    err_b = (out.actual - out.baseline_pred).to_numpy() ** 2
    err_a = (out.actual - out.augmented_pred).to_numpy() ** 2
    delta = err_b - err_a
    test = OLS(delta, np.ones((len(delta), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": LAGS[20]})
    stats = dict(n_oos=len(out), min_train=MIN_TRAIN, mse_baseline=float(err_b.mean()),
                 mse_augmented=float(err_a.mean()),
                 oos_mse_improvement=float(1 - err_a.mean() / err_b.mean())
                 if err_b.mean() > 0 else np.nan,
                 mean_squared_loss_gain=float(delta.mean()),
                 loss_gain_hac_t=float(test.tvalues[0]),
                 loss_gain_hac_p_two_sided=float(test.pvalues[0]))
    return out, stats


def write_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    root = Path(ap.parse_args().root)
    b = root / "output" / "M1_day14"
    s1 = b / "01_step1_m4_tail_risk_target_specification_freeze"
    s2 = b / "02_step2_m4_forward_tail_risk_labels"
    s3 = b / "03_step3_m1_m4_pit_alignment_summary"
    out = b / "04_step4_continuous_risk_state_validation"
    spec_file = s1 / "m4_tail_risk_target_specification.csv"
    panel_file = s3 / "m1_m4_pit_aligned_panel.csv"
    meta2_file = s2 / "day14_step2_metadata.json"
    meta3_file = s3 / "day14_step3_metadata.json"
    for p in (spec_file, panel_file, meta2_file, meta3_file):
        check(p.is_file(), f"Missing frozen input: {p}")
    m2, m3 = readmeta(meta2_file), readmeta(meta3_file)
    for m in (m2, m3):
        check(m.get("status") == "FROZEN" and m.get("m1_version") == "M1_v1.0"
              and m.get("formal_qa", {}).get("all_formal_qa_pass") is True,
              "Upstream freeze/QA invalid")
    verify_hash(m3["output_sha256"], panel_file)
    # Verify that Day13 and Day14 frozen upstream artifacts have not drifted.
    upstream = [
        root / "output/M1_day13/05_step5_m1_m4_frozen_interface/m1_to_m4_frozen_interface_panel.csv",
        root / "output/M1_day13/06_step6_m1_v1_final_freeze/m1_v1_0_final_freeze_manifest.csv",
        root / "output/M1_day13/06_step6_m1_v1_final_freeze/m1_v1_0_freeze_metadata.json",
        spec_file, s1 / "day14_step1_metadata.json",
        s2 / "m4_forward_tail_risk_labels.csv", s2 / "m4_tail_event_calibration.csv",
        s2 / "day14_step2_qa.csv", meta2_file,
    ]
    for p in upstream:
        check(p.is_file(), f"Missing upstream freeze artifact: {p}")
        verify_hash(m3["input_sha256"], p)
    for fname in ("m4_forward_tail_risk_labels.csv", "m4_tail_event_calibration.csv", "day14_step2_qa.csv"):
        verify_hash(m2["output_sha256"], s2 / fname)
    spec = pd.read_csv(spec_file)
    expected = spec.loc[spec.tier != "DERIVED_EVENT"].copy()
    check(len(spec) == 8 and len(expected) == 6 and len(spec.loc[spec.tier == "PRIMARY"]) == 1
          and spec.set_index("target_id").loc["T1", "target_name"] == "fwd_20d_max_drawdown_loss",
          "Frozen target specification changed")
    paths = {}
    for market in MARKETS:
        p = Path(m2["input_sources"][market])
        if not p.is_file():
            p = root / "data" / "m4_market_indexes" / PureWindowsPath(m2["input_sources"][market]).name
        check(p.is_file(), f"Missing original market file: {market} / {p}")
        verify_hash(m2["input_sha256"], p)
        paths[market] = p
    files = [out / name for name in (
        "m1_m4_continuous_regression.csv", "m1_m4_primary_expanding_oos.csv",
        "m1_m4_primary_oos_summary.csv", "day14_step4_qa.csv")]
    metadata_file = out / "day14_step4_metadata.json"
    inputs = [panel_file, meta3_file, *upstream, *paths.values()]
    signatures = {str(p): sha(p) for p in inputs}
    if any(p.exists() for p in [*files, metadata_file]):
        check(all(p.is_file() for p in [*files, metadata_file]), "Incomplete old Step4 outputs")
        old = readmeta(metadata_file)
        check(old.get("input_sha256") == signatures
              and old.get("script_sha256") == sha(Path(__file__))
              and old.get("output_sha256") == {p.name: sha(p) for p in files}
              and old.get("formal_qa", {}).get("all_formal_qa_pass") is True,
              "Step4 frozen file/code drift; refusing overwrite")
        print("Day14 Step4: existing frozen results verified, unchanged")
        return
    panel = pd.read_csv(panel_file)
    panel["analysis_date"] = pd.to_datetime(panel["analysis_date"], errors="raise")
    check(len(panel) == 312 and not panel.duplicated(["market_proxy", "analysis_date"]).any()
          and set(panel.market_proxy) == set(MARKETS), "Unexpected frozen panel")
    panel["score"] = pd.to_numeric(panel.m1_network_risk_state_score, errors="raise")
    check(panel.score.notna().all() and panel.score.between(0, 1).all(), "Invalid network score")
    reg_rows, market_panels = [], {}
    for market in MARKETS:
        d = panel.loc[panel.market_proxy == market].sort_values("analysis_date").reset_index(drop=True)
        check(len(d) == 104, f"{market}: unexpected row count")
        feats = historical_controls(price_table(paths[market]), pd.DatetimeIndex(d.analysis_date))
        d = pd.concat([d, feats], axis=1)
        check(d[CONTROLS].notna().all().all(), "Missing historical controls")
        market_panels[market] = d
        for z in expected.itertuples(index=False):
            result = regression(d, z.target_name, int(z.horizon_td))
            reg_rows.append(dict(market_proxy=market, target_id=z.target_id,
                                 tier=z.tier, target_name=z.target_name,
                                 horizon_td=int(z.horizon_td), **result))
    reg = pd.DataFrame(reg_rows)
    reg["adjusted_bh_q_18"] = multipletests(reg.adjusted_p, method="fdr_bh")[1]
    oos, oos_stats = rolling_primary(market_panels["CSI300"])
    qa = dict(m1_v1_0_and_step1_step3_verified=True,
              original_market_sha256_verified=True,
              market_count=len(MARKETS), continuous_targets=len(expected),
              regression_rows=len(reg), primary_target="CSI300_T1",
              oos_forecasts=len(oos), oos_min_train=MIN_TRAIN,
              no_oos_label_lookahead=bool((oos.last_train_label_end < oos.forecast_date).all()),
              historical_controls_only=True, target_or_market_selected_from_results=False,
              original_m1_state_reestimated=False)
    qa["all_formal_qa_pass"] = bool(len(reg) == 18 and len(oos) >= 20
                                    and reg.adjusted_p.notna().all() and qa["no_oos_label_lookahead"])
    check(qa["all_formal_qa_pass"], f"Step4 QA FAILED: {qa}")
    out.mkdir(parents=True, exist_ok=True)
    for p, df in zip(files, (reg, oos, pd.DataFrame([oos_stats]),
                             pd.DataFrame(qa.items(), columns=["qa_name", "qa_value"]))):
        write_csv(df, p)
    metadata = dict(research_day=14, step="Step4_Continuous_Risk_State_Predictive_Validation",
                    status="FROZEN", m1_version="M1_v1.0", primary_market="CSI300",
                    primary_target="T1", continuous_targets=[str(x) for x in expected.target_id],
                    controls=CONTROLS, historical_lookback_sessions=20,
                    control_availability="P[t-20]..P[t], usable after close t; no future prices",
                    hac_lags=LAGS, expanding_min_train=MIN_TRAIN,
                    expanding_training_rule="label_end_20d < current forecast_date",
                    interpretation="Exploratory after Step3 descriptive results; HAC is in-sample association, expanding pseudo-real-time OOS is primary only; no causality or independent prospective replication",
                    input_sha256=signatures, script_sha256=sha(Path(__file__)),
                    output_sha256={p.name: sha(p) for p in files}, formal_qa=qa)
    tmp = Path(str(metadata_file) + ".tmp")
    tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, metadata_file)
    primary = reg.loc[(reg.market_proxy == "CSI300") & (reg.target_id == "T1")].iloc[0]
    print(f"Day14 Step4 FROZEN / QA PASS | regressions={len(reg)}, primary expanding OOS={len(oos)}")
    print(f"CSI300 T1 adjusted HAC beta(+0.1 score)={primary.adjusted_beta_per_0p1:.6g}, p={primary.adjusted_p:.4g}")
    print(f"Primary OOS MSE improvement vs market-only={oos_stats['oos_mse_improvement']:.2%}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
