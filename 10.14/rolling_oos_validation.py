"""Day15 Step5-C: retrospective walk-forward OOS, calibrated edge-indicator model.
Only frozen Step5-B aggregated sufficient statistics; NOT independent holdout or causal inference.
Requires numpy, pandas, statsmodels. Run: python this.py --root D:\\M1_StockNetwork
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

NETS = ("B0_RAW_POSITIVE", "M1_RESIDUAL_POSITIVE")
METRICS = ("post_sector", "post_raw")
MODES = ("expanding", "rolling36")
MIN_TRAIN, PURGE, PRIOR, HAC = 36, 20, 50.0, 2
EPS = 1e-7


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def csv_bytes(df):
    return df.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8")


def scores(y, p):
    """Exact binary-label mean scores from group positive rate y."""
    p = float(np.clip(p, EPS, 1 - EPS))
    return y * (1 - p) ** 2 + (1 - y) * p ** 2, -y * np.log(p) - (1 - y) * np.log1p(-p)


def hac_test(values):
    z = np.asarray(values, float)
    if len(z) < 30 or not np.isfinite(z).all() or np.std(z) <= 1e-14:
        return dict(hac_t=np.nan, p_two_sided=np.nan, ci95_low=np.nan, ci95_high=np.nan)
    fit = sm.OLS(z, np.ones((len(z), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC})
    ci = fit.conf_int()[0]
    return dict(hac_t=float(fit.tvalues[0]), p_two_sided=float(fit.pvalues[0]),
                ci95_low=float(ci[0]), ci95_high=float(ci[1]))


def make_predictions(d, metric, mode, net):
    """Train only on completed historical dates, predict each future formation date."""
    d = d.sort_values("session_idx").reset_index(drop=True)
    records = []
    for r in d.itertuples(index=False):
        n = float(getattr(r, metric + "_n"))
        y_e = float(getattr(r, metric + "_edge"))
        y_n = float(getattr(r, metric + "_null"))
        row = dict(analysis_date=r.analysis_date, network=net, window=120,
                   metric=metric.upper(), mode=mode, session_idx=int(r.session_idx),
                   eval_n=n, observed_edge=y_e, observed_nonedge=y_n,
                   predicted_edge=np.nan, predicted_nonedge=np.nan,
                   predicted_pooled=np.nan, training_dates=0, train_last_date="",
                   brier_baseline=np.nan, brier_network=np.nan, brier_gain=np.nan,
                   logloss_baseline=np.nan, logloss_network=np.nan,
                   logloss_gain=np.nan, status="UNAVAILABLE")
        if n <= 0 or not np.isfinite(y_e + y_n):
            row["status"] = "LABEL_UNAVAILABLE"
            records.append(row)
            continue
        # Strict embargo: all training outcomes finish at least 20 market sessions before origin.
        train = d[(d.session_idx + PURGE < r.session_idx)
                  & (d[metric + "_n"] > 0)
                  & d[metric + "_edge"].notna()
                  & d[metric + "_null"].notna()]
        if mode == "rolling36":
            train = train.tail(MIN_TRAIN)
        row["training_dates"] = len(train)
        if len(train) < MIN_TRAIN:
            row["status"] = "TRAIN_WARMUP"
            records.append(row)
            continue
        w = train[metric + "_n"].to_numpy(float)
        e = train[metric + "_edge"].to_numpy(float)
        null = train[metric + "_null"].to_numpy(float)
        total = w.sum()
        require(total > 0, "Zero historical training weight")
        # Historical base rate; network model adds only a frozen binary edge indicator.
        p0 = np.clip(np.dot(w, e + null) / (2 * total), EPS, 1 - EPS)
        pe = np.clip((np.dot(w, e) + PRIOR * p0) / (total + PRIOR), EPS, 1 - EPS)
        pn = np.clip((np.dot(w, null) + PRIOR * p0) / (total + PRIOR), EPS, 1 - EPS)
        b0, l0 = [np.mean(z) for z in zip(scores(y_e, p0), scores(y_n, p0))]
        b1 = (scores(y_e, pe)[0] + scores(y_n, pn)[0]) / 2
        l1 = (scores(y_e, pe)[1] + scores(y_n, pn)[1]) / 2
        row.update(predicted_edge=pe, predicted_nonedge=pn, predicted_pooled=p0,
                   train_last_date=train.analysis_date.iloc[-1],
                   brier_baseline=b0, brier_network=b1, brier_gain=b0 - b1,
                   logloss_baseline=l0, logloss_network=l1, logloss_gain=l0 - l1,
                   status="OOS_EVALUATED")
        records.append(row)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    args = ap.parse_args()
    root = Path(args.root)
    b = root / "output/M1_day15/05b_step5b_common_shock_placebo"
    out = root / "output/M1_day15/05c_step5c_rolling_oos"
    meta_path = b / "day15_step5b_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    require(meta.get("status") == "FROZEN" and meta.get("formal_qa", {}).get("all_formal_qa_pass"),
            "Step5-B not formally frozen/PASS")
    require(meta.get("analyzed_windows") == [120] and meta.get("event_threshold") == -.05
            and meta.get("source_days") == [1, 10] and meta.get("target_followup") == [1, 5],
            "Step5-B design mismatch")
    inputs = {meta_path.name: sha(meta_path)}
    for name, expected in meta["output_sha256"].items():
        f = b / name
        require(f.is_file() and sha(f) == expected, f"Step5-B file SHA256 mismatch: {f}")
        inputs[name] = expected
    step2_path = root / "output/M1_day15/02_step2_stock_downside_target_freeze/day15_step2_metadata.json"
    require(sha(step2_path) == meta["input_sha256"]["day15_step2_metadata.json"],
            "Step2 metadata SHA256 mismatch")
    step2 = json.loads(step2_path.read_text(encoding="utf-8"))
    calendar_file = root / "output/M1_day1/01_stage1_final/trade_calendar_used.csv"
    cal_key = "output\\M1_day1\\01_stage1_final\\trade_calendar_used.csv"
    require(sha(calendar_file) == step2["input_sha256"][cal_key], "Trade calendar SHA256 mismatch")
    inputs["day15_step2_metadata.json"] = sha(step2_path)
    inputs["trade_calendar_used.csv"] = sha(calendar_file)
    calendar = pd.DatetimeIndex(pd.to_datetime(
        pd.read_csv(calendar_file, usecols=["trade_date"], low_memory=False)["trade_date"],
        errors="raise").dt.normalize().drop_duplicates().sort_values())
    require(calendar.is_unique and len(calendar) > 40, "Invalid exchange calendar")
    f = b / "temporal_common_shock_placebo_daily.csv"
    cols = ["analysis_date", "network", "window"] + [
        metric + suf for metric in METRICS for suf in ("_n", "_edge", "_null", "_delta")]
    data = pd.read_csv(f, usecols=cols, dtype={"analysis_date": "string", "network": "string"},
                       low_memory=False)
    require(len(data) == 208 and not data.duplicated(["analysis_date", "network", "window"]).any(),
            "Expected exactly 208 unique Step5-B graph/date cells")
    require(set(data.network) == set(NETS) and set(data.window) == {120}, "Network/window mismatch")
    dates = pd.to_datetime(data.analysis_date, errors="raise").dt.normalize()
    idx = calendar.get_indexer(dates)
    require((idx >= 0).all() and len(np.unique(idx)) == 104, "Formation dates/calendar mismatch")
    data["session_idx"] = idx
    for m in METRICS:
        n, e, q = (data[m + suffix].to_numpy(float) for suffix in ("_n", "_edge", "_null"))
        delta = data[m + "_delta"].to_numpy(float)
        valid = n > 0
        require((n >= 0).all() and np.isfinite(e[valid] + q[valid]).all()
                and ((e[valid] >= 0) & (e[valid] <= 1) & (q[valid] >= 0) & (q[valid] <= 1)).all()
                and np.allclose(delta[valid], e[valid] - q[valid], atol=1e-8)
                and np.isnan(delta[~valid]).all(), f"Invalid {m} sufficient statistics")
        # Step5-B uses 3 draws: aggregated mean n * 3 and n * p * 3 should be integer.
        for x in (3*n[valid], 3*n[valid]*e[valid], 3*n[valid]*q[valid]):
            require(np.max(np.abs(x - np.rint(x))) < 2e-6,
                    f"Step5-B {m} trial counts inconsistent with 3 random repeats")
    out_meta = out / "day15_step5c_metadata.json"
    script_hash = sha(Path(__file__))
    if out.exists() and any(out.iterdir()):
        require(out_meta.is_file(), "Existing partial Step5-C output; refusing overwrite")
        prior = json.loads(out_meta.read_text(encoding="utf-8"))
        require(prior.get("status") == "FROZEN" and prior.get("script_sha256") == script_hash
                and prior.get("input_sha256") == inputs, "Frozen Step5-C code/input drift")
        for filename, expected in prior["output_sha256"].items():
            require(sha(out / filename) == expected, f"Frozen Step5-C output drift: {filename}")
        print("Day15 Step5-C: EXISTING FROZEN OUTPUT VERIFIED; no overwrite")
        return
    rows = []
    for net in NETS:
        sub = data[data.network == net]
        for m in METRICS:
            for mode in MODES:
                rows.extend(make_predictions(sub, m, mode, net))
    daily = pd.DataFrame(rows)
    summary = []
    for (net, metric, mode), d in daily.groupby(["network", "metric", "mode"], sort=True):
        v = d[d.status == "OOS_EVALUATED"].sort_values("session_idx")
        n = v.eval_n.to_numpy(float)
        for score in ("brier", "logloss"):
            gain = v[score + "_gain"].to_numpy(float)
            base = v[score + "_baseline"].to_numpy(float)
            fitted = v[score + "_network"].to_numpy(float)
            summary.append(dict(network=net, window=120, metric=metric, mode=mode,
                                score=score, n_oos_dates=len(v), first_oos_date=v.analysis_date.iloc[0] if len(v) else "",
                                last_oos_date=v.analysis_date.iloc[-1] if len(v) else "",
                                mean_daily_gain=float(gain.mean()) if len(gain) else np.nan,
                                weighted_baseline=float(np.average(base, weights=n)) if len(n) else np.nan,
                                weighted_network=float(np.average(fitted, weights=n)) if len(n) else np.nan,
                                weighted_gain=float(np.average(gain, weights=n)) if len(n) else np.nan,
                                positive_gain_dates=int((gain > 0).sum()), **hac_test(gain)))
    summary = pd.DataFrame(summary)
    good = summary.p_two_sided.notna()
    summary["bh_q_16_tests"] = np.nan
    if good.any():
        summary.loc[good, "bh_q_16_tests"] = multipletests(
            summary.loc[good, "p_two_sided"].to_numpy(float), method="fdr_bh")[1]
    summary["test_status"] = np.where(good, "ESTIMABLE", "INSUFFICIENT_INFORMATION")
    count = daily[daily.status == "OOS_EVALUATED"].groupby(["network", "metric", "mode"]).size()
    predicted = daily[daily.status == "OOS_EVALUATED"]
    train_ix = calendar.get_indexer(pd.to_datetime(predicted.train_last_date, errors="raise"))
    purged_ok = bool(((train_ix >= 0) & (train_ix + PURGE < predicted.session_idx.to_numpy(int))).all())
    require(purged_ok, "Detected a training label that overlaps the forecast origin")
    qa = dict(step5b_frozen_pass=True, calendar_sha256_checked=True, graph_date_cells=208,
              distinct_formation_dates=104, walk_forward=True, embargo_market_sessions=PURGE,
              train_min_dates=MIN_TRAIN, all_prediction_training_purged=purged_ok,
              score_exact_from_group_sufficient_statistics=True,
              independent_new_unseen_holdout=False, stock_level_model=False,
              no_future_feature_or_label_used_in_training=True,
              oos_prediction_rows=int((daily.status == "OOS_EVALUATED").sum()),
              predicted_dates_min=int(count.min()) if len(count) else 0,
              predefined_tests=len(summary), estimable_tests=int(good.sum()))
    qa["all_formal_qa_pass"] = bool(len(daily) == 104*2*2*2 and len(summary) == 16
                                  and qa["predicted_dates_min"] >= 30 and good.all())
    require(qa["all_formal_qa_pass"], f"Step5-C QA BLOCKED: {qa}")
    files = {"rolling_oos_daily.csv": csv_bytes(daily),
             "rolling_oos_score_summary.csv": csv_bytes(summary),
             "day15_step5c_qa.csv": csv_bytes(pd.DataFrame(qa.items(), columns=["qa_name", "qa_value"]))}
    metadata = dict(research_day=15, step="Step5C_Retrospective_Rolling_OOS", status="FROZEN",
                    model="constant historical conditional probability; baseline pooled edge/nonedge, network separate edge/nonedge",
                    dependent_variables=list(METRICS), modes=list(MODES), min_train_dates=MIN_TRAIN,
                    purge_market_sessions=PURGE, prior_strength=PRIOR,
                    hac_lags=HAC, number_of_bh_tests=16,
                    scoring="exact binary Brier and log loss from fractional 3-replicate group event counts",
                    interpretation="retrospective walk-forward only; no per-stock conditional forecasts, no fresh independent holdout, no causal proof",
                    input_sha256=inputs, script_sha256=script_hash,
                    output_sha256={name: hashlib.sha256(value).hexdigest() for name,value in files.items()},
                    formal_qa=qa)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        (out / name).write_bytes(value)  # binary: Windows newline safe
    out_meta.write_bytes((json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print("Day15 Step5-C: FROZEN/PASS | OOS rows:", qa["oos_prediction_rows"])
    print(summary[(summary.network == NETS[1]) & (summary.metric == "POST_SECTOR")]
          [["mode", "score", "n_oos_dates", "weighted_gain", "p_two_sided", "bh_q_16_tests"]]
          .to_string(index=False))
    print("Output:", out)


if __name__ == "__main__":
    main()
