"""Day15 Step5: exploratory ordered downside shocks, frozen 1% graphs.
Requires numpy, pandas, pyarrow, statsmodels. No causal transmission claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

NETS = ("B0_RAW_POSITIVE", "M1_RESIDUAL_POSITIVE")
WINS = (60, 120, 252)
PRIMARY = ("M1_RESIDUAL_POSITIVE", 120)
DENSITY, SHOCK, SOURCE_DAYS, LAG, HIST, MIN_HIST = .01, -.05, 10, 5, 60, 40
SAMPLE, REPEATS, SEED, HAC = 12000, 3, 20261010, 2


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def need(ok, why):
    if not ok:
        raise RuntimeError(why)


def edge_keys(a, b, width):
    return np.minimum(a, b).astype(np.int64) * width + np.maximum(a, b)


def connected(keys, query):
    pos = np.searchsorted(keys, query)
    return (pos < len(keys)) & (keys[np.minimum(pos, len(keys) - 1)] == query)


def event_arrays(future):
    """Source first shock in sessions 1..10; target at risk and followed for 5."""
    shock = np.isfinite(future) & (future <= SHOCK)
    full_source = np.isfinite(future[:SOURCE_DAYS]).all(axis=0)
    first = np.where(shock[:SOURCE_DAYS].any(axis=0),
                     shock[:SOURCE_DAYS].argmax(axis=0) + 1, 0)
    first[~full_source] = 0
    risk = np.zeros((SOURCE_DAYS, future.shape[1]), dtype=bool)
    event = np.zeros_like(risk)
    for s in range(1, SOURCE_DAYS + 1):
        # Target must be observable, and have NO event, through source day s.
        risk[s-1] = (np.isfinite(future[:s + LAG]).all(axis=0)
                     & ~shock[:s].any(axis=0))
        event[s-1] = shock[s:s + LAG].any(axis=0)
    return first, risk, event


def hist_buckets(history, nodes):
    """PIT 60-session historical TOTAL return volatility (not residual IVOL)."""
    good = np.isfinite(history)
    count = good.sum(axis=0)
    x = np.where(good, history, 0.)
    sums, sq = x.sum(axis=0), (x*x).sum(axis=0)
    denom = np.maximum(count, 1)
    v = np.sqrt(np.maximum(0., (sq - sums*sums/denom) / np.maximum(count-1, 1)))
    usable = np.zeros(len(v), dtype=bool)
    usable[nodes] = count[nodes] >= MIN_HIST
    bins = np.full(len(v), -1, dtype=np.int8)
    if usable.any():
        cuts = np.quantile(v[usable], [.2, .4, .6, .8])
        bins[usable] = np.searchsorted(cuts, v[usable], side="right")
    return bins


def evaluate(a, b, industry, bins, keys, first, risk, event, width, rng):
    """Match on source/date, target industry + PIT-vol quintile, using endpoints.
    Match BEFORE future outcomes; exclude true edges and self. Use identical
    real/null source shock time and paired at-risk restriction for evaluation.
    """
    # Construct the candidate pool using ONLY formation-date information.
    # Crucially, do this BEFORE selecting source shocks (which occur in the future).
    groups_all = industry.astype(np.int64) * 5 + bins[b]
    strata_valid = (industry >= 0) & (bins[b] >= 0)
    pools = {int(k): b[strata_valid & (groups_all == k)]
             for k in np.unique(groups_all[strata_valid])}
    ix = np.flatnonzero(strata_valid & (first[a] > 0))
    if len(ix) == 0:
        return 0, 0, np.nan, np.nan, np.nan
    a, b, group = a[ix], b[ix], groups_all[ix]
    time = first[a] - 1
    row_groups = {k: np.flatnonzero(group == k) for k in np.unique(group)}
    diffs, pr, pn, ns = [], [], [], []
    for _ in range(REPEATS):
        cand = np.full(len(a), -1, dtype=np.int64)
        for k, members in row_groups.items():
            if len(pools[k]) > 1:
                cand[members] = rng.choice(pools[k], len(members), replace=True)
        # All draws are based solely on time-t network/history.
        for _retry in range(12):
            possible = cand >= 0
            bad = (~possible) | (cand == a)
            pos = np.flatnonzero(possible & ~bad)
            if len(pos):
                bad[pos] |= connected(keys, edge_keys(a[pos], cand[pos], width))
            if not np.any(bad & possible):
                break
            for k, members in row_groups.items():
                sub = members[bad[members]]
                if len(sub) and len(pools[k]) > 1:
                    cand[sub] = rng.choice(pools[k], len(sub), replace=True)
        good = (cand >= 0) & (cand != a)
        pos = np.flatnonzero(good)
        if len(pos):
            good[pos] &= ~connected(keys, edge_keys(a[pos], cand[pos], width))
        pos = np.flatnonzero(good)
        if len(pos):
            good[pos] &= risk[time[pos], b[pos]] & risk[time[pos], cand[pos]]
        pos = np.flatnonzero(good)
        if not len(pos):
            continue
        real = event[time[pos], b[pos]].astype(float)
        null = event[time[pos], cand[pos]].astype(float)
        ns.append(len(pos)); pr.append(real.mean()); pn.append(null.mean())
        diffs.append((real-null).mean())
    if not diffs:
        return len(ix), 0, np.nan, np.nan, np.nan
    return (len(ix), float(np.mean(ns)), float(np.mean(pr)),
            float(np.mean(pn)), float(np.mean(diffs)))


def hac(z):
    z = np.asarray(z, float)
    if len(z) < 30 or not np.isfinite(z).all() or np.std(z) < 1e-15:
        return dict(mean_delta=float(z.mean()) if len(z) else np.nan,
                    hac_t=np.nan, p_two_sided=np.nan, ci95_low=np.nan, ci95_high=np.nan)
    fit = sm.OLS(z, np.ones((len(z), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC})
    return dict(mean_delta=float(fit.params[0]), hac_t=float(fit.tvalues[0]),
                p_two_sided=float(fit.pvalues[0]),
                ci95_low=float(fit.conf_int()[0, 0]), ci95_high=float(fit.conf_int()[0, 1]))


def csv_bytes(df):
    return df.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    root = Path(ap.parse_args().root)
    base = root / "output/M1_day15"
    d2 = base / "02_step2_stock_downside_target_freeze"
    d3 = base / "03_step3_network_edge_codistress"
    d4 = base / "04_step4_community_cross_industry_pathways"
    out = base / "05_step5_temporal_order_matched_null"
    inputs = {}
    for folder, mname in ((d2, "day15_step2_metadata.json"),
                          (d3, "day15_step3_metadata.json"),
                          (d4, "day15_step4_metadata.json")):
        p = folder / mname
        need(p.is_file(), f"Missing upstream metadata: {p}")
        m = json.loads(p.read_text(encoding="utf-8"))
        need(m["status"] == "FROZEN" and m["formal_qa"]["all_formal_qa_pass"],
             f"Upstream not frozen/PASS: {p}")
        inputs[mname] = sha(p)
        for name, digest in m["output_sha256"].items():
            q = folder / name
            need(q.is_file() and sha(q) == digest, f"Frozen output drift: {q}")
            inputs[name] = digest
    need(m["density"] == DENSITY and m["primary"] == "M1_RESIDUAL_POSITIVE_W120",
         "Step4 density or primary changed")
    rfile = root / "output/M1_day1/03_stage3_return_validation/daily_return_panel.parquet"
    cfile = root / "output/M1_day1/01_stage1_final/trade_calendar_used.csv"
    mapfile = root / "output/M1_day4/02_step2_matched_networks/stock_master_mapping.parquet"
    step2meta = json.loads((d2 / "day15_step2_metadata.json").read_text(encoding="utf-8"))
    for file in (rfile, cfile, mapfile):
        rel = str(file.relative_to(root)).replace("/", "\\")
        need(sha(file) == step2meta["input_sha256"][rel], f"Day1/Day4 source drift: {file}")
    edge_m = pd.read_csv(d3 / "day15_step3_edge_manifest.csv", dtype={"analysis_date": str})
    node_m = pd.read_csv(d2 / "day15_step2_node_snapshot_manifest.csv", dtype={"analysis_date": str})
    need(len(edge_m) == 624 and len(node_m) == 312, "Manifest cardinality incorrect")
    need(not edge_m.duplicated(["analysis_date", "network", "window"]).any()
         and not node_m.duplicated(["analysis_date", "window"]).any(), "Duplicate manifest keys")
    # Check all frozen graph hashes before even considering a previous Step5 freeze.
    for rec in edge_m.itertuples(index=False):
        p = root / str(rec.source).replace("\\", "/")
        need(p.is_file() and sha(p) == rec.sha256, f"Frozen edge drift: {p}")
    for rec in node_m.itertuples(index=False):
        p = root / str(rec.path).replace("\\", "/")
        need(p.is_file() and sha(p) == rec.sha256, f"Frozen node drift: {p}")
    prior = out / "day15_step5_metadata.json"
    if prior.exists():
        old = json.loads(prior.read_text(encoding="utf-8"))
        need(old["status"] == "FROZEN" and old["input_sha256"] == inputs
             and old["script_sha256"] == sha(Path(__file__)), "Frozen Step5 version drift")
        for name, digest in old["output_sha256"].items():
            need((out / name).is_file() and sha(out / name) == digest,
                 f"Frozen Step5 output drift: {name}")
        print("Day15 Step5: FROZEN REVERIFIED / PASS")
        return
    need(not out.exists() or not list(out.iterdir()), "Nonempty unfinished Step5 directory")
    cal = pd.to_datetime(pd.read_csv(cfile)["trade_date"], errors="coerce")
    need(cal.notna().all(), "Invalid trade calendar dates")
    dates = pd.DatetimeIndex(cal.dt.normalize().sort_values())
    need(dates.is_unique and dates.is_monotonic_increasing, "Invalid trade calendar")
    mapping = pd.read_parquet(mapfile, columns=["master_index", "security_id"])
    mapping = mapping.sort_values("master_index")
    need(np.array_equal(mapping.master_index.to_numpy(), np.arange(len(mapping))), "Master index drift")
    security = pd.Index(mapping.security_id.astype("string").str.strip())
    pf = pq.ParquetFile(rfile)
    cols = ["trade_date", "security_id", "return_network_simple", "is_open", "is_consecutive_open_pair"]
    need(set(cols).issubset(pf.schema_arrow.names), "Day1 daily-return schema changed")
    R = np.full((len(dates), len(security)), np.nan, dtype=np.float64)
    seen = np.zeros(R.shape, dtype=bool)
    for batch in pf.iter_batches(batch_size=250_000, columns=cols):
        b = batch.to_pandas()
        ti = dates.get_indexer(pd.to_datetime(b.trade_date).dt.normalize())
        si = security.get_indexer(b.security_id.astype("string").str.strip())
        need((ti >= 0).all() and (si >= 0).all(), "Unknown date/security in daily returns")
        key = ti.astype(np.int64) * len(security) + si
        need(not seen[ti, si].any() and len(np.unique(key)) == len(key), "Duplicate daily-return key")
        v = pd.to_numeric(b.return_network_simple, errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan, copy=True)
        valid = (b.is_open.eq(True).fillna(False).to_numpy(dtype=bool)
                 & b.is_consecutive_open_pair.eq(True).fillna(False).to_numpy(dtype=bool))
        v[~valid] = np.nan
        need(not np.any((v < -1) | np.isinf(v)), "Invalid daily simple returns")
        R[ti, si] = v
        seen[ti, si] = True
    lc = ["analysis_date", "master_index", "in_w60", "in_w120", "in_w252"]
    labels = pd.read_csv(d2 / "stock_forward_downside_20d_panel.csv", usecols=lc,
                         dtype={"analysis_date": "string", "master_index": "int32",
                                **{f"in_w{w}": "int8" for w in WINS}}, low_memory=False)
    need(not labels.duplicated(["analysis_date", "master_index"]).any(), "Duplicate formation universe")
    groups = {str(d): g for d, g in labels.groupby("analysis_date", sort=False)}
    need(len(groups) == 104 and len(edge_m) == len(groups) * 6, "Formation dates changed")
    em = {(str(r.analysis_date), r.network, int(r.window)): r for r in edge_m.itertuples(index=False)}
    nm = {(str(r.analysis_date), int(r.window)): r for r in node_m.itertuples(index=False)}
    daily = []
    for day in sorted(groups):
        idx = dates.get_indexer([pd.Timestamp(day)])[0]
        need(idx >= 0, f"Missing formation trading date: {day}")
        within_horizon = idx + 20 < len(dates)
        if within_horizon:
            future = R[idx+1:idx+21]
            first, risk, event = event_arrays(future)
            history = R[max(0, idx-HIST+1):idx+1]
        for w in WINS:
            nr = nm[(day, w)]
            nodes = pd.read_parquet(root / nr.path.replace("\\", "/"), columns=["master_index"])
            ids = nodes.master_index.to_numpy(np.int64)
            expected = groups[day].loc[groups[day][f"in_w{w}"] == 1, "master_index"].to_numpy(np.int64)
            need(len(ids) == int(nr.n_nodes) == len(np.unique(ids))
                 and ((ids >= 0) & (ids < len(security))).all()
                 and np.array_equal(np.sort(ids), np.sort(expected)),
                 f"Step2 node membership drift: {day} W{w}")
            membership = np.zeros(len(security), dtype=bool)
            membership[ids] = True
            bins = hist_buckets(history, ids) if within_horizon else None
            for net in NETS:
                rec = em[(day, net, w)]
                p = root / rec.source.replace("\\", "/")
                e = pd.read_parquet(p, columns=["master_index_i", "master_index_j",
                                                "rank_fraction_common", "industry_id1_i", "industry_id1_j"])
                e = e.loc[pd.to_numeric(e.rank_fraction_common, errors="coerce") <= DENSITY]
                need(len(e) == int(rec.selected_1pct_edges), f"1% edge count drift: {p}")
                a, b = e.master_index_i.to_numpy(np.int64), e.master_index_j.to_numpy(np.int64)
                need(len(a) > 0 and ((a >= 0) & (a < len(security))
                     & (b >= 0) & (b < len(security))).all(), f"Invalid edge indices: {p}")
                need(membership[a].all() and membership[b].all(), f"Edge node drift: {p}")
                full_keys = np.sort(edge_keys(a, b, len(security)))
                need(len(np.unique(full_keys)) == len(full_keys) and (a != b).all(), f"Invalid edges: {p}")
                seed = int(hashlib.sha256(f"{SEED}|{day}|{net}|{w}".encode()).hexdigest()[:16], 16)
                rng = np.random.default_rng(seed)
                if len(a) > SAMPLE:
                    chosen = rng.choice(len(a), size=SAMPLE, replace=False)
                    e = e.iloc[chosen]
                    a, b = a[chosen], b[chosen]
                row = dict(analysis_date=day, network=net, window=w, edges_1pct=len(full_keys),
                           edges_sampled=len(a), sampled_edge_share=len(a)/len(full_keys),
                           directed_trials=2*len(a), source_shock_trials=0,
                           strata_eligible_trials=0, matched_n=0., matched_share=0.,
                           p_edge=np.nan, p_nonedge=np.nan, delta=np.nan)
                if within_horizon:
                    ia = pd.to_numeric(e.industry_id1_i, errors="coerce").fillna(-1).to_numpy(int)
                    ib = pd.to_numeric(e.industry_id1_j, errors="coerce").fillna(-1).to_numpy(int)
                    src = np.r_[a, b]
                    dst = np.r_[b, a]
                    sectors = np.r_[ib, ia]
                    row["source_shock_trials"] = int((first[src] > 0).sum())
                    counts = evaluate(src, dst, sectors, bins, full_keys, first, risk, event,
                                      len(security), rng)
                    row["strata_eligible_trials"], row["matched_n"], row["p_edge"], row["p_nonedge"], row["delta"] = counts
                    row["matched_share"] = row["matched_n"] / max(1, row["source_shock_trials"])
                daily.append(row)
        print(f"[{day}] Step5 complete", flush=True)
    daily = pd.DataFrame(daily)
    need(len(daily) == 624 and not daily.duplicated(["analysis_date", "network", "window"]).any(),
         "624 network-date cells required")
    summary = []
    for net in NETS:
        for w in WINS:
            d = daily[(daily.network == net) & (daily.window == w)]
            z = d.delta.dropna().to_numpy(float)
            summary.append(dict(network=net, window=w, test="EDGE_MINUS_NULL", n_dates=len(z), **hac(z),
                                mean_matched_share=float(d.loc[d.delta.notna(), "matched_share"].mean())))
    for w in WINS:
        dd = daily[daily.window == w].pivot(index="analysis_date", columns="network", values="delta")
        z = (dd[NETS[1]] - dd[NETS[0]]).dropna().to_numpy(float)
        summary.append(dict(network="RESIDUAL_MINUS_RAW", window=w, test="NETWORK_CONTRAST",
                            n_dates=len(z), **hac(z), mean_matched_share=np.nan))
    summary = pd.DataFrame(summary)
    available = np.isfinite(summary.p_two_sided.to_numpy(float))
    summary["bh_q_9_tests"] = np.where(available,
        multipletests(summary.p_two_sided.fillna(1), method="fdr_bh")[1], np.nan)
    summary["test_status"] = np.where(available, "ESTIMABLE", "INSUFFICIENT_INFORMATION")
    qa = dict(step2_3_4_frozen_pass=True, edge_sha256_checked=624, node_sha256_checked=312,
              unique_dates=len(groups), network_date_cells=len(daily), tests=len(summary),
              estimable_tests=int(available.sum()),
              missing_future_horizon_cells=int((~daily.delta.notna() &
                  daily.analysis_date.isin([d for d in groups if dates.get_indexer([pd.Timestamp(d)])[0] + 20 >= len(dates)])).sum()),
              no_future_matching=True, source_and_target_temporal_order_strict=True,
              no_network_reestimation=True, no_significance_gate=True, no_causal_claim=True,
              min_matched_n=float(daily.loc[daily.delta.notna(), "matched_n"].min())
                  if daily.delta.notna().any() else 0.)
    qa["all_formal_qa_pass"] = bool(len(daily) == 624 and len(summary) == 9
        and daily.matched_n.le(daily.strata_eligible_trials + 1e-9).all()
        and daily.matched_share.le(1 + 1e-9).all())
    need(qa["all_formal_qa_pass"], "Formal QA BLOCKED")
    outputs = {"temporal_order_daily.csv": csv_bytes(daily),
               "temporal_order_hac_summary.csv": csv_bytes(summary),
               "day15_step5_qa.csv": csv_bytes(pd.DataFrame(qa.items(), columns=["qa_name", "qa_value"]))}
    meta = dict(research_day=15, step="Step5_Temporal_Ordering_Matched_Null",
                status="FROZEN", primary="M1_RESIDUAL_POSITIVE_W120", density=DENSITY,
                daily_event="return_network_simple <= -0.05", source_first_days=[1, SOURCE_DAYS],
                target_followup_days=[1, LAG], historical_vol_days=HIST,
                historical_min_valid=MIN_HIST, vol_strata=5, max_sampled_edges=SAMPLE,
                null="fixed source + date; matching target industry and PIT historical-volatility quintile; draw from sampled edge endpoint pool; reject real edge/self; no future used for matching",
                analysis="observational ordering, not causal contagion; no independent OOS",
                repeats=REPEATS, hac_lags=HAC, tests=9,
                input_sha256=inputs, script_sha256=sha(Path(__file__)),
                output_sha256={k: hashlib.sha256(v).hexdigest() for k, v in outputs.items()},
                formal_qa=qa)
    out.mkdir(parents=True, exist_ok=True)
    for name, b in outputs.items():
        tmp = out / (name + ".tmp")
        tmp.write_bytes(b)
        os.replace(tmp, out / name)
    tmp = out / "day15_step5_metadata.json.tmp"
    tmp.write_bytes((json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    os.replace(tmp, prior)
    print(f"Day15 Step5: FROZEN/PASS | {len(daily)} cells | {int(available.sum())}/9 estimable tests")
    print(summary[["network", "window", "mean_delta", "p_two_sided", "bh_q_9_tests"]].to_string(index=False))
    print("Output:", out)


if __name__ == "__main__":
    main()
