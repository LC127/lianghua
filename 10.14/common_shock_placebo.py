"""Day15 Step5-B: common-shock-adjusted temporal risk ordering and pre-event placebo.
Use only frozen 1% M1 graphs; observational robustness, NOT causal or OOS.
Requires numpy, pandas, pyarrow, statsmodels. No CLI changes to frozen Step5-A.
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
WINS = (120,)  # fixed main W120; Raw is the pre-specified comparator
Q, THRESH, MAX_EDGE, REPS, SEED = .01, -.05, 4000, 3, 20261010
HIST, MIN_HIST, MIN_PEERS, HAC = 60, 40, 10, 2
METRICS = ("post_raw", "post_sector", "pre_raw", "pre_sector")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def keys(a, b, n):
    return np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)


def is_edge(ordered_keys, query):
    pos = np.searchsorted(ordered_keys, query)
    return (pos < len(ordered_keys)) & (ordered_keys[np.minimum(pos, len(ordered_keys)-1)] == query)


def strata(history, caps, nodes):
    """Formation-date volatility quintile + size tercile, no future data."""
    count = np.isfinite(history).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        vol = np.nanstd(history, axis=0, ddof=1)
    good = np.zeros(len(caps), bool)
    good[nodes] = (count[nodes] >= MIN_HIST) & np.isfinite(vol[nodes])
    good &= np.isfinite(caps) & (caps > 0)
    vb, sb = np.full(len(caps), -1, int), np.full(len(caps), -1, int)
    if good.any():
        vb[good] = np.searchsorted(np.quantile(vol[good], [.2, .4, .6, .8]), vol[good], "right")
        sb[good] = np.searchsorted(np.quantile(caps[good], [1/3, 2/3]), caps[good], "right")
    return vb, sb


def sector_excess(future, nodes, sector):
    """Return less formation-sector contemporaneous leave-one-out peer mean."""
    out = np.full_like(future, np.nan)
    good_nodes = nodes[sector[nodes] >= 0]
    if not len(good_nodes):
        return out
    names, codes = np.unique(sector[good_nodes], return_inverse=True)
    for k in range(min(15, len(future))):
        r = future[k, good_nodes]
        valid = np.isfinite(r)
        sums = np.bincount(codes[valid], weights=r[valid], minlength=len(names))
        counts = np.bincount(codes[valid], minlength=len(names))
        sufficient = valid & (counts[codes] >= MIN_PEERS + 1)
        ii = good_nodes[sufficient]
        out[k, ii] = r[sufficient] - (sums[codes[sufficient]] - r[sufficient]) / (counts[codes[sufficient]] - 1)
    return out


def first_shocks(future):
    shocks = np.isfinite(future[:10]) & (future[:10] <= THRESH)
    complete = np.isfinite(future[:10]).all(0)
    first = np.where(shocks.any(0), shocks.argmax(0)+1, 0)
    first[~complete] = 0
    return first


def compare(src, dst, target_sector, vol_bin, cap_bin, full_keys,
            first, future, excess, width, rng):
    """Build null pools using t-information; evaluate post/pre on separate risk sets."""
    valid = (target_sector >= 0) & (vol_bin[dst] >= 0) & (cap_bin[dst] >= 0)
    groups = target_sector.astype(np.int64) * 15 + vol_bin[dst]*3 + cap_bin[dst]
    pools = {int(g): dst[valid & (groups == g)] for g in np.unique(groups[valid])}
    ix = np.flatnonzero(valid & (first[src] > 0))
    shock_trials = int((first[src] > 0).sum())
    sums = {k: [0., 0., 0] for k in METRICS}  # n, count(edge events), count(null events)
    if not len(ix):
        return shock_trials, 0, {k: (0., np.nan, np.nan, np.nan) for k in METRICS}
    a, b, g, s = src[ix], dst[ix], groups[ix], first[src[ix]]
    gpos = {int(k): np.flatnonzero(g == k) for k in np.unique(g)}
    for _ in range(REPS):
        null = np.full(len(a), -1, dtype=np.int64)
        for grp, pos in gpos.items():
            if len(pools[grp]) > 1:
                null[pos] = rng.choice(pools[grp], size=len(pos), replace=True)
        for _attempt in range(5):
            active = null >= 0
            bad = active & (null == a)
            ix2 = np.flatnonzero(active & ~bad)
            if len(ix2):
                bad[ix2] = is_edge(full_keys, keys(a[ix2], null[ix2], width))
            if not bad.any():
                break
            for grp, pos in gpos.items():
                take = pos[bad[pos]]
                if len(take) and len(pools[grp]) > 1:
                    null[take] = rng.choice(pools[grp], size=len(take), replace=True)
        good = (null >= 0) & (null != a)
        ix2 = np.flatnonzero(good)
        if len(ix2):
            good[ix2] &= ~is_edge(full_keys, keys(a[ix2], null[ix2], width))
        for day in range(1, 11):
            take = np.flatnonzero(good & (s == day))
            if not len(take):
                continue
            bi, ni = b[take], null[take]
            raw = future[:day+5, :][:, np.r_[bi, ni]]
            ex = excess[:day+5, :][:, np.r_[bi, ni]]
            n = len(take)
            # Main risk set: no raw shock at or before source day; full observations;
            # same complete-case pairs used for raw and sector-excess outcomes.
            complete = np.isfinite(raw).all(0) & np.isfinite(ex[day:day+5]).all(0)
            no_prior = ~(raw[:day] <= THRESH).any(0)
            mask = (complete & no_prior)[:n] & (complete & no_prior)[n:]
            ix3 = np.flatnonzero(mask)
            if len(ix3):
                for metric, val in (("post_raw", raw[day:] <= THRESH),
                                    ("post_sector", ex[day:] <= THRESH)):
                    ev = val.any(0)
                    q = sums[metric]
                    q[0] += len(ix3)
                    q[1] += ev[ix3].sum()
                    q[2] += ev[ix3+n].sum()
            # Negative-control PRE-source placebo: no no-prior-shock condition!
            # Need five sessions wholly after formation, hence source day >= 6.
            if day >= 6:
                before = raw[day-6:day-1]
                before_ex = ex[day-6:day-1]
                m = (np.isfinite(before).all(0) & np.isfinite(before_ex).all(0))
                ix3 = np.flatnonzero(m[:n] & m[n:])
                if len(ix3):
                    for metric, val in (("pre_raw", before <= THRESH),
                                        ("pre_sector", before_ex <= THRESH)):
                        ev = val.any(0)
                        q = sums[metric]
                        q[0] += len(ix3)
                        q[1] += ev[ix3].sum()
                        q[2] += ev[ix3+n].sum()
    result = {}
    for key, (n, re, nu) in sums.items():
        result[key] = (n/REPS, re/n if n else np.nan,
                       nu/n if n else np.nan, (re-nu)/n if n else np.nan)
    return shock_trials, len(ix), result


def hac(z):
    z = np.asarray(z, float)
    if len(z) < 30 or not np.isfinite(z).all() or np.std(z) < 1e-15:
        return dict(mean_delta=z.mean() if len(z) else np.nan,
                    hac_t=np.nan, p_two_sided=np.nan, ci95_low=np.nan, ci95_high=np.nan)
    fit = sm.OLS(z, np.ones((len(z), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC})
    return dict(mean_delta=float(fit.params[0]), hac_t=float(fit.tvalues[0]),
                p_two_sided=float(fit.pvalues[0]), ci95_low=float(fit.conf_int()[0, 0]),
                ci95_high=float(fit.conf_int()[0, 1]))


def csv_data(frame):
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\M1_StockNetwork")
    root = Path(parser.parse_args().root)
    day15 = root / "output/M1_day15"
    folders = [day15 / n for n in ("02_step2_stock_downside_target_freeze",
                                     "03_step3_network_edge_codistress",
                                     "04_step4_community_cross_industry_pathways",
                                     "05_step5_temporal_order_matched_null")]
    out = day15 / "05b_step5b_common_shock_placebo"
    names = ["day15_step2_metadata.json", "day15_step3_metadata.json",
             "day15_step4_metadata.json", "day15_step5_metadata.json"]
    inputs, metas = {}, []
    for folder, name in zip(folders, names):
        p = folder/name
        check(p.is_file(), f"Missing frozen metadata: {p}")
        m = json.loads(p.read_text(encoding="utf-8"))
        check(m.get("status") == "FROZEN" and m.get("formal_qa", {}).get("all_formal_qa_pass"), f"Upstream not PASS: {p}")
        inputs[name] = sha(p)
        for filename, expected in m["output_sha256"].items():
            q = folder/filename
            check(q.is_file() and sha(q) == expected, f"Frozen output changed: {q}")
            inputs[filename] = expected
        metas.append(m)
    check(metas[3]["daily_event"] == "return_network_simple <= -0.05"
          and metas[3]["source_first_days"] == [1, 10]
          and metas[3]["target_followup_days"] == [1, 5]
          and metas[3]["density"] == Q, "Step5-A design drift")
    check(metas[2]["primary"] == "M1_RESIDUAL_POSITIVE_W120"
          and metas[2]["density"] == Q, "Step4 design drift")
    day1 = root/"output/M1_day1"
    rfile = day1/"03_stage3_return_validation/daily_return_panel.parquet"
    cfile = day1/"01_stage1_final/trade_calendar_used.csv"
    mapfile = root/"output/M1_day4/02_step2_matched_networks/stock_master_mapping.parquet"
    for file in (rfile, cfile, mapfile):
        rel = str(file.relative_to(root)).replace("/", "\\")
        check(file.is_file() and sha(file) == metas[0]["input_sha256"][rel], f"Day1/4 source drift: {file}")
    em = pd.read_csv(folders[1]/"day15_step3_edge_manifest.csv", dtype={"analysis_date":str}, low_memory=False)
    nm = pd.read_csv(folders[0]/"day15_step2_node_snapshot_manifest.csv", dtype={"analysis_date":str}, low_memory=False)
    check(len(em) == 624 and len(nm) == 312
          and not em.duplicated(["analysis_date", "network", "window"]).any()
          and not nm.duplicated(["analysis_date", "window"]).any(), "Bad edge/node manifest keys")
    for rec in em.itertuples(index=False):
        p = root/rec.source.replace("\\", "/")
        check(p.is_file() and sha(p) == rec.sha256, f"Frozen edge drift: {p}")
    for rec in nm.itertuples(index=False):
        p = root/rec.path.replace("\\", "/")
        check(p.is_file() and sha(p) == rec.sha256, f"Frozen node drift: {p}")
    prior = out/"day15_step5b_metadata.json"
    if prior.is_file():
        old = json.loads(prior.read_text(encoding="utf-8"))
        check(old["status"] == "FROZEN" and old["input_sha256"] == inputs
              and old["script_sha256"] == sha(Path(__file__)), "Step5-B version drift")
        for filename, digest in old["output_sha256"].items():
            check((out/filename).is_file() and sha(out/filename) == digest,
                  f"Step5-B frozen output drift: {filename}")
        print("Day15 Step5-B: FROZEN REVERIFIED / PASS")
        return
    check(not out.exists() or not list(out.iterdir()), "Existing nonempty unfinished Step5-B directory")
    cal = pd.to_datetime(pd.read_csv(cfile, usecols=["trade_date"], low_memory=False).trade_date, errors="coerce")
    check(cal.notna().all(), "Bad trading calendar")
    dates = pd.DatetimeIndex(cal.dt.normalize().sort_values())
    check(dates.is_unique and dates.is_monotonic_increasing, "Duplicate/unsorted calendar")
    mapping = pd.read_parquet(mapfile, columns=["master_index", "security_id"]).sort_values("master_index")
    check(np.array_equal(mapping.master_index.to_numpy(), np.arange(len(mapping))), "Day4 master index drift")
    securities = pd.Index(mapping.security_id.astype("string").str.strip())
    check(securities.is_unique, "Duplicate security mapping")
    formation = pd.read_csv(folders[0]/"stock_forward_downside_20d_panel.csv",
                            usecols=["analysis_date", "master_index", "in_w60", "in_w120", "in_w252"],
                            dtype={"analysis_date":"string", "master_index":"int64",
                                   **{f"in_w{w}":"int8" for w in WINS}}, low_memory=False)
    check(not formation.duplicated(["analysis_date", "master_index"]).any(), "Duplicate formation labels")
    fs = {str(t): g for t,g in formation.groupby("analysis_date", sort=True)}
    check(len(fs) == 104 and set(fs) == set(em.analysis_date) == set(nm.analysis_date), "104 formation dates changed")
    fdates = sorted(fs)
    findex = {dates.get_indexer([pd.Timestamp(d)])[0]: k for k,d in enumerate(fdates)}
    check(all(v >= 0 for v in findex), "Formation date outside trading calendar")
    R = np.full((len(dates), len(securities)), np.nan, dtype=np.float32)
    CAP = np.full((len(fdates), len(securities)), np.nan, dtype=np.float64)
    seen = np.zeros(R.shape, bool)
    pqfile = pq.ParquetFile(rfile)
    columns = ["trade_date", "security_id", "return_network_simple", "is_open",
               "is_consecutive_open_pair", "market_value"]
    check(set(columns).issubset(pqfile.schema_arrow.names), "Missing daily-return or market_value columns")
    frow = np.full(len(dates), -1, int)
    for k,v in findex.items():
        frow[k] = v
    for batch in pqfile.iter_batches(batch_size=250_000, columns=columns):
        b = batch.to_pandas()
        ti = dates.get_indexer(pd.to_datetime(b.trade_date).dt.normalize())
        si = securities.get_indexer(b.security_id.astype("string").str.strip())
        check((ti >= 0).all() and (si >= 0).all(), "Unknown return date/security")
        key = ti.astype(np.int64)*len(securities)+si
        check(not seen[ti,si].any() and len(np.unique(key)) == len(key), "Duplicate daily return key")
        ret = pd.to_numeric(b.return_network_simple, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)
        good = (b.is_open.eq(True).fillna(False).to_numpy(dtype=bool)
                & b.is_consecutive_open_pair.eq(True).fillna(False).to_numpy(dtype=bool))
        ret[~good] = np.nan
        check(not np.any((ret < -1) | np.isinf(ret)), "Invalid simple return")
        R[ti,si] = ret
        seen[ti,si] = True
        ii = frow[ti]
        at = ii >= 0
        if at.any():
            CAP[ii[at],si[at]] = pd.to_numeric(b.market_value, errors="coerce").to_numpy(dtype=float, na_value=np.nan)[at]
    edge_index = {(str(r.analysis_date),r.network,int(r.window)):r for r in em.itertuples(index=False)}
    node_index = {(str(r.analysis_date),int(r.window)):r for r in nm.itertuples(index=False)}
    rows = []
    censored_dates = []
    for day in fdates:
        ix = dates.get_indexer([pd.Timestamp(day)])[0]
        enough = ix+20 < len(dates)  # same right-censoring convention as Step5-A
        if not enough:
            censored_dates.append(day)
        if enough:
            future = R[ix+1:ix+16].astype(float)
            history = R[max(0,ix-HIST+1):ix+1].astype(float)
            first = first_shocks(future)
        for w in WINS:
            nrec = node_index[(day,w)]
            node = pd.read_parquet(root/nrec.path.replace("\\","/"), columns=["master_index","industry_id1"])
            ids = node.master_index.to_numpy(np.int64)
            expected = fs[day].loc[fs[day][f"in_w{w}"] == 1, "master_index"].to_numpy(np.int64)
            check(len(ids)==int(nrec.n_nodes)==len(np.unique(ids)) and (ids>=0).all()
                  and (ids<len(securities)).all() and np.array_equal(np.sort(ids),np.sort(expected)),
                  f"Frozen nodes not preserved: {day} W{w}")
            sector = np.full(len(securities), -1, int)
            sector[ids] = pd.to_numeric(node.industry_id1, errors="coerce").fillna(-1).to_numpy(int)
            vb,sb = strata(history, CAP[fdates.index(day)], ids) if enough else (None,None)
            excess = sector_excess(future, ids, sector) if enough else None
            for net in NETS:
                rec = edge_index[(day,net,w)]
                p = root/rec.source.replace("\\","/")
                edge = pd.read_parquet(p, columns=["master_index_i", "master_index_j",
                                                "rank_fraction_common", "industry_id1_i", "industry_id1_j"])
                edge = edge.loc[pd.to_numeric(edge.rank_fraction_common, errors="coerce") <= Q]
                check(len(edge) == int(rec.selected_1pct_edges) and len(edge)>0,
                      f"1% density budget changed: {p}")
                a = edge.master_index_i.to_numpy(np.int64)
                b = edge.master_index_j.to_numpy(np.int64)
                check(((a>=0)&(b>=0)&(a<len(securities))&(b<len(securities))&(a!=b)).all(),
                      f"Invalid edge endpoint: {p}")
                check((sector[a] == pd.to_numeric(edge.industry_id1_i,errors="coerce").fillna(-1).to_numpy(int)).all()
                      and (sector[b] == pd.to_numeric(edge.industry_id1_j,errors="coerce").fillna(-1).to_numpy(int)).all(),
                      f"Node/edge sector mismatch: {p}")
                full = np.sort(keys(a,b,len(securities)))
                check(len(np.unique(full))==len(full), f"Duplicate edges: {p}")
                rng = np.random.default_rng(int(hashlib.sha256(f"{SEED}|{day}|{net}|{w}".encode()).hexdigest()[:16],16))
                if len(a)>MAX_EDGE:
                    chosen = rng.choice(len(a),MAX_EDGE,replace=False)
                    a,b = a[chosen],b[chosen]
                row = dict(analysis_date=day,network=net,window=w,
                           edges_1pct=len(full),edges_sampled=len(a),
                           sampled_share=len(a)/len(full),source_shock_trials=0,strata_eligible_trials=0)
                if enough:
                    src,dst = np.r_[a,b],np.r_[b,a]
                    shocks,eligible,rr = compare(src,dst,sector[dst],vb,sb,full,first,future,excess,len(securities),rng)
                    row.update(source_shock_trials=shocks,strata_eligible_trials=eligible)
                else:
                    rr = {name:(0.,np.nan,np.nan,np.nan) for name in METRICS}
                for name,(n,real,control,delta) in rr.items():
                    row.update({f"{name}_n":n,f"{name}_edge":real,
                                f"{name}_null":control,f"{name}_delta":delta,
                                f"{name}_share":n/max(1,row["source_shock_trials"])})
                rows.append(row)
        print(f"[{day}] Step5-B done",flush=True)
    daily = pd.DataFrame(rows)
    check(len(daily)==208 and not daily.duplicated(["analysis_date","network","window"]).any(),
          "Expected 208 W120 graph/date cells")
    summary=[]
    for net in NETS:
        for w in WINS:
            d = daily[(daily.network==net)&(daily.window==w)]
            for metric in METRICS:
                z = d[f"{metric}_delta"].dropna().to_numpy(float)
                summary.append(dict(network=net,window=w,test=metric.upper(),n_dates=len(z),
                                    **hac(z),mean_match_share=d.loc[d[f"{metric}_delta"].notna(),f"{metric}_share"].mean()))
    for w in WINS:
        d = daily[daily.window==w].pivot(index="analysis_date",columns="network",values="post_sector_delta")
        z = (d[NETS[1]]-d[NETS[0]]).dropna().to_numpy(float)
        summary.append(dict(network="RESIDUAL_MINUS_RAW",window=w,test="POST_SECTOR_CONTRAST",n_dates=len(z),
                            **hac(z),mean_match_share=np.nan))
    summary = pd.DataFrame(summary)
    available = np.isfinite(summary.p_two_sided.to_numpy(float))
    summary["bh_q_9_tests"] = np.where(available,
        multipletests(summary.p_two_sided.fillna(1.0),method="fdr_bh")[1],np.nan)
    summary["test_status"] = np.where(available,"ESTIMABLE","INSUFFICIENT_INFORMATION")
    main = daily[(daily.network==NETS[1])&(daily.window==120)]
    qa = dict(step2_3_4_5a_frozen_pass=True,edge_hashes_checked=624,node_hashes_checked=312,
              unique_dates=len(fdates),network_date_cells=len(daily),analyzed_windows=[120],predeclared_tests=len(summary),
              estimable_tests=int(available.sum()),primary_post_sector_dates=int(main.post_sector_delta.notna().sum()),
              primary_pre_sector_dates=int(main.pre_sector_delta.notna().sum()),
              calendar_right_censored_cells=2*len(censored_dates),
              zero_post_matched_cells=int((daily.post_raw_n==0).sum()),
              no_future_matching=True,independent_pre_placebo_risk_set=True,
              contemporaneous_leave_one_out_sector_adjustment=True,
              pit_size_and_vol_strata=True,no_network_reestimation=True,
              no_significance_gate=True,no_causal_or_oos_claim=True)
    qa["all_formal_qa_pass"] = bool(len(daily)==208 and len(summary)==9
        and daily[[f"{k}_n" for k in METRICS]].ge(0).all().all()
        and daily[[f"{k}_share" for k in METRICS]].le(1+1e-9).all().all()
        and qa["primary_post_sector_dates"]>=30)
    check(qa["all_formal_qa_pass"],f"Formal QA BLOCKED: {qa}")
    output={"temporal_common_shock_placebo_daily.csv":csv_data(daily),
            "temporal_common_shock_placebo_hac.csv":csv_data(summary),
            "day15_step5b_qa.csv":csv_data(pd.DataFrame(qa.items(),columns=["qa_name","qa_value"]))}
    meta=dict(research_day=15,step="Step5B_Common_Shock_Controls_Placebo",status="FROZEN",
              main="M1_RESIDUAL_POSITIVE_W120",density=Q,event_threshold=THRESH,
              source_days=[1,10],target_followup=[1,5],pre_placebo_days=5,
              pre_placebo_source_day_min=6,historical_vol_sessions=HIST,min_hist=MIN_HIST,
              min_sector_other_peers=MIN_PEERS,size_strata=3,vol_strata=5,analyzed_windows=[120],
              sampling_max_edges=MAX_EDGE,repeats=REPS,seed=SEED,hac_lags=HAC,
              common_shock_control="target return minus same-day formation-sector leave-one-out equal-weight peer return; market component absorbed by sector; no factor-beta control",
              matched_null="same source/date; target formation sector, historical volatility quintile and market-cap tercile; non-edge endpoint pool",
              placebo="five market sessions before source shock, source shock in days 6..10; separate sample, no pre-source event exclusion",
              caution="Exploratory conditional associations, not independent OOS or causal contagion; post-vs-pre differ in risk set",
              input_sha256=inputs,script_sha256=sha(Path(__file__)),
              output_sha256={k:hashlib.sha256(v).hexdigest() for k,v in output.items()},formal_qa=qa)
    out.mkdir(parents=True,exist_ok=True)
    for filename,data in output.items():
        temp=out/(filename+".tmp")
        temp.write_bytes(data)
        os.replace(temp,out/filename)
    temp=out/"day15_step5b_metadata.json.tmp"
    temp.write_bytes((json.dumps(meta,ensure_ascii=False,indent=2,sort_keys=True)+"\n").encode("utf-8"))
    os.replace(temp,prior)
    print(f"Day15 Step5-B: FROZEN/PASS | 208 cells | {available.sum()}/9 estimable tests")
    print(summary[(summary.network==NETS[1])&(summary.window==120)][["test","n_dates","mean_delta","p_two_sided","bh_q_9_tests"]].to_string(index=False))
    print("Outputs:",out)


if __name__=="__main__":
    main()
