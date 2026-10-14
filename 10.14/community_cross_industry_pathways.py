"""Day15 Step4 | Frozen-network sector/community co-distress (exploratory).
Requires numpy, pandas, pyarrow, networkx>=2.8, statsmodels. No graph refitting.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

NETWORKS = ("B0_RAW_POSITIVE", "M1_RESIDUAL_POSITIVE")
WINDOWS = (60, 120, 252)
PRIMARY = ("M1_RESIDUAL_POSITIVE", 120)
Q, SEED, REPEATS, HAC = 0.01, 20261010, 3, 2
COLS = ["master_index_i", "master_index_j", "edge_weight",
        "rank_fraction_common", "industry_id1_i", "industry_id1_j"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def keys(a, b, width):
    return np.minimum(a, b).astype("int64") * width + np.maximum(a, b)


def is_real(sorted_keys, k):
    pos = np.searchsorted(sorted_keys, k)
    return (pos < len(sorted_keys)) & (sorted_keys[np.minimum(pos, len(sorted_keys)-1)] == k)


def paired(a, b, strata, loss, all_keys, width, rng):
    """Fix source; resample destination in same stratum; reject self/real edges."""
    dest = np.full(len(a), -1, dtype=np.int64)
    _, inv = np.unique(strata, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sections = np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)
    for ix in sections:
        if len(ix) < 2:
            continue
        dest[ix] = b[rng.permutation(ix)]
        for _ in range(8):
            bad = (dest[ix] == a[ix]) | is_real(all_keys, keys(a[ix], dest[ix], width))
            if not bad.any():
                break
            dest[ix[bad]] = rng.choice(b[ix], size=int(bad.sum()), replace=True)
    good = (dest >= 0) & (dest != a)
    ix = np.flatnonzero(good)
    if len(ix):
        good[ix] &= ~is_real(all_keys, keys(a[ix], dest[ix], width))
    if not good.any():
        return 0, np.nan, np.nan
    return int(good.sum()), float(np.mean(loss[a[good]] * loss[b[good]])), float(np.mean(loss[a[good]] * loss[dest[good]]))


def communities(e, node_ids):
    """New Day15 derivation from PIT frozen edges; NOT Day5 frozen labels."""
    require(hasattr(nx.community, "louvain_communities"), "Need networkx>=2.8 for Louvain")
    G = nx.Graph()
    G.add_nodes_from(node_ids.tolist())
    G.add_weighted_edges_from(e[["master_index_i", "master_index_j", "edge_weight"]]
                              .itertuples(index=False, name=None))
    groups = nx.community.louvain_communities(G, weight="weight", seed=SEED, resolution=1.)
    groups = sorted(groups, key=lambda v: min(v))
    return {int(i): c for c, members in enumerate(groups) for i in members}, len(groups)


def csv_bytes(df):
    return df.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8")


def infer(z):
    """Date-level HAC; missing when fewer than 30 usable dates or zero variation."""
    z = np.asarray(z, dtype=float)
    if len(z) < 30 or not np.isfinite(z).all() or np.std(z) < 1e-16:
        return dict(mean_delta=float(np.mean(z)) if len(z) else np.nan,
                    hac_t=np.nan, p_two_sided=np.nan, ci95_low=np.nan, ci95_high=np.nan)
    f = sm.OLS(z, np.ones((len(z), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC})
    return dict(mean_delta=float(f.params[0]), hac_t=float(f.tvalues[0]),
                p_two_sided=float(f.pvalues[0]), ci95_low=float(f.conf_int()[0, 0]),
                ci95_high=float(f.conf_int()[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    args = ap.parse_args()
    root = Path(args.root)
    base = root / "output/M1_day15"
    d2 = base / "02_step2_stock_downside_target_freeze"
    d3 = base / "03_step3_network_edge_codistress"
    out = base / "04_step4_community_cross_industry_pathways"
    graph = root / "output/M1_day4/02_step2_matched_networks"
    inputs = {}
    for folder, meta_name in ((d2, "day15_step2_metadata.json"),
                              (d3, "day15_step3_metadata.json")):
        p = folder / meta_name
        require(p.is_file(), f"Missing frozen input: {p}")
        m = json.loads(p.read_text(encoding="utf-8"))
        require(m["status"] == "FROZEN" and m["formal_qa"]["all_formal_qa_pass"],
                f"Upstream not FROZEN/PASS: {p}")
        inputs[meta_name] = sha(p)
        for name, expected in m["output_sha256"].items():
            q = folder / name
            require(q.is_file() and sha(q) == expected, f"Frozen output drift: {q}")
            inputs[name] = expected
    require(m["network_density"] == Q and m["primary"] == "M1_RESIDUAL_POSITIVE_W120",
            "Step3 network density or primary does not match")

    edgem = pd.read_csv(d3 / "day15_step3_edge_manifest.csv")
    nodem = pd.read_csv(d2 / "day15_step2_node_snapshot_manifest.csv")
    require(len(edgem) == 624 and len(nodem) == 312, "Expected 624 edges and 312 nodes")
    require(not edgem.duplicated(["analysis_date", "network", "window"]).any(), "Duplicate edge file")
    require(not nodem.duplicated(["analysis_date", "window"]).any(), "Duplicate node file")
    # Verify every actual frozen graph file before reading or allowing a rerun.
    for rec in edgem.itertuples(index=False):
        p = root / str(rec.source).replace("\\", "/")
        require(p.is_file() and sha(p) == rec.sha256, f"Frozen edge drift: {p}")
    for rec in nodem.itertuples(index=False):
        p = root / str(rec.path).replace("\\", "/")
        require(p.is_file() and sha(p) == rec.sha256, f"Frozen node drift: {p}")
    frozen = out / "day15_step4_metadata.json"
    if frozen.exists():
        old = json.loads(frozen.read_text(encoding="utf-8"))
        require(old["status"] == "FROZEN" and old["input_sha256"] == inputs
                and old["script_sha256"] == sha(Path(__file__))
                and old["networkx_version"] == nx.__version__, "Frozen Step4 version drift")
        for name, expected in old["output_sha256"].items():
            require((out / name).is_file() and sha(out / name) == expected,
                    f"Frozen Step4 output drift: {name}")
        print("Day15 Step4: FROZEN REVERIFIED / PASS")
        return

    labelcols = ["analysis_date", "master_index", "security_id", "downside_loss_20d"]
    lp = d2 / "stock_forward_downside_20d_panel.csv"
    labels = pd.read_csv(lp, usecols=labelcols, low_memory=False,
                         dtype={"analysis_date": "string", "master_index": "int64",
                                "security_id": "string", "downside_loss_20d": "float64"})
    require(not labels.duplicated(["analysis_date", "master_index"]).any(), "Duplicate label keys")
    require(labels.groupby("master_index").security_id.nunique().eq(1).all(),
            "Security identity inconsistent across formation dates")
    security = labels.drop_duplicates("master_index").set_index("master_index").security_id.to_dict()
    width = int(labels.master_index.max()) + 1
    edgem["analysis_date"] = edgem.analysis_date.astype(str).str[:10]
    nodem["analysis_date"] = nodem.analysis_date.astype(str).str[:10]
    dates = sorted(labels.analysis_date.unique())
    require(len(dates) == 104 and set(dates) == set(edgem.analysis_date), "Incorrect formation dates")
    edgeindex = {(r.analysis_date, r.network, int(r.window)): r for r in edgem.itertuples(index=False)}
    nodeindex = {(r.analysis_date, int(r.window)): r for r in nodem.itertuples(index=False)}
    daily, candidates = [], []
    for day, g in labels.groupby("analysis_date", sort=True):
        loss = np.full(width, np.nan)
        loss[g.master_index.to_numpy(int)] = g.downside_loss_20d.to_numpy(float)
        for net in NETWORKS:
            for w in WINDOWS:
                rec = edgeindex[(day, net, w)]
                p = root / rec.source.replace("\\", "/")
                e = pd.read_parquet(p, columns=COLS)
                e = e.loc[pd.to_numeric(e.rank_fraction_common, errors="coerce") <= Q].copy()
                require(len(e) == rec.selected_1pct_edges, f"1% edge budget changed: {day} {net} W{w}")
                e = e.sort_values(["master_index_i", "master_index_j"]).reset_index(drop=True)
                a = e.master_index_i.to_numpy(np.int64)
                b = e.master_index_j.to_numpy(np.int64)
                require(len(a) and min(a.min(), b.min()) >= 0 and max(a.max(), b.max()) < width,
                        f"Bad master_index: {day} {net} W{w}")
                sector_a = pd.to_numeric(e.industry_id1_i, errors="coerce").fillna(-1).to_numpy(int)
                sector_b = pd.to_numeric(e.industry_id1_j, errors="coerce").fillna(-1).to_numpy(int)
                valid = np.isfinite(loss[a]) & np.isfinite(loss[b])
                sector_ok = (sector_a >= 0) & (sector_b >= 0)
                real_keys = np.sort(keys(a, b, width))
                rng = np.random.default_rng(int(hashlib.sha256(f"{SEED}|{day}|{net}|{w}".encode()).hexdigest()[:16], 16))
                swap = rng.random(len(a)) < .5
                src, dst = np.where(swap, b, a), np.where(swap, a, b)
                dstsector = np.where(swap, sector_a, sector_b)
                types = {"SAME_INDUSTRY": sector_a == sector_b,
                         "CROSS_INDUSTRY": sector_a != sector_b}
                ncomm = 0
                if (net, w) == PRIMARY:
                    recn = nodeindex[(day, w)]
                    nodes = pd.read_parquet(root / recn.path.replace("\\", "/"), columns=["master_index"])
                    cmap, ncomm = communities(e, nodes.master_index.to_numpy(int))
                    ci = np.array([cmap[int(v)] for v in a], dtype=int)
                    cj = np.array([cmap[int(v)] for v in b], dtype=int)
                    dstcomm = np.where(swap, ci, cj)
                    types.update({"WITHIN_COMMUNITY": ci == cj, "CROSS_COMMUNITY": ci != cj})
                else:
                    dstcomm = None
                for typ, mask in types.items():
                    elig = valid & sector_ok & mask
                    idx = np.flatnonzero(elig)
                    base = dict(analysis_date=day, network=net, window=w,
                                group=typ, edges_total=len(e), group_edges=int(mask.sum()),
                                valid_group_edges=int(elig.sum()),
                                valid_group_share=float(elig.sum() / max(1, mask.sum())),
                                community_count=ncomm if "COMMUNITY" in typ else np.nan)
                    if len(idx) < 3:
                        daily.append(dict(**base, matched_n=0, matched_share=0.,
                                          edge_paired=np.nan, nonedge_matched=np.nan, delta=np.nan))
                        continue
                    strata = dstsector[idx].astype(np.int64)
                    if "COMMUNITY" in typ:
                        strata = strata * (ncomm + 1) + dstcomm[idx]
                    samples = [paired(src[idx], dst[idx], strata, loss, real_keys, width, rng)
                               for _ in range(REPEATS)]
                    samples = [v for v in samples if v[0] > 0]
                    n = float(np.mean([s[0] for s in samples])) if samples else 0.
                    daily.append(dict(**base, matched_n=n, matched_share=n / len(idx),
                                      edge_paired=np.mean([s[1] for s in samples]) if samples else np.nan,
                                      nonedge_matched=np.mean([s[2] for s in samples]) if samples else np.nan,
                                      delta=np.mean([s[1]-s[2] for s in samples]) if samples else np.nan))
                if (net, w) == PRIMARY and ncomm and valid.any():
                    bridge = valid & sector_ok & (sector_a != sector_b) & (ci != cj)
                    bidx = np.flatnonzero(bridge)
                    if len(bidx):
                        scores = loss[a[bidx]] * loss[b[bidx]]
                        top = bidx[np.argsort(-scores, kind="stable")[:5]]
                        for j in top:
                            candidates.append(dict(analysis_date=day, master_index_i=int(a[j]),
                                                   master_index_j=int(b[j]),
                                                   security_id_i=str(security[int(a[j])]),
                                                   security_id_j=str(security[int(b[j])]),
                                                   industry_i=int(sector_a[j]),
                                                   industry_j=int(sector_b[j]), community_i=int(ci[j]),
                                                   community_j=int(cj[j]), edge_weight=float(e.edge_weight.iloc[j]),
                                                   loss_i=float(loss[a[j]]), loss_j=float(loss[b[j]]),
                                                   realized_loss_product=float(loss[a[j]]*loss[b[j]])))
        print(f"[{day}] Step4 completed", flush=True)

    daily = pd.DataFrame(daily)
    require(len(daily) == 104 * (6*2 + 2), f"Daily group cells incorrect: {len(daily)}")
    summary = []
    pairs = [(n, w, x) for n in NETWORKS for w in WINDOWS
             for x in ("SAME_INDUSTRY", "CROSS_INDUSTRY")]
    pairs += [(PRIMARY[0], PRIMARY[1], x) for x in ("WITHIN_COMMUNITY", "CROSS_COMMUNITY")]
    for net, w, group in pairs:
        data = daily.loc[(daily.network == net) & (daily.window == w) & (daily.group == group)]
        z = data.delta.dropna().to_numpy(float)
        summary.append(dict(network=net, window=w, test_type="EXCESS_VS_MATCHED", group=group,
                            n_dates=len(z), **infer(z),
                            mean_matched_share=float(data.loc[data.delta.notna(), "matched_share"].mean())))
    for net in NETWORKS:
        for w in WINDOWS:
            groups = [("CROSS_INDUSTRY", "SAME_INDUSTRY")]
            if (net, w) == PRIMARY:
                groups.append(("CROSS_COMMUNITY", "WITHIN_COMMUNITY"))
            for x, y in groups:
                d = daily[(daily.network == net) & (daily.window == w)].pivot(index="analysis_date", columns="group", values="delta")
                z = (d[x] - d[y]).dropna().to_numpy(float)
                summary.append(dict(network=net, window=w, test_type="BETWEEN_GROUP_CONTRAST",
                                    group=f"{x}_MINUS_{y}", n_dates=len(z),
                                    **infer(z), mean_matched_share=np.nan))
    summary = pd.DataFrame(summary)
    require(len(summary) == 21, "Expected 14 subgroup tests plus 7 contrasts")
    # Keep all 21 pre-specified hypotheses in the multiplicity family.
    # Unestimable tests are counted conservatively as p=1, but displayed as NA.
    available = np.isfinite(summary.p_two_sided.to_numpy(float))
    q = multipletests(summary.p_two_sided.fillna(1.0), method="fdr_bh")[1]
    summary["bh_q_21_tests"] = np.where(available, q, np.nan)
    summary["test_status"] = np.where(available, "ESTIMABLE", "INSUFFICIENT_INFORMATION")
    qa = dict(step2_step3_frozen_pass=True, edge_sha256_checked=624, node_sha256_checked=312,
              graph_date_cells=624, date_count=104, sector_cells=1248,
              primary_community_cells=208, subgroup_tests=14, group_contrasts=7,
              estimable_tests=int(available.sum()),
              density=Q, fixed_seed=SEED, no_network_reestimation=True,
              derived_community_not_day5_frozen=True, no_causal_interpretation=True)
    qa["all_formal_qa_pass"] = bool(len(daily) == 1456 and len(summary) == 21
                                     and daily.valid_group_edges.le(daily.group_edges).all())
    require(qa["all_formal_qa_pass"], "Day15 Step4 formal QA FAILED")
    artifacts = {"network_pathway_group_daily.csv": csv_bytes(daily),
                 "network_pathway_hac_summary.csv": csv_bytes(summary),
                 "candidate_cross_industry_community_bridges.csv": csv_bytes(pd.DataFrame(
                     candidates, columns=["analysis_date", "master_index_i", "master_index_j",
                     "security_id_i", "security_id_j", "industry_i", "industry_j",
                     "community_i", "community_j", "edge_weight", "loss_i", "loss_j",
                     "realized_loss_product"])),
                 "day15_step4_qa.csv": csv_bytes(pd.DataFrame(qa.items(), columns=["qa_name", "qa_value"]))}
    meta = dict(research_day=15, step="Step4_Community_Cross_Industry_Risk_Pathways",
                status="FROZEN", primary="M1_RESIDUAL_POSITIVE_W120",
                community_method="Day15-derived Louvain on PIT frozen 1% weighted network, seed=20261010, resolution=1",
                community_scope="Residual W120 only; other 5 network settings are industry robustness",
                networkx_version=nx.__version__, matched_null="fixed source; destination same industry; community test additionally same destination community; exclude self/real edges",
                density=Q, repeats=REPEATS, hac_lags=HAC, tests=21,
                interpretation="Exploratory co-distress; ex-post bridge examples; not causal transmission or OOS",
                input_sha256=inputs, script_sha256=sha(Path(__file__)),
                output_sha256={name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()},
                formal_qa=qa)
    require(not out.exists() or not list(out.iterdir()), "Partial Step4 output directory exists; inspect before rerun")
    out.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        tmp = out / (name + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, out / name)
    tmp = out / "day15_step4_metadata.json.tmp"
    tmp.write_bytes((json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    os.replace(tmp, frozen)
    print(f"Day15 Step4: FROZEN/PASS | 1456 daily group cells | {int(available.sum())}/21 HAC tests estimable")
    print(summary.loc[(summary.network == PRIMARY[0]) & (summary.window == PRIMARY[1]),
                      ["test_type", "group", "mean_delta", "p_two_sided", "bh_q_21_tests"]].to_string(index=False))
    print("Output:", out)


if __name__ == "__main__":
    main()
