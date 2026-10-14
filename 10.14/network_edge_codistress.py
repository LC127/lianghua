"""Day15 Step3 | Frozen 1% network edge co-distress vs industry-matched non-edges.
No network re-estimation; exploratory, NOT causal contagion. Requires pandas/numpy/statsmodels/pyarrow.
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

NETWORKS = ("B0_RAW_POSITIVE", "M1_RESIDUAL_POSITIVE")
WINDOWS = (60, 120, 252)
Q, REPEATS, HAC_LAGS = 0.01, 3, 2
EDGE_COLS = ["master_index_i", "master_index_j", "rank_fraction_common", "rank",
             "edge_weight", "edge_sign", "industry_id1_i", "industry_id1_j",
             "analysis_date", "window", "network_id"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def pair_keys(a, b, width):
    return np.minimum(a, b).astype(np.int64) * width + np.maximum(a, b)


def in_sorted_pairs(sorted_keys, pairs):
    pos = np.searchsorted(sorted_keys, pairs)
    return (pos < len(sorted_keys)) & (sorted_keys[np.minimum(pos, len(sorted_keys) - 1)] == pairs)


def match_once(a, b, target_sector, losses, sorted_keys, width, rng):
    """Keep source fixed; shuffle destination endpoints within destination industry.
    Candidate must not be a real edge or self-pair. Return paired observed/null means.
    """
    cand = np.full(len(a), -1, dtype=np.int64)
    groups = {k: np.flatnonzero(target_sector == k) for k in np.unique(target_sector)}
    for idx in groups.values():
        if len(idx) > 1:
            cand[idx] = b[rng.permutation(idx)]
    for attempt in range(10):
        possible = cand >= 0
        bad = ~possible | (cand == a)
        pos = np.flatnonzero(possible & ~bad)
        if len(pos):
            bad[pos] |= in_sorted_pairs(sorted_keys, pair_keys(a[pos], cand[pos], width))
        if not bad.any():
            break
        for sector, idx in groups.items():
            fix = idx[bad[idx]]
            if len(fix) and len(idx) > 1:
                cand[fix] = rng.choice(b[idx], size=len(fix), replace=True)
    good = (cand >= 0) & (cand != a)
    pos = np.flatnonzero(good)
    if len(pos):
        good[pos] &= ~in_sorted_pairs(sorted_keys, pair_keys(a[pos], cand[pos], width))
    if not good.any():
        return np.nan, np.nan, 0
    return (float(np.mean(losses[a[good]] * losses[b[good]])),
            float(np.mean(losses[a[good]] * losses[cand[good]])), int(good.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    args = ap.parse_args()
    root = Path(args.root)
    base = root / "output/M1_day15"
    prev = base / "02_step2_stock_downside_target_freeze"
    out = base / "03_step3_network_edge_codistress"
    graph = root / "output/M1_day4/02_step2_matched_networks"
    meta2path = prev / "day15_step2_metadata.json"
    need(meta2path.is_file(), f"Missing Step2 metadata: {meta2path}")
    meta2 = json.loads(meta2path.read_text(encoding="utf-8"))
    need(meta2.get("status") == "FROZEN" and meta2["formal_qa"]["all_formal_qa_pass"],
         "Day15 Step2 is not frozen/PASS")
    inputs = {"day15_step2_metadata.json": sha(meta2path)}
    for name, expected in meta2["output_sha256"].items():
        p = prev / name
        need(p.is_file() and sha(p) == expected, f"Step2 output drift: {p}")
        inputs[name] = expected

    # Fast verification of a previously frozen run: validate original inputs,
    # ALL node/edge file hashes, and outputs, without expensive recomputation.
    frozen_meta = out / "day15_step3_metadata.json"
    if frozen_meta.exists():
        old = json.loads(frozen_meta.read_text(encoding="utf-8"))
        need(old.get("status") == "FROZEN" and old["script_sha256"] == sha(Path(__file__))
             and old["input_sha256"] == inputs and old["formal_qa"]["all_formal_qa_pass"],
             "Previously frozen Step3 version/input mismatch")
        for name, expected in old["output_sha256"].items():
            need((out / name).is_file() and sha(out / name) == expected,
                 f"Frozen Step3 output drift: {name}")
        previous_edges = pd.read_csv(out / "day15_step3_edge_manifest.csv")
        need(len(previous_edges) == 624, "Frozen edge manifest cardinality changed")
        for rec in previous_edges.itertuples(index=False):
            p = root / rec.source.replace("\\", "/")
            need(p.is_file() and sha(p) == rec.sha256, f"Frozen edge input drift: {p}")
        prior_nodes = pd.read_csv(prev / "day15_step2_node_snapshot_manifest.csv")
        for rec in prior_nodes.itertuples(index=False):
            p = root / rec.path.replace("\\", "/")
            need(p.is_file() and sha(p) == rec.sha256, f"Frozen node drift: {p}")
        print("Day15 Step3: FROZEN REVERIFIED / PASS | 624 edge hashes | 312 node hashes")
        return

    manifest = pd.read_csv(prev / "day15_step2_node_snapshot_manifest.csv", dtype={"analysis_date": str})
    need(len(manifest) == 312 and not manifest.duplicated(["analysis_date", "window"]).any(),
         "Step2 node manifest is not unique 104 x 3")
    for rec in manifest.itertuples(index=False):
        p = root / rec.path.replace("\\", "/")
        need(p.is_file() and sha(p) == rec.sha256, f"Frozen node snapshot drift: {p}")

    # Avoid pandas C-parser low_memory chunk-type inference error on the 460k-row CSV.
    label_cols = ["analysis_date", "master_index", "label_available",
                  "downside_loss_20d", "label_end_date", "label_status"]
    label_types = {"analysis_date": "string", "master_index": "int64",
                   "label_available": "boolean", "downside_loss_20d": "float64",
                   "label_end_date": "string", "label_status": "string"}
    label_path = prev / "stock_forward_downside_20d_panel.csv"
    try:
        labels = pd.read_csv(label_path, usecols=label_cols,
                             dtype=label_types, low_memory=False)
    except (IndexError, pd.errors.ParserError):
        # Some pandas installations have a C-parser chunk concatenation bug.
        labels = pd.read_csv(label_path, usecols=label_cols,
                             dtype=label_types, engine="python")
    need(labels["label_available"].notna().all(), "NA found in label_available")
    need(not labels.duplicated(["analysis_date", "master_index"]).any(), "Duplicate labels")
    need(labels.analysis_date.nunique() == meta2["formal_qa"]["frozen_formation_dates"]
         and int(labels.label_available.sum()) == meta2["formal_qa"]["available_labels"],
         "Frozen Day15 label sample unexpectedly changed")
    good = labels.label_available
    need(labels.loc[good, "downside_loss_20d"].notna().all()
         and labels.loc[~good, "downside_loss_20d"].isna().all()
         and (pd.to_datetime(labels.loc[good, "label_end_date"])
              > pd.to_datetime(labels.loc[good, "analysis_date"])).all(), "Invalid label PIT/NA")
    width = int(labels.master_index.max()) + 1
    panels = {d: g for d, g in labels.groupby("analysis_date", sort=False)}
    dates = sorted(panels)
    need(set(dates) == set(manifest.analysis_date), "Labels/node dates mismatch")

    daily, edge_files = [], []
    for day in dates:
        g = panels[day]
        loss = np.full(width, np.nan, dtype=float)
        loss[g.master_index.to_numpy(dtype=int)] = g.downside_loss_20d.to_numpy(float)
        # Form-date universe is frozen BEFORE filtering on future labels.
        node_index = {}
        for w in WINDOWS:
            m = manifest.loc[(manifest.analysis_date == day) & (manifest.window == w)]
            need(len(m) == 1, f"Node manifest missing W{w}: {day}")
            node = pd.read_parquet(root / m.iloc[0]["path"].replace("\\", "/"),
                                   columns=["master_index"])
            ids = node.master_index.to_numpy(dtype=int)
            need(len(ids) == len(np.unique(ids)) == int(m.iloc[0].n_nodes),
                 f"Node-universe inconsistent: {day}/W{w}")
            node_index[w] = np.zeros(width, dtype=bool)
            node_index[w][ids] = True

        for w in WINDOWS:
            for net in NETWORKS:
                p = graph / "edge_master" / net / f"W{w}" / f"{day}.parquet"
                need(p.is_file(), f"Missing frozen edge snapshot: {p}")
                file_hash = sha(p)
                e = pd.read_parquet(p, columns=EDGE_COLS)
                need(not e.empty, f"Empty edge snapshot: {p}")
                need(e.analysis_date.astype(str).str[:10].eq(day).all()
                     and e.window.eq(w).all() and e.network_id.eq(net).all(),
                     f"Edge snapshot identity mismatch: {p}")
                q = pd.to_numeric(e.rank_fraction_common, errors="coerce").to_numpy(float)
                need(np.isfinite(q).all() and (q > 0).all(), f"Invalid frozen ranks: {p}")
                e = e.loc[q <= Q].copy()
                need(len(e) > 0, f"No edges at fixed density 1%: {p}")
                a = e.master_index_i.to_numpy(dtype=np.int64)
                b = e.master_index_j.to_numpy(dtype=np.int64)
                ok = (a >= 0) & (b >= 0) & (a < width) & (b < width)
                need(ok.all() and node_index[w][a].all() and node_index[w][b].all(),
                     f"Edges outside frozen node universe: {p}")
                keys = pair_keys(a, b, width)
                need((a != b).all() and len(np.unique(keys)) == len(keys)
                     and e.edge_sign.eq(1).all()
                     and (pd.to_numeric(e.edge_weight, errors="coerce") > 0).all(),
                     f"Duplicate/self/non-positive edges: {p}")
                edge_files.append(dict(analysis_date=day, network=net, window=w,
                                       source=str(p.relative_to(root)), sha256=file_hash,
                                       selected_1pct_edges=len(e)))
                valid = np.isfinite(loss[a]) & np.isfinite(loss[b])
                ii = pd.to_numeric(e.industry_id1_i, errors="coerce").fillna(-1).to_numpy(int)
                jj = pd.to_numeric(e.industry_id1_j, errors="coerce").fillna(-1).to_numpy(int)
                eligible = valid & (ii >= 0) & (jj >= 0)
                base = dict(analysis_date=day, network=net, window=w,
                            edges_1pct=len(e), edges_valid=int(valid.sum()),
                            valid_edge_share=float(valid.mean()),
                            industry_eligible_edges=int(eligible.sum()),
                            edge_codistress=(float(np.mean(loss[a[valid]] * loss[b[valid]]))
                                              if valid.any() else np.nan))
                if eligible.sum() < 3:
                    daily.append(dict(**base, matched_n_mean=0, matched_share=0.0,
                                      edge_paired=np.nan, nonedge_matched=np.nan, delta=np.nan))
                    continue
                # Randomize orientation for undirected networks; deterministic per cell.
                seed = int(hashlib.sha256(f"{day}|{net}|{w}".encode()).hexdigest()[:16], 16)
                rng = np.random.default_rng(seed)
                a, b, ii, jj = a[eligible], b[eligible], ii[eligible], jj[eligible]
                swap = rng.random(len(a)) < .5
                src = np.where(swap, b, a)
                dst = np.where(swap, a, b)
                dest_industry = np.where(swap, ii, jj)
                sorted_keys = np.sort(keys)
                samples = [match_once(src, dst, dest_industry, loss, sorted_keys, width, rng)
                           for _ in range(REPEATS)]
                samples = [x for x in samples if x[2] > 0]
                nmatch = float(np.mean([x[2] for x in samples])) if samples else 0.0
                daily.append(dict(**base, matched_n_mean=nmatch,
                                  matched_share=nmatch / len(src),
                                  edge_paired=float(np.mean([x[0] for x in samples])) if samples else np.nan,
                                  nonedge_matched=float(np.mean([x[1] for x in samples])) if samples else np.nan,
                                  delta=float(np.mean([x[0] - x[1] for x in samples])) if samples else np.nan))
        print(f"[{day}] completed: 6 graph snapshots", flush=True)

    daily = pd.DataFrame(daily)
    need(len(daily) == 104 * 6 and len(edge_files) == 104 * 6, "Incomplete 6x104 cells")
    need((daily.edges_valid <= daily.edges_1pct).all()
         and (daily.industry_eligible_edges <= daily.edges_valid).all(), "Bad coverage counts")
    kcheck = daily.pivot_table(index=["analysis_date", "window"],
                               columns="network", values="edges_1pct", aggfunc="first")
    need(kcheck.notna().all().all() and kcheck[NETWORKS[0]].eq(kcheck[NETWORKS[1]]).all(),
         "Raw/Residual fixed-density edge budgets do not match")
    # Per-date HAC inference: NEVER treat millions of edges as independent observations.
    summary = []
    for net in NETWORKS:
        for w in WINDOWS:
            s = daily[(daily.network == net) & (daily.window == w)]
            z = s.delta.dropna().to_numpy(float)
            need(len(z) >= 20, f"Insufficient monthly matched observations: {net}/W{w}")
            fit = sm.OLS(z, np.ones((len(z), 1))).fit(
                cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
            ci = fit.conf_int()[0]
            summary.append(dict(network=net, window=w, primary=(net == NETWORKS[1] and w == 120),
                                dates_total=len(s), dates_tested=len(z),
                                total_edges=int(s.edges_1pct.sum()),
                                valid_edge_share=float(s.edges_valid.sum() / s.edges_1pct.sum()),
                                matched_share=float(s.matched_n_mean.sum() / s.industry_eligible_edges.sum()),
                                mean_edge_paired=s.edge_paired.mean(),
                                mean_nonedge=s.nonedge_matched.mean(),
                                delta_mean=float(fit.params[0]), hac_t=float(fit.tvalues[0]),
                                hac_p_two_sided=float(fit.pvalues[0]),
                                ci95_low=float(ci[0]), ci95_high=float(ci[1])))
    summary = pd.DataFrame(summary)
    summary["bh_q_six_tests"] = multipletests(summary.hac_p_two_sided, method="fdr_bh")[1]
    qa = dict(step2_frozen_pass=True, selected_density=Q, graph_cells=len(daily),
              unique_frozen_dates=len(dates), network_count=2, windows_count=3,
              missing_edge_files=0, node_hashes_checked=len(manifest),
              edge_hashes_recorded=len(edge_files), matched_density_budgets_equal=True,
              bad_edge_count_cells=int((daily.edges_1pct <= 0).sum()),
              matched_dates_primary=int(summary.loc[summary.primary, "dates_tested"].iloc[0]),
              no_network_reestimation=True, no_future_universe_filtering=True,
              no_significance_gate=True, no_causal_interpretation=True)
    qa["all_formal_qa_pass"] = (qa["graph_cells"] == 624
                               and qa["node_hashes_checked"] == 312
                               and qa["edge_hashes_recorded"] == 624
                               and qa["bad_edge_count_cells"] == 0)
    need(qa["all_formal_qa_pass"], "Step3 formal QA failed")

    outputs = {"network_edge_codistress_daily.csv": daily.to_csv(index=False, float_format="%.12g"),
               "network_edge_codistress_hac_summary.csv": summary.to_csv(index=False, float_format="%.12g"),
               "day15_step3_edge_manifest.csv": pd.DataFrame(edge_files).to_csv(index=False),
               "day15_step3_qa.csv": pd.DataFrame(list(qa.items()), columns=["qa_name", "qa_value"]).to_csv(index=False)}
    hashes = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in outputs.items()}
    meta = dict(research_day=15, step="Step3_Network_Edge_Codistress_Clustering",
                status="FROZEN", primary="M1_RESIDUAL_POSITIVE_W120",
                network_density=Q, repeats=REPEATS, hac_lags=HAC_LAGS,
                analysis="Exploratory observational co-distress; no causal contagion",
                null="Fixed source, permuted target within target industry among valid edge endpoints; reject real edges and self-loops; degree-weighted target exposure",
                missing="Complete-case edges; per-date coverage and matched share always reported",
                input_sha256=inputs, script_sha256=sha(Path(__file__)),
                output_sha256=hashes, formal_qa=qa)
    out.mkdir(parents=True, exist_ok=True)
    prevmeta = out / "day15_step3_metadata.json"
    if prevmeta.exists():
        need(json.loads(prevmeta.read_text(encoding="utf-8")) == meta,
             "Existing Step3 metadata differs: never silently overwrite frozen research")
        for name, expected in hashes.items():
            need((out / name).is_file() and sha(out / name) == expected,
                 f"Existing Step3 frozen output drift: {name}")
    else:
        need(not any((out / name).exists() for name in outputs),
             "Partial prior Step3 outputs exist; inspect before rerun")
        for name, content in outputs.items():
            temp = out / (name + ".tmp")
            temp.write_text(content, encoding="utf-8")
            os.replace(temp, out / name)
        temp = out / "day15_step3_metadata.json.tmp"
        temp.write_text(json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                        encoding="utf-8")
        os.replace(temp, prevmeta)
    print(f"Day15 Step3: FROZEN / PASS | network-date cells={len(daily)}")
    print(summary[["network", "window", "primary", "delta_mean", "hac_p_two_sided", "bh_q_six_tests"]].to_string(index=False))
    print("Output:", out)


if __name__ == "__main__":
    main()
