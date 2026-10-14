"""Day15 Step1: audit frozen network edges, nodes and stock-level labels; no re-estimation."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit("Missing pyarrow: run pip install pyarrow") from exc

WINS = (60, 120, 252)
NETS = ("B0_RAW_POSITIVE", "M1_RESIDUAL_POSITIVE")
EDGE_COLS = {"master_index_i", "master_index_j", "analysis_date", "window",
             "network_id", "edge_weight", "rank_fraction_common"}
NODE_COLS = {"master_index", "security_id", "analysis_date", "window"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parquet_info(path):
    pf = pq.ParquetFile(path)
    return set(pf.schema_arrow.names), int(pf.metadata.num_rows)


def write_once(path, payload, previous_pass):
    b = payload.encode("utf-8")
    if path.exists() and previous_pass:
        if path.read_bytes() != b:
            raise RuntimeError(f"Previously passed audit drift: {path}")
        return
    path.write_bytes(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    ap.add_argument("--hash-all-edges", action="store_true",
                    help="Hash every edge/node snapshot (slower but full-byte audit)")
    args = ap.parse_args()
    root = Path(args.root)
    base = root / "output/M1_day4/02_step2_matched_networks"
    edge_dir, node_dir = base / "edge_master", base / "node_universe"
    mapping = base / "stock_master_mapping.parquet"  # Day4-generated index map
    label = root / "output/M1_day6/02_step2_forward_labels/forward_label_panel.parquet"
    master = root / "output/M1_day1/01_stage1_final/stock_master.parquet"
    returns = root / "output/M1_day1/03_stage3_return_validation/daily_return_panel.parquet"
    m1_dir = root / "output/M1_day13"
    fm = m1_dir / "06_step6_m1_v1_final_freeze/m1_v1_0_freeze_metadata.json"
    manifest = m1_dir / "06_step6_m1_v1_final_freeze/m1_v1_0_final_freeze_manifest.csv"
    interface = m1_dir / "05_step5_m1_m4_frozen_interface/m1_to_m4_frozen_interface_panel.csv"
    out = root / "output/M1_day15/01_step1_frozen_network_risk_source_audit"
    out.mkdir(parents=True, exist_ok=True)

    qa, inventory, samples, fingerprints = {}, [], [], {}
    def add(role, path, row_count=None, note=""):
        inventory.append(dict(role=role, path=str(path), exists=path.exists(),
                              row_count=row_count, note=note))

    # 1. Verify M1 v1.0 lineage and exact source interface.
    required = [fm, manifest, interface, label, master, returns, mapping]
    qa["required_source_missing_count"] = sum(not p.is_file() for p in required)
    m1_dates = set()
    if qa["required_source_missing_count"] == 0:
        m = json.loads(fm.read_text(encoding="utf-8"))
        qa["m1_v1_frozen"] = (m.get("freeze_version") == "M1_v1.0"
            and m.get("status") == "FROZEN"
            and m.get("formal_qa", {}).get("all_formal_qa_pass") is True)
        f = pd.read_csv(manifest, dtype=str)
        rel = str(interface.relative_to(root)).replace("/", "\\").lower()
        match = f.loc[f.relative_path.str.lower() == rel]
        qa["interface_manifest_sha256_match"] = (len(match) == 1
            and sha256(interface) == match.iloc[0]["sha256"])
        d = pd.read_csv(interface, usecols=["analysis_date"])
        dates = pd.to_datetime(d.analysis_date, errors="coerce")
        m1_dates = set(dates.dropna().dt.strftime("%Y-%m-%d"))
        qa["interface_dates_valid_unique"] = dates.notna().all() and not dates.duplicated().any()
    else:
        qa.update(m1_v1_frozen=False, interface_manifest_sha256_match=False,
                  interface_dates_valid_unique=False)
    qa["m1_date_count"] = len(m1_dates)

    # 2. Inspect complete Day4 edge/node snapshot calendar via filenames;
    #    read parquet metadata only (never load whole all-market edge panels).
    edge_sets, node_sets = {}, {}
    schema_errors, missing_dates, invalid_date_names = [], 0, 0
    for w in WINS:
        nd = node_dir / f"W{w}"
        nf = sorted(nd.glob("*.parquet"))
        node_sets[w] = {p.stem for p in nf}
        add(f"nodes_W{w}", nd, len(nf))
        missing_dates += len(m1_dates - node_sets[w])
        for network in NETS:
            folder = edge_dir / network / f"W{w}"
            files = sorted(folder.glob("*.parquet"))
            edge_sets[network, w] = {p.stem for p in files}
            add(f"edges_{network}_W{w}", folder, len(files))
            missing_dates += len(m1_dates - edge_sets[network, w])
            invalid_date_names += sum(pd.isna(pd.to_datetime(p.stem, errors="coerce")) for p in files)
            if not files:
                schema_errors.append(f"No edges: {folder}")
                continue
            # Prefer representative snapshot on the actual frozen M1 calendar.
            chosen = next((p for p in files if p.stem in m1_dates), files[0])
            cols, n = parquet_info(chosen)
            node_file = nd / chosen.name
            if not EDGE_COLS.issubset(cols):
                schema_errors.append(f"Edge columns: {chosen}: {EDGE_COLS - cols}")
            if not node_file.is_file():
                schema_errors.append(f"Missing node partner: {node_file}")
            else:
                nc, nn = parquet_info(node_file)
                if not NODE_COLS.issubset(nc):
                    schema_errors.append(f"Node columns: {node_file}: {NODE_COLS - nc}")
                elif EDGE_COLS.issubset(cols):
                    # Validate actual endpoint identity mapping in sampled snapshot.
                    e = pd.read_parquet(chosen, columns=["master_index_i", "master_index_j",
                                                         "rank_fraction_common", "edge_weight",
                                                         "analysis_date", "window", "network_id"])
                    nodes = pd.read_parquet(node_file, columns=["master_index", "security_id"])
                    endpoints = pd.concat([e.master_index_i, e.master_index_j], ignore_index=True)
                    bad = int((~endpoints.isin(nodes.master_index)).sum())
                    if bad or nodes.master_index.duplicated().any() or nodes.security_id.duplicated().any() or e.master_index_i.eq(e.master_index_j).any():
                        schema_errors.append(f"Endpoint/node identity QA: {chosen}")
                    if (pd.to_datetime(e.analysis_date, errors="coerce").dt.strftime("%Y-%m-%d") != chosen.stem).any() or not e.window.eq(w).all() or not e.network_id.eq(network).all():
                        schema_errors.append(f"Snapshot identity mismatch: {chosen}")
                    # Max-density masters contain more than the frozen 1%-density selection.
                    fraction = pd.to_numeric(e.rank_fraction_common, errors="coerce")
                    if fraction.isna().any() or not fraction.between(0, 1).all() or not (fraction <= .01).any():
                        schema_errors.append(f"Bad rank fraction / 1% availability: {chosen}")
                    samples.append(dict(network=network, window=w, analysis_date=chosen.stem,
                                        edge_rows=n, node_rows=nn,
                                        sampled_1pct_edges=int((fraction <= .01).sum()),
                                        unmatched_endpoints=bad))
            fingerprints[str(chosen.relative_to(root))] = sha256(chosen)
    qa["missing_required_snapshot_dates"] = missing_dates
    qa["invalid_edge_filename_dates"] = invalid_date_names
    qa["snapshot_schema_or_identity_errors"] = len(schema_errors)
    qa["sampled_edge_network_window_cells"] = len(samples)

    # 3. Stock master, daily returns and Day6 canonical forward-label panel.
    for role, p, expected in [
        ("stock_master", master, {"security_id", "stock_code", "stock_name",
                                  "exchange", "list_date", "delist_date"}),
        ("day4_stock_mapping", mapping, {"master_index", "security_id"}),
        ("daily_returns", returns, {"security_id"}),
        ("day6_forward_labels", label, {"security_id", "analysis_date", "window"})]:
        if p.is_file():
            cols, n = parquet_info(p)
            add(role, p, n, "columns=" + ",".join(sorted(cols)))
            if not expected.issubset(cols):
                schema_errors.append(f"{role} missing {expected - cols}")
            fingerprints[str(p.relative_to(root))] = sha256(p)
        else:
            add(role, p, note="MISSING")
    # Day4 assigned master_index AFTER sorting the Day1 master by security_id.
    # Audit that preserved mapping; never create a new mapping for Day15.
    qa["day4_master_mapping_consistent"] = False
    if master.is_file() and mapping.is_file():
        mc, _ = parquet_info(master)
        xc, _ = parquet_info(mapping)
        if "security_id" in mc and {"master_index", "security_id"}.issubset(xc):
            original = pd.read_parquet(master, columns=["security_id"])["security_id"]
            original = original.astype("string").str.strip().sort_values().reset_index(drop=True)
            indexed = pd.read_parquet(mapping, columns=["master_index", "security_id"])
            indexed = indexed.sort_values("master_index").reset_index(drop=True)
            ids = indexed["security_id"].astype("string").str.strip()
            idx = indexed["master_index"].tolist()
            qa["day4_master_mapping_consistent"] = bool(
                len(original) == len(indexed)
                and idx == list(range(len(indexed)))
                and original.equals(ids)
                and not original.isna().any()
                and not original.duplicated().any()
            )
            if not qa["day4_master_mapping_consistent"]:
                schema_errors.append("Day4 stock_master_mapping differs from Day1 sorted security_id")
    qa["snapshot_schema_or_identity_errors"] = len(schema_errors)
    qa["day6_label_schema_valid"] = False
    qa["day6_label_dates_cover_m1"] = False
    qa["day6_label_duplicate_key_count"] = -1
    if label.is_file():
        cols, _ = parquet_info(label)
        if {"security_id", "analysis_date", "window"}.issubset(cols):
            d = pd.read_parquet(label, columns=["security_id", "analysis_date", "window"])
            day6_dates = set(pd.to_datetime(d.analysis_date, errors="coerce")
                             .dropna().dt.strftime("%Y-%m-%d"))
            qa["day6_label_schema_valid"] = True
            qa["day6_label_dates_cover_m1"] = m1_dates.issubset(day6_dates)
            qa["day6_label_duplicate_key_count"] = int(d.duplicated(
                ["security_id", "analysis_date", "window"]).sum())
    if args.hash_all_edges:
        for w in WINS:
            for folder in [node_dir / f"W{w}"] + [edge_dir / s / f"W{w}" for s in NETS]:
                for p in folder.glob("*.parquet"):
                    fingerprints[str(p.relative_to(root))] = sha256(p)
    qa["hash_all_snapshots"] = args.hash_all_edges
    qa["hashed_file_count"] = len(fingerprints)
    qa["all_formal_qa_pass"] = bool(
        qa["required_source_missing_count"] == 0 and qa["m1_v1_frozen"]
        and qa["interface_manifest_sha256_match"] and qa["interface_dates_valid_unique"]
        and qa["m1_date_count"] > 0 and qa["missing_required_snapshot_dates"] == 0
        and qa["invalid_edge_filename_dates"] == 0
        and qa["snapshot_schema_or_identity_errors"] == 0
        and qa["sampled_edge_network_window_cells"] == 6
        and qa["day4_master_mapping_consistent"]
        and qa["day6_label_schema_valid"] and qa["day6_label_dates_cover_m1"]
        and qa["day6_label_duplicate_key_count"] == 0)

    status = "SOURCE_AUDIT_PASS" if qa["all_formal_qa_pass"] else "SOURCE_AUDIT_BLOCKED"
    metadata = dict(research_day=15, step="Step1_Frozen_Network_Risk_Source_Audit",
                    status=status, m1_version="M1_v1.0", source_sha256=fingerprints,
                    audit_limit="All snapshot dates inventoried; representative snapshot per network/window fully inspected."
                                " Full edge file SHA256 requires --hash-all-edges.",
                    label_boundary="Day6 next-analysis-date label is NOT automatically an exact 20-trading-day loss.",
                    no_reestimation=True, schema_errors=schema_errors, formal_qa=qa)
    files = {
        "day15_step1_source_inventory.csv": pd.DataFrame(inventory).to_csv(index=False),
        "day15_step1_snapshot_samples.csv": pd.DataFrame(samples).to_csv(index=False),
        "day15_step1_qa.csv": pd.DataFrame([{"qa_name":k, "qa_value":v} for k,v in qa.items()]).to_csv(index=False),
        "day15_step1_metadata.json": json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    }
    oldmeta = out / "day15_step1_metadata.json"
    oldpass = oldmeta.is_file() and json.loads(oldmeta.read_text(encoding="utf-8")).get("status") == "SOURCE_AUDIT_PASS"
    for name, content in files.items():
        write_once(out / name, content, oldpass)
    print(f"Day15 Step1: {status} | M1 dates={len(m1_dates)} | sampled groups={len(samples)}/6")
    print(f"QA: {out / 'day15_step1_qa.csv'}")
    if schema_errors:
        print(f"Schema issues: {len(schema_errors)}")
        for problem in schema_errors[:15]:
            print(" -", problem)
    if not qa["all_formal_qa_pass"]:
        sys.exit(1)

if __name__ == "__main__":
    main()
