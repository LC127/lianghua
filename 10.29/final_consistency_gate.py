
from pathlib import Path
from datetime import datetime
from collections import Counter
import json
import re

import pandas as pd
import pyarrow.parquet as pq


# =========================================================
# 0. Configuration
# =========================================================
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/M1_day18/05_final_consistency_gate" / (
    f"run_{datetime.now():%Y%m%d_%H%M%S}"
)
OUT.mkdir(parents=True, exist_ok=True)

checks, key_results, network_results = [], [], []


def add(area, status, metric, detail="", path=""):
    checks.append({
        "area": area, "status": status,
        "metric": metric, "detail": str(detail), "path": str(path)
    })


def read_table(path, columns=None):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        for enc in ("utf-8-sig", "gb18030"):
            try:
                kw = {"encoding": enc, "dtype": str, "low_memory": False}
                if columns is not None:
                    kw["usecols"] = columns
                return pd.read_csv(path, **kw)
            except UnicodeDecodeError:
                continue
        raise ValueError("Cannot decode CSV")
    return pd.read_parquet(path, columns=columns)


def columns_of(path):
    if Path(path).suffix.lower() == ".csv":
        return list(read_table(path, columns=[]).columns)
    return pq.ParquetFile(path).schema_arrow.names


def first_existing(paths):
    return next((Path(p) for p in paths if Path(p).is_file()), None)


def sid(s):
    return s.astype("string").str.strip()


def date_key(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    try:
        if re.fullmatch(r"\d{8}", s):
            return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
        return pd.to_datetime(s, errors="raise").strftime("%Y-%m-%d")
    except Exception:
        return None


def window_num(x):
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else None


# =========================================================
# 1. Recheck the 10 known candidate-key files
# =========================================================
KEY_FILES = [
    ("day16_risk",
     "output/M1_day16/02_step2_network_risk_exposure_factors/network_risk_exposure_panel.parquet",
     [["analysis_date", "network", "window", "master_index"],
      ["analysis_date", "network", "window", "security_id"]]),

    ("stock_code_history",
     "output/M1_day1/01_stage1/stock_code_history.parquet",
     [["security_id", "stock_code"],
      ["security_id", "stock_code", "exchange"],
      ["security_id", "stock_code", "list_date"],
      ["security_id", "stock_code", "list_date", "delist_date"]]),

    ("stock_master_issues",
     "output/M1_day1/01_stage1_fixed/stock_master_issues.csv",
     [["security_id", "issue_type", "severity", "note"],
      ["security_id", "stock_code", "issue_type", "severity", "note"]]),

    ("core_data_anomalies",
     "output/M1_day1/02_stage2_daily_market_qa/core_data_anomalies.parquet",
     [["trade_date", "security_id", "issue_type", "severity", "note"],
      ["trade_date", "security_id", "issue_type", "severity"]]),

    ("stock_feasibility",
     "output/M1_day1/04_step4_network_feasibility/stock_level_feasibility_summary.parquet",
     [["window", "security_id", "min_valid_ratio"]]),

    ("target_weights",
     "output/M1_day8/02_stage2_trade_reconstruction/portfolio_target_weights.parquet",
     [["analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
      ["analysis_date", "window", "factor_name", "method", "security_id"]]),

    ("trade_weights",
     "output/M1_day8/02_stage2_trade_reconstruction/portfolio_trade_weight_panel.parquet",
     [["analysis_date", "previous_analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
      ["analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
      ["analysis_date", "window", "factor_name", "method", "security_id"]]),

    ("dedicated_drift",
     "output/M1_day8/02c_stage2_rerun_with_dedicated_drift/portfolio_trade_weight_panel_dedicated_drift.parquet",
     [["analysis_date", "previous_analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
      ["analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
      ["analysis_date", "window", "factor_name", "method", "security_id"]]),

    ("canonical_trade",
     "output/M1_day8/02e_stage2_final_canonical_trade_reconstruction/canonical_portfolio_trade_weight_panel.parquet",
     [["analysis_date", "previous_analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
      ["analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
      ["analysis_date", "window", "factor_name", "method", "security_id"]]),

    ("participation_base",
     "output/M1_day8/03_stage3_participation_capacity/stock_participation_base_panel.parquet",
     [["analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
      ["analysis_date", "window", "factor_name", "method", "security_id"]]),
]


def test_keys(name, path, candidates):
    if not path.is_file():
        for i, key in enumerate(candidates, 1):
            key_results.append({
                "file": name, "candidate": f"K{i}",
                "key": "|".join(key), "status": "BLOCKED",
                "null_key_rows": "", "duplicate_rows": "",
                "detail": "File not found"
            })
        return

    cols = columns_of(path)
    needed = list(dict.fromkeys(c for k in candidates for c in k if c in cols))
    df = read_table(path, needed) if needed else pd.DataFrame()

    for i, key in enumerate(candidates, 1):
        missing = [c for c in key if c not in cols]
        r = {
            "file": name, "candidate": f"K{i}", "key": "|".join(key),
            "null_key_rows": "", "duplicate_rows": "", "detail": ""
        }
        if missing:
            r.update(status="REVIEW", detail=f"Missing columns: {missing}")
        else:
            nulls = int(df[key].isna().any(axis=1).sum())
            dups = int(df.duplicated(key).sum())
            r.update(
                status="PASS" if nulls == 0 and dups == 0 else "REVIEW",
                null_key_rows=nulls, duplicate_rows=dups
            )
        key_results.append(r)


for name, rel, candidates in KEY_FILES:
    test_keys(name, ROOT / rel, candidates)


# =========================================================
# 2. Generate two non-destructive cleanup candidates
# =========================================================
hist_path = ROOT / "output/M1_day1/01_stage1/stock_code_history.parquet"

if hist_path.is_file():
    hist = pd.read_parquet(hist_path)
    identity = ["security_id", "stock_code"]
    dup = hist[hist.duplicated(identity, keep=False)]
    missing = hist[hist[identity].isna().any(axis=1)]

    dup.to_csv(OUT / "stock_code_history_duplicates.csv",
               index=False, encoding="utf-8-sig")
    missing.to_csv(OUT / "stock_code_history_missing_identity.csv",
                   index=False, encoding="utf-8-sig")

    # Only propose dropping a null-list_date row when all other
    # fields match and the same identity has exactly one valid date.
    drop_idx = []
    if "list_date" in hist.columns:
        for _, g in hist.groupby(identity, dropna=False, sort=False):
            if len(g) < 2:
                continue
            valid = int(g["list_date"].notna().sum())
            null = int(g["list_date"].isna().sum())
            other = g.drop(columns=["list_date"])
            same_other = bool((other.nunique(dropna=False) <= 1).all())
            if valid == 1 and null == len(g) - 1 and same_other:
                drop_idx.extend(g.index[g["list_date"].isna()].tolist())

    hist_candidate = hist.drop(index=drop_idx)
    hist_candidate.to_parquet(
        OUT / "stock_code_history_candidate.parquet", index=False
    )

    add(
        "source_exception",
        "REVIEW" if len(dup) or len(missing) else "PASS",
        "stock_code_history",
        f"duplicate_group_rows={len(dup)}; missing_identity_rows={len(missing)}; "
        f"candidate_rows_removed={len(drop_idx)}. Candidate only; source unchanged.",
        hist_path
    )
else:
    add("source_exception", "BLOCKED", "stock_code_history", "File not found", hist_path)


issues_path = ROOT / "output/M1_day1/01_stage1_fixed/stock_master_issues.csv"

if issues_path.is_file():
    issues = read_table(issues_path)
    exact_excess = int(issues.duplicated().sum())
    duplicate_rows = issues[issues.duplicated(keep=False)]

    duplicate_rows.to_csv(
        OUT / "stock_master_issues_exact_duplicates.csv",
        index=False, encoding="utf-8-sig"
    )
    issues.drop_duplicates().to_csv(
        OUT / "stock_master_issues_deduplicated_candidate.csv",
        index=False, encoding="utf-8-sig"
    )

    add(
        "source_exception",
        "REVIEW" if exact_excess else "PASS",
        "stock_master_issues",
        f"original_rows={len(issues)}; exact_duplicate_excess={exact_excess}; "
        f"candidate_rows={len(issues.drop_duplicates())}. Source unchanged.",
        issues_path
    )
else:
    add("source_exception", "BLOCKED", "stock_master_issues", "File not found", issues_path)


# =========================================================
# 3. Final master, global mapping and return-panel coverage
# =========================================================
master_path = first_existing([
    ROOT / "output/M1_day1/01_stage1_final/stock_master.parquet",
    ROOT / "output/M1_day1/01_stage1/stock_master.parquet",
    ROOT / "output/M1_day1/01_stage1_fixed/stock_master.parquet",
])
mapping_path = ROOT / "output/M1_day4/02_step2_matched_networks/stock_master_mapping.parquet"
return_path = ROOT / "output/M1_day1/03_stage3_return_validation/daily_return_panel.parquet"

master_ids = set()
mapping_ids = set()
mapping_by_index = {}
mapping_df = None

if master_path and mapping_path.is_file():
    master = read_table(master_path, ["security_id"])
    mapping_df = read_table(mapping_path, ["security_id", "master_index"])

    master_series = sid(master["security_id"])
    map_series = sid(mapping_df["security_id"])
    idx_series = sid(mapping_df["master_index"])

    master_ids = set(master_series.dropna())
    mapping_ids = set(map_series.dropna())
    mapping_by_index = dict(zip(idx_series.dropna(), map_series.dropna()))

    master_dups = int(master_series.duplicated().sum())
    map_id_dups = int(map_series.duplicated().sum())
    map_idx_dups = int(idx_series.duplicated().sum())
    only_master = master_ids - mapping_ids
    only_mapping = mapping_ids - master_ids

    pd.DataFrame(
        [{"difference": "FINAL_ONLY", "security_id": x} for x in sorted(only_master)]
        + [{"difference": "MAPPING_ONLY", "security_id": x} for x in sorted(only_mapping)]
    ).to_csv(OUT / "final_mapping_id_differences.csv",
             index=False, encoding="utf-8-sig")

    ok = not (master_dups or map_id_dups or map_idx_dups or only_master or only_mapping)
    add(
        "final_mapping", "PASS" if ok else "REVIEW",
        "master_vs_mapping",
        f"master={len(master_ids)}; mapping={len(mapping_ids)}; "
        f"master_duplicate_rows={master_dups}; mapping_duplicate_ids={map_id_dups}; "
        f"mapping_duplicate_indices={map_idx_dups}; final_only={len(only_master)}; "
        f"mapping_only={len(only_mapping)}",
        mapping_path
    )
else:
    add("final_mapping", "BLOCKED", "master_vs_mapping",
        "Final master or Day4 mapping file not found", mapping_path)

if return_path.is_file() and master_ids:
    ret = read_table(return_path, ["security_id"])
    return_ids = set(sid(ret["security_id"]).dropna())
    final_only = master_ids - return_ids
    return_only = return_ids - master_ids

    pd.DataFrame(
        [{"difference": "FINAL_WITHOUT_RETURN", "security_id": x}
         for x in sorted(final_only)]
        + [{"difference": "RETURN_OUTSIDE_FINAL", "security_id": x}
           for x in sorted(return_only)]
    ).to_csv(OUT / "final_return_id_differences.csv",
             index=False, encoding="utf-8-sig")

    add(
        "return_coverage",
        "PASS" if not final_only and not return_only else "REVIEW",
        "master_vs_return_panel",
        f"final_ids={len(master_ids)}; return_ids={len(return_ids)}; "
        f"final_without_return={len(final_only)}; return_outside_final={len(return_only)}. "
        "Differences require interpretation; no automatic deletion.",
        return_path
    )
else:
    add("return_coverage", "BLOCKED", "master_vs_return_panel",
        "Return panel or Final master unavailable", return_path)


# =========================================================
# 4. Day4 analysis-date coverage and node mapping
# =========================================================
day4 = ROOT / "output/M1_day4"
schedule_path = day4 / "01_step1_network_design/day4_analysis_dates.csv"
network_root = day4 / "02_step2_matched_networks"
node_root = network_root / "node_universe"
raw_root = network_root / "edge_master/B0_RAW_POSITIVE"
res_root = network_root / "edge_master/M1_RESIDUAL_POSITIVE"

def path_map(root):
    if not root.is_dir():
        return {}
    return {p.relative_to(root).as_posix(): p
            for p in root.rglob("*.parquet")}


nodes = path_map(node_root)
raw_files = path_map(raw_root)
res_files = path_map(res_root)
node_indices, node_bad = {}, {}
actual_pairs = set()

for rel, p in nodes.items():
    win = window_num(Path(rel).parent.name)
    dt = date_key(Path(rel).stem)
    if win is not None and dt:
        actual_pairs.add((win, dt))

    try:
        ndf = read_table(p, ["master_index", "security_id"])
        idx = sid(ndf["master_index"])
        sec = sid(ndf["security_id"])
        node_indices[rel] = set(idx.dropna())

        expected_sec = idx.map(mapping_by_index) if mapping_by_index else pd.Series(
            [None] * len(idx), index=idx.index, dtype="object"
        )
        bad = int((expected_sec.isna() | (sec != expected_sec)).sum())
        node_bad[rel] = bad
    except Exception:
        node_indices[rel] = set()
        node_bad[rel] = -1

add(
    "day4_nodes",
    "PASS" if nodes and all(v == 0 for v in node_bad.values()) else "REVIEW",
    "node_mapping",
    f"node_snapshots={len(nodes)}; snapshots_with_mapping_errors="
    f"{sum(v != 0 for v in node_bad.values())}; "
    f"mapping_error_rows={sum(max(v, 0) for v in node_bad.values())}",
    node_root
)

coverage_rows = []
if schedule_path.is_file() and nodes:
    sched = read_table(schedule_path)
    norm_cols = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in sched.columns}
    date_col = next((norm_cols[x] for x in
                     ["analysisdate", "date", "tradedate"] if x in norm_cols), None)
    win_col = next((norm_cols[x] for x in
                    ["window", "windowdays", "windowlength"] if x in norm_cols), None)

    if date_col:
        schedule_dates = {
            date_key(x) for x in sched[date_col].dropna()
            if date_key(x) is not None
        }
        expected_pairs = set()

        if win_col:
            for _, row in sched.iterrows():
                dt = date_key(row[date_col])
                w = window_num(row[win_col])
                if dt and w:
                    expected_pairs.add((w, dt))

        if expected_pairs:
            missing_pairs = expected_pairs - actual_pairs
            extra_pairs = actual_pairs - expected_pairs
            for w, dt in sorted(expected_pairs | actual_pairs):
                coverage_rows.append({
                    "window": w, "analysis_date": dt,
                    "expected": (w, dt) in expected_pairs,
                    "observed": (w, dt) in actual_pairs
                })
            coverage_ok = not missing_pairs and not extra_pairs
            detail = (
                f"expected_pairs={len(expected_pairs)}; observed_pairs={len(actual_pairs)}; "
                f"missing_pairs={len(missing_pairs)}; extra_pairs={len(extra_pairs)}"
            )
        else:
            observed_dates = {dt for _, dt in actual_pairs}
            missing_dates = schedule_dates - observed_dates
            extra_dates = observed_dates - schedule_dates
            for dt in sorted(schedule_dates | observed_dates):
                coverage_rows.append({
                    "window": "", "analysis_date": dt,
                    "expected": dt in schedule_dates,
                    "observed": dt in observed_dates
                })
            coverage_ok = not missing_dates and not extra_dates
            detail = (
                f"schedule_dates={len(schedule_dates)}; observed_dates={len(observed_dates)}; "
                f"dates_without_any_snapshot={len(missing_dates)}; "
                f"dates_outside_schedule={len(extra_dates)}; "
                "No window column: checked date coverage across all windows only."
            )

        add("day4_schedule",
            "PASS" if coverage_ok else "REVIEW",
            "schedule_vs_nodes", detail, schedule_path)
    else:
        add("day4_schedule", "REVIEW", "schedule_vs_nodes",
            f"Could not infer date column; columns={list(sched.columns)}", schedule_path)
else:
    add("day4_schedule", "BLOCKED", "schedule_vs_nodes",
        "Schedule or node snapshots not found", schedule_path)

pd.DataFrame(coverage_rows).to_csv(
    OUT / "day4_date_coverage.csv", index=False, encoding="utf-8-sig"
)


# =========================================================
# 5. Pairwise B0 vs M1 edge audit
# =========================================================
def edge_stats(path, node_idx):
    df = read_table(path, ["master_index_i", "master_index_j"])
    a = sid(df["master_index_i"])
    b = sid(df["master_index_j"])
    valid = a.notna() & b.notna()

    pairs = [
        tuple(sorted((x, y)))
        for x, y in zip(a[valid].tolist(), b[valid].tolist())
    ]
    pair_set = set(pairs)
    loops = sum(x == y for x, y in pairs)
    duplicates = len(pairs) - len(pair_set)
    unknown = (set(a[valid]) | set(b[valid])) - node_idx
    missing_rows = int((~valid).sum())

    n = len(node_idx)
    possible = n * (n - 1) / 2
    nonself = sum(x != y for x, y in pair_set)
    density = nonself / possible if possible else None

    return {
        "rows": len(df), "unique_edges": len(pair_set),
        "self_loops": loops, "duplicate_pairs": duplicates,
        "missing_endpoint_rows": missing_rows,
        "unknown_endpoint_ids": len(unknown),
        "density": density, "pairs": pair_set
    }


all_edge_rel = sorted(set(raw_files) | set(res_files))
for rel in all_edge_rel:
    rp, sp = raw_files.get(rel), res_files.get(rel)
    npth = nodes.get(rel)
    row = {"relative_path": rel, "status": "REVIEW"}

    try:
        if not rp or not sp or not npth:
            raise FileNotFoundError("Missing raw/residual edge file or node snapshot")

        idx = node_indices.get(rel, set())
        if not idx:
            raise ValueError("Node index set is empty or unavailable")

        a = edge_stats(rp, idx)
        b = edge_stats(sp, idx)

        union = a["pairs"] | b["pairs"]
        overlap = a["pairs"] & b["pairs"]
        jaccard = len(overlap) / len(union) if union else 1.0
        counts_equal = a["rows"] == b["rows"]

        ok = (
            counts_equal
            and node_bad.get(rel, -1) == 0
            and all(a[k] == 0 and b[k] == 0 for k in [
                "self_loops", "duplicate_pairs",
                "missing_endpoint_rows", "unknown_endpoint_ids"
            ])
        )

        row.update({
            "window": window_num(Path(rel).parent.name),
            "analysis_date": date_key(Path(rel).stem),
            "node_count": len(idx),
            "raw_edges": a["rows"], "residual_edges": b["rows"],
            "raw_density": a["density"], "residual_density": b["density"],
            "edge_counts_equal": counts_equal,
            "raw_self_loops": a["self_loops"],
            "residual_self_loops": b["self_loops"],
            "raw_duplicate_pairs": a["duplicate_pairs"],
            "residual_duplicate_pairs": b["duplicate_pairs"],
            "raw_unknown_endpoint_ids": a["unknown_endpoint_ids"],
            "residual_unknown_endpoint_ids": b["unknown_endpoint_ids"],
            "overlap_edges": len(overlap), "jaccard": jaccard,
            "status": "PASS" if ok else "REVIEW"
        })
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"

    network_results.append(row)

pd.DataFrame(network_results).to_csv(
    OUT / "day4_network_pair_audit.csv",
    index=False, encoding="utf-8-sig"
)

add(
    "day4_edges",
    "PASS" if network_results and all(
        r.get("status") == "PASS" for r in network_results
    ) else "REVIEW",
    "raw_residual_edge_audit",
    f"raw_files={len(raw_files)}; residual_files={len(res_files)}; "
    f"paired_results={len(network_results)}; "
    f"review_pairs={sum(r.get('status') != 'PASS' for r in network_results)}",
    network_root
)


# =========================================================
# 6. One final decision and all reports
# =========================================================
pd.DataFrame(key_results).to_csv(
    OUT / "candidate_key_recheck.csv", index=False, encoding="utf-8-sig"
)
pd.DataFrame(checks).to_csv(
    OUT / "final_gate_checks.csv", index=False, encoding="utf-8-sig"
)

counts = dict(Counter(x["status"] for x in checks))
gate = (
    "BLOCKED" if counts.get("BLOCKED", 0)
    else "REVIEW" if counts.get("REVIEW", 0)
    else "PASS"
)

summary = {
    "gate": gate,
    "checks": counts,
    "candidate_key_tests": len(key_results),
    "day4_node_snapshots": len(nodes),
    "day4_raw_edge_files": len(raw_files),
    "day4_residual_edge_files": len(res_files),
    "day4_edge_pairs_checked": len(network_results),
    "output_dir": str(OUT),
    "source_files_modified": False
}

(OUT / "day18_final_gate_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\nReports saved to:", OUT)
