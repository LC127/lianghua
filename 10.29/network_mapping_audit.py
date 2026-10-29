
# -*- coding: utf-8 -*-
"""Day18: Final master, node-universe and network-edge consistency audit."""
import csv
from pathlib import Path
from datetime import datetime
import pandas as pd


def canon(s):
    return (s.astype("string").str.strip()
            .str.replace(r"\.0$", "", regex=True).replace("", pd.NA))


def date_ok(df, expected):
    if "analysis_date" not in df:
        return False
    d = pd.to_datetime(df["analysis_date"], errors="coerce")
    return d.notna().all() and set(d.dt.strftime("%Y-%m-%d")) == {expected}


def save(path, rows):
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    root = Path(r"D:\M1_StockNetwork")
    day4 = root / "output" / "M1_day4"
    master_p = root / "output" / "M1_day1" / "01_stage1_final" / "stock_master.parquet"
    map_p = day4 / "02_step2_matched_networks" / "stock_master_mapping.parquet"
    node_root = day4 / "02_step2_matched_networks" / "node_universe"
    edge_root = day4 / "02_step2_matched_networks" / "edge_master"

    for p in [master_p, map_p, node_root, edge_root]:
        if not p.exists():
            raise SystemExit(f"BLOCKED: required input missing: {p}")

    out = root / "output" / "M1_day18" / "06_final_network_mapping_audit"
    out = out / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)

    # 1. Final master and global security-index mapping
    master = pd.read_parquet(master_p, columns=["security_id"])
    mapping = pd.read_parquet(map_p, columns=["security_id", "master_index"])

    final_ids = canon(master["security_id"])
    map_ids = canon(mapping["security_id"])
    map_idx = canon(mapping["master_index"])

    final_set = set(final_ids.dropna())
    map_id_set = set(map_ids.dropna())
    map_idx_set = set(map_idx.dropna())
    id_to_idx = dict(zip(map_ids.dropna(), map_idx.dropna()))

    global_check = [{
        "final_count": len(final_set),
        "mapping_count": len(map_id_set),
        "overlap": len(final_set & map_id_set),
        "final_only": len(final_set - map_id_set),
        "mapping_only": len(map_id_set - final_set),
        "final_missing_ids": int(final_ids.isna().sum()),
        "mapping_missing_ids": int(map_ids.isna().sum()),
        "mapping_missing_index": int(map_idx.isna().sum()),
        "mapping_duplicate_ids": int(map_ids.dropna().duplicated().sum()),
        "mapping_duplicate_indices": int(map_idx.dropna().duplicated().sum()),
    }]
    save(out / "global_master_mapping_check.csv", global_check)

    # 2. Check each date-specific node universe
    node_files = sorted(node_root.rglob("*.parquet"))
    node_checks, node_indices = [], {}

    for p in node_files:
        window, date = p.parent.name, p.stem
        try:
            df = pd.read_parquet(
                p, columns=["master_index", "security_id", "analysis_date", "window"]
            )
            ids, idx = canon(df["security_id"]), canon(df["master_index"])
            expected_idx = ids.map(id_to_idx)
            mismatch = expected_idx.isna() | idx.ne(expected_idx)

            unknown_final = len(set(ids.dropna()) - final_set)
            unknown_mapping = len(set(ids.dropna()) - map_id_set)
            dup_ids = int(ids.dropna().duplicated().sum())
            dup_idx = int(idx.dropna().duplicated().sum())
            d_ok = date_ok(df, date)
            w_ok = set(canon(df["window"]).dropna()) == {window}

            node_indices[(window, date)] = set(idx.dropna())
            status = (
                "PASS" if not any([
                    unknown_final, unknown_mapping, int(mismatch.sum()),
                    dup_ids, dup_idx, not d_ok, not w_ok
                ]) else "REVIEW"
            )
            node_checks.append({
                "file": str(p), "window": window, "date": date,
                "rows": len(df), "unique_security_ids": ids.nunique(),
                "unknown_final_ids": unknown_final,
                "unknown_mapping_ids": unknown_mapping,
                "index_mismatch_rows": int(mismatch.sum()),
                "duplicate_security_ids": dup_ids,
                "duplicate_indices": dup_idx,
                "date_match": d_ok, "window_match": w_ok, "status": status,
            })
        except Exception as e:
            node_checks.append({
                "file": str(p), "window": window, "date": date,
                "status": f"READ_ERROR: {e}"
            })

    save(out / "node_universe_checks.csv", node_checks)

    # 3. Check Raw and Residual edge endpoints against mapping and date nodes
    edge_files = sorted(edge_root.rglob("*.parquet"))
    edge_checks = []

    for p in edge_files:
        window, date = p.parent.name, p.stem
        family = p.parent.parent.name
        try:
            df = pd.read_parquet(
                p, columns=[
                    "master_index_i", "master_index_j", "analysis_date", "window"
                ]
            )
            a, b = canon(df["master_index_i"]), canon(df["master_index_j"])
            valid = a.notna() & b.notna()
            endpoints = set(pd.concat([a, b]).dropna())
            node_set = node_indices.get((window, date))

            unknown_map = len(endpoints - map_idx_set)
            unknown_node = len(endpoints - node_set) if node_set is not None else ""
            self_loops = int((a[valid] == b[valid]).sum())

            av = a[valid].reset_index(drop=True)
            bv = b[valid].reset_index(drop=True)
            lo, hi = av.where(av <= bv, bv), bv.where(av <= bv, av)
            duplicate_pairs = int(pd.DataFrame({
                "lo": lo, "hi": hi
            }).duplicated().sum())

            d_ok = date_ok(df, date)
            w_ok = set(canon(df["window"]).dropna()) == {window}
            missing_rows = int((~valid).sum())

            status = (
                "PASS" if node_set is not None and not any([
                    unknown_map, unknown_node, self_loops,
                    duplicate_pairs, missing_rows, not d_ok, not w_ok
                ]) else "REVIEW"
            )
            edge_checks.append({
                "file": str(p), "network_family": family,
                "window": window, "date": date, "rows": len(df),
                "unique_endpoint_indices": len(endpoints),
                "missing_endpoint_rows": missing_rows,
                "unknown_mapping_indices": unknown_map,
                "unknown_date_node_indices": unknown_node,
                "self_loops": self_loops,
                "duplicate_undirected_pairs": duplicate_pairs,
                "date_match": d_ok, "window_match": w_ok,
                "node_snapshot_found": node_set is not None,
                "status": status,
            })
        except Exception as e:
            edge_checks.append({
                "file": str(p), "network_family": family,
                "window": window, "date": date,
                "status": f"READ_ERROR: {e}"
            })

    save(out / "edge_endpoint_checks.csv", edge_checks)

    report = [
        "# Day18 Final 网络节点一致性核验", "",
        f"- Final 唯一证券数：{len(final_set)}",
        f"- 节点快照文件数：{len(node_files)}",
        f"- 网络边文件数：{len(edge_files)}",
        f"- 节点快照 PASS：{sum(x.get('status') == 'PASS' for x in node_checks)}",
        f"- 节点快照 REVIEW/ERROR：{sum(x.get('status') != 'PASS' for x in node_checks)}",
        f"- 边文件 PASS：{sum(x.get('status') == 'PASS' for x in edge_checks)}",
        f"- 边文件 REVIEW/ERROR：{sum(x.get('status') != 'PASS' for x in edge_checks)}",
        "", "## 输出",
        "- global_master_mapping_check.csv：Final 与全局证券映射。",
        "- node_universe_checks.csv：逐日期节点身份和索引检查。",
        "- edge_endpoint_checks.csv：逐日期边端点检查。",
        "", "PASS 仅表示通过本脚本列出的结构与映射检查；",
        "不能单独证明网络经济含义正确或正式冻结所需的所有条件均已满足。",
    ]
    (out / "day18_final_network_mapping_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("Audit complete")
    print(f"Nodes: {len(node_files)} | Edges: {len(edge_files)}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
