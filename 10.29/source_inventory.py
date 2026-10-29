# -*- coding: utf-8 -*-
"""Day18 Step2 v2: inventory sources and compare only logical version candidates."""
import argparse, csv, hashlib, os, re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

COMPARE = {
    "stock_master.parquet", "raw_universe_by_date.parquet",
    "stock_code_history.parquet", "stock_master_issues.csv",
    "stock_master_quality_summary.csv", "core_data_anomalies.parquet",
    "universe_mismatch_detail.parquet", "forward_label_panel.parquet",
    "network_edge_codistress_daily.csv",
    "network_edge_codistress_hac_summary.csv",
}
HASH_NAMES = COMPARE | {
    "daily_return_panel.parquet", "stock_master_mapping.parquet",
    "return_validation_and_panel.py", "freeze_baseline_design.py",
    "market_residual_correlation.py", "construct_matched_networks.py",
}
KEYS = ("stock_master", "daily_return_panel", "mapping", "return_validation",
        "freeze_baseline", "market_residual", "construct_matched", "network",
        "code_version_audit", "tail_event", "risk_source_audit",
        "downside_target", "continuous_risk", "day18_step1")
SKIP = {".git", ".venv", "venv", "__pycache__", "node_modules"}
DATE_PART = re.compile(r"^\d{4}-\d{2}-\d{2}\.parquet$", re.I)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scan(root, excluded):
    found = []
    if not root.is_dir():
        return found
    excluded = excluded.resolve()
    for folder, dirs, names in os.walk(root):
        here = Path(folder)
        dirs[:] = [d for d in dirs if d not in SKIP and
                   (here / d).resolve() != excluded and
                   excluded not in (here / d).resolve().parents]
        for name in names:
            p = here / name
            ext = p.suffix.lower()
            if ext in {".parquet", ".py"} or (
                ext in {".csv", ".json"} and
                any(k in name.lower() for k in KEYS)
            ):
                found.append(p)
    return found


def main():
    ap = argparse.ArgumentParser(description="Day18 Step2 v2 source inventory")
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    ap.add_argument("--extra-root", default="", help="Optional independent source root")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    extra = Path(args.extra_root).resolve() if args.extra_root else None
    if not root.is_dir():
        raise SystemExit(f"BLOCKED: project root not found: {root}")
    if extra and not extra.is_dir():
        print(f"WARNING: extra-root not found, ignored: {extra}")
        extra = None

    base = root / "output" / "M1_day18"
    out = base / "02_step2_source_inventory" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)

    paths = scan(root, base)
    # Do not scan a nested/overlapping root twice.
    if extra and extra != root and extra not in root.parents and root not in extra.parents:
        paths += scan(extra, base)
    paths = sorted({str(p.resolve()).lower(): p for p in paths}.values(),
                   key=lambda p: str(p).lower())

    records, groups = [], defaultdict(list)
    for p in paths:
        try:
            s = p.stat()
            digest = ""
            if p.name.lower() in HASH_NAMES:
                try:
                    digest = sha256(p)
                except OSError as e:
                    digest = f"READ_ERROR: {e}"
            row = {"file_name": p.name, "path": str(p.resolve()),
                   "size_bytes": s.st_size,
                   "modified_time": datetime.fromtimestamp(s.st_mtime).isoformat(timespec="seconds"),
                   "sha256": digest}
            records.append(row)
            groups[p.name.lower()].append(row)
        except OSError as e:
            records.append({"file_name": p.name, "path": str(p),
                            "size_bytes": "", "modified_time": "",
                            "sha256": f"READ_ERROR: {e}"})

    candidates = []
    for name in sorted(COMPARE):
        items = groups.get(name, [])
        if len(items) < 2:
            continue
        if any(x["sha256"].startswith("READ_ERROR:") or not x["sha256"] for x in items):
            verdict = "REVIEW_HASH_ERROR"
        else:
            verdict = ("EXACT_DUPLICATE" if len({x["sha256"] for x in items}) == 1
                       else "SAME_NAME_DIFFERENT_CONTENT")
        for item in items:
            candidates.append({"file_name": name, "verdict": verdict,
                               "candidate_count": len(items), **item})

    fields = ["file_name", "path", "size_bytes", "modified_time", "sha256"]
    with (out / "day18_source_inventory.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    with (out / "day18_duplicate_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "verdict", "candidate_count"] + fields[1:])
        w.writeheader()
        w.writerows(candidates)

    date_groups = {n: xs for n, xs in groups.items() if DATE_PART.match(n) and len(xs) > 1}
    report = (
        "# Day18 Step2 v2：数据源与逻辑版本候选核验\n\n"
        f"- 文件清单记录数：{len(records)}\n"
        f"- 进入逻辑版本比较的候选记录数：{len(candidates)}\n"
        f"- 被排除出版本比较的重复日期分片组数：{len(date_groups)}\n"
        f"- 日期分片涉及文件记录数：{sum(len(x) for x in date_groups.values())}\n\n"
        "## 判定规则\n"
        "- 只对预先列出的逻辑数据对象比较同名候选版本。\n"
        "- 按日期命名的 Parquet 文件仍保留在总清单中，但不跨目录直接比较；它们可能属于不同面板或不同网络阶段。\n"
        "- `EXACT_DUPLICATE` 表示 SHA256 相同；`SAME_NAME_DIFFERENT_CONTENT` 表示同一逻辑对象的候选版本内容不同。\n"
        "- 本脚本不自动选择权威版本，不修改、移动或删除任何源文件。\n\n"
        "## 输出\n"
        "- `day18_source_inventory.csv`：项目文件清单；指定关键文件附 SHA256。\n"
        "- `day18_duplicate_candidates.csv`：逻辑版本候选比较结果。\n"
    )
    (out / "day18_step2_report.md").write_text(report, encoding="utf-8")

    print(f"Day18 Step2: INVENTORY_COMPLETE | files={len(records)} | candidate rows={len(candidates)}")
    print(f"Date-partition groups excluded from comparison: {len(date_groups)}")
    print(f"Inventory : {out / 'day18_source_inventory.csv'}")
    print(f"Candidates: {out / 'day18_duplicate_candidates.csv'}")
    print(f"Report    : {out / 'day18_step2_report.md'}")


if __name__ == "__main__":
    main()
