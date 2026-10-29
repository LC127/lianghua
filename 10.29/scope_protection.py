
# -*- coding: utf-8 -*-
"""Day18 Step1: 只读审计范围与原始文件清单。"""
import argparse, csv, json, os
from datetime import datetime
from pathlib import Path

REQUIRED = {
    "Day1": ["stock_master.parquet", "daily_return_panel.parquet",
             "return_validation_and_panel.py"],
    "Day2": ["freeze_baseline_design.py"],
    "Day3": ["market_residual_correlation.py"],
    "Day4": ["construct_matched_networks.py",
             "stock_master_mapping.parquet"],
}
REFERENCE = {
    "Earlier_QA": ["code_version_audit.py",
                   "tail_event_high_low_validation.py",
                   "frozen_network_risk_source_audit.py",
                   "stock_downside_target_freeze.py",
                   "network_edge_codistress.py",
                   "continuous_risk_state_validation.py"],
}
SKIP = {".git", ".venv", "venv", "__pycache__", "node_modules"}


def find_files(root, names, exclude):
    result = {n: [] for n in names}
    if not root.is_dir():
        return result
    exclude = exclude.resolve()
    for folder, dirs, files in os.walk(root):
        here = Path(folder)
        dirs[:] = [
            d for d in dirs
            if d not in SKIP
            and (here / d).resolve() != exclude
            and exclude not in (here / d).resolve().parents
        ]
        for name in names.intersection(files):
            result[name].append(here / name)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Day18 Step1: read-only scope audit"
    )
    parser.add_argument("--root", default=r"D:\M1_StockNetwork")
    parser.add_argument("--legacy-root", default=r"D:\M1_StockNetwork\output")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    legacy = Path(args.legacy_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"BLOCKED: project root not found: {root}")

    audit_base = root / "output" / "M1_day18"
    out = audit_base / "01_step1_scope_protection" / datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )
    out.mkdir(parents=True, exist_ok=False)

    groups = {**REQUIRED, **REFERENCE}
    names = {n for group in groups.values() for n in group}
    project_files = find_files(root, names, audit_base)
    legacy_files = find_files(legacy, names, audit_base)

    rows, missing = [], []
    for group, expected in groups.items():
        required = group in REQUIRED
        for name in expected:
            matches = {
                str(f): ("project_root", f)
                for f in project_files[name]
            }
            matches.update({
                str(f): ("legacy_root", f)
                for f in legacy_files[name]
            })

            if not matches:
                rows.append([
                    group, name, "", "", "", "",
                    "MISSING", "REQUIRED" if required else "REFERENCE"
                ])
                if required:
                    missing.append(f"{group}: {name}")

            for path, (source, file) in sorted(matches.items()):
                stat = file.stat()
                rows.append([
                    group, name, source, path, stat.st_size,
                    datetime.fromtimestamp(stat.st_mtime).isoformat(
                        timespec="seconds"
                    ),
                    "FOUND", "REQUIRED" if required else "REFERENCE"
                ])

    with (out / "day18_step1_inventory.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "group", "expected_name", "root", "path", "size_bytes",
            "modified_time", "status", "requirement"
        ])
        writer.writerows(rows)

    status = (
        "SCOPE_READY_REVIEW_REQUIRED"
        if missing or not legacy.is_dir()
        else "SCOPE_READY"
    )
    record = {
        "step": "Day18 Step1",
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(root),
        "legacy_root": str(legacy),
        "legacy_root_exists": legacy.is_dir(),
        "allowed_write_directory": str(out),
        "required_files_missing": missing,
        "protection": (
            "仅读取源文件路径和元数据；不修改、移动、重命名或删除源文件。"
        ),
        "scope_note": (
            "本步骤仅登记审计范围；SHA256和数据内容核验属于后续步骤。"
        ),
    }
    (out / "day18_step1_scope.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# Day18 Step1：审计范围与原始文件保护",
        "",
        f"- 状态：`{status}`",
        f"- 项目目录：`{root}`",
        f"- 历史/上游目录：`{legacy}`（存在：{legacy.is_dir()}）",
        f"- 本次写入目录：`{out}`",
        f"- 必需文件缺失数：{len(missing)}",
        "",
        "## 保护规则",
        "- 原始文件只读取路径、大小和修改时间。",
        "- 不修改、移动、重命名或删除原始文件。",
        "- 本步骤不判定 SHA256 是否一致。",
        "- MISSING 仅表示未找到该文件名，需要核实真实路径。",
        "",
        "## 必需文件缺失",
        *([f"- {x}" for x in missing] if missing else ["- 无"]),
    ]
    (out / "day18_step1_scope_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(f"Day18 Step1: {status}")
    print(f"Output: {out}")
    if missing:
        print("Required files to review:\n - " + "\n - ".join(missing))


if __name__ == "__main__":
    main()
