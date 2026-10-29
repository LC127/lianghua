
# -*- coding: utf-8 -*-
"""Day18: audit Day4 file schemas and identify network edge columns."""
import os
import re
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

SKIP = {".git", ".venv", "venv", "__pycache__", "node_modules", "M1_day18"}

PAIRS = [
    ("source", "target"), ("src", "dst"), ("from", "to"),
    ("node_i", "node_j"), ("node1", "node2"),
    ("stock_i", "stock_j"), ("stock1", "stock2"),
    ("security_id_i", "security_id_j"),
    ("source_id", "target_id"), ("src_id", "dst_id"),
    ("from_id", "to_id"), ("from_stock", "to_stock"),
    ("stock_code_i", "stock_code_j"),
    ("stock_code1", "stock_code2"),
    ("ticker1", "ticker2"), ("code_i", "code_j"),
    ("id1", "id2"), ("stock_x", "stock_y"),
    ("u", "v"), ("i", "j"), ("left", "right"),
]

PATH_WORDS = re.compile(
    r"(^|[\\/_.-])(network|edge|graph|adjacency|pair|link|node)(?=$|[\\/_.-])",
    re.I,
)


def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def find_endpoints(columns):
    cmap = {norm(c): c for c in columns}
    for a, b in PAIRS:
        if norm(a) in cmap and norm(b) in cmap:
            return cmap[norm(a)], cmap[norm(b)]
    return None


def get_schema(path):
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            return pf.schema.names, pf.metadata.num_rows
        except ImportError:
            # 仅对较小文件使用完整读取作为后备方案。
            if path.stat().st_size > 100 * 1024 * 1024:
                raise RuntimeError(
                    "缺少 pyarrow，且文件超过100MB；请安装 pyarrow"
                )
            df = pd.read_parquet(path)
            return list(df.columns), len(df)

    cols = list(pd.read_csv(
        path, nrows=0, encoding="utf-8-sig"
    ).columns)
    return cols, ""


def write_csv(path, rows):
    fields = list(rows[0]) if rows else ["status", "detail"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\M1_StockNetwork")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    day4 = root / "output" / "M1_day4"

    if not day4.is_dir():
        raise SystemExit(f"BLOCKED: Day4目录不存在：{day4}")

    out = root / "output" / "M1_day18" / "05_day4_schema_audit"
    out /= datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)

    inventory, candidates = [], []

    for folder, dirs, files in os.walk(day4):
        dirs[:] = [d for d in dirs if d not in SKIP]

        for name in files:
            p = Path(folder) / name
            if p.suffix.lower() not in {".parquet", ".csv"}:
                continue

            rel = p.relative_to(day4).as_posix()
            path_match = bool(PATH_WORDS.search(rel))

            try:
                columns, nrows = get_schema(p)
                pair = find_endpoints(columns)

                row = {
                    "relative_path": rel,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "rows": nrows,
                    "column_count": len(columns),
                    "columns_json": json.dumps(
                        columns, ensure_ascii=False
                    ),
                    "endpoint_a": pair[0] if pair else "",
                    "endpoint_b": pair[1] if pair else "",
                    "path_suggests_network": path_match,
                    "schema_suggests_edges": bool(pair),
                    "status": "OK",
                }
            except Exception as e:
                row = {
                    "relative_path": rel,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "rows": "",
                    "column_count": "",
                    "columns_json": "",
                    "endpoint_a": "",
                    "endpoint_b": "",
                    "path_suggests_network": path_match,
                    "schema_suggests_edges": False,
                    "status": f"SCHEMA_ERROR: {e}",
                }

            inventory.append(row)
            if path_match or row["schema_suggests_edges"]:
                candidates.append(row)

    write_csv(out / "day4_schema_inventory.csv", inventory)
    write_csv(out / "day4_network_edge_candidates.csv", candidates)

    detected = sum(bool(r["endpoint_a"]) for r in candidates)
    report = [
        "# Day4 网络文件结构审计",
        "",
        f"- Day4目录：`{day4}`",
        f"- CSV/Parquet文件总数：{len(inventory)}",
        f"- 网络相关路径候选数：{sum(bool(r['path_suggests_network']) for r in inventory)}",
        f"- 识别出端点字段组合的文件数：{detected}",
        "",
        "## 输出文件",
        "- day4_schema_inventory.csv：全部文件的字段结构。",
        "- day4_network_edge_candidates.csv：网络相关候选文件。",
        "",
        "如果端点识别数仍为0，请检查候选文件的columns_json，",
        "再根据真实字段名补充PAIRS；不能据此断言网络文件不存在。",
    ]
    (out / "day18_day4_schema_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(
        f"Schema audit complete | files={len(inventory)} | "
        f"candidates={len(candidates)} | endpoint schemas={detected}"
    )
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
