
# -*- coding: utf-8 -*-
"""Day18: stock master content audit and source-reference trace."""
import os, re, csv, json, hashlib, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

VERSIONS = ["01_stage1", "01_stage1_fixed", "01_stage1_final"]
SKIP = {".git", ".venv", "venv", "__pycache__", "node_modules", "M1_day18"}
SRC_EXT = {".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
KEYS = ["security_id", "ts_code", "stock_code", "security_code",
        "wind_code", "symbol", "ticker", "stock_id", "security_code", "code"]
PAT = re.compile(
    r"stock_master|01_stage1(?:_fixed|_final)?|master_path|daily_return_panel",
    re.I
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def find_key(df):
    cols = {norm(c): c for c in df.columns}
    return next((cols[norm(k)] for k in KEYS if norm(k) in cols), None)


def clean(s):
    return s.astype("string").str.strip().replace("", pd.NA)


def locate(root, version):
    """Search all project copies; never silently choose an ambiguous path."""
    hits = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP]
        if Path(folder).name == version and "stock_master.parquet" in files:
            hits.append(Path(folder) / "stock_master.parquet")

    hits = sorted(set(p.resolve() for p in hits), key=str)
    if len(hits) == 1:
        return hits[0], "", hits
    if not hits:
        return root / "output" / "M1_day1" / version / "stock_master.parquet", \
               "FILE_NOT_FOUND", hits
    return root / "output" / "M1_day1" / version / "stock_master.parquet", \
           f"AMBIGUOUS_PATHS:{len(hits)}", hits


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    ap.add_argument("--return-panel", default="")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"BLOCKED: project root not found: {root}")

    out = root / "output" / "M1_day18" / "03_stock_master_audit"
    out /= datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)

    masters = []
    for version in VERSIONS:
        path, error, hits = locate(root, version)
        m = {"version": version, "path": path, "df": None, "key": None,
             "hash": "", "error": error, "hits": hits}
        if not error:
            try:
                m["hash"] = sha256(path)
                m["df"] = pd.read_parquet(path)
                m["key"] = find_key(m["df"])
                if m["key"] is None:
                    m["error"] = "NO_STOCK_ID_COLUMN"
            except Exception as e:
                m["error"] = f"READ_ERROR:{e}"
        masters.append(m)

    # A. Per-file schema and key quality
    summary = []
    for m in masters:
        d, p, k = m["df"], m["path"], m["key"]
        summary.append({
            "version": m["version"], "status": m["error"] or "READ_OK",
            "path": str(p), "candidate_paths": " | ".join(map(str, m["hits"])),
            "sha256": m["hash"],
            "size_bytes": p.stat().st_size if p.is_file() else "",
            "rows": len(d) if d is not None else "",
            "columns_json": json.dumps(list(d.columns), ensure_ascii=False)
                            if d is not None else "",
            "dtypes_json": json.dumps({str(c): str(t) for c, t in d.dtypes.items()},
                                       ensure_ascii=False) if d is not None else "",
            "master_index_present": any(norm(c) == "masterindex" for c in d.columns)
                                    if d is not None else "",
            "key_column": k or "",
            "key_missing": int(clean(d[k]).isna().sum()) if d is not None and k else "",
            "key_duplicates": int(clean(d[k]).dropna().duplicated().sum())
                              if d is not None and k else "",
        })

    # B. Pairwise comparison by a stock identity key
    pairs = []
    for i, a in enumerate(masters):
        for b in masters[i + 1:]:
            da, db = a["df"], b["df"]
            r = {
                "version_a": a["version"], "version_b": b["version"],
                "path_a": str(a["path"]), "path_b": str(b["path"]),
                "sha256_equal": (a["hash"] == b["hash"]
                                 if a["hash"] and b["hash"] else ""),
                "rows_a": len(da) if da is not None else "",
                "rows_b": len(db) if db is not None else "",
                "same_columns": "", "key_overlap": "",
                "a_only_keys": "", "b_only_keys": "",
                "shared_columns": "", "rows_with_value_differences": "",
                "note": "",
            }
            if da is None or db is None:
                r["note"] = "READ_ERROR_OR_AMBIGUOUS_PATH"
                pairs.append(r)
                continue

            r["same_columns"] = set(da.columns) == set(db.columns)
            ka, kb = a["key"], b["key"]
            if not ka or not kb or norm(ka) != norm(kb):
                r["note"] = "NO_COMMON_STOCK_KEY"
                pairs.append(r)
                continue

            xa, xb = da.copy(), db.copy()
            xa["__audit_key"], xb["__audit_key"] = clean(da[ka]), clean(db[kb])
            xa = xa.dropna(subset=["__audit_key"]).set_index("__audit_key")
            xb = xb.dropna(subset=["__audit_key"]).set_index("__audit_key")

            if xa.index.has_duplicates or xb.index.has_duplicates:
                r["note"] = "DUPLICATE_KEYS; VALUE_COMPARE_SKIPPED"
                pairs.append(r)
                continue

            sa, sb = set(xa.index), set(xb.index)
            both = sorted(sa & sb)
            r.update(key_overlap=len(both), a_only_keys=len(sa - sb),
                     b_only_keys=len(sb - sa))
            shared = [c for c in da.columns if c in db.columns]
            r["shared_columns"] = len(shared)
            changed = np.zeros(len(both), dtype=bool)

            for c in shared:
                va = xa.loc[both, c].astype("string").fillna("<NA>").reset_index(drop=True)
                vb = xb.loc[both, c].astype("string").fillna("<NA>").reset_index(drop=True)
                changed |= va.ne(vb).to_numpy()

            r["rows_with_value_differences"] = int(changed.sum())
            r["note"] = "KEYED_COMPARE"
            pairs.append(r)

    # C. Static source/config reference trace; output folders are excluded.
    trace = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP and d.lower() != "output"]
        for name in files:
            p = Path(folder) / name
            if p.suffix.lower() not in SRC_EXT:
                continue
            if p.name.lower().startswith(("day18_", "day17_", "day16_")):
                continue
            if p.resolve() == Path(__file__).resolve():
                continue
            try:
                for n, line in enumerate(
                    p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
                ):
                    if PAT.search(line):
                        trace.append({"path": str(p), "line_number": n,
                                      "line": line.strip()[:500]})
            except OSError:
                pass

    write_csv(out / "stock_master_summary.csv", summary, list(summary[0]))
    write_csv(out / "stock_master_pairwise_comparison.csv", pairs, list(pairs[0]))
    write_csv(out / "source_reference_trace.csv", trace, ["path", "line_number", "line"])

    # D. Optional stock coverage against a confirmed return panel
    if args.return_panel:
        panel = Path(args.return_panel).resolve()
        coverage = []
        try:
            ret = pd.read_parquet(panel)
            rk = find_key(ret)
            for m in masters:
                d, k = m["df"], m["key"]
                if d is None or not k or not rk:
                    coverage.append({"version": m["version"], "status": "NO_COMMON_KEY"})
                    continue
                x, y = set(clean(d[k]).dropna()), set(clean(ret[rk]).dropna())
                coverage.append({
                    "version": m["version"], "master_key": k, "panel_key": rk,
                    "master_keys": len(x), "panel_keys": len(y),
                    "overlap": len(x & y), "master_only": len(x - y),
                    "panel_only": len(y - x), "status": "CHECK_REQUIRED",
                })
        except Exception as e:
            coverage.append({"status": f"READ_ERROR:{e}"})
        write_csv(out / "stock_master_return_panel_mapping.csv",
                  coverage, list(coverage[0]))

    report = [
        "# Day18：证券主表比较与源码追踪", "",
        f"- 可读取主表：{sum(m['df'] is not None for m in masters)}/3",
        f"- 源码引用命中：{len(trace)}", "",
        "## 说明",
        "- 标准数据路径为 output/M1_day1 下的对应版本目录。",
        "- 同名文件存在多个路径时，标记歧义并停止内容比较。",
        "- master_index 不直接作为证券身份键。",
        "- 静态源码引用不等于运行时一定读取该文件。",
        "- 本脚本不会自动选择、覆盖或删除正式主表。",
    ]
    (out / "day18_stock_master_audit_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Audit complete | readable={sum(m['df'] is not None for m in masters)}/3")
    print(f"Source references={len(trace)}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
