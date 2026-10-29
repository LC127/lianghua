
from pathlib import Path
from datetime import datetime
from collections import Counter
import hashlib
import json
import re

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 1. Paths and configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

MANIFEST = (
    ROOT
    / "output/M1_day18/03_step3_frozen_fingerprints"
    / "day18_frozen_source_manifest.csv"
)

OUT = (
    ROOT
    / "output/M1_day18/04_step4_structure_key_time_audit"
    / f"run_v2_{datetime.now():%Y%m%d_%H%M%S_%f}"
)

DATE_NAMES = {
    "analysisdate", "date", "tradedate",
    "datetime", "timestamp",
    "labelenddate", "nextanalysisdate",
    "startdate", "enddate",
    "eventdate", "reportdate",
    "asofdate", "effectivedate",
    "periodstart", "periodend",
    "targetdate", "monthend"
}

# V2: recognize Day4 network endpoint columns
EDGE_PAIRS = [
    ("masterindexi", "masterindexj"),
    ("sourceindex", "targetindex"),
    ("sourcemasterindex", "targetmasterindex"),
    ("srcindex", "dstindex"),
    ("sourceid", "targetid"),
    ("src", "dst"),
    ("source", "target"),
    ("fromindex", "toindex"),
    ("from", "to"),
    ("u", "v"),
    ("nodei", "nodej"),
    ("i", "j")
]

ENTITY = [
    "masterindex", "securityid",
    "stockcode", "ticker", "symbol",
    "permno", "assetid",
    "instrumentid", "stockid"
]

ID_COLS = ["recordid", "rowid", "edgeid", "id"]


# ============================================================
# 2. Column and date utilities
# ============================================================

def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def parse_date(s):
    raw = (
        s.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    out = pd.to_datetime(raw, errors="coerce")

    formats = [
        (r"\d{8}", "%Y%m%d", None),
        (r"\d{6}", "%Y%m", None),
        (r"\d{10}", None, "s"),
        (r"\d{13}", None, "ms")
    ]

    for pattern, fmt, unit in formats:
        mask = raw.str.fullmatch(pattern, na=False)

        if mask.any():
            values = (
                pd.to_numeric(raw[mask], errors="coerce")
                if unit else raw[mask]
            )

            out.loc[mask] = (
                pd.to_datetime(
                    values, unit=unit, errors="coerce"
                )
                if unit
                else pd.to_datetime(
                    values, format=fmt, errors="coerce"
                )
            )

    return out


# ============================================================
# 3. Candidate primary-key inference
# ============================================================

def infer(cols):
    d = {norm(c): c for c in cols}

    # V2: avoid treating "candidate" as a date column
    dates = [
        c for c in cols
        if norm(c) in DATE_NAMES
        or str(c).lower().endswith(
            ("_date", "_datetime", "_timestamp")
        )
    ]

    date = next(
        (
            d[x] for x in [
                "analysisdate", "date",
                "tradedate", "datetime", "timestamp"
            ]
            if x in d
        ),
        None
    )

    window = d.get("window") or d.get("windowdays")

    pair = next(
        (
            (d[a], d[b])
            for a, b in EDGE_PAIRS
            if a in d and b in d
        ),
        None
    )

    prefix = (
        ([date] if date else [])
        + ([window] if window else [])
    )

    if pair:
        key = prefix + list(pair)
    else:
        entity = next(
            (d[x] for x in ENTITY if x in d),
            None
        )

        if entity:
            key = prefix + [entity]
        else:
            entity = next(
                (d[x] for x in ID_COLS if x in d),
                None
            )
            key = [entity] if entity else []

    return key, dates, window, pair


# ============================================================
# 4. File-level audit
# ============================================================

def audit_file(p):
    r = dict(
        path=str(p),
        file_type=p.suffix.lower(),
        row_count="",
        column_count=0,
        columns="",
        column_types="",
        schema_sha256="",
        candidate_primary_key="",
        key_status="NOT_INFERRED",
        duplicate_key_rows="",
        null_key_rows="",
        self_loops="",
        duplicate_undirected_pairs="",
        date_columns="",
        time_min="",
        time_max="",
        invalid_date_values=0,
        window_values="",
        status="REVIEW",
        error=""
    )

    if not p.is_file():
        r.update(
            status="ERROR",
            error="FILE_NOT_FOUND"
        )
        return r, [], [("FILE_NOT_FOUND", "")]

    ext = p.suffix.lower()

    try:
        # JSON metadata
        if ext == ".json":
            obj = json.loads(
                p.read_text(encoding="utf-8-sig")
            )

            cols = list(obj) if isinstance(obj, dict) else []
            types = (
                [type(obj[c]).__name__ for c in cols]
                if isinstance(obj, dict) else []
            )

            sig = "|".join(
                f"{c}:{t}" for c, t in zip(cols, types)
            )

            r.update(
                row_count=(
                    len(obj) if isinstance(obj, list) else 1
                ),
                column_count=len(cols),
                columns="|".join(cols),
                column_types="|".join(types),
                schema_sha256=hashlib.sha256(
                    sig.encode()
                ).hexdigest(),
                key_status="NOT_APPLICABLE",
                status="INFO"
            )

            return r, [], []

        # Read schemas
        if ext in {".parquet", ".pq"}:
            pf = pq.ParquetFile(p)
            cols = pf.schema_arrow.names
            types = [
                str(field.type)
                for field in pf.schema_arrow
            ]
            r["row_count"] = pf.metadata.num_rows
            enc = None

        elif ext == ".csv":
            enc = "utf-8-sig"

            try:
                cols = list(
                    pd.read_csv(
                        p, nrows=0, encoding=enc
                    ).columns
                )
            except UnicodeDecodeError:
                enc = "gb18030"
                cols = list(
                    pd.read_csv(
                        p, nrows=0, encoding=enc
                    ).columns
                )

            sample = pd.read_csv(
                p,
                nrows=1000,
                encoding=enc,
                low_memory=False
            )

            types = [
                str(sample[c].dtype) for c in cols
            ]
        else:
            return r, [], []

        # Infer primary key and date columns
        key, dates, window, pair = infer(cols)

        need = list(
            dict.fromkeys(
                key + dates + ([window] if window else [])
            )
        )

        if ext == ".csv" and not need and cols:
            need = [cols[0]]

        # Only load columns needed for auditing
        if ext in {".parquet", ".pq"}:
            df = (
                pd.read_parquet(p, columns=need)
                if need else None
            )
        else:
            df = (
                pd.read_csv(
                    p,
                    usecols=need,
                    encoding=enc,
                    low_memory=False
                )
                if need else None
            )

            r["row_count"] = (
                len(df) if df is not None else 0
            )

        sig = "|".join(
            f"{c}:{t}" for c, t in zip(cols, types)
        )

        r.update(
            column_count=len(cols),
            columns="|".join(map(str, cols)),
            column_types="|".join(types),
            schema_sha256=hashlib.sha256(
                sig.encode()
            ).hexdigest(),
            candidate_primary_key="|".join(key),
            key_status=(
                "INFERRED" if key else "NOT_INFERRED"
            ),
            date_columns="|".join(dates)
        )

        issues = []
        times = []

        # Primary-key checks
        if df is not None and key:
            r["null_key_rows"] = int(
                df[key].isna().any(axis=1).sum()
            )

            r["duplicate_key_rows"] = int(
                df.duplicated(key).sum()
            )

            if r["null_key_rows"]:
                issues.append(
                    ("NULL_PRIMARY_KEY", r["null_key_rows"])
                )

            if r["duplicate_key_rows"]:
                issues.append(
                    (
                        "DUPLICATE_PRIMARY_KEY",
                        r["duplicate_key_rows"]
                    )
                )

        # Window values
        if df is not None and window and window in df:
            vals = sorted(
                df[window].dropna().astype(str).unique()
            )

            r["window_values"] = (
                "|".join(vals[:20])
                + ("|..." if len(vals) > 20 else "")
            )

        # Date checks
        mins, maxs = [], []

        if df is not None:
            for c in dates:
                parsed = parse_date(df[c])
                valid = parsed.dropna()

                bad = int(
                    (df[c].notna() & parsed.isna()).sum()
                )

                lo = (
                    valid.min().strftime("%Y-%m-%d")
                    if len(valid) else ""
                )
                hi = (
                    valid.max().strftime("%Y-%m-%d")
                    if len(valid) else ""
                )

                times.append(
                    dict(
                        path=str(p),
                        date_column=c,
                        valid_dates=len(valid),
                        invalid_dates=bad,
                        min_date=lo,
                        max_date=hi
                    )
                )

                r["invalid_date_values"] += bad

                if lo:
                    mins.append(lo)
                if hi:
                    maxs.append(hi)

                if bad:
                    issues.append(
                        ("INVALID_DATE_VALUES", f"{c}:{bad}")
                    )

            r["time_min"] = min(mins, default="")
            r["time_max"] = max(maxs, default="")

            # Undirected edge checks
            if pair:
                a = df[pair[0]]
                b = df[pair[1]]

                r["self_loops"] = int(
                    (a == b).sum()
                )

                a = (
                    a.astype("string")
                    .fillna("<NA>")
                )
                b = (
                    b.astype("string")
                    .fillna("<NA>")
                )

                lo = a.where(a <= b, b)
                hi = b.where(a <= b, a)

                # V2: preserve date/window context
                context = [
                    c for c in key if c not in pair
                ]

                canon = df[context].copy()
                canon["_lo"] = lo
                canon["_hi"] = hi

                r["duplicate_undirected_pairs"] = int(
                    canon.duplicated().sum()
                )

                if r["self_loops"]:
                    issues.append(
                        ("SELF_LOOPS", r["self_loops"])
                    )

                if r["duplicate_undirected_pairs"]:
                    issues.append(
                        (
                            "DUPLICATE_UNDIRECTED_PAIRS",
                            r["duplicate_undirected_pairs"]
                        )
                    )

        if not key:
            issues.append(
                (
                    "KEY_NOT_INFERRED",
                    "需按表语义人工确认"
                )
            )

        r["status"] = (
            "REVIEW" if issues else "PASS"
        )

        return r, times, issues

    except Exception as e:
        r.update(
            status="ERROR",
            error=f"{type(e).__name__}: {e}"
        )
        return r, [], [("READ_ERROR", r["error"])]


# ============================================================
# 5. Main audit
# ============================================================

def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Frozen manifest not found: {MANIFEST}"
        )

    m = pd.read_csv(
        MANIFEST,
        encoding="utf-8-sig"
    )

    if "path" not in m.columns:
        raise ValueError(
            f"Manifest needs 'path'; columns={list(m.columns)}"
        )

    raw_paths = (
        m["path"].dropna().astype(str).tolist()
    )

    files = list(
        dict.fromkeys(
            Path(x)
            for x in raw_paths
            if Path(x).suffix.lower()
            in {".parquet", ".pq", ".csv", ".json"}
        )
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    audits = []
    times = []
    issues = []

    for i, p in enumerate(files, 1):
        r, t, found = audit_file(p)

        audits.append(r)
        times.extend(t)

        issues.extend(
            {
                "path": str(p),
                "issue": k,
                "detail": str(v)
            }
            for k, v in found
        )

        if i % 250 == 0:
            print(
                f"Audited {i:,}/{len(files):,}: {p.name}",
                flush=True
            )

    # File-level results
    adf = pd.DataFrame(audits)

    adf.to_csv(
        OUT / "data_structure_key_audit.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Directory-level schema consistency
    families = []

    grouping = adf.groupby(
        [
            adf["path"].map(
                lambda x: str(Path(x).parent)
            ),
            adf["file_type"]
        ]
    )

    for (parent, ext), g in grouping:
        sigs = (
            g.loc[
                g["schema_sha256"].astype(str).ne(""),
                "schema_sha256"
            ].nunique()
        )

        status = (
            "INFO" if len(g) == 1
            else "PASS" if sigs <= 1
            else "REVIEW"
        )

        families.append(
            {
                "parent": parent,
                "file_type": ext,
                "file_count": len(g),
                "schema_versions": int(sigs),
                "status": status
            }
        )

        if status == "REVIEW":
            issues.append(
                {
                    "path": parent,
                    "issue": "SCHEMA_VARIATION_IN_DIRECTORY",
                    "detail": (
                        f"{ext}: {sigs} schema signatures "
                        f"across {len(g)} files"
                    )
                }
            )

    pd.DataFrame(
        families,
        columns=[
            "parent", "file_type",
            "file_count", "schema_versions",
            "status"
        ]
    ).to_csv(
        OUT / "schema_consistency_audit.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        times,
        columns=[
            "path", "date_column",
            "valid_dates", "invalid_dates",
            "min_date", "max_date"
        ]
    ).to_csv(
        OUT / "time_range_audit.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        issues,
        columns=["path", "issue", "detail"]
    ).to_csv(
        OUT / "step4_issues.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Summary
    summary = {
        "manifest_rows": len(m),

        "duplicate_manifest_paths": (
            len(raw_paths)
            - m["path"].dropna().astype(str).nunique()
        ),

        "data_files_audited": len(files),

        "schema_families": len(families),

        "schema_families_with_variation": sum(
            x["status"] == "REVIEW"
            for x in families
        ),

        "status_counts": dict(
            Counter(x["status"] for x in audits)
        ),

        "files_with_inferred_key": sum(
            x["key_status"] == "INFERRED"
            for x in audits
        ),

        "files_with_duplicate_keys": sum(
            isinstance(x["duplicate_key_rows"], int)
            and x["duplicate_key_rows"] > 0
            for x in audits
        ),

        "files_with_invalid_dates": sum(
            x["invalid_date_values"] > 0
            for x in audits
        ),

        "time_range_rows": len(times),
        "issue_rows": len(issues),

        "note": (
            "主键为按列名推断的候选主键；"
            "需结合各表数据契约解释 REVIEW。"
        ),

        "output_dir": str(OUT)
    }

    (
        OUT / "day18_step4_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
