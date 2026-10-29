
from pathlib import Path
from datetime import datetime
from collections import Counter
import csv, hashlib, json, os, platform, re, sys

ROOT = Path(__file__).resolve().parent
INV = ROOT / "output/M1_day18/02_step2_source_inventory/run_20261029_120201_266156/day18_source_inventory.csv"
FREEZE = ROOT / "output/M1_day18/03_step3_frozen_fingerprints"
RUN = FREEZE / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
SKIP = {"output", ".git", "__pycache__", ".venv", "venv", "env", "site-packages"}


def norm(p):
    return os.path.normcase(os.path.abspath(str(p)))


def resolve(v):
    v = str(v or "").strip().strip('"')
    if not v:
        return None
    p = Path(v)
    return p if p.is_absolute() else ROOT / p


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def audit(items, baseline, report):
    exists = baseline.exists()
    if exists:
        with baseline.open(encoding="utf-8-sig", newline="") as f:
            old = {norm(resolve(r["path"])): r for r in csv.DictReader(f)}
    else:
        old = {}

    current, result, bad = set(items), [], False

    for k, (p, expected_size) in items.items():
        r = {"path": str(p), "size_bytes": "", "sha256": "", "status": ""}
        try:
            if not p.is_file():
                r["status"] = "MISSING"
            else:
                r["size_bytes"], r["sha256"] = p.stat().st_size, digest(p)
                r["status"] = (
                    "INVENTORY_SIZE_MISMATCH"
                    if expected_size is not None and r["size_bytes"] != expected_size
                    else "BASELINE_CREATED" if not exists
                    else "NEW" if k not in old
                    else "MATCH" if r["sha256"] == old[k].get("sha256")
                    else "CHANGED"
                )
        except Exception as e:
            r.update(status="HASH_ERROR", error=str(e))

        bad |= r["status"] in {
            "MISSING", "HASH_ERROR", "NEW", "CHANGED",
            "INVENTORY_SIZE_MISMATCH"
        }
        result.append(r)

        if len(result) % 500 == 0:
            print(f"Hashed {len(result):,}/{len(items):,}: {p.name}")

    if exists:
        for k, r in old.items():
            if k not in current:
                result.append({
                    "path": r["path"], "size_bytes": r.get("size_bytes", ""),
                    "sha256": r.get("sha256", ""),
                    "status": "NOT_IN_CURRENT_INVENTORY"
                })
                bad = True

    fields = ["path", "size_bytes", "sha256", "status", "error"]
    if not exists and result and not bad:
        write_csv(baseline, result, ["path", "size_bytes", "sha256"])
    write_csv(report, result, fields)

    return {
        "files": len(items),
        "status_counts": dict(Counter(r["status"] for r in result)),
        "baseline_created": not exists and bool(result) and not bad,
        "baseline_exists": baseline.exists(),
        "issues_found": bad or not baseline.exists()
    }


if not INV.is_file():
    raise FileNotFoundError(f"Step2 inventory not found: {INV}")

with INV.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    rows, cols = list(reader), reader.fieldnames or []

if not rows:
    raise ValueError(f"Empty inventory: {INV}")

score = {
    c: sum(1 for r in rows[:100]
           if (p := resolve(r.get(c))) and p.is_file())
    for c in cols
}
path_col = max(score, key=score.get)
if not score[path_col]:
    raise ValueError(f"Cannot find path column; columns={cols}; scores={score}")

size_col = next(
    (c for c in cols if re.sub(r"[^a-z0-9]", "", c.lower())
     in {"sizebytes", "filesizebytes", "filesize", "size"}), None
)

source = {}
for r in rows:
    p = resolve(r.get(path_col))
    if p:
        try:
            size = int(float(str(r.get(size_col, "")).replace(",", ""))) \
                if size_col and str(r.get(size_col, "")).strip() else None
        except ValueError:
            size = None
        source.setdefault(norm(p), (p, size))

code = {
    norm(p): (p, None) for p in ROOT.rglob("*.py")
    if not any(x.lower() in SKIP for x in p.relative_to(ROOT).parts)
}

FREEZE.mkdir(parents=True, exist_ok=True)
RUN.mkdir(parents=True, exist_ok=True)

s = audit(
    source, FREEZE / "day18_frozen_source_manifest.csv",
    RUN / "source_fingerprint_audit.csv"
)
c = audit(
    code, FREEZE / "day18_source_code_manifest.csv",
    RUN / "source_code_version_audit.csv"
)

summary = {
    "inventory": str(INV),
    "inventory_rows": len(rows),
    "path_column": path_col,
    "unique_source_files": len(source),
    "source_fingerprints": s,
    "source_code_files": len(code),
    "source_code_version": c,
    "python_version": platform.python_version(),
    "python_executable": sys.executable,
    "audit_script_sha256": digest(Path(__file__).resolve()),
    "run_dir": str(RUN)
}
(RUN / "day18_step3_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
print("Reports:", RUN)
