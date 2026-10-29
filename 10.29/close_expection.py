"""Close the single documented post-inventory size exception in Day18 Step3.

Read-only for all frozen inputs; creates a source SHA-256 baseline and a
separate exception report. Save under output/M1_day18/03_step3_frozen_fingerprints.
"""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import os

ROOT = Path(r"D:\M1_StockNetwork")
FREEZE = ROOT / "output/M1_day18/03_step3_frozen_fingerprints"
RUN = FREEZE / "run_20261029_144940_290846"
INV = ROOT / "output/M1_day18/02_step2_source_inventory/run_20261029_120201_266156/day18_source_inventory.csv"
BASE = FREEZE / "day18_frozen_source_manifest.csv"
EXC = FREEZE / "day18_step3_size_exception.json"
TARGET = ROOT / "source_inventory.py"
EXPECTED_SHA = "1e873bbc5cabdb699fbc54800c07605148206881ef9103bf11144b809360f267"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def key(path):
    return os.path.normcase(os.path.abspath(str(path)))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


if BASE.exists():
    raise FileExistsError(f"Source baseline already exists; will not overwrite: {BASE}")

inventory = read_csv(INV)
audit = read_csv(RUN / "source_fingerprint_audit.csv")
code = read_csv(FREEZE / "day18_source_code_manifest.csv")
inv_by_path = {key(r["path"]): r for r in inventory}
code_by_path = {key(r["path"]): r for r in code}
assert len(inventory) == len(audit) == len(inv_by_path) == 5436
assert len(code) == 114
assert key(TARGET) in inv_by_path and key(TARGET) in code_by_path
assert code_by_path[key(TARGET)]["sha256"] == EXPECTED_SHA

old = inv_by_path[key(TARGET)]
assert int(old["size_bytes"]) == 7066
assert datetime.fromisoformat(old["modified_time"]).timestamp() < TARGET.stat().st_mtime

manifest = []
for i, row in enumerate(audit, 1):
    p = Path(row["path"])
    k = key(p)
    assert k in inv_by_path and p.is_file(), f"Missing inventory/source: {p}"
    size = p.stat().st_size
    assert size == int(row["size_bytes"]), f"Changed size since audit: {p}"
    assert sha256(p) == row["sha256"], f"Changed SHA-256 since audit: {p}"
    if k == key(TARGET):
        assert size == 7060 and row["sha256"] == EXPECTED_SHA
        assert row["status"] == "INVENTORY_SIZE_MISMATCH"
    else:
        assert size == int(inv_by_path[k]["size_bytes"]), f"Size mismatch: {p}"
        assert row["status"] == "BASELINE_CREATED", f"Unexpected status: {p}"
    manifest.append({"path": str(p), "size_bytes": size, "sha256": row["sha256"]})
    if i % 500 == 0:
        print(f"Verified {i:,}/{len(audit):,}")

# Verify previously frozen source-code files have not changed.
for row in code:
    p = Path(row["path"])
    assert p.is_file() and sha256(p) == row["sha256"], f"Frozen code changed: {p}"

exception = {
    "status": "DOCUMENTED_POST_INVENTORY_CHANGE",
    "file": str(TARGET),
    "inventory_size_bytes": 7066,
    "current_size_bytes": 7060,
    "inventory_modified_time": old["modified_time"],
    "current_sha256": EXPECTED_SHA,
    "inventory_sha256": None,
    "note": "Changed after Step2 inventory; historical content difference cannot be verified."
}
EXC.write_text(json.dumps(exception, indent=2), encoding="utf-8")
with BASE.open("x", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
    w.writeheader()
    w.writerows(manifest)
print(f"SOURCE BASELINE CREATED WITH ONE DOCUMENTED EXCEPTION: {len(manifest)} files")
print(f"Frozen code hashes verified: {len(code)}")
print("Manifest:", BASE)
print("Exception:", EXC)
