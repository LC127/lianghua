"""Repair only accidental double-CR (CRCRLF -> CRLF) drift in Day15 Step3 CSVs.
Dry-run by default. With --apply: backup originals before atomic byte repair.
Does not recompute or alter the scientific results or metadata.
"""
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=r'D:\M1_StockNetwork')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    out = root / 'output/M1_day15/03_step3_network_edge_codistress'
    meta_path = out / 'day15_step3_metadata.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('status') != 'FROZEN' or not meta['formal_qa']['all_formal_qa_pass']:
        raise RuntimeError('Step3 metadata is not FROZEN/PASS')
    expected = meta['output_sha256']
    if set(expected) != {
        'day15_step3_edge_manifest.csv', 'day15_step3_qa.csv',
        'network_edge_codistress_daily.csv', 'network_edge_codistress_hac_summary.csv'
    }:
        raise RuntimeError('Unexpected output file set: refusing repair')

    changes = []
    for name, digest in expected.items():
        path = out / name
        raw = path.read_bytes()
        normalized = raw if sha(raw) == digest else raw.replace(b'\r\r\n', b'\r\n')
        raw_hash, normalized_hash = sha(raw), sha(normalized)
        if normalized_hash != digest:
            raise RuntimeError(f'CONTENT DRIFT: {name}: not a newline-only difference')
        if raw_hash != digest:
            changes.append((name, raw_hash, digest, normalized))
        print(f'{name}: {"NEEDS_CRLF_REPAIR" if raw_hash != digest else "PASS"}')

    if not changes:
        print('All Step3 output hashes PASS. No action taken.')
        return
    if not args.apply:
        print(f'DRY RUN: {len(changes)} file(s) need double-CR removal; use --apply to back up and repair.')
        return

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = root / 'output/M1_day15/03a_step3_newline_repair_audit' / stamp
    backup.mkdir(parents=True, exist_ok=False)
    for name in [*expected, meta_path.name]:
        shutil.copy2(out / name, backup / name)

    journal = {'reason': 'CRCRLF-to-CRLF byte repair only', 'backup': str(backup),
               'metadata_script_sha256_unchanged': meta['script_sha256'],
               'original_to_expected': {n: {'original': a, 'expected': b} for n, a, b, _ in changes}}
    (backup / 'repair_journal.json').write_text(json.dumps(journal, indent=2), encoding='utf-8')
    for name, old_hash, digest, new_bytes in changes:
        tmp = out / (name + '.lfrepair.tmp')
        tmp.write_bytes(new_bytes)
        if sha(tmp.read_bytes()) != digest:
            raise RuntimeError('Pre-commit hash failed: ' + name)
        os.replace(tmp, out / name)
    for name, digest in expected.items():
        if sha((out / name).read_bytes()) != digest:
            raise RuntimeError('Post-repair hash failed: ' + name)
    print(f'PASS: {len(changes)} output CSV files repaired without data changes. Backup: {backup}')


if __name__ == '__main__':
    main()
