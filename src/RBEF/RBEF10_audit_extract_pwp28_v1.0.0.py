#!/usr/bin/env python3
"""RBEF10: audit JPO04 and export the canonical 28 C daily area (read-only source)."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

SOURCE_SHA256 = '05D0D99884E248DEFF75D50971468F90932AE337DA7C3C2009BA2323299D80EB'
START = date(1981, 9, 1)
END = date(2026, 9, 30)
THRESHOLDS = ('28.0000000000', '28.5000000000', '29.0000000000')
EXPECTED_DAYS = (END - START).days + 1
EXPECTED_ROWS = EXPECTED_DAYS * 3
REQUIRED = {'date', 'threshold_c', 'canonical_area_million_km2', 'canonical_area_km2'}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Original JPO04 CSV, never modified')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--allow-hash-mismatch', action='store_true', help='For exploratory audits only; exported data flagged NON-CANONICAL')
    args = parser.parse_args()
    source = args.input.expanduser().resolve(strict=True)
    actual_sha = digest(source)
    hash_match = actual_sha == SOURCE_SHA256
    if not hash_match and not args.allow_hash_mismatch:
        raise SystemExit(f'FAIL: SHA256 mismatch. Expected {SOURCE_SHA256}; observed {actual_sha}. No output created.')

    all_keys = set()
    counts = Counter()
    dates = set()
    area28 = {}
    row_count = 0
    with source.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise SystemExit(f'FAIL: missing columns: {sorted(REQUIRED-set(reader.fieldnames or []))}')
        for row in reader:
            row_count += 1
            try:
                day = date.fromisoformat(row['date'].strip())
                temp = float(row['threshold_c'])
                area = float(row['canonical_area_million_km2'])
                area_km2 = float(row['canonical_area_km2'])
            except (ValueError, TypeError) as e:
                raise SystemExit(f'FAIL: invalid value at CSV data row {row_count}: {e}')
            if not all(map(math.isfinite, (temp, area, area_km2))):
                raise SystemExit(f'FAIL: non-finite numeric value at data row {row_count}')
            if temp not in (28.0, 28.5, 29.0):
                raise SystemExit(f'FAIL: unexpected threshold {temp} at data row {row_count}')
            if day < START or day > END:
                raise SystemExit(f'FAIL: unexpected date {day} at data row {row_count}')
            if area < 0 or area_km2 < 0:
                raise SystemExit(f'FAIL: negative area at data row {row_count}')
            if not math.isclose(area, area_km2 / 1e6, abs_tol=1e-7, rel_tol=1e-10):
                raise SystemExit(f'FAIL: area units disagree at data row {row_count}')
            key = (day, temp)
            if key in all_keys:
                raise SystemExit(f'FAIL: duplicate date/threshold: {key}')
            all_keys.add(key)
            dates.add(day)
            counts[temp] += 1
            if temp == 28.0:
                area28[day] = row['canonical_area_million_km2'].strip()

    expected_dates = {START + timedelta(days=i) for i in range(EXPECTED_DAYS)}
    errors = []
    if row_count != EXPECTED_ROWS: errors.append(f'row_count {row_count} != {EXPECTED_ROWS}')
    if dates != expected_dates: errors.append(f'date coverage: missing {len(expected_dates-dates)}, extra {len(dates-expected_dates)}')
    for t in (28.0, 28.5, 29.0):
        if counts[t] != EXPECTED_DAYS: errors.append(f'threshold {t}: {counts[t]} != {EXPECTED_DAYS}')
    if errors:
        raise SystemExit('FAIL: '+ '; '.join(errors) + '. No output created.')

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'RBEF_PWP28_daily_area_19810901_20260930.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['date', 'pwp28_area_million_km2'])
        for day in sorted(area28): writer.writerow([day.isoformat(), area28[day]])
    values = [float(v) for v in area28.values()]
    report = {
        'status': 'PASS' if hash_match else 'PASS_WITH_HASH_MISMATCH_NON_CANONICAL',
        'script': 'RBEF10_audit_extract_pwp28_v1.0.0.py',
        'source_file': source.name,
        'source_sha256_expected': SOURCE_SHA256,
        'source_sha256_observed': actual_sha,
        'source_sha256_matches': hash_match,
        'source_bytes': source.stat().st_size,
        'total_data_rows': row_count,
        'threshold_counts': {str(t): counts[t] for t in (28.0, 28.5, 29.0)},
        'first_date': START.isoformat(), 'last_date': END.isoformat(),
        'unique_dates': len(dates), 'missing_dates': 0, 'duplicate_date_threshold_pairs': 0,
        'pwp28_mean_million_km2': sum(values)/len(values),
        'pwp28_min_million_km2': min(values), 'pwp28_max_million_km2': max(values),
        'derived_file': csv_path.name, 'derived_sha256': digest(csv_path),
        'derived_bytes': csv_path.stat().st_size,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'notes': ['Area series only: does not reproduce spatial Figure 1.',
                  'The original JPO04 file is read only and never overwritten.',
                  'No Zenodo DOI is asserted; verify archive status before publication.',
                  'Mean/min/max are audit diagnostics, not an independent reanalysis of the manuscript.']
    }
    (output_dir/'RBEF10_audit_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    (output_dir/'SHA256SUMS.txt').write_text(f"{report['derived_sha256']}  {csv_path.name}\n", encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('\nAUDIT PASSED. Original source untouched.')

if __name__ == '__main__': main()
