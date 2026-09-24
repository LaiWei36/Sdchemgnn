#!/usr/bin/env python
"""Build the Materials Project magnetic-ordering side-table for the SD-ChemGNN runs.

For every material in mp_dataset.jsonl, fetch the MP magnetism doc (ordering,
is_magnetic, num_magnetic_sites, total_magnetization_normalized_formula_units)
and write ordering_sidetable.csv, one row per jsonl record, with an aux_mask
column that says whether the ordering label may be used as auxiliary
supervision for a model trained on the CURATED (manifest) labels.

Provenance rules:
  * MP `ordering` describes the RAW calculation. The manifest moved some
    training targets away from raw, so the ordering label is only valid aux
    supervision where the curated target still describes that calculation.
  * aux_mask reasons, in priority order:
      manifest_drop   id dropped by the manifest (never trains; mask for tidiness)
      jsonl_skipped   record missing required fields -> never entered the dataset
      missing_mp      id absent from current MP (deprecated/merged)
      fetch_failed    API chunk failed after retries (re-run to retry; cache keeps
                      successful chunks)
      mp_drift        re-fetched MP total != jsonl raw total (MP revised the entry
                      since the dataset snapshot -> ordering describes a DIFFERENT
                      calculation than our label)
      manifest_shift  |clamp(raw, +/-cap) - curated| > --mask-shift-tol (0.5 uB):
                      basin-flip / value-move relabels. Cap-only saturation
                      (raw>cap, no manifest row) is NOT masked - same basin.
  * The drift gate compares against the JSONL RAW value (the manifest never
    touches the jsonl), NOT the curated label.

The output row order is the jsonl order and the file's md5 is printed, so WHICH
side-table a later run used can be fingerprinted exactly like label_manifest.

Typical use on the cluster (login node, needs internet + MP_API_KEY):
    python fetch_mp_magnetic_ordering.py                       # defaults below
    python fetch_mp_magnetic_ordering.py --limit 500           # smoke test first!
The fetch is RESUMABLE: responses append to --cache (one json per line, plus
tombstones for ids MP does not know); a re-run only fetches what is missing.
Exit code is non-zero if any chunk failed, so a plain re-run completes the job.

Offline testing hook: --mock-docs FILE bypasses the network entirely and reads
"API responses" from a jsonl of {"material_id":..., "ordering":..., ...} dicts.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time

REQUIRED_KEYS = (  # a record missing any of these was skipped by build_crystal_graph_dataset.py
    'poscar', 'formation_energy_per_atom',
    'total_magnetization_normalized_formula_units', 'total_magnetization',
    'material_id', 'formula_pretty',
)
FIELDS = ['material_id', 'ordering', 'is_magnetic', 'num_magnetic_sites',
          'total_magnetization_normalized_formula_units']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--jsonl',    default='./dataset_mp/mp_dataset.jsonl')
    p.add_argument('--manifest', default='./label_manifest.csv')
    p.add_argument('--out',      default='./ordering_sidetable.csv')
    p.add_argument('--cache',    default='./mp_ordering_cache.jsonl',
                    help='resumable per-id response cache (append-only jsonl)')
    p.add_argument('--api-key',  default=os.environ.get('MP_API_KEY', ''),
                    help='MP API key (default: env MP_API_KEY; pymatgen config as last resort)')
    p.add_argument('--endpoint', choices=['magnetism', 'summary'], default='magnetism',
                    help='MP collection to query (summary mirrors the magnetism fields)')
    p.add_argument('--chunk', type=int, default=250,
                    help='ids per API request (kept small: ids travel in the query string)')
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--limit', type=int, default=0,
                    help='fetch only the first N dataset ids (smoke test); side-table still '
                         'covers every jsonl row, the un-fetched ones as fetch_failed')
    p.add_argument('--mask-shift-tol', type=float, default=0.5,
                    help='|clamp(raw)-curated| above this masks the aux label (uB)')
    p.add_argument('--drift-tol', type=float, default=1e-3,
                    help='|refetched MP total - jsonl raw| above this = mp_drift (uB)')
    p.add_argument('--cap', type=float, default=30.0,
                    help='target_cap used by training (clamp before the shift test)')
    p.add_argument('--mock-docs', default='',
                    help='OFFLINE TEST ONLY: jsonl of fake MP docs; no network is touched')
    return p.parse_args()


# ── step 1: stream the dataset jsonl (ids + raw totals only; poscars discarded) ──
def read_jsonl(path):
    rows, dup = [], 0
    seen = set()
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f'  [jsonl] line {ln}: unparseable, skipped', file=sys.stderr)
                continue
            mid = str(rec.get('material_id', '')) or f'<line {ln}>'
            ok = all(rec.get(k) is not None for k in REQUIRED_KEYS)
            raw = rec.get('total_magnetization_normalized_formula_units')
            if mid in seen:
                dup += 1
                continue
            seen.add(mid)
            rows.append({'material_id': mid,
                         'formula': rec.get('formula_pretty', ''),
                         'raw_fu': float(raw) if raw is not None else None,
                         'jsonl_ok': ok})
    if dup:
        print(f'  [jsonl] WARNING: {dup} duplicate material_id lines ignored (kept first).')
    return rows


# ── step 2: manifest (same parse semantics as the trainer's load_manifest) ──
def read_manifest(path):
    man = {}
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            try:
                new = float(row['new'])
            except (ValueError, TypeError, KeyError):
                new = None                      # tolerate blank 'new' on drop rows
            man[row['material_id']] = (row['action'], new)
    return man


# ── step 3: fetch (resumable) ──────────────────────────────────────────────────
def norm_doc(d):
    """Normalize one MP doc (object or dict) to plain python values."""
    def g(k):
        return d.get(k) if isinstance(d, dict) else getattr(d, k, None)
    o = g('ordering')
    o = getattr(o, 'value', o)                  # pymatgen Ordering enum -> str
    n = g('num_magnetic_sites')
    t = g('total_magnetization_normalized_formula_units')
    return {'material_id': str(g('material_id')),
            'ordering': str(o) if o is not None else '',
            'is_magnetic': g('is_magnetic'),
            'num_magnetic_sites': int(n) if n is not None else None,
            'mp_total_fu': float(t) if t is not None else None}


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                cache[str(d['material_id'])] = d
    return cache


def fetch(ids, args, cache):
    """Fill `cache` for every id in `ids`. Returns the set of ids whose chunk
    failed after retries (they stay OUT of the cache so a re-run retries them)."""
    todo = [m for m in ids if m not in cache]
    if not todo:
        print('  [fetch] cache already complete - nothing to fetch.')
        return set()

    if args.mock_docs:                          # offline test path: no network
        print(f'  [fetch] MOCK MODE: reading docs from {args.mock_docs}')
        mock = {}
        with open(args.mock_docs) as fh:
            for line in fh:
                if line.strip():
                    d = norm_doc(json.loads(line))
                    mock[d['material_id']] = d
        with open(args.cache, 'a') as out:
            for m in todo:
                d = mock.get(m, {'material_id': m, 'missing': True})
                cache[m] = d
                out.write(json.dumps(d) + '\n')
        return set()

    try:
        from mp_api.client import MPRester
    except Exception:                           # noqa: BLE001 - show the REAL cause
        import traceback
        traceback.print_exc()
        sys.exit(
            '\nFailed to import mp_api.client — the full traceback is above.\n'
            f'  interpreter: {sys.executable}\n'
            'If `conda list` shows mp-api in this env, the package is present but its\n'
            'IMPORT is failing (usually a pydantic / emmet-core version conflict).\n'
            'Two ways out:\n'
            '  (a) RECOMMENDED — dedicated minimal env; this script needs ONLY mp-api\n'
            '      (everything else is stdlib), so the training env stays untouched:\n'
            '          conda create -n mpfetch python=3.11 -y\n'
            '          conda activate mpfetch\n'
            '          pip install mp-api\n'
            '          python fetch_mp_magnetic_ordering.py --limit 500\n'
            '  (b) repair in place — may disturb pinned deps in a torch/PyG env:\n'
            '          pip install -U mp-api emmet-core pydantic pydantic-settings\n')

    failed = set()
    key = args.api_key or None                  # None -> pymatgen config fallback
    n_chunks = (len(todo) + args.chunk - 1) // args.chunk
    print(f'  [fetch] {len(todo)} ids to fetch in {n_chunks} chunks of <= {args.chunk} '
          f'({len(cache)} already cached)')
    with MPRester(key) as mpr:
        coll = getattr(mpr.materials, args.endpoint)
        for c in range(n_chunks):
            part = todo[c * args.chunk:(c + 1) * args.chunk]
            docs = None
            for attempt in range(1, args.retries + 1):
                try:
                    docs = coll.search(material_ids=part, fields=FIELDS)
                    break
                except Exception as exc:        # noqa: BLE001 - report and retry
                    print(f'  [fetch] chunk {c + 1}/{n_chunks} attempt {attempt} failed: {exc}',
                          file=sys.stderr)
                    time.sleep(5 * attempt)
            if docs is None:
                failed.update(part)             # NOT cached -> a re-run retries them
                continue
            got = {}
            for d in docs:
                nd = norm_doc(d)
                got[nd['material_id']] = nd
            with open(args.cache, 'a') as out:
                for m in part:
                    d = got.get(m, {'material_id': m, 'missing': True})  # tombstone
                    cache[m] = d
                    out.write(json.dumps(d) + '\n')
            if (c + 1) % 20 == 0 or c + 1 == n_chunks:
                print(f'  [fetch] {c + 1}/{n_chunks} chunks done')
    return failed


# ── step 4+5: build the side-table and report ──────────────────────────────────
def main():
    args = parse_args()
    print(f'jsonl:    {args.jsonl}\nmanifest: {args.manifest}\nout:      {args.out}')

    rows = read_jsonl(args.jsonl)
    man = read_manifest(args.manifest)
    n_drop_man = sum(1 for a, _ in man.values() if a == 'drop')
    print(f'  jsonl records: {len(rows)} ({sum(r["jsonl_ok"] for r in rows)} dataset-eligible)')
    print(f'  manifest rows: {len(man)} ({n_drop_man} drops)  '
          f'[expect 15959 / 573 for md5 3f0150930c]')
    missing_from_jsonl = [m for m in man if m not in {r['material_id'] for r in rows}]
    if missing_from_jsonl:
        print(f'  WARNING: {len(missing_from_jsonl)} manifest ids not in jsonl, '
              f'e.g. {missing_from_jsonl[:3]}')

    fetch_ids = [r['material_id'] for r in rows if r['jsonl_ok']]
    if args.limit:
        fetch_ids = fetch_ids[:args.limit]
        print(f'  [fetch] --limit {args.limit}: smoke run, side-table will mark the rest fetch_failed')
    cache = load_cache(args.cache)
    failed = fetch(fetch_ids, args, cache)

    reasons = {}
    out_rows = []
    for r in rows:
        mid = r['material_id']
        raw = r['raw_fu']
        act, new = man.get(mid, (None, None))
        doc = cache.get(mid, {})
        shift = drift = None
        # any non-drop manifest action is a relabel (mirrors relabel_master: r[0] != 'drop')
        if act is not None and act != 'drop' and new is not None and raw is not None:
            clamped = max(-args.cap, min(args.cap, raw))
            shift = abs(clamped - new)
        if not doc.get('missing', False) and doc.get('mp_total_fu') is not None and raw is not None:
            drift = abs(doc['mp_total_fu'] - raw)

        if act == 'drop':
            mask, why = 1, 'manifest_drop'
        elif not r['jsonl_ok']:
            mask, why = 1, 'jsonl_skipped'
        elif mid in failed or mid not in cache:
            mask, why = 1, 'fetch_failed'
        elif doc.get('missing'):
            mask, why = 1, 'missing_mp'
        elif drift is not None and drift > args.drift_tol:
            mask, why = 1, 'mp_drift'
        elif shift is not None and shift > args.mask_shift_tol:
            mask, why = 1, 'manifest_shift'
        else:
            mask, why = 0, 'ok'
        reasons[why] = reasons.get(why, 0) + 1

        out_rows.append({
            'material_id': mid, 'formula': r['formula'],
            'ordering': doc.get('ordering', ''),
            'is_magnetic': '' if doc.get('is_magnetic') is None else int(bool(doc.get('is_magnetic'))),
            'num_magnetic_sites': '' if doc.get('num_magnetic_sites') is None else doc.get('num_magnetic_sites'),
            'mp_total_fu': '' if doc.get('mp_total_fu') is None else f'{doc.get("mp_total_fu"):.6f}',
            'jsonl_raw_fu': '' if raw is None else f'{raw:.6f}',
            'drift': '' if drift is None else f'{drift:.6f}',
            'manifest_action': act or '',
            'manifest_new': '' if new is None else f'{new:.6f}',
            'manifest_shift': '' if shift is None else f'{shift:.6f}',
            'aux_mask': mask, 'mask_reason': why,
        })

    cols = list(out_rows[0].keys())
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    digest = hashlib.md5(open(args.out, 'rb').read()).hexdigest()[:10]

    print('\n── side-table report ──')
    print(f'  rows written: {len(out_rows)}  ->  {args.out}  (md5 {digest})')
    for why in ('ok', 'manifest_shift', 'mp_drift', 'missing_mp', 'fetch_failed',
                'jsonl_skipped', 'manifest_drop'):
        if why in reasons:
            print(f'  {why:>15}: {reasons[why]}')
    dist = {}
    for o in (r['ordering'] for r in out_rows if r['aux_mask'] == 0 and r['ordering']):
        dist[o] = dist.get(o, 0) + 1
    tot = sum(dist.values()) or 1
    print('  ordering distribution among aux-usable rows:')
    for o, n in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f'    {o:>8}: {n:7d}  ({100.0 * n / tot:.1f}%)')
    if failed:
        print(f'\n  {len(failed)} ids in FAILED chunks - re-run this script to retry them.')
        sys.exit(2)
    print('\nDONE. Next: python phase1_ordering_diagnostics.py --sidetable', args.out)


if __name__ == '__main__':
    main()
