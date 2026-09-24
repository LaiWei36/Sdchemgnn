#!/usr/bin/env python
"""Fetch per-site magnetic moments (magmoms) from the Materials Project for every
material in mp_dataset.jsonl, with site-order alignment checks against the
jsonl's own POSCARs. Output feeds the per-site moment-decomposition training
(node-level supervision) — the demonstrated fix for the shared-trunk
co-adaptation found in the earlier single-scalar model.

Per material this records:
  n_sites, species sequence (from the MP structure), per-site magmom list
  (signed, µB), alignment status vs the jsonl POSCAR's species sequence, and
  the residual |Σ magmom − total_magnetization(cell)| — site projections never
  sum exactly to the cell total (interstitial contribution), so the residual is
  RECORDED, not failed on; its distribution is the integration-time input for
  the learned-residual design.

Statuses (mutually exclusive, priority order):
  jsonl_skipped      record missing required fields -> never in the dataset
  fetch_failed       API chunk failed after retries (re-run to retry; cache
                     keeps successful chunks; exit code 2 signals a re-run)
  missing_mp         id absent from current MP
  no_structure       doc had no parsable structure
  no_magmom          structure carries no per-site magmom property (typical for
                     non-magnetic entries -> integration treats sites as 0)
  poscar_no_symbols  jsonl POSCAR is VASP4-style (no symbol line): MP data kept,
                     alignment UNVERIFIED for this id
  species_mismatch   POSCAR site sequence != MP structure site sequence:
                     magmoms kept but DO NOT align to this graph's node order —
                     exclude from node supervision unless remapped
  aligned            species sequences match site-for-site: safe to attach

Resumable: responses append to --cache (one json line per id, tombstones for
unknown ids); a re-run fetches only what is missing. Structures are heavier
than the ordering fetch (default --chunk 100; full run ~1540 requests).
Offline test hook: --mock-docs FILE supplies pre-extracted records, no network.

Typical use (needs MP_API_KEY):
    python fetch_mp_site_moments.py --limit 300      # smoke
    python fetch_mp_site_moments.py                  # full, resumable
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time

REQUIRED_KEYS = ('poscar', 'formation_energy_per_atom',
                 'total_magnetization_normalized_formula_units',
                 'total_magnetization', 'material_id', 'formula_pretty')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--jsonl',   default='./dataset_mp/mp_dataset.jsonl')
    p.add_argument('--out',     default='./magmom_sidetable.jsonl')
    p.add_argument('--cache',   default='./mp_magmom_cache.jsonl')
    p.add_argument('--api-key', default=os.environ.get('MP_API_KEY', ''))
    p.add_argument('--endpoint', choices=['materials', 'summary'], default='materials',
                    help='collection whose blessed structure to fetch')
    p.add_argument('--chunk', type=int, default=100)
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--limit', type=int, default=0, help='smoke: fetch only first N ids')
    p.add_argument('--mock-docs', default='', help='OFFLINE TEST ONLY: jsonl of records')
    return p.parse_args()


# ── jsonl streaming: ids + cell totals + POSCAR species sequence ────────────────
def poscar_species(poscar_text):
    """Site-ordered species list from a POSCAR string, or None if VASP4-style
    (no symbols line). ase-written POSCARs (this pipeline) are VASP5."""
    try:
        lines = [ln for ln in str(poscar_text).splitlines()]
        syms = lines[5].split()
        if not syms or any(s.isdigit() for s in syms):
            return None
        counts = [int(x) for x in lines[6].split()]
        if len(counts) != len(syms):
            return None
        out = []
        for s, n in zip(syms, counts):
            out.extend([s] * n)
        return out
    except (IndexError, ValueError):
        return None


def read_jsonl(path):
    rows, seen = [], set()
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = str(rec.get('material_id', '')) or f'<line {ln}>'
            if mid in seen:
                continue
            seen.add(mid)
            ok = all(rec.get(k) is not None for k in REQUIRED_KEYS)
            rows.append({'material_id': mid,
                         'formula': rec.get('formula_pretty', ''),
                         'cell_total': (float(rec['total_magnetization'])
                                        if rec.get('total_magnetization') is not None else None),
                         'poscar_species': poscar_species(rec.get('poscar')) if ok else None,
                         'jsonl_ok': ok})
    return rows


# ── fetch ────────────────────────────────────────────────────────────────────────
def extract(doc):
    """(species list, magmom list or None) from an MP doc's structure —
    handles both hydrated pymatgen Structure and plain-dict forms."""
    st = doc.get('structure') if isinstance(doc, dict) else getattr(doc, 'structure', None)
    if st is None:
        return None, None
    try:
        if hasattr(st, 'site_properties'):                       # pymatgen Structure
            species = [site.species_string for site in st]
            mm = st.site_properties.get('magmom')
        else:                                                    # dict form
            species = [s['species'][0]['element'] for s in st['sites']]
            mm = [s.get('properties', {}).get('magmom') for s in st['sites']]
            if all(v is None for v in mm):
                mm = None
        if mm is not None:
            mm = [float(v) if v is not None else 0.0 for v in mm]
            if len(mm) != len(species):
                mm = None
        return species, mm
    except Exception:                                            # noqa: BLE001
        return None, None


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    cache[str(d['material_id'])] = d
    return cache


def fetch(ids, args, cache):
    todo = [m for m in ids if m not in cache]
    if not todo:
        print('  [fetch] cache already complete - nothing to fetch.')
        return set()
    if args.mock_docs:
        print(f'  [fetch] MOCK MODE: {args.mock_docs}')
        mock = {}
        with open(args.mock_docs) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    mock[str(d['material_id'])] = d
        with open(args.cache, 'a') as out:
            for m in todo:
                d = mock.get(m, {'material_id': m, 'missing': True})
                cache[m] = d
                out.write(json.dumps(d) + '\n')
        return set()
    try:
        from mp_api.client import MPRester
    except Exception:                                            # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(f'\nFailed to import mp_api.client (traceback above).\n'
                 f'  interpreter: {sys.executable}\n'
                 f'Use the dedicated env:  conda activate mpfetch')
    failed = set()
    key = args.api_key or None
    n_chunks = (len(todo) + args.chunk - 1) // args.chunk
    print(f'  [fetch] {len(todo)} ids in {n_chunks} chunks of <= {args.chunk} '
          f'({len(cache)} cached). Structures are heavy: expect ~30-90 min full.')
    with MPRester(key) as mpr:
        coll = mpr.materials if args.endpoint == 'materials' else mpr.materials.summary
        for c in range(n_chunks):
            part = todo[c * args.chunk:(c + 1) * args.chunk]
            docs = None
            for attempt in range(1, args.retries + 1):
                try:
                    docs = coll.search(material_ids=part,
                                       fields=['material_id', 'structure'])
                    break
                except Exception as exc:                         # noqa: BLE001
                    print(f'  [fetch] chunk {c + 1}/{n_chunks} attempt {attempt} '
                          f'failed: {exc}', file=sys.stderr)
                    time.sleep(5 * attempt)
            if docs is None:
                failed.update(part)
                continue
            got = {}
            for d in docs:
                mid = str(d.get('material_id') if isinstance(d, dict)
                          else getattr(d, 'material_id', ''))
                species, mm = extract(d)
                got[mid] = ({'material_id': mid, 'no_structure': True} if species is None
                            else {'material_id': mid, 'species': species, 'magmom': mm})
            with open(args.cache, 'a') as out:
                for m in part:
                    d = got.get(m, {'material_id': m, 'missing': True})
                    cache[m] = d
                    out.write(json.dumps(d) + '\n')
            if (c + 1) % 10 == 0 or c + 1 == n_chunks:
                print(f'  [fetch] {c + 1}/{n_chunks} chunks done', flush=True)
    return failed


def main():
    args = parse_args()
    print(f'jsonl: {args.jsonl}\nout:   {args.out}')
    rows = read_jsonl(args.jsonl)
    print(f'  jsonl records: {len(rows)} ({sum(r["jsonl_ok"] for r in rows)} dataset-eligible)')
    fetch_ids = [r['material_id'] for r in rows if r['jsonl_ok']]
    if args.limit:
        fetch_ids = fetch_ids[:args.limit]
        print(f'  [fetch] --limit {args.limit}: smoke run')
    cache = load_cache(args.cache)
    failed = fetch(fetch_ids, args, cache)

    stat = {}
    residuals = []
    with open(args.out, 'w') as out:
        for r in rows:
            mid = r['material_id']
            doc = cache.get(mid, {})
            if not r['jsonl_ok']:
                s = 'jsonl_skipped'
            elif mid in failed or mid not in cache:
                s = 'fetch_failed'
            elif doc.get('missing'):
                s = 'missing_mp'
            elif doc.get('no_structure'):
                s = 'no_structure'
            elif doc.get('magmom') is None:
                s = 'no_magmom'
            elif r['poscar_species'] is None:
                s = 'poscar_no_symbols'
            elif r['poscar_species'] != doc.get('species'):
                s = 'species_mismatch'
            else:
                s = 'aligned'
            rec = {'material_id': mid, 'formula': r['formula'], 'status': s,
                   'n_sites': len(doc['species']) if doc.get('species') else None,
                   'species': doc.get('species'), 'magmom': doc.get('magmom'),
                   'cell_total': r['cell_total']}
            if s in ('aligned', 'species_mismatch', 'poscar_no_symbols') \
                    and doc.get('magmom') is not None and r['cell_total'] is not None:
                rec['residual'] = round(abs(sum(doc['magmom']) - r['cell_total']), 6)
                if s == 'aligned':
                    residuals.append(rec['residual'])
            out.write(json.dumps(rec) + '\n')
            stat[s] = stat.get(s, 0) + 1

    digest = hashlib.md5(open(args.out, 'rb').read()).hexdigest()[:10]
    print(f'\n── magmom side-table ──\n  rows: {len(rows)} -> {args.out} (md5 {digest})')
    for k in ('aligned', 'no_magmom', 'species_mismatch', 'poscar_no_symbols',
              'no_structure', 'missing_mp', 'fetch_failed', 'jsonl_skipped'):
        if k in stat:
            print(f'  {k:>18}: {stat[k]}')
    if residuals:
        residuals.sort()
        q = lambda f: residuals[min(int(f * len(residuals)), len(residuals) - 1)]
        print(f'  |Σmagmom − cell_total| on aligned rows: median {q(0.5):.3f}  '
              f'p90 {q(0.9):.3f}  p99 {q(0.99):.3f}  max {residuals[-1]:.3f}  '
              f'(interstitial contribution -> learned residual in the site-decomposition model)')
    if failed:
        print(f'\n  {len(failed)} ids in FAILED chunks — re-run this script to retry.')
        sys.exit(2)
    print('DONE')


if __name__ == '__main__':
    main()
