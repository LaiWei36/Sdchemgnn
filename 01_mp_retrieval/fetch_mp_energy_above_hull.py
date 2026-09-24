#!/usr/bin/env python
"""Fetch energy_above_hull from the Materials Project for our structures.

WHY. Kaba et al. (Phys. Rev. Materials 7, 044407) train and evaluate on MP
v2020.06 filtered to energy_above_hull <= 0.1 eV/atom, 78,462 structures. Our
corpus is unfiltered and deliberately retains the unstable, strongly magnetic
tail. Section 3.5 currently *infers* that the distance between their 0.043
uB/atom and our forest's 0.057 is corpus difficulty rather than architecture.
With a stability column that inference becomes a measurement: we can restrict
our own evaluation to their window and compare on a matched population.

This script only fetches. Everything downstream is offline.

Conventions follow fetch_mp_magnetic_ordering.py: chunked queries, retry with backoff,
an append-only jsonl cache so a re-run resumes, and tombstones for ids the API
does not return.

    export MP_API_KEY=...            # or rely on the pymatgen config
    python fetch_mp_energy_above_hull.py --ids ordering_sidetable.csv --out ehull_sidetable.csv

Needs mp-api and internet. If mp_api fails to import inside the training env
(usually a pydantic / emmet-core conflict), use a dedicated minimal env -- this
script needs nothing else:

    conda create -n mpfetch python=3.11 -y && conda activate mpfetch
    pip install mp-api
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time

FIELDS = ['material_id', 'energy_above_hull']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ids', default='./ordering_sidetable.csv',
                   help='CSV with a material_id column (default: the ordering side-table, '
                        'which covers all 154,377 structures)')
    p.add_argument('--out', default='./ehull_sidetable.csv')
    p.add_argument('--cache', default='./mp_ehull_cache.jsonl',
                   help='resumable append-only response cache')
    p.add_argument('--api-key', default=os.environ.get('MP_API_KEY', ''))
    p.add_argument('--endpoint', default='summary',
                   help="mpr.materials.<endpoint>; 'summary' carries energy_above_hull")
    p.add_argument('--chunk', type=int, default=500)
    p.add_argument('--retries', type=int, default=4)
    p.add_argument('--limit', type=int, default=0, help='fetch at most N ids (smoke test)')
    return p.parse_args()


def main():
    a = parse_args()
    with open(a.ids) as fh:
        ids = [r['material_id'] for r in csv.DictReader(fh) if r.get('material_id')]
    seen = set()
    ids = [m for m in ids if not (m in seen or seen.add(m))]
    print(f'ids file: {a.ids}  ->  {len(ids):,} unique material_ids')

    cache = {}
    if os.path.exists(a.cache):
        with open(a.cache) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    cache[d['material_id']] = d
                except Exception:
                    continue
        print(f'cache:    {a.cache}  ->  {len(cache):,} already fetched')

    todo = [m for m in ids if m not in cache]
    if a.limit:
        todo = todo[:a.limit]
    print(f'to fetch: {len(todo):,}')

    if todo:
        try:
            from mp_api.client import MPRester
        except Exception:
            import traceback; traceback.print_exc()
            sys.exit('\nFailed to import mp_api.client -- see the traceback above.\n'
                     'Try a dedicated env:  conda create -n mpfetch python=3.11 -y\n'
                     '                      conda activate mpfetch && pip install mp-api\n')
        n = (len(todo) + a.chunk - 1) // a.chunk
        with MPRester(a.api_key or None) as mpr:
            coll = getattr(mpr.materials, a.endpoint)
            for c in range(n):
                part = todo[c * a.chunk:(c + 1) * a.chunk]
                docs = None
                for attempt in range(1, a.retries + 1):
                    try:
                        docs = coll.search(material_ids=part, fields=FIELDS)
                        break
                    except Exception as exc:
                        print(f'  chunk {c+1}/{n} attempt {attempt} failed: {exc}',
                              file=sys.stderr)
                        time.sleep(5 * attempt)
                if docs is None:
                    print(f'  chunk {c+1}/{n} GAVE UP -- re-run to retry', file=sys.stderr)
                    continue
                got = {}
                for d in docs:
                    mid = str(getattr(d, 'material_id', None))
                    val = getattr(d, 'energy_above_hull', None)
                    got[mid] = {'material_id': mid,
                                'energy_above_hull': None if val is None else float(val)}
                with open(a.cache, 'a') as out:
                    for m in part:
                        rec = got.get(m, {'material_id': m, 'energy_above_hull': None,
                                          'missing': True})
                        cache[m] = rec
                        out.write(json.dumps(rec) + '\n')
                if (c + 1) % 20 == 0 or c + 1 == n:
                    print(f'  {c+1}/{n} chunks done')

    with open(a.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['material_id', 'energy_above_hull'])
        nok = 0
        for m in ids:
            v = cache.get(m, {}).get('energy_above_hull')
            w.writerow([m, '' if v is None else f'{v:.6f}'])
            nok += v is not None
    print(f'\nwrote {a.out}: {nok:,}/{len(ids):,} with a value '
          f'({len(ids)-nok:,} missing)')
    # NB: test `is None` explicitly. `value or 9e9` is WRONG here because a
    # ground-state structure has energy_above_hull == 0.0, which is falsy, and
    # would be counted as unstable (this cost 32,725 structures once).
    def _eh(m):
        v = cache.get(m, {}).get('energy_above_hull')
        return 9e9 if v is None else float(v)
    stable = sum(1 for m in ids if _eh(m) <= 0.1)
    print(f'energy_above_hull <= 0.1 eV/atom: {stable:,} of {len(ids):,} '
          f'({100*stable/max(len(ids),1):.1f}%)   [Kaba et al. retained 69%]')


if __name__ == '__main__':
    sys.exit(main())
