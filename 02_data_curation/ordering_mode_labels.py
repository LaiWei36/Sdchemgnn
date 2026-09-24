"""Ordering-mode labels for the auxiliary ordering-diagnostic head. Pure stdlib.

Derives one integer per material from the ordering side-table:
    -1  unsupervised (no mode loss)      0  "net ~ 0"      1  "partial" (FiM)
     2  "aligned" (FM)

Derivation rule (first match wins; y = the CURATED label the model trains on):
    aux_mask == 1 in the side-table  ................................  -1
    material_id in collapse_suspects.csv  ...........................  -1
    ordering not in {NM, AFM, FM, FiM}  (Unknown / missing)  ........  -1
    ordering in {NM, AFM} and |y| >= near_zero  (threshold anomaly) .  -1
    |y| < near_zero  ................................................   0
    ordering == FM  .................................................   2
    ordering == FiM  ................................................   1

STALENESS TRIPWIRE:
mode labels depend on the manifest through aux_mask and y. build_modes()
therefore cross-checks the side-table's embedded manifest columns against the
manifest the run actually loads, ROW BY ROW, and raises on any mismatch —
including a re-derivation of the manifest_shift mask condition. A manifest edit
(a new curation round) without regenerating the side-table is
a hard startup error, never a silent skew. Regeneration is cheap:
fetch_mp_magnetic_ordering.py rebuilds from its cache in seconds, no API calls.

Canary ids (present in the MP magnetic dataset) pin the derivation end-to-end;
they are skipped silently if absent (other datasets).
"""

import csv
import hashlib
import os

ORDER_CLASSES = ('NM', 'AFM', 'FM', 'FiM')
MODE_NAMES = {-1: 'unsup', 0: 'net~0', 1: 'partial', 2: 'aligned'}

# material_id -> expected mode on THIS dataset (audit round 2 ground truth):
#   mp-831163   LiMnBO3      manifest-relabeled 0->5   -> masked
#   mp-759193   FeOF         FM, y=5                   -> aligned
#   mp-1235435  LiGd6In2O12  FiM, y=0.16 (compensated) -> net~0
#   mp-1275201  Mn2CuO4      FiM, y=7                  -> partial
CANARIES = {'mp-831163': -1, 'mp-759193': 2, 'mp-1235435': 0, 'mp-1275201': 1}

MASK_SHIFT_TOL = 0.5      # must match fetch_mp_magnetic_ordering.py --mask-shift-tol
_NUM_TOL = 1e-6


def file_md5(path, n=10):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()[:n]


def load_collapse_ids(path):
    if not path:
        return set(), None
    ids = set()
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            ids.add(row['material_id'])
    return ids, file_md5(path)


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def build_modes(sidetable_path, manifest, collapse_path=None,
                cap=30.0, near_zero=0.5, canaries=CANARIES):
    """Returns (modes: dict material_id -> int, meta: dict).

    `manifest` is the dict the train script already loaded via load_manifest():
    {material_id: (action, new_or_None)}. Raises RuntimeError on ANY side-table /
    manifest inconsistency or canary mismatch.
    """
    collapse, collapse_md5 = load_collapse_ids(collapse_path)
    modes, census = {}, {-1: 0, 0: 0, 1: 0, 2: 0}
    man_seen, problems = set(), []
    post_edit_drops = []
    n_rows = 0

    with open(sidetable_path, newline='') as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            mid = row['material_id']
            act = (row.get('manifest_action') or '').strip()
            new = _f(row.get('manifest_new'))
            raw = _f(row.get('jsonl_raw_fu'))
            reason = (row.get('mask_reason') or '').strip()

            # ── staleness tripwire: side-table manifest columns vs loaded manifest ──
            m = manifest.get(mid)
            if act == '':
                if m is not None:
                    if m[0] == 'drop':
                        # A drop APPENDED to the manifest after this side-table
                        # was generated is harmless by construction: the id is
                        # removed from every split before mode attach, so its
                        # mode label is never consumed. Allowed (and counted);
                        # relabels added post-generation still fail loudly below
                        # because they DO change the curated y the modes read.
                        man_seen.add(mid)
                        post_edit_drops.append(mid)
                    else:
                        problems.append(f'{mid}: manifest has action {m[0]!r}, side-table has none')
            else:
                man_seen.add(mid)
                if m is None:
                    problems.append(f'{mid}: side-table action {act!r}, absent from manifest')
                elif m[0] != act:
                    problems.append(f'{mid}: action mismatch (side-table {act!r} vs manifest {m[0]!r})')
                elif act != 'drop' and m[1] is not None and new is not None \
                        and abs(m[1] - new) > _NUM_TOL:
                    problems.append(f'{mid}: new mismatch ({new} vs {m[1]})')
                # re-derive the manifest_shift mask condition (catches a side-table
                # built with a different tolerance or a different manifest). Rows
                # sitting exactly ON the tolerance boundary are skipped: the CSV
                # stores 6 decimals, so a full-precision shift of 0.5000004 masks
                # in the fetch script but re-derives as exactly 0.5 here.
                if act != 'drop' and raw is not None and new is not None \
                        and reason in ('ok', 'manifest_shift'):
                    shift = abs(max(-cap, min(cap, raw)) - new)
                    if abs(shift - MASK_SHIFT_TOL) > 1e-5:
                        expect = 'manifest_shift' if shift > MASK_SHIFT_TOL else 'ok'
                        if reason != expect:
                            problems.append(f'{mid}: mask_reason {reason!r}, re-derived {expect!r}')
            if len(problems) >= 8:
                break

            # ── curated label y (what the model trains on) ──
            if act not in ('', 'drop') and new is not None:
                y = new
            elif raw is not None:
                y = max(-cap, min(cap, raw))
            else:
                y = None

            # ── the mode rule (first match wins) ──
            ordering = (row.get('ordering') or '').strip()
            if str(row.get('aux_mask', '1')).strip() != '0':
                v = -1
            elif mid in collapse:
                v = -1
            elif ordering not in ORDER_CLASSES:
                v = -1
            elif y is None:
                v = -1
            elif ordering in ('NM', 'AFM') and abs(y) >= near_zero:
                v = -1                                   # threshold anomaly
            elif abs(y) < near_zero:
                v = 0
            elif ordering == 'FM':
                v = 2
            else:                                        # FiM, |y| >= near_zero
                v = 1
            modes[mid] = v
            census[v] += 1

    missing = [m for m in manifest if m not in man_seen][:5]
    if missing:
        problems.append(f'{len([m for m in manifest if m not in man_seen])} manifest ids '
                        f'absent from side-table, e.g. {missing[:3]}')
    if problems:
        raise RuntimeError(
            'ORDERING SIDE-TABLE IS STALE vs label_manifest — the mode labels would be '
            'silently wrong. Regenerate it: fetch_mp_magnetic_ordering.py '
            'rebuilds from its cache in seconds. First mismatches:\n  '
            + '\n  '.join(problems[:8]))

    for cid, expect in (canaries or {}).items():
        if cid in modes and modes[cid] != expect:
            raise RuntimeError(f'ORDERING-MODE CANARY FAILED: {cid} derived '
                               f'{modes[cid]} ({MODE_NAMES[modes[cid]]}), expected '
                               f'{expect} ({MODE_NAMES[expect]}) — derivation rule '
                               f'or inputs changed; do not train on these modes.')

    if post_edit_drops:
        print(f'    (ordering side-table: {len(post_edit_drops)} manifest drop(s) '
              f'added after generation — allowed, dropped ids never reach mode '
              f'attach: {", ".join(post_edit_drops[:4])}'
              + (' ...' if len(post_edit_drops) > 4 else '') + ')')
    meta = {
        'sidetable': os.path.abspath(sidetable_path),
        'post_edit_drops': len(post_edit_drops),
        'sidetable_md5': file_md5(sidetable_path),
        'collapse_suspects': os.path.abspath(collapse_path) if collapse_path else None,
        'collapse_md5': collapse_md5,
        'near_zero': near_zero, 'cap': cap, 'n_rows': n_rows,
        'census': {MODE_NAMES[k]: census[k] for k in (-1, 0, 1, 2)},
        'n_labeled': census[0] + census[1] + census[2],
    }
    return modes, meta


def _load_manifest_like_train(path):
    man = {}
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            try:
                new = float(row['new'])
            except (ValueError, TypeError, KeyError):
                new = None
            man[row['material_id']] = (row['action'], new)
    return man


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Standalone mode census / staleness check.')
    ap.add_argument('--sidetable', default='./ordering_sidetable.csv')
    ap.add_argument('--manifest', default='./label_manifest.csv')
    ap.add_argument('--collapse', default='./collapse_suspects.csv')
    ap.add_argument('--near-zero', type=float, default=0.5)
    a = ap.parse_args()
    man = _load_manifest_like_train(a.manifest)
    collapse = a.collapse if a.collapse and os.path.exists(a.collapse) else None
    modes, meta = build_modes(a.sidetable, man, collapse, near_zero=a.near_zero)
    print(f'side-table: {meta["sidetable"]} (md5 {meta["sidetable_md5"]}, {meta["n_rows"]} rows)')
    print(f'collapse suspects: {meta["collapse_suspects"]} (md5 {meta["collapse_md5"]})')
    print(f'census: {meta["census"]}  ->  {meta["n_labeled"]} labeled '
          f'({100.0 * meta["n_labeled"] / max(meta["n_rows"], 1):.1f}%)')
    print('consistency + canaries: OK')
