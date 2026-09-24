"""Per-site magnetic-moment labels for the per-site moment decomposition.

Turns magmom_sidetable.jsonl (fetch_mp_site_moments.py) + ordering_sidetable.csv
(fetch_mp_magnetic_ordering.py, for the RAW per-f.u. value) + label_manifest.csv into,
per dataset graph:
    node_magmom  [n_i]  signed uB per atom, POSCAR site order  (zeros if unsup)
    node_sup     {0,1}  1 = this graph's node/residual losses fire
    resid_target float  (cell_total - sum(m)) clipped to +/- resid_clip
    z_fu         float  formula units per cell (the y = cell_total / Z bridge)

MASKING POLICY (first match wins):
    missing from either side-table .......................... mask
    status != aligned/no_magmom (none exist in the fetched table) mask
    manifest action 'relabel0' ............................... mask (we distrust
        MP's magnetic solution there -- do NOT force zeros)
    in collapse_suspects.csv ................................. mask
    relabel with |capped(raw) - curated| > shift_tol ......... mask (site moments
        no longer sum to the training label)
    aligned but len(magmom) != graph node count .............. mask (+ HALT if
        > 0.5% of aligned rows -- graph/POSCAR assumption broken)
    z suspect (formula/ratio disagree, see below) ............ mask
    no_magmom and (|cell_total| > zero_tol or |curated| > zero_tol)  mask
    no_magmom, both totals ~ 0 ............................... SUPERVISED, all-
        zero site targets, resid_target = 0 (the negative examples)
    aligned, consistent ...................................... SUPERVISED

Z DERIVATION:
    z_ratio   = cell_total / raw_fu            where |raw_fu| > 0.1 (exact by
                                               construction: MP normalized raw)
    z_formula = graph_node_count / n_atoms(reduced formula)   must be a positive
                                               integer (tol 1e-3)
    both derivable and disagreeing by > z_ratio_tol (relative) -> z_suspect:
    row masked from node/resid supervision (total loss still applies), z takes
    z_ratio (it matches the label scale). > 1% z_suspect rows -> HALT.

VERIFY LAYERS:
    build:  residual recompute vs the fetch's own 'residual' field (sampled);
            canaries (mask policy + sign structure); sign-convention sweep
            (curated y > 5  =>  sum(m) > 0 on >= 98% of supervised rows).
    attach: full-sweep len(magmom) == slice width for EVERY row it writes.
    cache:  verify_magmom_in_cache -- every cached batch, every graph, exact
            tensor equality through the SAME accessors the loss consumes.

Accessors at the bottom are THE read path: the loss, the verifiers and the
dumps all go through them (never a parallel attribute read).
"""

import csv
import hashlib
import json
import math
import os
import re

import torch

MASK_SHIFT_TOL_GUARD = 1e-5      # boundary guard, same as ordering_mode_labels.py

# Directional canaries (skipped silently if the id is absent from the dataset):
#   mp-831163   LiMnBO3      relabelR2 0->5 (straddle median) -> MUST be masked
#                            by a MANIFEST-driven reason (label_shift here)
#   mp-1235435  LiGd6In2O12  compensated FiM     -> supervised, mixed signs,
#                                                  sum|m| > 5 (six Gd sites)
#   mp-759193   FeOF         FM, y = 5           -> supervised, sum(m) > 1
CANARY_MASKED = 'mp-831163'
CANARY_COMPENSATED = 'mp-1235435'
CANARY_FM = 'mp-759193'


def file_md5(path, n=10):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()[:n]


# ── reduced-formula atom count (for z_formula) ─────────────────────────────────
def _expand_formula(s):
    out = {}
    while s:
        if s[0] == '(':
            depth, j = 1, 1
            while depth > 0 and j < len(s):
                depth += {'(': 1, ')': -1}.get(s[j], 0)
                j += 1
            inner = _expand_formula(s[1:j - 1])
            m2 = re.match(r'(\d+)', s[j:])
            mult = int(m2.group(1)) if m2 else 1
            s = s[j + (m2.end() if m2 else 0):]
            for k, v in inner.items():
                out[k] = out.get(k, 0) + v * mult
        else:
            m = re.match(r'([A-Z][a-z]?)(\d*)', s)
            if not m:
                break
            out[m.group(1)] = out.get(m.group(1), 0) + int(m.group(2) or 1)
            s = s[m.end():]
    return out


def reduced_atom_count(formula):
    """Total atoms in one reduced formula unit, or None if unparsable."""
    counts = _expand_formula(str(formula or ''))
    total = sum(counts.values())
    return total if total > 0 else None


def _f(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_collapse_ids(path):
    if not path:
        return set(), None
    ids = set()
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            ids.add(row['material_id'])
    return ids, file_md5(path)


def load_raw_fu(ordering_csv):
    """material_id -> (raw per-f.u. value or None, formula) from the ordering
    side-table (its jsonl_raw_fu column is the RAW MP normalized value)."""
    out = {}
    with open(ordering_csv, newline='') as fh:
        for row in csv.DictReader(fh):
            out[row['material_id']] = (_f(row.get('jsonl_raw_fu')),
                                       (row.get('formula') or '').strip())
    return out


# ── build ──────────────────────────────────────────────────────────────────────
def build_magmom_labels(magmom_jsonl, ordering_csv, manifest, node_counts,
                        collapse_path=None, cap=30.0, shift_tol=0.5,
                        zero_tol=0.1, resid_clip=10.0, z_ratio_tol=0.02,
                        halt_nsite_frac=0.005, halt_zsuspect_frac=0.01,
                        resid_recompute_n=200):
    """Returns (labels, meta).

    labels: {material_id: {'sup': 0/1, 'm': list[float] len n_i (present iff
             sup), 'resid': float, 'z': float, 'why': str}} -- one entry for
             EVERY id in node_counts (= every graph in the dataset master).
    manifest: {material_id: (action, new)} as loaded by the train script.
    node_counts: {material_id: graph node count} from the dataset master store.
    Raises RuntimeError on any HALT condition, canary failure, or sign-sweep
    failure -- before a single epoch can train on bad node labels.
    """
    collapse, collapse_md5 = load_collapse_ids(collapse_path)
    raw_fu = load_raw_fu(ordering_csv)

    labels = {}
    census = {}
    n_sup = n_zero = 0
    n_sup_nodes = 0
    z_suspects = nsite_mismatch = n_aligned_seen = 0
    resid_used = []          # |cell_total - sum(m)| on supervised aligned rows
    resid_checked = resid_bad = 0
    sign_pos = sign_tot = 0

    def _mask(mid, why):
        labels[mid] = {'sup': 0, 'resid': 0.0,
                       'z': labels.get(mid, {}).get('z', 1.0), 'why': why}
        census[why] = census.get(why, 0) + 1

    with open(magmom_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mid = str(rec.get('material_id', ''))
            n_graph = node_counts.get(mid)
            if n_graph is None:
                continue                      # not in the dataset (drops etc.)
            status = rec.get('status', '')
            cell_total = _f(rec.get('cell_total'))
            magmom = rec.get('magmom')
            raw, o_formula = raw_fu.get(mid, (None, ''))
            formula = (rec.get('formula') or '').strip() or o_formula

            # ── curated label y (what the model trains on) ──
            act, new = ('', None)
            m_row = manifest.get(mid)
            if m_row is not None:
                act, new = m_row[0], m_row[1]
            if act not in ('', 'drop') and new is not None:
                y_cur = new
            elif raw is not None:
                y_cur = max(-cap, min(cap, raw))
            else:
                y_cur = None

            # ── Z (needed for EVERY row -- the prediction path divides by it) ──
            z_ratio = None
            if raw is not None and abs(raw) > 0.1 and cell_total is not None:
                zr = cell_total / raw
                if math.isfinite(zr) and zr > 0.4:
                    z_ratio = zr
            z_formula = None
            n_red = reduced_atom_count(formula)
            if n_red:
                zf = n_graph / n_red
                if abs(zf - round(zf)) <= 1e-3 and round(zf) >= 1:
                    z_formula = float(round(zf))
            z_ok = True
            if z_ratio is not None and z_formula is not None:
                if abs(z_ratio - z_formula) / z_formula > z_ratio_tol:
                    z_ok = False
            z = (z_formula if z_formula is not None else
                 (z_ratio if z_ratio is not None else 1.0))
            if not z_ok:
                z = z_ratio                   # matches the label scale
            labels[mid] = {'z': float(z)}     # provisional; filled below

            # ── mask cascade (first match wins) ──
            if mid not in raw_fu:
                _mask(mid, 'missing_ordering_row'); continue
            if not z_ok:
                z_suspects += 1
                _mask(mid, 'z_suspect'); continue
            if z_formula is None and z_ratio is None:
                _mask(mid, 'z_underivable'); continue
            if status == 'no_magmom':
                if (cell_total is not None and abs(cell_total) <= zero_tol
                        and y_cur is not None and abs(y_cur) <= zero_tol):
                    labels[mid] = {'sup': 1, 'm': [0.0] * n_graph, 'resid': 0.0,
                                   'z': float(z), 'why': 'zero'}
                    census['zero'] = census.get('zero', 0) + 1
                    n_sup += 1; n_zero += 1; n_sup_nodes += n_graph
                else:
                    _mask(mid, 'no_magmom_magnetic')
                continue
            if status != 'aligned':
                _mask(mid, f'status_{status}'); continue
            n_aligned_seen += 1
            if act == 'relabel0':
                _mask(mid, 'relabel0'); continue
            if mid in collapse:
                _mask(mid, 'collapse_suspect'); continue
            if act not in ('', 'drop') and new is not None and raw is not None:
                shift = abs(max(-cap, min(cap, raw)) - new)
                if shift > shift_tol and abs(shift - shift_tol) > MASK_SHIFT_TOL_GUARD:
                    _mask(mid, 'label_shift'); continue
            if not isinstance(magmom, list) or cell_total is None:
                _mask(mid, 'magmom_unusable'); continue
            if len(magmom) != n_graph:
                nsite_mismatch += 1
                _mask(mid, 'nsite_mismatch'); continue

            # ── SUPERVISED aligned row ──
            m = [float(v) for v in magmom]
            resid_raw = cell_total - sum(m)
            rec_resid = _f(rec.get('residual'))
            if rec_resid is not None and resid_checked < resid_recompute_n:
                resid_checked += 1
                if abs(abs(resid_raw) - rec_resid) > 1e-3:
                    resid_bad += 1
            resid = max(-resid_clip, min(resid_clip, resid_raw))
            labels[mid] = {'sup': 1, 'm': m, 'resid': float(resid),
                           'z': float(z), 'why': 'ok'}
            census['ok'] = census.get('ok', 0) + 1
            n_sup += 1; n_sup_nodes += n_graph
            resid_used.append(abs(resid_raw))
            if y_cur is not None and y_cur > 5.0:
                sign_tot += 1
                if sum(m) > 0:
                    sign_pos += 1

    # ids in the dataset but absent from the magmom side-table
    for mid, n_graph in node_counts.items():
        if mid not in labels:
            labels[mid] = {'z': 1.0}
            _mask(mid, 'missing_magmom_row')

    # ── HALT conditions ──
    problems = []
    if n_aligned_seen and nsite_mismatch / n_aligned_seen > halt_nsite_frac:
        problems.append(
            f'{nsite_mismatch}/{n_aligned_seen} aligned rows have '
            f'len(magmom) != graph node count (> {halt_nsite_frac:.1%}) — the '
            f'graph==POSCAR-site assumption is broken; do NOT train node labels.')
    n_all = max(len(node_counts), 1)
    if z_suspects / n_all > halt_zsuspect_frac:
        problems.append(
            f'{z_suspects}/{n_all} rows are z_suspect (> {halt_zsuspect_frac:.1%})'
            f' — Z derivation is unreliable; investigate before training.')
    if resid_bad:
        problems.append(
            f'{resid_bad}/{resid_checked} sampled rows fail the residual '
            f'recompute vs the fetch\'s own residual field — parsing or site '
            f'order is wrong.')
    if sign_tot >= 50 and sign_pos / sign_tot < 0.98:
        problems.append(
            f'sign-convention sweep failed: only {sign_pos}/{sign_tot} supervised '
            f'rows with curated y > 5 have sum(m) > 0 (>= 98% expected under the '
            f'MP gauge) — sign handling is broken somewhere.')

    # ── canaries (skipped silently if absent from the dataset) ──
    # Canary mask discipline: a MANIFEST/AUDIT-driven mask is legitimate
    # (curation can change); a data-integrity mask (z_suspect, nsite_mismatch,
    # status change, unusable magmom) means the inputs shifted under us and
    # must fail loudly, not silently exclude.
    _CANARY_OK_MASKS = ('relabel0', 'label_shift', 'collapse_suspect')
    c = labels.get(CANARY_MASKED)
    if c is not None and not (c['sup'] == 0 and c['why'] in _CANARY_OK_MASKS):
        problems.append(f'CANARY {CANARY_MASKED}: relabelR2 0->5 must be masked '
                        f'by a manifest-driven reason {_CANARY_OK_MASKS}, got '
                        f'sup={c["sup"]} why={c["why"]!r}.')
    c = labels.get(CANARY_COMPENSATED)
    if c is not None:
        if c.get('sup'):
            m = c['m']
            if not (min(m) < -0.5 and max(m) > 0.5 and sum(abs(v) for v in m) > 5):
                problems.append(f'CANARY {CANARY_COMPENSATED}: compensated FiM must '
                                f'have mixed signs and sum|m| > 5; got '
                                f'min={min(m):.2f} max={max(m):.2f} '
                                f'sum|m|={sum(abs(v) for v in m):.2f}.')
        elif c.get('why') not in _CANARY_OK_MASKS:
            problems.append(f'CANARY {CANARY_COMPENSATED}: unexpectedly masked '
                            f'({c.get("why")!r}) — data-integrity mask on a '
                            f'known-good row; inputs changed.')
    c = labels.get(CANARY_FM)
    if c is not None:
        if c.get('sup'):
            if sum(c['m']) <= 1.0:
                problems.append(f'CANARY {CANARY_FM}: FM row expected sum(m) > 1, '
                                f'got {sum(c["m"]):.2f}.')
        elif c.get('why') not in _CANARY_OK_MASKS:
            problems.append(f'CANARY {CANARY_FM}: unexpectedly masked '
                            f'({c.get("why")!r}) — data-integrity mask on a '
                            f'known-good row; inputs changed.')

    if problems:
        raise RuntimeError('MAGMOM NODE LABELS REFUSED (the site-decomposition model):\n  '
                           + '\n  '.join(problems))

    resid_used.sort()
    q = (lambda f: resid_used[min(int(f * len(resid_used)),
                                  len(resid_used) - 1)]) if resid_used else (lambda f: float('nan'))
    meta = {
        'magmom_sidetable': os.path.abspath(magmom_jsonl),
        'magmom_md5': file_md5(magmom_jsonl),
        'ordering_csv_md5': file_md5(ordering_csv),
        'collapse_md5': collapse_md5,
        'census': dict(sorted(census.items(), key=lambda kv: -kv[1])),
        'n_rows': len(labels), 'n_sup': n_sup, 'n_zero': n_zero,
        'n_sup_nodes': n_sup_nodes,
        'z_suspects': z_suspects, 'nsite_mismatch': nsite_mismatch,
        'resid_recompute': f'{resid_checked - resid_bad}/{resid_checked} OK',
        'sign_sweep': f'{sign_pos}/{sign_tot} positive',
        'resid_q': {'median': q(0.5), 'p90': q(0.9), 'p99': q(0.99)},
        'shift_tol': shift_tol, 'zero_tol': zero_tol, 'resid_clip': resid_clip,
    }
    return labels, meta


# ── attach into the collated master (+ slices) ─────────────────────────────────
def attach_magmom_labels(dataset, labels):
    """Write node_magmom [sum n_i] (node-aligned slices cloned from slices['x'])
    and the per-graph node_sup / resid_target / z_fu into the COLLATED MASTER
    store, so every split view and every cached batch carries them. Full-sweep
    invariant inside: every supervised row's list length must equal its x-slice
    width. Returns (ids_in_master_order, n_sup_graphs)."""
    store = getattr(dataset, '_data', None)
    if store is None:
        store = dataset.data
    sl = getattr(dataset, 'slices', None)
    if sl is None:
        sl = getattr(dataset, '_slices', None)
    if sl is None or 'x' not in sl:
        raise RuntimeError("cannot locate slices['x'] on the dataset (PyG API "
                           "drift?) — node-aligned attach would silently break.")
    raw_ids = store.material_id
    ids = [str(x[0] if isinstance(x, (list, tuple)) else x) for x in raw_ids]
    xb = sl['x'].tolist()                        # node boundaries [G+1]
    if len(xb) != len(ids) + 1:
        raise RuntimeError(f"slices['x'] has {len(xb)} offsets for {len(ids)} "
                           f'graphs — unexpected master layout.')
    n_nodes_total = int(xb[-1])
    node_m = torch.zeros(n_nodes_total, dtype=torch.float32)
    sup = torch.zeros(len(ids), dtype=torch.int8)
    resid = torch.zeros(len(ids), dtype=torch.float32)
    z_fu = torch.ones(len(ids), dtype=torch.float32)
    n_sup = 0
    for j, mid in enumerate(ids):
        rec = labels.get(mid)
        if rec is None:
            raise RuntimeError(f'ATTACH FAILED: {mid} is in the dataset master '
                               f'but has no label record — build_magmom_labels '
                               f'must cover every dataset id.')
        z_fu[j] = rec.get('z', 1.0)
        if rec.get('sup'):
            lo, hi = xb[j], xb[j + 1]
            m = rec['m']
            if len(m) != hi - lo:                # full-sweep invariant
                raise RuntimeError(f'ATTACH INVARIANT FAILED: {mid} has '
                                   f'{len(m)} site labels for a {hi - lo}-node '
                                   f'graph slice.')
            node_m[lo:hi] = torch.tensor(m, dtype=torch.float32)
            sup[j] = 1
            resid[j] = rec['resid']
            n_sup += 1
    store.node_magmom = node_m
    store.node_sup = sup
    store.resid_target = resid
    store.z_fu = z_fu
    sl['node_magmom'] = sl['x'].clone()          # node-aligned boundaries
    per_graph = torch.arange(len(ids) + 1)
    sl['node_sup'] = per_graph
    sl['resid_target'] = per_graph.clone()
    sl['z_fu'] = per_graph.clone()
    return ids, n_sup


# ── THE read path (loss, verifiers and dumps all go through these) ─────────────
def get_node_magmom(data):
    """Per-node signed site-moment target [num_nodes] (zeros where unsup)."""
    return data.node_magmom.reshape(-1)


def get_node_sup(data):
    """Per-graph supervision flag [num_graphs] (int8, 1 = supervised)."""
    return data.node_sup.reshape(-1)


def get_resid_target(data):
    """Per-graph clipped residual target [num_graphs] (0 where unsup)."""
    return data.resid_target.reshape(-1)


def get_z_fu(data):
    """Per-graph formula units per cell [num_graphs] (>= 1)."""
    return data.z_fu.reshape(-1)


# ── cache verification (full sweep, vectorized per batch) ──────────────────────
def verify_magmom_in_cache(cache, labels, tag):
    """End-to-end proof that the CACHED batches — the tensors the node/residual
    losses will actually consume — carry the built labels, read through the same
    accessors the loss uses. EVERY graph in EVERY batch is checked. Raises on
    the first mismatch; prints the per-split census."""
    n_sup = n_graphs = n_nodes_sup = 0
    for b in cache:
        mids = [str(x[0] if isinstance(x, (list, tuple)) else x)
                for x in b.material_id]
        sup = get_node_sup(b).tolist()
        resid = get_resid_target(b).tolist()
        z = get_z_fu(b).tolist()
        node_m = get_node_magmom(b)
        counts = torch.bincount(b.batch, minlength=len(mids)).tolist()
        if len(mids) != len(sup):
            raise RuntimeError(f'MAGMOM VERIFY FAILED [{tag}]: {len(mids)} ids vs '
                               f'{len(sup)} node_sup entries in one cached batch.')
        expect = []
        off = 0
        for m, s, r, zz, c in zip(mids, sup, resid, z, counts):
            rec = labels.get(m)
            if rec is None:
                raise RuntimeError(f'MAGMOM VERIFY FAILED [{tag}]: {m} has no '
                                   f'label record.')
            if int(s) != int(rec.get('sup', 0)):
                raise RuntimeError(f'MAGMOM VERIFY FAILED [{tag}]: {m} carries '
                                   f'node_sup={int(s)}, build says '
                                   f'{int(rec.get("sup", 0))} ({rec.get("why")}).')
            if abs(zz - rec.get('z', 1.0)) > 1e-4:
                raise RuntimeError(f'MAGMOM VERIFY FAILED [{tag}]: {m} z_fu={zz}, '
                                   f'build says {rec.get("z", 1.0)}.')
            if rec.get('sup'):
                if abs(r - rec['resid']) > 1e-4:
                    raise RuntimeError(f'MAGMOM VERIFY FAILED [{tag}]: {m} '
                                       f'resid_target={r}, build says {rec["resid"]}.')
                if len(rec['m']) != c:
                    raise RuntimeError(f'MAGMOM VERIFY FAILED [{tag}]: {m} has '
                                       f'{len(rec["m"])} site labels but {c} '
                                       f'nodes in the cached batch.')
                expect.extend(rec['m'])
                n_sup += 1
                n_nodes_sup += c
            else:
                expect.extend([0.0] * c)
            off += c
        n_graphs += len(mids)
        exp_t = torch.tensor(expect, dtype=node_m.dtype)
        if node_m.numel() != exp_t.numel() or not torch.allclose(
                node_m.cpu(), exp_t, atol=1e-5):
            bad = int((node_m.cpu() - exp_t).abs().argmax()) if node_m.numel() == exp_t.numel() else -1
            raise RuntimeError(f'MAGMOM VERIFY FAILED [{tag}]: cached node_magmom '
                               f'differs from the built labels (first bad flat '
                               f'index {bad}).')
    print(f'    magmom verify [{tag}]: {n_sup}/{n_graphs} graphs supervised '
          f'({n_nodes_sup} nodes) — cached batches match the built labels.')


# ── the sign-gauge node loss (per-graph; unit-tested offline) ──────────────────
def node_gauge_loss(m_hat, m_true, batch_vec, num_graphs):
    """Per-graph sign-gauge-invariant node L1:
        L_g = min( mean_i |m_hat - m|,  mean_i |m_hat + m| )   (one global flip
    per graph, never per atom). Returns (L_g [num_graphs], flip [num_graphs]
    bool: True where the flipped branch won -- used by the gauge-resolved
    metrics so they report exactly what the loss optimized)."""
    e_plus = (m_hat - m_true).abs()
    e_minus = (m_hat + m_true).abs()
    zeros = torch.zeros(num_graphs, dtype=m_hat.dtype, device=m_hat.device)
    s_plus = zeros.clone().index_add_(0, batch_vec, e_plus)
    s_minus = zeros.clone().index_add_(0, batch_vec, e_minus)
    cnt = torch.zeros(num_graphs, dtype=m_hat.dtype, device=m_hat.device)
    cnt.index_add_(0, batch_vec, torch.ones_like(m_hat))
    cnt = cnt.clamp(min=1)
    flip = s_minus < s_plus
    return torch.minimum(s_plus, s_minus) / cnt, flip
