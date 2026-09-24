"""
train_sdchemgnn.py — trains SD-ChemGNN, the spin-decorated ChemGNN.
The per-site moment decomposition CONDITIONED on a proposed spin arrangement:

    x_i (95ch)  ++  [seed sign, formal magnitude, seed flag]   -> 98 channels
                    (spin_decoration.decorate_batch, AT RUNTIME — data.pt
                     and the site-label attach pipeline are untouched)
    m_hat_i = node_moment_head(z_i)            [N] signed uB/atom
              (Xu-conditioned comparator: node scalar * seed sign)
    r_hat   = interstitial_residual_head(h)    [G] uB PER F.U.
    y_hat   = sum_i m_hat_i / Z  +  r_hat      <- the SIZE-CONSISTENT
              reconstruction: every term is replication-invariant; it
              supersedes the earlier (sum m + r)/Z

What this trainer adds on top of the plain site-decomposition trainer:
  1. Seed build/attach/verify after the site-moment attach (build_seed_table,
     attach_seed_channels, verify_seed_vs_sidetable tripwire, cache sweep,
     the gauge-conflict corner masked).
  2. Runtime decoration in every train/eval pass; the corruption schedule +
     the global co-flip (train mode; co-flips the seed AND every signed
     target).
  3. Node loss gauge-PINNED on decorated graphs, min-over-s retained for
     undecorated graphs only (combined_node_loss).
  4. Residual loss target read PER F.U. (resid_target / z_fu) to match the
     per-f.u. residual head.
  5. Parity tolerance max(20x the measured noise floor, 1e-3); the 0.05 hard
     baseline check is unchanged.
  6. Startup canaries: two-site symmetry-breaking check + global-flip check
     (exact asserts for the Xu-conditioned comparator; mechanism assert +
     report for SD-ChemGNN), repeated as REPORTS on the trained model at the
     final evaluation.
  7. Shuffled-decoration leakage probe (label-decorated vs undecorated vs
     shuffled val MAE) at disp_step and at the end: want oracle <
     undecorated <= shuffled.
  8. Final artifacts in two modes: *_undec / *_oracle regression + sitedecomp
     dumps (5th sitedecomp column = per-graph self-consistency). The
     proposer-decorated mode is MagSieve's job (04_magsieve/), not this
     trainer's.
Everything else — losses, weights, heads, trunk, chained-session / resume /
deadline / verify mechanics — is the site-decomposition trainer unchanged, so
the decoration is the single attributed variable.


Goal: screen generator-proposed structures for SIGNIFICANT magnetism (find novel
magnets). So the model must be accurate in the mid-to-upper moment range that
decides "promising or not", while the exact value of extreme moments (which the
generator will essentially never produce) does not matter.

One MyCEALNetwork regresses the per-formula-unit moment directly, on the same
seeded split as the original ChemGNN formation-energy trainer.

The heavy tail of |M| (0 -> ~214 uB/f.u.) is handled by CAPPING the target: the
moment magnitude is clamped to +/- cfg.target_cap for both training and metrics
(cap_target), so a |M|=200 structure teaches ">= cap" rather than its exact value.
This removes the tail distortion at the source and saturates (does NOT drop) those
structures. On the bounded target, cfg.loss_type selects the objective:
'binned_l1' (the default) reweights the L1 by |M| bin to shift gradient from the
easy bulk (|M| <= bulk_cutoff, ~69% of samples) into the tail; plain 'l1'
remains the unweighted baseline. cfg.select_metric decides which val number drives
the plateau scheduler, early stopping and best-model saving ('loss' = the training
objective on val, so a reweighted run is judged by its own objective).

Reporting is independent of the loss: an always-on dashboard prints the training
loss and per-f.u. MAE on the SAME scale for train/val/test (so the gap is a clean
overfitting signal), plus MEDIAN absolute error and the error split into
BULK (|M| <= bulk_cutoff, "not interesting") vs TAIL (|M| > bulk_cutoff, the
"worth-pursuing" tail). bulk_cutoff is set to the significance threshold, so
the tail-MAE is a direct readout of accuracy on candidates worth acting on.
cfg.scope_max (off by default) is the separate, explicit scope cut (DROPS data).

Works with the existing code unchanged: MPDataset (build_crystal_graph_dataset),
the model from sdchemgnn_model.make_model_bs, and the training_utils save/plot
helpers. All knobs come from sdchemgnn_config.py.

Performance: with cfg.cache_batches (default) the dense-graph collation happens ONCE
into a RAM batch cache and is reused every epoch (per-epoch CPU work ~zero, so the run
is GPU-compute-bound); the batch order still reshuffles each epoch and batches are
cloned per step so the cache is never mutated. Falls back to streaming DataLoaders
(persistent workers + prefetch, non_blocking copies over pinned memory) when disabled.
Also: large forward-only eval batch, on-GPU metric accumulation (one sync/epoch), and
the test set evaluated only at disp_step (val alone drives scheduler + early stopping).
"""

# ── repository path bootstrap (added for the GitHub release) ──────────────
# The stage folders import each other's modules; make all of them importable
# whichever folder this file is run from.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ('02_data_curation', '03_sdchemgnn', '04_magsieve'):
    _p = _os.path.join(_REPO_ROOT, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────
import os
import os.path as osp
import signal
import sys
import time
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from sdchemgnn_config import config_magnetic as cfg
from build_crystal_graph_dataset import MPDataset
from sdchemgnn_model import make_model_bs, SEED_CHANNELS
from site_moment_labels import (
    build_magmom_labels,
    attach_magmom_labels,
    verify_magmom_in_cache,
    get_node_magmom,
    get_node_sup,
    get_resid_target,
    get_z_fu,
    node_gauge_loss,
)
from spin_decoration import (          # ── SD-ChemGNN ──
    build_seed_table,
    attach_seed_channels,
    verify_seed_vs_sidetable,
    verify_seed_in_cache,
    get_seed_sign,
    get_seed_fmag,
    get_seed_elig,
    decorate_batch,
    combined_node_loss,
    self_consistency,
)
from training_utils import (
    worker_init_fn,
    save_hyper_parameter,
    save_model_CEAL,
    save_mae,
    save_regression_result,
    plot_regression_result,
)


# ── A100 throughput: TF32 tensor-core matmuls (opt-in: TRAIN_TF32=1) ─────────────
# PyTorch >= 1.12 ships with fp32-matmul TF32 turned OFF; on A100 the Linear/GEMM
# work inside CEALConv (per-edge message MLPs, pre/post MLPs) then runs on the slow
# fp32 path and leaves a 3-6x kernel speedup on the table. TF32 keeps fp32 range
# and fp32 ACCUMULATION with a 10-bit input mantissa — noise orders of magnitude
# below this problem's label noise.
# GATED behind TRAIN_TF32=1 so that, by default, this file is numerically
# IDENTICAL to the pre-patch script: with the gate off, only observability was
# added ([profile] line, env identity, s/ep), so it is safe to drop into a chain
# that is ALREADY RUNNING for diagnosis. Because TF32 changes numerics slightly,
# only enable it together with a FRESH run dir — never mid-chain.
if os.environ.get('TRAIN_TF32') == '1':
    try:
        torch.set_float32_matmul_precision('high')   # torch >= 1.12 canonical switch
    except AttributeError:
        pass                                          # older torch: flags below suffice
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET      = cfg.target                       # total_magnetization_normalized_formula_units
_MAG_COLUMN = 1                                 # data.y[:, 1] = magnetization / f.u.
_UNITS      = 'uB/f.u.'

# ── Chained short sessions (busy-queue mode) ────────────────────────────────────
# Two env vars (set by the launcher) switch on chain mode; when both
# are absent the script behaves exactly as before (fresh auto-numbered folder, no
# time budget):
#   TRAIN_RUN_DIR          fixed run folder shared by every session of one chain
#                          (resume.pt / best_model.pt / artifacts accumulate here)
#   TRAIN_SESSION_SECONDS  this session's wall budget; the loop YIELDS — saves
#                          resume.pt and exits 0 — once the next epoch + margin
#                          no longer fits (cfg.session_margin_s)
# SIGUSR1 / SIGTERM (the scheduler's pre-termination signal, e.g. sbatch
# --signal, fires ~2 min before the wall; the scheduler
# sends TERM at the kill) also request a graceful yield, as the backstop for a
# wrong budget estimate. The handler only sets a flag; safe points check it.
RUN_DIR_ENV = os.environ.get('TRAIN_RUN_DIR') or None
SESSION_S   = float(os.environ.get('TRAIN_SESSION_SECONDS') or 0)   # 0 = unlimited
# Absolute wall deadline (unix seconds), stamped by the SHELL at job start. More
# accurate than SESSION_S: python can start 1-3 min after the job on a busy
# filesystem (torch imports, dataset load), and an elapsed-since-python budget
# extends past the real wall by that lag — observed in the first chained run, where the
# T-120s SIGUSR1 backstop beat the budget in 5/5 sessions. With the deadline the
# budget path is primary again; SESSION_S remains the fallback.
SESSION_END = float(os.environ.get('TRAIN_SESSION_DEADLINE') or 0)  # 0 = use SESSION_S

_YIELD = {'flag': False, 'why': ''}


def _request_yield(signum, frame):
    _YIELD['flag'] = True
    _YIELD['why'] = 'SIGUSR1' if signum == signal.SIGUSR1 else f'signal {signum}'


signal.signal(signal.SIGUSR1, _request_yield)
signal.signal(signal.SIGTERM, _request_yield)


def get_magnetization(data):
    """Per-f.u. moment as a 1-D tensor [num_graphs] (may contain NaN). Mirrors
    get_target() in the original ChemGNN trainer so both pipelines read the same values."""
    if hasattr(data, TARGET) and getattr(data, TARGET) is not None:
        t = getattr(data, TARGET)
    else:
        t = data.y[:, _MAG_COLUMN]
    return t.reshape(-1)


def get_ordering_mode(data):
    """Per-graph ordering-mode label as a 1-D int tensor [num_graphs]; -1 =
    unsupervised. THE read path for the mixture-head aux loss — the verify layers
    read through this function, per an earlier lesson (verify what the loss
    consumes, never a parallel attribute)."""
    return data.ordering_mode.reshape(-1)


def _fresh_item_view(ds):
    """── SD-ChemGNN: drop InMemoryDataset's lazy per-index item cache
    (_data_list) so the NEXT item accesses re-separate from the CURRENT master
    store. PyG memoizes materialized items on first access, so an item read by
    an EARLIER read-back layer predates the later attaches and is served STALE
    on re-access — job 20883498: the seed read-back hit an item cached by the
    manifest read-back and died on a missing seed_sign. (The training path is
    unaffected: the split views build their own caches AFTER every attach.)
    Clearing the cache is pure de-memoization — numerics identical. Call this
    immediately after each attach, before its item-path read-back."""
    if hasattr(ds, '_data_list'):
        ds._data_list = None
    return ds


def cap_target(t):
    """Saturate the moment MAGNITUDE at +/- cfg.target_cap (clamp, don't drop), so
    the model learns to predict up to the ceiling and report ">= cap" beyond it.
    Applied to the target for BOTH training and metrics; errors above the cap don't
    count. clamp() propagates NaN, so it composes with the NaN mask either way."""
    return t.clamp(-cfg.target_cap, cfg.target_cap)


# ── Robust training losses (selected by cfg.loss_type) ──────────────────────────
def make_loss(cfg):
    lt = cfg.loss_type.lower()
    if lt in ('huber', 'smooth_l1'):
        d = float(cfg.huber_delta)
        if lt == 'huber':                       # quad for |r|<=d, linear beyond
            def f(pred, tgt):
                ar = (pred - tgt).abs()
                quad = torch.clamp(ar, max=d)
                lin = ar - quad
                return (0.5 * quad * quad + d * lin).mean()
        else:                                   # PyTorch SmoothL1 (beta = d)
            def f(pred, tgt):
                ar = (pred - tgt).abs()
                return torch.where(ar < d, 0.5 * ar * ar / d, ar - 0.5 * d).mean()
        return f
    if lt == 'logcosh':                         # stable log(cosh(r))
        ln2 = math.log(2.0)
        def f(pred, tgt):
            ar = (pred - tgt).abs()
            return (ar + F.softplus(-2.0 * ar) - ln2).mean()
        return f
    if lt == 'mag_weighted_l1':                 # L1 down-weighted by |M|
        s = float(cfg.mag_weight_scale)
        def f(pred, tgt):
            w = 1.0 / (1.0 + tgt.abs() / s)
            return (w * (pred - tgt).abs()).sum() / w.sum().clamp(min=1e-8)
        return f
    if lt == 'binned_l1':                       # L1 UP-weighted by |M| bin (tail)
        # Per-sample weight looked up by which |M| bin the (capped) target falls in:
        #   |M| <= e0 -> weights[0], (e0, e1] -> weights[1], ..., > e_last -> weights[-1]
        # (torch.bucketize, right=False: boundaries[i-1] < v <= boundaries[i], so the
        # first bin is |M| <= e0 — matching the dashboard's bulk = |M| <= bulk_cutoff.)
        # Loss is the WEIGHTED MEAN sum(w*|r|)/sum(w): global weight scale cancels, so
        # the weights are pure RELATIVE emphasis and the loss stays on the ~MAE scale
        # regardless of batch composition. w depends only on the target, so the
        # gradient is just w_i/sum(w) * sign(r_i) — plain L1 with a per-sample rate.
        edges_l = [float(e) for e in cfg.binned_l1_edges]
        w_l = [float(x) for x in cfg.binned_l1_weights]
        if edges_l != sorted(edges_l) or len(set(edges_l)) != len(edges_l):
            raise ValueError(f'binned_l1_edges must be strictly increasing, got {cfg.binned_l1_edges}')
        if len(w_l) != len(edges_l) + 1:
            raise ValueError(f'binned_l1_weights needs len(binned_l1_edges)+1 = {len(edges_l) + 1} '
                             f'entries, got {len(w_l)}')
        if min(w_l) <= 0:
            raise ValueError(f'binned_l1_weights must all be > 0, got {cfg.binned_l1_weights}')
        edges_cpu, w_cpu = torch.tensor(edges_l), torch.tensor(w_l)
        cache = {}                              # one device-resident copy per device
        def f(pred, tgt):
            if tgt.device not in cache:
                cache[tgt.device] = (edges_cpu.to(tgt.device), w_cpu.to(tgt.device))
            edges, w_tab = cache[tgt.device]
            w = w_tab[torch.bucketize(tgt.abs(), edges)]
            return (w * (pred - tgt).abs()).sum() / w.sum().clamp(min=1e-8)
        return f
    if lt == 'l1':
        return lambda pred, tgt: (pred - tgt).abs().mean()
    if lt == 'mse':
        return lambda pred, tgt: ((pred - tgt) ** 2).mean()
    raise ValueError(f"unknown loss_type '{cfg.loss_type}'")


# ── Optimizer grouping (weight decay on weight matrices only) ───────────────────
def build_optimizer(model, lr, wd_gnn, wd_mlp):
    gnn_prefixes  = ('ceal_1.', 'cealConvs.')
    mlp_prefixes  = ('pre_mlp.', 'post_mlp.', 'mode_head.', 'mode_value_head.',
                     'node_moment_head.', 'interstitial_residual_head.',
                     'spin_mlp.')                       # ── Xu-conditioned comparator ──
    norm_prefixes = ('norm_1.', 'norms.')
    gnn_decay, mlp_decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith('.bias') or any(name.startswith(x) for x in norm_prefixes):
            no_decay.append(p)
        elif any(name.startswith(x) for x in gnn_prefixes):
            gnn_decay.append(p)
        elif any(name.startswith(x) for x in mlp_prefixes):
            mlp_decay.append(p)
        else:
            gnn_decay.append(p)
    return torch.optim.AdamW(
        [
            {'params': gnn_decay, 'weight_decay': wd_gnn},
            {'params': mlp_decay, 'weight_decay': wd_mlp},
            {'params': no_decay,  'weight_decay': 0.0},
        ],
        lr=lr,
    )


def make_loader(ds, batch, shuffle, persistent, g, num_workers):
    kwargs = dict(
        batch_size=batch, shuffle=shuffle, num_workers=num_workers,
        worker_init_fn=worker_init_fn, generator=g, pin_memory=True,
    )
    if num_workers > 0:                          # these args are invalid at num_workers=0
        kwargs['persistent_workers'] = persistent  # keep workers alive across epochs
        kwargs['prefetch_factor'] = 4              # stage 4 batches/worker ahead (hides collation)
    return DataLoader(ds, **kwargs)


def build_cache(ds, batch, num_workers, g, shuffle_compose=False, tag=''):
    """Collate `ds` into a fixed list of Batch objects ONCE, held in CPU RAM. Workers
    are used only for this one-time build. If shuffle_compose, the dataset is permuted
    first so the batch COMPOSITION differs from a previous build (cache_reshuffle_every).
    The dense-graph collation that dominated per-epoch time thus happens once, not 1500x.

    Progress is printed every 50 batches (and at completion) so a stalled build is
    DISTINGUISHABLE from a slow one straight from the job log — a line that stops
    advancing means the build is stuck (seen once on a node with polluted /dev/shm:
    with num_workers > 0 every batch transits POSIX shared memory and the parent
    RETAINS those segments, so a full/dirty /dev/shm blocks the workers silently).
    num_workers=0 collates in the main process: slower but fork- and shm-free — the
    bulletproof fallback (cfg.cache_workers = 0) if a build ever stalls again."""
    if shuffle_compose:
        order = torch.randperm(len(ds), generator=g)
        ds = [ds[i] for i in order.tolist()] if isinstance(ds, list) else ds[order]
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers,
                        worker_init_fn=worker_init_fn, generator=g, pin_memory=False)
    total = len(loader)
    out, tb = [], time.time()
    for i, data in enumerate(loader, 1):
        out.append(data)
        if i % 50 == 0 or i == total:
            rate = i / max(time.time() - tb, 1e-9)
            print(f'    cache[{tag or "?"}] {i}/{total} batches  ({rate:.1f} batch/s)',
                  flush=True)
    return out


class _CudaPrefetcher:
    """Overlap batch staging with GPU compute (opt-in via env TRAIN_PREFETCH=1).

    A worker thread pulls from the wrapped loader (for CachedLoader that pull IS
    the per-batch clone) and issues the H2D copy on a dedicated side stream, up to
    two batches ahead of the training loop; the consumer then only waits on a CUDA
    event instead of doing clone+copy itself, so staging hides behind the previous
    batch's compute. record_stream() marks every delivered tensor as in-use by the
    compute stream, so the caching allocator cannot recycle its memory for the
    next staged batch while kernels are still reading it. Iteration order, RNG
    consumption and the delivered VALUES are identical to the wrapped loader —
    this changes when copies happen, never what is trained on. Any worker error
    is re-raised in the training thread (fail loud, not silently degraded)."""

    def __init__(self, loader, device):
        self._loader = loader
        self._device = device

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        if not torch.cuda.is_available():
            yield from self._loader
            return
        import queue as _queue
        import threading
        stream = torch.cuda.Stream()
        q = _queue.Queue(maxsize=2)                  # <= 2 batches staged ahead

        def _worker():
            try:
                for b in self._loader:               # CachedLoader yields fresh clones
                    with torch.cuda.stream(stream):
                        b = b.to(self._device, non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(stream)
                    q.put((b, ev))
                q.put(None)
            except Exception as exc:                 # surface in the consumer thread
                q.put(exc)

        threading.Thread(target=_worker, daemon=True).start()
        compute = torch.cuda.current_stream()

        def _rec(t):
            t.record_stream(compute)                 # allocator: defer reuse past compute
            return t

        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise RuntimeError(f'prefetch worker failed: {item!r} — rerun '
                                   f'without TRAIN_PREFETCH=1 to fall back') from item
            b, ev = item
            compute.wait_event(ev)                   # order the H2D before compute
            b.apply(_rec)
            yield b


class CachedLoader:
    """Iterable over pre-collated batches held in CPU RAM, so graph collation happens
    ONCE instead of every epoch. Matches the DataLoader interface (`len()` and
    `for data in loader`), so train/eval code is unchanged.

    Each iteration yields a CLONE of the cached batch: the training loop then moves the
    clone to GPU with the usual in-place `.to(device)`, leaving the CPU cache intact
    (PyG's `.to()` mutates in place, so without the clone the cache would migrate to GPU
    and accumulate). Cloning is a cheap CPU copy of one batch at a time.

    For training (shuffle=True) the batch ORDER is reshuffled every epoch via the seeded
    generator; batch COMPOSITION is fixed at build time (see build_cache)."""
    def __init__(self, batches, shuffle, generator):
        self._batches = batches
        self._shuffle = shuffle
        self._g = generator

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        if self._shuffle:
            order = torch.randperm(len(self._batches), generator=self._g).tolist()
        else:
            order = range(len(self._batches))
        for i in order:
            yield self._batches[i].clone()       # clone protects the cache from in-place .to()


@torch.no_grad()
def collect_M(ds, g):
    """Forward-free pass: signed M per graph (CPU), dataset order. One-shot, so it
    uses the light eval worker pool, not the full train pool."""
    loader = make_loader(ds, cfg.batch_size * cfg.eval_batch_factor, False, False,
                         g, cfg.eval_num_workers)
    vals = [get_magnetization(data).reshape(-1) for data in loader]
    return torch.cat(vals) if vals else torch.empty(0)


def _fmt(s):
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    return f'{h}h {m}m {sec}s' if h else f'{m}m {sec}s'


# ── Resume state (chained sessions) ──────────────────────────────────────────────
def _resume_path(folder):
    return osp.join(folder, 'resume.pt')


def save_resume(folder, *, epoch, model, optimizer, scheduler, g, best, best_epoch,
                no_improve, hist, timestr, cum_seconds, fingerprint, dec_g=None):
    """Everything needed to continue the run as if it was never interrupted:
    weights, optimizer + plateau-scheduler state, early-stop counters, dashboard
    history, cumulative trained time, and ALL RNG streams — the global CPU/CUDA
    states (dropout) and the loader generator g (epoch shuffle order) — plus a
    config fingerprint so a mid-chain config edit fails loudly instead of silently
    mixing two experiments. Written tmp-then-os.replace (ATOMIC), so a kill in the
    middle of the write can never leave a corrupt/partial resume.pt behind."""
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best': best, 'best_epoch': best_epoch, 'no_improve': no_improve,
        'hist': hist, 'timestr': timestr, 'cum_seconds': cum_seconds,
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'loader_g': g.get_state(),
        'dec_g': dec_g.get_state() if dec_g is not None else None,  # ── SD-ChemGNN ──
        'fingerprint': fingerprint,
    }
    tmp = _resume_path(folder) + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, _resume_path(folder))


def load_resume(folder):
    p = _resume_path(folder)
    if not osp.isfile(p):
        return None
    try:
        return torch.load(p, map_location='cpu', weights_only=False)
    except TypeError:                        # older torch without the weights_only arg
        return torch.load(p, map_location='cpu')


def check_fingerprint(saved, current):
    """A chain is ONE experiment: editing the config between its sessions silently
    breaks attribution, so any drift vs. the saved hyper snapshot is a HARD error.
    Deliberate override for the rare legitimate case: TRAIN_ALLOW_CONFIG_CHANGE=1.
    (Operational knobs like resume_every / session_margin_s are intentionally NOT
    in the fingerprint — tune them mid-chain freely.)"""
    if saved is None:
        return
    diff = sorted({k for k in current if repr(saved.get(k)) != repr(current[k])}
                  | {k for k in saved if k not in current})
    if diff and not os.environ.get('TRAIN_ALLOW_CONFIG_CHANGE'):
        raise RuntimeError(
            f'resume.pt was written by a DIFFERENT config — changed keys: {diff}. '
            f'Finish or abandon the chain, or set TRAIN_ALLOW_CONFIG_CHANGE=1 to '
            f'override knowingly.')


# ── Training / evaluation ───────────────────────────────────────────────────────
_PROFILE_BATCHES = 20        # batches phase-timed on each session's FIRST epoch


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_epoch(model, optimizer, loss_fn, loader, mask_nan, profile=False, aux=None,
                bopt=None, dec=None):
    """One epoch. With profile=True the first _PROFILE_BATCHES steps are split into
    fetch(=cache clone) / H2D copy / forward+loss / backward+step wall times (with
    cudaDeviceSynchronize fences, so each phase is charged its real GPU cost) and a
    one-line summary is printed. That line is THE diagnostic for a slow epoch:
    fetch+H2D dominating -> enable TRAIN_PREFETCH=1 (overlap); fwd/bwd dominating ->
    the conv itself is the wall (TF32 / AMP / architecture, not data movement)."""
    model.train()
    total = torch.zeros((), device=device)      # training-objective loss (e.g. Huber)
    total_ae = torch.zeros((), device=device)   # absolute error (for running train MAE)
    n = torch.zeros((), device=device)
    # auxiliary-head accumulators (labeled rows only). `total` above stays the
    # REGRESSION objective alone, so the printed train loss remains directly
    # comparable to pre-head runs.
    aux_ce = torch.zeros((), device=device)
    aux_l1 = torch.zeros((), device=device)
    n_lab = torch.zeros((), device=device)
    # OPTION-B accumulators (node / residual / shrink terms, printed at disp_step)
    assert aux is not None and bopt is not None, 'site-decomposition trainer requires both channels'
    assert dec is not None, 'SD-ChemGNN trainer requires the decoration channel'
    b_node = torch.zeros((), device=device)
    b_resid = torch.zeros((), device=device)
    b_shrink = torch.zeros((), device=device)
    nb_sup = torch.zeros((), device=device)
    nb_all = torch.zeros((), device=device)
    # ── SD-ChemGNN: decoration census over the epoch ──
    b_dec = torch.zeros((), device=device)
    b_flip = torch.zeros((), device=device)
    prof = {'fetch': 0.0, 'h2d': 0.0, 'fwd': 0.0, 'bwd': 0.0, 'k': 0} if profile else None
    it = iter(loader)
    i = -1
    while True:
        i += 1
        timed = prof is not None and i < _PROFILE_BATCHES
        if timed:
            _cuda_sync()
            t0 = time.time()
        try:
            data = next(it)                     # CachedLoader: clone of the cached batch
        except StopIteration:
            break
        # Backstop yield: a stop signal arrived MID-epoch. Exit now — the last
        # epoch-boundary resume.pt is the restart point, so this partial epoch is
        # simply redone next session (<= 1 epoch of work lost, state consistent).
        if _YIELD['flag'] and i % 16 == 0:
            print(f'\nSESSION YIELD mid-epoch ({_YIELD["why"]}); next session '
                  f'resumes from the last completed epoch.', flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)          # skip finalization (see loop-bottom yield note)
        if timed:
            t1 = time.time()
            prof['fetch'] += t1 - t0
        data = data.to(device, non_blocking=True)
        # ── SD-ChemGNN: runtime decoration (corruption + co-flip;
        # widens x by the three seed channels and, on flipped graphs, co-flips
        # node_magmom / resid_target / the per-f.u. total the losses read) ──
        data = decorate_batch(data, 'train', dec['dcfg'], dec['gen'])
        if timed:
            _cuda_sync()
            t2 = time.time()
            prof['h2d'] += t2 - t1
        M = get_magnetization(data)
        # ── OPTION-B step: one trunk pass, every head, the reconstruction ──
        out = model.forward_b(data)
        pred = out['y'].reshape(-1)
        km = get_ordering_mode(data)
        vg = ~torch.isnan(M) if mask_nan else torch.ones_like(M, dtype=torch.bool)
        if not bool(vg.any()):
            continue
        Mv = cap_target(M[vg])                  # train on the capped moment
        optimizer.zero_grad()
        reg_loss = loss_fn(pred[vg], Mv)                        # THE objective (reconstruction)
        aux_reg = loss_fn(out['y_aux'].reshape(-1)[vg], Mv)     # post_mlp deep supervision
        # node loss — SD-ChemGNN dispatch: gauge-PINNED to the (possibly
        # co-flipped) seed/label convention on decorated graphs, min-over-s
        # retained for undecorated graphs only; per-graph node-mean, mean over
        # SUPERVISED graphs only (masked rows contribute nothing, no dilution)
        n_graphs = M.numel()
        Lg, _flip = combined_node_loss(out['m_hat'], get_node_magmom(data),
                                       data.batch, n_graphs, data.seed_dec,
                                       node_gauge_loss)
        sup = get_node_sup(data).bool() & vg
        r_hat = out['r_hat'].reshape(-1)
        if bool(sup.any()):
            l_node = Lg[sup].mean()
            # ── SD-ChemGNN: residual head + target both PER F.U. (size-consistent) ──
            l_resid = _huber_elem(
                r_hat[sup],
                (get_resid_target(data) / get_z_fu(data))[sup],
                bopt['huber_d'])
        else:
            l_node = torch.zeros((), device=pred.device)
            l_resid = torch.zeros((), device=pred.device)
            bopt['empty_batches'] += 1
        shrink = r_hat[vg].abs().mean()                          # anti-bypass
        loss = (reg_loss + bopt['w_aux'] * aux_reg + bopt['w_node'] * l_node
                + bopt['w_resid'] * l_resid + bopt['w_shrink'] * shrink)
        # auxiliary mode diagnostics (unchanged weights; out of score)
        kmv = km[vg]
        lab = kmv >= 0
        if bool(lab.any()):
            kl = kmv[lab].long()
            ce = F.cross_entropy(out['mode_logits'][vg][lab], kl, weight=aux['ce_w'])
            l1 = (out['mode_vals'][vg][lab].gather(1, kl.view(-1, 1)).view(-1)
                  - Mv[lab]).abs().mean()
            loss = loss + aux['w_mode'] * ce + aux['w_val'] * l1
            kn = kl.numel()
            aux_ce += ce.detach() * kn
            aux_l1 += l1.detach() * kn
            n_lab += kn
        else:
            aux['empty_batches'] += 1
        ns = sup.sum()
        b_node += l_node.detach() * ns
        b_resid += l_resid.detach() * ns
        nb_sup += ns
        b_shrink += shrink.detach() * vg.sum()
        nb_all += vg.sum()
        b_dec += (data.seed_dec & vg).sum()      # ── SD-ChemGNN census ──
        b_flip += (data.seed_flip & vg).sum()
        pred, M = pred[vg], Mv                  # for the shared accumulators below
        if timed:
            _cuda_sync()
            t3 = time.time()
            prof['fwd'] += t3 - t2
        loss.backward()
        optimizer.step()
        if timed:
            _cuda_sync()
            prof['bwd'] += time.time() - t3
            prof['k'] += 1
        k = M.numel()
        total += reg_loss.detach() * k
        total_ae += (pred.detach() - M).abs().sum()
        n += k
    if prof and prof['k']:
        b = prof['k']
        per = (prof['fetch'] + prof['h2d'] + prof['fwd'] + prof['bwd']) / b
        print(f"    [profile] first {b} train batches: fetch(clone)={prof['fetch'] / b * 1e3:.0f}ms  "
              f"H2D={prof['h2d'] / b * 1e3:.0f}ms  fwd={prof['fwd'] / b * 1e3:.0f}ms  "
              f"bwd+opt={prof['bwd'] / b * 1e3:.0f}ms  ->  {per * 1e3:.0f} ms/batch "
              f"(~{per * len(loader) / 60:.1f} min/epoch train-side)", flush=True)
    n = n.clamp(min=1)
    aux_stats = None
    if aux is not None:
        nl = n_lab.clamp(min=1)
        aux_stats = {'ce': (aux_ce / nl).item(), 'l1': (aux_l1 / nl).item(),
                     'frac': (n_lab / n).item(), 'empty': aux['empty_batches']}
    bstats = {'node': (b_node / nb_sup.clamp(min=1)).item(),
              'resid': (b_resid / nb_sup.clamp(min=1)).item(),
              'shrink': (b_shrink / nb_all.clamp(min=1)).item(),
              'sup_frac': (nb_sup / n).item(),
              'dec_frac': (b_dec / nb_all.clamp(min=1)).item(),
              'flip_frac': (b_flip / nb_all.clamp(min=1)).item()}
    return (total / n).item(), (total_ae / n).item(), aux_stats, bstats


def money_zone_bands(e, absM):
    """MAE + count within each tail sub-band. The tail (|M| > bulk_cutoff,
    capped at target_cap) is split at cfg.money_zone_edges into bands
    [bulk_cutoff, e0), [e0, e1), ..., [e_last, target_cap]. The first lower bound is
    EXCLUSIVE so the bands partition exactly the tail (|M| > bulk_cutoff); the last
    band includes the cap. Returns [(lo, hi, mae, count, rel), ...] where rel is the
    band's relative error = mae / mean(|M| in band) (aggregate form, outlier-robust;
    everything in capped space). Empty list if no edges."""
    interior = sorted(x for x in cfg.money_zone_edges
                      if cfg.bulk_cutoff < x < cfg.target_cap)
    edges = [cfg.bulk_cutoff, *interior, cfg.target_cap]
    bands = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lo_ok = absM > lo if i == 0 else absM >= lo
        hi_ok = absM <= hi if i == len(edges) - 2 else absM < hi
        m = lo_ok & hi_ok
        cnt = int(m.sum())
        if cnt > 0:
            mae = e[m].mean().item()
            mean_abs = absM[m].mean().item()
            rel = mae / mean_abs if mean_abs > 0 else float('nan')
        else:
            mae, rel = float('nan'), float('nan')
        bands.append((lo, hi, mae, cnt, rel))
    return bands


def _dec_eval(data, dec):
    """── SD-ChemGNN: eval-mode decoration (no corruption, no flip; 'shuffled'
    draws from the caller-supplied fixed-seed generator)."""
    return decorate_batch(data, dec['mode'], dec['dcfg'], dec.get('gen'))


@torch.no_grad()
def evaluate(model, loader, loss_fn, dec):
    """Dashboard over a loader: the training-objective loss (cfg.loss_type) on the
    SAME scale as the train loss, per-f.u. MAE, median AE, bulk vs tail MAE, and a
    tail sub-band breakdown (money_zone_bands) showing where in the tail the
    error concentrates. Target is capped (cap_target) so all are in capped space.
    SD-ChemGNN: every batch is decorated in dec['mode'] first."""
    model.eval()
    errs, absMs = [], []
    total_loss = torch.zeros((), device=device)
    n = torch.zeros((), device=device)
    for data in loader:
        data = data.to(device, non_blocking=True)
        data = _dec_eval(data, dec)
        M = get_magnetization(data)
        v = ~torch.isnan(M)
        pred = model.forward_b(data)['y'].reshape(-1)      # the site-decomposition model: reconstruction
        pm, Mm = pred[v], cap_target(M[v])       # capped target; errors above cap don't count
        if Mm.numel() == 0:
            continue
        total_loss += loss_fn(pm, Mm).detach() * Mm.numel()
        n += Mm.numel()
        errs.append((pm - Mm).abs().cpu())
        absMs.append(Mm.abs().cpu())             # split by capped magnitude
    e = torch.cat(errs); absM = torch.cat(absMs)
    bulk = absM <= cfg.bulk_cutoff
    tail = ~bulk
    return {
        'loss':  (total_loss / n.clamp(min=1)).item(),   # same scale as train loss
        'mae':   e.mean().item(),
        'medae': e.quantile(0.5).item(),
        'bulk_mae': e[bulk].mean().item() if bool(bulk.any()) else float('nan'),
        'tail_mae': e[tail].mean().item() if bool(tail.any()) else float('nan'),
        'n_bulk': int(bulk.sum()),
        'n_tail': int(tail.sum()),
        'bands': money_zone_bands(e, absM),
    }


@torch.no_grad()
def collect_pred_true(model, loader, dec):
    model.eval()
    P, T = [], []
    for data in loader:
        data = data.to(device, non_blocking=True)
        data = _dec_eval(data, dec)                        # ── SD-ChemGNN ──
        M = get_magnetization(data)
        v = ~torch.isnan(M)
        pred = model.forward_b(data)['y'].reshape(-1)      # the site-decomposition model: reconstruction
        P.append(pred[v].cpu()); T.append(cap_target(M[v]).cpu())   # capped truth
    return torch.cat(P), torch.cat(T)


def fmt_breakdown(tag, d):
    """The MAE-dashboard detail for val/test: median + bulk/tail with counts."""
    return (f"{tag}[med={d['medae']:.3f} bulk={d['bulk_mae']:.3f}/{d['n_bulk']} "
            f"tail={d['tail_mae']:.3f}/{d['n_tail']}]")


def fmt_bands(d):
    """Money-zone sub-bands as 'MAE/count (rel%)' per |M| band (blank if no bands).
    rel% = band MAE as a percentage of the band's mean |M| (capped space)."""
    parts = ' '.join(
        f'[{lo:g}-{hi:g}]={mae:.3f}/{n} ({rel * 100:.0f}%)'
        for lo, hi, mae, n, rel in d['bands'])
    return parts if parts else '(no tail bands)'


def save_curve(folder, prefix, ep, tr_loss, va_loss, va_mae, loss_name):
    try:
        arr = np.array([ep, tr_loss, va_loss, va_mae], dtype=float).T
        np.savetxt(osp.join(folder, f'{prefix}_progress.txt'), arr,
                   header=f'epoch train_{loss_name} val_{loss_name} val_MAE', comments='')
    except Exception as exc:
        print(f'  [warn] could not save progress txt ({exc})')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure()
        # train vs val on the SAME loss scale -> the gap is the overfitting signal.
        plt.plot(ep, tr_loss, label=f'train {loss_name}')
        plt.plot(ep, va_loss, label=f'val {loss_name}')
        plt.xlabel('epoch'); plt.ylabel(f'{loss_name} loss'); plt.title(prefix)
        plt.legend(); plt.tight_layout()
        plt.savefig(osp.join(folder, f'{prefix}_progress.png'), dpi=120)
        plt.close()
    except Exception as exc:
        print(f'  [warn] could not plot progress ({exc})')


def load_manifest(path):
    """label_manifest.csv (from curate_magnetization_labels.py) -> {material_id: (action, new)}
    plus a short content hash recorded in the run's hyperparameters, so WHICH
    curation produced a run is part of its fingerprint."""
    import csv
    import hashlib
    man = {}
    with open(path, newline='') as fh:
        for i, row in enumerate(csv.DictReader(fh), 2):
            mid = (row.get('material_id') or '').strip()
            act, new = row.get('action'), row.get('new')
            if not mid or act is None or new is None:
                raise RuntimeError(
                    f'label_manifest.csv line {i} is MALFORMED: '
                    f'material_id={mid!r}, action={act!r}, new={new!r} — a row '
                    f'must carry every header column '
                    f'(material_id,formula,split,old,new,action,reason). '
                    f'A hand-appended line with too few fields does this; fix '
                    f'the line before training.')
            try:
                newf = float(new)
            except ValueError:
                raise RuntimeError(
                    f"label_manifest.csv line {i} ({mid}): new={new!r} is not "
                    f"a number — the 'new' column landed in the wrong field?")
            man[mid] = (act, newf)
    digest = hashlib.md5(open(path, 'rb').read()).hexdigest()[:10]
    return man, digest


def relabel_master(dataset, man):
    """Write manifest relabels VECTORIZED into the collated master y tensor.

    Lesson of an earlier run: per-item writes (d.y.view(-1)[i] = v) can
    silently land on a per-access materialized temporary under PyG's
    copy-on-access item semantics — 12,407 'relabeled' counters incremented
    while the model trained on the ORIGINAL labels (train-bin populations
    matched the earlier run to the unit). The master store is what every item view and
    every collate reads through, so writing it is effective by construction —
    and the caller VERIFIES both through the item path and through the cached
    batches before a single epoch runs.

    Returns (ids_in_original_order, n_relabeled)."""
    store = getattr(dataset, '_data', None)
    if store is None:
        store = dataset.data                       # older PyG attribute name
    raw_ids = store.material_id
    ids = [str(x[0] if isinstance(x, (list, tuple)) else x) for x in raw_ids]
    idx, vals = [], []
    for j, m in enumerate(ids):
        r = man.get(m)
        if r is not None and r[0] != 'drop':
            idx.append(j)
            vals.append(r[1])
    if idx:
        y2 = store.y.view(len(ids), -1)            # (N,), (N,1), (N,2) all fine
        col = _MAG_COLUMN if y2.size(1) > _MAG_COLUMN else 0
        y2[torch.tensor(idx), col] = torch.tensor(vals, dtype=y2.dtype)
        # A LESSON FROM AN EARLIER RUN: get_magnetization() PREFERS the NAMED
        # per-graph attribute (total_magnetization_normalized_formula_units) over
        # data.y — so rewriting y alone changes what the verifiers check but NOT
        # what the model trains and evaluates on. That run trained on the
        # ORIGINAL labels while both verification layers passed on y. Write the
        # named master tensor too: it is the tensor the read path consumes.
        named = getattr(store, TARGET, None)
        if named is not None:
            n2 = named.view(len(ids), -1)
            n2[torch.tensor(idx), 0] = torch.tensor(vals, dtype=n2.dtype)
    return ids, len(idx)


def verify_manifest_in_cache(cache, man, tag):
    """End-to-end proof that the cached batches — the tensors the model will
    actually train/evaluate on — carry the manifest labels: every relabeled id
    must hold its manifest value, and no dropped id may be present. Raises
    (before any epoch) on the first violation; a silent label failure is no
    longer possible."""
    relab = {m: v for m, (a, v) in man.items() if a != 'drop'}
    drops = {m for m, (a, _) in man.items() if a == 'drop'}
    checked = 0
    for batch in cache:
        mids = batch.material_id
        # Verify through the READ PATH: get_magnetization() is what the training
        # and eval loops consume — if THIS returns the manifest value, the model
        # provably trains on it. (Checking batch.y alone is how the named-
        # attribute bug slipped through an earlier run.) The y column is checked too.
        y_read = get_magnetization(batch).view(-1)
        ys = batch.y.view(len(mids), -1)
        col = _MAG_COLUMN if ys.size(1) > _MAG_COLUMN else 0
        for m, yv, yv_col in zip(mids, y_read.tolist(), ys[:, col].tolist()):
            m = str(m[0] if isinstance(m, (list, tuple)) else m)
            if m in drops:
                raise RuntimeError(f'MANIFEST VERIFY FAILED [{tag}]: dropped id '
                                   f'{m} is present in the cached batches.')
            v = relab.get(m)
            if v is not None:
                if abs(yv - v) > 1e-3:
                    raise RuntimeError(
                        f'MANIFEST VERIFY FAILED [{tag}]: the READ PATH '
                        f'(get_magnetization) delivers y={yv:.4f} for {m} but the '
                        f'manifest says {v:.4f}. The model would train on wrong '
                        f'labels — aborting before any epoch.')
                if abs(yv_col - v) > 1e-3:
                    raise RuntimeError(
                        f'MANIFEST VERIFY FAILED [{tag}]: data.y carries '
                        f'{yv_col:.4f} for {m} but the manifest says {v:.4f}.')
                checked += 1
    if checked == 0:
        raise RuntimeError(f'MANIFEST VERIFY [{tag}]: no relabeled entries seen '
                           f'— cannot confirm the manifest was applied.')
    print(f'    manifest verify [{tag}]: {checked} relabeled entries confirmed '
          f'in cached batches; no dropped ids present.')


def attach_ordering_mode_labels(dataset, modes_by_id):
    """Write the per-graph int8 `ordering_mode` into the COLLATED MASTER store
    (+ the slices dict), keyed by the material_id attribute every cached graph
    already carries — the same master-store discipline as relabel_master(), with
    the new-attribute twist: PyG item slicing serves a key only if it exists in
    BOTH _data and slices (one value per graph). Every split view and cached
    batch then carries it automatically. Returns (ids_in_master_order, n_labeled)."""
    store = getattr(dataset, '_data', None)
    if store is None:
        store = dataset.data                       # older PyG attribute name
    raw_ids = store.material_id
    ids = [str(x[0] if isinstance(x, (list, tuple)) else x) for x in raw_ids]
    t = torch.full((len(ids),), -1, dtype=torch.int8)
    n_lab = 0
    for j, m in enumerate(ids):
        v = modes_by_id.get(m, -1)
        if v >= 0:
            t[j] = v
            n_lab += 1
    store.ordering_mode = t
    sl = getattr(dataset, 'slices', None)
    if sl is None:
        sl = getattr(dataset, '_slices', None)
    if sl is None:
        raise RuntimeError('cannot locate the slices dict on the dataset (PyG API '
                           'drift?) — ordering_mode item slicing would silently break.')
    sl['ordering_mode'] = torch.arange(len(ids) + 1)
    return ids, n_lab


def verify_modes_in_cache(cache, modes_by_id, tag):
    """End-to-end proof that the CACHED batches — the tensors the aux loss will
    actually consume — carry the derived mode labels, read through
    get_ordering_mode. EVERY graph is checked (dict lookups; ~ms), not a sample.
    Prints the per-split histogram; raises on any mismatch."""
    hist = {-1: 0, 0: 0, 1: 0, 2: 0}
    for b in cache:
        km = get_ordering_mode(b)
        raw_ids = b.material_id
        mids = [str(x[0] if isinstance(x, (list, tuple)) else x) for x in raw_ids]
        if len(mids) != km.numel():
            raise RuntimeError(f'MODE VERIFY FAILED [{tag}]: {len(mids)} ids vs '
                               f'{km.numel()} mode entries in one cached batch.')
        for m, v in zip(mids, km.tolist()):
            if int(v) != int(modes_by_id.get(m, -1)):
                raise RuntimeError(f'MODE VERIFY FAILED [{tag}]: {m} carries mode '
                                   f'{int(v)}, side-table derivation says '
                                   f'{int(modes_by_id.get(m, -1))}.')
            hist[int(v)] += 1
    print(f'    mode verify [{tag}]: '
          + ' '.join(f'{k}:{hist[k]}' for k in (-1, 0, 1, 2))
          + '  — cached batches match the side-table.')


@torch.no_grad()
def evaluate_modes(model, loader, dec):
    """Mode-head dashboard over a loader, LABELED rows only: accuracy, balanced
    accuracy (mean per-class recall), per-class recall, 3x3 confusion. Purely
    diagnostic — never drives selection."""
    model.eval()
    C = torch.zeros(9, dtype=torch.long)
    for data in loader:
        data = data.to(device, non_blocking=True)
        data = _dec_eval(data, dec)                        # ── SD-ChemGNN ──
        _, logits, _ = model(data, return_aux=True)
        km = get_ordering_mode(data)
        lab = km >= 0
        if not bool(lab.any()):
            continue
        tk = km[lab].long()
        pk = logits[lab].argmax(dim=1)
        C += torch.bincount((tk * 3 + pk).cpu(), minlength=9)
    C = C.reshape(3, 3)
    n = int(C.sum())
    rec = [float(C[k, k]) / max(int(C[k].sum()), 1) for k in range(3)]
    return {'n': n, 'acc': float(C.diag().sum()) / max(n, 1),
            'bal': sum(rec) / 3.0, 'recall': rec, 'confusion': C}


@torch.no_grad()
def collect_modes(model, loader, dec):
    """Mode-head outputs aligned ROW-FOR-ROW with collect_pred_true (same loader
    order, same NaN mask), for the *_modes.txt artifact."""
    model.eval()
    P, MU, K = [], [], []
    for data in loader:
        data = data.to(device, non_blocking=True)
        data = _dec_eval(data, dec)                        # ── SD-ChemGNN ──
        M = get_magnetization(data)
        v = ~torch.isnan(M)
        _, logits, mus = model(data, return_aux=True)
        P.append(F.softmax(logits[v], dim=1).cpu())
        MU.append(mus[v].cpu())
        K.append(get_ordering_mode(data)[v].cpu())
    return torch.cat(P), torch.cat(MU), torch.cat(K)


def _huber_elem(pred, tgt, d):
    """Elementwise-Huber mean (knee d) for the residual loss — robust to the
    measured heavy tail (p99 ~ 7.9 uB/cell) without the quadratic blow-up."""
    ar = (pred - tgt).abs()
    quad = torch.clamp(ar, max=d)
    return (0.5 * quad * quad + d * (ar - quad)).mean()


@torch.no_grad()
def evaluate_nodes(model, loader, dec):
    """OPTION-B(-S) node dashboard over a loader: gauge-resolved per-site MAE
    (overall + banded by |m|: [0,0.5) / [0.5,2) / [2+) uB), r_hat honesty stats
    — ALL PER F.U. in SD-ChemGNN (mean|r_hat|, corr vs resid_target/Z, share of the
    predicted per-f.u. total on tail rows), the recon-vs-aux MAE pair, and the
    SELF-CONSISTENCY readout (fraction of committal seeded sites whose
    predicted sign reproduces the seed; per-graph, then averaged / thresholded).
    The gauge resolution matches the loss exactly: PINNED on decorated graphs,
    min-over-s on undecorated ones (combined_node_loss)."""
    model.eval()
    band_edges = None
    se_sum = torch.zeros(3)
    se_cnt = torch.zeros(3)
    tot_err = 0.0
    tot_sites = 0
    r_abs_sum = 0.0
    r_n = 0
    share_sum = 0.0
    share_n = 0
    mae_recon = 0.0
    mae_aux = 0.0
    mae_n = 0
    sc_sum = 0.0
    sc_n = 0
    sc_ok = 0
    n_dec = 0
    r_list, rt_list = [], []
    for data in loader:
        data = data.to(device, non_blocking=True)
        data = _dec_eval(data, dec)                        # ── SD-ChemGNN ──
        M = get_magnetization(data)
        v = ~torch.isnan(M)
        out = model.forward_b(data)
        Mc = cap_target(M)
        pred = out['y'].reshape(-1)
        paux = out['y_aux'].reshape(-1)
        mae_recon += (pred[v] - Mc[v]).abs().sum().item()
        mae_aux += (paux[v] - Mc[v]).abs().sum().item()
        mae_n += int(v.sum())
        n_graphs = M.numel()
        m_true = get_node_magmom(data)
        m_hat = out['m_hat']
        r_hat = out['r_hat'].reshape(-1)                   # uB PER F.U. (SD-ChemGNN)
        r_abs_sum += r_hat[v].abs().sum().item()
        r_n += int(v.sum())
        Lg, flip = combined_node_loss(m_hat, m_true, data.batch, n_graphs,
                                      data.seed_dec, node_gauge_loss)
        sup = get_node_sup(data).bool() & v
        if bool(sup.any()):
            node_sel = sup[data.batch]
            mh = torch.where(flip[data.batch], -m_hat, m_hat)[node_sel]
            mt = m_true[node_sel]
            err = (mh - mt).abs()
            if band_edges is None or band_edges.device != mt.device:
                band_edges = torch.tensor([0.5, 2.0], device=mt.device)
            band = torch.bucketize(mt.abs(), band_edges)
            for k in range(3):
                bm = band == k
                if bool(bm.any()):
                    se_sum[k] += err[bm].sum().cpu()
                    se_cnt[k] += int(bm.sum())
            tot_err += err.sum().item()
            tot_sites += int(err.numel())
            r_list.append(r_hat[sup].cpu())
            rt_list.append((get_resid_target(data)
                            / get_z_fu(data))[sup].cpu())  # per f.u. (SD-ChemGNN)
        tail = (Mc.abs() > cfg.bulk_cutoff) & v
        if bool(tail.any()):
            m_fu_abs = (out['m_sum'].reshape(-1) / get_z_fu(data)).abs()
            denom = m_fu_abs + r_hat.abs() + 1e-6
            share_sum += (r_hat[tail].abs() / denom[tail]).sum().item()
            share_n += int(tail.sum())
        # ── SD-ChemGNN: self-consistency on decorated, valid graphs ──
        n_dec += int((data.seed_dec & v).sum())
        frac, _n_comm = self_consistency(m_hat, data,
                                         tiny=float(getattr(cfg, 'sc_tiny',
                                                            0.05)))
        fv = frac[v & data.seed_dec]
        fv = fv[~torch.isnan(fv)]
        if fv.numel():
            sc_sum += float(fv.sum())
            sc_n += int(fv.numel())
            sc_ok += int((fv >= 0.999).sum())
    r_corr = float('nan')
    if r_list:
        ra = torch.cat(r_list)
        rb = torch.cat(rt_list)
        if ra.numel() > 2 and float(ra.std()) > 1e-8 and float(rb.std()) > 1e-8:
            r_corr = float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])
    bm = [float(se_sum[k] / se_cnt[k].clamp(min=1)) for k in range(3)]
    return {'site_mae': tot_err / max(tot_sites, 1), 'n_sites': tot_sites,
            'band0': bm[0], 'band1': bm[1], 'band2': bm[2],
            'n_band': [int(x) for x in se_cnt.tolist()],
            'r_abs': r_abs_sum / max(r_n, 1), 'r_corr': r_corr,
            'r_share': share_sum / max(share_n, 1),
            'mae_recon': mae_recon / max(mae_n, 1),
            'mae_aux': mae_aux / max(mae_n, 1),
            'sc_mean': sc_sum / max(sc_n, 1), 'sc_rate': sc_ok / max(sc_n, 1),
            'sc_n': sc_n, 'n_dec': n_dec}


@torch.no_grad()
def collect_sitedecomp(model, loader, dec):
    """Per-graph [sum|m_hat|, sum m_hat, r_hat_fu, z_fu, sc_frac], aligned
    ROW-FOR-ROW with collect_pred_true (same loader order, same NaN mask) —
    the mechanistic audit artifact: a row that reads "large sum|m|, small
    sum m" is the model representing big-moments-cancelling; the alignment
    ratio |sum m|/sum|m| is col2/col1. SD-ChemGNN: r_hat is PER F.U.; the 5th
    column is the per-graph self-consistency fraction (NaN where the graph
    is undecorated or has no committal seeded sites)."""
    model.eval()
    rows = []
    for data in loader:
        data = data.to(device, non_blocking=True)
        data = _dec_eval(data, dec)                        # ── SD-ChemGNN ──
        M = get_magnetization(data)
        v = ~torch.isnan(M)
        out = model.forward_b(data)
        r_hat = out['r_hat'].reshape(-1)                   # uB PER F.U. (SD-ChemGNN)
        n_graphs = M.numel()
        abs_sum = torch.zeros(n_graphs, dtype=out['m_hat'].dtype,
                              device=r_hat.device)
        abs_sum = abs_sum.index_add(0, data.batch, out['m_hat'].abs())
        m_sum = out['m_sum'].reshape(-1)
        z = get_z_fu(data)
        sc, _ = self_consistency(out['m_hat'], data,
                                 tiny=float(getattr(cfg, 'sc_tiny', 0.05)))
        rows.append(torch.stack([abs_sum[v], m_sum[v], r_hat[v], z[v],
                                 sc[v]], dim=1).cpu())
    return torch.cat(rows)


# ── SD-ChemGNN: model-side canaries ─────────────────────────────────────────────
def _toy_two_site_batch(sample_batches, base_channels, sign_pair):
    """Build a synthetic 2-node graph by CLONING a real magnetic-species node
    (formal magnitude > 0) from the cached batches: two identical atoms, one
    symmetric edge pair (both directed edges share one real edge_attr), seed
    signs per sign_pair. The two nodes are EXACTLY equivalent — same features,
    automorphic graph — so any between-node output difference is attributable
    to the seed decoration alone. No file dependency (atom_init.json etc.)."""
    import types as _types
    src = None
    for b in sample_batches:
        fm = get_seed_fmag(b)
        hit = (fm > 0).nonzero(as_tuple=True)[0]
        if hit.numel():
            i = int(hit[0])
            src = (b.x[i].clone(), b.edge_attr[0].clone(), float(fm[i]))
            break
    if src is None:
        raise RuntimeError('SEED CANARY: no magnetic-species node found in the '
                           'sampled cached batches — cannot build the toy '
                           'graph.')
    x0, e0, f0 = src
    if x0.numel() != base_channels:
        raise RuntimeError(f'SEED CANARY: sampled node has {x0.numel()} '
                           f'features, expected the undecorated base '
                           f'{base_channels}.')
    d = _types.SimpleNamespace(
        x=torch.stack([x0, x0]).to(device),
        edge_index=torch.tensor([[0, 1], [1, 0]], device=device),
        edge_attr=torch.stack([e0, e0]).to(device),
        batch=torch.tensor([0, 0], device=device),
        num_graphs=1,
        z_fu=torch.tensor([1.0], device=device),
        material_id=['toy'],
        seed_sign=torch.tensor([float(sign_pair[0]),
                                float(sign_pair[1])], device=device),
        seed_fmag=torch.tensor([f0, f0], device=device),
        seed_elig=torch.tensor([1], dtype=torch.int8, device=device),
    )
    return d


def run_seed_canaries(model, sample_batches, dcfg, arm, stage):
    """Spec §4 verification layers, model side, on the two-site toy graph.

    stage='startup' ASSERTS the mechanism before any epoch:
      - AFM seed (+1,-1) BREAKS the equivalent-site symmetry (the
        equivalent-site obstruction is lifted: the two embeddings/moments differ);
      - FM seed (+1,+1) and the undecorated input PRESERVE it (outputs equal);
      - the Xu-conditioned model only, exact by construction: AFM site sum == 0 (compensation),
        global seed flip co-flips m_hat and leaves y_aux invariant (r_hat
        asymmetry reported — it reads the flip-invariant pooled h; bounded by
        2|r_hat|, see sdchemgnn_model docstring).
    stage='final' prints the SAME numbers on the trained model as REPORTS
    (SD-ChemGNN's learned equivariance + AFM compensation are evaluation
    material, not startup asserts — augmentation teaches them, nothing
    enforces them)."""
    was_training = model.training
    model.eval()
    base = int(dcfg['base_channels'])
    r = {}
    with torch.no_grad():
        d = decorate_batch(_toy_two_site_batch(sample_batches, base, (1, -1)),
                           'oracle', dcfg)
        o = model.forward_b(d)
        m = o['m_hat']
        r['afm_split'] = float((m[0] - m[1]).abs())
        r['afm_sum'] = float(m.sum())
        r['afm_abs'] = float(m.abs().sum())
        d2 = decorate_batch(_toy_two_site_batch(sample_batches, base, (1, -1)),
                            'oracle', dcfg)
        d2.x[:, base] = -d2.x[:, base]              # global seed flip
        o2 = model.forward_b(d2)
        r['flip_m'] = float((o2['m_hat'] + m).abs().max())
        r['flip_aux'] = float((o2['y_aux'].reshape(-1)
                               - o['y_aux'].reshape(-1)).abs().max())
        r['flip_r'] = float((o2['r_hat'] - o['r_hat']).abs().max())
        dfm = decorate_batch(_toy_two_site_batch(sample_batches, base, (1, 1)),
                             'oracle', dcfg)
        mfm = model.forward_b(dfm)['m_hat']
        r['fm_split'] = float((mfm[0] - mfm[1]).abs())
        du = decorate_batch(_toy_two_site_batch(sample_batches, base, (0, 0)),
                            'undecorated', dcfg)
        mu_ = model.forward_b(du)['m_hat']
        r['undec_split'] = float((mu_[0] - mu_[1]).abs())
    if was_training:
        model.train()
    print(f"    seed canary [{stage}/{arm}]: AFM split={r['afm_split']:.2e} "
          f"sum={r['afm_sum']:+.2e} sum|m|={r['afm_abs']:.2e} | "
          f"flip: max|m+flip(m)|={r['flip_m']:.2e} d(y_aux)={r['flip_aux']:.2e} "
          f"d(r_hat)={r['flip_r']:.2e} | FM split={r['fm_split']:.2e} "
          f"undec split={r['undec_split']:.2e}")
    if stage == 'startup':
        problems = []
        if not r['afm_split'] > 1e-9:
            problems.append(
                f"AFM seed did NOT break the equivalent-site symmetry "
                f"(split {r['afm_split']:.3g}) — the decoration never reaches "
                f"the node embeddings; check the channel wiring.")
        if not r['fm_split'] < 1e-4:
            problems.append(
                f"FM-seeded equivalent sites differ (split {r['fm_split']:.3g})"
                f" — the toy graph is not symmetric; canary construction bug.")
        if not r['undec_split'] < 1e-4:
            problems.append(
                f"undecorated equivalent sites differ "
                f"(split {r['undec_split']:.3g}).")
        if arm == 'xu_baseline':
            if not abs(r['afm_sum']) < 1e-4:
                problems.append(
                    f"Xu-conditioned AFM site sum {r['afm_sum']:.3g} != 0 — the "
                    f"projected head must compensate by construction.")
            if not r['flip_m'] < 1e-4:
                problems.append(
                    f"Xu-conditioned seed flip does not co-flip m_hat exactly "
                    f"(max err {r['flip_m']:.3g}) — symmetrization broken.")
            if not r['flip_aux'] < 1e-4:
                problems.append(
                    f"Xu-conditioned y_aux not flip-invariant ({r['flip_aux']:.3g}).")
        if problems:
            raise RuntimeError('SEED CANARY FAILED (startup):\n  '
                               + '\n  '.join(problems))
    return r


def main():
    session_t0 = time.time()        # budget clock starts NOW (includes data + cache)
    print(f'Using device: {device}')
    # Environment identity: makes "which GPU / which stack ran this" answerable
    # from the log alone — the first thing to compare when two runs of the same
    # code differ in speed (node throttling, module/env drift between runs).
    if torch.cuda.is_available():
        try:
            import torch_geometric as _pyg
            _pygv = _pyg.__version__
        except Exception:
            _pygv = '?'
        print(f'GPU: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__} '
              f'/ cuda {torch.version.cuda} / PyG {_pygv}  |  '
              f'TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}  |  '
              f'prefetch: {"ON" if os.environ.get("TRAIN_PREFETCH") == "1" else "off"}')
    print(f'Target: {TARGET}   (MAE in {_UNITS})')
    print(f'Loss: {cfg.loss_type}'
          + (f' (huber_delta={cfg.huber_delta})' if cfg.loss_type.lower() in ('huber', 'smooth_l1') else '')
          + (f' (mag_weight_scale={cfg.mag_weight_scale})' if cfg.loss_type.lower() == 'mag_weighted_l1' else '')
          + (f' (edges={cfg.binned_l1_edges}, weights={cfg.binned_l1_weights})'
             if cfg.loss_type.lower() == 'binned_l1' else ''))
    print(f'Selection metric (scheduler / early stop / best model): val {cfg.select_metric}')
    print(f'Target cap: |M| clamped to +/-{cfg.target_cap} {_UNITS} (train + metrics)')
    print(f'Dashboard bulk_cutoff = {cfg.bulk_cutoff} {_UNITS}'
          f'  (bulk = not-interesting |M|<=cutoff; tail = worth-pursuing)')

    # ── Dataset + split (identical to the original ChemGNN trainer) ────────────
    dataset = MPDataset(root=cfg.dataset_root, jsonl_path='')
    # ── Seed separation (initialization-seed replicates) ───────────────────────
    # The SPLIT is pinned by split_seed (defaults to cfg.seed) and must NEVER
    # move — test membership is the project's measuring stick. init_seed, when
    # set, re-seeds the global RNG AFTER the split draw, giving fresh weight
    # init / dropout / shuffle streams on IDENTICAL membership. init_seed=None
    # reproduces the legacy single-seed behavior bit-exactly (no re-seed).
    split_seed = int(getattr(cfg, 'split_seed', None) or cfg.seed)
    torch.manual_seed(split_seed)
    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]
    n_total = len(dataset)
    n_train = int(n_total * cfg.train_p)
    n_val   = int(n_total * cfg.val_p)
    train_ds = dataset[:n_train]
    val_ds   = dataset[n_train:n_train + n_val]
    test_ds  = dataset[n_train + n_val:]
    print(f'Dataset sizes  —  train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}')

    _init_seed = getattr(cfg, 'init_seed', None)
    if _init_seed is not None:
        torch.manual_seed(int(_init_seed))       # fresh init + dropout streams
        print(f'SEED-2 MODE: split_seed={split_seed} (membership pinned), '
              f'init_seed={int(_init_seed)} (fresh init/dropout/shuffle streams)')
    g = torch.Generator()
    g.manual_seed(int(_init_seed) if _init_seed is not None else cfg.seed)

    # ── Optional scope cut (step 2) ────────────────────────────────────────────
    if cfg.scope_max is not None:
        smax = float(cfg.scope_max)
        print('*' * 78)
        print(f'  SCOPE-RESTRICTED RUN: dropping |M| > {smax} {_UNITS} from train AND eval.')
        print(f'  This is a NARROWER-SCOPE model ("moments in 0..{smax} {_UNITS}"). Compare')
        print(f'  ONLY against a baseline re-evaluated on the same restriction.')
        print('*' * 78)
        def restrict(ds, tag):
            M = collect_M(ds, g)
            keep = (M.abs() <= smax) & ~torch.isnan(M)
            idx = keep.nonzero(as_tuple=True)[0]
            print(f'    {tag}: kept {int(keep.sum())}/{len(ds)} (dropped {len(ds)-int(keep.sum())})')
            return ds[idx]
        train_ds = restrict(train_ds, 'train')
        val_ds   = restrict(val_ds, 'val')
        test_ds  = restrict(test_ds, 'test')
        assert len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0, 'scope_max removed everything.'

    # ── Curated labels (RUN 4): apply the manifest AFTER the split ─────────────
    # Split membership stays identical to runs 1-3 (drops only remove entries;
    # nothing is reordered or re-drawn), so metrics remain comparable — this is
    # the same protocol curate_magnetization_labels.py used for the zero-GPU rescore.
    manifest_md5 = None
    if getattr(cfg, 'label_manifest', None):
        if cfg.scope_max is not None:
            raise RuntimeError('label_manifest with scope_max is untested — '
                               'run them separately.')
        print(f'LABEL MANIFEST: {cfg.label_manifest} (curated-label run)')
        man, manifest_md5 = load_manifest(cfg.label_manifest)
        print(f'    manifest rows: {len(man)}; md5 {manifest_md5}')
        _tm = time.time()
        ids, n_rel = relabel_master(dataset, man)      # vectorized master write
        # Verification layer 1 — item-path read-back: the very access pattern
        # that fooled the previous implementation must now return the new value.
        inv = torch.argsort(perm)
        _checked = 0
        for _j, _m in enumerate(ids):
            _r = man.get(_m)
            if _r is not None and _r[0] != 'drop':
                _it = dataset[int(inv[_j])]
                # Read back through get_magnetization — the READ PATH — not
                # through _it.y (the named-attribute lesson).
                _got = float(get_magnetization(_it).view(-1)[0])
                if abs(_got - _r[1]) > 1e-3:
                    raise RuntimeError(f'MANIFEST WRITE FAILED at item path: {_m} '
                                       f'expected {_r[1]:.4f}, read {_got:.4f}.')
                _checked += 1
                if _checked >= 5:
                    break
        # Drops: index selection on each split VIEW (membership preserved).
        drop_mids = {m for m, (a, _) in man.items() if a == 'drop'}
        def _drop(view, orig_idx, tag):
            keep = [ids[j] not in drop_mids for j in orig_idx.tolist()]
            n_drop = len(keep) - sum(keep)
            print(f'    manifest[{tag}]: {n_drop} dropped, {sum(keep)} kept')
            if n_drop == 0:
                return view
            return view[torch.tensor([i for i, k in enumerate(keep) if k])]
        train_ds = _drop(train_ds, perm[:n_train], 'train')
        val_ds   = _drop(val_ds,   perm[n_train:n_train + n_val], 'val')
        test_ds  = _drop(test_ds,  perm[n_train + n_val:], 'test')
        print(f'    {n_rel} labels rewritten in the master store; item-path '
              f'read-back OK ({_checked} checked); applied in {_fmt(time.time() - _tm)}')

    # ── Ordering-mode aux labels: derive -> attach -> verify ───────────────────
    # Modes come from the ordering side-table (MP `ordering` x curated label);
    # build_modes() cross-checks the side-table against THIS run's manifest row by
    # row and refuses to start on a stale table (a stale table once trained a
    # run on the wrong labels). The labels
    # ride the collated master into every cached batch; two verify layers below
    # (item path here, cached batches after the cache build) read them back
    # through get_ordering_mode — the exact accessor the aux loss uses.
    ordering_meta, ordering_mode_labels_by_id = None, None
    if getattr(cfg, 'ordering_sidetable', None):
        if str(getattr(cfg, 'mixture_variant', 'aux')).lower() != 'aux':
            raise NotImplementedError("mixture_variant='full' (V2) is deliberately "
                                      "not implemented yet — run the 'aux' mixture variant first.")
        if not getattr(cfg, 'label_manifest', None):
            raise RuntimeError('ordering_sidetable requires label_manifest: mode '
                               'labels are derived from the CURATED targets.')
        from ordering_mode_labels import build_modes
        print(f'ORDERING MODES: {cfg.ordering_sidetable} (ordering-diagnostic head labels)')
        _tm2 = time.time()
        ordering_mode_labels_by_id, ordering_meta = build_modes(
            cfg.ordering_sidetable, man,
            collapse_path=getattr(cfg, 'collapse_suspects', None),
            cap=float(cfg.target_cap),
            near_zero=float(getattr(cfg, 'mode_near_zero', 0.5)))
        print(f'    side-table md5 {ordering_meta["sidetable_md5"]}; manifest '
              f'row-level consistency OK; census {ordering_meta["census"]}')
        ids_m, n_lab_total = attach_ordering_mode_labels(dataset, ordering_mode_labels_by_id)
        _fresh_item_view(dataset)          # ── SD-ChemGNN: re-separate post-attach ──
        inv2 = torch.argsort(perm)
        _checked2 = 0
        for _j, _mid in enumerate(ids_m):
            _want = ordering_mode_labels_by_id.get(_mid, -1)
            if _want >= 0:
                _got = int(get_ordering_mode(dataset[int(inv2[_j])]).view(-1)[0])
                if _got != _want:
                    raise RuntimeError(f'ORDERING-MODE ATTACH FAILED at item path: '
                                       f'{_mid} expected {_want}, read {_got}.')
                _checked2 += 1
                if _checked2 >= 5:
                    break
        print(f'    modes attached to master store: {n_lab_total}/{len(ids_m)} labeled; '
              f'item-path read-back OK ({_checked2} checked); {_fmt(time.time() - _tm2)}')

    # ── Per-site node labels: build -> attach -> verify ────────────────────────
    if not getattr(cfg, 'magmom_sidetable', None):
        raise RuntimeError('sdchemgnn_config.magmom_sidetable is unset — '
                           'the site decomposition has no meaning without node labels.')
    if ordering_meta is None:
        raise RuntimeError('the site decomposition requires ordering_sidetable (raw per-f.u. '
                           'values for Z + diagnostic mode labels).')
    print(f'MAGMOM NODE LABELS: {cfg.magmom_sidetable} (per-site moment targets)')
    _tb = time.time()
    _slb = getattr(dataset, 'slices', None)
    if _slb is None:
        _slb = getattr(dataset, '_slices', None)
    _storeb = getattr(dataset, '_data', None)
    if _storeb is None:
        _storeb = dataset.data
    _xb = _slb['x'].tolist()
    _mids_master = [str(x[0] if isinstance(x, (list, tuple)) else x)
                    for x in _storeb.material_id]
    node_counts = {m: int(_xb[j + 1] - _xb[j])
                   for j, m in enumerate(_mids_master)}
    magmom_labels, magmom_meta = build_magmom_labels(
        cfg.magmom_sidetable, cfg.ordering_sidetable, man, node_counts,
        collapse_path=getattr(cfg, 'collapse_suspects', None),
        cap=float(cfg.target_cap),
        shift_tol=float(cfg.nodesup_shift_tol),
        zero_tol=float(cfg.no_magmom_zero_tol),
        resid_clip=float(cfg.resid_target_clip))
    print(f'    magmom md5 {magmom_meta["magmom_md5"]}; manifest/collapse/shift '
          f'masking applied; residual recompute {magmom_meta["resid_recompute"]}; '
          f'sign sweep {magmom_meta["sign_sweep"]}')
    print(f'    census {magmom_meta["census"]}')
    print(f'    supervised: {magmom_meta["n_sup"]}/{magmom_meta["n_rows"]} graphs '
          f'({100.0 * magmom_meta["n_sup"] / max(magmom_meta["n_rows"], 1):.1f}%; '
          f'{magmom_meta["n_zero"]} all-zero, {magmom_meta["n_sup_nodes"]} nodes); '
          f'z_suspect {magmom_meta["z_suspects"]}; nsite_mismatch '
          f'{magmom_meta["nsite_mismatch"]}; |resid| median/p90/p99 = '
          f'{magmom_meta["resid_q"]["median"]:.3f}/{magmom_meta["resid_q"]["p90"]:.3f}/'
          f'{magmom_meta["resid_q"]["p99"]:.3f}')
    ids_b, n_sup_b = attach_magmom_labels(dataset, magmom_labels)
    _fresh_item_view(dataset)              # ── SD-ChemGNN: re-separate post-attach ──
    # item-path read-back through the SAME accessors the loss consumes
    _invb = torch.argsort(perm)
    _posb = {m: j for j, m in enumerate(ids_b)}
    _checkedb = 0
    for _mid, _rec in magmom_labels.items():
        if not _rec.get('sup') or _rec.get('why') != 'ok' or _mid not in _posb:
            continue
        _it = dataset[int(_invb[_posb[_mid]])]
        _gm = get_node_magmom(_it)
        _exp = torch.tensor(_rec['m'], dtype=_gm.dtype)
        if _gm.numel() != _exp.numel() or not torch.allclose(_gm, _exp, atol=1e-5):
            raise RuntimeError(f'MAGMOM ATTACH FAILED at item path: {_mid} node '
                               f'targets do not match the built labels.')
        if int(get_node_sup(_it).view(-1)[0]) != 1                 or abs(float(get_resid_target(_it).view(-1)[0]) - _rec['resid']) > 1e-4                 or abs(float(get_z_fu(_it).view(-1)[0]) - _rec['z']) > 1e-4:
            raise RuntimeError(f'MAGMOM ATTACH FAILED at item path: {_mid} '
                               f'per-graph fields do not match the built labels.')
        _checkedb += 1
        if _checkedb >= 5:
            break
    print(f'    node labels attached to master store: {n_sup_b}/{len(ids_b)} '
          f'supervised; item-path read-back OK ({_checkedb} checked); '
          f'{_fmt(time.time() - _tb)}')

    # ── SD-ChemGNN: seed table build -> attach -> tripwire ─────────────────────
    print(f'SEED DECORATION: deriving spin seeds from the node-label state '
          f'(eps={cfg.seed_epsilon} uB; sign-convention corner masked per '
          f'precondition 0.iv)')
    _tsd = time.time()
    seeds, seed_meta = build_seed_table(
        cfg.magmom_sidetable, cfg.ordering_sidetable, man, magmom_labels,
        node_counts, eps=float(cfg.seed_epsilon),
        gauge_min_y=float(cfg.seed_gauge_min_y),
        gauge_min_summ=float(cfg.seed_gauge_min_summ),
        cap=float(cfg.target_cap))
    _n_trip = verify_seed_vs_sidetable(seeds, cfg.magmom_sidetable,
                                       float(cfg.seed_epsilon))
    _ss = seed_meta['site_signs']
    print(f"    eligible {seed_meta['n_elig']}/{seed_meta['n_rows']} graphs "
          f"({seed_meta['n_nm_seed']} NM seeds, {seed_meta['n_mixed_sign']} "
          f"mixed-sign); {seed_meta['n_sites_decorated']} decorated sites "
          f"[+{_ss['plus']} -{_ss['minus']} 0:{_ss['zero']}]; eps-band "
          f"{seed_meta['n_eps_band_sites']}; promotable "
          f"{seed_meta['n_promotable_sites']}")
    print(f"    gauge conflicts seed-masked: {seed_meta['n_gauge_conflicts']}"
          + (f" (e.g. {', '.join(seed_meta['gauge_conflict_ids'][:6])}"
             + (' ...' if seed_meta['n_gauge_conflicts'] > 6 else '') + ')'
             if seed_meta['n_gauge_conflicts'] else '')
          + f"; census {seed_meta['census']}")
    print(f'    tripwire: {_n_trip} rows re-derived from the raw side-table — '
          f'consistent')
    ids_s, n_elig_s = attach_seed_channels(dataset, seeds)
    _fresh_item_view(dataset)              # ── SD-ChemGNN: re-separate post-attach ──
    # item-path read-back through the SAME accessors decoration consumes
    _invs = torch.argsort(perm)
    _poss = {m: j for j, m in enumerate(ids_s)}
    _checkeds = 0
    for _mid, _rec in seeds.items():
        if not _rec.get('elig') or _rec.get('why') != 'ok' or _mid not in _poss:
            continue
        _it = dataset[int(_invs[_poss[_mid]])]
        _gs = get_seed_sign(_it)
        _es = torch.tensor([float(s) for s in _rec['sign']], dtype=_gs.dtype)
        if _gs.numel() != _es.numel() or not torch.equal(_gs, _es):
            raise RuntimeError(f'SEED ATTACH FAILED at item path: {_mid} seed '
                               f'signs do not match the built table.')
        _gf = get_seed_fmag(_it)
        _ef = torch.tensor(_rec['fmag'], dtype=_gf.dtype)
        if not torch.allclose(_gf, _ef, atol=1e-6) \
                or int(get_seed_elig(_it).view(-1)[0]) != 1:
            raise RuntimeError(f'SEED ATTACH FAILED at item path: {_mid} '
                               f'fmag/elig do not match the built table.')
        _checkeds += 1
        if _checkeds >= 5:
            break
    print(f'    seed channels attached to master store: {n_elig_s}/{len(ids_s)}'
          f' eligible; item-path read-back OK ({_checkeds} checked); '
          f'{_fmt(time.time() - _tsd)}')

    base_in_channels = train_ds[0].x.size(dim=-1)
    in_channels = base_in_channels + SEED_CHANNELS         # ── SD-ChemGNN ──
    edge_dim    = train_ds[0].edge_attr.shape[1]
    print(f'in_channels: {base_in_channels} base + {SEED_CHANNELS} seed = '
          f'{in_channels}, edge_dim: {edge_dim}')

    # NaN-label check (drives whether the hot train loop needs masking).
    M_train = collect_M(train_ds, g)
    has_nan = bool(torch.isnan(M_train).any())
    print(f'NaN labels in train: {has_nan}'
          f'{"" if has_nan else "  (train loop runs mask-free)"}')

    # For binned_l1: report each bin's TRAIN population and its effective share of
    # the loss (share_b = frac_b * w_b / sum(frac * w)) so the actual gradient
    # allocation of this run is printed next to the config that produced it.
    if cfg.loss_type.lower() == 'binned_l1':
        Mc = cap_target(M_train[~torch.isnan(M_train)]).abs()
        _edges = torch.tensor([float(e) for e in cfg.binned_l1_edges])
        _w = torch.tensor([float(x) for x in cfg.binned_l1_weights])
        _counts = torch.bincount(torch.bucketize(Mc, _edges), minlength=len(_w)).float()
        _frac = _counts / _counts.sum().clamp(min=1)
        _share = (_frac * _w) / (_frac * _w).sum().clamp(min=1e-12)
        _lo = [None, *cfg.binned_l1_edges]
        _hi = [*cfg.binned_l1_edges, cfg.target_cap]
        print('binned_l1 train bins (capped |M|, per f.u.):')
        for i in range(len(_w)):
            rng = (f'|M| <= {_hi[i]:g}' if i == 0 else f'{_lo[i]:g} < |M| <= {_hi[i]:g}')
            print(f'    bin {i} [{rng:>18s}]  n={int(_counts[i]):6d} ({_frac[i] * 100:4.1f}%)  '
                  f'w={_w[i]:g}  ->  loss share {_share[i] * 100:4.1f}%')

    # ── Loaders: RAM-cached (collate once) or streaming (fallback) ─────────────
    eval_batch = cfg.batch_size * cfg.eval_batch_factor
    if cfg.cache_batches:
        print('Caching pre-collated batches into RAM (one-time; collation happens once)...')
        tc = time.time()
        train_cache = build_cache(train_ds, cfg.batch_size, cfg.cache_workers, g, tag='train')
        val_cache   = build_cache(val_ds,   eval_batch,     cfg.cache_workers, g, tag='val')
        test_cache  = build_cache(test_ds,  eval_batch,     cfg.cache_workers, g, tag='test')
        # Verification layer 2 — the tensors the model actually consumes:
        if getattr(cfg, 'label_manifest', None):
            verify_manifest_in_cache(train_cache, man, 'train')
            verify_manifest_in_cache(val_cache, man, 'val')
            verify_manifest_in_cache(test_cache, man, 'test')
        if ordering_meta is not None:
            verify_modes_in_cache(train_cache, ordering_mode_labels_by_id, 'train')
            verify_modes_in_cache(val_cache, ordering_mode_labels_by_id, 'val')
            verify_modes_in_cache(test_cache, ordering_mode_labels_by_id, 'test')
        # Verification layer 2, node edition — the tensors the node/residual
        # losses actually consume:
        verify_magmom_in_cache(train_cache, magmom_labels, 'train')
        verify_magmom_in_cache(val_cache, magmom_labels, 'val')
        verify_magmom_in_cache(test_cache, magmom_labels, 'test')
        # ── SD-ChemGNN: verification layer 2, seed edition ──
        verify_seed_in_cache(train_cache, seeds, 'train')
        verify_seed_in_cache(val_cache, seeds, 'val')
        verify_seed_in_cache(test_cache, seeds, 'test')
        compose_note = (f'train composition refreshed every {cfg.cache_reshuffle_every} epochs'
                        if cfg.cache_reshuffle_every else
                        'fixed composition, order reshuffled each epoch')
        print(f'  cached train={len(train_cache)} / val={len(val_cache)} / test={len(test_cache)} '
              f'batches in {_fmt(time.time() - tc)}  ({compose_note})')
        train_loader = CachedLoader(train_cache, shuffle=True,  generator=g)
        val_loader   = CachedLoader(val_cache,   shuffle=False, generator=g)
        test_loader  = CachedLoader(test_cache,  shuffle=False, generator=g)
        # Opt-in overlap of clone+H2D with compute (see _CudaPrefetcher). Purely
        # operational (not fingerprinted): flip via env between sessions freely.
        if os.environ.get('TRAIN_PREFETCH') == '1' and torch.cuda.is_available():
            train_loader = _CudaPrefetcher(train_loader, device)
            val_loader   = _CudaPrefetcher(val_loader, device)
            test_loader  = _CudaPrefetcher(test_loader, device)
            print('Prefetch: ON — a worker thread stages clone+H2D up to 2 batches '
                  'ahead on a side CUDA stream (TRAIN_PREFETCH=1)')
    else:
        train_loader = make_loader(train_ds, cfg.batch_size, cfg.shuffle, True, g, cfg.num_workers)
        val_loader   = make_loader(val_ds,   eval_batch,     False,      True, g, cfg.eval_num_workers)
        test_loader  = make_loader(test_ds,  eval_batch,     False,      True, g, int(cfg.eval_num_workers / 2))
        print(f'Streaming loaders: train workers={cfg.num_workers}, val/test='
              f'{cfg.eval_num_workers}/{int(cfg.eval_num_workers / 2)} '
              f'(cache_batches=False)')

    # ── Model / optimizer / schedule / loss ────────────────────────────────────
    model = make_model_bs(in_channels, edge_dim, cfg).to(device)   # ── SD-ChemGNN ──
    optimizer = build_optimizer(model, cfg.learning_rate, cfg.wd_gnn, cfg.wd_mlp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=cfg.plateau_factor, patience=cfg.plateau_patience,
        min_lr=cfg.plateau_min_lr)
    loss_fn = make_loss(cfg)
    print(f'\nModel:\n{model}')

    # ── auxiliary-head config: CE class weights from the TRAIN split ───────────
    # Computed from the tensors the loss will actually see (cached train batches /
    # the streaming loader), post-drop — not from the side-table's global census.
    aux = None
    if ordering_meta is not None:
        if cfg.cache_batches:
            _km_all = torch.cat([get_ordering_mode(b) for b in train_cache])
        else:
            _tmp_loader = make_loader(train_ds, cfg.batch_size * cfg.eval_batch_factor,
                                      False, False, g, cfg.eval_num_workers)
            _km_all = torch.cat([get_ordering_mode(b) for b in _tmp_loader])
            print('    (streaming loaders: cached-batch mode verify not applicable)')
        _lab_all = _km_all[_km_all >= 0].long()
        _cnt = torch.bincount(_lab_all, minlength=3).float()
        if int((_cnt > 0).sum()) < 3:
            raise RuntimeError(f'a mode class is MISSING from the train split '
                               f'(counts {[int(x) for x in _cnt.tolist()]}) — check '
                               f'side-table / split before training.')
        _w = _cnt.sum() / (3.0 * _cnt)
        aux = {'w_mode': float(cfg.mode_loss_weight),
               'w_val': float(cfg.mode_value_weight),
               'ce_w': _w.to(device), 'empty_batches': 0}
        print(f'Mode-head supervision on TRAIN: {int(_lab_all.numel())} labeled '
              f'({100.0 * _lab_all.numel() / max(_km_all.numel(), 1):.1f}% of graphs); '
              f'counts net~0/partial/aligned = {[int(x) for x in _cnt.tolist()]}; '
              f'CE weights {[round(float(x), 2) for x in _w.tolist()]}; '
              f'lambda_mode={cfg.mode_loss_weight} lambda_val={cfg.mode_value_weight}')

    # ── OPTION-B loss config + train-split node-supervision census ─────────────
    if aux is None:
        raise RuntimeError('the site-decomposition model requires the ordering aux channel — '
                           'ordering_sidetable must be set.')
    bopt = {'w_node': float(cfg.node_loss_weight),
            'w_resid': float(cfg.resid_loss_weight),
            'w_shrink': float(cfg.resid_shrink_weight),
            'w_aux': float(cfg.aux_post_mlp_weight),
            'huber_d': float(cfg.resid_huber_delta),
            'empty_batches': 0}

    # ── SD-ChemGNN: decoration policy + RNG stream ─────────────────────────────
    bs_arm = str(getattr(cfg, 'bs_arm', 'seed_input')).lower()
    _coflip = bool(getattr(cfg, 'seed_coflip', True))
    if bs_arm == 'xu_baseline' and _coflip:
        _coflip = False          # the Xu-conditioned model's gauge is architectural and exact —
        print('    (xu_baseline: co-flip augmentation forced OFF — the '
              'symmetrized embedding is exactly flip-invariant)')
    dcfg = {'base_channels': int(base_in_channels),
            'p_undecorate': float(cfg.seed_p_undecorate),
            'p_site_mask': float(cfg.seed_p_site_mask),
            'p_promote': float(cfg.seed_p_promote),
            'fmag_scale': float(cfg.seed_fmag_scale),
            'coflip': _coflip}
    # dedicated, resumable RNG stream for the corruption/co-flip draws — kept
    # OUT of the global/dropout and loader streams so decoration randomness
    # never perturbs the stochastic state of the undecorated predecessor it is
    # compared against
    dec_g = torch.Generator()
    dec_g.manual_seed((int(_init_seed) if _init_seed is not None
                       else int(cfg.seed)) + 202608)
    dec_train = {'dcfg': dcfg, 'gen': dec_g}
    # eval-mode maps: what "undecorated" means for this variant (the Xu-conditioned model's projected
    # head emits 0 on unseeded sites, so its fallback is the proposer's FM
    # hypothesis when xu_undecorated_policy == 'fm_formal')
    undec_mode = 'undecorated'
    if bs_arm == 'xu_baseline' and str(getattr(
            cfg, 'xu_undecorated_policy', 'fm_formal')).lower() == 'fm_formal':
        undec_mode = 'fm_formal'
    sel_mode = str(getattr(cfg, 'bs_select_mode', 'oracle')).lower()
    if sel_mode not in ('oracle', 'undecorated'):
        raise ValueError(f"bs_select_mode must be 'oracle' or 'undecorated', "
                         f'got {sel_mode!r}')
    dec_sel = {'mode': 'oracle' if sel_mode == 'oracle' else undec_mode,
               'dcfg': dcfg}
    dec_undec = {'mode': undec_mode, 'dcfg': dcfg}
    dec_oracle = {'mode': 'oracle', 'dcfg': dcfg}

    def dec_shuffled():
        g_sh = torch.Generator()
        g_sh.manual_seed(1729)               # fixed: probe reproducible per call
        return {'mode': 'shuffled', 'dcfg': dcfg, 'gen': g_sh}

    print(f"Seed decoration: variant={bs_arm}  select-mode={dec_sel['mode']}  "
          f"undecorated-mode={undec_mode}  coflip={dcfg['coflip']}  "
          f"p_undec={dcfg['p_undecorate']} p_site_mask={dcfg['p_site_mask']} "
          f"p_promote={dcfg['p_promote']}  eps={cfg.seed_epsilon} "
          f"fmag_scale={dcfg['fmag_scale']}")
    if cfg.cache_batches:
        _sup_all = torch.cat([get_node_sup(b) for b in train_cache])
        _nsup_tr = int((_sup_all == 1).sum())
        print(f'Node supervision on TRAIN: {_nsup_tr}/{_sup_all.numel()} graphs '
              f'({100.0 * _nsup_tr / max(_sup_all.numel(), 1):.1f}%); '
              f'lambda_node={cfg.node_loss_weight} lambda_resid={cfg.resid_loss_weight} '
              f'shrink={cfg.resid_shrink_weight} aux={cfg.aux_post_mlp_weight} '
              f'(resid clip +/-{cfg.resid_target_clip}, Huber d={cfg.resid_huber_delta})')
        # One-batch eval-mode parity: plain forward vs forward_b()['y_aux'].
        # forward_b runs the PARENT forward unmodified, so any difference beyond
        # the machine's own nondeterminism means the capture perturbed the
        # trunk. On GPU that floor is NOT zero: CEALConv's scatter aggregation
        # uses atomics, so two IDENTICAL forwards differ at ~1e-6..1e-3 (a
        # fixed atol=1e-6 once tripped on exactly this). CALIBRATE
        # first — two plain forwards on fresh clones measure the noise floor —
        # then require the captured path within the WIDENED margin
        # max(20x noise floor, 1e-3): an earlier run passed its 10x/1e-4 cap
        # with ZERO headroom and the next unlucky floor would HALT the run via
        # the crash breaker; the hard 0.05 baseline check below is unchanged.
        # SD-ChemGNN: parity runs on an UNDECORATED-mode
        # decorated batch (x widened by three zero/FM-hypothesis channels) so
        # both paths consume the width the model was built for.
        if len(val_cache):
            def _par_batch():
                return _dec_eval(val_cache[0].clone().to(device), dec_undec)
            model.eval()
            with torch.no_grad():
                _y1a = model(_par_batch()).reshape(-1)
                _y1b = model(_par_batch()).reshape(-1)
                _resb = model.forward_b(_par_batch())
                _y2 = _resb['y_aux'].reshape(-1)
            model.train()
            _noise = float((_y1a - _y1b).abs().max())
            _diff = float((_y1a - _y2).abs().max())
            if _noise > 0.05:
                raise RuntimeError(
                    f'PARITY BASELINE BROKEN: two plain forwards on identical '
                    f'fresh clones differ by {_noise:.3g} — forward mutates its '
                    f'input or the node is misbehaving; do not train.')
            _thresh = max(20.0 * _noise, 1e-3)         # ── SD-ChemGNN: widened ──
            if _diff > _thresh:
                raise RuntimeError(
                    f'PARITY CHECK FAILED: forward_b()["y_aux"] differs from the '
                    f'plain forward by {_diff:.3g} (machine noise floor '
                    f'{_noise:.3g}, threshold {_thresh:.3g}) — the capture '
                    f'perturbs the trunk; do not train.')
            print(f'    parity check OK: max|y_aux - plain|={_diff:.2e} within '
                  f'20x machine noise ({_noise:.2e}, min 1e-3); reconstruction '
                  f'live (first-batch mean|r_hat|='
                  f'{float(_resb["r_hat"].abs().mean()):.4f}, zero-init expects 0)')
            # ── SD-ChemGNN: startup canaries (mechanism asserts) ──
            run_seed_canaries(model, val_cache[:8], dcfg, bs_arm, 'startup')

    # ── Run folder ──────────────────────────────────────────────────────────────
    # Chain mode (TRAIN_RUN_DIR set): every session of the chain shares this ONE
    # folder — resume.pt, best_model.pt and all artifacts accumulate here.
    # Single-session mode: fresh auto-numbered folder (atomic; race-safe).
    if RUN_DIR_ENV:
        folder = RUN_DIR_ENV
        os.makedirs(folder, exist_ok=True)
    else:
        formatted_date = datetime.now().strftime('%m_%d_%Y')
        os.makedirs('./Results', exist_ok=True)
        next_id = 1
        while True:
            folder = f'./Results/ChemGNN_magneticBS_{formatted_date}_{next_id}'
            try:
                os.makedirs(folder, exist_ok=False)
                break
            except FileExistsError:
                next_id += 1
    prefix  = f'{cfg.model_type}_magneticBS_Mfu'
    timestr = str(int(time.time() * 1000000))
    print(f'Results folder: {folder}')

    hyper_dict = {
        'target': TARGET, 'model_type': cfg.model_type, 'target_cap': cfg.target_cap,
        'loss_type': cfg.loss_type, 'huber_delta': cfg.huber_delta,
        'mag_weight_scale': cfg.mag_weight_scale,
        'binned_l1_edges': cfg.binned_l1_edges, 'binned_l1_weights': cfg.binned_l1_weights,
        'label_manifest': getattr(cfg, 'label_manifest', None),
        'label_manifest_md5': manifest_md5,
        'select_metric': cfg.select_metric,
        'plateau_factor': cfg.plateau_factor, 'plateau_patience': cfg.plateau_patience,
        'plateau_min_lr': cfg.plateau_min_lr,
        'es_patience': cfg.es_patience, 'es_min_delta': cfg.es_min_delta,
        'bulk_cutoff': cfg.bulk_cutoff,
        'money_zone_edges': cfg.money_zone_edges, 'scope_max': cfg.scope_max,
        'batch_size': cfg.batch_size, 'eval_batch_factor': cfg.eval_batch_factor,
        'num_workers': cfg.num_workers, 'eval_num_workers': cfg.eval_num_workers,
        'cache_batches': cfg.cache_batches, 'cache_reshuffle_every': cfg.cache_reshuffle_every,
        'train_p': cfg.train_p, 'val_p': cfg.val_p,
        'epochs': cfg.epochs, 'learning_rate': cfg.learning_rate,
        'wd_gnn': cfg.wd_gnn, 'wd_mlp': cfg.wd_mlp,
        'out_channels': cfg.out_channels, 'numLayers': cfg.numLayers,
        'aggregators': cfg.aggregators, 'scalers': cfg.scalers,
        'aggMLP': cfg.aggMLP, 'aggMLP_factor': cfg.aggMLP_factor,
        'dropout': cfg.dropout, 'norm': cfg.norm, 'towers': cfg.towers,
        'pre_layers': cfg.pre_layers, 'post_layers': cfg.post_layers,
        'divide_input': cfg.divide_input,
    }
    # SEED-2: fingerprinted only when init_seed is set (fresh chains only).
    if getattr(cfg, 'init_seed', None) is not None:
        hyper_dict.update({'split_seed': split_seed, 'init_seed': int(cfg.init_seed)})
    # auxiliary head: fingerprint the aux-label channel ONLY when enabled, so with
    # ordering_sidetable=None this file resumes pre-head chains untouched.
    if ordering_meta is not None:
        hyper_dict.update({
            'ordering_sidetable': cfg.ordering_sidetable,
            'ordering_sidetable_md5': ordering_meta['sidetable_md5'],
            'collapse_suspects': ordering_meta['collapse_suspects'],
            'collapse_md5': ordering_meta['collapse_md5'],
            'mode_near_zero': ordering_meta['near_zero'],
            'mode_loss_weight': cfg.mode_loss_weight,
            'mode_value_weight': cfg.mode_value_weight,
            'mixture_variant': cfg.mixture_variant,
        })
    # the site-decomposition model: the node-label channel is always on in this trainer.
    hyper_dict.update({
        'optionB': True,
        'magmom_sidetable': cfg.magmom_sidetable,
        'magmom_sidetable_md5': magmom_meta['magmom_md5'],
        'node_loss_weight': cfg.node_loss_weight,
        'resid_loss_weight': cfg.resid_loss_weight,
        'resid_shrink_weight': cfg.resid_shrink_weight,
        'resid_target_clip': cfg.resid_target_clip,
        'resid_huber_delta': cfg.resid_huber_delta,
        'aux_post_mlp_weight': cfg.aux_post_mlp_weight,
        'nodesup_shift_tol': cfg.nodesup_shift_tol,
        'no_magmom_zero_tol': cfg.no_magmom_zero_tol,
    })
    # ── SD-ChemGNN: decoration policy fingerprint (any change = new chain) ──
    _dec_policy = {
        'optionBS': True,
        'bs_arm': bs_arm,
        'seed_epsilon': float(cfg.seed_epsilon),
        'seed_p_undecorate': dcfg['p_undecorate'],
        'seed_p_site_mask': dcfg['p_site_mask'],
        'seed_p_promote': dcfg['p_promote'],
        'seed_fmag_scale': dcfg['fmag_scale'],
        'seed_coflip': dcfg['coflip'],
        'seed_gauge_min_y': float(cfg.seed_gauge_min_y),
        'seed_gauge_min_summ': float(cfg.seed_gauge_min_summ),
        'bs_select_mode': dec_sel['mode'],
        'bs_undec_mode': undec_mode,
        'sc_tiny': float(getattr(cfg, 'sc_tiny', 0.05)),
        'resid_per_fu': True,
        'base_in_channels': int(base_in_channels),
        'seed_channels': int(SEED_CHANNELS),
        'n_seed_eligible': int(seed_meta['n_elig']),
        'n_gauge_conflicts': int(seed_meta['n_gauge_conflicts']),
    }
    if bs_arm == 'xu_baseline':
        _dec_policy.update({
            'xu_spin_embed_dim': int(getattr(cfg, 'xu_spin_embed_dim', 16)),
            'xu_undecorated_policy': str(getattr(cfg, 'xu_undecorated_policy',
                                                 'fm_formal')),
        })
    import hashlib as _hl
    _dec_policy['decoration_policy_hash'] = _hl.md5(
        repr(sorted((k, repr(v)) for k, v in _dec_policy.items()
                    )).encode()).hexdigest()[:10]
    hyper_dict.update(_dec_policy)
    # ── Selection metric (validated before any state is touched) ───────────────
    sel = cfg.select_metric.lower()
    if sel not in ('loss', 'mae'):
        raise ValueError(f"select_metric must be 'loss' or 'mae', got {cfg.select_metric!r}")

    # ── Fresh start vs. resume ─────────────────────────────────────────────────
    best, best_epoch, no_improve = float('inf'), 0, 0
    patience, min_delta = cfg.es_patience, cfg.es_min_delta
    ep_list, trl_hist, val_hist, vmae_hist = [], [], [], []
    start_epoch, cum_prev = 1, 0.0

    rs = load_resume(folder)
    if rs is not None:
        check_fingerprint(rs.get('fingerprint'), hyper_dict)
        model.load_state_dict(rs['model'])
        optimizer.load_state_dict(rs['optimizer'])
        scheduler.load_state_dict(rs['scheduler'])
        torch.set_rng_state(rs['torch_rng'])                    # dropout stream (CPU)
        if rs.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rs['cuda_rng'])        # dropout stream (GPU)
        g.set_state(rs['loader_g'])                             # shuffle-order stream
        if rs.get('dec_g') is not None:                         # ── SD-ChemGNN ──
            dec_g.set_state(rs['dec_g'])                        # corruption stream
        best, best_epoch, no_improve = rs['best'], rs['best_epoch'], rs['no_improve']
        ep_list, trl_hist, val_hist, vmae_hist = rs['hist']
        timestr, cum_prev = rs['timestr'], rs['cum_seconds']
        start_epoch = rs['epoch'] + 1
        print(f'RESUME from {_resume_path(folder)}: continuing at epoch {start_epoch} '
              f'(completed {rs["epoch"]}; best val {sel}={best:.4f} @ {best_epoch}; '
              f'no_improve={no_improve}; lr={optimizer.param_groups[0]["lr"]:.2e}; '
              f'{_fmt(cum_prev)} trained in earlier sessions)')
    else:
        save_hyper_parameter(folder,
                             hyper_para_filename=f'{prefix}_{timestr}_hyper_parameters.json',
                             hyper_dict=hyper_dict)

    # ── Training loop (early stop on the val SELECTION metric) ────────────────
    # select_metric='loss' tracks the TRAINING OBJECTIVE on val (for binned_l1 the
    # tail-weighted MAE; for 'l1' identical to plain MAE, so legacy runs are
    # unchanged). Scheduler, early stopping and best-model saving all follow it, so
    # a reweighted run is not steered/stopped/selected by the headline MAE it is
    # deliberately trading away. select_metric='mae' restores the old behavior.
    t0 = time.time()               # this session's loop start (rate / ETA basis)
    ep_times = []                  # recent per-epoch wall times (budget estimator)
    resume_every = max(1, int(getattr(cfg, 'resume_every', 1)))
    budget_end = SESSION_END or (session_t0 + SESSION_S if SESSION_S else 0.0)
    if budget_end:
        print(f'Session budget: {_fmt(max(budget_end - time.time(), 0))} remaining '
              f'({"absolute deadline" if SESSION_END else "elapsed-based"}; startup '
              f'consumed {_fmt(time.time() - session_t0)}; yields when < one epoch '
              f'+ {cfg.session_margin_s:g}s remain)')
    print(f'\n=== Training (max {cfg.epochs} epochs) ===')
    for epoch in range(start_epoch, cfg.epochs + 1):
        te = time.time()
        # Optional: periodically re-collate train batches with fresh composition.
        if (cfg.cache_batches and cfg.cache_reshuffle_every
                and epoch > 1 and (epoch - 1) % cfg.cache_reshuffle_every == 0):
            train_loader = CachedLoader(
                build_cache(train_ds, cfg.batch_size, cfg.cache_workers, g,
                            shuffle_compose=True, tag='train-reshuffle'),
                shuffle=True, generator=g)
        train_loss, train_mae, aux_stats, bstats = train_epoch(
            model, optimizer, loss_fn, train_loader, has_nan,
            profile=(epoch == start_epoch), aux=aux, bopt=bopt,
            dec=dec_train)                                  # ── SD-ChemGNN ──
        val = evaluate(model, val_loader, loss_fn, dec_sel)
        scheduler.step(val[sel])
        ep_times = (ep_times + [time.time() - te])[-5:]

        if epoch % cfg.disp_step == 0:
            test = evaluate(model, test_loader, loss_fn, dec_sel)   # disp_step only
            lr = optimizer.param_groups[0]['lr']
            rate = (time.time() - t0) / max(1, epoch - start_epoch + 1)
            eta = rate * (cfg.epochs - epoch)              # rate from THIS session only
            # Two same-scale groups so train-vs-val gaps are meaningful:
            #   <loss_type>: the training objective (train numbers are mid-epoch w/ dropout)
            #   MAE: the reportable per-f.u. metric
            # then the val/test median + bulk/tail detail.
            print(f'[epoch {epoch}/{cfg.epochs}] '
                  f'{cfg.loss_type} train={train_loss:.4f} val={val["loss"]:.4f} test={test["loss"]:.4f} | '
                  f'MAE train={train_mae:.4f} val={val["mae"]:.4f} test={test["mae"]:.4f} | '
                  f'{fmt_breakdown("val", val)} {fmt_breakdown("test", test)} | '
                  f'lr={lr:.2e} {rate:.0f}s/ep ETA={_fmt(eta)}')
            print(f'    val tail (|M|>{cfg.bulk_cutoff:g}) MAE/N (rel%): {fmt_bands(val)}')
            if aux is not None:
                mv = evaluate_modes(model, val_loader, dec_sel)
                print(f'    val mode-head: acc={mv["acc"]:.3f} bal={mv["bal"]:.3f} '
                      f'recall[net~0/partial/aligned]='
                      f'{mv["recall"][0]:.3f}/{mv["recall"][1]:.3f}/{mv["recall"][2]:.3f} '
                      f'(n={mv["n"]})'
                      + (f' | train aux: ce={aux_stats["ce"]:.3f} l1={aux_stats["l1"]:.3f} '
                         f'lab={aux_stats["frac"] * 100:.0f}%'
                         if aux_stats else ''))
                nv = evaluate_nodes(model, val_loader, dec_sel)
                print(f'    val node-head: site-MAE={nv["site_mae"]:.3f} uB '
                      f'[|m|<0.5]={nv["band0"]:.3f} [0.5-2)={nv["band1"]:.3f} '
                      f'[2+]={nv["band2"]:.3f} (n={nv["n_sites"]}) | '
                      f'r_hat(fu): mean|{nv["r_abs"]:.3f}| corr={nv["r_corr"]:.2f} '
                      f'tail-share={nv["r_share"] * 100:.1f}% | '
                      f'recon/aux val MAE {nv["mae_recon"]:.4f}/{nv["mae_aux"]:.4f}'
                      + (f' | train: node={bstats["node"]:.3f} '
                         f'resid={bstats["resid"]:.3f}' if bstats else ''))
                # ── SD-ChemGNN: self-consistency + decoration census + leakage probe ──
                print(f'    val seed: self-consistency mean={nv["sc_mean"]:.3f} '
                      f'rate(all-sites)={nv["sc_rate"]:.3f} (n={nv["sc_n"]}) | '
                      f'train dec={bstats["dec_frac"] * 100:.0f}% '
                      f'flip={bstats["flip_frac"] * 100:.0f}%')
                _p_or = evaluate(model, val_loader, loss_fn, dec_oracle)
                _p_un = evaluate(model, val_loader, loss_fn, dec_undec)
                _p_sh = evaluate(model, val_loader, loss_fn, dec_shuffled())
                _ok = (_p_or['mae'] < _p_un['mae'] <= _p_sh['mae'] + 1e-9)
                print(f'    val leakage probe MAE: oracle={_p_or["mae"]:.4f} '
                      f'{undec_mode}={_p_un["mae"]:.4f} '
                      f'shuffled={_p_sh["mae"]:.4f}  '
                      f'[want oracle < {undec_mode} <= shuffled: '
                      f'{"OK" if _ok else "NOT YET"}]')
            ep_list.append(epoch); trl_hist.append(train_loss)
            val_hist.append(val['loss']); vmae_hist.append(val['mae'])
            # Refresh the train/val-vs-epoch txt + plot live, so progress is visible
            # during the run and survives a walltime timeout.
            save_curve(folder, prefix, ep_list, trl_hist, val_hist, vmae_hist, cfg.loss_type)

        if val[sel] < best - min_delta:
            best, best_epoch, no_improve = val[sel], epoch, 0
            save_model_CEAL(model, folder, 'best_model.pt', 'best_model_para.json')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch} '
                      f'(best val {sel}={best:.4f} @ epoch {best_epoch}).')
                break

        # ── Restart point + graceful session yield (chain mode) ────────────────
        # resume.pt is (re)written every resume_every epochs — atomic, a few MB —
        # so even a hard kill loses at most resume_every epochs of work.
        cum_now = cum_prev + (time.time() - t0)
        if epoch % resume_every == 0:
            save_resume(folder, epoch=epoch, model=model, optimizer=optimizer,
                        scheduler=scheduler, g=g, best=best, best_epoch=best_epoch,
                        no_improve=no_improve,
                        hist=(ep_list, trl_hist, val_hist, vmae_hist),
                        timestr=timestr, cum_seconds=cum_now, fingerprint=hyper_dict,
                        dec_g=dec_g)
        # Yield when a stop signal arrived, or when the remaining budget no longer
        # fits the next epoch (estimated by the SLOWEST recent epoch) plus margin.
        over_budget = bool(budget_end) and (
            time.time() + (max(ep_times) if ep_times else 0.0)
            + float(cfg.session_margin_s) > budget_end)
        if _YIELD['flag'] or over_budget:
            if epoch % resume_every != 0:              # capture this epoch too
                save_resume(folder, epoch=epoch, model=model, optimizer=optimizer,
                            scheduler=scheduler, g=g, best=best, best_epoch=best_epoch,
                            no_improve=no_improve,
                            hist=(ep_list, trl_hist, val_hist, vmae_hist),
                            timestr=timestr, cum_seconds=cum_now,
                            fingerprint=hyper_dict, dec_g=dec_g)
            why = _YIELD['why'] if _YIELD['flag'] else 'session budget reached'
            print(f'\nSESSION YIELD after epoch {epoch}/{cfg.epochs} ({why}); '
                  f'best val {sel}={best:.4f} @ epoch {best_epoch}; resume.pt written '
                  f'— the next session continues at epoch {epoch + 1}.', flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            # os._exit, NOT return: a normal exit finalizes the interpreter —
            # GC over the manifest lists + batch caches (~10^6 tensors, ~120 GB)
            # can outlive the scheduler's T-120s USR1, which then kills a finalizing
            # interpreter whose Python-level handlers are no longer serviced ->
            # phantom rc=138 "crashes" (seen in an earlier chained run) that
            # trip the circuit breaker. resume.pt is already atomically on
            # disk; nothing else needs flushing; skip finalization entirely.
            os._exit(0)                                # chain continues

    model.load_state_dict(torch.load(osp.join(folder, 'best_model.pt')))
    print(f'Best val {sel}={best:.4f} @ epoch {best_epoch}; best weights restored.')
    save_curve(folder, prefix, ep_list, trl_hist, val_hist, vmae_hist, cfg.loss_type)

    # ── SD-ChemGNN: final evaluation in two modes (undecorated / label-decorated
    # 'oracle'); the proposer-decorated mode lives in MagSieve (04_magsieve/).
    # Only the proposer-decorated mode ever claims deployability; the oracle
    # dumps are the mechanism evidence, the undecorated dumps the no-harm read.
    scope_note = '' if cfg.scope_max is None else f'  [SCOPE-RESTRICTED |M|<={cfg.scope_max}]'
    print('\n── Final (best weights) ──' + scope_note)
    if cfg.cache_batches and len(val_cache):
        # canaries on the TRAINED model: reports, not asserts — the
        # AFM compensation + learned equivariance ARE the mechanism readout
        run_seed_canaries(model, val_cache[:8], dcfg, bs_arm, 'final')

    final_modes = [('oracle', dec_oracle), ('undec', dec_undec)]
    finals = {}
    regression_files = {}
    for _mtag, _dec in final_modes:
        val_final = evaluate(model, val_loader, loss_fn, _dec)
        test_final = evaluate(model, test_loader, loss_fn, _dec)
        finals[_mtag] = test_final
        _mode_note = _dec['mode']
        print(f'\n  [{_mtag} mode = {_mode_note}]')
        print(f'  VAL   {cfg.loss_type}={val_final["loss"]:.4f}  MAE={val_final["mae"]:.4f}  {fmt_breakdown("", val_final)}')
        print(f'        tail (|M|>{cfg.bulk_cutoff:g}) MAE/N (rel%): {fmt_bands(val_final)}')
        print(f'  TEST  {cfg.loss_type}={test_final["loss"]:.4f}  MAE={test_final["mae"]:.4f}  {fmt_breakdown("", test_final)}')
        print(f'        tail (|M|>{cfg.bulk_cutoff:g}) MAE/N (rel%): {fmt_bands(test_final)}')
        pred_t, true_t = collect_pred_true(model, test_loader, _dec)
        regression_filename = f'{prefix}_{timestr}_regression_{_mtag}.txt'
        regression_files[_mtag] = regression_filename
        save_regression_result(folder, regression_filename,
                               pred_t.reshape(-1, 1), true_t.reshape(-1, 1))
        plot_regression_result(
            folder, filename=regression_filename,
            title=f'{prefix} [{_mtag}]: Predicted vs. Truth ({_UNITS}, '
                  f'capped at {cfg.target_cap})',
            plotfilename=f'{prefix}_{timestr}_regression_{_mtag}.png',
            disp=False)
        for _tag, _ld in (('VAL ', val_loader), ('TEST', test_loader)):
            nv = evaluate_nodes(model, _ld, _dec)
            print(f'  {_tag} node-head: site-MAE={nv["site_mae"]:.3f} uB '
                  f'[|m|<0.5]={nv["band0"]:.3f}/{nv["n_band"][0]} '
                  f'[0.5-2)={nv["band1"]:.3f}/{nv["n_band"][1]} '
                  f'[2+]={nv["band2"]:.3f}/{nv["n_band"][2]}   '
                  f'r_hat(fu): mean|{nv["r_abs"]:.3f}| corr={nv["r_corr"]:.2f} '
                  f'tail-share={nv["r_share"] * 100:.1f}%   '
                  f'recon/aux MAE {nv["mae_recon"]:.4f}/{nv["mae_aux"]:.4f}   '
                  f'self-consistency mean={nv["sc_mean"]:.3f} '
                  f'rate={nv["sc_rate"]:.3f}/{nv["sc_n"]}'
                  f'   (site-MAE at the undecorated predecessor\'s levels; '
                  f'r-honesty per f.u.)')
        _sd = collect_sitedecomp(model, test_loader, _dec)
        sitedecomp_filename = f'{prefix}_{timestr}_sitedecomp_{_mtag}.txt'
        np.savetxt(osp.join(folder, sitedecomp_filename), _sd.numpy(),
                   header='sum_abs_m sum_m resid_pred_fu z_fu sc_frac',
                   comments='', fmt='%.6f')
        print(f'  sitedecomp artifact: {sitedecomp_filename} ({_sd.size(0)} '
              f'rows, row-aligned with {regression_filename}) — the mechanism read '
              f'is the alignment ratio |sum_m|/sum_abs_m leaving the diagonal '
              f'on compensated rows; col 5 = per-graph self-consistency fraction')

    # headline + MAE file follow the SELECTION mode ('oracle' by default)
    _head_tag = 'oracle' if dec_sel['mode'] == 'oracle' else 'undec'
    test_final = finals[_head_tag]
    print(f'\n  headline test per-f.u. MAE [{_head_tag}] = '
          f'{test_final["mae"]:.4f} {_UNITS}   |  tail '
          f'(|M|>{cfg.bulk_cutoff}) MAE = {test_final["tail_mae"]:.4f}'
          + scope_note)
    print(f'  undecorated read: undec test MAE {finals["undec"]["mae"]:.4f} / tail '
          f'{finals["undec"]["tail_mae"]:.4f}')
    save_mae(folder=folder, maeFile=f'{prefix}_MAE.txt',
             test_loss=test_final['mae'],
             msg=f'{prefix}_{timestr}_loss-{cfg.loss_type}-{_head_tag}')

    if aux is not None:
        for _tag, _ld in (('VAL ', val_loader), ('TEST', test_loader)):
            mv = evaluate_modes(model, _ld, dec_sel)
            print(f'  {_tag} mode-head: acc={mv["acc"]:.3f} bal={mv["bal"]:.3f} '
                  f'recall[net~0/partial/aligned]='
                  f'{mv["recall"][0]:.3f}/{mv["recall"][1]:.3f}/{mv["recall"][2]:.3f} '
                  f'(n={mv["n"]})   confusion rows=true [net~0, partial, aligned]:')
            for _r in mv['confusion'].tolist():
                print('        ' + '  '.join(f'{int(_x):6d}' for _x in _r))
        p3, mu3, km3 = collect_modes(model, test_loader, dec_sel)
        modes_filename = f'{prefix}_{timestr}_modes.txt'
        _arr = np.concatenate([p3.numpy(), mu3.numpy(),
                               km3.view(-1, 1).float().numpy()], axis=1)
        np.savetxt(osp.join(folder, modes_filename), _arr,
                   header='p_net0 p_partial p_aligned mu_net0 mu_partial mu_aligned mode_label',
                   comments='', fmt='%.6f')
        print(f'  mode artifacts: {modes_filename} ({_arr.shape[0]} rows, '
              f'row-aligned with {regression_files[_head_tag]}; '
              f'{dec_sel["mode"]} mode)')

    # ── SD-ChemGNN: final shuffled-seed leakage probe (val + test) ─────────────
    for _tag, _ld in (('VAL ', val_loader), ('TEST', test_loader)):
        _p_or = evaluate(model, _ld, loss_fn, dec_oracle)
        _p_un = evaluate(model, _ld, loss_fn, dec_undec)
        _p_sh = evaluate(model, _ld, loss_fn, dec_shuffled())
        _ok = (_p_or['mae'] < _p_un['mae'] <= _p_sh['mae'] + 1e-9)
        _verdict = 'PASS' if _ok else ('FAIL — the seed channel is not being '
                                       'used as arrangement information')
        print(f'  {_tag} leakage probe MAE: oracle={_p_or["mae"]:.4f} '
              f'{undec_mode}={_p_un["mae"]:.4f} shuffled={_p_sh["mae"]:.4f}  '
              f'[oracle < {undec_mode} <= shuffled: {_verdict}]')

    # DONE marker: the machine-readable end-of-chain signal. A launcher can
    # check it after python exits (cancel the queued successor) and at session
    # start (a stray successor no-ops). Manual chain stop: touch this file.
    with open(osp.join(folder, 'DONE'), 'w') as fh:
        fh.write(f'{datetime.now().isoformat()}  best val {sel}={best:.4f} @ epoch {best_epoch}; '
                 f'[{_head_tag}] test MAE={test_final["mae"]:.4f}, '
                 f'tail MAE={test_final["tail_mae"]:.4f}; '
                 f'[undec] test MAE={finals["undec"]["mae"]:.4f}, '
                 f'tail MAE={finals["undec"]["tail_mae"]:.4f}\n')
    print('DONE')


if __name__ == '__main__':
    main()
