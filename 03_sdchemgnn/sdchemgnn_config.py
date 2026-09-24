"""
Configuration for SD-ChemGNN, the spin-decorated magnetic regressor.
Target: total_magnetization_normalized_formula_units (uB/f.u.)

The per-site decomposition is CONDITIONED on a proposed spin arrangement.
Three decoration channels (sign / formal spin-only magnitude / presence flag)
are appended to the node features AT RUNTIME (95 -> 98) by
spin_decoration.decorate_batch — the on-disk dataset (data.pt) is NOT
rebuilt; the decoration rides the collated master exactly like the per-site
labels, and the per-session RAM batch cache is rebuilt as always. Gauge is
handled by augmentation (random global co-flip of the decoration and of every
signed target; node loss gauge-pinned on decorated graphs); the Xu-conditioned
comparator ('xu_baseline') instead realises the gauge architecturally
(symmetrized spin embedding + sign-projected head, Xu et al. PNAS 2025).

Everything outside the SD-ChemGNN block is carried over UNCHANGED from the
site-decomposition configuration, so the decoration stays the single
attributed variable (loss terms, weights, heads and trunk are fixed). Two
further changes are implemented in the TRAINER (not knobs): the
size-consistent reconstruction y = sum(m)/Z + r_hat with the residual target
read per f.u., and the parity tolerance max(20x the measured noise floor,
1e-3).

The values below are those of the released weights (trained_model/):
init_seed 90211, split_seed 23647, model selection on the undecorated
validation loss. The paper's second run used init_seed 77031 and selected on
the label-decorated ('oracle') validation loss.

This file is FULLY STANDALONE — read via
    from sdchemgnn_config import config_magnetic
by train_sdchemgnn.py and by the MagSieve scoring pass.
sdchemgnn_model.make_model_bs() reads the backbone attributes plus
the bs_* / seed_* / xu_* knobs below.
"""


class ConfigMagnetic:

    # == Target ================================================================
    target     = 'total_magnetization_normalized_formula_units'   # MAE in uB/f.u.
    model_type = 'ChemGNN'

    # == Dataset (restated -- standalone) ======================================
    # SAME dataset root as the site-decomposition model: the seed channels are appended at runtime,
    # so data.pt is untouched and needs no rebuild.
    dataset_root = './dataset_mp_magnetic'
    cutoff       = 7            # Angstrom; graph-edge cutoff (for reference/repro)

    # == Data split (same seed/fractions as the original ChemGNN trainer) ======
    train_p = 0.8              # test fraction = 1 - train_p - val_p = 0.1
    val_p   = 0.1

    # == DataLoader ============================================================
    seed              = 23647  # controls the split shuffle and loader seeding
    shuffle           = True   # shuffle the training loader each epoch
    batch_size        = 512   # training batch size
    eval_batch_factor = 2      # val/test loaders use batch_size * this (forward-only)
    num_workers       = 10     # TRAIN loader workers (streaming fallback only)
    eval_num_workers  = 4      # eval loader workers (streaming fallback only)

    # == Batch caching (throughput) ============================================
    # Identical to the site-decomposition model: pre-collate ALL batches once into RAM; batch order
    # reshuffles per epoch; every step trains on a CLONE, which is also what
    # makes the runtime decoration safe (the cache itself is never widened or
    # flipped). cache_workers = 0 is the fork/shm-free fallback for the
    # /dev/shm crawl; operational knob, NOT fingerprinted.
    cache_batches         = True
    cache_reshuffle_every = 0   # 0 = fixed composition
    cache_workers = 10          # 0 = fork/shm-free fallback

    # == Logging ===============================================================
    disp_step    = 10          # print progress (+ dashboard) every N epochs
    dispProgress = False       # live matplotlib loss plot during training

    # == Target capping (screening-relevant ceiling) ===========================
    target_cap = 30.0          # uB/f.u.; |M| clamped for train + metrics

    # == Training loss =========================================================
    loss_type = 'binned_l1'    # the per-f.u. total ('umbrella') loss, unchanged

    # == binned_l1 =============================================================
    binned_l1_edges   = [2.0, 5.0, 10.0]
    binned_l1_weights = [1.0, 3.0, 4.0, 5.0]
    huber_delta       = 3.0    # (huber/smooth_l1 only; unused under binned_l1)
    mag_weight_scale  = 4.0    # (mag_weighted_l1 only)

    # == Reporting dashboard (always on) =======================================
    bulk_cutoff      = 2.0
    money_zone_edges = [5.0, 10.0]

    # == Curated labels ========================================================
    label_manifest = './label_manifest.csv'

    # == Seed separation =======================================================
    # Two training runs with different initialization seeds price the
    # seed-to-seed spread: init_seed = None is NOT allowed here — pin explicit
    # init seeds (90211 for the released run, 77031 for the paper's second run).
    split_seed = 23647     # membership pinned; None would fall back to seed (23647)
    init_seed  = 90211     # the released weights (trained_model/) were trained with
                           # this seed; the paper's second run used 77031

    # == Ordering-diagnostic auxiliary head (unchanged) ========================
    ordering_sidetable = './ordering_sidetable.csv'  # required (Z + mode labels)
    collapse_suspects  = './collapse_suspects.csv'
    mode_near_zero     = 0.5
    mode_loss_weight   = 0.2
    mode_value_weight  = 0.1
    mixture_variant    = 'aux'

    # == Site-decomposition blocks (unchanged) ==================================
    magmom_sidetable    = './magmom_sidetable.jsonl'
    node_loss_weight    = 0.5
    resid_loss_weight   = 0.1
    resid_shrink_weight = 0.01
    resid_target_clip   = 10.0   # uB/cell clip on the STORED target; the loss
                                 # reads it PER F.U. (target / z_fu), so the
                                 # effective per-f.u. clip is 10/Z — tighter on
                                 # bigger cells, which is the physical direction
    resid_huber_delta   = 1.0    # Huber knee, now on the per-f.u. residual
    aux_post_mlp_weight = 0.2
    nodesup_shift_tol   = 0.5
    no_magmom_zero_tol  = 0.1

    # == SD-ChemGNN — spin-decorated node features ==============================
    # Model-variant switch: SD-ChemGNN, or the Xu-conditioned comparator.
    bs_arm = 'seed_input'        # 'seed_input' (SD-ChemGNN) | 'xu_baseline' (Xu-conditioned)

    # Dead zone for sign assignment from labels (|m| <= eps -> sign 0).
    # 0.1 is also Xu et al.'s SCF-ambiguity threshold. The 0.5 sensitivity
    # check runs in the seed dry run (spin_decoration --eps-alt), not as a
    # training sweep.
    seed_epsilon = 0.1           # uB

    # Corruption schedule (train mode only; all rates fingerprinted and
    # folded into the decoration policy hash):
    seed_p_undecorate = 0.20     # graphs trained undecorated (preserves the
                                 #   undecorated mode and makes it scoreable)
    seed_p_site_mask  = 0.10     # decorated sites with all 3 channels zeroed
    seed_p_promote    = 0.10     # zero-sign magnetic-species sites promoted to
                                 #   a random +/-1 at formal magnitude (closes
                                 #   the "converged magnetic" one-bit leak)

    # Magnitude channel scale: raw formal uB / this (Eu/Gd = 7 -> 1.0).
    seed_fmag_scale = 7.0

    # The sign-convention corner: rows with
    # |y_cur| > gauge_min_y whose sum(m) contradicts y_cur's sign (and
    # |sum(m)| > gauge_min_summ) are seed-masked until the corner is resolved.
    seed_gauge_min_y    = 1.0    # uB/f.u.
    seed_gauge_min_summ = 0.1    # uB

    # Random global co-flip on decorated graphs (train mode). Forced
    # OFF at runtime for bs_arm='xu_baseline' (its invariance is architectural
    # and exact; augmenting it would only add noise).
    seed_coflip = True

    # Selection mode: which decoration mode the val dashboard / scheduler /
    # early stop / best-model saving run in. 'oracle' = the label-decorated
    # (conditional) task; 'undecorated' = the deployable mode. Every mode is
    # scored from the final dumps regardless.
    bs_select_mode = 'undecorated'  # 'oracle' | 'undecorated'; the released
                                    # run selected on the deployable mode, the
                                    # paper's second run on 'oracle'

    # Self-consistency readout: |m_hat| <= sc_tiny counts as
    # non-committal at a seeded site.
    sc_tiny = 0.05               # uB

    # Xu-conditioned comparator only:
    xu_spin_embed_dim     = 16          # e_s width; trunk in = 95 + this
    xu_undecorated_policy = 'fm_formal' # 'fm_formal': undecorated-mode eval
                                        #   feeds the proposer's trivial FM
                                        #   hypothesis (sign +1 on magnetic
                                        #   species) — the projected head
                                        #   emits 0 on unseeded sites, so
                                        #   'strict' zeros are degenerate;
                                        # 'strict': true zeros anyway (m_hat
                                        #   collapses to r_hat alone)

    # == Scope cut (off) =======================================================
    scope_max = None

    # == Backbone (single block; names match MyCEALNetwork constructor args) ====
    # FROZEN at the original ChemGNN backbone for attribution.
    out_channels  = 64
    aggregators   = ['sum', 'mean', 'min', 'max', 'std']
    scalers       = ['identity']
    towers        = 1
    numLayers     = 3
    pre_layers    = 1
    post_layers   = 1
    divide_input  = False
    aggMLP        = True
    aggMLP_factor = 0.125
    dropout       = 0.10
    norm          = 'layer'    # 'layer' or 'graph'

    # == Optimizer / schedule (frozen) =========================================
    epochs        = 1500
    learning_rate = 0.001
    wd_gnn        = 3e-3
    wd_mlp        = 3e-3

    plateau_factor   = 0.5
    plateau_patience = 12
    plateau_min_lr   = 1e-6

    es_patience  = 60
    es_min_delta = 1e-4

    select_metric = 'loss'

    # == Chained short sessions (busy-queue mode) ==============================
    resume_every     = 1     # epochs between resume.pt writes
    session_margin_s = 180   # yield when remaining budget < one epoch + this


config_magnetic = ConfigMagnetic
