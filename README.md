# SD-ChemGNN and MagSieve

Code for the manuscript on screening hypothetical crystal structures for new
magnets with a spin-decorated crystal-graph network (SD-ChemGNN) and a
label-free screening stage (MagSieve). The programs are the ones the paper's
runs used, reorganized into four stages and renamed for readability. Every
file's original name and md5 is listed in `PROVENANCE.tsv`. Released under
the MIT License (`LICENSE`).

| folder | stage |
|---|---|
| `01_mp_retrieval/` | download structures, magnetization, site moments and magnetic ordering from the Materials Project |
| `02_data_curation/` | build the crystal-graph dataset; the curated labels the paper used |
| `03_sdchemgnn/` | the SD-ChemGNN model, trainer, tests, and the paper's trained weights |
| `04_magsieve/` | MagSieve: spin-configuration proposer, ordering router, selector, shortlist scorer |

## Installation

Training needs Linux with an NVIDIA GPU. The offline checks and everything
else also run on a CPU or a Mac.

### Step 0: check your CUDA version

```bash
nvidia-smi
```

Note the `CUDA Version` it shows (for example `12.7`).

### Option A: `environment.yml`

For CUDA 12.8 or newer:

```bash
git clone https://github.com/LaiWei36/Sdchemgnn.git sdchemgnn
cd sdchemgnn
conda env create -f environment.yml
conda activate sdchemgnn
```

Then go to [Verify](#verify). If this fails, remove the environment
(`conda env remove -n sdchemgnn`) and use Option B.

### Option B: step by step

**B1. Create the environment.**

```bash
git clone https://github.com/LaiWei36/Sdchemgnn.git sdchemgnn
cd sdchemgnn
conda create -n sdchemgnn python=3.11 -y
conda activate sdchemgnn
```

**B2. Pick the tag for your CUDA version.** Replace `cu126` in B3 and B4
with it.

| CUDA version from Step 0 | tag |
|---|---|
| 12.9 or newer (including 13.x) | `cu129` |
| 12.8 | `cu128` |
| 12.0 to 12.7 | `cu126` |
| no GPU | `cpu` |

**B3. Install PyTorch.**

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

**B4. Install torch-scatter** (same tag as B3):

```bash
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
```

Use this command rather than a plain `pip install torch-scatter`, which
compiles a CPU-only version that fails on the GPU with `Not compiled with
CUDA support`.

On older Linux (Ubuntu 20.04, CentOS 7/8, many HPC clusters), importing it
fails with ``GLIBC_2.32 not found``. Compile it instead, with `cuda-version`
matching your tag (`12.6`, `12.8` or `12.9`); this takes 10 to 30 minutes:

```bash
pip uninstall -y torch-scatter
conda install -y -c conda-forge cuda-version=12.6 cuda-nvcc cuda-cudart-dev cuda-cccl \
      libcusparse-dev libcublas-dev libcusolver-dev libcurand-dev
export CUDA_HOME=$CONDA_PREFIX
FORCE_CUDA=1 pip install --no-build-isolation --no-cache-dir torch-scatter==2.1.2
```

**B5. Install the rest.**

```bash
pip install torch-geometric==2.8.0
pip install numpy==2.3.3 scipy==1.17.0 pandas==2.3.3 matplotlib==3.10.8 \
            pymatgen==2025.10.7 spglib==2.7.0 scikit-learn==1.9.0 \
            ase==3.28.0 tqdm==4.67.1 mp-api==0.46.1
```

**macOS:** follow Option B, but in B3 and B4 run `pip install torch==2.8.0`
and `pip install torch-scatter==2.1.2`.

### Verify

From the repository root:

```bash
python -c "import torch, torch_scatter; print(torch.cuda.is_available(), torch.ops.torch_scatter.cuda_version())"
# on a GPU machine: True 12060 (or 12080 / 12090). If the second number is -1, redo B4.

python 03_sdchemgnn/test_cuda.py                 # GPU only: ALL GPU CHECKS PASSED

cd 03_sdchemgnn                                  # offline checks, no GPU needed
python test_spin_decoration.py                   # ALL 54 SEED/DECORATION CHECKS PASSED
python test_sdchemgnn_model.py                   # ALL 26 MODEL CHECKS PASSED
python test_training_step.py                     # ALL 34 TRAIN-STEP CHECKS PASSED
cd ../04_magsieve
python test_proposer_offline.py                  # 38 passed, 0 failed, 0 skipped
python test_selector_offline.py                  # 17 passed, 0 failed
cd ..
```

## Usage

Run everything from a `work/` directory inside the repository; the programs
use paths relative to it.

### Step 1: set up the working directory

```bash
mkdir -p work && cd work
cp ../02_data_curation/label_manifest.csv ../02_data_curation/collapse_suspects.csv .
```

### Step 2: download data from the Materials Project

Get an API key at https://next-gen.materialsproject.org/api.

```bash
export MP_API_KEY=...
python ../01_mp_retrieval/fetch_mp_structures.py --api-key "$MP_API_KEY" --output ./dataset_mp/mp_dataset.jsonl --num-workers 16
python ../01_mp_retrieval/fetch_mp_magnetic_ordering.py --jsonl ./dataset_mp/mp_dataset.jsonl --manifest ./label_manifest.csv --out ./ordering_sidetable.csv
python ../01_mp_retrieval/fetch_mp_site_moments.py --jsonl ./dataset_mp/mp_dataset.jsonl --out ./magmom_sidetable.jsonl
```

Add `--limit N` to try a few structures first. Interrupted fetches resume
where they stopped. The paper used a 2026 snapshot of the Materials
Project, so a fresh fetch will differ slightly.

### Step 3: build the dataset

```bash
python ../02_data_curation/build_crystal_graph_dataset.py --jsonl ./dataset_mp/mp_dataset.jsonl --root ./dataset_mp_magnetic --cutoff 7.0 --node-features cgcnn --num-workers 16
```

This builds periodic crystal graphs (7 Å cutoff) for PyTorch Geometric.
Use `label_manifest.csv` as shipped: it is the paper's curated,
manually reviewed magnetization labels.

### Step 4: train SD-ChemGNN (optional)

Skip this to use the paper's weights in `03_sdchemgnn/trained_model/`.

```bash
export TRAIN_TF32=1
CUDA_VISIBLE_DEVICES=0 python ../03_sdchemgnn/train_sdchemgnn.py
```

Results go to `./Results/ChemGNN_magneticBS_<date>_<n>/` (`best_model.pt`
and test predictions). Hyper-parameters are in `sdchemgnn_config.py`. To
resume a run, set `TRAIN_RUN_DIR` to its folder; to stop cleanly before a
time limit (e.g. on SLURM), set `TRAIN_SESSION_SECONDS`.

The shipped weights (205,349 parameters) give an undecorated test MAE of
0.543 μB per formula unit (0.312 label-decorated) on the 15,377-structure
test set.

### Step 5: run MagSieve

```bash
# 5a. score every proposed spin configuration with the trained model
python ../04_magsieve/score_configurations_with_sdchemgnn.py --run ../03_sdchemgnn/trained_model --out ./magsieve_features --parts 0d --splits train,val,test --k 8 --cache-workers 0

# 5b. fit the ordering router
python ../04_magsieve/ordering_router.py --dir ./magsieve_features --features base,emb --model gbm --class-weight rare --budget-weight 0 --out ./magsieve_features/router_test_proba.txt

# 5c. apply the selector and score the top-1,000 shortlist
python ../04_magsieve/screen_shortlist.py --dir ./magsieve_features --ids ./magsieve_features/phase0_test_ids.csv --proba ./magsieve_features/router_test_proba.txt --abstain 0 --fallback physical
```

Step 5a needs a lot of memory on the train split (the paper used 240 GB);
add `--max-rows N` to subsample it. With the shipped weights, the top 1,000
holds 22 false positives before screening and 18 after MagSieve.

### Step 6 (optional): Table 2 comparator

A random forest on composition descriptors, using the tables from Step 5a:

```bash
python ../03_sdchemgnn/composition_forest_comparator.py --train-meta ./magsieve_features/phase0d_train_meta.csv --test-meta ./magsieve_features/phase0d_test_meta.csv --sidetable ./ordering_sidetable.csv
```

It prints the forest's row of Table 2 (MAE 0.620) next to SD-ChemGNN.

## What is not included

- The Materials Project data (Step 2 downloads it with an API key).
- The SLURM scripts the paper's runs were submitted with.

## Third-party data

- `02_data_curation/atom_init.json` is the element embedding of CGCNN
  (T. Xie and J. C. Grossman, Phys. Rev. Lett. 120, 145301, 2018;
  https://github.com/txie-93/cgcnn), MIT License, Copyright (c) 2018 Tian Xie.
- `label_manifest.csv` and `collapse_suspects.csv` contain values derived
  from the Materials Project (A. Jain et al., APL Materials 1, 011002,
  2013), released under CC BY 4.0. Cite the Materials Project and the
  database version in any work that uses them.

## Provenance

`PROVENANCE.tsv` maps every file to its original name and md5. The programs
are the originals with imports and file names updated, a short path setup
added to files that import across folders, `sdchemgnn_config.py` set to the
released weights' seed and selection mode, and comments rewritten into the
paper's vocabulary. `screen_shortlist.py` no longer prints an internal
pre-paper baseline. `composition_forest_comparator.py` follows the paper's
protocol (one forest fitted on train, scored on test) and reproduces its
numbers to four decimals. `03_sdchemgnn/test_cuda.py` is a new GPU
installation check.
