# corsika_sparse_v2: run order

Generative model for the Cherenkov photons (`NPhotons`) of CORSIKA 8 in-ice
showers on a sparse 192³ voxel grid (4 m box, 2.08 cm voxels, records with
raw time < 12 ns). Unconditional: all current showers are 1 TeV.

```
noise -> LatentDiT -> latent 8x12^3 -> FieldAE decoder -> 48^3 field
      -> struct (48 -> 96 -> 192) -> active voxels -> attr -> photons + offset per voxel
```

Four models, trained separately: FieldAE (`field_ae.py`), LatentDiT (`dit.py`),
struct (`structure.py`, `struct_data.py`), attr (`attr.py`). Every `.py` file
starts with a short description of what it does. `README.md` explains the
design and the evaluation; this file lists the commands in the order they are run.

Steps:
[0 Paths](#0-paths-to-change-on-your-machine) ·
[1 Install](#1-install-and-test) ·
[2 How to run](#2-how-to-run-a-step) ·
[3 Data](#3-data-build-the-voxel-cache) ·
[4 Checks](#4-checks-before-training) ·
[5 FieldAE](#5-fieldae-model-1) ·
[6 DiT, struct, attr](#6-dit-struct-attr-models-2-4) ·
[7 Sample and evaluate](#7-sample-and-evaluate) ·
[8 Figures](#8-figures) ·
[Notes](#notes)

## 0. Paths to change on your machine

The paths in this repository are the ones on the IceCube NPX cluster.
Change them before running anything.

| File | Setting | What it is |
|---|---|---|
| `configs/data.yaml` | `paths.raw` | raw CORSIKA files (input of section 3) |
| | `paths.processed` | 192³ voxel cache, written by section 3 |
| | `paths.output` | where checkpoints, samples and plots are written |
| | `viz.events` | test event ids for the per-event figures (export3d, sample.ipynb); see below |
| `configs/preprocess_192.yaml` | `paths.raw`, `paths.processed` | the same values as in `data.yaml` |
| `jobs/*.sub` | `project = ...` | this code directory |
| | `env_activate = ...` | your python environment (`.../bin/activate`) |
| `jobs/v1_samples.sub` | `v1_project = ...` | the old v1 package (optional baseline only) |
| `jobs/preprocess.dag` | two `JOB` lines | absolute paths of the two .sub files |
| `run_pipeline.sh`, `scripts/chain_page.sh` | default project / env | used only if the job does not set them |
| `scripts/make_dag.py` | `--project`, `--env`, `--v1-project`, `--output` | pass them on the command line |

`viz.events` are ids of my cache (so is `EVENT=28` in section 8). With your
own cache, replace them with ids from your test split:

```bash
python -c "import json,sys; print(json.load(open(sys.argv[1]+'/metadata.json'))['split']['test'][:6])" <paths.processed>
```

Replace the paths in all job files at once (set the two new paths first):

```bash
NEW_CODE=/path/to/corsika_sparse_v2
NEW_ENV=/path/to/env/bin/activate
sed -i "s#/data/user/qwang/corsika_sparse_v2#$NEW_CODE#g; s#/data/user/qwang/env/env2025/bin/activate#$NEW_ENV#g" \
    jobs/*.sub jobs/preprocess.dag run_pipeline.sh scripts/chain_page.sh
```

Find anything left over:

```bash
grep -rn "/data/user/qwang\|/scratch/qwang" configs jobs scripts notebooks run_pipeline.sh
```

Data: the voxel cache is not in this repository. Build it from the raw files
with section 3 before training.

Other cluster-specific settings are the GPU requirement
(`GPUs_Capability >= 6.0 && < 9.0` in `jobs/*.sub` and `REQ_GPU` in
`scripts/make_dag.py`) and the memory requests. Submit condor
jobs from a scratch directory, not from the data disk (condor writes its logs
in the submit directory).

## 1. Install and test

Any machine; CPU is enough.

```bash
# torch, numpy, pyyaml, pyarrow, scipy, matplotlib, pytest
pip install -r requirements.txt
# 54 unit tests
python -m pytest tests -q
# whole chain on synthetic data; ends with "SELFTEST PASS"
python -m sparseshower.cli selftest
```

## 2. How to run a step

Without condor (local machine), from the code directory:

```bash
python -m sparseshower.cli <command> --config <config> --case v2a [options]
# config: configs/v2.yaml for train --kind ae, configs/v2_cal85.yaml for the later steps
# add --allow-cpu to run a GPU step on CPU (slow)
```

With condor (cluster), from a scratch directory:

```bash
P=/path/to/corsika_sparse_v2          # your code directory
CFG=configs/v2_cal85.yaml             # AE occupancy threshold 0.85
condor_submit VAR=value ... $P/jobs/<step>.sub
```

The variables of each `.sub` file are listed at its top. Most `.sub` files run
`run_pipeline.sh`, which activates the environment, goes to the code directory
and runs `python -m sparseshower.cli <arguments line> --config <CONFIG>`; that
is also the local command. The notebook jobs run `scripts/run_notebook.py`;
`chain_page.sub`, `chain_steps.sub` and `active_vs_grid.sub` call their own
scripts. Notebooks do not read `CONFIG` from the job; pass it with
`--set CONFIG=...` in `EXTRA`.

Outputs go to `<paths.output>/` and `<paths.output>/v2a/` (`v2a` is the case
name). Nothing is overwritten: a resubmitted training job resumes from
`<kind>_last.pt`, a resubmitted sample job skips finished samples (a RUN made
with other settings is refused), and evaluation names must be new.

## 3. Data: build the voxel cache

Run this once before training; it writes the cache to `paths.processed`.

```bash
# 16 CPU shards, then finalize (split + statistics)
condor_submit_dag $P/jobs/preprocess.dag
# local: python -m sparseshower.cli preprocess --config configs/preprocess_192.yaml

# describe the cache and check that it matches the config
# 20260922 = split seed of the cache (preprocess_192.yaml); the seed in data.yaml is the training seed
python scripts/check_cache.py --config configs/data.yaml --seed 20260922
# also re-process 3 raw events and compare
python scripts/check_cache.py --config configs/data.yaml --seed 20260922 --rebuild 3
```

What `preprocess` does (`sparseshower/data.py`):

| | |
|---|---|
| input | `<paths.raw>/shower_group_<eid>/cherenkov/light.parquet`, eid 0..15999; columns `NPhotons`, `time`, `posX`, `posY`, `posZ` (`dirX`, `dirY`, `dirZ` if present) |
| cuts | finite values, `NPhotons > 0`, raw time < 12 ns, x, y in [-2, 2] m, z in [46, 50] m |
| voxels | 192³ grid; per active voxel: `ijk` voxel index, `q` sum of `NPhotons`, `off` photon-weighted position in the voxel (-0.5..0.5), `t` mean time, `m` mean direction, `cnt` number of records |
| output | `<paths.processed>/events/event_XXXXX.npz`, one file per shower (empty voxels not stored) |
| | `<paths.processed>/metadata.json`: train / val / test split (80/10/10, seed 20260922) and statistics |
| | `<paths.processed>/signature.json`: hash of the settings; a different setting is refused |

The models use `ijk`, `q` and `off`. `t` and `m` are stored but not modelled yet.
My models were trained on a cache built by the earlier v1 code with the same
settings and split seed; this step should give the same cache (`check_cache.py` compares).

## 4. Checks before training

```bash
# synthetic end-to-end run (wiring check only)
condor_submit $P/jobs/selftest.sub
# GPU memory per model on the largest event; all must say "fits: true"
condor_submit $P/jobs/memtest.sub
```

## 5. FieldAE (model 1)

```bash
# train the autoencoder 48^3 field <-> 8x12^3 latent (uses configs/v2.yaml)
condor_submit KIND=ae $P/jobs/train.sub
# check reconstruction on 256 val events -> v2a/eval/ae_best/passfail.md
condor_submit CONFIG=$CFG $P/jobs/ae_eval.sub
# encode every event -> v2a/latents/ (input of the other three models)
condor_submit CONFIG=$CFG $P/jobs/encode.sub
```

Optional threshold scan (which decoded 48³ cells count as occupied):

```bash
for t in 0.5 0.7 0.8 0.85 0.9 0.95; do
  f=$P/configs/thr_$t.yaml
  [ -e $f ] || printf 'extends: v2.yaml\nfield:\n  occ_threshold: %s\n' $t > $f
  condor_submit CONFIG=configs/thr_$t.yaml NAME=thr_$t $P/jobs/ae_eval.sub
done
condor_submit NB=notebooks/ae_check.ipynb EXTRA="--set CONFIG=$CFG --set THR=0.85" $P/jobs/notebook_cpu.sub
```

## 6. DiT, struct, attr (models 2-4)

The three trainings are independent and can run at the same time.

```bash
# generate latents from noise
condor_submit KIND=dit    CONFIG=$CFG $P/jobs/train.sub
# which child voxels are active, 48 -> 96 -> 192
condor_submit KIND=struct CONFIG=$CFG $P/jobs/train.sub
# photons and offset of every active voxel
condor_submit KIND=attr   CONFIG=$CFG $P/jobs/train.sub

# training progress
tail -n 3 <paths.output>/v2a/attr_history.jsonl
# training curves
condor_submit NB=notebooks/viz.ipynb EXTRA="--set CONFIG=$CFG" $P/jobs/notebook_cpu.sub
```

## 7. Sample and evaluate

Paired chains, one sample for each of the first N test events (N=128 by
default; needs AE, struct, attr):

```bash
# true 48^3 field -> struct -> attr
condor_submit CONFIG=$CFG CHAIN=sa_truth RUN=sa_truth $P/jobs/sample.sub
# AE-reconstructed field -> struct -> attr
condor_submit CONFIG=$CFG CHAIN=sa_recon RUN=sa_recon $P/jobs/sample.sub
# same without T2 (the per-48^3-cell photon rescaling)
condor_submit CONFIG=$CFG CHAIN=sa_recon RUN=sa_recon_t2off EXTRA="--t2 off" $P/jobs/sample.sub
```

SDEdit and unconditional (needs all four models):

```bash
# test latent noised to sigma s, then DiT
for s in 0.25 0.5 1 2 5; do
  condor_submit CONFIG=$CFG CHAIN=sdedit RUN=sdedit_$s EXTRA="--sigma-start $s" NSHARD=2 $P/jobs/sample.sub
done
# pure noise -> new showers
condor_submit CONFIG=$CFG CHAIN=full RUN=full N=1000 NSHARD=10 $P/jobs/sample.sub
```

Evaluate (per-event errors, W1/R0, C2ST, pass/fail table; the metrics are
explained in `README.md`, section 5):

```bash
condor_submit CONFIG=$CFG NAME=main RUNARGS="--runs sa_truth sa_recon sa_recon_t2off sdedit_0.5 sdedit_1 full" $P/jobs/evaluate.sub
cat <paths.output>/v2a/eval/main/passfail.md
```

Optional baseline from the old v1 model: run `condor_submit $P/jobs/v1_samples.sub`,
then evaluate with `RUNARGS="--v1-dir $P/artifacts/v1_samples/truth128"`.

## 8. Figures

```bash
# interactive 3D page -> v2a/viz/chain_page.html
condor_submit CONFIG=$CFG RUNS="sa_truth sa_recon sdedit_1" FULL=full $P/jobs/chain_page.sub
# one sample step by step -> v2a/viz/chain_steps/
condor_submit CONFIG=$CFG EVENT=28 RUN=sdedit_1 $P/jobs/chain_steps.sub
# sample-side figures
condor_submit NB=notebooks/sample.ipynb EXTRA="--set CONFIG=$CFG" $P/jobs/notebook_cpu.sub
# full report -> v2a/samples/report/
condor_submit -a request_cpus=8 -a request_memory=32GB NB=notebooks/report.ipynb \
    EXTRA="--set CONFIG=$CFG --set WORKERS=8" $P/jobs/notebook_gpu.sub
# active voxels vs grid size (raw data)
condor_submit $P/jobs/active_vs_grid.sub
```

## Notes

- **One DAG for steps 5-7.** `make_dag.py` writes steps 5-7, the V1 baseline,
  the export3d page, `sample.ipynb` and `viz.ipynb` as one DAG (no threshold
  scan, chain_page, chain_steps or report). `--output` must equal `paths.output`
  of the config; add `--no-v1` if you do not have the v1 package. All nodes,
  the AE training included, use `--config`:
  ```bash
  python3 $P/scripts/make_dag.py --out full.dag --project $P --env <env> --config $CFG --output <paths.output>
  condor_submit_dag full.dag
  ```
- **No GPU free.** For `train.sub` and `sample.sub`, add
  `-a request_gpus=0 -a requirements=True -a request_cpus=8` and put `--allow-cpu`
  in `EXTRA` (sdedit: `EXTRA="--sigma-start 1 --allow-cpu"`). Never run GPU and
  CPU jobs on the same RUN at the same time.
- **.sub files.** No trailing `#` comments, and never name a variable `args` / `arguments`.
- **GPU size.** Sized for an 8 GB GTX 1080. Newer cards (compute capability 7.0
  and up) switch on mixed precision, and cards with 16 GB or more run without
  activation checkpointing, so they are faster.
- **Status.** Large-scale shower shape is reproduced; fine structure and the
  unconditional chain are not yet at truth level.
