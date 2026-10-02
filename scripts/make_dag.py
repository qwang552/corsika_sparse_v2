#!/usr/bin/env python3
"""Write the full HTCondor DAG of the pipeline (baseline, AE, encode, models, samples, evaluation).

Standard library only, so it runs on the submit node without the environment:

    cd /scratch/qwang/corsika_sparse_v2
    python3 /data/user/qwang/corsika_sparse_v2/scripts/make_dag.py --out full.dag
    condor_submit_dag full.dag

Every node gets its own .sub file under <dag dir>/<dag name>_nodes/ with
absolute paths only; logs go to <dag dir>/<dag name>_logs/ (scratch, never
/data).  Nodes whose result already exists on disk are written with DONE, so
re-running this script with a new --out after a partial run only schedules
what is missing.

Edges:
    AE -> AE_EVAL, ENCODE
    ENCODE -> DIT, STRUCT, ATTR
    STRUCT, ATTR -> S_SA_TRUTH, S_SA_RECON (+ _T2OFF)
    DIT, STRUCT, ATTR -> S_SDEDIT_<sigma>, S_FULL
    all samples, V1 -> EVALUATE -> NB_SAMPLE
    all samples, V1 -> EXPORT3D
    AE_EVAL, DIT, STRUCT, ATTR -> NB_VIZ
    V1 (old v1 package) -> EVAL_BASELINE (R0 + V1)

--upto stops the DAG after a stage (ae: AE + AE_EVAL; encode; models: the
three trainings; samples; all), e.g. to check the AE acceptance by hand
before spending GPU days on the rest.  V1 and EVAL_BASELINE are always
included unless --no-v1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQ_GPU = "(GPUs_Capability >= 6.0) && (GPUs_Capability < 9.0)"
STAGES = ("ae", "encode", "models", "samples", "all")


def sub_text(a, name, args, cpus, mem, disk, gpu, queue=1, executable=None, env=None):
    """Text of one node's .sub file (absolute paths, logs under the DAG's log directory)."""
    exe = executable or f"{a.project}/run_pipeline.sh"
    env = env or f"SPARSE_PROJECT={a.project} SPARSE_ENV={a.env} SPARSE_CONFIG={a.config}"
    logs = a.logs
    lines = [
        f"# node {name} of {a.out.name} (written by scripts/make_dag.py)",
        "universe = vanilla",
        f"executable = {exe}",
        f'arguments = "{args}"',
        f'environment = "{env}"',
        "getenv = True",
        f"request_cpus = {cpus}",
        f"request_memory = {mem}",
        f"request_disk = {disk}",
        f"request_gpus = {1 if gpu else 0}",
    ]
    if gpu:
        lines.append(f"requirements = {REQ_GPU}")
    lines += [
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        "transfer_executable = True",
        'transfer_output_files = ""',
        f"output = {logs}/{name}.$(Cluster).$(Process).out",
        f"error = {logs}/{name}.$(Cluster).$(Process).err",
        f"log = {logs}/{name}.log",
        f"queue {queue}",
    ]
    text = "\n".join(lines) + "\n"
    bad = [l for l in text.splitlines() if "#" in l and not l.startswith("#")]
    assert not bad, bad                     # the cluster rejects trailing comments
    return text


def n_files(path, pattern="sample_*.npz"):
    """Number of files matching a pattern in a directory (0 if it does not exist)."""
    p = Path(path)
    return len(list(p.glob(pattern))) if p.is_dir() else 0


def sigma_name(s):
    """Run name of the SDEdit run at sigma s, e.g. sdedit_1."""
    return f"sdedit_{s:g}"


def main(argv=None):
    """Write every node file and the DAG; nodes with results on disk are marked DONE."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="full.dag", help="DAG file (its directory holds node files and logs)")
    ap.add_argument("--project", default="/data/user/qwang/corsika_sparse_v2")
    ap.add_argument("--env", default="/data/user/qwang/env/env2025/bin/activate")
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--output", default=None, help="must equal paths.output of --config (not read from it; default <project>/artifacts)")
    ap.add_argument("--case", default="v2a")
    ap.add_argument("--v1-project", default="/data/user/qwang/corsika_sparse_v1")
    ap.add_argument("--no-v1", action="store_true", help="skip the V1 rung")
    ap.add_argument("--upto", choices=STAGES, default="all")
    ap.add_argument("--paired-n", type=int, default=128, help="test events of the paired runs")
    ap.add_argument("--full-n", type=int, default=1000, help="unconditional samples")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 5])
    ap.add_argument("--paired-shards", type=int, default=4)
    ap.add_argument("--sdedit-shards", type=int, default=2)
    ap.add_argument("--full-shards", type=int, default=10)
    ap.add_argument("--eval-name", default="main")
    ap.add_argument("--ae-eval-n", type=int, default=256)
    ap.add_argument("--retry", type=int, default=3, help="RETRY of the AE, ENCODE, DIT, STRUCT and ATTR nodes (trainings resume from <kind>_last.pt)")
    a = ap.parse_args(argv)

    a.out = Path(a.out).resolve()
    if str(a.out).startswith("/data/"):
        sys.exit("write the DAG under /scratch (condor logs must not be on /data)")
    if a.out.exists():
        sys.exit(f"{a.out} exists; pick another --out (DAG files are not overwritten)")
    nodes_dir = a.out.parent / f"{a.out.stem}_nodes"
    a.logs = a.out.parent / f"{a.out.stem}_logs"
    for d in (nodes_dir, a.logs):
        if d.exists() and any(d.iterdir()):
            sys.exit(f"{d} exists and is not empty; pick another --out")
    nodes_dir.mkdir(parents=True, exist_ok=True)
    a.logs.mkdir(parents=True, exist_ok=True)
    output = Path(a.output or f"{a.project}/artifacts")
    case = output / a.case
    stage = STAGES.index(a.upto)

    jobs, edges, retries = [], [], {}

    def node(name, done, *sub_args, **sub_kw):
        """Write one node's .sub file and register the node."""
        path = nodes_dir / f"{name}.sub"
        path.write_text(sub_text(a, name, *sub_args, **sub_kw))
        jobs.append((name, path, bool(done)))
        return name

    def after(parents, children):
        """Register PARENT -> CHILD edges."""
        parents = [p for p in parents if p]
        children = [c for c in children if c]
        if parents and children:
            edges.append((parents, children))

    C = f"--case {a.case}"
    # V1 baseline -----------------------------------------------------------
    v1 = base = None
    v1_dir = output / "v1_samples" / f"truth{a.paired_n}"
    if not a.no_v1:
        v1 = node("V1", n_files(v1_dir) >= a.paired_n,
                  f"--config configs/struct_opt.yaml --case main --checkpoint best --n-samples {a.paired_n} "
                  f"--macro-source truth --seed 0 --out-name {v1_dir} --shard $(Process) --num-shards {a.paired_shards}",
                  2, "16GB", "4GB", True, queue=a.paired_shards,
                  executable=f"{a.v1_project}/scripts/run_sample_opt.sh",
                  env=f"SPARSE_PROJECT={a.v1_project} SPARSE_ENV={a.env}")
        retries[v1] = 2
        base = node("EVAL_BASELINE", (case / "eval" / "baseline" / "summary.json").exists(),
                    f"evaluate {C} --v1-dir {v1_dir} --name baseline", 4, "32GB", "4GB", False)
        after([v1], [base])
    # AE ------------------------------------------------------------------
    ae = node("AE", (case / "ae_done.json").exists(), f"train --kind ae {C}", 4, "16GB", "4GB", True)
    ae_eval = node("AE_EVAL", (case / "eval" / "ae_best" / "summary.json").exists(),
                   f"ae-eval {C} --checkpoint best --n-samples {a.ae_eval_n} --name ae_best", 2, "16GB", "4GB", True)
    after([ae], [ae_eval])
    retries[ae] = a.retry
    enc = dit = struct = attr = None
    if stage >= 1:
        enc = node("ENCODE", (case / "latents" / "done.json").exists(), f"encode {C} --checkpoint best",
                   4, "16GB", "8GB", True)
        after([ae], [enc])
        retries[enc] = a.retry
    # DiT, struct, attr ---------------------------------------------------
    if stage >= 2:
        dit = node("DIT", (case / "dit_done.json").exists(), f"train --kind dit {C}", 4, "16GB", "4GB", True)
        struct = node("STRUCT", (case / "struct_done.json").exists(), f"train --kind struct {C}",
                      4, "16GB", "4GB", True)
        attr = node("ATTR", (case / "attr_done.json").exists(), f"train --kind attr {C}", 4, "16GB", "4GB", True)
        after([enc], [dit, struct, attr])
        for n in (dit, struct, attr):
            retries[n] = a.retry
    # samples ---------------------------------------------------------------
    sample_nodes, paired_runs = [], []
    if stage >= 3:
        def sample(name, run, chain, n, shards, extra, parents):
            done = n_files(case / "samples" / run) >= n
            s = node(name, done, f"sample {C} --chain {chain} --run {run} --n-samples {n} --seed 0 --checkpoint best "
                                 f"--shard $(Process) --num-shards {shards}{extra}", 2, "16GB", "4GB", True,
                     queue=shards)
            retries[s] = 2
            after(parents, [s])
            sample_nodes.append(s)
            return run

        for chain in ("sa_truth", "sa_recon"):
            paired_runs.append(sample(f"S_{chain.upper()}", chain, chain, a.paired_n, a.paired_shards, "",
                                      [struct, attr]))
            sample(f"S_{chain.upper()}_T2OFF", f"{chain}_t2off", chain, a.paired_n, a.paired_shards, " --t2 off",
                   [struct, attr])
        for s in a.sigmas:
            run = sigma_name(s)
            sample(f"S_SDEDIT_{str(s).replace('.', 'P')}", run, "sdedit", a.paired_n, a.sdedit_shards,
                   f" --sigma-start {s:g}", [dit, struct, attr])
            if s in (0.5, 2.0):             # these two also go on the 3D page
                paired_runs.append(run)
        sample("S_FULL", "full", "full", a.full_n, a.full_shards, "", [dit, struct, attr])
    # evaluation and pages --------------------------------------------------
    if stage >= 4:
        runs = ["sa_truth", "sa_recon", "sa_truth_t2off", "sa_recon_t2off"] + [sigma_name(s) for s in a.sigmas] + ["full"]
        v1_arg = "" if a.no_v1 else f" --v1-dir {v1_dir}"
        ev = node("EVALUATE", (case / "eval" / a.eval_name / "summary.json").exists(),
                  f"evaluate {C} --runs {' '.join(runs)} --name {a.eval_name}{v1_arg}", 4, "32GB", "4GB", False)
        after(sample_nodes + [v1], [ev])
        page = case / "viz" / f"chains_3d_{a.eval_name}.html"
        ex = node("EXPORT3D", page.exists(),
                  f"export3d {C} --runs {' '.join(paired_runs)} --full-run full --out {page}{v1_arg}",
                  2, "16GB", "2GB", False)
        after(sample_nodes + [v1], [ex])
        nb = node("NB_SAMPLE", False,
                  f"notebook notebooks/sample.ipynb --threads 4 --set DEVICE=cpu --set CONFIG={a.config} --set CASE={a.case} "
                  f"--set EVAL_NAME={a.eval_name}", 4, "24GB", "4GB", False)
        nbv = node("NB_VIZ", False, f"notebook notebooks/viz.ipynb --threads 4 --set DEVICE=cpu --set CONFIG={a.config} "
                   f"--set CASE={a.case}",
                   4, "24GB", "4GB", False)
        after([ev], [nb])
        after([ae_eval, dit, struct, attr], [nbv])

    lines = [f"# v2 DAG for case {a.case} (written by scripts/make_dag.py; stage up to '{a.upto}')",
             f"# condor_submit_dag {a.out}"]
    for name, path, done in jobs:
        lines.append(f"JOB {name} {path}" + (" DONE" if done else ""))
    for parents, children in edges:
        lines.append(f"PARENT {' '.join(parents)} CHILD {' '.join(children)}")
    for name, n in retries.items():
        if name:
            lines.append(f"RETRY {name} {n}")
    a.out.write_text("\n".join(lines) + "\n")
    todo = [n for n, _, d in jobs if not d]
    print(f"wrote {a.out} ({len(jobs)} nodes, {len(todo)} to run: {' '.join(todo) or 'none'})")
    print(f"node files: {nodes_dir}\nlogs: {a.logs}\nsubmit:  condor_submit_dag {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
