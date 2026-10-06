"""Command-line entry point:  python -m sparseshower.cli <command> [options]

    selftest     synthetic end-to-end run (CPU), checks that every step runs
    synth        write synthetic raw events (for experiments without cluster data)
    preprocess   build the voxel cache (not needed when an existing cache is reused)
    audit        data audit on the cache (box counts, linearity, ...)
    memtest      GPU memory / speed of one step per kind on the largest event
    train        --kind ae|dit|struct|attr
    ae-eval      AE acceptance on val events
    encode       write the latent cache of a trained AE (+ D4 variants if encode.d4 is true)
    sample       --chain sa_truth|sa_recon|sdedit|full --run NAME
    evaluate     --runs NAME [NAME ...] [--v1-dir DIR] --name EVAL
    export3d     self-contained 3D comparison page
    pack-model   copy a trained case into a model folder for `python -m sparseshower.generate`
    info         resolved grids, token counts, paths
"""
from __future__ import annotations

import argparse
import json

from .common import config

COMMANDS = ("selftest", "synth", "preprocess", "audit", "memtest", "train", "ae-eval",
            "encode", "sample", "evaluate", "export3d", "pack-model", "info")


def build_parser():
    """Command-line options of all commands."""
    ap = argparse.ArgumentParser(prog="sparseshower", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--case", default="v2a", help="output sub-directory of this experiment")
    ap.add_argument("--kind", choices=("ae", "dit", "struct", "attr"))
    ap.add_argument("--kinds", nargs="+", default=["ae", "dit", "struct", "attr"], help="memtest kinds")
    ap.add_argument("--n-events", type=int, help="train: training events (unset = all)")
    ap.add_argument("--steps", type=int, help="train: override the step budget")
    ap.add_argument("--checkpoint", choices=("best", "last"), default="best")
    ap.add_argument("--chain", choices=("sa_truth", "sa_recon", "sdedit", "full"))
    ap.add_argument("--run", help="sample: run name under <case>/samples/")
    ap.add_argument("--runs", nargs="+", help="evaluate / export3d: sample run names")
    ap.add_argument("--full-run", help="export3d: the unconditional run for the gallery")
    ap.add_argument("--v1-dir", help="evaluate / export3d: directory of v1 baseline samples (jobs/v1_samples.sub)")
    ap.add_argument("--name", help="evaluate / ae-eval: output name under <case>/eval/")
    ap.add_argument("--n-samples", type=int, default=8,
                    help="sample: number of samples (paired chains: the first N test events); "
                         "ae-eval: number of val events")
    ap.add_argument("--events", type=int, nargs="+", help="sample / export3d: explicit test event ids")
    ap.add_argument("--sigma-start", type=float, help="sample --chain sdedit: noise level sigma_s added to the test latent")
    ap.add_argument("--t2", choices=("on", "off"), help="sample: override sample.t2")
    ap.add_argument("--truth-n", type=int, help="evaluate: use only the first N test events as truth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, help="preprocess / encode: only the first N events")
    ap.add_argument("--finalize", action="store_true", help="preprocess: after all shards, merge their reports and write metadata.json")
    ap.add_argument("--out", help="export3d: output html path (default <case>/viz/chains_3d.html); "
                                   "pack-model: model folder (default trained_model)")
    ap.add_argument("--n-gallery", type=int, default=12, help="export3d: unconditional samples in the gallery")
    ap.add_argument("--keep", action="store_true", help="selftest: keep the previous smoke run")
    ap.add_argument("--allow-cpu", action="store_true")
    return ap


def main(argv=None):
    """Parse the command line and dispatch to the module that implements the command."""
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.command == "selftest" and a.config == ap.get_default("config"):
        a.config = "configs/smoke.yaml"          # selftest wipes its own paths: synthetic config only
    c = config(a.config)

    if a.command == "selftest":
        from .selftest import run

        run(c, keep=a.keep)
        print("SELFTEST PASS")
        return 0

    if a.command == "synth":
        from .synth import write_dataset

        n = int(c["data"]["event_stop"]) - int(c["data"]["event_start"])
        files = write_dataset(c["paths"]["raw"], c["data"]["source_pattern"], n, c["data"]["ranges"], seed=a.seed)
        print(f"wrote {len(files)} synthetic events under {c['paths']['raw']}")
        return 0

    if a.command == "preprocess":
        from pathlib import Path

        from .data import finalize_cache, preprocess

        done = Path(c["paths"]["processed"]) / "metadata.json"
        if done.exists():
            raise SystemExit(f"{done} exists: this cache is finished and read only. "
                             "Point paths.processed at a new directory to build another cache.")

        meta = finalize_cache(c) if a.finalize else preprocess(c, limit=a.limit, shard=a.shard if a.num_shards > 1 else None,
                                                               num_shards=a.num_shards)
        print(json.dumps({k: v for k, v in meta.items() if k not in ("stats", "split")}, indent=2))
        return 0

    if a.command == "audit":
        from .audit import audit, format_audit

        res = audit(c, n_events=int(c["audit"]["events"]), k=int(c["audit"]["k"]))
        print(format_audit(res))
        return 0

    if a.command == "memtest":
        from .memtest import memtest

        print(json.dumps(memtest(c, tuple(a.kinds), allow_cpu=a.allow_cpu), indent=2))
        return 0

    if a.command == "train":
        from .train import train

        if not a.kind:
            ap.error("train needs --kind ae|dit|struct|attr")
        print(json.dumps(train(c, a.kind, a.case, a.n_events, a.steps, a.allow_cpu), indent=2))
        return 0

    if a.command == "ae-eval":
        from .evaluate import evaluate_ae

        evaluate_ae(c, a.case, n_events=a.n_samples, checkpoint=a.checkpoint, name=a.name,
                    allow_cpu=a.allow_cpu)
        return 0

    if a.command == "encode":
        from .field_ae import encode_cache
        from .loader import case_dir

        print(json.dumps(encode_cache(c, case_dir(c, a.case), a.checkpoint, a.allow_cpu, limit=a.limit), indent=2))
        return 0

    if a.command == "sample":
        from .sample import generate

        if not (a.chain and a.run):
            ap.error("sample needs --chain and --run")
        t2 = None if a.t2 is None else a.t2 == "on"
        rows = generate(c, a.case, a.chain, a.run, a.n_samples, a.seed, a.shard, a.num_shards, a.sigma_start,
                        t2, a.checkpoint, a.allow_cpu, a.events)
        ok = [r for r in rows if not r.get("failed")]
        print(json.dumps(dict(written=len(ok), failed=len(rows) - len(ok)), indent=2))
        return 0

    if a.command == "evaluate":
        from .evaluate import evaluate

        runs = list(a.runs or [])
        if a.v1_dir:
            runs.append("v1:" + a.v1_dir)
        if not runs:
            ap.error("evaluate needs --runs and/or --v1-dir")
        evaluate(c, a.case, runs, a.name, a.truth_n)
        return 0

    if a.command == "export3d":
        from .export3d import export

        print(export(c, a.case, a.runs or [], a.full_run, a.v1_dir, a.events, n_gallery=a.n_gallery, out=a.out))
        return 0

    if a.command == "pack-model":
        from .pack import pack_model

        print(json.dumps(pack_model(c, a.case, a.out or "trained_model", a.checkpoint), indent=2))
        return 0

    if a.command == "info":
        from .dit import describe
        from .samples import level_grids

        ranges = c["data"]["ranges"]
        print(json.dumps(dict(level_grids=level_grids(int(c["data"]["grid"]), int(c["field"]["base_grid"])),
                              voxel_cm=[round(100 * (r[1] - r[0]) / int(c["data"]["grid"]), 3) for r in ranges],
                              base_cell_cm=[round(100 * (r[1] - r[0]) / int(c["field"]["base_grid"]), 3) for r in ranges],
                              dit=describe(c), paths=c["paths"]), indent=2))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
