"""End-to-end self-test on SYNTHETIC data (configs/smoke.yaml).

Runs every stage of the v2 chain with tiny models and a few steps each:
synthetic raw -> preprocess -> audit -> AE train / eval -> encode
(+ D4) -> DiT / struct / attr train -> 4 sampling chains -> pack-model
-> generate -> evaluate -> 3D page.  Run it first on a new machine (CPU is fine).  It checks the wiring;
it says nothing about physics quality.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def run(c, keep=False, verbose=True):
    """Run the whole chain on synthetic data with tiny models; raises on the first failure."""
    import torch

    from .data import preprocess
    from .evaluate import evaluate, evaluate_ae
    from .export3d import export
    from .field_ae import encode_cache
    from .generate import generate as generate_showers
    from .loader import case_dir, meta_for
    from .pack import pack_model
    from .sample import generate
    from .synth import write_dataset
    from .train import train

    say = print if verbose else (lambda *a, **k: None)
    if not c.get("synthetic"):
        # selftest deletes and rewrites its raw / processed / output paths: never on real data
        raise SystemExit("selftest needs a synthetic config (configs/smoke.yaml has `synthetic: true`)")
    raw, processed, out = (Path(c["paths"][k]) for k in ("raw", "processed", "output"))
    if not keep:
        for p in (raw, processed, out):
            if p.exists():
                shutil.rmtree(p)
    torch.manual_seed(0)
    n = int(c["data"]["event_stop"]) - int(c["data"]["event_start"])
    write_dataset(raw, c["data"]["source_pattern"], n, c["data"]["ranges"], seed=int(c["seed"]), max_records=4000)
    say(f"[1/9] synthetic raw data: {n} events (SYNTHETIC, not CORSIKA)")
    meta = preprocess(c)
    say(f"[2/9] preprocess: splits { {k: len(v) for k, v in meta['split'].items()} }")
    from .audit import audit
    audit(c, n_events=int(c["audit"]["events"]), k=int(c["audit"]["k"]))
    say("[3/9] audit")

    case = "selftest"
    info = {"ae": train(c, "ae", case, allow_cpu=True)}
    evaluate_ae(c, case, n_events=2, checkpoint="last", allow_cpu=True)
    say(f"[4/9] AE trained ({info['ae']['parameters']:,} parameters) and evaluated")
    lat = encode_cache(c, case_dir(c, case), "last", allow_cpu=True)
    assert lat["d4"], "smoke config should write D4 latents"
    say(f"[5/9] latent cache: {lat['latent_shape']} per event, splits {lat['splits']}")
    for kind in ("dit", "struct", "attr"):
        info[kind] = train(c, kind, case, allow_cpu=True)
        say(f"      trained {kind}: {info[kind]['parameters']:,} parameters")
    say("[6/9] DiT, struct, attr trained")

    meta = meta_for(c)
    k = int(c["eval"]["paired_events"])
    runs = []
    for chain, kw in (("sa_truth", {}), ("sa_recon", {}), ("sdedit", dict(sigma_start=1.0)), ("full", {})):
        name = chain if chain != "sdedit" else "sdedit_1"
        rows = generate(c, case, chain, name, n_samples=k, seed=1, checkpoint="last", allow_cpu=True,
                        verbose=False, **kw)
        ok = [r for r in rows if not r.get("failed")]
        say(f"      {name}: {len(ok)}/{len(rows)} samples, voxels {[r.get('n_active') for r in rows]}")
        if ok:
            runs.append(name)
    assert runs, "no chain produced a sample"
    model = case_dir(c, case) / "trained_model"
    if not model.exists():
        pack_model(c, case, model, checkpoint="last")
    files = generate_showers(model, 2, case_dir(c, case) / "showers", seed=1, verbose=False)
    say(f"      pack-model + generate: {len(files)}/2 showers from the packed model")
    say("[7/9] sampling chains ran")
    summary = evaluate(c, case, runs, name="selftest", verbose=False)
    assert summary["rungs"], "evaluation empty"
    say(f"[8/9] evaluation written ({len(summary['passfail'])} pass/fail rows)")
    events = [int(e) for e in meta["split"]["test"]][:2]
    page = export(c, case, [r for r in runs if r != "full"], "full" if "full" in runs else None,
                  events=events, n_gallery=2)
    say(f"[9/9] 3D page: {page}")
    return dict(info=info, runs=runs, page=page)
