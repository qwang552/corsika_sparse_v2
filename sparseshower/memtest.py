"""GPU memory and speed of one training step per kind, on the largest event.

Run this on the GPU type the jobs will actually get (e.g. GTX 1080) before any
long job.  Needs no trained model: struct / attr get the truth 48^3 field in
the decoded-field format (same tensor shapes as the reconstruction), the DiT gets
random latents.  Writes <output>/memtest/<gpu>_<date>.json.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .common import read_json, write_json
from .data import Codec, load_event
from .loader import meta_for


def largest_event(c, meta, scan=400):
    """Train event with the most active voxels (from report.json, else among the first `scan`)."""
    root = Path(c["paths"]["processed"])
    train = set(int(e) for e in meta["split"]["train"])
    rep = root / "report.json"
    if rep.exists():
        rows = [r for r in read_json(rep) if r.get("status") == "ok" and int(r["event_id"]) in train]
        if rows:
            best = max(rows, key=lambda r: int(r.get("n_active", 0)))
            return int(best["event_id"]), int(best["n_active"])
    best, n_best = None, -1
    for e in list(train)[:scan]:
        with np.load(root / "events" / f"event_{e:05d}.npz") as d:
            n = len(d["q"])
        if n > n_best:
            best, n_best = e, n
    return int(best), int(n_best)


def memtest(c, kinds=("ae", "dit", "struct", "attr"), steps=3, allow_cpu=False, write=True):
    """Time a few training steps of each kind on the largest event; report peak memory and fit."""
    import torch

    from .common import device_for, resolve_amp, amp_context
    from .fields import coarse_field, load_field_stats, stats_events, truth_field3
    from .train import resolve_checkpoint

    device = device_for(allow_cpu)
    meta = meta_for(c)
    eid, n_act = largest_event(c, meta)
    ev = load_event(c["paths"]["processed"], eid)
    stats = load_field_stats(c, create_from=stats_events(c, meta))
    use_ckpt = resolve_checkpoint(c["model"].get("checkpoint", "auto"),
                                  float(c["model"].get("checkpoint_below_gib", 16)), device)
    c["model"]["checkpoint"] = use_ckpt
    amp = resolve_amp(c["train"]["amp"], device)
    f2, counts, sums = coarse_field(ev["ijk"], ev["q"], int(c["data"]["grid"]), int(c["field"]["base_grid"]),
                                    float(c["data"]["q_eps"]), stats)
    F_t = torch.as_tensor(truth_field3(f2), device=device)
    rows = {}
    for kind in kinds:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        rng = np.random.default_rng(0)
        if kind == "ae":
            from .field_ae import ProfileIndex, ae_loss, build_ae

            net = build_ae(c, device)
            net.use_ckpt = use_ckpt
            pidx = ProfileIndex(c, device)
            b = int(c["ae"]["batch_size"])
            field = torch.as_tensor(np.repeat(f2[None], b, 0), device=device)
            s = torch.as_tensor(np.repeat(sums[None].astype(np.float32), b, 0), device=device)

            def step():
                out, mu, lv = net(field)
                return ae_loss(out, mu, lv, field, s, c["ae"], stats, pidx)[0]
        elif kind == "dit":
            from .diffusion import edm_loss_dense, sample_sigmas
            from .dit import build_dit, dit_denoiser

            net = build_dit(c, device)
            net.use_ckpt = use_ckpt
            z = torch.randn(int(c["dit"]["batch_size"]), net.c, net.size, net.size, net.size, device=device)

            def step():
                sig = sample_sigmas(len(z), c["edm"], device)
                return edm_loss_dense(dit_denoiser(net, c["edm"]), z, sig)[0]
        elif kind == "struct":
            from .structure import build_struct, plan_step, struct_step

            net = build_struct(c, device)
            cs = dict(c, struct=dict(c["struct"], recon_parent_p=0.0, aug_p=0.0))
            plan = plan_step(eid, ev, cs, rng, level=1)          # the largest parent set (96^3)

            def step():
                return struct_step(net, plan, F_t, cs, device, train=True)[0]
        elif kind == "attr":
            from .attr import attr_step, build_attr, plan_step

            net = build_attr(c, device)
            codec = Codec(meta["stats"], float(c["data"]["q_eps"]))
            plan = plan_step(eid, ev, dict(c, attr=dict(c["attr"], drop_p=0.0)), codec, rng)

            def step():
                return attr_step(net, plan, F_t, c, device)[0]
        else:
            raise ValueError(kind)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-5)
        times = []
        for i in range(int(steps)):
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            with amp_context(device, amp):
                loss = step()
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.time() - t0)
        row = dict(parameters=int(sum(p.numel() for p in net.parameters())),
                   seconds_per_step=float(np.median(times[1:] if len(times) > 1 else times)))
        if device.type == "cuda":
            peak = int(torch.cuda.max_memory_allocated(device))
            total = int(torch.cuda.get_device_properties(device).total_memory)
            row.update(peak_gib=round(peak / 1024 ** 3, 3), total_gib=round(total / 1024 ** 3, 3),
                       fits=bool(peak < 0.9 * total))
        rows[kind] = row
        print(f"memtest {kind}: {row}", flush=True)
        net = opt = None                     # free before the next kind
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    res = dict(gpu=gpu, event=eid, n_active=n_act, checkpointing=bool(use_ckpt), amp=amp, kinds=rows,
               date=datetime.now().isoformat(timespec="seconds"))
    if write:
        name = "".join(ch if ch.isalnum() else "_" for ch in gpu)
        write_json(Path(c["paths"]["output"]) / "memtest" / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json", res)
    return res
