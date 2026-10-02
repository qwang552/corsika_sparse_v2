"""Sampling chains.

Run with `sample --chain ... --run NAME` (jobs/sample.sub, usually several
shards in parallel). The models the chain needs are loaded once per job; each
sample is written as soon as it is done, so a killed job resumes where it
stopped.

    sa_truth   truth 48^3 field                              -> struct -> attr   paired
    sa_recon   test event latent mu -> AE decode             -> struct -> attr   paired
    sdedit     test latent, noised to sigma_s, DiT denoise   -> decode -> ...    paired (less so as sigma_s grows)
    full       pure noise -> DiT -> decode                   -> struct -> attr   unpaired

Output: <case>/samples/<run>/sample_{n:05d}.npz, one per sample, plus run.json
(the settings), checkpoints.json (training step of each model used) and
index_shard{k:03d}.jsonl (one line per sample).  A job with other settings or
other checkpoints on an existing run name is refused.

T2 rescales the photons of every 48^3 cell to the photon sum of the decoded
field in that cell.

npz keys: ijk int32, q float32 (after T2 if on), q_raw float32 (attr output),
off float32, xyz float32 (m), eid int64 (-1 for full), counts int64 (voxels per
level), z float16 (standardized latent; sdedit / full / sa_recon), F float16
(decoded field, or the truth field for sa_truth, (3, 48, 48, 48); only for the
first 32 indices, for plots) and chain (str).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .common import read_json, save_npz, write_json
from .data import Codec, load_event
from .fields import coarse_field, load_field_stats, occupied_cells, truth_field3
from .geometry import voxel_center
from .loader import case_dir, meta_for

CHAINS = ("sa_truth", "sa_recon", "sdedit", "full")
PAIRED = ("sa_truth", "sa_recon", "sdedit")


def load_chain(c, case, chain, checkpoint="best", device=None):
    """Load the checkpoints a chain needs (AE, DiT, struct, attr) and the latent store."""
    import torch

    from .attr import build_attr
    from .common import torch_load
    from .dit import build_dit
    from .field_ae import LatentStore, ReconSource, load_ae
    from .structure import build_struct

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = case_dir(c, case)
    meta = meta_for(c)
    ch = dict(c=c, case=case, device=device, out_dir=out_dir, meta=meta,
              root=Path(c["paths"]["processed"]), codec=Codec(meta["stats"], float(c["data"]["q_eps"])),
              stats=load_field_stats(c), steps={})
    for kind, build in (("struct", build_struct), ("attr", build_attr)):
        state = torch_load(out_dir / f"{kind}_{checkpoint}.pt", device)
        net = build(c, device)
        net.load_state_dict(state["ema"])
        net.eval()
        ch[kind], ch["steps"][kind] = net, int(state["step"])
    if chain != "sa_truth":
        ch["store"] = LatentStore(out_dir)
        ae, st = load_ae(out_dir, ch["store"].done["checkpoint"], device, c)
        ch["recon"] = ReconSource(ae, ch["store"], device)
        ch["steps"]["ae"] = int(st["step"])
    if chain in ("sdedit", "full"):
        state = torch_load(out_dir / f"dit_{checkpoint}.pt", device)
        net = build_dit(c, device)
        net.load_state_dict(state["ema"])
        net.eval()
        ch["dit"], ch["steps"]["dit"] = net, int(state["step"])
    return ch


def field_for(ch, chain, eid, gen, sigma_start=None):
    """Decoded (or truth) field F (3,B,B,B) on the device, and the standardized latent."""
    import torch

    from .dit import sample_latents

    c, device = ch["c"], ch["device"]
    if chain == "sa_truth":
        ev = load_event(ch["root"], eid)
        f2, _, _ = coarse_field(ev["ijk"], ev["q"], int(c["data"]["grid"]), int(c["field"]["base_grid"]),
                                float(c["data"]["q_eps"]), ch["stats"])
        return torch.as_tensor(truth_field3(f2), device=device), None
    store = ch["store"]
    if chain == "full":
        z = sample_latents(ch["dit"], c, 1, device, generator=gen)[0]
    else:
        mu, _ = store.posterior(eid)
        z = torch.as_tensor(store.standardize(mu), device=device)
        if chain == "sdedit":
            z = sample_latents(ch["dit"], c, 1, device, generator=gen, x_init=z[None],
                               sigma_start=float(sigma_start))[0]
    z_np = z.float().cpu().numpy()
    F_t = ch["recon"].decode_raw(store.unstandardize(z_np))[0]
    return F_t, z_np


def run_one(ch, chain, eid, seed, sigma_start=None, t2=True, save_field=False):
    """Generate one sample: field -> struct -> attr (-> T2); returns a dict with ijk, q, off, ...
    """
    import torch

    from .attr import sample_attr, t2_rescale
    from .structure import grow_structure

    c, device = ch["c"], ch["device"]
    t0 = time.time()
    gens = [torch.Generator(device=device).manual_seed(int(seed) * 7919 + k) for k in range(3)]
    F_t, z = field_for(ch, chain, eid, gens[0], sigma_start)
    start = occupied_cells(F_t[0].cpu().numpy(), float(c["field"]["occ_threshold"]))
    ijk, counts = grow_structure(ch["struct"], F_t, start, c, gens[1])
    out = dict(chain=chain, eid=int(eid), counts=counts, seed=int(seed))
    if ijk is None:
        out.update(failed=True, seconds=round(time.time() - t0, 1))
        return out
    q_raw, off, lookup = sample_attr(ch["attr"], ijk, F_t, c, ch["codec"], gens[2])
    F_np = F_t.cpu().numpy()
    q, scales = (t2_rescale(q_raw, lookup, F_np, c, ch["stats"]) if t2 else (q_raw, np.ones(0)))
    ranges = np.asarray(c["data"]["ranges"], dtype=np.float64)
    voxel = (ranges[:, 1] - ranges[:, 0]) / int(c["data"]["grid"])
    xyz = voxel_center(ijk, ranges, int(c["data"]["grid"])) + off * voxel
    out.update(ijk=ijk, q=q, q_raw=q_raw, off=off, xyz=xyz, z=z, F=F_np if save_field else None,
               t2_scale_median=float(np.median(scales)) if len(scales) else 1.0,
               seconds=round(time.time() - t0, 1))
    return out


def save_result(path, r):
    """Write one sample to its .npz (atomic)."""
    items = dict(ijk=r["ijk"].astype(np.int32), q=r["q"].astype(np.float32),
                 q_raw=r["q_raw"].astype(np.float32), off=r["off"].astype(np.float32),
                 xyz=r["xyz"].astype(np.float32), eid=np.int64(r["eid"]),
                 counts=np.asarray(r["counts"], dtype=np.int64), chain=np.asarray(r["chain"]))
    if r.get("z") is not None:
        items["z"] = np.asarray(r["z"], dtype=np.float16)
    if r.get("F") is not None:
        items["F"] = np.asarray(r["F"], dtype=np.float16)
    save_npz(path, **items)


def paired_events(meta, n, events=None):
    """Test event ids used by the paired chains (the first n, or the ones given)."""
    if events:
        return [int(e) for e in events]
    ids = [int(e) for e in (meta["split"]["test"] or meta["split"]["val"])]
    return ids[: int(n)]


def generate(c, case, chain, run, n_samples=8, seed=0, shard=0, num_shards=1, sigma_start=None,
             t2=None, checkpoint="best", allow_cpu=False, events=None, verbose=True):
    """Run a chain for this shard's share of the samples and write them to <case>/samples/<run>/.
    """
    import torch

    if chain not in CHAINS:
        raise ValueError(f"chain must be one of {CHAINS}")
    if chain == "sdedit" and sigma_start is None:
        raise ValueError("sdedit needs --sigma-start")
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("no GPU; pass --allow-cpu to run on CPU (slow)")
    t2 = bool(c["sample"]["t2"]) if t2 is None else bool(t2)
    run_dir = case_dir(c, case) / "samples" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_for(c)
    if chain in PAIRED:
        eids = paired_events(meta, n_samples, events)
    else:
        eids = [-1] * int(n_samples)
    settings = dict(chain=chain, case=case, checkpoint=checkpoint, n_samples=len(eids), seed=int(seed),
                    sigma_start=None if sigma_start is None else float(sigma_start), t2=t2,
                    events=eids if chain in PAIRED else None, heun_steps=int(c["sample"]["steps"]),
                    occ_threshold=float(c["field"]["occ_threshold"]))
    spec = run_dir / "run.json"
    if spec.exists():
        prev = read_json(spec)
        if {k: prev.get(k) for k in settings} != settings:
            raise FileExistsError(f"{run_dir} was made with other settings ({prev}); pick a new --run")
    else:
        write_json(spec, settings)
    ch = load_chain(c, case, chain, checkpoint)
    used = run_dir / "checkpoints.json"
    if used.exists():
        if read_json(used) != ch["steps"]:
            raise FileExistsError(f"{run_dir} was sampled with checkpoints {read_json(used)}, "
                                  f"now {ch['steps']}; pick a new --run")
    else:
        write_json(used, ch["steps"])
    if verbose:
        print(f"sample {chain} run {run}: checkpoints {ch['steps']} on {ch['device']}, "
              f"{len(eids)} samples, shard {shard}/{num_shards}, T2 {'on' if t2 else 'off'}", flush=True)
    rows = []
    log = run_dir / f"index_shard{int(shard):03d}.jsonl"
    for n in range(int(shard), len(eids), max(1, int(num_shards))):
        path = run_dir / f"sample_{n:05d}.npz"
        if path.exists():
            continue
        r = run_one(ch, chain, eids[n], int(seed) + n, sigma_start, t2, save_field=n < 32)
        row = dict(index=n, eid=r["eid"], counts=r["counts"], seconds=r["seconds"])
        if r.get("failed"):
            row["failed"] = True
        else:
            save_result(path, r)
            row.update(n_active=int(len(r["q"])), q_total=float(r["q"].sum()),
                       q_raw_total=float(r["q_raw"].sum()), t2_scale_median=r["t2_scale_median"])
        rows.append(row)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
        if verbose:
            print(f"  {n}: " + (f"FAILED counts {r['counts']}" if r.get("failed") else
                               f"{row['n_active']} voxels, Q {row['q_total']:.4g}, {r['seconds']} s"),
                  flush=True)
    return rows


def load_run(run_dir):
    """All samples of a run: list of dicts (ijk, q, off, eid, ...), sorted by index."""
    run_dir = Path(run_dir)
    out = []
    for f in sorted(run_dir.glob("sample_*.npz")):
        with np.load(f) as d:
            r = {k: np.asarray(d[k]) for k in d.files}
        r["index"] = int(f.stem.split("_")[1])
        r["eid"] = int(r["eid"]) if "eid" in r else -1
        out.append(r)
    return out


def load_v1_run(sample_dir):
    """Samples written by the previous (v1) package: sample_*.npz with macro_event."""
    out = []
    for f in sorted(Path(sample_dir).glob("sample_*.npz")):
        with np.load(f) as d:
            r = dict(ijk=np.asarray(d["ijk"]), q=np.asarray(d["q"]), off=np.asarray(d["off"]),
                     eid=int(d["macro_event"]) if "macro_event" in d.files else -1)
        r["index"] = int(f.stem.split("_")[1])
        out.append(r)
    return out
