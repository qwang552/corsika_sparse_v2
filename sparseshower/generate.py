"""Generate new showers from a packed model folder, without any training data.

    python -m sparseshower.generate --n 10

reads the trained model from trained_model/ and writes showers/shower_00000.npz,
... (--model and --out change the two folders).  Each file holds

    xyz        (N, 3) float32   position of each lit 2 cm voxel [m]
                                (photon-weighted point inside the voxel)
    nphotons   (N,)   float32   number of Cherenkov photons in that voxel
    ijk        (N, 3) int32     voxel index on the 192^3 grid (x, y, z)
    box_m      (3, 2) float64   x, y, z extent of the grid [m]

    import numpy as np
    s = np.load("showers/shower_00000.npz")
    s["xyz"], s["nphotons"]

All showers are 1 TeV (the model is not yet conditioned on energy or angle).
Shower n uses seed + n, so the same --seed gives the same showers, and a
second run with a larger --n only adds the missing files.  A GPU is used when
one is available; on a CPU one shower takes minutes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import read_json, save_npz, write_json
from .data import Codec


class LatentScale:
    """Per-channel latent mean / std (the part of LatentStore the unconditional chain uses)."""

    def __init__(self, path):
        s = read_json(path)
        self.mean = np.asarray(s["mean"], dtype=np.float32)
        self.std = np.asarray(s["std"], dtype=np.float32)

    def unstandardize(self, z):
        """Standardized latent -> raw latent (per channel)."""
        return z * self.std[:, None, None, None] + self.mean[:, None, None, None]


def load_model(model_dir, device):
    """Build the four networks from a folder written by `pack-model`."""
    import torch

    from .attr import build_attr
    from .dit import build_dit
    from .field_ae import ReconSource, build_ae
    from .structure import build_struct

    d = Path(model_dir)
    c = read_json(d / "config.json")
    ch = dict(c=c, device=device, codec=Codec(read_json(d / "codec_stats.json"), float(c["data"]["q_eps"])),
              stats=read_json(d / "field_stats.json"), store=LatentScale(d / "latent_stats.json"), steps={})
    nets = {}
    for kind, build in (("ae", build_ae), ("dit", build_dit), ("struct", build_struct), ("attr", build_attr)):
        state = torch.load(d / f"{kind}.pt", map_location=device, weights_only=False)
        net = build(c, device)
        net.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in state["ema"].items()})
        net.eval()
        nets[kind], ch["steps"][kind] = net, int(state["step"])
    ch.update(dit=nets["dit"], struct=nets["struct"], attr=nets["attr"],
              recon=ReconSource(nets["ae"], ch["store"], device))
    return ch


def generate(model_dir, n, out, seed=0, device=None, t2=None, verbose=True):
    """Write n showers to `out`; returns the list of written file names."""
    import torch

    from .sample import run_one

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ch = load_model(model_dir, device)
    c = ch["c"]
    t2 = bool(c["sample"]["t2"]) if t2 is None else bool(t2)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(model=str(Path(model_dir).resolve()), steps=ch["steps"], seed=int(seed), t2=t2)
    if (out / "run.json").exists():
        prev = read_json(out / "run.json")
        if {k: prev.get(k) for k in ("steps", "seed", "t2")} != {k: spec[k] for k in ("steps", "seed", "t2")}:
            raise FileExistsError(f"{out} holds showers from another model or seed; pick a new --out")
    else:
        write_json(out / "run.json", spec)
        (out / "run.json").chmod(0o644)
    box = np.asarray(c["data"]["ranges"], dtype=np.float64)
    if verbose:
        print(f"model {model_dir} (steps {ch['steps']}) on {device}; {n} showers -> {out}", flush=True)
    written = []
    for i in range(int(n)):
        path = out / f"shower_{i:05d}.npz"
        if path.exists():
            continue
        t0 = time.time()
        r = run_one(ch, "full", -1, int(seed) + i, t2=t2)
        if r.get("failed"):
            print(f"shower {i}: no voxels generated, skipped", flush=True)
            continue
        save_npz(path, xyz=r["xyz"].astype(np.float32), nphotons=r["q"].astype(np.float32),
                 ijk=r["ijk"].astype(np.int32), box_m=box)
        path.chmod(0o644)                   # readable by others on a shared disk
        written.append(path.name)
        if verbose:
            print(f"shower {i}: {len(r['q'])} voxels, {r['q'].sum():.3g} photons, {time.time() - t0:.0f} s",
                  flush=True)
    return written


def main(argv=None):
    """Command line of `python -m sparseshower.generate`."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="trained_model", help="folder written by `cli pack-model`")
    ap.add_argument("--n", type=int, default=10, help="number of showers")
    ap.add_argument("--out", default="showers", help="output folder")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    files = generate(a.model, a.n, a.out, a.seed)
    print(json.dumps(dict(written=len(files), out=a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
