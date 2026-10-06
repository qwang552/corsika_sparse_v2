"""Pack a trained case into a small model folder that `generate` can run on its own.

    python -m sparseshower.cli pack-model --config configs/v2_cal85.yaml

writes trained_model/ in the current directory.

The folder holds everything the unconditional chain needs and nothing from the
training data:

    ae.pt, dit.pt, struct.pt, attr.pt   EMA weights and training step of each model
    config.json                         the config used for packing (paths removed)
    latent_stats.json                   per-channel mean / std of the train latents (DiT scale)
    field_stats.json                    48^3 field statistics (AE decode, T2)
    codec_stats.json                    photon / offset scales of the attr output
    info.json                           source case, checkpoint steps, weight precision

Weights are stored as float16, so every file stays below GitHub's 100 MB
limit; `generate` converts them back to float32.  An existing folder is refused.
"""
from __future__ import annotations

import time
from pathlib import Path

from .common import read_json, torch_load, write_json
from .field_ae import latent_dir, load_latent_stats
from .fields import load_field_stats
from .loader import case_dir, meta_for

KINDS = ("ae", "dit", "struct", "attr")
GITHUB_LIMIT_MB = 100


def pack_model(c, case, out="trained_model", checkpoint="best", dtype="float16"):
    """Write the model folder `out` from <paths.output>/<case>; returns info.json."""
    import torch

    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; pick a new --out (model folders are not overwritten)")
    src = case_dir(c, case)
    done = read_json(latent_dir(src) / "done.json")
    names = dict(ae=f"ae_{done['checkpoint']}.pt", dit=f"dit_{checkpoint}.pt",
                 struct=f"struct_{checkpoint}.pt", attr=f"attr_{checkpoint}.pt")
    missing = [n for n in names.values() if not (src / n).exists()]
    if missing:
        raise FileNotFoundError(f"missing in {src}: {', '.join(missing)}")
    out.mkdir(parents=True)
    cast = torch.float16 if dtype == "float16" else torch.float32
    steps, sizes = {}, {}
    for kind in KINDS:
        state = torch_load(src / names[kind], torch.device("cpu"))
        ema = {k: (v.to(cast) if torch.is_floating_point(v) else v) for k, v in state["ema"].items()}
        steps[kind] = int(state["step"])
        torch.save(dict(ema=ema, step=steps[kind]), out / f"{kind}.pt")
        sizes[kind] = round((out / f"{kind}.pt").stat().st_size / 2 ** 20, 1)
    mean, std = load_latent_stats(src)
    write_json(out / "latent_stats.json", dict(mean=mean.tolist(), std=std.tolist()))
    write_json(out / "field_stats.json", load_field_stats(c))
    stats = meta_for(c)["stats"]
    write_json(out / "codec_stats.json", {k: stats[k] for k in ("q_log_mean", "q_log_std", "off_std")})
    write_json(out / "config.json", {k: v for k, v in c.items() if k != "paths"})
    info = dict(case=case, checkpoints=names, steps=steps, dtype=dtype, size_mb=sizes,
                occ_threshold=float(c["field"]["occ_threshold"]), packed=time.strftime("%Y-%m-%d %H:%M"))
    write_json(out / "info.json", info)
    for f in out.iterdir():
        f.chmod(0o644)                      # readable by others when shared on a cluster disk
    big = [k for k, s in sizes.items() if s >= GITHUB_LIMIT_MB]
    if big:
        print(f"warning: {', '.join(big)} above {GITHUB_LIMIT_MB} MB; GitHub refuses such files without Git LFS")
    return info
