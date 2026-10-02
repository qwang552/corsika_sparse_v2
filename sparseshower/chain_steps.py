"""Every intermediate of one SDEdit chain for one test event, in one figure.

    python -m sparseshower.chain_steps --config configs/v2_cal85.yaml --case v2a --event 28 --sigma 1
    python -m sparseshower.chain_steps --config configs/v2_cal85.yaml --case v2a --event 28 --match-run sdedit_1

The steps are those of sample.run_one, with the same generators in the same order:

  1  encoder latent mu of the test event, standardized              8 x 12^3
  2  + sigma * noise: the start of the DiT chain                     8 x 12^3
  3  DiT Heun steps sigma -> 0: state x_t and its estimate D(x_t)    8 x 12^3 per step
  4  AE decode -> F = (p_occ, ch0, ch1)                              3 x 48^3
  5  cells with p_occ > occ_threshold: the struct start set          48^3 cells
  6  struct 48 -> 96 -> 192                                          active cells per level
  7  attr photons and offsets per voxel, then T2 per-cell rescale    192^3 voxels

Truth and the decode of the clean latent (what sa_recon uses) are drawn next to
them.  With --match-run, seed, sigma, T2 and checkpoint come from that run and the result is
compared with the run's saved sample (bit-identical only on the same device type
and with the same checkpoints).

Output (never overwritten; a timestamp is added if the name exists):
  <case>/viz/chain_steps/event<E>_s<sigma>.png   the figure
  <case>/viz/chain_steps/event<E>_s<sigma>.npz   every intermediate array
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import read_json
from .data import load_event
from .evaluate import profiles
from .fields import coarse_field, occupied_cells, photon_sums, truth_field3
from .loader import case_dir, meta_for


# ------------------------------------------------------------------- run ---
def run_steps(c, case, eid, sigma, seed=0, t2=None, checkpoint="best", device=None, n_show=5, verbose=True):
    """Re-run one SDEdit sample of one event and keep every intermediate (latents, levels, photons).
    """
    import torch

    from .attr import sample_attr, t2_rescale
    from .diffusion import heun_sample
    from .dit import dit_denoiser
    from .sample import load_chain
    from .samples import level_grids
    from .structure import sample_level

    log = print if verbose else (lambda *a, **k: None)
    t2 = bool(c["sample"]["t2"]) if t2 is None else bool(t2)
    ch = load_chain(c, case, "sdedit", checkpoint, device)
    dev, store = ch["device"], ch["store"]
    log(f"event {eid}, sigma {sigma}, seed {seed}, T2 {'on' if t2 else 'off'}, checkpoints {ch['steps']} on {dev}")
    gens = [torch.Generator(device=dev).manual_seed(int(seed) * 7919 + k) for k in range(3)]
    secs = {}

    # 1-3: latent, noise, DiT (every denoiser call recorded; Heun calls it twice per step
    # except the last, so the even calls are the states x_t)
    mu, _ = store.posterior(eid)
    z0 = store.standardize(mu)
    calls = []
    den = dit_denoiser(ch["dit"], c["edm"])

    def recorded(x, s, **k):
        """Denoiser wrapper that records (sigma, x_t, D(x_t)) at every call."""
        d = den(x, s, **k)
        calls.append((float(s.reshape(-1)[0]), x[0].float().cpu().numpy(), d[0].float().cpu().numpy()))
        return d

    t = time.time()
    with torch.no_grad():
        zhat = heun_sample(recorded, (1,) + tuple(z0.shape), c["edm"], dev, steps=int(c["sample"]["steps"]),
                           generator=gens[0], x_init=torch.as_tensor(z0, device=dev)[None],
                           sigma_start=float(sigma))[0]
    zhat = zhat.float().cpu().numpy()
    secs["dit"] = time.time() - t
    states = calls[0::2]
    sig_t = np.array([s for s, _, _ in states])
    x_t = np.stack([x for _, x, _ in states])
    d_t = np.stack([d for _, _, d in states])
    log(f"  DiT: {len(states)} Heun steps, {len(calls)} network calls, {secs['dit']:.1f} s")

    # 4: decode (the chain's field, the clean latent's field, and a few intermediate estimates)
    t = time.time()
    F_t = ch["recon"].decode_raw(store.unstandardize(zhat))[0]
    secs["decode"] = time.time() - t
    F = F_t.cpu().numpy()
    F_z0 = ch["recon"].decode_raw(store.unstandardize(z0))[0].cpu().numpy()
    show = np.unique(np.linspace(0, len(states) - 1, int(n_show)).round().astype(int))
    p_show = np.stack([ch["recon"].decode_raw(store.unstandardize(d_t[i]))[0][0].cpu().numpy() for i in show])

    # 5-6: start set and struct, level by level (as structure.grow_structure)
    thr = float(c["field"]["occ_threshold"])
    grid, base = int(c["data"]["grid"]), int(c["field"]["base_grid"])
    start = occupied_cells(F[0], thr)
    levels, failed = [start], None
    t = time.time()
    if len(start) == 0 or len(start) > int(c["sample"].get("max_start", len(start))):
        failed = f"start set has {len(start)} cells"
    else:
        with torch.no_grad():
            g = ch["struct"].encoder(F_t)
        ijk = start
        for level, gr in enumerate(level_grids(grid, base)[:-1]):
            ijk, _ = sample_level(ch["struct"], ijk, gr, level, F_t, g, c, gens[1])
            levels.append(np.asarray(ijk, dtype=np.int64))
            if len(ijk) == 0 or len(ijk) > int(c["sample"]["max_active"]):
                failed = f"level {level + 1} has {len(ijk)} cells"
                break
    secs["struct"] = time.time() - t
    log(f"  struct: cells per level {[len(v) for v in levels]}, {secs['struct']:.1f} s" + (f" FAILED: {failed}" if failed else ""))

    # 7: attr and T2
    q_raw = off = q = None
    if failed is None:
        t = time.time()
        q_raw, off, lookup = sample_attr(ch["attr"], levels[-1], F_t, c, ch["codec"], gens[2])
        q = t2_rescale(q_raw, lookup, F, c, ch["stats"])[0] if t2 else q_raw
        secs["attr"] = time.time() - t
        log(f"  attr: {len(q)} voxels, Q raw {q_raw.sum():.4g}, Q after T2 {q.sum():.4g}, {secs['attr']:.1f} s")

    # truth at every level
    ev = load_event(Path(c["paths"]["processed"]), eid)
    f2, t_counts, t_sums = coarse_field(ev["ijk"], ev["q"], grid, base, float(c["data"]["q_eps"]), ch["stats"])
    truth_levels = [np.unique(np.asarray(ev["ijk"], dtype=np.int64) // (grid // gr), axis=0)
                    for gr in level_grids(grid, base)]
    return dict(eid=int(eid), sigma=float(sigma), seed=int(seed), t2=t2, steps=ch["steps"], device=str(dev),
                thr=thr, z0=z0, z_noised=calls[0][1], zhat=zhat, sig_t=sig_t, x_t=x_t, d_t=d_t, show=show,
                p_show=p_show, F=F, F_z0=F_z0, F_truth=truth_field3(f2), truth_sums=t_sums.reshape(base, base, base),
                levels=levels, truth_levels=truth_levels, failed=failed, q_raw=q_raw, q=q, off=off,
                truth_ijk=np.asarray(ev["ijk"]), truth_q=np.asarray(ev["q"], dtype=np.float64), secs=secs,
                field_stats=ch["stats"])


def match_run(c, case, eid, run):
    """seed, sigma, T2 and sample index of `eid` in an sdedit run."""
    run_dir = case_dir(c, case) / "samples" / run
    spec = read_json(run_dir / "run.json")
    if spec.get("chain") != "sdedit":
        raise ValueError(f"{run} is a {spec.get('chain')} run; --match-run needs an sdedit run")
    events = [int(e) for e in spec["events"]]
    if int(eid) not in events:
        raise ValueError(f"event {eid} is not in {run} (events {events[:10]}...)")
    n = events.index(int(eid))
    if abs(float(spec.get("occ_threshold", -1)) - float(c["field"]["occ_threshold"])) > 1e-9:
        print(f"WARNING: {run} used occ_threshold {spec.get('occ_threshold')}, this config "
              f"{c['field']['occ_threshold']}: pass the config the run was made with")
    return dict(seed=int(spec["seed"]) + n, sigma=float(spec["sigma_start"]), t2=bool(spec["t2"]), index=n,
                checkpoint=spec.get("checkpoint", "best"),
                path=run_dir / f"sample_{n:05d}.npz",
                steps=read_json(run_dir / "checkpoints.json") if (run_dir / "checkpoints.json").exists() else None)


def compare(r, m):
    """Check that the re-run matches the saved sample of the run (voxel count, photons)."""
    if not m["path"].exists():
        return f"sample {m['index']} of the run is not written yet: nothing to compare"
    with np.load(m["path"]) as d:
        ijk, q = np.asarray(d["ijk"]), np.asarray(d["q"], dtype=np.float64)
    same_ckpt = m["steps"] is None or all(int(m["steps"].get(k, -1)) == int(v) for k, v in r["steps"].items())
    if r["q"] is None:
        return "this chain failed in struct, the run's sample did not: they differ"
    if len(ijk) == len(r["levels"][-1]) and np.array_equal(ijk, r["levels"][-1]) and np.allclose(q, r["q"], rtol=1e-3):
        return f"identical to sample {m['index']} of the run ({len(ijk)} voxels)"
    why = ("checkpoints differ: run " + json.dumps(m["steps"]) + ", now " + json.dumps(r["steps"])) if not same_ckpt \
        else "same checkpoints; most likely another device type (GPU and CPU random numbers differ)"
    return f"differs from sample {m['index']} of the run ({len(ijk)} vs {len(r['levels'][-1])} voxels): {why}"


# ------------------------------------------------------------------ plot ---
def _mosaic(z, cols=4):
    """(C, L, L, L) latent -> 2D mosaic of the C channels, each the y-slice through the centre (z up, x right)."""
    C, L = z.shape[0], z.shape[-1]
    rows = int(np.ceil(C / cols))
    out = np.full((rows * (L + 1) - 1, cols * (L + 1) - 1), np.nan)
    for k in range(C):
        r, q = divmod(k, cols)
        out[r * (L + 1): r * (L + 1) + L, q * (L + 1): q * (L + 1) + L] = z[k][:, L // 2, :]
    return out


def _count_image(ijk, grid):
    """Active cells along y for every (z, x) column."""
    ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)
    return np.bincount(ijk[:, 2] * grid + ijk[:, 0], minlength=grid * grid).reshape(grid, grid).astype(float)


def _photon_image(ijk, q, grid):
    """Photon sum along y for every (z, x) column."""
    ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)
    return np.bincount(ijk[:, 2] * grid + ijk[:, 0], weights=q, minlength=grid * grid).reshape(grid, grid)


def plot(r, c, title=""):
    """The step-by-step figure of one chain."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FormatStrFormatter, NullFormatter

    ranges = np.asarray(c["data"]["ranges"], dtype=float)
    ext = [ranges[0, 0], ranges[0, 1], ranges[2, 0], ranges[2, 1]]
    grid, base = int(c["data"]["grid"]), int(c["field"]["base_grid"])
    fig = plt.figure(figsize=(22, 27))
    gs = fig.add_gridspec(6, 5, hspace=0.42, wspace=0.38, top=0.95, bottom=0.03, left=0.04, right=0.965)

    def img(ax, a, ttl, cmap="viridis", norm=None, vmin=None, vmax=None, extent=ext, cb=True):
        """imshow helper with a black background and an optional colour bar."""
        ax.set_facecolor("black")
        im = ax.imshow(a, origin="lower", extent=extent, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        ax.set_title(ttl, fontsize=10.5)
        if cb:
            bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if norm is not None:
                bar.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
                bar.ax.yaxis.set_minor_formatter(NullFormatter())
        return im

    # row 1: latents
    lim = 3.0
    lat = [(r["z0"], "1  encoder latent μ (standardized)"), (r["z_noised"], f"2  + σ·noise, σ = {r['sigma']:g}"),
           (r["zhat"], "3  after DiT (σ → 0)"), (r["zhat"] - r["z0"], "DiT output − encoder latent")]
    for k, (z, ttl) in enumerate(lat):
        ax = fig.add_subplot(gs[0, k])
        L = z.shape[-1]
        img(ax, _mosaic(z), ttl + f"\nrms {np.sqrt(np.mean(z ** 2)):.2f}", cmap="RdBu_r", vmin=-lim, vmax=lim,
            extent=None, cb=(k == 3))
        ax.set_xticks([(L + 1) * q + L / 2 - 0.5 for q in range(4)])
        ax.set_xticklabels([f"ch{q}" + (f"/ch{q + 4}" if q + 4 < z.shape[0] else "") for q in range(4)], fontsize=8)
        ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 4])
    e_x = np.sqrt(((r["x_t"] - r["z0"]) ** 2).mean(axis=(1, 2, 3, 4)))
    e_d = np.sqrt(((r["d_t"] - r["z0"]) ** 2).mean(axis=(1, 2, 3, 4)))
    ax.plot(r["sig_t"], e_x, "o-", ms=3, label="state x_t")
    ax.plot(r["sig_t"], e_d, "o-", ms=3, label="DiT estimate D(x_t)")
    ax.plot(r["sig_t"], r["sig_t"], ":", color="grey", label="σ (added noise level)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("σ (sampling runs left to right, towards 0)")
    ax.set_ylabel("rms distance to encoder latent")
    ax.set_title("3  DiT trajectory", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # row 2: what the DiT estimate decodes to along the way
    for k, i in enumerate(r["show"][:5]):
        ax = fig.add_subplot(gs[1, k])
        img(ax, r["p_show"][k].max(axis=1), f"3  decode(D(x_t)) at σ = {r['sig_t'][i]:.3g}\np_occ, max over y",
            vmin=0, vmax=1, cb=(k == len(r["show"][:5]) - 1))

    # row 3: 48^3 fields
    thr = r["thr"]
    stats = r.get("field_stats")
    sums_hat = photon_sums(r["F"][2], r["F"][0] > thr, stats, float(c["data"]["q_eps"])) if stats else None
    tsum = r["truth_sums"]
    ax = fig.add_subplot(gs[2, 0])
    img(ax, r["F_truth"][0].max(axis=1), "truth 48³: occupied (max over y)", vmin=0, vmax=1, cb=False)
    ax = fig.add_subplot(gs[2, 1])
    img(ax, r["F_z0"][0].max(axis=1), "decode(encoder latent) = sa_recon field\np_occ", vmin=0, vmax=1, cb=False)
    ax = fig.add_subplot(gs[2, 2])
    img(ax, r["F"][0].max(axis=1), "4  decode(DiT output): p_occ", vmin=0, vmax=1)
    pos = [a for a in (tsum.sum(axis=1), None if sums_hat is None else sums_hat.sum(axis=1)) if a is not None]
    vmax = max(float(a.max()) for a in pos) if pos else 1.0
    vmin = max(vmax * 1e-5, 1e-3)
    ax = fig.add_subplot(gs[2, 3])
    img(ax, np.where(tsum.sum(axis=1) > 0, tsum.sum(axis=1), np.nan), "truth 48³: photons (sum over y)",
        cmap="magma", norm=LogNorm(vmin, vmax), cb=False)
    if sums_hat is not None:
        ax = fig.add_subplot(gs[2, 4])
        s = sums_hat.sum(axis=1)
        img(ax, np.where(s > 0, s, np.nan), f"4  decode(DiT output): photons\n(ch1, cells with p_occ > {thr:g})",
            cmap="magma", norm=LogNorm(vmin, vmax))

    # rows 4-5: struct levels, generated over truth
    lv, tl = r["levels"], r["truth_levels"]
    grids = [base * 2 ** k for k in range(len(tl))]
    names = [f"5  start set: p_occ > {thr:g}"] + [f"6  struct → {g}³" for g in grids[1:]]
    for k, gr in enumerate(grids):
        ax = fig.add_subplot(gs[3, k])
        if k < len(lv):
            a = _count_image(lv[k], gr)
            n_t = len(tl[k])
            img(ax, np.where(a > 0, a, np.nan), f"{names[k]}\n{len(lv[k]):,} cells ({len(lv[k]) / max(n_t, 1):.3f}× truth)",
                cmap="viridis", norm=LogNorm(1, max(2.0, a.max())))
        else:
            ax.axis("off")
            ax.set_title(f"{names[k]}\nnot reached ({r['failed']})", fontsize=10.5)
        ax = fig.add_subplot(gs[4, k])
        a = _count_image(tl[k], gr)
        img(ax, np.where(a > 0, a, np.nan), f"truth at {gr}³: {len(tl[k]):,} cells\n(active cells along y)",
            cmap="viridis", norm=LogNorm(1, max(2.0, a.max())))
    ax = fig.add_subplot(gs[3, 3])
    ax.axis("off")
    ax.text(0, 1, _summary(r), va="top", fontsize=9.5, family="monospace", transform=ax.transAxes)

    # row 6: attr
    tp = _photon_image(r["truth_ijk"], r["truth_q"], grid)
    vmax = float(tp.max())
    vmin = max(vmax * 1e-6, 1e-3)
    ax = fig.add_subplot(gs[5, 0])
    img(ax, np.where(tp > 0, tp, np.nan), f"truth 192³ photons (sum over y)\nQ {r['truth_q'].sum():.4g}", cmap="magma",
        norm=LogNorm(vmin, vmax), cb=False)
    if r["q"] is not None:
        for k, (qq, ttl) in enumerate(((r["q_raw"], "7  attr output (before T2)"),
                                       (r["q"], "7  after T2 (final sample)" if r["t2"] else "7  final (T2 off)"))):
            ax = fig.add_subplot(gs[5, 1 + k])
            a = _photon_image(r["levels"][-1], qq, grid)
            img(ax, np.where(a > 0, a, np.nan), f"{ttl}\nQ {qq.sum():.4g} ({qq.sum() / r['truth_q'].sum():.4g}× truth)",
                cmap="magma", norm=LogNorm(vmin, vmax), cb=(k == 1))
        lt, rt = profiles(r["truth_ijk"], r["truth_q"], c)
        lr, rr = profiles(r["levels"][-1], r["q_raw"], c)
        lq, rq = profiles(r["levels"][-1], r["q"], c)
        ax = fig.add_subplot(gs[5, 3])
        for v, lab, sty in ((lt, "truth", "k-"), (lr, "attr output", "C1--"), (lq, "final", "C0-")):
            ax.plot(np.arange(len(v)), v, sty, lw=1.6, label=lab)
        ax.set_title("longitudinal photon share (32 z slabs, top of box left)", fontsize=10.5)
        ax.set_xlabel("z slab")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = fig.add_subplot(gs[5, 4])
        for v, lab, sty in ((rt, "truth", "k-"), (rr, "attr output", "C1--"), (rq, "final", "C0-")):
            ax.plot(np.arange(1, len(v) + 1), np.where(v > 0, v, np.nan), sty, lw=1.6, label=lab)
        ax.set_yscale("log")
        ax.set_title("radial photon share (16 rings, √r spacing)", fontsize=10.5)
        ax.set_xlabel("ring (inner → outer)")
        ax.grid(alpha=0.3)
    for ax in fig.axes:
        if ax.get_images() and ax.images[0].get_extent() == ext:
            ax.set_xlabel("x [m]", fontsize=9)
            ax.set_ylabel("z [m]", fontsize=9)
    fig.suptitle(title, fontsize=14)
    return fig


def _summary(r):
    """Text block with the event, seed, checkpoints and per-level cell counts."""
    lines = [f"event {r['eid']}   σ = {r['sigma']:g}   seed {r['seed']}", f"device {r['device']}",
             "checkpoints " + ", ".join(f"{k} {v}" for k, v in r["steps"].items()), ""]
    lv, tl = r["levels"], r["truth_levels"]
    for k in range(len(tl)):
        n = f"{len(lv[k]):>8,}" if k < len(lv) else "       –"
        lines.append(f"level {k}: {n} cells, truth {len(tl[k]):>8,}")
    if r["q"] is not None:
        lines += ["", f"Q final / truth  {r['q'].sum() / r['truth_q'].sum():.4g}",
                  f"Q attr  / truth  {r['q_raw'].sum() / r['truth_q'].sum():.4g}"]
    lines += ["", "seconds: " + ", ".join(f"{k} {v:.1f}" for k, v in r["secs"].items()),
              f"total {sum(r['secs'].values()):.1f} s"]
    if r.get("match"):
        lines += ["", r["match"]]
    return "\n".join(lines)


# ------------------------------------------------------------------ main ---
def save(r, c, case, out=None):
    """Write the figure (.png) and the intermediates (.npz) without overwriting."""
    stem = Path(out) if out else case_dir(c, case) / "viz" / "chain_steps" / f"event{r['eid']}_s{r['sigma']:g}"
    stem = stem.with_suffix("")
    if stem.with_suffix(".png").exists() or stem.with_suffix(".npz").exists():
        if out:
            raise FileExistsError(f"{stem}.png/.npz exists; pass another --out (nothing is overwritten)")
        stem = stem.with_name(f"{stem.name}_{time.strftime('%Y%m%d-%H%M%S')}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    title = (f"SDEdit chain, step by step · test event {r['eid']} · σ = {r['sigma']:g} · case {case}"
             + (" · [SYNTHETIC data]" if c.get("synthetic") else ""))
    fig = plot(r, c, title)
    fig.savefig(stem.with_suffix(".png"), dpi=100)
    arrays = dict(z0=r["z0"], z_noised=r["z_noised"], zhat=r["zhat"], sig_t=r["sig_t"],
                  x_t=r["x_t"].astype(np.float16), d_t=r["d_t"].astype(np.float16), F=r["F"], F_z0=r["F_z0"],
                  F_truth=r["F_truth"], start_ijk=r["levels"][0])
    for k, v in enumerate(r["levels"][1:], 1):
        arrays[f"ijk_level{k}"] = v
    if r["q"] is not None:
        arrays.update(q_raw=r["q_raw"], q=r["q"], off=r["off"])
    arrays["info"] = np.asarray(json.dumps(dict(eid=r["eid"], sigma=r["sigma"], seed=r["seed"], t2=r["t2"],
                                                steps=r["steps"], device=r["device"], thr=r["thr"],
                                                failed=r["failed"], secs=r["secs"], match=r.get("match"))))
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    return stem


def main(argv=None):
    """Command-line entry point (python -m sparseshower.chain_steps)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--case", default="v2a")
    ap.add_argument("--event", type=int, help="test event id (default: the first test event)")
    ap.add_argument("--sigma", type=float, default=1.0, help="SDEdit noise level (ignored with --match-run)")
    ap.add_argument("--seed", type=int, default=0, help="sample seed (ignored with --match-run)")
    ap.add_argument("--match-run", help="sdedit run to reproduce: takes seed, sigma, T2, checkpoint from it and "
                    "compares (none = use --sigma and --seed)")
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--out", help="output path without extension")
    a = ap.parse_args(argv)
    from .common import config

    c = config(a.config)
    eid = a.event if a.event is not None else int(meta_for(c)["split"]["test"][0])
    m = match_run(c, a.case, eid, a.match_run) if a.match_run not in (None, "", "none") else None
    seed, sigma, t2 = (m["seed"], m["sigma"], m["t2"]) if m else (a.seed, a.sigma, None)
    checkpoint = m["checkpoint"] if m else a.checkpoint
    device = None
    if a.device:
        import torch

        device = torch.device(a.device)
    r = run_steps(c, a.case, eid, sigma, seed, t2, checkpoint, device)
    if m:
        r["match"] = f"{a.match_run}: " + compare(r, m)
        print(r["match"])
    stem = save(r, c, a.case, a.out)
    print(f"figure -> {stem}.png\narrays -> {stem}.npz")


if __name__ == "__main__":
    main()
