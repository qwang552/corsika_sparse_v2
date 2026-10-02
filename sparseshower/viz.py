"""Plot helpers behind notebooks/viz.ipynb (training side) and notebooks/sample.ipynb
(sample side).  Matplotlib only; every figure title names its data source.

Slow results are cached under <case>/viz/cache/:
    ae_recon_<ckpt>_s<step>_<eid>.npz     AE reconstruction of one val event (keyed by AE step)
    metrics_<run>_<digest>.npz            event metrics of every sample of a run (keyed by a
                                          digest of the run's sample indices and voxel counts)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import digest, read_json
from .loader import case_dir, meta_for

SOURCE_COLOURS = {"truth": "#222222", "sa_truth": "#1f77b4", "sa_recon": "#2ca02c", "full": "#d62728",
                  "V1": "#9467bd"}


def num(v):
    """Summary values are JSON: NaN was written as null."""
    return float("nan") if v is None else float(v)


def data_label(c):
    """' [SYNTHETIC data]' for configs that say so (smoke.yaml), else ''."""
    return " [SYNTHETIC data]" if c.get("synthetic") else ""


def cache_dir(c, case):
    """Notebook cache directory <case>/viz/cache (created if missing)."""
    d = case_dir(c, case) / "viz" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


SDEDIT_ORDER = ["sdedit_0.25", "sdedit_0.5", "sdedit_1", "sdedit_2", "sdedit_5"]


def colour(name, i=0):
    """Fixed plot colour of a data source (truth, sa_truth, sdedit_*, ...)."""
    if name in SOURCE_COLOURS:
        return SOURCE_COLOURS[name]
    if name.startswith("sdedit"):
        k = SDEDIT_ORDER.index(name) if name in SDEDIT_ORDER else i
        return ["#ff7f0e", "#e6a100", "#bcbd22", "#8c564b", "#e377c2"][k % 5]
    return f"C{i % 10}"


# ------------------------------------------------------------ projections ---
def project(ijk, q, grid, axis):
    """Photon sum over one axis -> 2D image [u, z] (u = x for axis=1, y for axis=0)."""
    ijk = np.asarray(ijk, dtype=np.int64)
    keep = [a for a in (0, 1, 2) if a != axis]
    img = np.zeros((grid, grid))
    np.add.at(img, (ijk[:, keep[0]], ijk[:, keep[1]]), np.asarray(q, dtype=np.float64))
    return img


def show_projection(ax, img, c, title, vmin=None, vmax=None, axis_names=("x", "z")):
    """Draw a 2D photon projection (log10 colour scale) with physical axes."""
    ranges = np.asarray(c["data"]["ranges"], dtype=np.float64)
    a = 0 if axis_names[0] == "x" else 1
    extent = [ranges[a, 0], ranges[a, 1], ranges[2, 0], ranges[2, 1]]
    with np.errstate(divide="ignore"):
        lg = np.where(img > 0, np.log10(img), np.nan)
    im = ax.imshow(lg.T, origin="lower", extent=extent, aspect="auto", cmap="inferno", vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_facecolor("black")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(f"{axis_names[0]} [m]", fontsize=7)
    ax.set_ylabel("z [m]", fontsize=7)
    ax.tick_params(labelsize=6)
    return im


# --------------------------------------------------------------- training ---
def load_history(case_path, kind):
    """Training log <kind>_history.jsonl of a case as a list of dicts (None if missing)."""
    path = Path(case_path) / f"{kind}_history.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows or None


def smooth(y, n):
    """Running mean over n points (for loss curves)."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 3 or n <= 1:
        return y
    n = min(int(n), len(y))
    k = np.ones(n) / n
    pad = np.r_[np.full(n - 1, y[0]), y]
    return np.convolve(pad, k, mode="valid")


def plot_training(c, case):
    """One row per trained kind: loss (raw + smoothed) and the probe curves."""
    import matplotlib.pyplot as plt

    path = case_dir(c, case)
    kinds = [k for k in ("ae", "dit", "struct", "attr") if load_history(path, k)]
    if not kinds:
        print(f"no *_history.jsonl under {path}")
        return None
    fig, axes = plt.subplots(len(kinds), 2, figsize=(12, 3.0 * len(kinds)), squeeze=False)
    th = c.get("thresholds", {}).get("ae", {})
    for row, kind in zip(axes, kinds):
        h = load_history(path, kind)
        hl = [r for r in h if "loss" in r]          # skipped steps (OOM, empty item) have no loss
        step = np.array([r["step"] for r in hl])
        loss = np.array([r["loss"] for r in hl], dtype=np.float64)
        ax = row[0]
        ax.plot(step, loss, lw=0.4, alpha=0.35, color="C0")
        ax.plot(step, smooth(loss, max(1, len(loss) // 50)), lw=1.2, color="C0", label="loss (smoothed)")
        if kind == "struct" and hl and "level" in hl[0]:
            lv = np.array([r.get("level", -1) for r in hl])
            for L in sorted(set(lv.tolist())):
                m = lv == L
                if m.sum() > 2:
                    ax.plot(step[m], smooth(loss[m], max(1, m.sum() // 50)), lw=0.9, label=f"CE level {L}")
        ax.set_yscale("log" if np.all(loss > 0) else "linear")
        ax.set_title(f"{kind}: training loss ({case}/{kind}_history.jsonl)", fontsize=9)
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
        ax = row[1]
        pr = [r for r in h if "probe_R_mean" in r]
        if pr:
            ps = [r["step"] for r in pr]
            keys = sorted({k for r in pr for k in r if k.startswith("probe_") and k != "probe_R_mean"})
            if kind == "ae":
                keys = [k for k in keys if k[6:] in th or k in ("probe_extra_cells_frac",)]
            elif kind == "struct":
                keys = [k for k in keys if k.endswith(("_ce", "_expected_ratio", "lost_children"))]
            for i, k in enumerate(keys[:8]):
                ax.plot(ps, [r.get(k, np.nan) for r in pr], "o-", ms=2, lw=1, label=k[6:], color=f"C{i}")
                if kind == "ae" and k[6:] in th:
                    ax.axhline(float(th[k[6:]]), color=f"C{i}", ls=":", lw=0.8)
            ax.plot(ps, [r["probe_R_mean"] for r in pr], "k--", lw=1, label="R_mean (best.pt criterion)")
            ax.set_yscale("log")
            ax.legend(fontsize=6, ncol=2)
        ax.set_title(f"{kind}: fixed probe on val events" + (" (dotted: acceptance thresholds)" if kind == "ae" else ""),
                     fontsize=9)
        ax.set_xlabel("step")
    fig.tight_layout()
    return fig


def memtest_table(c):
    """Print and return the memtest results found under <output>/memtest."""
    d = Path(c["paths"]["output"]) / "memtest"
    files = sorted(d.glob("*.json")) if d.exists() else []
    if not files:
        print(f"no memtest results under {d}")
        return []
    out = []
    for f in files:
        r = read_json(f)
        print(f"== {f.name}: {r.get('gpu', '?')}  event {r.get('event')} ({r.get('n_active')} voxels)")
        for kind, v in r.get("kinds", {}).items():
            line = "  ".join(f"{k}={v[k]}" for k in v if not isinstance(v[k], (dict, list)))
            print(f"   {kind:<7} {line}")
        out.append(r)
    return out


# ----------------------------------------------------------------- AE side ---
def ae_recon(c, case, eids, checkpoint="best", device="cpu"):
    """Truth vs decoded 48^3 field for val events (cached per AE step).

    The photon sums of the decoded field depend on field.occ_threshold, so they
    are recomputed from the cached field on every call.
    """
    import torch

    from .data import load_event
    from .field_ae import decoded_field, event_field, load_ae
    from .fields import load_field_stats, photon_sums

    stats = load_field_stats(c)
    thr, q_eps = float(c["field"]["occ_threshold"]), float(c["data"]["q_eps"])
    path = case_dir(c, case)
    state_step = None
    ae = None
    rows = []
    for e in eids:
        if state_step is None:
            ae, state = load_ae(path, checkpoint, torch.device(device), c)
            state_step = int(state["step"])
        f = cache_dir(c, case) / f"ae_recon_{checkpoint}_s{state_step}_{int(e):05d}.npz"
        if f.exists():
            with np.load(f) as d:
                r = {k: d[k] for k in d.files if k != "sums_pred"}
        else:
            field, sums, counts = event_field(load_event(c["paths"]["processed"], e), c, stats)
            with torch.no_grad():
                out, _, _ = ae(torch.as_tensor(field[None], device=device), sample=False)
                F = decoded_field(out)[0].cpu().numpy()
            r = dict(eid=np.int64(e), step=np.int64(state_step), truth=field.astype(np.float32),
                     counts=counts.astype(np.float32), sums=sums.astype(np.float32), F=F.astype(np.float32))
            np.savez_compressed(f, **r)
        occ = r["F"][0] > thr
        r["sums_pred"] = photon_sums(r["F"][2].reshape(-1), occ.reshape(-1), stats, q_eps).astype(np.float32)
        rows.append(r)
    return rows


def plot_ae_recon(c, rows, checkpoint="best"):
    """Per event: truth photons | decoded photons (x-z sums), p_occ, and the z profile."""
    import matplotlib.pyplot as plt

    if not rows:
        return None
    B = int(c["field"]["base_grid"])
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.0 * len(rows)), squeeze=False)
    for ax, r in zip(axes, rows):
        st = np.asarray(r["sums"], dtype=np.float64).reshape(B, B, B)       # [z, y, x]
        sp = np.asarray(r["sums_pred"], dtype=np.float64).reshape(B, B, B)
        img_t, img_p = st.sum(axis=1).T, sp.sum(axis=1).T                     # [x, z]
        hi = np.log10(max(img_t.max(), 1e-3))
        lo = hi - 5
        show_projection(ax[0], img_t, c, f"truth {B}³ photons (y sum), event {int(r['eid'])}", lo, hi)
        show_projection(ax[1], img_p, c, f"AE decode (ae_{checkpoint}.pt step {int(r['step'])})", lo, hi)
        pocc = np.asarray(r["F"])[0].max(axis=1).T
        ranges = np.asarray(c["data"]["ranges"])
        ax[2].imshow(pocc.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                     extent=[ranges[0, 0], ranges[0, 1], ranges[2, 0], ranges[2, 1]])
        ax[2].set_title("decoded p_occ (max over y)", fontsize=8)
        ax[3].plot(st.sum(axis=(1, 2))[::-1] / max(st.sum(), 1e-30), label="truth", color="k")
        ax[3].plot(sp.sum(axis=(1, 2))[::-1] / max(sp.sum(), 1e-30), label="decoded", color="C3", ls="--")
        ax[3].set_title(f"z profile (photon fraction per {B}³ slab, top first); Q ratio "
                        f"{sp.sum() / max(st.sum(), 1e-30):.3f}", fontsize=8)
        ax[3].set_xlabel("slab from top")
        ax[3].legend(fontsize=7)
    fig.suptitle(f"AE reconstruction of val events{data_label(c)}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_latents(c, case, full_run=None, max_rows=512):
    """Standardized test latents (encoder mu) vs generated latents of a `full` run, per channel."""
    import matplotlib.pyplot as plt

    from .field_ae import LatentStore
    from .sample import load_run

    try:
        store = LatentStore(case_dir(c, case))
    except FileNotFoundError as exc:
        print(exc)
        return None
    ids = store.ids("test")[:max_rows]
    zt = np.stack([store.standardize(store.posterior(e, "test")[0]) for e in ids])
    zg = None
    if full_run:
        run = case_dir(c, case) / "samples" / full_run
        if run.exists():
            zs = [r["z"] for r in load_run(run) if "z" in r][:max_rows]
            zg = np.stack(zs).astype(np.float32) if zs else None
    C = zt.shape[1]
    fig, axes = plt.subplots(1, C, figsize=(2.2 * C, 2.4), squeeze=False)
    bins = np.linspace(-4, 4, 61)
    for k, ax in enumerate(axes[0]):
        ax.hist(zt[:, k].ravel(), bins=bins, density=True, histtype="step", color="k", label="test (encoder μ)")
        if zg is not None:
            ax.hist(zg[:, k].ravel(), bins=bins, density=True, histtype="step", color="C3",
                    label=f"{full_run} (DiT)")
        ax.set_title(f"latent ch {k}", fontsize=8)
        ax.tick_params(labelsize=6)
    axes[0][0].legend(fontsize=6)
    fig.suptitle(f"standardized latents: {len(zt)} test events" +
                 (f" vs {len(zg)} generated" if zg is not None else " (no generated latents yet)"), fontsize=9)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------- sample side ---
def load_summary(c, case, name=None):
    """summary.json of an evaluation (by name, or the newest sample evaluation)."""
    d = case_dir(c, case) / "eval"
    if name:
        p = d / name / "summary.json"
        return read_json(p) if p.exists() else None
    cands = sorted((p for p in d.glob("*/summary.json") if "rungs" in read_json(p)),
                   key=lambda p: p.stat().st_mtime)
    return read_json(cands[-1]) if cands else None


def load_runs(c, case, names, v1_dir=None):
    """Load several sample runs (and optionally a v1 run) by name."""
    from .sample import load_run, load_v1_run

    out = {}
    for n in names:
        p = case_dir(c, case) / "samples" / n
        if p.exists():
            out[n] = load_run(p)
        else:
            print(f"run {n}: not found under {p.parent}")
    if v1_dir and Path(v1_dir).exists():
        out["V1"] = load_v1_run(v1_dir)
    return out


def run_metrics(c, case, name, rows):
    """event_metrics of every sample of a run, cached by (run, sample indices)."""
    from .evaluate import _pack, _unpack, event_metrics

    key = digest(dict(run=name, idx=[int(r["index"]) for r in rows], n=[len(r["q"]) for r in rows]))[:10]
    f = cache_dir(c, case) / f"metrics_{name.replace('/', '_')}_{key}.npz"
    if f.exists():
        with np.load(f, allow_pickle=False) as d:
            n = int(d["n"])
            return [_unpack({k[len(f"m{i}_"):]: d[k] for k in d.files if k.startswith(f"m{i}_")})
                    for i in range(n)]
    ms = [event_metrics(r["ijk"], r["q"], r["off"], c) for r in rows]
    pack = {"n": np.int64(len(ms))}
    for i, m in enumerate(ms):
        for k, v in _pack(m).items():
            pack[f"m{i}_{k}"] = v
    np.savez_compressed(f, **pack)
    return ms


def event_panel(c, eid, runs, max_points=4000, seed=0):
    """One test event through every paired run: x-z / y-z sums, 3D scatter, profiles, a metrics table."""
    import matplotlib.pyplot as plt

    from .data import load_event
    from .evaluate import event_metrics, truth_metrics
    from .geometry import voxel_center

    grid = int(c["data"]["grid"])
    ranges = np.asarray(c["data"]["ranges"], dtype=np.float64)
    voxel = (ranges[:, 1] - ranges[:, 0]) / grid
    ev = load_event(c["paths"]["processed"], eid)
    cols = [("truth", ev, truth_metrics(c, eid))]
    for name, rows in runs.items():
        s = next((r for r in rows if int(r["eid"]) == int(eid)), None)
        cols.append((name, s, event_metrics(s["ijk"], s["q"], s["off"], c) if s is not None else None))
    n = len(cols)
    fig = plt.figure(figsize=(2.6 * n, 11))
    gs = fig.add_gridspec(4, n, height_ratios=[1, 1, 1.1, 0.9])
    img_t = project(ev["ijk"], ev["q"], grid, 1)
    hi = np.log10(max(img_t.max(), 1e-3))
    lo = hi - 5
    rng = np.random.default_rng(seed)
    for j, (name, s, m) in enumerate(cols):
        title = f"truth · event {eid}" if name == "truth" else name
        ax1, ax2 = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
        ax3 = fig.add_subplot(gs[2, j], projection="3d")
        if s is None:
            for ax in (ax1, ax2, ax3):
                ax.set_title(f"{name}: event not in run", fontsize=8)
                ax.set_axis_off()
            continue
        show_projection(ax1, project(s["ijk"], s["q"], grid, 1), c, f"{title}\nx-z sum", lo, hi, ("x", "z"))
        show_projection(ax2, project(s["ijk"], s["q"], grid, 0), c, "y-z sum", lo, hi, ("y", "z"))
        xyz = voxel_center(np.asarray(s["ijk"]), ranges, grid) + np.asarray(s["off"], dtype=np.float64) * voxel
        q = np.asarray(s["q"], dtype=np.float64)
        pick = np.arange(len(q)) if len(q) <= max_points else rng.choice(len(q), max_points, replace=False)
        ax3.scatter(xyz[pick, 0], xyz[pick, 1], xyz[pick, 2], c=np.log10(np.maximum(q[pick], 1e-3)), s=0.6,
                    cmap="inferno", vmin=lo, vmax=hi, depthshade=False)
        ax3.set_xlim(*ranges[0]), ax3.set_ylim(*ranges[1]), ax3.set_zlim(*ranges[2])
        ax3.set_title(f"3D, {len(pick):,}/{len(q):,} voxels", fontsize=7)
        ax3.tick_params(labelsize=5)
    axl, axr = fig.add_subplot(gs[3, : max(1, n // 2)]), fig.add_subplot(gs[3, max(1, n // 2):] if n > 1 else gs[3, 0])
    for j, (name, s, m) in enumerate(cols):
        if m is None:
            continue
        kw = dict(color=colour(name, j), lw=2.0 if name == "truth" else 1.1, label=name)
        axl.plot(m["longitudinal"], **kw)
        axr.plot(m["radial"], **kw)
    axl.set_title("longitudinal profile (32 slabs, top first)", fontsize=8)
    axr.set_title("radial profile (16 rings)", fontsize=8)
    axl.legend(fontsize=6)
    fig.suptitle(f"test event {eid}: truth and paired chains{data_label(c)}", fontsize=10)
    fig.tight_layout()
    table = [(name, None if m is None else {k: m["scalars"][k] for k in
                                             ("n_active", "q_total", "straight_frac", "halo_n_long",
                                              "halo_big_frac", "top5_share")}) for name, _, m in cols]
    return fig, table


def print_table(table):
    """Print a small table of key event scalars per source."""
    keys = ("n_active", "q_total", "straight_frac", "halo_n_long", "halo_big_frac", "top5_share")
    print(f"{'source':<14}" + "".join(f"{k:>15}" for k in keys))
    for name, m in table:
        if m is None:
            print(f"{name:<14}" + "".join(f"{'-':>15}" for _ in keys))
        else:
            print(f"{name:<14}" + "".join(f"{m[k]:>15.4g}" for k in keys))


def nearest_truth(c, gen_metrics, test_ids):
    """For each generated sample, the test event closest in standardized feature space."""
    from .evaluate import features, truth_metrics

    F = np.stack([features(truth_metrics(c, t)) for t in test_ids])
    mu, sd = np.nanmean(F, 0), np.nanstd(F, 0) + 1e-9
    Z = (F - mu) / sd
    out = []
    for m in gen_metrics:
        d = np.nansum((Z - (features(m) - mu) / sd) ** 2, axis=1)
        out.append(int(test_ids[int(np.argmin(d))]))
    return out


def gallery(c, case, full_name, rows, metrics, n=12, n_ref=300):
    """Unconditional samples next to their nearest test event (a visual reference, NOT a pair)."""
    import matplotlib.pyplot as plt

    from .data import load_event

    grid = int(c["data"]["grid"])
    test = [int(e) for e in meta_for(c)["split"]["test"]][:n_ref]
    rows, metrics = rows[:n], metrics[:n]
    near = nearest_truth(c, metrics, test)
    k = len(rows)
    ncol = 4
    nrow = int(np.ceil(k / (ncol // 2)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for i, (r, t) in enumerate(zip(rows, near)):
        a, b = axes[i // 2, 2 * (i % 2)], axes[i // 2, 2 * (i % 2) + 1]
        a.set_axis_on(), b.set_axis_on()
        img = project(r["ijk"], r["q"], grid, 1)
        hi = np.log10(max(img.max(), 1e-3))
        show_projection(a, img, c, f"{full_name} #{int(r['index'])} (unconditional)", hi - 5, hi)
        ev = load_event(c["paths"]["processed"], t)
        show_projection(b, project(ev["ijk"], ev["q"], grid, 1), c,
                        f"nearest truth: event {t}\n(visual reference only, not paired)", hi - 5, hi)
    fig.suptitle(f"unconditional gallery ({case}/samples/{full_name}); nearest truth among {len(test)} test events "
                 f"in standardized event-feature space{data_label(c)}", fontsize=9)
    fig.tight_layout()
    return fig


def eval_truth(c, summary=None):
    """Truth metrics of the test split (the evaluation's truth set)."""
    from .evaluate import truth_metrics

    test = [int(e) for e in meta_for(c)["split"]["test"]]
    if summary and summary.get("truth_events"):
        test = test[: int(summary["truth_events"])]
    return test, [truth_metrics(c, e) for e in test]


def plot_distributions(c, gen, truth, summary=None, rung=None, keys=None):
    """Histograms of event scalars: truth (all test) vs generated; W1 ratio from the summary."""
    import matplotlib.pyplot as plt

    from .evaluate import SCALARS

    keys = keys or [k for k in SCALARS]
    ncol = 5
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow), squeeze=False)
    for ax in axes.ravel()[len(keys):]:
        ax.set_axis_off()
    sc = (summary or {}).get("rungs", {}).get(rung, {}).get("distribution", {}).get("scalars", {})
    for ax, k in zip(axes.ravel(), keys):
        t = np.array([m["scalars"][k] for m in truth], dtype=np.float64)
        g = np.array([m["scalars"][k] for m in gen], dtype=np.float64)
        t, g = t[np.isfinite(t)], g[np.isfinite(g)]
        logx = k in ("n_active", "q_total") and len(t) and t.min() > 0
        if logx:
            t, g = np.log10(t), np.log10(np.maximum(g, 1e-30))
        both = np.r_[t, g]
        if len(both) == 0:
            continue
        lo, hi = np.quantile(both, [0.005, 0.995])
        bins = np.linspace(lo, hi + 1e-9, 31)
        ax.hist(t, bins=bins, density=True, color="0.6", alpha=0.6, label=f"truth test ({len(t)})")
        ax.hist(g, bins=bins, density=True, histtype="step", color="C3", lw=1.3, label=f"{rung} ({len(g)})")
        r = num(sc.get(k, {}).get("ratio"))
        ax.set_title(("log10 " if logx else "") + k + (f"   W1/R0 = {r:.2f}" if np.isfinite(r) else ""), fontsize=8)
        ax.tick_params(labelsize=6)
    axes[0][0].legend(fontsize=6)
    fig.suptitle(f"event-level distributions: {rung} vs test truth (W1/R0 = 1 is truth-vs-truth level)"
                 f"{data_label(c)}", fontsize=9)
    fig.tight_layout()
    return fig


def plot_profiles(summary, truth):
    """Mean longitudinal and radial profiles: truth vs each run."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, key, tk in ((axes[0], "long", "longitudinal"), (axes[1], "rad", "radial")):
        T = np.stack([m[tk] for m in truth])
        mu, sd = T.mean(0), T.std(0)
        x = np.arange(len(mu))
        ax.fill_between(x, mu - sd, mu + sd, color="0.8", label="truth ±1σ")
        ax.plot(x, mu, color="k", lw=2, label="truth mean")
        for i, (name, r) in enumerate(summary["rungs"].items()):
            p = r["distribution"]["profiles"]
            ax.plot(x, [num(v) for v in p[f"{key}_mean"]], color=colour(name, i), lw=1.1, label=name)
        ax.set_title(("longitudinal (32 slabs, top first)" if key == "long" else "radial (16 rings)") +
                     ": mean photon fraction", fontsize=9)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return fig


def plot_c2st(summary):
    """C2ST AUC per run with its confidence interval (0.5 = indistinguishable)."""
    import matplotlib.pyplot as plt

    names = list(summary["rungs"])
    vals = [summary["rungs"][n]["distribution"]["c2st"] for n in names]
    names = ["R0 (truth B)"] + names
    vals = [summary["R0"]["c2st"]] + vals
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(names)), 3.2))
    for i, v in enumerate(vals):
        auc, lo, hi = num(v.get("auc")), num(v.get("lo")), num(v.get("hi"))
        if not np.isfinite(auc):
            continue
        err = [[auc - lo], [hi - auc]] if np.isfinite(lo) and np.isfinite(hi) else None
        ax.errorbar(i, auc, yerr=err, fmt="o", color=colour(names[i], i), capsize=3)
    ax.axhline(0.5, color="k", lw=0.8)
    ax.axhline(0.65, color="C3", ls=":", lw=0.8, label="acceptance 0.65")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("C2ST AUC (0.5 ideal)")
    ax.legend(fontsize=7)
    ax.set_title("classifier two-sample test vs test truth, 95% bootstrap CI", fontsize=9)
    fig.tight_layout()
    return fig


def sdedit_curve(summary, left="sa_recon", right="full", right_sigma=80.0):
    """Points (sigma, rung) ordered from per-event (sa_recon, sigma 0) to unconditional (full)."""
    pts = []
    for name, r in summary["rungs"].items():
        st = r.get("settings", {})
        if r.get("chain") == "sdedit" and st.get("sigma_start") is not None and st.get("t2") is not False:
            pts.append((float(st["sigma_start"]), name))
    pts.sort()
    if left in summary["rungs"]:
        pts.insert(0, (0.0, left))
    if right in summary["rungs"]:
        pts.append((float(right_sigma), right))
    return pts


def plot_sdedit(summary, left="sa_recon", right="full"):
    """Metric vs sigma_s: sa_recon (0) -> sdedit -> full (80).  Paired metrics stop before `full`."""
    import matplotlib.pyplot as plt

    pts = sdedit_curve(summary, left, right)
    if len(pts) < 2:
        print("not enough sdedit / sa_recon / full rungs in this summary for the curve")
        return None
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    xs = np.array([max(s, 0.1) for s, _ in pts])

    def paired(key):
        """Mean and CI of one paired metric along the sigma curve."""
        m, lo, hi = [], [], []
        for _, n in pts:
            p = summary["rungs"][n].get("paired", {}).get(key)
            m.append(np.nan if p is None else num(p["mean"]))
            lo.append(np.nan if p is None else num(p["lo"]))
            hi.append(np.nan if p is None else num(p["hi"]))
        return np.array(m), np.array(lo), np.array(hi)

    for ax, key, title in ((axes[0], "long_l1", "paired longitudinal L1"),
                           (axes[1], "q_rel_err", "paired |Q ratio − 1|")):
        m, lo, hi = paired(key)
        ok = np.isfinite(m)
        ax.errorbar(xs[ok], m[ok], yerr=[m[ok] - lo[ok], hi[ok] - m[ok]], fmt="o-", capsize=3)
        ax.set_title(title + " (95% CI)", fontsize=9)
    oo = np.array([num(summary["rungs"][n].get("own_other_long")) for _, n in pts])
    axes[2].plot(xs, oo, "o-")
    axes[2].axhline(1.0, color="k", lw=0.8)
    axes[2].set_title("own / other longitudinal L1 (1 = no per-event information)", fontsize=9)
    auc = np.array([num(summary["rungs"][n]["distribution"]["c2st"]["auc"]) for _, n in pts])

    def median_ratio(n):
        """Median W1 / R0 ratio over all scalars of one run."""
        r = np.array([num(v["ratio"]) for v in summary["rungs"][n]["distribution"]["scalars"].values()])
        return float(np.median(r[np.isfinite(r)])) if np.isfinite(r).any() else np.nan

    w1r = np.array([median_ratio(n) for _, n in pts])
    axes[3].plot(xs, auc, "o-", label="C2ST AUC")
    axes[3].plot(xs, w1r, "s--", label="median W1/R0")
    axes[3].legend(fontsize=7)
    axes[3].set_title("distribution level", fontsize=9)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s:g}\n{n}" for (s, n) in pts], fontsize=6)
        ax.set_xlabel("σ_s (left: sa_recon at 0, right: full at 80)", fontsize=7)
    fig.tight_layout()
    return fig
