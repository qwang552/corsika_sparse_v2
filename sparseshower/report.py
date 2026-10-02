"""Sample report behind notebooks/report.ipynb: every sample-side figure and metric in one folder.

    <case>/samples/<name>/            (a timestamp is added if it exists; nothing is overwritten)
        NN_<figure>.png               numbered in notebook order
        *.csv, summary.json, passfail.md
        index.md                      every file with a one-line caption

Three parts: the metrics (everything evaluate computes plus voxel overlap and shape
ratios, recomputed here with a process pool, so no evaluate run is needed), static
renders of the chain_page view and charts, and chains traced step by step with the
generators of sample.run_one.  The first cell of notebooks/report.ipynb lists the
contents of each part.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .common import digest, read_json
from .data import load_event
from .loader import case_dir, meta_for

EXTRA_KEYS = ("n_ratio", "q_ratio", "iou_192", "precision_192", "recall_192", "iou_96", "iou_48",
              "precision_48", "recall_48", "q_recall_192", "q_recall_48", "centroid_shift_cm",
              "zmax_shift_m", "r50_ratio", "r90_ratio", "zext_ratio", "box_d1_diff", "box_d2_diff",
              "box_d3_diff", "box_d4_diff", "cells48_ratio", "cells96_ratio", "cells192_ratio")


# ------------------------------------------------------------------ output ---
class Report:
    """Figure / table sink: figures as NN_<name>.png in <case>/samples/<name>/, tables as CSV
    or text, all listed in index.md."""

    def __init__(self, c, case, name="report", dpi=150, formats=("png",)):
        d = case_dir(c, case) / "samples" / name
        if d.exists():
            d = d.with_name(f"{name}_{time.strftime('%Y%m%d-%H%M%S')}")
        d.mkdir(parents=True)
        self.dir, self.dpi, self.formats, self.items, self.n = d, int(dpi), tuple(formats), [], 0
        self.synthetic = bool(c.get("synthetic"))

    def _name(self, name):
        self.n += 1
        return f"{self.n:02d}_{name}"

    def fig(self, fig, name, caption=""):
        """Save a figure as the next numbered file and record its caption."""
        import matplotlib.pyplot as plt

        if fig is None:
            return None
        stem = self._name(name)
        for fmt in self.formats:
            fig.savefig(self.dir / f"{stem}.{fmt}", dpi=self.dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        self.items.append((f"{stem}.{self.formats[0]}", caption))
        plt.show()
        plt.close(fig)
        return stem

    def csv(self, rows, name, caption=""):
        """Save rows as a CSV and record its caption."""
        if not rows:
            return None
        keys = list(dict.fromkeys(k for r in rows for k in r))
        path = self.dir / f"{name}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: _fmt_cell(r.get(k)) for k in keys})
        self.items.append((path.name, caption))
        return path

    def text(self, text, name, caption=""):
        """Save a text file and record its caption."""
        path = self.dir / name
        path.write_text(text)
        self.items.append((path.name, caption))
        return path

    def index(self, title="Sample report"):
        """Write index.md: every saved file with its caption."""
        lines = [f"# {title}", "", f"made {time.strftime('%Y-%m-%d %H:%M')}" +
                 (" · SYNTHETIC data (smoke config)" if self.synthetic else ""), ""]
        lines += [f"- `{f}`: {cap}" for f, cap in self.items]
        (self.dir / "index.md").write_text("\n".join(lines) + "\n")
        return self.dir / "index.md"


def _fmt_cell(v):
    """Format a CSV cell (NaN -> empty, lists / dicts -> JSON)."""
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.6g}"
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return v


# -------------------------------------------------------------- inventory ---
def load_runs(c, case, names, v1_dir=None):
    """{name: (rows, spec, steps, index_rows)}; missing runs are reported and skipped."""
    from .sample import load_run, load_v1_run

    out = {}
    for n in names:
        d = case_dir(c, case) / "samples" / n
        if not (d / "run.json").exists():
            print(f"run {n}: not found ({d})")
            continue
        idx = []
        for f in sorted(d.glob("index_shard*.jsonl")):
            for line in open(f):
                try:
                    idx.append(json.loads(line))
                except ValueError:
                    pass
        steps = read_json(d / "checkpoints.json") if (d / "checkpoints.json").exists() else {}
        out[n] = (load_run(d), read_json(d / "run.json"), steps, idx)
    if v1_dir and Path(v1_dir).exists():
        out["V1"] = (load_v1_run(v1_dir), dict(chain="v1", source=str(v1_dir)), {}, [])
    return out


def inventory(runs):
    """One row per run: settings, sample count, checkpoint steps, time per sample, T2 scale."""
    rows = []
    for n, (rows_, spec, steps, idx) in runs.items():
        secs = [float(r["seconds"]) for r in idx if "seconds" in r and not r.get("failed")]
        t2s = [float(r["t2_scale_median"]) for r in idx if "t2_scale_median" in r]
        rows.append(dict(run=n, chain=spec.get("chain"), samples=len(rows_),
                         paired_events=len({int(r["eid"]) for r in rows_ if int(r["eid"]) >= 0}),
                         failed=sum(1 for r in idx if r.get("failed")), sigma=spec.get("sigma_start"),
                         t2=spec.get("t2"), occ_threshold=spec.get("occ_threshold"),
                         heun_steps=spec.get("heun_steps"), checkpoint=spec.get("checkpoint"),
                         steps=" ".join(f"{k}:{v}" for k, v in steps.items()),
                         sec_median=float(np.median(secs)) if secs else float("nan"),
                         t2_scale_median=float(np.median(t2s)) if t2s else float("nan"),
                         voxels_median=float(np.median([len(r["q"]) for r in rows_])) if rows_ else float("nan")))
    return rows


# ------------------------------------------------------ metrics (parallel) ---
def _metrics_job(args):
    """Worker job: event metrics of one sample."""
    ijk, q, off, c = args
    from .evaluate import event_metrics

    return event_metrics(ijk, q, off, c)


def _truth_job(args):
    """Worker job: metrics of one truth event."""
    c, e = args
    from .evaluate import truth_metrics

    return truth_metrics(c, e)


def _pool_map(fn, jobs, workers, label, verbose=True):
    """Map a job over a process pool (or serially with one worker), with progress output."""
    out = [None] * len(jobs)
    if not jobs:
        return out
    t0 = time.time()
    if int(workers) <= 1:
        for i, j in enumerate(jobs):
            out[i] = fn(j)
            if verbose and (i + 1) % 25 == 0:
                print(f"  {label}: {i + 1}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
        return out
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(int(workers), mp_context=mp.get_context("fork")) as ex:
        for i, m in enumerate(ex.map(fn, jobs, chunksize=1)):
            out[i] = m
            if verbose and (i + 1) % 25 == 0:
                print(f"  {label}: {i + 1}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    return out


def run_metrics(c, case, name, rows, workers=1, verbose=True):
    """evaluate.event_metrics of every row; same cache file (name and format) as viz.run_metrics."""
    from .evaluate import _pack, _unpack
    from .viz import cache_dir

    key = digest(dict(run=name, idx=[int(r["index"]) for r in rows], n=[len(r["q"]) for r in rows]))[:10]
    f = cache_dir(c, case) / f"metrics_{name.replace('/', '_')}_{key}.npz"
    if f.exists():
        with np.load(f, allow_pickle=False) as d:
            n = int(d["n"])
            return [_unpack({k[len(f"m{i}_"):]: d[k] for k in d.files if k.startswith(f"m{i}_")})
                    for i in range(n)]
    ms = _pool_map(_metrics_job, [(r["ijk"], r["q"], r["off"], c) for r in rows], workers, name, verbose)
    pack = {"n": np.int64(len(ms))}
    for i, m in enumerate(ms):
        for k, v in _pack(m).items():
            pack[f"m{i}_{k}"] = v
    np.savez_compressed(f, **pack)
    return ms


def truth_set(c, eids, workers=1, verbose=True):
    """Truth metrics of the given events, as {eid: metrics}."""
    ms = _pool_map(_truth_job, [(c, int(e)) for e in eids], workers, "truth", verbose)
    return dict(zip([int(e) for e in eids], ms))


def _keys(ijk, grid):
    """Integer keys of voxel indices (for set overlap)."""
    ijk = np.asarray(ijk, dtype=np.int64)
    return (ijk[:, 2] * grid + ijk[:, 1]) * grid + ijk[:, 0]


def _radii(ijk, q, c, fracs=(0.5, 0.9)):
    """Photon centroid, the radii (around the vertical line through it) that hold 50% and 90%
    of the photons, and the z extent of the central 80% of the photons."""
    from .geometry import voxel_center

    grid = int(c["data"]["grid"])
    p = voxel_center(np.asarray(ijk, dtype=np.int64), np.asarray(c["data"]["ranges"], dtype=np.float64), grid)
    q = np.asarray(q, dtype=np.float64)
    w = q / max(q.sum(), 1e-30)
    cen = w @ p
    r = np.hypot(p[:, 0] - cen[0], p[:, 1] - cen[1])
    o = np.argsort(r)
    cum = np.cumsum(w[o])
    rad = [float(r[o][min(len(o) - 1, np.searchsorted(cum, f))]) for f in fracs]
    oz = np.argsort(p[:, 2])
    cz = np.cumsum(w[oz])
    zext = float(p[oz][min(len(oz) - 1, np.searchsorted(cz, 0.9)), 2] - p[oz][min(len(oz) - 1, np.searchsorted(cz, 0.1)), 2])
    return cen, rad, zext


def extra_paired(row, ev, gm, tm, c):
    """Per-event metrics beyond evaluate.paired_errors (row: generated sample, ev: truth event)."""
    grid, base = int(c["data"]["grid"]), int(c["field"]["base_grid"])
    gi, ti = np.asarray(row["ijk"], dtype=np.int64), np.asarray(ev["ijk"], dtype=np.int64)
    gq, tq = np.asarray(row["q"], dtype=np.float64), np.asarray(ev["q"], dtype=np.float64)
    out = dict(n_ratio=len(gi) / max(len(ti), 1), q_ratio=gq.sum() / max(tq.sum(), 1e-30))
    for f, tag in ((1, "192"), (2, "96"), (grid // base, "48")):
        a = np.unique(_keys(gi // f, grid // f))
        b = np.unique(_keys(ti // f, grid // f))
        inter = len(np.intersect1d(a, b, assume_unique=True))
        out[f"iou_{tag}"] = inter / max(len(a) + len(b) - inter, 1)
        out[f"precision_{tag}"] = inter / max(len(a), 1)
        out[f"recall_{tag}"] = inter / max(len(b), 1)
        out[f"cells{tag}_ratio"] = len(a) / max(len(b), 1)
    for f, tag in ((1, "192"), (grid // base, "48")):
        hit = np.isin(_keys(ti // f, grid // f), np.unique(_keys(gi // f, grid // f)))
        out[f"q_recall_{tag}"] = float(tq[hit].sum() / max(tq.sum(), 1e-30))
    cg, rg, zg = _radii(gi, gq, c)
    ct, rt, zt = _radii(ti, tq, c)
    slab = (float(c["data"]["ranges"][2][1]) - float(c["data"]["ranges"][2][0])) / len(tm["longitudinal"])
    out.update(centroid_shift_cm=float(np.linalg.norm(cg - ct) * 100),
               zmax_shift_m=float((np.argmax(gm["longitudinal"]) - np.argmax(tm["longitudinal"])) * slab),
               r50_ratio=rg[0] / max(rt[0], 1e-9), r90_ratio=rg[1] / max(rt[1], 1e-9),
               zext_ratio=zg / max(zt, 1e-9))
    for k in (1, 2, 3, 4):
        out[f"box_d{k}_diff"] = gm["scalars"][f"box_d{k}"] - tm["scalars"][f"box_d{k}"]
    return out


def compute_summary(c, case, runs, n_truth=256, max_paired=128, n_full=256, workers=8, verbose=True):
    """The evaluate summary (same structure, so viz and evaluate.passfail work on it) plus the extra
    paired metrics.  Nothing is written to eval/.  Returns (summary, per_event, truth_list, gen_metrics)."""
    from .evaluate import (PAIRED_KEYS, bootstrap_mean, distribution, memorization, own_other,
                           paired_errors, passfail)

    say = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(int(c["eval"]["seed"]))
    test = [int(e) for e in (meta_for(c)["split"]["test"] or meta_for(c)["split"]["val"])][: int(n_truth)]
    t0 = time.time()
    truth = truth_set(c, test, workers, verbose)
    say(f"truth metrics: {len(test)} test events ({time.time() - t0:.0f} s)")
    order = rng.permutation(len(test))
    A = [truth[test[i]] for i in order[: len(test) // 2]]
    B = [truth[test[i]] for i in order[len(test) // 2:]]
    summary = dict(case=case, name="report", truth_events=len(test), rungs={},
                   R0=distribution(B, A, B, rng, c))
    per_event, gen_metrics = [], {}
    root = Path(c["paths"]["processed"])
    for name, (rows, spec, steps, idx) in runs.items():
        chain = spec.get("chain", "?")
        rows = [r for r in rows if int(r["eid"]) >= 0][: int(max_paired)] if chain != "full" else rows[: int(n_full)]
        if not rows:
            continue
        t0 = time.time()
        ms = run_metrics(c, case, name, rows, workers, verbose)
        for r, m in zip(rows, ms):
            r["m"] = m
        gen_metrics[name] = (rows, ms)
        entry = dict(chain=chain, settings=spec, n=len(rows), steps=steps)
        if chain != "full":
            need = [int(r["eid"]) for r in rows if int(r["eid"]) not in truth]
            truth.update(truth_set(c, need, workers, False))
            errs = []
            for r in rows:
                e = int(r["eid"])
                ev = load_event(root, e)
                errs.append(dict(run=name, eid=e, **paired_errors(r["m"], truth[e]),
                                 **extra_paired(r, ev, r["m"], truth[e], c)))
            per_event += errs
            nb = int(c["eval"]["bootstrap"])
            entry["paired"] = {k: bootstrap_mean([x[k] for x in errs], nb, rng) for k in PAIRED_KEYS + EXTRA_KEYS}
            entry["own_other_long"] = own_other(rows, {int(r["eid"]): truth[int(r["eid"])] for r in rows}, rng)
        entry["distribution"] = distribution(ms, A, B, rng, c)
        if chain == "full":
            entry["memorization"] = memorization(c, case, rows, rng)
        summary["rungs"][name] = entry
        say(f"{name}: {len(rows)} samples ({time.time() - t0:.0f} s)")
    summary["passfail"] = passfail(summary, c)
    return summary, per_event, A + B, gen_metrics


def paired_table(summary):
    """Rows: metric; columns: per paired run the mean and a '<run> CI' column '[lo, hi]'."""
    from .evaluate import PAIRED_KEYS

    names = [n for n, r in summary["rungs"].items() if "paired" in r]
    rows = []
    for k in PAIRED_KEYS + EXTRA_KEYS:
        row = dict(metric=k)
        for n in names:
            v = summary["rungs"][n]["paired"].get(k)
            row[n] = float("nan") if v is None else v["mean"]
            row[n + " CI"] = "" if v is None else f"[{v['lo']:.4g}, {v['hi']:.4g}]"
        rows.append(row)
    rows.append(dict(metric="own_other_long", **{n: summary["rungs"][n].get("own_other_long") for n in names}))
    return rows


def distribution_table(summary):
    """Rows: W1/R0 per event scalar, then C2ST AUC, Fréchet and memorization ratio;
    columns: R0 and each run."""
    names = list(summary["rungs"])
    rows = []
    keys = list(summary["R0"]["scalars"])
    for k in keys:
        row = dict(metric=f"W1/R0 {k}", R0=1.0)
        row.update({n: summary["rungs"][n]["distribution"]["scalars"][k]["ratio"] for n in names})
        rows.append(row)
    rows.append(dict(metric="C2ST AUC", R0=summary["R0"]["c2st"]["auc"],
                     **{n: summary["rungs"][n]["distribution"]["c2st"]["auc"] for n in names}))
    rows.append(dict(metric="Frechet", R0=summary["R0"]["frechet"],
                     **{n: summary["rungs"][n]["distribution"]["frechet"] for n in names}))
    rows.append(dict(metric="memorization ratio", R0=float("nan"),
                     **{n: (summary["rungs"][n].get("memorization") or {}).get("ratio", float("nan")) for n in names}))
    return rows


# ----------------------------------------------------------- plots: metrics ---
def _colour(name, i):
    """Plot colour of a run (same as viz)."""
    from .viz import colour

    return colour(name, i)


def plot_table(rows, cols, title, fmt="{:.3g}", good=None, figsize=None):
    """A table as a figure; good(metric, value, column) -> True/False/None colours the cell."""
    import matplotlib.pyplot as plt

    cell = [[r["metric"]] + [("–" if r.get(k) is None or (isinstance(r.get(k), float) and not np.isfinite(r[k]))
                              else (fmt.format(r[k]) if isinstance(r.get(k), (int, float)) else str(r[k])))
                             for k in cols] for r in rows]
    fig, ax = plt.subplots(figsize=figsize or (2.0 + 1.35 * len(cols), 0.32 * len(rows) + 1.0))
    ax.set_axis_off()
    t = ax.table(cellText=cell, colLabels=["metric"] + list(cols), loc="center", cellLoc="right", colLoc="right")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.25)
    for (i, j), cl in t.get_celld().items():
        if i == 0:
            cl.set_facecolor("#e8edf3")
            cl.set_text_props(weight="bold")
        elif j == 0:
            cl.set_text_props(ha="left")
            cl._loc = "left"
        elif good is not None:
            v = rows[i - 1].get(cols[j - 1])
            ok = good(rows[i - 1]["metric"], v, cols[j - 1]) if isinstance(v, (int, float)) else None
            if ok is True:
                cl.set_facecolor("#dff3e6")
            elif ok is False:
                cl.set_facecolor("#fbe3e0")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_paired_boxes(per_event, names, keys=None, title=""):
    """Box plots of the per-event paired metrics, all runs in one figure."""
    import matplotlib.pyplot as plt

    keys = keys or [("n_ratio", 1.0), ("q_ratio", 1.0), ("long_l1", 0.0), ("rad_l1", 0.0), ("iou_192", 1.0),
                    ("q_recall_192", 1.0), ("iou_48", 1.0), ("w1_logq", 0.0), ("w1_lin", 0.0),
                    ("straight_diff", 0.0), ("centroid_shift_cm", 0.0), ("zmax_shift_m", 0.0)]
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.9 * nrow), squeeze=False)
    for ax in axes.ravel()[len(keys):]:
        ax.set_axis_off()
    for ax, (k, ideal) in zip(axes.ravel(), keys):
        data = [np.array([r[k] for r in per_event if r["run"] == n and np.isfinite(r[k])]) for n in names]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True, flierprops=dict(ms=2))
        for i, p in enumerate(bp["boxes"]):
            p.set_facecolor(_colour(names[i], i))
            p.set_alpha(0.55)
        ax.axhline(ideal, color="k", lw=0.8, ls=":")
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
        ax.set_title(f"{k}  (ideal {ideal:g})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_vs_size(per_event, truth_by_eid, names, title=""):
    """Paired metrics against the size of the truth event."""
    import matplotlib.pyplot as plt

    keys = [("n_ratio", 1.0, True), ("q_ratio", 1.0, True), ("long_l1", 0.0, False), ("iou_192", 1.0, False)]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.4))
    for ax, (k, ideal, logy) in zip(axes, keys):
        for i, n in enumerate(names):
            rr = [r for r in per_event if r["run"] == n]
            x = [truth_by_eid[r["eid"]]["scalars"]["n_active"] for r in rr]
            ax.scatter(x, [r[k] for r in rr], s=9, alpha=0.7, color=_colour(n, i), label=n)
        ax.axhline(ideal, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("truth active voxels (event size)")
        ax.set_title(k, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_levels(per_event, names, title=""):
    """Cell-count ratio and IoU with truth at 48^3 / 96^3 / 192^3 per run."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    tags = ["48", "96", "192"]
    for i, n in enumerate(names):
        rr = [r for r in per_event if r["run"] == n]
        if not rr:
            continue
        for ax, key in ((axes[0], "cells{}_ratio"), (axes[1], "iou_{}")):
            v = np.array([[r[key.format(t)] for t in tags] for r in rr])
            med, lo, hi = np.median(v, 0), np.quantile(v, 0.25, 0), np.quantile(v, 0.75, 0)
            ax.errorbar(np.arange(3) + 0.05 * i, med, yerr=[med - lo, hi - med], fmt="o-", capsize=3,
                        color=_colour(n, i), label=n)
    axes[0].axhline(1, color="k", lw=0.8, ls=":")
    axes[0].set_title("active cells per level, gen / truth (median, IQR)", fontsize=9)
    axes[1].set_title("overlap with truth per level: IoU (median, IQR)", fontsize=9)
    for ax in axes:
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{t}³" for t in tags])
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_timing(runs, title=""):
    """Seconds per sample and T2 scale factor per run."""
    import matplotlib.pyplot as plt

    names = [n for n, v in runs.items() if v[3]]
    if not names:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    secs = [[float(r["seconds"]) for r in runs[n][3] if "seconds" in r and not r.get("failed")] for n in names]
    t2 = [[float(r["t2_scale_median"]) for r in runs[n][3] if "t2_scale_median" in r] for n in names]
    for ax, data, lab in ((axes[0], secs, "seconds per sample"), (axes[1], t2, "T2 scale (median per sample)")):
        ok = [i for i, d in enumerate(data) if d]
        if ok:
            ax.boxplot([data[i] for i in ok], widths=0.6)
            ax.set_xticks(range(1, len(ok) + 1))
            ax.set_xticklabels([names[i] for i in ok], rotation=25, ha="right", fontsize=7)
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_yscale("log")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# -------------------------------------------------------- plots: web charts ---
def plot_web_charts(gen_metrics, truth_by_eid, events, names, full=None, title=""):
    """The six charts of chain_page, from evaluate's per-event quantile vectors (each event weighs equally)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 7.6))
    T = [truth_by_eid[e] for e in events if e in truth_by_eid]
    series = [("truth", T, "k", 2.2, "-")]
    for i, n in enumerate(names):
        rows, ms = gen_metrics.get(n, ([], []))
        pick = [m for r, m in zip(rows, ms) if int(r["eid"]) in set(events)]
        if pick:
            series.append((n, pick, _colour(n, i + 1), 1.4, "-"))
    if full and full in gen_metrics:
        series.append((f"{full} (unpaired)", gen_metrics[full][1], _colour(full, 0), 1.4, "--"))
    pool = np.concatenate([m["q_logq"] for _, ms, *_ in series for m in ms])
    lo, hi = np.quantile(pool, [0.002, 0.999])
    specs = [(axes[0, 0], "q_logq", np.linspace(lo, hi, 49), False, "log(Q + ε) per voxel"),
             (axes[0, 1], "q_nn", np.linspace(0.015, 0.09, 41), True, "nearest-neighbour distance [m]"),
             (axes[0, 2], "q_lin", np.linspace(0.3, 1.0, 36), False, "local linearity")]
    for ax, key, edges, logy, lab in specs:
        for name, ms, col, lw, ls in series:
            v = np.concatenate([m[key] for m in ms])
            h, _ = np.histogram(v, bins=edges)
            h = h / max(h.sum(), 1)
            ax.step(edges[:-1], h, where="post", color=col, lw=lw, ls=ls, label=name)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel("share")
        ax.grid(alpha=0.25)
    for ax, key, logy, lab in ((axes[1, 0], "longitudinal", False, "z slab (top of box first)"),
                               (axes[1, 1], "radial", True, "ring (inner → outer)")):
        for name, ms, col, lw, ls in series:
            v = np.median(np.stack([m[key] for m in ms]), 0)
            ax.plot(np.arange(len(v)), np.where(v > 0, v, np.nan) if logy else v, color=col, lw=lw, ls=ls, label=name)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel("photon share (median over events)")
        ax.grid(alpha=0.25)
    ax = axes[1, 2]
    for name, ms, col, lw, ls in series:
        v = np.nanmedian(np.array([[m["scalars"][f"box_d{k}"] for k in (1, 2, 3, 4)] for m in ms]), 0)
        ax.plot([1, 2, 3, 4], v, "o-", color=col, lw=lw, ls=ls, label=name)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1→2", "2→4", "4→8", "8→16"])
    ax.set_xlabel("between coarsening factors")
    ax.set_ylabel("box-counting dimension (median)")
    ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------- plots: 3D renders ---
def render(ijk, ci, grid, lut, yaw=-0.6, pitch=0.3, size=520, zoom=1.0, gain=1.8, ps=2):
    """numpy port of chain_page's renderer: additive light along the line of sight, exponential tone map."""
    W = H = int(size)
    h = grid / 2
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    sc = zoom * min(W, H) / (grid * 1.85)

    def proj(x, y, z):
        """Project 3D grid coordinates to image pixels for the current view."""
        x, y, z = x - h, y - h, z - h
        x1, y1 = x * cy - y * sy, x * sy + y * cy
        return W / 2 + x1 * sc, H / 2 - (y1 * sp + z * cp) * sc

    ijk = np.asarray(ijk, dtype=np.float64).reshape(-1, 3)
    u, v = proj(ijk[:, 0], ijk[:, 1], ijk[:, 2])
    u, v = np.trunc(u).astype(np.int64), np.trunc(v).astype(np.int64)
    col = lut[np.asarray(ci, dtype=np.int64)]
    ps = max(1, int(round(ps * max(1.0, 192 / grid))))       # as the page: coarser grids get bigger points
    buf = np.zeros((3, H * W))
    for a in range(ps):
        for b in range(ps):
            uu, vv = u + a, v + b
            ok = (uu >= 0) & (vv >= 0) & (uu < W) & (vv < H)
            k = vv[ok] * W + uu[ok]
            for ch in range(3):
                buf[ch] += np.bincount(k, weights=col[ok, ch], minlength=H * W)
    img = np.stack([5 + 250 * (1 - np.exp(-gain * buf[0])), 7 + 248 * (1 - np.exp(-gain * buf[1])),
                    10 + 245 * (1 - np.exp(-gain * buf[2]))], -1).reshape(H, W, 3) / 255.0
    return img, proj


def _draw_box(ax, proj, grid):
    """Draw the edges of the grid box in the current projection."""
    G = grid
    E = [[0, 0, 0, G, 0, 0], [0, G, 0, G, G, 0], [0, 0, G, G, 0, G], [0, G, G, G, G, G], [0, 0, 0, 0, G, 0],
         [G, 0, 0, G, G, 0], [0, 0, G, 0, G, G], [G, 0, G, G, G, G], [0, 0, 0, 0, 0, G], [G, 0, 0, G, 0, G],
         [0, G, 0, 0, G, G], [G, G, 0, G, G, G]]
    for e in E:
        a, b = proj(*e[:3]), proj(*e[3:])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=(0.63, 0.67, 0.74, 0.35), lw=0.7)
    o = proj(0, 0, 0)
    for lab, p, cl in (("x", (G * 1.1, 0, 0), "#e66767"), ("y", (0, G * 1.1, 0), "#3fbf8a"), ("z", (0, 0, G * 1.1), "#5b9cf0")):
        q = proj(*p)
        ax.plot([o[0], q[0]], [o[1], q[1]], color=cl, lw=1.2)
        ax.text(q[0] + 4, q[1] - 4, lab, color=cl, fontsize=8)


def colour_index(q, lo, hi, q_eps):
    """Q -> colour-table index 0..255 of log(Q + q_eps) on [lo, hi]."""
    lq = np.log(np.asarray(q, dtype=np.float64) + float(q_eps))
    return np.clip(np.rint((lq - lo) / (hi - lo) * 255), 0, 255).astype(np.int64)


def lq_range(qs, q_eps):
    """Shared logQ colour range over several voxel sets (robust quantiles)."""
    pool = np.concatenate([np.log(np.asarray(q, dtype=np.float64) + float(q_eps)) for q in qs])
    lo, hi = np.quantile(pool, [0.002, 0.999])
    return float(lo), float(hi if hi > lo else lo + 1)


def plot_3d(c, panels, lq, views=(("perspective", -0.6, 0.3), ("side x–z", 0.0, 0.0)), title="", size=520,
            gain=1.8, ps=2):
    """panels: [(label, subtitle, ijk, q) or (label, subtitle, None, None) for a missing one]."""
    import matplotlib.pyplot as plt

    from .chain_page import lut256

    grid = int(c["data"]["grid"])
    lut = lut256()
    q_eps = float(c["data"]["q_eps"])
    n = len(panels)
    fig, axes = plt.subplots(len(views), n, figsize=(3.3 * n, 3.55 * len(views)), squeeze=False)
    fig.patch.set_facecolor("#0f1216")
    for j, (label, sub, ijk, q) in enumerate(panels):
        for i, (vname, yaw, pitch) in enumerate(views):
            ax = axes[i, j]
            ax.set_facecolor("#05070a")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if ijk is None:
                ax.text(0.5, 0.5, "not in this run", color="#9aa4b1", ha="center", va="center", transform=ax.transAxes)
            else:
                img, proj = render(ijk, colour_index(q, lq[0], lq[1], q_eps), grid, lut, yaw, pitch, size,
                                   gain=gain, ps=ps)
                ax.imshow(img, interpolation="nearest")
                _draw_box(ax, proj, grid)
                ax.set_xlim(0, size)
                ax.set_ylim(size, 0)
            if i == 0:
                ax.set_title(f"{label}\n{sub}", color="#e6e9ee", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(vname, color="#9aa4b1", fontsize=8)
    fig.suptitle(title, color="#e6e9ee", fontsize=11)
    fig.tight_layout()
    return fig


def event_panels(c, eid, runs, names):
    """3D panels of one test event: truth and every run's sample of it."""
    ev = load_event(Path(c["paths"]["processed"]), eid)
    tq = float(np.sum(ev["q"]))
    panels = [(f"truth · event {eid}", f"{len(ev['q']):,} voxels · Q {tq:.3g}", ev["ijk"], ev["q"])]
    for n in names:
        rows = runs[n][0] if n in runs else []
        x = next((r for r in rows if int(r["eid"]) == int(eid)), None)
        if x is None:
            panels.append((n, "", None, None))
        else:
            panels.append((n, f"{len(x['q']):,} vox ({len(x['q']) / len(ev['q']):.3f}×) · Q {np.sum(x['q']) / tq:.3f}×",
                           x["ijk"], x["q"]))
    return panels


# -------------------------------------------------------- chain tracing ---
def load_models(c, case, checkpoint="best", device=None):
    """Load all chain models once for step-by-step tracing."""
    import torch

    from .sample import load_chain

    dev = torch.device(device) if isinstance(device, str) else device
    return load_chain(c, case, "full", checkpoint, dev)


def run_seed(c, case, run, eid=None, index=None):
    """(seed, sigma, t2, index) that reproduce one sample of an existing run."""
    d = case_dir(c, case) / "samples" / run
    spec = read_json(d / "run.json")
    if eid is not None and spec.get("events"):
        index = [int(e) for e in spec["events"]].index(int(eid))
    index = int(index or 0)
    return dict(seed=int(spec["seed"]) + index, sigma=spec.get("sigma_start"), t2=spec.get("t2"), index=index,
                path=d / f"sample_{index:05d}.npz", chain=spec.get("chain"))


def trace_chain(ch, chain, eid=None, sigma=None, seed=0, t2=None, n_show=6, ref_eid=None):
    """One sample of sa_recon / sdedit / full, step by step (the generators of sample.run_one).

    Returns the keys of chain_steps.run_steps; for `full` the reference is `ref_eid`
    (e.g. the nearest test event), which is NOT a partner of the sample."""
    import torch

    from .attr import sample_attr, t2_rescale
    from .diffusion import heun_sample
    from .dit import dit_denoiser
    from .fields import occupied_cells
    from .samples import level_grids
    from .structure import sample_level

    c, dev, store = ch["c"], ch["device"], ch["store"]
    t2 = bool(c["sample"]["t2"]) if t2 is None else bool(t2)
    gens = [torch.Generator(device=dev).manual_seed(int(seed) * 7919 + k) for k in range(3)]
    secs = {}
    ref = int(eid) if chain != "full" else (None if ref_eid is None else int(ref_eid))
    z0 = store.standardize(store.posterior(ref)[0]) if chain != "full" else None
    L = int(c["field"]["base_grid"]) // (2 ** (len(c["ae"]["widths"]) - 1))
    shape = (1, int(c["ae"]["latent_channels"]), L, L, L)
    calls = []
    t = time.time()
    if chain == "sa_recon":
        zhat, sig_t, x_t, d_t, z_start = z0, np.zeros(0), np.zeros((0,) + shape[1:]), np.zeros((0,) + shape[1:]), z0
    else:
        den = dit_denoiser(ch["dit"], c["edm"])

        def recorded(x, s, **k):
            d = den(x, s, **k)
            calls.append((float(s.reshape(-1)[0]), x[0].float().cpu().numpy(), d[0].float().cpu().numpy()))
            return d

        with torch.no_grad():
            if chain == "sdedit":
                z = heun_sample(recorded, shape, c["edm"], dev, steps=int(c["sample"]["steps"]), generator=gens[0],
                                x_init=torch.as_tensor(z0, device=dev)[None], sigma_start=float(sigma))[0]
            else:
                z = heun_sample(recorded, shape, c["edm"], dev, steps=int(c["sample"]["steps"]), generator=gens[0])[0]
        zhat = z.float().cpu().numpy()
        states = calls[0::2]
        sig_t = np.array([s for s, _, _ in states])
        x_t = np.stack([x for _, x, _ in states])
        d_t = np.stack([d for _, _, d in states])
        z_start = calls[0][1]
    secs["dit"] = time.time() - t
    t = time.time()
    F_t = ch["recon"].decode_raw(store.unstandardize(zhat))[0]
    secs["decode"] = time.time() - t
    F = F_t.cpu().numpy()
    show = np.unique(np.linspace(0, max(len(sig_t) - 1, 0), int(n_show)).round().astype(int)) if len(sig_t) else np.zeros(0, int)
    p_show = np.stack([ch["recon"].decode_raw(store.unstandardize(d_t[i]))[0][0].cpu().numpy() for i in show]) \
        if len(show) else np.zeros((0,) + F.shape[1:])

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
    q_raw = off = q = None
    if failed is None:
        t = time.time()
        q_raw, off, lookup = sample_attr(ch["attr"], levels[-1], F_t, c, ch["codec"], gens[2])
        q = t2_rescale(q_raw, lookup, F, c, ch["stats"])[0] if t2 else q_raw
        secs["attr"] = time.time() - t
    out = dict(chain=chain, eid=-1 if chain == "full" else int(eid), ref_eid=ref, sigma=float(sigma or 0.0),
               seed=int(seed), t2=t2, steps=ch["steps"], device=str(dev), thr=thr, z0=z0, z_noised=z_start,
               zhat=zhat, sig_t=sig_t, x_t=x_t, d_t=d_t, show=show, p_show=p_show, F=F, F_z0=None,
               levels=levels, failed=failed, q_raw=q_raw, q=q, off=off, secs=secs, field_stats=ch["stats"])
    if ref is not None:
        out.update(reference(ch, ref))
    return out


def reference(ch, ref):
    """Truth-side keys for event `ref`: its latent, decoded field, truth field, levels and voxels."""
    from .fields import coarse_field, truth_field3
    from .samples import level_grids

    c, store = ch["c"], ch["store"]
    grid, base = int(c["data"]["grid"]), int(c["field"]["base_grid"])
    z0 = store.standardize(store.posterior(int(ref))[0])
    ev = load_event(Path(c["paths"]["processed"]), int(ref))
    f2, _, sums = coarse_field(ev["ijk"], ev["q"], grid, base, float(c["data"]["q_eps"]), ch["stats"])
    return dict(ref_eid=int(ref), z0=z0, F_z0=ch["recon"].decode_raw(store.unstandardize(z0))[0].cpu().numpy(),
                F_truth=truth_field3(f2), truth_sums=sums.reshape(base, base, base),
                truth_levels=[np.unique(np.asarray(ev["ijk"], dtype=np.int64) // (grid // gr), axis=0)
                              for gr in level_grids(grid, base)],
                truth_ijk=np.asarray(ev["ijk"]), truth_q=np.asarray(ev["q"], dtype=np.float64))


def nearest_reference(ch, r, n_ref=300):
    """Attach the test event nearest to a traced (full) sample in standardized event features."""
    from .evaluate import event_metrics
    from .viz import nearest_truth

    c = ch["c"]
    if r["q"] is None:
        return r
    test = [int(e) for e in meta_for(c)["split"]["test"]][: int(n_ref)]
    ref = nearest_truth(c, [event_metrics(r["levels"][-1], r["q"], r["off"], c)], test)[0]
    r.update(reference(ch, ref))
    return r


def compare_saved(r, path):
    """Check a traced sample against the saved sample of the run."""
    if not Path(path).exists():
        return "saved sample not found"
    with np.load(path) as d:
        ijk, q = np.asarray(d["ijk"]), np.asarray(d["q"], dtype=np.float64)
    if r["q"] is not None and len(ijk) == len(r["levels"][-1]) and np.array_equal(ijk, r["levels"][-1]) \
            and np.allclose(q, r["q"], rtol=1e-3):
        return "identical to the saved sample"
    return (f"differs from the saved sample ({len(ijk)} vs {len(r['levels'][-1])} voxels): other device type "
            f"or other checkpoints than the run")


# ------------------------------------------------------ plots: chain steps ---
def _mosaic(z):
    """Latent mosaic (same as chain_steps)."""
    from .chain_steps import _mosaic as m

    return m(z)


def _xz_counts(ijk, gr):
    """Active-cell image along y, empty columns as NaN."""
    from .chain_steps import _count_image

    a = _count_image(ijk, gr)
    return np.where(a > 0, a, np.nan)


def _xz_photons(ijk, q, grid):
    """Photon image along y, empty columns as NaN."""
    from .chain_steps import _photon_image

    a = _photon_image(ijk, q, grid)
    return np.where(a > 0, a, np.nan)


def plot_strip(r, c, title=""):
    """Two rows: the chain (start latent → DiT → decode → start set → struct levels → photons) and its reference."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    ranges = np.asarray(c["data"]["ranges"], dtype=float)
    ext = [ranges[0, 0], ranges[0, 1], ranges[2, 0], ranges[2, 1]]
    grid, base = int(c["data"]["grid"]), int(c["field"]["base_grid"])
    grids = [base * 2 ** k for k in range(int(np.log2(grid // base)) + 1)]
    ncol = 4 + len(grids)
    has_ref = "truth_ijk" in r
    fig, axes = plt.subplots(2 if has_ref else 1, ncol, figsize=(2.75 * ncol, 3.1 * (2 if has_ref else 1)),
                             squeeze=False)
    start_lab = {"sa_recon": "encoder latent μ (no DiT)", "sdedit": f"μ + noise, σ = {r['sigma']:g}",
                 "full": f"pure noise, σ = {r['sig_t'][0]:.3g}" if len(r["sig_t"]) else "noise"}[r["chain"]]
    tp = _xz_photons(r["truth_ijk"], r["truth_q"], grid) if has_ref else None
    vmax = np.nanmax(tp) if has_ref else (np.nanmax(_xz_photons(r["levels"][-1], r["q"], grid)) if r["q"] is not None else 1)
    pn = LogNorm(max(vmax * 1e-6, 1e-3), vmax)

    def im(ax, a, ttl, **kw):
        """imshow helper for one strip panel."""
        ax.set_facecolor("black")
        ax.imshow(a, origin="lower", extent=kw.pop("extent", ext), aspect="auto", interpolation="nearest", **kw)
        ax.set_title(ttl, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])

    row = axes[0]
    im(row[0], _mosaic(r["z_noised"]), f"1  {start_lab}", cmap="RdBu_r", vmin=-3, vmax=3, extent=None)
    im(row[1], _mosaic(r["zhat"]), "2  DiT output" if r["chain"] != "sa_recon" else "2  (no DiT)", cmap="RdBu_r",
       vmin=-3, vmax=3, extent=None)
    im(row[2], r["F"][0].max(axis=1), "3  AE decode: p_occ", cmap="viridis", vmin=0, vmax=1)
    for k, gr in enumerate(grids):
        ref_n = len(r["truth_levels"][k]) if has_ref else None
        if k < len(r["levels"]):
            n = len(r["levels"][k])
            ttl = (f"4  start set {gr}³" if k == 0 else f"5  struct → {gr}³") + f"\n{n:,} cells" + \
                (f" ({n / max(ref_n, 1):.3f}×)" if ref_n else "")
            im(row[3 + k], _xz_counts(r["levels"][k], gr), ttl, cmap="viridis", norm=LogNorm(1, max(2, gr // 4)))
        else:
            row[3 + k].set_axis_off()
            row[3 + k].set_title(f"{gr}³ not reached", fontsize=8.5)
    if r["q"] is not None:
        qs = r["q"].sum()
        im(row[-1], _xz_photons(r["levels"][-1], r["q"], grid),
           ("6  attr + T2: photons" if r["t2"] else "6  attr: photons (T2 off)") + (f"\nQ {qs / r['truth_q'].sum():.3f}× reference" if has_ref else f"\nQ {qs:.3g}"),
           cmap="magma", norm=pn)
    else:
        row[-1].set_axis_off()
    if has_ref:
        lab = "truth" if r["chain"] != "full" else "nearest test event (not paired)"
        row = axes[1]
        im(row[0], _mosaic(r["z0"]), f"{lab}: encoder latent μ", cmap="RdBu_r", vmin=-3, vmax=3, extent=None)
        row[1].set_axis_off()
        row[1].text(0.5, 0.5, f"{lab}\nevent {r['ref_eid']}", ha="center", va="center", fontsize=10,
                    transform=row[1].transAxes)
        im(row[2], r["F_truth"][0].max(axis=1), f"{lab}: occupied 48³", cmap="viridis", vmin=0, vmax=1)
        for k, gr in enumerate(grids):
            im(row[3 + k], _xz_counts(r["truth_levels"][k], gr), f"{lab} {gr}³\n{len(r['truth_levels'][k]):,} cells",
               cmap="viridis", norm=LogNorm(1, max(2, gr // 4)))
        im(row[-1], tp, f"{lab}: photons\nQ {r['truth_q'].sum():.3g}", cmap="magma", norm=pn)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_evolution(r, c, title=""):
    """DiT trajectory: state x_t (latent) and what its estimate D(x_t) decodes to, at a few σ."""
    import matplotlib.pyplot as plt

    if not len(r["show"]):
        return None
    ranges = np.asarray(c["data"]["ranges"], dtype=float)
    ext = [ranges[0, 0], ranges[0, 1], ranges[2, 0], ranges[2, 1]]
    n = len(r["show"])
    fig, axes = plt.subplots(2, n + 1, figsize=(2.8 * (n + 1), 5.6), squeeze=False)
    for k, i in enumerate(r["show"]):
        a = axes[0, k]
        a.imshow(_mosaic(r["x_t"][i]), origin="lower", cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto",
                 interpolation="nearest")
        a.set_title(f"state x_t, σ = {r['sig_t'][i]:.3g}", fontsize=8.5)
        a.set_xticks([])
        a.set_yticks([])
        b = axes[1, k]
        b.set_facecolor("black")
        b.imshow(r["p_show"][k].max(axis=1), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1,
                 aspect="auto", interpolation="nearest")
        b.set_title("decode(D(x_t)): p_occ", fontsize=8.5)
        b.set_xticks([])
        b.set_yticks([])
    a = axes[0, n]
    a.imshow(_mosaic(r["zhat"]), origin="lower", cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto",
             interpolation="nearest")
    a.set_title("DiT output (σ = 0)", fontsize=8.5)
    a.set_xticks([])
    a.set_yticks([])
    b = axes[1, n]
    b.set_facecolor("black")
    b.imshow(r["F"][0].max(axis=1), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1, aspect="auto",
             interpolation="nearest")
    b.set_title("decode(DiT output): p_occ", fontsize=8.5)
    b.set_xticks([])
    b.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_sweep(results, c, title=""):
    """One event through sa_recon (σ = 0) and sdedit at several σ: rows σ, last row truth."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from .evaluate import profiles

    ranges = np.asarray(c["data"]["ranges"], dtype=float)
    ext = [ranges[0, 0], ranges[0, 1], ranges[2, 0], ranges[2, 1]]
    grid = int(c["data"]["grid"])
    r0 = results[0]
    tp = _xz_photons(r0["truth_ijk"], r0["truth_q"], grid)
    vmax = np.nanmax(tp)
    pn = LogNorm(max(vmax * 1e-6, 1e-3), vmax)
    lt, _ = profiles(r0["truth_ijk"], r0["truth_q"], c)
    n = len(results) + 1
    fig, axes = plt.subplots(n, 5, figsize=(15, 2.7 * n), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.3, 1, 1, 1, 1.4]))
    for i, r in enumerate(results + [None]):
        row = axes[i]
        for a in row[:4]:
            a.set_xticks([])
            a.set_yticks([])
            a.set_facecolor("black")
        if r is None:
            row[0].imshow(_mosaic(r0["z0"]), origin="lower", cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto",
                          interpolation="nearest")
            row[0].set_ylabel("truth", fontsize=10)
            row[1].imshow(r0["F_truth"][0].max(axis=1), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1,
                          aspect="auto", interpolation="nearest")
            row[2].imshow(_xz_counts(r0["truth_levels"][-1], grid), origin="lower", extent=ext, cmap="viridis",
                          norm=LogNorm(1, max(2, grid // 4)), aspect="auto", interpolation="nearest")
            row[3].imshow(tp, origin="lower", extent=ext, cmap="magma", norm=pn, aspect="auto", interpolation="nearest")
            row[2].set_title(f"{len(r0['truth_ijk']):,} voxels", fontsize=8)
            row[3].set_title(f"Q {r0['truth_q'].sum():.3g}", fontsize=8)
            row[4].plot(lt, "k-", lw=2)
            row[4].set_title("longitudinal share (truth)", fontsize=8)
            continue
        lab = "σ = 0 (sa_recon)" if r["chain"] == "sa_recon" else f"σ = {r['sigma']:g}"
        row[0].imshow(_mosaic(r["zhat"]), origin="lower", cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto",
                      interpolation="nearest")
        row[0].set_ylabel(lab, fontsize=10)
        row[1].imshow(r["F"][0].max(axis=1), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1,
                      aspect="auto", interpolation="nearest")
        if i == 0:
            row[0].set_title("latent after DiT", fontsize=9)
            row[1].set_title("decoded p_occ (max over y)", fontsize=9)
        if r["q"] is not None:
            nv, qs = len(r["q"]), r["q"].sum()
            row[2].imshow(_xz_counts(r["levels"][-1], grid), origin="lower", extent=ext, cmap="viridis",
                          norm=LogNorm(1, max(2, grid // 4)), aspect="auto", interpolation="nearest")
            row[3].imshow(_xz_photons(r["levels"][-1], r["q"], grid), origin="lower", extent=ext, cmap="magma",
                          norm=pn, aspect="auto", interpolation="nearest")
            lg, _ = profiles(r["levels"][-1], r["q"], c)
            row[2].set_title(f"{nv:,} voxels ({nv / len(r['truth_ijk']):.3f}×)", fontsize=8)
            row[3].set_title(f"Q {qs / r['truth_q'].sum():.3f}× truth", fontsize=8)
            row[4].plot(lt, "k-", lw=1.2, alpha=0.5, label="truth")
            row[4].plot(lg, "C0-", lw=1.6, label=lab)
            row[4].set_title(f"longitudinal share, L1 = {np.abs(lg - lt).sum():.3f}", fontsize=8)
        else:
            row[2].set_title(f"failed: {r['failed']}", fontsize=8)
        row[4].grid(alpha=0.25)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
