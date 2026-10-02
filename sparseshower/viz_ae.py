"""AE check plots behind notebooks/ae_check.ipynb: where the FieldAE is right and
wrong at a given occupancy threshold.

Reuses the per-event cache of viz.ae_recon (truth field, photon sums, decoded
field F; keyed by checkpoint step and event).  Everything that depends on the
threshold is recomputed here from F, so one cache serves every threshold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import viz
from .loader import case_dir, meta_for

CATEGORY_COLOURS = ["#000000", "#8c8c8c", "#e8412c", "#2c8ce8", "#c040c0"]
CATEGORY_NAMES = ["empty", "matched only", "extra (pred, not true)", "missed (true, not pred)", "extra + missed"]


# ------------------------------------------------------------------ events ---
def pick_events(c, case, scan_name=None, n_first=4):
    """[(eid, reason)]: from eval/<scan_name>/per_event.jsonl the smallest, median and
    largest val event (by true non-empty 48^3 cells) and the worst by field_long_l1 and
    by q_total_rel_err; without a scan the first n_first val events."""
    val = [int(e) for e in meta_for(c)["split"]["val"]]
    p = case_dir(c, case) / "eval" / str(scan_name) / "per_event.jsonl" if scan_name else None
    if p is None or not p.exists():
        return [(e, f"val #{i}") for i, e in enumerate(val[:n_first])]
    rows = [json.loads(line) for line in open(p)]
    by_n = sorted(rows, key=lambda r: r["n48_true"])
    picks = [(by_n[0], "smallest"), (by_n[len(by_n) // 2], "median size"), (by_n[-1], "largest"),
             (max(rows, key=lambda r: r["field_long_l1"]), "worst long_l1"),
             (max(rows, key=lambda r: r["q_total_rel_err"]), "worst q_total")]
    reasons = {}
    for r, why in picks:
        reasons.setdefault(int(r["eid"]), []).append(why)
    return [(e, ", ".join(w)) for e, w in reasons.items()]


# ----------------------------------------------------------------- numbers ---
def at_threshold(c, r, thr, stats, pidx):
    """Threshold-dependent view of one cached ae_recon row (flat arrays of B^3)."""
    from .fields import field_profiles, photon_sums

    p = np.asarray(r["F"][0], dtype=np.float64).reshape(-1)
    occ_p = p > float(thr)
    occ_t = np.asarray(r["counts"]).reshape(-1) > 0
    sums_t = np.asarray(r["sums"], dtype=np.float64).reshape(-1)
    sums_p = photon_sums(np.asarray(r["F"][2]).reshape(-1), occ_p, stats, float(c["data"]["q_eps"]))
    lt, rt = field_profiles(sums_t, *pidx)
    lp, rp = field_profiles(sums_p, *pidx)
    n_t = max(int(occ_t.sum()), 1)
    tot = max(sums_t.sum(), 1e-30)
    m = dict(n48_true=int(occ_t.sum()), n48_ratio=float(occ_p.sum() / n_t),
             extra=float((occ_p & ~occ_t).sum() / n_t), missed=float((occ_t & ~occ_p).sum() / n_t),
             lost=float(sums_t[occ_t & ~occ_p].sum() / tot), q_ratio=float(sums_p.sum() / tot),
             long_l1=float(np.abs(lp - lt).sum()), rad_l1=float(np.abs(rp - rt).sum()))
    return dict(p=p, occ_p=occ_p, occ_t=occ_t, sums_t=sums_t, sums_p=sums_p, lt=lt, rt=rt, lp=lp, rp=rp,
                metrics=m)


def load_rows(c, case, eids, checkpoint="best", device="cpu"):
    """Cached truth + decoded fields (viz.ae_recon), plus what every plot here needs."""
    from .fields import cell_profile_index, load_field_stats

    rows = viz.ae_recon(c, case, [int(e) for e in eids], checkpoint, device)
    stats = load_field_stats(c)
    pidx = cell_profile_index(c, int(c["field"]["base_grid"]))
    return rows, stats, pidx


# ---------------------------------------------------------------- per event ---
def _extent_xz(c):
    """Physical x-z extent of the box for imshow."""
    ranges = np.asarray(c["data"]["ranges"], dtype=np.float64)
    return [ranges[0, 0], ranges[0, 1], ranges[2, 0], ranges[2, 1]]


def category_image(occ_p, occ_t, B):
    """[z, x] category of each y-column: 0 empty, 1 matched only, 2 extra, 3 missed, 4 both."""
    shape = (B, B, B)                                                  # flat index -> [z, y, x]
    matched = (occ_p & occ_t).reshape(shape).sum(axis=1) > 0
    extra = (occ_p & ~occ_t).reshape(shape).sum(axis=1) > 0
    missed = (occ_t & ~occ_p).reshape(shape).sum(axis=1) > 0
    img = np.zeros((B, B), dtype=np.int64)
    img[matched] = 1
    img[extra & ~missed] = 2
    img[missed & ~extra] = 3
    img[extra & missed] = 4
    return img


def plot_events(c, rows, labels, thr, stats, pidx, compare_thr=0.5, checkpoint="best"):
    """One row per event: truth | decoded at thr | extra/missed cell map | z profile | radial profile."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    if not rows:
        return None
    B = int(c["field"]["base_grid"])
    fig, axes = plt.subplots(len(rows), 5, figsize=(17, 3.1 * len(rows)), squeeze=False)
    for ax, r in zip(axes, rows):
        a = at_threshold(c, r, thr, stats, pidx)
        b = at_threshold(c, r, compare_thr, stats, pidx) if compare_thr is not None else None
        m = a["metrics"]
        eid = int(r["eid"])
        img_t = a["sums_t"].reshape(B, B, B).sum(axis=1).T                   # [x, z]
        img_p = a["sums_p"].reshape(B, B, B).sum(axis=1).T
        hi = np.log10(max(img_t.max(), 1e-3))
        viz.show_projection(ax[0], img_t, c, f"event {eid} ({labels.get(eid, '')}): truth, y sum", hi - 5, hi)
        viz.show_projection(ax[1], img_p, c, f"AE decode, p_occ > {thr} (step {int(r['step'])})", hi - 5, hi)
        cat = category_image(a["occ_p"], a["occ_t"], B)
        ax[2].imshow(cat, origin="lower", aspect="auto", extent=_extent_xz(c), interpolation="nearest",
                     cmap=ListedColormap(CATEGORY_COLOURS), vmin=-0.5, vmax=4.5)
        ax[2].set_title(f"cells: extra {m['extra']:.1%}, missed {m['missed']:.1%} "
                        f"(of {m['n48_true']} true)", fontsize=8)
        ax[2].set_xlabel("x [m]", fontsize=7)
        ax[2].tick_params(labelsize=6)
        ax[3].plot(a["lt"], color="k", label="truth")
        ax[3].plot(a["lp"], color="C3", ls="--", label=f"decode > {thr}")
        if b is not None:
            ax[3].plot(b["lp"], color="C0", ls=":", alpha=0.8, label=f"decode > {compare_thr}")
        ax[3].set_title(f"z profile: long_l1 {m['long_l1']:.4f}, Q ratio {m['q_ratio']:.3f}", fontsize=8)
        ax[3].set_xlabel("z slab (top first)", fontsize=7)
        pos = lambda v: np.where(np.asarray(v) > 0, v, np.nan)             # log axis: leave empty rings out
        ax[4].plot(pos(a["rt"]), color="k", label="truth")
        ax[4].plot(pos(a["rp"]), color="C3", ls="--", label=f"decode > {thr}")
        if b is not None:
            ax[4].plot(pos(b["rp"]), color="C0", ls=":", alpha=0.8, label=f"decode > {compare_thr}")
        ax[4].set_yscale("log")
        ax[4].set_title(f"radial profile: rad_l1 {m['rad_l1']:.4f}, lost {m['lost']:.2%}", fontsize=8)
        ax[4].set_xlabel("ring (sqrt(r) spacing)", fontsize=7)
        for k in (3, 4):
            ax[k].tick_params(labelsize=6)
            ax[k].legend(fontsize=6)
    handles = [Patch(color=col, label=name) for col, name in zip(CATEGORY_COLOURS[1:], CATEGORY_NAMES[1:])]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"FieldAE ae_{checkpoint}.pt on val events, threshold {thr}{viz.data_label(c)}",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ pooled ---
def pooled(c, rows, thr, stats, pidx):
    """Cell-level arrays pooled over events, plus one metrics dict per event."""
    keys = ("p_true", "p_empty", "q_matched_t", "q_matched_p", "q_missed", "q_extra")
    out = {k: [] for k in keys}
    per_event = []
    for r in rows:
        a = at_threshold(c, r, thr, stats, pidx)
        occ_t, occ_p = a["occ_t"], a["occ_p"]
        out["p_true"].append(a["p"][occ_t])
        out["p_empty"].append(a["p"][~occ_t & (a["p"] > 1e-3)])
        both = occ_t & occ_p
        out["q_matched_t"].append(a["sums_t"][both])
        out["q_matched_p"].append(a["sums_p"][both])
        out["q_missed"].append(a["sums_t"][occ_t & ~occ_p])
        out["q_extra"].append(a["sums_p"][occ_p & ~occ_t])
        per_event.append(dict(eid=int(r["eid"]), **a["metrics"]))
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}, per_event


def plot_pooled(c, rows, thr, stats, pidx, compare_thr=0.5):
    """p_occ of true / empty cells, photons of matched / missed / extra cells, per-cell
    photon accuracy, per-event total-photon ratio vs event size."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import NullFormatter

    if not rows:
        return None
    P, per_event = pooled(c, rows, thr, stats, pidx)
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    bins = np.linspace(0, 1, 51)
    ax[0].hist(P["p_true"], bins=bins, histtype="step", color="k", label="true non-empty cells")
    ax[0].hist(P["p_empty"], bins=bins, histtype="step", color="C3", label="true empty cells (p > 0.001)")
    for t, ls in ((compare_thr, ":"), (thr, "--")):
        if t is not None:
            ax[0].axvline(float(t), color="C0", ls=ls, label=f"threshold {t}")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("decoded p_occ")
    ax[0].set_title("p_occ by truth: the cut decides extra vs missed", fontsize=9)
    ax[0].legend(fontsize=7)

    lb = np.linspace(-1, 7, 65)
    for key, col, name in (("q_matched_t", "k", "matched (truth photons)"),
                           ("q_missed", "C0", "missed (truth photons)"),
                           ("q_extra", "C3", "extra (decoded photons)")):
        v = P[key][P[key] > 0]
        ax[1].hist(np.log10(v), bins=lb, histtype="step", color=col,
                   label=f"{name}: {len(P[key])} cells, {P[key].sum():.3g} photons")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("log10 photons per 48³ cell")
    ax[1].set_title(f"photons per cell at threshold {thr}", fontsize=9)
    ax[1].legend(fontsize=6)

    t, p = P["q_matched_t"], P["q_matched_p"]
    ok = (t > 0) & (p > 0)
    if ok.any():
        lt, lp = np.log10(t[ok]), np.log10(p[ok])
        ax[2].hist2d(lt, lp, bins=[lb, lb], cmap="Greys", norm=LogNorm())
        ax[2].plot(lb, lb, color="C3", lw=0.8)
        d = lp - lt
        ax[2].set_title(f"matched cells: log10(decoded/truth) mean {d.mean():+.3f}, sd {d.std():.3f}",
                        fontsize=9)
    ax[2].set_xlabel("log10 truth photons")
    ax[2].set_ylabel("log10 decoded photons")

    n = np.array([e["n48_true"] for e in per_event])
    q = np.array([e["q_ratio"] for e in per_event])
    ax[3].scatter(n, q, s=8, color="k")
    ax[3].axhline(1.0, color="C3", lw=0.8)
    ax[3].axhspan(0.99, 1.01, color="C3", alpha=0.12, label="±1% (q_total threshold)")
    ax[3].set_xscale("log")
    ax[3].xaxis.set_minor_formatter(NullFormatter())
    ax[3].set_xlabel("true non-empty 48³ cells")
    ax[3].set_ylabel("decoded / truth total photons")
    ax[3].set_title(f"total photons per event (mean {q.mean():.4f}, sd {q.std():.4f})", fontsize=9)
    ax[3].legend(fontsize=7)
    for a_ in ax:
        a_.tick_params(labelsize=7)
    fig.suptitle(f"FieldAE, {len(rows)} val events pooled, threshold {thr}{viz.data_label(c)}", fontsize=10)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------- scan ---
def load_scan(c, case, prefix="thr_"):
    """[(threshold, summary, per_event rows)] from eval/<prefix><t>/ (ae-eval outputs)."""
    out = []
    root = case_dir(c, case) / "eval"
    for d in root.glob(f"{prefix}*") if root.exists() else []:
        try:
            t = float(d.name[len(prefix):])
        except ValueError:
            continue
        s, pe = d / "summary.json", d / "per_event.jsonl"
        if s.exists() and pe.exists():
            out.append((t, json.loads(s.read_text()), [json.loads(line) for line in open(pe)]))
    return sorted(out, key=lambda x: x[0])


def plot_scan(c, scan, thr=None):
    """Acceptance numbers vs threshold (the ae-eval scan), acceptance thresholds dotted."""
    import matplotlib.pyplot as plt

    if not scan:
        return None
    ts = np.array([s[0] for s in scan])

    def mean(k):
        """Mean of one metric across the scan thresholds."""
        return np.array([viz.num(s[1]["metrics"][k]["mean"]) for s in scan])

    def ev(f):
        """Per-threshold mean of a function over the per-event rows."""
        return np.array([np.mean([f(r) for r in s[2]]) for s in scan])

    ratio = ev(lambda r: r["n48_pred"] / max(r["n48_true"], 1))
    extra = mean("extra_cells_frac")
    missed = extra - (ratio - 1.0)
    th = c.get("thresholds", {}).get("ae", {})
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(ts, ratio, "o-", color="k", label="n48 pred / true")
    ax[0].plot(ts, extra, "o-", color="C3", label="extra cells / true")
    ax[0].plot(ts, missed, "o-", color="C0", label="missed cells / true")
    ax[0].axhline(1.0, color="k", lw=0.6)
    if "n48_rel_err" in th:
        ax[0].axhspan(1 - float(th["n48_rel_err"]), 1 + float(th["n48_rel_err"]), color="k", alpha=0.1,
                      label=f"n48 threshold ±{th['n48_rel_err']}")
    ax[0].set_title("48³ cells vs threshold", fontsize=9)
    for i, k in enumerate(("lost_photon_frac", "field_long_l1", "field_rad_l1", "q_total_rel_err")):
        col = f"C{i + 1}"
        ax[1].plot(ts, mean(k), "o-", color=col, label=k)
        if k in th:
            ax[1].axhline(float(th[k]), color=col, ls=":", lw=1)
    ax[1].set_yscale("log")
    ax[1].set_title("acceptance numbers vs threshold (dotted: acceptance thresholds)", fontsize=9)
    for a_ in ax:
        a_.set_xlabel("occupancy threshold")
        if thr is not None:
            a_.axvline(float(thr), color="grey", ls="--", lw=0.8)
        a_.legend(fontsize=7)
        a_.tick_params(labelsize=7)
    n = scan[0][1].get("n_events")
    fig.suptitle(f"ae-eval scan: {len(scan)} thresholds, {n} val events each, "
                 f"ae step {scan[0][1].get('ae_step')}{viz.data_label(c)}", fontsize=10)
    fig.tight_layout()
    return fig


def save_new(fig, fig_dir, name):
    """Save a figure without overwriting: <name>.png, else <name>_<timestamp>.png."""
    import time

    if fig is None:
        return None
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    if path.exists():
        path = fig_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path
