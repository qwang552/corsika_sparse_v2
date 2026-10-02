"""Interactive 3D comparison page for sample runs.

    python -m sparseshower.chain_page --config configs/v2_cal85.yaml --case v2a \
        --runs sa_truth sa_recon sdedit_1 --full-run full --events 13 28 31

One self-contained HTML file (fonts from Google Fonts if online, nothing else
external):

* 3D view      one tab per paired test event (truth | every paired run | v1)
               and one per unconditional sample (generated | nearest test
               event, labelled "not paired").  Up to --max-points voxels per
               panel are drawn (a random subset beyond that); light is
               accumulated along the line of sight and coloured by log Q.
               All panels rotate and zoom together.
* charts       per-voxel log Q, nearest-neighbour distance, local linearity,
               longitudinal and radial photon share, box-counting dimension;
               truth and each run pooled over the events shown.
* tables       per run against truth (medians over events), and every sample.

Nothing is overwritten: an explicit --out that exists is refused, the default
name gets a timestamp if it exists.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import time
from pathlib import Path

import numpy as np

from .common import config, read_json
from .data import load_event
from .evaluate import profiles, w1
from .geometry import box_count, box_dimension, knn_within_grid, local_linearity, voxel_center
from .loader import case_dir, meta_for
from .sample import load_run, load_v1_run

# viridis from 0.12 to 1.0, dimmed so that accumulated light saturates softly
VIRIDIS = [(0.2803, 0.1657, 0.4765), (0.2412, 0.2965, 0.5397), (0.1872, 0.4147, 0.5565),
           (0.1448, 0.5191, 0.5566), (0.1201, 0.6222, 0.5349), (0.2140, 0.7221, 0.4696),
           (0.4310, 0.8085, 0.3465), (0.7099, 0.8688, 0.1693), (0.9932, 0.9062, 0.1439)]
LUT_SCALE = 0.6
COLOURS = ["#2a78d6", "#eb6834", "#1a9e77", "#9b59d0", "#c99a06", "#d63a6e", "#4aa3a3"]
COLOURS_DARK = ["#3f8ae6", "#f07a47", "#2fbf8f", "#b07ce6", "#e0b429", "#e85a8a", "#5fc0c0"]


def lut256():
    """256-entry viridis colour table used by the page's renderer."""
    a = np.asarray(VIRIDIS)
    t = np.linspace(0, 1, len(a))
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, t, a[:, k]) for k in range(3)], 1) * LUT_SCALE


def describe(run_dir):
    """Chain description of a run from its run.json."""
    spec = read_json(run_dir / "run.json") if (run_dir / "run.json").exists() else {}
    chain = spec.get("chain", "?")
    text = {"sa_truth": "true 48³ field → struct → attr",
            "sa_recon": "encoder latent → AE decode → struct → attr",
            "full": "noise → DiT → AE decode → struct → attr"}.get(chain, chain)
    if chain == "sdedit":
        text = f"latent noised to σ = {spec.get('sigma_start')} → DiT → AE decode → struct → attr"
    if spec and not spec.get("t2", True):
        text += " (T2 off: photons not rescaled per 48³ cell)"
    steps = read_json(run_dir / "checkpoints.json") if (run_dir / "checkpoints.json").exists() else {}
    return chain, text, spec, steps


def seconds_of(run_dir):
    """Seconds per sample of a run, read from its index_shard*.jsonl files."""
    out = {}
    for f in sorted(Path(run_dir).glob("index_shard*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "seconds" in r:
                out[int(r["index"])] = float(r["seconds"])
    return out


# ------------------------------------------------------------ statistics ---
def voxel_stats(ijk, q, off, c):
    """Small per-sample statistics shown next to the 3D view."""
    grid = int(c["data"]["grid"])
    ranges = np.asarray(c["data"]["ranges"], dtype=np.float64)
    voxel = (ranges[:, 1] - ranges[:, 0]) / grid
    ijk = np.asarray(ijk, dtype=np.int64)
    q = np.asarray(q, dtype=np.float64)
    off = np.zeros((len(q), 3)) if off is None else np.asarray(off, dtype=np.float64)
    points = voxel_center(ijk, ranges, grid) + off * voxel
    if len(q) > 2:
        nn = knn_within_grid(points, ijk, grid, 2)[0][:, 1]
        nn = nn[np.isfinite(nn)]
        lin = local_linearity(points, ijk, grid, k=8)
    else:
        nn, lin = np.zeros(0), np.zeros(0)
    long_p, rad_p = profiles(ijk, q, c)
    factors = [f for f in (1, 2, 4, 8, 16) if grid % f == 0]
    bd = (list(box_dimension(box_count(ijk, factors), factors)) + [float("nan")] * 4)[:4]
    order = np.sort(q)[::-1]
    share = np.cumsum(order) / max(order.sum(), 1e-30)
    top5 = float(share[min(len(share) - 1, int(0.05 * len(order)))]) if len(order) else float("nan")
    return dict(n=int(len(q)), q=float(q.sum()), nn=nn, lin=lin, logq=np.log(q + float(c["data"]["q_eps"])),
                long=long_p, rad=rad_p, box=[float(v) for v in bd], top5=top5,
                straight=float((lin > 0.95).mean()) if len(lin) else float("nan"),
                nn_med=float(np.median(nn)) if len(nn) else float("nan"))


def hist(values, edges):
    """Normalised histogram of concatenated values."""
    v = np.concatenate([np.asarray(a, dtype=np.float64) for a in values]) if values else np.zeros(0)
    h, _ = np.histogram(v, bins=edges)
    return (h / max(h.sum(), 1)).round(6).tolist()


def median_rows(rows):
    """Element-wise median over a list of equal-length rows."""
    return np.nanmedian(np.stack(rows), axis=0).round(6).tolist() if rows else []


def ratio(a, b):
    """a / b, NaN when b is zero."""
    return float(a / b) if b else float("nan")


# ------------------------------------------------------------------ build ---
def build(c, case, paired_runs, full_run=None, v1_dir=None, events=None, n_gallery=6, title=None,
          max_points=250000, verbose=True):
    """Collect all tabs (paired events, gallery) and their data for the page."""
    grid = int(c["data"]["grid"])
    if grid > 256:
        raise ValueError("chain_page packs voxel indices into uint8: grid must be <= 256")
    root = c["paths"]["processed"]
    meta = meta_for(c)
    cd = case_dir(c, case)
    log = print if verbose else (lambda *a, **k: None)

    # series: paired runs, v1, full (truth is added per tab)
    series, runs = [], {}
    for r in paired_runs:
        run_dir = cd / "samples" / r
        chain, text, spec, steps = describe(run_dir)
        rows = load_run(run_dir)
        runs[r] = dict(rows={int(x["eid"]): x for x in rows}, secs=seconds_of(run_dir))
        series.append(dict(key=r, name=r, desc=text, chain=chain, steps=steps, spec=spec, n=len(rows)))
        log(f"{r}: {len(rows)} samples ({text})")
    if v1_dir:
        rows = load_v1_run(v1_dir)
        runs["v1"] = dict(rows={int(x["eid"]): x for x in rows}, secs={})
        series.append(dict(key="v1", name="v1", desc="v1 model, conditioned on the (z, r) photon summary of the true event", chain="v1", steps={},
                           spec={}, n=len(rows)))
    gen_rows = []
    if full_run:
        run_dir = cd / "samples" / full_run
        chain, text, spec, steps = describe(run_dir)
        gen_rows = load_run(run_dir)[: int(n_gallery)]
        series.append(dict(key=full_run, name=full_run, desc=text + " (unpaired)", chain=chain, steps=steps,
                           spec=spec, n=len(gen_rows), unpaired=True))
        log(f"{full_run}: {len(gen_rows)} samples in the gallery ({text})")
    for i, s in enumerate(series):
        s["colour"], s["colour_dark"] = COLOURS[i % len(COLOURS)], COLOURS_DARK[i % len(COLOURS_DARK)]

    if events is None:
        found = set()
        for r in paired_runs:
            found |= set(runs[r]["rows"])
        test = [int(e) for e in meta["split"]["test"]]
        events = [e for e in test if e in found][:6] or [int(e) for e in c["viz"]["events"]]
    events = [int(e) for e in events]

    # per-panel raw data, coloured once the global log Q range is known
    raw, tabs = [], []
    stats = {"truth": {}}
    for s in series:
        stats[s["key"]] = {}
    for e in events:
        ev = load_event(root, e)
        st = voxel_stats(ev["ijk"], ev["q"], ev.get("off"), c)
        stats["truth"][e] = st
        panels = [dict(key="truth", name="Truth", desc=f"test event {e}", n=st["n"], q=st["q"], raw=len(raw))]
        raw.append((np.asarray(ev["ijk"]), st["logq"]))
        for s in series:
            if s.get("unpaired"):
                continue
            x = runs[s["key"]]["rows"].get(e)
            if x is None:
                panels.append(dict(key=s["key"], name=s["name"], desc=s["desc"], missing=True))
                continue
            sg = voxel_stats(x["ijk"], x["q"], x.get("off"), c)
            stats[s["key"]][e] = sg
            sec = runs[s["key"]]["secs"].get(int(x.get("index", -1)))
            panels.append(dict(key=s["key"], name=s["name"], desc=s["desc"], n=sg["n"], q=sg["q"], raw=len(raw),
                               n_ratio=ratio(sg["n"], st["n"]), q_ratio=ratio(sg["q"], st["q"]),
                               long_l1=float(np.abs(sg["long"] - st["long"]).sum()),
                               rad_l1=float(np.abs(sg["rad"] - st["rad"]).sum()),
                               straight=sg["straight"], seconds=sec,
                               counts=[int(v) for v in np.asarray(x.get("counts", [])).ravel()]))
            raw.append((np.asarray(x["ijk"]), sg["logq"]))
        tabs.append(dict(kind="paired", label=f"event {e}", event=e, panels=panels, truth_straight=st["straight"]))
        log(f"event {e}: {st['n']} truth voxels")

    if gen_rows:
        from .evaluate import event_metrics, features, truth_metrics
        test = [int(t) for t in meta["split"]["test"]][:200]
        tm = {t: truth_metrics(c, t) for t in test}
        F = np.stack([features(tm[t]) for t in test])
        mu, sd = np.nanmean(F, 0), np.nanstd(F, 0) + 1e-9
        for x in gen_rows:
            fg = (features(event_metrics(x["ijk"], x["q"], x.get("off"), c)) - mu) / sd
            t = test[int(np.nanargmin(np.nansum(((F - mu) / sd - fg) ** 2, axis=1)))]
            sg = voxel_stats(x["ijk"], x["q"], x.get("off"), c)
            stats[full_run][int(x["index"])] = sg
            ev = load_event(root, t)
            st = voxel_stats(ev["ijk"], ev["q"], ev.get("off"), c)
            sec = seconds_of(cd / "samples" / full_run).get(int(x["index"]))
            panels = [dict(key=full_run, name=f"{full_run} #{int(x['index'])}", desc="unconditional sample",
                           n=sg["n"], q=sg["q"], raw=len(raw), seconds=sec,
                           counts=[int(v) for v in np.asarray(x.get("counts", [])).ravel()]),
                      dict(key="truth", name=f"Nearest truth · event {t}",
                           desc="closest test event in event features; visual reference only, not paired",
                           n=st["n"], q=st["q"], raw=len(raw) + 1)]
            raw.append((np.asarray(x["ijk"]), sg["logq"]))
            raw.append((np.asarray(ev["ijk"]), st["logq"]))
            tabs.append(dict(kind="gen", label=f"#{int(x['index'])}", event=t, panels=panels))
        log(f"gallery: {len(gen_rows)} unconditional samples")

    # global colour scale and packing
    pool = np.concatenate([lq for _, lq in raw]) if raw else np.zeros(1)
    lo, hi = (float(np.quantile(pool, 0.002)), float(np.quantile(pool, 0.999))) if len(pool) > 1 else (0.0, 1.0)
    hi = hi if hi > lo else lo + 1.0
    packed, rng = [], np.random.default_rng(0)
    for ijk, lq in raw:
        if max_points and len(lq) > max_points:     # only the drawing is thinned; statistics use every voxel
            keep = np.sort(rng.choice(len(lq), int(max_points), replace=False))
            ijk, lq = np.asarray(ijk)[keep], lq[keep]
        ci = np.clip(np.rint((lq - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
        a = np.empty((len(ijk), 4), dtype=np.uint8)
        a[:, :3] = np.asarray(ijk, dtype=np.int64)
        a[:, 3] = ci
        packed.append((base64.b64encode(a.tobytes()).decode(), len(a)))
    for t in tabs:
        for p in t["panels"]:
            if "raw" in p:
                p["b64"], p["drawn"] = packed[p.pop("raw")]

    # charts (pooled over the events shown)
    keys = ["truth"] + [s["key"] for s in series]
    ed_q = np.linspace(lo, hi, 49)
    ed_nn = np.linspace(0.015, 0.09, 41)
    ed_lin = np.linspace(0.3, 1.0, 36)
    ranges = np.asarray(c["data"]["ranges"], dtype=np.float64)
    zc = (ranges[2, 1] - (np.arange(32) + 0.5) / 32 * (ranges[2, 1] - ranges[2, 0])).round(3).tolist()
    charts = dict(logq=dict(edges=ed_q.round(4).tolist()), nn=dict(edges=ed_nn.round(4).tolist()),
                  lin=dict(edges=ed_lin.round(4).tolist()), z=zc, long={}, rad={}, box={})
    for k in keys:
        S = list(stats[k].values())
        if not S:
            continue
        charts["logq"][k] = hist([s["logq"] for s in S], ed_q)
        charts["nn"][k] = hist([s["nn"] for s in S], ed_nn)
        charts["lin"][k] = hist([s["lin"] for s in S], ed_lin)
        charts["long"][k] = median_rows([s["long"] for s in S])
        charts["rad"][k] = median_rows([s["rad"] for s in S])
        charts["box"][k] = median_rows([np.asarray(s["box"]) for s in S])

    # table: each paired run against truth
    table = []
    T = stats["truth"]
    tpool = {k: np.concatenate([T[e][k] for e in events]) for k in ("lin", "nn", "logq")}
    for s in series:
        G = stats[s["key"]]
        if s.get("unpaired"):
            ids = list(G)
            row = dict(key=s["key"], unpaired=True, events=len(ids))
            if ids:
                gp = {k: np.concatenate([G[i][k] for i in ids]) for k in ("lin", "nn", "logq")}
                row.update(w1_lin=w1(gp["lin"], tpool["lin"]), w1_nn=w1(gp["nn"], tpool["nn"]),
                           w1_logq=w1(gp["logq"], tpool["logq"]),
                           straight=float(np.nanmean([G[i]["straight"] for i in ids])))
            table.append(row)
            continue
        es = [e for e in events if e in G]
        if not es:
            continue
        gp = {k: np.concatenate([G[e][k] for e in es]) for k in ("lin", "nn", "logq")}
        tp = {k: np.concatenate([T[e][k] for e in es]) for k in ("lin", "nn", "logq")}
        table.append(dict(
            key=s["key"], events=len(es),
            n_ratio=float(np.median([G[e]["n"] / T[e]["n"] for e in es])),
            q_ratio=float(np.median([G[e]["q"] / T[e]["q"] for e in es])),
            long_l1=float(np.median([np.abs(G[e]["long"] - T[e]["long"]).sum() for e in es])),
            rad_l1=float(np.median([np.abs(G[e]["rad"] - T[e]["rad"]).sum() for e in es])),
            top5_ratio=float(np.median([G[e]["top5"] / T[e]["top5"] for e in es])),
            nn_ratio=float(np.median([G[e]["nn_med"] / T[e]["nn_med"] for e in es])),
            w1_lin=w1(gp["lin"], tp["lin"]), w1_nn=w1(gp["nn"], tp["nn"]), w1_logq=w1(gp["logq"], tp["logq"]),
            straight=float(np.nanmean([G[e]["straight"] for e in es])),
            truth_straight=float(np.nanmean([T[e]["straight"] for e in es]))))

    span = ranges[:, 1] - ranges[:, 0]
    data = dict(
        grid=grid, lut=lut256().round(4).ravel().tolist(), lq_range=[lo, hi], case=case,
        synthetic=bool(c.get("synthetic")), series=[{k: v for k, v in s.items() if k != "spec"} for s in series],
        spec={s["key"]: {k: s["spec"].get(k) for k in ("heun_steps", "occ_threshold", "t2", "seed")}
              for s in series if s.get("spec")},
        tabs=tabs, charts=charts, table=table, events=events, max_points=int(max_points or 0),
        geometry=dict(voxel_cm=round(float(span[0] / grid) * 100, 2), z=[float(ranges[2, 0]), float(ranges[2, 1])],
                      box_m=round(float(span[0]), 2)),
        made=time.strftime("%Y-%m-%d %H:%M"))
    ttl = title or "v2 chains vs truth"
    if c.get("synthetic"):
        ttl += " [SYNTHETIC data]"
    return PAGE.replace("__TITLE__", html.escape(ttl)).replace("__DATA__", _json(data))


def _json(obj):
    """JSON dump with NaN / inf replaced by null."""
    def clean(o):
        """Recursively replace non-finite floats by None."""
        if isinstance(o, float):
            return None if not np.isfinite(o) else o
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.generic):
            return clean(o.item())
        return o
    return json.dumps(clean(obj), separators=(",", ":")).replace("</", "<\\/")


def export(c, case, paired_runs, full_run=None, v1_dir=None, events=None, n_gallery=6, out=None, title=None,
           max_points=250000):
    """Write the page to <case>/viz/chain_page.html (timestamped name if it exists)."""
    page = build(c, case, paired_runs, full_run, v1_dir, events, n_gallery, title, max_points)
    if out and Path(out).exists():
        raise FileExistsError(f"{out} exists; pass another --out (nothing is overwritten)")
    out = Path(out or case_dir(c, case) / "viz" / "chain_page.html")
    if out.exists():
        out = out.with_name(f"{out.stem}_{time.strftime('%Y%m%d-%H%M%S')}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return str(out)


def main(argv=None):
    """Command-line entry point (python -m sparseshower.chain_page)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--case", default="v2a")
    ap.add_argument("--runs", nargs="+", default=[], help="paired sample runs (sa_truth, sa_recon, sdedit_*)")
    ap.add_argument("--full-run", help="unconditional run for the gallery tabs (none = no gallery)")
    ap.add_argument("--v1-dir", help="v1 samples directory (jobs/v1_samples.sub)")
    ap.add_argument("--events", type=int, nargs="+", help="test events (default: first 6 present in the runs)")
    ap.add_argument("--n-gallery", type=int, default=6)
    ap.add_argument("--title")
    ap.add_argument("--out")
    ap.add_argument("--max-points", type=int, default=250000,
                    help="voxels drawn per panel at most (random subset; statistics use all); 0 = no limit")
    a = ap.parse_args(argv)
    c = config(a.config)
    if a.full_run in ("", "none"):
        a.full_run = None
    path = export(c, a.case, a.runs, a.full_run, a.v1_dir, a.events, a.n_gallery, a.out, a.title, a.max_points)
    size = Path(path).stat().st_size / 1e6
    print(f"page -> {path} ({size:.1f} MB)")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{--bg:#f3f5f7;--surface:#fff;--ink:#151a21;--muted:#5d6774;--faint:#8a939e;--rule:#dde2e8;--rule2:#eceff3;--chip:#e8edf3;--good:#1a7f4b;--screen:#05070a;--truth:#151a21;
--sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--cond:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0f1216;--surface:#161a20;--ink:#e6e9ee;--muted:#9aa4b1;--faint:#6f7985;--rule:#29303a;--rule2:#1f252d;--chip:#1f262f;--good:#46b37b;--truth:#e6e9ee}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0f1216;--surface:#161a20;--ink:#e6e9ee;--muted:#9aa4b1;--faint:#6f7985;--rule:#29303a;--rule2:#1f252d;--chip:#1f262f;--good:#46b37b;--truth:#e6e9ee}
*{box-sizing:border-box}body{background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans);margin:0}
.wrap{max-width:1320px;margin:0 auto;padding:28px 20px 56px;display:flex;flex-direction:column;gap:40px}
h1,h2,h3{font-family:var(--cond);font-weight:600;letter-spacing:.005em;text-wrap:balance;margin:0}
h1{font-size:clamp(28px,4vw,40px);line-height:1.1}h2{font-size:22px;line-height:1.2}h3{font-size:15px}
p{margin:0;max-width:78ch}.eyebrow{font:500 12px/1.4 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.muted{color:var(--muted)}.mono,.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
header{display:flex;flex-direction:column;gap:12px}.facts{display:flex;flex-wrap:wrap;gap:8px}
.chip{font:12.5px/1 var(--mono);background:var(--chip);border-radius:4px;padding:6px 8px;white-space:nowrap}.chip b{font-weight:500;color:var(--muted);margin-right:6px}
section{display:flex;flex-direction:column;gap:16px}.sec-head{display:flex;flex-direction:column;gap:6px}
.chains{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
.chainc{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px 14px;display:flex;flex-direction:column;gap:4px}
.chainc .n{font:600 14px/1.3 var(--cond)}.chainc .d{font-size:13px;color:var(--muted)}.chainc .s{font:12px/1.4 var(--mono);color:var(--faint)}
.tabrow{display:flex;flex-wrap:wrap;gap:6px;align-items:center}.tabrow .lbl{font:500 12px/1 var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-right:6px;min-width:120px}
.tab{font:500 13px/1 var(--mono);padding:8px 10px;border-radius:5px;border:1px solid var(--rule);background:var(--surface);color:var(--ink);cursor:pointer}
.tab[aria-selected="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.tab:focus-visible,.btn:focus-visible,input:focus-visible{outline:2px solid #2a78d6;outline-offset:2px}
.controls{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;font-size:13px;color:var(--muted)}.controls label{display:flex;gap:8px;align-items:center}
.btn{font:500 12.5px/1 var(--sans);padding:7px 10px;border-radius:5px;border:1px solid var(--rule);background:var(--surface);color:var(--ink);cursor:pointer}
.btn:hover,.tab:hover{border-color:var(--muted)}input[type=range]{width:110px;accent-color:#2a78d6}
.panes{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:10px;align-items:end}.pane{display:flex;flex-direction:column;gap:6px}
.screen{position:relative;background:var(--screen);border-radius:6px;overflow:hidden;height:420px;touch-action:none}
.screen canvas{display:block;width:100%;height:100%;cursor:grab}
.screen .none{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:24px;color:#9aa4b1;font-size:13.5px}
.pane-head{display:flex;flex-direction:column;gap:2px;min-height:58px}.pane-head .name{font:600 14px/1.3 var(--cond)}
.pane-head .desc{font-size:12.5px;line-height:1.35;color:var(--muted)}.pane-head .meta{font:12px/1.3 var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--muted);align-items:center}
.lut{display:inline-block;width:120px;height:9px;border-radius:2px;vertical-align:-1px}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:14px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:16px;display:flex;flex-direction:column;gap:8px;min-width:0}.card p{font-size:13.5px;color:var(--muted)}
svg.chart{width:100%;height:auto;display:block;overflow:visible}svg.chart text{fill:var(--muted);font:13px var(--mono)}
svg.chart .grid{stroke:var(--rule2);stroke-width:1}svg.chart .axis{stroke:var(--rule);stroke-width:1}svg.chart .cross{stroke:var(--faint);stroke-width:1;stroke-dasharray:3 3}svg.chart .hit{fill:transparent}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--rule);border-radius:6px;padding:8px 10px;font:12px/1.45 var(--mono);box-shadow:0 6px 18px rgba(0,0,0,.12);z-index:10;white-space:nowrap}
.tip .k{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--rule);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--rule2);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{font:600 12px/1.3 var(--sans);letter-spacing:.03em;text-transform:uppercase;color:var(--muted)}
tr.grp td{font:600 12px/1.3 var(--sans);letter-spacing:.05em;text-transform:uppercase;color:var(--muted);text-align:left;background:var(--bg)}
td.better{color:var(--good);font-weight:500}#tall th:nth-child(2),#tall td:nth-child(2){text-align:left}.notes{display:flex;flex-direction:column;gap:8px;font-size:14px;color:var(--muted)}.notes li{max-width:95ch}
@media (max-width:760px){.screen{height:330px}.tabrow .lbl{min-width:0;width:100%}}
</style></head><body>
<div class="wrap">
<header>
  <div class="eyebrow" id="eyebrow"></div>
  <h1>__TITLE__</h1>
  <p class="muted" id="intro"></p>
  <div class="facts" id="facts"></div>
</header>
<section>
  <div class="sec-head"><h2>What each column is</h2><p class="muted">Paired chains start from the test event itself, so each can be compared with that event voxel by voxel. Going left to right adds one more learned stage, so the change between neighbouring columns is what that stage costs.</p></div>
  <div class="chains" id="chains"></div>
</section>
<section id="viewer">
  <div class="sec-head"><h2>3D view</h2><p class="muted">Photon-weighted light, accumulated along the line of sight and coloured by log Q. All panels rotate and zoom together. Drag to rotate, scroll or pinch to zoom.</p></div>
  <div class="tabrow" id="tabs-p"></div><div class="tabrow" id="tabs-g"></div>
  <div class="controls">
    <button class="btn" id="v-side">Side x–z</button><button class="btn" id="v-side2">Side y–z</button><button class="btn" id="v-top">Top x–y</button><button class="btn" id="v-reset">Reset</button>
    <label><input type="checkbox" id="spin"> Rotate</label>
    <label for="qmin">Min log Q <input type="range" id="qmin" min="0" max="250" value="0"></label>
    <label for="gain">Gain <input type="range" id="gain" min="2" max="80" value="18"></label>
    <label for="psize">Point <input type="range" id="psize" min="1" max="4" value="2"></label>
  </div>
  <div class="panes" id="panes"></div>
  <div class="legend"><span>log Q</span><span class="mono" id="lq-lo"></span><span class="lut" id="lut"></span><span class="mono" id="lq-hi"></span><span id="boxnote"></span></div>
</section>
<section>
  <div class="sec-head"><h2>Against truth, pooled over the events shown</h2><p class="muted" id="pooled-note"></p></div>
  <div class="tablewrap"><table id="t0"></table></div>
  <div class="grid2">
    <div class="card"><h3>Per-voxel log(Q + ε)</h3><p>Share of voxels per bin.</p><svg class="chart" id="h-q" viewBox="0 0 520 250"></svg></div>
    <div class="card"><h3>Nearest-neighbour distance [m]</h3><p>Log scale. Excess at the right means isolated voxels.</p><svg class="chart" id="h-nn" viewBox="0 0 520 250"></svg></div>
    <div class="card"><h3>Local linearity</h3><p>Share near 1.0 is voxels on straight tracks.</p><svg class="chart" id="h-lin" viewBox="0 0 520 250"></svg></div>
    <div class="card"><h3>Longitudinal photon share</h3><p>Median over events, 32 z slabs; the top of the box (largest z) is at the right.</p><svg class="chart" id="p-z" viewBox="0 0 520 250"></svg></div>
    <div class="card"><h3>Radial photon share</h3><p>Median over events, 16 rings (√r spacing), log scale.</p><svg class="chart" id="p-r" viewBox="0 0 520 250"></svg></div>
    <div class="card"><h3>Box-counting dimension</h3><p>Median over events; local slope between coarsening factors 1, 2, 4, 8, 16.</p><svg class="chart" id="p-box" viewBox="0 0 520 250"></svg></div>
  </div>
  <div class="legend" id="legend"></div>
</section>
<section>
  <div class="sec-head"><h2>Every sample</h2><p class="muted">Ratios are generated / truth of the same event. Cells per level are the active cells at 48³, 96³ and 192³.</p></div>
  <div class="tablewrap"><table id="tall"></table></div>
</section>
<section>
  <h2>How this page was made</h2>
  <ul class="notes" id="notes"></ul>
</section>
</div>
<div class="tip" id="tip" hidden></div>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id),SVGNS='http://www.w3.org/2000/svg';
const fmt=(v,d=3)=>v==null?'–':Number(v).toFixed(d),pct=v=>v==null?'–':(100*v).toFixed(1)+'%';
const sci=v=>{if(v==null||!v)return '–';const e=Math.floor(Math.log10(Math.abs(v)));return (v/10**e).toFixed(2)+'e'+e;};
const dark=()=>document.documentElement.dataset.theme==='dark'||(document.documentElement.dataset.theme!=='light'&&matchMedia('(prefers-color-scheme: dark)').matches);
const SER={truth:{name:'truth',colour:'var(--truth)'}};
D.series.forEach(s=>{SER[s.key]={name:s.name,colour:dark()?s.colour_dark:s.colour,desc:s.desc,unpaired:!!s.unpaired,steps:s.steps}});
const col=k=>SER[k]?SER[k].colour:'var(--muted)';
const G=D.grid,GEO=D.geometry;
$('eyebrow').textContent=`CORSIKA 8 in-ice Cherenkov · ${G}³ voxels of ${GEO.voxel_cm} cm · z ${GEO.z[0]}–${GEO.z[1]} m · case ${D.case}`+(D.synthetic?' · SYNTHETIC data':'');
const paired=D.tabs.filter(t=>t.kind==='paired'),gen=D.tabs.filter(t=>t.kind==='gen');
$('intro').textContent=`${paired.length} test events through ${D.series.filter(s=>!s.unpaired).length} paired chain(s)`+(gen.length?`, and ${gen.length} unconditional samples, each next to the nearest test event (not paired).`:'.');
// facts
const chips=[];D.series.forEach(s=>{const st=s.steps||{};const parts=Object.entries(st).map(([k,v])=>`${k} ${Number(v).toLocaleString()}`).join(' · ');chips.push(`<span class="chip"><b>${s.name}</b>${parts||'steps n/a'}</span>`)});
const sp=Object.values(D.spec||{})[0];if(sp){chips.push(`<span class="chip"><b>sampler</b>Heun ${sp.heun_steps} steps</span>`);chips.push(`<span class="chip"><b>occ threshold</b>${sp.occ_threshold}</span>`);}
chips.push(`<span class="chip"><b>made</b>${D.made}</span>`);$('facts').innerHTML=chips.join('');
// chain cards
$('chains').innerHTML=`<div class="chainc"><span class="n"><span class="swatch" style="background:var(--truth)"></span>truth</span><span class="d">The CORSIKA 8 event itself (test split, never trained on).</span></div>`+
 D.series.map(s=>`<div class="chainc"><span class="n"><span class="swatch" style="background:${col(s.key)}"></span>${s.name}</span><span class="d">${s.desc}</span><span class="s">${s.n} samples${s.steps&&Object.keys(s.steps).length?' · checkpoints '+Object.entries(s.steps).map(([k,v])=>k+' '+v).join(', '):''}</span></div>`).join('');
// tooltip + svg helpers
const tip=$('tip');
function showTip(e,h){tip.innerHTML=h;tip.hidden=false;const r=tip.getBoundingClientRect();let x=e.clientX+14,y=e.clientY+14;if(x+r.width>innerWidth-8)x=e.clientX-r.width-14;if(y+r.height>innerHeight-8)y=e.clientY-r.height-14;tip.style.left=x+'px';tip.style.top=y+'px';}
const hideTip=()=>tip.hidden=true;
const el=(tag,a,p)=>{const n=document.createElementNS(SVGNS,tag);for(const k in a)n.setAttribute(k,a[k]);p&&p.appendChild(n);return n;};
const key=k=>`<span class="k" style="background:${col(k)}"></span>`;
function niceStep(r){const e=Math.pow(10,Math.floor(Math.log10(r))),f=r/e;return (f<1.5?1:f<3?2:f<7?5:10)*e;}
function niceTicks(a,b,n){const st=niceStep((b-a)/n),o=[];for(let v=Math.ceil(a/st)*st;v<=b+1e-9;v+=st)o.push(+v.toFixed(10));return o;}
function chart(id,{x,series,step=false,logY=false,xlab='',xfmt=v=>v,yfmt=v=>v,xticks,yticks,y0=0}){
  const svg=$(id),W=520,H=250,L=62,R=16,T=12,B=40;series=series.filter(s=>s.v&&s.v.length);if(!series.length)return;
  const xs=step?x.slice(0,-1).map((v,i)=>(v+x[i+1])/2):x;
  const xmin=step?x[0]:Math.min(...x),xmax=step?x[x.length-1]:Math.max(...x);
  const vals=series.flatMap(s=>s.v).filter(v=>v!=null&&v>0);if(!vals.length)return;
  let ymin=logY?Math.pow(10,Math.floor(Math.log10(Math.min(...vals)))):y0,ymax=Math.max(...vals)*1.08;if(logY)ymax=Math.pow(10,Math.ceil(Math.log10(ymax)));
  const X=v=>L+(W-L-R)*(v-xmin)/(xmax-xmin),Y=v=>logY?(H-B)-(H-B-T)*(Math.log10(Math.max(v,ymin))-Math.log10(ymin))/(Math.log10(ymax)-Math.log10(ymin)):(H-B)-(H-B-T)*(v-ymin)/(ymax-ymin);
  const yt=yticks||(logY?(()=>{const a=[];for(let e=Math.log10(ymin);e<=Math.log10(ymax)+1e-9;e++)a.push(10**e);return a;})():niceTicks(ymin,ymax,4));
  yt.forEach(v=>{el('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),class:'grid'},svg);el('text',{x:L-8,y:Y(v)+4,'text-anchor':'end'},svg).textContent=yfmt(v);});
  (xticks||niceTicks(xmin,xmax,5)).forEach(v=>{el('text',{x:X(v),y:H-B+16,'text-anchor':'middle'},svg).textContent=xfmt(v);});
  el('line',{x1:L,x2:W-R,y1:H-B,y2:H-B,class:'axis'},svg);el('text',{x:(L+W-R)/2,y:H-2,'text-anchor':'middle'},svg).textContent=xlab;
  series.forEach(s=>{let d='';if(step){s.v.forEach((v,i)=>{const y=Y(v==null||(logY&&v<=0)?ymin:v);d+=(i?'L':'M')+X(x[i])+','+y+'L'+X(x[i+1])+','+y;});}
    else{let pen=false;s.v.forEach((v,i)=>{if(v==null||(logY&&v<=0)){pen=false;return;}d+=(pen?'L':'M')+X(xs[i])+','+Y(v);pen=true;});}
    el('path',{d,fill:'none',stroke:col(s.k),'stroke-width':s.k==='truth'?2.4:1.8,'stroke-dasharray':SER[s.k]&&SER[s.k].unpaired?'5 3':''},svg);});
  const cross=el('line',{x1:0,x2:0,y1:T,y2:H-B,class:'cross',visibility:'hidden'},svg);
  const hit=el('rect',{x:L,y:T,width:W-L-R,height:H-B-T,class:'hit'},svg);
  hit.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect(),px=(e.clientX-r.left)*W/r.width;let i=0,b=1e9;xs.forEach((v,j)=>{const d=Math.abs(X(v)-px);if(d<b){b=d;i=j;}});
    cross.setAttribute('x1',X(xs[i]));cross.setAttribute('x2',X(xs[i]));cross.setAttribute('visibility','visible');
    showTip(e,(step?`${xfmt(x[i])} – ${xfmt(x[i+1])}`:xfmt(xs[i]))+series.map(s=>`<br>${key(s.k)}${SER[s.k]?SER[s.k].name:s.k} ${yfmt(s.v[i],true)}`).join(''));});
  hit.addEventListener('mouseleave',()=>{cross.setAttribute('visibility','hidden');hideTip();});
}
const C=D.charts,ORDER=['truth',...D.series.map(s=>s.key)];
const ser=h=>ORDER.filter(k=>h[k]).map(k=>({k,v:h[k]})).reverse();
chart('h-q',{x:C.logq.edges,series:ser(C.logq),step:true,xlab:'log(Q + ε)',xfmt:v=>(+v).toFixed(1),yfmt:(v,t)=>(+v).toFixed(t?4:2)});
chart('h-nn',{x:C.nn.edges,series:ser(C.nn),step:true,logY:true,xlab:'distance [m]',xfmt:v=>(+v).toFixed(3),yfmt:(v,t)=>t?(+v).toExponential(2):String(+v)});
chart('h-lin',{x:C.lin.edges,series:ser(C.lin),step:true,xlab:'local linearity',xfmt:v=>(+v).toFixed(2),yfmt:(v,t)=>(+v).toFixed(t?4:2)});
chart('p-z',{x:C.z,series:ser(C.long),xlab:'z [m]',xfmt:v=>(+v).toFixed(1),yfmt:(v,t)=>(+v).toFixed(t?4:2),xticks:niceTicks(Math.min(...C.z),Math.max(...C.z),5)});
chart('p-r',{x:C.z.slice(0,16).map((_,i)=>i+1),series:ser(C.rad),logY:true,xlab:'ring (inner → outer)',xfmt:v=>String(v),yfmt:(v,t)=>t?(+v).toExponential(2):String(+v)});
chart('p-box',{x:[1.5,3,6,12],series:ser(C.box),xlab:'between coarsening factors',xticks:[1.5,3,6,12],xfmt:v=>({1.5:'1→2',3:'2→4',6:'4→8',12:'8→16'})[v],yfmt:(v,t)=>(+v).toFixed(t?3:1),y0:1.0});
$('legend').innerHTML=ORDER.map(k=>`<span><span class="swatch" style="background:${col(k)}"></span>${SER[k].name}${SER[k].unpaired?' (dashed, unpaired)':''}</span>`).join('');
$('pooled-note').textContent=`Paired chains are compared with the same ${D.events.length} test events. ${gen.length?'The unconditional run (dashed) is pooled over its '+gen.length+' gallery samples and compared with the same pooled truth; it has no partner events, so only the distribution rows apply to it.':''} Green marks the paired chain closest to ideal in each row.`;
// table 0
(function(){const rows=[['Active voxels, gen / truth','n_ratio',1,'r'],['Total photons, gen / truth','q_ratio',1,'r'],['Longitudinal profile L1','long_l1',0,'a'],['Radial profile L1','rad_l1',0,'a'],
  ['Top-5% photon share, gen / truth','top5_ratio',1,'r'],['NN spacing, gen / truth','nn_ratio',1,'r'],['W1 distance, linearity','w1_lin',0,'a'],['W1 distance, NN spacing [m]','w1_nn',0,'a'],['W1 distance, log Q','w1_logq',0,'a'],['Straight-track voxels (linearity > 0.95)','straight',null,'s']];
  const T=D.table;if(!T.length){$('t0').innerHTML='<tbody><tr><td>no runs</td></tr></tbody>';return;}
  const ts=(T.find(r=>r.truth_straight!=null)||{}).truth_straight;
  let h=`<thead><tr><th>Metric</th><th>Ideal</th>${T.map(r=>`<th><span class="swatch" style="background:${col(r.key)}"></span>${SER[r.key].name}</th>`).join('')}</tr></thead><tbody>`;
  rows.forEach(([lab,k,ideal,mode])=>{const id=mode==='s'?ts:ideal;const dist=v=>v==null?1e9:mode==='r'?Math.abs(Math.log(v)):mode==='s'?Math.abs(v-id):Math.abs(v);
    const vals=T.map(r=>r[k]);const cand=T.filter(r=>!r.unpaired).map(r=>r[k]);const best=Math.min(...cand.map(dist));
    const f=v=>v==null?'–':mode==='s'?pct(v):(Math.abs(v)<0.01&&v!==0?v.toExponential(2):fmt(v,3));
    h+=`<tr><td>${lab}</td><td class="num">${mode==='s'?pct(id)+' (truth)':ideal}</td>${T.map((r,j)=>{const v=vals[j];return `<td class="num${v!=null&&!r.unpaired&&dist(v)===best&&cand.length>1?' better':''}">${f(v)}</td>`;}).join('')}</tr>`;});
  $('t0').innerHTML=h+'</tbody>';})();
// every sample
(function(){let s=`<thead><tr><th>Tab</th><th>Chain</th><th>Voxels</th><th>Voxels / truth</th><th>Photons</th><th>Photons / truth</th><th>Long L1</th><th>Radial L1</th><th>Straight</th><th>Cells per level</th><th>Time</th></tr></thead><tbody>`;
  D.tabs.forEach(t=>{s+=`<tr class="grp"><td colspan="11">${t.kind==='paired'?'test event '+t.event:'unconditional '+t.label+' · nearest truth event '+t.event+' (not paired)'}</td></tr>`;
    t.panels.forEach(p=>{if(p.missing){s+=`<tr><td></td><td>${p.name}</td><td colspan="9" class="muted">event not in this run</td></tr>`;return;}
      s+=`<tr><td></td><td><span class="swatch" style="background:${col(p.key)}"></span>${p.name}</td><td class="num">${p.n.toLocaleString()}</td><td class="num">${p.n_ratio!=null?fmt(p.n_ratio,3):'–'}</td><td class="num">${sci(p.q)}</td><td class="num">${p.q_ratio!=null?fmt(p.q_ratio,3):'–'}</td>
      <td class="num">${p.long_l1!=null?fmt(p.long_l1,3):'–'}</td><td class="num">${p.rad_l1!=null?fmt(p.rad_l1,3):'–'}</td><td class="num">${p.straight!=null?pct(p.straight):'–'}</td>
      <td class="num">${p.counts&&p.counts.length?p.counts.map(v=>v.toLocaleString()).join(' → '):'–'}</td><td class="num">${p.seconds!=null?p.seconds.toFixed(0)+' s':'–'}</td></tr>`;});});
  $('tall').innerHTML=s+'</tbody>';})();
// notes
$('notes').innerHTML=[`Source: <span class="mono">&lt;paths.output&gt;/${D.case}/samples/&lt;run&gt;/</span> for the generated showers and the voxel cache for truth. Checkpoint steps per run are the ones recorded in each run's <span class="mono">checkpoints.json</span>.`,
 `Metrics use the helpers of <span class="mono">sparseshower.evaluate</span> (profiles, W1) and <span class="mono">sparseshower.geometry</span> (nearest neighbours, local linearity from the 8 nearest neighbours, box counting) on voxel centres plus sub-voxel offsets.`,
 `The unconditional samples have no partner event. The truth shown next to each is the test event nearest in standardized event features (the first 200 test events); it is a visual reference, not a comparison.`,
 `${D.events.length} events is a small sample; use <span class="mono">evaluate</span> on the full runs for numbers with confidence intervals.`,
 `The 3D panels draw voxel centres; sub-voxel offsets are left out at this zoom level. Colour scale: log Q from the 0.2% to the 99.9% quantile of all panels.`+(D.max_points?` A panel with more than ${D.max_points.toLocaleString()} voxels draws a random subset of that size (its header then says so); the statistics always use every voxel.`:'')].map(t=>`<li>${t}</li>`).join('');
// 3D viewer
const LUT=D.lut;$('lq-lo').textContent=D.lq_range[0].toFixed(1);$('lq-hi').textContent=D.lq_range[1].toFixed(1);
$('boxnote').textContent=`· box ${GEO.box_m} m on a side, axes x red, y green, z blue`;
$('lut').style.background='linear-gradient(90deg,'+[0,64,128,192,255].map(i=>`rgb(${[0,1,2].map(k=>Math.round(255*Math.min(1,LUT[3*i+k]/0.6))).join(',')})`).join(',')+')';
function dec(b){const s=atob(b),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a;}
const V={yaw:-0.6,pitch:0.3,zoom:1};let panes=[];
function build(idx){const t=D.tabs[idx],box=$('panes');box.innerHTML='';
  document.querySelectorAll('.tab').forEach(b=>b.setAttribute('aria-selected',+b.dataset.i===idx?'true':'false'));
  const truth=t.panels.find(p=>p.key==='truth');
  panes=t.panels.map(p=>{const d=document.createElement('div');d.className='pane';
    const meta=p.missing?'event not in this run':`${p.n.toLocaleString()} voxels · Q ${sci(p.q)}`+(p.n_ratio!=null?` · ${fmt(p.n_ratio,3)}× / ${fmt(p.q_ratio,3)}× truth`:'')+(p.drawn<p.n?` · drawing ${p.drawn.toLocaleString()}`:'');
    d.innerHTML=`<div class="pane-head"><span class="name" style="color:${p.key==='truth'?'var(--ink)':col(p.key)}">${p.name}</span><span class="desc">${p.desc||''}</span><span class="meta">${meta}</span></div><div class="screen"></div>`;
    box.appendChild(d);const sc=d.querySelector('.screen');
    if(p.missing||!p.b64){sc.innerHTML=`<div class="none">This event is not in ${p.name}.</div>`;return null;}
    const cv=document.createElement('canvas');sc.appendChild(cv);bind(cv);const data=dec(p.b64);return {cv,sc,data,n:data.length/4};}).filter(Boolean);
  size();}
function size(){panes.forEach(p=>{const r=p.sc.getBoundingClientRect(),dpr=Math.min(2,window.devicePixelRatio||1);p.cv.width=Math.round(r.width*dpr);p.cv.height=Math.round(r.height*dpr);p.dpr=dpr;});draw();}
function draw(){const qmin=+$('qmin').value,gain=+$('gain').value/10,sz=+$('psize').value;
  const cy=Math.cos(V.yaw),sy=Math.sin(V.yaw),cp=Math.cos(V.pitch),sp=Math.sin(V.pitch),h=G/2;
  for(const p of panes){const W=p.cv.width,H=p.cv.height,ctx=p.cv.getContext('2d'),dp=p.dpr,ps=Math.max(1,Math.round(sz*dp/1.5*Math.max(1,192/G)));
    const sc=V.zoom*Math.min(W,H)/(G*1.85),buf=new Float32Array(W*H*3),d=p.data;
    for(let i=0;i<p.n;i++){const o=4*i,ci=d[o+3];if(ci<qmin)continue;
      const x=d[o]-h,y=d[o+1]-h,z=d[o+2]-h,x1=x*cy-y*sy,y1=x*sy+y*cy,z2=y1*sp+z*cp;
      const u=(W/2+x1*sc)|0,v=(H/2-z2*sc)|0,r=LUT[3*ci],g=LUT[3*ci+1],b=LUT[3*ci+2];
      for(let a=0;a<ps;a++)for(let c=0;c<ps;c++){const uu=u+a,vv=v+c;if(uu<0||vv<0||uu>=W||vv>=H)continue;const k=3*(vv*W+uu);buf[k]+=r;buf[k+1]+=g;buf[k+2]+=b;}}
    const img=ctx.createImageData(W,H),px=img.data;
    for(let k=0,j=0;k<buf.length;k+=3,j+=4){px[j]=5+250*(1-Math.exp(-gain*buf[k]));px[j+1]=7+248*(1-Math.exp(-gain*buf[k+1]));px[j+2]=10+245*(1-Math.exp(-gain*buf[k+2]));px[j+3]=255;}
    ctx.putImageData(img,0,0);
    const P=(x,y,z)=>{x-=h;y-=h;z-=h;const x1=x*cy-y*sy,y1=x*sy+y*cy,z2=y1*sp+z*cp;return [W/2+x1*sc,H/2-z2*sc];};
    ctx.strokeStyle='rgba(160,172,188,0.35)';ctx.lineWidth=dp;ctx.beginPath();
    const E=[[0,0,0,G,0,0],[0,G,0,G,G,0],[0,0,G,G,0,G],[0,G,G,G,G,G],[0,0,0,0,G,0],[G,0,0,G,G,0],[0,0,G,0,G,G],[G,0,G,G,G,G],[0,0,0,0,0,G],[G,0,0,G,0,G],[0,G,0,0,G,G],[G,G,0,G,G,G]];
    for(const e of E){const a=P(e[0],e[1],e[2]),b=P(e[3],e[4],e[5]);ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);}ctx.stroke();
    ctx.font=`${12*dp}px ui-monospace,monospace`;const o0=P(0,0,0);
    for(const [l,x,y,z,c] of [['x',G*1.1,0,0,'#e66767'],['y',0,G*1.1,0,'#3fbf8a'],['z',0,0,G*1.1,'#5b9cf0']]){const q=P(x,y,z);
      ctx.strokeStyle=c;ctx.fillStyle=c;ctx.lineWidth=2*dp;ctx.beginPath();ctx.moveTo(o0[0],o0[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.fillText(l,q[0]+4*dp,q[1]-4*dp);}}}
let drag=null;
function bind(cv){cv.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId);});
  cv.addEventListener('pointermove',e=>{if(!drag)return;V.yaw+=(e.clientX-drag[0])*0.008;V.pitch=Math.max(-1.57,Math.min(1.57,V.pitch+(e.clientY-drag[1])*0.008));drag=[e.clientX,e.clientY];draw();});
  cv.addEventListener('pointerup',()=>drag=null);cv.addEventListener('pointercancel',()=>drag=null);
  cv.addEventListener('wheel',e=>{e.preventDefault();V.zoom=Math.max(0.4,Math.min(6,V.zoom*Math.exp(-e.deltaY*0.001)));draw();},{passive:false});}
const setV=(y,p)=>{V.yaw=y;V.pitch=p;draw();};
$('v-side').onclick=()=>setV(0,0);$('v-side2').onclick=()=>setV(-Math.PI/2,0);$('v-top').onclick=()=>setV(0,Math.PI/2);$('v-reset').onclick=()=>{V.zoom=1;setV(-0.6,0.3);};
['qmin','gain','psize'].forEach(i=>$(i).addEventListener('input',draw));
const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
(function tick(){if($('spin').checked&&!reduce){V.yaw+=0.01;draw();}requestAnimationFrame(tick);})();
const tabBtn=(t,i)=>`<button class="tab" role="tab" data-i="${i}">${t.kind==='paired'?'event '+t.event:t.label}</button>`;
$('tabs-p').innerHTML=paired.length?'<span class="lbl">Paired · test event</span>'+D.tabs.map((t,i)=>t.kind==='paired'?tabBtn(t,i):'').join(''):'';
$('tabs-g').innerHTML=gen.length?'<span class="lbl">Unconditional</span>'+D.tabs.map((t,i)=>t.kind==='gen'?tabBtn(t,i):'').join(''):'';
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>build(+b.dataset.i));
let rt;window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(size,120);});
if(D.tabs.length)build(0);
</script></body></html>
"""


if __name__ == "__main__":
    main()
