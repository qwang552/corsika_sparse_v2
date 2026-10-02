"""Self-contained 3D comparison page (no external scripts; opens offline).

Two sections:
* per event   one row per test event: truth | each paired run (sa_truth,
              sa_recon, sdedit ...) | optional v1 run.  Same event, same view.
* gallery     the first N samples of an unconditional (`full`) run, each next to
              the nearest of the first 200 test events in standardized
              event-feature space, labelled "visual reference only, not paired".

Every panel title names its source.  Drag to rotate (all panels together),
wheel to zoom, double-click to reset.  Points are sub-sampled per panel
(brightest first, then random) and coloured by log Q.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from .data import load_event
from .geometry import voxel_center
from .loader import case_dir, meta_for

MAX_POINTS = 5000


def _points(ijk, q, off, c, rng, max_points=MAX_POINTS):
    """3D points (voxel centre + offset) and photon weights of one voxel set, subsampled if large.
    """
    ranges = np.asarray(c["data"]["ranges"], dtype=np.float64)
    grid = int(c["data"]["grid"])
    voxel = (ranges[:, 1] - ranges[:, 0]) / grid
    xyz = voxel_center(ijk, ranges, grid) + np.asarray(off, dtype=np.float64) * voxel
    q = np.asarray(q, dtype=np.float64)
    n = len(q)
    if n > max_points:
        top = np.argsort(q)[::-1][: max_points // 2]
        rest = np.setdiff1d(np.arange(n), top)
        pick = np.concatenate([top, rng.choice(rest, max_points - len(top), replace=False)])
    else:
        pick = np.arange(n)
    centre = ranges.mean(axis=1)
    p = np.rint((xyz[pick] - centre) * 1000).astype(int)          # mm, centred
    lq = np.log10(np.maximum(q[pick], 1e-3))
    return dict(p=p.reshape(-1).tolist(), c=np.round(lq, 2).tolist(), n=int(n), qt=float(q.sum()))


def build_page(c, case, paired_runs, full_run=None, v1_dir=None, events=None, n_gallery=6, title=None):
    """Collect the per-event rows and the gallery and fill the HTML template."""
    from .evaluate import features, truth_metrics
    from .sample import load_run, load_v1_run

    rng = np.random.default_rng(0)
    meta = meta_for(c)
    events = [int(e) for e in (events or c["viz"]["events"])]
    root = c["paths"]["processed"]
    runs = {r: {x["eid"]: x for x in load_run(case_dir(c, case) / "samples" / r)} for r in paired_runs}
    v1 = {x["eid"]: x for x in load_v1_run(v1_dir)} if v1_dir else {}
    rows = []
    for e in events:
        ev = load_event(root, e)
        panels = [dict(title=f"truth · event {e}", kind="truth", **_points(ev["ijk"], ev["q"], ev["off"], c, rng))]
        for name, byeid in runs.items():
            if e in byeid:
                s = byeid[e]
                panels.append(dict(title=f"{name}", kind="gen", **_points(s["ijk"], s["q"], s["off"], c, rng)))
            else:
                panels.append(dict(title=f"{name} (event not in run)", kind="missing", p=[], c=[], n=0, qt=0.0))
        if v1:
            s = v1.get(e)
            panels.append(dict(title="v1 chain (macro)", kind="v1", **(_points(s["ijk"], s["q"], s["off"], c, rng)
                                                                        if s else dict(p=[], c=[], n=0, qt=0.0))))
        rows.append(dict(label=f"event {e}", panels=panels))
    gallery = []
    if full_run:
        gen = load_run(case_dir(c, case) / "samples" / full_run)[: int(n_gallery)]
        test = [int(x) for x in meta["split"]["test"]][:200]
        tm = {t: truth_metrics(c, t) for t in test}
        F = np.stack([features(tm[t]) for t in test])
        mu, sd = np.nanmean(F, 0), np.nanstd(F, 0) + 1e-9
        from .evaluate import event_metrics
        for s in gen:
            fg = (features(event_metrics(s["ijk"], s["q"], s["off"], c)) - mu) / sd
            d = np.nansum(((F - mu) / sd - fg) ** 2, axis=1)
            t = test[int(np.argmin(d))]
            ev = load_event(root, t)
            gallery.append(dict(label=f"generated #{s['index']}", panels=[
                dict(title=f"{full_run} #{s['index']} (unconditional)", kind="gen",
                     **_points(s["ijk"], s["q"], s["off"], c, rng)),
                dict(title=f"nearest truth: event {t} (visual reference only, not paired)", kind="truth",
                     **_points(ev["ijk"], ev["q"], ev["off"], c, rng))]))
    data = dict(rows=rows, gallery=gallery, case=case, paired=list(paired_runs), full=full_run,
                v1=bool(v1_dir))
    title = (title or f"v2 chains · {case}") + (" [SYNTHETIC data]" if c.get("synthetic") else "")
    return PAGE.replace("__TITLE__", html.escape(title)) \
               .replace("__DATA__", json.dumps(data, separators=(",", ":")))


def export(c, case, paired_runs, full_run=None, v1_dir=None, events=None, n_gallery=12, out=None):
    """Write the page.  An explicit `out` that exists is refused; the default name
    (<case>/viz/chains_3d.html) gets a timestamp instead.  Nothing is overwritten."""
    import time

    page = build_page(c, case, paired_runs, full_run, v1_dir, events, n_gallery)
    if out and Path(out).exists():
        raise FileExistsError(f"{out} exists; pass another --out (nothing is overwritten)")
    out = Path(out or case_dir(c, case) / "viz" / "chains_3d.html")
    if out.exists():
        out = out.with_name(f"{out.stem}_{time.strftime('%Y%m%d-%H%M%S')}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return str(out)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f4f5f7;--surface:#fff;--ink:#16191d;--muted:#5e6670;--rule:#dde1e6;--screen:#07090c}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--surface:#171a1f;--ink:#e7e9ec;--muted:#9aa2ad;--rule:#2a2f36}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:24px 16px 48px;display:flex;flex-direction:column;gap:28px}
h1{font-size:26px;margin:0}h2{font-size:19px;margin:0 0 6px}p{margin:0;color:var(--muted);max-width:80ch}
.row{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:8px}
.lab{font-weight:600}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}
.cell{display:flex;flex-direction:column;gap:4px}.cell .t{font-size:12.5px;font-weight:600}.cell .m{font:12px ui-monospace,monospace;color:var(--muted)}
canvas{width:100%;aspect-ratio:1/1;background:var(--screen);border-radius:6px;cursor:grab;touch-action:none}
.legend{display:flex;align-items:center;gap:8px;font:12px ui-monospace,monospace;color:var(--muted)}
.bar{width:160px;height:10px;border-radius:3px;background:linear-gradient(90deg,#3b4cc0,#7aa6f7,#e8e8e8,#f5a582,#b40426)}
</style></head><body><div class="wrap">
<header><h1>__TITLE__</h1><p>Each panel names its source. Rows under “per event” show one test event through every paired chain; the gallery shows unconditional samples next to the nearest test event, which is a visual reference only. Drag to rotate all panels, wheel to zoom, double-click to reset. Up to 5,000 points per panel (brightest half, then random).</p>
<div class="legend">log10 Q <span class="bar"></span><span id="lq"></span></div></header>
<section id="paired"><h2>Per event (paired chains)</h2><div class="rows"></div></section>
<section id="gallery"><h2>Unconditional gallery</h2><div class="rows"></div></section>
</div>
<script>
const D=__DATA__;
const views=[];let yaw=0.6,pitch=0.35,zoom=1;
let lo=1e9,hi=-1e9;for(const R of [...D.rows,...D.gallery])for(const P of R.panels)for(const v of P.c){if(v<lo)lo=v;if(v>hi)hi=v}
if(!(hi>lo)){lo=0;hi=1}document.getElementById('lq').textContent=lo.toFixed(1)+' … '+hi.toFixed(1);
const stops=[[59,76,192],[122,166,247],[232,232,232],[245,165,130],[180,4,38]];
function col(v){let t=(v-lo)/(hi-lo);t=Math.max(0,Math.min(1,t))*(stops.length-1);const i=Math.min(stops.length-2,Math.floor(t)),f=t-i,a=stops[i],b=stops[i+1];return `rgb(${a[0]+(b[0]-a[0])*f|0},${a[1]+(b[1]-a[1])*f|0},${a[2]+(b[2]-a[2])*f|0})`}
function draw(v){const cv=v.cv,ctx=cv.getContext('2d'),w=cv.width=cv.clientWidth*devicePixelRatio,h=cv.height=cv.clientHeight*devicePixelRatio;
ctx.fillStyle='#07090c';ctx.fillRect(0,0,w,h);const P=v.P,n=P.c.length;if(!n)return;
const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),s=zoom*w/4200;
const idx=new Array(n),zz=new Float32Array(n),xs=new Float32Array(n),ys=new Float32Array(n);
for(let i=0;i<n;i++){const x=P.p[3*i],y=P.p[3*i+1],z=P.p[3*i+2];const x1=cy*x-sy*y,y1=sy*x+cy*y;const y2=cp*y1-sp*z,z2=sp*y1+cp*z;xs[i]=w/2+s*x1;ys[i]=h/2-s*z2;zz[i]=y2;idx[i]=i}
idx.sort((a,b)=>zz[b]-zz[a]);const r=Math.max(1,1.3*devicePixelRatio);
for(const i of idx){ctx.fillStyle=col(P.c[i]);ctx.fillRect(xs[i]-r/2,ys[i]-r/2,r,r)}}
function redraw(){for(const v of views)draw(v)}
function mount(sel,rows){const root=document.querySelector(sel+' .rows');if(!rows.length){document.querySelector(sel).style.display='none';return}
for(const R of rows){const row=document.createElement('div');row.className='row';row.innerHTML=`<div class="lab">${R.label}</div>`;const g=document.createElement('div');g.className='grid';
for(const P of R.panels){const cell=document.createElement('div');cell.className='cell';cell.innerHTML=`<div class="t"></div><canvas></canvas><div class="m"></div>`;
cell.querySelector('.t').textContent=P.title;cell.querySelector('.m').textContent=P.n?`${P.n.toLocaleString()} voxels · Q ${P.qt.toExponential(3)}`:'—';
const cv=cell.querySelector('canvas');views.push({cv,P});g.appendChild(cell)}row.appendChild(g);root.appendChild(row)}}
mount('#paired',D.rows);mount('#gallery',D.gallery);
let drag=null;addEventListener('pointerdown',e=>{if(e.target.tagName==='CANVAS'){drag=[e.clientX,e.clientY];e.target.setPointerCapture(e.pointerId)}});
addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*0.01;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*0.01));drag=[e.clientX,e.clientY];redraw()});
addEventListener('pointerup',()=>drag=null);
addEventListener('wheel',e=>{if(e.target.tagName!=='CANVAS')return;e.preventDefault();zoom*=Math.exp(-e.deltaY*0.001);redraw()},{passive:false});
addEventListener('dblclick',e=>{if(e.target.tagName==='CANVAS'){yaw=0.6;pitch=0.35;zoom=1;redraw()}});
addEventListener('resize',redraw);redraw();
</script></body></html>
"""
