"""Active voxels (nphotons > 0) per event vs grid resolution, from the RAW parquet.

Uses the same row selection as the cache (sparseshower.data.select_rows: finite,
NPhotons > 0, time < time_max, inside data.ranges), then voxelises the kept
photons at every grid in --grids in one pass over each file.

Output (never overwrites): <output>/active_vs_grid[_<stamp>]/
  counts.csv   one row per event: eid, n_rows, q_total, n_<G> for every grid
  summary.json median / p5 / p95 per grid, growth between grids, log-log slope
  active_vs_grid.png

  python scripts/active_vs_grid.py --config configs/v2.yaml --events 300 \
      --grids 32,48,64,96,128,192,256,384
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparseshower.common import config, read_json  # noqa: E402
from sparseshower.data import iter_batches, select_rows  # noqa: E402
from sparseshower.geometry import voxel_index  # noqa: E402


def event_ids(c, n, split):
    """First n event ids of a split (from metadata.json, or the config range without a cache)."""
    meta = Path(c["paths"]["processed"]) / "metadata.json"
    if meta.exists():
        ids = read_json(meta)["split"][split]
    else:
        ids = list(range(int(c["data"]["event_start"]), int(c["data"]["event_stop"])))
    return [int(i) for i in ids[:n]]


def count_event(path, d, grids):
    """Active voxels of one raw event at every grid, in one pass over the file."""
    keys = {g: [] for g in grids}
    rows, q = 0, 0.0
    for cols in iter_batches(path, int(d["parquet_batch_rows"])):
        sel = select_rows(cols, d)
        if not len(sel["w"]):
            continue
        rows += len(sel["w"])
        q += float(sel["w"].sum())
        for g in grids:
            ijk = voxel_index(sel["xyz"], d["ranges"], g).astype(np.int64)
            keys[g].append(np.unique((ijk[:, 2] * g + ijk[:, 1]) * g + ijk[:, 0]))
    counts = {g: int(len(np.unique(np.concatenate(k)))) if k else 0 for g, k in keys.items()}
    return rows, q, counts


def plot(grids, n, out):
    """Median and 5-95% band of active voxels vs grid (log-log)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = np.asarray(grids, float)
    med, lo, hi = (np.percentile(n, p, axis=0) for p in (50, 5, 95))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    a.fill_between(g, lo, hi, color="0.8", label="5-95% of events")
    a.plot(g, med, "o-", color="k", lw=2, label="median active cells")
    a.plot(g, g ** 3, "--", color="C3", label="all cells N$^3$ (dense grid)")
    for gi, m in zip(g, med):
        a.annotate(f"{m:,.0f}", (gi, m), textcoords="offset points", xytext=(-10, 10), fontsize=9)
    b = ax[1]
    occ = n / g ** 3 * 100
    b.fill_between(g, np.percentile(occ, 5, 0), np.percentile(occ, 95, 0), color="0.8")
    b.plot(g, np.median(occ, 0), "o-", color="k", lw=2)
    for x in ax:
        x.set_xscale("log", base=2); x.set_yscale("log"); x.set_xticks(g)
        x.set_xticklabels([f"{int(v)}$^3$" for v in g]); x.grid(alpha=.3, which="both")
        x.set_xlabel("grid resolution (same box)")
    a.set_ylabel("cells per event"); a.legend(fontsize=9, loc="upper left")
    a.set_title("active cells (nphotons > 0) vs dense grid")
    b.set_ylabel("occupied fraction [%]"); b.set_title("occupied fraction")
    fig.suptitle(f"Real data: {len(n)} events, voxelised from raw parquet at each grid", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=170)


def main():
    """Parse options, count every event, write counts.csv, summary.json and the plot."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--events", type=int, default=300)
    ap.add_argument("--split", default="train")
    ap.add_argument("--grids", default="32,48,64,96,128,192,256,384")
    ap.add_argument("--name", default="active_vs_grid")
    a = ap.parse_args()
    c = config(a.config)
    d = dict(c["data"])
    grids = sorted(int(x) for x in a.grids.split(","))
    out = Path(c["paths"]["output"]) / a.name
    if out.exists():
        out = out.with_name(f"{a.name}_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True)
    raw = Path(c["paths"]["raw"])
    ids = event_ids(c, a.events, a.split)
    rows_out, n = [], []
    t0 = time.time()
    for k, eid in enumerate(ids):
        path = raw / d["source_pattern"].format(eid=eid)
        if not path.exists():
            print(f"skip {eid}: {path} missing", flush=True)
            continue
        rows, q, counts = count_event(path, d, grids)
        rows_out.append(dict(eid=eid, n_rows=rows, q_total=q, **{f"n_{g}": counts[g] for g in grids}))
        n.append([counts[g] for g in grids])
        if k % 20 == 0:
            print(f"{k + 1}/{len(ids)} eid {eid} {counts} {time.time() - t0:.0f}s", flush=True)
    if not n:
        raise SystemExit("no events processed")
    with open(out / "counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)
    n = np.asarray(n, float)
    med = np.median(n, 0)
    summary = dict(events=len(n), grids=grids, box_m=d["ranges"], time_max_ns=d.get("time_max"),
                   median=med.tolist(), p5=np.percentile(n, 5, 0).tolist(),
                   p95=np.percentile(n, 95, 0).tolist(),
                   occupied_frac_median=(med / np.asarray(grids, float) ** 3).tolist(),
                   growth_between_grids=(med[1:] / med[:-1]).tolist(),
                   slope_loglog=(np.diff(np.log(med)) / np.diff(np.log(grids))).tolist())
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    plot(grids, n, out / "active_vs_grid.png")
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
