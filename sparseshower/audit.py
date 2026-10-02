"""Data audit of the voxel cache: active voxels, box counts, nearest-neighbour
distances, local linearity and photon concentration.

Run with `python -m sparseshower.cli audit` (no job file; it is fast). It reads
the first audit.events train events, prints a summary and writes
<paths.output>/audit.json.

These numbers describe what survives voxelisation, so they are the reference
for what the generative model can be expected to reproduce.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import read_json, write_json
from .data import Codec, load_event
from .geometry import box_count, box_dimension, knn_within_grid, local_linearity, voxel_center


def audit_event(ev, c, k=8):
    """Audit numbers of one cached event (active voxels, box counts, linearity, ...)."""
    grid = int(c["data"]["grid"])
    ranges = np.asarray(c["data"]["ranges"], dtype=np.float64)
    voxel = (ranges[:, 1] - ranges[:, 0]) / grid
    ijk = np.asarray(ev["ijk"], dtype=np.int64)
    q = np.asarray(ev["q"], dtype=np.float64)
    points = voxel_center(ijk, ranges, grid) + np.asarray(ev["off"], dtype=np.float64) * voxel
    factors = [f for f in (1, 2, 4, 8, 16) if grid % f == 0]
    counts = box_count(ijk, factors)
    dist, _ = knn_within_grid(points, ijk, grid, 2)
    nn = dist[:, 1]
    nn = nn[np.isfinite(nn)]
    lin = local_linearity(points, ijk, grid, k=k)
    order = np.sort(q)[::-1]
    share = np.cumsum(order) / max(order.sum(), 1e-30)
    top = lambda frac: float(share[min(len(share) - 1, int(frac * len(order)))])  # noqa: E731
    coh = np.linalg.norm(np.asarray(ev["m"], dtype=np.float64), axis=1)
    return dict(
        n_active=int(len(ijk)),
        box_factors=factors, box_counts=counts,
        box_dimension=box_dimension(counts, factors),
        nn_median_m=float(np.median(nn)) if len(nn) else float("nan"),
        linearity_median=float(np.median(lin)), linearity_frac_gt_08=float((lin > 0.8).mean()),
        q_top1=top(0.01), q_top5=top(0.05), q_top20=top(0.20),
        records_per_voxel=float(np.asarray(ev["cnt"]).mean()),
        coherence_median=float(np.median(coh)),
        q_total=float(q.sum()),
    )


def audit(c, n_events=16, k=8, out_name="audit.json"):
    """Audit the first `n_events` train events; writes <paths.output>/audit.json and returns it."""
    root = Path(c["paths"]["processed"])
    meta = read_json(root / "metadata.json")
    ids = meta["split"]["train"][: int(n_events)]
    if not ids:
        raise RuntimeError("no training events to audit")
    per_event, channel_logq, channel_off = [], [], []
    codec = Codec(meta["stats"], float(c["data"]["q_eps"]))
    for eid in ids:
        ev = load_event(root, eid)
        per_event.append(audit_event(ev, c, k=k))
        x = codec.encode(ev["q"], ev["off"])
        channel_logq.append(x[:, 0])
        channel_off.append(x[:, 1:4])
    med = lambda key: float(np.median([p[key] for p in per_event]))  # noqa: E731
    counts = np.array([p["box_counts"] for p in per_event], dtype=np.float64)
    factors = per_event[0]["box_factors"]
    summary = dict(
        events=len(per_event),
        n_active_min=int(min(p["n_active"] for p in per_event)),
        n_active_median=med("n_active"),
        n_active_max=int(max(p["n_active"] for p in per_event)),
        box_factors=factors,
        box_counts_median=np.median(counts, axis=0).tolist(),
        box_dimension_median=box_dimension(np.median(counts, axis=0), factors),
        nn_median_m=med("nn_median_m"),
        linearity_median=med("linearity_median"),
        linearity_frac_gt_08=med("linearity_frac_gt_08"),
        records_per_voxel_median=med("records_per_voxel"),
        coherence_median=med("coherence_median"),
        photon_share_top1=med("q_top1"), photon_share_top5=med("q_top5"),
        photon_share_top20=med("q_top20"),
        channel_std_logq=float(np.std(np.concatenate(channel_logq))),
        channel_std_off=float(np.std(np.concatenate(channel_off))),
        grid=int(c["data"]["grid"]),
    )
    # What the next finer grid would cost, from the finest measured slope.
    if len(summary["box_dimension_median"]):
        d_fine = summary["box_dimension_median"][0]
        summary["predicted_next_grid_factor"] = float(2 ** d_fine)
        summary["predicted_next_grid_active"] = float(summary["n_active_median"] * 2 ** d_fine)
    out = dict(summary=summary, per_event=per_event)
    write_json(Path(c["paths"]["output"]) / out_name, out)
    return out


def format_audit(result):
    """Human-readable text version of the audit summary."""
    s = result["summary"]
    lines = [
        f"events audited        : {s['events']}",
        f"active voxels         : min {s['n_active_min']}  median {s['n_active_median']:.0f}  max {s['n_active_max']}",
        f"box counts            : {['%d' % v for v in s['box_counts_median']]} at factors {s['box_factors']}",
        f"box dimension         : {[round(v, 2) for v in s['box_dimension_median']]}",
        f"-> next finer grid    : x{s.get('predicted_next_grid_factor', float('nan')):.2f}"
        f"  (~{s.get('predicted_next_grid_active', float('nan')):.0f} active voxels)",
        f"nearest neighbour     : {s['nn_median_m'] * 100:.2f} cm (median)",
        f"local linearity       : median {s['linearity_median']:.3f}, frac>0.8 {s['linearity_frac_gt_08']:.3f}",
        f"records per voxel     : {s['records_per_voxel_median']:.1f}",
        f"direction coherence   : {s['coherence_median']:.3f}",
        f"photon share top 1/5/20%: {s['photon_share_top1']:.3f} / {s['photon_share_top5']:.3f} / {s['photon_share_top20']:.3f}",
        f"channel std logQ/off  : {s['channel_std_logq']:.3f} / {s['channel_std_off']:.3f} (both should be ~1)",
    ]
    return "\n".join(lines)
