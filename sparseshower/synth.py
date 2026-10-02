"""Synthetic CORSIKA-like Cherenkov records.

Used by the self-test and by anyone who wants to exercise the pipeline without
cluster data.  The generator makes a branching bundle of straight-ish tracks
with ~1 cm steps, photon weights along each step and a light-front time, i.e.
the same shape of data the real writer produces (NPhotons, time, posX/Y/Z,
dirX/Y/Z) - not the same physics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

C_M_PER_NS = 0.299792458


def _unit(v):
    """Normalise a vector (fallback: straight down)."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([0.0, 0.0, -1.0])


def synth_event(seed, ranges, step=0.0105, max_records=20000, branches=6, depth=3):
    """One synthetic shower; returns a dict of 1-D arrays.

    `depth` is the maximum branch generation; `branches` only caps the number of pending branches.
    """
    rng = np.random.default_rng(seed)
    ranges = np.asarray(ranges, dtype=np.float64)
    top = ranges[2, 1] - 0.05
    axis = np.array([0.0, 0.0, -1.0])
    # (start, direction, weight scale, remaining length, generation)
    stack = [(np.array([0.0, 0.0, top]), axis.copy(), 1.0, 2.2, 0)]
    pos_all, dir_all, w_all, t_all = [], [], [], []
    total = 0
    while stack and total < max_records:
        start, direction, scale, length, gen = stack.pop()
        n_steps = max(2, int(length / step))
        p = start.copy()
        d = _unit(direction)
        t0 = float(np.linalg.norm(start - np.array([0.0, 0.0, top]))) / C_M_PER_NS
        for s in range(n_steps):
            if total >= max_records:
                break
            d = _unit(d + rng.normal(0, 0.05 + 0.02 * gen, 3))
            p = p + d * step
            if np.any(p < ranges[:, 0]) or np.any(p > ranges[:, 1]):
                break
            depth_frac = max(0.0, (top - p[2]) / 2.5)
            profile = np.exp(-((depth_frac - 0.35) ** 2) / 0.08)
            w = 300.0 * scale * profile * (1.0 + 0.1 * rng.standard_normal())
            if w <= 1.0:
                continue
            pos_all.append(p.copy())
            dir_all.append(d.copy())
            w_all.append(w)
            t_all.append(t0 + (s + 1) * step / C_M_PER_NS)
            total += 1
            if gen < depth and rng.random() < 0.02:
                child = _unit(d + rng.normal(0, 0.35, 3))
                stack.append((p.copy(), child, scale * 0.45, length * 0.5, gen + 1))
                if len(stack) > branches * (gen + 1) * 4:
                    stack.pop(0)
    if not pos_all:
        raise RuntimeError("synthetic generator produced no records")
    pos = np.asarray(pos_all)
    dirs = np.asarray(dir_all)
    return dict(
        NPhotons=np.asarray(w_all, dtype=np.float64),
        time=np.asarray(t_all, dtype=np.float64),
        posX=pos[:, 0], posY=pos[:, 1], posZ=pos[:, 2],
        dirX=dirs[:, 0], dirY=dirs[:, 1], dirZ=dirs[:, 2],
    )


def write_dataset(root, pattern, n_events, ranges, seed=0, max_records=20000):
    """Write `n_events` synthetic events under `root` using `pattern`."""
    root = Path(root)
    written = []
    for eid in range(int(n_events)):
        cols = synth_event(seed * 1000 + eid, ranges, max_records=max_records)
        dst = root / pattern.format(eid=eid)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **cols)
        written.append(str(dst))
    return written
