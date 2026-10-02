"""The 48^3 coarse field, its statistics, D4 transforms and profiles (NumPy only).

Role in the pipeline: turns a cached 192^3 event into the coarse 48^3 field
that the FieldAE encodes and that struct / attr are conditioned on. Also holds
the field statistics (field_stats_48.json), the D4 (x, y) rotations / mirrors
used for optional data augmentation (encode.d4, dit.d4), and the (z, r) photon
profiles used in evaluation.

Field layout: array (2, B, B, B)
indexed [channel, k(z), j(y), i(x)], flat cell index (k*B + j)*B + i.

    ch0 = log(1 + n) / log(1 + f^3)      n = occupied fine voxels in the cell, f = grid / B
    ch1 = (log(sum q + q_eps) - q_mean) / q_std   on occupied cells, 0 elsewhere

The decoded field the AE hands downstream has three channels
(p_occ, ch0, ch1); `truth_field3` builds the same thing from truth with
p_occ = 1{n > 0}, so struct/attr see one format in every chain.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import read_json, write_json
from .geometry import voxel_center


# ------------------------------------------------------------------ field ---
def coarse_field(ijk, q, grid, base, q_eps, stats=None):
    """(2, B, B, B) float32 field + per-cell counts and photon sums (flat, float64).

    ch1 is standardized only when `stats` is given; otherwise it is the raw log photon sum.
    """
    ijk = np.asarray(ijk, dtype=np.int64)
    grid, base = int(grid), int(base)
    f = grid // base
    co = ijk // f
    flat = (co[:, 2] * base + co[:, 1]) * base + co[:, 0]
    size = base ** 3
    counts = np.bincount(flat, minlength=size).astype(np.float64)
    sums = np.bincount(flat, weights=np.asarray(q, dtype=np.float64), minlength=size)
    occ = np.log1p(counts) / np.log1p(float(f ** 3))
    with np.errstate(divide="ignore"):
        logq = np.where(counts > 0, np.log(sums + float(q_eps)), 0.0)
    if stats is not None:
        logq = np.where(counts > 0, (logq - stats["q_mean"]) / stats["q_std"], 0.0)
    field = np.stack([occ, logq]).reshape(2, base, base, base).astype(np.float32)
    return field, counts, sums


def truth_field3(field2):
    """(2,B,B,B) truth field -> (3,B,B,B) (p_occ, ch0, ch1) with p_occ = 1{n > 0}."""
    occ = (np.asarray(field2)[0] > 0).astype(np.float32)
    return np.concatenate([occ[None], np.asarray(field2, dtype=np.float32)], axis=0)


def field_statistics(events, c, max_events=None):
    """Mean / std of log photon sum over occupied base cells (streams `events`)."""
    grid, base = int(c["data"]["grid"]), int(c["field"]["base_grid"])
    vals = []
    for i, ev in enumerate(events):
        if max_events is not None and i >= int(max_events):
            break
        field, counts, _ = coarse_field(ev["ijk"], ev["q"], grid, base, float(c["data"]["q_eps"]))
        vals.append(field[1].reshape(-1)[counts > 0])
    allv = np.concatenate(vals) if vals else np.zeros(1)
    return dict(q_mean=float(allv.mean()), q_std=float(max(allv.std(), 1e-6)), base_grid=base)


def field_stats_path(c):
    """Path of field_stats_<B>.json (shared by all cases)."""
    return Path(c["paths"]["output"]) / f"field_stats_{int(c['field']['base_grid'])}.json"


def stats_events(c, meta):
    """The events field_stats_{B}.json is computed from: the first field.stats_events
    train ids (default 256), identical for every caller so the file never depends on
    which job ran first."""
    from .data import load_event

    root = c["paths"]["processed"]
    ids = [int(e) for e in meta["split"]["train"]][: int(c["field"].get("stats_events", 256))]
    return (load_event(root, e) for e in ids)


def load_field_stats(c, create_from=None):
    """Shared by every case (it only depends on the data): computed once, then read."""
    path = field_stats_path(c)
    if path.exists():
        return read_json(path)
    if create_from is None:
        raise FileNotFoundError(f"{path} missing; train the AE first (it writes this file)")
    stats = field_statistics(create_from, c)
    write_json(path, stats)
    return stats


def photon_sums(ch1, occupied, stats, q_eps):
    """Invert ch1 -> photon sum per cell (0 where not occupied)."""
    s = np.exp(np.asarray(ch1, dtype=np.float64) * stats["q_std"] + stats["q_mean"]) - float(q_eps)
    return np.where(occupied, np.maximum(s, 0.0), 0.0)


def occupied_cells(p_occ, threshold):
    """(N, 3) indices (i, j, k) of the cells with p_occ > threshold, key-sorted like every parent set."""
    base = p_occ.shape[-1]
    flat = np.flatnonzero(np.asarray(p_occ).reshape(-1) > float(threshold))
    return np.stack([flat % base, (flat // base) % base, flat // (base * base)], 1).astype(np.int64)


# --------------------------------------------------------------- profiles ---
def cell_profile_index(c, base, z_slabs=32, r_rings=16):
    """For each base cell: its z slab (from the high-z end) and radial ring,
    with the same definitions as evaluate.profiles uses on voxels."""
    ranges = np.asarray(c["data"]["ranges"], dtype=np.float64)
    b = np.arange(int(base))
    k, j, i = np.meshgrid(b, b, b, indexing="ij")
    ijk = np.stack([i.reshape(-1), j.reshape(-1), k.reshape(-1)], axis=1)
    centres = voxel_center(ijk, ranges, int(base))
    frac = (ranges[2, 1] - centres[:, 2]) / (ranges[2, 1] - ranges[2, 0])
    zs = np.clip((frac * z_slabs).astype(np.int64), 0, z_slabs - 1)
    r = np.hypot(centres[:, 0] - float(c["macro"].get("axis_x", 0.0)),
                 centres[:, 1] - float(c["macro"].get("axis_y", 0.0)))
    rr = np.clip((np.sqrt(np.clip(r / float(c["macro"]["r_max"]), 0, 1)) * r_rings).astype(np.int64),
                 0, r_rings - 1)
    return zs, rr


def field_profiles(sums_flat, zs, rr, z_slabs=32, r_rings=16):
    """Normalised longitudinal (z slab) and radial (ring) photon profiles of a field."""
    s = np.asarray(sums_flat, dtype=np.float64)
    total = max(s.sum(), 1e-30)
    return (np.bincount(zs, weights=s, minlength=z_slabs) / total,
            np.bincount(rr, weights=s, minlength=r_rings) / total)


# --------------------------------------------------------------------- D4 ---
# The 8 symmetries of the square acting on (x, y): t = r + 4*m, rotate by r*90
# degrees about the box axis, then mirror x if m.  Integer arithmetic on
# U = 2i - (G - 1), so the box centre is exact.
def _d4_uv(u, v, t, inverse=False):
    """Apply (or undo) D4 element t to centred integer coordinates (u, v)."""
    r, m = int(t) % 4, int(t) // 4
    if not inverse:
        for _ in range(r):
            u, v = -v, u
        if m:
            u = -u
    else:
        if m:
            u = -u
        for _ in range(r):
            u, v = v, -u
    return u, v


def d4_ijk(ijk, grid, t):
    """Apply D4 element t (0..7) to voxel indices in the x-y plane."""
    ijk = np.asarray(ijk, dtype=np.int64).copy()
    g1 = int(grid) - 1
    u, v = _d4_uv(2 * ijk[:, 0] - g1, 2 * ijk[:, 1] - g1, t)
    ijk[:, 0], ijk[:, 1] = (u + g1) // 2, (v + g1) // 2
    return ijk


def d4_offsets(off, t):
    """Voxel-centroid offsets (in voxel units) under the same transform."""
    off = np.asarray(off, dtype=np.float64).copy()
    u, v = _d4_uv(off[:, 0].copy(), off[:, 1].copy(), t)      # copies: views would alias in the swap
    off[:, 0], off[:, 1] = u, v
    return off


def d4_field(field, t):
    """Apply t to a (..., B[z], B[y], B[x]) array."""
    field = np.asarray(field)
    b = field.shape[-1]
    g1 = b - 1
    jj, ii = np.meshgrid(np.arange(b), np.arange(b), indexing="ij")    # output (j, i)
    u, v = _d4_uv(2 * ii - g1, 2 * jj - g1, t, inverse=True)          # source
    return field[..., (v + g1) // 2, (u + g1) // 2]


def d4_enabled(c, setting):
    """Resolve encode.d4 / dit.d4 (true / false): use the 8 D4 (x, y) variants as augmentation."""
    if isinstance(setting, bool):
        return setting
    raise ValueError(f"d4 must be true or false, got {setting!r}")
