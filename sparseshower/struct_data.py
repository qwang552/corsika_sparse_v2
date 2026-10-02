"""NumPy side of the structure model.  No torch, so every piece is unit-testable on CPU.

Model 3 of 4 (data side). The structure model decides, for every active parent
voxel, which of its 8 children on the next finer grid are active. This file
prepares everything that model needs on CPU: the child configurations, the
input features, the training-time corruptions and the colour schedule.

* 256-way joint child configuration (`bits_to_config`, `CODE_BITS`), local
  shape features, parent-set corruption (drop real parents / add empty ones
  with target config 0).
* 26-neighbour known-children features.  For each of the 26 neighbour
  offsets, the children of that neighbour that touch this parent: 4 across a
  face, 2 across an edge, 1 across a corner -> 56 bits, + 26 "known" flags.
  `nbr26_slots` fixes the order.
* K-colour sequential sampling.  K = 8 uses the 2x2x2 parity colour
  (i%2) + 2(j%2) + 4(k%2): no two parents of one colour touch, even
  diagonally, so everything a parent sees of its neighbours is either final
  or unknown.  Training draws a colour step k: colours < k are known (truth),
  the loss covers colours >= k.  K = 2 is a checkerboard, K = 1 a single pass.
* Track-cut augmentation.  Along the principal direction of locally linear
  known parents, 1-3 consecutive known parents get their child bits zeroed
  (flag stays "known"), which mimics an earlier colour that missed a piece of
  track.  Targets are unchanged.

Child code convention: code = dx + 2*dy + 4*dz; a configuration is
sum_c bit_c * 2**c.
"""
from __future__ import annotations

import numpy as np

from .geometry import OFFSETS_6, OFFSETS_26, neighbour_table, pack, unpack

N_CONFIG = 256
N_SHAPE = 9          # principal axis v v^T (6) + linearity + planarity + fill fraction
N_CTX = 3            # decoded field at the 48^3 ancestor: p_occ, ch0, ch1

CODE_BITS = ((np.arange(N_CONFIG)[:, None] >> np.arange(8)[None, :]) & 1).astype(np.float32)
_POW2 = (1 << np.arange(8)).astype(np.int64)


def nbr26_slots():
    """(offset index k into OFFSETS_26, child code of that neighbour) for the 56 slots."""
    slots = []
    for k, o in enumerate(OFFSETS_26):
        choices = []
        for a in range(3):
            if o[a] > 0:
                choices.append((0,))      # neighbour is on the + side: its low children face us
            elif o[a] < 0:
                choices.append((1,))
            else:
                choices.append((0, 1))
        for cx in choices[0]:
            for cy in choices[1]:
                for cz in choices[2]:
                    slots.append((k, cx + 2 * cy + 4 * cz))
    return np.asarray(slots, dtype=np.int64)


SLOTS = nbr26_slots()
N_SLOTS = len(SLOTS)                       # 56
N_NBR = len(OFFSETS_26)                    # 26
N_IN = N_CTX + 1 + N_SHAPE + N_SLOTS + N_NBR   # 95


def bits_to_config(bits):
    """8 child bits (..., 8) -> configuration index 0..255 (code = dx + 2*dy + 4*dz)."""
    return (np.asarray(bits) > 0.5).astype(np.int64) @ _POW2


def config_to_bits(config):
    """Configuration index 0..255 -> 8 child bits."""
    return CODE_BITS[np.asarray(config, dtype=np.int64)]


def expand_children(parent_ijk, bits):
    """Per-parent 8-bit occupancy -> child voxel indices on the next grid."""
    parent_ijk = np.asarray(parent_ijk, dtype=np.int64)
    rows, codes = np.nonzero(np.asarray(bits) > 0.5)
    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    out = 2 * parent_ijk[rows] + np.stack([codes % 2, (codes // 2) % 2, codes // 4], axis=1)
    grid2 = 2 * (int(parent_ijk.max()) + 1)
    return out[np.argsort(pack(out, max(grid2, 2)), kind="stable")]


# ---------------------------------------------------------------- colours ---
def colour_of(ijk, colours):
    """Colour class of each parent (K = 1, 2 or 8) that sets the sequential sampling order."""
    ijk = np.asarray(ijk, dtype=np.int64)
    k = int(colours)
    if k == 1:
        return np.zeros(len(ijk), dtype=np.int64)
    if k == 2:
        return ijk.sum(axis=1) % 2
    if k == 8:
        return (ijk[:, 0] % 2) + 2 * (ijk[:, 1] % 2) + 4 * (ijk[:, 2] % 2)
    raise ValueError("struct.colours must be 1, 2 or 8")


# ------------------------------------------------------------------ shape ---
_SHAPE_OFFSETS = {}


def _cube_offsets(radius):
    """Offsets of a (2r+1)^3 cube without its centre, and their outer products (cached)."""
    if radius not in _SHAPE_OFFSETS:
        r = range(-radius, radius + 1)
        offs = np.array([(dx, dy, dz) for dx in r for dy in r for dz in r
                         if (dx, dy, dz) != (0, 0, 0)], dtype=np.int64)
        oo = offs[:, :, None].astype(np.float64) * offs[:, None, :]
        _SHAPE_OFFSETS[radius] = (offs, oo.reshape(len(offs), 9))
    return _SHAPE_OFFSETS[radius]


def shape_features(ijk, grid, radius=2):
    """(N, 9) float32: principal axis v v^T (6), linearity, planarity and fill fraction of the occupied (2r+1)^3 neighbourhood."""
    ijk = np.asarray(ijk, dtype=np.int64)
    n = len(ijk)
    out = np.zeros((n, N_SHAPE), dtype=np.float32)
    if n == 0:
        return out
    offs, oo = _cube_offsets(int(radius))
    r, g = int(radius), int(grid)
    if (g + 2 * r) ** 3 <= 64_000_000:
        occ = np.zeros((g + 2 * r,) * 3, dtype=bool)
        occ[ijk[:, 0] + r, ijk[:, 1] + r, ijk[:, 2] + r] = True
        base = ijk + r
        mask = np.empty((n, len(offs)), dtype=np.float32)
        for m, o in enumerate(offs):
            mask[:, m] = occ[base[:, 0] + o[0], base[:, 1] + o[1], base[:, 2] + o[2]]
    else:
        _, mask = neighbour_table(ijk, g, offs)
    count = mask.sum(axis=1)
    cov = (mask.astype(np.float64) @ oo).reshape(n, 3, 3) / np.maximum(count, 1.0)[:, None, None]
    evals, evecs = np.linalg.eigh(cov)
    l1, l2, l3 = evals[:, 2], evals[:, 1], evals[:, 0]
    v = evecs[:, :, 2]
    safe = np.maximum(l1, 1e-12)
    feats = np.stack([v[:, 0] ** 2, v[:, 1] ** 2, v[:, 2] ** 2,
                      v[:, 0] * v[:, 1], v[:, 0] * v[:, 2], v[:, 1] * v[:, 2],
                      (l1 - l2) / safe, (l2 - l3) / safe, count / len(offs)], axis=1)
    feats[count == 0] = 0.0
    out[:] = feats
    return out


def principal_direction(shape):
    """Recover a unit principal axis (sign arbitrary) from the 6 vv^T components."""
    s = np.asarray(shape, dtype=np.float64)
    vv = np.zeros((len(s), 3, 3))
    vv[:, 0, 0], vv[:, 1, 1], vv[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    vv[:, 0, 1] = vv[:, 1, 0] = s[:, 3]
    vv[:, 0, 2] = vv[:, 2, 0] = s[:, 4]
    vv[:, 1, 2] = vv[:, 2, 1] = s[:, 5]
    a = np.argmax(s[:, :3], axis=1)
    rows = vv[np.arange(len(s)), a]
    norm = np.sqrt(np.maximum(s[np.arange(len(s)), a], 1e-12))
    return rows / norm[:, None]


# ------------------------------------------------------------- corruption ---
def perturb_parents(parent, bits, grid, rng, drop_frac, add_frac):
    """Drop real parents / add empty face-neighbour parents (target config 0)."""
    parent = np.asarray(parent, dtype=np.int64)
    bits = np.asarray(bits, dtype=np.float32)
    n = len(parent)
    real_keys = pack(parent, grid) if n else np.zeros(0, dtype=np.int64)
    keep = np.ones(n, dtype=bool)
    if drop_frac > 0 and n > 8:
        n_drop = int(rng.uniform(0.0, drop_frac) * n)
        if n_drop:
            keep[rng.choice(n, n_drop, replace=False)] = False
    parent, bits = parent[keep], bits[keep]
    if add_frac > 0 and len(parent):
        n_add = int(rng.uniform(0.0, add_frac) * len(parent))
        if n_add:
            cand = (parent[:, None, :] + OFFSETS_6[None]).reshape(-1, 3)
            cand = cand[np.all((cand >= 0) & (cand < grid), axis=1)]
            keys = np.unique(pack(cand, grid))
            keys = keys[~np.isin(keys, real_keys)]
            if len(keys):
                pick = rng.choice(keys, min(n_add, len(keys)), replace=False)
                parent = np.concatenate([parent, unpack(pick, grid)])
                bits = np.concatenate([bits, np.zeros((len(pick), 8), dtype=np.float32)])
    order = np.argsort(pack(parent, grid), kind="stable")
    return parent[order], bits[order]


def track_cut(parent, grid, known, bits_known, shape, rng, frac, length=(1, 3), lin_min=0.9):
    """Track cut: zero the child bits of short runs of known parents along local tracks.

    Returns a modified copy of bits_known (flags are left as they are).
    """
    parent = np.asarray(parent, dtype=np.int64)
    bits_known = np.array(bits_known, dtype=np.float32, copy=True)
    seeds = np.flatnonzero((known > 0.5) & (shape[:, 6] > lin_min))
    if len(seeds) == 0 or frac <= 0:
        return bits_known
    n_cut = max(1, int(rng.uniform(0.0, frac) * int((known > 0.5).sum())))
    seeds = rng.choice(seeds, min(n_cut, len(seeds)), replace=False)
    keys = pack(parent, grid)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    v = principal_direction(shape[seeds])
    step = np.rint(v / np.maximum(np.abs(v).max(axis=1, keepdims=True), 1e-9)).astype(np.int64)
    for s, d in zip(seeds, step):
        n_len = int(rng.integers(int(length[0]), int(length[1]) + 1))
        sign = 1 if rng.random() < 0.5 else -1
        pts = parent[s][None] + sign * np.arange(n_len)[:, None] * d[None]
        pts = pts[np.all((pts >= 0) & (pts < grid), axis=1)]
        if not len(pts):
            continue
        pk = pack(pts, grid)
        pos = np.clip(np.searchsorted(sk, pk), 0, len(sk) - 1)
        hit = sk[pos] == pk
        rows = order[pos[hit]]
        rows = rows[known[rows] > 0.5]
        bits_known[rows] = 0.0
    return bits_known


# ------------------------------------------------------------------ inputs ---
def level_inputs(parent, grid, c, shape_radius, bundle=None):
    """Geometry + shape + 26-neighbour table of one parent set (NumPy)."""
    from .samples import ancestor_lookup, bundle_for

    parent = np.asarray(parent, dtype=np.int64)
    if bundle is None:
        bundle = bundle_for(parent, grid, c)
    nbr26_idx, nbr26_mask = neighbour_table(parent, int(grid), OFFSETS_26)
    return dict(parent=parent, grid=int(grid), bundle=bundle,
                lookup=ancestor_lookup(parent, grid, int(c["field"]["base_grid"])),
                shape=shape_features(parent, grid, int(shape_radius)),
                nbr26_idx=nbr26_idx, nbr26_mask=nbr26_mask)


def colour_step(parent, colours, k):
    """Training view for colour step k: known = colour < k, loss mask = colour >= k."""
    col = colour_of(parent, colours)
    return (col < int(k)).astype(np.float32), col >= int(k)


def level_prior_logits(child_prior, config0_prior):
    """Initial output bias: log prior of each of the 256 configurations."""
    k = CODE_BITS.sum(axis=1)
    p = float(child_prior)
    logp = k * np.log(p) + (8 - k) * np.log(1 - p)
    logp[0] = np.log(float(config0_prior))
    return (logp - logp.max()).astype(np.float32)
