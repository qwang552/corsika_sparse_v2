"""Voxel-set plumbing shared by struct and attr (NumPy only).

Role in the pipeline: the list of grid levels (48, 96, 192), the voxel set at each level, the
8 child-occupancy bits of each parent, and the geometry bundle (neighbour
table, attention windows, pooling map) the sparse network needs.

The coarse grid is `field.base_grid` (48).  `bits_for_parents` accepts parent
sets that do not contain every child's parent (decoded parent sets at level
48 -> 96 miss some truth cells).
"""
from __future__ import annotations

import numpy as np

from .geometry import (neighbour_offsets, neighbour_table, pack, parent_map, serialize_order,
                       window_partition)


def level_grids(grid, base):
    """[base, 2*base, ..., grid]; requires grid = base * 2**k."""
    grid, base = int(grid), int(base)
    if base < 1 or grid % base:
        raise ValueError("field.base_grid must divide data.grid")
    ratio = grid // base
    if ratio & (ratio - 1):
        raise ValueError("data.grid / field.base_grid must be a power of two")
    out, g = [base], base
    while g < grid:
        g *= 2
        out.append(g)
    return out


def normalized_pos(ijk, grid):
    """Voxel centres mapped to [-1, 1] on every axis (grid-relative)."""
    return ((np.asarray(ijk, dtype=np.float32) + 0.5) / float(grid) * 2.0 - 1.0).astype(np.float32)


def geometry_bundle(ijk, grid, window, neighbours, coarse_factor):
    """Neighbour table, two window partitions and the pooling map for one voxel set."""
    ijk = np.asarray(ijk, dtype=np.int64)
    offsets = neighbour_offsets(int(neighbours))
    nbr_idx, nbr_mask = neighbour_table(ijk, grid, offsets)
    win_a = window_partition(len(ijk), window, serialize_order(ijk, variant=0))
    win_b = window_partition(len(ijk), window, serialize_order(ijk, variant=1))
    pool_ijk, pool_parent = parent_map(ijk, max(1, int(coarse_factor)))
    return dict(
        ijk=ijk.astype(np.int32),
        pos=normalized_pos(ijk, grid),
        nbr_idx=nbr_idx.astype(np.int64),
        nbr_mask=nbr_mask.astype(np.float32),
        nbr_off=(offsets.astype(np.float32) / max(1.0, float(np.abs(offsets).max()))),
        win_idx_a=win_a[0].astype(np.int64), win_valid_a=win_a[1],
        win_idx_b=win_b[0].astype(np.int64), win_valid_b=win_b[1],
        pool_parent=pool_parent.astype(np.int64),
        pool_pos=normalized_pos(pool_ijk, max(1, grid // max(1, int(coarse_factor)))),
        n_pool=int(len(pool_ijk)),
        grid=int(grid),
    )


def bundle_for(ijk, grid, c):
    """geometry_bundle with the window / neighbour / pooling settings from the config."""
    return geometry_bundle(ijk, int(grid), int(c["model"]["window"]), int(c["model"]["neighbours"]),
                           max(1, int(grid) // int(c["model"]["pool_grid"])))


def level_sets(ijk, grids):
    """Unique voxel sets at each grid in `grids` (grids[-1] must be the finest), key-sorted."""
    ijk = np.asarray(ijk, dtype=np.int64)
    fine = grids[-1]
    sets = []
    for g in grids:
        coarse = np.unique(ijk // (fine // g), axis=0)
        sets.append(coarse[np.argsort(pack(coarse, g), kind="stable")])
    return sets


def bits_for_parents(parent_ijk, child_ijk, child_grid):
    """8 occupancy bits per parent (code = dx + 2*dy + 4*dz).

    Returns (bits, n_missing, missing): children whose parent is not in
    `parent_ijk` are ignored, counted in n_missing and flagged in the boolean
    mask `missing`, so a caller can report what a decoded parent set lost.
    """
    parent_ijk = np.asarray(parent_ijk, dtype=np.int64).reshape(-1, 3)
    child_ijk = np.asarray(child_ijk, dtype=np.int64).reshape(-1, 3)
    bits = np.zeros((len(parent_ijk), 8), dtype=np.float32)
    if len(child_ijk) == 0 or len(parent_ijk) == 0:
        return bits, int(len(child_ijk)), np.ones(len(child_ijk), dtype=bool)
    pgrid = int(child_grid) // 2
    p_keys = pack(parent_ijk, pgrid)
    own = child_ijk // 2
    c_keys = pack(own, pgrid)
    order = np.argsort(p_keys, kind="stable")
    sk = p_keys[order]
    pos = np.clip(np.searchsorted(sk, c_keys), 0, len(sk) - 1)
    hit = sk[pos] == c_keys
    rows = order[pos[hit]]
    ch = child_ijk[hit]
    code = (ch[:, 0] - 2 * own[hit, 0]) + 2 * (ch[:, 1] - 2 * own[hit, 1]) \
        + 4 * (ch[:, 2] - 2 * own[hit, 2])
    bits[rows, code] = 1.0
    return bits, int((~hit).sum()), ~hit


def ancestor_lookup(ijk, grid, base):
    """Flat index of each voxel's cell on the base grid."""
    f = max(1, int(grid) // int(base))
    co = np.asarray(ijk, dtype=np.int64) // f
    return ((co[:, 2] * base + co[:, 1]) * base + co[:, 0]).astype(np.int64)
