"""Pure-NumPy geometry bookkeeping for sparse voxel fields.

Everything the networks need about *where* voxels are is computed here, on CPU,
once per sample: voxel index <-> position, integer keys, Z-order (Morton)
serialisation for windowed attention, neighbour tables, parent / child maps
between grid levels, and box counting / local linearity for the data audit
and the evaluation.

Keeping this out of torch has two reasons: the structure is fixed while the
attributes are denoised, so neighbourhoods never depend on the noise level; and
the torch modules only do dense algebra on gathered indices, which is easier to
test.
"""
from __future__ import annotations

import numpy as np

# The 6 face neighbours.
OFFSETS_6 = np.array(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.int64
)


def offsets_26():
    """The 26 neighbour offsets of a voxel, faces first, then edges, then corners."""
    out = [(dx, dy, dz)
           for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
           if (dx, dy, dz) != (0, 0, 0)]
    out.sort(key=lambda o: (abs(o[0]) + abs(o[1]) + abs(o[2]), o))
    return np.array(out, dtype=np.int64)


OFFSETS_26 = offsets_26()


def neighbour_offsets(kind):
    """Offset table for a 6- or 26-neighbourhood."""
    if kind == 6:
        return OFFSETS_6
    if kind == 26:
        return OFFSETS_26
    raise ValueError("neighbours must be 6 or 26")


# --------------------------------------------------------------- indexing ---
def voxel_index(xyz, ranges, grid):
    """Map physical coordinates to integer voxel indices, clipped to the box.

    ranges: (3, 2) array of [lo, hi] per axis.  Points exactly on `hi` land in
    the last voxel.
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    span = ranges[:, 1] - ranges[:, 0]
    if np.any(span <= 0):
        raise ValueError("ranges must be increasing")
    frac = (np.asarray(xyz, dtype=np.float64) - ranges[:, 0]) / span
    idx = np.floor(frac * grid).astype(np.int64)
    return np.clip(idx, 0, grid - 1)


def voxel_center(ijk, ranges, grid):
    """Physical centre (metres) of voxels `ijk` on a grid of side `grid`."""
    ranges = np.asarray(ranges, dtype=np.float64)
    span = ranges[:, 1] - ranges[:, 0]
    return ranges[:, 0] + (np.asarray(ijk, dtype=np.float64) + 0.5) / grid * span


def pack(ijk, grid):
    """(N,3) voxel indices -> one int64 key per voxel (x fastest)."""
    ijk = np.asarray(ijk, dtype=np.int64)
    if ijk.ndim != 2 or ijk.shape[1] != 3:
        raise ValueError("ijk must be (N, 3)")
    return (ijk[:, 2] * grid + ijk[:, 1]) * grid + ijk[:, 0]


def unpack(keys, grid):
    """Inverse of `pack`: int64 keys -> (N,3) voxel indices."""
    keys = np.asarray(keys, dtype=np.int64)
    i = keys % grid
    j = (keys // grid) % grid
    k = keys // (grid * grid)
    return np.stack([i, j, k], axis=1)


def morton_key(ijk, bits=None, variant=0):
    """Z-order key.  `variant` permutes the axes so successive blocks use
    different orderings (cheap way to vary the window partition)."""
    ijk = np.asarray(ijk, dtype=np.int64)
    axes = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)][variant % 4]
    a = ijk[:, list(axes)]
    if bits is None:
        bits = int(max(1, np.ceil(np.log2(max(2, int(a.max()) + 1)))))
    key = np.zeros(len(a), dtype=np.int64)
    for b in range(bits):
        for axis in range(3):
            key |= ((a[:, axis] >> b) & 1) << (3 * b + axis)
    return key


def serialize_order(ijk, variant=0):
    """Permutation that sorts voxels along a Z-order curve."""
    return np.argsort(morton_key(ijk, variant=variant), kind="stable")


def neighbour_table(ijk, grid, offsets):
    """For each voxel, the row index of each offset neighbour.

    Returns (idx, mask): idx[n, k] is the index of the neighbour, or n itself
    when that neighbour is not occupied; mask[n, k] is 1.0 where the neighbour
    exists.  Self-index keeps gathers valid without branching.
    """
    ijk = np.asarray(ijk, dtype=np.int64)
    n = len(ijk)
    keys = pack(ijk, grid)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    idx = np.empty((n, len(offsets)), dtype=np.int64)
    mask = np.zeros((n, len(offsets)), dtype=np.float32)
    self_idx = np.arange(n, dtype=np.int64)
    for k, off in enumerate(offsets):
        nb = ijk + off
        inside = np.all((nb >= 0) & (nb < grid), axis=1)
        nb_keys = pack(np.clip(nb, 0, grid - 1), grid)
        pos = np.searchsorted(sorted_keys, nb_keys)
        pos_clipped = np.clip(pos, 0, n - 1)
        hit = inside & (sorted_keys[pos_clipped] == nb_keys)
        found = order[pos_clipped]
        idx[:, k] = np.where(hit, found, self_idx)
        mask[:, k] = hit.astype(np.float32)
    return idx, mask


def _loose_keys(ijk):
    """Collision-free packing for arbitrary non-negative index arrays."""
    ijk = np.asarray(ijk, dtype=np.int64)
    base = int(ijk.max()) + 1 if len(ijk) else 1
    if base > 2**20:
        raise ValueError("index range too large for packing")
    return (ijk[:, 2] * base + ijk[:, 1]) * base + ijk[:, 0]


def parent_map(ijk, factor):
    """Group voxels into parents of side `factor`.

    Returns (parent_ijk, child_to_parent): parent_ijk are the unique parent
    indices (sorted by packed key) and child_to_parent[n] indexes into them.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    ijk = np.asarray(ijk, dtype=np.int64)
    coarse = ijk // factor
    keys = _loose_keys(coarse)
    uniq, inverse = np.unique(keys, return_inverse=True)
    order = np.argsort(keys, kind="stable")
    first = order[np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1]]
    parent_ijk = coarse[first]
    # `np.unique` sorts by key, and `first` is ordered the same way.
    return parent_ijk.astype(np.int64), inverse.astype(np.int64).reshape(-1)


def window_partition(n, window, order=None):
    """Split `n` serialized rows into equal windows.

    Returns (index, valid): index[w, s] is a row index (padded with 0) and
    valid[w, s] is True where the slot is real.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    order = np.arange(n, dtype=np.int64) if order is None else np.asarray(order, dtype=np.int64)
    pad = (-n) % window
    padded = np.concatenate([order, np.zeros(pad, dtype=np.int64)])
    valid = np.concatenate([np.ones(n, dtype=bool), np.zeros(pad, dtype=bool)])
    return padded.reshape(-1, window), valid.reshape(-1, window)


# ------------------------------------------------------------- statistics ---
def box_count(ijk, factors):
    """Number of distinct voxels after coarsening by each factor."""
    ijk = np.asarray(ijk, dtype=np.int64)
    out = []
    for f in factors:
        out.append(int(len(np.unique(_loose_keys(ijk // int(f))))))
    return out


def box_dimension(counts, factors):
    """Local slope D between consecutive scales: N ~ h^-D."""
    counts = np.asarray(counts, dtype=np.float64)
    factors = np.asarray(factors, dtype=np.float64)
    d = []
    for a in range(len(counts) - 1):
        d.append(float(np.log(counts[a] / counts[a + 1]) / np.log(factors[a + 1] / factors[a])))
    return d


def knn_within_grid(points, ijk, grid, k, radius=2):
    """k nearest neighbours among voxel-resident points.

    Candidates come from the (2*radius+1)^3 voxel block around each point, so
    this is exact whenever the k-th neighbour lies within `radius` voxels.
    Returns (dist, idx) with shape (N, k); missing neighbours get inf/self.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n == 0:
        return np.zeros((0, k)), np.zeros((0, k), dtype=np.int64)
    offs = np.array([(dx, dy, dz)
                     for dx in range(-radius, radius + 1)
                     for dy in range(-radius, radius + 1)
                     for dz in range(-radius, radius + 1)], dtype=np.int64)
    idx, mask = neighbour_table(ijk, grid, offs)
    # Include self as a candidate (offset (0,0,0) is in `offs`).
    d = np.linalg.norm(points[idx] - points[:, None, :], axis=2)
    d = np.where(mask > 0, d, np.inf)
    take = min(k, d.shape[1])
    part = np.argsort(d, axis=1)[:, :take]
    rows = np.arange(n)[:, None]
    return d[rows, part], idx[rows, part]


def local_linearity(points, ijk, grid, k=8, radius=2):
    """Largest PCA eigenvalue share of each point's k-neighbourhood."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < k + 1:
        return np.zeros(len(points))
    dist, idx = knn_within_grid(points, ijk, grid, k + 1, radius)
    nb = points[idx]
    ok = np.isfinite(dist)
    nb = np.where(ok[..., None], nb, points[:, None, :])
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / nb.shape[1]
    ev = np.linalg.eigvalsh(cov)[:, ::-1]
    total = ev.sum(axis=1)
    return np.where(total > 0, ev[:, 0] / np.maximum(total, 1e-30), 0.0)
