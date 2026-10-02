"""Raw CORSIKA Cherenkov records -> sparse voxel cache (the `preprocess` step).

This is the code that built the 192^3 cache used for training
(paths.processed in configs/data.yaml).  For each shower it reads the parquet
file in batches, applies the cuts (finite values, NPhotons > 0, t < time_max,
inside the box), bins every record into a 192^3 voxel and sums, per voxel, the
photons and the photon-weighted position, time and direction.  The result is
stored as one .npz per event; `finalize_cache` then writes metadata.json with
the train / val / test split and the channel statistics.  The commands are in
RUN_ORDER.md, section 3.

Cache layout under paths.processed:

    signature.json            hash of the data settings (a changed setting is refused)
    metadata.json             grid, ranges, split {train, val, test}, stats
    report.json               per-event status (ok / cached / missing / error)
    events/event_XXXXX.npz    one file per shower:
        ijk    (N,3) int32    voxel indices on the 192^3 grid
        q      (N,)  float32  sum of NPhotons in the voxel
        off    (N,3) float32  photon-weighted centroid, in voxel units, in [-0.5, 0.5]
        m      (N,3) float32  photon-weighted mean direction vector (|m| = coherence)
        t      (N,)  float32  photon-weighted mean time (ns, per time_mode)
        cnt    (N,)  int32    number of raw records in the voxel
        macro  (2*z_slabs*r_rings,) float32   coarse (z-slab, r-ring) voxel counts and photon sums

`m` is zero when the source file has no direction columns.  The v2 models use
only ijk, q and off; t and m are stored but not modelled yet, and macro (the
v1 conditioning summary) is kept so rebuilt files match the existing cache.

`load_event` and `Codec` are used at training and sampling time: they read
cached events and normalise q / off for the networks, with the statistics that
`compute_stats` writes to metadata.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import digest, lock, read_json, save_npz, write_json
from .geometry import voxel_center, voxel_index

C_M_PER_NS = 0.299792458

REQUIRED = ["NPhotons", "time", "posX", "posY", "posZ"]
DIRECTION = ["dirX", "dirY", "dirZ"]


# ------------------------------------------------------------------ input ---
def event_source(raw_root, eid, pattern):
    """Path of the raw parquet file of event `eid` (data.source_pattern)."""
    return Path(raw_root) / pattern.format(eid=int(eid))


def iter_batches(path, batch_rows):
    """Yield dicts of 1-D float64 arrays from a parquet or npz source.

    The npz branch exists so the whole pipeline can run without pyarrow (tests,
    synthetic data); the parquet branch is what runs on the cluster.
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as f:
            cols = {k: np.asarray(f[k], dtype=np.float64) for k in f.files}
        missing = [c for c in REQUIRED if c not in cols]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        n = len(cols[REQUIRED[0]])
        for start in range(0, max(n, 1), batch_rows):
            stop = min(start + batch_rows, n)
            if stop > start:
                yield {k: v[start:stop] for k, v in cols.items()}
        return
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    available = set(handle.schema_arrow.names)
    missing = [c for c in REQUIRED if c not in available]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    columns = REQUIRED + [c for c in DIRECTION if c in available]
    for batch in handle.iter_batches(batch_size=batch_rows, columns=columns):
        yield {c: np.asarray(batch.column(i).to_numpy(zero_copy_only=False), dtype=np.float64)
               for i, c in enumerate(columns)}


# ------------------------------------------------------------ aggregation ---
def select_rows(cols, d):
    """Apply the finite / positive-weight / time / box cuts.  Pure NumPy."""
    w = cols["NPhotons"]
    t = cols["time"]
    xyz = np.stack([cols["posX"], cols["posY"], cols["posZ"]], axis=1)
    keep = np.isfinite(w) & np.isfinite(t) & np.all(np.isfinite(xyz), axis=1) & (w > 0)
    tmax = d.get("time_max")
    if tmax is not None:
        keep &= t < float(tmax)
    ranges = np.asarray(d["ranges"], dtype=np.float64)
    keep &= np.all((xyz >= ranges[:, 0]) & (xyz <= ranges[:, 1]), axis=1)
    out = dict(w=w[keep], t=t[keep], xyz=xyz[keep])
    if all(c in cols for c in DIRECTION):
        dirs = np.stack([cols[c] for c in DIRECTION], axis=1)[keep]
        norm = np.linalg.norm(dirs, axis=1, keepdims=True)
        out["dir"] = np.divide(dirs, np.where(norm > 0, norm, 1.0))
    else:
        out["dir"] = None
    return out


def _partial(sel, d):
    """Group one selected batch by voxel; returns additive sufficient stats."""
    g = int(d["grid"])
    ijk = voxel_index(sel["xyz"], d["ranges"], g)
    keys = (ijk[:, 2] * g + ijk[:, 1]) * g + ijk[:, 0]
    uniq, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.reshape(-1)
    n = len(uniq)
    w = sel["w"]
    out = dict(
        keys=uniq,
        q=np.bincount(inverse, weights=w, minlength=n),
        cnt=np.bincount(inverse, minlength=n).astype(np.float64),
        wt=np.bincount(inverse, weights=w * sel["t"], minlength=n),
        wx=np.stack([np.bincount(inverse, weights=w * sel["xyz"][:, a], minlength=n)
                     for a in range(3)], axis=1),
    )
    if sel["dir"] is None:
        out["wd"] = np.zeros((n, 3))
        out["has_dir"] = False
    else:
        out["wd"] = np.stack([np.bincount(inverse, weights=w * sel["dir"][:, a], minlength=n)
                              for a in range(3)], axis=1)
        out["has_dir"] = True
    return out


def _merge(parts):
    """Combine the per-batch voxel sums of one event into one set of sums."""
    keys = np.concatenate([p["keys"] for p in parts])
    uniq, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.reshape(-1)
    n = len(uniq)
    out = dict(keys=uniq, has_dir=all(p["has_dir"] for p in parts))
    for field in ("q", "cnt", "wt"):
        out[field] = np.bincount(inverse, weights=np.concatenate([p[field] for p in parts]),
                                 minlength=n)
    for field in ("wx", "wd"):
        stack = np.concatenate([p[field] for p in parts], axis=0)
        out[field] = np.stack([np.bincount(inverse, weights=stack[:, a], minlength=n)
                               for a in range(3)], axis=1)
    return out


def macro_vector(ijk, q, d, m):
    """Coarse (z-slab, r-ring) summary: counts and photon sums, row-major."""
    g = int(d["grid"])
    z_slabs, r_rings = int(m["z_slabs"]), int(m["r_rings"])
    centers = voxel_center(ijk, d["ranges"], g)
    ranges = np.asarray(d["ranges"], dtype=np.float64)
    # Slab 0 is the high-z end, matching the shower's direction of travel.
    frac = (ranges[2, 1] - centers[:, 2]) / (ranges[2, 1] - ranges[2, 0])
    zs = np.clip((frac * z_slabs).astype(np.int64), 0, z_slabs - 1)
    r = np.hypot(centers[:, 0] - float(m.get("axis_x", 0.0)),
                 centers[:, 1] - float(m.get("axis_y", 0.0)))
    r_max = float(m["r_max"])
    # Ring index ~ sqrt(r / r_max), so rings are narrowest near the axis; everything beyond r_max lands in the last ring.
    rr = np.clip((np.sqrt(np.clip(r / r_max, 0, 1)) * r_rings).astype(np.int64), 0, r_rings - 1)
    cell = zs * r_rings + rr
    size = z_slabs * r_rings
    counts = np.bincount(cell, minlength=size).astype(np.float64)
    sums = np.bincount(cell, weights=q, minlength=size)
    return np.concatenate([counts, sums]).astype(np.float32)


def process_event(path, c):
    """Read one raw event and return (arrays, macro, info)."""
    d = c["data"]
    parts, raw_rows, kept_rows, kept_q = [], 0, 0, 0.0
    for cols in iter_batches(path, int(d["parquet_batch_rows"])):
        raw_rows += len(cols[REQUIRED[0]])
        sel = select_rows(cols, d)
        if not len(sel["w"]):
            continue
        kept_rows += len(sel["w"])
        kept_q += float(sel["w"].sum())
        parts.append(_partial(sel, d))
    if not parts:
        raise ValueError(f"No records survived selection in {path}")
    agg = _merge(parts)
    g = int(d["grid"])
    keys = agg["keys"]
    ijk = np.stack([keys % g, (keys // g) % g, keys // (g * g)], axis=1).astype(np.int64)
    q = agg["q"]
    if np.any(q <= 0):
        raise ValueError("Non-positive voxel weight after aggregation")
    centroid = agg["wx"] / q[:, None]
    centers = voxel_center(ijk, d["ranges"], g)
    ranges = np.asarray(d["ranges"], dtype=np.float64)
    voxel_size = (ranges[:, 1] - ranges[:, 0]) / g
    off = (centroid - centers) / voxel_size
    if np.max(np.abs(off)) > 0.5 + 1e-6:
        raise ValueError("Centroid outside its voxel; check ranges/grid")
    t = agg["wt"] / q
    if d.get("time_mode", "raw") == "lightfront":
        t = t - (ranges[2, 1] - centers[:, 2]) / C_M_PER_NS
    m_vec = agg["wd"] / q[:, None]
    rel = abs(float(q.sum()) - kept_q) / max(kept_q, 1e-30)
    if rel > 1e-5:
        raise RuntimeError(f"Conservation failure: {rel}")
    arrays = dict(
        ijk=ijk.astype(np.int32),
        q=q.astype(np.float32),
        off=off.astype(np.float32),
        m=m_vec.astype(np.float32),
        t=t.astype(np.float32),
        cnt=agg["cnt"].astype(np.int32),
    )
    macro = macro_vector(ijk, q, d, c["macro"])
    info = dict(raw_rows=int(raw_rows), kept_rows=int(kept_rows), n_active=int(len(q)),
                q_total=float(q.sum()), conservation_rel=float(rel), has_dir=bool(agg["has_dir"]),
                records_per_voxel=float(kept_rows / max(len(q), 1)))
    return arrays, macro, info


# ------------------------------------------------------------- preprocess ---
def event_path(root, eid):
    """Path of the cached .npz of event `eid`."""
    return Path(root) / "events" / f"event_{int(eid):05d}.npz"


def data_signature(c):
    """Hash of the settings that define the cache (data, macro, raw path)."""
    d = dict(c["data"])
    d.pop("parquet_batch_rows", None)
    return digest(dict(data=d, macro=c["macro"], raw=str(Path(c["paths"]["raw"]).resolve()),
                       format=2))


def splits_for(ids, c):
    """Seeded random train / val / test split of the cached event ids (fractions from data.*)."""
    ids = np.asarray(sorted(ids))
    rng = np.random.default_rng(int(c["seed"]))
    perm = rng.permutation(len(ids))
    n_train = int(round(float(c["data"]["train_fraction"]) * len(ids)))
    n_val = int(round(float(c["data"]["val_fraction"]) * len(ids)))
    n_train = max(1, min(n_train, len(ids) - 1)) if len(ids) > 1 else len(ids)
    n_val = max(0, min(n_val, len(ids) - n_train))
    take = lambda sl: sorted(int(ids[i]) for i in perm[sl])  # noqa: E731
    return dict(train=take(slice(0, n_train)),
                val=take(slice(n_train, n_train + n_val)),
                test=take(slice(n_train + n_val, len(ids))))


def check_signature(root, sig):
    """Refuse to add events to a cache that was built with different settings."""
    sig_file = Path(root) / "signature.json"
    if sig_file.exists() and read_json(sig_file)["hash"] != sig:
        raise RuntimeError("Cache configuration changed. Use a NEW paths.processed root.")
    write_json(sig_file, dict(hash=sig))


def busy_shards(root):
    """Shard lock files whose owner is still alive (flock is released on exit)."""
    busy = []
    for f in sorted(Path(root).glob(".preprocess.shard*.lock")):
        try:
            with lock(f, wait=False):
                pass
        except RuntimeError:
            busy.append(f.name)
    return busy


def finalize_cache(c):
    """Scan the cache, merge shard reports and write metadata.json.

    Only reads the event files that are already there, so a partially
    preprocessed cache can be finalized.  It
    refuses to run while a shard is still writing, because the split and the
    codec statistics would then be built from half a dataset.
    """
    d, root = c["data"], Path(c["paths"]["processed"])
    sig = data_signature(c)
    still = busy_shards(root)
    if still:
        raise RuntimeError(f"{len(still)} preprocess shard(s) still running ({', '.join(still)}); "
                           "wait for them to finish before finalizing")
    with lock(root / ".preprocess.lock", wait=True):
        check_signature(root, sig)
        ok = sorted(int(f.stem.split("_")[1]) for f in (root / "events").glob("event_*.npz"))
        if not ok:
            raise RuntimeError("No cached events; run preprocess first")
        if len(ok) < int(d["min_events"]):
            raise RuntimeError(f"Only {len(ok)} cached events (< data.min_events)")
        report = []
        for f in sorted((root / "reports").glob("shard_*.json")) if (root / "reports").exists() else []:
            report.extend(read_json(f))
        if report:
            write_json(root / "report.json", report)
        meta = dict(signature=sig, grid=int(d["grid"]), ranges=d["ranges"],
                    macro=c["macro"], split=splits_for(ok, c), n_events=len(ok), partial=False)
        meta["stats"] = compute_stats(root, meta["split"]["train"], c)
        write_json(root / "metadata.json", meta)
    return meta


def preprocess(c, limit=None, shard=None, num_shards=1, finalize=True):
    """Aggregate events into the cache.

    With `shard`, only every `num_shards`-th event is processed and metadata is
    left to `finalize_cache`, so N of these can run in parallel on N workers.
    """
    d, root = c["data"], Path(c["paths"]["processed"])
    root.mkdir(parents=True, exist_ok=True)
    sig = data_signature(c)
    num_shards = max(1, int(num_shards))
    tag = "" if shard is None else f".shard{int(shard)}"
    with lock(root / f".preprocess{tag}.lock", wait=True):
        check_signature(root, sig)
        report = []
        ids = list(range(int(d["event_start"]), int(d["event_stop"])))
        if limit:
            ids = ids[: int(limit)]
        if shard is not None:
            if not 0 <= int(shard) < num_shards:
                raise ValueError("shard must be in [0, num_shards)")
            ids = ids[int(shard)::num_shards]
        for eid in ids:
            dst = event_path(root, eid)
            src = event_source(c["paths"]["raw"], eid, d["source_pattern"])
            if dst.exists():
                report.append(dict(event_id=eid, status="cached"))
                continue
            if not src.exists():
                report.append(dict(event_id=eid, status="missing"))
                continue
            try:
                arrays, macro, info = process_event(src, c)
            except Exception as exc:  # keep going; the report records the failure
                report.append(dict(event_id=eid, status="error", detail=f"{type(exc).__name__}: {exc}"))
                continue
            save_npz(dst, macro=macro, **arrays)
            report.append(dict(event_id=eid, status="ok", **info))
        ok = [r["event_id"] for r in report if r["status"] in ("ok", "cached")]
        if not ok:
            raise RuntimeError("No usable events; check paths.raw and data.source_pattern")
        if shard is None:
            write_json(root / "report.json", report)
        else:
            write_json(root / "reports" / f"shard_{int(shard):03d}.json", report)
        if not finalize or shard is not None:
            return dict(shard=shard, n_events=len(ok), finalized=False)
        if not limit and len(ok) < int(d["min_events"]):
            raise RuntimeError(f"Only {len(ok)} usable events (< data.min_events)")
        meta = dict(signature=sig, grid=int(d["grid"]), ranges=d["ranges"],
                    macro=c["macro"], split=splits_for(ok, c), n_events=len(ok),
                    partial=bool(limit))
        meta["stats"] = compute_stats(root, meta["split"]["train"], c)
        write_json(root / "metadata.json", meta)
    return meta


def load_event(root, eid):
    """Read one cached event as a dict of arrays (ijk, q, off, m, t, cnt, macro)."""
    with np.load(event_path(root, eid)) as f:
        return {k: np.asarray(f[k]) for k in f.files}


def compute_stats(root, train_ids, c, max_events=64):
    """Channel statistics from the first `max_events` training events (written to metadata.json for the Codec)."""
    q_eps = float(c["data"]["q_eps"])
    logs, offs, ts, macros, actives, ms = [], [], [], [], [], []
    for eid in list(train_ids)[:max_events]:
        ev = load_event(root, eid)
        logs.append(np.log(ev["q"].astype(np.float64) + q_eps))
        offs.append(ev["off"].astype(np.float64))
        ts.append(ev["t"].astype(np.float64))
        ms.append(np.linalg.norm(ev["m"].astype(np.float64), axis=1))
        macros.append(ev["macro"].astype(np.float64))
        actives.append(len(ev["q"]))
    if not logs:
        raise RuntimeError("No training events available for statistics")
    log_all = np.concatenate(logs)
    off_all = np.concatenate(offs)
    t_all = np.concatenate(ts)
    macro_all = np.log1p(np.stack(macros))
    std = lambda a: float(max(np.std(a), 1e-6))  # noqa: E731
    return dict(
        q_log_mean=float(np.mean(log_all)), q_log_std=std(log_all),
        off_std=std(off_all), t_mean=float(np.mean(t_all)), t_std=std(t_all),
        coh_mean=float(np.mean(np.concatenate(ms))),
        macro_log_mean=macro_all.mean(axis=0).tolist(),
        macro_log_std=np.maximum(macro_all.std(axis=0), 1e-6).tolist(),
        n_active_mean=float(np.mean(actives)), n_active_max=int(np.max(actives)),
        n_stats_events=len(logs),
    )


# ------------------------------------------------------------------ codec ---
class Codec:
    """Physical values <-> normalized network channels.

    Channel order: [logQ, off_x, off_y, off_z].  Every channel is scaled to
    unit standard deviation so that one sigma means the same disturbance in all
    of them; xyz share a single scale so the geometry is not distorted.
    """

    def __init__(self, stats, q_eps):
        self.q_log_mean = float(stats["q_log_mean"])
        self.q_log_std = float(stats["q_log_std"])
        self.off_std = float(stats["off_std"])
        self.q_eps = float(q_eps)

    @property
    def n_channels(self):
        return 4

    def encode(self, q, off):
        """(q, off) -> 4 unit-variance channels [logQ, off_x, off_y, off_z]."""
        logq = (np.log(np.asarray(q, dtype=np.float64) + self.q_eps) - self.q_log_mean) / self.q_log_std
        return np.concatenate([logq[:, None], np.asarray(off, dtype=np.float64) / self.off_std],
                              axis=1).astype(np.float32)

    def decode(self, x):
        """4 channels -> (q >= 0, off clipped to [-0.5, 0.5]); inverse of encode."""
        x = np.asarray(x, dtype=np.float64)
        q = np.exp(x[:, 0] * self.q_log_std + self.q_log_mean) - self.q_eps
        off = np.clip(x[:, 1:4] * self.off_std, -0.5, 0.5)
        return np.maximum(q, 0.0), off
