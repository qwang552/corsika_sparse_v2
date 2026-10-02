"""Describe a voxel cache and check that it matches a config (read only).

Answers two questions about paths.processed of the config:

1. What is stored, and how?  Prints the files, the number of events, the
   train / val / test sizes, the channel statistics and the arrays of one event.
2. Was it built by this code with these settings?  Recomputes the data
   signature (hash of data + macro + raw path) and the seeded split from the
   config and compares them with signature.json / metadata.json of the cache.
   With --rebuild N it also re-runs sparseshower.data.process_event on the raw
   parquet of N cached events and compares the arrays with the stored files.

From the code directory, with the environment active:

    python scripts/check_cache.py --config configs/data.yaml --seed 20260922              # the training cache
    python scripts/check_cache.py --config configs/data.yaml --seed 20260922 --rebuild 3  # + re-process 3 events

Nothing is written.  Exit code 1 if a check fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparseshower.common import config, read_json  # noqa: E402
from sparseshower.data import (data_signature, event_path, event_source, load_event,  # noqa: E402
                               process_event, splits_for)


def dir_size(path):
    """Total size in bytes of the files directly inside `path`."""
    return sum(f.stat().st_size for f in Path(path).glob("*") if f.is_file())


def describe_event(root, eid):
    """Print the arrays stored for one event."""
    ev = load_event(root, eid)
    size = event_path(root, eid).stat().st_size
    print(f"\nevent {eid}  ({event_path(root, eid).name}, {size / 2**20:.2f} MiB)")
    for k, v in ev.items():
        print(f"  {k:<6} {str(v.shape):<12} {str(v.dtype):<8} min {np.min(v):.4g}  max {np.max(v):.4g}")
    print(f"  active voxels {len(ev['q']):,}   photons {float(ev['q'].sum()):.4g}   "
          f"raw records {int(ev['cnt'].sum()):,}")


def rebuild_check(c, root, eids):
    """Re-run process_event on the raw file of each event and compare with the cached arrays."""
    ok = True
    for eid in eids:
        src = event_source(c["paths"]["raw"], eid, c["data"]["source_pattern"])
        if not src.exists():
            print(f"  event {eid}: raw file {src} not found, skipped")
            continue
        arrays, macro, _ = process_event(src, c)
        arrays["macro"] = macro
        cached = load_event(root, eid)
        diff = [k for k in arrays if k not in cached or cached[k].shape != arrays[k].shape
                or not np.array_equal(cached[k], arrays[k])]
        print(f"  event {eid}: " + ("identical" if not diff else f"DIFFERENT in {diff}"))
        ok &= not diff
    return ok


def main(argv=None):
    """Print the cache description and the results of the checks."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--event", type=int, help="event to describe (default: the first train event)")
    ap.add_argument("--rebuild", type=int, default=0, help="re-process this many train events from the raw files")
    ap.add_argument("--seed", type=int, help="split seed to check against (default: the config seed; the cache split used 20260922, data.yaml's seed is the training seed)")
    a = ap.parse_args(argv)
    c = config(a.config)
    if a.seed is not None:
        c["seed"] = a.seed
    root = Path(c["paths"]["processed"])
    print(f"config    : {a.config}")
    print(f"cache     : {root}")
    print(f"raw       : {c['paths']['raw']}  ({c['data']['source_pattern']})")
    if not (root / "metadata.json").exists():
        print("no metadata.json: the cache is not finished (run preprocess, then preprocess --finalize)")
        return 1
    meta = read_json(root / "metadata.json")
    d = c["data"]
    n_files = len(list((root / "events").glob("event_*.npz")))
    print(f"grid      : {meta['grid']}^3, box {meta['ranges']} m, voxel "
          f"{(d['ranges'][0][1] - d['ranges'][0][0]) / meta['grid'] * 100:.2f} cm")
    print(f"cuts      : NPhotons > 0, finite, time < {d['time_max']} ns, inside the box; time_mode {d['time_mode']}")
    print(f"events    : {meta['n_events']} in metadata, {n_files} event files, "
          f"{dir_size(root / 'events') / 2**30:.2f} GiB")
    print("split     : " + ", ".join(f"{k} {len(v)}" for k, v in meta["split"].items()))
    s = meta["stats"]
    print(f"stats     : logQ mean {s['q_log_mean']:.4f} std {s['q_log_std']:.4f}, offset std {s['off_std']:.4f}, "
          f"active voxels mean {s['n_active_mean']:.0f} max {s['n_active_max']} "
          f"(first {s['n_stats_events']} train events)")
    report = root / "report.json"
    if report.exists():
        status = {}
        for r in read_json(report):
            status[r["status"]] = status.get(r["status"], 0) + 1
        print(f"report    : {status}")

    ok = True
    sig = data_signature(c)
    stored = read_json(root / "signature.json")["hash"] if (root / "signature.json").exists() else None
    same_sig = stored == sig and meta.get("signature") == sig
    print(f"\nsignature : config {sig[:16]}...  cache {str(stored)[:16]}...  -> "
          + ("MATCH" if same_sig else "DIFFERENT (data / macro settings or raw path differ)"))
    ok &= same_sig
    ids = sorted(int(e) for part in meta["split"].values() for e in part)
    same_split = splits_for(ids, c) == {k: [int(e) for e in v] for k, v in meta["split"].items()}
    print(f"split     : recomputed with seed {c['seed']} -> " + ("MATCH" if same_split else "DIFFERENT"))
    ok &= same_split

    train = [int(e) for e in meta["split"]["train"]]
    describe_event(root, a.event if a.event is not None else train[0])
    if a.rebuild:
        print(f"\nre-processing {a.rebuild} event(s) from the raw files:")
        ok &= rebuild_check(c, root, train[: a.rebuild])
    print("\nALL CHECKS PASS" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
