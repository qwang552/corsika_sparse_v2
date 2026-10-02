"""Run bookkeeping shared by all four models.  Pure NumPy, testable without torch.

    case_dir, fingerprint       where a run writes, and a hash of its settings;
                                training refuses to resume after a change that
                                alters the maths (use a new --case instead)
    checkpoint_decision, amp_decision
                                the "auto" rules for activation checkpointing
                                and mixed precision, from the GPU actually given
    meta_for, split_ids, select_events
                                read metadata.json of the (read-only) cache
    Stream                      deterministic sequence of training items, built
                                a few steps ahead on a worker thread
    LRU                         small thread-safe cache of built items
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

from .common import digest, read_json

# Keys that do not change the trained weights (memory / speed settings, probes, sampling temperature).
NON_SCIENTIFIC = {
    "model": ("checkpoint", "checkpoint_below_gib"),
    "train": ("cache_samples", "prefetch", "probe_every", "amp"),
    "ae": ("probe_events",),
    "dit": ("probe_events",),
    "struct": ("probe_events", "temperature"),
    "attr": ("probe_events", "probe_sigmas"),
}
KIND_SECTIONS = {
    "ae": ("data", "field", "train", "ae"),
    "dit": ("data", "field", "edm", "train", "ae", "encode", "dit"),
    "struct": ("data", "field", "model", "train", "ae", "struct"),
    "attr": ("data", "field", "model", "edm", "train", "ae", "attr"),
}


def fingerprint(c, kind, ids, extra=None):
    """Hash of everything scientific this kind depends on (+ the events)."""
    keep = {}
    for section in KIND_SECTIONS[kind]:
        if section not in c:
            continue
        drop = NON_SCIENTIFIC.get(section, ())
        keep[section] = {k: v for k, v in c[section].items() if k not in drop}
    return digest(dict(kind=kind, scientific=keep, seed=c.get("seed"), events=list(ids),
                       extra=extra))


def checkpoint_decision(value, device_gib, below_gib=16.0):
    """`checkpoint`: True / False / "auto" -> bool, given the card's size."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() == "auto":
        return device_gib is not None and float(device_gib) < float(below_gib)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f'checkpoint must be true, false or "auto", got {value!r}')


def amp_decision(mode, capability, bf16_supported):
    """`train.amp`: "bf16" / "fp16" / "none" / "auto" -> the mode to actually use.

    bf16 needs compute capability 8.0; fp16 only pays off with tensor cores
    (7.0+).  On Pascal (6.1, GTX 1080) half precision runs at 1/64 of fp32, so
    "auto" picks none.  torch.cuda.is_bf16_supported() is True on old cards too
    (emulated), so capability is checked as well.
    """
    mode = str(mode).lower()
    if mode not in ("bf16", "fp16", "none", "auto"):
        raise ValueError(f"train.amp must be bf16/fp16/none/auto, got {mode!r}")
    if capability is None:
        return "none"
    native_bf16 = bool(bf16_supported) and int(capability[0]) >= 8
    if mode == "auto":
        if native_bf16:
            return "bf16"
        return "fp16" if capability[0] >= 7 else "none"
    if mode == "bf16" and not native_bf16:
        raise RuntimeError('train.amp=bf16 but this GPU does not support it; use "auto", fp16 or none.')
    return mode


def case_dir(c, case):
    """Output directory of one experiment: <paths.output>/<case>."""
    return Path(c["paths"]["output"]) / case


def meta_for(c):
    """Read metadata.json of the voxel cache."""
    return read_json(Path(c["paths"]["processed"]) / "metadata.json")


def split_ids(meta, split):
    """Event ids of one split ('train', 'val' or 'test'); raises if empty."""
    ids = list(meta["split"].get(split) or [])
    if not ids:
        raise RuntimeError(f"empty {split} split in metadata.json")
    return [int(e) for e in ids]


def select_events(meta, n_events, seed):
    """All train ids, or a seeded random subset of `n_events` of them."""
    train_ids = split_ids(meta, "train")
    if n_events in (None, 0) or int(n_events) >= len(train_ids):
        return train_ids
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(train_ids), size=int(n_events), replace=False)
    return sorted(int(train_ids[i]) for i in pick)


class LRU:
    """Thread-safe cache of the last `limit` built items (first in, first out: a hit does not refresh an item)."""

    def __init__(self, fn, limit=64):
        self.fn, self.limit = fn, int(limit)
        self.store, self.order = {}, []
        self._lock = threading.Lock()

    def get(self, key):
        """Return the cached item for `key`, building it with `fn` on a miss."""
        with self._lock:
            if key in self.store:
                return self.store[key]
        value = self.fn(key)
        with self._lock:
            if self.limit > 0 and key not in self.store:
                self.store[key] = value
                self.order.append(key)
                while len(self.order) > self.limit:
                    self.store.pop(self.order.pop(0), None)
        return value


class Stream:
    """Deterministic item sequence, built `depth` steps ahead on one thread.

    `build(eids, rng)` returns one training item for a list of event ids.  The
    event order and the rng live on the producer thread only, so the sequence
    is reproducible.  depth = 0 builds inline (tracebacks point at the call).
    """

    def __init__(self, build, ids, seed, batch=1, depth=4):
        self.build, self.ids, self.batch = build, list(ids), max(1, int(batch))
        self.rng = np.random.default_rng(seed)
        self.depth = max(0, int(depth))
        self._error = None
        self._stop = threading.Event()
        if self.depth:
            self.queue = queue.Queue(maxsize=self.depth)
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _pick(self):
        return [int(self.ids[self.rng.integers(len(self.ids))]) for _ in range(self.batch)]

    def _make(self):
        return self.build(self._pick(), self.rng)

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self._make()
            except Exception as exc:  # surfaced on the consumer side
                self._error = exc
                self.queue.put(None)
                return
            while not self._stop.is_set():
                try:
                    self.queue.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def next(self):
        """Next item in the sequence (raises if the worker thread failed)."""
        if not self.depth:
            return self._make()
        item = self.queue.get()
        if item is None:
            raise RuntimeError(f"training-item worker failed: {self._error}") from self._error
        return item

    def close(self):
        """Stop the worker thread and drop queued items."""
        self._stop.set()
        if self.depth:
            try:
                while True:
                    self.queue.get_nowait()
            except queue.Empty:
                pass
            self.thread.join(timeout=5.0)
