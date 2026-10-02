"""Small shared helpers used by every other module.

What is here:
    config / merge      load a YAML config and resolve its `extends:` parent
    digest              stable hash of a dict (run fingerprints, cache signature)
    write_json, save_npz, torch_save
                        atomic writes (temp file + rename), retried on transient
                        cluster filesystem errors, so a killed job never leaves
                        a half-written file behind
    JsonlLog            training log (one JSON record per line)
    lock                file lock so two jobs never write the same cache / case
    seed_all, device_for, resolve_amp, amp_context
                        seeding, device choice and mixed-precision settings
    EMA                 exponential moving average of the model weights
"""
from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml


# ----------------------------------------------------------------- config ---
def merge(a, b):
    """Recursively merge config dict b into a (b wins); used for `extends:`."""
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def config(path):
    """Load a YAML config, resolving a single `extends: <file>` chain."""
    path = Path(path).resolve()
    with path.open() as f:
        c = yaml.safe_load(f)
    if not isinstance(c, dict):
        raise ValueError(f"{path} must contain a mapping")
    parent = c.pop("extends", None)
    if parent:
        c = merge(config(path.parent / parent), c)
    return c


def digest(obj):
    """SHA-256 of a JSON-serialisable object with sorted keys (order independent)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


# --------------------------------------------------------------------- io ---
def read_json(path):
    """Read a JSON file."""
    return json.loads(Path(path).read_text())


# Errors a cluster filesystem (Ceph, NFS) can return for an ordinary write.
# They are usually transient, so writes are retried instead of failing the job.
TRANSIENT_IO = frozenset((errno.EAGAIN, errno.EWOULDBLOCK, errno.EINTR, errno.EBUSY))


def retry_io(fn, *args, tries=6, delay=0.25, **kwargs):
    """Run a filesystem write, backing off over the transient errors.

    Waits at most 0.25+0.5+1+2+4 = 7.75 s in total, then re-raises.  Anything
    that is not in TRANSIENT_IO (no space, no permission, bad path) is raised
    immediately: those do not get better by waiting.
    """
    for attempt in range(int(tries)):
        try:
            return fn(*args, **kwargs)
        except OSError as exc:
            if exc.errno not in TRANSIENT_IO or attempt == int(tries) - 1:
                raise
            time.sleep(float(delay) * (2 ** attempt))
    raise AssertionError("unreachable")


def _write_json(path, obj):
    """Write JSON to a temp file in the same directory, then rename it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, obj):
    """Atomic JSON write with retries."""
    return retry_io(_write_json, path, obj)


def _save_npz(path, **items):
    """Write an .npz to a temp file in the same directory, then rename it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        tmp = f.name
    try:
        np.savez_compressed(tmp, **items)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_npz(path, **items):
    """Atomic .npz write with retries."""
    return retry_io(_save_npz, path, **items)


class JsonlLog:
    """Append-only JSONL log that never stops a run because of a failed write.

    No file handle is kept open.  Ordinary lines are batched and appended with
    retries; if the filesystem still refuses, log lines are dropped instead of
    raising.  Important records (probes, the final step) use `now=True` so that
    `tail -f` stays up to date.
    """

    def __init__(self, path, batch=25, keep=2000, tries=6, delay=0.25):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pending, self.batch, self.keep = [], int(batch), int(keep)
        self.tries, self.delay = int(tries), float(delay)
        self.dropped, self.retries = 0, 0

    def _append(self, payload):
        with self.path.open("a") as f:
            f.write(payload)
            f.flush()

    def write(self, record, now=False):
        """Queue one record; flush when the batch is full or `now` is set."""
        self.pending.append(json.dumps(record, allow_nan=False))
        if now or len(self.pending) >= self.batch:
            return self.flush()
        return True

    def flush(self):
        """Append the queued records to the file; on repeated failure keep them for the next try.
        """
        if not self.pending:
            return True
        payload = "\n".join(self.pending) + "\n"
        try:
            retry_io(self._append, payload, tries=self.tries, delay=self.delay)
        except OSError:
            self.retries += 1
            if len(self.pending) > self.keep:      # bounded memory, oldest go
                self.dropped += len(self.pending) - self.keep
                self.pending = self.pending[-self.keep:]
            return False
        self.pending.clear()
        return True

    def close(self):
        """Final flush; returns False if some records could not be written."""
        ok = self.flush()
        self.dropped += 0 if ok else len(self.pending)
        return ok


@contextlib.contextmanager
def lock(path, wait=False):
    """Advisory exclusive file lock; prevents two writers on one cache/case."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {path}; do not run duplicate writers.") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ------------------------------------------------------------------ torch ---
def seed_all(seed):
    """Seed Python, NumPy and torch (CPU and CUDA) from one integer."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(allow_cpu=False):
    """Return the CUDA device, or CPU only when `allow_cpu` is set (otherwise raise)."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA unavailable. Use a GPU worker, or pass --allow-cpu for smoke tests.")


def resolve_amp(mode, device):
    """Turn train.amp (possibly "auto") into a concrete mode for this GPU."""
    import torch

    from .loader import amp_decision

    cap = bf16 = None
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        bf16 = bool(torch.cuda.is_bf16_supported())
    out = amp_decision(mode, cap, bf16)
    if str(mode).lower() == "auto":
        name = torch.cuda.get_device_name(device) if cap else str(device)
        print(f"train.amp: auto -> {out}  ({name}"
              + (f", compute capability {cap[0]}.{cap[1]}" if cap else "") + ")", flush=True)
    return out


def amp_context(device, mode):
    """Autocast context for the chosen AMP mode (a no-op for 'none' or CPU)."""
    import torch

    if mode == "none" or device.type == "cpu":
        return contextlib.nullcontext()
    if mode not in ("bf16", "fp16"):
        raise ValueError(f'train.amp must be bf16/fp16/none/auto, got {mode!r} '
                         "(auto must be resolved with resolve_amp first)")
    return torch.autocast("cuda", dtype=torch.bfloat16 if mode == "bf16" else torch.float16)


def _torch_save(path, state):
    """torch.save to a temp file in the same directory, then rename it into place."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    os.close(fd)
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def torch_save(path, state):
    """Atomic checkpoint write, retried over transient filesystem errors.

    Unlike the log, a failed checkpoint still raises: losing one silently is
    worse than stopping.
    """
    return retry_io(_torch_save, path, state)


def torch_load(path, device):
    """Load a checkpoint written by this package onto `device`."""
    import torch

    # Locally produced research checkpoints only.
    return torch.load(path, map_location=device, weights_only=False)


class EMA:
    """Exponential moving average of parameters (kept on the training device)."""

    def __init__(self, module, decay):
        import torch

        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float() for k, v in module.state_dict().items()
                       if v.dtype.is_floating_point}
        self.buffers = {k: v.detach().clone() for k, v in module.state_dict().items()
                        if not v.dtype.is_floating_point}
        self._torch = torch

    @property
    def state(self):
        """EMA weights plus non-float buffers, as a state_dict."""
        out = dict(self.buffers)
        out.update(self.shadow)
        return out

    def update(self, module):
        """shadow = decay * shadow + (1 - decay) * current weights."""
        with self._torch.no_grad():
            for k, v in module.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
                else:
                    self.buffers[k] = v.detach().clone()

    def copy_to(self, module):
        """Load the EMA weights into `module` (used for probes, checkpoints and sampling)."""
        module.load_state_dict({k: v for k, v in self.state.items()}, strict=True)

    def load_state(self, state):
        """Restore the EMA from a saved state_dict (resume)."""
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k] = v.detach().clone().float()
            else:
                self.buffers[k] = v.detach().clone()
