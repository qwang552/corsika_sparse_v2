"""One training loop for the four kinds: ae, dit, struct, attr.

Run with `train --kind ae|dit|struct|attr` (jobs/train.sub). Checkpoints are
<case>/<kind>_last.pt and <kind>_best.pt. Resubmitting the same job resumes
from <kind>_last.pt; resume refuses a changed config (start a NEW --case).
The best checkpoint is chosen by the probe score R_mean on fixed validation
events, lower is better (ae: loss; dit, attr: mean denoising ratio
|D - x|^2 / |sigma n|^2 at fixed sigmas; struct: mean cross-entropy).

Shared: event selection, fingerprint + resume, EMA, AMP, gradient clipping,
OOM skip, fixed-seed probes, best/last checkpoints, memory report at step 1.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .common import (EMA, JsonlLog, amp_context, device_for, lock, resolve_amp, seed_all,
                     torch_load, torch_save, write_json)
from .data import Codec, load_event
from .loader import (LRU, Stream, case_dir, checkpoint_decision, fingerprint, meta_for,
                     select_events)

KINDS = ("ae", "dit", "struct", "attr")
OOM_ERRORS = (getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError),)


def resolve_checkpoint(value, below_gib, device):
    """Decide activation checkpointing from model.checkpoint and the GPU memory, and log it."""
    gib = None
    if device.type == "cuda":
        gib = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    on = checkpoint_decision(value, gib, below_gib)
    if isinstance(value, str):
        name = torch.cuda.get_device_name(device) if gib else str(device)
        print(f"checkpoint: {value} -> {on}  ({name}" + (f", {gib:.2f} GiB" if gib else "") + ")",
              flush=True)
    return on


def memory_report(kind, device, meta, n_active, checkpointing):
    """Peak GPU memory after step 1 and the worst case it implies (sparse kinds scale
    with the number of active voxels; ae / dit are fixed-size)."""
    if device.type != "cuda":
        return None
    peak = int(torch.cuda.max_memory_allocated(device))
    total = int(torch.cuda.get_device_properties(device).total_memory)
    n_now = int(n_active or 1)
    n_max = int(meta["stats"].get("n_active_max", n_now))
    scale = max(1.0, n_max / n_now) if kind in ("struct", "attr") else 1.0
    projected = peak * scale
    gib = 1024 ** 3
    print(f"GPU memory after step 1: peak {peak / gib:.2f} GiB of {total / gib:.2f} GiB on "
          f"{torch.cuda.get_device_name(device)}"
          + (f"; worst case (n_active {n_now} -> {n_max}) ~ {projected / gib:.2f} GiB" if scale > 1 else ""),
          flush=True)
    if projected > total:
        print("  THE LARGEST EVENTS WILL NOT FIT" + ("" if checkpointing else
              " - set model.checkpoint: true and start a NEW --case"), flush=True)
    return dict(peak_bytes=peak, total_bytes=total, n_active=n_now, n_active_max=n_max,
                projected_bytes=int(projected), checkpointing=bool(checkpointing))


# ------------------------------------------------------------------ tasks ---
class Task:
    """Base class of one trainable kind: build items (CPU), take a step (GPU), probe."""
    kind = None
    batch = 1

    def __init__(self, c, case_path, device, meta, ids, probe_ids):
        self.c, self.case_path, self.device, self.meta = c, Path(case_path), device, meta
        self.ids, self.probe_ids = ids, probe_ids
        self.root = Path(c["paths"]["processed"])

    # build(eids, rng) -> item   (prefetch thread)
    # step(item) -> (loss, logs, n_active)
    # probe(module) -> dict with R_mean


class AETask(Task):
    """Training task of the FieldAE: batches of truth 48^3 fields."""
    kind = "ae"

    def __init__(self, *a):
        super().__init__(*a)
        from .field_ae import ProfileIndex, build_ae
        from .fields import load_field_stats, stats_events

        c = self.c
        self.stats = load_field_stats(c, create_from=stats_events(c, self.meta))
        self.batch = int(c["ae"]["batch_size"])
        self.model = build_ae(c, self.device)
        self.eval_model = build_ae(c, self.device)
        self.pidx = ProfileIndex(c, self.device)
        self.fields = LRU(self._field, int(c["train"]["cache_samples"]) * 4)

    def _field(self, eid):
        """Truth field and per-cell photon sums of one event."""
        from .field_ae import event_field

        f, sums, _ = event_field(load_event(self.root, eid), self.c, self.stats)
        return f, sums

    def build(self, eids, rng):
        """One batch of truth fields (built on the worker thread)."""
        items = [self.fields.get(int(e)) for e in eids]
        return dict(field=np.stack([f for f, _ in items]), sums=np.stack([s for _, s in items]),
                    seed=int(rng.integers(2 ** 31)))

    def step(self, item, model=None, sample=True):
        """Forward pass and AE loss of one batch."""
        from .field_ae import ae_loss

        model = model or self.model
        field = torch.as_tensor(item["field"], device=self.device)
        sums = torch.as_tensor(item["sums"], device=self.device)
        gen = torch.Generator(device=self.device).manual_seed(item["seed"])
        out, mu, lv = model(field, sample=sample, generator=gen)
        loss, logs = ae_loss(out, mu, lv, field, sums, self.c["ae"], self.stats, self.pidx)
        return loss, logs, None

    @torch.no_grad()
    def probe(self, model):
        """AE loss and acceptance numbers on the fixed probe events."""
        from .field_ae import ae_event_metrics, decoded_field, event_field
        from .fields import cell_profile_index

        pidx = cell_profile_index(self.c, int(self.c["field"]["base_grid"]))
        losses, rows = [], []
        for e in self.probe_ids:
            f, sums, counts = event_field(load_event(self.root, e), self.c, self.stats)
            item = dict(field=f[None], sums=sums[None], seed=0)
            loss, _, _ = self.step(item, model=model, sample=False)
            losses.append(float(loss))
            out, _, _ = model(torch.as_tensor(f[None], device=self.device), sample=False)
            rows.append(ae_event_metrics(decoded_field(out)[0].cpu().numpy(), counts, sums,
                                         self.c, self.stats, pidx))
        res = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        res["loss"] = float(np.mean(losses))
        res["R_mean"] = res["loss"]
        return res


class DiTTask(Task):
    """Training task of the LatentDiT: batches of standardized train latents (optionally D4)."""
    kind = "dit"

    def __init__(self, *a):
        super().__init__(*a)
        from .dit import build_dit
        from .field_ae import LatentStore
        from .fields import d4_enabled

        c = self.c
        self.store = LatentStore(self.case_path)
        self.batch = int(c["dit"]["batch_size"])
        self.model = build_dit(c, self.device)
        self.eval_model = build_dit(c, self.device)
        self.d4 = d4_enabled(c, c["dit"]["d4"]) and self.store.has_d4()
        if d4_enabled(c, c["dit"]["d4"]) and not self.store.has_d4():
            print("dit.d4 is on but the latent cache has no D4 variants; training without", flush=True)
        train_ids = self.store.ids("train")
        wanted = set(self.ids)
        self.rows = np.asarray([i for i, e in enumerate(train_ids) if e in wanted], dtype=np.int64)
        val = self.store.ids("val")[: int(c["dit"]["probe_events"])]
        self.probe_z = np.stack([self.store.standardize(self.store.posterior(e, "val")[0]) for e in val]) \
            if val else self.store.train_mu(self.rows[:8])
        print(f"dit: {len(self.rows)} train latents, D4 augmentation {'on' if self.d4 else 'off'}",
              flush=True)

    def build(self, eids, rng):
        """One batch of random train latents (random D4 variants when enabled)."""
        rows = self.rows[rng.integers(len(self.rows), size=self.batch)]
        tr = rng.integers(8, size=self.batch) if self.d4 else None
        return dict(z=self.store.train_mu(rows, tr), seed=int(rng.integers(2 ** 31)))

    def step(self, item, model=None):
        """EDM loss of the DiT on one batch."""
        from .diffusion import edm_loss_dense, sample_sigmas
        from .dit import dit_denoiser

        model = model or self.model
        z = torch.as_tensor(item["z"], device=self.device)
        gen = torch.Generator(device=self.device).manual_seed(item["seed"])
        sigma = sample_sigmas(len(z), self.c["edm"], self.device, gen)
        noise = torch.randn(z.shape, device=self.device, generator=gen)
        loss, _ = edm_loss_dense(dit_denoiser(model, self.c["edm"]), z, sigma, noise=noise)
        return loss, dict(sigma_mean=float(sigma.mean())), None

    @torch.no_grad()
    def probe(self, model):
        """Denoising ratio R at fixed sigmas on fixed latents (lower is better)."""
        from .dit import dit_denoiser

        den = dit_denoiser(model, self.c["edm"])
        z = torch.as_tensor(self.probe_z, device=self.device)
        out = {}
        for s in (0.1, 0.5, 1.0, 2.0, 5.0):
            gen = torch.Generator(device=self.device).manual_seed(int(1000 * s) + 3)
            noise = torch.randn(z.shape, device=self.device, generator=gen)
            sig = torch.full((len(z), 1, 1, 1, 1), s, device=self.device)
            pred = den(z + sig * noise, sig)
            out[f"R@{s:g}"] = float((pred - z).square().mean() / (sig * noise).square().mean())
        out["R_mean"] = float(np.mean(list(out.values())))
        return out


class _ReconMixin:
    """Gives struct / attr the AE-reconstructed 48^3 field of an event (their input)."""
    def _setup_recon(self):
        """Load the latent cache and the AE checkpoint it was encoded with."""
        from .field_ae import LatentStore, ReconSource, load_ae

        self.store = LatentStore(self.case_path)
        ae, _ = load_ae(self.case_path, self.store.done["checkpoint"], self.device, self.c)
        self.recon_src = ReconSource(ae, self.store, self.device)

    def recon(self, eid, posterior=False, seed=0):
        """Decoded field of one event (latent mean, or a posterior draw)."""
        return self.recon_src.recon(eid, posterior=posterior, rng=np.random.default_rng(seed))


class StructTask(Task, _ReconMixin):
    """Training task of the structure model (one event per step)."""
    kind = "struct"

    def __init__(self, *a):
        super().__init__(*a)
        from .structure import build_struct

        self._setup_recon()
        self.model = build_struct(self.c, self.device)
        self.eval_model = build_struct(self.c, self.device)
        self.events = LRU(lambda e: load_event(self.root, e), int(self.c["train"]["cache_samples"]))

    def build(self, eids, rng):
        """CPU part of one step (parent sets, targets, corruptions)."""
        from .structure import plan_step

        e = int(eids[0])
        return plan_step(e, self.events.get(e), self.c, rng)

    def step(self, item, model=None):
        """GPU part of one step: decode the field, compute the configuration loss."""
        from .structure import struct_step

        F_t = self.recon(item["eid"], posterior=item["posterior"], seed=item["seed"])
        loss, logs = struct_step(model or self.model, item, F_t, self.c, self.device, train=True)
        return loss, logs, item["n_active"]

    def probe(self, model):
        """Teacher-forced cross-entropy on the fixed probe events."""
        from .structure import probe

        events = [(e, load_event(self.root, e)) for e in self.probe_ids]
        return probe(model, events, lambda e, posterior=False: self.recon(e, posterior), self.c,
                     self.device)


class AttrTask(Task, _ReconMixin):
    """Training task of the attribute model (one event per step)."""
    kind = "attr"

    def __init__(self, *a):
        super().__init__(*a)
        from .attr import build_attr, fine_inputs

        self._setup_recon()
        self.codec = Codec(self.meta["stats"], float(self.c["data"]["q_eps"]))
        self.model = build_attr(self.c, self.device)
        self.eval_model = build_attr(self.c, self.device)
        self.events = LRU(lambda e: load_event(self.root, e), int(self.c["train"]["cache_samples"]))
        self.inputs = LRU(lambda e: fine_inputs(self.events.get(e)["ijk"], self.c),
                          int(self.c["train"]["cache_samples"]))

    def build(self, eids, rng):
        """CPU part of one step (voxel set, targets, structure dropout)."""
        from .attr import plan_step

        e = int(eids[0])
        return plan_step(e, self.events.get(e), self.c, self.codec, rng, cache=self.inputs)

    def step(self, item, model=None):
        """GPU part of one step: decode the field, compute the EDM loss."""
        from .attr import attr_step

        F_t = self.recon(item["eid"], posterior=item["posterior"], seed=item["seed"])
        gen = torch.Generator(device=self.device).manual_seed(item["seed"])
        loss, logs = attr_step(model or self.model, item, F_t, self.c, self.device, gen)
        return loss, logs, item["n_active"]

    def probe(self, model):
        """Denoising ratio R at fixed sigmas on the fixed probe events."""
        from .attr import probe

        items = []
        for e in self.probe_ids:
            ev = self.events.get(e)
            items.append((e, self.codec.encode(ev["q"], ev["off"]), self.inputs.get(e)))
        return probe(model, items, lambda e, posterior=False: self.recon(e, posterior), self.c,
                     self.device)


TASKS = dict(ae=AETask, dit=DiTTask, struct=StructTask, attr=AttrTask)


# ------------------------------------------------------------------- loop ---
def train(c, kind, case, n_events=None, steps=None, allow_cpu=False, resume=True):
    """Train one kind: resume if possible, loop over steps, probe, write last / best checkpoints.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    meta = meta_for(c)
    device = device_for(allow_cpu)
    seed_all(int(c["seed"]) + KINDS.index(kind))
    out_dir = case_dir(c, case)
    out_dir.mkdir(parents=True, exist_ok=True)
    sec = c[kind]
    ids = select_events(meta, n_events, int(c["seed"]))
    n_probe = int(sec.get("probe_events", 2))
    val_ids = list(meta["split"]["val"])[:n_probe]
    probe_ids = [int(e) for e in (val_ids or ids[:n_probe])]
    fp = fingerprint(c, kind, ids)
    # machine-dependent switches, resolved after the fingerprint
    ck_value = c["model"].get("checkpoint", "auto")
    below = float(c["model"].get("checkpoint_below_gib", 16))
    use_ckpt = resolve_checkpoint(ck_value, below, device)
    c["model"]["checkpoint"] = use_ckpt
    c["train"]["amp"] = resolve_amp(c["train"]["amp"], device)

    task = TASKS[kind](c, out_dir, device, meta, ids, probe_ids)
    net = task.model
    if hasattr(net, "use_ckpt"):
        net.use_ckpt = use_ckpt
    total_steps = int(steps or sec["steps"])
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=float(sec.get("lr") or c["train"]["lr"]),
                            weight_decay=float(c["train"]["weight_decay"]))
    ema = EMA(net, float(sec.get("ema") or c["train"]["ema"]))
    scaler = torch.amp.GradScaler("cuda", enabled=(c["train"]["amp"] == "fp16" and device.type == "cuda"))
    start, best = 0, float("inf")
    ckpt_path = out_dir / f"{kind}_last.pt"
    n_params = int(sum(p.numel() for p in net.parameters()))

    with lock(out_dir / f".{kind}.lock"):
        if resume and ckpt_path.exists():
            state = torch_load(ckpt_path, device)
            if state.get("fingerprint") != fp:
                raise RuntimeError(f"{kind} checkpoint fingerprint mismatch in case {case!r}: the "
                                   "config or events changed.  Start a NEW --case (nothing is overwritten).")
            net.load_state_dict(state["model"])
            ema.load_state(state["ema"])
            opt.load_state_dict(state["optimizer"])
            if state.get("scaler") and scaler.is_enabled():
                scaler.load_state_dict(state["scaler"])
            start, best = int(state["step"]), float(state.get("best", float("inf")))
        write_json(out_dir / f"{kind}_config.json",
                   dict(fingerprint=fp, kind=kind, events=len(ids), parameters=n_params,
                        total_steps=total_steps, device=str(device), probe_events=probe_ids,
                        checkpointing=bool(use_ckpt), amp=c["train"]["amp"], config=c))
        print(f"{kind}: {n_params:,} parameters, steps {start}->{total_steps}, {len(ids)} events, "
              f"device {device}", flush=True)
        accum = max(1, int(sec.get("accum", 1)))
        stream = Stream(task.build, ids, [int(c["seed"]) + 7 + KINDS.index(kind), start, 1],
                        batch=task.batch, depth=int(c["train"].get("prefetch", 4)))
        history = JsonlLog(out_dir / f"{kind}_history.jsonl")
        t0, oom_skips, oom_run = time.time(), 0, 0

        def payload(step):
            return dict(model=net.state_dict(), ema=ema.state, optimizer=opt.state_dict(),
                        step=step, best=best, fingerprint=fp, kind=kind, scaler=scaler.state_dict())

        try:
            for step in range(start + 1, total_steps + 1):
                net.train()
                opt.zero_grad(set_to_none=True)
                value, logs, n_act = None, {}, None
                try:
                    for _ in range(accum):
                        item = stream.next()
                        with amp_context(device, c["train"]["amp"]):
                            value, logs, n_act = task.step(item)
                        if value is None:
                            break
                        scaler.scale(value / accum).backward()
                except OOM_ERRORS:
                    opt.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    oom_skips += 1
                    oom_run += 1
                    print(f"step {step}: CUDA OOM (n_active {n_act}) - skipped ({oom_skips} so far)", flush=True)
                    history.write(dict(step=step, skipped="cuda oom", n_active=n_act), now=True)
                    if oom_run >= 20:
                        raise RuntimeError("20 consecutive CUDA OOM steps: enable model.checkpoint "
                                           "or reduce the model")
                    continue
                if value is None:
                    history.write(dict(step=step, skipped="empty item", **(logs or {})))
                    continue
                oom_run = 0
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), float(c["train"]["grad_clip"]))
                if not torch.isfinite(grad_norm):
                    opt.zero_grad(set_to_none=True)
                    scaler.update()
                    history.write(dict(step=step, skipped="non-finite gradient"))
                    continue
                scaler.step(opt)
                scaler.update()
                ema.update(net)
                record = dict(step=step, loss=float(value.detach()), grad=float(grad_norm), **logs)
                if step == start + 1:
                    mem = memory_report(kind, device, meta, n_act, use_ckpt)
                    if mem:
                        record["memory"] = mem
                        write_json(out_dir / f"{kind}_memory.json", mem)
                probed = step % int(c["train"]["probe_every"]) == 0 or step == total_steps
                if probed:
                    task.eval_model.load_state_dict(ema.state)
                    task.eval_model.eval()
                    now = task.probe(task.eval_model)
                    record.update({f"probe_{k}": v for k, v in now.items()})
                    torch_save(ckpt_path, payload(step))
                    if now["R_mean"] < best:
                        best = now["R_mean"]
                        torch_save(out_dir / f"{kind}_best.pt", payload(step))
                    record["elapsed_s"] = round(time.time() - t0, 1)
                    print(f"step {step}: probe " + "  ".join(f"{k} {v:.4g}" for k, v in now.items()
                                                             if isinstance(v, float))[:400], flush=True)
                history.write(record, now=probed)
        finally:
            stream.close()
            history.close()
        torch_save(ckpt_path, payload(total_steps))
        write_json(out_dir / f"{kind}_done.json", dict(step=total_steps, best=best,
                                                        seconds=round(time.time() - t0, 1)))
    return dict(kind=kind, case=case, steps=total_steps, best=best, parameters=n_params,
                oom_skipped=oom_skips, log_dropped=history.dropped)

