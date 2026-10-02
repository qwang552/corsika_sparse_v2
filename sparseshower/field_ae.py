"""FieldAE: 48^3 x 2 coarse field <-> 8 x 12^3 latent.

Model 1 of 4. The FieldAE compresses the 48^3 coarse field of an event into a
small latent (8 x 12^3) that the LatentDiT learns to generate, and decodes a
latent back into the field F = (p_occ, ch0, ch1) that struct and attr start
from. This file also holds the latent cache (`encode_cache`, `LatentStore`)
written by the `encode` command, and `ReconSource`, the one path from a latent
to F, used by struct / attr training and by every sampling chain.

    encoder   Conv3d 2 -> w0 @48, ResBlocks, stride-2 down -> w1 @24, ResBlocks,
              down -> w2 @12, ResBlocks + self-attention, -> (mu, logvar) x C
    decoder   mirror image; one 3-channel output conv on the 48^3 grid:
              occ logit (BCE), ch0 and ch1 (MSE on truth-occupied cells)

Loss = BCE(occ; pos_weight, photon weight 1 + alpha*log1p(sum q))
     + w_ch0 * MSE(ch0) + w_ch1 * MSE(ch1)            (occupied cells only)
     + w_profile * (L1 longitudinal + L1 radial + (log total ratio)^2)
     + kl_beta * KL(q(z|x) || N(0, 1))

Latent cache (written by `encode`), under <case>/latents/:
    {split}_ids.npy      (N,)            event ids
    {split}_mu.npy       (N, C, L, L, L) float16 posterior means
    {split}_logvar.npy   (N, C, L, L, L) float16
    train_d4_mu.npy      (N, 8, C, L, L, L) float16   only when encode.d4 is true (8 x-y rotations / mirrors)
    latent_stats.json    per-channel mean / std of train mu (DiT works on the standardized z)
    done.json            AE checkpoint step + fingerprint the cache was made from
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .common import read_json, write_json
from .data import load_event
from .fields import cell_profile_index, coarse_field, d4_field, field_profiles, photon_sums


# ------------------------------------------------------------------ model ---
def _gn(ch):
    """GroupNorm with a group count that suits the channel number."""
    return nn.GroupNorm(min(32, ch // 4) if ch >= 8 else 1, ch)


class ResBlock(nn.Module):
    """3D residual block: (GroupNorm -> SiLU -> Conv3d) x 2 plus a skip connection."""
    def __init__(self, cin, cout):
        super().__init__()
        self.n1, self.c1 = _gn(cin), nn.Conv3d(cin, cout, 3, padding=1)
        self.n2, self.c2 = _gn(cout), nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """Global self-attention over the D^3 cells of a small grid."""

    def __init__(self, ch, heads):
        super().__init__()
        self.norm = _gn(ch)
        self.attn = nn.MultiheadAttention(ch, int(heads), batch_first=True)

    def forward(self, x):
        b, c, d, h, w = x.shape
        tok = self.norm(x).flatten(2).transpose(1, 2)            # (B, DHW, C)
        out, _ = self.attn(tok, tok, tok, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, d, h, w)


class FieldAE(nn.Module):
    """Convolutional VAE-style autoencoder between the 48^3 field and the 8 x 12^3 latent."""
    def __init__(self, cfg, in_channels=2):
        super().__init__()
        widths = [int(w) for w in cfg["widths"]]
        nb = int(cfg["blocks"])
        heads = int(cfg.get("attn_heads", 0))
        self.latent_channels = int(cfg["latent_channels"])
        self.down_factor = 2 ** (len(widths) - 1)
        self.use_ckpt = False
        # encoder
        enc = [nn.Conv3d(in_channels, widths[0], 3, padding=1)]
        for i, w in enumerate(widths):
            cin = widths[i - 1] if i else widths[0]
            if i:
                enc.append(nn.Conv3d(cin, w, 3, stride=2, padding=1))
            enc += [ResBlock(w, w) for _ in range(nb)]
        if heads:
            enc.append(AttnBlock(widths[-1], heads))
        enc += [_gn(widths[-1]), nn.SiLU(), nn.Conv3d(widths[-1], 2 * self.latent_channels, 3, padding=1)]
        self.encoder = nn.ModuleList(enc)
        # decoder
        dec = [nn.Conv3d(self.latent_channels, widths[-1], 3, padding=1)]
        dec += [ResBlock(widths[-1], widths[-1]) for _ in range(nb)]
        if heads:
            dec.append(AttnBlock(widths[-1], heads))
        for i in reversed(range(len(widths) - 1)):
            dec += [nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv3d(widths[i + 1], widths[i], 3, padding=1)]
            dec += [ResBlock(widths[i], widths[i]) for _ in range(nb)]
        dec += [_gn(widths[0]), nn.SiLU(), nn.Conv3d(widths[0], 3, 3, padding=1)]
        self.decoder = nn.ModuleList(dec)

    def _run(self, layers, x):
        """Run a layer list, with activation checkpointing on the heavy blocks when enabled."""
        for layer in layers:
            if self.use_ckpt and self.training and torch.is_grad_enabled() \
                    and isinstance(layer, (ResBlock, AttnBlock)):
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

    def encode(self, field):
        """Field (B, 2, 48^3) -> latent mean and log-variance, each (B, 8, 12, 12, 12)."""
        h = self._run(self.encoder, field)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar.clamp(-30.0, 20.0)

    def decode(self, z):
        """-> (B, 3, D, D, D): occ logit, ch0, ch1."""
        return self._run(self.decoder, z)

    def forward(self, field, sample=True, generator=None):
        """Encode, draw z (or take the mean), decode; returns (decoder output, mu, logvar)."""
        mu, logvar = self.encode(field)
        z = mu
        if sample:
            eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
            z = mu + torch.exp(0.5 * logvar) * eps
        return self.decode(z), mu, logvar


def decoded_field(out):
    """Decoder output -> F = (p_occ, ch0, ch1), the downstream format."""
    return torch.cat([torch.sigmoid(out[:, :1].float()), out[:, 1:3].float()], dim=1)


def build_ae(c, device):
    """Construct the FieldAE from the `ae` config section."""
    return FieldAE(c["ae"]).to(device)


# ------------------------------------------------------------------- loss ---
class ProfileIndex:
    """z-slab / ring index of each base cell, as tensors (for the profile loss)."""

    def __init__(self, c, device, z_slabs=32, r_rings=16):
        zs, rr = cell_profile_index(c, int(c["field"]["base_grid"]), z_slabs, r_rings)
        self.zs = torch.as_tensor(zs, device=device)
        self.rr = torch.as_tensor(rr, device=device)
        self.z_slabs, self.r_rings = z_slabs, r_rings

    def profiles(self, sums):
        """sums (B, D^3) -> normalized (B, Z), (B, R) and totals (B,)."""
        b = sums.shape[0]
        zp = torch.zeros(b, self.z_slabs, device=sums.device, dtype=sums.dtype)
        rp = torch.zeros(b, self.r_rings, device=sums.device, dtype=sums.dtype)
        zp = zp.index_add(1, self.zs, sums)
        rp = rp.index_add(1, self.rr, sums)
        tot = sums.sum(dim=1).clamp_min(1e-6)
        return zp / tot[:, None], rp / tot[:, None], tot


def ae_loss(out, mu, logvar, field, sums, cfg, stats, pidx):
    """field (B,2,D,D,D) truth; sums (B, D^3) truth photon sums per cell."""
    out = out.float()
    logit, ch0, ch1 = out[:, 0], out[:, 1], out[:, 2]
    occ = (field[:, 0] > 0).float()
    w = torch.where(occ > 0, float(cfg["pos_weight"])
                    * (1.0 + float(cfg["photon_alpha"]) * torch.log1p(sums.reshape(occ.shape))),
                    torch.ones_like(occ))
    bce = (F.binary_cross_entropy_with_logits(logit, occ, reduction="none") * w).sum() / w.sum()
    n_occ = occ.sum().clamp_min(1.0)
    mse0 = ((ch0 - field[:, 0]).square() * occ).sum() / n_occ
    mse1 = ((ch1 - field[:, 1]).square() * occ).sum() / n_occ
    # expected photons per cell: p_occ * exp(ch1 * q_std + q_mean)
    log_s = (ch1.clamp(-8.0, 8.0) * float(stats["q_std"]) + float(stats["q_mean"]))
    pred_sums = (torch.sigmoid(logit) * torch.exp(log_s)).flatten(1)
    zp, rp, tp = pidx.profiles(pred_sums)
    zt, rt, tt = pidx.profiles(sums.float())
    prof = (zp - zt).abs().sum(1).mean() + (rp - rt).abs().sum(1).mean() \
        + (torch.log(tp) - torch.log(tt)).square().mean()
    kl = 0.5 * (mu.float().square() + logvar.float().exp() - 1.0 - logvar.float()).flatten(1).sum(1).mean()
    total = bce + float(cfg["w_ch0"]) * mse0 + float(cfg["w_ch1"]) * mse1 \
        + float(cfg["w_profile"]) * prof + float(cfg["kl_beta"]) * kl
    logs = dict(bce=float(bce), mse0=float(mse0), mse1=float(mse1), prof=float(prof), kl=float(kl))
    return total, logs


# ------------------------------------------------------------- per event ---
def event_field(ev, c, stats):
    """Truth field (2,B,B,B), per-cell photon sums and occupied-voxel counts (flat B^3) of one cached event."""
    field, counts, sums = coarse_field(ev["ijk"], ev["q"], int(c["data"]["grid"]),
                                       int(c["field"]["base_grid"]), float(c["data"]["q_eps"]), stats)
    return field, sums.astype(np.float32), counts


def ae_event_metrics(F_np, truth_counts, truth_sums, c, stats, pidx_np):
    """Acceptance numbers of one decoded field against its truth (NumPy).

    F_np (3,B,B,B) decoded; truth counts / sums flat (B^3,).
    """
    thr = float(c["field"]["occ_threshold"])
    occ_pred = F_np[0].reshape(-1) > thr
    occ_true = np.asarray(truth_counts) > 0
    sums_t = np.asarray(truth_sums, dtype=np.float64)
    sums_p = photon_sums(F_np[2].reshape(-1), occ_pred, stats, float(c["data"]["q_eps"]))
    zs, rr = pidx_np
    lt, rt = field_profiles(sums_t, zs, rr)
    lp, rp = field_profiles(sums_p, zs, rr)
    tot_t = max(sums_t.sum(), 1e-30)
    return dict(
        lost_photon_frac=float(sums_t[occ_true & ~occ_pred].sum() / tot_t),
        n48_rel_err=float(abs(occ_pred.sum() / max(occ_true.sum(), 1) - 1.0)),
        field_long_l1=float(np.abs(lp - lt).sum()),
        field_rad_l1=float(np.abs(rp - rt).sum()),
        q_total_rel_err=float(abs(sums_p.sum() / tot_t - 1.0)),
        extra_cells_frac=float((occ_pred & ~occ_true).sum() / max(occ_true.sum(), 1)),
        n48_true=int(occ_true.sum()), n48_pred=int(occ_pred.sum()))


# ----------------------------------------------------------------- latents ---
def latent_dir(case_path):
    """Directory of the latent cache of a case: <case>/latents."""
    return Path(case_path) / "latents"


def load_latent_stats(case_path):
    """Per-channel mean / std of the train latents (used to standardize for the DiT)."""
    s = read_json(latent_dir(case_path) / "latent_stats.json")
    mean = np.asarray(s["mean"], dtype=np.float32)
    std = np.asarray(s["std"], dtype=np.float32)
    return mean, std


class LatentStore:
    """Read-only access to the latent cache (memory-mapped)."""

    def __init__(self, case_path):
        self.dir = latent_dir(case_path)
        if not (self.dir / "done.json").exists():
            raise FileNotFoundError(f"{self.dir}/done.json missing: run `encode` for this case first")
        self.done = read_json(self.dir / "done.json")
        self.mean, self.std = load_latent_stats(case_path)
        self._arrays, self._index = {}, {}

    def _arr(self, name):
        if name not in self._arrays:
            self._arrays[name] = np.load(self.dir / f"{name}.npy", mmap_mode="r")
        return self._arrays[name]

    def ids(self, split):
        """Event ids of one split in the latent cache."""
        return [int(e) for e in self._arr(f"{split}_ids")]

    def _ensure_index(self, split):
        if split not in self._index:
            self._index[split] = {int(e): i for i, e in enumerate(self._arr(f"{split}_ids"))}
        return self._index[split]

    def row(self, split, eid):
        """Row of event `eid` in the arrays of `split`."""
        return self._ensure_index(split)[int(eid)]

    def split_of(self, eid):
        """Which split an event belongs to."""
        for split in ("train", "val", "test"):
            if (self.dir / f"{split}_ids.npy").exists() and int(eid) in self._ensure_index(split):
                return split
        raise KeyError(f"event {eid} not in the latent cache")

    def posterior(self, eid, split=None):
        """(mu, logvar) of one event, unstandardized."""
        split = split or self.split_of(eid)
        i = self.row(split, eid)
        mu = np.asarray(self._arr(f"{split}_mu")[i], dtype=np.float32)
        lv = np.asarray(self._arr(f"{split}_logvar")[i], dtype=np.float32)
        return mu, lv

    def has_d4(self):
        """True if D4-augmented train latents were written."""
        return (self.dir / "train_d4_mu.npy").exists()

    def train_mu(self, rows, transforms=None):
        """Standardized train latents; with D4, `transforms` picks the variant per row."""
        rows = np.asarray(rows, dtype=np.int64)
        if transforms is None or not self.has_d4():
            z = np.asarray(self._arr("train_mu")[rows], dtype=np.float32)
        else:
            arr = self._arr("train_d4_mu")
            z = np.stack([np.asarray(arr[r, t], dtype=np.float32)
                          for r, t in zip(rows, transforms)])
        return self.standardize(z)

    def standardize(self, z):
        """Raw latent -> standardized latent (per channel)."""
        return (z - self.mean[:, None, None, None]) / self.std[:, None, None, None]

    def unstandardize(self, z):
        """Standardized latent -> raw latent (per channel)."""
        return z * self.std[:, None, None, None] + self.mean[:, None, None, None]


class ReconSource:
    """Latent -> decoded field F = (p_occ, ch0, ch1) on the training device.

    Used by struct/attr training ("train on the reconstructed field") and by
    every sampling chain.  The decoder is frozen and in eval mode.
    """

    def __init__(self, ae, store, device):
        self.ae, self.store, self.device = ae, store, device
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def decode_raw(self, z_raw):
        """z_raw (B, C, L, L, L) unstandardized -> F (B, 3, D, D, D) float32."""
        z = torch.as_tensor(z_raw, device=self.device, dtype=torch.float32)
        if z.dim() == 4:
            z = z[None]
        return decoded_field(self.ae.decode(z))

    @torch.no_grad()
    def recon(self, eid, posterior=False, rng=None, split=None):
        """AE-reconstructed field of a cached event (mean latent, or a posterior draw)."""
        mu, lv = self.store.posterior(eid, split)
        z = mu
        if posterior:
            rng = rng or np.random.default_rng()
            z = mu + np.exp(0.5 * lv) * rng.standard_normal(mu.shape).astype(np.float32)
        return self.decode_raw(z)[0]


def load_ae(case_path, checkpoint_name, device, c):
    """Load the EMA weights of ae_<name>.pt from a case directory."""
    from .common import torch_load

    state = torch_load(Path(case_path) / f"ae_{checkpoint_name}.pt", device)
    ae = build_ae(c, device)
    ae.load_state_dict(state["ema"])
    ae.eval()
    return ae, state


# ------------------------------------------------------------------ encode ---
@torch.no_grad()
def encode_cache(c, case_path, checkpoint_name="best", allow_cpu=False, splits=("train", "val", "test"),
                 limit=None):
    """Encode every event of the voxel cache once; D4 variants of train if enabled."""
    from .common import device_for
    from .fields import d4_enabled, load_field_stats
    from .loader import meta_for

    device = device_for(allow_cpu)
    case_path = Path(case_path)
    out = latent_dir(case_path)
    ae, state = load_ae(case_path, checkpoint_name, device, c)
    done_path = out / "done.json"
    tag = dict(ae_step=int(state["step"]), ae_fingerprint=state.get("fingerprint"),
               checkpoint=checkpoint_name)
    if done_path.exists():
        prev = read_json(done_path)
        if {k: prev.get(k) for k in tag} == tag:
            print(f"latent cache already complete for {tag}; nothing to do", flush=True)
            return prev
        raise RuntimeError(f"{out} was made from a different AE checkpoint ({prev}); "
                           "use a NEW case for a new AE (the cache is never overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    meta = meta_for(c)
    root = Path(c["paths"]["processed"])
    stats = load_field_stats(c)
    use_d4 = d4_enabled(c, c["encode"]["d4"])
    bs = max(1, int(c["encode"]["batch_size"]))
    lc = int(c["ae"]["latent_channels"])
    lat = int(c["field"]["base_grid"]) // ae.down_factor
    t0 = time.time()
    for split in splits:
        ids = [int(e) for e in meta["split"][split]][: (int(limit) if limit else None)]
        if not ids:
            continue
        mu_mm = np.lib.format.open_memmap(out / f"{split}_mu.tmp.npy", mode="w+", dtype=np.float16,
                                          shape=(len(ids), lc, lat, lat, lat))
        lv_mm = np.lib.format.open_memmap(out / f"{split}_logvar.tmp.npy", mode="w+", dtype=np.float16,
                                          shape=(len(ids), lc, lat, lat, lat))
        d4_mm = None
        if use_d4 and split == "train":
            d4_mm = np.lib.format.open_memmap(out / "train_d4_mu.tmp.npy", mode="w+", dtype=np.float16,
                                              shape=(len(ids), 8, lc, lat, lat, lat))
        for start in range(0, len(ids), bs):
            chunk = ids[start:start + bs]
            fields = np.stack([event_field(load_event(root, e), c, stats)[0] for e in chunk])
            mu, lv = ae.encode(torch.as_tensor(fields, device=device))
            mu_mm[start:start + len(chunk)] = mu.cpu().numpy().astype(np.float16)
            lv_mm[start:start + len(chunk)] = lv.cpu().numpy().astype(np.float16)
            if d4_mm is not None:
                d4_mm[start:start + len(chunk), 0] = mu_mm[start:start + len(chunk)]
                for t in range(1, 8):
                    ft = np.stack([d4_field(f, t) for f in fields])
                    mt, _ = ae.encode(torch.as_tensor(np.ascontiguousarray(ft), device=device))
                    d4_mm[start:start + len(chunk), t] = mt.cpu().numpy().astype(np.float16)
            if (start // bs) % 50 == 0:
                print(f"  encode {split}: {start + len(chunk)}/{len(ids)}  "
                      f"({time.time() - t0:.0f} s)", flush=True)
        for mm in (mu_mm, lv_mm, d4_mm):
            if mm is not None:
                mm.flush()
        del mu_mm, lv_mm, d4_mm
        np.save(out / f"{split}_ids.npy", np.asarray(ids, dtype=np.int64))
        for name in (f"{split}_mu", f"{split}_logvar") + (("train_d4_mu",) if use_d4 and split == "train" else ()):
            Path(out / f"{name}.tmp.npy").replace(out / f"{name}.npy")
    train_mu = np.load(out / "train_mu.npy", mmap_mode="r")
    sub = np.asarray(train_mu[: min(len(train_mu), 2000)], dtype=np.float32)
    mean = sub.mean(axis=(0, 2, 3, 4))
    std = np.maximum(sub.std(axis=(0, 2, 3, 4)), 1e-6)
    write_json(out / "latent_stats.json", dict(mean=mean.tolist(), std=std.tolist(),
                                               n_events=int(len(sub))))
    info = dict(tag, d4=bool(use_d4), latent_shape=[lc, lat, lat, lat],
                splits={s: int(len(np.load(out / f"{s}_ids.npy"))) for s in splits
                        if (out / f"{s}_ids.npy").exists()},
                seconds=round(time.time() - t0, 1))
    write_json(done_path, info)
    return info

