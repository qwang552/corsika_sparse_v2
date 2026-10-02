"""Attribute model: sparse EDM for (logQ, offset) on a fixed 192^3 structure.

Model 4 of 4. Given the final 192^3 active voxels and the decoded field, attr
generates, for every voxel, the photon count (logQ) and the sub-voxel offset of
the light, by denoising from noise (EDM).

* Global conditioning: g = FieldEncoder(decoded field).  The attribute model
  has its own FieldEncoder, trained jointly.
* Per-voxel context: the decoded field F = (p_occ, ch0, ch1) at the voxel's
  48^3 ancestor cell, taken from the AE reconstruction of the event.
* Structure dropout (training): with probability drop_p, up to drop_frac of
  the voxels are removed; the remaining voxels keep their true targets.
  Voxels are only removed, never added, because an added voxel has no target.
* T2 (sampling option): photons are rescaled per 48^3 cell so that each cell
  sums to the photon count of the decoded field.

Input per voxel: x_sigma (4) + ctx (3) + occupied-neighbour fraction (1)
+ Fourier positions (inside SparseNet) -> 4 denoised channels.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .diffusion import Denoiser, edm_loss, heun_sample, sample_sigmas
from .fields import photon_sums
from .geometry import neighbour_offsets
from .nn_blocks import SparseNet
from .samples import ancestor_lookup, bundle_for
from .structure import FieldEncoder, geo_tensors, sparse_cfg

N_CH = 4


class AttrNet(nn.Module):
    """Attribute network: FieldEncoder (for g) + SparseNet with 4 outputs per voxel."""
    def __init__(self, c):
        super().__init__()
        a = c["attr"]
        self.encoder = FieldEncoder(int(a["g_dim"]))
        n_off = len(neighbour_offsets(int(c["model"]["neighbours"])))
        self.net = SparseNet(N_CH + 3 + 1, N_CH, int(a["g_dim"]), sparse_cfg(c, "attr"), n_off)

    def forward(self, x, c_noise, geo, g, ctx):
        """Network output for noised channels x at noise level c_noise."""
        occ = geo["nbr_mask"].mean(dim=1, keepdim=True)
        feats = torch.cat([x, ctx.to(x.dtype), occ.to(x.dtype)], dim=-1)
        return self.net(feats, geo, c_noise, g)


def build_attr(c, device):
    """Construct the attribute network."""
    return AttrNet(c).to(device)


def denoiser(net, geo, g, ctx, edm_cfg):
    """Wrap AttrNet into an EDM denoiser for one fixed voxel set."""
    return Denoiser(lambda x, c_noise, **_: net(x, c_noise, geo, g, ctx),
                    float(edm_cfg["sigma_data"]), edm_cfg.get("predict", "edm"))


def fine_inputs(ijk, c):
    """Geometry bundle and 48^3 ancestor lookup of a 192^3 voxel set."""
    grid = int(c["data"]["grid"])
    return dict(bundle=bundle_for(ijk, grid, c),
                lookup=ancestor_lookup(ijk, grid, int(c["field"]["base_grid"])))


def plan_step(eid, ev, c, codec, rng, cache=None):
    """CPU part (prefetch thread): the voxel set (possibly thinned) and its targets."""
    a = c["attr"]
    ijk = np.asarray(ev["ijk"], dtype=np.int64)
    x = codec.encode(ev["q"], ev["off"])
    keep = None
    if rng.random() < float(a["drop_p"]) and len(ijk) > 16:
        n_drop = int(rng.uniform(0.0, float(a["drop_frac"])) * len(ijk))
        if n_drop:
            keep = np.ones(len(ijk), dtype=bool)
            keep[rng.choice(len(ijk), n_drop, replace=False)] = False
    if keep is None:
        inp = cache.get(int(eid)) if cache is not None else fine_inputs(ijk, c)
    else:
        ijk, x = ijk[keep], x[keep]
        inp = fine_inputs(ijk, c)
    return dict(eid=int(eid), x=x, inp=inp, n_active=int(len(ijk)),
                posterior=bool(rng.random() < float(a["posterior_p"])), dropped=keep is not None,
                seed=int(rng.integers(2 ** 31)))


def ctx_from(F_t, lookup, device):
    """Per-voxel context: the decoded field (p_occ, ch0, ch1) at each voxel's 48^3 cell."""
    return F_t.reshape(3, -1)[:, torch.as_tensor(lookup, device=device)].transpose(0, 1).contiguous()


def photon_weights(x_noisy, alpha):
    """Per-voxel loss weight exp(alpha * noisy logQ), so bright voxels count more; it uses only the noisy input, so the optimum is unchanged."""
    if alpha <= 0:
        return None
    return torch.exp(float(alpha) * x_noisy[:, 0].detach().clamp(-3.0, 3.0))


def attr_step(model, plan, F_t, c, device, generator=None):
    """One training step: noise the true channels, denoise, return the weighted EDM loss."""
    geo = geo_tensors(plan["inp"]["bundle"], device)
    ctx = ctx_from(F_t, plan["inp"]["lookup"], device)
    x = torch.as_tensor(plan["x"], device=device)
    g = model.encoder(F_t)
    sigma = sample_sigmas(1, c["edm"], device, generator).reshape(1, 1)
    noise = torch.randn(x.shape, device=device, generator=generator)
    den = denoiser(model, geo, g, ctx, c["edm"])
    w = photon_weights(x + sigma * noise, float(c["attr"]["photon_alpha"]))
    value, _ = edm_loss(den, x, sigma, noise=noise, weights=w)
    return value, dict(sigma=float(sigma), dropped=bool(plan["dropped"]))


@torch.no_grad()
def probe(model, items, recon, c, device):
    """Fixed-sigma, fixed-seed R = |D - x|^2 / |sigma n|^2 on fixed events (posterior mean)."""
    sigmas = [float(s) for s in c["attr"]["probe_sigmas"]]
    ratios = {s: [] for s in sigmas}
    for eid, x_np, inp in items:
        F_t = recon(eid, posterior=False)
        geo = geo_tensors(inp["bundle"], device)
        ctx = ctx_from(F_t, inp["lookup"], device)
        g = model.encoder(F_t)
        x = torch.as_tensor(x_np, device=device)
        den = denoiser(model, geo, g, ctx, c["edm"])
        for s in sigmas:
            gen = torch.Generator(device=device).manual_seed(int(1e6 * s) + 17)
            noise = torch.randn(x.shape, device=device, generator=gen)
            sig = torch.full((1, 1), s, device=device)
            pred = den(x + sig * noise, sig)
            ratios[s].append(float((pred - x).square().mean())
                             / max(float((sig * noise).square().mean()), 1e-30))
    out = {f"R@{s:g}": float(np.mean(v)) for s, v in ratios.items()}
    out["R_mean"] = float(np.mean([np.mean(v) for v in ratios.values()]))
    return out


@torch.no_grad()
def sample_attr(model, ijk, F_t, c, codec, gen, steps=None):
    """Generate (q, off) for the voxels `ijk` with the Heun sampler; returns (q, off, 48^3 lookup)."""
    device = F_t.device
    inp = fine_inputs(ijk, c)
    geo = geo_tensors(inp["bundle"], device)
    ctx = ctx_from(F_t, inp["lookup"], device)
    g = model.encoder(F_t)
    den = denoiser(model, geo, g, ctx, c["edm"])
    x = heun_sample(den, (len(ijk), N_CH), c["edm"], device,
                    steps=int(steps or c["sample"]["steps"]), generator=gen)
    q, off = codec.decode(x.float().cpu().numpy())
    return q.astype(np.float64), off.astype(np.float64), inp["lookup"]


def t2_rescale(q, lookup, F_np, c, stats):
    """Per 48^3 cell, scale the photons so they sum to the decoded cell sum.

    Cells without a positive decoded sum (or without photons) are left alone.
    Returns (q, per-cell scale factors actually applied).
    """
    base = int(c["field"]["base_grid"])
    occupied = F_np[0].reshape(-1) > float(c["field"]["occ_threshold"])
    target = photon_sums(F_np[2].reshape(-1), occupied, stats, float(c["data"]["q_eps"]))
    have = np.bincount(lookup, weights=q, minlength=base ** 3)
    ok = (have > 0) & (target > 0)
    scale = np.ones(base ** 3)
    scale[ok] = target[ok] / have[ok]
    return q * scale[lookup], scale[ok]
