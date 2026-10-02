"""EDM preconditioning, loss and Heun sampler for the two diffusion models.

The two models are LatentDiT (on the 8x12^3 latent) and attr (on the per-voxel
channels). Training draws a noise level sigma, adds noise and trains
D(x + sigma*n, sigma) to return x; sampling runs the Heun ODE solver from
sigma_max down to 0 (or from a smaller sigma for SDEdit).

Inputs are standardized to unit variance (attr by data.Codec, the DiT latent by
latent_stats.json), so sigma_data = 1 and a given sigma means the same relative
disturbance in every channel.

`predict` selects the output parameterisation:
    "edm" : D = c_skip * x + c_out * F(c_in * x)          (Karras et al. 2022)
    "x"   : D = F(c_in * x)                               (direct clean-data)
The second is what large-patch pixel models prefer (Li & He 2025); it is a
config switch so both can be compared on the same data.

`edm_loss_dense` is the batched version for dense tensors (used by the latent
DiT).
"""
from __future__ import annotations

import torch


def scalings(sigma, sigma_data):
    """EDM preconditioning factors c_skip, c_out, c_in, c_noise for noise level sigma."""
    s2, d2 = sigma * sigma, sigma_data * sigma_data
    c_skip = d2 / (s2 + d2)
    c_out = sigma * sigma_data / torch.sqrt(s2 + d2)
    c_in = 1.0 / torch.sqrt(s2 + d2)
    c_noise = torch.log(sigma.clamp_min(1e-12)) / 4.0
    return c_skip, c_out, c_in, c_noise


def loss_weight(sigma, sigma_data):
    """EDM loss weight lambda(sigma), which makes the loss scale equal across sigma."""
    return (sigma * sigma + sigma_data ** 2) / (sigma * sigma_data) ** 2


def sample_sigmas(n, cfg, device, generator=None):
    """Training noise levels: log-normal(p_mean, p_std), optionally mixed with a fine range."""
    p_mean, p_std = float(cfg["p_mean"]), float(cfg["p_std"])
    z = torch.randn(n, device=device, generator=generator)
    sigma = torch.exp(z * p_std + p_mean)
    frac = float(cfg.get("fine_fraction", 0.0))
    if frac > 0:
        lo, hi = [float(v) for v in cfg["fine_range"]]
        u = torch.rand(n, device=device, generator=generator)
        fine = torch.exp(u * (torch.log(torch.tensor(hi, device=device))
                              - torch.log(torch.tensor(lo, device=device)))
                         + torch.log(torch.tensor(lo, device=device)))
        pick = torch.rand(n, device=device, generator=generator) < frac
        sigma = torch.where(pick, fine, sigma)
    return sigma.clamp(float(cfg["sigma_min"]), float(cfg["sigma_max"]))


def karras_schedule(steps, cfg, device, sigma_max=None):
    """Decreasing sigma schedule of Karras et al. (rho spacing), ending with 0."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    rho = float(cfg["rho"])
    s_max = float(cfg["sigma_max"] if sigma_max is None else sigma_max)
    s_min = min(float(cfg["sigma_min"]), s_max)
    t = torch.linspace(0, 1, steps, device=device, dtype=torch.float64)
    s = (s_max ** (1 / rho) + t * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho
    return torch.cat([s, torch.zeros(1, device=device, dtype=torch.float64)]).float()


class Denoiser:
    """Wraps a raw network into D(x, sigma) with EDM preconditioning."""

    def __init__(self, net_fn, sigma_data=1.0, predict="edm"):
        if predict not in ("edm", "x"):
            raise ValueError("edm.predict must be 'edm' or 'x'")
        self.net_fn = net_fn
        self.sigma_data = float(sigma_data)
        self.predict = predict

    def __call__(self, x, sigma, **ctx):
        """D(x, sigma): run the network on the scaled input and combine with the skip term."""
        sigma = sigma.to(torch.float32)
        c_skip, c_out, c_in, c_noise = scalings(sigma, self.sigma_data)
        f = self.net_fn(c_in * x, c_noise, **ctx).float()
        if self.predict == "x":
            return f
        return c_skip * x + c_out * f


def edm_loss(denoiser, x, sigma, noise=None, weights=None, **ctx):
    """Weighted denoising MSE.  `weights` may depend only on the noisy input, not on x, so the optimum is unchanged."""
    noise = torch.randn_like(x) if noise is None else noise
    noisy = x + sigma * noise
    pred = denoiser(noisy, sigma, **ctx)
    err = (pred - x).square().mean(dim=-1)
    if weights is not None:
        err = err * weights / weights.mean().clamp_min(1e-6)
    return (err.mean() * loss_weight(sigma, denoiser.sigma_data)).mean(), pred


def edm_loss_dense(denoiser, x, sigma, noise=None, **ctx):
    """Batched EDM loss for dense tensors: x (B, ...), sigma (B,)."""
    noise = torch.randn_like(x) if noise is None else noise
    s = sigma.reshape(-1, *([1] * (x.dim() - 1)))
    pred = denoiser(x + s * noise, s, **ctx)
    err = (pred - x).square().flatten(1).mean(dim=1)
    return (err * loss_weight(sigma.reshape(-1), denoiser.sigma_data)).mean(), pred


@torch.no_grad()
def heun_sample(denoiser, shape, cfg, device, steps=32, generator=None, dtype=torch.float32,
                x_init=None, sigma_start=None, **ctx):
    """Deterministic Heun sampler on the Karras schedule.

    SDEdit: with `x_init` (a clean sample) and `sigma_start`, the chain starts
    from x_init + sigma_start * noise and runs the schedule sigma_start -> 0.
    sigma_start = sigma_max with any x_init is the unconditional sampler up to
    an x_init / sigma_max shift of the starting point.
    """
    if x_init is None:
        sig = karras_schedule(steps, cfg, device)
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator) * sig[0]
    else:
        if sigma_start is None:
            raise ValueError("x_init needs sigma_start")
        sig = karras_schedule(steps, cfg, device, sigma_max=float(sigma_start))
        x = x_init.to(dtype) + torch.randn(x_init.shape, device=device, dtype=dtype,
                                           generator=generator) * sig[0]
    for i in range(len(sig) - 1):
        s_cur, s_next = sig[i], sig[i + 1]
        d = denoiser(x, s_cur.reshape(1), **ctx)
        v = (x - d) / s_cur
        x_next = x + (s_next - s_cur) * v
        if s_next > 0:
            d2 = denoiser(x_next, s_next.reshape(1), **ctx)
            v2 = (x_next - d2) / s_next
            x_next = x + (s_next - s_cur) * 0.5 * (v + v2)
        x = x_next
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"non-finite sampler state at step {i}")
    return x
