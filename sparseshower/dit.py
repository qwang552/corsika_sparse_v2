"""LatentDiT: unconditional EDM diffusion on the standardized AE latent.

Model 2 of 4. The LatentDiT generates new latents from noise (unconditional
generation), or denoises a partly noised test latent (SDEdit). Its output is
decoded by the FieldAE into a 48^3 field.

    z (C, L, L, L) --patch p--> (L/p)^3 tokens of C*p^3 --Linear--> width
    + fixed 3D sin-cos position embedding
    depth x DiT block (adaLN-Zero: LN -> MHA, LN -> MLP; shift/scale/gate from sigma)
    final adaLN + Linear -> unpatchify

The only conditioning is the noise level; SDEdit is a sampling mode, see
`sample_latents`.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .diffusion import Denoiser, heun_sample
from .nn_blocks import ScalarEmbedding, modulate


def sincos_3d(n, width):
    """(n^3, width) fixed position embedding, one third of the channels per axis."""
    per = (width // 6) * 2
    pos = torch.arange(n, dtype=torch.float64)
    omega = 1.0 / (10000 ** (torch.arange(per // 2, dtype=torch.float64) / max(per // 2, 1)))
    emb1 = torch.cat([torch.sin(pos[:, None] * omega), torch.cos(pos[:, None] * omega)], dim=1)
    k, j, i = torch.meshgrid(torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij")
    out = torch.cat([emb1[k.reshape(-1)], emb1[j.reshape(-1)], emb1[i.reshape(-1)]], dim=1)
    if out.shape[1] < width:
        out = torch.cat([out, torch.zeros(out.shape[0], width - out.shape[1], dtype=out.dtype)], 1)
    return out.float()


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning on the noise level."""
    def __init__(self, width, heads, mlp_ratio):
        super().__init__()
        self.n1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.n2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, width))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond):
        """x (B, T, width), cond (B, width) -> x after attention and MLP."""
        s1, b1, g1, s2, b2, g2 = self.ada(cond)[:, None].chunk(6, dim=-1)
        h = modulate(self.n1(x), b1, s1)
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        return x + g2 * self.mlp(modulate(self.n2(x), b2, s2))


class LatentDiT(nn.Module):
    """Diffusion transformer on the 8 x 12^3 latent (patch 2 -> 216 tokens)."""
    def __init__(self, cfg, channels, size):
        super().__init__()
        self.c, self.size, self.p = int(channels), int(size), int(cfg["patch"])
        if self.size % self.p:
            raise ValueError("latent size must be divisible by dit.patch")
        self.n = self.size // self.p
        width = int(cfg["width"])
        tok = self.c * self.p ** 3
        self.inp = nn.Linear(tok, width)
        self.register_buffer("pos", sincos_3d(self.n, width), persistent=False)
        self.t_emb = ScalarEmbedding(width)
        self.blocks = nn.ModuleList([DiTBlock(width, int(cfg["heads"]), float(cfg["mlp_ratio"]))
                                     for _ in range(int(cfg["depth"]))])
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.out = nn.Linear(width, tok)
        for m in (self.ada[-1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.use_ckpt = False

    def patchify(self, z):
        """(B, C, L, L, L) latent -> (B, tokens, C*p^3) patch tokens."""
        b, c, n, p = z.shape[0], self.c, self.n, self.p
        z = z.reshape(b, c, n, p, n, p, n, p).permute(0, 2, 4, 6, 1, 3, 5, 7)
        return z.reshape(b, n ** 3, c * p ** 3)

    def unpatchify(self, t):
        """Inverse of patchify."""
        b, c, n, p = t.shape[0], self.c, self.n, self.p
        t = t.reshape(b, n, n, n, c, p, p, p).permute(0, 4, 1, 5, 2, 6, 3, 7)
        return t.reshape(b, c, n * p, n * p, n * p)

    def forward(self, z, c_noise):
        """Network output for a noised latent at noise level c_noise."""
        b = z.shape[0]
        cond = self.t_emb(c_noise.reshape(-1).expand(b) if c_noise.numel() == 1
                          else c_noise.reshape(b))
        x = self.inp(self.patchify(z)) + self.pos
        for blk in self.blocks:
            if self.use_ckpt and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, cond, use_reentrant=False)
            else:
                x = blk(x, cond)
        shift, scale = self.ada(cond)[:, None].chunk(2, dim=-1)
        return self.unpatchify(self.out(modulate(self.norm(x), shift, scale)))


def build_dit(c, device):
    """Construct the LatentDiT for the latent shape implied by the config."""
    lat = int(c["field"]["base_grid"]) // (2 ** (len(c["ae"]["widths"]) - 1))
    return LatentDiT(c["dit"], int(c["ae"]["latent_channels"]), lat).to(device)


def dit_denoiser(net, edm_cfg):
    """Wrap the DiT into an EDM denoiser D(x, sigma)."""
    return Denoiser(lambda x, c_noise, **_: net(x, c_noise), float(edm_cfg["sigma_data"]),
                    edm_cfg.get("predict", "edm"))


@torch.no_grad()
def sample_latents(net, c, n, device, generator=None, steps=None, x_init=None, sigma_start=None):
    """Standardized latents.  x_init + sigma_start -> SDEdit, else unconditional."""
    den = dit_denoiser(net, c["edm"])
    shape = (int(n), net.c, net.size, net.size, net.size)
    return heun_sample(den, shape, c["edm"], device, steps=int(steps or c["sample"]["steps"]),
                       generator=generator, x_init=x_init, sigma_start=sigma_start)


def nn_distance(a, b, chunk=256):
    """Euclidean distance from each row of a to its nearest row of b (NumPy, float64)."""
    a = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    b = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    bb = (b * b).sum(1)
    out = np.empty(len(a))
    for s in range(0, len(a), chunk):
        x = a[s:s + chunk]
        d2 = (x * x).sum(1)[:, None] + bb[None] - 2.0 * x @ b.T
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def describe(c):
    """Latent shape, token count and head size implied by the config (for `info`)."""
    lat = int(c["field"]["base_grid"]) // (2 ** (len(c["ae"]["widths"]) - 1))
    n = lat // int(c["dit"]["patch"])
    return dict(latent=[int(c["ae"]["latent_channels"]), lat, lat, lat], tokens=n ** 3,
                token_dim=int(c["ae"]["latent_channels"]) * int(c["dit"]["patch"]) ** 3,
                head_dim=int(c["dit"]["width"]) // int(c["dit"]["heads"]))
