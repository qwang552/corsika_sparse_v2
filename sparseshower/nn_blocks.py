"""Torch building blocks: Fourier position features and a sparse voxel stack.

Role in the pipeline: `SparseNet` is the backbone of both struct and attr. It
works on a list of active voxels (not a dense grid): each voxel gets its
features plus Fourier features of its position, then passes through blocks of
(neighbour message passing -> windowed attention -> MLP), with a pooled global
attention branch every few blocks. Conditioning (noise level, level index and
the global vector g) enters through adaLN modulation.

Design rules:

* positions enter through multi-frequency Fourier features, never as raw
  coordinates (spectral bias: Rahaman 2019, Tancik 2020);
* neighbourhoods come from the *known* voxel structure and are passed in as
  precomputed index tables, so they never depend on the noise level.

Shapes (N = active voxels, K = neighbours, C = width, W = attention windows,
S = window size, M = pooled tokens):
    h          (N, C)
    nbr_idx    (N, K) int64       nbr_mask (N, K) float32
    win_idx_*  (W, S) int64       win_valid_* (W, S) bool
    pool_parent(N,)   int64       pool_pos (M, 3) float32
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class FourierFeatures(nn.Module):
    """[sin(pi f x), cos(pi f x)] over log-spaced frequencies, per axis."""

    def __init__(self, n_freq=7, max_freq=64.0, dims=3):
        super().__init__()
        if n_freq < 1:
            raise ValueError("n_freq must be >= 1")
        freqs = torch.exp(torch.linspace(0.0, math.log(float(max_freq)), n_freq))
        self.register_buffer("freqs", freqs, persistent=False)
        self.dims = int(dims)

    @property
    def out_dim(self):
        return self.dims * 2 * len(self.freqs)

    def forward(self, pos):
        """(..., dims) positions in [-1, 1] -> (..., dims * 2F) sin/cos features."""
        a = pos[..., None] * self.freqs * math.pi           # (..., dims, F)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1).flatten(-2)


class ScalarEmbedding(nn.Module):
    """Sinusoidal embedding of one scalar per sample (log sigma, level index)."""

    def __init__(self, width, n_freq=32):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(1000.0), n_freq)),
                             persistent=False)
        self.net = nn.Sequential(nn.Linear(2 * n_freq, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, value):
        """(B,) scalars -> (B, width) embedding."""
        t = value.reshape(-1, 1) * self.freqs[None]
        return self.net(torch.cat([torch.sin(t), torch.cos(t)], dim=-1))


class Conditioner(nn.Module):
    """cond = MLP(noise-level embedding + embedding of the global vector g, passed as `macro`)."""

    def __init__(self, macro_dim, width):
        super().__init__()
        self.scalar = ScalarEmbedding(width)
        self.macro = nn.Sequential(nn.Linear(macro_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(width, width))

    def forward(self, scalar, macro):
        """Conditioning vector from a scalar (noise level) and a global vector (here g)."""
        # macro may be (D,) for a single event or (B, D) for a dense batch.
        m = macro if macro.dim() == 2 else macro.reshape(1, -1)
        return self.out(self.scalar(scalar) + self.macro(m))


def modulate(x, shift, scale):
    """adaLN modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale) + shift


class NeighbourMessage(nn.Module):
    """Exact grid-neighbour message passing (a small sparse convolution).

    The neighbour direction enters as a learned per-offset bias on the hidden
    layer, so no (N, K, C) tensor of concatenated direction embeddings is ever
    built.  This keeps memory low on the largest events.
    """

    def __init__(self, width, n_offsets, hidden=None):
        super().__init__()
        hidden = hidden or max(32, width // 2)
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Linear(width, hidden)
        self.off_bias = nn.Parameter(torch.zeros(n_offsets, hidden))
        self.out = nn.Linear(hidden, width)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, nbr_idx, nbr_mask):
        """Add the gated mean message from each voxel's valid grid neighbours."""
        # proj is linear, so proj(z[j] - z[i]) = W z[j] - W z[i] + b: project
        # once per voxel (N, H) and gather the projection instead of projecting
        # every edge of an (N, K, C) tensor.
        z = self.norm(h)                                     # (N, C)
        p = F.linear(z, self.proj.weight)                    # (N, H), no bias
        hidden = p[nbr_idx] - p[:, None, :]                  # (N, K, H)
        hidden = hidden + (self.proj.bias + self.off_bias).to(hidden.dtype)  # + (K, H)
        nbr_mask = nbr_mask.to(hidden.dtype)
        msg = self.out(F.silu(hidden)) * nbr_mask[..., None]
        denom = nbr_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return h + torch.tanh(self.gate) * msg.sum(dim=1) / denom


class WindowAttention(nn.Module):
    """Self-attention inside fixed windows of a serialized voxel order."""

    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, v, win_idx, win_valid):
        """Attention among the voxels of each window; returns (N, C) in the original order."""
        n, width = v.shape
        tok = v[win_idx]                                     # (W, S, C)
        w, s, _ = tok.shape
        qkv = self.qkv(tok).view(w, s, 3, self.heads, self.dim)
        q, k, val = qkv.permute(2, 0, 3, 1, 4).unbind(0)      # (W, H, S, D)
        mask = win_valid[:, None, None, :]                   # (W, 1, 1, S)
        out = F.scaled_dot_product_attention(q, k, val, attn_mask=mask)
        out = out.permute(0, 2, 1, 3).reshape(w, s, width)
        flat = torch.zeros(n, width, dtype=out.dtype, device=out.device)
        flat = flat.index_copy(0, win_idx[win_valid], out[win_valid])
        return self.proj(flat)


class SparseBlock(nn.Module):
    """One block: neighbour message, then windowed attention and MLP with adaLN modulation."""
    def __init__(self, width, heads, cond_width, n_offsets):
        super().__init__()
        self.message = NeighbourMessage(width, n_offsets)
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = WindowAttention(width, heads)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.ff = nn.Sequential(nn.Linear(width, 3 * width), nn.GELU(), nn.Linear(3 * width, width))
        self.ada = nn.Linear(cond_width, 6 * width)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        with torch.no_grad():                                 # small non-zero residual gates
            self.ada.bias[2 * width:3 * width].fill_(0.1)
            self.ada.bias[5 * width:6 * width].fill_(0.1)

    def forward(self, h, geo, cond, variant):
        """Run the block; even / odd blocks use windows from two different Z-order curves."""
        h = self.message(h, geo["nbr_idx"], geo["nbr_mask"])
        s1, b1, g1, s2, b2, g2 = self.ada(cond).chunk(6, dim=-1)
        v = modulate(self.norm1(h), b1, s1)
        if variant % 2 == 0:
            out = self.attn(v, geo["win_idx_a"], geo["win_valid_a"])
        else:
            out = self.attn(v, geo["win_idx_b"], geo["win_valid_b"])
        h = h + g1 * out
        return h + g2 * self.ff(modulate(self.norm2(h), b2, s2))


class CoarseBranch(nn.Module):
    """Pool to <= a few hundred tokens, attend globally, broadcast back."""

    def __init__(self, width, heads, cond_width, pos_dim):
        super().__init__()
        self.pos = nn.Linear(pos_dim, width)
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ada = nn.Linear(cond_width, 3 * width)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        with torch.no_grad():
            self.ada.bias[2 * width:3 * width].fill_(0.1)

    def forward(self, h, geo, cond, pos_feats):
        """Average voxels into pooled tokens, attend over all tokens, add the result back."""
        parent = geo["pool_parent"]
        m = int(geo["n_pool"])
        counts = torch.zeros(m, 1, dtype=h.dtype, device=h.device)
        counts = counts.index_add(0, parent, torch.ones_like(h[:, :1]))
        pooled = torch.zeros(m, h.shape[1], dtype=h.dtype, device=h.device)
        pooled = pooled.index_add(0, parent, h) / counts.clamp_min(1.0)
        s, b, g = self.ada(cond).chunk(3, dim=-1)
        tok = modulate(self.norm(pooled + self.pos(pos_feats)), b, s)[None]
        out, _ = self.attn(tok, tok, tok, need_weights=False)
        return h + g * out[0][parent]


class SparseNet(nn.Module):
    """Stack of sparse blocks with a periodic global branch."""

    def __init__(self, in_channels, out_channels, macro_dim, cfg, n_offsets):
        super().__init__()
        width = int(cfg["width"])
        depth = int(cfg["depth"])
        heads = int(cfg["heads"])
        self.fourier = FourierFeatures(int(cfg["pos_freqs"]), float(cfg["pos_max_freq"]))
        self.input = nn.Linear(in_channels + self.fourier.out_dim, width)
        self.cond = Conditioner(macro_dim, width)
        self.blocks = nn.ModuleList([SparseBlock(width, heads, width, n_offsets)
                                     for _ in range(depth)])
        self.coarse_every = max(1, int(cfg["coarse_every"]))
        n_coarse = depth // self.coarse_every
        self.coarse = nn.ModuleList([CoarseBranch(width, heads, width, self.fourier.out_dim)
                                     for _ in range(n_coarse)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                  nn.Linear(width, out_channels))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        # Activation checkpointing: recompute block activations in the backward
        # pass instead of storing them (about a third more compute, much less
        # memory).  "auto" is resolved by the caller that knows the device
        # (train.resolve_checkpoint); only a real True switches it on here.
        value = cfg.get("checkpoint", False)
        self.checkpoint = value is True

    def _run(self, module, *args):
        """Call a block, with activation checkpointing during training when enabled."""
        if self.checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, feats, geo, scalar, macro):
        """(N, in) voxel features + geometry + conditioning -> (N, out) outputs."""
        cond = self.cond(scalar, macro)                      # (1, C)
        pos_feats = self.fourier(geo["pos"])
        h = self.input(torch.cat([feats, pos_feats], dim=-1))
        pool_feats = self.fourier(geo["pool_pos"])
        c_index = 0
        for i, block in enumerate(self.blocks):
            h = self._run(block, h, geo, cond, i)
            if (i + 1) % self.coarse_every == 0 and c_index < len(self.coarse):
                h = self._run(self.coarse[c_index], h, geo, cond, pool_feats)
                c_index += 1
        return self.head(h)
