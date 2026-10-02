"""Structure model: grow 48^3 -> 96^3 -> 192^3 from a decoded field.

Model 3 of 4 (torch side). Training (`plan_step` + `struct_step`) teaches the
network the 256-way child configuration of every parent, one level at a time,
with the field context taken from the AE-decoded field. Sampling
(`grow_structure`) starts from the occupied 48^3 cells of a decoded field and
grows the active set level by level, colour by colour.

Per parent and level the network sees (95 channels + Fourier positions):
    ctx (3)        decoded field F = (p_occ, ch0, ch1) at the 48^3 ancestor
    occupancy (1)  fraction of occupied face neighbours (message-passing table)
    shape (9)      direction tensor / linearity of the 5^3 neighbourhood
    nbr (56 + 26)  children of the 26 neighbours that touch this parent, and
                   which neighbours are already known (earlier colours)
and a global vector g = FieldEncoder(F) plus the level index in the adaLN
conditioning.  Output: 256 logits, one per joint child configuration.

Training on the reconstructed field (same input distribution as at sampling):
* ctx and g always come from the AE decode of the event's latent
  (posterior mean, or a posterior sample with prob. posterior_p);
* level 0 (48 -> 96) uses the decoded occupied set {p_occ > thr} as parents
  with prob. recon_parent_p: the same start set as at sampling.  Truth cells the
  decode missed cannot be grown (their children are counted in `lost_children`); extra
  decoded cells get target config 0.  Otherwise the truth 48^3 set, corrupted
  with prob. aug_p;
* level 1 (96 -> 192): truth 96^3 set, corrupted with prob. aug_p.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .geometry import neighbour_offsets
from .nn_blocks import SparseNet
from .samples import bits_for_parents, level_grids, level_sets
from .struct_data import (CODE_BITS, N_CONFIG, N_IN, SLOTS, bits_to_config, colour_of,
                          colour_step, expand_children, level_inputs, level_prior_logits,
                          perturb_parents, track_cut)


# ------------------------------------------------------------------ model ---
class FieldEncoder(nn.Module):
    """Decoded field (3, B, B, B) -> one global conditioning vector g."""

    def __init__(self, out_dim, in_channels=3, channels=(32, 64, 128)):
        super().__init__()
        layers, cin = [], in_channels
        for cout in channels:
            layers += [nn.Conv3d(cin, cout, 3, stride=2, padding=1),
                       nn.GroupNorm(min(8, cout), cout), nn.SiLU()]
            cin = cout
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(2 * cin, out_dim)

    def forward(self, field):
        """(3, B, B, B) or (N, 3, B, B, B) field -> (N, out_dim) global vector."""
        x = field if field.dim() == 5 else field[None]
        h = self.net(x)
        return self.out(torch.cat([h.mean(dim=(2, 3, 4)), h.amax(dim=(2, 3, 4))], dim=1))


def sparse_cfg(c, section):
    """SparseNet settings: `model` section with per-model overrides from `struct` / `attr`."""
    cfg = {k: c["model"][k] for k in ("width", "depth", "heads", "coarse_every",
                                      "pos_freqs", "pos_max_freq")}
    for k in ("width", "depth", "heads", "coarse_every"):
        if c[section].get(k) is not None:
            cfg[k] = c[section][k]
    cfg["checkpoint"] = c["model"].get("checkpoint") is True
    return cfg


class StructNet(nn.Module):
    """Structure network: FieldEncoder (for g) + SparseNet with 256 output logits per parent."""
    def __init__(self, c):
        super().__init__()
        s = c["struct"]
        self.encoder = FieldEncoder(int(s["g_dim"]))
        n_off = len(neighbour_offsets(int(c["model"]["neighbours"])))
        self.net = SparseNet(N_IN, N_CONFIG, int(s["g_dim"]), sparse_cfg(c, "struct"), n_off)
        with torch.no_grad():
            self.net.head[-1].bias.copy_(torch.as_tensor(level_prior_logits(s["child_prior"],
                                                                           s["config0_prior"])))
        self.register_buffer("slot_k", torch.as_tensor(SLOTS[:, 0]), persistent=False)
        self.register_buffer("slot_code", torch.as_tensor(SLOTS[:, 1]), persistent=False)
        self.register_buffer("code_bits", torch.as_tensor(CODE_BITS), persistent=False)

    def nbr_features(self, t, known, bits_known):
        """(N, 56 + 26) known children of the 26 neighbours + known flags."""
        idx, mask = t["nbr26_idx"], t["nbr26_mask"]
        flag = mask * known[idx]                                   # (N, 26)
        nb_bits = bits_known[idx[:, self.slot_k]]                  # (N, 56, 8)
        slot = torch.gather(nb_bits, 2, self.slot_code[None, :, None].expand(idx.shape[0], -1, 1))[..., 0]
        return torch.cat([slot * flag[:, self.slot_k], flag], dim=1)

    def forward(self, t, ctx, level, g, known=None, bits_known=None):
        """Logits (N, 256) of the child configuration of every parent at one level."""
        n = ctx.shape[0]
        if known is None:
            known = torch.zeros(n, device=ctx.device)
            bits_known = torch.zeros(n, 8, device=ctx.device)
        occ = t["geo"]["nbr_mask"].mean(dim=1, keepdim=True)
        feats = torch.cat([ctx, occ.to(ctx.dtype), t["shape"].to(ctx.dtype),
                           self.nbr_features(t, known.float(), bits_known.float()).to(ctx.dtype)],
                          dim=-1)
        scalar = torch.as_tensor([float(level)], device=ctx.device)
        return self.net(feats, t["geo"], scalar, g)


def build_struct(c, device):
    """Construct the structure network."""
    return StructNet(c).to(device)


# ---------------------------------------------------------------- tensors ---
def geo_tensors(bundle, device):
    """Move a NumPy geometry bundle to torch tensors on `device`."""
    keys = ("pos", "nbr_idx", "nbr_mask", "win_idx_a", "win_valid_a", "win_idx_b",
            "win_valid_b", "pool_parent", "pool_pos")
    out = {}
    for key in keys:
        v = bundle[key]
        if v.dtype == np.bool_:
            out[key] = torch.as_tensor(v, device=device)
        elif np.issubdtype(v.dtype, np.integer):
            out[key] = torch.as_tensor(v.astype(np.int64), device=device)
        else:
            out[key] = torch.as_tensor(v.astype(np.float32), device=device)
    out["n_pool"] = int(bundle["n_pool"])
    return out


def level_tensors(inp, F_t, device):
    """Torch inputs of one level: geometry, shape features, 26-neighbour table, field context."""
    t = dict(geo=geo_tensors(inp["bundle"], device),
             shape=torch.as_tensor(inp["shape"], device=device),
             nbr26_idx=torch.as_tensor(inp["nbr26_idx"], device=device),
             nbr26_mask=torch.as_tensor(inp["nbr26_mask"], device=device))
    ctx = F_t.reshape(3, -1)[:, torch.as_tensor(inp["lookup"], device=device)].transpose(0, 1)
    return t, ctx.contiguous()


# ------------------------------------------------------------------- loss ---
def config_loss(model, logits, target, mask, label_smoothing=0.0):
    """Cross-entropy over the 256 configurations on masked parents, plus recall / precision stats.
    """
    lg, tg = logits.float()[mask], target[mask]
    if lg.shape[0] == 0:
        return logits.float().sum() * 0.0, {}
    loss = F.cross_entropy(lg, tg, label_smoothing=float(label_smoothing))
    with torch.no_grad():
        prob = torch.softmax(lg, dim=-1)
        marg = prob @ model.code_bits
        bits = model.code_bits[tg]
        pred = marg > 0.5
        tp = float((pred & (bits > 0.5)).sum())
        stats = dict(ce=float(loss), joint_acc=float((lg.argmax(-1) == tg).float().mean()),
                     recall=tp / max(float(bits.sum()), 1.0),
                     precision=tp / max(float(pred.sum()), 1.0),
                     expected_ratio=float(marg.sum()) / max(float(bits.sum()), 1.0),
                     p_config0=float(prob[:, 0].mean()))
    return loss, stats


# ------------------------------------------------------------ training data ---
def truth_levels(ev, c):
    """Truth sets on [48, 96, 192] (key-sorted)."""
    grids = level_grids(int(c["data"]["grid"]), int(c["field"]["base_grid"]))
    return grids, level_sets(ev["ijk"], grids)


def plan_step(eid, ev, c, rng, level=None):
    """CPU part of one struct training step (runs on the prefetch thread).

    Everything that does not depend on the decoded field is decided and built
    here; `struct_step` finishes level 0 when the parents come from the decode.
    """
    s = c["struct"]
    grids, sets = truth_levels(ev, c)
    n_levels = len(grids) - 1
    level = int(rng.integers(n_levels)) if level is None else int(level)
    plan = dict(eid=int(eid), level=level, grid=int(grids[level]),
                child_set=sets[level + 1], posterior=bool(rng.random() < float(s["posterior_p"])),
                colour_k=int(rng.integers(max(1, int(s["colours"])))),
                cut=bool(rng.random() < float(s["cut_p"])), n_active=int(len(ev["ijk"])),
                seed=int(rng.integers(2 ** 31)))
    plan["recon_parent"] = level == 0 and bool(rng.random() < float(s["recon_parent_p"]))
    if not plan["recon_parent"]:
        parent = sets[level]
        bits, _, _ = bits_for_parents(parent, plan["child_set"], grids[level + 1])
        if rng.random() < float(s["aug_p"]):
            parent, bits = perturb_parents(parent, bits, grids[level], rng,
                                           float(s["drop_frac"]), float(s["add_frac"]))
        plan["parent"], plan["bits"] = parent, bits
        plan["inp"] = level_inputs(parent, grids[level], c, int(s["shape_radius"]))
    return plan


def struct_step(model, plan, F_t, c, device, train=True):
    """GPU part: finish the parent set if it comes from the decode, then CE."""
    s = c["struct"]
    rng = np.random.default_rng(plan["seed"])
    if plan["recon_parent"]:
        from .fields import occupied_cells
        parent = occupied_cells(F_t[0].detach().cpu().numpy(), float(c["field"]["occ_threshold"]))
        bits, lost, _ = bits_for_parents(parent, plan["child_set"], 2 * plan["grid"])
        inp = level_inputs(parent, plan["grid"], c, int(s["shape_radius"]))
    else:
        parent, bits, inp, lost = plan["parent"], plan["bits"], plan["inp"], 0
    if len(parent) == 0:
        return None, dict(empty=True)
    t, ctx = level_tensors(inp, F_t, device)
    target = torch.as_tensor(bits_to_config(bits), device=device)
    known_np, mask_np = colour_step(parent, int(s["colours"]), plan["colour_k"] if train else 0)
    bits_known_np = bits * known_np[:, None]
    if train and plan["cut"] and known_np.any():
        bits_known_np = track_cut(parent, plan["grid"], known_np, bits_known_np, inp["shape"], rng,
                                  float(s["cut_frac"]), tuple(s["cut_len"]))
    known = torch.as_tensor(known_np, device=device)
    bits_known = torch.as_tensor(bits_known_np, device=device, dtype=torch.float32)
    mask = torch.as_tensor(mask_np, device=device)
    g = model.encoder(F_t)
    logits = model(t, ctx, plan["level"], g, known, bits_known)
    loss, stats = config_loss(model, logits, target, mask, s["label_smoothing"])
    stats.update(level=plan["level"], colour_k=plan["colour_k"] if train else 0,
                 recon_parent=bool(plan["recon_parent"]), lost_children=int(lost),
                 n_parents=int(len(parent)))
    return loss, stats


# --------------------------------------------------------------- sampling ---
def _draw(logits, temperature, gen):
    """Sample one configuration per parent from the logits at a given temperature."""
    prob = torch.softmax(logits.float() / max(float(temperature), 1e-4), dim=-1)
    return torch.multinomial(prob, 1, generator=gen)[:, 0]


@torch.no_grad()
def sample_level(model, parent, grid, level, F_t, g, c, gen, temperature=None):
    """Children of one parent set, colour by colour."""
    s = c["struct"]
    device = F_t.device
    temperature = s["temperature"] if temperature is None else temperature
    inp = level_inputs(parent, grid, c, int(s["shape_radius"]))
    t, ctx = level_tensors(inp, F_t, device)
    col = torch.as_tensor(colour_of(inp["parent"], int(s["colours"])), device=device)
    n = len(inp["parent"])
    config = torch.zeros(n, dtype=torch.int64, device=device)
    known = torch.zeros(n, device=device)
    for k in range(max(1, int(s["colours"]))):
        sel = col == k
        if not bool(sel.any()):
            continue
        bits_known = model.code_bits[config] * known[:, None]
        logits = model(t, ctx, level, g, known, bits_known)
        draw = _draw(logits[sel], temperature, gen)
        config[sel] = draw
        known[sel] = 1.0
    cfg = config.cpu().numpy()
    return expand_children(inp["parent"], CODE_BITS[cfg]), cfg


@torch.no_grad()
def grow_structure(model, F_t, start_ijk, c, gen, temperature=None):
    """48^3 start set -> 192^3 active voxels.  Returns (ijk or None, counts)."""
    grids = level_grids(int(c["data"]["grid"]), int(c["field"]["base_grid"]))
    max_active = int(c["sample"]["max_active"])
    g = model.encoder(F_t)
    ijk = np.asarray(start_ijk, dtype=np.int64).reshape(-1, 3)
    counts = [int(len(ijk))]
    if len(ijk) == 0 or len(ijk) > int(c["sample"].get("max_start", len(ijk))):
        return None, counts                 # empty or exploded start set: recorded as a failed sample
    for level, grid in enumerate(grids[:-1]):
        ijk, _ = sample_level(model, ijk, grid, level, F_t, g, c, gen, temperature)
        counts.append(int(len(ijk)))
        if len(ijk) == 0 or len(ijk) > max_active:
            return None, counts
    return ijk, counts


# ------------------------------------------------------------------ probe ---
@torch.no_grad()
def probe(model, events, recon, c, device):
    """Teacher-forced CE on fixed events: every level, colour step 0 (no known
    neighbours) and the last colour step (all others known); decoded parents
    at level 0.  Posterior mean, no corruption."""
    rows = []
    s = c["struct"]
    last = max(1, int(s["colours"])) - 1
    for eid, ev in events:
        F_t = recon(eid, posterior=False)
        grids, _ = truth_levels(ev, c)
        for level in range(len(grids) - 1):
            for k in sorted({0, last}):
                rng = np.random.default_rng(0)
                plan = plan_step(eid, ev, dict(c, struct=dict(s, recon_parent_p=1.0, aug_p=0.0,
                                                              posterior_p=0.0, cut_p=0.0)),
                                 rng, level=level)
                plan["colour_k"] = k
                _, st = struct_step(model, plan, F_t, c, device, train=True)
                if st and "ce" in st:
                    rows.append(dict(st, step_k=k))
    out = {}
    for k, tag in ((0, "k0"), (last, "klast")):
        sel = [r for r in rows if r["step_k"] == k]
        if sel:
            for key in ("ce", "recall", "precision", "expected_ratio", "p_config0"):
                out[f"{tag}_{key}"] = float(np.mean([r[key] for r in sel]))
    out["lost_children"] = float(np.mean([r["lost_children"] for r in rows])) if rows else 0.0
    out["R_mean"] = float(np.mean([v for k, v in out.items() if k.endswith("_ce")])) if rows else float("inf")
    return out
