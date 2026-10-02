"""Structure model: 26-neighbour slots, 8-colour ordering, track cut, and the no-leak rule."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from sparseshower import geometry as G
from sparseshower.common import config
from sparseshower.struct_data import (CODE_BITS, N_IN, SLOTS, bits_to_config, colour_of, colour_step,
                                      config_to_bits, expand_children, level_prior_logits, perturb_parents,
                                      shape_features, track_cut)

try:
    import torch
except ImportError:                     # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def blob(rng, grid=16, n=300):
    ijk = np.unique(rng.integers(0, grid, size=(n, 3)), axis=0)
    return ijk[np.argsort(G.pack(ijk, grid), kind="stable")]


class TestSlots(unittest.TestCase):
    def test_56_slots_are_exactly_the_touching_children(self):
        self.assertEqual(len(SLOTS), 56)
        self.assertEqual(N_IN, 95)
        own = {(cx, cy, cz) for cx in (0, 1) for cy in (0, 1) for cz in (0, 1)}     # children of parent 0
        want = set()
        for k, o in enumerate(G.OFFSETS_26):
            for code in range(8):
                ch = 2 * np.asarray(o) + np.array([code % 2, (code // 2) % 2, code // 4])
                if any(max(abs(ch - np.array(c))) == 1 for c in own):
                    want.add((k, code))
        self.assertEqual({tuple(s) for s in SLOTS.tolist()}, want)

    def test_config_bits_roundtrip(self):
        self.assertTrue(np.array_equal(bits_to_config(config_to_bits(np.arange(256))), np.arange(256)))
        self.assertEqual(CODE_BITS.shape, (256, 8))

    def test_prior_logits(self):
        lp = level_prior_logits(0.35, 0.02)
        self.assertEqual(lp.shape, (256,))
        self.assertAlmostEqual(float(lp.max()), 0.0)


class TestColours(unittest.TestCase):
    def test_same_colour_parents_never_touch(self):
        rng = np.random.default_rng(0)
        ijk = blob(rng)
        col = colour_of(ijk, 8)
        idx, mask = G.neighbour_table(ijk, 16, G.OFFSETS_26)
        rows, ks = np.nonzero(mask > 0)
        self.assertTrue(len(rows) > 0)
        self.assertTrue((col[rows] != col[idx[rows, ks]]).all())

    def test_two_colours_separate_face_neighbours(self):
        rng = np.random.default_rng(1)
        ijk = blob(rng)
        col = colour_of(ijk, 2)
        idx, mask = G.neighbour_table(ijk, 16, G.OFFSETS_6)
        rows, ks = np.nonzero(mask > 0)
        self.assertTrue((col[rows] != col[idx[rows, ks]]).all())

    def test_colour_step_views(self):
        ijk = blob(np.random.default_rng(2))
        for k in range(8):
            known, loss = colour_step(ijk, 8, k)
            col = colour_of(ijk, 8)
            self.assertTrue(np.array_equal(known > 0.5, col < k))
            self.assertTrue(np.array_equal(loss, col >= k))


class TestAugmentation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        z = np.arange(2, 14)
        line = np.stack([np.full(len(z), 8), np.full(len(z), 8), z], 1)
        self.parent = np.unique(np.concatenate([line, blob(rng, n=40)]), axis=0)
        self.parent = self.parent[np.argsort(G.pack(self.parent, 16), kind="stable")]
        self.bits = (rng.random((len(self.parent), 8)) < 0.4).astype(np.float32)
        self.shape = shape_features(self.parent, 16, 2)

    def test_track_cut_only_touches_known_rows(self):
        rng = np.random.default_rng(4)
        known = (rng.random(len(self.parent)) < 0.6).astype(np.float32)
        bk = self.bits * known[:, None]
        before = bk.copy()
        out = track_cut(self.parent, 16, known, bk, self.shape, rng, frac=0.5, length=(1, 3), lin_min=0.5)
        self.assertTrue(np.array_equal(bk, before))                          # input not modified
        changed = np.any(out != bk, axis=1)
        self.assertTrue((known[changed] > 0.5).all())
        self.assertTrue((out[changed] == 0).all())
        same = track_cut(self.parent, 16, known, bk, self.shape, rng, frac=0.0)
        self.assertTrue(np.array_equal(same, bk))

    def test_perturb_parents(self):
        rng = np.random.default_rng(5)
        p, b = perturb_parents(self.parent, self.bits, 16, rng, 0.2, 0.2)
        keys = G.pack(p, 16)
        self.assertTrue((np.diff(keys) > 0).all())                           # sorted, unique
        orig = set(G.pack(self.parent, 16).tolist())
        added = np.array([k not in orig for k in keys.tolist()])
        self.assertTrue((b[added] == 0).all())

    def test_expand_children_is_sorted(self):
        ch = expand_children(self.parent, self.bits)
        self.assertEqual(len(ch), int(self.bits.sum()))
        self.assertTrue((np.diff(G.pack(ch, 32)) > 0).all())


@unittest.skipIf(torch is None, "torch not installed")
class TestStructNet(unittest.TestCase):
    def setUp(self):
        self.c = config(ROOT / "configs" / "smoke.yaml")
        from sparseshower.structure import build_struct
        torch.manual_seed(0)
        self.net = build_struct(self.c, torch.device("cpu"))

    def test_unknown_neighbours_do_not_leak(self):
        """Features of a parent may only see the child bits of KNOWN neighbours."""
        rng = np.random.default_rng(6)
        ijk = blob(rng)
        idx, mask = G.neighbour_table(ijk, 16, G.OFFSETS_26)
        t = dict(nbr26_idx=torch.as_tensor(idx), nbr26_mask=torch.as_tensor(mask, dtype=torch.float32))
        known, _ = colour_step(ijk, 8, 3)
        known = torch.as_tensor(known)
        bits = torch.as_tensor((rng.random((len(ijk), 8)) < 0.5).astype(np.float32)) * known[:, None]
        f1 = self.net.nbr_features(t, known, bits)
        noise = torch.as_tensor((rng.random((len(ijk), 8)) < 0.5).astype(np.float32)) * (1 - known)[:, None]
        f2 = self.net.nbr_features(t, known, bits + noise)
        self.assertTrue(torch.equal(f1, f2))
        self.assertEqual(f1.shape, (len(ijk), 56 + 26))

    def test_slot_feature_is_the_neighbours_child_bit(self):
        ijk = np.array([[4, 4, 4], [5, 4, 4]])                                # +x face neighbours
        idx, mask = G.neighbour_table(ijk, 16, G.OFFSETS_26)
        t = dict(nbr26_idx=torch.as_tensor(idx), nbr26_mask=torch.as_tensor(mask, dtype=torch.float32))
        known = torch.tensor([0.0, 1.0])
        bits = torch.zeros(2, 8)
        bits[1, 0] = 1.0                                                     # child (0,0,0) of the +x parent
        f = self.net.nbr_features(t, known, bits)
        k = [i for i, o in enumerate(G.OFFSETS_26) if tuple(o) == (1, 0, 0)][0]
        hit = [n for n, (kk, code) in enumerate(SLOTS.tolist()) if kk == k and code == 0][0]
        self.assertEqual(float(f[0, hit]), 1.0)
        self.assertEqual(float(f[0, :56].sum()), 1.0)
        self.assertEqual(float(f[1, :56].sum()), 0.0)                       # its neighbour is unknown

    def test_grow_structure_stays_in_the_box(self):
        from sparseshower.structure import grow_structure
        B = int(self.c["field"]["base_grid"])
        F = torch.zeros(3, B, B, B)
        F[0, 2:5, 3:5, 3:5] = 1.0
        start = np.argwhere(F[0].numpy().transpose(2, 1, 0) > 0.5)
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            ijk, counts = grow_structure(self.net, F, start, self.c, gen)
        self.assertEqual(counts[0], len(start))
        if ijk is not None:
            g = int(self.c["data"]["grid"])
            self.assertTrue(((ijk >= 0) & (ijk < g)).all())
            anc = ijk // (g // B)
            self.assertTrue(set(map(tuple, anc.tolist())) <= set(map(tuple, start.tolist())))


if __name__ == "__main__":
    unittest.main()
