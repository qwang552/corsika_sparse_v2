"""Evaluation statistics, halo connectivity, T2 rescaling (photons per 48^3 cell matched to the decoded field), pass/fail and plot helpers."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from sparseshower.common import config
from sparseshower.evaluate import (auc_score, bootstrap_mean, c2st, connected_components, event_metrics, frechet,
                                   halo_metrics, own_other, paired_errors, passfail, w1)
from sparseshower import geometry as G

ROOT = Path(__file__).resolve().parents[1]


def cfg(grid=64):
    c = config(ROOT / "configs" / "v2.yaml")
    c["data"]["grid"] = grid
    c["field"]["base_grid"] = grid // 4
    return c


class TestStatistics(unittest.TestCase):
    def test_w1(self):
        a = np.random.default_rng(0).normal(size=4000)
        self.assertAlmostEqual(w1(a, a), 0.0)
        self.assertAlmostEqual(w1(a, a + 0.5), 0.5, places=6)

    def test_bootstrap_ci_contains_the_mean(self):
        x = np.random.default_rng(1).normal(2.0, 1.0, size=300)
        r = bootstrap_mean(np.r_[x, np.nan], 500, np.random.default_rng(2))
        self.assertTrue(r["lo"] < r["mean"] < r["hi"])
        self.assertEqual(r["n"], 300)

    def test_auc(self):
        y = np.r_[np.zeros(50), np.ones(50)]
        self.assertAlmostEqual(auc_score(y, y), 1.0)
        self.assertAlmostEqual(auc_score(y, -y), 0.0)
        self.assertAlmostEqual(auc_score(y, np.zeros(100)), 0.5)

    def test_c2st_same_vs_different(self):
        rng = np.random.default_rng(3)
        A, B = rng.normal(size=(400, 6)), rng.normal(size=(400, 6))
        same = c2st(A, B, 5, rng, n_boot=100)
        self.assertLess(abs(same["auc"] - 0.5), 0.08)
        diff = c2st(A, B * 2.0, 5, rng, n_boot=100)                  # only the spread differs: squares see it
        self.assertGreater(diff["auc"], 0.8)
        self.assertLessEqual(diff["lo"], diff["auc"])

    def test_frechet_guard(self):
        self.assertTrue(np.isnan(frechet(np.zeros((2, 3)), np.zeros((5, 3)), np.zeros((5, 3)))))


class TestEventMetrics(unittest.TestCase):
    def test_connected_components(self):
        ijk = np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5], [6, 5, 5], [9, 9, 9]])
        idx, mask = G.neighbour_table(ijk, 16, G.OFFSETS_26)
        lab = connected_components(len(ijk), idx, mask)
        self.assertEqual(len(set(lab.tolist())), 3)
        self.assertEqual(lab[0], lab[1])
        self.assertEqual(lab[2], lab[3])

    def test_halo_long_component(self):
        c = cfg()
        c["eval"]["halo_r"] = 0.5
        c["eval"]["halo_long_m"] = 0.6
        track = np.stack([np.full(20, 60), np.full(20, 32), np.arange(20, 40)], 1)   # 6.25 cm voxels: x ~ 1.78 m, 1.25 m long
        dots = np.array([[4, 4, 5], [4, 60, 50]])                          # isolated halo voxels
        core = np.stack([np.full(10, 32), np.full(10, 32), np.arange(10)], 1)
        ijk = np.concatenate([track, dots, core])
        q = np.ones(len(ijk))
        h = halo_metrics(ijk, q, c)
        self.assertEqual(h["halo_n_comp"], 3)
        self.assertEqual(h["halo_n_long"], 1)
        self.assertAlmostEqual(h["halo_q_frac"], 22 / 32)

    def test_paired_errors_of_an_event_with_itself(self):
        c = cfg()
        rng = np.random.default_rng(4)
        ijk = np.unique(rng.integers(0, 64, size=(500, 3)), axis=0)
        m = event_metrics(ijk, rng.lognormal(2, 1, len(ijk)), rng.uniform(-0.5, 0.5, (len(ijk), 3)), c)
        e = paired_errors(m, m)
        for k, v in e.items():
            self.assertAlmostEqual(v, 0.0, msg=k)

    def test_own_other(self):
        prof = {e: dict(longitudinal=np.eye(8)[e]) for e in range(8)}
        gen = [dict(eid=e, m=dict(longitudinal=np.eye(8)[e])) for e in range(8)]
        self.assertAlmostEqual(own_other(gen, prof, np.random.default_rng(0)), 0.0)
        shuffled = [dict(eid=e, m=dict(longitudinal=np.eye(8)[(e + 1) % 8])) for e in range(8)]
        self.assertGreater(own_other(shuffled, prof, np.random.default_rng(0)), 0.9)


class TestT2(unittest.TestCase):
    def test_cell_sums_match_the_decoded_field(self):
        from sparseshower.attr import t2_rescale
        from sparseshower.fields import coarse_field
        from sparseshower.samples import ancestor_lookup
        c = cfg(32)
        stats = dict(q_mean=3.0, q_std=1.5)
        rng = np.random.default_rng(5)
        ijk = np.unique(rng.integers(0, 32, size=(300, 3)), axis=0)
        q_true = rng.lognormal(2, 1, len(ijk))
        f2, counts, sums = coarse_field(ijk, q_true, 32, 8, float(c["data"]["q_eps"]), stats)
        F = np.concatenate([(f2[0] > 0)[None].astype(np.float32), f2])
        lookup = ancestor_lookup(ijk, 32, 8)
        q_gen = rng.lognormal(2, 1, len(ijk))                              # wrong photons, right cells
        q2, scale = t2_rescale(q_gen, lookup, F, c, stats)
        got = np.bincount(lookup, weights=q2, minlength=8 ** 3)
        np.testing.assert_allclose(got, sums, rtol=1e-4)
        self.assertEqual(len(scale), int((counts > 0).sum()))


class TestPassFail(unittest.TestCase):
    def test_rows(self):
        c = config(ROOT / "configs" / "v2.yaml")
        dist = dict(scalars=dict(q_total=dict(ratio=1.5), n_active=dict(ratio=3.0)),
                    c2st=dict(auc=0.6, lo=0.55, hi=0.65))
        s = dict(rungs=dict(
            sa_recon=dict(chain="sa_recon", distribution=dist,
                          paired=dict(q_rel_err=dict(mean=0.01), long_l1=dict(mean=0.05))),
            full=dict(chain="full", distribution=dist, memorization=dict(ratio=0.95))))
        rows = {(r["rung"], r["metric"]): r["ok"] for r in passfail(s, c)}
        self.assertTrue(rows[("sa_recon", "W1 ratio q_total")])
        self.assertFalse(rows[("sa_recon", "W1 ratio n_active")])
        self.assertTrue(rows[("sa_recon", "paired |Q ratio - 1|")])
        self.assertFalse(rows[("sa_recon", "paired longitudinal L1")])
        self.assertTrue(rows[("full", "C2ST AUC")])
        self.assertTrue(rows[("full", "memorization ratio")])
        self.assertNotIn(("full", "paired |Q ratio - 1|"), rows)

    def test_sdedit_curve_order(self):
        from sparseshower.viz import sdedit_curve
        r = lambda ch, s=None: dict(chain=ch, settings=dict(sigma_start=s, t2=True))   # noqa: E731
        s = dict(rungs=dict(full=r("full"), sdedit_2=r("sdedit", 2.0), sa_recon=r("sa_recon"),
                            sdedit_0_5=r("sdedit", 0.5)))
        self.assertEqual([n for _, n in sdedit_curve(s)], ["sa_recon", "sdedit_0_5", "sdedit_2", "full"])


if __name__ == "__main__":
    unittest.main()
