"""FieldAE, LatentDiT and the EDM / SDEdit sampler (torch, CPU, small widths)."""
from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:                     # pragma: no cover
    torch = None

EDM = dict(sigma_min=0.002, sigma_max=80.0, rho=7.0, sigma_data=1.0, P_mean=-1.2, P_std=1.2)


@unittest.skipIf(torch is None, "torch not installed")
class TestFieldAE(unittest.TestCase):
    def setUp(self):
        from sparseshower.field_ae import FieldAE
        torch.manual_seed(0)
        self.cfg = dict(widths=[8, 16, 32], blocks=1, latent_channels=8, attn_heads=2)
        self.ae = FieldAE(self.cfg).eval()

    def test_shapes_48_to_12(self):
        from sparseshower.field_ae import decoded_field
        x = torch.randn(2, 2, 48, 48, 48)
        with torch.no_grad():
            out, mu, logvar = self.ae(x, sample=False)
        self.assertEqual(tuple(mu.shape), (2, 8, 12, 12, 12))
        self.assertEqual(tuple(logvar.shape), (2, 8, 12, 12, 12))
        F = decoded_field(out)
        self.assertEqual(tuple(F.shape), (2, 3, 48, 48, 48))
        self.assertTrue(((F[:, 0] >= 0) & (F[:, 0] <= 1)).all())

    def test_checkpointing_does_not_change_the_forward(self):
        x = torch.randn(1, 2, 48, 48, 48)
        self.ae.train()
        torch.manual_seed(1)
        a = self.ae(x, sample=False)[0]
        self.ae.use_ckpt = True
        torch.manual_seed(1)
        b = self.ae(x, sample=False)[0]
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        b.float().square().mean().backward()                    # and it backpropagates
        self.assertIsNotNone(next(self.ae.parameters()).grad)


@unittest.skipIf(torch is None, "torch not installed")
class TestDiT(unittest.TestCase):
    def setUp(self):
        from sparseshower.dit import LatentDiT
        torch.manual_seed(0)
        self.net = LatentDiT(dict(patch=2, width=32, depth=2, heads=4, mlp_ratio=2), 8, 12)

    def test_patchify_roundtrip(self):
        z = torch.randn(3, 8, 12, 12, 12)
        t = self.net.patchify(z)
        self.assertEqual(tuple(t.shape), (3, 216, 64))
        self.assertTrue(torch.equal(self.net.unpatchify(t), z))

    def test_adaln_zero_start(self):
        z = torch.randn(2, 8, 12, 12, 12)
        out = self.net(z, torch.tensor([0.3, -1.0]))
        self.assertEqual(out.shape, z.shape)
        self.assertEqual(float(out.abs().max()), 0.0)          # zero-initialised output layer

    def test_scalar_and_batched_noise_levels(self):
        for m in self.net.parameters():
            torch.nn.init.normal_(m, std=0.02)
        z = torch.randn(2, 8, 12, 12, 12)
        a = self.net(z, torch.tensor([0.5]))
        b = self.net(z, torch.tensor([0.5, 0.5]))
        self.assertTrue(torch.allclose(a, b, atol=1e-6))


@unittest.skipIf(torch is None, "torch not installed")
class TestSampler(unittest.TestCase):
    def test_schedule_with_sigma_start(self):
        from sparseshower.diffusion import karras_schedule
        s = karras_schedule(8, EDM, "cpu", sigma_max=2.0)
        self.assertAlmostEqual(float(s[0]), 2.0, places=5)
        self.assertEqual(float(s[-1]), 0.0)
        self.assertTrue((s[1:] < s[:-1]).all())
        full = karras_schedule(8, EDM, "cpu")
        self.assertAlmostEqual(float(full[0]), 80.0, places=3)

    def test_sdedit_with_a_perfect_denoiser_returns_the_clean_sample(self):
        from sparseshower.diffusion import heun_sample
        x0 = torch.randn(2, 3, 4)
        den = lambda x, s, **_: x0                                          # noqa: E731
        g = torch.Generator().manual_seed(0)
        out = heun_sample(den, x0.shape, EDM, "cpu", steps=6, generator=g, x_init=x0, sigma_start=1.0)
        self.assertTrue(torch.allclose(out, x0, atol=1e-5))
        with self.assertRaises(ValueError):
            heun_sample(den, x0.shape, EDM, "cpu", steps=6, x_init=x0)

    def test_small_sigma_start_stays_close(self):
        """With D = identity, SDEdit keeps x_init + sigma_start * noise: the shift is O(sigma_start)."""
        from sparseshower.diffusion import heun_sample
        x0 = torch.randn(4, 1000)
        den = lambda x, s, **_: x                                           # noqa: E731
        for s in (0.25, 2.0):
            out = heun_sample(den, x0.shape, EDM, "cpu", steps=4, generator=torch.Generator().manual_seed(1),
                              x_init=x0, sigma_start=s)
            self.assertAlmostEqual(float((out - x0).std()), s, delta=0.1 * s)

    def test_dense_loss_shapes(self):
        from sparseshower.diffusion import Denoiser, edm_loss_dense
        den = Denoiser(lambda x, c_noise, **_: torch.zeros_like(x), 1.0)
        x = torch.randn(4, 8, 6, 6, 6)
        loss, pred = edm_loss_dense(den, x, torch.full((4,), 0.5))
        self.assertEqual(pred.shape, x.shape)
        self.assertTrue(torch.isfinite(loss))


class TestNN(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_nn_distance(self):
        from sparseshower.dit import nn_distance
        a = np.random.default_rng(0).normal(size=(5, 2, 3, 3, 3))
        d = nn_distance(a, a)
        self.assertTrue(np.allclose(d, 0.0, atol=1e-6))           # float cancellation only
        d2 = nn_distance(a + 1.0, a)
        self.assertTrue((d2 > 0).all())


if __name__ == "__main__":
    unittest.main()
