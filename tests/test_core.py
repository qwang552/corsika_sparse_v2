"""Core unit tests (NumPy + PyYAML; the selftest refusal test also needs torch): geometry, data, config, 48^3 field, D4, levels, jobs.

Run from the project directory:  python -m pytest tests -q
(or  PYTHONPATH=. python -m unittest discover -s tests -v)
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sparseshower import geometry as G
from sparseshower.common import config
from sparseshower.data import Codec, process_event
from sparseshower.fields import (coarse_field, d4_field, d4_ijk, d4_offsets, field_statistics, occupied_cells,
                                 photon_sums, truth_field3)
from sparseshower.loader import KIND_SECTIONS, fingerprint
from sparseshower.samples import bits_for_parents, level_grids, level_sets
from sparseshower.struct_data import expand_children
from sparseshower.synth import synth_event

ROOT = Path(__file__).resolve().parents[1]
RANGES = [[-2.0, 2.0], [-2.0, 2.0], [46.0, 50.0]]


def random_event(rng, grid=32, n=400):
    ijk = np.unique(rng.integers(0, grid, size=(n, 3)), axis=0)
    q = rng.lognormal(3.0, 1.5, size=len(ijk))
    off = rng.uniform(-0.5, 0.5, size=(len(ijk), 3))
    return ijk, q, off


class TestGeometry(unittest.TestCase):
    def test_center_roundtrip(self):
        ijk = np.array([[0, 1, 2], [7, 7, 7]])
        self.assertTrue((G.voxel_index(G.voxel_center(ijk, RANGES, 8), RANGES, 8) == ijk).all())

    def test_pack_unpack(self):
        ijk = np.array([[0, 0, 0], [3, 1, 2], [7, 7, 7]])
        self.assertTrue((G.unpack(G.pack(ijk, 8), 8) == ijk).all())

    def test_neighbour_table_matches_brute_force(self):
        rng = np.random.default_rng(0)
        ijk = np.unique(rng.integers(0, 12, size=(200, 3)), axis=0)
        idx, mask = G.neighbour_table(ijk, 12, G.OFFSETS_26)
        lookup = {tuple(v): i for i, v in enumerate(ijk.tolist())}
        for n in rng.choice(len(ijk), size=20, replace=False):
            for k, off in enumerate(G.OFFSETS_26):
                target = tuple((ijk[n] + off).tolist())
                if target in lookup:
                    self.assertEqual((idx[n, k], mask[n, k]), (lookup[target], 1.0))
                else:
                    self.assertEqual((idx[n, k], mask[n, k]), (n, 0.0))

    def test_box_dimension_of_a_line(self):
        ijk = np.stack([np.arange(64), np.zeros(64, int), np.zeros(64, int)], axis=1)
        dims = G.box_dimension(G.box_count(ijk, [1, 2, 4, 8]), [1, 2, 4, 8])
        self.assertTrue(all(abs(d - 1.0) < 1e-9 for d in dims))


class TestData(unittest.TestCase):
    def test_codec_roundtrip(self):
        stats = dict(q_log_mean=5.0, q_log_std=1.5, off_std=0.28, macro_log_mean=[0.0] * 4,
                     macro_log_std=[1.0] * 4)
        codec = Codec(stats, 0.001)
        q = np.array([1.0, 100.0, 5e4])
        off = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [-0.5, 0.5, 0.25]])
        q2, off2 = codec.decode(codec.encode(q, off))
        np.testing.assert_allclose(q, q2, rtol=1e-5)
        np.testing.assert_allclose(off, off2, atol=1e-6)

    def test_aggregation_conserves_photons(self):
        c = dict(data=dict(ranges=RANGES, grid=8, time_max=None, q_eps=0.001, parquet_batch_rows=1024,
                           time_mode="raw"),
                 macro=dict(z_slabs=4, r_rings=2, r_max=2.0, axis_x=0.0, axis_y=0.0))
        cols = synth_event(3, RANGES, max_records=3000)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.npz"
            np.savez(path, **cols)
            arrays, _, info = process_event(path, c)
        self.assertLess(info["conservation_rel"], 1e-9)
        self.assertTrue((np.abs(arrays["off"]) <= 0.5 + 1e-6).all())


class TestConfig(unittest.TestCase):
    def test_configs_load(self):
        for name in ("data.yaml", "v2.yaml", "smoke.yaml"):
            c = config(ROOT / "configs" / name)
            self.assertEqual(int(c["data"]["grid"]) % int(c["field"]["base_grid"]), 0, name)
        c = config(ROOT / "configs" / "v2.yaml")
        self.assertEqual(level_grids(192, 48), [48, 96, 192])
        self.assertEqual(int(c["field"]["base_grid"]) // 2 ** (len(c["ae"]["widths"]) - 1), 12)
        self.assertIn("read", open(ROOT / "configs" / "data.yaml").read().lower())   # cache is read only

    def test_selftest_refuses_a_real_config(self):
        from sparseshower.selftest import run
        with self.assertRaises(SystemExit):              # it would wipe paths.processed / output
            run(config(ROOT / "configs" / "v2.yaml"), verbose=False)

    def test_every_kind_fingerprints_existing_sections(self):
        c = config(ROOT / "configs" / "v2.yaml")
        for kind, sections in KIND_SECTIONS.items():
            for s in sections:
                self.assertIn(s, c, f"{kind}: {s}")

    def test_fingerprint_ignores_other_kinds_and_machine_knobs(self):
        c = config(ROOT / "configs" / "v2.yaml")
        ids = [1, 2, 3]
        base = fingerprint(c, "attr", ids)
        c2 = config(ROOT / "configs" / "v2.yaml")
        c2["dit"]["depth"] = 99                         # another kind's section
        c2["model"]["checkpoint"] = True                # memory knob
        c2["train"]["amp"] = "fp16"
        self.assertEqual(fingerprint(c2, "attr", ids), base)
        c2["attr"]["drop_frac"] = 0.2                   # science of this kind
        self.assertNotEqual(fingerprint(c2, "attr", ids), base)
        self.assertNotEqual(fingerprint(c, "attr", [1, 2]), base)


class TestField(unittest.TestCase):
    def test_coarse_field_conserves_counts_and_photons(self):
        rng = np.random.default_rng(1)
        ijk, q, _ = random_event(rng)
        field, counts, sums = coarse_field(ijk, q, 32, 8, 1e-3)
        self.assertEqual(field.shape, (2, 8, 8, 8))
        self.assertEqual(int(counts.sum()), len(ijk))
        self.assertAlmostEqual(float(sums.sum()), float(q.sum()), places=6)
        self.assertLessEqual(float(field[0].max()), 1.0 + 1e-6)          # log(1+n)/log(1+64)
        f3 = truth_field3(field)
        self.assertTrue(np.array_equal(f3[0].reshape(-1) > 0, counts > 0))

    def test_photon_sums_inverts_ch1(self):
        rng = np.random.default_rng(2)
        ijk, q, _ = random_event(rng)
        stats = dict(q_mean=4.0, q_std=2.0)
        field, counts, sums = coarse_field(ijk, q, 32, 8, 1e-3, stats)
        back = photon_sums(field[1].reshape(-1), counts > 0, stats, 1e-3)
        np.testing.assert_allclose(back, sums, rtol=1e-4)

    def test_field_statistics_streams(self):
        c = dict(data=dict(grid=32, q_eps=1e-3), field=dict(base_grid=8))
        rng = np.random.default_rng(3)

        def gen():
            for _ in range(5):
                ijk, q, _ = random_event(rng)
                yield dict(ijk=ijk, q=q)
        s = field_statistics(gen(), c)
        self.assertTrue(np.isfinite(s["q_mean"]) and s["q_std"] > 0)

    def test_occupied_cells_order_matches_flat_layout(self):
        p = np.zeros((4, 4, 4))
        p[1, 2, 3] = 0.9                     # [z, y, x]
        self.assertEqual(occupied_cells(p, 0.5).tolist(), [[3, 2, 1]])


class TestD4(unittest.TestCase):
    def test_field_and_voxels_agree(self):
        rng = np.random.default_rng(4)
        ijk, q, _ = random_event(rng)
        field, _, _ = coarse_field(ijk, q, 32, 8, 1e-3)
        for t in range(8):
            moved, _, _ = coarse_field(d4_ijk(ijk, 32, t), q, 32, 8, 1e-3)
            np.testing.assert_allclose(d4_field(field, t), moved, err_msg=f"t={t}")

    def test_group(self):
        rng = np.random.default_rng(5)
        ijk, _, _ = random_event(rng, n=50)
        self.assertTrue(np.array_equal(d4_ijk(ijk, 32, 0), ijk))
        seen = {tuple(map(tuple, d4_ijk(ijk, 32, t))) for t in range(8)}
        self.assertEqual(len(seen), 8)
        for t in range(8):
            self.assertTrue((d4_ijk(ijk, 32, t)[:, 2] == ijk[:, 2]).all())        # z untouched
            self.assertTrue(((d4_ijk(ijk, 32, t) >= 0) & (d4_ijk(ijk, 32, t) < 32)).all())

    def test_offsets_move_with_the_voxel(self):
        rng = np.random.default_rng(6)
        ijk, _, off = random_event(rng, n=60)
        voxel = 4.0 / 32
        for t in range(8):
            p = G.voxel_center(ijk, RANGES, 32) + off * voxel
            p2 = G.voxel_center(d4_ijk(ijk, 32, t), RANGES, 32) + d4_offsets(off, t) * voxel
            # the same D4 map applied to (x, y) about the box axis
            u, v = p[:, 0], p[:, 1]
            r, m = t % 4, t // 4
            for _ in range(r):
                u, v = -v, u
            if m:
                u = -u
            np.testing.assert_allclose(np.stack([u, v, p[:, 2]], 1), p2, atol=1e-9, err_msg=f"t={t}")


class TestLevels(unittest.TestCase):
    def test_bits_roundtrip_and_lost_children(self):
        rng = np.random.default_rng(7)
        ijk, _, _ = random_event(rng, grid=32, n=600)
        s48, s96 = level_sets(ijk, [16, 32])
        bits, lost, _ = bits_for_parents(s48, s96, 32)
        self.assertEqual(lost, 0)
        back = expand_children(s48, bits)
        self.assertTrue(np.array_equal(back, s96))
        # a parent set missing some parents loses exactly their children
        keep = np.ones(len(s48), bool)
        keep[::5] = False
        bits2, lost2, mask = bits_for_parents(s48[keep], s96, 32)
        self.assertEqual(lost2, int(bits[~keep].sum()))
        self.assertEqual(int(mask.sum()), lost2)


class TestJobs(unittest.TestCase):
    """The cluster rejects trailing comments; GPU jobs need the capability requirement."""

    def test_sub_files(self):
        cli = (ROOT / "sparseshower" / "cli.py").read_text()
        commands = re.search(r"COMMANDS = \((.*?)\)", cli, re.S).group(1)
        for f in sorted((ROOT / "jobs").glob("*.sub")):
            text = f.read_text()
            for line in text.splitlines():
                self.assertFalse("#" in line and not line.lstrip().startswith("#"), f"{f.name}: {line}")
            if re.search(r"^request_gpus = 1", text, re.M):
                self.assertIn("(GPUs_Capability >= 6.0) && (GPUs_Capability < 9.0)", text, f.name)
            self.assertRegex(text, r"(?m)^queue ", f.name)
            m = re.search(r'^arguments = "([^\s"]+)', text, re.M)
            if "run_pipeline.sh" in text and m and m.group(1) not in ("selftest", "notebook"):
                self.assertIn(f'"{m.group(1)}"', commands, f"{f.name}: unknown command {m.group(1)}")

    def test_make_dag(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_dag.py"), "--out", f"{tmp}/t.dag",
                                  "--output", f"{tmp}/none"], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            dag = Path(tmp, "t.dag").read_text()
            jobs = set(re.findall(r"^JOB (\S+) ", dag, re.M))
            for line in re.findall(r"^PARENT (.*) CHILD (.*)$", dag, re.M):
                for n in " ".join(line).split():
                    self.assertIn(n, jobs)
            for f in Path(tmp, "t_nodes").glob("*.sub"):
                for line in f.read_text().splitlines():
                    self.assertFalse("#" in line and not line.startswith("#"), f"{f.name}: {line}")
            again = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_dag.py"), "--out", f"{tmp}/t.dag"],
                                   capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)                           # never overwritten


if __name__ == "__main__":
    unittest.main()
