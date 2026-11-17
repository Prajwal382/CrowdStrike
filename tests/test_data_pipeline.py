"""Tests for src/data. Run with:  python -m pytest tests -q

The synthetic tests always run. The real-data test (20 random Part B images)
runs only when data/ShanghaiTech and data/processed exist.
"""

import glob
import os
import random
import subprocess
import sys

import cv2
import numpy as np
import pytest
import scipy.io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.density import (density_adaptive, density_fixed, load_points,  # noqa: E402
                              split_train_val)
from src.data.paths import find_part_dir  # noqa: E402


def _save_gt(path, points):
    rec = {"location": np.asarray(points, dtype=np.float64), "number": float(len(points))}
    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = rec
    scipy.io.savemat(path, {"image_info": cell})


@pytest.fixture
def fake_dataset(tmp_path):
    """Tiny ShanghaiTech-shaped dataset: 10 train + 3 test images of 100x132."""
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    for sub, n in (("train_data", 10), ("test_data", 3)):
        (raw / "part_B" / sub / "images").mkdir(parents=True)
        (raw / "part_B" / sub / "ground-truth").mkdir(parents=True)
        for i in range(1, n + 1):
            img = rng.integers(0, 255, (100, 132, 3), dtype=np.uint8)
            cv2.imwrite(str(raw / "part_B" / sub / "images" / f"IMG_{i}.jpg"), img)
            pts = rng.uniform([-2, -2], [134, 102], size=(rng.integers(1, 40), 2))
            _save_gt(str(raw / "part_B" / sub / "ground-truth" / f"GT_IMG_{i}.mat"), pts)
    return raw, tmp_path / "processed"


def test_load_points_roundtrip(tmp_path):
    pts = np.array([[1.5, 2.5], [10.0, 20.0]])
    _save_gt(str(tmp_path / "GT_IMG_1.mat"), pts)
    np.testing.assert_allclose(load_points(str(tmp_path / "GT_IMG_1.mat")), pts)


@pytest.mark.parametrize("fn", [density_fixed, density_adaptive])
def test_density_sum_matches_count_including_edges(fn):
    rng = np.random.default_rng(1)
    pts = np.concatenate([rng.uniform(0, 200, (50, 2)),
                          [[0, 0], [199.9, 149.9], [-3, 5], [250, 160]]])  # border + outside
    d = fn((150, 200), pts)
    assert d.dtype == np.float32 and d.shape == (150, 200)
    assert abs(d.sum() - len(pts)) / len(pts) < 1e-3


def test_empty_and_single_point():
    assert density_fixed((32, 32), np.zeros((0, 2))).sum() == 0
    assert abs(density_adaptive((32, 32), [[5, 5]]).sum() - 1) < 1e-3


def test_split_is_deterministic_and_disjoint():
    ids = [f"IMG_{i}" for i in range(1, 401)]
    a, b = split_train_val(ids), split_train_val(list(reversed(ids)))
    assert a == b
    assert len(a["val"]) == 40 and len(a["train"]) == 360
    assert not set(a["val"]) & set(a["train"])


def test_make_maps_and_loader(fake_dataset):
    raw, processed = fake_dataset
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_density_maps.py"),
                    "--raw-root", str(raw), "--processed-root", str(processed)],
                   check=True, cwd=ROOT)
    assert len(glob.glob(str(processed / "part_B" / "val" / "*.npy"))) == 1
    assert len(glob.glob(str(processed / "part_B" / "train" / "*.npy"))) == 9

    from src.data.loader import get_dataset
    tr = get_dataset("B", "train", batch_size=2, raw_root=str(raw), processed_root=str(processed))
    img, dens = next(iter(tr))
    assert img.shape == (2, 48, 64, 3) and dens.shape == (2, 6, 8, 1)
    assert img.dtype.name == "float32" and dens.dtype.name == "float32"

    te = get_dataset("B", "test", batch_size=1, raw_root=str(raw), processed_root=str(processed))
    for (img, dens), npy in zip(te, sorted(glob.glob(str(processed / "part_B" / "test" / "*.npy")))):
        assert img.shape == (1, 104, 136, 3) and dens.shape == (1, 13, 17, 1)
        assert abs(float(dens.numpy().sum()) - float(np.load(npy).sum())) < 1e-3


REAL_RAW = os.path.join(ROOT, "data", "ShanghaiTech")
REAL_PROCESSED = os.path.join(ROOT, "data", "processed", "part_B")


@pytest.mark.skipif(not os.path.isdir(REAL_PROCESSED) or find_part_dir(REAL_RAW, "B") is None,
                    reason="real Part B data not downloaded / processed")
def test_real_part_b_sums_match_head_counts():
    part_dir = find_part_dir(REAL_RAW, "B")
    npys = sorted(glob.glob(os.path.join(REAL_PROCESSED, "*", "*.npy")))
    for npy in random.Random(0).sample(npys, 20):
        split = os.path.basename(os.path.dirname(npy))
        sub = "test_data" if split == "test" else "train_data"
        image_id = os.path.splitext(os.path.basename(npy))[0]
        n = len(load_points(os.path.join(part_dir, sub, "ground-truth", f"GT_{image_id}.mat")))
        assert abs(np.load(npy).sum() - n) / max(n, 1) < 1e-3, image_id
