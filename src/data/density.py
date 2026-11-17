"""Ground-truth density map generation for ShanghaiTech.

Contract (docs/CONTRACT.md):
  * Density maps are float32 arrays at full image resolution (H, W).
  * density.sum() == number of annotated heads (relative error < 1e-3).

Each head is spread with a 2-D Gaussian that is truncated at 3 sigma and at
the image border, then renormalised so it integrates to exactly 1. That
renormalisation is what keeps the count invariant for heads near the edge.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Sequence

import numpy as np
import scipy.io
from scipy.spatial import KDTree

PART_B_SIGMA = 15.0
ADAPTIVE_K = 3
ADAPTIVE_BETA = 0.3
VAL_FRACTION = 0.1
SPLIT_SEED = 42


def load_points(mat_path: str) -> np.ndarray:
    """Read head annotations from a ShanghaiTech GT_*.mat file.

    Returns:
        float array of shape (N, 2) with (x, y) pixel coordinates.
    """
    mat = scipy.io.loadmat(mat_path)
    points = mat["image_info"][0, 0][0, 0][0]
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _gaussian_1d(center: float, sigma: float, size: int):
    """Truncated 1-D Gaussian clipped to [0, size), normalised to sum 1.

    Returns (start, values) so the caller can add it into a slice.
    """
    radius = max(1, int(np.ceil(3 * sigma)))
    c = int(np.clip(np.floor(center), 0, size - 1))
    lo, hi = max(0, c - radius), min(size, c + radius + 1)
    xs = np.arange(lo, hi, dtype=np.float64)
    g = np.exp(-0.5 * ((xs - c) / sigma) ** 2)
    return lo, g / g.sum()


def _splat(shape, points: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    h, w = shape[:2]
    density = np.zeros((h, w), dtype=np.float64)
    for (x, y), s in zip(points, sigmas):
        x0, gx = _gaussian_1d(x, s, w)
        y0, gy = _gaussian_1d(y, s, h)
        density[y0:y0 + len(gy), x0:x0 + len(gx)] += np.outer(gy, gx)
    return density.astype(np.float32)


def density_fixed(shape, points: np.ndarray, sigma: float = PART_B_SIGMA) -> np.ndarray:
    """Density map with a fixed Gaussian sigma (used for Part B, sigma=15)."""
    points = np.asarray(points).reshape(-1, 2)
    return _splat(shape, points, [sigma] * len(points))


def adaptive_sigmas(points: np.ndarray, k: int = ADAPTIVE_K,
                    beta: float = ADAPTIVE_BETA, default: float = PART_B_SIGMA) -> np.ndarray:
    """Geometry-adaptive sigma: beta * mean distance to the k nearest heads."""
    points = np.asarray(points).reshape(-1, 2)
    n = len(points)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.array([default])
    kk = min(k + 1, n)  # +1 because the nearest neighbour is the point itself
    dists, _ = KDTree(points).query(points, k=kk)
    sig = beta * dists[:, 1:].mean(axis=1)
    return np.maximum(sig, 1.0)  # avoid degenerate kernels for duplicate points


def density_adaptive(shape, points: np.ndarray, k: int = ADAPTIVE_K,
                     beta: float = ADAPTIVE_BETA) -> np.ndarray:
    """Density map with geometry-adaptive sigma (used for Part A)."""
    points = np.asarray(points).reshape(-1, 2)
    return _splat(shape, points, adaptive_sigmas(points, k, beta))


def make_density(shape, points: np.ndarray, part: str) -> np.ndarray:
    """Pick the right kernel for a dataset part ('A' or 'B')."""
    if normalize_part(part) == "part_A":
        return density_adaptive(shape, points)
    return density_fixed(shape, points)


def normalize_part(part: str) -> str:
    """Accept 'B', 'b', 'part_B' and return 'part_B'."""
    p = part.strip()
    if p.lower().startswith("part_"):
        p = p[5:]
    p = p.upper()
    if p not in ("A", "B"):
        raise ValueError(f"Unknown part {part!r}; expected 'A' or 'B'")
    return f"part_{p}"


def split_train_val(image_ids: List[str], val_fraction: float = VAL_FRACTION,
                    seed: int = SPLIT_SEED) -> Dict[str, List[str]]:
    """Deterministically hold out a fraction of the training ids as validation."""
    ids = sorted(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_fraction))
    return {"train": sorted(ids[n_val:]), "val": sorted(ids[:n_val])}


def image_id_from_path(path: str) -> str:
    """'.../IMG_12.jpg' -> 'IMG_12'."""
    return os.path.splitext(os.path.basename(path))[0]
