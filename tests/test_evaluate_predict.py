"""Tests for src/evaluate.py and src/predict.py. Run with:  python -m pytest tests -q"""

import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluate import count_metrics  # noqa: E402
from src.model_utils import build_dummy_model  # noqa: E402
from src.predict import limit_size, make_overlay, pad_to_multiple, predict_count  # noqa: E402


def test_count_metrics():
    m = count_metrics([1.0, 3.0], [2.0, 1.0])
    assert m["mae"] == pytest.approx(1.5)
    assert m["mse"] == pytest.approx(2.5)
    assert m["rmse"] == pytest.approx(np.sqrt(2.5))


def test_pad_to_multiple():
    img = np.ones((301, 457, 3), dtype=np.uint8)
    out = pad_to_multiple(img)
    assert out.shape == (304, 464, 3)
    assert out[:301, :457].sum() == img.sum() and out[301:].sum() == 0


def test_predict_count_and_overlay(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (301, 457, 3), dtype=np.uint8)
    path = str(tmp_path / "img.png")
    cv2.imwrite(path, img)
    count, density = predict_count(build_dummy_model(), path)
    assert density.shape == (38, 58)
    assert count == pytest.approx(float(density.sum()))
    overlay = make_overlay(img, density, count)
    assert overlay.shape == img.shape and overlay.dtype == np.uint8


def test_limit_size():
    img = np.zeros((3000, 4000, 3), dtype=np.uint8)
    assert limit_size(img, 1024).shape == (768, 1024, 3)
    assert limit_size(img, None).shape == img.shape
    small = np.zeros((500, 600, 3), dtype=np.uint8)
    assert limit_size(small, 1024).shape == small.shape
