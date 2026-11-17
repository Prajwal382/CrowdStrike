"""Load a trained model for evaluation/inference, or build an untrained stand-in.

Used by src/evaluate.py and src/predict.py so both load checkpoints the same way.
"""

from __future__ import annotations

import os
from typing import Optional

from tensorflow import keras


def build_dummy_model(seed: int = 0) -> keras.Model:
    """Untrained model with CSRNet's input/output shapes, for smoke tests.

    (None, None, 3) -> 8x8 average pooling -> 1x1 conv with ReLU -> (H/8, W/8, 1).
    """
    keras.utils.set_random_seed(seed)
    inp = keras.Input(shape=(None, None, 3))
    x = keras.layers.AveragePooling2D(pool_size=8)(inp)
    out = keras.layers.Conv2D(1, 1, activation="relu")(x)
    return keras.Model(inp, out, name="dummy_1x1")


def load_model(weights: Optional[str] = None, dummy: bool = False) -> keras.Model:
    """Return a model ready for inference.

    Args:
        weights: a full saved model (.keras or .h5) or a weights-only file
            (.weights.h5). Weights-only files are loaded into build_csrnet().
        dummy: ignore weights and return the untrained build_dummy_model().
    """
    if dummy:
        return build_dummy_model()
    if not weights:
        raise ValueError("Pass a weights file, or use the dummy model for a smoke test.")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not weights.endswith(".weights.h5"):
        try:
            return keras.models.load_model(weights, compile=False)
        except Exception:  # not a full model; fall back to weights-only loading
            pass
    from src.model import build_csrnet
    model = build_csrnet()
    model.load_weights(weights)
    return model
