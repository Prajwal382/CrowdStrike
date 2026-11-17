"""tf.data input pipeline.

Contract (docs/CONTRACT.md):
    get_dataset(part, split, batch_size, crop_size=None) -> tf.data.Dataset
    yields (image, density_1_8)
      image        float32 (B, H, W, 3), RGB, ImageNet mean/std normalised
      density_1_8  float32 (B, H/8, W/8, 1), 8x8 sum-pooled, so its sum == count

Train split: random crop (default 1/4 of the image area, i.e. half height x
half width), random horizontal flip, shuffle.
Val/test splits: whole image, zero-padded on the bottom/right to a multiple
of 8 (padding adds no density, so the count is unchanged).
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from src.data.density import normalize_part
from src.data.paths import (DEFAULT_PROCESSED_ROOT, DEFAULT_RAW_ROOT,
                            find_part_dir, processed_dir, raw_split_dir)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DOWNSAMPLE = 8


def normalize_image(rgb: Union[np.ndarray, tf.Tensor]):
    """uint8/float RGB in [0, 255] -> float32 ImageNet-normalised.

    Works on NumPy arrays and tf tensors, so predict.py can reuse it.
    """
    if isinstance(rgb, np.ndarray):
        return ((rgb.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    rgb = tf.cast(rgb, tf.float32) / 255.0
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def sum_pool(density: tf.Tensor, factor: int = DOWNSAMPLE) -> tf.Tensor:
    """(H, W) -> (H/f, W/f, 1) by summing f x f blocks. H, W must divide by f."""
    h, w = tf.shape(density)[0], tf.shape(density)[1]
    d = tf.reshape(density, [h // factor, factor, w // factor, factor])
    return tf.reduce_sum(d, axis=[1, 3])[..., tf.newaxis]


def list_samples(part: str, split: str, raw_root: str = DEFAULT_RAW_ROOT,
                 processed_root: str = DEFAULT_PROCESSED_ROOT) -> List[Tuple[str, str]]:
    """Return sorted (image_path, density_path) pairs for a split."""
    dens_dir = processed_dir(part, split, processed_root)
    dens_paths = sorted(glob.glob(os.path.join(dens_dir, "*.npy")))
    if not dens_paths:
        raise FileNotFoundError(
            f"No density maps in {dens_dir}. Run scripts/make_density_maps.py first.")
    part_dir = find_part_dir(raw_root, part)
    if part_dir is None:
        raise FileNotFoundError(f"Could not find {normalize_part(part)} under {raw_root}")
    img_dir = os.path.join(part_dir, raw_split_dir(split), "images")
    pairs = []
    for dp in dens_paths:
        image_id = os.path.splitext(os.path.basename(dp))[0]
        ip = os.path.join(img_dir, image_id + ".jpg")
        if not os.path.exists(ip):
            raise FileNotFoundError(f"Image for {image_id} not found at {ip}")
        pairs.append((ip, dp))
    return pairs


def _load_pair(img_path, dens_path):
    img = tf.io.decode_jpeg(tf.io.read_file(img_path), channels=3)
    dens = tf.numpy_function(lambda p: np.load(p.decode()).astype(np.float32),
                             [dens_path], tf.float32)
    dens.set_shape([None, None])
    return img, dens


def _random_crop(img, dens, crop_size: Optional[Tuple[int, int]]):
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    if crop_size is None:
        ch, cw = h // 2, w // 2  # 1/4 of the image area
    else:
        ch, cw = tf.minimum(crop_size[0], h), tf.minimum(crop_size[1], w)
    ch = tf.maximum(ch // DOWNSAMPLE * DOWNSAMPLE, DOWNSAMPLE)
    cw = tf.maximum(cw // DOWNSAMPLE * DOWNSAMPLE, DOWNSAMPLE)
    y = tf.random.uniform([], 0, h - ch + 1, dtype=tf.int32)
    x = tf.random.uniform([], 0, w - cw + 1, dtype=tf.int32)
    return img[y:y + ch, x:x + cw], dens[y:y + ch, x:x + cw]


def _pad_to_multiple(img, dens):
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    ph = (-h) % DOWNSAMPLE
    pw = (-w) % DOWNSAMPLE
    img = tf.pad(img, [[0, ph], [0, pw], [0, 0]])  # zeros == mean colour after normalisation
    dens = tf.pad(dens, [[0, ph], [0, pw]])
    return img, dens


def get_dataset(part: str, split: str, batch_size: int,
                crop_size: Optional[Union[int, Tuple[int, int]]] = None,
                raw_root: str = DEFAULT_RAW_ROOT,
                processed_root: str = DEFAULT_PROCESSED_ROOT,
                shuffle: Optional[bool] = None,
                seed: Optional[int] = None) -> tf.data.Dataset:
    """Build the tf.data pipeline for one split.

    Args:
        part: 'A' or 'B' (also accepts 'part_B').
        split: 'train', 'val' or 'test'.
        batch_size: batch size. Images of different sizes cannot be batched,
            so use batch_size=1 for val/test on Part A and for Part A train
            unless crop_size is given.
        crop_size: train only. None -> random crop of half height x half width
            (1/4 area). An int or (h, w) -> fixed crop size.
        raw_root, processed_root: data locations (override on Colab/Drive).
        shuffle: defaults to True for train, False otherwise.
        seed: optional seed for shuffling and crops.

    Returns:
        tf.data.Dataset of (image, density_1_8).
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train/val/test, got {split!r}")
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    is_train = split == "train"
    shuffle = is_train if shuffle is None else shuffle

    pairs = list_samples(part, split, raw_root, processed_root)
    img_paths = [p[0] for p in pairs]
    dens_paths = [p[1] for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((img_paths, dens_paths))
    if shuffle:
        ds = ds.shuffle(len(pairs), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)

    def _prep(img, dens):
        if is_train:
            img, dens = _random_crop(img, dens, crop_size)
            if tf.random.uniform([]) < 0.5:
                img = tf.image.flip_left_right(img)
                dens = tf.reverse(dens, axis=[1])
        else:
            img, dens = _pad_to_multiple(img, dens)
        return normalize_image(img), sum_pool(dens)

    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
