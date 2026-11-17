"""Locate the raw ShanghaiTech folders and the processed density maps.

Expected raw layout (official zip or Kaggle mirror, either name works):

    data/ShanghaiTech/part_B[_final]/{train_data,test_data}/images/IMG_<n>.jpg
    data/ShanghaiTech/part_B[_final]/{train_data,test_data}/ground-truth/GT_IMG_<n>.mat

Processed layout (docs/CONTRACT.md):

    data/processed/part_B/{train,val,test}/IMG_<n>.npy
"""

from __future__ import annotations

import glob
import os
from typing import Optional

from src.data.density import normalize_part

DEFAULT_RAW_ROOT = os.path.join("data", "ShanghaiTech")
DEFAULT_PROCESSED_ROOT = os.path.join("data", "processed")


def find_part_dir(raw_root: str, part: str) -> Optional[str]:
    """Find part_B (or part_B_final) anywhere up to 3 levels below raw_root."""
    name = normalize_part(part)
    for depth in range(4):
        pattern = os.path.join(raw_root, *(["*"] * depth))
        for cand in (name, name + "_final"):
            hits = sorted(glob.glob(os.path.join(pattern, cand)))
            hits = [h for h in hits if os.path.isdir(os.path.join(h, "train_data"))]
            if hits:
                return hits[0]
    return None


def raw_split_dir(split: str) -> str:
    """Our val split is carved out of the official train_data folder."""
    return "test_data" if split == "test" else "train_data"


def processed_dir(part: str, split: str, processed_root: str = DEFAULT_PROCESSED_ROOT) -> str:
    return os.path.join(processed_root, normalize_part(part), split)
