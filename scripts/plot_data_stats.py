"""Make the dataset figures for the report and print per-split count statistics.

Writes:
  outputs/figures/count_hist_part_B.png      head-count histogram per split
  outputs/figures/pipeline_steps_part_B.png  one training sample through the loader:
                                             full image -> 1/4-area crop -> flip -> 1/8 target

Example:
    python scripts/plot_data_stats.py --part B
"""

import argparse
import glob
import json
import os
import random
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.density import normalize_part  # noqa: E402
from src.data.loader import DOWNSAMPLE  # noqa: E402
from src.data.paths import (DEFAULT_PROCESSED_ROOT, DEFAULT_RAW_ROOT,  # noqa: E402
                            find_part_dir, processed_dir, raw_split_dir)

SPLITS = ("train", "val", "test")
COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}


def split_counts(part, processed_root):
    counts = {}
    for split in SPLITS:
        paths = sorted(glob.glob(os.path.join(processed_dir(part, split, processed_root), "*.npy")))
        if not paths:
            sys.exit(f"No density maps for {split}. Run scripts/make_density_maps.py first.")
        counts[split] = np.array([float(np.load(p).sum()) for p in paths])
    return counts


def plot_histogram(counts, part, out_path):
    bins = np.linspace(0, max(c.max() for c in counts.values()) + 10, 30)
    fig, ax = plt.subplots(figsize=(7, 4))
    for split in SPLITS:
        c = counts[split]
        ax.hist(c, bins=bins, density=True, histtype="step", linewidth=2, color=COLORS[split],
                label=f"{split} (n={len(c)}, mean {c.mean():.1f})")
    ax.set_xlabel("people per image")
    ax.set_ylabel("fraction of images (normalised)")
    ax.set_title(f"ShanghaiTech {normalize_part(part).replace('_', ' ')}: head counts per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pipeline(part, raw_root, processed_root, seed, out_path):
    """Replays the loader's train transform with a fixed seed so the figure is reproducible."""
    rng = random.Random(seed)
    dens_path = rng.choice(sorted(glob.glob(os.path.join(processed_dir(part, "train", processed_root),
                                                         "*.npy"))))
    image_id = os.path.splitext(os.path.basename(dens_path))[0]
    img_path = os.path.join(find_part_dir(raw_root, part), raw_split_dir("train"), "images",
                            image_id + ".jpg")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    dens = np.load(dens_path)

    h, w = dens.shape
    ch, cw = h // 2 // DOWNSAMPLE * DOWNSAMPLE, w // 2 // DOWNSAMPLE * DOWNSAMPLE
    y, x = rng.randint(0, h - ch), rng.randint(0, w - cw)
    crop_img, crop_d = img[y:y + ch, x:x + cw], dens[y:y + ch, x:x + cw]
    flip_img, flip_d = crop_img[:, ::-1], crop_d[:, ::-1]
    target = flip_d.reshape(ch // DOWNSAMPLE, DOWNSAMPLE, cw // DOWNSAMPLE, DOWNSAMPLE).sum(axis=(1, 3))

    boxed = img.copy()
    cv2.rectangle(boxed, (x, y), (x + cw, y + ch), (255, 255, 0), 6)
    panels = [
        (boxed, f"1. {image_id} ({w}×{h}), crop box"),
        (dens, f"density map, sum = {dens.sum():.1f}"),
        (crop_img, f"2. random crop {cw}×{ch} (1/4 area)"),
        (crop_d, f"crop density, sum = {crop_d.sum():.1f}"),
        (flip_img, "3. horizontal flip"),
        (flip_d, f"flipped density, sum = {flip_d.sum():.1f}"),
        (None, ""),
        (target, f"4. 8×8 sum-pool → {target.shape[1]}×{target.shape[0]}, sum = {target.sum():.1f}"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(9, 11))
    for ax, (data, title) in zip(axes.ravel(), panels):
        ax.axis("off")
        if data is None:
            continue
        ax.imshow(data, cmap=None if data.ndim == 3 else "jet")
        ax.set_title(title, fontsize=9)
    fig.suptitle("Training input pipeline (counts are preserved by flip and sum-pooling)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return image_id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="B", choices=["A", "B"])
    ap.add_argument("--seed", type=int, default=3, help="picks the pipeline example and crop")
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    ap.add_argument("--processed-root", default=DEFAULT_PROCESSED_ROOT)
    ap.add_argument("--out-dir", default=os.path.join("outputs", "figures"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = normalize_part(args.part)

    counts = split_counts(args.part, args.processed_root)
    stats = {s: {"images": int(len(c)), "total": round(float(c.sum())), "min": round(float(c.min())),
                 "mean": round(float(c.mean()), 1), "median": round(float(np.median(c)), 1),
                 "max": round(float(c.max()))} for s, c in counts.items()}
    print(json.dumps(stats, indent=1))

    hist_path = os.path.join(args.out_dir, f"count_hist_{tag}.png")
    plot_histogram(counts, args.part, hist_path)
    print(f"Saved {hist_path}")
    pipe_path = os.path.join(args.out_dir, f"pipeline_steps_{tag}.png")
    image_id = plot_pipeline(args.part, args.raw_root, args.processed_root, args.seed, pipe_path)
    print(f"Saved {pipe_path} (example {image_id})")


if __name__ == "__main__":
    main()
