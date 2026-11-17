"""Save a figure of image / ground-truth density pairs for the report.

Example:
    python scripts/plot_density_examples.py --part B --split train --n 6
"""

import argparse
import glob
import os
import random
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.paths import (DEFAULT_PROCESSED_ROOT, DEFAULT_RAW_ROOT,  # noqa: E402
                            find_part_dir, processed_dir, raw_split_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="B", choices=["A", "B"])
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=6, help="number of pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    ap.add_argument("--processed-root", default=DEFAULT_PROCESSED_ROOT)
    ap.add_argument("--out", default=os.path.join("outputs", "figures", "density_examples.png"))
    args = ap.parse_args()

    dens_paths = sorted(glob.glob(os.path.join(processed_dir(args.part, args.split,
                                                             args.processed_root), "*.npy")))
    if not dens_paths:
        sys.exit("No density maps found. Run scripts/make_density_maps.py first.")
    img_dir = os.path.join(find_part_dir(args.raw_root, args.part), raw_split_dir(args.split), "images")
    chosen = random.Random(args.seed).sample(dens_paths, min(args.n, len(dens_paths)))

    cols = 2  # pairs per row -> 4 axes per row
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * cols * 2, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, dp in enumerate(chosen):
        image_id = os.path.splitext(os.path.basename(dp))[0]
        img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, image_id + ".jpg")), cv2.COLOR_BGR2RGB)
        dens = np.load(dp)
        r, c = divmod(i, cols)
        axes[r, 2 * c].imshow(img)
        axes[r, 2 * c].set_title(image_id, fontsize=9)
        axes[r, 2 * c + 1].imshow(dens, cmap="jet")
        axes[r, 2 * c + 1].set_title(f"GT count = {dens.sum():.0f}", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
