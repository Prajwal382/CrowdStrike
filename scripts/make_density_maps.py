"""Generate ground-truth density maps for ShanghaiTech.

Writes float32 .npy maps at full image resolution to
    <processed-root>/part_<X>/{train,val,test}/<image_id>.npy
and a splits.json listing which ids went to train/val/test.

Part B uses a fixed Gaussian (sigma=15). Part A uses a geometry-adaptive
Gaussian (k=3 nearest neighbours, beta=0.3). 10% of the official training
images are held out as validation with a fixed seed.

Example:
    python scripts/make_density_maps.py --part B
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.density import (SPLIT_SEED, VAL_FRACTION, image_id_from_path,  # noqa: E402
                              load_points, make_density, normalize_part,
                              split_train_val)
from src.data.paths import (DEFAULT_PROCESSED_ROOT, DEFAULT_RAW_ROOT,  # noqa: E402
                            find_part_dir, processed_dir)


def process_folder(src_dir, out_dir, ids, part, overwrite):
    os.makedirs(out_dir, exist_ok=True)
    worst = 0.0
    for i, image_id in enumerate(ids, 1):
        out_path = os.path.join(out_dir, image_id + ".npy")
        if os.path.exists(out_path) and not overwrite:
            continue
        img = cv2.imread(os.path.join(src_dir, "images", image_id + ".jpg"))
        if img is None:
            raise FileNotFoundError(f"Cannot read image {image_id} in {src_dir}")
        points = load_points(os.path.join(src_dir, "ground-truth", f"GT_{image_id}.mat"))
        dens = make_density(img.shape[:2], points, part)
        count = len(points)
        rel = abs(float(dens.sum()) - count) / max(count, 1)
        worst = max(worst, rel)
        if rel >= 1e-3:
            raise AssertionError(f"{image_id}: density sum {dens.sum():.3f} != count {count}")
        np.save(out_path, dens)
        if i % 50 == 0 or i == len(ids):
            print(f"  {out_dir}: {i}/{len(ids)}")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="B", choices=["A", "B"], help="dataset part")
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT,
                    help="folder containing part_A/ and part_B/ (default: %(default)s)")
    ap.add_argument("--processed-root", default=DEFAULT_PROCESSED_ROOT,
                    help="output folder (default: %(default)s)")
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--overwrite", action="store_true", help="regenerate existing maps")
    args = ap.parse_args()

    part_dir = find_part_dir(args.raw_root, args.part)
    if part_dir is None:
        sys.exit(f"Could not find {normalize_part(args.part)} under {args.raw_root}. "
                 "Run scripts/check_dataset.py for details.")

    train_src = os.path.join(part_dir, "train_data")
    test_src = os.path.join(part_dir, "test_data")
    train_ids = [image_id_from_path(p) for p in glob.glob(os.path.join(train_src, "images", "*.jpg"))]
    test_ids = sorted(image_id_from_path(p) for p in glob.glob(os.path.join(test_src, "images", "*.jpg")))
    splits = split_train_val(train_ids, args.val_fraction, args.seed)
    splits["test"] = test_ids

    worst = 0.0
    for split, src in (("train", train_src), ("val", train_src), ("test", test_src)):
        out_dir = processed_dir(args.part, split, args.processed_root)
        worst = max(worst, process_folder(src, out_dir, splits[split], args.part, args.overwrite))

    split_file = os.path.join(args.processed_root, normalize_part(args.part), "splits.json")
    with open(split_file, "w") as f:
        json.dump({"seed": args.seed, "val_fraction": args.val_fraction, **splits}, f, indent=1)
    print(f"Done. train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}; "
          f"worst relative count error (new maps) = {worst:.2e}; splits -> {split_file}")


if __name__ == "__main__":
    main()
