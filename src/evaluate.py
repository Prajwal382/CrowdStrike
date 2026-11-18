"""Evaluate a crowd-counting model on a ShanghaiTech split.

Reports MAE, MSE and RMSE of the per-image counts. Crowd-counting papers call
RMSE "MSE", so both are reported and labelled. Also writes a per-image CSV
(image_id, gt, pred, abs_error), a predicted-vs-GT scatter plot and a figure of
the 5 best and 5 worst images.

Examples:
    python src/evaluate.py --weights checkpoints/best.keras --part B --split test
    python src/evaluate.py --dummy --tag untrained          # smoke test, untrained model
    python src/evaluate.py --baseline                       # mean-training-count baseline
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.density import normalize_part  # noqa: E402
from src.data.paths import (DEFAULT_PROCESSED_ROOT, DEFAULT_RAW_ROOT,  # noqa: E402
                            processed_dir)

DEFAULT_OUT_DIR = "outputs"
DEFAULT_FIG_DIR = os.path.join("outputs", "figures")


def count_metrics(preds: Sequence[float], gts: Sequence[float]) -> Dict[str, float]:
    """MAE = mean|pred - gt|, MSE = mean (pred - gt)^2, RMSE = sqrt(MSE)."""
    err = np.asarray(preds, dtype=np.float64) - np.asarray(gts, dtype=np.float64)
    mse = float(np.mean(err ** 2))
    return {"mae": float(np.mean(np.abs(err))), "mse": mse, "rmse": float(np.sqrt(mse))}


def write_csv(rows: List[dict], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "gt", "pred", "abs_error"])
        w.writeheader()
        for r in rows:
            w.writerow({"image_id": r["image_id"], "gt": f"{r['gt']:.2f}",
                        "pred": f"{r['pred']:.2f}", "abs_error": f"{r['abs_error']:.2f}"})
    return path


def plot_scatter(rows: List[dict], metrics: Dict[str, float], title: str, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = np.array([r["gt"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    hi = max(gt.max(), pred.max(), 1.0) * 1.05
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gt, pred, s=12, alpha=0.6)
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="pred = GT")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Ground-truth count")
    ax.set_ylabel("Predicted count")
    ax.set_title(f"{title}\nMAE {metrics['mae']:.2f} | MSE {metrics['mse']:.1f} | "
                 f"RMSE {metrics['rmse']:.2f}", fontsize=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_best_worst(rows: List[dict], image_paths: Dict[str, str], path: str, k: int = 5) -> str:
    """Grid of the k lowest-error (top row) and k highest-error (bottom row) images."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(rows, key=lambda r: r["abs_error"])
    k = min(k, len(ranked))
    groups = [("Best", ranked[:k]), ("Worst", ranked[::-1][:k])]
    fig, axes = plt.subplots(2, k, figsize=(3.2 * k, 6.4), squeeze=False)
    for row_axes, (label, group) in zip(axes, groups):
        for ax, r in zip(row_axes, group):
            img = cv2.imread(image_paths[r["image_id"]])
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{label}: {r['image_id']}\nGT {r['gt']:.0f} | pred {r['pred']:.1f}",
                         fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def evaluate(model, part: str, split: str, out_dir: str = DEFAULT_OUT_DIR,
             fig_dir: str = DEFAULT_FIG_DIR, tag: str = None,
             raw_root: str = DEFAULT_RAW_ROOT, processed_root: str = DEFAULT_PROCESSED_ROOT,
             make_figures: bool = True) -> dict:
    """Compute MAE, MSE and RMSE of predicted counts on one split.

    Args:
        model: keras.Model mapping images to (H/8, W/8, 1) density maps.
        part: 'A' or 'B'.
        split: 'train', 'val' or 'test' (train is evaluated on random crops).
        out_dir: folder for the per-image CSV and metrics JSON.
        fig_dir: folder for the scatter plot and best/worst figure.
        tag: name used in output files (default 'part_B_test').
        raw_root, processed_root: data locations (override on Colab/Drive).
        make_figures: set False to skip the plots.

    Returns:
        dict with keys 'mae', 'mse', 'rmse', 'per_image_csv' and 'n'.
    """
    from src.data.loader import get_dataset, list_samples

    tag = tag or f"{normalize_part(part)}_{split}"
    samples = list_samples(part, split, raw_root, processed_root)
    # Val/test are not shuffled, so the dataset order matches list_samples.
    ds = get_dataset(part, split, batch_size=1, raw_root=raw_root,
                     processed_root=processed_root, shuffle=False)

    rows, image_paths = [], {}
    for (img_path, dens_path), (x, y) in zip(samples, ds):
        image_id = os.path.splitext(os.path.basename(dens_path))[0]
        pred = float(model(x, training=False).numpy().sum())
        gt = float(y.numpy().sum())
        rows.append({"image_id": image_id, "gt": gt, "pred": pred, "abs_error": abs(pred - gt)})
        image_paths[image_id] = img_path
    if len(rows) != len(samples):
        raise RuntimeError(f"Evaluated {len(rows)} images but the split has {len(samples)}")

    metrics = count_metrics([r["pred"] for r in rows], [r["gt"] for r in rows])
    csv_path = write_csv(rows, os.path.join(out_dir, f"eval_{tag}.csv"))
    result = {**metrics, "per_image_csv": csv_path, "n": len(rows)}
    if make_figures:
        result["scatter_png"] = plot_scatter(
            rows, metrics, f"{tag} ({len(rows)} images)",
            os.path.join(fig_dir, f"scatter_{tag}.png"))
        result["best_worst_png"] = plot_best_worst(
            rows, image_paths, os.path.join(fig_dir, f"best_worst_{tag}.png"))
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def mean_count_baseline(part: str = "B", split: str = "test", out_dir: str = DEFAULT_OUT_DIR,
                        processed_root: str = DEFAULT_PROCESSED_ROOT) -> dict:
    """Predict the mean training count for every image of `split`.

    The mean is taken over data/processed/part_X/train/*.npy (the 360 training
    images after the validation hold-out); GT counts are the sums of the split's maps.
    """
    def counts(s):
        paths = sorted(glob.glob(os.path.join(processed_dir(part, s, processed_root), "*.npy")))
        if not paths:
            raise FileNotFoundError(f"No density maps for {part}/{s}. "
                                    "Run scripts/make_density_maps.py first.")
        return paths, [float(np.load(p).sum()) for p in paths]

    _, train_counts = counts("train")
    mean_count = float(np.mean(train_counts))
    paths, gts = counts(split)
    rows = [{"image_id": os.path.splitext(os.path.basename(p))[0], "gt": g,
             "pred": mean_count, "abs_error": abs(mean_count - g)} for p, g in zip(paths, gts)]
    metrics = count_metrics([mean_count] * len(gts), gts)
    tag = f"baseline_{normalize_part(part)}_{split}"
    csv_path = write_csv(rows, os.path.join(out_dir, f"eval_{tag}.csv"))
    result = {**metrics, "per_image_csv": csv_path, "n": len(rows),
              "mean_train_count": mean_count, "n_train": len(train_counts)}
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--weights", help="saved model (.keras/.h5) or weights file (.weights.h5)")
    which.add_argument("--dummy", action="store_true",
                       help="use an untrained 1x1-conv model (smoke test)")
    which.add_argument("--baseline", action="store_true",
                       help="predict the mean training count for every image")
    ap.add_argument("--part", default="B", choices=["A", "B"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="CSV/JSON folder (default: %(default)s)")
    ap.add_argument("--fig-dir", default=DEFAULT_FIG_DIR, help="figure folder (default: %(default)s)")
    ap.add_argument("--tag", default=None, help="name used in output files (default: part_X_split)")
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    ap.add_argument("--processed-root", default=DEFAULT_PROCESSED_ROOT)
    ap.add_argument("--no-figures", action="store_true", help="skip the plots")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.baseline:
        res = mean_count_baseline(args.part, args.split, args.out_dir, args.processed_root)
        name = f"Mean-count baseline (mean train count {res['mean_train_count']:.2f})"
    else:
        from src.model_utils import load_model
        model = load_model(args.weights, dummy=args.dummy)
        res = evaluate(model, args.part, args.split, args.out_dir, args.fig_dir, args.tag,
                       args.raw_root, args.processed_root, make_figures=not args.no_figures)
        name = "Untrained 1x1-conv model" if args.dummy else os.path.basename(args.weights)

    print(f"\n{name} on part {args.part} {args.split} ({res['n']} images)")
    print("| Method | MAE | MSE | RMSE |\n|---|---|---|---|")
    print(f"| {name} | {res['mae']:.2f} | {res['mse']:.2f} | {res['rmse']:.2f} |")
    print("(MSE = mean squared count error; RMSE = its square root, reported as 'MSE' in papers)")
    for key in ("per_image_csv", "scatter_png", "best_worst_png"):
        if key in res:
            print(f"{key}: {res[key]}")


if __name__ == "__main__":
    main()
