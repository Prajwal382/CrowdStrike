"""Count people in any image and save a density heatmap overlay.

The image is read with OpenCV, converted BGR -> RGB, zero-padded on the
bottom/right to a multiple of 8 and ImageNet-normalised exactly as in the
loader. The count is the sum of the predicted density map.

Examples:
    python src/predict.py --weights checkpoints/best.keras --image crowd.jpg --out outputs/crowd_pred.jpg
    python src/predict.py --weights checkpoints/best.keras --image demo_images/ --out outputs/demo/
    python src/predict.py --dummy --image crowd.jpg          # smoke test, untrained model
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.loader import DOWNSAMPLE, normalize_image  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_MAX_SIDE = 1024  # Part B images are 1024 x 768


def limit_size(img: np.ndarray, max_side=DEFAULT_MAX_SIDE) -> np.ndarray:
    """Shrink img so its longest side is at most max_side (None = keep as is)."""
    h, w = img.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return img
    scale = max_side / float(max(h, w))
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def load_rgb(image_path: str) -> np.ndarray:
    """Read an image with OpenCV and return it as uint8 RGB."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pad_to_multiple(img: np.ndarray, multiple: int = DOWNSAMPLE) -> np.ndarray:
    """Zero-pad the bottom and right edges so height and width divide by `multiple`."""
    h, w = img.shape[:2]
    return np.pad(img, ((0, (-h) % multiple), (0, (-w) % multiple), (0, 0)))


def predict_count(model, image_path: str, max_side=None) -> Tuple[float, np.ndarray]:
    """Predict the head count for one image.

    Args:
        model: keras.Model mapping images to (H/8, W/8, 1) density maps.
        image_path: path to any image readable by OpenCV.
        max_side: if set, shrink the image so its longest side is at most this
            many pixels before inference (large photos run out of memory and
            show people at a very different scale from the training images).

    Returns:
        (count, density): count is a float, density is the (H/8, W/8) map
        for the (resized and) padded image.
    """
    rgb = pad_to_multiple(limit_size(load_rgb(image_path), max_side))
    x = normalize_image(rgb)[np.newaxis]
    density = model(x, training=False).numpy()[0, ..., 0]
    return float(density.sum()), density


def make_overlay(image_bgr: np.ndarray, density: np.ndarray, count: float) -> np.ndarray:
    """Blend a JET heatmap of `density` over the image and write the count on it."""
    h, w = image_bgr.shape[:2]
    ph, pw = density.shape[0] * DOWNSAMPLE, density.shape[1] * DOWNSAMPLE
    heat = cv2.resize(density.astype(np.float32), (pw, ph), interpolation=cv2.INTER_LINEAR)
    heat = heat[:h, :w]  # drop the padded border
    peak = float(heat.max())
    heat = (heat / peak * 255.0) if peak > 0 else np.zeros_like(heat)
    heat = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(image_bgr, 0.6, heat, 0.4, 0)

    text = f"Count: {count:.1f}"
    scale = max(0.6, w / 800.0)
    thick = max(1, int(round(2 * scale)))
    org = (int(10 * scale), int(35 * scale))
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def predict_and_save(model, image_path: str, out_path: str, max_side=None) -> float:
    count, density = predict_count(model, image_path, max_side)
    overlay = make_overlay(limit_size(cv2.imread(image_path), max_side), density, count)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not cv2.imwrite(out_path, overlay):
        raise OSError(f"Could not write {out_path}")
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--weights", help="saved model (.keras/.h5) or weights file (.weights.h5)")
    which.add_argument("--dummy", action="store_true",
                       help="use an untrained 1x1-conv model (smoke test)")
    ap.add_argument("--image", required=True, help="an image file, or a folder of images")
    ap.add_argument("--out", default=os.path.join("outputs", "predictions"),
                    help="output image path, or a folder (default: %(default)s)")
    ap.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                    help="shrink images so the longest side is at most this many pixels "
                         "(default: %(default)s, the Part B width; 0 = keep full size)")
    args = ap.parse_args()

    from src.model_utils import load_model
    model = load_model(args.weights, dummy=args.dummy)

    if os.path.isdir(args.image):
        images = sorted(p for p in glob.glob(os.path.join(args.image, "*"))
                        if p.lower().endswith(IMAGE_EXTS))
        if not images:
            sys.exit(f"No images found in {args.image}")
        out_dir = args.out
    else:
        images = [args.image]
        out_dir = None if args.out.lower().endswith(IMAGE_EXTS) else args.out

    rows = []
    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{stem}_pred.jpg") if out_dir else args.out
        count = predict_and_save(model, path, out_path, args.max_side)
        rows.append((os.path.basename(path), count))
        print(f"{os.path.basename(path)}: {count:.1f} people -> {out_path}")

    if out_dir and len(rows) > 1:
        csv_path = os.path.join(out_dir, "counts.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "count"])
            w.writerows((name, f"{c:.2f}") for name, c in rows)
        print(f"counts: {csv_path}")


if __name__ == "__main__":
    main()
