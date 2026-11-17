"""Check that the ShanghaiTech dataset was downloaded and unpacked correctly.

Example:
    python scripts/check_dataset.py --part B
    python scripts/check_dataset.py --part A --raw-root /content/drive/MyDrive/ShanghaiTech
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.density import normalize_part  # noqa: E402
from src.data.paths import DEFAULT_RAW_ROOT, find_part_dir  # noqa: E402

EXPECTED = {"part_A": {"train_data": 300, "test_data": 182},
            "part_B": {"train_data": 400, "test_data": 316}}


def check(raw_root: str, part: str) -> bool:
    name = normalize_part(part)
    part_dir = find_part_dir(raw_root, part)
    if part_dir is None:
        print(f"[FAIL] no {name}/ or {name}_final/ folder with train_data/ under {raw_root}")
        return False
    print(f"[ok]   found {part_dir}")
    ok = True
    for sub, n_expected in EXPECTED[name].items():
        img_dir = os.path.join(part_dir, sub, "images")
        gt_dir = os.path.join(part_dir, sub, "ground-truth")
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        missing_gt = [p for p in imgs if not os.path.exists(
            os.path.join(gt_dir, "GT_" + os.path.splitext(os.path.basename(p))[0] + ".mat"))]
        status = "ok" if len(imgs) == n_expected and not missing_gt else "FAIL"
        ok &= status == "ok"
        print(f"[{status:4}] {sub}: {len(imgs)} images (expected {n_expected}), "
              f"{len(missing_gt)} missing .mat files")
        for p in missing_gt[:5]:
            print(f"         missing GT for {os.path.basename(p)}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT,
                    help="folder containing part_A/ and part_B/ (default: %(default)s)")
    ap.add_argument("--part", default="B", choices=["A", "B"], help="dataset part")
    args = ap.parse_args()
    sys.exit(0 if check(args.raw_root, args.part) else 1)


if __name__ == "__main__":
    main()
