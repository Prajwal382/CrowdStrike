"""Train CSRNet on ShanghaiTech (owner: Ronish).

Pixel-wise MSE loss, Adam (lr 1e-5), a callback that tracks count MAE on the
validation split, ModelCheckpoint on best val MAE, EarlyStopping and CSVLogger.

Example:
    python src/train.py --part B --epochs 100
"""

import argparse


def train(part: str = "B", epochs: int = 100, batch_size: int = 8,
          lr: float = 1e-5, out_dir: str = "checkpoints"):
    """Train CSRNet and save the checkpoint with the best validation MAE.

    Args:
        part: 'A' or 'B'.
        epochs: maximum number of epochs (EarlyStopping may stop sooner).
        batch_size: training batch size (val uses batch_size=1).
        lr: Adam learning rate.
        out_dir: where checkpoints and the CSV log are written.

    Returns:
        The keras History object.
    """
    raise NotImplementedError("train is not implemented yet (Ronish)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="B", choices=["A", "B"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out-dir", default="checkpoints")
    args = ap.parse_args()
    train(args.part, args.epochs, args.batch_size, args.lr, args.out_dir)


if __name__ == "__main__":
    main()
