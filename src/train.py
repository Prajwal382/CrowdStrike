"""Train CSRNet on ShanghaiTech (owner: Ronish).

Pixel-wise MSE loss on density_1_8, Adam (lr 1e-5), a callback that tracks
count MAE on the validation split, ModelCheckpoint on best val MAE,
EarlyStopping and CSVLogger.

Examples:
    python src/train.py --part B --epochs 100
    python src/train.py --smoke --no-pretrained          # random tensors, no dataset needed
    python src/train.py --out-dir /content/drive/MyDrive/crowdstrike/checkpoints   # Colab
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from src.model import build_csrnet  # noqa: E402


def density_mse(y_true, y_pred):
    """Pixel-wise squared error, summed over each density map, averaged over the batch.

    Summing over pixels (as in the CSRNet paper) instead of averaging keeps the
    gradients large enough that the model does not collapse to all zeros at lr 1e-5.
    """
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2, 3]))


class CountMAE(keras.callbacks.Callback):
    """Count MAE and RMSE on the validation set at the end of every epoch.

    Adds logs['val_mae'] and logs['val_rmse'] so ModelCheckpoint, EarlyStopping
    and CSVLogger can use them. It must come first in the callbacks list.
    """

    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        errors = []
        for images, densities in self.val_ds:
            pred = self.model(images, training=False)
            pred_count = tf.reduce_sum(pred, axis=[1, 2, 3])
            gt_count = tf.reduce_sum(densities, axis=[1, 2, 3])
            errors.append((pred_count - gt_count).numpy())
        errors = np.concatenate(errors)
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        if logs is not None:
            logs["val_mae"] = mae
            logs["val_rmse"] = rmse
        print(f" - val_mae: {mae:.2f} - val_rmse: {rmse:.2f}")


def smoke_datasets(batch_size, crop=256, n_train=16, n_val=4):
    """Random tensors with the same shapes as the real loader's output."""
    def make(n):
        imgs = tf.random.normal((n, crop, crop, 3))
        dens = tf.random.uniform((n, crop // 8, crop // 8, 1), maxval=0.05)
        return tf.data.Dataset.from_tensor_slices((imgs, dens))
    return make(n_train).batch(batch_size), make(n_val).batch(1)


def plot_curves(csv_path, out_path):
    """Save training loss and validation MAE/RMSE curves from the CSV log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    epochs = df["epoch"] + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, df["loss"])
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Density squared error")
    axes[1].plot(epochs, df["val_mae"], label="val MAE")
    axes[1].plot(epochs, df["val_rmse"], label="val RMSE")
    axes[1].set(title="Validation count error", xlabel="Epoch", ylabel="People")
    axes[1].legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curve to {out_path}")


def train(part: str = "B", epochs: int = 100, batch_size: int = 8,
          lr: float = 1e-5, out_dir: str = "checkpoints",
          raw_root: str = None, processed_root: str = None, crop_size=None,
          patience: int = 20, resume: str = None, pretrained: bool = True,
          smoke: bool = False, figure_dir: str = os.path.join("outputs", "figures")):
    """Train CSRNet and save the checkpoint with the best validation MAE.

    Args:
        part: 'A' or 'B'.
        epochs: maximum number of epochs (EarlyStopping may stop sooner).
        batch_size: training batch size (val uses batch_size=1).
        lr: Adam learning rate.
        out_dir: where checkpoints and the CSV log are written.
        raw_root, processed_root: data locations (override on Colab/Drive).
        crop_size: training crop (h, w); None uses the loader's 1/4-area crop.
        patience: EarlyStopping patience on val MAE, in epochs.
        resume: optional .weights.h5 file to continue training from.
        pretrained: load ImageNet weights into the VGG16 front end.
        smoke: train 2 epochs on random tensors instead of the dataset.
        figure_dir: where the training-curve plot is saved.

    Returns:
        The keras History object.
    """
    if smoke:
        epochs = min(epochs, 2)
        train_ds, val_ds = smoke_datasets(batch_size)
    else:
        from src.data.loader import get_dataset
        roots = {}
        if raw_root:
            roots["raw_root"] = raw_root
        if processed_root:
            roots["processed_root"] = processed_root
        train_ds = get_dataset(part, "train", batch_size, crop_size=crop_size, **roots)
        val_ds = get_dataset(part, "val", 1, **roots)

    model = build_csrnet(pretrained=pretrained)
    if resume:
        model.load_weights(resume)
        print(f"Resumed from {resume}")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=density_mse)

    os.makedirs(out_dir, exist_ok=True)
    tag = f"part{part}" + ("_smoke" if smoke else "")
    ckpt_path = os.path.join(out_dir, f"csrnet_{tag}_best.weights.h5")
    csv_path = os.path.join(out_dir, f"train_log_{tag}.csv")

    callbacks = [
        CountMAE(val_ds),  # first, so the callbacks below can see val_mae
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_mae", mode="min",
                                        save_best_only=True, save_weights_only=True,
                                        verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_mae", mode="min",
                                      patience=patience, verbose=1),
        keras.callbacks.CSVLogger(csv_path, append=bool(resume)),
    ]

    history = model.fit(train_ds, epochs=epochs, callbacks=callbacks, verbose=2)

    plot_curves(csv_path, os.path.join(figure_dir, f"training_curve_{tag}.png"))
    print(f"Best weights: {ckpt_path}")
    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="B", choices=["A", "B"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out-dir", default="checkpoints",
                    help="checkpoints and CSV log (use a Drive path on Colab)")
    ap.add_argument("--raw-root", default=None, help="default: data/ShanghaiTech")
    ap.add_argument("--processed-root", default=None, help="default: data/processed")
    ap.add_argument("--crop-size", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="training crop; default is half height x half width")
    ap.add_argument("--patience", type=int, default=20, help="EarlyStopping patience")
    ap.add_argument("--resume", default=None, help=".weights.h5 file to resume from")
    ap.add_argument("--figure-dir", default=os.path.join("outputs", "figures"))
    ap.add_argument("--no-pretrained", action="store_true",
                    help="don't load ImageNet VGG16 weights")
    ap.add_argument("--smoke", action="store_true",
                    help="train 2 epochs on random tensors to test the pipeline")
    args = ap.parse_args()

    train(part=args.part, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          out_dir=args.out_dir, raw_root=args.raw_root,
          processed_root=args.processed_root,
          crop_size=tuple(args.crop_size) if args.crop_size else None,
          patience=args.patience, resume=args.resume,
          pretrained=not args.no_pretrained, smoke=args.smoke,
          figure_dir=args.figure_dir)


if __name__ == "__main__":
    main()
