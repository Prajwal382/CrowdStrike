# Interface contract

This file is the agreement that lets the data, model and evaluation work happen
in parallel. Change it only through a PR that all three of us see.

## Data on disk

| What | Where | Format |
|---|---|---|
| Raw dataset | `data/ShanghaiTech/part_B[_final]/{train_data,test_data}/{images,ground-truth}/` | `IMG_<n>.jpg`, `GT_IMG_<n>.mat` |
| Density maps | `data/processed/part_B/{train,val,test}/<image_id>.npy` | float32, full image resolution (H, W) |
| Split file | `data/processed/part_B/splits.json` | fixed 360/40 train/val split, seed 42 |

- Head points in the `.mat` files are at `image_info[0,0][0,0][0]`, as (x, y).
- Part B: fixed Gaussian, sigma = 15. Part A (stretch): geometry-adaptive sigma,
  k = 3 nearest neighbours, beta = 0.3.
- **Invariant:** `density.sum()` equals the annotated head count, relative error < 1e-3.
- `val` is 10% of the official `train_data`, so `val` images are read from `train_data/images/`.

## Functions

### `src/data/loader.py` (Prashant)

```python
get_dataset(part, split, batch_size, crop_size=None,
            raw_root="data/ShanghaiTech", processed_root="data/processed",
            shuffle=None, seed=None) -> tf.data.Dataset   # yields (image, density_1_8)
list_samples(part, split, raw_root=..., processed_root=...) -> [(image_path, density_path), ...]  # sorted
normalize_image(rgb) -> float32   # ImageNet mean/std, works on NumPy and tf tensors
```

- `image`: float32 `(B, H, W, 3)`, RGB, ImageNet mean/std normalised.
- `density_1_8`: float32 `(B, H/8, W/8, 1)`, made by sum-pooling 8x8 blocks, so its
  sum is still the head count.
- Train: random crop of 1/4 of the image area (half height x half width), random
  horizontal flip, shuffled.
- Val/test: whole image, zero-padded on the bottom/right to a multiple of 8, **not
  shuffled**, in the same order as `list_samples(part, split)`.
- **Use `batch_size=1` for val and test** (images have different sizes).
- `raw_root=` and `processed_root=` let Colab point at Google Drive paths.

### `src/model.py` (Ronish)

```python
build_csrnet() -> keras.Model
```

- Input `(None, None, 3)`, output `(H/8, W/8, 1)` with ReLU.
- Front end: first 10 conv layers of Keras VGG16 (ImageNet weights).
- Back end: dilated 3x3 convs 512, 512, 512, 256, 128, 64 (dilation 2), then a 1x1 conv.

### `src/train.py` (Ronish)

- Pixel-wise MSE loss on `density_1_8`, Adam lr 1e-5.
- Callback that computes count MAE on val; `ModelCheckpoint` on best val MAE,
  `EarlyStopping`, `CSVLogger`.
- Checkpoints go in `checkpoints/` (gitignored). Total training budget: about 4 GPU-hours on Colab.

### `src/evaluate.py` (Prajwal)

```python
evaluate(model, part, split) -> {"mae": float, "mse": float, "rmse": float, "per_image_csv": str}
```

- Per-image counts: `pred = model(x).numpy().sum()`, `gt = y.numpy().sum()`.
- `MAE = mean(|pred - gt|)`, `MSE = mean((pred - gt)^2)`, `RMSE = sqrt(MSE)`.
  Crowd-counting papers call RMSE "MSE", so our tables show both, labelled.
- Per-image CSV columns: `image_id, gt, pred, abs_error`.

### `src/predict.py` (Prajwal)

```python
predict_count(model, image_path) -> (count: float, density: np.ndarray)
```

- Read with OpenCV, BGR -> RGB, zero-pad bottom/right to a multiple of 8,
  `normalize_image`, run the model. `count = density.sum()`.

## Rules

- Every script runs from the command line with `argparse` and has `--help`.
- Never commit the dataset, `data/processed/` (2.1 GB) or trained weights.
- On Colab, upload only the dataset zip to Drive and rebuild the density maps in
  the notebook (`scripts/make_density_maps.py --part B` takes a few seconds).
