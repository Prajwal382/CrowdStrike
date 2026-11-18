# CrowdStrike: Crowd Counting and Density Estimation

Computer Vision course project (Prashant Panta, Ronish Karki, Prajwal Kunwar).
Predict a density map for a crowd image; its sum is the head count.
CSRNet in TensorFlow/Keras on ShanghaiTech Part B (Part A is a stretch goal).
Prefer simple, working code over ambitious code.

## Ownership

| Module | Owner |
|---|---|
| `src/data/` (density.py, loader.py, paths.py), `scripts/`, `tests/` | Prashant |
| `src/model.py`, `src/train.py`, `notebooks/train_colab.ipynb` | Ronish |
| `src/evaluate.py`, `src/predict.py`, README, integration | Prajwal |

Don't implement or commit a teammate's module for them.

## Interface contract (full version: docs/CONTRACT.md)

- Density maps: float32 `.npy` at full resolution in
  `data/processed/part_B/{train,val,test}/<image_id>.npy`; `density.sum()` equals the head count (rel. error < 1e-3).
- Fixed 360/40 train/val split (seed 42) in `data/processed/part_B/splits.json`.
- `get_dataset(part, split, batch_size, crop_size=None, raw_root=..., processed_root=...)`
  yields `(image, density_1_8)`: RGB float32 ImageNet-normalised image, and the map
  sum-pooled 8x8 so the count is unchanged. Use `batch_size=1` for val/test; the test
  split is not shuffled and follows `list_samples(part, split)` order.
  `raw_root=` / `processed_root=` are for Colab/Drive paths.
- `normalize_image(rgb)` in `src/data/loader.py` is the one normalisation used everywhere.
- `build_csrnet() -> keras.Model`: input `(None, None, 3)`, output `(H/8, W/8, 1)`, ReLU.
- `evaluate(model, part, split) -> dict` with keys `mae`, `mse`, `rmse`, `per_image_csv`.
- `predict_count(model, image_path) -> (count: float, density: np.ndarray)`.
- Report MSE and RMSE separately and label them; papers call RMSE "MSE".

## Data

1. Kaggle: search "ShanghaiTech" (by Thai Thien), "Download dataset as zip" (349 MB).
2. `unzip -q archive.zip "ShanghaiTech/*" -d data`
3. `python scripts/check_dataset.py --part B`
4. `python scripts/make_density_maps.py --part B`
On Colab, upload only the zip to Drive (not `data/processed`, which is 2.1 GB) and rebuild the maps.

## Rules for code

- Every script uses `argparse` and has `--help`.
- Never commit `data/`, checkpoints, `*.h5`, `*.keras` or `outputs/` (except `outputs/figures/`).
- Run tests with `python -m pytest tests -q`.

## Git rules (strict)

- No AI attribution anywhere: no Co-Authored-By trailers, no "Generated with" lines,
  no AI mentions in commit messages, PR titles/descriptions, code comments or the README.
  `.claude/settings.json` turns attribution off; keep it that way.
- Never set or override the git author or committer and never use `--author`.
  Commits use the local git config of whoever is committing.
- Branches are named `<firstname>/<task>` (e.g. `ronish/model`). Merge into `main`
  only through a PR reviewed by one other teammate.
- Small commits, imperative mood: "Add Gaussian density map generation for Part B".
- Ask before anything destructive: force pushes, deleting branches, rewriting history.
