# CrowdStrike: Crowd Counting and Density Estimation

Team CrowdStrike: Prashant Panta, Ronish Karki, Prajwal Kunwar.

We predict a density map for a crowd image; the sum of the map is the estimated
head count. The model is our own TensorFlow/Keras implementation of CSRNet
(Li et al., CVPR 2018), trained and evaluated on ShanghaiTech Part B.

## Setup

Python 3.10+.

```bash
git clone https://github.com/Prajwal382/CrowdStrike.git
cd CrowdStrike
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
```

## Data download

The dataset is not in this repo.

1. On Kaggle, search "ShanghaiTech" (dataset by Thai Thien) and choose
   **Download dataset as zip** (about 349 MB). Save it as `archive.zip` in the repo root.
2. Unzip only the dataset folder into `data/`:
   ```bash
   unzip -q archive.zip "ShanghaiTech/*" -d data
   ```
3. Check the folder layout:
   ```bash
   python scripts/check_dataset.py --part B
   ```
4. Build the ground-truth density maps and the fixed 360/40 train/val split (about 6 seconds):
   ```bash
   python scripts/make_density_maps.py --part B
   ```

Maps are written to `data/processed/part_B/{train,val,test}/` and the split to
`data/processed/part_B/splits.json`. On Colab, upload only the zip to Google Drive
and rebuild the maps in the notebook; `data/processed` is 2.1 GB.

## Train

CSRNet (`src/model.py`): VGG16 front end (first 10 conv layers, ImageNet weights)
plus a dilated back end. Training (`src/train.py`) uses the per-map sum of squared
pixel errors, Adam (lr 1e-5), batch size 8, random 1/4-area crops with horizontal
flips, and keeps the checkpoint with the best validation count MAE (early stopping,
patience 20).

```bash
python src/train.py --part B                     # full training
python src/train.py --smoke --no-pretrained      # 2-epoch pipeline check on random tensors
python src/train.py --part B --resume checkpoints/csrnet_partB_best.weights.h5   # continue after a disconnect
python src/train.py --help                       # all options
```

Outputs in `checkpoints/`: `csrnet_partB_best.weights.h5` and `train_log_partB.csv`.
The training curve goes to `outputs/figures/training_curve_partB.png`.

Colab (recommended, T4 GPU): `notebooks/train_colab.ipynb`. Pass
`--out-dir /content/drive/MyDrive/...` so checkpoints survive a disconnect.

Our Part B run: 100 epochs (~40 s each on a T4, about 66 minutes total). Best
validation MAE **11.13** (RMSE 15.63) at epoch 83.

![Training curve](outputs/figures/training_curve_partB.png)

## Evaluate

```bash
# trained model on the Part B test set
python src/evaluate.py --weights checkpoints/csrnet_partB_best.weights.h5 --part B --split test
# mean-count baseline (no model needed)
python src/evaluate.py --baseline
# smoke test with an untrained 1x1-conv model
python src/evaluate.py --dummy --tag untrained
```

`--weights` takes a full saved model (`.keras`/`.h5`) or a weights-only file
(`.weights.h5`, loaded into `build_csrnet()`). It writes:

- `outputs/eval_part_B_test.csv`: per-image `image_id, gt, pred, abs_error`
- `outputs/metrics_part_B_test.json`: MAE, MSE, RMSE
- `outputs/figures/scatter_part_B_test.png`: predicted vs ground-truth counts
- `outputs/figures/best_worst_part_B_test.png`: the 5 best and 5 worst images

Use `--raw-root` / `--processed-root` for Colab/Drive paths and `--tag` to name the output files.

## Predict

```bash
python src/predict.py --weights checkpoints/csrnet_partB_best.weights.h5 --image path/to/image.jpg --out outputs/prediction.jpg
python src/predict.py --weights checkpoints/csrnet_partB_best.weights.h5 --image demo_images/ --out outputs/demo/
```

Prints the count and saves the image with a density heatmap overlay and the count
written on it. Given a folder, it processes every image and also writes `counts.csv`.
Images are shrunk so the longest side is at most 1024 pixels (the Part B width)
before inference; change this with `--max-side` (`0` keeps full size, which can
run out of memory on large photos).

## Results (ShanghaiTech Part B, test set, 316 images)

MSE is the mean of squared count errors. RMSE is its square root; crowd-counting
papers report RMSE under the name "MSE".

| Method | MAE | MSE | RMSE |
|---|---|---|---|
| Mean-count baseline (predicts 120.84, the mean train count) | 71.44 | 9058.92 | 95.18 |
| CSRNet (ours, best-val checkpoint) | 11.78 | 303.83 | 17.43 |
| CSRNet (paper, Li et al. 2018) | 10.6 | n/a | 16.0 |

Validation (40 images) for the same checkpoint: MAE 11.13, RMSE 15.63.
Full write-up: [docs/report/report.md](docs/report/report.md)
(rebuild it from the section files with `python scripts/build_report.py`).

## Credits

- Y. Li, X. Zhang, D. Chen. *CSRNet: Dilated Convolutional Neural Networks for
  Understanding the Highly Congested Scenes.* CVPR 2018.
- Y. Zhang, D. Zhou, S. Chen, S. Gao, Y. Ma. *Single-Image Crowd Counting via
  Multi-Column Convolutional Neural Network.* CVPR 2016 (ShanghaiTech dataset).
- ShanghaiTech Kaggle mirror by Thai Thien.
- Public CSRNet implementations used for reference only: _list each one we looked at, with a link_.
- Demo photos (Unsplash License): Ryoji Iwata, Daniele Colucci, Cheung Yin,
  Fan WU, Dim 7, Viktor Solomonik, Tsuyoshi Kozu and Georg Eiermann on
  [Unsplash](https://unsplash.com). The photos are not stored in this repo;
  only our prediction overlays are.
