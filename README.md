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

_TODO (Ronish)._

```bash
python src/train.py --part B
```

Colab: `notebooks/train_colab.ipynb`.

## Evaluate

_TODO (Prajwal)._

```bash
python src/evaluate.py --weights checkpoints/best.keras --part B --split test
```

## Predict

_TODO (Prajwal)._

```bash
python src/predict.py --weights checkpoints/best.keras --image path/to/image.jpg --out outputs/prediction.jpg
```

## Results (ShanghaiTech Part B, test set, 316 images)

MSE is the mean of squared count errors. RMSE is its square root; crowd-counting
papers report RMSE under the name "MSE".

| Method | MAE | MSE | RMSE |
|---|---|---|---|
| Mean-count baseline | _TBD_ | _TBD_ | _TBD_ |
| CSRNet (ours) | _TBD_ | _TBD_ | _TBD_ |
| CSRNet (paper) | 10.6 | n/a | 16.0 |

## Credits

- Y. Li, X. Zhang, D. Chen. *CSRNet: Dilated Convolutional Neural Networks for
  Understanding the Highly Congested Scenes.* CVPR 2018.
- Y. Zhang, D. Zhou, S. Chen, S. Gao, Y. Ma. *Single-Image Crowd Counting via
  Multi-Column Convolutional Neural Network.* CVPR 2016 (ShanghaiTech dataset).
- ShanghaiTech Kaggle mirror by Thai Thien.
- Public CSRNet implementations used for reference only: _list each one we looked at, with a link_.
