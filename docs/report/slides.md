# Slides outline: Crowd Counting with CSRNet

About 10 slides, roughly 1 minute each. Each owner presents their own part.

---

## 1. Title (Prajwal)
- Crowd Counting and Density Estimation with CSRNet
- Team CrowdStrike: Prashant Panta, Ronish Karki, Prajwal Kunwar

## 2. Problem and approach (Prajwal)
- Goal: count people in a crowd image
- Detection breaks down when people overlap
- Predict a density map instead; its sum is the count
- Figure: one test image next to its density map

## 3. Dataset (Prashant)
- ShanghaiTech Part B: 716 street-scene images, 768×1024, one point per head
- Split: 360 train / 40 val (fixed seed 42) / 316 test
- 9 to 576 people per image, about 120 on average
- Figure: `outputs/figures/count_hist_part_B.png`

## 4. Ground-truth density maps (Prashant)
- Each head → Gaussian kernel (σ = 15 px) that sums to 1
- Border-safe: kernels clipped and renormalised at the image edge
- Map sum = head count; worst error 7.4 × 10⁻⁷ over all 716 images
- Figure: `outputs/figures/density_examples.png`

## 5. Input pipeline (Prashant)
- Train: random 1/4-area crop (384×512) + horizontal flip
- ImageNet normalisation for the VGG-16 front end
- 8×8 sum-pooling → 1/8-resolution target, count stays exact
- Figure: `outputs/figures/pipeline_steps_part_B.png`

## 6. Model: CSRNet (Ronish)
- Front end: first 10 conv layers of VGG-16 (ImageNet weights)
- Back end: 6 dilated 3×3 convs (dilation 2) + 1×1 output conv
- Fully convolutional: any image size in, 1/8-resolution density map out

## 7. Training (Ronish)
- Summed pixel-wise squared error, Adam lr 1e-5, batch 8
- Checkpoint on best validation count MAE
- 100 epochs, about 66 min on a Colab T4
- Figure: `outputs/figures/training_curve_partB.png`

## 8. Results (Prajwal)
| Method | MAE | MSE | RMSE |
|---|---:|---:|---:|
| Mean-count baseline | 71.44 | 9058.92 | 95.18 |
| CSRNet (ours) | 11.78 | 303.83 | 17.43 |
| CSRNet (paper) | 10.6 | n/a | 16.0 |
- About 6× lower MAE than the baseline, within 1.2 of the paper
- Figure: predicted vs ground-truth scatter

## 9. Where it works and where it fails (Prajwal)
- Figure: 5 best / 5 worst test images
- Figure: 2–3 overlays on new Unsplash photos

## 10. Limitations and next steps (all)
- Part B only; small validation split makes model selection noisy
- Next: learning-rate schedule, Part A with adaptive kernels
- Questions
