# Dataset and Preprocessing (draft — Prashant)

## Dataset
We use the ShanghaiTech Crowd Counting Dataset (Zhang et al., CVPR 2016).
Each image comes with a `.mat` file listing one (x, y) point per annotated head.

| Part | Train | Test | Typical scene | Resolution |
|------|------:|-----:|---------------|------------|
| B (required) | 400 | 316 | street scenes, sparser crowds | 768 × 1024 |
| A (stretch)  | 300 | 182 | dense crowds from the web | varies |

We hold out 10% of the official training images as a validation split
(Part B: 360 train / 40 val) with a fixed random seed (42), so every teammate
and every Colab run uses the same split. The ids are saved in
`data/processed/part_B/splits.json`. The test set is used only for final evaluation.

## Ground-truth density maps
Counting is framed as density regression. Each head point is replaced by a
2-D Gaussian kernel that integrates to 1, so the integral (sum) of the map
equals the head count.

- **Part B:** fixed kernel, σ = 15 px. Part B crowds are fairly uniform in
  scale, so a fixed σ works well (as in CSRNet).
- **Part A:** geometry-adaptive kernel. σᵢ = β · d̄ᵢ, where d̄ᵢ is the mean
  distance to the k = 3 nearest heads (via `scipy.spatial.KDTree`) and β = 0.3.
  Denser regions get narrower kernels.

Kernels are truncated at 3σ. Heads near the border would normally lose part
of their mass outside the image, which lowers the count. We clip each kernel
to the image and renormalise it, so every head still contributes exactly 1.
The generation script checks that every map satisfies
|sum − count| / count < 10⁻³ and refuses to save a map that fails.
Maps are stored as float32 `.npy` files at full image resolution.

## Input pipeline (`tf.data`)
- **Training:** random crop of 1/4 of the image area (half height × half
  width, i.e. 384 × 512 on Part B), random horizontal flip (p = 0.5),
  applied identically to the image and its density map.
- **Validation/test:** full image, zero-padded on the bottom and right to a
  multiple of 8. Padding adds no density, so counts are unchanged.
- **Normalisation:** RGB scaled to [0, 1], then standardised with the ImageNet
  mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225) to match the
  pretrained VGG-16 front end.
- **Target resolution:** CSRNet outputs a map at 1/8 resolution. We
  downsample the ground truth by **sum-pooling** 8 × 8 blocks, not by
  resizing. Sum-pooling preserves the total count exactly, while bilinear
  resizing would scale it by 1/64.

## Verification
- Unit tests check the count invariant, including heads on or outside
  the image border, and that the split is deterministic and disjoint. They
  also check loader shapes and dtypes, and that the 1/8 targets keep the count.
- A test on 20 random real Part B images checks that each density sum
  matches the number of annotated heads.
- `outputs/figures/density_examples.png` shows 6 image/density pairs.

## Slide bullets
- ShanghaiTech Part B: 400 train (360 + 40 val) / 316 test, 768×1024
- Head points → Gaussian density map (σ = 15); sum = count
- Border-safe kernels: count error < 0.1% on every image
- Augmentation: 1/4-area random crop + horizontal flip
- 8×8 sum-pooling → 1/8 target that keeps the count
