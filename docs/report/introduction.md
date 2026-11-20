# Introduction

Crowd counting estimates how many people appear in an image. It matters for
public safety, event management and urban planning, where counting by hand is
slow and detecting every person fails once people overlap in dense scenes.

We follow the density-estimation approach: instead of detecting people, a
network predicts a **density map** whose values sum to the number of people in
the image. This handles occlusion and perspective better than detection,
because the model only has to say how much "person" is in each region.

We implement **CSRNet** (Li, Zhang and Chen, CVPR 2018) in TensorFlow/Keras and
train it on **ShanghaiTech Part B**. The project has three parts, each owned by
one team member:

1. **Data pipeline** (Prashant Panta): ground-truth density maps, validation
   split and the `tf.data` loader.
2. **Model and training** (Ronish Karki): CSRNet and training on Google Colab.
3. **Evaluation and inference** (Prajwal Kunwar): MAE/MSE/RMSE evaluation, a
   mean-count baseline, and an inference script that draws a density heatmap
   on any image.

On the 316 Part B test images our model reaches **MAE 11.78** and
**RMSE 17.43**, compared with 71.44 / 95.18 for a mean-count baseline and
10.6 / 16.0 reported in the CSRNet paper.
