# Conclusion

We built a complete crowd-counting pipeline: border-safe Gaussian density maps
whose sums match the head counts, a `tf.data` loader that keeps counts exact
through augmentation and 1/8 sum-pooling, a Keras CSRNet trained on a single
Colab T4 GPU in about 66 minutes, and scripts that evaluate any checkpoint and
draw density heatmaps on new images.

On ShanghaiTech Part B the model reaches MAE 11.78 and RMSE 17.43 on the test
set. That is about 6× better than predicting the mean training count and close
to the published CSRNet result (MAE 10.6, RMSE 16.0).

The most promising next steps are a learning-rate schedule and a larger
validation split, to make training and model selection less noisy, and
training on Part A, which needs the geometry-adaptive density maps that are
already implemented.

# References

1. Y. Li, X. Zhang, D. Chen. CSRNet: Dilated Convolutional Neural Networks for
   Understanding the Highly Congested Scenes. CVPR 2018.
2. Y. Zhang, D. Zhou, S. Chen, S. Gao, Y. Ma. Single-Image Crowd Counting via
   Multi-Column Convolutional Neural Network. CVPR 2016 (ShanghaiTech dataset).
3. K. Simonyan, A. Zisserman. Very Deep Convolutional Networks for Large-Scale
   Image Recognition. ICLR 2015 (VGG-16).
