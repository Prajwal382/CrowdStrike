"""CSRNet model (owner: Ronish).

Contract (docs/CONTRACT.md):
    build_csrnet() -> keras.Model
    input (None, None, 3), output (H/8, W/8, 1) with ReLU.
"""

from tensorflow import keras


def build_csrnet() -> keras.Model:
    """Build CSRNet.

    Front end: the first 10 conv layers of Keras VGG16 with ImageNet weights.
    Back end: dilated 3x3 convs (512, 512, 512, 256, 128, 64; dilation rate 2)
    followed by a 1x1 output conv with ReLU.

    Returns:
        keras.Model mapping (None, None, 3) images to (H/8, W/8, 1) density maps.
    """
    raise NotImplementedError("build_csrnet is not implemented yet (Ronish)")
