"""CSRNet model (owner: Ronish).

Contract (docs/CONTRACT.md):
    build_csrnet() -> keras.Model
    input (None, None, 3), output (H/8, W/8, 1) with ReLU.

Front end: the first 10 conv layers of VGG16 (up to block4_conv3, three
max-pools, so the output is 1/8 of the input size).
Back end: dilated 3x3 convs (512, 512, 512, 256, 128, 64; dilation rate 2)
followed by a 1x1 output conv with ReLU.

Example:
    python src/model.py --no-pretrained --size 256
"""

import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BACKEND_FILTERS = (512, 512, 512, 256, 128, 64)


def build_csrnet(pretrained: bool = True) -> keras.Model:
    """Build CSRNet.

    Args:
        pretrained: load ImageNet weights into the VGG16 front end.

    Returns:
        keras.Model mapping (None, None, 3) images to (H/8, W/8, 1) density maps.
        H and W should be multiples of 8 (the loader pads/crops to this).
    """
    vgg = keras.applications.VGG16(include_top=False,
                                   weights="imagenet" if pretrained else None,
                                   input_shape=(None, None, 3))
    x = vgg.get_layer("block4_conv3").output  # 10th conv layer

    init = keras.initializers.RandomNormal(stddev=0.01)
    for i, filters in enumerate(BACKEND_FILTERS):
        x = layers.Conv2D(filters, 3, padding="same", dilation_rate=2,
                          activation="relu", kernel_initializer=init,
                          name=f"backend_conv{i + 1}")(x)
    out = layers.Conv2D(1, 1, activation="relu", kernel_initializer=init,
                        name="density")(x)
    return keras.Model(vgg.input, out, name="csrnet")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="skip downloading ImageNet weights")
    ap.add_argument("--size", type=int, default=256, help="side of a random test input")
    args = ap.parse_args()

    model = build_csrnet(pretrained=not args.no_pretrained)
    model.summary()
    x = tf.random.uniform((1, args.size, args.size, 3))
    print(f"input {tuple(x.shape)} -> output {tuple(model(x).shape)}")


if __name__ == "__main__":
    main()
