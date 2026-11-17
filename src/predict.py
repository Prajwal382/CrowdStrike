"""Count people in any image and draw a density heatmap overlay (owner: Prajwal).

Contract (docs/CONTRACT.md):
    predict_count(model, image_path) -> (count: float, density: np.ndarray)
"""


def predict_count(model, image_path: str):
    """Predict the head count for one image.

    Args:
        model: keras.Model mapping images to (H/8, W/8, 1) density maps.
        image_path: path to any RGB image readable by OpenCV.

    Returns:
        (count, density): count is a float, density is the (H/8, W/8) map.
    """
    raise NotImplementedError("predict_count is not implemented yet (Prajwal)")
