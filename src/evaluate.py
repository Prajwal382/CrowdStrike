"""Evaluate a crowd-counting model on a ShanghaiTech split (owner: Prajwal).

Contract (docs/CONTRACT.md):
    evaluate(model, part, split) -> dict with keys mae, mse, rmse, per_image_csv
"""


def evaluate(model, part: str, split: str) -> dict:
    """Compute MAE, MSE and RMSE of predicted counts on one split.

    Args:
        model: keras.Model mapping images to (H/8, W/8, 1) density maps.
        part: 'A' or 'B'.
        split: 'train', 'val' or 'test'.

    Returns:
        dict with keys 'mae', 'mse', 'rmse' and 'per_image_csv' (path to a CSV
        with columns image_id, gt, pred, abs_error).
    """
    raise NotImplementedError("evaluate is not implemented yet (Prajwal)")
