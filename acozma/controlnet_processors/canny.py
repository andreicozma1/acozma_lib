import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .utils import processor


@processor
def canny(
    image: Image.Image,
    sigma: float = 0.5,
    low_threshold: int | None = None,
    high_threshold: int | None = None,
    **kwargs,
):
    assert sigma > 0.0, f"sigma must be positive, got {sigma}"

    # apply gaussian blur
    image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

    # image = ImageOps.autocontrast(image)

    image = np.array(image)

    if low_threshold is None:
        # auto pick based on median
        median = np.median(image)
        low_threshold = int(max(0, (1.0 - sigma) * median))
        print("Auto low_threshold:", low_threshold)

    if high_threshold is None:
        # auto pick based on median
        median = np.median(image)
        high_threshold = int(min(255, (1.0 + sigma) * median))
        print("Auto high_threshold:", high_threshold)

    assert (
        0 <= low_threshold < 255
    ), f"low_threshold must be in [0, 255), got {low_threshold}"
    assert (
        0 < high_threshold <= 255
    ), f"high_threshold must be in (0, 255], got {high_threshold}"

    # TODO: Auto thresholding for low_threshold and high_threshold
    image = cv2.Canny(image, low_threshold, high_threshold)

    image = Image.fromarray(image)

    return image.convert("RGB")


def rand_canny(image: Image.Image):
    low_threshold: int = np.random.randint(0, 200)

    params = {
        "sigma": np.random.uniform(0.6, 2.4),
        "low_threshold": low_threshold,
        "high_threshold": np.random.randint(254 - low_threshold, 255),
    }

    res = canny(image, **params)

    return res, params
