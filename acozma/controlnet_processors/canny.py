import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .blur import BlurFuncs
from .utils import processor


def _verify_tresholds(low_threshold, high_threshold):
    assert (
        0 <= low_threshold <= 255
    ), f"expected 0 <= low_threshold <= 255; got low_threshold={low_threshold}"
    assert (
        0 <= high_threshold <= 255
    ), f"expected 0 <= high_threshold <= 255; got high_threshold={high_threshold}"
    assert (
        low_threshold < high_threshold
    ), f"expected low_threshold < high_threshold; got low_threshold={low_threshold}, high_threshold={high_threshold}"

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
    # image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    image = BlurFuncs.GAUSSIAN(image, sigma)

    # image = ImageOps.autocontrast(image)

    image = np.array(image)

    # Auto thresholding for low_threshold and high_threshold
    if low_threshold is None:
        # auto pick based on median
        median = np.median(image)
        low_threshold = int(max(0, (1.0 - sigma) * median))
        print("low_threshold:", low_threshold, "(auto)")

    if high_threshold is None:
        # auto pick based on median
        median = np.median(image)
        high_threshold = int(min(255, (1.0 + sigma) * median))
        print("high_threshold:", high_threshold, "(auto)")

    _verify_tresholds(low_threshold, high_threshold)

    image = cv2.Canny(image, low_threshold, high_threshold)

    image = Image.fromarray(image)

    return image.convert("RGB")


def rand_canny(
    image: Image.Image,
    threshold_bounds: tuple[int, int] = (0, 255),
    **kwargs,
):
    threshold_min, threshold_max = threshold_bounds
    _verify_tresholds(threshold_min, threshold_max)

    # TODO: Fix 200 here
    low_threshold: int = np.random.randint(threshold_min, threshold_max - 1)

    params = {
        # TODO: Add sigma bounds
        "sigma": np.random.uniform(0.6, 2.4),
        "low_threshold": low_threshold,
        "high_threshold": np.random.randint(low_threshold + 1, threshold_max),
        **kwargs,
    }

    res = canny(image, **params)

    return res, params
