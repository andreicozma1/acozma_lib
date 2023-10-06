import cv2
import numpy as np
from PIL import Image, ImageFilter

from acozma.image.processors.utils import processor


@processor
def canny(
    image: Image.Image,
    sigma: float = 1.0,
    low_threshold: int = 100,
    high_threshold: int = 200,
):
    assert sigma > 0.0, "sigma must be positive"
    assert (
        0 <= low_threshold < 255
    ), f"low_threshold must be in [0, 255), got {low_threshold}"
    assert (
        0 < high_threshold <= 255
    ), f"high_threshold must be in (0, 255], got {high_threshold}"

    # TODO: Apply autocontrast?
    # image = ImageOps.autocontrast(image)

    # TODO: Apply histogram equalization?

    # apply gaussian blur
    image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

    image = np.array(image)
    # print(image.min())
    # print(image.max())

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
