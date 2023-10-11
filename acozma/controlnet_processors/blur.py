import numpy as np
from PIL import Image, ImageFilter

from .utils import processor


@processor
def blur(image: Image.Image, radius: int = 5, **kwargs):
    blur_funcs = [
        ImageFilter.GaussianBlur,
        ImageFilter.BoxBlur,
        ImageFilter.MedianFilter,
    ]
    assert radius >= 0, "radius must be non-negative"

    if radius != 0:
        rand_func = np.random.choice(blur_funcs)

        # if median filter ensure radius is odd
        if rand_func == ImageFilter.MedianFilter and radius % 2 == 0:
            radius += 1

        image = image.filter(rand_func(radius))

    return image.convert("RGB")


def rand_blur(image: Image.Image):
    params = {
        "radius": np.random.randint(0, 25),
    }

    res = blur(image, **params)

    return res, params
