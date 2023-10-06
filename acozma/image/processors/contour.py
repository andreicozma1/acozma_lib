import numpy as np
from PIL import Image, ImageOps
from skimage import measure
from skimage.util import img_as_float, img_as_ubyte

from acozma.image.processors.utils import processor


@processor
def contour(image: Image.Image, level: float | list[float] | None = None):
    levels = level if isinstance(level, list) else [level]
    image = image.convert("L")
    image = np.array(image)
    image = img_as_float(image)

    # TODO: Histogram equalization?
    # image = exposure.equalize_hist(image)
    # image = exposure.equalize_adapthist(image, clip_limit=0.03)

    contours = [
        measure.find_contours(
            image,
            l,
            # fully_connected="low",
            # positive_orientation="low",
        )
        for l in levels
    ]
    image = np.zeros_like(image)
    for contour in contours:
        for c in contour:
            c = np.round(c).astype(int)
            image[c[:, 0], c[:, 1]] = 1

    image = img_as_ubyte(image)
    image = Image.fromarray(image)
    # autocontrast ensures we can find the edges
    image = ImageOps.autocontrast(image)
    return image.convert("RGB")


def rand_contour(image: Image.Image):
    # TODO: Implement random contours

    num_levels = np.random.randint(1, 5)

    params = {
        "level": list(np.random.uniform(0.0, 1.0, size=num_levels)),
    }

    res = contour(image, **params)

    return res, params
