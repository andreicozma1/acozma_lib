import numpy as np
from PIL import Image, ImageOps
from skimage import feature
from skimage.util import img_as_float, img_as_ubyte

from .utils import processor


@processor
def hog(image: Image.Image, orientations: int = 4, pixels_per_cell: int = 9, **kwargs):
    # histogram of oriented gradients
    assert orientations >= 4, "orientations must be at least 4"
    assert pixels_per_cell >= 5, "pixels_per_cell must be at least 5"
    image = np.array(image)
    image = img_as_float(image)

    # TODO: Histogram equalization?
    # image = exposure.equalize_hist(image)
    # image = exposure.equalize_adapthist(image, clip_limit=0.03)

    _, hog_image = feature.hog(
        image,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(1, 1),
        visualize=True,
        channel_axis=-1,
    )

    hog_image = img_as_ubyte(hog_image)
    hog_image = Image.fromarray(hog_image)
    hog_image = ImageOps.autocontrast(hog_image)
    return hog_image.convert("RGB")


def rand_hog(image: Image.Image, **kwargs):
    params = {
        "orientations": np.random.randint(4, 9),
        "pixels_per_cell": np.random.randint(5, 12),
        **kwargs,
    }

    res = hog(image, **params)

    return res, params
