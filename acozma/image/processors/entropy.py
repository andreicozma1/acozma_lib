import numpy as np
from PIL import Image
from skimage import morphology
from skimage.filters import rank
from skimage.util import img_as_ubyte

from acozma.image.processors.utils import processor


@processor
def entropy(img, radius: int = 5):
    assert radius >= 3, "radius must be non-negative"

    img = np.array(img)
    footprint = morphology.disk(radius)

    # compute entropy for each channel separately
    entropy_cs = []
    for channel in range(img.shape[-1]):
        ec = rank.entropy(img[..., channel], footprint=footprint)
        # ec = (ec - ec.min()) / (ec.max() - ec.min())
        entropy_cs.append(ec)

    # merge channels back into a single image
    img = np.stack(entropy_cs, axis=-1)
    img = (img - img.min()) / (img.max() - img.min())

    img = img_as_ubyte(img)
    img = Image.fromarray(img)
    # img = ImageOps.autocontrast(img)
    return img


def rand_entropy(image: Image.Image):
    params = {
        "radius": np.random.randint(3, 15),
    }

    res = entropy(image, **params)

    return res, params
