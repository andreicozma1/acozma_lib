import numpy as np
from PIL import Image
from skimage import morphology
from skimage.filters import rank
from skimage.util import img_as_ubyte

from .utils import processor


@processor
def entropy(img: Image.Image, radius: int = 5, **kwargs):
    assert radius >= 3, "radius must be non-negative"

    img_np = np.array(img)
    footprint = morphology.disk(radius)

    # compute entropy for each channel separately
    entropy_cs = []
    for channel in range(img_np.shape[-1]):
        ec = rank.entropy(img_np[..., channel], footprint=footprint)
        # ec = (ec - ec.min()) / (ec.max() - ec.min())
        entropy_cs.append(ec)

    # merge channels back into a single image
    img_np = np.stack(entropy_cs, axis=-1)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    img_np = img_as_ubyte(img_np)
    return Image.fromarray(img_np)


def rand_entropy(image: Image.Image, **kwargs):
    params = {
        "radius": np.random.randint(3, 10),
        **kwargs,
    }

    res = entropy(image, **params)

    return res, params
