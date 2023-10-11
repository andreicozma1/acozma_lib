import numpy as np
from PIL import Image

from acozma.image.processors.utils import processor

_resampling_modes: list[int] = [
    Image.NEAREST,
    Image.BILINEAR,
    Image.BICUBIC,
    Image.LANCZOS,
]


@processor
def colorgrid(
    image: Image.Image,
    grid_size: int = 5,
    downsample=Image.BICUBIC,
    upsample=Image.NEAREST,
    shuffle: bool = False,
    **kwargs,
):
    assert grid_size >= 5, "grid_size must be >= 5"

    width, height = image.size
    # TODO: Try the effect of different resampling filters
    # NEAREST: Literal[0]
    # LANCZOS: Literal[1]
    # BILINEAR: Literal[2]
    # BICUBIC: Literal[3]
    # BOX: Literal[4]
    # HAMMING: Literal[5]
    image = image.resize((grid_size, grid_size), resample=downsample)
    # shuffle pixels
    if shuffle:
        pixels = np.array(image)
        np.random.shuffle(pixels)
        image = Image.fromarray(pixels)
    image = image.resize((width, height), resample=upsample)
    return image.convert("RGB")


def rand_colorgrid(image: Image.Image):
    params = {
        "grid_size": np.random.randint(5, 25),
        "downsample": np.random.choice(_resampling_modes),
    }

    res = colorgrid(image, **params)

    return res, params
