import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageOps

from .utils import processor

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
    # return image.convert("RGB")
    return image


def rand_colorgrid(
    image: Image.Image,
    grid_size_bounds: tuple[int, int] = (5, 25),
    **kwargs,
):
    grid_size_min, grid_size_max = grid_size_bounds
    assert grid_size_min >= 5, "grid_size_min must be >= 5"
    assert grid_size_max >= grid_size_min, "grid_size_max must be >= grid_size_min"

    params = {
        "grid_size": np.random.randint(grid_size_min, grid_size_max),
        "downsample": np.random.choice(_resampling_modes),
        # TODO: Add kwargs to the rest of the functions
        **kwargs,
    }

    res = colorgrid(image, **params)

    return res, params


def _colorgrid_blend_fg_v1(image_bg: Image.Image, image_fg: Image.Image):
    image_bg = image_bg.convert("RGB")
    image_fg = image_fg.convert("RGB")

    image_bg_inv = ImageOps.invert(image_bg)
    image_bg_inv = ImageEnhance.Contrast(image_bg_inv).enhance(factor=10)

    image_fg_new = ImageChops.multiply(image_fg, image_bg_inv)

    return ImageChops.composite(
        image_bg, image_fg_new, ImageOps.invert(image_fg.convert("1"))
    )

def _colorgrid_blend_fg_v2(image_bg: Image.Image, image_fg: Image.Image):
    image_bg = image_bg.convert("CMYK")
    image_bg_c, image_bg_m, image_bg_y, _ = image_bg.split()

    img_out = Image.merge(
        "CMYK", (image_bg_c, image_bg_m, image_bg_y, image_fg.convert("L"))
    )
    img_out = img_out.convert("RGB")
    return img_out


def colorgrid_blend_fg(image_bg: Image.Image, image_fg: Image.Image):
    return _colorgrid_blend_fg_v2(image_bg, image_fg)
