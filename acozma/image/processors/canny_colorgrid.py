from PIL import Image, ImageChops, ImageEnhance, ImageOps

from acozma.image.processors.canny import canny, rand_canny
from acozma.image.processors.colorgrid import colorgrid, rand_colorgrid
from acozma.image.processors.utils import processor


def _canny_colorgrid_post(image_bg: Image.Image, image_fg: Image.Image):
    image_bg_inv = ImageOps.invert(image_bg.convert("L"))
    image_bg_inv = ImageEnhance.Contrast(image_bg_inv).enhance(factor=20)

    image_fg_new = ImageChops.multiply(image_fg.convert("L"), image_bg_inv)

    return ImageChops.composite(image_fg_new, image_bg, image_fg.convert("L")).convert(
        "RGB"
    )


@processor
def canny_colorgrid(image: Image.Image, grid_size=5):
    image_bg = colorgrid(image, grid_size=grid_size)
    image_fg = canny(image)

    return _canny_colorgrid_post(image_bg, image_fg)


def rand_canny_colorgrid(image: Image.Image):
    image_bg, colorgrid_params = rand_colorgrid(image)
    image_fg, canny_params = rand_canny(image)

    return (
        _canny_colorgrid_post(image_bg, image_fg),
        {**canny_params, **colorgrid_params},
    )
