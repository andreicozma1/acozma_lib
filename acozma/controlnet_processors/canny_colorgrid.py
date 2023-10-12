from PIL import Image, ImageChops, ImageEnhance, ImageOps

from .canny import canny, rand_canny
from .colorgrid import colorgrid, rand_colorgrid
from .utils import processor


def _canny_colorgrid_post(image_bg: Image.Image, image_fg: Image.Image):
    image_bg = image_bg.convert("RGB")
    image_fg = image_fg.convert("RGB")

    image_bg_inv = ImageOps.invert(image_bg)
    image_bg_inv = ImageEnhance.Contrast(image_bg_inv).enhance(factor=10)

    image_fg_new = ImageChops.multiply(image_fg, image_bg_inv)

    img_out = ImageChops.composite(
        image_bg, image_fg_new, ImageOps.invert(image_fg.convert("1"))
    )
    return img_out


@processor
def canny_colorgrid(
    image: Image.Image,
    color_image: Image.Image | None = None,
    **kwargs,
):
    image_bg = colorgrid(color_image or image, **kwargs)
    image_fg = canny(image, **kwargs)

    return _canny_colorgrid_post(image_bg, image_fg)


def rand_canny_colorgrid(
    image: Image.Image,
    color_image: Image.Image | None = None,
    **kwargs,
):
    image_bg, colorgrid_params = rand_colorgrid(color_image or image, **kwargs)
    image_fg, canny_params = rand_canny(image, **kwargs)

    return (
        _canny_colorgrid_post(image_bg, image_fg),
        {**canny_params, **colorgrid_params},
    )
