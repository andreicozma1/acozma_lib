from PIL import Image, ImageChops, ImageEnhance, ImageOps

from .canny import canny, rand_canny
from .colorgrid import colorgrid, colorgrid_blend_fg, rand_colorgrid
from .utils import processor


@processor
def colorgrid_canny(
    image: Image.Image,
    color_image: Image.Image | None = None,
    **kwargs,
):
    image_bg = colorgrid(color_image or image, **kwargs)
    image_fg = canny(image, **kwargs)

    return colorgrid_blend_fg(image_bg, image_fg)


def rand_colorgrid_canny(
    image: Image.Image,
    color_image: Image.Image | None = None,
    **kwargs,
):
    image_bg, colorgrid_params = rand_colorgrid(color_image or image, **kwargs)
    image_fg, canny_params = rand_canny(image, **kwargs)

    return (
        colorgrid_blend_fg(image_bg, image_fg),
        {**canny_params, **colorgrid_params},
    )
