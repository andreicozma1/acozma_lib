from PIL import Image

from .colorgrid import colorgrid, colorgrid_blend_fg, rand_colorgrid
from .contour import contour, rand_contour
from .utils import processor


@processor
def colorgrid_contour(
    image: Image.Image,
    color_image: Image.Image | None = None,
    **kwargs,
):
    if color_image is None:
        color_image = image
    image_bg = colorgrid(color_image, **kwargs)
    image_fg = contour(image, **kwargs)

    return colorgrid_blend_fg(image_bg, image_fg)


def rand_colorgrid_contour(
    image: Image.Image,
    color_image: Image.Image | None = None,
    **kwargs,
):
    image_bg, colorgrid_params = rand_colorgrid(color_image or image, **kwargs)
    image_fg, canny_params = rand_contour(image, **kwargs)

    return (
        colorgrid_blend_fg(image_bg, image_fg),
        {**canny_params, **colorgrid_params},
    )
