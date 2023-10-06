from PIL import Image, ImageChops, ImageEnhance, ImageOps

from acozma.image.processors.canny import canny, rand_canny
from acozma.image.processors.colorgrid import colorgrid, rand_colorgrid
from acozma.image.processors.utils import processor


def _canny_colorgrid_post(image_canny: Image.Image, image_colorgrid: Image.Image):
    image_cg20_inv = ImageOps.invert(image_colorgrid.convert("L"))
    image_cg20_inv = ImageEnhance.Contrast(image_cg20_inv).enhance(factor=20)
    image_canny_new = ImageChops.multiply(image_canny.convert("L"), image_cg20_inv)

    return ImageChops.composite(
        image_canny_new, image_colorgrid, image_canny.convert("L")
    ).convert("RGB")


@processor
def canny_colorgrid(image: Image.Image, grid_size=5):
    image_canny = canny(image)
    image_colorgrid = colorgrid(image, grid_size=grid_size)

    return _canny_colorgrid_post(image_canny, image_colorgrid)


@processor
def rand_canny_colorgrid(image: Image.Image):
    image_canny, canny_params = rand_canny(image)
    image_colorgrid, colorgrid_params = rand_colorgrid(image)

    return (
        _canny_colorgrid_post(image_canny, image_colorgrid),
        {**canny_params, **colorgrid_params},
    )
