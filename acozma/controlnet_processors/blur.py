
import numpy as np
from PIL import Image, ImageFilter

from .utils import processor


class BlurFuncs:
    @staticmethod
    def gaussian(image: Image.Image, radius: float) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius))

    @staticmethod
    def box(image: Image.Image, radius: int) -> Image.Image:
        return image.filter(ImageFilter.BoxBlur(radius))

    @staticmethod
    def median(image: Image.Image, radius: int) -> Image.Image:
        if radius % 2 == 0:
            radius += 1
        return image.filter(ImageFilter.MedianFilter(radius))


@processor
def blur(
    image: Image.Image,
    func: str = "gaussian",
    radius: int = 5,
    **kwargs,
):
    assert radius >= 0, "expected radius >= 0; got radius={radius}"

    if radius == 0:
        return image

    try:
        image = getattr(BlurFuncs, func.lower())(image, radius)
    except Exception as e:
        print(f"Params: {func}, {radius}")
        raise e

    return image


def rand_blur(
    image: Image.Image,
    radius_bounds: tuple[int, int] = (0, 15),
    **kwargs,
):
    radius_min, radius_max = radius_bounds
    assert radius_min >= 0, "expected radius_min >= 0; got radius_min={radius_min}"
    assert (
        radius_max >= radius_min
    ), "expected radius_max >= radius_min; got radius_max={radius_max}, radius_min={radius_min}"

    # TODO: Scale random radius based on image size. 0-15 is good for 512x512
    func_names = [
        func_name for func_name in dir(BlurFuncs) if not func_name.startswith("_")
    ]
    params = {
        "func": np.random.choice(func_names),
        # "func": np.random.choice(BlurFuncs),
        "radius": np.random.randint(radius_min, radius_max),
        **kwargs,
    }

    return blur(image, **params), params
