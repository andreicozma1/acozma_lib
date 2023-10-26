from enum import Enum

import numpy as np
from PIL import Image, ImageFilter

from .utils import processor


class BlurFuncs(Enum):
    @staticmethod
    def gaussian(image: Image.Image, radius: int):
        return image.filter(ImageFilter.GaussianBlur(radius))

    @staticmethod
    def box(image: Image.Image, radius: int):
        return image.filter(ImageFilter.BoxBlur(radius))

    @staticmethod
    def median(image: Image.Image, radius: int):
        if radius % 2 == 0:
            radius += 1
        return image.filter(ImageFilter.MedianFilter(radius))
    
    GAUSSIAN = (gaussian,)
    BOX = (box,)
    MEDIAN = (median,)
    
    def __call__(self, image: Image.Image, radius: int):
        return self.value[0](image, radius)
    
    

@processor
def blur(
    image: Image.Image,
    func: BlurFuncs,
    radius: int = 5,
    **kwargs,
):
    assert radius >= 0, "expected radius >= 0; got radius={radius}"

    if radius == 0:
        return image
    
    if isinstance(func, str):
        func = BlurFuncs[func]
        
    try:
        image = func(image, radius)
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
    
    params = {
        "func": np.random.choice(BlurFuncs._member_names_),
        # "func": np.random.choice(BlurFuncs),
        "radius": np.random.randint(radius_min, radius_max),
        **kwargs,
    }

    return blur(image, **params), params
