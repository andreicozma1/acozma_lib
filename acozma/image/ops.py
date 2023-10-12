import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
from skimage import exposure, feature, filters, measure, morphology
from skimage.util import img_as_float, img_as_ubyte


def inner_square_crop(image: Image.Image, shift: float):
    assert 0.0 <= shift <= 1.0, "shift must be between 0 and 1"

    width, height = image.size

    # crop randomly along the longer dimension
    if width != height:
        if width > height:
            diff = width - height
            x = int(diff * shift)
            y = 0
            width = height
        else:
            diff = height - width
            x = 0
            y = int(diff * shift)
            height = width

        image = image.crop((x, y, x + width, y + height))

    assert image.width == image.height, "Internal Error: image must be square"
    return image


def square_crop_center(image: Image.Image):
    return inner_square_crop(image, 0.5)


def square_crop_random(image: Image.Image):
    shift = np.random.uniform(0, 1)
    return inner_square_crop(image, shift)
