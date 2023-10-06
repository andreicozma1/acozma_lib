import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
from skimage import exposure, feature, filters, measure, morphology
from skimage.util import img_as_float, img_as_ubyte


def square_crop(image: Image.Image):
    width, height = image.size

    # print(width, height)
    # crop randomly along the longer dimension
    if width != height:
        if width > height:
            diff = width - height
            x = np.random.randint(0, diff)
            y = 0
            width = height
        else:
            diff = height - width
            x = 0
            y = np.random.randint(0, diff)
            height = width

        image = image.crop((x, y, x + width, y + height))

    assert image.width == image.height, "Internal Error: image must be square"
    return image
