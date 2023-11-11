from enum import Enum
from PIL import Image
from torchvision.transforms import v2


class ControlNetProcessorMode(Enum):
    TRAIN = 0
    TEST = 1


def processor(func):
    def wrapper(image: Image.Image, *args, **kwargs):
        assert isinstance(
            image, Image.Image
        ), "image must be a PIL image. Got {}".format(type(image))
        # assert image.mode == "RGB", "Input image must be an RGB image"
        res = func(image, *args, **kwargs)
        assert isinstance(
            res, Image.Image
        ), "Internal Error: processor must return a PIL image. Got {}".format(type(res))
        # assert (
        #     res.mode == "RGB"
        # ), "Internal Error: processor must return an RGB image. Got {}".format(res.mode)
        return res

    return wrapper


def params_to_str(params: dict):
    return ",".join(
        f"{k}={round(v, 2) if isinstance(v, float) else v}" for k, v in params.items()
    )


def rand_colorjitter(image: Image.Image, **kwargs):
    params = {
        "hue": (-0.5, 0.5),
        "saturation": (0.5, 1.5),
        "brightness": (0.5, 1.5),
        "contrast": (0.5, 1.5),
        **kwargs,
    }
    image = v2.ColorJitter(**params)(image)
    return image, params
