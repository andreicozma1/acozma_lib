from PIL import Image


def processor(func):
    def wrapper(image: Image.Image, *args, **kwargs):
        assert isinstance(
            image, Image.Image
        ), "image must be a PIL image. Got {}".format(type(image))
        assert image.mode == "RGB", "Input image must be an RGB image"
        res = func(image, *args, **kwargs)
        assert isinstance(
            res, Image.Image
        ), "Internal Error: processor must return a PIL image. Got {}".format(type(res))
        assert (
            res.mode == "RGB"
        ), "Internal Error: processor must return an RGB image. Got {}".format(res.mode)
        return res

    return wrapper


def params_to_str(params: dict):
    return ",".join(
        f"{k}={round(v, 2) if isinstance(v, float) else v}" for k, v in params.items()
    )
