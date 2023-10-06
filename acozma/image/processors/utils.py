from PIL import Image


def processor(func):
    def wrapper(image: Image.Image, *args, **kwargs):
        assert isinstance(image, Image.Image), "image must be a PIL image"
        assert image.mode == "RGB", "Input image must be an RGB image"
        res = func(image, *args, **kwargs)
        assert isinstance(
            res, Image.Image
        ), "Internal Error: processor must return a PIL image"
        assert res.mode == "RGB", "Internal Error: processor must return an RGB image"
        return res

    return wrapper
