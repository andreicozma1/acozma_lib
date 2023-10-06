import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
from skimage import exposure, feature, filters, measure, morphology
from skimage.util import img_as_float, img_as_ubyte


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


@processor
def colorgrid(
    image: Image.Image,
    grid_size: int = 5,
    downsample=Image.BICUBIC,
    upsample=Image.NEAREST,
):
    assert grid_size >= 5, "grid_size must be >= 5"

    width, height = image.size
    # TODO: Try the effect of different resampling filters
    # NEAREST: Literal[0]
    # LANCZOS: Literal[1]
    # BILINEAR: Literal[2]
    # BICUBIC: Literal[3]
    # BOX: Literal[4]
    # HAMMING: Literal[5]
    image = image.resize((grid_size, grid_size), resample=downsample)
    image = image.resize((width, height), resample=upsample)
    return image.convert("RGB")


@processor
def rand_colorgrid(image: Image.Image):
    grid_size = np.random.randint(5, 25)
    resampling_modes = [
        Image.NEAREST,
        Image.BILINEAR,
        Image.BICUBIC,
        Image.LANCZOS,
    ]
    downsample = np.random.choice(resampling_modes)
    return colorgrid(image, grid_size=grid_size, downsample=downsample)


@processor
def canny(
    image: Image.Image,
    sigma: float = 1.0,
    low_threshold: int = 100,
    high_threshold: int = 200,
):
    assert sigma > 0.0, "sigma must be positive"
    assert (
        0 <= low_threshold < 255
    ), f"low_threshold must be in [0, 255), got {low_threshold}"
    assert (
        0 < high_threshold <= 255
    ), f"high_threshold must be in (0, 255], got {high_threshold}"

    # TODO: Apply autocontrast?
    # image = ImageOps.autocontrast(image)

    # TODO: Apply histogram equalization?

    # apply gaussian blur
    image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

    image = np.array(image)
    # print(image.min())
    # print(image.max())

    # TODO: Auto thresholding for low_threshold and high_threshold
    image = cv2.Canny(image, low_threshold, high_threshold)

    image = Image.fromarray(image)

    return image.convert("RGB")


@processor
def rand_canny(image: Image.Image):
    sigma: float = np.random.uniform(0.6, 2.4)
    low_threshold: int = np.random.randint(0, 200)
    high_threshold: int = np.random.randint(254 - low_threshold, 255)
    return canny(image, sigma, low_threshold, high_threshold)


def _canny_colorgrid_post(image_canny, image_colorgrid):
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
    image_canny = rand_canny(image)
    image_colorgrid = rand_colorgrid(image)
    return _canny_colorgrid_post(image_canny, image_colorgrid)


@processor
def blur(image: Image.Image, radius: int = 5):
    blur_funcs = [
        ImageFilter.GaussianBlur,
        ImageFilter.BoxBlur,
        ImageFilter.MedianFilter,
    ]
    assert radius >= 0, "radius must be non-negative"

    if radius != 0:
        rand_func = np.random.choice(blur_funcs)

        # if median filter ensure radius is odd
        if rand_func == ImageFilter.MedianFilter and radius % 2 == 0:
            radius += 1

        image = image.filter(rand_func(radius))

    return image.convert("RGB")


def rand_blur(image: Image.Image):
    radius = np.random.randint(0, 25)
    print(radius)
    return blur(image, radius), radius


@processor
def hog(image: Image.Image, orientations: int = 4, pixels_per_cell: int = 9):
    # histogram of oriented gradients
    assert orientations >= 4, "orientations must be at least 4"
    assert pixels_per_cell >= 5, "pixels_per_cell must be at least 5"
    image = np.array(image)
    image = img_as_float(image)

    # TODO: Histogram equalization?
    # image = exposure.equalize_hist(image)
    # image = exposure.equalize_adapthist(image, clip_limit=0.03)

    _, hog_image = feature.hog(
        image,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(1, 1),
        visualize=True,
        channel_axis=-1,
    )

    hog_image = img_as_ubyte(hog_image)
    hog_image = Image.fromarray(hog_image)
    hog_image = ImageOps.autocontrast(hog_image)
    return hog_image.convert("RGB")


def rand_hog(image: Image.Image):
    orientations = np.random.randint(4, 9)
    pixels_per_cell = np.random.randint(5, 12)
    return hog(image, orientations, pixels_per_cell)


@processor
def entropy(img, radius: int = 5):
    assert radius >= 3, "radius must be non-negative"

    img = np.array(img)
    footprint = morphology.disk(radius)

    # compute entropy for each channel separately
    entropy_cs = []
    for channel in range(img.shape[-1]):
        ec = filters.rank.entropy(img[..., channel], footprint=footprint)
        # ec = (ec - ec.min()) / (ec.max() - ec.min())
        entropy_cs.append(ec)

    # merge channels back into a single image
    img = np.stack(entropy_cs, axis=-1)
    img = (img - img.min()) / (img.max() - img.min())

    img = img_as_ubyte(img)
    img = Image.fromarray(img)
    # img = ImageOps.autocontrast(img)
    return img


def rand_entropy(image: Image.Image):
    radius = np.random.randint(3, 15)
    return entropy(image, radius)


@processor
def contours(image: Image.Image):
    image = image.convert("L")
    image = np.array(image)
    image = img_as_float(image)

    # TODO: Histogram equalization?
    # image = exposure.equalize_hist(image)
    # image = exposure.equalize_adapthist(image, clip_limit=0.03)

    levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    contours = [measure.find_contours(image, level) for level in levels]
    image = np.zeros_like(image)
    for contour in contours:
        for c in contour:
            c = np.round(c).astype(int)
            image[c[:, 0], c[:, 1]] = 1

    image = img_as_ubyte(image)
    image = Image.fromarray(image)
    # autocontrast ensures we can find the edges
    image = ImageOps.autocontrast(image)
    return image.convert("RGB")


def rand_square_crop(image: Image.Image, resize: int = 512):
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
    return image.resize((resize, resize), resample=Image.LANCZOS)
