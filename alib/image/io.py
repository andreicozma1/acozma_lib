import itertools
import os
from typing import Callable, List, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import io, transform
from skimage.util import img_as_float32, img_as_ubyte

from . import processing, utils

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

IMAGE_TYPES = Union[np.ndarray, Image.Image]


def read(
    fname: str,
    resize_width: int = None,
    as_gray=False,
    as_ubyte: bool = False,
    apply_funcs: Union[List[Callable], Callable, None] = None,
    **kwargs,
):
    img = io.imread(fname, as_gray=as_gray)
    if resize_width is not None:
        height = int(img.shape[0] * resize_width / img.shape[1])
        img = transform.resize(img, (height, resize_width), anti_aliasing=True)
    img = img_as_ubyte(img) if as_ubyte else img_as_float32(img)
    if apply_funcs is not None:
        img = processing.apply(img, funcs=apply_funcs, **kwargs)
    return img


def show(
    images: Union[IMAGE_TYPES, List[IMAGE_TYPES], List[List[IMAGE_TYPES]]],
    captions: Union[str, List[str], List[List[str]], None] = None,
    title: Union[str, None] = None,
    show_hist: bool = False,
    vertical: bool = False,
    grid: bool = False,
    cmap="gray",
    interpolation="antialiased",
    save_path: Union[str, None] = None,
    size=4,
    **kwargs,
) -> None:
    if isinstance(images, IMAGE_TYPES):
        images = [[images]]
    elif isinstance(images[0], IMAGE_TYPES):
        images = [images]
    elif not isinstance(images[0][0], IMAGE_TYPES):
        raise ValueError(f"Invalid image type: {type(images[0][0])}")

    if captions is not None:
        if isinstance(captions, str):
            captions = [[captions]]
        elif isinstance(captions[0], str):
            captions = [captions]
        elif not isinstance(captions[0][0], str):
            raise ValueError(f"Invalid caption type: {type(captions[0][0])}")

    if not grid:
        num_rows = len(images)
        num_cols = len(images[0])
    else:
        # flatten images
        images = list(itertools.chain.from_iterable(images))

        # make both into a 2D grid
        num_images = len(images)
        # make into a square grid if possible but always show all images
        num_rows = int(np.sqrt(num_images))
        num_cols = int(np.ceil(num_images / num_rows))

        if num_rows * num_cols < num_images:
            raise ValueError("Error encountered while constructing grid of images")

        images = [images[i : i + num_cols] for i in range(0, num_images, num_cols)]

        if captions is not None:
            captions = list(itertools.chain.from_iterable(captions))
            captions = [
                captions[i : i + num_cols] for i in range(0, num_images, num_cols)
            ]

    if vertical:
        num_rows, num_cols = num_cols, num_rows

    print("=" * 80)
    print(f"Plotting {num_rows}x{num_cols} images")
    print("=" * 80)

    figsize = (num_cols * size, num_rows * size)
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        sharex=True,
        sharey=True,
        # dpi=150,
        figsize=figsize,
        constrained_layout=True,
    )

    for i, j in itertools.product(range(num_rows), range(num_cols)):
        caption = None
        if captions is None:
            caption = f"image[{i}][{j}]"

        elif vertical:
            caption = (
                captions[j][i]
                if j < len(captions) and i < len(captions[j])
                else f"image[{i}][{j}]"
            )
        else:
            caption = (
                captions[i][j]
                if i < len(captions) and j < len(captions[i])
                else f"image[{i}][{j}]"
            )
        try:
            image = images[j][i] if vertical else images[i][j]
        except IndexError:
            continue
        utils.print_info(image, caption)

        if num_rows == 1 and num_cols == 1:
            ax = axs
        elif num_rows == 1 and num_cols > 1:
            ax = axs[j]
        elif num_rows > 1 and num_cols == 1:
            ax = axs[i]
        else:
            ax = axs[i][j]

        image = processing.adjust(image, **kwargs)

        if show_hist:
            ax.hist(image.ravel(), bins=256)
        else:
            ax.imshow(np.squeeze(image), cmap=cmap, interpolation=interpolation)

        ax.set_title(caption)
        ax.set_axis_off()

    if title != None:
        plt.suptitle(title)

    if save_path is not None:
        dir_path = os.path.dirname(save_path).strip()
        if not os.path.exists(dir_path) and dir_path != "":
            os.makedirs(dir_path)

        plt.savefig(save_path)

    plt.show()
