import itertools
import os
from typing import Callable, List, Optional, Tuple, Union

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
    resize_width: Optional[int] = None,
    as_gray=False,
    as_ubyte: bool = False,
):
    img = io.imread(fname, as_gray=as_gray)
    if resize_width is not None:
        height = int(img.shape[0] * resize_width / img.shape[1])
        img = transform.resize(img, (height, resize_width), anti_aliasing=True)
    img = img_as_ubyte(img) if as_ubyte else img_as_float32(img)
    return img


def show(
    images: Union[IMAGE_TYPES, List[IMAGE_TYPES], List[List[IMAGE_TYPES]]],
    captions: Union[str, List[str], List[List[str]], None] = None,
    suptitle: Union[str, None] = None,
    show_hist: bool = False,
    vertical: bool = False,
    grid: bool = False,
    figsize=(5, 5),
    cmap="gray",
    interpolation="none",
    save_path: Union[str, None] = None,
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

    # heights, widths = [], []
    # for i, j in itertools.product(range(num_rows), range(num_cols)):
    #     img = images[j][i] if vertical else images[i][j]
    #     if img is None:
    #         continue
    #     # TODO: implement better way to get dimensions across types
    #     img = np.array(img)
    #     heights.append(img.shape[0])
    #     widths.append(img.shape[1])

    # height_avg, width_avg = np.mean(heights), np.mean(widths)
    # aspect_ratio = width_avg / height_avg
    # print(f"aspect_ratio: {aspect_ratio}")

    figsize = (
        num_cols * figsize[0],
        num_rows * figsize[1],
    )

    print(f"figsize: {figsize}")

    _, axs = plt.subplots(
        num_rows,
        num_cols,
        sharex=True,
        sharey=True,
        # dpi=150,
        figsize=figsize,
        constrained_layout=True,
    )

    for i, j in itertools.product(range(num_rows), range(num_cols)):
        img = images[j][i] if vertical else images[i][j]
        if img is None:
            continue

        img_caption = None
        if captions is None:
            img_caption = f"image[{i}][{j}]"

        elif vertical:
            img_caption = (
                captions[j][i]
                if j < len(captions) and i < len(captions[j])
                else f"image[{i}][{j}]"
            )
        else:
            img_caption = (
                captions[i][j]
                if i < len(captions) and j < len(captions[i])
                else f"image[{i}][{j}]"
            )

        utils.print_info(img, img_caption)

        if num_rows == 1 and num_cols == 1:
            ax = axs
        elif num_rows == 1 and num_cols > 1:
            ax = axs[j]
        elif num_rows > 1 and num_cols == 1:
            ax = axs[i]
        else:
            ax = axs[i][j]

        img = processing.adjust(img, **kwargs)

        if show_hist:
            ax.hist(img.ravel(), bins=256)
        else:
            ax.imshow(np.squeeze(img), cmap=cmap, interpolation=interpolation)

        # set to image mode
        ax.set_aspect("equal")
        ax.set_title(img_caption)
        ax.set_axis_off()

    if suptitle != None:
        plt.suptitle(suptitle)

    if save_path is not None:
        dir_path = os.path.dirname(save_path).strip()
        if not os.path.exists(dir_path) and dir_path != "":
            os.makedirs(dir_path)

        plt.savefig(save_path)

    plt.show()
