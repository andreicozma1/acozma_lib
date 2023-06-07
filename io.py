import itertools
import os
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import PIL
from skimage import io, transform
from skimage.util import img_as_float32, img_as_ubyte

from andrei_lib import process, utils

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
        img = process.apply(img, funcs=apply_funcs, **kwargs)
    return img


def show(
    images: Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]],
    titles: Union[str, List[str], List[List[str]], None] = None,
    plot_title: Union[str, None] = None,
    show_hist: bool = False,
    vertical: bool = False,
    cmap="gray",
    interpolation="antialiased",
    save_path: Union[str, None] = None,
    size=4,
    **kwargs,
) -> None:
    if isinstance(images, Union[np.ndarray, PIL.Image.Image]):
        images = [[images]]
    elif isinstance(images[0], Union[np.ndarray, PIL.Image.Image]):
        images = [images]
    elif not isinstance(images[0][0], Union[np.ndarray, PIL.Image.Image]):
        raise ValueError("Invalid image type")

    if titles is not None:
        if isinstance(titles, str):
            titles = [[titles]]
        elif isinstance(titles[0], str):
            titles = [titles]
        elif not isinstance(titles[0][0], str):
            raise ValueError("Invalid title type")

    num_rows = len(images)
    num_cols = len(images[0])
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
        title = None
        if titles is None:
            title = f"image[{i}][{j}]"

        elif vertical:
            title = (
                titles[j][i]
                if j < len(titles) and i < len(titles[j])
                else f"image[{i}][{j}]"
            )
        else:
            title = (
                titles[i][j]
                if i < len(titles) and j < len(titles[i])
                else f"image[{i}][{j}]"
            )
        image = images[j][i] if vertical else images[i][j]
        utils.print_info(image, title)

        if num_rows == 1 and num_cols == 1:
            ax = axs
        elif num_rows == 1 and num_cols > 1:
            ax = axs[j]
        elif num_rows > 1 and num_cols == 1:
            ax = axs[i]
        else:
            ax = axs[i][j]

        image = process.adjust(image, **kwargs)

        if show_hist:
            ax.hist(image.ravel(), bins=256)
        else:
            ax.imshow(np.squeeze(image), cmap=cmap, interpolation=interpolation)

        ax.set_title(title)
        ax.set_axis_off()

    if plot_title != None:
        plt.suptitle(plot_title)

    if save_path is not None:
        dir_path = os.path.dirname(save_path).strip()
        if not os.path.exists(dir_path) and dir_path != "":
            os.makedirs(dir_path)

        plt.savefig(save_path)

    plt.show()
