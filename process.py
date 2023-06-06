from typing import Callable, List, Optional, Union

import numpy as np
from skimage import exposure

import andrei_lib


def apply(
    images: Union[np.ndarray, List[np.ndarray]],
    funcs: Optional[Union[List[Callable], Callable]] = None,
    funcs_kwargs: Optional[Union[List[dict], dict]] = None,
    **kwargs,
):
    if funcs is None:
        return images

    return_single = False
    if isinstance(images, np.ndarray):
        return_single = True
        images = [images]

    if isinstance(funcs, Callable):
        funcs = [funcs]
    if isinstance(funcs_kwargs, dict):
        funcs_kwargs = [funcs_kwargs]
        if len(funcs) > 1:
            funcs_kwargs.extend([{}] * (len(funcs) - 1))
    if funcs_kwargs is None:
        funcs_kwargs = [{}] * len(funcs)

    processed_imgs = []

    print("=" * 80)
    print(f"Processing {len(images)} images")
    print("=" * 80)
    if len(images) == 1 and len(funcs) == 1 and len(funcs_kwargs) > 1:
        images = images * len(funcs_kwargs)
        funcs = funcs * len(funcs_kwargs)

        for i, (image, func, func_kwargs) in enumerate(
            zip(images, funcs, funcs_kwargs)
        ):
            image_proc = func(image, **func_kwargs)
            andrei_lib.utils.print_info(image_proc, f"Image[{i}]")

            image_proc = adjust(image_proc, **kwargs)
            processed_imgs.append(image_proc)

    else:
        for i, image in enumerate(images):
            image_proc = image.copy()
            andrei_lib.utils.print_info(image_proc, f"Image[{i}]")

            for func, func_kwargs in zip(funcs, funcs_kwargs):
                image_proc = func(image_proc, **func_kwargs)
                andrei_lib.utils.print_info(image_proc)

            image_proc = adjust(image_proc, **kwargs)
            processed_imgs.append(image_proc)

    print(f"=> Returning {len(processed_imgs)} images")
    print("=" * 80)
    return (
        processed_imgs[0]
        if return_single and len(processed_imgs) == 1
        else processed_imgs
    )


def adjust(image, adjust_log=None, adjust_gamma=None, clip=None):
    if adjust_log is not None and adjust_log is not False:
        gain = 1 if adjust_log is True else adjust_log
        image = exposure.adjust_log(image, gain)
        andrei_lib.utils.print_info(image, "adjust_log")

    if adjust_gamma is not None and adjust_gamma is not False:
        gamma = 1 if adjust_gamma is True else adjust_gamma
        image = exposure.adjust_gamma(image, gamma)
        andrei_lib.utils.print_info(image, "adjust_gamma")

    if clip is not None:
        if clip is True:
            clip = (0, 1)
        elif isinstance(clip, (int, float)):
            clip = (-clip, clip)
        elif isinstance(clip, (tuple, list)) and len(clip) == 1:
            clip = (-clip[0], clip[0])
        elif isinstance(clip, (tuple, list)) and len(clip) > 2:
            clip = clip[:2]
        image = np.clip(image, clip[0], clip[1])
        andrei_lib.utils.print_info(image, "clip")

    return image
