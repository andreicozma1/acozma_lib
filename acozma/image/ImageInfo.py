from typing import Tuple, Union

import numpy as np
from PIL import Image


class ImageInfo:
    def __init__(
        self,
        size: Tuple[int, int],
        channels: int,
        data_type: str,
        min_val: float,
        max_val: float,
    ):
        self.size = size
        self.channels = channels
        self.data_type = data_type
        self.min_val = min_val
        self.max_val = max_val

    @classmethod
    def from_pil(cls, pil_image: Image.Image) -> "ImageInfo":
        size = (pil_image.width, pil_image.height)
        channels = len(pil_image.getbands())
        data_type = str(pil_image.mode)
        extrema = pil_image.getextrema()
        if channels > 1:  # Multi-band image
            min_val = min([band[0] for band in extrema])
            max_val = max([band[1] for band in extrema])
        else:  # Single-band image
            min_val, max_val = extrema
        return cls(size, channels, data_type, min_val, max_val)

    @classmethod
    def from_numpy(cls, np_array: np.ndarray) -> "ImageInfo":
        if len(np_array.shape) > 3:
            raise ValueError(f"Unsupported array shape: {np_array.shape}")
        size = (np_array.shape[1], np_array.shape[0])
        channels = 1 if len(np_array.shape) == 2 else np_array.shape[2]
        data_type = str(np_array.dtype)
        min_val, max_val = np_array.min(), np_array.max()
        return cls(size, channels, data_type, min_val, max_val)

    @classmethod
    def from_any(cls, image: Union[Image.Image, np.ndarray]) -> "ImageInfo":
        if isinstance(image, np.ndarray):
            return cls.from_numpy(image)
        elif isinstance(image, Image.Image):
            return cls.from_pil(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def __str__(self) -> str:
        return f"{str(self.size)} {self.channels}C {self.data_type} {round(self.min_val, 2)}min/{round(self.max_val, 2)}max"

    @property
    def aspect_ratio(self) -> float:
        return self.size[0] / self.size[1]
