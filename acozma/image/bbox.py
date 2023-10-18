import enum
import math

import numpy as np
from PIL import Image


class BBoxMode(str, enum.Enum):
    XYXY = "xyxy"
    XYWH = "xywh"


class BBox:
    # This class should handle both XYXY and XYWH and convert between them
    def __init__(
        self,
        xyxy: tuple[int, int, int, int],
        label: str | None = None,
    ):
        self.xyxy: tuple[int, int, int, int] = xyxy
        self.label = label

    @staticmethod
    def xyxy_to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        return (x1, y1, w, h)

    @staticmethod
    def xywh_to_xyxy(bbox):
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        return (x1, y1, x2, y2)

    @property
    def xywh(self):
        return BBox.xyxy_to_xywh(self.xyxy)

    @property
    def wh(self):
        _, _, w, h = self.xywh
        return w, h

    @property
    def width(self):
        return self.wh[0]

    @property
    def height(self):
        return self.wh[1]

    @property
    def area(self):
        w, h = self.wh
        return w * h

    def __repr__(self):
        return f"BBox(xyxy={self.xyxy}, label={self.label}) (xywh={self.xywh})"

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_xyxy(cls, xyxy: tuple[int, int, int, int], **kwargs):
        return cls(xyxy, **kwargs)

    @classmethod
    def from_xywh(cls, xywh: tuple[int, int, int, int], **kwargs):
        return cls(cls.xywh_to_xyxy(xywh), **kwargs)

    @classmethod
    def random_for_image(
        cls,
        image: Image.Image,
        width_bounds: tuple[int, int],
        height_bounds: tuple[int, int],
        **kwargs,
    ):
        width_min, width_max = width_bounds
        height_min, height_max = height_bounds
        assert (
            width_min < width_max
        ), f"width_min >= width_max: {width_min} >= {width_max}"
        assert (
            height_min < height_max
        ), f"height_min >= height_max: {height_min} >= {height_max}"

        width = np.random.randint(width_min, width_max)
        height = np.random.randint(height_min, height_max)

        x = np.random.randint(0, image.width - width)
        y = np.random.randint(0, image.height - height)
        return cls.from_xywh((x, y, width, height), **kwargs)

    def rescale(self, scale: float):
        """
        Rescales the bbox by the given percentage
        For example, scale=0.5 will shrink the bbox by 50%
        And scale=2 will double the bbox size
        """
        x1, y1, x2, y2 = self.xyxy
        w, h = x2 - x1, y2 - y1

        x1 -= w * (scale - 1) / 2
        y1 -= h * (scale - 1) / 2
        x2 += w * (scale - 1) / 2
        y2 += h * (scale - 1) / 2

        self.xyxy = (int(x1), int(y1), int(x2), int(y2))
        return self

    def make_square(self):
        x1, y1, x2, y2 = self.xyxy
        w, h = x2 - x1, y2 - y1

        largest_side = max(w, h)
        diff = largest_side - min(w, h)
        diff_side = int(math.floor(diff / 2))
        if w > h:
            y1 -= diff_side
            y2 += diff_side
            if diff % 2 == 1:
                y2 += 1
        elif h > w:
            x1 -= diff_side
            x2 += diff_side
            if diff % 2 == 1:
                x2 += 1

        self.xyxy = (int(x1), int(y1), int(x2), int(y2))

        w, h = self.wh
        assert w == h, f"w != h: {w} != {h}"

        return self

    def copy(self):
        return BBox(**self.__dict__)
