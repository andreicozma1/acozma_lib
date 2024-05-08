from typing import Union
from pprint import pprint as _pprint
from pprint import pformat as _pformat

pprint_default_kwargs = dict(
    width=30, sort_dicts=False, compact=True, underscore_numbers=True
)


def pprint(*args, **kwargs):
    return _pprint(*args, **(pprint_default_kwargs | kwargs))


def pformat(*args, **kwargs):
    return _pformat(*args, **(pprint_default_kwargs | kwargs))


__pprint_width_map = {"xs": 20, "sm": 40, "md": 60, "lg": 80, "xl": 100}
__pprint_level_map = {1: "#", 2: "=", 3: "*", 4: "~", 5: "-"}


def pprint_sep(*values, level: int = 1, width: Union[str, int] = "lg", **kwargs):
    if isinstance(width, str):
        width = __pprint_width_map[width]
    print(__pprint_level_map[level] * width)
    if values:
        pprint(*values, **kwargs)
