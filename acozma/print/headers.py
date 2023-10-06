from typing import Union

sep_sz_map = {
    "xs": 20,
    "sm": 40,
    "md": 60,
    "lg": 80,
    "xl": 100,
}


def _print_with_separator(*values, sep: str, sz: Union[str, int] = "lg", **kwargs):
    if isinstance(sz, str):
        sz = sep_sz_map[sz]
    print(sep * sz)
    if values:
        print(*values, **kwargs)
        print(sep * sz)


def hr1(*values, **kwargs):
    _print_with_separator(*values, sep="#", **kwargs)


def hr2(*values, **kwargs):
    _print_with_separator(*values, sep="=", **kwargs)


def hr3(*values, **kwargs):
    _print_with_separator(*values, sep="~", **kwargs)


def hr4(*values, **kwargs):
    _print_with_separator(*values, sep="-", **kwargs)


def hr5(*values, **kwargs):
    _print_with_separator(*values, sep="*", **kwargs)
