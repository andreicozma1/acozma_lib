from ..utils import in_notebook

if in_notebook():
    from IPython.display import Markdown, display


def _print_with_header(*values, level=1, **kwargs):
    header = "#" * level
    if in_notebook():
        display(Markdown(f"{header} {' '.join(map(str, values))}"))
    else:
        print(header, *values, **kwargs)


def h1(*values, **kwargs):
    _print_with_header(*values, level=1, **kwargs)


def h2(*values, **kwargs):
    _print_with_header(*values, level=2, **kwargs)


def h3(*values, **kwargs):
    _print_with_header(*values, level=3, **kwargs)


def h4(*values, **kwargs):
    _print_with_header(*values, level=4, **kwargs)


def h5(*values, **kwargs):
    _print_with_header(*values, level=5, **kwargs)


def h6(*values, **kwargs):
    _print_with_header(*values, level=6, **kwargs)


def hr():
    if in_notebook():
        display(Markdown("---"))
    else:
        print("---")
