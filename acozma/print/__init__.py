from .headers import *
from .headers_md import *

from pprint import pprint as _pprint
from pprint import pformat as _pformat

pprint_default_kwargs = dict(
    depth=2, width=30, sort_dicts=False, compact=True, underscore_numbers=True
)

pprint = lambda *args, **kwargs: _pprint(
    *args,
    **(pprint_default_kwargs | kwargs),
)

pformat = lambda *args, **kwargs: _pformat(
    *args,
    **(pprint_default_kwargs | kwargs),
)
