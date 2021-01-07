#!/usr/bin/env python3
"""
Utilities used internally by climopy.
"""
from . import array, docstring, quack, warnings  # noqa: F401

try:  # print debugging
    from icecream import ic
except ImportError:  # graceful fallback if IceCream isn't installed
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


def _first_unique(args):
    """
    Return first unique argument.
    """
    seen = set()
    for arg in args:
        if arg not in seen:
            yield arg
        seen.add(arg)


def _is_numeric(data):
    """
    Test if object is numeric, i.e. not string or datetime-like.
    """
    import numpy as np
    return np.issubdtype(np.asarray(data).dtype, np.number)


def _is_scalar(data):
    """
    Test if object is scalar. Returns ``False`` if it is sized singleton object
    or has more than one entry.
    """
    # WARNING: np.isscalar of dimensionless data returns False
    import pint
    import numpy as np
    if isinstance(data, pint.Quantity):
        data = data.magnitude
    data = np.asarray(data)
    return data.ndim == 0
