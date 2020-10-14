#!/usr/bin/env python3
"""
Utilities used internally by climopy.
"""
from . import array, docstring, quack, warnings  # noqa: F401

try:  # print debugging
    from icecream import ic
except ImportError:  # graceful fallback if IceCream isn't installed
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa
