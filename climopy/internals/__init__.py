#!/usr/bin/env python3
"""
Utilities used internally by climopy.
"""
try:  # print debugging
    from icecream import ic
except ImportError:  # graceful fallback if IceCream isn't installed
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa
from . import docstring, quack, warnings  # noqa: F401


def _first_unique(args):
    """
    Return first unique argument.
    """
    seen = set()
    for arg in args:
        if arg not in seen:
            yield arg
        seen.add(arg)


def _make_stopwatch(verbose=True, fixedwidth=20):
    """
    Return a simple stopwatch.
    """
    import time
    t = time.time()
    def _stopwatch(message=None):  # noqa: E306
        nonlocal t
        delta = format(-t + (t := time.time()), '.6f')  # reassign t in enclosing scope
        if verbose and message:
            print(message + ':' + ' ' * (fixedwidth - len(message)), delta)
    return _stopwatch
