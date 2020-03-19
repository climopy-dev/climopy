#!/usr/bin/env python3
"""
Includes miscellaneous useful functions.
"""
import numpy as np
import itertools
from .utils import *  # noqa


def sin(x):
    """Sine in degrees."""
    return np.sin(x * np.pi / 180)


def cos(x):
    """Cosine in degrees."""
    return np.cos(x * np.pi / 180)


def tan(x):
    """Tangent in degrees."""
    return np.tan(x * np.pi / 180)


def csc(x):
    """Cosecant in degrees."""
    return 1 / np.sin(x * np.pi / 180)


def sec(x):
    """Secant in degrees."""
    return 1 / np.cos(x * np.pi / 180)


def cot(x):
    """Cotangent in degrees."""
    return 1 / np.tan(x * np.pi / 180)


def arcsin(x):
    """Inverse sine returning degrees."""
    return np.arcsin(x) * 180 / np.pi


def arccos(x):
    """Inverse cosine returning degrees."""
    return np.arccos(x) * 180 / np.pi


def arctan(x):
    """Inverse tangent returning degrees."""
    return np.arctan(x) * 180 / np.pi


def arccsc(x):
    """Inverse cosecant returning degrees."""
    return np.arcsin(1 / x) * 180 / np.pi


def arcsec(x):
    """Inverse secant returning degrees."""
    return np.arccos(1 / x) * 180 / np.pi


def arccot(x):
    """Inverse cotangent returning degrees."""
    return np.arctan(1 / x) * 180 / np.pi


def year(dt):
    """Get year from numpy datetime64."""
    return dt.astype('datetime64[Y]').astype(np.int32) + 1970


def month(dt):
    """Get month from numpy datetime64."""
    return dt.astype('datetime64[M]').astype(np.int32) % 12 + 1


def match(*args):
    """
    Match arbitrary number of 1D vectors; will return slices for producing the matching
    segment from either vector, and the vector itself, so use as follows:

    .. code-block:: python

        i1, i2, ..., vmatch = match(v1, v2, ...)
        v1[i1] == v2[i2] == ... == vmatch

    Useful e.g. for matching the time dimensions of 3D or 4D variables collected
    over different years and months.
    """
    vs = [np.array(v) for v in args]
    if not all(np.all(v == np.sort(v)) for v in vs):
        raise ValueError('Vectors must be sorted.')
    # Get common minima/maxima
    min_all, max_all = max(v.min() for v in vs), min(v.max() for v in vs)
    try:
        min_locs = [np.where(v == min_all)[0][0] for v in vs]
        max_locs = [np.where(v == max_all)[0][0] for v in vs]
    except IndexError:
        raise ValueError('Vectors do not have matching maxima/minima.')
    slices = [
        slice(min_i, max_i + 1) for min_i, max_i in zip(min_locs, max_locs)
    ]
    if any(
        v[slice_i].size != vs[0][slices[0]].size
        for v, slice_i in zip(vs, slices)
    ):
        raise ValueError(
            'Vectors are not identical between matching minima/maxima.'
        )
    elif any(
        not np.all(v[slice_i] == vs[0][slices[0]])
        for v, slice_i in zip(vs, slices)
    ):
        raise ValueError(
            'Vectors are not identical between matching minima/maxima.'
        )
    return slices + [vs[0][slices[0]]]


def intersection(x, segment1, segment2, xlog=False):
    """
    Find the (first) intersection point for two line segments.
    Optionally do this in log-space for the x-axis.
    """
    # Initial stuff
    segment1, segment2 = np.array(segment1), np.array(segment2)
    if xlog:  # transform x coordinates optionally
        def transform(x):
            return np.log10(x)
        def itransform(x):
            return 10 ** x
    else:
        def transform(x):
            return x
        def itransform(x):
            return x

    # Get intersection
    diff = segment1 - segment2
    if (diff > 0).all() or (diff < 0).all():
        print('Warning: No intersections found.')
        return np.nan, np.nan
    idx = np.where(diff > 0)[0][0]  # two-element vectors
    x, y = diff[idx - 1 : idx + 1], transform(x[idx - 1 : idx + 1])
    px = itransform(y[0] + (0 - x[0]) * ((y[1] - y[0]) / (x[1] - x[0])))
    x, y = y, segment2[idx - 1 : idx + 1]
    py = y[0] + (transform(px) - x[0]) * ((y[1] - y[0]) / (x[1] - x[0]))
    return px, py
