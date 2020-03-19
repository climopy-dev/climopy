#!/usr/bin/env python3
"""
Includes miscellaneous useful functions.
"""
import numpy as np


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
    Return the overlapping points for a group of 1D vectors. Useful e.g. for
    matching the time dimensions of 3D or 4D variables collected over
    different years and months.

    Parameters
    ----------
    v1, v2, ... : ndarray
        The coordinate vectors.

    Returns
    -------
    i1, i2, ..., v : ndarray
        The indices of matching coordinates for each vector, and the vector
        consisting of these coordinates. These satsify the following condition:

        .. code-block:: python

            v1[i1] == v2[i2] == ... == vmatch
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
        raise ValueError('Vectors do not have matching minima/maxima.')
    slices = [
        slice(min_i, max_i + 1) for min_i, max_i in zip(min_locs, max_locs)
    ]

    # Checks
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

    Parameters
    ----------
    x : ndarray
        The *x* coordinates.
    segment1, segment2 : ndarray
        The two lists of *y* coordinates.
    xlog : bool, optional
        Whether to find the *x* coordinate intersection in logarithmic space.
    """
    # Initial stuff
    segment1, segment2 = np.array(segment1), np.array(segment2)
    if xlog:  # transform x coordinates optionally
        transform = lambda x: np.log10(x)
        itransform = lambda x: 10 ** x
    else:
        transform = itransform = lambda x: x

    # Get intersection
    diff = segment1 - segment2
    if (diff > 0).all() or (diff < 0).all():
        print('Warning: No intersections found.')
        return np.nan, np.nan
    idx = np.where(diff > 0)[0][0]  # two-element vectors
    x, y = diff[idx - 1:idx + 1], transform(x[idx - 1:idx + 1])
    px = itransform(y[0] + (0 - x[0]) * ((y[1] - y[0]) / (x[1] - x[0])))
    x, y = y, segment2[idx - 1:idx + 1]
    py = y[0] + (transform(px) - x[0]) * ((y[1] - y[0]) / (x[1] - x[0]))
    return px, py
