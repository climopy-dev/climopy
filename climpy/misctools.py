#!/usr/bin/env python3
"""
Includes miscellaneous useful functions.
"""
import numpy as np
import itertools
from .arraytools import *

# Trigonometric functions in degrees
def sin(x):
    """Sine in degrees."""
    return np.sin(x*np.pi/180)
def cos(x):
    """Cosine in degrees."""
    return np.cos(x*np.pi/180)
def tan(x):
    """Tangent in degrees."""
    return np.tan(x*np.pi/180)
def csc(x):
    """Cosecant in degrees."""
    return 1/np.sin(x*np.pi/180)
def sec(x):
    """Secant in degrees."""
    return 1/np.cos(x*np.pi/180)
def cot(x):
    """Cotangent in degrees."""
    return 1/np.tan(x*np.pi/180)
def arcsin(x):
    """Inverse sine returning degrees."""
    return np.arcsin(x)*180/np.pi
def arccos(x):
    """Inverse cosine returning degrees."""
    return np.arccos(x)*180/np.pi
def arctan(x):
    """Inverse tangent returning degrees."""
    return np.arctan(x)*180/np.pi
def arccsc(x):
    """Inverse cosecant returning degrees."""
    return np.arcsin(1/x)*180/np.pi
def arcsec(x):
    """Inverse secant returning degrees."""
    return np.arccos(1/x)*180/np.pi
def arccot(x):
    """Inverse cotangent returning degrees."""
    return np.arctan(1/x)*180/np.pi

################################################################################
# Time handling stuff
################################################################################
def year(dt):
    """
    Gets year from numpy datetime object (used e.g. by xarray, pandas).
    """
    # The astype(int) is actually super fast (ns), and below is in general
    # faster than list comprehension with container of datetime objects
    # UNIX time starts at 1970-01-01 00:00:00
    return dt.astype('datetime64[Y]').astype(np.int32)+1970

def month(dt):
    """
    Gets month from numpy datetime object (used e.g. by xarray, pandas).
    """
    # Below will convert datetime64 units from [ns] (default) to months, then spit out months relative to year
    # UNIX time starts at 1970-01-01 00:00:00
    return dt.astype('datetime64[M]').astype(np.int32)%12 + 1

################################################################################
# Random vector stuff
# Doesn't really fit 'theme' of climpy but whatevs
################################################################################
def zip(*iterables):
    """
    Special kind of zip that fails when iterators not same length;
    see `this post <https://stackoverflow.com/a/32954700/4970632>`__.
    For purpose of ``sentinel`` see `this post
    <https://stackoverflow.com/a/28306434/4970632>`__.
    """
    sentinel = object() # filler object; point is, will always be unique!
    for combo in itertools.zip_longest(*iterables, fillvalue=sentinel):
        # if sentinel in combo:
        if any(sentinel is c for c in combo):
            print(combo)
            raise ValueError('Iterables have different lengths: '
                f'{", ".join(str(len(i)) for i in iterables)}.')
        yield combo

def match(*args):
    """
    Match arbitrary number of 1D vectors; will return slices for producing the matching
    segment from either vector, and the vector itself, so use as follows:

    ::

        i1, i2, ..., vmatch = match(v1, v2, ...)
        v1[i1] == v2[i2] == ... == vmatch

    Useful e.g. for matching the time dimensions of 3D or 4D variables collected
    over different years and months.
    """
    vs = [np.array(v) for v in args]
    if not all(np.all(v==np.sort(v)) for v in vs):
        raise ValueError('Vectors must be sorted.')
    # Get common minima/maxima
    min_all, max_all = max(v.min() for v in vs), min(v.max() for v in vs)
    try:
        min_locs = [np.where(v==min_all)[0][0] for v in vs]
        max_locs = [np.where(v==max_all)[0][0] for v in vs]
    except IndexError:
        raise ValueError('Vectors do not have matching maxima/minima.')
    slices = [slice(min_i, max_i+1) for min_i,max_i in zip(min_locs,max_locs)]
    if any(v[slice_i].size != vs[0][slices[0]].size for v,slice_i in zip(vs,slices)):
        raise ValueError('Vectors are not identical between matching minima/maxima.')
    elif any(not np.all(v[slice_i]==vs[0][slices[0]]) for v,slice_i in zip(vs,slices)):
        raise ValueError('Vectors are not identical between matching minima/maxima.')
    return slices + [vs[0][slices[0]]]

def intersection(x, segment1, segment2, xlog=False):
    """
    Find the (first) intersection point for two line segments.
    Optionally do this in log-space for the x-axis.
    """

    # Initial stuff
    segment1, segment2 = np.array(segment1), np.array(segment2)
    if xlog: # transform x coordinates optionally
        transform  = lambda x: np.log10(x)
        itransform = lambda x: 10**x
    else:
        transform  = lambda x: x
        itransform = lambda x: x

    # Get intersection
    diff = segment1 - segment2
    if (diff>0).all() or (diff<0).all():
        print("Warning: No intersections found.")
        return np.nan, np.nan
    idx = np.where(diff>0)[0][0]
    x, y = diff[idx-1:idx+1], transform(x[idx-1:idx+1]) # two-element vectors
    px = itransform(y[0] + (0-x[0])*((y[1]-y[0])/(x[1]-x[0])))
    x, y = y, segment2[idx-1:idx+1] # once again for the y-position; can also use segment1
    py = (y[0] + (transform(px)-x[0])*((y[1]-y[0])/(x[1]-x[0])))
    return px, py

# def match(v1, v2):
#     """
#     Match two 1D vectors; will return slices for producing the matching
#     segment from either vector, and the vector itself, so use as follows:
#         i1, i2, vmatch = match(v1, v2)
#         v1[i1] == v2[i2] == vmatch
#     Useful e.g. for matching the time dimensions of 3D or 4D variables collected
#     over different years and months.
#     """
#     v1, v2 = np.array(v1), np.array(v2)
#     if not np.all(v1==np.sort(v1)) or not np.all(v2==np.sort(v2)):
#         raise ValueError('Vectors must be sorted.')
#     # Get common minima/maxima
#     min12, max12 = max(v1.min(), v2.min()), min(v1.max(), v2.max())
#     try:
#         min1f, min2f = np.where(v1==min12)[0][0], np.where(v2==min12)[0][0]
#         max1f, max2f = np.where(v1==max12)[0][0], np.where(v2==max12)[0][0]
#     except IndexError:
#         raise ValueError('Vectors do not have matching maxima/minima.')
#     slice1, slice2 = slice(min1f, max1f+1), slice(min2f, max2f+1)
#     if v1[slice1].size != v2[slice2].size:
#         raise ValueError('Vectors are not identical between matching minima/maxima.')
#     elif not (v1[slice1]==v2[slice2]).all():
#         raise ValueError('Vectors are not identical between matching minima/maxima.')
#     return slice1, slice2, v1[slice1]
