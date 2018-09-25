#!/usr/bin/env python3
"""
Includes miscelaneous useful functions.
"""
import numpy as np
from itertools import zip_longest
from .arraytools import *

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
    Special kind of zip that fails when iterators not same length.
    See: https://stackoverflow.com/a/32954700/4970632
    For purpose of object() see: https://stackoverflow.com/a/28306434/4970632
    """
    sentinel = object() # filler object; point is, will always be unique!
    for combo in zip_longest(*iterables, fillvalue=sentinel):
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

#------------------------------------------------------------------------------
# Math/simple
#------------------------------------------------------------------------------
def intersection(x, segment1, segment2, xlog=False):
    """
    Find the (first) intersection point for two line segments.
    Optionally do this in log-space for the x-axis.
    """
    #--------------------------------------------------------------------------#
    # Initial stuff
    segment1, segment2 = np.array(segment1), np.array(segment2)
    if xlog:
        func = lambda x: np.log10(x)
        ifunc = lambda x: 10**x
    else:
        func = lambda x: x
        ifunc = lambda x: x
    #--------------------------------------------------------------------------#
    # Get intersection
    diff = segment1 - segment2
    if (diff>0).all() or (diff<0).all():
        print("Warning: No intersections found.")
        return np.nan, np.nan
    idx = np.where(diff>0)[0][0]
    x, y = diff[idx-1:idx+1], func(x[idx-1:idx+1]) # two-element vectors
    px = ifunc(y[0] + (0-x[0])*((y[1]-y[0])/(x[1]-x[0])))
    x, y = y, segment2[idx-1:idx+1] # once again for the y-position; can also use segment1
    py = (y[0] + (func(px)-x[0])*((y[1]-y[0])/(x[1]-x[0])))
    return px, py

def rolling(data, window=5, axis=-1):
    """
    Read this: https://stackoverflow.com/a/4947453/4970632
    Generates rolling numpy window along final axis; can then operate with
    functions like polyfit or mean along the new last axis of output.
    Just creates *view* of original array, without duplicating data, so no worries
    about efficiency.
    * Will generate a new axis in the -1 position that is a running representation
      of value in axis numver <axis>.
    * Strides are apparently the 'number of bytes' one has to skip in memory
      to move to next position *on the given axis*. For example, a 5 by 5
      array of 64bit (8byte) values will have array.strides == (40,8).
    * Should consider using swapaxes instead of these permute and unpermute
      functions, might be simpler.
    TODO: Add option that preserves *edges* i.e. does not reduces length
    of dimension to be 'rolled' by (window-1).
    """
    # Roll axis, reshape, and get generate running dimension
    # data = np.rollaxis(data, axis, data.ndim)
    if axis<0: axis = data.ndim+axis # e.g. if 3 dims, and want to axis dim -1, this is dim number 2
    data = permute(data, axis)
    shape = data.shape[:-1] + (data.shape[-1]-(window-1), window)
    strides = [*data.strides, data.strides[-1]] # repeat striding on end
    data = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return unpermute(data, axis, select=-2) # want to 'put back' axis -2;
        # axis -1 is our new "rolling"/"averaging" dimension, keep that in same position

def running(*args, **kwargs):
    """
    Easier to remember name.
    """
    return rolling(*args, **kwargs)

def roots(poly):
    """
    Find real-valued root for polynomial with input coefficients; pretty simple.
    Format of input array p: p[0]*x^n + p[1]*x^n-1 + ... + p[n-1]*x + p[n]
    """
    # Just use numpy's roots function, and filter results.
    r = np.roots(poly) # input polynomial; returns ndarray
    filt = (r==np.real(r))
    r = r[filt].astype(np.float32) # take only real-valued ones
    return r

def slope(x, y, axis=-1):
    """
    Get linear regression along axis, ignoring NaNs. Uses np.polyfit.
    """
    # First, reshape
    y = np.rollaxis(y, axis, y.ndim).T # reverses dimension order
    yshape = y.shape # save shape
    y = np.reshape(y, (yshape[0], np.prod(yshape[1:])), order='F')

    # Next, supply to polyfit
    coeff = np.polyfit(x, y, deg=1) # is ok with 2d input data
        # DOES NOT accept more than 2d; also, much faster than loop with stats.linregress

    # And finally, make
    # Coefficients are returned in reverse; e.g. 2-deg fit gives c[0]*x^2 + c[1]*x + c[2]
    slopes = coeff[0,:]
    slopes = np.reshape(slopes, (1, y.shape[1:]), order='F').T
    return np.rollaxis(slopes, y.ndim-1, axis)

