#!/usr/bin/env python3
"""
Includes several helper/base functions shared by other tools.
"""
import numpy as np

def trail_flatten(data, nflat=None):
    """
    Flatten *trailing* dimensions onto single dimension.
    Argument 'nflat' controls number of dimensions flattened; default is to
    choose <ndim-1> i.e. enough to flatten into a 2x2 array.
     * Note that numpy prod of an empty iterable will be 1; so, adds singleton dim.
     * Note we use tuple expansion of the [shape] tuple
     * Note default for numpy is is row-major.
    """
    if nflat is None:
        nflat = data.ndim-1 # all but last dimension
    return np.reshape(data, (*data.shape[:-nflat], np.prod(data.shape[-nflat:]).astype(int)), order='F'), list(data.shape)

def trail_unflatten(data, shape, nflat=None):
    """
    Undo action of trail_flatten.
    Argument 'shape' is desired shape of reconstructed array.
    Argument 'nflat' controls number of dimensions unflattened; default is to
    choose <ndim-1> i.e. enough to flatten into a 2x2 array.
    """
    if nflat is None:
        nflat = len(shape)-1 # all but last dimension
    if data.shape[-1] != np.prod(shape[-nflat:]):
        raise ValueError(f'Number of trailing elements {data.shape[-1]} does not match trailing shape {shape[nflat:]:s}.')
    if not all(s1==s2 for s1,s2 in zip(data.shape[:-1], shape[:-nflat])):
        raise ValueError(f'Leading dimensions on data, {data.shape[:-1]}, do not match leading dimensions on new shape, {shape[:-nflat]}.')
    return np.reshape(data, shape, order='F')

def lead_flatten(data, nflat=None):
    """
    Flatten *leading* dimensions onto single dimension.
    Argument 'nflat' controls number of dimensions flattened; default is to
    choose <ndim-1> i.e. enough to flatten into a 2x2 array.
    """
    if nflat is None:
        nflat = data.ndim-1 # all but last dimension
    return np.reshape(data, (np.prod(data.shape[:nflat]).astype(int), *data.shape[nflat:]), order='C'), list(data.shape) # make column major

def lead_unflatten(data, shape, nflat=None):
    """
    Undo action of lead_flatten.
    Argument 'shape' is desired shape of reconstructed array.
    Argument 'nflat' controls number of dimensions unflattened; default is to
    choose <ndim-1> i.e. enough to flatten into a 2x2 array.
    """
    if nflat is None:
        nflat = len(shape)-1 # all but last dimension
    if data.shape[0] != np.prod(shape[:nflat]):
        raise ValueError(f'Number of leading elements {data.shape[0]} does not match leading shape {shape[nflat:]}.')
    if not all(s1==s2 for s1,s2 in zip(data.shape[1:], shape[nflat:])):
        raise ValueError(f'Trailing dimensions on data, {data.shape[1:]}, do not match tailing dimensions on new shape, {shape[nflat:]}.')
    return np.reshape(data, shape, order='F')

def permute(data, axis=-1):
    """
    Permutes axis <axis> onto the *last* dimension.
    Rollaxis moves axis until to *left* of axis number data.ndim, i.e. to
    the end because that is 1 more than the last axis number.
    """
    select = data.ndim # so far no reason to make this variable
    return np.rollaxis(data, axis=axis, start=select)

def unpermute(data, axis=-1, select=-1):
    """
    Undoes action of permute; permutes *last* dimension back onto axis <axis>.
    Alternatively, roll the different axis <select> until it lies "before"
    (i.e. to the left) axis <axis>, which is useful e.g. if you added some new
    trailing dimensions.
    """
    if select<0:
        select = data.ndim + select
    return np.rollaxis(data, axis=select, start=axis)

