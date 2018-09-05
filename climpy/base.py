#!/usr/bin/env python3
"""
Includes several helper/base functions shared by other tools.
"""
def trail_flatten(data, nflat):
    """
    Flatten *trailing* dimensions onto single dimension (useful).
    Call this to manipulate data on first dimensions, then restore with unflatten.
    * Note that numpy prod of an empty iterable will be 1; so, adds singleton dim.
    * Note we use tuple expansion of the [shape] tuple
    * Note default for numpy is is row-major.
    """
    return np.reshape(data, (*data.shape[:-nflat], np.prod(data.shape[-nflat:]).astype(int)), order='F'), data.shape

def trail_unflatten(data, shape, nflat):
    """
    Undo action of flatten.
    Shape can be the original shape, or a new shape.
    """
    if data.shape[-1] != np.prod(shape[-nflat:]):
        raise ValueError(f'Number of trailing elements {data.shape[-1]} does not match trailing shape {shape[nflat:]:s}.')
    if not all(s1==s2 for s1,s2 in zip(data.shape[:-1], shape[:-nflat])):
        raise ValueError(f'Leading dimensions on data, {data.shape[:-1]}, do not match leading dimensions on new shape, {shape[:-nflat]}.')
    return np.reshape(data, shape, order='F')

def lead_flatten(data, nflat):
    """
    Flatten *leading* dimensions onto single dimension.
    """
    return np.reshape(data, (np.prod(data.shape[:nflat]).astype(int), *data.shape[nflat:]), order='C') # make column major

def lead_unflatten(data, shape, nflat):
    """
    Undo action of leadflatten.
    """
    if data.shape[0] != np.prod(shape[:nflat]):
        raise ValueError(f'Number of leading elements {data.shape[0]} does not match leading shape {shape[end:]:s}.')
    if not all(s1==s2 for s1,s2 in zip(data.shape[1:], shape[nflat:])):
        raise ValueError(f'Trailing dimensions on data, {data.shape[1:]}, do not match tailing dimensions on new shape, {shape[nflat:]}.')
    return np.reshape(data, shape, order='F')

def permute(data, axis=-1):
    """
    Permutes a given axis onto the LAST dimension.
    """
    return np.rollaxis(data, axis, data.ndim)

def unpermute(data, axis=-1, select=-1):
    """
    Undoes action of permute; permutes LAST dimension back onto original axis.
    Rolls axis <select> until it lies "before" (i.e. to the left) of axis <axis>.
    """
    if select<0: select = data.ndim+select
    return np.rollaxis(data, select, axis)

