#!/usr/bin/env python3
"""
Includes several helper/base functions shared by other tools.
"""
import numpy as np

def trail_flatten(data, nflat=None):
    """
    Flatten *trailing* dimensions onto single dimension.

    Parameters
    ----------
    data : array
        input data
    nflat : int or None
        number of dimensions flattened -- default is to
        choose rank-1 i.e. enough to flatten into a rank 2 array

    Notes
    -----
    * Note that numpy prod of an empty iterable will be 1; so, adds singleton dim.
    * Note we use tuple expansion of the [shape] tuple
    * Note default for numpy is is row-major.
    """
    shape = list(data.shape)
    if nflat is None:
        nflat = data.ndim-1 # all but last dimension
    if nflat<=0: # just apply singleton dimension
        return data[...,None], [*shape, 1]
    if nflat is None:
        nflat = data.ndim-1 # all but last dimension
    return np.reshape(data, (*data.shape[:-nflat], np.prod(data.shape[-nflat:]).astype(int)), order='F'), shape

def trail_unflatten(data, shape, nflat=None):
    """
    Undo action of trail_flatten.

    Parameters
    ----------
    data : array
        the input data
    shape : iterable of ints
        desired shape of reconstructed array.
    nflat : int or None
        number of dimensions unflattened -- default is to
        choose rank-1 i.e. enough to flatten into a rank 2 array
    """
    if nflat is None:
        nflat = len(shape)-1 # all but last dimension
    if nflat<=0: # just remove singleton dimension
        return data[...,0]
    if data.shape[-1] != np.prod(shape[-nflat:]):
        raise ValueError(f'Number of trailing elements {data.shape[-1]} does not match trailing shape {shape[nflat:]}.')
    if not all(s1==s2 for s1,s2 in zip(data.shape[:-1], shape[:-nflat])):
        raise ValueError(f'Leading dimensions on data, {data.shape[:-1]}, do not match leading dimensions on new shape, {shape[:-nflat]}.')
    return np.reshape(data, shape, order='F')

def lead_flatten(data, nflat=None):
    """
    Flatten *trailing* dimensions onto single dimension.

    Parameters
    ----------
    data : array
        input data
    nflat : int or None
        number of dimensions flattened -- default is to
        choose rank-1 i.e. enough to flatten into a rank 2 array
    """
    shape = list(data.shape)
    if nflat is None:
        nflat = data.ndim-1 # all but last dimension
    if nflat<=0: # just apply singleton dimension
        return data[None,...], [1, *shape]
    # if np.sum(data.shape[:nflat])==0:
    #     shape = [1, *shape]
    return np.reshape(data, (np.prod(data.shape[:nflat]).astype(int), *data.shape[nflat:]), order='C'), shape # make column major

def lead_unflatten(data, shape, nflat=None):
    """
    Undo action of lead_flatten.

    Flatten *trailing* dimensions onto single dimension.

    Parameters
    ----------
    data : array
        input data
    shape : iterable of ints
        desired shape of reconstructed array.
    nflat : int or None
        number of dimensions unflattened -- default is to
        choose rank-1 i.e. enough to restore from a rank 2 array
    """
    if nflat is None:
        nflat = len(shape)-1 # all but last dimension
    if nflat<=0: # just remove singleton dimension
        return data[0,...]
    if data.shape[0] != np.prod(shape[:nflat]):
        raise ValueError(f'Number of leading elements {data.shape[0]} does not match leading shape {shape[nflat:]}.')
    if not all(s1==s2 for s1,s2 in zip(data.shape[1:], shape[nflat:])):
        raise ValueError(f'Trailing dimensions on data, {data.shape[1:]}, do not match trailing dimensions on new shape, {shape[nflat:]}.')
    return np.reshape(data, shape, order='F')

def permute(data, source=-1, destination=-1):
    """
    Permutes axes, by default moving to the *last* dimension.

    Arguments
    ---------
    source : int
        dimension to be permuted
    destination : int
        destination for that dimension

    Notes
    -----
    This is now a simple wrapper around np.moveaxis.
    This used to use np.rollaxis but it sucks as acknowledged by maintainers:
    https://github.com/numpy/numpy/issues/9473
    """
    data = np.moveaxis(data, source, destination)
    return data

def unpermute(data, source=-1, destination=-1):
    """
    Arguments
    ---------
    source : int
        dimension that was previously moved
    destination : int
        destination for that moved dimension
    """
    data = np.moveaxis(data, destination, source)
    return data

