#!/usr/bin/env python3
"""
This module contains helper functions for manipulating arrays. Helps
us perform various complex analysis tasks "along axes" on arbirary
n-dimensional arrays. Generally you won't need to use them, but they are
public just in case.
"""
import numpy as np

def trail_flatten(data, nflat=None):
    """
    Flatten *trailing* dimensions onto single dimension.

    Parameters
    ----------
    data : array
        Input data.
    nflat : int or None
        Number of dimensions flattened -- default is to
        choose rank-1, i.e. enough to flatten into a rank 2 array.

    Notes
    -----
    * The `numpy.prod` of an empty array is 1. So in this case,
      a singleton dimension is added.
    * Note numpy arrays are row-major by default.
    """
    shape = list(data.shape)
    if nflat is None:
        nflat = data.ndim-1 # all but last dimension
    if nflat<=0: # just add singleton dimension
        return data[...,None], shape
        # return data[...,None], [*shape, 1]
    if nflat is None:
        nflat = data.ndim-1 # all but last dimension
    return np.reshape(data, (*data.shape[:-nflat], np.prod(data.shape[-nflat:]).astype(int)), order='F'), shape

def trail_unflatten(data, shape, nflat=None):
    """
    Undoes action of `trail_flatten`.

    Parameters
    ----------
    data : array
        Input data.
    shape : list of int
        Desired shape of reconstructed array.
    nflat : int or None
        Number of dimensions unflattened -- default is to
        choose rank-1, i.e. enough to restore from a rank 2 array.
    """
    if nflat is None:
        nflat = len(shape)-1 # all but last dimension
    # if nflat<=0 or shape[-1]==1: # just remove singleton dimension
    if nflat<=0:
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
        Input data.
    nflat : int or None
        Number of dimensions flattened -- default is to
        choose rank-1, i.e. enough to flatten into a rank 2 array.
    """
    shape = list(data.shape)
    if nflat is None:
        nflat = data.ndim-1 # all but last dimension
    if nflat<=0: # just apply singleton dimension
        # return data[None,...], [1, *shape]
        return data[None,...], shape
    return np.reshape(data, (np.prod(data.shape[:nflat]).astype(int), *data.shape[nflat:]), order='C'), shape # make column major

def lead_unflatten(data, shape, nflat=None):
    """
    Undoes action of `lead_flatten`.

    Parameters
    ----------
    data : array-like
        Input data.
    shape : list of int
        Desired shape of reconstructed array.
    nflat : int or None
        Number of dimensions unflattened -- default is to
        choose rank-1, i.e. enough to restore from a rank 2 array.
    """
    if nflat is None:
        nflat = len(shape) - 1 # all but last dimension
    # if nflat<=0 or shape[0]==1:
    if nflat<=0: # we artificially added a singleton dimension; remove it
        return data[0,...]
    if data.shape[0] != np.prod(shape[:nflat]):
        raise ValueError(f'Number of leading elements {data.shape[0]} does not match leading shape {shape[nflat:]}.')
    if not all(s1==s2 for s1,s2 in zip(data.shape[1:], shape[nflat:])):
        raise ValueError(f'Trailing dimensions on data, {data.shape[1:]}, do not match trailing dimensions on new shape, {shape[nflat:]}.')
    return np.reshape(data, shape, order='C')

def permute(data, source=-1, destination=-1):
    """
    Permutes axes, by default moving to the *last* dimension.

    Arguments
    ---------
    data : array-like
        Input data.
    source : int, optional
        Dimension to be permuted.
    destination : int, optional
        Destination for that dimension.

    Notes
    -----
    This is now a simple wrapper around `numpy.moveaxis`. This used to use
    `numpy.rollaxis`, but it sucks, as `acknowledged by maintainers
    <https://github.com/numpy/numpy/issues/9473>`_.
    """
    data = np.moveaxis(data, source, destination)
    return data

def unpermute(data, source=-1, destination=-1):
    """
    Undoes action of `permute`. Usage is identical.
    """
    data = np.moveaxis(data, destination, source)
    return data

