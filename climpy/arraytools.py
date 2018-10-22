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
    Argument 'shape' is desired shape of reconstructed array.
    Argument 'nflat' controls number of dimensions unflattened; default is to
    choose <ndim-1> i.e. enough to flatten into a 2x2 array.
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
    Flatten *leading* dimensions onto single dimension.
    Argument 'nflat' controls number of dimensions flattened; default is to
    choose <ndim-1> i.e. enough to flatten into a 2x2 array.
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
    Argument 'shape' is desired shape of reconstructed array.
    Argument 'nflat' controls number of dimensions unflattened; default is to
    choose <ndim-1> i.e. enough to flatten into a 2x2 array.
    """
    if nflat is None:
        nflat = len(shape)-1 # all but last dimension
    if nflat<=0: # just remove singleton dimension
        return data[0,...]
    if data.shape[0] != np.prod(shape[:nflat]):
        raise ValueError(f'Number of leading elements {data.shape[0]} does not match leading shape {shape[nflat:]}.')
    if not all(s1==s2 for s1,s2 in zip(data.shape[1:], shape[nflat:])):
        raise ValueError(f'Trailing dimensions on data, {data.shape[1:]}, do not match tailing dimensions on new shape, {shape[nflat:]}.')
    return np.reshape(data, shape, order='F')

def permute(data, source=-1, destination=-1):
    """
    Permutes axis <source> onto the *last* dimension by default.
    This used to use rollaxis but it sucks as acknowledged by maintainers:
    https://github.com/numpy/numpy/issues/9473
    The moveaxis command is way way more intuitive.
    """
    # print('before:', data.shape, source, destination)
    data = np.moveaxis(data, source, destination)
    # print('after:', data.shape)
    return data

def unpermute(data, source=-1, destination=-1):
    """
    Undoes action of permute, by default putting *last* dimension onto axis <source>.
    This used to use rollaxis but it sucks as acknowledged by maintainers:
    https://github.com/numpy/numpy/issues/9473
    The moveaxis command is way way more intuitive.
    """
    # print('before:', data.shape, source, destination)
    data = np.moveaxis(data, destination, source)
    # print('after:', data.shape)
    return data

