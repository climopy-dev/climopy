#!/usr/bin/env python3
"""
This module contains helper functions for manipulating arrays. Helps
us perform various complex analysis tasks "along axes" on arbirary
n-dimensional arrays. Generally you won't need to use them, but they are
public just in case.
"""
import numpy as np


def lead_flatten(data, nflat=None):
    """
    Flattens *leading* dimensions onto single dimension.

    Parameters
    ----------
    data : array
        Input data.
    nflat : int or None
        Number of dimensions flattened -- default is to
        choose rank-1, i.e. enough to flatten into a rank 2 array.
    """
    shape = [*data.shape]
    if nflat is None:
        nflat = data.ndim - 1  # all but last dimension
    if nflat <= 0:  # just apply singleton dimension
        return data[None, ...], shape
    data = np.reshape(
        data,
        (np.prod(data.shape[:nflat]).astype(int), *data.shape[nflat:]),
        order='C',
    )
    return data, shape  # make column major


def lead_unflatten(data, shape, nflat=None):
    """
    Undoes action of `lead_flatten`.

    Parameters
    ----------
    data : ndarray
        Input data.
    shape : list of int
        Desired shape of reconstructed array.
    nflat : int or None, optional
        Number of dimensions unflattened -- default is to
        choose rank-1, i.e. enough to restore from a rank 2 array.
    """
    if nflat is None:
        nflat = len(shape) - 1  # all but last dimension
    if nflat <= 0:  # we artificially added a singleton dimension; remove it
        return data[0, ...]
    if data.shape[0] != np.prod(shape[:nflat]):
        raise ValueError(
            f'Number of leading elements {data.shape[0]} does not match leading shape {shape[nflat:]}.')
    if not all(s1 == s2 for s1, s2 in zip(data.shape[1:], shape[nflat:])):
        raise ValueError(
            f'Trailing dimensions on data, {data.shape[1:]}, do not match trailing dimensions on new shape, {shape[nflat:]}.')
    data = np.reshape(data, shape, order='C')
    return data


def trail_flatten(data, nflat=None):
    """
    Flattens *trailing* dimensions onto single dimension.

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
    shape = [*data.shape]
    if nflat is None:
        nflat = data.ndim - 1  # all but last dimension
    if nflat <= 0:  # just add singleton dimension
        return data[..., None], shape
    if nflat is None:
        nflat = data.ndim - 1  # all but last dimension
    data = np.reshape(
        data,
        (*data.shape[:-nflat], np.prod(data.shape[-nflat:]).astype(int)),
        order='F',
    )
    return data, shape


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
        nflat = len(shape) - 1  # all but last dimension
    if nflat <= 0:
        return data[..., 0]
    if data.shape[-1] != np.prod(shape[-nflat:]):
        raise ValueError(
            f'Number of trailing elements {data.shape[-1]} does not match trailing shape {shape[nflat:]}.')
    if not all(s1 == s2 for s1, s2 in zip(data.shape[:-1], shape[:-nflat])):
        raise ValueError(
            f'Leading dimensions on data, {data.shape[:-1]}, do not match leading dimensions on new shape, {shape[:-nflat]}.')
    data = np.reshape(data, shape, order='F')
    return data
