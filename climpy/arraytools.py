#!/usr/bin/env python3
"""
This module contains helper functions for manipulating arrays. Helps
us perform various complex analysis tasks "along axes" on arbirary
n-dimensional arrays. Generally you won't need to use them, but they are
public just in case.
"""
import numpy as np
#------------------------------------------------------------------------------#
# Benchmarks
#------------------------------------------------------------------------------#
# f = lambda x: x
# def test1(data, axis=-1):
#     # Test the lead flatten approach
#     data, shape = lead_flatten(np.moveaxis(data, axis, -1))
#     output = np.empty(data.shape)
#     for i in range(data.shape[0]): # iterate along first dimension; each row is an autocorrelation spectrum
#         output[i,:] = f(data[i,:]) # arbitrary complex equation
#     return np.moveaxis(lead_unflatten(output, shape), -1, axis)
# def test2(data, axis=-1):
#     # Test the new approach
#     # TODO: Directly use nditer with more minute control?
#     axis = (axis % data.ndim) # e.g. for 3D array, -1 becomes 2
#     output = np.empty(data.shape)
#     s = slice(None)
#     s1 = slice(None,axis) # speed!
#     s2 = slice(axis,None) # speed!
#     # Fastest yet!
#     ss = tuple(s for i,s in enumerate(data.shape) if i!=axis)
#     for idx in np.ndindex(*ss):
#         idx = (*idx[s1], s, *idx[s2])
#         output[idx] = f(data[idx])
#     # Way way slower with class
#     # for o,d in zip(iter_1d(output), iter_1d(data)):
#     #     o[:] = f(d)
#     # Tuple indexing, faster but still not perfect!
#     # shape = (1 if i==axis else s for i,s in enumerate(data.shape))
#     # for idx in np.ndindex(*shape):
#     #     idx = tuple(s if j==axis else i for i,j in enumerate(idx))
#     #     output[idx] = f(data[idx])
#     # List indexing, slow!
#     # shape = (1 if i==axis else s for i,s in enumerate(data.shape))
#     # for idx in np.ndindex(*shape):
#     #     idx = [*idx]
#     #     idx[axis] = slice(None)
#     #     output[idx] = f(data[idx])
#     return output

#------------------------------------------------------------------------------#
# Class
# WARNING: Turns out permuting and reshaping were much faster
# in the end... but this approach would allow integration with xarray because
# would not have to fuck up and rebuild Dataset indices.
#------------------------------------------------------------------------------#
# # First the class where shape does not change
# class iter_1d(object):
#     """Magical class for iterating over arbitrary axis of arbitrarily-shaped
#     array. Will return slices of said array."""
#     def __init__(self, data, axis=-1):
#         """Store data."""
#         axis = (axis % data.ndim) # e.g. for 3D array, -1 becomes 2
#         self.data = data
#         self.axis = axis
#         self._s1 = slice(None,axis) # speed!
#         self._s2 = slice(None)
#         self._s3 = slice(axis+1,None) # speed!
#
#     def __iter__(self):
#         """Instantiate."""
#         s = (s for i,s in enumerate(self.data.shape) if i!=self.axis)
#         self.iter = np.ndindex(*s)
#         return self
#
#     def __next__(self):
#         """Get next."""
#         idx = self.iter.next()
#         return self.data[(*idx[self._s1], self._s2, *idx[self._s3])]
#
# # Next approach where shape does change
# class iter_1d_reshape(object):
#     """Magical class that permutes and stuff."""
#     def __init__(self, data, axis=-1):
#         """Store data."""
#         axis = (axis % data.ndim) # e.g. for 3D array, -1 becomes 2
#         self.data = data
#         self.axis = axis
#
#     def __iter__(self):
#         """Instantiate."""
#         shape = [*data.shape]
#         if nflat is None:
#             nflat = data.ndim - 1 # all but last dimension
#         if nflat < 0: # 1D array already
#             self.i = None
#             self.view = data
#         else:
#             new = (np.prod(data.shape[:nflat]).astype(int), *data.shape[nflat:])
#             self.i = 0
#             self.view = np.reshape(data, new, order='C')
#         return self
#
#     def __next__(self):
#         """Get next."""
#         self.i += 1
#         if self.i >= self.view.shape[0]:
#             raise StopIteration
#         return self.data[i,:]

#------------------------------------------------------------------------------#
# Functions
#------------------------------------------------------------------------------#
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
        nflat = data.ndim - 1 # all but last dimension
    if nflat<=0: # just apply singleton dimension
        return data[None,...], shape
    data = np.reshape(data, (np.prod(data.shape[:nflat]).astype(int), *data.shape[nflat:]), order='C')
    return data, shape # make column major

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
        nflat = len(shape) - 1 # all but last dimension
    if nflat<=0: # we artificially added a singleton dimension; remove it
        return data[0,...]
    if data.shape[0] != np.prod(shape[:nflat]):
        raise ValueError(f'Number of leading elements {data.shape[0]} does not match leading shape {shape[nflat:]}.')
    if not all(s1==s2 for s1,s2 in zip(data.shape[1:], shape[nflat:])):
        raise ValueError(f'Trailing dimensions on data, {data.shape[1:]}, do not match trailing dimensions on new shape, {shape[nflat:]}.')
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
        nflat = data.ndim - 1 # all but last dimension
    if nflat<=0: # just add singleton dimension
        return data[...,None], shape
    if nflat is None:
        nflat = data.ndim - 1 # all but last dimension
    data = np.reshape(data, (*data.shape[:-nflat], np.prod(data.shape[-nflat:]).astype(int)), order='F')
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
        nflat = len(shape)-1 # all but last dimension
    if nflat<=0:
        return data[...,0]
    if data.shape[-1] != np.prod(shape[-nflat:]):
        raise ValueError(f'Number of trailing elements {data.shape[-1]} does not match trailing shape {shape[nflat:]}.')
    if not all(s1==s2 for s1,s2 in zip(data.shape[:-1], shape[:-nflat])):
        raise ValueError(f'Leading dimensions on data, {data.shape[:-1]}, do not match leading dimensions on new shape, {shape[:-nflat]}.')
    data = np.reshape(data, shape, order='F')
    return data

