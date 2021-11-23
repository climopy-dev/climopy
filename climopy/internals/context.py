#!/usr/bin/env python3
"""
Context objects for managing datasets.
"""
import numpy as np

from . import ic  # noqa: F401
from . import _make_logger

# Set up ArrayContext logger
logger = _make_logger('ArrayContext', 'error')  # or 'info'


def _exe_axis_moves(array, moves, reverse=False):
    """
    Execute the input series of axis swaps.
    """
    slice_ = slice(None, None, -1) if reverse else slice(None)
    for move in moves[slice_]:
        move = move[slice_]
        array = np.moveaxis(array, *move)
        logger.info(f'Move {move[0]} to {move[1]}: {array.shape}')
    return array


def _get_axis_moves(push_left, push_right, left_base=0, right_base=-1):
    """
    Get the series of axis swaps given the input dimensionality.
    """
    logger.info(f'Push axes left: {push_left}')
    logger.info(f'Push axes right: {push_right}')
    moves = []
    left_base = 0
    right_base = -1
    for i, axis in enumerate(push_right):
        moves.append((axis, right_base))
        for push in (push_left, push_right):
            push[push > axis] -= 1  # NOTE: some of these changes have no effect
    for axis in push_left:
        moves.append((axis, left_base))
        for push in (push_left, push_right):
            push[push < axis] += 1  # NOTE: some of these changes have no effect
    return np.array(moves)


class _ArrayContext(object):
    """
    Temporarily reshape the input dataset(s). This is needed so we can do objective
    analysis tasks "along an axis". Some tasks can be done by just moving axes and using
    array[..., :] notation but this is not always possible. Should work with arbitrary
    duck-type arrays, including dask arrays.
    """
    def __init__(
        self, *args,
        push_right=None, push_left=None, nflat_right=None, nflat_left=None,
    ):
        """
        Parameters
        ----------
        *datas : numpy.ndarray
            The arrays to be reshaped
        push_left, push_right : int or list of int, optional
            Axis or axes to move to the left or right sides. Axes are moved in the input
            order. By default, if neither are provided, `push_right` is set to ``-1``.
        nflat_left, nflat_right : int, optional
            Number of dimensions to flatten on the left or right sides. By default, if
            only `push_left` is provided, `nflat_right` is set to ``data.ndim -
            len(push_left)``, and if only `push_right` is provided, `nflat_left` is set
            to ``data.ndim - len(push_right)``.

        Examples
        --------
        Here is a worked example used with the EOF algorithm:

        >>> import logging
        >>> import numpy as np
        >>> import xarray as xr
        >>> from climopy.internals.context import logger, _ArrayContext
        >>> logger.setLevel(logging.INFO)
        >>> # Generate neof, member, run, time, plev, lat array
        >>> dataarray = xr.DataArray(
        ...     np.random.rand(12, 8, 100, 40, 20),
        ...     dims=('member', 'run', 'time', 'plev', 'lat'),
        ... )
        >>> array = dataarray.data
        >>> with _ArrayContext(
        ...     array,
        ...     push_left=(0, 1), nflat_left=2,
        ...     push_right=(2, 3, 4), nflat_right=2,
        ... ) as context:
        ...     data = context.data
        ...     nextra, ntime, nspace = data.shape
        ...     eofs = np.random.rand(nextra, 5, 1, nspace)  # singleton time dimension
        ...     pcs = np.random.rand(nextra, 5, ntime, 1)  # singleton space dimension
        ...     context.replace_data(eofs, pcs, insert_left=1)
        >>> logger.setLevel(logging.ERROR)
        >>> eofs, pcs = context.data
        """
        # Set arrays
        # NOTE: No array standardization here. Assume duck-type arrays (numpy
        # arrays, pint quantities, xarray DataArrays, dask arrays).
        if not args:
            raise ValueError('Need at least one input argument.')
        self._arrays = args
        self._shapes = []
        self._moves = []
        ndim = self._arrays[0].ndim

        # Parse axis arguments and ensure they are positive
        if push_right is None and push_left is None:
            push_right = -1
        if push_right is None:
            push_right = np.array([])
        else:
            push_right = np.atleast_1d(push_right)
        if push_left is None:
            push_left = np.array([])
        else:
            push_left = np.atleast_1d(push_left)
        for push, side in zip((push_left, push_right), ('left', 'right')):
            push[push < 0] += ndim
            if any(push < 0) or any(push >= ndim) or np.unique(push).size != push.size:
                raise ValueError(f'Invalid push_{side}={push} for {ndim}D array.')
        self._push_left = push_left
        self._push_right = push_right

        # Parse nflat arguments. When user requests pushing to right, means we want
        # to flatten the remaining left dims. Same goes for pushing to left.
        # NOTE: There is distinction here between 'None' and '0'. The latter means
        # add a singleton dimension (useful when iterating over 'extra' dimensions)
        # while the former means add nothing.
        if nflat_left is None and not push_left.size and push_right.size:
            nflat_left = ndim - push_right.size
        if nflat_right is None and not push_right.size and push_left.size:
            nflat_right = ndim - push_left.size
        self._nflat_left = nflat_left
        self._nflat_right = nflat_right

    def replace_data(self, *args, insert_left=None, insert_right=None):
        """
        Replace the data attribute with new array(s).

        Parameters
        ----------
        *args : array-like, optional
            The new arrays. The unflattened middle-dimensions can be changed. The
            flattened leading or trailing dimensions can be reduced to singleton, but
            otherwise must be identical or it is unclear how they should be re-expanded.
        insert_left, insert_right : int, optional
            Number of new dimensions added to the left or right of the array.
            Dimensions can only be added to the left or the right of the
            unflattened middle-dimensions of the array. For example, `climopy.eof`
            adds a new `neof` dimension so that dimensions are transformed
            from ``(nextra, ntime, nspace)`` to ``(nextra, neof, ntime, nspace)``.
            Use lists of numbers to transform input arguments differently.

        Examples
        --------
        Inserting new dimensions does not mess up the order of values in dimensions
        that come before or after. This is revealed by playing with a simple example.

        >>> a = np.array(
        ...     [
        ...         [[1, 2, 1], [3, 4, 3]],
        ...         [[5, 6, 5], [7, 8, 7]],
        ...         [[9, 10, 9], [11, 12, 11]],
        ...     ]
        ... )
        >>> a.shape
        (3, 2, 3)
        >>> a[:, 0, 0]
        array([1, 5, 9])
        >>> np.reshape(a, (3, 6), order='F')[:, 0]
        array([1, 5, 9])
        >>> np.reshape(a, (3, 6), order='C')[:, 0]
        array([1, 5, 9])
        """
        # Parse arguments
        inserts_left, inserts_right = [], []
        for inserts, insert in zip((inserts_left, inserts_right), (insert_left, insert_right)):  # noqa: E501
            insert = np.atleast_1d(insert).tolist()
            if len(insert) == 1:
                insert = insert * len(args)
            elif len(insert) != len(args):
                raise ValueError(f'Got {len(insert)} inserts but {len(args)} args.')
            inserts[:] = insert

        # Check input array shapes
        # WARNING: The *flattened* dimensions of the new data must match the size
        # of the *flattened* dimensions of the input data. Flattened dimensions should
        # only be iterated over or reduced to length 1 by climopy functions like `eof`.
        shape_template = self._shapes[0]
        if not all(shape == shape_template for shape in self._shapes):
            raise ValueError(
                'Cannot reset dimensions when input data shapes '
                + ', '.join(map(repr, self._shapes)) + ' differ.'
            )

        # Loop through arrays
        nflat_left = self._nflat_left
        nflat_right = self._nflat_right
        shape_flat = self._arrays[0].shape
        shape_unflat_orig = self._shapes[0]
        self._arrays = []
        self._shapes = []
        self._moves = []
        for array, insert_left, insert_right in zip(args, inserts_left, inserts_right):
            # Check shape against flattened dimensions
            logger.info('')
            logger.info(f'Add new context array: {array.shape}')
            shape = list(array.shape)
            insert_left = insert_left or 0  # *number* of dimensions inserted left
            insert_right = insert_right or 0  # *number* of dimensions inserted right
            if (
                len(shape_flat) + insert_left + insert_right != len(shape)
            ) or (
                nflat_left is not None
                and shape[0] != shape_flat[0]
                and shape[0] > 1  # reduction to singleton is allowed
            ) or (
                nflat_right is not None
                and shape[-1] != shape_flat[-1]
                and shape[-1] > 1  # reduction to singleton is allowed
            ):
                raise ValueError(
                    f'New flattened array shape {shape!r} incompatible with '
                    f'existing flattened array shape {shape_flat!r}.'
                )

            # Determine *unflattened* shape from template shape
            shape_unflat = shape_unflat_orig.copy()
            if nflat_left is None:
                ileft_flat = 0
                nleft_unflat = 0
            else:
                ileft_flat = 1
                nleft_unflat = nflat_left
                if shape[0] <= 1:
                    for i in range(nflat_left):
                        shape_unflat[i] = 1
            if nflat_right is None:
                iright_flat = len(shape)
                nright_unflat = 0
            else:
                iright_flat = len(shape) - 1
                nright_unflat = nflat_right
                if shape[-1] <= 1:
                    for i in range(1, nflat_right + 1):
                        shape_unflat[-i] = 1

            # Build unflattened shape
            shape_left = shape_unflat[:nleft_unflat]
            shape_center = shape[ileft_flat:iright_flat]  # includes inserted
            shape_right = shape_unflat[len(shape_unflat) - nright_unflat:]
            shape = (*shape_left, *shape_center, *shape_right)
            logger.info(f'Change flattened shape {shape_flat} to {array.shape}.')
            logger.info(f'Number of left-flattened dimensions: {nflat_left}')
            logger.info(f'Number of right-flattened dimensions: {nflat_right}')
            logger.info(f'Flattened left dimensions: {shape_left}')
            logger.info(f'New center dimensions: {shape_center}')
            logger.info(f'Flattened right dimensions: {shape_right}')
            logger.info(f'Change unflattened shape {shape_unflat} to {shape}.')
            self._arrays.append(array)
            self._shapes.append(shape)

            # Correct the axis moves given new *inserted* dimensions
            # Example: Original array has shape [A, B, C, D, E] with push_left [1]
            # and push_right [0, 3]. Want the new array *final* shape (after swapping
            # axes) will be [X, Y, A, B, C, D, E, Z]. Now pretend this was the
            # *initial* dimensionality. Input push_left *and* push_right would be
            # plus 2 (shifted by 2 new axes), and input push_right unchanged.
            push_left = self._push_left + insert_left
            push_right = self._push_right + insert_left
            moves = _get_axis_moves(
                push_left,
                push_right,
                left_base=insert_left,
                right_base=(-1 - insert_right),
            )
            self._moves.append(moves)

    def __enter__(self):
        """
        Reshape the array.
        """
        # NOTE: Hard to build intuition for ND reshaping, but think of it as
        # just changing the *indices* used to refernece elements. For 2 x 3 x ...
        # array, row-major flattening creates 6 x ... array whose indices
        # correspond to A[0, 0], A[0, 1], A[0, 2], A[1, 0], A[1, 1], A[1, 2].
        # For column-major array, indices correspond to A[0, 0], A[1, 0],
        # A[0, 1], A[1, 1], A[0, 2], A[1, 2]. Other dimensions not affected.
        arrays = self._arrays
        nflat_left = self._nflat_left
        nflat_right = self._nflat_right
        self._arrays = []
        self._shapes = []
        self._moves = []
        for array in arrays:
            # Move axes
            logger.info('')
            logger.info(f'Flatten array: {array.shape}')
            push_left = self._push_left.copy()  # *must* be copy or replace_data fails!
            push_right = self._push_right.copy()
            moves = _get_axis_moves(push_left, push_right)
            array = _exe_axis_moves(array, moves)

            # Get new left shape
            ndim = array.ndim
            shape = list(array.shape)
            reshape = shape[nflat_left or 0:ndim - (nflat_right or 0)]
            if nflat_left is not None:
                s = shape[:nflat_left]
                N = np.prod(s).astype(int)
                reshape.insert(0, N)
                logger.info(f'Flatten {nflat_left} left dimensions: {s} to {N}')

            # Get new right shape
            if nflat_right is not None:
                s = shape[ndim - nflat_right:]
                N = np.prod(s).astype(int)
                reshape.append(N)
                logger.info(f'Flatten {nflat_right} right dimensions: {s} to {N}')

            # Reshape
            if shape != reshape:
                # WARNING: 'order' arg is invalid for dask arrays
                logger.info(f'Reshape from {array.shape} to {reshape}')
                array = np.reshape(array, reshape)
            self._arrays.append(array)
            self._moves.append(moves)
            self._shapes.append(shape)

        return self

    def __exit__(self, *args):  # noqa: U100
        """
        Restore the array to its original shape.
        """
        arrays = self._arrays
        shapes = self._shapes
        moves = self._moves
        self._arrays = []
        self._shapes = []
        self._moves = []
        for array, ishape, imoves in zip(arrays, shapes, moves):
            logger.info('')
            logger.info(f'Unflatten array: {array.shape}')
            if array.shape != ishape:
                # WARNING: 'order' arg is invalid for dask arrays
                logger.info(f'Reshape from {array.shape} to {ishape}')
                array = np.reshape(array, ishape)
            array = _exe_axis_moves(array, imoves, reverse=True)
            self._arrays.append(array)

    @property
    def data(self):
        """
        The arrays. Use this to retrieve reshaped arrays within the context block for
        your computation and outside the context block once they are reshaped back.
        """
        arrays = self._arrays
        if len(arrays) == 1:
            return arrays[0]
        else:
            return tuple(arrays)
