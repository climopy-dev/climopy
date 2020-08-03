#!/usr/bin/env python3
"""
Includes miscellaneous mathematical functions.
"""
import datetime
from functools import partial

import numpy as np
import pandas as pd

from .diff import deriv_half, deriv_uneven
from .internals import quack, warnings

__all__ = [
    'dt2cal',
    'intersection',
    'linetrack',
    'match',
    'zerofind',
]


def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, and seconds with these quantites indexed on the last axis.

    Parameters
    ----------
    dt : datetime64 array
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 6)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """
    # See: https://stackoverflow.com/a/56260054/4970632
    # Allocate output
    dt = np.asarray(dt)
    out = np.empty(dt.shape + (6,), dtype='u4')

    # Decompose calendar floors
    # NOTE: M8 is datetime64, m8 is timedelta64
    if isinstance(dt, pd.DatetimeIndex):
        # Datatype is subdtype of numpy.datetime64 but includes builtin
        # methods for getting calendar properties
        out[..., 0] = dt.year
        out[..., 1] = dt.month
        out[..., 2] = dt.day
        out[..., 3] = dt.hour
        out[..., 4] = dt.minute
        out[..., 5] = dt.second
    elif np.issubdtype(dt, np.datetime64):
        Y, M, D, h, m, s = [dt.astype(f'M8[{x}]') for x in 'YMDhms']
        out[..., 0] = Y + 1970  # Gregorian Year
        out[..., 1] = (M - Y) + 1  # month
        out[..., 2] = (D - M) + 1  # day
        out[..., 3] = (dt - D).astype('m8[h]')  # hour
        out[..., 4] = (dt - h).astype('m8[m]')  # minute
        out[..., 5] = (dt - m).astype('m8[s]')  # second
    elif dt.dtype == 'object' and all(isinstance(_, datetime.datetime) for _ in dt.flat):  # noqa: E501
        out[..., 0] = np.vectorize(partial(getattr, dt, 'year'))()
        out[..., 1] = np.vectorize(partial(getattr, dt, 'month'))()
        out[..., 2] = np.vectorize(partial(getattr, dt, 'day'))()
        out[..., 3] = np.vectorize(partial(getattr, dt, 'hour'))()
        out[..., 4] = np.vectorize(partial(getattr, dt, 'minute'))()
        out[..., 5] = np.vectorize(partial(getattr, dt, 'second'))()
    else:
        raise ValueError(f'Invalid data type for dt2cal: {dt.dtype}')
    return out


def match(*args):
    """
    Return the overlapping points for a group of 1D vectors. Useful e.g. for
    matching the time dimensions of 3D or 4D variables collected over
    different years and months.

    Parameters
    ----------
    v1, v2, ... : ndarray
        The coordinate vectors.

    Returns
    -------
    i1, i2, ..., v : ndarray
        The indices of matching coordinates for each vector, and the vector
        consisting of these coordinates. These satsify the following condition:

        .. code-block:: python

            v1[i1] == v2[i2] == ... == vmatch
    """
    vs = [np.array(v) for v in args]
    if not all(np.all(v == np.sort(v)) for v in vs):
        raise ValueError('Vectors must be sorted.')

    # Get common minima/maxima
    min_all, max_all = max(v.min() for v in vs), min(v.max() for v in vs)
    try:
        min_idxs = [np.where(v == min_all)[0][0] for v in vs]
        max_idxs = [np.where(v == max_all)[0][0] for v in vs]
    except IndexError:
        raise ValueError('Vectors do not have matching minima/maxima.')
    slices = [
        slice(min_i, max_i + 1) for min_i, max_i in zip(min_idxs, max_idxs)
    ]

    # Checks
    if any(
        v[slice_i].size != vs[0][slices[0]].size
        for v, slice_i in zip(vs, slices)
    ):
        raise ValueError(
            'Vectors are not identical between matching minima/maxima.'
        )
    elif any(
        not np.all(v[slice_i] == vs[0][slices[0]])
        for v, slice_i in zip(vs, slices)
    ):
        raise ValueError(
            'Vectors are not identical between matching minima/maxima.'
        )
    return slices + [vs[0][slices[0]]]


def intersection(x, y1, y2, xlog=False):
    """
    Find the (first) intersection point for two line segments.

    Parameters
    ----------
    x : ndarray
        The *x* coordinates.
    y1, y2 : ndarray
        The two lists of *y* coordinates.
    xlog : bool, optional
        Whether to find the *x* coordinate intersection in logarithmic space.

    Example
    -------

    >>> import climopy as climo
    ... x = 10 + np.arange(4)
    ... y1 = np.array([4, 2, 0, -2])
    ... y2 = np.array([0, 1, 2, 3])
    ... climo.intersection(x, y1, y2)

    """
    # Initial stuff
    x = np.asanyarray(x)
    y1 = np.asanyarray(y1)
    y2 = np.asanyarray(y2)
    if xlog:  # transform x coordinates optionally
        transform = lambda x: np.log10(x)  # noqa: E731
        itransform = lambda x: 10 ** x  # noqa: E731
    else:
        transform = itransform = lambda x: x  # noqa: E731
    if x.size != y1.size or x.size != y2.size:
        raise ValueError(f'Incompatible sizes {x.size=}, {y1.size=}, {y2.size=}.')

    # Get intersection
    dy = y1 - y2
    if np.all(dy > 0) or np.all(dy < 0):
        warnings._warn_climopy(f'No intersections found for data {y1!r} and {y2!r}.')
        return np.nan, np.nan
    idx, = np.where(np.diff(np.sign(dy)) != 0)  # e.g. 6, 2, -3 --> 1, 1, -1 --> 0, -2
    if idx.size > 1:
        warnings._warn_climopy('Multiple intersections found. Using the first one.')
    idx = idx[0]

    # Get coordinates
    x, y = dy[idx:idx + 2], transform(x[idx:idx + 2])
    px = itransform(y[0] + (0 - x[0]) * ((y[1] - y[0]) / (x[1] - x[0])))
    x, y = y, y2[idx:idx + 2]
    py = y[0] + (transform(px) - x[0]) * ((y[1] - y[0]) / (x[1] - x[0]))
    return px, py


# TODO: Support pint quantities here
def linetrack(xs, ys=None, /, sep=None, seed=None, ntrack=None):  # noqa: E225
    """
    Track individual "lines" across lists of coordinates.

    Parameters
    ----------
    xs : list of lists
        The locations to be grouped into lines.
    ys : list of lists, optional
        The values corresponding to the locations `xs`.
    sep : float, optional
        The maximum separation between points belonging to the same "track".
        Larger separations will cause the algorithm to begin a new track. The
        default behavior is to not separate into tracks this way.
    seed : float or list of float, optional
        The track or tracks you want the algorithm to pick up if the number of
        tracks is limited by `ntrack`.
    ntrack : int, optional
        The maximum number of values to be simultaneously tracked. This can
        be set to a low value to ignore spurious values in combination with `seed`.
        The default value is the maximum `xs` sublist length.

    Returns
    -------
    xs_sorted : ndarray
        2D array of *x* coordinates whose columns correspond to individual "lines".
        New "lines" may stop or start at rows in the middle of the array.
    ys_sorted : ndarray, optional
        The corresponding *y* coordinates. Returned if `ys` is not ``None``.

    Example
    -------

    >>> import climopy as climo
    ... climo.linetrack(
    ...    [
    ...        [30, 20],
    ...        [22],
    ...        [24],
    ...        [32, 25],
    ...        [26, 40, 33],
    ...        [45],
    ...        [20, 47],
    ...        [23, 50],
    ...    ],
    ...    ntrack=3,
    ... )

    """
    # Parse input
    if ys is None:
        ys = xs  # not pretty but this simplifies the loop code
    if sep is None:
        sep = np.inf
    if seed is None:
        seed = []
    if (
        len(xs) != len(ys)
        or any(np.atleast_1d(x).size != np.atleast_1d(y).size for x, y in zip(xs, ys))
    ):
        raise ValueError('Mismatched geometry between x and y lines.')
    if ntrack is None:
        # WARNING: np.isscalar(np.array(1)) returns False so need to take
        # great pains to avoid length of unsized object errors
        ntrack = max(
            size if (size := getattr(x, 'size', None)) is not None  # noqa: E203, E231
            else 1 if np.isscalar(x) else len(x) for x in xs
        )

    # Arrays containing sorted lines in the output columns
    # NOTE: Need twice the maximum number of simultaneously tracked lines as columns
    # in the array. For example the following sequence with ntrack == 1 and sep == 5:
    # [20, NaN]
    # [22, NaN]
    # [NaN, 40]  # bigger than sep, so "new" line
    # For another example, the following sequence with ntrack == 2 and sep == np.inf:
    # [18, 32, NaN]
    # [20, 30, NaN]
    # [NaN, 33, 40]
    # The algorithm recognizes that even if ntrack is 2, if the remaining unmatched
    # points are even *farther* from the remaining previous points, this is a new line.
    nslots = 2 * ntrack
    seed = np.atleast_1d(seed)[:ntrack]
    with np.errstate(invalid='ignore'):
        xs_sorted = np.empty((len(xs) + 1, nslots)) * np.nan
        ys_sorted = np.empty((len(ys) + 1, nslots)) * np.nan
    xs_sorted[0, :seed.size] = seed

    for i, (ixs, iys) in enumerate(zip(xs, ys)):
        i += 1
        # Simple cases: No line tracking, no lines in this group, *or* no
        # lines in previous group so every single point starts a new line.
        # NOTE: It's ok if columns are occupied by more than one "line" as
        # long as there are NaNs between them. This is really just for plotting.
        ixs = np.atleast_1d(ixs)
        iys = np.atleast_1d(iys)
        if ixs.size == 0 or np.all(np.isnan(xs_sorted[i - 1, :])):
            ixs = ixs[:ntrack]
            iys = iys[:ntrack]
            xs_sorted[i, :ixs.size] = ixs
            ys_sorted[i, :iys.size] = iys
            continue

        # Find the points in the latest unsorted record that are *closest*
        # to the points in existing tracks, and the difference between them.
        mindiffs = np.empty((nslots,)) * np.nan
        argmins = np.empty((nslots,)) * np.nan
        for j, ix_prev in enumerate(xs_sorted[i - 1, :]):
            if np.isnan(ix_prev):
                continue
            diffs = np.abs(ixs - ix_prev)
            if np.min(diffs) > sep:
                continue  # not a member of *any* existing track
            mindiffs[j] = np.min(diffs)
            argmins[j] = np.argmin(diffs)

        # Handle *existing* tracks that continued or died out
        # Note that NaNs always go last in an argsort
        idxs = set()
        nlines = 0
        lines_old = np.argsort(mindiffs)  # prefer *smallest* differences
        for j in lines_old:
            idx = argmins[j]
            if np.isnan(idx):  # track dies
                continue
            if idx in idxs:  # already continued the line from a closer candidate
                continue
            if nlines >= ntrack:
                continue
            nlines += 1
            idxs.add(idx)
            xs_sorted[i, j] = ixs[int(idx)]
            ys_sorted[i, j] = iys[int(idx)]

        # Handle brand new tracks
        # NOTE: Set comparison {1, 2, 3} - {1, 2, np.nan} is {3} (extra values omitted)
        # NOTE: Set comparison {1} - {1.0} is {} (no issues with mixed float/int types)
        # NOTE: Should never run out of jslots since 'nlines' limits possible lines
        # TODO: Better way to prioritize "new" lines than random approach
        jslots, = np.where(np.all(np.isnan(xs_sorted[i - 1:i + 1, :]), axis=0))
        lines_new = set(range(len(ixs))) - set(argmins)
        for j, idx in enumerate(lines_new):
            if nlines >= ntrack:
                continue
            nlines += 1
            xs_sorted[i, jslots[j]] = ixs[int(idx)]
            ys_sorted[i, jslots[j]] = iys[int(idx)]

    # Return lines ignoring the "seed" and removing empty tracks
    mask = np.any(~np.isnan(xs_sorted[1:, :]), axis=0)
    xs_sorted = xs_sorted[1:, mask]
    ys_sorted = ys_sorted[1:, mask]
    if xs is not ys:
        return xs_sorted, ys_sorted
    else:
        return xs_sorted


@quack._xarray_zerofind_wrapper
@quack._pint_wrapper(('=x', '=y'), ('=x', '=y'))
def zerofind(x, y, axis=0, diff=None, centered=True, which='both', **kwargs):
    """
    Find the location of the zero value for a given data array.

    Parameters
    ----------
    x : array-like
        The coordinates.
    y : array-like
        The data for which we find zeros.
    axis : int, optional
        The axis along which zeros are found and (optionally) derivatives
        are taken. These will be connected along the other axis with `linetrack`.
    diff : int, optional
        How many times to differentiate along the axis.
    centered : bool, optional
        Whether to use centered finite differencing or half level differencing.
    which : {'negpos', 'posneg', 'both'}, optional
        Whether to find values that go from negative to positive, positive
        to negative, or both (the ``'min'`` and ``'max'`` keys really
        only apply to when `diff` is ``1``).
    **kwargs
        Passed to `linetrack` and used to group the locations into
        coherent tracks.

    Returns
    -------
    zx : array-like
        The zero locations.
    zy : array-like
        The zero values. If ``diff == 0`` these should all be equal to zero
        up to floating point precision. Otherwise these are the minima and
        maxima corresponding to the zero derivative locations.

    Example
    -------

    >>> import xarray as xr
    ... import climopy as climo
    ... ureg = climo.ureg
    ... x = np.arange(100)
    ... y = np.sort(np.random.rand(50, 10) - 0.5, axis=0)
    ... y = np.concatenate((y, y[::-1, :]), axis=0)
    ... xarr = xr.DataArray(
    ...     x * ureg.s,
    ...     dims=('x',), attrs={'long_name': 'x coordinate'}
    ... )
    ... with climo.internals.warnings._unit_stripped_ignore():
    ...     yarr = xr.DataArray(
    ...         y * ureg.m, name='variable',
    ...         dims=('x', 'y'), coords={'x': xarr}
    ...     )
    ... zx, zy = climo.zerofind(xarr, yarr, ntrack=2)

    """
    # Tests
    # TODO: Support tracking across single axis
    if which not in ('negpos', 'posneg', 'both'):
        raise ValueError(f'Invalid which {which!r}.')
    if y.ndim > 2:
        raise ValueError(f'Currently y must be 2D, got {y.ndim}D.')
    if x.ndim != 1 or y.shape[axis] != x.size:
        raise ValueError(f'Invalid shapes {x.shape=} and {y.shape=}.')
    is1d = y.ndim == 1
    y = np.moveaxis(y, axis, -1)
    if is1d:
        y = y[None, ...]
    reverse = x[1] - x[0] < 0  # TODO: check this works?
    nextra, naxis = y.shape

    # Optionally take derivatives onto half-levels and interpolate to
    # points on those half-levels.
    # NOTE: Doesn't matter if units are degrees or meters for latitude.
    dy = y
    if diff:  # not zero or None
        if centered:
            # Centered differencing onto same levels
            dy = deriv_uneven(x, y, axis=-1, order=diff, keepedges=True)
        else:
            # More accurate differencing onto half levels, then inteprolate back
            dx, dy = deriv_half(x, y, axis=-1, order=diff)
            dyi = dy.copy()
            for i in range(nextra):
                dyi[i, :] = np.interp(dx, x, dy[i, :])
            x, y = dx, dyi

    # Find where sign switches from +ve to -ve and vice versa
    zxs = []
    zys = []
    for k in range(nextra):
        # Get indices where values go positive to negative and vice versa
        # NOTE: Always have False where NaNs present
        posneg = negpos = ()
        with np.errstate(invalid='ignore'):
            if which in ('negpos', 'both'):
                negpos = np.diff(np.sign(y[k, :])) > 0
            if which in ('posneg', 'both'):
                posneg = np.diff(np.sign(dy[k, :])) < 0

        # Interpolate to exact zero locations and values at those locations
        izxs = []
        izys = []
        for j, mask in enumerate((negpos, posneg)):
            idxs, = np.where(mask)  # NOTE: for empty array, yields nothing
            for idx in idxs:
                # Need 'x' of segment to be *increasing* for interpolation
                if (not reverse and j == 0) or (reverse and j == 1):
                    segment = slice(idx, idx + 2)
                else:
                    segment = slice(idx + 1, idx - 1, -1)
                x_segment = x[segment]
                y_segment = y[k, segment]
                dy_segment = dy[k, segment]
                if x_segment.size in (0, 1):
                    continue  # weird error
                zx = np.interp(0, dy_segment, x_segment, left=np.nan, right=np.nan)
                if np.isnan(zx):  # no extrapolation!
                    continue
                zy = np.interp(zx, x_segment, y_segment)
                izxs.append(zx)
                izys.append(zy)  # record

        # Add to list
        # NOTE: Must use lists because number of zeros varies
        zxs.append(izxs)
        zys.append(izys)

    # Return locations and values
    zxs, zys = linetrack(zxs, zys, **kwargs)
    if not zxs.size:
        warnings._warn_climopy(f'No zeros found for data {y!r}.')
    if is1d:
        zxs, zys = zxs[0, :], zys[0, :]
    zxs = np.moveaxis(zxs, -1, axis)
    zys = np.moveaxis(zys, -1, axis)
    return zxs, zys
