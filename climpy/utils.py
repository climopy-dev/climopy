#!/usr/bin/env python3
"""
Includes miscellaneous mathematical functions.
"""
import numpy as np
from .diff import deriv_half, deriv_uneven
from .internals import quack

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
    out = np.empty(dt.shape + (6,), dtype='u4')

    # Decompose calendar floors
    # NOTE: M8 is datetime64, m8 is timedelta64
    Y, M, D, h, m, s = [dt.astype(f'M8[{x}]') for x in 'YMDhms']
    out[..., 0] = Y + 1970  # Gregorian Year
    out[..., 1] = (M - Y) + 1  # month
    out[..., 2] = (D - M) + 1  # day
    out[..., 3] = (dt - D).astype('m8[h]')  # hour
    out[..., 4] = (dt - h).astype('m8[m]')  # minute
    out[..., 5] = (dt - m).astype('m8[s]')  # second
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


def intersection(x, segment1, segment2, xlog=False):
    """
    Find the (first) intersection point for two line segments.

    Parameters
    ----------
    x : ndarray
        The *x* coordinates.
    segment1, segment2 : ndarray
        The two lists of *y* coordinates.
    xlog : bool, optional
        Whether to find the *x* coordinate intersection in logarithmic space.
    """
    # Initial stuff
    segment1, segment2 = np.array(segment1), np.array(segment2)
    if xlog:  # transform x coordinates optionally
        transform = lambda x: np.log10(x)
        itransform = lambda x: 10 ** x
    else:
        transform = itransform = lambda x: x

    # Get intersection
    diff = segment1 - segment2
    if (diff > 0).all() or (diff < 0).all():
        print('Warning: No intersections found.')
        return np.nan, np.nan
    idx = np.where(diff > 0)[0][0]  # two-element vectors
    x, y = diff[idx - 1:idx + 1], transform(x[idx - 1:idx + 1])
    px = itransform(y[0] + (0 - x[0]) * ((y[1] - y[0]) / (x[1] - x[0])))
    x, y = y, segment2[idx - 1:idx + 1]
    py = y[0] + (transform(px) - x[0]) * ((y[1] - y[0]) / (x[1] - x[0]))
    return px, py


# TODO: Support pint quantities here
def linetrack(xs, ys=None, /, track=True, sep=np.inf, seed=None, N=10):
    """
    Track individual "lines" across lists of coordinates.

    Parameters
    ----------
    xs : list of lists
        The locations to be grouped into lines.
    ys : list of lists, optional
        The values corresponding to the locations `xs`.
    track : bool, optional
        Whether to track the lines. If ``False`` this function simply puts
        the sublists into rows of a 2D array in no particular order.
    sep : float, optional
        The maximum separation between points belonging to the same "track".
        Larger separations will cause the algorithm to begin a new track. The
        default behavior is to not separate into tracks this way.
    seed : float or list of float, optional
        The track or tracks you want the algorithm to pick up if the number of
        tracks is limited by `N`.
    N : int, optional
        The maximum number of values to be simultaneously tracked. This can
        be used to ignore spurious values in combination with `seed`.

    Returns
    -------
    xs_sorted : ndarray
        2D array of *x* coordinates whose columns correspond to individual "lines".
        New "lines" may stop or start at rows in the middle of the array.
    ys_sorted : ndarray, optional
        The corresponding *y* coordinates. Returned if `ys` is not ``None``.

    Example
    -------

    >>> import climpy
    ... climpy.linetrack(
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
    ...    N=3,
    ... )

    """
    # Parse input
    if ys is None:
        ys = xs  # not pretty but this simplifies the loop code
    if seed is None:
        seed = []
    if len(xs) != len(ys) or any(len(x) != len(y) for x, y in zip(xs, ys)):
        raise ValueError('Mismatched geometry between x and y lines.')

    # Arrays containing sorted lines in the output columns
    # NOTE: Need twice the maximum number of simultaneously tracked lines
    # as columns in the array. For example to generate the sequence with N = 1:
    # [20, NaN]
    # [22, NaN]
    # [NaN, 40]  # bigger than sep, so "new" line
    # [NaN, 42]
    seed = np.atleast_1d(seed)[:N]
    xs_sorted = np.empty((len(xs) + 1, N * 2)) * np.nan
    ys_sorted = np.empty((len(ys) + 1, N * 2)) * np.nan
    xs_sorted[0, :seed.size] = seed

    for i, (ixs, iys) in enumerate(zip(xs, ys)):
        i += 1
        # Simple cases: No line tracking, no lines in this group, *or* no
        # lines in previous group so every single point starts a new line.
        # NOTE: It's ok if columns are occupied by more than one "line" as
        # long as there are NaNs between them. This is really just for plotting.
        ixs = np.atleast_1d(ixs)[:N]
        iys = np.atleast_1d(iys)[:N]
        if not track or ixs.size == 0 or np.all(np.isnan(xs_sorted[i - 1, :])):
            xs_sorted[i, :ixs.size] = ixs
            ys_sorted[i, :iys.size] = iys
            continue

        # Find the points in the latest unsorted record that are *closest*
        # to the points in existing tracks, and the difference between them.
        mindiffs = np.empty((N * 2,)) * np.nan
        argmins = np.empty((N * 2,)) * np.nan
        for j, ix_prev in enumerate(xs_sorted[i - 1, :]):
            print('previous', j, ix_prev)
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
            if nlines >= N:
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
            if nlines >= N:
                continue
            nlines += 1
            xs_sorted[i, jslots[j]] = ixs[int(idx)]
            ys_sorted[i, jslots[j]] = iys[int(idx)]

    # Return lines, ignoring the "seed"
    if xs is not ys:
        return xs_sorted[1:, :], ys_sorted[1:, :]
    else:
        return xs_sorted[1:, :]


@quack._xarray_xy_wrapper
@quack._pint_wrapper(('=x', '=y'), ('=x', '=y'))
def zerofind(x, y, axis=-1, diff=0, centered=True, which='both', **kwargs):
    """
    Find the location of the zero value for a given data array.

    Parameters
    ----------
    x : array-like
        The coordinates.
    y : array-like
        The data for which we find zeros.
    axis : int, optional
        The axis along which zeros are tracked.
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
    """
    # Tests
    # TODO: Support tracking across single axis
    if which not in ('negpos', 'posneg', 'both'):
        raise ValueError(f'Invalid which {which!r}.')
    if y.ndim > 2:
        raise ValueError(f'Currently y must be 2D, got {y.ndim}D.')
    if x.ndim != 1 or y.shape[axis] != x.size:
        raise ValueError(f'Invalid shapes {x.shape=} and {y.shape=}.')
    y = np.moveaxis(y, axis, -1)
    reverse = x[1] - x[0] < 0  # TODO: check this works?
    nextra, naxis = y.shape

    # Optionally take derivatives onto half-levels and interpolate to
    # points on those half-levels.
    # NOTE: Doesn't matter if units are degrees or meters for latitude.
    dy = y
    if diff:  # not zero or None
        if centered:
            # Centered differencing onto same levels
            dy = deriv_uneven(x, y, axis=-1, order=diff)
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
        posneg = negpos = ()
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
    if nextra == 1:  # there were no other dimensions; only return the sublist
        zxs, zys = np.array(zxs[0]), np.array(zys[0])
    else:
        zxs, zys = linetrack(zxs, zys, **kwargs)
    if not zxs.size:
        raise ValueError(f'No zeros found for data {y!r}.')

    return zxs, zys
