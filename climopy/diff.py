#!/usr/bin/env python3
"""
Various finite difference schemes.
"""
# TODO: Add integration schemes! Will be simple to implement, they are just cumsums.
import numpy as np

from .internals import ic  # noqa: F401
from .internals import docstring, quack, warnings

__all__ = [
    'integral', 'deriv_even', 'deriv_half', 'deriv_uneven',
]

# Docstring snippets
# NOTE: Includes snippets used in other parts of code
docstring.snippets['params_axisdim'] = """
axis : int, optional
    Axis along which %(name)s is taken.
dim : str, optional
    *For `xarray.DataArray` input only*.
    Named dimension along which %(name)s is taken.
"""

docstring.snippets['params_uneven'] = """
x : float or array-like
    The step size, a 1-d coordinate vector, or an array of coordinates
    matching the shape of `y`.
y : array-like
    The data.
order : int, optional
    The order of the derivative. Default is ``1``.
"""


def _fornberg_coeffs(x, x0, order=1):
    """
    Retrieve the Fornberg (1988) coefficients for estimating derivatives of arbitrary
    order at arbitrary points as recommended by `this post \
<https://scicomp.stackexchange.com/a/481/24014>`__.
    Code was adapted from `this example \
<https://numdifftools.readthedocs.io/en/latest/_modules/numdifftools/fornberg.html>`__.

    Parameters
    ----------
    x : array-like, optional
        Array with rightmost dimension representing sample coordinates.
    x0 : array-like, optional
        Array representing coordinate selection. Rank must be one less than `x`.
    order : int, optional
        The order of the derivative.
    """
    # Begin loop
    # NOTE: The order of coordinates does not matter (can be descending or even
    # non-monotonic evidently).
    x = np.asarray(x)
    n = x.shape[-1]
    if order >= n:
        raise ValueError(f'Derivative order {order} must be smaller than {n}.')
    weights = np.zeros((*x.shape[:-1], n, order + 1))  # weights for all derivatives
    weights[..., 0, 0] = 1
    hprod_prev = 1
    for i in range(1, n):
        # Set terms up
        idxs = np.arange(0, min(i, order) + 1)
        hprod = np.prod(x[..., i:i + 1] - x[..., :i], axis=-1, keepdims=True)
        h0 = x[..., i:i + 1] - x0[..., None]
        h0_prev = x[..., i - 1:i] - x0[..., None]
        for ii in range(i):
            w = weights[..., ii, idxs]
            w_prev = weights[..., ii, idxs - 1]
            # The 'for m := 0 to min(n, M)' part
            h = x[..., i:i + 1] - x[..., ii:ii + 1]
            weights[..., ii, idxs] = (h0 * w - idxs * w_prev) / h

        # The 'for m := 0 to min(n, M)' part
        # Note we use w and w_prev from last loop iteration here
        weights[..., i, idxs] = (idxs * w_prev - h0_prev * w) * (hprod_prev / hprod)
        hprod_prev = hprod

    return weights[..., -1]  # weights for order'th derivative (rightmost selection)


@quack._xarray_xyy_wrapper
@quack._pint_wrapper(('=x', '=y'), '=x * y')
@docstring.inject_snippets(name='integral')
def integral(x, y, /, y0=0, axis=0):
    """
    Return the integral approximation along an arbitrary axis.

    Parameters
    ----------
    x : array-like
        A 1-d coordinate vector. Must match the shape of `y` on axis `axis`.
    y : array-like
        The data.
    y0 : float or array-like, optional
        Constant offset added to the integral. Must be scalar or match the
        shape of `y`.
    %(params_axisdim)s

    Returns
    -------
    array-like
        The "integral".
    """
    x = np.atleast_1d(x)
    if x.size == 1:
        dx = x.item()
    else:
        dx = x[1:] - x[:-1]
        dx = np.concatenate((dx[:1], dx))
        shape = [1] * y.ndim
        shape[axis] = dx.size
        dx = np.reshape(dx, shape)  # add singleton dimensions
    return y0 + (y * dx).cumsum(axis=axis)


def _restrict_accuracy(n, accuracy, order=1):
    """
    Restrict the accuracy based on length of dimension.
    """
    absmin = order + 1  # minimum number of points for deriv
    finitemin = 1 + 2 * ((order + 1) // 2)  # e.g. 1, 2 --> 3; 3, 4 --> 5
    if n < absmin:  # allows odd-numbered derivs on half-levels
        raise ValueError('Need at least 2 points on derivative axis.')
    elif n < finitemin:
        if accuracy > 0:
            warnings._warn_climopy(
                f'Setting accuracy to 0 for derivative on length-{n} axis.'
            )
            accuracy = 0
    elif n < finitemin + 2:
        if accuracy > 2:
            warnings._warn_climopy(
                f'Setting accuracy to 2 for derivative on length-{n} axis.'
            )
            accuracy = 2
    elif n < finitemin + 4:
        if accuracy > 4:
            warnings._warn_climopy(
                f'Setting accuracy to 4 for derivative on length-{n} axis.'
            )
            accuracy = 4
    return accuracy


@quack._xarray_xyy_wrapper
@quack._pint_wrapper(('=x', '=y'), '=y / x')
def _deriv_first(
    h, y, /, axis=0, accuracy=2, keepleft=False, keepright=False, keepedges=False
):
    """
    Return the first order centered finite difference.
    """
    # Simple Euler scheme
    h = quack._as_step(h)
    ldiff = rdiff = ()
    if keepedges:
        keepleft = keepright = True

    # Checks
    n = y.shape[axis]
    accuracy = _restrict_accuracy(n, accuracy, order=1)

    # Derivative
    y = np.asarray(y)  # for safety
    y = np.moveaxis(y, axis, -1)
    if accuracy == 0:
        diff = (y[..., 1:] - y[..., :-1]) / h
    elif accuracy == 2:
        diff = (1 / 2) * (-y[..., :-2] + y[..., 2:]) / h
        if keepleft:
            ldiff = _deriv_first(h, y[..., :2], axis=-1, keepleft=True, accuracy=0),
        if keepright:
            rdiff = _deriv_first(h, y[..., -2:], axis=-1, keepright=True, accuracy=0),
    elif accuracy == 4:
        diff = (
            (1 / 12)
            * (
                y[..., :-4]
                - 8 * y[..., 1:-3]
                + 8 * y[..., 3:-1]
                - y[..., 4:]
            )
            / h
        )
        if keepleft:
            ldiff = _deriv_first(h, y[..., :3], axis=-1, keepleft=True, accuracy=2),
        if keepright:
            rdiff = _deriv_first(h, y[..., -3:], axis=-1, keepright=True, accuracy=2),
    elif accuracy == 6:
        diff = (
            (1 / 60)
            * (
                - y[..., :-6]
                + 9 * y[..., 1:-5]
                - 45 * y[..., 2:-4]
                + 45 * y[..., 4:-2]
                - 9 * y[..., 5:-1]
                + y[..., 6:]
            )
            / h
        )
        if keepleft:
            ldiff = _deriv_first(h, y[..., :5], axis=-1, keepleft=True, accuracy=4),
        if keepright:
            rdiff = _deriv_first(h, y[..., -5:], axis=-1, keepright=True, accuracy=4),
    else:
        raise ValueError('Invalid accuracy. Choose form O(h^2), O(h^4), or O(h^6).')
    diff = np.concatenate((*ldiff, diff, *rdiff), axis=-1)
    return np.moveaxis(diff, -1, axis)


@quack._xarray_xyy_wrapper
@quack._pint_wrapper(('=x', '=y'), '=y / x ** 2')
def _deriv_second(
    h, y, /, axis=0, accuracy=2, keepleft=False, keepright=False, keepedges=False
):
    """
    Return the second order centered finite difference.
    """
    # Simple Euler scheme
    h = quack._as_step(h)
    ldiff = rdiff = ()
    if keepedges:
        keepleft = keepright = True

    # Checks
    n = y.shape[axis]
    accuracy = _restrict_accuracy(n, accuracy, order=2)

    # Derivative
    y = np.asarray(y)  # for safety
    y = np.moveaxis(y, axis, -1)
    if accuracy == 2:
        diff = (y[..., :-2] - 2 * y[..., 1:-1] + y[..., 2:]) / h ** 2
        if keepleft:  # just append the leftmost 2nd deriv
            ldiff = diff[..., :1],
        if keepright:  # just append the rightmost 2nd deriv
            rdiff = diff[..., -1:],
    elif accuracy == 4:
        diff = (
            (1 / 12)
            * (
                - y[..., :-4]
                + 16 * y[..., 1:-3]
                - 30 * y[..., 2:-2]
                + 16 * y[..., 3:-1]
                - y[..., 4:]
            )
            / h ** 2
        )
        if keepleft:
            ldiff = _deriv_second(h, y[..., :3], axis=-1, keepleft=True, accuracy=2),
        if keepright:
            rdiff = _deriv_second(h, y[..., -3:], axis=-1, keepright=True, accuracy=2),
    elif accuracy == 6:
        diff = (
            (1 / 180)
            * (
                2 * y[..., :-6]
                - 27 * y[..., 1:-5]
                + 270 * y[..., 2:-4]
                - 490 * y[..., 3:-3]
                + 270 * y[..., 4:-2]
                - 27 * y[..., 5:-1]
                + 2 * y[..., 6:]
            )
            / h ** 2
        )
        if keepleft:
            ldiff = _deriv_second(h, y[..., :5], axis=-1, keepleft=True, accuracy=4),
        if keepright:
            rdiff = _deriv_second(h, y[..., -5:], axis=-1, keepright=True, accuracy=4),
    else:
        raise ValueError('Invalid accuracy. Choose form O(h^2), O(h^4), or O(h^6).')
    diff = np.concatenate((*ldiff, diff, *rdiff), axis=-1)
    return np.moveaxis(diff, -1, axis)


@quack._xarray_xyy_wrapper
@quack._pint_wrapper(('=x', '=y'), '=y / x ** 3')
def _deriv_third(
    h, y, /, axis=0, accuracy=2, keepleft=False, keepright=False, keepedges=False
):
    """
    Return the third order centered finite difference.
    """
    # Simple Euler scheme
    h = quack._as_step(h)
    ldiff = rdiff = ()
    if keepedges:
        keepleft = keepright = True

    # Checks
    n = y.shape[axis]
    accuracy = _restrict_accuracy(n, accuracy, order=3)

    # Derivative
    y = np.asarray(y)  # for safety
    y = np.moveaxis(y, axis, -1)
    if accuracy == 0:
        diff = (
            -y[..., :-3]
            + 3 * y[..., 1:-2]
            - 3 * y[..., 2:-1]
            + y[..., 3:]
        ) / h ** 3
        if keepleft:  # just append the leftmost 3rd deriv
            ldiff = diff[..., :1],
        if keepright:  # just append the rightmost 3rd deriv
            rdiff = diff[..., -1:],
    elif accuracy == 2:
        diff = (
            (1 / 2)
            * (
                - y[..., :-4]
                + 2 * y[..., 1:-3]
                - 2 * y[..., 3:-1]
                + y[..., 4:]
            )
            / h ** 3
        )
        if keepleft:
            ldiff = _deriv_third(h, y[..., :4], axis=-1, keepleft=True, accuracy=0),
        if keepright:
            rdiff = _deriv_third(h, y[..., -4:], axis=-1, keepright=True, accuracy=0),
    elif accuracy == 4:
        diff = (
            (1 / 8)
            * (
                y[..., :-6]
                - 8 * y[..., 1:-5]
                + 13 * y[..., 2:-4]
                - 13 * y[..., 4:-2]
                + 8 * y[..., 5:-1]
                - y[..., 6:]
            )
            / h ** 3
        )
        if keepleft:
            ldiff = _deriv_third(h, y[..., :5], axis=-1, keepleft=True, accuracy=2),
        if keepright:
            rdiff = _deriv_third(h, y[..., -5:], axis=-1, keepright=True, accuracy=2),
    elif accuracy == 6:
        diff = (
            (1 / 240)
            * (
                - 7 * y[..., :-8]
                + 72 * y[..., 1:-7]
                - 338 * y[..., 2:-6]
                + 488 * y[..., 3:-5]
                - 488 * y[..., 5:-3]
                + 338 * y[..., 6:-2]
                - 72 * y[..., 7:-1]
                + 7 * y[..., 8:]
            )
            / h ** 3
        )
        if keepleft:
            ldiff = _deriv_third(h, y[..., :7], axis=-1, keepleft=True, accuracy=4),
        if keepright:
            rdiff = _deriv_third(h, y[..., -7:], axis=-1, keepright=True, accuracy=4),
    else:
        raise ValueError('Invalid accuracy. Choose form O(h^2), O(h^4), or O(h^6).')
    diff = np.concatenate((*ldiff, diff, *rdiff), axis=-1)
    return np.moveaxis(diff, -1, axis)


@docstring.inject_snippets(name='derivative')
def deriv_even(h, y, /, axis=0, order=1, **kwargs):
    """
    Return an estimate of the first, second, or third order derivative along an
    arbitrary axis using centered finite differencing.

    Parameters
    ----------
    h : float or array-like
        The step size. If non-singleton, the step size is ``h[1] - h[0]``. An error
    y : array-like
        The data.
    %(params_axisdim)s
    accuracy : {0, 2, 4, 6}, optional
        Accuracy of finite difference approximation. ``0`` corresponds to
        differentiation onto half-levels. ``2``, ``4``, and ``6`` correspond to
        centered accuracies of :math:`h^2`, :math:`h^4`, and :math:`h^6`, respectively.
        See `this wikipedia page \
<https://en.wikipedia.org/wiki/Finite_difference_coefficient>`__
        for the table of coefficients for each accuracy.
    keepleft, keepright, keepedges : bool, optional
        Whether to fill left, right, or both edge positions with progressively
        lower-`accuracy` finite difference estimates to prevent reducing
        the dimension size along axis `axis`.

    Returns
    -------
    diff : array-like
        The "derivative". The length of axis `axis` may differ from `y`
        depending on the `keepleft`, `keepright`, and `keepedges` settings.

    See Also
    --------
    deriv_half, deriv_uneven
    """
    if order == 1:
        return _deriv_first(h, y, axis=axis, **kwargs)
    elif order == 2:
        return _deriv_second(h, y, axis=axis, **kwargs)
    elif order == 3:
        return _deriv_third(h, y, axis=axis, **kwargs)
    else:
        raise ValueError(f'Invalid derivative {order=}. Must fall between 1 and 3.')


def _standardize_x_y(x, y, /, axis=0):
    """
    Standardize the coordiantes.
    """
    x = np.atleast_1d(x).astype(float)
    y = np.atleast_1d(y).astype(float)
    ylen = y.shape[axis]
    if x.size == 1:  # just used the step size
        x = np.linspace(0, x[0] * (ylen - 1), ylen)
    xlen = x.shape[axis] if x.ndim > 1 else x.size
    if xlen != y.shape[axis] or x.ndim > 1 and x.shape != y.shape:
        raise ValueError(f'{x.shape=} incompatible with {y.shape=}')
    return x, y


@quack._xarray_xyxy_wrapper
@quack._pint_wrapper(('=x', '=y'), ('=x', '=y / x ** {order}'), order=1)
@docstring.inject_snippets(name='derivative')
def deriv_half(x, y, /, order=1, axis=0):
    """
    Return an arbitrary order finite difference approximation by taking successive
    half-level differences. This will change both the length of the data and
    the *x* coordinates of the data. While this is not always practical, it
    retains data resolution better than the centered methods.

    Parameters
    ----------
    %(params_uneven)s
    %(params_axisdim)s

    Returns
    -------
    x : array-like
        The new *x* coordinates.
    diff : array-like
        The "derivative".

    See also
    --------
    deriv_even, deriv_uneven

    Examples
    --------
    >>> import xarray as xr
    >>> import climopy as climo
    >>> x = xr.DataArray([0, 2, 4], name='x', dims='p', coords={'p': [1000, 800, 600]})
    >>> y = xr.DataArray([0, 4, 16], name='y', dims='p')
    >>> dx, dy = climo.deriv_half(x, y)
    >>> dx
    <xarray.DataArray 'x' (p: 2)>
    array([1., 3.])
    Coordinates:
      * p        (p) float64 900.0 700.0
    >>> dy
    <xarray.DataArray 'y' (p: 2)>
    array([2., 6.])
    Dimensions without coordinates: p
    """
    # Standardize
    x, y = _standardize_x_y(x, y, axis)
    if x.ndim > 1:
        x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)

    # Take derivatives on half levels
    diff = y
    for i in range(order):
        diff = (diff[..., 1:] - diff[..., :-1]) / (x[..., 1:] - x[..., :-1])
        x = 0.5 * (x[..., 1:] + x[..., :-1])

    # Return derivative
    if x.ndim > 1:
        x = np.moveaxis(x, -1, axis)
    diff = np.moveaxis(diff, -1, axis)
    return x, diff


@quack._xarray_xyy_wrapper
@quack._pint_wrapper(('=x', '=y'), '=y / x ** {order}', order=1)
@docstring.inject_snippets(name='derivative')
def deriv_uneven(x, y, /, order=1, axis=0, accuracy=2, keepedges=False):
    r"""
    Return an arbitrary order centered finite difference approximation for
    arbitrarily spaced coordinates using the :cite:`1988:fornberg` method.

    Parameters
    ----------
    %(params_uneven)s
    %(params_axisdim)s
    accuracy : {2, 4, 6, ...}, optional
        Accuracy of the finite difference approximation. This determines the
        number of terms that go into the :cite:`1988:fornberg` algorithm.
        Using too many terms can result in overfitting.
    keepedges : bool, optional
        Whether to fill left, right, or both edge positions with progressively
        lower-`accuracy` finite difference estimates to prevent reducing
        the dimension size along axis `axis`.

    Returns
    -------
    diff : array-like
        The "derivative".

    References
    ----------
    .. bibliography:: ../bibs/diff.bib

    See Also
    --------
    deriv_even, deriv_half
    """
    # Standardize x and y
    x, y = _standardize_x_y(x, y, axis=axis)
    if x.ndim > 1:
        x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)

    # Initial stuff
    # nblock = 1 + accuracy + 2 * ((order - 1) // 2)
    nhalf = accuracy // 2 + (order - 1) // 2
    offset = 0 if keepedges else nhalf
    n = y.shape[-1]

    # Get coefficients for blocks of x-coordinates matching the length of respective
    # centered finite difference methods.
    # NOTE: We figure out edge derivatives with the fornberg algorithm using the same
    # number of points as centered samples, but could also take approach of even finite
    # difference methods and progressively reduce numbers of points used on edge.
    diff = np.full(y.shape, np.nan)
    for i in range(offset, n - offset):
        # Get segment of x to pass to algorithm
        # NOTE: To prevent overfitting we want to try to try to reduce segment length
        # such that evenly spaced points yield standard lower-accuracy finite diff
        # coefficients. If this is not possible, reduce to the bare minimum of points
        # required for finite differencing, and the resulting coefficients will be the
        # same independent of x0. This may pad the edges with identical finite diffs.
        if i < nhalf:
            # left, right = 0, nblock  # causes overfitting!
            left, right = 0, max(order + 1, i * 2 + 1)
        elif i > n - nhalf - 1:
            # left, right = n - nblock, n  # causes overfitting!
            left, right = n - max(order + 1, (n - 1 - i) * 2 + 1), n
        else:
            left, right = i - nhalf, i + nhalf + 1

        # Get finite difference
        coeffs = _fornberg_coeffs(x[..., left:right], x[..., i], order=order)
        diff[..., i] = np.sum(coeffs * y[..., left:right], axis=-1)

    # Pad edges simply with edge derivatives
    if not keepedges:
        diff = diff[..., nhalf:-nhalf]
    diff = np.moveaxis(diff, -1, axis)

    return diff
