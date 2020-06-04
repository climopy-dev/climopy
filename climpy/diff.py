#!/usr/bin/env python3
"""
Various finite difference schemes.
"""
# TODO: Add integration schemes! Will be simple to implement, they are just cumsums.
import numpy as np
from .internals import docstring, quack, warnings

__all__ = [
    'integral',
    'deriv1', 'deriv2', 'deriv3',
    'deriv_half', 'deriv_uneven',
]

# Used below
sentinel = object()

# Docstring snippets
_axis_dim = """
axis : int, optional
    Axis along which %(action)s.
dim : str, optional
    *For `xarray.DataArray` input only*.
    Named dimension along which the %(action)s.
"""

docstring.snippets['hist.axis'] = _axis_dim % {'action': 'histogram is computed'}

docstring.snippets['deriv.axis'] = _axis_dim % {'action': 'derivative is taken'}

docstring.snippets['interp.axis'] = _axis_dim % {'action': 'interpolation is done'}

docstring.snippets['integral.axis'] = _axis_dim % {'action': 'integral is taken'}

docstring.snippets['deriv.args'] = """
h : float or array-like
    The step size. If non-singleton, the step size is `h[1] - h[0]`.
y : array-like
    The data.
"""

docstring.snippets['deriv.kwargs'] = """
accuracy : {{0, 2, 4, 6}}, optional
    Accuracy of finite difference approximation. ``0`` corresponds
    to differentiation onto half-levels. ``2``, ``4``, and ``6`` correspond
    to centered accuracies of :math:`h^2`, :math:`h^4`, and :math:`h^6`,
    respectively.  See `this wikipedia page \
<https://en.wikipedia.org/wiki/Finite_difference_coefficient>`__
    for the table of coefficients for each accuracy.
keepleft, keepright, keepedges : bool, optional
    Whether to fill left, right, or both edge positions with progressively
    lower-`accuracy` finite difference estimates to prevent reducing
    the dimension size along axis `axis`.
"""

docstring.snippets['deriv.returns'] = """
diff : array-like
    The "derivative". The length of axis `axis` may differ from `y`
    depending on the `keepleft`, `keepright`, and `keepedges` settings.
"""

docstring.snippets['basic.params'] = """
x : array-like
    A 1-d coordinate vector. Must match the shape of `y` on axis `axis`.
y : array-like
    The data.
"""

docstring.snippets['uneven.params'] = """
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
    Retrieve the Fornberg (1988) coefficients for estimating derivatives
    of arbitrary order at arbitrary points as recommended by
    `this post <https://scicomp.stackexchange.com/a/481/24014>`__.
    Code was adapted from `this example \
<https://numdifftools.readthedocs.io/en/latest/_modules/numdifftools/fornberg.html>`__.
    """
    # NOTE: The order of coordinates does not matter (can be descending or
    # even non-monotonic evidently).
    x = np.asarray(x)
    n = x.shape[-1]
    if order >= n:
        raise ValueError(f'Derivative order {order} must be smaller than {n}.')
    weights = np.zeros((n, order + 1))  # includes zeroth weights
    weights[..., 0, 0] = 1
    hprod_prev = 1
    for i in range(1, n):
        # Set terms up
        idxs = np.arange(0, min(i, order) + 1)
        hprod = np.prod(x[..., i] - x[..., :i], axis=-1)
        h0 = x[..., i] - x0
        h0_prev = x[..., i - 1] - x0
        for ii in range(i):
            w = weights[..., ii, idxs]
            w_prev = weights[..., ii, idxs - 1]
            # The 'for m := 0 to min(n, M)' part
            h = x[..., i] - x[..., ii]
            weights[..., ii, idxs] = (h0 * w - idxs * w_prev) / h
        # The 'for m := 0 to min(n, M)' part
        # Note we use w and w_prev from last loop iteration here
        weights[..., i, idxs] = (idxs * w_prev - h0_prev * w) * (hprod_prev / hprod)
        hprod_prev = hprod
    return weights[..., -1]


@quack._xarray_xy_wrapper
@quack._pint_wrapper(('=x', '=y'), '=x * y')
@docstring.add_snippets
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
    %(integral.axis)s

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


def _accuracy_check(n, accuracy, order=1):
    """
    Restrict the accuracy based on length of dimension.
    """
    absmin = order + 1  # minimum number of points for deriv
    finitemin = 1 + 2 * ((order + 1) // 2)  # e.g. 1, 2 --> 3; 3, 4 --> 5
    if n < absmin:  # allows odd-numbered derivs on half-levels
        raise ValueError('Need at least 2 points on derivative axis.')
    elif n < finitemin:
        if accuracy > 0:
            warnings._warn_climpy(
                f'Setting accuracy to 0 for derivative on length-{n} axis.'
            )
            accuracy = 0
    elif n < finitemin + 2:
        if accuracy > 2:
            warnings._warn_climpy(
                f'Setting accuracy to 2 for derivative on length-{n} axis.'
            )
            accuracy = 2
    elif n < finitemin + 4:
        if accuracy > 4:
            warnings._warn_climpy(
                f'Setting accuracy to 4 for derivative on length-{n} axis.'
            )
            accuracy = 4
    return accuracy


def _get_step(h):
    """
    Determines scalar step h.
    """
    h = np.atleast_1d(h)
    if len(h) == 1:
        return h[0]
    else:
        warnings._warn_climpy('Using difference between first 2 points for step size.')
        return h[1] - h[0]


@quack._xarray_xy_wrapper
@quack._pint_wrapper(('=x', '=y'), '=y / x')
@docstring.add_snippets
def deriv1(
    h, y, /, axis=0, accuracy=2, keepleft=False, keepright=False, keepedges=False
):
    """
    Return an estimate of the first derivative along an arbitrary axis using
    first order centered finite differencing.

    Parameters
    ----------
    %(deriv.args)s
    %(deriv.axis)s
    %(deriv.kwargs)s

    Returns
    -------
    %(deriv.returns)s

    See Also
    --------
    deriv_half, deriv_uneven
    """
    # Simple Euler scheme
    h = _get_step(h)
    ldiff = rdiff = ()
    if keepedges:
        keepleft = keepright = True

    # Checks
    n = y.shape[axis]
    accuracy = _accuracy_check(n, accuracy, order=1)

    # Derivative
    y = np.asarray(y)  # for safety
    y = np.moveaxis(y, axis, -1)
    if accuracy == 0:
        diff = (y[..., 1:] - y[..., :-1]) / h
    elif accuracy == 2:
        diff = (1 / 2) * (-y[..., :-2] + y[..., 2:]) / h
        if keepleft:
            ldiff = (
                deriv1(h, y[..., :2], axis=-1, keepleft=True, accuracy=0),
            )  # one-tuple
        if keepright:
            rdiff = (
                deriv1(h, y[..., -2:], axis=-1, keepright=True, accuracy=0),
            )
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
            ldiff = (
                deriv1(h, y[..., :3], axis=-1, keepleft=True, accuracy=2),
            )  # one-tuple
        if keepright:
            rdiff = (
                deriv1(h, y[..., -3:], axis=-1, keepright=True, accuracy=2),
            )
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
            ldiff = (
                deriv1(h, y[..., :5], axis=-1, keepleft=True, accuracy=4),
            )  # one-tuple
        if keepright:
            rdiff = (
                deriv1(h, y[..., -5:], axis=-1, keepright=True, accuracy=4),
            )
    else:
        raise ValueError('Invalid accuracy. Choose form O(h^2), O(h^4), or O(h^6).')
    diff = np.concatenate((*ldiff, diff, *rdiff), axis=-1)
    return np.moveaxis(diff, -1, axis)


@quack._xarray_xy_wrapper
@quack._pint_wrapper(('=x', '=y'), '=y / x ** 2')
@docstring.add_snippets
def deriv2(
    h, y, /, axis=0, accuracy=2, keepleft=False, keepright=False, keepedges=False
):
    """
    Return an estimate of the second derivative along an arbitrary axis using
    second order centered finite differencing.

    Parameters
    ----------
    %(deriv.args)s
    %(deriv.axis)s
    %(deriv.kwargs)s

    Returns
    -------
    %(deriv.returns)s

    See Also
    --------
    deriv1, deriv_uneven
    """
    # Simple Euler scheme
    h = _get_step(h)
    ldiff = rdiff = ()
    if keepedges:
        keepleft = keepright = True

    # Checks
    n = y.shape[axis]
    accuracy = _accuracy_check(n, accuracy, order=2)

    # Derivative
    y = np.asarray(y)  # for safety
    y = np.moveaxis(y, axis, -1)
    if accuracy == 2:
        diff = (y[..., :-2] - 2 * y[..., 1:-1] + y[..., 2:]) / h ** 2
        if keepleft:  # just append the leftmost 2nd deriv
            ldiff = (diff[..., :1],)
        if keepright:  # just append the rightmost 2nd deriv
            rdiff = (diff[..., -1:],)
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
            ldiff = (
                deriv2(h, y[..., :3], axis=-1, keepleft=True, accuracy=2),
            )
        if keepright:
            rdiff = (
                deriv2(h, y[..., -3:], axis=-1, keepright=True, accuracy=2),
            )
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
            ldiff = (
                deriv2(h, y[..., :5], axis=-1, keepleft=True, accuracy=4),
            )
        if keepright:
            rdiff = (
                deriv2(h, y[..., -5:], axis=-1, keepright=True, accuracy=4),
            )
    else:
        raise ValueError('Invalid accuracy. Choose form O(h^2), O(h^4), or O(h^6).')
    diff = np.concatenate((*ldiff, diff, *rdiff), axis=-1)
    return np.moveaxis(diff, -1, axis)


@quack._xarray_xy_wrapper
@quack._pint_wrapper(('=x', '=y'), '=y / x ** 3')
@docstring.add_snippets
def deriv3(
    h, y, /, axis=0, accuracy=2, keepleft=False, keepright=False, keepedges=False
):
    """
    Return an estimate of the third derivative along an arbitrary axis using
    third order centered finite differencing.

    Parameters
    ----------
    %(deriv.args)s
    %(deriv.axis)s
    %(deriv.kwargs)s

    Returns
    -------
    %(deriv.returns)s

    See Also
    --------
    deriv1, deriv_uneven
    """
    # Simple Euler scheme
    h = _get_step(h)
    ldiff = rdiff = ()
    if keepedges:
        keepleft = keepright = True

    # Checks
    n = y.shape[axis]
    accuracy = _accuracy_check(n, accuracy, order=3)

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
            ldiff = (diff[..., :1],)
        if keepright:  # just append the rightmost 3rd deriv
            rdiff = (diff[..., -1:],)
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
            ldiff = (
                deriv3(h, y[..., :4], axis=-1, keepleft=True, accuracy=0),
            )
        if keepright:
            rdiff = (
                deriv3(h, y[..., -4:], axis=-1, keepright=True, accuracy=0),
            )
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
            ldiff = (
                deriv3(h, y[..., :5], axis=-1, keepleft=True, accuracy=2),
            )
        if keepright:
            rdiff = (
                deriv3(h, y[..., -5:], axis=-1, keepright=True, accuracy=2),
            )
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
            ldiff = (
                deriv2(h, y[..., :7], axis=-1, keepleft=True, accuracy=4),
            )
        if keepright:
            rdiff = (
                deriv2(h, y[..., -7:], axis=-1, keepright=True, accuracy=4),
            )
    else:
        raise ValueError('Invalid accuracy. Choose form O(h^2), O(h^4), or O(h^6).')
    diff = np.concatenate((*ldiff, diff, *rdiff), axis=-1)
    return np.moveaxis(diff, -1, axis)


def _xy_standardize(x, y, /, axis=0):
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


@quack._xarray_xy_wrapper
@quack._pint_wrapper(('=x', '=y'), ('=x', '=y / x ** {order}'), order=1)
@docstring.add_snippets
def deriv_half(x, y, /, order=1, axis=0):
    """
    Return an arbitrary order finite difference approximation by taking successive
    half-level differences. This will change both the length of the data and
    the *x* coordinates of the data. While this is not always practical, it
    retains data resolution better than the centered methods.

    Parameters
    ----------
    %(uneven.params)s
    %(deriv.axis)s

    Returns
    -------
    x : array-like
        The new *x* coordinates.
    diff : array-like
        The "derivative".
    """
    # Standardize
    x, y = _xy_standardize(x, y, axis)
    if x.ndim > 1:
        x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)

    # Take derivatives on half levels
    diff = y
    for i in range(order):
        diff = (diff[..., 1:] - diff[..., :-1]) / (x[..., 1:] - x[..., :-1])
        x = (x[..., 1:] + x[..., :-1]) / 2.0

    # Return derivative
    if x.ndim > 1:
        np.moveaxis(x, -1, axis)
    diff = np.moveaxis(diff, -1, axis)
    return x, diff


@quack._xarray_xy_wrapper
@quack._pint_wrapper(('=x', '=y'), ('=x', '=y / x ** {order}'), order=1)
@docstring.add_snippets
def deriv_uneven(x, y, /, order=1, axis=0, accuracy=2, keepedges=False):
    r"""
    Return an arbitrary order centered finite difference approximation for
    arbitrarily spaced coordinates using the :cite:`1988:fornberg` method.

    Parameters
    ----------
    %(uneven.params)s
    %(deriv.axis)s
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
    deriv1, deriv_half
    """
    # Standardize x and y
    x, y = _xy_standardize(x, y, axis=axis)
    if x.ndim > 1:
        x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)

    # Get coefficients for blocks of x-coordinates matching
    # the length of respective centered finite difference methods.
    # NOTE: We figure out edge derivatives with the fornberg algorithm using
    # the same number of points as centered samples, but could also take
    # approach of even finite difference methods and progressively reduce
    # numbers of points used on edge.
    n = y.shape[-1]
    nblock = 1 + accuracy + 2 * ((order - 1) // 2)
    nhalf = (nblock - 1) // 2
    diff = np.empty(y.shape) * np.nan
    offset = 0 if keepedges else nhalf
    for i in range(offset, n - offset):
        # Get segment of x to pass to algorithm
        # NOTE: To prevent overfitting we want to try to try to reduce segment
        # length such that evenly spaced points yield standard lower-accuracy
        # finite difference coefficients. If this is not possible, reduce
        # to the bare minimum of points required for finite differencing, and
        # the resulting coefficients will be the same independent of x0. This
        # may pad the edges with identical finite differences.
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
        x = x[..., nhalf:-nhalf]
        diff = diff[..., nhalf:-nhalf]
    if x.ndim > 1:
        x = np.moveaxis(x, -1, axis)
    diff = np.moveaxis(diff, -1, axis)
    return x, diff
