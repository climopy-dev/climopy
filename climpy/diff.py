#!/usr/bin/env python3
"""
Various finite difference schemes.
"""
# TODO: Add integration schemes! Will be simple to implement, they are just
# cumsums.
import numpy as np
from .arraytools import *

#------------------------------------------------------------------------------#
# Functions
#------------------------------------------------------------------------------#
def _step(h):
    """Determine scalar step h."""
    h = np.atleast_1d(h)
    if len(h)==1:
        return h[0]
    else:
        print('Warning: Using difference between first 2 points for step size.')
        return h[1]-h[0]

def integrate(x, y, y0=0, axis=0):
    """Integrate stuff."""
    dx = x[1:] - x[:-1]
    dx = np.concatenate((dx[:1], dx))
    shape = [1]*y.ndim
    shape[axis] = dx.size
    dx = np.reshape(dx, shape) # add singletons
    return y0 + (y*dx).cumsum()

def deriv(*args, **kwargs):
    """Alias for `deriv1`."""
    return deriv1(*args, **kwargs)

def deriv1(h, y, axis=0, accuracy=2, keepleft=False, keepright=False, keepedges=False):
    """
    First order finite differencing. Can be accurate to :math:`h^2`,
    :math:`h^4`, or :math:`h^6`. Reduces axis length by `accuracy`,
    except when `keepedges` = ``True``, in which case progressively
    lower-`accuracy` derivatives are used for edges.

    Parameters
    ----------
    h : float, array-like
        Step size. If array-like, step size is taken as `h[1] - h[0]`.
    y : array-like
        The data.
    axis : int, optional
        Axis along which derivative is taken.
    accuracy : {0, 2, 4, 6}
        Accuracy of Euler centered-finite difference method. "0" corresponds
        to differentiation onto half-levels, as in `diff`.
    keepleft, keepright, keepedges : bool, optional
        Whether to retain edge data with progressively lower-`accuracy`
        derivatives. That is when `keepedges` is ``True``, shape of output
        array is unchanged.

    Returns
    -------
    array-like
        The "derivative".

    Notes
    -----
    This was developed from the `wikipedia definition 
    <https://en.wikipedia.org/wiki/Finite_difference_coefficient>`_.

    See Also
    --------
    diff, deriv1_uneven
    """
    # Simple Euler scheme
    h = _step(h)
    ldiff, rdiff = (), ()
    if keepedges:
        keepleft = keepright = True

    # Checks
    n = y.shape[axis]
    if n<2:
        raise ValueError('Need at least 2 points on derivative axis.')
    elif n<3:
        if accuracy>0:
            print(f'Warning: Setting accuracy to 0 for derivative on length-{n} axis.')
            accuracy = 0
    elif n<5:
        if accuracy>2:
            print(f'Warning: Setting accuracy to 2 for derivative on length-{n} axis.')
            accuracy = 2
    elif n<7:
        if accuracy>4:
            print(f'Warning: Setting accuracy to 4 for derivative on length-{n} axis.')
            accuracy = 4

    # Derivative
    y = np.array(y) # for safety
    y = permute(y, axis)
    if accuracy==0:
        diff = (y[...,1:]-y[...,:-1])/h # keepleft and keepright are immaterial
    elif accuracy==2:
        diff = (1/2)*(y[...,2:]-y[...,:-2])/h
        if keepleft:
            ldiff = deriv1(h, y[...,:2], axis=-1, keepleft=True, accuracy=0), # one-tuple
        if keepright:
            rdiff = deriv1(h, y[...,-2:], axis=-1, keepright=True, accuracy=0),
        diff = np.concatenate((*ldiff, diff, *rdiff), axis=-1)
    elif accuracy==4:
        diff = (1/12)*(-y[...,4:] + 8*y[...,3:-1]
                - 8*y[...,1:-3] + y[...,:-4])/h
        if keepleft:
            ldiff = deriv1(h, y[...,:3], axis=-1, keepleft=True, accuracy=2), # one-tuple
        if keepright:
            rdiff = deriv1(h, y[...,-3:], axis=-1, keepright=True, accuracy=2),
        diff = np.concatenate((*ldiff, diff, *rdiff), axis=-1)
    elif accuracy==6:
        diff = (1/60)*(y[...,6:] - 9*y[...,5:-1] + 45*y[...,4:-2]
                - 45*y[...,2:-4] + 9*y[...,1:-5] - y[...,:-6])/h
        if keepleft:
            ldiff = deriv1(h, y[...,:5], axis=-1, keepleft=True, accuracy=4), # one-tuple
        if keepright:
            rdiff = deriv1(h, y[...,-5:], axis=-1, keepright=True, accuracy=4),
        diff = np.concatenate((*ldiff, diff, *rdiff), axis=-1)
    else:
        raise ValueError('Invalid accuracy; for now, choose form O(h^2), O(h^4), or O(h^6).')
    return unpermute(diff, axis)

def deriv2(h, y, axis=0, accuracy=2, keepleft=False, keepright=False, keepedges=False):
    """
    Second order finite differencing. Can be accurate to :math:`h^2`,
    :math:`h^4`, or :math:`h^6`. Reduces axis length by `accuracy`,
    except when `keepedges` = ``True``, in which case progressively
    lower-`accuracy` derivatives are used for edges.

    Notes
    -----
    For usage, see `deriv1`.

    See Also
    --------
    deriv2_uneven
    """
    # Simple Euler scheme
    # y = np.rollaxis(y, axis, y.ndim)
    h = _step(h)
    ldiff, rdiff = (), ()
    if keepedges:
        keepleft = keepright = True

    # Checks
    n = y.shape[axis]
    if n<2:
        raise ValueError('Need at least 2 points on derivative axis.')
    elif n<3:
        if accuracy>0:
            print(f'Warning: Setting accuracy to 0 for derivative on length-{n} axis.')
            accuracy = 0
    elif n<5:
        if accuracy>2:
            print(f'Warning: Setting accuracy to 2 for derivative on length-{n} axis.')
            accuracy = 2
    elif n<7:
        if accuracy>4:
            print(f'Warning: Setting accuracy to 4 for derivative on length-{n} axis.')
            accuracy = 4

    # Derivative
    y = np.array(y) # for safety
    y = permute(y, axis)
    if accuracy==2:
        diff = (y[...,2:] - 2*y[...,1:-1] + y[...,:-2])/h**2
        if keepleft: # just append the leftmost 2nd deriv
            ldiff = diff[...,:1],
        if keepright: # just append the rightmost 2nd deriv
            rdiff = diff[...,-1:],
        diff = np.concatenate((*ldiff, diff, *rdiff), axis=-1)
    elif accuracy==4:
        diff = (1/12)*(-y[...,4:] + 16*y[...,3:-1]
                - 30*y[...,2:-2] + 16*y[...,1:-3] - y[...,:-4])/h**2
        if keepleft:
            ldiff = deriv2(h, y[...,:3], axis=-1, keepleft=True, accuracy=2),
        if keepright:
            rdiff = deriv2(h, y[...,-3:], axis=-1, keepright=True, accuracy=2),
        diff = np.concatenate((*ldiff, diff, *rdiff), axis=-1)
    elif accuracy==6:
        diff = (1/180)*(2*y[...,6:] - 27*y[...,5:-1] + 270*y[...,4:-2]
                - 490*y[...,3:-3] + 270*y[...,2:-4] - 27*y[...,1:-5] + 2*y[...,:-6])/h**2
        if keepleft:
            ldiff = deriv2(h, y[...,:5], axis=-1, keepleft=True, accuracy=4),
        if keepright:
            rdiff = deriv2(h, y[...,-5:], axis=-1, keepright=True, accuracy=4),
        diff = np.concatenate((*ldiff, diff, *rdiff), axis=-1)
    else:
        raise ValueError('Invalid accuracy; for now, choose form O(h^2), O(h^4), or O(h^6).')
    # return np.rollaxis(diff, y.ndim-1, axis)
    return unpermute(diff, axis)

def deriv_uneven(*args, **kwargs):
    """Alias for `deriv1_uneven`."""
    return deriv_uneven(*args, **kwargs)

def deriv1_uneven(x, y, axis=0, keepedges=False):
    r"""
    Central numerical differentiation, uneven/even spacing.
    Reduces axis length by 2.

    Parameters
    ----------
    x : float, array-like
        If not array-like, the step size. If array-like, the *x*-coordinates.
        May exactly match shape of `y`, or match length of `y` along the axis
        `axis`.
    y : array-like
        The data.
    axis : int, optional
        Axis along which derivative is taken.
    keepedges : bool, default False
        When ``True``, result is padded along `axis` edges with the adjacent
        derivative estimates, to preserve the shape of `y`.

    Returns
    -------
    float
        The "derivative".

    Notes
    -----
    Equation is as follows:

    .. math::
        \frac{dy}{dx} = \dfrac{\frac{x_1 - x_0}{x_2 - x_1}(y_2 - y_1)
                        - \frac{x_2 - x_1}{x_1 - x_0}(y_1 - y_0)}{x_2 - x_0}

    * Weights the slope closer to center point more heavily.
    * Reduces to standard :math:`(y_2-y_0)/(x_2-x_0)` for even spacing,

    See Also
    --------
    diff, deriv1
    """
    # Preliminary stuff
    y = np.array(y)
    if not np.iterable(x):
        x = np.linspace(0, x*y.shape[axis]-1, y.shape[axis])
    else:
        x = np.array(x)
    xaxis = (axis if x.ndim>1 else 0) # if want x interpreted as vector
    if x.size==1: # is just the step size
        x = np.linspace(0, x[0]*y.shape[axis]-1, y.shape[axis])
    if x.shape[xaxis] != y.shape[axis]: # allow broadcasting rules to be used along other axes
        raise ValueError(f'x ({x.shape[xaxis]}) and y ({axis}) dimensions do not match along derivative axis.')

    # Checks
    n = y.shape[axis]
    if n<2:
        raise ValueError('Need at least 2 points on derivative axis.')
    elif n<3: # can only do a simple difference
        print('Warning: Taking difference between points as derivative.')
        diff = (y[...,1:] - y[...,:-1])/(x[...,1:] - x[...,:-1])
        return unpermute(diff, axis)

    # Formulation from stackoverflow, shown to be equivalent to the
    # one referenced below, but use x's instead of h's, and don't separte out terms
    # Original from this link: http://www.m-hikari.com/ijma/ijma-password-2009/ijma-password17-20-2009/bhadauriaIJMA17-20-2009.pdf
    x, y = permute(x, xaxis), permute(y, axis)
    x0, x1, x2 = x[...,:-2], x[...,1:-1], x[...,2:]
    y0, y1, y2 = y[...,:-2], y[...,1:-1], y[...,2:]
    h1, h2 = x1-x0, x2-x1 # the x-steps

    # Get
    # f = (x2 - x1)/(x2 - x0)
    # diff = (1-f)*(y2 - y1)/(x2 - x1) + f*(y1 - y0)/(x1 - x0) # version 1
    # print(h1.shape, h2.shape, y0.shape, y1.shape, y2.shape)
    diff = -h2*y0/(h1*(h1+h2)) - (h1-h2)*y1/(h1*h2) + h1*y2/(h2*(h1+h2))
    if keepedges: # pad with simple differences on edges
        bh = np.diff(x[...,:2], axis=-1)
        eh = np.diff(x[...,-2:], axis=-1)
        bdiff = deriv1(bh, y[...,:2], axis=-1, keepedges=True, accuracy=0)
        ediff = deriv1(eh, y[...,-2:], axis=-1, keepedges=True, accuracy=0)
        diff = np.concatenate((bdiff, diff, ediff), axis=-1)
    return unpermute(diff, axis)

def deriv2_uneven(x, y, axis=0, keepedges=False): # alternative
    """
    Second derivative adapted from Euler's method as in `deriv_uneven`.
    Formulation is found `here <https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/>`_.

    Notes
    -----
    For usage, see `deriv1_uneven`.

    See Also
    --------
    deriv2
    """
    # Preliminary stuff
    x, y = np.array(x), np.array(y) # precaution
    xaxis = (axis if x.ndim>1 else 0) # if want x interpreted as vector
    if x.shape[xaxis] != y.shape[axis]: # allow broadcasting rules to be used along other axes
        raise ValueError('x and y dimensions do not match along derivative axis.')
    x, y = permute(x, xaxis), permute(y, axis)

    # Formulation from this link: https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/#comments
    # Identical to this link: http://www.m-hikari.com/ijma/ijma-password-2009/ijma-password17-20-2009/bhadauriaIJMA17-20-2009.pdf
    x0, x1, x2 = x[...,:-2], x[...,1:-1], x[...,2:]
    y0, y1, y2 = y[...,:-2], y[...,1:-1], y[...,2:]
    h1, h2, H = x1-x0, x2-x1, x2-x0 # the x-steps
    # diff = 2*((x2-x1)*y0 - (x2-x0)*y1 + (x1-x0)*y2) / ((x2-x1)*(x1-x0)*(x2-x0)) # version 1
    # diff = 2*(y0/((x1-x0)*(x2-x0)) - y1/((x2-x1)*(x1-x0)) + y2/((x2-x1)*(x2-x0))) # version 2
    diff = 2*(h2*y0 - H*y1 + h1*y2)/(h1*h2*H) # version 3
    if keepedges: # need 3 points for 2nd derivative; can only pad edges with the nearest 2nd derivs
        diff = np.concatenate((diff[...,:1], diff, diff[...,-1:]), axis=-1)
    return unpermute(diff, axis)

def deriv3_uneven(x, y, axis=0, keepedges=False): # alternative
    """
    Third derivative adapted from Euler's method as in `deriv_uneven`.
    Formulation is found `here <https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/>`_.

    See Also
    --------
    deriv3
    """
    # Preliminary stuff
    x, y = np.array(x), np.array(y) # precaution
    xaxis = (axis if x.ndim>1 else 0) # if want x interpreted as vector
    if x.shape[xaxis] != y.shape[axis]: # allow broadcasting rules to be used along other axes
        raise ValueError('x and y dimensions do not match along derivative axis.')
    x, y = permute(x, xaxis), permute(y, axis)

    # Formulation from the same PDF shown above
    # First 4-point formula, which is uncentered and weird, so don't use it
    # x0, x1, x2, x3 = x[...,:-3], x[...,1:-2], x[...,2:-1], x[...,3:]
    # y0, y1, y2, y3 = y[...,:-3], y[...,1:-2], y[...,2:-1], y[...,3:]
    # h1, h2, h3, H = x1-x0, x2-x1, x3-x2, x3-x0 # the x-steps
    # diff = (6/h1)*(-y0/(h1*(h1+h2)*H) + y1/(h1*h2*(h1+h3)) - y2/((h1+h2)*h2*h3) + y3/(H*(h2+h3)*h))
    # Now the 5-point formula, evaluating on the center points
    # * Changed second line from h1+h3 to h2+h3; this was just by eyeballing the
    #   coefficients and trying to enforce symmetry.
    # * Turns out the *paper is incorrect*; original version leads to weird derivative,
    #   but by changing that term the results are as expected.
    x0, x1, x2, x3, x4 = x[...,:-4], x[...,1:-3], x[...,2:-2], x[...,3:-1], x[...,4:]
    y0, y1, y2, y3, y4 = y[...,:-4], y[...,1:-3], y[...,2:-2], y[...,3:-1], y[...,4:]
    h1, h2, h3, h4 = x1-x0, x2-x1, x3-x2, x4-x3
    H1, H2, H = h1+h2+h3, h2+h3+h4, h1+h2+h3+h4 # cleaned up their notation a bit; now
        # just replace H2 with H, and replace H2-h1 with H2; this makes more sense
    # for h in h1,h2,h3,h4,H1,H2,H: print(h.min(),h.max())
    # diff = (-.5*y0 + y1 - y3 + .5*y4)/(50e2**3) # Euler method; result is actually normal
    diff = 6*((h2-2*h3-h4)*y0/(h1*(h1+h2)*H1*H) \
            - (h1+h2-2*h3-h4)*y1/(h1*h2*(h2+h3)*H2) \
            + (h1+2*h2-2*h3-h4)*y2/((h1+h2)*h2*h3*(h3+h4)) \
            - (h1+2*h2-h3-h4)*y3/(H1*(h2+h3)*h3*h4) \
            + (h1+2*h2-h3)*y4/(H*H2*(h3+h4)*h4)) # holy shitballs
    if keepedges: # need 5 points for 3rd derivative; can only pad edges with the nearest 3rd derivs
        diff = np.concatenate((diff[...,:1], diff[...,:1], diff, diff[...,-1:], diff[...,-1:]), axis=-1)
    return unpermute(diff, axis)

def diff(x, y, axis=0):
    """
    Trivial differentiation onto half levels.
    Reduces axis length by 1.

    See Also
    --------
    deriv1, deriv1_uneven
    """
    if x.ndim>1: # if want x interpreted as vector
        xaxis = axis
    else:
        xaxis = 0
    if x.shape[xaxis] != y.shape[axis]: # allow broadcasting rules to be used along other axes
        raise ValueError('x and y dimensions do not match along derivative axis.')
    # y = np.rollaxis(y, axis, y.ndim) # broadcasting rules will then help us out
    # x = np.rollaxis(x, xaxis, x.ndim)
    y = permute(y, axis)
    x = permute(x, xaxis)
    return unpermute((y[...,1:] - y[...,:-1])/(x[...,1:] - x[...,:-1]), axis)

