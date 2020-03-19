#!/usr/bin/env python3
"""
Tools for objective analysis-related tasks. Many of these are adapted
from examples and course notes provided by `Elizabeth Barnes \
<http://barnes.atmos.colostate.edu/COURSES/AT655_S15/lecture_slides.html>`__
and `Dennis Hartmann \
<https://atmos.washington.edu/~dennis/552_Notes_ftp.html>`__.

Note
----
The convention throughout this package (consistent with conventions
in atmospheric science) is to use *linear* wave properties, i.e. the
wavelength in [units] per :math:`2\\pi` radians, and wavenumber
:math:`2\\pi` radians per [units].
"""
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import scipy.linalg as linalg
import scipy.optimize as optimize
import warnings
from .utils import lead_flatten, lead_unflatten, trail_flatten, trail_unflatten


def roots(poly):
    """
    Finds real-valued root for polynomial with input coefficients.
    Format of input array p: p[0]*x^n + p[1]*x^n-1 + ... + p[n-1]*x + p[n]
    """
    # Just use numpy's roots function, and filter results.
    r = np.roots(poly)  # input polynomial; returns ndarray
    r = r[np.imag(r) == 0].astype(np.float32)  # take only real-valued ones
    return r


def gaussian(N=1000, mean=0, stdev=None, sigma=1):
    """
    Returns sample points on Gaussian curve.
    """
    sigma = stdev if stdev is not None else sigma
    norm = stats.norm(loc=mean, scale=sigma)
    x = np.linspace(
        norm.ppf(0.0001), norm.ppf(0.9999), N
    )  # get x through percentile range
    pdf = norm.pdf(x, loc=mean, scale=sigma)
    return x, pdf


def rednoise(a, ntime, samples=1, mean=0, stdev=1):
    """
    Creates artificial red noise time series, i.e. a weighted sum
    of random perturbations.

    Equation is: x(t) = a*x(t-dt) + b*eps(t)
    where a is the lag-1 autocorrelation and b is a scaling term.

    Parameters
    ---------
    a : float
        The autocorrelation.
    ntime : int
        Number of timesteps.
    samples : int or list of int, optional
        Draw ``np.prod(samples)``. Final result will have
        shape `ntime` x `samples`.
    mean, stdev : float, optional
        The mean and standard deviation for the red noise
        time series.

    Note
    ----
    The output will have shape `ntime` by `nsamples`.
    This function enforces that the first timestep always equals the
    'starting' position.
    """
    # Initial stuff
    ntime -= 1  # exclude the initial timestep
    samples = np.atleast_1d(samples)
    data = np.empty((ntime + 1, *samples))  # user can make N-D array
    b = (1 - a ** 2) ** 0.5  # from OA class

    # Nested loop
    # data[1:,i] = a*data[:-1,i] + b*eps[:-1] # won't work because next state
    # function of previous state
    data, shape = trail_flatten(data)
    data[0, :] = 0  # initialize
    for i in range(data.shape[-1]):
        eps = np.random.normal(loc=0, scale=1, size=ntime)
        for t in range(1, ntime + 1):
            data[t, i] = a * data[t - 1, i] + b * eps[t - 1]

    # Return
    data = trail_unflatten(data, shape)
    if len(samples) == 1 and samples[0] == 1:
        data = data.squeeze()
    return mean + stdev * data  # rescale to have specified stdeviation/mean


def wilks(percentiles, alpha=0.10):
    """
    Return the Wilks et al. threshold given an array of percentiles.
    """
    percentiles = np.asarray(percentiles)
    pvals = list(percentiles.flat)  # want in decimal
    pvals = sorted(2 * min(pval, 1 - pval) for pval in pvals)
    ptest = [alpha * i / len(pvals) for i in range(len(pvals))]
    ppick = max(pv for pv, pt in zip(pvals, ptest) if pv <= pt) / 2
    mask = (percentiles <= ppick) | (percentiles >= (1 - ppick))
    return percentiles[mask]


def waves(x, wavenums=None, wavelens=None, phase=None):
    """
    Compose array of sine waves.
    Useful for testing performance of filters.

    Parameters
    ----------
    x : ndarray
        If scalar, *x* is ``np.arange(0,x)``. If iterable, can be
        n-dimensional, and will calculate sine from coordinates on
        every dimension.
    wavelens : float
        Wavelengths for sine function. Required if `wavenums` is ``None``.
    wavenums : float
        Wavenumbers for sine function. Required if `wavelens` is ``None``.
    phase : float, optional
        Array of phase offsets.

    Returns
    -------
    data : ndarray
        Data composed of sine waves.

    Note
    ----
    `x` will always be normalized so that wavelength is with reference to
    the first step. This make sense because when working with filters, for
    which we almost always need to use units corresponding to the axis.
    """
    # Wavelengths
    if wavenums is None and wavelens is None:
        raise ValueError('Must declare wavenums or wavelengths.')
    elif wavelens is not None:
        # dx = x[1] - x[0]
        wavenums = 1.0 / np.atleast_1d(wavelens)
    wavenums = np.atleast_1d(wavenums)
    if not hasattr(x, '__iter__'):
        x = np.arange(x)
    data = np.zeros(x.shape)  # user can make N-D array

    # Get waves
    if phase is None:
        phis = np.random.uniform(0, 2 * np.pi, len(wavenums))
    else:
        phis = phase * np.ones([len(wavenums)])
    for wavenum, phi in zip(wavenums, phis):
        data += np.sin(2 * np.pi * wavenum * x + phi)
    return data


def linefit(*args, axis=-1, build=False, stderr=False):
    """
    Get linear regression along axis, ignoring NaNs. Uses `~numpy.polyfit`.

    Parameters
    ----------
    x : ndarray, optional
        The *x* coordinates. Default is ``np.arange(0, len(y))``.
    y : ndarray
        The *y* coordinates.
    axis : int, optional
        regression axis
    build : bool, optional
        Whether to replace regression axis with scalar slope, or
        reconstructed best-fit line.
    stderr : bool, optional
        If not doing 'build', whether to add standard error on
        the slope in index 1 of regressoin axis.

    Returns
    -------
    y : ndarray
        Regression params on axis `axis`. The intercept is on index ``0``,
        slope on index ``1``.
    """
    # Perform regression
    # Polyfit can perform regressions on data with series in columns,
    # separate samples along rows.
    if len(args) == 1:
        (y,) = args
        x = np.arange(len(y))
    elif len(args) == 2:
        x, y = args
    else:
        raise ValueError("Must input 'x' or 'x, y'.")
    y, shape = lead_flatten(np.moveaxis(y, axis, -1))
    z, v = np.polyfit(x, y.T, deg=1, cov=True)
    z = np.fliplr(z.T)  # put a first, b next
    # Prepare output
    if build:
        # Repalce regression dimension with best-fit line
        z = z[:, :1] + x * z[:, 1:]
    elif stderr:
        # Replace the regression dimension with (slope, standard error)
        n = y.shape[1]
        s = np.array(z.shape[:1])
        resid = y - (z[:, :1] + x * z[:, 1:])  # residual
        mean = resid.mean(axis=1)
        var = resid.var(axis=1)
        rho = np.sum(
            (resid[:, 1:] - mean[:, None]) * (resid[:, :-1] - mean[:, None]),
            axis=1,
        ) / ((n - 1) * var)
        scale = (n - 2) / (n * ((1 - rho) / (1 + rho)) - 2)
        s = np.sqrt(v[0, 0, :] * scale)
        z[:, 0] = s  # the standard error
        z = np.fliplr(z)  # the array is <slope, standard error>
        shape[-1] = 2
    else:
        # Replace regression dimension with singleton (slope)
        z = z[:, 1:]
        shape[-1] = 1  # axis now occupied by the slope
    return np.moveaxis(lead_unflatten(z, shape), -1, axis)


def rednoisefit(
    data,
    dt=1,
    corr=False,
    nlag=None,
    axis=-1,
    lag1=False,
    series=False,
    verbose=False,
):
    """
    Calculates a best-fit red noise autocorrelation spectrum.
    Goes up to nlag-timestep autocorrelation.

    Parameters
    ----------
    data : ndarray
        The input data.
    dt : float, optional
        The timestep for your time series, used to scale the final timescale.
    corr : bool, optional
        Whether the input data indicates correlation at a series of lags.
    nlag : int, optional
        Number of lags to use for least-squares fit to red noise spectrum.
        Ignored if `lag1` is ``True``.
    lag1 : bool, optional
        If ``True``, estimate red noise spectrum from the lag-1
        autocorrelation.
    series : bool, optional
        Return the *e*-folding timescale associated with the red noise
        spectrum (``False``, default), or the red noise spectrum itself
        (``True``)?

    Returns
    -------
    taus : ndarray
        If `series` is ``False``, this is an array of timescales. Its shape
        is the same as `data`, with the time dimension replaced with a
        singleton dimesion.

        If `series` is ``True``, this is a reconstructed red noise
        autocorrelation spectrum. Its shape is the same as `data`.
        Present when `series` is ``False``. The *e*-folding timescales.
    sigmas : ndarray
        The standard error for the curve fit timescale estimate. If `lag1`
        is ``True``, this is an array of zeros (no curve fitting was
        performed).
    """
    # Initial stuff
    data, shape = lead_flatten(np.moveaxis(data, axis, -1))
    shape = [*shape]
    # First get the autocorrelation
    if corr:  # already was passed!
        auto = data
    elif nlag is None:
        raise ValueError(
            f"Must declare \"nlag\" argument; number of points to use for fit."
        )
    else:
        _, auto = globals()['corr'](data, nlag=nlag, axis=-1, verbose=verbose)
    # Shape stuff
    ne = auto.shape[0]  # extras
    nt = auto.shape[1]
    shape1 = [*shape]  # copy; this is where sigma is stored
    shape1[-1] = 1
    if series:
        ndim = nt
    else:
        ndim = 1
        shape = shape1  # same size as sigma array
    # Iterate
    time = np.arange(0, nt)  # time series for curve fit
    taus = np.empty((ne, ndim))
    sigmas = np.zeros((ne, 1))
    for i in range(ne):  # iterate along first dimension
        # Negative inverse natural log of lag-1 autocorrelation
        if lag1:
            p = -dt / np.log(auto[i, 1])
            s = 0  # no sigma, because no estimate

        # Standard error from covariance matrix; this is general solution,
        # but in this case have 1D array
        else:
            p, s = optimize.curve_fit(
                lambda t, tau: np.exp(-t * dt / tau), time, auto[i, :]
            )
            s = np.sqrt(np.diag(s))
            p, s = p[0], s[0]  # take only first param

        # Return
        if series:
            taus[i, :] = np.exp(-dt * time / p)  # best-fit spectrum
        else:
            taus[i, 0] = p  # just store the timescale
        if not lag1:
            sigmas[i, 0] = s
    taus = np.moveaxis(lead_unflatten(taus, shape), -1, axis)
    sigmas = np.moveaxis(lead_unflatten(sigmas, shape1), -1, axis)
    return taus, sigmas


def covar(*args, **kwargs):
    """As in `corr`, but returns the covariance."""
    return corr(*args, **kwargs, _normalize=False)


def corr(
    data1,
    data2=None,
    dt=1,
    lag=None,
    nlag=None,
    verbose=False,
    axis=-1,
    _normalize=True,
):
    """
    Gets the correlation spectrum at successive lags. For autocorrelation,
    pass only a single ndarray.

    Parameters
    ----------
    data1 : ndarray
        Input data.
    data2 : ndarray, optional
        Second input data, if cross-correlation is desired. Must have same
        shape as `data1`.
    axis : int, optional
        Axis along which correlation is taken.
    lag : int, optional
        Return correlation at the single lag `lag`.
    nlag : int, optional
        Return lag correlation up to `nlag` timesteps.

    Returns
    -------
    lags : ndarray
        The lags.
    corrs : ndarray
        The correlation as a function of lag.

    Note
    ----
    Uses `this estimator \
    <https://en.wikipedia.org/wiki/Autocorrelation#Estimation>`__:

    .. code-block::

        ((n-k)*sigma^2)^-1 * sum_i^(n-k)[X_t - mu][X_t+k - mu]

    This includes lag-zero values by default. However if you just want a
    single lag it will throw out those other values.
    """
    # Preparation, and stdev/means
    data1 = np.array(data1)
    if data2 is None:
        autocorr = True
    else:
        autocorr = False
        data2 = np.array(data2)
        if data1.shape != data2.shape:
            raise ValueError(
                f'Data 1 shape {data1.shape} and Data 2 shape {data2.shape} '
                'do not match.'
            )
    naxis = data1.shape[axis]  # length

    # Checks
    if not (nlag is None) ^ (lag is None):  # a wild xor appears!
        raise ValueError(
            f"Must specify either of the 'lag' or 'nlag' keyword args."
        )
    if nlag is not None and nlag >= naxis / 2:
        raise ValueError(
            f"Lag {nlag} must be greater than axis length {naxis}."
        )
    if verbose:
        if nlag is None:
            print(
                f'Calculating lag-{lag} autocorrelation.'
            )
        else:
            print(
                f'Calculating autocorrelation spectrum up to '
                f'lag {nlag} for axis length {naxis}.'
            )
    # Means and permute
    std1 = std2 = 1  # use for covariance
    data1 = np.moveaxis(data1, axis, -1)
    mean1 = data1.mean(axis=-1, keepdims=True)  # keep dims for broadcasting
    if autocorr:
        data2, mean2 = data1, mean1
    else:
        data2 = np.moveaxis(data2, axis, -1)

        mean2 = data2.mean(axis=-1, keepdims=True)

    # Standardize maybe
    if _normalize:
        # this is divided by the summation term, so should have annihilated
        # axis
        std1 = data1.std(axis=-1, keepdims=False)
        if autocorr:
            std2 = std1
        else:
            std2 = data2.std(axis=-1, keepdims=False)

    # This is just the variance, or one if autocorrelation mode is enabled
    # corrs = np.ones((*data1.shape[:-1], 1))
    if nlag is None and lag == 0:
        return np.moveaxis(
            np.sum((data1 - mean1) * (data2 - mean2)) / (naxis * std1 * std2),
            -1,
            axis,
        )

    # Correlation on specific lag
    elif nlag is None:
        lag = np.round(lag * dt).astype(int)
        return np.moveaxis(
            np.sum(
                (data1[..., :-lag] - mean1) * (data2[..., lag:] - mean2),
                axis=-1,
                keepdims=True,
            )
            / ((naxis - lag) * std1 * std2),
            -1,
            axis,
        )

    # Correlation up to n timestep-lags after 0-correlation
    else:
        # First figure out lags
        # Negative lag means data2 leads data1 (e.g. z corr m, left-hand side
        # is m leads z).
        # e.g. 20 day lag, at synoptic timesteps
        nlag = np.round(nlag / dt).astype(int)
        if not autocorr:
            n = nlag * 2 + 1  # the center point, and both sides
            lags = range(-nlag, nlag + 1)
        else:
            n = nlag + 1
            lags = range(0, nlag + 1)
        # Get correlation
        # will include the zero-lag autocorrelation
        corrs = np.empty((*data1.shape[:-1], n))
        for i, lag in enumerate(lags):
            if lag == 0:
                prod = (data1 - mean1) * (data2 - mean2)
            elif lag < 0:  # input 1 *trails* input 2
                prod = (data1[..., -lag:] - mean1) * (data2[..., :lag] - mean2)
            else:
                prod = (data1[..., :-lag] - mean1) * (data2[..., lag:] - mean2)
            corrs[..., i] = prod.sum(axis=-1) / ((naxis - lag) * std1 * std2)
        return np.array(lags) * dt, np.moveaxis(corrs, -1, axis)


def eof(
    data,
    neof=5,
    record=-2,
    space=-1,
    weights=None,
    percent=True,
    normalize=False,
):
    """
    Calculates the temporal EOFs, using the scipy algorithm for Hermetian (or
    real symmetric) matrices. This version allows calculating just 'n'
    most important ones, so should be faster.

    Parameters
    ----------
    data : ndarray
        Data of arbitrary shape.
    neof : int, optional
        Number of eigenvalues we want.
    record : int, optional
        Axis used as 'record' dimension.
    space : int or list of int, optional
        Axis or axes used as 'space' dimension.
    weights : ndarray, optional
        Area/mass weights; must be broadcastable on multiplication with 'data'
        weights will be normalized prior to application.
    percent : bool, optional
        Whether to return raw eigenvalue(s), or *percent* of variance
        explained by eigenvalue(s).
    normalize : bool, optional
        Whether to normalize the data by its standard deviation
        at every point prior to obtaining the EOFs.
    """
    # First query array shapes and stuff
    m_dims = np.atleast_1d(record)
    n_dims = np.atleast_1d(space)
    if m_dims.size > 1:
        raise ValueError('Record dimension must lie on only one axis.')
    m_dims[m_dims < 0] = data.ndim + m_dims[m_dims < 0]
    n_dims[n_dims < 0] = data.ndim + n_dims[n_dims < 0]
    if any(i < 0 or i >= data.ndim for i in [*m_dims, *n_dims]):
        raise ValueError('Invalid dimensions.')
    space_after = all(i > m_dims[0] for i in n_dims)
    if not space_after and not all(i < m_dims[0] for i in n_dims):
        raise ValueError(
            'Please reorder your data. '
            'All space dimensions must come before/after time dimension.'
        )

    # Remove the mean and optionally standardize the data
    data = data - data.mean(axis=m_dims[0], keepdims=True)  # remove mean
    if normalize:
        data = data / data.stdev(axis=m_dims[0], keepdims=True)

    # Next apply weights
    m = np.prod([data.shape[i] for i in n_dims])  # number samples/timesteps
    n = np.prod([data.shape[i] for i in m_dims])  # number space locations
    if weights is None:
        weights = 1
    weights = np.atleast_1d(weights)  # want numpy array
    weights = weights / weights.mean()  # so does not affect amplitude
    try:
        if m > n:  # more sampling than space dimensions
            data = data * np.sqrt(weights)
            dataw = data
        else:  # more space than sampling dimensions
            dataw = data * weights
    except ValueError:
        raise ValueError(
            f'Dimensionality of weights {weights.shape} incompatible with '
            f'dimensionality of space dimensions {data.shape}!'
        )

    # Turn matrix into *record* by *space*, or 'M' by 'N'
    # 1. Move record dimension to right
    data = np.moveaxis(data, m_dims[0], -1)
    dataw = np.moveaxis(dataw, m_dims[0], -1)
    # 2. Successively move space dimensions to far right, proceeding from the
    # rightmost space dimension to leftmost space dimension so axis numbers
    # do not change
    dims = n_dims.copy()
    if space_after:  # time was before, so new space dims changed
        dims -= 1
    dims = np.sort(dims)[::-1]
    for axis in dims:
        data = np.moveaxis(data, axis, -1)
        dataw = np.moveaxis(dataw, axis, -1)

    # Only flatten after apply weights (e.g. if have level and latitude
    # dimensions).
    shape_trail = data.shape[-n_dims.size:]
    data, _ = trail_flatten(data, n_dims.size)
    dataw, _ = trail_flatten(dataw, n_dims.size)
    shape_lead = data.shape[:-2]
    data, _ = lead_flatten(data, data.ndim - 2)
    dataw, _ = lead_flatten(dataw, dataw.ndim - 2)

    # Prepare output
    # Dimensions will be extraneous by eofs by time by space
    # m = record, n = time
    if data.ndim != 3:
        raise ValueError(f"Shit's on fire yo.")
    nextra, m, n = data.shape[0], data.shape[1], data.shape[2]
    pcs = np.empty((nextra, neof, m, 1))
    projs = np.empty((nextra, neof, 1, n))
    evals = np.empty((nextra, neof, 1, 1))
    nstar = np.empty((nextra, 1, 1, 1))

    # Get EOFs and PCs and stuff
    for i in range(data.shape[0]):
        # Get matrices
        x = data[i, :, :]  # array will be sampling by space
        xw = dataw[i, :, :]

        # Get reduced degrees of freedom for spatial eigenvalues
        # TODO: Fix the weight projection below
        rho = np.corrcoef(x.T[:, 1:], x.T[:, :-1])[0, 1]  # space x time
        rho_ave = (rho * weights).sum() / weights.sum()
        nstar[i, 0, 0, 0] = m * ((1 - rho_ave) / (1 + rho_ave))  # estimate

        # Get EOFs using covariance matrix on *shortest* dimension
        if x.shape[0] > x.shape[1]:
            # Get *temporal* covariance matrix since time dimension larger
            eigrange = [n - neof, n - 1]  # eigenvalues to get
            covar = (xw.T @ xw) / m
            # returns eigenvectors in *columns*
            l, v = linalg.eigh(covar, eigvals=eigrange, eigvals_only=False)
            Z = xw @ v  # (time by space) x (space x neof)
            z = (Z - Z.mean(axis=0)) / Z.std(axis=0)  # standardize pcs
            p = x.T @ z / m  # (space by time) x (time by neof)

        # Get *spatial* dispersion matrix since space dimension longer
        # This time 'eigenvectors' are actually the pcs
        else:
            eigrange = [m - neof, m - 1]  # eigenvalues to get
            covar = (xw @ x.T) / n
            l, Z = linalg.eigh(covar, eigvals=eigrange, eigvals_only=False)
            z = (Z - Z.mean(axis=0)) / Z.std(axis=0)  # standardize pcs
            p = (x.T @ z) / m  # (space by time) x (time by neof)

        # Store in big arrays
        pcs[i, :, :, 0] = z.T[::-1, :]  # neof by time
        projs[i, :, 0, :] = p.T[::-1, :]  # neof by space
        if percent:
            evals[i, :, 0, 0] = (
                100.0 * l[::-1] / np.trace(covar)
            )  # percent explained
        else:
            evals[i, :, 0, 0] = l[::-1]  # neof

        # # Sort
        # idx = L.argsort()[::-1]
        # L, Z = L[idx], Z[:,idx]

    # Return along the correct dimension
    # The 'lead's were *extraneous* dimensions; we got EOFs along them
    nlead = len(shape_lead)  # expand back to original; leave space for EOFs
    pcs = lead_unflatten(pcs, [*shape_lead, neof, m, 1], nlead)
    projs = lead_unflatten(projs, [*shape_lead, neof, 1, n], nlead)
    evals = lead_unflatten(evals, [*shape_lead, neof, 1, 1], nlead)
    nstar = lead_unflatten(nstar, [*shape_lead, 1, 1, 1], nlead)

    # The 'trail's were *spatial* dimensions, which were allowed to be more
    # than 1D
    ntrail = len(shape_trail)
    flat_trail = [1] * len(shape_trail)
    pcs = trail_unflatten(pcs, [*shape_lead, neof, m, *flat_trail], ntrail)
    projs = trail_unflatten(
        projs, [*shape_lead, neof, 1, *shape_trail], ntrail
    )
    evals = trail_unflatten(evals, [*shape_lead, neof, 1, *flat_trail], ntrail)
    nstar = trail_unflatten(nstar, [*shape_lead, 1, 1, *flat_trail], ntrail)

    # Permute 'eof' dimension onto the start (note we had to put it between
    # extraneous dimensions and time/space dimensions so we could perform
    # the above unflatten moves)
    init = len(shape_lead)  # eofs go after leading dimensions
    pcs = np.moveaxis(pcs, init, 0)
    projs = np.moveaxis(projs, init, 0)
    evals = np.moveaxis(evals, init, 0)
    nstar = np.moveaxis(nstar, init, 0)

    # Finally, permute stuff on right-hand dimensions back to original
    # positions. This time proceed from left-to-right so axis numbers onto
    # which we permute are correct.
    dims = n_dims.copy()
    dims += 1  # account for EOF
    if space_after:
        # The dims are *actually* one slot to left, since time was
        # not put back yet.
        dims -= 1
    dims = np.sort(dims)
    for axis in dims:
        pcs = np.moveaxis(pcs, -1, axis)
        projs = np.moveaxis(projs, -1, axis)
        evals = np.moveaxis(evals, -1, axis)
        nstar = np.moveaxis(nstar, -1, axis)
    pcs = np.moveaxis(pcs, -1, m_dims[0] + 1)
    projs = np.moveaxis(projs, -1, m_dims[0] + 1)
    evals = np.moveaxis(evals, -1, m_dims[0] + 1)
    nstar = np.moveaxis(nstar, -1, m_dims[0] + 1)

    return evals, nstar, projs, pcs


def eot(data, neof=5):  # noqa
    """
    EOTs, whatever they are.

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError


def reof(data, neof=5):  # noqa
    """
    Rotated EOFs, e.g. according to the "varimax" method. The EOFs will
    be rotated according only to the first `neof` EOFs.

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError


def rolling(*args, **kwargs):
    """Alias for `running`."""
    return running(*args, **kwargs)


def running(x, w, axis=-1, pad=True, pad_value=np.nan):
    """
    Apply running filter along axis.

    Parameters
    ----------
    x : ndarray
        Data, and we roll along axis `axis`.
    w : int or ndarray
        Boxcar window length, or custom weights.
    axis : int, optional
        Axis to filter.
    pad : bool, optional
        Whether to pad the edges of axis back to original size.
    pad_value : float, optional
        Pad value.

    Returns
    -------
    x : ndarray
        Data windowed along axis `axis`.

    Other parameters
    ----------------
    **kwargs
        Remaining keyword args passed to windowing function.

    Todo
    ----
    Merge this with `running`?

    Note
    ----
    Implementation is similar to `scipy.signal.lfilter`. Read
    `this post <https://stackoverflow.com/a/4947453/4970632>`_.

    Generates rolling numpy window along final axis. Can then operate with
    functions like polyfit or mean along the new last axis of output.
    Note this creates *view* of original array, without duplicating data, so
    no worries about efficiency.

    * For 1-D data numpy `convolve` would be appropriate, problem is `convolve`
      doesn't take multidimensional input!
    * If `x` has odd number of obs along axis, result will have last element
      trimmed. Just like `filter`.
    * Strides are apparently the 'number of bytes' one has to skip in memory
      to move to next position *on the given axis*. For example, a 5 by 5
      array of 64bit (8byte) values will have array.strides == (40,8).
    """
    # Roll axis, reshape, and get generate running dimension
    n_orig = x.shape[axis]
    if axis < 0:
        axis = (
            x.ndim + axis
        )  # e.g. if 3 dims, and want to axis dim -1, this is dim number 2
    x = np.moveaxis(x, axis, -1)

    # Determine weights
    if isinstance(w, str):
        raise NotImplementedError(
            "Need to allow string 'w' argument, e.g. w='Lanczos'"
        )
    w = np.atleast_1d(w)
    if len(w) == 1:
        # Boxcar window
        nw = w[0]
        w = 1 / nw
    else:
        # Arbitrary windowing function
        # TODO: Add windowing functions!
        nw = len(w)

    # Manipulate array
    shape = x.shape[:-1] + (x.shape[-1] - (nw - 1), nw)
    strides = [*x.strides, x.strides[-1]]  # repeat striding on end
    x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Next 'put back' axis, keep axis as
    # the 'rolling' dimension which can be averaged by arbitrary weights.
    x = np.moveaxis(x, -2, axis)  # want to 'put back' axis -2;

    # Finally take the weighted average
    # Note numpy will broadcast from right, so weights can be scalar or not
    # print(x.min(), x.max(), x.mean())
    x = (x * w).sum(axis=-1)  # broadcasts to right

    # Optionally fill the rows taken up
    if not pad:
        return x
    n_new = x.shape[axis]
    n_left = (n_orig - n_new) // 2
    n_right = n_orig - n_new - n_left
    if n_left != n_right:
        warnings.warn('Data shifted left by one.')
    d_left = pad_value * np.ones(
        (*x.shape[:axis], n_left, *x.shape[axis + 1:])
    )
    d_right = pad_value * np.ones(
        (*x.shape[:axis], n_right, *x.shape[axis + 1:])
    )
    x = np.concatenate((d_left, x, d_right), axis=axis)
    return x


def filter(x, b, a=1, n=1, axis=-1, fix=True, pad=True, pad_value=np.nan):
    """
    Apply scipy.signal.lfilter to data. By default this does *not* pad
    ends of data. May keep it this way.

    Parameters
    ----------
    x : ndarray
        Data to be filtered.
    b : ndarray
        *b* coefficients (non-recursive component).
    a : ndarray, optional
        *a* coefficients (recursive component); default of ``1`` indicates
        a non-recursive filter.
    n : int, optional
        Number of times to filter data; will go forward --> backward
        --> forward...
    axis : int, optional
        Axis along which we filter data. Defaults to last axis.
    fix : bool, optional
        Whether to trim leading part of axis by number of *b* coefficients.
        Will also attempt to *re-center* the data if a net-forward
        (e.g. f, fbf, fbfbf, ...) filtering was performed. This works
        for non-recursive filters only.
    pad : bool, optional
        Whether to pad trimmed values (when `fix` is ``True``) with
        `pad_value`.
    pad_value : float
        The pad value.

    Returns
    -------
    y : ndarray
        Data filtered along axis `axis`.

    Notes
    -----
    * Consider adding empirical method for trimming either side of recursive
      filter that trims up to where impulse response is negligible.
    * If `x` has odd number of obs along axis, lfilter will trim
      the last one. Just like `rolling`.
    * The *a* vector contains (index 0) the scalar used to normalize *b*
      coefficients, and (index 1,2,...) the coefficients on `y` in the
      filtering conversion from `x` to `y`. So, 1 implies
      array of [1, 0, 0...], implies non-recursive.
    """
    # Apply filter 'n' times to each sample
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    n_half = (max(len(a), len(b)) - 1) // 2
    if axis < 0:
        axis = x.ndim + axis  # necessary for concatenate below
    x, shape = lead_flatten(np.moveaxis(x, axis, -1))
    y = x.copy()  # then can filter multiple times
    ym = y.mean(axis=1, keepdims=True)
    y = y - ym  # remove mean
    for i in range(n):  # applications
        step = 1 if i % 2 == 0 else -1  # forward-backward application
        y[:, ::step] = signal.lfilter(b, a, y[:, ::step], axis=-1)
    y = y + ym  # add mean back in

    # Fancy manipulation
    if fix:
        # Capture component that (for non-recursive filter) doesn't include
        # datapoints with clipped edges. Forward-backward runs, so filtered
        # data is in correct position w.r.t. x e.g. if n is 2, we cut off
        # the (len(b)-1) from each side.
        n_2sides = (n // 2) * 2 * n_half

        # Net forward run, so filtered data is shifted right by n_half
        # Also have to trim data on both sides if it's
        # foward-->backward-->forward e.g.
        n_left = int((n % 2) == 1) * 2 * n_half

        # Determine part that 'sees' all coefficients
        if n_2sides == 0:
            y = y[:, n_left:]
        else:
            y = y[:, (n_2sides + n_left):(-n_2sides)]

        # Optionally pad with a 'fill value' (usually NaN)
        if pad:
            y_left = pad_value * np.ones((y.shape[0], n_2sides + n_left // 2))
            y_right = pad_value * np.ones((y.shape[0], n_2sides + n_left // 2))
            y = np.concatenate((y_left, y, y_right), axis=-1)

    # Return
    y = np.moveaxis(lead_unflatten(y, shape), -1, axis)
    return y


def response(dx, b, a=1, n=1000, simple=False):
    """
    Calculate response function given the a and b coefficients for some
    analog filter. Note we *need to make the exponent frequencies into
    rad/physical units* for results to make sense.

    Dennis Notes: https://atmos.washington.edu/~dennis/552_Notes_ftp.html

    Note
    ----
    The response function formula is depicted as follows:

    .. code-block::

                jw               -jw            -jmw
        jw  B(e)    b[0] + b[1]e + .... + b[m]e
        H(e) = ---- = ------------------------------------
                jw               -jw            -jnw
            A(e)    a[0] + a[1]e + .... + a[n]e

    and below we calculate simultaneously for vector of input omegas.
    """
    # Parse input
    a = np.atleast_1d(a)
    x = np.linspace(0, np.pi, n)

    # Simple calculation given 'b' coefficients, from Libby's notes
    if simple:
        if not len(a) == 1 and a[0] == 1:
            raise ValueError(
                'Cannot manually calculate response function '
                'for recursive filter.'
            )
        if len(b) % 2 == 0:
            raise ValueError(
                'Filter coefficient number should be odd, symmetric '
                'about a central value.'
            )
        nb = len(b)
        C0 = b[nb // 2]
        Ck = b[nb // 2 + 1:]  # should be symmetric
        tau = np.arange(1, nb // 2 + 1)  # lag time, up to nb+1
        x = x * 2 * np.pi * dx  # from cycles/unit --> rad/unit --> rad/step
        y = C0 + 2 * np.sum(
            Ck[None, :] * np.cos(tau[None, :] * x[:, None]), axis=1
        )

    # More complex freqz filter, generalized for arbitrary recursive filters,
    # with extra allowance for working in physical (non-timestep) space
    # Last entry will be Nyquist, i.e. 1/(dx*2)
    else:
        _, y = signal.freqz(b, a, x)
        x = x / (2 * np.pi * dx)
        y = np.abs(y)
    return x, y


def impulse():
    """
    Displays the *impulse* response function for a recursive filter.
    """
    # R2_q = 1./(1. + (omega/omega_c)**(2*N))
    raise NotImplementedError


def harmonics(x, k=4, axis=-1, absval=False):
    """
    Select the first `k` Fourier harmonics of the time series. Useful
    for example in removing seasonal cycle or something.
    """
    # Get fourier transform
    x = np.moveaxis(x, axis, -1)
    fft = np.fft.fft(x, axis=-1)

    # Remove frequencies outside range. The FFT will have some error and give
    # non-zero imaginary components, but we can get magnitude or naively cast
    # to real
    fft[..., 0] = 0
    fft[..., k + 1:-k] = 0
    if absval:
        y = np.moveaxis(np.abs(np.fft.ifft(fft)), -1, axis)
    else:
        y = np.moveaxis(np.real(np.fft.ifft(fft)), -1, axis)
    return y


def highpower(x, n, axis=-1):
    """
    Select only the highest power frequencies. Useful for
    crudely reducing noise.

    Parameters
    ----------
    x : `numpy.ndarray`
        The data.
    n : int
        The integer number of frequencies to select.
    """
    # Naively remove certain frequencies
    # Should ignore first coefficient, the mean
    x = np.moveaxis(x, axis, -1)
    fft = np.fft.fft(x, axis=-1)
    fftfreqs = np.arange(1, fft.shape[-1] // 2)  # up to fft.size/2 - 1

    # Get indices of n largest values. Use *argpartition* because it's more
    # efficient, will just put -nth element into sorted position, everything
    # after that unsorted but larger (don't need exact order!).
    p = np.abs(fft) ** 2
    f = np.argpartition(p, -n, axis=-1)[..., -n:]
    y = fft.copy()
    y[...] = 0  # fill in
    y[..., f] = fft[..., f]  # put back the high-power frequencies
    freqs = fftfreqs[..., f]
    return freqs, y  # frequencies and the high-power filter


def lanczos(dx, width, cutoff):
    """
    Returns *coefficients* for Lanczos high-pass filter with
    desired wavelength specified. See `this link \
<https://scitools.org.uk/iris/docs/v1.2/examples/graphics/SOI_filtering.html>`__.

    Parameters
    ----------
    dx : float
        Units of your x-dimension, so that cutoff can be translated
        from physical units to 'timestep' units.
    width : float
        Length of filter in time steps.
    cutoff : float
        Cutoff wavelength in physical units.

    Returns
    -------
    b : ndarray
        Numerator coeffs.
    a : ndarray
        Denominator coeffs.

    Notes
    -----
    * The smoothing should only be *approximate* (see Hartmann notes), response
      function never exactly perfect like with Butterworth filter.
    * The `cutoff` parameter must be provided in *time step units*. Change
      the converter `dx` otherwise.
    * The '2' factor appearing in multiple places may seem random. But this
      converts linear frequency (i.e. wavenumber) to angular frequency in
      sine call below. The '2' doesn't appear in any other factor just as a
      consequence of the math.

    Example
    -------
    n=9 returns 4+4+1=9 points in the 'concatenate' below.
    """
    # Coefficients and initial stuff
    # n = (width/dx)//1  # convert window width from 'time units' to 'steps'
    # n = width//2
    # convert alpha to wavenumber (new units are 'inverse timesteps')
    alpha = 1.0 / (cutoff / dx)
    n = width
    n = (n - 1) // 2 + 1
    print(f'Order-{n*2 - 1:.0f} Lanczos window')
    tau = np.arange(1, n + 1)  # lag time
    C0 = 2 * alpha  # integral of cutoff-response function is alpha*pi/pi
    Ck = np.sin(2 * np.pi * alpha * tau) / (np.pi * tau)
    Cktilde = Ck * np.sin(np.pi * tau / n) / (np.pi * tau / n)

    # Return filter
    window = np.concatenate((np.flipud(Cktilde), np.array([C0]), Cktilde))
    return window[1:-1], 1


def butterworth(dx, order, cutoff, btype='low'):
    """
    Applies Butterworth filter to data. Since this is a *recursive*
    filter, non-trivial to apply, so this uses scipy 'lfilter'.

    To get an 'impulse response function', pass a bunch of zeros with a single
    non-zero 'point' as the dx. See Libby's function for more details.

    Parameters
    ----------
    dx : float
        Data spacing.
    order : int
        Order of the filter.
    cutoff : float
        Cutoff frequency in 'x' units (i.e. *wavelengths*).

    Returns
    -------
    b : ndarray
        Numerator coeffs.
    a : ndarray
        Denominator coeffs.

    Note
    ----
    * Need to run *forward and backward* to prevent time-shifting.
    * The 'analog' means units of cutoffs are rad/s.
    * Unlike Lanczos filter, the *length* of this should be
      determined always as function of timestep, because really high
      order filters can get pretty wonky.
    * Cutoff is point at which gain reduces to 1/sqrt(2) of the
      initial frequency. If doing bandpass, can
    """
    # Initial stuff
    # N = (width/dx)//1 # convert to timestep units
    # N = (N//2)*2 + 1 # odd numbered
    N = order  # or order
    analog = False  # lfilter seems to need digital
    if analog:
        cutoff = 2 * np.pi / (cutoff / dx)  # from wavelengths to rad/steps
    else:
        cutoff = 1.0 / cutoff  # to Hz, or cycles/unit
        cutoff = cutoff * (2 * dx)  # to cycles / (2 timesteps)
    if cutoff > 1:
        raise ValueError(
            'Cuttoff frequency must be in [0, 1]. Remember you pass a cutoff '
            '*wavelength* to this function, not a frequency.'
        )

    # Apply filter
    print(f'Order-{N:.0f} Butterworth filter')
    b, a = signal.butter(
        N - 1, cutoff, btype=btype, analog=analog, output='ba'
    )
    return b, a


def window(wintype, n):
    """
    Retrieves weighting function window, identical to get_window(). Note
    the raw window weights must be normalized, or will screw up the power
    of your FFT coefficients.  For windows that require extra parameters,
    window must be a tuple of window-name, parameter.

    Parameters
    ----------
    wintype : str or (str, float) tuple
        The window name.
    n : int
        The window length.

    Returns
    -------
    win : ndarray
        The window coefficients.
    """
    # Default error messages are shit, make them better
    if wintype == 'welch':
        raise ValueError('Welch window needs 2-tuple of (name,beta).')
    if wintype == 'kaiser':
        raise ValueError('Welch window needs 2-tuple of (name,beta).')
    if wintype == 'gaussian':
        raise ValueError('Gaussian window needs 2-tuple of (name,stdev).')

    # Get window
    win = signal.get_window(wintype, n)
    return win


def power(
    y1,
    y2=None,
    dx=1,
    cyclic=False,
    coherence=False,
    nperseg=100,
    wintype='boxcar',
    center=np.pi,
    axis=-1,
    detrend='constant',
):
    """
    Gets the spectral decomposition for particular windowing technique.
    Uses simple numpy fft.

    Parameters
    ----------
    y1 : ndarray
        Input data.
    y2 : ndarray, default ``None``
        Second input data, if cross-spectrum is desired.
        Must have same shape as `y1`.
    dx : float, optional
        Timestep in physical units, used for scaling the
        frequency-coordinates.
    cyclic : bool, optional
        Whether data is cyclic along axis. When ``True``, the *nperseg*
        argument will be overridden
    coherence : bool, optional
        Ignored when `z2` is ``None``. If ``False`` (the default), `power`
        returns the co-power spectrum, quadrature spectrum, and individual
        power spectra. If ``True``, `power` returns the coherence
        and phase difference.

    Returns
    -------
    f : ndarray
        Wavenumbers, in [x units]**-1.
    P : ndarray
        Present when `z2` is ``None``. Power spectrum, in
        units [data units]**2.
    P, Q, Pz1, Pz2 : ndarray
        Present when `z2` is not ``None``, `coherence` = ``False``. Co-power
        spectrum, quadrature spectrum, power spectrum for `z1`, and
        power spectrum for `z2`, respectively.
    Coh, Phi : ndarray
        Present when `z2` is not ``None``, `coherence` = ``True``.
        Coherence and phase difference, respectively.

    Note
    ----
    The scaling conventions, and definitions of coefficients, change between
    references and packages! Libby's notes defines variance equals one half
    the sum of right-hand square coefficients, but numpy package defines
    variance as sum of square of all coefficients (or twice the right-hand
    coefficients). Also complex DFT convention in general seems to require
    normalizing by 1/N after FFT.

    See:

      * https://stackoverflow.com/a/19976162/4970632
      * https://stackoverflow.com/a/15148195/4970632

    Note that windowing reduces the power amplitudes, and results in loss
    of information! With 'boxcar' window, summing the components gives you
    the exact variance. Another issue is the necessary amplitude 'correction'
    factor varies depending on frequency and the signal analysed.

    Better perhaps to follow example of Randel and Held and smooth in
    frequency space with a Gaussian filter (running average with window
    summing to 1; should not reduce power).

    Example
    -------
    Below proves that extent of power reduction due to windowing
    depends on the signal.

    .. ipython:: python

        import numpy as np
        import climpy
        w = climpy.window('hanning', 200)
        y1 = np.sin(np.arange(0,8*np.pi-0.01,np.pi/25)) # simple signal
        y2 = np.random.rand(200) # complex signal
        for y in (y1, y2):
            yvar = ((y-y.mean())**2).mean()
            Y = (np.abs(np.fft.fft(y)[1:]/y.size)**2).sum()
            Yw = (np.abs(np.fft.fft(y*w)[1:]/y.size)**2).sum()
            print('Boxcar', Y/yvar)
            print('Hanning', Yw/yvar)

    """
    # Initial stuff
    N = y1.shape[axis]  # window count
    if cyclic:
        wintype = 'boxcar'
        nperseg = N
    if y2 is not None and y2.shape != y1.shape:
        raise ValueError(
            f'Got conflicting shapes for y1 {y1.shape} and y2 {y2.shape}.'
        )
    nperseg = 2 * (nperseg // 2)  # enforce even window size
    # Trim if necessary
    r = N % nperseg
    if r != 0:
        s = [slice(None) for i in range(y1.ndim)]
        s[axis] = slice(None, -r)
        y1 = y1[tuple(s)]  # slice it up
        N = y1.shape[axis]  # update after trim
        if y2 is not None:
            y2 = y2[tuple(s)]
        warnings.warn(
            f'Trimmed {r} out of {N} points to '
            f'accommodate length-{nperseg} window.'
        )

    # Just use scipy csd
    # 'one-sided' says to only return first symmetric half if data is real
    # 'scaling' queries whether to:
    #  * scale 'per wavenumber'/'per Hz', option 'density', default;
    #    this is analagous to a Planck curve with intensity per wavenumber
    #  * show the power (so units are just data units squared); this is
    #    usually what we want
    # if not manual and y2 is None:
    # f, P = signal.csd(y1, y1, window=wintype,
    #   return_onesided=True, scaling=scaling,
    #   nperseg=nperseg, noverlap=nperseg//2, detrend=detrend,
    #   axis=axis
    # )

    # Manual approach, have checked these and results are identical
    # Get copsectrum, quadrature spectrum, and powers for each window
    # shape is shape of *original* data
    y1, shape = lead_flatten(np.moveaxis(y1, axis, -1))
    if y2 is not None:
        y2, _ = lead_flatten(np.moveaxis(y2, axis, -1))
    extra = y1.shape[0]
    pm = nperseg // 2
    shape[-1] = pm  # new shape

    # List of *center* indices for windows
    win = window(wintype, nperseg)
    loc = np.arange(pm, N - pm + pm // 2, pm)  # jump by half window length
    if len(loc) == 0:
        raise ValueError('Window length too big.')

    # Ouput arrays
    Py1 = np.empty((extra, loc.size, pm))  # power
    if y2 is not None:
        CO = Py1.copy()
        Q = Py1.copy()
        Py2 = Py1.copy()

    for j in range(extra):
        # Loop through windows
        if np.any(~np.isfinite(y1[j, :])) or (
            y2 is not None and np.any(~np.isfinite(y2[j, :]))
        ):
            warnings.warn('Skipping array with missing values.')
            continue
        for i, l in enumerate(loc):
            if y2 is None:
                # Remember to double the size of power, because only
                # have half the coefficients (rfft not fft)
                # print(win.size, pm, y1[j,l-pm:l+pm].size)
                wy = win * signal.detrend(
                    y1[j, l - pm:l + pm], type=detrend
                )
                Fy1 = np.fft.rfft(wy)[1:] / win.sum()
                Py1[j, i, :] = np.abs(Fy1) ** 2
                Py1[j, i, :-1] *= 2

            else:
                # Frequencies
                wy1 = win * signal.detrend(
                    y1[j, l - pm:l + pm], type=detrend
                )
                wy2 = win * signal.detrend(
                    y2[j, l - pm:l + pm], type=detrend
                )
                Fy1 = np.fft.rfft(wy1)[1:] / win.sum()
                Fy2 = np.fft.rfft(wy2)[1:] / win.sum()
                # Powers
                Py1[j, i, :] = np.abs(Fy1) ** 2
                Py2[j, i, :] = np.abs(Fy2) ** 2
                CO[j, i, :] = Fy1.real * Fy2.real + Fy1.imag * Fy2.imag
                Q[j, i, :] = Fy1.real * Fy2.imag - Fy2.real * Fy1.imag
                for array in (Py1, Py2, CO, Q):
                    array[j, i, :-1] *= 2  # scale all but Nyquist frequency

    # Helper function
    def unshape(x):
        x = lead_unflatten(x, shape)
        x = np.moveaxis(x, -1, axis)
        return x

    # Get window averages, reshape, and other stuff
    # NOTE: For 'real' transform, all values but Nyquist must be divided by
    # two, so that an 'average' of the power equals the covariance.
    f = np.fft.rfftfreq(nperseg)[1:]  # frequencies
    if y2 is None:
        # Average windows
        Py1 = Py1.mean(axis=1)
        Py1 = unshape(Py1)
        return f / dx, Py1

    else:
        # Averages
        CO = CO.mean(axis=1)
        Q = Q.mean(axis=1)
        Py1 = Py1.mean(axis=1)
        Py2 = Py2.mean(axis=1)
        if coherence:  # return coherence and phase instead
            # Coherence and stuff
            Coh = (CO ** 2 + Q ** 2) / (Py1 * Py2)
            Phi = np.arctan2(Q, CO)  # phase
            Phi[Phi >= center + np.pi] -= 2 * np.pi
            Phi[Phi < center - np.pi] += 2 * np.pi
            Phi = Phi * 180 / np.pi  # convert to degrees!!!
            Coh = unshape(Coh)
            Phi = unshape(Phi)
            return f / dx, Coh, Phi
        else:
            # Reshape and return
            CO = unshape(CO)
            Q = unshape(Q)
            Py1 = unshape(Py1)
            Py2 = unshape(Py2)
            return f / dx, CO, Q, Py1, Py2


def power2d(
    z1,
    z2=None,
    dx=1,
    dy=1,
    coherence=False,
    nperseg=100,
    wintype='boxcar',
    center=np.pi,
    axes=(-2, -1),
):
    """
    Performs 2-d spectral decomposition, with windowing along only *one*
    dimension, in style of Randel and Held 1991. Therefore assumption is we
    have *cyclic* data along one dimension, the 'x' dimension.

    Parameters
    ----------
    z1 : ndarray
        Input data.
    z2 : ndarray, default ``None``
        Second input data, if cross-spectrum is desired. Must have same
        shape as `z1`.
    dx : float, optional
        Timestep in physical units, used for scaling the
        frequency-coordinates.
    dy : float, optional
        Physical units of cyclic dimension step, used for scaling the
        frequency-coordinates.
    axes : (int, int), optional
        Locations of the "time" and "cyclic" axes, respectively.
        By default the second-to-last and last axes are used.
    coherence : bool, optional
        Ignored when `z2` is ``None``. If ``False`` (the default), `power2d`
        returns the co-power spectrum, quadrature spectrum, and individual
        power spectra. If ``True``, `power2d` returns the coherence
        and phase difference.

    Returns
    -------
    f : ndarray
        Wavenumbers, in [x units]**-1.
    P : ndarray
        Present when `z2` is ``None``. Power spectrum, in
        units [data units]**2.
    P, Q, Pz1, Pz2 : ndarray
        Present when `z2` is not ``None``, `coherence` = ``False``. Co-power
        spectrum, quadrature spectrum, power spectrum for `z1`, and
        power spectrum for `z2`, respectively.
    Coh, Phi : ndarray
        Present when `z2` is not ``None``, `coherence` = ``True``.
        Coherence and phase difference, respectively.

    Note
    ----
    See notes for `power`.
    """
    # Checks
    taxis, caxis = axes
    if len(z1.shape) < 2:
        raise ValueError('Need at least rank 2 array.')
    if z2 is not None and not z1.shape == z2.shape:
        raise ValueError(
            f'Shapes of z1 {z1.shape} and z2 {z2.shape} must match.'
        )
    print(
        f'Cyclic dimension ({caxis}): Length {z1.shape[caxis]}.'
    )
    print(
        f'Windowed dimension ({taxis}): '
        f'Length {z1.shape[taxis]}, window length {nperseg}.'
    )
    if caxis < 0:
        caxis = z1.ndim - caxis
    if taxis < 0:
        taxis = z1.ndim - taxis
    nperseg = 2 * (nperseg // 2)  # enforce even window size because I said so
    l = z1.shape[taxis]  # noqa: E741
    r = l % nperseg
    if r > 0:
        s = [slice(None) for i in range(z1.ndim)]
        s[taxis] = slice(None, -r)
        z1 = z1[s]  # slice it up
        warnings.warn(
            f'Trimmed {r} out of {l} points to '
            f'accommodate length-{nperseg} window.'
        )

    # Helper function
    # Will put the *non-cyclic* axis on position 1, *cyclic* axis on position 2
    # Mirrors convention for row-major geophysical data array storage, time
    # by pressure by lat by lon (the cyclic one).
    offset = int(taxis < caxis)  # TODO: does this work, or is stuff messed up?
    nflat = z1.ndim - 2  # we overwrite z1, so must save this value!

    def reshape(x):
        x = np.moveaxis(x, taxis, -1)  # put on -1, then will be moved to -2
        x = np.moveaxis(x, caxis - offset, -1)  # put on -1
        x, shape = lead_flatten(x, nflat)  # flatten remaining dimensions
        return x, shape

    # Permute
    z1, shape = reshape(z1)
    extra = z1.shape[0]
    if z2 is not None:
        z2, _ = reshape(z2)
    # For output data
    N = z1.shape[1]  # non-cyclic dimension
    M = z1.shape[2]  # cyclic dimension
    pm = nperseg // 2
    shape[-2] = 2 * pm  # just store the real component
    shape[-1] = M // 2

    # Helper function
    # Gets 2D Fourier decomp, reorders negative frequencies on non-cyclic
    # axis so frequencies there are monotonically ascending.
    # wsum = (win ** 2).sum()
    win = window(wintype, nperseg)  # for time domain

    def freqs(x, pm):
        # Detrend
        # Or don't, since we shave the constant part anyway?
        # x = signal.detrend(x, type='constant', axis=1) # remove mean
        # x = signal.detrend(x, type=detrend, axis=0) # remove trend or mean

        # The 2D approach
        # NOTE: Read documentation regarding normalization. Default leaves
        # forward transform unnormalized, reverse normalized by 1/n; the ortho
        # option normalizes both by 1/sqrt(n).
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/routines.fft.html#module-numpy.fft
        # last axis specified should get a *real* transform
        X = np.fft.rfft2(win[:, None] * x, axes=(0, 1))
        X = X[:, 1:]  # remove the zero-frequency value
        X = X / (x.shape[0] * x.shape[1])  # normalize by sample size
        out = np.concatenate((X[pm:, :], X[1:pm + 1, :]), axis=0)

        # Manual approach, virtually identical
        # Follows Libby's recipe, where instead real is cosine and imag is
        # sine. Note only need to divide by 2 when conjugates are included.
        # xi = np.fft.rfft(x, axis=1)[:,1:]/x.shape[1]
        # xi = win[:,None]*xi # got a bunch of sines and cosines
        # C = np.fft.rfft(xi.real, axis=0)[1:,:]/x.shape[0]
        # S = np.fft.rfft(xi.imag, axis=0)[1:,:]/x.shape[0]
        # out = np.concatenate((
        #     (C.real + S.imag + 1j*(C.imag - S.real))[::-1,:],
        #      C.real - S.imag + 1j*(-C.imag - S.real),
        # ), axis=0)
        return out

    # The window *centers* for time windowing
    # jump by half window length
    loc = np.arange(pm, N - pm + 0.1, pm).astype(int)
    if len(loc) == 0:
        raise ValueError('Window length too big.')
    # Get the spectra
    Pz1 = np.nan * np.empty((extra, loc.size, *shape[-2:]))  # power
    if z2 is not None:
        CO = Pz1.copy()
        Q = Pz1.copy()
        Pz2 = Pz1.copy()
    for j in range(extra):
        # Missing values handling
        if np.any(~np.isfinite(z1[j, :, :])) or (
            z2 is not None and np.any(~np.isfinite(z2[j, :, :]))
        ):
            warnings.warn('Skipping array with missing values.')
            continue
        # 2D transform for each window on non-cyclic dimension
        for i, l in enumerate(loc):
            if z2 is None:
                # Note since we got the rfft (not fft) in one direction, only
                # have half the coefficients (they are symmetric); means for
                # correct variance, have to double the power.
                Fz1 = freqs(z1[j, l - pm:l + pm, :], pm)
                Pz1[j, i, :, :] = np.abs(Fz1) ** 2
                Pz1[j, i, :, :-1] *= 2
            else:
                # Frequencies
                Fz1 = freqs(z1[j, l - pm:l + pm, :], pm)
                Fz2 = freqs(z2[j, l - pm:l + pm, :], pm)
                # Powers
                # analagous to Libby's notes, for complex space
                Phi1 = np.arctan2(Fz1.imag, Fz1.real)
                Phi2 = np.arctan2(Fz2.imag, Fz2.real)
                CO[j, i, :, :] = (
                    np.abs(Fz1) * np.abs(Fz2) * np.cos(Phi1 - Phi2)
                )
                Q[j, i, :, :] = np.abs(Fz1) * np.abs(Fz2) * np.sin(Phi1 - Phi2)
                Pz1[j, i, :, :] = np.abs(Fz1) ** 2
                Pz2[j, i, :, :] = np.abs(Fz2) ** 2
                for array in (CO, Q, Pz1, Pz2):
                    array[j, i, :, :-1] *= 2

    # Frequencies
    # Make sure Nyquist frequency is appropriately signed on either side of
    # array
    fx = np.fft.fftfreq(2 * pm)  # just the positive-direction Fourier coefs
    fx = np.concatenate(
        (
            -np.abs(fx[pm:pm + 1]),
            fx[pm + 1:],
            fx[1:pm],
            np.abs(fx[pm:pm + 1]),
        ),
        axis=0,
    )
    fy = np.fft.rfftfreq(M)[1:]

    # Helper function
    # Reshapes final result, and scales powers so we can take dimensional
    # average without needing to divide by 2
    def unshape(x):
        x = lead_unflatten(x, shape, nflat)
        # put caxis back, accounting for if taxis moves to left
        x = np.moveaxis(x, -1, caxis - offset)
        x = np.moveaxis(x, -1, taxis)  # put taxis back
        return x

    # Get window averages, reshape, and other stuff
    # NOTE: For the 'real' transform, all values but Nyquist must
    # be divided by two, so that an 'average' of the power equals
    # the covariance.
    if z2 is None:
        # Return
        Pz1 = Pz1.mean(axis=1)
        Pz1 = unshape(Pz1)
        return fx / dx, fy / dy, Pz1
    else:
        # Averages
        print(Pz1.shape)
        CO = CO.mean(axis=1)
        Q = Q.mean(axis=1)
        Pz1 = Pz1.mean(axis=1)
        Pz2 = Pz2.mean(axis=1)
        if coherence:  # return coherence and phase instead
            # Coherence and stuff
            # NOTE: This Phi relationship is still valid. Check Libby's
            # notes; divide here Q by CO and the Ws cancel out, end up
            # with average phase difference indeed.
            Coh = (CO ** 2 + Q ** 2) / (Pz1 * Pz2)
            Phi = np.arctan2(Q, CO)  # phase
            Phi[Phi >= center + np.pi] -= 2 * np.pi
            Phi[Phi < center - np.pi] += 2 * np.pi
            # Reshape and return
            Coh = unshape(Coh)
            Phi = unshape(Phi)
            return fx / dx, fy / dy, Coh, Phi
        else:
            # Reshape
            CO = unshape(CO)
            Q = unshape(Q)
            Pz1 = unshape(Pz1)
            Pz2 = unshape(Pz2)
            return fx / dx, fy / dy, CO, Q, Pz1, Pz2


def autopower():
    """
    Wrapper around `power1d`, that generates co-spectral
    statistics and whatnot at **successive lags**.

    Warning
    -------
    Not yet implemented.
    """
    # Uses scipy.signal.welch windowing method to generate an estimate of the
    # *lagged* spectrum. Can also optionally do this with two variables.
    raise NotImplementedError


def autopower2d():
    """
    Wrapper around `power2d`, that generates co-spectral
    statistics and whatnot at **successive lags**.

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError
