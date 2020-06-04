#!/usr/bin/env python3
"""
Analyses of variance and trends. Many of these are adapted
from examples and course notes provided by `Elizabeth Barnes \
<http://barnes.atmos.colostate.edu/COURSES/AT655_S15/lecture_slides.html>`__
and `Dennis Hartmann \
<https://atmos.washington.edu/~dennis/552_Notes_ftp.html>`__.
"""
import numpy as np
import scipy.stats as stats
import scipy.linalg as linalg
import scipy.optimize as optimize
from .internals import quack
from .internals import docstring
from .internals.array import _ArrayContext

__all__ = [
    'corr',
    'covar',
    'eof',
    'eot',
    'reof',
    'gaussian',
    'hist',
    'linefit',
    'rednoise',
    'rednoisefit',
    'wilks',
]

# Docstring snippets
_corr_math = r"""
.. math::

    \dfrac{\sum_{i=0}^{n-k}\left(x_t - \overline{x}\right)\left(y_{t+k} - \overline{y}\right)}{(n - k) s_x s_y}

where :math:`\overline{x}` and :math:`\overline{y}` are the sample means and
:math:`s_x` and :math:`s_y` are the sample standard deviations.
"""  # noqa: E501

_covar_math = r"""
.. math::

    \dfrac{\sum_{i=0}^{n-k}\left(x_t - \overline{x}\right)\left(y_{t+k} - \overline{y}\right)}{n - k}

where :math:`\overline{x}` and :math:`\overline{y}` are the sample means.
"""  # noqa: E501

_corr_covar_docs = """
Return the %(name)s or auto%(name)s spectrum at successive lags.

Parameters
----------
z1 : array-like
    Input data.
z2 : array-like, optional
    Second input data. Must be same shape as `z1`. Leave empty if
    auto%(name)s is desired.
axis : int, optional
    Axis along which %(name)s is taken.
lag : int, optional
    Return %(name)s at the single lag `lag`.
nlag : int, optional
    Return lagged %(name)s from ``0`` timesteps up to `nlag` timesteps.

Returns
-------
lags : array-like
    The lags.
result : array-like
    The %(name)s as a function of lag.

Note
----
This function uses the following formula to estimate %(name)s at lag :math:`k`:
%(math)s
"""

docstring.snippets['corr'] = _corr_covar_docs % {
    'name': 'correlation', 'math': _corr_math
}
docstring.snippets['covar'] = _corr_covar_docs % {
    'name': 'covariance', 'math': _covar_math
}

docstring.snippets['power.bibliography'] = """
.. bibliography:: ../bibs/power.bib
"""

docstring.snippets['power.params'] = """
coherence : bool, optional
    Ignored if `z2` is ``None``. If ``False`` (the default), `power`
    returns the co-power spectrum, quadrature spectrum, and individual
    power spectra. If ``True``, `power` returns the coherence
    and phase difference.
wintype : str or (str, float), optional
    The window specification, passed to `get_window`. The resulting
    weights are used to window the data before carrying out spectral
    decompositions. See notes for details.
nperseg : int, optional
    The window or segment length, passed to `get_window`. If ``None``,
    windowing is not carried out. See notes for details.
detrend : {'constant', 'linear'}, optional
    Passed as the `type` argument to `scipy.signal.detrend`. ``'constant'``
    removes the mean and ``'linear'`` removes the linear trend.
"""

docstring.snippets['power.notes'] = """
The Fourier coefficients are scaled so that total variance is equal to one
half the sum of the right-hand coefficients. This is more natural for the
real-valued datasets typically used by physical scientists, and matches
the convention from Elizabeth Barnes's objective analysis `course notes \
<http://barnes.atmos.colostate.edu/COURSES/AT655_S15/lecture_slides.html>`__.
This differs from the numpy convention, which scales the coefficients so
that total variance is equal to the sum of squares of all coefficients,
or twice the right-hand coefficients.

Windowing is carried out by applying the `wintype` weights to successive
time segments of length `nperseg` (overlapping by one half the window
length), taking spectral decompositions of each weighted segment, then
taking the average of the result for all segments. Note that non-boxcar
windowing reduces the total power amplitude and results in loss of
information. It may often be preferable to follow the example of
:cite:`1991:randel` and smooth in *frequency* space with a Gaussian filter
after the decomposition has been carried out.

The below example shows that the extent of power reduction resulting from
non-boxcar windowing depends on the character of the signal.

.. code-block:: python

    import numpy as np
    import climpy
    w = climpy.window('hanning', 200)
    y1 = np.sin(np.arange(0, 8 * np.pi - 0.01, np.pi / 25)) # basic signal
    y2 = np.random.rand(200) # complex signal
    for y in (y1, y2):
        yvar = y.var()
        Y = (np.abs(np.fft.fft(y)[1:] / y.size) ** 2).sum()
        Yw = (np.abs(np.fft.fft(y * w)[1:] / y.size) ** 2).sum()
        print('Boxcar', Y / yvar)
        print('Hanning', Yw / yvar)
"""


@docstring.add_snippets
def corr(*args, **kwargs):
    """%(corr)s"""
    return corr(*args, **kwargs, corr=True)


@docstring.add_snippets
def covar(
    z1, z2=None, dt=1, lag=None, nlag=None,
    axis=-1, corr=False, verbose=False,
):
    """%(covar)s"""
    # Preparation, and stdev/means
    z1 = np.array(z1)
    if z2 is None:
        auto = True
    else:
        auto = False
        z2 = np.array(z2)
        if z1.shape != z2.shape:
            raise ValueError(
                f'Data 1 shape {z1.shape} and Data 2 shape {z2.shape} '
                'do not match.'
            )
    naxis = z1.shape[axis]  # length

    # Checks
    if not (nlag is None) ^ (lag is None):  # xor
        raise ValueError("Must specify either of the 'lag' or 'nlag' keyword args.")
    if nlag is not None and nlag >= naxis / 2:
        raise ValueError(f"Lag {nlag} must be greater than axis length {naxis}.")
    if verbose:
        prefix = 'auto' if auto else ''
        suffix = 'correlation' if corr else 'covariance'
        if nlag is None:
            print(f'Calculating lag-{lag} {prefix}{suffix}.')
        else:
            print(f'Calculating {prefix}{suffix} to lag {nlag} for axis size {naxis}.')

    # Means and permute
    std1 = std2 = 1  # use for covariance
    z1 = np.moveaxis(z1, axis, -1)
    mean1 = z1.mean(axis=-1, keepdims=True)  # keep dims for broadcasting
    if auto:
        z2, mean2 = z1, mean1
    else:
        z2 = np.moveaxis(z2, axis, -1)
        mean2 = z2.mean(axis=-1, keepdims=True)

    # Standardize maybe
    if corr:
        std1 = z1.std(axis=-1, keepdims=False)
        if auto:
            std2 = std1
        else:
            std2 = z2.std(axis=-1, keepdims=False)

    # This is just the variance, or *one* if autocorrelation mode is enabled
    # corrs = np.ones((*z1.shape[:-1], 1))
    if nlag is None and lag == 0:
        covar = np.sum((z1 - mean1) * (z2 - mean2)) / (naxis * std1 * std2),
        return np.moveaxis(covar, -1, axis)

    # Correlation on specific lag
    elif nlag is None:
        lag = np.round(lag * dt).astype(int)
        covar = np.sum(
            (z1[..., :-lag] - mean1) * (z2[..., lag:] - mean2), axis=-1, keepdims=True,
        ) / ((naxis - lag) * std1 * std2),
        return np.moveaxis(covar, -1, axis)

    # Correlation up to n timestep-lags after 0-correlation
    else:
        # First figure out lags
        # Negative lag means z2 leads z1 (e.g. z corr m, left-hand side
        # is m leads z).
        # e.g. 20 day lag, at synoptic timesteps
        nlag = np.round(nlag / dt).astype(int)
        if not auto:
            n = nlag * 2 + 1  # the center point, and both sides
            lags = range(-nlag, nlag + 1)
        else:
            n = nlag + 1
            lags = range(0, nlag + 1)

        # Get correlation
        # will include the zero-lag autocorrelation
        covar = np.empty((*z1.shape[:-1], n))
        for i, lag in enumerate(lags):
            if lag == 0:
                prod = (z1 - mean1) * (z2 - mean2)
            elif lag < 0:  # input 1 *trails* input 2
                prod = (z1[..., -lag:] - mean1) * (z2[..., :lag] - mean2)
            else:
                prod = (z1[..., :-lag] - mean1) * (z2[..., lag:] - mean2)
            covar[..., i] = prod.sum(axis=-1) / ((naxis - lag) * std1 * std2)
        return np.array(lags) * dt, np.moveaxis(covar, -1, axis)


def eof(
    data,
    neof=5,
    time=-2,
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
    data : array-like
        Data of arbitrary shape.
    neof : int, optional
        Number of eigenvalues we want.
    time : int, optional
        Axis used as the 'record' or 'time' dimension.
    space : int or list of int, optional
        Axis or axes used as 'space' dimension.
    weights : array-like, optional
        Area or mass weights; must be broadcastable on multiplication with
        `data` weights. Will be normalized prior to application.
    percent : bool, optional
        Whether to return raw eigenvalue(s) or the *percentage* of variance
        explained by eigenvalue(s).
    normalize : bool, optional
        Whether to normalize the data by its standard deviation
        at every point prior to obtaining the EOFs.

    Example
    -------

    >>> import xarray as xr
        import climpy
        array = xr.DataArray(
            np.random.rand(10, 5, 100, 40, 20),
            dims=('member', 'run', 'time', 'plev', 'lat'),
        )
        data = array.data
        result = climpy.eof(data, time=2, space=(3, 4))

    """
    # Parse input
    m_axis, n_axes = time, space
    np.atleast_1d(m_axis).item()  # ensure time axis is 1D
    shape = data.shape  # original shape

    # Remove the mean and optionally standardize the data
    data = data - data.mean(axis=m_axis, keepdims=True)  # remove mean
    if normalize:
        data = data / data.std(axis=m_axis, keepdims=True)

    # Next apply weights
    m = data.shape[m_axis]  # number timesteps
    n = np.prod([data.shape[_] for _ in n_axes])  # number space locations
    if weights is None:
        weights = 1
    weights = np.atleast_1d(weights)  # want numpy array
    weights = weights / weights.mean()  # so does not affect amplitude
    if m > n:  # longer time axis than space axis
        dataw = data = data * np.sqrt(weights)
    else:  # longer space axis than dime axis
        dataw = data * weights  # raises error if dimensions incompatible

    # Turn matrix in to extra (K) x time (M) x space (N)
    # Requires flatening space axes into one, and flattening extra axes into one
    with _ArrayContext(
        data, dataw,
        push_right=(m_axis, *n_axes),
        nflat_right=len(n_axes),  # flatten space axes
        nflat_left=data.ndim - len(n_axes) - 1,  # flatten
    ) as context:
        # Retrieve reshaped data
        data, dataw = context.data
        k, m, n = data.shape
        if m != shape[m_axis] or n != np.prod(np.asarray(shape)[n_axes]):
            raise RuntimeError('Array resizing algorithm failed.')

        # Prepare output arrays
        pcs = np.empty((k, neof, m, 1))
        projs = np.empty((k, neof, 1, n))
        evals = np.empty((k, neof, 1, 1))
        nstar = np.empty((k, 1, 1, 1))

        # Get EOFs and PCs and stuff
        for i in range(k):
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

        # Replace context data with new dimension inserted on left side
        context.replace(pcs, projs, evals, nstar, insert_left=1)

    # Return data restored to original dimensionality
    return context.data


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


def gaussian(N=1000, mean=0, stdev=None, sigma=1):
    """
    Returns sample points on Gaussian curve.
    """
    sigma = stdev if stdev is not None else sigma
    norm = stats.norm(loc=mean, scale=sigma)
    x = np.linspace(norm.ppf(0.0001), norm.ppf(0.9999), N)  # x through percentile range
    pdf = norm.pdf(x, loc=mean, scale=sigma)
    return x, pdf


@quack._xarray_xy_wrapper
@quack._pint_wrapper(('=y', '=y'), '')
@docstring.add_snippets
def hist(bins, y, /, axis=0):
    """
    Get the histogram along axis `axis`.

    Parameters
    ----------
    bins : array-like
        The bin location.
    y : array-like
        The data.
    %(hist.axis)s

    Example
    -------

    >>> import climpy
    ... import numpy as np
    ... import xarray as xr
    ... ureg = climpy.ureg
    ... data = xr.DataArray(
    ...     np.random.rand(20, 1000) * ureg.m,
    ...     dims=('x', 'y'),
    ...     coords={'x': np.arange(20), 'y': np.arange(1000) * 0.1}
    ... )
    ... bins = np.linspace(0, 1, 11) * ureg.m
    ... hist = climpy.hist(bins, data, axis=1)

    """
    if bins.ndim != 1:
        raise ValueError('Bins must be 1-dimensional.')

    with _ArrayContext(y, push_right=axis) as context:
        # Get flattened data
        y = context.data
        yhist = np.empty((y.shape[0], bins.size - 1))

        # Take histogram
        for k in range(y.shape[0]):
            yhist[k, :] = np.histogram(y[k, :], bins=bins)[0]

        # Replace data
        context.replace_data(yhist)

    # Return unflattened data
    return context.data


def linefit(x, y, /, axis=-1):
    """
    Get linear regression along axis, ignoring NaNs. Uses `~numpy.polyfit`.

    Parameters
    ----------
    x : array-like
        The *x* coordinates.
    y : array-like
        The *y* coordinates.
    axis : int, optional
        Regression axis

    Returns
    -------
    slope : array-like
        The slope estimates. The shape is the same as `y` but with
        dimension `axis` reduced to length 1.
    stderr : array-like
        The standard errors of the slope estimates. The shape is the
        same as `slope`.
    bestfit : array-like
        The reconstructed best-fit line. The shape is the same as `y`.
    """
    if x.ndim != 1 or x.size != y.shape[axis]:
        raise ValueError(
            f'Invalid x-shape {x.shape} for regression along axis {axis} '
            f'of y-shape {y.shape}.'
        )

    with _ArrayContext(y, push_right=axis) as context:
        # Get regression coefficients. Flattened data is shape (K, N)
        # where N is regression dimension. Permute to (N, K) then back again.
        # N gets replaced with length-2 dimension (slope, offset).
        y = context.data
        y_params, y_var = np.polyfit(x, y.T, deg=1, cov=True)
        y_params = np.fliplr(y_params.T)  # flip to (offset, slope)

        # Get best-fit line and slope
        y_fit = y_params[:, :1] + x * y_params[:, 1:]
        y_slope = y_params[:, 1:]

        # Get standard error
        # See Dave's paper (TODO: add citation)
        n = y.shape[1]
        resid = y - y_fit  # residual
        mean = resid.mean(axis=1)
        var = resid.var(axis=1)
        rho = np.sum(
            (resid[:, 1:] - mean[:, None]) * (resid[:, :-1] - mean[:, None]),
            axis=1,
        ) / ((n - 1) * var)  # correlation factor
        scale = (n - 2) / (n * ((1 - rho) / (1 + rho)) - 2)  # scale factor
        y_stderr = np.sqrt(y_var[0, 0, :] * scale)  # replace offset with stderr

        # Replace context data
        context.replace_data(y_slope, y_stderr, y_fit)

    return context.data


def rednoise(a, ntime, nsamples=1, mean=0, stdev=1):
    r"""
    Return one or more artificial red noise time series with prescribed mean
    and standard deviation. The time series are generated with the following
    equation:

    .. math::

        x(t) = a \cdot x(t - \Delta t) + b \cdot \epsilon(t)

    where *a* is the lag-1 autocorrelation and *b* is a scaling term.

    Parameters
    ---------
    a : float
        The autocorrelation.
    ntime : int
        Number of timesteps.
    nsamples : int or list of int, optional
        Axis size or list of axis sizes for the "sample" dimension(s). Shape of the
        `data` array will be ``(ntime,)`` if `nsamples` is not provided,
        ``(ntime, nsamples)`` if `nsamples` is scalar, or ``(ntime, *nsamples)``
        if `nsamples` is a list of axis sizes.
    mean, stdev : float, optional
        The mean and standard deviation for the red noise
        time series.

    Returns
    -------
    data : array-like
        The red noise data.

    See Also
    --------
    rednoisefit
    """
    # Initial stuff
    ntime -= 1  # exclude the initial timestep
    nsamples = np.atleast_1d(nsamples)
    data = np.empty((ntime + 1, *nsamples))  # user can make N-D array
    b = (1 - a ** 2) ** 0.5  # from OA class

    # Nested loop
    with _ArrayContext(data, push_left=0) as context:
        data = context.data
        data[0, :] = 0.0  # initialize
        for i in range(data.shape[-1]):
            eps = np.random.normal(loc=0, scale=1, size=ntime)
            for t in range(1, ntime + 1):
                data[t, i] = a * data[t - 1, i] + b * eps[t - 1]

    # Return
    data = context.data
    if len(nsamples) == 1 and nsamples[0] == 1:
        data = data.squeeze()
    return mean + stdev * data  # rescale to have specified stdeviation/mean


def rednoisefit(a, dt=1, nlag=None, nlag_fit=None, axis=-1):
    r"""
    Return the :math:`e`-folding autocorrelation timescale for the input
    autocorrelation spectra along an arbitrary axis. Depending on the length
    of `axis`, the timescale is obtained with either of the following two
    approaches:

    1. Find the :math:`e`-folding timescale(s) for the pure red noise
       autocorrelation spectra :math:`\exp(-x\Delta t / \tau)` with the
       least-squares best fit to the provided spectra.
    2. Assume the process *is* pure red noise and invert the
       red noise autocorrelation spectrum at lag 1 to solve for
       :math:`\tau = \Delta t / \log a_1`.

    Approach 2 is used if the length of the data along axis `axis` is ``1``,
    or the data is scalar. In these cases, the data is assumed to represent
    just the lag-1 autocorrelation(s).

    Parameters
    ----------
    a : array-like
        The autocorrelation spectra.
    dt : float, optional
        The timestep. This is used to scale timescales into physical units.
    nlag : int, optional
        The number of lag timesteps to include in the curve fitting. Default
        is all the available lags.
    nlag_fit : int, optional
        The number of lag timesteps to include in the reconstructed pure red
        noise autocorrelation spectrum. Default is 50 timesteps.
    axis : int, optional
        The "lag" dimension. Each slice along this axis should represent an
        autocorrelation spectrum generated with `corr`. If the length is ``1``, the
        data are assumed to be lag-1 autocorrelations and the timescale is computed
        from the red noise equation. Otherwise, the timescale is estimated from
        a least-squares curve fit to a red noise spectrum.

    Returns
    -------
    tau : array-like
        The autocorrelation timescales. The shape is the same as `data`
        but with `axis` reduced to length ``1``.
    sigma : array-like
        The standard errors. If the timescale was inferred using the lag-1
        equation, this is an array of zeros. The shape is the same as `tau`.
    bestfit : array-like
        The best fit autocorrelation curves. The shape is the same as `data`
        but with `axis` of length `nlag`.

    Example
    -------
    >>> import climpy
        auto = climpy.autocorr(data, axis=0)
        taus, sigmas = climpy.rednoise_fit(auto, axis=0)

    See Also
    --------
    corr, rednoise, rednoise_spectrum
    """
    # Best-fit function
    curve = lambda t, tau: np.exp(-t * dt / tau)
    nlag_fit = nlag_fit or 50
    lags_fit = np.arange(0, nlag_fit)

    with _ArrayContext(a, push_right=axis) as context:
        # Iterate over dimensions
        a = context.data
        nextra, nlag_in = a.shape
        nlag = nlag or nlag_in
        lags = np.arange(0, nlag)  # lags for the curve fit
        tau = np.empty((nextra, 1))
        sigma = np.empty((nextra, 1))
        afit = np.empty((nextra, nlag_fit))
        for i in range(nextra):
            if nlag <= 1:
                p = -dt / np.log(a[i, -1])
                s = 0  # no sigma, because no estimate
            else:
                p, s = optimize.curve_fit(curve, lags, a[i, :])
                s = np.sqrt(np.diag(s))
                p, s = p[0], s[0]  # take only first param
            tau[i, 0] = p  # just store the timescale
            sigma[i, 0] = s
            afit[i, :] = np.exp(-dt * lags_fit / p)  # best-fit spectrum

        # Replace context data
        context.replace_data(tau, sigma, afit)

    # Return permuted data
    return context.data


def wilks(percentiles, alpha=0.10):
    """
    Return the precentile threshold from an array of percentiles that satisfies
    the given false discovery rate. See :cite:`2006:wilks` for details.

    Parameters
    ----------
    percentiles : array-like
        The percentiles.
    alpha : float, optional
        The false discovery rate.

    References
    ----------
    .. bibliography:: ../bibs/wilks.bib
    """
    percentiles = np.asarray(percentiles)
    pvals = list(percentiles.flat)  # want in decimal
    pvals = sorted(2 * min(pval, 1 - pval) for pval in pvals)
    ptest = [alpha * i / len(pvals) for i in range(len(pvals))]
    ppick = max(pv for pv, pt in zip(pvals, ptest) if pv <= pt) / 2
    mask = (percentiles <= ppick) | (percentiles >= (1 - ppick))
    return percentiles[mask]
