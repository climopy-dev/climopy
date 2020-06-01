#!/usr/bin/env python3
"""
Tools for objective analysis-related tasks. Many of these are adapted
from examples and course notes provided by `Elizabeth Barnes \
<http://barnes.atmos.colostate.edu/COURSES/AT655_S15/lecture_slides.html>`__
and `Dennis Hartmann \
<https://atmos.washington.edu/~dennis/552_Notes_ftp.html>`__.

Note
----
The convention for this package is to use *linear* wave properties, i.e. the
wavelength in <units> per :math:`2\\pi` radians, and wavenumber
:math:`2\\pi` radians per <units>.
"""
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import scipy.linalg as linalg
import scipy.optimize as optimize
from .internals import docstring, warnings
from .internals.array import _ArrayContext

__all__ = [
    'autopower', 'autopower2d',
    'butterworth',
    'corr', 'covar',
    'eof', 'eot', 'reof',
    'filter', 'gaussian',
    'harmonics', 'highpower', 'impulse',
    'lanczos',
    'linefit',
    'power', 'power2d',
    'response',
    'roots',
    'rednoise', 'rednoise_fit', 'rednoise_spectrum',
    'rolling', 'running',
    'waves',
    'wilks',
    'window',
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


def roots(poly):
    """
    Finds real-valued root for polynomial with input coefficients.
    Format of input array p: p[0]*x^n + p[1]*x^n-1 + ... + p[n-1]*x + p[n]
    """
    # Just use numpy's roots function, and filter results.
    r = np.roots(poly)  # input polynomial; returns array-like
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
    rednoise_fit, rednoise_spectrum
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


def waves(x, wavenums=None, wavelens=None, phase=None):
    """
    Compose array of sine waves.
    Useful for testing performance of filters.

    Parameters
    ----------
    x : array-like
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
    data : array-like
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
    x : array-like, optional
        The *x* coordinates. Default is ``np.arange(0, len(y))``.
    y : array-like
        The *y* coordinates.
    axis : int, optional
        Regression axis
    stderr : bool, optional
        Whether to add standard error of the slope as index 1
        on the output array `axis`.
    build : bool, optional
        Whether to replace the output array `axis` with the
        reconstructed best-fit line.

    Returns
    -------
    array-like
        Array with the same shape as `y`, but with `axis` containing the
        the reconstructed best-fit line (if ``build=True``), `axis` of length
        2 containing the slope and slope standard error (if `stderr=True``),
        or `axis` of length 1 containing just the slope (if both ``build=False``
        and ``stderr=False``).
    """
    # Parse input
    if len(args) == 1:
        y, = args
        x = np.arange(y.shape[axis])
    elif len(args) == 2:
        x, y = args
    else:
        raise ValueError('Invalid input args.')
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

        # Prepare output
        y_fit = y_params[:, :1] + x * y_params[:, 1:]
        if build:
            # Replace regression dimension with best-fit line
            y_out = y_fit

        elif stderr:
            # Replace regression dimension with (slope, standard error)
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
            y_out = y_params
            y_out[:, 0] = np.sqrt(y_var[0, 0, :] * scale)  # replace offset with stderr
            y_out = np.fliplr(y_out)  # the array is (slope, standard error)

        else:
            # Replace regression dimension with (slope,)
            y_out = y_params[:, 1:]

        # Replace context data
        context.replace_data(y_out)

    return context.data


def rednoise_fit(auto, dt=1, nlag=None, axis=-1):
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
    auto : array-like
        The autocorrelation spectra.
    dt : float, optional
        The timestep. This is used to scale timescales into physical units.
    axis : int, optional
        The "lag" dimension. Each slice along this axis should represent an
        autocorrelation spectrum generated with `corr`.
        Axis along which the autocorrelation timescale is inferred. Data
        should consist of autocorrelation spectra generated with `corr`. If
        the length is ``1``, the data are assumed to be lag-1 autocorrelations
        and the timescale is computed from the red noise equation.
        Otherwise, the timescale is estimated from a least-squares curve fit
        to a red noise spectrum.

    Returns
    -------
    taus : array-like
        The autocorrelation timescales along `axis`. The shape is the same
        as `data` but with `axis` reduced to length ``1``.
    sigmas : array-like
        The standard errors for the curve fits. If the timescale was inferred
        using the lag-1 equation, this is an array of zeros.

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

    with _ArrayContext(auto, push_right=axis) as context:
        # Iterate over dimensions
        auto = context.data
        nextra, nlag = auto.shape
        lags = np.arange(0, nlag)  # lags for the curve fit
        taus = np.empty((nextra, 1))
        sigmas = np.zeros((nextra, 1))
        for i in range(nextra):
            # np.exp(-dt * lags / p)  # best-fit spectrum
            if nlag <= 1:
                p = -dt / np.log(auto[i, -1])
                s = 0  # no sigma, because no estimate
            else:
                p, s = optimize.curve_fit(curve, lags, auto[i, :])
                s = np.sqrt(np.diag(s))
                p, s = p[0], s[0]  # take only first param
            taus[i, 0] = p  # just store the timescale
            if nlag > 1:
                sigmas[i, 0] = s

        # Replace context data
        context.replace_data(taus, sigmas)

    # Return permuted data
    return context.data


def rednoise_spectrum():
    """
    Return the red noise autocorrelation spectra for the given input
    autocorrelation timescales.

    See Also
    --------
    rednoise, rednoise_fit

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError


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


def rolling(*args, **kwargs):
    """
    Alias for `running`.
    """
    return running(*args, **kwargs)


def running(x, w, axis=-1, pad=True, pad_value=np.nan):
    """
    Apply running average to array.

    Parameters
    ----------
    x : array-like
        Data, and we roll along axis `axis`.
    w : int or array-like
        Boxcar window length, or custom weights.
    axis : int, optional
        Axis to filter.
    pad : bool, optional
        Whether to pad the edges of axis back to original size.
    pad_value : float, optional
        The pad value.

    Returns
    -------
    x : array-like
        Data windowed along axis `axis`.

    Note
    ----
    Implementation is similar to `scipy.signal.lfilter`. Read
    `this post <https://stackoverflow.com/a/4947453/4970632>`__.

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
        axis += x.ndim
    x = np.moveaxis(x, axis, -1)

    # Determine weights
    if isinstance(w, str):
        raise NotImplementedError("Need to allow string 'w' argument, e.g. w='Lanczos'")
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
    strides = (*x.strides, x.strides[-1])  # repeat striding on end
    x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Next 'put back' axis, keep axis as
    # the 'rolling' dimension which can be averaged by arbitrary weights.
    x = np.moveaxis(x, -2, axis)  # want to 'put back' axis -2;

    # Finally take the weighted average
    # Note numpy will broadcast from right, so weights can be scalar or not
    # print(x.min(), x.max(), x.mean())
    x = (x * w).sum(axis=-1)  # broadcasts to right

    # Optionally fill the rows taken up
    if pad:
        n_new = x.shape[axis]
        n_left = (n_orig - n_new) // 2
        n_right = n_orig - n_new - n_left
        if n_left != n_right:
            warnings._warn_climpy('Data shifted left by one.')
        d_left = pad_value * np.ones((*x.shape[:axis], n_left, *x.shape[axis + 1:]))
        d_right = pad_value * np.ones((*x.shape[:axis], n_right, *x.shape[axis + 1:]))
        x = np.concatenate((d_left, x, d_right), axis=axis)
    return x


def filter(x, b, a=1, n=1, axis=-1, fix=True, pad=True, pad_value=np.nan):
    """
    Apply scipy.signal.lfilter to data. By default this does *not* pad
    ends of data. May keep it this way.

    Parameters
    ----------
    x : array-like
        Data to be filtered.
    b : array-like
        *b* coefficients (non-recursive component).
    a : array-like, optional
        *a* coefficients (recursive component); default of ``1`` indicates
        a non-recursive filter.
    n : int, optional
        Number of times to filter data. Will go forward --> backward
        --> forward...
    axis : int, optional
        Axis along which we filter data. Defaults to last axis.
    fix : bool, optional
        Whether to trim leading part of axis by number of *b* coefficients. Will
        also attempt to *re-center* the data if a net-forward (e.g. f, fbf, fbfbf, ...)
        filtering was performed. This works for non-recursive filters only.
    pad : bool, optional
        Whether to pad trimmed values (when `fix` is ``True``) with `pad_value`.
    pad_value : float
        The pad value.

    Returns
    -------
    y : array-like
        Data filtered along axis `axis`.

    Note
    ----
    * Consider adding empirical method for trimming either side of recursive
      filter that trims up to where impulse response is negligible.
    * If `x` has odd number of obs along axis, lfilter will trim
      the last one. Just like `rolling`.
    * The *a* vector contains (index 0) the scalar used to normalize *b*
      coefficients, and (index 1,2,...) the coefficients on `y` in the
      filtering conversion from `x` to `y`. So, 1 implies
      array of [1, 0, 0...], implies non-recursive.
    """
    # Parse input
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    n_half = (max(len(a), len(b)) - 1) // 2

    # Apply filter 'n' times to each sample
    with _ArrayContext(x, push_right=axis) as context:
        # Take mean
        y_filtered = context.data.copy()
        y_mean = y_filtered.mean(axis=1, keepdims=True)

        # Filter data
        y_filtered -= y_mean  # remove mean
        for i in range(n):  # applications of filter
            step = 1 if i % 2 == 0 else -1  # forward-backward application
            y_filtered[:, ::step] = signal.lfilter(b, a, y_filtered[:, ::step], axis=-1)
        y_filtered += y_mean  # add mean back in

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
                y_filtered = y_filtered[:, n_left:]
            else:
                y_filtered = y_filtered[:, (n_2sides + n_left):(-n_2sides)]

        # Optionally pad with a 'fill value' (usually NaN)
        if pad:
            y_left = pad_value * np.ones((y_filtered.shape[0], n_2sides + n_left // 2))
            y_right = pad_value * np.ones((y_filtered.shape[0], n_2sides + n_left // 2))
            y_filtered = np.concatenate((y_left, y_filtered, y_right), axis=-1)

        # Replace context data
        context.replace_data(y_filtered)

    # Return
    return context.data


def response(dx, b, a=1, n=1000, simple=False):
    """
    Calculate the response function given the *a* and *b* coefficients for some
    analog filter. For details, see Dennis Hartmann's objective analysis
    `course notes <https://atmos.washington.edu/~dennis/552_Notes_ftp.html>`__.

    Note
    ----
    The response function formula can be depicted as follows:

    .. code-block::

                     jw               -jw               -jmw
            jw   B(e)     b[0] + b[1]e    + .... + b[m]e
        H(e)   = ------ = ----------------------------------
                     jw               -jw               -jnw
                 A(e)     a[0] + a[1]e    + .... + a[n]e

    """
    # Parse input
    a = np.atleast_1d(a)
    x = np.linspace(0, np.pi, n)

    # Simple calculation given 'b' coefficients, from Libby's notes
    # Note we *need to make the exponent frequencies into
    # rad/physical units* for results to make sense.
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
    # with extra allowance for working with physical units.
    # Last entry will be Nyquist, i.e. 1/(dx*2)
    else:
        _, y = signal.freqz(b, a, x)
        x = x / (2 * np.pi * dx)
        y = np.abs(y)
    return x, y


def impulse():
    """
    Displays the *impulse* response function for a recursive filter.

    Warning
    -------
    Not yet implemented.
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
    x : `numpy.array-like`
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
    b : array-like
        Numerator coeffs.
    a : array-like
        Denominator coeffs.

    Note
    ----
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
    b : array-like
        Numerator coeffs.
    a : array-like
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
    Retrieve the `~scipy.signal.get_window` weighting function window.

    Parameters
    ----------
    wintype : str or (str, float) tuple
        The window name or ``(name, param1, ...)`` tuple containing the window
        name and required parameter(s).
    n : int
        The window length.

    Returns
    -------
    win : array-like
        The window coefficients.

    power of your FFT coefficients. If your window requires some parameter,
    `wintype` must be a ``(name, parameter1, ...)`` tuple.
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


@docstring.add_snippets
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
    Return the spectral decomposition of a real-valued dataset along an
    arbitrary axis with arbitrary windowing behavior.

    Parameters
    ----------
    y1 : array-like
        Input data.
    y2 : array-like, default ``None``
        Second input data, if cross-spectrum is desired.
        Must have same shape as `y1`.
    dx : float, optional
        Time dimension step size in physical units. Used to scale `f`.
    cyclic : bool, optional
        Whether data is cyclic along axis. If ``True``, the *nperseg*
        argument will be overridden
    %(power.params)s

    Returns
    -------
    f : array-like
        Frequencies in units <x units>**-1. Scaled with `dx`.
    P : array-like, optional
        Power spectrum in units <data units>**2.
        Returned if `z2` is ``None``.
    P, Q, Pz1, Pz2 : array-like, optional
        Co-power spectrum, quadrature spectrum, power spectrum for `z1`, and
        power spectrum for `z2`, respectively.
        Returned if `z2` is not ``None`` and `coherence` is ``False``.
    Coh, Phi : array-like, optional
        Coherence and phase difference, respectively.
        Returned if `z2` is not ``None`` and `coherence` is ``True``.

    Note
    ----
    %(power.notes)s

    References
    ----------
    %(power.bibliography)s
    """
    # Initial stuff
    copower = y2 is not None
    N = y1.shape[axis]  # window count
    if cyclic:
        wintype = 'boxcar'
        nperseg = N
    if copower and y2.shape != y1.shape:
        raise ValueError(f'Got conflicting shapes for y1 {y1.shape} and y2 {y2.shape}.')
    nperseg = 2 * (nperseg // 2)  # enforce even window size

    # Trim if necessary
    rem = N % nperseg
    if rem != 0:
        ndim = y1.ndim
        slices = [slice(None, -rem) if i == axis else slice(None) for i in range(ndim)]
        y1 = y1[tuple(slices)]
        if copower:
            y2 = y2[tuple(slices)]
        warnings._warn_climpy(
            f'Trimmed {rem} out of {N} points to accommodate length-{nperseg} window.'
        )

    # Auto approach with scipy.csd. 'one-sided' says to only return first symmetric
    # half if data is real 'scaling' queries whether to:
    #  * Scale 'per wavenumber'/'per Hz', option 'density', default.
    #    This is analagous to a Planck curve with intensity per wavenumber
    #  * Show the power (so units are just data units squared).
    #    This is usually what we want.
    # if not manual and not copower:
    #     f, P = signal.csd(y1, y1, window=wintype,
    #       return_onesided=True, scaling=scaling,
    #       nperseg=nperseg, noverlap=nperseg//2, detrend=detrend,
    #       axis=axis
    #     )

    # Manual approach (have compared to auto and results are identical)
    # Get copsectrum, quadrature spectrum, and powers for each window
    # shape is shape of *original* data
    pm = nperseg // 2
    args = (y1, y2) if copower else (y1,)
    with _ArrayContext(*args, push_right=axis) as context:
        # Get flattened data
        if copower:
            y1, y2 = context.data
        else:
            y1 = context.data

        # List of *center* indices for windows
        K, N = y1.shape[1]
        win = window(wintype, nperseg)
        winloc = np.arange(pm, N - pm + pm // 2, pm)  # jump by half window length
        nwindows = winloc.size
        if nwindows == 0:
            raise ValueError(f'Window length {nperseg} too big for size-{N} dimension.')

        # Setup output arrays
        Py1 = np.empty((K, nwindows, pm))  # we take the mean of dimension 1
        if copower:
            Py2 = Py1.copy()
            C = Py1.copy()
            Q = Py1.copy()

        # Loop through windows. Remember to double the size of power, because we
        # only have half the coefficients (rfft not fft).
        for k in range(K):
            if (
                np.any(~np.isfinite(y1[k, :]))
                or copower and np.any(~np.isfinite(y2[k, :]))
            ):
                warnings._warn_climpy('Skipping array with missing values.')
                continue
            for i, l in enumerate(winloc):
                wy1 = win * signal.detrend(y1[k, l - pm:l + pm], type=detrend)
                Fy1 = np.fft.rfft(wy1)[1:] / win.sum()
                Py1[k, i, :] = np.abs(Fy1) ** 2
                Py1[k, i, :-1] *= 2
                if copower:
                    wy2 = win * signal.detrend(y2[k, l - pm:l + pm], type=detrend)
                    Fy2 = np.fft.rfft(wy2)[1:] / win.sum()
                    Py2[k, i, :] = np.abs(Fy2) ** 2
                    C[k, i, :] = Fy1.real * Fy2.real + Fy1.imag * Fy2.imag
                    Q[k, i, :] = Fy1.real * Fy2.imag - Fy2.real * Fy1.imag
                    Py2[k, i, :-1] *= 2  # scale all but the Nyquist frequency
                    C[k, i, :-1] *= 2
                    Q[k, i, :-1] *= 2

        # Get window averages
        f = np.fft.rfftfreq(nperseg)[1:]  # frequencies
        Py1 = Py1.mean(axis=1)
        if copower:
            Py2 = Py2.mean(axis=1)
            C = C.mean(axis=1)
            Q = Q.mean(axis=1)

        # Get output arrays
        if not copower:
            arrays = (Py1,)
        elif not coherence:
            arrays = (C, Q, Py1, Py2)
        else:
            # Coherence and phase difference
            Coh = (C ** 2 + Q ** 2) / (Py1 * Py2)
            Phi = np.arctan2(Q, C)  # phase
            Phi[Phi >= center + np.pi] -= 2 * np.pi
            Phi[Phi < center - np.pi] += 2 * np.pi
            Phi = Phi * 180 / np.pi  # convert to degrees!!!
            arrays = (Coh, Phi)

        # Replace arrays
        context.replace_data(*arrays)

    if copower:
        return (f / dx, *context.data)
    else:
        return (f / dx, context.data)


def _decomp_2d(pm, win, x, detrend='constant'):
    """
    Get 2D Fourier decomp and reorder negative frequencies on non-cyclic
    axis so frequencies there are monotonically ascending.
    """
    # Detrend
    x = signal.detrend(x, type=detrend, axis=0)  # remove trend or mean from "time"
    x = signal.detrend(x, type='constant', axis=1)  # remove mean from "longitude"

    # Use 1D numpy.fft.rfft (identical)
    # Follows Libby's recipe, where instead real is cosine and imag is
    # sine. Note only need to divide by 2 when conjugates are included.
    # xi = np.fft.rfft(x, axis=1)[:,1:]/x.shape[1]
    # xi = win[:,None]*xi # got a bunch of sines and cosines
    # C = np.fft.rfft(xi.real, axis=0)[1:,:]/x.shape[0]
    # S = np.fft.rfft(xi.imag, axis=0)[1:,:]/x.shape[0]
    # return np.concatenate(
    #     (
    #          (C.real + S.imag + 1j * (C.imag - S.real))[::-1, :],
    #          C.real - S.imag + 1j * (-C.imag - S.real),
    #     ),
    #     axis=0,
    # )

    # Use 2D numpy.fft.rfft2
    # NOTE: Read documentation regarding normalization. Default leaves
    # forward transform unnormalized, reverse normalized by 1 / n. The ortho
    # option normalizes both by 1/sqrt(n).
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/routines.fft.html#module-numpy.fft
    # last axis specified should get a *real* transform
    X = np.fft.rfft2(win[:, None] * x, axes=(0, 1))  # last axis gets real transform
    X = X[:, 1:]  # remove the zero-frequency value
    X = X / (x.shape[0] * x.shape[1])  # normalize by sample size
    return np.concatenate((X[pm:, :], X[1:pm + 1, :]), axis=0)


@docstring.add_snippets
def power2d(
    z1,
    z2=None,
    dx=1,
    dy=1,
    coherence=False,
    nperseg=None,
    wintype='boxcar',
    center=np.pi,
    axes=(-2, -1),
    detrend='constant',
):
    """
    Return the 2-d spectral decomposition of a real-valued dataset with
    one *time* dimension and one *cyclic* dimension along arbitrary axes with
    arbitrary windowing behavior. For details, see :cite:`1991:randel`.

    Parameters
    ----------
    z1 : array-like
        Input data.
    z2 : array-like, default ``None``
        Second input data, if cross-spectrum is desired. Must have same
        shape as `z1`.
    dx : float, optional
        Time dimension step size in physical units. Used to scale `fx`.
    dy : float, optional
        Cyclic dimension step size in physical units. Used to scale `fy`.
    axes : (int, int), optional
        Locations of the "time" and "cyclic" axes, respectively.
        By default the second-to-last and last axes are used.
    %(power.params)s

    Returns
    -------
    fx : array-like
        Time dimension frequencies in units <x units>**-1. Scaled with `dx`.
    fy : array-like
        Cyclic dimension wavenumbers in units <y units>**-1. Scaled with `dy`.
    P : array-like, optional
        Power spectrum in units <data units>**2.
        Returned if `z2` is ``None``.
    P, Q, Pz1, Pz2 : array-like, optional
        Co-power spectrum, quadrature spectrum, power spectrum for `z1`, and
        power spectrum for `z2`, respectively.
        Returned if `z2` is not ``None`` and `coherence` is ``False``.
    Coh, Phi : array-like, optional
        Coherence and phase difference, respectively.
        Returned if `z2` is not ``None`` and `coherence` is ``True``.

    Note
    ----
    %(power.notes)s

    References
    ----------
    %(power.bibliography)s
    """
    # Checks
    copower = z2 is not None
    if len(z1.shape) < 2:
        raise ValueError('Need at least rank 2 array.')
    if copower and not z1.shape == z2.shape:
        raise ValueError(f'Shapes of z1 {z1.shape} and z2 {z2.shape} must match.')
    taxis, caxis = axes
    if caxis < 0:
        caxis += z1.ndim
    if taxis < 0:
        taxis += z1.ndim
    nperseg = 2 * (nperseg // 2)  # enforce even window size

    # Trim data to fit window length
    print(f'Cyclic dim ({caxis}): Length {z1.shape[caxis]}.')
    print(f'Windowed dim ({taxis}): Length {z1.shape[taxis]}, window length {nperseg}.')
    N = z1.shape[taxis]
    rem = N % nperseg
    if rem > 0:
        ndim = z1.ndim
        slices = [slice(None, -rem) if i == taxis else slice(None) for i in range(ndim)]
        z1 = z1[tuple(slices)]
        if copower:
            z2 = z2[tuple(slices)]
        warnings._warn_climpy(
            f'Trimmed {rem} out of {N} points to accommodate length-{nperseg} window.'
        )

    # Permute and flatten
    args = (z1, z2) if copower else (z1,)
    with _ArrayContext(*args, push_right=(taxis, caxis)) as context:
        # Get flattened data
        if copower:
            z1, z2 = context.data
        else:
            z1 = context.data

        # The window *centers* for time windowing. Jump by half window length
        K, N, M = z1.shape
        pm = nperseg // 2
        win = window(wintype, nperseg)  # for time domain
        winloc = np.arange(pm, N - pm + 0.1, pm).astype(int)
        nwindows = winloc.size
        if nwindows == 0:
            raise ValueError('Window length too big.')

        # Setup output arrays
        Pz1 = np.nan * np.empty((K, nwindows, 2 * pm, M // 2))  # power array
        if copower:
            Pz2 = Pz1.copy()
            C = Pz1.copy()
            Q = Pz1.copy()

        # 2D transform for each window on non-cyclic dimension
        # Note since we got the rfft (not fft) in one direction, only have half the
        # coefficients (they are symmetric); means for correct variance, have to
        # double the power. These are analagous to Libby's notes for complex space
        for k in range(K):
            if (
                np.any(~np.isfinite(z1[k, :, :]))
                or copower and np.any(~np.isfinite(z2[k, :, :]))
            ):
                warnings._warn_climpy('Skipping array with missing values.')
                continue
            for i, idx in enumerate(winloc):
                Fz1 = _decomp_2d(pm, win, z1[k, idx - pm:idx + pm, :], detrend)
                Pz1[k, i, :, :] = np.abs(Fz1) ** 2
                Pz1[k, i, :, :-1] *= 2
                if copower:
                    Fz2 = _decomp_2d(pm, win, z2[k, idx - pm:idx + pm, :], detrend)
                    Pz2[k, i, :, :] = np.abs(Fz2) ** 2
                    Phi1 = np.arctan2(Fz1.imag, Fz1.real)
                    Phi2 = np.arctan2(Fz2.imag, Fz2.real)
                    C[k, i, :, :] = np.abs(Fz1) * np.abs(Fz2) * np.cos(Phi1 - Phi2)
                    Q[k, i, :, :] = np.abs(Fz1) * np.abs(Fz2) * np.sin(Phi1 - Phi2)
                    Pz2[k, i, :, :-1] *= 2
                    C[k, i, :, :-1] *= 2
                    Q[k, i, :, :-1] *= 2

        # Output frequencies.
        # TODO: Why remove mean power?
        # NOTE: Default order is to go 0 1 ... N/2 -N/2 ... -1. We reorder so
        # frequencies are from -N/2 ... -1 1 ... N/2.
        fy = np.fft.rfftfreq(M)[1:]
        fx = np.fft.fftfreq(2 * pm)  # start with the positive Fourier coefficients
        fq = np.abs(fx[pm:pm + 1])  # Nyquist frequency singleton array
        fx = np.concatenate((-fq, fx[pm + 1:], fx[1:pm], fq), axis=0)

        # Take average along windows
        Pz1 = Pz1.mean(axis=1)
        if copower:
            Pz2 = Pz2.mean(axis=1)
            C = C.mean(axis=1)
            Q = Q.mean(axis=1)

        # Get output arrays
        if not copower:
            arrays = (Pz1,)
        elif not coherence:
            arrays = (C, Q, Pz1, Pz2)
        else:
            # Get coherence and phase difference
            # NOTE: This Phi relationship is still valid. Check Libby notes. Divide
            # here Q by C and the Ws cancel out, end up with average phase diff.
            Coh = (C ** 2 + Q ** 2) / (Pz1 * Pz2)
            Phi = np.arctan2(Q, C)  # phase
            Phi[Phi >= center + np.pi] -= 2 * np.pi
            Phi[Phi < center - np.pi] += 2 * np.pi
            arrays = (Coh, Phi)

        # Replace context data
        context.replace_data(*arrays)

    # Return unflattened data
    if copower:
        return (fx / dx, fy / dy, *context.data)
    else:
        return (fx / dx, fy / dy, context.data)


def autopower():
    """
    Wrapper around `power` that generates co-spectral
    statistics and whatnot at *successive lags*.

    Warning
    -------
    Not yet implemented.
    """
    # Uses scipy.signal.welch windowing method to generate an estimate of the
    # *lagged* spectrum. Can also optionally do this with two variables.
    raise NotImplementedError


def autopower2d():
    """
    Wrapper around `power2d` that generates co-spectral
    statistics and whatnot at *successive lags*.

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError
