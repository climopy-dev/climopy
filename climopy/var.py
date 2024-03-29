#!/usr/bin/env python3
"""
Analyses of variance and trends. Many of these are adapted
from examples and course notes provided by Professors `Elizabeth Barnes \
<http://barnes.atmos.colostate.edu/COURSES/AT655_S15/lecture_slides.html>`__
and `Dennis Hartmann \
<https://atmos.washington.edu/~dennis/552_Notes_ftp.html>`__.
"""
import numpy as np
import numpy.ma as ma
import scipy.linalg as linalg
import scipy.optimize as optimize

from .internals import ic  # noqa: F401
from .internals import docstring, permute, quack, quant

__all__ = [
    'autocorr',
    'autocovar',
    'corr',
    'covar',
    'eof',
    'eot',
    'reof',
    'hist',
    'rednoise',
    'slope',
    'timescale',
    'wilks',
]

# Docstring snippets
_var_data = """
z : array-like
    The input data.
"""
_covar_data = """
z1 : array-like
    The input data.
z2 : array-like, optional
    The second input data. Must be same shape as `z1`.
"""
_corr_math = r"""
.. math::

    \dfrac{%
        \sum_{i=0}^{n-k}
        \left(x_t - \overline{x}\right)
        \left(y_{t+k} - \overline{y}\right)
    }{%
        (n - k) s_x s_y
    }

where :math:`\overline{x}` and :math:`\overline{y}` are the sample means and
:math:`s_x` and :math:`s_y` are the sample standard deviations.
"""  # noqa: E501
_covar_math = r"""
.. math::

    \dfrac{%
        \sum_{i=0}^{n-k}
        \left(x_t - \overline{x}\right)
        \left(y_{t+k} - \overline{y}\right)
    }{%
        n - k
    }

where :math:`\overline{x}` and :math:`\overline{y}` are the sample means.
"""  # noqa: E501
_var_template = """
Return the %(name)s spectrum at a single lag or successive lags. Default
behavior returns the lag-0 %(name)s.

Parameters
----------
dt : float or array-like, optional
    The timestep or time series (from which the timestep is inferred). Default is
    ``1`` (note this makes `lag` and `ilag`, `maxlag` and `imaxlag` identical).
%(data)s
axis : int, optional
    Axis along which %(name)s is taken.
dim : str, optional
    *For `xarray.DataArray` input only*.
    Named dimension along which %(name)s is taken.
lag : float, optional
    Return %(name)s for the single lag `lag` (must be divisible by `dt`).
ilag : int, default: 0
    As with `lag` but specifies the index instead of the physical time.
maxlag : float, optional
    Return lagged %(name)s up to the lag `maxlag` (must be divisible by `dt`).
imaxlag : int, optional
    As with `maxlag` but specifies the index instead of the physical time.

Returns
-------
lags : array-like, optional
    The lags. Only returned if `maxlag` or `imaxlag` was used.
result : array-like
    The %(name)s as a function of lag.

Notes
-----
This function uses the following formula to estimate %(name)s at lag :math:`k`:

%(math)s
"""
docstring._snippet_manager['template_var'] = _var_template


def rednoise(a, ntime=100, nsamples=1, mean=0, stdev=1, state=None):
    r"""
    Return one or more artificial red noise time series with prescribed mean and
    standard deviation. The time series are generated with the following equation:

    .. math::

        x(t) = a \cdot x(t - \Delta t) + b \cdot \epsilon(t)

    where *a* is the lag-1 autocorrelation and *b* is a scaling term.

    Parameters
    ---------
    a : float
        The autocorrelation.
    ntime : int, optional
        Number of timesteps.
    nsamples : int or sequence of int, optional
        Axis size or sequence of axis sizes for the "sample" dimension(s). Shape of
        the output array will be ``(ntime,)`` if `nsamples` is not provided,
        ``(ntime, nsamples)`` if `nsamples` is scalar, or ``(ntime, *nsamples)``
        if `nsamples` is a sequence of axis sizes.
    mean, stdev : float, optional
        The mean and standard deviation for the red noise time series.
    state : `numpy.RandomState`, optional
        The random state to use for generating the data.

    Returns
    -------
    data : array-like
        The red noise data.

    See Also
    --------
    timescale
    """
    # Initial stuff
    ntime -= 1  # exclude the initial timestep
    nsamples = np.atleast_1d(nsamples)
    data = np.empty((ntime + 1, *nsamples))  # user can make N-D array
    b = (1 - a ** 2) ** 0.5  # from OA class

    # Nested loop
    if state is None:
        state = np.random
    with permute._PermuteContext(data, push_left=0) as ctx:
        data = ctx.data
        data[0, :] = 0.0  # initialize
        for i in range(data.shape[-1]):
            eps = state.normal(loc=0, scale=1, size=ntime)
            for t in range(1, ntime + 1):
                data[t, i] = a * data[t - 1, i] + b * eps[t - 1]

    # Return
    data = ctx.data
    if len(nsamples) == 1 and nsamples[0] == 1:
        data = data.squeeze()
    return mean + stdev * data  # rescale to have specified stdeviation/mean


def _covar_driver(
    dt, z1, z2, /, *, lag=None, ilag=None, maxlag=None, imaxlag=None,
    axis=0, standardize=False, verbose=False,
):
    """
    Driver function for getting covariance.
    """
    # Preparation, and stdev/means
    dt = quack._as_step(dt)
    auto = z1 is z2
    if z1.shape != z2.shape:
        raise ValueError(f'Incompatible shapes {z1.shape=} and {z2.shape=}.')
    naxis = z1.shape[axis]  # length

    # Parse input args
    npassed = sum(_ is not None for _ in (lag, ilag, maxlag, imaxlag))
    if npassed == 0:
        ilag = 0
    elif npassed != 1:
        raise ValueError(f'Conflicting kwargs {lag=}, {ilag=}, {maxlag=}, {imaxlag=}.')
    if any(_ is not None and not 0 <= _ < naxis - 3 for _ in (ilag, imaxlag)):
        raise ValueError(f'Lag index must satisfy 0 <= lag < {naxis - 3}.')
    if any(_ is not None and not 0 <= _ < dt * (naxis - 3) for _ in (lag, maxlag)):
        raise ValueError(f'Lag time must satisfy 0 <= lag < {dt * (naxis - 3)}.')
    if any(_ is not None and not np.isclose(_ % dt, 0) for _ in (lag, maxlag)):
        raise ValueError(f'Lag time must be divisible by timestep {dt}.')
    if lag is not None:
        ilag = np.round(lag / dt).astype(int)
    if maxlag is not None:
        imaxlag = np.round(maxlag / dt).astype(int)
    if verbose:
        prefix = 'auto' if auto else ''
        suffix = 'correlation' if standardize else 'covariance'
        if maxlag is None:
            print(f'Calculating lag-{lag} {prefix}{suffix}.')
        else:
            print(f'Calculating {prefix}{suffix} to lag {maxlag} for axis size {naxis}.')  # noqa: E501

    # Mask data
    z1 = ma.masked_invalid(z1)
    z2 = ma.masked_invalid(z2)

    # Means and permute
    z1 = np.moveaxis(z1, axis, -1)
    mean1 = z1.mean(axis=-1, keepdims=True)  # keep dims for broadcasting
    if auto:
        z2, mean2 = z1, mean1
    else:
        z2 = np.moveaxis(z2, axis, -1)
        mean2 = z2.mean(axis=-1, keepdims=True)

    # Standardize maybe
    std1 = std2 = np.array([1])  # use for covariance
    if standardize:
        std1 = z1.std(axis=-1, keepdims=True)
        if auto:
            std2 = std1
        else:
            std2 = z2.std(axis=-1, keepdims=True)
    std1[std1 == 0] = std2[std2 == 0] = 1  # avoid nan error for constant series

    # Covariance at zero-lag (included for consistency)
    if ilag == 0 or imaxlag == 0:
        lags = [0]
        covar = np.sum(
            (z1 - mean1) * (z2 - mean2),
            axis=-1, keepdims=True,
        ) / (naxis * std1 * std2)

    # Covariance on specific lag
    elif ilag is not None:
        lags = np.array([dt * ilag])
        covar = np.sum(
            (z1[..., :-ilag] - mean1) * (z2[..., ilag:] - mean2),
            axis=-1, keepdims=True,
        ) / ((naxis - ilag) * std1 * std2)

    # Covariance up to n timestep-lags after 0-correlation. Make this
    # symmetric if this is not an 'auto' function (i.e. extend to negative lags).
    else:
        if not auto:
            ilags = np.arange(-imaxlag, imaxlag + 1)
        else:
            ilags = np.arange(0, imaxlag + 1)
        lags = dt * ilags
        covar = np.empty((*z1.shape[:-1], ilags.size))
        for i, ilag in enumerate(ilags):
            if ilag == 0:
                prod = (z1 - mean1) * (z2 - mean2)
            elif ilag < 0:  # input 1 *trails* input 2
                prod = (z1[..., -ilag:] - mean1) * (z2[..., :ilag] - mean2)
            else:
                prod = (z1[..., :-ilag] - mean1) * (z2[..., ilag:] - mean2)
            covar[..., i] = (
                prod.sum(axis=-1, keepdims=True)
                / ((naxis - ilag) * std1 * std2)
            )[..., 0]

    # Return lags and covariance
    # TODO: Only return lags if 'lag' and 'ilag' was passed.
    # Figure out how to do this with wrapper functions.
    return lags, np.moveaxis(covar, -1, axis)


@quack._covar_metadata
@quant.while_dequantified(('=t', '=z'), ('=t', ''))
@docstring._snippet_manager(name='autocorrelation', data=_var_data, math=_corr_math)
def autocorr(dt, z, axis=0, **kwargs):
    """
    %(template_var)s
    """
    # NOTE: covar checks if z1 and z2 are the same to to reduce computational cost
    kwargs.setdefault('standardize', True)
    return _covar_driver(dt, z, z, axis=axis, **kwargs)


@quack._covar_metadata
@quant.while_dequantified(('=t', '=z'), ('=t', '=z ** 2'))
@docstring._snippet_manager(name='autocovariance', data=_var_data, math=_covar_math)
def autocovar(dt, z, axis=0, **kwargs):
    """
    %(template_var)s
    """
    # NOTE: covar checks if z1 and z2 are the same to reduce computational cost
    return _covar_driver(dt, z, z, axis=axis, **kwargs)


@quack._covar_metadata
@quant.while_dequantified(('=t', '=z1', '=z2'), ('=t', ''))
@docstring._snippet_manager(name='correlation', data=_covar_data, math=_corr_math)
def corr(dt, z1, z2, axis=0, **kwargs):
    """
    %(template_var)s
    """
    kwargs.setdefault('standardize', True)
    return _covar_driver(dt, z1, z2, axis=axis, **kwargs)


@quack._covar_metadata
@quant.while_dequantified(('=t', '=z1', '=z2'), ('=t', '=z1 * z2'))
@docstring._snippet_manager(name='covariance', data=_covar_data, math=_covar_math)
def covar(dt, z1, z2, axis=0, **kwargs):
    """
    %(template_var)s
    """
    return _covar_driver(dt, z1, z2, axis=axis, **kwargs)


@quack._eof_metadata
@quant.while_dequantified('=z', ('', '=z', '=z ** 2', 'count'))
def eof(
    data, /, neof=5, axis_time=-2, axis_space=-1,
    weights=None, percent=False, normalize=False,
):
    r"""
    Calculate the first `N` EOFs using the scipy algorithm for Hermetian matrices
    on the covariance matrix.

    Parameters
    ----------
    data : array-like
        Data of arbitrary shape.
    neof : int, optional
        Number of eigenvalues we want.
    axis_time : int, optional
        Axis used as the 'record' or 'time' dimension.
    axis_space : int or sequence of int, optional
        Axis or axes used as 'space' dimension.
    weights : array-like, optional
        Area or mass weights; must be broadcastable on multiplication with
        `data` weights. Will be normalized prior to application.
    percent : bool, optional
        Whether to return raw eigenvalue(s) or the *percentage* of total
        variance explained by eigenvalue(s). Default is ``False``.
    normalize : bool, optional
        Whether to normalize the data by its standard deviation
        at every point prior to obtaining the EOFs.

    Returns
    -------
    pcs : array-like
        The standardized principal components. The `axis_space` dimensions
        are reduced to length 1.
    projs : array-like
        Projections of the standardized principal components onto the
        original data. The `axis_time` dimension is reduced to length 1.
    evals
        If `percent` is ``Flase``, these are the eigenvalues. Otherwise, this is
        the percentage of total variance explained by the corresponding eigenvector.
        The `axis_time` and `axis_space` dimensions are reduced to length 1.
    nstars
        The approximate degrees of freedom as determined by the :cite:`2011:wilks`
        autocorrelation critereon. This can be used to compute the approximate 95%
        error bounds for the eigenvalues using the :cite:`1982:north` critereon of
        :math:`\lambda \sqrt{2 / N^*}`. The `axis_time` and `axis_space` dimensions
        are reduced to length 1.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import climopy as climo
    >>> state = np.random.RandomState(51423)
    >>> data = xr.DataArray(
    ...     state.rand(10, 6, 100, 40, 20),
    ...     dims=('member', 'run', 'time', 'lev', 'lat'),
    ...     coords={
    ...         'member': np.arange(1, 11),
    ...         'run': np.arange(1, 7),
    ...         'time': np.arange(100.0),
    ...         'lev': np.linspace(0.0, 1000.0, 40),
    ...         'lat': np.linspace(-90.0, 90.0, 20),
    ...     }
    ... )
    >>> pcs, projs, evals, nstars = climo.eof(data, axis_time=2, axis_space=(3, 4))
    >>> pcs.sizes
    Frozen({'eof': 5, 'member': 10, 'run': 6, 'time': 100, 'lev': 1, 'lat': 1})
    >>> projs.sizes
    Frozen({'eof': 5, 'member': 10, 'run': 6, 'time': 1, 'lev': 40, 'lat': 20})
    >>> pcs.head(time=1, run=1, member=1).T
    <xarray.DataArray (lat: 1, lev: 1, time: 1, run: 1, member: 1, eof: 5)>
    array([[[[[[-0.13679781,  1.08751657,  2.52901891,  0.00737416,
                 0.55085823]]]]]])
    Coordinates:
      * member   (member) int64 1
      * run      (run) int64 1
      * time     (time) float64 0.0
      * eof      (eof) int64 1 2 3 4 5
    Dimensions without coordinates: lat, lev
    >>> projs.head(lat=1, lev=1, run=1, member=1).T
    <xarray.DataArray (lat: 1, lev: 1, time: 1, run: 1, member: 1, eof: 5)>
    array([[[[[[-0.02304145, -0.01572039,  0.02761249, -0.06884522,
                 0.04163672]]]]]])
    Coordinates:
      * member   (member) int64 1
      * run      (run) int64 1
      * lev      (lev) float64 0.0
      * lat      (lat) float64 -90.0
      * eof      (eof) int64 1 2 3 4 5
    Dimensions without coordinates: time

    References
    ----------
    .. bibliography:: ../bibs/eofs.bib
    """
    # Parse input
    # WARNING: Have to explicitly 'push_left' extra dimensions (even though just
    # specifying nflat_left is enough to flattened the desired dimensions) or the new
    # 'eof' dimension is not pushed to the left of the extra dimensions. Think this
    # is not a bug but consistent with design.
    axis_space = np.atleast_1d(axis_space)
    np.atleast_1d(axis_time).item()  # ensure time axis is 1D
    axis_space[axis_space < 0] += data.ndim
    if axis_time < 0:
        axis_time += data.ndim
    if np.any(axis_space == axis_time):
        raise ValueError(f'Cannot have {axis_time=} same as {axis_space=}.')
    axis_extra = tuple(set(range(data.ndim)) - {axis_time, *axis_space})

    # Remove the mean and optionally standardize the data
    data = data - data.mean(axis=axis_time, keepdims=True)  # remove mean
    if normalize:
        data = data / data.std(axis=axis_time, keepdims=True)

    # Next apply weights
    ntime = data.shape[axis_time]  # number timesteps
    nspace = np.prod(np.asarray(data.shape)[axis_space])  # number space locations
    if weights is None:
        weights = 1
    weights = np.atleast_1d(weights)  # want numpy array
    weights = weights / weights.mean()  # so does not affect amplitude
    if ntime > nspace:
        dataw = data = data * np.sqrt(weights)
    else:
        dataw = data * weights  # raises error if dimensions incompatible

    # Turn matrix in to extra (K) x time (M) x space (N)
    # Requires flatening space axes into one, and flattening extra axes into one
    shape_orig = data.shape
    with permute._PermuteContext(
        data, dataw,
        push_left=axis_extra,
        push_right=(axis_time, *axis_space),
        nflat_right=len(axis_space),  # flatten space axes
        nflat_left=data.ndim - len(axis_space) - 1,  # flatten
    ) as ctx:
        # Retrieve reshaped data
        data, dataw = ctx.data
        k = data.shape[0]  # ensure 3D
        if data.ndim != 3 or ntime != data.shape[1] or nspace != data.shape[2]:
            raise RuntimeError(
                'Array resizing algorithm failed. '
                f'Expected (N x {ntime} x {nspace}). '
                f'Turned shape from {shape_orig} to {data.shape}.'
            )

        # Prepare output arrays
        pcs = np.empty((k, neof, ntime, 1))
        projs = np.empty((k, neof, 1, nspace))
        evals = np.empty((k, neof, 1, 1))
        nstars = np.empty((k, 1, 1, 1))

        # Get EOFs and PCs and stuff
        for i in range(k):
            # Get matrices
            x = data[i, :, :]  # array will be sampling by space
            xw = dataw[i, :, :]

            # Get reduced degrees of freedom for spatial eigenvalues
            # TODO: Fix the weight projection below
            rho = np.corrcoef(x.T[:, 1:], x.T[:, :-1])[0, 1]  # space x time
            rho_ave = (rho * weights).sum() / weights.sum()
            nstars[i, 0, 0, 0] = ntime * ((1 - rho_ave) / (1 + rho_ave))  # estimate

            # Get EOFs using covariance matrix on *shortest* dimension
            # NOTE: Eigenvalues are in *ascending* order, so get the last ones
            if x.shape[0] > x.shape[1]:
                # Get *temporal* covariance matrix since time dimension larger
                # Returns eigenvectors in columns
                eigrange = [nspace - neof, nspace - 1]  # eigenvalues to get
                covar = (xw.T @ xw) / ntime
                l, v = linalg.eigh(covar, eigvals=eigrange, eigvals_only=False)
                pc = xw @ v  # (time x space) x (space x neof) = (time x neof)

            else:
                # Get *spatial* dispersion matrix since space dimension longer
                # This time 'eigenvectors' are actually the pcs
                eigrange = [ntime - neof, ntime - 1]  # eigenvalues to get
                covar = (xw @ x.T) / nspace
                l, pc = linalg.eigh(covar, eigvals=eigrange, eigvals_only=False)

            # Store in big arrays
            # NOTE: We store projection of PC onto data to get 1 standard
            # deviation associated with the PC rather than actual eigenvector,
            # because eigenvector values may be damped by area weighting.
            pc = (pc - pc.mean(axis=0)) / pc.std(axis=0)  # standardize pcs
            proj = (x.T @ pc) / ntime  # (space by time) x (time by neof)
            pcs[i, :, :, 0] = pc.T[::-1, :]  # neof by time
            projs[i, :, 0, :] = proj.T[::-1, :]  # neof by space
            if percent:  # *percent explained* rather than total
                evals[i, :, 0, 0] = (100.0 * l[::-1] / np.trace(covar))
            else:
                evals[i, :, 0, 0] = l[::-1]  # actual eigenvalues

        # Replace context data with new dimension inserted on left side
        ctx.replace_data(pcs, projs, evals, nstars, insert_left=1)

    # Return data restored to original dimensionality
    return ctx.data


def eot(data, neof=5):  # noqa
    """
    Empirical orthogonal teleconnections.

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError


def reof(data, neof=5):  # noqa
    """
    Rotated empirical orthogonal functions, e.g. according to the "varimax"
    method. The EOFs will be rotated according only to the first `neof` EOFs.

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError


@quack._hist_metadata
@quant.while_dequantified(('=y', '=y'), 'count')
@docstring._snippet_manager(name='count')
def hist(bins, y, /, axis=0):
    """
    Get the histogram along axis `axis`.

    Parameters
    ----------
    bins : array-like
        The bin location.
    y : array-like
        The data.
    %(params_axisdim)s

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import climopy as climo
    >>> from climopy import ureg
    >>> state = np.random.RandomState(51423)
    >>> data = xr.DataArray(
    ...     state.rand(20, 1000) * ureg.m,
    ...     name='distance',
    ...     dims=('x', 'y'),
    ...     coords={'x': np.arange(20), 'y': np.arange(1000) * 0.1}
    ... )
    >>> bins = np.linspace(0, 1, 11) * ureg.m
    >>> hist = climo.hist(bins, data, axis=1)
    >>> bins
    <Quantity([0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ], 'meter')>
    >>> hist
    <xarray.DataArray 'count' (x: 20, distance: 10)>
    <Quantity([[100. 102. 101. 112.  99.  98.  93.  97.  84. 114.]
     [ 96.  94. 117.  81.  93. 111.  99.  92. 116. 101.]
     [116.  92. 101.  93. 100. 100. 101. 106.  95.  96.]
     [101. 113. 100.  96. 103. 112.  99.  85.  96.  95.]
     [102.  97.  85. 111.  94. 116. 101.  98.  94. 102.]
     [ 95. 112.  93. 105. 104.  87. 101. 103.  95. 105.]
     [103.  86.  98.  89. 110. 100. 101.  81. 132. 100.]
     [ 90.  98.  99. 130.  97. 106.  86.  97. 101.  96.]
     [ 95. 110.  96.  92.  88.  87. 118. 101. 112. 101.]
     [ 97.  85.  77. 102.  97. 119.  90. 106. 108. 119.]
     [ 87.  96.  95. 105.  91. 118. 109.  97.  99. 103.]
     [113.  99. 102.  97.  91.  97.  89. 110. 104.  98.]
     [100. 107. 110.  97.  85. 114. 104.  95.  97.  91.]
     [110. 102.  87.  98.  84.  99. 119.  92. 109. 100.]
     [ 95.  96. 101. 118. 103.  93.  89. 102.  90. 113.]
     [ 94.  87. 119. 102. 106. 100. 110. 108.  83.  91.]
     [ 98.  85.  96. 101. 101. 122.  85.  95. 111. 106.]
     [ 93. 111.  87.  95.  93. 103. 107. 111.  92. 108.]
     [ 86.  95.  89. 109.  90.  98. 119.  90. 116. 108.]
     [103. 100. 106.  87. 102.  88. 103. 121.  93.  97.]], 'count')>
    Coordinates:
      * x         (x) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
      * distance  (distance) float64 0.05 0.15 0.25 0.35 ... 0.65 0.75 0.85 0.95
    """
    if bins.ndim != 1:
        raise ValueError('Bins must be 1-dimensional.')

    with permute._PermuteContext(y, push_right=axis) as ctx:
        # Get flattened data
        y = ctx.data
        yhist = np.empty((y.shape[0], bins.size - 1))

        # Take histogram
        for k in range(y.shape[0]):
            yhist[k, :] = np.histogram(y[k, :], bins=bins)[0]

        # Replace data
        ctx.replace_data(yhist)

    # Return unflattened data
    return ctx.data


@quack._lls_metadata
@quant.while_dequantified(('=x', '=y'), ('=y / x', '=y / x', '=y'))
def slope(x, y, /, axis=0):
    """
    Get linear regression along axis, ignoring NaNs. Uses `~numpy.polyfit`.

    Parameters
    ----------
    x : array-like
        The *x* coordinates.
    y : array-like
        The *y* coordinates.
    axis : int, optional
        The regression axis.
    dim : str, optional
        *For `xarray.DataArray` input only*.
        The named regression dimension.

    Returns
    -------
    slope : array-like
        The slope estimates. The shape is the same as `y` but with
        dimension `axis` reduced to length 1.
    stderr : array-like
        The standard errors of the slope estimates. The shape is the
        same as `slope`.
    fit : array-like
        The reconstructed best-fit line. The shape is the same as `y`.

    Examples
    --------
    >>> import numpy as np
    >>> import climopy as climo
    >>> state = np.random.RandomState(51423)
    >>> x = np.arange(500)
    >>> y = state.rand(10, 500) + 0.1 * x
    >>> slope, stderr, fit = climo.slope(x, y, axis=1)
    >>> slope
    array([[0.10000399],
           [0.09997467],
           [0.09980544],
           [0.10004589],
           [0.10002195],
           [0.09996018],
           [0.10009204],
           [0.09992162],
           [0.10014288],
           [0.10011434]])
    """
    if x.ndim != 1 or x.size != y.shape[axis]:
        raise ValueError(
            f'Invalid x-shape {x.shape} for regression along axis {axis} '
            f'of y-shape {y.shape}.'
        )

    with permute._PermuteContext(y, push_right=axis) as ctx:
        # Get regression coefficients. Flattened data is shape (K, N)
        # where N is regression dimension. Permute to (N, K) then back again.
        # N gets replaced with length-2 dimension (slope, offset).
        y = ctx.data
        params, covar = np.polyfit(x, y.T, deg=1, cov=True)
        params = np.fliplr(params.T)  # flip to (offset column 0, slope column 1)

        # Get best-fit line and slope
        fit = params[:, :1] + x * params[:, 1:]
        slope = params[:, 1:]

        # Get standard error
        # See Dave's paper (TODO: add citation)
        n = y.shape[1]
        resid = y - fit  # residual
        mean = resid.mean(axis=1, keepdims=True)
        var = resid.var(axis=1, keepdims=True)
        rho = np.sum(
            (resid[:, 1:] - mean) * (resid[:, :-1] - mean), axis=1, keepdims=True,
        ) / ((n - 1) * var)  # correlation factor
        scale = (n - 2) / (n * ((1 - rho) / (1 + rho)) - 2)  # scale factor
        stderr = np.sqrt(scale * covar[0, 0, :, None])

        # Replace context data
        ctx.replace_data(slope, stderr, fit)

    return ctx.data


@quack._lls_metadata
@quant.while_dequantified(('=t', ''), ('=t', '=t', ''))
def timescale(
    dt, a, /, maxlag=None, imaxlag=None, maxlag_fit=None, imaxlag_fit=None, axis=0
):
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

    Approach 2 is used if `axis` is singleton or the data is scalar. In these
    cases, the data is assumed to represent just the lag-1 autocorrelation.

    Parameters
    ----------
    dt : float or array-like, optional
        The timestep or time series (from which the timestep is inferred).
    a : array-like
        The autocorrelation spectra.
    maxlag : float, optional
        The maximum time lag to include in the red noise fit.
    imaxlag : int, optional
        As with `maxlag` but specifies the index instead of the physical time.
    maxlag_fit : float, optional
        The maximum time lag to include in the output pure red noise spectrum.
    imaxlag_fit : int, optional
        As with `maxlag` but specifies the index instead of the physical time.
    axis : int, optional
        The lag axis. Each slice along this axis should represent an
        autocorrelation spectrum generated with `corr`.

        * If `axis` is singleton, the data are assumed to be lag-1 autocorrelations
          and the timescale is computed from the lag-1 pure red noise formula.
        * If `axis` is non-singleton, the timescale is estimated from a least-squares
          curve fit to a pure red noise spectrum up to `maxlag`.

    dim : str, optional
        *For `xarray.DataArray` input only*.
        The named lag dimension.

    Returns
    -------
    tau : array-like
        The autocorrelation timescales. The shape is the same as `data`
        but with `axis` reduced to length ``1``.
    sigma : array-like
        The standard errors. If the timescale was inferred using the lag-1
        equation, this is an array of zeros. The shape is the same as `tau`.
    fit : array-like
        The best fit autocorrelation spectrum. The shape is the same as `data`
        but with `axis` of length `nlag`.

    Examples
    --------
    >>> import numpy as np
    >>> import climopy as climo
    >>> state = np.random.RandomState(51423)
    >>> data = climo.rednoise(0.8, 500, 10, state=state)
    >>> lag, auto = climo.autocorr(1, data, axis=0, maxlag=50)
    >>> taus, sigmas, fit = climo.timescale(lag, auto, axis=0)
    >>> taus
    array([[5.97691453, 4.29275329, 4.91997185, 4.87781027, 3.46404331,
            4.23444888, 4.91852921, 4.39283164, 4.79466674, 3.81250855]])

    See Also
    --------
    corr, rednoise
    """
    # Initial stuff
    dt = quack._as_step(dt)
    curve_func = lambda t, tau: np.exp(-t * dt / tau)  # noqa: E731

    # Parse arguments
    if maxlag is not None and imaxlag is not None:
        raise ValueError(f'Conflicting kwargs {maxlag=} and {imaxlag=}.')
    if maxlag_fit is not None and imaxlag_fit is not None:
        raise ValueError(f'Conflicting kwargs {maxlag_fit=} and {imaxlag_fit=}.')
    if any(_ is not None and not np.isclose(_ % dt, 0) for _ in (maxlag, maxlag_fit)):
        raise ValueError(f'Lag time must be divisible by timestep {dt}.')
    if maxlag is not None:
        imaxlag = np.round(maxlag / dt).astype(int)
    if maxlag_fit is not None:
        imaxlag_fit = np.round(maxlag_fit / dt).astype(int)

    with permute._PermuteContext(a, push_right=axis) as ctx:
        # Set defaults
        a = ctx.data
        nlag = a.shape[1] - 1  # not including 0-lag entry
        nextra = a.shape[0]
        imaxlag = max(1, nlag) if imaxlag is None else min(imaxlag, nlag)
        imaxlag_fit = imaxlag if imaxlag_fit is None else imaxlag_fit  # can be anything
        lags = np.arange(0, imaxlag + 1)  # lags for the curve fit
        lags_fit = np.arange(0, imaxlag_fit + 1)

        # Iterate over dimensions
        taus = np.empty((nextra, 1))
        sigmas = np.empty((nextra, 1))
        fit = np.empty((nextra, imaxlag_fit + 1))
        for i in range(nextra):
            if imaxlag <= 1:
                # Scalar function
                tau = -dt / np.log(a[i, -1])
                sigma = 0
            else:
                # Curve fit
                tau, sigma = optimize.curve_fit(curve_func, lags, a[i, :])
                sigma = np.sqrt(np.diag(sigma))
                tau, sigma = tau[0], sigma[0]
            # Timescales, uncertainty and the curve itself
            taus[i, 0] = tau  # just store the timescale
            sigmas[i, 0] = sigma
            fit[i, :] = np.exp(-dt * lags_fit / tau)  # best-fit spectrum

        # Replace context data
        ctx.replace_data(taus, sigmas, fit)

    # Return permuted data
    return ctx.data


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
    .. bibliography:: ../bibs/fdr.bib
    """
    percentiles = np.asarray(percentiles)
    pvals = list(percentiles.flat)  # want in decimal
    pvals = sorted(2 * min(pval, 1 - pval) for pval in pvals)
    ptest = [alpha * i / len(pvals) for i in range(len(pvals))]
    ppick = max(pv for pv, pt in zip(pvals, ptest) if pv <= pt) / 2
    mask = (percentiles <= ppick) | (percentiles >= (1 - ppick))
    return percentiles[mask]
