#!/usr/bin/env python3
"""
Tools for objective analysis-related tasks.
Most functions should work with arbitrary array shapes.
Note: Convention throughout module (consistent with common conventions
in atmospheric science) is to use *linear* wave properties, i.e. the
wavelength in <units> per 2pi rad, and wavenumber 2pi rad per <units>.
"""
import os
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import scipy.linalg as linalg
import scipy.optimize as optimize
from . import const # as its own module
from .arraytools import *

#------------------------------------------------------------------------------
# Algebra
#------------------------------------------------------------------------------
def solve(poly):
    """
    Find real-valued root for polynomial with input coefficients; pretty simple.
    Format of input array p: p[0]*x^n + p[1]*x^n-1 + ... + p[n-1]*x + p[n]
    """
    # Just use numpy's roots function, and filter results.
    r = np.roots(poly) # input polynomial; returns ndarray
    r = r[np.imag(r)==0].astype(np.float32) # take only real-valued ones
    return r

#------------------------------------------------------------------------------#
# Random distributions
#------------------------------------------------------------------------------#
def gaussian(N=1000, mean=0, stdev=None, sigma=1):
    """
    Returns sample points on Gaussian curve.
    """
    sigma = stdev if stdev is not None else sigma
    norm = stats.norm(loc=mean, scale=sigma)
    x = np.linspace(norm.ppf(0.0001), norm.ppf(0.9999), N) # get x through percentile range
    pdf = norm.pdf(x, loc=mean, scale=sigma)
    return x, pdf

#------------------------------------------------------------------------------
# Artificial data
#------------------------------------------------------------------------------
def rednoise(a, ntime, samples=1, mean=0, stdev=1, nested=False):
    """
    Creates artificial red noise time series, i.e. a weighted sum
    of random perturbations.

    Equation is: x(t) = a*x(t-dt) + b*eps(t)
    where a is the lag-1 autocorrelation and b is a scaling term.

    Parameters
    ---------
    a : scalar
        autocorrelation
    ntime : integer
        number of timesteps
    samples : integer or iterable of integers
        shape of output array; final result will be ntime by this shape

    Notes
    -----
    * Output will have shape ntime by nsamples.
    * Enforce that the first timestep always equals the 'starting' position.
    * Use 'nested' flag to control which algorithm to use.
    """
    # Initial stuff
    ntime -= 1 # exclude the initial timestep
    samples = np.atleast_1d(samples)
    data = np.empty((ntime+1,*samples)) # user can make N-D array
    b = (1-a**2)**0.5  # from OA class

    # Nested loop
    data, shape = trail_flatten(data)
    data[0,:] = 0 # initialize
    for i in range(data.shape[-1]):
        eps = np.random.normal(loc=0, scale=1, size=ntime)
        # data[1:,i] = a*data[:-1,i] + b*eps[:-1] # won't work because next state function of previous state
        for t in range(1,ntime+1):
            data[t,i] = a*data[t-1,i] + b*eps[t-1]

    # Seriously overkill
    # init = np.atleast_1d(init)
    # if hasattr(init,'__iter__') and len(init)==2:
    #     data = data + np.random.uniform(*init, size=data.shape[1]) # overkill
    # if data.shape[-1]!=init.shape[-1] and init.size!=1:
    #     raise ValueError('Length of vector of initial positions must equal number of sample time series.')
    # data = data + init # scalar or iterable (in which case, right-broadcasting takes place)
    # Trying to be fancy, just turned out fucking way slower
    # aseries = b*np.array([a**(ntime-i) for i in range(1,ntime+1)])
    # for i in range(samples):
    #     eps = np.random.normal(loc=0, scale=1, size=ntime)
    #     vals = eps[:,None] @ aseries[None,:] # matrix for doing math on
    #     data[1:,i] = [np.trace(vals,ntime-i) for i in range(1,ntime+1)]

    # Return
    data = trail_unflatten(data, shape)
    if len(samples)==1 and samples[0]==1:
        data = data.squeeze()
    return mean + stdev*data # rescale to have specified stdeviation/mean

def waves(x, wavenums=None, wavelens=None, phase=None):
    """
    Compose array of sine waves.
    Useful for testing performance of filters.

    Parameters
    ----------
    x : scalar or iterable
        * if scalar, 'x' is np.arange(0,x)
        * if iterable, can be n-dimensional, and will calculate sine from coordinates on every dimension
    wavelens :
        wavelengths for sine function (use either this or wavenums)
    wavenums :
        wavenumbers for sine function (use either this or wavelens)

    Returns
    -------
    data :
        data composed of waves.

    Notes
    -----
    'x' will always be normalized so that wavelength is with reference to
    first step. This make sense because when working with filters, almost
    always need to use units corresponding to axis.
    """
    # Wavelengths
    if wavenums is None and wavelens is None:
        raise ValueError('Must declare wavenums or wavelengths.')
    elif wavelens is not None:
        # dx = x[1] - x[0]
        wavenums = 1.0/np.atleast_1d(wavelens)
    wavenums = np.atleast_1d(wavenums)
    if not hasattr(x, '__iter__'):
        x = np.arange(x)
    data = np.zeros(x.shape) # user can make N-D array
    # Get waves
    if phase is None:
        phis = np.random.uniform(0,2*np.pi,len(wavenums))
    else:
        phis = phase*np.ones([len(wavenums)])
    for wavenum,phi in zip(wavenums,phis):
        data += np.sin(2*np.pi*wavenum*x + phi)
    return data

#------------------------------------------------------------------------------#
# Least squares fits
#------------------------------------------------------------------------------#
def linefit(*args, axis=-1, build=False, stderr=False):
    """
    Get linear regression along axis, ignoring NaNs. Uses np.polyfit.

    Parameters
    ----------
    y :
        assumed 'x' is 0 to len(y)-1
    x, y :
        arbitrary 'x', must monotonically increase
    axis :
        regression axis
    build :
        whether to replace regression axis with scalar slope, or
        reconstructed best-fit line.
    stderr :
        if not doing 'build', whether to add standard error on
        the slope in index 1 of regressoin axis.

    Returns
    -------
    y : regression params on axis 'axis', with offset at index 0,
        slope at index 1.
    """
    # Perform regression
    # Polyfit can perform regressions on data with series in columns,
    # separate samples along rows.
    if len(args)==1:
        y, = args
        x  = np.arange(len(y))
    elif len(args)==2:
        x, y = args
    else:
        raise ValueError("Must input 'x' or 'x, y'.")
    y, shape = lead_flatten(permute(y, axis))
    z, v = np.polyfit(x, y.T, deg=1, cov=True)
    z = np.fliplr(z.T) # put a first, b next
    # Prepare output
    if build:
        # Repalce regression dimension with best-fit line
        z = z[:,:1] + x*z[:,1:]
    elif stderr:
        # Replace the regression dimension with (slope, standard error)
        n = y.shape[1]
        s = np.array(z.shape[:1])
        resid = y - (z[:,:1] + x*z[:,1:]) # residual
        mean = resid.mean(axis=1)
        var  = resid.var(axis=1)
        rho  = np.sum((resid[:,1:]-mean[:,None])*(resid[:,:-1]-mean[:,None]), axis=1)/((n-1)*var)
        scale = (n-2)/(n*((1-rho)/(1+rho))-2)
        s = np.sqrt(v[0,0,:]*scale)
        z[:,0] = s # the standard error
        z = np.fliplr(z) # the array is <slope, standard error>
        shape[-1] = 2
    else:
        # Replace regression dimension with singleton (slope)
        z = z[:,1:]
        shape[-1] = 1 # axis now occupied by the slope
    return unpermute(lead_unflatten(z, shape), axis)

def rednoisefit(data, nlag=None, axis=-1, lag1=False, series=False, verbose=False):
    """
    Calculates a best-fit red noise autocorrelation spectrum.
    Goes up to nlag-timestep autocorrelation.

    Parameters
    ----------
    data :
        the input data (this function computes necessary correlation coeffs).
    nlag : integer
        number of lags to use for fit
    lag1 : bool
        if True, just return the lag-1 best-fit
    series : bool
        return the red noise fit spectrum (True), or just the
        associted e-folding time scale (False)?

    Returns
    -------
    spectrum :
        the autocorrelation spectrum out to nlag lags.
    """
    # Initial stuff
    if nlag is None:
        raise ValueError(f"Must declare \"nlag\" argument; number of points to use for fit.")
    data, shape = lead_flatten(permute(data, axis))
    # if len(time)!=data.shape[-1]:
    #     raise ValueError(f"Got {len(time)} time values, but {data.shape[-1]} timesteps for data.")
    # First get the autocorrelation spectrum, and flatten leading dimensions
    # Dimensions will need to be flat because we gotta loop through each 'time series' and get curve fits.
    # TODO: Add units support, i.e. a 'dx'
    # time = time[:nlag+1] # for later
    _, autocorrs = corr(data, nlag=nlag, axis=-1, verbose=verbose)
    # Next iterate over the flattened dimensions, and perform successive curve fits
    ndim = data.shape[-1] if series else 1
    output = np.empty((autocorrs.shape[0], ndim))
    time = np.arange(autocorrs.shape[-1]) # time series for curve fit
    for i in range(autocorrs.shape[0]): # iterate along first dimension; each row is an autocorrelation spectrum
        if lag1:
            # dt = time[1]-time[0] # use this for time diff
            p = [-1/np.log(autocorrs[i,1])] # -ve inverse natural log of lag-1 autocorrelation
        else:
            p, _ = optimize.curve_fit(lambda t,tau: np.exp(-t/tau), time, autocorrs[i,:])
        if series:
            output[i,:] = np.exp(-np.arange(ndim)/p[0]) # return the best-fit red noise spectrum
        else:
            output[i,0] = p[0] # add just the timescale
    # shape = (*data.shape[:-1], ndim) # not sure what this was for
    return unpermute(lead_unflatten(output, shape), axis)

#------------------------------------------------------------------------------#
# Correlation analysis
#------------------------------------------------------------------------------#
def corr(data1, data2=None, dx=1, nlag=None, lag=None,
         verbose=False, axis=0, _normalize=True):
    """
    Gets the correlation spectrum at successive lags. For autocorrelation,
    pass only a single ndarray.

    Parameters
    ----------
    data1 :
        The input data.
    data2 : (optional)
        The input data with which we compare data
    nlag :
        Get correlation at multiple lags ('n' is number after the 0-lag).
    lag :
        Get correlation at just this lag.
    _normalize :
        Used for autocovar wrapper. generally user shouldn't touch this.

    Returns
    -------
    autocorrs :
        the autocorrelations as a function of lag.

    Notes
    -----
    * Uses following estimator: ((n-k)*sigma^2)^-1 * sum_i^(n-k)[X_t - mu][X_t+k - mu]
    See: https://en.wikipedia.org/wiki/Autocorrelation#Estimation
    * By default, includes lag-zero values; if user just wants single lag
    will, however, throw out those values.
    """
    # Preparation, and stdev/means
    data1 = np.array(data1)
    if data2 is None:
        autocorr = True
        data2 = data1.copy()
    else:
        autocorr = False
        data2 = np.array(data2)
        if data1.shape != data2.shape:
            raise ValueError(f'Data 1 shape {data1.shape} and Data 2 shape {data2.shape} do not match.')
    naxis = data1.shape[axis] # length
    if (nlag is None and lag is None) or (nlag is not None and lag is not None):
        raise ValueError(f"Must specify *either* a lag <lag> or range of lags <nlags>.")
    if nlag is not None and nlag>=naxis/2:
        raise ValueError(f"Lag {nlag} must be greater than axis length {naxis}.")
    if verbose:
        if nlag is None:
            print(f"Calculating lag-{lag} autocorrelation.")
        else:
            print(f"Calculating autocorrelation spectrum up to lag {nlag} for axis length {naxis}.")
    # Standardize maybe
    var1 = var2 = 1
    data1 = permute(data1, axis)
    data2 = permute(data2, axis)
    mean1 = data1.mean(axis=-1, keepdims=True) # keepdims for broadcasting in (data1 minus mean)
    mean2 = data2.mean(axis=-1, keepdims=True)
    if _normalize:
        std1 = data1.std(axis=-1, keepdims=False) # this is divided by the summation term, so should have annihilated axis
        std2 = data2.std(axis=-1, keepdims=False)
    else:
        std1 = std2 = 1 # use for covariance

    # Trivial autocorrelation done, just fill with ones
    if nlag is None and lag==0:
        # autocorrs = np.ones((*data1.shape[:-1], 1))
        return unpermute(np.sum((data1 - mean1)*(data2 - mean2)) / (naxis*std1*std2), axis)
    # Autocorrelation on specific lag
    elif nlag is None:
        lag = np.round(lag*dx).astype(int)
        return unpermute(np.sum((data1[...,:-lag] - mean1)*(data2[...,lag:] - mean2), axis=-1, keepdims=True)
            / ((naxis - lag)*std1*std2), axis)
    # Autocorrelation up to n timestep-lags after 0-correlation
    else:
        # First figure out lags
        # Negative lag means data2 leads data1 (e.g. z corr m, left-hand side
        # is m leads z).
        nlag = np.round(nlag/dx).astype(int) # e.g. 20 day lag, at synoptic timesteps
        if not autocorr:
            # print('Getting autocorr-sided correlation.')
            n = nlag*2 + 1 # the center point, and both sides
            lags = range(-nlag, nlag+1)
        else:
            # print('Getting one-sided correlation.')
            n = nlag + 1
            lags = range(0, nlag+1)
        # Get correlation
        autocorrs = np.empty((*data1.shape[:-1], n)) # will include the zero-lag autocorrelation
        for i,lag in enumerate(lags):
            if lag==0:
                prod = (data1 - mean1)*(data2 - mean2)
            elif lag<0: # input 1 *trails* input 2
                prod = (data1[...,-lag:] - mean1)*(data2[...,:lag] - mean2)
            else:
                prod = (data1[...,:-lag] - mean1)*(data2[...,lag:] - mean2)
            autocorrs[...,i] = prod.sum(axis=-1) / ((naxis - lag)*std1*std2)
        return np.array(lags)*dx, unpermute(autocorrs, axis)

def covar(*args, **kwargs):
    """
    As above, but gets the covariance.
    """
    return corr(*args, **kwargs, _normalize=False)

#------------------------------------------------------------------------------#
# Empirical orthogonal functions and related decomps
#------------------------------------------------------------------------------#
def eof(data, neof=5, record=-2, space=-1,
        percent=True,
        weights=None,
        debug=False, normalize=False):
    """
    Calculates the temporal EOFs, using the scipy algorithm for Hermetian (or
    real symmetric) matrices. This version allows calculating just 'n'
    most important ones, so should be faster.

    Parameters
    ----------
        data :
            Data of arbitrary shape.
        neof :
            Number of eigenvalues we want.
        percent : bool
            Whether to return raw eigenvalue(s), or *percent* of variance
            explained by eigenvalue(s).
        record :
            Axis used as 'record' dimension -- should only be 1.
        space :
            Axes used as 'space' dimension -- can be many.
        weights :
            Area/mass weights; must be broadcastable on multiplication with 'data'
            weights will be normalized prior to application.
    """
    # First query array shapes and stuff
    m_dims = np.atleast_1d(record)
    n_dims = np.atleast_1d(space)
    if m_dims.size>1:
        raise ValueError('Record dimension must lie on only one axis.')
    m_dims[m_dims<0] = data.ndim + m_dims[m_dims<0]
    n_dims[n_dims<0] = data.ndim + n_dims[n_dims<0]
    if any(i<0 or i>=data.ndim for i in [*m_dims, * n_dims]):
        raise ValueError('Invalid dimensions.')
    space_after  = all(i>m_dims[0] for i in n_dims)
    if not space_after and not all(i<m_dims[0] for i in n_dims):
        raise ValueError('Reorder your data! Need space dimensions to come before/after time dimension.')

    # Remove the mean and optionally standardize the data
    data = data - data.mean(axis=m_dims[0], keepdims=True) # remove mean
    if normalize:
        data = data / data.stdev(axis=m_dims[0], keepdims=True) # optionally standardize, usually not wanted for annular mode stuff
    # Next apply weights
    m = np.prod([data.shape[i] for i in n_dims]) # number samples/timesteps
    n = np.prod([data.shape[i] for i in m_dims]) # number space locations
    if weights is None:
        weights = 1
    weights = np.atleast_1d(weights) # want numpy array
    weights = weights/weights.mean() # so does not affect amplitude
    try:
        if m > n: # more sampling than space dimensions
            data  = data*np.sqrt(weights)
            dataw = data
        else: # more space than sampling dimensions
            dataw = data*weights
    except ValueError:
        raise ValueError(f'Dimensionality of weights {weights.shape} incompatible with dimensionality of space dimensions {data.shape}!')

    # Turn matrix into *record* by *space*, or 'M' by 'N'
    # 1) Move record dimension to right
    data  = permute(data, m_dims[0], -1)
    dataw = permute(dataw, m_dims[0], -1)
    # 2) successively move space dimensions to far right, proceeding from the
    # rightmost space dimension to leftmost space dimension so axis numbers
    # do not change
    dims = n_dims.copy()
    if space_after: # time was before, so new space dims changed
        dims -= 1
    dims = np.sort(dims)[::-1]
    for axis in dims:
        data = permute(data, axis, -1)
        dataw = permute(dataw, axis, -1)
    # Only flatten after apply weights (e.g. if have level and latitude dimensoins)
    shape_trail = data.shape[-n_dims.size:]
    data,  _ = trail_flatten(data,  n_dims.size)
    dataw, _ = trail_flatten(dataw, n_dims.size)
    shape_lead = data.shape[:-2]
    data,  _ = lead_flatten(data,  data.ndim-2)
    dataw, _ = lead_flatten(dataw, dataw.ndim-2)
    # Prepare output
    # Dimensions will be extraneous by eofs by time by space
    if data.ndim!=3:
        raise ValueError(f"Shit's on fire yo.")
    nextra, m, n = data.shape[0], data.shape[1], data.shape[2] # n extra, record, and space
    pcs   = np.empty((nextra, neof, m, 1))
    projs = np.empty((nextra, neof, 1, n))
    evals = np.empty((nextra, neof, 1, 1))
    nstar = np.empty((nextra, 1,    1, 1))

    # Get EOFs and PCs and stuff
    for i in range(data.shape[0]):
        # Initial
        x = data[i,:,:] # array will be sampling by space
        xw = dataw[i,:,:]
        # Get reduced degrees of freedom for spatial eigenvalues
        # TODO: Fix the weight projection below
        rho = np.corrcoef(x.T[:,1:], x.T[:,:-1])[0,1] # must be (space x time)
        rho_ave = (rho*weights).sum()/weights.sum()
        nstar[i,0,0,0] = m*((1 - rho_ave)/(1 + rho_ave)) # simple degrees of freedom estimation
        # Get EOFs using covariance matrix on *shortest* dimension
        if x.shape[0] > x.shape[1]:
            # Get *temporal* covariance matrix since time dimension larger
            eigrange = [n-neof, n-1] # eigenvalues to get
            covar = (xw.T @ xw)/m
            l, v = linalg.eigh(covar, eigvals=eigrange, eigvals_only=False) # returns eigenvectors in *columns*
            Z = xw @ v # i.e. multiply (time x space) by (space x neof), get (time x neof)
            z = (Z - Z.mean(axis=0))/Z.std(axis=0) # standardize pcs
            p = x.T @ z/m # i.e. multiply (space x time) by (time x neof), get (space x neof)
        else:
            # Get *spatial* dispersion matrix since space dimension longer
            # This time 'eigenvectors' are actually the pcs
            eigrange = [m-neof, m-1] # eigenvalues to get
            covar = (xw @ x.T)/n
            l, Z = linalg.eigh(covar, eigvals=eigrange, eigvals_only=False)
            z = (Z - Z.mean(axis=0))/Z.std(axis=0) # standardize pcs
            p = (x.T @ z)/m # i.e. multiply (space x time) by (time by neof), get (space x neof)
        # Store in big arrays
        pcs[i,:,:,0] = z.T[::-1,:] # neof by time
        projs[i,:,0,:] = p.T[::-1,:] # neof by space
        if percent:
            evals[i,:,0,0] = 100.0*l[::-1]/np.trace(covar) # percent explained
        else:
            evals[i,:,0,0] = l[::-1] # neof
        # # Sort
        # idx = L.argsort()[::-1]
        # L, Z = L[idx], Z[:,idx]

    # Return along the correct dimension
    # The 'lead's were *extraneous* dimensions; we got EOFs along them
    nlead = len(shape_lead) # expand back to original; leave space for EOFs
    pcs   = lead_unflatten(pcs,   [*shape_lead, neof, m, 1], nlead)
    projs = lead_unflatten(projs, [*shape_lead, neof, 1, n], nlead)
    evals = lead_unflatten(evals, [*shape_lead, neof, 1, 1], nlead)
    nstar = lead_unflatten(nstar, [*shape_lead, 1,    1, 1], nlead)
    # The 'trail's were *spatial* dimensions, which were allowed to be more than 1D
    ntrail = len(shape_trail)
    flat_trail = [1]*len(shape_trail)
    pcs   = trail_unflatten(pcs,   [*shape_lead, neof, m, *flat_trail],  ntrail)
    projs = trail_unflatten(projs, [*shape_lead, neof, 1, *shape_trail], ntrail)
    evals = trail_unflatten(evals, [*shape_lead, neof, 1, *flat_trail],  ntrail)
    nstar = trail_unflatten(nstar, [*shape_lead, 1,    1, *flat_trail],  ntrail)
    # Permute 'eof' dimension onto the start (note we had to put it between
    # extraneous dimensions and time/space dimensions so we could perform
    # the above unflatten moves)
    init = len(shape_lead) # eofs are on the one *after* those leading dimensions
    pcs   = np.moveaxis(pcs, init, 0)
    projs = np.moveaxis(projs, init, 0)
    evals = np.moveaxis(evals, init, 0)
    nstar = np.moveaxis(nstar, init, 0)
    # Finally, permute stuff on right-hand dimensions back to original positions
    # 1) The spatial dimensions. This time proceed from left-to-right so
    # axis numbers onto which we permute are correct.
    dims = n_dims.copy()
    dims += 1 # account for EOF
    if space_after:
        dims -= 1 # the dims are *actually* one slot to left, since time was not put back yet
    dims = np.sort(dims)
    for axis in dims:
        pcs = unpermute(pcs, axis)
        projs = unpermute(projs, axis)
        evals = unpermute(evals, axis)
        nstar = unpermute(nstar, axis)
    pcs = unpermute(pcs, m_dims[0]+1)
    projs = unpermute(projs, m_dims[0]+1)
    evals = unpermute(evals, m_dims[0]+1)
    nstar = unpermute(nstar, m_dims[0]+1)
    # And return everything! Beautiful!
    return evals, nstar, projs, pcs

def eot(data, neof=5):
    """
    EOTs, used in some other research contexts.
    """
    raise NotImplementedError('Not yet implemented.')

def reof(data, neof=5):
    """
    Rotated EOFs, e.g. according to Varimax.
    The EOFs will be rotated according only to the first N signals.
    """
    raise NotImplementedError('Not yet implemented.')

#------------------------------------------------------------------------------#
# Application of filters
# Mimick lfilter, but in pure python and only for non-recursive
# real-space filters, and without shifting the data in x (i.e.
# filling the trails of the data with NaNs)
#------------------------------------------------------------------------------#
def rolling(x, w, axis=-1, btype='lowpass',
                  pad=True, padvalue=np.nan, **kwargs):
    """
    Implementation is similar to scipy.signal.lfilter. Note for non-recursive
    (i.e. windowing) filters, the 'a' vector is just 1, followed by zeros.
    Read this: https://stackoverflow.com/a/4947453/4970632

    Generates rolling numpy window along final axis; can then operate with
    functions like polyfit or mean along the new last axis of output.

    Just creates *view* of original array, without duplicating data, so no worries
    about efficiency.

    Parameters
    ----------
    x :
        data, and we roll along axis 'axis'.
    w : int or iterable
        boxcar window length, or custom weights
    pad : bool
        whether to pad the edges of axis back to original size
    padvalue : float
        what to pad with (default np.nan)
    btype : string
        whether to apply lowpass, highpass, or bandpass
    **kwargs :
        remaining kwargs passed to windowing function

    Returns
    -------
    x :
        data windowed along arbitrary dimension

    Notes
    -----
    * For 1-D data numpy 'convolve' would be appropriate, problem is 'concolve'
    doesn't take multidimensional input! Likely that the source code employs
    something similar to what I do below anyway.
    * If x has odd number of obs along axis, result will have last element
    trimmed. Just like filter().
    * Will generate a new axis in the -1 position that is a running representation
    of value in axis numver <axis>.
    * Strides are apparently the 'number of bytes' one has to skip in memory
    to move to next position *on the given axis*. For example, a 5 by 5
    array of 64bit (8byte) values will have array.strides == (40,8).
    * Should consider using swapaxes instead of these permute and unpermute
    functions, might be simpler.
    """
    # Roll axis, reshape, and get generate running dimension
    n_orig = x.shape[axis]
    if axis<0:
        axis = x.ndim + axis # e.g. if 3 dims, and want to axis dim -1, this is dim number 2
    x = permute(x, axis)
    # Determine weights
    if type(w) is str:
        raise NotImplementedError("Need to allow string 'w' argument, e.g. w='Lanczos'")
    w = np.atleast_1d(w)
    if len(w)==1:
        # Boxcar window
        nw = w[0]
        w = 1/nw
    else:
        # Arbitrary windowing function
        # TODO: Add windowing functions!
        nw = len(w)
    # Manipulate array
    shape = x.shape[:-1] + (x.shape[-1]-(nw-1), nw)
    strides = [*x.strides, x.strides[-1]] # repeat striding on end
    x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    # Next 'put back' axis, keep axis=1 as
    # the 'rolling' dimension which can be averaged by arbitrary weights.
    x = unpermute(x, axis, select=-2) # want to 'put back' axis -2;
    # Finally take the weighted average
    # Note numpy will broadcast from right, so weights can be scalar or not
    # print(x.min(), x.max(), x.mean())
    x = (x*w).sum(axis=-1) # broadcasts to right
    # print(x.min(), x.max(), x.mean())
    # Optionally fill the rows taken up
    if not pad:
        return x
    n_new   = x.shape[axis]
    n_left  = (n_orig-n_new)//2
    n_right = n_orig - n_new - n_left
    if n_left!=n_right:
        print('Warning: Data shifted left by one.')
    d_left  = padvalue*np.ones((*x.shape[:axis], n_left, *x.shape[axis+1:]))
    d_right = padvalue*np.ones((*x.shape[:axis], n_right, *x.shape[axis+1:]))
    x = np.concatenate((d_left, x, d_right), axis=axis)
    return x

def running(*args, **kwargs):
    """
    Alias for climpy.rolling().
    """
    return rolling(*args, **kwargs)

def filter(x, b, a=1, n=1, axis=-1, fix=True, pad=True, fixvalue=np.nan):
    """
    Apply scipy.signal.lfilter to data. By default this does *not* pad
    ends of data. May keep it this way.

    Parameters
    ----------
    x :
        data to be filtered
    b :
        b coefficients (non-recursive component)
    a :
        scale factor in index 0, followed by a coefficients (recursive component)
    n :
        number of times to filter data (will go forward-->backward-->forward...)
    axis :
        axis along which we filter data
    fix :
        whether to (a) trim leading part of axis by number of a/b coefficients
        and (b) fill trimmed values with NaNs; will also attempt to *re-center*
        the data if a net-forward (e.g. f, fbf, fbfbf, ...) filtering was
        performed (this works for non-recursive filters only)

    Output
    ------
    y :
        filtered data
    Notes:
      * Consider adding **empirical method for trimming either side of recursive
        filter that trims up to where impulse response is negligible**
      * If x has odd number of obs along axis, lfilter will trim
        the last one. Just like rolling().
      * For non-recursive time-space filters, 'a' should just be
        set to 1. The 'a' vector contains (index 0) the scalar use to normalize 'b'
        coefficients, and (index 1,2,...) the coefficients on 'y' in the filtering
        conversion from 'x' to 'y': so, 1 implies array of [1, 0, 0...], implies
        non-recursive.
    """
    # Apply filter 'n' times to each sample
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    n_half = (max(len(a), len(b))-1)//2
    if axis<0:
        axis = x.ndim + axis # necessary for concatenate below
    x, shape = lead_flatten(permute(x, axis))
    y = x.copy() # then can filter multiple times
    ym = y.mean(axis=1, keepdims=True)
    y = y-ym # remove mean
    for i in range(y.shape[0]): # iterate through samples
        for j in range(n): # applications
            step = 1 if j%2==0 else -1 # forward-backward application
            y[i,::step] = signal.lfilter(b, a, y[i,::step], axis=-1)
    y = y+ym # add mean back in
    # Fancy manipulation
    if fix:
        # Capture component that (for non-recursive filter) doesn't include datapoints with clipped edges
        # Forward-backward runs, so filtered data is in correct position w.r.t. x
        # e.g. if n is 2, we cut off the (len(b)-1) from each side
        n_2sides = (n//2)*2*n_half
        # Net forward run, so filtered data is shifted right by n_half
        # Also have to trim data on both sides if it's foward-->backward-->forward e.g.
        n_left = int((n%2)==1)*2*n_half
        # Determine part that 'sees' all coefficients
        if n_2sides==0:
            y = y[:,n_left:]
        else:
            y = y[:,(n_2sides + n_left):(-n_2sides)]
        # Optionally pad with a 'fill value' (usually NaN)
        if pad:
            y_left  = fixvalue*np.ones((y.shape[0], n_2sides+n_left//2))
            y_right = fixvalue*np.ones((y.shape[0], n_2sides+n_left//2))
            y = np.concatenate((y_left, y, y_right), axis=-1)
    # Return
    y = unpermute(lead_unflatten(y, shape), axis)
    return y

def response(dx, b, a=1, n=1000, simple=False):
    """
    Calculate response function given the a and b coefficients for some
    analog filter. Note we *need to make the exponent frequencies into
    rad/physical units* for results to make sense.

    Dennis Notes: https://atmos.washington.edu/~dennis/552_Notes_ftp.html

    Formula
    -------
                jw               -jw            -jmw
        jw  B(e)    b[0] + b[1]e + .... + b[m]e
        H(e) = ---- = ------------------------------------
                jw               -jw            -jnw
            A(e)    a[0] + a[1]e + .... + a[n]e
    and below we calculate simultaneously for vector of input omegas.
    """
    # Simple calculation given 'b' coefficients, from Libby's notes
    if simple:
        # Initial stuff
        if not (a==1 or a[0]==1):
            raise ValueError('Cannot yet manually calculate response function for recursive filter.')
        if len(b)%2==0:
            raise ValueError('Filter coefficient number should be odd, symmetric about a central value.')
        nb = len(b)
        C0 = b[nb//2]
        Ck = b[nb//2+1:] # should be symmetric; need to flip if choose leading side instead of trailing
        tau = np.arange(1,nb//2+1) # lag time, up to nb+1
        x = x*2*np.pi*dx # from cycles/unit --> rad/unit --> rad/timestep
        # Calculate and return
        y = C0 + 2*np.sum(Ck[None,:]*np.cos(tau[None,:]*x[:,None]), axis=1)
    # More complex freqz filter, generalized for arbitrary recursive filters,
    # with extra allowance for working in physical (non-timestep) space
    else:
        # Simply pass to freqz
        a = np.atleast_1d(a)
        x = np.linspace(0,np.pi,n)
        if len(a)>1: # e.g. butterworth, probably digital
            # _, y = signal.freqs(b, a, x)
            _, y = signal.freqz(b, a, x)
        else: # e.g. Lanczos, need freqz
            _, y = signal.freqz(b, a, x)
        x = x/(2*np.pi*dx) # the last entry is the **Nyquist frequency**, or 1/(dx*2)
        y = np.abs(y)
    return x, y

def impulse():
    """
    Displays the *impulse* response function for a recursive filter.
    """
    # R2_q = 1./(1. + (omega/omega_c)**(2*N))
    pass

#------------------------------------------------------------------------------#
# Filters and windows
#------------------------------------------------------------------------------#
def harmonics(x, k=4, axis=-1, absval=False): #n=np.inf, kmin=0, kmax=np.inf): #, kscale=1, krange=None, k=None):
    """
    Select the first k Fourier harmonics of the time series. Useful
    for example in removing seasonal cycle or something.
    """
    # Get fourier transform
    x   = permute(x, axis)
    fft = np.fft.fft(x, axis=-1)
    # Remove frequencies outside range. The FFT will have some error and give
    # non-zero imaginary components, but we can get magnitude or naively cast to real
    fft[...,0] = 0
    fft[...,k+1:-k] = 0
    if absval:
        y = unpermute(np.abs(np.fft.ifft(fft)), axis)
    else:
        y = unpermute(np.real(np.fft.ifft(fft)), axis)
    return y

def highpower(x):
    """
    Select only the highest power frequencies. Useful for
    crudely reducing noise.
    """
    # Naively remove certain frequencies
    # Should ignore first coefficient, the mean
    x   = permute(x, axis)
    fft = np.fft.fft(x, axis=-1)
    fftfreqs = np.arange(1, fft.shape[-1]//2) # up to fft.size/2 - 1, units cycles per sample
    # Get indices of n largest values
    # Use *argpartition* because it's more efficient, will just put
    # -nth element into sorted position, everything after that unsorted
    # but larger (don't need exact order!)
    p = np.abs(fft)**2
    f = np.argpartition(p, -n, axis=-1)[...,-n:]
    y = fft.copy()
    y[...] = 0 # fill in
    y[...,f] = fft[...,f] # put back the high-power frequencies
    freqs = fftfreqs[...,f]
    return freqs, y # frequencies and the high-power filter

def lanczos(dx, width, cutoff):
    """
    Returns *coefficients* for Lanczos high-pass filter with
    desired wavelength specified.
    See: https://scitools.org.uk/iris/docs/v1.2/examples/graphics/SOI_filtering.html

    Parameters
    ----------
    dx :
         units of your x-dimension (so that cutoff can be translated
         from physical units to 'timestep' units)
    width :
        length of filter in time steps
    cutoff :
        cutoff wavelength in physical units

    Returns
    -------
    b :
        numerator coeffs
    a :
        denominator coeffs

    Notes
    -----
    * The smoothing should only be *approximate* (see Hartmann notes), response
    function never exactly perfect like with Butterworth filter.
    * The '2' factor appearing in multiple places may seem random. But actually
    converts linear frequency (i.e. wavenumber) to angular frequency in
    sine call below. The '2' doesn't appear in any other factor just as a
    consequence of the math.
    * Code is phrased slightly differently, more closely follows Libby's discription in class.
    * Keep in mind 'cutoff' must be provided in *time step units*. Change
    the converter 'dx' otherwise.

    Example
    -------
    n=9 returns 4+4+1=9 points in the 'concatenate' below.
    """
    # Coefficients and initial stuff
    alpha = 1.0/(cutoff/dx) # convert alpha to wavenumber (new units are 'inverse timesteps')
    # n = (width/dx)//1 # convert window width from 'time units' to 'time steps'
    n = width
    n = (n - 1)//2 + 1
    # n = width//2
    print(f'Order-{n*2 - 1:.0f} Lanczos window')
    tau = np.arange(1,n+1) # lag time
    C0 = 2*alpha # integral of cutoff-response function is alpha*pi/pi
    Ck = np.sin(2*np.pi*alpha*tau)/(np.pi*tau)
    Cktilde = Ck*np.sin(np.pi*tau/n)/(np.pi*tau/n)
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
      dx :
        data spacing
      order :
        order of the filter
      cutoff :
        cutoff frequency in 'x' units (i.e. *wavelengths*)

    Returns
    -------
      b :
        numerator coeffs
      a :
        denominator coeffs

    Notes
    -----
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
    N = order # or order
    analog = False # seem to need digital, lfilter needed to actually filter stuff and doesn't accept analog
    if analog:
        cutoff = 2*np.pi/(cutoff/dx) # from wavelengths to rad/time steps
    else:
        cutoff = 1.0/cutoff # to Hz, or cycles/unit
        cutoff = cutoff*(2*dx) # to cycles/(2 timesteps), must be relative to nyquist
    if cutoff > 1:
        raise ValueError('Cuttoff frequency must be in [0, 1]. Remember you pass a cutoff *wavelength* to this function, not a frequency.')
        # cutoff = (1/cutoff)*(2/(1/dx)) # python takes this in frequency units!
    # Apply filter
    # print(f'Order-{N:.0f} Butterworth filter')
    b, a = signal.butter(N-1, cutoff, btype=btype, analog=analog, output='ba')
    return b, a

#------------------------------------------------------------------------------#
# Spectral analysis
#------------------------------------------------------------------------------#
def window(wintype, n, normalize=False):
    """
    Retrieves weighting function window, identical to get_window(). Note
    the raw window weights must be normalized, or will screw up the power
    of your FFT coefficients.

    For windows that require extra parameters, window
    must be a tuple of window-name, parameter.
    """
    # Default error messages are shit, make them better
    if wintype=='welch':
        raise ValueError('Welch window needs 2-tuple of (name,beta).')
    if wintype=='kaiser':
        raise ValueError('Welch window needs 2-tuple of (name,beta).')
    if wintype=='gaussian':
        raise ValueError('Gaussian window needs 2-tuple of (name,stdev).')
    # Get window
    # NOTE: Window will be normalized to have maximum of 1. Windowing decreases
    # total power of resulting transform; to get it back, divide FFT by sum
    # of window coefficients instead of length of the FFT'd dimension.
    win = signal.get_window(wintype, n)
    return win

def power(y1, y2=None, dx=1, cyclic=False, coherence=False,
        nperseg=100, wintype='boxcar',
        center=np.pi, axis=-1,
        detrend='constant', scaling='spectrum'):
    """
    Gets the spectral decomposition for particular windowing technique.
    Uses simple numpy fft.

    Parameters
    ----------
    y1 :
        the data
    y2 :
        the data, if you want cross-spectral power
    dx :
        timestep in physical units (used for scaling the frequency-coordinates)
    cyclic :
        whether data is cyclic along axis; in this case the nperseg
        will be overridden

    Returns (single transform)
    -------
    f :
        wavenumbers, in <x units>**-1
    P :
        power spectrum, in units <data units>**2

    Returns (cross transform, coherence == False)
    -------
    f :
        wavenumbers, in <x units>**-1
    P :
        co-power spectrum
    Q :
        quadrature power spectrum
    Py1 :
        power spectrum for y1
    Py2 :
        power spectrum for y2

    Returns (cross transform, coherence == True)
    -------
    f :
        wavenumbers, in <x units>**-1
    Coh :
        coherence squared
    Phi :
        average phase relationship

    Notes
    -----
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
    # Power reduction depends on signal
    for y in (np.sin(np.arange(0,8*np.pi-0.01,np.pi/25)),
        np.random.rand(200)):
        yvar = ((y-y.mean())**2).mean()
        Y = (np.abs(np.fft.fft(y)[1:]/y.size)**2).sum()
        Yw = (np.abs(np.fft.fft(y*w)[1:]/y.size)**2).sum()
        print('test')
        print(Yw/yvar)
        print(Yw*(y.size/w.sum())/yvar)
        print(Y/yvar)
    """
    # Initial stuff
    N = y1.shape[axis] # window count
    if cyclic:
        wintype = 'boxcar'
        nperseg = N
    if y2 is not None and y2.shape != y1.shape:
        raise ValueError(f'Got conflicting shapes for y1 {y1.shape} and y2 {y2.shape}.')
    nperseg = 2*(nperseg // 2) # enforce even window size
    # Trim if necessary
    r = N % nperseg
    if r!=0:
        s = [slice(None) for i in range(y1.ndim)]
        s[axis] = slice(None,-r)
        y1 = y1[tuple(s)] # slice it up
        N = y1.shape[axis] # update after trim
        if y2 is not None:
            y2 = y2[tuple(s)]
        print(f'Warning: Trimmed {r} out of {N} points to accommodate length-{nperseg} window.')

    # Just use scipy csd
    # 'one-sided' says to only return first symmetric half if data is real
    # 'scaling' queries whether to:
    #  * scale 'per wavenumber'/'per Hz', option 'density', default;
    #    this is analagous to a Planck curve with intensity per wavenumber
    #  * show the power (so units are just data units squared); this is
    #    usually what we want
    # if not manual and y2 is None:
    # f, P = signal.csd(y1, y1, window=wintype,
    #         return_onesided=True, scaling=scaling,
    #         nperseg=nperseg, noverlap=nperseg//2, detrend=detrend, axis=axis)

    # Manual approach
    # Have checked these and results are identical
    # Get copsectrum, quadrature spectrum, and powers for each window
    y1, shape = lead_flatten(permute(y1, axis)) # shape is shape of *original* data
    if y2 is not None:
        y2, _ = lead_flatten(permute(y2, axis)) # shape is shape of *original* data
    extra = y1.shape[0]
    pm = nperseg//2
    shape[-1] = pm # new shape
    # List of *center* indices for windows
    win = window(wintype, nperseg)
    loc = np.arange(pm, N-pm+pm//2, pm) # jump by half window length
    if len(loc)==0:
        raise ValueError('Window length too big.')
    # Ouput arrays
    Py1 = np.empty((extra, loc.size, pm)) # power
    if y2 is not None:
        CO = Py1.copy()
        Q = Py1.copy()
        Py2 = Py1.copy()
    for j in range(extra):
        # Loop through windows
        if np.any(~np.isfinite(y1[j,:])) or (y2 is not None and np.any(~np.isfinite(y2[j,:]))):
            print('Warning: Skipping array with missing values.')
            continue
        for i,l in enumerate(loc):
            if y2 is None:
                # Remember to double the size of power, because only
                # have half the coefficients (rfft not fft)
                # print(win.size, pm, y1[j,l-pm:l+pm].size)
                wy = win*signal.detrend(y1[j,l-pm:l+pm], type=detrend)
                Fy1 = np.fft.rfft(wy)[1:]/win.sum()
                Py1[j,i,:] = np.abs(Fy1)**2
                Py1[j,i,:-1] *= 2
            else:
                # Frequencies
                wy1 = win*signal.detrend(y1[j,l-pm:l+pm], type=detrend)
                wy2 = win*signal.detrend(y2[j,l-pm:l+pm], type=detrend)
                Fy1 = np.fft.rfft(wy1)[1:]/win.sum()
                Fy2 = np.fft.rfft(wy2)[1:]/win.sum()
                # Powers
                Py1[j,i,:] = np.abs(Fy1)**2
                Py2[j,i,:] = np.abs(Fy2)**2
                CO[j,i,:]  = (Fy1.real*Fy2.real + Fy1.imag*Fy2.imag)
                Q[j,i,:]   = (Fy1.real*Fy2.imag - Fy2.real*Fy1.imag)
                for array in (Py1,Py2,CO,Q):
                    array[j,i,:-1] *= 2 # scale all but Nyquist frequency

    # Helper function
    def unshape(x):
        x = lead_unflatten(x, shape)
        x = unpermute(x, axis)
        return x
    # Get window averages, reshape, and other stuff
    # NOTE: For the 'real' transform, all values but Nyquist must
    # be divided by two, so that an 'average' of the power equals
    # the covariance.
    f = np.fft.rfftfreq(nperseg)[1:] # frequencies
    if y2 is None:
        # Average windows
        Py1 = Py1.mean(axis=1)
        Py1 = unshape(Py1)
        return f/dx, Py1
    else:
        # Averages
        CO  = CO.mean(axis=1)
        Q   = Q.mean(axis=1)
        Py1 = Py1.mean(axis=1)
        Py2 = Py2.mean(axis=1)
        if coherence: # return coherence and phase instead
            # Coherence and stuff
            Coh = (CO**2 + Q**2)/(Py1*Py2)
            Phi = np.arctan2(Q, CO) # phase
            Phi[Phi >= center + np.pi] -= 2*np.pi
            Phi[Phi <  center - np.pi] += 2*np.pi
            Phi = Phi*180/np.pi # convert to degrees!!!
            Coh = unshape(Coh)
            Phi = unshape(Phi)
            return f/dx, Coh, Phi
        else:
            # Reshape and return
            CO  = unshape(CO)
            Q   = unshape(Q)
            Py1 = unshape(Py1)
            Py2 = unshape(Py2)
            return f/dx, CO, Q, Py1, Py2

def power2d(z1, z2=None, dx=1, dy=1, coherence=False,
        nperseg=100, wintype='boxcar',
        center=np.pi, axes=(-2,-1), # first position is *cyclic* (perform simple real transform), second is *not* (windowed)
        manual=False, detrend='constant', scaling='spectrum'):
    """
    Performs 2-d spectral decomposition, with windowing along only *one* dimension,
    in style of Randel and Held 1991. Therefore assumption is we have *cyclic*
    data along one dimension, the 'x' dimension.

    Returns (single transform)
    -------
    f :
        wavenumbers, in <x units>**-1
    P :
        power spectrum, in units <data units>**2

    Returns (cross transform, coherence == False)
    -------
    f :
        wavenumbers, in <x units>**-1
    P :
        co-power spectrum
    Q :
        quadrature power spectrum
    Pz1 :
        power spectrum for y1
    Pz2 :
        power spectrum for y2

    Returns (cross transform, coherence == True)
    -------
    f :
        wavenumbers, in <x units>**-1
    Coh :
        coherence squared
    Phi :
        average phase relationship

    Notes
    -----
    See notes for the 1D version.
    """
    # Checks
    taxis, caxis = axes
    if len(z1.shape)<2:
        raise ValueError('Need at least rank 2 array.')
    if z2 is not None and not z1.shape==z2.shape:
        raise ValueError(f'Shapes of x {x.shape} and y {y.shape} must match.')
    print(f'Cyclic dimension ({caxis}): Length {z1.shape[caxis]}.')
    print(f'Windowed dimension ({taxis}): Length {z1.shape[taxis]}, window length {nperseg}.')
    if caxis<0:
        caxis = z1.ndim-caxis
    if taxis<0:
        taxis = z1.ndim-taxis
    # if caxis<=taxis: # could generalize, but tried that for eof2D and it was huge pain in the ass
    #     # TODO: Actually perhaps this is not necessary?
    #     raise ValueError('Cyclic axis must be to right of non-cyclic axis.')
    nperseg = 2*(nperseg // 2) # enforce even window size because I said so
    l = z1.shape[taxis]
    r = l % nperseg
    if r>0:
        s = [slice(None) for i in range(z1.ndim)]
        s[taxis] = slice(None,-r)
        z1 = z1[s] # slice it up
        print(f'Warning: Trimmed {r} out of {l} points to accommodate length-{nperseg} window.')
        # raise ValueError(f'Window width {nperseg} does not divide axis length {z1.shape[axis]}.')

    # Helper function
    # Will put the *non-cyclic* axis on position 1, *cyclic* axis on position 2
    # Mirrors convention for row-major geophysical data array storage, time
    # by pressure by lat by lon (the cyclic one).
    offset = int(taxis<caxis) # TODO: does this work, or is stuff messed up?
    nflat = z1.ndim - 2 # we overwrite z1, so must save this value!
    def reshape(x):
        x = permute(x, taxis, -1) # put on -1, then will be moved to -2
        x = permute(x, caxis - offset, -1) # put on -1
        x, shape = lead_flatten(x, nflat) # flatten remaining dimensions
        return x, shape

    # Permute
    z1, shape = reshape(z1)
    extra = z1.shape[0]
    if z2 is not None:
        z2, _ = reshape(z2)
    # For output data
    N = z1.shape[1] # non-cyclic dimension
    M = z1.shape[2] # cyclic dimension
    pm = nperseg//2
    shape[-2] = 2*pm # just store the real component
    shape[-1] = M//2

    # Helper function
    # Gets 2D Fourier decomp, reorders negative frequencies on non-cyclic
    # axis so frequencies there are monotonically ascending.
    win = window(wintype, nperseg) # for time domain
    wsum = (win**2).sum()
    def freqs(x, pm):
        # Detrend
        # Or don't, since we shave the constant part anyway?
        # x = signal.detrend(x, type='constant', axis=1) # remove mean for cyclic one
        # x = signal.detrend(x, type=detrend, axis=0) # remove trend or mean
        # The 2D approach
        # NOTE: Read documentation regarding normalization. Default leaves
        # forward transform unnormalized, reverse normalized by 1/n; the ortho
        # option normalizes both by 1/sqrt(n).
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/routines.fft.html#module-numpy.fft
        X = np.fft.rfft2(win[:,None]*x, axes=(0,1)) # last axis specified should get a *real* transform
        X = X[:,1:] # remove the zero-frequency value
        X = X/(x.shape[0]*x.shape[1]) # complex FFT has to be normalized by sample size
        # print(w.sum()/x.shape[0])
        # print((w**2).sum()/x.shape[0])
        return np.concatenate((X[pm:,:], X[1:pm+1,:]), axis=0)
        # Manual approach, virtually identical
        # Follows Libby's recipe, where instead real is cosine and imag is
        # sine. Note only need to divide by 2 when conjugates are included.
        # xi = np.fft.rfft(x, axis=1)[:,1:]/x.shape[1]
        # xi = win[:,None]*xi # got a bunch of sines and cosines
        # C = np.fft.rfft(xi.real, axis=0)[1:,:]/x.shape[0]
        # S = np.fft.rfft(xi.imag, axis=0)[1:,:]/x.shape[0]
        # return np.concatenate((
        #     (C.real + S.imag + 1j*(C.imag - S.real))[::-1,:], # frequency increasing in absolute magnitude along axis
        #      C.real - S.imag + 1j*(-C.imag - S.real),
        #     ), axis=0)

    # The window *centers* for time windowing
    loc = np.arange(pm, N - pm + 0.1, pm).astype(int) # jump by half window length
    if len(loc)==0:
        raise ValueError('Window length too big.')
    # Get the spectra
    Pz1 = np.nan*np.empty((extra, loc.size, *shape[-2:])) # power
    if z2 is not None:
        CO = Pz1.copy()
        Q = Pz1.copy()
        Pz2 = Pz1.copy()
    for j in range(extra):
        # Missing values handling
        if np.any(~np.isfinite(z1[j,:,:])) or (z2 is not None and np.any(~np.isfinite(z2[j,:,:]))):
            print('Warning: Skipping array with missing values.')
            continue
        # 2D transform for each window on non-cyclic dimension
        for i,l in enumerate(loc):
            if z2 is None:
                # Note since we got the rfft (not fft) in one direction, only
                # have half the coefficients (they are symmetric); means for
                # correct variance, have to double the power.
                Fz1 = freqs(z1[j,l-pm:l+pm,:], pm)
                Pz1[j,i,:,:] = np.abs(Fz1)**2
                Pz1[j,i,:,:-1] *= 2
            else:
                # Frequencies
                Fz1 = freqs(z1[j,l-pm:l+pm,:], pm)
                Fz2 = freqs(z2[j,l-pm:l+pm,:], pm)
                # Powers
                Phi1 = np.arctan2(Fz1.imag, Fz1.real) # analagous to Libby's notes, for complex space
                Phi2 = np.arctan2(Fz2.imag, Fz2.real)
                CO[j,i,:,:]  = np.abs(Fz1)*np.abs(Fz2)*np.cos(Phi1 - Phi2)
                Q[j,i,:,:]   = np.abs(Fz1)*np.abs(Fz2)*np.sin(Phi1 - Phi2)
                Pz1[j,i,:,:] = np.abs(Fz1)**2
                Pz2[j,i,:,:] = np.abs(Fz2)**2
                for array in (CO,Q,Pz1,Pz2):
                    array[j,i,:,:-1] *= 2

    # Frequencies
    # Make sure Nyquist frequency is appropriately signed on either side of array
    fx = np.fft.fftfreq(2*pm) # just the positive-direction Fourier coefs
    fx = np.concatenate((-np.abs(fx[pm:pm+1]), fx[pm+1:], fx[1:pm], np.abs(fx[pm:pm+1])), axis=0)
    fy = np.fft.rfftfreq(M)[1:]

    # Helper function
    # Reshapes final result, and scales powers so we can take dimensional
    # average without needing to divide by 2
    def unshape(x):
        x = lead_unflatten(x, shape, nflat)
        x = unpermute(x, caxis - offset) # put caxis back, accounting for if taxis moves to left
        x = unpermute(x, taxis) # put taxis back
        return x
    # Get window averages, reshape, and other stuff
    # NOTE: For the 'real' transform, all values but Nyquist must
    # be divided by two, so that an 'average' of the power equals
    # the covariance.
    if z2 is None:
        # Return
        Pz1 = Pz1.mean(axis=1)
        Pz1 = unshape(Pz1)
        return fx/dx, fy/dy, Pz1
    else:
        # Averages
        print(Pz1.shape)
        CO  = CO.mean(axis=1)
        Q   = Q.mean(axis=1)
        Pz1 = Pz1.mean(axis=1)
        Pz2 = Pz2.mean(axis=1)
        if coherence: # return coherence and phase instead
            # Coherence and stuff
            # NOTE: This Phi relationship is still valid. Check Libby's
            # notes; divide here Q by CO and the Ws cancel out, end up
            # with average phase difference indeed.
            Coh = (CO**2 + Q**2)/(Pz1*Pz2)
            Phi = np.arctan2(Q, CO) # phase
            Phi[Phi >= center + np.pi] -= 2*np.pi
            Phi[Phi <  center - np.pi] += 2*np.pi
            # Reshape and return
            Coh = unshape(Coh)
            Phi = unshape(Phi)
            return fx/dx, fy/dy, Coh, Phi
        else:
            # Reshape
            CO  = unshape(CO)
            Q   = unshape(Q)
            Pz1 = unshape(Pz1)
            Pz2 = unshape(Pz2)
            return fx/dx, fy/dy, CO, Q, Pz1, Pz2

def autopower():
    """
    This will turn into a wrapper around power1d, that generates co-spectral
    statistics and whatnot at ***successive lags***.

    Uses scipy.signal.welch windowing method to generate an estimate of the
    *lagged* spectrum. Can also optionally do this with two variables.
    """
    raise NotImplementedError()

