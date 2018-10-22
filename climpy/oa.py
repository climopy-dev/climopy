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
def gaussian(N=1000, mean=0, sigma=1):
    """
    Returns sample points on Gaussian curve.
    """
    norm = stats.norm(loc=mean, scale=sigma)
    x = np.linspace(norm.ppf(0.0001), norm.ppf(0.9999), N) # get x through percentile range
    pdf = norm.pdf(x, loc=mean, scale=sigma)
    return x, pdf

#------------------------------------------------------------------------------
# Artificial data
#------------------------------------------------------------------------------
def rednoise(ntime, a, init=1, samples=1, mean=0, stdev=1, nested=False):
    """
    Creates artificial red noise time series, i.e. a weighted sum of random perturbations.
    Equation is: x(t) = a*x(t-dt) + b*eps(t)
    where a is the lag-1 autocorrelation and b is a scaling term.
     * Output will have shape ntime by nsamples.
     * Enforce that the first timestep always equals the 'starting' position.
     * Use 'nested' flag to control which algorithm to use.
     * Use 'init' to either a) declare integer starting point or b) pick
       some random number from a uniform distribution in that range.
    """
    # Initial stuff
    ntime -= 1 # exclude the initial timestep
    init = np.atleast_1d(init)
    samples = np.atleast_1d(samples)
    data = np.empty((ntime+1,*samples)) # user can make N-D array
    b = (1-a**2)**0.5  # from OA class
    # Nested loop
    data, shape = trail_flatten(data)
    data[0,:] = 0 # initialize
    for i in range(data.shape[-1]):
        eps = np.random.normal(loc=0, scale=1, size=ntime)
        for t in range(1,ntime+1):
            data[t,i] = a*data[t-1,i] + b*eps[t-1]
    # if hasattr(init,'__iter__') and len(init)==2:
    #     data = data + np.random.uniform(*init, size=data.shape[1]) # overkill
    if data.shape[-1]!=init.shape[-1] and init.size!=1:
        raise ValueError('Length of vector of initial positions must equal number of sample time series.')
    data = data + init # scalar or iterable (in which case, right-broadcasting takes place)
    # Trying to be fancy, just turned out fucking way slower
    # aseries = b*np.array([a**(ntime-i) for i in range(1,ntime+1)])
    # for i in range(samples):
    #     eps = np.random.normal(loc=0, scale=1, size=ntime)
    #     vals = eps[:,None]@aseries[None,:] # matrix for doing math on
    #     data[1:,i] = [np.trace(vals,ntime-i) for i in range(1,ntime+1)]
    data = trail_unflatten(data, shape)
    if len(samples)==1 and samples[0]==1:
        data = data.squeeze()
    return mean + stdev*data # rescale to have specified stdeviation/mean

def waves(x, wavenums=None, wavelens=None, phase=None):
    """
    Compose array of sine waves.
    Useful for testing performance of filters.
    Input:
        x: if scalar, 'x' is np.arange(0,x)
           if iterable, can be n-dimensional, and will calculate sine
           from coordinates on every dimension
    Required kwarg -- either of:
        wavelens: wavelengths for sine function
        wavenums: wavenumbers for sine function
    Output:
        data: data composed of waves.
    Notes:
      * 'x' will always be normalized so that wavelength
        is with reference to first step.
      * this make sense because when working with filters, almost
        always need to use units corresponding to axis.
    """
    # Wavelengths
    if wavenums is None and wavelens is None:
        raise ValueError('Must declare wavenums or wavelengths.')
    elif wavelens is not None:
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
    Input:
        y:    assumed 'x' is 0 to len(y)-1
        x, y: arbitrary 'x', must monotonically increase
    Optional:
        axis: regression axis
        build: whether to replace regression axis with scalar slope, or
            reconstructed best-fit line.
        stderr: if not doing 'build', whether to add standard error on
            the slope in index 1 of regressoin axis.
    Output:
        y: regression params on axis 'axis', with offset at index 0,
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
    Input:
        data: the input data (this function computes necessary correlation coeffs).
    Output:
        spectrum: the autocorrelation spectrum out to nlag lags.
    Optional:
        * If 'series' is True, return the red noise spectrum full series.
        * If 'series' is False, return just the *timescale* associated with the red noise spectrum.
        * If 'lag1' is True, just return the lag-1 version.
    """
    # Initial stuff
    if nlag is None:
        raise ValueError(f"Must declare \"nlag\" argument; number of points to use for fit.")
    data, shape = lead_flatten(permute(data, axis))
    # if len(time)!=data.shape[-1]:
    #     raise ValueError(f"Got {len(time)} time values, but {data.shape[-1]} timesteps for data.")
    # First get the autocorrelation spectrum, and flatten leading dimensions
    # Dimensions will need to be flat because we gotta loop through each 'time series' and get curve fits.
    # time = time[:nlag+1] # for later
    autocorrs = autocorr(data, nlag, axis=-1, verbose=verbose)
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
def autocorr(data, nlag=None, lag=None, verbose=False, axis=0, _normalize=True):
    """
    Gets the autocorrelation spectrum at successive lags.
    Input:
        data: the input data.
    Output:
        autocorrs: the autocorrelations as a function of lag.
    Optional:
        nlag: get correlation at multiple lags ('n' is number after the 0-lag).
        lag: get correlation at just this lag.
        _normalize: used for autocovar wrapper. generally user shouldn't touch this.
    Notes:
      * Uses following estimator: ((n-k)*sigma^2)^-1 * sum_i^(n-k)[X_t - mu][X_t+k - mu]
        See: https://en.wikipedia.org/wiki/Autocorrelation#Estimation
      * By default, includes lag-zero values; if user just wants single lag
        will, however, throw out those values.
    """
    # Preparation, and stdev/means
    data = np.array(data)
    naxis = data.shape[axis] # length
    if (nlag is None and lag is None) or (nlag is not None and lag is not None):
        raise ValueError(f"Must specify *either* a lag (\"lag=x\") or range of lags (\"nlag=y\").")
    if nlag is not None and nlag>=naxis/2:
        raise ValueError(f"Lag {nlag} must be greater than axis length {naxis}.")
    if verbose:
        if nlag is None:
            print(f"Calculating lag-{lag} autocorrelation.")
        else:
            print(f"Calculating autocorrelation spectrum up to lag {nlag} for axis length {naxis}.")
    data = permute(data, axis)
    mean = data.mean(axis=-1, keepdims=True) # keepdims for broadcasting in (data minus mean)
    if _normalize:
        var = data.var(axis=-1, keepdims=False) # this is divided by the summation term, so should have annihilated axis
    else:
        var = 1 # dummy
    # Trivial autocorrelation done, just fill with ones
    if nlag is None and lag==0:
        autocorrs = np.ones((*data.shape[:-1],1))
    # Calculate on specific lag
    elif nlag is None:
        autocorrs = np.sum((data[...,:-lag]-mean)*(data[...,lag:]-mean),axis=-1)/((naxis-lag)*var)
        autocorrs = autocorrs[...,None] # add dimension back in
    # Up to n timestep-lags after 0-correlation
    else:
        autocorrs = np.empty((*data.shape[:-1], nlag+1)) # will include the zero-lag autocorrelation
        autocorrs[...,0] = 1 # lag-0 autocorrelation
        for i,lag in enumerate(range(1,nlag+1)):
            autocorrs[...,i+1] = np.sum((data[...,:-lag]-mean)*(data[...,lag:]-mean),axis=-1)/((naxis-lag)*var)
    return unpermute(autocorrs, axis)

def autocovar(*args, **kwargs):
    """
    As above, but gets the covariance.
    """
    return autocorr(*args, **kwargs, _normalize=False)

#------------------------------------------------------------------------------#
# Empirical orthogonal functions and related decomps
#------------------------------------------------------------------------------#
def eof(data, record=-2, space=-1, weights=None, neof=5, normalize=False):
    """
    Calculates the temporal EOFs, using the scipy algorithm for
    Hermetian (or real symmetric) matrices. This version allows
    calculating just 'n' most important ones.
    Input:
        data: data of arbitrary shape
    Kwargs:
        neof: number of eigenvalues we want
        record: axis used as 'record' dimension -- should only be 1
        space: axes used as 'space' dimension -- can be many
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
        raise ValueError('Organize your data! Need space dimensions to come before/after time dimension.')
    # Remove the mean and optionally standardize the data
    data = data - data.mean(axis=m_dims[0], keepdims=True) # remove mean
    if normalize:
        data = data / data.stdev(axis=m_dims[0], keepdims=True) # optionally standardize, usually not wanted for annular mode stuff
    # Next apply weights
    m = np.prod([data.shape[i] for i in n_dims])
    n = np.prod([data.shape[i] for i in m_dims])
    if weights is None:
        weights = 1
    weights = np.atleast_1d(weights) # want numpy array
    try:
        if m>n: # more sampling than space dimensions
            data  = data*np.sqrt(weights)
            dataw = data
        else: # more space than sampling dimensions
            dataw = data*weights
    except ValueError:
        raise ValueError(f'Dimensionality of weights {weights.shape} incompatible with dimensionality of space dimensions {data.shape[-2:]}!')
    # Turn matrix into record by space
    # print('initial', data.shape)
    data  = permute(data, record, -1)
    dataw = permute(dataw, record, -1)
    # print('record permute', data.shape)
    for i,axis in enumerate(n_dims):
        axis = axis-i-1 if space_after else axis # if permuting when space comes *after* record, actual axes of our data keep changing
        data  = permute(data, axis, -1)
        dataw = permute(dataw, axis, -1)
    # Only flatten after apply weights (e.g. if have level and latitude dimensoins)
    shape_trail = data.shape[-n_dims.size:]
    data,  _ = trail_flatten(data,  n_dims.size)
    dataw, _ = trail_flatten(dataw, n_dims.size)
    shape_lead = data.shape[:-2]
    data,  _ = lead_flatten(data,  data.ndim-2)
    dataw, _ = lead_flatten(dataw, dataw.ndim-2)
    # Prepare output; will add a new 'eof dimension' to the trailing side
    if data.ndim!=3:
        raise ValueError(f"Shit's on fire yo.")
    nextra, m, n = data.shape[0], data.shape[1], data.shape[2] # n extra, record, and space
    pcs   = np.empty((nextra, neof,  m, 1))
    projs = np.empty((nextra, neof,  1,  n))
    evals = np.empty((nextra, neof,  1,  1))
    nstar = np.empty((nextra, neof, 1,  1))

    # Get EOFs and PCs and stuff
    for i in range(data.shape[0]):
        # Initial
        x = data[i,:,:] # array will be sampling by space
        xw = dataw[i,:,:]
        # Get reduced degrees of freedom for spatial eigenvalues
        # TODO: Fix the weight projection below
        rho = np.corrcoef(x.T[:,1:], x.T[:,:-1])[0,1] # must be (space x time)
        rho_ave = (rho*weights).sum()/weights.sum()
        nstar[i,:,0,0] = m*((1-rho_ave)/(1+rho_ave)) # simple degrees of freedom estimation
        # Get EOFs using covariance matrix on *shortest* dimension
        if x.shape[0] > x.shape[1]:
            # Get *temporal* covariance matrix since time dimension larger
            eigrange = [n-neof, n-1] # eigenvalues to get
            l, v = linalg.eigh((xw.T@xw)/m, eigvals=eigrange, eigvals_only=False)
            z = xw@v # i.e. multiply (time x space) by (space x neof), get (time x neof)
            z = (z-z.mean(axis=0))/z.std(axis=0) # standardize pcs
            p = x.T@z/m # i.e. multiply (space x time) by (time x neof), get (space x neof)
        else:
            # Get *spatial* dispersion matrix since space dimension longer
            # This time 'eigenvectors' are actually the pcs
            eigrange = [m-neof, m-1] # eigenvalues to get
            l, z = linalg.eigh((xw@x.T)/n, eigvals=eigrange, eigvals_only=False)
            z = (z-z.mean(axis=0))/z.std(axis=0) # standardize pcs
            p = x.T@z/m # i.e. multiply (space x time) by (time by neof), get (space x neof)
        # Store in big arrays
        pcs[i,:,:,0]   = z.T[::-1,:] # neof by time
        projs[i,:,0,:] = p.T[::-1,:] # neof by space
        evals[i,:,0,0] = l[::-1] # neof
        # # Sort
        # idx = L.argsort()[::-1]
        # L, Z = L[idx], Z[:,idx]

    # Return them along the correct dimension
    nlead = len(shape_lead)
    pcs   = lead_unflatten(pcs,   [*shape_lead, neof, m, 1], nlead)
    projs = lead_unflatten(projs, [*shape_lead, neof, 1, n], nlead)
    evals = lead_unflatten(evals, [*shape_lead, neof, 1, 1],  nlead)
    nstar = lead_unflatten(nstar, [*shape_lead, neof, 1, 1],  nlead)
    ntrail = len(shape_trail)
    flat_trail = [1]*len(shape_trail)
    pcs   = trail_unflatten(pcs,   [*shape_lead, neof, m, *flat_trail],  ntrail)
    projs = trail_unflatten(projs, [*shape_lead, neof, 1,  *shape_trail], ntrail)
    evals = trail_unflatten(evals, [*shape_lead, neof, 1,  *flat_trail],  ntrail)
    nstar = trail_unflatten(nstar, [*shape_lead, neof, 1,  *flat_trail],  ntrail)
    # Permute 'eof' dimension onto the end (note we had to put it before the
    # record and space dimensions so we could perform 'trail_unflatten')
    di, df = len(shape_lead), pcs.ndim-1 # eofs are on the one *after* those leading dimensions
    pcs   = np.moveaxis(pcs, di, df)
    projs = np.moveaxis(projs, di, df)
    evals = np.moveaxis(evals, di, df)
    nstar = np.moveaxis(nstar, di, df)
    # Finally, permute stuff back to original positions
    for i,axis in enumerate(space):
        axis = axis-i-1 if space_after else axis
        pcs = unpermute(pcs, axis, -2)
        projs = unpermute(projs, axis, -2)
        evals = unpermute(evals, axis, -2)
        nstar = unpermute(nstar, axis, -2)
    pcs = unpermute(pcs, record, -2)
    projs = unpermute(projs, record, -2)
    evals = unpermute(evals, record, -2)
    nstar = unpermute(nstar, record, -2)
    # And return everything! Beautiful!
    return evals, nstar, projs, pcs

    #--------------------------------------------------------------------------#
    # Data should be time by location
    # # Standardize
    # Z = (Z-Z.mean(0))/Z.std(0)
    # # Fractions
    # frac = np.abs(L)/L.sum()
    # # And re-project; now *rows* are anomalies associated with 1std of each PC of original time series
    # Xun[np.isnan(Xun)] = Z[np.isnan(Z)] = 0 # replace with zero, so matrix multiplication will work
    # Doutput = (Z.T@Xun)/Xun.shape[0] # use the new shape
    # D = np.full(old_shape,np.nan)
    # D[:,goodfilt] = Doutput
    # print('EOFs calculated.')
    # # Autocorrelations
    # out = (D, Z)
    # if return_frac:
    #     rho = np.zeros((X.shape[1],))
    #     for i in range(X.shape[1]):
    #         if i%10000 == 0: print(i)
    #         rho[i] = np.corrcoef(X[:-1,i],X[1:,i])[1,0]
    #     rho_ave = (rho*w).sum()/w.sum()
    #     Nstar = X.shape[0]*((1-rho_ave)/(1+rho_ave))
    #     delta_frac = (L*np.sqrt(2/Nstar))/L.sum()
    #     print('Spatial mean autocorrelation: %.5f' % rho_ave)
    #     out = out + (frac, delta_frac)
    # if return_filt:
    #     out = out + (goodfilt,)
    # return out

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
    Input:
        x: data, and we roll along axis 'axis'.
    Optional:
        w (int or iterable): boxcar window length, or custom weights
        pad (bool):        whether to pad the edges of axis back to original size
        padvalue (float): what to pad with (default np.nan)
        btype (string):    whether to apply lowpass, highpass, or bandpass
        kwargs: remaining kwargs passed to windowing function
    Output:
        x: data windowed along arbitrary dimension
    Notes:
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

def filter(x, b, a=1, n=1, axis=-1,
              fix=True, fixvalue=np.nan):
    """
    Apply scipy.signal.lfilter to data. By default this does *not* pad
    ends of data. May keep it this way.
    Input:
        x: data to be filtered
        b: b coefficients (non-recursive component)
        a: scale factor in index 0, followed by a coefficients (recursive component)
        n: number of times to filter data (will go forward-->backward-->forward...)
        axis: axis along which we filter data
    Optional:
        fix: whether to (a) trim leading part of axis by number of a/b coefficients
        and (b) fill trimmed values with NaNs; will also attempt to *re-center*
        the data if a net-forward (e.g. f, fbf, fbfbf, ...) filtering was performed; this
        won't work for recursive filters, but does for non-recursive filters
    Output:
        y: filtered data
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
            y = y[:,n_2sides+n_left:-n_2sides]
        # shape[axis] = shape[axis] - 2*n_2sides - n_left # use if not padding 
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
    Formula:
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

def lanczos(dx, width, cutoff, response=True):
    """
    Returns *coefficients* for Lanczos high-pass filter with
    desired wavelength specified.
    Input:
        width: length of filter in physical units
        cutoff: cutoff wavelength in physical units
        dx: units of your x-dimension (so that cutoff can be translated
        from physical units to 'timestep' units)
    Returns:
      b: numerator coeffs
      a: denominator coeffs
    See: https://scitools.org.uk/iris/docs/v1.2/examples/graphics/SOI_filtering.html
    Notes:
      * The smoothing should only be *approximate* (see Hartmann notes), response
        function never exactly perfect like with Butterworth filter.
      * The '2' factor appearing in multiple places may seem random. But actually
        converts linear frequency (i.e. wavenumber) to angular frequency in
        sine call below. The '2' doesn't appear in any other factor just as a
        consequence of the math.
      * Code is phrased slightly differently, more closely follows Libby's discription in class.
      * Keep in mind 'cutoff' must be provided in *time step units*. Change
        the converter 'dx' otherwise.
      * Example, n=9 returns 4+4+1=9 points in the 'concatenate' below.
    """
    # Coefficients and initial stuff
    alpha = 1.0/(cutoff/dx) # convert alpha to wavenumber (new units are 'inverse timesteps')
    n = (width/dx)//1 # convert window width from 'time units' to 'time steps'
    n = (n - 1)//2 + 1
    # n = width//2
    print(f'order-{n*2 - 1:.0f} Lanczos window')
    tau = np.arange(1,n+1) # lag time
    C0 = 2*alpha # integral of cutoff-response function is alpha*pi/pi
    Ck = np.sin(2*np.pi*alpha*tau)/(np.pi*tau)
    Cktilde = Ck*np.sin(np.pi*tau/n)/(np.pi*tau/n)
    # Return filter
    window = np.concatenate((np.flipud(Cktilde), np.array([C0]), Cktilde))
    return window[1:-1], 1

def butterworth(dx, width, cutoff, btype='low'):
    """
    Applies Butterworth filter to data. Since this is a *recursive*
    filter, non-trivial to apply, so this uses scipy 'lfilter'.
    Get an 'impulse response function' by passing a bunch of zeros, and
    single non-zero 'point'.
    See Libby's function for more details.
    Notes:
      * Unlike Lanczos filter, the *length* of this should be
        determined always as function of timestep, because really high
        order filters can get pretty wonky.
      * Need to run *forward and backward* to prevent time-shifting.
      * Cutoff is point at which gain reduces to 1/sqrt(2) of the
        initial frequency. If doing bandpass, can 
      * The 'analog' means units of cutoffs are rad/s.
    Returns:
      b: numerator coeffs
      a: denominator coeffs
    """
    # Initial stuff
    # n = (width/dx)//1 # convert to timestep units
    # n = (n//2)*2 + 1 # odd numbered
    n = width # or order
    analog = False # seem to need digital, lfilter needed to actually filter stuff and doesn't accept analog
    if analog:
        cutoff = 2*np.pi/(cutoff/dx) # from wavelengths to rad/time steps
    else:
        cutoff = 1.0/cutoff # to Hz, or cycles/unit
        cutoff = cutoff*(2*dx) # to cycles/(2 timesteps), must be relative to nyquist
        # cutoff = (1/cutoff)*(2/(1/dx)) # python takes this in frequency units!
    print(f'order-{n:.0f} Butterworth filter')
    # Apply filter
    b, a = signal.butter(n-1, cutoff, btype=btype, analog=analog, output='ba')
    return b, a

#------------------------------------------------------------------------------#
# Spectral analysis
#------------------------------------------------------------------------------#
def window(wintype, n, normalize=False):
    """
    Retrieves weighting function window, identical to get_window().
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
    win = signal.get_window(wintype, n)
    if normalize:
        win = win/win.sum() # default normalizes so *maximum (usually central) value is 1*
    return win

def spectrum(dx, y, nperseg=72, wintype='boxcar', axis=-1,
        manual=False, detrend='linear', scaling='spectrum'):
    """
    Gets the spectral decomposition for particular windowing technique.
    Uses simple numpy fft.
    Input:
        dx: timestep in physical units (used for scaling the frequency-coordinates)
        y:  the data
    Output:
        f: wavenumbers, in <x units>**-1
        P: power spectrum, in units <data units>**2
    """
    # Checks
    l = y.shape[axis]
    r = l % nperseg
    if r>0:
        s = [slice(None) for i in range(y.ndim)]
        s[axis] = slice(None,-r)
        y = y[s] # slice it up
        print(f'Warning: Trimmed {r} out of {l} points to accommodate length-{nperseg} window.')
        # raise ValueError(f'Window width {nperseg} does not divide axis length {y.shape[axis]}.')
    if manual:
        # Get copsectrum, quadrature spectrum, and powers for each window
        y, shape = lead_flatten(permute(y, axis)) # shape is shape of permuted data
        N = y.shape[1] # window count
        win = window(wintype, nperseg)
        pm = nperseg//2
        loc = np.arange(pm, N-pm+pm//2, pm) # jump by half window length
        P = np.empty(y.shape) # power
        for j in range(y.shape[0]):
            Cx = np.empty((loc.size, pm)) # have half/window size number of freqs
            for i,l in enumerate(loc):
                # Note np.fft gives power A+Bi, so want sqrt(A^2 + B^2)/2
                # for +/- wavenumbers, easy
                wy = win*signal.detrend(y[j,l-pm:l+pm], type=detrend)
                Cx[i,:] = np.abs(np.fft.fft(wy)[:pm])**2
            P[j,:] = Cx.mean(axis=0)
        f = np.fft.fftfreq(nperseg)[:pm] # frequency
        P = unpermute(lead_unflatten(P, shape), axis)
        P = unpermute(P, axis)
    else:
        # Just use scipy csd
        # 'one-sided' says to only return first symmetric half if data is real
        # 'scaling' queries whether to:
        # * scale 'per wavenumber'/'per Hz', option 'density', default;
        #   this is analagous to a Planck curve with intensity per wavenumber
        # * show the power (so units are just data units squared); this is
        #   usually what we want
        wintype = window(wintype, nperseg) # has better error messages
        f, P = signal.csd(y, y, window=wintype,
                return_onesided=True, scaling=scaling,
                nperseg=nperseg, noverlap=nperseg//2, detrend=detrend, axis=axis)
    # Convert frequency to physical units
    # Scale so variance of windows (sum of power spectrum) equals variance of this
    # scale = y.var(axis=axis, keepdims=True)/P.sum(axis=axis, keepdims=True)
    # Scale so power is in proportion to total
    scale = 1.0/P.sum(axis=axis, keepdims=True)
    return f/dx, P*scale

def spectrum2d(dx, dy, z, nperseg, wintype='boxcar',
        axes=(-2,-1), # first position is *cyclic* (perform simple real transform), second is *not* (windowed)
        manual=False, detrend='linear', scaling='spectrum'):
    """
    Performs 2-d spectral decomposition, with windowing along only *one* dimension,
    in style of Randel and Held 1991. Therefore assumption is we have *cyclic*
    data along one dimension.
    Input:
        dx: cylic dimension physical units step
        dy: non-cyclic dimension physical units step
        z:  the data
        nperseg: window width
        axes: first position is cyclic axis (we perform no windowing), second
            position is non-cyclic (perform windowing, and get complex coeffs)
    Output:
        f: wavenumbers, in <x units>**-1
        P: power spectrum, in units <data units>**2
    """
    # Checks
    caxis = axes[0]
    taxis = axes[1] # axis for windowing
    print(f'Cyclic dimension in position {caxis}, length {z.shape[caxis]}.')
    print(f'Data to be windowed in position {taxis}, length {z.shape[taxis]}, window length {nperseg}.')
    if caxis<0:
        caxis = z.ndim-caxis
    if taxis<0:
        taxis = z.ndim-taxis
    l = z.shape[taxis]
    r = l % nperseg
    if r>0:
        s = [slice(None) for i in range(z.ndim)]
        s[taxis] = slice(None,-r)
        z = z[s] # slice it up
        print(f'Warning: Trimmed {r} out of {l} points to accommodate length-{nperseg} window.')
        # raise ValueError(f'Window width {nperseg} does not divide axis length {z.shape[axis]}.')
    # Permute
    # Axis where we get *complex* coefficients (i.e. have negative frequencies) in -2 position
    # Axis where we retrieve *real* coefficients in -1 position
    # print('initial:', z.shape)
    z = permute(z, taxis, -1) # put on -1, then will be moved to -2
    query = int(taxis<caxis)
    z = permute(z, caxis-query, -1) # put on -1
    # print('final:', z.shape)
    nflat = z.ndim-2 # we overwrite z, so must save this value!
    z, shape = lead_flatten(z, nflat=nflat) # flatten remaining dimensions
    # Get copsectrum, quadrature spectrum, and powers for each window
    win = window(wintype, nperseg)
    M = z.shape[1] # cyclic dimension is in position 1
    N = z.shape[2] # non-cyclic dimension in position 2
    pm = nperseg//2
    loc = np.arange(pm, N-pm+0.1, pm).astype(int) # jump by half window length
    if len(loc)==0:
        raise ValueError('Window length too big.')
    shape[-2] = M//2
    shape[-1] = pm*2
    P = np.empty((z.shape[0], *shape[-2:])) # power
    for j in range(z.shape[0]):
        Cx = np.empty((loc.size, *shape[-2:])) # have half/window size number of freqs
        for i,l in enumerate(loc):
            # Detrend and compute Fourier transform; note cyclic axis has no trend
            wy = z[j,:,l-pm:l+pm] # apply window
            wy = signal.detrend(wy, type='constant', axis=0) # remove mean
            wy = signal.detrend(wy, type=detrend, axis=1) # remove trend or mean
            # print('cx container:', Cx.shape)
            # print('original data:', z.shape)
            # print('weighted data:', wy.shape)
            coefs = np.fft.rfft2(wy, axes=(1,0))
            coefs = coefs[1:,:] # remove the zero-frequency value; keep both sides of complex transform
            # print('result:', coefs.shape)
            Cx[i,:,:] = np.abs(coefs)**2
        P[j,:,:] = Cx.mean(axis=0)
    # Dimensions
    fx = np.fft.rfftfreq(2*shape[-2]-1) # just the positive-direction Fourier coefs
    fy = np.fft.fftfreq(shape[-1]) # we use the whole thing
    # Fix array
    P = lead_unflatten(P, shape, nflat=nflat)
    # print('initial:', P.shape)
    P = unpermute(P, caxis-query, -2) # put caxis back, accounting for if taxis moves to left
    # print('middle:', P.shape)
    P = unpermute(P, taxis, -1) # put taxis back
    P.max()
    # print('final:', P.shape)
    # Convert frequency to physical units
    # Scale so variance of windows (sum of power spectrum) equals variance of this
    # scale = z.var(axis=axis, keepdims=True)/P.sum(axis=axis, keepdims=True)
    # Scale so power is in proportion to total
    scale = 1.0/P.sum(axis=axes, keepdims=True) # sum over the multiple axes
    return fx/dx, fy/dy, P*scale # power density (sum over grid is 1)

def xspectrum(x, y, nperseg=72, wintype='boxcar', centerphase=np.pi, axis=-1,
        manual=False, detrend='linear'):
    """
    Calculates the cross spectrum for particular windowing technique.
    Uses simply numpy fft.
    """
    if manual:
        # Initial stuff; get integer number of half-overlapped windows
        N = x.size
        pm = nperseg//2
        Nround = pm*(N//pm)
        x, y = x[:Nround], y[:Nround]
        if N-Nround>0: print(f'Points removed: {N-Nround:d}.')
        # Get copsectrum, quadrature spectrum, and powers for each window
        win = window(wintype, nperseg)
        loc = np.arange(pm, Nround-pm+pm//2, pm) # jump by half window length
        shape = (loc.size, pm) # have half window size number of freqs
        Fxx, Fyy, CO, Q = np.empty(shape), np.empty(shape), np.empty(shape), np.empty(shape)
        for i,l in enumerate(loc):
            Cx = np.fft.fft(win*signal.detrend(x[l-pm:l+pm], detrend=detrend))[:pm]
            Cy = np.fft.fft(win*signal.detrend(y[l-pm:l+pm], detrend=detrend))[:pm]
            Fxx[i,:] = np.abs(Cx)**2
            Fyy[i,:] = np.abs(Cy)**2
            CO[i,:] = Cx.real*Cy.real + Cx.imag*Cy.imag
            Q[i,:] = Cx.real*Cy.imag - Cy.real*Cx.imag
        # Get average cospectrum and other stuff, return
        Fxx, Fyy, CO, Q = Fxx.mean(0), Fyy.mean(0), CO.mean(0), Q.mean(0)
        Coh = (CO**2 + Q**2)/(Fxx*Fyy) # coherence
        p = np.arctan2(Q, CO) # phase
        p[p >= centerphase+np.pi] -= 2*np.pi
        p[p < centerphase-np.pi] += 2*np.pi
        freq = np.fft.fftfreq(nperseg)[:pm] # frequency
    else:
        # Just use builtin scipy method
        f, P = signal.csd(x, x, window=wintype,
                return_onesided=True, scaling='spectrum',
                nperseg=nperseg, noverlap=nperseg//2, detrend=detrend, axis=axis)
    return freq, Coh, p

def spectrum2():
    """
    Compute the 2-dimensional Fourier decomposition, and produce power estimates
    with windowing as above.
    """
    raise NotImplementedError()

def autospectrum():
    """
    Uses scipy.signal.welch windowing method to generate an estimate of the
    lagged spectrum.
    """
    raise NotImplementedError()

def autoxspectrum():
    """
    Uses scipy.signal.csd automated method to generate an estimate of the
    lagged cross-spectrum.
    """
    raise NotImplementedError()
