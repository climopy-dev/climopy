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
def rednoise(ntime, a, init=[-1,1], samples=1, mean=0, stdev=1, nested=False):
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
    if not hasattr(samples,'__iter__'):
        samples = [samples]
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

def waves(x, wavelens, phase=None, samples=1):
    """
    Compose array of sine waves.
    Useful for testing performance of filters.
    Input:
        x: iterable or scalar, in which case 'x' is np.arange(0,x)
        wavelens: wavelengths for sine function
    Output:
        data: data composed of waves.
    Notes:
      * 'x' will always be normalized so that wavelength
        is with reference to first step.
      * this make sense because when working with filters, almost
        always need to use units corresponding to axis.
    """
    samples = np.atleast_1d(samples)
    wavelens = np.atleast_1d(wavelens)
    if not hasattr(x, '__iter__'):
        x = np.arange(x)
    data = np.zeros((len(x),*samples)) # user can make N-D array
    data, shape = trail_flatten(data)
    # Get waves
    for i in range(data.shape[-1]):
        if phase is None:
            phis = np.random.uniform(0,2*np.pi,len(wavelens))
        else:
            phis = phase*np.ones([len(wavelens)])
        for wavelen,phi in zip(wavelens,phis):
            data[:,i] += np.sin(2*np.pi*(x/wavelen) + phi)
    data = trail_unflatten(data, shape)
    if len(samples)==1 and samples[0]==1:
        data = data.squeeze()
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
    if build:
        # Repalce regression dimension with best-fit line
        z = z[:,:1] + x*z[:,1:]
    elif stderr:
        # Replace the regression dimension with (slope, standard error)
        f = z[:,:1] + x*z[:,1:] # build the best-fit
        s = np.array(f.shape[:1])
        n = y.shape[1]
        resid = y - f # get residual
        mean = resid.mean(axis=1)
        var = resid.var(axis=1)
        rho = np.sum((resid[:,1:]-mean[:,None])*(resid[:,:-1]-mean[:,None]), axis=1)/((n-1)*var)
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
def eof(data, neof=5):
    """
    Calculates the temporal EOFs, using most efficient method.
    """
    raise NotImplementedError('Not yet implemented.')

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
              pad=True, padvalue=np.nan):
    """
    Apply scipy.signal.lfilter to data. By default this does *not* pad
    ends of data. May keep it this way.
    Input:
        x: data to be filtered
        b: b coefficients (non-recursive component)
        a: scale factor in index 0, followed by a coefficients (recursive component)
        n: number of times to filter data (will go forward-->backward-->forward...)
        axis: axis along which we filter data
    Output:
        y: filtered data
    Notes:
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
    # Capture component that, if filter was non-recursive, does not include
    # data points with clipped edges
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
    if pad:
        y_left  = padvalue*np.ones((y.shape[0], n_2sides+n_left//2))
        y_right = padvalue*np.ones((y.shape[0], n_2sides+n_left//2))
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
def window(wintype, M=100):
    """
    Retrieves weighting function window.
    """
    # Prepare window
    if wintype=='boxcar':
        win = np.ones(M)
    elif wintype=='hanning':
        win = np.hanning(M) # window
    elif wintype=='hamming':
        win = np.hamming(M)
    elif wintype=='blakman':
        win = np.blackman(M)
    elif wintype=='kaiser':
        win = np.kaiser(M)
    else:
        raise ValueError('Unknown window type: %s' % (wintype,))
    return win/win.sum()

def fboxcar(x, k=4, axis=-1): #n=np.inf, kmin=0, kmax=np.inf): #, kscale=1, krange=None, k=None):
    """
    Apply boxcar window in Fourier space.
    Extracts the time series associated with cycle, given by the first k
    Fourier harmonics for the time series.
    """
    # Get fourier transform
    x   = permute(x, axis)
    fft = np.fft.fft(x, axis=-1)
    # Remove frequencies outside range
    # FFT will have some error and give non-zero imaginary components, but
    # we just naively cast to real
    fft[...,0] = 0
    fft[...,k+1:-k] = 0
    return unpermute(np.fft.ifft(fft).real, axis)
    # # Naively remove certain frequencies
    # # Should ignore first coefficient, the mean
    # f = np.where((freq[1:]>=kmin) | (freq[1:]<=kmax))
    # # Gets indices of n largest values
    # if n==np.inf:
    #     fremove = f
    # else:
    #     p = np.abs(fft)**2
    #     fremove = f[np.argpartition(p, -n)[-n:]]
    # # And filter back
    # # FFT should be symmetric, so remove locations at corresponding negative freqs
    # fft[fremove+1], fft[-fremove-1] = 0, 0

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
def spectrum(x, M=72, wintype='boxcar', param=None, axis=-1):
    """
    Gets the spectral decomposition for particular windowing technique.
    Uses simple numpy fft.
    """
    # Initital stuff; get integer number of half-overlapped windows
    N = x.size
    pm = M//2
    Nround = pm*(N//pm)
    x = x[:Nround]
    if N-Nround>0: print(f'Points removed: {N-Nround:d}.')
    # Get copsectrum, quadrature spectrum, and powers for each window
    win = window(wintype, M)
    loc = np.arange(pm, Nround-pm+pm//2, pm) # jump by half window length
    Cx = np.empty((loc.size, pm)) # have half/window size number of freqs
    for i,l in enumerate(loc):
        # numpy fft gives power A+Bi, so want sqrt(A^2 + B^2)/2 for +/- wavenumbers
        Cx[i,:] = np.abs(np.fft.fft(win*signal.detrend(x[l-pm:l+pm]))[:pm])**2
    freq = np.fft.fftfreq(M)[:pm] # frequency
    return freq, Cx.mean(axis=0)

def xspectrum(x, y, M=72, wintype='boxcar', param=None, centerphase=np.pi):
    """
    Calculates the cross spectrum for particular windowing technique.
    Uses simply numpy fft.
    """
    # Iniital stuff; get integer number of half-overlapped windows
    N = x.size
    pm = M//2
    Nround = pm*(N//pm)
    x, y = x[:Nround], y[:Nround]
    if N-Nround>0: print(f'Points removed: {N-Nround:d}.')

    # Get copsectrum, quadrature spectrum, and powers for each window
    win = window(wintype, M)
    loc = np.arange(pm, Nround-pm+pm//2, pm) # jump by half window length
    shape = (loc.size, pm) # have half window size number of freqs
    Fxx, Fyy, CO, Q = np.empty(shape), np.empty(shape), np.empty(shape), np.empty(shape)
    for i,l in enumerate(loc):
        Cx = np.fft.fft(win*signal.detrend(x[l-pm:l+pm]))[:pm]
        Cy = np.fft.fft(win*signal.detrend(y[l-pm:l+pm]))[:pm]
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
    freq = np.fft.fftfreq(M)[:pm] # frequency
    return freq, Coh, p

def autospectrum():
    """
    Uses scipy.signal.welch windowing method to generate an estimate of the spectrum.
    """
    return

def autoxspectrum():
    """
    Uses scipy.signal.cpd automated method to generate cross-spectrum estimate.
    """
    return

