#!/usr/bin/env python3
"""
Includes objective analysis-related functions.
"""
import os
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import scipy.optimize as optimize
from . import const # as its own module
from .base import *

#------------------------------------------------------------------------------
# Statistics
#------------------------------------------------------------------------------
def gaussian(mean, sigma):
    """
    Returns sample points on Gaussian curve.
    """
    norm = stats.norm(loc=mean, scale=sigma)
    x = np.linspace(norm.ppf(0.0001), norm.ppf(0.9999), 1000) # get x through percentile range
    pdf = norm.pdf(x)
    return x, pdf

#------------------------------------------------------------------------------
# Changing phase space (to EOFs, spectral decomposition, etc.)
#------------------------------------------------------------------------------
def rednoise(ntime, nsamples, a, mean=0, stdev=1, nested=False):
    """
    Creates artificial red noise time series, i.e. a weighted sum of random perturbations.
    Equation is: x(t) = a*x(t-dt) + b*eps(t)
    where a is the lag-1 autocorrelation and b is a scaling term.
     * Output will have shape ntime by nsamples.
     * Enforce that the first timestep always equals the 'starting' position.
     * Use 'nested' flag to control which algorithm to use.
    """
    #--------------------------------------------------------------------------#
    # Initial stuff
    ntime -= 1 # exclude the initial timestep
    output = np.empty((ntime+1,nsamples))
    output[0,:] = 0 # initiation
    b = (1-a**2)**0.5 # from OA class
    #--------------------------------------------------------------------------#
    # Nested loop
    for i in range(nsamples):
        eps = np.random.normal(loc=0, scale=1, size=ntime)
        for t in range(1,ntime+1):
            output[t,i] = a*output[t-1,i] + b*eps[t-1]
    #--------------------------------------------------------------------------#
    # This formula was nonsense; only works for autocorrelation 1
    # return np.concatenate((np.ones((1,nsamples))*mean,
    #   mean + np.random.normal(loc=0, scale=stdev, size=(ntime-1,nsamples)).cumsum(axis=0)), axis=0).squeeze()
    #--------------------------------------------------------------------------#
    # Trying to be fancy, just turned out fucking way slower
    # aseries = b*np.array([a**(ntime-i) for i in range(1,ntime+1)])
    # for i in range(nsamples):
    #     eps = np.random.normal(loc=0, scale=1, size=ntime)
    #     vals = eps[:,None]@aseries[None,:] # matrix for doing math on
    #     output[1:,i] = [np.trace(vals,ntime-i) for i in range(1,ntime+1)]
    return mean + stdev*output.squeeze() # rescale to have specified stdeviation/mean

def rednoisefit(data, nlag=None, axis=-1, lag1=False, series=False, verbose=False):
    """
    Returns a best-fit red noise autocorrelation spectrum.
    * If 'series' is True, return the red noise spectrum full series.
    * If 'series' is False, return just the *timescale* associated with the red noise spectrum.
    * If 'lag1' is True, just return the lag-1 version.
    Go up to nlag-timestep autocorrelation.
    """
    #--------------------------------------------------------------------------#
    # Initial stuff
    if nlag is None:
        raise ValueError(f"Must declare \"nlag\" argument; number of points to use for fit.")
    nflat = data.ndim-1
    data = lead_flatten(permute(data, axis), nflat)
    # if len(time)!=data.shape[-1]:
    #     raise ValueError(f"Got {len(time)} time values, but {data.shape[-1]} timesteps for data.")
    #--------------------------------------------------------------------------#
    # First get the autocorrelation spectrum, and flatten leading dimensions
    # Dimensions will need to be flat because we gotta loop through each 'time series' and get curve fits.
    # time = time[:nlag+1] # for later
    autocorrs = autocorr(data, nlag, axis=-1, verbose=verbose)
    #--------------------------------------------------------------------------#
    # Next iterate over the flattened dimensions, and perform successive curve fits
    ndim = data.shape[-1] if series else 1
    shape = (*data.shape[:-1], ndim) # when we perform *unflattening*
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
    return unpermute(lead_unflatten(output, shape, nflat), axis)

def eof(data, neof=5):
    """
    Calculates the temporal EOFs, using most efficient method.
    """
    return

def autocorr(data, nlag=None, lag=None, verbose=False, axis=0):
    """
    Gets the autocorrelation spectrum at successive lags.
      * Estimator is: ((n-k)*sigma^2)^-1 * sum_i^(n-k)[X_t - mu][X_t+k - mu]
        See: https://en.wikipedia.org/wiki/Autocorrelation#Estimation
      * By default, includes lag-zero values; if user just wants single lag
        will, however, throw out those values.
    """
    #--------------------------------------------------------------------------#
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
    var = data.var(axis=-1, keepdims=False) # this is divided by the summation term, so should have annihilated axis
    #--------------------------------------------------------------------------#
    # Loop through lags
    # Include an extra lag under some circumstances
    if nlag is None and lag==0:
        autocorrs = np.ones((*data.shape[:-1],1))
    elif nlag is None:
        autocorrs = np.sum((data[...,:-lag]-mean)*(data[...,lag:]-mean),axis=-1)/((naxis-lag)*var)
        autocorrs = autocorrs[...,None] # add dimension back in
    else:
        autocorrs = np.empty((*data.shape[:-1], nlag+1)) # will include the zero-lag autocorrelation
        autocorrs[...,0] = 1 # lag-0 autocorrelation
        for i,lag in enumerate(range(1,nlag+1)):
            autocorrs[...,i+1] = np.sum((data[...,:-lag]-mean)*(data[...,lag:]-mean),axis=-1)/((naxis-lag)*var)
    return unpermute(autocorrs, axis)

def lowpass(x, k=4, axis=-1): #n=np.inf, kmin=0, kmax=np.inf): #, kscale=1, krange=None, k=None):
    """
    Extracts the time series associated with cycle, given by the first k
    Fourier harmonics for the time series.
    Does not apply any windowing.
    """
    # Naively remove certain frequencies
    # p = np.abs(fft)**2
    # f = np.where((freq[1:]>=kmin) | (freq[1:]<=kmax))
    #     # should ignore first coefficient, the mean
    # if n==np.inf: fremove = f
    # else: fremove = f[np.argpartition(p, -n)[-n:]]
    #         # gets indices of n largest values

    # # And filter back
    # fft[fremove+1], fft[-fremove-1] = 0+0j, 0+0j
    #     # fft should be symmetric, so remove locations at corresponding negative freqs

    # Get fourier transform
    # x = np.rollaxis(x, axis, x.ndim)
    x = permute(x, axis)
    fft = np.fft.fft(x, axis=-1)
    # freq = np.fft.fftfreq(x.size)*scale

    # Remove the edge case frequencies
    fft[...,0] = 0
    fft[...,k+1:-k] = 0
    # return np.rollaxis(np.fft.ifft(fft).real, x.ndim-1, axis)
    return unpermute(np.fft.ifft(fft).real, axis)
        # FFT will have some error, and give non-zero imaginary components;
        # just naively cast to real

def lanczos(alpha, J):
    """
    Lanczos filtering of data; gives an abrupt high-frequency (low wavenumber)
    cutoff at omega = alpha*pi, the number of datapoints needed.
    """
    C0 = alpha # integral of cutoff-response function is alpha*pi/pi
    Ck = np.sin(alpha*np.pi*np.arange(1,J+1))*(1/(np.pi*np.arange(1,J+1)))
    Cktilde = Ck*np.sin(np.pi*np.arange(1,J+1)/J)/(np.pi*np.arange(1,J+1)/J)
    filt = np.concatenate((np.flipud(Cktilde), np.array([C0]), Cktilde))
    return filt/filt.sum()
    # for j,J in enumerate(Jsamp):
    #     pass
    # R = lambda Cfunc, omega: C0 + np.sum(
    #         Cfunc(alpha)*np.cos(omega*np.arange(1,J+1))
    #         ) # C_tau * cos(omega*tau)
    # omega = np.linspace(0,np.pi,1000)
    # Romega = np.empty(omega.size)
    # Romegatilde = np.empty(omega.size)
    # for i,o in enumerate(omega): Romega[i] = R(Ck, o)
    # for i,o in enumerate(omega): Romegatilde[i] = R(Cktilde, o)
    # a1.plot(omega, Romega, color=colors[j], label=('J=%d' % J))
    # a2.plot(omega, Romegatilde, color=colors[j], label=('J=%d' % J))
    # return

def butterworth(order, *args, **kwargs):
    """
    Return Butterworth filter.
    Wraps around the builtin scipy method.
    """
    kwargs.update({'analog':True}) # must be analog or something
    b, a = signal.butter(order-1, *args, **kwargs)
    return b/a # gives numerate/demoninator of coefficients; we just want floating points
        # an order of N will return N+1 denominator values; we fix this

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
        win = np.kaiser(M, param)
    elif wintype=='lanczos':
        win = lanczos(M, param)
    elif wintype=='butterworth':
        win = butterworth(M, param)
    else:
        raise ValueError('Unknown window type: %s' % (wintype,))
    return win/win.sum()

def spectrum(x, M=72, wintype='boxcar', param=None, axis=-1):
    """
    Gets the spectral decomposition for particular windowing technique.
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
        Cx[i,:] = np.abs(np.fft.fft(win*signal.detrend(x[l-pm:l+pm]))[:pm])**2
        # numpy fft gives power A+Bi, so want sqrt(A^2 + B^2)/2 for +/- wavenumbers
    freq = np.fft.fftfreq(M)[:pm] # frequency
    return freq, Cx.mean(axis=0)

def cspectrum(x, y, M=72, wintype='boxcar', param=None, centerphase=np.pi):
    """
    Calculates the cross spectrum for particular windowing technique; kwargs
    are passed to scipy.signal.cpd
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

def autocspectrum():
    """
    Uses scipy.signal.cpd automated method to generate cross-spectrum estimate.
    """
    return

#------------------------------------------------------------------------------#
# TODO Finish parsing this; was copied from sst project
# Gets the significance accounting for autocorrelation or something
#------------------------------------------------------------------------------#
# Simple function for getting error, accounting for autocorrelation
# def error(r):
#     """
#     Manually calculate the error according to Wilks definition; compare to bse.
#     Proves that the bse parameter is equal to Equation 7.18b in Wilks.
#     """
#     # Use fittedvalues and resid attributes
#     # xfact = (x**2).sum()/(r.nobs*(x-x.mean())**2).sum() # for constant, a
#     # x = r.model.exog[:,-1] # exogeneous variable
#     # xfact = 1/((x-x.mean())**2).sum()
#     se2 = (r.resid.values**2).sum()/(r.nobs-2) # Eq 7.9, Wilks
#     xfact2 = 12/(r.nobs**3-r.nobs) # Eq 3, Thompson et. al 2015
#     sigma = (se2**0.5)*(xfact2**0.5)
#     print('Provided: %.10f, manually calculated: %.10f' % (r.bse.x1, sigma))
#     return

#------------------------------------------------------------------------------#
# Function for performing regression, along with significance
# on regression coefficient or something
# Below uses statsmodels, but slower for small vectors
# X = sm.add_constant(T.index.values) # use the new index values
# r = sm.OLS(T[name,region_get], X).fit() # regress column [name] against X
# scale = ((r.nobs-2)/(r.nobs*((1-autocorr)/(1+autocorr))-2))**0.5
# L[name,region] = [unit*r.params.x1, scale*unit*r.bse.x1]
# Below uses np.corrcoef, but is slower than pearsonr
# autocorr = np.corrcoef(r.resid.values[1:], r.resid.values[:-1])[0,1] # outputs 2by2 matrix
# def regress(x, y, unit=10, sigma=False, ignorenan=False):
#     """
#     Gets regression results with fastest methods possible.
#     See the IPython noteobok for comparisons.
#     """
#     # NaN check
#     x, y = x.squeeze(), y.squeeze()
#     if y.ndim>1:
#         raise ValueError('y is not 1-dimensional.')
#     nan = np.isnan(y)
#     if nan[0] and ignorenan:
#         # If we are getting continuous trends, don't bother computing this one, because
#         # we will also get trends from x[1:], x[2:], etc.
#         return np.nan, np.nan, np.nan, np.nan
#     if nan.any():
#         # Filter out NaNs, get regression on what remains
#         x, y = x[~nan], y[~nan]
#     if x.size<5:
#         # Cannot get decent sigma estimate, in this case
#         return np.nan, np.nan, np.nan, np.nan
#     # Regress, and get sigma if requested
#     # First value is estimated change K through record, second is K/unit
#     if sigma:
#         p, V = np.polyfit(x, y, deg=1, cov=True)
#         resid = y - (x*p[0] + p[1]) # very fast step; don't worry about this one
#         autocorr, _ = st.pearsonr(resid[1:], resid[:-1])
#         scale = (x.size-2)/(x.size*((1-autocorr)/(1+autocorr))-2)
#             # scale factor from Thompson et. al, 2015, Quantifying role of internal variability...
#         stderr = np.sqrt(V[0,0]*scale)
#         return (x[-1]-x[0])*p[0], unit*p[0], (x[-1]-x[0])*stderr, unit*stderr
#     else:
#         p = np.polyfit(x, y, deg=1)
#         return (x[-1]-x[0])*p[0], unit*p[0]

#-------------------------------------------------------------------------------
# TODO: Finish harvesting this original function written for the Objective
# Analysis assignment. Consider deleting it.
#-------------------------------------------------------------------------------
# def spectral(fnm, nm, data, norm=True, win=501,
#             freq_scale=1, scale='days',
#             xlog=True, ylog=False, mc='k',
#             xticks=None,
#             rcolors=('C3','C6'), pcolors=('C0','C1'), alpha=0.99, marker=None,
#             xlim=None, ylim=None, # optional override
#             linewidth=1.5,
#             red_discrete=True, red_contin=True, manual=True, welch=True):
#     '''
#     Spectral transform function. Needs work; was copied from OA
#     assignment and right now just plots a bunch of stuff.
#     '''
#     # Iniital stuff
#     N = len(data)
#     pm = int((win-1)/2)
#     fig, a = plt.subplots(figsize=(15,5))
#
#     # Confidence intervals
#     dof_num = 1.2*2*2*(N/(win/2))
#     trans = 0.5
#     exp = 1
#     F99 = stats.f.ppf(1-(1-alpha)**exp,dof_num,1000)
#     F01 = stats.f.ppf((1-alpha)**exp,dof_num,1000)
#     print('F stats:',F01,F99)
#     rho = np.corrcoef(data[1:],data[:-1])[0,1]
#     kr = np.arange(0,win//2+1)
#     fr = freq_scale*kr/win
#
#     def xtrim(f):
#         if xlim is None:
#             return np.ones(f.size, dtype=bool)
#         else:
#             return ((f>=xlim[0]) & (f<=xlim[-1]))
#
#     # Power spectra
#     if manual:
#         label = 'power spectrum'
#         if welch: label = 'manual method'
#         # Now, manual method with proper overlapping etc.
#         if False:
#             data = data[:(N//pm)*pm]
#         loc = np.linspace(pm,N-pm-1,2*int(np.round(N/win))).round().astype(int) # sample loctaions
#         han = np.hanning(win)
#         han = han/han.sum()
#         phi = np.empty((len(loc),win//2))
#         for i,l in enumerate(loc):
#             pm = int((win-1)/2)
#             C = np.fft.fft(han*signal.detrend(data[l-pm:l+pm+1]))
#             phii = np.abs(C)**2/2
#             phii = 2*phii[1:win//2+1]
#             phi[i,:] = phii
#         phi = phi.mean(axis=0)
#         print('phi sum:',phi.sum())
#         f = np.fft.fftfreq(win)[1:win//2+1]*freq_scale
#         if norm: phi = phi/phi.sum()
#         f, phi = f[xtrim(f)], phi[xtrim(f)] # trim
#         a.plot(f, phi, label=label,
#                mec=mc, mfc=mc, mew=linewidth,
#                marker=marker, color=pcolors[0], linewidth=linewidth)
#         if xlim is None: xlim = ((f*freq_scale).min(), (f*freq_scale).max())
#         if ylim is None: ylim = ((phi.min()*0.95, phi.max()*1.05))
#
#     if welch:
#         label = 'power spectrum'
#         if manual: label = 'welch method'
#         # Welch
#         fw, phi_w = signal.welch(data, nperseg=win, detrend='linear', window='hanning', scaling='spectrum',
#                               return_onesided=False)
#         fw, phi_w = fw[1:win//2+1]*freq_scale, phi_w[1:win//2+1]
#         if norm: phi_w = phi_w/phi_w.sum()
#         fw, phi_w = fw[xtrim(fw)], phi_w[xtrim(fw)] # trim
#         print('phiw sum:',phi_w.sum())
#         a.plot(fw, phi_w, label=label,
#               mec=mc, mfc=mc, mew=linewidth,
#                marker=marker, color=pcolors[-1], linewidth=linewidth)
#         if xlim is None: xlim = ((fw).min(), (fw).max())
#         if ylim is None: ylim = (phi_w.min()*0.95, phi_w.max()*1.05)
#
#     # Best fit red noise spectrum
#     if red_discrete:
#         print('Autocorrelation',rho)
#         phi_r1 = (1-rho**2)/(1+rho**2-2*rho*np.cos(kr*np.pi/(win//2)))
#         print('phi_r1 sum:',phi_r1.sum())
#         if norm: phi_r1 = phi_r1/phi_r1.sum()
#         frp, phi_r1 = fr[xtrim(fr)], phi_r1[xtrim(fr)]
#         a.plot(fr[xtrim(fr)], phi_r1, label=r'red noise, $\rho(\Delta t)$',
#                marker=None, color=rcolors[0], linewidth=linewidth)
#         a.plot(frp, phi_r1*F99, linestyle='--',
#                marker=None, alpha=trans, color=rcolors[0], linewidth=linewidth)
#
#     # Alternate best fit
#     if red_contin:
#         Te = -1/np.log(rho)
#         omega = (kr/win)*np.pi*2
#         phi_r2 = 2*Te/(1+(Te**2)*(omega**2))

#         print('phi_r2 sum:',phi_r2.sum())
#         if norm: phi_r2 = phi_r2/phi_r2.sum()
#         frp, phi_r2 = fr[xtrim(fr)], phi_r2[xtrim(fr)]
#         a.plot(frp, phi_r2, label=r'red noise, $T_e$',
#                marker=None, color=rcolors[1], linewidth=linewidth)
#         a.plot(frp, phi_r2*F99, linestyle='--',
#                marker=None, alpha=trans, color=rcolors[-1], linewidth=linewidth)
#     # Variance
#     print('true variance:',data.std()**2)
#     # Figure formatting
#     a.legend()
#     if ylog:
#         a.set_yscale('log')
#     if xlog:
#         a.set_xscale('log')
#     a.set_title('%s power spectrum' % nm)
#     a.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%5.3g'))
#     a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
#     if xticks is None: xticks = a.get_xticks()
#     my.format(a, xlabel=('frequency (%s${}^{-1}$)' % scale), ylabel='proportion variance explained',
#              xlim=xlim, ylim=ylim, xticks=xticks)
#     suffix = 'pdf'
#     fig.savefig('a5_' + fnm + '.' + suffix, format=suffix, dpi='figure')
#     plt.show()
