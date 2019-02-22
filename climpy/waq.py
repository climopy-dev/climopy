#!/usr/bin/env python3
"""
Module for performing `Noboru Nakamura's <https://geosci.uchicago.edu/people/noboru-nakamura/>`_
finite-amplitude wave activity quantification. Incorporates code developed
by `Clare Huang <https://github.com/csyhuang/hn2016_falwa/>`_.

Warnings
--------
This module is very old and needs work!

Todo
----
Incorporate xarray Datasets into this. Incorporate Noboru's grad
student's Github code.
"""
# Imports
import numpy as np
from .arraytools import *
from . import const

#------------------------------------------------------------------------------#
# Grid description, useful for WAQ analysis
#------------------------------------------------------------------------------#
# @dataclass # consider for python3.7?
class GridDes(object):
    """
    For storing latitude/longitude grid properties. Assumes global grid, and
    borders halfway between each grid center.
    """
    def __init__(self, lon, lat):
        # First, guess cell widths and edges
        lon, lat = lon.astype(np.float32), lat.astype(np.float32)
        dlon1, dlon2 = lon[1]-lon[0], lon[-1]-lon[-2]
        dlat1, dlat2 = lat[1]-lat[0], lat[-1]-lat[-2]
        self.latb = np.concatenate((lat[:1]-dlat1/2, (lat[1:]+lat[:-1])/2, lat[-1:]+dlat2/2))
        self.lonb = np.concatenate((lon[:1]-dlon1/2, (lon[1:]+lon[:-1])/2, lon[-1:]+dlon2/2))
        self.latc, self.lonc = lat.copy(), lon.copy()

        # Use corrections for dumb grids with 'centers' at poles
        if lat[0]==-90:
            self.latc[0], self.latb[0] = -90 + dlat1/4, -90
        if lat[-1]==90:
            self.latc[-1], self.latb[-1] = 90 - dlat2/4, 90

        # Corrected grid widths (cells half as tall near pole)
        self.dlon = self.lonb[1:]-self.lonb[:-1]
        self.dlat = self.latb[1:]-self.latb[:-1]
        if lat[0]==-90:
            self.dlat[0] /= 2
        if lat[-1]==90:
            self.dlat[-1] /= 2

        # Radians
        self.phic, self.phib, self.dphi = self.latc*np.pi/180, self.latb*np.pi/180, self.dlat*np.pi/180
        self.thetac, self.thetab, self.dtheta = self.lonc*np.pi/180, self.lonb*np.pi/180, self.dlon*np.pi/180

        # Area weights (function of latitude only). Close approximation to area,
        # since cosine is extremely accurate. Also make lon by lat, so
        # broadcasting rules can apply.
        self.weights = self.dphi[None,:]*self.dtheta[:,None]*np.cos(self.phic[None,:])
        self.areas = self.dphi[None,:]*self.dtheta[:,None]*np.cos(self.phic[None,:])*(const.a**2)
        # areas = dphi*dtheta*np.cos(phic)*(const.a**2)[None,:]

#------------------------------------------------------------------------------
# Wave activity stuff
#------------------------------------------------------------------------------
def eqlat(lon, lat, q, skip=10, sigma=None, fix=False): #n=1001, skip=10):
    """
    Get series of equivalent latitudes corresponding to PV levels evenly
    sampled from distribution of sorted PVs. Solves the equation:

    .. math::

        Area == integral_(-pi/2)^x a^2*2*pi*cos(phi) dphi

    which is the area occupied by PV zonalized below a given contour.

    Returns: band, the equivalent latitude; qband, its associated Q threshold;
    w, grid weights (for future use).
    """
    # Initial stuff
    # Delivers grid areas, as function of latitude
    areas = GridDes(lon, lat).areas

    # Flatten
    q, shape = lead_flatten(q, 2) # gives current q, and former shape

    # And consider mass-weighting this stuff
    if sigma is not None:
        # Note that at least singleton dimension is added
        mass = lead_flatten(sigma, 2)*areas[...,None]
        # Get cumulative mass from pole
        masscum = mass.cumsum(axis=1).sum(axis=0, keepdims=True)

    # Determing Q contour values for evaluating eqlat (no interpolation; too slow)
    K = q.shape[-1] # number of extra dims
    N = (np.prod(shape[:2])-1)//skip + 1 # e.g. // counts number of complete length-<skip> blocks after index 0, then add 0 position
    offset = (np.prod(shape[:2]) % skip)//2 # want to center the list if possible; e.g. [0,1,2,3,4,5,6,7,8], skip=3, will give [1,4,7]
    # N = N-2 # want to eliminate start/end, in case? no should be fine

    # Solve equivalent latitudes; options include mass-weighted solution with sigma, or area-weighted
    bands, q_bands = np.empty((1, N, K)), np.empty((1, N, K))
    for k in range(K): # iterate through extra dimensions
        # Iterate through Q contours
        q_bands[0,:,k] = np.sort(q[:,:,k], axis=None)[offset::skip] # test q-values
        for n in range(N):
            f = (q[:,:,k] <= q_bands[0,n,k]) # filter
            if sigma is None: # normal weighting
                # Get sine of eqlat, and correct for rounding errors
                sin = areas[f].sum()/(2*np.pi*const.a**2)-1
                if sin>1: sin = 1
                if sin<-1: sin = -1
                bands[0,n,k] = np.arcsin(sin)*180/np.pi
            else: # mass weighting
                # Interpolate to latitude of mass contour
                massk, masscumk = mass[:,:,k], masscum[:,:,k].squeeze() # latter is coordinate
                mass = massk[f].sum() # total mass below Q
                bands[0,n,k] = np.interp(mass, masscumk, lat) # simple interpolation to one point

    # Reshape, and return
    return lead_unflatten(bands, shape, 2), lead_unflatten(q_bands, shape, 2)

def waqlocal(lon, lat, q,
        nh=True, skip=10):
    """
    Get local wave activity measure. Input `skip` is
    the interval of sorted q you choose (passed to eqlat).
    """
    # Grid considerations
    if nh:
        lat, q = -np.flipud(lat), -np.flip(q, axis=1)
            # negated q, so monotonically increasing "northward"
    grid = GridDes(lon, lat)
    areas, dphi, phib = grid.areas, grid.dphi, grid.phib
    integral = const.a*phib[None,:]

    # Flatten (eqlat can do this, but not necessary here)
    q, shape = lead_flatten(q, 2)

    # Get equivalent latiitudes
    bands, q_bands = eqlat(lon, lat, q, skip=skip) # note w is just lonbylat
    L, M, N, K = q.shape[0], q.shape[1], bands.shape[1], q.shape[-1] # number of eqlats, number of extra dims

    # Get local wave activity measure, as simple line integrals
    waq = np.empty((L, M, K))
    percent = 0
    for k in range(K):
        if (k/K)>(.01*percent):
            print('%d%% finished' % (100*k/K,))
            percent = percent+10
        # Loop through each contour
        waq_k = np.empty((L,N))
        for n in range(N):
            # Setup, large areas
            band = bands[0,n,k]*np.pi/180
            if np.isnan(band): # happens if contours intersect at edge
                waq_k[:,n] = np.nan
            else:
                anom = q[:,:,k] - q_bands[0,n,k]
                f_pos = (anom >= 0) & (phib[None,1:] < band) # high anomalies at low lat (below top graticule)
                f_neg = (anom < 0) & (phib[None,:-1] >= band) # low anomalies at high lat (above bottom graticule)
                # See if band is sandwiched between latitudes
                mid = np.where((phib[:-1] <= band) & (phib[1:] > band))[0] # want scalar id (might be zero)
                if mid.size>0:
                    f_pos_mid = (anom[:,mid] >= 0) # longitudes where positive
                    p_int, m_int = const.a*(band-phib[mid]), const.a*(phib[mid+1]-band)
                        # partial integrals, positive and negative
                for l in range(L):
                    # Get individual integral
                    integral_pos = (anom[l,f_pos[l,:]]*integral[:,f_pos[l,:]]).sum()
                    integral_neg = -(anom[l,f_neg[l,:]]*integral[:,f_neg[l,:]]).sum() # minus a negative
                    if mid.size>0:
                        if f_pos_mid[l]: # if positive at this latitude, we add anomaly
                            integral_extra = anom[l,mid]*p_int
                        else: # else, subtract it
                            integral_extra = -anom[l,mid]*m_int
                    else:
                        integral_extra = 0
                    # Put it all together
                    waq_k[l,n] = integral_pos + integral_neg + integral_extra # no normalization here
        # Interpolate
        for l in range(L):
            waq[l,:,k] = np.interp(lat, bands[0,:,k], waq_k[l,:])

    # Return
    if nh: waq = np.flip(waq, axis=1)
    return lead_unflatten(waq, shape)

def waq(lon, lat, q, sigma=None, omega=None,
        nh=True, skip=10): #, ignore=None): #N=1001, ignore=None):
    """
    Get finite-amplitude wave activity. Input `omega` is (quantity being
    integrated), and `skip` is interval of sorted `q` you choose. See
    :cite:`nakamura_finite-amplitude_2010` for details.

    .. bibliography:: ../refs.bib
    """
    # Grid considerations
    if nh:
        lat, q = -np.flipud(lat), -np.flip(q, axis=1)
        if omega is not None: omega = -np.flip(omega, axis=1)
        if sigma is not None: sigma = np.flipd(sigma, axis=1)
        # negated q/omega, so monotonically increasing "northward"
    grid = GridDes(lon, lat)
    areas, dphi, phib = grid.areas, grid.dphi, grid.phib

    # Flatten (eqlat can do this, but not necessary here)
    q, shape = lead_flatten(q, 2)
    if omega is not None: omega, _ = lead_flatten(omega, 2)
    if sigma is not None: sigma, _ = lead_flatten(sigma, 2)

    # Get equivalent latiitudes
    bands, q_bands = eqlat(lon, lat, q, sigma=sigma, skip=skip) # note w is just lonbylat
        # will infer area weights, to get equivalent latitude
    M, N, K = q.shape[1], bands.shape[1], q.shape[-1] # number of eqlats, number of extra dims

    # Get activity
    waq = np.empty((1, M, K))
    percent = 0
    for k in range(K):
        if (k/K)>(.01*percent):
            print('%d%% finished' % (percent,))
            percent = percent+10
        # Loop through each contour
        waq_k = np.empty(N)
        for n in range(N): #i, bandki in enumerate(bandk[0,:,k]): #, q_bandki) in enumerate(zip(bandk, q_bandk)):
            # First, main blocks
            band = bands[0,n,k]*np.pi/180
            if np.isnan(band):
                waq_k[n] = np.nan
            else:
                # anom = q[:,:,k] - q_bands[0,n,k]
                qk, Qk = q[:,:,k], q_bands[0,n,k] # should be identical, since positive/negative regions have same area by construction
                if omega is None:
                    qint = q[:,:,k] # the thing being integrated
                else:
                    qint = omega[:,:,k]
                # f_pos = (anom >= 0) & (phib[None,1:] < band) # high anomalies at low lat (below top graticule)
                # f_neg = (anom < 0) & (phib[None,:-1] >= band) # low anomalies at high lat (above bottom graticule)
                # integral = (anom[f_pos]*areas[f_pos]).sum() - (anom[f_neg]*areas[f_neg]).sum() # minus a negative
                f_pos = (qk >= Qk) & (phib[None,1:] < band) # high anomalies at low lat (below top graticule)
                f_neg = (qk < Qk) & (phib[None,:-1] >= band) # low anomalies at high lat (above bottom graticule)
                integral = (qint[f_pos]*areas[f_pos]).sum() - (qint[f_neg]*areas[f_neg]).sum() # minus a negative
                # Next, account for tiny pieces along equivalent latitude cells
                mid = np.where((phib[:-1] <= band) & (phib[1:] > band))[0] # want scalar id
                try: mid = mid[0]
                except IndexError:
                    integral_extra = 0
                else:
                    # f_pos_mid = (anom[:,mid] >= 0) # longitudes where positive
                    f_pos_mid = (qk[:,mid] >= Qk) # longitudes where positive
                    p_dphi, m_dphi = (
                            np.cos((band+phib[mid])/2)*(band-phib[mid]), # positive, low lat
                            np.cos((band+phib[mid+1])/2)*(phib[mid+1]-band) # negative, high lat
                            )
                    integral_extra = (
                            qint[f_pos_mid,mid].sum()*(areas[mid]*m_dphi/dphi[mid])
                            - qint[~f_pos_mid,mid].sum()*(areas[mid]*p_dphi/dphi[mid])
                            )
                # Put it all together
                waq_k[n] = (integral + integral_extra)/(2*np.pi*const.a*np.cos(band))
        # Interpolate
        nanfilt = np.isnan(waq_k)
        if sum(~nanfilt)==0:
            print('Warning: no valid waqs calculated for k %d.' % k)
            waq[0,:,k] = np.nan
        else:
            waq[0,:,k] = np.interp(grid.latc, bands[0,~nanfilt,k], waq_k[~nanfilt])

    # Return
    if nh:
        waq = np.flip(waq, axis=1)
    return lead_unflatten(waq, shape)

