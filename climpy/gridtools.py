#!/usr/bin/env python3
"""
Various geographic utilities. Includes calculus on spherical geometry
and some other stuff. Padding arrays and whatnot.

Todo
----
Update this for pairing with xarray! Can add useful tools, e.g. shifting
longitudes and stuff, with a "standardize" function.
"""
# Imports
import numpy as np
from . import const
# TODO: SST project scripts will fail because nc() removed. Must be updated.
# # Enforce longitude ordering convention
# # Not always necessary, but this is safe/fast; might as well standardize
# values = data.lon.values-720 # equal mod 360
# while True: # loop only adds 360s to longitudes
#     filter_ = values<lonmin
#     if filter_.sum()==0: # once finished, write new longitudes and roll
#         roll = values.argmin()
#         data = data.roll(lon=-roll)
#         data['lon'] = np.roll(values, -roll)
#         break
#     values[filter_] += 360
#
# # Make latitudes monotonic (note extracting values way way faster)
# try: data.lat.values[1]
# except IndexError:
#     pass
# else:
#     if data.lat.values[0]>data.lat.values[1]:
#         data = data.isel(lat=slice(None,None,-1))
# # Fix precision of time units. Some notes:
# # 1) Sometimes get weird useless round-off error; convert to days, then
# #    restore to numpy datetime64[ns] because xarray seems to require it
# # 2) Ran into mysterious problem where dataset could be loaded, but then
# #    could not be saved because one of the datetimes wasn't serializable...
# #    this was in normal data, the CCSM4 CMIP5 results, made no sense; range
# #    was 1850-2006
# if 'time' in data.indexes:
#     data['time'] = data.time.values.astype('datetime64[D]').astype('datetime64[ns]')

#------------------------------------------------------------------------------#
# Standardize Dataset/DataArray
# Idea is should work similar to CDO; detect data with X/Y/Z/T dimensions
# by checking attributes, or with user specification
#------------------------------------------------------------------------------#
def standardize():
    """
    Standardize dimension names on xarray datasets, cyclic re-ordering,
    interpolation to poles, and more.

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError

#------------------------------------------------------------------------------#
# Basic geographic stuff
#------------------------------------------------------------------------------#
def geopad(lon, lat, data, nlon=1, nlat=0):
    """
    Returns array padded circularly along lons, and over the earth pole,
    for finite difference methods.
    """
    # Get padded array
    if nlon>0:
        pad  = ((nlon,nlon),) + (data.ndim-1)*((0,0),)
        data = np.pad(data, pad, mode='wrap')
        lon  = np.pad(lon, nlon, mode='wrap') # should be vector
    if nlat>0:
        if (data.shape[0] % 2)==1:
            raise ValueError('Data must have even number of longitudes, if you wish to pad over the poles.')
        data_append = np.roll(np.flip(data, axis=1), data.shape[0]//2, axis=0)
            # data is now descending in lat, and rolled 180 degrees in lon
        data = np.concatenate((
            data_append[:,-nlat:,...], # -87.5, -88.5, -89.5 (crossover)
            data, # -89.5, -88.5, -87.5, ..., 87.5, 88.5, 89.5 (crossover)
            data_append[:,:nlat,...], # 89.5, 88.5, 87.5
            ), axis=1)
        lat = np.pad(lat, nlat, mode='symmetric')
        lat[:nlat], lat[-nlat:] = 180-lat[:nlat], 180-lat[-nlat:]
            # much simpler for lat; but want monotonic ascent, so allow these to be >90, <-90
    return lon, lat, data

def geomean(lon, lat, data,
        box=(None,None,None,None),
        weights=1, keepdims=False):
    """
    Takes area mean of data time series; zone and F are 2d, but data is 3d.
    Since array is masked, this is super easy... just use the masked array
    implementation of the mean, and it will adjust weights accordingly.

    Parameters
    ----------
    lon : array-like
        Grid longitude centers.
    lat : array-like
        Grid latitude centers.
    data : array-like
        Should be lon by lat by (extra dims); will preserve dimensionality.
    box : length-4 list of float, optional
        Region for averaging. Note if edges don't fall on graticule, will
        just subsample the grid cells that do -- haven't bothered to
        account for partial coverage, because it would be pain in the butt
        and not that useful.
    weights : array-like, optional
        Extra, custom weights to apply to data -- could be land/ocean
        fractions, for example.

    Notes
    -----
    Data should be loaded with myfuncs.ncutils.ncload, and provide this function
    with metadata in 'm' structure.

    Todo
    ----
    Allow applying land/ocean mask as part of module functionality.
    """
    # TODO: Make default lat by lon.
    # Get cell areas
    a = np.cos(lat*np.pi/180)[:,None]

    # Get zone for average
    delta = lat[1]-lat[0]
    zone = np.ones((1,lat.size), dtype=bool)
    if box[1] is not None: # south
        zone = zone & ((lat[:-1]-delta/2 >= box[1])[None,:])
    if box[3] is not None: # north
        zone = zone & ((lat[1:]+delta/2 <= box[3])[None,:])

    # Get lune for average
    delta = lon[1]-lon[0]
    lune = np.ones((lon.size,1), dtype=bool)
    if box[0] is not None: # west
        lune = lune & ((lon[:-1]-delta/2 >= box[0])[:,None]) # left-hand box edges >= specified edge
    if box[2] is not None: # east
        lune = lune & ((lon[1:]+delta/2 <= box[2])[:,None]) # right-hand box edges <= specified edge

    # Get total weights
    weights = a*zone*lune*weights
    for s in data.shape[2:]:
        weights = weights[...,None]
    weights = np.tile(weights, (1,1,*data.shape[2:])) # has to be tiled, to match shape of data exactly

    # And apply; note that np.ma.average was tested to be slower than the
    # three-step procedure for NaN ndarrays used below
    if type(data) is np.ma.MaskedArray:
        data = data.filled(np.nan)
    isvalid = np.isfinite(data) # boolean function extremely fast
    data[~isvalid] = 0 # scalar assignment extremely fast
    try:
        ave = np.average(data, weights=weights*isvalid, axis=(0,1))
    except ZeroDivisionError:
        ave = np.nan # weights sum to zero

    # Return, optionally restoring the lon/lat dimensions to singleton
    if keepdims:
        return ave[None,None,...]
    else:
        return ave

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : ndarrays
        Arrays of longitude and latitude in degrees. If one pair is scalar,
        distances are calculated between the scalar pair and each element of
        the non-scalar pair. Otherwise, pair shapes must be identical.

    Returns
    -------
    d : ndarray
        Distances in km.
    """
    # Earth radius, in km
    R = const.a*1e-3

    # Convert to radians, get differences
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine, in km
    km = 2*R*np.arcsin(np.sqrt(
        np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        ))
    return km

#------------------------------------------------------------------------------#
# Spherical coordinate mathematics
#------------------------------------------------------------------------------#
def gradient():
    """
    Gradient in spherical coordinates.
    """
    raise NotImplementedError

def laplacian(lon, lat, data, accuracy=4):
    r"""
    Get Laplacian in spherical coordinates.

    Parameters
    ----------
    lon, lat : ndarray
        The longitude and latitude.
    data : ndarray
        The data.

    Notes
    -----
    The solved equation is as follows:
    .. math::

        \nabla^2 = \frac{\cos^2(\phi)}{a^2}\frac{\partial^2}{\partial^2\theta}
          + \frac{\cos(\phi)}{a^2}\frac{\partial}{\partial\phi}\frac{\cos\phi\partial}{\partial\phi}
    """
    # Setup
    npad = accuracy//2 # need +/-1 for O(h^2) approx, +/-2 for O(h^4), etc.
    data = geopad(lon, lat, data, nlon=npad, nlat=npad)[2] # pad lons/lats
    phi, theta = lat*np.pi/180, lon*np.pi/180 # from north pole

    # Execute
    h_phi, h_theta = abs(phi[2]-phi[1]), abs(theta[2]-theta[1]) # avoids weird edge cases
    phi = phi[None,...]
    for i in range(2,data.ndim):
        phi = phi[...,None]
    laplacian = ( # below avoids repeating finite differencing scheme
            (1/(const.a**2 * np.cos(phi)**2)) * deriv2(h_theta, data[:,npad:-npad,...], axis=0, accuracy=accuracy) # unpad latitudes
            + (-np.tan(phi)/(const.a**2)) * deriv1(h_phi, data[npad:-npad,...], axis=1, accuracy=accuracy) # unpad longitudes
            + (1/(const.a**2)) * deriv2(h_phi, data[npad:-npad,...], axis=1, accuracy=accuracy) # unpad longitudes
            ) # note, no axis rolling required here; the deriv schemes do that
    return laplacian

