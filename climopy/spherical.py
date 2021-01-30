#!/usr/bin/env python3
"""
Mathematical operations on the surface of the sphere.

Warning
-------
This submodule out of date and poorly tested. It will eventually be cleaned up.
In the meantime, feel free to copy and modify it.
"""
import numpy as np

from . import const, diff

__all__ = [
    'geopad', 'geomean', 'geogradient', 'geolaplacian', 'haversine',
]


def geopad(lon, lat, data, /, nlon=1, nlat=0):
    """
    Return array padded circularly along longitude and over the poles for finite
    difference methods.
    """
    # Pad over longitude seams
    if nlon > 0:
        pad = ((nlon, nlon),) + (data.ndim - 1) * ((0, 0),)
        data = np.pad(data, pad, mode='wrap')
        lon = np.pad(lon, nlon, mode='wrap')  # should be vector

    # Pad over poles
    if nlat > 0:
        if (data.shape[0] % 2) == 1:
            raise ValueError(
                'Data must have even number of longitudes '
                'if you wish to pad over the poles.'
            )
        append = np.roll(  # descending in lat
            np.flip(data, axis=1), data.shape[0] // 2, axis=0
        )
        data = np.concatenate(
            (
                append[:, -nlat:, ...],  # -87.5, -88.5, -89.5 (crossover)
                data,  # -89.5, -88.5, -87.5, ..., 87.5, 88.5, 89.5 (crossover)
                append[:, :nlat, ...],  # 89.5, 88.5, 87.5
            ),
            axis=1,
        )
        lat = np.pad(lat, nlat, mode='symmetric')
        lat[:nlat] = 180 - lat[:nlat]  # monotonic ascent
        lat[-nlat:] = 180 - lat[-nlat:]
    return lon, lat, data


def geomean(lon, lat, data, /, box=None, weights=1, keepdims=False):
    """
    Take area mean of data time series.

    Parameters
    ----------
    lon : array-like
        Grid longitude centers.
    lat : array-like
        Grid latitude centers.
    data : array-like
        Should be lon by lat by (extra dims); will preserve dimensionality.
    box : length-4 list of float, optional
        Region for averaging. If the edges don't fall on the graticule, we
        subsample the grid cells that do.
    weights : array-like, optional
        Weights to apply to the averages.

    Todo
    ----
    Allow applying land or ocean mask as part of module functionality.
    """
    # Parse input
    box = (None, None, None, None)

    # Get zone for average
    delta = lat[1] - lat[0]
    zone = np.ones((1, lat.size), dtype=bool)
    if box[1] is not None:  # south
        zone = zone & ((lat[:-1] - delta / 2 >= box[1])[None, :])
    if box[3] is not None:  # north
        zone = zone & ((lat[1:] + delta / 2 <= box[3])[None, :])

    # Get lune for average
    delta = lon[1] - lon[0]
    lune = np.ones((lon.size, 1), dtype=bool)
    if box[0] is not None:  # west
        # left-hand box edges >= specified edge
        lune = lune & ((lon[:-1] - delta / 2 >= box[0])[:, None])
    if box[2] is not None:  # east
        # right-hand box edges <= specified edge
        lune = lune & ((lon[1:] + delta / 2 <= box[2])[:, None])

    # Get total weights
    a = np.cos(lat * np.pi / 180)[:, None]
    weights = a * zone * lune * weights
    for s in data.shape[2:]:
        weights = weights[..., None]
    weights = np.tile(weights, (1, 1, *data.shape[2:]))

    # And apply; note that np.ma.average was tested to be slower than the
    # three-step procedure for NaN ndarrays used below
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(np.nan)
    isvalid = np.isfinite(data)  # boolean function extremely fast
    data[~isvalid] = 0  # scalar assignment extremely fast
    try:
        ave = np.average(data, weights=weights * isvalid, axis=(0, 1))
    except ZeroDivisionError:
        ave = np.nan  # weights sum to zero

    # Return, optionally restoring the lon/lat dimensions to singleton
    if keepdims:
        return ave[None, None, ...]
    else:
        return ave


def geogradient(lon, lat, data, /):  # noqa: U100
    """
    Calculate gradient in spherical coordinates.
    """
    return NotImplementedError


def geolaplacian(lon, lat, data, /, accuracy=4):
    r"""
    Calculate Laplacian in spherical coordinates.

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

        \nabla^2
        = \frac{1}{a^2\cos^2(\phi)} \frac{\partial^2}{\partial^2\theta}
        + \frac{1}{a^2\cos(\phi)} \frac{\partial}{\partial\phi} \cos\phi \frac{\partial}{\partial\phi}
    """  # noqa
    # Setup
    npad = accuracy // 2  # need +/-1 for O(h^2) approx, +/-2 for O(h^4), etc.
    data = geopad(lon, lat, data, nlon=npad, nlat=npad)[2]  # pad lons/lats
    phi, theta = lat * np.pi / 180, lon * np.pi / 180  # from north pole

    # Execute with chain rule
    # TODO: Is chain rule best, or double even first order differentiation?
    h_phi = phi[2] - phi[1]
    h_theta = theta[2] - theta[1]
    phi = phi[None, ...]
    for i in range(2, data.ndim):
        phi = phi[..., None]
    return (
        1 / (const.a ** 2 * np.cos(phi) ** 2)
        * diff.deriv_even(
            h_theta, data[:, npad:-npad, ...], axis=0, order=2, accuracy=accuracy
        )  # unpad latitudes
        - np.tan(phi) / (const.a ** 2)
        * diff.deriv_even(
            h_phi, data[npad:-npad, ...], axis=1, order=1, accuracy=accuracy
        )  # unpad longitudes
        + 1 / (const.a ** 2)
        * diff.deriv_even(
            h_phi, data[npad:-npad, ...], axis=1, order=2, accuracy=accuracy
        )  # unpad longitudes
    )


def haversine(lon1, lat1, lon2, lat2, /):
    """
    Calculate the great circle distance between two points
    on Earth, specified in degrees.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : ndarray
        Coordinate arrays for the first and second point(s). If one pair is
        scalar, distances are calculated between the scalar pair and each
        element of the non-scalar pair. Otherwise, the pair shapes must
        be identical.

    Returns
    -------
    ndarray
        Distances in km.
    """
    # Earth radius in km
    R = const.a * 1e-3

    # Convert to radians
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine in km
    km = 2 * R * np.arcsin(np.sqrt(
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    ))
    return km
