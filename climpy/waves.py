#!/usr/bin/env python3
"""
Various diagnostics for quantifying wave activity, tracking wave packets,
and detecting blocking patterns. Incorporates code developed by `Clare Huang \
<https://github.com/csyhuang/hn2016_falwa/>`__.

Warning
-------
This codebase out of date and poorly tested. It will eventually be cleaned up.
In the meantime, feel free to copy and modify it.
"""
# Imports
import numpy as np
from . import const
from .internals import docstring, warnings
from .internals.array import _ArrayContext

__all__ = [
    'eqlat', 'waq', 'waqlocal',
]

docstring.snippets['eqlat.params'] = """
lon, lat : ndarray
    The longitude and latitude coordinates.
q : ndarray
    The potential vorcitiy data. Should be sorted lon x lat x ... (this
    will be changed in a future version).
skip : int, optional
    The interval of sorted `q` from which we select output `Q` values.
"""

docstring.snippets['waq.params'] = """
flip : bool, optional
    Whether to flip the input data along the latitude dimension. Use this
    if your zonal average gradient is negative poleward.
omega : ndarray, optional
    The vertical wind. Shape must match `q`.
sigma : ndarray, optional
    The data weights, useful for isentropic coordinates. Shape must match `q`.
"""


class _LongitudeLatitude(object):
    """
    Class for storing longitude and latitude grid properties. Assumes global
    grid with borders halfway between each grid center.
    """
    def __init__(self, lon, lat):
        # Guess cell widths and edges
        # TODO: Output lat, lon array instead of lon, lat array
        dlon1 = lon[1] - lon[0]
        dlon2 = lon[-1] - lon[-2]
        dlat1 = lat[1] - lat[0]
        dlat2 = lat[-1] - lat[-2]
        self.lonb = np.concatenate(
            (
                lon[:1] - dlon1 / 2,
                (lon[1:] + lon[:-1]) / 2,
                lon[-1:] + dlon2 / 2,
            )
        )
        self.latb = np.concatenate(
            (
                lat[:1] - dlat1 / 2,
                (lat[1:] + lat[:-1]) / 2,
                lat[-1:] + dlat2 / 2,
            )
        )
        self.lonc = lon.copy()
        self.latc = lat.copy()

        # Use corrections for dumb grids with 'centers' at poles
        if lat[0] == -90:
            self.latc[0] = -90 + dlat1 / 4
            self.latb[0] = -90
        if lat[-1] == 90:
            self.latc[-1] = 90 - dlat2 / 4
            self.latb[-1] = 90

        # Corrected grid widths when cells half as tall near pole
        self.dlon = self.lonb[1:] - self.lonb[:-1]
        self.dlat = self.latb[1:] - self.latb[:-1]
        if lat[0] == -90:
            self.dlat[0] /= 2
        if lat[-1] == 90:
            self.dlat[-1] /= 2

        # Convert to radians
        self.phic = self.latc * np.pi / 180
        self.phib = self.latb * np.pi / 180
        self.dphi = self.dlat * np.pi / 180
        self.thetac = self.lonc * np.pi / 180
        self.thetab = self.lonb * np.pi / 180
        self.dtheta = self.dlon * np.pi / 180

        # Area weights. Close approximation to area, since cosine is extremely
        # accurate. Also make lon by lat, so broadcasting rules can apply.
        self.weights = np.cos(self.phic) * self.dphi * self.dtheta[:, None]
        self.areas = self.weights * const.a ** 2


@docstring.add_snippets
def eqlat(lon, lat, q, skip=10, sigma=None):
    r"""
    Get equivalent latitudes corresponding to PV levels evenly sampled from
    the distribution of sorted PVs. This solves the equation:

    .. math::

        A = \int_{-\pi/2}^x a^2 2\pi\cos(\phi) d\phi

    which is the area occupied by PV zonalized below a given contour.

    Parameters
    ----------
    %(eqlat.params)s

    Returns
    -------
    eqlat : ndarray
        The equivalent latitudes. Same shape as `q` but with longitude dimension
        replaced with singleton and latitude dimension replaced with the number
        of equivalent latitudes selected.
    Q : ndarray
        The associated Q thresholds for the equivalent latitudes. Same shape
        as `eqlat`.

    Warning
    -------
    For now assumed dimensionality is lon x lat x ...
    """
    # Get grid areas, as function of latitude
    areas = _LongitudeLatitude(lon, lat).areas

    # Flatten
    # TODO: Use C-style time x plev x lat x lon instead as prototypical dimensionality
    mass = sigma is not None
    arrays = (q, sigma) if mass else (q,)
    with _ArrayContext(*arrays, nflat_right=(q.ndim - 2)) as context:
        # Get flattened arrays
        if not mass:
            q = context.data
        else:
            # Get cumulative mass from equator to pole at each latitude
            # and add them together
            q, sigma = context.data
            mass = sigma * areas[..., None]
            masscum = mass.cumsum(axis=1).sum(axis=0, keepdims=True)

        # Determing Q contour values for evaluating eqlat
        M, K = q.shape  # horizontal space x extra dimensions
        # Count the number of complete length-<skip> blocks after
        # index 0, then add 0 position.
        N = (M - 1) // skip + 1
        # Want to center the list if possible; e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # with skip=3 will give [1, 4, 7].
        offset = (M % skip) // 2

        # Solve equivalent latitudes; options include mass-weighted solution with
        # sigma, or area-weighted
        bands = np.empty((1, N, K))
        q_bands = np.empty((1, N, K))
        for k in range(K):  # iterate through extra dimensions
            # Iterate through Q contours
            q_bands[0, :, k] = np.sort(q[:, :, k], axis=None)[offset::skip]
            for n in range(N):
                f = q[:, :, k] <= q_bands[0, n, k]  # filter
                if sigma is None:
                    # Get sine of eqlat, and correct for rounding errors
                    sin = areas[f].sum() / (2 * np.pi * const.a ** 2) - 1
                    sin = np.clip(sin, -1, 1)
                    bands[0, n, k] = np.arcsin(sin) * 180 / np.pi
                else:
                    # Interpolate to latitude of mass contour
                    massk = mass[:, :, k]
                    masscumk = masscum[:, :, k].squeeze()
                    mass = massk[f].sum()  # total mass below Q
                    bands[0, n, k] = np.interp(mass, masscumk, lat)

        # Unshape data
        context.replace_data(bands, q_bands)

    # Return unflattened data
    return context.data


def waq(lon, lat, q, sigma=None, omega=None, flip=True, skip=10):
    """
    Return the finite-amplitude wave activity. See
    :cite:`2010:nakamura` for details.

    Parameters
    ----------
    %(eqlat.params)s
    %(waq.params)s

    References
    ----------
    .. bibliography:: ../bibs/waq.bib
    """
    # Graticule considerations
    has_omega = omega is not None
    has_sigma = sigma is not None
    if flip:
        lat, q = -np.flipud(lat), -np.flip(q, axis=1)
        if has_omega:
            omega = -np.flip(omega, axis=1)
        if has_sigma:
            sigma = np.flipd(sigma, axis=1)
    grid = _LongitudeLatitude(lon, lat)
    areas, dphi, phib = grid.areas, grid.dphi, grid.phib

    # Flatten (eqlat can do this, but not necessary here)
    arrays = [q]
    if has_omega:
        arrays.append(omega)
    if has_sigma:
        arrays.append(sigma)
    with _ArrayContext(*arrays, nflat_right=(q.ndim - 2)) as context:
        # Get flattened data
        if has_omega and has_sigma:
            q, omega, sigma = context.data
        elif has_omega:
            q, omega = context.data
        elif has_sigma:
            q, sigma = context.data
        else:
            q = context.data

        # Get equivalent latiitudes
        bands, q_bands = eqlat(lon, lat, q, sigma=sigma, skip=skip)
        M = lat.size  # number of latitudes onto which we interpolate
        N = bands.shape[1]  # number of equivalent latitudes
        K = q.shape[2]  # number of extra dimensions

        # Get activity
        waq = np.empty((1, M, K))
        percent = 0
        for k in range(K):
            # Status message
            if (k / K) > (0.01 * percent):
                print('%d%% finished' % (percent,))
                percent = percent + 10

            # Loop through each contour
            # i, bandki in enumerate(bandk[0,:,k]):
            # enumerate(zip(bandk, q_bandk)):
            waq_k = np.empty(N)
            for n in range(N):
                # First, main blocks
                band = bands[0, n, k] * np.pi / 180
                if np.isnan(band):
                    waq_k[n] = np.nan
                    continue

                # Work with anomalies? Should be identical, since
                # positive/negative regions have same area by construction.
                # anom = q[:,:,k] - q_bands[0,n,k]
                # f_pos = (anom >= 0) & (phib[None,1:] < band)
                # f_neg = (anom < 0) & (phib[None,:-1] >= band)
                # integral = (anom[f_pos]*areas[f_pos]).sum() -
                # (anom[f_neg]*areas[f_neg]).sum() # minus a negative

                # Get the integrand
                qk, Qk = q[:, :, k], q_bands[0, n, k]
                if has_omega:
                    qint = q[:, :, k]  # the thing being integrated
                else:
                    qint = omega[:, :, k]

                # Get high anomalies at low latitude (below top graticule) and
                # low anomalies at high latitude (above bottom graticule)
                f_pos = (qk >= Qk) & (phib[None, 1:] < band)
                f_neg = (qk < Qk) & (phib[None, :-1] >= band)
                integral = (
                    (qint[f_pos] * areas[f_pos]).sum()
                    - (qint[f_neg] * areas[f_neg]).sum()  # minus a negative
                )

                # Account for tiny pieces along equivalent latitude cells
                mid, = np.where((phib[:-1] <= band) & (phib[1:] > band))
                integral_extra = 0
                if mid:
                    # Find longitudes where positive
                    f_pos_mid = qk[:, mid] >= Qk
                    # Positive high latitude and negative, low latitude
                    p_dphi = np.cos((band + phib[mid]) / 2) * (band - phib[mid])
                    m_dphi = np.cos((band + phib[mid + 1]) / 2) * (phib[mid + 1] - band)
                    # Extra pieces due to discrete grid
                    integral_extra = (
                        qint[f_pos_mid, mid].sum() * (areas[mid] * m_dphi / dphi[mid])
                        - qint[~f_pos_mid, mid].sum() * (areas[mid] * p_dphi / dphi[mid])  # noqa: E501
                    )

                # Put it all together
                waq_k[n] = (
                    (integral + integral_extra) / (2 * np.pi * const.a * np.cos(band))
                )

            # Interpolate
            nanfilt = np.isnan(waq_k)
            if sum(~nanfilt) == 0:
                warnings._warn_climpy(f'Warning: No valid waqs calculated for k {k}.')
                waq[0, :, k] = np.nan
            else:
                waq[0, :, k] = np.interp(lat, bands[0, ~nanfilt, k], waq_k[~nanfilt])

        # Reapply data
        context.replace_data(waq)

    # Return
    waq = context.waq
    if flip:
        waq = np.flip(waq, axis=1)
    return waq


@docstring.add_snippets
def waqlocal(lon, lat, q, omega=None, sigma=None, flip=True, skip=10):
    """
    Return the local finite-amplitude wave activity. See
    :cite:`2016:huang` for details.

    Parameters
    ----------
    %(eqlat.params)s
    %(waq.params)s

    References
    ----------
    .. bibliography:: ../bibs/waqlocal.bib

    Todo
    ----
    Support using `omega`.
    """
    # Graticule considerations
    has_omega = omega is not None
    has_sigma = sigma is not None
    if flip:
        lat, q = -np.flipud(lat), -np.flip(q, axis=1)
        if has_omega:
            omega = -np.flip(omega, axis=1)
        if sigma is not None:
            sigma = np.flipd(sigma, axis=1)
    grid = _LongitudeLatitude(lon, lat)
    phib = grid.phib
    integral = const.a * phib[None, :]

    # Flatten (eqlat can do this, but not necessary here)
    arrays = [q]
    if has_omega:
        arrays.append(omega)
    if has_sigma:
        arrays.append(sigma)

    # Flatten (eqlat can do this, but not necessary here)
    with _ArrayContext(*arrays, nflat_right=(q.ndim - 2)) as context:
        # Get flattened data
        if has_omega and has_sigma:
            q, omega, sigma = context.data
        elif has_omega:
            q, omega = context.data
        elif has_sigma:
            q, sigma = context.data
        else:
            q = context.data

        # Get equivalent latiitudes
        bands, q_bands = eqlat(lon, lat, q, sigma=sigma, skip=skip)
        L = lon.size
        M = lat.size
        N = bands.shape[1]  # number of equivalent latitudes
        K = q.shape[2]  # number of extra dimensions

        # Get local wave activity measure, as simple line integrals
        waq = np.empty((L, M, K))
        percent = 0
        for k in range(K):
            if (k / K) > (0.01 * percent):
                print('%d%% finished' % (100 * k / K,))
                percent = percent + 10

            # Loop through each contour
            waq_k = np.empty((L, N))
            for n in range(N):
                # Setup, large areas
                band = bands[0, n, k] * np.pi / 180
                if np.isnan(band):  # happens if contours intersect at edge
                    waq_k[:, n] = np.nan
                else:
                    # Get high anomalies at low latitude (below top graticule) and
                    # low anomalies at high latitude (above bottom graticule)
                    anom = q[:, :, k] - q_bands[0, n, k]
                    f_pos = (anom >= 0) & (phib[None, 1:] < band)
                    f_neg = (anom < 0) & (phib[None, :-1] >= band)

                    # See if band is sandwiched between latitudes
                    # want scalar id (might be zero)
                    mid, = np.where((phib[:-1] <= band) & (phib[1:] > band))
                    if mid.size > 0:
                        # Find longitudes where positive
                        # Perform partial integrals, positive and negative
                        f_pos_mid = anom[:, mid] >= 0
                        p_int = const.a * (band - phib[mid])
                        m_int = const.a * (phib[mid + 1] - band)

                    for l in range(L):
                        # Get individual integrals
                        integral_pos = (
                            anom[l, f_pos[l, :]] * integral[:, f_pos[l, :]]
                        ).sum()
                        integral_neg = -(  # *minus* a *negative*
                            anom[l, f_neg[l, :]] * integral[:, f_neg[l, :]]
                        ).sum()

                        # Get extra bits
                        if mid.size > 0:
                            if f_pos_mid[l]:
                                integral_extra = anom[l, mid] * p_int
                            else:
                                integral_extra = -anom[l, mid] * m_int
                        else:
                            integral_extra = 0

                        # Put it all together
                        waq_k[l, n] = integral_pos + integral_neg + integral_extra

        # Interpolate to actual latitudes
        for l in range(L):
            waq[l, :, k] = np.interp(lat, bands[0, :, k], waq_k[l, :])

        # Replace context data
        context.replace_data(waq)

    # Return
    waq = context.data
    if flip:
        waq = np.flip(waq, axis=1)
    return waq
