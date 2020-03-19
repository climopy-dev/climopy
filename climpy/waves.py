#!/usr/bin/env python3
"""
Various diagnostics for quantifying wave activity, tracking wave packets,
and detecting blocking patterns. Incorporates code developed by `Clare Huang \
<https://github.com/csyhuang/hn2016_falwa/>`__.
"""
# Imports
import numpy as np
from . import const, utils


class Graticule(object):
    """
    Class for storing longitude and latitude grid properties. Assumes
    a global grid with borders halfway between each grid center.
    """
    def __init__(self, lon, lat):
        # First, guess cell widths and edges
        lon, lat = lon.astype(np.float32), lat.astype(np.float32)
        dlon1, dlon2 = lon[1] - lon[0], lon[-1] - lon[-2]
        dlat1, dlat2 = lat[1] - lat[0], lat[-1] - lat[-2]
        self.latb = np.concatenate(
            (
                lat[:1] - dlat1 / 2,
                (lat[1:] + lat[:-1]) / 2,
                lat[-1:] + dlat2 / 2,
            )
        )
        self.lonb = np.concatenate(
            (
                lon[:1] - dlon1 / 2,
                (lon[1:] + lon[:-1]) / 2,
                lon[-1:] + dlon2 / 2,
            )
        )
        self.latc, self.lonc = lat.copy(), lon.copy()

        # Use corrections for dumb grids with 'centers' at poles
        if lat[0] == -90:
            self.latc[0], self.latb[0] = -90 + dlat1 / 4, -90
        if lat[-1] == 90:
            self.latc[-1], self.latb[-1] = 90 - dlat2 / 4, 90

        # Corrected grid widths when cells half as tall near pole
        self.dlon = self.lonb[1:] - self.lonb[:-1]
        self.dlat = self.latb[1:] - self.latb[:-1]
        if lat[0] == -90:
            self.dlat[0] /= 2
        if lat[-1] == 90:
            self.dlat[-1] /= 2

        # Convert to radians
        self.phic, self.phib, self.dphi = (
            self.latc * np.pi / 180,
            self.latb * np.pi / 180,
            self.dlat * np.pi / 180,
        )
        self.thetac, self.thetab, self.dtheta = (
            self.lonc * np.pi / 180,
            self.lonb * np.pi / 180,
            self.dlon * np.pi / 180,
        )

        # Area weights. Close approximation to area, since cosine is extremely
        # accurate. Also make lon by lat, so broadcasting rules can apply.
        self.weights = (
            self.dphi[None, :]
            * self.dtheta[:, None]
            * np.cos(self.phic[None, :])
        )
        self.areas = (
            self.dphi[None, :]
            * self.dtheta[:, None]
            * np.cos(self.phic[None, :])
            * (const.a ** 2)
        )


def eqlat(lon, lat, q, skip=10, sigma=None):
    r"""
    Get equivalent latitudes corresponding to PV levels evenly sampled from
    the distribution of sorted PVs. This solves the equation:

    .. math::

        A == \int_{-\pi/2}^x a^2 2\pi\cos(\phi) d\phi

    which is the area occupied by PV zonalized below a given contour.

    Parameters
    ----------
    lon, lat : ndarray
        The longitude and latitude coordinates.
    q : ndarray
        The data array.
    skip : int, optional
        The interval of sorted `q` from which we select output `Q` values.

    Returns
    -------
    eqlat : ndarray
        The equivalent latitudes.
    Q : ndarray
        The associated Q thresholds for the equivalent latitudes.
    """
    # Initial stuff
    # Delivers grid areas, as function of latitude
    areas = Graticule(lon, lat).areas

    # Flatten
    q, shape = utils.lead_flatten(q, 2)  # gives current q, and former shape

    # And consider mass-weighting this stuff
    if sigma is not None:
        # Note that at least singleton dimension is added
        mass = utils.lead_flatten(sigma, 2) * areas[..., None]
        # Get cumulative mass from pole
        masscum = mass.cumsum(axis=1).sum(axis=0, keepdims=True)

    # Determing Q contour values for evaluating eqlat
    K = q.shape[-1]  # number of extra dims
    # Count the number of complete length-<skip> blocks after index 0,
    # then add 0 position.
    N = (np.prod(shape[:2]) - 1) // skip + 1
    # Want to center the list if possible; e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # with skip=3 will give [1, 4, 7].
    offset = (np.prod(shape[:2]) % skip) // 2

    # Solve equivalent latitudes; options include mass-weighted solution with
    # sigma, or area-weighted
    bands, q_bands = np.empty((1, N, K)), np.empty((1, N, K))
    for k in range(K):  # iterate through extra dimensions
        # Iterate through Q contours
        q_bands[0, :, k] = np.sort(q[:, :, k], axis=None)[offset::skip]
        for n in range(N):
            f = q[:, :, k] <= q_bands[0, n, k]  # filter
            if sigma is None:  # normal weighting
                # Get sine of eqlat, and correct for rounding errors
                sin = areas[f].sum() / (2 * np.pi * const.a ** 2) - 1
                if sin > 1:
                    sin = 1
                if sin < -1:
                    sin = -1
                bands[0, n, k] = np.arcsin(sin) * 180 / np.pi
            else:  # mass weighting
                # Interpolate to latitude of mass contour
                massk, masscumk = (
                    mass[:, :, k],
                    masscum[:, :, k].squeeze(),
                )  # latter is coordinate
                mass = massk[f].sum()  # total mass below Q
                bands[0, n, k] = np.interp(mass, masscumk, lat)

    # Reshape, and return
    return (
        utils.lead_unflatten(bands, shape, 2),
        utils.lead_unflatten(q_bands, shape, 2)
    )


def waq(
    lon, lat, q, sigma=None, omega=None, flip=True, skip=10
):
    """
    Return the finite-amplitude wave activity. See
    :cite:`2010:nakamura` for details.

    Parameters
    ----------
    lon, lat : ndarray
        The longitude and latitude coordiantes.
    q : ndarray
        The data array.
    omega : ndarray, optional
        The data weights, useful for isentropic coordinates. Shape must match
        shape of `q`.
    flip : bool, optional
        Whether to flip the input data along the latitude dimension. Use this
        if your zonal average gradient is negative poleward.
    skip : int, optional
        Passed to `eqlat`.

    References
    ----------
    .. bibliography:: ../waq.bib
    """
    # Graticule considerations
    if flip:
        lat, q = -np.flipud(lat), -np.flip(q, axis=1)
        if omega is not None:
            omega = -np.flip(omega, axis=1)
        if sigma is not None:
            sigma = np.flipd(sigma, axis=1)
    grid = Graticule(lon, lat)
    areas, dphi, phib = grid.areas, grid.dphi, grid.phib

    # Flatten (eqlat can do this, but not necessary here)
    q, shape = utils.lead_flatten(q, 2)
    if omega is not None:
        omega, _ = utils.lead_flatten(omega, 2)
    if sigma is not None:
        sigma, _ = utils.lead_flatten(sigma, 2)

    # Get equivalent latiitudes
    bands, q_bands = eqlat(
        lon, lat, q, sigma=sigma, skip=skip
    )  # note w is just lon x lat
    M = q.shape[1]  # number of eqlats
    N = bands.shape[1]
    K = q.shape[-1]  # number of extra dimensions

    # Get activity
    waq = np.empty((1, M, K))
    percent = 0
    for k in range(K):
        if (k / K) > (0.01 * percent):
            print('%d%% finished' % (percent,))
            percent = percent + 10
        # Loop through each contour
        waq_k = np.empty(N)
        # i, bandki in enumerate(bandk[0,:,k]): #, q_bandki) in
        # enumerate(zip(bandk, q_bandk)):
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
            if omega is None:
                qint = q[:, :, k]  # the thing being integrated
            else:
                qint = omega[:, :, k]

            # Get high anomalies at low latitude (below top graticule) and
            # low anomalies at high latitude (above bottom graticule)
            f_pos = (qk >= Qk) & (phib[None, 1:] < band)
            f_neg = (qk < Qk) & (phib[None, :-1] >= band)
            integral = (qint[f_pos] * areas[f_pos]).sum() - (
                qint[f_neg] * areas[f_neg]
            ).sum()  # minus a negative

            # Account for tiny pieces along equivalent latitude cells
            mid = np.where((phib[:-1] <= band) & (phib[1:] > band))[0]
            integral_extra = 0
            if mid:
                mid = mid[0]
                f_pos_mid = qk[:, mid] >= Qk  # longitudes where positive
                p_dphi, m_dphi = (
                    np.cos((band + phib[mid]) / 2)
                    * (band - phib[mid]),  # positive, high lat
                    np.cos((band + phib[mid + 1]) / 2)
                    * (phib[mid + 1] - band),  # negative, low lat
                )
                integral_extra = qint[f_pos_mid, mid].sum() * (
                    areas[mid] * m_dphi / dphi[mid]
                ) - qint[~f_pos_mid, mid].sum() * (
                    areas[mid] * p_dphi / dphi[mid]
                )

            # Put it all together
            waq_k[n] = (integral + integral_extra) / (
                2 * np.pi * const.a * np.cos(band)
            )

        # Interpolate
        nanfilt = np.isnan(waq_k)
        if sum(~nanfilt) == 0:
            print('Warning: No valid waqs calculated for k %d.' % k)
            waq[0, :, k] = np.nan
        else:
            waq[0, :, k] = np.interp(
                grid.latc, bands[0, ~nanfilt, k], waq_k[~nanfilt]
            )

    # Return
    if flip:
        waq = np.flip(waq, axis=1)
    return utils.lead_unflatten(waq, shape)


def waqlocal(lon, lat, q, flip=True, skip=10):
    """
    Return the local finite-amplitude wave activity. See
    :cite:`2016:huang` for details.

    Parameters
    ----------
    lon, lat : ndarray
        The longitude and latitude coordiantes.
    q : ndarray
        The data array.
    omega : ndarray, optional
        The data weights, useful for isentropic coordinates. Shape must match
        shape of `q`.
    flip : bool, optional
        Whether to flip the input data along the latitude dimension. Use this
        if your zonal average gradient is negative poleward.
    skip : int, optional
        Passed to `eqlat`.

    References
    ----------
    .. bibliography:: ../waqlocal.bib
    """
    # Graticule considerations
    if flip:
        lat, q = -np.flipud(lat), -np.flip(q, axis=1)
    phib = Graticule(lon, lat).phib
    integral = const.a * phib[None, :]

    # Flatten (eqlat can do this, but not necessary here)
    q, shape = utils.lead_flatten(q, 2)

    # Get equivalent latiitudes
    bands, q_bands = eqlat(lon, lat, q, skip=skip)
    L, M, N, K = q.shape[0], q.shape[1], bands.shape[1], q.shape[-1]

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
                mid = np.where((phib[:-1] <= band) & (phib[1:] > band))[0]
                if mid.size > 0:
                    # Find longitudes where positive
                    # Perform partial integrals, positive and negative
                    f_pos_mid = anom[:, mid] >= 0
                    p_int, m_int = (
                        const.a * (band - phib[mid]),
                        const.a * (phib[mid + 1] - band),
                    )

                for l in range(L):
                    # Get individual integral
                    integral_pos = (
                        anom[l, f_pos[l, :]] * integral[:, f_pos[l, :]]
                    ).sum()
                    integral_neg = -(  # *minus* a *negative*
                        anom[l, f_neg[l, :]] * integral[:, f_neg[l, :]]
                    ).sum()
                    if mid.size > 0:
                        # If positive at this latitude, we add anomaly
                        # Else, subtract it
                        if f_pos_mid[l]:
                            integral_extra = anom[l, mid] * p_int
                        else:
                            integral_extra = -anom[l, mid] * m_int
                    else:
                        integral_extra = 0

                    # Put it all together
                    waq_k[l, n] = (
                        integral_pos + integral_neg + integral_extra
                    )  # no normalization here

        # Interpolate
        for l in range(L):
            waq[l, :, k] = np.interp(lat, bands[0, :, k], waq_k[l, :])

    # Return
    if flip:
        waq = np.flip(waq, axis=1)
    return utils.lead_unflatten(waq, shape)
