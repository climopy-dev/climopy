#!/usr/bin/env python3
"""
Wrappers on python APIs for downloading reanalysis data.
"""
# Imports
import xarray as xr
import numpy as np
import calendar

# -------------------------------------------------------------------------------
# ERA-interim downloads
# -------------------------------------------------------------------------------


def eraint(params, stream, levtype,
           daterange=None, yearrange=None, monthrange=None, dayrange=None,
           years=None, months=None,  # can specify list
           format='netcdf',  # file format
           # whether we want the *forecast* field? and forecast timestep if so
           forecast=False, step=12,
           # means number of lats per hemisphere, we want 64
           levrange=None, levs=None, grid='N32',
           hours=(0, 6, 12, 18), hour=None,
           res=1.0, box=None,
           filename='eraint.nc'
           ):
    """
    Retrieves ERA-Interim data using the provided API. User must have, in home
    directory, a file named ``.ecmwfapirc``. See API documentation, but should
    look like:
    ::

        {
        "url"   : "https://api.ecmwf.int/v1",
        "key"   : "abcdefghijklmnopqrstuvwxyz",
        "email" : "email@gmail.com"
        }

    with the key found on your user/profile page on the ECMWF website.

    Parameters
    ----------
    params : str or list of str
        Variable name. Gets translated to MARS id name by dictionary below.
        Add to this from the `online GRIB table <https://rda.ucar.edu/datasets/ds627.0/docs/era_interim_grib_table.html>`_.
        Pay attention to *available groups*. If not available for the group
        you selected (e.g. pressure levs, moda), get ``ERROR 6 (MARS_EXPECTED_FIELDS)``.
        For rates of change of *parameterized* processes (i.e. diabatic) see
        `this link <https://confluence.ecmwf.int/pages/viewpage.action?pageId=57448466>`_.
    stream : {'oper', 'moda', 'mofm', 'mdfa', 'mnth'}
        The data stream.
    levtype : {'ml', 'pl', 'sfc', 'pt', 'pv'}
        Level type: model, pressure, surface, potential temperature, and 2PVU
        surface, respectively.
    levrange : float or (float, float), optional
        Individual level or range of levels.
    levs : float or ndarray, optional
        Individual level or list of levels.
    yearrange : int or (int, int)
        Individual year or range of years.
    years : int or ndarray, optional
        Individual year or list of years.
    monthrange : int or (int, int), optional
        Individual month or range of months.
    months : int or ndarray, optional
        Individual month or list of months.
    daterange : (datetime.datetime, datetime.datetime), optional
        Range of dates.
    hours : {0, 6, 12, 18} or list thereof, optional
        Hour(s) (UTC) of observation.
    forecast : bool, optional
        Whether we want forecast `'fc'` or analysis `'an'` data. Note that
        some data is only available in `'fc'` mode, e.g. diabatic heating.
    res : float, optional
        Desired output resolution in degrees. Not sure if this is arbitrary,
        or if ERA-interim only has a select few valid resolution options.
    box : str or length-4 list of float, optional
        String name for particular region, e.g. ``'europe'``, or the west,
        south, east, and north boundaries, respectively.
    format : {'grib1', 'grib2', 'netcdf'}, optional
        Output format.
    filename : str, optional
        Name of file output.


    Notes
    -----
    Some fields (seems true for most model fields) are not archived as
    monthly means for some reason! Have no idea why because it would need
    almost zero storage requirements.
    """
    # Data stream
    import ecmwfapi as ecmwf  # only do so inside function

    # Variable id conversion (see:
    # https://rda.ucar.edu/datasets/ds627.0/docs/era_interim_grib_table.html)
    if isinstance(params, str) or not np.iterable(params):
        params = (params,)
    params = [{
        'tdt': '110.162',
        # model-level surface pressure; use this whenever getting tdt and stuff
        # (requires lev=1)
        'msp': '152.128',
        'sp': '134.128',  # surface pressure
        't2m': '167.128',  # 2m temp
        'd2m': '168.128',  # 2m dew point
        'sst': '34.128',  # sst
        'msl': '151.128',  # sea level pressure
        'slp': '151.128',  # same
        'z': '129.128',  # geopotential
        't': '130.128',  # temp
        'u': '131.128',  # u wind
        'v': '132.128',  # v wind
        'w': '135.128',  # w wind
        'q': '133.128',  # specific humidity
        'r': '157.128',  # relative humidity
        'vort': '138.128',  # relative vorticity
        'vo': '138.128',  # same
        'zeta': '138.128',  # same
        'pt': '3.128',  # potential temp (available on 2pvu surf)
        'theta': '3.128',  # same
        'p': '54.128',  # pressure (availble on pt, 2pvu surfaces)
        'pres': '54.128',  # same
        # potential vorticity (available on p, pt surfaces)
        'pv': '60.128',
        'precip': '228.128',
    }.get(p) for p in params]  # returns generator object for each param
    if None in params:
        raise ValueError(
            'MARS id for variable is unknown (might need to be added to this script).')
    params = '/'.join(params)

    # Time selection as various RANGES or LISTS
    # Priority; just use daterange as datetime or date objects
    if daterange is not None:
        if not np.iterable(daterange):
            daterange = (daterange,)  # want a SINGLE DAY
        # options for monthly or daily data
        if stream != 'oper':
            y0, m0, y1, m1 = daterange[0].year, daterange[0].month, daterange[1].year, daterange[1].month
            N = max(y1 - y0 - 1, 0) * 12 + (13 - m0) + \
                m1  # number of months in range
            dates = '/'.join('%04d%02d00' %
                             (y0 + (m0 + n - 1) // 12, (m0 + n - 1) %
                              12 + 1) for n in range(N))
        else:
            # MARS will get calendar days in range
            dates = '/to/'.join(d.strftime('%Y%m%d') for d in daterange)

    # Alternative; list the years/months desired, and if synoptic, get all
    # calendar days within
    else:
        # First, years
        if years is not None:
            if not np.iterable(years):
                years = (years,)  # single month
        elif yearrange is not None:
            if not np.iterable(yearrange):
                years = (yearrange,)
            else:
                years = tuple(range(yearrange[0], yearrange[1] + 1))
        else:
            raise ValueError('You must use "years" or "yearrange" kwargs.')
        # Next, months (this way, can just download JJA data, for example)
        if months is not None:
            if not np.iterable(months):
                months = (months,)  # single month
        elif monthrange is not None:
            if not np.iterable(monthrange):
                months = (monthrange, monthrange)
            else:
                months = tuple(range(monthrange[0], monthrange[1] + 1))
        else:
            months = tuple(range(1, 13))
        # And get dates; options for monthly means and daily stuff
        if stream != 'oper':
            dates = '/'.join(
                '/'.join('%04d%02d00' % (y, m) for m in months)
                for y in years)
        else:
            dates = '/'.join(
                '/'.join(
                    '/'.join('%04d%02d%02d' % (y, m, i + 1)
                             for i in range(calendar.monthrange(y, m)[1]))
                    for m in months)
                for y in years)

    # Level selection as RANGE or LIST
    # Update this list if you modify script for ERA5, etc.
    levchoices = {
        'sfc': None,
        'pv': None,
        'ml': np.arange(1, 137 + 1),
        'pl': np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]),
        'pt': np.array([265, 270, 285, 300, 315, 330, 350, 370, 395, 430, 475, 530, 600, 700, 850]),
    }.get(levtype, [])
    if levchoices == []:
        raise ValueError(
            'Invalid level type. Choose from "pl", "pt", "pv", "sfc".')
    # More options
    # WARNING: Some vars are technically on "levels" like hybrid level surface
    # pressure, but we still need 60.
    if levtype not in ('sfc', 'pv'):  # these have multiple options
        # Require input
        if levs is None and levrange is None:
            raise ValueError(
                'Must specify list of levels to "levs" kwarg, range of levels to "levrange" kwarg, or single level to either one.')
        # Convert levels to mars request
        if levs is not None:
            if not np.iterable(levs):
                levs = (levs,)
        else:
            if not np.iterable(levrange):
                levs = (levrange,)
            else:
                levs = levchoices[(levchoices >= levrange[0])
                                  & (levchoices <= levrange[1])].flat
        levs = '/'.join(str(l) for l in levs)

    # Other parameters
    # Resolution
    # same in latitude/longitude required, for now
    res = '%.5f/%.5f' % (res, res)
    # Area - can be specified as pre-defined region (e.g. string 'europe') OR
    # n/s/w/e boundary
    if box is not None and not isinstance(box, str):
        box = '/'.join(str(b) for b in (box[3], box[0], box[2], box[1]))
    # Hour conversion
    if not np.iterable(hours):
        hours = (hours,)
    hours = '/'.join(str(h).zfill(2) for h in hours)  # zfill padds 0s on left
    # Forecast type
    # TODO: Fix the 12 hour thing. Works for some params (e.g. diabatic
    # heating, has 3, 6, 9, 12) but others have 0, 6, 12, 18.
    if forecast:
        dtype, step = 'fc', str(step)
    else:
        dtype, step = 'an', '0'

    # Server instructions
    # Not really sure what happens in some situations: list so far:
    # 1) evidently if you provide with variable string-name instead of numeric ID,
    #       MARS will search for correct one; if there is name ambiguity/conflict will throw error
    # 2) on GUI framework, ECMWF only offers a few resolution options, but program seems
    #       to run when requesting custom resolutions like 5deg/5deg
    request = {
        # Important ones
        'class': 'ei',  # ecmwf classifiction; choose ERA-Interim
        'expver': '1',
        'dataset': 'interim',  # thought we already did that; *shrug*
        'type': dtype,  # type of field; analysis 'an' or forecast 'fc'
        'resol': 'av',  # prevents truncation before transformation to geo grid
        'gaussian': 'reduced',
        'format': format,
        'step': step,  # NOTE: ignored for non-forecast type
        # 'grid':     res,
        # Can also spit raw output into GRIB; apparently ERA-Interim uses
        # bilinear interpolation to make grid of point obs, which makes sense,
        # because their reanalysis model just picks out point observations
        # from spherical harmonics; so maybe grid cell concept is dumb? maybe need to focus
        # on just using cosine weightings, forget about rest?
        'grid': grid,  # 64 latitudes, i.e. T42 truncation
        'stream': stream,  # product monthly, raw, etc.
        'date': dates,
        'time': hours,
        'levtype': levtype,
        'param': params,
        'target': filename,  # save location
    }
    maxlen = max(map(len, request.keys()))
    if levs is not None:
        request.update(levelist=levs)
    if box is not None:
        request.update(area=box)
    if stream == 'oper':  # TODO: change?
        request.update(hour=hour)
    print("REQUEST\n" +
          "\n".join(f'"{key}": ' +
                    " " *
                    (maxlen -
                     len(key)) +
                    f"{value}" for key, value in request.items()))
    # Retrieve DATA with settings
    server = ecmwf.ECMWFDataServer()
    server.retrieve(request)
    return

#------------------------------------------------------------------------------#
# Merra
#------------------------------------------------------------------------------#


def merra():
    """
    Download MERRA data. Is this possible?

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError


def ncar():
    """
    Download NCAR CFSL data. Is this possible?

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError

#------------------------------------------------------------------------------#
# Satellite data
#------------------------------------------------------------------------------#


def satellite():
    """
    Download satellite data.

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError

#------------------------------------------------------------------------------#
# CMIP5 data
#------------------------------------------------------------------------------#


def cmip():
    """
    Download CMIP5 model data.

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError
