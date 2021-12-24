#!/usr/bin/env python3
"""
Convenient wrappers for python APIs used to download
climatological datasets.

Warning
-------
This submodule out of date and poorly tested. It will eventually be cleaned up.
In the meantime, feel free to copy and modify it.
"""
import calendar

import numpy as np

__all__ = [
    'cmip',
    'era',
    'merra',
    'ncar',
    'satellite',
]


# CMIP constants. Results of get_facet_options() for SearchContext(project='CMIP5')
# and SearchContext(project='CMIP6') using https://esgf-node.llnl.gov/esg-search
# for the SearchConnection URL. Conventions changed between projects so e.g.
# 'experiment', 'ensemble', 'cmor_table', and 'time_frequency' in CMIP5 must be
# changed to 'experiment_id', 'variant_label', 'table_id', and 'frequency' in CMIP6.
# Note 'member_id' is equivalent to 'variant_label' if 'sub_experiment_id' is unset
# and for some reason 'variable' and 'variable_id' are kepts synonyms in CMIP5.
# URL https://esgf-node.llnl.gov/esg-search:     11900116 hits for CMIP6 (use this one!)
# URL https://esgf-data.dkrz.de/esg-search:      01009809 hits for CMIP6
# URL https://esgf-node.ipsl.upmc.fr/esg-search: 01452125 hits for CMIP6
CMIP5_FACETS = [
    'access', 'cera_acronym', 'cf_standard_name', 'cmor_table', 'data_node',
    'ensemble', 'experiment', 'experiment_family', 'forcing', 'format',
    'index_node', 'institute', 'model', 'product', 'realm', 'time_frequency',
    'variable', 'variable_long_name', 'version'
]
CMIP6_FACETS = [
    'access', 'activity_drs', 'activity_id', 'branch_method', 'creation_date',
    'cf_standard_name', 'data_node', 'data_specs_version', 'datetime_end',
    'experiment_id', 'experiment_title', 'frequency', 'grid', 'grid_label',
    'index_node', 'institution_id', 'member_id', 'nominal_resolution', 'realm',
    'short_description', 'source_id', 'source_type', 'sub_experiment_id', 'table_id',
    'variable', 'variable_id', 'variable_long_name', 'variant_label', 'version'
]

# ECMWF constants. Update this list if you modify script for ERA5, etc.
# NOTE: Some variables are technically on "levels" like hybrid
# level surface pressure but we still need 60 "levels".
# TODO: Fix the 12 hour thing. Works for some parameters (e.g. diabatic
# heating, has 3, 6, 9, 12) but other parameters have 0, 6, 12, 18.
ECMWF_LEVOPTS = {
    'ml': range(1, 137 + 1),
    'pl': [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000],  # noqa: E501
    'pt': [265, 270, 285, 300, 315, 330, 350, 370, 395, 430, 475, 530, 600, 700, 850],
    'pv': None,
    'sfc': None,
}
ECMWF_VAROPTS = {
    'tdt': '110.162',  # diabatic temp tendency
    'msp': '152.128',  # model-level surface pressure (use for tdt etc., requires lev=1)
    'sp': '134.128',  # surface pressure
    't2m': '167.128',  # 2 meter air temp
    'd2m': '168.128',  # 2 meter dew point
    'sst': '34.128',  # sea surface temp
    'msl': '151.128',  # sea level pressure
    'slp': '151.128',
    'z': '129.128',  # geopotential
    't': '130.128',  # temp
    'u': '131.128',  # u wind
    'v': '132.128',  # v wind
    'w': '135.128',  # w wind
    'q': '133.128',  # specific humidity
    'r': '157.128',  # relative humidity
    'vo': '138.128',  # relative vorticity
    'vort': '138.128',
    'zeta': '138.128',
    'pt': '3.128',  # potential temp (available on 2PVU surface)
    'theta': '3.128',
    'p': '54.128',  # pressure (availble on potential temp and 2PVU surfaces)
    'pres': '54.128',
    'pv': '60.128',  # potential voriticy (available on pressure and potential temp)
    'precip': '228.128',  # precipitation accumulation
}


def cmip(url=None):
    """
    Download CMIP5 model data.

    Parameters
    ----------
    url : str, default: 'https://esgf-node.llnl.gov/esg-search'
        The search URL.
    """
    # Data requests
    from pyesgf.search import SearchConnection
    if url is None:
        url = 'https://esgf-node.llnl.gov/esg-search'
    conn = SearchConnection(url, distrib=True)
    ctx = conn.new_context()
    return ctx


def era(
    params,
    stream,
    levtype,
    daterange=None,
    yearrange=None,
    monthrange=None,
    # dayrange=None,  # not yet used
    years=None,
    months=None,
    format='netcdf',
    forecast=False,
    step=12,
    levrange=None,
    levs=None,
    grid=None,
    hours=(0, 6, 12, 18),
    hour=None,
    res=None,
    box=None,
    filename='era.nc',
):
    """
    Retrieve ERA reanalysis data using the provided API. User must have a file
    named ``.ecmwfapirc`` in the home directory. Please see the API documentation
    for detalis, but it should look something like this:

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
        Level type: model, pressure, surface, potential
        temperature, and 2PVU surface, respectively.
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
    grid : str, optional
        The grid type. Default is ``N32`` which returns data on 64 latitudes.
    res : float, optional
        Alternative to `grid` that specifies the desired output grid resolution
        in degrees. ERA-Interim has a few valid preset resolutions and will
        choose the resolution that most closely matches the input.
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
    """  # noqa: E501
    # Data stream
    import ecmwfapi as ecmwf  # only do so inside function

    # Variable id conversion (see:
    # https://rda.ucar.edu/datasets/ds627.0/docs/era_interim_grib_table.html)
    if isinstance(params, str) or not np.iterable(params):
        params = (params,)
    params = [ECMWF_VAROPTS.get(p, None) for p in params]
    if None in params:
        raise ValueError('MARS ID for param unknown (consider adding to this script).')
    params = '/'.join(params)

    # Time selection as various ranges or lists
    # Priority. Use daterange as datetime or date objects
    if daterange is not None:
        if not np.iterable(daterange):
            daterange = (daterange,)  # want a single day
        if stream != 'oper':
            y0, m0, y1, m1 = (
                daterange[0].year,
                daterange[0].month,
                daterange[1].year,
                daterange[1].month,
            )
            N = max(y1 - y0 - 1, 0) * 12 + (13 - m0) + m1  # number of months in range
            dates = '/'.join(
                '%04d%02d00' % (y0 + (m0 + n - 1) // 12, (m0 + n - 1) % 12 + 1)
                for n in range(N)
            )
        else:
            # MARS will get calendar days in range
            dates = '/to/'.join(d.strftime('%Y%m%d') for d in daterange)

    # Alternative. List years/months desired, and if synoptic, get all days within
    else:
        # Year specification
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
        # Month specification (helpful for e.g. JJA data)
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
        # Construct dates ranges
        if stream != 'oper':
            dates = '/'.join(
                '/'.join('%04d%02d00' % (y, m) for m in months) for y in years
            )
        else:
            dates = '/'.join(
                '/'.join(
                    '/'.join(
                        '%04d%02d%02d' % (y, m, i + 1)
                        for i in range(calendar.monthrange(y, m)[1])
                    )
                    for m in months
                )
                for y in years
            )

    # Level selection as range or list
    levopts = np.array(ECMWF_LEVOPTS.get(levtype))  # could be np.array(None)
    if not levopts:
        raise ValueError('Invalid level type. Choose from "pl", "pt", "pv", "sfc".')
    if levtype not in ('sfc', 'pv'):  # these have multiple options
        if levs is None and levrange is None:
            raise ValueError(
                'Must specify list of levels with the "levs" keyword, a range of '
                'levels with the "levrange" keyword, or a single level to either one.'
            )
        if levs is not None:
            levs = np.atleast_1d(levs)
        elif not np.iterable(levrange) or len(levrange) == 1:
            levs = np.atleast_1d(levrange)
        else:
            levs = levopts[(levopts >= levrange[0]) & (levopts <= levrange[1])]
        levs = '/'.join(str(l) for l in levs.flat)
    # Resolution (same in latitude/longitude is required for now)
    if res is not None:
        grid = '%.5f/%.5f' % (res, res)
    elif grid is None:
        grid = 'N32'
    # Area specified as pre-defined region (e.g. string 'europe') or n/s/w/e boundary
    if box is not None and not isinstance(box, str):
        box = '/'.join(str(b) for b in (box[3], box[0], box[2], box[1]))
    # Hour conversion
    if not np.iterable(hours):
        hours = (hours,)
    hours = '/'.join(str(h).zfill(2) for h in hours)  # zfill padds 0s on left
    # Forecast type
    if forecast:
        dtype, step = 'fc', str(step)
    else:
        dtype, step = 'an', '0'

    # Server instructions
    # Can also spit raw output into GRIB; apparently ERA-Interim uses
    # bilinear interpolation to make grid of point obs, which makes sense,
    # because their reanalysis model just picks out point observations
    # from spherical harmonics; so maybe grid cell concept is dumb? Maybe
    # need to focus on just using cosine weightings, forget about rest?
    # Not really sure what happens in some situations: list so far:
    # 1. If you provide with variable string-name instead of numeric ID, MARS will
    #    search for correct one; if there is name ambiguity/conflict will throw error.
    # 2. On GUI framework, ECMWF only offers a few resolution options, but program
    #    seems to run when requesting custom resolutions like 5deg/5deg
    request = {
        'class': 'ei',  # ecmwf classifiction; choose ERA-Interim
        'expver': '1',
        'dataset': 'interim',  # thought we already did that; *shrug*
        'type': dtype,  # type of field; analysis 'an' or forecast 'fc'
        'resol': 'av',  # prevents truncation before transformation to geo grid
        'gaussian': 'reduced',
        'format': format,
        'step': step,  # NOTE: ignored for non-forecast type
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
    parts = (f'{k!r}: ' + ' ' * (maxlen - len(k)) + f'{v}' for k, v in request.items())
    print('MARS request:', *parts, sep='\n')
    server = ecmwf.ECMWFDataServer()
    server.retrieve(request)
    return request


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


def satellite():
    """
    Download satellite data.

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError
