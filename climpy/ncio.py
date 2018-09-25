#!/usr/bin/env python3
import numpy as np
import xarray as xr

#-------------------------------------------------------------------------------
# NetCDF loading tool
#-------------------------------------------------------------------------------
def nc(filename, param, lonmin=0, times=None, lons=None, lats=None, levs=None, **kwargs):
    """
    Function for loading up NetCDF data, and standardizing dimension names and order,
    and standardizing longitude range (i.e. -180-180 or 0-360). Extremely simple.
    * Also converts time dimension from np.datetime64 to list of python datetime.datetime objects.
    * For slicing, specify coordinate/tuple range/length-3 tuple range + skip
    """
    #--------------------------------------------------------------------------#
    # Helper function; create slice
    def makeslice(arg):
        if arg is None:
            arg = (None,)
        try: iter(arg)
        except TypeError:
            pass
        else:
            arg = slice(*arg)
        return arg # a scalar, Noneslice (select all, or ':'), or range
    #--------------------------------------------------------------------------#
    # Load dataset
    if not os.path.exists(filename):
        raise ValueError(f'{filename} does not exist!')
    with xr.open_dataset(filename, engine='netcdf4', cache=False, **kwargs) as file:
        #----------------------------------------------------------------------#
        # Standardize remaining dimension names (to avoid making copies, must
        # be done on Dataset object, not Dataarray object; the latter has no inplace option)
        querynames = { # time must ALWAYS be named 'time'
                'lat':  ['y','j','lat','latitude','la'],
                'lon':  ['x','i','lon','longitude','lo','long'], # have seen i/j in files
                'lev':  ['z','lev','level','p','pres','pressure','pt','theta','h','height'], # t is 'transport-z'
                } # standardized enough, that this should be pretty safe
        for assignee, options in querynames.items():
            count = 0
            for opt in options:
                if opt in file.indexes:
                    file.rename({opt: assignee}, inplace=True)
                    count += 1
            if count==0:
                if assignee not in ('lev','time'):
                    raise IOError(f'Candidate for "{assignee}" dimension not found.')
            elif count>1:
                raise IOError(f'Multiple candidates for "{assignee}" dimension found.')
        slices = { 'lon': makeslice(lons),
                   'lat': makeslice(lats) }
        data = file[param]
        if 'lev' in data.indexes:
            slices.update(lev=makeslice(levs))
        if 'time' in data.indexes:
            slices.update(time=makeslice(times))
        data = data.sel(**slices) # slice dat shit up yo
            # this action also copies it from filesystem, it seems
        data.load() # load view of DataArray from disk into memory
    #--------------------------------------------------------------------------#
    # Fix precision of time units... some notes:
    # 1) sometimes get weird useless round-off error; convert to days, then restore to numpy datetime64[ns]
    #   because xarray seems to require it
    # 2) ran into mysterious problem where dataset could be loaded, but then
    #   COULD NOT BE SAVED because one of the datetimes wasn't serializable... this was
    #   in normal data, the CCSM4 CMIP5 results, made no sense; range was 1850-2006
    if 'time' in data.indexes:
        data['time'] = data.time.values.astype('datetime64[D]').astype('datetime64[ns]')
    #--------------------------------------------------------------------------#
    # Enforce longitude ordering convention
    # Not always necessary, but this is safe/fast; might as well standardize
    values = data.lon.values-720 # equal mod 360
    while True: # loop only adds 360s to longitudes
        filter_ = values<lonmin
        if filter_.sum()==0: # once finished, write new longitudes and roll
            roll = values.argmin()
            data = data.roll(lon=-roll)
            data['lon'] = np.roll(values, -roll)
            break
        values[filter_] += 360
    #--------------------------------------------------------------------------#
    # Make latitudes monotonic (note extracting values way way faster)
    try: data.lat.values[1]
    except IndexError:
        pass
    else:
        if data.lat.values[0]>data.lat.values[1]:
            data = data.isel(lat=slice(None,None,-1))
    #--------------------------------------------------------------------------#
    # Re-order dims to my expected order before returning
    order = ['time','lev','lat','lon'] # better for numpy broadcasting
    if 'lev' not in data.indexes:
        order.remove('lev')
    if 'time' not in data.indexes:
        order.remove('time')
    data.transpose(*order)
    return data

#-------------------------------------------------------------------------------
# Possible other io wrappers below
#-------------------------------------------------------------------------------

