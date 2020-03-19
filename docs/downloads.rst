Data downloads
==============

Reanalysis downloads
--------------------

Easily download ERA-interim reanalysis data with the
`~climpy.downloads.eraint` function! Hoping to expand this to provide
tools for downloading MERRA reanalysis data, NCEP-NCAR reanalysis data,
and archived CMIP model data.

.. code:: ipython3

    # Load
    # NOTE: For now just get July and January values, store in same place
    # NOTE: Full decade is only 1GB or so, so we store one file
    import proplot as plot
    import climpy
    import numpy as np
    plot.nbsetup()
    levs = range(1,61) # all 60 of them, so we can interpolate easily
    hours = (0, 12) # only hours 0, 12 available; tdt is average over 12 ho
    urs it seems
    years = [(1981, 1990), (1991, 2000), (2001, 2010)]
    # For testing
    # levs = 58
    # hours = 12
    # years = [(2010, 2010)]
    for year in years:
        for month in (1,7):
            # Temperature tendency
            filename = f'{base}/mlevs/tdt_{year[0]:04d}-{year[1]:04d}_{month:02d}.grb2'
            print(f'\n\n\nTemperature tendency for years {year}, months {month}, file {filename}.')
            climpy.eraint(('tdt','msp'), 'oper', 'ml', levs=levs,
                    yearrange=year, months=month,
                    # days=1, # for testing
                    # years=year, month=months,
                    filename=filename, grid='F32',
                    forecast=True, format='grib2',
                    # forecast=True, format='netcdf',
                    step=12, hours=hours)

