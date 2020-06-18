Downloading datasets
====================

.. warning::

   These examples are out of date and may no longer work. Please refer
   first to the :ref:`API Reference` until the examples are updated.

Reanalysis downloads
--------------------

Easily download ERA-interim reanalysis data with the
`~climopy.downloads.era` function! Hoping to expand this to provide
tools for downloading MERRA reanalysis data, NCEP-NCAR reanalysis data,
and archived CMIP model data.

.. code-block:: python

   import proplot as plot
   import climopy as climo
   import numpy as np
   levs = range(1, 61)  # all 60 of them, so we can interpolate easily
   hours = (0, 12)  # only hours 0, 12 available; tdt is average over 12 ho
   years = [(1981, 1990), (1991, 2000), (2001, 2010)]
   # levs = 58
   # hours = 12
   # years = [(2010, 2010)]
   for year in years:
       for month in (1, 7):
           # Temperature tendency
           file = f'{base}/mlevs/tdt_{year[0]:04d}-{year[1]:04d}_{month:02d}.grb2'
           print(
               f'\n\nTemperature tendency for years {year}, months {month}, file {file}.'
           )
           climo.era(
               ('tdt', 'msp'),
               'oper',
               'ml',
               levs=levs,
               yearrange=year,
               months=month,
               # days=1, years=year, month=months,
               filename=file,
               grid='F32',
               forecast=True,
               format='grib2',
               # forecast=True, format='netcdf',
               step=12,
               hours=hours,
           )
