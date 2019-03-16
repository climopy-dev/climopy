.. Docstrings formatted according to:
   numpy guide:      https://numpydoc.readthedocs.io/en/latest/format.html
   matplotlib guide: https://matplotlib.org/devel/documenting_mpl.html
.. Sphinx is used following this guide (less traditional approach):
   https://daler.github.io/sphinxdoc-test/includeme.html

ClimPy
======

This package includes utilities for analyzing and processing datasets
related to atmospheric science, including several common statistical and
objective analysis techniques, formulas for derived physical quantities,
and physical constants.

Many atmospheric scientists may already have scripts for data analysis tasks lying around. But this package is meant to be a **user-friendly**, **well-documented**, **flexible** consolidation of these tasks.

This is a work-in-progress -- currently, there is no formal release
on PyPi. For the time being, you may install directly from Github using:

.. code-block:: bash

   pip install git+https://github.com/lukelbd/climpy.git#egg=climpy

I may also consider merging this project with `MetPy <https://unidata.github.io/MetPy/latest/index.html>`_ at some point. If you are an atmospheric scientist, you should check that project out -- it's awesome. But for the time being, it cannot perform many of the data analysis tasks used by *climate* scientists.

The dependencies are `xarray <http://xarray.pydata.org/en/stable/>`_, `numpy <http://www.numpy.org/>`_, and `scipy <https://www.scipy.org/>`_.
To use the (optional) `ECMWF <https://www.ecmwf.int/>`_ `ERA-Interim <https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/>`_ downloading tool, you will also need to install the `ECMWF python API <https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets>`_.

For a walkthrough of the features, check out the tutorial. Once you get the hang of the API,
see the :ref:`Full documentation`.
