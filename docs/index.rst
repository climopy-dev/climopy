.. ClimPy documentation master file, created by
   sphinx-quickstart on Wed Feb 20 21:30:48 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ClimPy
======

This package includes utilities for analyzing and processing geophysical
datasets, including several common statistical and
objective analysis techniques, formulas for derived physical quantities,
and physical constants. The Github page is `here <https://github.com/lukelbd/climpy>`__.

Many atmospheric scientists may already have scripts for data analysis tasks lying around. But this package is meant to be a standardized, well-documented, flexible consolidation of these tasks.

This is a work-in-progress -- currently, there is no formal release
on PyPi. For the time being, you may install directly from Github using:

.. code-block:: bash

   pip install git+https://github.com/lukelbd/climpy.git#egg=climpy

I may also consider merging this project with `MetPy <https://unidata.github.io/MetPy/latest/index.html>`_ at some point. If you are an atmospheric scientist, you should check that project out -- it's awesome. But for the time being, it cannot perform many of the data analysis tasks used by *climate* scientists.

The dependencies are `xarray <http://xarray.pydata.org/en/stable/>`_, `numpy <http://www.numpy.org/>`_, and `scipy <https://www.scipy.org/>`_.
To use the (optional) `ECMWF <https://www.ecmwf.int/>`_ `ERA-Interim <https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/>`_ downloading tool, you will also need to install the `ECMWF python API <https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets>`_.

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Getting Started

   quickstart

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: API Reference

   oa
   diff
   gridtools
   arraytools
   misctools
   downloads
   waq

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
