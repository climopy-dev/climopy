.. ClimPy documentation master file, created by
   sphinx-quickstart on Wed Feb 20 21:30:48 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ClimPy
======

This package is meant to help climate scientists complete a variety of data analysis tasks.
It provides handy functions for objective and statistical analysis methods, deriving physical quantities, and downloading archived observational and model data.
The Github page is `here <https://github.com/lukelbd/climpy>`__.

This is a work-in-progress -- currently, there is no formal release
on PyPi. For the time being, you may install directly from Github using:

.. code-block:: bash

   pip install git+https://github.com/lukelbd/climpy.git#egg=climpy

I may consider merging this project with `MetPy <https://unidata.github.io/MetPy/latest/index.html>`_ eventually. But for the time being, MetPy cannot perform many of the objective and statistical analysis tasks used by climate scientists.

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
   :caption: Reference

   api
   changelog
   authors

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
