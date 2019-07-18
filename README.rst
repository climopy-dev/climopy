.. Docstrings formatted according to:
   numpy guide:      https://numpydoc.readthedocs.io/en/latest/format.html
   matplotlib guide: https://matplotlib.org/devel/documenting_mpl.html
.. Sphinx is used following this guide (less traditional approach):
   https://daler.github.io/sphinxdoc-test/includeme.html

ClimPy
======

|build-status| |coverage| |docs| |license| |pr-welcome|

This package is meant to help geophysical scientists complete a variety of analysis tasks.
It provides handy functions for objective and statisical analysis methods, formulas for deriving physical quantities, and utilities for downloading data.
The documentation is `published on readthedocs <https://climpy.readthedocs.io>`__.

Many atmospheric scientists may already have scripts for data analysis tasks lying around. But this is meant to be a standardized, user-friendly consolidation of these tasks.


This is a work-in-progress -- currently, there is no formal release
on PyPi. For the time being, you may install directly from Github using:

.. code-block:: bash

   pip install git+https://github.com/lukelbd/climpy.git#egg=climpy

I may also consider merging this project with `MetPy <https://unidata.github.io/MetPy/latest/index.html>`_ at some point. If you are an atmospheric scientist, you should check that project out -- it's awesome. But for the time being, it cannot perform many of the data analysis tasks used by *climate* scientists.

The dependencies are `xarray <http://xarray.pydata.org/en/stable/>`_, `numpy <http://www.numpy.org/>`_, and `scipy <https://www.scipy.org/>`_.
To use the (optional) `ECMWF <https://www.ecmwf.int/>`_ `ERA-Interim <https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/>`_ downloading tool, you will also need to install the `ECMWF python API <https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets>`_.

.. |build-status| image:: https://img.shields.io/travis/lukelbd/climpy.svg?style=flat
    :alt: build status
    :target: https://travis-ci.org/lukelbd/climpy

.. |coverage| image:: https://codecov.io/gh/lukelbd/climpy.org/branch/master/graph/badge.svg
    :alt: coverage
    :target: https://codecov.io/gh/lukelbd/climpy.org

.. |license| image:: https://img.shields.io/github/license/lukelbd/climpy.svg
   :alt: license
   :target: LICENSE.txt

.. |docs| image:: https://readthedocs.org/projects/climpy/badge/?version=latest
    :alt: docs
    :target: https://climpy.readthedocs.io/en/latest/?badge=latest

.. |pr-welcome| image:: https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?
   :alt: PR welcome
   :target: https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project

..
.. |quality| image:: https://api.codacy.com/project/badge/Grade/931d7467c69c40fbb1e97a11d092f9cd
   :alt: quality
   :target: https://www.codacy.com/app/lukelbd/proplot?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=lukelbd/proplot&amp;utm_campaign=Badge_Grade

