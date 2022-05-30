.. Docstrings formatted according to:
   numpy guide:      https://numpydoc.readthedocs.io/en/latest/format.html
   matplotlib guide: https://matplotlib.org/devel/documenting_mpl.html
.. Sphinx is used following this guide (less traditional approach):
   https://daler.github.io/sphinxdoc-test/includeme.html

ClimoPy
=======

|build-status| |docs| |pypi| |code-style| |pr-welcome| |license|

This package provides tools to help climate scientists complete a variety of
data analysis tasks, including objective and statistical analysis methods,
1D and 2D spectral decompositions, deriving physical quantities, working with
physical units, and downloading archived observational and model data.
The source code is `published on Github <https://github.com/lukelbd/climopy>`__.

Please note this package is very much a work-in-progress! The examples and
documentation are incomplete, and undiscovered bugs may exist.

Documentation
-------------

The documentation is `published on readthedocs <https://climopy.readthedocs.io>`__.

Installation
------------

Climopy is published on `PyPi <https://pypi.org/project/climopy/>`__.
It can be installed or upgraded with ``pip`` as follows:

.. code-block:: bash

   pip install climopy
   pip install --upgrade climopy

To install a development version of climopy, you can use
``pip install git+https://github.com/lukelbd/climopy.git``
or clone the repository and run ``pip install -e .`` inside
the ``climopy`` folder.



.. |build-status| image:: https://img.shields.io/travis/lukelbd/climopy.svg?style=flat
    :alt: build status
    :target: https://travis-ci.com/lukelbd/climopy

.. |docs| image:: https://readthedocs.org/projects/climopy/badge/?version=latest
    :alt: docs
    :target: https://climopy.readthedocs.io/en/latest/?badge=latest

.. |pypi| image:: https://img.shields.io/pypi/v/climopy?color=83%20197%2052
   :alt: pypi
   :target: https://pypi.org/project/climopy/

.. |code-style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :alt: black
   :target: https://github.com/psf/black

.. |pr-welcome| image:: https://img.shields.io/badge/PR-Welcome-green.svg?
   :alt: PR welcome
   :target: https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project

.. |license| image:: https://img.shields.io/github/license/lukelbd/climopy.svg
   :alt: license
   :target: LICENSE.txt
