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

Installation
------------

ClimoPy is published on `PyPi <https://pypi.org/project/climopy/>`__.
It can be installed or upgraded with ``pip`` as follows:

.. code-block:: bash

   pip install climopy
   pip install --upgrade climopy

To install a development version of ClimoPy, you can use
``pip install git+https://github.com/lukelbd/climopy.git``
or clone the repository and run ``pip install -e .`` inside
the ``climopy`` folder.


Documentation
-------------
The documentation is `published on readthedocs <https://climopy.readthedocs.io>`__.


.. |build-status| image:: https://img.shields.io/travis/lukelbd/climopy.svg?style=flat
    :alt: build status
    :target: https://travis-ci.org/lukelbd/climopy

.. |code-style| image:: https://img.shields.io/badge/code%20style-pep8-green.svg
   :alt: pep8
   :target: https://www.python.org/dev/peps/pep-0008/

.. |coverage| image:: https://codecov.io/gh/lukelbd/climopy.org/branch/master/graph/badge.svg
    :alt: coverage
    :target: https://codecov.io/gh/lukelbd/climopy.org

.. |license| image:: https://img.shields.io/github/license/lukelbd/climopy.svg
   :alt: license
   :target: LICENSE.txt

.. |docs| image:: https://readthedocs.org/projects/climopy/badge/?version=latest
    :alt: docs
    :target: https://climopy.readthedocs.io/en/latest/?badge=latest

.. |pr-welcome| image:: https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?
   :alt: PR welcome
   :target: https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project

.. |pypi| image:: https://img.shields.io/pypi/v/climopy?color=83%20197%2052
   :alt: pypi
   :target: https://pypi.org/project/climopy/

..
.. |quality| image:: https://api.codacy.com/project/badge/Grade/931d7467c69c40fbb1e97a11d092f9cd
   :alt: quality
   :target: https://www.codacy.com/app/lukelbd/climopy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=lukelbd/climopy&amp;utm_campaign=Badge_Grade
