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

The dependencies are `xarray <http://xarray.pydata.org/en/stable/>`_, `numpy
<http://www.numpy.org/>`_, and `scipy <https://www.scipy.org/>`_. To use the (optional)
`ECMWF <https://www.ecmwf.int/>`_ `ERA-Interim
<https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/>`_ downloading
tool, you will also need to install the `ECMWF python API
<https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets>`_.

..
  Note that I may consider merging this project with `MetPy
  <https://unidata.github.io/MetPy/latest/index.html>`__ eventually. But for the time
  being, MetPy cannot perform many of the objective and statistical analysis tasks used
  by climate scientists.
