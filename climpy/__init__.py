#!/usr/bin/env python3
#------------------------------------------------------------------------------#
# Import everything in this folder into a giant module
# Files are segretated by function, so we don't end up with
# giant 5,000-line single file
#------------------------------------------------------------------------------#
from .misctools import *  # misc tools
from .gridtools import *  # managing xarray grids, geo tools
from .downloads import *  # downloading data
from .diff import *      # finite differencing
from .oa import *        # statistical and objective analysis stuff
from . import waq        # wave-activity related stuff, pretty specific
from . import const      # separate submodule
name = 'ClimPy'
