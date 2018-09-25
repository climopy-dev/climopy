#!/usr/bin/env python3
#------------------------------------------------------------------------------#
# Import everything in this folder into a giant module
# Files are segretated by function, so we don't end up with
# giant 5,000-line single file
#------------------------------------------------------------------------------#
name = 'ClimPy'
from . import const      # separate submodule
from . import waq        # wave-activity related stuff, pretty specific
from .diff import *      # finite differencing
from .oa import *        # statistical and objective analysis stuff
from .ncio import *      # data input/output
from .downloads import * # downloading data
from .geotools import *  # geography related stuff
from .misctools import * # misc tools

