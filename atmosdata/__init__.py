#!/usr/bin/env python3
#------------------------------------------------------------------------------#
# Import everything in this folder into a giant module
# Files are segretated by function, so we don't end up with
# giant 5,000-line single file
#------------------------------------------------------------------------------#
name = 'AtmosData'
from .analysis import * # statistical and objective analysis stuff
from .geocalc import *  # geography related stuff
from .stats import *
