#!/usr/bin/env python3
"""
Warnings used internally by climo.
"""
import re
import sys
import warnings

import pint
import xarray as xr

# Store methods
catch_warnings = warnings.catch_warnings  # avoid overwriting module name
simplefilter = warnings.simplefilter  # avoid overwriting module name

# Filter warnings
simplefilter('error', category=pint.UnitStrippedWarning)
simplefilter('ignore', category=xr.core.extensions.AccessorRegistrationWarning)

# Warning class
ClimopyWarning = type('ClimopyWarning', (UserWarning,), {})


def _warn_climopy(message):
    """
    Emit a `ClimopyWarning` and try to show the stack level corresponding
    to user code by jumping to the stack outside of climopy, numpy, scipy,
    pandas, xarray, pint, and scipy.
    """
    frame = sys._getframe()
    stacklevel = 1
    while True:
        if frame is None:
            break  # when called in embedded context may hit frame is None
        if not re.match(
            r'\A(climopy|numpy|scipy|xarray|pandas|pint).',
            frame.f_globals.get('__name__', '')
        ):
            break
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, ClimopyWarning, stacklevel=stacklevel)
