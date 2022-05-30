#!/usr/bin/env python3
"""
A python package for climate scientists.
"""
# SCM versioning
import pkg_resources as pkg
name = 'climopy'
try:
    version = __version__ = pkg.get_distribution(__name__).version
except pkg.DistributionNotFound:
    version = __version__ = 'unknown'

# Required imports
# NOTE: These modules are for organization and should not be accessible by
# users. However recommended syntax for using unit and variable registries
# and constants library is 'from climopy import ureg, vreg, const'.
from .unit import *  # noqa: F401, F403
from .context import *  # noqa: F401, F403
from .cfvariable import *  # noqa: F401 F403
from .derivations import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from .diff import *  # noqa: F401, F403
from .var import *  # noqa: F401, F403
from .waves import *  # noqa: F401, F403
from .spectral import *  # noqa: F401, F403
from .spherical import *  # noqa: F401, F403
from .internals.quack import *  # noqa: F401, F403
from .internals.quant import *  # noqa: F401, F403
from . import const  # noqa: F401
from . import physics  # noqa: F401
from . import internals  # noqa: F401

# Optional imports
# NOTE: All other modules use xarray as an optional import but here it is required.
# Using e.g. CLIMOPY_NO_ACCESSORS=True can result in speedup when using climopy
# in processing pipeline, e.g. to handle continuous output of model runs.
import os
_no_accessors = os.environ.get('CLIMOPY_NO_ACCESSORS', '0')
if _no_accessors.isdecimal() and len(_no_accessors) == 1 and int(_no_accessors):
    pass
else:
    try:
        import xarray as xr  # noqa: F401
        import cf_xarray as cf  # noqa: F401
    except ImportError:
        pass
    else:
        from .accessor import *  # noqa: F401, F403
