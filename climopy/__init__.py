#!/usr/bin/env python3
"""
A python package for climate scientists.
"""
# Import functions to top-level. Recommended syntax for registries is one of:
# from climopy import const, units, descriptors
# from climopy import const, ureg, dreg
# NOTE: Submodules are for organization and should not be accessible by users
import pkg_resources as _pkg
from .unit import *  # noqa: F401, F403
from .cfvariable import *  # noqa: F401 F403
from .accessor import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from .diff import *  # noqa: F401, F403
from .var import *  # noqa: F401, F403
from .waves import *  # noqa: F401, F403
from .spectral import *  # noqa: F401, F403
from .spherical import *  # noqa: F401, F403
from .downloads import *  # noqa: F401, F403
from . import const  # noqa: F401
from . import internals  # noqa: F401

# SCM versioning
name = 'climopy'
try:
    version = __version__ = _pkg.get_distribution(__name__).version
except _pkg.DistributionNotFound:
    version = __version__ = 'unknown'
