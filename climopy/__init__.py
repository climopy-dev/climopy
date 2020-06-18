#!/usr/bin/env python3
"""
Instantiate the `pint.UnitRegistry` unit registry.
"""
# Import functions to top-level
# NOTE: Syntax is 'from climopy import ureg, const'
import pkg_resources as _pkg
from .units import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from .spherical import *  # noqa: F401, F403
from .downloads import *  # noqa: F401, F403
from .diff import *  # noqa: F401, F403
from .var import *  # noqa: F401, F403
from .spectral import *  # noqa: F401, F403
from .waves import *  # noqa: F401, F403
from . import const  # noqa: F401
from . import internals  # noqa: F401

# SCM versioning
name = 'climopy'
try:
    version = __version__ = _pkg.get_distribution(__name__).version
except _pkg.DistributionNotFound:
    version = __version__ = 'unknown'
