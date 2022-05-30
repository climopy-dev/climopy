#!/usr/bin/env python3
"""
A python package for climate scientists.
"""
# Required imports
# NOTE: These modules are for organization and should not be accessible by
# users. However recommended syntax for using unit and variable registries
# and constants library is 'from climopy import ureg, vreg, const'.
import pkg_resources as _pkg
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
from .accessor import *  # noqa: F401, F403
from .internals.quack import *  # noqa: F401, F403
from .internals.quant import *  # noqa: F401, F403
from . import const  # noqa: F401
from . import physics  # noqa: F401
from . import internals  # noqa: F401

# SCM versioning
name = 'climopy'
try:
    version = __version__ = _pkg.get_distribution(__name__).version
except _pkg.DistributionNotFound:
    version = __version__ = 'unknown'
