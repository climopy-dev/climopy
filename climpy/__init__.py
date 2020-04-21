#!/usr/bin/env python3
import pkg_resources as _pkg

# Import everything into top-level namespace
from .math import *  # noqa
from .spherical import *  # noqa
from .downloads import *  # noqa
from .diff import *  # noqa
from .oa import *  # noqa
from .waves import *  # noqa
from .const import *  # noqa
from . import cbook

# SCM versioning
name = 'climpy'
try:
    version = __version__ = _pkg.get_distribution(__name__).version
except _pkg.DistributionNotFound:
    version = __version__ = 'unknown'
