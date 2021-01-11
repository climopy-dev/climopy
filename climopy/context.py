#!/usr/bin/env python3
"""
Climate-related `phnt.Context` objects.
"""
import functools
import itertools

import pint

from . import const
from .unit import ureg


# Add forward and inverse context definitions for scalar multiple transformations
# NOTE: This feature is extremely limited for now, e.g. defining the transformation
# [length]**2 to [mass] does not work for [length]**-2 to [mass]**-1 and vice versa,
# *additional* units like an extra [joule] cause this to fail, and adding things
# together e.g. with [length]**2 + [mass] fails.
# NOTE: While converters are not multiplicative (do not work for arbitrary additional
# units appended to source and dest) they are commutative. For example first two
# transformations permit converting 'temperature' to 'length'! For this reason
# do not make the context object global; would yield unexpected results.
def _add_transformation(context, source, dest, scale):  # noqa: E302
    """
    Add linear forward and inverse unit transformations.
    """
    context.add_transformation(
        source, dest, functools.partial(lambda scale, _, x: x * scale, scale)
    )
    context.add_transformation(
        dest, source, functools.partial(lambda scale, _, x: x / scale, scale)
    )


# Static energy components, their rates of change, their fluxes, and all of the
# above integrated with respect to mass, area, or mass per unit area.
# TODO: Consider removing latent heat transformations? May result in weird bugs
# where inadvertantly nondimensional data is given restored dimensions!
# NOTE: Pint context transformations are recursive (e.g. below permits converting
# [length] to [temperature]) but do not work with arbitrary additional unit multipliers
# (e.g. below does not cover [length] * [length] to [energy] * [length] / [mass]).
# NOTE: Common to want to convert [energy] / [mass] to [energy] / [area] / [pressure]
# for displaying static or Lorenz energy terms. But this is already covered by
# the geopotential height transformations! Latter units are equivalent to [length]!
# NOTE: The [area] multiplier supports longitudinal integration of meridional
# fluxes per unit pressure (e.g. PW 100hPa-1). Could be thought of as the static energy
# density (in [energy] / [area] / [pressure]), i.e. [length] scaled by the area of the
# zonal band traversed by meridional fluxes in one second. Vertically integrated
# meridional fluxes work without this multiplier because [area] / [time] * [mass] /
# [area] = [mass] / [time] which is already covered. Unsure of other uses.
climo = pint.Context('climo')
for suffix1, suffix2 in itertools.product(
    ('', ' / [time]', ' * [velocity]'),
    ('', ' * [area]', ' * [mass]', ' * [mass] / [area]'),
):
    suffix = suffix1 + suffix2
    _add_transformation(
        climo,
        '[length]' + suffix,  # potential energy (dependent on geopotential height)
        '[energy] / [mass]' + suffix,
        const.g,
    )
    _add_transformation(
        climo,
        '[temperature]' + suffix,  # sensible heat (dependent on temperature)
        '[energy] / [mass]' + suffix,
        const.cp,
    )
    _add_transformation(
        climo,
        '[]' + suffix,  # latent heat (dependent on mixing ratio)
        '[energy] / [mass]' + suffix,
        const.Lv,
    )

# Register context object
ureg.add_context(climo)
