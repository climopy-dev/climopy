"""
A variety of physical constants.
"""
# WARNING: Putting hyperlink on first line seems to break the sphinx docstring
# because colon is interpreted as start of docstring. Must appear on line 2.
import functools
import itertools
import math

import pint

from .unit import ureg

__all__ = [
    'G', 'H', 'Md', 'Mw', 'Na', 'Omega', 'R',
    'a', 'c', 'cp', 'cv', 'e', 'g', 'h', 'p0', 'psfc', 'pi', 'tau',
    'Rd', 'Rm',
    'kappa', 'kb', 'sigma',
]

#: `Gravitational constant\
#: <https://en.wikipedia.org/wiki/Gravitational_constant>`__
G = ureg.Quantity(6.67408e-11, 'm^3 kg^-1 s^-2')

#: Rule-of-thumb 7km atmospheric
#: `scale height <http://glossary.ametsoc.org/wiki/Scale_height>`__
H = ureg.Quantity(7.0e3, 'm')

#: Dry air molar mass
Md = ureg.Quantity(28.9645e-3, 'kg mol^-1')

#: Water vapor molar mass
Mw = ureg.Quantity(18.0153e-3, 'kg mol^-1')

#: `Avogadro constant\
#: <https://en.wikipedia.org/wiki/Avogadro_constant>`__.
Na = ureg.Quantity(6.02214e23, 'mol^-1')

#: Earth `rotation rate\
#: <https://en.wikipedia.org/wiki/Earth%27s_rotation#Angular_speed>`__
Omega = ureg.Quantity(7.292115e-5, 'rad s^-1')

#: `Ideal gas constant\
#: <https://en.wikipedia.org/wiki/Gas_constant>`__
R = ureg.Quantity(8.31446, 'J mol^-1 K^-1')

#: Earth `mean radius\
#: <https://en.wikipedia.org/wiki/Earth_radius#Mean_radius>`__
a = ureg.Quantity(6.3710088e6, 'm')

#: `Speed of light\
#: <http://glossary.ametsoc.org/wiki/Speed_of_light>`__ in a vacuum
c = ureg.Quantity(2.99792458e8, 'm s^-1')

#: `Specific heat capacity\
#: <http://glossary.ametsoc.org/wiki/Specific_heat_capacity>`__
#: at constant pressure for dry air at :math:`0^{\circ}\mathrm{C}`
cp = ureg.Quantity(1.0057e3, 'J kg^-1 K^-1')

#: `Specific heat capacity\
#: <http://glossary.ametsoc.org/wiki/Specific_heat_capacity>`__
#: at constant volume for dry air at :math:`0^{\circ}\mathrm{C}`
cv = ureg.Quantity(0.719e3, 'J kg^-1 K^-1')

#: `Latent heat of vaporization\
#:  <https://glossary.ametsoc.org/wiki/Latent_heat>`__
#: at :math:`0^{\circ}\mathrm{C}`
Lv = ureg.Quantity(2.501e6, 'J kg^-1')

#: `Latent heat of fusion\
#:  <https://glossary.ametsoc.org/wiki/Latent_heat>`__
#: at :math:`0^{\circ}\mathrm{C}`
Lf = ureg.Quantity(3.371e5, 'J kg^-1')

#: `Latent heat of sublimation\
#:  <https://glossary.ametsoc.org/wiki/Latent_heat>`__
#: at :math:`0^{\circ}\mathrm{C}`
Ls = ureg.Quantity(2.834e6, 'J kg^-1')

#: `Euler's number\
#: <https://en.wikipedia.org/wiki/E_(mathematical_constant)>`__
e = ureg.Quantity(math.e, '')

#: `Standard acceleration due to gravity\
#: <https://en.wikipedia.org/wiki/Standard_gravity>`__
g = ureg.Quantity(9.80665, 'm s^-2')

#: `Planck constant\
#: <http://glossary.ametsoc.org/wiki/Planck%27s_constant>`__
h = ureg.Quantity(6.62607e-34, 'J s')

#: Earth `mean sea-level pressure\
#: <https://en.wikipedia.org/wiki/Atmospheric_pressure#Mean_sea-level_pressure>`__
psfc = ureg.Quantity(101325.0, 'Pa')

#: Standard reference pressure
p0 = ureg.Quantity(1e5, 'Pa')

#: :math:`\pi` (3.14159...)
pi = ureg.Quantity(math.pi, '')

#: :math:`\tau` (6.28318...)
tau = ureg.Quantity(2.0 * math.pi, '')

#: Dry air gas constant
Rd = R / Md

#: Water vapor gas constant
Rm = R / Mw

#: `Poisson constant\
#: <http://glossary.ametsoc.org/wiki/Poisson_constant>`__
#: for dry air. Equivalent to :math:`R_d / c_p`.
kappa = Rd / cp

#: `Boltzmann constant\
#: <https://en.wikipedia.org/wiki/Boltzmann_constant>`__
kb = R / Na

#: `Stefan-Boltzmann constant\
#: <https://en.wikipedia.org/wiki/Stefan–Boltzmann_constant>`__
sigma = ((2 * (pi**5) * (kb**4)) / (15 * (h**3) * (c**2))).to('W K^-4 m^-2')


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
# above integrated with respect to mass or mass per unit area.
# TODO: Consider adding latent heat transformations? May result in weird bugs
# where inadvertantly nondimensional data is given restored dimensions!
# NOTE: Pint context transformations are recursive (e.g. below permits converting
# [length] to [temperature]) but not multiplicative (e.g. below does not cover
# converting [temperature] * [mass] to [energy]).
# NOTE: Common to want to convert [energy] / [mass] to [energy] / [area] / [pressure]
# for displaying static or Lorenz energy terms. But this is already covered by
# the geopotential height transformations! Latter units are equivalent to [length]!
climo = pint.Context('climo')
for suffix1, suffix2 in itertools.product(
    ('', ' / [time]', ' * [velocity]'),
    ('', ' * [mass]', ' * [mass] / [area]'),
):
    suffix = suffix1 + suffix2
    _add_transformation(
        climo,
        '[length]' + suffix,  # potential energy (dependent on geopotential height)
        '[energy] / [mass]' + suffix,
        g,
    )
    _add_transformation(
        climo,
        '[temperature]' + suffix,  # sensible heat (dependent on temperature)
        '[energy] / [mass]' + suffix,
        cp,
    )
    _add_transformation(
        climo,
        '[]' + suffix,  # latent heat (dependent on mixing ratio)
        '[energy] / [mass]' + suffix,
        Lv,
    )

# Register context object
ureg.add_context(climo)
