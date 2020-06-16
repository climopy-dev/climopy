"""
A variety of physical constants.
"""
# WARNING: Putting hyperlink on first line seems to break the sphinx docstring
# because colon is interpreted as start of docstring. Must appear on line 2.
import math
import pint
import functools
import itertools
from .units import ureg
Quant = ureg.Quantity

__all__ = [
    'G', 'H', 'Md', 'Mw', 'Na', 'Omega', 'R',
    'a', 'c', 'cp', 'cv', 'e', 'g', 'h', 'p0', 'psfc', 'pi', 'tau',
    'Rd', 'Rm',
    'kappa', 'kb', 'sigma',
]

#: `Gravitational constant\
#: <https://en.wikipedia.org/wiki/Gravitational_constant>`__
G = Quant(6.67408e-11, 'm^3 kg^-1 s^-2')

#: Rule-of-thumb 7km atmospheric
#: `scale height <http://glossary.ametsoc.org/wiki/Scale_height>`__
H = Quant(7.0e3, 'm')

#: Dry air molar mass
Md = Quant(28.9645e-3, 'kg mol^-1')

#: Water vapor molar mass
Mw = Quant(18.0153e-3, 'kg mol^-1')

#: `Avogadro constant\
#: <https://en.wikipedia.org/wiki/Avogadro_constant>`__.
Na = Quant(6.02214e23, 'mol^-1')

#: Earth `rotation rate\
#: <https://en.wikipedia.org/wiki/Earth%27s_rotation#Angular_speed>`__
Omega = Quant(7.292115e-5, 'rad s^-1')

#: `Ideal gas constant\
#: <https://en.wikipedia.org/wiki/Gas_constant>`__
R = Quant(8.31446, 'J mol^-1 K^-1')

#: Earth `mean radius\
#: <https://en.wikipedia.org/wiki/Earth_radius#Mean_radius>`__
a = Quant(6.3710088e6, 'm')

#: `Speed of light\
#: <http://glossary.ametsoc.org/wiki/Speed_of_light>`__ in a vacuum
c = Quant(2.99792458e8, 'm s^-1')

#: `Specific heat capacity\
#: <http://glossary.ametsoc.org/wiki/Specific_heat_capacity>`__
#: at constant pressure for dry air at :math:`0^{\circ}\mathrm{C}`
cp = Quant(1.0057e3, 'J kg^-1 K^-1')

#: `Specific heat capacity\
#: <http://glossary.ametsoc.org/wiki/Specific_heat_capacity>`__
#: at constant volume for dry air at :math:`0^{\circ}\mathrm{C}`
cv = Quant(0.719e3, 'J kg^-1 K^-1')

#: `Euler's number\
#: <https://en.wikipedia.org/wiki/E_(mathematical_constant)>`__
e = math.e

#: `Standard acceleration due to gravity\
#: <https://en.wikipedia.org/wiki/Standard_gravity>`__
g = Quant(9.80665, 'm s^-2')

#: `Planck constant\
#: <http://glossary.ametsoc.org/wiki/Planck%27s_constant>`__
h = Quant(6.62607e-34, 'J s')

#: Earth `mean sea-level pressure\
#: <https://en.wikipedia.org/wiki/Atmospheric_pressure#Mean_sea-level_pressure>`__
psfc = Quant(101325.0, 'Pa')

#: Standard reference pressure
p0 = Quant(1e5, 'Pa')

#: :math:`\pi` (3.14159...)
pi = math.pi

#: :math:`\tau` (6.28318...)
tau = 2.0 * pi

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

# Add context definitions for otherwise impossible definitions
# NOTE: This feature is extremely limited for now, e.g. defining the transformation
# [length]**2 to [mass] does not work for [length]**-2 to [mass]**-1 and vice versa,
# *additional* units like an extra [joule] cause this to fail, and adding things
# together e.g. with [length]**2 + [mass] fails.
context = pint.Context('climpy')

# Transform temperature to heat energy
def _add_transformation(source, dest, scale):
    """
    Add a custom unit transformation.
    """
    context.add_transformation(
        source, dest, functools.partial(lambda scale, ureg, x : x * scale, scale)
    )
    context.add_transformation(
        dest, source, functools.partial(lambda scale, ureg, x : x / scale, scale)
    )

# Dry static energy components, their rates of change (1/s), their fluxes (m/s),
# and their *absolute* fluxes integrated over the latitude band (m^2/s)
# Thermal energy
_add_transformation(
    '[temperature]',
    '[energy] / [mass]',
    cp,
)
_add_transformation(
    '[temperature] / [time]',
    '[power] / [mass]',
    cp,
)
_add_transformation(
    '[temperature] * [length] / [time]',
    '[power] * [length] / [mass]',
    cp,
)
_add_transformation(
    '[temperature] * [area] / [time]',
    '[power] * [area] / [mass]',
    cp,
)

# Geopotential energy
_add_transformation(
    '[length]',
    '[energy] / [mass]',
    g,
)
_add_transformation(
    '[length] / [time]',
    '[power] / [mass]',
    g,
)
_add_transformation(
    '[length] * [length] / [time]',
    '[power] * [length] / [mass]',
    g,
)
_add_transformation(
    '[length] * [area] / [time]',
    '[power] * [area] / [mass]',
    g,
)

# Static energy or Lorenz energy measured in energy per unit mass
_add_transformation(
    '[energy] / [mass]',
    '[energy] / [area] / [pressure]',
    1.0 / g,
)
_add_transformation(
    '[power] / [mass]',
    '[power] / [area] / [pressure]',
    1.0 / g,
)
_add_transformation(
    '[power] * [length] / [mass]',
    '[power] * [length] / [area] / [pressure]',
    1.0 / g,
)
_add_transformation(
    '[power] * [area] / [mass]',
    '[power] / [pressure]',
    1.0 / g,
)

# Thermal energy integrated with respect to pressure
_add_transformation(
    '[temperature] * [pressure]',
    '[energy] / [area]',
    cp / g,
)
_add_transformation(
    '[temperature] * [pressure] / [time]',
    '[power] / [area]',
    cp / g,
)
_add_transformation(
    '[temperature] * [pressure] * [length] / [time]',
    '[power] * [length] / [area]',
    cp / g,
)
_add_transformation(
    '[temperature] * [pressure] * [area] / [time]',
    '[power]',
    cp / g,
)

# Geopotential or static energy integrated with respect to pressure
# NOTE: Converting integrated geopotential height m * hPa to J / m^2 does
# not need extra transformation. Functionally this is (height * g) / g to get
# geopotential then convert the pressure integration to a mass integration.
_add_transformation(
    '[energy] * [pressure] / [mass]',
    '[energy] / [area]',
    1.0 / g,
)
_add_transformation(
    '[power] * [pressure] / [mass]',
    '[power] / [area]',
    1.0 / g,
)
_add_transformation(
    '[power] * [pressure] * [length] / [mass]',
    '[power] * [length] / [area]',
    1.0 / g,
)
_add_transformation(
    '[power] * [pressure] * [area] / [mass]',
    '[power]',
    1.0 / g,
)

# Add context object
ureg.enable_contexts(context)
