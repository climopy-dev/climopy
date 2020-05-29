"""
A variety of physical constants.
"""
# WARNING: Putting hyperlink on first line seems to break the sphinx docstring
# because colon is interpreted as start of docstring. Must appear on line 2.
import math
from . import ureg
Quant = ureg.Quantity

__all__ = [
    'G', 'H', 'Md', 'Mw', 'Na', 'Omega', 'R',
    'a', 'c', 'cp', 'cv', 'e', 'g', 'h', 'p0', 'psfc', 'pi', 'tau',
    'Rd', 'Rm',
    'kappa', 'kb', 'sigma',
]

#: `Gravitational constant\
#: <https://en.wikipedia.org/wiki/Gravitational_constant>`__
#: :math:`(\mathrm{m}^3 \cdot \mathrm{kg}^{-1} \cdot \mathrm{s}^{-2})`.
G = Quant(6.67408e-11, 'm^3 kg^-1 s^-2')

#: Rule-of-thumb 7km atmospheric
#: `scale height <http://glossary.ametsoc.org/wiki/Scale_height>`__
#: :math:`(\mathrm{m})`.
H = Quant(7.0e3, 'm')

#: Dry air molar mass :math:`(\mathrm{kg} \cdot \mathrm{mol}^{-1})`.
Md = Quant(28.9645e-3, 'kg mol^-1')

#: Water vapor molar mass :math:`(\mathrm{kg} \cdot \mathrm{mol}^{-1})`.
Mw = Quant(18.0153e-3, 'kg mol^-1')

#: `Avogadro constant\
#: <https://en.wikipedia.org/wiki/Avogadro_constant>`__.
Na = Quant(6.02214e23, 'mol^-1')

#: Earth `rotation rate\
#: <https://en.wikipedia.org/wiki/Earth%27s_rotation#Angular_speed>`__
#: :math:`(\mathrm{rad} \cdot \mathrm{s}^{-1})`.
Omega = Quant(7.292115e-5, 'rad s^-1')

#: `Ideal gas constant\
#: <https://en.wikipedia.org/wiki/Gas_constant>`__
#: :math:`(\mathrm{J} \cdot \mathrm{K}^{-1} \cdot \mathrm{mol}^{-1})`.
R = Quant(8.31446, 'J mol^-1 K^-1')

#: Earth `mean radius\
#: <https://en.wikipedia.org/wiki/Earth_radius#Mean_radius>`__
#: :math:`(\mathrm{m})`.
a = Quant(6.3710088e6, 'm')

#: `Speed of light\
#: <http://glossary.ametsoc.org/wiki/Speed_of_light>`__
#: in a vacuum :math:`(\mathrm{m} \cdot \mathrm{s}^{-1})`.
c = Quant(2.99792458e8, 'm s^-1')

#: `Specific heat capacity\
#: <http://glossary.ametsoc.org/wiki/Specific_heat_capacity>`__
#: at constant pressure for dry air at :math:`0^{\circ}\mathrm{C}`
#: :math:`(\mathrm{J} \cdot \mathrm{kg}^{-1} \cdot \mathrm{K}^{-1})`.
cp = Quant(1.0057e3, 'J kg^-1 K^-1')

#: `Specific heat capacity\
#: <http://glossary.ametsoc.org/wiki/Specific_heat_capacity>`__
#: at constant volume for dry air at :math:`0^{\circ}\mathrm{C}`
#: :math:`(\mathrm{J} \cdot \mathrm{kg}^{-1} \cdot \mathrm{K}^{-1})`.
cv = Quant(0.719e3, 'J kg^-1 K^-1')

#: `Euler's number\
#: <https://en.wikipedia.org/wiki/E_(mathematical_constant)>`__
e = math.e

#: `Standard acceleration due to gravity\
#: <https://en.wikipedia.org/wiki/Standard_gravity>`__
#: :math:`(\mathrm{m} \cdot \mathrm{s}^{-2})`.
g = Quant(9.80665, 'm s^-2')

#: `Planck constant\
#: <http://glossary.ametsoc.org/wiki/Planck%27s_constant>`__
#: :math:`(\mathrm{J} \cdot \mathrm{s})`.
h = Quant(6.62607e-34, 'J s')

#: Earth `mean sea-level pressure\
#: <https://en.wikipedia.org/wiki/Atmospheric_pressure#Mean_sea-level_pressure>`__
#: :math:`(\mathrm{Pa})`.
psfc = Quant(101325.0, 'Pa')

#: Standard reference pressure :math:`(\mathrm{Pa})`
p0 = Quant(1e5, 'Pa')

#: :math:`\pi` (3.14159...)
pi = math.pi

#: :math:`\tau` (6.28318...)
tau = 2.0 * pi

#: Dry air gas constant
#: :math:`(\mathrm{J} \cdot \mathrm{kg}^{-1} \cdot \mathrm{K}^{-1})`.
Rd = R / Md

#: Water vapor gas constant
#: :math:`(\mathrm{J} \cdot \mathrm{kg}^{-1} \cdot \mathrm{K}^{-1})`.
Rm = R / Mw

#: `Poisson constant\
#: <http://glossary.ametsoc.org/wiki/Poisson_constant>`__
#: for dry air. Equivalent to :math:`R_d / c_p`.
kappa = Rd / cp

#: `Boltzmann constant\
#: <https://en.wikipedia.org/wiki/Boltzmann_constant>`__
#: :math:`(\mathrm{J} \cdot \mathrm{K}^{-1})`.
kb = R / Na

#: `Stefan-Boltzmann constant\
#: <https://en.wikipedia.org/wiki/Stefan–Boltzmann_constant>`__
#: :math:`(\mathrm{W} \cdot \mathrm{m}^{-2} \cdot \mathrm{K}^{-4})`.
sigma = ((2 * (pi**5) * (kb**4)) / (15 * (h**3) * (c**2))).to('W K^-4 m^-2')
