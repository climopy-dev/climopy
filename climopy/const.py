"""
A variety of physical constants.
"""
# WARNING: Putting hyperlink on first line seems to break the sphinx docstring
# because colon is interpreted as start of docstring. Must appear on line 2.
import math

from .unit import ureg

__all__ = [
    'a', 'c', 'cp', 'cv', 'e', 'eps', 'G', 'g', 'H', 'h',
    'kappa', 'kb', 'Lf', 'Ls', 'Lv', 'Md', 'Mw', 'Na', 'Omega',
    'p0', 'pi', 'psfc', 'R', 'Rd', 'Rm', 'sigma', 'tau',
]

#: `Gravitational constant\
#: <https://glossary.ametsoc.org/wiki/Gravitation>`__
G = ureg.Quantity(6.67408e-11, 'm^3 kg^-1 s^-2')

#: Rule-of-thumb 7km atmospheric
#: `scale height <http://glossary.ametsoc.org/wiki/Scale_height>`__
H = ureg.Quantity(7.0e3, 'm')

#: Dry air molar mass
Md = ureg.Quantity(28.9645e-3, 'kg mol^-1')

#: Water vapor molar mass
Mw = ureg.Quantity(18.0153e-3, 'kg mol^-1')

#: `Avogadro constant\
#: <https://glossary.ametsoc.org/wiki/Avogadro%27s_number>`__.
Na = ureg.Quantity(6.02214e23, 'mol^-1')

#: Earth `rotation rate\
#: <https://glossary.ametsoc.org/wiki/Angular_velocity_of_the_earth>`__
Omega = ureg.Quantity(7.292115e-5, 'rad s^-1')

#: `Ideal gas constant\
#: <https://glossary.ametsoc.org/wiki/Gas_constant>`__
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
#: <https://glossary.ametsoc.org/wiki/Latent_heat>`__
#: at :math:`0^{\circ}\mathrm{C}`
Lv = ureg.Quantity(2.501e6, 'J kg^-1')

#: `Latent heat of fusion\
#: <https://glossary.ametsoc.org/wiki/Latent_heat>`__
#: at :math:`0^{\circ}\mathrm{C}`
Lf = ureg.Quantity(3.371e5, 'J kg^-1')

#: `Latent heat of sublimation\
#: <https://glossary.ametsoc.org/wiki/Latent_heat>`__
#: at :math:`0^{\circ}\mathrm{C}`
Ls = ureg.Quantity(2.834e6, 'J kg^-1')

#: `Euler's number\
#: <https://en.wikipedia.org/wiki/E_(mathematical_constant)>`__
e = ureg.Quantity(math.e, '')

#: `Standard acceleration due to gravity\
#: <https://glossary.ametsoc.org/wiki/Standard_gravity>`__
g = ureg.Quantity(9.80665, 'm s^-2')

#: `Planck constant\
#: <http://glossary.ametsoc.org/wiki/Planck%27s_constant>`__
h = ureg.Quantity(6.62607e-34, 'J s')

#: `Standard atmospheric pressure\
#: <https://glossary.ametsoc.org/wiki/Standard_atmospheric_pressure>`__
psfc = ureg.Quantity(101325.0, 'Pa')

#: Standard reference pressure
p0 = ureg.Quantity(1e5, 'Pa')

#: :math:`\pi` (3.14159...)
pi = ureg.Quantity(math.pi, '')

#: :math:`\tau` (6.28318...)
tau = ureg.Quantity(2.0 * math.pi, '')

#: Standard `liquid water density\
#: <https://glossary.ametsoc.org/wiki/Standard_density>`__
rhow = ureg.Quantity(999.8, 'kg m^-3')

#: Dry air gas constant
#: (equivalent to :math:`R / M_d`)
Rd = R / Md

#: Water vapor gas constant
#: (equivalent to :math:`R / M_w`)
Rm = R / Mw

#: Ratio of molar masses of vapor and dry air (used in
#: converting vapor pressures to mass mixing ratios)
eps = Mw / Md

#: `Poisson constant\
#: <http://glossary.ametsoc.org/wiki/Poisson_constant>`__
#: for dry air (equivalent to :math:`R_d / c_p`)
kappa = Rd / cp

#: `Boltzmann constant\
#: <https://en.wikipedia.org/wiki/Boltzmann_constant>`__
#: (equivalent to :math:`R / N_a`)
kb = R / Na

#: `Stefan-Boltzmann constant\
#: <https://en.wikipedia.org/wiki/Stefan–Boltzmann_constant>`__
sigma = ((2 * (pi ** 5) * (kb ** 4)) / (15 * (h ** 3) * (c ** 2))).to('W K^-4 m^-2')
