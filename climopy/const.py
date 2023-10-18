"""
A variety of physical constants.
"""
# WARNING: Putting hyperlink on first line seems to break the sphinx docstring
# because colon is interpreted as start of docstring. Must appear on line 2.
import math

from .unit import ureg

__all__ = [
    'a', 'Omega', 'g', 'G', 'H', 'p0', 'psfc', 'rhoa', 'rhow',
    'Ma', 'Mo', 'cp', 'cv', 'cw', 'Ca', 'Co', 'Lv', 'Lf', 'Ls',
    'Md', 'Mw', 'Mc', 'Na', 'R', 'Rd', 'Rm', 'kb', 'kappa', 'eps',
    'h', 'c', 'sigma', 'e', 'pi', 'tau',
]

#: Earth `mean radius\
#: <https://en.wikipedia.org/wiki/Earth_radius#Mean_radius>`__
a = ureg.Quantity(6.3710088e6, 'm')

#: Earth `rotation rate\
#: <https://glossary.ametsoc.org/wiki/Angular_velocity_of_the_earth>`__
Omega = ureg.Quantity(7.292115e-5, 'rad s^-1')

#: `Standard acceleration due to gravity\
#: <https://glossary.ametsoc.org/wiki/Standard_gravity>`__
g = ureg.Quantity(9.80665, 'm s^-2')

#: `Gravitational constant\
#: <https://glossary.ametsoc.org/wiki/Gravitation>`__
G = ureg.Quantity(6.67408e-11, 'm^3 kg^-1 s^-2')

#: Rule-of-thumb 7km atmospheric
#: `scale height <http://glossary.ametsoc.org/wiki/Scale_height>`__
H = ureg.Quantity(7.0e3, 'm')

#: `Standard reference pressure\
#: <https://glossary.ametsoc.org/wiki/Standard_pressure>`__
p0 = ureg.Quantity(1e5, 'Pa')

#: `Standard atmospheric pressure\
#: <https://glossary.ametsoc.org/wiki/Standard_atmospheric_pressure>`__
psfc = ureg.Quantity(101325.0, 'Pa')

#: Density of atmospheric column per square meter
#: (equivalent to :math:`p_{\mathrm{sfc}} / g`).
rhoa = (psfc / g).to('kg m^-2')

#: `Standard liquid water density\
#: <https://glossary.ametsoc.org/wiki/Standard_density>`__
rhow = ureg.Quantity(999.8, 'kg m^-3')

#: Mass of entire atmosphere
#: (equivalent to :math:`4 \pi a^2 p_{\mathrm{sfc}} / g`)
Ma = (4 * math.pi * a ** 2 * psfc / g).to('Pg')

#: Mass of entire ocean per meter depth
#: (equivalent to :math:`4 \pi a^2 \rho_w`)
Mo = (4 * math.pi * a ** 2 * rhow).to('Pg / m')

#: `Specific heat capacity\
#: <http://glossary.ametsoc.org/wiki/Specific_heat_capacity>`__
#: at constant pressure for dry air at :math:`0^{\circ}\mathrm{C}`
cp = ureg.Quantity(1.0057e3, 'J kg^-1 K^-1')

#: `Specific heat capacity\
#: <http://glossary.ametsoc.org/wiki/Specific_heat_capacity>`__
#: at constant volume for dry air at :math:`0^{\circ}\mathrm{C}`
cv = ureg.Quantity(0.719e3, 'J kg^-1 K^-1')

#: `Specific heat capacity\
#: <https://thermexcel.com/english/tables/eau_atm.htm>`__
#: of liquid water at :math:`20^{\circ}\mathrm{C}`.
cw = ureg.Quantity(4.184e3, 'J kg^-1 K^-1')

#: Heat capacity of atmospheric column per square meter
#: (equivalent to :math:`c_p p_{\mathrm{sfc}} / g`).
Ca = (cp * psfc / g).to('J m^-2 K^-1')

#: Heat capacity of ocean per square meter per meter depth
#: (equivalent to :math:`c_w \rho_w`).
Co = (cw * rhow).to('J m^-3 K^-1')

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

#: `Dry air molar mass\
#: <https://glossary.ametsoc.org/wiki/Standard_atmosphere>`__
Md = ureg.Quantity(28.9645e-3, 'kg mol^-1')

#: `Water vapor molar mass\
#: <https://en.wikipedia.org/wiki/Water_vapor>`__
Mw = ureg.Quantity(18.0153e-3, 'kg mol^-1')

#: `Carbon molar mass\
#: <https://en.wikipedia.org/wiki/carbon>`__
Mc = ureg.Quantity(12.0096e-3, 'kg mol^-1')

#: `Carbon dioxide molar mass\
#: <https://en.wikipedia.org/wiki/Carbon_dioxide>`__
M2 = ureg.Quantity(44.0090e-3, 'kg mol^-1')

#: `Avogadro constant\
#: <https://glossary.ametsoc.org/wiki/Avogadro%27s_number>`__
Na = ureg.Quantity(6.02214e23, 'mol^-1')

#: `Ideal gas constant\
#: <https://glossary.ametsoc.org/wiki/Gas_constant>`__
R = ureg.Quantity(8.31446, 'J mol^-1 K^-1')

#: Dry air gas constant
#: (equivalent to :math:`R / M_d`)
Rd = R / Md

#: Water vapor gas constant
#: (equivalent to :math:`R / M_w`)
Rm = R / Mw

#: `Boltzmann constant\
#: <https://en.wikipedia.org/wiki/Boltzmann_constant>`__
#: (equivalent to :math:`R / N_a`)
kb = R / Na

#: `Poisson constant\
#: <http://glossary.ametsoc.org/wiki/Poisson_constant>`__
#: for dry air (equivalent to :math:`R_d / c_p`)
kappa = Rd / cp

#: Ratio of molar masses of vapor and dry air (used in
#: converting vapor pressures to mass mixing ratios)
eps = Mw / Md

#: `Planck constant\
#: <http://glossary.ametsoc.org/wiki/Planck%27s_constant>`__
h = ureg.Quantity(6.62607e-34, 'J s')

#: `Speed of light\
#: <http://glossary.ametsoc.org/wiki/Speed_of_light>`__ in a vacuum
c = ureg.Quantity(2.99792458e8, 'm s^-1')

#: `Stefan-Boltzmann constant\
#: <https://en.wikipedia.org/wiki/Stefan–Boltzmann_constant>`__
sigma = ((2 * (math.pi ** 5) * (kb ** 4)) / (15 * (h ** 3) * (c ** 2))).to('W K^-4 m^-2')  # noqa: E501

#: `Euler's number\
#: <https://en.wikipedia.org/wiki/E_(mathematical_constant)>`__
e = ureg.Quantity(math.e, '')

#: :math:`\pi` (3.14159...)
pi = ureg.Quantity(math.pi, '')

#: :math:`\tau` (6.28318...)
tau = ureg.Quantity(2.0 * math.pi, '')
