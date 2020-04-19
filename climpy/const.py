"""
A variety of physical constants.
"""
# WARNING: Putting hyperlink on first line seems to break the sphinx docstring
# because colon is interpreted as start of docstring. Must appear on line 2.
import math

#: `Gravitational constant\
#: <https://en.wikipedia.org/wiki/Gravitational_constant>`__
#: :math:`(\mathrm{m}^3 \cdot \mathrm{kg}^{-1} \cdot \mathrm{s}^{-2})`.
G = 6.67408e-11

#: Rule-of-thumb 7km atmospheric
#: `scale height <http://glossary.ametsoc.org/wiki/Scale_height>`__
#: :math:`(\mathrm{m})`.
H = 7.0e3

#: Dry air molar mass :math:`(\mathrm{kg} \cdot \mathrm{mol}^{-1})`.
Md = 28.9645e-3

#: Water vapor molar mass :math:`(\mathrm{kg} \cdot \mathrm{mol}^{-1})`.
Mw = 18.0153e-3

#: `Avogadro constant\
#: <https://en.wikipedia.org/wiki/Avogadro_constant>`__.
Na = 6.02214e23

#: Earth `rotation rate\
#: <https://en.wikipedia.org/wiki/Earth%27s_rotation#Angular_speed>`__
#: :math:`(\mathrm{rad} \cdot \mathrm{s}^{-1})`.
Omega = 7.292115e-5

#: `Ideal gas constant\
#: <https://en.wikipedia.org/wiki/Gas_constant>`__
#: :math:`(\mathrm{J} \cdot \mathrm{K}^{-1} \cdot \mathrm{mol}^{-1})`.
R = 8.31446

#: Earth `mean radius\
#: <https://en.wikipedia.org/wiki/Earth_radius#Mean_radius>`__
#: :math:`(\mathrm{m})`.
a = 6.3710088e6

#: `Speed of light\
#: <http://glossary.ametsoc.org/wiki/Speed_of_light>`__
#: in a vacuum :math:`(\mathrm{m} \cdot \mathrm{s}^{-1})`.
c = 2.99792458e8

#: `Specific heat capacity\
#: <http://glossary.ametsoc.org/wiki/Specific_heat_capacity>`__
#: at constant pressure for dry air at :math:`0^{\circ}\mathrm{C}`
#: :math:`(\mathrm{J} \cdot \mathrm{kg}^{-1} \cdot \mathrm{K}^{-1})`.
cp = 1.0057e3

#: `Specific heat capacity\
#: <http://glossary.ametsoc.org/wiki/Specific_heat_capacity>`__
#: at constant pressure for dry air at :math:`0^{\circ}\mathrm{C}`
#: :math:`(\mathrm{J} \cdot \mathrm{kg}^{-1} \cdot \mathrm{K}^{-1})`.
cv = 0.719e3

#: `Standard acceleration due to gravity\
#: <https://en.wikipedia.org/wiki/Standard_gravity>`__
#: :math:`(\mathrm{m} \cdot \mathrm{s}^{-2})`.
g = 9.80665

#: `Planck constant\
#: <http://glossary.ametsoc.org/wiki/Planck%27s_constant>`__
#: :math:`(\mathrm{J} \cdot \mathrm{s}^{-1})`.
h = 6.62607e-34

#: Earth `mean sea-level pressure\
#: <https://en.wikipedia.org/wiki/Atmospheric_pressure#Mean_sea-level_pressure>`__
#: :math:`(\mathrm{Pa})`.
p0 = 101325

#: :math:`\pi` (3.14159...)
pi = math.pi

#: Seconds per day.
sec_per_day = 3600 * 24

#: Dry air gas constant
#: :math:`(\mathrm{J} \cdot \mathrm{kg}^{-1} \cdot \mathrm{K}^{-1})`.
Rd = R / Md

#: Water vapor gas constant
#: :math:`(\mathrm{J} \cdot \mathrm{kg}^{-1} \cdot \mathrm{K}^{-1})`.
Rm = R / Mw

#: `Poisson constant\
#: <http://glossary.ametsoc.org/wiki/Poisson_constant>`__
#: for dry air.
kappa = Rd / cp

#: `Boltzmann constant\
#: <https://en.wikipedia.org/wiki/Boltzmann_constant>`__
#: :math:`(\mathrm{J} \cdot \mathrm{K}^{-1})`.
kb = R / Na

#: `Stefan-Boltzmann constant\
#: <https://en.wikipedia.org/wiki/Stefan–Boltzmann_constant>`__
#: :math:`(\mathrm{W} \cdot \mathrm{m}^{-2} \cdot \mathrm{K}^{-4})`.
sigma = (2 * (pi**5) * (kb**4)) / (15 * (h**3) * (c**2))
