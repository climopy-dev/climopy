"""
A variety of physical constants.
"""
#: :math:`\pi` (3.14159...)
from math import pi

#: Earth radius in meters.
a = 6.371e6

#: Earth rotation rate in rad / s.
Omega = 7.292115e-5

#: Scale height rule-of-thumb in m.
H = 7.0e3

#: Standard gravity in m / s^2.
g = 9.80665

#: Gravitational constant in m^3 / kg * s^2.
G = 6.67408e-11

#: Ideal gas constant in J / K * mol.
R = 8.31446

#: Avo's number.
Na = 6.02214e23

#: Planck constant in J / s.
h = 6.62607e-34

#: Speed of light in m / s.
c = 2.99792458e8

#: Dry air molar mass in kg / mol.
Md = 28.9645e-3

#: Water vapor molar mass in kg / mol.
Mw = 18.0153e-3

#: Specific heat at T = 300K in J / kg * K.
cp = 1.005e3

#: Surface pressure in Pa.
p0 = 101325

#: Boltzmann constant in J / K.
kb = R / Na

#: Dry air gas constant in J / kg * K.
Rd = R / Md

#: Water vapor gas constant in J / kg * K.
Rm = R / Mw

#: Poisson constant.
kappa = Rd / cp

#: Stefan-Boltzmann constant in W / m^2 * K^4.
sigma = (2 * (pi**5) * (kb**4)) / (15 * (h**3) * (c**2))
