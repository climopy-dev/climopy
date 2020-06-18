#!/usr/bin/env python3
"""
Integration with `pint`.
"""
import pint

__all__ = ['ureg']

#: The `pint.UnitRegistry` used throughout climopy. Adds flexible aliases for
#: the meter, Kelvin, degree Celsius, and degree Fahrenheit.
#: Also adds definitions for the bar, percent, permille, potential vorticity
#: unit, inches mercury, and the "degree North" and "degree East" coordinate
#: variables. The units are defined as follows:
#:
#: .. code-block:: txt
#:
#:    bar = 10^5 Pa = b
#:    percent = 0.01 * count = % = per_cent
#:    permille = 0.001 * count = %% = per_mille
#:    potential_vorticity_unit = 10^6 K m^2 kg^-1 s^-1 = PVU
#:    inch_mercury = 3386.389 Pa = inch_Hg = in_mercury = in_Hg = ...
#:    degree_North = degree = °N = degree_north = degN = deg_N = ...
#:    degree_East = degree = °E = degree_east = degE = deg_E = ...
#:
ureg = pint.UnitRegistry(
    preprocessors=[
        lambda s: s.replace('%%', ' permille '),
        lambda s: s.replace('%', ' percent '),
    ]
)

# Percent definitions (see https://github.com/hgrecco/pint/issues/185)
ureg.define(
    pint.unit.UnitDefinition(
        'permille', '%%', (), pint.converters.ScaleConverter(0.001),
    )
)
ureg.define(
    pint.unit.UnitDefinition(
        'percent', '%', (), pint.converters.ScaleConverter(0.01),
    )
)

# Canonical unit definitions
ureg.define(  # cannot add alias for 'dimensionless', this is best way
    'none = 1'
)
ureg.define(  # automatically adds milli, hecta, etc.
    'bar = 10^5 Pa = b'
)
ureg.define(
    'potential_vorticity_unit = 10^-6 K m^2 s^-1 kg^-1 = PVU'
)
ureg.define(
    'vorticity_unit = 10^-5 s^-1 = 10^-5 s^-1 = VU'
)
ureg.define(
    'inch_mercury = 3386.389 Pa = inHg = inchHg = inchesHg = '
    'in_Hg = inch_Hg = inches_Hg = inches_mercury'
)
ureg.define(
    'degree_North = degree = °N = degree_north = degrees_North = degrees_north = '
    'degree_N = degrees_N = deg_North = deg_north = deg_N = '
    'degN = degreeN = degreesN = degNorth = degreeNorth = degreesNorth'
)
ureg.define(
    'degree_East = degree = °E = degree_east = degrees_East = degrees_east = '
    'degree_E = degrees_E = deg_East = deg_east = deg_E = '
    'degE = degreeE = degreesE = degEast = degreeEast = degreesEast'
)

# Additional aliases
ureg.define(
    '@alias meter = metre = geopotential_meter = geopotential_metre = gpm'
)
ureg.define(
    '@alias kelvin = Kelvin = K = k = '
    'degree_kelvin = degree_Kelvin = degrees_kelvin = degrees_Kelvin = '
    'deg_K = deg_k = degK = degk'
)
ureg.define(
    '@alias degree_Celsius = degree_celsius = '
    'degrees_Celsius = degrees_celsius = '
    'degree_C = degree_c = degrees_C = degrees_c = '
    'deg_Celsius = deg_celsius = '
    'deg_C = deg_c = degC = degc'
)
ureg.define(
    '@alias degree_Fahrenheit = degree_fahrenheit = '
    'degrees_Fahrenheit = degrees_fahrenheit = '
    'degree_F = degree_f = degrees_F = degrees_f = '
    'deg_Fahrenheit = deg_fahrenheit = '
    'deg_F = deg_f = degF = degf'
)

# Common units with constants
ureg.define(
    '_10km = 10 * km = 10km'
)
ureg.define(
    '_100km = 100 * km = 100km'
)
ureg.define(
    '_1000km = 1000 * km = 1000km'
)
ureg.define(
    '_100hPa = 100 * hPa = 100hPa'
)
