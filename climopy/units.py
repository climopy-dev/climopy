#!/usr/bin/env python3
"""
Integration with `pint`.
"""
import pint

__all__ = ['ureg', 'units']  # 'ureg' is pint convention, 'units' is metpy convention


#: The `pint.UnitRegistry` used throughout climopy. Adds flexible aliases for
#: temperature, pressure, vorticity, and various dimensionless quantities with
#: support for nice short-form ``'~'`` formatting as follows:
#:
#: .. code-block:: txt
#:
#:     percent = 0.01 * count = % = per_cent
#:     permille = 0.001 * count = %% = per_mille
#:     bar = 10^5 Pa = b
#:     inch_mercury = 3386.389 Pa = inch_Hg = in_mercury = in_Hg = ...
#:     vorticity_unit = 10^-5 s^-1 = VU
#:     potential_vorticity_unit = 10^-6 K m^2 kg^-1 s^-1 = PVU
#:     degree =  π / 180 * radian = ° = deg = arcdeg = arcdegree = angular_degree
#:     arcminute = degree / 60 = ′ = arcmin = arc_minute = angular_minute
#:     arcsecond = arcminute / 60 = ″ = arcsec = arc_second = angular_second
#:     degree_North = degree = °N = degree_north = degN = deg_N = ...
#:     degree_East = degree = °E = degree_east = degE = deg_E = ...
#:
#: This also registers a `pint.Context` manager named ``'climo'`` for converting
#: static energy components, their rates of change (:math:`s^{-1}`), their fluxes
#: (:math:`m\,s^{-1}`), and their *absolute* fluxes integrated over latitude bands
#: (:math:`m^2\,s^{-1}`) between temperature and sensible heat (:math:`x \cdot c_p`),
#: between absolute humidity and latent heat (:math:`x \cdot L`), and between
#: geopotential height and geopotential (:math:`x \cdot g`).
ureg = units = pint.UnitRegistry(
    preprocessors=[
        lambda s: s.replace('%%', ' permille '),
        lambda s: s.replace('%', ' percent '),
    ],
    force_ndarray_like=True,  # good practice and make compatible with pint-xarray
)

# Dimensionless definitions (see https://github.com/hgrecco/pint/issues/185)
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

# Ohter dimensionless definitions
ureg.define('none = 1')  # cannot add alias for 'dimensionless', this is best way
ureg.define('level = 1')  # specialized CF units for 'model level'
ureg.define('layer = 1')  # specialized CF units for 'vertical layer'
ureg.define('sigma_level = 1')  # specialized CF units for 'sigma coordinate'

# Pressure definitions (replace 'barn' with 'bar' as default 'b' unit)
ureg.define(  # automatically adds milli, hecta, etc.
    'bar = 10^5 Pa = b'
)
ureg.define(
    'inch_mercury = 3386.389 Pa = inHg = inchHg = inchesHg = '
    'in_Hg = inch_Hg = inches_Hg = inches_mercury'
)

# Vorticity definitions
ureg.define(
    'potential_vorticity_unit = 10^-6 K m^2 s^-1 kg^-1 = PVU'
)
ureg.define(
    'vorticity_unit = 10^-5 s^-1 = 10^-5 s^-1 = VU'
)

# Flexible degree definitions with unicode short repr
ureg.define(
    'degree = π / 180 * radian = ° = deg = arcdeg = arcdegree = angular_degree'
)
ureg.define(
    'arcminute = degree / 60 = ′ = arcmin = arc_minute = angular_minute'
)
ureg.define(
    'arcsecond = arcminute / 60 = ″ = arcsec = arc_second = angular_second'
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

# Temperature aliases
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

# Length aliases relatd to geopotential height
ureg.define(
    '@alias meter = metre = geopotential_meter = geopotential_metre = gpm'
)

# Common unit constants
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

# Set up for use with matplotlib
ureg.setup_matplotlib()
