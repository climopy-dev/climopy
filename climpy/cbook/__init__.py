"""
Helper functions used internally by climpy.
"""
import inspect
import pint

#: Dictionary of docstring snippets added with `add_snippets`.
snippets = {}

#: The `pint.UnitRegistry` used throughout climpy. Adds the potential vorticity
#: unit with ``potential_vorticity_unit = 10^6 K m^2 kg^-1 s^-1 = PVU``,
#: the "bar" unit with ``bar = 10^5 Pa = b``,
#: the "inch of mercury" unit with
#: ``inch_mercury = 3386.389 Pa = inch_Hg = in_mercury = in_Hg``
#: flexible aliases for the Kelvin, degree Celsius, degree Fahrenheit,
#: and the spherical coordinate "degree North" and "degree East" units.
ureg = pint.UnitRegistry(preprocessors=[
    lambda s: s.replace('%%', ' permille '),
    lambda s: s.replace('%', ' percent '),
])

# Percent definitions (see https://github.com/hgrecco/pint/issues/185)
ureg.define(pint.unit.UnitDefinition(
    'permille', '%%', (), pint.converters.ScaleConverter(0.001),
))
ureg.define(pint.unit.UnitDefinition(
    'percent', '%', (), pint.converters.ScaleConverter(0.01),
))

# Canonical unit definitions
ureg.define('bar = 10^5 Pa = b')
ureg.define('potential_vorticity_unit = 10^6 K m^2 kg^-1 s^-1 = PVU')
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
    '@alias kelvin = Kelvin = K = '
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


def add_snippets(func):
    """Decorator that dedents docstrings with `inspect.getdoc` and adds
    un-indented snippets from the global `snippets` dictionary. This function
    uses ``%(name)s`` substitution rather than `str.format` substitution so
    that the `snippets` keys can be invalid variable names."""
    func.__doc__ = inspect.getdoc(func)
    if func.__doc__:
        func.__doc__ %= snippets
    return func
