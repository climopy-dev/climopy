"""
Helper functions used internally by climpy.
"""
import inspect
import pint

#: The `pint.UnitRegistry` used throughout climpy. Adds a definition for
#: potential vorticity units and more flexible aliases for the Kelvin, degree
#: Celsius, degree Fahrenheit, and the common spherical coordinate variables
#: "degree North" and "degree East".
ureg = pint.UnitRegistry()
ureg.define('potential_vorticity_unit = 10^6 K m^2 kg^-1 s^-1 = PVU')
ureg.define(
    '@alias kelvin = '
    'Kelvin = K = degree_kelvin = degree_Kelvin = '
    'deg_k = deg_K = degk = degK'
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
ureg.define(
    '@alias degree = degree_North = degree_north = '
    'degrees_North = degrees_north = '
    'deg_North = deg_north = '
    'degree_N = degree_n = degrees_N = degrees_n = '
    'deg_N = deg_n = degN = degn'
)
ureg.define(
    '@alias degree = degree_East = degree_east = '
    'degrees_East = degrees_east = '
    'deg_East = deg_east = '
    'degree_E = degree_e = degrees_E = degrees_e = '
    'deg_E = deg_e = degE = dege'
)

#: Dictionary of docstring snippets added with `add_snippets`.
snippets = {}


def add_snippets(func):
    """Decorator that dedents docstrings with `inspect.getdoc` and adds
    un-indented snippets from the global `snippets` dictionary. This function
    uses ``%(name)s`` substitution rather than `str.format` substitution so
    that the `snippets` keys can be invalid variable names."""
    func.__doc__ = inspect.getdoc(func)
    if func.__doc__:
        func.__doc__ %= snippets
    return func
