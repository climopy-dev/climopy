"""
Helper functions used internally by climpy.
"""
import inspect
import pint

#: The `pint.UnitRegistry` used throughout climpy.
ureg = pint.UnitRegistry()
ureg.define('potential_vorticity_units = 10^6 K m^2 kg^-1 s^-1 = PVU')
ureg.define('@alias kelvin = Kelvin = K = deg_k = deg_K = degk = degK = degree_kelvin = degree_Kelvin')  # noqa: E501
ureg.define('@alias degree_Celsius = deg_c = deg_C = degc = degC = degree_celsius')  # noqa: E501

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
