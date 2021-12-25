#!/usr/bin/env python3
"""
A `pint` unit registry for climate scientists and tools for parsing and
formatting CF-compliant unit strings.
"""
import re

from pint import Unit, UnitRegistry, formatter

__all__ = [
    'ureg',  # pint convention
    'units',  # metpy convention
    'decode_units',
    'encode_units',
    'format_units',
]


#: The default `pint.UnitRegistry` used throughout climopy. Includes flexible aliases
#: for temperature, pressure, vorticity, dimensionless quantities, and units with
#: constants, with support for nice short-form ``'~'`` formatting as follows:
#:
#: .. code-block:: txt
#:
#:     none = 1
#:     level = 1
#:     layer = 1
#:     sigma_level = 1
#:     _10km = 10 * km = 10km
#:     _100km = 100 * km = 100km
#:     _1000km = 1000 * km = 1000km
#:     _100hPa = 100 * hPa = 100hPa
#:     bar = 10^5 Pa = b
#:     inch_mercury = 3386.389 Pa = inHg = inchHg = inchesHg = in_Hg = inch_Hg = ...
#:     potential_vorticity_unit = 10^-6 K m^2 s^-1 kg^-1 = PVU
#:     vorticity_unit = 10^-5 s^-1 = 10^-5 s^-1 = VU
#:     degree = π / 180 * radian = ° = deg = arcdeg = arcdegree = angular_degree
#:     arcminute = degree / 60 = ′ = arcmin = arc_minute = angular_minute
#:     arcsecond = arcminute / 60 = ″ = arcsec = arc_second = angular_second
#:     degree_North = degree = °N = degree_north = degrees_North = degrees_north = ...
#:     degree_East = degree = °E = degree_east = degrees_East = degrees_east = ...
#:     @alias kelvin = Kelvin = K = k = degree_kelvin = degree_Kelvin = ...
#:     @alias degree_Celsius = degree_celsius = degrees_Celsius = degrees_celsius = ...
#:     @alias degree_Fahrenheit = degree_fahrenheit = degrees_Fahrenheit = ...
#:     @alias meter = metre = geopotential_meter = geopotential_metre = gpm
#:
#: This also registers a `pint.Context` manager named ``'climo'`` for converting
#: static energy components, their rates of change (:math:`s^{-1}`), and their fluxes
#: (:math:`m\,s^{-1}`) between temperature and sensible heat (:math:`x \cdot c_p`),
#: between absolute humidity and latent heat (:math:`x \cdot L`), and between
#: geopotential height and geopotential (:math:`x \cdot g`). It also supports
#: transforming static energy terms between those normalized with respect to unit
#: vertical pressure distance and terms normalized per unit mass per unit area
#: (:math:`x \cdot g`).
ureg = UnitRegistry(
    preprocessors=[
        lambda s: s.replace('%%', ' permille '),
        lambda s: s.replace('%', ' percent '),
    ],
    # NOTE: Pint-xarray mysteriously declares that force_ndarary_like=True is
    # required for things to "work properly" but can't find other information.
    # Everything seems to work just fine without it so far...
    # https://pint-xarray.readthedocs.io/en/stable/creation.html#attaching-units
    # WARNING: Encountered this issue after enabling this option:
    # https://github.com/hgrecco/pint/issues/1203
    # force_ndarray_like=True,
)

#: Alias for the default `pint.UnitRegistry` `ureg`. The name "units" is consistent
#: with the `metpy` convention, while "ureg" is consistent with the `pint` convention.
units = ureg

# Dimensionless definitions (see https://github.com/hgrecco/pint/issues/185)
ureg.define('permille = 0.001 = %%')
ureg.define('percent = 0.01 = %')
ureg.define('none = 1')

# CF definitions for vertical coordinates with dummy 'units' attributes
ureg.define('level = 1')
ureg.define('layer = 1')
ureg.define('sigma_level = 1')

# Common unit constants
ureg.define('_10km = 10 * km = 10km')
ureg.define('_100km = 100 * km = 100km')
ureg.define('_1000km = 1000 * km = 1000km')
ureg.define('_100hPa = 100 * hPa = 100hPa')

# Pressure definitions (replace 'barn' with 'bar' as default 'b' unit)
ureg.define('bar = 10^5 Pa = b')
ureg.define('inch_mercury = 3386.389 Pa = inHg = inchHg = inchesHg = in_Hg = inch_Hg = inches_Hg = inches_mercury')  # noqa: E501

# Vorticity definitions
ureg.define('potential_vorticity_unit = 10^-6 K m^2 s^-1 kg^-1 = PVU')
ureg.define('vorticity_unit = 10^-5 s^-1 = 10^-5 s^-1 = VU')

# Degree definitions with unicode short repr
ureg.define('degree = π / 180 * radian = ° = deg = arcdeg = arcdegree = angular_degree')
ureg.define('arcminute = degree / 60 = ′ = arcmin = arc_minute = angular_minute')
ureg.define('arcsecond = arcminute / 60 = ″ = arcsec = arc_second = angular_second')

# Coordinate degree definitions
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

# Geopotential meter aliases
ureg.define('@alias meter = metre = geopotential_meter = geopotential_metre = gpm')

# Set up for use with matplotlib
ureg.setup_matplotlib()


def _standardize_string(unit):
    """
    Convert the input string into a unit string recognized by `pint`. Translates
    original CF constructs and additional climopy constructs in preparation for
    parsing by `pint.parse_units`. See `decode_units` for details.
    """
    unit = re.sub(r'([a-zA-Z]+)([-+]?[0-9]+)', r'\1^\2', unit or '')  # exponents
    unit = re.sub(r'\b([-+]?[0-9]+[a-zA-Z]+)', r'_\1', unit)  # constants
    if ' since ' in unit:  # hours since, days since, etc.
        unit, *_ = unit.split()
    unit, *denominator = unit.strip().split('/')
    if denominator:
        unit = unit.strip() + ' / (' + ' '.join(d.strip() for d in denominator) + ')'
    return unit


def encode_units(unit, /):
    """
    Convert `pint.Unit` into an unambiguous unit string. This is used with
    `~.accessor.ClimoDataArrayAccessor.dequantify` to record units in the
    `xarray.DataArray` attributes for future use.
    """
    if isinstance(unit, str):
        unit = decode_units(unit)
    return ' '.join(formatter([(u, e)], as_ratio=False) for u, e in unit._units.items())


def decode_units(unit, /):
    """
    Convert unit string into a `pint.Unit` with support for CF compliant constructs.
    This is used with `~.accessor.ClimoDataArrayAccessor.quantify` and
    `~.accessor.ClimoDataArrayAccessor.to_units`. Includes the following features:

    * Interpret CF standard where exponents are expressed as numbers adjacent to
      letters without any exponentiation marker, e.g. ``m2`` for ``m^2``.
    * Interpret CF standard time units with offsets declared in the string, e.g.
      ``days since 0001-01-01`` is decoded as ``days``.
    * Interpret climopy-defined units with constants without the leading dummy
      underscore (e.g. ``100km`` instead of ``_100km``; see `ureg` for details).
    * Interpret everything to the right of the first slash as part of the denominator
      (e.g. ``W / m2 Pa`` instead of ``W / m^2 / Pa``; additional slashes are optional).
    """
    if isinstance(unit, str):
        unit = _standardize_string(unit)
    return ureg.parse_units(unit)


def format_units(unit, /, *, long_form=None):
    r"""
    Fussily format the unit string or `pint.Unit` object into TeX-compatible form
    suitable for (e.g.) matplotlib figure text. Includes the following features:

    * Removes alphabetical sorting of component units to retain logical grouping
      from when the unit was first instantiated.
    * Raises component units to negative exponents instead of using fractions and
      increases separation between component units using the 3-mu ``\,`` separator.
    * Uses long form for units in `long_form` (default is ``('day', 'month', 'year')``,
      short form otherwise. Long form units in the numerator are written in plural.
    * Parses units on either side of the first slash independently. For example,
      the sensitivity parameter ``K / (W m^-2)`` is formatted as ``K / W m^-2``.
    """
    # Format the accessor "active units" by default. Use the string descriptor
    # representation of the standard units are active, to apply fussy formatting.
    long_form = long_form or ('day', 'month', 'year')
    if isinstance(unit, str):
        units = list(map(decode_units, unit.split('/')))
    elif not isinstance(unit, Unit):
        raise ValueError(f'Invalid units {unit!r}.')
    elif unit == 'dimensionless':
        units = []
    else:
        units = [unit]

    # Apply units sorting and name standardization. Put 'constant units'
    # like 100hPa and 1000km at the end of the unit string.
    # WARNING: 'sort' option requires pint 0.15 and 'L' results in failure to look up
    # and format babel unit since key is surrounded by \\mathrm{} by that point.
    # Instead we format the symbol and long form unit manually as shown below.
    parts = []
    for i, unit in enumerate(units):
        part = []
        for key, exp in unit._units.items():
            if key not in long_form:
                key = ureg._get_symbol(key)
            elif exp > 0 and i == 0:
                key += 's'
            part.append((exp, key))
        parts.append(part)
    string = r' \, / \, '.join(
        formatter(
            part,
            sort=False,
            as_ratio=False,
            power_fmt='{}^{{{}}}',
            product_fmt=r' \, ',
        )
        for part in parts
    )
    if '\N{DEGREE SIGN}' in string:
        string = ''
    elif string:
        string = '$' + string + '$'
    return string
