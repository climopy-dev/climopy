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

# Regular expressions for translation from CF-compliant exponential
# and constant indictions to pint compatible strings.
REGEX_DIVIDE = re.compile(r'(/(?![^\(]*\))|\Z)')
REGEX_EXPONENTS = re.compile(r'([a-zA-Z]+)([-+]?[0-9]+)')
REGEX_CONSTANTS = re.compile(r'\b([-+]?[0-9]+[a-zA-Z]+)')


#: The default `pint.UnitRegistry` used throughout climopy. Includes flexible
#: definitions and aliases for time, pressure, temperature, vorticity, coordinates,
#: and units with constants, with support for nice short-form ``'~'`` formatting.
#: The unit definitions can be summarized as follows:
#:
#: .. code-block:: txt
#:
#:     none = 1
#:     percent = 0.01 = %
#:     permille = 0.001 = %%
#:     _10km = 10 * km = 10km
#:     _100km = 100 * km = 100km
#:     _1000km = 1000 * km = 1000km
#:     _10hPa = 10 * hPa = 10hPa
#:     _100hPa = 100 * hPa = 100hPa
#:     _1000hPa = 1000 * hPa = 1000hPa
#:     _10mb = 10 * mb = 10mb
#:     _100mb = 100 * mb = 100mb
#:     _1000mb = 1000 * mb = 1000mb
#:     minute = 60 * second = min
#:     hour = 60 * minute = hr = h
#:     day = 24 * hour = d
#:     month = year / 12 = mon
#:     year = 365.25 * day = yr = annum = a
#:     julian_year = 365.25 * day = julian_year = julian
#:     gregorian_year = 365.2425 * day = gregorian_year = gregorian = proleptic_...
#:     _366_day_year = 365 * day = _366_day = all_leap_year = allleap_year
#:     _365_day_year = 365 * day = _365_day = no_leap_year = noleap_year
#:     _360_day_year = 360 * day = _360_day
#:     bar = 10^5 Pa = b = baro
#:     level = 1 = layer = sigma_level = sigma_layer = model_level = model_layer
#:     inch_mercury = 3386.389 Pa = inHg = inchHg = inchesHg = in_Hg = inch_Hg = ...
#:     potential_vorticity_unit = 10^-6 K m^2 s^-1 kg^-1 = PVU
#:     vorticity_unit = 10^-5 s^-1 = 10^-5 s^-1 = VU
#:     degree = π / 180 * radian = ° = deg = arcdeg = arcdegree = angular_degree
#:     arcminute = degree / 60 = ′ = arcmin = arc_minute = angular_minute
#:     arcsecond = arcminute / 60 = ″ = arcsec = arc_second = angular_second
#:     degree_North = degree = °N = degree_north = degrees_North = degrees_north = ...
#:     degree_East = degree = °E = degree_east = degrees_East = degrees_east = ...
#:     @alias meter = metre = geopotential_meter = geopotential_metre = gpm
#:     @alias kelvin = Kelvin = K = k = degree_kelvin = degree_Kelvin = ...
#:     @alias degree_Celsius = degree_celsius = degrees_Celsius = degrees_celsius = ...
#:     @alias degree_Fahrenheit = degree_fahrenheit = degrees_Fahrenheit = ...
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
    # NOTE: If logging messages are enabled e.g. due to another module calling
    # logging.basicConfig() then the below will emit warnings. Can temporarily
    # disable these messages using 'on_redefinition'. Note also the below definitions
    # that overwrite existing definitions are always a superset of existing aliases
    # -- we only omit @alias to change the default long or short string repr.
    # See: https://github.com/hgrecco/pint/issues/1304
    # NOTE: Pint-xarray asserts that force_ndarray_like=True is required for scalar
    # conversions to work but haven't run into issues so far, and encountered an
    # issue due to negative integer exponentiation after enabling this option.
    # See: https://pint-xarray.readthedocs.io/en/stable/creation.html#attaching-units
    # See: https://github.com/hgrecco/pint/issues/1203
    preprocessors=[
        lambda s: s.replace('%%', ' permille '),
        lambda s: s.replace('%', ' percent '),
    ],
    on_redefinition='ignore',
)

#: Alias for the default `pint.UnitRegistry` `ureg`. The name "units" is consistent
#: with the `metpy` convention, while "ureg" is consistent with the `pint` convention.
units = ureg

# Dimensionless unit definitions
# See: https://github.com/hgrecco/pint/issues/185)
ureg.define('none = 1')
ureg.define('percent = 0.01 = %')
ureg.define('permille = 0.001 = %%')

# Common unit constants. These are used in things like meridional
# gradients and vertical mass or energy budget terms.
ureg.define('_10km = 10 * km = 10km')
ureg.define('_100km = 100 * km = 100km')
ureg.define('_1000km = 1000 * km = 1000km')
ureg.define('_10hPa = 10 * hPa = 10hPa')
ureg.define('_100hPa = 100 * hPa = 100hPa')
ureg.define('_1000hPa = 1000 * hPa = 1000hPa')
ureg.define('_10mb = 10 * mb = 10mb')
ureg.define('_100mb = 100 * mb = 100mb')
ureg.define('_1000mb = 1000 * mb = 1000mb')

# Time definitions. Uses 'yr' and 'hr' as default short-form year and hour, adds 'mon'
# alias, ensures only 'day' and 'second' have single-character short-form.
ureg.define('minute = 60 * second = min')
ureg.define('hour = 60 * minute = hr = h')
ureg.define('day = 24 * hour = d')
ureg.define('month = year / 12 = mon')
ureg.define('year = 365.25 * day = yr = annum = a')

# Calendar definitions. Puts 'julian_year' on its own instead of as part of 'year'
# and adds other CF compliant calendars. See: https://en.wikipedia.org/wiki/Year
ureg.define('julian_year = 365.25 * day = julian_year = julian')
ureg.define('gregorian_year = 365.2425 * day = gregorian_year = gregorian = proleptic_gregorian = proleptic_gregorian_year')  # noqa: E501
ureg.define('_366_day_year = 365 * day = _366_day = all_leap_year = allleap_year')
ureg.define('_365_day_year = 365 * day = _365_day = no_leap_year = noleap_year')
ureg.define('_360_day_year = 360 * day = _360_day')

# Vorticity definitions. Includes official 'PVU' unit and analogous (but fake)
# 'VU' unit characteristic of magnitudes observed on earth.
ureg.define('vorticity_unit = 10^-5 s^-1 = VU')
ureg.define('potential_vorticity_unit = 10^-6 K m^2 s^-1 kg^-1 = PVU')

# Pressure definitions (replace 'barn' with 'bar' as default 'b' unit) and CF
# definitions for vertical coordinates with dummy 'units' attributes
ureg.define('bar = 10^5 Pa = b = baro')
ureg.define('level = 1 = layer = sigma_level = sigma_layer = model_level = model_layer')
ureg.define('inch_mercury = 3386.389 Pa = inHg = inchHg = inchesHg = in_Hg = inch_Hg = inches_Hg = inches_mercury')  # noqa: E501

# Degree definitions with unicode short repr. These are redefinitions
# of existing units with identical supported aliases.
ureg.define('degree = π / 180 * radian = ° = deg = arcdeg = arcdegree = angular_degree')
ureg.define('arcminute = degree / 60 = ′ = arcmin = arc_minute = angular_minute')
ureg.define('arcsecond = arcminute / 60 = ″ = arcsec = arc_second = angular_second')

# Coordinate degree definitions. These are also used in the subclassed
# cf xarray accessor to detect longitude and latitude coordinates.
_longitude_units = (
    'degree_East = degree = °E = degree_east = degrees_East = degrees_east = '
    'degree_E = degrees_E = deg_East = deg_east = deg_E = '
    'degE = degreeE = degreesE = degEast = degreeEast = degreesEast'
)
_latitude_units = (
    'degree_North = degree = °N = degree_north = degrees_North = degrees_north = '
    'degree_N = degrees_N = deg_North = deg_north = deg_N = '
    'degN = degreeN = degreesN = degNorth = degreeNorth = degreesNorth'
)
ureg.define(_longitude_units)
ureg.define(_latitude_units)

# Temperature and height aliases. These include all varitions and
# combinations of abbreviation and capitalization.
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

# Set up for use with matplotlib
ureg.setup_matplotlib()

# Re-enable redefinition warnings
ureg._on_redefinition = 'warn'


def _to_pint_string(unit):
    """
    Translate CF and climopy constructs in preparation for parsing by
    `pint.decode_units`. See `decode_units` for details.
    """
    unit = REGEX_EXPONENTS.sub(r'\1^\2', unit or '')
    unit = REGEX_CONSTANTS.sub(r'_\1', unit)
    if ' since ' in unit:  # hours since, days since, etc.
        unit, *_ = unit.split()
    start = 0
    units = []
    for m in REGEX_DIVIDE.finditer(unit):
        end = m.start()
        units.append(unit[start:end].strip())
        start = m.end()
    unit, *denominator = units
    if denominator:
        unit = unit.strip() + ' / (' + ' '.join(d.strip() for d in denominator) + ')'
    return unit


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

    Parameters
    ----------
    unit : str or pint.Unit
        The pint units. If `pint.Unit` they are sipmly returned.

    See also
    --------
    encode_units
    format_units
    """
    if isinstance(unit, str):
        unit = _to_pint_string(unit)
    return ureg.parse_units(unit)


def encode_units(unit, /, long_form=None):
    """
    Convert `pint.Unit` into an unambiguous unit string. This is used with
    `~.accessor.ClimoDataArrayAccessor.dequantify` to assign the units attribute.

    Parameters
    ----------
    unit : pint.Unit or str
        The pint units. If string they are converted to `pint.Unit` then re-encoded.
    long_form : sequence, default: ('day', 'month', 'year')
        List of units that should be formatted in long form instead of symbols.

    See also
    --------
    decode_units
    format_units
    """
    long_form = long_form or ('day', 'month', 'year')
    if isinstance(unit, str):
        unit = decode_units(unit)
    units = {
        unit + ('s' if exp > 0 else '')
        if unit in long_form else ureg._get_symbol(unit): exp  # noqa: E501
        for unit, exp in unit._units.items()
    }
    string = ' '.join(
        formatter([(unit, exp)], as_ratio=False, power_fmt='{}^{}')
        for unit, exp in units.items()
    )
    return string


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

    Parameters
    ----------
    unit : str or pint.Unit
        The pint units. If string they are converted to `pint.Unit` first.
    long_form : sequence, default: ('day', 'month', 'year')
        List of units that should be formatted in long form instead of symbols.

    See also
    --------
    encode_units
    decode_units
    """
    # Parsing helper functions
    # WARNING: Critical to not split slashed inside of unit parentheses.
    # See: https://stackoverflow.com/a/3519601/4970632
    def _parse_unit(unit):
        start = 0
        parts = []
        for m in REGEX_DIVIDE.finditer(unit):
            end = m.start()
            part = unit[start:end]
            if '/' in part:
                part = _parse_unit(part.strip().strip('()'))
            else:
                part = decode_units(part)
            parts.append(part)
            start = m.end()
        return parts

    # Processing helper function
    # WARNING: 'sort' option requires pint 0.15 and 'L' results in failure to look up
    # and format babel unit since key is surrounded by \\mathrm{} by that point.
    # Instead we format the symbol and long form unit manually as shown below.
    def _process_unit(unit):
        exps = []
        longs = long_form or ('day', 'month', 'year')
        for key, exp in unit._units.items():
            if key not in longs:
                key = ureg._get_symbol(key)
            elif exp > 0:
                key += 's'
            key = rf'\mathrm{{{key}}}'
            exps.append((key, exp))
        return formatter(
            exps,
            sort=False,
            as_ratio=False,
            power_fmt='{}^{{{}}}',
            product_fmt=r' \, ',
        )
    def _process_units(*parts, outside=True):  # noqa: E306
        strings = []
        for part in parts:
            if not isinstance(part, Unit):
                string = _process_units(*part, outside=False)
            else:
                string = _process_unit(part)
            strings.append(string)
        string = r' \, / \, '.join(filter(None, strings))
        if not string:
            pass
        elif outside:
            string = f'${string}$'
        elif '/' in string:
            string = rf'({string})'
        return string

    # Apply units using sorting and name standardization.
    # NOTE: This formats accessor units attributes or cfvariable standard_units by
    # default. Uses the string descriptor to apply fussy formatting.
    # Also move 'constant_ units' like 100hPa and 1000km to the end of the unit string.
    if isinstance(unit, Unit):
        unit = (unit,)
    elif isinstance(unit, str):
        unit = _parse_unit(unit)
    else:
        raise ValueError(f'Invalid units {unit!r}.')
    string = _process_units(*unit)
    string = string.replace('%', r'\%')
    return string
