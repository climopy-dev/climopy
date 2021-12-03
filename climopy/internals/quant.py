#!/usr/bin/env python3
"""
Tools for working with pint quantities.
"""
import functools
import itertools
import re

import pint
import pint.util as putil

from ..unit import ureg
from . import docstring

__all__ = ['while_quantified', 'while_dequantified']

# Regex to find terms surrounded by curly braces that can be filled with str.format()
REGEX_FORMAT = re.compile(r'\{([^{}]+?)\}')  # '+?' is non-greedy, group inside brackets

# Docstring snippets
_quant_docstring = """
A decorator that executes functions with %(descrip)s data values and enforcing the
specified input and output units. Pint quantities passed to the function will
result in quantities returned by the function. Non-pint quantiites passed to
the function are assumed to be in the correct units and will result in
non-quantities returned by the function.

Parameters
----------
units_in : unit-like, string, or sequence thereof
    The units for the positional input arguments. Can be a `pint.Unit`, a unit
    string specification like ``'cm'``, or a relational string specification like
    ``'=x ** 2'``. You can include keyword arguments in the unit specification
    using e.g. ``while_%(descrip)s(('=x', '=y'), '=y / x ** {{order}}')``
    for a function that takes the nth derivative using the keyword `order`.
units_out : unit-like, string, or sequence thereof
    As with `units_in` but for the return values.
strict : bool, default: False
    Whether to forbid non-quantity input arguments. If ``False`` then these
    are assumed to be in the correct units.
**fmt_defaults
    Default values for the terms surrounded by curly braces in relational
    or string unit specifications.

Example
-------
Here is a simple example for an nth derivative wrapper.

>>> from climopy import ureg, while_dequantified
>>> @while_dequantified(('=x', '=y'), '=y / x ** {order}', order=1)
... def deriv(x, y, order=1):
...     return y / x ** order
>>> deriv(1 * ureg.m, 1 * ureg.s, order=2)
<Quantity(1.0, 'second / meter ** 2')>
"""
docstring.snippets['quant.quantified'] = _quant_docstring


def _pint_replace_units(container, definitions):
    """
    Convert the references in a UnitsContainer into valid pint Units, using
    positional data argument units for each reference.
    """
    unit = ureg.dimensionless
    for name, exponent in container.items():
        if name in definitions:
            unit *= definitions[name] ** exponent
        else:
            raise RuntimeError(f'Missing definition for container reference {name!r}.')
    return unit


def _pint_units_container(arg):
    """
    Convert a pint unit type to a UnitsContainer, checking if it is a reference
    (i.e. a string prefixed with an equal sign).
    """
    is_ref = isinstance(arg, str) and '=' in arg
    if is_ref:
        unit = putil.to_units_container(arg.split('=', 1)[1])
    else:
        unit = putil.to_units_container(arg, ureg)
    return unit, is_ref


def _pint_parse_args(units, *, quantify=False):
    """
    Parse pint wrapper unit specifications and return a function that standardizes the
    units of positional data arguments. The `quantity` parameter controls whether data
    arguments are quantified or dequantified after unit standardization.
    """
    # Helper variables
    containers = [_pint_units_container(unit) for unit in units]
    independents = {}  # indices of independent definitions
    dependents = set()  # indices of arguments that depend on definitions
    constants = set()  # indices of constant units

    # Check for references in args and remove None values
    for idx, (container, is_ref) in enumerate(containers):
        if container is None:
            continue
        elif is_ref:
            if len(container) == 1:
                (key, value), = container.items()
                if value == 1 and key not in independents:
                    independents[key] = idx
                    containers[idx] = (key, True)
                else:
                    dependents.add(idx)
            else:
                dependents.add(idx)  # definition found elsewhere
        else:
            constants.add(idx)

    # Check that all dependent variables are defined
    for idx in dependents:
        container, is_ref = containers[idx]
        if isinstance(container, dict) and not container.keys() <= independents.keys():
            raise ValueError(
                'Found a missing token while wrapping a function: Not '
                f'all variable referenced in {units[idx]} are defined!'
            )

    # Generate converter
    def _converter(args, strict):
        # Translate containers into units
        units = {}
        definitions = {}
        for key, idx in independents.items():
            arg  = args[idx]
            unit = arg.units if isinstance(arg, pint.Quantity) else ureg.dimensionless
            units[idx] = definitions[key] = unit
        for idx in dependents:
            units[idx] = _pint_replace_units(containers[idx][0], definitions)
        for idx in constants:
            units[idx] = pint.Unit(containers[idx][0])

        # Standardize data units
        args_new = []
        for idx, arg in enumerate(args):
            if isinstance(arg, str):  # parse expressions e.g. '5cm'
                arg = ureg.parse_expression(arg)
            if isinstance(arg, pint.Quantity):
                arg = arg.to(units[idx])
            elif not strict:
                arg = ureg.Quantity(arg, units[idx])
            else:
                raise ValueError('Pint quantities are required in strict mode.')
            if not quantify:
                arg = arg.magnitude
            args_new.append(arg)

        return args_new, definitions

    return _converter


def _while_converted(units_in, units_out, quantify=False, strict=False, **fmt_defaults):  # noqa: E501
    """
    Driver function for `while_quantified` and `while_dequantified`.
    See above for the full documentation.
    """
    # Handle singleton input or multiple input
    # NOTE: Pint cannot handle singleton-tuple of return value unit specifications.
    # So when passing to wrapper simply expand singleton tuples.
    is_container_in = isinstance(units_in, (tuple, list))
    is_container_out = isinstance(units_out, (tuple, list))
    if not is_container_in:
        units_in = (units_in,)
    if not is_container_out:
        units_out = (units_out,)

    # Ensure valid keyword arguments
    fmt_args = []
    for units in (units_in, units_out):
        for unit in units:
            if isinstance(unit, str):
                fmt_args.extend(REGEX_FORMAT.findall(unit))
    if set(fmt_args) != set(fmt_defaults):
        raise ValueError(
            f'Invalid or insufficient keyword args {tuple(fmt_defaults)} '
            f'when string unit specification includes terms {tuple(fmt_args)}.'
        )

    # Ensure valid input and return arguments
    for arg in units_in:
        if arg is not None and not isinstance(arg, (ureg.Unit, str)):
            raise TypeError(
                f'Wraps arguments must by of type str or Unit, not {type(arg)} ({arg}).'
            )
    for arg in units_out:
        if arg is not None and not isinstance(arg, (ureg.Unit, str)):
            raise TypeError(
                "Wraps 'units_out' argument must by of type str or Unit, "
                f'not {type(arg)} ({arg}).'
            )

    # Parse input
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Test input
            n_expect = len(units_in)
            n_result = len(args)
            if len(units_in) != len(args):
                raise ValueError(f'Expected {n_expect} positional args, got {n_result}.')  # noqa: E501

            # Fill parameters inside units
            units_in_fmt = []
            units_out_fmt = []
            for units, units_fmt in zip((units_in, units_out), (units_in_fmt, units_out_fmt)):  # noqa: E501
                for unit in units:
                    if isinstance(unit, str):
                        # Get format values from user input keyword args
                        fmt_keys = REGEX_FORMAT.findall(unit)
                        fmt_kwargs = {key: val for key, val in kwargs.items() if key in fmt_keys}  # noqa: E501
                        # Fill missing format values and format string
                        for key, value in fmt_defaults.items():
                            fmt_kwargs.setdefault(key, value)
                        unit = unit.format(**fmt_kwargs)
                    # Add new unit string
                    units_fmt.append(unit)

            # Standardize input
            converter = _pint_parse_args(units_in_fmt, quantify=quantify)
            args_new, definitions = converter(args, strict)

            # Call main function and check output
            result = func(*args_new, **kwargs)
            n_result = 1 if not isinstance(result, tuple) else len(result)
            n_expect = len(units_out)
            if not is_container_out and isinstance(result, tuple):
                raise ValueError('Got tuple of return values, expected one value.')
            if is_container_out and n_result != len(units_out):
                raise ValueError(f'Expected {n_expect} return values, got {n_result}.')

            # Quantify output, but *only* if input were quantities
            result_new = []
            result_quantify = any(isinstance(arg, ureg.Quantity) for arg in args)
            if not is_container_out:
                result = (result,)
            for res, unit in itertools.zip_longest(result, units_out_fmt):
                container, is_ref = _pint_units_container(unit)
                if is_ref:
                    unit = _pint_replace_units(container, definitions)
                else:
                    unit = pint.Unit(container)
                if result_quantify and not isinstance(res, pint.Quantity):
                    res = ureg.Quantity(res, unit)
                elif not result_quantify and isinstance(res, pint.Quantity):
                    res = res.magnitude
                result_new.append(res)

            # Return sanitized values
            if not is_container_out:
                return result_new[0]
            else:
                return tuple(result_new)

        return wrapper

    return decorator


@docstring.inject_snippets(descrip='dequantified')
def while_quantified(units_in, units_out, strict=False, **fmt_defaults):
    """
    %(quant.quantified)s
    """
    return _while_converted(
        units_in, units_out, quantify=True, strict=strict, **fmt_defaults
    )


@docstring.inject_snippets(descrip='dequantified')
def while_dequantified(units_in, units_out, strict=False, **fmt_defaults):
    """
    %(quant.quantified)s
    """
    return _while_converted(
        units_in, units_out, quantify=False, strict=strict, **fmt_defaults
    )
