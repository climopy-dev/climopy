#!/usr/bin/env python3
"""
Tools for working with pint quantities.
"""
# NOTE: isinstance(..., ureg.Unit) or isinstance(..., ureg.Quantity) returns False
# for instances derived from other registries. So always test against pint namespace
# class definitions and defer to incompatible registry errors down the line.
import functools
import itertools
import re

import pint
import pint.util as putil

from ..unit import ureg
from . import docstring

__all__ = ['while_quantified', 'while_dequantified']

# Regex to find terms surrounded by curly braces that can be filled with str.format()
REGEX_FORMAT = re.compile(r'\{[a-zA-Z_]\w*\}')  # valid identifiers

# Docstring snippets
_quant_docstring = """
A decorator that executes functions with %(descrip)s data values and enforces the
specified input and output units. Pint quantities passed to the function will
result in quantities returned by the function. Non-pint quantiites passed to
the function are assumed to be in the correct units and will result in
non-quantities returned by the function.

Parameters
----------
units_in : unit-like, string, or sequence thereof
    The units for the positional input arguments. Can be a `pint.Unit`, a unit
    string specification like ``'cm'``, or a relational string specification
    like ``'=x^2'``. You can include keyword arguments in the unit specification
    by embedding the keyword in a unit string using `str.format` notation, for
    example ``while_%(descrip)s(('=x', '=y'), '=y / x^{{order}}')`` might be
    used for a function that takes the nth derivative using the keyword `order`.
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

>>> from climopy import ureg, while_%(descrip)s
>>> @while_%(descrip)s(('=x', '=y'), '=y / x^{order}', order=1)
... def deriv(x, y, order=1):
...     return y / x ** order
>>> deriv(1 * ureg.m, 1 * ureg.s, order=2)
<Quantity(1.0, 'second / meter ** 2')>
"""
docstring.snippets['quant.quantified'] = _quant_docstring


def _replace_units(container, definitions):
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


def _units_container(arg, **fmt_kwargs):
    """
    Convert a pint unit type to a UnitsContainer, checking if it is a reference
    (i.e. a string prefixed with an equal sign).
    """
    is_ref = isinstance(arg, str) and '=' in arg
    if isinstance(arg, str):
        arg = arg.format(**fmt_kwargs)  # permits extra keyword arguments
    types = (str, dict, pint.Unit, pint.Quantity, putil.UnitsContainer)
    if arg is not None and not isinstance(arg, types):
        raise ValueError(f'Invalid unit argument {arg}. Must be any of {types}.')
    if is_ref:
        unit = putil.to_units_container(arg.split('=', 1)[1])
    else:
        unit = putil.to_units_container(arg, ureg)  # None returns None
    return unit, is_ref


def _while_converted(units_in, units_out, quantify=False, strict=False, **fmt_defaults):
    """
    Driver function for `while_quantified` and `while_dequantified`. See above
    for the full documentation.
    """
    # Ensure input arguments are valid units or references
    is_container_in = isinstance(units_in, (tuple, list))
    is_container_out = isinstance(units_out, (tuple, list))
    if not is_container_in:
        units_in = (units_in,)
    if not is_container_out:
        units_out = (units_out,)
    containers_in = [_units_container(unit, **fmt_defaults) for unit in units_in]
    containers_out = [_units_container(unit, **fmt_defaults) for unit in units_out]  # noqa: E501, F841

    # Detect references in args and remove None values
    independents = {}  # indices of independent definitions
    dependents = set()  # indices of arguments that depend on definitions
    constants = set()  # indices of constant units
    for idx, (container, is_ref) in enumerate(containers_in):
        if container is None:
            continue
        elif is_ref:
            if len(container) == 1:
                (key, value), = container.items()
                if value == 1 and key not in independents:
                    independents[key] = idx
                    containers_in[idx] = (key, True)
                else:
                    dependents.add(idx)
            else:
                dependents.add(idx)  # definition found elsewhere
        else:
            constants.add(idx)

    # Ensure that all dependent variables are defined
    for idx in dependents:
        container, is_ref = containers_in[idx]
        if isinstance(container, dict) and not container.keys() <= independents.keys():
            raise ValueError(f'Not all variables referenced in {units_in[idx]} are defined.')  # noqa: E501

    # Declare decorator
    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            # Test input
            n_result = len(args)
            n_expect = len(units_in)
            if n_result != n_expect:
                raise ValueError(f'Expected {n_expect} positional args, got {n_result}.')  # noqa: E501

            # Translate containers into units
            units = {}
            definitions = {}
            for key, idx in independents.items():
                arg = args[idx]
                unit = arg.units if isinstance(arg, pint.Quantity) else ureg.dimensionless  # noqa: E501
                units[idx] = definitions[key] = unit
            fmt_kwargs = {key: val for key, val in kwargs.items() if key in fmt_defaults}  # noqa: E501
            for key, val in fmt_defaults.items():
                fmt_kwargs.setdefault(key, value)
            for idx in dependents:
                container, _ = _units_container(units_in[idx], **fmt_kwargs)
                units[idx] = _replace_units(container, definitions)
            for idx in constants:
                container, _ = _units_container(units_in[idx], **fmt_kwargs)
                units[idx] = pint.Unit(container)

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

            # Call main function and check output
            result = func(*args_new, **kwargs)
            n_result = 1 if not isinstance(result, tuple) else len(result)
            n_expect = len(units_out)
            if not is_container_out and isinstance(result, tuple):
                raise ValueError('Got tuple of return values, expected one value.')
            if is_container_out and n_result != n_expect:
                raise ValueError(f'Expected {n_expect} return values, got {n_result}.')

            # Quantify output if input arguments were quantities
            result_new = []
            result_quantify = any(isinstance(arg, pint.Quantity) for arg in args)
            if not is_container_out:
                result = (result,)
            for res, unit in itertools.zip_longest(result, units_out):
                container, is_ref = _units_container(unit, **fmt_kwargs)
                if is_ref:
                    unit = _replace_units(container, definitions)
                else:
                    unit = pint.Unit(container)
                if isinstance(res, pint.Quantity):
                    res = res.to(unit)
                else:
                    res = ureg.Quantity(res, unit)
                if not result_quantify:
                    res = res.magnitude
                result_new.append(res)

            # Return sanitized values
            if not is_container_out:
                return result_new[0]
            else:
                return tuple(result_new)

        return _wrapper

    return _decorator


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
