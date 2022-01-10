#!/usr/bin/env python3
"""
Tools for working with pint quantities.
"""
# NOTE: isinstance(..., ureg.Unit) or isinstance(..., ureg.Quantity) returns False
# for instances derived from other registries. So always test against pint namespace
# class definitions and defer to incompatible registry errors down the line.
import functools
import re

import numpy as np
import pint
import xarray as xr
from pint.util import to_units_container

from ..unit import _to_pint_string, ureg
from . import docstring

__all__ = ['while_quantified', 'while_dequantified']

# Regex to find terms surrounded by curly braces that can be filled with str.format()
REGEX_FORMAT = re.compile(r'\{[a-zA-Z_]\w*\}')  # valid identifiers

# Docstring snippets
_quant_docstring = """
A decorator that executes functions with %(descrip)s data values and enforces the
specified input and output units or dimensionalities. Comapre to `pint.wraps` and
`pint.check`. Pint quantities passed to the function will result in quantities returned
by the function. Non-pint quantiites passed to the function are assumed to be in the
correct units and will result in non-quantities returned by the function.

Parameters
----------
units_in : unit-spec or str or sequence
    The units for the positional input arguments. Can be a `pint.Unit`, a unit
    string specification like ``'cm'``, a dimensionality string specification
    like ``'[length]'``, or a relational variable specification like ``'=x'``.
    if the argument is not associated with any particular unit or dimensionality
    (for example ``while_%(descrip)s(('=x', '=y'), '=y / x')`` might be used
    for a function whose output units are the units of the second argument
    divided by the units of the first argument). Keyword arguments can be included
    in the variable specification using curly brace `str.format` notation after
    providing the default value via keyword argument to the decorator, for example
    ``while_%(descrip)s(('=x', '=y'), '=y / x^{{order}}', order=1)`` might be
    used for a function that takes the nth derivative using the keyword `order`.
    Vertical bars can be used to allow multiple incompatible units, for
    example ``while_%(descrip)s('J | K', 'J / s | K / s')`` converts energy
    or temperature input values into corresponding rate of change terms. This
    can be useful for designing functions that execute similar physical
    operations with different (but related) physical quantities.
units_out : unit-spec or str or sequence
    As with `units_in`, but for the return values.
convert_units : bool, default: True
    Whether to convert input argument and return value units to the specified
    units or merely assert compatibility with the specified units. This can be
    used as an alternative to dimensionality specifications like ``'[length]'.
require_quantity : bool, default: False
    Whether to forbid input arguments that are not pint quantities and have no units
    attribures. If ``False`` then these arguments are assumed to have correct units.
require_metadata : bool, default: False
    Whether to forbid input arguments that are not `xarray.DataArray`\\ s.
    Useful for derivations that require operations along CF coordinates.
**fmt_defaults
    Default values for the terms surrounded by curly braces in relational
    or string unit specifications.

Returns
-------
callable
    The function decorated by `~internals.quant.while_%(descrip)s`.

Example
-------
Here is a simple example for an nth derivative wrapper.

>>> from climopy import ureg, while_%(descrip)s
>>> @while_%(descrip)s(('=x', '=y'), '=y / x^{{order}}', order=1)
... def deriv(x, y, order=1):
...     return y / x ** order
>>> deriv(1 * ureg.m, 1 * ureg.s, order=2)
<Quantity(1.0, 'second / meter ** 2')>
"""
docstring.snippets['quant.quantified'] = _quant_docstring


def _as_units_container(arg, **fmt_kwargs):
    """
    Convert a unit type to a UnitsContainer after checking if it is a reference.
    """
    # NOTE: This parses units when applying quantify decorators, standardizing return
    # value units, and standardizing dependent input argument units. Also validates
    # input units and dimensions when decorator is defined rather than when used.
    category = 0  # unit spec (0), dimensionality string (1), or reference string (2)
    if isinstance(arg, str):
        arg = arg.format(**fmt_kwargs)  # permit extra keyword arguments
        if '[' in arg and ']' in arg:
            category = 1
        elif '=' in arg:
            category = 2
        else:
            arg = _to_pint_string(arg)  # support conventions, possible error later
    elif arg is None or isinstance(arg, pint.Unit):
        pass
    else:  # should be impossible since _group_args checks type
        raise ValueError(f'Unrecognized pint unit argument {arg}.')
    if category == 0:
        container = to_units_container(arg, ureg)  # validates string units
    elif category == 1:
        container = ureg.get_dimensionality(arg)  # validates dimensions
    elif category == 2:
        container = to_units_container(arg.split('=', 1)[1])  # skips validation
    else:
        raise RuntimeError(f'Invalid {category=}.')
    return container, category


def _enforce_dimensionality(arg, dimensionality):
    """
    Ensure quantity conforms to specified dimensionality.
    """
    # WARNING: Error has to be raised manually because is_compatible_with does not
    # accept dimensionality containers and no other public methods are available.
    if not isinstance(arg, (pint.Unit, pint.Quantity)):
        raise RuntimeError(f'Invalid input argument {arg!r}.')
    dimensionality = ureg.get_dimensionality(dimensionality)
    arg_dimensionality = ureg.get_dimensionality(arg)
    if arg_dimensionality != dimensionality:
        raise pint.DimensionalityError(
            arg, 'a quantity of', arg_dimensionality, dimensionality
        )


def _get_pint_units(arg):
    """
    Get the pint units associated with the object argument.
    """
    # NOTE: This parses units when comparing input argument units against the
    # declared units in order to select the correct standardization group.
    units = None
    if isinstance(arg, str):
        arg = ureg.parse_expression(arg)  # multiplies raw unit strings by '1'
    elif isinstance(arg, pint.Quantity):
        units = arg.units
    elif isinstance(arg, xr.DataArray):
        if arg.climo._has_units:
            units = arg.climo.units
    return units


def _group_args(args_in, args_out):
    """
    Parse specifications for input arguments and return values. Used for
    `while_quantified`, `while_dequantified`, and `register_derivation`.
    """
    # Enforce iterable input argument and return value specs
    # NOTE: Type checking for decorators happens here first.
    from ..cfvariable import CFVariable  # depends on internals so import here
    if isinstance(args_in, str) or not np.iterable(args_in):
        args_in = () if args_in is None else (args_in,)
    if is_scalar_out := isinstance(args_out, str) or not np.iterable(args_out):
        args_out = () if args_out is None else (args_out,)

    # Split string specs into groups of options (separated by |). Ensure same
    # number of non-scalar options for input arguments and return values.
    args = []
    sizes = set()
    for i, arg in enumerate((*args_in, *args_out)):
        if isinstance(arg, str):
            arg = [a.strip() for a in arg.split('|')]
        elif arg is None or isinstance(arg, (pint.Unit, CFVariable)):
            arg = [arg]
        else:
            raise TypeError(f'Input must be str, dict, Unit, or UnitsContainer. Instead got {arg!r}.')  # noqa: E501
        if len(arg) > 1:
            sizes.add(len(arg))
        if len(sizes) > 1:
            raise TypeError('Non-scalar name sequences must be equal length.')
        if sizes and len(arg) == 1 and i >= len(args_in):
            raise TypeError('Non-scalar input name sequences require non-scalar output')
        args.append(arg)

    # Split input argument and return value specs into option groups. For example get
    # [['a', 'c'], ['b', 'c']] and [['x'], ['y']] from _group_args(('a|b', 'c'), 'x|y').
    args = [arg * max(sizes, default=1) if len(arg) == 1 else arg for arg in args]
    args_in = [arg[:-len(args_out)] for arg in zip(*args)]
    args_out = [arg[-len(args_out):] for arg in zip(*args)]
    return args_in, args_out, is_scalar_out


def _standardize_independent(
    arg, quantify=False, require_quantity=False, require_metadata=False
):
    """
    Return a quantified version of the input argument using its own units. If it
    has no units then assign dimensionless units.
    """
    # Apply existing units
    if isinstance(arg, str):  # parse expressions e.g. '5cm'
        arg = ureg.parse_expression(arg)
    if isinstance(arg, xr.DataArray):
        is_quantity = arg.climo._is_quantity
        if arg.climo._has_units:
            arg = arg.climo.quantify()
        if not arg.climo._is_quantity:
            if not require_quantity:
                arg = arg.climo.quantify(units='dimensionless')
            else:
                raise TypeError('Pint quantity data or units attributes are required.')
        units = arg.data.units
    elif not require_metadata:
        is_quantity = isinstance(arg, pint.Quantity)
        if not is_quantity:
            if not require_quantity:
                arg = arg * ureg.dimensionless
            else:
                raise TypeError('Pint quantity data are required.')
        units = arg.units
    else:
        raise TypeError('Xarray DataArrays are required.')

    # Optionally dequantify result after converting
    if not quantify:
        if isinstance(arg, xr.DataArray):
            arg = arg.climo.dequantify()
        else:
            arg = arg.magnitude
    return arg, units, is_quantity


def _standardize_dependent(
    arg, unit=None, quantify=False, definitions=None,
    convert_units=True, require_quantity=False, require_metadata=False, **fmt_kwargs
):
    """
    Return a quantified version of the input argument possibly applying the
    declared units or inferring them from the independent variable units.
    """
    # Parse input units
    container, category = _as_units_container(unit, **fmt_kwargs)
    if container is None:
        return arg, False

    # Apply units definitions
    if category == 0:
        dest = pint.Unit(container)
    elif category == 1:
        dest = container  # keep the raw dimensionality container
    else:
        dest = ureg.dimensionless
        definitions = definitions or {}
        for name, exponent in container.items():
            if name in definitions:
                dest *= definitions[name] ** exponent
            else:
                raise RuntimeError(f'Missing unit definition for variable {name!r}.')

    # Enforce argument units
    # NOTE: Important to record whether we started with units
    if isinstance(arg, str):  # parse expressions e.g. '300 K'
        arg = ureg.parse_expression(arg)
    if isinstance(arg, xr.DataArray):
        is_quantity = arg.climo._is_quantity
        if arg.climo._has_units:
            arg = arg.climo.quantify()
        if arg.climo._is_quantity:
            if not convert_units or category == 1:
                _enforce_dimensionality(arg.data, dest)
            else:
                arg = arg.climo.to(dest)
        else:
            if not require_quantity and category != 1:
                arg = arg.climo.quantify(units=unit)
            else:
                raise TypeError('Pint quantity data or units attributes are required.')
    elif not require_metadata:
        if is_quantity := isinstance(arg, pint.Quantity):
            if not convert_units or category == 1:
                _enforce_dimensionality(arg, dest)
            else:
                arg = arg.to(dest)
        else:
            if not require_quantity and category != 1:
                arg = ureg.Quantity(arg, unit)
            else:
                raise TypeError('Pint quantity data are required.')
    else:
        raise TypeError('Xarray DataArrays are required.')

    # Optionally dequantify result after converting
    if not quantify:
        if isinstance(arg, xr.DataArray):
            arg = arg.climo.dequantify()
        else:
            arg = arg.magnitude
    return arg, is_quantity


def _while_converted(
    units_in=None,
    units_out=None,
    grouped=False,
    quantify=False,
    convert_units=True,
    require_quantity=False,
    require_metadata=False,
    **fmt_defaults  # noqa: E501
):
    """
    Driver function for `while_quantified` and `while_dequantified`. See above
    for the full documentation.
    """
    # Group and categorize the input argument units
    # NOTE: Resulting units_in, units_out will be singleton lists unless | was used.
    categories = []
    containers = []  # check dependent against independent variables
    if grouped:  # used by register_derivation
        pass
    else:
        units_in, units_out, is_scalar_out = _group_args(units_in, units_out)
    for units in units_in:
        independent = {}
        dependent = set()
        constant = set()
        for idx, unit in enumerate(units):
            container, category = _as_units_container(unit, **fmt_defaults)
            if container is None:
                pass
            elif category == 2:
                if len(container) == 1:
                    (key, value), = container.items()
                    if value == 1 and key not in independent:
                        independent[key] = idx
                    else:
                        dependent.add(idx)
                else:
                    dependent.add(idx)  # definition is found elsewhere
            else:
                constant.add(idx)
            containers.append(container)
        for idx in dependent:
            container, unit = containers[idx], unit[idx]
            if not container.keys() <= independent.keys():
                raise ValueError(f'Not all variables referenced in {unit} are defined.')
        categories.append((independent, dependent, constant))

    # Declare decorator
    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            # Test input arguments. Bypass extra arguments
            args = list(args)
            n_result = len(args)
            n_expect = len(units_in[0])
            if n_expect > n_result:
                raise ValueError(f'Expected {n_expect} positional args, got {n_result}.')  # noqa: E501

            # Select group for parsing
            # NOTE: Behavior is subtle. Iterate over possible inputs and approve each
            # member of the grouping if either (1) it has no units, (2) the declared
            # unit is a reference, (3) there were no declared units, or (4) the units
            # are compatible with the declared units. If this fails we use the final
            # grouping by default and an error will be raised down the line.
            for grp, (independents, dependents, constants) in enumerate(categories):
                units_input = [_get_pint_units(args[idx]) for idx in constants]
                units_expect = [units_in[grp][idx] for idx in constants]
                if all(
                    unit_input is None
                    or unit_expect is None
                    or unit_input.dimensionality == ureg.get_dimensionality(unit_expect)
                    for unit_input, unit_expect in zip(units_input, units_expect)
                ):
                    break

            # Quantify independent input arguments and record units
            args_new = args.copy()
            definitions = {}
            quantify_results = False
            for key, idx in independents.items():
                arg, units, is_quantity = _standardize_independent(
                    args[idx],
                    quantify=quantify,
                    require_quantity=require_quantity,
                    require_metadata=require_metadata,
                )
                args_new[idx] = arg
                definitions[key] = units
                quantify_results = is_quantity or quantify_results

            # Quantify remaining arguments using recorded units
            fmt_kwargs = {key: val for key, val in kwargs.items() if key in fmt_defaults}  # noqa: E501
            for key, val in fmt_defaults.items():
                fmt_kwargs.setdefault(key, value)
            for idx in (*dependents, *constants):
                arg, is_quantity = _standardize_dependent(
                    args[idx],
                    units_in[grp][idx],
                    definitions=definitions,
                    quantify=quantify,
                    convert_units=convert_units,
                    require_quantity=require_quantity,
                    require_metadata=require_metadata,
                    **fmt_kwargs
                )
                args_new[idx] = arg
                quantify_results = is_quantity or quantify_results

            # Call main function and standardize results. Bypass extra values
            results = func(*args_new, **kwargs)
            n_result = 1 if not isinstance(results, tuple) else len(results)
            n_expect = len(units_out[grp])
            if is_scalar_out and isinstance(results, tuple):
                raise ValueError('Got tuple of return values, expected one value.')
            if not is_scalar_out and n_expect > n_result:
                raise ValueError(f'Expected {n_expect} return values, got {n_result}.')
            results = [results] if is_scalar_out else list(results)
            results_new = results.copy()
            for idx in range(n_expect):
                res, _ = _standardize_dependent(
                    results[idx],
                    units_out[grp][idx],
                    definitions=definitions,
                    quantify=quantify_results,
                    convert_units=convert_units,
                    **fmt_kwargs
                )
                results_new[idx] = res

            # Return sanitized values
            if is_scalar_out:
                return results_new[0]
            else:
                return tuple(results_new)

        return _wrapper

    return _decorator


@docstring.inject_snippets(descrip='dequantified')
def while_quantified(*args, **kwargs):
    """
    %(quant.quantified)s
    """
    kwargs['quantify'] = True
    return _while_converted(*args, **kwargs)


@docstring.inject_snippets(descrip='dequantified')
def while_dequantified(*args, **kwargs):
    """
    %(quant.quantified)s
    """
    kwargs['quantify'] = False
    return _while_converted(*args, **kwargs)
