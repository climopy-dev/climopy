#!/usr/bin/env python3
"""
Wrappers that permit duck-type array input to various functions.

Todo
----
Add wrappers for variance functions, spectral functions, and EOF function.

Warning
-------
It is clunky to require duplicate ``axis=axis`` but otherwise if you change it
in the function you may forget to change it on the wrapper. And cannot do xarray
processing inside main functions because it must happen outside of pint processing.
"""
import re
import functools
import itertools
import numpy as np
import xarray as xr
import pint.util as putil
from .. import ureg
from . import warnings

# Regex to find terms surrounded by curly braces that can be filled with str.format()
REGEX_FORMAT = re.compile(r'\{([^{}]+?)\}')  # '+?' is non-greedy, group inside brackets


def _get_step(h):
    """
    Determine scalar step h.
    """
    h = np.atleast_1d(h)
    if h.size == 1:
        return h[0]
    elif h.ndim != 1:
        raise ValueError(f'x coordinates must be 1D, not {h.ndim}D.')
    elif not np.allclose(np.diff(h), h[1] - h[0]):
        raise ValueError(f'x coordinate steps must be identical, but got {h}.')
    else:
        return h[1] - h[0]


def _apply_units(data):
    """
    Apply unit attribute units and get magnitudes.
    """
    if isinstance(data.data, ureg.Quantity):
        data = data.data
    else:
        try:
            data = data.data * ureg(data.attrs['units'])
        except Exception:  # many, many things could go wrong here
            data = data.data
    return data


def _remove_units(data):
    """
    Remove units before assigning as coordinate.
    """
    if isinstance(data, xr.DataArray) and isinstance(data.data, ureg.Quantity):
        data = data.copy()
        data.attrs.setdefault('units', format(data.data.units, '~'))
        data.data = data.data.magnitude
    return data


def _to_arraylike(x, y, axis_default, *, infer_axis=True, **kwargs):
    """
    Sanitize input *x* and *y* DataArrays, return the axis and dim
    along which we are calculating stuff,
    """
    # Interpret 'dim' argument when input is Dataarray
    axis = kwargs.pop('axis', None)
    if isinstance(y, xr.DataArray):
        dim = kwargs.pop('dim', None)
        if dim is not None:
            if axis is not None:
                warnings._warn_climpy('Ambiguous axis specification.')
            axis = y.dims.index(dim)

    # If both are dataarrays and *x* is coordinates, infer axis from dimension names
    # Detect if user input axis or dim conflicts with inferred one
    # NOTE: Easier to do this than permit *omitting* x coordinates and retrieving
    # coordinates from DataArray. Variable call signatures get tricky.
    if (
        infer_axis
        and isinstance(x, xr.DataArray)
        and isinstance(y, xr.DataArray)
        and x.ndim == 1
        and x.name in y.dims
    ):
        axis_inferred = y.dims.index(x.dims[0])
        if axis is not None and axis != axis_inferred:
            raise ValueError(f'Input {axis=} different from {axis_inferred=}.')
        axis = axis_inferred

    # Apply units and get data
    kwargs['axis'] = axis_default if axis is None else axis
    if isinstance(x, xr.DataArray):
        x = _apply_units(x)
    if isinstance(y, xr.DataArray):
        y = _apply_units(y)

    return x, y, kwargs


def _from_dataarray(
    dataarray, data, name=None, dims=None, attrs=None, coords=None,
    dim_rename=None, dim_coords=None, keep_attrs=False,
):
    """
    Create a copy of the DataArray with various modifications.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The source.
    data : array-like
        The new data.
    name, dims, attrs, coords : optional
        Replacement values.
    dim_rename : 2-tuple of str, optional
        Used to rename dimensions.
    dim_coords : (str, array-like), optional
        The new array coordinates for an arbitrary dimension
    """
    # Get source info
    name = name or dataarray.name
    dims = dims or list(dataarray.dims)
    if coords is None:
        coords = dict(dataarray.coords)
    if attrs is None:
        attrs = dict(dataarray.attrs) if keep_attrs else {}

    # Rename dimension and optionally apply coordinates
    if dim_rename is not None:
        dim_in, dim_out = dim_rename
        coords.pop(dim_in, None)
        dims[dims.index(dim_in)] = dim_out
    if dim_coords is not None:
        dim, coord = dim_coords
        if coord is not None:
            coords[dim] = _remove_units(coord)
        elif dim in coords:  # e.g. ('lat', None) is instruction do delete dimension
            del coords[dim]

    # Return new dataarray
    return xr.DataArray(data, name=name, dims=dims, attrs=attrs, coords=coords)


def _xarray_zerofind_wrapper(*, axis):
    """
    Support `xarray.DataArray` for zerofind function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Sanitize input
            x, y = args
            x_in, y_in, kwargs = _to_arraylike(x, y, axis, **kwargs)

            # Call main function
            x_out, y_out = func(x_in, y_in, **kwargs)

            # Add metadata to x_out and y_out
            if isinstance(y, xr.DataArray):
                attrs = {}
                dim = y.dims[kwargs['axis']]
                if dim in y.coords:
                    attrs = y.coords[dim]
                x_out = _from_dataarray(
                    y, y_out, name=dim, attrs=attrs, dim_rename=(dim, 'track'),
                )
                y_out = _from_dataarray(
                    y, y_out, dim_rename=(dim, 'track'),
                )

            return x_out, y_out

        return wrapper

    return decorator


def _xarray_hist_wrapper(*, axis):
    """
    Support `xarray.DataArray` for `hist` function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Sanitize input
            # NOTE: This time 'x' coordinates are bins so do not infer
            yb, y = args
            yb_in, y_in, kwargs = _to_arraylike(
                yb, y, axis, infer_axis=False, **kwargs
            )

            # Call main function
            y_out = func(yb_in, y_in, **kwargs)

            # Add metadata to y_out
            if isinstance(y, xr.DataArray):
                dim = y.dims[kwargs['axis']]
                name = y.name
                yb_centers = 0.5 * (yb_in[1:] + yb_in[:-1])
                coords = _from_dataarray(
                    y, yb_centers, dims=(name,), coords={}
                )
                y_out = _from_dataarray(
                    y, y_out, name='count', attrs={},
                    dim_rename=(dim, name), dim_coords=(name, coords),
                )

            return y_out

        return wrapper

    return decorator


def _xarray_fit_wrapper(*, axis):
    """
    Generic `xarray.DataArray` wrapper for functions accepting *x* and *y*
    and returning a fit parameter, the fit standard error, and the reconstruction.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Sanitize input
            x, y = args  # *both* or *one* of these is dataarray
            x_in, y_in, kwargs = _to_arraylike(x, y, axis, **kwargs)

            # Call main function
            fit_val, fit_err, fit_line = func(x_in, y_in, **kwargs)

            # Create new output array
            if isinstance(y, xr.DataArray):
                dim_coords = (y.dims[kwargs['axis']], None)
                fit_val = _from_dataarray(y, fit_val, dim_coords=dim_coords)
                fit_err = _from_dataarray(y, fit_err, dim_coords=dim_coords)
                fit_line = _from_dataarray(y, fit_line)  # same everything

            return fit_val, fit_err, fit_line

        return wrapper

    return decorator


def _xarray_xy_y_wrapper(*, axis):
    """
    Generic `xarray.DataArray` wrapper for functions accepting *x* and *y*
    coordinates and returning just *y* coordinates. Permits situation
    where dimension coordinates on returned data are symmetrically trimmed.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Sanitize input
            x, y = args  # *both* or *one* of these is dataarray
            x_in, y_in, kwargs = _to_arraylike(x, y, axis, **kwargs)

            # Call main function
            y_out = func(x_in, y_in, **kwargs)

            # Create new output array
            if isinstance(y, xr.DataArray):
                axis_ = kwargs['axis']
                dim = y.dims[axis_]
                ntrim = (y_in.shape[axis_] - y_out.shape[axis_]) // 2
                if ntrim > 0 and dim in y.coords:
                    dim_coords = (dim, x[ntrim:-ntrim])
                y_out = _from_dataarray(y, y_out, dim_coords=dim_coords)

            return y_out

        return wrapper

    return decorator


def _xarray_xy_xy_wrapper(*, axis):
    """
    Generic `xarray.DataArray` wrapper for functions accepting *x* and *y*
    coordinates and returning new *x* and *y* coordinates.

    Warning
    -------
    So far this fails for 2D `xarray.DataArray` *x* data with non-empty
    coordinates for the `dim` dimension.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Sanitize input
            x, y = args  # *both* or *one* of these is dataarray
            x_in, y_in, kwargs = _to_arraylike(x, y, axis, **kwargs)

            # Call main function
            x_out, y_out = func(x_in, y_in, **kwargs)

            # Create new output array with x coordinates either trimmed
            # or interpolated onto half-levels
            # NOTE: This may fail for 2D DataArray x coordinates
            axis_ = kwargs['axis']
            dim_coods = None
            if x.ndim == 1 and any(isinstance(_, xr.DataArray) for _ in (x, y)):
                dim = x.dims[0] if isinstance(x, xr.DataArray) else y.dims[axis_]
                dim_coods = (dim, x_out)
            if isinstance(x, xr.DataArray):
                x_out = _from_dataarray(x, x_out, dim_coords=dim_coods)
            if isinstance(y, xr.DataArray):
                y_out = _from_dataarray(y, y_out, dim_coords=dim_coods)

            return y_out

        return wrapper

    return decorator


def _pint_wrapper(units_in, units_out, strict=False, **fmt_defaults):
    """
    Handle pint units, similar to `~pint.UnitRegistry.wraps`. Put input units
    as the first argument, set `strict` to ``False`` by default, and if
    non-quantities are passed into the function, ensure non-quantities are returned
    by the function rather than a quantity with ``'dimensionless'`` units, and
    default to ``strict=False``.

    Parameters
    ----------
    units_in : unit-like, string, or list thereof
        A pint unit like `~pint.UnitRegistry.meter`, a relational string
        like ``'=x ** 2'``, or list thereof, specifying the units of the
        input data. You can put keyword arguments into the unit specification
        with, for example ``_pint_wrapper(('=x', '=y'), '=y / x ** {{n}}')``
        for the ``n``th derivative.
    units_out : unit-like, string, or list thereof
        As with `units_in` but for the output arguments.
    strict : bool, optional
        Whether non-relational (absolute) unit specifications are strict.
        If ``False``, non-quantity input is cast to the default units.
    **fmt_defaults
        Default values for the terms surrounded by curly braces in relational
        or string unit specifications.

    Example
    -------
    Here is a simple example for a derivative wrapper.

    >>> import climpy
    ... from climpy.internals import quack
    ... ureg = climpy.ureg
    ... @quack._pint_wrapper(('=x', '=y'), '=y / x ** {order}', order=1)
    ... def deriv(x, y, order=1):
    ...     return y / x ** order
    ... deriv(1 * ureg.m, 1 * ureg.s, order=2)
    1.0 <Unit('second / meter ** 2')>

    """
    # Handle singleton input or multiple input
    is_container_in = isinstance(units_in, (tuple, list))
    is_container_out = isinstance(units_out, (list, tuple))
    if not is_container_in:
        units_in = (units_in,)
    if not is_container_out:
        units_out = (units_out,)

    # Ensure valid kwargs
    # NOTE: Pint cannot handle singleton-tuple of return value unit specifications.
    # So when passing to wrapper simply expand singleton tuples.
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

    # Ensure valid args unit specifications
    for arg in units_in:
        if arg is not None and not isinstance(arg, (ureg.Unit, str)):
            raise TypeError(
                f'Wraps arguments must by of type str or Unit, not {type(arg)} ({arg}).'
            )

    # Ensure valid return unit specifications
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
            if len(units_in) != len(args):
                raise ValueError(
                    f'Expected {len(units_in)} positional args, got {len(args)}.'
                )

            # Fill parameters inside units
            units_in_fmt = []
            units_out_fmt = []
            for units, units_fmt in zip(
                (units_in, units_out), (units_in_fmt, units_out_fmt)
            ):
                for unit in units:
                    if isinstance(unit, str):
                        # Get format values from user input keyword args
                        fmt_keys = REGEX_FORMAT.findall(unit)
                        fmt_kwargs = {
                            key: value for key, value in kwargs.items()
                            if key in fmt_keys
                        }
                        # Fill missing format values and format string
                        for key, value in fmt_defaults.items():
                            fmt_kwargs.setdefault(key, value)
                        unit = unit.format(**fmt_kwargs)
                    # Add new unit string
                    units_fmt.append(unit)

            # Dequantify input
            converter = _parse_pint_args(units_in_fmt)
            args_new, args_by_name = converter(args, strict)

            # Call main function and check output
            result = func(*args_new, **kwargs)
            n_result = 1 if not isinstance(result, tuple) else len(result)
            n_expect = len(units_out)
            if not is_container_out and isinstance(result, tuple):
                raise ValueError('Got tuple of return values, expected one value.')
            if is_container_out and n_result != len(units_out):
                raise ValueError(f'Expected {n_expect} return values, got {n_result}.')

            # Quantify output, but *only* if input was quantities
            if not is_container_out:
                result = (result,)
            pairs = tuple(_to_units_container(arg) for arg in units_out_fmt)
            no_quantities = not any(isinstance(arg, ureg.Quantity) for arg in args)
            units = tuple(
                _replace_units(unit, args_by_name) if is_ref else unit
                for (unit, is_ref) in pairs
            )
            result = tuple(
                res if unit is None or no_quantities else ureg.Quantity(res, unit)
                for unit, res in itertools.zip_longest(units, result)
            )

            # Return sanitized values
            if not is_container_out:
                result = result[0]
            return result

        return wrapper

    return decorator


def _parse_pint_args(args):
    """
    Parse pint wrapper arguments.
    """
    # Helper variables
    defs_args = set()  # arguments which contain definitions
    defs_args_ndx = set()  # (i.e. names that appear alone and for the first time)
    dependent_args_ndx = set()  # arguments which depend on others
    unit_args_ndx = set()  # arguments which have units.
    args_as_uc = [_to_units_container(arg) for arg in args]

    # Check for references in args, remove None values
    for ndx, (arg, is_ref) in enumerate(args_as_uc):
        if arg is None:
            continue
        elif is_ref:
            if len(arg) == 1:
                [(key, value)] = arg.items()
                if value == 1 and key not in defs_args:
                    # This is the first time that
                    # a variable is used => it is a definition.
                    defs_args.add(key)
                    defs_args_ndx.add(ndx)
                    args_as_uc[ndx] = (key, True)
                else:
                    # The variable was already found elsewhere,
                    # we consider it a dependent variable.
                    dependent_args_ndx.add(ndx)
            else:
                dependent_args_ndx.add(ndx)
        else:
            unit_args_ndx.add(ndx)

    # Check that all valid dependent variables
    for ndx in dependent_args_ndx:
        arg, is_ref = args_as_uc[ndx]
        if not isinstance(arg, dict):
            continue
        if not set(arg.keys()) <= defs_args:
            raise ValueError(
                'Found a missing token while wrapping a function: '
                f'Not all variable referenced in {args[ndx]} are defined!'
            )

    # Generate converter
    def _converter(args, strict):
        args_new = list(args)
        args_by_name = {}

        # First pass: Grab named values
        for ndx in defs_args_ndx:
            arg = args[ndx]
            args_by_name[args_as_uc[ndx][0]] = arg
            args_new[ndx] = getattr(arg, '_magnitude', arg)

        # Second pass: calculate derived values based on named values
        for ndx in dependent_args_ndx:
            arg = args[ndx]
            assert _replace_units(args_as_uc[ndx][0], args_by_name) is not None
            args_new[ndx] = ureg._convert(
                getattr(arg, '_magnitude', arg),
                getattr(arg, '_units', putil.UnitsContainer({})),
                _replace_units(args_as_uc[ndx][0], args_by_name),
            )

        # Third pass: convert other arguments
        for ndx in unit_args_ndx:
            if isinstance(args[ndx], ureg.Quantity):
                args_new[ndx] = ureg._convert(
                    args[ndx]._magnitude, args[ndx]._units, args_as_uc[ndx][0]
                )
            elif strict:
                if isinstance(args[ndx], str):
                    # If the value is a string, we try to parse it
                    tmp_value = ureg.parse_expression(args[ndx])
                    args_new[ndx] = ureg._convert(
                        tmp_value._magnitude, tmp_value._units, args_as_uc[ndx][0]
                    )
                else:
                    raise ValueError(
                        'A wrapped function using strict=True requires '
                        'quantity or a string for all arguments with not None units. '
                        f'(error found for {args_as_uc[ndx][0]}, {args_new[ndx]})'
                    )

        return args_new, args_by_name

    return _converter


def _replace_units(original_units, values_by_name):
    """
    Convert a unit compatible type to a UnitsContainer.
    """
    q = 1
    with np.errstate(divide='ignore', invalid='ignore'):
        # NOTE: Multiply quantities successively here just to pull out units
        # after everything is done. But avoid broadcasting errors as shown.
        for arg_name, exponent in original_units.items():
            q = q * values_by_name[arg_name] ** exponent
            m = getattr(q, 'magnitude', q)
            u = getattr(q, 'units', 1)
            if not np.isscalar(m):
                m = m.flat[0]
            q = m * u
    return getattr(q, '_units', putil.UnitsContainer({}))


def _to_units_container(arg):
    """
    Convert a unit compatible type to a UnitsContainer, checking if it is string
    field prefixed with an equal (which is considered a reference). Return the
    unit container and a boolean indicating whether this is an equal string.
    """
    if isinstance(arg, str) and '=' in arg:
        return putil.to_units_container(arg.split('=', 1)[1]), True
    return putil.to_units_container(arg, ureg), False
