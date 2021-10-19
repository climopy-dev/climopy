#!/usr/bin/env python3
"""
Wrappers that support duck-type array input to various functions (hence the filename).
"""
import functools
import inspect
import itertools
import numbers
import re

import numpy as np
import pint
import pint.util as putil
import xarray as xr

from ..unit import ureg
from . import ic  # noqa: F401
from . import _make_logger, warnings

# Set up ArrayContext logger
logger = _make_logger('ArrayContext', 'error')  # or 'info'

# Regex to find terms surrounded by curly braces that can be filled with str.format()
REGEX_FORMAT = re.compile(r'\{([^{}]+?)\}')  # '+?' is non-greedy, group inside brackets


def _as_arraylike(*args):
    """
    Convert list and tuple input to arrays.
    """
    return tuple(
        np.atleast_1d(_) if isinstance(_, (list, tuple, numbers.Number)) else _
        for _ in args
    )


def _as_step(h):
    """
    Convert coordinate to scalar step h. Ensure spacings are constant.
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


def _default_param(func, param):
    """
    Get the default value from the call signature.
    """
    return inspect.signature(func).parameters[param].default


def _interp_safe(x, xp, yp):
    """
    Safe interpolation accounting for pint units. The `yp` can be a DataArray.
    """
    if any(isinstance(_, ureg.Quantity) for _ in (x, xp)):
        if not all(isinstance(_, ureg.Quantity) for _ in (x, xp)) or x.units != xp.units:  # noqa: E501
            raise ValueError('Source and destination coordinates must have same units.')
        xp = xp.magnitude
        x = x.magnitude
    units = None
    yp_in = yp
    if isinstance(yp, xr.DataArray):
        yp_in = yp.data
    if isinstance(yp_in, ureg.Quantity):
        units = yp_in.units
        yp_in = yp_in.magnitude
    if xp[1] - xp[0] < 0:
        xp, yp = xp[::-1], yp_in[::-1]
    y = np.interp(x, xp, yp_in)
    if units is not None:
        y = y * units
    if isinstance(yp, xr.DataArray):
        y = _dataarray_from(yp, y, dim_drop=yp.dims, keep_attrs=True)
    return y


def _is_numeric(data):
    """
    Test if object is numeric, i.e. not string or datetime-like.
    """
    return np.issubdtype(np.asarray(data).dtype, np.number)


def _is_scalar(data):
    """
    Test if object is scalar. Returns ``False`` if it is sized singleton object
    or has more than one entry.
    """
    # WARNING: np.isscalar of dimensionless data returns False
    if isinstance(data, pint.Quantity):
        data = data.magnitude
    data = np.asarray(data)
    return data.ndim == 0


def _dataarray_from(
    dataarray, data, name=None, dims=None, attrs=None, coords=None,
    dim_change=None, dim_coords=None, dim_drop=None, keep_attrs=False,
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
    dim_change : (dim1_old, dim1_new, dim2_old, dim2_new, ...), optional
        Replace an old dimension with a new dimension and drop the old coordinates.
    dim_coords : (dim1, coords1, dim2, coords2, ...), optional
        Mapping to new array coordinates for arbitrary dimension(s).
    dim_drop : (dim1, dim2, ...), optional
        List of dimension coordinates to drop.
    keep_attrs : bool, optional
        Whether to keep the original attributes.
    """
    # Get source info
    name = name or dataarray.name
    dims = dims or list(dataarray.dims)
    if coords is None:
        coords = dict(dataarray.coords)
    if attrs is None:
        attrs = dict(dataarray.attrs) if keep_attrs else {}

    # Rename dimension and optionally apply coordinates
    for dim_in, dim_out in (dim_change or {}).items():
        coords.pop(dim_in, None)
        dims[dims.index(dim_in)] = dim_out
    for dim, coord in (dim_coords or {}).items():
        if coord is not None:
            if isinstance(coord, xr.DataArray):
                coord = coord.climo.dequantify()
            coords[dim] = coord
        elif dim in coords:  # e.g. ('lat', None) is instruction do delete dimension
            del coords[dim]

    # Strip unit if present
    coords_fixed = {}
    dim_drop = dim_drop or ()
    for dim, coord in coords.items():
        if dim not in dims:  # missing coords
            continue
        if dim in dim_drop:
            continue
        if not isinstance(coord, xr.DataArray):
            coord = xr.DataArray(coord, dims=dim, name=dim)
        if coord.climo._is_quantity:
            coord = coord.climo.dequantify()
        coords_fixed[dim] = coord

    # Return new dataarray
    # NOTE: Avoid xarray bug creating data array from scalar quantity
    if isinstance(data, pint.Quantity):
        units, data = data.units, data.magnitude
    else:
        units = 1
    data = xr.DataArray(data, name=name, dims=dims, attrs=attrs, coords=coords_fixed)
    with xr.set_options(keep_attrs=True):
        data *= units
    if not dataarray.climo._is_quantity and data.climo._is_quantity:
        data = data.climo.dequantify()
    return data


def _dataarray_strip(func, x, *ys, suffix='', infer_axis=True, **kwargs):
    """
    Get data from *x* and *y* DataArrays, interpret the `dim` argument, and try to
    determine the `axis` used for computations automatically if ``infer_axis=True``.
    The axis suffix can be e.g. ``'_time'`` and we look for the ``dim_time`` keyword.
    """
    # Convert builtin python types to arraylike
    x, = _as_arraylike(x)
    ys = _as_arraylike(*ys)
    x_dataarray = isinstance(x, xr.DataArray)
    y_dataarray = all(isinstance(y, xr.DataArray) for y in ys)
    axis_default = _default_param(func, 'axis' + suffix)

    # Ensure all *ys* have identical dimensions, type, and shape
    # TODO: Permit automatic broadcasting for various functions?
    y_types = {type(y) for y in ys}
    if len(y_types) != 1:
        raise ValueError(f'Expected one type for y inputs, got {y_types=}.')
    shapes = tuple(y.shape for y in ys)
    if any(_ != shapes[0] for _ in shapes):
        raise ValueError(f'Shapes should be identical, got {shapes}.')
    if y_dataarray:
        dims = tuple(y.dims for y in ys)
        if any(_ != dims[0] for _ in dims):
            raise ValueError(f'Dimensions should be identical, got {dims}.')

    # Interpret 'dim' argument when input is Dataarray
    axis = kwargs.pop('axis' + suffix, None)
    if y_dataarray:
        dim = kwargs.pop('dim' + suffix, None)
        if dim is not None:
            if axis is not None:
                warnings._warn_climopy('Ambiguous axis specification.')
            axis = ys[0].dims.index(dim)

    # If both are dataarrays and *x* is coordinates, infer axis from dimension names
    # Detect if user input axis or dim conflicts with inferred one
    # NOTE: Easier to do this than permit *omitting* x coordinates and retrieving
    # coordinates from DataArray. Variable call signatures get tricky.
    if infer_axis and x_dataarray and y_dataarray and x.ndim == 1 and x.dims[0] in ys[0].dims:  # noqa: E501
        axis_inferred = ys[0].dims.index(x.dims[0])
        if axis is not None and axis != axis_inferred:
            raise ValueError(f'Input {axis=} different from {axis_inferred=}.')
        axis = axis_inferred

    # Finally apply units and strip dataarray data
    args = []
    for arg in (x, *ys):
        if isinstance(arg, xr.DataArray):
            if 'units' in arg.attrs:
                arg = arg.climo.quantify()
            arg = arg.data
        args.append(arg)
    kwargs['axis' + suffix] = axis_default if axis is None else axis

    return (*args, kwargs)


def _xarray_yy_wrapper(func):
    """
    Generic `xarray.DataArray` wrapper for functions accepting and returning
    arrays of the same shape. Permits situation where dimension coordinates
    of returned data are symmetrically trimmed.
    """
    @functools.wraps(func)
    def wrapper(y, *args, **kwargs):
        _, y_in, kwargs = _dataarray_strip(func, 0, y, **kwargs)
        y_out = func(y_in, *args, **kwargs)

        # Build back the DataArray
        if isinstance(y, xr.DataArray):
            axis = kwargs['axis']
            dim = y.dims[axis]
            ntrim = (y_in.shape[axis] - y_out.shape[axis]) // 2
            dim_coords = None
            if ntrim > 0 and dim in y.coords:
                dim_coords = {dim: y.coords[dim][ntrim:-ntrim]}
            y_out = _dataarray_from(y, y_out, dim_coords=dim_coords)

        return y_out

    return wrapper


def _xarray_xyy_wrapper(func):
    """
    Generic `xarray.DataArray` wrapper for functions accepting *x* and *y*
    coordinates and returning just *y* coordinates. Permits situation
    where dimension coordinates of returned data are symmetrically trimmed.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, y = args
        x_in, y_in, kwargs = _dataarray_strip(func, x, y, **kwargs)
        y_out = func(x_in, y_in, **kwargs)

        # Build back the DataArray
        # NOTE: Account for symmetrically trimmed coords here
        if isinstance(y, xr.DataArray):
            axis = kwargs['axis']
            dim = y.dims[axis]
            dx = (y_in.shape[axis] - y_out.shape[axis]) // 2
            if dx > 0 and dim in y.coords:
                dim_coords = {dim: y.coords[y.dims[axis]][dx:-dx]}
            else:
                dim_coords = None
            y_out = _dataarray_from(y, y_out, dim_coords=dim_coords)

        return y_out

    return wrapper


def _xarray_xyxy_wrapper(func):
    """
    Generic `xarray.DataArray` wrapper for functions accepting *x* and *y*
    coordinates and returning new *x* and *y* coordinates.

    Warning
    -------
    So far this fails for 2D `xarray.DataArray` *x* data with non-empty
    coordinates for the `dim` dimension.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, y = args  # *both* or *one* of these is dataarray
        x_in, y_in, kwargs = _dataarray_strip(func, x, y, **kwargs)
        x_out, y_out = func(x_in, y_in, **kwargs)

        # Build back the DataArray. Trim x coordinates or interpolate to half-levels
        # NOTE: Also modify coordinates associated with x array, which may differ
        # from array values themselves (e.g. heights converted from pressure).
        axis = kwargs['axis']
        if x_in.size == 1:
            x_in_sel = np.arange(x_out.size)
            x_out_sel = x_out
        elif x_in.ndim == 1 and x_out.ndim == 1:
            x_in_sel = x_in
            x_out_sel = x_out
        elif x_in.ndim > 1 and x_out.ndim > 1:
            slice_ = [0] * x_in.ndim
            slice_[axis] = slice(None)
            x_in_sel = x_in[tuple(slice_)]
            x_out_sel = x_out[tuple(slice_)]
        else:
            raise ValueError('Unexpected dimensionality of x coordinates.')
        if isinstance(x, xr.DataArray):
            dim = x.dims[0] if x.ndim == 1 else x.dims[axis]
            if dim in x.coords:
                dim_coords = {dim: _interp_safe(x_out_sel, x_in_sel, x.coords[dim])}
            else:
                dim_coords = None
            x_out = _dataarray_from(x, x_out, dim_coords=dim_coords)
        if isinstance(y, xr.DataArray):
            dim = y.dims[axis]
            if dim in y.coords:
                dim_coords = {dim: _interp_safe(x_out_sel, x_in_sel, y.coords[dim])}
            else:
                dim_coords = None
            y_out = _dataarray_from(y, y_out, dim_coords=dim_coords)

        return x_out, y_out

    return wrapper


def _xarray_power_wrapper(func):
    """
    Support `xarray.DataArray` for `power` and `copower`.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, *ys = args  # *both* or *one* of these is dataarray
        y = ys[0]
        x_in, *ys_in, kwargs = _dataarray_strip(func, x, *ys, **kwargs)
        f, *Ps = func(x_in, *ys_in, **kwargs)

        # Build back the DataArray
        if isinstance(x, xr.DataArray):
            f = _dataarray_from(
                x, f, dims=('f',), coords={}, attrs={'long_name': 'frequency'},
            )
        if isinstance(y, xr.DataArray):
            dim = y.dims[kwargs['axis']]
            Ps = (
                _dataarray_from(y, P, dim_change={dim: 'f'}, dim_coords={'f': f})
                for P in Ps
            )

        return f, *Ps

    return wrapper


def _xarray_power2d_wrapper(func):
    """
    Support `xarray.DataArray` for `power2d` and `copower2d`.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x1, x2, *ys = args  # *both* or *one* of these is dataarray
        y = ys[0]
        x1_in, *ys_in, kwargs = _dataarray_strip(func, x1, *ys, suffix='_lon', **kwargs)
        x2_in, *_, kwargs = _dataarray_strip(func, x2, *ys, suffix='_time', **kwargs)
        k, f, *Ps = func(x1_in, x2_in, *ys_in, **kwargs)

        # Build back the DataArray
        if isinstance(x1, xr.DataArray):
            k = _dataarray_from(
                x1, dims=('k',), coords={}, attrs={'long_name': 'wavenumber'},
            )
        if isinstance(x2, xr.DataArray):
            f = _dataarray_from(
                x2, dims=('f',), coords={}, attrs={'long_name': 'frequency'},
            )
        if isinstance(y, xr.DataArray):
            dim1 = y.dims[kwargs['axis_lon']]
            dim2 = y.dims[kwargs['axis_time']]
            Ps = (
                _dataarray_from(
                    y, P, dim_change={dim1: 'k', dim2: 'f'}, dim_coords={'k': k, 'f': f}
                )
                for P in Ps
            )

        return k, *Ps

    return wrapper


def _xarray_lls_wrapper(func):
    """
    Generic `xarray.DataArray` wrapper for functions accepting *x* and *y*
    and returning a fit parameter, the fit standard error, and the reconstruction.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, y = args  # *both* or *one* of these is dataarray
        x_in, y_in, kwargs = _dataarray_strip(func, x, y, **kwargs)
        fit_val, fit_err, fit_line = func(x_in, y_in, **kwargs)

        # Build back the DataArray
        if isinstance(y, xr.DataArray):
            dim_coords = {y.dims[kwargs['axis']]: None}
            fit_val = _dataarray_from(y, fit_val, dim_coords=dim_coords)
            fit_err = _dataarray_from(y, fit_err, dim_coords=dim_coords)
            fit_line = _dataarray_from(y, fit_line)  # same everything

        return fit_val, fit_err, fit_line

    return wrapper


def _xarray_covar_wrapper(func):
    """
    Support `xarray.DataArray` for `corr`, `covar`, `autocorr`, and `autocovar` funcs.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, *ys = args  # *both* or *one* of these is dataarray
        y = ys[0]
        x_in, *ys_in, kwargs = _dataarray_strip(func, x, *ys, **kwargs)
        lag, C = func(x_in, *ys_in, **kwargs)

        # Build back the DataArray
        if isinstance(x, xr.DataArray):
            lag = _dataarray_from(x, lag, name='lag', dims=('lag',), coords={})
        if isinstance(y, xr.DataArray):
            dim = y.dims[kwargs['axis']]
            C = _dataarray_from(y, C, dim_change={dim: 'lag'}, dim_coords={'lag': lag})

        return lag, C

    return wrapper


def _xarray_eof_wrapper(func):
    """
    Support `xarray.DataArray` for `eof` functions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        data, = data_in, = args
        axis_time = kwargs.pop('axis_time', _default_param(func, 'axis_time'))
        axis_space = kwargs.pop('axis_space', _default_param(func, 'axis_space'))
        if isinstance(data, xr.DataArray):
            data_in = data.data
            if 'dim_time' in kwargs:  # no warning for duplicate args, no big deal
                axis_time = data.dims.index(kwargs.pop('dim_time'))
            if 'dim_space' in kwargs:
                axis_space = data.dims.index(kwargs.pop('dim_space'))

        pcs, projs, evals, nstars = func(
            data_in, axis_time=axis_time, axis_space=axis_space, **kwargs,
        )

        # Build back the DataArray
        if isinstance(data, xr.DataArray):
            # Add EOF dimension
            dims = ['eof'] + list(data.dims)  # add 'EOF number' leading dimension
            eofs = np.arange(1, pcs.shape[0] + 1)

            # Remove dimension coordinates where necessary
            time_flat = {data.dims[axis_time]: None}
            space_flat = {data.dims[axis]: None for axis in np.atleast_1d(axis_space)}
            both_flat = {**time_flat, **space_flat}
            all_flat = both_flat.copy()  # with singleton EOF dimension
            both_flat['eof'] = time_flat['eof'] = space_flat['eof'] = eofs

            # Create DataArrays
            pcs = _dataarray_from(data, pcs, dims=dims, dim_coords=space_flat)
            projs = _dataarray_from(data, projs, dims=dims, dim_coords=time_flat)
            evals = _dataarray_from(data, evals, dims=dims, dim_coords=both_flat)
            nstars = _dataarray_from(data, nstars, dims=dims, dim_coords=all_flat)

        return pcs, projs, evals, nstars

    return wrapper


def _xarray_hist_wrapper(func):
    """
    Support `xarray.DataArray` for `hist` function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # NOTE: This time 'x' coordinates are bins so do not infer
        yb, y = args
        yb_in, y_in, kwargs = _dataarray_strip(func, yb, y, infer_axis=False, **kwargs)
        y_out = func(yb_in, y_in, **kwargs)

        # Build back the DataArray
        if isinstance(y, xr.DataArray):
            dim = y.dims[kwargs['axis']]
            name = y.name
            yb_centers = 0.5 * (yb_in[1:] + yb_in[:-1])
            coords = _dataarray_from(
                y, yb_centers, dims=(name,), coords={}, keep_attrs=True,
            )
            y_out = _dataarray_from(
                y, y_out, name='count', attrs={},
                dim_change={dim: name}, dim_coords={name: coords},
            )

        return y_out

    return wrapper


def _xarray_find_wrapper(func):
    """
    Support `xarray.DataArray` for find function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, y = args
        x_in, y_in, kwargs = _dataarray_strip(func, x, y, **kwargs)
        x_out, y_out = func(x_in, y_in, **kwargs)

        # Build back the DataArray
        # NOTE: x_out inherits *shape* from y_out but should inherit *attrs* from x_in.
        if isinstance(y, xr.DataArray):
            attrs = {}
            dim = y.dims[kwargs['axis']]
            if dim in y.coords:
                attrs = y.coords[dim].attrs.copy()
            if 'units' in attrs:
                del attrs['units']  # units should have been applied with pint
            x_out = _dataarray_from(
                y, x_out, name=dim, attrs=attrs, dim_change={dim: 'track'},
            )
            y_out = _dataarray_from(
                y, y_out, keep_attrs=True, dim_change={dim: 'track'},
            )

        return x_out, y_out

    return wrapper


def _pint_units_container(arg):
    """
    Convert a unit compatible type to a UnitsContainer, checking if it is string field
    prefixed with an equal (which is considered a reference). Return the unit container
    and a boolean indicating whether this is an equal string.
    """
    if isinstance(arg, str) and '=' in arg:
        return putil.to_units_container(arg.split('=', 1)[1]), True
    return putil.to_units_container(arg, ureg), False


def _pint_replace_units(original_units, values_by_name):
    """
    Convert a unit compatible type to a UnitsContainer.
    """
    q = 1
    with np.errstate(divide='ignore', invalid='ignore'):
        # NOTE: Multiply quantities successively here just to pull out units
        # after everything is done. But avoid broadcasting errors as shown.
        for arg_name, exponent in original_units.items():
            try:
                q = q * values_by_name[arg_name] ** exponent
            except ValueError:  # avoid negative integer powers of integer arrays
                q = q * values_by_name[arg_name].astype(float) ** exponent
            m = getattr(q, 'magnitude', q)
            u = getattr(q, 'units', 1)
            if not np.isscalar(m):
                m = m.flat[0]
            q = m * u
    return getattr(q, '_units', putil.UnitsContainer({}))


def _pint_parse_args(args):
    """
    Parse pint wrapper arguments.
    """
    # Helper variables
    defs_args = set()  # arguments which contain definitions
    defs_args_ndx = set()  # (i.e. names that appear alone and for the first time)
    dependent_args_ndx = set()  # arguments which depend on others
    unit_args_ndx = set()  # arguments which have units.
    args_as_uc = [_pint_units_container(arg) for arg in args]

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
            assert _pint_replace_units(args_as_uc[ndx][0], args_by_name) is not None
            args_new[ndx] = ureg._convert(
                getattr(arg, '_magnitude', arg),
                getattr(arg, '_units', putil.UnitsContainer({})),
                _pint_replace_units(args_as_uc[ndx][0], args_by_name),
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


def _pint_wrapper(units_in, units_out, strict=False, **fmt_defaults):
    """
    Handle pint units, similar to `~pint.UnitRegistry.wraps`. Put input units as the
    first argument, set `strict` to ``False`` by default, and if non-quantities are
    passed into the function, ensure non-quantities are returned by the function rather
    than a quantity with ``'dimensionless'`` units.

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

    >>> from climopy.internals import quack
    >>> from climopy import ureg
    >>> @quack._pint_wrapper(('=x', '=y'), '=y / x ** {order}', order=1)
    ... def deriv(x, y, order=1):
    ...     return y / x ** order
    >>> deriv(1 * ureg.m, 1 * ureg.s, order=2)
    <Quantity(1.0, 'second / meter ** 2')>
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
            n_expect = len(units_in)
            n_result = len(args)
            if len(units_in) != len(args):
                raise ValueError(f'Expected {n_expect} positional args, got {n_result}.')  # noqa: E501

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
            converter = _pint_parse_args(units_in_fmt)
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
            pairs = tuple(_pint_units_container(arg) for arg in units_out_fmt)
            no_quantities = not any(isinstance(arg, ureg.Quantity) for arg in args)
            units = tuple(
                _pint_replace_units(unit, args_by_name) if is_ref else unit
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


class _ArrayContext(object):
    """
    Temporarily reshape the input dataset(s). This is needed so we can do objective
    analysis tasks "along an axis". Some tasks can be done by just moving axes and using
    array[..., :] notation but this is not always possible. Should work with arbitrary
    duck-type arrays, including dask arrays.
    """
    def __init__(
        self, *args,
        push_right=None, push_left=None, nflat_right=None, nflat_left=None,
    ):
        """
        Parameters
        ----------
        *datas : numpy.ndarray
            The arrays to be reshaped
        push_left, push_right : int or list of int, optional
            Axis or axes to move to the left or right sides. Axes are moved in the input
            order. By default, if neither are provided, `push_right` is set to ``-1``.
        nflat_left, nflat_right : int, optional
            Number of dimensions to flatten on the left or right sides. By default, if
            only `push_left` is provided, `nflat_right` is set to ``data.ndim -
            len(push_left)``, and if only `push_right` is provided, `nflat_left` is set
            to ``data.ndim - len(push_right)``.

        Examples
        --------
        Here is a worked example used with the EOF algorithm:

        >>> import logging
        >>> import numpy as np
        >>> import xarray as xr
        >>> from climopy.internals.quack import logger, _ArrayContext
        >>> logger.setLevel(logging.INFO)
        >>> # Generate neof, member, run, time, plev, lat array
        >>> dataarray = xr.DataArray(
        ...     np.random.rand(12, 8, 100, 40, 20),
        ...     dims=('member', 'run', 'time', 'plev', 'lat'),
        ... )
        >>> array = dataarray.data
        >>> with _ArrayContext(
        ...     array,
        ...     push_left=(0, 1), nflat_left=2,
        ...     push_right=(2, 3, 4), nflat_right=2,
        ... ) as context:
        ...     data = context.data
        ...     nextra, ntime, nspace = data.shape
        ...     eofs = np.random.rand(nextra, 5, 1, nspace)  # singleton time dimension
        ...     pcs = np.random.rand(nextra, 5, ntime, 1)  # singleton space dimension
        ...     context.replace_data(eofs, pcs, insert_left=1)
        >>> logger.setLevel(logging.ERROR)
        >>> eofs, pcs = context.data
        """
        # Set arrays
        # NOTE: No array standardization here. Assume duck-type arrays (numpy
        # arrays, pint quantities, xarray DataArrays, dask arrays).
        if not args:
            raise ValueError('Need at least one input argument.')
        self._arrays = args
        self._shapes = []
        self._moves = []
        ndim = self._arrays[0].ndim

        # Parse axis arguments and ensure they are positive
        if push_right is None and push_left is None:
            push_right = -1
        if push_right is None:
            push_right = np.array([])
        else:
            push_right = np.atleast_1d(push_right)
        if push_left is None:
            push_left = np.array([])
        else:
            push_left = np.atleast_1d(push_left)
        for push, side in zip((push_left, push_right), ('left', 'right')):
            push[push < 0] += ndim
            if any(push < 0) or any(push >= ndim) or np.unique(push).size != push.size:
                raise ValueError(f'Invalid push_{side}={push} for {ndim}D array.')
        self._push_left = push_left
        self._push_right = push_right

        # Parse nflat arguments. When user requests pushing to right, means we want
        # to flatten the remaining left dims. Same goes for pushing to left.
        # NOTE: There is distinction here between 'None' and '0'. The latter means
        # add a singleton dimension (useful when iterating over 'extra' dimensions)
        # while the former means add nothing.
        if nflat_left is None and not push_left.size and push_right.size:
            nflat_left = ndim - push_right.size
        if nflat_right is None and not push_right.size and push_left.size:
            nflat_right = ndim - push_left.size
        self._nflat_left = nflat_left
        self._nflat_right = nflat_right

    def replace_data(self, *args, insert_left=None, insert_right=None):
        """
        Replace the data attribute with new array(s).

        Parameters
        ----------
        *args : array-like, optional
            The new arrays. The unflattened middle-dimensions can be changed. The
            flattened leading or trailing dimensions can be reduced to singleton, but
            otherwise must be identical or it is unclear how they should be re-expanded.
        insert_left, insert_right : int, optional
            Number of new dimensions added to the left or right of the array.
            Dimensions can only be added to the left or the right of the
            unflattened middle-dimensions of the array. For example, `climopy.eof`
            adds a new `neof` dimension so that dimensions are transformed
            from ``(nextra, ntime, nspace)`` to ``(nextra, neof, ntime, nspace)``.
            Use lists of numbers to transform input arguments differently.

        Examples
        --------
        Inserting new dimensions does not mess up the order of values in dimensions
        that come before or after. This is revealed by playing with a simple example.

        >>> a = np.array(
        ...     [
        ...         [[1, 2, 1], [3, 4, 3]],
        ...         [[5, 6, 5], [7, 8, 7]],
        ...         [[9, 10, 9], [11, 12, 11]],
        ...     ]
        ... )
        >>> a.shape
        (3, 2, 3)
        >>> a[:, 0, 0]
        array([1, 5, 9])
        >>> np.reshape(a, (3, 6), order='F')[:, 0]
        array([1, 5, 9])
        >>> np.reshape(a, (3, 6), order='C')[:, 0]
        array([1, 5, 9])
        """
        # Parse arguments
        inserts_left, inserts_right = [], []
        for inserts, insert in zip((inserts_left, inserts_right), (insert_left, insert_right)):  # noqa: E501
            insert = np.atleast_1d(insert).tolist()
            if len(insert) == 1:
                insert = insert * len(args)
            elif len(insert) != len(args):
                raise ValueError(f'Got {len(insert)} inserts but {len(args)} args.')
            inserts[:] = insert

        # Check input array shapes
        # WARNING: The *flattened* dimensions of the new data must match the size
        # of the *flattened* dimensions of the input data. Flattened dimensions should
        # only be iterated over or reduced to length 1 by climopy functions like `eof`.
        shape_template = self._shapes[0]
        if not all(shape == shape_template for shape in self._shapes):
            raise ValueError(
                'Cannot reset dimensions when input data shapes '
                + ', '.join(map(repr, self._shapes)) + ' differ.'
            )

        # Loop through arrays
        nflat_left = self._nflat_left
        nflat_right = self._nflat_right
        shape_flat = self._arrays[0].shape
        shape_unflat_orig = self._shapes[0]
        self._arrays = []
        self._shapes = []
        self._moves = []
        for array, insert_left, insert_right in zip(args, inserts_left, inserts_right):
            # Check shape against flattened dimensions
            logger.info('')
            logger.info(f'Add new context array: {array.shape}')
            shape = list(array.shape)
            insert_left = insert_left or 0  # *number* of dimensions inserted left
            insert_right = insert_right or 0  # *number* of dimensions inserted right
            if (
                len(shape_flat) + insert_left + insert_right != len(shape)
            ) or (
                nflat_left is not None
                and shape[0] != shape_flat[0]
                and shape[0] > 1  # reduction to singleton is allowed
            ) or (
                nflat_right is not None
                and shape[-1] != shape_flat[-1]
                and shape[-1] > 1  # reduction to singleton is allowed
            ):
                raise ValueError(
                    f'New flattened array shape {shape!r} incompatible with '
                    f'existing flattened array shape {shape_flat!r}.'
                )

            # Determine *unflattened* shape from template shape
            shape_unflat = shape_unflat_orig.copy()
            if nflat_left is None:
                ileft_flat = 0
                nleft_unflat = 0
            else:
                ileft_flat = 1
                nleft_unflat = nflat_left
                if shape[0] <= 1:
                    for i in range(nflat_left):
                        shape_unflat[i] = 1
            if nflat_right is None:
                iright_flat = len(shape)
                nright_unflat = 0
            else:
                iright_flat = len(shape) - 1
                nright_unflat = nflat_right
                if shape[-1] <= 1:
                    for i in range(1, nflat_right + 1):
                        shape_unflat[-i] = 1

            # Build unflattened shape
            shape_left = shape_unflat[:nleft_unflat]
            shape_center = shape[ileft_flat:iright_flat]  # includes inserted
            shape_right = shape_unflat[len(shape_unflat) - nright_unflat:]
            shape = (*shape_left, *shape_center, *shape_right)
            logger.info(f'Change flattened shape {shape_flat} to {array.shape}.')
            logger.info(f'Number of left-flattened dimensions: {nflat_left}')
            logger.info(f'Number of right-flattened dimensions: {nflat_right}')
            logger.info(f'Flattened left dimensions: {shape_left}')
            logger.info(f'New center dimensions: {shape_center}')
            logger.info(f'Flattened right dimensions: {shape_right}')
            logger.info(f'Change unflattened shape {shape_unflat} to {shape}.')
            self._arrays.append(array)
            self._shapes.append(shape)

            # Correct the axis moves given new *inserted* dimensions
            # Example: Original array has shape [A, B, C, D, E] with push_left [1]
            # and push_right [0, 3]. Want the new array *final* shape (after swapping
            # axes) will be [X, Y, A, B, C, D, E, Z]. Now pretend this was the
            # *initial* dimensionality. Input push_left *and* push_right would be
            # plus 2 (shifted by 2 new axes), and input push_right unchanged.
            push_left = self._push_left + insert_left
            push_right = self._push_right + insert_left
            moves = self._get_axis_moves(
                push_left,
                push_right,
                left_base=insert_left,
                right_base=(-1 - insert_right),
            )
            self._moves.append(moves)

    def __enter__(self):
        """
        Reshape the array.
        """
        # NOTE: Hard to build intuition for ND reshaping, but think of it as
        # just changing the *indices* used to refernece elements. For 2 x 3 x ...
        # array, row-major flattening creates 6 x ... array whose indices
        # correspond to A[0, 0], A[0, 1], A[0, 2], A[1, 0], A[1, 1], A[1, 2].
        # For column-major array, indices correspond to A[0, 0], A[1, 0],
        # A[0, 1], A[1, 1], A[0, 2], A[1, 2]. Other dimensions not affected.
        arrays = self._arrays
        nflat_left = self._nflat_left
        nflat_right = self._nflat_right
        self._arrays = []
        self._shapes = []
        self._moves = []
        for array in arrays:
            # Move axes
            logger.info('')
            logger.info(f'Flatten array: {array.shape}')
            push_left = self._push_left.copy()  # *must* be copy or replace_data fails!
            push_right = self._push_right.copy()
            moves = self._get_axis_moves(push_left, push_right)
            array = self._run_axis_moves(array, moves)

            # Get new left shape
            ndim = array.ndim
            shape = list(array.shape)
            reshape = shape[nflat_left or 0:ndim - (nflat_right or 0)]
            if nflat_left is not None:
                s = shape[:nflat_left]
                N = np.prod(s).astype(int)
                reshape.insert(0, N)
                logger.info(f'Flatten {nflat_left} left dimensions: {s} to {N}')

            # Get new right shape
            if nflat_right is not None:
                s = shape[ndim - nflat_right:]
                N = np.prod(s).astype(int)
                reshape.append(N)
                logger.info(f'Flatten {nflat_right} right dimensions: {s} to {N}')

            # Reshape
            if shape != reshape:
                # WARNING: 'order' arg is invalid for dask arrays
                logger.info(f'Reshape from {array.shape} to {reshape}')
                array = np.reshape(array, reshape)
            self._arrays.append(array)
            self._moves.append(moves)
            self._shapes.append(shape)

        return self

    def __exit__(self, *args):  # noqa: U100
        """
        Restore the array to its original shape.
        """
        arrays = self._arrays
        shapes = self._shapes
        moves = self._moves
        self._arrays = []
        self._shapes = []
        self._moves = []
        for array, ishape, imoves in zip(arrays, shapes, moves):
            logger.info('')
            logger.info(f'Unflatten array: {array.shape}')
            if array.shape != ishape:
                # WARNING: 'order' arg is invalid for dask arrays
                logger.info(f'Reshape from {array.shape} to {ishape}')
                array = np.reshape(array, ishape)
            array = self._run_axis_moves(array, imoves, reverse=True)
            self._arrays.append(array)

    @staticmethod
    def _get_axis_moves(push_left, push_right, left_base=0, right_base=-1):
        """
        Get the series of axis swaps given the input dimensionality.
        """
        logger.info(f'Push axes left: {push_left}')
        logger.info(f'Push axes right: {push_right}')
        moves = []
        left_base = 0
        right_base = -1
        for i, axis in enumerate(push_right):
            moves.append((axis, right_base))
            for push in (push_left, push_right):
                push[push > axis] -= 1  # NOTE: some of these changes have no effect
        for axis in push_left:
            moves.append((axis, left_base))
            for push in (push_left, push_right):
                push[push < axis] += 1  # NOTE: some of these changes have no effect
        return np.array(moves)

    @staticmethod
    def _run_axis_moves(array, moves, reverse=False):
        """
        Execute the input axis moves.
        """
        slice_ = slice(None, None, -1) if reverse else slice(None)
        for move in moves[slice_]:
            move = move[slice_]
            array = np.moveaxis(array, *move)
            logger.info(f'Move {move[0]} to {move[1]}: {array.shape}')
        return array

    @property
    def data(self):
        """
        The arrays. Use this to retrieve reshaped arrays within the context block for
        your computation and outside the context block once they are reshaped back.
        """
        arrays = self._arrays
        if len(arrays) == 1:
            return arrays[0]
        else:
            return tuple(arrays)
