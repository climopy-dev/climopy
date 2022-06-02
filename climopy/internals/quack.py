#!/usr/bin/env python3
"""
Tools for managing duck-type data arrays.
"""
import functools
import inspect
import numbers

import numpy as np
import pint
import xarray as xr

from ..unit import ureg
from . import ic  # noqa: F401
from . import warnings

# TODO: Make basic utilities public in future
# TODO: Add support for NetCDF4 dataset types
__all__ = []


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


def _keep_cell_attrs(func):
    """
    Preserve attributes for duration of function call with `update_cell_attrs`.
    """
    @functools.wraps(func)
    def _wrapper(self, *args, no_keep_attrs=False, **kwargs):
        result = func(self, *args, **kwargs)  # must return a DataArray
        if no_keep_attrs:
            return result
        if not isinstance(result, (xr.DataArray, xr.Dataset)):
            raise TypeError('Wrapped function must return a DataArray or Dataset.')
        result.climo.update_cell_attrs(self)
        return result

    return _wrapper


def _while_quantified(func, always_quantify=False):
    """
    Return a wrapper that temporarily quantifies the data.
    Compare to `~.internals.quant.while_quantified`.
    """
    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        # Quantify
        data = self.data
        if isinstance(data, xr.Dataset):
            data = data.copy(deep=False)
            quantified = set()
            for da in data.values():
                if da.climo._is_coordinate_bounds:  # i.e. missing attributes
                    pass
                elif da.climo._has_units:
                    if not da.climo._is_quantity:
                        da.climo._quantify()
                        quantified.add(da.name)
                else:
                    if always_quantify:
                        msg = f'Cannot quantify DataArray with no units {da.name!r}.'
                        raise RuntimeError(msg)
        elif not self._is_quantity:
            data = data.climo.quantify()

        # Main function
        result = func(data.climo, *args, **kwargs)

        # Dequantify
        if isinstance(data, xr.Dataset):
            result = result.copy(deep=False)
            for name in quantified:
                result[name].climo._dequantify()
        elif not self._is_quantity:
            result = result.climo.dequantify()

        return result

    return _wrapper


def _while_dequantified(func):
    """
    Return a wrapper that temporarily dequantifies the data.
    Compare to `~.internals.quant.while_dequantified`.
    """
    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        # Dequantify
        data = self.data
        if isinstance(data, xr.Dataset):
            data = data.copy(deep=False)
            dequantified = {}
            for da in data.values():
                if da.climo._is_quantity:
                    dequantified[da.name] = da.data.units
                    da.climo._dequantify()
        elif self._is_quantity:
            units = data.data.units
            data = data.climo.dequantify()

        # Main function
        result = func(data.climo, *args, **kwargs)

        # Quantify
        # NOTE: In _find_extrema, units actually change! Critical that we avoid
        # overwriting (this is default behavior when passing units to quantify).
        if isinstance(data, xr.Dataset):
            result = result.copy(deep=False)
            for name, units in dequantified.items():
                units = None if 'units' in result[name].attrs else units
                result[name].climo._quantify(units=units)
        elif self._is_quantity:
            units = None if 'units' in result.attrs else units
            result.climo.quantify(units=units)

        return result

    return _wrapper


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


def _dataarray_strip(x, *ys, suffix='', infer_axis=True, **kwargs):
    """
    Get data from *x* and *y* DataArrays, interpret the `dim` argument, and try to
    determine the `axis` used for computations automatically if ``infer_axis=True``.
    Can also translate suffixed dimensions to suffixed axes, e.g. `dim_time`.

    Parameters
    ----------
    x, *ys : array-like
        The input data.
    suffix : str or sequence,
        The axis keyword suffix(es).
    infer_axis : bool, optional
        Whether to infer the default `axis` keyword. Suffixed keywords are ignored.
    **kwargs
        Function keyword arguments. This may include axis and dimension specifiers.
    """
    # Convert builtin python types to arraylike
    x, = _as_arraylike(x)
    ys = _as_arraylike(*ys)
    x_dataarray = isinstance(x, xr.DataArray)
    y_dataarray = all(isinstance(y, xr.DataArray) for y in ys)

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

    # Interpret 'dim' argument when input is DataArray, and if both inputs are
    # DataArray and *x* is coordinates, infer axis from dimension names
    # NOTE: Easier to do this than permit *omitting* x coordinates and retrieving
    # coordinates from DataArray... but should still allow the latter in the future.
    suffixes = (suffix,) if isinstance(suffix, str) else tuple(suffix)
    for suffix in suffixes:
        axis = kwargs.pop('axis' + suffix, None)
        if y_dataarray:
            dim = kwargs.pop('dim' + suffix, None)
            if dim is not None:
                msg = f'Ambiguous axis specification {dim=} and {axis=}. Using {dim=}.'
                if axis is not None:
                    warnings._warn_climopy(msg)
                axis = ys[0].dims.index(dim)
            infer = infer_axis and not suffix  # only for naked 'axis' keyword
            if infer and x_dataarray and x.ndim == 1 and x.dims[0] in ys[0].dims:
                axis_inferred = ys[0].dims.index(x.dims[0])
                if axis is not None and axis != axis_inferred:
                    raise ValueError(f'Input {axis=} different from {axis_inferred=}.')
                axis = axis_inferred
        if axis is not None:
            kwargs['axis' + suffix] = axis

    # Finally apply units and strip dataarray data
    # NOTE: Use .data not .values to preserve e.g. Quantity,
    # dask array, or numpy masked array data.
    args = []
    for arg in (x, *ys):
        if isinstance(arg, xr.DataArray):
            if 'units' in arg.attrs:
                arg = arg.climo.quantify()
            arg = arg.data
        args.append(arg)
    return (*args, kwargs)


def _yy_metadata(func):
    """
    Generic `xarray.DataArray` wrapper for functions accepting and returning
    arrays of the same shape. Permits situation where dimension coordinates
    of returned data are symmetrically trimmed.
    """
    @functools.wraps(func)
    def wrapper(y, *args, **kwargs):
        _, y_in, kwargs = _dataarray_strip(0, y, **kwargs)
        y_out = func(y_in, *args, **kwargs)

        # Build back the DataArray
        # NOTE: Data might be shifted left by one if had even numbered
        # runmean window, so the padding may not be exactly symmetrical.
        if isinstance(y, xr.DataArray):
            axis = kwargs['axis']
            dim = y.dims[axis]
            d1 = (y_in.shape[axis] - y_out.shape[axis]) // 2
            d2 = y_in.shape[axis] - y_out.shape[axis] - d1
            dim_coords = None
            if d1 > 0 and dim in y.coords:
                if kwargs.get('center', True):
                    dim_coords = {dim: y.coords[dim][d1:-d2]}
                else:
                    dim_coords = {dim: y.coords[dim][d1 + d2:]}
            y_out = _dataarray_from(y, y_out, dim_coords=dim_coords)

        return y_out

    return wrapper


def _xyy_metadata(func):
    """
    Generic `xarray.DataArray` wrapper for functions accepting *x* and *y*
    coordinates and returning just *y* coordinates. Permits situation
    where dimension coordinates of returned data are symmetrically trimmed.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, y = args
        x_in, y_in, kwargs = _dataarray_strip(x, y, **kwargs)
        y_out = func(x_in, y_in, **kwargs)

        # Build back the DataArray
        # NOTE: Account for symmetrically trimmed coords here
        if isinstance(y, xr.DataArray):
            axis = kwargs['axis']
            dim = y.dims[axis]
            dx = (y_in.shape[axis] - y_out.shape[axis]) // 2
            dim_coords = None
            if dx > 0 and dim in y.coords:
                dim_coords = {dim: y.coords[y.dims[axis]][dx:-dx]}
            y_out = _dataarray_from(y, y_out, dim_coords=dim_coords)

        return y_out

    return wrapper


def _xyxy_metadata(func):
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
        x_in, y_in, kwargs = _dataarray_strip(x, y, **kwargs)
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


def _power_metadata(func):
    """
    Support `xarray.DataArray` for `power` and `copower`.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, *ys = args  # *both* or *one* of these is dataarray
        y = ys[0]
        x_in, *ys_in, kwargs = _dataarray_strip(x, *ys, **kwargs)
        f, *Ps = func(x_in, *ys_in, **kwargs)

        # Build back the DataArray
        if isinstance(x, xr.DataArray):
            f = _dataarray_from(
                x, f, dims=('f',), coords={}, attrs={'long_name': 'frequency'}
            )
        if isinstance(y, xr.DataArray):
            dim = y.dims[kwargs['axis']]
            Ps = (
                _dataarray_from(
                    y, P, dim_change={dim: 'f'}, dim_coords={'f': f}
                ) for P in Ps
            )

        return f, *Ps

    return wrapper


def _power2d_metadata(func):
    """
    Support `xarray.DataArray` for `power2d` and `copower2d`.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x1, x2, *ys = args  # *both* or *one* of these is dataarray
        y = ys[0]
        x1_in, *ys_in, kwargs = _dataarray_strip(x1, *ys, suffix='_lon', **kwargs)
        x2_in, *_, kwargs = _dataarray_strip(x2, *ys, suffix='_time', **kwargs)
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
                ) for P in Ps
            )

        return k, *Ps

    return wrapper


def _lls_metadata(func):
    """
    Generic `xarray.DataArray` wrapper for functions accepting *x* and *y*
    and returning a fit parameter, the fit standard error, and the reconstruction.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, y = args  # *both* or *one* of these is dataarray
        x_in, y_in, kwargs = _dataarray_strip(x, y, **kwargs)
        fit_val, fit_err, fit_line = func(x_in, y_in, **kwargs)

        # Build back the DataArray
        if isinstance(y, xr.DataArray):
            dim_coords = {y.dims[kwargs['axis']]: None}
            fit_val = _dataarray_from(y, fit_val, dim_coords=dim_coords)
            fit_err = _dataarray_from(y, fit_err, dim_coords=dim_coords)
            fit_line = _dataarray_from(y, fit_line)  # same everything

        return fit_val, fit_err, fit_line

    return wrapper


def _covar_metadata(func):
    """
    Support `xarray.DataArray` for `corr` and `covar` funcs.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nargs = 2 if 'auto' in func.__name__ else 3
        if len(args) == nargs - 1:
            args = 1, *args  # time step of 1
        if len(args) != nargs:
            raise TypeError(f'Expected {nargs} positional arguments. Got {len(args)}.')
        x, *ys = args  # *both* or *one* of these is dataarray
        y = ys[0]  # the first one
        x_in, *ys_in, kwargs = _dataarray_strip(x, *ys, **kwargs)
        lag, cov = func(x_in, *ys_in, **kwargs)

        # Build back the DataArray
        if isinstance(x, xr.DataArray):
            lag = _dataarray_from(x, lag, name='lag', dims=('lag',), coords={})
        if isinstance(y, xr.DataArray):
            dim = y.dims[kwargs['axis']]
            cov = _dataarray_from(y, cov, dim_change={dim: 'lag'}, dim_coords={'lag': lag})  # noqa: E501

        return lag, cov

    return wrapper


def _eof_metadata(func):
    """
    Support `xarray.DataArray` for `eof` functions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # NOTE: Here we need the time and space axis explicitly so that array coords
        # can be propertly applied below. In future might have function that permits
        # collapse of arbitrary dimensions to scalar and auto strips coords.
        data, = args
        data_in, kwargs = _dataarray_strip(data, suffix=('_time', '_space'), **kwargs)
        default = lambda param: inspect.signature(func).parameters[param].default
        axis_time = kwargs.pop('axis_time', default('axis_time'))
        axis_space = kwargs.pop('axis_space', default('axis_space'))
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


def _hist_metadata(func):
    """
    Support `xarray.DataArray` for `hist` function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # NOTE: This time 'x' coordinates are bins so do not infer
        yb, y = args
        yb_in, y_in, kwargs = _dataarray_strip(yb, y, infer_axis=False, **kwargs)
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


def _find_metadata(func):
    """
    Support `xarray.DataArray` for find function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, y = args
        x_in, y_in, kwargs = _dataarray_strip(x, y, suffix=('', '_track'), **kwargs)
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
