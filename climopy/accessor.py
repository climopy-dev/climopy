#!/usr/bin/env python3
"""
A set of `xarray accessors \
<http://xarray.pydata.org/en/stable/internals.html#extending-xarray>`__
for working with pint units and CF variables, calculating derived quantities,
and reducing dimensions in myrid ways.
"""
import functools
import itertools
import math
import numbers
import re

import cf_xarray as xcf  # noqa: F401
import numpy as np
import pint
import xarray as xr
from cf_xarray import accessor as _cf_accessor
from icecream import ic  # noqa: F401

from . import const, diff, ureg, utils, var
from .cfvariable import CFVariableRegistry, vreg
from .internals import _first_unique, _is_numeric, _is_scalar, warnings
from .unit import encode_units, latex_units, parse_units

__all__ = [
    'ClimoAccessor',
    'ClimoDataArrayAccessor',
    'ClimoDatasetAccessor',
    'register_derivation',
    'register_transformation',
]

# Add custom cell measures to cf accessor. 'Width' and 'depth' are generally for
# longitude and latitude directions. Naming conventions are to be consistent with
# existing 'cell' style names and prevent conflicts with axes names and standard names.
# NOTE: width * depth = area and width * depth * height = volume
# NOTE: height should generally be a mass-per-unit-area weighting rather than distance
CELL_MEASURES = (
    'width', 'depth', 'height', 'duration', 'area', 'volume',
)
CELL_MEASURES_BY_COORD = {
    'longitude': 'width', 'latitude': 'depth', 'vertical': 'height', 'time': 'duration'
}
if hasattr(_cf_accessor, '_CELL_MEASURES'):
    _cf_accessor._CELL_MEASURES = CELL_MEASURES
else:
    warnings._warn_climopy('cf_xarray API changed. Cannot update cell measures.')

# Expand regexes for automatic coordinate detection with standardize_coords
if hasattr(_cf_accessor, 'regex'):
    _cf_accessor.regex = {
        'time': 'time[0-9]*|lag[0-9]*|min|hour|day|week|month|year',
        'vertical': '([a-z]*lev|lv_|bottom_top|sigma|h(ei)?ght|altitude|depth|isobaric|pres|isotherm)[a-z0-9_]*',  # noqa: E501
        'latitude': 'y?lat[a-z0-9_]*',
        'longitude': 'x?lon[a-z0-9_]*',
        'X': 'xc?',
        'Y': 'yc?',
    }
    _cf_accessor.regex['Z'] = _cf_accessor.regex['vertical']
    _cf_accessor.regex['T'] = _cf_accessor.regex['time']
else:
    warnings._warn_climopy('cf_xarray API changed. Cannot update regexes.')

# Internal global variables
CFVARIABLE_ARGS = ('long_name', 'short_name', 'standard_name', 'prefix', 'suffix')
TRANSFORMATIONS = {}
DERIVATIONS = {}


def _expand_variable_args(func):
    """
    Expand single positional argument into multiple positional arguments with optional
    keyword dicts. Permits e.g. get(('t', {'lat': 'mean'})) tuple pairs.
    """
    def _wrapper(self, arg, **kwargs):
        args = []
        kwargs = kwargs.copy()
        def _iter_args(*iargs):  # noqa: E306
            for arg in iargs:
                if isinstance(arg, (tuple, list)):
                    _iter_args(*arg)
                elif isinstance(arg, str):
                    args.append(arg)
                elif isinstance(arg, dict):
                    kwargs.update(arg)
                else:
                    raise ValueError(f'Invalid variable spec {arg!r}.')
        _iter_args(arg)
        return func(self, *args, **kwargs)
    return _wrapper


def _expand_indexer(key, ndim):
    """
    Expand an indexer to a tuple with length `ndim`. Given a key for indexing an
    ndarray, return an equivalent key which is a tuple with length equal to the number
    of dimensions.  The expansion is done by replacing all `Ellipsis` items with the
    right number of full slices and then padding the key with full slices so that it
    reaches the appropriate dimensionality.
    """
    # Numpy treats non-tuple keys equivalent to tuples of length 1
    if not isinstance(key, tuple):
        key = (key,)
    new_key = []

    # Handling Ellipsis right is a little tricky, see:
    # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
    found_ellipsis = False
    for k in key:
        if k is Ellipsis:
            if not found_ellipsis:
                new_key.extend((ndim + 1 - len(key)) * [slice(None)])
                found_ellipsis = True
            else:
                new_key.append(slice(None))
        else:
            new_key.append(k)
    if len(new_key) > ndim:
        raise IndexError('Too many indices.')
    new_key.extend((ndim - len(new_key)) * [slice(None)])

    return tuple(new_key)


def _manage_reduced_coords(func):
    """
    Add back singleton NaN dummy coordinates after some dimension reduction, so that
    we can continue relating dimension names to CF axis and coordinate names, and
    identically reduce cell weights. See `add_scalar_coords` for details on motivation.
    """
    @functools.wraps(func)
    def _wrapper(self, *args, no_manage_coords=False, **kwargs):
        # Call wrapped function
        # NOTE: Existing scalar coordinates should be retained by xarray been retained
        coords = self.data.coords
        result = func(self, *args, **kwargs)
        if no_manage_coords:
            return result

        # Add back lost coords
        coords_lost = coords.keys() - result.coords.keys()
        for name in coords_lost:
            prev = coords[name]
            measure = self._to_cf_measure_name(name)

            # Replace lost dimension coordinates with scalar NaNs. Drop 1D cell
            # measure coordinates, as that information is no longer needed
            coord = None
            if prev.ndim == 1 and not measure:
                coord = xr.DataArray(np.nan, name=name, attrs=coords[name].attrs)

            # Replace lost cell measures using unweighted sum. Drop non-cell measure
            # coordinates; usually makes no sense to get an 'average' coordinate
            elif (
                prev.ndim > 1 and measure and prev.sizes.keys() & result.sizes.keys()
                and func.__name__ in ('average', 'integrate')
            ):
                coord = prev.climo.sum(*args, no_manage_coords=True, **kwargs)

            # Add weights
            if coord is not None:
                result = result.assign_coords({name: coord})

        return result

    return _wrapper


def _keep_tracked_attrs(func):
    """
    Preserve special tracked attributes for duration of function call.
    """
    @functools.wraps(func)
    def _wrapper(self, *args, no_keep_attrs=False, **kwargs):
        # Initial stuff
        # WARNING: For datasets, we use data array with longest cell_methods, to try to
        # accomodate variable derivations from source variables with identical methods
        # and ignore variables like 'bounds' with only partial cell_methods. But this
        # is ugly kludge with side effects... should be refined.
        data = self.data
        attrs = {}
        if isinstance(data, xr.Dataset):  # get longest cell_methods
            data = max(data.values(), key=lambda _: len(_.attrs.get('cell_methods') or ''))  # noqa: E501
        else:
            attrs = {k: v for k, v in data.attrs.items() if k in CFVARIABLE_ARGS}

        # Call wrapped function
        result = func(self, *args, **kwargs)
        if no_keep_attrs:
            return result

        # Build back attributes
        result.attrs.update(attrs)  # naively copied over
        for attr in ('cell_methods', 'cell_measures'):
            value = self._build_cf_attr(data.attrs.get(attr), result.attrs.get(attr))
            if value:
                result.attrs[attr] = value
        return result

    return _wrapper


def _while_quantified(func):
    """
    Wrapper that temporarily quantifies the data.
    """
    def _wrapper(self, *args, **kwargs):
        isquantity = self._is_quantity
        if not isquantity:
            self = self.quantify().climo
        if not self._is_quantity:
            raise RuntimeError('Failed to quantify data.')  # e.g. if no units attribute
        result = func(self, *args, **kwargs)
        if not isquantity:
            result = result.climo.dequantify()
        return result

    return _wrapper


def _while_dequantified(func):
    """
    Wrapper that temporarily dequantifies the data. Works with `LocIndexer` and
    `ClimoDataArrayAccessor`.
    """
    def _wrapper(self, *args, **kwargs):
        isquantity = self._is_quantity
        if isquantity:
            self = self.dequantify().climo
        result = func(self, *args, **kwargs)
        if isquantity:
            result = result.climo.quantify()
        return result

    return _wrapper


class _GroupByQuantified(object):
    """
    A unit-friendly `.groupby` indexer. Dequantify the "group" `DataArray` before use
    and preserve attributes on the resulting coordinates.

    Examples
    --------
    >>> ds = xr.tutorial.open_dataset('rasm').load()
    ... ds = ds.coarsen(x=25, y=25, boundary='trim').mean()
    ... ds.Tair.attrs['units'] = 'degC'
    ... group = ureg.kg * (ds.Tair > 0)  # arbitrary group with units
    ... group.name = 'group'
    ... ds.climo.quantify().climo.groupby(group).mean()
    """
    def __init__(self, obj, group, *args, **kwargs):
        # Infer non-data group
        if not isinstance(group, (xr.DataArray, xr.IndexVariable)):
            try:
                hash(group)
            except TypeError:
                raise TypeError(
                    'Group must be an xarray.DataArray or the '
                    'name of an xarray variable or dimension.'
                )
            group = obj[group]
            if len(group) == 0:
                raise ValueError(f'{group.name} must not be empty.')
            if group.name not in obj.coords and group.name in obj.dims:
                group = xr.core.groupby._DummyGroup(obj, group.name, group.coords)

        # Store attrs and dequantify group (group could be e.g. quantified dataset var)
        group = group.climo.dequantify()
        self._group_name = group.name  # NOTE: could be nameless
        self._group_attrs = group.attrs.copy()

        super().__init__(obj, group, *args, **kwargs)

    def _combine(self, applied, *args, **kwargs):
        # Reapply stripped group attributes to coordinates (including critical units)
        # NOTE: keep_attrs fails to preserve attributes
        res = super()._combine(applied, *args, **kwargs)
        if self._group_name in res.coords:
            coords = res.coords[self._group_name]
            coords.attrs.clear()
            coords.attrs.update(self._group_attrs)
        return res


class _DataArrayGroupByQuantified(
    _GroupByQuantified, xr.core.groupby.DataArrayGroupBy
):
    pass


class _DatasetGroupByQuantified(
    _GroupByQuantified, xr.core.groupby.DatasetGroupBy
):
    pass


class _DataArrayLocIndexerQuantified(object):
    """
    A unit-wrapped ``.loc`` indexer for `xarray.DataArray`\\ s.
    """
    def __init__(self, data_array):
        self._data = data_array

    def _expand_ellipsis(self, key):
        if not isinstance(key, dict):
            labels = _expand_indexer(key, self._data.ndim)
            key = dict(zip(self._data.dims, labels))
        return key

    def __getitem__(self, key):
        """
        Request slices optionally with pint quantity indexers.
        """
        key, _ = self._data.climo._parse_indexers(self._expand_ellipsis(key))
        key = self._data.climo._reassign_quantity_indexer(key)
        return self._data.loc[key]

    def __setitem__(self, key, value):
        """
        Request and set slices optionally with pint quantity indexers and
        pint quantity assignments.
        """
        # Standardize indexers
        # NOTE: Xarray does not support boolean loc indexing
        # See: https://github.com/pydata/xarray/issues/3546
        key, _ = self._data.climo._parse_indexers(self._expand_ellipsis(key))
        key = self._data.climo._reassign_quantity_indexer(key)

        # Standardize value
        data = self._data
        if isinstance(value, xr.DataArray):
            if value.climo._has_units and data.climo._has_units:
                value = value.climo.to_units(data.climo.units)
            value = value.data
        elif isinstance(value, pint.Quantity):
            if not data.climo._has_units:
                raise ValueError('Cannot assign pint quantities to data with unclear units.')  # noqa: E501
            value = value.to(data.climo.units)
        if isinstance(value, pint.Quantity) and not self._data.climo._is_quantity:
            value = value.magnitude  # we always apply to dequantified data
        elif not isinstance(value, pint.Quantity) and self._data.climo._is_quantity:
            value = value * self._data.data.units

        self._data.loc[key] = value


class _DatasetLocIndexerQuantified(object):
    """
    A unit-wrapped ``.loc`` indexer for `xarray.Dataset`\\ s.
    """
    def __init__(self, dataset):
        self._data = dataset

    def __getitem__(self, key):
        parsed_key = self._data.climo._reassign_quantity_indexer(key)
        return self._data.loc[parsed_key]


class _CoordsQuantified(object):
    """
    A unit-friendly `.coords` container. Returns quantified and derived
    coordinate vectors as `xarray.DataArray` objects.
    """
    def __init__(self, data, registry):
        """
        Parameters
        ----------
        data : xarray.Dataset or xarray.DataArray
            The data.
        registry : CFVariableRegistry
            The variable registry.
        """
        # NOTE: xarray currently uses _data for xarray object. We assign to _data
        # here *and* call super().__init__() in case API changes.
        self._data = data
        self._registry = registry

    def __contains__(self, key):
        return self._parse_key(key) is not None

    def __getattr__(self, attr):
        if attr[:1] == '_':
            return super().__getattribute__(attr)
        try:
            return self.__getitem__(attr)
        except KeyError:
            return super().__getattribute__(attr)

    def __getitem__(self, key):
        """
        Retrieve a quantified coordinate or derived coordinate.
        """
        tup = self._parse_key(key)
        if not tup:
            raise KeyError(f'Invalid coordinate spec {key!r}.')
        return self._make_coords(*tup)

    def _parse_key(self, key, use_registry=True):
        """
        Return the coordinates, transformation function, and flag.
        """
        # Interpret bounds specification
        m = re.match(r'\A(.*?)(_top|_bot(?:tom)?|_del(?:ta)?)?\Z', key)
        key, flag = m.groups()  # fix bounds flag
        flag = flag or ''

        # Find transformed coordinate or aliased coordinate
        # WARNING: Cannot call native items() or values() because they call
        # overridden __getitem__ internally. So recreate coordinate mapping below.
        # Also super() alone fails possibly because it returns the super() of
        # e.g. _DataArrayCoordsQuantified, which would be _CoordsQuantified.
        data = self._data
        everything = (super(_CoordsQuantified, self).__getitem__(key) for key in self)
        transformation = coord = None
        if tup := _find_any_transformation(everything, key):
            transformation, coord = tup
        elif super(_CoordsQuantified, self).__contains__(key):
            coord = super(_CoordsQuantified, self).__getitem__(key)
        elif key in data.cf.coordinates or key in data.cf.axes:
            coord = data.climo.cf[key]  # may raise error if multiple options

        # Recursively check if any aliases are valid
        elif use_registry:
            var = self._registry.get(key)
            identifiers = var.identifiers if var else ()
            for name in identifiers:
                tup = self._parse_key(name + flag, use_registry=False)
                if tup:
                    return tup

        # Return tuple
        if coord is not None:
            return coord, transformation, flag[1:]

    def _make_coords(self, coord, transformation, flag):
        """
        Return the coordinates, accounting for CF and CFVariableRegistry names.
        """
        # Select bounds
        # WARNING: Get bounds before doing transformation because halfway points in
        # actual lattice may not equal halfway points after nonlinear transformation
        dest = coord
        suffix = ''
        if flag:
            bnds = self._get_bounds(coord)
            if bnds is not None:
                if flag[:3] in ('bot', 'del'):
                    dest = bottom = bnds[..., 0]  # NOTE: scalar coord bnds could be 1D
                    suffix = ' bottom edge'
                if flag[:3] in ('top', 'del'):
                    dest = top = bnds[..., 1]
                    suffix = ' top edge'
                if flag[:3] == 'del':
                    dest = np.abs(top - bottom)  # e.g. lev --> z0, order reversed
                    suffix = ' thickness'

        # Build quantified copy of coordinate array and take transformation
        # NOTE: coord.copy() and coord.copy(data=data) both fail, have to make
        # new data array from scratch to prevent unit stripping. Perhaps there
        # is some way to check if DataArray is pandas Index. See:
        # https://github.com/pydata/xarray/issues/525#issuecomment-514452182
        dest = xr.DataArray(
            dest.data.copy(),
            dims=coord.dims,
            name=coord.name,
            coords=coord.coords,
            attrs=coord.attrs.copy(),  # also copies over units
        )
        dest = dest.climo.quantify()
        if transformation:
            dest = transformation(dest)  # will also adjust name

        # Return coords with cleaned up attributes. Only long_name and standard_name
        # are kept if math was performed.
        flag = '_' + flag if flag else ''
        dest.name += flag
        if long_name := coord.attrs.get('long_name'):
            dest.attrs['long_name'] = long_name + suffix
        if standard_name := coord.attrs.get('standard_name'):
            dest.attrs['standard_name'] = standard_name

        return dest

    def _get_bounds(self, coord):
        """
        Return bounds inferred from the coordinates or generated on-the-fly.
        """
        # Retrieve actual bounds from dataset
        # WARNING: When reading/writing datasets with default decode_cf=True behavior,
        # xarray silently strips attributes from variables whose name correspond to
        # 'bounds' of coordinate variable, in accordance with CF conventions.
        # See: http://xarray.pydata.org/en/stable/whats-new.html#id283
        # See: http://cfconventions.org/cf-conventions/cf-conventions#cell-boundaries
        data = self._data
        bnds = None
        bounds = coord.attrs.get('bounds', None)
        if bounds and isinstance(data, xr.Dataset):
            try:
                bnds = data[bounds]
            except KeyError:
                warnings._warn_climopy(
                    f'Coordinate {coord.name!r} bounds variable {bounds!r} '
                    f'missing from dataset with variables {tuple(data)}.'
                )
        if bnds is not None:
            if bnds.ndim != 2 or 2 not in bnds.shape:
                raise RuntimeError(
                    f'Expected bounds variable {bnds.name!r} to be 2-dimensional '
                    f'and have a length-2 bounds dimension. Got shape {bnds.shape!r}.'
                )
            if bnds.climo._is_quantity and bnds.climo.units != coord.climo.units:
                warnings._warn_climopy(
                    f'Replacing coordinate bounds units {bnds.climo.units} '
                    f'with coordinate units {coord.climo.units}.'
                )
            bnds.data = bnds.climo.magnitude  # strip units if they exist
            bnds.attrs.clear()
            bnds.attrs.update(coord.attrs)  # add back attrs stripped by xarray
            bdim = bnds.dims[bnds.shape.index(2)]  # bounds dimension name
            return bnds.transpose(..., bdim)  # put bounds dimension on final axis

        # Special consideration for singleton longitude, latitude, and height
        # dimensions! Consider 'bounds' to be entire domain.
        if not _is_numeric(coord):
            # Bail out for string and datetime coords
            warnings._warn_climopy(
                f'Cannot make coordinate bounds for non-numeric data {coord.name!r}.'
            )
            return
        elif coord.size == 1:
            if not coord.isnull():
                raise RuntimeError(
                    f'Cannot infer bounds for singleton non-NaN coord {coord!r}.'
                )
            vertical = data.climo.vertical_type
            cfcoords = data.cf.coordinates
            if coord.name in cfcoords.get('longitude', ()):
                bounds = [-180.0, 180.0] * ureg.deg
            elif coord.name in cfcoords.get('latitude', ()):
                bounds = [-90.0, 90.0] * ureg.deg
            elif coord.name in cfcoords.get('vertical', ()) and vertical == 'pressure':
                bounds = [0.0, 1013.25] * ureg.hPa
            else:
                raise RuntimeError(
                    f'Cannot infer bounds for singleton NaN coord {coord!r}. Must '
                    'be a longitude, latitude, or vertical pressure dimension.'
                )
            bounds = bounds.to(coord.climo.units).magnitude
            if not _is_scalar(coord):
                bounds = bounds[None, :]

        # Infer distances from *surface* up to TOA, padding the ends with
        # NOTE: All dimensions are assumed monotonically increasing
        else:
            edges = np.concatenate(
                (
                    coord.data[:1] - 0.5 * np.diff(coord.data[:2]),
                    0.5 * (coord.data[1:] + coord.data[:-1]),
                    coord.data[-1:] + 0.5 * np.diff(coord.data[-2:]),
                )
            )
            bounds = np.hstack((edges[:-1, None], edges[1:, None]))

        # Fix boundary conditions at meridional domain edge
        # NOTE: Includes kludge where we ignore data from other hemisphere if we
        # have hemispheric data with single latitude from other hemisphere.
        if coord.name in self._data.climo.cf.coordinates['latitude']:
            bnd_lo = 1e-10
            bnd_hi = 90
            bounds[bounds < -bnd_hi] = -bnd_hi
            bounds[bounds > bnd_hi] = bnd_hi
            if bounds[bounds < -bnd_lo].size == 1:
                bounds[bounds < -bnd_lo] = -bnd_lo
            if bounds[bounds > bnd_lo].size == 1:
                bounds[bounds > bnd_lo] = bnd_lo

        # Return new DataArray
        bounds = xr.DataArray(
            bounds,
            name=coord.name + '_bnds',
            dims=(*coord.dims[:1], 'bnds'),  # nameless 'bnds' dimension
            coords=coord.coords,
        )

        return bounds

    def get(self, key, default=None):
        """
        Return the coordinate if it is present, otherwise return a default value.
        """
        tup = self._parse_key(key)
        if tup is None:
            return default
        else:
            return self._make_coords(*tup)


class _DataArrayCoordsQuantified(
    _CoordsQuantified, xr.core.coordinates.DataArrayCoordinates
):
    pass


class _DatasetCoordsQuantified(
    _CoordsQuantified, xr.core.coordinates.DatasetCoordinates
):
    pass


class _VarsQuantified(object):
    """
    A data array container. Returns quantified variables, mirroring behavior
    of `_CoordsQuantified`.
    """
    def __init__(self, dataset, registry):
        """
        Parameters
        ----------
        dataset : xarray.Dataset
            The data.
        registry : CFVariableRegistry
            The variable registry.
        """
        self._data = dataset
        self._registry = registry

    def __contains__(self, key):
        return self._get_item(key) is not None

    def __getattr__(self, attr):
        if attr[:1] == '_':
            return super().__getattribute__(attr)
        try:
            return self.__getitem__(attr)
        except KeyError:
            return super().__getattribute__(attr)

    def __getitem__(self, key):
        """
        Return a quantified variable.
        """
        da = self._get_item(key)
        if da is None:
            raise KeyError(f'Invalid variable name {key!r}.')
        return da.climo.quantify()

    def _get_item(self, key, use_registry=True):
        """
        Return a function that generates the variable, accounting for CF and
        CFVariableRegistry names.
        """
        # NOTE: Compare with _CoordsQuantified._get_item and ClimoDatasetAccessor
        data = self._data
        if key in data:
            return data[key]
        if key in data.cf.standard_names or key in data.cf.cell_measures:  # e.g. 'area'
            return data.climo.cf[key]
        # Try to locate using identifier synonyms
        if use_registry:
            var = self._registry.get(key)
            identifiers = var.identifiers if var else ()
            for name in identifiers:
                da = self._get_item(name, use_registry=False)
                if da is not None:
                    return da

    def get(self, key, default=None):
        """
        Return the variable if it is present, otherwise return a default value.
        """
        da = self._get_item(key)
        if da is not None:
            return da.climo.quantify()
        else:
            return default


class _CFDataArrayAccessor(_cf_accessor.CFDataArrayAccessor):
    """
    Accessor that ignores annoying CF warnings when bounds variables are missing from
    data. See `related xarray issue <https://github.com/pydata/xarray/issues/1475>`_.
    """
    def __getitem__(self, key):
        import warnings
        # TODO: Update this once stable API changes
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            return super().__getitem__(key)


class ClimoAccessor(object):
    """
    Accessor with properties and methods shared by `DataArray`s and `Dataset`s

    Warning
    -------
    This adds unit support for the operations `.loc`, `.sel`, `.interp`, and `.groupby`.
    Otherwise, `.weighted` and `.coarsen` already work, but `.resample` and `.rolling`
    are broken and may be quite tricky to fix.
    """
    def _build_cf_attr(self, old, new):
        """
        Merge and encode parts into CF `cell_methods`-like attribute.
        """
        seen = set()
        parts = [
            ((dims,) if isinstance(dims, str) else tuple(dims), value)
            for parts in (old, new)
            for dims, value in self._decode_cf_attr(parts)  # decode if string
            if dims  # drop empty tuples (could happen in edge cases?)
        ]
        return ' '.join(
            ': '.join(dims) + ': ' + value for dims, value in parts
            if (dims, value) not in seen and not seen.add((dims, value))
        ).strip()

    def _cf_attr_values_by_key(self, attr, key):
        """
        Get the values corresponding to the input key, e.g. all ``'lon'`` cell methods.
        """
        attr = self.attrs.get(attr, '')
        for dims, value in self._decode_cf_attr(attr):
            if key in dims:
                yield value

    def _decode_cf_attr(self, attr):
        """
        Expand CF `cell_methods`-like attribute into parts.
        """
        attr = attr or ''
        if not isinstance(attr, str):  # already decoded
            return attr
        attr = attr.strip()
        if not attr:
            return []
        starts = tuple(m.start() for m in re.finditer(r'(\w+:\s+)+', attr))
        if not starts or starts[0] != 0:
            raise ValueError(f'Invalid CF-style attribute {attr!r}.')
        parts = []  # do not use dict so we can have duplicate 'keys'
        for start, end in zip(starts, starts[1:] + (None,)):
            substring = attr[start:end].strip()
            m = re.match(r'\s*(?:\w+:\s*)+(.*)', substring)
            if not m:
                raise ValueError(f'Invalid CF-style attribute {attr!r}.')
            idx = m.start(1)  # start of *value*
            dims = tuple(dim.strip() for dim in substring[:idx].split(':'))
            dims = tuple(filter(None, dims))
            value = substring[idx:].strip()
            parts.append((dims, value))
        return parts

    def _iter_data_arrays(self):
        """
        Iterate over DataArray(s). If this is a DataArray itself just yield it.
        """
        data = self.data
        if isinstance(data, xr.DataArray):
            yield data
        else:
            yield from data.values()

    def _iter_by_indexer_coords(self, func, indexers, **kwargs):
        """
        Apply function `func` (currently `.sel` or `.interp`) to each scalar value
        in the indexers, then merge along the indexer coordinate axes. This supports
        arbitrary ND multiple selections. Example usage: Selecting the storm track
        latitude as a function of some parametric or ensemble member axis.
        """
        # Iterate over non-scalar indexer coordinates
        # NOTE: Input indexers should alredy have been standardized and translated
        # by _reassign_quantity_indexer, e.g. 'latitude' --> 'lat'
        # NOTE: If coordinates are present on indexers, they must match! For example:
        # lat=xr.DataArray([30, 60, 90], coords={'dummy': [10, 20, 30]})
        # lev=xr.DataArray([250, 500, 750], coords={'dummy': [10, 20, 30]})
        indexers_fancy = {k: v for k, v in indexers.items() if isinstance(v, xr.DataArray)}  # noqa: E501
        indexers = {k: indexers[k] for k in indexers.keys() - indexers_fancy.keys()}
        datas = np.empty((1,), dtype='O')
        dims = ()
        if indexers_fancy:
            sample = tuple(indexers_fancy.values())[0]
            if any(key in da.dims for key, da in indexers_fancy.items()):
                raise ValueError(
                    'Coordinates on DataArray indexers should not match any of the '
                    'coordinates they are indexing.'
                )
            if any(da.sizes != sample.sizes for da in indexers_fancy.values()):
                raise ValueError(  # NOTE: this check is incomplete
                    'Dimensionality of DataArray indexers must be identical '
                    'to one another.'
                )
            dims, coords = sample.dims, sample.coords
            if dims:
                datas = np.empty(sample.shape, dtype='O')

        # Make selections or interpolations
        data = self.data
        for idx in np.ndindex(datas.shape):  # ignore 'startstop' dimension
            isel = {k: v for k, v in zip(dims, idx)}
            idata = data.isel(isel, drop=True)
            indexer = {
                **{k: v.isel(isel, drop=True) for k, v in indexers_fancy.items()},
                **indexers,
            }
            datas[idx] = getattr(idata, func)(indexer, **kwargs)

        # Merge along indexer coordinates, and return to original permution order
        if indexers_fancy:
            data = xr.combine_nested(
                datas.tolist(),
                concat_dim=dims,
                join='exact',
                compat='identical',
                combine_attrs='identical',
            )
            data = data.climo.replace_coords(coords)
        else:
            data = datas[0]

        return data

    def _parse_dims(self, dim=None, single=False, **kwargs):
        """
        Parse positional dimension indexers. Defer to _parse_indexers for algorithm.
        """
        dims = dim or ()
        if isinstance(dims, str):
            dims = (dims,)
        kwargs.setdefault('allow_kwargs', False)
        indexers = {dim: None for dim in dims}
        indexers, _ = self._parse_indexers(indexers, **kwargs)
        if not single:
            return tuple(indexers) or None
        elif len(indexers) != 1:
            raise ValueError(f'Expected one dimension, got {len(indexers)}.')
        else:
            return tuple(indexers)[0]

    def _parse_indexers(
        self,
        indexers=None,
        allow_kwargs=True,
        ignore_scalar=False,
        include_scalar=False,
        include_pseudo=False,
        include_transforms=False,
        **kwargs
    ):
        """
        Parse and translate keyword dimension indexers.
        """
        # NOTE: Currently this only supports dimensions with attached coordinates!
        # Pretty sure every function that invokes this requires coordinate info
        dims = self.data.dims
        coords = self.data.coords
        filtered = {}
        indexers = indexers or {}
        kwargs = {**indexers, **kwargs}
        for key in tuple(kwargs):
            dim = key
            if dim in dims and dim not in coords:  # but not coordinates
                raise RuntimeError(f'Dimension {key!r} is missing coordinate data.')
            try:
                dim = self._to_native_name(dim)
            except KeyError:
                pass
            if dim in coords and coords[dim].size == 1:
                if ignore_scalar:  # used for .sum() and .mean()
                    del kwargs[key]
                    continue
                if not include_scalar:
                    raise RuntimeError(f'Coordinate {key!r} is scalar.')
            if (
                dim in coords
                or include_pseudo and dim in ('area', 'volume')  # e.g. integral('area')
                or include_transforms and _find_any_transformation(coords.values(), dim)
            ):
                filtered[dim] = kwargs.pop(key)
            elif not allow_kwargs:
                raise ValueError(f'Invalid argument or unknown dimension {key!r}.')

        return filtered, kwargs

    def _parse_truncate_args(self, **kwargs):
        """
        Parse arguments used to truncate data. Returns tuple of dictionaries used
        to limit data range. Used by both `.truncate` and `.reduce`.
        """
        # Limit range of dimension reduction
        # NOTE: This permits *multiple* bounds that get reduced to 'track' dimension,
        # either explicitly with e.g. latmin=(0, 30) or when parameter bounds like
        # latmin='ehf_lat' returns more than one value.
        data = self.data
        dims = []
        startstops = []
        for key in tuple(kwargs):
            # Interpret dimension and bounds
            # WARNING: Below precludes us from using _[min|max|lim] suffix for other
            # keyword args. Might reconsider but we use "special" suffixes and prefixes
            # everywhere (e.g. _[lat|strength]) so this is consistent with API.
            m = re.match(r'\A(.*?)_(min|max|lim|)\Z', key)
            if not m:
                continue
            if key not in kwargs:  # happens with e.g. latitude_min=x latitude_max=y
                continue
            dim, mode = m.groups()
            try:
                dim = self._to_native_name(dim)
            except KeyError:
                raise TypeError(f'Invalid truncation arg {key!r}.')
            units = data.coords[dim].climo.units

            # Get start and stop locations
            # Handle passing e.g. latmin=x latmax=y instead of latlim=z
            loc = kwargs.pop(key)
            if mode == 'max':
                start = kwargs.pop(dim + 'min', None)
                stop = loc
            elif mode == 'min':
                start = loc
                stop = kwargs.pop(dim + 'max', None)
            else:
                start, stop = loc

            # Get 'variable-spec' bounds and translate units
            # Then add to the list of starts and stops
            dims.append(dim)
            for bound, mode in zip((start, stop), ('min', 'max')):
                # Translate 'parameter' bounds
                if isinstance(bound, (str, tuple)):  # 'name' or ('name', {})
                    if not isinstance(data, xr.Dataset):
                        raise ValueError('Dataset required to get bounds {bound!r}.')
                    bound = data.climo.get(bound)  # may add a 'track' dimension
                else:
                    if bound is None:
                        bound = getattr(data.climo.coords[dim].climo.magnitude, mode)()
                    bound = np.atleast_1d(bound)
                    if bound.ndim > 1:
                        raise ValueError('Too many dimensions for bounds {bound!r}.')
                    bound = xr.DataArray(bound, dims='track')
                # Handle units
                if not isinstance(bound.data, pint.Quantity):
                    bound.data = bound.data * units
                else:
                    bound = bound.climo.to_units(units)
                bound = bound.climo.dequantify()
                if 'track' not in bound.dims:
                    bound = bound.expand_dims('track')
                startstops.append(bound)

        # Match dimensionality between 'start' and 'stop' bounds for all dim names
        # Example: a (5) x b (4) x track (3) 'starts' and b (4) x track (2) 'stops'
        # Below if we pass list(zip(*list(itertools.product(range(2), range(3)))))
        # get idx series for each DataArray: [(0, 0, 0, 1, 1, 1), (0, 1, 2, 0, 1, 2)]
        idxs = zip(*itertools.product(*(range(_.sizes['track']) for _ in startstops)))
        startstops = tuple(_.isel(track=list(idx)) for idx, _ in zip(idxs, startstops))
        startstops = xr.broadcast(*startstops)  # match dimensionality!

        # Create bounds dictionary
        # NOTE: The zerofind 'track' dims have no coordinates
        # NOTE: Xarray concat() does automatic dimension broadcasting so we
        # just need to get 'outer' combination of all possible start/stop tracks
        # NOTE: Remove coordinates from bounds specifications to prevent weird
        # errors when applying coords in reduce(). The bounds specifications
        # *themselves* are supposed to be coordinates.
        bounds = {
            dim: xr.concat(
                (start, stop),
                dim='startstop',
                compat='no_conflicts',
                combine_attrs='no_conflicts'
            ).reset_coords(drop=True)
            for dim, start, stop in zip(dims, startstops[::2], startstops[1::2])
        }
        return bounds, kwargs

    def _reassign_quantity_indexer(self, indexers):
        """
        Reassign a `pint.Quantity` indexer to units of relevant coordinate.
        """
        def _to_magnitude(value, units, scalar=True):
            if isinstance(value, xr.DataArray):
                if value.climo._has_units and value.dtype.kind != 'b':
                    value = value.climo.to_units(units)
                value = value.climo.dequantify().data
            elif isinstance(value, pint.Quantity):
                value = value.to(units).magnitude
            if np.atleast_1d(value).size == 1:
                value = np.atleast_1d(value)[0]  # convert to scalar index
            elif scalar:
                raise ValueError(f'Expected scalar indexer, got {value=}.')
            return value

        # Update indexers to handle quantities and slices of quantities
        data = self.data
        indexers_scaled = {}
        for name, sel in indexers.items():
            if name in data.climo.coords:
                units = data.climo.coords[name].climo.units
            else:
                units = None
            if isinstance(sel, slice):
                start = _to_magnitude(sel.start, units)
                stop = _to_magnitude(sel.stop, units)
                step = _to_magnitude(sel.step, units)
                indexers_scaled[name] = slice(start, stop, step)
            else:
                indexers_scaled[name] = _to_magnitude(sel, units, scalar=False)

        return indexers_scaled

    def _to_native_name(self, key, use_cf=True, use_registry=True):
        """
        Translate input name into variable name, with support for active CF
        identifiers and the variable registry.
        """
        # NOTE: This should *really* be a dedicated function in cf_xarray
        if not isinstance(key, str):
            raise KeyError('Key must be string.')
        data = self.data
        if key in data.coords or isinstance(data, xr.Dataset) and key in data:
            return key
        if use_cf and key in self.cf:
            return self.cf[key].name  # may raise error if spec is ambiguous!
        if use_registry:
            var = self.registry.get(key)
            identifiers = var.identifiers if var else ()
            for name in identifiers:
                if name in data.coords or isinstance(data, xr.Dataset) and name in data:
                    return name
        raise KeyError(f'Invalid variable {key!r}.')

    def _to_cf_measure_name(self, key):
        """
        Translate input variable name into cell mesure name. Return None if not found.
        """
        try:
            key = self._to_native_name(key)
        except KeyError:
            pass
        for measure, names in self.cf.cell_measures.items():
            if key in names:
                return measure

    def _to_cf_coord_name(self, key):
        """
        Translate input variable name into coordinate name. Return None if not found.
        """
        try:
            key = self._to_native_name(key)
        except KeyError:
            pass
        for coord, names in self.cf.coordinates.items():
            if key in names:
                return coord

    def add_boundaries(self, latitude=True, vertical=False, zero=None):
        """
        Add latitude coordinates for north and south poles and pressure coordinates
        for the mean sea-level pressure and zero pressure levels. This ensures plots
        of latitude-pressure cross-sections have data coverage over the whole atmosphere
        and improves the accuracy of budget term calculations.

        Parameters
        ----------
        latitude, vertical : bool, optional
            Indicates whether to enforce global conditions on the latitude and
            vertical coordinates.
        zero : bool or list of str, optional
            If this is a `DataArray` accessor, should be boolean indicating whether
            data should be zeroed out on the poles (i.e. wind and extensive properties).
            If this is a `Dataset` accessor, should be list of variables that
            should be zeroed out on the poles.
        """
        # Add latitude coordinates at poles
        data = self.data
        if latitude:
            coords = data.climo.coords['latitude']
            lat = coords.name
            parts = []
            if np.min(coords) < -80 * ureg.deg and -90 * ureg.deg not in coords:
                part = data.isel({lat: slice(0, 1)})
                part = part.climo.replace_coords({lat: [-90] * ureg.deg})
                parts.append(part)
            parts.append(data)
            if np.max(coords) > 80 * ureg.deg and 90 * ureg.deg not in coords:
                part = data.isel({lat: slice(-1, None)})
                part = part.climo.replace_coords({lat: [90] * ureg.deg})
                parts.append(part)
            data = xr.concat(
                parts,
                dim=lat,
                data_vars='minimal',
                compat='no_conflicts',
                combine_attrs='no_conflicts'
            )

        # Add pressure coordinates at surface and "top of atmosphere"
        # WARNING: This part is untested and has questionable value
        if vertical:
            coords = data.climo.coords['vertical']
            if not coords.climo.units.is_compatible_with('Pa'):
                vertical = False
        if vertical:
            lev = coords.name
            parts = []
            if 0 * ureg.hPa not in coords:
                part = data.isel({lev: slice(0, 1)})
                part = part.climo.replace_coords({lev: [0] * ureg.hPa})
                parts.append(part)
            parts.append(data)
            if 1013.25 * ureg.hPa not in coords:
                part = data.isel({lev: slice(-1, None)})
                part = part.climo.replace_coords({lev: [1013.25] * ureg.hPa})
                parts.append(part)
            data = xr.concat(
                parts,
                dim=lev,
                data_vars='minimal',
                compat='no_conflicts',
                combine_attrs='no_conflicts',
            )
            if isinstance(data, xr.Dataset) and 'bounds' in coords.attrs:
                bnds = data[coords.attrs['bounds']]
                bnds[-2, 1] = bnds[-1, 0] = bnds[-1, :].mean()
                bnds[0, 1] = bnds[1, 0] = bnds[0, :].mean()

        # Repair values at polar singularity
        # NOTE: Xarray does not support boolean loc indexing
        # See: https://github.com/pydata/xarray/issues/3546
        if latitude:
            if isinstance(data, xr.DataArray):
                zero = (data.name,) if zero else ()
            else:
                zero = zero or ()
            for da in data.climo._iter_data_arrays():
                if da.name in zero and lat in da.coords:
                    coord = da.climo.coords[lat]
                    loc = coord[np.abs(coord) == 90 * ureg.deg]
                    da.climo.loc[{lat: loc}] = 0

        return data

    def add_cell_methods(self, methods=None, **kwargs):
        """
        Update the `cell_methods` attribute on the `xarray.DataArray` or on every array
        in the `xarray.Dataset` with the input methods.

        Parameters
        ----------
        methods : dict-like, optional
            A cell methods dictionary, whose keys are dimension names. To associate
            multiple dimensions with a single method, use tuples of dimension names
            as dictionary keys.
        **kwargs
            Cell methods passed as keyword args.
        """
        attr = 'cell_methods'
        methods = methods or {}
        methods.update(kwargs)
        for da in self._iter_data_arrays():
            da.attrs[attr] = self._build_cf_attr(da.attrs.get(attr), methods.items())

    def add_cell_measures(
        self, measures=None, *, dataset=None, overwrite=False, **kwargs
    ):
        """
        Update the `cell_measures` attribute on the `xarray.DataArray` or on every array
        in the `xarray.Dataset` with the input measures and add them to the coordinates.

        Parameters
        ----------
        measures : dict-like, optional
            Dictionary of cell measures. If none are provided, the default `width`,
            `depth`, and `height` measures are automatically calculated. If this is a
            DataArray, surface pressure will not be taken into account and isentropic
            grids will error out.
        dataset : xarray.Dataset, optional
            The dataset associated with this `xarray.DataArray`. Needed when
            calculating cell measures automatically.
        overwrite : bool, optional
            Whether to overwrite existing cell measures. Default is ``False``.
        **kwargs
            Cell measures passed as keyword args.
        """
        data = self.data.copy(deep=False)
        action = 'default'
        measures = measures or {}
        measures.update(kwargs)
        if isinstance(data, xr.Dataset):
            dataset = data
        elif dataset is None:
            # Get weights from temporary dataset. Ignore CF UserWarning and
            # ClimoPyWarnings due to missing bounds
            dataset = data.to_dataset(name=data.name or 'unknown')
            action = 'ignore'

        # Get default cell measures
        if not measures:
            import warnings
            for measure in ('width', 'depth', 'height', 'duration'):
                if (cell := 'cell_' + measure) in data.coords and not overwrite:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter(action)  # possibly ignore warnings
                    try:
                        weight = dataset.climo._get_item(cell, add_cell_measures=False)
                    except (KeyError, RuntimeError):
                        continue
                if weight.sizes.keys() - data.sizes.keys():  # dims are not subset
                    continue
                weight.name = cell  # just in case
                measures[measure] = weight

        # Add measures as dequantified coordinate variables
        # NOTE: This approach is used as an example in cf_xarray docs:
        # https://cf-xarray.readthedocs.io/en/latest/examples/introduction.html#Feature:-Weight-by-Cell-Measures
        for measure, da in measures.items():
            if not isinstance(da, xr.DataArray):
                raise ValueError('Input cell measures must be DataArrays.')
            data.coords[da.name] = da.climo.dequantify()
            objs = [data]
            attr = 'cell_measures'
            if isinstance(data, xr.Dataset):
                objs.extend(data.climo._iter_data_arrays())
            for obj in objs:
                attr = 'cell_measures'
                obj.attrs[attr] = self._build_cf_attr(
                    obj.attrs.get(attr), ((measure, da.name),)
                )

        return data

    def add_scalar_coords(self):
        """
        Add dummy scalar coordinates for missing longitude, latitude, and vertical
        dimensions. See final paragraph of CF manual section 7.3.2:

            A dimension of size one may be the result of "collapsing" an axis by some
            statistical operation, for instance by calculating a variance from time
            series data. We strongly recommend that dimensions of size one be retained
            (or scalar coordinate variables be defined) to enable documentation of the
            method (through the cell_methods attribute) and its domain (through the
            bounds attribute).

        In accordance with the CF standards, this function adds scalar coordinates and
        updates data variable `cell_methods` to indicate that 0D coordinate "cells" were
        reduced from their original 1D sampling by averaging over the domain (i.e. all
        longitudes or latitudes or the full column height). This is analogous to the
        more standard use case where `cell_methods` indicates how "cells" in 1D
        coordinate arrays were constructed from a finer original sampling interval.
        """
        data = self.data.copy(deep=False)
        coords = ('longitude', 'latitude', 'vertical')
        attrs = {  # default variable names and attributes if dimension not present
            'lon': {'axis': 'X', 'standard_name': 'longitude', 'units': 'degrees_east'},
            'lat': {'axis': 'Y', 'standard_name': 'latitude', 'units': 'degrees_north'},
            'lev': {'axis': 'Z', 'standard_name': 'air_pressure', 'positive': 'down', 'units': 'hPa'},  # noqa: E501
        }
        if data.cf.sizes.get('time', None) == 1:  # time dimension exists
            data = data.cf.squeeze('time')  # may remove time coordinate
        for dim, coord in zip(attrs, coords):
            if coord not in data.climo.cf:
                if dim in data.dims:
                    if data.sizes[dim] == 1:  # exists as singleton dim but not coord
                        data = data.squeeze(dim)
                    else:
                        raise ValueError('Dimension already exists without coords.')
                data.coords[dim] = ((), np.nan, attrs[dim])
            elif data.climo.cf[coord].size == 1:
                dim = data.climo._to_native_name(coord)  # coord alreday shown to exist
                if dim in data.sizes:  # i.e. is a dimension
                    data = data.squeeze(dim)
                data = data.climo.replace_coords({dim: np.nan})
                for key, value in attrs.items():
                    data.attrs.setdefault(key, value)
            else:
                continue
            data.climo.add_cell_methods({dim: 'mean'})

        return data

    def groupby(self, group, *args, **kwargs):
        """
        Wrap `xarray.DataArray.groupby` to support pint quantity groups.

        Parameters
        ----------
        *args, **kwargs
            Passed to `.groupby`.
        """
        return self._cls_groupby(self.data, group, *args, **kwargs)

    @_while_dequantified
    def interp(self, indexers=None, method='linear', assume_sorted=False, **kwargs):
        """
        Wrap `.interp` to handle units, preserve coordinate attributes, units, and
        perform extrapolation for out-of-range coordinates by default. Also permit
        interpolating to different points as a function of other coordinates.

        Parameters
        ----------
        *args, **kwargs
            Passed to `.interp`.
        """
        indexers = indexers or {}
        indexers.update(kwargs)
        indexers, _ = self._parse_indexers(indexers)
        indexers = self._reassign_quantity_indexer(indexers)
        return self._iter_by_indexer_coords(
            'interp', indexers, method=method, assume_sorted=assume_sorted,
            kwargs={'fill_value': 'extrapolate'},
        )

    def replace_coords(self, indexers=None, **kwargs):
        """
        Return a copy with replaced coordinate values and preserved attributes. If
        input coordinates are already `DataArray`s, its existing attributes are not
        overwritten. Inspired by `xarray.DataArray.assign_coords`.

        Parameters
        ----------
        indexers : dict-like, optional
            The new coordinates.
        **kwargs
            Coordinates passed as keyword args.
        """
        indexers, _ = self._parse_indexers(
            indexers, allow_kwargs=False, include_scalar=True, **kwargs
        )
        indexers_new = {}
        for name, coord in indexers.items():
            if name not in self.data.coords:
                raise ValueError(f'Coordinate {name!r} not found.')
            if isinstance(coord, tuple):
                raise ValueError('Coordinate data must be array-like.')
            prev = self.data.coords[name]
            if isinstance(coord, xr.DataArray):
                for key, value in prev.attrs.items():
                    coord.attrs.setdefault(key, value)
            else:
                # WARNING: containers of scalar quantities like [90 * ureg.deg]
                # silently have units stripped and are transformed to 1.
                dims = () if _is_scalar(coord) else (name,)
                coord = xr.DataArray(coord, dims=dims, name=name, attrs=prev.attrs)
            if coord.climo._has_units:
                coord = coord.climo.to_units(prev.climo.units)  # units support
                coord = coord.climo.dequantify()
                coord.attrs['units'] = prev.attrs['units']  # ensure exact match
            indexers_new[name] = coord
        return self.data.assign_coords(indexers_new)

    def sel(self, indexers=None, method=None, tolerance=None, drop=False, **kwargs):
        """
        Wrap `.sel` to handle units. Also permit selecting different points as
        a function of other coordinates.

        Parameters
        ----------
        *args, **kwargs
            Passed to `.sel`.
        """
        indexers = indexers or {}
        indexers.update(kwargs)
        indexers, _ = self._parse_indexers(indexers)
        indexers = self._reassign_quantity_indexer(indexers)
        return self._iter_by_indexer_coords(
            'sel', indexers, method=method, tolerance=tolerance, drop=drop
        )

    def sel_hemisphere(self, which, invert=None):
        """
        Select a hemisphere or average the hemispheres. A single negative latitude
        is included across the equator and data points are added at the poles so that
        contours, lines, and whatnot encompass the entire 0-90 degree range.

        Parameters
        ----------
        which : {'globe', 'inverse', 'ave', 'nh', 'sh'}
            The "hemisphere". May be the globe, the globe with hemispheres flipped, the
            average of both hemispheres, or either of the northern and southern.
        invert : bool or list of str, optional
            If this is a `DataArray` accessor, should be boolean indicating whether
            data should be inverted when averaging northern and southern hemispheres.
            If this is a `Dataset` accessor, should be list of variables that
            should be inverted.
        """
        # Bail out if already is single hemisphere
        data = self.data
        lat = self.cf['latitude'].values
        if np.all(np.sign(lat) == np.sign(lat[0])):
            return data

        # Change the "positive" meridional direction for all variables in the dataset.
        # NOTE: PV is -ve in SH and +ve in NH so flux does not need adjustment.
        def _invert_hemi(data):  # noqa: E306
            data = data.copy(deep=False)
            rlat = -1 * data.cf.coords['latitude']
            data = data.climo.replace_coords(lat=rlat)
            if isinstance(data, xr.DataArray):
                if invert:
                    data.data *= -1
            else:
                for name, da in data.items():
                    if name in invert:
                        da.data *= -1
            return data

        # Select region (requires temporarily removing "bnd" variables)
        # WARNING: keep_attrs fails to preserve dataset attributes for 'ave' (bug)
        attrs = data.attrs.copy()
        nhmin = np.min(lat[lat > 0])
        shmax = np.max(lat[lat < 0])
        with xr.set_options(keep_attrs=True):
            which = which.lower()
            if which == 'globe':
                pass
            elif which == 'inverse':  # global, but SH on top
                data = _invert_hemi(data)
            elif which == 'nh':
                data = data.cf.sel({'latitude': slice(shmax, 90)})
            elif which == 'sh':
                data = _invert_hemi(data.cf.sel({'latitude': slice(nhmin, -90, -1)}))
            elif which == 'ave':
                data_nh = data.cf.sel({'latitude': slice(shmax, 90)})
                data_sh = _invert_hemi(data.cf.sel({'latitude': slice(nhmin, -90, -1)}))
                data = 0.5 * (data_nh + data_sh)
            else:
                raise ValueError(f'Unknown hemisphere identifier {which!r}.')
        data.attrs.update(attrs)
        return data

    def sel_pair(self, key, drop=True):
        """
        Return selection from pseudo "pair" or "parameter" axis. This searches for
        "parameter" axes as any axis whose associated coordinate `CFVariable` has a
        "reference" attribute.

        Parameters
        ----------
        key : str, optional
            The pair key. If the parameter dimension is length 2, the key ``1`` returns
            the first position and the key ``2`` the second position. Otherwise, the
            key ``1`` returns the 'base' position and the key ``2`` is a no-op that
            returns the original data. To return the difference between keys
            ``2`` and ``1``, pass ``'anom'``. The associated `CFVariable` names are
            modified by adding `prefix` and `suffix` attributes to the data that get
            passed to the `CFVariable.update` method.
        drop : bool, optional
            Whether to drop the coordinates when making selections or keep them
            as scalar values. Default is ``True``.
        """
        data = self.data
        key = str(key)
        if key not in ('1', '2', 'anom'):
            raise ValueError(f'Invalid pair spec {key!r}.')

        # Determine pseudo 'pair' axis
        dims_base = {
            dim: coord.climo.base for dim, coord in data.coords.items()
            if dim in data.dims and getattr(coord.climo, 'base') is not None
        }
        dims = tuple(dim for dim in dims_base if data.sizes[dim] == 2)
        if dims:
            if len(dims) > 1:
                warnings._warn_climopy(f'Ambiguous anomaly pair dimensions {dims}. Using first one.')  # noqa: E501
            sels = tuple({dims[0]: data.coords[dims[0]].values[i]} for i in range(2))
        elif dims_base:
            if len(dims_base) > 1:
                warnings._warn_climopy(f'Ambiguous parameter dimensions {tuple(dims_base)}. Using first one.')  # noqa: E501
            sels = (dims_base, {})
        else:
            raise ValueError('No parameter dimensions found.')

        # Make selection and repair cfvariable
        prefix = suffix = None
        modify = isinstance(data, xr.DataArray) and not re.search('(force|forcing)', data.name or '')  # noqa: E501
        if key == '1':
            prefix = 'unforced' if dims and modify else None
            data = data.sel(sels[0], drop=drop)
        elif key == '2':
            prefix = 'forced' if dims and modify else None
            data = data.sel(sels[1], drop=drop)
        else:
            suffix = 'response'
            with xr.set_options(keep_attrs=True):
                data = data.climo.sel(sels[1]) - data.climo.sel(sels[0])
        if prefix or suffix:
            data.climo._add_prefix_suffix(prefix=prefix, suffix=suffix)
        return data

    def sel_time(self, date=None, **kwargs):
        """
        Select indices matching date or "virtual" datetime component.
        See: http://xarray.pydata.org/en/stable/time-series.html

        Parameters
        ----------
        date : date-like, optional
            Particular date selection(s).
        year, month, day, hour, minute, second, dayofyear, week, dayofweek, weekday
            "Virtual" datetime selection(s).
        """
        data = self.data
        try:
            time = self._to_native_name('time')  # translate native dimension name
        except KeyError:
            raise RuntimeError('Time dimension not found.')
        if date is not None:
            data = data.sel({time: date})
        for key, value in kwargs.items():
            if value is None:
                continue
            data = data.sel({time: data[f'{time}.{key}'] == value})
        return data

    def standardize_coords(self, verbose=False):
        """
        Standardize coordinates to satisfy CF conventions using `guess_coord_axis`
        and `rename_like`. May also support variable standardization in the future
        using `CFVariableRegistry`. For now, this function does the following:

        * Adds ``longitude`` and ``latitude`` standard names and ``degrees_east``
          and ``degrees_north`` units to detected ``X`` and ``Y`` axes.
        * Ensures detected longitude and latitude coordinates are designated
          as ``X`` and ``Y`` axes if none are present.
        * Ensures unique ``Z`` axis is also detected as ``vertical`` and transforms
          height-like, pressure-like, and temperature-like vertical coordinate units
          to kilometers, hectopascals, and kelvin, respectively.
        * Renames longitude, latitude, vertical, and time coordinate names
          to ``'lon'``, ``'lat'``, ``'lev'``, and ``'time'``, respectively.
        * Renames coordinate bounds to the coordinate names followed by a
          ``'_bnds'`` suffix and removes all attributes from bounds variables.

        Existing attributes are not overwritten.

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, print statements are issued.
        """
        # Update 'axis' attributes and 'longitude', 'latitude' standard names and units
        data = self.data
        for coord in data.coords.values():
            if 'cartesian_axis' in coord.attrs:  # rename non-standard axis specifier
                coord.attrs.setdefault('axis', coord.attrs.pop('cartesian_axis'))
        data = data.cf.guess_coord_axis(verbose=verbose)

        # Ensure unique longitude and latitude axes are designated as X and Y
        for axis, coord in zip(('X', 'Y'), ('longitude', 'latitude')):
            if coord in data.cf and axis not in data.cf:
                da = data.climo.cf[coord]
                data.coords[da.name].attrs['axis'] = axis
                if verbose:
                    print(
                        f'Set {coord} coordinate {da.name!r} axis type to {axis!r}.'
                    )

        # Manage all Z axis units and interpret 'positive' direction if not set
        # (guess_coord_axis does not otherwise detect 'positive' attribute)
        for name in data.cf.axes.get('Z', []):
            da = data.climo.coords[name]  # climopy makes unit-transformable copy
            units = data.coords[name].attrs.get('units', None)
            units = units if units is None else parse_units(units)
            positive = None
            if units is None:
                pass
            elif units == 'level' or units == 'layer':  # ureg.__eq__ handles strings
                positive = 'up'  # +ve vertical direction is increasing vals
            elif units == 'sigma_level':  # special CF unit
                positive = 'down'
            elif units.is_compatible_with('Pa'):
                positive = 'down'
                da = da.climo.to_units('hPa')
            elif units.is_compatible_with('m'):
                positive = 'up'
                da = da.climo.to_units('km')
            elif units.is_compatible_with('K'):
                positive = 'up'
                da = da.climo.to_units('K')
            if positive is None:
                positive = 'up'
                warnings._warn_climopy(
                    f'Ambiguous positive direction for vertical coordinate {name!r}. '
                    'Assuming up.'
                )
            da = da.climo.dequantify()
            da.attrs.setdefault('positive', positive)
            data = data.assign_coords({da.name: da})
            if verbose:
                print(
                    f'Set vertical coordinate {name!r} units to {da.climo.units} '
                    f'with positive direction {positive!r}.'
                )

        # Rename longitude, latitude, vertical, and time coordinates if present
        # WARNING: If multiples of each coordinate type are found, this triggers error
        coords = {  # dummy CF-compliant coordinates used with rename_like
            'lon': ('lon', [], {'standard_name': 'longitude'}),
            'lat': ('lat', [], {'standard_name': 'latitude'}),
            'lev': ('lev', [], {'positive': 'up'}),
            'time': ('time', [], {'standard_name': 'time'}),
        }
        coords_prev = data.cf.coordinates
        sample = xr.Dataset(coords=coords)
        data = data.cf.rename_like(sample)
        if verbose:
            coords_curr = data.cf.coordinates
            for coord, names_curr in coords_curr.items():
                names_prev = coords_prev.get(coord, [])
                for name_prev, name_curr in zip(names_prev, names_curr):
                    if name_prev != name_curr:
                        print(
                            f'Renamed coordinate {coord!r} name '
                            f'{name_prev!r} to {name_curr!r}.'
                        )

        # Manage bounds variables
        for name, da in data.coords.items():
            if isinstance(data, xr.Dataset):
                # Delete missing bounds attributes
                bounds = da.attrs.get('bounds')
                if bounds and bounds not in data:
                    del da.attrs['bounds']
                    if verbose:
                        print(
                            f'Deleted coordinate {name!r} bounds attribute {bounds!r} '
                            '(bounds variable not present in dataset).'
                        )
                # Infer unset bounds attributes
                for suffix in ('bnds', 'bounds'):
                    bounds = name + '_' + suffix
                    if bounds in data and 'bounds' not in da.attrs:
                        da.attrs['bounds'] = bounds
                        if verbose:
                            print(
                                f'Set coordinate {name!r} bounds to discovered '
                                f'bounds-like variable {bounds!r}.'
                            )
                # Standardize bounds name and remove attributes (similar to rename_like)
                bounds = da.attrs.get('bounds')
                if bounds and bounds != (bounds_new := da.name + '_bnds'):
                    da.attrs['bounds'] = bounds_new
                    data = data.rename_vars({bounds: bounds_new})
                    data[bounds_new].attrs.clear()
                    if verbose:
                        print(
                            'Renamed bounds variable {bounds!r} to {bounds_new!r}.'
                        )
            # Delete bounds variables for DataArrays, to prevent CF warning issue
            elif 'bounds' in da.attrs:
                del da.attrs['bounds']

        return data

    def truncate(self, bounds=None, *, ignore_extra=False, **kwargs):
        """
        Restrict the data into exact bounds using `.interp`. The bounds are specified
        with e.g. ``latmin=x``, ``latmax=y``, ``latlim=(x, y)``, or ``lat=(x, y)``.

        Parameters
        ----------
        bounds : dict-like, optional
            The bounds specifications. For e.g. latitude dimension `lat`, the entries
            should look like ``lat_min=min_value``, ``lat_max=max_value``,
            ``lat_lim=(min, max)``, or the shorthand ``lat=(min, max)``.
        **kwargs
            The bounds specifications passed as keyword args.
        """
        # Parse truncation args
        # NOTE: Data attributes are conserved during sel, interp, concat operations.
        # NOTE: This uses the unit-friendly accessor sel method. Range is limited
        # *exactly* by interpolating onto requested bounds.
        data = self.data
        bounds = bounds or {}
        bounds.update(kwargs)
        bounds, kwargs = self._parse_truncate_args(**bounds)
        if kwargs and not ignore_extra:
            raise ValueError(f'truncate() got unexpected keyword args {kwargs}.')
        if any(_.size > 2 for _ in bounds.values()):
            raise ValueError(f'truncate() args {kwargs} yield non-scalar bounds.')

        # Iterate through truncations
        # NOTE: The below uses uses _iter_by_indexer_coords
        for dim, bound in bounds.items():
            lo, hi = bound.values.squeeze()  # pull out of array
            coords = data.coords[dim]  # must be unquantified
            attrs = coords.attrs.copy()
            dimlo = dimhi = None
            if lo is not None and lo not in coords:
                dimlo = data.climo.interp({dim: lo})  # interpolate to bounds
            if hi is not None and hi not in coords:
                dimhi = data.climo.interp({dim: hi})
            data = data.climo.sel({dim: slice(lo, hi)})
            if data.sizes[dim] == 0:
                raise ValueError(f'Invalid bounds {dim}=({lo!r}, {hi!r}).')
            data = xr.concat(
                [data_ for data_ in (dimlo, data, dimhi) if data_ is not None],
                dim=dim,
                compat='no_conflicts',
                combine_attrs='no_conflicts',
            )
            data.coords[dim].attrs.update(attrs)

        return data

    @property
    def cf(self):
        """
        Redirection to the `CFAccessor`.
        """
        return self._cls_cf(self.data)

    @property
    def coords(self):
        """
        Wrap `.coords` with a mapping that returns quantified coordinates, or
        variables transformed from the native coordinates using `to_variable`
        (e.g. `meridional_coordinate` from `latitude`). The coordinate top boundaries,
        bottom boundaries, or thicknesses can also be returned by appending the
        name with ``_top``, ``_bot``, or ``_del`` (or ``_delta``), respectively. Always
        returns copies of the coordinate variables since xarray coordinates
        `cannot be quantified in-place <https://github.com/pydata/xarray/issues/525>`__.
        """
        return self._cls_coords(self.data, self.registry)

    @property
    def data(self):
        """
        Redirect to the underlying `xarray.Dataset` or `xarray.DataArray`.
        """
        return self._data

    @property
    def dims(self):
        """
        Redirect to `.dims`.
        """
        return self.data.dims

    @property
    def loc(self):
        """
        Wrap `.loc` with an indexer that handles units and coordinate types.
        """
        return self._cls_loc(self.data)

    @property
    def param(self):
        """
        The parameter corresponding to the major parameter sweep axis. Sweep axes
        are detected as any coordinate whose cfvariable has a ``base`` attribute.
        """
        dims = tuple(
            dim for dim, coord in self.data.coords.items()
            if coord.climo.cfvariable.base is not None  # is param if has 'base' value
        )
        if len(dims) == 0:
            raise RuntimeError('No parameter dimensions found.')
        return self.data.coords[dims[0]].climo.dequantify()

    @property
    def registry(self):
        """
        The active `CFVariableRegistry` used to look up variables with `.cfvariable`.
        """
        return self._registry

    @registry.setter
    def registry(self, reg):
        if not isinstance(reg, CFVariableRegistry):
            raise ValueError('Input must be a CFVariableRegistry instance.')
        self._registry = reg

    @property
    def vertical_type(self):
        """
        Return type of `z` axis, i.e. one of ``'temperature'``, ``'pressure'``,
        ``'height'``, or ``'unknown'``. Model levels not yet supported.
        """
        units = self.cf['vertical'].climo.units
        if units.is_compatible_with('K'):
            return 'temperature'
        elif units.is_compatible_with('Pa'):
            return 'pressure'
        elif units.is_compatible_with('m'):
            return 'height'
        else:
            return 'unknown'


@xr.register_dataarray_accessor('climo')
class ClimoDataArrayAccessor(ClimoAccessor):
    """
    Accessor for `xarray.DataArray`\\ s. Includes several stub functions for
    integration with free-standing climopy functions (similar to numpy design).
    """
    _cls_groupby = _DataArrayGroupByQuantified
    _cls_coords = _DataArrayCoordsQuantified
    _cls_loc = _DataArrayLocIndexerQuantified
    _cls_cf = _CFDataArrayAccessor

    def __repr__(self):
        return f'<climopy.ClimoDataArrayAccessor>({self._cf_repr(brackets=False)})'

    def __init__(self, data_array, registry=vreg):
        """
        Parameters
        ----------
        data_array : xarray.DataArray
            The data.
        registry : CFVariableRegistry
            The active registry used to look up variables with `.cfvariable`.
        """
        self._data = data_array
        self._registry = registry

    def __contains__(self, key):
        """
        Contains the coordinate.
        """
        return key in self.coords

    def __getitem__(self, key):
        """
        Retrieve a coordinate variable or derived coordinate variable.
        """
        return self.coords[key]

    def __getattr__(self, attr):
        """
        Retrieve a coordinate variable, attribute, or cfvariable property.
        """
        if attr[:1] == '_' or attr == 'cfvariable':
            return super().__getattribute__(attr)  # trigger builtin AttributeError
        elif attr in self.coords:
            return self.coords[attr]
        elif self._has_cf_attr(attr):
            return self._get_cf_attr(attr)
        else:
            return super().__getattribute__(attr)  # trigger builtin AttributeError

    def _add_prefix_suffix(self, prefix=None, suffix=None):
        """
        Add prefix or suffix to CFVariable, without overwriting existing ones.
        """
        attrs = self.data.attrs
        if prefix is not None:
            attrs['prefix'] = (prefix + ' ' + attrs.get('prefix', '')).strip()
        if suffix is not None:
            attrs['suffix'] = (attrs.get('suffix', '') + ' ' + suffix).strip()

    def _has_cf_attr(self, attr):
        """
        Return whether CFVariable attribute exists.
        """
        try:
            self._get_cf_attr(attr)
        except AttributeError:
            return False
        else:
            return True

    def _get_cf_attr(self, attr, default=None):
        """
        Return CFVariable attribute if it exists.
        """
        try:
            var = self.cfvariable
        except AttributeError:
            raise AttributeError(f'Unknown CFVariable for name {self.data.name}.')
        try:
            return getattr(var, attr)
        except AttributeError:
            raise AttributeError(f'Invalid CFVariable attribute {attr!r}.')

    @_while_quantified
    def reduce(
        self, indexers=None, dataset=None, centroid=False, weight=None, mask=None,
        **kwargs
    ):
        """
        Reduce the dimension of a `xarray.DataArray` with arbitrary method(s).

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values representing the
            "reduction modes" for the dimensions. Values can be any of the following:

            ===============  =========================================================
            Reduction mode   Description
            ===============  =========================================================
            array-like       The value(s) at this location using ``self.interp``.
            param-spec       Parameter name passed to ``self.get``, e.g. ``'trop'``.
            ``'int'``        Integral along the dimension.
            ``'avg'``        Weighted mean along the dimension.
            ``'anom'``       Weighted anomaly along the dimension.
            ``'lcumint'``    Cumulative integral from the left.
            ``'rcumint'``    Cumulative integral from the right.
            ``'lcumavg'``    Cumulative average from the left.
            ``'rcumavg'``    Cumulative average from the right.
            ``'lcumanom'``   Anomaly w.r.t. cumulative average from the left.
            ``'rcumanom'``   Anomaly w.r.t. cumulative average from the right.
            ``'min'``        Local minima along the dimension.
            ``'max'``        Local maxima along the dimension.
            ``'argmin'``     Location(s) of local minima along the dimension.
            ``'argmax'``     Location(s) of local maxima along the dimension.
            ``'argzero'``    Location(s) of zeros along the dimension.
            ``'absmin'``     Global minimum along the dimension.
            ``'absmax'``     Global maximum along the dimension.
            ``'absargmin'``  Location of global minimum along the dimension.
            ``'absargmax'``  Location of global maximum along the dimension.
            ``'timescale'``  *For time dimension only.* The e-folding timescale.
            ``'autocorr'``   *For time dimension only.* The autocorrelation.
            ``'hist'``       *For time dimension only.* The histogram.
            ===============  =========================================================

        dataset : `xarray.Dataset`, optional
            The associated dataset. This is needed for 2D reduction
            of isentropic data, and may also be needed for 2D reduction
            of horizontal data with 2D latitude/longitude coords in the future.
        centroid : bool, optional
            Get the value-weighted average wavenumber using `.centroid`. Units
            are distance.
        weight : str or `xarray.DataArray`, optional
            Additional weighting parameter name or `xarray.DataArray`, used for
            averages and integrations. Mass weighting is applied automatically.
        mask : {None, 'land', 'sea', 'trop', 'pv'}, optional
            The 2-dimensional mask to apply before taking the weighted average. Presets
            will be added to this.
        **indexers_kwargs
            The keyword arguments form of `indexers`.
            One of `indexers` or `indexers_kwargs` must be provided.
        **truncate_kwargs
            Dimension names with the suffix ``_min``, ``_max``, or ``_lim`` used
            to constrain the range within which the above reduction methods are
            executed. For example ``data.reduce(lev='mean', lev_min=500 * ureg.hPa)``
            takes the average in the bottom half of the atmosphere. Bounds do not
            have to have units attached. Bounds can *also* take on the value of a
            `param` variable, e.g. ``lev_max='tropopause'``. Multiple-valued ``min``
            or ``max`` bounds will be reduced for each bounds selection, then the
            resulting reduction will be concatenated along a `track` dimension.
        **kwargs
            Remaining keyword arguments are passed to relevant functions
            like `zerofind` and `rednoisefit`.

        Returns
        -------
        xarray.DataArray
            The data with relevant dimension(s) reduced.
        """
        # Initial stuff
        data = self.data
        name = data.name
        find_names = (
            'min', 'max', 'absmin', 'absmax', 'argmin', 'argmin',
            'absargmin', 'absargmax', 'argzero',
        )
        average_names = {
            'int': ('integrate', {}),
            'avg': ('average', {}),
            'anom': ('anomaly', {}),
            'lcumint': ('cumintegrate', {}),
            'rcumint': ('cumintegrate', {'reverse': True}),
            'lcumavg': ('cumaverage', {}),
            'rcumavg': ('cumaverage', {'reverse': True}),
            'lcumanom': ('cumanomaly', {}),
            'rcumanom': ('cumanomaly', {'reverse': True}),
        }
        method_keys = {  # keyword args that can be passed to different methods
            'autocorr': ('lag', 'ilag', 'maxlag', 'imaxlag'),
            'autocovar': ('lag', 'ilag', 'maxlag', 'imaxlag'),
            'average': ('skipna',),
            'find': ('centered', 'which', 'diff', 'sep', 'seed', 'ntrack', 'dim_track'),
            'hist': ('bins',),
            'slope': (),
            'timescale': ('maxlag', 'imaxlag', 'maxlag_fit', 'imaxlag_fit'),
        }

        # Parse indexers
        # NOTE: Prefer dataset here to allow for things like lat_min='ehf_lat'
        indexers, kwargs = self._parse_indexers(  # NOTE: include pseudo for integrating
            indexers, include_scalar=True, include_pseudo=True, **kwargs
        )
        if isinstance(weight, str):
            if dataset is None:  # supplied by get_variable
                raise ValueError(f'Dataset required to infer weighting {weight!r}.')
            weight = dataset.climo[weight]

        # Parse truncation args
        trunc, kwargs = (dataset or data).climo._parse_truncate_args(**kwargs)
        if trunc.keys() - indexers.keys():
            raise ValueError(
                f'One of truncation dims {tuple(trunc)!r} missing from '
                f'list of reduction dims {tuple(indexers)!r} for var {data.name!r}.'
            )
        if trunc:
            sample = tuple(trunc.values())[0]
            dims = sample.dims[1:]
            datas = np.empty(sample.shape[1:], dtype='O')
            coords = {key: c for key, c in sample.coords.items() if key != 'startstop'}
        else:
            sample = None
            dims = ('track',)
            datas = np.array([None])
            coords = {}

        # Iterate through bounds combinations across dimensions
        # NOTE: _parse_truncate ensures all bounds passed are put into DataArrays with
        # a 'startstop' dim, an at least singleton 'track' dim, and matching shapes.
        used_kw = set()
        for idx in np.ndindex(datas.shape):  # ignore 'startstop' dimension
            # Limit range exactly be interpolating to bounds
            # NOTE: Common to use e.g. reduce(lat='int', latmin='ehf_lat') for data
            # with e.g. time axis. Here we are iterating through extra axes.
            isel_trunc = dict(zip(dims, idx))
            itrunc = {k: tuple(v.isel(isel_trunc).data) for k, v in trunc.items()}
            isel_data = {dim: i for dim, i in zip(dims, idx) if dim != 'track'}
            idata = data.isel(isel_data).climo.truncate(**itrunc)
            if weight is None:
                iweight = None
            else:
                iweight = data.isel(isel_data).climo.truncate(**itrunc)

            # Single dimension reductions
            # WARNING: Need to include *coords* so we can 'reduce' singleton lon
            for dim, method in indexers.items():
                # Verify dimensions
                if dim not in idata.climo.coords:
                    warnings._warn_climopy(
                        f'Skipping reduction {dim}={method!r} (has no coords).'
                    )

                # Various simple reduction modes
                # NOTE: Integral does not need dataset because here we are only
                # integrating unknown dimensions; attenuated spatial ones earlier.
                # TODO: Add '_hist' entry to __getitem__ that (1) gets jet latitudes
                # with argmax and (2) calls hist on the resulting latitudes.
                if isinstance(method, str) and (
                    method in method_keys
                    or method in find_names
                    or method in average_names
                ):
                    kw = {}
                    if method in find_names:
                        keys = method_keys['find']
                    elif method in average_names:
                        method, kw = average_names[method]
                        keys = method_keys['average']
                        kw.update({'weight': iweight})
                    kw.update({k: kwargs[k] for k in kwargs.keys() & set(keys)})
                    idata = getattr(idata.climo, method)(dim, **kw)
                    used_kw |= kw.keys()

                # Select single or multiple points with interpolation
                # For example: climo.get('dtdy', lev='mean', lat='ehf_lat')
                else:
                    loc = getattr(method, 'magnitude', method)
                    if dim in self.cf.coordinates.get('time', ()):
                        idata = idata.climo.sel_time({dim: loc})
                    elif _is_numeric(loc):  # i.e. not datetime, string, etc.
                        idata = idata.climo.interp({dim: loc})
                    else:
                        try:
                            loc = dataset.climo.get_variable(loc, standardize=True)
                            idata = idata.climo.interp({dim: loc})
                        except (KeyError, ValueError, AttributeError):
                            raise ValueError(f'Invalid {method=}.')

            # Add to list of reductions along different subselections
            datas[idx] = idata

        # Detect invalid kwargs
        extra_kw = kwargs.keys() - used_kw
        if extra_kw:
            raise ValueError('Unexpected kwargs: ' + ', '.join(map(repr, extra_kw)))

        # Concatente with combine_nested, then fix weird dimension reordering
        data = xr.combine_nested(
            datas.tolist(),
            concat_dim=dims,
            join='exact',
            compat='identical',
            combine_attrs='identical',
        )
        data = data.assign_coords(coords)
        dims = (*dims, *(_ for _ in tuple(data.dims)[::-1] if _ not in dims))
        data = data.transpose(*dims)
        if data.sizes['track'] == 1:
            data = data.isel(track=0, drop=True)

        # Add back name and attributes
        # NOTE: Climopy DataArray wrappers may rename output dimension to coordinate
        # name e.g. with 'argmax'. Here we keep names the same.
        data.name = name
        return data

    @_while_quantified
    @_keep_tracked_attrs
    def _integrate_or_average(
        self,
        dims,
        weight=None,
        integral=True,
        cumulative=False,
        reverse=False,
        skipna=None,
        **kwargs
    ):
        """
        Perform integration or average for all of climopy's weighted integration
        and averaging functions.
        """
        # Apply truncations to data and extra weight
        # NOTE: Great way
        data = self.data.climo.truncate(**kwargs)
        name = data.name
        weights_explicit = []  # quantification necessary for integrate()
        weights_implicit = []  # quantification not necessary, slows things down a bit
        if weight is not None:
            weight = weight.climo.truncate(**kwargs)
            weights_implicit.append(weight.climo.dequantify())

        # Translate dims. When none are passed, interpret this as integrating over the
        # entire atmosphere. Support irregular grids by preferring 'volume' or 'area'
        if dims:
            # Handle special case: Integral along a latitude line without integration
            # over longitude applies unnormalized 'cell depth' latitude weights (km) and
            # normalized longitude and vertical weights. In practice however we almost
            # always integrate over just longitude or both longitude or latitude... and
            # therefore we want unnormalized 'cell width' longitude (cosine latitude)
            # weights. Here we manually add these with a warning message.
            dims_orig = self._parse_dims(dims, include_scalar=True, include_pseudo=True)
            dims_std = tuple(self._to_cf_coord_name(dim) for dim in dims_orig)
            if integral and 'latitude' in dims_std and 'longitude' not in dims_std:
                warnings._warn_climopy(
                    'Only latitude integral was specified, but almost always want to '
                    'integrate over just longitudes or both longitudes and latitudes. '
                    'Adding longitude as an integration dimension.'
                )
                dims_orig = ('longitude', *dims_orig)
        elif 'volume' in self.cf.cell_measures:
            dims_orig = ('volume',)
        elif 'area' in self.cf.cell_measures:
            dims_orig = ('area', 'vertical')
        else:
            dims_orig = ('longitude', 'latitude', 'vertical')

        # Get quantified cell measure weights for dimensions we are integrating over,
        # and translate 'area' and 'volume' to their component coordinates
        dims = []
        measures = set()
        for dim in dims_orig:
            # Get the corresponding cell measure
            # NOTE: Why not allow 'width', 'depth', 'height' user input? Because 'area'
            # and 'volume' are required for non-standard horizontal grids where
            # 'latitude' and 'longitude' *never* make sense on their own.
            if dim in ('area', 'volume'):  # pseudo-dimensions
                measure = dim
                coords = ('longitude', 'latitude')
                if dim == 'volume':
                    coords += ('vertical',)
            else:
                coord = self._to_cf_coord_name(dim)  # including time!
                if coord is None:
                    warnings._warn_climopy(f'Unknown weights for non-CF dim {dim!r}.')
                    continue
                measure = CELL_MEASURES_BY_COORD[coord]
                coords = (coord,)
            try:
                weight = self.cf[measure]
            except KeyError:
                raise ValueError(f'Cell measure {measure!r} for dim {dim!r} not found.')
            dims.extend(self._to_native_name(coord) for coord in coords)
            measures.add(measure)
            weights_explicit.append(weight.climo.quantify())

        # Add unquantified cell measure weights for measures whose dimensions match any
        # of the dimensions we are already integrating over (e.g. 'depth' is added
        # for an areal integral to account for differing cell thickness)
        cell_measures = data.attrs.get('cell_measures')
        for (measure,), varname in self._decode_cf_attr(cell_measures):
            if measure in measures:
                continue
            try:
                weight = self.cf[measure]
            except KeyError:
                warnings._warn_climopy(
                    f'Cell measure {measure!r} with name {varname!r} not found.'
                )
                continue
            if set(dims) & set(weight.dims):
                weights_implicit.append(weight.climo.dequantify())

        # Use math.prod to reduce items in list
        # NOTE: numpy.prod just returns 0 for some reason. math.prod seems to work
        # with arbitrary objects, similar to builtin sum()
        one = xr.DataArray(1)  # ensure returned 'weight' is DataArray
        weights = (*weights_explicit, *weights_implicit)
        if integral:
            cell_method = 'integral'
            normalize_denom = True
            weight_num = math.prod(weights, start=one)
            weight_denom = math.prod(weights_implicit, start=one)
        else:
            cell_method = 'average'
            normalize_denom = False
            weight_num = weight_denom = math.prod(weights, start=one).climo.dequantify()

        # Run integration
        dims_sum = tuple(dim for dim in dims if dim in data.dims)
        data = (
            data.climo._weighted_sum(
                dims_sum,
                weight_num,
                skipna=skipna,
                cumulative=cumulative,
                reverse=reverse
            ) / data.climo._sum_of_weights(
                dims_sum,
                weight_denom,
                normalize=normalize_denom,
                cumulative=cumulative,
                reverse=reverse,
            )
        )
        data.climo.add_cell_methods({tuple(dims): cell_method})
        data.name = name
        return data

    def _weighted_sum(
        self, dims, weights, skipna=None, cumulative=False, reverse=False
    ):
        """
        Return the weighted sum, accounting for where weights and data are NaN,
        Optionally sum cumulatively.
        """
        # NOTE: Prefer xr.dot to multiplication, broadcasting, and summation because
        # xr.dot doesn't have to build extra giant array. See xarray weighted.py source
        data = self.data
        if skipna or skipna is None and data.dtype.kind in 'cfO':
            data = data.fillna(0.0)  # works with pint quantities
        if cumulative:
            if len(dims) > 1:
                raise ValueError('Too many dimensions for cumulative integration.')
            isel = {dims[0]: slice(None, None, -1) if reverse else slice(None)}
            res = (data * weights).isel(isel).cumsum(dims).isel(isel)
        else:
            res = xr.dot(data, weights, dims=dims)
        return res

    def _sum_of_weights(
        self, dims, weights, cumulative=False, normalize=False, reverse=False
    ):
        """
        Return the sum of weights, accounting for NaN data values and masking where
        weights sum to zero. Optionally sum cumulatively.
        """
        mask = self.data.notnull().astype(int)
        if normalize:  # needed for denominator when integrating
            mask = mask / xr.ones_like(mask).sum(dims)
        if cumulative:
            if len(dims) > 1:
                raise ValueError('Too many dimensions for cumulative integration.')
            isel = {dims[0]: slice(None, None, -1) if reverse else slice(None)}
            res = (mask * weights).isel(isel).cumsum(dims).isel(isel)
        else:
            res = xr.dot(mask, weights, dims=dims)
        res = res.where(res != 0.0)  # 0.0 --> NaN; works with pint.Quantity data
        return res

    @_manage_reduced_coords  # need access to cell_measures, so place before keep_attrs
    def integrate(self, dim=None, **kwargs):
        """
        Mass-weighted integration.

        Parameters
        ----------
        dim : dim-spec or {'area', 'volume'}, optional
            The integration dimensions. Weights are applied automatically using CF
            `cell_measures` stored in the coordinates. If not specified, the data
            is integrated over the entire domain.
        skipna : bool, optional
            Whether to skip NaN values.
        weight : xr.DataArray, optional
            Optional additional weighting.
        **kwargs
            Passed to `.truncate`. Used to limit bounds of integration.

        Notes
        -----
        ClimoPy's mass weighted operations work with dedicated functions rather than
        `xarray.Weighted` objects because the selection of mass weights depends on the
        dimensions being integrated.
        """
        kwargs.update(integral=True, cumulative=False)
        return self._integrate_or_average(dim, **kwargs)

    @_manage_reduced_coords  # need access to cell_measures, so place before keep_attrs
    def average(self, dim=None, **kwargs):
        """
        Mass-weighted average.

        Parameters
        ----------
        dim : dim-spec or {'area', 'volume'}, optional
            The averaging dimensions. Weights are applied automatically using CF
            `cell_measures` stored in the coordinates. If not specified, the data
            is averaged over the entire domain.
        skipna : bool, optional
            Whether to skip NaN values.
        weight : xr.DataArray, optional
            Optional additional weighting.
        **kwargs
            Passed to `.truncate`. Used to limit bounds of average.

        Notes
        -----
        ClimoPy makes artifical distinction in `cell_method` between `mean` (the naive
        mean along axes) and `average` (the mass-weighted mean).
        """
        kwargs.update(integral=False, cumulative=False)
        return self._integrate_or_average(dim, **kwargs)

    def anomaly(self, *args, **kwargs):
        """
        Anomaly with respect to mass-weighted average.

        Parameters
        ----------
        *args, **kwargs
            Passed to `ClimoDataArrayAccessor.average`.
        """
        # TODO: Indicate anomalous data with cell method
        with xr.set_options(keep_attrs=True):
            return self.data - self.average(*args, **kwargs)

    def cumintegrate(self, dim, skipna=None, **kwargs):
        """
        Cumulative integral.

        Parameters
        ----------
        dim : dim-spec
            The integration dimension. Weights are applied automatically using
            `cell_measures` stored in the coordinates.
        skipna : bool, optional
            Whether to skip NaN values.
        reverse : bool, optional
            Whether to change the direction of the accumulation to right-to-left.
        **kwargs
            Passed to `.truncate`. Used to limit bounds of integration.
        """
        kwargs.update(integral=True, cumulative=True)
        return self._integrate_or_average(dim, **kwargs)

    def cumaverage(self, dim, reverse=False, weight=None, skipna=None, **kwargs):
        """
        Cumulative average.

        Parameters
        ----------
        dim : dim-spec
            The averaging dimension. Weights are applied automatically using
            `cell_measures` stored in the coordinates.
        skipna : bool, optional
            Whether to skip NaN values.
        reverse : bool, optional
            Whether to change the direction of the accumulation to right-to-left.
        **kwargs
            Passed to `.truncate`. Used to limit bounds of integration.
        """
        kwargs.update(integral=False, cumulative=True)
        return self._integrate_or_average(dim, **kwargs)

    def cumanomaly(self, *args, **kwargs):
        """
        Anomaly relative to cumulative mass-weighted average.

        Parameters
        ----------
        *args, **kwargs
            Passed to `ClimoDataArrayAccessor.cumaverage`.
        """
        # TODO: Indicate anomalous data with cell method
        with xr.set_options(keep_attrs=True):
            return self.data - self.cumaverage(*args, **kwargs)

    @_manage_reduced_coords
    @_keep_tracked_attrs
    def _mean_or_sum(self, method, dim=None, skipna=None, weight=None, **kwargs):
        """
        Simple average or summation.
        """
        data = self.truncate(**kwargs)
        dims = data.dims if dim is None else self._parse_dims(dim, ignore_scalar=True)
        if weight is not None:
            data = data.weighted(weight.climo.truncate(**kwargs))
        data = getattr(data, method)(dims, skipna=skipna, keep_attrs=True)
        data.climo.add_cell_methods({dims: method})
        return data

    def mean(self, dim=None, skipna=None, weight=None, **kwargs):
        """
        Take simple mean along dimension(s), preserving attributes and coordinates.

        Parameters
        ----------
        dim : str or list of str, optional
            The dimensions.
        skipna : bool, optional
            Whether to skip NaN values.
        weight : xr.DataArray, optional
            Optional weighting.
        **kwargs
            Passed to `xarray.DataArray.mean`. Used to limit bounds of mean.
        """
        return self._mean_or_sum('mean', dim, **kwargs)

    def sum(self, dim=None, skipna=None, weight=None, **kwargs):
        """
        Take simple summation along dimension(s), preserving attributes and coordinates.

        Parameters
        ----------
        dim : str or list of str, optional
            The dimensions.
        skipna : bool, optional
            Whether to skip NaN values.
        weight : xr.DataArray, optional
            Optional weighting.
        **kwargs
            Passed to `.truncate`. Used to limit bounds of summation.
        """
        return self._mean_or_sum('sum', dim, **kwargs)

    @_keep_tracked_attrs
    def runmean(self, indexers=None, **kwargs):
        """
        Return the running mean along different dimensions.

        Parameters
        ----------
        indexers : dict, optional
            Dictionary mapping of dimension names to window lengths. For example, to get
            the 11-item or 11-day running mean, use ``time=11`` or ``time=11 * ureg.d``.
        **indexers_kwargs
            The keyword arguments form of `indexers`.
            One of `indexers` or `indexers_kwargs` must be provided.
        **kwargs
            Passed to `climopy.deriv_uneven`. The `order` keyword arg is ignored.
        """
        data = self.data
        indexers, _ = self._parse_indexers(indexers, allow_kwargs=False, **kwargs)
        for dim, window in indexers.items():
            if isinstance(window, ureg.Quantity):
                coords = data.climo.coords[dim]
                window = int(np.round(window / (coords[1] - coords[0])).magnitude)
                if window <= 0:
                    raise ValueError('Invalid window length.')
            data = var.runmean(data, window, dim=dim)
        return data

    @_while_quantified
    @_keep_tracked_attrs
    def derivative(self, indexers=None, half=False, **kwargs):
        """
        Take the nth order centered finite difference for the specified dimensions.

        Parameters
        ----------
        indexers : dict, optional
            Dictionary mapping of dimension names to derivative order. For example,
            to get the second time derivative, use ``time=0``.
        half : bool, optional
            Whether to use more accurate (but less convenient) half-level
            differentiation rather than centered differentiation.
        **indexers_kwargs
            The keyword arguments form of `indexers`.
            One of `indexers` or `indexers_kwargs` must be provided.
        **kwargs
            Passed to `climopy.deriv_uneven`. The `order` keyword arg is ignored.
        """
        data = self.data
        indexers, kwargs = self._parse_indexers(
            indexers, include_transforms=True, **kwargs
        )
        kwargs.pop('order', None)
        for dim, order in indexers.items():
            coords = data.climo.coords[dim]
            if half:
                _, data = diff.deriv_half(coords, data, order=order, **kwargs)
            else:
                kwargs.setdefault('keepedges', True)
                data = diff.deriv_uneven(coords, data, order=order, **kwargs)
            data.climo.add_cell_methods({dim: 'derivative'})

        return data

    def convergence(self, *args, **kwargs):
        """
        Take the spherical meridional convergence. To calculating divergence at the
        poles, the numerator is assumed to vanish and l'Hopital's rule is invoked.

        Parameters
        ----------
        half : bool, optional
            Whether to use more accurate (but less convenient) half-level
            differentiation rather than centered differentiation.
        **kwargs
            Passed to `climopy.deriv_uneven` or `climopy.deriv_half`.
        """
        result = self.divergence(*args, **kwargs)
        with xr.set_options(keep_attrs=True):
            return -1 * result

    @_while_quantified
    @_keep_tracked_attrs
    def divergence(self, half=False, **kwargs):
        """
        Take the spherical meridional divergence. To calculating divergence at the
        poles, the numerator is assumed to vanish and l'Hopital's rule is invoked.

        Parameters
        ----------
        half : bool, optional
            Whether to use more accurate (but less convenient) half-level
            differentiation rather than centered differentiation.
        **kwargs
            Passed to `climopy.deriv_uneven` or `climopy.deriv_half`.
        """
        # Compute divergence in spherical coordinates
        # div = diff.deriv1(y[:2], data * cos, **kwargs) / cos
        y = self.coords['meridional_coordinate']
        cos = self.coords['cosine_latitude']
        data = self.data
        kwargs['order'] = 1
        if half:
            cos2 = 0.5 * (cos.data[1:] + cos.data[:-1])
            y, div = diff.deriv_half(y, data * cos, **kwargs)
            div /= cos2
        else:
            kwargs.setdefault('keepedges', True)
            div = diff.deriv_uneven(y, data * cos, **kwargs) / cos

        # If numerator vanishes, divergence at poles is precisely 2 * dflux / dy.
        # See Hantel 1974, Journal of Applied Meteorology, or just work it out
        # for yourself (simple l'Hopital's rule application).
        lat = div.coords['lat']  # without units
        for lat, isel in ((lat[0], slice(None, 2)), (lat[-1], slice(-2, None))):
            if abs(lat) == 90:
                div.climo.loc[{'lat': lat}] = (
                    2 * data.isel(lat=isel).diff(dim='lat').isel(lat=0).data
                    / (y.data[isel][1] - y.data[isel][0])
                )

        div.climo.add_cell_methods({'area': 'divergence'})
        return div

    @_keep_tracked_attrs
    def autocorr(self, dim, **kwargs):
        """
        Calculate the autocorrelation.

        Parameters
        ----------
        dim : str
            The dimension.
        **kwargs
            Passed to `climopy.autocorr`.
        """
        data = self.data
        if not kwargs.keys() & {'lag', 'ilag', 'maxlag', 'imaxlag'}:
            kwargs['ilag'] = 1
        _, data = var.autocorr(data.coords[dim], data, dim=dim, **kwargs)
        data.climo.add_cell_methods({dim: 'correlation'})
        return data

    @_keep_tracked_attrs
    def autocovar(self, dim, **kwargs):
        """
        Calculate the autocovariance.

        Parameters
        ----------
        dim : str
            The dimension.
        **kwargs
            Passed to `climopy.autocorr`.
        """
        data = self.data
        if not kwargs.keys() & {'lag', 'ilag', 'maxlag', 'imaxlag'}:
            kwargs['ilag'] = 1
        _, data = var.autocovar(data.coords[dim], data, dim=dim, **kwargs)
        data.climo.add_cell_methods({dim: 'covariance'})
        return data

    @_keep_tracked_attrs
    def centroid(self, dataset=None, **kwargs):
        """
        Calculate the value-weighted average wavenumber.

        Parameters
        ----------
        dataset : `xarray.Dataset`, optional
            The dataset.
        **kwargs
            Passed to `.truncate`. Used to restrict the bounds of the calculation.
        """
        # Multi-dimensional reduction: power-weighted centroid (wavenumber)
        # Mask region for taking power-weighted average
        # NOTE: Wavenumber dimension scaling bug is fixed in newer files
        # Need to rerun spectral decompositions on all experiments
        data = self.data.truncate(**kwargs)
        lat = self.coords['lat']
        k = self.coords['k']
        if np.all(k < 1):
            k /= (k[1] - k[0])
        kmask = k.climo.magnitude >= 1.5
        latmask = (lat.climo.magnitude >= 25) & (lat.climo.magnitude <= 70)

        # Get centroid as the *power-weighted average*. This prevents
        # recording big discrete jumps in wavelength.
        if dataset is None:
            raise ValueError('Dataset is required.')  # TODO: loosen restriction
        power = data.isel(k=kmask, lat=latmask)
        weight = dataset.climo['depth'] * dataset.climo['height'] * power
        denom = weight.sum()  # over all dims
        data = (k[kmask] * weight).sum() / denom
        lat = (lat[latmask] * weight).sum() / denom

        # Now convert to *physical* wavenumber. Start with circles per wave,
        # times mteres per circle, times 0.25 to get *quarter* wavelength.
        # TODO: Check Frierson et al 2006, make sure this is what people use
        # data.data[data.data >= 10e3 * ureg.km] = np.nan * ureg.km
        circum = 2 * np.pi * const.a * np.cos(np.pi * lat / 180)
        data = 0.25 * (circum / data)  # from dimensionless to meters
        data.climo.add_cell_methods({('lat', 'k'): 'centroid'})
        return data

    @_manage_reduced_coords
    @_keep_tracked_attrs
    def _find_extrema(
        self, dim, abs=False, arg=False, which='max', dim_track=None, **kwargs
    ):
        """
        Find local or global extrema or their locations.
        """
        # Manage keyword args
        method = 'arg' + which if arg else which
        if abs:
            kwargs.setdefault('track', False)
        else:
            if dim_track:
                kwargs['axis_track'] = self.data.dims.index(dim_track)
                kwargs.setdefault('track', True)
            elif 'axis_track' not in kwargs:
                warnings._warn_climopy('Tracking dim for local zeros not provided.')
                kwargs['track'] = False
        if which in ('min', 'max'):
            kwargs['diff'] = 1
            if which == 'min':
                kwargs['which'] = 'negpos'
            if which == 'max':
                kwargs['which'] = 'posneg'

        # Get precise local values using linear interpolation
        # NOTE: The zerofind function applies pint units if not already applied
        dim = self._parse_dims(dim, single=True)
        data = self.data
        if dim == 'lat':  # TODO: remove kludge! error is with uweight lat=absmax
            data = data.transpose(..., dim)
        locs, values = utils.zerofind(data.coords[dim], data, **kwargs)

        # Get global values
        # TODO: Incorporate this into zerofind, and rename to 'extrema()'?
        if not abs:
            data = locs if arg else values

        # If no extrema were found (e.g. there are only extrema on edges)
        # revert to native min max functions.
        elif abs and locs.sizes['track'] == 0:
            result = getattr(data, method)(dim)  # e.g. min() or argmin()
            if arg:
                data = data.climo.coords[dim][result].drop(dim)  # drop unused dim
            else:
                data = result.drop(dim)

        # Otherwise select from the identified 'sandwiched' extrema and possible
        # extrema on the array edges. We merge zerofind values with array edges
        else:
            locs = [locs]
            values = [values]
            for i, idx in enumerate((0, -1)):
                ilocs = data.climo.coords[dim].isel({dim: idx}, drop=True)
                ivalues = data.isel({dim: idx}, drop=True)
                locs.append(ilocs.expand_dims('track'))
                values.append(ivalues.expand_dims('track'))
            locs = xr.concat(
                locs,
                dim='track',
                compat='override',  # needed to merge cell weights
                coords='minimal',
                combine_attrs='no_conflicts'
            )
            values = xr.concat(
                values,
                dim='track',
                compat='override',
                coords='minimal',
                combine_attrs='no_conflicts'
            )
            # Select location of largest minimum or maximum
            isel = {'track': getattr(values, 'arg' + which)('track')}
            data = (locs if arg else values).isel(isel, drop=True)

        # Use either actual locations or interpolated values. Restore attributes
        # TODO: By default, zerofind changes 'locs' name to coordinate name. Here,
        # instead, we keep the original variable name and attributes, but the change in
        # units is indicated with an 'argmin' cell method.
        # values = data.climo.interp({dim: locs.squeeze()}, method='cubic')
        data.name = self.data.name  # original name
        data.attrs.clear()
        data.attrs.update(self.data.attrs)  # original attrs
        data.climo.add_cell_methods({dim: method})

        return data

    def min(self, dim=None, **kwargs):
        """
        Return the local minima along the dimension.

        Parameters
        ----------
        dim : str, optional
            The dimension.
        dim_track : str, optional
            The dimension along which minima are tracked.
        **kwargs
            Passed to `climopy.zerofind`.
        """
        kwargs.update(which='min', abs=False, arg=False)
        return self._find_extrema(dim, **kwargs)

    def max(self, dim=None, **kwargs):
        """
        Return the local maxima along the dimension.

        Parameters
        ----------
        dim : str, optional
            The dimension.
        dim_track : str, optional
            The dimension along which maxima are tracked.
        **kwargs
            Passed to `climopy.zerofind`.
        """
        kwargs.update(which='max', abs=False, arg=False)
        return self._find_extrema(dim, **kwargs)

    def absmin(self, dim=None, **kwargs):
        """
        Return the global minimum along the dimension.

        Parameters
        ----------
        dim : str, optional
            The dimension.
        **kwargs
            Passed to `climopy.zerofind`.
        """
        kwargs.update(which='min', abs=True, arg=False)
        return self._find_extrema(dim, **kwargs)

    def absmax(self, dim=None, **kwargs):
        """
        Return the global maximum along the dimension.

        Parameters
        ----------
        dim : str, optional
            The dimension.
        **kwargs
            Passed to `climopy.zerofind`.
        """
        kwargs.update(which='max', abs=True, arg=False)
        return self._find_extrema(dim, **kwargs)

    def argmin(self, dim=None, **kwargs):
        """
        Return locations of the local minima along the dimension.

        Parameters
        ----------
        dim : str, optional
            The dimension.
        dim_track : str, optional
            The dimension along which minima are tracked.
        **kwargs
            Passed to `climopy.zerofind`.
        """
        kwargs.update(which='min', abs=False, arg=True)
        return self._find_extrema(dim, **kwargs)

    def argmax(self, dim=None, **kwargs):
        """
        Return locations of the local maxima along the dimension.

        Parameters
        ----------
        dim : str, optional
            The dimension.
        dim_track : str, optional
            The dimension along which maxima are tracked.
        **kwargs
            Passed to `climopy.zerofind`.
        """
        kwargs.update(which='max', abs=False, arg=True)
        return self._find_extrema(dim, **kwargs)

    def absargmin(self, dim=None, **kwargs):
        """
        Return location of the global minimum along the dimension.

        Parameters
        ----------
        dim : str, optional
            The dimension.
        **kwargs
            Passed to `climopy.zerofind`.
        """
        kwargs.update(which='min', abs=True, arg=True)
        return self._find_extrema(dim, **kwargs)

    def absargmax(self, dim=None, **kwargs):
        """
        Return location of the global maximum along the dimension.

        Parameters
        ----------
        dim : str, optional
            The dimension.
        **kwargs
            Passed to `climopy.zerofind`.
        """
        kwargs.update(which='max', abs=True, arg=True)
        return self._find_extrema(dim, **kwargs)

    def argzero(self, dim=None, **kwargs):
        """
        Return locations of the zero-valued points along the dimension.

        Parameters
        ----------
        dim : str, optional
            The dimension.
        dim_track : str, optional
            The dimension along which zeros are tracked.
        **kwargs
            Passed to `climopy.zerofind`.
        """
        kwargs.update(which='zero', abs=False, arg=True)
        return self._find_extrema(dim, **kwargs)

    @_keep_tracked_attrs
    def hist(self, dim, bins=None):
        """
        Return the histogram along the given dimension(s).

        Parameters
        ----------
        dim : str
            The dimension name.
        bins : int or list of float, optional
            The bin boundaries or the integer number of bins from the minimum datum to
            the maximum datum. Default is ``11``.
        """
        data = self.data
        if bins is None:
            bins = 11
        if isinstance(bins, numbers.Integral):
            bins = np.linspace(np.nanmin(data.data), np.nanmax(data.data), bins)
        else:
            bins = bins.copy()
        data = var.hist(bins, data, dim=dim)
        if 'track' in data.dims:
            data = data.sum(dim='track')
        data.climo.add_cell_methods({dim: 'hist'})
        return data

    def mask(self, mask, dataset=None):
        """
        Mask out data according to a preset pattern.
        """
        # TODO: Expand this function
        # NOTE: DataArray math operations ignore NaNs by default
        data = self.data
        if mask is not None:
            if dataset is None:
                raise ValueError('Dataset required for applying preset masks.')
            elif mask in ('pv', 'trop'):
                pv = dataset.climo['pv']
                mask = (pv.data >= 0.1 * ureg.PVU) & (pv.data <= 2.0 * ureg.PVU)
                if mask == 'pv':  # mask region os positive poleward PV gradient
                    lat = dataset.climo['latitude']
                    dpvdy = dataset.climo['dpvdy']
                    mask = (
                        mask
                        & (dpvdy.data > 0 * ureg('PVU / km'))
                        & (np.abs(lat) >= 30 * ureg.deg)
                        & (np.abs(lat) <= 60 * ureg.deg)
                    )
            elif mask in ('land', 'sea'):
                raise NotImplementedError
            else:
                raise ValueError(f'Unknown mask preset {mask!r}.')
            data = data.copy(deep=True)
            data[~mask] = np.nan * getattr(data.data, 'units', 1)
        return data

    @_keep_tracked_attrs
    def normalize(self):
        """
        Normalize with respect to time.
        """
        data = self.data
        data = data / data.cf.mean('time')
        data.climo.add_cell_methods({'time': 'normalized'})
        return data

    @_keep_tracked_attrs
    def slope(self, dim):
        """
        Return a best-fit estimate of the slope.
        """
        data = self.data
        data, _, _ = var.linefit(data.coords[dim], data)
        data.climo.add_cell_methods({dim: 'slope'})
        return data

    def quantify(self):
        """
        Convert data and coordinates to `pint.Quantity` using the units attribute.
        If the units attribute is missing, a warning is issued.
        """
        # WARNING: In-place conversion resulted in endless bugs related to
        # ipython %autoreload, was departure from metpy convention, was possibly
        # confusing for users, and not even sure if faster. So abandoned this.
        data = self.data.copy(deep=True)
        if not isinstance(data.data, pint.Quantity) and _is_numeric(data.data):
            if 'units' in data.attrs:
                data.data = data.data * self.units
            else:
                warnings._warn_climopy(
                    f'Failed to quantify {data.name=} (units attribute not found).'
                )
            data.attrs.pop('units', None)
        return data

    def dequantify(self):
        """
        Convert data and coordinates to magnitude with units as an attribute.
        """
        # WARNING: Try to preserve *order* of units for fussy formatting later on.
        # Avoid default alphabetical sorting by pint.__format__.
        data = self.data.copy(deep=True)
        if isinstance(self.data.data, pint.Quantity):
            data.data = data.data.magnitude
            data.attrs['units'] = encode_units(self.units)
        return data

    @_while_quantified
    def to_units(self, units):
        """
        Return a copy converted to the desired units. Supports CF compliant constructs
        like `'m2'` for "meters squared" and `'100km'` for "100 kilometers."
        """
        if not self._is_quantity:
            raise ValueError('Data should be quantified.')
        data = self.data.copy(deep=True)
        if isinstance(units, str):
            units = parse_units(units)
        try:
            data.data = data.data.to(units, 'climo')  # use climopy context
        except Exception:
            raise RuntimeError(
                f'Failed to convert {data.name!r} from current units '
                f'{self.units!r} to units {units!r}.'
            )
        return data

    @_while_quantified
    def to_base_units(self, coords=False):
        """
        Return a copy converted to base units.
        """
        # NOTE: assign_coords has issues with multiple DataArray values. See:
        # https://github.com/pydata/xarray/issues/3483
        data = self.data.copy(deep=True)
        data.data = data.data.to_base_units()
        if coords:
            data = data.assign_coords({
                dim: data.climo.coords[dim].climo.to_base_units().climo.dequantify()
                .variable for dim in data.coords
            })
        return data

    @_while_quantified
    def to_compact_units(self, coords=False):
        """
        Return a copy converted to compact units.
        """
        # NOTE: assign_coords has issues with multiple DataArray values. See:
        # https://github.com/pydata/xarray/issues/3483
        data = self.data.copy(deep=True)
        data.data = data.data.to_compact_units()
        if coords:
            data = data.assign_coords({
                dim: data.climo.coords[dim].climo.to_compact_units().climo.dequantify()
                .variable for dim in data.coords
            })
        return data

    def to_standard_units(self, coords=False):
        """
        Return a copy converted to standardized units. This will only work if
        the variable name matches a valid cfvariable.
        """
        # NOTE: assign_coords has issues with multiple DataArray values. See:
        # https://github.com/pydata/xarray/issues/3483
        units = self._get_cf_attr('standard_units')
        if units is None:  # unspecified "standard" units
            units = self.units  # just convert to current units
        try:
            data = self.to_units(units)
        except Exception:
            raise RuntimeError(
                f'Failed to convert {self.data.name!r} from current units '
                f'{self.units!r} to standard units {units!r} with '
                f"reduction methods {self.data.attrs.get('cell_methods')!r}."
            )
        if coords:
            data = data.assign_coords({
                dim: data.climo.coords[dim].climo.to_standard_units().climo.dequantify()
                .variable for dim in data.coords
            })
        return data

    @_while_quantified
    def to_variable(self, dest, standardize=False, **kwargs):
        """
        Transform this variable to another variable using two-way transformations
        registered with `register_transformation`. Transformations work recursively,
        i.e. definitions for A-->B and B-->C permit transforming A-->C.

        Parameters
        ----------
        dest : str
            The destination variable.
        standardize : bool, optional
            Whether to standardize the units afterward.
        **kwargs
            Passed to the transformation function.
        """
        data = self.data
        if data.name is None:
            raise RuntimeError('DataArray name required to do transformation.')
        func = _find_this_transformation(data, dest)
        if func is None:
            raise ValueError(f'Transformation {data.name!r} --> {dest!r} not found.')
        with xr.set_options(keep_attrs=False):  # ensure invalid attributes are lost
            param = func(data, **kwargs)
        param.name = dest
        if standardize:
            param = param.climo.to_standard_units()
        return param

    @_keep_tracked_attrs
    def timescale(self, dim, maxlag=None, imaxlag=None, **kwargs):
        """
        Return a best-fit estimate of the autocorrelation timescale.

        Parameters
        ----------
        dim : str, optional
            The dimension.
        **kwargs
            Passed to `climopy.rednoisefit`.
        """
        data = self.data
        lag = data.coords[dim]
        if maxlag is None and imaxlag is None:
            maxlag = 50.0  # default value is 50 days
        if dim != 'lag':
            lag, data = var.autocorr(lag, data, maxlag=maxlag, imaxlag=imaxlag)
        data, _, _ = var.rednoisefit(lag, data, maxlag=maxlag, imaxlag=imaxlag, **kwargs)  # noqa: E501
        data.climo.add_cell_methods({dim: 'timescale'})
        return data

    def _cf_repr(self, brackets=True, maxlength=None, fixedwidth=None):
        """
        Get representation even if `cfvariable` is not present.
        """
        if var := getattr(self, 'cfvariable', None):
            repr_ = re.match(r'\A.*?\((.*)\)\Z', repr(var)).group(1)
        else:
            repr_ = self.data.name or 'unknown'
        if fixedwidth is not None and (m := re.match(r'\A(\w+),\s*(.*)\Z', repr_)):
            grp1, grp2 = m.groups()  # pad between canonical name and subsequent info
            repr_ = grp1[:fixedwidth] + ',' + ' ' * (fixedwidth - len(grp1)) + grp2
        if maxlength is not None and len(repr_) > maxlength:
            repr_ = repr_[:maxlength - 4]
            repr_ = repr_[:repr_.rfind(' ')] + ' ...'
        if brackets:
            repr_ = re.sub(r'\A(\w+),(\s*)(.*)\Z', r'\1\2<\3>', repr_)
        return repr_

    @property
    def _is_quantity(self):
        """
        Return whether data is quantified.
        """
        return isinstance(self.data.data, pint.Quantity)

    @property
    def _has_units(self):
        """
        Return whether 'units' attribute exists or data is quantified.
        """
        return 'units' in self.data.attrs or self._is_quantity

    @property
    def cfvariable(self):
        """
        Return a `CFVariable` based on the DataArray name, the scalar coordinates,
        and the coordinate reductions referenced in `cell_methods`.
        """
        data = self.data
        if data.name is None:
            raise AttributeError('DataArray name required to get cfvariable.')
        kwargs = {key: val for key, val in data.attrs.items() if key in CFVARIABLE_ARGS}
        methods = self._decode_cf_attr(data.attrs.get('cell_methods', ''))
        for dim, da in data.coords.items():
            if da.size > 1:
                continue
            coord = self._to_cf_coord_name(da.name)
            if not coord:
                continue
            if not da.isnull():  # selection of single coordinate
                units = parse_units(da.units) if 'units' in da.attrs else 1
                kwargs[coord] = units * da.item()
            elif any(da.name in d for d, m in methods):
                kwargs[coord] = tuple(m for d, m in methods if da.name in d)
            # else:
            #     warnings._warn_climopy(f'Unknown cell method for {coord.name!r}.')
        try:
            return self.registry(data.name, accessor=self, **kwargs)
        except KeyError:
            raise AttributeError(f'CFVariable not found for name {data.name!r}.')

    @property
    def magnitude(self):
        """
        The magnitude of the data values of this DataArray (i.e., without units).
        """
        if isinstance(self.data.data, pint.Quantity):
            return self.data.data.magnitude
        else:
            return self.data.data

    @property
    def quantity(self):
        """
        The data values of this DataArray as a `pint.Quantity`.
        """
        if isinstance(self.data.data, pint.Quantity):
            return self.data.data
        else:
            return ureg.Quantity(self.data.data, self.units)

    @property
    def units(self):
        """
        The units of this DataArray as a `pint.Unit`, taken from the underlying
        `pint.Quantity` or the ``units`` attribute. See `parse_units` for details.
        """
        if isinstance(self.data.data, pint.Quantity):
            return self.data.data.units
        elif 'units' in self.data.attrs:
            return parse_units(self.data.attrs['units'])
        else:
            raise RuntimeError('Units not present in attributes or as pint.Quantity.')

    @property
    def units_latex(self):
        """
        Units label formatted for matplotlib.
        """
        return latex_units(self.units)


@xr.register_dataset_accessor('climo')
class ClimoDatasetAccessor(ClimoAccessor):
    """
    Accessor for xarray datasets.
    """
    _cls_groupby = _DatasetGroupByQuantified
    _cls_coords = _DatasetCoordsQuantified
    _cls_loc = _DatasetLocIndexerQuantified
    _cls_cf = _cf_accessor.CFDatasetAccessor

    def __repr__(self):
        pad = 4
        data = self.data
        rows = ['<climopy.ClimoDatasetAccessor>']
        width = max(
            (
                len(da.name) for grp in (data, data.coords) for da in grp.values()
                if isinstance(da.name, str)
            ), default=10
        )
        for row, src in zip(('Coordinates:', 'Data variables:'), (data.coords, data)):
            if not src:
                continue
            rows.append(row)
            rows.extend(
                pad * ' ' + da.climo._cf_repr(maxlength=88 - pad, fixedwidth=width + 2)
                for da in src.values()
            )
        return '\n'.join(rows)

    def __init__(self, dataset, registry=vreg):
        """
        Parameters
        ----------
        dataset : xarray.Dataset
            The data.
        registry : CFVariableRegistry
            The active registry used to look up variables with `.cfvariable`.
        """
        self._data = dataset
        self._registry = registry

    def __contains__(self, key):
        """
        Is a dataset variable or derived coordinate.
        """
        return self._get_item_or_func(key) is not None

    def __getattr__(self, attr):
        """
        Return a possibly derived parameter with `coords` or `param`. Use the
        `cf_xarray` accessor to search for parameters by CF names.
        """
        if attr[:1] == '_':
            return super().__getattribute__(attr)  # trigger builtin AttributeError
        try:
            return self[attr]
        except KeyError:
            return super().__getattribute__(attr)  # raises builtin error

    @_keep_tracked_attrs
    def __getitem__(self, key):
        """
        Return a coordinate, variable, or transformed or derived variable registered
        with `register_transformation` or `register_derivation`. Translates CF axis,
        coordinate, and standard names and CFVariableRegistry identifiers.
        """
        return self._get_item(key)  # with weights attached

    def _get_item(self, key, add_cell_measures=True):
        """
        Return a quantified DataArray with weights optionally added. This is separated
        from _get_item_or_func to facillitate fast __contains__.
        """
        # Retrieve quantity
        da = self._get_item_or_func(key)
        if da is None:
            raise KeyError(f'Invalid variable name {key!r}.')
        if callable(da):
            da = da()  # ta-da!
        data = da.climo.quantify()  # should already be quantified, but just in case
        data.name = data.name or 'unknown'  # just in case
        if add_cell_measures and key not in self.coords:
            data = data.climo.add_cell_measures(dataset=self.data)

        return data

    def _get_item_or_func(self, key, use_registry=True):
        """
        Return a DataArray or function that generates the DataArray.
        """
        # Return either a DataArra or callable function
        # NOTE: This lets us implement a quick __contains__ that works on derived vars
        da = None
        data = self._data
        if tup := _find_any_transformation(data.values(), key):
            da = functools.partial(*tup)  # DataArray is passed with partial func
        elif func := _find_derivation(key):
            da = functools.partial(func, self)
        elif (da := self.coords.get(key)) is not None:
            pass
        elif (da := self.vars.get(key)) is not None:
            pass
        elif use_registry:
            var = self._registry.get(key)
            identifiers = var.identifiers if var else ()
            for name in identifiers:
                da = self._get_item_or_func(name, use_registry=False)
                if da is not None:
                    return da

        # Adjust final name, removing special suffixes
        # TODO: Add robust method for automatically removing dimension reduction
        # suffixes from variable names.
        if isinstance(da, xr.DataArray):
            regex = r'\A(.*?)(_zonal|_horizontal|_atmosphere)?(_timescale|_autocorr)?\Z'
            da.name = re.sub(regex, r'\1', da.name)

        return da

    def add_variable(self, *args, **kwargs):
        """
        Call `get_variable` and add the result to the dataset.
        """
        data = self.data.copy(deep=False)
        kwargs.setdefault('add_cell_measures', False)
        da = self.get_variable(*args, **kwargs)
        if da.name in self.data:
            raise RuntimeError(f'Quantity {da.name!r} already present in dataset.')
        data[da.name] = da
        return data

    @_expand_variable_args  # standardize args are passed to lookup cache
    # @functools.lru_cache(maxsize=64)  # TODO: fix issue where recursion breaks cache
    @_keep_tracked_attrs
    def get_variable(
        self,
        *keys,
        add_cell_measures=True,
        normalize=False,
        quantify=None,
        units=None,
        running=None,
        standardize=False,
        **kwargs
    ):
        """
        Call `.__getitem__`, with optional post-processing steps and special behavior
        when variables are prefixed or suffixed with certain values.

        Parameters
        ----------
        arg : var-spec or 2-tuple thereof
            The variable name. The following prefix and suffix shorthands
            are supported:

            * Prepend ``abs_`` to return the absolute value of the result.
            * Append ``_lat`` or ``_strength`` to return vertically and zonally
              integrated or maximum energy and momentum budget terms.
            * Append ``_1``, ``_2``, or ``_diff`` to make a selection or take an
              anomaly pair difference using `.sel_pair`.

            You can also pass a 2-tuple to return the difference between two variables.
            And all names can be replaced with 2-tuples of the form ('name', kwargs)
            to pass keyword arguments positionally.
        add_cell_measures : bool, optional
            Whether to add default cell measures to the coordinates.
        normalize : bool, optional
            Whether to normalize the resulting data by the time-mean.
        quantify : bool, optional
            Whether to quantify the data with `.quantify()`.
        units : unit-like, optional
            Convert the result to these units using `.to_units()`.
        running : bool, optional
            Apply a running mean with `.runmean`.
        standardize : bool, optional
            Whether to standardize the resulting units with `.to_standard_units`.
        **kwargs
            Passed to `reduce`. Used to reduce the dimensions.

        Returns
        -------
        data : xarray.DataArray
            The data.
        """
        # Parse positional arguments and optionally return an anomaly
        # NOTE: Derivations are for variables for which user-specified reductions
        # come *after* all the other math. For things like forced-unforced anomaly
        # the reduction comes *before* the math (difference), so put in get_variable.
        if len(keys) == 1:
            key = keys[0]
        elif len(keys) == 2:
            data1 = self.get_variable(keys[0], **kwargs)
            data2 = self.get_variable(keys[1], **kwargs)
            with xr.set_options(keep_attrs=True):
                data = data1 - data2
            data.name = data1.name
            return data
        else:
            raise ValueError(f'Invalid variable spec {keys!r}.')

        # Get the variable, translating meta-variable actions
        # NOTE: This supports e.g. edsef_strength_anom
        regex = r'\A(abs_)?(.*?)(_lat|_strength)?(_1|_2|_anom)?\Z'
        abs, key, reduce, pair = re.match(regex, key).groups()
        data = self._get_item(key, add_cell_measures=add_cell_measures)

        # Automatically determine 'reduce' kwargs for energy and momentum budget
        # NOTE: For tendency 'strength' terms we integrate over lons and lats
        # WARNING: Flux convergence terms are subgroups of flux terms, not tendency
        if reduce:
            reduce = reduce.strip('_')
            content = key in vreg['energy'] or key in vreg['momentum']
            tendency = key in vreg['energy_flux'] or key in vreg['acceleration']
            transport = key in vreg['meridional_energy_flux'] or key in vreg['meridional_momentum_flux']  # noqa: E501
            if not content and not transport and not tendency:
                raise ValueError(f'Invalid parameter {key!r}.')
            lon = 'int' if tendency and reduce == 'strength' or transport else 'avg'
            if tendency:  # e.g. magnitude of cooling over polar cap
                lat = 'int' if reduce == 'strength' else 'argzero'
            else:
                lat = 'absmax' if reduce == 'strength' else 'absargmax'
            kwargs = {'lon': lon, 'lev': 'int', 'lat': lat, **kwargs}
            if data.sizes.get('lev', 1) == 1:  # horizontal grid
                del kwargs['lev']

        # Reduce dimensionality using keyword args
        # NOTE: For timescale variables take inverse before and after possible average
        for dim in tuple(kwargs):
            if dim in self.dims and dim not in data.dims:
                warnings._warn_climopy(f'Dim {dim!r} was already reduced for {key!r}.')
                del kwargs[dim]
        if kwargs:
            invert = key in vreg.timescale or key in vreg.tau
            if invert:
                with xr.set_options(keep_attrs=True):
                    data = 1.0 / data
                warnings._warn_climopy(f'Taking inverse reduced inverse of {key!r}.')
            try:
                data = data.climo.reduce(dataset=self.data, **kwargs)
            except Exception:
                raise ValueError(f'Failed to reduce data {key!r} with kwargs {kwargs}.')
            if invert:
                with xr.set_options(keep_attrs=True):
                    data = 1.0 / data

        # Normalize the data
        if normalize:
            data = data.climo.normalize()

        # Take the rolling mean
        if running:
            data = data.climo.running(time=running)

        # Take the absolute value, accounting for attribute-stripping xarray bug
        if abs:
            data = _keep_tracked_attrs(np.abs)(data)

        # Select pair only *after* doing all the math. This is a convenient way
        # to get difference between reduced values
        if pair:
            data = data.climo.sel_pair(pair.strip('_'))

        # Change the units
        if units is not None:  # permit units='' to translate to dimensionless
            data = data.climo.to_units(units)
        elif standardize:
            data = data.climo.to_standard_units()

        # Quantify or dequantify data
        if quantify is not None:
            if quantify:  # should *already* be quantified but just to make sure
                data = data.climo.quantify()
            else:
                data = data.climo.dequantify()

        # Re-order spectral dimensions for 3 plot types: YZ, CY, YK, and CK (with
        # row-major ordering), or simply preserve original dimension order based on
        # dimension order that appears in dataset variables.
        # See: https://github.com/pydata/xarray/issues/2811#issuecomment-473319350
        if 'k' in data.dims:
            dims = ('lev', 'k', 'lat', 'c')
        else:
            dims = _first_unique(d for v in self.data.values() for d in v.dims)
        data = data.transpose(..., *(d for d in dims if d in data.dims))

        return data

    def quantify(self):
        """
        Convert all xarray.DataArray data into pint Quantities, excluding bounds.
        """
        return self.data.map(
            lambda da: da if self._is_bounds(da.name) else da.climo.quantify(),
            keep_attrs=True
        )

    def dequantify(self):
        """
        Convert all xarray.DataArray data into normal arrays.
        """
        return self.data.map(lambda da: da.climo.dequantify(), keep_attrs=True)

    @property
    def vars(self):
        """
        Variable analogue to `.coords`. Indexing this always returns quantified arrays.
        """
        return _VarsQuantified(self.data, self.registry)

    def _is_bounds(self, key):
        """
        Return if variable is a coordinate bounds.
        """
        coords = self.data.coords
        sentinel = object()
        return any(key == da.attrs.get('bounds', sentinel) for da in coords.values())


def register_derivation(spec, /, overwrite=True):
    """
    Register a function that derives one variable from one or more others, for use
    with `ClimoDatasetAccessor.get_variable`. All derivations are carried out with
    data arrays quantified by `pint`. Input can be a string or compiled regular
    expression pattern (in the latter case the function must also accept a `name`
    keyword argument, and the behavior of the derivation should depend on the name).

    Parameters
    ----------
    spec : str, tuple, or re.Pattern
        The destination variable name, a tuple of valid destination names, or an
        `re.compile`'d pattern matching a set of valid destination names. In the latter
        two cases, the derivation function must accept a `name` keyword argument.
    overwrite : bool, optional
        Whether to overwrite existing transformations. ``True`` by default.

    Examples
    --------
    >>> import climopy as climo
    ... from climopy import const
    ... @climo.register_derivation('pt')
    ... def potential_temp(self):
    ...     return self['t'] * (const.p0 / self['p']).climo.to_units('') ** (2 / 7)
    """
    if not isinstance(spec, (str, tuple, re.Pattern)):
        raise TypeError(f'Invalid name or regex {spec!r}.')

    def _decorator(func):  # noqa: E306
        # Possibly overwrite
        if spec in DERIVATIONS:
            if overwrite:
                raise TypeError(f'Derivation of {spec!r} already registered.')
            else:
                warnings._warn_climopy(f'Overwriting existing derivation {spec!r}.')

        # Wrap function to assign a DataArray name. Also ensure we use the
        # registered name rather than a CF-style alias
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            data.name = spec if isinstance(spec, str) else kwargs.get('name', None)
            return data

        DERIVATIONS[spec] = _wrapper

    return _decorator


def register_transformation(src, dest, /, *, overwrite=True):
    """
    Register a function that transforms one variable to another, for use with
    `ClimoDataArrayAccessor.to_variable`. Transformations should depend only on the
    initial variable and (optionally) the coordinates. All transformations are
    carried out with data arrays quantified by `pint`.

    Parameters
    ----------
    src : str
        The source variable name.
    dest : str, tuple, or re.Pattern
        The destination variable name, a tuple of valid destination names, or an
        `re.compile`'d pattern matching a set of valid destination names. In the latter
        two cases, the transformation function must accept a `name` keyword argument.
    overwrite : bool, optional
        Whether to overwrite existing transformations. ``True`` by default.

    Examples
    --------
    >>> import climopy as climo
    ... from climopy import const
    ... @climo.register_transformation('lat', 'y')
    ... def meridional_coordinate(da):
    ...     return (da * const.a).climo.to_units('km')  # implicit deg-->rad conversion
    """
    if not isinstance(src, str):
        raise ValueError(f'Invalid source {src!r}. Must be string.')
    if not isinstance(dest, (str, tuple, re.Pattern)):
        raise ValueError(f'Invalid destination {dest!r}. Must be string, tuple, regex.')

    def _decorator(func):
        # Possibly overwrite
        if (src, dest) in TRANSFORMATIONS:
            if overwrite:
                raise TypeError(f'Transformation {src!r}->{dest!r} already registered.')
            else:
                warnings._warn_climopy(f'Overwriting existing {src!r}->{dest!r} transformation.')  # noqa: E501

        # Wrap function to assign a DataArray name. Also ensure we use the
        # registered name rather than a CF-style alias
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            data.name = dest if isinstance(dest, str) else kwargs.get('name', None)
            return data

        TRANSFORMATIONS[(src, dest)] = _wrapper

    return _decorator


def _matching_function(key, func, name):
    """
    Return the function if the input string matches a string, tuple, or regex key. In
    the latter two cases a `name` keyword arguments is added with `functools.partial`.
    """
    if isinstance(key, str) and name == key:
        return func
    if isinstance(key, tuple) and name in key:
        return functools.partial(func, name=name)
    if isinstance(key, re.Pattern) and key.match(name):
        return functools.partial(func, name=name)


def _find_derivation(dest):
    """
    Find derivation that generates the variable name. Return `None` if not found.
    """
    for idest, derivation in DERIVATIONS.items():
        if func := _matching_function(idest, derivation, dest):
            return func


def _find_any_transformation(data_arrays, dest):
    """
    Find transformation that generates the variable name. Return `None` if not found.
    Otherwise return the generating function and a source variable.
    """
    for data_array in data_arrays:
        if func := _find_this_transformation(data_array, dest):
            return func, data_array


def _find_this_transformation(src, dest, error=False, registry=None):
    """
    Find possibly nested series of transformations that get from variable A --> C.
    Account for CF and CFVariableRegistry names.
    """
    # TODO: Also support inverse transformations?
    # First get list of *source* identifiers
    if isinstance(src, str):
        identifiers = [src]
        if registry and (var := registry.get(src)):
            identifiers.extend(var.identifiers)
    else:
        # Internally, we only pass registry keyword arg on recursion; retrieve
        # it from the accessor on first pass.
        registry = registry or src.climo.registry
        identifiers = [src.name]
        if 'standard_name' in src.attrs:
            identifiers.append(src.attrs['standard_name'])
        if src.climo._has_cf_attr('identifiers'):
            identifiers.extend(src.climo._get_cf_attr('identifiers') or ())
    # Next find the transformation
    if dest in identifiers:
        return lambda da: da.copy()
    for (isrc, idest), transformation in TRANSFORMATIONS.items():
        if isrc not in identifiers:
            continue
        if func := _matching_function(idest, transformation, dest):  # allow regexes
            return func
        # Perform nested invocation of transformations. Inner func goes from
        # A --> ?, then outer func from ? --> B (returned above)
        if outer := _find_this_transformation(idest, dest, error=True, registry=registry):  # noqa: E501
            return lambda da, **kwargs: outer(transformation(da, **kwargs))


# Register coordinate-related transformations and derivations
register_transformation('latitude', 'meridional_coordinate')(
    lambda da: (da * const.a).climo.to_units('km')
)
register_transformation('latitude', 'cosine_latitude')(
    lambda da: np.cos(da)
)
register_transformation('latitude', 'coriolis_parameter')(
    lambda da: 2 * const.Omega * np.sin(da)
)
register_transformation('latitude', 'beta_parameter')(
    lambda da: np.abs(2 * const.Omega * np.cos(da)) / const.a
)
register_transformation('vertical', 'mass')(
    lambda da: (da / const.g).climo.to_units('kg m^-2')
)
register_transformation('vertical', 'reference_height')(
    lambda da: (const.H * np.log(const.p0 / da)).climo.to_units('km')
)
register_transformation('vertical', 'reference_density')(
    lambda da: (da / (const.Rd * ureg('300K'))).climo.to_units('kg m^-3')
)
register_transformation('vertical', 'reference_potential_temperature')(
    lambda da: ureg('300K') * (const.p0 / da).climo.to_units('') ** const.kappa
)


@register_derivation('cell_width')
def _cell_width(self):
    coord = const.a * self.coords['cosine_latitude'] * self.coords['longitude_del']
    return coord.climo.to_units('km')


@register_derivation('cell_depth')
def _cell_depth(self):
    coord = const.a * self.coords['latitude_del']
    return coord.climo.to_units('km')


@register_derivation('cell_duration')
def _cell_duration(self):
    coord = self.coords['time_del']
    return coord.climo.to_units('days')


@register_derivation('cell_height')
def _cell_height(self):
    # WARNING: Must use _get_item with add_cell_measures=False to avoid recursion
    vertical = self.vertical_type
    if vertical == 'temperature':
        data = self.vars['pseudo_density'] * self.coords['vertical_del']
    elif vertical == 'pressure':
        ps = None
        data = self.coords['vertical_del']
        for candidate in ('surface_air_pressure', 'air_pressure_at_mean_sea_level'):
            if candidate not in self.vars:
                continue
            ps = self.vars[candidate]
            loc = {'vertical': np.max(self.coords['vertical'].data)}
            bott = ps - self.coords['vertical_bot'].climo.sel(loc)
            data = data + 0 * ps  # conform shape
            data.climo.loc[loc] = bott
        if ps is None:
            warnings._warn_climopy(
                'Could not find surface pressure for accurate mass weighting.'
            )
        data = data / const.g
    elif vertical == 'hybrid':
        raise NotImplementedError(
            'Cell height for hybrid coordinates not yet implemented.'
        )
    else:
        raise NotImplementedError(
            f'Unknown cell height for vertical type {vertical!r}.'
        )
    data = data.climo.to_units('kg m^-2')
    zero = 0 * ureg.kg / ureg.m ** 2
    data.data[data.data < zero] = zero
    data = data.fillna(0)
    return data
