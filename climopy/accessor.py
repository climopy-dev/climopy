#!/usr/bin/env python3
"""
A pair of `xarray accessors \
<http://xarray.pydata.org/en/stable/internals.html#extending-xarray>`__
for working with pint units and CF variables, calculating transformed and derived
quantities, and reducing dimensions in myrid ways.
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

from . import DERIVATIONS, TRANSFORMATIONS, const, diff, spectral, utils, var
from .cfvariable import CFVariableRegistry, vreg
from .internals import _make_stopwatch  # noqa: F401
from .internals import ic  # noqa: F401
from .internals import _first_unique, docstring, quack, warnings
from .unit import (
    _latitude_units,
    _longitude_units,
    decode_units,
    encode_units,
    format_units,
    ureg,
)

__all__ = [
    'ClimoAccessor',
    'ClimoDataArrayAccessor',
    'ClimoDatasetAccessor',
    'register_derivation',
    'register_transformation',
]

# Attributes passed to cfvariable
CFVARIABLE_ARGS = (
    'long_name',
    'short_name',
    'standard_name',
    'long_prefix',
    'long_suffix',
    'short_prefix',
    'short_suffix',
)

# Arguments passed to parse key
PARSEKEY_ARGS = (
    'add_cell_measures'
    'search_vars',
    'search_coords',
    'search_derivations',
    'search_transformations',
    'search_registry'
)

# Regular expressions compiled for speed
REGEX_BOUNDS = re.compile(r'\A(.*?)(?:_(top|bot(?:tom)?|del(?:ta)?|b(?:ou)?nds))?\Z')  # coordinate suffixes implying changes  # noqa: E501
REGEX_IGNORE = re.compile(r'\A(.*?)(_zonal|_horizontal|_atmosphere)?(_timescale|_autocorr)?\Z')  # variable suffixes to ignore  # noqa: E501
REGEX_MODIFY = re.compile(r'\A(abs_)?(.*?)(_latitude|_strength)?(_1|_2|_anomaly|_ratio)?\Z')  # variable suffixes implying changes  # noqa: E501
REGEX_REPR_COMMA = re.compile(r'\A(\w+),(\s*)(.*)\Z')  # content around first comma
REGEX_REPR_PAREN = re.compile(r'\A.*?\((.*)\)\Z')  # content inside first parentheses

# Custom cell measures and associated coordinates. Naming conventions are consistent
# with existing 'cell' style names and avoid conflicts with axes names / standard names.
# NOTE: The width * depth = area and width * depth * height = volume
# TODO: The height cell measures are generally mass-per-unit-area implying volume
# is more like mass but may want to support actual volume and height in future.
CELL_MEASURE_COORDS = {
    'length': ('longitude',),
    'width': ('latitude',),
    'height': ('vertical',),
    'duration': ('time',),
    'area': ('longitude', 'latitude'),
    'volume': ('longitude', 'latitude', 'vertical'),
}
COORD_CELL_MEASURE = {
    coords[0]: m for m, coords in CELL_MEASURE_COORDS.items() if len(coords) == 1
}
if hasattr(_cf_accessor, '_CELL_MEASURES'):
    _cf_accessor._CELL_MEASURES = tuple(CELL_MEASURE_COORDS)
else:
    warnings._warn_climopy('cf_xarray API changed. Cannot update cell measures.')

# Expand regexes for automatic coordinate detection with standardize_coords
# NOTE: The new 'vertical' regex covers old options 'nav_lev', 'gdep', 'lv_', '[o]*lev',
# 'depth' and the time regex includes new matches 'date', 'datetime', 'lag'.
# NOTE: Suffixes are irrelevant because we only test re.match without end word
# or end string atoms. The old regex dictionary needlessly includes suffixes.
_cf_regex = getattr(_cf_accessor, 'regex', None)
if _cf_regex:
    _cf_regex.update({
        'longitude': '(?:x|nav_)?(?:lon|g?lam)',
        'latitude': '(?:y|nav_)?(?:lat|g?phi)',
        'vertical': (
            r'z\Z|(?:[a-z_]+)?(?:le?v|g?dep)|pres|sigma|h(?:ei)?ght|altitude'
            r'|isobar(?:ic)?|isotherm(?:al)?|isentrop(?:e|ic)|top_bottom|bottom_top'
        ),
        'time': r't\Z|time|date|datetime|lag|min|hour|day|week|month|year',
        'X': r'x\Z|i\Z|n(?:lon|phi|i)',
        'Y': r'y\Z|j\Z|n(?:lat|phi|j)',
    })
    _cf_regex['Z'] = _cf_regex['vertical']
    _cf_regex['T'] = _cf_regex['time']
    if all(isinstance(_, re.Pattern) for _ in _cf_accessor.regex.values()):
        _cf_regex.update({key: re.compile(value) for key, value in _cf_regex.items()})
    elif not all(isinstance(_, str) for _ in _cf_regex.values()):
        _cf_regex = None
if not _cf_regex:
    warnings._warn_climopy('cf_xarray API changed. Cannot update coordinate regexes.')

# Expand regexes for coordinate detection
# NOTE: The default criteria only sees 'degree_E' not 'deg_E', 'degree_east' not
# 'degree_East'. This is annoying so we make it as flexible as custom unit definition.
_cf_criteria = getattr(_cf_accessor, 'coordinate_criteria', None)
if _cf_criteria:
    for key, opt in zip(('longitude', 'latitude'), (_longitude_units, _latitude_units)):
        values = [s.strip() for i, s in enumerate(opt.split('=')) if i not in (1, 2)]
        _cf_criteria[key]['units'] = tuple(values)  # skips definition and abbreviation
if not _cf_criteria:
    warnings._warn_climopy('cf_xarray API changed. Cannot update coordinate criteria.')


# Mean and average templates
_template_meansum = """
Return the %(operator)s along dimension(s), preserving attributes and coordinates.

Parameters
----------
dim : str or sequence of str, optional
    The dimensions.
skipna : bool, optional
    Whether to skip NaN values.
weight : xr.DataArray, optional
    Optional weighting.
**kwargs
    Passed to `~ClimoAccessor.truncate`. Used to limit bounds of %(operator)s.
"""
_template_avgint = """
Return the mass-weighted %(operator)s.

Parameters
----------
dim : dim-spec or {'area', 'volume'}, optional
    The %(action)s dimension(s). Weights are applied automatically using cell
    measure variables stored in the coodinates and referenced by the
    `cell_measures` attribute (see `~ClimoAccessor.add_cell_measures`). If not
    specified, the entire 3-dimensional domain is used.
weight : xr.DataArray, optional
    Optional additional weighting.
skipna : bool, optional
    Whether to skip NaN values.
**kwargs
    Passed to `~ClimoAccessor.truncate`. Used to limit the bounds of the
    %(action)s.
"""
_template_cumavgint = """
Return the cumulative mass-weighted %(operator)s.

Parameters
----------
dim : dim-spec
    The %(action)s dimension. Weights are applied automatically using cell
    measure variables stored in the coodinates and referenced by the
    `cell_measures` attribute (see `~ClimoAccessor.add_cell_measures`).
skipna : bool, optional
    Whether to skip NaN values.
reverse : bool, optional
    Whether to change the direction of the accumulation to right-to-left.
**kwargs
    Passed to `~ClimoAccessor.truncate`. Used to limit bounds of integration.
"""
docstring.snippets['template_meansum'] = _template_meansum
docstring.snippets['template_avgint'] = _template_avgint
docstring.snippets['template_cumavgint'] = _template_cumavgint

# Mean and average notes
_notes_avgmean = """
ClimoPy makes an artifical distinction between the "mean" as a naive, unweighted
average and the "average" as a cell measures-aware, mass-weighted average.
"""
_notes_weighted = """
This was added as a dedicated accessor function rather than creating a
custom `~xarray.core.weighted.Weighted` object because the selection of
mass weights depends on the dimension(s) passed by the user.
"""
docstring.snippets['notes_avgmean'] = _notes_avgmean
docstring.snippets['notes_weighted'] = _notes_weighted

# Extrema templates
_template_minmax = """
Return the %(prefix)s local %(extrema)s along the dimension. Multiple %(extrema)s are
concatenated along a 'track' dimension.

Parameters
----------
dim : str, optional
    The dimension. This is replaced with a ``'track'`` dimension on the output
    `~xarray.DataArray`.
dim_track : str, optional
    The dimension along which %(extrema)s are grouped into lines and tracked with
    `~.utils.linetrack`.
**kwargs
    Passed to `~.utils.find` or `~ClimoAccessor.truncate`.
"""
_template_absminmax = """
Return the %(prefix)s global %(extrema)s along the dimension.

Parameters
----------
dim : str, optional
    The dimension.
**kwargs
    Passed to `~.utils.find` or `~ClimoAccessor.truncate`.
"""
_template_argloc = """
Return the coordinate(s) of a given value along the dimension.

Parameters
----------
dim : str, optional
    The dimension.
value : int, optional
    The value we are searching for. Default is ``0``.
dim_track : str, optional
    The dimension along which coordinates are grouped into lines and tracked with
    `~.utils.linetrack`.
**kwargs
    Passed to `~.utils.find` or `~ClimoAccessor.truncate`.
"""
docstring.snippets['template_minmax'] = _template_minmax
docstring.snippets['template_absminmax'] = _template_absminmax
docstring.snippets['template_argloc'] = _template_argloc

# Differentiation
_template_divcon = r"""
Return the spherical meridional %(operator)s. To calculate the %(operator)s at the
poles, the numerator is assumed to vanish and l'Hopital's rule is invoked.

Parameters
----------
cos_power : int, optional
    Exponent to which the cosines in the numerator and denominator is raised.
    Default is ``1``, but the contribution of the angular momentum flux convergence
    to the zonal wind budget requires ``2`` (this can be seen by writing the budget
    equation for the angular momentum :math:`L` and solving for :math:`\partial_t u`).
centered : bool, optional
    If False, use more accurate (but less convenient) half-level
    differentiation rather than centered differentiation.
**kwargs
    Passed to `~.diff.deriv_uneven` or `~.diff.deriv_half`.
"""
docstring.snippets['template_divcon'] = _template_divcon

# Auto-variance
_template_auto = """
Return the auto%(operator)s along the input dimension.

Parameters
----------
dim : str
    The dimension. This is replaced with a ``'lag'`` dimension on the
    output `~xarray.DataArray`.
**kwargs
    Passed to `~.var.auto%(func)s`.
"""
docstring.snippets['template_auto'] = _template_auto

# Variable derivations
_params_register = r"""
dest : str, tuple, or re.Pattern
    The destination variable name, a tuple of valid destination names, or an
    `re.compile`'d pattern matching a set of valid destination names. In the
    latter two cases, the function must accept a `name` keyword argument. This is
    useful if you want to register a single function capable of deriving multiple
    related variables (e.g., registering the regex ``r'\Ad.*dy\Z'`` to return the
    meridional gradient of an arbitrary variable).
assign_name : bool, optional
    Whether to assign the user-input string as the output `xarray.DataArray.name`.
    Default is ``True``.
"""
docstring.snippets['params_register'] = _params_register

# Messages
_warning_inplace = """
Warning
-------
Unlike most other public methods, this modifies the data in-place rather
than returning a copy.
"""
docstring.snippets['warning_inplace'] = _warning_inplace


def _expand_variable_args(func):
    """
    Expand single positional argument into multiple positional arguments with optional
    keyword dicts. Permits e.g. `get(('t', {'lat': 'mean'}))` tuple pairs.
    """
    @functools.wraps(func)
    def _wrapper(self, *keys, **kwargs):
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
        _iter_args(*keys)
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


def _manage_coord_reductions(func):
    """
    Add back singleton NaN dummy coordinates after some dimension reduction, so that
    we can continue relating dimension names to CF axis and coordinate names, and
    identically reduce cell weights. See `add_scalar_coords` for details on motivation.
    """
    @functools.wraps(func)
    def _wrapper(self, dim=None, *, manage_coords=True, **kwargs):
        # Call wrapped function
        # NOTE: Existing scalar coordinates should be retained by xarray
        coords = self.data.coords
        result = func(self, dim, **kwargs)
        if not manage_coords:
            return result

        # Treat lost coordinates
        coords_lost = coords.keys() - result.coords.keys()
        for name in coords_lost:
            prev = coords[name]
            try:
                measure = self.cf._encode_name(name, 'cell_measures')
            except KeyError:
                measure = None
            if prev.ndim == 1 and not measure:
                # Replace lost dimension coordinates with scalar NaNs. Drop 0D cell
                # measure coordinates, as that information is no longer needed
                coord = xr.DataArray(np.nan, name=name, attrs=coords[name].attrs)
                result = result.assign_coords({name: coord})
            if prev.ndim > 1 and measure and prev.sizes.keys() & result.sizes.keys():
                # Replace lost cell measures using unweighted sum. Drop non-cell measure
                # coordinates; usually makes no sense to get an 'average' coordinate
                # TODO: Support preserving measures for max, min, etc.
                if func.__name__ in ('sum', 'integral'):
                    method = prev.climo.sum
                elif func.__name__ in ('mean', 'average'):
                    method = prev.climo.mean
                else:
                    continue  # skip max, min, etc. for now
                if not dim:  # entire domain
                    dim = 'volume'
                if isinstance(dim, str):
                    dim = (dim,)
                dim = tuple(d for m in dim for d in CELL_MEASURE_COORDS.get(m, (m,)))
                coord = method(dim, manage_coords=False, **kwargs)
                result = result.assign_coords({name: coord})

        return result

    return _wrapper


def _matching_function(key, func, name):
    """
    Return function if the input string matches a string, tuple, or regex key. In
    latter two cases a `name` keyword arguments is added with `functools.partial`.
    """
    if isinstance(key, str) and name == key:
        return func
    if isinstance(key, tuple) and name in key:
        return functools.partial(func, name=name)
    if isinstance(key, re.Pattern) and key.match(name):
        return functools.partial(func, name=name)


class _CFAccessor(object):
    """
    CF accessor with cacheing to improve speed during iterative lookups.

    Notes
    -----
    Some features that could be added to this accessor are for now implemented
    in the climopy accessors. For example, key translation for dictionaries passed
    to ``data.loc[...]`` and ``data[...]``.
    """
    def __init__(self, data, registry=vreg):
        """
        Parameters
        ----------
        data : xarray.Dataset or xarray.DataArray
            The data.
        registry : cvariable.CFVariableRegistry
            The active registry.
        """
        self._obj = data
        self._src = data.coords if isinstance(data, xr.DataArray) else data
        self._stale_cache = False
        self._variable_registry = registry

    @staticmethod
    def _is_cfname(key):
        """
        Return whether the input string is already a CF standard
        axis name, coordinate name, or cell measure name.
        """
        return any(
            key == name for attr in ('_AXIS_NAMES', '_COORD_NAMES', '_CELL_MEASURES')
            for name in getattr(_cf_accessor, attr, ())
        )

    @staticmethod
    def _clear_cache(func):
        """
        Return a wrapper that clears cache before running. Use this with every public
        method where names are translated. Must come before `_manage_coord_reductions`.
        """
        # WARNING: Critical to only put this on user-facing top-level functions.
        # Tried putting it on _parse_key but get() ends up regenerating CF properties
        # tons of times due to derivations, adding cell measures, etc.
        @functools.wraps(func)
        def _wrapper(self, *args, **kwargs):
            cf = self._data.climo.cf  # permit use on .loc and .coord class funcs
            if not cf._stale_cache:
                return func(self, *args, **kwargs)
            for attr in ('axes', 'coordinates', 'cell_measures', 'standard_names'):
                cache = '_' + attr + '_cache'
                if hasattr(cf, cache):
                    delattr(cf, cache)
            cf._stale_cache = False
            try:
                return func(self, *args, **kwargs)
            finally:
                cf._stale_cache = True
        return _wrapper

    @staticmethod
    def _encode_attr(*parts):
        """
        Merge and encode parts into CF `cell_methods`-like attribute.
        """
        seen = set()
        parts = tuple(
            ((dims,) if isinstance(dims, str) else tuple(dims), value)
            for attr in parts for dims, value in _CFAccessor._decode_attr(attr) if dims
        )
        attr = ' '.join(
            ': '.join(dims) + ': ' + value for dims, value in parts
            if (dims, value) not in seen and not seen.add((dims, value))
        )
        return attr.strip()

    @staticmethod
    def _decode_attr(attr):
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
            dims = tuple(dim.strip() for dim in substring[:idx].split(':') if dim.strip())  # noqa: E501
            value = substring[idx:].strip()
            parts.append((dims, value))
        return parts

    def _encode_name(self, key, *attrs, search_registry=True):
        """
        Translate a dataset variable name or registry variable name or alias
        into a standard CF name. Check only the specified attributes.
        """
        if not isinstance(key, str):
            raise KeyError('Key must be string.')
        if self._is_cfname(key):
            return key
        # Decode variable aliases into native dataset names used by CF accessor
        if search_registry:
            var = self._variable_registry.get(key, None)
            for name in getattr(var, 'identifiers', ()):
                if name in self._src:
                    key = name
                    break
        # Check if key is present in CF accessor properties
        attrs = attrs or ('axes', 'coordinates', 'cell_measures', 'standard_names')
        for attr in attrs:
            mapping = getattr(self, attr)
            for coord, names in mapping.items():
                if key in names:
                    return coord
        raise KeyError(f'Failed to find CF name for variable {key!r}.')

    def _decode_name(self, key, *attrs, search_registry=True, return_if_missing=False):
        """
        Translate a standard CF name or registry variable alias into dataset variable
        name or registry variable name. Check only the specified attributes.
        """
        if not isinstance(key, str):
            raise KeyError('Key must be string.')
        # Check if already valid variable name
        if key in self._src:
            return key
        # Check if key matches CF name
        attrs = attrs or ('axes', 'coordinates', 'cell_measures', 'standard_names')
        for attr in attrs:
            names = getattr(self, attr).get(key, None)
            if not names:  # unknown in dictionary
                pass
            elif len(names) > 1:
                raise KeyError(f'Too many options for CF key {key!r}: {names!r}')
            elif names[0] in self._src:
                return names[0]
            elif return_if_missing:
                return names[0]  # e.g. missing cell_measures variable
        # Check if key matches registry variable alias
        if search_registry:
            var = self._variable_registry.get(key, None)
            for name in getattr(var, 'identifiers', ()):
                if name in self._src:
                    return name
            if var and return_if_missing:
                return var.name  # return *standard* registered name
        raise KeyError(
            f'Failed to find dataset or registry variable for CF name {key!r}.'
        )

    def _get_cache_attr(self, attr):
        """
        Return attribute, deferring to cache if it exists and creating cache if not.
        """
        cache = '_' + attr + '_cache'
        value = getattr(self, cache, None)
        if value is None:
            value = getattr(super(_CFAccessor, self), attr)
            setattr(self, cache, value)
        return value

    def _get_item(self, key, *attrs, **kwargs):
        """
        Try to get item using CF name. Search only the input attributes.
        """
        try:
            name = self._decode_name(key, *attrs, **kwargs)
        except KeyError:
            return
        if name in self._src:
            return self._src[name]

    # Properties
    # NOTE: CF accessor only looks for .axes and coordinates in the .coords object
    # WARNING: CF accessor .axes, .coordinates, etc. is extremely slow due simply to
    # repeated generation and lookups of modestly sized dictionaries in nested loops.
    # Try to avoid this using simple cacheing approach.
    axes = property(functools.partial(_get_cache_attr, attr='axes'))
    coordinates = property(functools.partial(_get_cache_attr, attr='coordinates'))
    cell_measures = property(functools.partial(_get_cache_attr, attr='cell_measures'))
    standard_names = property(functools.partial(_get_cache_attr, attr='standard_names'))

    @property
    def vertical_type(self):
        """
        The type of the unique ``'vertical'`` axis. Result is one of
        ``'temperature'``, ``'pressure'``, ``'height'``, or ``'unknown'``.
        Model levels and hybrid sigma coordinates are not yet supported.
        """
        da = self._get_item('vertical', 'coordinates')
        if da is None:
            return 'unknown'
        units = da.climo.units
        if units.is_compatible_with('K'):
            return 'temperature'
        elif units.is_compatible_with('Pa'):
            return 'pressure'
        elif units.is_compatible_with('m'):
            return 'height'
        else:
            return 'unknown'


class _CFDataArrayAccessor(
    _CFAccessor, _cf_accessor.CFDataArrayAccessor
):
    pass


class _CFDatasetAccessor(
    _CFAccessor, _cf_accessor.CFDatasetAccessor
):
    pass


class _GroupByQuantified(object):
    """
    A unit-friendly `ClimoAccessor.groupby` indexer.
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
    A unit-friendly `ClimoAccessor.loc` indexer for `xarray.DataArray`\\ s.
    """
    def __init__(self, data_array):
        self._data = data_array

    @_CFAccessor._clear_cache
    def __getitem__(self, key):
        """
        Request slices optionally with pint quantity indexers.
        """
        data = self._data
        key, _ = data.climo._parse_indexers(data.climo._expand_ellipsis(key))
        key = data.climo._dequantify_indexers(key)
        return data.loc[key]

    @_CFAccessor._clear_cache
    def __setitem__(self, key, value):
        """
        Request and set slices optionally with pint quantity indexers and
        pint quantity assignments.
        """
        # Standardize indexers
        # NOTE: Xarray does not support boolean loc indexing
        # See: https://github.com/pydata/xarray/issues/3546
        data = self._data
        key, _ = data.climo._parse_indexers(data.climo._expand_ellipsis(key))
        key = data.climo._dequantify_indexers(key)
        data.loc[key].climo._assign_value(value)


class _DatasetLocIndexerQuantified(object):
    """
    A unit-friendly `ClimoAccessor.loc` indexer for `xarray.Dataset`\\ s.
    """
    def __init__(self, dataset):
        self._data = dataset

    def __getitem__(self, key):
        data = self._data
        key, _ = data.climo._parse_indexers(key)
        key = data.climo._dequantify_indexers(key)
        return data.loc[key]


class _CoordsQuantified(object):
    """
    A unit-friendly `ClimoAccessor.coords` container.
    """
    def __init__(self, data, registry=vreg):
        """
        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            The data.
        registry : cfvariable.CFVariableRegistry
            The variable registry.
        """
        self._data = data
        self._variable_registry = registry

    @_CFAccessor._clear_cache
    def __contains__(self, key):
        """
        Is a valid derived coordinate.
        """
        return self._parse_key(key) is not None

    def __getattr__(self, attr):
        """
        Try to return a coordinate with `__getitem__`.
        """
        if attr in super().__dir__():  # can happen if @property triggers error
            return super().__getattribute__(attr)
        if attr in self:
            return self.__getitem__(attr)
        raise AttributeError(
            f'Attribute {attr!r} does not exist and is not a coordinate or '
            'transformed coordinate.'
        )

    @_CFAccessor._clear_cache
    def __getitem__(self, key):
        """
        Return a quantified coordinate or derived coordinate.
        """
        tup = self._parse_key(key)
        if not tup:
            raise KeyError(f'Invalid coordinate spec {key!r}.')
        return self._build_coord(*tup)

    def _build_coord(self, transformation, coord, flag, **kwargs):
        """
        Return the coordinates, accounting for `CF` and `CFVariableRegistry` names.
        """
        # Select or return bounds
        # WARNING: Get bounds before doing transformation because halfway points in
        # actual lattice may not equal halfway points after nonlinear transformation
        dest = coord
        suffix = ''
        if flag:
            bnds = self._get_bounds(coord, **kwargs)
            if flag in ('bnds', 'bounds'):
                return bnds.climo.quantify()
            if flag[:3] in ('bot', 'del'):
                dest = bottom = bnds[..., 0]  # NOTE: scalar coord bnds could be 1D
                suffix = ' bottom edge'
            if flag[:3] in ('top', 'del'):
                dest = top = bnds[..., 1]
                suffix = ' top edge'
            if flag[:3] == 'del':
                # NOTE: If top and bottom are cftime or native python datetime,
                # xarray coerces array of resulting native python timedeltas to a
                # numpy timedelta64 array (not the case with numpy arrays). See:
                # http://xarray.pydata.org/en/stable/time-series.html#creating-datetime64-data
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
            dest.name = None  # ensure name is modified
            dest = transformation(dest)  # adjusts name if it is unset

        # Return coords with cleaned up attributes. Only long_name and standard_name
        # are kept if math was performed.
        flag = '_' + flag if flag else ''
        dest.name += flag
        if long_name := coord.attrs.get('long_name'):
            dest.attrs['long_name'] = long_name + suffix
        if standard_name := coord.attrs.get('standard_name'):
            dest.attrs['standard_name'] = standard_name

        return dest

    def _get_bounds(self, coord, sharp_cutoff=None):
        """
        Return bounds inferred from the coordinates or generated on-the-fly. See
        `.get` for `sharp_cutoff` details.
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
                    f'Coordinate {coord.name!r} bounds variable {bounds!r} missing '
                    f'from dataset with variables {tuple(data)}. Calculating '
                    'bounds on-the-fly instead.'
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
        type_ = data.climo.cf.vertical_type
        coordinates = data.climo.cf.coordinates
        if coord.size == 1:
            if not coord.isnull():
                raise RuntimeError(
                    f'Cannot infer bounds for singleton non-NaN coord {coord!r}.'
                )
            if coord.name in coordinates.get('longitude', ()):
                bounds = [-180.0, 180.0] * ureg.deg
            elif coord.name in coordinates.get('latitude', ()):
                bounds = [-90.0, 90.0] * ureg.deg
            elif coord.name in coordinates.get('vertical', ()) and type_ == 'pressure':
                bounds = [0.0, 1013.25] * ureg.hPa
            else:
                raise RuntimeError(
                    f'Cannot infer bounds for singleton NaN coord {coord!r}. Must '
                    'be a longitude, latitude, or vertical pressure dimension.'
                )
            bounds = bounds.to(coord.climo.units).magnitude
            if not quack._is_scalar(coord):
                bounds = bounds[None, :]

        # Cell bounds for time coordinates. Unlike spatial cells, time cell coordinates
        # almost always indicates the end of the cell rather than the center.
        # WARNING: Requires standard_name set to 'time'
        elif coord.name in coordinates.get('time', ()):
            diffs = coord.data[1:] - coord.data[:-1]
            diffs = np.hstack((diffs[:1], diffs))
            upper = coord.data
            lower = upper - diffs
            bounds = np.hstack((lower[:, None], upper[:, None]))

        # Construct default cell bounds
        elif quack._is_numeric(coord):
            if sharp_cutoff or sharp_cutoff is None:
                delta1 = delta2 = 0
            else:
                delta1 = 0.5 * np.diff(coord.data[:2])
                delta2 = 0.5 * np.diff(coord.data[-2:])
            edges = np.concatenate(
                (
                    coord.data[:1] - delta1,
                    0.5 * (coord.data[1:] + coord.data[:-1]),
                    coord.data[-1:] + delta2
                )
            )
            bounds = np.hstack((edges[:-1, None], edges[1:, None]))

        # Non-numeric fallback
        else:
            raise RuntimeError(
                f'Cannot infer bounds for non-numeric non-time coord {coord.name!r}.'
            )

        # Fix boundary conditions at meridional domain edge
        # NOTE: Includes kludge where we ignore data from other hemisphere if we
        # have hemispheric data with single latitude from other hemisphere.
        if coord.name in coordinates.get('latitude', ()):
            bnd_lo = 1e-10
            bnd_hi = 90
            bounds[bounds < -bnd_hi] = -bnd_hi
            bounds[bounds > bnd_hi] = bnd_hi
            mask = bounds < -bnd_lo
            if bounds[mask].size == 1:
                bounds[mask] = -bnd_lo
            mask = bounds > bnd_lo
            if bounds[mask].size == 1:
                bounds[mask] = bnd_lo

        # Return new DataArray
        bounds = xr.DataArray(
            bounds,
            name=coord.name + '_bnds',
            dims=(*coord.dims[:1], 'bnds'),  # nameless 'bnds' dimension
            coords=coord.coords,
            attrs=coord.attrs,
        )

        return bounds

    def _parse_key(
        self,
        key,
        search_cf=True,
        search_registry=True,
        search_transformations=True,
    ):
        """
        Return the coordinates, transformation function, and flag.
        """
        # Interpret bounds specification
        if not isinstance(key, str):
            raise TypeError(f'Invalid key {key!r}. Must be string.')
        key, flag = REGEX_BOUNDS.match(key).groups()
        flag = flag or ''

        # Find native coordinate
        # WARNING: super() alone fails possibly because it returns the super() of
        # e.g. _DataArrayCoordsQuantified, which would be _CoordsQuantified.
        transformation = None
        if super(_CoordsQuantified, self).__contains__(key):
            coord = super(_CoordsQuantified, self).__getitem__(key)
            return transformation, coord, flag

        # Find CF alias
        data = self._data
        if search_cf:
            coord = data.climo.cf._get_item(key, 'axes', 'coordinates')
            if coord is not None:
                return transformation, coord, flag

        # Find transformed coordinate
        # WARNING: Cannot call native items() or values() because they call
        # overridden __getitem__ internally. So recreate coordinate mapping below.
        if search_transformations:
            coords = (super(_CoordsQuantified, self).__getitem__(key) for key in self)
            if tup := data.climo._find_any_transformation(coords, key):
                transformation, coord = tup
                return transformation, coord, flag

        # Recursively check if any aliases are valid
        if search_registry:
            var = self._variable_registry.get(key)
            flag = flag and '_' + flag  # '' if empty, '_flag' if non-empty
            identifiers = var.identifiers if var else ()
            for name in set(identifiers):
                if tup := self._parse_key(
                    name + flag,
                    search_cf=search_cf,
                    search_transformations=search_transformations,
                    search_registry=False,
                ):
                    return tup

    @_CFAccessor._clear_cache
    def get(self, key, default=None, sharp_cutoff=None, **kwargs):
        """
        Return the coordinate if it is present, otherwise return a default value.

        Parameters
        ----------
        key : str
            The coordinate key.
        default : optional
            The default return value.
        search_cf : bool, optional
            Whether to translate CF names.
        search_registry : bool, optional
            Whether to translate registered names and aliases.
        search_transformations : bool, optional
            Whether to perform registered transformations of coordinates.
        sharp_cutoff : bool, optional
            The cutoff behavior used when calculating default non-datetime coordinate
            bounds in the event that an explicit ``'bounds'`` variable is unavailable.
            When ``True``, the end coordinate centers are also treated as coordinate
            edges. When ``False``, the end coordinate edges are calculated as half the
            distance between the closest coordinate centers away from the edgemost
            centers. Default is ``True``, which should yield correct results when
            working with datasets whose coordinate centers cover the entire domain (360
            degrees of longitude, 180 degrees of latitude, and 1013.25 hectoPascals of
            pressure), as with datasets modified with `~ClimoAccessor.enforce_global`.
        """
        tup = self._parse_key(key, **kwargs)  # potentially limit search
        if tup is None:
            return default
        else:
            return self._build_coord(*tup, sharp_cutoff=sharp_cutoff)


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
        self._variable_registry = registry

    def __iter__(self):
        """
        Iterate over non-coordinate variables.
        """
        return self._data.data_vars.__iter__()

    @_CFAccessor._clear_cache
    def __contains__(self, key):
        """
        Is a valid variable.
        """
        return self._parse_key(key) is not None

    def __getattr__(self, attr):
        """
        Try to return a variable with `__getitem__`.
        """
        if attr in super().__dir__():  # can happen if @property triggers error
            return super().__getattribute__(attr)
        if attr in self:
            return self.__getitem__(attr)
        raise AttributeError(
            f'Attribute {attr!r} does not exist and is not a variable.'
        )

    @_CFAccessor._clear_cache
    def __getitem__(self, key):
        """
        Return a quantified variable.
        """
        da = self._parse_key(key)
        if da is None:
            raise KeyError(f'Invalid variable name {key!r}.')
        return da.climo.quantify()

    def _parse_key(self, key, search_cf=True, search_registry=True):
        """
        Return a function that generates the variable, accounting for CF and
        CFVariableRegistry names.
        """
        # Find native variable
        # NOTE: Compare with _CoordsQuantified._parse_key and ClimoDatasetAccessor
        data = self._data
        if not isinstance(key, str):
            raise TypeError(f'Invalid key {key!r}. Must be string.')
        if key in data.data_vars:  # exclude coords
            return data[key]

        # Find CF alias
        if search_cf:
            da = data.climo.cf._get_item(key, 'cell_measures', 'standard_names')
            if da is not None and da.name in data.data_vars:
                return da

        # Locate using identifier synonyms
        if search_registry:
            var = self._variable_registry.get(key)
            identifiers = var.identifiers if var else ()
            for name in set(identifiers):
                da = self._parse_key(name, search_cf=search_cf, search_registry=False)
                if da is not None:
                    return da

    @_CFAccessor._clear_cache
    def get(self, key, default=None, **kwargs):
        """
        Return the variable if it is present, otherwise return a default value.

        Parameters
        ----------
        default : optional
            The default return value.
        search_cf : bool, optional
            Whether to translate CF names.
        search_registry : bool, optional
            Whether to translate registered names and aliases.
        """
        da = self._parse_key(key, **kwargs)  # potentially limit search
        if da is not None:
            return da.climo.quantify()
        else:
            return default


class ClimoAccessor(object):
    """
    Accessor with properties and methods shared by `xarray.DataArray`\\ s and
    `xarray.Dataset`\\ s. Registered under the name ``climo`` (i.e, usage is
    ``data_array.climo`` and ``dataset.climo``).
    """
    def __init__(self, data, registry=vreg):
        """
        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            The data.
        registry : cfvariable.CFVariableRegistry
            The variable registry.

        Notes
        -----
        This adds `pint.Quantity` support for the operations `~xarray.DataArray.loc`,
        `~xarray.DataArray.sel`, `~xarray.DataArray.interp`, and
        `~xarray.DataArray.groupby`. Otherwise, `~xarray.DataArray.weighted` and
        `~xarray.DataArray.coarsen` already work, but `~xarray.DataArray.resample`
        and `~xarray.DataArray.rolling` are broken and may be quite tricky to fix.
        """
        self._data = data
        self._cf_accessor = None
        self._variable_registry = registry

    def _dequantify_indexers(self, indexers):
        """
        Reassign a `pint.Quantity` indexer to units of relevant coordinate.
        """
        def _dequantify_value(value, units, scalar=True):
            if isinstance(value, xr.DataArray):
                # Strip non-dimensional coordinates to avoid indexer/indexee conflicts
                # WARNING: Keep DataArray indexers as DataArrays! Xarray can read from
                # them, and coords could be interpreted by _iter_by_indexer_coords.
                if value.climo._has_units and value.dtype.kind != 'b':
                    value = value.climo.to_units(units)
                value = value.climo.dequantify()
                value = value.squeeze(drop=True)
                value = value.drop_vars(value.coords.keys() - value.sizes.keys())
            elif isinstance(value, pint.Quantity):
                value = value.to(units).magnitude
            if np.asarray(value).size == 1:
                value = np.asarray(value).item()  # pull out of ndarray or DataArray
            elif scalar:
                raise ValueError(f'Expected scalar indexer, got {value=}.')
            return value

        # Update indexers to handle quantities and slices of quantities
        data = self.data
        indexers_scaled = {}
        for name, sel in indexers.items():
            if (coord := data.climo.coords.get(name)) is not None:
                units = coord.climo.units
            else:
                units = None
            if isinstance(sel, slice):
                start = _dequantify_value(sel.start, units, scalar=True)
                stop = _dequantify_value(sel.stop, units, scalar=True)
                step = _dequantify_value(sel.step, units, scalar=True)
                indexers_scaled[name] = slice(start, stop, step)
            else:
                indexers_scaled[name] = _dequantify_value(sel, units, scalar=False)

        return indexers_scaled

    def _find_derivation(self, dest):
        """
        Find derivation that generates the variable name. Return `None` if not found.
        """
        # TODO: Merge 'transformations' with 'derivations' concepts with
        # metpy.calc-like functions that can be supplied with dataset variables.
        for idest, derivation in DERIVATIONS.items():
            if func := _matching_function(idest, derivation, dest):
                return func

    def _find_any_transformation(self, data_arrays, dest):
        """
        Find transformation that generates the variable name. Return `None` if not
        found. Otherwise return the generating function and a source variable.
        """
        # TODO: Merge 'transformations' with 'derivations' concepts with
        # metpy.calc-like functions that can be supplied with dataset variables.
        for data in data_arrays:
            if func := self._find_this_transformation(data.name, dest):
                return func, data

    def _find_this_transformation(self, src, dest):
        """
        Find possibly nested series of transformations that get from variable A --> C.
        Account for `CF` and `CFVariableRegistry` names.
        """
        # Translate names to dataset variable names or registry variable names
        try:
            src = self.cf._decode_name(src, return_if_missing=True)
        except KeyError:
            return  # source not available!
        try:
            dest = self.cf._decode_name(dest, return_if_missing=True)
        except KeyError:
            pass
        if src == dest:
            return lambda da: da.copy()
        # Find the transformation
        for (isrc, idest), transformation in TRANSFORMATIONS.items():
            try:
                isrc = self.cf._decode_name(isrc, return_if_missing=True)
            except KeyError:
                continue  # source not available!
            if isrc != src:
                continue
            if func := _matching_function(idest, transformation, dest):
                return func
            # Perform nested invocation of transformations. Inner func goes from
            # A --> ?, then outer func from ? --> B (returned above)
            if outer := self._find_this_transformation(idest, dest):  # noqa: E501
                return lambda da, **kwargs: outer(transformation(da, **kwargs))

    def _find_params(self, allow_empty=False, return_reference=False):
        """
        Iterte over parameter coordinates (identified as cfvariables with references).
        """
        coords = {}
        for dim, coord in self.data.coords.items():
            try:
                cfvariable = coord.climo._cf_variable(use_cf_attrs=False)
                reference = cfvariable.reference
            except AttributeError:
                continue
            if reference is not None:
                coords[dim] = reference if return_reference else coord
        if not allow_empty and not coords:
            raise RuntimeError('No parameter dimensions found.')
        return coords

    def _iter_data_vars(self, include_dataset=False):
        """
        Iterate over non-coordinate DataArrays. If this is a DataArray just yield it.
        """
        data = self.data
        if isinstance(data, xr.DataArray):
            yield data
        else:
            if include_dataset:
                yield data
            yield from data.values()

    def _iter_by_indexer_coords(
        self,
        func,
        indexers,
        drop_cell_measures=False,
        **kwargs
    ):
        """
        Apply function `func` (currently `.sel` or `.interp`) using each scalar value
        in the indexers, then merge along the indexer coordinate axes. Always preserves
        coordinate attributes and supports arbitrary ND multiple selections (e.g.,
        selecting the storm track latitude at every time step).
        """
        # Iterate over non-scalar indexer coordinates
        # NOTE: If coordinates are present on indexers, they must match! For example:
        # lat=xr.DataArray([30, 60, 90], coords={'dummy': [10, 20, 30]})
        # lev=xr.DataArray([250, 500, 750], coords={'dummy': [10, 20, 30]})
        # NOTE: The native xarray approach is: "if DataArrays are passed as new
        # coordinates, their dimensions are used for the broadcasting." This includes
        # ugly behavior of *replacing* existing coordinate values. By contrast, we
        # *require* indexer coordinates must correspond to data coordinates, and
        # *require* that they match the existing coordinate values.
        indexers_fancy = {k: v for k, v in indexers.items() if isinstance(v, xr.DataArray)}  # noqa: E501
        indexers = {k: indexers[k] for k in indexers.keys() - indexers_fancy.keys()}
        datas = np.empty((1,), dtype='O')  # stores xarray objects
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
        for idx in np.ndindex(datas.shape):
            # Interpolate or select
            isel = {k: v for k, v in zip(dims, idx)}
            idata = data.isel(isel, drop=True)
            indexer = indexers.copy()
            for key, val in indexers_fancy.items():
                indexer[key] = val.isel(isel, drop=True)
            idata = getattr(idata, func)(indexer, **kwargs)

            # Repair output
            for dim, value in indexer.items():
                # Preserve attributes of indexed coordinate
                # NOTE: This is critical for CF interpretation of coords! Right now
                # interp drops attrs. See: https://github.com/pydata/xarray/issues/4239
                if dim in idata.coords:  # wasn't dropped
                    idata.coords[dim].attrs.update(data.coords[dim].attrs)
                # Drop corresponding cell measures for indexed coordinates
                # NOTE: Without this get bizarre behavior where using .interp(lev=1050)
                # extrapolates to below surface but subsequent .reduce(lat='avg')
                # omits those points.
                if drop_cell_measures:
                    try:
                        coordinate = self.cf._encode_name(dim, 'coordinates')
                    except KeyError:
                        continue
                    measure = COORD_CELL_MEASURE[coordinate]
                    try:
                        measure = self.cf._decode_name(measure, 'cell_measures', search_registry=False)  # noqa: E501
                    except KeyError:
                        continue
                    if measure is not None and measure in idata.coords:
                        idata = idata.drop_vars(measure)

            # Save DataArray subset
            datas[idx] = idata

        # Merge along indexer coordinates, and return to original permution order
        if indexers_fancy:
            data = xr.combine_nested(
                datas.tolist(),
                concat_dim=dims,
                join='exact',
                compat='identical',
                combine_attrs='identical',
            )
            data = data.climo.replace_coords(dict(coords))
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
            return tuple(indexers)
        elif len(indexers) > 1:
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
        include_no_coords=False,
        search_transformations=False,
        **kwargs
    ):
        """
        Parse and translate keyword dimension indexers.
        """
        dims = self.data.dims
        coords = self.data.coords
        indexers = indexers or {}
        indexers = indexers.copy()
        indexers.update(kwargs)
        indexers_filtered = {}
        for key in tuple(indexers):
            # Translate indexer name
            dim = key
            if dim in dims and dim not in coords:  # but not coordinates
                if include_no_coords:
                    indexers_filtered[dim] = indexers.pop(key)
                    continue
                else:
                    raise RuntimeError(f'Dimension {key!r} is missing coordinate data.')
            try:
                dim = self.cf._decode_name(dim, 'axes', 'coordinates')
            except KeyError:
                pass
            # Handle salar indexer
            if dim in coords and coords[dim].size == 1:
                if ignore_scalar:  # used for .sum() and .mean()
                    del indexers[key]
                    continue
                elif not include_scalar:  # used for .integral() and .average()
                    raise RuntimeError(f'Coordinate {key!r} is scalar.')
            # Validate indexer
            if (
                dim in coords
                or include_pseudo and dim in ('area', 'volume')
                or search_transformations and self._find_any_transformation(coords.values(), dim)  # noqa: E501
            ):
                # e.g. integral('area') or derivative('meridional_coordinate')
                indexers_filtered[dim] = indexers.pop(key)
            elif not allow_kwargs:
                raise ValueError(f'Invalid argument or unknown dimension {key!r}.')

        # Warn if empty
        if not indexers_filtered:
            raise ValueError(f'No valid indexers found in kwargs {indexers!r}.')
        return indexers_filtered, indexers

    def _parse_truncate_args(self, **kwargs):
        """
        Parse arguments used to truncate data. Returns tuple of dictionaries used
        to limit data range. Used by both `~ClimoAccessor.truncate` and
        `~ClimoDataArrayAccessor.reduce`.
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
            # everywhere (e.g. _[lat|strength]) so this is consistent with API. Makes
            # things much easier if we can just detect special suffix without trying
            # to figure out if the rest of the string matches a dimension yet.
            m = re.match(r'\A(.*?)_(min|max|lim)\Z', key)
            if not m:
                continue
            if key not in kwargs:  # happens with e.g. latitude_min=x latitude_max=y
                continue
            dim, mode = m.groups()
            try:
                dim = self.cf._decode_name(dim, 'axes', 'coordinates')
            except KeyError:
                raise TypeError(f'Invalid truncation arg {key!r}.')
            units = data.coords[dim].climo.units

            # Get start and stop locations
            # Handle passing e.g. latmin=x latmax=y instead of latlim=z
            loc = kwargs.pop(key)
            if mode == 'max':
                start = kwargs.pop(dim + '_min', None)
                stop = loc
            elif mode == 'min':
                start = loc
                stop = kwargs.pop(dim + '_max', None)
            else:
                start, stop = loc

            # Get 'variable-spec' bounds and translate units
            # Then add to the list of starts and stops
            dims.append(dim + '_lim')
            for bound, mode in zip((start, stop), ('min', 'max')):
                # Translate 'parameter' bounds
                if isinstance(bound, (str, tuple)):  # 'name' or ('name', {})
                    if not isinstance(data, xr.Dataset):
                        raise ValueError('Dataset required to get bounds {bound!r}.')
                    bound = data.climo.get(bound)  # may add a 'track' dim
                else:
                    if bound is None:
                        bound = getattr(data.coords[dim], mode)()
                        # bound = getattr(data.climo.coords[dim].climo.magnitude, mode)()  # noqa: E501
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
        # NOTE: The find 'track' dims have no coordinates
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

    @_CFAccessor._clear_cache
    def add_cell_measures(
        self, measures=None, *, dataset=None, override=False, verbose=False, **kwargs
    ):
        """
        Add cell measures to `~xarray.DataArray.coords`, remove cell measures missing
        from `~xarray.DataArray.coords`, and update the ``cell_measures`` attribute(s).

        Parameters
        ----------
        measures : dict-like, optional
            Dictionary of cell measures. If none are provided, the default `width`,
            `depth`, `height`, and `period` measures are automatically calculated.
            If this is a DataArray, surface pressure will not be taken into account
            and isentropic grids will error out.
        dataset : xarray.Dataset, optional
            The dataset associated with this `xarray.DataArray`. Needed when
            calculating cell measures automatically.
        override : bool, optional
            Whether to override existing cell measures. Default is ``False``.
        verbose : bool, optional
            If ``True``, print statements are issued.
        **kwargs
            Cell measures passed as keyword args.
        """
        stopwatch = _make_stopwatch(verbose=False)
        cf = self.cf
        data = self.data.copy(deep=False)
        action = 'default'
        measures = measures or {}
        measures.update(kwargs)
        if isinstance(data, xr.Dataset):
            dataset = data
        elif dataset is None:
            # Get weights from temporary dataset. Ignore CF UserWarning and
            # ClimoPyWarnings due to missing bounds and ClimoPyWarnings due
            # to missing surface pressure coordinate for vertical bounds.
            dataset = data.to_dataset(name=data.name or 'unknown')
            action = 'ignore'

        # Add default cell measures
        if not measures:
            stopwatch('init')
            for measure in ('length', 'width', 'height', 'duration'):
                # Skip measures that already exist in coordinates and measures that
                # aren't subset of existing spatial coordinates
                if (
                    not override
                    and measure in cf.cell_measures
                    and cf.cell_measures[measure][0] in data.coords
                    or set(CELL_MEASURE_COORDS[measure]) - set(cf.coordinates)
                ):
                    continue

                # Calculate new measures
                # NOTE: This catches RuntimeErrors emitted from _get_bounds if fail to
                # calculate bounds and NotImplementedErrors from the definitions e.g.
                # if there is no algorithm for cell height (child of RuntimeError)
                name = 'cell_' + measure
                with warnings.catch_warnings():
                    warnings.simplefilter(action)  # possibly ignore warnings
                    try:
                        weight = dataset.climo._get_item(
                            name,
                            search_cf=False,
                            search_transformations=False,
                            search_registry=False,
                            add_cell_measures=False
                        )
                    except RuntimeError as err:
                        if verbose:
                            print(f'Failed to add cell measure {measure!r} with name {name!r}. Error message: {err!s}')  # noqa: E501
                        continue
                    else:
                        if weight.sizes.keys() - data.sizes.keys():
                            continue  # e.g. 'length' for data with no latitude dim
                        if verbose:
                            print(f'Added cell measure {measure!r} with name {name!r}.')
                        weight.name = name  # just in case
                        measures[measure] = weight
                    finally:
                        stopwatch(measure)

        # Add measures as dequantified coordinate variables
        # TODO: Stop adding cell measures attribute to whole dataset
        # NOTE: This approach is used as an example in cf_xarray docs:
        # https://cf-xarray.readthedocs.io/en/latest/examples/introduction.html#Feature:-Weight-by-Cell-Measures
        missing = set()  # only emit warning once
        if not all(isinstance(da, xr.DataArray) for da in measures.values()):
            raise ValueError('Input cell measures must be DataArrays.')
        if any(da.name is None for da in measures.values()):
            raise ValueError('Input cell measures must have names.')
        for obj in data.climo._iter_data_vars(include_dataset=True):
            measures_old = cf._decode_attr(obj.attrs.get('cell_measures', ''))
            for key in measures_old:
                (measure,), name = key
                if name in data.coords:
                    continue
                measures_old.remove(key)
                if verbose and name not in missing:
                    print(f'Removed missing {measure!r} cell measure {name!r}.')
                missing.add(name)
            measures_new = measures_old.copy()
            for measure, da in measures.items():
                if not isinstance(da, xr.DataArray):
                    raise ValueError('Input cell measures must be DataArrays.')
                if da.name is None:
                    raise ValueError('Input cell measures must have names.')
                data.coords[da.name] = da.climo.dequantify()
                if isinstance(obj, xr.DataArray) and obj.climo._is_coordinate_bounds:
                    continue
                measures_new.append((measure, da.name))
            obj.attrs['cell_measures'] = cf._encode_attr(measures_new)

        return data

    @_CFAccessor._clear_cache
    def add_scalar_coords(self, verbose=False):
        """
        Add dummy scalar coordinates for missing longitude, latitude, and vertical
        dimensions and update the `cell_methods` attribute(s) to indicate the missing
        coordinates were reduced by averaging. For motivation, see the final paragraph
        of CF manual section 7.3.2:

            A dimension of size one may be the result of "collapsing" an axis by some
            statistical operation, for instance by calculating a variance from time
            series data. We strongly recommend that dimensions of size one be retained
            (or scalar coordinate variables be defined) to enable documentation of the
            method (through the cell_methods attribute) and its domain (through the
            bounds attribute).

        In other words, while `cell_methods` are typically used to indicate how each
        cell in a 1D coordinate vector was constructed from a notional "original"
        sampling interval, they can also be used to indicate how a single scalar
        coordinate was reduced from a 1D coordinate vector.

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, print statements are issued.
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
            da = self.cf._get_item(coord, 'coordinates')
            if da is None:
                if dim in data.dims:
                    if data.sizes[dim] == 1:  # exists as singleton dim but not coord
                        data = data.squeeze(dim)
                    else:
                        raise ValueError('Dimension already exists without coords.')
                data.coords[dim] = ((), np.nan, attrs[dim])
                if verbose:
                    print(f'Added missing scalar {coord} coordinate {dim!r}.')
            elif da.size == 1:
                dim = da.name
                if dim in data.sizes:  # i.e. is a dimension
                    data = data.squeeze(dim)
                data = data.climo.replace_coords({dim: np.nan})
                for key, value in attrs.items():
                    data.attrs.setdefault(key, value)
                if verbose:
                    print(f'Set scalar {coord} coordinate {dim!r} value to NaN.')
            else:
                continue
            data.climo.update_cell_methods({dim: 'mean'})

        return data

    @_CFAccessor._clear_cache
    def enforce_global(self, longitude=True, latitude=True, vertical=False, zero=None):
        """
        Add a circularly overlapping longitude coordinate, latitude coordinates for
        the north and south poles, and pressure coordinates for the mean sea-level
        and "zero" pressure levels. This ensures plots data coverage over the whole
        atmosphere and improves the accuracy of budget term calculations.

        Parameters
        ----------
        longitude : bool, optional
            Whether to enforce circular longitudes. Default is ``True``.
        latitude : bool, optional
            Whether to enforce latitude coverage from pole-to-pole. Default is ``True``.
        vertical : bool, optional
            Whether to enforce pressure level coverage from 0 hectoPascals to
            1013.25 hectoPascals (mean sea-level pressure). Default is ``False``.
        zero : bool or sequence of str, optional
            If this is a `~ClimoDataArrayAccessor`, should be boolean indicating whether
            data at the pole coordinates should be zeroed (e.g. wind variables and
            extensive properties like fluxes). If this is a `~ClimoDatasetAccessor`,
            should be sequence of variables that should be zeroed.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> import climopy as climo
        >>> ds = xr.Dataset(
        ...     coords={
        ...         'lon': np.arange(0, 360, 30),
        ...         'lat': np.arange(-85, 86, 10),
        ...         'lev': ('lev', np.arange(100, 1000, 100), {'units': 'hPa'}),
        ...     }
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (lat: 18, lev: 9, lon: 12)
        Coordinates:
          * lon      (lon) int64 0 30 60 90 120 150 180 210 240 270 300 330
          * lat      (lat) int64 -85 -75 -65 -55 -45 -35 -25 ... 25 35 45 55 65 75 85
          * lev      (lev) int64 100 200 300 400 500 600 700 800 900
        Data variables:
            *empty*
        >>> ds = ds.climo.standardize_coords()
        >>> ds = ds.climo.enforce_global(vertical=True)
        >>> ds = ds.climo.add_cell_measures()
        >>> ds
        <xarray.Dataset>
        Dimensions:      (lat: 20, lev: 11, lon: 13)
        Coordinates:
          * lon          (lon) float64 -2.03e+04 0.0 30.0 60.0 ... 270.0 300.0 330.0
          * lat          (lat) float64 -90.0 -85.0 -75.0 -65.0 ... 65.0 75.0 85.0 90.0
          * lev          (lev) float64 0.0 100.0 200.0 300.0 ... 800.0 900.0 1.013e+03
            cell_width   (lat, lon) float64 6.91e-11 6.92e-11 ... 2.043e-13 1.021e-13
            cell_depth   (lat) float64 278.0 834.0 1.112e+03 ... 1.112e+03 834.0 278.0
            cell_height  (lev) float64 509.9 1.02e+03 1.02e+03 ... 1.087e+03 577.4
        Data variables:
            *empty*
        Attributes:
            cell_measures:  width: cell_width depth: cell_depth height: cell_height
        """
        # Add circular longitude coordinates
        data = self.data
        stopwatch = _make_stopwatch(verbose=False)
        concatenate = functools.partial(
            xr.concat, data_vars='minimal', combine_attrs='no_conflicts'
        )
        if longitude and 'longitude' in self.cf.coordinates:
            coord = data.climo.coords['longitude']
            lon = coord.name
            if coord.size > 1 and not np.isclose(coord[-1], coord[0] + 360 * ureg.deg):
                edge = data.isel({lon: slice(-1, None)})
                edge = edge.climo.replace_coords({lon: coord[-1] - 360})
                data = concatenate((edge, data), dim=lon)
                stopwatch('longitude')

        # Add latitude coordinates at poles
        # WARNING: Containers of scalar quantities like [90 * ureg.deg] silently have
        # units stripped and are transformed to 1. Submit github issue?
        if latitude and 'latitude' in self.cf.coordinates:
            coord = data.climo.coords['latitude']
            if coord.size > 1:
                lat = coord.name
                parts = []
                if np.min(coord) < -80 * ureg.deg and -90 * ureg.deg not in coord:
                    part = data.isel({lat: slice(0, 1)})
                    part = part.climo.replace_coords({lat: [-90] * ureg.deg})
                    parts.append(part)
                parts.append(data)
                if np.max(coord) > 80 * ureg.deg and 90 * ureg.deg not in coord:
                    part = data.isel({lat: slice(-1, None)})
                    part = part.climo.replace_coords({lat: [90] * ureg.deg})
                    parts.append(part)
                data = concatenate(parts, dim=lat)
                stopwatch('latitude')

        # Add pressure coordinates at surface and "top of atmosphere"
        if vertical and 'vertical' in self.cf.coordinates:
            coord = data.climo.coords['vertical']
            if coord.climo.units.is_compatible_with('Pa'):
                lev = coord.name
                parts = []
                if 0 * ureg.hPa not in coord:
                    part = data.isel({lev: slice(0, 1)})
                    part = part.climo.replace_coords({lev: [0] * ureg.hPa})
                    parts.append(part)
                parts.append(data)
                if 1013.25 * ureg.hPa not in coord:
                    part = data.isel({lev: slice(-1, None)})
                    part = part.climo.replace_coords({lev: [1013.25] * ureg.hPa})
                    parts.append(part)
                data = concatenate(parts, dim=lev)
                if isinstance(data, xr.Dataset) and 'bounds' in coord.attrs:
                    bnds = data[coord.attrs['bounds']]
                    bnds[-2, 1] = bnds[-1, 0] = bnds[-1, :].mean()
                    bnds[0, 1] = bnds[1, 0] = bnds[0, :].mean()
                stopwatch('vertical')

        # Repair values at polar singularity
        # WARNING: Climopy loc indexing with units is *very* slow for now
        # WARNING: Xarray does not support boolean loc indexing
        # See: https://github.com/pydata/xarray/issues/3546
        if latitude:
            if isinstance(data, xr.DataArray):
                zero = (data.name,) if zero else ()
            else:
                zero = zero or ()
            # coord = data.climo.coords[lat]
            # loc = coord[np.abs(coord) == 90 * ureg.deg]
            coord = data.coords[lat]
            loc = coord[np.abs(coord) == 90]
            for da in data.climo._iter_data_vars():
                if da.name in zero and lat in da.coords:
                    # da.climo.loc[{lat: loc}] = 0
                    da.loc[{lat: loc}] = 0
            stopwatch('zero')

        return data

    @_CFAccessor._clear_cache
    def groupby(self, group, *args, **kwargs):
        """
        Return a unit-friendly `~xarray.DataArray.groupby` indexer. Dequantifies the
        `DataArray` before use and preserve attributes on the resulting coordinates.

        Parameters
        ----------
        *args, **kwargs
            Passed to `~xarray.DataArray.groupby`.

        Examples
        --------
        >>> ds = xr.tutorial.open_dataset('rasm', decode_times=False)
        >>> ds = ds.coarsen(x=25, y=25, boundary='trim').mean()
        >>> ds.Tair.attrs['units'] = 'degC'
        >>> T = ds.Tair.climo.quantify()
        >>> grp = ureg.kg * (T > 273 * ureg.K)  # arbitrary group with units
        >>> grp.name = 'above_freezing'
        >>> T.climo.groupby(grp).mean()
        <xarray.DataArray 'Tair' (above_freezing: 2)>
        <Quantity([-13.66380631  11.57033461], 'degree_Celsius')>
        Coordinates:
          * above_freezing  (above_freezing) int64 0 1
        """
        return self._cls_groupby(self.data, group, *args, **kwargs)

    @quack._keep_cell_attrs
    def _mean_or_sum(self, method, dim=None, skipna=None, weight=None, **kwargs):
        """
        Return an average or summation.
        """
        # NOTE: Unweighted mean or sum along scalar coords conceptually is an identity
        # operation, so ignore scalar coords. This is also important when running
        # integral() and _manage_coord_reductions adjusted the cell methods.
        data = self.truncate(**kwargs)
        dims = data.dims if dim is None else self._parse_dims(
            dim, ignore_scalar=True, include_no_coords=True,
        )
        if weight is not None:
            data = data.weighted(weight.climo.truncate(**kwargs))
        data = getattr(data, method)(dims, skipna=skipna, keep_attrs=True)
        data.climo.update_cell_methods({dims: method})
        return data

    @_CFAccessor._clear_cache
    @_manage_coord_reductions
    @docstring.inject_snippets(operator='mean')
    def mean(self, dim=None, **kwargs):
        """
        %(template_meansum)s

        Notes
        -----
        %(notes_avgmean)s
        """
        return self._mean_or_sum('mean', dim, **kwargs)

    @_CFAccessor._clear_cache
    @_manage_coord_reductions
    @docstring.inject_snippets(operator='sum')
    def sum(self, dim=None, **kwargs):
        """
        %(template_meansum)s
        """
        return self._mean_or_sum('sum', dim, **kwargs)

    @_CFAccessor._clear_cache
    @quack._while_dequantified
    def interp(
        self,
        indexers=None,
        method='linear',
        assume_sorted=False,
        kwargs=None,
        drop_cell_measures=True,
        **indexers_kwargs
    ):
        """
        Call `~xarray.DataArray.interp` with support for units and indexer aliases. Also
        preserve coordinate attributes, perform extrapolation for out-of-range
        coordinates by default, permit interpolating to different points as a function
        of other coordinates, and drop cell measures associated with the interpolated
        dimension.

        Parameters
        ----------
        *args, **kwargs
            Passed to `~xarray.DataArray.interp`.
        """
        kwargs = kwargs or {}
        kwargs.setdefault('fill_value', 'extrapolate')
        indexers, _ = self._parse_indexers(
            indexers, allow_kwargs=False, **indexers_kwargs
        )
        indexers = self._dequantify_indexers(indexers)
        return self._iter_by_indexer_coords(
            'interp',
            indexers,
            method=method,
            assume_sorted=assume_sorted,
            kwargs=kwargs,
            drop_cell_measures=drop_cell_measures,
        )

    @_CFAccessor._clear_cache
    def invert_hemisphere(self, which=None, invert=None):
        """
        Invert the sign of data in one or both hemispheres. This can be used e.g.
        to make meridional wind positive in the poleward direction. This is used
        internally by `~ClimoAccessor.sel_hemisphere`.

        Parameters
        ----------
        which : {'nh', 'sh', None}
            The hemisphere to invert. May be both or the northern or
            southern hemispheres. The default of ``which=None`` inverts both.
        invert : bool, str, or sequence of str, optional
            If boolean, indicates whether or not to invert the `~xarray.DataArray`
            (or, if this is a `~xarray.Dataset`, every array in the dataset). If
            sequence of strings, the `~xarray.DataArray` is only inverted if its
            ``name`` appears in the sequence (or, if this is an `~xarray.Dataset`,
            only those arrays in the dataset with matching names are inverted).
            For example, if the dataset contains the meridional wind ``'v'`` and
            potential vorticity ``'pv'``, one might use ``invert=('v', 'pv')``.
        """
        # Get latitude slice
        # NOTE: Exclude equator data from inversion
        data = self.data
        dim = self.cf._decode_name('latitude', 'coordinates')
        lat = data.coords[dim]
        sel = {}
        if which == 'sh':
            sel = {dim: slice(-90, np.max(lat[lat < 0]))}
        elif which == 'nh':
            sel = {dim: slice(np.min(lat[lat > 0]), 90)}
        elif which is not None:
            raise ValueError(f'Invalid {which=}. Must be sh or nh.')
        # Invert the data
        data = data.copy(deep=False)  # shallow copy by default
        if invert is None:
            invert = True
        elif isinstance(invert, str):
            invert = (invert,)
        for da in data.climo._iter_data_vars():
            if np.iterable(invert) and da.name not in invert:
                continue
            elif not invert:
                continue
            da.data = da.data.copy()  # deep copy when modifying the data
            da.loc[sel] = -1 * da.loc[sel]  # should preserve attributes
        return data

    @_CFAccessor._clear_cache
    def isel(
        self,
        indexers=None,
        drop=None,
        drop_cell_measures=False,
        **indexers_kwargs
    ):
        """
        Call `~xarray.DataArray.isel` with support for units and indexer aliases. Also
        permit selecting different points as a function of other coordinates.


        Parameters
        ----------
        *args, **kwargs
            Passed to `~xarray.DataArray.isel`.
        """
        indexers, _ = self._parse_indexers(
            indexers, allow_kwargs=False, **indexers_kwargs
        )
        return self._iter_by_indexer_coords(
            'isel', indexers, drop=drop, drop_cell_measures=drop or drop_cell_measures,
        )

    @_CFAccessor._clear_cache
    def replace_coords(self, indexers=None, **kwargs):
        """
        Return a copy with coordinate values added or replaced (if they already exist).
        If the input coordinates are `~xarray.DataArray`\\ s, the non-conflicting
        coordinate attributes are kept.

        Parameters
        ----------
        indexers : dict-like, optional
            The new coordinates.
        **kwargs
            Coordinates passed as keyword args.
        """
        indexers, _ = self._parse_indexers(
            indexers, include_no_coords=True, include_scalar=True, allow_kwargs=False, **kwargs  # noqa: E501
        )
        indexers_new = {}
        for name, coord in indexers.items():
            if isinstance(coord, tuple):
                raise ValueError('Coordinate data must be array-like.')
            # Build coordinate DataArray
            if not isinstance(coord, xr.DataArray):
                dims = () if quack._is_scalar(coord) else (name,)
                coord = xr.DataArray(coord, dims=dims, name=name)
            # Fix coordinate units and attributes
            # WARNING: Absolutely *critical* that DataArray unit string exactly
            # matches old one. Otherwise concatenate will strip unit attribute.
            coord = coord.climo.dequantify()
            if name in self.data.coords:
                prev = self.data.coords[name]
                if coord.climo._has_units:
                    coord = coord.climo.to_units(prev.climo.units)  # may raise error
                    coord.attrs['units'] = prev.attrs['units']  # *always* identical
                for key, value in prev.attrs.items():
                    coord.attrs.setdefault(key, value)  # avoid overriding
            indexers_new[name] = coord.climo.dequantify()
        return self.data.assign_coords(indexers_new)

    @_CFAccessor._clear_cache
    def reverse_hemisphere(self):
        """
        Reverse the direction and sign of the latitude coordinates (e.g., turn the
        southern hemisphere into the northern hemisphere). This is used internally
        by `~ClimoAccessor.sel_hemisphere`.
        """
        data = self.data
        data = data.copy(deep=False)
        dim = self.cf._decode_name('latitude', 'coordinates')
        with xr.set_options(keep_attrs=True):  # retain latitude attributes
            lat = -1 * data.coords[dim]
        data = data.climo.replace_coords({dim: lat})
        data = data.isel({dim: slice(None, None, -1)})  # retain original direction
        return data

    @_CFAccessor._clear_cache
    def sel(
        self,
        indexers=None,
        method=None,
        tolerance=None,
        drop=None,
        drop_cell_measures=False,
        **indexers_kwargs
    ):
        """
        Call `~xarray.DataArray.sel` with support for units and indexer aliases. Also
        permit selecting different points as a function of other coordinates.

        Parameters
        ----------
        *args, **kwargs
            Passed to `~xarray.DataArray.sel`.
        """
        indexers, _ = self._parse_indexers(
            indexers, allow_kwargs=False, **indexers_kwargs
        )
        indexers = self._dequantify_indexers(indexers)
        return self._iter_by_indexer_coords(
            'sel',
            indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
            drop_cell_measures=drop or drop_cell_measures,
        )

    @_CFAccessor._clear_cache
    def sel_hemisphere(self, which, invert=None):
        """
        Select a hemisphere or average of hemispheres. A single negative latitude
        is always included so that contours, lines, and whatnot extend to the equator
        exactly.

        Parameters
        ----------
        which : {'nh', 'sh', 'avg'}
            The hemisphere to select. May be the northern, southern, or an
            average of both hemispheres. This also controls the value of
            `which` passed to `~ClimoAccessor.invert_hemisphere`.
        invert : bool, str, or sequence of str, optional
            Passed to `~ClimoAccessor.invert_hemisphere`.
        """
        # Select and transform data
        # NOTE: keep data point on either side of equator so plots drawn
        # with resulting data will have a zero latitude point.
        data = self.data
        dim = self.cf._decode_name('latitude', 'coordinates')
        lat = data.coords[dim]
        which = which.lower()
        if invert is None:
            invert = False
        if which in ('nh', 'avg'):
            ndata = data.sel({dim: slice(np.max(lat[lat <= 0]), 90)})
        if which in ('sh', 'avg'):
            sdata = data.sel({dim: slice(-90, np.min(lat[lat >= 0]))})
            sdata = sdata.climo.invert_hemisphere('sh', invert)
            sdata = sdata.climo.reverse_hemisphere()
        # Possibly average hemispheres
        if which == 'sh':
            data = sdata
        elif which == 'nh':
            data = ndata
        elif which == 'avg':
            with xr.set_options(keep_attrs=True):
                data = 0.5 * (sdata + ndata)
        else:
            raise ValueError(f'Unknown hemisphere identifier {which!r}.')
        return data

    @_CFAccessor._clear_cache
    def sel_pair(self, key, *, modify=None):
        """
        Return selection from a pseudo "parameter" axis. "Parameter"
        axes are identified as any non-scalar coordinate whose associated
        `~ClimoDataArrayAccessor.cfvariable` has a "reference" value (e.g.,
        a coordinate named ``'forcing'`` with a "reference" value of ``0``).

        Parameters
        ----------
        key : str, optional
            The pair key. If the parameter axis is length 2, the key ``1`` returns
            the first position and the key ``2`` the second position. Otherwise, the
            key ``1`` returns the `~.cfvariable.CFVariable.reference` position and the
            key ``2`` is a no-op that returns the original data. To return the
            difference between keys ``2`` and ``1``, pass ``'anomaly'``. To
            return the ratio of key ``2`` over key ``1``, pass ``'ratio'``.
        modify : bool, optional
            Whether to modify the associated `~ClimoDataArrayAccessor.cfvariable` names
            by adding ``long_prefix`` and ``long_suffix`` attributes to the resulting
            `~xarray.DataArray`\\ (s). Default is ``False`` for variable(s) containing
            the substrings ``'force'`` or ``'forcing'`` and ``True`` otherwise.
        """
        # Find "anomaly-pair" axes and parametric axes
        # NOTE: This behavior differently depending on available parameters. If
        # pair is present then always select second minus first (either may or may not
        # be reference). Otherwise select value minus reference.
        key = str(key)
        if key not in ('1', '2', 'anom', 'anomaly', 'ratio'):
            raise ValueError(f'Invalid pair spec {key!r}.')
        data = self.data
        dims_param = self._find_params(return_reference=True)
        dims_pair = tuple(dim for dim in dims_param if data.sizes.get(dim, 0) == 2)
        if dims_pair:
            if len(dims_pair) > 1:
                warnings._warn_climopy(
                    f'Ambiguous anomaly-pair dimensions {dims_pair}. Using first.'
                )
            sels = tuple(
                {dims_pair[0]: data.coords[dims_pair[0]].values[i]} for i in range(2)
            )
        elif dims_param:
            if len(dims_param) > 1:
                warnings._warn_climopy(
                    f'Ambiguous parameter dimensions {tuple(dims_param)}. Using first.'  # noqa: E501
                )
            sels = (dims_param, {})
        else:
            raise ValueError('No anomaly-pair dimensions found.')

        # Make selection and repair cfvariable
        # NOTE: We are careful here to track parent_name variables found in
        # coordinates, i.e. variables to which we applied _find_extrema.
        prefix = suffix = None
        if key == '1':
            prefix = 'unforced'
            result = data.sel(sels[0])
        elif key == '2':
            prefix = 'forced'
            result = data.sel(sels[1])
        else:
            suffix = 'anomaly'
            with xr.set_options(keep_attrs=True):
                name = data.attrs.get('parent_name', None)
                data0 = data.climo.sel(sels[0])
                data1 = data.climo.sel(sels[1])
                result = data1 / data0 if key == 'ratio' else data1 - data0
                if name and name in data.coords and name not in result.coords:
                    coord = 0.5 * (data0.coords[name] + data1.coords[name])
                    result.coords[name] = coord

        # Add prefixes and suffixes
        for da in result.climo._iter_data_vars():
            attrs = da.attrs
            combine = lambda *args: ' '.join(filter(None, args))  # noqa: E731
            if modify is None:
                skip = re.search('(force|forcing)', da.name or '')
            else:
                skip = not modify
            if skip:
                continue
            if prefix:
                attrs['long_prefix'] = combine(prefix, attrs.get('long_prefix'))
            if suffix:
                attrs['long_suffix'] = combine(attrs.get('long_suffix'), suffix)

        return result

    @_CFAccessor._clear_cache
    def sel_time(self, date=None, **kwargs):
        """
        Return an `~xarray.DataArray` or `~xarray.Dataset` with the time coordinate
        filtered to times matching some datetime component. For details, see the
        `xarray documentation on virtual datetime coordinates \
        <http://xarray.pydata.org/en/stable/time-series.html>`__.

        Parameters
        ----------
        date : date-like, optional
            Itemized selection of dates. Data type should match the time coordinate
            data type (e.g. `numpy.datetime64`).
        year, month, day, hour, minute, second, dayofyear, week, dayofweek, weekday
            The datetime component, e.g. ``year=2000`` or ``season='JJA'``.
        """
        data = self.data
        try:
            time = self.cf._decode_name('time', 'coordinates')
        except KeyError:
            raise RuntimeError('Time dimension not found.')
        if date is not None:
            data = data.sel({time: date})
        for key, value in kwargs.items():
            if value is None:
                continue
            data = data.sel({time: data[f'{time}.{key}'] == value})
        return data

    @_CFAccessor._clear_cache
    def standardize_coords(self, verbose=False):
        """
        Infer and standardize coordinates to satisfy CF conventions with the help of
        `~cf_xarray.CFAccessor.guess_coord_axis` and `cf_xarray.CFAccessor.rename_like`.
        This function does the following:

        * Adds ``longitude`` and ``latitude`` standard names and ``degrees_east``
          and ``degrees_north`` units to detected ``X`` and ``Y`` axes.
        * Ensures detected longitude and latitude coordinates are designated
          as ``X`` and ``Y`` axes if none are present.
        * Adds ``positive`` direction attribute to detected ``Z`` axes so they
          are also interpted as ``vertical`` coordinates.
        * Enforces vertical coordinate units of kilometers, hectopascals, and kelvin,
          for height-like, pressure-like, and temperature-like data, respectively.
        * Renames longitude, latitude, vertical, and time coordinate names
          to ``'lon'``, ``'lat'``, ``'lev'``, and ``'time'``, respectively.
        * Renames coordinate bounds to the coordinate names followed by a
          ``'_bnds'`` suffix and removes all attributes from bounds variables.

        Note that this function never overwrites existing attributes.

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, print statements are issued.
        """
        # Update 'axis' attributes and 'longitude', 'latitude' standard names and units
        for coord in self.data.coords.values():
            if 'cartesian_axis' in coord.attrs:  # rename non-standard axis specifier
                coord.attrs.setdefault('axis', coord.attrs.pop('cartesian_axis'))
        data = self.cf.guess_coord_axis(verbose=verbose)

        # Ensure unique longitude and latitude axes are designated as X and Y
        for axis, coord in zip(('X', 'Y'), ('longitude', 'latitude')):
            da = self.cf._get_item(coord, 'coordinates')
            if da is not None and axis not in data.climo.cf.axes:
                da.attrs['axis'] = axis
                if verbose:
                    print(f'Set {coord} coordinate {da.name!r} axis type to {axis!r}.')

        # Manage all Z axis units and interpret 'positive' direction if unset.
        # Otherwise guess_coord_axis does not detect the 'positive' attribute.
        for name in data.climo.cf.axes.get('Z', []):
            da = data.climo.coords[name]  # climopy makes unit-transformable copy
            units = data.coords[name].attrs.get('units', None)
            units = units if units is None else decode_units(units)
            positive = None
            if units is None:
                pass
            elif units == 'level' or units == 'layer':  # ureg.__eq__ handles strings
                positive = 'up'  # positive vertical direction is increasing values
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
                    'Assumed up.'
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
        coords_prev = data.climo.cf.coordinates
        sample = xr.Dataset(coords=coords)
        data = data.climo.cf.rename_like(sample)
        if verbose:
            coords_curr = data.climo.cf.coordinates
            for coord, names_curr in coords_curr.items():
                names_prev = coords_prev.get(coord, [])
                for name_prev, name_curr in zip(names_prev, names_curr):
                    if name_prev != name_curr:
                        print(f'Renamed coordinate {coord!r} name {name_prev!r} to {name_curr!r}.')  # noqa: E501

        # Manage bounds variables
        for name, da in data.coords.items():
            if isinstance(data, xr.Dataset):
                # Delete bounds indicators when the bounds variable is missing
                bounds = da.attrs.get('bounds')
                if bounds and bounds not in data:
                    del da.attrs['bounds']
                    if verbose:
                        print(f'Deleted coordinate {name!r} bounds attribute {bounds!r} (bounds variable not present in dataset).')  # noqa: E501
                # Infer unset bounds attributes
                for suffix in ('bnds', 'bounds'):
                    bounds = name + '_' + suffix
                    if bounds in data and 'bounds' not in da.attrs:
                        da.attrs['bounds'] = bounds
                        if verbose:
                            print(f'Set coordinate {name!r} bounds to discovered bounds-like variable {bounds!r}.')  # noqa: E501
                # Standardize bounds name and remove attributes (similar to rename_like)
                bounds = da.attrs.get('bounds')
                if bounds and bounds != (bounds_new := da.name + '_bnds'):
                    da.attrs['bounds'] = bounds_new
                    data = data.rename_vars({bounds: bounds_new})
                    if verbose:
                        print(f'Renamed bounds variable {bounds!r} to {bounds_new!r}.')
                # Delete all bounds attributes as recommended by CF manual
                if bounds:
                    data[bounds_new].attrs.clear()
            # Delete bounds variables for DataArrays, to prevent CF warning issue
            elif 'bounds' in da.attrs:
                del da.attrs['bounds']

        return data

    @_CFAccessor._clear_cache
    def truncate(self, bounds=None, *, source=None, ignore_extra=False, **kwargs):
        """
        Restrict the coordinate range using `ClimoAccessor.interp`. Conceptually,
        inserts conincident centers and boundaries that mark the new edges of the
        coordinate range. The cell measure weights are redistributed accordingly.

        Parameters
        ----------
        bounds : dict-like, optional
            The bounds specifications. For e.g. the latitude dimension `lat`, the
            entries should look like ``lat_min=min_value``, ``lat_max=max_value``,
            ``lat_lim=(min, max)``, or the shorthand ``lat=(min, max)``.
        source : xarray.Dataset, optional
            The source dataset. Required to retrieve coordinate bounds in order
            to adjust cell measure weights.
        **kwargs
            The bounds specifications passed as keyword args.

        Notes
        -----
        If cell measures are present, we assume internal level boundaries are unchanged
        by edges of coordinate range. So, cell measures on new coordinate edges are just
        scaled-down version of old cell measures. By contrast, if cell measures are
        missing, boundaries and measures auto-generated by `~ClimoAccessor.coords` will
        assume new boundary is halfway between new edge and old internal boundary.
        """
        # Parse truncation args
        # NOTE: Data attributes are conserved during sel, interp, concat operations.
        # NOTE: This uses the unit-friendly accessor sel method. Range is limited
        # *exactly* by interpolating onto requested bounds.
        data = self.data
        source = source or data
        bounds = bounds or {}
        bounds.update(kwargs)
        bounds, kwargs = source.climo._parse_truncate_args(**bounds)
        if kwargs and not ignore_extra:
            raise ValueError(f'truncate() got unexpected keyword args {kwargs}.')
        if any(_.size > 2 for _ in bounds.values()):
            raise ValueError(f'truncate() args {kwargs} yield non-scalar bounds.')

        # Iterate through truncations
        # NOTE: The below uses uses _iter_by_indexer_coords
        for dim, bound in bounds.items():
            dim = re.sub(r'_lim\Z', '', dim)
            data_orig = data
            coord_orig = data.coords[dim]  # must be unquantified
            bnds_orig = source.climo.coords._get_bounds(coord_orig, sharp_cutoff=True)
            attrs = coord_orig.attrs.copy()

            # Interpolate to new edges. When 'truncating' outside the coordinate range,
            # simply replace edge coordinates but keep everything else the same.
            lo, hi = bound.values.squeeze()  # pull out of array
            edges = [None, None]
            center = data.climo.sel({dim: slice(lo, hi)})
            if center.sizes[dim] == 0:
                raise ValueError(f'Invalid bounds {dim}=({lo!r}, {hi!r}).')
            for idx, val in enumerate((lo, hi)):
                if val is None or val in coord_orig:
                    continue
                if coord_orig.min() < val < coord_orig.max():
                    edges[idx] = data.climo.interp({dim: val}, drop_cell_measures=False)
                else:
                    sel = coord_orig.min() if val < coord_orig.min() else coord_orig.max()  # noqa: E501
                    edges[idx] = data.climo.sel({dim: sel}).climo.replace_coords({dim: val})  # noqa: E501

            # Concatenate efficiently
            parts = tuple(_ for _ in (edges[0], center, edges[1]) if _ is not None)
            concatenate = functools.partial(
                xr.concat,
                dim=dim,
                coords='minimal',
                compat='override',
                combine_attrs='no_conflicts'
            )
            if isinstance(data, xr.Dataset):
                concatenate = functools.partial(concatenate, data_vars='minimal')
            data = concatenate(parts)
            coord = data.coords[dim]
            coord.attrs.update(attrs)

            # Delete old bounds variables
            # TODO: Also preserve bounds like we preserve cell measures
            bounds = coord.attrs.get('bounds')
            if bounds and isinstance(data, xr.Dataset) and bounds in data:
                data = data.drop_vars(bounds)

            # Update relevant cell measures with scale factor. For example, if
            # we are truncating latitude, only scale 'width', 'area', and 'volume'
            try:
                coordinate = self.cf._encode_name(dim, 'coordinates')
            except KeyError:
                continue
            for idx, offset in ((0, 1), (-1, -1)):
                if np.any(coord_orig == coord[idx]):
                    continue  # we did nothing
                loc, = np.where(coord_orig == coord[idx + offset])
                if loc.size != 1:
                    continue  # found double coordinates, unclear how to proceed
                loc, = loc - offset

                # Get scale factors
                factor_edge = None
                if 0 <= loc < bnds_orig.shape[0]:
                    bnds = bnds_orig[loc, :]
                    bound = bnds[idx + 1]  # the "inner" bound
                    if bnds.min() < coord[idx] < bnds.max():
                        factor_edge = np.abs((bound - coord[idx]) / (bnds[1] - bnds[0]))
                bnds = bnds_orig[loc + offset, :]
                bound = 0.5 * (coord[idx] + coord[idx + offset])
                factor_offset = np.abs((bound - bnds[idx]) / (bnds[1] - bnds[0]))

                # Adjust cell measures
                # NOTE: This strictly prevents adding mass. "Truncating" to points
                # outside of coordinate range only re-distributes edge weights.
                for measure, (name,) in self.cf.cell_measures.items():
                    if coordinate not in CELL_MEASURE_COORDS[measure]:
                        continue
                    weight = data.coords[name]
                    weight_orig = data_orig.coords[name]
                    weight[{dim: idx}] = (
                        factor_offset * weight_orig[{dim: loc + offset}]
                        + (factor_edge * weight_orig[{dim: loc}] if factor_edge else 0)
                    )
                    weight[{dim: idx + offset}] = (
                        (1 - factor_offset) * weight_orig[{dim: loc + offset}]
                    )

        return data

    @docstring.inject_snippets()
    def update_cell_attrs(self, other):
        """
        Update `cell_methods` and `cell_measures` attributes from another object onto
        the `xarray.DataArray` or every array in the `xarray.Dataset`. Used internally
        throughout `climopy`.

        %(warning_inplace)s
        """
        # NOTE: No longer track CFVARIABLE_ARGS attributes. Too complicated, and
        # yields weird behavior like adding back long_name='zonal wind' after 'argmax'
        # TODO: Stop defining cell measures for whole dataset, just like cell methods,
        # to accomodate situation with multiple grids.
        # WARNING: For datasets, we use data array with longest cell_methods, to try to
        # accomodate variable derivations from source variables with identical methods
        # and ignore variables like 'bounds' with only partial cell_methods. But this
        # is ugly kludge with side effects... should be refined.
        if isinstance(other, (xr.DataArray, xr.Dataset)):
            pass
        elif isinstance(other, ClimoAccessor):
            other = other.data
        else:
            raise TypeError('Other must be a DataArray, Dataset, or ClimoAccessor.')
        other_methods = other_measures = other
        if isinstance(other, xr.Dataset):  # get longest cell_methods
            other_methods = max(
                other.values(),
                key=lambda _: len(_.attrs.get('cell_methods') or ''),
                default=other,  # no attributes found anyway
            )
        for da in self._iter_data_vars():
            for other, attr in zip((other_methods, other_measures), ('cell_methods', 'cell_measures')):  # noqa: E501
                if value := self.cf._encode_attr(
                    other.attrs.get(attr, None), da.attrs.get(attr, None)
                ):
                    da.attrs[attr] = value

    @docstring.inject_snippets()
    def update_cell_methods(self, methods=None, **kwargs):
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

        %(warning_inplace)s
        """
        methods = methods or {}
        methods.update(kwargs)
        if not methods:
            return
        for da in self._iter_data_vars():
            if da.climo._is_coordinate_bounds:
                continue
            da.attrs['cell_methods'] = self.cf._encode_attr(
                da.attrs.get('cell_methods', None), methods.items()
            )

    @property
    def cf(self):
        """
        A version of `~cf_xarray.accessors.CFAccessor` that supports automatic cacheing
        for speed boosts. The accessor instance is preserved.
        """
        cf = self._cf_accessor
        if cf is None:
            cf = self._cf_accessor = self._cls_cf(self.data, self.variable_registry)
        return cf

    @property
    def coords(self):
        """
        A version of `~xarray.DataArray.coords` that returns always-quantified
        coordinate variables or variables *transformed* from the native coordinates
        using `ClimoDataArrayAccessor.to_variable` (e.g. ``'latitude'`` to
        ``'meridional_coordinate'``). Coordinates can be requested by their name (e.g.
        ``'lon'``), axis attribute (e.g. ``'X'``), CF coordinate name (e.g.
        ``'longitude'``), or `~.cfvariable.CFVariableRegistry` identifier.

        The coordinate top boundaries, bottom boundaries, or thicknesses can be returned
        by appending the key with ``_top``, ``_bot``, or ``_del`` (or ``_delta``),
        respectively, or the N x 2 bounds array can be returned by apppending ``_bnds``
        (or ``_bounds``). If explicit boundary variables do not exist, boundaries are
        inferred by assuming datetime-like coordinates represent end-points of temporal
        cells and numeric coordinates represent center-points of spatial cells (i.e.,
        numeric coordinate bounds are found halfway between the coordinates).
        """
        # NOTE: Creating class instance is O(100 microseconds). Very fast.
        # NOTE: Quantifying in-place: https://github.com/pydata/xarray/issues/525
        return self._cls_coords(self.data, self.variable_registry)

    @property
    def data(self):
        """
        Redirect to the underlying `xarray.Dataset` or `xarray.DataArray`.
        """
        return self._data

    @property
    def loc(self):
        """
        Call `~xarray.DataArray.loc` with support for `pint.Quantity` indexers
        and assignments and coordinate name aliases.
        """
        return self._cls_loc(self.data)

    @property
    @_CFAccessor._clear_cache
    def parameter(self):
        """
        The coordinate `~xarray.DataArray` for the "parameter" axis. Detected as the
        first coordinate whose `~ClimoDataArrayAccessor.cfvariable` has a non-empty
        ``reference`` attribute.
        """
        coords = self._find_params()
        if len(coords) > 1:
            warnings._warn_climopy(
                f'Ambiguous parameter dimensions {tuple(coords)}. Using first.'
            )
        return coords[tuple(coords)[0]]

    @property
    @_CFAccessor._clear_cache
    def parameters(self):
        """
        A tuple of the coordinate `~xarray.DataArray`\\ s for "parameter" axes.
        Detected as coordinates whose `~climoDataArrayAccessor.cfvariable`\\ s
        have non-empty ``reference`` attributes.
        """
        return tuple(self._find_params(allow_empty=True).values())

    @property
    def variable_registry(self):
        """
        The active `~.cfvariable.CFVariableRegistry` used to look up variables
        with `~ClimoDataArrayAccessor.cfvariable`.
        """
        return self._variable_registry

    @variable_registry.setter
    def variable_registry(self, reg):
        if not isinstance(reg, CFVariableRegistry):
            raise ValueError('Input must be a CFVariableRegistry instance.')
        self._variable_registry = reg


@xr.register_dataarray_accessor('climo')
class ClimoDataArrayAccessor(ClimoAccessor):
    """
    Accessor for `xarray.DataArray`\\ s. Includes methods for working with `pint`
    quantities and `~.cfvariable.CFVariable` variables, several stub functions for
    integration with free-standing climopy functions (similar to numpy design), and an
    interface for transforming one physical variable to another. Registered under the
    name ``climo`` (i.e, usage is ``data_array.climo``). The string representation of
    this accessor displays its `~ClimoDataArrayAccessor.cfvariable` information (if the
    data array name is found in the `~ClimoAccessor.variable_registry`).
    """
    _cls_cf = _CFDataArrayAccessor
    _cls_groupby = _DataArrayGroupByQuantified
    _cls_coords = _DataArrayCoordsQuantified
    _cls_loc = _DataArrayLocIndexerQuantified

    @_CFAccessor._clear_cache
    def __repr__(self):
        return f'<climopy.ClimoDataArrayAccessor>({self._cf_repr(brackets=False)})'

    def __contains__(self, key):
        """
        Is a valid derived coordinate.
        """
        return key in self.coords

    def __dir__(self):
        """
        Support name lookup and completion. Derivations and aliases are excluded.
        """
        data = self.data
        try:
            cfattrs = dir(self.cfvariable)
        except AttributeError:
            cfattrs = ()
        return sorted({*dir(type(self)), *cfattrs, *data.attrs, *data.coords})

    def __getattr__(self, attr):
        """
        Return a coordinate, attribute, or cfvariable property.
        """
        if attr in super().__dir__():  # can happen if @property triggers error
            return super().__getattribute__(attr)  # trigger builtin AttributeError
        try:
            return self.data.attrs[attr]
        except KeyError:
            pass
        try:
            return self.coords[attr]
        except KeyError:
            pass
        try:
            return getattr(self.cfvariable, attr)
        except AttributeError:
            pass
        raise AttributeError(
            f'Attribute {attr!r} does not exist and is not a valid coordinate or '
            f'DataArray or CFVariable attribute, or a CFVariable was not found for '
            f'the name {self.data.name!r}.'
        )

    @_CFAccessor._clear_cache
    def __getitem__(self, key):
        """
        Return a quantified coordinate or a selection along dimensions with translated
        dictionary indexing. Translates CF names and `~.cfvariable.CFVariableRegistry`
        identifiers.
        """
        if isinstance(key, str):
            return self.coords[key]
        else:  # see also .loc.__getitem__
            key, _ = self._parse_indexers(self._expand_ellipsis(key), include_no_coords=True)  # noqa: E501
            return self.data[key]

    @_CFAccessor._clear_cache
    def __setitem__(self, key, value):
        """
        Set values along dimensions with translated dictionary indexing.
        """
        key, _ = self._parse_indexers(self._expand_ellipsis(key), include_no_coords=True)  # noqa: E501
        self.data[key].climo._assign_value(value)

    def _assign_value(self, value):
        """
        Assign value to the entire `xarray.DataArray`. Generally the underlying
        data should be a sliced view of another `xarray.DataArray`.
        """
        # Standardize value
        data = self.data
        if isinstance(value, xr.DataArray):
            if value.climo._has_units and self._has_units:
                value = value.climo.to_units(self.units)
            value = value.data
        if isinstance(value, pint.Quantity):
            if not self._has_units:
                raise ValueError('Cannot assign pint quantities to data with unclear units.')  # noqa: E501
            value = value.to(self.units)
            if not self._is_quantity:
                value = value.magnitude  # apply to dequantified data
        elif self._is_quantity:
            value = value * data.data.units
        if data.shape:  # i.e. has a non-empty shape tuple, i.e. not scalar
            value = np.atleast_1d(value)  # fix assignment of scalar pint quantities
        data[...] = value

    def _cf_repr(self, brackets=True, varwidth=None, maxlength=None, padlength=None, **kwargs):  # noqa: E501
        """
        Get representation even if `cfvariable` is not present.
        """
        # Get content inside CFVariable(...) repr
        try:
            var = self._cf_variable(**kwargs)
        except AttributeError:
            repr_ = self.data.name or 'unknown'
        else:
            repr_ = REGEX_REPR_PAREN.match(repr(var)).group(1)

        # Align names and truncate key=value pairs
        padlength = padlength or 0
        if varwidth is not None and (m := REGEX_REPR_COMMA.match(repr_)):
            name, _, info = m.groups()  # pad between canonical name + subsequent info
            repr_ = name[:varwidth] + ',' + ' ' * (varwidth - len(name)) + info
        if maxlength is not None and len(repr_) > maxlength - padlength:
            repr_ = repr_[:maxlength - padlength - 4]
            repr_ = repr_[:repr_.rfind(' ')] + ' ...'
        if brackets:
            repr_ = REGEX_REPR_COMMA.sub(r'\1\2<\3>', repr_)
        return ' ' * padlength + repr_

    def _cf_variable(self, use_kw_attrs=True, use_cf_attrs=True):
        """
        Return a `CFVariable` with options for excluding information.
        """
        # Get the name
        data = self.data.copy(deep=False)
        data.coords[data.name] = data  # whoa dude... this is so CF searches self
        self = data.climo
        name = data.name

        # Get override attributes
        kwargs = {}
        if use_kw_attrs:
            for key, val in data.attrs.items():
                if key in CFVARIABLE_ARGS:
                    kwargs[key] = val

        # Get cell methods that modify descriptive names
        if use_cf_attrs:
            # Get methods applied to each coordinate by reading cell_methods and
            # scalar coordinates. Apply units if it can be done safely.
            # WARNING: Looking up cf attributes can be slow when looping over dataset
            # variables so optionally skip this (see _CFAccessor above).
            # WARNING: To detect e.g. latitude='argmax' reductions in the cell methods
            # list this requires the data array to match the original coordinate name.
            # For example you cannot change the name from 'latitude' to 'jet_latitude'.
            methods = self.cf._decode_attr(data.attrs.get('cell_methods', ''))
            cf_name = None
            for coord, da in data.coords.items():
                if coord == name:
                    attrs = ('coordinates', 'cell_measures')
                else:
                    attrs = ('coordinates',)
                try:
                    key = self.cf._encode_name(coord, *attrs)
                except KeyError:
                    continue
                if coord == name:
                    cf_name = key
                if any(coord in dims for dims, _ in methods):
                    kwargs[key] = [m for dims, m in methods if coord in dims]
                elif da.size == 1 and not da.isnull():
                    value = da.item()
                    if 'units' in da.attrs and np.issubdtype(da.data.dtype, np.number):
                        value *= decode_units(da.attrs['units'])
                    kwargs[key] = [value]  # exact coordinate

            # Find if DataArray corresponds to a standard coordinate or cell measure,
            # with corresponding standard cfvariables registered by climopy. For
            # example this will return CFVariable('latitude') for a variable with
            # deg_E units or CFVariable('cell_area') for a variable defined as the
            # cell area even if their name and standard_name do not match. This also
            # handles e.g. latitude='argmax' reduction methods in which the DataArray
            # corresponds to some coordinate measure of a different physical variable.
            parent_name = data.attrs.get('parent_name', None)
            if cf_name in kwargs and not self._is_coordinate_bounds:
                if parent_name is None:
                    raise RuntimeError(f'Unknown parent name for coordinate {name!r}.')
                if parent_name not in data.coords:
                    raise RuntimeError(f'Parent coordinate {parent_name!r} not found.')
                name = parent_name
                parent = data.coords[name]
                for key in ('long_name', 'short_name', 'standard_name'):
                    if key in parent.attrs:
                        kwargs[key] = parent.attrs[key]
                    else:
                        kwargs.pop(key, None)
            elif cf_name is None:  # this is a physical variable
                pass
            else:  # this is a coordinate or cell measure
                name = cf_name

        # Create the CFVariable
        for identifier in (name, kwargs.pop('standard_name', None)):
            if identifier is None:
                continue
            try:
                return self.variable_registry(identifier, accessor=self, **kwargs)
            except KeyError:
                pass
        raise AttributeError(f'CFVariable not found for name {name!r}.')

    def _expand_ellipsis(self, key):
        """
        Expand out ellipses of tuple indexer. Reproduces xarray internals.
        """
        if not isinstance(key, dict):
            labels = _expand_indexer(key, self._data.ndim)
            key = dict(zip(self._data.dims, labels))
        return key

    def _budget_reduce_kwargs(self, method):
        """
        Automatically determine reduction keyword arguments for the ``'latitude'``
        or ``'strength``' of an energy or momentum budget term.
        """
        # WARNING: Flux convergence terms are subgroups of flux terms, not tendency
        kw = {}
        reg = self.variable_registry
        name = self.data.name
        content = name in reg.energy or name in reg.momentum
        tendency = name in reg.energy_flux or name in reg.acceleration
        transport = name in reg.meridional_energy_flux or name in reg.meridional_momentum_flux  # noqa: E501
        if not content and not transport and not tendency:
            raise ValueError(f'Invalid parameter {name!r}.')
        if self.cf.sizes.get('vertical', 1) > 1:
            kw['vertical'] = 'int'  # NOTE: order of reduction is important
        kw['longitude'] = 'avg' if content or tendency else 'int'
        if method == 'strength':
            kw['latitude'] = 'absmax'
        elif method == 'latitude':
            kw['latitude'] = 'absargmax'
        else:
            raise ValueError(f'Invalid energy or momentum reduce method {method!r}.')
        return kw

    @_CFAccessor._clear_cache
    @quack._while_quantified
    def reduce(
        self,
        indexers=None,
        source=None,
        invert=False,
        weight=None,
        mask=None,
        **kwargs
    ):
        """
        Reduce the dimension of a `xarray.DataArray` with arbitrary method(s).

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values representing the
            "reduction methods" for the dimensions. Options are as follows:

            ===============  ==========================================================
            Reduction mode   Description
            ===============  ==========================================================
            array-like       The value(s) at this location using ``self.interp``.
            var-spec         Variable name passed to ``self.get``, e.g. ``'trop'``.
            ``'autocorr'``   Autocorrelation along the dimension.
            ``'autocovar'``  Autocovariance along the dimension.
            ``'centroid'``   The centroid. Note dimension could be e.g. ``'volume'``.
            ``'hist'``       Histogram along the dimension.
            ``'int'``        Integral along the dimension.
            ``'avg'``        Weighted mean along the dimension.
            ``'anom'``       Weighted anomaly along the dimension.
            ``'lcumint'``    Cumulative integral from the left.
            ``'rcumint'``    Cumulative integral from the right.
            ``'lcumavg'``    Cumulative average from the left.
            ``'rcumavg'``    Cumulative average from the right.
            ``'lcumanom'``   Anomaly w.r.t. cumulative average from the left.
            ``'rcumanom'``   Anomaly w.r.t. cumulative average from the right.
            ``'mean'``       Simple arithmetic mean.
            ``'sum'``        Simple arithmetic sum.
            ``'min'``        Local minima along the dimension.
            ``'max'``        Local maxima along the dimension.
            ``'absmin'``     Global minimum along the dimension.
            ``'absmax'``     Global maximum along the dimension.
            ``'argmin'``     Location(s) of local minima along the dimension.
            ``'argmax'``     Location(s) of local maxima along the dimension.
            ``'argzero'``    Location(s) of zeros along the dimension.
            ``'absargmin'``  Location of global minimum along the dimension.
            ``'absargmax'``  Location of global maximum along the dimension.
            ``'slope'``      Least-squares fit linear slope along the dimension.
            ``'timescale'``  Least-squares fit e-folding timescale along the dimension.
            ===============  ==========================================================

        source : `xarray.Dataset`, optional
            The source dataset. Required for complex dimension reduction methods,
            e.g. interpolating to the tropopause with ``lev='trop'``. Also required
            for retrieving averaging weights, e.g. coordinate bounds ``lev_bnds``.
        invert : bool, optional
            Whether to invert the data before and after the reduction. Can be
            useful for e.g. averages of timescales or wavelengths.
        weight : str or `xarray.DataArray`, optional
            Additional weighting parameter name or `xarray.DataArray`, used for
            averages and integrations. Mass weighting is applied automatically.
        mask : {None, 'land', 'sea', 'trop', 'pv'}, optional
            The 2-dimensional mask to apply before taking the weighted average.
            Presets will be added to this.
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
            like `find` and `rednoisefit`.

        Returns
        -------
        xarray.DataArray
            The data with relevant dimension(s) reduced.
        """
        # Initial stuff
        data = self.data
        if invert:
            with xr.set_options(keep_attrs=True):
                data = 1.0 / data
        name = data.name
        dims = data.dims
        source = source or data
        hist_keys = ('bins',)
        avg_keys = ('skipna', 'weight')
        cov_keys = ('lag', 'ilag', 'maxlag', 'imaxlag')
        fit_keys = ('maxlag', 'imaxlag', 'maxlag_fit', 'imaxlag_fit')
        find_keys = ('diff', 'which', 'centered', 'sep', 'seed', 'ntrack', 'dim_track')
        reduce_methods = {
            'autocorr': ('autocorr', cov_keys, {}),
            'autocovar': ('autocovar', cov_keys, {}),
            'centroid': ('centroid', (), {}),
            'hist': ('hist', hist_keys, {}),
            'int': ('integral', avg_keys, {}),
            'avg': ('average', avg_keys, {}),
            'anom': ('anomaly', avg_keys, {}),
            'lcumint': ('cumintegral', avg_keys, {}),
            'rcumint': ('cumintegral', avg_keys, {'reverse': True}),
            'lcumavg': ('cumaverage', avg_keys, {}),
            'rcumavg': ('cumaverage', avg_keys, {'reverse': True}),
            'lcumanom': ('cumanomaly', avg_keys, {}),
            'rcumanom': ('cumanomaly', avg_keys, {'reverse': True}),
            'mean': ('mean', avg_keys, {}),
            'sum': ('sum', avg_keys, {}),
            'min': ('min', find_keys, {}),
            'max': ('max', find_keys, {}),
            'absmin': ('absmin', find_keys, {}),
            'absmax': ('absmax', find_keys, {}),
            'argmin': ('argmin', find_keys, {}),
            'argmax': ('argmax', find_keys, {}),
            'argzero': ('argloc', find_keys, {'value': 0}),
            'absargmin': ('absargmin', find_keys, {}),
            'absargmax': ('absargmax', find_keys, {}),
            'slope': ('slope', (), {}),
            'timescale': ('timescale', fit_keys, {}),
        }
        if mask is not None:
            raise NotImplementedError('Mask application not yet implemented')

        # Parse indexers
        # NOTE: Prefer dataset here to allow for things like lat_min='ehf_lat'
        # NOTE: Include 'pseudo' indexers of already-reduced dims for integration
        indexers, kwargs = self._parse_indexers(
            indexers, include_scalar=True, include_pseudo=True, **kwargs
        )
        if isinstance(weight, str):
            if not isinstance(source, xr.Dataset):  # supplied by get
                raise ValueError(f'Dataset required to infer weighting {weight!r}.')
            weight = source.climo[weight]

        # Parse truncation args
        kw_trunc, kwargs = source.climo._parse_truncate_args(**kwargs)
        if kw_trunc:
            sample = tuple(kw_trunc.values())[0]
            dims_sample = sample.dims[1:]  # exclude 'startstop'
            datas = np.empty(sample.shape[1:], dtype='O')
            coords = {key: c for key, c in sample.coords.items() if key != 'startstop'}
        else:
            sample = None
            dims_sample = ('track',)
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
            isel_trunc = dict(zip(dims_sample, idx))
            isel_data = {dim: i for dim, i in zip(dims_sample, idx) if dim != 'track'}
            ikw_trunc = {k: tuple(v.isel(isel_trunc).data) for k, v in kw_trunc.items()}
            idata = data.isel(isel_data).climo.truncate(source=source, **ikw_trunc)
            iweight = weight
            if iweight is not None:
                iweight = data.isel(isel_data).climo.truncate(source=source, **ikw_trunc)  # noqa: E501

            # Single dimension reductions
            # WARNING: Need to include *coords* so we can 'reduce' singleton lon
            for dim, method in indexers.items():
                # Various simple reduction modes
                # NOTE: Integral does not need dataset because here we are only
                # integrating unknown dimensions; attenuated spatial ones earlier.
                # TODO: Add '_hist' entry to __getitem__ that (1) gets jet latitudes
                # with argmax and (2) calls hist on the resulting latitudes.
                if isinstance(method, str) and method in reduce_methods:
                    method, keys, kw = reduce_methods[method]
                    for key in set(keys) & set(kwargs):
                        kw[key] = kwargs[key]
                    if 'weight' in keys:
                        kw['weight'] = iweight
                    idata = getattr(idata.climo, method)(dim, **kw)
                    used_kw |= kw.keys()

                # Select single or multiple points with interpolation
                # For example: climo.get('dtdy', lat='ehf_lat')
                else:
                    loc = getattr(method, 'magnitude', method)
                    if dim in self.cf.coordinates.get('time', ()):
                        idata = idata.climo.sel_time({dim: loc})
                    elif quack._is_numeric(loc):  # i.e. not datetime, string, etc.
                        idata = idata.climo.interp({dim: loc})
                    else:
                        msg = (
                            f'Invalid reduce method or variable spec {loc!r}. Valid '
                            'reduce methods are: ' + ', '.join(reduce_methods) + '.'
                        )
                        if not isinstance(source, xr.Dataset):
                            raise ValueError(
                                msg + ' Variable specs require a source dataset.'
                            )
                        try:
                            loc = source.climo.get(loc)
                        except KeyError:
                            raise ValueError(
                                msg + ' Failed to treat {loc!r} as variable spec.'
                            )
                        try:
                            idata = idata.climo.interp({dim: loc})
                        except ValueError:
                            raise ValueError(
                                'Falied to interpolate to variable spec {loc!r}.'
                            )

            # Add to list of reductions along different subselections
            datas[idx] = idata

        # Detect invalid kwargs
        extra_kw = set(kwargs) - used_kw
        if extra_kw:
            raise ValueError('Unexpected kwargs: ' + ', '.join(map(repr, extra_kw)))

        # Concatente with combine_nested, then fix weird dimension reordering
        data = xr.combine_nested(
            datas.tolist(),
            concat_dim=dims_sample,
            join='exact',
            compat='identical',
            combine_attrs='identical',
        )
        data = data.assign_coords(coords)
        data = data.transpose(..., *(dim for dim in dims if dim in data.dims))
        if data.sizes['track'] == 1:
            data = data.isel(track=0, drop=True)

        # Possibly add back name
        # TODO: Figure out which functions remove the name!
        if data.name is None:
            data.name = name
        if invert:
            with xr.set_options(keep_attrs=True):
                data = 1.0 / data
        return data

    @quack._while_quantified
    @quack._keep_cell_attrs
    def _integral_or_average(
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
        dims = dims or ('volume',)
        data = self.data.climo.truncate(**kwargs)
        name = data.name
        cell_method = 'integral' if integral else 'average'
        weights_explicit = []  # quantification necessary for integral()
        weights_implicit = []  # quantification not necessary, slows things down a bit
        if weight is not None:
            weight = weight.climo.truncate(**kwargs)
            weights_implicit.append(weight.climo.dequantify())

        # Translate dims. When none are passed, interpret this as integrating over the
        # entire atmosphere. Support irregular grids by preferring 'volume' or 'area'
        dims_orig = self._parse_dims(dims, include_scalar=True, include_pseudo=True)
        dims_std = tuple(
            coordinate for coordinate, names in self.cf.coordinates.items()
            if any(dim in names for dim in dims_orig)
        )
        # Handle special case: Integral along a latitude line without integration
        # over longitude applies unnormalized 'cell depth' latitude weights (km) and
        # normalized longitude and vertical weights. In practice however we almost
        # always integrate over just longitude or both longitude or latitude... and
        # therefore we want unnormalized 'cell width' longitude (cosine latitude)
        # weights. Here we manually add these with a warning message.
        if 'latitude' in dims_std and 'longitude' not in dims_std:
            warnings._warn_climopy(
                f'Only latitude {cell_method} was specified, but almost always '
                'want to integrate over just longitudes or both longitudes and '
                'latitudes. Adding longitude as an integration dimension.'
            )
            dims_orig = ('longitude', *dims_orig)

        # Translate 'area' to longitude latitude and 'volume' to
        # all dimensions if these cell measures are not present.
        # NOTE: This permits 2D and 3D cell measures for non-standard grids and using
        # 'area' and 'volume' as shorthands for 1D cell measures for standard grids
        dims_orig = list(dims_orig)
        if 'volume' in dims_orig and 'volume' not in self.cf.cell_measures:
            dims_orig.remove('volume')
            dims_orig.extend(('area', 'vertical'))
        if 'area' in dims_orig and 'area' not in self.cf.cell_measures:
            dims_orig.remove('area')
            dims_orig.extend(('longitude', 'latitude'))

        # Get quantified cell measure weights for dimensions we are integrating over,
        # and translate 'area' and 'volume' to their component coordinates
        # NOTE: Below error messages are critical. Pretty much the whole reason
        # we wrote those two functions is to facillitate averages.
        dims = []
        measures = set()
        for dim in dims_orig:
            is_coord = dim in COORD_CELL_MEASURE
            is_measure = dim in CELL_MEASURE_COORDS
            if not is_coord and not is_measure:
                try:
                    coordinate = self.cf._encode_name(dim, 'coordinates')
                except KeyError:
                    raise ValueError(f'Missing {cell_method} dimension {dim!r}.')
                names = (dim,)
                measure = COORD_CELL_MEASURE[coordinate]
                coordinates = (coordinate,)
            else:
                names = ()
                measure = dim if is_measure else COORD_CELL_MEASURE[dim]
                coordinates = (dim,) if is_coord else CELL_MEASURE_COORDS[dim]
                for coord in coordinates:
                    try:
                        names += (self.cf._decode_name(coord, 'coordinates'),)
                    except KeyError:
                        raise ValueError(
                            f'Missing {cell_method} coordinate {{!r}}. If data is '
                            'already reduced you may need to call add_scalar_coords.'
                        )
            try:  # is cell measure missing from dictionary?
                key = self.cf._decode_name(measure, 'cell_measures', search_registry=False, return_if_missing=True)  # noqa: E501
            except KeyError:
                raise ValueError(
                    f'Missing cell measure {measure!r} for {cell_method} dimension '
                    f'{dim!r}. You may need to call add_cell_measures.'
                )
            try:  # is cell measure missing from coords? (common for external source)
                weight = self.cf._src[key]
            except KeyError:
                raise ValueError(
                    f'Missing cell measure {measure!r} variable {key!r} for '
                    f'{cell_method} dimension {dim!r}.'
                )
            dims.extend(names)
            measures.add(measure)
            weights_explicit.append(weight.climo.quantify())

        # Add unquantified cell measure weights for measures whose dimensions match any
        # of the dimensions we are already integrating over (e.g. 'width' is added
        # for an areal integral to account for differing cell thickness)
        for measure, (varname,) in self.cf.cell_measures.items():
            if measure in measures:  # explicit weight
                continue
            if varname not in data.coords:  # already reduced or accidentally missing
                continue
            weight = data.coords[varname]
            if set(dims) & set(weight.dims):
                weights_implicit.append(weight.climo.dequantify())

        # Use math.prod to reduce items in list
        # NOTE: numpy.prod just returns 0 for some reason. math.prod seems to work
        # with arbitrary objects, similar to builtin sum()
        one = xr.DataArray(1)  # ensure returned 'weight' is DataArray
        weights = (*weights_explicit, *weights_implicit)
        if integral:
            normalize_denom = True
            weight_num = math.prod(weights, start=one)
            weight_denom = math.prod(weights_implicit, start=one)
        else:
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
        data.climo.update_cell_methods({tuple(dims): cell_method})
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

    @_CFAccessor._clear_cache
    @_manage_coord_reductions  # need access to cell_measures, so put before keep_attrs
    @docstring.inject_snippets(operator='integral', action='integration',)
    def integral(self, dim=None, **kwargs):
        """
        %(template_avgint)s

        Notes
        -----
        %(notes_weighted)s
        """
        kwargs.update(integral=True, cumulative=False)
        return self._integral_or_average(dim, **kwargs)

    @_CFAccessor._clear_cache
    @_manage_coord_reductions  # need access to cell_measures, so put before keep_attrs
    @docstring.inject_snippets(operator='average', action='averaging')
    def average(self, dim=None, **kwargs):
        """
        %(template_avgint)s

        Notes
        -----
        %(notes_avgmean)s

        %(notes_weighted)s
        """
        kwargs.update(integral=False, cumulative=False)
        return self._integral_or_average(dim, **kwargs)

    def anomaly(self, *args, **kwargs):
        """
        Anomaly with respect to mass-weighted average.

        Parameters
        ----------
        *args, **kwargs
            Passed to `~ClimoDataArrayAccessor.average`.
        """
        # TODO: Indicate anomalous data with cell method
        with xr.set_options(keep_attrs=True):
            return self.data - self.average(*args, **kwargs)

    @_CFAccessor._clear_cache
    @docstring.inject_snippets(operator='integral', action='integration')
    def cumintegral(self, dim, **kwargs):
        """
        %(template_cumavgint)s

        Notes
        -----
        %(notes_weighted)s
        """
        kwargs.update(integral=True, cumulative=True)
        return self._integral_or_average(dim, **kwargs)

    @_CFAccessor._clear_cache
    @docstring.inject_snippets(operator='average', action='averaging')
    def cumaverage(self, dim, **kwargs):
        """
        %(template_cumavgint)s

        Notes
        -----
        %(notes_avgmean)s

        %(notes_weighted)s
        """
        kwargs.update(integral=False, cumulative=True)
        return self._integral_or_average(dim, **kwargs)

    def cumanomaly(self, *args, **kwargs):
        """
        Anomaly of cumulative to full mass-weighted average.

        Parameters
        ----------
        *args, **kwargs
            Passed to `ClimoDataArrayAccessor.cumaverage`.
        """
        # TODO: Indicate anomalous data with a cell method
        with xr.set_options(keep_attrs=True):
            return self.average(*args, **kwargs) - self.cumaverage(*args, **kwargs)

    @_CFAccessor._clear_cache
    @quack._keep_cell_attrs
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
            Passed to `~.spectral.runmean`.
        """
        data = self.data
        indexers, _ = self._parse_indexers(indexers, allow_kwargs=False, **kwargs)
        for dim, window in indexers.items():
            if isinstance(window, ureg.Quantity):
                coords = data.climo.coords[dim]
                window = np.round(window / (coords[1] - coords[0]))
                window = int(window.climo.magnitude)
                if window <= 0:
                    raise ValueError('Invalid window length.')
            data = spectral.runmean(data, window, dim=dim)
        return data

    @_CFAccessor._clear_cache
    @quack._while_quantified
    @quack._keep_cell_attrs
    def derivative(self, indexers=None, centered=True, **kwargs):
        """
        Take the nth order centered finite difference for the specified dimensions.

        Parameters
        ----------
        indexers : dict, optional
            Dictionary mapping of dimension names to derivative order. For example,
            to get the second time derivative, use ``time=0``.
        centered : bool, optional
            If False, use more accurate (but less convenient) half-level
            differentiation rather than centered differentiation.
        **indexers_kwargs
            The keyword arguments form of `indexers`.
            One of `indexers` or `indexers_kwargs` must be provided.
        **kwargs
            Passed to `~.diff.deriv_uneven`. The `order` keyword arg is ignored.
        """
        data = self.data
        coords = data.coords
        indexers, kwargs = self._parse_indexers(
            indexers, search_transformations=True, **kwargs
        )
        kwargs.pop('order', None)
        for dim, order in indexers.items():
            coord = data.climo.coords[dim]
            if centered:
                kwargs.setdefault('keepedges', True)
                data = diff.deriv_uneven(coord, data, order=order, **kwargs)
            else:
                _, data = diff.deriv_half(coord, data, order=order, **kwargs)
            data.climo.update_cell_methods({dim: 'derivative'})
        data.coords.update(
            {key: da for key, da in coords.items() if centered or key not in da.dims}
        )
        return data

    @docstring.inject_snippets(operator='convergence')
    def convergence(self, *args, **kwargs):
        """
        %(template_divcon)s
        """
        result = self.divergence(*args, **kwargs)
        with xr.set_options(keep_attrs=True):
            return -1 * result

    @_CFAccessor._clear_cache
    @quack._while_quantified
    @quack._keep_cell_attrs
    @docstring.inject_snippets(operator='divergence')
    def divergence(self, cos_power=1, centered=True, **kwargs):
        """
        %(template_divcon)s
        """
        # Divergence far away from poles
        # Allow for eddy zonal wind flux correction for its contribution to zonal
        # wind budget under angular momentum conservation that requires cos_power == 2.
        y = self.coords['meridional_coordinate']
        cos = self.coords['cosine_latitude']
        data = self.data
        coords = data.coords
        kwargs['order'] = 1
        if centered:
            kwargs.setdefault('keepedges', True)
            cos **= cos_power
            res = diff.deriv_uneven(y, data * cos, **kwargs) / cos
        else:
            cos2 = 0.5 * (cos.data[1:] + cos.data[:-1])
            y, res = diff.deriv_half(y, data * cos ** cos_power, **kwargs)
            res /= cos2 ** cos_power

        # If numerator vanishes, divergence at poles is precisely 2 * dflux / dy.
        # See Hantel 1974, Journal of Applied Meteorology, or just work it out
        # for yourself (simple l'Hopital's rule application).
        lat = self.coords['latitude']
        for lat, isel in ((lat[0], slice(None, 2)), (lat[-1], slice(-2, None))):
            if abs(lat) == 90 * ureg.deg:
                res.climo.loc[{'lat': lat}] = (
                    2 * data.isel(lat=isel).diff(dim='lat').isel(lat=0).data
                    / (y.data[isel][1] - y.data[isel][0])
                )
        res.climo.update_cell_methods({'area': 'divergence'})
        res.coords.update(
            {key: da for key, da in coords.items() if centered or key not in da.dims}
        )
        return res

    @_CFAccessor._clear_cache
    @quack._while_quantified
    @quack._keep_cell_attrs
    @docstring.inject_snippets(operator='correlation', func='corr')
    def autocorr(self, dim, **kwargs):
        """
        %(template_auto)s
        """
        dim = self.cf._decode_name(dim, 'axes', 'coordinates')
        data = self.data
        if not kwargs.keys() & {'lag', 'ilag', 'maxlag', 'imaxlag'}:
            kwargs['ilag'] = 1
        _, data = var.autocorr(data.climo.coords[dim], data, dim=dim, **kwargs)
        data.climo.update_cell_methods({dim: 'correlation'})
        return data

    @_CFAccessor._clear_cache
    @quack._while_quantified
    @quack._keep_cell_attrs
    @docstring.inject_snippets(operator='covariance', func='covar')
    def autocovar(self, dim, **kwargs):
        """
        %(template_auto)s
        """
        dim = self.cf._decode_name(dim, 'axes', 'coordinates')
        data = self.data
        if not kwargs.keys() & {'lag', 'ilag', 'maxlag', 'imaxlag'}:
            kwargs['ilag'] = 1
        _, data = var.autocovar(data.climo.coords[dim], data, dim=dim, **kwargs)
        data.climo.update_cell_methods({dim: 'covariance'})
        return data

    @_CFAccessor._clear_cache
    @quack._keep_cell_attrs
    def centroid(self, dim, source=None, **kwargs):
        """
        Return the value-weighted average wavenumber.

        Parameters
        ----------
        dim : str
            The dimension(s) along which centroid is computed. Might be
            a pseudo dimension e.g. ``'area'`` or ``'volume``'.
        source : `xarray.Dataset`
            The source dataset. Required for retrieving averaging weights, e.g.
            coordinate bounds ``lev_bnds``.
        **kwargs
            Passed to `~ClimoAccessor.truncate`. Used to limit the bounds of the
            calculation.
        """
        # Multi-dimensional reduction: power-weighted centroid (wavenumber)
        # Mask region for taking power-weighted average
        # NOTE: Previously this included a default truncation between
        # latitudes 25 and 75. Now this should be done manually.
        # NOTE: Wavenumber dimension scaling bug is fixed in newer files
        # Need to rerun spectral decompositions on all experiments
        power = self.data.truncate(source=source, **kwargs)
        source = source or power
        wavenum = power.climo.coords['wavenumber']
        if np.all(wavenum < 1):
            wavenum /= (wavenum[1] - wavenum[0])
        mask = wavenum.climo.magnitude >= 1.5  # at least 2 cycles per circle
        circum = 2 * np.pi * const.a * power.climo.coords['cosine_latitude']
        wavenum = wavenum / circum  # cycles per latitude circle to cycles per meter

        # Get centroid as the *power-weighted average*. This prevents
        # recording big discrete jumps in wavelength.
        # NOTE: This should correctly handle 'volume' dimension as a simple
        # latitude-height average. The 'longitude' addition does band scaling.
        # TODO: Support integral or average weight dimensions that exceed
        # dimensions of data being integrated.
        # TODO: Check Frierson et al 2006, make sure this is what people use
        # data.data[data.data >= 10e3 * ureg.km] = np.nan * ureg.km
        power = power.isel(wavenumber=mask)
        source = source.isel(wavenumber=mask)  # e.g. for vertical weights
        wavenum = wavenum.isel(wavenumber=mask)
        wavenum = wavenum.climo._integral_or_average(dim, source=source, weight=power)
        wavelen = 0.25 / wavenum  # radius of given wave crest
        wavelen.climo.update_cell_methods({dim: 'centroid'})

        return wavelen

    @quack._while_dequantified
    @quack._keep_cell_attrs
    def _find_extrema(
        self, dim, abs=False, arg=False, which='max', dim_track=None, **kwargs
    ):
        """
        Find local or global extrema or their locations.
        """
        # Helper function for converting local extrema to global extrema. Appends
        # source array (i.e. data or coordinates) to extrema data (i.e. values or
        # locations). Also adds an all-NaN dummy slice.
        def _concatenate_edges(data, source):
            datas = [data]
            for idx in (0, -1):
                isel = source.isel({dim: idx}, drop=True)
                datas.append(isel.expand_dims('track'))
            datas.append(xr.full_like(datas[-1], np.nan))  # dummy slice final position
            data = xr.concat(
                datas,
                dim='track',
                coords='minimal',
                compat='override',
                combine_attrs='override'
            )
            data.coords['track'] = np.arange(data.sizes['track'])
            return data

        # Parse and truncate
        dim = self._parse_dims(dim, single=True)
        trunc, kwargs = self._parse_truncate_args(**kwargs)
        data = self.truncate(**trunc)
        if dim == 'lat':  # TODO: remove kludge! error is with uweight lat=absmax
            data = data.transpose(..., dim)

        # Manage find keyword args
        if dim_track is not None:
            kwargs['axis_track'] = data.dims.index(dim_track)
        if abs:
            kwargs.setdefault('track', False)  # will not matter in the end
        elif kwargs.get('axis_track', None) is not None:
            kwargs.setdefault('track', True)  # implied that user wants tracking
        elif kwargs.get('track', None):
            warnings._warn_climopy(
                'Keyword args axis_track or dim_track required when track=True. '
                'Setting track=False.'
            )
            kwargs['track'] = False
        if which in ('negpos', 'posneg', 'both'):  # super users
            kwargs['which'] = which
        elif which in ('min', 'max'):
            kwargs['diff'] = 1
            if which == 'min':
                kwargs['which'] = 'negpos'
            if which == 'max':
                kwargs['which'] = 'posneg'
        elif which:
            raise ValueError(f'Unknown argument {which=}.')

        # Get precise local values using linear interpolation
        # NOTE: The find function applies pint units
        coord = data.climo.coords[dim]  # return modifiable copy of coords
        coord = coord.climo.dequantify()  # units not necessary
        locs, values = utils.find(coord, data, **kwargs)
        locs = locs.climo.dequantify()
        values = values.climo.dequantify()

        # Get global extrema. If none were found (e.g. there are only extrema
        # on edges) revert to native min max functions.
        # WARNING: Need xarray >= 0.16.0 'idxmin' and 'idxmax' to avoid all-NaN
        # slice errors. See: https://github.com/pydata/xarray/issues/4481
        if abs and locs.sizes['track'] == 0:
            locs = getattr(data, 'idx' + which)(dim, fill_value=np.nan).drop_vars(dim)
            values = getattr(data, which)(dim).drop_vars(dim)

        # Otherwise select from the identified 'sandwiched' extrema and possible
        # extrema on the array edges. We merge find values with array edges
        # NOTE: Here 'idxmin' and 'idxmax' followed by 'sel' probably more expensive
        # then 'argmin' and 'argmax' but needed to set fill_value to dummy positoin
        elif abs:
            locs = _concatenate_edges(locs, coord)
            values = _concatenate_edges(values, data)
            sel = getattr(values, 'idx' + which)('track', fill_value=locs['track'][-1])
            locs = locs.sel(track=sel, drop=True)
            values = values.sel(track=sel, drop=True)

        # Use either actual locations or interpolated values. Restore attributes
        # NOTE: Add locs to coordinates for 'min', 'max', etc. and add values to
        # coordinates for 'argmax', 'argmin', etc. Then e.g. for 'argmax' can leverage
        # coordinate information to have '.cfvariable' interpret 'lat' with e.g.
        # 'zonal_wind' coordinate data as e.g. the latitude of maximum zonal wind.
        # values = data.climo.interp({dim: locs.squeeze()}, method='cubic')
        locs.attrs.update(coord.attrs)
        values.attrs.update(data.attrs)
        if arg:
            locs.coords[data.name] = values
            locs.attrs['parent_name'] = data.name
            data = locs
        else:
            values.coords[coord.name] = locs
            data = values

        data.climo.update_cell_methods({dim: 'arg' + which if arg else which})
        return data

    @_CFAccessor._clear_cache
    # @_manage_coord_reductions
    @docstring.inject_snippets(extrema='mimima', prefix='')
    def min(self, dim=None, **kwargs):
        """
        %(template_minmax)s
        """
        kwargs.update(which='min', abs=False, arg=False)
        return self._find_extrema(dim, **kwargs)

    # @_manage_coord_reductions
    @docstring.inject_snippets(extrema='maxima', prefix='')
    def max(self, dim=None, **kwargs):
        """
        %(template_minmax)s
        """
        kwargs.update(which='max', abs=False, arg=False)
        return self._find_extrema(dim, **kwargs)

    # @_manage_coord_reductions
    @docstring.inject_snippets(extrema='minima', prefix='')
    def absmin(self, dim=None, **kwargs):
        """
        %(template_absminmax)s
        """
        kwargs.update(which='min', abs=True, arg=False)
        return self._find_extrema(dim, **kwargs)

    # @_manage_coord_reductions
    @docstring.inject_snippets(extrema='maxima', prefix='')
    def absmax(self, dim=None, **kwargs):
        """
        %(template_absminmax)s
        """
        kwargs.update(which='max', abs=True, arg=False)
        return self._find_extrema(dim, **kwargs)

    # @_manage_coord_reductions
    @docstring.inject_snippets(extrema='minima', prefix='coordinates of')
    def argmin(self, dim=None, **kwargs):
        """
        %(template_minmax)s
        """
        kwargs.update(which='min', abs=False, arg=True)
        return self._find_extrema(dim, **kwargs)

    # @_manage_coord_reductions
    @docstring.inject_snippets(extrema='maxima', prefix='coordinates of')
    def argmax(self, dim=None, **kwargs):
        """
        %(template_minmax)s
        """
        kwargs.update(which='max', abs=False, arg=True)
        return self._find_extrema(dim, **kwargs)

    # @_manage_coord_reductions
    @docstring.inject_snippets(extrema='minima', prefix='coordinates of')
    def absargmin(self, dim=None, **kwargs):
        """
        %(template_absminmax)s
        """
        kwargs.update(which='min', abs=True, arg=True)
        return self._find_extrema(dim, **kwargs)

    # @_manage_coord_reductions
    @docstring.inject_snippets(extrema='maxima', prefix='coordinates of')
    def absargmax(self, dim=None, **kwargs):
        """
        %(template_absminmax)s
        """
        kwargs.update(which='max', abs=True, arg=True)
        return self._find_extrema(dim, **kwargs)

    # @_manage_coord_reductions
    @docstring.inject_snippets()
    def argloc(self, dim=None, value=None, **kwargs):
        """
        %(template_argloc)s
        """
        kwargs.update(abs=False, arg=True)
        kwargs.setdefault('which', 'both')
        data = self.data
        if value is not None:
            data = data - value  # find points meeting this value (default 0)
        return data.climo._find_extrema(dim, **kwargs)

    @_CFAccessor._clear_cache
    @quack._keep_cell_attrs
    def hist(self, dim, bins=None):
        """
        Return the histogram along the given dimension.

        Parameters
        ----------
        dim : str
            The dimension name.
        bins : int or sequence of float, optional
            The bin boundaries or the integer number of bins from the minimum datum to
            the maximum datum. Default is ``11``.

        Todo
        ----
        Support multiple dimensions.
        """
        data = self.data
        if bins is None:
            bins = 11
        if isinstance(bins, numbers.Integral):
            bins = np.linspace(np.nanmin(data.data), np.nanmax(data.data), bins)
        else:
            bins = bins.copy()
        dim = self.cf._decode_name(dim, 'axes', 'coordinates')
        data = var.hist(bins, data, dim=dim)
        if 'track' in data.dims:
            data = data.sum(dim='track')
        data.climo.update_cell_methods({dim: 'hist'})
        return data

    def mask(self, mask, dataset=None):
        """
        Return a copy of the data with a mask applied according to some preset pattern.

        Warning
        -------
        This method is incomplete.
        """
        # TODO: Expand this function
        # NOTE: DataArray math operations ignore NaNs by default
        data = self.data.copy(deep=True)
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
            data[~mask] = np.nan * getattr(data.data, 'units', 1)
        return data

    @_CFAccessor._clear_cache
    @quack._while_dequantified
    @quack._keep_cell_attrs
    def normalize(self):
        """
        Return a copy of the data normalized with respect to time.
        """
        time = self.cf._decode_name('time', 'coordinates')
        data = self.data
        data = data / data.mean(time)
        data.attrs['units'] = 'dimensionless'
        data.climo.update_cell_methods({time: 'normalized'})
        return data

    @_CFAccessor._clear_cache
    @quack._while_quantified
    @quack._keep_cell_attrs
    def slope(self, dim):
        """
        Return the best-fit slope with respect to some dimension.

        Parameters
        ----------
        dim : str, optional
            The dimension.
        **kwargs
            Passed to `~.var.slope`.
        """
        dim = self.cf._decode_name(dim, 'axes', 'coordinates')
        data = self.data
        data, _, _ = var.slope(data.climo.coords[dim], data)
        data.climo.update_cell_methods({dim: 'slope'})
        return data

    @_CFAccessor._clear_cache
    @quack._while_quantified
    @quack._keep_cell_attrs
    def timescale(self, dim, maxlag=None, imaxlag=None, **kwargs):
        """
        Return a best-fit estimate of the autocorrelation timescale.

        Parameters
        ----------
        dim : str, optional
            The dimension.
        **kwargs
            Passed to `~.var.timescale`.
        """
        dim = self.cf._decode_name(dim, 'axes', 'coordinates')
        data = self.data
        time = data.climo.coords[dim]
        if maxlag is None and imaxlag is None:
            maxlag = 50.0  # default value is 50 days
        if dim != 'lag':
            time, data = var.autocorr(time, data, maxlag=maxlag, imaxlag=imaxlag)
        data, _, _ = var.timescale(time, data, maxlag=maxlag, imaxlag=imaxlag, **kwargs)  # noqa: E501
        data.climo.update_cell_methods({dim: 'timescale'})
        return data

    @_CFAccessor._clear_cache
    @quack._while_quantified
    def to_variable(self, dest, standardize=False, **kwargs):
        """
        Transform this variable to another variable using transformations
        registered with `register_transformation`. Transformations work recursively,
        i.e. definitions for A --> B and B --> C permit transforming A --> C.

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
            raise RuntimeError('DataArray name is empty. Cannot get transformation.')
        func = self._find_this_transformation(data.name, dest)
        if func is None:
            raise ValueError(f'Transformation {data.name!r} --> {dest!r} not found.')
        with xr.set_options(keep_attrs=False):  # ensure invalid attributes are lost
            param = func(data, **kwargs)
        param.name = dest
        if standardize:
            param = param.climo.to_standard_units()
        return param

    @quack._while_quantified
    def to_units(self, units, context='climo'):
        """
        Return a copy converted to the desired units.

        Parameters
        ----------
        units : str or `pint.Unit`
            The destination units. Strings are parsed with `~.unit.decode_units`.
        context : str or `pint.Context`, optional
            The `pint context <https://pint.readthedocs.io/en/0.10.1/contexts.html>`_.
            Default is the ClimoPy context ``'climo'`` (see `~.unit.ureg` for details).
        """
        if not self._is_quantity:
            raise ValueError('Data should be quantified.')
        units = decode_units(units)
        data = self.data.copy(deep=False)
        args = (context,) if context else ()
        try:
            data.data = data.data.to(units, *args)  # NOTE: not ito()
        except Exception:
            raise RuntimeError(
                f'Failed to convert {data.name!r} from current units '
                f'{self.units!r} to units {units!r}.'
            )
        return data

    @quack._while_quantified
    def to_base_units(self, coords=False):
        """
        Return a copy with the underlying data converted to base units.
        """
        # NOTE: assign_coords has issues with multiple DataArray values. See:
        # https://github.com/pydata/xarray/issues/3483
        data = self.data.copy(deep=False)
        data.data = data.data.to_base_units()
        if coords:
            data = data.assign_coords({
                dim: data.climo.coords[dim].climo.to_base_units().climo.dequantify()
                .variable for dim in data.coords
            })
        return data

    @quack._while_quantified
    def to_compact_units(self, coords=False):
        """
        Return a copy with the underlying data converted to "compact" units.
        """
        # NOTE: assign_coords has issues with multiple DataArray values. See:
        # https://github.com/pydata/xarray/issues/3483
        data = self.data.copy(deep=False)
        data.data = data.data.to_compact_units()
        if coords:
            data = data.assign_coords({
                dim: data.climo.coords[dim].climo.to_compact_units().climo.dequantify()
                .variable for dim in data.coords
            })
        return data

    def to_standard_units(self, coords=False):
        """
        Return a copy with the underyling data converted to the
        `~ClimoDataArrayAccessor.cfvariable` `standard_units` value. This will only
        work if the variable name matches a valid `~.cfvariable.CFVariable` identifier.
        """
        # NOTE: assign_coords has issues with multiple DataArray values. See:
        # https://github.com/pydata/xarray/issues/3483
        units = self.cfvariable.standard_units
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

    def quantify(self, units=None):
        """
        Return a copy of the `xarray.DataArray` with underlying data converted to
        `pint.Quantity` using the ``'units'`` attribute. If the data is already
        quantified, nothing is done. If the ``'units'`` attribute is missing, a warning
        is raised. Units are parsed with `~.unit.decode_units`.
        """
        # WARNING: In-place conversion resulted in endless bugs related to
        # ipython %autoreload, was departure from metpy convention, was possibly
        # confusing for users, and not even sure if faster. So abandoned this.
        data = self.data.copy(deep=False)
        data.climo._quantify(units=units)
        return data

    def dequantify(self):
        """
        Return a copy of the `xarray.DataArray` with underlying data stripped of
        its units and units written to the ``'units'`` attribute. If the data is already
        dequantified, nothing is done. Units are written with `~.unit.encode_units`.
        """
        # WARNING: Try to preserve *order* of units for fussy formatting later on.
        # Avoid default alphabetical sorting by pint.__format__.
        data = self.data.copy(deep=False)
        data.climo._dequantify()
        return data

    def _quantify(self, units=None):
        """
        In-place version of `~ClimoDataArrayAccessor.quantify`.
        """
        # NOTE: This won't affect shallow DataArray or Dataset copy parents
        data = self.data
        if isinstance(data.data, pint.Quantity) or not quack._is_numeric(data.data):
            if units is None:
                units = data.attrs.get('units', None)
            if units is not None:
                warnings._warn_climopy(f'Ignoring {units=}. Data is non-numeric or already quantified.')  # noqa: E501
        else:
            if units is not None:
                data.attrs['units'] = encode_units(units)
            data.data = data.data * self.units  # may raise error
        data.attrs.pop('units', None)

    def _dequantify(self):
        """
        In-place version of `~ClimoDataArrayAccessor.dequantify`.
        """
        # NOTE: This won't affect shallow DataArray or Dataset copy parents
        data = self.data
        if not isinstance(data.data, pint.Quantity):
            return
        units = data.data.units
        data.data = data.data.magnitude
        data.attrs['units'] = encode_units(units)

    @property
    @_CFAccessor._clear_cache
    def cfvariable(self):
        """
        Return a `~.cfvariable.CFVariable` based on the `~xarray.DataArray` name, the
        scalar coordinates, and the coordinate reductions referenced in `cell_methods`.
        As a shorthand, you can access ``data_array.climo.cfvariable`` properties
        directly using ``data_array.climo.property``.
        """
        return self._cf_variable()

    @property
    def magnitude(self):
        """
        The magnitude of the data values of this `~xarray.DataArray`
        (i.e., without units).
        """
        if self._is_quantity:
            return self.data.data.magnitude
        else:
            return self.data.data

    @property
    def quantity(self):
        """
        The data values of this `~xarray.DataArray` as a `pint.Quantity`.
        """
        if self._is_quantity:
            return self.data.data
        else:
            return ureg.Quantity(self.data.data, self.units)

    @property
    def units(self):
        """
        The units of this `~xarray.DataArray` as a `pint.Unit`, taken from the
        underlying `pint.Quantity` or the ``'units'`` attribute. Unit strings are
        parsed with `~.unit.decode_units`.
        """
        if self._is_quantity:
            return self.data.data.units
        elif 'units' in self.data.attrs:
            return decode_units(self.data.attrs['units'])
        else:
            raise RuntimeError(
                'Units not present in attributes or as pint.Quantity '
                f'for DataArray with name {self.data.name!r}.'
            )

    @property
    def units_label(self):
        """
        The units of this `~xarray.DataArray` formatted LaTeX-style. Suitable for
        adding text to matplotlib figures. This works even when a
        `~ClimoDataArrayAccessor.cfvariable` is not available (see also the
        `units_label` `~.cfvariable.CFVariable` property).
        """
        # NOTE: When pint parses units string, the associated _units dictionary
        # retains original insertion order. No need to hang onto old string.
        units = self.units
        try:
            units_standard = self.cfvariable.units_standard
        except AttributeError:
            pass
        else:
            if units_standard is not None and units == decode_units(units_standard):
                return format_units(units_standard)
        return format_units(units)

    @property
    def _has_units(self):
        """
        Boolean indiating whether 'units' attribute exists or data is quantified.
        """
        return self._is_quantity or 'units' in self.data.attrs

    @property
    def _is_quantity(self):
        """
        Boolean indiating whether data is quantified.
        """
        return isinstance(self.data.data, pint.Quantity)

    @property
    def _is_coordinate_bounds(self):
        """
        Boolean indiating whether data is a coordinate bounds.
        """
        key = self.data.name
        coords = self.data.coords
        sentinel = object()
        return any(key == da.attrs.get('bounds', sentinel) for da in coords.values())


@xr.register_dataset_accessor('climo')
class ClimoDatasetAccessor(ClimoAccessor):
    """
    Accessor for `xarray.Dataset`\\ s. Includes methods for working with `pint`
    quantities and `~.cfvariable.CFVariable` variables and an interface for deriving one
    physical variable from other variables in the dataset. Registered under the name
    ``climo`` (i.e, usage is ``data_array.climo``). The string representation of this
    accessor displays `~ClimoDataArrayAccessor.cfvariable` information for every
    variable whose name is found in the `~ClimoAccessor.variable_registry`.
    """
    _cls_cf = _CFDatasetAccessor
    _cls_coords = _DatasetCoordsQuantified
    _cls_groupby = _DatasetGroupByQuantified
    _cls_loc = _DatasetLocIndexerQuantified

    @_CFAccessor._clear_cache
    def __repr__(self):
        data = self.data
        rows = ['<climopy.ClimoDatasetAccessor>']
        width = max(
            (
                len(da.name) for grp in (data, data.coords) for da in grp.values()
                if isinstance(da.name, str)
            ), default=10
        )
        for start, src in zip(('Coordinates:', 'Data variables:'), (data.coords, data)):
            if not src:
                continue
            rows.append(start)
            rows.extend(
                da.climo._cf_repr(maxlength=88, padlength=4, varwidth=width + 2)
                for da in src.values()
            )
        return '\n'.join(rows)

    @_CFAccessor._clear_cache
    def __contains__(self, key):
        """
        Is a dataset variable or derived coordinate.
        """
        return self._parse_key(key) is not None

    def __dir__(self):
        """
        Support name lookup and completion. Derivations and aliases are excluded.
        """
        return sorted({*dir(type(self)), *self.coords, *self.vars})

    def __getattr__(self, attr):
        """
        Try to return a variable with `~ClimoDatasetAccessor.__getitem__`.
        """
        if attr in super().__dir__():  # can happen if @property triggers error
            return super().__getattribute__(attr)  # trigger builtin AttributeError
        if attr in self:
            return self.__getitem__(attr)
        raise AttributeError(
            f'Attribute {attr!r} does not exist and is not a variable, '
            'transformed variable, or derived variable.'
        )

    @_CFAccessor._clear_cache
    def __getitem__(self, key):
        """
        Return a quantified coordinate or variable, including transformations and
        derivations registered with `register_transformation` or `register_derivation`,
        or return a selection along dimensions with translated dictionary indexing.
        Translates CF names and `~.cfvariable.CFVariableRegistry` identifiers. Attaches
        cell measures to coordinates using `~ClimoAccessor.add_cell_measures`.
        """
        if isinstance(key, dict):  # see also .loc.__getitem__
            key, _ = self._parse_indexers(key)
            return self.data[key]
        else:
            return self._get_item(key)  # with weights attached

    def _get_item(self, key, add_cell_measures=True, **kwargs):
        """
        Return a quantified DataArray with weights optionally added. This is
        separated from `_parse_key` to facilitate fast `__contains__`.
        """
        # Retrieve and compute quantity
        tup = self._parse_key(key, **kwargs)
        if not tup:
            raise KeyError(f'Invalid variable name {key!r}.')
        type_, da = tup
        if callable(da):
            da = da()  # ta-da!

        # Add units and cell measures
        data = da.climo.quantify()  # should already be quantified, but just in case
        if data.name is None:
            data.name = 'unknown'  # just in case
        if type_ != 'coord' and add_cell_measures:
            data = data.climo.add_cell_measures(dataset=self.data)

        # Transpose, potentially after derivation or transformation moved dims around
        # See: https://github.com/pydata/xarray/issues/2811#issuecomment-473319350
        # NOTE: Also re-order spectral dimensions for 3 plot types: YZ, CY, YK, and CK
        # (with row-major ordering), or simply preserve original dimension order based
        # on dimension order that appears in dataset variables.
        if 'k' in data.dims:
            dims = ('lev', 'k', 'lat', 'c')
        else:
            dims = _first_unique(dim for da in self.data.values() for dim in da.dims if not da.climo._is_coordinate_bounds)  # noqa: E501
        data = data.transpose(..., *(dim for dim in dims if dim in data.dims))

        return data

    def _parse_key(
        self,
        key,
        search_vars=True,
        search_coords=True,
        search_derivations=True,
        search_transformations=True,
        search_registry=True,
        **kwargs
    ):
        """
        Return a DataArray or function that generates the data and a string indicating
        the object type. Extra args are passed to `.vars.get` and `.coords.get`.
        """
        # Return a variable, removing special suffixes from variable names
        # NOTE: This lets us implement a quick __contains__ that works on derived vars
        # TODO: Add robust method for automatically removing dimension reduction
        # suffixes from var names and adding them as cell methods (currently do this in
        # Experiment.load()), or excluding suffixes when looking up in CFRegistry.
        if not isinstance(key, str):
            raise TypeError(f'Invalid key {key!r}. Must be string.')
        if search_vars:
            da = self.vars.get(key, search_registry=search_registry, **kwargs)
            if da is not None:  # noqa: E501
                da.name = REGEX_IGNORE.sub(r'\1', da.name)
                return 'var', da

        # Return a coord, transformation, or derivation
        # NOTE: Coordinate search rules out coordinate transformations
        # TODO: Make derivations look like transformations; define functions that
        # accept data arrays with particular names.
        if search_coords:
            coord = self.coords.get(key, search_registry=search_registry, **kwargs)
            if coord is not None:
                return 'coord', coord
        if search_derivations:
            func = self._find_derivation(key)
            if func is not None:
                return 'derivation', functools.partial(func, self)
        if search_transformations:
            tup = self._find_any_transformation(self.data.values(), key)
            if tup is not None:
                return 'transformation', functools.partial(*tup)

        # Recursively check if any aliases are valid
        if search_registry:
            var = self._variable_registry.get(key)
            identifiers = var.identifiers if var else ()
            for name in set(identifiers):
                if tup := self._parse_key(
                    name,
                    search_vars=search_vars,
                    search_coords=search_coords,
                    search_derivations=search_derivations,
                    search_transformations=search_transformations,
                    search_registry=False
                ):
                    return tup

    def add(self, *args, **kwargs):
        """
        Call `get` and return a copy of the dataset with the result added to it.
        """
        data = self.data.copy(deep=False)
        kwargs.setdefault('add_cell_measures', False)
        da = self.get(*args, **kwargs)
        if da.name in self.data:
            raise RuntimeError(f'Quantity {da.name!r} already present in dataset.')
        data[da.name] = da
        return data

    @_CFAccessor._clear_cache
    @_expand_variable_args  # standardize args are passed to lookup cache
    # @functools.lru_cache(maxsize=64)  # TODO: fix issue where recursion breaks cache
    def get(
        self, key, *,
        quantify=None,
        standardize=False,
        units=None,
        normalize=False,
        runmean=None,
        add=None,
        subtract=None,
        multiply=None,
        divide=None,
        **kwargs
    ):
        """
        Call `~ClimoDatasetAccessor.__getitem__`, with optional post-processing
        steps and special behavior when variables are prefixed or suffixed with
        certain values.

        Parameters
        ----------
        key : var-spec or 2-tuple
            The variable specification. See `~ClimoDatasetAccessor.__getitem__` for
            details. The following prefix suffix shorthands are also supported:

            * Prepend ``abs_`` to return the absolute value of the result.
            * Append ``_latitude`` or ``_strength`` to return vertically and zonally
              integrated energy and momentum budget maxima or latitudes of maxima.
            * Append ``_1``, ``_2``, ``_anomaly``, or ``_ratio`` to make a selection or
              take an anomaly pair difference using `~ClimoAccessor.sel_pair`.

            All names can be replaced with 2-tuples of the form ('name', kwargs) to
            pass keyword arguments positionally.
        search_coords : bool, optional
            Whether to search for coordinates.
        search_vars : bool, optional
            Whether to search for variables.
        search_cf : bool, optional
            Whether to translate CF names.
        search_registry : bool, optional
            Whether to translate registered names and aliases.
        search_derivations : bool, optional
            Whether to perform registered derivations of coordinates or variables.
        search_transformations : bool, optional
            Whether to perform registered transformations of coordinates or variables.
        add_cell_measures : bool, optional
            Whether to add default cell measures to the coordinates.
        quantify : bool, optional
            Whether to quantify the data with `~ClimoDataArrayAccessor.quantify()`.
        units : unit-spec, optional
            Convert the result to the input units with
            `~ClimoDataArrayAccessor.to_units`.
        standardize : bool, optional
            Convert the result to the standard units with
            `~ClimoDataArrayAccessor.to_standard_units`.
        normalize : bool, optional
            Whether to normalize the resulting data with
            `~ClimoDataArrayAccessor.normalize`.
        runmean : bool, optional
            Apply a length-`runmean` running mean to the time dimension with
            `~ClimoDataArrayAccessor.runmean`.
        add, subtract, multiply, divide : var-spec, optional
            Modify the resulting variable by adding, subtracting, multiplying, or
            dividing by this variable (passed to `~ClimoDatasetAccessor.get`).
        long_name, short_name, standard_name, long_prefix, long_suffix, short_prefix, \
short_suffix : str, optional
            Arguments to be passed to `~.cfvariable.CFVariableRegistry` when
            constructing the `~ClimoDataArrayAccessor.cfvariable`. Added as
            attributes to the `~xarray.DataArray`.
        **kwargs
            Passed to `~ClimoDataArrayAccessor.reduce`. Used to reduce dimensionality.

        Returns
        -------
        data : xarray.DataArray
            The data.
        """
        # Get the variable, translating meta-variable actions and interpreting
        # CFVariable attributes passed as keyword arguments.
        # TODO: Avoid name conflicts with derivations and variables?
        flag_abs, key, flag_reduce, flag_pair = REGEX_MODIFY.match(key).groups()
        kw_reduce = kwargs.copy()
        kw_attrs = {k: kw_reduce.pop(k) for k in CFVARIABLE_ARGS if k in kw_reduce}
        kw_getitem = {k: kw_reduce.pop(k) for k in PARSEKEY_ARGS if k in kw_reduce}
        data = self._get_item(key, **kw_getitem)

        # Reduce dimensionality using keyword args
        # WARNING: For timescale variables take inverse before and after possible
        # average. Should move this kludge away.
        if flag_reduce := flag_reduce and flag_reduce.strip('_'):
            kw_reduce.update(data.climo._budget_reduce_kwargs(flag_reduce))
        if kw_reduce:
            try:
                data = data.climo.reduce(source=self.data, **kw_reduce)
            except Exception:
                raise ValueError(f'Failed to reduce data {key!r} with {kw_reduce}.')

        # Normalize the data
        if normalize:
            data = data.climo.normalize()

        # Take the rolling mean
        if runmean:
            data = data.climo.runmean(time=runmean)

        # Take the absolute value, accounting for attribute-stripping xarray bug
        if flag_abs:
            data = quack._keep_cell_attrs(np.abs)(data)

        # Select pair only *after* doing all the math. This is a convenient way
        # to get difference between reduced values
        if flag_pair:
            data = data.climo.sel_pair(flag_pair.strip('_'))

        # Modify the variable. This was used to compare storm track intensity changes
        # under different temperature gradient adjustments.
        for other, method in zip(
            (multiply, divide, add, subtract),
            ('mul', 'truediv', 'add', 'sub'),
        ):
            if other is not None:
                if isinstance(other, (str, tuple)):
                    other = self.get(other)
                with xr.set_options(keep_attrs=True):
                    data = getattr(data, f'__{method}__')(other)  # should keep name

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

        data.attrs.update(kw_attrs)
        return data

    def quantify(self):
        """
        Return a copy of the `xarray.Dataset` with underlying `xarray.DataArray` data
        converted to `pint.Quantity` using the ``'units'`` attributes. Coordinate bounds
        variables are excluded. Already-quantified data is left alone.
        """
        return self.data.map(lambda d: d if d.climo._is_coordinate_bounds else d.climo.quantify())  # noqa: E501

    def dequantify(self):
        """
        Return a copy of the `xarray.Dataset` with underlying `xarray.DataArray` data
        stripped of its units and the units written to the ``'units'`` attributes.
        Already-dequantified data is left alone.
        """
        return self.data.map(lambda d: d.climo.dequantify())

    @property
    def vars(self):
        """
        Analogue to `ClimoAccessor.coords` for retreiving always-quantified data
        variables based on their actual names, standard name attributes, or
        `~.cfvariable.CFVariableRegistry` identifiers.
        """
        return _VarsQuantified(self.data, self.variable_registry)


@docstring.inject_snippets()
def register_derivation(dest, /, *, assign_name=True):
    """
    Register a function that derives one variable from one or more others, for use
    with `ClimoDatasetAccessor.get`.

    Parameters
    ----------
    %(params_register)s

    Examples
    --------
    >>> import climopy as climo
    >>> from climopy import const
    >>> @climo.register_derivation('pot_temp')
    ... def potential_temp(self):
    ...     return self.temp * (const.p0 / self.pres).climo.to_units('') ** (2 / 7)
    >>> ds = xr.Dataset(
    ...     {
    ...         'temp': ((), 273, {'units': 'K'}),
    ...         'pres': ((), 100, {'units': 'hPa'})
    ...     }
    ... )
    >>> ds.climo['pot_temp']  # or ds.climo.pt
    <xarray.DataArray 'pot_temp' ()>
    <Quantity(527.08048, 'kelvin')>
    """
    if not isinstance(dest, (str, tuple, re.Pattern)):
        raise TypeError(f'Invalid name or regex {dest!r}.')
    if dest in DERIVATIONS:
        warnings._warn_climopy(f'Overriding existing derivation {dest!r}.')

    def _decorator(func):
        @quack._keep_cell_attrs
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            if data is NotImplemented:
                raise NotImplementedError(f'Derivation {func} is not implemnted.')
            if not isinstance(data, xr.DataArray):
                raise TypeError(f'Derivation {func} did not return a DataArray.')
            if assign_name:
                data.name = dest if isinstance(dest, str) else kwargs.get('name', None)
            return data

        DERIVATIONS[dest] = _wrapper
        return _wrapper

    return _decorator


@docstring.inject_snippets()
def register_transformation(src, dest, /, *, assign_name=True):
    """
    Register a function that transforms one variable to another, for use with
    `ClimoDataArrayAccessor.to_variable`. Transformations should depend only on the
    initial variable and (optionally) the coordinates.

    Parameters
    ----------
    src : str
        The source variable name.
    %(params_register)s

    Examples
    --------
    In this example, we define a simple derivation to convert pressure to the
    log-pressure height.

    >>> import climopy as climo
    >>> from climopy import const
    >>> @climo.register_transformation('p', 'z_logp')
    ... def meridional_coordinate(da):
    ...     return (const.H * np.log(const.p0 / da)).climo.to_units('km')
    >>> da = xr.DataArray([1000, 800, 600, 400], name='p', attrs={'units': 'hPa'})
    >>> da.climo.to_variable('z_logp')
    <xarray.DataArray 'z_logp' (dim_0: 4)>
    array([0.        , 1.56200486, 3.57577937, 6.41403512])
    Dimensions without coordinates: dim_0
    Attributes:
        units:    kilometer
    """
    if not isinstance(src, str):
        raise ValueError(f'Invalid source {src!r}. Must be string.')
    if not isinstance(dest, (str, tuple, re.Pattern)):
        raise ValueError(f'Invalid destination {dest!r}. Must be string, tuple, regex.')
    if (src, dest) in TRANSFORMATIONS:
        warnings._warn_climopy(f'Overriding existing {src!r}->{dest!r} transformation.')  # noqa: E501

    def _decorator(func):
        @quack._keep_cell_attrs
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            if data is NotImplemented:
                raise NotImplementedError(f'Transformation {func} is not implemnted.')
            if not isinstance(data, xr.DataArray):
                raise TypeError(f'Transformation {func} did not return a DataArray.')
            if assign_name:
                data.name = dest if isinstance(dest, str) else kwargs.get('name', None)
            return data

        TRANSFORMATIONS[(src, dest)] = _wrapper
        return _wrapper

    return _decorator
