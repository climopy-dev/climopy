#!/usr/bin/env python3
"""
Tools for managing CF-style variable metadata. Includes a class for holding variable
metadata and grouping variables into related hierarchies, and a registry for storing
variables.
"""
import copy
import numbers
import re

import numpy as np
import pint

from .internals import _make_stopwatch  # noqa: F401
from .internals import ic  # noqa: F401
from .internals import warnings
from .unit import decode_units, format_units

__all__ = [
    'vreg',  # pint convention
    'variables',  # metpy convention
    'CFVariable',
    'CFVariableRegistry',
]


class CFVariable(object):
    """
    Lightweight CF-style representation of physical variables. Integrated
    with the xarray accessor via `~.accessor.ClimoDataArrayAccessor.cfvariable`.
    """
    def __str__(self):
        names = ', '.join(self.identifiers)  # name, standard_name, aliases
        return f'CFVariable({names})'

    def __repr__(self):
        standard_units = self.standard_units
        long_name = self.long_name
        short_name = self.short_name
        standard_name = self.standard_name
        aliases = self.aliases
        string = self.name
        if long_name:
            string += f', {long_name=}'
        if standard_units is not None:
            string += f', {standard_units=}'
        if short_name and short_name != long_name:
            string += f', {short_name=}'
        if standard_name:
            string += f', {standard_name=}'
        if aliases:
            string += f', aliases={aliases!r}'
        return f'CFVariable({string})'

    def __new__(cls, *args, **kwargs):
        # Initialize with a registry always declared.
        inst = object.__new__(cls, *args, **kwargs)
        if not hasattr(inst, '_registry'):  # use the 'app' registry
            inst._registry = vreg
        return inst

    def __init__(self, name, *args, **kwargs):
        """
        Parameters
        ----------
        name : str
            The canonical variable name.
        *args, **kwargs
            Passed to `CFVariable.update`. The `long_name`, `standard_units`, and
            `short_name` can be passed positionally (in that order).
        """
        # NOTE: Accessor can only be added by querying the registry to make an
        # ad hoc copy. The 'reference' variables in general should be conceptually
        # isolated and separate from any particular data.
        self._name = name
        self._aliases = []  # manged through variable registry
        self._parents = []
        self._children = []
        self._accessor = None
        self.update(*args, **kwargs)

    def __contains__(self, other):
        """
        Whether input variable is child of this one.
        """
        for var in self:
            if var == other:
                return True
        return False

    def __eq__(self, other):
        """
        Whether variables are equivalent.
        """
        if isinstance(other, str):
            try:
                other = self._registry._get_item(other)
            except KeyError:
                return False
        # NOTE: CFVariableRegistry.define() *very* robustly ensures no overlap, but
        # below accommodates edge cases arising from users making their own variables.
        # Could compare identifiers but then variables with swapped standard names
        # and canonical names would be shown as 'True'.
        self_names = [self.name]
        other_names = [other.name]
        for var, names in zip((self, other), (self_names, other_names)):
            if name := var.standard_name:
                names.append(name)
            names.extend(sorted(var.aliases))
        if self_names == other_names:
            return True
        elif all(s != o for s, o in zip(self_names, other_names)):
            return False
        else:
            warnings._warn_climopy(f'Partial overlap between {self} and {other}.')
            return True

    def __iter__(self):
        """
        Iterate through self and children recursively.
        """
        yield self
        for var in self._children:
            yield from var

    @staticmethod
    def _modify_name(name, prefix, suffix):
        """
        Modify variable name with prefix and suffix, squeeze consective spaces,
        and correct clunky names like "jet stream strength latitude".
        """
        if name is None:
            return None
        prefix = prefix or ''
        suffix = suffix or ''
        name = ' '.join((prefix, name, suffix)).strip()
        name = re.sub(r'\s\+', ' ', name)  # squeeze consecutive spaces
        for modifier in ('strength', 'intensity'):
            for ending in ('latitude', 'anomaly', 'response'):
                name = name.replace(f'{modifier} {ending}', ending)
        return name

    def _inherit(self, attr, value=None, default=None):
        """
        Try to inherit property if `value` was not provided.
        """
        return getattr(self, '_' + attr, default) if value is None else value

    def child(self, name, *args, other_parents=None, **kwargs):
        """
        Return a new child variable with properties inherited from the current one.
        The resulting parent and child variables support basic grouping behavior, i.e.
        ``var.child('foo') in var`` returns ``True`` and ``iter(var)`` iterates over
        the variable and all of its descendents. Use the `CFVariableRegistry.define`
        interface to add child variables to the registry.

        Parameters
        ----------
        name : str
            The child variable name.
        other_parents : tuple of `CFVariable`, optional
            Additional parents. Variable information will not be copied from these
            variables, but this can be useful for the purpose of grouping behavior.
        *args, **kwargs
            Passed to `CFVariable.update`.
        """
        # Parse input args
        parents = other_parents or ()
        if isinstance(parents, CFVariable):
            parents = (parents,)
        if not all(isinstance(parent, CFVariable) for parent in parents):
            raise ValueError('Parents must be Desciptor instances.')

        # Create child variable
        child = copy.copy(self)
        child._parents = (self, *parents)  # static parent list
        child._children = []  # new mutable list of children
        child._name = name
        child._aliases = []
        child._standard_name = None
        child.update(*args, **kwargs)

        # Add to children of parent variables
        def _add_child_to_parents(obj):
            for parent in obj._parents:
                if set(parent.identifiers) & set(child.identifiers):
                    raise ValueError(f'Name conflict between {child} and parent {parent}.')  # noqa: E501
                if child in parent._children:
                    continue
                parent._children.append(child)
                _add_child_to_parents(parent)
        _add_child_to_parents(child)

        return child

    def update(
        self,
        long_name=None, standard_units=None, short_name=None, standard_name=None, *,
        long_prefix=None, long_suffix=None, short_prefix=None, short_suffix=None,
        symbol=None, reference=None, colormap=None, axis_scale=None, axis_reverse=None,
        axis_formatter=None, scalar_formatter=None,
    ):
        """
        Update the variable. This is called during initialization. Unspecified variables
        are kept at their current state. Internally, `CFVariable.child` variables are
        constructed by calling `CFVariable.update` on existing `CFVariable` objects
        copied with `copy.copy`.

        Parameters
        ----------
        long_name : str, optional
            The plot-friendly variable name.
        standard_units : str, optional
            The plot-friendly CF-compatible units string. Parsed with
            `~.unit.decode_units`.
        short_name : str, optional
            The shorter plot-friendly variable name. This is useful for describing
            the "category" of a variable and its descendents, e.g. "energy flux" for
            something with units watts per meters squared. If not provided, this is
            set to `long_name`. It is often useful to leave this unset and create
            child variables that inherit the parent `short_name`.
        standard_name : str, optional
            The unambiguous CF-defined variable name. If one does not exist, you may
            construct a reasonable one based on the `CF guidelines \
            <http://cfconventions.org/Data/cf-standard-names/docs/guidelines.html>`_.
            If not provided, an *ad hoc* `standard_name` is constructed by replacing
            the non-alphanumeric characters in `long_name` with underscores. Since
            the `standard_name` is supposed to be a *unique* variable identifier,
            it is never inherited from parent variables.
        long_prefix, long_suffix : str, optional
            Prefix and suffix to be added to the long name. Defaults to `short_prefix`
            and `short_suffix`.
        short_prefix, short_suffix : str, optional
            Prefix and suffix to be added to the short name. Also sets `long_prefix`
            and `long_suffix` if they were not specified.
        symbol : str, optional
            The TeX-style symbol used to represent the variable, e.g. ``R_o`` for the
            Rossby number or ``\\lambda`` for the climate sensitivity parameter.
        sigfig : int, optional
            The number of significant figures when printing this variable with
            `~CFVariable.scalar_label`.
        reference : float, optional
            The notional "reference" value for the variable (in units `standard_units`).
            Useful for parameter sweeps with respect to some value.
        colormap : colormap-spec, optional
            The appropriate colormap when plotting this quantity. Generally this should
            be parsed by `~proplot.constructor.Colormap`.
        axis_scale : scale-spec, optional
            The axis scale name when using this quantity as an axis variable. Generally
            this should be parsed by `~proplot.constructor.Scale`.
        axis_reverse : bool, optional
            Whether to reverse the axis by default when using this quantity as
            an axis variable.
        axis_formatter : formatter-spec, optional
            The number formatter when using this quantity as an axis variable. Parsed
            by `proplot.constructor.Formatter`. Set to ``False`` to revert to the
            default formatter instead of inheriting the formatter.
        scalar_formatter : formatter-spec, optional
            The number formatter when using this quantity as a scalar label. Parsed
            by `proplot.constructor.Formatter`. Set to ``False`` to revert to the
            default formatter instead of inheriting the formatter.
        """
        # Parse input names and apply prefixes and suffixes
        # NOTE: Important to add prefixes and suffixes to long_name *after* using
        # as default short_name. Common use case is to create "child" variables with
        # identical short_name using e.g. vreg.define('var', long_prefix='prefix')
        standard_units = self._inherit('standard_units', standard_units)
        long_name = self._inherit('long_name', long_name)
        short_name = self._inherit('short_name', short_name, default=long_name)
        long_name = self._modify_name(long_name, long_prefix or short_prefix, long_suffix or short_suffix)  # noqa: E501
        short_name = self._modify_name(short_name, short_prefix, short_suffix)
        standard_name = self._inherit('standard_name', standard_name)
        if standard_name is None and long_name is not None:
            standard_name = re.sub(r'\W', '_', long_name).strip('_').lower()
            standard_name = re.sub(r'_+', '_', standard_name)  # squeeze consecutive

        # Add attributes
        default_axis_formatter = 'auto'
        if axis_formatter is False:
            axis_formatter = default_axis_formatter
        default_scalar_formatter = ('sigfig', 2)
        if scalar_formatter is False:
            scalar_formatter = default_scalar_formatter
        self._standard_units = standard_units
        self._long_name = long_name
        self._short_name = short_name
        self._standard_name = standard_name
        self._symbol = self._inherit('symbol', symbol, default='').strip('$')
        self._reference = self._inherit('reference', reference)
        self._colormap = self._inherit('colormap', colormap, default='Greys')
        self._axis_formatter = self._inherit('axis_formatter', axis_formatter, default=default_axis_formatter)  # noqa: E501
        self._axis_reverse = self._inherit('axis_reverse', axis_reverse, default=False)
        self._axis_scale = self._inherit('axis_scale', axis_scale)
        self._scalar_formatter = self._inherit('scalar_formatter', scalar_formatter, default=default_scalar_formatter)  # noqa: E501

    # Core properties
    @property
    def name(self):
        """
        The canonical variable name.
        """
        return self._name

    @property
    def identifiers(self):
        """
        Tuple of unique variable identifiers (the `~CFVariable.name`, the
        `~CFVariable.standard_name`, and the `~CFVariable.aliases`).
        """
        names = (self.name,)
        if self.standard_name:
            names += (self.standard_name,)
        names += self.aliases
        return names

    @property
    def aliases(self):
        """
        Aliases for the variable name.
        """
        return tuple(self._aliases)  # return an unmodifiable copy

    @property
    def long_name(self):
        """
        The plot-friendly variable name.
        """
        return self._long_name

    @property
    def short_name(self):
        """
        The shorter plot-friendly variable name.
        """
        return self._short_name

    @property
    def standard_name(self):
        """
        The CF-style standard name.
        """
        return self._standard_name

    @property
    def standard_units(self):
        """
        The plot-friendly standard units.
        """
        return self._standard_units

    @property
    def children(self):
        """
        The variable children.
        """
        return self._children

    @property
    def symbol(self):
        """
        Mathematical symbol for this variable. Default is the first letter of the name.
        """
        return '$' + (self._symbol or self._name[:1]) + '$'

    @property
    def reference(self):
        """
        The "reference" value associated with this variable.
        """
        return self._reference

    # Axis properties
    @property
    def axis_formatter(self):
        """
        The axis formatter used when plotting against this variable.
        """
        from proplot import Formatter
        return Formatter(self._axis_formatter)

    @property
    def axis_reverse(self):
        """
        Whether to reverse the axis when plotting against this variable.
        """
        return self._axis_reverse

    @property
    def axis_scale(self):
        """
        The axis scale when plotting against this variable.
        """
        return self._axis_scale

    @property
    def scalar_formatter(self):
        """
        The formatter used for scalar labels of this variable.
        """
        from proplot import Formatter
        return Formatter(self._scalar_formatter)

    # Color properties
    @property
    def color(self):
        """
        The color associated with the variable.
        """
        return self.colormap_line(0.5)

    @property
    def colormap(self):
        """
        The colormap associated with the variable.
        """
        import proplot as plot
        cmap = plot.Colormap(self._colormap)
        if self.axis_reverse:
            cmap = cmap.reversed()
        return cmap

    @property
    def colormap_line(self):
        """
        The truncated colormap associated with the variable.
        """
        # WARNING: Kludge for truncating blacks from stellar map
        cmap = self.colormap
        trunc = 0.1  # truncation
        offset = 0.05  # offset from white or black
        for i, (side, coord) in enumerate(zip(('right', 'left'), (1 - trunc, trunc))):
            if np.mean(cmap(i)[:3]) >= 1 - offset or np.mean(cmap(i)[:3]) <= offset:
                cmap = cmap.truncate(**{side: coord})
        return cmap

    # Label properties
    @property
    def long_label(self):
        """
        Readable plot label using the long name and units.
        """
        label = self.long_name
        units = self.units_label
        if units:
            label += f' ({units})'
        return label

    @property
    def short_label(self):
        """
        Readable plot label using the short name and units.
        """
        label = self.short_name
        units = self.units_label
        if units:
            label += f' ({units})'
        return label

    @property
    def scalar_label(self, pad=2):
        """
        Scalar label. This *always* shows the standardized form.
        """
        units = self.units_label.strip('$')
        symbol = self.symbol
        accessor = self._accessor
        formatter = self.scalar_formatter
        value = accessor.to_standard_units()
        value = value.climo.dequantify()
        value = value.item()
        if np.isnan(value):
            value = 'NaN'
        else:
            value = formatter(value)
        if '.' in value:
            if value[-1] == '1':  # close enough
                value = value[:-1]
            value = value.rstrip('0').rstrip('.')
        string = rf'{symbol} = {value}$ \, {units}$'
        string = ' ' * pad + string
        return string

    @property
    def units_pint(self):
        """
        The standard unit string translated to `pint.Unit`.
        """
        if self.standard_units is None:
            raise RuntimeError(f'No standard units for variable {self!r}.')
        else:
            return decode_units(self.standard_units)

    @property
    def units_label(self):
        """
        The active accessor units or the standard units formatted for matplotlib
        labels. When they are equivalent (as determined by `pint`), the
        `~CFVariable.standard_units` string is used rather than the `pint.Unit` object.
        This permits specifying a numerator and denominator by the position of the
        forward slash in the standard units string. See `~.unit.format_units` for
        details.
        """
        if self.standard_units is None:
            if not self._accessor:
                raise RuntimeError(f'Accessor required for variable with no default units {self!r}.')  # noqa: E501
            units = self._accessor.units
        elif self._accessor and self.units_pint != self._accessor.units:
            units = self._accessor.units
        else:
            units = self.standard_units
        return format_units(units)  # format, retaining slashes and whatnot


class CFVariableRegistry(object):
    """
    Container of `CFVariable` instances supporting aliases and *ad hoc* generation
    of `CFVariable` copies with properties modified by the coordinate cell methods.
    Integrated with the xarray accessor via
    `~.accessor.ClimoDataArrayAccessor.cfvariable`.
    """
    def __init__(self):
        self._database = {}
        self.CFVariable = self._build_cfvariable_class()

    def __contains__(self, key):
        try:
            self._get_item(key)
        except KeyError:
            return False
        else:
            return True

    def __getitem__(self, key):
        return self._get_item(key)

    def __getattr__(self, attr):
        if attr[:1] == '_':
            return super().__getattribute__(attr)  # trigger error
        try:
            return self._get_item(attr)
        except KeyError:  # critical to raise AttributeError so can use e.g. hasattr()
            raise AttributeError(f'Unknown CFVariable {attr!r}.')

    def __setattr__(self, attr, value):
        if attr[:1] == '_':
            super().__setattr__(attr, value)
        else:
            raise RuntimeError('Cannot set attributes on variable registry.')

    def __iter__(self):
        for key, value in self._database.items():
            if key[:1] != '_':
                yield key, value

    def __call__(
        self,
        name,
        accessor=None,
        longitude=None,
        latitude=None,
        vertical=None,
        time=None,
        **kwargs
    ):
        """
        Return a copy of a variable with optional name and unit modifications based
        on the coordinate reduction methods and optional pairing to an existing
        `~.accessor.ClimoDataArrayAccessor`. Inspired by `pint.UnitRegistry.__call__`.

        Parameters
        ----------
        name : str
            The variable name
        accessor : `~.accessor.ClimoDataArrayAccessor`
            The accessor (required for certain labels). Automatically passed
            when requesting the accessor property
            `~.accessor.ClimoDataArrayAccessor.cfvariable`
        longitude, latitude, vertical, time : optional
            Reduction method(s) for standard CF coordinate axes. Taken from
            corresponding dimensions in the `cell_methods` attribute when requesting
            the accessor property `~.accessor.ClimoDataArrayAccessor.cfvariable`.
        **kwargs
            Passed to `CFVariable.update`. Prefixes and suffixes are applied at
            the end, while name replacements are applied at the beginning

        Note
        ----
        For ``integral`` coordinate reductions, instructions for dealing with unit
        changes must be hardcoded in this function. So far this is done for energy
        budget, momentum budget, and Lorenz budget terms and their children.

        Todo
        ----
        Add non-standard `wavenumber`, `frequency`, and `ensemble` dimension options.
        We're already using non-standard reduction methods, so this isn't a stretch.
        """
        # Helper functions
        def _pop_integral(*args):
            b = all('integral' in arg for arg in args)
            if b:
                for arg in args:
                    arg.remove('integral')
            return b
        def _as_set(arg):  # noqa: E301
            arg = arg or set()
            if isinstance(arg, (tuple, list, set)):
                return set(arg)
            else:
                return {arg}

        # Parse input arguments and support reduction method specifications on varname
        # TODO: Expand this section
        longitude = _as_set(longitude)
        latitude = _as_set(latitude)
        vertical = _as_set(vertical)
        time = _as_set(time)
        for method in ('autocorr',):
            if m := re.match(rf'\A(.+)_{method}\Z', name or ''):
                name = m.group(1)
                time.add(method)

        # Get variable
        var = self._get_item(name)
        if var.name[0] == 'c' and 'convergence' in var.long_name and _pop_integral(latitude):  # noqa: E501
            var = self._get_item(name[1:])  # Green's theorem; e.g. cehf --> ehf
        var = copy.copy(var)
        var._accessor = accessor

        # Apply basic overrides
        kwmod = {  # update later!
            key: kwargs.pop(key) for key in tuple(kwargs)
            if key in ('long_prefix', 'long_suffix', 'short_prefix', 'short_suffix')
        }
        var.update(**kwargs)

        # Handle unit changes due to integration
        # NOTE: Vertical integration should always be with units kg/m^2
        # NOTE: Pint contexts apply required multiplication/division by c_p and g.
        if var in self.meridional_momentum_flux and _pop_integral(longitude):
            units = 'TN' if _pop_integral(vertical) else 'TN 100hPa^-1'
            var.update(
                long_name=var.long_name.replace('flux', 'transport'),
                short_name='momentum transport',
                standard_units=units,
            )
        elif var in self.meridional_energy_flux and _pop_integral(longitude):
            units = 'PW' if _pop_integral(vertical) else 'PW 100hPa^-1'
            var.update(
                long_name=var.long_name.replace('flux', 'transport'),
                short_name='energy transport',
                standard_units=units,
            )
        elif _pop_integral(vertical):
            # NOTE: Earth surface area is 500 x 10^12 m^2 and 10^12 is Tera,
            # 10^15 is Peta, 10^18 is Exa, 10^21 is Zeta.
            if var in self.energy:
                units = 'ZJ' if _pop_integral(longitude, latitude) else 'MJ m^-2'
                var.update(standard_units=units)  # same short name 'energy content'
            elif var in self.energy_flux:
                units = 'PW' if _pop_integral(longitude, latitude) else 'W m^-2'
                var.update(standard_units=units)  # same short name 'energy flux'
            elif var in self.acceleration:  # includes flux convergence
                units = 'TN' if _pop_integral(longitude, latitude) else 'Pa'
                var.update(standard_units=units, short_name='eastward stress')
            else:
                vertical.add('integral')  # raises error below

        # If *integral* reduction is ignored raise error, because integration
        # always changes units and that means the 'standard' units are incorrect!
        for dim, methods in zip(
            ('longitude', 'latitude', 'vertical'),
            (longitude, latitude, vertical),
        ):
            if 'integral' in methods:
                raise ValueError(
                    f'Failed to adjust units for {name!r} with {dim}={methods!r}.'
                )

        # Latitude dimension reduction of variable in question
        args = latitude & {'argmin', 'argmax', 'argzero'}
        if args:
            var.update(
                short_name='latitude',
                standard_units='deg_north',
                symbol=fr'\phi_{{{var.symbol}}}',
                axis_formatter='deg',
                long_suffix=f'{args.pop()[3:]} latitude',  # use the first one
                # long_suffix='latitude',
            )

        # Centroid reduction
        if 'centroid' in latitude:
            var.update(
                long_suffix='centroid',
                short_name='centroid',
                standard_units='km',
                axis_formatter=False,
            )

        # Time dimension reductions of variable in question
        if 'timescale' in time:
            var.update(  # modify existing
                long_suffix='e-folding timescale',
                short_name='timesale',
                standard_units='day',
                symbol=fr'T_e({var.symbol})',
                axis_formatter=False,
            )
        elif 'autocorr' in time:
            var.update(  # modify existing
                long_suffix='autocorrelation',
                short_name='autocorrelation',
                standard_units='',
                symbol=fr'\rho({var.symbol})',
                axis_formatter=False,
            )
        elif 'hist' in time:
            var.update(
                long_suffix='histogram',
                short_name='count',
                standard_units='',
                axis_formatter=False,
            )

        # Exact coordinates
        coords = [
            rf'{method.magnitude}$\,${format_units(method.units)}'
            if isinstance(method, pint.Quantity) else str(method)
            for methods in (longitude, latitude, vertical, time) for method in methods
            if isinstance(method, (pint.Quantity, numbers.Number))
        ]
        if coords:
            var.update(long_suffix='at ' + ', '.join(coords))
        if any('normalized' in dim for dim in (longitude, latitude, vertical, time)):
            var.update(standard_units='')

        # Finally add user-specified prefixes and suffixes
        var.update(**kwmod)
        return var

    def _build_cfvariable_class(self):
        """
        Create a `CFVariable` subclass that uses the input registry. This mimics
        pint's registry-specific Unit and Quantity internals.
        """
        class CFVariable(_CFVariable):
            _registry = self
        return CFVariable

    def _get_item(self, key):
        """
        Efficiently retrieve a variable based on its canonical name, name alias, or
        standard name.
        """
        if not isinstance(key, str) or key[:1] == '_':
            raise TypeError(f'Invalid variable name {key!r}.')
        try:
            return self._database[key]  # avoid iterating through registry if possible
        except KeyError:
            pass
        for var in self._database.values():
            if key == var.standard_name or key in var.aliases:
                return var
        raise KeyError(f'Unknown CFVariable {key!r}.')

    def alias(self, *args):
        """
        Add aliases for an existing registered `CFVariable`. This can be useful
        for identifying dataset variables from a wide variety of sources that have
        empty `standard_name` attributes. The variable aliasing system was
        inspired by the robust unit aliasing system of `pint.UnitRegistry`.

        Parameters
        ----------
        *args
            Strings which should identically refer to the variable. At least
            one of these must match the canonical name, standard name, or
            existing alias of an existing variable.
        """
        vars = []
        for arg in args:  # iterate over a copy
            try:
                var = self._get_item(arg)
            except KeyError:
                continue
            if var not in vars:
                vars.append(var)
        if not vars:
            raise ValueError(f'No variables found using {args=}.')
        if len(vars) > 1:
            raise ValueError(f'Multiple variables found with {args=}: {tuple(map(str, vars))}')  # noqa: E501
        var = vars[0]
        var._aliases.extend(arg for arg in args if arg not in var.identifiers)

    def define(self, name, *args, aliases=None, parents=None, **kwargs):
        """
        Define a `CFVariable` instance. Inspired by `pint.UnitRegistry.define`.

        Parameters
        ----------
        name : str
            The registered variable name. The CFVariable property ``property`` can be
            retrieved for data arrays whose `name` match this name using
            ``data_array.climo.cfvariable.property`` or the shorthand form
            ``data_array.climo.property``.
        aliases : str or list of str, optional
            Aliases for the `CFVariable`. This can be useful for identifying dataset
            variables from a wide variety of sources that have empty `standard_name`
            attributes. As an example you might register an air temperature variable
            ``t`` with ``aliases=('ta', 'temp', 'air_temp')``.
        parents : str or list of str, optional
            The parent variable name(s). Unspecified variable properties are
            inherited from the first one, and variable grouping is based on all
            of them. For example, defining the variable ``eddy_potential_energy`` with
            ``parents=('lorenz_energy_budget_term', 'energy_flux')`` will yield both
            ``'eddy_potential_energy' in vreg['lorenz_energy_budget_term']``
            and ``'eddy_potential_energy' in vreg['energy_flux']``.
        *args, **kwargs
            Passed to `CFVariable`.

        Notes
        -----
        If the input variable's `~CFVariable.identifiers` conflict with an existing
        variable's `~CFVariable.identifiers`, a warning message is printed and the
        conflicting variable is removed from the registry (along with its children).
        """
        # Create new variable or child variable
        if not parents:
            var = self.CFVariable(name, *args, **kwargs)
        else:
            if isinstance(parents, str):
                parents = (parents,)
            parents = tuple(self._get_item(parent) for parent in parents)
            var = parents[0].child(name, *args, other_parents=parents[1:], **kwargs)

        # Add aliases to variable
        aliases = aliases or ()
        if isinstance(aliases, str):
            aliases = (aliases,)
        var._aliases.extend(aliases)

        # Delete old variables and children, ensuring zero conflict between sets
        # of unique identifiers and forbidding deletion of parent variables.
        for identifier in set(var.identifiers):
            try:
                prev = self._get_item(identifier)
            except KeyError:
                pass
            else:
                warnings._warn_climopy(f'Overriding {prev} with {var}.')
                for other in prev:  # iterate over self and all children
                    if other is var:  # i.e. we are trying to override a parent!
                        raise ValueError(f'Name conflict between {var} and parent {other}.')  # noqa: E501
                    self._database.pop(other.name, None)

        # Add variable to database and return it
        self._database[name] = var
        return var

    def get(self, key, default=None):
        """
        Get the variable if present, otherwise return a default value.
        """
        try:
            return self._get_item(key)
        except (KeyError, TypeError):  # allow e.g. vreg.get(None, default)
            return default


# Alternate name for registry-specific subclass inheritance. This mimics
# pint's registry-specific Unit and Quantity internals.
_CFVariable = CFVariable

#: The default `CFVariableRegistry` paired with `~.accessor.ClimoDataArrayAccessor`
#: xarray accessor instances. `CFVariable` properties can be retrieved from an
#: `xarray.DataArray` whose name matches a variable name using
#: ``data_array.climo.cfvariable.property`` or the shorthand form
#: ``data_array.climo.property``.
vreg = CFVariableRegistry()

#: Alias for the default `CFVariableRegistry` `vreg`. Mimics the alias `~.unit.units`
#: used for the default unit registry `~.unit.ureg`.
variables = vreg
