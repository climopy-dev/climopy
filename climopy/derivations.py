#!/usr/bin/env python3
"""
Tools for defining physical variable derivations. These are integrated into the
climopy accessors for efficient retrieval based on `xarray.Varaible` names.
"""
import functools

import cf_xarray as xcf  # noqa: F401
import xarray as xr

from .cfvariable import CFVariable, vreg
from .internals import ic  # noqa: F401
from .internals import docstring, quack, quant, warnings

# Global constants
# NOTE: Here autoreload will reset the derivations but not huge deal.
DERIVATIONS = {}


@docstring._snippet_manager()
def register_derivation(vars_in, vars_out, assign_name=True, **kwargs):
    """
    Register a function that derives one output variable from arbitrarily many input
    variables for use with `ClimoDatasetAccessor.get` (e.g. ``ds['derived_var']``).
    The decorated function should accept arrays corresponding to the input variable
    names as input arguments and return a single array corresponding to the output
    variable name (see below). It should also be able to handle `~xarray.DataArray`
    input without stripping metadata (e.g. by using other climopy utiltiies).

    Parameters
    ----------
    vars_in : var-spec or str or sequence
        The variables for the positional input arguments. Can be a
        `~.cfvariable.CFVariable` or a variable specification string like ``'t'``
        or ``'air_temperature'``. Input arrays passed to the decorated function are
        quantified during execution using `~.internals.quant.while_quantified` with
        ``convert=True`` to enforce unit compatibility. The units are inferred from
        the associated `~.cfvariable.CFVariable.standard_units`. Vertical bars
        can be used to allow multiple distinct variables, for example
        ``register_derivation('u | v', 'dudy | dvdy')`` could be used for a
        function that computes meridional shear in the zonal or meridional
        components of the wind (similar to `~.internals.quant.while_quantified`).
    vars_out : var-spec or str or sequence
        As with `vars_in`, but for the return values.
    assign_name : bool, default: True
        Whether to assign the user-input string as the output `xarray.DataArray.name`.
    **kwargs
        Additional keyword arguments passed to `~.internals.quant.while_quantified`.

    Returns
    -------
    callable
        The function decorated by `~internals.quant.while_quantified` and registered
        for use with `ClimoDatasetAccessor.get` (e.g. ``ds['derived_var']``).

    Note
    ----
    The available dataset variables are automatically translated using
    `~ClimoAccessor.variable_registry` whenever a derivation is requested
    through `ClimoDatasetAccessor.get`. For example, if a derivation requiring
    only temperature ``'t'`` is registered, and the dataset contains a data array
    named ``'temp'`` that points to the same `~ClimoDataArrayAccessor.cfvariable`
    (e.g., because they have the same ``standard_name``), then this data array
    will be used for the derivation.

    Examples
    --------
    >>> import climopy as climo
    >>> from climopy import const
    >>> @climo.register_derivation('temp', 'pres', 'pot_temp')
    ... def potential_temp(temp, pres):
    ...     return temp * (const.p0 / pres).climo.to_units('') ** (2 / 7)
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
    # Define helper function
    # TODO: Use the variable registry specific to each dataset?
    # NOTE: Initially considered permitting non-cfvariable names that are recognized
    # by the CF standard (axis names, coordinates, cell measures), but that would
    # require extra step of validating names against internal cf-xarray dictionaries
    # and manually defining associated units. Simpler instead to register coordinates
    # and cell measures as cfvariables and enforce all register_derivation arguments
    # are valid cfvariable names. Then we added an exception to climo.cfvariable so that
    # it returns the appropriate cfvariable even for data arrays without matching names
    # or standard names (e.g. longitude coordinate named 'foobar' defined by its units).
    def _parse_variable(arg):
        if isinstance(arg, CFVariable):
            var = arg
        elif isinstance(arg, str):
            var = vreg[arg]
        else:
            raise TypeError(f'Invalid variable specification {arg!r}.')
        return var.name, var.standard_units

    # Parse the variables and units
    # NOTE: Initially considered permitting mixed cfvariable specifications and unit
    # specifications with e.g. '=meter', analogous to dependent unit specifications
    # passed to while_quantified. However any definition that requires non-cfvariable
    # input could not be retrieved from the dataset accessor, and any definition that
    # creates non-cfvariable output could not be looked up.
    seen = set()
    names_in, names_out = [], []
    units_in, units_out = [], []
    vars_in, vars_out, is_scalar_out = quant._group_args(vars_in, vars_out)
    for ivars, ovars in zip(vars_in, vars_out):  # iterate over groups
        inames, iunits = zip(*map(_parse_variable, ivars))
        onames, ounits = zip(*map(_parse_variable, ovars))
        for name in (*inames, *onames):
            if name in seen:
                raise ValueError(f'Variable {name!r} already used in definition.')
            seen.add(name)
        names_in.append(inames)
        names_out.append(onames)
        units_in.append(iunits)
        units_out.append(ounits)

    def _decorator(func):
        # Quantify original function
        # NOTE: Here we skip applicaton of quant._group_args in while_quantified.
        # TODO: Add public xarray wrappers so that user physical variable definitions
        # can utilize xarray constructions.
        kwargs['grouped'] = True
        kwargs.setdefault('convert_units', False)
        kwargs.setdefault('require_quantity', False)
        kwargs.setdefault('require_metadata', False)
        func_quantified = quant.while_quantified(func, units_in, units_out, **kwargs)

        # Create dataarray or dataset version of function. Will be used by
        # ClimoDataArrayAccessor.to_variable and ClimoDatasetAccessor.get.
        # TODO: Derivations should accept input arrays passed by helper function
        # that searches registered derivations (do this before commit)!
        @quack._keep_cell_attrs
        @functools.wraps(func_quantified)
        def _derivation(self, names_in, names_out):  # self is the ClimoAccessor
            arrays = [self.get(name) for name in names_in]
            results = func_quantified(*arrays)
            results = [results] if is_scalar_out else list(results)
            for name, result in zip(names_out, results):  # allow extra return values
                if not assign_name:
                    pass
                elif isinstance(result, xr.DataArray):
                    result.name = name
                else:
                    warnings._warn_climopy(
                        f'Failed to assign name {name!r}. '
                        f'Output is {type(result)!r} instead of DataArray.'
                    )
            if is_scalar_out:
                return results[0]
            else:
                return tuple(results)

        # Register the resulting function as a (names_in, names_out) tuple.
        # This was originally how transformations were registered.
        for key in zip(names_in, names_out):
            if key in DERIVATIONS:
                warnings._warn_climopy(f'Overriding existing derivation {key!r}.')
            DERIVATIONS[key] = _derivation

        return func_quantified

    return _decorator
