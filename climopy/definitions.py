#!/usr/bin/env python3
"""
Define and register the default variables, transformations, and derivations.
"""
import warnings

import numpy as np

from . import const
from .accessor import register_derivation, register_transformation
from .cfvariable import vreg
from .internals import ic  # noqa: F401
from .internals.warnings import ClimoPyWarning, _warn_climopy
from .unit import ureg

# Ignore override warnings arising due to autoreload. Only want to issue warnings
# when users are *manually* overriding things.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=ClimoPyWarning)

    # Coordinates
    vreg.define('latitude', 'latitude', 'degrees_north', axis_scale='sine', axis_formatter='deg')  # noqa: E501
    vreg.define('longitude', 'longitude', 'degrees_east', axis_formatter='deg')
    vreg.define('meridional_coordinate', 'meridional coordinate', 'km')
    vreg.define('cosine_latitude', 'cosine latitude', '')
    vreg.define('coriolis_parameter', 'Coriolis parameter', 's^-1')
    vreg.define('beta_parameter', 'beta parameter', 'm^-1 s^-1')
    vreg.define('reference_height', 'reference height', 'km')
    vreg.define('reference_density', 'reference density', 'kg m^-3')
    vreg.define('reference_potential_temperature', 'reference potential temperature', 'K')  # noqa: E501

    # Physical variables
    # TODO: Add to these
    vreg.define('energy', 'energy', 'MJ m^-2 100hPa^-1')  # static/kinetic
    vreg.define('energy_flux', 'vertical energy flux', 'W m^-2 100hPa^-1')
    vreg.define('meridional_energy_flux', 'meridional energy flux', 'MJ m^-2 100hPa^-1 m s^-1')  # noqa: E501
    vreg.define('momentum', 'momentum', 'm s^-1')
    vreg.define('meridional_momentum_flux', 'meridional momentum flux', 'm^2 s^-2')
    vreg.define('acceleration', 'acceleration', 'm s^-1 day^-1')

    # Register coordinate-related transformations and derivations
    @register_transformation('latitude', 'meridional_coordinate')
    def meridional_coordinate(da):
        return (da * const.a).climo.to_units('km')

    @register_transformation('latitude', 'cosine_latitude')
    def cosine_latitude(da):
        return np.cos(da)

    @register_transformation('latitude', 'coriolis_parameter')
    def coriolis_parameter(da):
        return 2 * const.Omega * np.sin(da)

    @register_transformation('latitude', 'beta_parameter')
    def beta_parameter(da):
        return np.abs(2 * const.Omega * np.cos(da)) / const.a

    @register_transformation('vertical', 'reference_height')
    def reference_height(da):
        return (const.H * np.log(const.p0 / da)).climo.to_units('km')

    @register_transformation('vertical', 'reference_density')
    def reference_density(da):
        return (da / (const.Rd * ureg('300K'))).climo.to_units('kg m^-3')

    @register_transformation('vertical', 'reference_potential_temperature')
    def reference_potential_temperature(da):
        return ureg('300K') * (const.p0 / da).climo.to_units('') ** const.kappa

    @register_derivation('cell_width')
    def cell_width(self):
        coord = const.a * self.coords['cosine_latitude'] * self.coords['longitude_del']
        return coord.climo.to_units('km')

    @register_derivation('cell_depth')
    def cell_depth(self):
        coord = const.a * self.coords['latitude_del']
        return coord.climo.to_units('km')

    @register_derivation('cell_duration')
    def cell_duration(self):
        coord = self.coords['time_del']
        return coord.climo.to_units('days')

    @register_derivation('cell_height')
    def cell_height(self):
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
                _warn_climopy(
                    'Surface pressure not found. '
                    'Vertical mass weighting will be inaccurate in lower levels.'
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
