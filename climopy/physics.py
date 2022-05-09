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
from .internals.warnings import ClimoPyWarning
from .unit import ureg

# Ignore override warnings arising due to autoreload. Only want to issue warnings
# when users are *manually* overriding things.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=ClimoPyWarning)

    # Coordinates and cell methods
    # NOTE: Vertical and time units are so varied that we do not include them.
    vreg.define('longitude', 'longitude', 'degE', axis_formatter='deg')
    vreg.define('latitude', 'latitude', 'degN', axis_scale='sine', axis_formatter='deg')  # noqa: E501
    vreg.define('cell_length', 'cell length', 'km')  # longitude
    vreg.define('cell_width', 'cell depth', 'km')  # latitude
    vreg.define('cell_height', 'cell height', 'kg m^-2')
    vreg.define('cell_duration', 'cell duration', 'days')
    vreg.define('cell_area', 'cell area', 'km^2')
    vreg.define('cell_volume', 'cell volume', 'kg')

    # Coordinate derivations
    # TODO: Rely on default standard name construction
    vreg.define('meridional_coordinate', 'meridional coordinate', 'km')
    vreg.define('cosine_latitude', 'cosine latitude', '')
    vreg.define('coriolis_parameter', 'Coriolis parameter', 's^-1')
    vreg.define('beta_parameter', 'beta parameter', 'm^-1 s^-1')
    vreg.define('reference_height', 'reference height', 'km')
    vreg.define('reference_density', 'reference density', 'kg m^-3')
    vreg.define('reference_potential_temperature', 'reference potential temperature', 'K')  # noqa: E501

    # Variable derivations
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
        return (const.a * da).climo.to_units('km')

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

    @register_derivation('cell_length')
    def cell_length(self):
        # NOTE: Add measures as coords to be consistent with result of ds['cell_length']
        # for datasets with measures already present in coordinates, while preventing
        # the recalculation of exact same measures.
        data = const.a * self.coords['cosine_latitude'] * self.coords['longitude_delta']  # noqa: E501
        data = data.climo.to_units('km')  # removes 'deg'
        data.name = 'cell_length'  # avoids calculation of other measures
        return data.climo.add_cell_measures(length=data)

    @register_derivation('cell_width')
    def cell_width(self):
        # NOTE: Width is interpreted as if looking northward at 3D cell rectangle.
        # Think of width as 'into the back' instead of 'across the front'
        data = const.a * self.coords['latitude_delta']
        data = data.climo.to_units('km')  # removes 'deg'
        data.name = 'cell_width'
        return data.climo.add_cell_measures(width=data)

    @register_derivation('cell_duration')
    def cell_duration(self):
        # NOTE: Xarray coerces datetime.timedelta arrays resulting from cftime
        # subtraction into native timedelta64 arrays. See _make_coords for details.
        data = self.coords['time_delta']
        if data.data.dtype.kind == 'm':  # datetime64, if time coordinate was decoded
            data = ureg.days * data.dt.days
        data = data.climo.to_units('days')
        data.name = 'cell_duration'
        return data.climo.add_cell_measures(duration=data)

    @register_derivation('cell_height')
    def cell_height(self):
        # WARNING: Previously used air_pressure_at_mean_sea_level for aquaplanet
        # models but better instead to always use surface pressure... otherwise
        # this will give weird weightings for datasets with realistic topography.
        type_ = self.cf.vertical_type
        if type_ == 'temperature':
            data = self.vars['pseudo_density'] * self.coords['vertical_delta']
        elif type_ == 'pressure':
            ps = None
            data = bnds = self.coords['vertical_bnds']
            if 'surface_air_pressure' in self.vars:
                # Set hard minimum bounds (possibly for multiple levels)
                ps = 0 * data + self.vars['surface_air_pressure']
                data = data + 0 * ps
                data = data.transpose(*ps.dims)  # conform dimensionality
                mask = data.data > ps.data
                data.data[mask] = ps.data[mask]
                # Expand near-surface vertical bounds to actual surface
                idx = {}
                if 'vertical' not in bnds.cf.sizes:
                    idx['bnds'] = bnds.data.argmax()  # e.g. single-level
                    pass
                else:
                    idx['bnds'] = bnds.cf.isel(vertical=0).data.argmax()
                    idx['vertical'] = bnds.isel(bnds=0).data.argmax()
                mask = data.climo[idx].data < ps.climo[idx].data
                data.climo[idx].data[mask] = ps.climo[idx].data[mask]
            data = data.diff('bnds').isel(bnds=0)
            data = data / const.g
        elif type_ == 'hybrid':
            raise NotImplementedError(
                'Cell height for hybrid coordinates not yet implemented.'
            )
        else:
            raise NotImplementedError(
                f'Unknown cell height for vertical type {type_!r}.'
            )
        data = data.climo.to_units('kg m^-2')
        if data.dims:  # non-scalar
            data.data[data.data < 0] = ureg.Quantity(0, 'kg m^-2')
        elif data.data < 0:
            data.data = ureg.Quantity(0, 'kg m^-2')
        data = data.fillna(0)  # seems to work without units in the 'fill' part
        data.name = 'cell_height'
        return data.climo.add_cell_measures(height=data)
