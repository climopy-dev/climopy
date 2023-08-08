#!/usr/bin/env python3
"""
Define and register the default variables, transformations, and derivations.
"""
import numpy as np

from . import const
from .accessor import register_derivation, register_transformation
from .cfvariable import vreg
from .internals import ic  # noqa: F401
from .internals import warnings
from .unit import ureg

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


# Register coordinate-related transformations
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


# Register coordinate-related derivations
@register_derivation('cell_width')
def cell_width(self):
    # NOTE: Add measures as coords to be consistent with result of ds['cell_width']
    # for datasets with measures already present in coordinates, while preventing
    # recalculation of exact same measures.
    data = const.a * self.coords['cosine_latitude'] * self.coords['longitude_delta']  # noqa: E501
    data = data.astype(np.float32)
    data = data.climo.to_units('km')  # removes 'deg'
    data.name = 'cell_width'  # avoids calculation of other measures
    data = data.climo.add_cell_measures(width=data)  # auto dequantified
    return data


@register_derivation('cell_depth')
def cell_depth(self):
    # NOTE: Depth is interpreted as if looking northward at 3D cell rectangle.
    # Think of depth as 'into the distance' instead of 'into the ground'.
    data = const.a * self.coords['latitude_delta']
    data = data.astype(np.float32)
    data = data.climo.to_units('km')  # removes 'deg'
    data.name = 'cell_depth'
    data = data.climo.add_cell_measures(depth=data)  # auto dequantified
    return data


@register_derivation('cell_duration')
def cell_duration(self):
    # NOTE: Xarray coerces datetime.timedelta arrays resulting from cftime
    # subtraction into native timedelta64 arrays. See _make_coords for details.
    # TODO: Auto-detect monthly data and apply days-per-month weightings independent
    # of actual days in the time coordinate (e.g. common to use central month day of
    # 14, 15, or 16 which would mess up the correct days-per-month weighting).
    data = self.coords['time_delta']
    if data.data.dtype.kind == 'm':  # datetime64, if time coordinate was decoded
        data = ureg.days * data.dt.days
    data = data.astype(np.float32)
    data = data.climo.to_units('days')
    data.name = 'cell_duration'
    data = data.climo.add_cell_measures(duration=data)  # auto dequantified
    return data


@register_derivation('cell_height')
def cell_height(self, surface=False, tropopause=False):
    # Initial stuff
    # NOTE: Previously used air_pressure_at_mean_sea_level for aquaplanet
    # models but better instead to always use surface pressure... otherwise
    # this will give weird weightings for datasets with realistic topography.
    type_ = self.cf.vertical_type
    data = self.coords['vertical_bnds']
    data = data.astype(np.float32)
    dbot = dtop = None
    name = self.cf._decode_name('vertical')
    bools = data.isel(bnds=1) >= data.isel(bnds=0)
    if np.all(bools):
        sign = 1
    elif not np.any(bools):
        sign = -1
    else:
        raise RuntimeError('Mixed directionality of vertical coordinates.')
    if type_ == 'height':
        scale = self['density']
        bot = 'surface_altitude'
        top = 'tropopause_altitude'
    elif type_ == 'pressure':
        scale = 1.0 / const.g
        bot = 'surface_air_pressure'
        top = 'tropopause_air_pressure'
    elif type_ == 'temperature':
        scale = self['pseudo_density']
        bot == 'surface_potential_temperature'  # TODO add this
        top = 'tropopause_air_potential_tmeperature'  # TODO add this
    else:
        raise NotImplementedError(f'Unknown cell height for vertical type {type_!r}.')  # noqa: E501

    # Apply surface and tropopause dependence
    if surface:
        # Contract too-low near-surface bounds onto surface pressure
        if bot in self:
            dbot = self.get(bot, add_cell_measures=False)
        else:
            raise RuntimeError(f'Variable {bot!r} is not available for cell heights.')
        levs = 0 * data + dbot
        data = (data + 0 * levs).transpose(*levs.dims)
        mask = data.data > levs.data
        data.data[mask] = levs.data[mask]
        # Expand too-high surface bounds onto surface pressure
        idx = -1 if sign == 1 else 0
        loc = {'bnds': idx}
        if 'vertical' in data.cf.sizes:
            loc['vertical'] = idx
        levs_loc = levs.climo[loc].data
        data_loc = data.climo[loc].data
        mask = data_loc < levs_loc
        data_loc[mask] = levs_loc[mask]
    if tropopause:
        # Contract too-high near-tropopause bounds onto tropopuse
        # NOTE: Algorithm most often fails in cold stably stratified regions
        # e.g. near poles so use a low default tropopause pressure.
        if top in self:
            dtop = self.get(top, add_cell_measures=False)
        else:
            raise RuntimeError(f'Variable {top!r} is not available for cell heights.')
        if dtop.isnull().any():
            prop = 100 * dtop.isnull().sum().item() / dtop.size
            warnings._warn_climopy(f'Setting {prop:.2f}% NaN-valued tropopause levels to 300hPa.')  # noqa: E501
            dtop.data[dtop.isnull().data] = 300 * ureg.hPa
        levs = 0 * data + dtop
        data = (data + 0 * levs).transpose(*levs.dims)
        mask = data.data < levs.data
        data.data[mask] = levs.data[mask]

    # Convert units and correct invalid weights
    # TODO: Have add_cell_measures() handle the result and add the coordinate
    # internally instead of calling recursively and modifying the dataset in-place.
    data.data[data.data < 0 * data.data.units] = 0 * data.data.units
    data = data.diff('bnds').isel(bnds=0)
    data = scale * sign * data
    data = data.climo.to_units('kg m^-2')
    mask = np.isnan(data.data.magnitude) | np.isclose(data.data.magnitude, 0)
    data.data.magnitude[mask] = 0
    data.name = 'cell_height'
    data = data.climo.add_cell_measures(height=data)  # auto dequantified
    if dbot is not None:
        self.data.coords[f'{name}_bot'] = dbot.climo.dequantify()
    if dtop is not None:
        self.data.coords[f'{name}_top'] = dtop.climo.dequantify()
    return data
