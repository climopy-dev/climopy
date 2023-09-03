#!/usr/bin/env python3
"""
Todo
----
These are the remaining EOF plotting utilities that have not been
transferred to the generalized plotting functions.
"""
# Use pint.UnitRegistry() just like metpy! Then do *every* calculation
# with units! Maybe we *parse* the units string for each DataArray and
# then just naturally handle units like that.
import re

import numpy as np
import xarray as xr

import climo as climo


def param_eof(
    vars, data, mode=None,
    lag=None, eof=None, nlag=None,  # nlag in days
    order=None, cutoff=None,
    lowpass=False, highpass=False, time=None,
    wintype='boxcar', nperseg=1000,
    plevrange=(0, 1000), latrange=(0, 90)
):
    """
    Returns a 1D numpy array for some EOF parameter in a variety of ways.
    So far this supports:

    * Co-power spectra.
    * Cross-correlations.
    * 1D power spectra.
    * Maximum phase speeds or wavenumbers.

    Parameters
    ----------
    vars : str or (str, str)
        The variable name or the two variable names corresponding to the
        projection of the first parameter onto an EOF of the second parameter.
    data : `xarray.Dataset`
        The dataset.
    mode : {'power', 'coherence', 'phase', 'cospectrum', 'quadrature', \
'cross', 'corr', 'var', 'lagvar'}
        The retrieval mode. This indicates what kind of parameter we want
        and is used to reduce an axis dimension in some way. This is not
        relevant for variable types that are already 1-dimensional, for example
        PC time series or eigenvalues.
    nlag : int
        Number of lags to use for autocorrelation plot.
    **kwargs
        Various other options relevant to the specific retrieval modes.

    Returns
    -------
    x : `Variable`
        The parameter coordinates.
    param : `Variable`
        The parameter. If list of 3 vectors, the edge 2 indicate error bounds.
    """
    # Mode used to interpret data
    if eof is None:
        eof = 1
    if lag is None:
        lag = 0
    if nlag is None:
        nlag = 50
    if order is None:
        order = 21
    if cutoff is None:
        cutoff = 10

    # Coordinates
    time = data['time'].values
    dt = time[1] - time[0]

    # Filter
    plevsel = slice(None)
    if plevrange is not None:
        plevsel = slice(*plevrange)
    latsel = slice(None)
    if latrange is not None:
        latsel = slice(*latrange)

    # Cross-correlation
    if mode == 'corr':
        sel1 = {'eof': eof}  # project on which lagged pattern?
        if isinstance(vars, str):
            vars = [f'{vars}_pc']
            sel2 = sel1
        else:
            vars = [f'{vars[0]}_pc', f'{vars[1]}_on_{vars[0]}']
            sel2 = {'eof': eof, 'lag': lag}

        # Get data
        values = [
            data[var].sel(**sel).values.squeeze()
            for sel, var in zip((sel1, sel2), vars)
        ]
        if any(ivalues.ndim != 1 for ivalues in values):
            raise ValueError('Did not get 1D arrays.')

        # Optionally filter
        # TODO: Fix this, right now get weird results
        if lowpass or highpass:
            # b, a = climo.butterworth(
            #    dt, order, cutoff, btype=('low' if lowpass else 'high')
            # ) # cut off at *10 day* cycles
            # b, a = climo.butterworth(
            #     dt, order, cutoff, btype='low'
            # ) # weird results!
            # length 50, cutoff 10 days
            b, a = climo.lanczos(dt, order, cutoff)
            valuesf = [None] * len(values)
            for i, ivalues in enumerate(values):
                # do not pad with NaNs
                ivaluesf = climo.filter(ivalues, b, a, pad=False)
                if highpass:
                    ivalues = ivalues[order // 2:-order // 2 + 2] - ivaluesf
                else:
                    ivalues = ivaluesf
                valuesf[i] = ivalues

        # Get correlation; x is lag (in units days)
        xparam, param = climo.corr(*values, nlag=nlag, dt=dt)

        # Variables
        xparam = xr.DataArray(xparam, name='lag')
        param = xr.DataArray(param, name='correlation')

    # Spectral analysis
    # NOTE: The 'cross' is just sum of cospectrum and quadrature spectrum
    elif mode in (
        'power', 'cospectrum', 'quadrature', 'coherence', 'phase', 'cross'
    ):
        sel1 = {'eof': eof}
        if isinstance(vars, str):
            if mode != 'power':
                raise ValueError(
                    'Can only get power spectrum for one variable.'
                )
            vars = [f'{vars}_pc']
            sels = [sel1]
        else:
            var1, var2 = vars
            sel2 = {'eof': eof, 'lag': lag}
            if mode == 'power':
                vars = [f'{var1}_on_{var2}']
                sels = [sel2]
            else:
                vars = [f'{var2}_pc', f'{var1}_on_{var2}']
                sels = [sel1, sel2]
        # Get power
        # We normalize because almost always interested in comparative *shape*
        # of time series rather than absolute power.
        # want the quadrature contribution, or just co-spectrum?
        values = [
            data[var].sel(**sel).values.squeeze() for sel, var in zip(sels, vars)  # noqa
        ]
        if any(ivalues.ndim != 1 for ivalues in values):
            raise ValueError(
                f'Got values of shape {[vals.shape for vals in values]}.')
        result = climo.power(
            *values, dx=dt, nperseg=nperseg, wintype=wintype,
            coherence=(mode in ('coherence', 'phase'))
        )
        if mode == 'coherence':  # inherently normalized
            xparam, param, _ = result
        elif mode == 'phase':  # different units
            xparam, _, param = result
        elif mode == 'power':  # normalize this
            xparam, param = result
            param = param / param.sum()
        else:
            # Select result
            if mode == 'cospectrum':
                xparam, param, _, py1, py2 = result
            elif mode == 'quadrature':
                xparam, _, param, py1, py2 = result
            elif mode == 'cross':
                xparam, y1, y2, py1, py2 = result
                param = np.sqrt(y1**2 + y2**2)
            # Potentially normalize
            # Just normalize by standard devs of each series
            param = param / (np.sqrt(py1.sum()) * np.sqrt(py2.sum()))
            # Doesn't make sense, we want *absolute* power at each freq
            # param = param / (np.sqrt(py1) * np.sqrt(py2))
            # Doens't make sense, since co-spectrum can be negative
            # param = param/param.sum()

        # Get x coordinates
        xparam = xr.DataArray(xparam, name='f')  # frequency
        param = xr.DataArray(param, name=mode)

    # Plot the variance explained by vertically averaging
    # WARNING: If you try to get variance explained in one variable by its
    # projection onto ***more than 1*** EOF of another variable, can get over
    # 100%, because those *projections* are not necessarily orthogonal! We
    # prohibit this!
    elif mode == 'var' or mode == 'lagvar':
        # Checks
        if isinstance(vars, str):
            raise ValueError(
                f'For variance explained plot, need two variables. '
                f'Received {vars}.'
            )
        var1, var2 = vars  # generally you want var1
        if np.iterable(eof):
            if var1 != var2:
                raise ValueError(
                    'Cannot get percent variance explained in variable '
                    'by more than one EOF of one variable, since projections '
                    'are not necessarily orthogonal!'
                )
            eof = slice(eof[0], eof[1])  # remember sel is endpoint inclusive!
        else:
            eof = slice(eof, eof)

        # Variance explained by projection of some "responding" quantity
        # to the scalar PC time series for some "forcing" quantity
        # First get this as function of *latitude*
        var1 = f'{var1}_on_{var2}_var'  # this is a *percentage*!
        var2 = f'{var1}_var'  # this is absolute
        if mode == 'var':
            # Get weights
            # print(f'Percent range: {percent.values.min():.2f},
            # print(f'Using vars: {var1} and {var2}')
            dplev = data['plev_bnds'].sel(plev=plevsel).values
            dplev = dplev[:, 1] - dplev[:, 0]
            weights = xr.DataArray(dplev / dplev.sum(), dims='plev')

            # Get data
            sel1 = {'eof': eof, 'plev': plevsel}
            sel2 = {'plev': plevsel}  # variance itself
            if 'lag' in data[var1].coords:
                sel1['lag'] = lag
            xparam = data['lat'].values
            part = 0.01 * (
                weights * data[var1].sel(**sel1) * data[var2].sel(**sel2)
            ).sum(dim=('plev', 'eof')).values
            total = (
                weights * data[var2].sel(**sel2)
            ).sum(dim='plev').values

            # Coordinate
            xparam = xr.DataArray(xparam, name='lat')  # frequency

        # Next get this as function of *lag*
        else:
            # Get weights
            # NOTE: Why weight
            dplev = data['plev_bnds'].sel(plev=plevsel).values
            dplev = dplev[:, 1:] - dplev[:, :1]  # keep 2nd dimensoin
            cos = np.cos(data['lat'].sel(lat=latsel).values * np.pi / 180)
            weights = xr.DataArray(dplev * cos, dims=('plev', 'lat'))
            weights = weights * data[var2].sel(lat=latsel, plev=plevsel)
            weights = weights / weights.sum(dim=('plev', 'lat'))
            if 'lag' not in data[var1].coords:
                raise ValueError(
                    f'Invalid parameter {var1}. '
                    'Need to pick one that has a lag dimension.'
                )

            # Variance in projection onto EOF
            sel1 = {'eof': eof, 'plev': plevsel, 'lat': latsel}

            # Actual variance (TODO: Make sure units are correct)
            sel2 = {'plev': plevsel, 'lat': latsel}
            xparam = data['lag'].values
            part = 0.01 * (
                weights * data[var1].sel(**sel1) * data[var2].sel(**sel2)
            ).sum(dim=(*sel2, 'eof')).values
            total = (
                weights * data[var2].sel(**sel2)
            ).sum(dim=(*sel2,)).values

            # Coordinate
            xparam = climo.Variable('lag', xparam)

        # Get variable
        param = climo.Variable('variance', part / total)

    # Just want to plot the eigenvalues
    # NOTE: Should already be in 'variance explained' units because
    # that's usually what is most useful.
    elif isinstance(vars, str) and 'eval' in vars:
        # Get values
        nstar = re.sub('eval', 'nstar', vars)  # e.g. u_eval
        xparam = data['eof'].values  # the EOF number
        param = data[vars].values.squeeze() / 100  # they are stored as percent

        # Get *error bounds*, return as 3xN array (this option is anticipated
        # by the stack() plotting function). This is 95% confidence interval,
        # assumed symmetric, so make each whisker *half* this size
        yerr = param * np.sqrt(2 / data[nstar].values.item()) / 2
        param = np.vstack((yerr, param, yerr))

        # Variables
        xparam = climo.Variable('eof', xparam)
        param = climo.Variable('eval', param)

    # The PC time series, e.g. ke_pc
    elif isinstance(vars, str) and 'pc' in vars:  # e.g. u_pc
        xparam = data['time'].values  # the EOF number
        xparam -= xparam[0]
        xparam += xparam[1] - xparam[0]
        xparam = climo.Variable('time', xparam)
        param = data[vars].sel(eof=eof).values.squeeze()
        param = climo.Variable(vars, param)

    # Error
    else:
        raise ValueError(f'Unknown plotting mode {mode!r}.')

    # Properties and stuff
    return xparam, param
