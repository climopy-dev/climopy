#!/usr/bin/env python3
"""
Spectral analysis tools. Many of these are adapted
from examples and course notes provided by Professors `Elizabeth Barnes \
<http://barnes.atmos.colostate.edu/COURSES/AT655_S15/lecture_slides.html>`__
and `Dennis Hartmann \
<https://atmos.washington.edu/~dennis/552_Notes_ftp.html>`__.

Note
----
The convention for this package is to use *linear* wave properties, i.e. the
wavelength in <units> per :math:`2\\pi` radians, and wavenumber
:math:`2\\pi` radians per <units>.
"""
# TODO: Support applying 'lanczos' filter with lanczos
import numpy as np
import scipy.signal as signal

from .internals import ic  # noqa: F401
from .internals import docstring, quack, warnings
from .internals.array import _ArrayContext

__all__ = [
    'butterworth',
    'copower',
    'copower2d',
    'filter',
    'harmonics',
    'highpower',
    'lanczos',
    'power',
    'power2d',
    'response',
    'runmean',
    'waves',
    'window',
]

# Input values
_power_params = """
wintype : str or (str, float), optional
    The window specification, passed to `window`. The resulting
    weights are used to window the data before carrying out spectral
    decompositions. See notes for details.
nperseg : int, optional
    The time dimension window or segment length, passed to `window`. If
    ``None``, windowing is not carried out. See notes for details.
detrend : {{'constant', 'linear'}}, optional
    Passed as the `type` argument to `scipy.signal.detrend`. ``'constant'``
    removes the mean and ``'linear'`` removes the linear trend.
"""

_power_data = """
y : array-like
    The input data.
"""

_copower_data = """
y1 : array-like
   First input data.
y2 : array-like
    Second input data. Must have same shape as `y1`.
"""

_power1d_params = """
dx : float or array-like
    Dimension step size in physical units. Used to scale `fx`.
{}
""".rstrip() + _power_params

_power2d_params = """
dx_lon : float or array-like
    Longitude or cyclic dimension step size in physical units. Used to scale `fx_lon`.
dx_time : float or array-like
    Time dimension step size in physical units. Used to scale `fx_time`.
{}
axis_lon : int, optional
    Location of the cyclic "space" axis, generally longitude.
axis_time : int, optional
    Location of the "time" axis.
""".rstrip() + _power_params

docstring.snippets['power.params'] = _power1d_params.format(_power_data.strip())
docstring.snippets['power2d.params'] = _power2d_params.format(_power_data.strip())
docstring.snippets['copower.params'] = _power1d_params.format(_copower_data.strip())
docstring.snippets['copower2d.params'] = _power2d_params.format(_copower_data.strip())


# Return values
_power1d_returns = """
fx : array-like
    Frequencies. Units are the inverse of the `dx` units.
"""

_power2d_returns = """
fx_lon : array-like
    Frequencies for *longitude* or *cyclic* axis. Units are the inverse of `dx_lon`.
fx_time : array-like
    Frequencies for *time* axis. Units are the inverse of the `dx_time`.
"""

_power_returns = """
P : array-like
    Power spectrum. Units are the input units squared.
"""

_copower_returns = """
C : array-like
    Co-power spectrum.
Q : array-like
    Quadrature spectrum.
P1 : array-like
    Power spectrum for the first input data.
P2 : array-like
    Power spectrum for the second input data.
Coh : array-like, optional
    Coherence. Values should range from 0 to 1.
Phi : array-like, optional
    Phase difference in degrees.
"""

docstring.snippets['power.returns'] = _power1d_returns.strip() + _power_returns.rstrip()  # noqa: E501
docstring.snippets['power2d.returns'] = _power2d_returns.strip() + _power_returns.rstrip()  # noqa: E501
docstring.snippets['copower.returns'] = _power1d_returns.strip() + _copower_returns.rstrip()  # noqa: E501
docstring.snippets['copower2d.returns'] = _power2d_returns.strip() + _copower_returns.rstrip()  # noqa: E501

# Notes
docstring.snippets['power.notes'] = """
Notes
-----
The Fourier coefficients are scaled so that total variance is equal to one
half the sum of the right-hand coefficients. This is more natural for the
real-valued datasets typically used by physical scientists, and matches
the convention from Professor Elizabeth Barnes's objective analysis
`course notes \
<http://barnes.atmos.colostate.edu/COURSES/AT655_S15/lecture_slides.html>`__.
This differs from the numpy convention, which scales the coefficients so
that total variance is equal to the sum of squares of all coefficients,
or twice the right-hand coefficients.

Windowing is carried out by applying the `wintype` weights to successive
time segments of length `nperseg` (overlapping by one half the window
length), taking spectral decompositions of each weighted segment, then
taking the average of the result for all segments. Note that non-boxcar
windowing reduces the total power amplitude and results in loss of
information. It may often be preferable to follow the example of
:cite:`1991:randel` and smooth in *frequency* space with a Gaussian filter
after the decomposition has been carried out.

The below example shows that the extent of power reduction resulting from
non-boxcar windowing depends on the character of the signal.

>>> import numpy as np
>>> import climopy as climo
>>> state = np.random.RandomState(51423)
>>> w = climo.window(200, 'hanning')
>>> y1 = np.sin(np.arange(0, 8 * np.pi - 0.01, np.pi / 25)) # basic signal
>>> y2 = state.rand(200) # complex signal
>>> for y in (y1, y2):
...     yvar = y.var()
...     Y = (np.abs(np.fft.fft(y)[1:] / y.size) ** 2).sum()
...     Yw = (np.abs(np.fft.fft(y * w)[1:] / y.size) ** 2).sum()
...     print('Boxcar', Y / yvar)
...     print('Hanning', Yw / yvar)
Boxcar 1.0
Hanning 0.375
Boxcar 0.9999999999999999
Hanning 0.5728391743988162

References
----------
.. bibliography:: ../bibs/power.bib
"""


@quack._pint_wrapper(('=x', '', '=x'), '')
def butterworth(dx, order, cutoff, /, btype='low'):
    """
    Applies Butterworth filter to data. Since this is a *recursive*
    filter, non-trivial to apply, so this uses scipy 'lfilter'.

    To get an 'impulse response function', pass a bunch of zeros with a single
    non-zero 'point' as the dx. See Libby's function for more details.

    Parameters
    ----------
    dx : float
        Data spacing.
    order : int
        Order of the filter.
    cutoff : float
        Cutoff frequency in 'x' units (i.e. *wavelengths*).

    Returns
    -------
    b : array-like
        Numerator coeffs.
    a : array-like
        Denominator coeffs.
    """
    # Initial stuff
    # * Need to run *forward and backward* to prevent time-shifting.
    # * The 'analog' means units of cutoffs are rad/s.
    # * Unlike Lanczos filter, the *length* of this should be
    #   determined always as function of timestep, because really high
    #   order filters can get pretty wonky.
    # * Cutoff is point at which gain reduces to 1/sqrt(2) of the
    #   initial frequency. If doing bandpass, can
    # N = (width/dx)//1 # convert to timestep units
    # N = (N//2)*2 + 1 # odd numbered
    N = order  # or order
    analog = False  # lfilter seems to need digital
    if analog:
        cutoff = 2 * np.pi / (cutoff / dx)  # from wavelengths to rad/steps
    else:
        cutoff = 1.0 / cutoff  # to Hz, or cycles/unit
        cutoff = cutoff * (2 * dx)  # to cycles / (2 timesteps)
    if cutoff > 1:
        raise ValueError(
            'Cuttoff frequency must be in [0, 1]. Remember you pass a cutoff '
            '*wavelength* to this function, not a frequency.'
        )

    # Apply filter
    print(f'Order-{order} Butterworth filter')
    b, a = signal.butter(N - 1, cutoff, btype=btype, analog=analog, output='ba')
    return b, a


@quack._pint_wrapper(('=x', '', '=x'), '')
def lanczos(dx, width, cutoff, /):
    """
    Returns *coefficients* for Lanczos high-pass filter with
    desired wavelength specified. See `this link \
<https://scitools.org.uk/iris/docs/v1.2/examples/graphics/SOI_filtering.html>`__.

    Parameters
    ----------
    dx : float
        Units of your x-dimension, so that cutoff can be translated
        from physical units to 'timestep' units.
    width : float
        Length of filter in time steps.
    cutoff : float
        Cutoff wavelength in physical units.

    Returns
    -------
    b : array-like
        Numerator coeffs.
    a : array-like
        Denominator coeffs.

    Notes
    -----
    * The smoothing should only be *approximate* (see Hartmann notes), response
      function never exactly perfect like with Butterworth filter.
    * The `cutoff` parameter must be provided in *time step units*. Change
      the converter `dx` otherwise.
    * The '2' factor appearing in multiple places may seem random. But this
      converts linear frequency (i.e. wavenumber) to angular frequency in
      sine call below. The '2' doesn't appear in any other factor just as a
      consequence of the math.
    """
    # Coefficients and initial stuff
    # n = (width/dx)//1  # convert window width from 'time units' to 'steps'
    # n = width//2
    # Convert alpha to wavenumber (new units are 'inverse timesteps')
    alpha = 1.0 / (cutoff / dx)
    n = width
    n = (n - 1) // 2 + 1
    tau = np.arange(1, n + 1)  # lag time
    C0 = 2 * alpha  # integral of cutoff-response function is alpha*pi/pi
    Ck = np.sin(2 * np.pi * alpha * tau) / (np.pi * tau)
    Cktilde = Ck * np.sin(np.pi * tau / n) / (np.pi * tau / n)

    # Return filter
    # Example: n = 9 returns 4 + 4 + 1 points
    order = n * 2 - 1
    print(f'Order-{order} Lanczos window')
    window = np.concatenate((np.flipud(Cktilde), np.array([C0]), Cktilde))
    return window[1:-1], 1


@quack._xarray_yy_wrapper
@quack._pint_wrapper(('=x', ''), '=x')
def filter(x, b, /, a=1, n=1, axis=-1, center=True, pad=True, pad_value=np.nan):
    """
    Apply scipy.signal.lfilter to data. By default this does *not* pad
    ends of data. May keep it this way.

    Parameters
    ----------
    x : array-like
        Data to be filtered.
    b : array-like
        *b* coefficients (non-recursive component).
    a : array-like, optional
        *a* coefficients (recursive component). Default of ``1`` indicates
        a non-recursive filter.
    n : int, optional
        Number of times to filter data. Will go forward --> backward --> forward...
    axis : int, optional
        Axis along which data is filtered. Defaults to last axis.
    center : bool, optional
        Whether to trim leading part of axis by number of *b* coefficients. Will
        also attempt to *re-center* the data if a net-forward (e.g. f, fbf, fbfbf, ...)
        filtering was performed. This works for non-recursive filters only.
    pad : bool, optional
        Whether to pad trimmed values with `pad_value` when `center` is ``True``.
    pad_value : float
        The pad value.

    Returns
    -------
    array-like
        Data filtered along axis `axis`.
    """
    # NOTE:
    # * Consider adding empirical method for trimming either side of recursive
    #   filter that trims up to where impulse response is negligible.
    # * If `x` has odd number of obs along axis, lfilter will trim
    #   the last one. Just like `rolling`.
    # * The *a* vector contains (index 0) the scalar used to normalize *b*
    #   coefficients, and (index 1,2,...) the coefficients on `y` in the
    #   filtering conversion from `x` to `y`. So, 1 implies
    #   array of [1, 0, 0...], implies non-recursive.
    # Parse input
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    n_half = (max(len(a), len(b)) - 1) // 2

    # Apply filter 'n' times to each sample
    with _ArrayContext(x, push_right=axis) as context:
        # Take mean
        y_filtered = context.data.copy()
        y_mean = y_filtered.mean(axis=1, keepdims=True)

        # Filter data
        y_filtered -= y_mean  # remove mean
        for i in range(n):  # applications of filter
            step = 1 if i % 2 == 0 else -1  # forward-backward application
            y_filtered[:, ::step] = signal.lfilter(b, a, y_filtered[:, ::step], axis=-1)
        y_filtered += y_mean  # add mean back in

        # Fancy manipulation
        if center:
            # Capture component that (for non-recursive filter) doesn't include points
            # with clipped edges. Forward-backward runs, so filtered data is in correct
            # position w.r.t. x e.g. if n is 2, we cut off the b.size - 1 from each side
            n_2sides = (n // 2) * 2 * n_half

            # Net forward run, so filtered data is shifted right by n_half
            # Also have to trim data on both sides if it's
            # foward-->backward-->forward e.g.
            n_left = int((n % 2) == 1) * 2 * n_half

            # Determine part that 'sees' all coefficients
            if n_2sides == 0:
                y_filtered = y_filtered[:, n_left:]
            else:
                y_filtered = y_filtered[:, (n_2sides + n_left):(-n_2sides)]

        # Optionally pad with a 'fill value' (usually NaN)
        if center and pad:
            y_left = pad_value * np.ones((y_filtered.shape[0], n_2sides + n_left // 2))
            y_right = pad_value * np.ones((y_filtered.shape[0], n_2sides + n_left // 2))
            y_filtered = np.concatenate((y_left, y_filtered, y_right), axis=-1)

        # Replace context data
        context.replace_data(y_filtered)

    # Return
    return context.data


@quack._xarray_yy_wrapper
@quack._pint_wrapper(('=y', ''), '=y')
def harmonics(y, n, /, axis=-1):
    """
    Select the first Fourier harmonics of the time series.
    Useful for example in removing seasonal cycles.

    Parameters
    ----------
    y : array-like
        The data.
    n : int
        The number of harmonics.
    axis : int, optional
        The axis along which harmonics are taken.

    Returns
    -------
    array-like
        The Fourier harmonics.
    """
    # Get fourier transform
    y = np.moveaxis(y, axis, -1)
    fft = np.fft.fft(y, axis=-1)

    # Remove frequencies outside range. The FFT will have some error and give
    # non-zero imaginary components, but we can get magnitude or naively cast to real
    fft[..., 0] = 0
    fft[..., n + 1:-n] = 0
    yf = np.real(np.fft.ifft(fft, axis=-1))
    # yf = np.abs(np.fft.ifft(fft, axis=-1))
    return np.moveaxis(yf, -1, axis)


@quack._xarray_yy_wrapper
@quack._pint_wrapper(('=y', ''), '=y')
def highpower(y, n, /, axis=-1):
    """
    Select only the highest power frequencies. Useful for crudely reducing noise.

    Parameters
    ----------
    y : `numpy.array-like`
        The data.
    n : int
        The integer number of frequencies to select.
    axis : int, optional
        Axis along which the power is computed.

    Returns
    -------
    array-like
        The filtered data.
    """
    # Get transform
    y = np.moveaxis(y, axis, -1)
    fft = np.fft.fft(y, axis=-1)
    p = np.abs(fft) ** 2

    # Naively remove certain frequencies. Use *argpartition* because it's more
    # efficient, will just put -nth element into sorted position, everything
    # after that unsorted but larger (don't need exact order!).
    f = np.argpartition(p, -n, axis=-1)[..., -n:]
    fft_hi = fft[..., f]
    fft[:] = 0.0
    fft[..., f] = fft_hi
    yf = np.real(np.fft.ifft(fft, axis=-1))
    # yf = np.abs(np.fft.ifft(fft, axis=-1))
    return np.moveaxis(yf, -1, axis)


def _fft2d(pm, win, x, /, detrend='constant'):
    """
    Get 2D Fourier decomp and reorder negative frequencies on non-cyclic
    axis so frequencies there are monotonically ascending.
    """
    x = signal.detrend(x, type=detrend, axis=0)  # remove trend or mean from "time"
    x = signal.detrend(x, type='constant', axis=1)  # remove mean from "longitude"

    # Use 1D numpy.fft.rfft (identical)
    # Follows Libby's recipe, where instead real is cosine and imag is
    # sine. Note only need to divide by 2 when conjugates are included.
    # xi = np.fft.rfft(x, axis=1)[:,1:]/x.shape[1]
    # xi = win[:,None]*xi # got a bunch of sines and cosines
    # C = np.fft.rfft(xi.real, axis=0)[1:,:]/x.shape[0]
    # S = np.fft.rfft(xi.imag, axis=0)[1:,:]/x.shape[0]
    # part1 = (C.real + S.imag + 1j * (C.imag - S.real))[::-1, :]
    # part2 = C.real - S.imag + 1j * (-C.imag - S.real)
    # return np.concatenate((part1, part2), axis=0)

    # Use 2D numpy.fft.rfft2
    # NOTE: Read documentation regarding normalization. Default leaves forward
    # transform unnormalized, reverse normalized by 1 / n. The ortho option
    # normalizes both by 1/sqrt(n).
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/routines.fft.html#module-numpy.fft
    # last axis specified should get a *real* transform
    X = np.fft.rfft2(win[:, None] * x, axes=(0, 1))  # last axis gets real transform
    X = X[:, 1:]  # remove the zero-frequency value
    X = X / (x.shape[0] * x.shape[1])  # normalize by sample size
    return np.concatenate((X[pm:, :], X[1:pm + 1, :]), axis=0)


def _window_data(data1, data2, /, nperseg=None, wintype=None):
    """
    Return window properties.
    """
    # Trim if necessary
    ntime = data1.shape[1]
    if nperseg is None:
        nperseg = ntime
    nperseg = 2 * (nperseg // 2)  # enforce even window size
    rem = ntime % nperseg
    if rem != 0:
        data1 = data1[:, :-rem, ...]
        data2 = data2[:, :-rem, ...]
        warnings._warn_climopy(
            f'Trimmed {rem} out of {ntime} points to accommodate '
            f'length-{nperseg} window.'
        )

    # Get window values and *center* indices for window locations
    pm = nperseg // 2
    win = window(nperseg, wintype)
    winloc = np.arange(pm, ntime - pm + pm // 2, pm)  # jump by half window length
    if winloc.size == 0:
        raise ValueError(f'Window length {nperseg} too big for length-{ntime} axis.')
    return win, winloc, data1, data2


def _power_driver(
    dx, y1, y2, /,
    nperseg=None,
    wintype='boxcar',
    center=np.pi,
    axis=-1,
    detrend='constant',
):
    """
    Driver function for 1D spectral estimates.
    """
    # Initial stuff
    dx = quack._get_step(dx)
    copower = y1 is not y2
    if y1.shape != y2.shape:
        raise ValueError(f'Got conflicting shapes for y1 {y1.shape} and y2 {y2.shape}.')

    # Get copsectrum, quadrature spectrum, and powers for each window
    with _ArrayContext(y1, y2, push_right=axis) as context:
        # Get window and flattened, trimmed data
        y1, y2 = context.data
        win, winloc, y1, y2 = _window_data(y1, y2, nperseg=nperseg, wintype=wintype)
        pm = win.size // 2
        nwindows = winloc.size
        nextra, ntime = y1.shape

        # Setup output arrays
        Py1 = np.empty((nextra, nwindows, pm))  # we take the mean of dimension 1
        if copower:
            Py2 = Py1.copy()
            C = Py1.copy()
            Q = Py1.copy()

        # Loop through windows. Remember to double the size of power, because we
        # only have half the coefficients (rfft not fft).
        for k in range(nextra):
            if np.any(~np.isfinite(y1[k, :])) or np.any(~np.isfinite(y2[k, :])):
                warnings._warn_climopy('Skipping array with missing values.')
                continue
            for i, l in enumerate(winloc):
                # Auto approach with scipy.csd. 'one-sided' says to only return first
                # symmetric half if data is real 'scaling' queries whether to:
                #  * Scale 'per wavenumber'/'per Hz', option 'density', default.
                #    This is analagous to a Planck curve with intensity per wavenumber
                #  * Show the power (so units are just data units squared).
                #    This is usually what we want.
                # f, P = signal.csd(y1, y1, window=wintype,
                #   return_onesided=True, scaling=scaling,
                #   nperseg=nperseg, noverlap=nperseg//2, detrend=detrend,
                #   axis=axis
                # )
                # Manual approach
                wy1 = win * signal.detrend(y1[k, l - pm:l + pm], type=detrend)
                Fy1 = np.fft.rfft(wy1)[1:] / win.sum()
                Py1[k, i, :] = np.abs(Fy1) ** 2
                Py1[k, i, :-1] *= 2
                if copower:
                    wy2 = win * signal.detrend(y2[k, l - pm:l + pm], type=detrend)
                    Fy2 = np.fft.rfft(wy2)[1:] / win.sum()
                    Py2[k, i, :] = np.abs(Fy2) ** 2
                    C[k, i, :] = Fy1.real * Fy2.real + Fy1.imag * Fy2.imag
                    Q[k, i, :] = Fy1.real * Fy2.imag - Fy2.real * Fy1.imag
                    Py2[k, i, :-1] *= 2  # scale all but the Nyquist frequency
                    C[k, i, :-1] *= 2
                    Q[k, i, :-1] *= 2

        # Get window averages and output arrays
        # NOTE: For sanity check, ensure (C ** 2 + Q ** 2) / (Py1 * Py2) === 1.
        # NOTE: This Phi relationship is still valid. Check Libby notes. Divide
        # here Q by C and the Ws cancel out, end up with average phase diff.
        f = np.fft.rfftfreq(2 * pm)[1:]  # frequencies
        Py1 = Py1.mean(axis=1)
        arrays = (Py1,)
        if copower:
            Py2 = Py2.mean(axis=1)
            C = C.mean(axis=1)
            Q = Q.mean(axis=1)
            Coh = C ** 2 / (Py1 * Py2)
            Phi = np.arctan2(Q, C)  # phase
            Phi[Phi >= center + np.pi] -= 2 * np.pi
            Phi[Phi < center - np.pi] += 2 * np.pi
            Phi = Phi * 180.0 / np.pi  # convert to degrees!!!
            arrays = (C, Q, Py1, Py2, Coh, Phi)

        # Replace arrays
        context.replace_data(*arrays)

    return (f / dx, *context.data) if copower else (f / dx, context.data)


def _power2d_driver(
    dx_lon, dx_time, y1, y2, /,
    nperseg=None,
    wintype='boxcar',
    center=np.pi,
    axis_lon=-1,
    axis_time=0,
    detrend='constant',
):
    """
    Driver function for 2D spectral estimates.
    """
    # Checks
    dx_lon = quack._get_step(dx_lon)
    dx_time = quack._get_step(dx_time)
    copower = y1 is not y2
    if len(y1.shape) < 2:
        raise ValueError('Need at least rank 2 array.')
    if y1.shape != y2.shape:
        raise ValueError(f'Shapes of y1 {y1.shape} and y2 {y2.shape} must match.')

    # Permute and flatten
    with _ArrayContext(y1, y2, push_right=(axis_time, axis_lon)) as context:
        # Get window and flattened, trimmed data
        y1, y2 = context.data
        win, winloc, y1, y2 = _window_data(y1, y2, nperseg=nperseg, wintype=wintype)
        pm = win.size // 2
        nwindows = winloc.size
        nextra, ntime, ncyclic = y1.shape

        # Setup output arrays
        Py1 = np.nan * np.empty((nextra, nwindows, pm * 2, ncyclic // 2))
        if copower:
            Py2 = Py1.copy()
            C = Py1.copy()
            Q = Py1.copy()

        # 2D transform for each window on non-cyclic dimension
        # Note since we got the rfft (not fft) in one direction, only have half the
        # coefficients (they are symmetric); means for correct variance, have to
        # double th power. These are analagous to Libby's notes for complex space
        for k in range(nextra):
            if (
                np.any(~np.isfinite(y1[k, :, :]))
                or np.any(~np.isfinite(y2[k, :, :]))
            ):
                warnings._warn_climopy('Skipping array with missing values.')
                continue
            for i, idx in enumerate(winloc):
                Fy1 = _fft2d(pm, win, y1[k, idx - pm:idx + pm, :], detrend)
                Py1[k, i, :, :] = np.abs(Fy1) ** 2
                Py1[k, i, :, :-1] *= 2
                if copower:
                    Fy2 = _fft2d(pm, win, y2[k, idx - pm:idx + pm, :], detrend)
                    Py2[k, i, :, :] = np.abs(Fy2) ** 2
                    Phi1 = np.arctan2(Fy1.imag, Fy1.real)
                    Phi2 = np.arctan2(Fy2.imag, Fy2.real)
                    C[k, i, :, :] = np.abs(Fy1) * np.abs(Fy2) * np.cos(Phi1 - Phi2)
                    Q[k, i, :, :] = np.abs(Fy1) * np.abs(Fy2) * np.sin(Phi1 - Phi2)
                    Py2[k, i, :, :-1] *= 2
                    C[k, i, :, :-1] *= 2
                    Q[k, i, :, :-1] *= 2

        # Get output arrays
        # TODO: Why remove mean power?
        # NOTE: This Phi relationship is still valid. Check Libby notes. Divide
        # here Q by C and the Ws cancel out, end up with average phase diff.
        # NOTE: Default order is to go 0 1 ... N/2 -N/2 ... -1. We reorder so
        # frequencies are from -N/2 ... -1 1 ... N/2.
        fx_time = np.fft.fftfreq(2 * pm)
        fq = np.abs(fx_time[pm:pm + 1])  # Nyquist frequency singleton array
        fx_time = np.concatenate((-fq, fx_time[pm + 1:], fx_time[1:pm], fq), axis=0)
        fx_lon = np.fft.rfftfreq(ncyclic)[1:]
        Py1 = Py1.mean(axis=1)
        arrays = (Py1,)
        if copower:
            Py2 = Py2.mean(axis=1)
            C = C.mean(axis=1)
            Q = Q.mean(axis=1)
            Coh = (C ** 2 + Q ** 2) / (Py1 * Py2)
            Phi = np.arctan2(Q, C)  # phase
            Phi[Phi >= center + np.pi] -= 2 * np.pi
            Phi[Phi < center - np.pi] += 2 * np.pi
            arrays = (C, Q, Py1, Py2, Coh, Phi)

        # Replace context data
        context.replace_data(*arrays)

    # Return unflattened data
    if copower:
        return (fx_lon / dx_lon, fx_time / dx_time, *context.data)
    else:
        return (fx_lon / dx_lon, fx_time / dx_time, context.data)


@quack._xarray_power_wrapper
@quack._pint_wrapper(('=x', '=y'), ('=1 / x', '=y ** 2'))
@docstring.add_snippets
def power(dx, y1, /, axis=0, **kwargs):
    """
    Return the spectral decomposition of a real-valued array along an
    arbitrary axis.

    Parameters
    ----------
    %(power.params)s

    Returns
    -------
    %(power.returns)s

    %(power.notes)s

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import climopy as climo
    >>> state = np.random.RandomState(51423)
    >>> ureg = climo.ureg
    >>> x = xr.DataArray(
    ...     np.arange(1000, dtype=float) * ureg.day, dims=('time',), name='time'
    ... )
    >>> y = xr.DataArray(
    ...     state.rand(1000, 50) * ureg.K, dims=('time', 'space'), name='variable'
    ... )
    >>> f, P = climo.power(x, y, axis=0)
    >>> f.head()
    <xarray.DataArray 'time' (f: 5)>
    <Quantity([0.001 0.002 0.003 0.004 0.005], '1 / day')>
    Dimensions without coordinates: f
    Attributes:
        long_name:  frequency
    >>> P.head()
    <xarray.DataArray 'variable' (f: 5, space: 5)>
    <Quantity([[1.18298459e-04 1.09036599e-04 3.76719588e-04 3.36341980e-04
      1.62219721e-04]
     [1.74596838e-04 2.09816475e-04 1.26945159e-04 3.71464785e-04
      4.18144064e-04]
     [6.38233218e-05 1.52714331e-05 8.25923743e-05 3.31492680e-04
      3.06457271e-05]
     [1.71082223e-04 3.95808831e-05 5.23555579e-04 6.15207584e-05
      5.59359839e-05]
     [1.34026670e-04 3.90031444e-05 2.11700762e-04 2.69345283e-05
      6.86978186e-05]], 'kelvin ** 2')>
    Coordinates:
      * f        (f) float64 0.001 0.002 0.003 0.004 0.005
    Dimensions without coordinates: space
    """
    return _power_driver(dx, y1, y1, axis=axis, **kwargs)


@quack._xarray_power_wrapper
@quack._pint_wrapper(
    ('=x', '=y1', '=y2'),
    ('=1 / x', '=y1 * y2', '=y1 * y2', '=y1 ** 2', '=y2 ** 2', '', 'deg'),
)
@docstring.add_snippets
def copower(dx, y1, y2, /, axis=0, **kwargs):
    """
    Return the co-spectral decomposition and related quantities for two
    real-valued arrays along an arbitrary axis.

    Parameters
    ----------
    %(copower.params)s

    Returns
    -------
    %(copower.returns)s

    %(power.notes)s

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import climopy as climo
    >>> state = np.random.RandomState(51423)
    >>> ureg = climo.ureg
    >>> x = xr.DataArray(
    ...     np.arange(1000, dtype=float) * ureg.day, dims=('time',), name='time'
    ... )
    >>> y1 = xr.DataArray(
    ...     state.rand(1000, 50) * ureg.K, dims=('time', 'space'), name='variable'
    ... )
    >>> y2 = xr.DataArray(
    ...     state.rand(1000, 50) * ureg.m / ureg.s,
    ...     dims=('time', 'space'), name='variable',
    ... )
    >>> f, C, Q, P1, P2, Coh, Phi = climo.copower(x, y1, y2, axis=0)
    >>> f.head()
    <xarray.DataArray 'time' (f: 5)>
    <Quantity([0.001 0.002 0.003 0.004 0.005], '1 / day')>
    Dimensions without coordinates: f
    Attributes:
        long_name:  frequency
    >>> C.head()
    <xarray.DataArray 'variable' (f: 5, space: 5)>
    <Quantity([[-1.55726091e-04 -2.51099398e-04 -4.51984603e-05  3.38682324e-04
       1.02699316e-04]
     [ 6.03921637e-05  1.84940350e-04 -6.98685563e-06  1.25120193e-04
      -5.62898627e-04]
     [ 2.89041840e-05  1.62381854e-05  5.51445757e-05 -2.75903456e-04
      -1.26185549e-05]
     [ 5.04572726e-05 -6.94459577e-05  2.43580083e-04 -4.08724137e-05
       6.86952412e-05]
     [-2.18000152e-04  1.43110211e-05 -2.26535472e-04 -3.73436882e-05
      -1.28860163e-04]], 'kelvin * meter / second')>
    Coordinates:
      * f        (f) float64 0.001 0.002 0.003 0.004 0.005
    Dimensions without coordinates: space
    >>> Q.head()
    <xarray.DataArray 'variable' (f: 5, space: 5)>
    <Quantity([[ 3.39059554e-05 -5.49872491e-05  3.83785647e-05  1.03330854e-04
      -9.02664704e-06]
     [ 1.65124562e-05 -4.40473709e-05 -5.06947002e-07  1.96874623e-04
      -1.18780187e-04]
     [-1.09859742e-04  2.31464562e-05 -4.98439846e-05  5.67827280e-05
       3.39652860e-05]
     [ 1.22685839e-04 -6.57993024e-05  2.23153696e-04 -9.57458276e-06
       1.21946552e-06]
     [-2.82846001e-05  4.62368401e-05  1.54712600e-05 -6.94771696e-05
      -2.77645650e-05]], 'kelvin * meter / second')>
    Coordinates:
      * f        (f) float64 0.001 0.002 0.003 0.004 0.005
    Dimensions without coordinates: space
    """
    return _power_driver(dx, y1, y2, axis=axis, **kwargs)


@quack._xarray_power2d_wrapper
@quack._pint_wrapper(('=x1', '=x2', '=y'), ('=1 / x1', '=1 / x2', '=y ** 2'))
@docstring.add_snippets
def power2d(dx_lon, dx_time, y, /, axis_lon=-1, axis_time=0, **kwargs):
    """
    Return the spectral decomposition of a real-valued array along an
    arbitrary axis.

    Parameters
    ----------
    %(power2d.params)s

    Returns
    -------
    %(power2d.returns)s

    %(power.notes)s
    """
    return _power2d_driver(
        dx_lon, dx_time, y, axis_lon=axis_lon, axis_time=axis_time, **kwargs
    )


@quack._xarray_power2d_wrapper
@quack._pint_wrapper(
    ('=x1', '=x2', '=y1', '=y2'),
    ('=1 / x1', '=1 / x2', '=y1 * y2', '=y1 * y2', '=y1 ** 2', '=y2 ** 2', '', 'deg'),
)
@docstring.add_snippets
def copower2d(dx_lon, dx_time, y1, y2, /, axis_lon=0, axis_time=-1, **kwargs):
    """
    Return the 2D spectral decomposition of two real-valued arrays with
    along an arbitrary *time* dimension and *cyclic* dimension.
    For details, see :cite:`1991:randel`.

    Parameters
    ----------
    %(copower2d.params)s

    Returns
    -------
    %(copower2d.returns)s

    %(power.notes)s
    """
    return _power2d_driver(
        dx_lon, dx_time, y1, y2, axis_lon=axis_lon, axis_time=axis_time, **kwargs
    )


@quack._pint_wrapper(('=x', '', ''), ('', ''))
def response(dx, b, a=1, /, n=1000, simple=False):
    """
    Calculate the response function given the *a* and *b* coefficients for some
    analog filter. For details, see Dennis Hartmann's objective analysis
    `course notes <https://atmos.washington.edu/~dennis/552_Notes_ftp.html>`__.

    Parameters
    ----------
    dx : int or array-like
        The step size.
    b : array-like
        The window `b` coefficients.
    a : array-like
        The window `a` coefficients.

    Notes
    -----
    The response function formula can be depicted as follows:

    .. code-block::

                     jw               -jw               -jmw
            jw   B(e)     b[0] + b[1]e    + .... + b[m]e
        H(e)   = ------ = ----------------------------------
                     jw               -jw               -jnw
                 A(e)     a[0] + a[1]e    + .... + a[n]e

    """
    # Parse input
    a = np.atleast_1d(a)
    x = np.linspace(0, np.pi, n)

    # Simple calculation given 'b' coefficients, from Libby's notes
    # Note we *need to make the exponent frequencies into
    # rad/physical units* for results to make sense.
    if simple:
        if not len(a) == 1 and a[0] == 1:
            raise ValueError(
                'Cannot manually calculate response function '
                'for recursive filter.'
            )
        if len(b) % 2 == 0:
            raise ValueError(
                'Filter coefficient number should be odd, symmetric '
                'about a central value.'
            )
        nb = len(b)
        C0 = b[nb // 2]
        Ck = b[nb // 2 + 1:]  # should be symmetric
        tau = np.arange(1, nb // 2 + 1)  # lag time, up to nb+1
        x = x * 2 * np.pi * dx  # from cycles/unit --> rad/unit --> rad/step
        y = C0 + 2 * np.sum(
            Ck[None, :] * np.cos(tau[None, :] * x[:, None]), axis=1
        )

    # More complex freqz filter, generalized for arbitrary recursive filters,
    # with extra allowance for working with physical units.
    # Last entry will be Nyquist, i.e. 1/(dx*2)
    else:
        _, y = signal.freqz(b, a, x)
        x = x / (2 * np.pi * dx)
        y = np.abs(y)
    return x, y


@quack._xarray_yy_wrapper
@quack._pint_wrapper(('=y', ''), '=y')
def runmean(y, n, /, wintype='boxcar', axis=-1, pad=np.nan):
    """
    Apply running average to array.

    Parameters
    ----------
    y : array-like
        Data, and we roll along axis `axis`.
    n : int, optional
        Window length. Passed to `window`.
    wintype : int or array-like
        Window type. Passed to `window`.
    axis : int, optional
        Axis to filter.
    pad : bool, optional
        The pad value used to fill the array back to its original size.
        Set to `None` to disable padding.

    Returns
    -------
    y : array-like
        Data windowed along axis `axis`.

    Notes
    -----
    Implementation is similar to `scipy.signal.lfilter`. Read
    `this post <https://stackoverflow.com/a/4947453/4970632>`__. This creates *view*
    of original array, without duplicating data, so very efficient approach.
    """
    # * For 1-D data numpy `convolve` would be appropriate, problem is `convolve`
    #   doesn't take multidimensional input!
    # * If `y` has odd number of obs along axis, result will have last element
    #   trimmed. Just like `filter`.
    # * Strides are apparently the 'number of bytes' one has to skip in memory
    #   to move to next position *on the given axis*. For example, a 5 by 5
    #   array of 64bit (8byte) values will have array.strides == (40, 8).
    # Roll axis, reshape, and get generate running dimension
    if axis < 0:
        axis += y.ndim
    y = np.moveaxis(y, axis, -1)
    w = window(n, wintype)

    # Manipulate array
    shape = y.shape[:-1] + (y.shape[-1] - (n - 1), n)
    strides = (*y.strides, y.strides[-1])  # repeat striding on end
    yr = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    yr = (yr * w).mean(axis=-1)  # broadcasts to right

    # Optionally fill the rows taken up
    # NOTE: Data might be shifted left by one if had even numbered window
    if pad is not None:
        n1 = (y.shape[-1] - yr.shape[-1]) // 2
        n2 = (y.shape[-1] - yr.shape[-1]) - n1
        y1 = pad * np.ones((*yr.shape[:-1], n1))
        y2 = pad * np.ones((*yr.shape[:-1], n2))
        yr = np.concatenate((y1, yr, y2), axis=-1)

    return np.moveaxis(yr, -1, axis)


@quack._pint_wrapper('=x', '=x')
def waves(x, /, wavenums=None, wavelengths=None, phase=None, state=None):
    """
    Compose array of sine waves. Useful for testing the performance of filters.

    Parameters
    ----------
    x : array-like
        If scalar, *x* is ``np.arange(0, x)``. If iterable, can be n-dimensional,
        and will calculate sine from coordinates on every dimension.
    wavelengths : float
        Wavelengths for sine function. Required if `wavenums` is ``None``.
    wavenums : float
        Wavenumbers for sine function. Required if `wavelengths` is ``None``.
    phase : float, optional
        Array of phase offsets.
    state : `numpy.RandomState`, optional
        The random state to use for generating the data.

    Returns
    -------
    data : array-like
        Data composed of sine waves.

    Notes
    -----
    `x` will always be normalized so that wavelength is with reference to
    the first step. This make sense because when working with filters, for
    which we almost always need to use units corresponding to the axis.
    """
    # Wavelengths
    if wavenums is None and wavelengths is None:
        raise ValueError('Must declare wavenums or wavelengths.')
    elif wavelengths is not None:
        wavenums = 1.0 / np.atleast_1d(wavelengths)
    wavenums = np.atleast_1d(wavenums)
    if np.isscalar(x):
        x = np.arange(x)
    data = np.zeros(x.shape)  # user can make N-D array

    # Get waves
    if state is None:
        state = np.random
    if phase is None:
        phis = state.uniform(0, 2 * np.pi, len(wavenums))
    else:
        phis = phase * np.ones((len(wavenums),))
    for wavenum, phi in zip(wavenums, phis):
        data += np.sin(2 * np.pi * wavenum * x + phi)
    return data


def window(n, /, wintype='boxcar'):
    """
    Retrieve the `~scipy.signal.get_window` weighting function window.

    Parameters
    ----------
    n : int
        The window length.
    wintype : str or (str, float, ...) tuple
        The window name or ``(name, param1, ...)`` tuple containing the
        name and the required parameter(s).
    normalize : bool, optional
        Whether to divide the resulting coefficients by their sum.

    Returns
    -------
    win : array-like
        The window coefficients.
    """
    if wintype == 'welch':
        raise ValueError('Welch window needs 2-tuple of (name, beta).')
    elif wintype == 'kaiser':
        raise ValueError('Kaiser window needs 2-tuple of (name, beta).')
    elif wintype == 'gaussian':
        raise ValueError('Gaussian window needs 2-tuple of (name, stdev).')
    else:
        w = signal.get_window(wintype, n)
    # if normalize:
    #     w /= np.sum(w)
    return w
