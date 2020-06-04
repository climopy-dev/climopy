#!/usr/bin/env python3
"""
Spectral analysis tools. Many of these are adapted
from examples and course notes provided by `Elizabeth Barnes \
<http://barnes.atmos.colostate.edu/COURSES/AT655_S15/lecture_slides.html>`__
and `Dennis Hartmann \
<https://atmos.washington.edu/~dennis/552_Notes_ftp.html>`__.

Note
----
The convention for this package is to use *linear* wave properties, i.e. the
wavelength in <units> per :math:`2\\pi` radians, and wavenumber
:math:`2\\pi` radians per <units>.
"""
import numpy as np
import scipy.signal as signal
# from .internals import quack
from .internals import docstring, warnings
from .internals.array import _ArrayContext

__all__ = [
    'autopower',
    'autopower2d',
    'butterworth',
    'filter',
    'harmonics',
    'highpower',
    'impulse',
    'lanczos',
    'power',
    'power2d',
    'response',
    'rolling',
    'running',
    'waves',
    'window',
]


def autopower():
    """
    Wrapper around `power` that generates co-spectral
    statistics and whatnot at *successive lags*.

    Warning
    -------
    Not yet implemented.
    """
    # Uses scipy.signal.welch windowing method to generate an estimate of the
    # *lagged* spectrum. Can also optionally do this with two variables.
    raise NotImplementedError


def autopower2d():
    """
    Wrapper around `power2d` that generates co-spectral
    statistics and whatnot at *successive lags*.

    Warning
    -------
    Not yet implemented.
    """
    raise NotImplementedError


def butterworth(dx, order, cutoff, btype='low'):
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

    Note
    ----
    * Need to run *forward and backward* to prevent time-shifting.
    * The 'analog' means units of cutoffs are rad/s.
    * Unlike Lanczos filter, the *length* of this should be
      determined always as function of timestep, because really high
      order filters can get pretty wonky.
    * Cutoff is point at which gain reduces to 1/sqrt(2) of the
      initial frequency. If doing bandpass, can
    """
    # Initial stuff
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
    print(f'Order-{N:.0f} Butterworth filter')
    b, a = signal.butter(
        N - 1, cutoff, btype=btype, analog=analog, output='ba'
    )
    return b, a


def filter(x, b, a=1, n=1, axis=-1, fix=True, pad=True, pad_value=np.nan):
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
        *a* coefficients (recursive component); default of ``1`` indicates
        a non-recursive filter.
    n : int, optional
        Number of times to filter data. Will go forward --> backward
        --> forward...
    axis : int, optional
        Axis along which we filter data. Defaults to last axis.
    fix : bool, optional
        Whether to trim leading part of axis by number of *b* coefficients. Will
        also attempt to *re-center* the data if a net-forward (e.g. f, fbf, fbfbf, ...)
        filtering was performed. This works for non-recursive filters only.
    pad : bool, optional
        Whether to pad trimmed values (when `fix` is ``True``) with `pad_value`.
    pad_value : float
        The pad value.

    Returns
    -------
    y : array-like
        Data filtered along axis `axis`.

    Note
    ----
    * Consider adding empirical method for trimming either side of recursive
      filter that trims up to where impulse response is negligible.
    * If `x` has odd number of obs along axis, lfilter will trim
      the last one. Just like `rolling`.
    * The *a* vector contains (index 0) the scalar used to normalize *b*
      coefficients, and (index 1,2,...) the coefficients on `y` in the
      filtering conversion from `x` to `y`. So, 1 implies
      array of [1, 0, 0...], implies non-recursive.
    """
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
        if fix:
            # Capture component that (for non-recursive filter) doesn't include
            # datapoints with clipped edges. Forward-backward runs, so filtered
            # data is in correct position w.r.t. x e.g. if n is 2, we cut off
            # the (len(b)-1) from each side.
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
        if pad:
            y_left = pad_value * np.ones((y_filtered.shape[0], n_2sides + n_left // 2))
            y_right = pad_value * np.ones((y_filtered.shape[0], n_2sides + n_left // 2))
            y_filtered = np.concatenate((y_left, y_filtered, y_right), axis=-1)

        # Replace context data
        context.replace_data(y_filtered)

    # Return
    return context.data


def harmonics(x, k=4, axis=-1, absval=False):
    """
    Select the first `k` Fourier harmonics of the time series. Useful
    for example in removing seasonal cycle or something.
    """
    # Get fourier transform
    x = np.moveaxis(x, axis, -1)
    fft = np.fft.fft(x, axis=-1)

    # Remove frequencies outside range. The FFT will have some error and give
    # non-zero imaginary components, but we can get magnitude or naively cast
    # to real
    fft[..., 0] = 0
    fft[..., k + 1:-k] = 0
    if absval:
        y = np.moveaxis(np.abs(np.fft.ifft(fft)), -1, axis)
    else:
        y = np.moveaxis(np.real(np.fft.ifft(fft)), -1, axis)
    return y


def highpower(x, n, axis=-1):
    """
    Select only the highest power frequencies. Useful for
    crudely reducing noise.

    Parameters
    ----------
    x : `numpy.array-like`
        The data.
    n : int
        The integer number of frequencies to select.
    """
    # Naively remove certain frequencies
    # Should ignore first coefficient, the mean
    x = np.moveaxis(x, axis, -1)
    fft = np.fft.fft(x, axis=-1)
    fftfreqs = np.arange(1, fft.shape[-1] // 2)  # up to fft.size/2 - 1

    # Get indices of n largest values. Use *argpartition* because it's more
    # efficient, will just put -nth element into sorted position, everything
    # after that unsorted but larger (don't need exact order!).
    p = np.abs(fft) ** 2
    f = np.argpartition(p, -n, axis=-1)[..., -n:]
    y = fft.copy()
    y[...] = 0  # fill in
    y[..., f] = fft[..., f]  # put back the high-power frequencies
    freqs = fftfreqs[..., f]
    return freqs, y  # frequencies and the high-power filter


def impulse():
    """
    Get the *impulse* response function for a recursive filter.

    Warning
    -------
    Not yet implemented.
    """
    # R2_q = 1./(1. + (omega/omega_c)**(2*N))
    raise NotImplementedError


def lanczos(dx, width, cutoff):
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

    Note
    ----
    * The smoothing should only be *approximate* (see Hartmann notes), response
      function never exactly perfect like with Butterworth filter.
    * The `cutoff` parameter must be provided in *time step units*. Change
      the converter `dx` otherwise.
    * The '2' factor appearing in multiple places may seem random. But this
      converts linear frequency (i.e. wavenumber) to angular frequency in
      sine call below. The '2' doesn't appear in any other factor just as a
      consequence of the math.

    Example
    -------
    n=9 returns 4+4+1=9 points in the 'concatenate' below.
    """
    # Coefficients and initial stuff
    # n = (width/dx)//1  # convert window width from 'time units' to 'steps'
    # n = width//2
    # convert alpha to wavenumber (new units are 'inverse timesteps')
    alpha = 1.0 / (cutoff / dx)
    n = width
    n = (n - 1) // 2 + 1
    print(f'Order-{n*2 - 1:.0f} Lanczos window')
    tau = np.arange(1, n + 1)  # lag time
    C0 = 2 * alpha  # integral of cutoff-response function is alpha*pi/pi
    Ck = np.sin(2 * np.pi * alpha * tau) / (np.pi * tau)
    Cktilde = Ck * np.sin(np.pi * tau / n) / (np.pi * tau / n)

    # Return filter
    window = np.concatenate((np.flipud(Cktilde), np.array([C0]), Cktilde))
    return window[1:-1], 1


@docstring.add_snippets
def power(
    y1,
    y2=None,
    dx=1,
    cyclic=False,
    coherence=False,
    nperseg=100,
    wintype='boxcar',
    center=np.pi,
    axis=-1,
    detrend='constant',
):
    """
    Return the spectral decomposition of a real-valued dataset along an
    arbitrary axis with arbitrary windowing behavior.

    Parameters
    ----------
    y1 : array-like
        Input data.
    y2 : array-like, default ``None``
        Second input data, if cross-spectrum is desired.
        Must have same shape as `y1`.
    dx : float, optional
        Time dimension step size in physical units. Used to scale `f`.
    cyclic : bool, optional
        Whether data is cyclic along axis. If ``True``, the *nperseg*
        argument will be overridden
    %(power.params)s

    Returns
    -------
    f : array-like
        Frequencies in units <x units>**-1. Scaled with `dx`.
    P : array-like, optional
        Power spectrum in units <data units>**2.
        Returned if `z2` is ``None``.
    P, Q, Pz1, Pz2 : array-like, optional
        Co-power spectrum, quadrature spectrum, power spectrum for `z1`, and
        power spectrum for `z2`, respectively.
        Returned if `z2` is not ``None`` and `coherence` is ``False``.
    Coh, Phi : array-like, optional
        Coherence and phase difference, respectively.
        Returned if `z2` is not ``None`` and `coherence` is ``True``.

    Note
    ----
    %(power.notes)s

    References
    ----------
    %(power.bibliography)s
    """
    # Initial stuff
    copower = y2 is not None
    N = y1.shape[axis]  # window count
    if cyclic:
        wintype = 'boxcar'
        nperseg = N
    if copower and y2.shape != y1.shape:
        raise ValueError(f'Got conflicting shapes for y1 {y1.shape} and y2 {y2.shape}.')
    nperseg = 2 * (nperseg // 2)  # enforce even window size

    # Trim if necessary
    rem = N % nperseg
    if rem != 0:
        ndim = y1.ndim
        slices = [slice(None, -rem) if i == axis else slice(None) for i in range(ndim)]
        y1 = y1[tuple(slices)]
        if copower:
            y2 = y2[tuple(slices)]
        warnings._warn_climpy(
            f'Trimmed {rem} out of {N} points to accommodate length-{nperseg} window.'
        )

    # Auto approach with scipy.csd. 'one-sided' says to only return first symmetric
    # half if data is real 'scaling' queries whether to:
    #  * Scale 'per wavenumber'/'per Hz', option 'density', default.
    #    This is analagous to a Planck curve with intensity per wavenumber
    #  * Show the power (so units are just data units squared).
    #    This is usually what we want.
    # if not manual and not copower:
    #     f, P = signal.csd(y1, y1, window=wintype,
    #       return_onesided=True, scaling=scaling,
    #       nperseg=nperseg, noverlap=nperseg//2, detrend=detrend,
    #       axis=axis
    #     )

    # Manual approach (have compared to auto and results are identical)
    # Get copsectrum, quadrature spectrum, and powers for each window
    # shape is shape of *original* data
    pm = nperseg // 2
    args = (y1, y2) if copower else (y1,)
    with _ArrayContext(*args, push_right=axis) as context:
        # Get flattened data
        if copower:
            y1, y2 = context.data
        else:
            y1 = context.data

        # List of *center* indices for windows
        K, N = y1.shape[1]
        win = window(wintype, nperseg)
        winloc = np.arange(pm, N - pm + pm // 2, pm)  # jump by half window length
        nwindows = winloc.size
        if nwindows == 0:
            raise ValueError(f'Window length {nperseg} too big for size-{N} dimension.')

        # Setup output arrays
        Py1 = np.empty((K, nwindows, pm))  # we take the mean of dimension 1
        if copower:
            Py2 = Py1.copy()
            C = Py1.copy()
            Q = Py1.copy()

        # Loop through windows. Remember to double the size of power, because we
        # only have half the coefficients (rfft not fft).
        for k in range(K):
            if (
                np.any(~np.isfinite(y1[k, :]))
                or copower and np.any(~np.isfinite(y2[k, :]))
            ):
                warnings._warn_climpy('Skipping array with missing values.')
                continue
            for i, l in enumerate(winloc):
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

        # Get window averages
        f = np.fft.rfftfreq(nperseg)[1:]  # frequencies
        Py1 = Py1.mean(axis=1)
        if copower:
            Py2 = Py2.mean(axis=1)
            C = C.mean(axis=1)
            Q = Q.mean(axis=1)

        # Get output arrays
        if not copower:
            arrays = (Py1,)
        elif not coherence:
            arrays = (C, Q, Py1, Py2)
        else:
            # Coherence and phase difference
            Coh = (C ** 2 + Q ** 2) / (Py1 * Py2)
            Phi = np.arctan2(Q, C)  # phase
            Phi[Phi >= center + np.pi] -= 2 * np.pi
            Phi[Phi < center - np.pi] += 2 * np.pi
            Phi = Phi * 180 / np.pi  # convert to degrees!!!
            arrays = (Coh, Phi)

        # Replace arrays
        context.replace_data(*arrays)

    if copower:
        return (f / dx, *context.data)
    else:
        return (f / dx, context.data)


def _decomp_2d(pm, win, x, detrend='constant'):
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
    # return np.concatenate(
    #     (
    #          (C.real + S.imag + 1j * (C.imag - S.real))[::-1, :],
    #          C.real - S.imag + 1j * (-C.imag - S.real),
    #     ),
    #     axis=0,
    # )
    # Use 2D numpy.fft.rfft2
    # NOTE: Read documentation regarding normalization. Default leaves
    # forward transform unnormalized, reverse normalized by 1 / n. The ortho
    # option normalizes both by 1/sqrt(n).
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/routines.fft.html#module-numpy.fft
    # last axis specified should get a *real* transform
    X = np.fft.rfft2(win[:, None] * x, axes=(0, 1))  # last axis gets real transform
    X = X[:, 1:]  # remove the zero-frequency value
    X = X / (x.shape[0] * x.shape[1])  # normalize by sample size
    return np.concatenate((X[pm:, :], X[1:pm + 1, :]), axis=0)


@docstring.add_snippets
def power2d(
    z1,
    z2=None,
    dx=1,
    dy=1,
    coherence=False,
    nperseg=None,
    wintype='boxcar',
    center=np.pi,
    axes=(-2, -1),
    detrend='constant',
):
    """
    Return the 2-d spectral decomposition of a real-valued dataset with
    one *time* dimension and one *cyclic* dimension along arbitrary axes with
    arbitrary windowing behavior. For details, see :cite:`1991:randel`.

    Parameters
    ----------
    z1 : array-like
        Input data.
    z2 : array-like, default ``None``
        Second input data, if cross-spectrum is desired. Must have same
        shape as `z1`.
    dx : float, optional
        Time dimension step size in physical units. Used to scale `fx`.
    dy : float, optional
        Cyclic dimension step size in physical units. Used to scale `fy`.
    axes : (int, int), optional
        Locations of the "time" and "cyclic" axes, respectively.
        By default the second-to-last and last axes are used.
    %(power.params)s

    Returns
    -------
    fx : array-like
        Time dimension frequencies in units <x units>**-1. Scaled with `dx`.
    fy : array-like
        Cyclic dimension wavenumbers in units <y units>**-1. Scaled with `dy`.
    P : array-like, optional
        Power spectrum in units <data units>**2.
        Returned if `z2` is ``None``.
    P, Q, Pz1, Pz2 : array-like, optional
        Co-power spectrum, quadrature spectrum, power spectrum for `z1`, and
        power spectrum for `z2`, respectively.
        Returned if `z2` is not ``None`` and `coherence` is ``False``.
    Coh, Phi : array-like, optional
        Coherence and phase difference, respectively.
        Returned if `z2` is not ``None`` and `coherence` is ``True``.

    Note
    ----
    %(power.notes)s

    References
    ----------
    %(power.bibliography)s
    """
    # Checks
    copower = z2 is not None
    if len(z1.shape) < 2:
        raise ValueError('Need at least rank 2 array.')
    if copower and not z1.shape == z2.shape:
        raise ValueError(f'Shapes of z1 {z1.shape} and z2 {z2.shape} must match.')
    taxis, caxis = axes
    if caxis < 0:
        caxis += z1.ndim
    if taxis < 0:
        taxis += z1.ndim
    nperseg = 2 * (nperseg // 2)  # enforce even window size

    # Trim data to fit window length
    print(f'Cyclic dim ({caxis}): Length {z1.shape[caxis]}.')
    print(f'Windowed dim ({taxis}): Length {z1.shape[taxis]}, window length {nperseg}.')
    N = z1.shape[taxis]
    rem = N % nperseg
    if rem > 0:
        ndim = z1.ndim
        slices = [slice(None, -rem) if i == taxis else slice(None) for i in range(ndim)]
        z1 = z1[tuple(slices)]
        if copower:
            z2 = z2[tuple(slices)]
        warnings._warn_climpy(
            f'Trimmed {rem} out of {N} points to accommodate length-{nperseg} window.'
        )

    # Permute and flatten
    args = (z1, z2) if copower else (z1,)
    with _ArrayContext(*args, push_right=(taxis, caxis)) as context:
        # Get flattened data
        if copower:
            z1, z2 = context.data
        else:
            z1 = context.data

        # The window *centers* for time windowing. Jump by half window length
        K, N, M = z1.shape
        pm = nperseg // 2
        win = window(wintype, nperseg)  # for time domain
        winloc = np.arange(pm, N - pm + 0.1, pm).astype(int)
        nwindows = winloc.size
        if nwindows == 0:
            raise ValueError('Window length too big.')

        # Setup output arrays
        Pz1 = np.nan * np.empty((K, nwindows, 2 * pm, M // 2))  # power array
        if copower:
            Pz2 = Pz1.copy()
            C = Pz1.copy()
            Q = Pz1.copy()

        # 2D transform for each window on non-cyclic dimension
        # Note since we got the rfft (not fft) in one direction, only have half the
        # coefficients (they are symmetric); means for correct variance, have to
        # double the power. These are analagous to Libby's notes for complex space
        for k in range(K):
            if (
                np.any(~np.isfinite(z1[k, :, :]))
                or copower and np.any(~np.isfinite(z2[k, :, :]))
            ):
                warnings._warn_climpy('Skipping array with missing values.')
                continue
            for i, idx in enumerate(winloc):
                Fz1 = _decomp_2d(pm, win, z1[k, idx - pm:idx + pm, :], detrend)
                Pz1[k, i, :, :] = np.abs(Fz1) ** 2
                Pz1[k, i, :, :-1] *= 2
                if copower:
                    Fz2 = _decomp_2d(pm, win, z2[k, idx - pm:idx + pm, :], detrend)
                    Pz2[k, i, :, :] = np.abs(Fz2) ** 2
                    Phi1 = np.arctan2(Fz1.imag, Fz1.real)
                    Phi2 = np.arctan2(Fz2.imag, Fz2.real)
                    C[k, i, :, :] = np.abs(Fz1) * np.abs(Fz2) * np.cos(Phi1 - Phi2)
                    Q[k, i, :, :] = np.abs(Fz1) * np.abs(Fz2) * np.sin(Phi1 - Phi2)
                    Pz2[k, i, :, :-1] *= 2
                    C[k, i, :, :-1] *= 2
                    Q[k, i, :, :-1] *= 2

        # Output frequencies.
        # TODO: Why remove mean power?
        # NOTE: Default order is to go 0 1 ... N/2 -N/2 ... -1. We reorder so
        # frequencies are from -N/2 ... -1 1 ... N/2.
        fy = np.fft.rfftfreq(M)[1:]
        fx = np.fft.fftfreq(2 * pm)  # start with the positive Fourier coefficients
        fq = np.abs(fx[pm:pm + 1])  # Nyquist frequency singleton array
        fx = np.concatenate((-fq, fx[pm + 1:], fx[1:pm], fq), axis=0)

        # Take average along windows
        Pz1 = Pz1.mean(axis=1)
        if copower:
            Pz2 = Pz2.mean(axis=1)
            C = C.mean(axis=1)
            Q = Q.mean(axis=1)

        # Get output arrays
        if not copower:
            arrays = (Pz1,)
        elif not coherence:
            arrays = (C, Q, Pz1, Pz2)
        else:
            # Get coherence and phase difference
            # NOTE: This Phi relationship is still valid. Check Libby notes. Divide
            # here Q by C and the Ws cancel out, end up with average phase diff.
            Coh = (C ** 2 + Q ** 2) / (Pz1 * Pz2)
            Phi = np.arctan2(Q, C)  # phase
            Phi[Phi >= center + np.pi] -= 2 * np.pi
            Phi[Phi < center - np.pi] += 2 * np.pi
            arrays = (Coh, Phi)

        # Replace context data
        context.replace_data(*arrays)

    # Return unflattened data
    if copower:
        return (fx / dx, fy / dy, *context.data)
    else:
        return (fx / dx, fy / dy, context.data)


def response(dx, b, a=1, n=1000, simple=False):
    """
    Calculate the response function given the *a* and *b* coefficients for some
    analog filter. For details, see Dennis Hartmann's objective analysis
    `course notes <https://atmos.washington.edu/~dennis/552_Notes_ftp.html>`__.

    Note
    ----
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


def rolling(*args, **kwargs):
    """
    Alias for `running`.
    """
    return running(*args, **kwargs)


def running(x, w, axis=-1, pad=True, pad_value=np.nan):
    """
    Apply running average to array.

    Parameters
    ----------
    x : array-like
        Data, and we roll along axis `axis`.
    w : int or array-like
        Boxcar window length, or custom weights.
    axis : int, optional
        Axis to filter.
    pad : bool, optional
        Whether to pad the edges of axis back to original size.
    pad_value : float, optional
        The pad value.

    Returns
    -------
    x : array-like
        Data windowed along axis `axis`.

    Note
    ----
    Implementation is similar to `scipy.signal.lfilter`. Read
    `this post <https://stackoverflow.com/a/4947453/4970632>`__.

    Generates rolling numpy window along final axis. Can then operate with
    functions like polyfit or mean along the new last axis of output.
    Note this creates *view* of original array, without duplicating data, so
    no worries about efficiency.

    * For 1-D data numpy `convolve` would be appropriate, problem is `convolve`
      doesn't take multidimensional input!
    * If `x` has odd number of obs along axis, result will have last element
      trimmed. Just like `filter`.
    * Strides are apparently the 'number of bytes' one has to skip in memory
      to move to next position *on the given axis*. For example, a 5 by 5
      array of 64bit (8byte) values will have array.strides == (40,8).
    """
    # Roll axis, reshape, and get generate running dimension
    n_orig = x.shape[axis]
    if axis < 0:
        axis += x.ndim
    x = np.moveaxis(x, axis, -1)

    # Determine weights
    if isinstance(w, str):
        raise NotImplementedError("Need to allow string 'w' argument, e.g. w='Lanczos'")
    w = np.atleast_1d(w)
    if len(w) == 1:
        # Boxcar window
        nw = w[0]
        w = 1 / nw
    else:
        # Arbitrary windowing function
        # TODO: Add windowing functions!
        nw = len(w)

    # Manipulate array
    shape = x.shape[:-1] + (x.shape[-1] - (nw - 1), nw)
    strides = (*x.strides, x.strides[-1])  # repeat striding on end
    x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Next 'put back' axis, keep axis as
    # the 'rolling' dimension which can be averaged by arbitrary weights.
    x = np.moveaxis(x, -2, axis)  # want to 'put back' axis -2;

    # Finally take the weighted average
    # Note numpy will broadcast from right, so weights can be scalar or not
    # print(x.min(), x.max(), x.mean())
    x = (x * w).sum(axis=-1)  # broadcasts to right

    # Optionally fill the rows taken up
    if pad:
        n_new = x.shape[axis]
        n_left = (n_orig - n_new) // 2
        n_right = n_orig - n_new - n_left
        if n_left != n_right:
            warnings._warn_climpy('Data shifted left by one.')
        d_left = pad_value * np.ones((*x.shape[:axis], n_left, *x.shape[axis + 1:]))
        d_right = pad_value * np.ones((*x.shape[:axis], n_right, *x.shape[axis + 1:]))
        x = np.concatenate((d_left, x, d_right), axis=axis)
    return x


def waves(x, wavenums=None, wavelens=None, phase=None):
    """
    Compose array of sine waves.
    Useful for testing performance of filters.

    Parameters
    ----------
    x : array-like
        If scalar, *x* is ``np.arange(0,x)``. If iterable, can be
        n-dimensional, and will calculate sine from coordinates on
        every dimension.
    wavelens : float
        Wavelengths for sine function. Required if `wavenums` is ``None``.
    wavenums : float
        Wavenumbers for sine function. Required if `wavelens` is ``None``.
    phase : float, optional
        Array of phase offsets.

    Returns
    -------
    data : array-like
        Data composed of sine waves.

    Note
    ----
    `x` will always be normalized so that wavelength is with reference to
    the first step. This make sense because when working with filters, for
    which we almost always need to use units corresponding to the axis.
    """
    # Wavelengths
    if wavenums is None and wavelens is None:
        raise ValueError('Must declare wavenums or wavelengths.')
    elif wavelens is not None:
        # dx = x[1] - x[0]
        wavenums = 1.0 / np.atleast_1d(wavelens)
    wavenums = np.atleast_1d(wavenums)
    if not hasattr(x, '__iter__'):
        x = np.arange(x)
    data = np.zeros(x.shape)  # user can make N-D array

    # Get waves
    if phase is None:
        phis = np.random.uniform(0, 2 * np.pi, len(wavenums))
    else:
        phis = phase * np.ones([len(wavenums)])
    for wavenum, phi in zip(wavenums, phis):
        data += np.sin(2 * np.pi * wavenum * x + phi)
    return data


def window(wintype, n):
    """
    Retrieve the `~scipy.signal.get_window` weighting function window.

    Parameters
    ----------
    wintype : str or (str, float) tuple
        The window name or ``(name, param1, ...)`` tuple containing the window
        name and required parameter(s).
    n : int
        The window length.

    Returns
    -------
    win : array-like
        The window coefficients.

    power of your FFT coefficients. If your window requires some parameter,
    `wintype` must be a ``(name, parameter1, ...)`` tuple.
    """
    # Default error messages are shit, make them better
    if wintype == 'welch':
        raise ValueError('Welch window needs 2-tuple of (name,beta).')
    if wintype == 'kaiser':
        raise ValueError('Welch window needs 2-tuple of (name,beta).')
    if wintype == 'gaussian':
        raise ValueError('Gaussian window needs 2-tuple of (name,stdev).')

    # Get window
    win = signal.get_window(wintype, n)
    return win
