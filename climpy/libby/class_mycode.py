#------------------------------------------------------------------------------#
# TODO Finish parsing this; was copied from sst project
# Gets the significance accounting for autocorrelation or something
#------------------------------------------------------------------------------#
# Simple function for getting error, accounting for autocorrelation
def error(r):
    """
    Manually calculate the error according to Wilks definition; compare to bse.
    Proves that the bse parameter is equal to Equation 7.18b in Wilks.
    """
    # Use fittedvalues and resid attributes
    # xfact = (x**2).sum()/(r.nobs*(x-x.mean())**2).sum() # for constant, a
    # x = r.model.exog[:,-1] # exogeneous variable
    # xfact = 1/((x-x.mean())**2).sum()
    se2 = (r.resid.values**2).sum()/(r.nobs-2) # Eq 7.9, Wilks
    xfact2 = 12/(r.nobs**3-r.nobs) # Eq 3, Thompson et. al 2015
    sigma = (se2**0.5)*(xfact2**0.5)
    print('Provided: %.10f, manually calculated: %.10f' % (r.bse.x1, sigma))
    return

#------------------------------------------------------------------------------#
# Function for performing regression, along with significance
# on regression coefficient or something
# Below uses statsmodels, but slower for small vectors
# X = sm.add_constant(T.index.values) # use the new index values
# r = sm.OLS(T[name,region_get], X).fit() # regress column [name] against X
# scale = ((r.nobs-2)/(r.nobs*((1-autocorr)/(1+autocorr))-2))**0.5
# L[name,region] = [unit*r.params.x1, scale*unit*r.bse.x1]
# Below uses np.corrcoef, but is slower than pearsonr
# autocorr = np.corrcoef(r.resid.values[1:], r.resid.values[:-1])[0,1] # outputs 2by2 matrix
def regress(x, y, unit=10, sigma=False, ignorenan=False):
    """
    Gets regression results with fastest methods possible.
    See the IPython noteobok for comparisons.
    """
    # NaN check
    x, y = x.squeeze(), y.squeeze()
    if y.ndim>1:
        raise ValueError('y is not 1-dimensional.')
    nan = np.isnan(y)
    if nan[0] and ignorenan:
        # If we are getting continuous trends, don't bother computing this one, because
        # we will also get trends from x[1:], x[2:], etc.
        return np.nan, np.nan, np.nan, np.nan
    if nan.any():
        # Filter out NaNs, get regression on what remains
        x, y = x[~nan], y[~nan]
    if x.size<5:
        # Cannot get decent sigma estimate, in this case
        return np.nan, np.nan, np.nan, np.nan
    # Regress, and get sigma if requested
    # First value is estimated change K through record, second is K/unit
    if sigma:
        p, V = np.polyfit(x, y, deg=1, cov=True)
        resid = y - (x*p[0] + p[1]) # very fast step; don't worry about this one
        autocorr, _ = st.pearsonr(resid[1:], resid[:-1])
        scale = (x.size-2)/(x.size*((1-autocorr)/(1+autocorr))-2)
            # scale factor from Thompson et. al, 2015, Quantifying role of internal variability...
        stderr = np.sqrt(V[0,0]*scale)
        return (x[-1]-x[0])*p[0], unit*p[0], (x[-1]-x[0])*stderr, unit*stderr
    else:
        p = np.polyfit(x, y, deg=1)
        return (x[-1]-x[0])*p[0], unit*p[0]

#-------------------------------------------------------------------------------
# TODO: Finish harvesting this original function written for the Objective
# Analysis assignment. Consider deleting it.
#-------------------------------------------------------------------------------
def spectral(fnm, nm, data, norm=True, win=501,
            freq_scale=1, scale='days',
            xlog=True, ylog=False, mc='k',
            xticks=None,
            rcolors=('C3','C6'), pcolors=('C0','C1'), alpha=0.99, marker=None,
            xlim=None, ylim=None, # optional override
            linewidth=1.5,
            red_discrete=True, red_contin=True, manual=True, welch=True):
    '''
    Spectral transform function. Needs work; was copied from OA
    assignment and right now just plots a bunch of stuff.
    '''
    # Iniital stuff
    N = len(data)
    pm = int((win-1)/2)
    fig, a = plt.subplots(figsize=(15,5))

    # Confidence intervals
    dof_num = 1.2*2*2*(N/(win/2))
    trans = 0.5
    exp = 1
    F99 = stats.f.ppf(1-(1-alpha)**exp,dof_num,1000)
    F01 = stats.f.ppf((1-alpha)**exp,dof_num,1000)
    print('F stats:',F01,F99)
    rho = np.corrcoef(data[1:],data[:-1])[0,1]
    kr = np.arange(0,win//2+1)
    fr = freq_scale*kr/win

    def xtrim(f):
        if xlim is None:
            return np.ones(f.size, dtype=bool)
        else:
            return ((f>=xlim[0]) & (f<=xlim[-1]))

    # Power spectra
    if manual:
        label = 'power spectrum'
        if welch: label = 'manual method'
        # Now, manual method with proper overlapping etc.
        if False:
            data = data[:(N//pm)*pm]
        loc = np.linspace(pm,N-pm-1,2*int(np.round(N/win))).round().astype(int) # sample loctaions
        han = np.hanning(win)
        han = han/han.sum()
        phi = np.empty((len(loc),win//2))
        for i,l in enumerate(loc):
            pm = int((win-1)/2)
            C = np.fft.fft(han*signal.detrend(data[l-pm:l+pm+1]))
            phii = np.abs(C)**2/2
            phii = 2*phii[1:win//2+1]
            phi[i,:] = phii
        phi = phi.mean(axis=0)
        print('phi sum:',phi.sum())
        f = np.fft.fftfreq(win)[1:win//2+1]*freq_scale
        if norm: phi = phi/phi.sum()
        f, phi = f[xtrim(f)], phi[xtrim(f)] # trim
        a.plot(f, phi, label=label,
               mec=mc, mfc=mc, mew=linewidth,
               marker=marker, color=pcolors[0], linewidth=linewidth)
        if xlim is None: xlim = ((f*freq_scale).min(), (f*freq_scale).max())
        if ylim is None: ylim = ((phi.min()*0.95, phi.max()*1.05))

    if welch:
        label = 'power spectrum'
        if manual: label = 'welch method'
        # Welch
        fw, phi_w = signal.welch(data, nperseg=win, detrend='linear', window='hanning', scaling='spectrum',
                              return_onesided=False)
        fw, phi_w = fw[1:win//2+1]*freq_scale, phi_w[1:win//2+1]
        if norm: phi_w = phi_w/phi_w.sum()
        fw, phi_w = fw[xtrim(fw)], phi_w[xtrim(fw)] # trim
        print('phiw sum:',phi_w.sum())
        a.plot(fw, phi_w, label=label,
              mec=mc, mfc=mc, mew=linewidth,
               marker=marker, color=pcolors[-1], linewidth=linewidth)
        if xlim is None: xlim = ((fw).min(), (fw).max())
        if ylim is None: ylim = (phi_w.min()*0.95, phi_w.max()*1.05)

    # Best fit red noise spectrum
    if red_discrete:
        print('Autocorrelation',rho)
        phi_r1 = (1-rho**2)/(1+rho**2-2*rho*np.cos(kr*np.pi/(win//2)))
        print('phi_r1 sum:',phi_r1.sum())
        if norm: phi_r1 = phi_r1/phi_r1.sum()
        frp, phi_r1 = fr[xtrim(fr)], phi_r1[xtrim(fr)]
        a.plot(fr[xtrim(fr)], phi_r1, label=r'red noise, $\rho(\Delta t)$',
               marker=None, color=rcolors[0], linewidth=linewidth)
        a.plot(frp, phi_r1*F99, linestyle='--',
               marker=None, alpha=trans, color=rcolors[0], linewidth=linewidth)

    # Alternate best fit
    if red_contin:
        Te = -1/np.log(rho)
        omega = (kr/win)*np.pi*2
        phi_r2 = 2*Te/(1+(Te**2)*(omega**2))

        print('phi_r2 sum:',phi_r2.sum())
        if norm: phi_r2 = phi_r2/phi_r2.sum()
        frp, phi_r2 = fr[xtrim(fr)], phi_r2[xtrim(fr)]
        a.plot(frp, phi_r2, label=r'red noise, $T_e$',
               marker=None, color=rcolors[1], linewidth=linewidth)
        a.plot(frp, phi_r2*F99, linestyle='--',
               marker=None, alpha=trans, color=rcolors[-1], linewidth=linewidth)
    # Variance
    print('true variance:',data.std()**2)
    # Figure formatting
    a.legend()
    if ylog:
        a.set_yscale('log')
    if xlog:
        a.set_xscale('log')
    a.set_title('%s power spectrum' % nm)
    a.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%5.3g'))
    a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if xticks is None: xticks = a.get_xticks()
    my.format(a, xlabel=('frequency (%s${}^{-1}$)' % scale), ylabel='proportion variance explained',
             xlim=xlim, ylim=ylim, xticks=xticks)
    suffix = 'pdf'
    fig.savefig('a5_' + fnm + '.' + suffix, format=suffix, dpi='figure')
    plt.show()
