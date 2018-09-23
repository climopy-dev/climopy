#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
#.............................................
# INTITIAL SETUP
#.............................................

#.............................................
# IMPORT STATEMENTS
#.............................................
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
import scipy.signal as sig
#from scipy import stats
#from scipy import interpolate
#import numpy.ma as ma
#import csv
#from numpy import genfromtxt
#from mpl_toolkits.basemap import Basemap

# import general_functions as gf
# reload(gf)


#scipy.linalg

#.............................................
# PLOTTING COMMANDS
#.............................................
# gf.cc()
plt.ion()
#%matplotlib inline
#%matplotlib qt



#%% START CODE
N = 4
b, a = sig.butter(N, .25)

freq = np.arange(0.,0.51,0.01)

omega = 2.*np.pi*freq
omega_c = .25*2*np.pi

#%%
N = 4
R2_4 = 1./(1. + (omega/omega_c)**(2*N))

N = 9
R2_9 = 1./(1. + (omega/omega_c)**(2*N))

N = 30
R2_q = 1./(1. + (omega/omega_c)**(2*N))

#%%
# gf.cfig(1)
plt.figure()
plt.plot(freq, R2_4,'-k',linewidth = 2, label = 'Butterworth N = 4')
plt.plot(freq, R2_9,'-r',linewidth = 2, label = 'Butterworth N = 9')
plt.plot(freq, R2_q,'--b',linewidth = 2, label = 'Butterworth N = 30')

plt.xlabel('frequency')
plt.ylabel('|R^2| response')
plt.legend(fontsize = 14, frameon = False)

plt.ylim(0,1.1)

# gf.show_plot()

#%% impluse response of butterworth filters

x = np.zeros((100,))
x[0] = 1.

# gf.cfig(2)
plt.figure()

N = 4
b, a = sig.butter(N, .25)
y = sig.lfilter(b,a,x)
plt.plot(y,'-k',linewidth = 2, label = 'N=4')

N = 9
b, a = sig.butter(N, .25)
y = sig.lfilter(b,a,x)
plt.plot(y,'-r',linewidth = 2, label = 'N=9')

N = 30
b, a = sig.butter(N, .25)
y = sig.lfilter(b,a,x)
plt.plot(y,'--b',linewidth = 2, label = 'N=30')

# gf.plot_zero_lines()

plt.xlabel('time')
plt.ylabel('impulse response')
plt.title('Impulse response of Butterworth filter')

plt.legend()

# gf.show_plot()

#%% impluse response at both ends of butterworth filters

x = np.zeros((200,))
x[0] = 1.
x[100] = 1.

# gf.cfig(5)
plt.figure()

plt.plot(x,'-k',linewidth = 2, label = 'raw data')


N = 30
b, a = sig.butter(N, .25)
y = sig.lfilter(b,a,x)
plt.plot(y,'--g',linewidth = 2, label = 'forward')

y2 = sig.lfilter(b,a,y[::-1])
y2 = y2[::-1]
plt.plot(y2,'--r',linewidth = 2, label = 'forward-backard')

# gf.plot_zero_lines()

plt.xlabel('time')
plt.ylabel('impulse response')
plt.title('Impulse response of Butterworth filter')

plt.xlim(-1,np.size(x)+1)

plt.legend(frameon = False, fontsize = 16)

# gf.show_plot()

#%% butterworth filter of actual data

chunk_length = 200
num_chunks= 1
n = chunk_length*num_chunks

# generate red noise time series with autocorrelation
alpha = 0.5
height = 2.0
factor = np.sqrt(1.-alpha*alpha)

x = np.zeros((n,))
pnum = 0

x[0] = x[-1]*alpha + factor*np.random.randn()
for j in np.arange(1,n):
    x[j] = x[j-1]*alpha + factor*np.random.randn()+1.0*np.cos(2.*np.pi*(1.-0.01*np.random.randn())*52./256.*j) + 0.75*np.cos(2.*np.pi*(1.-.001*np.random.randn())*100./256*j-np.pi/4.)
        
xa = x - np.mean(x)


# gf.cfig(10)
plt.figure()
N = 9
b, a = sig.butter(N, .25)
y = sig.lfilter(b,a,xa)
plt.plot(x,'-k',linewidth = 1.5, label = 'raw input')
plt.plot(y,'-r',linewidth = 2.5, label = '1 pass')

y2 = sig.lfilter(b,a,y[::-1])
plt.plot(y2[::-1],'--b',linewidth = 2.5, label = '2 passes (forward-backward)')

plt.title('butterworth filter')

plt.legend(frameon = False,fontsize = 12)
# gf.show_plot()

#%% same data, 1-2-1 filter, or moving average filter
# gf.cfig(11)
plt.figure()
plt.plot(x,'-k',linewidth = 1.5, label = 'raw input')

b = (1,2,1)
y = sig.lfilter(b,np.sum(b),x)
plt.plot(y,'-r', label = 'forward')
y2 = sig.lfilter(b,np.sum(b),y[::-1])
plt.plot(y2[::-1],'-b', label = 'forward-backward')

plt.xlim(-1,np.size(x)+1)

plt.legend(frameon = False, fontsize = 16)

plt.title('1-2-1 filter')

# gf.show_plot()


#%% Lanczos Filter

def low_pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.

    Args:

    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.

    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
        
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    
    return w[1:-1]

#------------------------------------------------------------------------------
# window length for filters
window = 41
wgts24 = low_pass_weights(window, 1. / 11.)

y  = sig.lfilter(wgts24,np.sum(wgts24),x)
y2 = sig.lfilter(wgts24,np.sum(wgts24),y[::-1])
plt.plot(y2[::-1], color = 'orange', label = 'Lanczos forward-backward')

plt.legend(frameon = False, fontsize = 16)

# gf.cfig(13)
plt.figure()
plt.plot(wgts24/np.sum(wgts24))

plt.title('Lanczos window/kernel')
# gf.show_plot()

#%%
################################################################################
# shows the same thing?
################################################################################
# cutoff = 1./11.
#
# order = ((window - 1) // 2 ) + 1
# nwts = 2 * order + 1
# w = np.zeros([nwts])
# n = nwts // 2
# w[n] = 2 * cutoff
# k = np.arange(1., n)
# sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
# firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
#
# w[n-1:0:-1] = firstfactor * sigma
# w[n+1:-1] = firstfactor * sigma
#
# #w = w/(2.*cutoff)
# # gf.cfig(13)
# plt.figure()
# plt.plot(w)
# gf.show_plot()
