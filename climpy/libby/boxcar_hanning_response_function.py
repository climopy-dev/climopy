#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 20:47:26 2017

@author: eabarnes
"""

#.............................................
# IMPORT STATEMENTS
#.............................................
#import time
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
#import scipy.signal as sig
#import scipy.stats as stats
#import numpy.ma as ma
#import csv
#import scipy.io as sio
#import numpy.linalg as LA
#from mpl_toolkits.basemap import Basemap
#from matplotlib.patches import Polygon



#.............................................
# PLOTTING COMMANDS
#.............................................
#gf.cc()
plt.ion()
#%matplotlib inline
#%matplotlib qt
#%%
LW = 3
FS = 16
T = 3
omega = np.arange(-2.*np.pi*3., 2.*np.pi*3.+.1, .1)
t = np.arange(0.01,T+0.01,0.01)

#%% Windows in time space


plt.xlim(-0.05,T+0.05)
plt.ylim(-0.02,1.5)

b = np.ones(np.shape(t))
plt.plot(t,b,'-b', linewidth = LW, label = 'Boxcar window')

plt.plot((0,0),(0,1),'-b',linewidth = LW)
plt.plot((T,T),(0,1),'-b',linewidth = LW)

w = 0.5*(1-np.cos(2.*np.pi*t/T))
plt.plot(t,w,'-r',linewidth = LW, label = 'Hanning window')

plt.xlabel('time')
plt.ylabel('value of data (as a fraction)')

plt.legend(frameon = False)
#%% Response functions in frequency space


# ------- setup response functions ----------------
B = np.sinc(omega*T/(2.*np.pi))
#B2 = (T*np.sinc(omega*T))
W = np.sinc(omega*T/(2.*np.pi)) + (1./2.)*(np.sinc(omega*T/(2.*np.pi) + 1.) + np.sinc(omega*T/(2.*np.pi) - 1.))
Wa = np.sinc(omega*T/(2.*np.pi))
Wb = (1./2.)*(np.sinc(omega*T/(2.*np.pi) + 1.) + np.sinc(omega*T/(2.*np.pi) - 1.))
#W2 =  T*np.sinc(omega*T) + (T/2.)*(np.sinc(omega*T + 1.) + np.sinc(omega*T - 1.))
#--------------------------------------------------

#%% plot Boxcar response function

plt.plot(omega,B/np.max(B),'-b',linewidth = LW, label = 'Boxcar response')
plt.xlabel('radial frequency')
plt.ylabel('spectral power')
plt.xlim(-16,16)

plt.legend(frameon = False, loc = 'upper left', fontsize = FS)

#%% plot terms 2 and 3 of Hanning response

plt.plot(omega, Wb,'--r', linewidth = LW, label = 'terms 2 & 3 of Hanning response')
plt.legend(frameon = False, loc = 'upper left', fontsize = FS)

#%% plot full Hanning response

plt.plot(omega, W,'-r', linewidth = LW, label = 'full Hanning response')
plt.legend(frameon = False, loc = 'upper left', fontsize = FS)

