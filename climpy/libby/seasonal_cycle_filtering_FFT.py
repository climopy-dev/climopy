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
#import scipy.signal as sig
#from scipy import stats
import scipy.io as sio
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
DATA = sio.loadmat('mslp_erainterim_daily_33N_170E.mat')
X = DATA['Xin']
TIME = DATA['TIME']


#%% calculate the daily climatology
TIME_climate = np.vstack({tuple(row) for row in TIME[:,2:4]})
TIME_climate = TIME_climate[np.lexsort((TIME_climate[:,1],TIME_climate[:,0])),:]

days_in_year = np.size(TIME_climate, axis=0)
Yclim = np.empty(days_in_year)
Yclim[:] = np.NAN

for iday in np.arange(0,days_in_year):
    month = TIME_climate[iday,0]
    day = TIME_climate[iday,1]

    t = np.where(np.logical_and(TIME[:,2]==month,TIME[:,3]==day))
    Yclim[iday] = np.nanmean(X[t])

 
    
#%% plot mean seasonal cycle with all of the daily wiggles
# gf.cfig(1)
plt.figure()
plt.plot(np.arange(0,np.size(Yclim)),Yclim,'-k')
plt.ylabel('hPa')
plt.xlabel('day of year')
plt.xlim(-1,366)
plt.title('pressure at 33N, 170E')
# gf.show_plot()

#%% plot anomalous pressure

#remove the mean of the time series for the FFT
X = Yclim - np.mean(Yclim);

# gf.cfig(10)
plt.figure()
plt.plot(np.arange(0,np.size(X)),X,'-k')
plt.ylabel('hPa')
plt.xlabel('day of year')
plt.xlim(-1,366)
plt.title('anomalous pressure at 33N, 170E')
# # gf.plot_zero_lines()
# gf.show_plot()

#%% calculate the FFT of the seasonal cycle

#remove the mean of the time series for the FFT
X = Yclim - np.mean(Yclim);
#don't remove the mean of the time series
#X = Yclim;

Z = np.fft.fft(X)

# to get the right variance, need to normalize by the length of the time
# series, however, don't want to to do this if inputting back into "ifft"
Yfft = Z/np.size(Yclim);


#%% plot the out FFT power

# gf.cfig(2)
plt.figure()
plt.plot(np.arange(0,np.size(Yfft)),np.abs(Yfft)**2)
plt.plot(0,np.abs(Yfft[0])**2.,'sg',markersize=10)
plt.plot(1,np.abs(Yfft[1])**2.,'*r',markersize=20)
plt.plot(np.size(Yfft)-1,np.abs(Yfft[-1])**2.,'*r',markersize=20)

plt.xlabel('index')
plt.ylabel('$C_k^2$ (power)')

plt.title('Python numpy FFT output')

plt.ylim(-.5,4)
plt.xlim(-5,365+5)
# gf.show_plot()

#%% combine symmetric parts of the FFT and plot the power spectrum as a function of frequency
#
freq = np.arange(0,np.size(Yclim)/2+1)/float(np.size(Yclim))

Ck2 = 2.*np.abs(Yfft[0:np.size(Yclim)//2+1])**2 # the factor of 2 in front is needed or the sum won't equal the total variance of X

# gf.cfig(3)
plt.figure()
plt.plot(freq,Ck2/np.sum(Ck2),'-k',linewidth = 3)


plt.ylabel('normalized power')
plt.xlabel('frequency (cycles per day)')
plt.xlim(-.001,.1)
plt.title('normalized power spectrum')

# gf.show_plot()

#%% checking the variance compared to the total variance in our spectrum
# this is checking Parseval's Theorem
#
# actual variance of your data
var_actual = np.var(X)

# variance in your spectrum, it should be close to the actual variance of
# your data
a = Yfft[np.arange(0,np.size(Yclim)//2+1)]
s=np.sum(a[1::]*np.conj(a[1::])); # don't want to include the mean, as this doesn't show up in the variance calculation
var_spectrum = np.real(2*s) # multiply by two in order to conserve variance

print(str(var_actual) + ', ' + str(var_spectrum))


#%% plot spectrum and how we will only retain the first 3 harmonics (which includes the mean)

# #gf.cfig(4)
plt.figure()
plt.plot((0,0),(0,1),'-r', linewidth = 1.5)
plt.plot((freq[2],freq[2]),[0,1],'-r', linewidth = 1.5)

#plt.xlim([-0.001, 0.05])
plt.ylim([0,.69])

A = Ck2/np.sum(Ck2)
A[3::] = 0.
plt.plot(freq[0:3],A[0:3],'--r',linewidth = 3)

# gf.show_plot()
#%% high-pass filter (everything you didn't remove before)

ndelete = 4
Z3 = np.copy(Z)
Z3[0:ndelete+1:] = 0.
Z3[-ndelete::] = 0.

X_hp = np.real(np.fft.ifft(Z3))

# plot seasonal cycle

# gf.cfig(10)
plt.figure()
plt.plot(np.arange(0,np.size(X)),X,'-k')
plt.ylabel('hPa')
plt.xlabel('day of year')
plt.xlim(-1,366)
plt.title('anomalous pressure at 33N, 170E')
# gf.plot_zero_lines()

plt.plot(np.arange(0,np.size(X)),X_hp,'-b',linewidth = 3)

# gf.show_plot()


#%% retain only the mean and the first two harmonics, set all other frequencies to zero

Z2 = np.copy(Z)
Z2[ndelete+1:-ndelete:] = 0.0 # so retain first two
#Z2[4:-3:] = 0.0

X_smoothed = np.real(np.fft.ifft(Z2))

# plot seasonal cycle

# gf.cfig(10)
plt.figure()
plt.plot(np.arange(0,np.size(X)),X,'-k', label = 'data')
plt.ylabel('hPa')
plt.xlabel('day of year')
plt.xlim(-1,366)
plt.title('anomalous pressure at 33N, 170E')
# gf.plot_zero_lines()

plt.plot(np.arange(0,np.size(X)),X_hp,'-b',linewidth = 3, label='high-pass filter')

plt.plot(np.arange(0,np.size(X)),X_smoothed,'-r',linewidth = 5, label ='low-pass filter')

# plot sum of low pass and high pass
plt.plot(np.arange(0,np.size(X)),X_smoothed+X_hp,'-',color = 'orange',linewidth = 1, label='low-pass high-pass sum')

plt.legend()

# gf.show_plot()






