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
from scipy import stats
#import scipy.io as sio
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
num_iters = 1000

chunk_length = 256
num_chunks= 20
n = chunk_length*num_chunks

# generate red noise time series with autocorrelation
alpha = 0.5
height = 2.0
factor = np.sqrt(1.-alpha*alpha)

xa = np.empty((num_iters,chunk_length*num_chunks))

for k in np.arange(0,num_iters):
    
    x = np.zeros((n,))
    pnum = 0
    x[0] = x[-1]*alpha + factor*np.random.randn()
    for j in np.arange(1,n):
        x[j] = x[j-1]*alpha + factor*np.random.randn()
            
    xa[k,:] = x  - np.mean(x)


#%% calculate the FFT of the seasonal cycle

win_rect = np.ones(chunk_length,)

Pxx_HO = np.empty((num_iters,chunk_length//2+1), dtype=np.complex_)
Pxx_H = np.empty((num_iters,chunk_length//2+1), dtype=np.complex_)

for k in np.arange(0,num_iters):
    F, pxx = sig.csd(xa[k,:], xa[k,:], window = 'hann', noverlap = 0, nperseg = chunk_length,\
                     nfft = np.size(xa,1)/float(num_chunks), scaling = 'density', detrend = False)
    Pxx_H[k,:] = np.abs(pxx)/np.sum(np.abs(pxx))

    F, pxx = sig.csd(xa[k,:], xa[k,:], window = 'hann', noverlap = chunk_length//2, nperseg = chunk_length, \
                     nfft = np.size(xa,1)/float(num_chunks), scaling = 'density', detrend = False)
    Pxx_HO[k,:] = np.abs(pxx)/np.sum(np.abs(pxx))
    
#%% plot the out FFT power

#==============================================================================
# PLOT HANNING + OVERLAP
plot_Power = Pxx_HO
plot_Power_upper = np.percentile(plot_Power,95,axis=0)

plt.figure()
# gf.cfig(20)
for k in np.arange(0,num_iters):
    plt.plot(F,plot_Power[k,:],linewidth = .5, color = 'gray')

plt.plot(F,plot_Power_upper,linewidth = 3, color = 'k')

plt.xlabel('frequency')
plt.ylabel('power')

plt.title('Hanning + overlap')

#--------------------------------------------------
# calculate and plot red-noise fit
N = chunk_length
h = np.arange(0,N//2+1)
Te = -1./np.log(alpha)
rnoise = (1.-alpha**2)/(1.-2.*alpha*np.cos(h*np.pi/(N//2.))+alpha**2)

rnoise = rnoise/np.sum(rnoise)

# hanning and overlap
dof = 1.2*(2.*(num_chunks*2.-1.))  #for 50% overlap    #2*2.8*num_chunks*chunk_length*np.diff(F)[0] # Welch's way of writing things

fst = stats.f.ppf(.95,dof,1000)
spec99 = fst*rnoise
plt.plot(F,spec99,'--',color = 'r', label = '95% conf. bound')
plt.legend()
#--------------------------------------------------

# gf.show_plot()
#==============================================================================

#==============================================================================
# PLOT HANNING only
plot_Power = Pxx_H
plot_Power_upper = np.percentile(plot_Power,95,axis=0)

plt.figure()
# gf.cfig(30)
for k in np.arange(0,num_iters):
    plt.plot(F,plot_Power[k,:],linewidth = .5, color = 'gray')

plt.plot(F,plot_Power_upper,linewidth = 3, color = 'k')

plt.xlabel('frequency')
plt.ylabel('power')

plt.title('Hanning only')

#--------------------------------------------------
# calculate and plot red-noise fit
N = chunk_length
h = np.arange(0,N//2+1)
Te = -1./np.log(alpha)
rnoise = (1.-alpha**2)/(1.-2.*alpha*np.cos(h*np.pi/(N//2.))+alpha**2)

rnoise = rnoise/np.sum(rnoise)

# hanning but no overlap
dof = (2.*num_chunks)*1.2

fst = stats.f.ppf(.95,dof,1000)
spec99 = fst*rnoise
plt.plot(F,spec99,'--',color = 'r', label = '95% conf. bound')
plt.legend()

#--------------------------------------------------

# gf.show_plot()
#==============================================================================

#%% checking the variance compared to the total variance in our spectrum
# this is checking Parseval's Theorem
#
# actual variance of your data
#var_actual = np.var(xa[10,:])

# variance in your spectrum, it should be close to the actual variance of
# your data...ASSUMING YOU HAVEN"T NORMALIZED TO NO LONGER BE DENSITY
#var_spectrum = np.sum(np.abs(Pxx_HO[10,:])*np.diff(F)[0]) # since spectral DENSITY, need to multiply by DELTA_F

#print '--------------------------------------------------------------'
#print 'var(x) = ' + str(var_actual) + ', sum of spectrum = ' + str(var_spectrum)
#print '--------------------------------------------------------------'






