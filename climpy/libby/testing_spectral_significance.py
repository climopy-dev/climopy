#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:38:49 2017

@author: eabarnes
"""

#.............................................
# IMPORT STATEMENTS
#.............................................
#import time
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
import scipy.signal as sig
import scipy.stats as stats
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

#%% input parameters
T = 256         #length of window
N = 40          #number of realizations

alpha = 0.5     #red noise lag-one autocorrelation
#%%

T2 = T/2
freq = np.arange(0.,T2+1.)/T

# contstruct expected rednoise spectrum
rspec = []
for i in np.arange(1,T2+2,1):
    rspec.append((1.-alpha*alpha)/(1.-2.*alpha*np.cos(np.pi*(i-1.)/T2)+alpha*alpha))
    
factor = np.sqrt(1.-alpha*alpha)

x = np.zeros(T,)
pnum = 0
# loop realizations
for ir in np.arange(0,1,1):
    
    x[0] = x[-1]*alpha + factor*np.random.randn()

    for j in np.arange(1,T,1):
        x[j] = x[j-1]*alpha + factor*np.random.randn()+0.5*np.cos(2.*np.pi*(1.-0.01*np.random.randn())*50./256.*j)
        

    p = sig.welch(x,window='hanning', nperseg=T);
    if(ir==0):
        psum = p[1]
    else:
        psum = psum + p[1]

    # calculate average    
    pave = psum/(ir+1.0)
    #normalize the spectrum
    pave = pave/np.mean(pave)
 
    
    # calculate significance
    dof = 2.*(ir+1.)
    fstat = stats.f.ppf(.99,dof,1000)
    spec99 = [fstat*m for m in rspec]
    
    if((ir+1.) % 5 == 0 or ir==0):       
        plt.xlabel('frequency (cycles per time step)')
        plt.ylabel('power')
        plt.title('# Realizations = ' + str(ir+1))
        plt.ylim(0,20.)
        plt.plot(freq,pave,'-k', label = 'data')
        plt.plot(freq,rspec,'-', label = 'red-noise fit', color = 'blue')
        plt.plot(freq,spec99,'--', label = '99% confidence', color = 'red')
        plt.legend(frameon = False)
        plt.pause(3)




