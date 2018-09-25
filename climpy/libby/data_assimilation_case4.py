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
#from scipy import interpolate
#import numpy.ma as ma
#import csv
#from numpy import genfromtxt
#from mpl_toolkits.basemap import Basemap

# import general_functions as gf
# )eload(gf)


#scipy.linalg

#.............................................
# PLOTTING COMMANDS
#.............................................
# gf.cc()
plt.ion()

#%% START CODE

deltax = np.arange(-8.,8.,.1)
x = np.arange(-4.,4,.1)


pt1_pos = -2.
alpha = 0.25
rho_10 = 0.406
rho_2 = (1.+deltax)*np.exp(-deltax)


w1 = np.empty(np.size(x))
w2 = np.empty(np.size(x))
e = np.empty(np.size(x))

for ix,xval in enumerate(x):
    
    rho_12 = (1.+np.abs(xval-pt1_pos))*np.exp(-np.abs(xval-pt1_pos))
    rho_20 = (1.+np.abs(xval-0.))*np.exp(-np.abs(xval-0.))
    
    w1[ix] = (rho_10*(1.+alpha) - rho_12*rho_20)/((1.+alpha)**2 - rho_12**2)
    w2[ix] = (rho_20*(1.+alpha) - rho_12*rho_10)/((1.+alpha)**2 - rho_12**2)

    e[ix] = 1. - ( (1.+alpha)*(rho_10**2 + rho_20**2) - 2.*rho_10*rho_20*rho_12 )/ ( (1.+alpha)**2 - rho_12**2 )

# gf.cfig(1)
plt.figure()
plt.plot(x,w1,'-',color = 'red', label = 'w1')
plt.plot(x,w2,'-', color = 'blue', label = 'w2')
plt.plot(x,e,'-', color = 'black', label = 'normalized error')

plt.plot((-2,-2),(-.2,1.0),'--r')

plt.ylabel('value')

plt.xlabel('position of obs. 2')

# gf.plot_zero_lines()
plt.legend(frameon = False, loc = 0)

# gf.show_plot()
