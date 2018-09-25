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
# reload(gf)


#scipy.linalg

#.............................................
# PLOTTING COMMANDS
#.............................................
# gf.cc()
plt.ion()

#%% START CODE

x = np.array((0.,1.,2.,3.,4.,5.,7.))
y = np.array((0.,2.,4.,3.,3.,4.,6.))

# gf.cfig(1)
plt.plot(x,y,'sk', markersize = 10, markerfacecolor = 'none', markeredgewidth = 2)

plt.xlim(-0.5,8.)
plt.ylim(-0.5,8.)

# gf.show_plot()

#%% ployfit all points with 5th order polynomial

xplot = np.arange(0.,7.+.01,.01)

p = np.polyfit(x, y, 5, rcond=None, full=False, w=None, cov=False)
plt.plot(xplot,np.poly1d(p)(xplot),'--k')

# gf.show_plot()
#%% remove one point and repeat
xnew = np.delete(x,2)
ynew = np.delete(y,2)

plt.plot(xnew,ynew,'or',markersize = 8)

p = np.polyfit(xnew, ynew, 5, rcond=None, full=False, w=None, cov=False)
plt.plot(xplot,np.poly1d(p)(xplot),'--r')

# gf.show_plot()
