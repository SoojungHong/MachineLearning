#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 23:52:26 2018

@author: soojunghong

Python plotting using matplotlib
"""
#------------------
# font
#------------------
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']

import matplotlib.pyplot as plt
fig, ax = plt.subplots() #create a figure and a set of subplots, return figure and axes
ax.plot([1, 2, 3], label='test')

ax.legend()
plt.show()


#----------------
# unicode minus 
#----------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


matplotlib.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots() #create a figure and a set of subplots, return figure and axes
ax.plot(10*np.random.randn(100), 10*np.random.randn(100), 'o') #Plot y versus x as lines and/or markers. plot(x,y)
ax.set_title('Using hyphen instead of Unicode minus')
plt.show()

#-----------------
# Simple plot 
#-----------------
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.01) #from 0.0 until 2.0 with interval 0.01 
s = 1 + np.sin(2 * np.pi * t)

# Note that using plt.subplots below is equivalent to using
# fig = plt.figure() and then ax = fig.add_subplot(111)
fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()

#-----------------
# Text watermark
#-----------------
import numpy as np
#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


fig, ax = plt.subplots()
ax.plot(np.random.rand(20), '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
ax.grid()

# position bottom right
fig.text(0.95, 0.05, 'Property of MPL',
         fontsize=50, color='gray',
         ha='right', va='bottom', alpha=0.5)

plt.show()
