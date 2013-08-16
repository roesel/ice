#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Load five CSV files and plot profiles and r into two subplots to compare.'''

# ---------------- Imports
import numpy as np
import matplotlib.pyplot as plt


def readArray(filename, dtype, separator=','):
    ''' (not ours) Read a file with an arbitrary number of columns. The type
        of data in each column is also arbitrary - it will be cast to the
        given dtype at runtime. '''

    cast = np.cast
    data = [[] for dummy in xrange(len(dtype))]
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            fields = line.strip().split(separator)
            for i, number in enumerate(fields):
                data[i].append(number)
    for i in xrange(len(dtype)):
        data[i] = cast[dtype[i]](data[i])
    return np.rec.array(data, dtype=dtype)


def multiFit(y, z):
    ''' Will fit and substract a z baseline from all values in one line.
        Returns leveled z-line values. '''

    pfit = np.polyfit(y, z, 3)
    zbaseline = np.polyval(pfit, y)
    zfixed = z-zbaseline

    return zfixed


def linearFit(y, z):
    ''' Older way of linear fit that is substracted from all z values in line.
        Returns leveled z-line values. '''
    # Fitting with linearly generated sequence
    A = np.array([y, np.ones(y.size)])
    w = np.linalg.lstsq(A.T, z)[0]  # obtaining the parameters
    zline = w[0]*y+w[1]
    zfixed = z-zline  # substracting baseline from every point

    return zfixed

# -- BEGIN PARAMETERS ----
profile = '1200_3d_snp5_p'
# ------ END PARAMETERS --

# Create the figure
fig = plt.figure()

# Create "profile" subplot
ax1 = fig.add_subplot(2, 1, 1)

# Create "r" subplot
ax2 = fig.add_subplot(2, 1, 2)

# Plot 5 different profiles and r values into two subplots
for i in range(1, 6):
    # Load data from a single CSV into array
    mydescr = np.dtype([('x', 'float32'), ('z', 'float32')])
    data = readArray(profile+str(i)+'.csv', mydescr)

    # Fitting with (^3, ^2, ^1) fit, can use linearFit() if necessary
    zfixed = multiFit(data.x, data.z)

    # Getting the slope in every point
    dydz = np.diff(zfixed)/np.diff(data.x)
    r = 1-(1/(1+dydz**2))**(0.5)

    # Transposing each profile by a bit
    zfixed = zfixed + (i - 1)*4
    data.x = data.x + (i)*20

    # Transposing r by a bit each time
    r = r + (i - 1)*0.4

    # Plot profile
    ax1.plot(data.x, zfixed)

    # Plot r
    ax2.plot(data.x[1:], r)

# Setting appropriate labels for all subplots
ax1.set_xlabel(r"y $[  \mu m]$")
ax1.set_ylabel(r"z $[  \mu m]$")

ax2.set_xlabel(r"y $[  \mu m]$")
ax2.set_ylabel(r"r(y)")

ax2.set_ylim((0, 2))

# Saving figure to disk
plt.savefig(profile+'ublish.png', dpi=100)
plt.show()
