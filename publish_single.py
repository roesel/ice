#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Load single-measurement CSV file and plot profile, r and r distribution.'''

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

    pfit = np.polyfit(y, np.ones(y.size), 3)
    zbaseline = np.polyval(pfit, y)
    zfixed = z-zbaseline

    return zfixed

# -- BEGIN PARAMETERS ----
profile = '1200_3d_snp5_p2'
# ------ END PARAMETERS --

# Load data from a single CSV into array
mydescr = np.dtype([('x', 'float32'), ('z', 'float32')])  # ('y', 'float32'),
data = readArray(profile+'.csv', mydescr)

# Fitting with (^3, ^2, ^1) fit, can use linearFit() if necessary
zfixed = multiFit(data.x, data.z)

# Plot profile
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(data.x, zfixed)
ax1.set_xlabel(r"y")
ax1.set_ylabel(r"z(y)")
#plt.savefig(profile+'_p.png', dpi=100)
#plt.show()

# Getting the slope in every point
dydz = np.diff(zfixed)/np.diff(data.x)
r = 1-(1/(1+dydz**2))**(0.5)

# Plot r
#plt.figure(2)
#plt.clf()  # name figure, clear plot
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(data.x[1:], r)
ax2.set_xlabel(r"y")
ax2.set_ylabel(r"r(y)")
plt.savefig(profile+'_publish.png', dpi=100)
plt.show()

print("The mean is: "+str(np.mean(r)))
