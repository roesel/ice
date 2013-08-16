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

    pfit = np.polyfit(y, z, 3)
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
plt.plot(data.x, zfixed)
plt.savefig(profile+'_p.png', dpi=100)
plt.show()

# Getting the slope in every point
dydz = np.diff(zfixed)/np.diff(data.x)
r = 1-(1/(1+dydz**2))**(0.5)

# Plot r
plt.figure(2)
plt.clf()  # name figure, clear plot
plt.plot(data.x[1:], r)
plt.savefig(profile+'_r.png', dpi=100)
plt.show()

print("The mean is: "+str(np.mean(r)))

hist, rhist = np.histogram(r, bins=40, range=(0, 0.25), density=True)

# Plot histogram
fig = plt.figure(3)
plt.clf()  # name figure, clear plot
plt.plot(rhist[:-1], hist)

# Creating dual X axis
ax1 = plt.subplot(111)
ax2 = ax1.twiny()

# Setting proper labels
ax1.set_xlabel(r"$r$")

# Setting up functions to convert r <---> phi
r_to_phi = lambda x: 180*np.arccos(1-x)/np.pi
phi_to_r = lambda x: 1 - np.cos((x*np.pi) / 180)

new_tick_locations = phi_to_r(np.array([5, 10, 15, 20, 25, 30, 35, 40]))

tick_function = lambda x: ["%.0f" % z for z in r_to_phi(x)]

ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(new_tick_locations))
ax2.set_xlabel(r"$\phi$")
ax2.set_xlim((0, 0.25))

# Plot and save file
plt.savefig(profile+'_h.png', dpi=100)
plt.show()
