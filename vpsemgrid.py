#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Plot 2D profiles of a subset of an entire image
   and 3D plot of the surface. '''

# ---------------- Imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# ---------------- File importing
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


def loadGrid(profile):
    ''' Reads data from a single-image CSV file and distributes x,y,z values
        into separate arrays of picture dimensions. '''

    # Setting appropriate descriptions/type to columns for easy access
    mydescr = np.dtype([('x', 'float32'), ('y', 'float32'), ('z', 'float32')])
    data = readArray(profile+'.csv', mydescr)  # Loading the .csv file

    # Counting the number of 0.00000 in arrays to get dimensions
    ''' Note: This can possibly be done better, namely because there can be
        other zeros around in other than the first row/column. This is not
        likely to happen at all though. '''
    Ny = (0 == data['y']).sum()  # Y dimension
    Nx = (0 == data['x']).sum()  # X dimension

    # Reshaping into three separate fields of picture dimension
    xgrid = data['x'].reshape(Nx, Ny)
    ygrid = data['y'].reshape(Nx, Ny)
    zgrid = data['z'].reshape(Nx, Ny)

    return xgrid, ygrid, zgrid


# Get the data
xgrid, ygrid, zgrid = loadGrid("1200_3d_snp5_img")

# Select a subset to focus on
Nx, Ny = xgrid.shape
nxstart = 100
nxend = 300  # Have to display a subset because it's too dense
nystart = 200
nyend = 640

# Massage it a bit
zgrid_s = np.zeros(zgrid.shape)
for j in range(Nx):
    pfit = np.polyfit(ygrid[:, 0], zgrid[:, j], 3)
    zbaseline = np.polyval(pfit, ygrid[:, 0])
    zgrid_s[:, j] = zgrid[:, j]-zbaseline

# Graph it in 2d
fignum = 1
plt.close(fignum)
ax = plt.figure(fignum).gca(projection='3d')  # Set up a 3d graphics window
ax.plot_wireframe(
    xgrid[nystart:nyend, nxstart:nxend],
    ygrid[nystart:nyend, nxstart:nxend],
    zgrid_s[nystart:nyend, nxstart:nxend],
)  # Make the mesh plot
ax.set_xlabel(r'X ($\mu$m)')  # Label axes
ax.set_ylabel(r'Z ($\mu$m)')
ax.set_zlabel(r'Y ($\mu$m)')
plt.show()

# Graph it in 1d
fignum = 2
plt.close(fignum)
plt.figure(fignum)  # Set up a graphics window
for j in range(nxstart, nxend, 4):
    plt.plot(
        ygrid[:, 0],
        zgrid_s[:, j])
#    plt.plot(\
#        ygrid[nystart:nyend,0],\
#        zgrid_s[nystart:nyend,j])
plt.xlabel(r'Z ($\mu$m)')  # Label the x axis
plt.ylabel(r'Y ($\mu$m)')  # Label the y axis
plt.grid()
plt.show()  # Make python display the graph

# Save the results
np.savetxt('xgrid.txt', xgrid)
np.savetxt('ygrid.txt', ygrid)
np.savetxt('zgrid.txt', zgrid_s)
