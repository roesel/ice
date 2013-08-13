#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Plot histogram of an entire 3D image and fit it with Weibull.''' 

# ---------------- Imports
from matplotlib.pyplot import * #figure, clf, semilogy, grid, legend, subplot, text, savefig, show
from scipy.optimize import leastsq

import numpy as np

# ---------------- Statistics
def pWeibull(r,sigma,eta):
    ''' Weibull function to be fit. '''
    
    from numpy import exp
    
    mu = 1-r
    return 2*eta/sigma**2/mu**3 * ( ((mu**(-2)-1)/sigma**2)**(eta-1) )*exp(-((mu**(-2)-1)/sigma**2)**eta)

def residuals(p, y, r):
    ''' Error function for fitting. '''
    
    from numpy import log
    
    sigma = p[0]
    eta = p[1]
    err = log(y)-log(pWeibull(r,sigma,eta))
    return err
    
# ---------------- File importing
def readArray(filename, dtype, separator=','):
    ''' (not ours) Read a file with an arbitrary number of columns. The type of data in 
        each column is also arbitrary - it will be cast to the given dtype 
        at runtime. '''
            
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
    data = readArray(profile+'.csv', mydescr) # Loading the .csv file
    
    # Counting the number of 0.00000 in arrays to get dimensions
    ''' Note: This can possibly be done better, namely because there can be
        other zeros around in other than the first row/column. This is not
        likely to happen at all though. '''
    Ny = (0 == data['y']).sum() # Y dimension
    Nx = (0 == data['x']).sum() # X dimension
    
    # Reshaping into three separate fields of picture dimension
    xgrid = data['x'].reshape(Nx,Ny)
    ygrid = data['y'].reshape(Nx,Ny)
    zgrid = data['z'].reshape(Nx,Ny)
    
    return xgrid, ygrid, zgrid

# ---------------- Histogram
def getR(profile, linemin, linemax):
    ''' Reads every line of z(y), eliminates slope for each line separately 
        and computes the r value. Returns all results of r in one list. '''
    
    # Fetching grid from file
    xgrid, ygrid, zgrid = loadGrid(profile)
    
    # Transposing y and z, because we're doing vertical measurements
    yT = np.transpose(ygrid)
    zT = np.transpose(zgrid)
    
    # Going through all lines
    for i in range(linemin, linemax):
        # Selecting the i-th line from each transposed array
        y = yT[i]
        z = zT[i]
        
        # Fitting with linearly generated sequence
        A = np.array([ y, np.ones(y.size)])
        w = np.linalg.lstsq(A.T,z)[0] # obtaining the parameters
        zline=w[0]*y+w[1]
        zfixed = z-zline # substracting baseline from every point 
    
        # Getting the slope in every point 
        dydz = np.diff(zfixed)/np.diff(y) #Note: try reversing this? dz/dy?
        r = 1-(1/(1+dydz**2))**(0.5)
        
        # If first line, start r_total, else append
        if i==linemin:
            r_total = r
        else:
            r_total = np.concatenate((r_total,r))
    
    return r_total 

def getHistogram(filebase, linemin, linemax, bins, rangemax):
    ''' Filebase is the name of the folder, where the CSVs are. 
        Num is the number of CSVs that are to be parsed and put together.
        Bins is the number of bins to use when making a histogram. Returns 
        spacing (labels), hist (values), data (all r values in one list).'''
    
    # Getting all r values from file
    data = getR(filebase, linemin, linemax)
    
    # Creating the histogram    
    hist, rhist = np.histogram(data, bins=bins, range=(0, rangemax), density=True)
    
    # Making our own spacing - middle of all intervals.
    spacing = rhist[1:]-(rhist[1]-rhist[0])/2
    
    return spacing, hist, data  

# -- BEGIN PARAMETERS ----
bins = 25
rangemax = 0.25
namebase = '1200_3d_snp5_img'
linemin = 10
linemax = 50
# ------ END PARAMETERS --

# Get histogram for set number of bins 
labels, values, data = getHistogram(namebase, linemin, linemax, bins, rangemax)

# If there are gaps, find the maximum possible number of bins to not get them
while not (values > 0).all():
    bins = bins-1
    labels, values, data = getHistogram(namebase, linemin, linemax, bins, rangemax)
    if bins==5:
        print('WARNING: Ideal bins are lower than 5, printing them anyway.')
        break

print("Number of bins determined as: " + str(bins))
print("Completed histogram from "+str(data.size)+" r values.")

total_mean = np.mean(data)
print("The total mean is: "+str(total_mean)+".")

# Setting estimated values for sigma and eta
sigma_0 = .2
eta_0 = 1.0
p0 = ([sigma_0 , eta_0]) # initial set of parameters
plsq = leastsq(residuals, p0, args=(values, labels), maxfev=200) #actual fit

# Report sigma and eta in commandline
sigma_ret = plsq[0][0]
eta_ret = plsq[0][1]
print('Sigma is: '+str(sigma_ret))
print('Eta is: '+str(eta_ret))

# Plot it
figure(1); clf()
semilogy(labels,values,labels,pWeibull(labels,sigma_ret, eta_ret)) # log scale
grid()
legend(('Expt','Best fit'))

# Creating dual X axis
ax1 = subplot(111)
ax2 = ax1.twiny()

# Setting proper labels
ax1.set_xlabel(r"$r$")

# Setting up functions to convert r <---> phi
r_to_phi = lambda x: 180*np.arccos(1-x)/np.pi
phi_to_r = lambda x: 1 - np.cos( (x*np.pi) / 180 )

new_tick_locations = phi_to_r(np.array([5, 10, 15, 20, 25, 30, 35, 40]))

def tick_function(x):
    V = r_to_phi(x)
    return ["%.0f" % z for z in V]

ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(new_tick_locations))
ax2.set_xlabel(r"$\phi$")
ax2.set_xlim((0, rangemax))

# Printing sigma and eta onto the plot
text(0.87, 0.82,
     r'$\sigma$: %.3f $\eta$: %.3f' % (sigma_ret,eta_ret),
     fontsize=12,
     horizontalalignment='center',
     verticalalignment='center',
     transform=ax1.transAxes)

# Printing bins and mean onto the plot     
text(0.84, 0.77,
     r'mean: %.3f bins: %i' % (total_mean,bins),
     fontsize=12,
     horizontalalignment='center',
     verticalalignment='center',
     transform=ax1.transAxes)

#Saving figure to disk
savefig(namebase+'weib_comp.png', dpi=100)

show()