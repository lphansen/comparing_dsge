#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
#######################################
######Bansal & Yaron Example#######
#######################################
'''

from matplotlib import pyplot as plt

import numpy as np
import mfr.sem as sem
import mfr.sdm as sdm
from scipy import interpolate

#######################################
############ Parameters ###############
#######################################

## Step 0: Set parameters
muX = np.matrix('-0.021 0; 0 -0.013')
iota = np.matrix('0; -1')
sigma = np.matrix('0.00031 -0.00015 0; 0 0 -0.038')
beta0 = 0.0015; beta1 = 1.0; beta2 = 0.0;
alpha = np.matrix('0.0034 0.007 0')
delta = 0.0; gamma = 8.0;

########################################
############ Setting up ################
########################################

## Step 1: Setting up the state space, drifts, and vols.
# ----------------------------------------------------------

### Step 1.1: Initialize state space
#### Initialize state space
n1 = 100; n2 = 100; ## use 100 points in each direction
stateMat = (np.linspace(-0.01, 0.01, n1), np.linspace(0,3,n2)) ## set lower and upper limits

### Step 1.2: Create model dictionary
model = {}

### Step 1.3: Set drifts and vols of the state variables
###Create function handle for the drifts
model['muX'] = lambda x: (x + iota.T) * muX

###Create function handles for the vols
sigmaX1Func = lambda x: np.sqrt(x[:,1]) * sigma[0,:]
sigmaX2Func = lambda x: np.sqrt(x[:,1]) * sigma[1,:]
model['sigmaX'] = [sigmaX1Func, sigmaX2Func]

## Step 2: Compute stationary density
# ----------------------------------------------------------

### Step 2.1: Set up boundary conditions for stationary density

bc = {}
bc['a0'] = 0
bc['first'] = np.matrix([1, 1], 'd')
bc['second'] = np.matrix([0, 0], 'd')
bc['third'] = np.matrix([0, 0], 'd')
bc['level']  = np.matrix([0, 0], 'd')
bc['natural'] = False

### Step 2.2: Use the toolbox to compute stationary density
dent, FKmat, stateGrid = sdm.computeDent(stateMat, model, bc)


## Step 3: Compute shock elasticities, for EZ utility
# ----------------------------------------------------------

### Step 3.1: Set up consumption process
model['muC'] = lambda x: x[:,0] + beta0
model['sigmaC'] = lambda x: np.sqrt(x[:,1]) * alpha

### Step 3.2: Set up SDFs for EZ utility
###Create function handles for drift and vol of EZ utility function
v1 = - beta1 / (muX[0,0] - delta);

###Quadratuc formula
A = (1.0 - gamma) / 2.0 * ( sigma[0,:] * sigma[0,:].T )
B = -delta + muX[1,1] + (1.0 - gamma) * alpha * sigma[1,:].T + 2.0 * v1 * \
    (1.0 - gamma) / 2.0 * (sigma[0,:] * sigma[1,:].T)
D = muX[0,1] * v1 + beta2 + (1.0 - gamma) * alpha * sigma[0,:].T * v1 + \
    (1.0 - gamma) / 2.0 * ( sigma[0,:] * sigma[0,:].T * (v1 ** 2) - alpha * alpha.T)
v2 = (-B - np.sqrt(np.power(B, 2) - 4.0 * A * D)) / (2.0 * A)

alphaTilde = (1.0 - gamma) * (sigma[0,:] * v1 + sigma[1,:] * v2.item() + alpha)

model['muS'] = lambda x: (-delta - 1.0 * (beta0 + beta1 * x[:,0] + \
     beta2 * ( x[:,1] - 1.0 ) ) - (alphaTilde * alphaTilde.T).item() / 2.0 * x[:,1])
model['sigmaS'] = lambda x: np.sqrt(x[:,1]) * (alphaTilde - 1.0 * alpha)

### Step 3.3: Time settings
###Time settings
model['T'] = 120 * 3
model['dt'] = 1.0

### Step 3.4: choose starting points
#### Here, we will use the 10th, 50th, and 90th quintiles of stochastic volatility.
density = dent.reshape([n1,n2], order ='F')

#### unconditional density of growth and stochastic vol
uncondG = np.sum(density, axis=0); uncondS = np.sum(density, axis=1)

#### use interpolation to get the 10th, 50th, and 90th quintiles of stochastic vol.
#### and get the 50th quintile of growth.

#### First, get the cumulative distribution. Second, interpolate to get results
cumden  = np.cumsum( uncondG )
cdfG    = interpolate.interp1d(cumden, stateMat[0])
cumden  = np.cumsum( uncondS )
cdfS    = interpolate.interp1d(cumden, stateMat[1])

#### Now get the quintiles desired
g = np.array(cdfG(.5)); ss = cdfS([.1, .5, .9])

#### Create points x0
x0 = np.matrix(np.array([np.full([1,3],g).tolist()[0], list(ss) ])).T

#### In other words, we are choose starting points to be the 50th quintile of growth
#### and 10th, 50th, and 90th quintiles of stochastaic volatility.

### Step 3.5: Compute shock elasticities
EZresults = sem.computeElas(stateMat, model, bc, x0)

## Step 4: Compute shock elasticities, for power utility
# ----------------------------------------------------------

### Note that to get the shock elasticities, we only need to change the SDF.


model['muS'] = lambda x: (-delta - gamma * model['muC'](x))
model['sigmaS'] = lambda x: np.sqrt(x[:,1]) * (- gamma * alpha)

powerResults = sem.computeElas(stateMat, model, bc, x0)

## Step 5: Plot results
# ----------------------------------------------------------

#### First plot: exposure elasticities
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))

ax1.fill_between(range(model['T']),np.exp(3*EZresults[0].firstType[0,1,:]) -1,  \
                 np.exp(3*EZresults[0].firstType[2,1,:]) -1, alpha = 0.2)
ax1.plot(np.exp(3*EZresults[0].firstType[1,1,:]) -1)
ax1.set_title('Temporary Shock Exposure Elasticity')
ax1.set_xlabel('Quarters')

ax2.fill_between(range(model['T']),np.exp(3*EZresults[0].firstType[0,0,:]) -1,  \
                 np.exp(3*EZresults[0].firstType[2,0,:]) -1, alpha = 0.2)
ax2.plot(np.exp(3*EZresults[0].firstType[1,0,:]) -1)
ax2.set_title('Permanent Shock Exposure Elasticity')
ax2.set_xlabel('Quarters')

fig.savefig('BY_shockExpo.png')

#### Second plot: price elasaticities of the two utility functions

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))

ax1.fill_between(range(model['T']),np.exp(3*EZresults[1].firstType[0,1,:]) -1,  \
                 np.exp(3*EZresults[1].firstType[2,1,:]) -1, alpha = 0.2)
ax1.plot(np.exp(3*EZresults[1].firstType[1,1,:]) -1)
ax1.plot(np.exp(3*powerResults[1].firstType[1,1,:]) -1, 'r', linestyle='dashed')
ax1.set_title('Temporary Shock Price Elasticity')
ax1.set_xlabel('Quarters')
ax1.set_ylim(-0.02,.7)

ax2.fill_between(range(model['T']),np.exp(3*EZresults[1].firstType[0,0,:]) -1,  \
                 np.exp(3*EZresults[1].firstType[2,0,:]) -1, alpha = 0.2)
ax2.plot(np.exp(3*EZresults[1].firstType[1,0,:]) -1)
ax2.plot(np.exp(3*powerResults[1].firstType[1,0,:]) -1, 'r', linestyle='dashed')

ax2.set_title('Permanent Shock Price Elasticity')
ax2.set_xlabel('Quarters')
ax2.set_ylim(-0.02,.7)

fig.savefig('BY_shockPrice.png')
