'''
#######################################
######He & Krushnamurthy Example#######
#######################################
'''

import numpy as np
from matplotlib import pyplot as plt
import mfr.sem as sem
import mfr.sdm as sdm
from scipy import interpolate

'''
Step 0: Setting up inputs
'''

##############
##Parameters##
##############

muY = 0.02
sigmaY = 0.09
m = 4
lambbda = 0.6
rho = 0.04
l = 1.84
x_star = (1 - lambbda) / (1 - lambbda + m)


alpha = lambda x: np.multiply( np.power( (1 - lambbda * (np.ones([np.size(x),1]) - x) ), -1) , (x > x_star)) + np.multiply( np.power( (1+m) * x, -1) , (x <= x_star)) ;

## Step 0: Setting up

###########################
#####Initialize state######
###########################

stateMat = (np.linspace(0.00001,1 - 0.00001,10000),)

model = {};
model['muX'] = lambda x: np.multiply(x , (- l / (1+l)*rho + np.power( (alpha(x) - np.ones([np.size(x),1]) ), (2)) * np.power(sigmaY,2 ) ))
model['sigmaX'] = lambda x: np.multiply(x , ( ( (alpha(x) - np.ones([np.size(x),1])) ) * sigmaY ) )
model['sigmaX'] = [model['sigmaX']]
model['muS'] = lambda x: -(rho / (1+l) + muY - alpha(x) *(sigmaY ** 2) + 0.5 * np.power(alpha(x), (2)) *(sigmaY ** 2) )
model['sigmaS'] = lambda x: -(alpha(x) * sigmaY)


###Boundary conditions
bc = {}
bc['a0'] = 0
bc['first'] = np.matrix([1], 'd')
bc['second'] = np.matrix([0], 'd')
bc['third'] = np.matrix([0], 'd')
bc['level']  = np.matrix([0], 'd')
bc['natural'] = False

###Time settings
model['T'] = 100
model['dt'] = 1.0

###x0
dent, FKmat, stateGrid = sdm.computeDent(stateMat, model, bc, usePardiso = True)
cumden  = np.cumsum( dent )
cdf     = interpolate.interp1d(cumden, stateMat[0])
x0 = np.matrix(cdf([0.1, 0.5, 0.9])).T


## Step 1: Compute shock elasticities for the first cash flow

## Step 1.1: setting up for the first cashf low
####Aggregate consumption
model['muC'] = lambda x: np.matrix( np.tile(muY - (0.5 * (sigmaY ** 2 )), [np.size(x), 1]) )
model['sigmaC'] = lambda x: np.matrix( np.tile( sigmaY , [np.size(x), 1] ) )

#####Create matrix#####
res = sem.computeElas(stateMat, model, bc, x0, usePardiso = True)
expoElas = res[0]; priceElas = res[1]

## Step 1.2: plot results
#### First plot: exposure elasticities
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))

ax1.plot(expoElas.firstType[0,0,:])
ax1.plot(expoElas.firstType[1,0,:])
ax1.plot(expoElas.firstType[2,0,:])
ax1.set_title('Shock Exposure Elasticity (First Type)')
ax1.set_ylim(-1,1.5)
ax1.set_xlabel('Time')

ax2.plot(expoElas.secondType[0,0,:])
ax2.plot(expoElas.secondType[1,0,:])
ax2.plot(expoElas.secondType[2,0,:])
ax2.set_title('Shock Exposure Elasticity (Second Type)')
ax2.set_ylim(-1,1.5)
ax2.set_xlabel('Time')

fig.savefig('HK_expoC1.png')


#### Second plot: exposure elasticities
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))

ax1.plot(priceElas.firstType[0,0,:])
ax1.plot(priceElas.firstType[1,0,:])
ax1.plot(priceElas.firstType[2,0,:])
ax1.set_title('Shock Price Elasticity (First Type)')
ax1.set_xlabel('Time')

ax2.plot(priceElas.secondType[0,0,:])
ax2.plot(priceElas.secondType[1,0,:])
ax2.plot(priceElas.secondType[2,0,:])
ax2.set_title('Shock Price Elasticity (Second Type)')
ax2.set_xlabel('Time')
fig.savefig('HK_priceC1.png')

## Step 2: Compute shock elasticities for the second cash flow

## Step 2.1: setting up
model['muC'] = lambda x: -model['muS'](x) - rho
model['sigmaC'] = lambda x: -model['sigmaS'](x)

res = sem.computeElas(stateMat, model, bc, x0)
expoElas = res[0]; priceElas = res[1]

## Step 2.2: plot results
#### First plot: exposure elasticities
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))

ax1.plot(expoElas.firstType[0,0,:])
ax1.plot(expoElas.firstType[1,0,:])
ax1.plot(expoElas.firstType[2,0,:])
ax1.set_title('Shock Exposure Elasticity (First Type)')
#ax1.set_ylim(-1,1.5)
ax1.set_xlabel('Time')

ax2.plot(expoElas.secondType[0,0,:])
ax2.plot(expoElas.secondType[1,0,:])
ax2.plot(expoElas.secondType[2,0,:])
ax2.set_title('Shock Exposure Elasticity (Second Type)')
#ax2.set_ylim(-1,1.5)
ax2.set_xlabel('Time')

fig.savefig('HK_expoC2.png')


#### Second plot: exposure elasticities
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))

ax1.plot(priceElas.firstType[0,0,:])
ax1.plot(priceElas.firstType[1,0,:])
ax1.plot(priceElas.firstType[2,0,:])
ax1.set_title('Shock Price Elasticity (First Type)')
ax1.set_xlabel('Time')

ax2.plot(priceElas.secondType[0,0,:])
ax2.plot(priceElas.secondType[1,0,:])
ax2.plot(priceElas.secondType[2,0,:])
ax2.set_title('Shock Price Elasticity (Second Type)')
ax2.set_xlabel('Time')
fig.savefig('HK_priceC2.png')
