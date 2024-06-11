import os
import time
import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator as RGI
from utils_FDM import finiteDiff_1D_first, finiteDiff_1D_second
from utils_pde_shock_elasticity import computeElas

import argparse
parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--Delta",type=float,default=1.0)
parser.add_argument("--gamma",type=float,default=8.0)
parser.add_argument("--rho",type=float,default=1.00001)
parser.add_argument("--alpha",type=float,default=0.0922)
parser.add_argument("--action_name",type=str,default="tests")
parser.add_argument("--q",type=float,default=0.05)
parser.add_argument("--twoparameter",type=int,default=1)

args = parser.parse_args()

Delta = args.Delta
gamma = args.gamma
rho = args.rho
alpha = args.alpha
action_name = args.action_name
q = args.q
twoparameter = args.twoparameter

outputdir = f"./output/{action_name}/Delta_{Delta}_twoparameter_{twoparameter}/q_{q}_gamma_{gamma}_rho_{rho}_alpha_{alpha}/"

def cal_drift_diffusion_term(logvar, zscale, zdrift, zdiffusion):

    dlogvardz = finiteDiff_1D_first(logvar,0,zscale)
    ddlogvardz = finiteDiff_1D_second(logvar,0,zscale)

    drift = dlogvardz*zdrift + 1/2*ddlogvardz*(zdiffusion[0]**2+zdiffusion[1]**2)
    diffusion = [dlogvardz*zdiffusion[i] for i in range(len(zdiffusion))]
    
    return drift, diffusion

def marginal_quantile_func_factory(dent, statespace, statename):

    inverseCDFs = {}
    
    nRange   = list(range(len(statespace)))
    for i, state in enumerate(statespace):
        axes     = list(filter(lambda x: x != i,nRange))
        condDent = dent.sum(axis = tuple(axes))
        cumden   = np.cumsum(condDent.copy())
        cdf      = interpolate.interp1d(cumden, state, fill_value= (state.min(), state.max()), bounds_error = True)
        inverseCDFs[statename[i]] = cdf
        
    return inverseCDFs

def compute_pde_shock_elasticity(statespace, dt, muX, sigmaX, mulogM, sigmalogM, mulogS, sigmalogS, initial_point, T, boundary_condition):
    """
    Computes the shock elasticity of the model using the PDE method. It uses an independent module from mfrSuite.

    Parameters:
    - statespace: list of flatten numpy arrays
        List of state variables in flattened numpy arrays for each state dimension. Grid points should be unique and sorted in ascending order.
    - dt: float or int
        Time step for the shock elasticity PDE.
    - muX: list of numpy arrays
        List of state variable drift terms as numpy arrays, evaluated at the grid points. The order should match the state variables.
    - sigmaX: list of lists of numpy arrays
        List of lists of state variable diffusion terms as numpy arrays, corresponding to different shocks. Evaluated at the grid points. The order should match the state variables.
    - mulogM: numpy array
        The log drift term for the response variable M.
    - sigmalogM: list of numpy arrays
        The log diffusion terms for the response variable M.
    - mulogS: numpy array
        The log drift term for the SDF.
    - sigmalogS: list of numpy arrays
        The log diffusion terms for the SDF.
    - initial_point: list of lists of floats or ints
        List of initial state variable points as lists for the shock elasticity.
    - T: float
        The calculation period for the shock elasticity given dt.
    - boundary_condition: dict
        The boundary condition for the shock elasticity PDE.

    Returns:
    dict
        A dictionary with the computed exposure and price elasticities.
    """
    nDims = len(statespace)
    nShocks = len(sigmalogM)

    modelInput = {}

    modelInput['T'] = T; 
    modelInput['dt'] = dt;
    modelInput['nDims'] = nDims
    modelInput['nShocks'] = nShocks

    ## Drift and Diffusion terms for the state variables
    muXs = []; 
    sigmaXs = [];
    stateVolsList = []
    for n in range(nDims):
        muXs.append(RGI(statespace,muX[n]))
        stateVols = [];
        for s in range(nShocks):
            stateVols.append(RGI(statespace, sigmaX[n][s]))
        stateVolsList.append(stateVols)
        def sigmaXfn(n):
            return lambda x: np.transpose([vol(x) for vol in stateVolsList[n] ])
        sigmaXs.append(sigmaXfn(n))
    modelInput['muX']    = lambda x: np.transpose([mu(x) for mu in muXs])
    modelInput['sigmaX'] = sigmaXs

    ## Drift and Diffusion terms for the logM and logSDF
    modelInput['muC'] = RGI(statespace, mulogM)
    modelInput['muS'] = RGI(statespace, mulogS)
    sigmalogMs = [];
    sigmalogSs = [];
    for s in range(nShocks):
        sigmalogMs.append(RGI(statespace, sigmalogM[s]))
        sigmalogSs.append(RGI(statespace, sigmalogS[s]))
    modelInput['sigmaC'] = lambda x: np.transpose([vol(x) for vol in sigmalogMs])
    modelInput['sigmaS'] = lambda x: np.transpose([vol(x) for vol in sigmalogSs])

    ## Compute the shock elasticity
    start_time = time.time()
    exposure_elasticity, price_elasticity, _, _, _ = computeElas(statespace, modelInput, boundary_condition, np.matrix(initial_point))
    print("--- %s seconds for the elasticity computation ---" % (time.time() - start_time))

    return {'exposure_elasticity':exposure_elasticity, 'price_elasticity':price_elasticity}

res = np.load(outputdir + "res.npz")

# Parameter Values
phi, beta, eta, a11, delta, alpha = res['phi'], res['beta'], res['eta'], res['a11'], res['delta'], res['alpha']
sigma_k, sigma_z = res['sigma_k'], res['sigma_z']

# Model Solution
Z = res['zz']
zscale = (np.max(Z) - np.min(Z))/(int(res['I'])-1)
logvmk = res['V']
logcmk = np.log(res['cons'])
Hk = (1-gamma)*res['hk']
Hz = (1-gamma)*res['hz']
H = [Hk, Hz]
sk = res['s1']
sz = res['s2']
S = [sk, sz]
dent = res['g']*zscale

# Compute the drift and diffusion terms from the model solution
kdrift = np.log(1+res['d']*phi)/phi + beta*Z - eta*np.ones(res['I']) - np.dot(sigma_k,sigma_k)/2
kdiffusion = [sigma_k[i]*np.ones(res['I']) for i in range(np.size(sigma_k))]      

zdrift = -a11*Z 
zdiffusion = [sigma_z[i]*np.ones(res['I']) for i in range(np.size(sigma_z))]

logcmk_drift, logcmk_diffusion = cal_drift_diffusion_term(logcmk, zscale, zdrift, zdiffusion)
logvmk_drift, logvmk_diffusion = cal_drift_diffusion_term(logvmk, zscale, zdrift, zdiffusion)

logc_drift = logcmk_drift + kdrift
logc_diffusion = [logcmk_diffusion[i] + kdiffusion[i] for i in range(len(zdiffusion))]

loghsdrift = - 0.5*((Hk + sk)**2 + (Hz + sz)**2)
loghsdiffusion = [H[i]+S[i] for i in range(len(logc_diffusion))]

# Compute the elasticity
statespace = [np.unique(Z)]
T = 45
dt = 1
boundary_condition = {'natural':True} ## natural boundary condition (see mfrSuite Readme p32/p37 for details)
marginal_quantile = marginal_quantile_func_factory(dent, [np.unique(Z)], ['Z'])
initial_points = [[marginal_quantile['Z'](0.1)],\
                  [marginal_quantile['Z'](0.5)],\
                  [marginal_quantile['Z'](0.9)]]
muX = [zdrift]
sigmaX = [zdiffusion]

uncertainty_priceelas = compute_pde_shock_elasticity(statespace, dt, muX, sigmaX, logc_drift, logc_diffusion, loghsdrift, loghsdiffusion, initial_points, T, boundary_condition)
np.savez(os.path.join(outputdir, 'uncertainty_priceelas.npz'), **uncertainty_priceelas)