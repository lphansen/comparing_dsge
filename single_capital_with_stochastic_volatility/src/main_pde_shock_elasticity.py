import os
import time
import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator as RGI
from utils_FDM import finiteDiff_2D_first, finiteDiff_2D_second, finiteDiff_2D_cross
from utils_pde_shock_elasticity import computeElas
import argparse

parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--Delta",type=float,default=1.)
parser.add_argument("--delta",type=float,default=0.1)
parser.add_argument("--gamma",type=float,default=8.0)
parser.add_argument("--rho",type=float,default=1.00001)
parser.add_argument("--alpha",type=float,default=1.0)
parser.add_argument("--action_name",type=str,default="tests")
args = parser.parse_args()

Delta = args.Delta
delta = args.delta
gamma = args.gamma
rho = args.rho
alpha = args.alpha
action_name = args.action_name

outputdir = f"./output/{args.action_name}/Delta_{args.Delta}/delta_{args.delta}/gamma_{args.gamma}_rho_{args.rho}_alpha_{args.alpha}/"

def cal_drift_diffusion_term(logvar, zscale, yscale, zdrift, zdiffusion, ydrift, ydiffusion):

    dlogvardz = finiteDiff_2D_first(logvar,0,zscale)
    ddlogvardz = finiteDiff_2D_second(logvar,0,zscale)
    dlogvardy = finiteDiff_2D_first(logvar,1,yscale)
    ddlogvardy = finiteDiff_2D_second(logvar,1,yscale)
    ddlogvardzdy = finiteDiff_2D_cross(logvar,zscale,yscale)

    drift = dlogvardz*zdrift+dlogvardy*ydrift +\
    1/2*ddlogvardz*(zdiffusion[0]**2+zdiffusion[1]**2+zdiffusion[2]**2) + \
    1/2*ddlogvardy*(ydiffusion[0]**2+ydiffusion[1]**2+ydiffusion[2]**2) + \
    1/2*2*ddlogvardzdy*(zdiffusion[0]*ydiffusion[0]+zdiffusion[1]*ydiffusion[1]+zdiffusion[2]*ydiffusion[2])

    diffusion = [dlogvardz*zdiffusion[i]+dlogvardy*ydiffusion[i] for i in range(len(ydiffusion))]
    
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
phi, beta, eta, a11, a22, alpha = res['phi'], res['beta'], res['eta'], res['a11'], res['a22'], res['alpha']
sigma_k, sigma_z, sigma_y = np.array([res['sigma_k']]), np.array([res['sigma_z']]), np.array([res['sigma_y']])

# Model Solution
Z, Y = res['zz'], res['yy']
zscale = (np.max(Z) - np.min(Z))/(res['I']-1)
yscale = (np.max(Y) - np.min(Y))/(res['J']-1)
logvmk = res['V']
logcmk = np.log(res['cons'])
logimo = np.log(res['d']/alpha)
dent = res['g']*zscale*yscale

# Compute the drift and diffusion terms from the model solution
zdrift = -a11 * Z
ydrift = -a22 * (Y - res['ymean'])
kdrift = np.log(1+res['d']*phi)/phi + beta*Z - eta*np.ones(res['I']) -  np.sum(sigma_k**2)/2*Y

zdiffusion = [sigma_z[0,i]*np.sqrt(Y) for i in range(np.size(sigma_z,1))]
ydiffusion = [sigma_y[0,i]*np.sqrt(Y) for i in range(np.size(sigma_y,1))]
kdiffusion = [sigma_k[0,i]*np.sqrt(Y) for i in range(np.size(sigma_k,1))]

logcmk_drift, logcmk_diffusion = cal_drift_diffusion_term(logcmk, zscale, yscale, zdrift, zdiffusion, ydrift, ydiffusion)
logimo_drift, logimo_diffusion = cal_drift_diffusion_term(logimo, zscale, yscale, zdrift, zdiffusion, ydrift, ydiffusion)
logvmk_drift, logvmk_diffusion = cal_drift_diffusion_term(logvmk, zscale, yscale, zdrift, zdiffusion, ydrift, ydiffusion)

logc_drift = logcmk_drift + kdrift
logv_drift = logvmk_drift + kdrift
logc_diffusion = [logcmk_diffusion[i] + kdiffusion[i] for i in range(len(ydiffusion))]
logv_diffusion = [logvmk_diffusion[i] + kdiffusion[i] for i in range(len(ydiffusion))]

logsdfdrift = - delta - rho * logc_drift + (gamma-1)*(rho-gamma)/2*np.sum([vd**2 for vd in logv_diffusion],axis=0)
logsdfdiffusion = [-rho*logc_diffusion[i]+(rho-gamma)*logv_diffusion[i] for i in range(len(logc_diffusion))]

logndrift = -0.5*((1-gamma)**2)*np.sum([vd**2 for vd in logv_diffusion],axis=0)
logndiffusion = [(1-gamma)*logv_diffusion[i] for i in range(len(logc_diffusion))]

# Compute the elasticity
statespace = [np.unique(Z),np.unique(Y)]
T = 45
dt = 1
boundary_condition = {'natural':True} ## natural boundary condition (see mfrSuite Readme p32/p37 for details)
marginal_quantile = marginal_quantile_func_factory(dent, [np.unique(Z), np.unique(Y)], ['Z','Y'])
initial_points = [[marginal_quantile['Z'](0.5),marginal_quantile['Y'](0.1)],
                  [marginal_quantile['Z'](0.5),marginal_quantile['Y'](0.5)],
                  [marginal_quantile['Z'](0.5),marginal_quantile['Y'](0.9)]]

muX = [zdrift, ydrift]
sigmaX = [zdiffusion, ydiffusion]

elasticities_logimo = compute_pde_shock_elasticity(statespace, dt, muX, sigmaX, logimo_drift, logimo_diffusion, logsdfdrift, logsdfdiffusion, initial_points, T, boundary_condition)
np.savez(os.path.join(outputdir, 'elasticity_logimo.npz'), **elasticities_logimo)

elasticities_logc = compute_pde_shock_elasticity(statespace, dt, muX, sigmaX, logc_drift, logc_diffusion, logsdfdrift, logsdfdiffusion, initial_points, T, boundary_condition)
np.savez(os.path.join(outputdir, 'elasticity_logc.npz'), **elasticities_logc)

uncertainty_priceelas = compute_pde_shock_elasticity(statespace, dt, muX, sigmaX, logc_drift, logc_diffusion, logndrift, logndiffusion, initial_points, T, boundary_condition)
np.savez(os.path.join(outputdir, 'uncertainty_priceelas.npz'), **uncertainty_priceelas)
