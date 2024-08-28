import os
import time
import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator as RGI
from utils_FDM import finiteDiff_3D
from utils_pde_shock_elasticity import computeElas
import argparse

parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--Delta",type=float,default=1.)
parser.add_argument("--gamma",type=float,default=8.0)
parser.add_argument("--rho",type=float,default=1.00001)
parser.add_argument("--kappa",type=float,default=0.0)
parser.add_argument("--zeta",type=float,default=0.5)
parser.add_argument("--alpha",type=float,default=1.0)
parser.add_argument("--beta1",type=float,default=0.04)
parser.add_argument("--beta2",type=float,default=0.04)
parser.add_argument("--action_name",type=str,default="tests")
args = parser.parse_args()

Delta = args.Delta
gamma = args.gamma
rho = args.rho
alpha = args.alpha
kappa = args.kappa
zeta = args.zeta
beta1 = args.beta1
beta2 = args.beta2
action_name = args.action_name

outputdir = "./output/"+action_name+"/Delta_"+str(Delta)+'/beta1_'+str(beta1)+'_beta2_'+str(beta2)+"/kappa_"+str(kappa)+"_zeta_"+str(zeta)+"/gamma_"+str(gamma)+"_rho_"+str(rho)+'_alpha_'+str(alpha)+"/"

def cal_drift_diffusion_term(logvar, rscale, zscale, sscale, rdrift, rdiffusion, zdrift, zdiffusion, sdrift, sdiffusion):

    dlogvardr = finiteDiff_3D(logvar,0,1,rscale)
    ddlogvardr = finiteDiff_3D(logvar,0,2,rscale)
    dlogvardz = finiteDiff_3D(logvar,1,1,zscale)
    ddlogvardz = finiteDiff_3D(logvar,1,2,zscale)
    dlogvards = finiteDiff_3D(logvar,2,1,sscale)
    ddlogvards = finiteDiff_3D(logvar,2,2,sscale)
    ddlogvardrdz = finiteDiff_3D(dlogvardr,1,1,zscale)
    ddlogvardrds = finiteDiff_3D(dlogvardr,2,1,sscale)
    ddlogvardzds = finiteDiff_3D(dlogvardz,2,1,sscale)
    
    drift = dlogvardr*rdrift+dlogvardz*zdrift + dlogvards*sdrift +\
    1/2*ddlogvardr*(rdiffusion[0]**2+rdiffusion[1]**2+rdiffusion[2]**2+rdiffusion[3]**2) + \
    1/2*ddlogvardz*(zdiffusion[0]**2+zdiffusion[1]**2+zdiffusion[2]**2+zdiffusion[3]**2) + \
    1/2*ddlogvards*(sdiffusion[0]**2+sdiffusion[1]**2+sdiffusion[2]**2+sdiffusion[3]**2) + \
    1/2*2*ddlogvardrdz*(rdiffusion[0]*zdiffusion[0]+rdiffusion[1]*zdiffusion[1]+rdiffusion[2]*zdiffusion[2]+rdiffusion[3]*zdiffusion[3]) + \
    1/2*2*ddlogvardrds*(rdiffusion[0]*sdiffusion[0]+rdiffusion[1]*sdiffusion[1]+rdiffusion[2]*sdiffusion[2]+rdiffusion[3]*sdiffusion[3]) + \
    1/2*2*ddlogvardzds*(zdiffusion[0]*sdiffusion[0]+zdiffusion[1]*sdiffusion[1]+zdiffusion[2]*sdiffusion[2]+zdiffusion[3]*sdiffusion[3])
    diffusion = [dlogvardr*rdiffusion[i]+dlogvardz*zdiffusion[i]+dlogvards*sdiffusion[i] for i in range(len(rdiffusion))]

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
    For the details of the PDE method, see the mfrSuite Readme file p34.
    
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
delta, phi1, phi2, beta1, beta2, eta1, eta2, a11, a22, alpha = res['delta'], res['phi1'], res['phi2'], res['beta1'], res['beta2'], res['eta1'], res['eta2'], res['a11'], res['a22'], res['alpha']
sigma_k1, sigma_k2, sigma_z, sigma_s = res['sigma_k1'], res['sigma_k2'], res['sigma_z'], res['sigma_s']

# Model Solution
ll, zz, ss = res['rrr'], res['zzz'], res['sss']
rscale = (np.max(ll) - np.min(ll))/(int(res['I'])-1)
zscale = (np.max(zz) - np.min(zz))/(int(res['J'])-1)
sscale = (np.max(ss) - np.min(ss))/(int(res['S'])-1)
logvmk = res['V']
logcmk = np.log(res['cons'])
logimo = np.log(1-res['cons']/res['alpha'])
dent = res['g']*rscale*zscale*sscale
k1a, k2a = res['k1a'], res['k2a']
dkadk1dk1 = (kappa-1)*(1-zeta)**2*(k1a)**(-2*kappa+2) - kappa*(1-zeta)*(k1a)**(-kappa+1);
dkadk1dk2 = (kappa-1)*zeta*(1-zeta)*(k1a)**(-kappa+1)*(k2a)**(-kappa+1);
dkadk2dk2 = (kappa-1)*zeta**2*(k2a)**(-2*kappa+2) - kappa*(1-zeta)*(k2a)**(-kappa+1)

# Compute the drift and diffusion terms from the model solution
k1_drift = np.log(1+phi1*res['d1'])/phi1 + beta1*zz - eta1*np.ones([res['I'],res['J'],res['S']])
k2_drift = np.log(1+phi2*res['d2'])/phi2 + beta2*zz - eta2*np.ones([res['I'],res['J'],res['S']])
ldrift = k2_drift - k1_drift - ss/2*(np.sum(sigma_k2**2) - np.sum(sigma_k1**2))

ldiffusion = [(sigma_k2 - sigma_k1)[i]*np.ones([res['I'],res['J'],res['S']])*np.sqrt(ss) for i in range(np.size(sigma_k1,0))]

kdrift = k1_drift*(1-zeta)*(k1a)**(1-kappa)+ k2_drift*(zeta)*(k2a)**(1-kappa)+ \
            ss/2*(np.sum(sigma_k1**2)*dkadk1dk1 + np.sum(sigma_k2**2)*dkadk2dk2 + \
            2*np.sum(sigma_k1*sigma_k2)*dkadk1dk2)
kdiffusion = [(sigma_k1[i]*(1-zeta)*(k1a)**(1-kappa) + sigma_k2[i]*(zeta)*(k2a)**(1-kappa))*np.sqrt(ss) for i in range(np.size(sigma_k1,0))]

zdrift = -a11*zz
zdiffusion = [sigma_z[i]*np.sqrt(ss) for i in range(np.size(sigma_z,0))]

sdrift = -a22*(ss-res['smean'])
sdiffusion = [sigma_s[i]*np.sqrt(ss) for i in range(np.size(sigma_s,0))]

logcmk_drift, logcmk_diffusion = cal_drift_diffusion_term(logcmk, rscale, zscale, sscale, ldrift, ldiffusion, zdrift, zdiffusion, sdrift, sdiffusion)
logimo_drift, logimo_diffusion = cal_drift_diffusion_term(logimo, rscale, zscale, sscale, ldrift, ldiffusion, zdrift, zdiffusion, sdrift, sdiffusion)
logvmk_drift, logvmk_diffusion = cal_drift_diffusion_term(logvmk, rscale, zscale, sscale, ldrift, ldiffusion, zdrift, zdiffusion, sdrift, sdiffusion)

logc_drift = logcmk_drift + kdrift
logv_drift = logvmk_drift + kdrift
logc_diffusion = [logcmk_diffusion[i] + kdiffusion[i] for i in range(len(logcmk_diffusion))]
logv_diffusion = [logvmk_diffusion[i] + kdiffusion[i] for i in range(len(logcmk_diffusion))]

logsdfdrift = - delta - rho * logc_drift + (gamma-1)*(rho-gamma)/2*np.sum([vd**2 for vd in logv_diffusion],axis=0)
logsdfdiffusion = [-rho*logc_diffusion[i]+(rho-gamma)*logv_diffusion[i] for i in range(len(logc_diffusion))]

# Compute the elasticity
statespace = [np.unique(ll), np.unique(zz), np.unique(ss)]
T = 45
dt = 1
bc = {'natural':True} ## natural boundary condition (see mfrSuite Readme p32/p37 for details)
marginal_quantile = marginal_quantile_func_factory(dent, statespace, ['l','z','s'])
initial_points = [[marginal_quantile['l'](0.5),marginal_quantile['z'](0.5),marginal_quantile['s'](0.5)]]

muX = [ldrift, zdrift, sdrift]
sigmaX = [ldiffusion, zdiffusion, sdiffusion]

print("Computing the elasticity for investment over output ratio...")
elasticities_logimo = compute_pde_shock_elasticity(statespace, dt, muX, sigmaX, logimo_drift, logimo_diffusion, logsdfdrift, logsdfdiffusion, initial_points, T, bc)
np.savez(os.path.join(outputdir, 'elasticity_logimo.npz'), **elasticities_logimo)

print("Computing the elasticity for comsumption...")
elasticities_logc = compute_pde_shock_elasticity(statespace, dt, muX, sigmaX, logc_drift, logc_diffusion, logsdfdrift, logsdfdiffusion, initial_points, T, bc)
np.savez(os.path.join(outputdir, 'elasticity_logc.npz'), **elasticities_logc)