import tensorflow as tf 
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from utils_DGM import DGMNet
import time 
import os
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator as RGI
from utils_pde_shock_elasticity import computeElas
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU') # To enable GPU acceleration, comment out this line and ensure CUDA and cuDNN libraries are properly installed
import argparse

## Parameter parser
parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--action_name",type=str,default="")
parser.add_argument("--nWealth",type=int,default=100)
parser.add_argument("--nZ",type=int,default=30)
parser.add_argument("--nV",type=int,default=30)
parser.add_argument("--V_bar",type=float,default=1.0)
parser.add_argument("--sigma_K_norm",type=float,default=0.04)
parser.add_argument("--sigma_Z_norm",type=float,default=0.0141)
parser.add_argument("--sigma_V_norm",type=float,default=0.132)
parser.add_argument("--wMin",type=float,default=0.01)
parser.add_argument("--wMax",type=float,default=0.99)

parser.add_argument("--chiUnderline",type=float,default=1.0)
parser.add_argument("--a_e",type=float,default=0.14)
parser.add_argument("--a_h",type=float,default=0.135)
parser.add_argument("--gamma_e",type=float,default=1.0)
parser.add_argument("--gamma_h",type=float,default=1.0)
parser.add_argument("--rho_e",type=float,default=1.0)
parser.add_argument("--rho_h",type=float,default=1.0)
parser.add_argument("--delta_e",type=float,default=1.0)
parser.add_argument("--delta_h",type=float,default=1.0)
parser.add_argument("--lambda_d",type=float,default=1.0)
parser.add_argument("--nu",type=float,default=1.0)
parser.add_argument("--shock_expo",type=str,default="")

parser.add_argument("--n_layers",type=int,default=5)
parser.add_argument("--units",type=int,default=16)
parser.add_argument("--points_size",type=int,default=2)
parser.add_argument("--iter_num",type=int,default=10)
parser.add_argument("--seed",type=int,default=1)
parser.add_argument("--penalization",type=int,default=10000)

parser.add_argument("--BFGS_maxiter",type=int,default=100)
parser.add_argument("--BFGS_maxfun",type=int,default=1000)
args = parser.parse_args()

action_name = args.action_name

## Domain parameters
nWealth           = args.nWealth
nZ                = args.nZ
nV                = args.nV
V_bar             = args.V_bar
sigma_K_norm      = args.sigma_K_norm
sigma_Z_norm      = args.sigma_Z_norm
sigma_V_norm      = args.sigma_V_norm
wMin              = args.wMin
wMax              = args.wMax

domain_list       = [nWealth, nZ, nV, V_bar, sigma_K_norm, sigma_Z_norm, sigma_V_norm, wMin, wMax]
wMin, wMax = [str("{:0.3f}".format(param)).replace('.', '', 1)  for param in [wMin, wMax]]
domain_folder = 'nW_' + str(nWealth) + '_nZ_' + str(nZ) + '_nV_' + str(nV) + '_wMin_' + wMin + '_wMax_' + wMax
nDims = 3

## Model parameters
chiUnderline      = args.chiUnderline
a_e               = args.a_e
a_h               = args.a_h
gamma_e           = args.gamma_e
gamma_e_val       = args.gamma_e
gamma_h           = args.gamma_h
gamma_h_val       = args.gamma_h
rho_e             = args.rho_e
rho_h             = args.rho_h
delta_e           = args.delta_e
delta_h           = args.delta_h
lambda_d          = args.lambda_d
nu                = args.nu
shock_expo        = args.shock_expo
parameter_list    = [chiUnderline, a_e, a_h, gamma_e, gamma_h, rho_e, rho_h, delta_e, delta_h, lambda_d, nu]
chiUnderline, a_e, a_h, gamma_e, gamma_h, rho_e, rho_h, delta_e, delta_h, lambda_d, nu = [str("{:0.3f}".format(param)).replace('.', '', 1)  for param in parameter_list]
model_folder = 'chiUnderline_' + chiUnderline + '_a_e_' + a_e + '_a_h_' + a_h  + '_gamma_e_' + gamma_e + '_gamma_h_' + gamma_h + '_rho_e_' + rho_e + '_rho_h_' + rho_h + '_delta_e_' + delta_e + '_delta_h_' + delta_h + '_lambda_d_' + lambda_d + '_nu_' + nu

## NN layer parameters
n_layers          = args.n_layers
units             = args.units
points_size       = args.points_size
iter_num          = args.iter_num
seed              = args.seed
penalization      = args.penalization
layer_folder =  'seed_' + str(seed) + '_n_layers_' + str(n_layers) + '_units_' + str(units) +'_points_size_' + str(points_size) + '_iter_num_' + str(iter_num) + '_penalization_' + str(penalization)

## Working directory
workdir = os.getcwd()
srcdir = workdir + '/src/'
datadir = workdir + '/data/'  + action_name + '/' + shock_expo + '/' + domain_folder + '/' + model_folder + '/'
outputdir = workdir + '/output/' + action_name + '/' + shock_expo + '/' + domain_folder + '/' + model_folder + '/' + layer_folder + '/'
os.makedirs(datadir,exist_ok=True)
os.makedirs(outputdir,exist_ok=True)

def read_npy_state(filename):
    """
    Load state variable from .npy file, return unique grid points as flattened numpy array.

    Parameters:
    filename: str
        The name of the .npy file to be loaded.

    Returns:
    numpy array
        Unique grid points as flattened numpy array.
    """
    data = np.load(outputdir + filename + '.npy')
    return np.unique(data)

def read_npy_drift_term(filename, statespace_shape):
    """
    Load drift term from .npy file, return reshaped numpy array with the same shape as the state space grid.

    Parameters:
    filename: str
        The name of the .npy file to be loaded.
    statespace_shape: list of int
        The shape of the state space grid.
    
    Returns:
    numpy array
        The drift term as a reshaped numpy array with the same shape as the state space grid.
    """
    data = np.load(outputdir + filename + '.npy')
    return data.reshape(statespace_shape, order='F')

def read_npy_diffusion_term(filename, statespace_shape):
    """
    Load diffusion term from .npy file, return the exposure to the shock as a reshaped numpy array with the same shape as the state space grid. 
    Incorporate the shock exposure to all shocks as a list of numpy arrays.

    Parameters:
    filename: str
        The name of the .npy file to be loaded.
    statespace_shape: list of int
        The shape of the state space grid.

    Returns:
    list of numpy arrays
        The exposure to all shocks as a list of numpy arrays with the same shape as the state space grid.
    """
    data = np.load(outputdir + filename + '.npy')
    return [data[:,col].reshape(statespace_shape,order='F') for col in range(data.shape[1])]

def marginal_quantile_func_factory(dent, statespace, statename):
    """
    Load stationary density from .npy file, return the marginal quantile function for each state variable.

    Parameters:
    dent: numpy array
        The stationary density as a numpy array, with the shape of the state space grid.
    statespace: list of numpy arrays
        List of state variables in numpy arrays for each state dimension. Grid points should be unique and sorted in ascending order.
    statename: list of str
        List of state variable names.
    
    Returns:
    dict
        The marginal quantile function for each state variable.
    """
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

## Load results
W = read_npy_state('W_NN')
Z = read_npy_state('Z_NN')
V = read_npy_state('V_NN')
nW = len(W)
nZ = len(Z)
nV = len(V)
muW = read_npy_drift_term('muW_NN',[nW,nZ,nV])
muZ = read_npy_drift_term('muZ_NN',[nW,nZ,nV])
muV = read_npy_drift_term('muV_NN',[nW,nZ,nV])
sigmaW = read_npy_diffusion_term('sigmaW_NN',[nW,nZ,nV])
sigmaZ = read_npy_diffusion_term('sigmaZ_NN',[nW,nZ,nV])
sigmaV = read_npy_diffusion_term('sigmaV_NN',[nW,nZ,nV])

dlogxiedw = read_npy_drift_term('dW_logXiE_NN',[nW,nZ,nV])
dlogxiedz = read_npy_drift_term('dZ_logXiE_NN',[nW,nZ,nV])
dlogxiedv = read_npy_drift_term('dV_logXiE_NN',[nW,nZ,nV])
ddlogxieddw = read_npy_drift_term('dW2_logXiE_NN',[nW,nZ,nV])
ddlogxieddz = read_npy_drift_term('dZ2_logXiE_NN',[nW,nZ,nV])
ddlogxieddv = read_npy_drift_term('dV2_logXiE_NN',[nW,nZ,nV])
ddlogxiedwdz = read_npy_drift_term('dWdZ_logXiE_NN',[nW,nZ,nV])
ddlogxiedwdv = read_npy_drift_term('dWdV_logXiE_NN',[nW,nZ,nV])
ddlogxiedzdv = read_npy_drift_term('dZdV_logXiE_NN',[nW,nZ,nV])

dlogxihdw = read_npy_drift_term('dW_logXiE_NN',[nW,nZ,nV])
dlogxihdz = read_npy_drift_term('dZ_logXiE_NN',[nW,nZ,nV])
dlogxihdv = read_npy_drift_term('dV_logXiE_NN',[nW,nZ,nV])
ddlogxihddw = read_npy_drift_term('dW2_logXiE_NN',[nW,nZ,nV])
ddlogxihddz = read_npy_drift_term('dZ2_logXiE_NN',[nW,nZ,nV])
ddlogxihddv = read_npy_drift_term('dV2_logXiE_NN',[nW,nZ,nV])
ddlogxihddwdz = read_npy_drift_term('dWdZ_logXiE_NN',[nW,nZ,nV])
ddlogxihddwdv = read_npy_drift_term('dWdV_logXiE_NN',[nW,nZ,nV])
ddlogxihddzdv = read_npy_drift_term('dZdV_logXiE_NN',[nW,nZ,nV])

logce_drift = read_npy_drift_term('mulogCe_NN',[nW,nZ,nV])
logce_diffusion = read_npy_diffusion_term('sigmalogCe_NN',[nW,nZ,nV])
logch_drift = read_npy_drift_term('mulogCh_NN',[nW,nZ,nV])
logch_diffusion = read_npy_diffusion_term('sigmalogCh_NN',[nW,nZ,nV])
logsdfe_drift = read_npy_drift_term('mulogSe_NN',[nW,nZ,nV])
logsdfe_diffusion = read_npy_diffusion_term('sigmalogSe_NN',[nW,nZ,nV])

dent = read_npy_drift_term('dent_NN',[nW,nZ,nV])

marginal_quantile = marginal_quantile_func_factory(dent, [W,Z,V], ['W','Z','V'])

## Consutruct the log drift and diffusion terms for the response variable M and the SDF (compontents of the SDF)
logve_diffusion = [dlogxiedw * sigmaW[i] + dlogxiedz * sigmaZ[i] + dlogxiedv * sigmaV[i] for i in range(len(sigmaW))]
logvh_diffusion = [dlogxihdw * sigmaW[i] + dlogxihdz * sigmaZ[i] + dlogxihdv * sigmaV[i] for i in range(len(sigmaW))]

logne_drift = -0.5*((1-gamma_e_val)**2)*np.sum([vd**2 for vd in logve_diffusion],axis=0)
logne_diffusion = [(1-gamma_e_val)*logve_diffusion[i] for i in range(len(logve_diffusion))]

lognh_drift = -0.5*((1-gamma_h_val)**2)*np.sum([vd**2 for vd in logvh_diffusion],axis=0)
lognh_diffusion = [(1-gamma_h_val)*logvh_diffusion[i] for i in range(len(logvh_diffusion))]

W_state = read_npy_drift_term('W_NN',[nW,nZ,nV])

logw_drift = muW/W_state - 1/(2*W_state**2) *(np.sum([sigmaW[i]**2 for i in range(len(sigmaW))],axis=0))
logw_diffusion = [sigmaW[i]/W_state for i in range(len(sigmaW))]

## Compute the shock elasticities
bc = {}
bc['natural'] = True ## natural boundary condition (see mfrSuite Readme p32/p37 for details)
statespace = [np.unique(W), np.unique(Z), np.unique(V)] ## statespace, unique grid points for each state variable
T = 45      ## calculation period                                                      
dt = 1      ## time step for the shock elasticity PDE
muX = [muW, muZ, muV]              ## drift terms for the state variables
sigmaX = [sigmaW, sigmaZ, sigmaV]  ## diffusion terms for the state variables

if shock_expo == 'lower_triangular':
    initial_points = np.matrix([
        [marginal_quantile['W'](0.05), 0, marginal_quantile['V'](0.5)],
        [marginal_quantile['W'](0.1), 0, marginal_quantile['V'](0.5)],
        [marginal_quantile['W'](0.5), 0, marginal_quantile['V'](0.5)]
    ])

    print("Computing the elasticity for relative wealth...")
    elasticities_logw = compute_pde_shock_elasticity(statespace, dt, muX, sigmaX, logw_drift, logw_diffusion, logsdfe_drift, logsdfe_diffusion, initial_points, T, bc)
    np.savez(os.path.join(outputdir, 'elasticity_logw.npz'), **elasticities_logw)
elif shock_expo == 'upper_triangular':
    initial_points = [[marginal_quantile['W'](0.5),0,marginal_quantile['V'](0.1)],\
                    [marginal_quantile['W'](0.5),0,marginal_quantile['V'](0.5)],\
                    [marginal_quantile['W'](0.5),0,marginal_quantile['V'](0.9)]]

    print("Computing the elasticity for relative wealth...")
    uncertaintye_priceelas = compute_pde_shock_elasticity(statespace, dt, muX, sigmaX, logce_drift, logce_diffusion, logne_drift, logne_diffusion, initial_points, T, bc)
    np.savez(os.path.join(outputdir, 'uncertaintye_priceelas.npz'), **uncertaintye_priceelas)

    print("Computing the elasticity for relative wealth...")
    uncertaintyh_priceelas = compute_pde_shock_elasticity(statespace, dt, muX, sigmaX, logch_drift, logch_diffusion, lognh_drift, lognh_diffusion, initial_points, T, bc)
    np.savez(os.path.join(outputdir, 'uncertaintyh_priceelas.npz'), **uncertaintyh_priceelas)