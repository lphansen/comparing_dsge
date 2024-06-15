import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
import os
from scipy import interpolate
from utils_sim_shock_elasticity import simulate_elasticities
import warnings
warnings.filterwarnings("ignore")
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
parser.add_argument("--W_percentile",type=float,default=0.5)
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
gamma_h           = args.gamma_h
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
dent = read_npy_drift_term('dent_NN',[nW,nZ,nV])
logsdfe_drift = read_npy_drift_term('mulogSe_NN',[nW,nZ,nV])
logsdfe_diffusion = read_npy_diffusion_term('sigmalogSe_NN',[nW,nZ,nV])

marginal_quantile = marginal_quantile_func_factory(dent, [W,Z,V], ['W','Z','V'])
W_state = read_npy_drift_term('W_NN',[nW,nZ,nV])

## Consutruct the log drift and diffusion terms for the response variable M
mulogW = muW/W_state - 1/(2*W_state**2) *(sigmaW[0]**2+sigmaW[1]**2+sigmaW[2]**2)
sigmalogW = [sigmaW[0]/W_state, sigmaW[1]/W_state, sigmaW[2]/W_state]

## Simulate elasticities
sim_num = 100                       ## number of simulations, actual number of simulations is sim_num*100
sim_length = 50                     ## simulation length
shock_index = 0                     ## shock index, 0 for the capital shock
dx = [0.01,0.01,1e-7]               ## state derivative 
dt = 1                              ## time step
statespace = [W,Z,V]                ## state space
muX = [muW, muZ, muV]               ## state variable drift terms
sigmaX = [sigmaW, sigmaZ, sigmaV]   ## state variable diffusion terms
mulogM = mulogW                     ## log drift term for the response variable M
sigmalogM = sigmalogW               ## log diffusion terms for the response variable M
mulogS = logsdfe_drift              ## log drift term for the SDF
sigmalogS = logsdfe_diffusion       ## log diffusion terms for the SDF

print('Simulating elasticities for W')
initial_point = [marginal_quantile['W'](args.W_percentile).item(), 0.0, marginal_quantile['V'](0.5).item()]
elastictities_W = simulate_elasticities(statespace, shock_index, dx, dt, muX, sigmaX, mulogM, sigmalogM, mulogS, sigmalogS, initial_point, sim_length, sim_num)
np.savez(outputdir + '/elasticities_W_percentile_'+str(args.W_percentile)+'.npz',**elastictities_W)

