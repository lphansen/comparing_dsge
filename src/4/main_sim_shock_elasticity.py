import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
import os
from scipy import interpolate
from utils_sim_shock_elasticity import simulate_elasticities
import warnings
warnings.filterwarnings("ignore")
import argparse

def main(action_name="",
         nWealth=100, nZ=30, nV=30, V_bar=1.0, sigma_K_norm=0.04, sigma_Z_norm=0.0141, sigma_V_norm=0.132,
         wMin=0.01, wMax=0.99,
         chiUnderline=1.0, a_e=0.14, a_h=0.135, gamma_e=1.0, gamma_h=1.0, rho_e=1.0, rho_h=1.0,
         delta_e=1.0, delta_h=1.0, lambda_d=1.0, nu=1.0, shock_expo="",
         n_layers=5, units=16, points_size=2, iter_num=10, seed=1, penalization=10000,
         W_percentile=0.5, BFGS_maxiter=100, BFGS_maxfun=1000):

    ## Domain parameters
    domain_list = [nWealth, nZ, nV, V_bar, sigma_K_norm, sigma_Z_norm, sigma_V_norm, wMin, wMax]
    wMin_str, wMax_str = [str("{:0.3f}".format(param)).replace('.', '', 1) for param in [wMin, wMax]]
    domain_folder = f'nW_{nWealth}_nZ_{nZ}_nV_{nV}_wMin_{wMin_str}_wMax_{wMax_str}'
    nDims = 3

    ## Model parameters
    parameter_list = [chiUnderline, a_e, a_h, gamma_e, gamma_h, rho_e, rho_h, delta_e, delta_h, lambda_d, nu]
    chiUnderline_str, a_e_str, a_h_str, gamma_e_str, gamma_h_str, rho_e_str, rho_h_str, delta_e_str, delta_h_str, lambda_d_str, nu_str = \
        [str("{:0.3f}".format(param)).replace('.', '', 1) for param in parameter_list]
    model_folder = (f'chiUnderline_{chiUnderline_str}_a_e_{a_e_str}_a_h_{a_h_str}'
                    f'_gamma_e_{gamma_e_str}_gamma_h_{gamma_h_str}_rho_e_{rho_e_str}'
                    f'_rho_h_{rho_h_str}_delta_e_{delta_e_str}_delta_h_{delta_h_str}'
                    f'_lambda_d_{lambda_d_str}_nu_{nu_str}')

    ## NN layer parameters
    layer_folder = (f'seed_{seed}_n_layers_{n_layers}_units_{units}_points_size_{points_size}'
                    f'_iter_num_{iter_num}_penalization_{penalization}')
    ## Working directory
    workdir = os.getcwd()
    outputdir = workdir + '/output/' + action_name + '/' + shock_expo + '/' + domain_folder + '/' + model_folder + '/' + layer_folder + '/'
    # os.makedirs(datadir,exist_ok=True)
    # os.makedirs(outputdir,exist_ok=True)

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
    initial_point = [marginal_quantile['W'](W_percentile).item(), 0.0, marginal_quantile['V'](0.5).item()]
    elastictities_W = simulate_elasticities(statespace, shock_index, dx, dt, muX, sigmaX, mulogM, sigmalogM, mulogS, sigmalogS, initial_point, sim_length, sim_num)
    # np.savez(outputdir + '/elasticities_W_percentile_'+str(W_percentile)+'.npz',**elastictities_W)

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



