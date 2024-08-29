import numpy as np
import tensorflow as tf
import os
from utils_pde_stationary_density import computeDent
from utils_para import setModelParameters
from utils_training import HJB_loss, calc_var
from scipy import interpolate
from utils_DGM import DGMNet
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU')  # To enable GPU acceleration, comment out this line and ensure CUDA and cuDNN libraries are properly installed
import argparse

def main_var(action_name, nWealth, nZ, nV, V_bar, sigma_K_norm, sigma_Z_norm, sigma_V_norm, wMin, wMax, chiUnderline, a_e, a_h, gamma_e, gamma_h, rho_e, rho_h, delta_e, delta_h, lambda_d, nu, shock_expo, n_layers, units, points_size, iter_num, seed, penalization, BFGS_maxiter, BFGS_maxfun):
    ## Domain parameters
    domain_list = [nWealth, nZ, nV, V_bar, sigma_K_norm, sigma_Z_norm, sigma_V_norm, wMin, wMax]
    wMin_str, wMax_str = [str("{:0.3f}".format(param)).replace('.', '', 1) for param in [wMin, wMax]]
    domain_folder = f'nW_{nWealth}_nZ_{nZ}_nV_{nV}_wMin_{wMin_str}_wMax_{wMax_str}'
    nDims = 3

    ## Model parameters
    parameter_list = [chiUnderline, a_e, a_h, gamma_e, gamma_h, rho_e, rho_h, delta_e, delta_h, lambda_d, nu]
    chiUnderline_str, a_e_str, a_h_str, gamma_e_str, gamma_h_str, rho_e_str, rho_h_str, delta_e_str, delta_h_str, lambda_d_str, nu_str = [str("{:0.3f}".format(param)).replace('.', '', 1) for param in parameter_list]
    model_folder = f'chiUnderline_{chiUnderline_str}_a_e_{a_e_str}_a_h_{a_h_str}_gamma_e_{gamma_e_str}_gamma_h_{gamma_h_str}_rho_e_{rho_e_str}_rho_h_{rho_h_str}_delta_e_{delta_e_str}_delta_h_{delta_h_str}_lambda_d_{lambda_d_str}_nu_{nu_str}'

    ## NN layer parameters
    layer_folder = f'seed_{seed}_n_layers_{n_layers}_units_{units}_points_size_{points_size}_iter_num_{iter_num}_penalization_{penalization}'

    ## Working directory
    workdir = os.getcwd()
    datadir = f'{workdir}/data/{action_name}/{shock_expo}/{domain_folder}/{model_folder}/'
    outputdir = f'{workdir}/output/{action_name}/{shock_expo}/{domain_folder}/{model_folder}/{layer_folder}/'
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(outputdir, exist_ok=True)

    ## Load parameter set
    params = setModelParameters(parameter_list, domain_list, nDims, datadir, shock_expo)

    ## Load trained NNs back from memory
    sim_NN_tf = tf.saved_model.load(outputdir + 'sim_NN')

    ## Calculate and dump equilibrium variables
    W_NN, Z_NN, V_NN, dent_NN = dump_variables(sim_NN_tf, params, nWealth, nZ, nV, outputdir)

    ## Compute and save marginal quantiles
    compute_and_save_quantiles(sim_NN_tf, params, nWealth, nZ, nV, outputdir,W_NN, Z_NN, V_NN, dent_NN)

def dump_variables(sim_NN_tf, params, nWealth, nZ, nV, outputdir):
    total_points = nWealth * nZ * nV 
    W_NN = np.tile(np.linspace(params['wMin'], params['wMax'], num=nWealth), int(total_points/nWealth))
    Z_NN = np.tile(np.repeat(np.linspace(params['zMin'], params['zMax'], num=nZ), nWealth), nV)
    V_NN = np.repeat(np.linspace(params['vMin'], params['vMax'], num=nV), int(total_points/nV))

    variables = calc_var(sim_NN_tf, W_NN.reshape(-1,1), Z_NN.reshape(-1,1), V_NN.reshape(-1,1), params)
    
    logXiE_NN       = variables['logXiE'].numpy();  
    logXiH_NN       = variables['logXiH'].numpy();  
    XiE_NN          = np.exp(logXiE_NN)
    XiH_NN          = np.exp(logXiH_NN)
    kappa_NN        = variables['kappa'].numpy()
    q_NN            = variables['Q'].numpy();            
    sigmaK_NN       = variables['sigmaK'].numpy();    
    sigmaZ_NN       = variables['sigmaZ'].numpy();    
    sigmaV_NN       = variables['sigmaV'].numpy();      
    dX_logXiE_NN    = variables['dX_logXiE'].numpy(); 
    dX_logXiH_NN    = variables['dX_logXiH'].numpy(); 
    dX2_logXiE_NN   = variables['dX2_logXiE'].numpy(); 
    dX2_logXiH_NN   = variables['dX2_logXiH'].numpy(); 
    dW_logXiE_NN    = variables['dW_logXiE'].numpy();
    dZ_logXiE_NN    = variables['dZ_logXiE'].numpy();
    dV_logXiE_NN    = variables['dV_logXiE'].numpy();
    dW_logXiH_NN    = variables['dW_logXiH'].numpy();
    dZ_logXiH_NN    = variables['dZ_logXiH'].numpy();
    dV_logXiH_NN    = variables['dV_logXiH'].numpy();
    dW2_logXiE_NN   = variables['dW2_logXiE'].numpy();
    dZ2_logXiE_NN   = variables['dZ2_logXiE'].numpy();
    dV2_logXiE_NN   = variables['dV2_logXiE'].numpy();
    dW2_logXiH_NN   = variables['dW2_logXiH'].numpy();
    dZ2_logXiH_NN   = variables['dZ2_logXiH'].numpy();
    dV2_logXiH_NN   = variables['dV2_logXiH'].numpy();
    dWdZ_logXiE_NN  = variables['dWdZ_logXiE'].numpy();
    dWdZ_logXiH_NN  = variables['dWdZ_logXiH'].numpy();
    dWdV_logXiE_NN  = variables['dWdV_logXiE'].numpy();
    dWdV_logXiH_NN  = variables['dWdV_logXiH'].numpy();
    dZdV_logXiE_NN  = variables['dZdV_logXiE'].numpy();
    dZdV_logXiH_NN  = variables['dZdV_logXiH'].numpy(); 
    muK_NN          = variables['muK'].numpy();        
    muZ_NN          = variables['muZ'].numpy();          
    muV_NN          = variables['muV'].numpy(); 
    mulogSe_NN      = variables['mu_logSe'].numpy();      
    mulogSh_NN      = variables['mu_logSh'].numpy();    
    sigmalogSe_NN   = variables['sigma_logSe'].numpy();
    sigmalogSh_NN   = variables['sigma_logSh'].numpy();  
    mulogCe_NN      = variables['mu_logCe'].numpy();      
    mulogCh_NN      = variables['mu_logCh'].numpy();    
    sigmalogCe_NN   = variables['sigma_logCe'].numpy();
    sigmalogCh_NN   = variables['sigma_logCh'].numpy();        
    mulogC_NN       = variables['mu_logC'].numpy();   
    sigmalogC_NN    = variables['sigma_logC'].numpy();  
    chi_NN          = variables['chi'].numpy(); 
    sigmaQ_NN       = variables['sigmaQ'].numpy();  
    sigmaR_NN       = variables['sigmaR'].numpy();    
    sigmaW_NN       = variables['sigmaW'].numpy();    
    deltaE_NN       = variables['Delta_E'].numpy();     
    deltaH_NN       = variables['Delta_H'].numpy();
    PiH_NN          = variables['Pi_H'].numpy();        
    PiE_NN          = variables['Pi_E'].numpy();          
    betaE_NN        = variables['beta_E'].numpy();      
    betaH_NN        = variables['beta_H'].numpy(); 
    muW_NN          = variables['muW'].numpy();        
    muQ_NN          = variables['muQ'].numpy();         
    muX_NN          = np.matrix(variables['muX']);       
    sigmaX_NN       = [np.matrix(el) for el in variables['sigmaX']];  
    r_NN            = variables['r'].numpy();                                   

    HJB_E_NN, HJB_H_NN, kappa_min_NN = HJB_loss(sim_NN_tf,W_NN.reshape(-1,1), Z_NN.reshape(-1,1),V_NN.reshape(-1,1), params)

    HJBE_validation_MSE = tf.reduce_mean(tf.square(HJB_E_NN)).numpy()
    HJBH_validation_MSE = tf.reduce_mean(tf.square(HJB_H_NN)).numpy()
    kappa_validation_MSE = tf.reduce_mean(tf.square(kappa_min_NN)).numpy()

    tf.print('HJB Experts validation MSE: ', HJBE_validation_MSE)
    tf.print('HJB Households validation MSE: ', HJBH_validation_MSE)
    tf.print('FOC kappa validation MSE: ', kappa_validation_MSE)

    HJB_E_NN        = HJB_E_NN.numpy();
    HJB_H_NN        = HJB_H_NN.numpy();
    kappa_min_NN    = kappa_min_NN.numpy()

    bc = {}
    bc['a0']  = 0
    bc['first'] = np.matrix([1,1,1], 'd')
    bc['second'] = np.matrix([0,0,0], 'd')
    bc['third'] = np.matrix([0,0,0], 'd')
    bc['level'] = np.matrix([0,0,0])
    bc['natural'] = False

    dent_NN, _, _ = computeDent(np.matrix(np.array([W_NN,Z_NN,V_NN]).transpose()), {'muX': muX_NN, 'sigmaX': sigmaX_NN}, bc = bc)

    dump_list = ['W_NN', 'Z_NN', 'V_NN', 'logXiE_NN', 'logXiH_NN', 'XiE_NN', 'XiH_NN', 'kappa_NN', 'q_NN', 'sigmaK_NN',
                 'sigmaZ_NN', 'sigmaV_NN', 'muK_NN', 'muZ_NN', 'muV_NN', 'chi_NN', 'muW_NN', 'muQ_NN', 'r_NN',
                 'sigmaW_NN', 'sigmaQ_NN', 'sigmaR_NN', 'deltaE_NN', 'deltaH_NN', 'PiH_NN', 'PiE_NN', 'betaE_NN',
                 'betaH_NN', 'HJB_E_NN', 'HJB_H_NN', 'kappa_min_NN', 'HJBE_validation_MSE', 'HJBH_validation_MSE',
                 'kappa_validation_MSE', 'dent_NN', 'dX_logXiE_NN', 'dX_logXiH_NN', 'dX2_logXiE_NN', 'dX2_logXiH_NN',
                 'dWdZ_logXiE_NN', 'dWdZ_logXiH_NN', 'dWdV_logXiE_NN', 'dWdV_logXiH_NN', 'dZdV_logXiE_NN', 'dZdV_logXiH_NN',
                 'dW_logXiE_NN', 'dZ_logXiE_NN', 'dV_logXiE_NN', 'dW_logXiH_NN', 'dZ_logXiH_NN', 'dV_logXiH_NN',
                 'dW2_logXiE_NN', 'dZ2_logXiE_NN', 'dV2_logXiE_NN', 'dW2_logXiH_NN', 'dZ2_logXiH_NN', 'dV2_logXiH_NN',
                 'mulogSe_NN', 'mulogSh_NN', 'sigmalogSe_NN', 'sigmalogSh_NN', 'mulogCe_NN', 'mulogCh_NN', 'sigmalogCe_NN',
                 'sigmalogCh_NN', 'mulogC_NN', 'sigmalogC_NN']

    for var_name in dump_list:
        np.save(outputdir + var_name, eval(var_name, globals(), locals()))
    return W_NN, Z_NN, V_NN, dent_NN

def compute_and_save_quantiles(sim_NN_tf, params, nWealth, nZ, nV, outputdir,W_NN, Z_NN, V_NN, dent_NN):

    marginal_quantile = marginal_quantile_func_factory(dent_NN.reshape([nWealth, nZ, nV], order = 'F'), [np.unique(W_NN),np.unique(Z_NN),np.unique(V_NN)], ['W','Z','V'])

    compute_and_save_evaluation(sim_NN_tf, params, marginal_quantile, W_NN, Z_NN, V_NN, dent_NN, outputdir)

def compute_and_save_evaluation(sim_NN_tf, params, marginal_quantile, W_NN, Z_NN, V_NN, dent_NN, outputdir):
    num = 1800
    W = np.linspace(np.min(W_NN), np.max(W_NN), num)
    Z = np.zeros(num)

    for quantile in [0.1, 0.5, 0.9]:
        V = np.ones(num) * marginal_quantile['V'](quantile)
        variables = calc_var(sim_NN_tf, W.reshape(-1, 1), Z.reshape(-1, 1), V.reshape(-1, 1), params)

        kappa_NN = variables['kappa'].numpy()
        chi_NN = variables['chi'].numpy()
        PiH_NN = variables['Pi_H'].numpy()
        PiE_NN = variables['Pi_E'].numpy()
        
        tf.print(PiE_NN[:, 0].mean())
        np.savez(outputdir + f'/eva_V_{int(quantile * 100)}.npz', kappa_NN=kappa_NN, chi_NN=chi_NN, PiH_NN=PiH_NN, PiE_NN=PiE_NN, W=W, Z=Z, V=V)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameter settings")
    parser.add_argument("--action_name", type=str, default="")
    parser.add_argument("--nWealth", type=int, default=100)
    parser.add_argument("--nZ", type=int, default=30)
    parser.add_argument("--nV", type=int, default=30)
    parser.add_argument("--V_bar", type=float, default=1.0)
    parser.add_argument("--sigma_K_norm", type=float, default=0.04)
    parser.add_argument("--sigma_Z_norm", type=float, default=0.0141)
    parser.add_argument("--sigma_V_norm", type=float, default=0.132)
    parser.add_argument("--wMin", type=float, default=0.01)
    parser.add_argument("--wMax", type=float, default=0.99)
    parser.add_argument("--chiUnderline", type=float, default=1.0)
    parser.add_argument("--a_e", type=float, default=0.14)
    parser.add_argument("--a_h", type=float, default=0.135)
    parser.add_argument("--gamma_e", type=float, default=1.0)
    parser.add_argument("--gamma_h", type=float, default=1.0)
    parser.add_argument("--rho_e", type=float, default=1.0)
    parser.add_argument("--rho_h", type=float, default=1.0)
    parser.add_argument("--delta_e", type=float, default=1.0)
    parser.add_argument("--delta_h", type=float, default=1.0)
    parser.add_argument("--lambda_d", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--shock_expo", type=str, default="")
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--units", type=int, default=16)
    parser.add_argument("--points_size", type=int, default=2)
    parser.add_argument("--iter_num", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--penalization", type=int, default=10000)
    parser.add_argument("--BFGS_maxiter", type=int, default=100)
    parser.add_argument("--BFGS_maxfun", type=int, default=1000)
    args = parser.parse_args()

    main_var(
        args.action_name, args.nWealth, args.nZ, args.nV, args.V_bar, args.sigma_K_norm, args.sigma_Z_norm, args.sigma_V_norm,
        args.wMin, args.wMax, args.chiUnderline, args.a_e, args.a_h, args.gamma_e, args.gamma_h, args.rho_e, args.rho_h,
        args.delta_e, args.delta_h, args.lambda_d, args.nu, args.shock_expo, args.n_layers, args.units, args.points_size,
        args.iter_num, args.seed, args.penalization, args.BFGS_maxiter, args.BFGS_maxfun
    )
