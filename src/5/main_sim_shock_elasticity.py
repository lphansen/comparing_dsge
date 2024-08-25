import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)
import time 
import os
import pandas as pd
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from utils_sim_shock_elasticity import simulate_elasticities
import argparse

## Parameter parser
parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--nW",type=int,default=30)
parser.add_argument("--nZ",type=int,default=30)
parser.add_argument("--initial_index",type=int,default=10)
parser.add_argument("--dt",type=float,default=0.1)

parser.add_argument("--a_e",type=float,default=0.14)
parser.add_argument("--a_h",type=float,default=0.135)
parser.add_argument("--rho_e",type=float,default=1.0)
parser.add_argument("--rho_h",type=float,default=1.0)
parser.add_argument("--gamma_e",type=float,default=1.0)
parser.add_argument("--gamma_h",type=float,default=1.0)
parser.add_argument("--chiUnderline",type=float,default=1.0)
parser.add_argument("--action_name",type=str,default='')
parser.add_argument("--shock_expo",type=str,default='')
parser.add_argument("--delta_e",type=float,default=0.01)
parser.add_argument("--delta_h",type=float,default=0.01)
parser.add_argument("--lambda_d",type=float,default=0.02)
parser.add_argument("--nu",type=float,default=0.01)

args = parser.parse_args()
action_name = args.action_name
initial_index = args.initial_index

rho_e = "{:0.3f}".format(args.rho_e)
rho_h = "{:0.3f}".format(args.rho_h)
gamma_e = "{:0.3f}".format(args.gamma_e)
gamma_h = "{:0.3f}".format(args.gamma_h)
a_e = "{:0.3f}".format(args.a_e)
a_h = "{:0.3f}".format(args.a_h)
chiUnderline = "{:0.3f}".format(args.chiUnderline)
dt = "{:0.3f}".format(args.dt)
delta_e = "{:0.3f}".format(args.delta_e)
delta_h = "{:0.3f}".format(args.delta_h)
lambda_d = "{:0.3f}".format(args.lambda_d)
nu = "{:0.3f}".format(args.nu)

outputdir = ('output/' + action_name + '/' + args.shock_expo + '/dt_'+str(args.dt)+'/nW_'+str(args.nW)+'_nZ_'+str(args.nZ)+'/chiUnderline_' + chiUnderline + '/a_e_' + a_e + '_a_h_' + a_h  + '/gamma_e_' + gamma_e + '_gamma_h_' + gamma_h + '/rho_e_' + rho_e + '_rho_h_' + rho_h + '/delta_e_' + delta_e + '_delta_h_' + delta_h + '/lambda_d_' + lambda_d + '_nu_' + nu)


def read_dat(filename):
    with open(outputdir + '/'+filename+'.dat', 'r') as file:
        data = [float(line.strip()) for line in file if line.strip()]
    return pd.DataFrame(data, columns=[filename])

def read_dat_state(filename):
    with open(outputdir + '/'+filename+'.dat', 'r') as file:
        data = [float(line.strip()) for line in file if line.strip()]
    return np.unique(np.array(data))

def read_dat_drift_term(filename, statespace_shape):
    with open(outputdir + '/'+filename+'.dat', 'r') as file:
        data = [float(line.strip()) for line in file if line.strip()]
    return np.array(data).reshape(statespace_shape,order='F')

def read_dat_diffusion_term(filename, statespace_shape):
    with open(outputdir + '/'+filename+'.dat', 'r') as file:
        data = [float(line.strip()) for line in file if line.strip()]
    data = split_vector(np.array(data).reshape(-1,1),len(statespace_shape))
    return [data[:,col].reshape(statespace_shape,order='F') for col in range(data.shape[1])]

def read_txt_stationary_density(statespace_shape):
    dent = pd.read_csv(outputdir + '/dent.txt',names = ['dent']).values
    return dent.reshape(statespace_shape,order='F')

def split_vector(vector, num_parts):
    """
    Split the vector into the specified number of parts and arrange these parts side by side to form a matrix.
    
    Parameters:
    vector (np.ndarray): The NumPy vector to be split.
    num_parts (int): The number of parts to split the vector into.
    
    Returns:
    np.ndarray: The matrix formed by arranging the parts side by side.
    """
    if num_parts <= 0:
        raise ValueError("num_parts must be a positive integer.")
    
    length = vector.shape[0]
    if length % num_parts != 0:
        raise ValueError("The length of the vector must be divisible by num_parts.")
    
    part_length = length // num_parts
    parts = [vector[i*part_length:(i+1)*part_length] for i in range(num_parts)]
    
    # Stack the parts vertically and transpose to arrange them side by side
    matrix = np.hstack(parts)
    
    return matrix

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

W = read_dat_state('W')
Z = read_dat_state('Z')
nW = len(W)
nZ = len(Z)
muW = read_dat_drift_term('muw_final',[nW,nZ])
muZ = read_dat_drift_term('muZ_final',[nW,nZ])
muCe = read_dat_drift_term('muCe_final',[nW,nZ])
muCh = read_dat_drift_term('muCh_final',[nW,nZ])
muSe = read_dat_drift_term('muSe_final',[nW,nZ])
muSh = read_dat_drift_term('muSh_final',[nW,nZ])

sigmaW = read_dat_diffusion_term('sigmaw_final',[nW,nZ])
sigmaZ = read_dat_diffusion_term('sigmaZ_final',[nW,nZ])
sigmaCe = read_dat_diffusion_term('sigmaCe_final',[nW,nZ])
sigmaCh = read_dat_diffusion_term('sigmaCh_final',[nW,nZ])
sigmaSe = read_dat_diffusion_term('sigmaSe_final',[nW,nZ])
sigmaSh = read_dat_diffusion_term('sigmaSh_final',[nW,nZ])
dent = read_txt_stationary_density([nW,nZ])

marginal_quantile = marginal_quantile_func_factory(dent, [W,Z], ['W','Z'])

W_state = read_dat_drift_term('W',[nW,nZ])

mulogW = muW/W_state - 1/(2*W_state**2) *(sigmaW[0]**2+sigmaW[1]**2)
sigmalogW = [sigmaW[0]/W_state, sigmaW[1]/W_state]

dt = 1/12
sim_length = 40*12
sim_num = 90
shock_index = 0
dx = [0.01,0.01]
statespace = [np.unique(W), np.unique(Z)]
muX = [muW, muZ]
sigmaX = [sigmaW, sigmaZ]
mulogM = mulogW
sigmalogM = sigmalogW
mulogS = muSe
sigmalogS = sigmaSe

initial_points = [[marginal_quantile['W'](0.05).item(),0],\
                    [marginal_quantile['W'](0.1).item(),0],\
                    [marginal_quantile['W'](0.5).item(),0]]

print('Simulating elasticities for W')
initial_point = initial_points[initial_index]
elastictities_W = simulate_elasticities(statespace, shock_index, dx, dt, muX, sigmaX, mulogM, sigmalogM, mulogS, sigmalogS, initial_point, sim_length, sim_num)
np.savez(outputdir + '/elasticity_W_' + str(initial_index) + '.npz', **elastictities_W)
