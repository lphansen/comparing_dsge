import mfr.modelSoln as m
import numpy as np
import argparse
import os
import pickle
from scipy.interpolate import griddata
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--nW",type=int,default=30)
parser.add_argument("--nZ",type=int,default=30)
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

folder_name = ('output/' + args.action_name + '/' + args.shock_expo +'/dt_'+str(args.dt)+'/nW_'+str(args.nW)+'_nZ_'+str(args.nZ)+'/chiUnderline_' + chiUnderline + '/a_e_' + a_e + '_a_h_' + a_h  + '/gamma_e_' + gamma_e + '_gamma_h_' + gamma_h + '/rho_e_' + rho_e + '_rho_h_' + rho_h + '/delta_e_' + delta_e + '_delta_h_' + delta_h + '/lambda_d_' + lambda_d + '_nu_' + nu)

def read_dat(filename):
    with open(folder_name + '/'+filename+'.dat', 'r') as file:
        data = [float(line.strip()) for line in file if line.strip()]
    return pd.DataFrame(data, columns=[filename])

W = read_dat('W')
Z = read_dat('Z')
W_sample = W['W'].unique()
Z_sample = np.array([0]) 
W_grid, Z_grid = np.meshgrid(W_sample, Z_sample, indexing='ij')
grid_points = np.column_stack((W_grid.ravel(), Z_grid.ravel()))

print('interpolating PiE_final_capital')

PiE_final = read_dat('PiE_final')
PiE_final_capital = pd.concat([W,Z,PiE_final['PiE_final'][:len(PiE_final)//2]], axis=1)
points = PiE_final_capital[['W', 'Z']].values
values = PiE_final_capital['PiE_final'].values
PiE_final_capital = griddata(points, values, grid_points, method='linear')

with open(os.getcwd()+"/" + folder_name + "/PiE_final_capital.pkl", 'wb') as file:   
    pickle.dump(PiE_final_capital, file)

print('interpolating kappa_final')

kappa_final = read_dat('kappa_final')
kappa_final = pd.concat([W,Z,kappa_final['kappa_final']], axis=1)
points = kappa_final[['W', 'Z']].values
values = kappa_final['kappa_final'].values
kappa_final = griddata(points, values, grid_points, method='linear')

with open(os.getcwd()+"/" + folder_name + "/kappa_final.pkl", 'wb') as file:
    pickle.dump(kappa_final, file)
