

import os.path
import pandas as pd
from collections import OrderedDict
from datetime import datetime
import subprocess
import os

############################################################
###################Parameters of Interest###################
############################################################
params = {}
#######Parameters of the model#######
params['nu_newborn']             = 0.1;
params['lambda_d']               = 0.02;
params['lambda_Z']               = 0.252;
params['lambda_V']               = 0.156;
params['lambda_Vtilde']          = 1.38;
params['Vtilde_bar']             = 0.0;
params['Z_bar']                  = 0.0;
params['V_bar']                  = 1.0;
params['delta_e']                = 0.05;
params['delta_h']                = 0.05;
params['a_e']                    = 0.14;
params['a_h']                    = -1;  ###Any negative number means -infty
params['rho_e']                  = 1.25;
params['rho_h']                  = 1.25;
params['phi']                    = 3;
params['gamma_e']                = 1;
params['gamma_h']                = 1;
params['sigma_K_norm']           = 0.04;
params['sigma_Z_norm']           = 0#0.0141;
params['sigma_V_norm']           = 0#0.132;
params['sigma_Vtilde_norm']      = 0#0.17;
params['equityIss']              = 2;
params['hhCap']                  = 0;
params['chiUnderline']           = 0.5;
params['alpha_K']                = 0.05;
params['numSds']                 = 5

#######Parameters of the numerical algorithm#######
params['method']                 = 2;   ###1: explicit; 2: implicit
params['dt']                     = 0.1;
params['dtInner']                = 0.1;
params['tol']                    = 0.00001;
params['innerTol']               = 0.00001;
params['verbatim']               = -1
params['maxIters']               = 1000000;
params['maxItersInner']          = 2000000;
params['preLoad']                = 'zero'; ### 'zero' means do not load any previous solution as initial guess
params['folderName']             = 'defaultModel'
params['exportFreq']             = 1000000
params['CGscale']                = 1.0

#######Parameters of Pardiso#######

params['iparm_2']              = 28;  ####Number of threads
params['iparm_3']              = 0;   ####0: direct solver; 1: Enable preconditioned CG
params['iparm_28']             = 0;   ####IEEE precision; 0: 64 bit; 1: 32 bit
params['iparm_31']             = 0;   ####0: direct solver; 1: recursive solver (sym matrices only)

#######Parameters of the grid#######
params['nDims']                  = 1;
params['nShocks']                = 1;
params['nWealth']                = 100;
params['logW']                   = -1;   ### If 1, solve model on log(w) grid
params['nZ']                     = 0;  ### Program will ignore this parameter if nDims < 2 or useG = -1
params['nV']                     = 0;  ### Program will ignore this parameter if nDims < 2 or useG = 1
params['nVtilde']                = 0;   ### Program will ignore this parameter if nDims < 4
params['wMin']                   = 0.01;
params['wMax']                   = 0.99;
params['Remark']                 = 'Testing a different finite diff scheme approx. for 2nd derivs ';


params['cov11']                  = 1.0
params['cov12']                  = 0.0
params['cov13']                  = 0.0
params['cov14']                  = 0.0
params['cov21']                  = 0.0
params['cov22']                  = 1.0
params['cov23']                  = 0.0
params['cov24']                  = 0.0
params['cov31']                  = 0.0
params['cov32']                  = 0.0
params['cov33']                  = 1.0
params['cov34']                  = 0.0
params['cov41']                  = 0.0
params['cov42']                  = 0.0
params['cov43']                  = 0.0
params['cov44']                  = 1.0

############################################################
##########           Create command          ###############
############################################################

cmdCols = ['numSds', 'nDims', 'nShocks', 'nWealth', 'nZ', 'nV', 'nVtilde', 'wMin', 'wMax',
   'nu_newborn', 'lambda_d', 'lambda_Z', 'lambda_V', 'lambda_Vtilde',
   'Z_bar', 'V_bar', 'Vtilde_bar', 'rho_e', 'rho_h', 'a_e', 'a_h',
   'delta_e', 'delta_h', 'phi', 'gamma_e', 'gamma_h', 'sigma_K_norm',
   'sigma_Z_norm', 'sigma_V_norm', 'sigma_Vtilde_norm', 'equityIss',
   'hhCap','chiUnderline', 'alpha_K', 'method', 'dt', 'dtInner',
   'tol', 'innerTol', 'maxIters', 'maxItersInner', 'verbatim', 'logW',
   'iparm_2', 'iparm_3', 'iparm_28', 'iparm_31', 'folderName', 'preLoad', 'cov11',
   'cov12', 'cov13', 'cov14', 'cov21', 'cov22', 'cov23', 'cov24',
   'cov31', 'cov32', 'cov33', 'cov34', 'cov41', 'cov42', 'cov43',
   'cov44', 'exportFreq', 'CGscale']

command = "./longrunrisk " + ' '.join(['--' + key + ' ' + \
                                       str(value) for key, value in \
                                       [(x, params[x]) for x in cmdCols]] )

print('Using the following command to run model:')
print(command)

createFolder = input("\nWould you like to submit create a folder " + params['folderName'] + "? ")
if createFolder == 'y':
    os.makedirs(params['folderName'])
    print('Folder created.')
