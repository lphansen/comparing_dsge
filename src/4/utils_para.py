import tensorflow as tf 
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU') # To enable GPU acceleration, comment out this line and ensure CUDA and cuDNN libraries are properly installed

def setModelParameters(parameter_list, domain_list, nDims, datadir, triangular):

  chiUnderline, a_e, a_h, gamma_e, gamma_h, rho_e, rho_h, delta_e, delta_h, lambda_d, nu = parameter_list 
  nWealth, nZ, nV, V_bar, sigma_K_norm, sigma_Z_norm, sigma_V_norm, wMin, wMax = domain_list 

  params = {}

  ## Dimensionality params
  params['nDims'] = nDims;      
  params['nShocks'] = nDims;  
  params['numSds'] = 5;

  ## Grid parameters 
  params['nWealth'] = nWealth;  
  params['nZ']  = nZ; 
  params['nV']  = nV;

  ## Domain params
  params['wMin']              = wMin
  params['wMax']              = wMax
  params['Z_bar']             = 0.0
  params['V_bar']             = V_bar
  params['sigma_K_norm']      = sigma_K_norm
  params['sigma_Z_norm']      = sigma_Z_norm
  params['sigma_V_norm']      = sigma_V_norm

  ## Economic params
  params['chiUnderline']      = chiUnderline
  params['a_e']               = a_e
  params['a_h']               = a_h
  params['gamma_e']           = gamma_e
  params['gamma_h']           = gamma_h
  params['rho_e']             = rho_e
  params['rho_h']             = rho_h

  params['nu_newborn']        = nu
  params['lambda_d']          = lambda_d
  params['lambda_Z']          = 0.056
  params['lambda_V']          = -np.log(0.984)*12
  params['delta_e']           = delta_e
  params['delta_h']           = delta_h
  params['alpha_K']           = 0.04
  params['phi']               = 8.0
  
  ## Shock correlation params
  if triangular == 'lower_triangular':
    params['cov11'] = 1.0;    params['cov12'] = 0.0;    params['cov13'] = 0.0;   
    params['cov21'] = 0.4027386142660167;    params['cov22'] = 0.9153150324227657;    params['cov23'] = 0.0;  
    params['cov31'] = 0.0;    params['cov32'] = 0.0;    params['cov33'] = 1.0;   
  elif triangular == 'upper_triangular':
    params['cov11'] = 0.9153150324227657;    params['cov12'] = 0.4027386142660167;    params['cov13'] = 0.0;   
    params['cov21'] = 0.0;    params['cov22'] = 1.0;    params['cov23'] = 0.0;  
    params['cov31'] = 0.0;    params['cov32'] = 0.0;    params['cov33'] = 1.0;   
  
  with open(datadir + 'parameters_NN.json', 'w') as f:
      json.dump(params,f)

  NN_param_list = ['chiUnderline','a_e','a_h','gamma_e','gamma_h','delta_e','delta_h',\
                   'V_bar','Z_bar','sigma_K_norm','sigma_Z_norm','sigma_V_norm','lambda_d','lambda_Z','lambda_V',\
                   'nu_newborn','phi','rho_e','rho_h','alpha_K', 'numSds',\
                   'cov11','cov12','cov13','cov21','cov22','cov23','cov31','cov32','cov33','wMin','wMax']
  
  for i in range(len(NN_param_list)):
    params[NN_param_list[i]]             = tf.constant(params[NN_param_list[i]], dtype=tf.float64)

  ## Covariance matrices 
  params['sigmaK']                 = tf.concat([params['cov11'] * params['sigma_K_norm'],       params['cov12'] * params['sigma_K_norm'],       params['cov13'] * params['sigma_K_norm']], 0)
  params['sigmaZ']                 = tf.concat([params['cov21'] * params['sigma_Z_norm'],       params['cov22'] * params['sigma_Z_norm'],       params['cov23'] * params['sigma_Z_norm']], 0)
  params['sigmaV']                 = tf.concat([params['cov31'] * params['sigma_V_norm'],       params['cov32'] * params['sigma_V_norm'],       params['cov33'] * params['sigma_V_norm']], 0)

  ## Min and max of state variables
  ## min/max for W
  params['wMin'] = tf.constant(params['wMin'], dtype=tf.float64)
  params['wMax'] = tf.constant(params['wMax'], dtype=tf.float64)

  ## min/max for Z
  zVar  = tf.pow(tf.sqrt(params['V_bar']) * params['sigma_Z_norm'], 2) / (2 * params['lambda_Z'])
  params['zMin'] = params['Z_bar'] - params['numSds'] * tf.sqrt(zVar)
  params['zMax'] = params['Z_bar'] + params['numSds'] * tf.sqrt(zVar)

  ## min/max for V
  shape = 2 * params['lambda_V'] * params['V_bar']  /  (tf.pow(params['sigma_V_norm'],2));
  rate = 2 * params['lambda_V'] / (tf.pow(params['sigma_V_norm'],2));
  params['vMin'] = tf.constant(0.0000001, dtype=tf.float64)
  params['vMax'] = params['V_bar'] + params['numSds'] * tf.sqrt( shape / tf.pow(rate, 2));

  return params