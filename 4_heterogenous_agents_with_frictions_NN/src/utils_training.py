import numpy as np
import sys
import tensorflow as tf 
import time 
from scipy import optimize
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU') # To enable GPU acceleration, comment out this line and ensure CUDA and cuDNN libraries are properly installed

@tf.function 
def calc_var(NN, W, Z, V, params):

    """
    This function calculates equilibrium variables for the model given the neural network
    """

    X = tf.concat([W,Z,V], axis=1)
    data_generator = NN(X)
    logXiE, logXiH, kappa = data_generator[:,0:1], data_generator[:,1:2], data_generator[:,2:3]
    
    a_e = params['a_e']
    rho_e = params['rho_e']
    delta_e = params['delta_e']
    gamma_e = params['gamma_e']

    a_h = params['a_h']
    rho_h = params['rho_h']
    delta_h = params['delta_h']
    gamma_h = params['gamma_h']
    chiUnderline = params['chiUnderline']

    phi = params['phi']
    sigmaK = params['sigmaK']
    sigmaZ = params['sigmaZ']
    sigmaV = params['sigmaV']

    lambda_Z = params['lambda_Z']
    Z_bar = params['Z_bar']
    lambda_V = params['lambda_V']
    V_bar = params['V_bar']
    
    alpha_K = params['alpha_K']
    lambda_d = params['lambda_d']
    nu_newborn = params['nu_newborn']

    if (params['a_h'] > 0) and (params['chiUnderline'] < 0.0001):
        kappa        = tf.ones_like(W, dtype = tf.float64)

    xiE = tf.exp(logXiE)
    xiH = tf.exp(logXiH)

    dW_logXiE     = tf.gradients(logXiE, W)[0];      
    dZ_logXiE     = tf.gradients(logXiE, Z)[0];  
    dV_logXiE     = tf.gradients(logXiE, V)[0];    
    dW2_logXiE    = tf.gradients(dW_logXiE, W)[0];    
    dZ2_logXiE    = tf.gradients(dZ_logXiE, Z)[0];
    dV2_logXiE    = tf.gradients(dV_logXiE, V)[0];    
    dWdZ_logXiE   = tf.gradients(dW_logXiE, Z)[0];    
    dWdV_logXiE   = tf.gradients(dW_logXiE, V)[0];
    dZdV_logXiE   = tf.gradients(dZ_logXiE, V)[0];

    dW_logXiH     = tf.gradients(logXiH, W)[0];
    dZ_logXiH     = tf.gradients(logXiH, Z)[0];
    dV_logXiH     = tf.gradients(logXiH, V)[0];
    dW2_logXiH    = tf.gradients(dW_logXiH, W)[0];
    dZ2_logXiH    = tf.gradients(dZ_logXiH, Z)[0];
    dV2_logXiH    = tf.gradients(dV_logXiH, V)[0];
    dWdZ_logXiH   = tf.gradients(dW_logXiH, Z)[0];
    dWdV_logXiH   = tf.gradients(dW_logXiH, V)[0];
    dZdV_logXiH   = tf.gradients(dZ_logXiH, V)[0];

    dX_logXiE = tf.concat([dW_logXiE, dZ_logXiE, dV_logXiE], axis=1)
    dX_logXiH = tf.concat([dW_logXiH, dZ_logXiH, dV_logXiH], axis=1)
    dX2_logXiE = tf.concat([dW2_logXiE, dZ2_logXiE, dV2_logXiE], axis=1)
    dX2_logXiH = tf.concat([dW2_logXiH, dZ2_logXiH, dV2_logXiH], axis=1)
    Q = ((1-kappa) * a_h + kappa * a_e + 1/phi)/\
    ((1-W)*delta_h**(1/rho_h) * xiH**(1-1/rho_h) + W*delta_e**(1/rho_e) * xiE**(1-1/rho_e) + 1/phi)

    Qtilde = ((1-1) * a_h + 1 * a_e + 1/phi)/\
        ((1-W)*delta_h**(1/rho_h) * xiH**(1-1/rho_h) + W*delta_e**(1/rho_e) * xiE**(1-1/rho_e) + 1/phi)
    
    logQ = tf.math.log(Q)
    logQtilde = tf.math.log(Qtilde)
    dW_Q = tf.gradients(Q, W)[0]
    dZ_Q = tf.gradients(Q, Z)[0]
    dV_Q = tf.gradients(Q, V)[0]
    dX_Q = tf.concat([dW_Q, dZ_Q, dV_Q], axis=1)
    dWdZ_Q = tf.gradients(dW_Q, Z)[0]
    dWdV_Q = tf.gradients(dW_Q, V)[0]
    dZdV_Q = tf.gradients(dZ_Q, V)[0]
    dW2_Q = tf.gradients(dW_Q, W)[0]
    dZ2_Q = tf.gradients(dZ_Q, Z)[0]
    dV2_Q = tf.gradients(dV_Q, V)[0]

    dW_logQ = tf.gradients(logQ, W)[0]
    dZ_logQ = tf.gradients(logQ, Z)[0]
    dV_logQ = tf.gradients(logQ, V)[0]
    dX_logQ = tf.concat([dW_logQ, dZ_logQ, dV_logQ], axis=1)
    dW_logQtilde = tf.gradients(logQtilde, W)[0]
    dZ_logQtilde = tf.gradients(logQtilde, Z)[0]
    dV_logQtilde = tf.gradients(logQtilde, V)[0]
    dX_logQtilde = tf.concat([dW_logQtilde, dZ_logQtilde, dV_logQtilde], axis=1)

    sigmaK      = sigmaK * tf.sqrt(V)                                                                                                                                                                                  
    sigmaZ      = sigmaZ * tf.sqrt(V)                                                                                                 
    sigmaV      = sigmaV * tf.sqrt(V)  

    muK         = 0.04*Z + logQ / phi - alpha_K  - 0.5*tf.reduce_sum(sigmaK*sigmaK, axis=1, keepdims=True)                      
    muZ         = lambda_Z * (Z_bar - Z)                                                                                   
    muV         = lambda_V * (V_bar - V)          
    
    if chiUnderline == 1:
        chi         = tf.ones_like(W, dtype = tf.float64)
        DXtilde     = tf.zeros_like(W, dtype = tf.float64)
        DXiW        = tf.zeros_like(W, dtype = tf.float64)
        DXiXtilde   = tf.zeros_like(W, dtype = tf.float64)
    else:
        DXtilde = sigmaK + sigmaZ*dZ_logQtilde + sigmaV * dV_logQtilde
        DXiW = W*(1-W)*tf.reduce_sum(DXtilde**2, axis=1, keepdims=True)* ((gamma_h -1)*dW_logXiH - (gamma_e -1)*dW_logXiE) 
        DXiXtilde = W*(1-W)* (DXtilde[:,0:1] * (sigmaZ[:,0:1]* ((gamma_h -1)*dZ_logXiH - (gamma_e -1)*dZ_logXiE) + \
                                                sigmaV[:,0:1]* ((gamma_h -1)*dV_logXiH - (gamma_e -1)*dV_logXiE)) +\
                                DXtilde[:,1:2] * (sigmaZ[:,1:2]* ((gamma_h -1)*dZ_logXiH - (gamma_e -1)*dZ_logXiE)+\
                                                  sigmaV[:,1:2]* ((gamma_h -1)*dV_logXiH - (gamma_e -1)*dV_logXiE)))
        chi = -(W*(1-W)*(gamma_e-gamma_h)*tf.reduce_sum(DXtilde**2, axis = 1,keepdims=True)  - DXiXtilde)/\
                (((1-W)*gamma_e+W*gamma_h)*tf.reduce_sum(DXtilde**2, axis = 1,keepdims=True) + dW_logQtilde * DXiXtilde - DXiW)+W
        chi = tf.clip_by_value(chi, chiUnderline, 10000)

    sigmaQ = ((chi*kappa-W) * dW_logQ * sigmaK + dZ_logQ * sigmaZ + dV_logQ * sigmaV) / \
            (1 - (chi*kappa-W) * dW_logQ)
    
    sigmaR = sigmaQ + sigmaK
    sigmaRnorm2 = tf.reduce_sum(sigmaR**2, axis=1, keepdims=True)
    sigmaW = (chi*kappa-W)*sigmaR
    sigmaX = [sigmaW, sigmaZ, sigmaV]
    
    sigmaRsigmaXDerivs =sigmaR[:,0:1] * (sigmaX[0][:,0:1] * (dX_logXiH[:,0:1] * (gamma_h - 1) - dX_logXiE[:,0:1] * (gamma_e - 1))+\
                                    sigmaX[1][:,0:1] * (dX_logXiH[:,1:2] * (gamma_h - 1) - dX_logXiE[:,1:2] * (gamma_e - 1))+\
                                    sigmaX[2][:,0:1] * (dX_logXiH[:,2:3] * (gamma_h - 1) - dX_logXiE[:,2:3] * (gamma_e - 1)))+\
                    sigmaR[:,1:2] * (sigmaX[0][:,1:2] * (dX_logXiH[:,0:1] * (gamma_h - 1) - dX_logXiE[:,0:1] * (gamma_e - 1))+\
                                    sigmaX[1][:,1:2] * (dX_logXiH[:,1:2] * (gamma_h - 1) - dX_logXiE[:,1:2] * (gamma_e - 1))+\
                                    sigmaX[2][:,1:2] * (dX_logXiH[:,2:3] * (gamma_h - 1) - dX_logXiE[:,2:3] * (gamma_e - 1)))+\
                    sigmaR[:,2:3] * (sigmaX[0][:,2:3] * (dX_logXiH[:,0:1] * (gamma_h - 1) - dX_logXiE[:,0:1] * (gamma_e - 1))+\
                                    sigmaX[1][:,2:3] * (dX_logXiH[:,1:2] * (gamma_h - 1) - dX_logXiE[:,1:2] * (gamma_e - 1))+\
                                    sigmaX[2][:,2:3] * (dX_logXiH[:,2:3] * (gamma_h - 1) - dX_logXiE[:,2:3] * (gamma_e - 1)))
    
    beta_E   = chi * kappa / W                                                                                                                   
    beta_H   = (1 - kappa) / (1 - W) 

    Delta_E = gamma_e * (chi*kappa / W) * sigmaRnorm2 - gamma_h*(1-chi*kappa)/(1-W)*sigmaRnorm2 - sigmaRsigmaXDerivs
    Delta_H = chiUnderline * Delta_E - (a_e-a_h) / Q
    
    Pi_E  = gamma_e * (chi*kappa / W) * sigmaR + (gamma_e -1) * (sigmaX[0]*dX_logXiE[:,0:1] + sigmaX[1]*dX_logXiE[:,1:2] + sigmaX[2]*dX_logXiE[:,2:3])
    Pi_H  = gamma_h * (1-chi*kappa)/(1-W) * sigmaR + (gamma_h -1) * (sigmaX[0]*dX_logXiH[:,0:1] + sigmaX[1]*dX_logXiH[:,1:2] + sigmaX[2]*dX_logXiH[:,2:3])
    
    muW = W*(1-W)*(delta_h**(1/rho_h) * xiH**(1-1/rho_h) - delta_e**(1/rho_e) * xiE**(1-1/rho_e) + beta_E*Delta_E - beta_H*Delta_H) +\
     (chi*kappa-W)*tf.reduce_sum(sigmaR * (Pi_H - sigmaR),axis=1, keepdims=True)+\
    lambda_d * (nu_newborn - W)
    muX = tf.concat([muW, muZ, muV], axis=1)
    
    traceQ = dW2_Q * tf.reduce_sum(sigmaW**2, axis=1, keepdims=True) +\
        dZ2_Q * tf.reduce_sum(sigmaZ**2, axis=1, keepdims=True) +\
        dV2_Q * tf.reduce_sum(sigmaV**2, axis=1, keepdims=True) +\
        2 * dWdZ_Q * tf.reduce_sum(sigmaW * sigmaZ, axis=1, keepdims=True) +\
        2 * dWdV_Q * tf.reduce_sum(sigmaW * sigmaV, axis=1, keepdims=True) +\
        2 * dZdV_Q * tf.reduce_sum(sigmaZ * sigmaV, axis=1, keepdims=True)

    muQ = 1/Q * (tf.reduce_sum(muX*dX_Q,axis=1,  keepdims=True) + 0.5*traceQ)
    
    r = muQ + muK + tf.reduce_sum(sigmaK*sigmaQ,axis=1, keepdims=True) -  tf.reduce_sum(sigmaR*Pi_H,axis=1,  keepdims=True) - \
    (1-W)*(beta_H*Delta_H - delta_h**(1/rho_h) * xiH**(1-1/rho_h)) - \
    W*(beta_E*Delta_E - delta_e**(1/rho_e) * xiE**(1-1/rho_e))
    
    sigma_logSh = -1.0 * Pi_H
    mu_logSh = - r - 0.5 * tf.reduce_sum(sigma_logSh*sigma_logSh, axis=1, keepdims=True)
    sigma_logSe = -1.0 * Pi_E
    mu_logSe = - r - 0.5 * tf.reduce_sum(sigma_logSe*sigma_logSe, axis=1, keepdims=True)

    Ce = tf.pow(delta_e,1/rho_e) * tf.exp((1-1/rho_e)*xiE) 
    Ch = tf.pow(delta_h,1/rho_h) * tf.exp((1-1/rho_h)*xiH)
    logCe = tf.math.log(Ce)
    logCh = tf.math.log(Ch)
    logC = tf.math.log(Ce*W + Ch*(1-W))

    dW_logCe = tf.gradients(logCe, W)[0];         dZ_logCe = tf.gradients(logCe, Z)[0];         dV_logCe = tf.gradients(logCe, V)[0];      
    dW_logCh = tf.gradients(logCh, W)[0];         dZ_logCh = tf.gradients(logCh, Z)[0];         dV_logCh = tf.gradients(logCh, V)[0];      
    dW_logC = tf.gradients(logC, W)[0];           dZ_logC = tf.gradients(logC, Z)[0];           dV_logC = tf.gradients(logC, V)[0];        
    sigma_logCe = dW_logCe*sigmaW + dZ_logCe*sigmaZ + dV_logCe*sigmaV + sigmaQ + sigmaK + sigmaW/W
    sigma_logCh = dW_logCh*sigmaW + dZ_logCh*sigmaZ + dV_logCh*sigmaV + sigmaQ + sigmaK - sigmaW/(1-W)
    sigma_logC = dW_logC*sigmaW + dZ_logC*sigmaZ + dV_logC*sigmaV + sigmaQ + sigmaK

    dW2_logCe = tf.gradients(dW_logCe, W)[0];         dZ2_logCe = tf.gradients(dZ_logCe, Z)[0];         dV2_logCe = tf.gradients(dV_logCe, V)[0];      
    dW2_logCh = tf.gradients(dW_logCh, W)[0];         dZ2_logCh = tf.gradients(dZ_logCh, Z)[0];         dV2_logCh = tf.gradients(dV_logCh, V)[0];    
    dW2_logC = tf.gradients(dW_logC, W)[0];           dZ2_logC = tf.gradients(dZ_logC, Z)[0];           dV2_logC = tf.gradients(dV_logC, V)[0];      
 
    dWZ_logCe = tf.gradients(dW_logCe, Z)[0];         dZV_logCe = tf.gradients(dZ_logCe, V)[0];         dVW_logCe = tf.gradients(dV_logCe, W)[0];     
    dWZ_logCh = tf.gradients(dW_logCh, Z)[0];         dZV_logCh = tf.gradients(dZ_logCh, V)[0];         dVW_logCh = tf.gradients(dV_logCh, W)[0];      
    dWZ_logC = tf.gradients(dW_logC, Z)[0];           dZV_logC = tf.gradients(dZ_logC, V)[0];           dVW_logC = tf.gradients(dV_logC, W)[0];      

    mu_logCe = dW_logCe*muW + dZ_logCe*muZ + dV_logCe*muV +\
    0.5*(dW2_logCe* tf.reduce_sum(sigmaW * sigmaW, axis=1, keepdims=True) + dZ2_logCe*tf.reduce_sum(sigmaZ * sigmaZ, axis=1, keepdims=True)  + dV2_logCe*tf.reduce_sum(sigmaV * sigmaV, axis=1, keepdims=True) ) +\
    0.5*2*(dWZ_logCe*tf.reduce_sum(sigmaW * sigmaZ, axis=1, keepdims=True) + dZV_logCe*tf.reduce_sum(sigmaZ * sigmaV, axis=1, keepdims=True) + dVW_logCe*tf.reduce_sum(sigmaV * sigmaW, axis=1, keepdims=True)) +\
    muQ - 0.5*tf.reduce_sum(sigmaQ * sigmaQ, axis=1, keepdims=True) + muK - 0.5*tf.reduce_sum(sigmaK * sigmaK, axis=1, keepdims=True) + 1/W*muW - (1/W**2)*0.5*tf.reduce_sum(sigmaW * sigmaW, axis=1, keepdims=True)
    
    mu_logCh = dW_logCh*muW + dZ_logCh*muZ + dV_logCh*muV +\
    0.5*(dW2_logCh*tf.reduce_sum(sigmaW * sigmaW, axis=1, keepdims=True) + dZ2_logCh*tf.reduce_sum(sigmaZ * sigmaZ, axis=1, keepdims=True) + dV2_logCh*tf.reduce_sum(sigmaV * sigmaV, axis=1, keepdims=True)) +\
    0.5*2*(dWZ_logCh*tf.reduce_sum(sigmaW * sigmaZ, axis=1, keepdims=True) + dZV_logCh*tf.reduce_sum(sigmaZ * sigmaV, axis=1, keepdims=True) + dVW_logCh*tf.reduce_sum(sigmaV * sigmaW, axis=1, keepdims=True)) +\
    muQ - 0.5*tf.reduce_sum(sigmaQ * sigmaQ, axis=1, keepdims=True) + muK - 0.5*tf.reduce_sum(sigmaK * sigmaK, axis=1, keepdims=True) - 1/(1-W)*muW - (1/(1-W)**2)*0.5*tf.reduce_sum(sigmaW * sigmaW, axis=1, keepdims=True)

    mu_logC = dW_logC*muW + dZ_logC*muZ + dV_logC*muV +\
    0.5*(dW2_logC*tf.reduce_sum(sigmaW * sigmaW, axis=1, keepdims=True) + dZ2_logC*tf.reduce_sum(sigmaZ * sigmaZ, axis=1, keepdims=True) + dV2_logC*tf.reduce_sum(sigmaV * sigmaV, axis=1, keepdims=True)) +\
    0.5*2*(dWZ_logC*tf.reduce_sum(sigmaW * sigmaZ, axis=1, keepdims=True) + dZV_logC*tf.reduce_sum(sigmaZ * sigmaV, axis=1, keepdims=True) + dVW_logC*tf.reduce_sum(sigmaV * sigmaW, axis=1, keepdims=True)) +\
    muQ - 0.5*tf.reduce_sum(sigmaQ * sigmaQ, axis=1, keepdims=True) + muK - 0.5*tf.reduce_sum(sigmaK * sigmaK, axis=1, keepdims=True)

    variables = {'logXiE'    : logXiE,      
                 'logXiH'    : logXiH,       
                 'xiE'       : xiE,          
                 'xiH'         : xiH,              
                 'kappa'     : kappa,     
                 'Pi_H'      : Pi_H, 
                 'Pi_E'      : Pi_E,  
                 'Delta_E'   : Delta_E, 
                 'Delta_H'   : Delta_H,  
                 'r'         : r,
                 'Q'         : Q,
                 'sigmaR'    : sigmaR,
                 'sigmaRnorm2': sigmaRnorm2, 
                 'sigmaRsigmaXDerivs' : sigmaRsigmaXDerivs,
                 'sigmaX'    : sigmaX,
                 'sigmaW'    : sigmaW,
                 'sigmaZ'    : sigmaZ,
                 'sigmaV'    : sigmaV,
                 'muX'       : muX,
                'dX_logXiE' : dX_logXiE, 
                 'dW_logXiE' : dW_logXiE,
                 'dZ_logXiE' : dZ_logXiE,
                 'dV_logXiE' : dV_logXiE,
                 'dW2_logXiE': dW2_logXiE,
                 'dZ2_logXiE': dZ2_logXiE,
                 'dV2_logXiE': dV2_logXiE,
                 'dWdZ_logXiE' : dWdZ_logXiE,
                 'dWdV_logXiE' : dWdV_logXiE,
                 'dZdV_logXiE' : dZdV_logXiE,
                 'dX2_logXiE' : dX2_logXiE,
                 'dX_logXiH' : dX_logXiH,  
                'dW_logXiH' : dW_logXiH,
                'dZ_logXiH' : dZ_logXiH,
                'dV_logXiH' : dV_logXiH,
                'dW2_logXiH': dW2_logXiH,
                'dZ2_logXiH': dZ2_logXiH,
                'dV2_logXiH': dV2_logXiH,   
                 'dWdZ_logXiH' : dWdZ_logXiH,
                 'dWdV_logXiH' : dWdV_logXiH,
                 'dZdV_logXiH' : dZdV_logXiH,
                 'dX2_logXiH' : dX2_logXiH,
                 'sigmaK'    : sigmaK,      
                 'sigmaZ'    : sigmaZ,       
                 'sigmaV'    : sigmaV,       
                 'muK'       : muK,         
                 'muZ'       : muZ,          
                 'muV'       : muV,          
                 'chi'       : chi,
                 'sigmaQ'    : sigmaQ,             
                 'beta_E'     : beta_E,       
                 'beta_H'     : beta_H,
                 'muW'       : muW,        
                'muQ'       : muQ,
                'dX_Q'      : dX_Q,
                'dX_logQ'   : dX_logQ,
                'sigma_logSh'  : sigma_logSh,
                'mu_logSh'     : mu_logSh,
                'sigma_logSe'  : sigma_logSe,
                'mu_logSe'     : mu_logSe,
                'mu_logC'    : mu_logC,
                'sigma_logC' : sigma_logC,
                'mu_logCe'   : mu_logCe,
                'sigma_logCe': sigma_logCe,
                'mu_logCh'   : mu_logCh,
                'sigma_logCh': sigma_logCh}
    
    return variables

@tf.function 
def HJB_loss(NN, W, Z, V, params):
    
    """
    This function calculates the loss function for the experts, households HJB equations and first order conditions w.r.t kappa policy function
    """

    X = tf.concat([W,Z,V], axis=1)

    a_e = params['a_e']
    rho_e = params['rho_e']
    delta_e = params['delta_e']
    gamma_e = params['gamma_e']

    a_h = params['a_h']
    rho_h = params['rho_h']
    delta_h = params['delta_h']
    gamma_h = params['gamma_h']
    chiUnderline = params['chiUnderline']

    phi = params['phi']
    sigmaK = params['sigmaK']
    sigmaZ = params['sigmaZ']
    sigmaV = params['sigmaV']

    lambda_Z = params['lambda_Z']
    Z_bar = params['Z_bar']
    lambda_V = params['lambda_V']
    V_bar = params['V_bar']
    
    alpha_K = params['alpha_K']
    lambda_d = params['lambda_d']
    nu_newborn = params['nu_newborn']
    
    variables = calc_var(NN, W, Z, V, params)
    logXiE = variables['logXiE']
    logXiH = variables['logXiH']
    xiE = variables['xiE']
    xiH = variables['xiH']
    kappa = variables['kappa']
    Pi_H = variables['Pi_H']
    Pi_E = variables['Pi_E']
    Delta_E = variables['Delta_E']
    Delta_H = variables['Delta_H']
    r = variables['r']
    Q = variables['Q']
    sigmaR = variables['sigmaR']
    sigmaRnorm2 = variables['sigmaRnorm2']
    sigmaRsigmaXDerivs = variables['sigmaRsigmaXDerivs']
    sigmaX = variables['sigmaX']
    sigmaW = variables['sigmaW']
    sigmaZ = variables['sigmaZ']
    sigmaV = variables['sigmaV']
    muX = variables['muX']
    dX_logXiE = variables['dX_logXiE']
    dWdZ_logXiE = variables['dWdZ_logXiE']
    dWdV_logXiE = variables['dWdV_logXiE']
    dZdV_logXiE = variables['dZdV_logXiE']
    dX2_logXiE = variables['dX2_logXiE']
    dX_logXiH = variables['dX_logXiH']
    dWdZ_logXiH = variables['dWdZ_logXiH']
    dWdV_logXiH = variables['dWdV_logXiH']
    dZdV_logXiH = variables['dZdV_logXiH']
    dX2_logXiH = variables['dX2_logXiH']

    if rho_e == 1:
        HJB_E_immediate_reward =  (-logXiE + tf.math.log(delta_e)) * delta_e - delta_e +\
                                r + 1/(2*gamma_e) * (Delta_E + tf.reduce_sum(Pi_H * sigmaR, axis=1, keepdims=True))**2 / sigmaRnorm2 
    else:
        HJB_E_immediate_reward = rho_e/(1-rho_e) * delta_e**(1/rho_e) * xiE**(1-1/rho_e) - delta_e/(1-rho_e)+\
                                    r + 1/(2*gamma_e) * (Delta_E + tf.reduce_sum(Pi_H * sigmaR, axis=1, keepdims=True))**2 / sigmaRnorm2 
    
    HJB_E_drift_term = (muX[:,0:1]+(1-gamma_e)/gamma_e*tf.reduce_sum(sigmaX[0]*sigmaR, axis=1, keepdims=True)*\
        (Delta_E + tf.reduce_sum(Pi_H * sigmaR, axis=1, keepdims=True))/sigmaRnorm2)*dX_logXiE[:,0:1]+\
    (muX[:,1:2]+(1-gamma_e)/gamma_e*tf.reduce_sum(sigmaX[1]*sigmaR, axis=1, keepdims=True)*\
        (Delta_E + tf.reduce_sum(Pi_H * sigmaR, axis=1, keepdims=True))/sigmaRnorm2)*dX_logXiE[:,1:2]+\
    (muX[:,2:3]+(1-gamma_e)/gamma_e*tf.reduce_sum(sigmaX[2]*sigmaR, axis=1, keepdims=True)*\
        (Delta_E + tf.reduce_sum(Pi_H * sigmaR, axis=1, keepdims=True))/sigmaRnorm2)*dX_logXiE[:,2:3]
    
    crosspartialXiE = dWdZ_logXiE*(tf.reduce_sum(sigmaW*sigmaZ, axis = 1, keepdims = True))+\
        dWdV_logXiE*(tf.reduce_sum(sigmaW*sigmaV, axis = 1, keepdims = True))+\
        dZdV_logXiE*(tf.reduce_sum(sigmaZ*sigmaV, axis = 1, keepdims = True))
    
    
    quardraticXiE = 0.5*(1-gamma_e)/gamma_e*( \
        (sigmaX[0][:,0:1]*dX_logXiE[:,0:1] + sigmaX[1][:,0:1]*dX_logXiE[:,1:2] + sigmaX[2][:,0:1]*dX_logXiE[:,2:3])\
        *((sigmaR[:,0:1]*sigmaR[:,0:1])*(1-gamma_e)/(sigmaRnorm2)+gamma_e)\
        *(sigmaX[0][:,0:1]*dX_logXiE[:,0:1] + sigmaX[1][:,0:1]*dX_logXiE[:,1:2] + sigmaX[2][:,0:1]*dX_logXiE[:,2:3])+\
        (sigmaX[0][:,0:1]*dX_logXiE[:,0:1] + sigmaX[1][:,0:1]*dX_logXiE[:,1:2] + sigmaX[2][:,0:1]*dX_logXiE[:,2:3])\
        *(sigmaR[:,0:1]*sigmaR[:,1:2])*(1-gamma_e)/(sigmaRnorm2)\
        *(sigmaX[0][:,1:2]*dX_logXiE[:,0:1] + sigmaX[1][:,1:2]*dX_logXiE[:,1:2] + sigmaX[2][:,1:2]*dX_logXiE[:,2:3])+\
        (sigmaX[0][:,0:1]*dX_logXiE[:,0:1] + sigmaX[1][:,0:1]*dX_logXiE[:,1:2] + sigmaX[2][:,0:1]*dX_logXiE[:,2:3])\
        *(sigmaR[:,0:1]*sigmaR[:,2:3])*(1-gamma_e)/(sigmaRnorm2)\
        *(sigmaX[0][:,2:3]*dX_logXiE[:,0:1] + sigmaX[1][:,2:3]*dX_logXiE[:,1:2] + sigmaX[2][:,2:3]*dX_logXiE[:,2:3])+\
        (sigmaX[0][:,1:2]*dX_logXiE[:,0:1] + sigmaX[1][:,1:2]*dX_logXiE[:,1:2] + sigmaX[2][:,1:2]*dX_logXiE[:,2:3])\
        *(sigmaR[:,1:2]*sigmaR[:,0:1])*(1-gamma_e)/(sigmaRnorm2)\
        *(sigmaX[0][:,0:1]*dX_logXiE[:,0:1] + sigmaX[1][:,0:1]*dX_logXiE[:,1:2] + sigmaX[2][:,0:1]*dX_logXiE[:,2:3])+\
        (sigmaX[0][:,1:2]*dX_logXiE[:,0:1] + sigmaX[1][:,1:2]*dX_logXiE[:,1:2] + sigmaX[2][:,1:2]*dX_logXiE[:,2:3])\
        *((sigmaR[:,1:2]*sigmaR[:,1:2])*(1-gamma_e)/(sigmaRnorm2)+gamma_e)\
        *(sigmaX[0][:,1:2]*dX_logXiE[:,0:1] + sigmaX[1][:,1:2]*dX_logXiE[:,1:2] + sigmaX[2][:,1:2]*dX_logXiE[:,2:3])+\
        (sigmaX[0][:,1:2]*dX_logXiE[:,0:1] + sigmaX[1][:,1:2]*dX_logXiE[:,1:2] + sigmaX[2][:,1:2]*dX_logXiE[:,2:3])\
        *(sigmaR[:,1:2]*sigmaR[:,2:3])*(1-gamma_e)/(sigmaRnorm2)\
        *(sigmaX[0][:,2:3]*dX_logXiE[:,0:1] + sigmaX[1][:,2:3]*dX_logXiE[:,1:2] + sigmaX[2][:,2:3]*dX_logXiE[:,2:3])+\
        (sigmaX[0][:,2:3]*dX_logXiE[:,0:1] + sigmaX[1][:,2:3]*dX_logXiE[:,1:2] + sigmaX[2][:,2:3]*dX_logXiE[:,2:3])\
        *(sigmaR[:,2:3]*sigmaR[:,0:1])*(1-gamma_e)/(sigmaRnorm2)\
        *(sigmaX[0][:,0:1]*dX_logXiE[:,0:1] + sigmaX[1][:,0:1]*dX_logXiE[:,1:2] + sigmaX[2][:,0:1]*dX_logXiE[:,2:3])+\
        (sigmaX[0][:,2:3]*dX_logXiE[:,0:1] + sigmaX[1][:,2:3]*dX_logXiE[:,1:2] + sigmaX[2][:,2:3]*dX_logXiE[:,2:3])\
        *(sigmaR[:,2:3]*sigmaR[:,1:2])*(1-gamma_e)/(sigmaRnorm2)\
        *(sigmaX[0][:,1:2]*dX_logXiE[:,0:1] + sigmaX[1][:,1:2]*dX_logXiE[:,1:2] + sigmaX[2][:,1:2]*dX_logXiE[:,2:3])+\
        (sigmaX[0][:,2:3]*dX_logXiE[:,0:1] + sigmaX[1][:,2:3]*dX_logXiE[:,1:2] + sigmaX[2][:,2:3]*dX_logXiE[:,2:3])\
        *((sigmaR[:,2:3]*sigmaR[:,2:3])*(1-gamma_e)/(sigmaRnorm2)+gamma_e)\
        *(sigmaX[0][:,2:3]*dX_logXiE[:,0:1] + sigmaX[1][:,2:3]*dX_logXiE[:,1:2] + sigmaX[2][:,2:3]*dX_logXiE[:,2:3]))
    
    secondCoefsE = 0.5*(dX2_logXiE[:,0:1]*tf.reduce_sum(sigmaX[0]**2, axis=1, keepdims=True)+\
                            dX2_logXiE[:,1:2]*tf.reduce_sum(sigmaX[1]**2, axis=1, keepdims=True)+\
                            dX2_logXiE[:,2:3]*tf.reduce_sum(sigmaX[2]**2, axis=1, keepdims=True))
    HJB_E_diffusion_term = secondCoefsE + crosspartialXiE + quardraticXiE
        
    HJB_E = HJB_E_immediate_reward + HJB_E_drift_term + HJB_E_diffusion_term


    if rho_h == 1:
        HJB_H_immediate_reward =(-logXiH + tf.math.log(delta_h)) * delta_h - delta_h+\
        r + 1/(2*gamma_h) * tf.reduce_sum(Pi_H**2, axis=1, keepdims=True)
    else:
        HJB_H_immediate_reward = rho_h/(1-rho_h) * delta_h**(1/rho_h) * xiH**(1-1/rho_h) - delta_h/(1-rho_h)+\
        r + 1/(2*gamma_h) * tf.reduce_sum(Pi_H**2, axis=1, keepdims=True)
    
    crosspartialXiH = dWdZ_logXiH*(tf.reduce_sum(sigmaW*sigmaZ, axis = 1, keepdims = True))+\
        dWdV_logXiH*(tf.reduce_sum(sigmaW*sigmaV, axis = 1, keepdims = True))+\
        dZdV_logXiH*(tf.reduce_sum(sigmaZ*sigmaV, axis = 1, keepdims = True))
    
    HJB_H_drift_term = (muX[:,0:1]+(1-gamma_h)/gamma_h*tf.reduce_sum(sigmaX[0]*Pi_H, axis=1, keepdims=True))*dX_logXiH[:,0:1]+\
    (muX[:,1:2]+(1-gamma_h)/gamma_h*tf.reduce_sum(sigmaX[1]*Pi_H, axis=1, keepdims=True))*dX_logXiH[:,1:2]+\
    (muX[:,2:3]+(1-gamma_h)/gamma_h*tf.reduce_sum(sigmaX[2]*Pi_H, axis=1, keepdims=True))*dX_logXiH[:,2:3]

    quardraticXiH = 0.5*((1-gamma_h)/gamma_h *(\
                            (sigmaW[:,0:1]*dX_logXiH[:,0:1] + sigmaZ[:,0:1]*dX_logXiH[:,1:2] + sigmaV[:,0:1]*dX_logXiH[:,2:3])**2 +\
                            (sigmaW[:,1:2]*dX_logXiH[:,0:1] + sigmaZ[:,1:2]*dX_logXiH[:,1:2] + sigmaV[:,1:2]*dX_logXiH[:,2:3])**2 +\
                            (sigmaW[:,2:3]*dX_logXiH[:,0:1] + sigmaZ[:,2:3]*dX_logXiH[:,1:2] + sigmaV[:,2:3]*dX_logXiH[:,2:3])**2))  
    secondCoefsH = 0.5*(tf.reduce_sum(sigmaX[0]**2, axis=1, keepdims=True)*dX2_logXiH[:,0:1]+\
                        tf.reduce_sum(sigmaX[1]**2, axis=1, keepdims=True)*dX2_logXiH[:,1:2]+\
                        tf.reduce_sum(sigmaX[2]**2, axis=1, keepdims=True)*dX2_logXiH[:,2:3])
    HJB_H_diffusion_term = quardraticXiH + crosspartialXiH + secondCoefsH                         
    HJB_H = HJB_H_immediate_reward + HJB_H_drift_term + HJB_H_diffusion_term   
    
    rightTerm          = W * gamma_h * (1 - chiUnderline * kappa) * sigmaRnorm2 - \
                          (1 - W) * gamma_e * chiUnderline * kappa * sigmaRnorm2 + \
                            W * (1 - W) * (a_e - a_h) / (chiUnderline * Q) + W * (1 - W) * sigmaRsigmaXDerivs 

    kappa_min          = tf.math.minimum(1 - kappa, rightTerm) 
    
    return HJB_E, HJB_H, kappa_min 


def function_factory(model, loss, W, Z, V, params, loss_type, targets, penalization, funcname):

    """
    This function creates a function that calculates the loss and gradients for the BFGS algorithm
    """

    ## Obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n
    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # Create a function that will compute the value and gradient. This can be the function that the factory returns
    @tf.function
    def val_and_grad(params_1d):
        with tf.GradientTape() as tape:
          ## Update the parameters in the model
            assign_new_model_parameters(params_1d)
            ## Calculate the loss 
            loss_value_vec = loss(model, W, Z, V, params)
            logXiE_loss = loss_value_vec[0]
            logXiH_loss = loss_value_vec[1]
            kappa_loss = loss_value_vec[2]

            loss_value =  loss_type(logXiE_loss, targets) + loss_type(logXiH_loss, targets) + loss_type(kappa_loss, targets)*penalization

        ## Calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)
        del tape

        ## Print out iteration & loss
        f.iter.assign_add(1)
        tf.print(funcname + " Iter:", f.iter, "loss:", loss_value)
        sys.stdout.flush()

        ## Store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    def f(params_1d):
      return [vv.numpy().astype(np.float64)  for vv in val_and_grad(params_1d)]

    ## Store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f
  

def training_step_BFGS(NN_func, W, Z, V, params, targets, penalization, maxiter, maxfun, gtol, maxcor, maxls, ftol):
    """
    This function trains the neural network using the BFGS algorithm
    """

    loss_fun = tf.keras.losses.MeanSquaredError()
    NN_func_adj = function_factory(NN_func, HJB_loss, W, Z, V, params, loss_fun, targets, penalization, 'BFGS')
    init_params = tf.dynamic_stitch(NN_func_adj.idx, NN_func.trainable_variables)
    start = time.time()
    results = optimize.minimize(NN_func_adj, x0 = init_params.numpy(), method = 'L-BFGS-B', jac = True, options = {'maxiter': maxiter, 'maxfun': maxfun, 'gtol': gtol, 'maxcor': maxcor, 'maxls': maxls, 'ftol' : ftol})
    end = time.time()
    tf.print('Elapsed time {:.4f} sec'.format(end - start))
    NN_func_adj.assign_new_model_parameters(results.x)

    return results

