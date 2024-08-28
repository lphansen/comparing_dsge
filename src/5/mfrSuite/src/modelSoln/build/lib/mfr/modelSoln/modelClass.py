#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This is the main program that interfaces with C++ via pybind11 for
the model solution developed in Hansen, Khorrami, and Tourre (working paper).


"""

###################################################
####### Load dependencies  ########################
###################################################

## Load dependent packages

### Python numerical packages
import modelSolnCore as m
from scipy.special import comb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate
import math
from numpy import linalg as LA
from numba import jit


## Python graphics packages
import plotly as py
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
from plotly import tools
import plotly.io as pio

## Other python packages
import copy
import warnings
import os
import itertools
from collections import OrderedDict, Counter
import functools
import json
from ipywidgets import widgets, Layout, Button, HBox, VBox, interactive
import matplotlib.pyplot as plt

### MFM packages
import mfr.sdm as sdm
import mfr.sem as sem

## ================================================================= ##
## ================================================================= ##
## ================================================================= ##

## Section 1: It contains the universal items, such as default parameters,
## dictionaries for variable, symbol, and latex conversions.

########################
## Default parameters ##
########################

paramsDefault = OrderedDict({});

#######Parameters of the model#######
paramsDefault['nu_newborn']             = 0.1;
paramsDefault['lambda_d']               = 0.02;
paramsDefault['lambda_Z']               = 0.252;
paramsDefault['lambda_V']               = 0.156;
paramsDefault['lambda_Vtilde']        = 1.38;
paramsDefault['Vtilde_bar']             = 0.0;
paramsDefault['Z_bar']                  = 0.0;
paramsDefault['V_bar']                  = 1.0;
paramsDefault['delta_e']                = 0.05;
paramsDefault['delta_h']                = 0.05;
paramsDefault['a_e']                    = 0.14;
paramsDefault['a_h']                    = -1;  ###Any negative number means -infty
paramsDefault['rho_e']                  = 1;
paramsDefault['rho_h']                  = 1;
paramsDefault['phi']                    = 3;
paramsDefault['gamma_e']                = 1;
paramsDefault['gamma_h']                = 1;
paramsDefault['sigma_K_norm']             = 0.04;
paramsDefault['sigma_Z_norm']           = 0.0;
paramsDefault['sigma_V_norm']           = 0.0;
paramsDefault['sigma_Vtilde_norm']      = 0.0;
paramsDefault['equityIss']              = 2;
paramsDefault['chiUnderline']           = 0.5;
paramsDefault['alpha_K']                = 0.05;

#######Parameters of the numerical algorithm#######
paramsDefault['method']                 = 2;   ###1: explicit; 2: implicit
paramsDefault['dt']                     = 0.1;
paramsDefault['dtInner']                = 0.1;
paramsDefault['tol']                    = 0.00001;
paramsDefault['innerTol']               = 0.00001;
paramsDefault['verbatim']               = -1
paramsDefault['maxIters']               = 4000;
paramsDefault['maxItersInner']          = 2000000;

#######Parameters of Pardiso#######

## Note: these parameters are disabled for now by default. For advanced users,
## you may activate them by modifying the C++ program to include Pardiso.

paramsDefault['iparm_2']              = 28;  ####Number of threads
paramsDefault['iparm_3']              = 0;   ####0: direct solver; 1: Enable preconditioned CG
paramsDefault['iparm_28']             = 0;   ####IEEE precision; 0: 64 bit; 1: 32 bit
paramsDefault['iparm_31']             = 0;   ####0: direct solver; 1: recursive solver (sym matrices only)

#######Parameters of the grid#######
paramsDefault['numSds']                 = 5;
paramsDefault['nWealth']                = 100;
paramsDefault['logW']                   = -1;  ### If 1, solve model on log(w) grid
paramsDefault['nZ']                     = 0;   ### Program will ignore this parameter if nDims < 2 or useG = -1
paramsDefault['nV']                     = 0;   ### Program will ignore this parameter if nDims < 2 or useG = 1
paramsDefault['nVtilde']                = 0;   ### Program will ignore this parameter if nDims < 4
paramsDefault['wMin']                   = 0.01;
paramsDefault['wMax']                   = 0.99;

#######Parameters of IO#######
paramsDefault['folderName']             = 'model0'
paramsDefault['overwrite']              = 'Yes'
paramsDefault['exportFreq']             = 10000
paramsDefault['CGscale']                = 1.0
paramsDefault['precondFreq']            = 1.0

####################################
## Conversion dictionaries        ##
####################################

dictOrder = ['nu_newborn', 'lambda_d', 'lambda_Z', 'lambda_V', \
'lambda_Vtilde', 'Z_bar', 'V_bar', 'Vtilde_bar', \
'sigma_K_norm', 'sigma_Z_norm', 'sigma_V_norm', 'sigma_Vtilde_norm',\
'nWealth', 'nZ','nV','nVtilde', 'nDims',\
'delta_e', 'delta_h', 'a_e', 'a_h', 'rho_e', 'rho_h', 'phi', 'gamma_e', 'gamma_h',\
'equityIss', 'hhCap', 'chiUnderline', 'method','dt','dtInner',\
'tol', 'innerTol', 'maxIters', 'maxItersInner', 'iparm_2',
'iparm_3', 'iparm_28', 'iparm_31', 'numSds', 'wMin', 'wMax', 'logW', \
'folderName', 'overwrite', 'verbatim', 'exportFreq']

label2Var = {'Risk Price (Experts): TFP Shock':'piE1()', 'Risk Price (Households): TFP Shock':'piH1()',
                     'Experts\' Effective Leverage':'leverage()', 'Experts\' Equity Retention' : 'chi()',
                     'Consumption-Wealth Ratio (Experts)' : 'chatE()',
                     'Consumption-Wealth Ratio (Households)' : 'chatH()',
                     'Wealth Share Drift' : 'muW()',
                     'Experts\' Bonus Risk Premium': 'deltaE()',
                     'Experts\' Share of Capital': 'kappa()',
                     'Risk Free Rate': 'r()', 'Wealth Share':'W()', 'Capital Price' : 'q()',
                     'Value Function of Experts': 'xiE()', 'Value Function of Households': 'xiH()',
                     'Conditional Expected GDP Growth' : 'muY()', 'Conditional GDP Growth Volatility' : 'sigmaYNorm()',
                     'Aggregate Consumption-to-Investment Ratio': 'CoverI()',
                     'Aggregate Investment-to-Capital Ratio': 'IoverK()',
                     'Aggregate Consumption-to-Capital Ratio' : 'CoverK()',
                     'Aggregate Investment-to-Output Ratio' : 'IoverY()',
                     'Aggregate Consumption-to-Output Ratio' : 'CoverY()',
                     'Wealth Share Diffusion: Coordinate 1' : 'sigmaW1()',
                     'Return on Capital Diffusion: Coordinate 1' : 'sigmaR1()',
                     'Conditional Expected Excess Return on Capital (Experts)':'excessReturnKExperts()',
                     'Conditional Expected Excess Return on Capital (Households)':'excessReturnKHouseholds()',
                     'Idiosyncratic Risk Price (Experts)' : 'piETilde()',
                     'Idiosyncratic Risk Price (Households)' : 'piHTilde()',
                     'Risk Price Diff. (Experts - Households): TFP Shock' : 'piDiff1()',
                     'Experts\' Consumption to Households\' Consumption Ratio' : 'CeOverCh()'}
var2Label = {'piE1()':'Risk Price (Experts): TFP Shock','piH1()':'Risk Price (Households): TFP Shock',
                     'leverage()':'Experts\' Effective Leverage',  'muW()':'Wealth Share Drift',
                      'chatE()': 'Consumption-Wealth Ratio (Experts)',
                      'chatH()': 'Consumption-Wealth Ratio (Households)',
                      'chi()': 'Experts\' Equity Retention',
                      'deltaE()': 'Experts\' Bonus Risk Premium',
                      'kappa()': 'Experts\' Share of Capital',
                      'r()': 'Risk Free Rate','W()': 'Wealth Share', 'q()': 'Capital Price',
                      'xiE()' : 'Value Function of Experts', 'xiH()' : 'Value Function of Households',
                      'muY()' : 'Conditional Expected GDP Growth', 'sigmaYNorm()' : 'Conditional GDP Growth Volatility',
                      'CoverI()' : 'Aggregate Consumption-to-Investment Ratio',
                      'IoverK()' : 'Aggregate Investment-to-Capital Ratio',
                      'CoverK()' : 'Aggregate Consumption-to-Capital Ratio',
                      'IoverY()' : 'Aggregate Investment-to-Output Ratio',
                      'CoverY()' : 'Aggregate Consumption-to-Output Ratio',
                      'sigmaW1()' : 'Wealth Share Diffusion: Coordinate 1',
                      'sigmaR1()' : 'Return on Capital Diffusion: Coordinate 1',
                      'excessReturnKExperts()':'Conditional Expected Excess Return on Capital (Experts)',
                      'excessReturnKHouseholds()':'Conditional Expected Excess Return on Capital (Households)',
                      'piETilde()' : 'Idiosyncratic Risk Price (Experts)',
                      'piHTilde()' : 'Idiosyncratic Risk Price (Households)',
                      'piDiff1()'  : 'Risk Price Diff. (Experts - Households): TFP Shock',
                      'CeOverCh()' : 'Experts\' Consumption to Households\' Consumption Ratio'}
label2Sym = {'Risk Price (Experts): TFP Shock':'\pi_e^{1}', 'Risk Price (Households): TFP Shock':'\pi_h^{1}',
                                         'Experts\' Effective Leverage':'\kappa\chi/W', 'Experts\' Equity Retention' : '\chi',
                                         'Consumption-Wealth Ratio (Experts)' : '\hat{c}_e',
                                         'Consumption-Wealth Ratio (Households)' : '\hat{c}_h',
                                         'Wealth Share Drift' : '\mu_W',
                                         'Experts\' Bonus Risk Premium': '\Delta_e',
                                         'Experts\' Share of Capital': '\kappa',
                                         'Risk Free Rate': 'r', 'Wealth Share':'W',
                                         'Exogenous Growth':'Z','Agg. Stochastic Variance':'V','Idio. Stochastic Variance':'\widetilde{V}',
                                         'Value Function of Experts': '\ksi_e',
                                         'Value Function of Households': '\ksi_h',
                                         'Capital Price' : 'q', 'Conditional Expected GDP Growth' : '\mu_Y',
                                         'Conditional GDP Growth Volatility' : '||\sigma_Y||^2',
                                         'Aggregate Consumption-to-Investment Ratio' : 'C/I',
                                         'Aggregate Investment-to-Capital Ratio' : 'I/K',
                                         'Aggregate Consumption-to-Capital Ratio' : 'C/K',
                                         'Aggregate Investment-to-Output Ratio' : 'I/Y',
                                         'Aggregate Consumption-to-Output Ratio' : 'C/Y',
                                         'Wealth Share Diffusion: Coordinate 1' : '\sigma_W^{1}',
                                         'Return on Capital Diffusion: Coordinate 1' : '\sigma_R^{1}',
                                         'Conditional Expected Excess Return on Capital (Experts)' : '\mu_{R_e} - r',
                                         'Conditional Expected Excess Return on Capital (Households)' : '\mu_{R_h} - r',
                                         'Idiosyncratic Risk Price (Experts)' : '\widetilde{\pi}_e',
                                         'Idiosyncratic Risk Price (Households)' : '\widetilde{\pi}_h',
                                         'Risk Price Diff. (Experts - Households): TFP Shock' : '\pi_e^{1} - \pi_h^{1}',
                                         'Experts\' Consumption to Households\' Consumption Ratio' : 'C_e / C_h'}

macroMomentsList = ['Consumption-Wealth Ratio (Experts)', 'Consumption-Wealth Ratio (Households)', \
'Aggregate Consumption-to-Investment Ratio', 'Wealth Share', \
'Exogenous Growth', 'Idio. Stochastic Variance', 'Agg. Stochastic Variance', 'Conditional GDP Growth Volatility',\
'Conditional Expected GDP Growth', 'Aggregate Investment-to-Capital Ratio', 'Aggregate Consumption-to-Capital Ratio',\
'Aggregate Investment-to-Output Ratio', 'Aggregate Consumption-to-Output Ratio',\
'Experts\' Effective Leverage', 'Experts\' Share of Capital','Experts\' Equity Retention']
apMomentsList    = [x for x in list(label2Sym.keys()) if x not in macroMomentsList]

## ================================================================= ##
## ================================================================= ##
## ================================================================= ##

## Section 2: This is the main section of the modelSoln package. It contains
## the definition of the class.

###################################################
####### Start of class model    ###################
###################################################

class Model(m.model):

    def __init__(self, params):
        self.status        =  -2; # -3: user termination; -2: unattempted; # -1: error when solving; # 0: tol not met; # 1: tol met
        self.params        = {}
        self.S             = None
        self.gridSizeList  = None
        self.stateVarList  = None

        ## Initialize model
        self.initializeModel(params)

        self.sigmaXList = None;
        self.stateMat   = pd.DataFrame()

        ### Stationary density and shock elasticities
        ##### self.dent stores the stationary density; self.FKmat stores the
        ##### Feynman Kac matrix used to solve the density.
        self.dent = None; self.FKmat = None;
        self.marginals = {};
        self.inverseCDFs = {};
        self.macroMoments = {};
        self.apMoments = {};
        self.corrs   = OrderedDict({});
        self.model = None; self.stateMatInput = None;
        self.expoElas = None; self.priceElasExperts = None; self.priceElasHouseholds = None;
        self.expoElasMap = {}; self.priceElasExpertsMap = {}; self.priceElasHouseholdsMap = {};
        self.x0 = None;
        self.linSysExpo = None; self.linSysE = None; self.linSysH = None;
        self.pcts = None;

        ### Labels (for plotting, correlation dictionary, etc.)
        self.label2Var = label2Var
        self.var2Label = var2Label
        self.stateVar2Label = {'W': 'Wealth Share', 'Z': 'Exogenous Growth', 'V': 'Agg. Stochastic Variance',
                          'Vtilde': 'Idio. Stochastic Variance'}
        self.label2stateVar = {'Wealth Share' : 'W', 'Exogenous Growth' : 'Z', 'Agg. Stochastic Variance' : 'V',
                          'Idio. Stochastic Variance' : 'Vtilde'}
        self.perturbLabel2Var = {'Aggregate Consumption':'C', 'Experts\' Consumption':'Ce',
                            'Households\' Consumption':'Ch', 'Capital':'K', 'Investment': 'Phi',
                            'Output': 'Y', 'Wealth Share' : 'LogW'}
        self.var2PerturbLabel = {'C':'Aggregate Consumption', 'Ce':'Experts\' Consumption',
                            'Ch':'Households\' Consumption', 'K':'Capital', 'Phi' : 'Investment',
                            'Y' : 'Output', 'LogW' : 'Wealth Share'}
        self.label2Sym = label2Sym
        
        
    def initializeModel(self, paramsInput, reset = False):

        ## This function initializes the model object via C++ when
        ## parameters are given and well defined.

        ## Step 0: Safety checks
        params = self.params.copy()
        params.update(paramsInput)

        #### Check parameters and make sure they are well-defined.
        checkParams(params)


        ## Step 1: Check if the oarameters are well defined.

        ## Change and fill in the parameters that can be deduced
        ## from the parameters given.
        params, self.S, self.gridSizeList, self.stateVarList = completeParams(params, reset)
        self.params = params

        ## Step 2: Initialize object
        if not reset:
            m.model.__init__(self, int(params['numSds']),
                                     float(params['sigma_K_norm']),
                                     float(params['sigma_Z_norm']),
                                     float(params['sigma_V_norm']),
                                     float(params['sigma_Vtilde_norm']),
                                     int(params['logW']),
                                     float(params['wMin']),
                                     float(params['wMax']),
                                     int(params['nDims']),
                                     int(params['nWealth']),
                                     int(params['nZ']),
                                     int(params['nV']),
                                     int(params['nVtilde']),
                                     int(params['nShocks']),
                                     int(params['verbatim']),
                                     str(params['folderName']),
                                     str(params['preLoad']),
                                     int(params['method']),
                                     float(params['dt']),
                                     float(params['dtInner']),
                                     int(params['maxIters']),
                                     int(params['maxItersInner']),
                                     float(params['tol']),
                                     float(params['innerTol']),
                                     int(params['equityIss']),
                                     int(params['hhCap']),
                                     int(params['iparm_2']),
                                     int(params['iparm_3']),
                                     int(params['iparm_28']),
                                     int(params['iparm_31']),
                                     float(params['lambda_d']),
                                     float(params['nu_newborn']),
                                     float(params['lambda_Z']),
                                     float(params['lambda_V']),
                                     float(params['lambda_Vtilde']),
                                     float(params['Z_bar']),
                                     float(params['V_bar']),
                                     float(params['Vtilde_bar']),
                                     float(params['delta_e']),
                                     float(params['delta_h']),
                                     float(params['a_e']),
                                     float(params['a_h']),
                                     float(params['rho_e']),
                                     float(params['rho_h']),
                                     float(params['phi']),
                                     float(params['alpha_K']),
                                     float(params['gamma_e']),
                                     float(params['gamma_h']),
                                     float(params['chiUnderline']),
                                     float(params['cov11']),
                                     float(params['cov12']),
                                     float(params['cov13']),
                                     float(params['cov14']),
                                     float(params['cov21']),
                                     float(params['cov22']),
                                     float(params['cov23']),
                                     float(params['cov24']),
                                     float(params['cov31']),
                                     float(params['cov32']),
                                     float(params['cov33']),
                                     float(params['cov34']),
                                     float(params['cov41']),
                                     float(params['cov42']),
                                     float(params['cov43']),
                                     float(params['cov44']),
                                     int(params['exportFreq']),
                                     params['xiEGuess'],
                                     params['xiHGuess'],
                                     params['chiGuess'],
                                     params['kappaGuess'],
                                     float(params['CGscale']),
                                     int(params['precondFreq']))
        else:
            self.reset(int(params['numSds']),
                                     float(params['sigma_K_norm']),
                                     float(params['sigma_Z_norm']),
                                     float(params['sigma_V_norm']),
                                     float(params['sigma_Vtilde_norm']),
                                     int(params['logW']),
                                     float(params['wMin']),
                                     float(params['wMax']),
                                     int(params['nDims']),
                                     int(params['nWealth']),
                                     int(params['nZ']),
                                     int(params['nV']),
                                     int(params['nVtilde']),
                                     int(params['nShocks']),
                                     int(params['verbatim']),
                                     str(params['folderName']),
                                     str(params['preLoad']),
                                     int(params['method']),
                                     float(params['dt']),
                                     float(params['dtInner']),
                                     int(params['maxIters']),
                                     int(params['maxItersInner']),
                                     float(params['tol']),
                                     float(params['innerTol']),
                                     int(params['equityIss']),
                                     int(params['hhCap']),
                                     int(params['iparm_2']),
                                     int(params['iparm_3']),
                                     int(params['iparm_28']),
                                     int(params['iparm_31']),
                                     float(params['lambda_d']),
                                     float(params['nu_newborn']),
                                     float(params['lambda_Z']),
                                     float(params['lambda_V']),
                                     float(params['lambda_Vtilde']),
                                     float(params['Z_bar']),
                                     float(params['V_bar']),
                                     float(params['Vtilde_bar']),
                                     float(params['delta_e']),
                                     float(params['delta_h']),
                                     float(params['a_e']),
                                     float(params['a_h']),
                                     float(params['rho_e']),
                                     float(params['rho_h']),
                                     float(params['phi']),
                                     float(params['alpha_K']),
                                     float(params['gamma_e']),
                                     float(params['gamma_h']),
                                     float(params['chiUnderline']),
                                     float(params['cov11']),
                                     float(params['cov12']),
                                     float(params['cov13']),
                                     float(params['cov14']),
                                     float(params['cov21']),
                                     float(params['cov22']),
                                     float(params['cov23']),
                                     float(params['cov24']),
                                     float(params['cov31']),
                                     float(params['cov32']),
                                     float(params['cov33']),
                                     float(params['cov34']),
                                     float(params['cov41']),
                                     float(params['cov42']),
                                     float(params['cov43']),
                                     float(params['cov44']),
                                     int(params['exportFreq']),
                                     params['xiEGuess'],
                                     params['xiHGuess'],
                                     params['chiGuess'],
                                     params['kappaGuess'],
                                     float(params['CGscale']),
                                     int(params['precondFreq']))

    def updateParameters(self, params = {}):

        ## This function is called to update parameters. For example, if
        ## the user did not input parameters when it was initialized, the user
        ## can update the parameters via this function.
        if 'preLoad' in self.params and self.params['preLoad'] == 'None':
            self.params.pop('preLoad', None)
        if not params:
            ## If arg params not given, the attribute params would be the new
            ## dictionary
            params = self.params.copy()

        self.initializeModel(params, True)

    def loadParams(self, fileName):
        with open(fileName) as f:
            loadedParams = json.load(f)
            loadedParams.pop('preLoad', None)
        self.params.updateParameters(loadedParams)
    
    def calDriftDiffusion(self,G):
        G = np.array(G)
        # Calculate gradients at all grid points
        G_grad = cal_gradient_grids(G=G,
                                    stateMat=np.array(self.stateMat),
                                    gridSizeList=np.array(self.gridSizeList))
        G_grad = np.array(G_grad)
        # Calculate hessian at all grid points
        G_hess = cal_hessian_grids(G=G,
                                   G_grad=G_grad,
                                   stateMat=np.array(self.stateMat),
                                   gridSizeList=np.array(self.gridSizeList))
        G_hess = np.array(G_hess)
        # Calculate drift and vol terms
        muG_grids, sigmaG_grids = cal_muG_sigma_grids(G_grad,
                                                      G_hess,
                                                      self.muX(),
                                                      np.array(self.sigmaXList))
        return muG_grids, sigmaG_grids
    
    # Han: added to calculate the drift and diffusion term of LogW
    def calDriftDiffusionLogW(self):
        G = np.log(np.array(self.W()))
        self.muLogW_grids, self.sigmaLogW_grids = self.calDriftDiffusion(G)
    
    # Han: added to return muLogW
    def muLogW(self): return self.muLogW_grids
    
    # Han: added to return sigmaLogW
    def sigmaLogW(self): return self.sigmaLogW_grids
        
    def Vtilde(self):
        return self.H()
    
    def muW(self): return self.muX()[:,0]

    def muV(self):
        errorMsg = 'Stochastic vol is not included.'
        if self.params['nV'] > 0:
            return self.muX()[:,self.stateVarList.index('V')]
        else:
            raise Exception(errorMsg)

    def muZ(self):
        errorMsg = 'Growth is not included.'
        if self.params['nZ'] > 0:
            return self.muX()[:,self.stateVarList.index('Z')]
        else:
            raise Exception(errorMsg)

    def muVtilde(self):
        errorMsg = 'Idiosyncratic vol is not included.'
        if self.params['nVtilde'] > 0:
            return self.muX()[:,self.stateVarList.index('Vtilde')]
        else:
            raise Exception(errorMsg)    
    
    def sigmaW(self): return self.sigmaXList[0]

    def sigmaV(self):
        errorMsg = 'Stochastic vol is not included.'
        if self.params['nV'] > 0:
            return self.sigmaXList[self.stateVarList.index('V')]
        else:
            raise Exception(errorMsg)

    def sigmaZ(self):
        errorMsg = 'Growth is not included.'
        if self.params['nZ'] > 0:
            return self.sigmaXList[self.stateVarList.index('Z')]
        else:
            raise Exception(errorMsg)

    def sigmaVtilde(self):
        errorMsg = 'Idiosyncratic vol is not included.'
        if self.params['nVtilde'] > 0:
            return self.sigmaXList[self.stateVarList.index('Vtilde')]
        else:
            raise Exception(errorMsg)
    
    def excessReturnKExperts(self):
        return self.muRe() - self.r()

    def excessReturnKHouseholds(self):
        return self.muRh() - self.r()

    def piE1(self):
        return self.piE()[:,0]

    def piE2(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 1:
            return self.piE()[:,1]
        else:
            raise Exception(errorMsg)

    def piE3(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 2:
            return self.piE()[:,2]
        else:
            raise Exception(errorMsg)

    def piE4(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 3:
            return self.piE()[:,3]
        else:
            raise Exception(errorMsg)

    def piDiff1(self):
        return self.piE()[:,0] - self.piH()[:,0]

    def piDiff2(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 1:
            return self.piE()[:,1] - self.piH()[:,1]
        else:
            raise Exception(errorMsg)

    def piDiff3(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 2:
            return self.piE()[:,2] - self.piH()[:,2]
        else:
            raise Exception(errorMsg)

    def piDiff4(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 3:
            return self.piE()[:,3] - self.piH()[:,3]
        else:
            raise Exception(errorMsg)

    def sigmaW1(self):
        return self.sigmaXList[0][:,0]

    def sigmaW2(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 1:
            return self.sigmaXList[0][:,1]
        else:
            raise Exception(errorMsg)

    def sigmaW3(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 2:
            return self.sigmaXList[0][:,2]
        else:
            raise Exception(errorMsg)

    def sigmaW4(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 3:
            return self.sigmaXList[0][:,3]
        else:
            raise Exception(errorMsg)

    def sigmaR1(self):
        return self.sigmaR()[:,0]

    def sigmaR2(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 1:
            return self.sigmaR()[:,1]
        else:
            raise Exception(errorMsg)

    def sigmaR3(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 2:
            return self.sigmaR()[:,2]
        else:
            raise Exception(errorMsg)

    def sigmaR4(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 3:
            return self.sigmaR()[:,3]
        else:
            raise Exception(errorMsg)

    def piH1(self):
        return self.piH()[:,0]

    def piH2(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 1:
            return self.piH()[:,1]
        else:
            raise Exception(errorMsg)

    def piH3(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 2:
            return self.piH()[:,2]
        else:
            raise Exception(errorMsg)

    def piH4(self):
        errorMsg = 'There are(is) only ' + str(self.params['nShocks']) + ' shocks.'
        if self.params['nShocks'] > 3:
            return self.piH()[:,3]
        else:
            raise Exception(errorMsg)

    def sigmaYNorm(self):
        return LA.norm(self.sigmaY(),axis=1)

    def solve(self):
        ###########################
        ## Method to solve model ##
        ###########################
        if not self.params:
            raise Exception('Model parameters are not correctly configured. Cannot solve model.')
        else:

            ## Create directory to save numerical output
            self.dent = np.full([self.S,], 0.0) ## clear stationary density
            if not os.path.exists(self.params['folderName']):
                os.makedirs(self.params['folderName'])
            else:
                if self.params['overwrite'] == 'No':
                    raise Exception('Folder already exists and not allowed to overwrite. Change the permission to overwrite or folder name.')

            self.status = self.solveModel()

            ## If used explicit scheme and model solution did not converge, prompt a warning
            ## on the choice of dt based on the CFL condition.

            if (self.status == -1) and ( (self.params['method'] == 1) or (self.params['a_h'] > 0)):
                ## method = 1 means explicit scheme.
                dVec = []
                for stateVar in self.stateVarList:
                    a = np.unique(np.array(eval('self.' + stateVar + '()')))
                    dVec.append(a[1] - a[0]) ## a is guaranteed to be monotonically increasing

                suggestDT = np.min([np.power(x,2) for x in dVec]) / 10.0
                dist      = '%e' % suggestDT
                dist      = int(dist.partition('-')[2]) ## find the nearest nonzero integer
                suggestDT = np.round(suggestDT, dist + 1 )
                if (self.params['method'] == 1):
                    msg = 'Model solution failed to converge with explicit scheme.'\
                            ' We suggest that you reduce the time step size of the outer loop (dt) to ' + str(suggestDT) + '.'
                    warnings.warn(msg)
                if (self.params['a_h'] > 0):
                    msg = 'Model solution failed to converge with explicit scheme.'\
                            ' We suggest that you reduce the time step size of the inner loop (dtInner) to ' + str(suggestDT) + '.'
                    warnings.warn(msg)
            ## After solving model, need to handle data based on the model solved.
            self.stateMat = pd.DataFrame()
            if self.status == 1:
                ## Since model was solved successfully, need to handle data
                ## First, create a pandas dataframe to store state space
                for stateVar in self.stateVarList:
                    self.stateMat[stateVar] = np.array(eval('self.' + stateVar + '()'))

                ## Second, turn sigmaX into a list
                self.sigmaXList = [self.sigmaX()[:,n * self.params['nShocks'] : \
                (n+1) * self.params['nShocks']] for n in range(0,self.params['nDims'],1)]

                ## Third, process labels
                self.processLabels()

                ## Fourth, clear moments and corrs

                self.macroMoments = {};
                self.apMoments    = {};
                self.corrs        = OrderedDict();

                ## Last, clear model and other items related to shock elas and stationary density
                self.model = None; self.stateMatInput = None;
                self.expoElas = None; self.priceElasExperts = None; self.priceElasHouseholds = None;
                self.expoElasMap = {}; self.priceElasExpertsMap = {}; self.priceElasHouseholdsMap = {};
                self.marginals = {}; self.inverseCDFs = {}
            ## Regardless of the status, export parameters
            paramsExport = self.params.copy()

            ## Don't export guesses since they could be big
            paramsExport.pop('xiEGuess', None); paramsExport.pop('xiHGuess', None);
            paramsExport.pop('kappaGuess', None); paramsExport.pop('chiGuess', None);

            with open(self.params['folderName'] + '/parameters.json', 'w') as fp:
                json.dump(paramsExport, fp)
            
            

    def restart(self):
        ## This function restarts the .solve() method by using the vectors stored
        ## in .xiE(), .xiH(), .kappa(), and .chi() as initial guesses without
        ## making any changes to other keys of .params.

        self.params['xiEGuess']        = self.xiE()
        self.params['xiHGuess']        = self.xiH()
        self.params['kappaGuess']      = self.kappa()
        self.params['chiGuess']        = self.chi()
        self.params.pop('preLoad', None)
        params                         = self.params.copy()

        self.updateParameters(params)
        self.solve()

    def processLabels(self):
        #####################################################
        ## Method to process labels after model is solved  ##
        #####################################################

        if self.status == 1:
            ## After solving a model, fix/change labels that are dependent on the parameters
            ## of the mdoel.
            self.label2Var = label2Var.copy()
            self.var2Label = var2Label.copy()
            self.label2Sym = label2Sym.copy()

            for n in range(1,self.params['nDims']):

                ## Take care of drifts
                self.label2Var['Drift of ' + self.stateVar2Label[self.stateVarList[n]]] = 'mu' + self.stateVarList[n].capitalize() + '()'
                self.var2Label['mu' + self.stateVarList[n].capitalize() + '()'] =  'Drift of ' + self.stateVar2Label[self.stateVarList[n]]
                self.label2Sym['Drift of ' + self.stateVar2Label[self.stateVarList[n]]] = '\mu_' + self.stateVarList[n]

                self.label2Var[self.stateVar2Label[self.stateVarList[n]]] = self.stateVarList[n] + '()'
                self.var2Label[self.stateVarList[n] + '()'] =  self.stateVar2Label[self.stateVarList[n]]

            ## Take care of risk prices and vol of wealth
            for s in range(1, self.params['nShocks']):
                ## risk prices
                self.label2Var['Risk Price (Experts): ' \
                + self.stateVar2Label[self.stateVarList[s]]  + ' Shock']\
                = 'piE' + str(s + 1) + '()'

                self.var2Label['piE' + str(s + 1) + '()'] \
                = 'Risk Price (Experts): ' \
                + self.stateVar2Label[self.stateVarList[s]] + ' Shock'

                self.label2Sym['Risk Price (Experts): ' \
                + self.stateVar2Label[self.stateVarList[s]]  + ' Shock']\
                 = '\pi_e^{' + str(s + 1) + '}'

                self.label2Var['Risk Price (Households): ' \
                + self.stateVar2Label[self.stateVarList[s]]  + ' Shock'] \
                = 'piH' + str(s + 1) + '()'

                self.var2Label['piH' + str(s + 1) + '()'] \
                = 'Risk Price (Households): ' \
                + self.stateVar2Label[self.stateVarList[s]]  + ' Shock'

                self.label2Sym['Risk Price (Households): ' \
                + self.stateVar2Label[self.stateVarList[s]]  + ' Shock' ]\
                 = '\pi_h^{' + str(s + 1) + '}'

                ## vol of wealth
                self.label2Var['Wealth Share Diffusion: Coordinate ' \
                + str(s+1)]\
                = 'sigmaW' + str(s + 1) + '()'

                self.var2Label['sigmaW' + str(s + 1) + '()'] \
                = 'Wealth Share Diffusion: Coordinate ' \
                + str(s+1)

                self.label2Sym['Wealth Share Diffusion: Coordinate ' \
                + str(s+1)]\
                 = '\sigma_W^{' + str(s + 1) + '}'

                ## vol of capital return
                self.label2Var['Return on Capital Diffusion: Coordinate ' \
                + str(s+1)]\
                = 'sigmaR' + str(s + 1) + '()'

                self.var2Label['sigmaR' + str(s + 1) + '()'] \
                = 'Return on Capital Diffusion: Coordinate ' \
                + str(s+1)

                self.label2Sym['Return on Capital Diffusion: Coordinate ' \
                + str(s+1)]\
                 = '\sigma_R^{' + str(s + 1) + '}'

                ## risk price diff.
                self.label2Var['Risk Price Diff. (Experts - Households): ' \
                + self.stateVar2Label[self.stateVarList[s]]  + ' Shock'] \
                = 'piDiff' + str(s + 1) + '()'

                self.var2Label['piDiff' + str(s + 1) + '()'] \
                = 'Risk Price Diff. (Experts - Households): ' \
                + self.stateVar2Label[self.stateVarList[s]]  + ' Shock'

                self.label2Sym['Risk Price Diff. (Experts - Households): ' \
                + self.stateVar2Label[self.stateVarList[s]]  + ' Shock' ] \
                 = '\pi_e^{' + str(s + 1) + '} - ' + '\pi_h^{' + str(s + 1) + '}'

        else:
            print('Model not solved. Try .solve() before processing labels.')
    def printInfo(self):
        #####################################################
        ## Method to print information after solving model ##
        #####################################################
        cwd = os.getcwd()
        checkLogMsg = 'Please check file log.txt in folder ' \
        + cwd + '/%(folderName)s to diagnose.' % self.params
        if self.status == -3:
            print('User terminated program before completion. ' + checkLogMsg)
            if self.params['a_h'] > 0:
                print('Since you solved a model with capital misallocation, we suggest that you decrease parameter'\
                ' params[\'dtInner\'] or the time step of the inner loop if you\'re using the GUI interface.')
        elif self.status == -2:
            print('There has been no attempt to solve to model. Plesae use method solve().')
        elif self.status == -1:
            print('Program resulted in error. Value functions contain nan or inf. ' + checkLogMsg)
            if self.params['a_h'] > 0:
                print('Since you solved a model with capital misallocation, we suggest that you decrease parameter'\
                ' params[\'dtInner\'] or the time step of the inner loop (if you\'re using the GUI interface).')
        elif self.status == 0:
            print('Program did not converge after reaching %(maxIters)s iterations (the maximum). ' % self.params  + checkLogMsg)
        elif self.status == 1:
            print('Program converged. Took %(totalIters)s iterations and %(totalTime)s seconds. %(pctTime)s%% '\
                  'of the time was spent on dealing with the linear systems.' % \
                  {'totalIters' : len(self.timeOuterloop),\
                   'totalTime'  : round(sum(self.timeOuterloop),2),\
                   'pctTime'    : round(100 * sum(self.timeLinSys) / sum(self.timeOuterloop),2) })

    def printParams(self, latex = False):
        #####################################################
        ## Method to print out parameters dictionary       ##
        #####################################################

        ## If latex = True, the function will print out latex code
        ## for Jupyter notebooks. If not, the function will print
        ## out a formatted table. You should only use latex = True
        ## in an environment where latex code can be rendered.
        if latex:
            ## To come
            return 0
        else:
            print("{:<20} {:<10}".format(str('Parameter'),str('Value')))
            for k in dictOrder:
                print("{:<20} {:<10}".format(k,str(self.params[k])))

            for i in range(self.params['nShocks']):
                for j in range(self.params['nShocks']):
                    print("{:<20} {:<10}".format('cov' + str(i + 1) + str(j + 1), \
                    str(self.params['cov' + str(i + 1) + str(j + 1)])))

    def plot(self, varName, pts = [], col = 0, useQuintiles = True, title = '', height = 500, width = 500, show = True, fancy = False, zoomed = False, showLegend = True, filterLegends = []):

        ## pts should be a list of dictionaries, with n - 1 of the state variables
        ## as keys and the values of those keys would be the points at which
        ## the user would like to fix
        plt.close()
        plt.ioff()
        data     = []
        if pts:
            ptsFixed = [stateVar for stateVar in self.stateVarList if stateVar in pts[0].keys()]
            x        = [x for x in self.stateVarList if x not in ptsFixed][0]
        else:
            x        = 'W'

        marginalDensity = self.marginals[x]

        if zoomed:
            ## If zoomed is True, only show support of the density distribution
            supportIdx      = marginalDensity > 0.0001
        else:
            supportIdx      = marginalDensity > -999999999

        statDent = go.Scatter(
                x = np.unique(eval('self.' + x + '()'))[supportIdx]  / (self.params['Vtilde_bar'] if x == 'Vtilde' else 1),
                y = (self.marginals[x])[supportIdx], yaxis = 'y2',
                fillcolor = 'rgba(66, 134, 244, 0.1)',
                showlegend=False, hoverinfo='none', fill='tozeroy',
                mode= 'none'
        )
        ## Before plotting, check user's inputs make sense
        if not pts and self.params['nDims'] > 1:
            raise Exception("No points are given.")

        if not fancy:
            defaultDPI = 80.0
            fig, ax1 = plt.subplots(figsize = (width/defaultDPI,height/defaultDPI))
            ax2 = ax1.twinx()
        if self.params['nDims'] > 1:
            for d in pts:
                if fancy:
                    idx  = self.findIndices(d, useQuintiles)
                    legendNames = [x for x in ptsFixed if x not in filterLegends]
                    name = [x + ' at ' +  (str(round(d[x]*100,0)) + '%' if \
                    useQuintiles else str(round(self.stateMat.iloc[idx,:][x].iloc[0], 2))) +'; ' for x in legendNames]
                    name = ''.join(name)[:-1]

                    trace = go.Scatter(
                    x = np.unique(eval('self.' + x + '()'))[supportIdx] / (self.params['Vtilde_bar'] if x == 'Vtilde' else 1),
                    y = eval('self.' + varName + '()[idx]')[supportIdx] \
                    if len(eval('self.' + varName + '()').shape) < 2\
                    else eval('self.' + varName + '()[idx,col]')[supportIdx],
                    mode = 'lines',
                    name = name[:-1], yaxis = 'y1', showlegend=showLegend
                    )
                    data.append(trace)
                else:
                    idx  = self.findIndices(d, useQuintiles)

                    name = [x + ' at ' +  (str(round(d[x]*100,0)) + '%' if \
                    useQuintiles else str(round(self.stateMat.iloc[idx,:][x].iloc[0], 2))) +'; ' for x in legendNames]
                    name = ''.join(name)[:-1]
                    stateVar = np.unique(eval('self.' + x + '()'))[supportIdx]
                    plotVar = eval('self.' + varName + '()[idx]')[supportIdx] \
                    if len(eval('self.' + varName + '()').shape) < 2\
                    else eval('self.' + varName + '()[idx,col]')[supportIdx]
                    ax1.plot(stateVar,plotVar, label = name[:-1])
                    ax1.set_xlabel(self.stateVar2Label[x])
                    ax1.set_title(self.var2Label[varName + '()'])
        else:
            if fancy:
                ## if fancy is True, use plotly
                trace = go.Scatter(
                x = np.unique(eval('self.' + x + '()'))[supportIdx] / (self.params['Vtilde_bar'] if x == 'Vtilde' else 1),
                y = eval('self.' + varName + '()')[supportIdx],
                    mode = 'lines',
                    yaxis = 'y1',
                    hoverinfo='x,y'
                )
                data.append(trace)
            else:
                ## if not, use matplotlib
                stateVar = np.unique(eval('self.' + x + '()'))[supportIdx]
                plotVar = eval('self.' + varName + '()')[supportIdx]
                ax1.plot(stateVar,plotVar)
                ax1.set_xlabel(self.stateVar2Label[x])
                ax1.set_title(self.var2Label[varName + '()'])

        if fancy:
            plotVar = eval('self.' + varName + '()[idx]')[supportIdx] \
            if len(eval('self.' + varName + '()').shape) < 2\
            else eval('self.' + varName + '()[idx,col]')[supportIdx]
            if np.max(plotVar) < 0.00001:
                layout = dict(title = self.var2Label[varName + '()'],
                              xaxis = dict(title = self.stateVar2Label[x]),
                              yaxis = dict(tickformat="0.2r"),
                              width=width, height=height, legend=dict(y = -0.25, orientation = 'h'))
            else:
                layout = dict(title = self.var2Label[varName + '()'],
                              xaxis = dict(title = self.stateVar2Label[x]),
                              width=width, height=height, legend=dict(y = -0.25, orientation = 'h'))
            configs ={'showLink': False}
            data.append(statDent)
            fig = go.Figure(data=data, layout = layout)

            fig['layout']['yaxis2'] = dict(showgrid=False,
                    zeroline=False,
                    showline=False,
                    ticks='',
                    showticklabels=False,
                    overlaying='y1',
                    side='right', range=(0, np.max(self.marginals[x]) ) )
            if np.std(eval('self.' + varName + '()')) < 0.00001:
                fig['layout']['yaxis1'].update(range = (np.min(np.min(eval('self.' + varName + '()'))\
                 - 0.2 * np.abs(np.min(eval('self.' + varName + '()')) ),0),
                np.max(eval('self.' + varName + '()')) + 0.2 * np.abs(np.max(eval('self.' + varName + '()')) )) )
            if self.params['nDims'] < 2:
                fig['layout'].update(showlegend=False)
        else:
            ax2.fill(np.unique(eval('self.' + x + '()')), self.marginals[x], 'b', alpha=0.1)
            ax2.yaxis.set_visible(False)
            ax2.set_ylim(0, np.max(self.marginals[x]))
            ax1.legend()

        if show:
            if fancy:
                init_notebook_mode(connected=False)
                py.offline.iplot(fig, filename='customPlot', config = configs)
            else:
                plt.show()
        else:
            return fig
    def dumpPlots(self, pts = [], fancy = False, zoomed = False, showLegend = True, height = 500, width = 500, filterLegends = []):
        for var in self.var2Label.keys():
            var = var.replace('()', '')
            fig = self.plot(var, pts = pts, show = False, height = height, width = width, fancy = fancy, zoomed = zoomed, showLegend = showLegend, filterLegends = filterLegends )
            legendLabel = '_noLegend' if not showLegend else ''
            if fancy:
                pio.write_image(fig, self.params['folderName'] + '/' + var + legendLabel + '.png')
            else:
                fig.savefig(self.params['folderName'] + '/' + var + legendLabel +  '.png')

    def plotElasPanel(self, perturb, W = [], Z = [], V = [], Vtilde = []):
        ## This function plots a panel of shock elasticities.
        ## It will fix points and show plots
        pDict = {}
        for state in self.stateVarList:
            pDict[state] = eval(state)

        init_notebook_mode(connected=False)
        defaultColors = [    '#1f77b4',  ## muted blue
        '#ff7f0e',  ## safety orange
        '#2ca02c',  ## cooked asparagus green
        '#d62728',  ## brick red
        '#9467bd',  ## muted purple
        '#8c564b',  ## chestnut brown
        '#e377c2',  ## raspberry yogurt pink
        '#7f7f7f',  ## middle gray
        '#bcbd22',  ## curry yellow-green
        '#17becf'   ## blue-teal
                    ]
        
        fig = tools.make_subplots(rows=self.params['nShocks'], cols = 3, subplot_titles=('Exposure', 'Price (Experts)',
                                                              'Price (Households)'),
                                                              vertical_spacing = 0.05) #, print_grid=False         
            
        for row in range(self.params['nShocks']):
            idx = functools.reduce(lambda x,y: x & y, \
            [(self.priceElasExpertsMap[perturb][x].isin(pDict[x])) for x in self.stateVarList] \
            + [ self.priceElasExpertsMap[perturb]['shock'].isin([str(int(row + 1))]) ] )

            expoData                   = self.expoElasMap[perturb]\
            [self.expoElasMap[perturb]['shock'] == str(row + 1)].iloc[:,self.params['nDims']+1:].values
            priceElasExpertsData       = self.priceElasExpertsMap[perturb]\
            [self.priceElasExpertsMap[perturb]['shock'] == str(row + 1)].iloc[:,self.params['nDims']+1:].values
            priceElasHouseholdsData    = self.priceElasHouseholdsMap[perturb]\
            [self.priceElasHouseholdsMap[perturb]['shock'] == str(row + 1)].iloc[:,self.params['nDims']+1:].values

            ### Determine ylims
            ylimMaxExpo  = 0.5
            ylimMinExpo  = -0.5
            ylimMaxPrice = 0.5
            ylimMinPrice = -0.5
            ylimMax     = expoData.max()
            ylimMin     = expoData.min()
            diff        = np.concatenate(expoData)[0:-1] - np.concatenate(expoData)[1:]
            diff        = [0 if (diff[i+1] == 0) or ((diff[i] / diff[i + 1])) > 0 \
            else 1 for i in range(len(diff) - 1)] ## change number of times the sign changes in diff
            if (not (abs(ylimMin) < 0.001 and abs(ylimMax) < 0.001)) \
            or ( sum(diff) / len(diff) < 0.4 ):
                ## Not too close to zero OR continuous -> fix axes
                ylimMaxExpo = ylimMax * 1.0/1.2 if ylimMax < -0.00001 else ylimMax * 1.2
                ylimMinExpo = ylimMin * 1.2 if ylimMin < -0.00001 else ylimMin * 1.0/1.2


            ylimMax      = max(priceElasExpertsData.max(),\
            priceElasHouseholdsData.max())
            ylimMin      = min(priceElasExpertsData.min(),\
            priceElasHouseholdsData.min())

            diff        = np.append(priceElasExpertsData,priceElasHouseholdsData)[0:-1] \
            - np.append(priceElasExpertsData,priceElasHouseholdsData)[1:]
            diff        = [0 if (diff[i+1] == 0) or ((diff[i] / diff[i + 1])) > 0 \
            else 1 for i in range(len(diff) - 1)] ## change number of times the sign changes in diff
            if (not (abs(ylimMin) < 0.001 and abs(ylimMax) < 0.001)) \
            or ( sum(diff) / len(diff) < 0.4 ):
                ylimMaxPrice = ylimMax * 1.0/1.2 if ylimMax < -0.00001 else ylimMax * 1.2
                ylimMinPrice = ylimMin * 1.2 if ylimMin < -0.00001  else ylimMin * 1.0/1.2

            yaxisLabel = 'y' if row == 0 else 'y' + str(row + 1)
            xaxisLabel = 'x' if row == 0 else 'x' + str(row + 1)

            for s in range(sum(idx)):
                ## Extend color list if necessary
                if len(defaultColors) < sum(idx) - 1:
                    defaultColors = defaultColors * int((round(sum(idx)/2) + 1))
                ## Figure out legends
                if '%' in self.expoElasMap[perturb][idx].iloc[s]['W']:
                    name = [x + ' at ' +  self.expoElasMap[perturb][idx].iloc[s][x] +'; ' for x in self.stateVarList]
                else:
                    name = [x + ' = '  +  str(round(float(self.expoElasMap[perturb][idx].iloc[s][x]),2)) +'; ' for x in self.stateVarList]
                name = ''.join(name)[:-1]
                legendTF = True if (row == 0) else False ## only need to show legends once

                ## Plot
                trace = go.Scatter(x =
                                np.linspace(0, int(float(self.expoElasMap[list(self.expoElasMap.keys())[0]].columns[-1])),
                                            int(float(self.expoElasMap[list(self.expoElasMap.keys())[0]].columns[-1])) ),
                                y = self.expoElasMap[perturb][idx].iloc[s].values[self.params['nDims']+1:],
                                  line = dict(color = defaultColors[s]), name = name, showlegend = legendTF,
                                  yaxis=yaxisLabel, xaxis = xaxisLabel)
                fig.append_trace(trace, row + 1, 1)
                
                trace = go.Scatter(x =
                                np.linspace(0,int(float(self.priceElasExpertsMap[list(self.priceElasExpertsMap.keys())[0]].columns[-1])),
                                            int(float(self.priceElasExpertsMap[list(self.priceElasExpertsMap.keys())[0]].columns[-1])) ),
                                y = self.priceElasExpertsMap[perturb][idx].iloc[s].values[self.params['nDims']+1:],
                                  line = dict(color = defaultColors[s]), name = name, showlegend = False,
                                  yaxis=yaxisLabel, xaxis = xaxisLabel)
                fig.append_trace(trace, row + 1, 2)
                
                trace = go.Scatter(x =
                                np.linspace(0, int(float(self.priceElasHouseholdsMap[list(self.priceElasHouseholdsMap.keys())[0]].columns[-1])) ,
                                            int(float(self.priceElasHouseholdsMap[list(self.priceElasHouseholdsMap.keys())[0]].columns[-1])) ),
                                y = self.priceElasHouseholdsMap[perturb][idx].iloc[s].values[self.params['nDims']+1:],
                                  line = dict(color = defaultColors[s]), name = name, showlegend = False,
                                  yaxis=yaxisLabel, xaxis = xaxisLabel)
                fig.append_trace(trace, row + 1, 3)
                    
                yaxisLabelLayout = 'yaxis' if row == 0 else 'yaxis' + str(row + 1)

            ## Handle x axes
            for j in range(3):
                ## There are #shocks rows and 3 columns
                xaxisLabelLayout = 'xaxis' if (row == 0 and j == 0) else 'xaxis' + str(3 * row + j + 1)
                if row == self.params['nShocks'] - 1:
                    fig['layout'][xaxisLabelLayout].update(title='Time (Years)', showline=True)
                else:
                    fig['layout'][xaxisLabelLayout].update(showline=True)

                yaxisLabelLayout = 'yaxis' if (row == 0 and j == 0) else 'yaxis' + str(3 * row + j + 1)
                if j == 0:
                    fig['layout'][yaxisLabelLayout].update(title='Shock ' \
                    + str(row + 1), range = [ylimMinExpo, ylimMaxExpo])
                else:
                    fig['layout'][yaxisLabelLayout].update(range = [ylimMinPrice, ylimMaxPrice])
        fig['layout'].update(height=self.params['nShocks'] * 400, width=1000, title='Shock Elasticities (Perturbed Variable: '
                             + self.var2PerturbLabel[perturb] + ')' )
        configs ={'showLink': False}
        py.offline.iplot(fig, config = configs, filename='shockElasticitiesPanelPlot')

    def plotElas(self, perturb = 'C',  W = [], Z = [], V = [], Vtilde = [], showAgent = True, showLegend = False, height = 500, width = 500, filterLegends = []):
        ## This function plots shock elasticities stored in *Map
        ## elasType: either Exposure or Price
        ## agent: Either Experts or Households
        ## titleName: title of charts
        ## shockNum: shock number
        ## perturbed: perturb variable

        plt.close()
        plt.ioff()
        init_notebook_mode(connected=False)
        defaultColors = [    '#1f77b4',  ## muted blue
        '#ff7f0e',  ## safety orange
        '#2ca02c',  ## cooked asparagus green
        '#d62728',  ## brick red
        '#9467bd',  ## muted purple
        '#8c564b',  ## chestnut brown
        '#e377c2',  ## raspberry yogurt pink
        '#7f7f7f',  ## middle gray
        '#bcbd22',  ## curry yellow-green
        '#17becf'   ## blue-teal
                    ]
        pDict = {}
        for state in self.stateVarList:
            pDict[state] = eval(state)

        for row in range(self.params['nShocks']):
            shockNum = row + 1

            idx = functools.reduce(lambda x,y: x & y, \
                [(self.priceElasExpertsMap[perturb][x].isin(pDict[x])) for x in self.stateVarList] \
                + [ self.priceElasExpertsMap[perturb]['shock'].isin([str(int(shockNum))]) ] )

            ## Find elasticities data
            expoData                   = self.expoElasMap[perturb]\
                    [self.expoElasMap[perturb]['shock'] == str(shockNum)].iloc[:,self.params['nDims']+1:].values
            priceElasExpertsData       = self.priceElasExpertsMap[perturb]\
                    [self.priceElasExpertsMap[perturb]['shock'] == str(shockNum)].iloc[:,self.params['nDims']+1:].values
            priceElasHouseholdsData    = self.priceElasHouseholdsMap[perturb]\
                    [self.priceElasHouseholdsMap[perturb]['shock'] == str(shockNum)].iloc[:,self.params['nDims']+1:].values

            ### Determine ylims
            ylimMaxExpo            = 0.5
            ylimMinExpo            = -0.5
            ylimMaxPriceHouseholds = 0.5
            ylimMinPriceHouseholds = -0.5
            ylimMaxPriceExperts    = 0.5
            ylimMinPriceExperts    = -0.5

            ylimMax     = expoData.max()
            ylimMin     = expoData.min()
            diff        = np.concatenate(expoData)[0:-1] - np.concatenate(expoData)[1:]
            diff        = [0 if (diff[i+1] == 0) or ((diff[i] / diff[i + 1])) > 0 \
            else 1 for i in range(len(diff) - 1)] ## change number of times the sign changes in diff
            if (not (abs(ylimMin) < 0.001 and abs(ylimMax) < 0.001)) \
            or ( sum(diff) / len(diff) < 0.4 ):
                ## Not too close to zero OR continuous -> fix axes
                ylimMaxExpo = ylimMax * 1.0/1.2 if ylimMax < -0.00001 else ylimMax * 1.2
                ylimMinExpo = ylimMin * 1.2 if ylimMin < -0.00001 else ylimMin * 1.0/1.2


            ylimMax      = priceElasExpertsData.max()
            ylimMin      = priceElasExpertsData.min()

            diff        = np.append(priceElasExpertsData,priceElasHouseholdsData)[0:-1] \
            - np.append(priceElasExpertsData,priceElasHouseholdsData)[1:]
            diff        = [0 if (diff[i+1] == 0) or ((diff[i] / diff[i + 1])) > 0 \
            else 1 for i in range(len(diff) - 1)] ## change number of times the sign changes in diff
            if (not (abs(ylimMin) < 0.001 and abs(ylimMax) < 0.001)) \
            or ( sum(diff) / len(diff) < 0.4 ):
                ylimMaxPriceExperts = ylimMax * 1.0/1.2 if ylimMax < -0.00001 else ylimMax * 1.2
                ylimMinPriceExperts = ylimMin * 1.2 if ylimMin < -0.00001  else ylimMin * 1.0/1.2

            ylimMax      = priceElasHouseholdsData.max()
            ylimMin      = priceElasHouseholdsData.min()

            diff        = np.append(priceElasExpertsData,priceElasHouseholdsData)[0:-1] \
            - np.append(priceElasExpertsData,priceElasHouseholdsData)[1:]
            diff        = [0 if (diff[i+1] == 0) or ((diff[i] / diff[i + 1])) > 0 \
            else 1 for i in range(len(diff) - 1)] ## change number of times the sign changes in diff
            if (not (abs(ylimMin) < 0.001 and abs(ylimMax) < 0.001)) \
            or ( sum(diff) / len(diff) < 0.4 ):
                ylimMaxPriceHouseholds = ylimMax * 1.0/1.2 if ylimMax < -0.00001 else ylimMax * 1.2
                ylimMinPriceHouseholds = ylimMin * 1.2 if ylimMin < -0.00001  else ylimMin * 1.0/1.2


            ## Empty list to store traces
            expoTraces               = []
            priceExpertsTraces       = []
            priceHouseholdsTraces    = []

            ## Indicator variable as to reformating the axes
            expoReformat              = False
            priceExpertsReformat      = False
            priceHouseholdsReformat   = False

            for s in range(sum(idx)):
                ## Extend color list if necessary
                if len(defaultColors) < sum(idx) - 1:
                    defaultColors = defaultColors * int((round(sum(idx)/2) + 1))
                ## Figure out legends
                legendNames = [x for x in self.stateVarList if x not in filterLegends]
                if '%' in self.expoElasMap[perturb][idx].iloc[s]['W']:
                    name = [x + ' at ' +  self.expoElasMap[perturb][idx].iloc[s][x] +'; ' for x in legendNames]
                else:
                    name = [x + ' = '  +  str(round(float(self.expoElasMap[perturb][idx].iloc[s][x]),2)) +'; ' for x in legendNames]
                name = ''.join(name)[:-1]

                ## Plot
                trace = go.Scatter(x =
                                    np.linspace(0, int(float(self.expoElasMap[list(self.expoElasMap.keys())[0]].columns[-1])) * self.model['dt'],
                                                int(float(self.expoElasMap[list(self.expoElasMap.keys())[0]].columns[-1])) ),
                                    y = self.expoElasMap[perturb][idx].iloc[s].values[self.params['nDims']+1:],
                                      line = dict(color = defaultColors[s]), name = name, showlegend = showLegend )
                expoTraces.append(trace)
                if np.max(abs(self.expoElasMap[perturb][idx].iloc[s].values[self.params['nDims']+1:])) < 0.001:
                    expoReformat              = True

                trace = go.Scatter(x =
                                    np.linspace(0,int(float(self.priceElasExpertsMap[list(self.priceElasExpertsMap.keys())[0]].columns[-1])) * self.model['dt'],
                                                int(float(self.priceElasExpertsMap[list(self.priceElasExpertsMap.keys())[0]].columns[-1])) ),
                                    y = self.priceElasExpertsMap[perturb][idx].iloc[s].values[self.params['nDims']+1:],
                                      line = dict(color = defaultColors[s]), name = name, showlegend = showLegend )
                priceExpertsTraces.append(trace)
                if np.max(abs(self.priceElasExpertsMap[perturb][idx].iloc[s].values[self.params['nDims']+1:])) < 0.001:
                    priceExpertsReformat      = True


                trace = go.Scatter(x =
                                    np.linspace(0, int(float(self.priceElasHouseholdsMap[list(self.priceElasHouseholdsMap.keys())[0]].columns[-1])) * self.model['dt'],
                                                int(float(self.priceElasHouseholdsMap[list(self.priceElasHouseholdsMap.keys())[0]].columns[-1])) ),
                                    y = self.priceElasHouseholdsMap[perturb][idx].iloc[s].values[self.params['nDims']+1:],
                                      line = dict(color = defaultColors[s]), name = name, showlegend = showLegend )
                priceHouseholdsTraces.append(trace)
                if np.max(abs(self.priceElasHouseholdsMap[perturb][idx].iloc[s].values[self.params['nDims']+1:])) < 0.001:
                    priceHouseholdsReformat   = True
            ## Configure layout
            layoutExpo               = dict(title = 'Exposure Elasticities: Shock ' + str(int(shockNum)) + '<br>Perturbed Variable: '
                                 + self.var2PerturbLabel[perturb],
                          xaxis = dict(title = 'Time'),
                          width=width, height=height, legend=dict(y = -0.25, orientation = 'h') )
            layoutPriceExperts       = dict(title = ('Price Elasticities (Experts): Shock ' + str(int(shockNum)) if showAgent else 'Price Elasticities: Shock ' + str(int(shockNum))) + '<br>Perturbed Variable: '
                                 + self.var2PerturbLabel[perturb],
                          xaxis = dict(title = 'Time'),
                          width=width, height=height, legend=dict(y = -0.25, orientation = 'h') )
            layoutPriceHouseholds    = dict(title = ('Price Elasticities (Households): Shock '  + str(int(shockNum)) if showAgent else 'Price Elasticities: Shock '  + str(int(shockNum))) + '<br>Perturbed Variable: '
                                 + self.var2PerturbLabel[perturb],
                          xaxis = dict(title = 'Time'),
                          width=width, height=height, legend=dict(y = -0.25, orientation = 'h') )


            figExpo                = go.Figure(data = expoTraces, layout = layoutExpo)
            figPriceExperts        = go.Figure(data = priceExpertsTraces, layout = layoutPriceExperts)
            figPriceHouseholds     = go.Figure(data = priceHouseholdsTraces, layout = layoutPriceHouseholds)

            figExpo['layout']['yaxis1'].update(range = [ylimMinExpo, ylimMaxExpo])
            figPriceExperts['layout']['yaxis1'].update(range = [ylimMinPriceExperts, ylimMaxPriceExperts])
            figPriceHouseholds['layout']['yaxis1'].update(range = [ylimMinPriceHouseholds, ylimMaxPriceHouseholds])
            if expoReformat:
                figExpo['layout']['yaxis1'].update(tickformat="0.2r")
            if priceExpertsReformat:
                figPriceExperts['layout']['yaxis1'].update(tickformat="0.2r")
            if priceHouseholdsReformat:
                figPriceHouseholds['layout']['yaxis1'].update(tickformat="0.2r")

            ## Export charts
            legendLabel = '_noLegend' if not showLegend else ''
            pio.write_image(figExpo, self.params['folderName'] + '/shockElas_' + perturb + '_Expo_shock' + str(int(shockNum)) + legendLabel + '.png')
            pio.write_image(figPriceExperts, self.params['folderName'] + '/shockElas_' + perturb + '_PriceExperts_shock' + str(int(shockNum)) + legendLabel + '.png')
            pio.write_image(figPriceHouseholds, self.params['folderName'] + '/shockElas_' + perturb + '_PriceHouseholds_shock' + str(int(shockNum)) + legendLabel + '.png')

    def plotPanel(self, varName1 = 'piE1()', varName2 = 'piH1()', xaxisState = 'Wealth Share'):
        ## This method plots a 1x2 plot at the user's choice.

        ## This methid allows the generalization to mulitple dimensions.
        ## xaxisState is the state variable that will be displayed on the x-axis.
        xaxisState = self.label2stateVar[xaxisState]
        shapes = []
        init_notebook_mode(connected=False)

        if self.params['nDims'] == 1:
            ## Select data
            plotVar = np.squeeze( eval('self.' + varName1))
            trace1 = go.Scatter(
                x = self.stateMat['W'],
                y = plotVar,
                               showlegend = False,
                               name = self.var2Label[varName1], line=dict(color='#00CED1', width=6)
            )

            ## Density
            statDent = go.Scatter(
                x = self.stateMat['W'],
                y = np.squeeze( self.dent ), yaxis = 'y3',
                fillcolor = 'rgba(66, 134, 244, 0.1)',
                showlegend=False, hoverinfo='none', fill='tozeroy',
                mode= 'none'
            )

            ## Add pct lines
            try:
               pctPt1 = go.Scatter(
                                   x = [0.5,.6,.7],
                                   y = [ np.min(np.squeeze( eval('self.' + varName1))) * \
                                   (0.01 if np.min(np.squeeze( eval('self.' + varName1))) else 2.0) - 1.0 ] * 3 , yaxis = 'y2',
                                   name = '10th Pct',
                                   mode = 'lines',
                                   line = {
                                   'color': 'rgb(220,20,60)',
                                   'width': 1.0
                                },
                                hoverinfo = 'none'
               )
               pctPt2 = go.Scatter(
                                   x = [0.5,.6,.7],
                                   y = [ np.min(np.squeeze( eval('self.' + varName1))) * \
                                   (0.01 if np.min(np.squeeze( eval('self.' + varName1))) else 2.0) - 1.0 ] * 3 , yaxis = 'y2',
                                   name = '50th Pct',
                                   mode = 'lines',
                                   line = {
                                   'color': 'rgb(220,20,60)',
                                   'width': 1.0,
                                   'dash' : 'dot'
                                },
                                hoverinfo = 'none'
               )
               pctPt3 = go.Scatter(
                                   x = [0.5,.6,.7],
                                   y = [ np.min(np.squeeze( eval('self.' + varName1))) * \
                                   (0.01 if np.min(np.squeeze( eval('self.' + varName1))) else 2.0) - 1.0 ] * 3 , yaxis = 'y2',
                                   name = '90th Pct',
                                   mode = 'lines',
                                   line = {
                                   'color': 'rgb(220,20,60)',
                                   'width': 1.0,
                                   'dash' : 'dashdot'
                                },
                                hoverinfo = 'none'
               )

               pcts = [0.1, 0.5, 0.9]
               dashes = ['solid', 'dot', 'dashdot']
               for pct, dash in zip(pcts,dashes):
                  shapes.append({'type': 'line',
                     'xref': 'x1',
                     'yref': 'y2',
                     'x0': self.inverseCDFs['W'](pct),
                     'y0': 0,
                     'x1': self.inverseCDFs['W'](pct),
                     'y1': np.max(self.dent),
                     'line': {
                     'color': 'rgb(220,20,60)',
                     'width': 1.0,
                     'dash': dash,
                  }})
            except:
                pass

            if not (self.dent is None or np.max(self.dent) < 0.0001):
                fig = go.Figure(data=[trace1, statDent, pctPt1, pctPt2, pctPt3])
            else:
                fig = go.Figure(data=[trace1, statDent])
            ## Add density and data to figure

            #### First two lines deal with variables; second two lines deal with
            #### density

            ## Change styles
            fig['data'][0].update(yaxis='y1')
            fig['data'][1].update(yaxis='y2')

            fig['layout']['xaxis1'].update(title='Wealth Share')
            #fig['layout']['xaxis2'].update(title='Wealth Share')

            fig['layout']['xaxis1'].update(hoverformat = '.2f')
            ##fig['layout']['xaxis2'].update(hoverformat = '.2f')
            fig['layout']['yaxis1'].update(hoverformat = '.2f', range = (  -0.001 if np.abs(np.min( plotVar.flatten() ) - 0) < 0.0001 else ( np.min( plotVar.flatten() ) * .8 \
                                         if np.min( plotVar.flatten() ) > 0.0001 else \
                                         np.min( plotVar.flatten() ) * 1.1) \
                                        , 0.001 if np.abs(np.max( plotVar.flatten() ) - 0) < 0.0001 else  ( np.max( plotVar.flatten() ) * 1.1 \
                                                                      if np.max( plotVar.flatten() ) > 0.0001 else \
                                                                      np.max( plotVar.flatten() ) * 0.8)  ) )

            if np.std(plotVar) < 0.001:
                fig['layout']['yaxis1'].update(range = (np.mean(plotVar) * .75, np.mean(plotVar) * 1.25))

            fig['layout'].update(title=self.var2Label[varName1], width = 900, height = 500)
            fig['layout']['yaxis2'] = dict(showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False,
                overlaying='y1',
                side='right', range=(0, np.max(self.dent) ) )

            fig['layout'].update(shapes=shapes, margin = dict(
    r = 40,
    t = 40,
    b = 40,
    l = 40, pad = 4
  ))
            configs ={'showLink': False }
            py.offline.iplot(fig, config = configs, filename='customizing-subplot-axes')
        elif self.params['nDims'] == 2:
            plotVar  = np.squeeze( eval('self.' + varName1  )).reshape(self.gridSizeList, order = 'F')

            statDent = None; pctsLine1 = None;
            pctsLine2 = None; pctsLine3 = None;
            marginalDent = None; y2max = None;
            fs = None; wPcts10 = None; wPcts50 = None; wPcts90 = None;
            dataList = None
            sliderNum    = 1 if (xaxisState == 'W') else 0
            ## Handle distributions if the user has computed it
            try:
                dent         = self.dent.reshape(self.gridSizeList, order = 'F')
                conds        = dent.sum(axis=0)
                y2max        = np.max(self.marginals[xaxisState])

                statDent = [dict(
                        visible = False,
                        x = np.unique(self.W()) if (xaxisState == 'W') else np.unique(eval('self.' + xaxisState + '()' )),
                        line=dict(color='rgba(143, 19, 131, 0.1)', width=6),
                        fillcolor = 'rgba(66, 134, 244, 0.1)',yaxis = 'y2',
                        y = dent[:,step] if (xaxisState == 'W') else dent[step,:],
                        showlegend=False, hoverinfo='none', fill='tozeroy' ) \
                        for step in range(self.gridSizeList[sliderNum]) ]
                statDent[round(self.gridSizeList[1]/2)]['visible'] = True

                ## Density
                marginalDent = [dict(
                    visible = False,
                    x = np.unique( eval('self.' + xaxisState + '()' ) ),
                    fillcolor = 'rgba(176, 224, 255, 0.15)',yaxis = 'y2',
                    y = np.squeeze(self.marginals[xaxisState]),
                    showlegend=False, hoverinfo='none', fill='tozeroy', mode = 'none' ) \
                    for step in range(self.gridSizeList[sliderNum])]
                marginalDent[round(self.gridSizeList[1]/2)]['visible'] = True

                pctsLine1 = [dict(
                    visible = False,
                    line=dict(color='rgb(220,20,60)', width=1, dash = 'solid'),
                    name = '10th Pct',yaxis = 'y2', mode = 'lines',
                    x = [np.asscalar( self.inverseCDFs[xaxisState](0.1) )] * 2,
                    y = [0, y2max] ) for step in range(self.gridSizeList[sliderNum]) ]
                pctsLine1[round(self.gridSizeList[1]/2)]['visible'] = True

                pctsLine2 = [dict(
                    visible = False,
                    line=dict(color='rgb(220,20,60)', width=1, dash = 'dot'),
                    name = '50th Pct',yaxis = 'y2', mode = 'lines',
                    x = [np.asscalar( self.inverseCDFs[xaxisState](0.5) )] * 2,
                    y = [0, y2max] ) for step in range(self.gridSizeList[sliderNum]) ]
                pctsLine2[round(self.gridSizeList[1]/2)]['visible'] = True

                pctsLine3 = [dict(
                    visible = False,
                    line=dict(color='rgb(220,20,60)', width=1, dash = 'dashdot'),
                    name = '90th Pct',yaxis = 'y2', mode = 'lines',
                    x = [np.asscalar( self.inverseCDFs[xaxisState](0.9) )] * 2,
                    y = [0, y2max] ) for step in range(self.gridSizeList[sliderNum]) ]
                pctsLine3[round(self.gridSizeList[1]/2)]['visible'] = True
            except:
                print('Stationary density not computed or degenerate.')
            ## Equilibrium quantity to be displayed
            data = [dict(
                    visible = False,
                    line=dict(color='#00CED1', width=6),
                    name = self.var2Label[varName1], showlegend = False,
                    x = np.unique(self.W()) if (xaxisState == 'W') else np.unique(eval('self.' + xaxisState + '()' )),
                    y = plotVar[:,step] if (xaxisState == 'W') else plotVar[step,:] ) for step in range(self.gridSizeList[sliderNum]) ]

            ## Determine active spot
            if not (xaxisState == 'W'):
                try:
                    activeIdx = np.argmin(np.abs(np.unique(self.W()) - self.inverseCDFs['W'](.5)))
                except:
                    activeIdx = round(self.gridSizeList[sliderNum]/2)
            else:
                try:
                    activeIdx = np.argmin(np.abs( np.unique(eval('self.' + self.stateVarList[sliderNum] + '()' )) - \
                    self.params[self.stateVarList[sliderNum] + '_bar'] ))
                except:
                    activeIdx = round(self.gridSizeList[sliderNum]/2)

            data[activeIdx]['visible'] = True

            steps        = []
            state2       = np.unique(self.stateMat[self.stateVarList[sliderNum]])

            for i in range(self.gridSizeList[sliderNum]):
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(data)],
                    label = str(round(state2[i],3))
                )
                step['args'][1][i] = True # Toggle i'th trace to "visible"
                steps.append(step)

            sliders = [dict(
                active = activeIdx,
                currentvalue = {"prefix": self.stateVar2Label[self.stateVarList[sliderNum]] + ': '},
                pad = {"t": 50},
                steps = steps
            )]

            layout = dict(sliders = sliders, width = 900, height = 500, title = self.var2Label[varName1],
                          xaxis = dict(title =  self.stateVar2Label[xaxisState] ),
                      yaxis=dict(range= (  -0.001 if np.abs(np.min( plotVar.flatten() ) - 0) < 0.0001 else ( np.min( plotVar.flatten() ) * .8 \
                                                   if np.min( plotVar.flatten() ) > 0.0001 else \
                                                   np.min( plotVar.flatten() ) * 1.1) \
                                                  , 0.001 if np.abs(np.max( plotVar.flatten() ) - 0) < 0.0001 else  ( np.max( plotVar.flatten() ) * 1.1 \
                                                                                if np.max( plotVar.flatten() ) > 0.0001 else \
                                                                                np.max( plotVar.flatten() ) * 0.8)  ) ),
                     yaxis2=dict(showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False,
                overlaying='y',
                side='right', range = (0, y2max),
            ),   margin = dict(
    r = 40,
    t = 40,
    b = 40,
    l = 40, pad = 4
  ))

            configs ={'showLink': False}
            if statDent is None:
                fig = dict(data= data, layout=layout)
            else:
                fig = dict(data= data + statDent  + marginalDent + pctsLine1 + pctsLine2 + pctsLine3, layout=layout)

            py.offline.iplot(fig, config = configs, filename='sliders_chart')
        else:
            sliderList   = [x for x in self.stateVarList if not (x == xaxisState)]
            sliderNums   = [self.stateVarList.index(x) for x in sliderList]
            x            = np.unique(eval('self.' + xaxisState + '()' ))
            state2       = np.unique(self.stateMat[self.stateVarList[sliderNums[0]]])
            state3       = np.unique(self.stateMat[self.stateVarList[sliderNums[1]]])
            y            = np.squeeze( eval('self.' + varName1  )).reshape(self.gridSizeList, order = 'F')
            f            = None

            dent         = self.dent.reshape(self.gridSizeList, order = 'F')

            trace1 = go.Scatter(
                        x = x,
                        y = y,
                                       showlegend = False,
                                       name = self.var2Label[varName1], line=dict(color='#00CED1', width=6)
                    )
            try:
                statDent = go.Scatter(
                            x = self.stateMat[xaxisState],
                            y = np.squeeze( dent[:,0,0] ), yaxis = 'y3',
                            line=dict(color='rgba(143, 19, 131, 0.1)', width=6),
                            fillcolor = 'rgba(66, 134, 244, 0.1)',
                            showlegend=False, hoverinfo='none', fill='tozeroy'
                )

                marginal = go.Scatter(
                            x = self.stateMat[xaxisState],
                            y = np.squeeze( self.marginals[xaxisState] ), yaxis = 'y3',
                            fillcolor = 'rgba(66, 134, 244, 0.1)',
                            showlegend=False, hoverinfo='none', fill='tozeroy',
                            mode= 'none'
                )

                ## Add pct lines
                pcts = [0.1, 0.5, 0.9]
                dashes = ['solid', 'dot', 'dashdot']
                for pct, dash in zip(pcts,dashes):
                    shapes.append({'type': 'line',
                       'xref': 'x1',
                       'yref': 'y3',
                       'x0': self.inverseCDFs[xaxisState](pct),
                       'y0': 0,
                       'x1': self.inverseCDFs[xaxisState](pct),
                       'y1': np.max(self.marginals[xaxisState]),
                       'line': {
                       'color': 'rgb(220,20,60)',
                       'width': 1.0,
                       'dash': dash,
                    }})

                pctPt1 = go.Scatter(
                                    x = [0.5,.6,.7],
                                    y = [ np.min(np.squeeze( eval('self.' + varName1))) * \
                                    (0.01 if np.min(np.squeeze( eval('self.' + varName1))) else 2.0) - 1.0 ] * 3 , yaxis = 'y3',
                                    name = '10th Pct',
                                    mode = 'lines',
                                    line = {
                                    'color': 'rgb(220,20,60)',
                                    'width': 1.0
                                 },
                                 hoverinfo = 'none'
                )
                pctPt2 = go.Scatter(
                                    x = [0.5,.6,.7],
                                    y = [ np.min(np.squeeze( eval('self.' + varName1))) * \
                                    (0.01 if np.min(np.squeeze( eval('self.' + varName1))) else 2.0) - 1.0 ] * 3 , yaxis = 'y3',
                                    name = '50th Pct',
                                    mode = 'lines',
                                    line = {
                                    'color': 'rgb(220,20,60)',
                                    'width': 1.0,
                                    'dash' : 'dot'
                                 },
                                 hoverinfo = 'none'
                )
                pctPt3 = go.Scatter(
                                    x = [0.5,.6,.7],
                                    y = [ np.min(np.squeeze( eval('self.' + varName1))) * \
                                    (0.01 if np.min(np.squeeze( eval('self.' + varName1))) else 2.0) - 1.0 ] * 3 , yaxis = 'y3',
                                    name = '90th Pct',
                                    mode = 'lines',
                                    line = {
                                    'color': 'rgb(220,20,60)',
                                    'width': 1.0,
                                    'dash' : 'dashdot'
                                 },
                                 hoverinfo = 'none'
                )
                f = go.FigureWidget(
                    data=[trace1, statDent, pctPt1, pctPt2, pctPt3, marginal]
                )

                f['layout'].update(shapes=shapes)

            except:
                f = go.FigureWidget(
                    data=[trace1]
                )
            ## Change styles
            try:
                f['layout']['yaxis3'] = dict(showgrid=False,
                            zeroline=False,
                            showline=False,
                            ticks='',
                            showticklabels=False,
                            overlaying='y1',
                            side='right', range=(0, np.max(self.marginals[xaxisState]) ) )
            except:
                pass

            f['data'][0].update(yaxis='y1')
            try:
                f['data'][1].update(yaxis='y3')
                f['layout']['yaxis3'].update(range=(0, np.max(self.marginals[xaxisState]) ) )
            except:
                pass

            f['layout']['xaxis1'].update(title= self.stateVar2Label[xaxisState] )

            f['layout']['xaxis1'].update(hoverformat = '.2f')
            f['layout']['yaxis1'].update(hoverformat = '.2f', range = ( -0.001 if np.abs(np.min( y.flatten() ) - 0) < 0.0001 else ( np.min( y.flatten() ) * .8 \
                                         if np.min( y.flatten() ) > 0.0001 else \
                                         np.min( y.flatten() ) * 1.1) \
                                        , 0.001 if np.abs(np.max( y.flatten() ) - 0) < 0.0001 else  ( np.max( y.flatten() ) * 1.1 \
                                                                      if np.max( y.flatten() ) > 0.0001 else \
                                                                      np.max( y.flatten() ) * 0.8)  ) )

            f['layout']['title'] = self.var2Label[varName1]
            f['layout']['xaxis1'].update(range= (min(x),\
            max(x)) )
            f['layout'].update( width = 900, height = 500, margin = dict(
                r = 40,
                t = 40,
                b = 40,
                l = 40, pad = 4
              ))
            def updateState2(state2Val, state3Val):
                ## find indices
                state2Idx = np.argmin(np.abs(state2 - state2Val))
                state3Idx = np.argmin(np.abs(state3 - state3Val))
                f.data[0].y = y[:,state2Idx, state3Idx] if (self.stateVarList.index(xaxisState) == 0 )\
                else y[state2Idx, :, state3Idx] if (self.stateVarList.index(xaxisState) == 1) \
                else y[state2Idx, state3Idx, :]
                f.data[0].x =  np.unique( self.stateMat[xaxisState] )

                try:
                    f.data[1].y = dent[:,state2Idx, state3Idx] if (self.stateVarList.index(xaxisState) == 0) \
                    else dent[state2Idx, :, state3Idx] if (self.stateVarList.index(xaxisState) == 1) \
                    else dent[state2Idx, state3Idx, :]
                    f.data[1].x = np.unique(self.stateMat[xaxisState])

                    f.data[-1].y = np.squeeze( self.marginals[xaxisState] )
                    f.data[-1].x = np.unique( self.stateMat[xaxisState] )
                except:
                    pass

            slider = interactive(updateState2, state2Val=(min(state2), max(state2), state2[2] - state2[1]),
                                      state3Val=(min(state3), max(state3), state3[2] - state3[1]))
            slider.children[0].description = self.stateVar2Label[self.stateVarList[sliderNums[0]]]
            slider.children[0].layout = Layout(width='500px')
            slider.children[0].style = {'description_width': '150px'}
            ## Determine active spot for 1
            if (self.stateVarList[sliderNums[0]] == 'W'):
                try:
                    val0 = self.inverseCDFs[self.stateVarList[sliderNums[0]]](.5)
                except:
                    val0 = round(self.gridSizeList[sliderNums[0]]/2)
            else:
                val0 = self.params[self.stateVarList[sliderNums[0]] + '_bar']

            slider.children[0].value = val0

            slider.children[1].description = self.stateVar2Label[self.stateVarList[sliderNums[1]]]
            slider.children[1].layout = Layout(width='500px')
            slider.children[1].style = {'description_width': '150px'}
            ## Determine active spot for 2
            if (self.stateVarList[sliderNums[1]] == 'W'):
                try:
                    val1 = self.inverseCDFs[self.stateVarList[sliderNums[1]]](.5)
                except:
                    val1 = round(self.gridSizeList[sliderNums[1]]/2)
            else:
                val1 = self.params[self.stateVarList[sliderNums[1]] + '_bar']

            slider.children[1].value = val1

            vb = VBox((f, slider))
            vb.layout.align_items = 'center'
            display(vb)

    def findIndices(self, pcts = {}, useQuintiles = True):

        '''
        This method finds indices of the state space by fixing state variables
        at the points given in pcts. It returns a list of indices.
        '''

        ## Find points
        pDict     = {} ## dictionary to store points
        for stateVar in pcts.keys():
            if useQuintiles:
                if self.dent is None or np.max(self.dent) < 0.0001:
                    raise Exception("Stationary density not computed.")
                pDict[stateVar] = np.asscalar(self.inverseCDFs[stateVar](pcts[stateVar]))
            else:
                pDict[stateVar] = pcts[stateVar]

        stateMat = self.stateMat.copy()
        for stateVar in pDict.keys():
            if pDict[stateVar] < np.min(self.stateMat[stateVar]) or \
            pDict[stateVar] > np.max(self.stateMat[stateVar]):
                ## Issue a warning when the user has input a value out of range
                warnings.warn('The value of {point} of {stateVar} is out of the state pace. Program will still find the closest point.'.format( \
                **{'point' : str(pDict[stateVar]), 'stateVar' : stateVar} ))
            anchor   = np.argmin( np.abs(stateMat[stateVar] - pDict[stateVar]) )
            stateMat = stateMat[stateMat[stateVar] == stateMat[stateVar][anchor]]

        return stateMat.index.tolist()
    def smoothResults(self, degree = 9):
        '''
        This function smoothes numerical results after solving badly behaving models
        such as models with capital misallocation through polynomial fit.
        '''
        idx   = self.kappa() < 0.9999999
        if sum(idx) < 1:
            print("No need to smooth. Households cannot hold capital.")
            return
        ## Step 1: Create features
        self.X = []
        ### Polynomial terms
        powers = range(1,degree)

        for n in range(self.params['nDims']):
            polynomials = [np.power(eval('self.' + self.stateVarList[n] + '()'), p) for p in powers]
            self.X.extend(polynomials)

        ### Interaction terms
        intersList =  list(itertools.combinations(range(self.params['nDims']),\
                                                  2))
        if self.params['nDims'] > 1:
            for pair in intersList:
                self.X.append(eval('self.' + self.stateVarList[pair[0]] + '()'\
                              + '* self.' + self.stateVarList[pair[1]] + '()'))

        ### Add ones
        self.X.append(np.ones(self.W().shape))

        ## Step 2: Find indices that need to be smoothed and smooth data
        idx   = self.kappa() < 0.9999999
        self.X = (np.array(self.X).T)[idx,:]

        #### Smooth q
        q_smoothed = self.q()[idx]
        coefs = np.linalg.lstsq(self.X, q_smoothed, rcond= -1)

        q_smoothed = np.matrix(self.X) * np.matrix(coefs[0]).T
        smoothedQ  = self.q().copy()
        smoothedQ[idx] = np.squeeze(np.array(q_smoothed))

        #### Smooth kappa

        kappa_smoothed = self.kappa()[idx]
        coefsKappa = np.linalg.lstsq(self.X, kappa_smoothed, rcond= -1)

        kappa_smoothed = np.matrix(self.X) * np.matrix(coefsKappa[0]).T
        smoothedKappa = self.kappa().copy()
        smoothedKappa[idx] = np.squeeze(np.array(kappa_smoothed))

        self.smoothDataCPP(smoothedQ, smoothedKappa)

    def computeStatDent(self, usePardiso = False, iparms = {}, explicit = False, dt = 0.1, tol = 1e-5, maxIters = 100000, verb = False, forceCompute = False):

        '''
        This function computes the stationary density *after* the program solves
        the model. It cannot be called unless the model is solved (i.e.
        self.status has to be 1).

        It would export density if computed successfully. It would also attempt
        to load the density in the preLoad folder if the file exists.
        '''

        ## Check whether model is solved.
        if not self.status == 1:
            ## Raise exception if status is not 1, i.e. model not solved.
            raise Exception('Model is not solved. Cannot compute stationary density.')
        if (self.params['chiUnderline'] < 0.05) or (self.params['chiUnderline'] < self.params['nu_newborn']) \
        or math.isclose(self.params['a_e'], self.params['a_h'], rel_tol = 0.01):
            warnings.warn('One of the following is prompting this warning:'\
            ' (1) experts’ required equity retention (chiUnderline) is very close to zero,'\
            ' (2) experts’ required equity retention (chiUnderline) is less than experts’ population share (nu_newborn),'\
            ' or (3) expert and household productivities (a_e, a_h) are very close to each other'\
            '. You may get a degenerate wealth distribution or may not '\
            'compute stationary density at all since the model is close to a complete markets model (cases 1 and 3)'\
            ' or the distribution becomes degenerate in case 2 (check the HKT paper).')
        ## if model is solved, start computing stationary density.

        ## The underlying tool is contained in stationaryDensityModules.py. It
        ## uses the fact that the Fokker Planck matrix is the transpose of
        ## the Feynman Kac matrix. Since the matrix is not invertible, we fix
        ## the first element in the solution vector to be 1. After that,
        ## we solve the sub-linear system and normalize afterwards.
        ## Also note that model could be solved on both log(w) and w grids.
        ## If model was solved on a log(w) grid, need to convert it to the
        ## corresponding w grid via interpolation.

        ## Try to load stationary density if it exists
        try:
            self.dent = np.loadtxt(self.params['preLoad'] + '/dent.txt')
        except:
            forceCompute = True

        self.model = {} ## dictionary to contain model info

        ## Attempt to compute density if the user sets forceCompute to True
        ## or the attempt to load fails.

        if forceCompute:
            self.model['muX'] = None; self.model['sigmaX'] = [];

            ## Step 1: Configure state space
            stateMat = []; stateMatLogW = [];

            for i in range(self.params['nDims']):
                ## Note that i starts at zero but our variable names start at 1.
                stateMat.append(np.linspace(np.min(self.stateMat.iloc[:,i]),
                                            np.max(self.stateMat.iloc[:,i]),
                                            np.unique(np.array(self.stateMat.iloc[:,i])).shape[0]) )

                if (abs( self.params['logW'] - (1)) < 0.00000001):
                    ## Model was solved on grid based on log(w). Need to use
                    ## interpolation.

                    if i == 0:
                        stateMatLogW.append( np.linspace(np.min(np.array(self.logW)),
                                                         np.max(np.array(self.logW)),
                                                         np.unique(np.array(self.stateMat.iloc[:,i])).shape[0]) )



                    else:
                        stateMat.append(np.linspace(np.min(self.stateMat.iloc[:,i]),
                                            np.max(self.stateMat.iloc[:,i]),
                                            np.unique(np.array(self.stateMat.iloc[:,i])).shape[0]))

            #### Outputs of Step 1:
            ####  stateMat:     a tuple that contains np.linspace arrays based on
            ####                w grid; will be used for stationary density computations.
            ####  stateMatLogW: a tuple that contains np.linspace arrays based on
            ####                log(w) grid. Will be used to create interpolants.
            ####  stateGrid:    a matrix based on w grid. Will be the grid on which
            ####                the interpolants will be applied.

            stateMat = tuple(stateMat); stateMatLogW = tuple(stateMatLogW)
            stateGrid = np.meshgrid(*stateMat,  indexing='ij'); stateGrid = [np.matrix(x.flatten(order='F')).T for x in stateGrid]
            stateGrid = np.concatenate(stateGrid, axis = 1)
            stateGrid[:, 0] = np.log( stateGrid[:, 0] )
            self.stateMatInput = stateMat


            ## Step 2: Convert muX and sigmaX from log(w) grid to w grid if necessary
            ##         and create interpolants to solve for stationary denisty.
            ##         Also, create boundary conditions dict.


            ## Step 2.1: Configure the drift for dict model

            if (abs( self.params['logW'] - (1)) < 0.00000001):
                self.model['muX'] = sdm.convertLogW(self.muX(), stateGrid, stateMatLogW)
            else:
                self.model['muX'] = np.matrix(self.muX())

            #### Boundary conditions dictionary
            bc = {}
            ##### Use natural boundaries
            bc['natural'] = True


            for i in range(self.params['nDims']):
                ## Iterate over dimension

                ## Step 2.1: Take care of sigmaX and convert from w to log(w)

                if (i == 0 and (abs( self.params['logW'] - (1)) < 0.00000001)):
                    #### Solved on log(w) grid; convert via interpolation for
                    #### sigmaX1 only

                    sigmaXn = sdm.convertLogW(self.sigmaXList[i], stateGrid, stateMatLogW)
                else:
                    #### No conversion needed, because either it was not solved on
                    #### log(w) grid or it's not state variable w

                    sigmaXn = self.sigmaXList[i]


                ## Step 2.3: Create interpolants for sigmaX

                #### Given a state variable, for each shock (iterant s), the loop
                #### will create an interpolant and append it to sigmaXnInterps.
                #### After that, it will create an anonymous function sigamXn and
                #### append it to model['sigmaX'], which starts out as an empty list.

                self.model['sigmaX'].append(np.matrix(sigmaXn))

            ## Step 3: Compute stationary density
            stateGrid[:, 0] = np.exp( stateGrid[:, 0] ) ## need to convert log (w) grid
                                                        ## back to w grid
            res = sdm.computeDent(stateGrid, self.model, bc, \
            usePardiso, iparms, explicit, \
            dt, tol, maxIters, verb, betterCP = False)

            self.dent = res[0]; self.FKmat = res[1];

        ## Export data
        np.savetxt(self.params['folderName'] + '/dent.txt', self.dent)

        ## Step 4: Compute cumulative distributions
        nRange = list(range(self.params['nDims']))
        for n in range(self.params['nDims']):
            ## Step 4.1 filter out the element that we want to sum over
            axes     = list(filter(lambda x: x != n,nRange))
            condDent = self.dent.reshape(self.gridSizeList, order = 'F').sum( axis = tuple(axes) )
            self.marginals[self.stateVarList[n]] = condDent.copy()

            cumden  = np.cumsum( self.marginals[ self.stateVarList[n] ] )
            cdf     = interpolate.interp1d(cumden, np.unique(eval('self.' + self.stateVarList[n] + '()' )),\
             fill_value= (np.unique(eval('self.' + self.stateVarList[n] + '()'))[1] ,  np.unique(eval('self.' + self.stateVarList[n] + '()'))[-2]), bounds_error = False)
            self.inverseCDFs[self.stateVarList[n]] = cdf

    def computeMoments(self, varNames = []):

        ## This method computes moments provided by the user. It will compute
        ## stationary density if the user has not already done so.
        ## It computes for the state variable(s) regardless of user input.
        ## Afterwards, it will compute the moments of the variables selected
        ## by user.

        if np.max(np.abs(self.dent)) < 0.000001:
            ## Stationary density not computed.
            self.computeStatDent()

        varNames      = [x + '()' for x in varNames if x not in self.stateVarList]
        varsToCompute = [x + '()' for x in self.stateVarList] + varNames

        for varName in varsToCompute:
            if self.var2Label[varName] in macroMomentsList:
                self.macroMoments[self.var2Label[varName]] =  computeMeanSd(eval('self.' + varName ), self.dent)
            else:
                self.apMoments[self.var2Label[varName]]    =  computeMeanSd(eval('self.' + varName ), self.dent)

    def computeCorrs(self, varNames = []):
        ## Compute Correlation
        varNames      = [x + '()' for x in varNames if x not in self.stateVarList]
        varsToCompute = [x + '()' for x in self.stateVarList] + varNames
        if len(varsToCompute) > 1:
            ## Only makes sense to compute correlation when more than one
            ## variable is selected.
            pairs         = list(itertools.combinations(varsToCompute, 2))
            self.corrs    = OrderedDict(( (pair[0].replace('()',''),pair[1].replace('()','')), \
            computeCorr(eval('self.' + pair[0]), eval('self.' + pair[1]), self.dent)) \
                                        for pair in pairs) ## using OrderedDict because we want to preserve the ordering of variables
        else:
            print('Only one variable selected. Cannot compute correlations.')

    def computeShockElas(self, pcts = {'W':[.1,.5,.9]}, points = np.matrix([]), T = 100, dt = 1, perturb = 'C', bc = {}):
        ## This method computes the shock elasticitie. By default, it selects
        ## the 10th, 50th, and 90th pcts of w and computes shock elasticites
        ## for T = 100 years with dt = 1 year. It perturbs aggregate consumption
        ## by default.

        ## Create empty dataframes to store shock elasticities for the perturbed variable
        #### Shock elasaticities data will be saved in a dictionary, while each entry
        #### is for a perturbed variable.

        self.expoElasMap[perturb]            = pd.DataFrame(columns = self.stateVarList + ['shock'] \
        + [str(t) for t in list(np.linspace(1,T*dt,T))])
        self.priceElasExpertsMap[perturb]    = pd.DataFrame(columns = self.stateVarList + ['shock'] \
        + [str(t) for t in list(np.linspace(1,T*dt,T))])
        self.priceElasHouseholdsMap[perturb] = pd.DataFrame(columns = self.stateVarList + ['shock'] \
        + [str(t) for t in list(np.linspace(1,T*dt,T))])

        ## Create input stateMat for shock elasticities, a tuple of ranges of the state space
        self.stateMatInput = []

        for i in range(self.params['nDims']):
            ## Note that i starts at zero but our variable names start at 1.
            self.stateMatInput.append(np.linspace(np.min(self.stateMat.iloc[:,i]),
                                        np.max(self.stateMat.iloc[:,i]),
                                        np.unique(np.array(self.stateMat.iloc[:,i])).shape[0]) )
        ## Create dictionary to store model
        if self.model is None:
            self.model = {}

        ## Find perturbed variables and create empty list to store poitns

        perturb = perturb + '()'
        allPcts = []

        ## Find points
        if points.shape[1] == 0:
            allPts = []
            for stateVar in self.stateVarList:
                if self.dent is None or np.max(self.dent) < 0.0001:
                    raise Exception("Stationary density not computed or degenerate.")
                allPts.append([self.inverseCDFs[stateVar](pct) for pct in pcts[stateVar]])
                allPcts.append(pcts[stateVar])
            self.x0 = np.matrix(list(itertools.product(*allPts)))
            allPcts = [list(x) for x in list(itertools.product(*allPcts))]
            self.pcts = pcts
        else:
            self.x0 = points

        ## Prepare inputs and create interpolants

        muXs          = []
        stateVols     = []
        SDFeVols      = []
        SDFhVols      = []
        sigmaCVols    = []
        stateVolsList = []
        sigmaXs       = []

        ## Perturbed variables
        ## Han: added to compute shock elasticity for wealth share
        if perturb == 'LogW()':
            self.calDriftDiffusionLogW()
        
        muM    = eval('self.mu' + perturb)
        sigmaM = eval('self.sigma' + perturb)

        for n in range(self.params['nDims']):

            ## Drifts of state variables
            fn = RegularGridInterpolator(self.stateMatInput, \
                                            self.muX()[:,n].reshape(self.gridSizeList, order = 'F'))
            muXs.append(fn)

            if n == 0:
                ## Drift of perturbed variable
                fn = RegularGridInterpolator(self.stateMatInput, \
                                             muM.reshape(self.gridSizeList, order = 'F'))

                self.model['muC'] = fn

                ## Drift of experts SDF
                fn = RegularGridInterpolator(self.stateMatInput, \
                                             self.muSe().reshape(self.gridSizeList, order = 'F'))
                self.model['muSe'] = fn

                ## Drift of hhs SDF
                fn = RegularGridInterpolator(self.stateMatInput, \
                                             self.muSh().reshape(self.gridSizeList, order = 'F'))
                self.model['muSh'] = fn

            for s in range(self.params['nShocks']):

                ## Diffusions of state variables
                fn = RegularGridInterpolator(self.stateMatInput, \
                                             self.sigmaXList[n][:,s].reshape(self.gridSizeList, order = 'F'))
                stateVols.append(fn)

                if n == 0:
                    ## Only need to do this once
                    fn = RegularGridInterpolator(self.stateMatInput, \
                                             self.sigmaSe()[:,s].reshape(self.gridSizeList, order = 'F'))
                    SDFeVols.append(fn)

                    fn = RegularGridInterpolator(self.stateMatInput, \
                                             self.sigmaSh()[:,s].reshape(self.gridSizeList, order = 'F'))
                    SDFhVols.append(fn)

                    fn = RegularGridInterpolator(self.stateMatInput, \
                                             sigmaM[:,s].reshape(self.gridSizeList, order = 'F'))
                    sigmaCVols.append(fn)

            stateVolsList.append(stateVols)
            def sigmaXfn(n):
                return lambda x: np.transpose([vol(x) for vol in stateVolsList[n] ])
            #sigmaXfn = lambda x: np.transpose([vol(x) for vol in stateVolsList[n]])
            sigmaXs.append(sigmaXfn(n))
            stateVols = []

            if n == 0:
                sigmaSefn = lambda x: np.transpose([vol(x) for vol in SDFeVols])
                self.model['sigmaSe'] = sigmaSefn

                sigmaShfn = lambda x: np.transpose([vol(x) for vol in SDFhVols])
                self.model['sigmaSh'] = sigmaShfn

                sigmaCfn = lambda x: np.transpose([vol(x) for vol in sigmaCVols])
                self.model['sigmaC'] = sigmaCfn

        self.model['sigmaX'] = sigmaXs
        self.model['muX']    = lambda x: np.transpose([mu(x) for mu in muXs])

        self.model['sigmaS'] = self.model['sigmaSe']
        self.model['muS']    = self.model['muSe']

        ###Boundary conditions
        ones = []; zeros = [];
        for n in range(self.params['nDims']):
            ones.append(1)
            zeros.append(0)

        if not bool(bc):
            bc['a0']     = 0;
            bc['first']  = np.matrix(ones, 'd')
            bc['second'] = np.matrix(zeros, 'd')
            bc['third']  = np.matrix(zeros, 'd')
            bc['level']  = np.matrix(zeros, 'd')
            bc['natural']= False

        self.model['T'] = T; self.model['dt'] = dt;
        modelInput = self.model.copy()

        self.expoElas, self.priceElasExperts, self.linSysExpo, self.linSysE = sem.computeElas(self.stateMatInput, modelInput, bc, self.x0)

        self.model['sigmaS'] = self.model['sigmaSh']
        self.model['muS']    = self.model['muSh']

        modelInput = self.model.copy()
        self.expoElas, self.priceElasHouseholds, self.linSysExpo, self.linSysH = sem.computeElas(self.stateMatInput, modelInput, bc, self.x0)

        #### After computing shock elasticities, add them to a map for better organization.
        stem = None
        for i in range(self.expoElas.firstType.shape[0]):
            for j in range(self.expoElas.firstType.shape[1]):
                if points.shape[1] == 0:
                    ## used quintiles
                    stem = [str(round(x * 100,0)) + '%' for x in allPcts[i]]
                else:
                    ## used points
                    stem = np.array(self.x0[i,:]).tolist()
                    stem = [str(x) for x in list(itertools.chain(*stem))]
                self.expoElasMap[perturb.replace('()','')].loc[i * self.expoElas.firstType.shape[1] + j] = stem \
                + [str(int(j + 1))] + list(np.squeeze(self.expoElas.firstType[i,j,:]))
                self.priceElasExpertsMap[perturb.replace('()','')].loc[i * self.expoElas.firstType.shape[1] + j] = stem \
                + [str(int(j + 1))] + list(np.squeeze(self.priceElasExperts.firstType[i,j,:]))
                self.priceElasHouseholdsMap[perturb.replace('()','')].loc[i * self.expoElas.firstType.shape[1] + j] = stem \
                + [str(int(j + 1))] + list(np.squeeze(self.priceElasHouseholds.firstType[i,j,:]))
## ================================================================= ##
## ================================================================= ##
## ================================================================= ##

## Section 3: It has the auxiliary functions for class Model() defined
## above.

def checkParams(params):
    '''
    This function checks the parameters and makes sure that the user has not
    put in a set of degenerate parameters.
    '''

    #### Check that numbers of discretization points are correctly given to
    #### the state variables with positive sigmas.
    if not ( ( abs(params['sigma_Z_norm']) > 0.00001 and params['nZ'] > 1 ) or \
             (math.isclose(params['sigma_Z_norm'], 0) and \
              math.isclose(params['nZ'], 0) ) ):

        beg = ('Trivial' if params['sigma_Z_norm'] < 0.000001 else 'Nontrivial' )
        end = ('Set the number of grid points to zero.' if params['sigma_Z_norm'] \
               < 0.000001 else 'Increase the number of grid points in g.' )
        errorMsg = (beg + ' volatility is assigned to g but {nG} grid points are assigned to g. ' + end).format( **params )

        raise Exception(errorMsg)

    if not ( ( abs(params['sigma_V_norm']) > 0.00001 and params['nV'] > 1 ) or \
             ( math.isclose(params['sigma_V_norm'], 0) and \
              math.isclose(params['nV'], 0) ) ):

        beg = ('Trivial' if params['sigma_V_norm'] < 0.000001 else 'Nontrivial' )
        end = ('Set the number of grid points to zero.' if params['sigma_V_norm'] \
               < 0.000001 else 'Increase the number of grid points in s.' )
        errorMsg = (beg + ' volatility is assigned to s but {nS} grid points are assigned to s. ' + end).format( **params )

        raise Exception(errorMsg)

    if not ( ( abs(params['sigma_Vtilde_norm']) > 0.00001 and params['nVtilde'] > 1 ) or \
             ( math.isclose(params['sigma_Vtilde_norm'], 0) and \
              math.isclose(params['nVtilde'], 0)) ) :

        beg = ('Trivial' if params['sigma_Vtilde_norm'] < 0.000001 else 'Nontrivial' )
        end = ('Set the number of grid points to zero.' if params['sigma_Vtilde_norm'] \
               < 0.000001 else 'Increase the number of grid points in Vtilde.' )
        errorMsg = (beg + ' volatility is assigned to Vtilde but {nVtilde} grid points are assigned to Vtilde. '  + end).format( **params )

        raise Exception(errorMsg)

    ## Forbid 4d models

    if ( ( abs(params['sigma_Vtilde_norm']) > 0.00001) and ( abs(params['sigma_Z_norm']) > 0.00001) \
    and ( abs(params['sigma_V_norm']) > 0.00001) ):
        raise Exception('Currently we do not support four state variables.')

    if math.isclose(params['numSds'], 0):
        errorMsg = 'Please insert a positive number for numSds'
        raise Exception(errorMsg)

    ## Check financial constraints.
    if math.isclose(params['equityIss'], 1, rel_tol = 1e-5):
        warnings.warn('equityIss is set to one (or Expert\'s constant equity retention constraint is selected). Equilibrium may not be reached.')

    ## Check parameters related to the algo
    if params['dt'] < 0.000001:
        wnMsg = 'Time step is {dt}; beware that a small time step could take excessive time.'.format( **params )
        warnings.warn(wnMsg)

    ## Check covariance matrix was given correctly
    sigmas = [(key, value) for key, value in params.items() if 'sigma' in key and 'norm' in key and value > 0]

    for i in range(1,len(sigmas)):
        if i < len(sigmas):
            ## This forloop is only effective when the number of shocks > 2
            for j in range(i):
                if not ('cov' + str(i + 1) + str(j + 1) in params):
                    raise Exception("Inputs for the covariance matrix are incorrect. Check the inputs for shock " + str(i+1) + ".")

    ## Make sure mean of Vtilde is strictly greater than zero if vol of idio vol is > 0
    if params['sigma_Vtilde_norm'] > 0 and params['Vtilde_bar'] <= 0.000000000001:
        raise Exception("Vol of idiosyncratic vol is positive implies mean of idiosyncratic vol has to be positive.")

    ## Make sure mean of V is strictly greater than zero if vol of stochastic vol is > 0
    if params['sigma_V_norm'] > 0 and params['V_bar'] <= 0.000000000001:
        raise Exception("Vol of stochastic vol is positive implies mean of stochastic vol has to be positive.")

    ## Check guesses if ONLY guesses are given
    if ('preLoad' not in params ) or (params['preLoad'] == 'None'):
        S = params['nWealth'] * (params['nZ'] if params['nZ'] > 0 else 1 )\
            * (params['nV'] if params['nV'] > 0 else 1 )\
            * (params['nVtilde'] if params['nVtilde'] > 0 else 1 )
        if 'xiEGuess' in params and ( (not (params['xiEGuess'].shape == (S,1))) and ( not (params['xiEGuess'].shape == (S,)) ) ):
            raise Exception("Dimensions of your guess of xiE is incorrect. It must be (" + str(int(S)) + ", 1) or (" + str(int(S)) + ",)")
        if 'xiHGuess' in params and ( (not (params['xiHGuess'].shape == (S,1))) and ( not (params['xiHGuess'].shape == (S,)) )):
            raise Exception("Dimensions of your guess of xiH is incorrect. It must be (" + str(int(S)) + ", 1) or (" + str(int(S)) + ",)")
        if 'kappaGuess' in params and ( (not (params['kappaGuess'].shape == (S,1))) and ( not (params['kappaGuess'].shape == (S,)) )):
            raise Exception("Dimensions of your guess of kappa is incorrect. It must be (" + str(int(S)) + ", 1) or (" + str(int(S)) + ",)")
        if 'chiGuess' in params and ( (not (params['chiGuess'].shape == (S,1))) and ( not (params['chiGuess'].shape == (S,)) )):
            raise Exception("Dimensions of your guess of chi is incorrect. It must be (" + str(int(S)) + ", 1) or (" + str(int(S)) + ",)")

def completeParams(params, reset = False):

    ## This function completes the parameters input. It completes:
    ## params['nDims']: number of dimensions based on the number of positive vols
    ## params['covij']: correlation matrix

    ## (1) convert negative a_h to -infty and fill in hhCap based on a_h

    if params['a_h'] < 0:
        params['a_h'] = -9999999999999999;
        params['hhCap'] = 0
    else:
        params['hhCap'] = 1

    ## (2) Fill in nDims
    ## Figure out what state variables
    ## are used and the total number of grid points

    S = params['nWealth'] * (params['nZ'] if params['nZ'] > 0 else 1 )\
        * (params['nV'] if params['nV'] > 0 else 1 )\
        * (params['nVtilde'] if params['nVtilde'] > 0 else 1 )

    gridSizeList = [x for x in [params['nWealth'], params['nZ'], params['nV'],\
                             params['nVtilde']] if x > 0.0000001]

    stateVarList = [x for x in ['Z','V','Vtilde'] if params['sigma_' + x + '_norm'] > 0]
    stateVarList = ['W'] + stateVarList

    params['nDims'] = len(stateVarList)
    params['nShocks'] = params['nDims'] ## force # shocks = # ndims

    ## (3) Fill in correlation matrix
    # params['cov11'] = 1; params['cov12'] = 0;
    # params['cov13'] = 0; params['cov14'] = 0

    # for i in range(1,4):
    #     if i < params['nShocks']:
    #         ## This forloop is only effective when the number of shocks > 2
    #         sumSquares = 0.0
    #         for j in range(i):
    #             sumSquares += np.power(params['cov' + str(i + 1) + str(j + 1)], 2.0)
    #         params['cov' + str(i + 1) + str(i + 1)] = np.sqrt(1.0 - sumSquares)

    #         ## Fill in zero for all j > #shocks
    #         for j in range(i + 1, 4):
    #             params['cov' + str(i + 1) + str(j + 1)] = 0

    #     else:
    #         ## Fill in zero for all i > #shocks
    #         for j in range(4):
    #             params['cov' + str(i + 1) + str(j + 1)] = 0
    ## (4) Fill in default guesses if their keys don't exist in the dict

    if 'preLoad' in params.keys() and (not (params['preLoad'] == 'None')):

        #### If 'preLoad' is given, load data from the folder
        suffix = 'final' if not 'suffix' in params.keys() else params['suffix']
        ## load in geuss for xiE
        if os.path.isfile(params['preLoad'] + '/xi_e_' + suffix + '.dat'):
            try:
                params['xiEGuess'] = np.array(np.loadtxt(params['preLoad'] + '/xi_e_' + suffix + '.dat')).reshape([S,1])
            except:
                raise Exception("Dimensions of your guess of xiE is incorrect. It must be (" + str(int(S)) + ", 1) or (" + str(int(S)) + ",)")
        else:
            warnings.warn('File ' + params['preLoad'] + '/xi_e_' + suffix + '.dat does not exit; fill in a vector of zeroes as guess xiE.')
            params['xiEGuess'] = np.full((S,1),0.0)

        ## load in guess for xiH
        if os.path.isfile(params['preLoad'] + '/xi_h_' + suffix + '.dat'):
            try:
                params['xiHGuess'] = np.array(np.loadtxt(params['preLoad'] + '/xi_h_' + suffix + '.dat')).reshape([S,1])
            except:
                raise Exception("Dimensions of your guess of xiH is incorrect. It must be (" + str(int(S)) + ", 1) or (" + str(int(S)) + ",)")
        else:
            warnings.warn('File ' + params['preLoad'] + '/xi_h_' + suffix + '.dat does not exit; fill in a vector of zeroes as guess xiH.')
            params['xiHGuess'] = np.full((S,1),0.0)

        ## load in guess for kappa
        if os.path.isfile(params['preLoad'] + '/kappa_' + suffix + '.dat'):
            try:
                params['kappaGuess'] = np.array(np.loadtxt(params['preLoad'] + '/kappa_' + suffix + '.dat')).reshape([S,1])
            except:
                raise Exception("Dimensions of your guess of kappa is incorrect. It must be (" + str(int(S)) + ", 1) or (" + str(int(S)) + ",)")
        else:
            warnings.warn('File ' + params['preLoad'] + '/kappa_' + suffix + '.dat does not exit; fill in w as guess for kappa.')
            params['kappaGuess'] = np.full((S,1),-1.0)

        ## load in guess for chi
        if os.path.isfile(params['preLoad'] + '/chi_' + suffix + '.dat'):
            try:
                params['chiGuess'] = np.array(np.loadtxt(params['preLoad'] + '/chi_' + suffix + '.dat')).reshape([S,1])
            except:
                raise Exception("Dimensions of your guess of chi is incorrect. It must be (" + str(int(S)) + ", 1) or (" + str(int(S)) + ",)")
        else:
            warnings.warn('File ' + params['preLoad'] + '/chi_' + suffix + '.dat does not exit; fill in a vector of chiUnderline as guess for chi.')
            params['chiGuess'] = np.full((S,1),-1.0)

    else:
        #### If 'preLoad' is not given, program will either use guesses given or default guesses
        #### By default, the guesses for the value functions will be zero vectors
        #### and chi will be set to the constraint and kappa = w.
        #### (Default values of chi and kappa will be handled by the c++ core code.)
        params['preLoad'] = 'None'
        if not 'xiEGuess' in params:
            params['xiEGuess'] = np.full((S,1),0.0)
        if not 'xiHGuess' in params:
            params['xiHGuess'] = np.full((S,1),0.0)
        if not 'kappaGuess' in params:
            params['kappaGuess'] = np.full((S,1),-1.0)
        if not 'chiGuess' in params:
            params['chiGuess'] = np.full((S,1),-1.0)

    return params, S, gridSizeList, stateVarList

def computeMeanSd(varData, h):

    ## This function computes mean and sd of a variable.

    ev    = round((varData * h).sum(),6)
    std   = np.sqrt( round(( (varData ** 2) * h ).sum()  -  ( ( varData * h ).sum() ** 2) ,9))

    return [ev, std]

def computeCorr(var1, var2, h):

    ## This function computes the correlation between two variables
    coVar  = ( var1 * var2 * h ).sum() - ( var1 * h ).sum() * ( var2 * h ).sum() ;
    std1   = np.sqrt( round( ( (var1 ** 2) * h ).sum()  -  ( ( var1 * h ).sum() ** 2),10 ))
    std2   = np.sqrt( round( ( (var2 ** 2) * h ).sum()  -  ( ( var2 * h ).sum() ** 2),10 ))

    ## The if statement catches constants (where std is 0)
    corRel = (0 if (math.isclose(std1, 0) or math.isclose(std2.sum(), 0)) else coVar / ( std1 * std2));

    return corRel


## ================================================================= ##
## Section 4: helper functions for drift and diffusion terms calculation

# Convert index of grid points to vectorized index
# Note points start from 0 not 1. 
@jit
def points_to_vec_index(points_index,gridSizeList):
    for i in range(len(points_index)):
        if i == 0:
            vec_index = points_index[i]
        else:
            vec_index = vec_index + points_index[i]*np.prod(gridSizeList[:i])
    return vec_index

# Convert vectorized index to index of grid points
# Note points start from 0 not 1. 
@jit
def vec_to_points_index(vec_index,gridSizeList):
    points_index = [0]*len(gridSizeList)
    for i in range(len(gridSizeList)):
        j = len(gridSizeList)-1 - i
        if j != 0:
            points_index[j] = np.floor(vec_index / np.prod(gridSizeList[:j]))
            vec_index = vec_index % np.prod(gridSizeList[:j])
        else:
            points_index[j] = vec_index
    return points_index

# Calculate derivative of G(X0,X1,...) wrt Xi, at grid point (n0,n1,...)
# Note that ni is between 0 and ni_max
@jit
def cal_derivative(G,i,vec_index,stateMat,gridSizeList):
    points_index = vec_to_points_index(vec_index,gridSizeList)
    points_index = np.array(points_index)
    
    ni = points_index[i]
    ni_max = gridSizeList[i]-1
    
    step_size = (stateMat[:,i].max() - stateMat[:,i].min())/ni_max

    # Use right derivative at left boundary
    if ni == 0:
        points_right_index = points_index.copy()
        points_right_index[i] += 1
        vec_right_index = points_to_vec_index(points_right_index,gridSizeList)
        return (G[vec_right_index] - G[vec_index])/step_size
    # Use left derivative at right boundary
    elif ni == ni_max:
        points_left_index = points_index.copy()
        points_left_index[i] -= 1
        vec_left_index = points_to_vec_index(points_left_index,gridSizeList)   
        return (G[vec_index] - G[vec_left_index])/step_size
    # Use central derivative at middle points
    else:
        points_right_index = points_index.copy()
        points_right_index[i] += 1
        vec_right_index = points_to_vec_index(points_right_index,gridSizeList)
        points_left_index = points_index.copy()
        points_left_index[i] -= 1
        vec_left_index = points_to_vec_index(points_left_index,gridSizeList)   
        return (G[vec_right_index] - G[vec_left_index])/(step_size*2)

# Apply (+1,-2,+1) weighting to 2nd order derivative when i=j
@jit
def cal_2nd_derivative(G,i,vec_index,stateMat,gridSizeList):
    points_index = vec_to_points_index(vec_index,gridSizeList)
    points_index = np.array(points_index)
    
    ni = points_index[i]
    ni_max = gridSizeList[i]-1
    
    step_size = (stateMat[:,i].max() - stateMat[:,i].min())/ni_max

    # Use the 2nd order derivative of the point to it at left boundary
    if ni == 0:
        points_right_index = points_index.copy()
        points_right_index[i] += 1
        vec_right_index = points_to_vec_index(points_right_index,gridSizeList)
        return cal_2nd_derivative(G,i,vec_right_index,stateMat,gridSizeList)
    # Use the 2nd order derivative of the point to it at right boundary
    elif ni == ni_max:
        points_left_index = points_index.copy()
        points_left_index[i] -= 1
        vec_left_index = points_to_vec_index(points_left_index,gridSizeList)   
        return cal_2nd_derivative(G,i,vec_left_index,stateMat,gridSizeList)
    # Use (+1,-2,+1) weighting at middle points
    else:
        points_right_index = points_index.copy()
        points_right_index[i] += 1
        vec_right_index = points_to_vec_index(points_right_index,gridSizeList)
        points_left_index = points_index.copy()
        points_left_index[i] -= 1
        vec_left_index = points_to_vec_index(points_left_index,gridSizeList)   
        return (G[vec_right_index] - 2*G[vec_index] + G[vec_left_index])/(step_size**2) 
    
# Calculate gradient of G at given points, where G is a function of states
@jit
def cal_gradient(G,vec_index,stateMat,gridSizeList):
    grad = np.zeros(len(gridSizeList))
    for i in range(len(gridSizeList)):
        grad[i] = cal_derivative(G,i,vec_index,stateMat,gridSizeList)
    return grad

# Calculate gradient of G at all grid points, where G is a function of states
@jit
def cal_gradient_grids(G,stateMat,gridSizeList):
    grad_grids = [cal_gradient(G,vec_index,stateMat,gridSizeList) for vec_index in range(len(stateMat))]
    return grad_grids

# Calculate hessian of G at given points, where G is a function of states
# G_grad is k x n, where k is number of vectorized grids and n is number of states
@jit
def cal_hessian(G,G_grad,vec_index,stateMat,gridSizeList):
    dim = len(gridSizeList)
    hessian = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            if i==j:
                hessian[i,j] = cal_2nd_derivative(G,i,vec_index,stateMat,gridSizeList)
            else:
                hessian[i,j] = cal_derivative(G_grad[:,i],j,vec_index,stateMat,gridSizeList)
    return hessian     

# Calculate hessian of G at all grid points, where G is a function of states
# G_grad is k x n, where k is number of vectorized grids and n is number of states
@jit
def cal_hessian_grids(G,G_grad,stateMat,gridSizeList):
    hessian_grids = [cal_hessian(G,G_grad,vec_index,stateMat,gridSizeList) for vec_index in range(len(stateMat))]
    return hessian_grids

@jit
def cal_muG_sigma_grids(G_grad,G_hess,muX,sigmaXList):
    n_grids = G_grad.shape[0]
    dim = G_grad.shape[1]
    muG_grids = np.zeros((n_grids))
    sigmaG_grids = np.zeros((n_grids,dim))
    for m in range(n_grids):
        # Get simgaX at a single point
        sigmaX_single = np.zeros((dim,dim))
        # Order of states
        for i in range(dim):
            # Order of shocks
            for j in range(dim):
                sigmaX_single[i][j] = sigmaXList[i][m][j]
        # Get muX,G_grad,G_hess at a single point
        muX_single = muX[m:m+1].T
        G_grad_single = G_grad[m:m+1].T
        G_hess_single = G_hess[m]
        # Calculate G_grad_single.T@muX_single in this way to avoid numba warning
        temp = 0
        for i in range(dim):
            temp = temp + G_grad_single[i,0]*muX_single[i,0]
        muG = temp + 0.5*np.trace(sigmaX_single.T@G_hess_single@sigmaX_single)
        sigmaG = G_grad_single.T@sigmaX_single
        muG_grids[m] = muG
        sigmaG_grids[m] = sigmaG
    return muG_grids,sigmaG_grids