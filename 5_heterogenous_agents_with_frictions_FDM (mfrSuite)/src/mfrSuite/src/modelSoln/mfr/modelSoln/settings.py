from collections import OrderedDict

########################
## Default parameters ##
########################

paramsDefault = OrderedDict({});

#######Parameters of the model#######
paramsDefault['nu_newborn']             = 0.1;
paramsDefault['lambda_d']               = 0.02;
paramsDefault['lambda_Z']               = 0.252;
paramsDefault['lambda_V']               = 0.156;
paramsDefault['lambda_Vtilde']          = 1.38;
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
paramsDefault['sigma_K_norm']           = 0.04;
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
paramsDefault['nVtilde']              = 0;   ### Program will ignore this parameter if nDims < 4
paramsDefault['wMin']                   = 0.01;
paramsDefault['wMax']                   = 0.99;

#######Parameters of IO#######
paramsDefault['folderName']             = 'model0'
paramsDefault['overwrite']              = 'Yes'
paramsDefault['exportFreq']             = 10000
paramsDefault['CGscale']                = 1.0
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
