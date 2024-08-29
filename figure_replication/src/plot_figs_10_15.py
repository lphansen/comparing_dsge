##Plot Figures 1-3
import os
import sys
import numpy as np
import pandas as pd
import pickle
np.set_printoptions(suppress=True, linewidth=200)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", font_scale=1.13, rc={"lines.linewidth": 3.5})
plt.rcParams['axes.formatter.useoffset'] = True
from utils_pde_shock_elasticity import computeElas

#Return neural networks solutions
def return_NN_solution(shock_expo, seed, chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h, delta_e, delta_h, lambda_d, nu,n_layers,units,iter_num, points_size,penalization, action_name):
        
        wMin = 0.01
        wMax = 0.99

        nWealth           = 180
        nZ                = 30
        nV                = 30
        
        wMin_t, wMax_t = [str("{:0.3f}".format(param)).replace('.', '', 1)  for param in [wMin, wMax]]
        domain_folder = 'nW_' + str(nWealth) + '_nZ_' + str(nZ) + '_nV_' + str(nV) + '_wMin_' + wMin_t + '_wMax_' + wMax_t

        parameter_list    = [chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h, delta_e, delta_h, lambda_d, nu]
        chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h, delta_e, delta_h, lambda_d, nu = [str("{:0.3f}".format(param)).replace('.', '', 1)  for param in parameter_list]
        model_folder = 'chiUnderline_' + chiUnderline + '_a_e_' + a_e + '_a_h_' + a_h  + '_gamma_e_' + gamma_e + '_gamma_h_' + gamma_h + '_rho_e_' + psi_e + '_rho_h_' + psi_h + '_delta_e_' + delta_e + '_delta_h_' + delta_h + '_lambda_d_' + lambda_d + '_nu_' + nu
        layer_folder =  'seed_' + str(seed) + '_n_layers_' + str(n_layers) + '_units_' + str(units) +'_points_size_' + str(points_size) + '_iter_num_' + str(iter_num) + '_penalization_' + str(penalization)

        outputdir = '../4_heterogenous_agents_with_frictions_NN/output/' + action_name + '/' + shock_expo + '/'+ domain_folder + '/' + model_folder + '/' + layer_folder + '/'

        eva_V_10 = np.load(outputdir + 'eva_V_10.npz')
        eva_V_50 = np.load(outputdir + 'eva_V_50.npz')
        eva_V_90 = np.load(outputdir + 'eva_V_90.npz')

        try:
            elasticity_logw = np.load(outputdir + 'elasticity_logw.npz', allow_pickle=True)
        except:
            elasticity_logw = None
        try:
            uncertaintye_priceelas = np.load(outputdir + 'uncertaintye_priceelas.npz', allow_pickle=True)
        except:
            uncertaintye_priceelas = None
        try:
            uncertaintyh_priceelas = np.load(outputdir + 'uncertaintyh_priceelas.npz', allow_pickle=True)
        except:
            uncertaintyh_priceelas = None

        W = np.load(outputdir + 'W_NN.npy')
        W = pd.DataFrame(W, columns = ['W'])
        Z = np.load(outputdir + 'Z_NN.npy')
        Z = pd.DataFrame(Z, columns = ['Z'])
        V = np.load(outputdir + 'V_NN.npy')
        V = pd.DataFrame(V, columns = ['V'])

        dent = np.load(outputdir + 'dent_NN.npy')
        dent = pd.DataFrame(dent, columns = ['dent'])
        dent = pd.concat([W,Z,V,dent['dent']], axis=1)
        dentW = dent.groupby('W').sum()['dent']
        dentV = dent.groupby('V').sum()['dent']

        try:
                elasticities_W_percentile_005 = np.load(outputdir + 'elasticities_W_percentile_0.05.npz', allow_pickle=True)
        except:
                elasticities_W_percentile_005 = None
        try:
                elasticities_W_percentile_01 = np.load(outputdir + 'elasticities_W_percentile_0.1.npz', allow_pickle=True)
        except:
                elasticities_W_percentile_01 = None
        try:
                elasticities_W_percentile_05 = np.load(outputdir + 'elasticities_W_percentile_0.5.npz', allow_pickle=True)
        except:
                elasticities_W_percentile_05 = None

        return {'eva_V_10':eva_V_10, 'eva_V_50':eva_V_50, 'eva_V_90':eva_V_90, 'dentW':dentW, 'dentV':dentV,
                'W':W, 'Z':Z, 'V':V, 'elasticity_logw':elasticity_logw,  'uncertaintye_priceelas':uncertaintye_priceelas, 'uncertaintyh_priceelas':uncertaintyh_priceelas,\
                'elasticities_W_percentile_005':elasticities_W_percentile_005, 'elasticities_W_percentile_01':elasticities_W_percentile_01, 'elasticities_W_percentile_05':elasticities_W_percentile_05}


#Return finite difference solution
def return_fdm_solution(shock_expo, dt, nW, chiUnderline, a_e, a_h, gamma_e, gamma_h, rho_e, rho_h, delta_e, delta_h, lambda_d, nu, action_name, nZ):

    rho_e_t = "{:0.3f}".format(rho_e)
    rho_h_t = "{:0.3f}".format(rho_h)
    gamma_e_t = "{:0.3f}".format(gamma_e)
    gamma_h_t = "{:0.3f}".format(gamma_h)
    a_e_t = "{:0.3f}".format(a_e)
    a_h_t = "{:0.3f}".format(a_h)
    chiUnderline_t = "{:0.3f}".format(chiUnderline)
    delta_e_t = "{:0.3f}".format(delta_e)
    delta_h_t = "{:0.3f}".format(delta_h)
    lambda_d_t = "{:0.3f}".format(lambda_d)
    nu_t = "{:0.3f}".format(nu)

    folder_name = ('../5_heterogenous_agents_with_frictions_FDM (mfrSuite)/output/' + action_name + '/' + shock_expo + '/dt_'+str(dt)+'/nW_'+str(nW)+'_nZ_'+str(nZ)+'/chiUnderline_' + chiUnderline_t + '/a_e_' + a_e_t + '_a_h_' + a_h_t  + '/gamma_e_' + gamma_e_t + '_gamma_h_' + gamma_h_t + '/rho_e_' + rho_e_t + '_rho_h_' + rho_h_t + '/delta_e_' + delta_e_t + '_delta_h_' + delta_h_t + '/lambda_d_' + lambda_d_t + '_nu_' + nu_t)

    def read_dat(filename):
        with open(folder_name + '/'+filename+'.dat', 'r') as file:
            data = [float(line.strip()) for line in file if line.strip()]
        return pd.DataFrame(data, columns=[filename])
    
    W = read_dat('W')
    Z = read_dat('Z')
    
    with open(folder_name + '/PiE_final_capital.pkl' , 'rb') as file:
        PiE_final_capital = pickle.load(file)
    with open(folder_name + '/kappa_final.pkl' , 'rb') as file:
        kappa_final = pickle.load(file)

    dents = pd.read_csv(folder_name + '/dent.txt',names = ['dent'])
    dents = pd.concat([W,Z,dents], axis=1)
    dents = dents.groupby('W').sum()['dent']

    try:
        elasticities_W0 = np.load(folder_name+'/elasticity_W_0.npz',allow_pickle=True)
        elasticities_W1 = np.load(folder_name+'/elasticity_W_1.npz',allow_pickle=True)
        elasticities_W2 = np.load(folder_name+'/elasticity_W_2.npz',allow_pickle=True)
    except:
        elasticities_W0 = None
        elasticities_W1 = None
        elasticities_W2 = None
    
    return {'W':W, 'Z':Z, 'PiE_final_capital':PiE_final_capital, 'dents':dents, 'kappa_final':kappa_final, 'elasticities_W0':elasticities_W0, 'elasticities_W1':elasticities_W1, 'elasticities_W2':elasticities_W2}
            


#Load Neural Network Results
print('Loading results...')
try: 
    modelRF_lower = return_NN_solution(shock_expo = 'lower_triangular', seed = 256, chiUnderline = 1.0, a_e = 0.0922, a_h = 0.0, gamma_e = 4.0, gamma_h = 8.0, psi_e = 1.0, psi_h = 1.0, delta_e = 0.0115, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, n_layers = 2, units = 16, iter_num = 5, points_size = 10, penalization = 10000, action_name = 'neural_net')
    modelRF_upper = return_NN_solution(shock_expo = 'upper_triangular', seed = 256, chiUnderline = 1.0, a_e = 0.0922, a_h = 0.0, gamma_e = 4.0, gamma_h = 8.0, psi_e = 1.0, psi_h = 1.0, delta_e = 0.0115, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, n_layers = 2, units = 16, iter_num = 5, points_size = 10, penalization = 10000, action_name = 'neural_net')
    modelSG_lower = return_NN_solution(shock_expo = 'lower_triangular', seed = 256, chiUnderline = 0.2, a_e = 0.0922, a_h = 0.0, gamma_e = 4.0, gamma_h = 8.0, psi_e = 1.0, psi_h = 1.0, delta_e = 0.0115, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, n_layers = 2, units = 16, iter_num = 5, points_size = 10, penalization = 10000, action_name = 'neural_net')
    modelSG_upper = return_NN_solution(shock_expo = 'upper_triangular', seed = 256, chiUnderline = 1.0, a_e = 0.0922, a_h = 0.0, gamma_e = 4.0, gamma_h = 8.0, psi_e = 1.0, psi_h = 1.0, delta_e = 0.0115, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, n_layers = 2, units = 16, iter_num = 5, points_size = 10, penalization = 10000, action_name = 'neural_net')
    modelPR_lower = return_NN_solution(shock_expo = 'lower_triangular', seed = 256, chiUnderline = 0.00001, a_e = 0.0922, a_h = 0.0, gamma_e = 4.0, gamma_h = 8.0, psi_e = 1.0, psi_h = 1.0, delta_e = 0.0115, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, n_layers = 2, units = 16, iter_num = 5, points_size = 10, penalization = 10000, action_name = 'neural_net')
    modelPR_upper = return_NN_solution(shock_expo = 'upper_triangular', seed = 256, chiUnderline = 0.00001, a_e = 0.0922, a_h = 0.0, gamma_e = 4.0, gamma_h = 8.0, psi_e = 1.0, psi_h = 1.0, delta_e = 0.0115, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, n_layers = 2, units = 16, iter_num = 5, points_size = 10, penalization = 10000, action_name = 'neural_net')

    modelRF_lower_gammae_3 = return_NN_solution(shock_expo = 'lower_triangular',seed = 256, chiUnderline = 1.0, a_e = 0.0922, a_h = 0.0, gamma_e = 3.0, gamma_h = 8.0, psi_e = 1.0, psi_h = 1.0, delta_e = 0.0115, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, n_layers = 2, units = 16, iter_num = 5, points_size = 10, penalization = 10000, action_name = 'neural_net')
    modelRF_lower_gammae_5 = return_NN_solution(shock_expo = 'lower_triangular',seed = 256, chiUnderline = 1.0, a_e = 0.0922, a_h = 0.0, gamma_e = 5.0, gamma_h = 8.0, psi_e = 1.0, psi_h = 1.0, delta_e = 0.0115, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, n_layers = 2, units = 16, iter_num = 5, points_size = 10, penalization = 10000, action_name = 'neural_net')

    modelSG_lower_gammae_3 = return_NN_solution(shock_expo = 'lower_triangular',seed = 256, chiUnderline = 0.2, a_e = 0.0922, a_h = 0.0, gamma_e = 3.0, gamma_h = 8.0, psi_e = 1.0, psi_h = 1.0, delta_e = 0.0115, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, n_layers = 2, units = 16, iter_num = 5, points_size = 10, penalization = 10000, action_name = 'neural_net')
    modelSG_lower_gammae_5 = return_NN_solution(shock_expo = 'lower_triangular',seed = 256, chiUnderline = 0.2, a_e = 0.0922, a_h = 0.0, gamma_e = 5.0, gamma_h = 8.0, psi_e = 1.0, psi_h = 1.0, delta_e = 0.0115, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, n_layers = 2, units = 16, iter_num = 5, points_size = 10, penalization = 10000, action_name = 'neural_net')

    modelPR_lower_gammae_3 = return_NN_solution(shock_expo = 'lower_triangular',seed = 256, chiUnderline = 0.00001, a_e = 0.0922, a_h = 0.0, gamma_e = 3.0, gamma_h = 8.0, psi_e = 1.0, psi_h = 1.0, delta_e = 0.0115, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, n_layers = 2, units = 16, iter_num = 5, points_size = 10, penalization = 10000, action_name = 'neural_net')
    modelPR_lower_gammae_5 = return_NN_solution(shock_expo = 'lower_triangular',seed = 256, chiUnderline = 0.00001, a_e = 0.0922, a_h = 0.0, gamma_e = 5.0, gamma_h = 8.0, psi_e = 1.0, psi_h = 1.0, delta_e = 0.0115, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, n_layers = 2, units = 16, iter_num = 5, points_size = 10, penalization = 10000, action_name = 'neural_net')
except:
    print("Please run the model first.")

#Load Finite Difference Results
try:
    model_070_lower_triangular = return_fdm_solution(shock_expo = 'lower_triangular',dt = 0.01, nW = 1800, nZ = 30, chiUnderline = 1.0, a_e = 0.0922, a_h = 0.070, gamma_e = 2.0, gamma_h = 2.0, rho_e = 1.0, rho_h = 1.0, delta_e = 0.03, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, action_name = 'finite_difference')
    model_080_lower_triangular = return_fdm_solution(shock_expo = 'lower_triangular',dt = 0.01, nW = 1800, nZ = 30, chiUnderline = 1.0, a_e = 0.0922, a_h = 0.080, gamma_e = 2.0, gamma_h = 2.0, rho_e = 1.0, rho_h = 1.0, delta_e = 0.03, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, action_name = 'finite_difference')
    model_085_lower_triangular = return_fdm_solution(shock_expo = 'lower_triangular',dt = 0.01, nW = 1800, nZ = 30, chiUnderline = 1.0, a_e = 0.0922, a_h = 0.085, gamma_e = 2.0, gamma_h = 2.0, rho_e = 1.0, rho_h = 1.0, delta_e = 0.03, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, action_name = 'finite_difference')

    model_070_upper_triangular = return_fdm_solution(shock_expo = 'upper_triangular',dt = 0.01, nW = 1800, nZ = 30, chiUnderline = 1.0, a_e = 0.0922, a_h = 0.070, gamma_e = 2.0, gamma_h = 2.0, rho_e = 1.0, rho_h = 1.0, delta_e = 0.03, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, action_name = 'finite_difference')
    model_080_upper_triangular = return_fdm_solution(shock_expo = 'upper_triangular',dt = 0.01, nW = 1800, nZ = 30, chiUnderline = 1.0, a_e = 0.0922, a_h = 0.080, gamma_e = 2.0, gamma_h = 2.0, rho_e = 1.0, rho_h = 1.0, delta_e = 0.03, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, action_name = 'finite_difference')
    model_085_upper_triangular = return_fdm_solution(shock_expo = 'upper_triangular',dt = 0.01, nW = 1800, nZ = 30, chiUnderline = 1.0, a_e = 0.0922, a_h = 0.085, gamma_e = 2.0, gamma_h = 2.0, rho_e = 1.0, rho_h = 1.0, delta_e = 0.03, delta_h = 0.01, lambda_d = 0.0, nu = 0.1, action_name = 'finite_difference')
except:
    print("Please run the model first.")

colors = ['#1f77b4', '#d62728', 'green']

#Figure 10
print('Figure 10')
fig, axes = plt.subplots(1,3, figsize=(12,4))
W_dense = np.unique(modelRF_lower['eva_V_10']['W'])
W_sparse = np.unique(modelRF_lower['W'].values)
sns.lineplot(x = W_dense, y = modelRF_lower_gammae_3['eva_V_10']['PiE_NN'][:,0], ax = axes[0], label = '$Z^2$ 10th percentile',color=colors[0])
sns.lineplot(x = W_dense, y = modelRF_lower_gammae_3['eva_V_90']['PiE_NN'][:,0], ax = axes[0], label = '$Z^2$ 90th percentile',color=colors[1])
axes[0].legend()
ax2 = axes[0].twinx()
sns.lineplot(x = W_sparse, y = modelRF_lower_gammae_3['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
ax2.set_ylim([0,0.03])
axes[0].set_title(r'$\gamma_e = 3$', fontsize=20)

sns.lineplot(x = W_dense, y = modelRF_lower['eva_V_10']['PiE_NN'][:,0], ax = axes[1],color=colors[0])
sns.lineplot(x = W_dense, y = modelRF_lower['eva_V_90']['PiE_NN'][:,0], ax = axes[1],color=colors[1])
ax2 = axes[1].twinx()
sns.lineplot(x = W_sparse, y = modelRF_lower['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
ax2.set_ylim([0,0.03])
axes[1].set_title(r'$\gamma_e = 4$', fontsize=20)

sns.lineplot(x = W_dense, y = modelRF_lower_gammae_5['eva_V_10']['PiE_NN'][:,0], ax = axes[2],color=colors[0])
sns.lineplot(x = W_dense, y = modelRF_lower_gammae_5['eva_V_90']['PiE_NN'][:,0], ax = axes[2],color=colors[1])
ax2 = axes[2].twinx()
sns.lineplot(x = W_sparse, y = modelRF_lower_gammae_5['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
ax2.set_ylim([0,0.03])
axes[2].set_title(r'$\gamma_e = 5$', fontsize=20)

for ax in axes.flatten():
    ax.set_xlim(0,1.0)
    ax.grid(False)
    ax.set_ylim([0.0,1.0])
    ax.set_xlabel('W', fontsize=20)
# plt.suptitle('Experts capital risk prices')  

plt.subplots_adjust(wspace=0.3, hspace=0.25)
plt.tight_layout()
fig.savefig("figures/figure_10.pdf")

#Figure 11
print('Figure 11')
fig, axes = plt.subplots(2,3, figsize=(12, 6))
W_large = modelPR_lower['eva_V_50']['W']
W_small = np.unique(modelPR_lower['W'].values)


sns.lineplot(x=W_large, y=modelPR_lower_gammae_3['eva_V_10']['PiE_NN'][:,0], ax = axes[0,0], label=r'$Z^2$ 10th percentile', color=colors[0])
sns.lineplot(x=W_large, y=modelPR_lower_gammae_3['eva_V_90']['PiE_NN'][:,0], ax = axes[0,0], label=r'$Z^2$ 90th percentile', color=colors[1])
ax2 = axes[0,0].twinx()
sns.lineplot(x = W_sparse, y = modelPR_lower_gammae_3['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
axes[0,0].set_title("$\gamma_e = 3$", fontsize=20)
axes[0,0].set_ylim([-0.01,2.3])
axes[0,0].legend(prop={'size': 14})
ax2.set_ylim([0,0.2])
ax2.set_ylim([0,0.03])

sns.lineplot(x=W_large, y=modelPR_lower['eva_V_10']['PiE_NN'][:,0], ax = axes[0,1], color=colors[0])
sns.lineplot(x=W_large, y=modelPR_lower['eva_V_90']['PiE_NN'][:,0], ax = axes[0,1], color=colors[1])
ax2 = axes[0,1].twinx()
sns.lineplot(x = W_sparse, y = modelPR_lower['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
axes[0,1].set_title("$\gamma_e = 4$", fontsize=20)
axes[0,1].set_ylim([-0.01,2.3])
ax2.set_ylim([0,0.03])


sns.lineplot(x=W_large, y=modelPR_lower_gammae_5['eva_V_10']['PiE_NN'][:,0], ax = axes[0,2], color=colors[0])
sns.lineplot(x=W_large, y=modelPR_lower_gammae_5['eva_V_90']['PiE_NN'][:,0], ax = axes[0,2], color=colors[1])
ax2 = axes[0,2].twinx()
sns.lineplot(x = W_sparse, y = modelPR_lower_gammae_5['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
axes[0,2].set_title("$\gamma_e = 5$", fontsize=20)
axes[0,2].set_ylim([-0.01,2.3])
ax2.set_ylim([0,0.5])

for ax in axes.flatten():
    ax.set_xlim(0,1.0)
    ax.grid(False)
    ax.set_ylim(0,0.3)

sns.lineplot(x=W_large, y=modelPR_lower_gammae_3['eva_V_50']['chi_NN'][:,0], ax = axes[1,0])
ax2 = axes[1,0].twinx()
sns.lineplot(x = W_sparse, y = modelPR_lower_gammae_3['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
axes[1,0].set_ylim([0,1.1])
axes[1,0].set_xlabel('w', fontsize=20)
ax2.set_ylim([0,0.03])


sns.lineplot(x=W_large, y=modelPR_lower['eva_V_50']['chi_NN'][:,0], ax = axes[1,1])
ax2 = axes[1,1].twinx()
sns.lineplot(x = W_sparse, y = modelPR_lower['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
axes[1,1].set_ylim([0,1.1])
axes[1,1].set_xlabel('w', fontsize=20)
ax2.set_ylim([0,0.03])

sns.lineplot(x=W_large, y=modelPR_lower_gammae_5['eva_V_50']['chi_NN'][:,0], ax = axes[1,2])
ax2 = axes[1,2].twinx()
sns.lineplot(x = W_sparse, y = modelPR_lower_gammae_5['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
axes[1,2].set_ylim([0,1.1])
axes[1,2].set_xlabel('w', fontsize=20)
ax2.set_ylim([0,0.5])
for ax in axes.flatten()[3:]:
    ax.set_xlim(0,1.0)
    ax.grid(False)
    ax.set_ylim(0,1.2)

axes[0,0].get_legend().remove()
fig.savefig("figures/figure_11.pdf",transparent=False)

#Figure 12
print('Figure 12')
fig, axes = plt.subplots(2,3, figsize=(12, 6))
W_large = modelSG_lower['eva_V_50']['W']
W_small = np.unique(modelSG_lower['W'].values)


sns.lineplot(x=W_large, y=modelSG_lower_gammae_3['eva_V_10']['PiE_NN'][:,0], ax = axes[0,0], label=r'$Z^2$ 10th percentile', color=colors[0])
sns.lineplot(x=W_large, y=modelSG_lower_gammae_3['eva_V_90']['PiE_NN'][:,0], ax = axes[0,0], label=r'$Z^2$ 90th percentile', color=colors[1])
ax2 = axes[0,0].twinx()
sns.lineplot(x = W_sparse, y = modelSG_lower_gammae_3['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
axes[0,0].set_title("$\gamma_e = 3$", fontsize=20)
axes[0,0].set_ylim([-0.01,2.3])
axes[0,0].legend(prop={'size': 14})
ax2.set_ylim([0,0.2])
ax2.set_ylim([0,0.03])

sns.lineplot(x=W_large, y=modelSG_lower['eva_V_10']['PiE_NN'][:,0], ax = axes[0,1], color=colors[0])
sns.lineplot(x=W_large, y=modelSG_lower['eva_V_90']['PiE_NN'][:,0], ax = axes[0,1], color=colors[1])
ax2 = axes[0,1].twinx()
sns.lineplot(x = W_sparse, y = modelSG_lower['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
axes[0,1].set_title("$\gamma_e = 4$", fontsize=20)
axes[0,1].set_ylim([-0.01,2.3])
ax2.set_ylim([0,0.03])


sns.lineplot(x=W_large, y=modelSG_lower_gammae_5['eva_V_10']['PiE_NN'][:,0], ax = axes[0,2], color=colors[0])
sns.lineplot(x=W_large, y=modelSG_lower_gammae_5['eva_V_90']['PiE_NN'][:,0], ax = axes[0,2], color=colors[1])
ax2 = axes[0,2].twinx()
sns.lineplot(x = W_sparse, y = modelSG_lower_gammae_5['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
axes[0,2].set_title("$\gamma_e = 5$", fontsize=20)
axes[0,2].set_ylim([-0.01,2.3])
ax2.set_ylim([0,0.1])

for ax in axes.flatten():
    ax.set_xlim(0,1.0)
    ax.grid(False)
    ax.set_ylim(0,0.4)

sns.lineplot(x=W_large, y=modelSG_lower_gammae_3['eva_V_50']['chi_NN'][:,0], ax = axes[1,0])
ax2 = axes[1,0].twinx()
sns.lineplot(x = W_sparse, y = modelSG_lower_gammae_3['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
axes[1,0].set_ylim([0,1.1])
axes[1,0].set_xlabel('w', fontsize=20)
ax2.set_ylim([0,0.03])


sns.lineplot(x=W_large, y=modelSG_lower['eva_V_50']['chi_NN'][:,0], ax = axes[1,1])
ax2 = axes[1,1].twinx()
sns.lineplot(x = W_sparse, y = modelSG_lower['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
axes[1,1].set_ylim([0,1.1])
axes[1,1].set_xlabel('w', fontsize=20)
ax2.set_ylim([0,0.03])

sns.lineplot(x=W_large, y=modelSG_lower_gammae_5['eva_V_50']['chi_NN'][:,0], ax = axes[1,2])
ax2 = axes[1,2].twinx()
sns.lineplot(x = W_sparse, y = modelSG_lower_gammae_5['dentW'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
axes[1,2].set_ylim([0,1.1])
axes[1,2].set_xlabel('w', fontsize=20)
ax2.set_ylim([0,0.1])
for ax in axes.flatten()[3:]:
    ax.set_xlim(0,1.0)
    ax.grid(False)
    ax.set_ylim(0,1.2)

axes[0,0].get_legend().remove()

fig.savefig("figures/figure_12.pdf",transparent=False)

#Figure 13
print('Figure 13')
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
W = model_070_lower_triangular['W']['W'].unique()
W2 = model_070_upper_triangular['W']['W'].unique()

sns.lineplot(x = W, y = model_070_lower_triangular['PiE_final_capital'], ax = axes[0,0])
ax2 = axes[0,0].twinx()
sns.lineplot(x = W2, y = model_070_upper_triangular['dents'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
ax2.set_yticklabels([])
axes[0,0].set_title(r"$\alpha_h=0.070$", fontsize=20)
axes[0,0].set_ylim(0,1.0)
axes[0,0].set_xlim(0,1.0)
ax2.set_ylim([0,0.016])

sns.lineplot(x = W, y = model_080_lower_triangular['PiE_final_capital'], ax = axes[0,1])
ax2 = axes[0,1].twinx()
sns.lineplot(x = W2, y = model_080_upper_triangular['dents'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
ax2.set_yticklabels([])
axes[0,1].set_title(r"$\alpha_h=0.080$", fontsize=20)
axes[0,1].set_ylim(0,1.0)
axes[0,1].set_xlim(0,1.0)
ax2.set_ylim([0,0.016])

sns.lineplot(x = W, y = model_085_lower_triangular['PiE_final_capital'], ax = axes[0,2])
ax2 = axes[0,2].twinx()
sns.lineplot(x = W2, y = model_085_upper_triangular['dents'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
ax2.set_yticklabels([])
axes[0,2].set_title(r"$\alpha_h=0.085$", fontsize=20)
axes[0,2].set_ylim(0,1.1)
axes[0,2].set_xlim(0,1.0)
ax2.set_ylim([0,0.016])

sns.lineplot(x = W2, y = model_070_upper_triangular['kappa_final'], ax = axes[1,0])
ax2 = axes[1,0].twinx()
sns.lineplot(x = W2, y = model_070_upper_triangular['dents'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
ax2.set_yticklabels([])
axes[1,0].set_ylim(0,1.1)
axes[1,0].set_xlim(0,1.0)
axes[1,0].set_xlabel('W')
ax2.set_ylim([0,0.016])

sns.lineplot(x = W2, y = model_080_upper_triangular['kappa_final'], ax = axes[1,1])
ax2 = axes[1,1].twinx()
sns.lineplot(x = W2, y = model_080_upper_triangular['dents'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
ax2.set_yticklabels([])
axes[1,1].set_ylim(0,1.1)
axes[1,1].set_xlim(0,1.0)
axes[1,1].set_xlabel('W')
ax2.set_ylim([0,0.016])

sns.lineplot(x = W2, y = model_085_upper_triangular['kappa_final'], ax = axes[1,2])
ax2 = axes[1,2].twinx()
sns.lineplot(x = W2, y =  model_085_upper_triangular['dents'].values, ax = ax2, ls='--', color='grey',lw=2.0, alpha=0.5)
ax2.grid(False)
ax2.set_yticks([])
ax2.set_yticklabels([])
axes[1,2].set_ylim(0,1.1)
axes[1,2].set_xlim(0,1.0)
axes[1,2].set_xlabel('W')
ax2.set_ylim([0,0.016])

for ax in axes.flatten():
    ax.grid(False)

plt.tight_layout()
fig.savefig("figures/figure_13.pdf")

#Figure 14
print('Figure 14')
fig, axes = plt.subplots(2,2, figsize=(8,7))

sns.lineplot(modelRF_lower['elasticities_W_percentile_005']['exposure_elasticity'], ax = axes[0,0],color = colors[0])
sns.lineplot(modelRF_lower['elasticities_W_percentile_01']['exposure_elasticity'], ax = axes[0,0],color = colors[1])
sns.lineplot(modelRF_lower['elasticities_W_percentile_05']['exposure_elasticity'], ax = axes[0,0],color = colors[2])
axes[0,0].set_title(r'Model RF')

sns.lineplot(modelSG_lower['elasticities_W_percentile_005']['exposure_elasticity'], ax = axes[1,1],color = colors[0])
sns.lineplot(modelSG_lower['elasticities_W_percentile_01']['exposure_elasticity'], ax = axes[1,1],color = colors[1])
sns.lineplot(modelSG_lower['elasticities_W_percentile_05']['exposure_elasticity'], ax = axes[1,1],color = colors[2])
axes[1,1].set_title(r'Model SG')
axes[1,1].set_xlabel('Years')

sns.lineplot(modelPR_lower['elasticities_W_percentile_005']['exposure_elasticity'], ax = axes[1,0],color = colors[0])
sns.lineplot(modelPR_lower['elasticities_W_percentile_01']['exposure_elasticity'], ax = axes[1,0],color = colors[1])
sns.lineplot(modelPR_lower['elasticities_W_percentile_05']['exposure_elasticity'], ax = axes[1,0],color = colors[2])
axes[1,0].set_title(r'Model PR')
axes[1,0].set_xlabel('Years')

sns.lineplot(x=np.linspace(0,49,480), y=model_080_lower_triangular['elasticities_W0']['exposure_elasticity'], label='W 5th percentile', ls='-',color = colors[0], ax = axes[0,1])
sns.lineplot(x=np.linspace(0,49,480), y=model_080_lower_triangular['elasticities_W1']['exposure_elasticity'], label='W 10th percentile', ls='-',color = colors[1], ax=axes[0,1])
sns.lineplot(x=np.linspace(0,49,480), y=model_080_lower_triangular['elasticities_W2']['exposure_elasticity'], label='W Median', ls='-',color = colors[2], ax=axes[0,1])


for ax in axes.flatten():
    ax.set_xlim(0,40)
    ax.grid(False)
    ax.set_ylim([0,0.018])

axes[0,1].set_ylim(0,0.4)
# plt.suptitle('Experts Wealth Share Exposure Elasticities')  

plt.subplots_adjust(wspace=0.3, hspace=0.25)
plt.tight_layout()
fig.savefig("figures/figure_14.pdf")

#Figure 15
print('Figure 15')
fig, axes = plt.subplots(2,2, figsize=(8, 7))
sns.lineplot(modelPR_upper['uncertaintye_priceelas']['price_elasticity'].item().secondType[0,1,:], ax = axes[0,0],color=colors[0])
sns.lineplot(modelPR_upper['uncertaintye_priceelas']['price_elasticity'].item().secondType[1,1,:], ax = axes[0,0],color='red')
sns.lineplot(modelPR_upper['uncertaintye_priceelas']['price_elasticity'].item().secondType[2,1,:], ax = axes[0,0],color=colors[2])
axes[0,0].set_title("PR")
axes[0,0].set_ylabel("Experts")
axes[0,0].set_ylim([0,0.15])

sns.lineplot(modelSG_upper['uncertaintye_priceelas']['price_elasticity'].item().secondType[0,1,:], ax = axes[0,1],color=colors[0])
sns.lineplot(modelSG_upper['uncertaintye_priceelas']['price_elasticity'].item().secondType[1,1,:], ax = axes[0,1],color='red')
sns.lineplot(modelSG_upper['uncertaintye_priceelas']['price_elasticity'].item().secondType[2,1,:], ax = axes[0,1],color=colors[2])
axes[0,1].set_title("SG")
axes[0,1].set_ylim([0,0.15])

sns.lineplot(modelPR_upper['uncertaintyh_priceelas']['price_elasticity'].item().secondType[0,1,:], ax = axes[1,0],color=colors[0])
sns.lineplot(modelPR_upper['uncertaintyh_priceelas']['price_elasticity'].item().secondType[1,1,:], ax = axes[1,0],color='red')
sns.lineplot(modelPR_upper['uncertaintyh_priceelas']['price_elasticity'].item().secondType[2,1,:], ax = axes[1,0],color=colors[2])
axes[1,0].set_ylabel("Households")
axes[1,0].set_xlabel("years")
axes[1,0].set_ylim([0,0.3])

sns.lineplot(modelSG_upper['uncertaintyh_priceelas']['price_elasticity'].item().secondType[0,1,:], ax = axes[1,1],color=colors[0])
sns.lineplot(modelSG_upper['uncertaintyh_priceelas']['price_elasticity'].item().secondType[1,1,:], ax = axes[1,1],color='red')
sns.lineplot(modelSG_upper['uncertaintyh_priceelas']['price_elasticity'].item().secondType[2,1,:], ax = axes[1,1],color=colors[2])
axes[1,1].set_xlabel("years")
axes[1,1].set_ylim([0,0.3])

for ax in axes.flatten():
    ax.set_xlim(0,40)
    ax.grid(False)

# plt.suptitle('Uncertainty Prices Term Structure')
plt.subplots_adjust(wspace=0.3, hspace=0.25)
plt.tight_layout()
fig.savefig("figures/figure_15.pdf")