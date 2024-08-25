import os
import sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sns.set(style="whitegrid", font_scale=1.13, rc={"lines.linewidth": 3.5})
plt.rcParams['axes.formatter.useoffset'] = True
sys.path.append('./src')
from utils_pde_shock_elasticity import computeElas

parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--Delta",type=float,default=1.)
parser.add_argument("--delta",type=float,default=0.1)
parser.add_argument("--gamma",type=float,default=8.0)
parser.add_argument("--rho",type=float,default=1.00001)
parser.add_argument("--alpha",type=float,default=1.0)
parser.add_argument("--action_name",type=str,default="tests")
args = parser.parse_args()

Delta = args.Delta
delta = args.delta
gamma = args.gamma
rho = args.rho
alpha = args.alpha
action_name = args.action_name

def return_solution(rho, gamma, Delta, delta, alpha, action_name):

    outputdir = f"./output/{action_name}/Delta_{Delta}/delta_{delta}/gamma_{gamma}_rho_{rho}_alpha_{alpha}/"
    
    res = np.load(os.path.join(outputdir, "res.npz"))
    elasticity_logc = np.load(os.path.join(outputdir, "elasticity_logc.npz"), allow_pickle=True)
    elasticity_logimo = np.load(os.path.join(outputdir, "elasticity_logimo.npz"), allow_pickle=True)
    uncertainty_priceelas = np.load(os.path.join(outputdir, "uncertainty_priceelas.npz"), allow_pickle=True)

    return {
        'res': res,
        'elasticity_logc': elasticity_logc,
        'elasticity_logimo': elasticity_logimo,
        'uncertainty_priceelas': uncertainty_priceelas
    }


action_name = "simple"
gamma_1_rho_100_delta_001 = return_solution(rho = 1.0, gamma = 1.0, Delta = 1.0, alpha = 0.0922, delta=0.01, action_name = action_name)


#Plot 1
fig, axes = plt.subplots(1,1, figsize=(8, 6.5))

sns.lineplot(data = gamma_1_rho_100_delta_001['elasticity_logimo']['exposure_elasticity'].item().firstType[1,1,:], ax = axes)
axes.set_xlim([0,40])
axes.set_ylabel('$\\gamma = 1$', fontsize=20)
axes.set_title(r'$\delta=0.01$', fontsize=20)
axes.set_ylim([-0.005,0.005])
plt.savefig("plots/simple_plot_1.png")


#Plot 2
fig, axes = plt.subplots(1,1, figsize=(8,6.5))

sns.lineplot(data = gamma_1_rho_100_delta_001['elasticity_logc']['exposure_elasticity'].item().firstType[1,1,:], ax = axes)
axes.set_xlim([0,40])
axes.set_ylabel('$\\gamma = 1$', fontsize=20)
axes.set_title(r'Exposure elasticity', fontsize=20)
axes.set_ylim([-0.02,0.05])
plt.savefig("plots/simple_plot_2.png")


#Plot 3
fig, axes = plt.subplots(1, 1, figsize=(12,4))

sns.lineplot(data = gamma_1_rho_100_delta_001['elasticity_logc']['exposure_elasticity'].item().firstType[1,1,:],  ax = axes)
axes.set_xlim([0,40])
axes.set_ylim([0,0.06])
axes.set_title('Exposure elasticity', fontsize=20)
plt.savefig("plots/simple_plot_3.png")
