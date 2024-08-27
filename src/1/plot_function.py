import os
import sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

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


def plot_solution(rho, gamma, Delta, delta, alpha, action_name):
    test = return_solution(rho, gamma, Delta, delta, alpha, action_name)


    #Plot 1
    fig, axes = plt.subplots(1,1, figsize=(5,4))

    sns.lineplot(data = test['elasticity_logc']['exposure_elasticity'].item().firstType[0,1,:], ax = axes,label="$Z^2$ 10 pct")
    sns.lineplot(data = test['elasticity_logc']['exposure_elasticity'].item().firstType[1,1,:], ax = axes,label="$Z^2$ 50 pct")
    sns.lineplot(data = test['elasticity_logc']['exposure_elasticity'].item().firstType[2,1,:], ax = axes,label="$Z^2$ 90 pct")
    axes.set_xlim([0,40])
    plt.suptitle(r"$\hat{C}$ exposure elasticity")
    plt.ylabel('Elasticity')
    plt.xlabel('years')

    plt.tight_layout()
    # plt.savefig("plots/simple_plot_2.png")


