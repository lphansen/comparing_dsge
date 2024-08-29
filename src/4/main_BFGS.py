import json
import numpy as np
import tensorflow as tf
import time
import os
from utils_para import setModelParameters
from utils_training import training_step_BFGS
from utils_DGM import DGMNet
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU') # To enable GPU acceleration, comment out this line and ensure CUDA and cuDNN libraries are properly installed
import argparse

def main(action_name, nWealth, nZ, nV, V_bar, sigma_K_norm, sigma_Z_norm, sigma_V_norm, wMin, wMax, chiUnderline, a_e, a_h, gamma_e, gamma_h, rho_e, rho_h, delta_e, delta_h, lambda_d, nu, shock_expo, n_layers, points_size, iter_num, units, seed, penalization, BFGS_maxiter, BFGS_maxfun):
    ## Domain parameters
    domain_list = [nWealth, nZ, nV, V_bar, sigma_K_norm, sigma_Z_norm, sigma_V_norm, wMin, wMax]
    wMin, wMax = [str("{:0.3f}".format(param)).replace('.', '', 1) for param in [wMin, wMax]]
    domain_folder = f'nW_{nWealth}_nZ_{nZ}_nV_{nV}_wMin_{wMin}_wMax_{wMax}'
    nDims = 3

    ## Model parameters
    parameter_list = [chiUnderline, a_e, a_h, gamma_e, gamma_h, rho_e, rho_h, delta_e, delta_h, lambda_d, nu]
    chiUnderline, a_e, a_h, gamma_e, gamma_h, rho_e, rho_h, delta_e, delta_h, lambda_d, nu = [str("{:0.3f}".format(param)).replace('.', '', 1) for param in parameter_list]
    model_folder = f'chiUnderline_{chiUnderline}_a_e_{a_e}_a_h_{a_h}_gamma_e_{gamma_e}_gamma_h_{gamma_h}_rho_e_{rho_e}_rho_h_{rho_h}_delta_e_{delta_e}_delta_h_{delta_h}_lambda_d_{lambda_d}_nu_{nu}'

    ## NN layer parameters
    layer_folder = f'seed_{seed}_n_layers_{n_layers}_units_{units}_points_size_{points_size}_iter_num_{iter_num}_penalization_{penalization}'

    ## Working directory
    workdir = os.getcwd()
    srcdir = workdir + '/src/'
    datadir = f'{workdir}/data/{action_name}/{shock_expo}/{domain_folder}/{model_folder}/'
    outputdir = f'/output/{action_name}/{shock_expo}/{domain_folder}/{model_folder}/{layer_folder}/'
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(outputdir, exist_ok=True)

    tf.random.set_seed(seed)
    np.random.seed(seed)
    tolerance = 1e-4

    ## Generate parameter set
    params = setModelParameters(parameter_list, domain_list, nDims, datadir, shock_expo)
    batchSize = int(2 ** points_size)
    dimension = 3
    activation = 'tanh'
    final_transformation = 'sigmoid'

    ## scipy BFGS optimization parameters
    BFGS_gtol = 1.0 * np.finfo(float).eps
    BFGS_maxcor = 100
    BFGS_maxls = 100
    BFGS_ftol = 1.0 * np.finfo(float).eps

    ## NN structure
    tf.keras.backend.set_floatx("float64")
    sim_NN = DGMNet(layer_width=units, n_layers=n_layers, input_dim=dimension, activation=activation, final_trans=final_transformation, seed=seed)

    ## Training
    losses = []
    times = []
    start = time.time()
    targets = tf.zeros(shape=(batchSize, 1), dtype=tf.float64)
    for iter in range(iter_num):
        W = tf.random.uniform(shape=(batchSize, 1), minval=params['wMin'], maxval=params['wMax'], dtype=tf.float64, seed=seed)
        Z = tf.random.uniform(shape=(batchSize, 1), minval=params['zMin'], maxval=params['zMax'], dtype=tf.float64, seed=seed)
        V = tf.random.uniform(shape=(batchSize, 1), minval=params['vMin'], maxval=params['vMax'], dtype=tf.float64, seed=seed)
        tf.print('Training Batch', iter + 1)
        results = training_step_BFGS(sim_NN, W, Z, V, params, targets, penalization, maxiter=BFGS_maxiter, maxfun=BFGS_maxfun, gtol=BFGS_gtol, maxcor=BFGS_maxcor, maxls=BFGS_maxls, ftol=BFGS_ftol)

        current_time = (time.time() - start) / 60
        losses.append(results.fun)
        times.append(current_time)

        if (iter + 1) % 5 == 0:
            if results.fun < tolerance:
                tf.print('Tolerance reached. Current loss:', results.fun, 'Batches:', iter + 1)
                break
            else:
                tf.print('Tolerance not reached yet. Current loss:', results.fun, 'need to increase batches')

    end = time.time()
    training_time = '{:.4f}'.format((end - start) / 60)
    print('Elapsed time for training {:.4f} sec'.format(end - start))

    ## Save trained neural network approximations and respective model parameters
    tf.saved_model.save(sim_NN, outputdir + 'sim_NN')

    NN_info = {
        'n_layers': n_layers, 'points_size': points_size, 'dimension': dimension, 'units': units,
        'activation': activation, 'iter_num': iter_num, 'training_time': training_time, 'batchSize': batchSize,
        'penalization': penalization, 'BFGS_maxiter': BFGS_maxiter, 'BFGS_maxfun': BFGS_maxfun,
        'BFGS_gtol': BFGS_gtol, 'BFGS_maxcor': BFGS_maxcor, 'BFGS_maxls': BFGS_maxls, 'BFGS_ftol': BFGS_ftol,
        'losses': losses, 'times': times
    }

    with open(outputdir + "/NN_info.json", "w") as f:
        json.dump(NN_info, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameter settings")
    parser.add_argument("--action_name", type=str, default="")
    parser.add_argument("--nWealth", type=int, default=100)
    parser.add_argument("--nZ", type=int, default=30)
    parser.add_argument("--nV", type=int, default=30)
    parser.add_argument("--V_bar", type=float, default=1.0)
    parser.add_argument("--sigma_K_norm", type=float, default=0.04)
    parser.add_argument("--sigma_Z_norm", type=float, default=0.0141)
    parser.add_argument("--sigma_V_norm", type=float, default=0.132)
    parser.add_argument("--wMin", type=float, default=0.01)
    parser.add_argument("--wMax", type=float, default=0.99)
    parser.add_argument("--chiUnderline", type=float, default=1.0)
    parser.add_argument("--a_e", type=float, default=0.14)
    parser.add_argument("--a_h", type=float, default=0.135)
    parser.add_argument("--gamma_e", type=float, default=1.0)
    parser.add_argument("--gamma_h", type=float, default=1.0)
    parser.add_argument("--rho_e", type=float, default=1.0)
    parser.add_argument("--rho_h", type=float, default=1.0)
    parser.add_argument("--delta_e", type=float, default=1.0)
    parser.add_argument("--delta_h", type=float, default=1.0)
    parser.add_argument("--lambda_d", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--shock_expo", type=str, default="")
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--points_size", type=int, default=2)
    parser.add_argument("--iter_num", type=int, default=10)
    parser.add_argument("--units", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--penalization", type=int, default=10000)
    parser.add_argument("--BFGS_maxiter", type=int, default=100)
    parser.add_argument("--BFGS_maxfun", type=int, default=1000)
    args = parser.parse_args()

    main(
        args.action_name, args.nWealth, args.nZ, args.nV, args.V_bar, args.sigma_K_norm, args.sigma_Z_norm, args.sigma_V_norm,
        args.wMin, args.wMax, args.chiUnderline, args.a_e, args.a_h, args.gamma_e, args.gamma_h, args.rho_e, args.rho_h, args.delta_e, args.delta_h, args.lambda_d, args.nu, args.shock_expo,
        args.n_layers, args.points_size, args.iter_num, args.units, args.seed, args.penalization,
        args.BFGS_maxiter, args.BFGS_maxfun
    )

