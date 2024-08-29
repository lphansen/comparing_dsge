#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --time=0-23:00:00
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --job-name=main
#SBATCH --mem=50G
#SBATCH --output=main.out
#SBATCH --error=main.err

# Define the arguments
chiUnderline=1.0
gamma_e=4.0
a_e=0.0922
a_h=0.0
gamma_h=4.0
delta_e=0.0115
delta_h=0.01
lambda_d=0.0
rho_e=1.0
rho_h=1.0
nu=0.1

V_bar=0.0000063030303030303026
sigma_K_norm=3.1707442821755683
sigma_Z_norm=19.835431735873996
sigma_V_norm=0.0010882177801089308
wMin=0.01
wMax=0.99

nWealth=180
nZ=30
nV=30

seed=256
n_layers=2
units=16
points_size=10
iter_num=5
penalization=10000

BFGS_maxiter=100
BFGS_maxfun=100
action_name='test'
shock_expo='upper_triangular'

# Additional inputs
W_percentile=0.5
Z_percentile=0.5
V_percentile=0.9

module load python/anaconda-2022.05

# Run the Python script with the arguments
python3 src/4/main_sim_shock_elasticity.py \
    --chiUnderline $chiUnderline \
    --gamma_e $gamma_e \
    --a_e $a_e \
    --a_h $a_h \
    --gamma_h $gamma_h \
    --delta_e $delta_e \
    --delta_h $delta_h \
    --lambda_d $lambda_d \
    --rho_e $rho_e \
    --rho_h $rho_h \
    --nu $nu \
    --V_bar $V_bar \
    --sigma_K_norm $sigma_K_norm \
    --sigma_Z_norm $sigma_Z_norm \
    --sigma_V_norm $sigma_V_norm \
    --wMin $wMin \
    --wMax $wMax \
    --nWealth $nWealth \
    --nZ $nZ \
    --nV $nV \
    --seed $seed \
    --n_layers $n_layers \
    --units $units \
    --points_size $points_size \
    --iter_num $iter_num \
    --penalization $penalization \
    --BFGS_maxiter $BFGS_maxiter \
    --BFGS_maxfun $BFGS_maxfun \
    --action_name $action_name \
    --shock_expo $shock_expo \
    --W_percentile $W_percentile \
    --Z_percentile $Z_percentile \
    --V_percentile $V_percentile
