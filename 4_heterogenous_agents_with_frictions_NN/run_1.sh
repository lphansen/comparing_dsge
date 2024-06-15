#!/bin/bash

# Description: This script solve the models for the three-dimensional RF / SG / PR economy in Section 5.3.1, 5.3.2, 5.3.3 using neural networks
# see parameter explanation in the main_BFGS.py file
chiUnderlinearray=(1.0 1.0 1.0 0.2 0.2 0.2 0.00001 0.00001 0.00001)
gamma_earray=(3.0 4.0 5.0 3.0 4.0 5.0 3.0 4.0 5.0)
a_earray=(0.0922)
a_harray=(0.0)
gamma_harray=(8.0)
delta_earray=(0.0115)
delta_harray=(0.01)
lambda_darray=(0.0)
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

seed_array=(256)
n_layers_array=(2)
units_array=(16)
points_size_array=(10)
iter_num_array=(5)
penalization=10000

BFGSmaxiter=100
BFGSmaxfun=100

count=0
for index in ${!chiUnderlinearray[@]}; do
    chiUnderline=${chiUnderlinearray[$index]}
    gamma_e=${gamma_earray[$index]}
    for a_e in "${a_earray[@]}"; do
        for a_h in "${a_harray[@]}"; do
            for gamma_h in "${gamma_harray[@]}"; do
                for delta_e in "${delta_earray[@]}"; do
                    for delta_h in "${delta_harray[@]}"; do
                        for lambda_d in "${lambda_darray[@]}"; do
                            for n_layers in "${n_layers_array[@]}"; do
                                for units in "${units_array[@]}"; do
                                    for points_size in "${points_size_array[@]}"; do
                                        for iter_num in "${iter_num_array[@]}"; do
                                            for seed in "${seed_array[@]}"; do
                                                for shock_expo in "upper_triangular" "lower_triangular"; do

                                                    action_name="neural_net"
                                            
                                                    mkdir -p ./job-outs/${action_name}/${shock_expo}/nW_${nWealth}_nZ_${nZ}_nV_${nV}/chiUnderline_${chiUnderline}_a_e_${a_e}_a_h_${a_h}_gamma_e_${gamma_e}_gamma_h_${gamma_h}_delta_e_${delta_e}_delta_h_${delta_h}_lambda_d_${lambda_d}_nu_${nu}/n_layers_${n_layers}_points_size_${points_size}_units_${units}_penalization_${penalization}_iter_num_${iter_num}/

                                                    if [ -f ./bash/${action_name}/${shock_expo}/nW_${nWealth}_nZ_${nZ}_nV_${nV}/chiUnderline_${chiUnderline}_a_e_${a_e}_a_h_${a_h}_gamma_e_${gamma_e}_gamma_h_${gamma_h}_delta_e_${delta_e}_delta_h_${delta_h}_lambda_d_${lambda_d}_nu_${nu}/n_layers_${n_layers}_points_size_${points_size}_units_${units}_penalization_${penalization}_iter_num_${iter_num}/run_${seed}.sh ]; then
                                                        rm ./bash/${action_name}/${shock_expo}/nW_${nWealth}_nZ_${nZ}_nV_${nV}/chiUnderline_${chiUnderline}_a_e_${a_e}_a_h_${a_h}_gamma_e_${gamma_e}_gamma_h_${gamma_h}_delta_e_${delta_e}_delta_h_${delta_h}_lambda_d_${lambda_d}_nu_${nu}/n_layers_${n_layers}_points_size_${points_size}_units_${units}_penalization_${penalization}_iter_num_${iter_num}/run_${seed}.sh
                                                    fi

                                                    mkdir -p ./bash/${action_name}/${shock_expo}/nW_${nWealth}_nZ_${nZ}_nV_${nV}/chiUnderline_${chiUnderline}_a_e_${a_e}_a_h_${a_h}_gamma_e_${gamma_e}_gamma_h_${gamma_h}_delta_e_${delta_e}_delta_h_${delta_h}_lambda_d_${lambda_d}_nu_${nu}/n_layers_${n_layers}_points_size_${points_size}_units_${units}_penalization_${penalization}_iter_num_${iter_num}/

                                                    touch ./bash/${action_name}/${shock_expo}/nW_${nWealth}_nZ_${nZ}_nV_${nV}/chiUnderline_${chiUnderline}_a_e_${a_e}_a_h_${a_h}_gamma_e_${gamma_e}_gamma_h_${gamma_h}_delta_e_${delta_e}_delta_h_${delta_h}_lambda_d_${lambda_d}_nu_${nu}/n_layers_${n_layers}_points_size_${points_size}_units_${units}_penalization_${penalization}_iter_num_${iter_num}/run_${seed}.sh

                                                    tee -a ./bash/${action_name}/${shock_expo}/nW_${nWealth}_nZ_${nZ}_nV_${nV}/chiUnderline_${chiUnderline}_a_e_${a_e}_a_h_${a_h}_gamma_e_${gamma_e}_gamma_h_${gamma_h}_delta_e_${delta_e}_delta_h_${delta_h}_lambda_d_${lambda_d}_nu_${nu}/n_layers_${n_layers}_points_size_${points_size}_units_${units}_penalization_${penalization}_iter_num_${iter_num}/run_${seed}.sh <<EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=solve_${seed}_${action_name}_${chiUnderline}_${a_e}_${a_h}_${gamma_e}_${gamma_h}_${delta_e}_${delta_h}_${lambda_d}_${nu}_${n_layers}_${points_size}_${units}_${penalization}_${iter_num}
#SBATCH --output=./job-outs/${action_name}/${shock_expo}/nW_${nWealth}_nZ_${nZ}_nV_${nV}/chiUnderline_${chiUnderline}_a_e_${a_e}_a_h_${a_h}_gamma_e_${gamma_e}_gamma_h_${gamma_h}_delta_e_${delta_e}_delta_h_${delta_h}_lambda_d_${lambda_d}_nu_${nu}/n_layers_${n_layers}_points_size_${points_size}_units_${units}_penalization_${penalization}_iter_num_${iter_num}/run_${seed}.out
#SBATCH --error=./job-outs/${action_name}/${shock_expo}/nW_${nWealth}_nZ_${nZ}_nV_${nV}/chiUnderline_${chiUnderline}_a_e_${a_e}_a_h_${a_h}_gamma_e_${gamma_e}_gamma_h_${gamma_h}_delta_e_${delta_e}_delta_h_${delta_h}_lambda_d_${lambda_d}_nu_${nu}/n_layers_${n_layers}_points_size_${points_size}_units_${units}_penalization_${penalization}_iter_num_${iter_num}/run_${seed}.err
#SBATCH --time=0-23:00:00
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=5G

module load python/anaconda-2022.05  

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)

python3 ./src/main_BFGS.py --chiUnderline ${chiUnderline} --a_e ${a_e} --a_h ${a_h} --gamma_e ${gamma_e} --gamma_h ${gamma_h} --rho_e ${rho_e} --rho_h ${rho_h} --nWealth ${nWealth} --nZ ${nZ} --nV ${nV} --V_bar ${V_bar} --sigma_V_norm ${sigma_V_norm} --n_layers ${n_layers} --points_size ${points_size} --iter_num ${iter_num} --seed ${seed} --BFGS_maxiter ${BFGSmaxiter} --BFGS_maxfun ${BFGSmaxfun} --sigma_K_norm ${sigma_K_norm} --sigma_Z_norm ${sigma_Z_norm} --delta_e ${delta_e} --delta_h ${delta_h} --lambda_d ${lambda_d} --nu ${nu} --wMin ${wMin} --wMax ${wMax} --action_name ${action_name} --penalization ${penalization} --units ${units} --shock_expo ${shock_expo}

echo "Program ends \$(date)"
end_time=\$(date +%s)
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
                                                    count=$(($count + 1))
                                                    sbatch ./bash/${action_name}/${shock_expo}/nW_${nWealth}_nZ_${nZ}_nV_${nV}/chiUnderline_${chiUnderline}_a_e_${a_e}_a_h_${a_h}_gamma_e_${gamma_e}_gamma_h_${gamma_h}_delta_e_${delta_e}_delta_h_${delta_h}_lambda_d_${lambda_d}_nu_${nu}/n_layers_${n_layers}_points_size_${points_size}_units_${units}_penalization_${penalization}_iter_num_${iter_num}/run_${seed}.sh                                                
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
echo "Total jobs submitted: $count" 
