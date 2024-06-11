#!/bin/bash

# Description: This script computes shock elasticities for the three-dimensional economy in Section 4.6 for \rho = 0.67, 1.0, 1.5 with stochastic volatility.

#### Parameter Configuration Arrays: each array contains the values of the parameters for the different scenarios
Deltaarray=(1.0)
kappaarray=(0.0)

beta1array=(0.04 0.0)
beta2array=(0.04 0.08)

rhoarray=(0.67 1.0 1.5) 
alphaarray=(0.1638 0.1844 0.216)
gammaarray=(1.0 4.0 8.0)
zetaarray=(0.5)

for index in ${!Deltaarray[@]}; do
    Delta=${Deltaarray[$index]}
    kappa=${kappaarray[$index]}
        for index2 in ${!rhoarray[@]}; do
            rho=${rhoarray[$index2]}
            alpha=${alphaarray[$index2]}
            for index3 in ${!beta1array[@]}; do
                beta1=${beta1array[$index3]}
                beta2=${beta2array[$index3]}
                    for gamma in "${gammaarray[@]}"; do
                        for zeta in "${zetaarray[@]}"; do
                            if [[ "$beta1" == "0.0" && ("$kappa" == "1.0" || "$kappa" == "2.0") ]]; then
                                continue
                            fi
                            if [[ ("$rho" == "0.67" || "$rho" == "1.5") && ("$kappa" == "1.0" || "$kappa" == "2.0") ]]; then
                                continue
                            fi

                            action_name="twocap_model_with_stochastic_volatility"

                            mkdir -p ./job-outs/${action_name}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/

                            if [ -f ./bash/${action_name}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/cal_ela_Delta_${Delta}.sh ]; then
                                rm ./bash/${action_name}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/cal_ela_Delta_${Delta}.sh
                            fi

                            mkdir -p ./bash/${action_name}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/

                            touch ./bash/${action_name}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/cal_ela_Delta_${Delta}.sh

                            tee -a ./bash/${action_name}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/cal_ela_Delta_${Delta}.sh <<EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=beta1_${beta1}_beta2_${beta2}_gamma_${gamma}_rho_${rho}_kappa_${kappa}_zeta_${zeta}_${action_name}
#SBATCH --output=./job-outs/$job_name/${action_name}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/cal_ela_Delta_${Delta}.out
#SBATCH --error=./job-outs/$job_name/${action_name}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/cal_ela_Delta_${Delta}.err
#SBATCH --time=1-11:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

module load python/anaconda-2022.05

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)

python3 ./src/main_pde_shock_elasticity.py  --Delta ${Delta} --gamma ${gamma} --rho ${rho} --kappa ${kappa} --zeta ${zeta} --alpha ${alpha} --beta1 ${beta1} --beta2 ${beta2} --action_name ${action_name} 

echo "Program ends \$(date)"
end_time=\$(date +%s)
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
                    sbatch ./bash/${action_name}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/cal_ela_Delta_${Delta}.sh
                done
            done
        done
    done
done