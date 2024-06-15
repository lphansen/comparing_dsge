#!/bin/bash

# Description: This script solves the model and computes shock elasticities for the two-dimensional economy in Section 4.6 for \rho = 1.0 without stochastic volatility.

# see parameter explanation in the main_twocapitals_two_dimensions.jl file
Deltaarray=(1.0 0.1)
beta1array=(0.04 0.0)
beta2array=(0.04 0.08)

alphaarray=(0.1844)
rhoarray=(1.0)
zetaarray=(0.5)
gammaarray=(1.0 4.0 8.0)
kappaarray=(0.0 1.0 2.0)

for index in ${!Deltaarray[@]}; do
    Delta=${Deltaarray[$index]}
    beta1=${beta1array[$index]}
    beta2=${beta2array[$index]}
    for index2 in ${!rhoarray[@]}; do
        rho=${rhoarray[$index2]}
        alpha=${alphaarray[$index2]}
        for gamma in "${gammaarray[@]}"; do
            for kappa in "${kappaarray[@]}"; do
                for zeta in "${zetaarray[@]}"; do
                    if [[ "$beta1" == "0.0" && ("$kappa" == "1.0" || "$kappa" == "2.0") ]]; then
                        continue
                    fi

                    action_name="twocap_model_without_stochastic_volatility"

                    mkdir -p ./job-outs/${action_name}/Delta_${Delta}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/

                    if [ -f ./bash/${action_name}/Delta_${Delta}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/run.sh ]; then
                        rm ./bash/${action_name}/Delta_${Delta}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/run.sh
                    fi

                    mkdir -p ./bash/${action_name}/Delta_${Delta}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/

                    touch ./bash/${action_name}/Delta_${Delta}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/run.sh

                    tee -a ./bash/${action_name}/Delta_${Delta}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/run.sh <<EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=beta1_${beta1}_beta2_${beta2}_gamma_${gamma}_rho_${rho}_kappa_${kappa}_zeta_${zeta}_${action_name}
#SBATCH --output=./job-outs/${action_name}/Delta_${Delta}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/run.out
#SBATCH --error=./job-outs/${action_name}/Delta_${Delta}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/run.err
#SBATCH --time=0-23:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

module load julia/1.7.3

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)

srun julia ./src/main_twocapitals_two_dimensions.jl  --Delta ${Delta} --gamma ${gamma}  --rho ${rho} --kappa ${kappa} --zeta ${zeta} --action_name ${action_name} --beta1 ${beta1} --beta2 ${beta2} --alpha ${alpha}

echo "Program ends \$(date)"
end_time=\$(date +%s)
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
                    count=$(($count + 1))
                    sbatch ./bash/${action_name}/Delta_${Delta}/beta1_${beta1}_beta2_${beta2}/kappa_${kappa}_zeta_${zeta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}/run.sh
                done
            done
        done
    done
done