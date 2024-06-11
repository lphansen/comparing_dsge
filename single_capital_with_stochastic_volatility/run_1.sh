#!/bin/bash

# Description: This script solves the model and computes shock elasticities for the example economy in Section 4.4 for \rho = 1.0

# see parameter explanation in the main_onecapital.jl file
Deltaarray=(1.0)
gammaarray=(1.0 8.0)

deltaarray=(0.01 0.015)
rhoarray=(1.0 1.0)
alphaarray=(0.0922 0.1002)

# Run Model Execution
# Loop through all configurations and execute the models.
for Delta in ${Deltaarray[@]}; do
    for index in ${!deltaarray[@]}; do
        delta=${deltaarray[$index]}
        alpha=${alphaarray[$index]}
        rho=${rhoarray[$index]}
            for gamma in "${gammaarray[@]}"; do   

                action_name="single_capital_with_stochastic_volatility"

                mkdir -p ./job-outs/${action_name}/Delta_${Delta}/delta_${delta}/

                if [ -f ./bash/${action_name}/Delta_${Delta}/delta_${delta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}.sh ]; then
                    rm ./bash/${action_name}/Delta_${Delta}/delta_${delta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}.sh
                fi

                mkdir -p ./bash/${action_name}/Delta_${Delta}/delta_${delta}/

                touch ./bash/${action_name}/Delta_${Delta}/delta_${delta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}.sh

                tee -a ./bash/${action_name}/Delta_${Delta}/delta_${delta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}.sh <<EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=D_${Delta}_gamma_${gamma}_rho_${rho}_${action_name}
#SBATCH --output=./job-outs/$job_name/${action_name}/Delta_${Delta}/delta_${delta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}.out
#SBATCH --error=./job-outs/$job_name/${action_name}/Delta_${Delta}/delta_${delta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}.err
#SBATCH --time=0-23:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G

module load julia/1.7.3
module load python/anaconda-2022.05

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)

srun julia ./src/main_onecapital.jl  --Delta ${Delta} --delta ${delta} --gamma ${gamma} --rho ${rho} --action_name ${action_name} --alpha ${alpha}
python3 ./src/main_pde_shock_elasticity.py --Delta ${Delta} --delta ${delta} --gamma ${gamma} --rho ${rho} --action_name ${action_name} --alpha ${alpha}

echo "Program ends \$(date)"
end_time=\$(date +%s)
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
            sbatch ./bash/${action_name}/Delta_${Delta}/delta_${delta}/rho_${rho}_gamma_${gamma}_alpha_${alpha}.sh
        done
    done
done