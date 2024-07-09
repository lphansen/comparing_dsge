#!/bin/bash

# Description: This script solves the model and computes shock elasticities for the example economy in Section 4.4 for \rho = 0.67, 1.5 with stochastic volatility.

#### Parameter set: this is the only thing you need to change.
Delta=1.0
gamma=1.0
delta=0.01
rho=1.0
alpha=0.0922
action_name="simple"

####

mkdir -p ./job-outs/simple/

if [ -f ./bash/simple/simple.sh ]; then
    rm ./bash/simple/simple.sh
fi

mkdir -p ./bash/simple/
mkdir -p ./plots/

touch ./bash/simple/simple.sh

tee -a ./bash/simple/simple.sh <<EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=simple
#SBATCH --output=./job-outs/simple/job.out
#SBATCH --error=./job-outs/simple/job.err
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

srun julia ./src/main_onecapital.jl --Delta ${Delta} --delta ${delta} --gamma ${gamma} --rho ${rho} --action_name ${action_name} --alpha ${alpha}
python3 ./src/main_pde_shock_elasticity.py --Delta ${Delta} --delta ${delta} --gamma ${gamma} --rho ${rho} --action_name ${action_name} --alpha ${alpha}
python3 ./src/simple_plot.py --Delta ${Delta} --delta ${delta} --gamma ${gamma} --rho ${rho} --action_name ${action_name} --alpha ${alpha}

echo "Program ends \$(date)"
end_time=\$(date +%s)
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'%H hr %M min %S sec')"

EOF
sbatch ./bash/simple/simple.sh
