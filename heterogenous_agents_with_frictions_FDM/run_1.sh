#! /bin/bash

# Description: This script solves the model for the two-dimensional IP economy in Section 5.3.4 
# see parameter explanation in the main_solve.py file
chiUnderlinearray=(1.0)
a_earray=(0.0922)
a_harray=(0.07 0.08 0.085)
gamma_earray=(2.0)
gamma_harray=(2.0)
rho_earray=(1.0)
rho_harray=(1.0)
delta_earray=(0.03)
delta_harray=(0.01)
lambda_darray=(0.0)
nuarray=(0.1)
shock_expo_array=("lower_triangular" "upper_triangular")

nWarray=(1800)
nZ=30
dtarray=(0.01)

for nW in ${nWarray[@]}; do
    for dt in ${dtarray[@]}; do
        for chiUnderline in ${chiUnderlinearray[@]}; do
            for a_e in "${a_earray[@]}"; do
                for a_h in "${a_harray[@]}"; do
                    for gamma_e in "${gamma_earray[@]}"; do
                        for gamma_h in "${gamma_harray[@]}"; do
                            for rho_e in "${rho_earray[@]}"; do
                                for rho_h in "${rho_harray[@]}"; do
                                    for delta_e in "${delta_earray[@]}"; do
                                        for delta_h in "${delta_harray[@]}"; do
                                            for lambda_d in "${lambda_darray[@]}"; do
                                                for nu in "${nuarray[@]}"; do
                                                    for shock_expo in "lower_triangular"; do
                                                        for nu in "${nuarray[@]}"; do

                                                            count=0

                                                            action_name="finite_difference"

                                                            mkdir -p ./job-outs/${action_name}/${shock_expo}/${dt}_nW_${nW}_nZ_${nZ}/chiUnderline_${chiUnderline}/a_e_${a_e}_a_h_${a_h}/gamma_e_${gamma_e}_gamma_h_${gamma_h}/rho_e_${rho_e}_rho_h_${rho_h}/delta_e_${delta_e}_delta_h_${delta_h}/lambda_d_${lambda_d}_nu_${nu}/

                                                            if [ -f ./bash/${action_name}/${shock_expo}/${dt}_nW_${nW}_nZ_${nZ}/chiUnderline_${chiUnderline}/a_e_${a_e}_a_h_${a_h}/gamma_e_${gamma_e}_gamma_h_${gamma_h}/rho_e_${rho_e}_rho_h_${rho_h}/delta_e_${delta_e}_delta_h_${delta_h}/lambda_d_${lambda_d}_nu_${nu}/run.sh ]; then
                                                                rm ./bash/${action_name}/${shock_expo}/${dt}_nW_${nW}_nZ_${nZ}/chiUnderline_${chiUnderline}/a_e_${a_e}_a_h_${a_h}/gamma_e_${gamma_e}_gamma_h_${gamma_h}/rho_e_${rho_e}_rho_h_${rho_h}/delta_e_${delta_e}_delta_h_${delta_h}/lambda_d_${lambda_d}_nu_${nu}/run.sh
                                                            fi

                                                            mkdir -p ./bash/${action_name}/${shock_expo}/${dt}_nW_${nW}_nZ_${nZ}/chiUnderline_${chiUnderline}/a_e_${a_e}_a_h_${a_h}/gamma_e_${gamma_e}_gamma_h_${gamma_h}/rho_e_${rho_e}_rho_h_${rho_h}/delta_e_${delta_e}_delta_h_${delta_h}/lambda_d_${lambda_d}_nu_${nu}/

                                                            touch ./bash/${action_name}/${shock_expo}/${dt}_nW_${nW}_nZ_${nZ}/chiUnderline_${chiUnderline}/a_e_${a_e}_a_h_${a_h}/gamma_e_${gamma_e}_gamma_h_${gamma_h}/rho_e_${rho_e}_rho_h_${rho_h}/delta_e_${delta_e}_delta_h_${delta_h}/lambda_d_${lambda_d}_nu_${nu}/run.sh

                                                            tee -a ./bash/${action_name}/${shock_expo}/${dt}_nW_${nW}_nZ_${nZ}/chiUnderline_${chiUnderline}/a_e_${a_e}_a_h_${a_h}/gamma_e_${gamma_e}_gamma_h_${gamma_h}/rho_e_${rho_e}_rho_h_${rho_h}/delta_e_${delta_e}_delta_h_${delta_h}/lambda_d_${lambda_d}_nu_${nu}/run.sh <<EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=${shock_expo}_dt_${dt}_nW_${nW}_nZ_${nZ}_${chiUnderline}_${a_e}_${a_h}_${gamma_e}_${gamma_h}_${rho_e}_${rho_h}_${action_name}_${delta_e}_${delta_h}_${lambda_d}_${nu}
#SBATCH --output=./job-outs/${action_name}/${shock_expo}/${dt}_nW_${nW}_nZ_${nZ}/chiUnderline_${chiUnderline}/a_e_${a_e}_a_h_${a_h}/gamma_e_${gamma_e}_gamma_h_${gamma_h}/rho_e_${rho_e}_rho_h_${rho_h}/delta_e_${delta_e}_delta_h_${delta_h}/lambda_d_${lambda_d}_nu_${nu}/run.out
#SBATCH --error=./job-outs/${action_name}/${shock_expo}/${dt}_nW_${nW}_nZ_${nZ}/chiUnderline_${chiUnderline}/a_e_${a_e}_a_h_${a_h}/gamma_e_${gamma_e}_gamma_h_${gamma_h}/rho_e_${rho_e}_rho_h_${rho_h}/delta_e_${delta_e}_delta_h_${delta_h}/lambda_d_${lambda_d}_nu_${nu}/run.err
#SBATCH --time=0-23:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G

module load python/anaconda-2022.05

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)

python3 ./src/main_solve.py --chiUnderline $chiUnderline --a_e $a_e --a_h $a_h --gamma_e $gamma_e --gamma_h $gamma_h --rho_e $rho_e --rho_h $rho_h --nW $nW  --action_name $action_name --delta_e $delta_e --delta_h $delta_h --lambda_d $lambda_d --nu $nu --dt $dt --nZ $nZ --shock_expo $shock_expo
python3 ./src/main_evaluate.py --chiUnderline $chiUnderline --a_e $a_e --a_h $a_h --gamma_e $gamma_e --gamma_h $gamma_h --rho_e $rho_e --rho_h $rho_h --nW $nW  --action_name $action_name --delta_e $delta_e --delta_h $delta_h --lambda_d $lambda_d --nu $nu --dt $dt --nZ $nZ --shock_expo $shock_expo

echo "Program ends \$(date)"
end_time=\$(date +%s)
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
                                                            count=$(($count + 1))
                                                            sbatch ./bash/${action_name}/${shock_expo}/${dt}_nW_${nW}_nZ_${nZ}/chiUnderline_${chiUnderline}/a_e_${a_e}_a_h_${a_h}/gamma_e_${gamma_e}_gamma_h_${gamma_h}/rho_e_${rho_e}_rho_h_${rho_h}/delta_e_${delta_e}_delta_h_${delta_h}/lambda_d_${lambda_d}_nu_${nu}/run.sh
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
    done
done