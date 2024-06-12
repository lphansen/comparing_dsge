#!/bin/bash


#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=23:00:00
#SBATCH --output=solveModel192.out
#SBATCH --error=solveModel192.err

module unload boost
module load boost/1.62.0
module load ddt/7.0

export OMP_NUM_THREADS=28
python solveModelsParallel.py

