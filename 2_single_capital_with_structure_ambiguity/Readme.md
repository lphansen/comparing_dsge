# Single Capital with Structure Ambiguity

The subfolder solves single capital production economies with structure ambiguity in section 4.5.

## Scripts Overview

1. **run.sh**
    - **main_onecapital.jl**: Solves the two dimensional model using Finite Diffence.
        - **utils_onecapital.jl**: Contains source files for HJB equation computations.
    - **main_pde_shock_elasticity.py**: Solve the uncertainty price shock elasticity using Finite Diffence.
        - **utils_pde_shock_elasticity.py**: Contains source files for continuous-time shock elasticity PDE computations.
        - **utils_FDM.py**: Computes finite differences for input data across various dimensions and orders.
2. **results.ipynb**: Load model solutions, plot distorted drifts, ambiguity sets, and uncertainty price elasticities.

## Generated Directory Structure

Running the above bash scripts organizes the outputs and logs into specific directories to streamline troubleshooting and monitoring of script execution:

1. **job-outs**
   - Contains all log and error files associated with the script runs.
2. **bash**
   - Contains sbatch files for each parameter set, which are used to submit jobs to a computing cluster.
3. **output**
   - Contains the computed model solutions and other outputs.
        - ***res.npz**: Contains state variables, control variables, stationary densities, important derivatives, etc.
        - **uncertainty_priceelas.npz**: Contains the uncertainty price shock elasticity

## Estimated Running Time
1. **run.sh**: < 1 min
Each task tested on a single core of Intel Xeon Gold 6248R using the parameters in the bash file, with multiple tasks run simultaneously